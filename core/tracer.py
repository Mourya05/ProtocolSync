"""
core/tracer.py
──────────────────────────────────────────────────────────────────────────────
Per-request branch-coverage measurement using coverage.py in branch-arc mode.

Architecture
────────────
• ``coverage.Coverage(branch=True, concurrency="thread")`` is used so that
  branch arcs are collected across the Starlette TestClient worker thread
  (where FastAPI route handlers execute) in addition to the main thread.

• Each :meth:`BranchTracer.measure_sync` call writes coverage data to a
  temporary directory that is cleaned up immediately, leaving no persistent
  ``.coverage`` artefacts in the project root.

• The global ``_accumulated`` :class:`frozenset` is updated atomically under a
  :class:`threading.Lock`, guaranteeing a monotonically non-decreasing branch
  set even across sequential test calls from the same process.

Public API
──────────
  BranchTracer          Callable wrapper / context manager for measuring coverage.
  ExecutionResult       Immutable telemetry dataclass returned by HarnessRunner.
"""
from __future__ import annotations

import os
import tempfile
import threading
from dataclasses import dataclass
from typing import Any, Callable, TypeVar

import coverage as coverage_lib

_T = TypeVar("_T")


# ─────────────────────────────────────────────────────────────────────────────
# Immutable result dataclass
# ─────────────────────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class ExecutionResult:
    """
    Immutable telemetry record for a single HTTP request execution.

    Attributes
    ----------
    status_code         : HTTP integer response status.
    latency_ms          : Wall-clock request latency in milliseconds.
    covered_lines       : Set of source line numbers executed during the request.
    covered_branches    : Set of branch arc pairs ``(from_line, to_line)`` executed.
    new_branches        : Delta — arcs not seen in any prior call on this runner.
    accumulated_branches: Global monotonic set after incorporating this request.
    """

    status_code: int
    latency_ms: float
    covered_lines: frozenset[int]
    covered_branches: frozenset[tuple[int, int]]     # (from_line → to_line)
    new_branches: frozenset[tuple[int, int]]         # delta vs. accumulated before this call
    accumulated_branches: frozenset[tuple[int, int]] # global set after this call

    def __repr__(self) -> str:  # noqa: D105
        return (
            f"ExecutionResult("
            f"status={self.status_code}, "
            f"latency={self.latency_ms:.2f}ms, "
            f"branches={len(self.covered_branches)}, "
            f"new={len(self.new_branches)}, "
            f"accumulated={len(self.accumulated_branches)})"
        )


# ─────────────────────────────────────────────────────────────────────────────
# Branch tracer
# ─────────────────────────────────────────────────────────────────────────────

class BranchTracer:
    """
    Wraps a callable invocation with ``coverage.py`` branch-arc tracing.

    Parameters
    ----------
    source : Python module / package names to instrument.
             Defaults to ``["target"]`` (the Protocol-Sync target service).

    Thread safety
    -------------
    Sequential calls from a single thread are safe.  Concurrent parallel calls
    from multiple threads are **not** supported in Phase 1.

    Coverage data files
    -------------------
    Each call writes to a temporary directory that is deleted before returning,
    so no ``.coverage`` files accumulate in the project tree.
    """

    def __init__(self, source: list[str] | None = None) -> None:
        self._source: list[str] = source or ["target"]
        self._accumulated: frozenset[tuple[int, int]] = frozenset()
        self._lock = threading.Lock()

    # ── Public ──────────────────────────────────────────────────────────────

    def measure_sync(
        self,
        fn: Callable[..., _T],
        *args: Any,
        **kwargs: Any,
    ) -> tuple[
        _T,
        frozenset[tuple[int, int]],
        frozenset[tuple[int, int]],
        frozenset[tuple[int, int]],
    ]:
        """
        Invoke ``fn(*args, **kwargs)`` under ``coverage.py`` branch tracing.

        Returns
        -------
        ``(return_value, covered_arcs, new_arcs, accumulated_arcs)``

        * ``covered_arcs``   — branch arcs executed during *this* call.
        * ``new_arcs``       — subset of ``covered_arcs`` not in prior accumulated set.
        * ``accumulated_arcs``— updated global set (``prior ∪ covered_arcs``).

        The caller is responsible for timing if wall-clock latency is needed.
        """
        with tempfile.TemporaryDirectory(prefix="ps_cov_") as tmpdir:
            data_path = os.path.join(tmpdir, ".coverage_data")

            cov = coverage_lib.Coverage(
                branch=True,
                source=self._source,
                # Thread-aware tracing so worker threads (TestClient / anyio)
                # are instrumented alongside the main thread.
                concurrency="thread",
                data_file=data_path,
                config_file=False,
            )

            cov.start()
            try:
                retval: _T = fn(*args, **kwargs)
            finally:
                cov.stop()
                cov.save()

            covered_arcs = self._extract_arcs(cov)

        # Atomic accumulated-set update
        with self._lock:
            old_accumulated = self._accumulated
            new_arcs = covered_arcs - old_accumulated
            new_accumulated = old_accumulated | covered_arcs
            self._accumulated = new_accumulated

        return retval, covered_arcs, new_arcs, new_accumulated

    def reset(self) -> None:
        """
        Clear the accumulated branch set.

        Use between isolated test scenarios to start fresh coverage accounting.
        """
        with self._lock:
            self._accumulated = frozenset()

    @property
    def accumulated_branches(self) -> frozenset[tuple[int, int]]:
        """Current global accumulated arc set (thread-safe snapshot)."""
        with self._lock:
            return self._accumulated

    # ── Internal ────────────────────────────────────────────────────────────

    @staticmethod
    def _extract_arcs(cov: coverage_lib.Coverage) -> frozenset[tuple[int, int]]:
        """
        Extract executed branch arc pairs from a stopped :class:`coverage.Coverage` object.

        Filters out coverage.py's synthetic pseudo-arcs that use negative line
        numbers to represent module entry (``-1``) and exit (``-2``) events.
        These carry no useful branch information for the RL reward model.
        """
        arcs: set[tuple[int, int]] = set()

        try:
            cov_data = cov.get_data()
            for filename in cov_data.measured_files():
                file_arcs = cov_data.arcs(filename)
                if not file_arcs:
                    continue
                for src, dst in file_arcs:
                    # Skip pseudo-arcs: negative line numbers are internal to coverage.py
                    if src > 0 and dst > 0:
                        arcs.add((src, dst))
        except Exception:  # noqa: BLE001
            # Coverage data extraction is best-effort.  If the coverage object
            # contains no data (e.g. the target module was not imported yet),
            # return an empty set rather than crashing the harness.
            pass

        return frozenset(arcs)
