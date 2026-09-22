"""
harness/runner.py
──────────────────────────────────────────────────────────────────────────────
Executes HTTP requests against the target FastAPI app entirely in-process
(no network, no Docker) and returns rich telemetry including per-request
branch-coverage deltas.

Architecture
────────────
  ┌────────────────────────────────────────────────────────────────────────┐
  │ HarnessRunner.execute(method, path, …)                                │
  │   └─ BranchTracer.measure_sync(lambda: TestClient.request(…))         │
  │         └─ coverage.Coverage(branch=True, concurrency="thread")       │
  │               └─ FastAPI ASGI app  (same process, same PID)           │
  └────────────────────────────────────────────────────────────────────────┘

Key design choices
──────────────────
* ``starlette.testclient.TestClient`` is used (rather than ``httpx.AsyncClient``)
  to keep the execution path synchronous and avoid nested event-loop
  complications inside pytest.

* ``raise_server_exceptions=False`` on ``TestClient`` ensures that HTTP 500
  responses from deliberate bugs are returned as normal response objects rather
  than being re-raised as Python exceptions in the test thread.  This allows
  the harness to measure and report error responses alongside their coverage.

* ``BranchTracer`` uses ``concurrency="thread"`` so that branches executed
  inside Starlette's worker thread (where FastAPI route handlers run) are
  captured alongside main-thread branches.
"""
from __future__ import annotations

import time
from typing import Any

from starlette.testclient import TestClient

from core.tracer import BranchTracer, ExecutionResult


class HarnessRunner:
    """
    In-process HTTP test harness with branch-coverage instrumentation.

    Parameters
    ----------
    app    : ASGI application to instrument (typically the FastAPI ``app``).
    source : Python module / package names to instrument.  Forwarded verbatim to
             :class:`~core.tracer.BranchTracer`.  Defaults to ``["target"]``.

    Usage
    -----
    .. code-block:: python

        from target.app import app
        from harness.runner import HarnessRunner

        runner = HarnessRunner(app)

        result = runner.execute("POST", "/order/create", json={
            "items": [{"product_id": "X", "quantity": 1, "unit_price": 9.99}],
            "discount_pct": 10.0,
            "total_amount": 9.99,
        })
        print(result)
        # ExecutionResult(status=201, latency=3.12ms, branches=24, new=24, accumulated=24)
    """

    def __init__(
        self,
        app: Any,
        *,
        source: list[str] | None = None,
    ) -> None:
        # raise_server_exceptions=False → 500 responses returned, not re-raised
        self._client = TestClient(app, raise_server_exceptions=False)
        self._tracer = BranchTracer(source=source or ["target"])

    # ── Public API ───────────────────────────────────────────────────────────

    def execute(
        self,
        method: str,
        path: str,
        *,
        json: Any = None,
        params: dict[str, Any] | None = None,
        headers: dict[str, str] | None = None,
        content: bytes | None = None,
    ) -> ExecutionResult:
        """
        Fire a single HTTP request under branch-coverage instrumentation.

        Parameters
        ----------
        method  : HTTP verb (case-insensitive).
        path    : URL path string, e.g. ``"/order/create"`` or
                  ``"/users/USR-1/profile"``.
        json    : JSON-serialisable body dict.  Sets ``Content-Type: application/json``
                  automatically (via ``httpx``).
        params  : URL query parameters as a ``{key: value}`` mapping.
        headers : Additional HTTP request headers.
        content : Raw bytes body — mutually exclusive with *json*.

        Returns
        -------
        :class:`~core.tracer.ExecutionResult`
            Immutable telemetry record containing status code, latency, and the
            full branch-coverage delta for this specific request.
        """
        def _send():
            return self._client.request(
                method.upper(),
                path,
                json=json,
                params=params,
                headers=headers,
                content=content,
            )

        t0 = time.perf_counter()
        response, covered_arcs, new_arcs, accumulated = self._tracer.measure_sync(_send)
        latency_ms = (time.perf_counter() - t0) * 1_000.0

        # Derive covered line numbers from arc endpoints
        covered_lines: frozenset[int] = frozenset(
            line for arc in covered_arcs for line in arc
        )

        return ExecutionResult(
            status_code=response.status_code,
            latency_ms=latency_ms,
            covered_lines=covered_lines,
            covered_branches=covered_arcs,
            new_branches=new_arcs,
            accumulated_branches=accumulated,
        )

    def reset(self) -> None:
        """
        Reset the accumulated branch coverage set.

        Call this between isolated test scenarios where you want each scenario
        to compute coverage deltas relative to a fresh baseline.
        """
        self._tracer.reset()

    @property
    def accumulated_branches(self) -> frozenset[tuple[int, int]]:
        """
        Current global accumulated arc set (read-only snapshot).

        This set is monotonically non-decreasing: it only grows or stays equal
        across successive :meth:`execute` calls (until :meth:`reset` is called).
        """
        return self._tracer.accumulated_branches
