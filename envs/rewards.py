"""
envs/rewards.py
──────────────────────────────────────────────────────────────────────────────
Multi-objective reward computation engine for Protocol-Sync Phase 2.

Reward formula
──────────────
  R_t = (w_crash   × I_crash)
      + (w_branch  × Δbranches)
      + (w_latency × norm_latency)
      - (w_reject  × I_reject)
      - step_penalty

Where:
  I_crash       = 1.0   iff  status_code == 500           (bug found)
  Δbranches     = |new_branches|                           (coverage expansion)
  norm_latency  = latency_ms / baseline_ms, capped at latency_cap
  I_reject      = 1.0   iff  status_code ∈ {400, 422}     (early validation drop)
  step_penalty  = constant per-step cost (encourages sample efficiency)

Design rationale
────────────────
* w_crash (50.0) dominates: finding server crashes is the primary objective.
* w_branch (10.0) ensures the policy explores new code paths, not just known bugs.
* w_latency (2.0) rewards finding abnormally slow paths (potential DoS vectors).
  norm_latency is always positive, providing a dense base signal every step.
* w_reject (1.0) gently penalises schema-rejection payloads (Pydantic 422s) to
  discourage trivially broken payloads without blocking structural exploration.
* step_penalty (0.1) is subtracted every step to encourage finding bugs quickly.
"""
from __future__ import annotations

from dataclasses import dataclass

from core.tracer import ExecutionResult

# HTTP status codes that indicate early schema validation rejection
_REJECTION_CODES: frozenset[int] = frozenset({400, 422})


# ─────────────────────────────────────────────────────────────────────────────
# Configuration
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class RewardConfig:
    """
    Configurable weights and hyper-parameters for the multi-objective reward.

    All defaults match the Protocol-Sync Phase 2 specification.

    Attributes
    ----------
    w_crash       : Reward weight for triggering an HTTP 500 response.
    w_branch      : Reward per newly discovered branch arc.
    w_latency     : Reward multiplier for normalised latency.
    w_reject      : Penalty weight for schema validation rejections (400/422).
    step_penalty  : Constant per-step cost to encourage sample efficiency.
    latency_cap   : Upper bound on ``norm_latency`` (prevents unbounded rewards
                    from pathologically slow responses).
    """
    w_crash:      float = 50.0
    w_branch:     float = 10.0
    w_latency:    float =  2.0
    w_reject:     float =  1.0
    step_penalty: float =  0.1
    latency_cap:  float = 10.0   # norm_latency is capped at this value


# ─────────────────────────────────────────────────────────────────────────────
# Engine
# ─────────────────────────────────────────────────────────────────────────────

class RewardEngine:
    """
    Stateless reward calculator for a single fuzzing step.

    Parameters
    ----------
    config : :class:`RewardConfig` instance.  Uses spec-default values if None.

    Example
    -------
    .. code-block:: python

        engine = RewardEngine()
        r = engine.compute(result=execution_result, baseline_latency_ms=50.0)
        # r ≈ 51.9 for a crash (500) with baseline latency and no new branches
    """

    def __init__(self, config: RewardConfig | None = None) -> None:
        self._cfg: RewardConfig = config if config is not None else RewardConfig()

    # ── Public ───────────────────────────────────────────────────────────────

    def compute(
        self,
        result: ExecutionResult,
        baseline_latency_ms: float,
    ) -> float:
        """
        Compute the scalar reward for one execution step.

        Parameters
        ----------
        result              : Telemetry returned by ``HarnessRunner.execute()``.
        baseline_latency_ms : Latency of a nominal request (used for
                              normalisation).  Must be > 0; clamped to 1.0 ms
                              if zero or negative.

        Returns
        -------
        float
            Scalar reward value.  May be negative when only penalty terms fire.
        """
        cfg = self._cfg

        # ── Crash indicator ───────────────────────────────────────────────
        i_crash: float = 1.0 if result.status_code == 500 else 0.0

        # ── Branch coverage expansion ─────────────────────────────────────
        delta_branches: float = float(len(result.new_branches))

        # ── Latency, normalised and bounded ───────────────────────────────
        safe_baseline: float = max(float(baseline_latency_ms), 1.0)
        norm_latency: float = min(
            result.latency_ms / safe_baseline,
            cfg.latency_cap,
        )

        # ── Schema rejection indicator ─────────────────────────────────────
        i_reject: float = 1.0 if result.status_code in _REJECTION_CODES else 0.0

        reward: float = (
            cfg.w_crash   * i_crash
            + cfg.w_branch  * delta_branches
            + cfg.w_latency * norm_latency
            - cfg.w_reject  * i_reject
            - cfg.step_penalty
        )

        return float(reward)

    # ── Properties ───────────────────────────────────────────────────────────

    @property
    def config(self) -> RewardConfig:
        """Return the active :class:`RewardConfig`."""
        return self._cfg

    def __repr__(self) -> str:  # noqa: D105
        c = self._cfg
        return (
            f"RewardEngine(w_crash={c.w_crash}, w_branch={c.w_branch}, "
            f"w_latency={c.w_latency}, w_reject={c.w_reject}, "
            f"step_penalty={c.step_penalty})"
        )
