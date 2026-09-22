"""
envs/api_fuzz_env.py
──────────────────────────────────────────────────────────────────────────────
Gymnasium-compliant RL environment for adversarial API fuzzing.

Architecture overview
─────────────────────
                      ┌──────────────────────────────────┐
  RL Policy           │         APIFuzzEnv               │
  action: (f, op) ──▶ │  MutationEngine.apply(payload)   │
                      │  HarnessRunner.execute(path, …)   │
  observation ◀────── │  RewardEngine.compute(result, …)  │
  reward      ◀────── │  _build_obs()                     │
                      └──────────────────────────────────┘

Action Space
────────────
  ``MultiDiscrete([num_body_fields, NUM_OPERATORS])``

  • action[0] → index into the endpoint's body-field list
  • action[1] → MutationOperator enum value (0–5)

Observation Space (OBS_DIM = 5)
────────────────────────────────
  Index  Feature                          Range
  ─────  ───────────────────────────────  ──────────
    0    endpoint_indicator               1.0 (constant, single-endpoint)
    1    last status code / 1000          [0.0, 1.0]
    2    accumulated branch ratio         [0.0, 1.0]
    3    new-branch discovery flag        {0.0, 1.0}
    4    step progress  (step / max)      [0.0, 1.0]

All observation components are clipped to [0.0, 1.0] and returned as float32.

Episode lifecycle
─────────────────
  reset()      → restore payload to baseline; clear runner accumulation
  step(action) → mutate payload; execute request; compute reward
  truncated    → True when step_count >= max_steps
  terminated   → always False (no natural terminal state in Phase 2)

Gymnasium compliance
────────────────────
  Passes ``gymnasium.utils.env_checker.check_env(env, skip_render_check=True)``
  for any well-formed endpoint with at least one body field.
"""
from __future__ import annotations

import copy
from typing import Any

import numpy as np
import gymnasium as gym
from gymnasium import spaces

from core.parser import EndpointSpec, FieldSpec, OpenAPIParser
from envs.actions import MutationEngine, NUM_OPERATORS
from envs.rewards import RewardConfig, RewardEngine
from harness.runner import HarnessRunner


# ─────────────────────────────────────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────────────────────────────────────

OBS_DIM: int = 5
"""Fixed observation vector length (see module docstring for index map)."""

DEFAULT_BRANCH_CEILING: int = 500
"""Assumed upper bound on the total reachable branch count for normalisation.

Tune this upward for complex targets to prevent the branch-ratio observation
from saturating at 1.0 prematurely.
"""

DEFAULT_BASELINE_LATENCY_MS: float = 50.0
"""Initial baseline latency (ms) before any real measurement is available.

Updated on the first ``step()`` call to the actual measured latency, giving
subsequent norm_latency computations a data-driven reference point.
"""


# ─────────────────────────────────────────────────────────────────────────────
# Environment
# ─────────────────────────────────────────────────────────────────────────────

class APIFuzzEnv(gym.Env):
    """
    RL fuzzing environment that wraps the Phase 1 harness into a Gymnasium API.

    Parameters
    ----------
    parser   : ``OpenAPIParser`` pre-initialised with the target OpenAPI document.
    runner   : ``HarnessRunner`` wrapping the target ASGI application.
    endpoint : URL path of the target endpoint (e.g. ``"/order/create"``).
    method   : HTTP verb (default ``"POST"``).
    max_steps: Maximum steps before truncation.  Default 50.
    reward_config    : Optional :class:`~envs.rewards.RewardConfig`; uses spec
                       defaults when None.
    branch_ceiling   : Upper bound for branch-ratio normalisation.
    baseline_latency_ms : Initial latency baseline (ms).
    baseline_payload : Pre-built payload dict to use on ``reset()``.  When None,
                       the env auto-generates one from the schema.
    """

    metadata: dict = {"render_modes": []}

    def __init__(
        self,
        parser: OpenAPIParser,
        runner: HarnessRunner,
        endpoint: str,
        method: str = "POST",
        max_steps: int = 50,
        reward_config: RewardConfig | None = None,
        branch_ceiling: int = DEFAULT_BRANCH_CEILING,
        baseline_latency_ms: float = DEFAULT_BASELINE_LATENCY_MS,
        baseline_payload: dict | None = None,
    ) -> None:
        super().__init__()

        self._runner = runner
        self._endpoint_path = endpoint
        self._method = method.upper()
        self._max_steps = int(max_steps)
        self._branch_ceiling = max(1, int(branch_ceiling))
        self._baseline_latency_ms: float = float(baseline_latency_ms)
        self._baseline_latency_initialised: bool = False
        self._reward_engine = RewardEngine(reward_config)
        self._mutation_engine = MutationEngine()

        # ── Resolve EndpointSpec ──────────────────────────────────────────
        parsed = parser.parse()
        ep_spec: EndpointSpec | None = parsed.endpoint(endpoint, method)
        if ep_spec is None:
            available = [(e.method.upper(), e.path) for e in parsed.endpoints]
            raise ValueError(
                f"Endpoint {method.upper()} {endpoint!r} not found in parsed spec. "
                f"Available: {available}"
            )
        self._ep_spec: EndpointSpec = ep_spec

        # ── Body-field catalogue ──────────────────────────────────────────
        self._fields: list[FieldSpec] = list(ep_spec.body_fields)
        if not self._fields:
            raise ValueError(
                f"Endpoint {endpoint!r} has no body fields — cannot build "
                "a meaningful action space."
            )
        self._field_name_to_spec: dict[str, FieldSpec] = {
            f.name: f for f in self._fields
        }

        # ── Baseline payload ──────────────────────────────────────────────
        self._baseline_payload: dict = (
            baseline_payload
            if baseline_payload is not None
            else self._auto_build_baseline()
        )
        self._payload: dict = {}           # working payload (set on reset)

        # ── Episode state ─────────────────────────────────────────────────
        self._step_count: int = 0
        self._last_status: int = 0         # 0 = "no request yet"
        self._new_branch_found: bool = False

        # ── Gymnasium spaces ──────────────────────────────────────────────
        self.action_space = spaces.MultiDiscrete(
            nvec=[len(self._fields), NUM_OPERATORS],
            dtype=np.int64,
        )
        self.observation_space = spaces.Box(
            low=np.zeros(OBS_DIM, dtype=np.float32),
            high=np.ones(OBS_DIM, dtype=np.float32),
            dtype=np.float32,
        )

    # ── Gymnasium API ─────────────────────────────────────────────────────────

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict | None = None,
    ) -> tuple[np.ndarray, dict]:
        """
        Reset environment to the start of a new episode.

        Restores the working payload to the baseline, zeroes all episode
        counters, and clears the runner's accumulated branch set so that
        ``new_branches`` deltas in subsequent steps reflect fresh exploration.

        Returns
        -------
        observation : np.ndarray of shape (OBS_DIM,), dtype float32
        info        : dict with episode metadata
        """
        super().reset(seed=seed)

        self._payload = copy.deepcopy(self._baseline_payload)
        self._step_count = 0
        self._last_status = 0
        self._new_branch_found = False
        self._runner.reset()

        obs = self._build_obs()
        info: dict = {
            "episode_step"   : 0,
            "endpoint"       : self._endpoint_path,
            "num_fields"     : len(self._fields),
            "payload_keys"   : list(self._payload.keys()),
        }
        return obs, info

    def step(
        self, action: np.ndarray | tuple | list,
    ) -> tuple[np.ndarray, float, bool, bool, dict]:
        """
        Apply a mutation action, fire the HTTP request, and return telemetry.

        Parameters
        ----------
        action : Array-like ``[field_idx, operator_idx]``.
                 Values outside valid range are **clipped** (not rejected), to
                 tolerate exploratory policy actions near space boundaries.

        Returns
        -------
        observation  : np.ndarray (OBS_DIM,) float32 — next observation
        reward       : float — scalar reward for this step
        terminated   : bool  — always False in Phase 2 (no natural terminal)
        truncated    : bool  — True when ``step_count >= max_steps``
        info         : dict  — execution telemetry
        """
        # ── Unpack and guard action ───────────────────────────────────────
        action_arr = np.asarray(action, dtype=np.int64).ravel()
        field_idx = int(np.clip(action_arr[0], 0, len(self._fields) - 1))
        op_idx    = int(np.clip(action_arr[1], 0, NUM_OPERATORS - 1))

        target_field = self._fields[field_idx].name
        field_spec   = self._fields[field_idx]

        # ── Mutate payload ────────────────────────────────────────────────
        self._payload = self._mutation_engine.apply(
            self._payload, target_field, op_idx, field_spec
        )

        # ── Execute request ───────────────────────────────────────────────
        filled_path = self._resolve_path_params(self._endpoint_path)
        result = self._runner.execute(
            self._method,
            filled_path,
            json=self._payload if self._payload else None,
        )

        # ── Calibrate latency baseline on first real measurement ──────────
        if not self._baseline_latency_initialised and result.latency_ms > 0.0:
            self._baseline_latency_ms = result.latency_ms
            self._baseline_latency_initialised = True

        # ── Update episode state ──────────────────────────────────────────
        self._step_count += 1
        self._last_status = result.status_code
        self._new_branch_found = len(result.new_branches) > 0

        # ── Reward ────────────────────────────────────────────────────────
        reward = self._reward_engine.compute(result, self._baseline_latency_ms)

        # ── Termination / truncation ──────────────────────────────────────
        terminated: bool = False                           # no natural terminal
        truncated: bool = self._step_count >= self._max_steps

        obs = self._build_obs()
        info: dict = {
            "episode_step"        : self._step_count,
            "status_code"         : result.status_code,
            "latency_ms"          : result.latency_ms,
            "new_branches"        : len(result.new_branches),
            "accumulated_branches": len(result.accumulated_branches),
            "field_mutated"       : target_field,
            "operator_id"         : op_idx,
            "operator_name"       : str(MutationEngine._infer_json_type.__name__),
        }
        return obs, float(reward), terminated, truncated, info

    def render(self) -> None:
        """No-op render.  Rendering is not implemented in Phase 2."""
        return None

    # ── Internal helpers ──────────────────────────────────────────────────────

    def _build_obs(self) -> np.ndarray:
        """
        Construct a normalised float32 observation vector of shape (OBS_DIM,).

        All components are clipped to [0.0, 1.0] and explicitly cast to float32
        so that ``observation_space.contains(obs)`` always returns True.
        """
        obs = np.empty(OBS_DIM, dtype=np.float32)

        # [0] Endpoint indicator: constant 1.0 for single-endpoint environment
        obs[0] = 1.0

        # [1] Last HTTP status code, normalised by dividing by 1000
        #     200 → 0.200 | 400 → 0.400 | 422 → 0.422 | 500 → 0.500
        obs[1] = float(np.clip(self._last_status / 1000.0, 0.0, 1.0))

        # [2] Accumulated branch coverage ratio
        accumulated = len(self._runner.accumulated_branches)
        obs[2] = float(np.clip(accumulated / self._branch_ceiling, 0.0, 1.0))

        # [3] New-branch discovery flag: 1.0 if the last step found new arcs
        obs[3] = 1.0 if self._new_branch_found else 0.0

        # [4] Step progress: fraction of max_steps consumed
        obs[4] = float(np.clip(self._step_count / self._max_steps, 0.0, 1.0))

        return obs

    def _auto_build_baseline(self) -> dict:
        """
        Generate a best-effort valid JSON payload from the endpoint's body schema.

        Produces values that are within schema constraints (no boundary overflows)
        so that the baseline request is highly likely to return 2xx.  Used when
        no explicit ``baseline_payload`` is provided to ``__init__``.
        """
        payload: dict = {}
        for field in self._fields:
            # Default value: use as-is when declared
            if field.default is not None:
                payload[field.name] = field.default
                continue

            ftype = field.type
            if ftype == "string":
                if field.enum:
                    payload[field.name] = field.enum[0]
                elif field.pattern and r"\d{4}" in field.pattern:
                    payload[field.name] = "2024-01-01"   # valid ISO date
                else:
                    payload[field.name] = "test_value"
            elif ftype == "integer":
                lo = field.minimum
                payload[field.name] = int(lo) + 1 if lo is not None else 1
            elif ftype == "number":
                lo = field.minimum
                payload[field.name] = float(lo) + 1.0 if lo is not None else 1.0
            elif ftype == "boolean":
                payload[field.name] = True
            elif ftype == "array":
                payload[field.name] = []
            elif ftype == "object":
                payload[field.name] = {}
            else:
                payload[field.name] = None

        return payload

    def _resolve_path_params(self, path: str) -> str:
        """
        Fill ``{param_name}`` placeholders in *path* with deterministic defaults.

        Path parameters are not part of the Phase 2 action space; they receive
        fixed sentinel values so that the target endpoint is consistently
        reachable across all mutation steps.
        """
        filled = path
        for pp in self._ep_spec.path_params:
            filled = filled.replace(f"{{{pp.name}}}", f"FUZZ-{pp.name.upper()}")
        return filled

    # ── Read-only properties ──────────────────────────────────────────────────

    @property
    def num_fields(self) -> int:
        """Number of body fields available for mutation."""
        return len(self._fields)

    @property
    def field_names(self) -> list[str]:
        """Ordered list of body-field names (matches action_space index 0)."""
        return [f.name for f in self._fields]

    @property
    def baseline_payload(self) -> dict:
        """Deep copy of the baseline payload used on ``reset()``."""
        return copy.deepcopy(self._baseline_payload)

    def __repr__(self) -> str:  # noqa: D105
        return (
            f"APIFuzzEnv("
            f"endpoint={self._endpoint_path!r}, "
            f"method={self._method!r}, "
            f"num_fields={self.num_fields}, "
            f"max_steps={self._max_steps})"
        )
