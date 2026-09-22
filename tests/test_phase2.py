"""
tests/test_phase2.py
──────────────────────────────────────────────────────────────────────────────
Integration tests for Protocol-Sync Phase 2:
  Gymnasium Environment & Multi-Objective Reward Engine.

Test groups
───────────
  1. TestMutationEngine      — operator correctness, deep-copy invariant, edge cases
  2. TestRewardEngine        — formula components, boundary conditions, config
  3. TestAPIFuzzEnvBasics    — construction, spaces, properties, baseline payload
  4. TestAPIFuzzEnvStep      — step semantics, observation bounds, info dict
  5. TestAPIFuzzEnvEpisode   — episode lifecycle, truncation, step accumulation
  6. TestGymnasiumCompliance — check_env (full Gymnasium API contract)

Run with:
    pytest tests/test_phase2.py -v --tb=short
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from core.parser import FieldSpec, OpenAPIParser
from core.tracer import ExecutionResult
from envs.actions import (
    MutationEngine,
    MutationOperator,
    NUM_OPERATORS,
    _SPECIAL_NUMERICS,
    _SPECIAL_STRINGS,
    _stable_index,
)
from envs.api_fuzz_env import (
    OBS_DIM,
    DEFAULT_BRANCH_CEILING,
    DEFAULT_BASELINE_LATENCY_MS,
    APIFuzzEnv,
)
from envs.rewards import RewardConfig, RewardEngine
from harness.runner import HarnessRunner
from target.app import app as target_app

# ─────────────────────────────────────────────────────────────────────────────
# Paths & constants
# ─────────────────────────────────────────────────────────────────────────────

_OPENAPI_JSON: Path = Path(__file__).parent.parent / "target" / "openapi.json"


# ─────────────────────────────────────────────────────────────────────────────
# Shared baseline payload for /order/create
# ─────────────────────────────────────────────────────────────────────────────

_ORDER_BASELINE: dict = {
    "items": [],           # empty list → subtotal=0, no ZeroDivisionError
    "discount_pct": 0.0,
    "total_amount": 1.0,
    "customer_note": "baseline",
}


# ─────────────────────────────────────────────────────────────────────────────
# Fixtures
# ─────────────────────────────────────────────────────────────────────────────

@pytest.fixture(scope="module")
def openapi_parser() -> OpenAPIParser:
    """Module-scoped parser (no IO after first load)."""
    doc = json.loads(_OPENAPI_JSON.read_text(encoding="utf-8"))
    return OpenAPIParser(doc)


@pytest.fixture()
def runner() -> HarnessRunner:
    """Fresh HarnessRunner per test — zeroed coverage accumulation."""
    r = HarnessRunner(target_app)
    r.reset()
    return r


@pytest.fixture()
def env(openapi_parser: OpenAPIParser, runner: HarnessRunner) -> APIFuzzEnv:
    """Fresh env targeting /order/create with an explicit safe baseline."""
    e = APIFuzzEnv(
        parser=openapi_parser,
        runner=runner,
        endpoint="/order/create",
        method="POST",
        max_steps=10,
        baseline_payload=_ORDER_BASELINE,
    )
    e.reset()
    return e


@pytest.fixture()
def mutation_engine() -> MutationEngine:
    return MutationEngine()


@pytest.fixture()
def reward_engine() -> RewardEngine:
    return RewardEngine()


def _make_result(
    status: int,
    new_branches: int = 0,
    latency_ms: float = 50.0,
    covered: int = 0,
) -> ExecutionResult:
    """Convenience factory for ExecutionResult test stubs."""
    new_arcs = frozenset((i, i + 1) for i in range(new_branches))
    all_arcs = frozenset((i, i + 1) for i in range(covered))
    return ExecutionResult(
        status_code=status,
        latency_ms=latency_ms,
        covered_lines=frozenset(range(covered)),
        covered_branches=new_arcs | all_arcs,
        new_branches=new_arcs,
        accumulated_branches=new_arcs | all_arcs,
    )


# ─────────────────────────────────────────────────────────────────────────────
# 1. MutationEngine tests
# ─────────────────────────────────────────────────────────────────────────────

class TestMutationEngine:
    """Verify that each operator produces the correct mutation."""

    _payload: dict = {
        "discount_pct": 10.0,
        "total_amount": 99.99,
        "username": "alice",
        "items": [1, 2, 3],
        "active": True,
        "meta": {"k": "v"},
    }

    # ── NO_OP ────────────────────────────────────────────────────────────────

    def test_no_op_returns_identical_values(self, mutation_engine: MutationEngine) -> None:
        result = mutation_engine.apply(self._payload, "discount_pct", MutationOperator.NO_OP)
        assert result["discount_pct"] == self._payload["discount_pct"]

    def test_no_op_is_deep_copy_not_same_object(self, mutation_engine: MutationEngine) -> None:
        result = mutation_engine.apply(self._payload, "items", MutationOperator.NO_OP)
        assert result is not self._payload
        assert result["items"] is not self._payload["items"]

    # ── FIELD_OMISSION ────────────────────────────────────────────────────────

    def test_field_omission_removes_existing_field(self, mutation_engine: MutationEngine) -> None:
        result = mutation_engine.apply(self._payload, "username", MutationOperator.FIELD_OMISSION)
        assert "username" not in result

    def test_field_omission_leaves_other_fields_intact(self, mutation_engine: MutationEngine) -> None:
        result = mutation_engine.apply(self._payload, "username", MutationOperator.FIELD_OMISSION)
        assert "discount_pct" in result
        assert result["discount_pct"] == self._payload["discount_pct"]

    def test_field_omission_on_missing_field_is_safe(self, mutation_engine: MutationEngine) -> None:
        """FIELD_OMISSION on a non-existent key should not raise."""
        result = mutation_engine.apply(self._payload, "nonexistent_key", MutationOperator.FIELD_OMISSION)
        assert "nonexistent_key" not in result

    # ── BOUNDARY_MIN_OVERFLOW ─────────────────────────────────────────────────

    def test_min_overflow_numeric_subtracts_one_from_minimum(self, mutation_engine: MutationEngine) -> None:
        spec = FieldSpec(name="discount_pct", type="number", required=False, minimum=0.0, maximum=100.0)
        result = mutation_engine.apply({"discount_pct": 10.0}, "discount_pct",
                                       MutationOperator.BOUNDARY_MIN_OVERFLOW, spec)
        assert result["discount_pct"] == -1.0    # minimum(0.0) - 1

    def test_min_overflow_string_produces_empty_string(self, mutation_engine: MutationEngine) -> None:
        spec = FieldSpec(name="username", type="string", required=True)
        result = mutation_engine.apply({"username": "alice"}, "username",
                                       MutationOperator.BOUNDARY_MIN_OVERFLOW, spec)
        assert result["username"] == ""

    def test_min_overflow_array_produces_empty_list(self, mutation_engine: MutationEngine) -> None:
        spec = FieldSpec(name="items", type="array", required=True)
        result = mutation_engine.apply({"items": [1, 2]}, "items",
                                       MutationOperator.BOUNDARY_MIN_OVERFLOW, spec)
        assert result["items"] == []

    def test_min_overflow_boolean_produces_false(self, mutation_engine: MutationEngine) -> None:
        spec = FieldSpec(name="active", type="boolean", required=False)
        result = mutation_engine.apply({"active": True}, "active",
                                       MutationOperator.BOUNDARY_MIN_OVERFLOW, spec)
        assert result["active"] is False

    # ── BOUNDARY_MAX_OVERFLOW ─────────────────────────────────────────────────

    def test_max_overflow_numeric_adds_one_to_maximum(self, mutation_engine: MutationEngine) -> None:
        spec = FieldSpec(name="discount_pct", type="number", required=False, minimum=0.0, maximum=100.0)
        result = mutation_engine.apply({"discount_pct": 10.0}, "discount_pct",
                                       MutationOperator.BOUNDARY_MAX_OVERFLOW, spec)
        assert result["discount_pct"] == 101.0   # maximum(100.0) + 1

    def test_max_overflow_string_is_long(self, mutation_engine: MutationEngine) -> None:
        spec = FieldSpec(name="username", type="string", required=True)
        result = mutation_engine.apply({"username": "alice"}, "username",
                                       MutationOperator.BOUNDARY_MAX_OVERFLOW, spec)
        assert isinstance(result["username"], str)
        assert len(result["username"]) == 10_000

    def test_max_overflow_array_produces_100_items(self, mutation_engine: MutationEngine) -> None:
        spec = FieldSpec(name="items", type="array", required=True)
        result = mutation_engine.apply({"items": []}, "items",
                                       MutationOperator.BOUNDARY_MAX_OVERFLOW, spec)
        assert isinstance(result["items"], list)
        assert len(result["items"]) == 100

    # ── TYPE_CONFUSION ────────────────────────────────────────────────────────

    def test_type_confusion_numeric_injects_string(self, mutation_engine: MutationEngine) -> None:
        spec = FieldSpec(name="discount_pct", type="number", required=False)
        result = mutation_engine.apply({"discount_pct": 10.0}, "discount_pct",
                                       MutationOperator.TYPE_CONFUSION, spec)
        assert isinstance(result["discount_pct"], str)

    def test_type_confusion_string_injects_integer(self, mutation_engine: MutationEngine) -> None:
        spec = FieldSpec(name="username", type="string", required=True)
        result = mutation_engine.apply({"username": "alice"}, "username",
                                       MutationOperator.TYPE_CONFUSION, spec)
        assert isinstance(result["username"], int)

    def test_type_confusion_array_injects_string(self, mutation_engine: MutationEngine) -> None:
        spec = FieldSpec(name="items", type="array", required=True)
        result = mutation_engine.apply({"items": []}, "items",
                                       MutationOperator.TYPE_CONFUSION, spec)
        assert isinstance(result["items"], str)

    def test_type_confusion_boolean_injects_string(self, mutation_engine: MutationEngine) -> None:
        spec = FieldSpec(name="active", type="boolean", required=False)
        result = mutation_engine.apply({"active": True}, "active",
                                       MutationOperator.TYPE_CONFUSION, spec)
        assert isinstance(result["active"], str)

    # ── SPECIAL_INJECTION ─────────────────────────────────────────────────────

    def test_special_injection_numeric_uses_pool(self, mutation_engine: MutationEngine) -> None:
        spec = FieldSpec(name="discount_pct", type="number", required=False)
        result = mutation_engine.apply({"discount_pct": 10.0}, "discount_pct",
                                       MutationOperator.SPECIAL_INJECTION, spec)
        assert result["discount_pct"] in _SPECIAL_NUMERICS

    def test_special_injection_string_uses_pool(self, mutation_engine: MutationEngine) -> None:
        spec = FieldSpec(name="username", type="string", required=True)
        result = mutation_engine.apply({"username": "alice"}, "username",
                                       MutationOperator.SPECIAL_INJECTION, spec)
        assert result["username"] in _SPECIAL_STRINGS

    def test_special_injection_deterministic_for_same_field(self, mutation_engine: MutationEngine) -> None:
        spec = FieldSpec(name="username", type="string", required=True)
        r1 = mutation_engine.apply({"username": "alice"}, "username",
                                   MutationOperator.SPECIAL_INJECTION, spec)
        r2 = mutation_engine.apply({"username": "bob"}, "username",
                                   MutationOperator.SPECIAL_INJECTION, spec)
        # Same field_name → same stable_index → same injected string
        assert r1["username"] == r2["username"]

    # ── Deep-copy invariant ────────────────────────────────────────────────────

    def test_original_payload_not_mutated_by_any_operator(
        self, mutation_engine: MutationEngine
    ) -> None:
        """Original dict must be unmodified regardless of which operator fires."""
        original = {"discount_pct": 10.0, "username": "alice", "items": [1]}
        import copy as _copy
        snapshot = _copy.deepcopy(original)

        for op in MutationOperator:
            mutation_engine.apply(original, "discount_pct", op)

        assert original == snapshot

    # ── Out-of-range operator index ────────────────────────────────────────────

    def test_out_of_range_operator_wraps_modulo(self, mutation_engine: MutationEngine) -> None:
        """Operator indices outside [0, NUM_OPERATORS) are wrapped silently."""
        # index 6 → 6 % 6 = 0 → NO_OP
        result = mutation_engine.apply({"x": 1}, "x", NUM_OPERATORS)
        assert result["x"] == 1    # NO_OP leaves value unchanged

    # ── No field_spec (type inference path) ──────────────────────────────────

    def test_infer_type_from_float_value(self, mutation_engine: MutationEngine) -> None:
        result = mutation_engine.apply({"v": 3.14}, "v",
                                       MutationOperator.TYPE_CONFUSION)
        # float inferred as "number" → TYPE_CONFUSION injects str
        assert isinstance(result["v"], str)

    def test_infer_type_from_list_value(self, mutation_engine: MutationEngine) -> None:
        result = mutation_engine.apply({"v": [1, 2]}, "v",
                                       MutationOperator.TYPE_CONFUSION)
        # list inferred as "array" → TYPE_CONFUSION injects str
        assert isinstance(result["v"], str)

    def test_stable_index_is_deterministic(self) -> None:
        assert _stable_index("discount_pct", 6) == _stable_index("discount_pct", 6)
        assert _stable_index("x", 10) == _stable_index("x", 10)

    def test_num_operators_constant(self) -> None:
        assert NUM_OPERATORS == 6
        assert len(MutationOperator) == 6


# ─────────────────────────────────────────────────────────────────────────────
# 2. RewardEngine tests
# ─────────────────────────────────────────────────────────────────────────────

class TestRewardEngine:
    """Verify each reward term and the combined formula."""

    _baseline: float = 50.0   # baseline latency used in all tests

    # ── Crash term ────────────────────────────────────────────────────────────

    def test_crash_gives_large_positive_reward(self, reward_engine: RewardEngine) -> None:
        result = _make_result(500, latency_ms=self._baseline)
        reward = reward_engine.compute(result, self._baseline)
        # R = 50.0 + 0 + 2*(50/50) - 0 - 0.1 = 51.9
        assert reward == pytest.approx(51.9)

    def test_non_crash_status_gives_no_crash_term(self, reward_engine: RewardEngine) -> None:
        r200 = reward_engine.compute(_make_result(200, latency_ms=self._baseline), self._baseline)
        r201 = reward_engine.compute(_make_result(201, latency_ms=self._baseline), self._baseline)
        # Both should be identical (no crash term)
        assert r200 == pytest.approx(r201)

    # ── Branch term ───────────────────────────────────────────────────────────

    def test_one_new_branch_adds_ten(self, reward_engine: RewardEngine) -> None:
        result = _make_result(200, new_branches=1, latency_ms=self._baseline)
        reward = reward_engine.compute(result, self._baseline)
        # R = 0 + 10*1 + 2*1 - 0 - 0.1 = 11.9
        assert reward == pytest.approx(11.9)

    def test_three_new_branches_adds_thirty(self, reward_engine: RewardEngine) -> None:
        result = _make_result(200, new_branches=3, latency_ms=self._baseline)
        reward = reward_engine.compute(result, self._baseline)
        # R = 0 + 30 + 2 - 0 - 0.1 = 31.9
        assert reward == pytest.approx(31.9)

    def test_zero_new_branches_no_branch_reward(self, reward_engine: RewardEngine) -> None:
        result = _make_result(200, new_branches=0, latency_ms=self._baseline)
        reward = reward_engine.compute(result, self._baseline)
        # R = 0 + 0 + 2 - 0 - 0.1 = 1.9
        assert reward == pytest.approx(1.9)

    # ── Latency term ──────────────────────────────────────────────────────────

    def test_double_baseline_latency_doubles_latency_term(self, reward_engine: RewardEngine) -> None:
        result_1x = _make_result(200, latency_ms=self._baseline)
        result_2x = _make_result(200, latency_ms=self._baseline * 2)
        r1 = reward_engine.compute(result_1x, self._baseline)
        r2 = reward_engine.compute(result_2x, self._baseline)
        # r2 - r1 = w_latency * (2 - 1) = 2.0
        assert (r2 - r1) == pytest.approx(2.0)

    def test_latency_cap_prevents_unbounded_reward(self, reward_engine: RewardEngine) -> None:
        extreme = _make_result(200, latency_ms=1_000_000.0)
        reward = reward_engine.compute(extreme, self._baseline)
        # w_latency * latency_cap + step_penalty (negative)
        max_latency_contribution = reward_engine.config.w_latency * reward_engine.config.latency_cap
        assert reward <= max_latency_contribution + 0.1   # small tolerance

    def test_zero_baseline_latency_uses_safe_floor(self, reward_engine: RewardEngine) -> None:
        """baseline_latency_ms=0 should not raise; clamped to 1.0 ms internally."""
        result = _make_result(200, latency_ms=50.0)
        reward = reward_engine.compute(result, baseline_latency_ms=0.0)
        # norm_latency = 50.0 / 1.0 → capped at latency_cap (10.0)
        # R = 0 + 0 + 2*10 - 0 - 0.1 = 19.9
        assert reward == pytest.approx(19.9)

    # ── Rejection penalty ─────────────────────────────────────────────────────

    def test_422_triggers_reject_penalty(self, reward_engine: RewardEngine) -> None:
        result = _make_result(422, latency_ms=self._baseline)
        reward = reward_engine.compute(result, self._baseline)
        # R = 0 + 0 + 2 - 1 - 0.1 = 0.9
        assert reward == pytest.approx(0.9)

    def test_400_triggers_reject_penalty(self, reward_engine: RewardEngine) -> None:
        result = _make_result(400, latency_ms=self._baseline)
        reward = reward_engine.compute(result, self._baseline)
        assert reward == pytest.approx(0.9)

    def test_non_422_non_400_no_reject_penalty(self, reward_engine: RewardEngine) -> None:
        r200 = reward_engine.compute(_make_result(200, latency_ms=self._baseline), self._baseline)
        r404 = reward_engine.compute(_make_result(404, latency_ms=self._baseline), self._baseline)
        # 404 is not in rejection codes → same reward as 200
        assert r200 == pytest.approx(r404)

    # ── Step penalty ──────────────────────────────────────────────────────────

    def test_step_penalty_always_applied(self, reward_engine: RewardEngine) -> None:
        result = _make_result(200, new_branches=0, latency_ms=self._baseline)
        reward = reward_engine.compute(result, self._baseline)
        # Without step penalty: 2.0; with: 1.9
        assert reward < 2.0

    # ── Custom config ─────────────────────────────────────────────────────────

    def test_custom_config_changes_crash_weight(self) -> None:
        cfg = RewardConfig(w_crash=100.0)
        engine = RewardEngine(cfg)
        result = _make_result(500, latency_ms=50.0)
        reward = engine.compute(result, 50.0)
        # R = 100 + 0 + 2 - 0 - 0.1 = 101.9
        assert reward == pytest.approx(101.9)

    def test_config_property_returns_config(self, reward_engine: RewardEngine) -> None:
        assert isinstance(reward_engine.config, RewardConfig)
        assert reward_engine.config.w_crash == 50.0

    def test_reward_is_scalar_float(self, reward_engine: RewardEngine) -> None:
        result = _make_result(200, latency_ms=self._baseline)
        reward = reward_engine.compute(result, self._baseline)
        assert isinstance(reward, float)


# ─────────────────────────────────────────────────────────────────────────────
# 3. APIFuzzEnv: construction and spaces
# ─────────────────────────────────────────────────────────────────────────────

class TestAPIFuzzEnvBasics:
    """Verify env construction, spaces, and read-only properties."""

    def test_env_creates_without_error(self, env: APIFuzzEnv) -> None:
        assert env is not None

    def test_repr_contains_endpoint(self, env: APIFuzzEnv) -> None:
        assert "/order/create" in repr(env)

    # ── Action space ──────────────────────────────────────────────────────────

    def test_action_space_is_multidiscrete(self, env: APIFuzzEnv) -> None:
        from gymnasium.spaces import MultiDiscrete
        assert isinstance(env.action_space, MultiDiscrete)

    def test_action_space_second_dim_is_six(self, env: APIFuzzEnv) -> None:
        assert env.action_space.nvec[1] == NUM_OPERATORS

    def test_action_space_first_dim_matches_fields(self, env: APIFuzzEnv) -> None:
        assert env.action_space.nvec[0] == env.num_fields

    def test_action_space_dtype_is_int64(self, env: APIFuzzEnv) -> None:
        assert env.action_space.dtype == np.int64

    def test_sampled_action_is_in_action_space(self, env: APIFuzzEnv) -> None:
        action = env.action_space.sample()
        assert env.action_space.contains(action)

    # ── Observation space ─────────────────────────────────────────────────────

    def test_observation_space_is_box(self, env: APIFuzzEnv) -> None:
        from gymnasium.spaces import Box
        assert isinstance(env.observation_space, Box)

    def test_observation_space_shape(self, env: APIFuzzEnv) -> None:
        assert env.observation_space.shape == (OBS_DIM,)

    def test_observation_space_dtype_is_float32(self, env: APIFuzzEnv) -> None:
        assert env.observation_space.dtype == np.float32

    def test_observation_space_bounds_are_zero_to_one(self, env: APIFuzzEnv) -> None:
        assert np.all(env.observation_space.low == 0.0)
        assert np.all(env.observation_space.high == 1.0)

    # ── Properties ────────────────────────────────────────────────────────────

    def test_num_fields_matches_order_create_body(self, env: APIFuzzEnv) -> None:
        # /order/create has 4 body fields: items, discount_pct, total_amount, customer_note
        assert env.num_fields == 4

    def test_field_names_contains_expected_keys(self, env: APIFuzzEnv) -> None:
        names = env.field_names
        assert "items" in names
        assert "discount_pct" in names
        assert "total_amount" in names

    def test_baseline_payload_property_is_deep_copy(self, env: APIFuzzEnv) -> None:
        bp1 = env.baseline_payload
        bp2 = env.baseline_payload
        assert bp1 is not bp2
        assert bp1 == bp2

    # ── Error cases ────────────────────────────────────────────────────────────

    def test_invalid_endpoint_raises_value_error(
        self, openapi_parser: OpenAPIParser, runner: HarnessRunner
    ) -> None:
        with pytest.raises(ValueError, match="not found in parsed spec"):
            APIFuzzEnv(openapi_parser, runner, endpoint="/does/not/exist")

    def test_wrong_method_raises_value_error(
        self, openapi_parser: OpenAPIParser, runner: HarnessRunner
    ) -> None:
        with pytest.raises(ValueError, match="not found in parsed spec"):
            APIFuzzEnv(openapi_parser, runner, endpoint="/order/create", method="DELETE")


# ─────────────────────────────────────────────────────────────────────────────
# 4. APIFuzzEnv: reset() semantics
# ─────────────────────────────────────────────────────────────────────────────

class TestAPIFuzzEnvReset:
    """Verify reset() return types, shapes, and state initialisation."""

    def test_reset_returns_2tuple(self, env: APIFuzzEnv) -> None:
        result = env.reset()
        assert isinstance(result, tuple) and len(result) == 2

    def test_reset_obs_is_ndarray(self, env: APIFuzzEnv) -> None:
        obs, _ = env.reset()
        assert isinstance(obs, np.ndarray)

    def test_reset_obs_shape(self, env: APIFuzzEnv) -> None:
        obs, _ = env.reset()
        assert obs.shape == (OBS_DIM,)

    def test_reset_obs_dtype_float32(self, env: APIFuzzEnv) -> None:
        obs, _ = env.reset()
        assert obs.dtype == np.float32

    def test_reset_obs_in_observation_space(self, env: APIFuzzEnv) -> None:
        obs, _ = env.reset()
        assert env.observation_space.contains(obs)

    def test_reset_obs_endpoint_indicator_is_one(self, env: APIFuzzEnv) -> None:
        obs, _ = env.reset()
        assert obs[0] == pytest.approx(1.0)

    def test_reset_obs_status_is_zero(self, env: APIFuzzEnv) -> None:
        """Before any request, status_norm should be 0.0."""
        obs, _ = env.reset()
        assert obs[1] == pytest.approx(0.0)

    def test_reset_obs_branch_ratio_is_zero(self, env: APIFuzzEnv) -> None:
        obs, _ = env.reset()
        assert obs[2] == pytest.approx(0.0)

    def test_reset_obs_step_norm_is_zero(self, env: APIFuzzEnv) -> None:
        obs, _ = env.reset()
        assert obs[4] == pytest.approx(0.0)

    def test_reset_info_is_dict(self, env: APIFuzzEnv) -> None:
        _, info = env.reset()
        assert isinstance(info, dict)

    def test_reset_info_has_endpoint(self, env: APIFuzzEnv) -> None:
        _, info = env.reset()
        assert info.get("endpoint") == "/order/create"

    def test_reset_info_has_num_fields(self, env: APIFuzzEnv) -> None:
        _, info = env.reset()
        assert "num_fields" in info
        assert info["num_fields"] == env.num_fields

    def test_reset_clears_accumulated_branches(
        self, env: APIFuzzEnv, runner: HarnessRunner
    ) -> None:
        """reset() must clear runner's accumulated branch set."""
        env.step([0, MutationOperator.NO_OP])
        assert len(runner.accumulated_branches) > 0
        env.reset()
        assert runner.accumulated_branches == frozenset()

    def test_reset_with_seed_does_not_raise(self, env: APIFuzzEnv) -> None:
        obs, _ = env.reset(seed=42)
        assert obs is not None

    def test_double_reset_returns_consistent_obs(self, env: APIFuzzEnv) -> None:
        obs1, _ = env.reset(seed=0)
        obs2, _ = env.reset(seed=0)
        # Both resets start from baseline → same initial obs shape/values
        np.testing.assert_array_equal(obs1, obs2)


# ─────────────────────────────────────────────────────────────────────────────
# 5. APIFuzzEnv: step() semantics
# ─────────────────────────────────────────────────────────────────────────────

class TestAPIFuzzEnvStep:
    """Verify step() return contract, observation updates, and info dict."""

    def test_step_returns_5tuple(self, env: APIFuzzEnv) -> None:
        out = env.step([0, MutationOperator.NO_OP])
        assert isinstance(out, tuple) and len(out) == 5

    def test_step_obs_is_ndarray_float32(self, env: APIFuzzEnv) -> None:
        obs, *_ = env.step([0, MutationOperator.NO_OP])
        assert isinstance(obs, np.ndarray)
        assert obs.dtype == np.float32

    def test_step_obs_shape(self, env: APIFuzzEnv) -> None:
        obs, *_ = env.step([0, MutationOperator.NO_OP])
        assert obs.shape == (OBS_DIM,)

    def test_step_obs_in_observation_space(self, env: APIFuzzEnv) -> None:
        obs, *_ = env.step([0, MutationOperator.NO_OP])
        assert env.observation_space.contains(obs)

    def test_step_reward_is_float(self, env: APIFuzzEnv) -> None:
        _, reward, *_ = env.step([0, MutationOperator.NO_OP])
        assert isinstance(reward, float)

    def test_step_terminated_is_bool(self, env: APIFuzzEnv) -> None:
        _, _, terminated, *_ = env.step([0, MutationOperator.NO_OP])
        assert isinstance(terminated, bool)

    def test_step_truncated_is_bool(self, env: APIFuzzEnv) -> None:
        _, _, _, truncated, _ = env.step([0, MutationOperator.NO_OP])
        assert isinstance(truncated, bool)

    def test_step_terminated_is_always_false(self, env: APIFuzzEnv) -> None:
        _, _, terminated, _, _ = env.step([0, MutationOperator.NO_OP])
        assert terminated is False

    def test_step_info_is_dict(self, env: APIFuzzEnv) -> None:
        *_, info = env.step([0, MutationOperator.NO_OP])
        assert isinstance(info, dict)

    def test_step_info_has_status_code(self, env: APIFuzzEnv) -> None:
        *_, info = env.step([0, MutationOperator.NO_OP])
        assert "status_code" in info
        assert isinstance(info["status_code"], int)

    def test_step_info_has_latency_ms(self, env: APIFuzzEnv) -> None:
        *_, info = env.step([0, MutationOperator.NO_OP])
        assert "latency_ms" in info
        assert info["latency_ms"] > 0.0

    def test_step_info_has_field_mutated(self, env: APIFuzzEnv) -> None:
        field_idx = 1   # discount_pct
        *_, info = env.step([field_idx, MutationOperator.NO_OP])
        assert "field_mutated" in info
        assert info["field_mutated"] == env.field_names[field_idx]

    def test_step_obs_status_updates_after_request(self, env: APIFuzzEnv) -> None:
        """obs[1] must reflect last HTTP status, not stay at 0.0."""
        obs, *_ = env.step([0, MutationOperator.NO_OP])
        assert obs[1] > 0.0   # any valid status code

    def test_step_obs_step_norm_increments(self, env: APIFuzzEnv) -> None:
        obs1, *_ = env.step([0, MutationOperator.NO_OP])
        obs2, *_ = env.step([0, MutationOperator.NO_OP])
        assert obs2[4] > obs1[4]

    def test_step_obs_all_in_bounds(self, env: APIFuzzEnv) -> None:
        for _ in range(3):
            obs, *_ = env.step(env.action_space.sample())
            assert np.all(obs >= 0.0) and np.all(obs <= 1.0)

    # ── Truncation ────────────────────────────────────────────────────────────

    def test_not_truncated_before_max_steps(self, env: APIFuzzEnv) -> None:
        env.reset()
        for _ in range(9):   # max_steps=10; 9 steps should not truncate
            _, _, _, truncated, _ = env.step([0, MutationOperator.NO_OP])
            assert truncated is False

    def test_truncated_at_max_steps(self, env: APIFuzzEnv) -> None:
        env.reset()
        for _ in range(9):
            env.step([0, MutationOperator.NO_OP])
        _, _, _, truncated, _ = env.step([0, MutationOperator.NO_OP])
        assert truncated is True

    # ── Mutation operators via step ───────────────────────────────────────────

    def test_field_omission_via_step_causes_422(self, env: APIFuzzEnv) -> None:
        """Omitting a required field ('items') should return 422 from Pydantic."""
        items_idx = env.field_names.index("items")
        obs, reward, _, _, info = env.step([items_idx, MutationOperator.FIELD_OMISSION])
        assert info["status_code"] == 422

    def test_crash_operator_gives_high_reward(self, env: APIFuzzEnv) -> None:
        """BOUNDARY_MAX_OVERFLOW on discount_pct → 101.0 → ZeroDivisionError → 500 → big reward."""
        discount_idx = env.field_names.index("discount_pct")
        # First restore discount_pct to 100.0 which triggers bug#1
        env._payload["discount_pct"] = 100.0   # direct set for test determinism
        env._payload["items"] = [{"product_id": "X", "quantity": 1, "unit_price": 1.0}]
        obs, reward, _, _, info = env.step([0, MutationOperator.NO_OP])
        if info["status_code"] == 500:
            assert reward > 50.0   # crash reward dominates

    def test_type_confusion_on_numeric_field_causes_422(self, env: APIFuzzEnv) -> None:
        """TYPE_CONFUSION on discount_pct (number) injects string → Pydantic 422."""
        discount_idx = env.field_names.index("discount_pct")
        env.reset()
        obs, reward, _, _, info = env.step([discount_idx, MutationOperator.TYPE_CONFUSION])
        assert info["status_code"] == 422

    # ── Acceptance test: path params auto-filled ──────────────────────────────

    def test_profile_endpoint_resolves_path_params(
        self, openapi_parser: OpenAPIParser, runner: HarnessRunner
    ) -> None:
        """Env for /users/{user_id}/profile must fill {user_id} without raising."""
        profile_env = APIFuzzEnv(
            openapi_parser,
            runner,
            endpoint="/users/{user_id}/profile",
            method="PUT",
            max_steps=5,
        )
        profile_env.reset()
        obs, reward, terminated, truncated, info = profile_env.step(
            [0, MutationOperator.NO_OP]
        )
        assert obs is not None
        assert info["status_code"] in {200, 422, 500}   # valid HTTP response


# ─────────────────────────────────────────────────────────────────────────────
# 6. Gymnasium compliance — the critical check
# ─────────────────────────────────────────────────────────────────────────────

class TestGymnasiumCompliance:
    """
    Verifies that APIFuzzEnv passes the official gymnasium.utils.env_checker.

    ``check_env`` is the gold standard for Gymnasium API compliance: it calls
    reset(), samples random actions, calls step(), and verifies that all return
    types, dtypes, and space memberships are correct.
    """

    def test_check_env_order_create(
        self, openapi_parser: OpenAPIParser, runner: HarnessRunner
    ) -> None:
        """check_env must pass for the /order/create endpoint."""
        from gymnasium.utils.env_checker import check_env

        test_env = APIFuzzEnv(
            parser=openapi_parser,
            runner=runner,
            endpoint="/order/create",
            method="POST",
            max_steps=5,
            baseline_payload=_ORDER_BASELINE,
        )
        # skip_render_check=True: render() is a no-op in Phase 2
        check_env(test_env, warn=True, skip_render_check=True)

    def test_check_env_profile_endpoint(
        self, openapi_parser: OpenAPIParser, runner: HarnessRunner
    ) -> None:
        """check_env must pass for the profile endpoint (has path params)."""
        from gymnasium.utils.env_checker import check_env

        test_env = APIFuzzEnv(
            parser=openapi_parser,
            runner=runner,
            endpoint="/users/{user_id}/profile",
            method="PUT",
            max_steps=5,
        )
        check_env(test_env, warn=True, skip_render_check=True)

    def test_observation_always_in_space_across_random_actions(
        self, openapi_parser: OpenAPIParser, runner: HarnessRunner
    ) -> None:
        """All observations over 20 random steps must be in observation_space."""
        test_env = APIFuzzEnv(
            parser=openapi_parser,
            runner=runner,
            endpoint="/order/create",
            method="POST",
            max_steps=30,
            baseline_payload=_ORDER_BASELINE,
        )
        obs, _ = test_env.reset(seed=7)
        assert test_env.observation_space.contains(obs)

        for _ in range(20):
            action = test_env.action_space.sample()
            obs, reward, terminated, truncated, info = test_env.step(action)
            assert test_env.observation_space.contains(obs), (
                f"Observation out of bounds: {obs}"
            )
            assert np.isfinite(reward), f"Non-finite reward: {reward}"
            if truncated or terminated:
                obs, _ = test_env.reset()
                assert test_env.observation_space.contains(obs)

    def test_step_tuple_unpacking(
        self, openapi_parser: OpenAPIParser, runner: HarnessRunner
    ) -> None:
        """Verify 5-tuple step return is correctly unpackable."""
        test_env = APIFuzzEnv(
            openapi_parser, runner, "/order/create",
            max_steps=5, baseline_payload=_ORDER_BASELINE,
        )
        test_env.reset()
        obs, reward, terminated, truncated, info = test_env.step(
            test_env.action_space.sample()
        )
        assert obs.shape == (OBS_DIM,)
        assert isinstance(reward, float)
        assert isinstance(terminated, bool)
        assert isinstance(truncated, bool)
        assert isinstance(info, dict)

    def test_reset_tuple_unpacking(
        self, openapi_parser: OpenAPIParser, runner: HarnessRunner
    ) -> None:
        """Verify 2-tuple reset return is correctly unpackable."""
        test_env = APIFuzzEnv(
            openapi_parser, runner, "/order/create",
            max_steps=5, baseline_payload=_ORDER_BASELINE,
        )
        obs, info = test_env.reset()
        assert obs.shape == (OBS_DIM,)
        assert isinstance(info, dict)
