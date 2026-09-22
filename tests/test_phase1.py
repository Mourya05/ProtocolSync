"""
tests/test_phase1.py
──────────────────────────────────────────────────────────────────────────────
End-to-end integration tests for Protocol-Sync Phase 1:
  Deterministic Schema Ingestion & Target Instrumentation Harness.

Test groups
───────────
  1. TestParser             — ParsedAPI structure, field constraints, $ref resolution
  2. TestHappyPath          — valid payloads → expected 2xx status codes + non-zero coverage
  3. TestEdgeCaseCrashes    — deliberate bugs detected as HTTP 500 (not masked)
  4. TestCoverageDelta      — new_branches is empty on identical repeated requests
  5. TestAccumulationMonotonicity — accumulated_branches is non-decreasing across calls

Run with:
    pytest tests/test_phase1.py -v --tb=short
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from core.parser import (
    EndpointSpec,
    FieldSpec,
    ParsedAPI,
    parse_file,
    parse_string,
)
from core.tracer import ExecutionResult
from harness.runner import HarnessRunner
from target.app import app as target_app

# ─────────────────────────────────────────────────────────────────────────────
# Paths
# ─────────────────────────────────────────────────────────────────────────────

_OPENAPI_JSON: Path = Path(__file__).parent.parent / "target" / "openapi.json"


# ─────────────────────────────────────────────────────────────────────────────
# Fixtures
# ─────────────────────────────────────────────────────────────────────────────

@pytest.fixture(scope="module")
def parsed_api() -> ParsedAPI:
    """Parse target/openapi.json once per test session."""
    return parse_file(_OPENAPI_JSON)


@pytest.fixture()
def runner() -> HarnessRunner:
    """
    Fresh HarnessRunner with a zeroed accumulated-branch state per test.

    Resetting between tests guarantees that ``new_branches`` computations
    are isolated and cannot be influenced by earlier test runs.
    """
    r = HarnessRunner(target_app)
    r.reset()
    return r


# ─────────────────────────────────────────────────────────────────────────────
# 1. Parser unit tests
# ─────────────────────────────────────────────────────────────────────────────

class TestParser:
    """Validate that the OpenAPI parser correctly ingests target/openapi.json."""

    # ── Top-level structure ────────────────────────────────────────────────

    def test_parse_file_returns_parsed_api(self, parsed_api: ParsedAPI) -> None:
        assert isinstance(parsed_api, ParsedAPI)

    def test_title_matches_app(self, parsed_api: ParsedAPI) -> None:
        assert parsed_api.title == "Protocol-Sync Target Service"

    def test_version_matches_app(self, parsed_api: ParsedAPI) -> None:
        assert parsed_api.version == "0.1.0"

    def test_base_url_present(self, parsed_api: ParsedAPI) -> None:
        assert parsed_api.base_url.startswith("http")

    def test_exactly_three_endpoints(self, parsed_api: ParsedAPI) -> None:
        assert len(parsed_api.endpoints) == 3

    def test_all_endpoints_are_endpoint_spec(self, parsed_api: ParsedAPI) -> None:
        for ep in parsed_api.endpoints:
            assert isinstance(ep, EndpointSpec)

    # ── /order/create ──────────────────────────────────────────────────────

    def test_order_create_endpoint_found(self, parsed_api: ParsedAPI) -> None:
        ep = parsed_api.endpoint("/order/create", "post")
        assert ep is not None

    def test_order_create_method_is_post(self, parsed_api: ParsedAPI) -> None:
        ep = parsed_api.endpoint("/order/create", "post")
        assert ep is not None
        assert ep.method == "post"

    def test_order_create_operation_id(self, parsed_api: ParsedAPI) -> None:
        ep = parsed_api.endpoint("/order/create", "post")
        assert ep is not None
        assert ep.operation_id == "create_order"

    def test_order_create_body_required(self, parsed_api: ParsedAPI) -> None:
        ep = parsed_api.endpoint("/order/create", "post")
        assert ep is not None
        assert ep.request_body_required is True

    def test_order_create_body_fields_present(self, parsed_api: ParsedAPI) -> None:
        ep = parsed_api.endpoint("/order/create", "post")
        assert ep is not None
        names = {f.name for f in ep.body_fields}
        assert "items" in names
        assert "discount_pct" in names
        assert "total_amount" in names

    def test_discount_pct_type_is_number(self, parsed_api: ParsedAPI) -> None:
        ep = parsed_api.endpoint("/order/create", "post")
        assert ep is not None
        f = next(x for x in ep.body_fields if x.name == "discount_pct")
        assert f.type == "number"

    def test_discount_pct_minimum_constraint(self, parsed_api: ParsedAPI) -> None:
        ep = parsed_api.endpoint("/order/create", "post")
        assert ep is not None
        f = next(x for x in ep.body_fields if x.name == "discount_pct")
        assert f.minimum == 0.0

    def test_discount_pct_maximum_constraint(self, parsed_api: ParsedAPI) -> None:
        ep = parsed_api.endpoint("/order/create", "post")
        assert ep is not None
        f = next(x for x in ep.body_fields if x.name == "discount_pct")
        assert f.maximum == 100.0

    def test_total_amount_type_is_number(self, parsed_api: ParsedAPI) -> None:
        ep = parsed_api.endpoint("/order/create", "post")
        assert ep is not None
        f = next(x for x in ep.body_fields if x.name == "total_amount")
        assert f.type == "number"

    def test_items_type_is_array(self, parsed_api: ParsedAPI) -> None:
        ep = parsed_api.endpoint("/order/create", "post")
        assert ep is not None
        f = next(x for x in ep.body_fields if x.name == "items")
        assert f.type == "array"

    def test_order_create_has_201_response_schema(self, parsed_api: ParsedAPI) -> None:
        ep = parsed_api.endpoint("/order/create", "post")
        assert ep is not None
        assert "201" in ep.response_schemas
        assert isinstance(ep.response_schemas["201"], dict)

    # ── /users/{user_id}/profile ───────────────────────────────────────────

    def test_profile_endpoint_found(self, parsed_api: ParsedAPI) -> None:
        ep = parsed_api.endpoint("/users/{user_id}/profile", "put")
        assert ep is not None

    def test_profile_operation_id(self, parsed_api: ParsedAPI) -> None:
        ep = parsed_api.endpoint("/users/{user_id}/profile", "put")
        assert ep is not None
        assert ep.operation_id == "update_profile"

    def test_profile_has_one_path_param(self, parsed_api: ParsedAPI) -> None:
        ep = parsed_api.endpoint("/users/{user_id}/profile", "put")
        assert ep is not None
        assert len(ep.path_params) == 1

    def test_profile_path_param_is_user_id(self, parsed_api: ParsedAPI) -> None:
        ep = parsed_api.endpoint("/users/{user_id}/profile", "put")
        assert ep is not None
        assert ep.path_params[0].name == "user_id"
        assert ep.path_params[0].required is True

    def test_profile_body_fields_present(self, parsed_api: ParsedAPI) -> None:
        ep = parsed_api.endpoint("/users/{user_id}/profile", "put")
        assert ep is not None
        names = {f.name for f in ep.body_fields}
        assert "username" in names
        assert "start_date" in names
        assert "end_date" in names

    def test_start_date_has_pattern(self, parsed_api: ParsedAPI) -> None:
        ep = parsed_api.endpoint("/users/{user_id}/profile", "put")
        assert ep is not None
        f = next(x for x in ep.body_fields if x.name == "start_date")
        assert f.pattern is not None and len(f.pattern) > 0

    def test_profile_required_fields(self, parsed_api: ParsedAPI) -> None:
        ep = parsed_api.endpoint("/users/{user_id}/profile", "put")
        assert ep is not None
        required = {f.name for f in ep.body_fields if f.required}
        assert "username" in required
        assert "start_date" in required
        assert "end_date" in required

    # ── /transfer ──────────────────────────────────────────────────────────

    def test_transfer_endpoint_found(self, parsed_api: ParsedAPI) -> None:
        ep = parsed_api.endpoint("/transfer", "post")
        assert ep is not None

    def test_transfer_operation_id(self, parsed_api: ParsedAPI) -> None:
        ep = parsed_api.endpoint("/transfer", "post")
        assert ep is not None
        assert ep.operation_id == "transfer"

    def test_transfer_required_fields(self, parsed_api: ParsedAPI) -> None:
        ep = parsed_api.endpoint("/transfer", "post")
        assert ep is not None
        required = {f.name for f in ep.body_fields if f.required}
        assert "from_account" in required
        assert "to_account" in required
        assert "amount" in required

    def test_transfer_currency_has_enum(self, parsed_api: ParsedAPI) -> None:
        ep = parsed_api.endpoint("/transfer", "post")
        assert ep is not None
        f = next(x for x in ep.body_fields if x.name == "currency")
        assert f.enum is not None
        assert "USD" in f.enum

    def test_transfer_amount_is_string(self, parsed_api: ParsedAPI) -> None:
        ep = parsed_api.endpoint("/transfer", "post")
        assert ep is not None
        f = next(x for x in ep.body_fields if x.name == "amount")
        assert f.type == "string"

    # ── Cross-cutting parser tests ─────────────────────────────────────────

    def test_ref_resolution_removes_ref_keys(self, parsed_api: ParsedAPI) -> None:
        """After $ref resolution, no EndpointSpec.request_body_schema should contain '$ref'."""
        for ep in parsed_api.endpoints:
            if ep.request_body_schema:
                assert "$ref" not in ep.request_body_schema, (
                    f"Unresolved $ref found in {ep.path} {ep.method} request body schema."
                )

    def test_ref_resolution_inlines_properties(self, parsed_api: ParsedAPI) -> None:
        ep = parsed_api.endpoint("/order/create", "post")
        assert ep is not None
        assert ep.request_body_schema is not None
        assert "properties" in ep.request_body_schema

    def test_parse_string_json_roundtrip(self) -> None:
        raw_text = _OPENAPI_JSON.read_text(encoding="utf-8")
        api = parse_string(raw_text, fmt="json")
        assert isinstance(api, ParsedAPI)
        assert api.title == "Protocol-Sync Target Service"
        assert len(api.endpoints) == 3

    def test_parse_string_yaml_roundtrip(self) -> None:
        import yaml as _yaml
        raw = json.loads(_OPENAPI_JSON.read_text(encoding="utf-8"))
        yaml_text = _yaml.dump(raw)
        api = parse_string(yaml_text, fmt="yaml")
        assert isinstance(api, ParsedAPI)
        assert len(api.endpoints) == 3

    def test_unsupported_version_raises_value_error(self) -> None:
        bad_doc = json.dumps({
            "openapi": "2.0.0",
            "info": {"title": "Old", "version": "1.0"},
            "paths": {},
        })
        with pytest.raises(ValueError, match="Unsupported OpenAPI version"):
            parse_string(bad_doc, fmt="json")

    def test_missing_openapi_field_raises(self) -> None:
        bad_doc = json.dumps({"info": {"title": "X", "version": "1"}, "paths": {}})
        with pytest.raises(ValueError, match="missing required 'openapi'"):
            parse_string(bad_doc, fmt="json")

    def test_empty_string_raises_value_error(self) -> None:
        with pytest.raises(ValueError, match="Empty"):
            parse_string("   ", fmt="json")

    def test_invalid_json_raises_value_error(self) -> None:
        with pytest.raises(ValueError, match="Failed to parse"):
            parse_string("{broken json!!!", fmt="json")

    def test_invalid_yaml_raises_value_error(self) -> None:
        with pytest.raises(ValueError, match="Failed to parse"):
            parse_string(":\n  - [unclosed", fmt="yaml")

    def test_bad_ref_raises_key_error(self) -> None:
        doc = json.dumps({
            "openapi": "3.0.3",
            "info": {"title": "T", "version": "1"},
            "paths": {
                "/x": {
                    "get": {
                        "requestBody": {
                            "content": {
                                "application/json": {
                                    "schema": {"$ref": "#/components/schemas/DoesNotExist"}
                                }
                            }
                        },
                        "responses": {"200": {"description": "ok"}},
                    }
                }
            },
        })
        with pytest.raises(KeyError):
            parse_string(doc, fmt="json")

    def test_external_ref_raises_value_error(self) -> None:
        doc = json.dumps({
            "openapi": "3.0.3",
            "info": {"title": "T", "version": "1"},
            "paths": {
                "/x": {
                    "get": {
                        "requestBody": {
                            "content": {
                                "application/json": {
                                    "schema": {"$ref": "./external.json#/Foo"}
                                }
                            }
                        },
                        "responses": {"200": {"description": "ok"}},
                    }
                }
            },
        })
        with pytest.raises(ValueError, match="External \\$ref"):
            parse_string(doc, fmt="json")

    def test_field_spec_is_hashable(self, parsed_api: ParsedAPI) -> None:
        ep = parsed_api.endpoint("/order/create", "post")
        assert ep is not None
        # FieldSpec is frozen=True → must be usable in a set
        field_set = set(ep.body_fields)
        assert len(field_set) == len(ep.body_fields)

    def test_endpoint_spec_is_frozen(self, parsed_api: ParsedAPI) -> None:
        ep = parsed_api.endpoint("/order/create", "post")
        assert ep is not None
        with pytest.raises((AttributeError, TypeError)):
            ep.path = "/new/path"  # type: ignore[misc]


# ─────────────────────────────────────────────────────────────────────────────
# 2. Happy-path harness tests
# ─────────────────────────────────────────────────────────────────────────────

class TestHappyPath:
    """Valid payloads should return expected 2xx status codes with non-zero coverage."""

    _ORDER = {
        "items": [{"product_id": "PROD-001", "quantity": 2, "unit_price": 49.99}],
        "discount_pct": 10.0,
        "total_amount": 100.0,
    }
    _PROFILE = {
        "username": "alice",
        "start_date": "2024-01-01",
        "end_date": "2024-12-31",
    }
    _TRANSFER = {
        "from_account": "ACC-001",
        "to_account": "ACC-002",
        "amount": "100.00",
    }

    def test_create_order_returns_201(self, runner: HarnessRunner) -> None:
        result = runner.execute("POST", "/order/create", json=self._ORDER)
        assert result.status_code == 201

    def test_create_order_returns_execution_result(self, runner: HarnessRunner) -> None:
        result = runner.execute("POST", "/order/create", json=self._ORDER)
        assert isinstance(result, ExecutionResult)

    def test_create_order_covered_branches_nonzero(self, runner: HarnessRunner) -> None:
        result = runner.execute("POST", "/order/create", json=self._ORDER)
        assert len(result.covered_branches) > 0

    def test_create_order_covered_lines_nonzero(self, runner: HarnessRunner) -> None:
        result = runner.execute("POST", "/order/create", json=self._ORDER)
        assert len(result.covered_lines) > 0

    def test_create_order_latency_positive(self, runner: HarnessRunner) -> None:
        result = runner.execute("POST", "/order/create", json=self._ORDER)
        assert result.latency_ms > 0.0

    def test_create_order_latency_is_float(self, runner: HarnessRunner) -> None:
        result = runner.execute("POST", "/order/create", json=self._ORDER)
        assert isinstance(result.latency_ms, float)

    def test_update_profile_returns_200(self, runner: HarnessRunner) -> None:
        result = runner.execute("PUT", "/users/USR-42/profile", json=self._PROFILE)
        assert result.status_code == 200

    def test_update_profile_has_coverage(self, runner: HarnessRunner) -> None:
        result = runner.execute("PUT", "/users/USR-42/profile", json=self._PROFILE)
        assert len(result.covered_branches) > 0

    def test_transfer_returns_200(self, runner: HarnessRunner) -> None:
        result = runner.execute("POST", "/transfer", json=self._TRANSFER)
        assert result.status_code == 200

    def test_transfer_has_coverage(self, runner: HarnessRunner) -> None:
        result = runner.execute("POST", "/transfer", json=self._TRANSFER)
        assert len(result.covered_branches) > 0

    def test_execution_result_is_frozen(self, runner: HarnessRunner) -> None:
        result = runner.execute("POST", "/order/create", json=self._ORDER)
        with pytest.raises((AttributeError, TypeError)):
            result.status_code = 999  # type: ignore[misc]

    def test_covered_lines_are_integers(self, runner: HarnessRunner) -> None:
        result = runner.execute("POST", "/order/create", json=self._ORDER)
        for line in result.covered_lines:
            assert isinstance(line, int)

    def test_covered_branches_are_int_pairs(self, runner: HarnessRunner) -> None:
        result = runner.execute("POST", "/order/create", json=self._ORDER)
        for arc in result.covered_branches:
            assert len(arc) == 2
            assert all(isinstance(x, int) for x in arc)


# ─────────────────────────────────────────────────────────────────────────────
# 3. Edge-case crash detection
# ─────────────────────────────────────────────────────────────────────────────

class TestEdgeCaseCrashes:
    """
    Verify that the harness correctly detects all deliberate bugs as HTTP 500
    rather than masking them or raising Python exceptions.
    """

    # ── /order/create bugs ─────────────────────────────────────────────────

    def test_discount_100_causes_500(self, runner: HarnessRunner) -> None:
        """Bug #1: discount_pct=100 → ZeroDivisionError → HTTP 500."""
        payload = {
            "items": [{"product_id": "X", "quantity": 1, "unit_price": 10.0}],
            "discount_pct": 100.0,
            "total_amount": 10.0,
        }
        result = runner.execute("POST", "/order/create", json=payload)
        assert result.status_code == 500

    def test_negative_total_amount_returns_201(self, runner: HarnessRunner) -> None:
        """Bug #2: Negative total_amount slips through validation → silent 201."""
        payload = {
            "items": [{"product_id": "X", "quantity": 1, "unit_price": 10.0}],
            "discount_pct": 5.0,
            "total_amount": -50.0,
        }
        result = runner.execute("POST", "/order/create", json=payload)
        # The bug produces a wrong answer (negative discount), not a crash
        assert result.status_code == 201

    # ── /users/.../profile bugs ────────────────────────────────────────────

    def test_inverted_dates_cause_500(self, runner: HarnessRunner) -> None:
        """Bug #4: start_date > end_date → ValueError → HTTP 500."""
        payload = {
            "username": "bob",
            "start_date": "2024-12-31",
            "end_date": "2024-01-01",
        }
        result = runner.execute("PUT", "/users/USR-1/profile", json=payload)
        assert result.status_code == 500

    def test_equal_dates_cause_500(self, runner: HarnessRunner) -> None:
        """Bug #4 (boundary): start_date == end_date → delta.days == 0 → HTTP 500."""
        payload = {
            "username": "carol",
            "start_date": "2024-06-15",
            "end_date": "2024-06-15",
        }
        result = runner.execute("PUT", "/users/USR-2/profile", json=payload)
        assert result.status_code == 500

    def test_malformed_start_date_causes_500(self, runner: HarnessRunner) -> None:
        """Bug #3: Non-ISO start_date string → ValueError → HTTP 500."""
        payload = {
            "username": "dave",
            "start_date": "not-a-date",
            "end_date": "2024-12-31",
        }
        result = runner.execute("PUT", "/users/USR-3/profile", json=payload)
        assert result.status_code == 500

    def test_malformed_end_date_causes_500(self, runner: HarnessRunner) -> None:
        """Bug #3 (end_date): Non-ISO end_date string → ValueError → HTTP 500."""
        payload = {
            "username": "eve",
            "start_date": "2024-01-01",
            "end_date": "32-13-2024",
        }
        result = runner.execute("PUT", "/users/USR-4/profile", json=payload)
        assert result.status_code == 500

    def test_short_email_domain_causes_500(self, runner: HarnessRunner) -> None:
        """Bug #5: username='user@x' → domain='x' → domain[1] → IndexError → HTTP 500."""
        payload = {
            "username": "user@x",
            "start_date": "2024-01-01",
            "end_date": "2024-12-31",
        }
        result = runner.execute("PUT", "/users/USR-5/profile", json=payload)
        assert result.status_code == 500

    def test_short_numeric_username_causes_500(self, runner: HarnessRunner) -> None:
        """Bug #6: username starts with digit and len < 11 → username[10] → IndexError."""
        payload = {
            "username": "1short",    # starts with digit, len=6 < 11
            "start_date": "2024-01-01",
            "end_date": "2024-12-31",
        }
        result = runner.execute("PUT", "/users/USR-6/profile", json=payload)
        assert result.status_code == 500

    # ── /transfer bugs ─────────────────────────────────────────────────────

    def test_self_transfer_causes_500(self, runner: HarnessRunner) -> None:
        """Bug #7: from_account == to_account → AssertionError → HTTP 500."""
        payload = {
            "from_account": "ACC-001",
            "to_account": "ACC-001",
            "amount": "50.00",
        }
        result = runner.execute("POST", "/transfer", json=payload)
        assert result.status_code == 500

    def test_non_numeric_amount_causes_500(self, runner: HarnessRunner) -> None:
        """Bug #8: amount='not-a-number' → InvalidOperation → HTTP 500."""
        payload = {
            "from_account": "ACC-001",
            "to_account": "ACC-002",
            "amount": "not-a-number",
        }
        result = runner.execute("POST", "/transfer", json=payload)
        assert result.status_code == 500

    def test_inf_amount_causes_500(self, runner: HarnessRunner) -> None:
        """Bug #8: amount='Infinity' → InvalidOperation → HTTP 500."""
        payload = {
            "from_account": "ACC-001",
            "to_account": "ACC-002",
            "amount": "Infinity",
        }
        result = runner.execute("POST", "/transfer", json=payload)
        assert result.status_code == 500

    def test_overflow_amount_causes_500(self, runner: HarnessRunner) -> None:
        """Bug #9: amount > 10^38 → ArithmeticError → HTTP 500."""
        payload = {
            "from_account": "ACC-001",
            "to_account": "ACC-002",
            "amount": "1" + "0" * 39,   # 10^39 > _MAX_TRANSFER
        }
        result = runner.execute("POST", "/transfer", json=payload)
        assert result.status_code == 500

    # ── Cross-cutting ─────────────────────────────────────────────────────

    def test_crashing_request_still_yields_coverage(self, runner: HarnessRunner) -> None:
        """
        Even crash-inducing requests must produce branch data.

        The route handler executes code before the exception is raised, so
        coverage.py should capture those branches.
        """
        payload = {
            "items": [{"product_id": "X", "quantity": 1, "unit_price": 10.0}],
            "discount_pct": 100.0,
            "total_amount": 10.0,
        }
        result = runner.execute("POST", "/order/create", json=payload)
        assert result.status_code == 500
        assert len(result.covered_branches) > 0


# ─────────────────────────────────────────────────────────────────────────────
# 4. Coverage delta tests
# ─────────────────────────────────────────────────────────────────────────────

class TestCoverageDelta:
    """Verify branch-delta computation semantics."""

    _PAYLOAD = {
        "items": [{"product_id": "PROD-1", "quantity": 1, "unit_price": 10.0}],
        "discount_pct": 0.0,
        "total_amount": 10.0,
    }

    def test_first_request_yields_new_branches(self, runner: HarnessRunner) -> None:
        """On a fresh runner, every branch is new."""
        result = runner.execute("POST", "/order/create", json=self._PAYLOAD)
        assert len(result.new_branches) > 0

    def test_second_identical_request_has_no_new_branches(
        self, runner: HarnessRunner
    ) -> None:
        """Identical payload on second call → zero new branches (all already accumulated)."""
        runner.execute("POST", "/order/create", json=self._PAYLOAD)
        result2 = runner.execute("POST", "/order/create", json=self._PAYLOAD)
        assert result2.new_branches == frozenset()

    def test_new_branches_is_subset_of_covered_branches(
        self, runner: HarnessRunner
    ) -> None:
        result = runner.execute("POST", "/order/create", json=self._PAYLOAD)
        assert result.new_branches <= result.covered_branches

    def test_new_branches_is_subset_of_accumulated_branches(
        self, runner: HarnessRunner
    ) -> None:
        result = runner.execute("POST", "/order/create", json=self._PAYLOAD)
        assert result.new_branches <= result.accumulated_branches

    def test_covered_branches_is_subset_of_accumulated(
        self, runner: HarnessRunner
    ) -> None:
        result = runner.execute("POST", "/order/create", json=self._PAYLOAD)
        assert result.covered_branches <= result.accumulated_branches

    def test_new_branches_type_is_frozenset(self, runner: HarnessRunner) -> None:
        result = runner.execute("POST", "/order/create", json=self._PAYLOAD)
        assert isinstance(result.new_branches, frozenset)

    def test_accumulated_branches_type_is_frozenset(self, runner: HarnessRunner) -> None:
        result = runner.execute("POST", "/order/create", json=self._PAYLOAD)
        assert isinstance(result.accumulated_branches, frozenset)

    def test_all_branch_fields_contain_int_pairs(self, runner: HarnessRunner) -> None:
        result = runner.execute("POST", "/order/create", json=self._PAYLOAD)
        for arc_set in (result.covered_branches, result.new_branches, result.accumulated_branches):
            for arc in arc_set:
                assert len(arc) == 2
                assert all(isinstance(x, int) for x in arc)


# ─────────────────────────────────────────────────────────────────────────────
# 5. Accumulation monotonicity
# ─────────────────────────────────────────────────────────────────────────────

class TestAccumulationMonotonicity:
    """Verify that accumulated_branches is strictly non-decreasing."""

    def test_accumulated_grows_across_different_routes(
        self, runner: HarnessRunner
    ) -> None:
        payloads = [
            ("POST", "/order/create", {
                "items": [{"product_id": "A", "quantity": 1, "unit_price": 5.0}],
                "discount_pct": 0.0,
                "total_amount": 5.0,
            }),
            ("PUT", "/users/USR-99/profile", {
                "username": "alice",
                "start_date": "2024-01-01",
                "end_date": "2024-06-30",
            }),
            ("POST", "/transfer", {
                "from_account": "ACC-001",
                "to_account": "ACC-002",
                "amount": "1.00",
            }),
        ]
        prev_size = 0
        for method, path, payload in payloads:
            result = runner.execute(method, path, json=payload)
            assert len(result.accumulated_branches) >= prev_size, (
                f"accumulated_branches shrank from {prev_size} to "
                f"{len(result.accumulated_branches)} after {method} {path}"
            )
            prev_size = len(result.accumulated_branches)

    def test_accumulated_never_shrinks(self, runner: HarnessRunner) -> None:
        """After two different calls, r2.accumulated ⊇ r1.accumulated."""
        r1 = runner.execute("POST", "/order/create", json={
            "items": [{"product_id": "Z", "quantity": 1, "unit_price": 1.0}],
            "discount_pct": 5.0,
            "total_amount": 1.0,
        })
        r2 = runner.execute("PUT", "/users/U/profile", json={
            "username": "zara",
            "start_date": "2023-01-01",
            "end_date": "2023-12-31",
        })
        assert r1.accumulated_branches <= r2.accumulated_branches

    def test_accumulated_identical_requests_does_not_shrink(
        self, runner: HarnessRunner
    ) -> None:
        payload = {
            "items": [{"product_id": "P", "quantity": 2, "unit_price": 3.0}],
            "discount_pct": 0.0,
            "total_amount": 6.0,
        }
        r1 = runner.execute("POST", "/order/create", json=payload)
        r2 = runner.execute("POST", "/order/create", json=payload)
        # Identical request → accumulated must stay same or grow (never shrink)
        assert r1.accumulated_branches <= r2.accumulated_branches
        assert r2.accumulated_branches == r1.accumulated_branches  # no new branches

    def test_runner_reset_clears_accumulated(self, runner: HarnessRunner) -> None:
        runner.execute("POST", "/order/create", json={
            "items": [{"product_id": "X", "quantity": 1, "unit_price": 1.0}],
            "discount_pct": 0.0,
            "total_amount": 1.0,
        })
        assert len(runner.accumulated_branches) > 0
        runner.reset()
        assert runner.accumulated_branches == frozenset()

    def test_after_reset_branches_are_new_again(self, runner: HarnessRunner) -> None:
        """After reset(), the same request should yield new_branches > 0 again."""
        payload = {
            "items": [{"product_id": "X", "quantity": 1, "unit_price": 1.0}],
            "discount_pct": 0.0,
            "total_amount": 1.0,
        }
        r1 = runner.execute("POST", "/order/create", json=payload)
        runner.reset()
        r2 = runner.execute("POST", "/order/create", json=payload)
        assert len(r2.new_branches) > 0
        # Same route → same branches exercised
        assert r1.covered_branches == r2.covered_branches

    def test_property_matches_last_result(self, runner: HarnessRunner) -> None:
        """runner.accumulated_branches property matches result.accumulated_branches."""
        result = runner.execute("POST", "/order/create", json={
            "items": [{"product_id": "Y", "quantity": 1, "unit_price": 2.0}],
            "discount_pct": 0.0,
            "total_amount": 2.0,
        })
        assert runner.accumulated_branches == result.accumulated_branches

    def test_cross_route_coverage_union(self, runner: HarnessRunner) -> None:
        """Accumulated set after two different routes should equal union of their covered sets."""
        r1 = runner.execute("POST", "/order/create", json={
            "items": [{"product_id": "A", "quantity": 1, "unit_price": 1.0}],
            "discount_pct": 0.0,
            "total_amount": 1.0,
        })
        r2 = runner.execute("PUT", "/users/U/profile", json={
            "username": "alice",
            "start_date": "2024-01-01",
            "end_date": "2024-06-01",
        })
        expected_accumulated = r1.covered_branches | r2.covered_branches
        assert r2.accumulated_branches == expected_accumulated
