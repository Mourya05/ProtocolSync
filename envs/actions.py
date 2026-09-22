"""
envs/actions.py
──────────────────────────────────────────────────────────────────────────────
Mutation operators and the MutationEngine that applies them to JSON payloads.

The six operators are intentionally ordered from *conservative* (NO_OP) to
*destructive* (FIELD_OMISSION), making it natural for an RL policy to learn
coarse-to-fine exploration strategies.

Operator taxonomy
─────────────────
  0  NO_OP               — return payload unchanged
  1  BOUNDARY_MIN_OVERFLOW — underflow: numeric (min−1), string (""), array ([])
  2  BOUNDARY_MAX_OVERFLOW — overflow:  numeric (max+1), string ("A"×10000), array (100×None)
  3  TYPE_CONFUSION       — inject a value of the wrong JSON type
  4  SPECIAL_INJECTION    — format strings, zero-division triggers, null bytes, NaN, Infinity
  5  FIELD_OMISSION       — delete the key entirely from the payload

Design invariants
─────────────────
* ``MutationEngine.apply()`` always returns a **deep copy** — the caller's
  original payload is never mutated.
* Unknown fields (not present in payload) are handled gracefully: the field is
  added with the mutated value for all operators except NO_OP.
* Operator index arithmetic uses ``% NUM_OPERATORS`` to guard against
  out-of-range action values from exploratory RL policies.
"""
from __future__ import annotations

import copy
from enum import IntEnum
from typing import Any

from core.parser import FieldSpec


# ─────────────────────────────────────────────────────────────────────────────
# Operator catalogue
# ─────────────────────────────────────────────────────────────────────────────

class MutationOperator(IntEnum):
    """Enumeration of six mutation strategies applied by the fuzzing engine."""
    NO_OP                 = 0
    BOUNDARY_MIN_OVERFLOW = 1
    BOUNDARY_MAX_OVERFLOW = 2
    TYPE_CONFUSION        = 3
    SPECIAL_INJECTION     = 4
    FIELD_OMISSION        = 5


NUM_OPERATORS: int = len(MutationOperator)   # 6


# ─────────────────────────────────────────────────────────────────────────────
# Special injection value pools
# (deterministic: we use a stable sum-of-ASCII index, not hash(), which is
# randomised by PYTHONHASHSEED in Python 3.3+)
# ─────────────────────────────────────────────────────────────────────────────

_SPECIAL_NUMERICS: tuple[Any, ...] = (
    100,            # potential ÷0 trigger for percentage fields (discount_pct)
    0,              # explicit zero (triggers division / modulo edge cases)
    -1,             # underflow
    1e308,          # near float max — overflow
    float("nan"),   # NaN propagation
    float("inf"),   # Infinity propagation
)

_SPECIAL_STRINGS: tuple[str, ...] = (
    "0",                   # numeric disguised as string (triggers Decimal errors)
    "Infinity",            # Infinity string  → Decimal("Infinity") raises
    "nan",                 # NaN string       → Decimal("nan") raises
    "not-a-date",          # malformed ISO date → date.fromisoformat raises
    "x@x",                 # single-char domain → domain[1] IndexError
    "\x00",                # null byte injection
    "%s%s%s%n",            # printf format string
    "../../../../etc/passwd",  # path traversal
    "",                    # empty string (min-boundary for strings)
    "null",                # JSON-null-as-string
)


def _stable_index(field_name: str, pool_size: int) -> int:
    """Return a deterministic pool index based on the field name's character sum."""
    return sum(ord(c) for c in field_name) % pool_size


# ─────────────────────────────────────────────────────────────────────────────
# Mutation engine
# ─────────────────────────────────────────────────────────────────────────────

class MutationEngine:
    """
    Applies a single mutation operator to one named field within a JSON payload.

    Usage
    -----
    .. code-block:: python

        engine = MutationEngine()
        mutated = engine.apply(
            payload    = {"discount_pct": 10.0, "total_amount": 50.0},
            field_name = "discount_pct",
            operator_id = MutationOperator.BOUNDARY_MAX_OVERFLOW,
            field_spec  = discount_field_spec,   # optional but recommended
        )
        # → {"discount_pct": 101.0, "total_amount": 50.0}
    """

    def apply(
        self,
        payload: dict,
        field_name: str,
        operator_id: int,
        field_spec: FieldSpec | None = None,
    ) -> dict:
        """
        Apply mutation *operator_id* to *field_name* in *payload*.

        Returns a deep copy with the mutation applied.  The original *payload*
        dict is **never** modified.

        Parameters
        ----------
        payload     : Source JSON payload dict.
        field_name  : Key of the field to mutate.
        operator_id : Integer in [0, NUM_OPERATORS).  Values outside this range
                      are wrapped modulo NUM_OPERATORS.
        field_spec  : Optional schema descriptor enabling type-aware boundary
                      values (min/max from OpenAPI spec).
        """
        mutated = copy.deepcopy(payload)
        op = MutationOperator(int(operator_id) % NUM_OPERATORS)

        # ── FIELD_OMISSION is a structural mutation (always safe) ─────────
        if op == MutationOperator.FIELD_OMISSION:
            mutated.pop(field_name, None)
            return mutated

        # ── NO_OP: return verbatim copy ───────────────────────────────────
        if op == MutationOperator.NO_OP:
            return mutated

        # ── Value-level mutations ─────────────────────────────────────────
        current: Any = mutated.get(field_name)
        field_type: str = (
            field_spec.type
            if field_spec is not None
            else self._infer_json_type(current)
        )

        mutated[field_name] = self._dispatch(
            current, op, field_type, field_name, field_spec
        )
        return mutated

    # ── Dispatch ─────────────────────────────────────────────────────────────

    def _dispatch(
        self,
        current: Any,
        op: MutationOperator,
        field_type: str,
        field_name: str,
        field_spec: FieldSpec | None,
    ) -> Any:
        if field_type in ("integer", "number"):
            return self._mutate_numeric(current, op, field_spec)
        if field_type == "string":
            return self._mutate_string(current, op, field_name, field_spec)
        if field_type == "array":
            return self._mutate_array(current, op)
        if field_type == "boolean":
            return self._mutate_boolean(current, op)
        if field_type == "object":
            return self._mutate_object(current, op)
        # Unknown / null type: degenerate mutations
        return None if op in (MutationOperator.TYPE_CONFUSION, MutationOperator.SPECIAL_INJECTION) else current

    # ── Per-type mutators ────────────────────────────────────────────────────

    def _mutate_numeric(
        self,
        current: Any,
        op: MutationOperator,
        field_spec: FieldSpec | None,
    ) -> Any:
        minimum: float = (
            float(field_spec.minimum)
            if field_spec is not None and field_spec.minimum is not None
            else 0.0
        )
        maximum: float = (
            float(field_spec.maximum)
            if field_spec is not None and field_spec.maximum is not None
            else 1_000_000.0
        )

        if op == MutationOperator.BOUNDARY_MIN_OVERFLOW:
            return minimum - 1          # underflow: below declared minimum
        if op == MutationOperator.BOUNDARY_MAX_OVERFLOW:
            return maximum + 1          # overflow: above declared maximum
        if op == MutationOperator.TYPE_CONFUSION:
            return "not_a_number"       # string injected into numeric field
        if op == MutationOperator.SPECIAL_INJECTION:
            return _SPECIAL_NUMERICS[
                _stable_index(str(current), len(_SPECIAL_NUMERICS))
            ]
        return current

    def _mutate_string(
        self,
        current: Any,
        op: MutationOperator,
        field_name: str,
        field_spec: FieldSpec | None,
    ) -> Any:
        if op == MutationOperator.BOUNDARY_MIN_OVERFLOW:
            return ""                   # empty string: minimum boundary
        if op == MutationOperator.BOUNDARY_MAX_OVERFLOW:
            return "A" * 10_000        # extreme length: maximum boundary
        if op == MutationOperator.TYPE_CONFUSION:
            return 12345               # integer injected into string field
        if op == MutationOperator.SPECIAL_INJECTION:
            return _SPECIAL_STRINGS[
                _stable_index(field_name, len(_SPECIAL_STRINGS))
            ]
        return current

    def _mutate_array(self, current: Any, op: MutationOperator) -> Any:
        if op == MutationOperator.BOUNDARY_MIN_OVERFLOW:
            return []                   # empty array
        if op == MutationOperator.BOUNDARY_MAX_OVERFLOW:
            return [None] * 100        # extreme-length array of nulls
        if op == MutationOperator.TYPE_CONFUSION:
            return "not_an_array"      # string injected into array field
        if op == MutationOperator.SPECIAL_INJECTION:
            return [None, {}, [], "\x00"]  # mixed special sentinel values
        return current

    def _mutate_boolean(self, current: Any, op: MutationOperator) -> Any:
        if op == MutationOperator.BOUNDARY_MIN_OVERFLOW:
            return False
        if op == MutationOperator.BOUNDARY_MAX_OVERFLOW:
            return True
        if op == MutationOperator.TYPE_CONFUSION:
            return "true"              # string "true" instead of bool
        if op == MutationOperator.SPECIAL_INJECTION:
            return None                # null injected into boolean field
        return current

    def _mutate_object(self, current: Any, op: MutationOperator) -> Any:
        if op == MutationOperator.BOUNDARY_MIN_OVERFLOW:
            return {}                  # empty object
        if op == MutationOperator.BOUNDARY_MAX_OVERFLOW:
            return {str(i): i for i in range(100)}  # large object
        if op == MutationOperator.TYPE_CONFUSION:
            return [1, 2, 3]           # array injected into object field
        if op == MutationOperator.SPECIAL_INJECTION:
            return {"__proto__": {}, "$where": "1=1", "\x00": None}
        return current

    # ── Type inference ────────────────────────────────────────────────────────

    @staticmethod
    def _infer_json_type(value: Any) -> str:
        """
        Infer the JSON Schema primitive type of *value* at runtime.

        Used when no :class:`~core.parser.FieldSpec` is available.
        """
        if isinstance(value, bool):
            return "boolean"   # must come before int — bool subclasses int
        if isinstance(value, int):
            return "integer"
        if isinstance(value, float):
            return "number"
        if isinstance(value, str):
            return "string"
        if isinstance(value, list):
            return "array"
        if isinstance(value, dict):
            return "object"
        return "null"
