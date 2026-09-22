"""
target/app.py
──────────────────────────────────────────────────────────────────────────────
FastAPI microservice engineered with deliberate, realistic edge-case branches
for RL-guided adversarial fuzzing in Protocol-Sync Phase 1.

╔══════════════════════════════════════════════════════════════════════════════╗
║  INTENTIONAL BUGS — DO NOT FIX                                             ║
║                                                                            ║
║  Route               │ Bug                    │ Exception                  ║
║  ──────────────────  │ ──────────────────────  │ ──────────────────────    ║
║  POST /order/create  │ discount_pct == 100     │ ZeroDivisionError         ║
║                      │ negative total_amount   │ (silent, returns 201)     ║
║  PUT  /users/.../    │ start_date >= end_date  │ ValueError                ║
║       profile        │ short email domain      │ IndexError                ║
║                      │ malformed date string   │ ValueError                ║
║  POST /transfer      │ self-transfer           │ AssertionError            ║
║                      │ non-numeric amount      │ decimal.InvalidOperation  ║
║                      │ astronomical amount     │ ArithmeticError           ║
╚══════════════════════════════════════════════════════════════════════════════╝

Design constraint: NO blanket try/except in route handlers.  Exceptions must
bubble up through Starlette's ServerErrorMiddleware and appear as HTTP 500
responses so the harness can detect them.
"""
from __future__ import annotations

from datetime import date
from decimal import Decimal, InvalidOperation
from typing import Any

from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI(
    title="Protocol-Sync Target Service",
    version="0.1.0",
    description=(
        "Deliberately buggy microservice for adversarial fuzzing in Protocol-Sync. "
        "Three routes with engineered edge-case branches that surface as HTTP 500."
    ),
)


# ─────────────────────────────────────────────────────────────────────────────
# Route 1 — POST /order/create
# ─────────────────────────────────────────────────────────────────────────────

class OrderItem(BaseModel):
    """A single line item in an order."""
    product_id: str
    quantity: int
    unit_price: float


class CreateOrderRequest(BaseModel):
    """
    Order creation payload.

    Edge cases
    ----------
    * ``discount_pct == 100``  → ZeroDivisionError in multiplier calculation.
    * ``total_amount < 0``     → slips through Pydantic (no ge constraint) and
                                 produces a negative ``final_amount``.
    """
    items: list[OrderItem]
    discount_pct: float = 0.0   # Intended range: 0–100; 100 breaks arithmetic
    total_amount: float          # Negative values are NOT rejected — deliberate
    customer_note: str | None = None


class CreateOrderResponse(BaseModel):
    order_id: str
    final_amount: float
    applied_discount: float


@app.post("/order/create", response_model=CreateOrderResponse, status_code=201)
def create_order(req: CreateOrderRequest) -> CreateOrderResponse:
    """
    Create an order and return the final price after discount.

    DELIBERATE BUG #1 (ZeroDivisionError)
    ──────────────────────────────────────
    The formula below computes a per-unit ``multiplier`` by dividing the subtotal
    by ``(100 - discount_pct)``.  When ``discount_pct == 100``, the divisor is 0,
    causing an unhandled ``ZeroDivisionError`` → HTTP 500.

    DELIBERATE BUG #2 (silent negative amount)
    ───────────────────────────────────────────
    Negative ``total_amount`` is arithmetically valid here and produces a
    negative ``applied_discount`` without any validation error → HTTP 201 with
    semantically wrong data (a bug the harness should detect via response inspection).
    """
    subtotal: float = sum(item.quantity * item.unit_price for item in req.items)

    # BUG #1: ZeroDivisionError when discount_pct == 100
    multiplier: float = subtotal / (100.0 - req.discount_pct)

    final: float = multiplier * (100.0 - req.discount_pct) - (
        req.discount_pct / 100.0 * req.total_amount
    )

    # BUG #2: abs() call means this is always non-negative, but final can still
    #         go negative; no guard present.
    applied: float = req.discount_pct / 100.0 * abs(req.total_amount)

    # Order ID derived from items hash — deterministic for the same payload
    order_id = f"ORD-{abs(hash(str([(i.product_id, i.quantity) for i in req.items]))) % 100_000:05d}"

    return CreateOrderResponse(
        order_id=order_id,
        final_amount=final,
        applied_discount=applied,
    )


# ─────────────────────────────────────────────────────────────────────────────
# Route 2 — PUT /users/{user_id}/profile
# ─────────────────────────────────────────────────────────────────────────────

class ProfileUpdateRequest(BaseModel):
    """
    Profile update payload.

    Edge cases
    ----------
    * ``start_date >= end_date``              → explicit ``ValueError``.
    * Malformed ISO date string               → ``ValueError`` from ``date.fromisoformat``.
    * ``username`` contains ``@`` with a very
      short domain (``len(domain) < 2``)     → ``IndexError`` on ``domain[1]``.
    * ``username`` starts with a digit AND
      has fewer than 11 characters            → ``IndexError`` on ``username[10]``.
    """
    username: str
    start_date: str         # Expected: ISO 8601 YYYY-MM-DD
    end_date: str           # Must be strictly after start_date
    metadata: dict[str, Any] | None = None


class ProfileUpdateResponse(BaseModel):
    user_id: str
    username: str
    duration_days: int


@app.put("/users/{user_id}/profile", response_model=ProfileUpdateResponse)
def update_profile(user_id: str, req: ProfileUpdateRequest) -> ProfileUpdateResponse:
    """
    Update a user's profile date range and username.

    DELIBERATE BUG #3 (ValueError — malformed date)
    ─────────────────────────────────────────────────
    ``date.fromisoformat`` raises ``ValueError`` for any string that is not a
    valid ISO 8601 date.  No try/except → HTTP 500.

    DELIBERATE BUG #4 (ValueError — date ordering invariant)
    ──────────────────────────────────────────────────────────
    ``end_date`` must be strictly after ``start_date``.  If not, we raise
    ``ValueError`` explicitly (no guard → HTTP 500).

    DELIBERATE BUG #5 (IndexError — email domain boundary)
    ────────────────────────────────────────────────────────
    If ``username`` contains ``@``, the code extracts the domain and indexes
    ``domain[1]`` without bounds-checking.  A one-character domain (e.g.
    ``user@x``) raises ``IndexError`` → HTTP 500.

    DELIBERATE BUG #6 (IndexError — numeric username)
    ──────────────────────────────────────────────────
    Usernames that start with a digit trigger an alternative path that accesses
    ``username[10]`` unconditionally; usernames shorter than 11 characters
    raise ``IndexError`` → HTTP 500.
    """
    # BUG #3: ValueError if date string is not ISO 8601 (no try/except)
    start: date = date.fromisoformat(req.start_date)
    end: date = date.fromisoformat(req.end_date)

    # BUG #4: explicit ValueError when ordering invariant is violated
    delta = end - start
    if delta.days <= 0:
        raise ValueError(
            f"end_date ({req.end_date!r}) must be strictly after "
            f"start_date ({req.start_date!r}); got delta={delta.days} days."
        )

    if "@" in req.username:
        # BUG #5: IndexError when domain has fewer than 2 characters
        domain = req.username.split("@", maxsplit=1)[1]
        _domain_initial = domain[1]   # raises IndexError if len(domain) < 2
    else:
        if req.username and req.username[0].isdigit():
            # BUG #6: IndexError when numeric username is shorter than 11 chars
            _tag = req.username[10]   # raises IndexError if len(username) < 11

    return ProfileUpdateResponse(
        user_id=user_id,
        username=req.username,
        duration_days=delta.days,
    )


# ─────────────────────────────────────────────────────────────────────────────
# Route 3 — POST /transfer
# ─────────────────────────────────────────────────────────────────────────────

class TransferRequest(BaseModel):
    """
    Fund transfer payload.

    Edge cases
    ----------
    * ``from_account == to_account``          → ``AssertionError``.
    * Non-numeric ``amount`` string           → ``decimal.InvalidOperation``.
    * ``"inf"`` / ``"nan"`` ``amount``        → ``decimal.InvalidOperation``.
    * ``amount > 10^38``                      → ``ArithmeticError``.
    """
    from_account: str
    to_account: str
    amount: str          # Decimal string for precision; invalid strings raise
    currency: str = "USD"
    memo: str | None = None


class TransferResponse(BaseModel):
    transaction_id: str
    from_account: str
    to_account: str
    amount: str
    status: str


# Simulated in-memory ledger — shared across requests (intentional mutable state)
_LEDGER: dict[str, Decimal] = {
    "ACC-001": Decimal("10000.00"),
    "ACC-002": Decimal("5000.00"),
    "ACC-003": Decimal("0.00"),
}

_MAX_TRANSFER: Decimal = Decimal("1" + "0" * 38)   # 10^38 — near Decimal practical limit


@app.post("/transfer", response_model=TransferResponse)
def transfer(req: TransferRequest) -> TransferResponse:
    """
    Transfer funds between two accounts.

    DELIBERATE BUG #7 (AssertionError — self-transfer)
    ───────────────────────────────────────────────────
    ``from_account == to_account`` fails the assertion → HTTP 500.

    DELIBERATE BUG #8 (InvalidOperation — malformed amount)
    ─────────────────────────────────────────────────────────
    ``Decimal(req.amount)`` raises ``decimal.InvalidOperation`` for non-numeric
    strings (``"abc"``, ``"inf"``, ``"nan"``) and very long digit strings that
    exceed Decimal's internal capacity.

    DELIBERATE BUG #9 (ArithmeticError — overflow guard)
    ──────────────────────────────────────────────────────
    Amounts exceeding ``_MAX_TRANSFER`` (10^38) raise ``ArithmeticError``
    rather than being silently truncated.

    DELIBERATE BUG #10 (silent negative balance)
    ─────────────────────────────────────────────
    No balance sufficiency check: any positive amount can be transferred even
    if the source account lacks funds, silently creating negative balances.
    """
    # BUG #7: AssertionError on self-transfer (not caught → HTTP 500)
    assert req.from_account != req.to_account, (
        f"Self-transfer not permitted: from_account and to_account are both "
        f"{req.from_account!r}."
    )

    # BUG #8: InvalidOperation for non-numeric or special-value strings
    amount: Decimal = Decimal(req.amount)

    # BUG #9: Explicit arithmetic guard — unhandled → HTTP 500
    if amount > _MAX_TRANSFER:
        raise ArithmeticError(
            f"Transfer amount {amount} exceeds system maximum ({_MAX_TRANSFER})."
        )

    # BUG #10: No balance check — negative balances created silently
    _LEDGER[req.from_account] = (
        _LEDGER.get(req.from_account, Decimal("0")) - amount
    )
    _LEDGER[req.to_account] = (
        _LEDGER.get(req.to_account, Decimal("0")) + amount
    )

    txn_id = (
        f"TXN-{abs(hash(req.from_account + req.to_account + req.amount)) % 10**9:09d}"
    )

    return TransferResponse(
        transaction_id=txn_id,
        from_account=req.from_account,
        to_account=req.to_account,
        amount=str(amount),
        status="COMPLETED",
    )
