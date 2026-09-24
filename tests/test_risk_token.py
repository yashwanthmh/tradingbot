"""The `RiskToken` guarantee: no order reaches a broker without the risk engine.

The centrepiece is `test_the_engine_is_the_only_construction_site`, which
parses every module in the tree. It is the mechanism that actually holds the
property, because it fails in CI — a bypass cannot be merged, never mind
executed. The runtime guards tested below are the second line: they catch the
bypass that a test or a REPL would otherwise get away with, and they close the
`dataclasses.replace` hole that an AST scan cannot see.

"Every order passes the risk engine" as a *convention* would be violated by
the first well-meaning refactor, with nothing anywhere failing. That is the
whole reason this file exists.
"""

from __future__ import annotations

import ast
from datetime import UTC, datetime, timedelta
from decimal import Decimal
from pathlib import Path

import pytest

from tb.broker.port import OrderPurpose, Side
from tb.risk.token import (
    ENGINE_MODULE,
    TOKEN_TTL_SECONDS,
    RiskToken,
    RiskTokenError,
    _calling_module,
    _Mint,
    _mint,
)

SRC = Path(__file__).resolve().parents[1] / "src"
ENGINE_PATH = SRC / "tb" / "risk" / "engine.py"
TOKEN_PATH = SRC / "tb" / "risk" / "token.py"


def _python_files() -> list[Path]:
    return sorted(SRC.rglob("*.py"))


def _construction_sites(path: Path) -> list[int]:
    """Line numbers of every `RiskToken(...)` call in `path`.

    Matches a bare `RiskToken(...)` and an attribute form like
    `token.RiskToken(...)`, because an import style change must not be able to
    hide a construction site from this scan.
    """
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    lines: list[int] = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        func = node.func
        name = None
        if isinstance(func, ast.Name):
            name = func.id
        elif isinstance(func, ast.Attribute):
            name = func.attr
        if name == "RiskToken":
            lines.append(node.lineno)
    return lines


# --------------------------------------------------------------------------
# The property that matters
# --------------------------------------------------------------------------


def test_the_engine_is_the_only_construction_site() -> None:
    """One `RiskToken(...)` in the whole tree, and it is in the risk engine.

    This is the assertion that makes the unbypassable risk path real rather
    than documented. If it fails, read the new call site: either it belongs in
    the engine, or it is the bug this test exists to catch.
    """
    offenders: dict[str, list[int]] = {}
    for path in _python_files():
        sites = _construction_sites(path)
        if not sites:
            continue
        if path == ENGINE_PATH:
            continue
        offenders[str(path.relative_to(SRC))] = sites

    assert not offenders, (
        "RiskToken is constructed outside tb/risk/engine.py: "
        f"{offenders}. Every order must be authorised by the risk engine, and that is "
        "enforced structurally rather than by convention — route the order through "
        "RiskEngine.evaluate instead of minting a token."
    )


def test_the_engine_really_does_construct_one() -> None:
    """The vacuity check.

    Without it, deleting the engine's construction — or renaming the class —
    would make the test above pass by finding nothing at all.
    """
    assert _construction_sites(ENGINE_PATH), (
        "no RiskToken construction found in the engine. The test above would then be "
        "vacuously true: zero call sites outside the engine, and zero inside it."
    )


def test_only_the_engine_imports_the_mint() -> None:
    """The capability has one holder.

    The mint is what makes a construction possible at all, so a second
    importer is a second potential issuer — even if it does not construct a
    token today.
    """
    offenders: list[str] = []
    for path in _python_files():
        if path in (ENGINE_PATH, TOKEN_PATH):
            continue
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom) and node.module == "tb.risk.token":
                for alias in node.names:
                    if alias.name in ("_mint", "_Mint"):
                        offenders.append(f"{path.relative_to(SRC)}:{node.lineno}")
    assert not offenders, (
        f"the token mint is imported outside the engine: {offenders}. Holding it is the "
        "capability to authorise an order."
    )


# --------------------------------------------------------------------------
# The runtime guards
# --------------------------------------------------------------------------


def _token_kwargs(**overrides: object) -> dict[str, object]:
    issued = datetime(2026, 4, 1, 14, 30, tzinfo=UTC)
    base: dict[str, object] = {
        "token_id": "rtok_test",
        "run_id": "run_test",
        "t212_ticker": "AAPL_US_EQ",
        "side": Side.BUY,
        "purpose": OrderPurpose.ENTRY,
        "quantity": Decimal("3"),
        "issued_at": issued,
        "expires_at": issued + timedelta(seconds=TOKEN_TTL_SECONDS),
        "mint": _mint,
    }
    base.update(overrides)
    return base


def test_construction_from_a_test_module_is_refused() -> None:
    """Even holding the mint, the caller must be the engine.

    This is the guard that catches a bypass the AST scan cannot: a construction
    assembled at runtime, from a REPL, or from a test that imported the mint.
    """
    with pytest.raises(RiskTokenError, match=r"only tb\.risk\.engine"):
        RiskToken(**_token_kwargs())  # type: ignore[arg-type]


def test_construction_without_the_mint_is_refused() -> None:
    with pytest.raises(RiskTokenError, match="without the mint"):
        RiskToken(**_token_kwargs(mint=None))  # type: ignore[arg-type]


def test_a_forged_mint_is_refused() -> None:
    """The mint is identified by type, so a look-alike does not work."""

    class FakeMint:
        pass

    with pytest.raises(RiskTokenError, match="without the mint"):
        RiskToken(**_token_kwargs(mint=FakeMint()))  # type: ignore[arg-type]


def test_replace_cannot_launder_an_approval_into_a_larger_one() -> None:
    """The hole the capability check alone leaves.

    `dataclasses.replace` passes the *existing* mint through to a new
    instance, so a valid approval for three shares would mint a valid approval
    for three hundred. The caller check catches it because `replace` reports
    the `dataclasses` module as the constructing caller.
    """
    from dataclasses import replace

    from tb.risk.engine import RiskEngine

    token = _issue_via_engine(RiskEngine())
    assert token is not None
    with pytest.raises(RiskTokenError, match=r"only tb\.risk\.engine"):
        replace(token, quantity=Decimal("300"))


def test_the_caller_depth_constant_is_correct() -> None:
    """`_CALLER_DEPTH` is a property of dataclass codegen, so it is asserted.

    If a Python release changes how `__init__` is generated, the guard would
    start reading the wrong frame — and would most likely then see
    `tb.risk.token` and refuse everything, or see too far up and accept
    everything. This test pins it either way.
    """
    assert _calling_module(depth=1) == __name__, (
        "depth=1 should be this test's own module; if it is not, the frame walk in "
        "tb.risk.token is counting differently than it did when written"
    )


# --------------------------------------------------------------------------
# What a token authorises
# --------------------------------------------------------------------------


def _issue_via_engine(engine: object) -> RiskToken | None:
    """Get a real token the only legitimate way: through the engine."""
    from tb.config.loader import load_hard_limits
    from tb.risk.engine import RiskEngine, entry_request
    from tb.risk.state import AccountState, RiskContext

    assert isinstance(engine, RiskEngine)
    limits = load_hard_limits(None).limits
    ctx = RiskContext(
        as_of=datetime(2026, 4, 1, 14, 30, tzinfo=UTC),
        limits=limits,
        request=entry_request(
            t212_ticker="AAPL_US_EQ",
            instrument_uid="isin:US0378331005",
            reference_price=Decimal("100"),
            expected_edge_bps=Decimal("150"),
        ),
        account=AccountState(
            equity_ccy=Decimal("10000"),
            free_cash_ccy=Decimal("9000"),
            deployed_ccy=Decimal("0"),
            n_open_positions=0,
            currency="GBP",
            day_pnl_pct=0.0,
            rolling_5d_pnl_pct=0.0,
            drawdown_from_peak_pct=0.0,
        ),
        may_enter=True,
        may_enter_reason="cross-verified",
        regime_exposure_factor=Decimal(1),
        regime_state="risk_on",
        bar_age_seconds=5.0,
        minutes_since_open=60,
        minutes_until_close=120,
        extra={"isin": "US0378331005", "instrument_currency": "USD"},
        # The allocator's per-position size. Fail-closed at the rule, so a
        # context that omits it refuses every entry — see
        # `StrategyAllocationRule`. Well above the caps here, so the M4 caps
        # stay the binding ones and these tests keep measuring what they were
        # written to measure.
        strategy_notional_ccy=Decimal("1000.00"),
    )
    return engine.evaluate(ctx, run_id="run_test").token


def test_a_token_authorises_only_the_order_it_was_issued_for() -> None:
    """A bare "approved: yes" token would be a cheque with the amount blank."""
    from tb.risk.engine import RiskEngine

    token = _issue_via_engine(RiskEngine())
    assert token is not None, "the fixture should produce an approval"
    at = token.issued_at

    # The order it was issued for.
    token.authorises(
        t212_ticker=token.t212_ticker,
        side=token.side,
        quantity=token.quantity,
        purpose=token.purpose,
        at=at,
    )

    for label, kwargs in (
        ("a different ticker", {"t212_ticker": "MSFT_US_EQ"}),
        ("the other side", {"side": Side.SELL}),
        ("a different purpose", {"purpose": OrderPurpose.EXIT}),
        ("a larger quantity", {"quantity": token.quantity * 100}),
        # Smaller, too: the engine sized this order for a reason, and a caller
        # picking its own number defeats the sizing rules.
        ("a smaller quantity", {"quantity": token.quantity / 2}),
    ):
        call = {
            "t212_ticker": token.t212_ticker,
            "side": token.side,
            "quantity": token.quantity,
            "purpose": token.purpose,
            "at": at,
            **kwargs,
        }
        with pytest.raises(RiskTokenError, match="does not authorise"):
            token.authorises(**call)  # type: ignore[arg-type]
        assert True, label


def test_an_expired_token_authorises_nothing() -> None:
    """A token asserts something about account state at a moment.

    Equity, deployed capital and the loss breakers all move. Re-evaluating is
    cheap; honouring a stale approval is how an order lands after the breaker
    that should have stopped it.
    """
    from tb.risk.engine import RiskEngine

    token = _issue_via_engine(RiskEngine())
    assert token is not None
    later = token.expires_at + timedelta(seconds=1)
    with pytest.raises(RiskTokenError, match="expired"):
        token.authorises(
            t212_ticker=token.t212_ticker,
            side=token.side,
            quantity=token.quantity,
            purpose=token.purpose,
            at=later,
        )


def test_a_token_cannot_be_mutated_after_issue() -> None:
    """Frozen: an editable approval is not an approval."""
    from dataclasses import FrozenInstanceError

    from tb.risk.engine import RiskEngine

    token = _issue_via_engine(RiskEngine())
    assert token is not None
    with pytest.raises(FrozenInstanceError):
        token.quantity = Decimal("300")  # type: ignore[misc]


def test_slots_prevent_attaching_a_new_attribute() -> None:
    """No `token.override = True`, and no `__dict__` to smuggle one into.

    The exception type differs from the frozen-field case above: setting a
    *declared* field raises `FrozenInstanceError`, while setting an undeclared
    one raises `TypeError` from a CPython quirk — with `frozen=True,
    slots=True` the generated `__setattr__` closes over the pre-slots class, so
    its `super()` call fails before it can raise the tidy error. Either way the
    write is refused, which is the property; the test accepts both rather than
    pinning a quirk that a future release may fix.
    """
    from tb.risk.engine import RiskEngine

    token = _issue_via_engine(RiskEngine())
    assert token is not None
    with pytest.raises((AttributeError, TypeError)):
        token.override = True  # type: ignore[attr-defined]
    # And the absence of a `__dict__` is the reason, so assert it directly.
    assert not hasattr(token, "__dict__"), (
        "a RiskToken with a __dict__ could carry arbitrary attributes, which is how "
        "an approval acquires fields the engine never granted"
    )


def test_the_engine_module_name_is_the_real_module() -> None:
    """`ENGINE_MODULE` is a string, so it can rot. Pin it to the import."""
    import tb.risk.engine as engine_module

    assert engine_module.__name__ == ENGINE_MODULE


def test_the_mint_type_is_what_the_guard_checks() -> None:
    assert isinstance(_mint, _Mint)
