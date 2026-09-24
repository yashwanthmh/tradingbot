"""Sizing arithmetic: the minimum of every opinion, and the floor underneath it.

The engine's sizing step is short and every line of it is a policy. These tests
pin the two that interact badly if either is written the obvious way:

* the regime factor scales a position *down*, and
* the floor notional refuses a position that is too small to be worth placing.

Compose them naively and a position sized at exactly the floor can never be
opened while the regime is reduced — and since the regime gate's cold-start
reading is 0.5, "reduced" is the state of every account without ten months of
reference history. The same collision arrives by a second route: `max_quantity`
rounds *down* to the quantum, so a cap sitting exactly on the floor yields a
notional a fraction under it on any instrument whose price does not divide the
floor.

Both are exercised here through the per-position cap, by choosing an equity that
puts that cap exactly on the floor. That keeps these cases independent of which
*other* rule happens to bind in a given release.
"""

from __future__ import annotations

from datetime import UTC, datetime
from decimal import Decimal

from tb.broker.port import OrderPurpose, Side
from tb.config.loader import load_hard_limits
from tb.risk.engine import RiskEngine
from tb.risk.state import AccountState, OrderRequest, RiskContext
from tb.strategy.base import Action

LIMITS = load_hard_limits(None).limits
AS_OF = datetime(2026, 4, 1, 15, 30, tzinfo=UTC)
TICKER = "AAPL_US_EQ"
UID = "isin:US0378331005"

FLOOR = LIMITS.capital.floor_notional_ccy
# The equity at which `per_position_pct` of it is exactly the floor, so the
# per-position cap and the minimum ticket coincide. Derived rather than written
# down, so the tests follow the shipped config instead of a stale constant.
EQUITY_AT_FLOOR = FLOOR * Decimal(100) / Decimal(str(LIMITS.capital.per_position_pct))


def _context(
    *,
    regime_factor: Decimal | None,
    equity: Decimal,
    price: Decimal = Decimal("150.00"),
    # The allocator's per-position size, deliberately far above every other cap
    # so the *per-position cap* stays the binding one. These tests are about
    # which bound wins and what happens at the floor, and an allocation that
    # also bound would make it ambiguous which rule produced the size.
    notional: Decimal = Decimal("100000.00"),
) -> RiskContext:
    return RiskContext(
        as_of=AS_OF,
        limits=LIMITS,
        request=OrderRequest(
            t212_ticker=TICKER,
            instrument_uid=UID,
            side=Side.BUY,
            purpose=OrderPurpose.ENTRY,
            action=Action.ENTER,
            reference_price=price,
            expected_edge_bps=Decimal("450"),
        ),
        account=AccountState(
            equity_ccy=equity,
            free_cash_ccy=equity,
            deployed_ccy=Decimal(0),
            n_open_positions=0,
            currency="GBP",
            day_pnl_pct=0.0,
            rolling_5d_pnl_pct=0.0,
            drawdown_from_peak_pct=0.0,
        ),
        may_enter=True,
        regime_exposure_factor=regime_factor,
        regime_state="risk_on" if regime_factor == Decimal(1) else "unavailable",
        bar_age_seconds=5.0,
        bar_period_seconds=86400,
        minutes_since_open=60,
        minutes_until_close=120,
        strategy_notional_ccy=notional,
        extra={"isin": "US0378331005", "instrument_currency": "USD"},
    )


def test_the_regime_factor_shrinks_a_position_that_has_room_to_shrink() -> None:
    """The ordinary case, unchanged: a position well above the floor is halved."""
    roomy = Decimal("10000.00")
    full = RiskEngine().evaluate(
        _context(regime_factor=Decimal(1), equity=roomy), run_id="run_sizing"
    )
    halved = RiskEngine().evaluate(
        _context(regime_factor=Decimal("0.5"), equity=roomy), run_id="run_sizing"
    )

    assert full.approved and halved.approved, (full.refusal_summary, halved.refusal_summary)
    assert full.approved_quantity is not None and halved.approved_quantity is not None
    assert halved.approved_quantity == full.approved_quantity / 2
    assert any("regime factor" in note for note in halved.sizing_notes)


def test_a_floor_size_position_survives_a_reduced_regime() -> None:
    """**The interaction this module exists for.**

    When the binding cap is exactly `floor_notional_ccy`, multiplying by the
    cold-start factor of 0.5 puts it under the floor and the floor rule would
    refuse it — so a floor-size position could never be opened on a fresh
    install. The factor is therefore bounded below by the minimum ticket: below
    the floor an order is unplaceable rather than small.
    """
    reduced = RiskEngine().evaluate(
        _context(regime_factor=Decimal("0.5"), equity=EQUITY_AT_FLOOR), run_id="run_sizing"
    )

    assert reduced.approved, reduced.refusal_summary
    assert reduced.approved_notional_ccy is not None
    assert reduced.approved_notional_ccy >= FLOOR
    assert any("minimum ticket" in note for note in reduced.sizing_notes), reduced.sizing_notes


def test_the_clamp_never_enlarges_an_order_past_the_cap_that_bound_it() -> None:
    """The clamp is a bound on the *scaling*, not a licence to round up.

    An account whose per-position cap is genuinely below the minimum ticket is
    refused, and stays refused with the regime reduced. Rescuing it to the floor
    would hand out more capital than the cap allowed, which would make every cap
    above advisory.
    """
    small = EQUITY_AT_FLOOR / Decimal(3)

    for factor in (Decimal(1), Decimal("0.5")):
        evaluation = RiskEngine().evaluate(
            _context(regime_factor=factor, equity=small), run_id="run_sizing"
        )
        assert not evaluation.approved, f"a sub-floor cap was placed at factor {factor}"
        assert "floor_notional" in evaluation.refusal_summary


def test_a_zero_regime_factor_still_blocks_outright() -> None:
    """The clamp is about scaling, not about the gate's refusal.

    `exposure_factor == 0` is the regime rule blocking, not a sizing opinion —
    and a clamp that read it as "size to the minimum ticket" would convert a
    refusal into an order.
    """
    evaluation = RiskEngine().evaluate(
        _context(regime_factor=Decimal(0), equity=EQUITY_AT_FLOOR), run_id="run_sizing"
    )
    assert not evaluation.approved
    assert "regime" in evaluation.refusal_summary


def test_an_unpriced_regime_blocks_rather_than_sizing() -> None:
    """A missing factor is a programming error, not full exposure."""
    evaluation = RiskEngine().evaluate(
        _context(regime_factor=None, equity=EQUITY_AT_FLOOR), run_id="run_sizing"
    )
    assert not evaluation.approved
    assert "regime" in evaluation.refusal_summary


def test_quantisation_alone_cannot_refuse_a_floor_size_entry() -> None:
    """The same shortfall by a different route, with the regime at full exposure.

    `max_quantity` rounds down to the quantum, which is right for a cap: rounding
    up turns a 1% cap into 1.4% on a high-priced instrument. But a cap sitting
    exactly on the floor then yields 14.99999998 at a £7 share, which the floor
    rule refuses — so a floor-size position would be unopenable on every
    instrument whose price does not divide the floor, in any regime.
    """
    evaluation = RiskEngine().evaluate(
        _context(regime_factor=Decimal(1), equity=EQUITY_AT_FLOOR, price=Decimal("7.00")),
        run_id="run_sizing",
    )
    assert evaluation.approved, evaluation.refusal_summary
    assert evaluation.approved_notional_ccy is not None
    assert evaluation.approved_notional_ccy >= FLOOR


def test_both_causes_together_still_reach_the_minimum_ticket() -> None:
    """A reduced regime and an awkward price at once, which is the live case."""
    evaluation = RiskEngine().evaluate(
        _context(regime_factor=Decimal("0.5"), equity=EQUITY_AT_FLOOR, price=Decimal("7.00")),
        run_id="run_sizing",
    )
    assert evaluation.approved, evaluation.refusal_summary
    assert evaluation.approved_notional_ccy is not None
    assert evaluation.approved_notional_ccy >= FLOOR
