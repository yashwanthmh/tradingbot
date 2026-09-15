"""The cost model.

These tests are where the project's central claim gets checked: that on this
venue the fee schedule, not the signal, decides whether a strategy exists. If
`test_the_venue_arithmetic_that_shapes_the_whole_design` ever starts passing
with a much smaller number, the cost model has a bug and every gate above it
is reading a flattering number.
"""

from __future__ import annotations

from decimal import Decimal
from pathlib import Path

import pytest

from tb.backtest.costs import (
    FRENCH_FTT_RATE,
    IRISH_STAMP_DUTY_RATE,
    UK_STAMP_DUTY_RATE,
    CostError,
    CostModel,
    Jurisdiction,
    jurisdiction_from_isin,
)
from tb.config.loader import load_hard_limits
from tb.data.fx import FX_FEE_RATE

NOTIONAL = Decimal("1000.00")


@pytest.fixture
def model(limits_file: Path) -> CostModel:
    return CostModel(load_hard_limits(limits_file).limits)


# --------------------------------------------------------------------------
# Jurisdiction
# --------------------------------------------------------------------------


def test_jurisdiction_comes_from_the_isin_not_the_venue() -> None:
    """A `.L` suffix means listed in London, not UK-incorporated.

    The difference is 50bps on every entry, and an Irish company traded in
    London attracts Irish duty at 100bps rather than UK duty at 50.
    """
    assert jurisdiction_from_isin("GB00B03MLX29") is Jurisdiction.UK
    assert jurisdiction_from_isin("IE00B4BNMY34") is Jurisdiction.IRELAND
    assert jurisdiction_from_isin("US0378331005") is Jurisdiction.US
    assert jurisdiction_from_isin("FR0000120271") is Jurisdiction.FRANCE
    assert jurisdiction_from_isin("DE0007164600") is Jurisdiction.OTHER


@pytest.mark.parametrize("isin", [None, "", "U", "12345", "  "])
def test_an_unusable_isin_is_unknown_not_cheap(isin: str | None) -> None:
    """Never default to the low-tax case.

    `OTHER` charges nothing. Silently assuming it for an instrument whose
    issuer we cannot identify understates cost by up to 100bps on every entry.
    """
    assert jurisdiction_from_isin(isin) is Jurisdiction.UNKNOWN


def test_an_unknown_jurisdiction_refuses_to_be_costed(model: CostModel) -> None:
    with pytest.raises(CostError, match="jurisdiction is unknown"):
        model.leg(
            notional_ccy=NOTIONAL,
            side="buy",
            instrument_currency="GBP",
            jurisdiction=Jurisdiction.UNKNOWN,
        )


def test_the_published_tax_rates_are_what_the_jurisdictions_charge() -> None:
    assert Jurisdiction.UK.buy_tax_rate == UK_STAMP_DUTY_RATE == Decimal("0.005")
    assert Jurisdiction.IRELAND.buy_tax_rate == IRISH_STAMP_DUTY_RATE == Decimal("0.010")
    assert Jurisdiction.FRANCE.buy_tax_rate == FRENCH_FTT_RATE == Decimal("0.003")
    assert Jurisdiction.US.buy_tax_rate == Decimal(0)
    assert Jurisdiction.OTHER.buy_tax_rate == Decimal(0)


# --------------------------------------------------------------------------
# One leg
# --------------------------------------------------------------------------


def test_stamp_duty_is_charged_on_the_buy_and_never_on_the_sale(model: CostModel) -> None:
    """Charging it symmetrically would overstate a UK round trip by 50bps.

    Which sounds conservative and is not: it would push the search loop away
    from the venue that is *cheaper* for a GBP account, towards paying FX.
    """
    buy = model.leg(
        notional_ccy=NOTIONAL,
        side="buy",
        instrument_currency="GBP",
        jurisdiction=Jurisdiction.UK,
    )
    sell = model.leg(
        notional_ccy=NOTIONAL,
        side="sell",
        instrument_currency="GBP",
        jurisdiction=Jurisdiction.UK,
    )
    assert buy.transaction_tax_ccy == NOTIONAL * UK_STAMP_DUTY_RATE
    assert sell.transaction_tax_ccy == Decimal(0)


def test_fx_is_charged_only_when_the_currency_is_crossed(model: CostModel) -> None:
    assert model.account_currency == "GBP"
    domestic = model.leg(
        notional_ccy=NOTIONAL,
        side="buy",
        instrument_currency="GBP",
        jurisdiction=Jurisdiction.UK,
    )
    foreign = model.leg(
        notional_ccy=NOTIONAL,
        side="buy",
        instrument_currency="USD",
        jurisdiction=Jurisdiction.US,
    )
    assert not domestic.crossed_currency
    assert domestic.fx_fee_ccy == Decimal(0)
    assert foreign.crossed_currency
    assert foreign.fx_fee_ccy == NOTIONAL * FX_FEE_RATE


def test_the_fx_rate_is_the_one_the_data_layer_charges(model: CostModel) -> None:
    """Reused rather than restated.

    A conversion charged once in the cost model and twice in the FX store is
    a systematically optimistic backtest, and two copies of 0.0015 is how that
    happens.
    """
    leg = model.leg(
        notional_ccy=Decimal("10000.00"),
        side="buy",
        instrument_currency="USD",
        jurisdiction=Jurisdiction.US,
    )
    assert leg.fx_fee_ccy == Decimal("10000.00") * FX_FEE_RATE == Decimal("15.00")


def test_a_leg_itemises_rather_than_totalling(model: CostModel) -> None:
    """ "Dies to stamp duty" and "dies to the spread" are different problems."""
    leg = model.leg(
        notional_ccy=NOTIONAL,
        side="buy",
        instrument_currency="USD",
        jurisdiction=Jurisdiction.US,
    )
    items = leg.itemised()
    assert set(items) == {
        "fx_fee",
        "transaction_tax",
        "half_spread",
        "slippage",
        "total",
        "jurisdiction",
    }
    # Strings, so no float can enter a stored payload and change a hash.
    assert all(isinstance(v, str) for v in items.values())
    assert leg.total_ccy == leg.fx_fee_ccy + leg.half_spread_ccy + leg.slippage_ccy


@pytest.mark.parametrize("bad", [Decimal(0), Decimal("-1")])
def test_a_non_positive_notional_is_refused(model: CostModel, bad: Decimal) -> None:
    with pytest.raises(CostError, match="notional must be positive"):
        model.leg(
            notional_ccy=bad,
            side="buy",
            instrument_currency="GBP",
            jurisdiction=Jurisdiction.UK,
        )


def test_an_unknown_side_is_refused(model: CostModel) -> None:
    with pytest.raises(CostError, match="side must be"):
        model.leg(
            notional_ccy=NOTIONAL,
            side="short",
            instrument_currency="GBP",
            jurisdiction=Jurisdiction.UK,
        )


# --------------------------------------------------------------------------
# The venue arithmetic
# --------------------------------------------------------------------------


def test_the_venue_arithmetic_that_shapes_the_whole_design(model: CostModel) -> None:
    """The finding the entire system is built around, as a number.

    A GBP account buying a US name pays FX twice plus two half-spreads plus two
    slippages. Against 5-20bps of gross minute-bar edge in liquid names, this
    is why the design targets minute *features* with hour-to-day position
    changes rather than minute-by-minute trading.
    """
    us = model.round_trip(
        notional_ccy=NOTIONAL,
        instrument_currency="USD",
        jurisdiction=Jurisdiction.US,
    )
    # 2 x 15bps FX + 2 x 2bps spread + 2 x 3bps slippage = 40bps.
    assert us.total_bps == pytest.approx(Decimal("40"), abs=Decimal("0.5"))
    assert us.total_bps > Decimal("20"), (
        "a US round trip has become cheaper than the top of the gross minute-bar "
        "edge band; re-read the cost model before believing any backtest"
    )

    uk = model.round_trip(
        notional_ccy=NOTIONAL,
        instrument_currency="GBP",
        jurisdiction=Jurisdiction.UK,
    )
    # 50bps stamp duty on the buy only + 2 x 2bps spread + 2 x 3bps slippage.
    assert uk.total_bps == pytest.approx(Decimal("60"), abs=Decimal("0.5"))
    assert uk.total_bps > us.total_bps, (
        "UK stamp duty should make a UK round trip dearer than a US one even "
        "after the FX a GBP account pays to reach the US"
    )


def test_the_exit_leg_is_costed_on_what_is_actually_sold(model: CostModel) -> None:
    """Assuming the exit notional equals the entry understates cost on a winner."""
    flat = model.round_trip(
        notional_ccy=NOTIONAL,
        instrument_currency="USD",
        jurisdiction=Jurisdiction.US,
    )
    doubled = model.round_trip(
        notional_ccy=NOTIONAL,
        instrument_currency="USD",
        jurisdiction=Jurisdiction.US,
        exit_notional_ccy=NOTIONAL * 2,
    )
    assert doubled.total_ccy > flat.total_ccy
    assert doubled.exit.total_ccy == flat.exit.total_ccy * 2


def test_a_cost_below_the_floor_is_raised_to_it(limits_file: Path) -> None:
    """The backstop against this module's own arithmetic.

    Every other error here is recoverable. "The model said it was free" is how
    a strategy gets funded, so a computed cost under the operator's floor is
    refused rather than believed.
    """
    import yaml

    payload = yaml.safe_load(Path(limits_file).read_text(encoding="utf-8"))
    payload["costs"]["assumed_half_spread_bps"] = 0.0
    payload["costs"]["assumed_slippage_bps"] = 0.0
    target = Path(limits_file).parent / "free.yaml"
    target.write_text(yaml.safe_dump(payload), encoding="utf-8")

    model = CostModel(load_hard_limits(target).limits)
    # A domestic UK sell-side-free instrument with no spread would otherwise
    # compute well under the floor.
    trip = model.round_trip(
        notional_ccy=NOTIONAL,
        instrument_currency="GBP",
        jurisdiction=Jurisdiction.OTHER,
    )
    assert trip.floored_to_bps is not None
    assert trip.total_bps == Decimal(str(model.limits.costs.min_round_trip_cost_bps))


# --------------------------------------------------------------------------
# The gate
# --------------------------------------------------------------------------


def test_the_gate_rejects_a_trade_whose_costs_eat_it(model: CostModel) -> None:
    verdict = model.gate(
        expected_cost_bps=Decimal("40"),
        expected_edge_bps=Decimal("50"),
    )
    assert not verdict.allowed
    assert "eats this trade" in verdict.reason
    with pytest.raises(CostError):
        verdict.raise_if_refused()


def test_the_gate_admits_a_trade_with_enough_edge(model: CostModel) -> None:
    verdict = model.gate(
        expected_cost_bps=Decimal("40"),
        expected_edge_bps=Decimal("200"),
    )
    assert verdict.allowed
    assert verdict.ratio is not None
    assert verdict.ratio <= verdict.max_ratio
    verdict.raise_if_refused()


def test_an_absurd_declared_edge_cannot_divide_its_way_through_the_gate(
    model: CostModel,
) -> None:
    """The hole this closes is the gate's own arithmetic.

    The gate computes cost/edge, and the strategy declares the edge. Without a
    ceiling, a spec claiming 10,000bps passes trivially — so the one control
    that keeps the search loop out of the fee trap would be defeatable by the
    search loop itself.
    """
    verdict = model.gate(
        expected_cost_bps=Decimal("40"),
        expected_edge_bps=Decimal("10000"),
    )
    assert not verdict.allowed
    assert "malformed spec" in verdict.reason
    # Refused before the ratio is computed, so no flattering number is produced.
    assert verdict.ratio is None


@pytest.mark.parametrize("edge", [Decimal(0), Decimal("-5")])
def test_a_strategy_that_declares_no_edge_cannot_trade(model: CostModel, edge: Decimal) -> None:
    verdict = model.gate(expected_cost_bps=Decimal("40"), expected_edge_bps=edge)
    assert not verdict.allowed
    assert verdict.ratio is None


def test_an_edge_below_the_declarable_floor_is_refused(model: CostModel) -> None:
    verdict = model.gate(
        expected_cost_bps=Decimal("40"),
        expected_edge_bps=Decimal("1"),
    )
    assert not verdict.allowed
    assert "min_expected_edge_bps" in verdict.reason


def test_gate_trade_gates_on_exactly_the_cost_it_computed(model: CostModel) -> None:
    """One call, so the cost charged and the cost gated on cannot differ."""
    verdict, trip = model.gate_trade(
        notional_ccy=NOTIONAL,
        instrument_currency="USD",
        jurisdiction=Jurisdiction.US,
        expected_edge_bps=Decimal("200"),
    )
    assert verdict.expected_cost_bps == trip.total_bps
    assert verdict.allowed


def test_a_minute_bar_edge_does_not_survive_a_us_round_trip(model: CostModel) -> None:
    """The conclusion, stated as a test rather than as a paragraph.

    20bps is the optimistic top of the gross minute-bar edge band in liquid
    names. It does not clear the gate against a ~40bps round trip, and the
    honest answer is that such a strategy is not a business on this venue.
    """
    verdict, trip = model.gate_trade(
        notional_ccy=NOTIONAL,
        instrument_currency="USD",
        jurisdiction=Jurisdiction.US,
        expected_edge_bps=Decimal("20"),
    )
    assert not verdict.allowed
    assert trip.total_bps > Decimal("20")
