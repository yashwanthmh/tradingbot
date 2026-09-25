"""The backtester, its metrics, and the calibration that judges it.

The centrepiece is `test_the_engine_cannot_manufacture_an_edge`: a population
of strategies with no edge by construction, run over a random walk, must show
no net edge. If that test ever passes with a materially positive net Sharpe,
every result the engine has ever produced is suspect — so it is a release gate
rather than a diagnostic.

`test_a_fill_never_uses_a_price_the_decision_could_see` is the specific
property behind it, asserted directly rather than only statistically.
"""

from __future__ import annotations

import random
from dataclasses import replace
from datetime import UTC, datetime, timedelta
from decimal import Decimal

import pytest

from tb.backtest.calibrate import MIN_MEANINGFUL_COST_DRAG_BPS, run_calibration
from tb.backtest.costs import CostModel, Jurisdiction
from tb.backtest.engine import Backtester, BacktestError, InstrumentMeta
from tb.backtest.metrics import (
    CurvePoint,
    max_drawdown_pct,
    returns_of,
    sharpe,
)
from tb.backtest.null_strategies import (
    AlternatingStrategy,
    AlwaysFlatStrategy,
    AlwaysLongStrategy,
    CoinFlipStrategy,
    population,
)
from tb.config.loader import load_hard_limits
from tb.data.asof import ForwardOnlyReader, InMemoryBarSource
from tb.data.provider import Bar, Provenance, Resolution, Session
from tb.features.pipeline import default_pipeline

BASE = datetime(2024, 1, 2, tzinfo=UTC)
UID = "isin:US0378331005"
OTHER = "isin:US5949181045"


def make_bar(uid: str, day: int, open_: str, close: str) -> Bar:
    opened = BASE + timedelta(days=day)
    o, c = Decimal(open_), Decimal(close)
    return Bar(
        instrument_uid=uid,
        resolution=Resolution.DAILY,
        bar_open_utc=opened,
        available_at_utc=opened + timedelta(days=1),
        ingested_at_utc=opened + timedelta(days=1),
        provider="fixture",
        provenance=Provenance.BACKFILL,
        session=Session.REGULAR,
        open=o,
        high=max(o, c) + Decimal("0.50"),
        low=min(o, c) - Decimal("0.50"),
        close=c,
        volume=1_000_000,
    )


def random_walk(uid: str, *, days: int, seed: int) -> list[Bar]:
    """A pure random walk: zero expected drift, so no gross edge exists."""
    walk = random.Random(seed)
    price = Decimal("100.00")
    bars = []
    for day in range(days):
        price = max(price + Decimal(str(round(walk.gauss(0, 1.2), 4))), Decimal("1.00"))
        close = max(price + Decimal(str(round(walk.gauss(0, 0.8), 4))), Decimal("1.00"))
        bars.append(make_bar(uid, day, str(price), str(close)))
    return bars


def decision_times(days: int) -> list[datetime]:
    """One decision per session, an hour after the previous bar became knowable."""
    return [BASE + timedelta(days=day + 1, hours=1) for day in range(days)]


def meta(uid: str, *, currency: str = "USD", j: Jurisdiction = Jurisdiction.US) -> InstrumentMeta:
    return InstrumentMeta(instrument_uid=uid, currency=currency, jurisdiction=j)


@pytest.fixture
def cost_model(limits_file) -> CostModel:  # type: ignore[no-untyped-def]
    return CostModel(load_hard_limits(limits_file).limits)


def build(cost_model: CostModel, uids: list[str], **kwargs: object) -> Backtester:
    return Backtester(
        cost_model=cost_model,
        pipeline=default_pipeline(),
        instruments={uid: meta(uid) for uid in uids},
        **kwargs,  # type: ignore[arg-type]
    )


def reader_over(bars: list[Bar], uids: list[str]) -> ForwardOnlyReader:
    return ForwardOnlyReader(
        source=InMemoryBarSource(bars=bars),
        resolution=Resolution.DAILY,
        instrument_uids=tuple(uids),
    )


# --------------------------------------------------------------------------
# The property that stops the engine cheating
# --------------------------------------------------------------------------


def test_a_fill_never_uses_a_price_the_decision_could_see(cost_model: CostModel) -> None:
    """Filling at the decision bar's close is how a backtest invents returns.

    Worth roughly the entire gross edge at minute resolution, because the close
    that produced the signal is the price you would be filled at. Every fill
    here must come from a bar whose *knowledge* time is after the decision.
    """
    bars = random_walk(UID, days=60, seed=3)
    engine = build(cost_model, [UID])
    result = engine.run(
        strategy=AlternatingStrategy(),
        reader=reader_over(bars, [UID]),
        decision_times=decision_times(60),
    )
    assert result.trades, "the fixture should produce trades"

    by_open = {bar.bar_open_utc: bar for bar in bars}
    for trade in result.trades:
        entry_bar = by_open[trade.entry_at]
        # The fill bar became knowable strictly after the decision that caused
        # it — which is the only way the decision could not have used it.
        assert entry_bar.available_at_utc > trade.entry_at - timedelta(hours=1)
        assert trade.entry_price == entry_bar.open


def test_a_backtest_over_a_real_backfill_sees_its_history(cost_model: CostModel) -> None:
    """A regression. A real backfill is stamped with the moment it was fetched,
    after every bar in it, and the read path used to hide a bar until it had
    been ingested — so a backtest over real history had nothing to fill
    against: every decision dropped, no trade, no promotion, ever. Every fixture
    here stamps a bar as ingested when it closed, which is why none showed it.
    The same history must backtest the same way however late it was fetched."""
    fetched = datetime(2026, 9, 25, 9, tzinfo=UTC)
    as_closed = random_walk(UID, days=60, seed=3)
    as_fetched = [replace(bar, ingested_at_utc=fetched) for bar in as_closed]

    results = [
        build(cost_model, [UID]).run(
            strategy=AlternatingStrategy(),
            reader=reader_over(bars, [UID]),
            decision_times=decision_times(60),
        )
        for bars in (as_closed, as_fetched)
    ]

    assert results[0].trades
    assert results[1].n_dropped_no_next_bar == 0
    assert [trade.net_pnl_ccy for trade in results[1].trades] == [
        trade.net_pnl_ccy for trade in results[0].trades
    ]


def test_a_decision_on_the_last_bar_is_dropped_not_filled(cost_model: CostModel) -> None:
    """There is no next bar, so the only available price is one it could see.

    Dropping is the honest option; filling at the last close would be the
    cheat, and silently so.
    """
    bars = random_walk(UID, days=20, seed=5)
    engine = build(cost_model, [UID])
    result = engine.run(
        strategy=AlternatingStrategy(),
        reader=reader_over(bars, [UID]),
        decision_times=decision_times(20),
    )
    # The loop breaks before deciding on the final step, so nothing is even
    # proposed there — which is the same guarantee reached earlier.
    assert result.window_end == decision_times(20)[-1]
    assert all(trade.exit_at < decision_times(20)[-1] for trade in result.trades)


def test_the_reader_refuses_to_rewind_so_a_run_cannot_be_reused(
    cost_model: CostModel,
) -> None:
    """A reader that could revisit an earlier moment could fit and then predict."""
    bars = random_walk(UID, days=30, seed=11)
    reader = reader_over(bars, [UID])
    engine = build(cost_model, [UID])
    engine.run(
        strategy=AlwaysFlatStrategy(),
        reader=reader,
        decision_times=decision_times(30),
    )
    with pytest.raises(Exception, match="rewind"):
        reader.advance_to(BASE)


def test_an_out_of_order_schedule_is_refused_before_the_run(cost_model: CostModel) -> None:
    """Better than raising part-way through."""
    bars = random_walk(UID, days=30, seed=13)
    times = decision_times(30)
    scrambled = [*times[5:], *times[:5]]
    with pytest.raises(BacktestError, match="ascending"):
        build(cost_model, [UID]).run(
            strategy=AlwaysFlatStrategy(),
            reader=reader_over(bars, [UID]),
            decision_times=scrambled,
        )


def test_a_single_step_run_is_refused(cost_model: CostModel) -> None:
    """A fill comes from the bar after the decision, so one step cannot trade."""
    bars = random_walk(UID, days=5, seed=17)
    with pytest.raises(BacktestError, match="at least two"):
        build(cost_model, [UID]).run(
            strategy=AlternatingStrategy(),
            reader=reader_over(bars, [UID]),
            decision_times=decision_times(1),
        )


# --------------------------------------------------------------------------
# Costs reach the equity curve
# --------------------------------------------------------------------------


def test_the_flat_control_moves_not_at_all(cost_model: CostModel) -> None:
    """Any movement is P&L with no position to attribute it to."""
    bars = random_walk(UID, days=40, seed=19)
    result = build(cost_model, [UID]).run(
        strategy=AlwaysFlatStrategy(),
        reader=reader_over(bars, [UID]),
        decision_times=decision_times(40),
    )
    assert result.metrics.n_trades == 0
    assert result.metrics.net_return_pct == Decimal(0)
    assert result.metrics.total_cost_ccy == Decimal(0)
    assert "vacuous" in " ".join(result.caveats)


def test_costs_are_charged_once_per_leg(cost_model: CostModel) -> None:
    """Buy and hold pays the entry leg only: it never sold.

    20bps for a US name from a GBP account — 15 FX + 2 spread + 3 slippage —
    and not the 40bps of a round trip.
    """
    bars = random_walk(UID, days=40, seed=23)
    result = build(cost_model, [UID]).run(
        strategy=AlwaysLongStrategy(),
        reader=reader_over(bars, [UID]),
        decision_times=decision_times(40),
    )
    assert result.metrics.cost_drag_bps == pytest.approx(Decimal("20"), abs=Decimal("0.5"))
    # One entry, no completed round trip.
    assert result.metrics.n_trades == 0
    assert result.metrics.total_cost_ccy > 0


def test_net_is_always_worse_than_gross_when_anything_traded(
    cost_model: CostModel,
) -> None:
    """The cost drag, as a measured quantity rather than an argument."""
    bars = random_walk(UID, days=200, seed=29)
    result = build(cost_model, [UID]).run(
        strategy=AlternatingStrategy(),
        reader=reader_over(bars, [UID]),
        decision_times=decision_times(200),
    )
    assert result.metrics.n_trades > 10
    assert result.metrics.net_return_pct < result.metrics.gross_return_pct
    assert result.metrics.cost_drag_bps > 0


def test_maximum_turnover_costs_exactly_what_the_fee_schedule_says(
    cost_model: CostModel,
) -> None:
    """The honest upper bound on frequent trading, checked as arithmetic.

    Asserting the *relationship* rather than a number: total cost must equal
    the round-trip cost times the number of round trips. A magic threshold
    would pass or fail on the fixture's length and instrument count rather
    than on whether both legs are charged — which is the thing being tested.

    Two years of alternate-session trading on one name at 1,000 notional
    against 10,000 of equity comes to about 10% of the account in fees, with
    no bad predictions at all. Scale that across a 25-symbol universe and it
    is the whole account.
    """
    bars = random_walk(UID, days=500, seed=31)
    engine = build(cost_model, [UID])
    result = engine.run(
        strategy=AlternatingStrategy(),
        reader=reader_over(bars, [UID]),
        decision_times=decision_times(500),
    )
    assert result.metrics.n_trades > 100

    per_trip = cost_model.round_trip(
        notional_ccy=engine.position_notional_ccy,
        instrument_currency="USD",
        jurisdiction=Jurisdiction.US,
    ).total_ccy
    expected = per_trip * Decimal(result.metrics.n_trades)
    # Within one leg's worth: the final entry may be open at the end of the
    # window, so its exit leg was never charged.
    assert abs(result.metrics.total_cost_ccy - expected) <= per_trip

    gap = result.metrics.gross_return_pct - result.metrics.net_return_pct
    expected_gap = expected / engine.starting_equity_ccy * Decimal(100)
    assert gap == pytest.approx(expected_gap, abs=Decimal("0.5")), (
        f"cost gap {gap}% does not match the fee schedule's {expected_gap}% — "
        "check that both legs are charged, once each"
    )


def test_a_rejected_signal_never_reaches_the_equity_curve(cost_model: CostModel) -> None:
    """And the rejection count is reported, because it *is* the result.

    "400 signals, 3 affordable trades" is the finding for a minute-resolution
    strategy on this venue, not a footnote.
    """
    bars = random_walk(UID, days=100, seed=37)
    # A declared edge far too small to clear a ~40bps round trip.
    stingy = CoinFlipStrategy(seed=1, declared_edge_bps=Decimal("10"))
    result = build(cost_model, [UID]).run(
        strategy=stingy,
        reader=reader_over(bars, [UID]),
        decision_times=decision_times(100),
    )
    assert result.n_signals > 0
    assert result.n_rejected_by_cost_gate == result.n_signals
    assert result.metrics.n_trades == 0
    assert "cost gate" in " ".join(result.caveats)


def test_an_unknown_jurisdiction_refuses_the_whole_run(cost_model: CostModel) -> None:
    """Rather than defaulting to the zero-tax case for the entire backtest."""
    bars = random_walk(UID, days=30, seed=41)
    engine = Backtester(
        cost_model=cost_model,
        pipeline=default_pipeline(),
        instruments={UID: meta(UID, j=Jurisdiction.UNKNOWN)},
    )
    with pytest.raises(Exception, match="jurisdiction is unknown"):
        engine.run(
            strategy=AlternatingStrategy(),
            reader=reader_over(bars, [UID]),
            decision_times=decision_times(30),
        )


# --------------------------------------------------------------------------
# Metrics
# --------------------------------------------------------------------------


def test_sharpe_is_none_rather_than_zero_when_unmeasurable() -> None:
    """Zero means "measured, no edge"; None means "not measured".

    A promotion gate that could not tell them apart would treat an unmeasured
    strategy as a measured flat one.
    """
    assert sharpe([], periods_per_year=252) is None
    assert sharpe([Decimal("0.01")], periods_per_year=252) is None
    # A perfectly flat curve has no dispersion, so no Sharpe — not an
    # infinite one.
    assert sharpe([Decimal(0)] * 50, periods_per_year=252) is None


def test_the_annualisation_factor_is_passed_not_assumed() -> None:
    """Hardcoding 252 would overstate a minute strategy's Sharpe ~8x."""
    returns = [Decimal("0.001"), Decimal("0.002"), Decimal("-0.001"), Decimal("0.003")]
    daily = sharpe(returns, periods_per_year=252)
    minute = sharpe(returns, periods_per_year=252 * 390)
    assert daily is not None and minute is not None
    assert minute > daily * 19  # sqrt(390) ~ 19.7


def test_returns_skip_a_non_positive_level_rather_than_dividing_by_it() -> None:
    """`inf` would propagate into a mean and emerge as a plausible number."""
    assert returns_of([Decimal(0), Decimal(100)]) == []
    assert len(returns_of([Decimal(100), Decimal(110), Decimal(121)])) == 2


def test_max_drawdown_finds_the_worst_fall() -> None:
    assert max_drawdown_pct([Decimal(100), Decimal(120), Decimal(90)]) == Decimal(25)
    assert max_drawdown_pct([Decimal(100)]) == Decimal(0)
    assert max_drawdown_pct([Decimal(100), Decimal(200)]) == Decimal(0)


def test_cost_drag_is_measured_against_traded_notional_not_equity() -> None:
    """Over equity it would shrink by trading less of the account.

    Which would reward a strategy for being small rather than for being cheap.
    """
    from tb.backtest.metrics import compute

    metrics = compute(
        curve=[CurvePoint(equity_ccy=Decimal(1000), gross_equity_ccy=Decimal(1000))],
        n_trades=1,
        total_cost_ccy=Decimal(4),
        traded_notional_ccy=Decimal(1000),
        starting_equity_ccy=Decimal(10000),
        periods_per_year=252,
    )
    # 4 on 1000 traded is 40bps, regardless of the 10000 of equity.
    assert metrics.cost_drag_bps == Decimal(40)


# --------------------------------------------------------------------------
# Calibration — the release gate
# --------------------------------------------------------------------------


def test_the_engine_cannot_manufacture_an_edge(cost_model: CostModel) -> None:
    """The most important test in the backtester.

    Strategies with no edge by construction, over a random walk with no drift.
    Every net Sharpe must sit at or below the tolerance. A positive one is a
    fill-timing error, a mark from a bar the position could not see, or a cost
    charged on one leg — never a discovery.
    """
    uids = [UID, OTHER]
    bars = random_walk(UID, days=500, seed=1000) + random_walk(OTHER, days=500, seed=1001)
    result = run_calibration(
        cost_model=cost_model,
        source=InMemoryBarSource(bars=bars),
        instruments={uid: meta(uid) for uid in uids},
        decision_times=decision_times(500),
        seed=7,
    )
    assert result.passed, "\n".join(result.failures)
    assert result.total_trades > 100, "a calibration with few trades proves little"

    best = result.best_net_sharpe
    assert best is not None
    assert best <= result.tolerance

    # Gross scatters around zero on a random walk; net is pushed below it by
    # costs. That shift *is* the cost drag, and it is the whole point.
    for run in result.results:
        if run.metrics.n_trades:
            assert run.metrics.net_sharpe is not None
            assert run.metrics.gross_sharpe is not None
            assert run.metrics.net_sharpe < run.metrics.gross_sharpe


def test_a_calibration_that_charges_nothing_is_reported_as_vacuous(
    cost_model: CostModel,
) -> None:
    """The half of the assertion people leave out.

    A run where nothing traded passes the Sharpe test trivially while proving
    nothing about whether costs are charged at all.
    """
    bars = random_walk(UID, days=6, seed=53)
    result = run_calibration(
        cost_model=cost_model,
        source=InMemoryBarSource(bars=bars),
        instruments={UID: meta(UID)},
        decision_times=decision_times(6),
        seed=3,
    )
    if result.total_trades == 0:
        assert not result.passed
        assert any("proves nothing" in f for f in result.failures)


def test_the_cost_drag_floor_is_a_real_threshold() -> None:
    assert MIN_MEANINGFUL_COST_DRAG_BPS > 0


def test_the_null_population_covers_four_distinct_engine_bugs() -> None:
    """Each null catches something the others cannot."""
    nulls = population(seed=0)
    kinds = {type(n).__name__ for n in nulls}
    assert kinds == {
        "AlwaysFlatStrategy",
        "AlwaysLongStrategy",
        "AlternatingStrategy",
        "CoinFlipStrategy",
    }
    # Several coin flips, because a fill-timing error shows up over many
    # entries rather than in any single one.
    assert sum(isinstance(n, CoinFlipStrategy) for n in nulls) >= 5


def test_a_coin_flip_is_reproducible_from_its_seed(cost_model: CostModel) -> None:
    """An unreproducible calibration cannot be compared across commits.

    Which is the only way to notice that a refactor broke the fill timing.
    """
    bars = random_walk(UID, days=120, seed=59)
    runs = []
    for _ in range(2):
        result = build(cost_model, [UID]).run(
            strategy=CoinFlipStrategy(seed=99),
            reader=reader_over(bars, [UID]),
            decision_times=decision_times(120),
        )
        runs.append(result)
    assert runs[0].metrics.n_trades == runs[1].metrics.n_trades
    assert runs[0].metrics.net_return_pct == runs[1].metrics.net_return_pct
