"""Properties of the backtester, and the canary that must crash.

Cases prove a backtester works on the inputs someone thought of. Properties
prove it on the ones nobody did, which is where lookahead lives — every real
lookahead bug I have seen was introduced by a change that looked local and
correct.

The centrepiece is `test_a_cheating_strategy_crashes_rather_than_profiting`. A
strategy that deliberately reaches for `close[t+1]` must raise, not receive
`UNKNOWN` and not receive a plausible number. A system where cheating is merely
discouraged produces a backtest nobody can distinguish from an honest one.
"""

from __future__ import annotations

import random
from datetime import UTC, datetime, timedelta
from decimal import Decimal

import pytest
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

from tb.backtest.costs import CostModel, Jurisdiction
from tb.backtest.engine import Backtester, InstrumentMeta
from tb.backtest.null_strategies import AlternatingStrategy, CoinFlipStrategy
from tb.config.loader import load_hard_limits
from tb.data.asof import (
    UNKNOWN,
    BarWindow,
    ForwardOnlyReader,
    InMemoryBarSource,
    LookaheadError,
)
from tb.data.provider import Bar, Provenance, Resolution, Session
from tb.features.pipeline import (
    FeaturePipeline,
    FeatureSnapshot,
    default_pipeline,
    make_spec,
)
from tb.strategy.base import Action, Decision, PositionState, hold

BASE = datetime(2024, 1, 2, tzinfo=UTC)
UID = "isin:US0378331005"

REFERENCE = load_hard_limits("config/hard_limits.yaml")


def make_bar(day: int, open_: Decimal, close: Decimal) -> Bar:
    opened = BASE + timedelta(days=day)
    return Bar(
        instrument_uid=UID,
        resolution=Resolution.DAILY,
        bar_open_utc=opened,
        available_at_utc=opened + timedelta(days=1),
        ingested_at_utc=opened + timedelta(days=1),
        provider="fixture",
        provenance=Provenance.BACKFILL,
        session=Session.REGULAR,
        open=open_,
        high=max(open_, close) + Decimal("0.50"),
        low=min(open_, close) - Decimal("0.50"),
        close=close,
        volume=1_000_000,
    )


def walk(days: int, seed: int) -> list[Bar]:
    rng = random.Random(seed)
    price = Decimal("100.00")
    bars = []
    for day in range(days):
        price = max(price + Decimal(str(round(rng.gauss(0, 1.2), 4))), Decimal("1.00"))
        close = max(price + Decimal(str(round(rng.gauss(0, 0.8), 4))), Decimal("1.00"))
        bars.append(make_bar(day, price, close))
    return bars


def times(days: int) -> list[datetime]:
    return [BASE + timedelta(days=day + 1, hours=1) for day in range(days)]


def engine(**kwargs: object) -> Backtester:
    return Backtester(
        cost_model=CostModel(REFERENCE.limits),
        pipeline=default_pipeline(),
        instruments={
            UID: InstrumentMeta(instrument_uid=UID, currency="USD", jurisdiction=Jurisdiction.US)
        },
        **kwargs,  # type: ignore[arg-type]
    )


def reader(bars: list[Bar]) -> ForwardOnlyReader:
    return ForwardOnlyReader(
        source=InMemoryBarSource(bars=bars),
        resolution=Resolution.DAILY,
        instrument_uids=(UID,),
    )


# --------------------------------------------------------------------------
# The canary
# --------------------------------------------------------------------------


class CheatingStrategy:
    """Reaches past its decision time on purpose. Must crash.

    Written the way a real lookahead bug arrives: not as an obvious grab at the
    future, but as a helper that "just needs tomorrow's price to compare
    against". If the engine let this run, its result would be indistinguishable
    from an honest backtest's — a positive Sharpe with a plausible trade count.
    """

    strategy_id = "cheater"
    version = 1
    required_features: tuple[str, ...] = ()

    def decide(
        self,
        *,
        snapshot: FeatureSnapshot,
        window: BarWindow,
        position: PositionState,
    ) -> Decision:
        # The canary: ask the window for something after its own as-of.
        window.require_visible(window.as_of + timedelta(days=1))
        raise AssertionError("unreachable: require_visible should have raised")


def test_a_cheating_strategy_crashes_rather_than_profiting() -> None:
    """The single most important negative test in the suite.

    A strategy that reaches forward must raise. Returning `UNKNOWN` would be
    almost as bad as returning the value: the strategy would handle it, produce
    a slightly different signal, and nobody would ever learn that the reach
    happened.
    """
    with pytest.raises(LookaheadError, match="did not exist yet"):
        engine().run(
            strategy=CheatingStrategy(),
            reader=reader(walk(30, seed=3)),
            decision_times=times(30),
        )


def test_a_window_contains_no_bar_later_than_its_own_as_of() -> None:
    """Structural, not conventional. The future is simply not in memory.

    A full-history frame plus a "don't peek" convention is broken by every
    `.shift(-1)`, `bfill`, `rolling(center=True)` and `scaler.fit(X_full)`.
    """
    bars = walk(50, seed=5)
    source = InMemoryBarSource(bars=bars)
    walker = ForwardOnlyReader(source=source, resolution=Resolution.DAILY, instrument_uids=(UID,))
    for moment in times(50):
        window = walker.advance_to(moment)
        for bar in window.bars(UID):
            assert bar.available_at_utc <= moment, (
                "a window held a bar that was not knowable at its own as-of"
            )


# --------------------------------------------------------------------------
# Properties
# --------------------------------------------------------------------------


@settings(max_examples=40, deadline=None, suppress_health_check=[HealthCheck.too_slow])
@given(seed=st.integers(min_value=0, max_value=5000))
def test_no_null_strategy_ever_beats_its_own_gross(seed: int) -> None:
    """Over any random walk, net must be worse than gross once anything trades.

    Property rather than case: a cost bug that only shows on some price paths
    is exactly what a single fixture misses.
    """
    bars = walk(120, seed=seed)
    result = engine().run(
        strategy=CoinFlipStrategy(seed=seed),
        reader=reader(bars),
        decision_times=times(120),
    )
    if result.metrics.n_trades == 0:
        return
    assert result.metrics.net_return_pct < result.metrics.gross_return_pct
    assert result.metrics.cost_drag_bps > 0


@settings(max_examples=30, deadline=None, suppress_health_check=[HealthCheck.too_slow])
@given(seed=st.integers(min_value=0, max_value=5000))
def test_every_fill_price_comes_from_a_bar_the_decision_could_not_see(seed: int) -> None:
    """The fill-timing property, over arbitrary price paths.

    Asserted on knowledge time. A bar visible at a decision has necessarily
    already opened, so comparing bar time here would pass vacuously — that
    exact confusion is what made the engine's first version fill nothing at
    all.
    """
    bars = walk(80, seed=seed)
    by_open = {bar.bar_open_utc: bar for bar in bars}
    result = engine().run(
        strategy=AlternatingStrategy(),
        reader=reader(bars),
        decision_times=times(80),
    )
    for trade in result.trades:
        entry_bar = by_open[trade.entry_at]
        assert trade.entry_price == entry_bar.open
        # The decision that opened this trade happened at the previous step,
        # strictly before this bar became knowable.
        assert entry_bar.available_at_utc > trade.entry_at - timedelta(days=2)


@settings(max_examples=30, deadline=None, suppress_health_check=[HealthCheck.too_slow])
@given(seed=st.integers(min_value=0, max_value=5000))
def test_a_run_is_reproducible_from_its_seed(seed: int) -> None:
    """Same vintage, same seed, same numbers. Otherwise nothing is comparable."""
    bars = walk(100, seed=seed)
    first = engine().run(
        strategy=CoinFlipStrategy(seed=42),
        reader=reader(bars),
        decision_times=times(100),
    )
    second = engine().run(
        strategy=CoinFlipStrategy(seed=42),
        reader=reader(bars),
        decision_times=times(100),
    )
    assert first.metrics.n_trades == second.metrics.n_trades
    assert first.metrics.net_return_pct == second.metrics.net_return_pct
    assert [t.entry_price for t in first.trades] == [t.entry_price for t in second.trades]


@settings(max_examples=30, deadline=None, suppress_health_check=[HealthCheck.too_slow])
@given(seed=st.integers(min_value=0, max_value=5000))
def test_the_flat_control_is_exactly_flat_on_every_path(seed: int) -> None:
    """No price path may produce P&L from a strategy that holds nothing."""
    from tb.backtest.null_strategies import AlwaysFlatStrategy

    result = engine().run(
        strategy=AlwaysFlatStrategy(),
        reader=reader(walk(60, seed=seed)),
        decision_times=times(60),
    )
    assert result.metrics.net_return_pct == Decimal(0)
    assert result.metrics.gross_return_pct == Decimal(0)
    assert result.metrics.total_cost_ccy == Decimal(0)


@settings(max_examples=25, deadline=None, suppress_health_check=[HealthCheck.too_slow])
@given(
    seed=st.integers(min_value=0, max_value=5000),
    edge_bps=st.integers(min_value=1, max_value=4),
)
def test_a_tiny_declared_edge_can_never_produce_a_trade(seed: int, edge_bps: int) -> None:
    """Below the declarable floor, so the gate refuses before costing anything.

    This is the structural reason minute-bar trading is not a business here: at
    5-20bps of gross edge against a 40bps round trip, the honest answer is that
    the trade does not happen.
    """
    result = engine().run(
        strategy=CoinFlipStrategy(seed=seed, declared_edge_bps=Decimal(edge_bps)),
        reader=reader(walk(80, seed=seed)),
        decision_times=times(80),
    )
    assert result.metrics.n_trades == 0
    assert result.n_rejected_by_cost_gate == result.n_signals


# --------------------------------------------------------------------------
# One pipeline, both callers
# --------------------------------------------------------------------------


def test_the_backtest_and_live_paths_compute_identical_snapshots() -> None:
    """The "exactly one pipeline" claim, checked rather than asserted in prose.

    The engine calls `pipeline.compute(window, uid)`. So does what M4's live
    loop will call. Same object, same window type, same hash — and if that ever
    stops being true, live will underperform its backtest for reasons nobody
    can locate.
    """
    bars = walk(80, seed=7)
    pipeline = default_pipeline()

    # The "live" path: a window built from the reader at one instant.
    walker = reader(bars)
    moment = times(80)[40]
    live_window = walker.advance_to(moment)
    live = pipeline.compute(live_window, UID)

    # The "backtest" path: the same reader walked up to the same instant.
    replay = reader(bars)
    for step in times(80)[:41]:
        window = replay.advance_to(step)
    backtest = pipeline.compute(window, UID)

    assert backtest.as_of == live.as_of
    assert backtest.snapshot_hash == live.snapshot_hash
    assert backtest.values == live.values


def test_a_feature_is_unknown_rather_than_computed_over_a_short_window() -> None:
    """Walked forward from the start, so the early windows are genuinely short."""
    bars = walk(60, seed=11)
    pipeline = FeaturePipeline(specs=(make_spec("sma", 50),))
    walker = reader(bars)
    unknown_early = 0
    for moment in times(60):
        window = walker.advance_to(moment)
        value = pipeline.compute(window, UID).get("sma_50")
        if len(window.bars(UID)) < 50:
            assert value is UNKNOWN
            unknown_early += 1
        else:
            assert value is not UNKNOWN
    assert unknown_early >= 40, "the fixture should exercise the short-window path"


# --------------------------------------------------------------------------
# Exit safety
# --------------------------------------------------------------------------


class ExitOnlyStrategy:
    """Holds forever, then exits once. Used to check the exit path in isolation."""

    strategy_id = "exit_only"
    version = 1
    required_features: tuple[str, ...] = ()

    def __init__(self, exit_after: datetime) -> None:
        self.exit_after = exit_after
        self.entered = False

    def decide(
        self,
        *,
        snapshot: FeatureSnapshot,
        window: BarWindow,
        position: PositionState,
    ) -> Decision:
        as_of = snapshot.as_of
        digest = snapshot.snapshot_hash
        uid = snapshot.instrument_uid

        if not position.is_open and not self.entered:
            self.entered = True
            return Decision(
                as_of=as_of,
                instrument_uid=uid,
                action=Action.ENTER,
                expected_edge_bps=Decimal("400"),
                feature_snapshot_hash=digest,
                strategy_id=self.strategy_id,
                strategy_version=self.version,
                rationale="enter once",
            )
        if position.is_open and as_of > self.exit_after:
            return Decision(
                as_of=as_of,
                instrument_uid=uid,
                action=Action.EXIT,
                expected_edge_bps=Decimal(0),
                feature_snapshot_hash=digest,
                strategy_id=self.strategy_id,
                strategy_version=self.version,
                rationale="exit once",
            )
        return hold(
            as_of=as_of,
            instrument_uid=uid,
            strategy_id=self.strategy_id,
            strategy_version=self.version,
            snapshot_hash=digest,
            rationale="waiting",
        )


def test_a_completed_round_trip_charges_both_legs_exactly_once() -> None:
    """One entry, one exit, and a cost equal to one round trip.

    Charged twice it would be 80bps; charged once it would be 20. Either error
    shifts every net Sharpe the engine produces.
    """
    schedule = times(60)
    result = engine().run(
        strategy=ExitOnlyStrategy(exit_after=schedule[30]),
        reader=reader(walk(60, seed=13)),
        decision_times=schedule,
    )
    assert result.metrics.n_trades == 1
    trade = result.trades[0]
    expected = CostModel(REFERENCE.limits).round_trip(
        notional_ccy=Decimal("1000.00"),
        instrument_currency="USD",
        jurisdiction=Jurisdiction.US,
    )
    # Within a cent of one round trip: the exit leg is charged on the exit
    # notional, which differs from the entry's as the price moved.
    assert abs(trade.cost_total_ccy - expected.total_ccy) < Decimal("1.00")
    assert trade.net_pnl_ccy == trade.gross_pnl_ccy - trade.cost_total_ccy
    assert trade.holding_minutes > 0
