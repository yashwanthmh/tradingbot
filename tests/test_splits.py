"""Splits reach every price a decision or a result is computed from.

Found by the pre-M7 audit. The factor algebra and the pipeline's `actions`
parameter existed, and nothing in production passed any actions: the loop, the
regime gate, the backtester, the research search and the holdout evaluation all
computed features over raw prices. So a 4-for-1 read as a 75% fall in every
feature across it; a split in the index fund left its 200-day average four times
the price for most of a year, which is RISK_OFF and half exposure; and a
backtest holding across a split booked the old share count at the new price — a
75% loss on a trade that made nothing, which the search would learn to avoid as
though it were a signal.

Passing the actions is only right if three other things are, and each is pinned
here:

* Yahoo's history is already split-adjusted up to the day it was fetched, so a
  bar is scaled from the last session its price reflects (`Bar.quoted_through`);
  scaling it from its own session would divide by the ratio twice.
* A holding follows a split whether or not anyone had recorded it
  (`holding_factor`) — shares change at the ex-date regardless.
* A vintage's backtest reads the actions it was sealed with
  (`SnapshotStore.actions_of`), not whatever a later backfill added.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from datetime import UTC, date, datetime, timedelta
from decimal import Decimal
from fractions import Fraction
from pathlib import Path
from typing import Any

from tb.backtest.costs import CostModel, Jurisdiction
from tb.backtest.engine import Backtester, InstrumentMeta
from tb.config.loader import load_hard_limits
from tb.data.actions import ActionStore
from tb.data.adjustments import (
    RESIDUAL_DETECTOR,
    ActionType,
    CorporateAction,
    holding_factor,
)
from tb.data.asof import BarWindow, ForwardOnlyReader, InMemoryBarSource, visible_bars
from tb.data.barstore import BarStore
from tb.data.calendar import TradingCalendar
from tb.data.provider import Bar, BarBatch, Provenance, Resolution, Session
from tb.data.regime import RegimeGate, RegimeState
from tb.data.snapshot import SnapshotStore
from tb.features.pipeline import FeaturePipeline, FeatureSnapshot, make_spec
from tb.ledger.store import Ledger
from tb.strategy.base import Action, Decision, PositionState, hold
from tb.strategy.trivial import specs
from tests.test_loop import AS_OF, UID, _broker, _loop, _rising_bars, _seed

OTHER = "isin:US5949181045"
BASE = datetime(2026, 3, 2, tzinfo=UTC)
# Known long before any bar here, so knowledge time never gets in the way of
# what a test is about.
EARLY = datetime(2025, 1, 1, tzinfo=UTC)


def daily(
    offset: int,
    close: str,
    *,
    provider: str = "alpaca",
    uid: str = UID,
    ingested: datetime | None = None,
) -> Bar:
    opened = BASE + timedelta(days=offset)
    price = Decimal(close)
    return Bar(
        instrument_uid=uid,
        resolution=Resolution.DAILY,
        bar_open_utc=opened,
        available_at_utc=opened + timedelta(days=1),
        ingested_at_utc=ingested or opened + timedelta(days=1),
        provider=provider,
        provenance=Provenance.BACKFILL,
        session=Session.REGULAR,
        open=price,
        high=price,
        low=price,
        close=price,
        volume=1_000_000,
    )


def split(
    on: date,
    ratio: Fraction,
    *,
    known: datetime = EARLY,
    uid: str = UID,
    inferred: bool = False,
) -> CorporateAction:
    return CorporateAction(
        action_id=f"act_{uid}_{on}_{ratio.numerator}_{ratio.denominator}_{known.date()}",
        instrument_uid=uid,
        action_type=ActionType.SPLIT,
        effective_date=on,
        known_at_utc=known,
        ratio_num=ratio.numerator,
        ratio_den=ratio.denominator,
        source_provider=RESIDUAL_DETECTOR if inferred else "alpaca",
        inferred_from_price_jump=inferred,
    )


def window(bars: list[Bar], as_of: datetime | None = None) -> BarWindow:
    return BarWindow(
        as_of=as_of or max(bar.available_at_utc for bar in bars),
        resolution=Resolution.DAILY,
        _by_uid={bars[0].instrument_uid: tuple(bars)},
    )


FOUR_FOR_ONE = Fraction(4, 1)
# BASE + 2 days: the third session, on which the price is quoted post-split.
EX_DATE = date(2026, 3, 4)
SHAPE = FeaturePipeline(specs=(make_spec("return_pct", 4), make_spec("sma", 4)))


# --------------------------------------------------------------------------
# The pipeline: each bar scaled from the scale it is already on
# --------------------------------------------------------------------------


def test_a_vendor_adjusted_history_is_not_divided_again() -> None:
    """The same four sessions, raw from Alpaca and as Yahoo serves them after
    the split. Both must read as no move at all. Scaled from each bar's own
    session, Yahoo's pre-split bars — already divided by four — would be
    divided again: a 300% "rise" over four flat days."""
    action = split(EX_DATE, FOUR_FOR_ONE)
    fetched = datetime(2026, 3, 9, 15, tzinfo=UTC)
    raw = [daily(0, "200"), daily(1, "200"), daily(2, "50"), daily(3, "50")]
    yahoo = [daily(offset, "50", provider="yahoo", ingested=fetched) for offset in range(4)]

    for bars in (raw, yahoo):
        snapshot = SHAPE.compute(window(bars), UID, actions=[action])
        assert snapshot.get("return_pct_4") == 0, bars[0].provider
        assert snapshot.get("sma_4") == Decimal("50"), bars[0].provider


def test_a_yahoo_history_fetched_before_a_split_and_extended_after_it() -> None:
    """Fetched on the old scale, then extended with bars quoted after the
    split: the older bars carry no adjustment yet, so they are the ones to
    scale, and the series is continuous."""
    before_the_split = datetime(2026, 3, 4, 2, tzinfo=UTC)  # 21:00 New York, 3 March
    bars = [
        daily(0, "200", provider="yahoo", ingested=before_the_split),
        daily(1, "200", provider="yahoo", ingested=before_the_split),
        daily(2, "50", provider="yahoo"),
        daily(3, "50", provider="yahoo"),
    ]
    snapshot = SHAPE.compute(window(bars), UID, actions=[split(EX_DATE, FOUR_FOR_ONE)])
    assert snapshot.get("return_pct_4") == 0


def test_another_instruments_split_does_not_touch_this_one() -> None:
    """`price_factor` never looks at the instrument, so the pipeline does:
    `compute_all` hands every instrument the same list."""
    bars = [daily(0, "200"), daily(1, "200"), daily(2, "50"), daily(3, "50")]
    theirs = split(EX_DATE, FOUR_FOR_ONE, uid=OTHER)
    ours = split(EX_DATE, FOUR_FOR_ONE)

    assert SHAPE.compute(window(bars), UID, actions=[theirs]).get("return_pct_4") == -75
    assert SHAPE.compute(window(bars), UID, actions=[theirs, ours]).get("return_pct_4") == 0


# --------------------------------------------------------------------------
# A holding across a split
# --------------------------------------------------------------------------


def test_a_holding_follows_each_split_between_its_scales() -> None:
    actions = [split(EX_DATE, FOUR_FOR_ONE), split(date(2026, 3, 11), Fraction(1, 2))]

    assert holding_factor(actions, quoted_through=date(2026, 3, 3), to=EX_DATE) == 4
    # A bar dated the ex-date is already quoted after it.
    assert holding_factor(actions, quoted_through=EX_DATE, to=date(2026, 3, 10)) == 1
    assert holding_factor(actions, quoted_through=date(2026, 3, 3), to=date(2026, 3, 12)) == 2
    # Backwards, as at a seam from a vendor-adjusted feed onto a raw one.
    assert holding_factor(actions, quoted_through=date(2026, 3, 12), to=date(2026, 3, 3)) == (
        Fraction(1, 2)
    )


def test_a_holding_follows_a_split_learned_late_but_never_a_guess() -> None:
    """Shares change at the ex-date whether or not anyone had recorded the
    split; an inferred ratio resizes nothing; a restated ratio counts at its
    newest vintage."""
    span = {"quoted_through": date(2026, 3, 3), "to": date(2026, 3, 5)}
    learned_late = split(EX_DATE, FOUR_FOR_ONE, known=datetime(2027, 1, 1, tzinfo=UTC))
    guessed = split(EX_DATE, FOUR_FOR_ONE, inferred=True)
    first = split(EX_DATE, Fraction(3, 1))
    corrected = split(EX_DATE, FOUR_FOR_ONE, known=EARLY + timedelta(days=1))

    assert holding_factor([learned_late], **span) == 4
    assert holding_factor([guessed], **span) == 1
    assert holding_factor([first, corrected], **span) == 4


@dataclass(frozen=True)
class EnterThenExit:
    """Buys at the first decision and sells at the first one after `exit_at`."""

    exit_at: datetime
    strategy_id: str = "enter_then_exit"
    version: int = 1
    required_features: tuple[str, ...] = ()

    def decide(
        self, *, snapshot: FeatureSnapshot, window: BarWindow, position: PositionState
    ) -> Decision:
        action = Action.HOLD
        if position.is_open and snapshot.as_of >= self.exit_at:
            action = Action.EXIT
        elif not position.is_open and snapshot.as_of < self.exit_at:
            action = Action.ENTER
        if action is Action.HOLD:
            return hold(
                as_of=snapshot.as_of,
                instrument_uid=snapshot.instrument_uid,
                strategy_id=self.strategy_id,
                strategy_version=self.version,
                snapshot_hash=snapshot.snapshot_hash,
                rationale="waiting",
            )
        return Decision(
            as_of=snapshot.as_of,
            instrument_uid=snapshot.instrument_uid,
            action=action,
            expected_edge_bps=Decimal("300") if action is Action.ENTER else Decimal(0),
            feature_snapshot_hash=snapshot.snapshot_hash,
            strategy_id=self.strategy_id,
            strategy_version=self.version,
        )


def _through_a_split(limits_file: Path, actions: list[CorporateAction]) -> Any:
    """Ten sessions at 100, a 4-for-1, ten at 25: flat in value throughout."""
    bars = [daily(offset, "100" if offset < 10 else "25") for offset in range(20)]
    engine = Backtester(
        cost_model=CostModel(load_hard_limits(limits_file).limits),
        pipeline=FeaturePipeline(specs=(make_spec("last", 1),)),
        instruments={UID: InstrumentMeta(UID, "USD", Jurisdiction.US)},
        actions={UID: actions},
    )
    return engine.run(
        strategy=EnterThenExit(exit_at=BASE + timedelta(days=15)),
        reader=ForwardOnlyReader(
            source=InMemoryBarSource(bars=bars),
            resolution=Resolution.DAILY,
            instrument_uids=(UID,),
        ),
        decision_times=[BASE + timedelta(days=offset + 1, hours=1) for offset in range(20)],
    )


def test_a_backtest_holding_across_a_split_keeps_its_value(limits_file: Path) -> None:
    """Bought at 100, sold at 25 after a 4-for-1: four times the shares, so the
    trade made exactly nothing before costs, and the curve never dipped. Left
    unadjusted it was a 75% loss, and a drawdown the gate would read as risk."""
    ex_date = (BASE + timedelta(days=10)).date()
    for known in (EARLY, datetime(2027, 1, 1, tzinfo=UTC)):
        result = _through_a_split(limits_file, [split(ex_date, FOUR_FOR_ONE, known=known)])
        (trade,) = result.trades
        assert trade.gross_pnl_ccy == 0, known
        assert trade.quantity == 4 * (Decimal(1000) / Decimal(100))
        assert {point.gross_equity_ccy for point in result.curve} == {result.starting_equity_ccy}

    (unadjusted,) = _through_a_split(limits_file, []).trades
    assert unadjusted.gross_pnl_ccy == Decimal("-750.00")


# --------------------------------------------------------------------------
# The regime gate
# --------------------------------------------------------------------------


def test_a_split_in_the_index_is_not_a_regime_change(limits_file: Path) -> None:
    """A steady uptrend with a 4-for-1 thirty sessions from the end. Unadjusted,
    the last close sits at a quarter of its 200-day average: RISK_OFF, and
    every strategy at half size on a number that is not a signal."""
    gate = RegimeGate(load_hard_limits(limits_file).limits)
    n = gate.min_sessions + 10
    split_at = n - 30

    def price(index: int) -> str:
        level = Decimal(100) + Decimal(index) / Decimal(2)
        return str(level if index < split_at else level / 4)

    bars = [daily(index, price(index), uid=gate.instrument_uid) for index in range(n)]
    ex_date = (BASE + timedelta(days=split_at)).date()
    as_of = BASE + timedelta(days=n, hours=2)

    unadjusted = gate.read_bars(bars, as_of=as_of)
    adjusted = gate.read_bars(
        bars,
        as_of=as_of,
        actions=[split(ex_date, FOUR_FOR_ONE, uid=gate.instrument_uid)],
    )

    assert unadjusted.state is RegimeState.RISK_OFF
    assert adjusted.state is RegimeState.RISK_ON, adjusted.detail


# --------------------------------------------------------------------------
# The loop
# --------------------------------------------------------------------------


def _quartered(bar: Bar) -> Bar:
    return replace(
        bar,
        open=bar.open / 4,
        high=bar.high / 4,
        low=bar.low / 4,
        close=bar.close / 4,
    )


def test_the_loop_reads_a_split_in_its_history_as_no_move(env: dict[str, Any]) -> None:
    """A steady rise with a 4-for-1 fifteen sessions ago. On raw prices the fast
    average has fallen far below the slow one and the strategy holds; the loop
    reads the split the ledger records, sees the rise, and enters."""
    rising = _rising_bars(days=140)
    split_at = len(rising) - 15
    bars = [bar if index < split_at else _quartered(bar) for index, bar in enumerate(rising)]
    action = split(bars[split_at].session_date, FOUR_FOR_ONE)
    _seed(env, bars)

    pinned = env["pinned"]
    with Ledger(env["db"], config_hash=pinned.config_hash) as ledger:
        ActionStore(ledger).record([action])
        loop = _loop(env, ledger, _broker())
        loop.lock.acquire(at=AS_OF)
        result = loop.run_cycle()
        store = BarStore(ledger, root=env["bars"], scale=pinned.limits.data.price_scale)
        seen = visible_bars(store, UID, Resolution.DAILY, as_of=AS_OF)

    raw = FeaturePipeline(specs=specs()).compute(window(list(seen), AS_OF), UID)
    fast, slow = raw.get("sma_20"), raw.get("sma_100")
    assert isinstance(fast, Decimal) and isinstance(slow, Decimal)
    assert fast < slow, "vacuous: on raw prices the split should read as a crash"
    (decision,) = result.decisions
    assert decision.action is Action.ENTER, decision.rationale


# --------------------------------------------------------------------------
# A vintage's actions
# --------------------------------------------------------------------------


def test_a_vintage_backtests_on_the_actions_it_was_sealed_with(
    ledger: Ledger, tmp_path: Path
) -> None:
    """An action a later backfill records carries the time it was *public*,
    which can be years before the seal, so the knowledge-time filter would let
    it into a re-run of the vintage. The recording is what decides."""
    calendar = TradingCalendar()
    store = BarStore(ledger, root=tmp_path / "bars")
    snapshots = SnapshotStore(ledger, store, calendar=calendar)
    bars = [daily(offset, "100") for offset in range(5)]
    store.ingest(
        BarBatch(
            bars=tuple(bars),
            provider="alpaca",
            symbol="AAPL",
            resolution=Resolution.DAILY,
            requested_start=bars[0].bar_open_utc,
            requested_end=bars[-1].bar_open_utc,
        )
    )
    sealed_with = split(date(2020, 8, 31), FOUR_FOR_ONE)
    ActionStore(ledger).record([sealed_with])
    vintage = snapshots.seal(resolutions=[Resolution.DAILY])
    ActionStore(ledger).record([split(date(2021, 1, 5), Fraction(1, 10))])

    assert snapshots.actions_of(vintage.vintage_id) == {UID: (sealed_with,)}
