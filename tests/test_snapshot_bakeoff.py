"""Sealed vintages, and the paid-data verdict as arithmetic.

`test_a_vintage_is_immutable_under_later_ingestion` is the property the whole
mechanism exists for. Yahoo back-adjusts history routinely, so re-running the
same backtest against "the store" next month legitimately produces different
numbers with nothing recording why. A vintage is the answer, and it is only an
answer if loading one after arbitrary further ingestion — *including revisions
inside its own window* — returns byte-identical bars.

`test_the_verdict_is_arithmetic_not_an_opinion` is the other one: the bake-off's
output is a single number, the gross edge a strategy would need before the
feed's own error is small enough to trade through, compared against what a
round trip actually costs.
"""

from __future__ import annotations

from datetime import UTC, date, datetime, timedelta
from decimal import Decimal
from pathlib import Path

import pytest

from tb.data.actions import ActionStore
from tb.data.adjustments import ActionType, CorporateAction
from tb.data.asof import LookaheadError
from tb.data.bakeoff import (
    MIN_COMPARABLE_BARS,
    ROUND_TRIP_COST_BPS,
    Bakeoff,
    BakeoffError,
    Verdict,
    compare_feeds,
    delay_p95,
    disagreement_bps,
    may_widen_live_resolutions,
    percentile,
)
from tb.data.barstore import BarStore
from tb.data.calendar import TradingCalendar
from tb.data.fx import FxStore, flat_rate
from tb.data.provider import Bar, BarBatch, Provenance, Resolution, Session
from tb.data.snapshot import (
    DatasetVintage,
    PitCompleteness,
    SnapshotError,
    SnapshotStore,
    VintageAlteredError,
    calendar_hash,
    observed_delays,
)
from tb.data.universe import SurvivorshipFlag, UniverseStore, build_candidates, select
from tb.ledger.events import EventType
from tb.ledger.store import Ledger

UID = "isin:US0378331005"
OTHER = "isin:US5949181045"
CAL = TradingCalendar()


def daily(
    day: date,
    *,
    close: str = "100.00",
    uid: str = UID,
    provider: str = "alpaca",
    provenance: Provenance = Provenance.BACKFILL,
    ingested: datetime | None = None,
) -> Bar:
    bar_open = datetime(day.year, day.month, day.day, tzinfo=UTC)
    price = Decimal(close)
    return Bar(
        instrument_uid=uid,
        resolution=Resolution.DAILY,
        bar_open_utc=bar_open,
        available_at_utc=bar_open + timedelta(days=1),
        ingested_at_utc=ingested or (bar_open + timedelta(days=1)),
        provider=provider,
        provenance=provenance,
        session=Session.REGULAR,
        open=price,
        high=price,
        low=price,
        close=price,
        volume=1_000_000,
    )


def minute(
    moment: datetime,
    *,
    close: str,
    uid: str = UID,
    provider: str = "alpaca",
    session: Session = Session.REGULAR,
    provenance: Provenance = Provenance.BACKFILL,
    delay_seconds: int = 0,
) -> Bar:
    price = Decimal(close)
    return Bar(
        instrument_uid=uid,
        resolution=Resolution.MINUTE,
        bar_open_utc=moment,
        available_at_utc=moment + timedelta(minutes=1, seconds=delay_seconds),
        ingested_at_utc=moment + timedelta(minutes=1, seconds=delay_seconds),
        provider=provider,
        provenance=provenance,
        session=session,
        open=price,
        high=price,
        low=price,
        close=price,
        volume=1000,
    )


@pytest.fixture
def store(ledger: Ledger, tmp_path: Path) -> BarStore:
    return BarStore(ledger, root=tmp_path / "bars")


@pytest.fixture
def snapshots(ledger: Ledger, store: BarStore) -> SnapshotStore:
    return SnapshotStore(ledger, store, calendar=CAL, run_id="run-test")


def ingest(store: BarStore, bars: list[Bar], *, provider: str = "alpaca") -> None:
    store.ingest(
        BarBatch(
            bars=tuple(bars),
            provider=provider,
            symbol="AAPL",
            resolution=bars[0].resolution,
            requested_start=bars[0].bar_open_utc,
            requested_end=bars[-1].bar_open_utc,
        )
    )


def week(store: BarStore, *, uid: str = UID, close: str = "100.00") -> list[date]:
    days = [s.day for s in CAL.sessions_between(date(2026, 3, 2), date(2026, 3, 6))]
    ingest(store, [daily(day, uid=uid, close=close) for day in days])
    return days


# ==========================================================================
# Sealing
# ==========================================================================


def test_sealing_records_a_vintage_and_its_event(
    ledger: Ledger, store: BarStore, snapshots: SnapshotStore
) -> None:
    week(store)
    vintage = snapshots.seal(resolutions=[Resolution.DAILY])

    assert vintage.vintage_id.startswith("vint_")
    assert vintage.row_count == 5
    assert vintage.n_files == 1
    assert vintage.instrument_uids == (UID,)

    events = list(ledger.iter_events(event_type=EventType.DATA_SNAPSHOT_SEALED))
    assert len(events) == 1
    assert events[0]["aggregate_id"] == vintage.vintage_id

    loaded = snapshots.get(vintage.vintage_id)
    assert loaded is not None
    assert loaded.manifest_hash == vintage.manifest_hash


def test_sealing_compacts_hot_rows_first(store: BarStore, snapshots: SnapshotStore) -> None:
    """A vintage names sealed Parquet, so staged rows must be sealed first.

    Otherwise the snapshot is silently missing the newest data while the
    catalog looks complete and the row count looks plausible.
    """
    week(store)
    assert store.has_hot_rows()
    vintage = snapshots.seal(resolutions=[Resolution.DAILY])
    assert not store.has_hot_rows()
    assert vintage.row_count == 5


def test_sealing_without_compacting_refuses_rather_than_omitting_data(
    store: BarStore, snapshots: SnapshotStore
) -> None:
    week(store)
    with pytest.raises(SnapshotError, match="omit the newest"):
        snapshots.seal(resolutions=[Resolution.DAILY], compact_first=False)


def test_sealing_the_same_data_twice_is_idempotent(
    ledger: Ledger, store: BarStore, snapshots: SnapshotStore
) -> None:
    """The id is content-derived, so `tb data seal` can be run twice safely.

    Which matters, because it is exactly the command someone runs again when
    unsure whether the first one worked.
    """
    week(store)
    first = snapshots.seal(resolutions=[Resolution.DAILY])
    second = snapshots.seal(resolutions=[Resolution.DAILY])
    assert first.vintage_id == second.vintage_id
    assert len(list(ledger.iter_events(event_type=EventType.DATA_SNAPSHOT_SEALED))) == 1


def test_sealing_an_empty_store_refuses(snapshots: SnapshotStore) -> None:
    """An empty vintage is admissible-looking evidence for a backtest over nothing."""
    with pytest.raises(SnapshotError, match="nothing to seal"):
        snapshots.seal(resolutions=[Resolution.DAILY])


def test_a_vintage_can_be_restricted_to_some_instruments(
    store: BarStore, snapshots: SnapshotStore
) -> None:
    week(store, uid=UID)
    week(store, uid=OTHER)
    vintage = snapshots.seal(resolutions=[Resolution.DAILY], instrument_uids=[UID])
    assert vintage.instrument_uids == (UID,)
    assert vintage.row_count == 5


# ==========================================================================
# What a vintage pins
# ==========================================================================


def test_a_restated_action_produces_a_different_vintage(
    ledger: Ledger, store: BarStore, snapshots: SnapshotStore
) -> None:
    """A changed split ratio changes every adjusted price before its ex-date.

    So the same bars with a different action table are genuinely a different
    dataset, and must not share a vintage id with the first.
    """
    week(store)
    before = snapshots.seal(resolutions=[Resolution.DAILY])

    ActionStore(ledger).record(
        [
            CorporateAction(
                action_id="act_split",
                instrument_uid=UID,
                action_type=ActionType.SPLIT,
                effective_date=date(2026, 3, 4),
                known_at_utc=datetime(2026, 2, 1, tzinfo=UTC),
                source_provider="alpaca",
                ratio_num=4,
                ratio_den=1,
            )
        ]
    )
    after = snapshots.seal(resolutions=[Resolution.DAILY])
    assert after.vintage_id != before.vintage_id
    assert after.action_table_hash != before.action_table_hash


def test_a_revised_fx_rate_produces_a_different_vintage(
    ledger: Ledger, store: BarStore, snapshots: SnapshotStore
) -> None:
    """The limits are GBP and the universe is USD.

    A revised rate changes every position size and every P&L figure, so it is
    part of what a backtest depended on.
    """
    week(store)
    before = snapshots.seal(resolutions=[Resolution.DAILY])

    FxStore(ledger).record(
        flat_rate(base="GBP", quote="USD", rate=Decimal("1.27"), days=[date(2026, 3, 4)])
    )
    after = snapshots.seal(resolutions=[Resolution.DAILY])
    assert after.vintage_id != before.vintage_id
    assert after.fx_table_hash != before.fx_table_hash


def test_the_calendar_is_pinned_by_content() -> None:
    """Which days existed changes a backtest's results."""
    default = calendar_hash(TradingCalendar())
    narrower = calendar_hash(
        TradingCalendar(covers_from=date(2026, 1, 1), covers_through=date(2026, 12, 31))
    )
    assert default != narrower


def test_the_manifest_hash_does_not_depend_on_row_order(
    store: BarStore, snapshots: SnapshotStore
) -> None:
    """A hash that varied with SQL ordering would reseal to a new id each time."""
    week(store, uid=UID)
    week(store, uid=OTHER)
    vintage = snapshots.seal(resolutions=[Resolution.DAILY])

    shuffled = DatasetVintage(
        vintage_id=vintage.vintage_id,
        as_of_utc=vintage.as_of_utc,
        manifest_hash=vintage.manifest_hash,
        file_sha256s=tuple(reversed(vintage.file_sha256s)),
        instrument_uids=tuple(reversed(vintage.instrument_uids)),
        resolutions=vintage.resolutions,
        window_start=vintage.window_start,
        window_end=vintage.window_end,
        row_count=vintage.row_count,
        calendar_hash=vintage.calendar_hash,
        action_table_hash=vintage.action_table_hash,
        fx_table_hash=vintage.fx_table_hash,
        universe_snapshot_id=vintage.universe_snapshot_id,
        survivorship_flag=vintage.survivorship_flag,
        pit_completeness_flag=vintage.pit_completeness_flag,
    )
    assert shuffled.manifest() == vintage.manifest()


# ==========================================================================
# Immutability — the property the mechanism exists for
# ==========================================================================


def test_a_vintage_is_immutable_under_later_ingestion(
    store: BarStore, snapshots: SnapshotStore
) -> None:
    """Including revisions *inside* its own window.

    This is the property Yahoo's back-adjustment breaks if nothing defends it:
    re-running the same backtest against "the store" would legitimately return
    different numbers. A vintage reads exactly the files it named.
    """
    week(store)
    vintage = snapshots.seal(resolutions=[Resolution.DAILY])
    original = snapshots.bars_of(vintage.vintage_id)
    original_rows = [bar.storage_row(scale=store.scale) for bar in original]

    # A restatement of a bar inside the sealed window, plus a new bar after it.
    ingest(
        store,
        [
            daily(
                date(2026, 3, 4),
                close="999.00",
                ingested=datetime(2026, 6, 1, tzinfo=UTC),
            ),
            daily(date(2026, 3, 9), close="101.00"),
        ],
    )
    store.compact()

    reloaded = snapshots.bars_of(vintage.vintage_id)
    assert [bar.storage_row(scale=store.scale) for bar in reloaded] == original_rows
    assert not any(bar.close == Decimal("999.000000") for bar in reloaded)


def test_a_vintage_whose_file_changed_refuses_to_load(
    store: BarStore, snapshots: SnapshotStore
) -> None:
    """Silently continuing would make every decision resting on it unfalsifiable."""
    week(store)
    vintage = snapshots.seal(resolutions=[Resolution.DAILY])

    target = next(store.root.rglob("*.parquet"))
    target.write_bytes(b"not a parquet file")

    assert snapshots.verify(vintage.vintage_id)
    with pytest.raises(VintageAlteredError, match="no longer matches"):
        snapshots.bars_of(vintage.vintage_id)


def test_a_vintage_whose_file_vanished_is_reported(
    store: BarStore, snapshots: SnapshotStore
) -> None:
    week(store)
    vintage = snapshots.seal(resolutions=[Resolution.DAILY])
    for path in store.root.rglob("*.parquet"):
        path.unlink()

    problems = snapshots.verify(vintage.vintage_id)
    assert problems
    assert "missing from disk" in problems[0]


def test_an_intact_vintage_verifies_clean(store: BarStore, snapshots: SnapshotStore) -> None:
    week(store)
    vintage = snapshots.seal(resolutions=[Resolution.DAILY])
    assert snapshots.verify(vintage.vintage_id) == []


# ==========================================================================
# Admissibility and the flags
# ==========================================================================


def test_a_backtest_citing_an_unknown_vintage_is_not_admissible(
    snapshots: SnapshotStore,
) -> None:
    """The rule that gives the whole mechanism teeth.

    Without it, "we backtested this and it worked" cannot be checked, because
    the data it worked on has since been free to change.
    """
    ok, why = snapshots.is_admissible("vint_2026-03-04_deadbeef1234")
    assert not ok
    assert "not in the ledger" in why
    assert "cannot be reproduced" in why


def test_a_sealed_vintage_is_admissible(store: BarStore, snapshots: SnapshotStore) -> None:
    week(store)
    vintage = snapshots.seal(resolutions=[Resolution.DAILY])
    ok, why = snapshots.is_admissible(vintage.vintage_id)
    assert ok
    assert "sealed with 1 file" in why


def test_a_backfill_only_vintage_is_labelled_vendor_current_view(
    store: BarStore, snapshots: SnapshotStore
) -> None:
    """Still usable evidence — but calling it point-in-time would be a lie.

    Its knowledge times were assumed, not measured, so the as-of machinery is
    inert across the whole window.
    """
    week(store)
    vintage = snapshots.seal(resolutions=[Resolution.DAILY])
    assert vintage.pit_completeness_flag is PitCompleteness.VENDOR_CURRENT_VIEW
    assert not vintage.pit_completeness_flag.is_point_in_time
    assert vintage.first_live_observation_at is None
    # Still admissible: on free data this is the only option for a long window.
    assert vintage.admissible_for_promotion


def test_a_live_only_vintage_is_point_in_time(store: BarStore, snapshots: SnapshotStore) -> None:
    days = [s.day for s in CAL.sessions_between(date(2026, 3, 2), date(2026, 3, 6))]
    ingest(store, [daily(day, provenance=Provenance.LIVE) for day in days])
    vintage = snapshots.seal(resolutions=[Resolution.DAILY])
    assert vintage.pit_completeness_flag is PitCompleteness.POINT_IN_TIME
    assert vintage.first_live_observation_at is not None


def test_a_mixed_vintage_is_partial(store: BarStore, snapshots: SnapshotStore) -> None:
    """Read from the bars, not from a flag someone set.

    A vintage that *claims* to be point-in-time because a config said so is
    exactly the artefact this layer exists to prevent.
    """
    days = [s.day for s in CAL.sessions_between(date(2026, 3, 2), date(2026, 3, 6))]
    mixed = [
        daily(day, provenance=Provenance.LIVE if index > 2 else Provenance.BACKFILL)
        for index, day in enumerate(days)
    ]
    ingest(store, mixed)
    vintage = snapshots.seal(resolutions=[Resolution.DAILY])
    assert vintage.pit_completeness_flag is PitCompleteness.PARTIAL


def test_survivorship_is_stamped_on_the_vintage(
    ledger: Ledger, store: BarStore, snapshots: SnapshotStore
) -> None:
    week(store)
    vintage = snapshots.seal(resolutions=[Resolution.DAILY])
    assert vintage.survivorship_flag is SurvivorshipFlag.UNMEASURED
    assert "no universe snapshots exist" in vintage.survivorship_note


def test_the_universe_snapshot_is_pinned_when_one_exists(
    ledger: Ledger, store: BarStore, snapshots: SnapshotStore
) -> None:
    from tb.broker.port import Instrument

    universe = UniverseStore(ledger)
    candidates, _ = build_candidates(
        [
            Instrument(
                ticker="AAPL_US_EQ",
                instrument_type="STOCK",
                isin="US0378331005",
                currency_code="USD",
            )
        ],
        symbol_for={"AAPL_US_EQ": "AAPL"},
    )
    snapshot = select(candidates, max_symbols=25, taken_at=datetime(2026, 1, 1, tzinfo=UTC))
    universe.record(snapshot)

    week(store)
    vintage = snapshots.seal(resolutions=[Resolution.DAILY])
    assert vintage.universe_snapshot_id == snapshot.snapshot_id


def test_caveats_collect_everything_a_promotion_should_weigh(
    store: BarStore, snapshots: SnapshotStore
) -> None:
    """Assembled on the object so M5 cannot read one flag and miss the other."""
    week(store)
    vintage = snapshots.seal(resolutions=[Resolution.DAILY])
    caveats = vintage.caveats
    assert any("survivorship" in note for note in caveats)
    assert any("point-in-time" in note for note in caveats)


# ==========================================================================
# The reader handed to M3
# ==========================================================================


def test_the_reader_contains_only_this_vintages_files(
    store: BarStore, snapshots: SnapshotStore
) -> None:
    """Structural, not conventional.

    Handing M3 the store with a note saying "only read these files" is a
    convention, and a convention is broken by the first helper that takes a
    shortcut.
    """
    week(store)
    vintage = snapshots.seal(resolutions=[Resolution.DAILY])

    ingest(store, [daily(date(2026, 3, 9), close="101.00")])
    store.compact()

    source = snapshots.source_for(vintage.vintage_id)
    opens = {bar.bar_open_utc.date() for bar in source.bars}
    assert date(2026, 3, 9) not in opens


def test_the_reader_walks_forward_and_refuses_to_rewind(
    store: BarStore, snapshots: SnapshotStore
) -> None:
    week(store)
    vintage = snapshots.seal(resolutions=[Resolution.DAILY])
    reader = snapshots.reader_for(vintage.vintage_id, resolution=Resolution.DAILY)

    later = reader.advance_to(datetime(2026, 3, 6, tzinfo=UTC))
    assert len(later.bars(UID)) >= 3
    with pytest.raises(LookaheadError, match="cannot rewind"):
        reader.advance_to(datetime(2026, 3, 3, tzinfo=UTC))


def test_the_reader_shows_nothing_past_the_decision_time(
    store: BarStore, snapshots: SnapshotStore
) -> None:
    week(store)
    vintage = snapshots.seal(resolutions=[Resolution.DAILY])
    reader = snapshots.reader_for(vintage.vintage_id, resolution=Resolution.DAILY)

    window = reader.advance_to(datetime(2026, 3, 4, tzinfo=UTC))
    for bar in window.bars(UID):
        assert bar.available_at_utc <= window.as_of


def test_a_reader_for_an_unknown_vintage_raises(snapshots: SnapshotStore) -> None:
    with pytest.raises(SnapshotError, match="not in the ledger"):
        snapshots.reader_for("vint_nope")


def test_observed_delays_ignore_backfilled_bars() -> None:
    """Backfilled bars have a delay of zero by construction.

    Including them reports a delayed feed as instantaneous, which is precisely
    the number that makes it look live-capable.
    """
    moment = datetime(2026, 3, 4, 15, tzinfo=UTC)
    bars = [
        minute(moment, close="100.00", provenance=Provenance.BACKFILL),
        minute(
            moment + timedelta(minutes=1),
            close="100.01",
            provenance=Provenance.LIVE,
            delay_seconds=900,
        ),
    ]
    assert observed_delays(bars) == {"alpaca": 900.0}


# ==========================================================================
# The bake-off
# ==========================================================================


def test_disagreement_is_symmetric_in_its_arguments() -> None:
    """Against the midpoint, so which feed is "primary" cannot change the number."""
    forward = disagreement_bps(Decimal("100.00"), Decimal("100.10"))
    backward = disagreement_bps(Decimal("100.10"), Decimal("100.00"))
    assert forward == pytest.approx(backward)
    assert forward == pytest.approx(9.995, abs=0.01)


def test_percentiles_are_nearest_rank_not_interpolated() -> None:
    """An interpolated p95 invents a value no bar actually disagreed by.

    The number has to be traceable to an observation someone can go and look
    at, which is the whole reason the worst comparisons are kept.
    """
    values = [float(i) for i in range(1, 101)]
    assert percentile(values, 0.50) in {50.0, 51.0}
    assert percentile(values, 0.95) in {95.0, 96.0}
    assert percentile([], 0.95) is None


def test_only_matching_bar_periods_are_compared() -> None:
    """Comparing 15:59 to 15:45 measures fourteen minutes of price movement.

    On a trending name that produces a large, confident, meaningless number.
    """
    base = datetime(2026, 3, 4, 15, tzinfo=UTC)
    primary = [minute(base, close="100.00", provider="alpaca")]
    secondary = [minute(base + timedelta(minutes=14), close="140.00", provider="yahoo")]
    pairs, only_primary, only_secondary = compare_feeds(primary, secondary)
    assert pairs == ()
    assert (only_primary, only_secondary) == (1, 1)


def test_a_bar_present_on_one_side_only_is_counted_not_compared() -> None:
    """A minute with no IEX print is not a 0bps agreement.

    It is its own statistic, and on a 2%-of-volume feed it is a large one.
    """
    base = datetime(2026, 3, 4, 15, tzinfo=UTC)
    primary = [minute(base, close="100.00", provider="alpaca")]
    secondary = [
        minute(base, close="100.02", provider="yahoo"),
        minute(base + timedelta(minutes=1), close="100.03", provider="yahoo"),
    ]
    pairs, only_primary, only_secondary = compare_feeds(primary, secondary)
    assert len(pairs) == 1
    assert (only_primary, only_secondary) == (0, 1)


def test_extended_hours_bars_are_excluded_by_default() -> None:
    """One earnings-night print moves the p95 on its own."""
    base = datetime(2026, 3, 4, 23, tzinfo=UTC)
    primary = [minute(base, close="100.00", provider="alpaca", session=Session.EXTENDED)]
    secondary = [minute(base, close="130.00", provider="yahoo", session=Session.EXTENDED)]
    pairs, _, _ = compare_feeds(primary, secondary)
    assert pairs == ()

    included, _, _ = compare_feeds(primary, secondary, include_extended=True)
    assert len(included) == 1


def test_a_feed_cannot_be_baked_off_against_itself(ledger: Ledger) -> None:
    with pytest.raises(BakeoffError, match="against itself"):
        Bakeoff(ledger).run(
            primary_bars=[],
            secondary_bars=[],
            resolution=Resolution.MINUTE,
            primary="alpaca",
            secondary="alpaca",
        )


def test_too_few_bars_is_insufficient_not_a_pass(ledger: Ledger) -> None:
    """Absence of evidence is not a pass, or the gate is decoration."""
    base = datetime(2026, 3, 4, 15, tzinfo=UTC)
    primary = [minute(base + timedelta(minutes=i), close="100.00") for i in range(10)]
    secondary = [
        minute(base + timedelta(minutes=i), close="100.01", provider="yahoo") for i in range(10)
    ]
    result = Bakeoff(ledger).run(
        primary_bars=primary,
        secondary_bars=secondary,
        resolution=Resolution.MINUTE,
        primary="alpaca",
        secondary="yahoo",
        emit_event=False,
    )
    assert result.verdict is Verdict.INSUFFICIENT_SAMPLE
    assert not result.verdict.permits_live_resolution
    assert str(MIN_COMPARABLE_BARS) in result.rationale


def test_the_verdict_is_arithmetic_not_an_opinion(ledger: Ledger) -> None:
    """The headline number, and what it is compared against.

    At a 12bps p95 and a required 3x ratio, a strategy needs 36bps gross —
    against a 30bps round trip. That subtraction is the paid-data decision, and
    it is arithmetic rather than judgement.
    """
    base = datetime(2026, 3, 4, 14, 30, tzinfo=UTC)
    primary = [
        minute(base + timedelta(minutes=i), close="100.00") for i in range(MIN_COMPARABLE_BARS)
    ]
    # A uniform 12bps disagreement: 100.00 against 100.12.
    secondary = [
        minute(base + timedelta(minutes=i), close="100.12", provider="yahoo")
        for i in range(MIN_COMPARABLE_BARS)
    ]
    result = Bakeoff(ledger, min_edge_to_noise_ratio=3.0).run(
        primary_bars=primary,
        secondary_bars=secondary,
        resolution=Resolution.MINUTE,
        primary="alpaca",
        secondary="yahoo",
        emit_event=False,
    )
    assert result.p95_bps == pytest.approx(11.99, abs=0.05)
    assert result.minimum_viable_edge_bps == pytest.approx(35.96, abs=0.2)
    assert result.verdict is Verdict.PAID_DATA_REQUIRED
    assert "is not a business" in result.rationale
    assert str(ROUND_TRIP_COST_BPS) in result.rationale


def test_a_tight_feed_passes(ledger: Ledger) -> None:
    base = datetime(2026, 3, 4, 14, 30, tzinfo=UTC)
    primary = [
        minute(base + timedelta(minutes=i), close="100.00") for i in range(MIN_COMPARABLE_BARS)
    ]
    secondary = [
        minute(base + timedelta(minutes=i), close="100.01", provider="yahoo")
        for i in range(MIN_COMPARABLE_BARS)
    ]
    result = Bakeoff(ledger, min_edge_to_noise_ratio=3.0).run(
        primary_bars=primary,
        secondary_bars=secondary,
        resolution=Resolution.MINUTE,
        primary="alpaca",
        secondary="yahoo",
        emit_event=False,
    )
    assert result.verdict is Verdict.FREE_DATA_SUFFICIENT
    assert result.verdict.permits_live_resolution


def test_the_bakeoff_records_its_verdict(ledger: Ledger) -> None:
    base = datetime(2026, 3, 4, 14, 30, tzinfo=UTC)
    primary = [
        minute(base + timedelta(minutes=i), close="100.00") for i in range(MIN_COMPARABLE_BARS)
    ]
    secondary = [
        minute(base + timedelta(minutes=i), close="100.12", provider="yahoo")
        for i in range(MIN_COMPARABLE_BARS)
    ]
    bakeoff = Bakeoff(ledger)
    result = bakeoff.run(
        primary_bars=primary,
        secondary_bars=secondary,
        resolution=Resolution.MINUTE,
        primary="alpaca",
        secondary="yahoo",
    )
    events = list(ledger.iter_events(event_type=EventType.DATA_BAKEOFF_COMPLETED))
    assert len(events) == 1

    latest = bakeoff.latest(Resolution.MINUTE)
    assert latest is not None
    assert latest["verdict"] == result.verdict.value
    assert bakeoff.latest(Resolution.DAILY) is None


def test_missing_bars_are_measured_against_the_calendar_not_the_better_feed(
    ledger: Ledger,
) -> None:
    """Otherwise the better-covered feed looks perfect by definition."""
    base = datetime(2026, 3, 4, 14, 30, tzinfo=UTC)
    primary = [minute(base + timedelta(minutes=i), close="100.00") for i in range(100)]
    secondary = [
        minute(base + timedelta(minutes=i), close="100.01", provider="yahoo") for i in range(390)
    ]
    result = Bakeoff(ledger).run(
        primary_bars=primary,
        secondary_bars=secondary,
        resolution=Resolution.MINUTE,
        primary="alpaca",
        secondary="yahoo",
        expected_bars=390,
        emit_event=False,
    )
    by_provider = {stat.provider: stat for stat in result.stats}
    assert by_provider["alpaca"].missing_fraction == pytest.approx(290 / 390)
    assert by_provider["yahoo"].missing_fraction == pytest.approx(0.0)


def test_the_staleness_coverage_is_measured_on_knowledge_time(ledger: Ledger) -> None:
    """Never on bar time: a delayed feed's timestamps look current."""
    base = datetime(2026, 3, 4, 14, 30, tzinfo=UTC)
    fresh = [
        minute(
            base + timedelta(minutes=i),
            close="100.00",
            provenance=Provenance.LIVE,
            delay_seconds=10,
        )
        for i in range(50)
    ]
    delayed = [
        minute(
            base + timedelta(minutes=50 + i),
            close="100.00",
            provenance=Provenance.LIVE,
            delay_seconds=900,
        )
        for i in range(50)
    ]
    result = Bakeoff(ledger, max_delay_seconds=180).run(
        primary_bars=fresh + delayed,
        secondary_bars=[],
        resolution=Resolution.MINUTE,
        primary="alpaca",
        secondary="yahoo",
        emit_event=False,
    )
    assert result.cycles_meeting_staleness_pct == pytest.approx(50.0)


def test_delay_p95_ignores_backfilled_bars() -> None:
    base = datetime(2026, 3, 4, 15, tzinfo=UTC)
    bars = [
        minute(base, close="100.00", provenance=Provenance.BACKFILL),
        minute(
            base + timedelta(minutes=1),
            close="100.00",
            provenance=Provenance.LIVE,
            delay_seconds=900,
        ),
    ]
    assert delay_p95(bars) == pytest.approx(900.0)
    assert delay_p95([bars[0]]) is None


def test_widening_a_live_resolution_needs_evidence_and_stays_manual(
    ledger: Ledger,
) -> None:
    """The measurement says whether the edit is justified; a human makes it.

    A passing measurement that silently widened a hard limit would be the
    control layer editing itself, which is the one thing the limits file exists
    to prevent.
    """
    base = datetime(2026, 3, 4, 14, 30, tzinfo=UTC)
    primary = [
        minute(base + timedelta(minutes=i), close="100.00") for i in range(MIN_COMPARABLE_BARS)
    ]
    wide = [
        minute(base + timedelta(minutes=i), close="100.12", provider="yahoo")
        for i in range(MIN_COMPARABLE_BARS)
    ]
    tight = [
        minute(base + timedelta(minutes=i), close="100.01", provider="yahoo")
        for i in range(MIN_COMPARABLE_BARS)
    ]
    bakeoff = Bakeoff(ledger)

    failing = bakeoff.run(
        primary_bars=primary,
        secondary_bars=wide,
        resolution=Resolution.MINUTE,
        primary="alpaca",
        secondary="yahoo",
        emit_event=False,
    )
    allowed, why = may_widen_live_resolutions(failing, currently_allowed=["daily"])
    assert not allowed
    assert "paid_data_required" in why

    passing = bakeoff.run(
        primary_bars=primary,
        secondary_bars=tight,
        resolution=Resolution.MINUTE,
        primary="alpaca",
        secondary="yahoo",
        emit_event=False,
    )
    allowed, why = may_widen_live_resolutions(passing, currently_allowed=["daily"])
    assert allowed
    assert "does not widen anything by itself" in why


def test_an_already_permitted_resolution_needs_no_evidence(ledger: Ledger) -> None:
    result = Bakeoff(ledger).run(
        primary_bars=[],
        secondary_bars=[],
        resolution=Resolution.DAILY,
        primary="alpaca",
        secondary="yahoo",
        emit_event=False,
    )
    allowed, why = may_widen_live_resolutions(result, currently_allowed=["daily"])
    assert allowed
    assert "already permitted" in why


def test_the_worst_comparisons_are_kept_for_inspection(ledger: Ledger) -> None:
    """So the headline p95 can be traced to bars somebody can look at."""
    base = datetime(2026, 3, 4, 14, 30, tzinfo=UTC)
    primary = [
        minute(base + timedelta(minutes=i), close="100.00") for i in range(MIN_COMPARABLE_BARS)
    ]
    secondary = [
        minute(
            base + timedelta(minutes=i),
            close="130.00" if i == 7 else "100.01",
            provider="yahoo",
        )
        for i in range(MIN_COMPARABLE_BARS)
    ]
    result = Bakeoff(ledger).run(
        primary_bars=primary,
        secondary_bars=secondary,
        resolution=Resolution.MINUTE,
        primary="alpaca",
        secondary="yahoo",
        emit_event=False,
    )
    assert result.worst
    assert result.worst[0].bar_open_utc == base + timedelta(minutes=7)


def test_a_non_positive_price_in_a_comparison_is_refused() -> None:
    with pytest.raises(BakeoffError, match="non-positive"):
        disagreement_bps(Decimal("0"), Decimal("100.00"))
