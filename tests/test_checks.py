"""Integrity checks, and the audit that runs them.

The tests that matter most are the ones about *classification*. A check that
reports "4,312 gaps" is a check nobody reads, and a check nobody reads is worse
than no check because its silence is mistaken for health. So the assertions
here are mostly about a gap on a half-day being called a half-day, a thin
name's missing minutes being called thin, and only the genuinely unexplained
ones counting against the feed.

`test_a_frozen_feed_is_caught_where_per_symbol_staleness_misses_it` is the
other one worth reading: it is the failure mode that per-instrument checks
structurally cannot see.
"""

from __future__ import annotations

from datetime import UTC, date, datetime, timedelta
from decimal import Decimal

import pytest

from tb.broker.reconcile import Severity
from tb.data.actions import ActionStore
from tb.data.adjustments import ActionType, CorporateAction
from tb.data.audit import DataAuditor, summarise_gaps
from tb.data.barstore import BarStore
from tb.data.calendar import TradingCalendar
from tb.data.checks import (
    CheckKind,
    GapCause,
    check_coverage,
    check_frozen_feed,
    check_intrabar_range,
    check_jumps,
    check_knowledge_times,
    check_ordering,
    check_provenance,
    check_revision_clusters,
    check_stale_repeats,
    check_zero_volume_sessions,
    classify_gap,
)
from tb.data.provider import Bar, BarBatch, Provenance, Resolution, Session
from tb.ledger.events import EventType
from tb.ledger.store import Ledger

UID = "isin:US0378331005"
THIN_UID = "isin:US0000000001"
CAL = TradingCalendar()


def daily(
    day: date,
    *,
    close: str = "100.00",
    open_: str | None = None,
    volume: int | None = 1_000_000,
    provider: str = "alpaca",
    provenance: Provenance = Provenance.BACKFILL,
    uid: str = UID,
    ingested: datetime | None = None,
) -> Bar:
    bar_open = datetime(day.year, day.month, day.day, tzinfo=UTC)
    closing = Decimal(close)
    opening = Decimal(open_ if open_ is not None else close)
    return Bar(
        instrument_uid=uid,
        resolution=Resolution.DAILY,
        bar_open_utc=bar_open,
        available_at_utc=bar_open + timedelta(days=1),
        ingested_at_utc=ingested or (bar_open + timedelta(days=1)),
        provider=provider,
        provenance=provenance,
        session=Session.REGULAR,
        open=opening,
        high=max(opening, closing),
        low=min(opening, closing),
        close=closing,
        volume=volume,
    )


def minute(
    moment: datetime,
    *,
    close: str = "100.00",
    volume: int | None = 1000,
    uid: str = UID,
) -> Bar:
    price = Decimal(close)
    return Bar(
        instrument_uid=uid,
        resolution=Resolution.MINUTE,
        bar_open_utc=moment,
        available_at_utc=moment + timedelta(minutes=1),
        ingested_at_utc=moment + timedelta(minutes=1),
        provider="alpaca",
        provenance=Provenance.LIVE,
        session=Session.REGULAR,
        open=price,
        high=price,
        low=price,
        close=price,
        volume=volume,
    )


# --------------------------------------------------------------------------
# Gap classification — the point of the whole module
# --------------------------------------------------------------------------


def test_a_weekend_gap_is_not_a_feed_problem() -> None:
    cause = classify_gap(
        datetime(2026, 3, 7, 15, tzinfo=UTC), calendar=CAL, resolution=Resolution.DAILY
    )
    assert cause is GapCause.MARKET_CLOSED
    assert not cause.counts_against_feed


def test_a_holiday_gap_is_not_a_feed_problem() -> None:
    cause = classify_gap(
        datetime(2026, 12, 25, 15, tzinfo=UTC), calendar=CAL, resolution=Resolution.DAILY
    )
    assert cause is GapCause.MARKET_CLOSED


def test_a_half_day_afternoon_is_a_half_day_not_a_gap() -> None:
    """Without this, every early close reads as 180 missing minutes.

    The unexplained-gap percentage would then be dominated by days when
    nothing was wrong, and the metric would be useless for its actual job.
    """
    # 15:00 Eastern on the day after Thanksgiving 2026, which closes at 13:00.
    cause = classify_gap(
        datetime(2026, 11, 27, 20, tzinfo=UTC), calendar=CAL, resolution=Resolution.MINUTE
    )
    assert cause is GapCause.HALF_DAY
    assert not cause.counts_against_feed


def test_a_thin_name_missing_a_minute_is_expected() -> None:
    """IEX is ~2% of consolidated volume, so small-caps genuinely have holes.

    Counting those against the feed would make the metric unusable across the
    universe, and the only available fix would be loosening the threshold for
    everyone — including the megacaps where a hole is a real failure.
    """
    mid_session = datetime(2026, 3, 4, 16, tzinfo=UTC)
    assert (
        classify_gap(mid_session, calendar=CAL, resolution=Resolution.MINUTE, thin_name=True)
        is GapCause.THIN_NAME_NO_PRINT
    )
    assert (
        classify_gap(mid_session, calendar=CAL, resolution=Resolution.MINUTE, thin_name=False)
        is GapCause.UNEXPLAINED
    )


def test_a_thin_name_exemption_does_not_apply_to_daily_bars() -> None:
    """Every listed name prints *something* in a whole session.

    The no-print exemption is about minutes, and extending it to daily bars
    would excuse a genuinely missing session.
    """
    assert (
        classify_gap(
            datetime(2026, 3, 4, tzinfo=UTC),
            calendar=CAL,
            resolution=Resolution.DAILY,
            thin_name=True,
        )
        is GapCause.UNEXPLAINED
    )


def test_a_known_halt_is_explained() -> None:
    cause = classify_gap(
        datetime(2026, 3, 4, 16, tzinfo=UTC),
        calendar=CAL,
        resolution=Resolution.MINUTE,
        halted_dates=[date(2026, 3, 4)],
    )
    assert cause is GapCause.KNOWN_HALT


def test_a_date_outside_the_calendar_is_not_reported_as_a_closure() -> None:
    """ "We do not know" and "the market was shut" are different claims.

    Only one of them needs someone to go and extend a list, so conflating them
    would hide the maintenance task behind a reassuring label.
    """
    cause = classify_gap(
        datetime(2031, 6, 3, 15, tzinfo=UTC), calendar=CAL, resolution=Resolution.DAILY
    )
    assert cause is GapCause.OUTSIDE_COVERAGE
    assert not cause.counts_against_feed


# --------------------------------------------------------------------------
# Coverage
# --------------------------------------------------------------------------


def test_full_coverage_over_trading_days_reports_nothing() -> None:
    """The denominator is sessions, not calendar days.

    Counting calendar days would report both weekend days as missing and the
    real gaps would vanish into the noise.
    """
    days = [session.day for session in CAL.sessions_between(date(2026, 3, 2), date(2026, 3, 6))]
    bars = [daily(day) for day in days]
    report, findings = check_coverage(
        bars,
        instrument_uid=UID,
        resolution=Resolution.DAILY,
        start=date(2026, 3, 2),
        end=date(2026, 3, 8),
        calendar=CAL,
        max_unexplained_pct=1.0,
    )
    assert report.expected == 5
    assert report.present == 5
    assert findings == ()


def test_a_missing_session_beyond_the_bound_blocks() -> None:
    """A feature over a window with holes is a different and wrong number."""
    days = [session.day for session in CAL.sessions_between(date(2026, 3, 2), date(2026, 3, 6))]
    bars = [daily(day) for day in days if day != date(2026, 3, 4)]
    report, findings = check_coverage(
        bars,
        instrument_uid=UID,
        resolution=Resolution.DAILY,
        start=date(2026, 3, 2),
        end=date(2026, 3, 6),
        calendar=CAL,
        max_unexplained_pct=1.0,
    )
    assert report.unexplained == 1
    assert report.unexplained_pct == pytest.approx(20.0)
    assert len(findings) == 1
    assert findings[0].severity is Severity.BLOCKING
    assert findings[0].cause is GapCause.UNEXPLAINED


def test_explained_gaps_do_not_count_against_the_bound() -> None:
    """The whole design, as one assertion.

    A thin name missing most of its minutes must not fail a coverage check it
    has no way to pass — while a megacap missing the same minutes must.
    """
    session = CAL.classify(date(2026, 3, 4))
    assert session.open_utc is not None
    # Only the first ten minutes of the session printed.
    bars = [minute(session.open_utc + timedelta(minutes=i), uid=THIN_UID) for i in range(10)]

    thin, thin_findings = check_coverage(
        bars,
        instrument_uid=THIN_UID,
        resolution=Resolution.MINUTE,
        start=date(2026, 3, 4),
        end=date(2026, 3, 4),
        calendar=CAL,
        max_unexplained_pct=1.0,
        thin_name=True,
    )
    assert thin.unexplained == 0
    assert thin.explained == 380
    assert thin_findings == ()

    liquid, liquid_findings = check_coverage(
        bars,
        instrument_uid=UID,
        resolution=Resolution.MINUTE,
        start=date(2026, 3, 4),
        end=date(2026, 3, 4),
        calendar=CAL,
        max_unexplained_pct=1.0,
        thin_name=False,
    )
    assert liquid.unexplained == 380
    assert liquid_findings[0].severity is Severity.BLOCKING


def test_a_half_day_does_not_produce_a_coverage_finding() -> None:
    """The 210 minutes that exist are all of them, not 210 out of 390."""
    session = CAL.classify(date(2026, 11, 27))
    assert session.open_utc is not None
    assert session.expected_minute_bars == 210
    bars = [minute(session.open_utc + timedelta(minutes=i)) for i in range(210)]
    report, findings = check_coverage(
        bars,
        instrument_uid=UID,
        resolution=Resolution.MINUTE,
        start=date(2026, 11, 27),
        end=date(2026, 11, 27),
        calendar=CAL,
        max_unexplained_pct=1.0,
    )
    assert report.expected == 210
    assert report.unexplained == 0
    assert findings == ()


def test_gaps_within_the_bound_are_reported_at_info() -> None:
    days = [session.day for session in CAL.sessions_between(date(2026, 1, 2), date(2026, 6, 30))]
    bars = [daily(day) for day in days if day != days[10]]
    report, findings = check_coverage(
        bars,
        instrument_uid=UID,
        resolution=Resolution.DAILY,
        start=date(2026, 1, 2),
        end=date(2026, 6, 30),
        calendar=CAL,
        max_unexplained_pct=1.0,
    )
    assert 0 < report.unexplained_pct < 1.0
    assert findings[0].severity is Severity.INFO


# --------------------------------------------------------------------------
# Series checks
# --------------------------------------------------------------------------


def test_out_of_order_bars_block() -> None:
    bars = [daily(date(2026, 3, 4)), daily(date(2026, 3, 3))]
    (finding,) = check_ordering(bars, instrument_uid=UID, resolution=Resolution.DAILY)
    assert finding.kind is CheckKind.OUT_OF_ORDER
    assert finding.severity is Severity.BLOCKING


def test_a_duplicated_vintage_blocks() -> None:
    """Doubling a bar doubles its weight in every feature over the window."""
    bar = daily(date(2026, 3, 4))
    (finding,) = check_ordering([bar, bar], instrument_uid=UID, resolution=Resolution.DAILY)
    assert finding.kind is CheckKind.DUPLICATE_BAR


def test_two_vintages_of_one_bar_are_not_a_duplicate() -> None:
    """A revision and its original are both legitimate rows."""
    original = daily(date(2026, 3, 4), ingested=datetime(2026, 3, 5, tzinfo=UTC))
    revised = daily(date(2026, 3, 4), close="101.00", ingested=datetime(2026, 3, 20, tzinfo=UTC))
    assert (
        check_ordering([original, revised], instrument_uid=UID, resolution=Resolution.DAILY) == ()
    )


def test_stale_repeats_are_caught() -> None:
    """A flat market still prints varying volume."""
    start = datetime(2026, 3, 4, 15, tzinfo=UTC)
    bars = [minute(start + timedelta(minutes=i), close="100.00") for i in range(20)]
    (finding,) = check_stale_repeats(
        bars, instrument_uid=UID, resolution=Resolution.MINUTE, max_run=12
    )
    assert finding.kind is CheckKind.STALE_REPEAT
    assert "repeating its last value" in finding.detail


def test_a_short_flat_run_is_not_a_stale_repeat() -> None:
    start = datetime(2026, 3, 4, 15, tzinfo=UTC)
    bars = [minute(start + timedelta(minutes=i), close="100.00") for i in range(4)]
    assert (
        check_stale_repeats(bars, instrument_uid=UID, resolution=Resolution.MINUTE, max_run=12)
        == ()
    )


def test_varying_volume_breaks_a_repeat_run() -> None:
    """Identical OHLC with moving volume is a quiet market, not an echo."""
    start = datetime(2026, 3, 4, 15, tzinfo=UTC)
    bars = [minute(start + timedelta(minutes=i), close="100.00", volume=100 + i) for i in range(20)]
    assert (
        check_stale_repeats(bars, instrument_uid=UID, resolution=Resolution.MINUTE, max_run=12)
        == ()
    )


def test_an_implausible_intrabar_range_is_flagged() -> None:
    bar = Bar(
        instrument_uid=UID,
        resolution=Resolution.MINUTE,
        bar_open_utc=datetime(2026, 3, 4, 15, tzinfo=UTC),
        available_at_utc=datetime(2026, 3, 4, 15, 1, tzinfo=UTC),
        ingested_at_utc=datetime(2026, 3, 4, 15, 1, tzinfo=UTC),
        provider="alpaca",
        provenance=Provenance.LIVE,
        session=Session.REGULAR,
        open=Decimal("100.00"),
        high=Decimal("200.00"),
        low=Decimal("99.00"),
        close=Decimal("101.00"),
        volume=1000,
    )
    (finding,) = check_intrabar_range([bar], instrument_uid=UID, resolution=Resolution.MINUTE)
    assert finding.kind is CheckKind.IMPLAUSIBLE_RANGE


def test_mostly_zero_volume_is_flagged() -> None:
    start = datetime(2026, 3, 4, 15, tzinfo=UTC)
    bars = [
        minute(start + timedelta(minutes=i), close="100.00", volume=0 if i % 4 else 500)
        for i in range(40)
    ]
    (finding,) = check_zero_volume_sessions(bars, instrument_uid=UID, resolution=Resolution.MINUTE)
    assert finding.kind is CheckKind.ZERO_VOLUME


def test_unreported_volume_is_not_counted_as_zero() -> None:
    """None means unmeasured — spot FX, an index — and zero means measured.

    Treating None as zero would flag every FX series as barely trading.
    """
    start = datetime(2026, 3, 4, 15, tzinfo=UTC)
    bars = [minute(start + timedelta(minutes=i), volume=None) for i in range(40)]
    assert check_zero_volume_sessions(bars, instrument_uid=UID, resolution=Resolution.MINUTE) == ()


def test_a_backfill_only_series_is_labelled() -> None:
    """The as-of machinery is inert across it, and the vintage must say so."""
    bars = [daily(date(2026, 3, 3)), daily(date(2026, 3, 4))]
    (finding,) = check_provenance(bars, instrument_uid=UID, resolution=Resolution.DAILY)
    assert finding.kind is CheckKind.BACKFILL_ONLY
    assert finding.severity is Severity.INFO
    assert "vendor-current-view" in finding.detail


def test_one_live_bar_is_enough_to_stop_the_backfill_label() -> None:
    bars = [
        daily(date(2026, 3, 3)),
        daily(date(2026, 3, 4), provenance=Provenance.LIVE),
    ]
    assert check_provenance(bars, instrument_uid=UID, resolution=Resolution.DAILY) == ()


def test_a_bar_knowable_before_it_closed_blocks() -> None:
    """`Bar.__post_init__` refuses to build one, so this catches other paths.

    A hand-edited Parquet file or a future writer is exactly the case the
    constructor cannot see, and a lookahead channel is not a warning.
    """
    good = daily(date(2026, 3, 4))
    # Bypass the constructor the way a bad migration would.
    broken = object.__new__(Bar)
    for name, value in (
        ("instrument_uid", UID),
        ("resolution", Resolution.DAILY),
        ("bar_open_utc", good.bar_open_utc),
        ("available_at_utc", good.bar_open_utc),
        ("ingested_at_utc", good.ingested_at_utc),
        ("provider", "alpaca"),
        ("provenance", Provenance.BACKFILL),
        ("session", Session.REGULAR),
        ("open", good.open),
        ("high", good.high),
        ("low", good.low),
        ("close", good.close),
        ("volume", good.volume),
        ("currency", None),
    ):
        object.__setattr__(broken, name, value)

    (finding,) = check_knowledge_times([broken], instrument_uid=UID, resolution=Resolution.DAILY)
    assert finding.kind is CheckKind.KNOWLEDGE_TIME_INVALID
    assert finding.severity is Severity.BLOCKING
    assert "do not adjust the row" in (finding.suggested_action or "")


# --------------------------------------------------------------------------
# Jumps
# --------------------------------------------------------------------------


def test_a_jump_explained_by_a_recorded_action_is_info() -> None:
    split = CorporateAction(
        action_id="act_split",
        instrument_uid=UID,
        action_type=ActionType.SPLIT,
        effective_date=date(2026, 3, 4),
        known_at_utc=datetime(2026, 2, 1, tzinfo=UTC),
        source_provider="alpaca",
        ratio_num=4,
        ratio_den=1,
    )
    bars = [daily(date(2026, 3, 3), close="400.00"), daily(date(2026, 3, 4), close="100.00")]
    findings, suspicions = check_jumps(
        bars, instrument_uid=UID, resolution=Resolution.DAILY, known_actions=[split]
    )
    assert suspicions == ()
    assert findings[0].kind is CheckKind.EXPLAINED_JUMP
    assert findings[0].severity is Severity.INFO


def test_an_unexplained_jump_fitting_a_split_ratio_blocks() -> None:
    bars = [daily(date(2026, 3, 3), close="400.00"), daily(date(2026, 3, 4), close="100.00")]
    findings, suspicions = check_jumps(bars, instrument_uid=UID, resolution=Resolution.DAILY)
    assert len(suspicions) == 1
    assert suspicions[0].implied_ratio.numerator == 4
    assert findings[0].severity is Severity.BLOCKING
    assert "do not adjust prices by the inferred ratio" in (findings[0].suggested_action or "")


def test_a_large_move_fitting_no_ratio_is_a_warning_not_a_block() -> None:
    """Most likely a real move, but worth confirming against a second feed."""
    bars = [daily(date(2026, 3, 3), close="100.00"), daily(date(2026, 3, 4), close="58.00")]
    findings, suspicions = check_jumps(bars, instrument_uid=UID, resolution=Resolution.DAILY)
    assert suspicions == ()
    assert findings[0].severity is Severity.WARN


def test_an_ordinary_session_produces_no_jump_finding() -> None:
    bars = [daily(date(2026, 3, 3), close="100.00"), daily(date(2026, 3, 4), close="103.00")]
    findings, _ = check_jumps(bars, instrument_uid=UID, resolution=Resolution.DAILY)
    assert findings == ()


# --------------------------------------------------------------------------
# The frozen feed
# --------------------------------------------------------------------------


def test_a_frozen_feed_is_caught_where_per_symbol_staleness_misses_it() -> None:
    """Each symbol alone looks like one unremarkable stale name.

    All of them stale at the *same* instant means the provider stopped
    publishing, which is a different fact with a different response — and no
    per-instrument check can see it.
    """
    frozen_at = datetime(2026, 3, 4, 15, tzinfo=UTC)
    latest = {
        f"isin:US000000000{i}": minute(frozen_at - timedelta(minutes=1), uid=f"isin:US000000000{i}")
        for i in range(5)
    }
    (finding,) = check_frozen_feed(latest, now=frozen_at + timedelta(minutes=30), limit_seconds=180)
    assert finding.kind is CheckKind.FROZEN_FEED
    assert finding.severity is Severity.BLOCKING
    assert "provider having stopped" in finding.detail
    assert "exits are unaffected" in (finding.suggested_action or "")


def test_one_stale_symbol_among_fresh_ones_is_not_a_frozen_feed() -> None:
    now = datetime(2026, 3, 4, 16, tzinfo=UTC)
    latest = {
        "isin:US0000000001": minute(now - timedelta(minutes=1), uid="isin:US0000000001"),
        "isin:US0000000002": minute(now - timedelta(minutes=1), uid="isin:US0000000002"),
        "isin:US0000000003": minute(now - timedelta(hours=2), uid="isin:US0000000003"),
    }
    assert check_frozen_feed(latest, now=now, limit_seconds=180) == ()


def test_stale_at_different_instants_is_not_a_frozen_feed() -> None:
    """Staggered staleness is a set of data gaps, not one provider outage."""
    now = datetime(2026, 3, 4, 18, tzinfo=UTC)
    latest = {
        f"isin:US000000000{i}": minute(now - timedelta(hours=1 + i), uid=f"isin:US000000000{i}")
        for i in range(4)
    }
    assert check_frozen_feed(latest, now=now, limit_seconds=180) == ()


def test_too_few_symbols_cannot_establish_a_frozen_feed() -> None:
    now = datetime(2026, 3, 4, 18, tzinfo=UTC)
    latest = {"isin:US0000000001": minute(now - timedelta(hours=2))}
    assert check_frozen_feed(latest, now=now, limit_seconds=180) == ()


# --------------------------------------------------------------------------
# Revision clusters
# --------------------------------------------------------------------------


def test_a_revision_cluster_is_flagged() -> None:
    """A trickle is expected; a cluster means a span was rewritten wholesale."""
    base = datetime(2026, 3, 4, tzinfo=UTC)
    rows = [
        {"instrument_uid": UID, "revised_at": (base + timedelta(minutes=i)).isoformat()}
        for i in range(30)
    ]
    (finding,) = check_revision_clusters(rows, min_cluster=20)
    assert finding.kind is CheckKind.REVISION_CLUSTER
    assert "no longer exists" in finding.detail


def test_a_trickle_of_revisions_is_not_a_cluster() -> None:
    base = datetime(2026, 1, 1, tzinfo=UTC)
    rows = [
        {"instrument_uid": UID, "revised_at": (base + timedelta(days=i)).isoformat()}
        for i in range(30)
    ]
    assert check_revision_clusters(rows, min_cluster=20) == ()


def test_unparseable_revision_rows_are_skipped_not_fatal() -> None:
    rows: list[dict[str, object]] = [
        {"instrument_uid": UID, "revised_at": "not-a-timestamp"},
        {"instrument_uid": "", "revised_at": None},
    ]
    assert check_revision_clusters(rows) == ()


# --------------------------------------------------------------------------
# The orchestrator
# --------------------------------------------------------------------------


@pytest.fixture
def store(ledger: Ledger, tmp_path: object) -> BarStore:
    return BarStore(ledger, root=tmp_path / "bars")  # type: ignore[operator]


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


def test_a_clean_store_audits_clean(ledger: Ledger, store: BarStore) -> None:
    days = [session.day for session in CAL.sessions_between(date(2026, 3, 2), date(2026, 3, 6))]
    ingest(store, [daily(day) for day in days])

    auditor = DataAuditor(ledger, store, calendar=CAL)
    report = auditor.run(resolutions=[Resolution.DAILY])
    assert report.clean
    assert report.n_bars_checked == 5
    assert report.gaps_unexplained == 0


def test_the_audit_emits_its_event_with_blocking_findings_first(
    ledger: Ledger, store: BarStore
) -> None:
    """A truncated sample must never drop the findings that stop trading."""
    days = [session.day for session in CAL.sessions_between(date(2026, 3, 2), date(2026, 3, 6))]
    ingest(store, [daily(day) for day in days if day != date(2026, 3, 4)])

    auditor = DataAuditor(ledger, store, calendar=CAL)
    report = auditor.run(resolutions=[Resolution.DAILY])
    assert not report.clean

    events = list(ledger.iter_events(event_type=EventType.DATA_AUDIT_COMPLETED))
    assert len(events) == 1
    import json

    payload = json.loads(str(events[0]["payload_json"]))
    assert payload["n_blocking"] >= 1
    assert payload["findings"][0]["severity"] == "blocking"
    assert payload["gaps_unexplained"] >= 1


def test_the_audit_rebuilds_bar_coverage_rather_than_incrementing(
    ledger: Ledger, store: BarStore
) -> None:
    """A counter that drifts from the files it summarises gets trusted anyway."""
    days = [session.day for session in CAL.sessions_between(date(2026, 3, 2), date(2026, 3, 6))]
    ingest(store, [daily(day) for day in days])

    auditor = DataAuditor(ledger, store, calendar=CAL)
    auditor.run(resolutions=[Resolution.DAILY], emit_event=False)
    auditor.run(resolutions=[Resolution.DAILY], emit_event=False)

    rows = auditor.coverage_rows()
    assert len(rows) == 1
    assert rows[0]["row_count"] == 5
    assert auditor.last_audit_at() is not None


def test_the_audit_uses_the_stores_own_span_by_default(ledger: Ledger, store: BarStore) -> None:
    """Auditing a window nobody asked for reports it all as missing.

    True, and completely uninformative about the feed.
    """
    ingest(store, [daily(date(2026, 3, 4)), daily(date(2026, 3, 5))])
    auditor = DataAuditor(ledger, store, calendar=CAL)
    report = auditor.run(resolutions=[Resolution.DAILY], emit_event=False)
    assert report.coverage[0].expected == 2
    assert report.clean


def test_the_audit_surfaces_a_split_suspicion(ledger: Ledger, store: BarStore) -> None:
    ingest(
        store,
        [
            daily(date(2026, 3, 3), close="400.00"),
            daily(date(2026, 3, 4), close="100.00"),
        ],
    )
    auditor = DataAuditor(ledger, store, calendar=CAL, actions=ActionStore(ledger))
    report = auditor.run(resolutions=[Resolution.DAILY], emit_event=False)
    assert len(report.suspicions) == 1
    assert not report.clean


def test_a_recorded_action_removes_the_suspicion(ledger: Ledger, store: BarStore) -> None:
    actions = ActionStore(ledger)
    actions.record(
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
    ingest(
        store,
        [
            daily(date(2026, 3, 3), close="400.00"),
            daily(date(2026, 3, 4), close="100.00"),
        ],
    )
    auditor = DataAuditor(ledger, store, calendar=CAL, actions=actions)
    report = auditor.run(resolutions=[Resolution.DAILY], emit_event=False)
    assert report.suspicions == []
    assert any(f.kind is CheckKind.EXPLAINED_JUMP for f in report.findings)


def test_warnings_alone_do_not_make_an_audit_dirty(ledger: Ledger, store: BarStore) -> None:
    """An audit that cries wolf over a thin name's flat minutes gets ignored.

    And then the blocking findings get ignored with it, which is the actual
    risk of an over-sensitive report.
    """
    days = [session.day for session in CAL.sessions_between(date(2026, 3, 2), date(2026, 3, 6))]
    ingest(store, [daily(day, volume=0, close="100.00") for day in days])
    auditor = DataAuditor(ledger, store, calendar=CAL)
    report = auditor.run(resolutions=[Resolution.DAILY], emit_event=False)
    assert report.clean


def test_a_missing_partition_file_is_a_blocking_finding(ledger: Ledger, store: BarStore) -> None:
    """The ledger named a file the store cannot produce.

    Mirrors `tb ledger verify`: the catalog defines the dataset, so a file it
    promises and cannot deliver is a hard failure rather than a warning.
    """
    days = [session.day for session in CAL.sessions_between(date(2026, 3, 2), date(2026, 3, 6))]
    ingest(store, [daily(day) for day in days])
    store.compact()

    for path in store.root.rglob("*.parquet"):
        path.unlink()

    auditor = DataAuditor(ledger, store, calendar=CAL)
    findings = auditor._check_partitions()
    assert findings
    assert findings[0].severity is Severity.BLOCKING
    assert findings[0].kind is CheckKind.PARTITION_MISSING


def test_gap_totals_are_reported_per_cause(ledger: Ledger, store: BarStore) -> None:
    """The total is the number that gets ignored; only unexplained is a claim."""
    days = [session.day for session in CAL.sessions_between(date(2026, 3, 2), date(2026, 3, 6))]
    ingest(store, [daily(day) for day in days if day != date(2026, 3, 4)])
    auditor = DataAuditor(ledger, store, calendar=CAL)
    report = auditor.run(resolutions=[Resolution.DAILY], emit_event=False)
    totals = summarise_gaps(report.coverage)
    assert totals.get(GapCause.UNEXPLAINED.value) == 1


def test_an_empty_store_audits_without_error(ledger: Ledger, store: BarStore) -> None:
    auditor = DataAuditor(ledger, store, calendar=CAL)
    report = auditor.run(resolutions=[Resolution.DAILY], emit_event=False)
    assert report.clean
    assert report.n_instruments == 0
