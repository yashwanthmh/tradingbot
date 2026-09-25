"""The properties M2 claims, and the two bugs an end-to-end run found.

The property tests here are the ones the milestone's verification list asks
for, and they are properties rather than cases because the cases that matter
are the ones nobody thinks to write: an ingest order somebody did not try, a
revision landing inside a sealed window, a DST boundary.

`test_two_providers_do_not_share_a_partition` and
`test_series_checks_run_per_provider` are regressions for bugs found by running
`tb data audit` against a store holding two feeds. Both were invisible to the
unit tests because every fixture used a single provider — and both were the
"wrong but plausible" shape: compaction merged Alpaca's and Yahoo's bars into
one partition labelled with one provider's name, which would have left the
cross-venue check comparing a feed against itself.
"""

from __future__ import annotations

from datetime import UTC, date, datetime, timedelta
from decimal import Decimal
from pathlib import Path

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from tb.data.asof import UNKNOWN, ForwardOnlyReader, InMemoryBarSource, LookaheadError, visible_bars
from tb.data.audit import DataAuditor
from tb.data.barstore import BarStore
from tb.data.calendar import TradingCalendar
from tb.data.checks import CheckKind, check_ordering
from tb.data.provider import (
    US_EASTERN,
    Bar,
    BarBatch,
    DataError,
    Provenance,
    Resolution,
    Session,
    TimestampConvention,
    classify_us_session,
    normalise_bar_open,
    provider_preference,
)
from tb.data.snapshot import SnapshotStore
from tb.ledger.store import Ledger

UID = "isin:US0378331005"
CAL = TradingCalendar()


def daily(
    day: date,
    *,
    close: str = "100.00",
    provider: str = "alpaca",
    ingested: datetime | None = None,
    provenance: Provenance = Provenance.BACKFILL,
) -> Bar:
    bar_open = datetime(day.year, day.month, day.day, tzinfo=UTC)
    price = Decimal(close)
    return Bar(
        instrument_uid=UID,
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


@pytest.fixture
def store(ledger: Ledger, tmp_path: Path) -> BarStore:
    return BarStore(ledger, root=tmp_path / "bars")


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


def sessions(start: date, end: date) -> list[date]:
    return [s.day for s in CAL.sessions_between(start, end)]


# ==========================================================================
# The two-provider bugs
# ==========================================================================


def test_two_providers_do_not_share_a_partition(ledger: Ledger, store: BarStore) -> None:
    """A regression, and a data-integrity one.

    Compaction detected an *overlapping* partition without checking the
    provider, so sealing Yahoo's bars superseded Alpaca's and merged both feeds
    into one file labelled `provider='yahoo'`. The catalog then lied about where
    its rows came from — and the cross-venue price check, the only defence
    against a mismapped ticker, would have been comparing a feed against
    itself.
    """
    days = sessions(date(2026, 3, 2), date(2026, 3, 6))
    ingest(store, [daily(day, provider="alpaca") for day in days], provider="alpaca")
    ingest(
        store,
        [daily(day, close="100.02", provider="yahoo") for day in days],
        provider="yahoo",
    )
    store.compact()

    live = store.live_partitions(resolution=Resolution.DAILY)
    assert len(live) == 2
    assert {info.provider for info in live} == {"alpaca", "yahoo"}
    assert all(info.row_count == 5 for info in live)

    for info in live:
        bars = store.read_partition(info.relative_path)
        assert {bar.provider for bar in bars} == {info.provider}


def test_sealing_a_mixed_provider_partition_is_refused(store: BarStore) -> None:
    """Asserted at the boundary, so the bug cannot recur by another route."""
    day = date(2026, 3, 4)
    with pytest.raises(DataError, match="one provider's bars"):
        store.seal(
            [daily(day, provider="alpaca"), daily(day, close="100.02", provider="yahoo")],
            provider="alpaca",
        )


def test_series_checks_run_per_provider(ledger: Ledger, store: BarStore) -> None:
    """The other half of the same regression.

    Two feeds' bars interleave by bar time into something that is not a time
    series. The audit reported a `duplicate_bar` for every single date — which
    looked alarming and meant nothing — and `check_jumps` would have compared
    one feed's close against the other's open.
    """
    days = sessions(date(2026, 3, 2), date(2026, 3, 6))
    ingest(store, [daily(day, provider="alpaca") for day in days], provider="alpaca")
    ingest(
        store,
        [daily(day, close="100.02", provider="yahoo") for day in days],
        provider="yahoo",
    )

    report = DataAuditor(ledger, store, calendar=CAL).run(
        resolutions=[Resolution.DAILY], emit_event=False
    )
    assert report.clean
    assert not [f for f in report.findings if f.kind is CheckKind.DUPLICATE_BAR]
    # One coverage report per provider, each over its own series.
    assert len(report.coverage) == 2
    assert all(coverage.expected == 5 for coverage in report.coverage)


def test_a_duplicate_needs_the_same_provider_to_count() -> None:
    """Same period from two feeds is two observations, not a duplicate."""
    day = date(2026, 3, 4)
    cross = [daily(day, provider="alpaca"), daily(day, close="100.02", provider="yahoo")]
    assert check_ordering(cross, instrument_uid=UID, resolution=Resolution.DAILY) == ()

    same = [daily(day, provider="alpaca"), daily(day, provider="alpaca")]
    (finding,) = check_ordering(same, instrument_uid=UID, resolution=Resolution.DAILY)
    assert finding.kind is CheckKind.DUPLICATE_BAR
    assert "from alpaca" in finding.detail


# ==========================================================================
# As-of monotonicity
# ==========================================================================


@given(
    ingest_order=st.permutations(list(range(8))),
    revise=st.integers(min_value=0, max_value=7),
    as_of_day=st.integers(min_value=0, max_value=20),
    backfilled=st.booleans(),
)
@settings(max_examples=150, deadline=None)
def test_visible_bars_are_monotonic_in_as_of(
    ingest_order: list[int], revise: int, as_of_day: int, backfilled: bool
) -> None:
    """For `t1 < t2`, `visible_bars(t1)` is a subset of `visible_bars(t2)`.

    Over random ingest orders *and* a revision, because the failure mode is
    order-dependent: a read path that collapsed vintages before filtering on
    knowledge time would return the restated value at an as-of instant before
    the restatement existed, and only some ingest orders would show it.

    And over both ways history arrives: bar by bar as each closed, or all at
    once by a backfill stamped with the day it was fetched, which is after
    every bar in it.
    """
    base = date(2026, 3, 2)
    fetched = datetime(2026, 5, 1, tzinfo=UTC) if backfilled else None
    bars = [daily(base + timedelta(days=index), ingested=fetched) for index in range(8)]
    # One bar restated much later, ingested out of band.
    restated_at = datetime(2026, 6, 1, tzinfo=UTC)
    bars.append(daily(base + timedelta(days=revise), close="999.00", ingested=restated_at))
    source = InMemoryBarSource(bars=[bars[i] for i in ingest_order] + [bars[-1]])

    earlier = datetime(2026, 3, 1, tzinfo=UTC) + timedelta(days=as_of_day)
    later = earlier + timedelta(days=1)

    first = visible_bars(source, UID, Resolution.DAILY, as_of=earlier)
    second = visible_bars(source, UID, Resolution.DAILY, as_of=later)

    assert {bar.bar_open_utc for bar in first} <= {bar.bar_open_utc for bar in second}
    for bar in first:
        assert bar.available_at_utc <= earlier
        # Every visible bar is either the first vintage of its period, which
        # stands for that period from its close, or was ingested by `as_of`.
        # The restatement is neither until it has happened.
        assert bar.ingested_at_utc != restated_at
    # Every period that had closed is there, however late it was fetched.
    assert len(first) == sum(1 for bar in bars[:8] if bar.available_at_utc <= earlier)


@given(as_of_day=st.integers(min_value=0, max_value=40))
@settings(max_examples=100, deadline=None)
def test_a_revision_is_invisible_before_it_was_ingested(as_of_day: int) -> None:
    """Both directions: the old value before, the new value after.

    This is the property Yahoo's back-adjustment breaks. Without it a backtest
    over last month silently uses a value restated last week.
    """
    day = date(2026, 3, 4)
    original = daily(day, close="100.00", ingested=datetime(2026, 3, 5, tzinfo=UTC))
    restated = daily(day, close="999.00", ingested=datetime(2026, 4, 1, tzinfo=UTC))
    source = InMemoryBarSource(bars=[restated, original])

    as_of = datetime(2026, 3, 5, tzinfo=UTC) + timedelta(days=as_of_day)
    visible = visible_bars(source, UID, Resolution.DAILY, as_of=as_of)
    assert len(visible) == 1
    expected = Decimal("999.00") if as_of >= datetime(2026, 4, 1, tzinfo=UTC) else Decimal("100.00")
    assert visible[0].close == expected


def test_backfilled_history_is_visible_from_each_bars_close() -> None:
    """A regression, and the one that made every backtest on real data empty.

    Both real feeds stamp a backfill with the moment it was fetched, which is
    after every bar in it, and the read path required every vintage to have
    been ingested by `as_of`. So a backtest over last year, run today, saw no
    bars at all — every decision dropped, nothing ever traded, nothing could
    ever be promoted. The fixtures hid it by stamping each bar as ingested the
    moment it closed, which no real backfill does.

    The first vintage of a period stands for it from its close; a restatement
    fetched later stays invisible until then, in both directions.
    """
    fetched = datetime(2026, 9, 25, 9, tzinfo=UTC)
    days = sessions(date(2025, 3, 3), date(2025, 3, 7))
    original = [daily(day, ingested=fetched) for day in days]
    restated = daily(days[2], close="777.00", ingested=datetime(2026, 10, 1, tzinfo=UTC))
    source = InMemoryBarSource(bars=[*original, restated])

    a_year_ago = visible_bars(source, UID, Resolution.DAILY, as_of=datetime(2025, 3, 8, tzinfo=UTC))
    before_the_restatement = visible_bars(
        source, UID, Resolution.DAILY, as_of=datetime(2026, 9, 30, tzinfo=UTC)
    )
    after_it = visible_bars(source, UID, Resolution.DAILY, as_of=datetime(2026, 10, 2, tzinfo=UTC))

    assert [bar.session_date for bar in a_year_ago] == days
    assert {bar.close for bar in a_year_ago} == {Decimal("100.00")}
    assert {bar.close for bar in before_the_restatement} == {Decimal("100.00")}
    assert after_it[2].close == Decimal("777.00")


def test_two_feeds_read_as_one_bar_per_period_from_the_primary() -> None:
    """A regression. The store keeps every feed's bars, and the read path used
    to return all of them: a symbol fetched from Alpaca and Yahoo came back as
    two bars a session, interleaved, so a 50-bar average spanned 25 days and a
    "return" ran from one feed's close to the other's. Now the primary's bar
    wins a period, and the fallback fills only a period the primary lacks."""
    days = sessions(date(2026, 3, 2), date(2026, 3, 6))
    alpaca = [daily(day, close="100.00") for day in days if day != days[2]]
    yahoo = [daily(day, close="25.00", provider="yahoo") for day in days]
    source = InMemoryBarSource(bars=[*yahoo, *alpaca])

    seen = visible_bars(source, UID, Resolution.DAILY, as_of=datetime(2026, 3, 9, tzinfo=UTC))

    assert [bar.session_date for bar in seen] == days
    assert [bar.provider for bar in seen] == ["alpaca", "alpaca", "yahoo", "alpaca", "alpaca"]


@given(
    held=st.lists(
        st.sets(st.sampled_from(["alpaca", "csv_fixture", "yahoo"])), min_size=1, max_size=8
    ),
    reverse=st.booleans(),
)
@settings(max_examples=100, deadline=None)
def test_each_period_is_read_from_the_most_preferred_feed_that_has_it(
    held: list[set[str]], reverse: bool
) -> None:
    """Whatever mix of feeds holds whichever days, in whatever order they are
    read: one bar per period, and it is the preferred feed's."""
    base = date(2026, 3, 2)
    bars = [
        daily(base + timedelta(days=index), provider=name)
        for index, names in enumerate(held)
        for name in sorted(names)
    ]
    source = InMemoryBarSource(bars=bars[::-1] if reverse else bars)

    seen = visible_bars(source, UID, Resolution.DAILY, as_of=datetime(2026, 4, 1, tzinfo=UTC))

    assert len({bar.bar_open_utc for bar in seen}) == len(seen)
    expected = [min(names, key=provider_preference) for names in held if names]
    assert [bar.provider for bar in seen] == expected


def test_a_revision_that_deletes_a_bar_stays_deleted_only_after_it_is_known(
    ledger: Ledger, store: BarStore
) -> None:
    """A deletion is invisible to a row-wise diff, so it is detected over the set.

    Yahoo drops bad prints. An as-of query before the deletion must still return
    the bar, because that is what was believed at the time.
    """
    days = sessions(date(2026, 3, 2), date(2026, 3, 6))
    ingest(store, [daily(day) for day in days])

    # Refetch the same window with one bar gone.
    kept = [daily(day) for day in days if day != date(2026, 3, 4)]
    result = store.ingest(
        BarBatch(
            bars=tuple(kept),
            provider="alpaca",
            symbol="AAPL",
            resolution=Resolution.DAILY,
            requested_start=datetime(2026, 3, 2, tzinfo=UTC),
            requested_end=datetime(2026, 3, 6, tzinfo=UTC),
        )
    )
    assert any(rev.kind == "deleted" for rev in result.revisions)

    # The original observation is still readable as of its own vintage.
    everything = list(store.bars_for(UID, Resolution.DAILY))
    assert any(bar.bar_open_utc.date() == date(2026, 3, 4) for bar in everything)


# ==========================================================================
# Vintage immutability
# ==========================================================================


@given(extra_days=st.integers(min_value=1, max_value=10))
@settings(max_examples=25, deadline=None)
def test_a_sealed_vintage_is_byte_identical_after_more_ingestion(
    tmp_path_factory: pytest.TempPathFactory, extra_days: int
) -> None:
    """Re-reading a sealed vintage returns the same rows and the same hash.

    Run as a property over how much data arrives afterwards, because the
    failure mode is a vintage that follows the *catalog* rather than its own
    file list — and that only shows up once compaction has re-sealed something.
    """
    root = tmp_path_factory.mktemp("vintage")
    with Ledger(root / "l.db") as ledger:
        ledger.initialise(created_by="test")
        store = BarStore(ledger, root=root / "bars")
        snapshots = SnapshotStore(ledger, store, calendar=CAL)

        ingest(store, [daily(day) for day in sessions(date(2026, 3, 2), date(2026, 3, 6))])
        vintage = snapshots.seal(resolutions=[Resolution.DAILY])
        before = [
            bar.storage_row(scale=store.scale) for bar in snapshots.bars_of(vintage.vintage_id)
        ]

        later = [
            daily(date(2026, 3, 9) + timedelta(days=index), close=f"10{index}.00")
            for index in range(extra_days)
        ]
        ingest(store, later)
        # And a restatement inside the sealed window.
        ingest(
            store,
            [daily(date(2026, 3, 4), close="777.00", ingested=datetime(2026, 7, 1, tzinfo=UTC))],
        )
        store.compact()

        after = [
            bar.storage_row(scale=store.scale) for bar in snapshots.bars_of(vintage.vintage_id)
        ]
        assert after == before


# ==========================================================================
# Timestamp-convention invariance
# ==========================================================================


@pytest.mark.parametrize(
    ("stamp", "expected_date"),
    [
        # Spring forward: 2026-03-08. The Monday after is EDT.
        ("2026-03-09T13:30:00+00:00", "2026-03-09"),
        # Fall back: 2026-11-01. The Monday after is EST.
        ("2026-11-02T14:30:00+00:00", "2026-11-02"),
        # UK and US DST disagree between 2026-03-08 and 2026-03-29.
        ("2026-03-16T13:30:00+00:00", "2026-03-16"),
        # A half-day: the day after Thanksgiving 2026.
        ("2026-11-27T14:30:00+00:00", "2026-11-27"),
    ],
)
def test_a_daily_bar_lands_on_its_own_session_across_dst(stamp: str, expected_date: str) -> None:
    """Anchored to midnight UTC of the *Eastern* session date.

    The bug this guards: read in UTC, a 00:00-Eastern stamp is the previous UTC
    day for four months of the year, and every date-keyed lookup lands one
    session early — a corporate action stops matching its own ex-date and a
    coverage gap is attributed to a day the market was shut.
    """
    anchored = normalise_bar_open(
        datetime.fromisoformat(stamp),
        convention=TimestampConvention.BAR_OPEN,
        resolution=Resolution.DAILY,
        session_tz=US_EASTERN,
    )
    assert anchored.isoformat() == f"{expected_date}T00:00:00+00:00"


@pytest.mark.parametrize(
    "stamp",
    [
        "2026-03-09T13:30:00+00:00",  # 09:30 EDT
        "2026-11-02T14:30:00+00:00",  # 09:30 EST
        "2026-03-16T13:30:00+00:00",  # US on EDT, UK still on GMT
    ],
)
def test_the_session_open_is_regular_on_both_sides_of_a_dst_change(stamp: str) -> None:
    """A fixed UTC offset is right for four months and an hour wrong for eight."""
    assert classify_us_session(datetime.fromisoformat(stamp), Resolution.MINUTE) is Session.REGULAR


def test_an_ambiguous_convention_fails_ingestion_at_every_resolution() -> None:
    """A guess here is a uniform one-bar lookahead, which looks like alpha."""
    for resolution in Resolution:
        with pytest.raises(DataError, match="ambiguous"):
            normalise_bar_open(
                datetime(2026, 3, 4, 15, tzinfo=UTC),
                convention=TimestampConvention.AMBIGUOUS,
                resolution=resolution,
            )


def test_a_bar_is_knowable_on_its_own_session_day_across_dst() -> None:
    """The daily anchor has to put `available_at` after the close, both seasons.

    In winter the US closes at 21:00 UTC and in summer at 20:00; midnight UTC
    the following day is after both, and before the next pre-open either way.
    """
    for stamp in ("2026-03-04T14:30:00+00:00", "2026-07-01T13:30:00+00:00"):
        anchored = normalise_bar_open(
            datetime.fromisoformat(stamp),
            convention=TimestampConvention.BAR_OPEN,
            resolution=Resolution.DAILY,
            session_tz=US_EASTERN,
        )
        close = anchored + Resolution.DAILY.duration
        eastern = close.astimezone(US_EASTERN)
        assert eastern.date() == anchored.date()
        assert eastern.hour >= 19


# ==========================================================================
# The lookahead canary
# ==========================================================================


def test_a_strategy_reaching_past_its_decision_time_crashes(
    ledger: Ledger, store: BarStore
) -> None:
    """The canary. A cheating strategy must raise, not receive `UNKNOWN`.

    Silently returning a sentinel would let a strategy handle it badly and
    produce a plausible-looking backtest nobody questions.
    """
    days = sessions(date(2026, 3, 2), date(2026, 3, 6))
    ingest(store, [daily(day) for day in days])
    reader = ForwardOnlyReader(source=store, resolution=Resolution.DAILY, instrument_uids=(UID,))
    window = reader.advance_to(datetime(2026, 3, 4, tzinfo=UTC))

    with pytest.raises(LookaheadError, match="did not exist yet"):
        window.require_visible(datetime(2026, 3, 5, tzinfo=UTC))


def test_a_short_window_returns_unknown_rather_than_a_shorter_average(
    ledger: Ledger, store: BarStore
) -> None:
    """A 200-day average over 40 days is a different and wrong number.

    `UNKNOWN` raises on arithmetic, so the feature code cannot use it by
    accident — which is the only reason "insufficient history" stays visible.
    """
    days = sessions(date(2026, 3, 2), date(2026, 3, 6))
    ingest(store, [daily(day) for day in days])
    reader = ForwardOnlyReader(source=store, resolution=Resolution.DAILY, instrument_uids=(UID,))
    window = reader.advance_to(datetime(2026, 3, 10, tzinfo=UTC))
    assert window.closes(UID, 3) is not UNKNOWN
    assert window.closes(UID, 200) is UNKNOWN


# ==========================================================================
# Crash atomicity
# ==========================================================================


def test_an_unrecorded_parquet_file_is_ignorable(ledger: Ledger, store: BarStore) -> None:
    """A crash between the rename and the ledger append leaves garbage.

    Harmless garbage: reads go through the catalog, so a file no event names is
    never opened. Reported as ORPHAN and safe to delete.
    """
    days = sessions(date(2026, 3, 2), date(2026, 3, 6))
    ingest(store, [daily(day) for day in days])
    store.compact()

    orphan = next(store.root.rglob("*.parquet")).parent / "deadbeef.parquet"
    orphan.write_bytes(b"junk")

    findings = store.verify_partitions()
    assert any(f.startswith("ORPHAN") for f in findings)
    assert not any(f.startswith(("MISSING", "ALTERED")) for f in findings)

    # And the orphan is not part of the dataset.
    assert all("deadbeef" not in info.relative_path for info in store.live_partitions())


def test_an_event_naming_a_missing_file_is_a_hard_failure(ledger: Ledger, store: BarStore) -> None:
    """The inverse ordering, which is why the ordering is what it is.

    A catalog row whose file is gone is data the ledger claims we hold and
    cannot produce — unrecoverable rather than merely untidy.
    """
    days = sessions(date(2026, 3, 2), date(2026, 3, 6))
    ingest(store, [daily(day) for day in days])
    store.compact()
    for path in store.root.rglob("*.parquet"):
        path.unlink()

    findings = store.verify_partitions()
    assert any(f.startswith("MISSING") for f in findings)


def test_an_altered_file_is_detected_by_hash(ledger: Ledger, store: BarStore) -> None:
    days = sessions(date(2026, 3, 2), date(2026, 3, 6))
    ingest(store, [daily(day) for day in days])
    store.compact()
    target = next(store.root.rglob("*.parquet"))
    target.write_bytes(b"tampered")

    findings = store.verify_partitions()
    assert any(f.startswith("ALTERED") for f in findings)


# ==========================================================================
# Structural staleness
# ==========================================================================


def test_a_provider_that_is_not_live_capable_says_so_with_a_reason() -> None:
    """Refused by arithmetic on the observed delay, not by a claim.

    A provider cannot assert itself into the decision path: `live_capable`
    computes from the measured delay, and the measurement wins over the
    declared value.
    """
    from tb.core.http import RecordingTransport
    from tb.data.pacing import PacingSpec, ProviderPacer
    from tb.data.providers import YahooProvider

    caps = YahooProvider(
        transport=RecordingTransport(),
        pacer=ProviderPacer(spec=PacingSpec(requests=1000, period_seconds=1.0), clock=lambda: 0.0),
    ).capabilities

    allowed, reason = caps.live_capable(Resolution.MINUTE, max_delay_seconds=180.0)
    assert not allowed
    assert "Usable for history, not for a live decision" in reason

    # An observed delay overrides the declared one, in both directions.
    ok, _ = caps.live_capable(Resolution.MINUTE, max_delay_seconds=180.0, observed_delay=20.0)
    assert ok
    bad, _ = caps.live_capable(Resolution.DAILY, max_delay_seconds=180.0, observed_delay=4000.0)
    assert not bad


def test_an_unmeasured_delay_is_assumed_unusable() -> None:
    """Fail-closed: no measurement means no live decision."""
    from tb.data.provider import ProviderCapabilities

    caps = ProviderCapabilities(
        name="mystery",
        resolutions=frozenset({Resolution.MINUTE}),
        timestamp_convention=TimestampConvention.BAR_OPEN,
    )
    allowed, reason = caps.live_capable(Resolution.MINUTE, max_delay_seconds=180.0)
    assert not allowed
    assert "assumed unusable" in reason
