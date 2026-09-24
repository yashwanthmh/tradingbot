"""The sealed holdout: structurally invisible, and evaluated exactly once."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from decimal import Decimal

import pytest

from tb.data.asof import ForwardOnlyReader, HoldoutViolation, InMemoryBarSource, LookaheadError
from tb.data.provider import Bar, Provenance, Resolution, Session
from tb.data.snapshot import DatasetVintage, PitCompleteness
from tb.data.universe import SurvivorshipFlag
from tb.ledger.events import EventType
from tb.ledger.store import Ledger
from tb.research.holdout import (
    MIN_HOLDOUT_DAYS,
    HoldoutAlreadyEvaluated,
    HoldoutError,
    HoldoutRegistry,
    SealedBarSource,
    evaluation_reader,
    holdout_boundary,
    training_reader,
)

BASE = datetime(2024, 1, 2, tzinfo=UTC)
UID = "isin:US0378331005"
SEAL = BASE + timedelta(days=60)


def bar(day: int, *, available_after_days: int = 1) -> Bar:
    opened = BASE + timedelta(days=day)
    return Bar(
        instrument_uid=UID,
        resolution=Resolution.DAILY,
        bar_open_utc=opened,
        available_at_utc=opened + timedelta(days=available_after_days),
        ingested_at_utc=opened + timedelta(days=available_after_days),
        provider="fixture",
        provenance=Provenance.BACKFILL,
        session=Session.REGULAR,
        open=Decimal("100.00"),
        high=Decimal("101.00"),
        low=Decimal("99.00"),
        close=Decimal("100.50"),
        volume=1_000,
    )


def source(days: int = 120) -> InMemoryBarSource:
    return InMemoryBarSource(bars=[bar(day) for day in range(days)])


def a_vintage(*, start: datetime = BASE, days: int = 400) -> DatasetVintage:
    return DatasetVintage(
        vintage_id="vint_test",
        as_of_utc=start + timedelta(days=days),
        manifest_hash="h",
        file_sha256s=("f",),
        instrument_uids=(UID,),
        resolutions=(Resolution.DAILY,),
        window_start=start,
        window_end=start + timedelta(days=days),
        row_count=days,
        survivorship_flag=SurvivorshipFlag.UNMEASURED,
        pit_completeness_flag=PitCompleteness.VENDOR_CURRENT_VIEW,
    )


# --------------------------------------------------------------------------
# The boundary
# --------------------------------------------------------------------------


def test_the_boundary_is_derived_from_the_vintage() -> None:
    """Pure and derived, so a caller cannot move it after seeing a result."""
    window = holdout_boundary(a_vintage(days=400))
    assert window.train_days == 300
    assert window.holdout_days == 100
    assert window.is_usable
    assert "train" in window.summary()


def test_the_same_vintage_always_gives_the_same_boundary() -> None:
    vintage = a_vintage(days=400)
    assert holdout_boundary(vintage) == holdout_boundary(vintage)


def test_a_degenerate_fraction_is_refused() -> None:
    for fraction in (0.0, 1.0, -0.5, 1.5):
        with pytest.raises(HoldoutError, match="strictly between 0 and 1"):
            holdout_boundary(a_vintage(), fraction=fraction)


def test_a_windowless_vintage_cannot_be_split() -> None:
    vintage = a_vintage()
    empty = DatasetVintage(
        vintage_id=vintage.vintage_id,
        as_of_utc=vintage.as_of_utc,
        manifest_hash=vintage.manifest_hash,
        file_sha256s=vintage.file_sha256s,
        instrument_uids=vintage.instrument_uids,
        resolutions=vintage.resolutions,
        window_start=None,
        window_end=None,
        row_count=0,
    )
    with pytest.raises(HoldoutError, match="has no window"):
        holdout_boundary(empty)


def test_too_short_a_holdout_is_reported_as_unusable() -> None:
    """Not an error — a fact the gate reads. A 10-day holdout produces
    out-of-sample statistics that are noise, and promoting on them while
    believing there was an independent check is the failure."""
    window = holdout_boundary(a_vintage(days=40))
    assert window.holdout_days < MIN_HOLDOUT_DAYS
    assert not window.is_usable


# --------------------------------------------------------------------------
# The sealed source
# --------------------------------------------------------------------------


def test_the_sealed_source_holds_no_bar_past_the_boundary() -> None:
    sealed = SealedBarSource(inner=source(), sealed_from=SEAL)
    bars = list(sealed.bars_for(UID, Resolution.DAILY))
    assert bars
    assert all(b.bar_open_utc < SEAL for b in bars)
    assert all(b.available_at_utc < SEAL for b in bars)


def test_the_sealed_source_filters_on_knowledge_time_too() -> None:
    """The subtler half. A bar whose open precedes the boundary but whose
    `available_at` follows it was not knowable before the boundary, so
    including it leaks knowledge time while respecting event time.
    """
    # Opens one day before the seal, but is not knowable for another ten.
    late = bar(59, available_after_days=10)
    assert late.bar_open_utc < SEAL < late.available_at_utc
    sealed = SealedBarSource(inner=InMemoryBarSource(bars=[late]), sealed_from=SEAL)
    assert list(sealed.bars_for(UID, Resolution.DAILY)) == []


def test_a_request_naming_the_holdout_raises() -> None:
    sealed = SealedBarSource(inner=source(), sealed_from=SEAL)
    with pytest.raises(HoldoutViolation, match="at or past the holdout boundary"):
        list(sealed.bars_for(UID, Resolution.DAILY, start=SEAL + timedelta(days=1)))
    with pytest.raises(HoldoutViolation, match="at or past the holdout boundary"):
        list(sealed.bars_for(UID, Resolution.DAILY, end=SEAL + timedelta(days=1)))


def test_a_violation_can_be_recorded_as_well_as_raised() -> None:
    """A raise stops one process; the event makes a pattern visible."""
    seen: list[tuple[datetime, str]] = []
    sealed = SealedBarSource(
        inner=source(),
        sealed_from=SEAL,
        on_violation=lambda when, which: seen.append((when, which)),
    )
    with pytest.raises(HoldoutViolation):
        list(sealed.bars_for(UID, Resolution.DAILY, start=SEAL))
    assert seen == [(SEAL, "start")]


def test_the_sealed_source_still_reports_its_instruments() -> None:
    sealed = SealedBarSource(inner=source(), sealed_from=SEAL)
    assert list(sealed.instruments()) == [UID]


# --------------------------------------------------------------------------
# The reader
# --------------------------------------------------------------------------


def test_a_training_reader_refuses_to_advance_into_the_holdout() -> None:
    reader = training_reader(
        source(), sealed_from=SEAL, resolution=Resolution.DAILY, instrument_uids=[UID]
    )
    reader.advance_to(SEAL - timedelta(days=1))
    with pytest.raises(HoldoutViolation, match="the holdout is sealed from"):
        reader.advance_to(SEAL + timedelta(days=5))


def test_a_training_reader_refuses_the_boundary_itself() -> None:
    """At the boundary, not merely past it: `available_at <= as_of` includes a
    bar available at exactly `as_of`, so a decision at the boundary instant is
    already the first holdout decision."""
    reader = training_reader(
        source(), sealed_from=SEAL, resolution=Resolution.DAILY, instrument_uids=[UID]
    )
    with pytest.raises(HoldoutViolation):
        reader.advance_to(SEAL)


def test_a_violation_is_fatal_rather_than_an_empty_window() -> None:
    """The empty window is the dangerous version. A strategy that sees no bars
    concludes it has no signal, and the searcher records a negative result
    about a window it never tested."""
    reader = training_reader(
        source(), sealed_from=SEAL, resolution=Resolution.DAILY, instrument_uids=[UID]
    )
    with pytest.raises(HoldoutViolation):
        reader.advance_to(SEAL + timedelta(days=30))


def test_a_holdout_violation_is_a_lookahead() -> None:
    """Which is what it is: the holdout is data the process that produced the
    strategy must not have seen."""
    assert issubclass(HoldoutViolation, LookaheadError)


def test_an_unsealed_reader_is_unaffected() -> None:
    """The default stays exactly as it was, so M3's readers are untouched."""
    reader = ForwardOnlyReader(source=source(), resolution=Resolution.DAILY, instrument_uids=(UID,))
    window = reader.advance_to(SEAL + timedelta(days=30))
    assert len(window.bars(UID)) > 0


def test_the_evaluation_reader_can_see_the_holdout() -> None:
    """It is the one process that must, and it is a separate function with its
    own name rather than a flag whose default decides safety."""
    reader = evaluation_reader(source(), resolution=Resolution.DAILY, instrument_uids=[UID])
    window = reader.advance_to(SEAL + timedelta(days=30))
    assert any(b.bar_open_utc >= SEAL for b in window.bars(UID))


def test_the_evaluation_readers_lookback_reaches_into_training_data() -> None:
    """Necessary and correct. A feature needing 100 bars on the first holdout
    decision has to get them from somewhere, and the boundary is about what may
    inform the *choice* of strategy, not what a fixed strategy computes from.
    """
    reader = evaluation_reader(
        source(),
        resolution=Resolution.DAILY,
        instrument_uids=[UID],
        lookback=timedelta(days=90),
    )
    window = reader.advance_to(SEAL + timedelta(days=1))
    assert any(b.bar_open_utc < SEAL for b in window.bars(UID))


# --------------------------------------------------------------------------
# The single evaluation
# --------------------------------------------------------------------------


@pytest.fixture
def holdouts(ledger: Ledger) -> HoldoutRegistry:
    return HoldoutRegistry(ledger, run_id="run_test")


def record_one(holdouts: HoldoutRegistry, *, passed: bool = True) -> object:
    return holdouts.record(
        strategy_id="stg_1",
        version=1,
        lineage_id="lin_1",
        spec_hash="hash_1",
        vintage_id="vint_test",
        sealed_from=SEAL,
        passed=passed,
        n_trades=42,
        net_sharpe=0.9,
        returns=[Decimal("0.001")] * 10,
    )


def test_an_evaluation_is_recorded_with_its_event(
    holdouts: HoldoutRegistry, ledger: Ledger
) -> None:
    record_one(holdouts)
    row = ledger.conn.execute("SELECT * FROM holdout_evaluations").fetchone()
    assert row["strategy_id"] == "stg_1"
    assert row["passed"] == 1
    events = ledger.conn.execute(
        "SELECT COUNT(*) FROM event_log WHERE event_type = ?",
        (EventType.HOLDOUT_EVALUATED.value,),
    ).fetchone()
    assert events[0] == 1


def test_a_second_evaluation_of_the_same_version_is_refused(
    holdouts: HoldoutRegistry,
) -> None:
    """The mechanism the whole holdout rests on. 'Failed, tweak, resubmit' fits
    the holdout one bit per attempt."""
    record_one(holdouts, passed=False)
    with pytest.raises(HoldoutAlreadyEvaluated, match="already evaluated"):
        record_one(holdouts, passed=True)


def test_the_refusal_reports_what_the_first_answer_was(
    holdouts: HoldoutRegistry,
) -> None:
    record_one(holdouts, passed=False)
    with pytest.raises(HoldoutAlreadyEvaluated, match="failed"):
        record_one(holdouts)


def test_the_refusal_points_at_the_only_legitimate_next_step(
    holdouts: HoldoutRegistry,
) -> None:
    record_one(holdouts, passed=False)
    with pytest.raises(HoldoutAlreadyEvaluated, match="multiplicity haircut"):
        record_one(holdouts)


def test_the_database_constraint_refuses_a_second_row_even_without_the_check(
    holdouts: HoldoutRegistry, ledger: Ledger
) -> None:
    """The check can be raced; the constraint cannot. This asserts the
    constraint is really there rather than trusting the read-then-write."""
    import sqlite3

    record_one(holdouts)
    with pytest.raises(sqlite3.IntegrityError):
        ledger.conn.execute(
            "INSERT INTO holdout_evaluations (evaluation_id, strategy_id, version, "
            "lineage_id, spec_hash, vintage_id, sealed_from, passed, evaluated_at, "
            "evaluating_event_seq) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            ("hold_other", "stg_1", 1, "lin_1", "hash_1", "v", SEAL.isoformat(), 1, "x", 1),
        )


def test_a_different_version_gets_its_own_single_evaluation(
    holdouts: HoldoutRegistry,
) -> None:
    record_one(holdouts, passed=False)
    holdouts.record(
        strategy_id="stg_1",
        version=2,
        lineage_id="lin_1",
        spec_hash="hash_2",
        vintage_id="vint_test",
        sealed_from=SEAL,
        passed=True,
    )
    assert holdouts.has_been_evaluated("stg_1", 1)
    assert holdouts.has_been_evaluated("stg_1", 2)
    assert len(holdouts.in_lineage("lin_1")) == 2


def test_an_evaluation_round_trips_including_its_returns(
    holdouts: HoldoutRegistry,
) -> None:
    record_one(holdouts)
    stored = holdouts.existing("stg_1", 1)
    assert stored is not None
    assert stored.n_trades == 42
    assert stored.net_sharpe == 0.9
    assert len(stored.returns) == 10
    assert stored.label == "stg_1@v1"


def test_an_unevaluated_version_reports_nothing(holdouts: HoldoutRegistry) -> None:
    assert holdouts.existing("stg_unknown") is None
    assert not holdouts.has_been_evaluated("stg_unknown")


def test_a_violation_attempt_is_recorded(holdouts: HoldoutRegistry, ledger: Ledger) -> None:
    holdouts.record_violation(
        sealed_from=SEAL,
        requested_at=SEAL + timedelta(days=1),
        strategy_id="stg_1",
        lineage_id="lin_1",
        caller="searcher",
    )
    row = ledger.conn.execute(
        "SELECT * FROM event_log WHERE event_type = ?",
        (EventType.HOLDOUT_VIOLATION_ATTEMPTED.value,),
    ).fetchone()
    assert row is not None
    assert "searcher" in str(row["payload_json"])
