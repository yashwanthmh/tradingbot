"""Session health: which sessions the loop ran cleanly, read from the ledger.

The live gate counts these, so the properties that matter are the ones that
would let a bad session through or keep a good one out: a late start is not a
session run, a crash is a fault even when nothing records it, a stop withdrawn
before an exit is not a protection failure, modes are counted apart, and a
verdict is not given until nothing more can change it.

Events are written at chosen instants by pinning the ledger's clock, so each
scenario is a real ledger read by the real reader rather than a hand-built
list of findings.
"""

from __future__ import annotations

from collections.abc import Iterator
from datetime import UTC, date, datetime, timedelta
from decimal import Decimal
from pathlib import Path
from typing import Any

import pytest
from typer.testing import CliRunner

from tb.cli import app
from tb.config.hard_limits import LiveLimits
from tb.config.loader import load_hard_limits
from tb.core.clock import to_iso
from tb.ledger.events import (
    EventType,
    HaltRaisedPayload,
    InstanceLockPayload,
    IntentCommittedPayload,
    KillswitchPayload,
    LoopCyclePayload,
    OrderOutcomePayload,
    PositionOrphanedPayload,
    ProtectionPayload,
    TradeClosedPayload,
    WatchdogPayload,
)
from tb.ledger.store import Ledger
from tb.ops.sessions import (
    FaultKind,
    ModeRecord,
    NoteKind,
    SessionVerdict,
    coverage,
    read_sessions,
)
from tests.conftest import REFERENCE_LIMITS

LIVE: LiveLimits = load_hard_limits(REFERENCE_LIMITS).limits.live

# A week with no holiday in it. US Eastern is UTC-4 throughout, so the
# regular session is 13:30 to 20:00 UTC every day.
MON = date(2026, 9, 21)
TUE = date(2026, 9, 22)
WED = date(2026, 9, 23)
THU = date(2026, 9, 24)
FRI = date(2026, 9, 25)
SAT = date(2026, 9, 26)
NEXT_MON = date(2026, 9, 28)

EVERY = timedelta(minutes=10)


def _utc(day: date, hour: int, minute: int = 0) -> datetime:
    return datetime(day.year, day.month, day.day, hour, minute, tzinfo=UTC)


class _Clock:
    """The ledger's clock, pinned, so an event lands at the instant a test says."""

    def __init__(self) -> None:
        self.at = _utc(MON, 12)

    def iso(self) -> str:
        return to_iso(self.at)


@pytest.fixture
def clock(monkeypatch: pytest.MonkeyPatch) -> _Clock:
    pinned = _Clock()
    monkeypatch.setattr("tb.ledger.store.now_iso", pinned.iso)
    return pinned


@pytest.fixture
def book(ledger_path: Path, clock: _Clock) -> Iterator[Ledger]:
    with Ledger(ledger_path) as opened:
        opened.initialise(created_by="test")
        yield opened


def _append(ledger: Ledger, clock: _Clock, at: datetime, event: EventType, payload: Any) -> None:
    clock.at = at
    ledger.append(event, getattr(payload, "run_id", None) or "subject", payload)


def _cycle(ledger: Ledger, clock: _Clock, run_id: str, at: datetime, n: int, **extra: Any) -> None:
    _append(
        ledger,
        clock,
        at,
        EventType.LOOP_CYCLE_COMPLETED,
        LoopCyclePayload(
            run_id=run_id,
            cycle=n,
            as_of_utc=at.isoformat(),
            n_instruments_considered=1,
            n_decisions=extra.get("decisions", 0),
            n_orders_submitted=extra.get("orders", 0),
            n_risk_refusals=0,
            duration_ms=5.0,
            detail=extra.get("detail", ""),
            n_fills_recorded=extra.get("fills", 0),
        ),
    )


def _run(
    ledger: Ledger,
    clock: _Clock,
    *,
    run_id: str,
    mode: str = "demo",
    day: date = MON,
    start: tuple[int, int] = (13, 0),
    stop: tuple[int, int] = (20, 0),
    skip: tuple[datetime, datetime] | None = None,
    end: str | None = "completed",
    error_type: str | None = None,
    error_detail: str | None = None,
) -> datetime:
    """A run of the loop: its start, a cycle every ten minutes, and its end.

    Returns the instant of its last cycle.
    """
    at = _utc(day, *start)
    clock.at = at
    ledger.record_run_start(run_id=run_id, mode=mode)
    last = at
    n = 0
    while at <= _utc(day, *stop):
        if skip is None or not skip[0] <= at < skip[1]:
            n += 1
            _cycle(ledger, clock, run_id, at, n)
            last = at
        at += EVERY
    if end is not None:
        clock.at = last + timedelta(seconds=30)
        ledger.record_run_end(
            run_id=run_id, exit_reason=end, error_type=error_type, error_detail=error_detail
        )
    return last


def _read(ledger: Ledger, *, now: datetime, mode: str = "demo") -> ModeRecord:
    return read_sessions(ledger, limits=LIVE, mode=mode, now=now)


def _only(record: ModeRecord) -> Any:
    assert len(record.sessions) == 1, [s.session_date for s in record.sessions]
    return record.sessions[0]


def _kinds(findings: Any) -> list[str]:
    return [finding.kind for finding in findings]


# --------------------------------------------------------------------------
# Coverage
# --------------------------------------------------------------------------


def test_coverage_is_the_time_within_reach_of_a_cycle() -> None:
    start, end = _utc(MON, 13, 30), _utc(MON, 14, 30)
    reach = timedelta(minutes=15)
    cycles = [
        start - timedelta(minutes=5),  # before the open: covers its first ten minutes
        start + timedelta(minutes=5),
        start + timedelta(minutes=40),
    ]
    covered, longest = coverage(cycles, start=start, end=end, reach=reach)

    # 13:30-13:50 covered, 13:50-14:10 not, 14:10-14:25 covered, 14:25-14:30 not.
    assert covered == 35 * 60
    assert longest == 20 * 60


def test_no_cycles_covers_nothing() -> None:
    start, end = _utc(MON, 13, 30), _utc(MON, 20)
    covered, longest = coverage([], start=start, end=end, reach=timedelta(minutes=15))
    assert covered == 0
    assert longest == (end - start).total_seconds()


# --------------------------------------------------------------------------
# What a clean session is
# --------------------------------------------------------------------------


def test_a_session_the_loop_ran_through_is_clean(book: Ledger, clock: _Clock) -> None:
    _run(book, clock, run_id="run_a")

    record = _read(book, now=_utc(MON, 21))
    session = _only(record)

    assert session.session_date == MON
    assert session.verdict is SessionVerdict.CLEAN, session.summary()
    assert session.coverage_pct == pytest.approx(100.0)
    assert session.run_ids == ("run_a",)
    assert session.n_cycles == 43
    assert not session.faults
    assert record.streak == (session,)


def test_a_session_is_not_judged_until_the_close_and_the_silence_bound_pass(
    book: Ledger, clock: _Clock
) -> None:
    """A run silent a minute before the bell may be a crash or a cycle about to land."""
    _run(book, clock, run_id="run_a", end=None)

    for now in (_utc(MON, 15), _utc(MON, 20, 5), _utc(MON, 20, 14)):
        assert _only(_read(book, now=now)).verdict is SessionVerdict.IN_PROGRESS, now
    # Past the bound the run is presumed dead, and its crash is a fault.
    judged = _only(_read(book, now=_utc(MON, 20, 16)))
    assert judged.verdict is SessionVerdict.FAULTED
    assert _kinds(judged.faults) == [FaultKind.RUN_CRASHED]


def test_a_run_still_cycling_after_the_close_leaves_a_clean_session(
    book: Ledger, clock: _Clock
) -> None:
    """The loop running on overnight is not a crash, and not the next session."""
    _run(book, clock, run_id="run_a", stop=(23, 50), end=None)

    record = _read(book, now=_utc(MON, 23, 55))
    session = _only(record)
    assert session.session_date == MON
    assert session.verdict is SessionVerdict.CLEAN, session.summary()


def test_stopping_the_loop_after_the_close_does_not_open_the_next_session(
    book: Ledger, clock: _Clock
) -> None:
    """Otherwise the next day would appear, empty, and be judged incomplete."""
    _run(book, clock, run_id="run_a", stop=(20, 20), end="interrupted")

    record = _read(book, now=_utc(TUE, 22))
    session = _only(record)
    assert session.session_date == MON
    assert session.verdict is SessionVerdict.CLEAN
    assert record.streak == (session,)


def test_a_late_start_is_incomplete_not_clean(book: Ledger, clock: _Clock) -> None:
    """Noon to the close is 61.5% of the session: the loop did not run it."""
    _run(book, clock, run_id="run_a", start=(16, 0))

    session = _only(_read(book, now=_utc(MON, 21)))
    assert session.verdict is SessionVerdict.INCOMPLETE
    assert session.coverage_pct == pytest.approx(100 * 240 / 390, abs=0.1)
    assert session.longest_gap_seconds == 150 * 60
    assert "covered" in session.summary()


def test_a_short_gap_is_tolerated_and_a_long_one_is_not(ledger_path: Path, clock: _Clock) -> None:
    """Twenty minutes without a cycle loses five; an hour and a half loses enough."""
    short = (_utc(MON, 16), _utc(MON, 16, 20))
    with Ledger(ledger_path) as book:
        book.initialise(created_by="test")
        _run(book, clock, run_id="run_a", skip=short)
        session = _only(_read(book, now=_utc(MON, 21)))
    assert session.verdict is SessionVerdict.CLEAN
    # Cycles at 15:50 and 16:20: the one at 15:50 reaches 16:05.
    assert session.longest_gap_seconds == 15 * 60

    long = (_utc(TUE, 16), _utc(TUE, 17, 30))
    with Ledger(ledger_path) as book:
        _run(book, clock, run_id="run_b", day=TUE, skip=long)
        tuesday = _read(book, now=_utc(TUE, 21)).session(TUE)
    assert tuesday is not None
    assert tuesday.verdict is SessionVerdict.INCOMPLETE
    assert tuesday.coverage_pct < LIVE.session_min_coverage_pct


# --------------------------------------------------------------------------
# Faults
# --------------------------------------------------------------------------


def test_a_halted_run_faults_its_session(book: Ledger, clock: _Clock) -> None:
    _run(
        book,
        clock,
        run_id="run_a",
        stop=(15, 0),
        end="halted",
        error_type="LoopHalted",
        error_detail="self-check failed: the kill switch forbids trading",
    )

    session = _only(_read(book, now=_utc(MON, 21)))
    assert session.verdict is SessionVerdict.FAULTED
    assert _kinds(session.faults) == [FaultKind.HALTED]
    assert "kill switch" in session.faults[0].detail
    assert session.faults[0].run_id == "run_a"


def test_an_error_ending_faults_its_session(book: Ledger, clock: _Clock) -> None:
    _run(
        book,
        clock,
        run_id="run_a",
        stop=(19, 0),
        end="error",
        error_type="LedgerUnwritableError",
        error_detail="disk full",
    )

    session = _only(_read(book, now=_utc(MON, 21)))
    assert _kinds(session.faults) == [FaultKind.RUN_ERROR]
    assert "LedgerUnwritableError: disk full" in session.faults[0].detail


def test_an_operator_stop_is_a_note_and_the_coverage_decides(book: Ledger, clock: _Clock) -> None:
    """Ctrl-C is not a fault; stopping at two o'clock still did not run the session."""
    _run(book, clock, run_id="run_a", stop=(18, 0), end="interrupted")

    session = _only(_read(book, now=_utc(MON, 21)))
    assert not session.faults
    assert _kinds(session.notes) == [NoteKind.STOPPED]
    assert session.verdict is SessionVerdict.INCOMPLETE


def test_the_kill_switch_counts_only_while_a_run_is_up(book: Ledger, clock: _Clock) -> None:
    """Engaged mid-session is a fault; engaged after the run ended is nobody's."""
    _run(book, clock, run_id="run_a")
    _run(book, clock, run_id="run_b", day=TUE)
    engaged = KillswitchPayload(
        path="var/KILL", determinable=True, detail="drill", engaged_by="ops"
    )
    _append(book, clock, _utc(MON, 15), EventType.KILLSWITCH_ENGAGED, engaged)
    _append(book, clock, _utc(TUE, 22), EventType.KILLSWITCH_ENGAGED, engaged)

    record = _read(book, now=_utc(WED, 12))
    monday, tuesday = record.sessions
    assert _kinds(monday.faults) == [FaultKind.KILL_SWITCH]
    assert "engaged by ops" in monday.faults[0].detail
    assert tuesday.verdict is SessionVerdict.CLEAN


def test_a_manual_halt_and_a_watchdog_trip_while_up_are_faults(book: Ledger, clock: _Clock) -> None:
    """Neither names the trading run — `tb halt` mints its own id — so time decides."""
    _run(book, clock, run_id="run_a")
    _append(
        book,
        clock,
        _utc(MON, 15),
        EventType.HALT_RAISED,
        HaltRaisedPayload(halt_id="halt_1", trigger="manual", detail="checking", run_id="run_x"),
    )
    _append(
        book,
        clock,
        _utc(MON, 16),
        EventType.WATCHDOG_TRIPPED,
        WatchdogPayload(direction="watchdog_to_trader", action_taken="engaged the kill switch"),
    )

    session = _only(_read(book, now=_utc(MON, 21)))
    assert _kinds(session.faults) == [FaultKind.HALT_RAISED, FaultKind.WATCHDOG]


def test_a_wedged_loop_owns_the_watchdog_trip_its_silence_caused(
    book: Ledger, clock: _Clock
) -> None:
    """A wedged loop writes nothing; the trip minutes later is still its fault."""
    last = _run(book, clock, run_id="run_a", stop=(15, 0), end=None)
    _append(
        book,
        clock,
        last + timedelta(minutes=5),
        EventType.WATCHDOG_TRIPPED,
        WatchdogPayload(direction="watchdog_to_trader", action_taken="engaged the kill switch"),
    )

    session = _only(_read(book, now=_utc(MON, 21)))
    assert set(_kinds(session.faults)) == {FaultKind.WATCHDOG, FaultKind.RUN_CRASHED}


def _protection(**fields: Any) -> ProtectionPayload:
    base: dict[str, Any] = {"t212_ticker": "AAPL_US_EQ", "run_id": "run_a", "quantity": Decimal(1)}
    return ProtectionPayload(**(base | fields))


@pytest.mark.parametrize(
    ("cause", "detail", "faulted"),
    [
        ("withdrawn", "stop withdrawn: exit", False),
        ("placement_failed", "the protective stop could not be placed: 503", True),
        ("risk_refused", "risk refused the protective stop: ...", True),
        ("no_price", "no entry price and no usable bar to place a stop from", True),
        # Written before `cause` existed: the detail is all there is.
        (None, "stop withdrawn: exit", False),
        (None, "the protective stop could not be placed: 503", True),
    ],
)
def test_a_withdrawn_stop_is_not_a_fault_and_a_failed_one_is(
    book: Ledger, clock: _Clock, cause: str | None, detail: str, faulted: bool
) -> None:
    _run(book, clock, run_id="run_a")
    _append(
        book,
        clock,
        _utc(MON, 15),
        EventType.POSITION_UNPROTECTED,
        _protection(protected=False, cause=cause, detail=detail),
    )

    session = _only(_read(book, now=_utc(MON, 21)))
    assert (session.verdict is SessionVerdict.FAULTED) is faulted
    if faulted:
        assert _kinds(session.faults) == [FaultKind.PROTECTION_FAILED]


def test_an_entry_left_bare_too_long_faults_its_session(book: Ledger, clock: _Clock) -> None:
    _run(book, clock, run_id="run_a")
    _run(book, clock, run_id="run_b", day=TUE)
    bound = LIVE.max_unprotected_seconds
    _append(
        book,
        clock,
        _utc(MON, 15),
        EventType.POSITION_PROTECTED,
        _protection(protected=True, unprotected_seconds=bound - 1.0),
    )
    _append(
        book,
        clock,
        _utc(TUE, 15),
        EventType.POSITION_PROTECTED,
        _protection(run_id="run_b", protected=True, unprotected_seconds=bound + 60.0),
    )

    monday, tuesday = _read(book, now=_utc(WED, 12)).sessions
    assert monday.verdict is SessionVerdict.CLEAN
    assert _kinds(tuesday.faults) == [FaultKind.UNPROTECTED_TOO_LONG]


def _commit(ledger: Ledger, clock: _Clock, at: datetime, intent_id: str) -> None:
    _append(
        ledger,
        clock,
        at,
        EventType.INTENT_COMMITTED,
        IntentCommittedPayload(
            intent_id=intent_id,
            run_id="run_a",
            t212_ticker="AAPL_US_EQ",
            side="buy",
            order_type="market",
            purpose="entry",
            priority_class="risk_increasing",
            quantity=Decimal(1),
            risk_token_id="tok",
        ),
    )


def _ack(ledger: Ledger, clock: _Clock, at: datetime, intent_id: str) -> None:
    _append(
        ledger,
        clock,
        at,
        EventType.ORDER_ACKNOWLEDGED,
        OrderOutcomePayload(
            intent_id=intent_id,
            run_id="run_a",
            t212_ticker="AAPL_US_EQ",
            broker_order_id="1",
            status="working",
        ),
    )


def test_an_order_left_in_an_unknown_state_faults_its_session(book: Ledger, clock: _Clock) -> None:
    """Answered at once is fine; answered after twenty minutes, or never, is not."""
    _run(book, clock, run_id="run_a")
    _commit(book, clock, _utc(MON, 14), "int_prompt")
    _ack(book, clock, _utc(MON, 14) + timedelta(seconds=1), "int_prompt")
    _commit(book, clock, _utc(MON, 15), "int_slow")
    _ack(book, clock, _utc(MON, 15, 20), "int_slow")
    _commit(book, clock, _utc(MON, 16), "int_never")

    session = _only(_read(book, now=_utc(MON, 21)))
    assert _kinds(session.faults) == [FaultKind.ORDER_UNKNOWN, FaultKind.ORDER_UNKNOWN]
    slow, never = session.faults
    assert "int_slow" in slow.detail and "stayed unknown for 20m00s" in slow.detail
    assert "int_never" in never.detail and "is still unknown" in never.detail


def test_an_order_just_sent_is_not_yet_unknown(book: Ledger, clock: _Clock) -> None:
    _run(book, clock, run_id="run_a", end=None, stop=(16, 0))
    _commit(book, clock, _utc(MON, 16, 1), "int_in_flight")

    session = _only(_read(book, now=_utc(MON, 16, 2)))
    assert not session.faults
    assert session.verdict is SessionVerdict.IN_PROGRESS


# --------------------------------------------------------------------------
# Attribution
# --------------------------------------------------------------------------


def test_modes_are_counted_apart(book: Ledger, clock: _Clock) -> None:
    """Thirty clean days on the simulated broker say nothing about Trading 212."""
    _run(book, clock, run_id="run_paper", mode="paper", day=MON)
    _run(book, clock, run_id="run_demo", mode="demo", day=TUE, stop=(19, 0), end="halted")

    paper = _read(book, now=_utc(WED, 12), mode="paper")
    demo = _read(book, now=_utc(WED, 12), mode="demo")
    assert [s.session_date for s in paper.sessions] == [MON]
    assert [s.session_date for s in demo.sessions] == [TUE]
    assert _only(paper).clean
    assert _only(demo).verdict is SessionVerdict.FAULTED
    # Nothing that is not a trading run makes a session.
    assert _read(book, now=_utc(WED, 12), mode="live").sessions == ()


def test_a_crash_over_the_weekend_breaks_monday(book: Ledger, clock: _Clock) -> None:
    """The run was waiting for Monday; its death belongs to Monday, not to nobody."""
    _run(book, clock, run_id="run_a", day=FRI, stop=(23, 50), end=None)
    _cycle(book, clock, "run_a", _utc(SAT, 10), 99)

    record = _read(book, now=_utc(NEXT_MON, 21))
    friday, monday = record.sessions
    assert friday.session_date == FRI and friday.clean
    assert monday.session_date == NEXT_MON
    assert _kinds(monday.faults) == [FaultKind.RUN_CRASHED]
    assert record.streak == ()


def test_notes_do_not_make_a_session_unclean(book: Ledger, clock: _Clock) -> None:
    _run(book, clock, run_id="run_a")
    _append(
        book,
        clock,
        _utc(MON, 14),
        EventType.ORDER_REJECTED,
        OrderOutcomePayload(
            intent_id="int_1",
            run_id="run_a",
            t212_ticker="AAPL_US_EQ",
            status="rejected",
            detail="insufficient funds",
            broker_message="InsufficientFreeForStocksBuy",
        ),
    )
    _append(
        book,
        clock,
        _utc(MON, 15),
        EventType.POSITION_ORPHANED,
        PositionOrphanedPayload(
            t212_ticker="MSFT_US_EQ",
            quantity="1",
            reason="no funded strategy holds it",
            action_taken="flattened",
        ),
    )
    _append(
        book,
        clock,
        _utc(MON, 16),
        EventType.INSTANCE_LOCK_REFUSED,
        InstanceLockPayload(
            lock_name="trading_loop", run_id="run_second", host="h", pid=2, acquired=False
        ),
    )
    _cycle(book, clock, "run_a", _utc(MON, 17, 5), 100, detail="settlement deferred: 503")

    session = _only(_read(book, now=_utc(MON, 21)))
    assert session.verdict is SessionVerdict.CLEAN
    assert _kinds(session.notes) == [
        NoteKind.ORDER_REJECTED,
        NoteKind.POSITION_ORPHANED,
        NoteKind.SECOND_INSTANCE,
        NoteKind.SETTLEMENT_DEFERRED,
    ]
    assert "InsufficientFreeForStocksBuy" in session.notes[0].detail


def test_activity_is_counted_per_session(book: Ledger, clock: _Clock) -> None:
    _run(book, clock, run_id="run_a", stop=(14, 0))
    _run(book, clock, run_id="run_b", start=(14, 10))
    _cycle(book, clock, "run_b", _utc(MON, 19, 55), 50, decisions=2, orders=1, fills=1)
    _append(
        book,
        clock,
        _utc(MON, 19),
        EventType.TRADE_CLOSED,
        TradeClosedPayload(
            closing_fill_id="fill_1",
            run_id="run_b",
            t212_ticker="AAPL_US_EQ",
            quantity=Decimal(1),
            admissible=True,
            charged=True,
        ),
    )

    record = _read(book, now=_utc(MON, 21))
    session = _only(record)
    assert session.verdict is SessionVerdict.CLEAN
    assert session.run_ids == ("run_a", "run_b")
    assert (session.n_decisions, session.n_orders, session.n_fills) == (2, 1, 1)
    assert session.n_trades_closed == 1
    assert record.trades_in_streak == 1


# --------------------------------------------------------------------------
# The streak
# --------------------------------------------------------------------------


def test_the_streak_counts_back_to_the_first_unclean_session(book: Ledger, clock: _Clock) -> None:
    """A fault resets it; a session still being judged neither counts nor breaks it."""
    _run(book, clock, run_id="run_mon", day=MON)
    _run(book, clock, run_id="run_tue", day=TUE, stop=(19, 0), end="halted")
    _run(book, clock, run_id="run_wed", day=WED)
    _run(book, clock, run_id="run_thu", day=THU)
    _run(book, clock, run_id="run_fri", day=FRI, stop=(15, 0), end=None)

    record = _read(book, now=_utc(FRI, 15, 5))
    verdicts = [s.verdict for s in record.sessions]
    assert verdicts == [
        SessionVerdict.CLEAN,
        SessionVerdict.FAULTED,
        SessionVerdict.CLEAN,
        SessionVerdict.CLEAN,
        SessionVerdict.IN_PROGRESS,
    ]
    assert [s.session_date for s in record.streak] == [WED, THU]
    assert record.n_clean == 3


def test_a_day_without_a_run_neither_counts_nor_breaks_the_streak(
    book: Ledger, clock: _Clock
) -> None:
    _run(book, clock, run_id="run_mon", day=MON)
    _run(book, clock, run_id="run_thu", day=THU)

    record = _read(book, now=_utc(FRI, 12))
    assert [s.session_date for s in record.streak] == [MON, THU]


def test_the_reader_refuses_an_unknown_mode(book: Ledger) -> None:
    with pytest.raises(ValueError, match="no such trading mode"):
        read_sessions(book, limits=LIVE, mode="research", now=_utc(MON, 12))


# --------------------------------------------------------------------------
# tb sessions
# --------------------------------------------------------------------------


def _sessions_cli(ledger_path: Path, *args: str) -> Any:
    return CliRunner().invoke(
        app,
        ["sessions", "--limits", str(REFERENCE_LIMITS), "--db", str(ledger_path), *args],
    )


def test_tb_sessions_lists_the_sessions_and_the_streak_the_gate_counts(
    ledger_path: Path, clock: _Clock, monkeypatch: pytest.MonkeyPatch
) -> None:
    with Ledger(ledger_path) as book:
        book.initialise(created_by="test")
        _run(book, clock, run_id="run_mon", day=MON)
        _run(
            book,
            clock,
            run_id="run_tue",
            day=TUE,
            stop=(19, 0),
            end="halted",
            error_detail="lease lost",
        )
        _run(book, clock, run_id="run_wed", day=WED)
    monkeypatch.setattr("tb.ops.sessions.now_utc", lambda: _utc(THU, 12))

    listed = _sessions_cli(ledger_path)
    assert listed.exit_code == 0, listed.output
    assert "2026-09-21" in listed.output and "2026-09-23" in listed.output
    assert "faulted" in listed.output and "lease lost" in listed.output
    assert "1 consecutive clean demo session(s), since 2026-09-23" in listed.output
    assert f"needs {LIVE.min_clean_demo_sessions}" in listed.output

    detail = _sessions_cli(ledger_path, "--date", "2026-09-22")
    assert detail.exit_code == 0, detail.output
    assert "[halted]" in detail.output
    assert "run_tue" in detail.output

    missing = _sessions_cli(ledger_path, "--date", "2026-09-26")
    assert missing.exit_code == 1
    assert "no demo session on 2026-09-26" in missing.output

    paper = _sessions_cli(ledger_path, "--mode", "paper")
    assert paper.exit_code == 0
    assert "no paper sessions" in paper.output


def test_tb_sessions_refuses_bad_arguments(ledger_path: Path) -> None:
    with Ledger(ledger_path) as book:
        book.initialise(created_by="test")

    assert _sessions_cli(ledger_path, "--mode", "research").exit_code == 2
    assert _sessions_cli(ledger_path, "--date", "tuesday").exit_code == 2
    assert _sessions_cli(ledger_path.with_name("absent.db")).exit_code == 2
