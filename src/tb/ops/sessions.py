"""Which trading sessions ran cleanly, per mode, read from the ledger alone.

`tb arm --live` asks for thirty clean sessions on the demo account. That count
has to come from somewhere nobody can round up — not a log someone skims, not a
file the bot keeps about itself, but the chained record of what each run did.
So a verdict here is a pure function of the ledger, the calendar, the `live`
limits and the moment of asking: the same inputs give the same answer on any
machine, and an arming record can cite the sessions it counted.

**A session** is one trading day, per mode. Paper, demo and live are counted
apart, because thirty clean days against the simulated broker say nothing about
Trading 212. Each event belongs to the first session whose close is at or after
it, so a crash overnight or on a Sunday counts against the session the run was
waiting for rather than falling between two. A session exists for a mode when a
run of that mode was up during its regular hours, or when a fault lands in it;
a day on which neither happened is not a session at all, so a week off neither
counts nor breaks anything.

**Clean** means three things, each read from events the loop already writes:

* *It ran the session.* A cycle within `session_max_cycle_gap_seconds` of at
  least `session_min_coverage_pct` of regular hours. A loop that started at
  noon, or died at two, did not run the session, however tidy the half it saw.
* *Nothing stopped it.* No halt, error or crash of a run of the mode, and no
  kill switch, watchdog trip or drift while one was up.
* *What it held was protected, and what it sent was known.* No protective stop
  that failed to land, no entry bare for longer than `max_unprotected_seconds`,
  and no order left in an unknown state for longer than the cycle-gap bound.

Rejections, orphaned positions, a refused second instance, rate limiting, stale
data, deferred settlement and an operator's stop are *notes*: each is a control
or a venue rule working as designed, and a session that met one and carried on
is still evidence that the loop copes.

**Judging waits** until the close plus the bound after which a silent run is
presumed dead. Before that, a run that stopped recording a minute before the
bell could be a crash or a cycle about to land, and a verdict that changed
after it was given would be worse than one given late.

**The streak** is what the gate counts: consecutive clean sessions, newest
first, stopping at the first judged session that was not clean. A faulted or
incomplete session resets it; one still being judged neither counts nor breaks
it.

**Drills** (`tb drill`) halt the loop on purpose. The halt, the kill switch and
the watchdog trip inside a passed drill's window are the drill's, filed as
notes, and a session with a passed drill in it is judged `drill`: neither clean
nor a break, since the loop was stopped by design and not by fault. A failed
drill is a fault, and so is anything a drill does not cause — a stop that
fails to land during one is exactly what the drill exists to find.

Not attributed: `tb reconcile`, an operator's check against whichever account
its key reaches, records no mode, so its verdict cannot be pinned on one.
"""

from __future__ import annotations

import json
from bisect import bisect_left
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta
from enum import StrEnum
from typing import Any

from tb.config.hard_limits import LiveLimits
from tb.core.clock import from_iso, now_utc
from tb.data.calendar import DayKind, TradingCalendar, TradingDay
from tb.ledger.events import EventType
from tb.ledger.store import Ledger
from tb.ops.watchdog import LEASE_SECONDS

# The modes a trading run records. Anything else that records a run — a probe,
# a reconcile — is not a session of anything.
TRADING_MODES: tuple[str, ...] = ("paper", "demo", "live")

# Why a position lost its stop, when that was a failure to protect it rather
# than a stop withdrawn on purpose ahead of an exit.
_PROTECTION_FAILURES = frozenset({"no_price", "risk_refused", "placement_failed"})

# The events that settle an intent's fate: past any of them, whether an order
# exists at the broker is known.
_SETTLING = frozenset(
    {EventType.ORDER_ACKNOWLEDGED, EventType.ORDER_REJECTED, EventType.INTENT_RESOLVED}
)

_READ = frozenset(
    {
        EventType.RUN_STARTED,
        EventType.RUN_ENDED,
        EventType.LOOP_CYCLE_COMPLETED,
        EventType.HALT_RAISED,
        EventType.KILLSWITCH_ENGAGED,
        EventType.WATCHDOG_TRIPPED,
        EventType.WATCHDOG_UNREACHABLE,
        EventType.CONFIG_DRIFT_DETECTED,
        EventType.BROKER_SCHEMA_DRIFT,
        EventType.POSITION_PROTECTED,
        EventType.POSITION_UNPROTECTED,
        EventType.INTENT_COMMITTED,
        EventType.TRADE_CLOSED,
        EventType.POSITION_ORPHANED,
        EventType.INSTANCE_LOCK_REFUSED,
        EventType.BROKER_RATE_LIMITED,
        EventType.DATA_STALENESS_BREACH,
        EventType.DRILL_STARTED,
        EventType.DRILL_COMPLETED,
        *_SETTLING,
    }
)

# What a drill causes on purpose, and so what a passed drill's window excuses.
# Nothing else is excused: a protection failure mid-drill is the finding.
_DRILLED = frozenset({"halted", "halt_raised", "kill_switch", "watchdog"})


class FaultKind(StrEnum):
    """What makes a session unclean. Each one names the event it comes from."""

    HALTED = "halted"
    RUN_ERROR = "run_error"
    RUN_CRASHED = "run_crashed"
    HALT_RAISED = "halt_raised"
    KILL_SWITCH = "kill_switch"
    WATCHDOG = "watchdog"
    CONFIG_DRIFT = "config_drift"
    SCHEMA_DRIFT = "schema_drift"
    PROTECTION_FAILED = "protection_failed"
    UNPROTECTED_TOO_LONG = "unprotected_too_long"
    ORDER_UNKNOWN = "order_unknown"
    DRILL_FAILED = "drill_failed"


class NoteKind(StrEnum):
    """Worth reading, and no mark against the session."""

    STOPPED = "stopped"
    ORDER_REJECTED = "order_rejected"
    POSITION_ORPHANED = "position_orphaned"
    SECOND_INSTANCE = "second_instance"
    RATE_LIMITED = "rate_limited"
    STALE_DATA = "stale_data"
    SETTLEMENT_DEFERRED = "settlement_deferred"
    DRILL = "drill"


class SessionVerdict(StrEnum):
    CLEAN = "clean"
    FAULTED = "faulted"
    INCOMPLETE = "incomplete"
    IN_PROGRESS = "in_progress"
    DRILL = "drill"


@dataclass(frozen=True, slots=True)
class Finding:
    """One fault or note: what, when, which event said so, in which run."""

    kind: str
    at: datetime
    seq: int
    run_id: str | None
    detail: str


@dataclass(frozen=True, slots=True)
class SessionHealth:
    """One trading day in one mode, judged."""

    session_date: date
    mode: str
    open_utc: datetime
    close_utc: datetime
    half_day: bool
    verdict: SessionVerdict
    coverage_pct: float
    longest_gap_seconds: float
    n_cycles: int
    n_decisions: int
    n_orders: int
    n_fills: int
    n_trades_closed: int
    run_ids: tuple[str, ...]
    code_shas: tuple[str, ...]
    config_hashes: tuple[str, ...]
    faults: tuple[Finding, ...]
    notes: tuple[Finding, ...]

    @property
    def clean(self) -> bool:
        return self.verdict is SessionVerdict.CLEAN

    @property
    def judged(self) -> bool:
        return self.verdict is not SessionVerdict.IN_PROGRESS

    def summary(self) -> str:
        """Why, in one line."""
        if self.faults:
            first = self.faults[0]
            more = f" (+{len(self.faults) - 1} more)" if len(self.faults) > 1 else ""
            return f"{first.kind}: {first.detail}{more}"
        if self.verdict is SessionVerdict.INCOMPLETE:
            return (
                f"the loop covered {self.coverage_pct:.1f}% of the session; the longest "
                f"stretch without a cycle was {_duration(self.longest_gap_seconds)}"
            )
        if self.verdict is SessionVerdict.IN_PROGRESS:
            return "not judged until the close and the silence bound have passed"
        if self.notes:
            kinds = sorted({note.kind for note in self.notes})
            return f"{len(self.notes)} note(s): {', '.join(kinds)}"
        return ""


@dataclass(frozen=True, slots=True)
class ModeRecord:
    """Every session of one mode, oldest first, as of one moment."""

    mode: str
    as_of: datetime
    sessions: tuple[SessionHealth, ...]

    @property
    def streak(self) -> tuple[SessionHealth, ...]:
        """Consecutive clean sessions up to the newest judged one, oldest first."""
        run: list[SessionHealth] = []
        for session in reversed(self.sessions):
            if not session.judged or session.verdict is SessionVerdict.DRILL:
                continue
            if not session.clean:
                break
            run.append(session)
        return tuple(reversed(run))

    @property
    def trades_in_streak(self) -> int:
        return sum(session.n_trades_closed for session in self.streak)

    @property
    def n_clean(self) -> int:
        return sum(1 for session in self.sessions if session.clean)

    def session(self, day: date) -> SessionHealth | None:
        for session in self.sessions:
            if session.session_date == day:
                return session
        return None


# --------------------------------------------------------------------------
# Reading
# --------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class _Event:
    seq: int
    at: datetime
    type: EventType
    run_id: str | None
    payload: dict[str, Any]


@dataclass(slots=True)
class _Run:
    run_id: str
    mode: str
    started_at: datetime
    code_sha: str | None
    config_hash: str | None
    last_seen: datetime
    last_seq: int
    ended_at: datetime | None = None

    def alive(self, now: datetime, bound: timedelta) -> bool:
        """Unended and recently heard from: a run that may still be working."""
        return self.ended_at is None and now - self.last_seen <= bound

    def up_until(self, now: datetime, bound: timedelta) -> datetime:
        """The last instant the run was demonstrably up: its end, now, or its last word."""
        if self.ended_at is not None:
            return self.ended_at
        return now if self.alive(now, bound) else self.last_seen

    def presumed_until(self, now: datetime, bound: timedelta) -> datetime:
        """The last instant the run may have been up, for events that name no run.

        Longer than `up_until` for a run that died silently: a wedged loop
        writes nothing, and the watchdog trip that its silence causes belongs
        to it rather than to nobody.
        """
        if self.ended_at is not None:
            return self.ended_at
        return min(now, self.last_seen + bound)


@dataclass(slots=True)
class _Bucket:
    day: TradingDay
    runs: set[str] = field(default_factory=set)
    n_cycles: int = 0
    n_decisions: int = 0
    n_orders: int = 0
    n_fills: int = 0
    n_trades_closed: int = 0
    faults: list[Finding] = field(default_factory=list)
    notes: list[Finding] = field(default_factory=list)
    drilled: bool = False


@dataclass(frozen=True, slots=True)
class _Drill:
    drill_id: str
    kind: str
    run_id: str
    started_at: datetime
    completed_at: datetime | None
    passed: bool

    def covers(self, run_id: str, at: datetime) -> bool:
        return (
            self.passed
            and self.completed_at is not None
            and run_id == self.run_id
            and self.started_at <= at <= self.completed_at
        )


def _drills(events: Sequence[_Event]) -> list[_Drill]:
    started: dict[str, _Event] = {}
    completed: dict[str, _Event] = {}
    for event in events:
        if event.type is EventType.DRILL_STARTED:
            started[str(event.payload["drill_id"])] = event
        elif event.type is EventType.DRILL_COMPLETED:
            completed[str(event.payload["drill_id"])] = event
    return [
        _Drill(
            drill_id=drill_id,
            kind=str(start.payload.get("kind")),
            run_id=str(start.payload.get("run_id")),
            started_at=start.at,
            completed_at=completed[drill_id].at if drill_id in completed else None,
            passed=drill_id in completed and bool(completed[drill_id].payload.get("passed")),
        )
        for drill_id, start in started.items()
    ]


def read_sessions(
    ledger: Ledger,
    *,
    limits: LiveLimits,
    mode: str,
    calendar: TradingCalendar | None = None,
    now: datetime | None = None,
    through_seq: int | None = None,
) -> ModeRecord:
    """Every session of `mode` in the ledger, judged as of `now`."""
    return read_all_sessions(
        ledger,
        limits=limits,
        modes=(mode,),
        calendar=calendar,
        now=now,
        through_seq=through_seq,
    )[mode]


def read_all_sessions(
    ledger: Ledger,
    *,
    limits: LiveLimits,
    modes: Sequence[str] = TRADING_MODES,
    calendar: TradingCalendar | None = None,
    now: datetime | None = None,
    through_seq: int | None = None,
) -> dict[str, ModeRecord]:
    """Every session of each mode, from one pass over the ledger.

    `through_seq` reads the ledger as it stood at that event and no later, so
    a record computed from it can be computed again, identically, tomorrow.
    """
    unknown = [mode for mode in modes if mode not in TRADING_MODES]
    if unknown:
        raise ValueError(f"no such trading mode: {unknown[0]!r}; one of {', '.join(TRADING_MODES)}")
    events = [_parse(row) for row in ledger.iter_events(end_seq=through_seq, event_types=_READ)]
    runs = _runs(events)
    drills = _drills(events)
    cal = calendar or TradingCalendar()
    moment = now or now_utc()
    return {
        mode: _record(mode, events, runs, drills, limits=limits, cal=cal, moment=moment)
        for mode in modes
    }


def silence_bound(limits: LiveLimits) -> timedelta:
    """How long a run may be silent before it is presumed dead.

    Never shorter than the lease: until the lease lapses, the run may still
    hold it. A session is judged once its close is this far behind.
    """
    return max(
        timedelta(seconds=limits.session_max_cycle_gap_seconds),
        timedelta(seconds=LEASE_SECONDS),
    )


def _record(
    mode: str,
    events: Sequence[_Event],
    runs: dict[str, _Run],
    drills: Sequence[_Drill],
    *,
    limits: LiveLimits,
    cal: TradingCalendar,
    moment: datetime,
) -> ModeRecord:
    reach = timedelta(seconds=limits.session_max_cycle_gap_seconds)
    silence = silence_bound(limits)
    ordered = sorted(runs.values(), key=lambda r: (r.started_at, r.run_id))

    def owner(event: _Event) -> _Run | None:
        """The trading run an event belongs to: by id, else the one up at the time."""
        if event.run_id is not None and event.run_id in runs:
            return runs[event.run_id]
        up = [r for r in ordered if r.started_at <= event.at <= r.presumed_until(moment, silence)]
        return up[-1] if up else None

    buckets: dict[date, _Bucket] = {}

    def bucket(at: datetime) -> _Bucket | None:
        """The session `at` belongs to, brought into being: for a fault."""
        day = session_of(cal, at)
        if day is None:
            return None
        return buckets.setdefault(day.day, _Bucket(day=day))

    # The sessions each run of this mode was up for during regular hours,
    # whether or not it wrote anything in them: a run wedged all day is a
    # session with no coverage. Regular hours rather than the whole window,
    # or a run stopped ten minutes after the bell would bring the next
    # session into being, empty, and have it judged incomplete.
    for run in ordered:
        if run.mode != mode:
            continue
        first = session_of(cal, run.started_at)
        if first is None:
            continue
        until = run.up_until(moment, silence)
        last = session_of(cal, until)
        through = last.day if last is not None else cal.coverage[1]
        for day in cal.sessions_between(first.day, max(first.day, through)):
            assert day.open_utc is not None and day.close_utc is not None
            if run.started_at < day.close_utc and until >= day.open_utc:
                buckets.setdefault(day.day, _Bucket(day=day)).runs.add(run.run_id)

    cycle_times: list[datetime] = []
    committed: dict[str, tuple[_Event, _Run]] = {}
    settled: dict[str, datetime] = {}

    for event in events:
        if event.type in _SETTLING:
            settled.setdefault(str(event.payload.get("intent_id", "")), event.at)
            # A rejection settles its intent and is also worth a note.
            if event.type is not EventType.ORDER_REJECTED:
                continue
        who = owner(event)
        if who is None or who.mode != mode:
            continue
        if event.type is EventType.INTENT_COMMITTED:
            committed[str(event.payload["intent_id"])] = (event, who)
            continue
        if event.type is EventType.LOOP_CYCLE_COMPLETED:
            cycle_times.append(event.at)
        home = session_of(cal, event.at)
        if home is None:
            continue
        sink = buckets.get(home.day) or _Bucket(day=home)
        _attribute(event, who, sink, limits=limits, drills=drills)
        # Counts and notes land only in a session the mode ran; a fault brings
        # its session into being, so a crash on a Sunday breaks Monday even if
        # nothing ran on Monday.
        if home.day in buckets or sink.faults:
            buckets[home.day] = sink
            sink.runs.add(who.run_id)

    # A run that never wrote its end and has gone quiet did not stop: it died.
    for run in ordered:
        if run.mode != mode or run.ended_at is not None or run.alive(moment, silence):
            continue
        crashed = bucket(run.last_seen)
        if crashed is not None:
            crashed.faults.append(
                Finding(
                    kind=FaultKind.RUN_CRASHED,
                    at=run.last_seen,
                    seq=run.last_seq,
                    run_id=run.run_id,
                    detail=(
                        f"run {run.run_id} stopped recording at {run.last_seen.isoformat()} "
                        "and never recorded an end: a crash, a kill -9 or a lost host"
                    ),
                )
            )

    # An order is known once the broker answered or recovery looked. One still
    # unknown past the silence bound is an order that may exist with nothing
    # accounting for it.
    for intent_id, (event, run) in committed.items():
        deadline = event.at + reach
        answered = settled.get(intent_id)
        if answered is not None and answered <= deadline:
            continue
        if answered is None and moment <= deadline:
            continue
        unknown = bucket(event.at)
        if unknown is None:
            continue
        purpose = event.payload.get("purpose", "")
        ticker = event.payload.get("t212_ticker", "")
        how_long = (
            f"stayed unknown for {_duration((answered - event.at).total_seconds())}"
            if answered is not None
            else "is still unknown"
        )
        unknown.faults.append(
            Finding(
                kind=FaultKind.ORDER_UNKNOWN,
                at=event.at,
                seq=event.seq,
                run_id=run.run_id,
                detail=(
                    f"{purpose} order {intent_id} on {ticker} {how_long}; the bound is "
                    f"{_duration(reach.total_seconds())}"
                ),
            )
        )

    cycle_times.sort()
    sessions = tuple(
        _judge(
            buckets[key],
            mode=mode,
            runs=runs,
            cycles=cycle_times,
            limits=limits,
            reach=reach,
            silence=silence,
            now=moment,
        )
        for key in sorted(buckets)
    )
    return ModeRecord(mode=mode, as_of=moment, sessions=sessions)


def session_of(calendar: TradingCalendar, at: datetime) -> TradingDay | None:
    """The first session whose close is at or after `at`.

    `None` past the calendar's reach, which the caller drops: a day the
    calendar cannot classify cannot be a clean session either.
    """
    day = calendar.day_of(at)
    if day.is_trading_day and day.close_utc is not None and at <= day.close_utc:
        return day
    return calendar.next_session(day.day)


def coverage(
    cycles: Sequence[datetime], *, start: datetime, end: datetime, reach: timedelta
) -> tuple[float, float]:
    """Seconds of `[start, end)` within `reach` after some cycle, and the longest gap.

    `cycles` must be sorted. A cycle a little before `start` covers the first
    minutes after it, which is how a loop that was already running at the open
    is credited with the open.
    """
    if end <= start:
        return 0.0, 0.0
    covered = 0.0
    longest = 0.0
    cursor = start
    for at in cycles[bisect_left(cycles, start - reach) :]:
        if at >= end:
            break
        until = min(at + reach, end)
        if until <= cursor:
            continue
        begins = max(at, cursor)
        if begins > cursor:
            longest = max(longest, (begins - cursor).total_seconds())
        covered += (until - begins).total_seconds()
        cursor = until
    longest = max(longest, (end - cursor).total_seconds())
    return covered, longest


# --------------------------------------------------------------------------
# Internals
# --------------------------------------------------------------------------


def _parse(row: Any) -> _Event:
    payload = json.loads(row["payload_json"])
    run_id = payload.get("run_id") or row["run_id"]
    return _Event(
        seq=int(row["seq"]),
        at=from_iso(str(row["ts_utc"])),
        type=EventType(row["event_type"]),
        run_id=str(run_id) if run_id else None,
        payload=payload,
    )


def _runs(events: Iterable[_Event]) -> dict[str, _Run]:
    """Every trading run, with when it started, ended and was last heard from."""
    runs: dict[str, _Run] = {}
    for event in events:
        if event.type is EventType.RUN_STARTED:
            mode = str(event.payload.get("mode", ""))
            if mode in TRADING_MODES and event.run_id is not None:
                runs[event.run_id] = _Run(
                    run_id=event.run_id,
                    mode=mode,
                    started_at=event.at,
                    code_sha=event.payload.get("code_git_sha"),
                    config_hash=event.payload.get("config_hash"),
                    last_seen=event.at,
                    last_seq=event.seq,
                )
            continue
        run = runs.get(event.run_id) if event.run_id is not None else None
        if run is None:
            continue
        if event.at >= run.last_seen:
            run.last_seen = event.at
            run.last_seq = event.seq
        if event.type is EventType.RUN_ENDED:
            run.ended_at = event.at
    return runs


def _attribute(
    event: _Event,
    run: _Run,
    target: _Bucket,
    *,
    limits: LiveLimits,
    drills: Sequence[_Drill] = (),
) -> None:
    """File one event under its session: a count, a fault or a note."""
    p = event.payload

    def fault(kind: FaultKind, detail: str) -> None:
        if kind.value in _DRILLED:
            drill = next((d for d in drills if d.covers(run.run_id, event.at)), None)
            if drill is not None:
                target.drilled = True
                note(
                    NoteKind.DRILL, f"{kind.value} in {drill.kind} drill {drill.drill_id}: {detail}"
                )
                return
        target.faults.append(
            Finding(kind=kind, at=event.at, seq=event.seq, run_id=run.run_id, detail=detail)
        )

    def note(kind: NoteKind, detail: str) -> None:
        target.notes.append(
            Finding(kind=kind, at=event.at, seq=event.seq, run_id=run.run_id, detail=detail)
        )

    match event.type:
        case EventType.LOOP_CYCLE_COMPLETED:
            target.n_cycles += 1
            target.n_decisions += int(p.get("n_decisions", 0))
            target.n_orders += int(p.get("n_orders_submitted", 0))
            target.n_fills += int(p.get("n_fills_recorded", 0))
            if p.get("detail"):
                note(NoteKind.SETTLEMENT_DEFERRED, f"cycle {p.get('cycle')}: {p['detail']}")
        case EventType.RUN_ENDED:
            reason = str(p.get("exit_reason", ""))
            if reason == "halted":
                fault(FaultKind.HALTED, str(p.get("error_detail") or "the loop halted"))
            elif reason == "error":
                fault(FaultKind.RUN_ERROR, f"{p.get('error_type')}: {p.get('error_detail')}")
            elif reason == "interrupted":
                note(NoteKind.STOPPED, f"run {run.run_id} was stopped by the operator")
        case EventType.HALT_RAISED:
            fault(FaultKind.HALT_RAISED, f"[{p.get('trigger')}] {p.get('detail')}")
        case EventType.KILLSWITCH_ENGAGED:
            who = p.get("engaged_by") or "unknown"
            fault(FaultKind.KILL_SWITCH, f"engaged by {who}: {p.get('detail')}")
        case EventType.WATCHDOG_TRIPPED:
            fault(FaultKind.WATCHDOG, f"watchdog {p.get('action_taken')}: {p.get('detail')}")
        case EventType.WATCHDOG_UNREACHABLE:
            fault(FaultKind.WATCHDOG, f"self-check {p.get('direction')}: {p.get('detail')}")
        case EventType.CONFIG_DRIFT_DETECTED:
            fault(FaultKind.CONFIG_DRIFT, str(p.get("detail")))
        case EventType.BROKER_SCHEMA_DRIFT:
            fault(
                FaultKind.SCHEMA_DRIFT,
                f"{p.get('endpoint')} {p.get('model')}: {p.get('error_detail')}",
            )
        case EventType.POSITION_UNPROTECTED:
            cause = p.get("cause")
            detail = str(p.get("detail", ""))
            # Before `cause` existed only the detail said which it was, and a
            # withdrawal's detail has always begun this way.
            failed = (
                cause in _PROTECTION_FAILURES
                if cause is not None
                else not detail.startswith("stop withdrawn")
            )
            if failed:
                fault(FaultKind.PROTECTION_FAILED, f"{p.get('t212_ticker')}: {detail}")
        case EventType.POSITION_PROTECTED:
            bare = p.get("unprotected_seconds")
            if bare is not None and float(bare) > limits.max_unprotected_seconds:
                fault(
                    FaultKind.UNPROTECTED_TOO_LONG,
                    f"{p.get('t212_ticker')} held without a stop for "
                    f"{_duration(float(bare))}; the bound is "
                    f"{_duration(limits.max_unprotected_seconds)}",
                )
        case EventType.TRADE_CLOSED:
            target.n_trades_closed += 1
        case EventType.DRILL_COMPLETED:
            if p.get("passed"):
                target.drilled = True
                note(NoteKind.DRILL, f"{p.get('kind')} drill {p.get('drill_id')} passed")
            else:
                fault(
                    FaultKind.DRILL_FAILED,
                    f"{p.get('kind')} drill {p.get('drill_id')} failed: "
                    + "; ".join(str(f) for f in p.get("failures", [])),
                )
        case EventType.ORDER_REJECTED:
            said = p.get("broker_message")
            note(
                NoteKind.ORDER_REJECTED,
                f"{p.get('t212_ticker')}: {p.get('detail')}" + (f" ({said})" if said else ""),
            )
        case EventType.POSITION_ORPHANED:
            note(
                NoteKind.POSITION_ORPHANED,
                f"{p.get('t212_ticker')}: {p.get('reason')}; {p.get('action_taken')}",
            )
        case EventType.INSTANCE_LOCK_REFUSED:
            note(
                NoteKind.SECOND_INSTANCE,
                f"a second instance (run {p.get('run_id')}) was refused the lease",
            )
        case EventType.BROKER_RATE_LIMITED:
            note(NoteKind.RATE_LIMITED, f"{p.get('endpoint')} answered {p.get('status_code')}")
        case EventType.DATA_STALENESS_BREACH:
            note(
                NoteKind.STALE_DATA,
                f"{p.get('instrument_uid')} {p.get('resolution')}: {p.get('action')}",
            )
        case _:
            pass


def _judge(
    bucket: _Bucket,
    *,
    mode: str,
    runs: dict[str, _Run],
    cycles: Sequence[datetime],
    limits: LiveLimits,
    reach: timedelta,
    silence: timedelta,
    now: datetime,
) -> SessionHealth:
    day = bucket.day
    assert day.open_utc is not None and day.close_utc is not None  # trading days have both
    # Coverage so far while the session runs; of the whole session once closed.
    end = min(day.close_utc, max(now, day.open_utc))
    covered, longest = coverage(cycles, start=day.open_utc, end=end, reach=reach)
    span = (end - day.open_utc).total_seconds()
    pct = 100.0 * covered / span if span > 0 else 0.0

    faults = tuple(sorted(bucket.faults, key=lambda f: (f.at, f.seq)))
    if faults:
        verdict = SessionVerdict.FAULTED
    elif now < day.close_utc + silence:
        verdict = SessionVerdict.IN_PROGRESS
    elif bucket.drilled:
        verdict = SessionVerdict.DRILL
    elif pct < limits.session_min_coverage_pct:
        verdict = SessionVerdict.INCOMPLETE
    else:
        verdict = SessionVerdict.CLEAN

    up = [runs[run_id] for run_id in sorted(bucket.runs) if run_id in runs]
    return SessionHealth(
        session_date=day.day,
        mode=mode,
        open_utc=day.open_utc,
        close_utc=day.close_utc,
        half_day=day.kind is DayKind.HALF_DAY,
        verdict=verdict,
        coverage_pct=pct,
        longest_gap_seconds=longest,
        n_cycles=bucket.n_cycles,
        n_decisions=bucket.n_decisions,
        n_orders=bucket.n_orders,
        n_fills=bucket.n_fills,
        n_trades_closed=bucket.n_trades_closed,
        run_ids=tuple(run.run_id for run in up),
        code_shas=tuple(sorted({run.code_sha for run in up if run.code_sha})),
        config_hashes=tuple(sorted({run.config_hash for run in up if run.config_hash})),
        faults=faults,
        notes=tuple(sorted(bucket.notes, key=lambda n: (n.at, n.seq))),
    )


def _duration(seconds: float) -> str:
    whole = round(seconds)
    if whole < 120:
        return f"{whole}s"
    if whole < 7200:
        return f"{whole // 60}m{whole % 60:02d}s"
    return f"{whole // 3600}h{(whole % 3600) // 60:02d}m"
