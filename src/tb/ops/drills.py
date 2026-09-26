"""Drills: the kill switch and the watchdog, fired on purpose, with money at risk.

The plan's M8 verification is "kill-switch and watchdog drills run during
market hours with open positions". Both conditions are the point. A kill
switch tested on a quiet evening proves a file is read; one tested with a
position open in a moving market proves what matters — that stopping the bot
does not leave what it holds unprotected. So a drill refuses to start without
them, and what it records is what it saw at the broker, before and after.

**The kill-switch drill** engages the switch the way `tb halt` does. The loop
must halt at its next cycle and send nothing after the engagement, and every
position must still have its broker-side stop.

**The watchdog drill** freezes the trading process (SIGSTOP) just after a
cycle completes, so its heartbeat goes stale the way a wedged process's does.
The watchdog — a separate process — must notice and engage the switch. Then
the process is thawed (SIGCONT) and must halt on the switch at its first
cycle. If the watchdog never notices, the drill engages the switch itself
before thawing: a failed drill must not hand a frozen loop back to a live
market with nothing stopping it. The process is thawed whatever happens.

Neither drill restarts trading. A drill ends with the switch engaged and the
loop stopped; resuming (`tb resume`, then `tb run`) is a person's decision,
as it is after any halt.

Drills run against the demo account only: the evidence the live gate wants is
that the mechanisms work before real money depends on them.
"""

from __future__ import annotations

import json
import os
import signal
import socket
import time
import uuid
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from decimal import Decimal
from enum import StrEnum
from pathlib import Path
from typing import Any, TypeVar

from tb.broker.port import OrderType, ReadOnlyBroker, Side
from tb.core.clock import from_iso, now_utc
from tb.core.errors import TbError
from tb.data.calendar import TradingCalendar
from tb.ledger.events import (
    Actor,
    DrillCompletedPayload,
    DrillStartedPayload,
    EventType,
    KillswitchPayload,
)
from tb.ledger.store import Ledger
from tb.ops.killswitch import engage_kill_switch, read_kill_switch
from tb.ops.watchdog import LOCK_NAME

DRILL_MODES = frozenset({"demo"})

_T = TypeVar("_T")


class DrillKind(StrEnum):
    KILL_SWITCH = "kill_switch"
    WATCHDOG = "watchdog"


class DrillError(TbError):
    """A drill could not start, or could not be run safely."""


@dataclass(frozen=True, slots=True)
class Holding:
    """A position, with the quantity its working stops at the broker cover."""

    ticker: str
    quantity: Decimal
    covered: Decimal

    @property
    def protected(self) -> bool:
        return self.covered >= self.quantity

    def as_dict(self) -> dict[str, str]:
        return {
            "ticker": self.ticker,
            "quantity": str(self.quantity),
            "covered": str(self.covered),
        }


def holdings(broker: ReadOnlyBroker) -> tuple[Holding, ...]:
    """Each held position, as the broker reports it, and its stop coverage."""
    orders = broker.get_open_orders()
    found: list[Holding] = []
    for position in sorted(broker.get_positions(), key=lambda p: p.ticker):
        if position.quantity <= 0:
            continue
        covered = sum(
            (
                (order.quantity or Decimal(0)) - (order.filled_quantity or Decimal(0))
                for order in orders
                if order.ticker == position.ticker
                and order.side is Side.SELL
                and order.order_type in (OrderType.STOP, OrderType.STOP_LIMIT)
                and order.status.is_open
            ),
            Decimal(0),
        )
        found.append(Holding(position.ticker, position.quantity, covered))
    return tuple(found)


@dataclass(frozen=True, slots=True)
class Target:
    """The trading run a drill fires at: the holder of the live lease."""

    run_id: str
    mode: str
    pid: int
    host: str


def running_target(ledger: Ledger, *, now: datetime) -> Target | None:
    row = ledger.conn.execute(
        "SELECT run_id, pid, host, expires_at, released_at FROM instance_locks WHERE lock_name = ?",
        (LOCK_NAME,),
    ).fetchone()
    if row is None or row["released_at"] is not None:
        return None
    if from_iso(str(row["expires_at"])) <= now:
        return None
    run_id = str(row["run_id"])
    mode = ledger.conn.execute("SELECT mode FROM runs WHERE run_id = ?", (run_id,)).fetchone()
    return Target(
        run_id=run_id,
        mode="" if mode is None else str(mode["mode"]),
        pid=int(row["pid"]),
        host=str(row["host"]),
    )


def _stop_process(pid: int) -> None:
    os.kill(pid, signal.SIGSTOP)


def _continue_process(pid: int) -> None:
    os.kill(pid, signal.SIGCONT)


@dataclass
class DrillWorld:
    """Everything a drill touches, so each piece can be supplied."""

    ledger: Ledger
    broker: ReadOnlyBroker
    kill_switch_path: Path
    heartbeat_path: Path
    calendar: TradingCalendar = field(default_factory=TradingCalendar)
    clock: Callable[[], datetime] = now_utc
    wait: Callable[[float], None] = time.sleep
    suspend: Callable[[int], None] = _stop_process
    resume: Callable[[int], None] = _continue_process
    host: str = field(default_factory=socket.gethostname)
    poll_seconds: float = 1.0


@dataclass(frozen=True, slots=True)
class DrillResult:
    drill_id: str
    kind: DrillKind
    run_id: str
    passed: bool
    halted_after_seconds: float | None
    observations: tuple[str, ...]
    failures: tuple[str, ...]
    before: tuple[Holding, ...]
    after: tuple[Holding, ...]


def run_drill(
    world: DrillWorld,
    kind: DrillKind,
    *,
    halt_within: timedelta,
    notice_within: timedelta,
    cycling_within: timedelta,
) -> DrillResult:
    """Check the preconditions, fire the mechanism, and record what happened."""
    target, before = _preconditions(world, kind, cycling_within=cycling_within)
    if kind is DrillKind.WATCHDOG:
        # Freeze between cycles, never inside one: a process frozen holding the
        # ledger's write lock would take the watchdog's record down with it.
        seen = _cycle_count(world.ledger, target.run_id)
        fresh = _wait_for(
            world,
            lambda: True if _cycle_count(world.ledger, target.run_id) > seen else None,
            cycling_within,
        )
        if fresh is None:
            raise DrillError(
                f"run {target.run_id} did not complete a cycle to freeze after; it is not "
                "cycling, so there is nothing for the drill to stop."
            )
    drill_id = f"drl_{uuid.uuid4().hex[:12]}"
    now = world.clock()
    world.ledger.append(
        EventType.DRILL_STARTED,
        drill_id,
        DrillStartedPayload(
            drill_id=drill_id,
            kind=kind.value,
            run_id=target.run_id,
            mode=target.mode,
            pid=target.pid,
            host=target.host,
            session_date=world.calendar.day_of(now).day.isoformat(),
            market_open=world.calendar.is_open_at(now),
            holdings=[h.as_dict() for h in before],
        ),
        actor=Actor.HUMAN,
    )

    observations: list[str] = []
    failures: list[str] = []
    fired_at = world.clock()
    if kind is DrillKind.KILL_SWITCH:
        _engage(world, drill_id, reason="kill-switch drill")
        observations.append("engaged the kill switch")
    else:
        fired_at = _freeze_and_wait(world, target, drill_id, notice_within, observations, failures)

    halted_after: float | None = None
    ended = _wait_for(world, lambda: _run_end(world.ledger, target.run_id), halt_within)
    if ended is None:
        failures.append(
            f"the loop did not stop within {halt_within.total_seconds():.0f}s of the switch"
        )
    else:
        ended_at, reason, detail = ended
        halted_after = (ended_at - fired_at).total_seconds()
        if reason == "halted":
            observations.append(f"the loop halted {halted_after:.0f}s later: {detail}")
        else:
            failures.append(f"the loop ended '{reason}' rather than halting on the switch")

    sent = _orders_since(world.ledger, target.run_id, fired_at)
    if sent:
        failures.append(f"{sent} order(s) were sent after the switch was engaged")
    else:
        observations.append("no order was sent after the switch")

    after = holdings(world.broker)
    for holding in after:
        if not holding.protected:
            failures.append(
                f"{holding.ticker}: {holding.quantity} held, only {holding.covered} covered by "
                "a working stop after the drill"
            )
    gone = sorted({h.ticker for h in before} - {h.ticker for h in after})
    if gone:
        # A stop that fired while the loop was stopped is protection working.
        observations.append(f"no longer held after the drill: {', '.join(gone)}")
    if after and all(h.protected for h in after):
        observations.append(f"all {len(after)} position(s) still protected at the broker")

    passed = not failures
    world.ledger.append(
        EventType.DRILL_COMPLETED,
        drill_id,
        DrillCompletedPayload(
            drill_id=drill_id,
            kind=kind.value,
            run_id=target.run_id,
            passed=passed,
            halted_after_seconds=halted_after,
            failures=failures,
            observations=observations,
            holdings_after=[h.as_dict() for h in after],
        ),
        actor=Actor.HUMAN,
    )
    return DrillResult(
        drill_id=drill_id,
        kind=kind,
        run_id=target.run_id,
        passed=passed,
        halted_after_seconds=halted_after,
        observations=tuple(observations),
        failures=tuple(failures),
        before=before,
        after=after,
    )


def _preconditions(
    world: DrillWorld, kind: DrillKind, *, cycling_within: timedelta
) -> tuple[Target, tuple[Holding, ...]]:
    now = world.clock()
    target = running_target(world.ledger, now=now)
    if target is None:
        raise DrillError(
            "no trading run holds the lease. A drill fires at a running loop: start "
            "`tb run --mode demo` and let it take a position first."
        )
    if target.mode not in DRILL_MODES:
        raise DrillError(
            f"run {target.run_id} is in {target.mode or 'an unrecorded'} mode. Drills run "
            "against the demo account, before real money depends on what they prove."
        )
    if not world.calendar.is_open_at(now):
        raise DrillError(
            "the market is closed. A drill proves what happens with the market moving, "
            "so it runs during regular hours."
        )
    last = _last_cycle(world.ledger, target.run_id)
    if last is None or now - last > cycling_within:
        raise DrillError(
            f"run {target.run_id} has not completed a cycle in the last "
            f"{cycling_within.total_seconds():.0f}s, so it is not trading and there is "
            "nothing for the drill to stop."
        )
    before = holdings(world.broker)
    if not before:
        raise DrillError(
            "no position is held. The drill's point is that stopping the bot leaves what "
            "it holds protected; with nothing held it proves nothing. Let the loop open a "
            "position first."
        )
    bare = [h for h in before if not h.protected]
    if bare:
        raise DrillError(
            "not fully protected before the drill: "
            + ", ".join(f"{h.ticker} ({h.covered} of {h.quantity})" for h in bare)
            + ". That is a fault to fix, not a drill to run."
        )
    if kind is DrillKind.WATCHDOG:
        if target.host != world.host:
            raise DrillError(
                f"the loop runs on {target.host}. The watchdog drill freezes the process, "
                "so it runs on the host the process does."
            )
        beat = _heartbeat(world.heartbeat_path)
        if beat.get("run_id") != target.run_id or beat.get("pid") != target.pid:
            raise DrillError(
                f"the heartbeat at {world.heartbeat_path} is not the running loop's "
                f"(it names run {beat.get('run_id')}, pid {beat.get('pid')}). Freezing a "
                "process the watchdog is not watching would prove nothing."
            )
    return target, before


def _freeze_and_wait(
    world: DrillWorld,
    target: Target,
    drill_id: str,
    notice_within: timedelta,
    observations: list[str],
    failures: list[str],
) -> datetime:
    """Freeze the loop, wait for the watchdog to notice, and always thaw it."""
    world.suspend(target.pid)
    frozen_at = world.clock()
    observations.append(f"froze the loop (pid {target.pid}) just after a cycle")
    try:
        tripped = _wait_for(
            world, lambda: _watchdog_tripped_since(world.ledger, frozen_at), notice_within
        )
        if tripped is None:
            failures.append(
                f"the watchdog did not notice the frozen loop within "
                f"{notice_within.total_seconds():.0f}s"
            )
        else:
            observations.append(
                f"the watchdog noticed after {(tripped - frozen_at).total_seconds():.0f}s"
            )
        if read_kill_switch(world.kill_switch_path).may_trade:
            _engage(world, drill_id, reason="watchdog drill: the watchdog did not engage it")
            failures.append(
                "the kill switch was not engaged when the loop was thawed; the drill "
                "engaged it rather than hand a frozen loop back to the market"
            )
        else:
            observations.append("the kill switch was engaged")
    finally:
        world.resume(target.pid)
        observations.append("thawed the loop")
    return frozen_at


def _engage(world: DrillWorld, drill_id: str, *, reason: str) -> None:
    engaged_by = f"drill {drill_id}"
    reading = engage_kill_switch(world.kill_switch_path, engaged_by=engaged_by, reason=reason)
    world.ledger.append(
        EventType.KILLSWITCH_ENGAGED,
        str(world.kill_switch_path),
        KillswitchPayload(
            path=str(world.kill_switch_path),
            determinable=reading.determinable,
            detail=reading.detail,
            engaged_by=engaged_by,
        ),
        actor=Actor.HUMAN,
    )


def _wait_for(world: DrillWorld, probe: Callable[[], _T | None], within: timedelta) -> _T | None:
    """Poll until `probe` answers or `within` passes on the world's clock."""
    deadline = world.clock() + within
    while True:
        answer = probe()
        if answer is not None:
            return answer
        if world.clock() >= deadline:
            return None
        world.wait(world.poll_seconds)


def _heartbeat(path: Path) -> dict[str, Any]:
    try:
        beat: dict[str, Any] = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}
    return beat


def _events_for_run(ledger: Ledger, event_type: EventType, run_id: str) -> list[Any]:
    return list(
        ledger.conn.execute(
            "SELECT seq, ts_utc, payload_json FROM event_log WHERE event_type = ?"
            " AND run_id = ? ORDER BY seq",
            (event_type.value, run_id),
        ).fetchall()
    )


def _last_cycle(ledger: Ledger, run_id: str) -> datetime | None:
    rows = _events_for_run(ledger, EventType.LOOP_CYCLE_COMPLETED, run_id)
    return from_iso(str(rows[-1]["ts_utc"])) if rows else None


def _cycle_count(ledger: Ledger, run_id: str) -> int:
    return len(_events_for_run(ledger, EventType.LOOP_CYCLE_COMPLETED, run_id))


def _run_end(ledger: Ledger, run_id: str) -> tuple[datetime, str, str] | None:
    rows = _events_for_run(ledger, EventType.RUN_ENDED, run_id)
    if not rows:
        return None
    payload = json.loads(rows[-1]["payload_json"])
    return (
        from_iso(str(rows[-1]["ts_utc"])),
        str(payload.get("exit_reason", "")),
        str(payload.get("error_detail") or ""),
    )


def _orders_since(ledger: Ledger, run_id: str, since: datetime) -> int:
    return sum(
        1
        for row in _events_for_run(ledger, EventType.ORDER_SUBMITTED, run_id)
        if from_iso(str(row["ts_utc"])) > since
    )


def _watchdog_tripped_since(ledger: Ledger, since: datetime) -> datetime | None:
    for row in ledger.iter_events(event_type=EventType.WATCHDOG_TRIPPED):
        at = from_iso(str(row["ts_utc"]))
        if at >= since:
            return at
    return None


# --------------------------------------------------------------------------
# Reading drills back
# --------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class DrillRecord:
    drill_id: str
    kind: str
    run_id: str
    mode: str
    market_open: bool
    n_holdings: int
    started_at: datetime
    completed_at: datetime | None
    passed: bool
    failures: tuple[str, ...] = ()


def drills(ledger: Ledger) -> list[DrillRecord]:
    """Every drill recorded, oldest first. One never completed has not passed."""
    started: dict[str, tuple[datetime, dict[str, Any]]] = {}
    completed: dict[str, tuple[datetime, dict[str, Any]]] = {}
    for row in ledger.iter_events(event_types=(EventType.DRILL_STARTED, EventType.DRILL_COMPLETED)):
        payload = json.loads(row["payload_json"])
        target = started if row["event_type"] == EventType.DRILL_STARTED.value else completed
        target[str(payload["drill_id"])] = (from_iso(str(row["ts_utc"])), payload)
    records: list[DrillRecord] = []
    for drill_id, (at, payload) in started.items():
        done = completed.get(drill_id)
        records.append(
            DrillRecord(
                drill_id=drill_id,
                kind=str(payload["kind"]),
                run_id=str(payload["run_id"]),
                mode=str(payload["mode"]),
                market_open=bool(payload["market_open"]),
                n_holdings=len(payload.get("holdings", [])),
                started_at=at,
                completed_at=None if done is None else done[0],
                passed=done is not None and bool(done[1]["passed"]),
                failures=() if done is None else tuple(done[1].get("failures", [])),
            )
        )
    return sorted(records, key=lambda r: r.started_at)
