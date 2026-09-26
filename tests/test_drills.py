"""Drills: the kill switch and the watchdog fired at a real loop holding a position.

A harness stands in for time passing and for the other processes. Its `wait`
advances one pinned clock that the loop, the simulated broker, the ledger
and the heartbeat all read; runs a loop cycle whenever an interval has gone
by (unless the loop is frozen or has stopped); and runs a watchdog check the
way `tb watchdog` does. So the loop that halts here is the real `TradingLoop`
reading the real kill switch file, and the watchdog is the real `Watchdog`
reading the real heartbeat's age.
"""

from __future__ import annotations

import os
import socket
from collections.abc import Iterator
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from decimal import Decimal
from pathlib import Path
from typing import Any

import pytest
from typer.testing import CliRunner

from tb.broker.simulated import SimulatedBroker
from tb.cli import app
from tb.core.clock import to_iso
from tb.engine.loop import LoopHalted, TradingLoop
from tb.ledger.store import Ledger
from tb.ops.drills import (
    DrillError,
    DrillKind,
    DrillWorld,
    drills,
    holdings,
    run_drill,
)
from tb.ops.journal import render_page
from tb.ops.killswitch import engage_kill_switch, read_kill_switch
from tb.ops.sessions import NoteKind, SessionVerdict, read_sessions
from tb.ops.watchdog import Watchdog
from tests.test_loop import AS_OF, TICKER, _broker, _loop, _rising_bars, _seed
from tests.test_protection import _book

INTERVAL = 60.0
HALT_WITHIN = timedelta(minutes=5)
NOTICE_WITHIN = timedelta(minutes=5)
CYCLING_WITHIN = timedelta(minutes=15)


@dataclass
class Harness:
    """Time, the loop and the watchdog, advanced together."""

    env: dict[str, Any]
    ledger: Ledger
    broker: SimulatedBroker
    loop: TradingLoop
    watchdog: Watchdog | None
    at: datetime = AS_OF
    frozen: bool = False
    stopped: bool = False
    in_episode: bool = False
    since_cycle: float = 0.0
    suspended: list[int] = field(default_factory=list)
    resumed: list[int] = field(default_factory=list)

    def now(self) -> datetime:
        return self.at

    def iso(self) -> str:
        return to_iso(self.at)

    def cycle(self) -> None:
        try:
            self.loop.run_cycle()
        except LoopHalted as exc:
            # What `tb run` writes when its loop halts.
            self.ledger.record_run_end(
                run_id=self.loop.run_id,
                exit_reason="halted",
                error_type=type(exc).__name__,
                error_detail=str(exc),
            )
            self.stopped = True
            return
        heartbeat = Path(self.env["pinned"].limits.safety.heartbeat_path)
        stamp = self.at.timestamp()
        os.utime(heartbeat, (stamp, stamp))

    def wait(self, seconds: float) -> None:
        self.at += timedelta(seconds=seconds)
        if self.watchdog is not None:
            verdict = self.watchdog.check(record=not self.in_episode)
            self.in_episode = verdict.tripped
        if self.frozen or self.stopped:
            return
        self.since_cycle += seconds
        if self.since_cycle >= INTERVAL:
            self.since_cycle = 0.0
            self.cycle()

    def suspend(self, pid: int) -> None:
        self.suspended.append(pid)
        self.frozen = True

    def resume(self, pid: int) -> None:
        self.resumed.append(pid)
        self.frozen = False

    def world(self) -> DrillWorld:
        safety = self.env["pinned"].limits.safety
        return DrillWorld(
            ledger=self.ledger,
            broker=self.broker,
            kill_switch_path=Path(safety.kill_switch_path),
            heartbeat_path=Path(safety.heartbeat_path),
            clock=self.now,
            wait=self.wait,
            suspend=self.suspend,
            resume=self.resume,
            poll_seconds=5.0,
        )


def _harness(
    env: dict[str, Any],
    ledger: Ledger,
    monkeypatch: pytest.MonkeyPatch,
    *,
    mode: str = "demo",
    watchdog: bool = True,
    enter: bool = True,
) -> Harness:
    """A demo run holding the lease, with an entry filled and its stop working."""
    broker = _broker()
    loop = _loop(
        env, ledger, broker, run_id="run_drill", book=_book(env, ["enter"] if enter else [])
    )
    safety = env["pinned"].limits.safety
    dog = (
        Watchdog(
            heartbeat_path=Path(safety.heartbeat_path),
            kill_switch_path=Path(safety.kill_switch_path),
            liveness_path=env["run_dir"] / "watchdog",
            stale_after_seconds=safety.heartbeat_stale_seconds,
            ledger=ledger,
        )
        if watchdog
        else None
    )
    harness = Harness(env=env, ledger=ledger, broker=broker, loop=loop, watchdog=dog)
    monkeypatch.setattr("tb.ledger.store.now_iso", harness.iso)
    monkeypatch.setattr("tb.ops.killswitch.now_utc", harness.now)
    broker.clock = harness.now
    loop.clock = harness.now

    ledger.record_run_start(run_id=loop.run_id, mode=mode)
    loop.lock.acquire(at=harness.at)
    harness.cycle()
    for _ in range(3):
        harness.wait(INTERVAL)
    return harness


@pytest.fixture
def ledger(env: dict[str, Any]) -> Iterator[Ledger]:
    _seed(env, _rising_bars(days=140))
    with Ledger(env["db"], config_hash=env["pinned"].config_hash) as opened:
        yield opened


# --------------------------------------------------------------------------
# The kill-switch drill
# --------------------------------------------------------------------------


def test_the_kill_switch_drill_halts_the_loop_and_leaves_the_stops(
    env: dict[str, Any], ledger: Ledger, monkeypatch: pytest.MonkeyPatch
) -> None:
    harness = _harness(env, ledger, monkeypatch)
    [held] = holdings(harness.broker)
    assert held.protected, "the fixture needs a position with its stop working"

    result = run_drill(
        harness.world(),
        DrillKind.KILL_SWITCH,
        halt_within=HALT_WITHIN,
        notice_within=NOTICE_WITHIN,
        cycling_within=CYCLING_WITHIN,
    )

    assert result.passed, result.failures
    assert result.halted_after_seconds is not None
    assert result.halted_after_seconds <= INTERVAL + 5
    assert harness.stopped
    assert not read_kill_switch(env["pinned"].limits.safety.kill_switch_path).may_trade
    assert [h.ticker for h in result.after] == [TICKER] and result.after[0].protected
    assert any("still protected at the broker" in line for line in result.observations)

    [record] = drills(ledger)
    assert record.passed and record.kind == "kill_switch" and record.mode == "demo"
    assert record.market_open and record.n_holdings == 1

    # The day's journal page tells it.
    page = render_page(
        ledger,
        session=AS_OF.date(),
        limits=env["pinned"].limits,
        limits_hash=env["pinned"].config_hash,
        as_of=AS_OF.replace(hour=23),
    )
    assert f"kill_switch drill {result.drill_id} passed" in page.text
    assert "| demo | drill |" in page.text


def test_a_passed_drill_is_the_sessions_drill_not_its_fault(
    env: dict[str, Any], ledger: Ledger, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Halted on purpose: neither clean nor a break in the streak."""
    harness = _harness(env, ledger, monkeypatch)
    run_drill(
        harness.world(),
        DrillKind.KILL_SWITCH,
        halt_within=HALT_WITHIN,
        notice_within=NOTICE_WITHIN,
        cycling_within=CYCLING_WITHIN,
    )

    record = read_sessions(
        ledger,
        limits=env["pinned"].limits.live,
        mode="demo",
        now=AS_OF.replace(hour=23),
    )
    [session] = record.sessions
    assert session.verdict is SessionVerdict.DRILL, session.faults
    assert not session.faults
    assert {n.kind for n in session.notes} >= {NoteKind.DRILL}
    assert record.streak == ()


def test_a_drill_that_finds_a_stop_gone_fails_and_faults_the_session(
    env: dict[str, Any], ledger: Ledger, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The drill's reason to exist: a position left bare while the bot is stopped."""
    harness = _harness(env, ledger, monkeypatch)
    real_wait = harness.wait

    def wait_and_lose_the_stop(seconds: float) -> None:
        real_wait(seconds)
        if harness.stopped:
            for order in harness.broker.protective_orders_for(TICKER):
                harness.broker._orders.pop(order.broker_order_id)

    world = harness.world()
    world.wait = wait_and_lose_the_stop
    result = run_drill(
        world,
        DrillKind.KILL_SWITCH,
        halt_within=HALT_WITHIN,
        notice_within=NOTICE_WITHIN,
        cycling_within=CYCLING_WITHIN,
    )

    assert not result.passed
    assert any("covered by a working stop" in f for f in result.failures)
    [session] = read_sessions(
        ledger, limits=env["pinned"].limits.live, mode="demo", now=AS_OF.replace(hour=23)
    ).sessions
    assert session.verdict is SessionVerdict.FAULTED
    assert "drill_failed" in {f.kind for f in session.faults}


def test_a_loop_that_ignores_the_switch_fails_the_drill(
    env: dict[str, Any], ledger: Ledger, monkeypatch: pytest.MonkeyPatch
) -> None:
    harness = _harness(env, ledger, monkeypatch)
    harness.frozen = True  # a loop that never runs another cycle, never halts

    result = run_drill(
        harness.world(),
        DrillKind.KILL_SWITCH,
        halt_within=timedelta(minutes=2),
        notice_within=NOTICE_WITHIN,
        cycling_within=CYCLING_WITHIN,
    )
    assert not result.passed
    assert any("did not stop within 120s" in f for f in result.failures)


# --------------------------------------------------------------------------
# The watchdog drill
# --------------------------------------------------------------------------


def test_the_watchdog_drill_freezes_the_loop_until_the_watchdog_notices(
    env: dict[str, Any], ledger: Ledger, monkeypatch: pytest.MonkeyPatch
) -> None:
    harness = _harness(env, ledger, monkeypatch)

    result = run_drill(
        harness.world(),
        DrillKind.WATCHDOG,
        halt_within=HALT_WITHIN,
        notice_within=NOTICE_WITHIN,
        cycling_within=CYCLING_WITHIN,
    )

    assert result.passed, result.failures
    assert harness.suspended == [os.getpid()] and harness.resumed == [os.getpid()]
    assert harness.stopped
    assert any("the watchdog noticed after" in line for line in result.observations)
    trips = ledger.conn.execute(
        "SELECT COUNT(*) FROM event_log WHERE event_type = 'watchdog.tripped'"
    ).fetchone()[0]
    assert trips == 1, "one stale episode, one trip"

    # The trip and the halt were the drill's: the session is a drill session.
    [session] = read_sessions(
        ledger, limits=env["pinned"].limits.live, mode="demo", now=AS_OF.replace(hour=23)
    ).sessions
    assert session.verdict is SessionVerdict.DRILL, session.faults


def test_a_watchdog_that_never_notices_fails_the_drill_and_the_loop_is_not_thawed_live(
    env: dict[str, Any], ledger: Ledger, monkeypatch: pytest.MonkeyPatch
) -> None:
    """No watchdog running: the drill engages the switch itself before thawing."""
    harness = _harness(env, ledger, monkeypatch, watchdog=False)

    result = run_drill(
        harness.world(),
        DrillKind.WATCHDOG,
        halt_within=HALT_WITHIN,
        notice_within=timedelta(minutes=3),
        cycling_within=CYCLING_WITHIN,
    )

    assert not result.passed
    assert any("did not notice" in f for f in result.failures)
    assert any("the drill engaged it" in f for f in result.failures)
    assert harness.resumed == [os.getpid()], "the loop is thawed whatever happens"
    assert harness.stopped, "and it halts on the switch the drill engaged"


# --------------------------------------------------------------------------
# Preconditions: refused before anything is fired
# --------------------------------------------------------------------------


def _refused(harness: Harness, kind: DrillKind = DrillKind.KILL_SWITCH) -> str:
    with pytest.raises(DrillError) as caught:
        run_drill(
            harness.world(),
            kind,
            halt_within=HALT_WITHIN,
            notice_within=NOTICE_WITHIN,
            cycling_within=CYCLING_WITHIN,
        )
    assert drills(harness.ledger) == [], "a refused drill records nothing"
    assert read_kill_switch(harness.env["pinned"].limits.safety.kill_switch_path).may_trade
    return str(caught.value)


def test_a_drill_needs_a_demo_run(
    env: dict[str, Any], ledger: Ledger, monkeypatch: pytest.MonkeyPatch
) -> None:
    harness = _harness(env, ledger, monkeypatch, mode="paper")
    assert "demo account" in _refused(harness)


def test_a_drill_needs_an_open_market(
    env: dict[str, Any], ledger: Ledger, monkeypatch: pytest.MonkeyPatch
) -> None:
    harness = _harness(env, ledger, monkeypatch)
    harness.at = AS_OF.replace(hour=22)
    harness.loop.lock.renew(at=harness.at)
    assert "market is closed" in _refused(harness)


def test_a_drill_needs_a_position(
    env: dict[str, Any], ledger: Ledger, monkeypatch: pytest.MonkeyPatch
) -> None:
    harness = _harness(env, ledger, monkeypatch, enter=False)
    assert "no position is held" in _refused(harness)


def test_a_drill_needs_every_position_protected_first(
    env: dict[str, Any], ledger: Ledger, monkeypatch: pytest.MonkeyPatch
) -> None:
    harness = _harness(env, ledger, monkeypatch)
    for order in harness.broker.protective_orders_for(TICKER):
        harness.broker._orders.pop(order.broker_order_id)
    assert "a fault to fix" in _refused(harness)


def test_a_drill_needs_a_loop_that_is_cycling(
    env: dict[str, Any], ledger: Ledger, monkeypatch: pytest.MonkeyPatch
) -> None:
    harness = _harness(env, ledger, monkeypatch)
    harness.at += timedelta(minutes=20)
    harness.loop.lock.renew(at=harness.at)
    assert "not completed a cycle" in _refused(harness)


def test_a_drill_needs_a_run_holding_the_lease(
    env: dict[str, Any], ledger: Ledger, monkeypatch: pytest.MonkeyPatch
) -> None:
    harness = _harness(env, ledger, monkeypatch)
    harness.loop.lock.release(at=harness.at)
    assert "holds the lease" in _refused(harness)


def test_the_watchdog_drill_runs_on_the_loops_host_against_its_heartbeat(
    env: dict[str, Any], ledger: Ledger, monkeypatch: pytest.MonkeyPatch
) -> None:
    harness = _harness(env, ledger, monkeypatch)
    ledger.conn.execute("UPDATE instance_locks SET host = ?", (socket.gethostname() + "-other",))
    assert "on the host the process does" in _refused(harness, DrillKind.WATCHDOG)

    ledger.conn.execute("UPDATE instance_locks SET host = ?", (socket.gethostname(),))
    heartbeat = Path(env["pinned"].limits.safety.heartbeat_path)
    heartbeat.write_text('{"run_id": "someone_else", "pid": 1}', encoding="utf-8")
    assert "not the running loop's" in _refused(harness, DrillKind.WATCHDOG)


def test_the_kill_switch_left_engaged_before_a_drill_is_not_a_drill(
    env: dict[str, Any], ledger: Ledger, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A switch already engaged has already stopped the loop: nothing is cycling."""
    harness = _harness(env, ledger, monkeypatch)
    engage_kill_switch(
        env["pinned"].limits.safety.kill_switch_path, engaged_by="test", reason="earlier"
    )
    for _ in range(2):
        harness.wait(INTERVAL)
    assert harness.stopped
    harness.at += timedelta(minutes=20)
    with pytest.raises(DrillError):
        run_drill(
            harness.world(),
            DrillKind.KILL_SWITCH,
            halt_within=HALT_WITHIN,
            notice_within=NOTICE_WITHIN,
            cycling_within=CYCLING_WITHIN,
        )


def test_holdings_count_only_open_sell_stops() -> None:
    broker = _broker()
    broker.seed_position(TICKER, quantity=Decimal("2"), average_price=Decimal("150"))
    [held] = holdings(broker)
    assert held.quantity == Decimal("2") and held.covered == 0 and not held.protected


# --------------------------------------------------------------------------
# tb drill
# --------------------------------------------------------------------------


def test_tb_drill_needs_the_demo_key_and_lists_what_ran(
    env: dict[str, Any], ledger: Ledger, monkeypatch: pytest.MonkeyPatch
) -> None:
    harness = _harness(env, ledger, monkeypatch)
    run_drill(
        harness.world(),
        DrillKind.KILL_SWITCH,
        halt_within=HALT_WITHIN,
        notice_within=NOTICE_WITHIN,
        cycling_within=CYCLING_WITHIN,
    )
    runner = CliRunner()
    monkeypatch.delenv("T212_DEMO_API_KEY", raising=False)
    monkeypatch.delenv("T212_LIVE_API_KEY", raising=False)
    refused = runner.invoke(
        app, ["drill", "killswitch", "--limits", str(env["limits"]), "--db", str(env["db"])]
    )
    assert refused.exit_code == 2, refused.output
    assert "T212_DEMO_API_KEY" in refused.output

    listed = runner.invoke(app, ["drill", "list", "--db", str(env["db"])])
    assert listed.exit_code == 0, listed.output
    assert "kill_switch" in listed.output and "passed" in listed.output
