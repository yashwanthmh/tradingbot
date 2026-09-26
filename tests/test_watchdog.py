"""The dead-man switch, both directions, and the single-instance lease.

Two mechanisms that fail in opposite circumstances, which is the whole reason
there are two:

* a watchdog that only halts a stalled trader misses a *dead watchdog* — the
  trader carries on unsupervised and nothing notices;
* a trader that only halts itself misses a *wedged trader* — a wedged process
  does not run its own check.

The direction people leave out is the second one, and
`test_a_trader_that_cannot_write_the_ledger_halts` is why it matters here: a
trader that cannot write the ledger can still place orders, and orders it
cannot record are orders nothing will ever reconcile.
"""

from __future__ import annotations

from collections.abc import Iterator
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import pytest
from typer.testing import CliRunner

from tb.cli import app
from tb.ledger.store import Ledger
from tb.ops.killswitch import read_kill_switch, write_heartbeat
from tb.ops.watchdog import (
    InstanceLock,
    InstanceLockRefused,
    SelfCheck,
    Watchdog,
    WatchdogError,
)

NOW = datetime(2026, 4, 1, 14, 30, tzinfo=UTC)


@pytest.fixture
def ledger(tmp_path: Path) -> Iterator[Ledger]:
    with Ledger(tmp_path / "ledger.db") as opened:
        opened.initialise(created_by="test")
        yield opened


@pytest.fixture
def paths(tmp_path: Path) -> dict[str, Path]:
    run = tmp_path / "run"
    run.mkdir(exist_ok=True)
    return {
        "heartbeat": run / "heartbeat",
        "kill": run / "KILL",
        "liveness": run / "watchdog",
    }


def _watchdog(paths: dict[str, Path], ledger: Ledger | None = None) -> Watchdog:
    return Watchdog(
        heartbeat_path=paths["heartbeat"],
        kill_switch_path=paths["kill"],
        liveness_path=paths["liveness"],
        stale_after_seconds=30,
        ledger=ledger,
    )


def _self_check(
    ledger: Ledger, paths: dict[str, Path], *, require_watchdog: bool = True
) -> SelfCheck:
    return SelfCheck(
        ledger=ledger,
        kill_switch_path=paths["kill"],
        liveness_path=paths["liveness"],
        run_id="run_test",
        require_watchdog=require_watchdog,
    )


# --------------------------------------------------------------------------
# Direction one: the watchdog halts a stalled trader
# --------------------------------------------------------------------------


def test_a_fresh_heartbeat_passes(paths: dict[str, Path]) -> None:
    write_heartbeat(paths["heartbeat"], run_id="run_test", state="running")
    verdict = _watchdog(paths).check()
    assert verdict.healthy
    assert read_kill_switch(paths["kill"]).may_trade


def test_a_missing_heartbeat_engages_the_kill_switch(paths: dict[str, Path]) -> None:
    """Missing is stale, not "not started yet".

    The alternative reading would leave a trader that died before its first
    beat permanently unsupervised, which is the moment it is most likely to be
    misconfigured.
    """
    assert not paths["heartbeat"].exists()
    verdict = _watchdog(paths).check()

    assert verdict.tripped
    assert "engaged the kill switch" in verdict.action_taken
    assert not read_kill_switch(paths["kill"]).may_trade


def test_a_stale_heartbeat_engages_the_kill_switch(
    paths: dict[str, Path], monkeypatch: pytest.MonkeyPatch
) -> None:
    """A wedged trader — the case its own self-check cannot catch."""
    write_heartbeat(paths["heartbeat"], run_id="run_test", state="running")
    # Age the file past the bound rather than sleeping.
    old = paths["heartbeat"].stat().st_mtime - 600
    import os

    os.utime(paths["heartbeat"], (old, old))

    verdict = _watchdog(paths).check()
    assert verdict.tripped
    assert not read_kill_switch(paths["kill"]).may_trade
    assert monkeypatch is not None


def test_the_watchdog_records_why_it_tripped(paths: dict[str, Path], ledger: Ledger) -> None:
    _watchdog(paths, ledger).check()
    rows = ledger.conn.execute(
        "SELECT payload_json FROM event_log WHERE event_type = 'watchdog.tripped'"
    ).fetchall()
    assert len(rows) == 1
    assert "watchdog_to_trader" in rows[0]["payload_json"]


def test_a_pass_that_does_not_record_still_engages_the_switch(
    paths: dict[str, Path], ledger: Ledger
) -> None:
    """Fail-closed every pass; recorded once per episode by the caller's choice."""
    verdict = _watchdog(paths, ledger).check(record=False)
    assert verdict.tripped
    assert not read_kill_switch(paths["kill"]).may_trade
    trips = ledger.conn.execute(
        "SELECT COUNT(*) FROM event_log WHERE event_type = 'watchdog.tripped'"
    ).fetchone()[0]
    assert trips == 0


def test_tb_watchdog_records_one_trip_per_stale_episode(env: dict[str, Any]) -> None:
    """A trader stopped overnight must not leave a trip event every fifteen seconds."""
    with Ledger(env["db"]) as ledger:
        ledger.initialise(created_by="test")
    result = CliRunner().invoke(
        app,
        [
            "watchdog",
            "--limits",
            str(env["limits"]),
            "--db",
            str(env["db"]),
            "--cycles",
            "3",
            "--interval",
            "0",
        ],
    )
    assert result.exit_code == 1, result.output
    assert "tripped on 3 of 3" in result.output
    with Ledger(env["db"]) as ledger:
        trips = ledger.conn.execute(
            "SELECT COUNT(*) FROM event_log WHERE event_type = 'watchdog.tripped'"
        ).fetchone()[0]
    assert trips == 1


def test_the_watchdog_writes_its_own_liveness_before_deciding(
    paths: dict[str, Path],
) -> None:
    """Ordering that matters.

    If the decision path raised, the trader must still be able to tell the
    watchdog was alive up to that moment — otherwise a bug in `check()` looks
    identical to a dead watchdog and halts a healthy trader.
    """
    assert not paths["liveness"].exists()
    _watchdog(paths).check()  # trips, because there is no heartbeat
    assert paths["liveness"].exists(), "liveness must be written even on a trip"


# --------------------------------------------------------------------------
# Direction two: the trader halts itself. The half people leave out.
# --------------------------------------------------------------------------


def test_a_healthy_trader_passes_its_own_check(ledger: Ledger, paths: dict[str, Path]) -> None:
    write_heartbeat(paths["liveness"], run_id="watchdog", state="supervising")
    _self_check(ledger, paths).assert_alive(at=NOW)


def test_a_trader_that_cannot_write_the_ledger_halts(
    ledger: Ledger, paths: dict[str, Path]
) -> None:
    """**The most important test in this file.**

    A trader that cannot write the ledger can still place orders — and orders
    it cannot record are orders nothing will ever reconcile, with this process
    the only thing that knows they exist. So it halts, and the message says
    why in those terms.

    The unwritable ledger is simulated by closing the connection, which is
    what a corrupted or full-disk database looks like from here.
    """
    write_heartbeat(paths["liveness"], run_id="watchdog", state="supervising")
    check = _self_check(ledger, paths)
    check.assert_alive(at=NOW)  # healthy first, so the failure below is the change

    ledger.conn.close()
    with pytest.raises(WatchdogError, match="cannot write the ledger"):
        check.assert_alive(at=NOW)


def test_a_dead_watchdog_halts_the_trader(ledger: Ledger, paths: dict[str, Path]) -> None:
    """Running unsupervised is not a degraded mode, it is a stop.

    Being wedged is precisely the state a process cannot detect about itself,
    so losing the thing that would detect it is a reason to stop rather than
    to carry on carefully.
    """
    assert not paths["liveness"].exists()
    with pytest.raises(WatchdogError, match="watchdog is not alive"):
        _self_check(ledger, paths).assert_alive(at=NOW)


def test_a_stale_watchdog_liveness_halts_the_trader(ledger: Ledger, paths: dict[str, Path]) -> None:
    import os

    write_heartbeat(paths["liveness"], run_id="watchdog", state="supervising")
    old = paths["liveness"].stat().st_mtime - 6000
    os.utime(paths["liveness"], (old, old))

    with pytest.raises(WatchdogError, match="watchdog is not alive"):
        _self_check(ledger, paths).assert_alive(at=NOW)


def test_an_engaged_kill_switch_halts_the_trader(ledger: Ledger, paths: dict[str, Path]) -> None:
    from tb.ops.killswitch import engage_kill_switch

    write_heartbeat(paths["liveness"], run_id="watchdog", state="supervising")
    engage_kill_switch(paths["kill"], engaged_by="human", reason="testing")
    with pytest.raises(WatchdogError, match="kill switch forbids trading"):
        _self_check(ledger, paths).assert_alive(at=NOW)


def test_an_unreadable_kill_switch_counts_as_engaged(
    ledger: Ledger, paths: dict[str, Path]
) -> None:
    """Fail-closed. A kill switch that fails open is not a kill switch.

    The unreadable case is reached by making the file a directory, which is
    the most portable way to produce a read error without depending on running
    as an unprivileged user.
    """
    write_heartbeat(paths["liveness"], run_id="watchdog", state="supervising")
    paths["kill"].mkdir()

    with pytest.raises(WatchdogError, match="kill switch forbids trading"):
        _self_check(ledger, paths).assert_alive(at=NOW)


def test_the_self_check_records_the_reason_it_halted(
    ledger: Ledger, paths: dict[str, Path], tmp_path: Path
) -> None:
    """A halt with no record is a halt nobody can diagnose."""
    with pytest.raises(WatchdogError):
        _self_check(ledger, paths).assert_alive(at=NOW)

    rows = ledger.conn.execute(
        "SELECT payload_json FROM event_log WHERE event_type = 'watchdog.unreachable'"
    ).fetchall()
    assert len(rows) == 1
    assert "watchdog_unreachable" in rows[0]["payload_json"]


def test_a_drill_may_run_without_a_watchdog_but_must_say_so(
    ledger: Ledger, paths: dict[str, Path]
) -> None:
    """`require_watchdog=False` is named rather than inferred from absence.

    "No watchdog configured" and "the watchdog died" must not look the same,
    or the second silently becomes the first the moment someone runs a drill.
    """
    _self_check(ledger, paths, require_watchdog=False).assert_alive(at=NOW)


def test_the_ledger_probe_leaves_no_events_behind(ledger: Ledger, paths: dict[str, Path]) -> None:
    """A liveness check that logged itself would drown the log it protects."""
    write_heartbeat(paths["liveness"], run_id="watchdog", state="supervising")
    before = ledger.conn.execute("SELECT COUNT(*) AS n FROM event_log").fetchone()["n"]

    check = _self_check(ledger, paths)
    for _ in range(20):
        check.assert_alive(at=NOW)

    after = ledger.conn.execute("SELECT COUNT(*) AS n FROM event_log").fetchone()["n"]
    assert after == before, "twenty liveness checks appended twenty events"


# --------------------------------------------------------------------------
# Two instances, exactly one trades
# --------------------------------------------------------------------------


def test_the_second_instance_is_refused(ledger: Ledger) -> None:
    """The drill the plan asks for.

    Two loops against one account double every position and reconcile to
    nonsense — and the failure is silent, because both processes look healthy
    the whole time.
    """
    first = InstanceLock(ledger, run_id="run_a")
    second = InstanceLock(ledger, run_id="run_b")

    lease = first.acquire(at=NOW)
    assert lease.run_id == "run_a"

    with pytest.raises(InstanceLockRefused, match="run_a"):
        second.acquire(at=NOW)

    holder = first.holder(at=NOW)
    assert holder is not None and holder[0] == "run_a"


def test_a_refusal_is_recorded_not_only_raised(ledger: Ledger) -> None:
    """The refusal is the only trace the second instance was ever started."""
    InstanceLock(ledger, run_id="run_a").acquire(at=NOW)
    with pytest.raises(InstanceLockRefused):
        InstanceLock(ledger, run_id="run_b").acquire(at=NOW)

    rows = ledger.conn.execute(
        "SELECT payload_json FROM event_log WHERE event_type = 'instance.lock_refused'"
    ).fetchall()
    assert len(rows) == 1
    assert "run_b" in rows[0]["payload_json"]


def test_an_expired_lease_can_be_taken_over(ledger: Ledger) -> None:
    """A crashed holder must not lock the account out permanently.

    That would be a worse failure than the one being prevented: a crash during
    market hours would leave positions unmanaged until someone noticed.
    """
    InstanceLock(ledger, run_id="run_crashed").acquire(at=NOW, ttl_seconds=60)

    later = NOW + timedelta(seconds=61)
    lease = InstanceLock(ledger, run_id="run_new").acquire(at=later)
    assert lease.run_id == "run_new"


def test_a_released_lease_is_immediately_available(ledger: Ledger) -> None:
    first = InstanceLock(ledger, run_id="run_a")
    first.acquire(at=NOW)
    first.release(at=NOW)
    assert first.holder(at=NOW) is None

    lease = InstanceLock(ledger, run_id="run_b").acquire(at=NOW)
    assert lease.run_id == "run_b"


def test_reacquiring_your_own_lease_is_not_a_conflict(ledger: Ledger) -> None:
    """A restart that reuses a run id must not deadlock against itself."""
    lock = InstanceLock(ledger, run_id="run_a")
    lock.acquire(at=NOW)
    again = lock.acquire(at=NOW + timedelta(seconds=1))
    assert again.run_id == "run_a"


def test_renewing_a_lease_someone_else_took_is_refused(ledger: Ledger) -> None:
    """The case that keeps a stalled instance from trading after a takeover.

    Ours expired, another instance took it, and ours woke up. Continuing would
    be exactly the two-instance state the lock exists to prevent, so the renew
    refuses rather than silently re-taking it.
    """
    ours = InstanceLock(ledger, run_id="run_a")
    ours.acquire(at=NOW, ttl_seconds=60)

    later = NOW + timedelta(seconds=61)
    InstanceLock(ledger, run_id="run_b").acquire(at=later)

    with pytest.raises(InstanceLockRefused, match="run_b"):
        ours.renew(at=later + timedelta(seconds=1))


def test_renewing_extends_the_expiry(ledger: Ledger) -> None:
    lock = InstanceLock(ledger, run_id="run_a")
    lease = lock.acquire(at=NOW, ttl_seconds=60)
    extended = lock.renew(at=NOW + timedelta(seconds=30), ttl_seconds=60)
    assert extended > lease.expires_at


def test_an_expired_lease_reports_no_holder(ledger: Ledger) -> None:
    """`holder` answers "who is live", not "who was last".

    A caller deciding whether to start needs the first question; the second
    would keep it out forever after any crash.
    """
    lock = InstanceLock(ledger, run_id="run_a")
    lock.acquire(at=NOW - timedelta(days=1), ttl_seconds=60)
    assert lock.holder(at=NOW) is None
