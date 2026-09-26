"""The dead-man switch, in both directions, and the single-instance lease.

**Bidirectional, because one direction alone leaves a hole each way.**

A watchdog that only halts a stalled trader misses the case where the watchdog
itself dies: the trader carries on unsupervised, and nothing notices for as
long as nobody looks. A trader that only halts itself when it cannot write the
ledger misses the case where it is wedged — a hung socket, a deadlock, a stop
the world — because a wedged process does not run the check.

So there are two mechanisms here, and they fail in opposite circumstances:

* `Watchdog.check()` runs in a separate process. It reads the trader's
  heartbeat and engages the kill switch if it is stale. This catches a wedged
  or dead trader.
* `SelfCheck.assert_alive()` runs *inside* the trader, every cycle. It halts
  the trader if it cannot write the ledger, cannot read the kill switch, or
  finds the watchdog's own liveness file stale. This catches a dead watchdog,
  and a trader that has lost its ability to record what it is doing.

The second one is the one people leave out, and it is the one that matters
most on this venue: a trader that cannot write the ledger can still place
orders, and orders it cannot record are orders nothing will ever reconcile.

**Fail-closed throughout.** Missing, unreadable and permission-denied all mean
stop. There is no reading of a heartbeat file that means "probably fine".

The instance lease is here rather than in its own module because it is the
same kind of control: two trading loops against one account double every
position and reconcile to nonsense, and the failure is silent — both
processes look healthy. A lease with an expiry rather than a boolean, so a
crashed holder does not lock the account out permanently.
"""

from __future__ import annotations

import contextlib
import os
import socket
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path

from tb.core.clock import from_iso, now_utc, to_iso
from tb.core.errors import TbError
from tb.ledger.events import Actor, EventType, InstanceLockPayload, WatchdogPayload
from tb.ledger.store import Ledger
from tb.ops.killswitch import (
    engage_kill_switch,
    read_heartbeat,
    read_kill_switch,
    write_heartbeat,
)

# How long the trader's own liveness claim about the watchdog stays good. A
# separate bound from the trader's heartbeat: the watchdog does less work per
# cycle, so a longer silence from it is a stronger signal.
WATCHDOG_STALE_SECONDS = 120

# The lease. Long enough that an ordinary GC pause or a slow ledger write does
# not drop it, short enough that a crashed instance's lock frees within a
# cycle or two rather than at the end of the session.
LEASE_SECONDS = 90

LOCK_NAME = "trading_loop"


class WatchdogError(TbError):
    """The watchdog could not establish that the system is safe to run."""


class InstanceLockRefused(TbError):
    """Another instance holds the trading lease."""

    def __init__(self, holder_run_id: str, holder_pid: int, expires_at: str) -> None:
        super().__init__(
            f"another instance holds the trading lease: run {holder_run_id} (pid "
            f"{holder_pid}), expiring {expires_at}. Two loops against one account double "
            "every position and reconcile to nonsense, and neither process looks unhealthy "
            "while it happens."
        )
        self.holder_run_id = holder_run_id
        self.holder_pid = holder_pid


@dataclass(frozen=True, slots=True)
class WatchdogVerdict:
    """What the watchdog concluded, and what it did about it."""

    healthy: bool
    detail: str
    observed_age_seconds: float | None = None
    limit_seconds: float | None = None
    action_taken: str = ""

    @property
    def tripped(self) -> bool:
        return not self.healthy


@dataclass(frozen=True, slots=True)
class Watchdog:
    """Runs outside the trader. Halts it when its heartbeat goes stale.

    Holds the paths rather than a `HardLimits`, because the watchdog has to be
    able to run when the config is the thing that is broken — a supervisor
    that could not start without a valid limits file would be unavailable
    exactly when a bad config had taken the trader down.
    """

    heartbeat_path: Path
    kill_switch_path: Path
    liveness_path: Path
    stale_after_seconds: int
    ledger: Ledger | None = None

    def check(self, *, at: datetime | None = None, record: bool = True) -> WatchdogVerdict:
        """One supervision pass.

        Writes the watchdog's own liveness *first*, before deciding anything.
        That ordering matters: if the decision path raises, the trader must
        still be able to tell that the watchdog was alive up to that moment,
        or a bug in this function would look identical to a dead watchdog and
        halt a perfectly healthy trader.

        A stale heartbeat engages the switch on every pass — cheap, idempotent,
        and fail-closed if someone releases it while the trader is still
        silent — but `record=False` leaves the ledger alone. A supervisor
        passes it for every pass after the first in one stale episode, or a
        trader stopped overnight would leave a trip event every fifteen
        seconds until morning.
        """
        moment = at or now_utc()
        self.note_alive(at=moment)

        reading = read_heartbeat(self.heartbeat_path, stale_after_seconds=self.stale_after_seconds)
        if not reading.stale:
            return WatchdogVerdict(
                healthy=True,
                detail=reading.detail,
                observed_age_seconds=reading.age_seconds,
                limit_seconds=self.stale_after_seconds,
            )

        # Stale, missing or unreadable — all the same conclusion.
        engaged = engage_kill_switch(
            self.kill_switch_path,
            engaged_by="watchdog",
            reason=f"heartbeat stale: {reading.detail}",
        )
        # `may_trade` is False once the switch is engaged, and also False if
        # it could not be read — which is the fail-closed reading and still
        # the outcome we want, so both count as success here.
        action = (
            "engaged the kill switch"
            if not engaged.may_trade
            else f"could not engage the kill switch: {engaged.detail}"
        )
        verdict = WatchdogVerdict(
            healthy=False,
            detail=reading.detail,
            observed_age_seconds=reading.age_seconds,
            limit_seconds=self.stale_after_seconds,
            action_taken=action,
        )
        if record:
            self._record(verdict, direction="watchdog_to_trader")
        return verdict

    def note_alive(self, *, at: datetime | None = None) -> None:
        """Write the watchdog's own liveness marker.

        The trader reads this to detect a dead supervisor. Written with the
        same atomic replace the heartbeat uses, so a torn file cannot be
        mistaken for a stale one.
        """
        write_heartbeat(self.liveness_path, run_id="watchdog", state="supervising")

    def _record(self, verdict: WatchdogVerdict, *, direction: str) -> None:
        if self.ledger is None:
            return
        self.ledger.append(
            EventType.WATCHDOG_TRIPPED,
            "watchdog",
            WatchdogPayload(
                direction=direction,
                observed_age_seconds=verdict.observed_age_seconds,
                limit_seconds=verdict.limit_seconds,
                action_taken=verdict.action_taken,
                detail=verdict.detail,
            ),
            actor=Actor.WATCHDOG,
        )


@dataclass(frozen=True, slots=True)
class SelfCheck:
    """Runs inside the trader, every cycle. The direction people leave out.

    Three questions, each of which can be false while the process is otherwise
    perfectly healthy:

    1. **Can I write the ledger?** If not, any order placed from here is an
       order nothing will ever reconcile. This is the most important of the
       three and the least obvious.
    2. **Can I read the kill switch?** An unreadable kill switch is an engaged
       one, by the fail-closed rule — so a permission change on that file
       stops trading rather than being ignored.
    3. **Is the watchdog alive?** A trader running unsupervised is running
       without the mechanism that would catch it wedging.
    """

    ledger: Ledger
    kill_switch_path: Path
    liveness_path: Path
    run_id: str
    watchdog_stale_after_seconds: int = WATCHDOG_STALE_SECONDS
    # Set False only for a drill that deliberately runs without a supervisor.
    # Named rather than inferred from the file's absence, because "no watchdog
    # configured" and "the watchdog died" must not look the same.
    require_watchdog: bool = True

    def assert_alive(self, *, at: datetime | None = None) -> None:
        """Raise unless all three checks pass. Called before deciding anything."""
        moment = at or now_utc()

        kill = read_kill_switch(self.kill_switch_path)
        if not kill.may_trade:
            raise WatchdogError(
                f"the kill switch forbids trading: {kill.detail}. "
                + (
                    ""
                    if kill.determinable
                    else "It could not be read, which counts as engaged — a kill switch "
                    "that fails open is not a kill switch."
                )
            )

        try:
            self._probe_ledger()
        except Exception as exc:
            self._record(
                direction="ledger_unwritable",
                detail=f"{type(exc).__name__}: {exc}",
                action="halting",
            )
            raise WatchdogError(
                f"cannot write the ledger ({type(exc).__name__}: {exc}). Halting rather "
                "than trading: an order placed from here would be an order nothing can "
                "reconcile, and this process would be the only thing that knew it "
                "existed."
            ) from exc

        if self.require_watchdog:
            reading = read_heartbeat(
                self.liveness_path, stale_after_seconds=self.watchdog_stale_after_seconds
            )
            if reading.stale:
                self._record(
                    direction="watchdog_unreachable",
                    detail=reading.detail,
                    action="halting",
                    observed_age_seconds=reading.age_seconds,
                )
                raise WatchdogError(
                    f"the watchdog is not alive: {reading.detail}. Halting rather than "
                    "running unsupervised — without it, a wedged trader has nothing "
                    "watching it, and being wedged is precisely the state this process "
                    "cannot detect about itself."
                )
        assert moment is not None  # the parameter is part of the contract

    def beat(self, *, state: str, heartbeat_path: Path) -> None:
        """Write the trader's heartbeat. The watchdog's only evidence."""
        write_heartbeat(heartbeat_path, run_id=self.run_id, state=state)

    def _probe_ledger(self) -> None:
        """Prove the ledger is writable, without writing an event.

        A `BEGIN IMMEDIATE` that is immediately rolled back takes the write
        lock and releases it, which is exactly the capability in question —
        and unlike appending a probe event, it leaves the chain untouched. A
        liveness check that filled the audit log with its own liveness checks
        would make the log unreadable for its actual purpose.
        """
        self.ledger.conn.execute("BEGIN IMMEDIATE")
        self.ledger.conn.rollback()

    def _record(
        self,
        *,
        direction: str,
        detail: str,
        action: str,
        observed_age_seconds: float | None = None,
    ) -> None:
        # Best effort: the ledger may be exactly what is broken. A failure to
        # record the failure must not replace the diagnosis with a different
        # exception.
        with contextlib.suppress(Exception):
            self.ledger.append(
                EventType.WATCHDOG_UNREACHABLE,
                "watchdog",
                WatchdogPayload(
                    direction=direction,
                    run_id=self.run_id,
                    observed_age_seconds=observed_age_seconds,
                    limit_seconds=self.watchdog_stale_after_seconds,
                    action_taken=action,
                    detail=detail,
                ),
                actor=Actor.SYSTEM,
            )


# --------------------------------------------------------------------------
# The single-instance lease
# --------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class Lease:
    """A held trading lease."""

    lock_name: str
    run_id: str
    host: str
    pid: int
    acquired_at: datetime
    expires_at: datetime


class InstanceLock:
    """One trading loop per account, enforced in the ledger.

    In the ledger rather than in a lock file, for one reason: the ledger is
    already the thing both instances must share, and a lock file on a
    different filesystem from the database would let two instances each hold
    "the" lock while writing to the same chain.

    A lease with an expiry rather than a boolean flag. A crashed instance
    cannot release its own lock, and a permanent lock would mean a crash
    during market hours locks the account out until someone notices — which
    is a worse failure than the one being prevented.
    """

    def __init__(self, ledger: Ledger, *, run_id: str, lock_name: str = LOCK_NAME) -> None:
        self._ledger = ledger
        self._run_id = run_id
        self._lock_name = lock_name
        self._host = socket.gethostname()
        self._pid = os.getpid()

    def acquire(self, *, at: datetime | None = None, ttl_seconds: int = LEASE_SECONDS) -> Lease:
        """Take the lease, or refuse.

        The read and the write happen in one `BEGIN IMMEDIATE` transaction, so
        two instances starting simultaneously cannot both observe a free lock.
        Without that, the race is not theoretical — a supervisor restarting a
        crashed process while the old one is still dying is exactly how it
        happens.
        """
        moment = at or now_utc()
        expires = moment + timedelta(seconds=ttl_seconds)

        # The refusal is decided inside the transaction but **recorded and
        # raised outside it**. Appending the refusal event and then raising
        # from within the same transaction rolls the event back with
        # everything else, so the only trace that a second instance was ever
        # started would be destroyed by the act of refusing it. That is the
        # opposite of what this control is for.
        conflict: tuple[str, int, str] | None = None

        with self._ledger.transaction() as tx:
            row = tx.execute(
                "SELECT run_id, pid, host, expires_at, released_at FROM instance_locks"
                " WHERE lock_name = ?",
                (self._lock_name,),
            ).fetchone()

            if row is not None:
                held_by_us = str(row["run_id"]) == self._run_id
                released = row["released_at"] is not None
                expired = from_iso(str(row["expires_at"])) <= moment
                if not (held_by_us or released or expired):
                    conflict = (
                        str(row["run_id"]),
                        int(row["pid"]),
                        str(row["expires_at"]),
                    )

            if conflict is None:
                tx.execute(
                    "INSERT INTO instance_locks (lock_name, run_id, host, pid, acquired_at,"
                    " renewed_at, expires_at, released_at) VALUES (?,?,?,?,?,?,?,NULL)"
                    " ON CONFLICT(lock_name) DO UPDATE SET run_id = excluded.run_id,"
                    " host = excluded.host, pid = excluded.pid,"
                    " acquired_at = excluded.acquired_at,"
                    " renewed_at = excluded.renewed_at,"
                    " expires_at = excluded.expires_at, released_at = NULL",
                    (
                        self._lock_name,
                        self._run_id,
                        self._host,
                        self._pid,
                        to_iso(moment),
                        to_iso(moment),
                        to_iso(expires),
                    ),
                )
                tx.append(
                    EventType.INSTANCE_LOCK_ACQUIRED,
                    self._lock_name,
                    InstanceLockPayload(
                        lock_name=self._lock_name,
                        run_id=self._run_id,
                        host=self._host,
                        pid=self._pid,
                        acquired=True,
                        expires_at=to_iso(expires),
                    ),
                    actor=Actor.SYSTEM,
                    run_id=self._run_id,
                )

        if conflict is not None:
            holder_run, holder_pid, holder_expiry = conflict
            self._ledger.append(
                EventType.INSTANCE_LOCK_REFUSED,
                self._lock_name,
                InstanceLockPayload(
                    lock_name=self._lock_name,
                    run_id=self._run_id,
                    host=self._host,
                    pid=self._pid,
                    acquired=False,
                    held_by_run_id=holder_run,
                    held_by_pid=holder_pid,
                    detail=(
                        "refused: a live lease is held by another instance. Recorded "
                        "rather than only raised, because a second instance starting is "
                        "a silent fault otherwise — both processes look healthy."
                    ),
                ),
                actor=Actor.SYSTEM,
                run_id=self._run_id,
            )
            raise InstanceLockRefused(holder_run, holder_pid, holder_expiry)

        return Lease(
            lock_name=self._lock_name,
            run_id=self._run_id,
            host=self._host,
            pid=self._pid,
            acquired_at=moment,
            expires_at=expires,
        )

    def renew(self, *, at: datetime | None = None, ttl_seconds: int = LEASE_SECONDS) -> datetime:
        """Extend the lease. Called every cycle.

        Refuses if the lease is no longer ours — which means another instance
        took it after ours expired, and continuing to trade would be exactly
        the two-instance state the lock exists to prevent.
        """
        moment = at or now_utc()
        expires = moment + timedelta(seconds=ttl_seconds)
        with self._ledger.transaction() as tx:
            row = tx.execute(
                "SELECT run_id, pid, expires_at FROM instance_locks WHERE lock_name = ?",
                (self._lock_name,),
            ).fetchone()
            if row is None or str(row["run_id"]) != self._run_id:
                holder = "nobody" if row is None else str(row["run_id"])
                raise InstanceLockRefused(
                    holder,
                    0 if row is None else int(row["pid"]),
                    "n/a" if row is None else str(row["expires_at"]),
                )
            tx.execute(
                "UPDATE instance_locks SET renewed_at = ?, expires_at = ? WHERE lock_name = ?",
                (to_iso(moment), to_iso(expires), self._lock_name),
            )
        return expires

    def release(self, *, at: datetime | None = None) -> None:
        """Give the lease up on a clean shutdown.

        Best effort by nature — a crashed process cannot call this, which is
        why the expiry exists. Releasing explicitly just means the next
        instance does not have to wait for it.
        """
        moment = at or now_utc()
        with self._ledger.transaction() as tx:
            tx.execute(
                "UPDATE instance_locks SET released_at = ? WHERE lock_name = ? AND run_id = ?",
                (to_iso(moment), self._lock_name, self._run_id),
            )
            tx.append(
                EventType.INSTANCE_LOCK_RELEASED,
                self._lock_name,
                InstanceLockPayload(
                    lock_name=self._lock_name,
                    run_id=self._run_id,
                    host=self._host,
                    pid=self._pid,
                    acquired=False,
                    detail="released on clean shutdown",
                ),
                actor=Actor.SYSTEM,
                run_id=self._run_id,
            )

    def holder(self, *, at: datetime | None = None) -> tuple[str, int] | None:
        """Who holds a *live* lease, if anyone.

        Answers "who is live", not "who was last". A caller deciding whether
        to start needs the first question; the second would keep it out
        forever after any crash.

        `at` is injectable for the same reason it is on `acquire` and `renew`:
        expiry is the entire semantics here, so reaching for the wall clock
        would make the answer depend on when the caller asked rather than on
        the lease.
        """
        moment = at or now_utc()
        row = self._ledger.conn.execute(
            "SELECT run_id, pid, expires_at, released_at FROM instance_locks WHERE lock_name = ?",
            (self._lock_name,),
        ).fetchone()
        if row is None or row["released_at"] is not None:
            return None
        if from_iso(str(row["expires_at"])) <= moment:
            return None
        return str(row["run_id"]), int(row["pid"])
