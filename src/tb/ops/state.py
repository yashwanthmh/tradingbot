"""The run state machine.

One property carries almost all the weight: **orders can only be placed in
`TRADING`**. Every other state — booting, reconciling, halted — physically
cannot mint the risk token that `place_order` will require in M4. That includes
`RECONCILING`, with no exception for protective stops: until the reconciler has
established what the account actually holds, "place a protective stop" is a
decision made without knowing the position it is protecting.

`HALT_UNRECONCILED` is deliberately separate from `HALTED`. A normal halt
clears itself once its condition passes; an unreconciled one requires a human
to acknowledge it. That is the single place this otherwise-autonomous system
insists on a person, because the alternative is guessing about real money.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass
from decimal import Decimal
from enum import StrEnum
from pathlib import Path

from tb.config.loader import PinnedLimits
from tb.core.clock import now_iso
from tb.core.errors import HaltRequired, Killed, TbError
from tb.ledger.events import (
    Actor,
    EventType,
    HaltClearedPayload,
    HaltRaisedPayload,
    KillswitchPayload,
    StateTransitionPayload,
)
from tb.ledger.store import Ledger
from tb.ops.killswitch import KillSwitchState, read_kill_switch


class RunState(StrEnum):
    BOOT = "boot"
    RECONCILING = "reconciling"
    TRADING = "trading"
    HALTED = "halted"
    HALT_UNRECONCILED = "halt_unreconciled"
    STOPPED = "stopped"

    @property
    def may_place_orders(self) -> bool:
        return self is RunState.TRADING

    @property
    def requires_human_ack(self) -> bool:
        return self is RunState.HALT_UNRECONCILED


# Transitions that are permitted. Anything absent here is a bug in the caller,
# raised rather than silently allowed: an undeclared transition means some code
# path has a different mental model of the lifecycle than this file does.
_ALLOWED: dict[RunState, frozenset[RunState]] = {
    RunState.BOOT: frozenset(
        {RunState.RECONCILING, RunState.HALTED, RunState.HALT_UNRECONCILED, RunState.STOPPED}
    ),
    RunState.RECONCILING: frozenset(
        {RunState.TRADING, RunState.HALTED, RunState.HALT_UNRECONCILED, RunState.STOPPED}
    ),
    RunState.TRADING: frozenset(
        {
            RunState.RECONCILING,
            RunState.HALTED,
            RunState.HALT_UNRECONCILED,
            RunState.STOPPED,
        }
    ),
    # A halt returns through reconciliation, never straight to trading: whatever
    # the halt was, the account may have moved while we were stopped.
    RunState.HALTED: frozenset(
        {RunState.RECONCILING, RunState.HALT_UNRECONCILED, RunState.STOPPED}
    ),
    RunState.HALT_UNRECONCILED: frozenset({RunState.RECONCILING, RunState.STOPPED}),
    RunState.STOPPED: frozenset({RunState.BOOT}),
}


class InvalidTransition(TbError):
    def __init__(self, current: RunState, requested: RunState) -> None:
        allowed = ", ".join(sorted(s.value for s in _ALLOWED.get(current, frozenset())))
        super().__init__(
            f"cannot go from {current.value} to {requested.value}; "
            f"permitted from {current.value}: {allowed or '(none)'}"
        )
        self.current = current
        self.requested = requested


@dataclass(frozen=True, slots=True)
class StateReading:
    state: RunState
    since: str
    reason: str | None
    run_id: str | None
    event_seq: int | None


@dataclass(frozen=True, slots=True)
class OpenHalt:
    halt_id: str
    raised_at: str
    trigger: str
    detail: str | None
    run_id: str | None


@dataclass(frozen=True, slots=True)
class TradingPermission:
    """Why the system may or may not trade right now, in full.

    Returned rather than a bare bool so the caller — and the operator reading
    `tb status` — sees every blocking reason at once instead of discovering them
    one restart at a time.
    """

    allowed: bool
    state: RunState
    kill_switch: KillSwitchState
    open_halts: tuple[OpenHalt, ...]
    config_drift: str | None
    reasons: tuple[str, ...]

    def raise_if_blocked(self) -> None:
        if self.allowed:
            return
        if self.kill_switch is not KillSwitchState.CLEAR:
            raise Killed("; ".join(self.reasons))
        trigger = self.open_halts[0].trigger if self.open_halts else "state"
        raise HaltRequired(trigger, "; ".join(self.reasons))


class StateMachine:
    """Reads and moves the run state, recording every move in the ledger."""

    def __init__(self, ledger: Ledger, pinned: PinnedLimits, *, run_id: str | None = None) -> None:
        self._ledger = ledger
        self._pinned = pinned
        self._run_id = run_id

    # -- reading -----------------------------------------------------------

    def current(self) -> StateReading:
        """The current state, defaulting to BOOT for a fresh ledger."""
        row = self._ledger.conn.execute("SELECT * FROM run_state WHERE id = 1").fetchone()
        if row is None:
            return StateReading(
                state=RunState.BOOT, since=now_iso(), reason=None, run_id=None, event_seq=None
            )
        return StateReading(
            state=RunState(row["state"]),
            since=row["since"],
            reason=row["reason"],
            run_id=row["run_id"],
            event_seq=int(row["event_seq"]) if row["event_seq"] is not None else None,
        )

    def open_halts(self) -> tuple[OpenHalt, ...]:
        rows = self._ledger.conn.execute(
            "SELECT halt_id, raised_at, trigger, detail, run_id FROM halts "
            "WHERE cleared_at IS NULL ORDER BY raised_at ASC"
        ).fetchall()
        return tuple(
            OpenHalt(
                halt_id=r["halt_id"],
                raised_at=r["raised_at"],
                trigger=r["trigger"],
                detail=r["detail"],
                run_id=r["run_id"],
            )
            for r in rows
        )

    @property
    def kill_switch_path(self) -> Path:
        return Path(self._pinned.limits.safety.kill_switch_path)

    def check_trading_permission(self) -> TradingPermission:
        """Every gate between here and placing an order.

        Deliberately checks all of them rather than short-circuiting, so the
        answer to "why is it not trading" is complete on the first look.
        """
        reasons: list[str] = []

        reading = self.current()
        if not reading.state.may_place_orders:
            reasons.append(
                f"run state is {reading.state.value}"
                + (f" ({reading.reason})" if reading.reason else "")
            )

        switch = read_kill_switch(self.kill_switch_path)
        if not switch.may_trade:
            reasons.append(switch.detail)

        halts = self.open_halts()
        for halt in halts:
            reasons.append(f"open halt {halt.halt_id} [{halt.trigger}]: {halt.detail}")

        drift: str | None = None
        try:
            self._pinned.verify_unchanged()
        except Exception as exc:
            drift = str(exc)
            reasons.append(drift)

        return TradingPermission(
            allowed=not reasons,
            state=reading.state,
            kill_switch=switch.state,
            open_halts=halts,
            config_drift=drift,
            reasons=tuple(reasons),
        )

    # -- writing -----------------------------------------------------------

    def transition_to(
        self,
        target: RunState,
        *,
        reason: str,
        actor: Actor = Actor.SYSTEM,
    ) -> StateReading:
        """Move state, recording the move and its projection atomically."""
        reading = self.current()
        if target == reading.state:
            return reading
        if target not in _ALLOWED.get(reading.state, frozenset()):
            raise InvalidTransition(reading.state, target)

        payload = StateTransitionPayload(
            from_state=reading.state.value,
            to_state=target.value,
            reason=reason,
            run_id=self._run_id,
        )
        with self._ledger.transaction() as tx:
            event = tx.append(EventType.STATE_TRANSITIONED, target.value, payload, actor=actor)
            tx.execute(
                """
                INSERT INTO run_state (id, state, since, reason, run_id, event_seq)
                VALUES (1, ?, ?, ?, ?, ?)
                ON CONFLICT(id) DO UPDATE SET
                    state = excluded.state,
                    since = excluded.since,
                    reason = excluded.reason,
                    run_id = excluded.run_id,
                    event_seq = excluded.event_seq
                """,
                (target.value, event.ts_utc, reason, self._run_id, event.seq),
            )
        return self.current()

    def raise_halt(
        self,
        trigger: str,
        detail: str,
        *,
        actor: Actor = Actor.SYSTEM,
        observed_value: Decimal | float | None = None,
        limit_value: Decimal | float | None = None,
        unreconciled: bool = False,
    ) -> str:
        """Record a halt and stop trading.

        `observed_value` and `limit_value` are worth the columns: "the daily loss
        breaker fired" is much less useful six weeks later than "fired at -2.4%
        against a 2.0% limit".
        """
        halt_id = f"halt_{uuid.uuid4().hex[:12]}"
        payload = HaltRaisedPayload(
            halt_id=halt_id,
            trigger=trigger,
            detail=detail,
            run_id=self._run_id,
            observed_value=observed_value,
            limit_value=limit_value,
        )
        with self._ledger.transaction() as tx:
            event = tx.append(EventType.HALT_RAISED, halt_id, payload, actor=actor)
            tx.execute(
                """
                INSERT INTO halts (
                    halt_id, raised_at, trigger, detail, run_id, observed_value, limit_value
                ) VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    halt_id,
                    event.ts_utc,
                    trigger,
                    detail,
                    self._run_id,
                    None if observed_value is None else str(observed_value),
                    None if limit_value is None else str(limit_value),
                ),
            )

        target = RunState.HALT_UNRECONCILED if unreconciled else RunState.HALTED
        current = self.current().state
        if target in _ALLOWED.get(current, frozenset()):
            self.transition_to(target, reason=f"{trigger}: {detail}", actor=actor)
        return halt_id

    def clear_halt(self, halt_id: str, *, cleared_by: str, clear_reason: str) -> None:
        """Acknowledge a halt.

        Clearing does not resume trading. The caller must transition through
        `RECONCILING`, because while we were stopped the account may have moved
        — a stop may have filled, a position may have been closed by hand.
        """
        row = self._ledger.conn.execute(
            "SELECT halt_id, cleared_at FROM halts WHERE halt_id = ?", (halt_id,)
        ).fetchone()
        if row is None:
            raise TbError(f"no such halt: {halt_id}")
        if row["cleared_at"] is not None:
            return

        payload = HaltClearedPayload(
            halt_id=halt_id, cleared_by=cleared_by, clear_reason=clear_reason
        )
        with self._ledger.transaction() as tx:
            event = tx.append(EventType.HALT_CLEARED, halt_id, payload, actor=Actor.HUMAN)
            tx.execute(
                "UPDATE halts SET cleared_at = ?, cleared_by = ?, clear_reason = ? "
                "WHERE halt_id = ?",
                (event.ts_utc, cleared_by, clear_reason, halt_id),
            )

    def record_kill_switch(
        self, *, engaged: bool, path: str, determinable: bool, detail: str, engaged_by: str | None
    ) -> None:
        """Note a kill switch change in the ledger."""
        payload = KillswitchPayload(
            path=path, determinable=determinable, detail=detail, engaged_by=engaged_by
        )
        self._ledger.append(
            EventType.KILLSWITCH_ENGAGED if engaged else EventType.KILLSWITCH_RELEASED,
            path,
            payload,
            actor=Actor.HUMAN if engaged_by else Actor.SYSTEM,
        )
