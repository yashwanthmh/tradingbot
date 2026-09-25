"""The intent write-ahead log: exactly-once submission, synthesised client-side.

Trading 212 offers no idempotency key. There is no header to send that makes a
duplicate POST a no-op, so "place this order exactly once" is not something the
venue will do for us — it has to be built here, and the only tool available is
ordering: **the intent is committed to the ledger before the socket write.**

That ordering is the whole design, and the alternative is worse in a way that
is easy to miss. Send-then-record loses the intent entirely if the process dies
mid-flight, leaving a live order at the broker that nothing in this system
knows about — no id to cancel, no record it exists, and a position that appears
from nowhere at the next reconcile. Record-then-send can only ever leave the
opposite: a row saying "we were about to send this", which is a question that
can be *answered* by looking.

So the states are:

| state | meaning |
|---|---|
| `PENDING_SUBMIT` | committed, not yet sent. After a crash: **UNKNOWN** |
| `SUBMITTED` | the POST left the process, no response yet. Also UNKNOWN |
| `ACKNOWLEDGED` | the broker gave us an order id |
| `REJECTED` | the broker refused it, explicitly |
| `RESOLVED_FILLED` / `RESOLVED_CANCELLED` | terminal, known |
| `ABANDONED` | deliberately given up on, by a human or by reconciliation |

**`UNKNOWN` is never silently converted to failed.** That conversion is
precisely how a double fill happens: the process crashes after the POST, the
retry decides the first attempt "must have failed", and two orders exist for
one intention. `resolve_unknown` therefore refuses to guess — it either finds
evidence at the broker or leaves the intent unresolved and blocks trading,
which is the fail-closed direction.

`intent_id` is **deterministic** over the facts that define the order, so a
retry of the same logical order computes the same id and collides with the
existing row rather than creating a second one. The run id is deliberately
*not* in the hash: a restart is a new run, and including it would make every
recovery attempt a fresh intent — defeating the whole mechanism at exactly the
moment it matters.
"""

from __future__ import annotations

import sqlite3
from dataclasses import dataclass
from datetime import datetime
from decimal import Decimal
from enum import StrEnum

from tb.broker.port import BrokerOrder, OrderPurpose, OrderStatus, OrderType, Side, TimeValidity
from tb.core.clock import from_iso, now_utc, to_iso
from tb.core.errors import TbError
from tb.core.ids import deterministic_id, new_id
from tb.ledger.events import (
    Actor,
    EventType,
    IntentCommittedPayload,
    IntentResolvedPayload,
    OrderOutcomePayload,
    OrderSubmittedPayload,
)
from tb.ledger.store import Ledger
from tb.risk.token import RiskToken


class IntentError(TbError):
    """An intent could not be committed, submitted or resolved."""


class IntentState(StrEnum):
    """Where an intent is in its lifecycle.

    `is_unknown` is the property the recovery pass keys on, and it covers two
    states rather than one: an intent that was committed but never sent, and
    one that was sent but never answered. Both are "we do not know whether an
    order exists", and both must be resolved by looking rather than assuming.
    """

    PENDING_SUBMIT = "pending_submit"
    SUBMITTED = "submitted"
    ACKNOWLEDGED = "acknowledged"
    REJECTED = "rejected"
    RESOLVED_FILLED = "resolved_filled"
    RESOLVED_CANCELLED = "resolved_cancelled"
    ABANDONED = "abandoned"

    @property
    def is_unknown(self) -> bool:
        """Whether an order may exist at the broker that we cannot account for."""
        return self in (IntentState.PENDING_SUBMIT, IntentState.SUBMITTED)

    @property
    def is_terminal(self) -> bool:
        return self in (
            IntentState.REJECTED,
            IntentState.RESOLVED_FILLED,
            IntentState.RESOLVED_CANCELLED,
            IntentState.ABANDONED,
        )

    @property
    def may_submit(self) -> bool:
        """Only a freshly committed intent may be sent.

        Deliberately excludes `SUBMITTED`: re-sending something already on the
        wire is the double-fill path. A `SUBMITTED` intent whose response was
        lost is resolved by looking, never by sending again.
        """
        return self is IntentState.PENDING_SUBMIT


class PriorityClass(StrEnum):
    """How a submission competes for the rate-limit budget.

    Risk-reducing work pre-empts risk-increasing work under saturation, which
    is the property that keeps an exit possible on a day when entries have
    eaten the whole budget.
    """

    RISK_REDUCING = "risk_reducing"
    PROTECTIVE = "protective"
    RISK_INCREASING = "risk_increasing"

    @property
    def rank(self) -> int:
        """Lower sorts first."""
        return {
            PriorityClass.PROTECTIVE: 0,
            PriorityClass.RISK_REDUCING: 1,
            PriorityClass.RISK_INCREASING: 2,
        }[self]


def priority_for(purpose: OrderPurpose) -> PriorityClass:
    """A protective stop outranks even other risk-reducing work.

    The unprotected window is the one interval where the sizing assumptions
    are load-bearing, so closing it is the most urgent thing the system ever
    does — ahead of a discretionary exit, which is merely important.
    """
    if purpose is OrderPurpose.PROTECTIVE_STOP:
        return PriorityClass.PROTECTIVE
    if purpose.is_risk_reducing:
        return PriorityClass.RISK_REDUCING
    return PriorityClass.RISK_INCREASING


@dataclass(frozen=True, slots=True)
class OrderIntent:
    """One intention to place one order, and everything needed to replay it."""

    intent_id: str
    run_id: str
    t212_ticker: str
    side: Side
    order_type: OrderType
    purpose: OrderPurpose
    priority_class: PriorityClass
    quantity: Decimal
    risk_token_id: str
    state: IntentState
    wal_committed_at: datetime
    decision_id: str | None = None
    parent_intent_id: str | None = None
    instrument_uid: str | None = None
    limit_price: Decimal | None = None
    stop_price: Decimal | None = None
    time_validity: TimeValidity | None = None
    expected_cost_bps: float | None = None
    submitted_at: datetime | None = None
    broker_order_id: str | None = None
    resolved_at: datetime | None = None
    resolution_note: str = ""
    n_submit_attempts: int = 0

    @property
    def is_unknown(self) -> bool:
        return self.state.is_unknown

    @property
    def is_replaceable(self) -> bool:
        return self.purpose in REPLACEABLE_PURPOSES

    def fingerprint(self) -> tuple[str, str, str, Decimal]:
        """What an order at the broker must match to be *this* intent.

        Ticker, side, type and quantity — the fields the venue echoes back.
        Not price: a market order has none, and a stop price may be adjusted
        by the venue's tick rules, so including it would make a legitimate
        match fail.
        """
        return (self.t212_ticker, self.side.value, self.order_type.value, self.quantity)

    def matches(self, order: BrokerOrder) -> bool:
        """Whether a broker order could be this intent's order.

        Deliberately loose on quantity for a *partially* filled order, and
        deliberately strict on ticker and side. A false positive here adopts
        someone else's order as ours; a false negative leaves a real order
        orphaned. Both are bad, so the match is reported rather than acted on
        silently — see `resolve_unknown`.
        """
        if order.ticker != self.t212_ticker:
            return False
        if order.side is not None and order.side is not self.side:
            return False
        # A `None` quantity is not a mismatch: the venue omits it on some
        # order shapes, and refusing to match on a field the broker did not
        # send would orphan an order that is genuinely ours.
        return order.quantity is None or order.quantity == self.quantity


def compute_intent_id(
    *,
    t212_ticker: str,
    side: Side,
    order_type: OrderType,
    purpose: OrderPurpose,
    quantity: Decimal,
    decision_id: str | None,
    stop_price: Decimal | None = None,
) -> str:
    """The deterministic id for one logical order.

    **`run_id` is not an input, on purpose.** A crash-and-restart is a new run,
    so including it would give the recovery attempt a fresh id, a fresh row,
    and a second order — which is the exact failure this function exists to
    prevent.

    `decision_id` *is* an input, because two entries in the same instrument
    from two different decisions are genuinely two orders. Without it they
    would collide and the second would be silently dropped as a duplicate.

    `stop_price` is included so that re-placing a protective stop at a new
    level is a new intent rather than a collision with the old one — a
    trailing stop moves, and each position of it is a distinct order.
    """
    return deterministic_id(
        "int",
        parts={
            "ticker": t212_ticker,
            "side": side.value,
            "type": order_type.value,
            "purpose": purpose.value,
            "quantity": str(quantity),
            "decision": decision_id or "",
            "stop": str(stop_price) if stop_price is not None else "",
        },
    )


# Orders the engine places to keep a state true rather than to answer one
# decision, so the same order legitimately recurs with every field identical:
# a protective stop withdrawn for an exit the venue then refuses is put back at
# the same level, for the same quantity, under the same entry decision; and
# every flatten of the same size has no decision at all. Entries and exits are
# not here: each answers exactly one decision, and a second submission for the
# same decision is a bug the id must keep catching.
REPLACEABLE_PURPOSES: frozenset[OrderPurpose] = frozenset(
    {OrderPurpose.PROTECTIVE_STOP, OrderPurpose.FLATTEN}
)


def replacement_id(base_intent_id: str, generation: int) -> str:
    """The id of the `generation`-th placement of one replaceable order.

    Generation 0 is the base id itself, so every intent committed before
    replacements existed keeps its id. Later generations are as deterministic
    as the base — derived from it and the count — so a crash while placing a
    replacement recomputes the same id on restart and finds the same row,
    which is the property the base id exists for.
    """
    if generation == 0:
        return base_intent_id
    return deterministic_id("int", parts={"base": base_intent_id, "generation": str(generation)})


class IntentLog:
    """The write-ahead log. Commits before the wire, resolves by looking.

    Every method that changes state does so in one ledger transaction together
    with its event, so a projection row without the chain entry behind it is
    impossible — the intent table is a projection of `event_log` like every
    other table here.
    """

    def __init__(self, ledger: Ledger, *, run_id: str) -> None:
        self._ledger = ledger
        self._run_id = run_id

    # -- the write-ahead commit -------------------------------------------

    def commit(
        self,
        *,
        token: RiskToken,
        order_type: OrderType,
        limit_price: Decimal | None = None,
        stop_price: Decimal | None = None,
        time_validity: TimeValidity | None = None,
        expected_cost_bps: float | None = None,
        parent_intent_id: str | None = None,
        instrument_uid: str | None = None,
        at: datetime | None = None,
    ) -> OrderIntent:
        """Record the intention to place this order. **Before** sending it.

        Takes a `RiskToken` rather than loose order parameters, and derives the
        order from it. That is not convenience: it means an intent physically
        cannot describe an order the risk engine did not approve, because the
        approved quantity, side, ticker and purpose come from the token itself.

        Returns the existing intent unchanged if this id is already committed.
        A re-commit is not an error — it is what a retry of the same logical
        order looks like, and the correct response is to hand back the row
        that already exists rather than to create a second one.

        Except for a replaceable order (`REPLACEABLE_PURPOSES`) whose
        predecessor is settled. That is not a retry: the predecessor was
        withdrawn, filled or refused, and handing it back would either refuse
        the replacement as a duplicate or — for a stop re-placed at its old
        level — report protection that is no longer there. So the next
        generation is committed instead. A predecessor still in flight or at
        the venue is handed back as before.
        """
        moment = at or now_utc()
        token.authorises(
            t212_ticker=token.t212_ticker,
            side=token.side,
            quantity=token.quantity,
            purpose=token.purpose,
            at=moment,
        )
        if order_type.needs_limit_price and limit_price is None:
            raise IntentError(
                f"{order_type.value} needs a limit price. Sending one without would "
                "either be rejected or, worse, be treated as a market order."
            )
        if order_type.needs_stop_price and stop_price is None:
            raise IntentError(
                f"{order_type.value} needs a stop price. A protective stop with no level "
                "is not protection."
            )

        base_id = compute_intent_id(
            t212_ticker=token.t212_ticker,
            side=token.side,
            order_type=order_type,
            purpose=token.purpose,
            quantity=token.quantity,
            decision_id=token.decision_id,
            stop_price=stop_price,
        )
        intent_id = base_id
        existing = self.get(intent_id)
        if token.purpose in REPLACEABLE_PURPOSES:
            generation = 0
            while existing is not None and existing.state.is_terminal:
                generation += 1
                intent_id = replacement_id(base_id, generation)
                existing = self.get(intent_id)
        if existing is not None:
            return existing

        priority = priority_for(token.purpose)
        with self._ledger.transaction() as tx:
            event = tx.append(
                EventType.INTENT_COMMITTED,
                intent_id,
                IntentCommittedPayload(
                    intent_id=intent_id,
                    run_id=self._run_id,
                    decision_id=token.decision_id,
                    parent_intent_id=parent_intent_id,
                    t212_ticker=token.t212_ticker,
                    side=token.side.value,
                    order_type=order_type.value,
                    purpose=token.purpose.value,
                    priority_class=priority.value,
                    quantity=token.quantity,
                    risk_token_id=token.token_id,
                    limit_price=limit_price,
                    stop_price=stop_price,
                    time_validity=time_validity.value if time_validity else None,
                    expected_cost_bps=expected_cost_bps,
                ),
                actor=Actor.RISK,
                run_id=self._run_id,
            )
            tx.execute(
                "INSERT INTO order_intents (intent_id, decision_id, run_id, parent_intent_id,"
                " instrument_uid, t212_ticker, side, order_type, purpose, priority_class,"
                " quantity, limit_price, stop_price, time_validity, expected_cost_bps,"
                " risk_token_id, state, wal_committed_at, n_submit_attempts,"
                " committing_event_seq)"
                " VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                (
                    intent_id,
                    token.decision_id,
                    self._run_id,
                    parent_intent_id,
                    instrument_uid,
                    token.t212_ticker,
                    token.side.value,
                    order_type.value,
                    token.purpose.value,
                    priority.value,
                    str(token.quantity),
                    str(limit_price) if limit_price is not None else None,
                    str(stop_price) if stop_price is not None else None,
                    time_validity.value if time_validity else None,
                    expected_cost_bps,
                    token.token_id,
                    IntentState.PENDING_SUBMIT.value,
                    to_iso(moment),
                    0,
                    event.seq,
                ),
            )

        committed = self.get(intent_id)
        if committed is None:  # pragma: no cover - the insert above would have raised
            raise IntentError(f"{intent_id}: committed but not readable back")
        return committed

    # -- the wire ----------------------------------------------------------

    def mark_submitted(self, intent_id: str, *, at: datetime | None = None) -> OrderIntent:
        """Record that a POST is about to leave the process.

        Called immediately *before* the socket write, not after. The window
        between this row and the actual send is the one place a crash can
        leave `SUBMITTED` with no order at the broker — which is a false
        positive for "an order may exist", and false positives here are the
        safe direction: they cause a look, not a second order.

        `n_submit_attempts` is incremented rather than set, because the count
        is the evidence for the crash drills' "exactly one POST per intent_id"
        assertion.
        """
        intent = self._require(intent_id)
        if not intent.state.may_submit:
            raise IntentError(
                f"{intent_id} is {intent.state.value}, which may not be submitted. "
                + (
                    "It is already on the wire; re-sending is how one intention becomes "
                    "two orders. Resolve it by looking at the broker instead."
                    if intent.state is IntentState.SUBMITTED
                    else "It has already reached a terminal state."
                )
            )
        moment = at or now_utc()
        with self._ledger.transaction() as tx:
            tx.append(
                EventType.ORDER_SUBMITTED,
                intent_id,
                OrderSubmittedPayload(
                    intent_id=intent_id,
                    run_id=self._run_id,
                    t212_ticker=intent.t212_ticker,
                    attempt=intent.n_submit_attempts + 1,
                    sent_at=to_iso(moment),
                ),
                actor=Actor.SYSTEM,
                run_id=self._run_id,
            )
            tx.execute(
                "UPDATE order_intents SET state = ?, submitted_at = ?,"
                " n_submit_attempts = n_submit_attempts + 1 WHERE intent_id = ?",
                (IntentState.SUBMITTED.value, to_iso(moment), intent_id),
            )
        return self._require(intent_id)

    def mark_acknowledged(
        self,
        intent_id: str,
        *,
        broker_order_id: str,
        detail: str = "",
    ) -> OrderIntent:
        """The broker returned an order id. The intent is no longer unknown.

        Takes no `at`, unlike its neighbours: there is no `acknowledged_at`
        column, so a timestamp passed here would be silently discarded. The
        appended event carries the moment. Accepting a parameter and ignoring
        it is worse than not offering one.
        """
        intent = self._require(intent_id)
        with self._ledger.transaction() as tx:
            tx.append(
                EventType.ORDER_ACKNOWLEDGED,
                intent_id,
                OrderOutcomePayload(
                    intent_id=intent_id,
                    run_id=self._run_id,
                    t212_ticker=intent.t212_ticker,
                    broker_order_id=broker_order_id,
                    status=OrderStatus.WORKING.value,
                    detail=detail,
                ),
                actor=Actor.BROKER,
                run_id=self._run_id,
            )
            tx.execute(
                "UPDATE order_intents SET state = ?, broker_order_id = ? WHERE intent_id = ?",
                (IntentState.ACKNOWLEDGED.value, broker_order_id, intent_id),
            )
        return self._require(intent_id)

    def mark_rejected(
        self,
        intent_id: str,
        *,
        detail: str,
        broker_message: str | None = None,
        at: datetime | None = None,
    ) -> OrderIntent:
        """The broker refused the order, explicitly and in so many words.

        Only for an *explicit* refusal — a 4xx naming a reason. A timeout, a
        connection reset or a 5xx is **not** a rejection: the order may well
        have been accepted before the response was lost, so those leave the
        intent `SUBMITTED` and unknown. Calling this for a transport failure
        would be the assumption that causes double fills.
        """
        intent = self._require(intent_id)
        moment = at or now_utc()
        with self._ledger.transaction() as tx:
            tx.append(
                EventType.ORDER_REJECTED,
                intent_id,
                OrderOutcomePayload(
                    intent_id=intent_id,
                    run_id=self._run_id,
                    t212_ticker=intent.t212_ticker,
                    status=OrderStatus.REJECTED.value,
                    detail=detail,
                    broker_message=broker_message,
                ),
                actor=Actor.BROKER,
                run_id=self._run_id,
            )
            tx.execute(
                "UPDATE order_intents SET state = ?, resolved_at = ?, resolution_note = ?"
                " WHERE intent_id = ?",
                (IntentState.REJECTED.value, to_iso(moment), detail, intent_id),
            )
        return self._require(intent_id)

    def resolve(
        self,
        intent_id: str,
        *,
        state: IntentState,
        resolved_by: str,
        broker_order_id: str | None = None,
        detail: str = "",
        at: datetime | None = None,
    ) -> OrderIntent:
        """Settle an intent into a terminal state.

        `resolved_by` distinguishes a response we *received* from a state we
        *discovered*. Collapsing the two would make an unacknowledged order
        that turned out to have filled indistinguishable from a normal one,
        and the unprotected window it implies would go unmeasured.
        """
        if not state.is_terminal:
            raise IntentError(
                f"{state.value} is not a terminal state, so it cannot resolve an intent. "
                "Use mark_submitted or mark_acknowledged for intermediate states."
            )
        intent = self._require(intent_id)
        moment = at or now_utc()
        with self._ledger.transaction() as tx:
            tx.append(
                EventType.INTENT_RESOLVED,
                intent_id,
                IntentResolvedPayload(
                    intent_id=intent_id,
                    run_id=self._run_id,
                    final_state=state.value,
                    resolved_by=resolved_by,
                    broker_order_id=broker_order_id or intent.broker_order_id,
                    detail=detail,
                ),
                actor=Actor.RECONCILER,
                run_id=self._run_id,
            )
            tx.execute(
                "UPDATE order_intents SET state = ?, resolved_at = ?, resolution_note = ?,"
                " broker_order_id = COALESCE(?, broker_order_id) WHERE intent_id = ?",
                (state.value, to_iso(moment), detail, broker_order_id, intent_id),
            )
        return self._require(intent_id)

    # -- recovery ----------------------------------------------------------

    def unknown(self) -> tuple[OrderIntent, ...]:
        """Every intent that might have an order behind it.

        Across all runs, not just this one: the whole point is that a crashed
        previous run's intents are this run's problem. Ordered oldest first, so
        recovery addresses the one that has been outstanding longest — it is
        the most likely to have filled.
        """
        rows = self._ledger.conn.execute(
            "SELECT * FROM order_intents WHERE state IN (?, ?) ORDER BY wal_committed_at",
            (IntentState.PENDING_SUBMIT.value, IntentState.SUBMITTED.value),
        ).fetchall()
        return tuple(_from_row(row) for row in rows)

    def resolve_unknown(
        self,
        intent: OrderIntent,
        *,
        broker_orders: tuple[BrokerOrder, ...],
        at: datetime | None = None,
    ) -> OrderIntent:
        """Settle one unknown intent against what the broker actually shows.

        Three outcomes, and the third is the important one:

        * A matching order exists → adopt its id and acknowledge. The order is
          ours; we simply never saw the response.
        * No matching order, and the intent was never sent → abandon it. This
          is the one case where absence is conclusive: nothing left the
          process, so nothing can exist.
        * No matching order, and the intent *was* sent → **leave it unknown**
          and say so. It may have filled and closed already, in which case it
          is not in the open-orders list — so absence proves nothing, and
          calling it failed is the double-fill path. The caller halts.
        """
        if not intent.is_unknown:
            return intent

        matches = tuple(order for order in broker_orders if intent.matches(order))
        if len(matches) == 1:
            order = matches[0]
            return self.mark_acknowledged(
                intent.intent_id,
                broker_order_id=order.broker_order_id,
                detail=(
                    "adopted during recovery: the order exists at the broker and this "
                    "intent describes it, so the response was lost rather than the order"
                ),
            )
        if len(matches) > 1:
            # Two orders matching one intent is the double-fill this whole
            # mechanism exists to prevent, so it is reported rather than
            # tidied away. Resolving it needs a human: cancelling the wrong
            # one of a pair is worse than leaving both visible.
            raise IntentError(
                f"{intent.intent_id} matches {len(matches)} live broker orders "
                f"({', '.join(o.broker_order_id for o in matches)}). That is the duplicate "
                "submission the write-ahead log exists to prevent, so it will not be "
                "resolved automatically — cancel the extra order by hand and re-run."
            )

        if intent.state is IntentState.PENDING_SUBMIT:
            return self.resolve(
                intent.intent_id,
                state=IntentState.ABANDONED,
                resolved_by="recovery_never_sent",
                detail=(
                    "committed but never submitted, and no matching order exists. Nothing "
                    "left the process, so nothing can be at the broker — this is the one "
                    "case where absence is conclusive."
                ),
                at=at,
            )
        return intent

    def blocking_unknowns(
        self, *, broker_orders: tuple[BrokerOrder, ...]
    ) -> tuple[OrderIntent, ...]:
        """Unknowns that survived recovery and must stop trading.

        A `SUBMITTED` intent with no matching open order is unresolvable from
        the open-orders list alone, because a filled-and-closed order is not in
        it. Order history would settle it, but that endpoint is rate-limited to
        six calls a minute and falls behind exactly when this matters — so the
        honest answer is to halt and let the reconciler's three-axis diff,
        which does read history, settle it.
        """
        survivors: list[OrderIntent] = []
        for intent in self.unknown():
            if intent.state is IntentState.SUBMITTED and not any(
                intent.matches(order) for order in broker_orders
            ):
                survivors.append(intent)
        return tuple(survivors)

    # -- reads -------------------------------------------------------------

    def get(self, intent_id: str) -> OrderIntent | None:
        row = self._ledger.conn.execute(
            "SELECT * FROM order_intents WHERE intent_id = ?", (intent_id,)
        ).fetchone()
        return None if row is None else _from_row(row)

    def for_ticker(self, t212_ticker: str) -> tuple[OrderIntent, ...]:
        rows = self._ledger.conn.execute(
            "SELECT * FROM order_intents WHERE t212_ticker = ? ORDER BY wal_committed_at",
            (t212_ticker,),
        ).fetchall()
        return tuple(_from_row(row) for row in rows)

    def by_broker_order_id(self, broker_order_id: str) -> OrderIntent | None:
        """The intent behind a broker order, or `None` for an order we did not place.

        What turns a row from the broker's order list back into something with a
        purpose and an owner. `None` is a real answer: an order placed by hand in
        the venue's app has no intent, and code acting on it must say so rather
        than invent one.
        """
        row = self._ledger.conn.execute(
            "SELECT * FROM order_intents WHERE broker_order_id = ?"
            " ORDER BY wal_committed_at DESC LIMIT 1",
            (broker_order_id,),
        ).fetchone()
        return None if row is None else _from_row(row)

    def protective_for(self, t212_ticker: str) -> tuple[OrderIntent, ...]:
        """Live protective stops for a ticker.

        What the unprotected-position check reads. Only non-terminal intents
        count: a cancelled stop is not protection, and a rejected one never
        was.
        """
        return tuple(
            intent
            for intent in self.for_ticker(t212_ticker)
            if intent.purpose is OrderPurpose.PROTECTIVE_STOP and not intent.state.is_terminal
        )

    def counts_today(self, *, day: datetime, t212_ticker: str | None = None) -> tuple[int, int]:
        """Orders committed today, in total and for one ticker.

        Counted from intents rather than from the broker's order list on
        purpose: `GET /equity/orders` is one call per five seconds against a
        write path that permits fifty a minute, so the broker's view is always
        behind — and an order we sent but never got a response for still
        counts against a runaway-loop budget, which is exactly what the
        broker's list omits.
        """
        prefix = day.astimezone().strftime("%Y-%m-%d")
        total = self._ledger.conn.execute(
            "SELECT COUNT(*) AS n FROM order_intents WHERE wal_committed_at LIKE ?",
            (f"{prefix}%",),
        ).fetchone()
        for_symbol = 0
        if t212_ticker is not None:
            row = self._ledger.conn.execute(
                "SELECT COUNT(*) AS n FROM order_intents"
                " WHERE wal_committed_at LIKE ? AND t212_ticker = ?",
                (f"{prefix}%", t212_ticker),
            ).fetchone()
            for_symbol = int(row["n"])
        return int(total["n"]), for_symbol

    def _require(self, intent_id: str) -> OrderIntent:
        intent = self.get(intent_id)
        if intent is None:
            raise IntentError(
                f"no intent {intent_id!r} in the log. An order cannot be submitted or "
                "resolved without its write-ahead row: that row is the only record that "
                "the order was ever intended."
            )
        return intent


def _from_row(row: sqlite3.Row) -> OrderIntent:
    get = row.__getitem__

    def _dec(key: str) -> Decimal | None:
        value = get(key)
        return None if value is None else Decimal(str(value))

    validity = get("time_validity")
    submitted = get("submitted_at")
    resolved = get("resolved_at")
    return OrderIntent(
        intent_id=str(get("intent_id")),
        run_id=str(get("run_id")),
        t212_ticker=str(get("t212_ticker")),
        side=Side(str(get("side"))),
        order_type=OrderType(str(get("order_type"))),
        purpose=OrderPurpose(str(get("purpose"))),
        priority_class=PriorityClass(str(get("priority_class"))),
        quantity=Decimal(str(get("quantity"))),
        risk_token_id=str(get("risk_token_id")),
        state=IntentState(str(get("state"))),
        wal_committed_at=from_iso(str(get("wal_committed_at"))),
        decision_id=None if get("decision_id") is None else str(get("decision_id")),
        parent_intent_id=(
            None if get("parent_intent_id") is None else str(get("parent_intent_id"))
        ),
        instrument_uid=None if get("instrument_uid") is None else str(get("instrument_uid")),
        limit_price=_dec("limit_price"),
        stop_price=_dec("stop_price"),
        time_validity=None if validity is None else TimeValidity(str(validity)),
        expected_cost_bps=(
            None if get("expected_cost_bps") is None else float(get("expected_cost_bps"))
        ),
        submitted_at=None if submitted is None else from_iso(str(submitted)),
        broker_order_id=(None if get("broker_order_id") is None else str(get("broker_order_id"))),
        resolved_at=None if resolved is None else from_iso(str(resolved)),
        resolution_note=str(get("resolution_note") or ""),
        n_submit_attempts=int(get("n_submit_attempts") or 0),
    )


def new_fill_id() -> str:
    return new_id("fill")
