"""Submission: the one path from an approved order to the venue.

Everything about exactly-once lives in the *order* of the four steps below,
and in what each failure between them is allowed to conclude:

    1. commit the intent to the ledger        <- before anything is sent
    2. mark it submitted                      <- before the socket write
    3. place_order                            <- the wire
    4. record the outcome                     <- after

A crash between 1 and 2 leaves `PENDING_SUBMIT`: nothing was sent, so absence
at the broker is conclusive and recovery may abandon it. A crash anywhere at
or after 3 leaves `SUBMITTED`: the order may exist, absence proves nothing,
and recovery must resolve it by looking.

Step 2 is deliberately *before* step 3 rather than after. It produces a false
positive — an intent marked submitted whose POST never left — and that is the
safe direction: a false positive causes a look, while a false negative causes
a second order.

`submit` is the only function here that places an order, and it is the only
caller of `Broker.place_order` outside the drills. That is not enforced by a
type — `place_order` already requires a token, which is the enforceable half —
but it is why the recovery pass and the fill recorder live beside it rather
than in the loop.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import datetime
from decimal import Decimal

from tb.broker.port import (
    Broker,
    OrderOutcomeUnknown,
    OrderPurpose,
    OrderRejected,
    OrderStatus,
    OrderType,
    TimeValidity,
)
from tb.broker.t212.errors import BrokerHttpError
from tb.core.clock import now_utc
from tb.core.errors import TbError, TransportError
from tb.core.ids import new_id
from tb.engine.intents import IntentLog, IntentState, OrderIntent
from tb.ledger.events import (
    Actor,
    EventType,
    FillPayload,
    OrderOutcomePayload,
    ProtectionPayload,
)
from tb.ledger.store import Ledger
from tb.risk.token import RiskToken, RiskTokenError


class SubmissionError(TbError):
    """A submission could not be completed, and its state is known."""


class SubmissionUnknown(TbError):
    """A submission's outcome is unknown. **Never retry on this.**

    Raised where an order may or may not exist at the venue. The caller's only
    correct responses are to halt, or to run recovery against the broker's
    order list — never to send the order again.
    """

    def __init__(self, intent: OrderIntent, detail: str) -> None:
        super().__init__(
            f"{intent.intent_id} ({intent.t212_ticker} {intent.purpose.value}) is in an "
            f"unknown state: {detail}. An order may exist at the broker. Do not re-send; "
            "resolve it against the broker's orders, or halt."
        )
        self.intent = intent
        self.detail = detail


class FillSource(str):
    """Where a fill price came from. A string subclass so it lands in SQL as-is."""

    API_HISTORY = "api_history"
    INFERRED = "inferred_from_position_delta"


@dataclass(frozen=True, slots=True)
class Submission:
    """The result of one successful submission."""

    intent: OrderIntent
    broker_order_id: str
    status: OrderStatus

    @property
    def filled(self) -> bool:
        return self.status is OrderStatus.FILLED


@dataclass(frozen=True, slots=True)
class Withdrawal:
    """What became of one attempt to withdraw a working order.

    `withdrawn` is the fact a caller acts on: the order is no longer working, so
    whatever it had reserved at the venue is released. It is deliberately not
    "the cancel call returned" — on this venue a cancel that finds the order
    already gone succeeds silently, and gone can mean *filled*.
    """

    broker_order_id: str
    status: OrderStatus
    withdrawn: bool
    detail: str = ""

    @property
    def filled_first(self) -> bool:
        """The order traded before the withdrawal reached it."""
        return self.status is OrderStatus.FILLED


@dataclass
class OrderSubmitter:
    """Commits, sends, and records. Holds the ordering that makes it safe."""

    ledger: Ledger
    broker: Broker
    log: IntentLog
    run_id: str
    # Injected so a drill can pin it, for the same reason the simulator's is
    # injected: a token's ten-second life makes wall-clock timing part of the
    # test otherwise.
    clock: Callable[[], datetime] | None = field(default=None)

    def _now(self) -> datetime:
        return self.clock() if self.clock is not None else now_utc()

    # -- the path ----------------------------------------------------------

    def submit(
        self,
        token: RiskToken,
        *,
        order_type: OrderType,
        purpose: OrderPurpose,
        limit_price: Decimal | None = None,
        stop_price: Decimal | None = None,
        time_validity: TimeValidity | None = None,
        parent_intent_id: str | None = None,
        instrument_uid: str | None = None,
    ) -> Submission:
        """Place one order exactly once, or leave a resolvable record."""
        moment = self._now()

        # 1. The write-ahead commit. If this raises, nothing was sent and
        #    nothing is recorded — the `pre_wal` crash point, where the
        #    correct state is simply "it never happened".
        intent = self.log.commit(
            token=token,
            order_type=order_type,
            limit_price=limit_price,
            stop_price=stop_price,
            time_validity=time_validity,
            parent_intent_id=parent_intent_id,
            instrument_uid=instrument_uid,
            at=moment,
        )

        # A re-entry into an intent that is already resolved is not a retry,
        # it is a duplicate call. Returning its outcome is right; sending
        # again is not.
        if intent.state is IntentState.ACKNOWLEDGED and intent.broker_order_id:
            return Submission(intent, intent.broker_order_id, OrderStatus.WORKING)
        if intent.state.is_terminal:
            raise SubmissionError(
                f"{intent.intent_id} is already {intent.state.value} "
                f"({intent.resolution_note or 'no note'}). This call would be a second "
                "order for an intention that has already been settled."
            )
        if intent.state is IntentState.SUBMITTED:
            raise SubmissionUnknown(
                intent,
                "it was already submitted and never resolved. Re-sending is the double-fill path",
            )

        # 2. Marked before the wire, on purpose. The false positive this can
        #    produce — submitted, but the POST never left — causes a look. The
        #    false negative the other ordering would produce causes a second
        #    order.
        intent = self.log.mark_submitted(intent.intent_id, at=moment)

        # 3. The wire.
        try:
            placed = self.broker.place_order(
                token,
                order_type=order_type,
                purpose=purpose,
                limit_price=limit_price,
                stop_price=stop_price,
                time_validity=time_validity,
            )
        except OrderOutcomeUnknown as exc:
            # Every adapter raises this for the same situation, so the
            # submitter does not need to know which one it is talking to.
            raise SubmissionUnknown(intent, str(exc)) from exc
        except RiskTokenError as exc:
            # Conclusive, and before the wire: every adapter checks the token
            # before building a request, so an expired or mismatched token
            # means nothing was sent. It happens when approval and send are
            # separated by other broker calls — an exit waits on its stop's
            # withdrawal — and treating it as unknown would halt the loop over
            # an order that provably does not exist.
            self.log.mark_rejected(
                intent.intent_id,
                detail=f"the risk token was refused before sending: {exc}",
                at=self._now(),
            )
            raise SubmissionError(f"{intent.intent_id} was not sent: {exc}") from exc
        except OrderRejected as exc:
            # Conclusive: the venue read the order and said no.
            self.log.mark_rejected(
                intent.intent_id,
                detail=exc.detail,
                broker_message=exc.code,
                at=self._now(),
            )
            raise SubmissionError(f"{intent.intent_id} was rejected: {exc}") from exc
        except BrokerHttpError as exc:
            if 400 <= exc.status_code < 500 and exc.status_code not in (401, 403, 429):
                # Conclusive. The venue read the order and said no.
                self.log.mark_rejected(
                    intent.intent_id,
                    detail=f"HTTP {exc.status_code}",
                    broker_message=exc.body[:400],
                    at=self._now(),
                )
                raise SubmissionError(
                    f"{intent.intent_id} was rejected: HTTP {exc.status_code} {exc.body[:200]}"
                ) from exc
            raise SubmissionUnknown(intent, f"HTTP {exc.status_code}") from exc
        except TransportError as exc:
            raise SubmissionUnknown(intent, f"transport failure: {exc}") from exc
        except Exception as exc:
            # Anything unclassified is **unknown**, not failed. A bug in an
            # adapter lands here, and so would an exception type nobody
            # thought to categorise — and in both cases the order may exist.
            # Fail-closed is the only safe default this far down the path.
            raise SubmissionUnknown(intent, f"{type(exc).__name__}: {exc}") from exc

        # 4. Record the outcome. A crash between 3 and 4 is the
        #    `post_response_pre_persist` point: the order is acknowledged at
        #    the venue and unrecorded here, which recovery resolves by
        #    matching on the broker's list.
        intent = self.log.mark_acknowledged(
            intent.intent_id,
            broker_order_id=placed.broker_order_id,
            detail=f"venue status {placed.raw_status or placed.status.value}",
        )
        return Submission(intent, placed.broker_order_id, placed.status)

    # -- withdrawal --------------------------------------------------------

    def cancel(
        self,
        token: RiskToken,
        *,
        broker_order_id: str,
        t212_ticker: str,
        reason: str,
    ) -> Withdrawal:
        """Withdraw one working order, then read it back and record what it became.

        Read back rather than trusted, because the cancel call's return says
        little: this venue answers a cancel for an order that is already gone as
        a success, and gone can mean a protective stop that *fired* a moment
        earlier. So afterwards the order is fetched and:

        * `CANCELLED` — withdrawn, and its intent (if we placed it) is resolved so;
        * `FILLED` — it traded first, resolved as filled, and the caller must not
          act as though the shares are still held;
        * absent from the venue's active orders — withdrawn one way or the other.
          The intent is left for the fill sweep, which reads history, to settle;
        * still working (the venue reports `CANCELLING` as working) — not
          withdrawn yet. Nothing is resolved, and the caller must not assume the
          shares are free.

        No write-ahead row, unlike `submit`. A cancel is idempotent — withdrawing
        an order twice has the effect of withdrawing it once — so a lost
        response is settled by reading the order back, never by a duplicate.
        """
        intent = self.log.by_broker_order_id(broker_order_id)
        detail = reason
        try:
            self.broker.cancel_order(token, broker_order_id=broker_order_id)
        except OrderRejected as exc:
            # Usually "already terminal". Which terminal state is what the
            # read-back below is for.
            detail = f"{reason}; the venue refused the cancel: {exc}"
        except (OrderOutcomeUnknown, TransportError, BrokerHttpError) as exc:
            detail = f"{reason}; the cancel's outcome is unknown: {exc}"

        try:
            order = self.broker.get_order(broker_order_id)
        except (TransportError, BrokerHttpError) as exc:
            return Withdrawal(
                broker_order_id,
                OrderStatus.UNKNOWN,
                withdrawn=False,
                detail=f"{detail}; could not read the order back: {exc}",
            )

        if order is None:
            status = OrderStatus.UNKNOWN
            withdrawn = True
            detail = f"{detail}; no longer among the venue's active orders"
        else:
            status = order.status
            withdrawn = order.status.is_terminal

        moment = self._now()
        if status is OrderStatus.CANCELLED or (order is None and withdrawn):
            self.ledger.append(
                EventType.ORDER_CANCELLED,
                t212_ticker,
                OrderOutcomePayload(
                    intent_id=None if intent is None else intent.intent_id,
                    run_id=self.run_id,
                    t212_ticker=t212_ticker,
                    broker_order_id=broker_order_id,
                    status=OrderStatus.CANCELLED.value if order else "gone",
                    detail=detail,
                ),
                actor=Actor.RISK,
                run_id=self.run_id,
            )
        if intent is not None and not intent.state.is_terminal:
            if status is OrderStatus.CANCELLED:
                self.log.resolve(
                    intent.intent_id,
                    state=IntentState.RESOLVED_CANCELLED,
                    resolved_by="withdrawn_by_engine",
                    detail=detail,
                    at=moment,
                )
            elif status is OrderStatus.FILLED:
                self.log.resolve(
                    intent.intent_id,
                    state=IntentState.RESOLVED_FILLED,
                    resolved_by="discovered_filled_at_withdrawal",
                    detail=detail,
                    at=moment,
                )
        return Withdrawal(broker_order_id, status, withdrawn=withdrawn, detail=detail)

    # -- recovery ----------------------------------------------------------

    def recover(self) -> tuple[tuple[OrderIntent, ...], tuple[OrderIntent, ...]]:
        """Resolve what a previous run left unknown.

        Returns `(resolved, still_unknown)`. The second tuple is why this
        returns a pair rather than a count: anything left in it must stop
        trading, and a caller that only saw a number would have to decide what
        to do about it without knowing which orders were involved.

        Runs before the first decision of every run, not only after a known
        crash — a run that ended cleanly cannot be distinguished from one that
        did not without looking at this table.
        """
        unknowns = self.log.unknown()
        if not unknowns:
            return (), ()

        broker_orders = self.broker.get_open_orders()
        resolved: list[OrderIntent] = []
        for intent in unknowns:
            settled = self.log.resolve_unknown(intent, broker_orders=broker_orders, at=self._now())
            if not settled.is_unknown:
                resolved.append(settled)

        return tuple(resolved), self.log.blocking_unknowns(broker_orders=broker_orders)

    # -- fills -------------------------------------------------------------

    def record_fill(
        self,
        *,
        intent: OrderIntent,
        quantity: Decimal,
        price: Decimal | None,
        source: str,
        filled_at: datetime | None = None,
        fees: dict[str, object] | None = None,
        fx_rate: Decimal | None = None,
        instrument_uid: str | None = None,
    ) -> str:
        """Record a fill, with the honesty of its price attached.

        `admissible_for_pnl` follows from `source` and is computed here rather
        than passed in, so a caller cannot mark an inferred price admissible by
        supplying the wrong flag. An inferred price is a guess derived from a
        position change; letting one into the realised series would teach the
        allocator an edge nobody earned.
        """
        admissible = source == FillSource.API_HISTORY and price is not None
        fill_id = new_id("fill")
        moment = filled_at or self._now()

        with self.ledger.transaction() as tx:
            event = tx.append(
                EventType.FILL_RECORDED,
                intent.t212_ticker,
                FillPayload(
                    fill_id=fill_id,
                    run_id=self.run_id,
                    t212_ticker=intent.t212_ticker,
                    side=intent.side.value,
                    quantity=quantity,
                    source=source,
                    confidence="observed" if admissible else "inferred",
                    admissible_for_pnl=admissible,
                    intent_id=intent.intent_id,
                    broker_order_id=intent.broker_order_id,
                    instrument_uid=instrument_uid or intent.instrument_uid,
                    price=price,
                    filled_at=moment.isoformat(),
                    fees=dict(fees or {}),
                    fx_rate=fx_rate,
                ),
                actor=Actor.BROKER,
                run_id=self.run_id,
            )
            tx.execute(
                "INSERT INTO fills (fill_id, intent_id, broker_order_id, t212_ticker,"
                " instrument_uid, side, quantity, price, filled_at, fees_json, fx_rate,"
                " source, confidence, admissible_for_pnl, recorded_at, recording_event_seq)"
                " VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                (
                    fill_id,
                    intent.intent_id,
                    intent.broker_order_id,
                    intent.t212_ticker,
                    instrument_uid or intent.instrument_uid,
                    intent.side.value,
                    str(quantity),
                    str(price) if price is not None else None,
                    moment.isoformat(),
                    None if not fees else repr(dict(fees)),
                    str(fx_rate) if fx_rate is not None else None,
                    source,
                    "observed" if admissible else "inferred",
                    1 if admissible else 0,
                    self._now().isoformat(),
                    event.seq,
                ),
            )
        return fill_id

    # -- protection --------------------------------------------------------

    def record_protection(
        self,
        *,
        t212_ticker: str,
        quantity: Decimal,
        protected: bool,
        stop_intent_id: str | None = None,
        stop_price: Decimal | None = None,
        entry_fill_id: str | None = None,
        unprotected_seconds: float | None = None,
        detail: str = "",
    ) -> None:
        """Record that a position gained or lost its protective stop.

        `unprotected_seconds` is the number this exists for. The window
        between an entry fill and its stop is unavoidable on a venue with no
        bracket orders, and the sizing rules assume a bound on it — so the
        actual duration has to be measurable rather than assumed. A drill that
        leaves the window open forever should be able to prove it did.
        """
        self.ledger.append(
            EventType.POSITION_PROTECTED if protected else EventType.POSITION_UNPROTECTED,
            t212_ticker,
            ProtectionPayload(
                t212_ticker=t212_ticker,
                run_id=self.run_id,
                quantity=quantity,
                protected=protected,
                stop_intent_id=stop_intent_id,
                stop_price=stop_price,
                entry_fill_id=entry_fill_id,
                unprotected_seconds=unprotected_seconds,
                detail=detail,
            ),
            actor=Actor.RISK,
            run_id=self.run_id,
        )
