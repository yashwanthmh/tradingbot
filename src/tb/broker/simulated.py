"""A deterministic broker, for the crash drills and the paper loop.

Not a mock. A mock asserts that a call was made; this models the venue's
*behaviour*, including the parts that make Trading 212 awkward — because a
simulator that only does the happy path would let the engine pass every drill
and still fail on the first real rejection.

What is modelled on purpose, each because it produces a distinct bug:

**No brackets.** `place_order` will not accept a protective level attached to
an entry, because the venue has no such call. The unprotected window between
an entry fill and its stop is therefore real here too, and measurable.

**`min_trade_quantity` rejections.** An order below the instrument minimum is
refused with the venue's own error shape. This is the one that matters most:
the *protective stop* is rejected for the same reason as the entry that
preceded it, so an engine that sizes an entry it cannot protect ends up with a
naked position — and that has to be reproducible in a test.

**The 50-pending-per-ticker ceiling.** Exhausting it causes a stop to be
rejected, which is the same failure by a different route.

**`currentPrice` only on held positions.** The read side reports no price for
an instrument not in the portfolio, which is what makes the cross-venue gate
unable to speak for a first entry.

**Injectable crash points.** `fail_at` raises *at* a named point in the
submission path, so the five drills can each be reproduced exactly. The
important ones are the two where the order reaches the venue and the caller
never learns: `post_send_pre_response` leaves a live order with no
acknowledgement, and `post_response_pre_persist` leaves one acknowledged but
unrecorded.

**Every POST is recorded.** `posts` is the evidence for "exactly one POST per
intent_id", which is the property the drills exist to assert. It counts
attempts that *reached the venue*, so a crash before the send does not
increment it and a crash after does — which is exactly the distinction the
drills are about.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import datetime
from decimal import Decimal
from enum import StrEnum

from tb.broker.port import (
    AccountInfo,
    AccountSnapshot,
    BrokerOrder,
    CashBalance,
    Instrument,
    OrderPurpose,
    OrderStatus,
    OrderType,
    PlacedOrder,
    Position,
    Side,
    TimeValidity,
)
from tb.core.clock import now_utc
from tb.core.errors import TbError
from tb.core.ids import new_id
from tb.risk.token import RiskToken


class SimulatedBrokerError(TbError):
    """The simulator refused something, the way the venue would."""


class SimulatedRejection(SimulatedBrokerError):
    """An explicit venue rejection, with the venue's own reason.

    Distinct from a transport failure on purpose: a rejection is conclusive
    (the order does not exist) while a transport failure is not (it may). The
    intent log treats them completely differently, so the simulator must be
    able to produce each without the other.
    """

    def __init__(self, reason: str, *, code: str) -> None:
        super().__init__(f"{code}: {reason}")
        self.code = code
        self.reason = reason


class SimulatedTransportFailure(SimulatedBrokerError):
    """The request did not complete. Whether the order exists is unknown.

    The failure the write-ahead log exists for. A caller that treats this as a
    rejection and retries is the double-fill path, so the simulator raises a
    type that cannot be mistaken for `SimulatedRejection`.
    """


class CrashPoint(StrEnum):
    """Where a drill injects a failure.

    The five points of the plan's crash matrix, named for what has and has not
    happened yet. `POST_FILL_PRE_STOP` is the one that is not about
    submission at all: it leaves a filled position with no protective order
    behind it, which is the state the unprotected-window sizing assumes is
    brief.
    """

    PRE_WAL = "pre_wal"
    POST_WAL_PRE_SEND = "post_wal_pre_send"
    POST_SEND_PRE_RESPONSE = "post_send_pre_response"
    POST_RESPONSE_PRE_PERSIST = "post_response_pre_persist"
    POST_FILL_PRE_STOP = "post_fill_pre_stop"

    @property
    def order_may_exist(self) -> bool:
        """Whether a crash here can leave an order at the broker.

        The property that decides what recovery is allowed to conclude. Before
        the send, absence is conclusive; at or after it, absence proves
        nothing.
        """
        return self in (
            CrashPoint.POST_SEND_PRE_RESPONSE,
            CrashPoint.POST_RESPONSE_PRE_PERSIST,
            CrashPoint.POST_FILL_PRE_STOP,
        )


@dataclass(frozen=True, slots=True)
class PostRecord:
    """One POST that reached the venue.

    Keyed by the order's identity rather than by a call counter, so "exactly
    one POST per intent" can be asserted by grouping rather than by trusting
    the order of a list.
    """

    at: datetime
    ticker: str
    side: Side
    order_type: OrderType
    quantity: Decimal
    purpose: OrderPurpose
    token_id: str
    broker_order_id: str | None = None
    outcome: str = "accepted"


@dataclass
class SimulatedBroker:
    """Implements the whole port, deterministically.

    Takes its clock as a field and holds no randomness at all: a fill either
    happens because the simulator was told to fill, or it does not. A
    simulator that filled probabilistically — or that read the wall clock —
    would make a failing drill impossible to reproduce, which is the one thing
    a drill needs.
    """

    environment: str = "demo"
    currency: str = "GBP"
    # Injectable, and not a convenience. A simulator that read the wall clock
    # could not produce a reproducible drill: a token is minted at a decision
    # instant and expires ten seconds later, so a test with a fixed `as_of`
    # would find every token expired against `now_utc()`. The default keeps
    # the paper loop honest; a drill pins it.
    clock: Callable[[], datetime] = now_utc
    equity: Decimal = Decimal("10000.00")
    free_cash: Decimal = Decimal("10000.00")
    # Instrument minimums, keyed by ticker. Absent means unconstrained, which
    # matches the read side: the venue does not always report one.
    min_trade_quantity: dict[str, Decimal] = field(default_factory=dict)
    max_open_quantity: dict[str, Decimal] = field(default_factory=dict)
    # A price per ticker, used for fills and for `currentPrice` on held
    # positions. Not a feed — the simulator is not a data provider.
    prices: dict[str, Decimal] = field(default_factory=dict)
    # Fill immediately on accept. False models a working order that has not
    # traded, which is what a limit order usually is.
    fill_on_accept: bool = True
    fail_at: CrashPoint | None = None
    # Rejections the drill wants, keyed by ticker. Each fires once, so a
    # retry after a rejection can be shown to succeed.
    reject_once: dict[str, tuple[str, str]] = field(default_factory=dict)

    _positions: dict[str, Position] = field(default_factory=dict, init=False)
    _orders: dict[str, BrokerOrder] = field(default_factory=dict, init=False)
    posts: list[PostRecord] = field(default_factory=list, init=False)
    _closed: bool = field(default=False, init=False)

    # Trading 212's documented ceiling. Exhausting it is how a protective stop
    # gets rejected for a reason that has nothing to do with the stop.
    MAX_PENDING_PER_TICKER = 50

    # -- the write half ----------------------------------------------------

    def place_order(
        self,
        token: RiskToken,
        *,
        order_type: OrderType,
        purpose: OrderPurpose,
        limit_price: Decimal | None = None,
        stop_price: Decimal | None = None,
        time_validity: TimeValidity | None = None,
    ) -> PlacedOrder:
        """Place one order, or refuse the way the venue would.

        The token is checked first and hardest. An implementation that accepted
        a token not matching the order in hand would be a hole in the risk path
        that type-checks perfectly.
        """
        token.authorises(
            t212_ticker=token.t212_ticker,
            side=token.side,
            quantity=token.quantity,
            purpose=purpose,
            at=self.clock(),
        )
        if purpose is not token.purpose:
            raise SimulatedBrokerError(
                f"the token authorises {token.purpose.value} but this order is "
                f"{purpose.value}. A token is bound to the order the engine evaluated."
            )
        self._check_order_shape(order_type, limit_price, stop_price)

        # Before the send. A crash here leaves nothing at the venue, which is
        # the one case where absence is conclusive.
        if self.fail_at is CrashPoint.POST_WAL_PRE_SEND:
            raise SimulatedTransportFailure(
                "injected crash before the send. No POST was made, so no order exists — "
                "this is the drill where recovery may safely abandon the intent."
            )

        ticker = token.t212_ticker
        quantity = token.quantity
        self._check_quantity(ticker, quantity, purpose=purpose)
        self._check_pending_ceiling(ticker, purpose=purpose)

        rejection = self.reject_once.pop(ticker, None)
        if rejection is not None:
            code, reason = rejection
            self.posts.append(
                PostRecord(
                    at=self.clock(),
                    ticker=ticker,
                    side=token.side,
                    order_type=order_type,
                    quantity=quantity,
                    purpose=purpose,
                    token_id=token.token_id,
                    outcome=f"rejected:{code}",
                )
            )
            raise SimulatedRejection(reason, code=code)

        broker_order_id = new_id("ord")
        record = PostRecord(
            at=self.clock(),
            ticker=ticker,
            side=token.side,
            order_type=order_type,
            quantity=quantity,
            purpose=purpose,
            token_id=token.token_id,
            broker_order_id=broker_order_id,
        )
        self.posts.append(record)

        # The order exists at the venue from here on, whatever happens next.
        # This is where the simulator stops being able to un-place it, which
        # is precisely the situation the two crash points below model.
        order = BrokerOrder(
            broker_order_id=broker_order_id,
            ticker=ticker,
            side=token.side,
            order_type=order_type,
            status=OrderStatus.WORKING,
            quantity=quantity,
            filled_quantity=Decimal(0),
            limit_price=limit_price,
            stop_price=stop_price,
            time_validity=time_validity,
            created_at=self.clock(),
            purpose=purpose,
        )
        self._orders[broker_order_id] = order

        if self.fail_at is CrashPoint.POST_SEND_PRE_RESPONSE:
            raise SimulatedTransportFailure(
                f"injected crash after the POST reached the venue. Order "
                f"{broker_order_id} exists and the caller never learned its id — "
                "recovery must find it by looking, and must not re-send."
            )

        if self.fill_on_accept and order_type is OrderType.MARKET:
            self._fill(broker_order_id)

        if self.fail_at is CrashPoint.POST_RESPONSE_PRE_PERSIST:
            raise SimulatedTransportFailure(
                f"injected crash after the response, before it was recorded. Order "
                f"{broker_order_id} is acknowledged at the venue and unrecorded here."
            )

        settled = self._orders[broker_order_id]
        return PlacedOrder(
            broker_order_id=broker_order_id,
            ticker=ticker,
            side=token.side,
            status=settled.status,
            quantity=quantity,
            raw_status=settled.status.value.upper(),
            accepted_at=settled.created_at,
        )

    def cancel_order(self, token: RiskToken, *, broker_order_id: str) -> None:
        """Cancel one order.

        Also behind a token, which is not obviously necessary — cancelling
        reduces risk. It is here because cancelling a *protective stop* is
        risk-*increasing*, and the only way to tell the two apart is the
        purpose the engine approved.
        """
        token.authorises(
            t212_ticker=token.t212_ticker,
            side=token.side,
            quantity=token.quantity,
            purpose=token.purpose,
            at=self.clock(),
        )
        order = self._orders.get(broker_order_id)
        if order is None:
            raise SimulatedRejection(f"no order {broker_order_id}", code="OrderNotFound")
        if order.status.is_terminal:
            raise SimulatedRejection(
                f"order {broker_order_id} is already {order.status.value}",
                code="OrderNotCancellable",
            )
        self._orders[broker_order_id] = _with_status(order, OrderStatus.CANCELLED)

    # -- the venue's refusals ----------------------------------------------

    def _check_order_shape(
        self,
        order_type: OrderType,
        limit_price: Decimal | None,
        stop_price: Decimal | None,
    ) -> None:
        if order_type.needs_limit_price and limit_price is None:
            raise SimulatedRejection("limit price required", code="LimitPriceRequired")
        if order_type.needs_stop_price and stop_price is None:
            raise SimulatedRejection("stop price required", code="StopPriceRequired")
        if not order_type.needs_limit_price and limit_price is not None:
            raise SimulatedRejection(
                f"a {order_type.value} order takes no limit price", code="UnexpectedLimitPrice"
            )

    def _check_quantity(self, ticker: str, quantity: Decimal, *, purpose: OrderPurpose) -> None:
        """The rejection that turns a sizing bug into a naked position.

        A protective stop below the instrument minimum is refused for exactly
        the same reason an entry of that size would be — so an engine that
        sizes an entry it cannot protect discovers it here, after the entry has
        filled. The message says so, because that is the expensive half.
        """
        minimum = self.min_trade_quantity.get(ticker)
        if minimum is not None and quantity < minimum:
            extra = (
                " This is a PROTECTIVE STOP: the position it was meant to protect is now unhedged."
                if purpose is OrderPurpose.PROTECTIVE_STOP
                else ""
            )
            raise SimulatedRejection(
                f"{quantity} is below the minimum {minimum} for {ticker}.{extra}",
                code="MinQuantityRequired",
            )
        maximum = self.max_open_quantity.get(ticker)
        if maximum is not None:
            held = self._positions.get(ticker)
            total = (held.quantity if held else Decimal(0)) + quantity
            if total > maximum:
                raise SimulatedRejection(
                    f"{total} would exceed the maximum open quantity {maximum} for {ticker}",
                    code="MaxQuantityExceeded",
                )

    def _check_pending_ceiling(self, ticker: str, *, purpose: OrderPurpose) -> None:
        pending = sum(
            1 for order in self._orders.values() if order.ticker == ticker and order.status.is_open
        )
        if pending >= self.MAX_PENDING_PER_TICKER:
            extra = (
                " The rejected order is a PROTECTIVE STOP, so the position is unhedged."
                if purpose is OrderPurpose.PROTECTIVE_STOP
                else ""
            )
            raise SimulatedRejection(
                f"{ticker} already has {pending} pending orders, at the venue ceiling of "
                f"{self.MAX_PENDING_PER_TICKER}.{extra}",
                code="TooManyPendingOrders",
            )

    # -- fills -------------------------------------------------------------

    def _fill(self, broker_order_id: str, *, price: Decimal | None = None) -> None:
        order = self._orders[broker_order_id]
        quantity = order.quantity or Decimal(0)
        fill_price = price or self.prices.get(order.ticker) or Decimal("100.00")

        held = self._positions.get(order.ticker)
        # Annotated, because a sell keeps whatever basis the position had —
        # which may be `None` if the position was seeded without one — while a
        # buy always computes a number.
        average: Decimal | None
        if order.side is Side.BUY:
            new_quantity = (held.quantity if held else Decimal(0)) + quantity
            # Weighted average, so a second entry does not overwrite the first
            # one's basis — an average price that jumped to the latest fill
            # would misreport P&L on every add.
            if held and held.average_price is not None and held.quantity > 0:
                total_cost = held.average_price * held.quantity + fill_price * quantity
                average = total_cost / new_quantity
            else:
                average = fill_price
        else:
            new_quantity = (held.quantity if held else Decimal(0)) - quantity
            if new_quantity < 0:
                raise SimulatedRejection(
                    f"selling {quantity} of {order.ticker} would leave "
                    f"{new_quantity}, and this account cannot go short",
                    code="InsufficientQuantity",
                )
            average = held.average_price if held else None

        if new_quantity > 0:
            self._positions[order.ticker] = Position(
                ticker=order.ticker,
                quantity=new_quantity,
                average_price=average,
                current_price=fill_price,
                initial_fill_date=(held.initial_fill_date if held else self.clock().isoformat()),
            )
        else:
            self._positions.pop(order.ticker, None)

        self._orders[broker_order_id] = BrokerOrder(
            broker_order_id=order.broker_order_id,
            ticker=order.ticker,
            side=order.side,
            order_type=order.order_type,
            status=OrderStatus.FILLED,
            quantity=order.quantity,
            filled_quantity=quantity,
            limit_price=order.limit_price,
            stop_price=order.stop_price,
            time_validity=order.time_validity,
            created_at=order.created_at,
            purpose=order.purpose,
        )

    def fill(self, broker_order_id: str, *, price: Decimal | None = None) -> None:
        """Fill a working order. For a drill that needs an explicit fill."""
        self._fill(broker_order_id, price=price)

    # -- the read half -----------------------------------------------------

    def get_account_info(self) -> AccountInfo:
        return AccountInfo(account_id=1, currency_code=self.currency)

    def get_cash(self) -> CashBalance:
        invested = sum(
            (p.quantity * (p.average_price or Decimal(0)) for p in self._positions.values()),
            Decimal(0),
        )
        return CashBalance(
            currency=self.currency,
            free=self.free_cash - invested,
            total=self.equity,
            invested=invested,
            blocked=Decimal(0),
        )

    def get_positions(self) -> tuple[Position, ...]:
        return tuple(self._positions.values())

    def get_position(self, ticker: str) -> Position | None:
        """Only a held position has a `currentPrice`.

        The venue's actual behaviour, and the reason the cross-venue price gate
        cannot speak for a first entry — which is what the two-tier symbol
        verification exists to work around.
        """
        return self._positions.get(ticker)

    def get_open_orders(self) -> tuple[BrokerOrder, ...]:
        return tuple(o for o in self._orders.values() if o.status.is_open)

    def get_order(self, broker_order_id: str) -> BrokerOrder | None:
        return self._orders.get(broker_order_id)

    def get_instruments(self) -> tuple[Instrument, ...]:
        return tuple(
            Instrument(
                ticker=ticker,
                instrument_type="STOCK",
                currency_code="USD",
                min_trade_quantity=self.min_trade_quantity.get(ticker),
                max_open_quantity=self.max_open_quantity.get(ticker),
            )
            for ticker in sorted({*self.prices, *self.min_trade_quantity})
        )

    def snapshot(self) -> AccountSnapshot:
        return AccountSnapshot(
            snap_id=new_id("snap"),
            taken_at=self.clock(),
            environment=self.environment,
            cash=self.get_cash(),
            positions=self.get_positions(),
            open_orders=self.get_open_orders(),
            account=self.get_account_info(),
        )

    def close(self) -> None:
        self._closed = True

    # -- drill helpers -----------------------------------------------------

    def posts_for_token(self, token_id: str) -> tuple[PostRecord, ...]:
        return tuple(p for p in self.posts if p.token_id == token_id)

    def posts_for_ticker(self, ticker: str) -> tuple[PostRecord, ...]:
        return tuple(p for p in self.posts if p.ticker == ticker)

    def live_orders_for(self, ticker: str) -> tuple[BrokerOrder, ...]:
        return tuple(o for o in self._orders.values() if o.ticker == ticker and o.status.is_open)

    def protective_orders_for(self, ticker: str) -> tuple[BrokerOrder, ...]:
        """Live stops behind a position. What "unprotected" is measured against."""
        return tuple(o for o in self.live_orders_for(ticker) if o.is_protective)

    def seed_position(
        self,
        ticker: str,
        *,
        quantity: Decimal,
        average_price: Decimal,
        entered_at: datetime | None = None,
    ) -> None:
        """Start with a position already held, for an exit-path drill."""
        self._positions[ticker] = Position(
            ticker=ticker,
            quantity=quantity,
            average_price=average_price,
            current_price=self.prices.get(ticker, average_price),
            initial_fill_date=(entered_at or self.clock()).isoformat(),
        )

    def clear_crash(self) -> None:
        """Stop injecting. What a restarted process looks like."""
        self.fail_at = None


def _with_status(order: BrokerOrder, status: OrderStatus) -> BrokerOrder:
    return BrokerOrder(
        broker_order_id=order.broker_order_id,
        ticker=order.ticker,
        side=order.side,
        order_type=order.order_type,
        status=status,
        quantity=order.quantity,
        filled_quantity=order.filled_quantity,
        limit_price=order.limit_price,
        stop_price=order.stop_price,
        time_validity=order.time_validity,
        created_at=order.created_at,
        purpose=order.purpose,
    )
