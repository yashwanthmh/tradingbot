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

**A market, when the paper loop needs one.** A drill pins its prices and sets
equity by hand, because it is testing what the engine does with a number. The
paper loop is testing the engine against something that moves, so with
`mark_to_market` the account follows `price_source`: fills are priced from it,
cash moves with every fill, equity is cash plus each holding at its mark, and a
working stop fires once the mark reaches it. `BarMarks` is that source for
`tb run --mode paper` — the newest close the loop itself can see, and never a
later one.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass, field, replace
from datetime import datetime
from decimal import Decimal
from enum import StrEnum

from tb.broker.port import (
    AccountInfo,
    AccountSnapshot,
    BrokerOrder,
    CashBalance,
    Execution,
    Instrument,
    OrderOutcomeUnknown,
    OrderPurpose,
    OrderRejected,
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
from tb.data.asof import BarSource, visible_bars
from tb.data.provider import Resolution
from tb.risk.token import RiskToken


class SimulatedBrokerError(TbError):
    """The simulator refused something, the way the venue would."""


class SimulatedRejection(OrderRejected):
    """An explicit venue rejection, with the venue's own reason.

    Subclasses the *port's* `OrderRejected` rather than defining a parallel
    type. That matters more than it looks: the submitter branches on refused
    versus unknown, and if the simulator's exceptions were unrelated to the
    real adapter's, the submitter would have to recognise them by name — which
    would mean production code sniffing for its test double.
    """

    def __init__(self, reason: str, *, code: str) -> None:
        super().__init__(reason, code=code)
        self.reason = reason


class SimulatedTransportFailure(OrderOutcomeUnknown):
    """The request did not complete. Whether the order exists is unknown.

    The failure the write-ahead log exists for. A caller that treats this as a
    rejection and retries is the double-fill path, so it is a sibling of
    `SimulatedRejection` under the port's taxonomy and cannot be caught by an
    `except OrderRejected`.
    """

    def __init__(self, detail: str) -> None:
        super().__init__("simulated order", detail)


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

    Takes its clock as a field and holds no randomness at all: a fill happens
    because the simulator was told to fill, or — with `mark_to_market` —
    because the mark reached a stop, and never by chance. A simulator that
    filled probabilistically — or that read the wall clock — would make a
    failing drill impossible to reproduce, which is the one thing a drill
    needs.
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
    # Where prices come from instead, when set: a ticker in, the venue's price
    # now out, or `None` when it has none. The paper loop passes `BarMarks`, so
    # a paper fill is at the close the loop decided on rather than at the
    # 100.00 a drill's fill defaults to.
    price_source: Callable[[str], Decimal | None] | None = None
    # The account follows the market: cash moves with every fill, equity is
    # cash plus each holding at its mark, and a working stop fires when the
    # mark reaches it. Off by default, because the drills and the loop suite
    # set `equity` by hand to drive the loss breakers. The paper loop turns it
    # on: a paper account whose equity never moves can never trip a breaker,
    # and a paper stop that never fires is protection nothing has exercised.
    # Cash starts at `free_cash`, and `equity` is not read in this mode.
    mark_to_market: bool = False
    # Fill immediately on accept. False models a working order that has not
    # traded, which is what a limit order usually is — and what a Trading 212
    # market order is too, since the venue fills asynchronously and a market
    # order placed outside the session waits for the open.
    fill_on_accept: bool = True
    # Shares committed to a pending sell cannot be sold again. Trading 212
    # reports this per position as `maxSell`, so the venue is expected to
    # refuse a market exit for shares its own protective stop has reserved.
    # Modelled rather than assumed away: without it an engine that forgot to
    # withdraw the stop before exiting fills in simulation and fails live.
    reserve_pending_sells: bool = True
    fail_at: CrashPoint | None = None
    # Rejections the drill wants, keyed by ticker. Each fires once, so a
    # retry after a rejection can be shown to succeed.
    reject_once: dict[str, tuple[str, str]] = field(default_factory=dict)
    # Whether order history shows finished orders yet. The venue's history is
    # rationed and can trail the fill, so an order can leave the open list
    # before history says what became of it; a drill of that gap sets this
    # False.
    publish_history: bool = True

    _positions: dict[str, Position] = field(default_factory=dict, init=False)
    _orders: dict[str, BrokerOrder] = field(default_factory=dict, init=False)
    posts: list[PostRecord] = field(default_factory=list, init=False)
    _closed: bool = field(default=False, init=False)
    _cash: Decimal = field(default=Decimal(0), init=False)
    # Price and time of each fill, which history reports and an order does not.
    _executed: dict[str, tuple[Decimal, datetime]] = field(default_factory=dict, init=False)

    # Trading 212's documented ceiling. Exhausting it is how a protective stop
    # gets rejected for a reason that has nothing to do with the stop.
    MAX_PENDING_PER_TICKER = 50

    def __post_init__(self) -> None:
        self._cash = self.free_cash

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
        # The market moves before the order arrives, not after: a stop the mark
        # has already reached has sold its shares, and a sell sent now finds
        # them gone.
        self._trigger_stops()
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
        if token.side is Side.SELL:
            self._check_free_to_sell(ticker, quantity, purpose=purpose)
        if self.mark_to_market and order_type is OrderType.MARKET:
            self._check_marketable(ticker, quantity, side=token.side)

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
        # A cancel races the market. A stop the mark has already reached filled
        # before the cancel arrived, and the refusal below says so.
        self._trigger_stops()
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

    def _reserved_by_pending_sells(self, ticker: str) -> Decimal:
        return sum(
            (
                (order.quantity or Decimal(0)) - (order.filled_quantity or Decimal(0))
                for order in self._orders.values()
                if order.ticker == ticker and order.status.is_open and order.side is Side.SELL
            ),
            Decimal(0),
        )

    def _free_to_sell(self, ticker: str) -> Decimal:
        held = self._positions.get(ticker)
        owned = held.quantity if held else Decimal(0)
        if not self.reserve_pending_sells:
            return owned
        return owned - self._reserved_by_pending_sells(ticker)

    def _check_free_to_sell(self, ticker: str, quantity: Decimal, *, purpose: OrderPurpose) -> None:
        """A sell for shares a pending sell has already committed is refused.

        The case this exists for is an exit placed while the position's own
        protective stop is working: the stop holds every share, so the exit has
        nothing to sell. An engine has to withdraw the stop first — and one that
        does not finds out here rather than on the live venue.
        """
        held = self._positions.get(ticker)
        owned = held.quantity if held else Decimal(0)
        if quantity > owned:
            # Not a reservation question at all: there are not that many shares.
            # The same refusal the fill would give, raised before the order
            # exists, since this account cannot go short.
            raise SimulatedRejection(
                f"selling {quantity} of {ticker} with {owned} held would go short, "
                "and this account cannot",
                code="InsufficientQuantity",
            )
        free = self._free_to_sell(ticker)
        if quantity > free:
            raise SimulatedRejection(
                f"selling {quantity} of {ticker}: {owned} held, "
                f"{owned - free} already committed to pending sell orders, {free} free. "
                "Withdraw the working sell (usually the position's protective stop) first.",
                code="InsufficientFreeQuantity",
            )

    def _check_marketable(self, ticker: str, quantity: Decimal, *, side: Side) -> None:
        """A market order needs a price to fill at, and a buy needs the cash.

        Only in `mark_to_market`, where the account's cash is real to the
        simulation. Refused before the order exists, as the venue refuses an
        order the account cannot pay for — a cash account cannot borrow, and a
        simulator that let cash go negative would be paper-trading on margin.
        A missing price is refused rather than filled at a stand-in: a paper
        fill at a price nothing saw would put a made-up number into equity.
        """
        price = self._price(ticker)
        if price is None or price <= 0:
            raise SimulatedRejection(
                f"no price for {ticker}, so a market order has nothing to fill at",
                code="NoPrice",
            )
        if side is Side.BUY and quantity * price > self._cash:
            raise SimulatedRejection(
                f"buying {quantity} of {ticker} at {price} costs {quantity * price}, "
                f"and {self._cash} is free",
                code="InsufficientFunds",
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
        fill_price = price if price is not None else self._price(order.ticker)
        if fill_price is None:
            if self.mark_to_market:
                # Unreachable through `place_order`, which refuses an unpriced
                # market order. Raised rather than filled at a stand-in, because
                # a stand-in here would become the account's equity.
                raise SimulatedBrokerError(f"no price for {order.ticker} to fill at")
            # A drill that pinned no price: the fill is about the order's path,
            # not its value, so any positive number serves.
            fill_price = Decimal("100.00")

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

        if self.mark_to_market:
            notional = fill_price * quantity
            self._cash += notional if order.side is Side.SELL else -notional
        self._executed[broker_order_id] = (fill_price, self.clock())

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

    # -- the market ----------------------------------------------------------

    def _price(self, ticker: str) -> Decimal | None:
        """The venue's price for `ticker` now: the source if one is set, else `prices`."""
        if self.price_source is not None:
            return self.price_source(ticker)
        return self.prices.get(ticker)

    def _trigger_stops(self) -> None:
        """Fill every working sell stop the mark has reached, at the mark.

        A stop becomes a market order once the price trades at or through its
        level, so it fills at the mark — below the stop when the price gapped
        through it, which is the overnight exposure the sizing rules budget
        for, and the reason a stop is not a guaranteed exit price. Evaluated on
        every call into the venue, since that is when time has passed.

        Only in `mark_to_market`: a drill fills a stop explicitly, or not at all.
        """
        if not self.mark_to_market:
            return
        for order in list(self._orders.values()):
            if (
                not order.status.is_open
                or order.order_type is not OrderType.STOP
                or order.side is not Side.SELL
                or order.stop_price is None
            ):
                continue
            mark = self._price(order.ticker)
            if mark is None or mark > order.stop_price:
                continue
            held = self._positions.get(order.ticker)
            if held is None or held.quantity < (order.quantity or Decimal(0)):
                # Triggered with the shares gone: a cash account cannot sell
                # what it does not hold, so the venue refuses the sell.
                self._orders[order.broker_order_id] = _with_status(order, OrderStatus.REJECTED)
                continue
            self._fill(order.broker_order_id, price=mark)

    # -- the read half -----------------------------------------------------

    def get_account_info(self) -> AccountInfo:
        return AccountInfo(account_id=1, currency_code=self.currency)

    def get_cash(self) -> CashBalance:
        self._trigger_stops()
        invested = sum(
            (p.quantity * (p.average_price or Decimal(0)) for p in self._positions.values()),
            Decimal(0),
        )
        if not self.mark_to_market:
            return CashBalance(
                currency=self.currency,
                free=self.free_cash - invested,
                total=self.equity,
                invested=invested,
                blocked=Decimal(0),
            )
        # Every holding at its mark, so equity moves with the market and the
        # loss breakers read a real number. A holding with no mark is carried
        # at its last known price rather than at zero or at cost, either of
        # which would invent a move that did not happen.
        marked = sum(
            (p.market_value or Decimal(0) for p in self.get_positions()),
            Decimal(0),
        )
        return CashBalance(
            currency=self.currency,
            free=self._cash,
            total=self._cash + marked,
            invested=invested,
            ppl=marked - invested,
            blocked=Decimal(0),
        )

    def get_positions(self) -> tuple[Position, ...]:
        self._trigger_stops()
        return tuple(self._as_reported(position) for position in self._positions.values())

    def get_position(self, ticker: str) -> Position | None:
        """Only a held position has a `currentPrice`.

        The venue's actual behaviour, and the reason the cross-venue price gate
        cannot speak for a first entry — which is what the two-tier symbol
        verification exists to work around.
        """
        self._trigger_stops()
        position = self._positions.get(ticker)
        return None if position is None else self._as_reported(position)

    def _as_reported(self, position: Position) -> Position:
        """A holding as the venue reports it.

        `maxSell` is what no pending sell has committed. In `mark_to_market`
        the price is the mark and `ppl` follows it; otherwise the price is the
        one the position was filled or seeded at, as a drill expects.
        """
        position = replace(position, max_sell=self._free_to_sell(position.ticker))
        if not self.mark_to_market:
            return position
        mark = self._price(position.ticker)
        current = mark if mark is not None else position.current_price
        ppl = (
            None
            if current is None or position.average_price is None
            else (current - position.average_price) * position.quantity
        )
        return replace(position, current_price=current, ppl=ppl)

    def get_open_orders(self) -> tuple[BrokerOrder, ...]:
        self._trigger_stops()
        return tuple(o for o in self._orders.values() if o.status.is_open)

    def get_order(self, broker_order_id: str) -> BrokerOrder | None:
        self._trigger_stops()
        return self._orders.get(broker_order_id)

    def get_executions(self, *, limit: int = 50) -> tuple[Execution, ...]:
        """Finished orders, most recently placed first, as history reports them."""
        self._trigger_stops()
        if not self.publish_history:
            return ()
        finished = [order for order in self._orders.values() if order.status.is_terminal]
        out: list[Execution] = []
        for order in reversed(finished[-limit:] if limit > 0 else []):
            price, at = self._executed.get(order.broker_order_id, (None, None))
            out.append(
                Execution(
                    broker_order_id=order.broker_order_id,
                    ticker=order.ticker,
                    status=order.status,
                    filled_quantity=order.filled_quantity or Decimal(0),
                    fill_price=price,
                    executed_at=at,
                )
            )
        return tuple(out)

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
        self._trigger_stops()
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


@dataclass
class BarMarks:
    """The paper venue's price: the newest close the loop itself can see.

    `visible_bars` at the broker's clock, so a paper fill is priced from what
    the deciding cycle saw and never from a bar that becomes knowable later —
    which would be a lookahead dressed up as a fill. It is the same raw close
    the loop sizes from, so a paper fill carries no slippage against the
    decision: the cost gate's estimate is the only cost the paper account
    pays, and a paper run tests the engine rather than the edge.

    Memoised per ticker within a clock minute, because the loop reads the venue
    many times a cycle and each read would otherwise be a store query. A memo
    is only returned at or after the instant it was read, so it can be staler
    than a fresh read but never newer.
    """

    bars: BarSource
    instruments: Mapping[str, str]
    resolution: Resolution
    clock: Callable[[], datetime] = now_utc
    _memo: dict[str, tuple[datetime, Decimal | None]] = field(default_factory=dict, init=False)

    def __call__(self, ticker: str) -> Decimal | None:
        uid = self.instruments.get(ticker)
        if uid is None:
            return None
        now = self.clock()
        memo = self._memo.get(ticker)
        if memo is not None and memo[0] <= now and _same_minute(memo[0], now):
            return memo[1]
        rows = visible_bars(self.bars, uid, self.resolution, as_of=now)
        close = rows[-1].close if rows else None
        self._memo[ticker] = (now, close)
        return close


def _same_minute(a: datetime, b: datetime) -> bool:
    return a.replace(second=0, microsecond=0) == b.replace(second=0, microsecond=0)


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
