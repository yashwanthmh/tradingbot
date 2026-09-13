"""The broker port: venue-neutral types, and the interface the engine talks to.

Two things are encoded in the types rather than left to convention, because
both are properties of the venue that no strategy may violate:

**Long-only.** The Trading 212 API covers Invest and Stocks ISA accounts, which
cannot short and cannot use leverage. So `Position.quantity` is non-negative by
construction and there is no short side to express. A strategy that wants
downside exposure has exactly one instrument available — cash — and making that
a type error rather than a runtime rejection means the search loop in M6 cannot
waste a generation discovering it.

**No brackets.** There is no OCO or bracket order type here because the venue
has none. A protective stop is a separate order with its own lifecycle, placed
after the entry fills, which is why `OrderPurpose` exists: the reconciler has to
be able to tell an unprotected position from a protected one, and that is only
possible if each order records what it was *for*.

M1 implements the read-only half of this port. `place_order` and `cancel_order`
arrive in M4, behind a `RiskToken` that only the risk engine can construct.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from decimal import Decimal
from enum import StrEnum
from typing import Protocol, runtime_checkable


class Side(StrEnum):
    BUY = "buy"
    SELL = "sell"


class OrderType(StrEnum):
    """The four order types Trading 212 supports. There is no bracket type."""

    MARKET = "market"
    LIMIT = "limit"
    STOP = "stop"
    STOP_LIMIT = "stop_limit"

    @property
    def needs_limit_price(self) -> bool:
        return self in (OrderType.LIMIT, OrderType.STOP_LIMIT)

    @property
    def needs_stop_price(self) -> bool:
        return self in (OrderType.STOP, OrderType.STOP_LIMIT)


class TimeValidity(StrEnum):
    DAY = "DAY"
    GOOD_TILL_CANCEL = "GTC"


class OrderPurpose(StrEnum):
    """Why an order exists.

    Load-bearing for two separate mechanisms. The reconciler uses it to find
    positions with no live `PROTECTIVE_STOP` behind them — the state that the
    unprotected window between entry and protection can leave behind. The rate
    governor uses it to decide whether a call may consume the reserve that
    keeps an exit possible when entries have eaten the budget.
    """

    ENTRY = "entry"
    PROTECTIVE_STOP = "protective_stop"
    TAKE_PROFIT = "take_profit"
    EXIT = "exit"
    REBALANCE = "rebalance"
    FLATTEN = "flatten"
    UNKNOWN = "unknown"

    @property
    def is_risk_reducing(self) -> bool:
        """True if this order can only ever decrease exposure.

        `REBALANCE` is deliberately excluded: it may increase a position, and a
        rebalance that consumed the reserve intended for exits would defeat the
        point of having one.
        """
        return self in (
            OrderPurpose.PROTECTIVE_STOP,
            OrderPurpose.TAKE_PROFIT,
            OrderPurpose.EXIT,
            OrderPurpose.FLATTEN,
        )


class OrderStatus(StrEnum):
    """Order lifecycle, normalised across whatever the venue calls things.

    `UNKNOWN` is a real, necessary state rather than a failure to classify. An
    order the broker no longer lists and that we cannot find in history is
    genuinely unknown, and the one thing that must never happen is quietly
    reclassifying it as `REJECTED` — that is the assumption that turns a crash
    into a double fill.
    """

    PENDING = "pending"
    WORKING = "working"
    PARTIALLY_FILLED = "partially_filled"
    FILLED = "filled"
    CANCELLED = "cancelled"
    REJECTED = "rejected"
    EXPIRED = "expired"
    UNKNOWN = "unknown"

    @property
    def is_open(self) -> bool:
        return self in (OrderStatus.PENDING, OrderStatus.WORKING, OrderStatus.PARTIALLY_FILLED)

    @property
    def is_terminal(self) -> bool:
        return self in (
            OrderStatus.FILLED,
            OrderStatus.CANCELLED,
            OrderStatus.REJECTED,
            OrderStatus.EXPIRED,
        )


@dataclass(frozen=True, slots=True)
class Instrument:
    """A tradable instrument as the broker describes it.

    `min_trade_quantity` and `max_open_quantity` are not decoration: an order
    below the minimum is rejected, and a *protective stop* rejected for being
    below the minimum leaves a naked position. M4's sizing has to respect them.
    """

    ticker: str
    instrument_type: str | None = None
    isin: str | None = None
    currency_code: str | None = None
    short_name: str | None = None
    full_name: str | None = None
    exchange_id: int | None = None
    working_schedule_id: int | None = None
    min_trade_quantity: Decimal | None = None
    max_open_quantity: Decimal | None = None
    added_on: str | None = None

    @property
    def is_equity_or_etf(self) -> bool:
        return (self.instrument_type or "").upper() in {"STOCK", "ETF", "EQUITY", ""}


@dataclass(frozen=True, slots=True)
class Position:
    """A holding. Long-only: `quantity` is non-negative."""

    ticker: str
    quantity: Decimal
    average_price: Decimal | None = None
    current_price: Decimal | None = None
    ppl: Decimal | None = None
    initial_fill_date: str | None = None
    max_buy: Decimal | None = None
    max_sell: Decimal | None = None

    def __post_init__(self) -> None:
        if self.quantity < 0:
            raise ValueError(
                f"{self.ticker}: negative quantity {self.quantity}. The Trading 212 "
                "Invest API cannot hold a short position, so this means the response "
                "was misparsed or the account is not the one we think it is."
            )

    @property
    def market_value(self) -> Decimal | None:
        if self.current_price is None:
            return None
        return self.quantity * self.current_price


@dataclass(frozen=True, slots=True)
class CashBalance:
    """The cash side of the account.

    `blocked` matters more than it looks: cash reserved against pending orders
    is not available, so sizing off `free` alone will produce rejections.
    """

    currency: str | None = None
    free: Decimal | None = None
    total: Decimal | None = None
    invested: Decimal | None = None
    ppl: Decimal | None = None
    result: Decimal | None = None
    blocked: Decimal | None = None
    pie_cash: Decimal | None = None

    @property
    def equity(self) -> Decimal | None:
        """Total account value, which is what percentage caps are applied to."""
        return self.total


@dataclass(frozen=True, slots=True)
class AccountInfo:
    account_id: int | None = None
    currency_code: str | None = None


@dataclass(frozen=True, slots=True)
class BrokerOrder:
    """An order as the broker reports it."""

    broker_order_id: str
    ticker: str
    side: Side | None = None
    order_type: OrderType | None = None
    status: OrderStatus = OrderStatus.UNKNOWN
    quantity: Decimal | None = None
    filled_quantity: Decimal | None = None
    limit_price: Decimal | None = None
    stop_price: Decimal | None = None
    time_validity: TimeValidity | None = None
    created_at: datetime | None = None
    # Purpose is ours, not the broker's: Trading 212 has no concept of "this
    # order protects that position". It comes from the intent write-ahead log
    # in M4, and is UNKNOWN for anything we did not place.
    purpose: OrderPurpose = OrderPurpose.UNKNOWN

    @property
    def is_protective(self) -> bool:
        return self.purpose is OrderPurpose.PROTECTIVE_STOP or (
            self.order_type in (OrderType.STOP, OrderType.STOP_LIMIT) and self.side is Side.SELL
        )


@dataclass(frozen=True, slots=True)
class AccountSnapshot:
    """Everything the broker will tell us at one moment.

    Taken as one object because the three axes of reconciliation have to be
    compared against each other, and comparing a position list fetched now
    against a cash balance fetched forty seconds ago (which the rate limits
    would happily produce) is how phantom mismatches appear.
    """

    snap_id: str
    taken_at: datetime
    environment: str
    cash: CashBalance
    positions: tuple[Position, ...] = ()
    open_orders: tuple[BrokerOrder, ...] = ()
    account: AccountInfo | None = None
    # Set when the snapshot could not be taken atomically enough to trust.
    staleness_warnings: tuple[str, ...] = field(default_factory=tuple)

    def position_for(self, ticker: str) -> Position | None:
        return next((p for p in self.positions if p.ticker == ticker), None)

    def orders_for(self, ticker: str) -> tuple[BrokerOrder, ...]:
        return tuple(o for o in self.open_orders if o.ticker == ticker)

    @property
    def held_tickers(self) -> tuple[str, ...]:
        return tuple(p.ticker for p in self.positions if p.quantity > 0)


@runtime_checkable
class ReadOnlyBroker(Protocol):
    """The read surface. This is all of the port that M1 implements.

    Split from the write surface on purpose: a milestone that cannot place an
    order cannot lose money, so the beta-API and symbol-mapping risks get
    retired before anything is at stake.
    """

    @property
    def environment(self) -> str:
        """`demo` or `live`. Derived from which API key is present."""
        ...

    def get_account_info(self) -> AccountInfo: ...

    def get_cash(self) -> CashBalance: ...

    def get_positions(self) -> tuple[Position, ...]: ...

    def get_position(self, ticker: str) -> Position | None: ...

    def get_open_orders(self) -> tuple[BrokerOrder, ...]: ...

    def get_order(self, broker_order_id: str) -> BrokerOrder | None: ...

    def get_instruments(self) -> tuple[Instrument, ...]: ...

    def snapshot(self) -> AccountSnapshot: ...
