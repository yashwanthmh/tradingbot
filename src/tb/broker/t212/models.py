"""Trading 212 response schemas.

**These shapes are reconstructions.** The official reference is unreachable
from the build environment and the API is in beta, so field names came from SDK
documentation and community reports. `tb broker probe` validates them against a
live demo account and archives every raw body, so the first real response either
confirms the model or produces a precise, replayable failure.

The parsing policy is deliberately asymmetric, and the asymmetry is the whole
design:

* **Unknown extra fields are ignored.** `extra="ignore"`. If Trading 212 adds a
  field, this system should not fall over. Strict-failing on an addition would
  turn a harmless upstream change into an inability to read the portfolio,
  which means an inability to reconcile or place protective stops — a parsing
  problem escalated into an unhedged position.
* **Consumed fields are required and strictly typed.** A field the system
  actually uses, gone missing or changed type, raises `SchemaDriftError`, which
  halts. There is no default and no coercion, and in particular a missing
  number never becomes `0` — a quantity of zero reads as "flat", which is the
  most dangerous possible wrong answer about a position.
* **Unrecognised enum values become `UNKNOWN`, not a guess.** A new order
  status must not be mapped to the nearest familiar one; `UNKNOWN` is handled
  safely everywhere downstream, whereas a wrong `FILLED` is not.
"""

from __future__ import annotations

from collections.abc import Iterable
from datetime import UTC, datetime
from decimal import Decimal, InvalidOperation
from typing import Annotated, Any, TypeVar

from pydantic import BaseModel, BeforeValidator, ConfigDict, Field, ValidationError

from tb.broker.port import (
    AccountInfo,
    BrokerOrder,
    CashBalance,
    Execution,
    Instrument,
    OrderStatus,
    OrderType,
    PlacedOrder,
    Position,
    Side,
    TimeValidity,
)
from tb.broker.t212.errors import SchemaDriftError


def _decimal_from_json(value: Any) -> Any:
    """Convert a JSON number to Decimal without going through binary float.

    `Decimal(0.1)` is 0.1000000000000000055511151231257827. Routing through
    `str` keeps the value the broker actually sent, which matters because these
    numbers become position sizes and monetary limits.
    """
    if value is None or isinstance(value, Decimal):
        return value
    if isinstance(value, bool):
        # bool is an int subclass; a boolean where a price belongs is drift.
        raise ValueError("boolean where a number was expected")
    if isinstance(value, (int, float, str)):
        text = str(value).strip()
        if text == "":
            return None
        try:
            parsed = Decimal(text)
        except InvalidOperation as exc:
            raise ValueError(f"not a number: {value!r}") from exc
        if not parsed.is_finite():
            raise ValueError(f"non-finite number: {value!r}")
        return parsed
    return value


# PEP 695 generics need 3.12; this project targets 3.11.
ModelT = TypeVar("ModelT", bound="T212Model")

Money = Annotated[Decimal, BeforeValidator(_decimal_from_json)]
OptMoney = Annotated[Decimal | None, BeforeValidator(_decimal_from_json)]


class T212Model(BaseModel):
    """Base for every broker response model.

    `extra="ignore"` is the fail-open half of the policy; every field declared
    without a default is the fail-closed half.
    """

    model_config = ConfigDict(extra="ignore", populate_by_name=True, frozen=True)


# --------------------------------------------------------------------------
# Status and type mapping
# --------------------------------------------------------------------------

# Trading 212's own vocabulary, as far as it is known. Anything absent maps to
# UNKNOWN rather than to the closest-looking entry.
_STATUS_MAP: dict[str, OrderStatus] = {
    "LOCAL": OrderStatus.PENDING,
    "UNCONFIRMED": OrderStatus.PENDING,
    "CONFIRMED": OrderStatus.WORKING,
    "NEW": OrderStatus.WORKING,
    "WORKING": OrderStatus.WORKING,
    "SUBMITTED": OrderStatus.WORKING,
    "REPLACING": OrderStatus.WORKING,
    "REPLACED": OrderStatus.WORKING,
    "PARTIALLY_FILLED": OrderStatus.PARTIALLY_FILLED,
    "PART_FILLED": OrderStatus.PARTIALLY_FILLED,
    "FILLED": OrderStatus.FILLED,
    "CANCELLING": OrderStatus.WORKING,
    "CANCELLED": OrderStatus.CANCELLED,
    "CANCELED": OrderStatus.CANCELLED,
    "REJECTED": OrderStatus.REJECTED,
    "EXPIRED": OrderStatus.EXPIRED,
}

_TYPE_MAP: dict[str, OrderType] = {
    "MARKET": OrderType.MARKET,
    "LIMIT": OrderType.LIMIT,
    "STOP": OrderType.STOP,
    "STOP_LIMIT": OrderType.STOP_LIMIT,
    "STOPLIMIT": OrderType.STOP_LIMIT,
}

_VALIDITY_MAP: dict[str, TimeValidity] = {
    "DAY": TimeValidity.DAY,
    "GTC": TimeValidity.GOOD_TILL_CANCEL,
    "GOOD_TILL_CANCEL": TimeValidity.GOOD_TILL_CANCEL,
}

# Populated as unrecognised values are seen, so the probe can report them
# without any single occurrence taking the system down.
UNMAPPED_VALUES: set[str] = set()


def map_status(raw: str | None) -> OrderStatus:
    if raw is None:
        return OrderStatus.UNKNOWN
    mapped = _STATUS_MAP.get(raw.strip().upper())
    if mapped is None:
        UNMAPPED_VALUES.add(f"status={raw}")
        return OrderStatus.UNKNOWN
    return mapped


def map_order_type(raw: str | None) -> OrderType | None:
    if raw is None:
        return None
    mapped = _TYPE_MAP.get(raw.strip().upper())
    if mapped is None:
        UNMAPPED_VALUES.add(f"type={raw}")
    return mapped


def map_validity(raw: str | None) -> TimeValidity | None:
    if raw is None:
        return None
    mapped = _VALIDITY_MAP.get(raw.strip().upper())
    if mapped is None:
        UNMAPPED_VALUES.add(f"timeValidity={raw}")
    return mapped


def infer_side(quantity: Decimal | None) -> Side | None:
    """Trading 212 encodes direction in the sign of `quantity`.

    A negative order quantity is a sell. Note this is the *order* field, not the
    portfolio field — a negative *position* quantity is impossible on an Invest
    account and `Position` rejects it.
    """
    if quantity is None:
        return None
    return Side.SELL if quantity < 0 else Side.BUY


# --------------------------------------------------------------------------
# Account
# --------------------------------------------------------------------------


class CashResponse(T212Model):
    """`GET /equity/account/cash`.

    `free` and `total` are consumed — `total` is the equity that every
    percentage cap is applied against — so both are required. `blocked` is cash
    reserved against pending orders and is optional because it is reported as
    null when nothing is pending; treating that null as zero happens to be
    correct here and is done explicitly at the call site rather than by schema
    default.
    """

    free: Money
    total: Money
    invested: OptMoney = None
    ppl: OptMoney = None
    result: OptMoney = None
    blocked: OptMoney = None
    pie_cash: OptMoney = Field(default=None, alias="pieCash")

    def to_domain(self, currency: str | None = None) -> CashBalance:
        return CashBalance(
            currency=currency,
            free=self.free,
            total=self.total,
            invested=self.invested,
            ppl=self.ppl,
            result=self.result,
            blocked=self.blocked,
            pie_cash=self.pie_cash,
        )


class AccountInfoResponse(T212Model):
    """`GET /equity/account/info`.

    `currencyCode` is required: it is asserted against `hard_limits.currency` at
    startup, because an absolute ceiling of 500 means something very different
    against a JPY account than a GBP one.
    """

    currency_code: str = Field(alias="currencyCode")
    account_id: int | None = Field(default=None, alias="id")

    def to_domain(self) -> AccountInfo:
        return AccountInfo(account_id=self.account_id, currency_code=self.currency_code)


# --------------------------------------------------------------------------
# Portfolio
# --------------------------------------------------------------------------


class PositionResponse(T212Model):
    """One entry from `GET /equity/portfolio`.

    `ticker` and `quantity` are required. A missing quantity must never default
    to zero: zero reads as "flat", and acting on a false flat means re-entering
    a position already held, or leaving a real one unprotected.

    `currentPrice` is optional. It is needed for the cross-venue price check,
    but its absence should block that one symbol rather than halt the system.
    """

    ticker: str = Field(min_length=1)
    quantity: Money
    average_price: OptMoney = Field(default=None, alias="averagePrice")
    current_price: OptMoney = Field(default=None, alias="currentPrice")
    ppl: OptMoney = None
    fx_ppl: OptMoney = Field(default=None, alias="fxPpl")
    initial_fill_date: str | None = Field(default=None, alias="initialFillDate")
    max_buy: OptMoney = Field(default=None, alias="maxBuy")
    max_sell: OptMoney = Field(default=None, alias="maxSell")
    pie_quantity: OptMoney = Field(default=None, alias="pieQuantity")

    def to_domain(self) -> Position:
        return Position(
            ticker=self.ticker,
            quantity=self.quantity,
            average_price=self.average_price,
            current_price=self.current_price,
            ppl=self.ppl,
            initial_fill_date=self.initial_fill_date,
            max_buy=self.max_buy,
            max_sell=self.max_sell,
        )


# --------------------------------------------------------------------------
# Orders
# --------------------------------------------------------------------------


class OrderResponse(T212Model):
    """One entry from `GET /equity/orders` or `GET /equity/orders/{id}`.

    Only `id` and `ticker` are required. Everything else is optional because an
    order in an early lifecycle state may legitimately not have it yet, and
    because the open-orders list is used mainly to answer "what is live right
    now" — a question `id` and `ticker` already answer.
    """

    order_id: int | str = Field(alias="id")
    ticker: str = Field(min_length=1)
    quantity: OptMoney = None
    filled_quantity: OptMoney = Field(default=None, alias="filledQuantity")
    limit_price: OptMoney = Field(default=None, alias="limitPrice")
    stop_price: OptMoney = Field(default=None, alias="stopPrice")
    value: OptMoney = None
    filled_value: OptMoney = Field(default=None, alias="filledValue")
    status: str | None = None
    strategy: str | None = None
    order_type: str | None = Field(default=None, alias="type")
    time_validity: str | None = Field(default=None, alias="timeValidity")
    creation_time: str | None = Field(default=None, alias="creationTime")

    def to_domain(self) -> BrokerOrder:
        return BrokerOrder(
            broker_order_id=str(self.order_id),
            ticker=self.ticker,
            side=infer_side(self.quantity),
            order_type=map_order_type(self.order_type),
            status=map_status(self.status),
            quantity=None if self.quantity is None else abs(self.quantity),
            filled_quantity=self.filled_quantity,
            limit_price=self.limit_price,
            stop_price=self.stop_price,
            time_validity=map_validity(self.time_validity),
        )


class PlacedOrderResponse(T212Model):
    """The response to a successful order POST.

    **`order_id` is the one field that is strictly required**, and the only
    place in this module where a missing field is fatal rather than tolerated.
    Everything else here follows the drift policy — optional, ignored if
    absent — because the rest is confirmation of what we already sent.

    The id is different in kind. Without it there is an order at the venue we
    cannot name: nothing to cancel, nothing to reconcile against, nothing to
    attach a fill to. An accepted order with no id is an *unknown* order, which
    is the state the whole write-ahead mechanism exists to avoid, so a response
    lacking it fails parsing and becomes schema drift — a halt — rather than a
    `PlacedOrder` with an empty string in it.

    `side` and `quantity` are not read back from the response at all. They are
    passed in from the token, because the token is what was authorised; taking
    them from the venue's echo would mean an order whose reported side differed
    from the approved one would be recorded as the venue described it.
    """

    order_id: int | str = Field(alias="id")
    ticker: str | None = None
    quantity: OptMoney = None
    status: str | None = None
    order_type: str | None = Field(default=None, alias="type")
    creation_time: str | None = Field(default=None, alias="creationTime")

    def to_domain(self, *, ticker: str, side: Side, quantity: Decimal) -> PlacedOrder:
        """Build the domain object, trusting the token over the echo.

        If the venue echoes a different ticker than the one sent, that is
        worth knowing about loudly rather than silently adopting — so it is
        checked rather than preferred.
        """
        if self.ticker is not None and self.ticker != ticker:
            raise ValueError(
                f"the venue accepted an order for {self.ticker!r} but {ticker!r} was sent. "
                "Recording the venue's answer would attach this order to the wrong "
                "instrument; recording ours would hide a real mismatch."
            )
        return PlacedOrder(
            broker_order_id=str(self.order_id),
            ticker=ticker,
            side=side,
            status=map_status(self.status),
            quantity=quantity,
            raw_status=self.status,
        )


# --------------------------------------------------------------------------
# Metadata
# --------------------------------------------------------------------------


class InstrumentResponse(T212Model):
    """One entry from `GET /equity/metadata/instruments`.

    `minTradeQuantity` is worth noticing: an order below it is rejected, and a
    *protective stop* rejected for being too small leaves a naked position. M4
    sizing reads it from here.
    """

    ticker: str = Field(min_length=1)
    instrument_type: str | None = Field(default=None, alias="type")
    isin: str | None = None
    currency_code: str | None = Field(default=None, alias="currencyCode")
    name: str | None = None
    short_name: str | None = Field(default=None, alias="shortName")
    working_schedule_id: int | None = Field(default=None, alias="workingScheduleId")
    min_trade_quantity: OptMoney = Field(default=None, alias="minTradeQuantity")
    max_open_quantity: OptMoney = Field(default=None, alias="maxOpenQuantity")
    added_on: str | None = Field(default=None, alias="addedOn")

    def to_domain(self) -> Instrument:
        return Instrument(
            ticker=self.ticker,
            instrument_type=self.instrument_type,
            isin=self.isin,
            currency_code=self.currency_code,
            short_name=self.short_name,
            full_name=self.name,
            working_schedule_id=self.working_schedule_id,
            min_trade_quantity=self.min_trade_quantity,
            max_open_quantity=self.max_open_quantity,
            added_on=self.added_on,
        )


class TimeEventResponse(T212Model):
    date: str | None = None
    event_type: str | None = Field(default=None, alias="type")


class WorkingScheduleResponse(T212Model):
    schedule_id: int | None = Field(default=None, alias="id")
    time_events: list[TimeEventResponse] = Field(default_factory=list, alias="timeEvents")


class ExchangeResponse(T212Model):
    """One entry from `GET /equity/metadata/exchanges`.

    The source of truth for market hours: Trading 212 provides no market-data
    feed, so its own working schedules are the most authoritative calendar
    available without adding another vendor.
    """

    exchange_id: int | None = Field(default=None, alias="id")
    name: str | None = None
    working_schedules: list[WorkingScheduleResponse] = Field(
        default_factory=list, alias="workingSchedules"
    )


class DividendResponse(T212Model):
    """One entry from `GET /equity/history/dividends`.

    The cash the broker *actually* paid, which is what makes dividend
    reconciliation the highest-value identity check in the data layer: a
    provider dividend with no matching credit on a position held through the
    ex-date means either the action data is wrong or the symbol map points at a
    different company than the one in the account. The second is the failure
    the symbol map exists to prevent, and this is the only evidence that
    surfaces it.

    Every field optional, per the drift policy: an unknown extra field is
    ignored, but a *consumed* field going missing halts rather than coercing to
    zero. A dividend amount silently read as 0 would reconcile as "paid
    nothing" and mask exactly the mismatch being looked for.
    """

    ticker: str | None = None
    reference: str | None = None
    amount: OptMoney = None
    amount_in_euro: OptMoney = Field(default=None, alias="amountInEuro")
    gross_amount_per_share: OptMoney = Field(default=None, alias="grossAmountPerShare")
    paid_on: str | None = Field(default=None, alias="paidOn")
    quantity: OptMoney = None
    dividend_type: str | None = Field(default=None, alias="type")


class HistoricalOrderResponse(T212Model):
    """One entry from `GET /equity/history/orders`.

    The confirmed fill record. Rate limited to six calls a minute, which cannot
    keep up with fifty orders a minute — hence M4's distinction between a fill
    confirmed here and one inferred from a position delta.
    """

    order_id: int | str | None = Field(default=None, alias="id")
    ticker: str | None = None
    ordered_quantity: OptMoney = Field(default=None, alias="orderedQuantity")
    filled_quantity: OptMoney = Field(default=None, alias="filledQuantity")
    fill_price: OptMoney = Field(default=None, alias="fillPrice")
    fill_cost: OptMoney = Field(default=None, alias="fillCost")
    fill_result: OptMoney = Field(default=None, alias="fillResult")
    status: str | None = None
    order_type: str | None = Field(default=None, alias="type")
    date_created: str | None = Field(default=None, alias="dateCreated")
    date_executed: str | None = Field(default=None, alias="dateExecuted")
    fill_type: str | None = Field(default=None, alias="fillType")
    taxes: list[dict[str, Any]] = Field(default_factory=list)


def executions_from_history(entries: Iterable[HistoricalOrderResponse]) -> tuple[Execution, ...]:
    """History entries as executions: one per order id, newest first.

    An order that filled in parts can appear once per fill, and is combined —
    quantities summed, the price volume-weighted, the charges added — because a
    caller placed one order and settles one order. Quantities are unsigned: the
    venue signs a sell's quantity negative, and the side is already known from
    the order that was placed.

    An entry with no id is skipped, since nothing we placed can be matched to
    it. A filled part with no price leaves the whole order's price `None`: the
    shares traded, and a volume-weighted average over the parts that happen to
    carry a price would be a number the venue never reported.
    """
    grouped: dict[str, list[HistoricalOrderResponse]] = {}
    for entry in entries:
        if entry.order_id is not None:
            grouped.setdefault(str(entry.order_id), []).append(entry)
    return tuple(_combine(order_id, parts) for order_id, parts in grouped.items())


def _combine(order_id: str, parts: list[HistoricalOrderResponse]) -> Execution:
    filled = Decimal(0)
    value = Decimal(0)
    priced = True
    fees: dict[str, Decimal] = {}
    executed: datetime | None = None
    for part in parts:
        quantity = abs(part.filled_quantity) if part.filled_quantity is not None else Decimal(0)
        if quantity > 0:
            if part.fill_price is None:
                priced = False
            else:
                value += quantity * part.fill_price
            filled += quantity
        for tax in part.taxes:
            charged = _charge(tax)
            if charged is not None:
                name, amount = charged
                fees[name] = fees.get(name, Decimal(0)) + amount
        moment = _executed_at(part.date_executed)
        if moment is not None and (executed is None or moment > executed):
            executed = moment
    first = parts[0]
    return Execution(
        broker_order_id=order_id,
        ticker=first.ticker or "",
        status=map_status(first.status),
        filled_quantity=filled,
        fill_price=(value / filled) if filled > 0 and priced else None,
        executed_at=executed,
        fees=tuple(sorted(fees.items())),
    )


def _charge(tax: dict[str, Any]) -> tuple[str, Decimal] | None:
    """One itemised charge, unsigned. `None` for one whose amount is unreadable."""
    try:
        amount = Decimal(str(tax.get("quantity")))
    except (InvalidOperation, ValueError):
        return None
    if not amount.is_finite():
        return None
    return str(tax.get("name") or "UNNAMED"), abs(amount)


def _executed_at(raw: str | None) -> datetime | None:
    """When a part filled. A timestamp without an offset is unknown, not UTC."""
    if not raw:
        return None
    try:
        moment = datetime.fromisoformat(raw)
    except ValueError:
        return None
    return None if moment.tzinfo is None else moment.astimezone(UTC)


# --------------------------------------------------------------------------
# Parsing entry point
# --------------------------------------------------------------------------


def parse_one(
    model: type[ModelT], body: Any, *, endpoint: str, msg_id: str | None = None
) -> ModelT:
    """Validate one object, converting failure into a halt-worthy drift error."""
    if not isinstance(body, dict):
        raise SchemaDriftError(
            endpoint,
            model.__name__,
            f"expected a JSON object, got {type(body).__name__}",
            msg_id=msg_id,
        )
    try:
        return model.model_validate(body)
    except ValidationError as exc:
        raise SchemaDriftError(endpoint, model.__name__, _summarise(exc), msg_id=msg_id) from exc


def parse_many(
    model: type[ModelT], body: Any, *, endpoint: str, msg_id: str | None = None
) -> list[ModelT]:
    """Validate a list of objects.

    One bad element fails the whole list rather than being skipped. A silently
    dropped position is an invisible holding, which is precisely the state the
    reconciler exists to make impossible.
    """
    if not isinstance(body, list):
        raise SchemaDriftError(
            endpoint,
            model.__name__,
            f"expected a JSON array, got {type(body).__name__}",
            msg_id=msg_id,
        )
    parsed: list[ModelT] = []
    for index, element in enumerate(body):
        if not isinstance(element, dict):
            raise SchemaDriftError(
                endpoint,
                model.__name__,
                f"element {index} is {type(element).__name__}, not an object",
                msg_id=msg_id,
            )
        try:
            parsed.append(model.model_validate(element))
        except ValidationError as exc:
            raise SchemaDriftError(
                endpoint,
                model.__name__,
                f"element {index}: {_summarise(exc)}",
                msg_id=msg_id,
            ) from exc
    return parsed


def _summarise(exc: ValidationError) -> str:
    """Name the offending fields.

    "3 validation errors" is useless at 09:30; "missing: total, quantity" says
    what changed and where to look in the archived body.
    """
    parts = []
    for error in exc.errors()[:6]:
        location = ".".join(str(p) for p in error["loc"]) or "(root)"
        parts.append(f"{location}: {error['msg']}")
    if len(exc.errors()) > 6:
        parts.append(f"... and {len(exc.errors()) - 6} more")
    return "; ".join(parts)
