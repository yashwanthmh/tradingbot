"""Trading 212 endpoint registry and rate limits.

**These limits are conservative reconstructions, not documented fact.** The
official reference is not reachable from the build environment and the API is
in beta, so they came from SDK documentation and community reports. Two things
follow, and both are implemented rather than hoped for:

* `tb broker probe` reads the real `x-ratelimit-*` headers from a live demo
  account and writes what it finds into `endpoint_observations`, reporting any
  disagreement with the table below.
* At runtime the server's own headers override the local estimate on every
  response. Our count is a guess; the broker's count is the one that returns
  429.

The arithmetic worth internalising, because it constrains the whole design:
limit-class orders go out at **one per two seconds**, and a protective stop is
a limit-class order. Turning over a portfolio of N symbols therefore needs 2N
seconds of governor budget for protection alone — 50 seconds for 25 symbols,
which is longer than the minute bar that generated the signal. Universe size
is a consequence of this table, not a preference.

Note also that writes outrun reads: market orders POST at 50/60s while the
order list reads at 1/5s and fill history at 6/60s. The system can create
state roughly five times faster than it can observe it, which is why M4
synthesises exactly-once submission client-side instead of polling for truth.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum


class Endpoint(StrEnum):
    ACCOUNT_CASH = "account_cash"
    ACCOUNT_INFO = "account_info"
    PORTFOLIO = "portfolio"
    PORTFOLIO_TICKER = "portfolio_ticker"
    ORDERS_LIST = "orders_list"
    ORDER_GET = "order_get"
    ORDER_MARKET = "order_market"
    ORDER_LIMIT = "order_limit"
    ORDER_STOP = "order_stop"
    ORDER_STOP_LIMIT = "order_stop_limit"
    ORDER_CANCEL = "order_cancel"
    INSTRUMENTS = "instruments"
    EXCHANGES = "exchanges"
    HISTORY_ORDERS = "history_orders"
    HISTORY_DIVIDENDS = "history_dividends"
    HISTORY_TRANSACTIONS = "history_transactions"


@dataclass(frozen=True, slots=True)
class EndpointSpec:
    """One endpoint: where it is, and how often it may be called.

    `reserve_for_risk_reducing` withholds part of a bucket so that an exit or a
    protective stop can still get through after entries have spent the budget.
    It only means anything where `capacity > 1`; for the many endpoints limited
    to a single call per period there is nothing to divide, and priority
    ordering among waiters does that job instead (see `ratelimit.py`).
    """

    endpoint: Endpoint
    method: str
    path: str
    capacity: int
    period_seconds: float
    reserve_for_risk_reducing: int = 0
    # True where the path contains a placeholder, so the archive records the
    # template rather than one instrument's URL.
    templated: bool = False
    note: str = ""

    @property
    def is_write(self) -> bool:
        return self.method in {"POST", "DELETE"}


# The `/equity` prefix is applied to most paths but, per the sources available,
# dividends and transactions sit directly under `/history`. The probe checks
# both spellings and records which one answers, because guessing wrong here
# looks identical to an auth failure.
_SPECS: tuple[EndpointSpec, ...] = (
    EndpointSpec(
        Endpoint.ACCOUNT_CASH,
        "GET",
        "/equity/account/cash",
        capacity=1,
        period_seconds=5.0,
        note="account equity; drives every percentage cap",
    ),
    EndpointSpec(
        Endpoint.ACCOUNT_INFO,
        "GET",
        "/equity/account/info",
        capacity=1,
        period_seconds=30.0,
        note="base currency; asserted against hard_limits.currency at startup",
    ),
    EndpointSpec(
        Endpoint.PORTFOLIO,
        "GET",
        "/equity/portfolio",
        capacity=1,
        period_seconds=1.0,
        note="the only source that survives order purging; reconciliation axis 3",
    ),
    EndpointSpec(
        Endpoint.PORTFOLIO_TICKER,
        "GET",
        "/equity/portfolio/{ticker}",
        capacity=1,
        period_seconds=1.0,
        templated=True,
    ),
    EndpointSpec(
        Endpoint.ORDERS_LIST,
        "GET",
        "/equity/orders",
        capacity=1,
        period_seconds=5.0,
        note="open orders only; filled orders vanish from here",
    ),
    EndpointSpec(
        Endpoint.ORDER_GET,
        "GET",
        "/equity/orders/{id}",
        capacity=1,
        period_seconds=1.0,
        templated=True,
    ),
    # The one generous bucket. Reserve a fifth of it so a flatten is always
    # possible even if a runaway entry loop has consumed the rest.
    EndpointSpec(
        Endpoint.ORDER_MARKET,
        "POST",
        "/equity/orders/market",
        capacity=50,
        period_seconds=60.0,
        reserve_for_risk_reducing=10,
        note="market orders only fill during market hours",
    ),
    EndpointSpec(
        Endpoint.ORDER_LIMIT,
        "POST",
        "/equity/orders/limit",
        capacity=1,
        period_seconds=2.0,
    ),
    # Protective stops come through here, which is why 2s/order sets the
    # universe ceiling.
    EndpointSpec(
        Endpoint.ORDER_STOP,
        "POST",
        "/equity/orders/stop",
        capacity=1,
        period_seconds=2.0,
        note="protective stops; the binding constraint on portfolio turnover",
    ),
    EndpointSpec(
        Endpoint.ORDER_STOP_LIMIT,
        "POST",
        "/equity/orders/stop_limit",
        capacity=1,
        period_seconds=2.0,
    ),
    EndpointSpec(
        Endpoint.ORDER_CANCEL,
        "DELETE",
        "/equity/orders/{id}",
        capacity=50,
        period_seconds=60.0,
        reserve_for_risk_reducing=10,
        templated=True,
        note="cancelling is risk-reducing; the orphan sweeper depends on it",
    ),
    EndpointSpec(
        Endpoint.INSTRUMENTS,
        "GET",
        "/equity/metadata/instruments",
        capacity=1,
        period_seconds=50.0,
        note="very large payload, very tight limit; cached in the instruments table",
    ),
    EndpointSpec(
        Endpoint.EXCHANGES,
        "GET",
        "/equity/metadata/exchanges",
        capacity=1,
        period_seconds=30.0,
        note="working schedules; the source of truth for market hours",
    ),
    EndpointSpec(
        Endpoint.HISTORY_ORDERS,
        "GET",
        "/equity/history/orders",
        capacity=6,
        period_seconds=60.0,
        note="the fill record. At 6/60s it cannot keep up with 50 orders/min, "
        "which is why fills carry a source field distinguishing confirmed "
        "from position-delta-inferred",
    ),
    EndpointSpec(
        Endpoint.HISTORY_DIVIDENDS,
        "GET",
        "/history/dividends",
        capacity=6,
        period_seconds=60.0,
    ),
    EndpointSpec(
        Endpoint.HISTORY_TRANSACTIONS,
        "GET",
        "/history/transactions",
        capacity=6,
        period_seconds=60.0,
    ),
)

SPECS: dict[Endpoint, EndpointSpec] = {spec.endpoint: spec for spec in _SPECS}

# Endpoints the read-only M1 client is allowed to touch. Enforced by the client
# rather than left to discipline: the milestone's whole value is that it cannot
# place an order, and that guarantee should not depend on nobody adding a call.
READ_ONLY_ENDPOINTS: frozenset[Endpoint] = frozenset(
    {
        Endpoint.ACCOUNT_CASH,
        Endpoint.ACCOUNT_INFO,
        Endpoint.PORTFOLIO,
        Endpoint.PORTFOLIO_TICKER,
        Endpoint.ORDERS_LIST,
        Endpoint.ORDER_GET,
        Endpoint.INSTRUMENTS,
        Endpoint.EXCHANGES,
        Endpoint.HISTORY_ORDERS,
        Endpoint.HISTORY_DIVIDENDS,
        Endpoint.HISTORY_TRANSACTIONS,
    }
)


def spec_for(endpoint: Endpoint) -> EndpointSpec:
    return SPECS[endpoint]


def seconds_to_place_protective_stops(symbol_count: int) -> float:
    """Governor time needed to protect `symbol_count` fresh positions.

    Exposed because it is the arithmetic behind `max_universe_symbols`, and a
    number in a docstring gets stale while a function gets tested.
    """
    stop = SPECS[Endpoint.ORDER_STOP]
    return symbol_count * (stop.period_seconds / stop.capacity)
