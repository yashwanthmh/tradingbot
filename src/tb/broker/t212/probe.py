"""The capability probe.

Everything in `endpoints.py` and `models.py` is a reconstruction: the official
Trading 212 reference is not reachable from the build environment and the API is
in beta. This module is how those reconstructions become facts.

It walks every read-only endpoint against a **demo** account and records, for
each one: whether it answers, which auth header shape it accepts, what
`x-ratelimit-*` says the real limit is, whether the response parses against our
model, and the raw body for later replay. Anything that disagrees with the
assumed table comes out as a line in a report rather than a 429 during
reconciliation three weeks later.

It refuses to run against a live account. There is no reason to characterise an
API using real money, and `require_demo` makes that structural rather than a
matter of remembering.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from tb.broker.t212.client import T212Client
from tb.broker.t212.endpoints import READ_ONLY_ENDPOINTS, Endpoint, spec_for
from tb.broker.t212.errors import (
    AuthError,
    BrokerHttpError,
    RateLimited,
    SchemaDriftError,
    TransportError,
)
from tb.broker.t212.models import UNMAPPED_VALUES
from tb.core.clock import now_iso
from tb.ledger.events import Actor, BrokerProbedPayload, EventType
from tb.ledger.store import Ledger

# Endpoints whose rate limit makes them slow to probe, and which are not needed
# to characterise the account. `--skip-slow` omits them.
SLOW_ENDPOINTS: frozenset[Endpoint] = frozenset(
    {Endpoint.INSTRUMENTS, Endpoint.EXCHANGES, Endpoint.ACCOUNT_INFO}
)


@dataclass(slots=True)
class EndpointResult:
    endpoint: Endpoint
    ok: bool
    status_code: int | None = None
    parsed: bool = False
    item_count: int | None = None
    observed_limit: int | None = None
    observed_period_seconds: int | None = None
    waited_seconds: float = 0.0
    detail: str = ""
    sample_keys: tuple[str, ...] = ()

    def as_dict(self) -> dict[str, Any]:
        spec = spec_for(self.endpoint)
        return {
            "endpoint": self.endpoint.value,
            "method": spec.method,
            "path": spec.path,
            "ok": self.ok,
            "status_code": self.status_code,
            "parsed": self.parsed,
            "item_count": self.item_count,
            "observed_limit": self.observed_limit,
            "observed_period_s": self.observed_period_seconds,
            "configured_limit": spec.capacity,
            "configured_period_s": spec.period_seconds,
            "detail": self.detail,
            "sample_keys": list(self.sample_keys),
        }


@dataclass(slots=True)
class ProbeReport:
    environment: str
    auth_scheme: str
    results: list[EndpointResult] = field(default_factory=list)
    base_currency: str | None = None
    account_id: int | None = None
    n_positions: int | None = None
    n_open_orders: int | None = None
    disagreements: list[str] = field(default_factory=list)
    unmapped_enum_values: list[str] = field(default_factory=list)
    unknown_fields: dict[str, list[str]] = field(default_factory=dict)

    @property
    def probed(self) -> int:
        return len(self.results)

    @property
    def succeeded(self) -> int:
        return sum(1 for r in self.results if r.ok)

    @property
    def drifted(self) -> list[EndpointResult]:
        return [r for r in self.results if r.ok and not r.parsed]

    @property
    def usable(self) -> bool:
        """Whether the adapter can be trusted against this account.

        Requires the three endpoints the system cannot function without:
        equity (for every percentage cap), positions, and open orders.
        """
        essential = {Endpoint.ACCOUNT_CASH, Endpoint.PORTFOLIO, Endpoint.ORDERS_LIST}
        by_endpoint = {r.endpoint: r for r in self.results}
        return all((r := by_endpoint.get(e)) is not None and r.ok and r.parsed for e in essential)

    def summary(self) -> str:
        return (
            f"{self.succeeded}/{self.probed} endpoints answered, "
            f"{len(self.drifted)} failed to parse, "
            f"{len(self.disagreements)} rate-limit disagreement(s)"
        )


def _keys_of(body: Any) -> tuple[str, ...]:
    """The field names actually present, for the report.

    Useful precisely because the models ignore unknown fields: this is how a
    newly added field becomes visible instead of silently dropped.
    """
    if isinstance(body, dict):
        return tuple(sorted(body.keys()))
    if isinstance(body, list) and body and isinstance(body[0], dict):
        return tuple(sorted(body[0].keys()))
    return ()


def _declared_fields(endpoint: Endpoint) -> set[str]:
    from tb.broker.t212 import models as m

    mapping: dict[Endpoint, type[m.T212Model]] = {
        Endpoint.ACCOUNT_CASH: m.CashResponse,
        Endpoint.ACCOUNT_INFO: m.AccountInfoResponse,
        Endpoint.PORTFOLIO: m.PositionResponse,
        Endpoint.PORTFOLIO_TICKER: m.PositionResponse,
        Endpoint.ORDERS_LIST: m.OrderResponse,
        Endpoint.ORDER_GET: m.OrderResponse,
        Endpoint.INSTRUMENTS: m.InstrumentResponse,
        Endpoint.EXCHANGES: m.ExchangeResponse,
        Endpoint.HISTORY_ORDERS: m.HistoricalOrderResponse,
    }
    model = mapping.get(endpoint)
    if model is None:
        return set()
    names: set[str] = set()
    for name, info in model.model_fields.items():
        names.add(name)
        if info.alias:
            names.add(info.alias)
    return names


def estimate_duration_seconds(*, skip_slow: bool = False) -> float:
    """Roughly how long a probe takes on a cold governor.

    Worth printing before starting: the governor assumes its budget is spent on
    a cold boot, so the first call to a one-per-fifty-seconds endpoint waits
    fifty seconds, and a probe that looks hung is a probe someone kills.
    """
    total = 0.0
    for endpoint in READ_ONLY_ENDPOINTS:
        if skip_slow and endpoint in SLOW_ENDPOINTS:
            continue
        spec = spec_for(endpoint)
        total += spec.period_seconds / spec.capacity
    return total


def run_probe(
    client: T212Client,
    *,
    ledger: Ledger | None = None,
    skip_slow: bool = False,
    on_progress: Any = None,
) -> ProbeReport:
    """Characterise the account's API surface."""
    report = ProbeReport(environment=client.environment, auth_scheme=client.auth_scheme.value)

    # Ordered cheapest-first, so an auth failure surfaces in seconds rather
    # than after a fifty-second wait on the instruments endpoint.
    order: list[Endpoint] = [
        Endpoint.ACCOUNT_CASH,
        Endpoint.PORTFOLIO,
        Endpoint.ORDERS_LIST,
        Endpoint.ACCOUNT_INFO,
        Endpoint.HISTORY_ORDERS,
        Endpoint.EXCHANGES,
        Endpoint.INSTRUMENTS,
    ]

    for endpoint in order:
        if skip_slow and endpoint in SLOW_ENDPOINTS:
            continue
        if on_progress is not None:
            on_progress(endpoint, client.governor.wait_estimate(endpoint))
        report.results.append(_probe_one(client, endpoint, report))

        # An auth failure will not fix itself by trying more endpoints, and
        # every attempt spends rate limit that the next run needs.
        if report.results[-1].status_code in (401, 403):
            break

    report.auth_scheme = client.auth_scheme.value
    report.disagreements = client.governor.disagreements()
    report.unmapped_enum_values = sorted(UNMAPPED_VALUES)

    if ledger is not None:
        _persist(report, ledger=ledger, client=client)

    return report


def _probe_one(client: T212Client, endpoint: Endpoint, report: ProbeReport) -> EndpointResult:
    """Probe one endpoint.

    Reaches into the client's raw request path on purpose: the probe needs the
    undecoded body to report which fields actually arrived, which is the one
    thing the models cannot tell us because they ignore unknown fields.
    """
    result = EndpointResult(endpoint=endpoint, ok=False)
    try:
        if endpoint is Endpoint.ACCOUNT_CASH:
            body, _ = client._request(endpoint)
            result.sample_keys = _keys_of(body)
            cash = client.get_cash()
            result.parsed = True
            result.detail = f"free={cash.free}, total={cash.total}"
        elif endpoint is Endpoint.ACCOUNT_INFO:
            body, _ = client._request(endpoint)
            result.sample_keys = _keys_of(body)
            info = client.get_account_info()
            report.base_currency = info.currency_code
            report.account_id = info.account_id
            result.parsed = True
            result.detail = f"currency={info.currency_code}"
        elif endpoint is Endpoint.PORTFOLIO:
            body, _ = client._request(endpoint)
            result.sample_keys = _keys_of(body)
            positions = client.get_positions()
            report.n_positions = len(positions)
            result.item_count = len(positions)
            result.parsed = True
            result.detail = f"{len(positions)} position(s)"
        elif endpoint is Endpoint.ORDERS_LIST:
            body, _ = client._request(endpoint)
            result.sample_keys = _keys_of(body)
            orders = client.get_open_orders()
            report.n_open_orders = len(orders)
            result.item_count = len(orders)
            result.parsed = True
            result.detail = f"{len(orders)} open order(s)"
        elif endpoint is Endpoint.INSTRUMENTS:
            instruments = client.get_instruments()
            result.item_count = len(instruments)
            result.parsed = True
            result.detail = f"{len(instruments)} instrument(s)"
        elif endpoint is Endpoint.EXCHANGES:
            exchanges = client.get_exchanges()
            result.item_count = len(exchanges)
            result.parsed = True
            result.detail = f"{len(exchanges)} exchange(s)"
        elif endpoint is Endpoint.HISTORY_ORDERS:
            history = client.get_order_history(limit=5)
            result.item_count = len(history)
            result.parsed = True
            result.detail = f"{len(history)} historical order(s)"
        else:  # pragma: no cover - templated endpoints need an id to probe
            result.detail = "skipped: needs a path parameter"
            return result

        result.ok = True
        result.status_code = 200

        if result.sample_keys:
            declared = _declared_fields(endpoint)
            extra = [k for k in result.sample_keys if k not in declared]
            if extra:
                # Not an error — the models ignore unknown fields by design.
                # Reporting them is how a useful new field gets noticed.
                report.unknown_fields[endpoint.value] = extra

    except SchemaDriftError as drift:
        result.ok = True
        result.status_code = 200
        result.parsed = False
        result.detail = f"answered but did not parse: {drift.detail}"
    except AuthError as exc:
        result.status_code = exc.status_code or 401
        result.detail = str(exc)
    except RateLimited as exc:
        result.status_code = 429
        result.detail = str(exc)
    except BrokerHttpError as exc:
        result.status_code = exc.status_code
        result.detail = f"HTTP {exc.status_code}: {exc.body[:200]}"
    except TransportError as exc:
        result.detail = str(exc)

    observations = {o["endpoint"]: o for o in client.governor.observations()}
    observed = observations.get(endpoint.value, {})
    result.observed_limit = observed.get("observed_limit")
    result.observed_period_seconds = observed.get("observed_period_s")
    return result


def _persist(report: ProbeReport, *, ledger: Ledger, client: T212Client) -> None:
    for result in report.results:
        spec = spec_for(result.endpoint)
        agrees: bool | None = None
        if result.observed_limit is not None and result.observed_period_seconds:
            agrees = (
                abs(
                    spec.capacity / spec.period_seconds
                    - result.observed_limit / result.observed_period_seconds
                )
                < 1e-9
            )
        ledger.conn.execute(
            """
            INSERT INTO endpoint_observations (
                endpoint, environment, observed_limit, observed_period_s,
                configured_limit, configured_period_s, agrees, last_status,
                last_seen_at, note
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(endpoint) DO UPDATE SET
                environment = excluded.environment,
                observed_limit = excluded.observed_limit,
                observed_period_s = excluded.observed_period_s,
                configured_limit = excluded.configured_limit,
                configured_period_s = excluded.configured_period_s,
                agrees = excluded.agrees,
                last_status = excluded.last_status,
                last_seen_at = excluded.last_seen_at,
                note = excluded.note
            """,
            (
                result.endpoint.value,
                report.environment,
                result.observed_limit,
                result.observed_period_seconds,
                spec.capacity,
                int(spec.period_seconds),
                None if agrees is None else int(agrees),
                result.status_code,
                now_iso(),
                result.detail[:500],
            ),
        )
    ledger.conn.commit()

    ledger.append(
        EventType.BROKER_PROBED,
        report.environment,
        BrokerProbedPayload(
            environment=report.environment,
            auth_scheme=client.auth_scheme.value,
            endpoints_probed=report.probed,
            endpoints_ok=report.succeeded,
            base_currency=report.base_currency,
            observations=[r.as_dict() for r in report.results],
            disagreements=report.disagreements,
        ),
        actor=Actor.BROKER,
    )


def cache_instruments(ledger: Ledger, instruments: Any) -> int:
    """Store instrument metadata.

    Cached because the endpoint allows about one call per fifty seconds and
    returns a very large payload — reading it on a schedule would consume the
    budget that reconciliation needs.
    """
    fetched = now_iso()
    count = 0
    for instrument in instruments:
        ledger.conn.execute(
            """
            INSERT INTO instruments (
                ticker, instrument_type, isin, currency_code, short_name, full_name,
                exchange_id, working_schedule_id, min_trade_quantity, max_open_quantity,
                added_on, fetched_at, raw_json
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(ticker) DO UPDATE SET
                instrument_type = excluded.instrument_type,
                isin = excluded.isin,
                currency_code = excluded.currency_code,
                short_name = excluded.short_name,
                full_name = excluded.full_name,
                working_schedule_id = excluded.working_schedule_id,
                min_trade_quantity = excluded.min_trade_quantity,
                max_open_quantity = excluded.max_open_quantity,
                fetched_at = excluded.fetched_at
            """,
            (
                instrument.ticker,
                instrument.instrument_type,
                instrument.isin,
                instrument.currency_code,
                instrument.short_name,
                instrument.full_name,
                instrument.exchange_id,
                instrument.working_schedule_id,
                None
                if instrument.min_trade_quantity is None
                else str(instrument.min_trade_quantity),
                None if instrument.max_open_quantity is None else str(instrument.max_open_quantity),
                instrument.added_on,
                fetched,
                None,
            ),
        )
        count += 1
    ledger.conn.commit()
    return count


def cached_instruments(ledger: Ledger) -> list[Any]:
    """Instruments from the local cache, as domain objects."""
    from decimal import Decimal

    from tb.broker.port import Instrument

    rows = ledger.conn.execute("SELECT * FROM instruments ORDER BY ticker").fetchall()
    return [
        Instrument(
            ticker=row["ticker"],
            instrument_type=row["instrument_type"],
            isin=row["isin"],
            currency_code=row["currency_code"],
            short_name=row["short_name"],
            full_name=row["full_name"],
            exchange_id=row["exchange_id"],
            working_schedule_id=row["working_schedule_id"],
            min_trade_quantity=(
                None if row["min_trade_quantity"] is None else Decimal(row["min_trade_quantity"])
            ),
            max_open_quantity=(
                None if row["max_open_quantity"] is None else Decimal(row["max_open_quantity"])
            ),
            added_on=row["added_on"],
        )
        for row in rows
    ]
