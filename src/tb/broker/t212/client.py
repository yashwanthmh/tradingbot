"""The Trading 212 client. Read-only in M1.

Order placement is absent, and its absence is enforced rather than trusted:
`_request` refuses any endpoint outside `READ_ONLY_ENDPOINTS`. The point of
this milestone is that it cannot lose money, and that guarantee should not
depend on nobody adding a `POST` in six weeks.

The environment is derived from **which API key is present**, never from a mode
flag. Trading 212 mints a key per environment and there is no way to tell them
apart by inspection, so `T212_DEMO_API_KEY` and `T212_LIVE_API_KEY` are read
from deliberately different variables and the base URL follows from that. A
single key plus `MODE=demo` is one stale shell away from sending
demo-intended orders to a real account.

The auth header format is determined empirically. Trading 212's own
documentation is not reachable from the build environment, so the client tries
the bare `Authorization: <key>` form first (what the SDKs use), falls back to
`Bearer` on a 401, and records which one worked.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from enum import StrEnum
from typing import Any, TypeVar

from tb.broker.port import (
    AccountInfo,
    AccountSnapshot,
    BrokerOrder,
    CashBalance,
    Instrument,
    Position,
)
from tb.broker.t212.endpoints import READ_ONLY_ENDPOINTS, Endpoint, spec_for
from tb.broker.t212.errors import (
    AuthError,
    BrokerHttpError,
    RateLimited,
    SchemaDriftError,
)
from tb.broker.t212.models import (
    AccountInfoResponse,
    CashResponse,
    DividendResponse,
    ExchangeResponse,
    HistoricalOrderResponse,
    InstrumentResponse,
    OrderResponse,
    PositionResponse,
    parse_many,
    parse_one,
)
from tb.broker.t212.ratelimit import RateGovernor, RateLimitHeaders
from tb.broker.t212.raw_archive import RawArchive
from tb.core.clock import now_utc
from tb.core.http import HttpxTransport, Transport
from tb.core.ids import new_id
from tb.ledger.events import Actor, BrokerRateLimitedPayload, EventType
from tb.ledger.store import Ledger
from tb.ops.secrets import BrokerEnvironment, inspect_secrets

# PEP 695 generics need 3.12; this project targets 3.11.
ParsedT = TypeVar("ParsedT")


class AuthScheme(StrEnum):
    """How the key is presented.

    `RAW` first because that is what every working SDK does; the alternatives
    exist because the authoritative documentation could not be consulted and
    guessing wrong looks exactly like a bad key.
    """

    RAW = "raw"
    BEARER = "bearer"

    def header(self, api_key: str) -> dict[str, str]:
        if self is AuthScheme.BEARER:
            return {"Authorization": f"Bearer {api_key}"}
        return {"Authorization": api_key}


@dataclass(frozen=True, slots=True)
class ClientConfig:
    api_key: str
    base_url: str
    environment: str
    auth_scheme: AuthScheme = AuthScheme.RAW
    timeout: float = 20.0


class T212Client:
    """Read-only Trading 212 adapter."""

    def __init__(
        self,
        config: ClientConfig,
        *,
        transport: Transport | None = None,
        governor: RateGovernor | None = None,
        ledger: Ledger | None = None,
        run_id: str | None = None,
    ) -> None:
        self._config = config
        self._transport = transport if transport is not None else HttpxTransport()
        self._governor = governor if governor is not None else RateGovernor()
        self._archive = RawArchive(
            ledger,
            environment=config.environment,
            run_id=run_id,
            # So the key is scrubbed if it ever appears in an archived body.
            redact_values=(config.api_key,),
        )
        self._ledger = ledger
        self._auth_scheme = config.auth_scheme
        self._auth_confirmed = False

    # -- construction ------------------------------------------------------

    @classmethod
    def from_env(
        cls,
        *,
        transport: Transport | None = None,
        governor: RateGovernor | None = None,
        ledger: Ledger | None = None,
        run_id: str | None = None,
        require_demo: bool = False,
    ) -> T212Client:
        """Build a client from whichever key is in the environment.

        `require_demo` is passed by anything that must not be able to reach a
        real-money account regardless of what is set — the probe's default, and
        later the research and backtest paths.
        """
        report = inspect_secrets()
        if not report.usable:
            raise AuthError(
                0,
                report.environment.value,
                "; ".join(report.findings) or "no usable broker credentials",
            )
        if require_demo and report.environment is BrokerEnvironment.LIVE:
            raise AuthError(
                0,
                "live",
                "this command refuses to run against a real-money account; "
                "unset T212_LIVE_API_KEY and set T212_DEMO_API_KEY instead",
            )

        import os

        from tb.ops.secrets import DEMO_KEY_VAR, LIVE_KEY_VAR

        key_var = LIVE_KEY_VAR if report.environment is BrokerEnvironment.LIVE else DEMO_KEY_VAR
        api_key = os.environ[key_var].strip()
        base_url = report.environment.base_url
        assert base_url is not None  # guaranteed by report.usable

        return cls(
            ClientConfig(
                api_key=api_key,
                base_url=base_url,
                environment=report.environment.value,
            ),
            transport=transport,
            governor=governor,
            ledger=ledger,
            run_id=run_id,
        )

    # -- properties --------------------------------------------------------

    @property
    def environment(self) -> str:
        return self._config.environment

    @property
    def is_real_money(self) -> bool:
        return self._config.environment == "live"

    @property
    def auth_scheme(self) -> AuthScheme:
        return self._auth_scheme

    @property
    def governor(self) -> RateGovernor:
        return self._governor

    @property
    def archive(self) -> RawArchive:
        return self._archive

    def close(self) -> None:
        self._transport.close()

    # -- the one request path ---------------------------------------------

    def _request(
        self,
        endpoint: Endpoint,
        *,
        path_params: dict[str, str] | None = None,
        params: dict[str, Any] | None = None,
        risk_reducing: bool = False,
        timeout: float | None = None,
        _retrying_auth: bool = False,
    ) -> tuple[Any, str]:
        """Perform one governed, archived, validated request.

        Returns the decoded JSON body and the archive id, so a parse failure
        downstream can name the exact stored response.
        """
        if endpoint not in READ_ONLY_ENDPOINTS:
            raise BrokerHttpError(
                endpoint.value,
                0,
                f"{endpoint.value} is not a read-only endpoint. M1 cannot place or "
                "cancel orders; that arrives in M4 behind a RiskToken.",
            )

        spec = spec_for(endpoint)
        path = spec.path
        if path_params:
            for key, value in path_params.items():
                path = path.replace("{" + key + "}", str(value))

        self._governor.acquire(endpoint, risk_reducing=risk_reducing, timeout=timeout)

        url = f"{self._config.base_url}{path}"
        response = self._transport.request(
            spec.method,
            url,
            headers={
                **self._auth_scheme.header(self._config.api_key),
                "Accept": "application/json",
            },
            params=params,
            timeout=timeout or self._config.timeout,
        )

        rate_headers = RateLimitHeaders.from_response_headers(response.headers)
        self._governor.observe_headers(endpoint, rate_headers)

        # 429 first: the body is not useful and the governor needs to know now.
        if response.status_code == 429:
            backoff = self._governor.note_rate_limited(endpoint, rate_headers)
            self._archive.record(
                endpoint=endpoint.value,
                method=spec.method,
                url_path=path,
                status_code=429,
                raw_body=response.text,
                ratelimit=rate_headers.as_dict(),
                params=params,
                duration_ms=response.elapsed_ms,
                parse_ok=False,
                parse_error="rate limited",
            )
            if self._ledger is not None:
                self._ledger.append(
                    EventType.BROKER_RATE_LIMITED,
                    endpoint.value,
                    BrokerRateLimitedPayload(
                        endpoint=endpoint.value,
                        status_code=429,
                        retry_after_seconds=backoff,
                        observed_limit=rate_headers.limit,
                        observed_period_seconds=rate_headers.period_seconds,
                        configured_limit=spec.capacity,
                        configured_period_seconds=int(spec.period_seconds),
                    ),
                    actor=Actor.BROKER,
                )
            raise RateLimited(endpoint.value, backoff)

        if response.status_code in (401, 403):
            # Try the other auth shape once before concluding the key is bad.
            # The documentation is unreachable, so this is the empirical answer.
            if not _retrying_auth and not self._auth_confirmed:
                self._archive.record(
                    endpoint=endpoint.value,
                    method=spec.method,
                    url_path=path,
                    status_code=response.status_code,
                    raw_body=response.text,
                    ratelimit=rate_headers.as_dict(),
                    params=params,
                    duration_ms=response.elapsed_ms,
                    parse_ok=False,
                    parse_error=f"auth rejected with scheme={self._auth_scheme.value}",
                )
                self._auth_scheme = (
                    AuthScheme.BEARER if self._auth_scheme is AuthScheme.RAW else AuthScheme.RAW
                )
                return self._request(
                    endpoint,
                    path_params=path_params,
                    params=params,
                    risk_reducing=risk_reducing,
                    timeout=timeout,
                    _retrying_auth=True,
                )
            self._archive.record(
                endpoint=endpoint.value,
                method=spec.method,
                url_path=path,
                status_code=response.status_code,
                raw_body=response.text,
                ratelimit=rate_headers.as_dict(),
                params=params,
                duration_ms=response.elapsed_ms,
                parse_ok=False,
                parse_error="auth rejected with both header schemes",
            )
            raise AuthError(response.status_code, self.environment, response.text[:200])

        if not response.ok:
            self._archive.record(
                endpoint=endpoint.value,
                method=spec.method,
                url_path=path,
                status_code=response.status_code,
                raw_body=response.text,
                ratelimit=rate_headers.as_dict(),
                params=params,
                duration_ms=response.elapsed_ms,
                parse_ok=False,
                parse_error=f"HTTP {response.status_code}",
            )
            raise BrokerHttpError(endpoint.value, response.status_code, response.text)

        try:
            body = json.loads(response.text) if response.text.strip() else None
        except ValueError as exc:
            archived = self._archive.record(
                endpoint=endpoint.value,
                method=spec.method,
                url_path=path,
                status_code=response.status_code,
                raw_body=response.text,
                ratelimit=rate_headers.as_dict(),
                params=params,
                duration_ms=response.elapsed_ms,
                parse_ok=False,
                parse_error=f"body is not JSON: {exc}",
            )
            raise SchemaDriftError(
                endpoint.value, "json", str(exc), msg_id=archived.msg_id
            ) from exc

        archived = self._archive.record(
            endpoint=endpoint.value,
            method=spec.method,
            url_path=path,
            status_code=response.status_code,
            raw_body=response.text,
            ratelimit=rate_headers.as_dict(),
            params=params,
            duration_ms=response.elapsed_ms,
            parse_ok=True,
        )
        self._auth_confirmed = True
        return body, archived.msg_id

    def _parse_one(
        self, model: type[ParsedT], body: Any, endpoint: Endpoint, msg_id: str
    ) -> ParsedT:
        try:
            return parse_one(model, body, endpoint=endpoint.value, msg_id=msg_id)  # type: ignore[type-var]
        except SchemaDriftError as drift:
            self._note_drift(endpoint, msg_id, model.__name__, drift.detail)
            raise

    def _parse_many(
        self, model: type[ParsedT], body: Any, endpoint: Endpoint, msg_id: str
    ) -> list[ParsedT]:
        try:
            return parse_many(model, body, endpoint=endpoint.value, msg_id=msg_id)  # type: ignore[type-var]
        except SchemaDriftError as drift:
            self._note_drift(endpoint, msg_id, model.__name__, drift.detail)
            raise

    def _note_drift(self, endpoint: Endpoint, msg_id: str, model: str, detail: str) -> None:
        spec = spec_for(endpoint)
        if self._ledger is not None:
            self._ledger.conn.execute(
                "UPDATE broker_messages SET parse_ok = 0, parse_error = ? WHERE msg_id = ?",
                (f"{model}: {detail}", msg_id),
            )
            self._ledger.conn.commit()
        self._archive.record_drift(
            endpoint=endpoint.value,
            url_path=spec.path,
            msg_id=msg_id,
            model=model,
            detail=detail,
        )

    # -- the read surface --------------------------------------------------

    def get_account_info(self) -> AccountInfo:
        body, msg_id = self._request(Endpoint.ACCOUNT_INFO)
        return self._parse_one(AccountInfoResponse, body, Endpoint.ACCOUNT_INFO, msg_id).to_domain()

    def get_cash(self, *, currency: str | None = None) -> CashBalance:
        body, msg_id = self._request(Endpoint.ACCOUNT_CASH)
        parsed = self._parse_one(CashResponse, body, Endpoint.ACCOUNT_CASH, msg_id)
        return parsed.to_domain(currency=currency)

    def get_positions(self) -> tuple[Position, ...]:
        body, msg_id = self._request(Endpoint.PORTFOLIO)
        parsed = self._parse_many(PositionResponse, body, Endpoint.PORTFOLIO, msg_id)
        return tuple(p.to_domain() for p in parsed)

    def get_position(self, ticker: str) -> Position | None:
        body, msg_id = self._request(Endpoint.PORTFOLIO_TICKER, path_params={"ticker": ticker})
        if body is None:
            return None
        return self._parse_one(
            PositionResponse, body, Endpoint.PORTFOLIO_TICKER, msg_id
        ).to_domain()

    def get_open_orders(self) -> tuple[BrokerOrder, ...]:
        """Open orders only.

        Filled orders vanish from this endpoint, which is why an order's absence
        here says nothing about whether it executed. That question is answered
        by history, or by a position delta.
        """
        body, msg_id = self._request(Endpoint.ORDERS_LIST)
        parsed = self._parse_many(OrderResponse, body, Endpoint.ORDERS_LIST, msg_id)
        return tuple(o.to_domain() for o in parsed)

    def get_order(self, broker_order_id: str) -> BrokerOrder | None:
        try:
            body, msg_id = self._request(Endpoint.ORDER_GET, path_params={"id": broker_order_id})
        except BrokerHttpError as exc:
            if exc.status_code == 404:
                # Absent, which is not the same as never placed. The caller must
                # keep looking; only the reconciler may conclude anything.
                return None
            raise
        if body is None:
            return None
        return self._parse_one(OrderResponse, body, Endpoint.ORDER_GET, msg_id).to_domain()

    def get_instruments(self) -> tuple[Instrument, ...]:
        """Every tradable instrument.

        One call per fifty seconds and a very large payload, so callers should
        use the cached copy in the `instruments` table rather than calling this
        on a schedule.
        """
        body, msg_id = self._request(Endpoint.INSTRUMENTS)
        parsed = self._parse_many(InstrumentResponse, body, Endpoint.INSTRUMENTS, msg_id)
        return tuple(i.to_domain() for i in parsed)

    def get_exchanges(self) -> list[ExchangeResponse]:
        """Exchange working schedules — the market-hours calendar."""
        body, msg_id = self._request(Endpoint.EXCHANGES)
        return self._parse_many(ExchangeResponse, body, Endpoint.EXCHANGES, msg_id)

    def get_order_history(
        self, *, limit: int = 50, cursor: int | None = None
    ) -> list[HistoricalOrderResponse]:
        """Confirmed fills.

        Six calls a minute, against a write path that can place fifty orders a
        minute. M4 therefore distinguishes a fill confirmed here from one
        inferred from a position delta, and never lets an inferred price into
        the realised-PnL series the allocator learns from.
        """
        params: dict[str, Any] = {"limit": limit}
        if cursor is not None:
            params["cursor"] = cursor
        body, msg_id = self._request(Endpoint.HISTORY_ORDERS, params=params)
        # The endpoint is paginated: `{items: [...], nextPagePath: ...}`.
        items = body.get("items", []) if isinstance(body, dict) else body
        return self._parse_many(HistoricalOrderResponse, items, Endpoint.HISTORY_ORDERS, msg_id)

    def get_dividends(
        self, *, limit: int = 50, cursor: int | None = None
    ) -> list[DividendResponse]:
        """Cash dividends the broker actually credited.

        The evidence behind the data layer's highest-value identity check. A
        provider dividend with no matching credit, on a position held through
        the ex-date, means the action data is wrong *or* the symbol map points
        at a different company than the one in the account — and only the
        broker's own payment record can tell us which.

        Six calls a minute, same bucket class as order history.
        """
        params: dict[str, Any] = {"limit": limit}
        if cursor is not None:
            params["cursor"] = cursor
        body, msg_id = self._request(Endpoint.HISTORY_DIVIDENDS, params=params)
        items = body.get("items", []) if isinstance(body, dict) else body
        return self._parse_many(DividendResponse, items, Endpoint.HISTORY_DIVIDENDS, msg_id)

    # -- composite ---------------------------------------------------------

    def snapshot(self) -> AccountSnapshot:
        """One coherent read of the account.

        Taken as a unit because reconciliation compares its three axes against
        each other, and comparing a position list fetched now against a cash
        balance fetched forty seconds ago — which the rate limits would happily
        produce — manufactures mismatches that are not real.

        The elapsed time across the reads is measured and reported as a
        staleness warning rather than hidden, since on this venue the reads
        genuinely cannot be simultaneous.
        """
        started = now_utc()
        warnings: list[str] = []

        account = None
        try:
            account = self.get_account_info()
        except (BrokerHttpError, RateLimited) as exc:
            # Not fatal: the base currency is cached after the first successful
            # call, and the snapshot is still useful without it.
            warnings.append(f"account info unavailable: {exc}")

        cash = self.get_cash(currency=account.currency_code if account else None)
        positions = self.get_positions()
        orders = self.get_open_orders()

        taken_at = now_utc()
        spread = (taken_at - started).total_seconds()
        if spread > 10.0:
            warnings.append(
                f"the reads in this snapshot span {spread:.1f}s because of per-endpoint "
                "rate limits, so positions and cash are not simultaneous"
            )

        return AccountSnapshot(
            snap_id=new_id("snap", length=16),
            taken_at=taken_at,
            environment=self.environment,
            cash=cash,
            positions=positions,
            open_orders=orders,
            account=account,
            staleness_warnings=tuple(warnings),
        )
