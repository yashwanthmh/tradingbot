"""Alpaca market data, free tier.

The primary provider, and the one honest thing to say about it is in
`capabilities`: **`consolidated_tape=False`**.

The free tier serves IEX only — roughly 2% of US consolidated volume. That is
not a freshness problem, which is how it is usually described, and it is the
reason this provider alone does not unlock minute-resolution live trading. The
consequences are structural:

* Its minute *close* is the last IEX print, not the consolidated last price.
* Its *high and low* are IEX's extremes, not the session's.
* A thin name has minutes with **no IEX print at all**, so the bar is simply
  absent rather than flat — which is why nothing in this layer forward-fills.

The systematic disagreement against the consolidated tape runs 5-20bps, the
same order as the entire gross edge a minute-bar strategy would be trading. So
`data.allowed_live_resolutions` is `[daily]` until `tb data bakeoff` measures
that number and says otherwise. The bake-off is what this provider exists to
be measured by.

What it does better than Yahoo, and why it is nonetheless primary:

* **`adjustment=raw` is honoured.** The OHLC is what the venue printed, so
  `returns_raw_prices=True` and a split does not retroactively rewrite stored
  history. Yahoo cannot offer this at all.
* **Low delay on IEX.** Free IEX data is not the 15-minute-delayed SIP feed, so
  it can drive a live decision once the representativeness question is settled.
* **Corporate actions carry a declaration date**, making `known_at` a fact
  rather than "whenever we happened to look".

Credentials are read from `ALPACA_DATA_KEY_ID` / `ALPACA_DATA_SECRET_KEY` —
names chosen to say *data*, following the same discipline as the broker keys.
Alpaca's own SDK reads `APCA_API_KEY_ID`, and on a funded Alpaca account that
same key can place orders. This process executes through Trading 212 and has no
business holding a key that can trade somewhere else, so the SDK names are not
read as a fallback; they are reported as a finding.
"""

from __future__ import annotations

import json
import os
from datetime import UTC, datetime, timedelta
from decimal import Decimal
from fractions import Fraction
from typing import Any

from tb.core.clock import now_utc
from tb.core.errors import TransportError
from tb.core.http import HttpxTransport, Transport
from tb.data.pacing import ALPACA_PACING, PacingSpec, ProviderPacer
from tb.data.provider import (
    US_EASTERN,
    Bar,
    BarBatch,
    DataError,
    Provenance,
    ProviderCapabilities,
    ProviderUnavailable,
    RawAction,
    Resolution,
    Session,
    TimestampConvention,
    classify_us_session,
    knowledge_time,
    normalise_bar_open,
    parse_price,
)

DATA_BASE_URL = "https://data.alpaca.markets"

KEY_ID_VAR = "ALPACA_DATA_KEY_ID"
# The *name* of the variable, never its value. Nothing in this module stores a
# credential at module scope, and the archive scrubs bodies before writing.
SECRET_VAR = "ALPACA_DATA_SECRET_KEY"  # noqa: S105
# The names Alpaca's SDK uses. Read only to *report* them, never to authenticate
# with: a key under this name is likely a full-permission trading key.
SDK_KEY_VARS = ("APCA_API_KEY_ID", "APCA_API_SECRET_KEY")

# The free plan. Pinned as a constant rather than a parameter: `sip` would 403
# on a free account, and worse, a paid account switching to it mid-history
# would splice two feeds with a systematic 5-20bps step at the join.
FREE_FEED = "iex"

_TIMEFRAME = {
    Resolution.MINUTE: "1Min",
    Resolution.HOURLY: "1Hour",
    Resolution.DAILY: "1Day",
}

# IEX data is not the delayed SIP feed, so the delay here is network latency
# plus Alpaca's own aggregation, not a licensing embargo. Still a conservative
# default until the probe measures it.
_DECLARED_DELAY = {
    Resolution.MINUTE: 30.0,
    Resolution.HOURLY: 30.0,
    Resolution.DAILY: 60.0,
}

# Alpaca's IEX archive effectively begins in 2016, so the ten-year daily
# backfill the plan calls for has to come from Yahoo. These figures bound what
# this provider will be *asked* for; `tb data audit` reports where its history
# actually starts, which is the number to trust.
_MAX_HISTORY_DAYS = {
    Resolution.MINUTE: 2555,
    Resolution.HOURLY: 2555,
    Resolution.DAILY: 3650,
}

# Alpaca pages long windows. A backfill that ignored `next_page_token` would
# return the first page and look complete, so paging is mandatory — and capped,
# because an unbounded loop against a paid-by-request API is its own hazard.
MAX_PAGES = 200
PAGE_LIMIT = 10_000


class AlpacaProvider:
    """Free-tier IEX bars and corporate actions."""

    name = "alpaca"

    def __init__(
        self,
        key_id: str,
        secret_key: str,
        *,
        transport: Transport | None = None,
        pacing: PacingSpec | None = None,
        pacer: ProviderPacer | None = None,
        base_url: str = DATA_BASE_URL,
        timeout: float = 30.0,
        archive: Any = None,
        feed: str = FREE_FEED,
    ) -> None:
        if not key_id or not secret_key:
            raise ProviderUnavailable(
                f"alpaca needs both {KEY_ID_VAR} and {SECRET_VAR}; one of them is empty"
            )
        self._key_id = key_id
        self._secret_key = secret_key
        self._transport = transport if transport is not None else HttpxTransport()
        self._pacer = pacer or ProviderPacer(spec=pacing or ALPACA_PACING)
        self._base_url = base_url.rstrip("/")
        self._timeout = timeout
        self._archive = archive
        self._feed = feed

    @classmethod
    def from_env(cls, **kwargs: Any) -> AlpacaProvider:
        """Build from the environment, naming the exact variables if absent."""
        key_id = (os.environ.get(KEY_ID_VAR) or "").strip()
        secret = (os.environ.get(SECRET_VAR) or "").strip()
        if not key_id or not secret:
            hint = ""
            if any(os.environ.get(var) for var in SDK_KEY_VARS):
                hint = (
                    f" ({'/'.join(SDK_KEY_VARS)} is set, but those are deliberately not read: "
                    "on a funded Alpaca account such a key can place orders, and this process "
                    "executes through Trading 212. Generate data-only keys and export them "
                    "under the names above.)"
                )
            raise ProviderUnavailable(
                f"alpaca credentials not found. Set {KEY_ID_VAR} and {SECRET_VAR}.{hint}"
            )
        return cls(key_id, secret, **kwargs)

    @classmethod
    def credential_findings(cls) -> tuple[str, ...]:
        """Anything worth reporting about the Alpaca credentials in scope.

        Surfaced by `tb status` rather than raised, because an over-privileged
        key is a posture problem, not a reason to refuse to fetch a price.
        """
        findings: list[str] = []
        for var in SDK_KEY_VARS:
            if os.environ.get(var):
                findings.append(
                    f"{var} is set. This process never reads it, but a key under that name "
                    "usually carries trading permission on Alpaca — unnecessary blast radius "
                    f"for a process that only needs prices. Use {KEY_ID_VAR}/{SECRET_VAR} "
                    "with data-only keys."
                )
        return tuple(findings)

    @property
    def capabilities(self) -> ProviderCapabilities:
        return ProviderCapabilities(
            name=self.name,
            resolutions=frozenset(_TIMEFRAME),
            # Alpaca stamps a bar with the start of its interval. Verified
            # against the daily bar, whose `t` is midnight of the session date
            # rather than the following midnight.
            timestamp_convention=TimestampConvention.BAR_OPEN,
            declared_delay_seconds=dict(_DECLARED_DELAY),
            max_history_days=dict(_MAX_HISTORY_DAYS),
            supports_extended_hours=True,
            supports_corporate_actions=True,
            # The headline limitation, and the reason minute-resolution live
            # trading is gated on the bake-off rather than on this flag.
            consolidated_tape=self._feed != "iex",
            # The headline advantage: `adjustment=raw` is honoured, so stored
            # prices stay what the venue printed even after a split.
            returns_raw_prices=True,
            note=(
                f"feed={self._feed}. IEX is ~2% of consolidated volume: its minute close is "
                "not the consolidated last price and thin names have minutes with no print. "
                "The systematic disagreement is the same order as the gross edge, which is "
                "what tb data bakeoff measures. Raw (unadjusted) OHLC; IEX history starts "
                "around 2016, so a ten-year daily backfill needs Yahoo."
            ),
        )

    def close(self) -> None:
        self._transport.close()

    # -- bars --------------------------------------------------------------

    def fetch_bars(
        self,
        symbol: str,
        *,
        instrument_uid: str,
        resolution: Resolution,
        start: datetime,
        end: datetime,
        provenance: Provenance = Provenance.BACKFILL,
        include_extended: bool = False,
    ) -> BarBatch:
        warnings: list[str] = []
        bars: list[Bar] = []
        msg_id: str | None = None
        page_token: str | None = None
        ingested = now_utc()

        for _page in range(MAX_PAGES):
            params: dict[str, Any] = {
                "symbols": symbol,
                "timeframe": _TIMEFRAME[resolution],
                "start": _rfc3339(start),
                "end": _rfc3339(end),
                # Raw prices, always. A store of vendor-adjusted bars cannot
                # reconstruct what a stop would have been triggered by.
                "adjustment": "raw",
                "feed": self._feed,
                "limit": PAGE_LIMIT,
                "sort": "asc",
            }
            if page_token:
                params["page_token"] = page_token

            payload, page_msg_id = self._request("/v2/stocks/bars", params=params)
            msg_id = msg_id or page_msg_id

            page_bars, page_warnings = self._parse_bars(
                payload,
                symbol=symbol,
                instrument_uid=instrument_uid,
                resolution=resolution,
                provenance=provenance,
                include_extended=include_extended,
                ingested=ingested,
            )
            bars.extend(page_bars)
            warnings.extend(page_warnings)

            page_token = payload.get("next_page_token") if isinstance(payload, dict) else None
            if not page_token:
                break
        else:
            # Falling out of the loop with a token left means the window is
            # larger than MAX_PAGES can cover. Reported loudly: silently
            # truncated history is a backtest that trades a window it never saw.
            warnings.append(
                f"stopped after {MAX_PAGES} pages with more data available for {symbol}; "
                "the requested window is incomplete — narrow it and refetch"
            )

        return BarBatch(
            bars=tuple(bars),
            provider=self.name,
            symbol=symbol,
            resolution=resolution,
            requested_start=start,
            requested_end=end,
            raw_msg_id=msg_id,
            warnings=tuple(warnings),
        )

    def latest_bar(self, symbol: str, *, instrument_uid: str, resolution: Resolution) -> Bar | None:
        """The most recent finished bar, or None.

        `/bars/latest` will return the bar currently being formed, whose high,
        low and volume are partial. `is_settled_at` drops it: storing a partial
        bar and replacing it when the minute ends would register as a revision
        on every single poll, burying the revisions that actually matter.
        """
        payload, _ = self._request(f"/v2/stocks/{symbol}/bars/latest", params={"feed": self._feed})
        if not isinstance(payload, dict):
            raise DataError(f"alpaca latest-bar response for {symbol} is not an object")
        entry = payload.get("bar")
        if not isinstance(entry, dict):
            return None

        now = now_utc()
        bar = self._build_bar(
            entry,
            instrument_uid=instrument_uid,
            resolution=resolution,
            provenance=Provenance.LIVE,
            ingested=now,
            currency=payload.get("currency"),
        )
        if bar is None or not bar.is_settled_at(now):
            return None
        return bar

    def _parse_bars(
        self,
        payload: Any,
        *,
        symbol: str,
        instrument_uid: str,
        resolution: Resolution,
        provenance: Provenance,
        include_extended: bool,
        ingested: datetime,
    ) -> tuple[list[Bar], list[str]]:
        if not isinstance(payload, dict):
            raise DataError(f"alpaca bars response for {symbol} is not an object")

        container = payload.get("bars")
        if container is None:
            # A window with no trading in it. Distinct from an error, and not a
            # warning either — a weekend is allowed to be empty.
            return [], []
        if not isinstance(container, dict):
            raise DataError(
                f"alpaca returned bars for {symbol} as {type(container).__name__}, expected an "
                "object keyed by symbol. The multi-symbol endpoint's shape has changed."
            )
        entries = container.get(symbol)
        if entries is None:
            # We asked for exactly one symbol; anything else is a mismatch worth
            # noticing rather than treating as "no data".
            other = sorted(container)
            if other:
                raise DataError(f"alpaca returned bars for {other} when {symbol} was requested")
            return [], []
        if not isinstance(entries, list):
            raise DataError(f"alpaca bars for {symbol} is not a list")

        currency = payload.get("currency")
        bars: list[Bar] = []
        warnings: list[str] = []
        skipped_extended = 0

        for entry in entries:
            if not isinstance(entry, dict):
                warnings.append(f"skipped a non-object bar entry for {symbol}")
                continue
            try:
                bar = self._build_bar(
                    entry,
                    instrument_uid=instrument_uid,
                    resolution=resolution,
                    provenance=provenance,
                    ingested=ingested,
                    currency=currency,
                )
            except DataError as exc:
                # One bad print must not fail a whole backfill, and must not
                # enter the store either. Reported, dropped, and visible in the
                # batch's warnings so `tb data audit` can count them.
                warnings.append(f"dropped bar for {symbol}: {exc}")
                continue
            if bar is None:
                continue
            if not include_extended and bar.session is not Session.REGULAR:
                skipped_extended += 1
                continue
            bars.append(bar)

        if skipped_extended:
            warnings.append(f"{skipped_extended} extended-hours bar(s) excluded as requested")
        return bars, warnings

    def _build_bar(
        self,
        entry: dict[str, Any],
        *,
        instrument_uid: str,
        resolution: Resolution,
        provenance: Provenance,
        ingested: datetime,
        currency: object,
    ) -> Bar | None:
        stamped = entry.get("t")
        if stamped is None:
            raise DataError("alpaca bar has no timestamp")
        bar_open = normalise_bar_open(
            _parse_rfc3339(str(stamped)),
            convention=TimestampConvention.BAR_OPEN,
            resolution=resolution,
            # Alpaca serves US equities only, and stamps a daily bar at local
            # midnight Eastern — which is the previous UTC day for four months
            # of the year. Reading the session date in Eastern rather than UTC
            # is what keeps a daily bar on the right date across DST.
            session_tz=US_EASTERN,
        )

        for field_name in ("o", "h", "l", "c"):
            if entry.get(field_name) is None:
                # Never coerced to zero or to the previous value. A missing
                # price is missing; the audit's job is to report the hole.
                raise DataError(f"alpaca bar at {bar_open.isoformat()} is missing '{field_name}'")

        volume = entry.get("v")
        return Bar(
            instrument_uid=instrument_uid,
            resolution=resolution,
            bar_open_utc=bar_open,
            available_at_utc=knowledge_time(
                bar_open=bar_open,
                resolution=resolution,
                provenance=provenance,
                delay_seconds=_DECLARED_DELAY[resolution],
                ingested_at=ingested if provenance is Provenance.LIVE else None,
            ),
            ingested_at_utc=ingested,
            provider=self.name,
            provenance=provenance,
            session=classify_us_session(bar_open, resolution),
            open=parse_price(entry["o"]),
            high=parse_price(entry["h"]),
            low=parse_price(entry["l"]),
            close=parse_price(entry["c"]),
            volume=None if volume is None else int(volume),
            currency=str(currency) if currency else "USD",
        )

    # -- corporate actions -------------------------------------------------

    def fetch_actions(
        self, symbol: str, *, instrument_uid: str, start: datetime, end: datetime
    ) -> tuple[RawAction, ...]:
        """Splits and dividends, with declaration dates where Alpaca has them.

        The split ratios arrive as `old_rate`/`new_rate` integers and are kept
        as integers. A 3-for-1 stored as 0.3333333 drifts over a twenty-year
        factor product, and `canonical_json` would faithfully hash the drift.
        """
        actions: list[RawAction] = []
        page_token: str | None = None
        observed = now_utc()

        for _ in range(MAX_PAGES):
            params: dict[str, Any] = {
                "symbols": symbol,
                "types": "forward_split,reverse_split,cash_dividend",
                "start": start.astimezone(UTC).date().isoformat(),
                "end": end.astimezone(UTC).date().isoformat(),
                "limit": 1000,
            }
            if page_token:
                params["page_token"] = page_token

            payload, _ = self._request("/v1/corporate-actions", params=params)
            if not isinstance(payload, dict):
                raise DataError(f"alpaca corporate-actions response for {symbol} is not an object")
            container = payload.get("corporate_actions") or {}
            if not isinstance(container, dict):
                raise DataError(f"alpaca corporate_actions for {symbol} is not an object")

            for key in ("forward_splits", "reverse_splits"):
                for entry in container.get(key) or []:
                    action = _split_action(entry, instrument_uid=instrument_uid, observed=observed)
                    if action is not None:
                        actions.append(action)

            for entry in container.get("cash_dividends") or []:
                action = _dividend_action(entry, instrument_uid=instrument_uid, observed=observed)
                if action is not None:
                    actions.append(action)

            page_token = payload.get("next_page_token")
            if not page_token:
                break

        return tuple(sorted(actions, key=lambda a: (a.effective_date, a.action_type)))

    # -- transport ---------------------------------------------------------

    def _request(self, path: str, *, params: dict[str, Any]) -> tuple[Any, str | None]:
        self._pacer.acquire()
        url = f"{self._base_url}{path}"
        response = self._transport.request(
            "GET",
            url,
            headers={
                "Accept": "application/json",
                "APCA-API-KEY-ID": self._key_id,
                "APCA-API-SECRET-KEY": self._secret_key,
            },
            params=params,
            timeout=self._timeout,
        )

        msg_id: str | None = None
        if self._archive is not None:
            archived = self._archive.record(
                endpoint="alpaca_data",
                method="GET",
                url_path=path,
                status_code=response.status_code,
                raw_body=response.text,
                params=params,
                duration_ms=response.elapsed_ms,
                parse_ok=response.ok,
                parse_error=None if response.ok else f"HTTP {response.status_code}",
            )
            msg_id = archived.msg_id

        if response.status_code == 429:
            retry_after = response.headers.get("retry-after")
            backoff = self._pacer.note_throttled(
                retry_after_seconds=float(retry_after) if retry_after else None
            )
            raise TransportError(
                f"alpaca rate-limited the request; backing off {backoff:.0f}s", endpoint=path
            )
        if response.status_code in (401, 403):
            # Never echo the body: an auth error can carry the submitted key.
            raise ProviderUnavailable(
                f"alpaca rejected the credentials with HTTP {response.status_code}. Check "
                f"{KEY_ID_VAR}/{SECRET_VAR}, and that the plan includes feed={self._feed}."
            )
        if not response.ok:
            raise TransportError(
                f"alpaca returned HTTP {response.status_code}: {response.text[:200]}",
                endpoint=path,
            )

        try:
            # `parse_float=Decimal` is not a nicety. Every price in this system
            # is a Decimal, and letting json build floats first would put a
            # binary round trip in front of the conversion — which changes the
            # stored value's last digits, and therefore every hash computed
            # over it.
            return json.loads(response.text, parse_float=Decimal), msg_id
        except ValueError as exc:
            raise DataError(f"alpaca returned a non-JSON body for {path}: {exc}") from exc


# --------------------------------------------------------------------------
# Response helpers
# --------------------------------------------------------------------------


def _rfc3339(moment: datetime) -> str:
    if moment.tzinfo is None:
        raise DataError("alpaca window bounds must be timezone-aware")
    return moment.astimezone(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")


def _parse_rfc3339(text: str) -> datetime:
    """Parse Alpaca's timestamps, which carry a `Z` and variable precision."""
    normalised = text.strip()
    if normalised.endswith("Z"):
        normalised = f"{normalised[:-1]}+00:00"
    try:
        parsed = datetime.fromisoformat(normalised)
    except ValueError as exc:
        raise DataError(f"alpaca sent an unparseable timestamp {text!r}") from exc
    if parsed.tzinfo is None:
        raise DataError(f"alpaca timestamp {text!r} carries no offset")
    return parsed.astimezone(UTC)


def _effective_date(entry: dict[str, Any]) -> str:
    """The date the action takes effect on the price.

    `ex_date` first: that is the session on which the price gaps, which is what
    the factor algebra keys on. `process_date` is Alpaca's bookkeeping date and
    can differ by a day.
    """
    for key in ("ex_date", "effective_date", "process_date"):
        value = entry.get(key)
        if value:
            return str(value)
    raise DataError(f"alpaca corporate action has no usable effective date: {sorted(entry)}")


def _split_action(entry: Any, *, instrument_uid: str, observed: datetime) -> RawAction | None:
    if not isinstance(entry, dict):
        return None
    new_rate = entry.get("new_rate")
    old_rate = entry.get("old_rate")
    if not new_rate or not old_rate:
        return None
    numerator, denominator = _integer_ratio(new_rate, old_rate)
    return RawAction(
        instrument_uid=instrument_uid,
        action_type="split",
        effective_date=_effective_date(entry),
        known_at_utc=observed,
        provider="alpaca",
        # A 4-for-1 arrives as new_rate=4, old_rate=1: one old share becomes
        # four, so the price divides by 4. `ratio_num/ratio_den` is that 4/1,
        # and `adjustments.py` inverts it — stated here because getting the
        # direction backwards squares the error instead of cancelling it.
        ratio_num=numerator,
        ratio_den=denominator,
        declared_date=_optional_date(entry.get("declaration_date")),
    )


def _dividend_action(entry: Any, *, instrument_uid: str, observed: datetime) -> RawAction | None:
    if not isinstance(entry, dict):
        return None
    rate = entry.get("rate")
    if rate is None:
        return None
    return RawAction(
        instrument_uid=instrument_uid,
        action_type="cash_dividend",
        effective_date=_effective_date(entry),
        known_at_utc=observed,
        provider="alpaca",
        gross_amount=parse_price(rate),
        currency="USD",
        declared_date=_optional_date(entry.get("declaration_date")),
    )


def _optional_date(value: Any) -> str | None:
    """A declaration date if there is one, and None if there is not.

    Never imputed. A split whose declaration date is guessed at is a factor
    table that can contain tomorrow's split, which defeats the entire point of
    keeping `known_at` separate from `effective_date`.
    """
    return str(value) if value else None


def _integer_ratio(new_rate: Any, old_rate: Any) -> tuple[int, int]:
    """Reduce a split ratio to exact integers, refusing to round.

    Alpaca sends these as numbers, and a reverse split can legitimately be
    fractional (`new_rate=0.1`). The ratio is scaled to integers exactly rather
    than rounded: `Fraction` over the Decimal keeps 1-for-10 as 1/10 instead of
    0.1, which matters because the factor product is a running multiplication
    over a twenty-year history.
    """
    try:
        ratio = Fraction(parse_price(new_rate)) / Fraction(parse_price(old_rate))
    except (ZeroDivisionError, ValueError) as exc:
        raise DataError(f"unusable split ratio {new_rate!r}/{old_rate!r}") from exc
    if ratio <= 0:
        raise DataError(f"non-positive split ratio {new_rate!r}/{old_rate!r}")
    return ratio.numerator, ratio.denominator


def alpaca_window_for(resolution: Resolution, *, days: int) -> tuple[datetime, datetime]:
    """A fetch window clamped to what the free plan will actually serve."""
    cap = _MAX_HISTORY_DAYS[resolution]
    end = now_utc()
    return end - timedelta(days=min(days, cap)), end
