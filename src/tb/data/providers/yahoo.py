"""Yahoo Finance chart endpoint, called directly.

Deliberately not the `yfinance` package. Going through our own `Transport`
means the entire provider is deterministic under `RecordingTransport` — no
network in the test suite, no global session state, our own pacing and our own
raw archive. That matters more here than anywhere else in the system, because
of what this provider is:

**A reverse-engineered API whose failure mode is silently different data.** It
is unofficial, undocumented, and free. It does not return an error when it
changes; it returns plausible numbers. Which is why the cross-provider check
and the raw archive are load-bearing rather than nice-to-have, and why the
integrity audit exists at all.

Two honest limitations, both encoded in `capabilities` rather than buried:

* **Yahoo back-adjusts historical OHLC for splits and offers no raw option.**
  So `returns_raw_prices=False`: what gets stored is the vendor's *current*
  adjusted view, not what the venue printed. A "raw price store" is simply not
  achievable from this feed. The consequence is that a split silently rewrites
  history here, which is exactly what revision detection is watching for.
* **Intraday history is shallow** — roughly a month of 1-minute bars — and
  **delayed**, so this provider can supply history but can never drive a live
  decision. `declared_delay_seconds` says so, and `live_capable` turns that
  into a refusal rather than a warning.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime
from decimal import Decimal
from typing import Any
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

from tb.core.clock import now_utc
from tb.core.errors import TransportError
from tb.core.http import HttpxTransport, Transport
from tb.data.pacing import YAHOO_PACING, PacingSpec, ProviderPacer
from tb.data.provider import (
    Bar,
    BarBatch,
    DataError,
    Provenance,
    ProviderCapabilities,
    RawAction,
    Resolution,
    Session,
    TimestampConvention,
    knowledge_time,
    normalise_bar_open,
    parse_price,
)

BASE_URL = "https://query1.finance.yahoo.com/v8/finance/chart"

_INTERVAL = {
    Resolution.MINUTE: "1m",
    Resolution.HOURLY: "1h",
    Resolution.DAILY: "1d",
}

# Yahoo's free intraday feed is delayed. The figure is a conservative default,
# replaced by the measured value in `provider_observations` once the probe has
# run — a declared delay is a guess and a measured one is not.
_DECLARED_DELAY = {
    Resolution.MINUTE: 900.0,
    Resolution.HOURLY: 900.0,
    # Daily bars are only meaningful after the session closes, so "delay" for a
    # daily bar is not a latency problem: a once-a-day decision does not care
    # whether the close arrived at 21:00 or 21:15.
    Resolution.DAILY: 60.0,
}

_MAX_HISTORY_DAYS = {
    Resolution.MINUTE: 30,
    Resolution.HOURLY: 730,
    Resolution.DAILY: 36500,
}


class YahooProvider:
    """History from Yahoo's chart endpoint. Not usable for live decisions."""

    name = "yahoo"

    def __init__(
        self,
        *,
        transport: Transport | None = None,
        pacing: PacingSpec | None = None,
        pacer: ProviderPacer | None = None,
        timeout: float = 30.0,
        archive: Any = None,
    ) -> None:
        self._transport = (
            transport
            if transport is not None
            # Yahoo serves different content to something that looks automated,
            # so the browser UA is a functional requirement, not a disguise for
            # anything: the request volume is paced well below any published
            # threshold and the endpoint is publicly readable.
            else HttpxTransport(user_agent=HttpxTransport.BROWSER_USER_AGENT)
        )
        self._pacer = pacer or ProviderPacer(spec=pacing or YAHOO_PACING)
        self._timeout = timeout
        self._archive = archive

    @property
    def capabilities(self) -> ProviderCapabilities:
        return ProviderCapabilities(
            name=self.name,
            resolutions=frozenset(_INTERVAL),
            timestamp_convention=TimestampConvention.BAR_OPEN,
            declared_delay_seconds=dict(_DECLARED_DELAY),
            max_history_days=dict(_MAX_HISTORY_DAYS),
            supports_extended_hours=True,
            supports_corporate_actions=True,
            # Yahoo's daily closes come from the consolidated tape, which is the
            # one thing it does better than a free IEX-only feed.
            consolidated_tape=True,
            # The important caveat: historical OHLC is split-adjusted by the
            # vendor, with no raw option.
            returns_raw_prices=False,
            note=(
                "unofficial endpoint; back-adjusts historical OHLC for splits, so stored "
                "prices are the vendor's current view rather than what the venue printed. "
                "Intraday history is ~30 days and delayed, so history only."
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
        payload, msg_id = self._request(
            symbol,
            params={
                "interval": _INTERVAL[resolution],
                "period1": int(start.timestamp()),
                "period2": int(end.timestamp()),
                "includePrePost": "true" if include_extended else "false",
                "events": "div,splits",
            },
        )
        result, warnings = _unwrap(payload, symbol)
        if result is None:
            return BarBatch(
                bars=(),
                provider=self.name,
                symbol=symbol,
                resolution=resolution,
                requested_start=start,
                requested_end=end,
                raw_msg_id=msg_id,
                warnings=tuple(warnings),
            )

        bars, more_warnings = self._parse_bars(
            result,
            symbol=symbol,
            instrument_uid=instrument_uid,
            resolution=resolution,
            provenance=provenance,
            include_extended=include_extended,
        )
        return BarBatch(
            bars=bars,
            provider=self.name,
            symbol=symbol,
            resolution=resolution,
            requested_start=start,
            requested_end=end,
            raw_msg_id=msg_id,
            warnings=tuple(warnings + more_warnings),
        )

    def latest_bar(self, symbol: str, *, instrument_uid: str, resolution: Resolution) -> Bar | None:
        """The most recent finished bar, or None.

        The chart endpoint includes the bar currently being formed, whose high,
        low and volume are partial. Unsettled bars are dropped: storing a
        partial bar and replacing it when the period ends would register as a
        revision on every poll, burying the revisions that actually matter.
        """
        now = now_utc()
        lookback = resolution.duration * 40
        batch = self.fetch_bars(
            symbol,
            instrument_uid=instrument_uid,
            resolution=resolution,
            start=now - lookback,
            end=now,
            provenance=Provenance.LIVE,
        )
        settled = [bar for bar in batch.sorted_bars() if bar.is_settled_at(now)]
        return settled[-1] if settled else None

    def _parse_bars(
        self,
        result: dict[str, Any],
        *,
        symbol: str,
        instrument_uid: str,
        resolution: Resolution,
        provenance: Provenance,
        include_extended: bool,
    ) -> tuple[tuple[Bar, ...], list[str]]:
        warnings: list[str] = []
        timestamps = result.get("timestamp") or []
        indicators = result.get("indicators") or {}
        quotes = (indicators.get("quote") or [{}])[0]
        meta = result.get("meta") or {}
        currency = meta.get("currency")

        opens = quotes.get("open") or []
        highs = quotes.get("high") or []
        lows = quotes.get("low") or []
        closes = quotes.get("close") or []
        volumes = quotes.get("volume") or []

        if not (len(opens) == len(highs) == len(lows) == len(closes) == len(timestamps)):
            raise DataError(
                f"yahoo returned mismatched array lengths for {symbol}: "
                f"{len(timestamps)} timestamps against {len(closes)} closes. Refusing to "
                "align them by position — a one-element offset would shift every bar."
            )

        regular_window = _regular_window(meta)
        session_tz = _exchange_tz(meta)
        volume_is_meaningful = symbol_has_volume(symbol)
        ingested = now_utc()
        delay = _DECLARED_DELAY[resolution]
        bars: list[Bar] = []
        skipped_null = 0

        for index, epoch in enumerate(timestamps):
            raw = (opens[index], highs[index], lows[index], closes[index])
            if any(value is None for value in raw):
                # Yahoo pads its arrays with nulls for minutes that had no
                # print. Skipped rather than forward-filled: an invented bar in
                # the raw store is indistinguishable from a real one later, and
                # the audit's job is to report the gap, not to paper over it.
                skipped_null += 1
                continue

            stamped = datetime.fromtimestamp(int(epoch), tz=UTC)
            bar_open = normalise_bar_open(
                stamped,
                convention=TimestampConvention.BAR_OPEN,
                resolution=resolution,
                # Yahoo stamps a daily bar at the market open in exchange time,
                # so the session date has to be read in that zone: a 09:30 ET
                # stamp is the same UTC date, but a 09:00 Sydney stamp is the
                # UTC day before.
                session_tz=session_tz,
            )
            session = _classify_session(bar_open, regular_window, resolution)
            if session is Session.EXTENDED and not include_extended:
                continue

            volume = volumes[index] if index < len(volumes) else None
            if not volume_is_meaningful:
                # Spot FX and index symbols have no share count, and Yahoo sends
                # a literal 0 for them. Stored as 0 that is a *measurement* of
                # no trading, which collides with `Bar`'s synthetic-bar guard
                # and would drop every FX bar — silently producing an empty rate
                # table. None is the honest value: not reported.
                volume = None
            try:
                bar = Bar(
                    instrument_uid=instrument_uid,
                    resolution=resolution,
                    bar_open_utc=bar_open,
                    available_at_utc=knowledge_time(
                        bar_open=bar_open,
                        resolution=resolution,
                        provenance=provenance,
                        delay_seconds=delay,
                        ingested_at=ingested if provenance is Provenance.LIVE else None,
                    ),
                    ingested_at_utc=ingested,
                    provider=self.name,
                    provenance=provenance,
                    session=session,
                    open=parse_price(raw[0]),
                    high=parse_price(raw[1]),
                    low=parse_price(raw[2]),
                    close=parse_price(raw[3]),
                    volume=None if volume is None else int(volume),
                    currency=currency,
                )
            except DataError as exc:
                # A bar that fails validation is reported and dropped, not
                # coerced. One bad print must not take down a whole backfill,
                # but it must not enter the store either.
                warnings.append(f"dropped bar at {bar_open.isoformat()}: {exc}")
                continue
            bars.append(bar)

        if skipped_null:
            warnings.append(f"{skipped_null} null-padded bar(s) skipped rather than forward-filled")
        return tuple(bars), warnings

    # -- corporate actions -------------------------------------------------

    def fetch_actions(
        self, symbol: str, *, instrument_uid: str, start: datetime, end: datetime
    ) -> tuple[RawAction, ...]:
        """Splits and dividends, as reported at roughly the ex-date.

        Yahoo gives no declaration date, so `declared_date` is None — stored as
        None rather than imputed. Pretending to know when a split became public
        is how a factor table ends up containing tomorrow's split.
        """
        payload, _ = self._request(
            symbol,
            params={
                "interval": "1d",
                "period1": int(start.timestamp()),
                "period2": int(end.timestamp()),
                "events": "div,splits",
            },
        )
        result, _ = _unwrap(payload, symbol)
        if result is None:
            return ()

        events = result.get("events") or {}
        observed = now_utc()
        actions: list[RawAction] = []

        for entry in (events.get("splits") or {}).values():
            numerator = entry.get("numerator")
            denominator = entry.get("denominator")
            if not numerator or not denominator:
                continue
            actions.append(
                RawAction(
                    instrument_uid=instrument_uid,
                    action_type="split",
                    effective_date=_date_of(entry.get("date")),
                    known_at_utc=observed,
                    provider=self.name,
                    ratio_num=int(numerator),
                    ratio_den=int(denominator),
                )
            )

        for entry in (events.get("dividends") or {}).values():
            amount = entry.get("amount")
            if amount is None:
                continue
            actions.append(
                RawAction(
                    instrument_uid=instrument_uid,
                    action_type="cash_dividend",
                    effective_date=_date_of(entry.get("date")),
                    known_at_utc=observed,
                    provider=self.name,
                    gross_amount=Decimal(str(amount)),
                    currency=(result.get("meta") or {}).get("currency"),
                )
            )
        return tuple(sorted(actions, key=lambda a: (a.effective_date, a.action_type)))

    # -- transport ---------------------------------------------------------

    def _request(self, symbol: str, *, params: dict[str, Any]) -> tuple[Any, str | None]:
        self._pacer.acquire()
        url = f"{BASE_URL}/{symbol}"
        response = self._transport.request(
            "GET",
            url,
            headers={"Accept": "application/json"},
            params=params,
            timeout=self._timeout,
        )

        msg_id: str | None = None
        if self._archive is not None:
            archived = self._archive.record(
                endpoint="yahoo_chart",
                method="GET",
                url_path=f"/v8/finance/chart/{symbol}",
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
                f"yahoo throttled the request; backing off {backoff:.0f}s. The endpoint is "
                "unofficial and publishes no limit, so the pacing budget is a guess.",
                endpoint=url,
            )
        if not response.ok:
            raise TransportError(
                f"yahoo returned HTTP {response.status_code}: {response.text[:200]}",
                endpoint=url,
            )

        try:
            # `parse_float=Decimal` is not a nicety. Every price here is a
            # Decimal, and letting json build a float first puts a binary round
            # trip in front of the conversion — changing the stored value's last
            # digits, and therefore every hash computed over it.
            return json.loads(response.text, parse_float=Decimal), msg_id
        except ValueError as exc:
            raise DataError(
                f"yahoo returned a non-JSON body for {symbol} ({exc}). For a "
                "reverse-engineered endpoint this usually means a block page rather "
                "than an outage."
            ) from exc


# --------------------------------------------------------------------------
# Response helpers
# --------------------------------------------------------------------------


def _unwrap(payload: Any, symbol: str) -> tuple[dict[str, Any] | None, list[str]]:
    """Pull the single result out of the chart envelope."""
    if not isinstance(payload, dict):
        raise DataError(f"yahoo response for {symbol} is not an object")
    chart = payload.get("chart")
    if not isinstance(chart, dict):
        raise DataError(f"yahoo response for {symbol} has no chart object")
    if chart.get("error"):
        raise DataError(f"yahoo reported an error for {symbol}: {chart['error']}")
    results = chart.get("result")
    if not results:
        return None, [f"yahoo returned no result for {symbol}"]
    first = results[0]
    if not isinstance(first, dict):
        raise DataError(f"yahoo result for {symbol} is not an object")
    return first, []


def symbol_has_volume(symbol: str) -> bool:
    """Whether a share count means anything for this Yahoo symbol.

    Vendor symbology, which is exactly the kind of knowledge an adapter should
    hold: Yahoo suffixes spot FX with `=X` (`GBPUSD=X`) and prefixes indices
    with `^` (`^GSPC`). Neither has a share count, and Yahoo reports a literal
    `0` rather than omitting the field.

    That matters because 0 and "not reported" are different claims. Stored as
    0, an FX bar with a real price range trips `Bar`'s synthetic-bar guard — the
    check that catches forward-filled equity bars — and every FX bar is dropped,
    leaving an empty rate table and a sizing path that can never convert.
    """
    return not (symbol.endswith("=X") or symbol.startswith("^"))


def _exchange_tz(meta: dict[str, Any]) -> ZoneInfo | None:
    """The listing venue's timezone, if Yahoo names one we recognise.

    Used only to decide which calendar date a daily bar belongs to. None falls
    back to UTC, which is correct for every market whose session does not cross
    local midnight — so an unknown zone name degrades rather than fails.
    """
    name = meta.get("exchangeTimezoneName")
    if not isinstance(name, str) or not name:
        return None
    try:
        return ZoneInfo(name)
    except (ZoneInfoNotFoundError, ValueError):
        return None


def _regular_window(meta: dict[str, Any]) -> tuple[datetime, datetime] | None:
    """The regular-session bounds Yahoo reports, if it reports them.

    Used to tell a regular-hours bar from a pre/post-market print. Without it
    every bar is `UNKNOWN` session, which the cross-venue check then refuses to
    compare — the conservative outcome rather than a wrong comparison.
    """
    period = (meta.get("currentTradingPeriod") or {}).get("regular")
    if not isinstance(period, dict):
        return None
    start, end = period.get("start"), period.get("end")
    if start is None or end is None:
        return None
    return (
        datetime.fromtimestamp(int(start), tz=UTC),
        datetime.fromtimestamp(int(end), tz=UTC),
    )


def _classify_session(
    bar_open: datetime, window: tuple[datetime, datetime] | None, resolution: Resolution
) -> Session:
    if resolution is Resolution.DAILY:
        # A daily bar spans the session by definition.
        return Session.REGULAR
    if window is None:
        return Session.UNKNOWN
    start, end = window
    # The reported window belongs to one specific day, so compare clock time
    # rather than the absolute instant; otherwise every historical bar falls
    # outside it and gets classified as extended.
    minutes = bar_open.hour * 60 + bar_open.minute
    open_minutes = start.hour * 60 + start.minute
    close_minutes = end.hour * 60 + end.minute
    if open_minutes <= minutes < close_minutes:
        return Session.REGULAR
    return Session.EXTENDED


def _date_of(epoch: Any) -> str:
    if epoch is None:
        raise DataError("corporate action with no date")
    return datetime.fromtimestamp(int(epoch), tz=UTC).date().isoformat()
