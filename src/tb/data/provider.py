"""The market data port: the `Bar` type, and what a provider must offer.

Trading 212 serves no market data, so every price in this system comes from a
different vendor than the one that fills the orders. Two properties of the
`Bar` type below are what keep that from becoming a silent disaster.

**Three time axes, not one.** A bar has an *event* time (`bar_open_utc`), a
*knowledge* time (`available_at_utc`) and a *vintage* time (`ingested_at_utc`).
The knowledge time is the one almost every system gets wrong: the earliest
moment a bar can actually be acted on is `bar_close + provider_delay +
ingest_latency`, not bar close. Gating features on "bars before the decision
bar" with a fifteen-minute-delayed feed produces a systematic lookahead
*exactly the size of the delay* — and the edge this system is hunting lives in
the first seconds after a bar closes, so it is precisely the wrong thing to get
wrong. `available_at_utc` is therefore **stored**, never recomputed at read
time from whatever the provider's delay is thought to be today.

**Bar-open normalisation, asserted.** Providers disagree about whether a
minute bar's timestamp marks its open or its close. Getting that backwards
shifts every feature by one bar of future information, so the convention is
declared per provider, normalised at the boundary, and an ambiguous payload
**fails ingestion** rather than being guessed at.

`live_capable` is deliberately not a property a provider can simply claim. It
is computed from *observed* delay against the configured bound, so a delayed
feed cannot assert itself into the decision path.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from decimal import Decimal, InvalidOperation
from enum import StrEnum
from typing import Protocol, runtime_checkable

from tb.core.canonical import hash_payload
from tb.core.errors import TbError


class DataError(TbError):
    """Base for market-data failures."""


class AmbiguousTimestampError(DataError):
    """A payload whose bar timestamps cannot be placed on the open/close axis.

    Fails ingestion on purpose. A guess here is a one-bar lookahead applied
    uniformly to every feature, which is both invisible and profitable-looking.
    """


class PrecisionLossError(DataError):
    """A price carries more precision than the configured scale can store.

    Refused rather than rounded: silently truncating the last digit of a price
    changes returns, and the whole point of scaled integers is that the stored
    value is exactly what the venue printed.
    """


class ProviderUnavailable(DataError):
    """The provider could not be reached or is not configured."""


class Resolution(StrEnum):
    MINUTE = "minute"
    HOURLY = "hourly"
    DAILY = "daily"

    @property
    def duration(self) -> timedelta:
        return {
            Resolution.MINUTE: timedelta(minutes=1),
            Resolution.HOURLY: timedelta(hours=1),
            Resolution.DAILY: timedelta(days=1),
        }[self]

    @property
    def seconds(self) -> int:
        return int(self.duration.total_seconds())

    @property
    def is_intraday(self) -> bool:
        return self in (Resolution.MINUTE, Resolution.HOURLY)


class Session(StrEnum):
    """Which trading session a bar belongs to.

    Load-bearing for the cross-venue price check. Trading 212's overnight
    `currentPrice` is the last regular-hours close, while a provider's "latest
    bar" may be a pre- or post-market print. Earnings-night extended-hours
    moves of 200-500bps are routine, so comparing across sessions against a
    75bps threshold would block half the universe every evening.
    """

    REGULAR = "regular"
    EXTENDED = "extended"
    UNKNOWN = "unknown"


class Provenance(StrEnum):
    """Whether a bar's knowledge time was observed or assumed.

    `LIVE` means we saw the bar arrive, so `available_at_utc` is a measurement.
    `BACKFILL` means we asked for history: the values are the vendor's *current*
    view, and `available_at_utc` is set to bar close as the earliest defensible
    knowledge time. That is the right as-of semantics for a backtest, and it is
    still not point-in-time — the values may have been restated since, which is
    what a vintage's `pit_completeness_flag` records rather than hides.
    """

    LIVE = "live"
    BACKFILL = "backfill"


class TimestampConvention(StrEnum):
    BAR_OPEN = "bar_open"
    BAR_CLOSE = "bar_close"
    # Declared by a provider we have not yet pinned down. Ingestion refuses it.
    AMBIGUOUS = "ambiguous"


# --------------------------------------------------------------------------
# Scaled-integer prices
# --------------------------------------------------------------------------


def to_scaled(price: Decimal, scale: int) -> int:
    """Convert a price to an integer scaled by `10**scale`, exactly.

    Floats are not used anywhere in the bar store: a float round-trip through
    Parquet changes the value's bits, which changes every hash computed over it
    and quietly breaks the reproducibility the whole vintage mechanism exists
    to provide.
    """
    if not isinstance(price, Decimal):
        raise TypeError(f"prices must be Decimal, got {type(price).__name__}")
    if not price.is_finite():
        raise PrecisionLossError(f"non-finite price: {price!r}")
    shifted = price.scaleb(scale)
    if shifted != shifted.to_integral_value():
        raise PrecisionLossError(
            f"price {price} needs more than {scale} decimal places; storing it would "
            "silently drop precision. Raise data.price_scale or reject the payload."
        )
    return int(shifted)


def from_scaled(value: int, scale: int) -> Decimal:
    """Recover the exact Decimal a scaled integer represents."""
    return Decimal(value).scaleb(-scale)


def parse_price(raw: object) -> Decimal:
    """Parse a provider's number into a Decimal without going via float."""
    if isinstance(raw, Decimal):
        parsed = raw
    elif isinstance(raw, bool):
        raise DataError("boolean where a price was expected")
    elif isinstance(raw, (int, float, str)):
        text = str(raw).strip()
        if not text:
            raise DataError("empty string where a price was expected")
        try:
            parsed = Decimal(text)
        except InvalidOperation as exc:
            raise DataError(f"not a price: {raw!r}") from exc
    else:
        raise DataError(f"not a price: {raw!r}")
    if not parsed.is_finite():
        raise DataError(f"non-finite price: {raw!r}")
    return parsed


# --------------------------------------------------------------------------
# Instrument identity
# --------------------------------------------------------------------------


def make_instrument_uid(
    *, isin: str | None = None, t212_ticker: str | None = None, data_symbol: str | None = None
) -> str:
    """A stable identity for an instrument, preferring ISIN.

    Bars are keyed by this rather than by a ticker string. Tickers get reused:
    keying on one lets a delisted company's history splice onto a new issuer's
    under the same symbol, producing a continuous-looking price series that
    belongs to two different businesses. An ISIN does not move.

    The `t212:` and `sym:` fallbacks are marked as such so a later audit can
    find every instrument whose identity is only as stable as its ticker.
    """
    if isin:
        return f"isin:{isin.strip().upper()}"
    if t212_ticker:
        return f"t212:{t212_ticker.strip()}"
    if data_symbol:
        return f"sym:{data_symbol.strip().upper()}"
    raise ValueError("an instrument_uid needs at least one of isin, t212_ticker, data_symbol")


def uid_is_stable(instrument_uid: str) -> bool:
    """True only for ISIN-derived identities."""
    return instrument_uid.startswith("isin:")


# --------------------------------------------------------------------------
# The bar
# --------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class Bar:
    """One OHLCV bar, with its knowledge time.

    Validation in `__post_init__` is deliberately strict. Each check below
    corresponds to a real way a feed goes wrong, and every one of them is
    cheaper to catch here than to diagnose later from a backtest that looked
    plausible.
    """

    instrument_uid: str
    resolution: Resolution
    bar_open_utc: datetime
    available_at_utc: datetime
    ingested_at_utc: datetime
    provider: str
    provenance: Provenance
    session: Session
    open: Decimal
    high: Decimal
    low: Decimal
    close: Decimal
    volume: int | None = None
    currency: str | None = None

    def __post_init__(self) -> None:
        for name in ("bar_open_utc", "available_at_utc", "ingested_at_utc"):
            value: datetime = getattr(self, name)
            if value.tzinfo is None:
                raise DataError(f"{self.instrument_uid}: {name} is a naive datetime")

        if self.low > min(self.open, self.close) or self.high < max(self.open, self.close):
            # Catches transposed columns and bad merges for free, and it is
            # impossible for a genuine bar to violate it.
            raise DataError(
                f"{self.instrument_uid} @ {self.bar_open_utc.isoformat()}: OHLC out of order "
                f"(o={self.open} h={self.high} l={self.low} c={self.close})"
            )
        if self.low <= 0:
            raise DataError(
                f"{self.instrument_uid} @ {self.bar_open_utc.isoformat()}: non-positive price "
                f"(low={self.low})"
            )
        if self.volume is not None:
            if self.volume < 0:
                raise DataError(f"{self.instrument_uid}: negative volume {self.volume}")
            if self.volume == 0 and self.high != self.low:
                # A price range with no volume cannot happen. It means a
                # synthetic or forward-filled bar reached the raw store.
                raise DataError(
                    f"{self.instrument_uid} @ {self.bar_open_utc.isoformat()}: zero volume with "
                    f"a non-zero range (h={self.high} l={self.low}) — this is a synthetic or "
                    "filled bar, not something the venue printed"
                )

        # The single most important invariant in the data layer. A bar that
        # claims to have been knowable before it finished is a lookahead
        # channel, so it is a hard error rather than a warning.
        if self.available_at_utc < self.bar_close_utc:
            raise DataError(
                f"{self.instrument_uid} @ {self.bar_open_utc.isoformat()}: available_at "
                f"({self.available_at_utc.isoformat()}) precedes bar close "
                f"({self.bar_close_utc.isoformat()}). A bar cannot be knowable before it ends."
            )

    # -- derived ----------------------------------------------------------

    @property
    def bar_close_utc(self) -> datetime:
        return self.bar_open_utc + self.resolution.duration

    @property
    def delay_seconds(self) -> float:
        """How long after closing the bar became knowable."""
        return (self.available_at_utc - self.bar_close_utc).total_seconds()

    def is_visible_at(self, as_of: datetime) -> bool:
        """Whether this bar was knowable at `as_of`.

        The only admissible visibility test. Comparing against `bar_open_utc`
        or `bar_close_utc` instead is the lookahead described in the module
        docstring.
        """
        if as_of.tzinfo is None:
            raise DataError("as_of must be timezone-aware")
        return self.available_at_utc <= as_of

    def age_seconds_at(self, now: datetime) -> float:
        """Seconds since this bar became knowable — what staleness is measured on."""
        if now.tzinfo is None:
            raise DataError("now must be timezone-aware")
        return (now - self.available_at_utc).total_seconds()

    @property
    def typical_price(self) -> Decimal:
        return (self.high + self.low + self.close) / Decimal(3)

    def row_values(self, *, scale: int) -> dict[str, object]:
        """The canonical row, for hashing and for storage.

        Hashes are computed over this dict rather than over a DataFrame: a
        pandas version bump changes NaN handling and dtype promotion, which
        would silently change every hash and make sealed vintages unverifiable.
        """
        return {
            "instrument_uid": self.instrument_uid,
            "resolution": self.resolution.value,
            "bar_open_utc": self.bar_open_utc.astimezone(UTC).isoformat(),
            "available_at_utc": self.available_at_utc.astimezone(UTC).isoformat(),
            "provider": self.provider,
            "provenance": self.provenance.value,
            "session": self.session.value,
            "open_scaled": to_scaled(self.open, scale),
            "high_scaled": to_scaled(self.high, scale),
            "low_scaled": to_scaled(self.low, scale),
            "close_scaled": to_scaled(self.close, scale),
            "volume": self.volume,
            "price_scale": scale,
            "currency": self.currency,
        }

    def row_hash(self, *, scale: int) -> str:
        """Content hash of the bar's values.

        Deliberately excludes `ingested_at_utc`: two fetches of the same
        unchanged bar must hash identically, or every poll would look like a
        revision and `bar_revisions` would fill with noise until the real
        signal was undetectable.
        """
        return hash_payload(self.row_values(scale=scale))

    def storage_row(self, *, scale: int) -> dict[str, object]:
        """The full row as stored, including vintage time.

        Distinct from `row_values` on purpose: the hash input must exclude
        `ingested_at_utc` so two fetches of an unchanged bar hash identically,
        but the *stored* row needs it, because that is the axis an as-of query
        collapses revisions along.
        """
        return {
            **self.row_values(scale=scale),
            "ingested_at_utc": self.ingested_at_utc.astimezone(UTC).isoformat(),
        }

    @property
    def identity(self) -> tuple[str, str, str, str]:
        """What makes two bars the same bar, ignoring vintage."""
        return (
            self.instrument_uid,
            self.resolution.value,
            self.bar_open_utc.astimezone(UTC).isoformat(),
            self.provider,
        )


# --------------------------------------------------------------------------
# Normalisation
# --------------------------------------------------------------------------


def normalise_bar_open(
    timestamp: datetime, *, convention: TimestampConvention, resolution: Resolution
) -> datetime:
    """Put a provider's timestamp onto the bar-open axis.

    An `AMBIGUOUS` convention raises. That is the point: a provider whose
    stamping we have not established cannot be ingested, because the failure
    mode of guessing is a uniform one-bar lookahead rather than an error.
    """
    if timestamp.tzinfo is None:
        raise DataError("provider timestamps must be timezone-aware")
    aware = timestamp.astimezone(UTC)

    if convention is TimestampConvention.BAR_OPEN:
        return aware
    if convention is TimestampConvention.BAR_CLOSE:
        return aware - resolution.duration
    raise AmbiguousTimestampError(
        f"provider declares an ambiguous timestamp convention for {resolution.value} bars. "
        "Refusing to ingest: treating a close-stamped bar as open-stamped shifts every "
        "feature by one bar of future information, which looks like alpha."
    )


def knowledge_time(
    *,
    bar_open: datetime,
    resolution: Resolution,
    provenance: Provenance,
    delay_seconds: float,
    ingested_at: datetime | None = None,
) -> datetime:
    """The earliest moment a bar could have been acted on.

    For `LIVE` bars this is `bar_close + provider_delay`, floored at the actual
    ingest time when that is later — we cannot have known something before it
    arrived. For `BACKFILL` it is bar close exactly: the earliest defensible
    knowledge time, with the caveat about restated values recorded on the
    vintage rather than buried here.
    """
    bar_close = bar_open + resolution.duration
    if provenance is Provenance.BACKFILL:
        return bar_close
    available = bar_close + timedelta(seconds=max(0.0, delay_seconds))
    if ingested_at is not None and ingested_at > available:
        return ingested_at
    return available


# --------------------------------------------------------------------------
# Provider capabilities and the protocol
# --------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class ProviderCapabilities:
    """What a provider can do, and how far behind it is.

    `declared_delay_seconds` is a conservative default, not a promise. The real
    value is measured and written to `provider_observations`, and
    `live_capable` reads the measurement where one exists.
    """

    name: str
    resolutions: frozenset[Resolution]
    timestamp_convention: TimestampConvention
    declared_delay_seconds: dict[Resolution, float] = field(default_factory=dict)
    max_history_days: dict[Resolution, int] = field(default_factory=dict)
    supports_extended_hours: bool = False
    supports_corporate_actions: bool = False
    # IEX is a few percent of consolidated volume, so its minute close is not
    # the consolidated last price. Recorded because it bounds what any strategy
    # built on this feed can honestly claim.
    consolidated_tape: bool = False
    note: str = ""

    def delay_for(self, resolution: Resolution) -> float:
        return self.declared_delay_seconds.get(resolution, float("inf"))

    def live_capable(
        self,
        resolution: Resolution,
        *,
        max_delay_seconds: float,
        observed_delay: float | None = None,
    ) -> tuple[bool, str]:
        """Whether this provider may drive a live decision at this resolution.

        Returns the reason as well as the verdict, because "the bot is not
        trading" needs to be answerable without reading code. The observed
        delay wins over the declared one when available: the declared value is
        a guess and the measurement is not.
        """
        if resolution not in self.resolutions:
            return False, f"{self.name} does not serve {resolution.value} bars"
        delay = observed_delay if observed_delay is not None else self.delay_for(resolution)
        if delay == float("inf"):
            return False, (
                f"{self.name} has no known delay for {resolution.value} bars, so it is "
                "assumed unusable for live decisions until measured"
            )
        if delay > max_delay_seconds:
            return False, (
                f"{self.name} {resolution.value} bars arrive about {delay:.0f}s after close, "
                f"beyond the {max_delay_seconds:.0f}s bound. Usable for history, not for a "
                "live decision."
            )
        return True, f"{self.name} {resolution.value} delay {delay:.0f}s is within bound"


@dataclass(frozen=True, slots=True)
class BarBatch:
    """Bars as one provider returned them, with what went wrong alongside."""

    bars: tuple[Bar, ...]
    provider: str
    symbol: str
    resolution: Resolution
    requested_start: datetime | None = None
    requested_end: datetime | None = None
    raw_msg_id: str | None = None
    warnings: tuple[str, ...] = field(default_factory=tuple)

    def __len__(self) -> int:
        return len(self.bars)

    @property
    def first_open(self) -> datetime | None:
        return self.bars[0].bar_open_utc if self.bars else None

    @property
    def last_open(self) -> datetime | None:
        return self.bars[-1].bar_open_utc if self.bars else None

    def sorted_bars(self) -> tuple[Bar, ...]:
        return tuple(sorted(self.bars, key=lambda b: b.bar_open_utc))


@dataclass(frozen=True, slots=True)
class RawAction:
    """A corporate action as a provider reported it.

    `known_at_utc` is when *we* observed it, not when it was declared. Free
    feeds report actions at roughly the ex-date with no declaration date, so
    `declared_date` is usually None — and that None is stored rather than
    imputed, because pretending to know when a split became public is how a
    factor table ends up containing tomorrow's split.
    """

    instrument_uid: str
    action_type: str
    effective_date: str
    known_at_utc: datetime
    provider: str
    ratio_num: int | None = None
    ratio_den: int | None = None
    gross_amount: Decimal | None = None
    currency: str | None = None
    new_symbol: str | None = None
    declared_date: str | None = None


@runtime_checkable
class MarketDataProvider(Protocol):
    """What the data layer needs from a price vendor."""

    @property
    def name(self) -> str: ...

    @property
    def capabilities(self) -> ProviderCapabilities: ...

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
    ) -> BarBatch: ...

    def latest_bar(
        self, symbol: str, *, instrument_uid: str, resolution: Resolution
    ) -> Bar | None: ...

    def fetch_actions(
        self, symbol: str, *, instrument_uid: str, start: datetime, end: datetime
    ) -> tuple[RawAction, ...]: ...

    def close(self) -> None: ...


def dedupe_latest(bars: Iterable[Bar]) -> tuple[Bar, ...]:
    """Collapse to one bar per identity, keeping the most recently ingested.

    This is the **read-side** collapse, applied only after visibility filtering
    — `visible_bars` calls it once it has already discarded anything ingested
    after the as-of instant, so "latest" means latest *as of then*.

    It must never be used on the write path. Collapsing vintages before storage
    discards the pre-revision value, which is precisely the value an as-of query
    over an earlier instant needs to return; see `dedupe_vintages`.
    """
    latest: dict[tuple[str, str, str, str], Bar] = {}
    for bar in bars:
        existing = latest.get(bar.identity)
        if existing is None or bar.ingested_at_utc >= existing.ingested_at_utc:
            latest[bar.identity] = bar
    return tuple(sorted(latest.values(), key=lambda b: b.bar_open_utc))


def dedupe_vintages(bars: Iterable[Bar]) -> tuple[Bar, ...]:
    """Drop exact duplicates while keeping every distinct vintage.

    The **write-side** dedupe. Two fetches of the same unchanged bar collapse to
    one row, but a restatement and its original are both kept, keyed by
    `ingested_at_utc`. That retention is what makes an as-of query able to
    answer "what did this bar say at the time" rather than only "what does the
    vendor say now".
    """
    seen: dict[tuple[str, str, str, str, str], Bar] = {}
    for bar in bars:
        key = (*bar.identity, bar.ingested_at_utc.astimezone(UTC).isoformat())
        seen.setdefault(key, bar)
    return tuple(sorted(seen.values(), key=lambda b: (b.bar_open_utc, b.ingested_at_utc)))
