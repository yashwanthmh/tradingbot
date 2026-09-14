"""Point-in-time foreign exchange.

The limits are GBP. The universe is USD. Nothing in between is optional:

* `capital.absolute_ceiling_ccy: 500` is GBP, so sizing a USD position needs a
  rate, and a stale rate sizes it wrong in a direction nobody notices until the
  ceiling is breached.
* Trading 212 charges **0.15% per conversion**, so a US round trip from a GBP
  account pays it twice. The cost gate cannot evaluate
  `expected_cost_bps / expected_edge_bps` without knowing that the position
  crosses a currency boundary at all.
* P&L in the account currency is a different number from P&L in the
  instrument's currency, and the difference is not noise — it is a second
  return series layered on the first. A strategy that looks profitable in USD
  can be flat in GBP.

Same three-axis discipline as bars, for the same reason. A rate has an
observation date, a knowledge time, and a vintage; published rates get revised,
and a backtest that uses today's view of a 2019 rate has quietly imported
information from the future. `rate_at` filters on knowledge time and nothing
else.

**Missing is `UNKNOWN`, never 1.0.** The seductive bug here is defaulting a
missing rate to unity: it makes the code run, and it makes a USD position look
like a GBP position of the same magnitude, understating the ceiling by ~25%.
`convert` returns `UNKNOWN` and the sizing path has to handle it.
"""

from __future__ import annotations

import sqlite3
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from datetime import UTC, date, datetime, timedelta
from decimal import ROUND_HALF_EVEN, Decimal, localcontext
from typing import TypeAlias

from tb.core.canonical import hash_payload
from tb.core.clock import from_iso, to_iso
from tb.data.asof import UNKNOWN, Unknown
from tb.data.provider import DataError, Provenance, parse_price
from tb.ledger.store import Ledger

# A rate older than this is not a rate. Two trading days of slack covers a
# weekend; beyond that the world has moved and sizing against it is guesswork.
MAX_RATE_AGE_DAYS = 4

# Trading 212's FX fee, as a fraction. Not a hard limit because it is a *fact*
# about the venue rather than a policy choice — but it belongs here, beside the
# rate, because the cost model needs both together and a conversion charged
# once in the model and twice in reality is a systematically optimistic
# backtest.
FX_FEE_RATE = Decimal("0.0015")

_WORKING_PRECISION = 40


class FxError(DataError):
    """A rate could not be read, stored, or applied."""


def pair_key(base: str, quote: str) -> str:
    """A canonical pair name, so `GBPUSD` and `gbp/usd` are the same row."""
    base_code, quote_code = base.strip().upper(), quote.strip().upper()
    if len(base_code) != 3 or len(quote_code) != 3:
        raise FxError(f"currency codes must be three letters, got {base!r}/{quote!r}")
    if base_code == quote_code:
        raise FxError(f"{base_code} to {quote_code} is not a conversion")
    return f"{base_code}{quote_code}"


@dataclass(frozen=True, slots=True)
class FxRate:
    """One observation of one pair.

    `rate` is units of `quote` per one unit of `base`: `GBPUSD = 1.27` means one
    pound buys 1.27 dollars. Stating the direction here is not pedantry — an
    inverted rate is a 60% sizing error that still looks like a plausible
    number, which is the worst kind.
    """

    pair: str
    as_of_date: date
    available_at_utc: datetime
    ingested_at_utc: datetime
    rate: Decimal
    provider: str
    provenance: Provenance

    def __post_init__(self) -> None:
        for name in ("available_at_utc", "ingested_at_utc"):
            moment: datetime = getattr(self, name)
            if moment.tzinfo is None:
                raise FxError(f"{self.pair}: {name} is a naive datetime")
        if self.rate <= 0:
            raise FxError(f"{self.pair}: non-positive rate {self.rate}")
        if len(self.pair) != 6:
            raise FxError(f"{self.pair}: not a canonical pair key")

    @property
    def base(self) -> str:
        return self.pair[:3]

    @property
    def quote(self) -> str:
        return self.pair[3:]

    @property
    def inverse(self) -> FxRate:
        """The same observation, stated the other way round.

        Computed at high precision and left as a Decimal rather than rounded to
        a display scale: the reciprocal of 1.27 does not terminate, and rounding
        it before use would make `convert(convert(x))` drift away from `x`.
        """
        with localcontext() as ctx:
            ctx.prec = _WORKING_PRECISION
            flipped = Decimal(1) / self.rate
        return FxRate(
            pair=f"{self.quote}{self.base}",
            as_of_date=self.as_of_date,
            available_at_utc=self.available_at_utc,
            ingested_at_utc=self.ingested_at_utc,
            rate=flipped,
            provider=self.provider,
            provenance=self.provenance,
        )

    def is_visible_at(self, as_of: datetime) -> bool:
        if as_of.tzinfo is None:
            raise FxError("as_of must be timezone-aware")
        return self.available_at_utc <= as_of and self.ingested_at_utc <= as_of

    def age_days_at(self, moment: date) -> int:
        return (moment - self.as_of_date).days


@dataclass(frozen=True, slots=True)
class Conversion:
    """A converted amount, with the rate that produced it.

    The rate rides along because the cost model needs it and because a
    conversion whose rate cannot be shown is a number nobody can check months
    later. `fee` is separate from `amount` so the fee never silently disappears
    into the converted figure.
    """

    amount: Decimal
    fee: Decimal
    rate: Decimal
    pair: str
    as_of_date: date
    stale_days: int

    @property
    def net(self) -> Decimal:
        """What actually lands, after the conversion fee."""
        return self.amount - self.fee


# A conversion, or the refusal. Spelled as an alias so every caller's signature
# says out loud that "no rate" is a possible answer.
MaybeConversion: TypeAlias = "Conversion | Unknown"


class FxStore:
    """Reads and writes `fx_rates`, with as-of semantics.

    No event per rate. A daily rate per pair is routine bookkeeping, and one
    event each would bury the ledger under thousands of entries that nothing
    would ever read — the same reasoning that keeps routine broker responses in
    the archive rather than the chain. A *revision* to a rate already used for
    sizing is a different matter, and `record` reports those to its caller.
    """

    def __init__(self, ledger: Ledger, *, run_id: str | None = None) -> None:
        self._ledger = ledger
        self._run_id = run_id

    # -- writing -----------------------------------------------------------

    def record(self, rates: Iterable[FxRate]) -> tuple[int, tuple[str, ...]]:
        """Store rates. Returns `(written, revisions)`.

        A revision is an observation whose value differs from one already held
        for the same pair and date. Reported rather than swallowed: a restated
        rate changes what a past sizing decision *would* have been, and that is
        the kind of thing the reconciler needs to know about when a position's
        GBP cost basis stops matching.
        """
        pending = list(rates)
        if not pending:
            return 0, ()

        revisions: list[str] = []
        written = 0
        with self._ledger.transaction() as tx:
            for rate in pending:
                previous = tx.execute(
                    "SELECT rate FROM fx_rates WHERE pair = ? AND as_of_date = ? "
                    "ORDER BY ingested_at_utc DESC LIMIT 1",
                    (rate.pair, rate.as_of_date.isoformat()),
                ).fetchone()
                if previous is not None and Decimal(str(previous["rate"])) != rate.rate:
                    revisions.append(
                        f"{rate.pair} {rate.as_of_date}: {previous['rate']} -> {rate.rate}"
                    )
                tx.execute(
                    "INSERT OR IGNORE INTO fx_rates (pair, as_of_date, available_at_utc, "
                    "ingested_at_utc, rate, provider, provenance) VALUES (?, ?, ?, ?, ?, ?, ?)",
                    (
                        rate.pair,
                        rate.as_of_date.isoformat(),
                        to_iso(rate.available_at_utc),
                        to_iso(rate.ingested_at_utc),
                        # Text, not REAL. A rate read back as a float changes the
                        # last digits of every converted amount, and the ceiling
                        # check is an exact comparison.
                        str(rate.rate),
                        rate.provider,
                        rate.provenance.value,
                    ),
                )
                written += 1
        return written, tuple(revisions)

    # -- reading -----------------------------------------------------------

    def rate_at(
        self,
        base: str,
        quote: str,
        *,
        as_of: datetime,
        on_or_before: date | None = None,
        max_age_days: int = MAX_RATE_AGE_DAYS,
    ) -> FxRate | None:
        """The freshest rate knowable at `as_of`, or None.

        Tries the pair, then its inverse. Storing only one direction and
        deriving the other keeps a pair from drifting against itself — two
        independently stored directions eventually disagree, and then position
        sizing and P&L use different rates for the same conversion.
        """
        if as_of.tzinfo is None:
            raise FxError("as_of must be timezone-aware")
        target = on_or_before or as_of.astimezone(UTC).date()

        direct = self._newest(pair_key(base, quote), as_of=as_of, target=target)
        if direct is not None:
            return None if direct.age_days_at(target) > max_age_days else direct

        flipped = self._newest(pair_key(quote, base), as_of=as_of, target=target)
        if flipped is not None and flipped.age_days_at(target) <= max_age_days:
            return flipped.inverse
        return None

    def _newest(self, pair: str, *, as_of: datetime, target: date) -> FxRate | None:
        """The latest visible observation for a pair on or before `target`.

        Ordering by `as_of_date` then `ingested_at_utc` is what makes this
        as-of correct: the freshest *observation date* wins, and among vintages
        of that date the newest one we had *ingested by `as_of`*. Filtering on
        `ingested_at_utc` is easy to forget and is exactly the leak — a rate
        restated last week must not appear in a backtest of last month.
        """
        row: sqlite3.Row | None = self._ledger.conn.execute(
            "SELECT * FROM fx_rates WHERE pair = ? AND as_of_date <= ? "
            "AND available_at_utc <= ? AND ingested_at_utc <= ? "
            "ORDER BY as_of_date DESC, ingested_at_utc DESC LIMIT 1",
            (pair, target.isoformat(), to_iso(as_of), to_iso(as_of)),
        ).fetchone()
        return None if row is None else _from_row(row)

    def convert(
        self,
        amount: Decimal,
        *,
        base: str,
        quote: str,
        as_of: datetime,
        on_or_before: date | None = None,
        charge_fee: bool = True,
        places: int = 2,
    ) -> MaybeConversion:
        """Convert an amount, or return `UNKNOWN`.

        Never falls back to 1.0. That default is the seductive bug in this
        module: it makes everything run, and it makes a USD position look like a
        GBP position of the same magnitude — understating the drawdown against a
        GBP ceiling by about a quarter, silently, in the permissive direction.
        """
        if not isinstance(amount, Decimal):
            raise FxError(f"amounts must be Decimal, got {type(amount).__name__}")

        rate = self.rate_at(base, quote, as_of=as_of, on_or_before=on_or_before)
        if rate is None:
            return UNKNOWN

        with localcontext() as ctx:
            ctx.prec = _WORKING_PRECISION
            converted = amount * rate.rate
            fee = converted * FX_FEE_RATE if charge_fee else Decimal(0)
            quantum = Decimal(1).scaleb(-places)
            converted = converted.quantize(quantum, rounding=ROUND_HALF_EVEN)
            fee = fee.quantize(quantum, rounding=ROUND_HALF_EVEN)

        target = on_or_before or as_of.astimezone(UTC).date()
        return Conversion(
            amount=converted,
            fee=fee,
            rate=rate.rate,
            pair=rate.pair,
            as_of_date=rate.as_of_date,
            stale_days=rate.age_days_at(target),
        )

    def pairs_held(self) -> tuple[str, ...]:
        rows = self._ledger.conn.execute("SELECT DISTINCT pair FROM fx_rates").fetchall()
        return tuple(sorted(str(row["pair"]) for row in rows))

    def table_hash(self) -> str:
        """A content hash of the whole rate table, for a sealed vintage.

        Part of what makes a backtest reproducible: a run whose FX table has
        changed is not the same run, even if every bar is identical.
        """
        rows = self._ledger.conn.execute(
            "SELECT pair, as_of_date, available_at_utc, ingested_at_utc, rate, provider, "
            "provenance FROM fx_rates ORDER BY pair, as_of_date, ingested_at_utc"
        ).fetchall()
        return hash_payload(
            [
                {key: (None if row[key] is None else str(row[key])) for key in row.keys()}  # noqa: SIM118
                for row in rows
            ]
        )


# --------------------------------------------------------------------------
# Building rates from bars
# --------------------------------------------------------------------------


def rates_from_bars(
    bars: Iterable[object], *, base: str, quote: str, provider: str
) -> tuple[FxRate, ...]:
    """Turn provider bars for an FX symbol into rates.

    Yahoo serves `GBPUSD=X` as an ordinary daily bar, so the FX feed reuses the
    provider layer and inherits its validation, its knowledge times and its
    revision detection rather than growing a second, weaker ingest path.

    The rate taken is the **close**, and the knowledge time is the bar's
    `available_at_utc` unchanged — not the bar's date. A rate is knowable when
    the bar carrying it became knowable, which is the same rule as everywhere
    else in this layer.
    """
    pair = pair_key(base, quote)
    built: list[FxRate] = []
    for bar in bars:
        close = getattr(bar, "close", None)
        bar_open = getattr(bar, "bar_open_utc", None)
        available = getattr(bar, "available_at_utc", None)
        ingested = getattr(bar, "ingested_at_utc", None)
        provenance = getattr(bar, "provenance", Provenance.BACKFILL)
        if close is None or bar_open is None or available is None or ingested is None:
            raise FxError("an FX bar is missing the fields a rate needs")
        built.append(
            FxRate(
                pair=pair,
                as_of_date=bar_open.astimezone(UTC).date(),
                available_at_utc=available,
                ingested_at_utc=ingested,
                rate=parse_price(close),
                provider=provider,
                provenance=provenance,
            )
        )
    return tuple(built)


def flat_rate(
    *,
    base: str,
    quote: str,
    rate: Decimal,
    days: Sequence[date],
    provider: str = "manual",
) -> tuple[FxRate, ...]:
    """A constant rate over a span, for tests and for a same-currency account.

    Marked `provider="manual"` so an audit can find every number a human typed.
    Knowledge time is the end of each observation date, which is the earliest
    defensible instant for a daily close.
    """
    pair = pair_key(base, quote)
    return tuple(
        FxRate(
            pair=pair,
            as_of_date=day,
            available_at_utc=datetime(day.year, day.month, day.day, tzinfo=UTC) + timedelta(days=1),
            ingested_at_utc=datetime(day.year, day.month, day.day, tzinfo=UTC) + timedelta(days=1),
            rate=rate,
            provider=provider,
            provenance=Provenance.BACKFILL,
        )
        for day in days
    )


def _from_row(row: sqlite3.Row) -> FxRate:
    return FxRate(
        pair=str(row["pair"]),
        as_of_date=date.fromisoformat(str(row["as_of_date"])),
        available_at_utc=from_iso(str(row["available_at_utc"])),
        ingested_at_utc=from_iso(str(row["ingested_at_utc"])),
        rate=Decimal(str(row["rate"])),
        provider=str(row["provider"]),
        provenance=Provenance(str(row["provenance"])),
    )
