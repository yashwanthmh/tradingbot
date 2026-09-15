"""Mapping Trading 212 tickers to market-data symbols, and policing the join.

Trading 212 provides no market data, so every signal is computed on one venue's
prices and filled on another's. That join is the most dangerous piece of
plumbing in the system: a mismapped ticker produces a signal that looks
perfectly valid, passes every risk check, and buys the wrong company.

So mapping is treated as a risk control rather than a string transform:

* A derived mapping is **not tradable**. It becomes tradable only after a
  verification that compares the broker's own quote against the data feed's and
  finds them in agreement. On a fresh install nothing is tradable until
  `tb symbols audit` has run, which is the correct default.
* The gate is asymmetric. An unverified or disagreeing symbol blocks **new
  entries only** — exits and protective stops are always permitted. Refusing to
  sell something because its mapping looks doubtful would turn a data problem
  into an unhedged position.
* A price ratio near 100 is reported as a **unit mismatch**, not a
  disagreement. London quotes in pence and most feeds quote pounds; a naive
  comparison reads as a 9,900bps divergence, which is technically a block but
  tells whoever reads it nothing useful.

The ticker grammar below is a reconstruction from observed Trading 212 tickers
and is deliberately conservative: anything it cannot parse confidently is
marked `UNMAPPED` rather than guessed at. `currencyCode` and `isin` from the
instruments endpoint carry more weight than the ticker string, because they are
data rather than convention.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, replace
from decimal import Decimal
from enum import StrEnum
from typing import Any

from tb.broker.port import Instrument
from tb.core.clock import now_iso
from tb.ledger.events import Actor, EventType, SymbolBlockedPayload, SymbolsAuditedPayload
from tb.ledger.store import Ledger

# Trading 212 tickers look like `AAPL_US_EQ` for US listings, and for European
# listings a single lowercase venue letter is appended to the base symbol
# before `_EQ` — `VODl_EQ` for Vodafone on the LSE, for example.
_TICKER_RE = re.compile(
    r"^(?P<base>[A-Z0-9._-]+?)(?P<venue>[a-z])?(?:_(?P<market>[A-Z]{2}))?_(?P<kind>EQ|ETF|CFD)$"
)

# Venue letter to a yfinance suffix. Reconstructed and incomplete on purpose:
# an unrecognised letter yields an unmapped symbol rather than a wrong one.
_VENUE_SUFFIX: dict[str, str] = {
    "l": ".L",  # London
    "d": ".DE",  # Xetra
    "a": ".AS",  # Amsterdam
    "p": ".PA",  # Paris
    "m": ".MC",  # Madrid
    "i": ".MI",  # Milan
    "s": ".SW",  # Swiss
    "b": ".BR",  # Brussels
    "h": ".HE",  # Helsinki
    "e": ".IR",  # Ireland
}

# Currency codes that mean the quote is in minor units (pence, not pounds).
_MINOR_UNIT_CURRENCIES = frozenset({"GBX", "GBP_MINOR", "ZAC"})


class Confidence(StrEnum):
    """How much the mapping is trusted, in two verified tiers.

    Two tiers rather than one, because one tier deadlocks. Trading 212 reports
    `currentPrice` only on `/equity/portfolio` — that is, only for positions
    **already held**. So a broker quote is obtainable only for a symbol we own,
    and if owning one requires a broker quote first, the first position can
    never be opened. `tb symbols audit` printing "nothing is verified" was not a
    pending TODO; it was that deadlock.

    * `CROSS_VERIFIED` (weak) is corroboration that does not need the broker:
      two independent feeds agreeing on the price, plus matching currency and
      name against the broker's own instrument record. It authorises the
      **first floor-notional entry and nothing more** — which is what breaks
      the cycle, at a bounded cost.
    * `VERIFIED` (strong) is the broker's own quote agreeing with the feed,
      which becomes possible once the position exists. It permits full size.

    The weak tier is deliberately cheap to obtain and expensive to rely on: its
    whole job is to risk one floor-notional position in order to unlock the
    evidence the strong tier needs.
    """

    VERIFIED = "verified"
    CROSS_VERIFIED = "cross_verified"
    DERIVED = "derived"
    AMBIGUOUS = "ambiguous"
    UNMAPPED = "unmapped"

    @property
    def tradable(self) -> bool:
        """Whether a new position may be opened at all — at *some* size."""
        return self in (Confidence.VERIFIED, Confidence.CROSS_VERIFIED)

    @property
    def permits_full_size(self) -> bool:
        """Whether the position may exceed the floor notional.

        Only the broker's own quote earns this. Cross-provider agreement says
        two vendors describe the same instrument; it does not say the broker
        agrees that the ticker we are about to send maps to it.
        """
        return self is Confidence.VERIFIED

    @property
    def floor_size_only(self) -> bool:
        return self is Confidence.CROSS_VERIFIED


@dataclass(frozen=True, slots=True)
class ParsedTicker:
    raw: str
    base: str
    venue_letter: str | None
    market: str | None
    kind: str

    @property
    def is_us(self) -> bool:
        return self.market == "US"


@dataclass(frozen=True, slots=True)
class SymbolMapping:
    t212_ticker: str
    data_symbol: str
    provider: str
    confidence: Confidence
    derivation: str
    currency_code: str | None = None
    exchange_hint: str | None = None
    verified_at: str | None = None
    last_disagreement_bps: float | None = None
    last_checked_at: str | None = None
    blocked: bool = False
    blocked_reason: str | None = None

    @property
    def may_enter(self) -> bool:
        """New positions need a verified, unblocked mapping — at some size.

        Coarse on purpose: whether the *size* is capped to the floor notional is
        `permits_full_size`, and the gate that enforces it is
        `tb.data.verification.SymbolVerifier.entry_permission`.
        """
        return self.confidence.tradable and not self.blocked

    @property
    def permits_full_size(self) -> bool:
        return self.confidence.permits_full_size and not self.blocked

    @property
    def may_exit(self) -> bool:
        """Always true.

        Refusing to sell because a mapping is doubtful converts a data problem
        into an unhedged position. Exit prices come from the broker anyway.
        """
        return True

    @property
    def quotes_in_minor_units(self) -> bool:
        return (self.currency_code or "").upper() in _MINOR_UNIT_CURRENCIES


class DisagreementKind(StrEnum):
    AGREES = "agrees"
    UNIT_MISMATCH = "unit_mismatch"
    DIVERGED = "diverged"
    MISSING_PRICE = "missing_price"


@dataclass(frozen=True, slots=True)
class PriceComparison:
    t212_ticker: str
    data_symbol: str
    broker_price: Decimal | None
    data_price: Decimal | None
    disagreement_bps: float | None
    kind: DisagreementKind
    detail: str

    @property
    def should_block(self) -> bool:
        return self.kind is not DisagreementKind.AGREES


def parse_ticker(ticker: str) -> ParsedTicker | None:
    """Split a Trading 212 ticker. Returns None if the grammar does not fit."""
    match = _TICKER_RE.match(ticker.strip())
    if match is None:
        return None
    return ParsedTicker(
        raw=ticker,
        base=match.group("base"),
        venue_letter=match.group("venue"),
        market=match.group("market"),
        kind=match.group("kind"),
    )


def derive_mapping(instrument: Instrument, *, provider: str) -> SymbolMapping:
    """Propose a data-provider symbol for a Trading 212 instrument.

    Uses the instrument's `currencyCode` in preference to the ticker string
    wherever they disagree, because currency is data and the ticker suffix is
    convention.
    """
    ticker = instrument.ticker
    parsed = parse_ticker(ticker)
    currency = (instrument.currency_code or "").upper()

    if parsed is None:
        return SymbolMapping(
            t212_ticker=ticker,
            data_symbol="",
            provider=provider,
            confidence=Confidence.UNMAPPED,
            derivation="ticker did not match the known Trading 212 grammar",
            currency_code=instrument.currency_code,
        )

    # Alpaca's free feed covers US equities and ETFs only, so a non-US listing
    # is unmapped there rather than mapped to a same-named US security — which
    # is exactly the mismapping that buys the wrong company.
    if provider == "alpaca":
        if parsed.is_us or currency == "USD":
            return SymbolMapping(
                t212_ticker=ticker,
                data_symbol=parsed.base,
                provider=provider,
                confidence=Confidence.DERIVED,
                derivation="US listing: base symbol used directly",
                currency_code=instrument.currency_code,
                exchange_hint="US",
            )
        return SymbolMapping(
            t212_ticker=ticker,
            data_symbol="",
            provider=provider,
            confidence=Confidence.UNMAPPED,
            derivation=(
                f"Alpaca's free feed is US-only and this instrument is quoted in "
                f"{instrument.currency_code or 'an unknown currency'}"
            ),
            currency_code=instrument.currency_code,
        )

    if provider == "yfinance":
        if parsed.is_us or (currency == "USD" and parsed.venue_letter is None):
            return SymbolMapping(
                t212_ticker=ticker,
                data_symbol=parsed.base,
                provider=provider,
                confidence=Confidence.DERIVED,
                derivation="US listing: base symbol used directly",
                currency_code=instrument.currency_code,
                exchange_hint="US",
            )
        if parsed.venue_letter is not None:
            suffix = _VENUE_SUFFIX.get(parsed.venue_letter)
            if suffix is None:
                return SymbolMapping(
                    t212_ticker=ticker,
                    data_symbol="",
                    provider=provider,
                    confidence=Confidence.UNMAPPED,
                    derivation=(
                        f"venue letter {parsed.venue_letter!r} is not in the known "
                        "suffix table; refusing to guess an exchange"
                    ),
                    currency_code=instrument.currency_code,
                )
            return SymbolMapping(
                t212_ticker=ticker,
                data_symbol=f"{parsed.base}{suffix}",
                provider=provider,
                confidence=Confidence.DERIVED,
                derivation=f"venue letter {parsed.venue_letter!r} -> {suffix}",
                currency_code=instrument.currency_code,
                exchange_hint=suffix.lstrip("."),
            )
        return SymbolMapping(
            t212_ticker=ticker,
            data_symbol=parsed.base,
            provider=provider,
            confidence=Confidence.AMBIGUOUS,
            derivation=(
                "no venue letter and not obviously a US listing, so the bare base "
                "symbol may resolve to a different company on the provider"
            ),
            currency_code=instrument.currency_code,
        )

    return SymbolMapping(
        t212_ticker=ticker,
        data_symbol="",
        provider=provider,
        confidence=Confidence.UNMAPPED,
        derivation=f"no derivation rules for provider {provider!r}",
        currency_code=instrument.currency_code,
    )


def compare_prices(
    mapping: SymbolMapping,
    *,
    broker_price: Decimal | None,
    data_price: Decimal | None,
    limit_bps: float,
) -> PriceComparison:
    """Check the data feed against the broker's own quote.

    A ratio near 100 or 0.01 is called out as a unit mismatch. London lists in
    pence while most feeds quote pounds, and reporting that as a 9,900bps
    divergence is technically a block but tells the reader nothing.
    """
    if broker_price is None or data_price is None:
        missing = "broker" if broker_price is None else "data feed"
        return PriceComparison(
            t212_ticker=mapping.t212_ticker,
            data_symbol=mapping.data_symbol,
            broker_price=broker_price,
            data_price=data_price,
            disagreement_bps=None,
            kind=DisagreementKind.MISSING_PRICE,
            detail=f"no price from the {missing}; cannot verify the mapping",
        )

    if broker_price <= 0 or data_price <= 0:
        return PriceComparison(
            t212_ticker=mapping.t212_ticker,
            data_symbol=mapping.data_symbol,
            broker_price=broker_price,
            data_price=data_price,
            disagreement_bps=None,
            kind=DisagreementKind.DIVERGED,
            detail=f"non-positive price (broker {broker_price}, data {data_price})",
        )

    ratio = float(data_price / broker_price)
    for factor, label in ((100.0, "x100"), (0.01, "/100")):
        if abs(ratio / factor - 1.0) < 0.05:
            return PriceComparison(
                t212_ticker=mapping.t212_ticker,
                data_symbol=mapping.data_symbol,
                broker_price=broker_price,
                data_price=data_price,
                disagreement_bps=None,
                kind=DisagreementKind.UNIT_MISMATCH,
                detail=(
                    f"data price is {label} the broker price "
                    f"({data_price} vs {broker_price}) — almost certainly a "
                    "major/minor unit mismatch such as pence against pounds, "
                    "not a real divergence. Fix the mapping's unit rather than "
                    "widening the tolerance."
                ),
            )

    mid = (broker_price + data_price) / Decimal(2)
    bps = float(abs(broker_price - data_price) / mid) * 10_000.0

    if bps <= limit_bps:
        return PriceComparison(
            t212_ticker=mapping.t212_ticker,
            data_symbol=mapping.data_symbol,
            broker_price=broker_price,
            data_price=data_price,
            disagreement_bps=bps,
            kind=DisagreementKind.AGREES,
            detail=f"agree within {bps:.1f}bps (limit {limit_bps:.0f}bps)",
        )

    return PriceComparison(
        t212_ticker=mapping.t212_ticker,
        data_symbol=mapping.data_symbol,
        broker_price=broker_price,
        data_price=data_price,
        disagreement_bps=bps,
        kind=DisagreementKind.DIVERGED,
        detail=(
            f"diverge by {bps:.1f}bps (limit {limit_bps:.0f}bps): broker "
            f"{broker_price}, data {data_price}. Could be a stale feed, an "
            "unhandled corporate action, or a mismapped ticker — none of which "
            "is a condition to open a position in."
        ),
    )


class SymbolMap:
    """The persisted mapping, and the entry gate built on it."""

    def __init__(self, ledger: Ledger, *, provider: str = "alpaca") -> None:
        self._ledger = ledger
        self._provider = provider

    @property
    def provider(self) -> str:
        return self._provider

    # -- persistence -------------------------------------------------------

    def upsert(self, mapping: SymbolMapping) -> None:
        self._ledger.conn.execute(
            """
            INSERT INTO symbol_map (
                t212_ticker, data_symbol, provider, currency_code, exchange_hint,
                confidence, derivation, verified_at, last_disagreement_bps,
                last_checked_at, blocked, blocked_reason
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(t212_ticker) DO UPDATE SET
                data_symbol = excluded.data_symbol,
                provider = excluded.provider,
                currency_code = excluded.currency_code,
                exchange_hint = excluded.exchange_hint,
                confidence = excluded.confidence,
                derivation = excluded.derivation,
                -- Verification is sticky: a re-derivation must not silently
                -- discard a confirmation that a price comparison earned.
                verified_at = COALESCE(excluded.verified_at, symbol_map.verified_at),
                last_disagreement_bps = COALESCE(
                    excluded.last_disagreement_bps, symbol_map.last_disagreement_bps
                ),
                last_checked_at = COALESCE(excluded.last_checked_at, symbol_map.last_checked_at),
                blocked = excluded.blocked,
                blocked_reason = excluded.blocked_reason
            """,
            (
                mapping.t212_ticker,
                mapping.data_symbol,
                mapping.provider,
                mapping.currency_code,
                mapping.exchange_hint,
                mapping.confidence.value,
                mapping.derivation,
                mapping.verified_at,
                mapping.last_disagreement_bps,
                mapping.last_checked_at,
                int(mapping.blocked),
                mapping.blocked_reason,
            ),
        )
        self._ledger.conn.commit()

    def get(self, t212_ticker: str) -> SymbolMapping | None:
        row = self._ledger.conn.execute(
            "SELECT * FROM symbol_map WHERE t212_ticker = ?", (t212_ticker,)
        ).fetchone()
        return None if row is None else _row_to_mapping(row)

    def all(self) -> list[SymbolMapping]:
        rows = self._ledger.conn.execute("SELECT * FROM symbol_map ORDER BY t212_ticker").fetchall()
        return [_row_to_mapping(r) for r in rows]

    def data_symbol_for(self, t212_ticker: str) -> str | None:
        mapping = self.get(t212_ticker)
        return mapping.data_symbol if mapping and mapping.data_symbol else None

    def t212_ticker_for(self, data_symbol: str) -> str | None:
        row = self._ledger.conn.execute(
            "SELECT t212_ticker FROM symbol_map WHERE data_symbol = ? AND provider = ?",
            (data_symbol, self._provider),
        ).fetchone()
        return None if row is None else str(row["t212_ticker"])

    # -- the gate ----------------------------------------------------------

    def may_enter(self, t212_ticker: str) -> tuple[bool, str]:
        """Whether a new position may be opened in this instrument."""
        mapping = self.get(t212_ticker)
        if mapping is None:
            return False, (
                f"{t212_ticker} has no symbol mapping. Run `tb symbols audit` — "
                "an unmapped instrument cannot be priced, so it cannot be sized."
            )
        if mapping.blocked:
            return False, f"{t212_ticker} is blocked: {mapping.blocked_reason}"
        if not mapping.confidence.tradable:
            return False, (
                f"{t212_ticker} -> {mapping.data_symbol or '(none)'} is "
                f"{mapping.confidence.value}, not verified ({mapping.derivation}). "
                "A derived mapping has been confirmed by nothing at all — neither a "
                "second feed nor the broker's own quote."
            )
        if mapping.confidence.floor_size_only:
            # Said out loud rather than left to the caller to infer. A bool that
            # means "yes, but only at the floor" read as plain "yes" is the whole
            # risk of having a weak tier at all.
            return True, (
                "cross-provider verified only: FLOOR NOTIONAL ONLY. Two feeds agree and "
                "the reference data matches, but the broker has not confirmed this "
                "mapping — it cannot until a position exists. Size with "
                "`SymbolVerifier.entry_permission`, which enforces the cap."
            )
        return True, "broker-quote verified; full size permitted"

    def may_exit(self, t212_ticker: str) -> tuple[bool, str]:
        """Always permitted. See `SymbolMapping.may_exit`."""
        return True, "exits are never gated on mapping confidence"

    def block(self, comparison: PriceComparison, *, limit_bps: float | None = None) -> None:
        """Stop a symbol accepting new entries after a cross-venue disagreement."""
        self.block_for(
            comparison.t212_ticker,
            reason=f"{comparison.kind.value}: {comparison.detail}",
            data_symbol=comparison.data_symbol,
            disagreement_bps=comparison.disagreement_bps,
            limit_bps=limit_bps,
            broker_price=comparison.broker_price,
            data_price=comparison.data_price,
        )

    def block_for(
        self,
        t212_ticker: str,
        *,
        reason: str,
        data_symbol: str | None = None,
        disagreement_bps: float | None = None,
        limit_bps: float | None = None,
        broker_price: Decimal | None = None,
        data_price: Decimal | None = None,
    ) -> None:
        """Stop a symbol accepting new entries, for any reason.

        Generalised from the cross-venue case because that is not the only
        thing that should stop an entry. An unexplained split is the other one:
        a price jump that fits a small-integer ratio with no action row behind
        it means the vendor restated history without reporting why, and sizing
        a position against that series is sizing against a guess.

        Blocking is deliberately asymmetric and `may_exit` is unaffected —
        refusing to *sell* over a data problem would convert it into an
        unhedged position, which is worse than the problem.
        """
        self._ledger.conn.execute(
            "UPDATE symbol_map SET blocked = 1, blocked_reason = ?, "
            "last_disagreement_bps = ?, last_checked_at = ? WHERE t212_ticker = ?",
            (reason, disagreement_bps, now_iso(), t212_ticker),
        )
        self._ledger.conn.commit()
        self._ledger.append(
            EventType.SYMBOL_BLOCKED,
            t212_ticker,
            SymbolBlockedPayload(
                t212_ticker=t212_ticker,
                data_symbol=data_symbol or "",
                reason=reason,
                disagreement_bps=disagreement_bps,
                limit_bps=limit_bps,
                broker_price=broker_price,
                data_price=data_price,
            ),
            actor=Actor.SYSTEM,
        )

    def verify(
        self,
        comparison: PriceComparison,
        *,
        limit_bps: float,
        confidence: Confidence = Confidence.VERIFIED,
    ) -> None:
        """Promote a mapping to a verified tier after prices agree.

        `confidence` names which tier the evidence supports. It defaults to the
        strong tier so the M1 call sites keep their meaning, and
        `SymbolVerifier` passes `CROSS_VERIFIED` for the weak one — see
        `tb.data.verification` for why one tier deadlocks.
        """
        if not confidence.tradable:
            raise ValueError(
                f"verify() was asked to record {confidence.value}, which is not a "
                "verified tier. Use block() to record a failure."
            )
        timestamp = now_iso()
        self._ledger.conn.execute(
            "UPDATE symbol_map SET confidence = ?, verified_at = ?, "
            "last_disagreement_bps = ?, last_checked_at = ?, blocked = 0, "
            "blocked_reason = NULL WHERE t212_ticker = ?",
            (
                confidence.value,
                timestamp,
                comparison.disagreement_bps,
                timestamp,
                comparison.t212_ticker,
            ),
        )
        self._ledger.conn.commit()
        self._ledger.append(
            EventType.SYMBOL_UNBLOCKED,
            comparison.t212_ticker,
            SymbolBlockedPayload(
                t212_ticker=comparison.t212_ticker,
                data_symbol=comparison.data_symbol,
                reason=f"verified: {comparison.detail}",
                disagreement_bps=comparison.disagreement_bps,
                limit_bps=limit_bps,
                broker_price=comparison.broker_price,
                data_price=comparison.data_price,
            ),
            actor=Actor.SYSTEM,
        )

    # -- bulk operations ---------------------------------------------------

    def derive_all(self, instruments: tuple[Instrument, ...]) -> dict[str, int]:
        """Derive and store a mapping for every instrument.

        Derivation alone never makes anything tradable — see `may_enter`.
        """
        counts = {c.value: 0 for c in Confidence}
        for instrument in instruments:
            mapping = derive_mapping(instrument, provider=self._provider)
            existing = self.get(instrument.ticker)
            # Preserve an existing verification rather than demoting it. Both
            # tiers are preserved: re-deriving would otherwise reset every
            # cross-verified symbol to untradable on each audit, re-creating the
            # deadlock one audit at a time.
            if existing is not None and existing.confidence.tradable:
                if existing.data_symbol == mapping.data_symbol:
                    counts[existing.confidence.value] += 1
                    continue
                # The derived symbol changed, which invalidates the old
                # verification: whatever was confirmed is not what we would use.
                mapping = replace(
                    mapping,
                    blocked=True,
                    blocked_reason=(
                        f"derived symbol changed from {existing.data_symbol!r} to "
                        f"{mapping.data_symbol!r}; the earlier verification no "
                        "longer applies"
                    ),
                )
            self.upsert(mapping)
            counts[mapping.confidence.value] += 1
        return counts

    def audit_summary(self, *, n_instruments: int) -> dict[str, Any]:
        rows = self.all()
        return {
            "provider": self._provider,
            "n_instruments": n_instruments,
            "n_mapped": sum(1 for m in rows if m.data_symbol),
            "n_unmapped": sum(1 for m in rows if not m.data_symbol),
            "n_verified": sum(1 for m in rows if m.confidence is Confidence.VERIFIED),
            "n_cross_verified": sum(1 for m in rows if m.confidence is Confidence.CROSS_VERIFIED),
            # What the deadlock check actually asks: is *anything* enterable?
            "n_enterable": sum(1 for m in rows if m.may_enter),
            "n_derived": sum(1 for m in rows if m.confidence is Confidence.DERIVED),
            "n_ambiguous": sum(1 for m in rows if m.confidence is Confidence.AMBIGUOUS),
            "n_blocked": sum(1 for m in rows if m.blocked),
        }

    def record_audit(self, summary: dict[str, Any]) -> None:
        unmapped = [m.t212_ticker for m in self.all() if not m.data_symbol][:20]
        self._ledger.append(
            EventType.SYMBOLS_AUDITED,
            self._provider,
            SymbolsAuditedPayload(
                provider=self._provider,
                n_instruments=int(summary["n_instruments"]),
                n_mapped=int(summary["n_mapped"]),
                n_unmapped=int(summary["n_unmapped"]),
                n_low_confidence=int(summary["n_derived"]) + int(summary["n_ambiguous"]),
                n_blocked=int(summary["n_blocked"]),
                unmapped_sample=unmapped,
            ),
        )


def _row_to_mapping(row: Any) -> SymbolMapping:
    return SymbolMapping(
        t212_ticker=row["t212_ticker"],
        data_symbol=row["data_symbol"],
        provider=row["provider"],
        confidence=Confidence(row["confidence"]),
        derivation=row["derivation"],
        currency_code=row["currency_code"],
        exchange_hint=row["exchange_hint"],
        verified_at=row["verified_at"],
        last_disagreement_bps=row["last_disagreement_bps"],
        last_checked_at=row["last_checked_at"],
        blocked=bool(row["blocked"]),
        blocked_reason=row["blocked_reason"],
    )
