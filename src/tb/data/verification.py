"""Proving that a Trading 212 ticker and a data symbol are the same instrument.

This module exists to fix a deadlock in the shipped M1 code, and the deadlock
is worth stating precisely because the fix only makes sense against it.

Trading 212 reports `currentPrice` **only on `/equity/portfolio`** — only for
positions already held. `compare_prices` returns `MISSING_PRICE` when either
side is absent, and only `AGREES` leads to `verify()`. So a *candidate* symbol
can never obtain a broker price, never reaches `AGREES`, never gets verified,
and `may_enter` stays False permanently. Owning a position required a broker
quote; getting a broker quote required owning a position. `tb symbols audit`
reporting "nothing is verified" was not a pending TODO. It was the reason the
first trade could never happen.

**The fix is a two-tier gate.** The weak tier corroborates the mapping without
the broker at all — two independent feeds agreeing on the price, plus currency
and name matching the broker's own instrument record — and authorises exactly
one floor-notional entry. That entry produces a held position, which produces a
broker quote, which the strong tier checks. The cycle is broken by deliberately
risking a bounded amount (`capital.floor_notional_ccy`, £15) on weaker evidence.

Three things make that safe rather than merely convenient:

**Absence is not contradiction.** A missing ISIN on the provider side is no
evidence; two *different* ISINs is proof of a mismapping. The weak tier needs
at least one positive corroboration and zero contradictions, so a single
unreachable feed degrades the evidence instead of deadlocking again — and a
genuine mismatch is fatal no matter how much other evidence agrees.

**The weak tier cannot clear a block.** If the strong tier blocked a symbol for
price divergence, cross-provider agreement must not unblock it. The feeds
agreeing with each other says nothing about the broker disagreeing with both.

**The weak tier expires.** Its evidence is one price comparison at one instant;
a symbol change or an unannounced action invalidates it. After
`CROSS_VERIFIED_TTL` it stops authorising entries until re-checked.

Two landmines in the same path, both handled explicitly:

*Session.* Trading 212's overnight `currentPrice` is the last regular-hours
close, while a provider's "latest bar" may be a pre- or post-market print.
Earnings-night extended-hours moves of 200-500bps are routine, so comparing
across sessions against a 75bps threshold would block half the universe every
evening. Comparisons run on regular-session bars of the same bar period only.

*Currency.* Whether T212's `currentPrice` is quoted in the instrument's
currency or the account's is not documented. A USD name reported in GBP is a
~2600bps gap that the existing x100 unit-mismatch detector would mislabel as a
mismapped ticker — sending the investigation in exactly the wrong direction. So
the basis is a recorded observation (`QuoteCurrencyBasis`), it can be *inferred*
from a held position when the two hypotheses are far enough apart to be
distinguishable, and while it is undetermined and the currencies differ the
strong tier refuses to verify rather than guess. That refusal leaves the symbol
stuck at floor size — reported, bounded, and never blocking an exit.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timedelta
from decimal import Decimal
from enum import StrEnum

from tb.broker.port import Instrument
from tb.core.clock import from_iso, now_utc
from tb.data.calendar import TradingCalendar
from tb.data.provider import Bar, DataError, Session
from tb.data.symbols import (
    Confidence,
    DisagreementKind,
    PriceComparison,
    SymbolMap,
    SymbolMapping,
    compare_prices,
)

# How long cross-provider evidence authorises entries before it must be
# re-checked. One trading day: the evidence is a single price comparison, and a
# symbol change or an unannounced corporate action between then and now would
# invalidate it silently.
CROSS_VERIFIED_TTL = timedelta(hours=30)

# Below this, two names are not the same company. Deliberately lenient — "Apple
# Inc." against "Apple" must pass, and so must "Alphabet Inc. Class A" against
# "Alphabet". It is a contradiction detector, not a matching algorithm: its job
# is to catch AAPL mapped to something called "Applied Signal Technology".
MIN_NAME_SIMILARITY = 0.34

# How far apart the two currency hypotheses must be before an observation can
# tell them apart. At GBPUSD ≈ 1.27 the instrument-currency and
# account-currency readings differ by 27%, which is unmistakable; near parity
# they are indistinguishable and inferring either would be a coin flip.
MIN_BASIS_SEPARATION_PCT = Decimal("8")


class VerificationError(DataError):
    """A verification could not be carried out."""


class Tier(StrEnum):
    """What kind of evidence a mapping rests on."""

    NONE = "none"
    CROSS_PROVIDER = "cross_provider"
    BROKER_QUOTE = "broker_quote"

    @property
    def permits_entry(self) -> bool:
        return self is not Tier.NONE

    @property
    def permits_full_size(self) -> bool:
        return self is Tier.BROKER_QUOTE

    @property
    def confidence(self) -> Confidence:
        return {
            Tier.NONE: Confidence.DERIVED,
            Tier.CROSS_PROVIDER: Confidence.CROSS_VERIFIED,
            Tier.BROKER_QUOTE: Confidence.VERIFIED,
        }[self]


class QuoteCurrencyBasis(StrEnum):
    """Which currency Trading 212's `currentPrice` is quoted in.

    Undocumented, so it is an observation rather than an assumption.
    `UNDETERMINED` is the honest default and is fail-closed for the strong
    tier: a USD name reported in GBP is a ~2600bps gap, and the existing
    unit-mismatch detector would call that a mismapped ticker.
    """

    INSTRUMENT = "instrument"
    ACCOUNT = "account"
    UNDETERMINED = "undetermined"


class Corroboration(StrEnum):
    """One piece of positive evidence that two symbols are the same thing."""

    PRICE_AGREES = "price_agrees"
    ISIN_MATCHES = "isin_matches"
    CURRENCY_MATCHES = "currency_matches"
    NAME_MATCHES = "name_matches"


class Contradiction(StrEnum):
    """One piece of evidence that they are *not*.

    Distinct from absence, and fatal regardless of how much else agrees. A
    mapping with a matching name, a matching currency and a *different ISIN* is
    a mismapping with good luck, not a match.
    """

    ISIN_DIFFERS = "isin_differs"
    CURRENCY_DIFFERS = "currency_differs"
    NAME_DIFFERS = "name_differs"
    PRICE_DIVERGES = "price_diverges"
    UNIT_MISMATCH = "unit_mismatch"
    SESSION_MISMATCH = "session_mismatch"
    BAR_PERIOD_MISMATCH = "bar_period_mismatch"
    SAME_PROVIDER = "same_provider"


@dataclass(frozen=True, slots=True)
class VerificationResult:
    """What a verification attempt found, and what it changed.

    Carries every corroboration and every contradiction rather than a verdict
    alone, because "why is this symbol not tradable" is the question the audit
    exists to answer and a bare False answers nothing.
    """

    t212_ticker: str
    data_symbol: str
    tier: Tier
    corroborations: tuple[Corroboration, ...] = field(default_factory=tuple)
    contradictions: tuple[Contradiction, ...] = field(default_factory=tuple)
    disagreement_bps: float | None = None
    detail: str = ""

    @property
    def verified(self) -> bool:
        return self.tier.permits_entry and not self.contradictions

    @property
    def fatal(self) -> bool:
        """Whether the evidence positively contradicts the mapping."""
        return bool(self.contradictions)


@dataclass(frozen=True, slots=True)
class EntryPermission:
    """Whether an entry may be placed, and how large it may be.

    **This is the gate the order path calls**, not `SymbolMap.may_enter`. That
    returns a bare bool which reads as plain permission; the size cap lives
    here, where it cannot be dropped by reading only the first element of a
    tuple.
    """

    allowed: bool
    t212_ticker: str
    tier: Tier
    max_notional: Decimal | None
    reasons: tuple[str, ...] = field(default_factory=tuple)

    @property
    def detail(self) -> str:
        return "; ".join(self.reasons) if self.reasons else "verified for full size"

    def cap(self, requested: Decimal) -> Decimal:
        """The largest notional this permission allows for a request.

        Returns the cap rather than raising, so the caller's own sizing rules
        compose with it — but it never returns more than was asked for, and
        never more than the tier allows.
        """
        if not self.allowed:
            raise VerificationError(f"{self.t212_ticker}: no entry is permitted ({self.detail})")
        if self.max_notional is None:
            return requested
        return min(requested, self.max_notional)


# --------------------------------------------------------------------------
# Reference-data cross-matching
# --------------------------------------------------------------------------


def name_similarity(left: str | None, right: str | None) -> float | None:
    """Token overlap between two instrument names, or None if either is absent.

    Deliberately crude and dependency-free. Precision is not the point: this is
    a contradiction detector, and its only job is to notice that the symbol we
    derived belongs to a company with an unrelated name. Returning None for a
    missing name matters more than the number — no name is no evidence, and
    treating it as a mismatch would reject every instrument the broker
    describes tersely.
    """
    if not left or not right:
        return None
    noise = {
        "inc",
        "inc.",
        "corp",
        "corp.",
        "corporation",
        "co",
        "co.",
        "company",
        "plc",
        "ltd",
        "limited",
        "the",
        "class",
        "cl",
        "a",
        "b",
        "c",
        "common",
        "stock",
        "shares",
        "ordinary",
        "sa",
        "nv",
        "ag",
        "holdings",
        "group",
    }

    def tokens(text: str) -> set[str]:
        cleaned = "".join(ch if ch.isalnum() or ch.isspace() else " " for ch in text.lower())
        return {word for word in cleaned.split() if word and word not in noise}

    first, second = tokens(left), tokens(right)
    if not first or not second:
        return None
    return len(first & second) / len(first | second)


@dataclass(frozen=True, slots=True)
class ProviderReference:
    """What a data provider says an instrument is.

    Everything is optional because free feeds report different subsets, and
    absence must stay distinguishable from disagreement.
    """

    symbol: str
    provider: str
    name: str | None = None
    currency: str | None = None
    isin: str | None = None
    exchange: str | None = None
    tradable: bool | None = None


def cross_match_reference(
    instrument: Instrument,
    reference: ProviderReference,
    *,
    min_name_similarity: float = MIN_NAME_SIMILARITY,
) -> tuple[tuple[Corroboration, ...], tuple[Contradiction, ...], str]:
    """Compare broker and provider reference data, field by field.

    Returns corroborations, contradictions and a human sentence. The asymmetry
    is the whole design: a field present on both sides and *disagreeing* is a
    contradiction, while a field missing on either side contributes nothing at
    all. Scoring absence as disagreement would reject most of the universe;
    scoring it as agreement would let a mismapping through on a shrug.
    """
    corroborations: list[Corroboration] = []
    contradictions: list[Contradiction] = []
    notes: list[str] = []

    broker_isin = (instrument.isin or "").strip().upper()
    provider_isin = (reference.isin or "").strip().upper()
    if broker_isin and provider_isin:
        if broker_isin == provider_isin:
            corroborations.append(Corroboration.ISIN_MATCHES)
            notes.append(f"ISIN matches ({broker_isin})")
        else:
            contradictions.append(Contradiction.ISIN_DIFFERS)
            notes.append(
                f"ISIN differs: broker {broker_isin}, {reference.provider} "
                f"{provider_isin}. These are different securities."
            )
    elif broker_isin or provider_isin:
        notes.append(
            f"only one side reports an ISIN, so it is no evidence either way "
            f"({'broker' if broker_isin else reference.provider} only)"
        )

    broker_ccy = (instrument.currency_code or "").strip().upper()
    provider_ccy = (reference.currency or "").strip().upper()
    if broker_ccy and provider_ccy:
        if broker_ccy == provider_ccy:
            corroborations.append(Corroboration.CURRENCY_MATCHES)
            notes.append(f"currency matches ({broker_ccy})")
        else:
            contradictions.append(Contradiction.CURRENCY_DIFFERS)
            notes.append(
                f"currency differs: broker {broker_ccy}, {reference.provider} "
                f"{provider_ccy}. Either the mapping is wrong or the quote needs "
                "converting before any comparison means anything."
            )

    similarity = name_similarity(instrument.full_name or instrument.short_name, reference.name)
    if similarity is not None:
        if similarity >= min_name_similarity:
            corroborations.append(Corroboration.NAME_MATCHES)
            notes.append(f"name overlap {similarity:.2f}")
        else:
            contradictions.append(Contradiction.NAME_DIFFERS)
            notes.append(
                f"names share almost nothing ({similarity:.2f}): broker "
                f"{instrument.full_name or instrument.short_name!r}, "
                f"{reference.provider} {reference.name!r}"
            )

    if reference.tradable is False:
        notes.append(
            f"{reference.provider} reports {reference.symbol} as not tradable, which "
            "usually means delisted or halted"
        )

    return tuple(corroborations), tuple(contradictions), "; ".join(notes)


# --------------------------------------------------------------------------
# Comparing two feeds
# --------------------------------------------------------------------------


def compare_provider_bars(
    mapping: SymbolMapping,
    primary: Bar | None,
    secondary: Bar | None,
    *,
    limit_bps: float,
) -> tuple[tuple[Corroboration, ...], tuple[Contradiction, ...], float | None, str]:
    """Check two feeds against each other on the same bar.

    Three preconditions, each of which would otherwise produce a confident
    wrong answer:

    * **Different providers.** Comparing a feed against itself proves nothing
      and would verify every symbol in the universe.
    * **Same bar period.** On a moving stock, comparing one feed's 15:59 bar to
      another's 15:45 bar measures the price change, not the mapping.
    * **Regular session both sides.** An extended-hours print can sit hundreds
      of basis points from the regular close, so a cross-session comparison
      fails for a reason that has nothing to do with the mapping.

    Absence of either bar is not a contradiction — it is no evidence, which
    matters because a single unreachable feed must degrade the weak tier rather
    than deadlock it again.
    """
    if primary is None or secondary is None:
        missing = "primary" if primary is None else "secondary"
        return (), (), None, f"no {missing}-feed bar, so the feeds cannot corroborate"

    if primary.provider == secondary.provider:
        return (
            (),
            (Contradiction.SAME_PROVIDER,),
            None,
            f"both bars came from {primary.provider}; a feed agreeing with itself is "
            "not corroboration",
        )

    if primary.bar_open_utc != secondary.bar_open_utc:
        return (
            (),
            (Contradiction.BAR_PERIOD_MISMATCH,),
            None,
            f"bar periods differ ({primary.bar_open_utc.isoformat()} vs "
            f"{secondary.bar_open_utc.isoformat()}); comparing them measures the price "
            "change over the gap, not the mapping",
        )

    if primary.session is not Session.REGULAR or secondary.session is not Session.REGULAR:
        return (
            (),
            (Contradiction.SESSION_MISMATCH,),
            None,
            f"sessions are {primary.session.value}/{secondary.session.value}; only "
            "regular-session prints are comparable",
        )

    comparison = compare_prices(
        mapping,
        broker_price=primary.close,
        data_price=secondary.close,
        limit_bps=limit_bps,
    )
    if comparison.kind is DisagreementKind.AGREES:
        return (
            (Corroboration.PRICE_AGREES,),
            (),
            comparison.disagreement_bps,
            f"{primary.provider} and {secondary.provider} agree: {comparison.detail}",
        )
    if comparison.kind is DisagreementKind.UNIT_MISMATCH:
        return (
            (),
            (Contradiction.UNIT_MISMATCH,),
            comparison.disagreement_bps,
            comparison.detail,
        )
    return (
        (),
        (Contradiction.PRICE_DIVERGES,),
        comparison.disagreement_bps,
        comparison.detail,
    )


# --------------------------------------------------------------------------
# Inferring the quote basis
# --------------------------------------------------------------------------


def infer_quote_basis(
    *,
    broker_price: Decimal,
    data_price: Decimal,
    fx_rate: Decimal | None,
    tolerance_pct: Decimal = Decimal("3"),
    min_separation_pct: Decimal = MIN_BASIS_SEPARATION_PCT,
) -> tuple[QuoteCurrencyBasis, str]:
    """Work out which currency the broker's quote is in, from a held position.

    This is how the undocumented fact becomes a measurement rather than a
    guess. With the instrument's price from the feed and the account-currency
    rate, there are two hypotheses and they predict different numbers; whichever
    the broker's quote matches is the answer.

    It returns `UNDETERMINED` in the case that matters most: when the two
    hypotheses are too *close* to tell apart. Near parity, `price` and
    `price * rate` are almost the same number, so matching one is not evidence
    against the other — and confidently recording the wrong basis would apply a
    spurious conversion to every future comparison.
    """
    if broker_price <= 0 or data_price <= 0:
        raise VerificationError(
            f"cannot infer a quote basis from non-positive prices "
            f"(broker {broker_price}, data {data_price})"
        )

    def gap_pct(expected: Decimal) -> Decimal:
        return abs(broker_price - expected) / expected * Decimal(100)

    as_instrument = gap_pct(data_price)
    if fx_rate is None:
        if as_instrument <= tolerance_pct:
            return (
                QuoteCurrencyBasis.INSTRUMENT,
                f"broker quote matches the feed within {as_instrument:.1f}%, and no FX "
                "rate was supplied to test the account-currency hypothesis",
            )
        return (
            QuoteCurrencyBasis.UNDETERMINED,
            f"broker quote is {as_instrument:.1f}% from the feed and no FX rate was "
            "supplied, so the account-currency hypothesis cannot be tested",
        )

    if fx_rate <= 0:
        raise VerificationError(f"non-positive FX rate {fx_rate}")

    converted = data_price * fx_rate
    as_account = gap_pct(converted)
    separation = abs(converted - data_price) / data_price * Decimal(100)

    if separation < min_separation_pct:
        return (
            QuoteCurrencyBasis.UNDETERMINED,
            f"the two hypotheses are only {separation:.1f}% apart (rate {fx_rate}), "
            "which is inside the noise. Matching one would not be evidence against "
            "the other.",
        )

    instrument_fits = as_instrument <= tolerance_pct
    account_fits = as_account <= tolerance_pct
    if instrument_fits and not account_fits:
        return (
            QuoteCurrencyBasis.INSTRUMENT,
            f"broker quote matches the instrument's own currency within "
            f"{as_instrument:.1f}% (account-currency hypothesis is off by "
            f"{as_account:.1f}%)",
        )
    if account_fits and not instrument_fits:
        return (
            QuoteCurrencyBasis.ACCOUNT,
            f"broker quote matches the feed converted at {fx_rate} within "
            f"{as_account:.1f}% (instrument-currency hypothesis is off by "
            f"{as_instrument:.1f}%)",
        )
    return (
        QuoteCurrencyBasis.UNDETERMINED,
        f"neither hypothesis fits: instrument-currency off by {as_instrument:.1f}%, "
        f"account-currency off by {as_account:.1f}%. Something other than the currency "
        "is wrong — most likely the mapping itself.",
    )


# --------------------------------------------------------------------------
# The verifier
# --------------------------------------------------------------------------


class SymbolVerifier:
    """Runs the two tiers and owns the entry permission.

    Holds no prices of its own: bars, broker quotes and FX rates are passed in.
    That keeps every state transition testable without a network and makes the
    evidence behind a verification visible at the call site rather than fetched
    invisibly inside it.
    """

    def __init__(
        self,
        symbol_map: SymbolMap,
        *,
        calendar: TradingCalendar | None = None,
        account_currency: str = "GBP",
        quote_basis: QuoteCurrencyBasis = QuoteCurrencyBasis.UNDETERMINED,
        cross_verified_ttl: timedelta = CROSS_VERIFIED_TTL,
    ) -> None:
        self._map = symbol_map
        self._calendar = calendar or TradingCalendar()
        self._account_currency = account_currency.strip().upper()
        self._quote_basis = quote_basis
        self._ttl = cross_verified_ttl

    @property
    def quote_basis(self) -> QuoteCurrencyBasis:
        return self._quote_basis

    def observe_quote_basis(self, basis: QuoteCurrencyBasis, detail: str = "") -> None:
        """Record what the probe or a held position established.

        Only ever tightens: an observation cannot reset a determined basis back
        to `UNDETERMINED`, because a single ambiguous reading is not grounds to
        discard a clear one.
        """
        if basis is QuoteCurrencyBasis.UNDETERMINED:
            return
        self._quote_basis = basis

    # -- tier 1: cross-provider -------------------------------------------

    def verify_by_cross_provider(
        self,
        t212_ticker: str,
        *,
        instrument: Instrument,
        reference: ProviderReference | None = None,
        primary_bar: Bar | None = None,
        secondary_bar: Bar | None = None,
        limit_bps: float,
        commit: bool = True,
    ) -> VerificationResult:
        """The weak tier: corroborate without the broker.

        Needs at least one corroboration and zero contradictions. That rule is
        what stops a single unreachable feed re-creating the deadlock while
        still making a genuine mismatch fatal — and the bound on being wrong is
        one floor-notional position, which the strong tier then re-examines.

        Never clears an existing block. Two feeds agreeing with each other says
        nothing about the broker disagreeing with both, which is what a block
        from the strong tier means.
        """
        mapping = self._require_mapping(t212_ticker)
        if not mapping.data_symbol:
            return VerificationResult(
                t212_ticker=t212_ticker,
                data_symbol="",
                tier=Tier.NONE,
                detail="no data symbol is mapped, so there is nothing to corroborate",
            )

        corroborations: list[Corroboration] = []
        contradictions: list[Contradiction] = []
        notes: list[str] = []

        if reference is not None:
            ref_ok, ref_bad, ref_note = cross_match_reference(instrument, reference)
            corroborations.extend(ref_ok)
            contradictions.extend(ref_bad)
            if ref_note:
                notes.append(ref_note)
        else:
            notes.append("no provider reference data supplied")

        price_ok, price_bad, bps, price_note = compare_provider_bars(
            mapping, primary_bar, secondary_bar, limit_bps=limit_bps
        )
        corroborations.extend(price_ok)
        contradictions.extend(price_bad)
        if price_note:
            notes.append(price_note)

        if contradictions:
            result = VerificationResult(
                t212_ticker=t212_ticker,
                data_symbol=mapping.data_symbol,
                tier=Tier.NONE,
                corroborations=tuple(corroborations),
                contradictions=tuple(contradictions),
                disagreement_bps=bps,
                detail="; ".join(notes),
            )
            if commit:
                self._block(mapping, result, limit_bps=limit_bps)
            return result

        if not corroborations:
            return VerificationResult(
                t212_ticker=t212_ticker,
                data_symbol=mapping.data_symbol,
                tier=Tier.NONE,
                disagreement_bps=bps,
                detail=(
                    "nothing corroborates the mapping: "
                    + ("; ".join(notes) if notes else "no evidence at all")
                ),
            )

        result = VerificationResult(
            t212_ticker=t212_ticker,
            data_symbol=mapping.data_symbol,
            tier=Tier.CROSS_PROVIDER,
            corroborations=tuple(corroborations),
            contradictions=(),
            disagreement_bps=bps,
            detail="; ".join(notes),
        )
        if commit:
            self._promote(mapping, result, limit_bps=limit_bps)
        return result

    # -- tier 2: broker quote ---------------------------------------------

    def verify_by_broker_quote(
        self,
        t212_ticker: str,
        *,
        instrument: Instrument,
        broker_price: Decimal | None,
        data_bar: Bar | None,
        limit_bps: float,
        now: datetime | None = None,
        fx_rate: Decimal | None = None,
        commit: bool = True,
    ) -> VerificationResult:
        """The strong tier: the broker's own quote, available once held.

        Refuses to run rather than produce a wrong answer in three cases, each
        of which would otherwise look like a divergence and block a perfectly
        good symbol:

        * **Market closed.** T212's overnight `currentPrice` is the last
          regular close while the feed's latest bar may be a post-market print.
        * **Extended-hours bar.** Same problem, from the other side.
        * **Undetermined currency basis with differing currencies.** A USD name
          reported in GBP is ~2600bps, which the unit-mismatch detector would
          call a mismapped ticker. Fail-closed: the symbol stays at floor size,
          which is reported and bounded, and exits are never affected.
        """
        mapping = self._require_mapping(t212_ticker)
        moment = now or now_utc()

        if broker_price is None or data_bar is None:
            missing = "broker quote" if broker_price is None else "feed bar"
            return VerificationResult(
                t212_ticker=t212_ticker,
                data_symbol=mapping.data_symbol,
                tier=Tier.NONE,
                detail=(
                    f"no {missing}. The broker only quotes instruments it holds, so "
                    "this tier is unavailable until a position exists — which is what "
                    "the cross-provider tier is for."
                ),
            )

        if not self._calendar.is_open_at(moment):
            return VerificationResult(
                t212_ticker=t212_ticker,
                data_symbol=mapping.data_symbol,
                tier=Tier.NONE,
                detail=(
                    "the market is closed, so the broker's currentPrice is the last "
                    "regular close while the feed may have moved on. Not comparable."
                ),
            )

        if data_bar.session is not Session.REGULAR:
            return VerificationResult(
                t212_ticker=t212_ticker,
                data_symbol=mapping.data_symbol,
                tier=Tier.NONE,
                detail=(
                    f"the feed's bar is an {data_bar.session.value}-session print; an "
                    "earnings-night move of several hundred bps would read as a "
                    "divergence that has nothing to do with the mapping"
                ),
            )

        quoted, basis_note = self._to_broker_basis(
            instrument, data_price=data_bar.close, fx_rate=fx_rate
        )
        if quoted is None:
            return VerificationResult(
                t212_ticker=t212_ticker,
                data_symbol=mapping.data_symbol,
                tier=Tier.NONE,
                detail=basis_note,
            )

        comparison = compare_prices(
            mapping, broker_price=broker_price, data_price=quoted, limit_bps=limit_bps
        )
        if comparison.kind is DisagreementKind.AGREES:
            result = VerificationResult(
                t212_ticker=t212_ticker,
                data_symbol=mapping.data_symbol,
                tier=Tier.BROKER_QUOTE,
                corroborations=(Corroboration.PRICE_AGREES,),
                disagreement_bps=comparison.disagreement_bps,
                detail=f"{comparison.detail}{f'; {basis_note}' if basis_note else ''}",
            )
            if commit:
                self._promote(mapping, result, limit_bps=limit_bps, unblock=True)
            return result

        kind = (
            Contradiction.UNIT_MISMATCH
            if comparison.kind is DisagreementKind.UNIT_MISMATCH
            else Contradiction.PRICE_DIVERGES
        )
        result = VerificationResult(
            t212_ticker=t212_ticker,
            data_symbol=mapping.data_symbol,
            tier=Tier.NONE,
            contradictions=(kind,),
            disagreement_bps=comparison.disagreement_bps,
            detail=comparison.detail,
        )
        if commit:
            self._block(mapping, result, limit_bps=limit_bps)
        return result

    def _to_broker_basis(
        self, instrument: Instrument, *, data_price: Decimal, fx_rate: Decimal | None
    ) -> tuple[Decimal | None, str]:
        """Put the feed's price on whatever basis the broker quotes in."""
        instrument_ccy = (instrument.currency_code or "").strip().upper()
        if not instrument_ccy or instrument_ccy == self._account_currency:
            # No ambiguity to resolve: both hypotheses predict the same number.
            return data_price, ""

        if self._quote_basis is QuoteCurrencyBasis.INSTRUMENT:
            return data_price, f"broker quotes in {instrument_ccy} (observed)"

        if self._quote_basis is QuoteCurrencyBasis.ACCOUNT:
            if fx_rate is None:
                return None, (
                    f"the broker quotes in {self._account_currency} but no "
                    f"{instrument_ccy}/{self._account_currency} rate was supplied, so "
                    "the feed's price cannot be put on the same basis"
                )
            return data_price * fx_rate, (
                f"feed price converted to {self._account_currency} at {fx_rate}"
            )

        return None, (
            f"whether the broker quotes {instrument.ticker} in {instrument_ccy} or in "
            f"{self._account_currency} has not been established, and the two differ by "
            "far more than the disagreement limit — a wrong choice would read as a "
            "mismapped ticker. Run `tb broker probe` against a held position, or call "
            "`infer_quote_basis`, then re-verify. The symbol stays at floor size "
            "meanwhile; exits are unaffected."
        )

    # -- the gate ----------------------------------------------------------

    def entry_permission(
        self,
        t212_ticker: str,
        *,
        floor_notional: Decimal,
        now: datetime | None = None,
    ) -> EntryPermission:
        """Whether an entry may be placed, and the largest notional allowed.

        The gate the order path calls. `SymbolMap.may_enter` answers the coarser
        question and its bool reads as full permission; the cap lives here.
        """
        moment = now or now_utc()
        mapping = self._map.get(t212_ticker)
        if mapping is None:
            return EntryPermission(
                allowed=False,
                t212_ticker=t212_ticker,
                tier=Tier.NONE,
                max_notional=None,
                reasons=(
                    f"{t212_ticker} has no symbol mapping, so it cannot be priced or "
                    "sized. Run `tb symbols audit`.",
                ),
            )

        reasons: list[str] = []
        if mapping.blocked:
            reasons.append(f"blocked: {mapping.blocked_reason}")
        if not mapping.confidence.tradable:
            reasons.append(
                f"{mapping.confidence.value}, not verified by either tier ({mapping.derivation})"
            )

        tier = (
            Tier.BROKER_QUOTE
            if mapping.confidence is Confidence.VERIFIED
            else Tier.CROSS_PROVIDER
            if mapping.confidence is Confidence.CROSS_VERIFIED
            else Tier.NONE
        )

        if tier is Tier.CROSS_PROVIDER and self._weak_evidence_expired(mapping, moment):
            reasons.append(
                f"cross-provider evidence is older than {self._ttl}. It was one price "
                "comparison at one instant, and a symbol change or unannounced action "
                "since then would have invalidated it silently. Re-run the check."
            )

        if reasons:
            return EntryPermission(
                allowed=False,
                t212_ticker=t212_ticker,
                tier=tier,
                max_notional=None,
                reasons=tuple(reasons),
            )

        if tier is Tier.CROSS_PROVIDER:
            return EntryPermission(
                allowed=True,
                t212_ticker=t212_ticker,
                tier=tier,
                max_notional=floor_notional,
                reasons=(
                    f"cross-provider verified: capped at the floor notional "
                    f"{floor_notional}. The broker has not confirmed this mapping and "
                    "cannot until a position exists, so this entry is the evidence-"
                    "gathering trade rather than a sized position.",
                ),
            )

        return EntryPermission(
            allowed=True,
            t212_ticker=t212_ticker,
            tier=tier,
            max_notional=None,
            reasons=(),
        )

    def _weak_evidence_expired(self, mapping: SymbolMapping, now: datetime) -> bool:
        if mapping.last_checked_at is None:
            # No timestamp means we cannot show the evidence is fresh, which is
            # the same as it not being fresh.
            return True
        try:
            checked = from_iso(mapping.last_checked_at)
        except ValueError:
            return True
        return now - checked > self._ttl

    # -- persistence -------------------------------------------------------

    def _require_mapping(self, t212_ticker: str) -> SymbolMapping:
        mapping = self._map.get(t212_ticker)
        if mapping is None:
            raise VerificationError(
                f"{t212_ticker} has no symbol mapping; run `derive_all` before verifying"
            )
        return mapping

    def _promote(
        self,
        mapping: SymbolMapping,
        result: VerificationResult,
        *,
        limit_bps: float,
        unblock: bool = False,
    ) -> None:
        """Record a verification at the tier the evidence supports.

        `unblock` is False for the weak tier and True for the strong one. That
        asymmetry is load-bearing: a block from the strong tier records that the
        *broker* disagreed, and two feeds agreeing with each other is not an
        answer to that.
        """
        if mapping.blocked and not unblock:
            return
        comparison = PriceComparison(
            t212_ticker=result.t212_ticker,
            data_symbol=result.data_symbol,
            broker_price=None,
            data_price=None,
            disagreement_bps=result.disagreement_bps,
            kind=DisagreementKind.AGREES,
            detail=f"{result.tier.value}: {result.detail}",
        )
        self._map.verify(comparison, limit_bps=limit_bps, confidence=result.tier.confidence)

    def _block(
        self, mapping: SymbolMapping, result: VerificationResult, *, limit_bps: float
    ) -> None:
        kinds = ", ".join(c.value for c in result.contradictions)
        comparison = PriceComparison(
            t212_ticker=result.t212_ticker,
            data_symbol=result.data_symbol,
            broker_price=None,
            data_price=None,
            disagreement_bps=result.disagreement_bps,
            kind=DisagreementKind.DIVERGED,
            detail=f"{kinds}: {result.detail}",
        )
        self._map.block(comparison, limit_bps=limit_bps)

    # -- reporting ---------------------------------------------------------

    def tier_summary(self) -> dict[str, int]:
        """Counts per tier, for `tb symbols audit`.

        `n_enterable` is the number that matters: while it is zero the bot
        cannot open its first position, which is the condition this module
        exists to make visible rather than mysterious.
        """
        rows = self._map.all()
        return {
            "n_broker_verified": sum(1 for m in rows if m.confidence is Confidence.VERIFIED),
            "n_cross_verified": sum(1 for m in rows if m.confidence is Confidence.CROSS_VERIFIED),
            "n_derived": sum(1 for m in rows if m.confidence is Confidence.DERIVED),
            "n_blocked": sum(1 for m in rows if m.blocked),
            "n_enterable": sum(1 for m in rows if m.may_enter),
            "n_full_size": sum(1 for m in rows if m.permits_full_size),
        }
