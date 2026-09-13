"""Factor algebra over corporate actions. Pure, exact, no I/O.

The store holds **facts** — a split happened on this date, at this ratio, and we
learned about it at this moment — and this module derives **factors** from them
on demand. Never the reverse. A materialised adjustment column is how a table
ends up containing tomorrow's split: the column gets recomputed by a backfill
job that knows about actions the backtest's as-of instant could not have known,
and every price before the ex-date silently becomes information from the future.
No schema check catches it, because the number looks perfectly ordinary.

Four things carry the correctness here.

**Two filters, never conflated, with different names and different types.**
As-of visibility asks `known_at <= as_of` and takes a `datetime`. Price-path
reconstruction asks `effective_date > t` and takes a `date`. Conflating them is
the classic bug in this area, so they cannot even be passed to each other's
parameters without mypy objecting.

**Exact rationals.** Ratios are `Fraction`, not float and not rounded Decimal.
A 3-for-1 stored as `0.3333333` drifts across a twenty-year product of factors,
`canonical_json` faithfully hashes the drift, and two runs that should agree do
not. Decimal appears only at the boundary, in `apply_factor`, with the rounding
stated and the precision pinned locally rather than inherited from a global
context some other library may have changed.

**The volume factor is the exact inverse of the price factor.** Not
approximately: `1/Fraction` is exact. Getting this wrong step-changes ADV at
every split, and M3's slippage model would inherit a 4x liquidity error on any
name that has split.

**Three named series, never a bare "price."** `RAW` is the only series allowed
near execution — staleness checks, cross-venue comparison, stop placement, tick
rounding, minimum quantity. `SPLIT_ADJUSTED` is for price-shape features.
`TOTAL_RETURN` is for P&L and Sharpe only, and is labelled gross of withholding
and FX. `require_raw` turns that rule into a raise rather than a comment.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from datetime import date, datetime
from decimal import ROUND_HALF_EVEN, Decimal, localcontext
from enum import StrEnum
from fractions import Fraction

from tb.data.provider import DataError

# Enough precision that a factor product over decades is exact before the
# single explicit quantize at the end. Pinned locally because the decimal
# context is process-global mutable state: a library that lowers `prec`
# elsewhere would otherwise change prices here.
_WORKING_PRECISION = 60


class FactorError(DataError):
    """A factor could not be computed from the actions given."""


class SeriesMisuseError(DataError):
    """An adjusted price was used somewhere only a raw price is admissible."""


class ActionType(StrEnum):
    SPLIT = "split"
    CASH_DIVIDEND = "cash_dividend"
    # The two that matter most for survivorship: a ticker that changed hands
    # and a company that stopped existing are exactly the rows a
    # today's-universe backfill silently omits.
    SYMBOL_CHANGE = "symbol_change"
    DELISTING = "delisting"


class Series(StrEnum):
    """Which price series a number belongs to.

    Carried alongside every adjusted value because the three are not
    interchangeable and the failure is silent: a split-adjusted price compared
    against the broker's quote disagrees by the whole split ratio, and the
    cross-venue check would read that as a mismapped ticker.
    """

    RAW = "raw"
    SPLIT_ADJUSTED = "split_adjusted"
    TOTAL_RETURN = "total_return"

    @property
    def admissible_for_execution(self) -> bool:
        """Whether a value on this series may touch the order path."""
        return self is Series.RAW


@dataclass(frozen=True, slots=True)
class CorporateAction:
    """A dated, revisable fact about an instrument.

    `known_at_utc` is when *we* learned it, which is the only thing an as-of
    query may filter on. `declared_date` is when it became public and is
    frequently unknown on free data — stored as None rather than imputed,
    because a guessed declaration date is how a factor table acquires
    information from the future.
    """

    action_id: str
    instrument_uid: str
    action_type: ActionType
    effective_date: date
    known_at_utc: datetime
    source_provider: str
    ratio_num: int | None = None
    ratio_den: int | None = None
    gross_amount: Decimal | None = None
    currency: str | None = None
    new_symbol: str | None = None
    declared_date: date | None = None
    superseded_by: str | None = None
    inferred_from_price_jump: bool = False

    def __post_init__(self) -> None:
        if self.known_at_utc.tzinfo is None:
            raise DataError(f"{self.action_id}: known_at_utc is a naive datetime")
        if self.action_type is ActionType.SPLIT:
            if not self.ratio_num or not self.ratio_den:
                raise DataError(f"{self.action_id}: a split needs both ratio_num and ratio_den")
            if self.ratio_num <= 0 or self.ratio_den <= 0:
                raise DataError(
                    f"{self.action_id}: non-positive split ratio {self.ratio_num}/{self.ratio_den}"
                )
        if self.action_type is ActionType.CASH_DIVIDEND:
            if self.gross_amount is None:
                raise DataError(f"{self.action_id}: a cash dividend needs a gross_amount")
            if self.gross_amount <= 0:
                raise DataError(
                    f"{self.action_id}: non-positive dividend {self.gross_amount}. A zero or "
                    "negative distribution is a data error, not an event."
                )

    @property
    def split_ratio(self) -> Fraction:
        """New shares per old share, exactly.

        A 4-for-1 is `Fraction(4, 1)`; a 1-for-10 reverse split is
        `Fraction(1, 10)` and not `0.1`.
        """
        if self.action_type is not ActionType.SPLIT:
            raise FactorError(f"{self.action_id} is a {self.action_type.value}, not a split")
        assert self.ratio_num is not None and self.ratio_den is not None  # __post_init__
        return Fraction(self.ratio_num, self.ratio_den)

    @property
    def identity(self) -> tuple[str, str, str]:
        """What makes two rows the same action, ignoring vintage."""
        return (self.instrument_uid, self.action_type.value, self.effective_date.isoformat())


# --------------------------------------------------------------------------
# The two filters. Different names, different parameter types.
# --------------------------------------------------------------------------


def known_by(actions: Iterable[CorporateAction], as_of: datetime) -> tuple[CorporateAction, ...]:
    """**As-of visibility.** Actions we had learned of by `as_of`.

    Takes a `datetime` because knowledge time is an instant. This is the filter
    that keeps a backtest from adjusting a price for a split that had not been
    announced yet.
    """
    if as_of.tzinfo is None:
        raise DataError("as_of must be timezone-aware")
    return tuple(action for action in actions if action.known_at_utc <= as_of)


def effective_after(
    actions: Iterable[CorporateAction], moment: date
) -> tuple[CorporateAction, ...]:
    """**Price-path reconstruction.** Actions taking effect strictly after `moment`.

    Takes a `date` because an ex-date is a session, not an instant. Strictly
    after: a bar dated on the ex-date is already quoted post-split, so adjusting
    it would apply the ratio twice.
    """
    return tuple(action for action in actions if action.effective_date > moment)


def effective_in(
    actions: Iterable[CorporateAction], *, after: date, through: date
) -> tuple[CorporateAction, ...]:
    """Actions in the half-open span `(after, through]`.

    For returns over a span rather than a level at a point — the adjustment
    that belongs to a return from `after` to `through`.
    """
    if through < after:
        raise FactorError(f"span runs backwards: ({after}, {through}]")
    return tuple(action for action in actions if after < action.effective_date <= through)


def latest_vintages(
    actions: Iterable[CorporateAction], as_of: datetime
) -> tuple[CorporateAction, ...]:
    """Resolve restatements as of an instant.

    A vendor that corrects a ratio produces a second row for the same
    `(instrument, type, effective_date)`. Only the newest row *known by*
    `as_of` counts — so a backtest run over an earlier instant still sees the
    ratio that was believed at the time, which is the whole point of keeping
    both rows.
    """
    newest: dict[tuple[str, str, str], CorporateAction] = {}
    for action in known_by(actions, as_of):
        existing = newest.get(action.identity)
        if existing is None or action.known_at_utc >= existing.known_at_utc:
            newest[action.identity] = action
    return tuple(sorted(newest.values(), key=lambda a: (a.effective_date, a.action_type.value)))


# --------------------------------------------------------------------------
# Factors
# --------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class Factor:
    """A multiplicative adjustment, exact, with its own completeness.

    `complete` is not decoration. A total-return factor needs a reference close
    per dividend, and a missing one means the dividend was *left out* —
    understating total return for exactly the names that pay dividends. That
    bias is downward, which is the direction nobody investigates, so it is
    recorded on the value rather than logged.
    """

    value: Fraction
    series: Series
    n_actions: int = 0
    complete: bool = True
    missing: tuple[str, ...] = field(default_factory=tuple)

    def __post_init__(self) -> None:
        if self.value <= 0:
            raise FactorError(
                f"non-positive {self.series.value} factor {self.value}. A factor multiplies a "
                "price, so zero or negative would invert or erase it."
            )

    @property
    def is_identity(self) -> bool:
        return self.value == 1

    @property
    def inverse(self) -> Fraction:
        return 1 / self.value


@dataclass(frozen=True, slots=True)
class AdjustedPrice:
    """A price that knows which series it is on.

    The type exists so that "never a bare price" is enforceable. A function
    handed one of these cannot mistake a split-adjusted value for something it
    can send to the broker.
    """

    value: Decimal
    series: Series
    factor: Fraction
    as_of: datetime
    complete: bool = True

    @property
    def admissible_for_execution(self) -> bool:
        return self.series.admissible_for_execution


def price_factor(actions: Iterable[CorporateAction], *, at: date, as_of: datetime) -> Factor:
    """Multiply a raw price dated `at` by this to reach the current split scale.

    Only splits count, and only those effective after `at` and known by
    `as_of`. A 4-for-1 after the bar divides the price by four, so the factor
    contributes `ratio_den / ratio_num`.
    """
    relevant = [
        action
        for action in effective_after(latest_vintages(actions, as_of), at)
        if action.action_type is ActionType.SPLIT
    ]
    value = Fraction(1)
    for action in relevant:
        value /= action.split_ratio
    return Factor(value=value, series=Series.SPLIT_ADJUSTED, n_actions=len(relevant))


def volume_factor(actions: Iterable[CorporateAction], *, at: date, as_of: datetime) -> Factor:
    """The exact inverse of the price factor.

    Exact, not approximate: a 4-for-1 quadruples the share count precisely as
    it quarters the price, so dollar volume is invariant across the split. That
    invariance is asserted in the tests, because without it ADV step-changes at
    every split and everything sized off liquidity inherits the error.
    """
    prices = price_factor(actions, at=at, as_of=as_of)
    return Factor(
        value=prices.inverse,
        series=Series.SPLIT_ADJUSTED,
        n_actions=prices.n_actions,
        complete=prices.complete,
        missing=prices.missing,
    )


def total_return_factor(
    actions: Iterable[CorporateAction],
    *,
    at: date,
    as_of: datetime,
    reference_close: Mapping[str, Decimal],
) -> Factor:
    """Splits *and* dividends, for P&L and Sharpe only.

    A dividend's adjustment is `(C - D) / C`, where `C` is the close on the last
    session before the ex-date. That close is a price, not an action, so it has
    to be supplied — `reference_close` maps `action_id` to it. Keeping the
    dependency explicit is what lets this module stay pure; deriving it from a
    store inside here would make the factor depend on whatever the store
    currently holds rather than on what was knowable at `as_of`.

    A dividend whose reference close is missing, or which exceeds that close, is
    **excluded and named** in `missing` with `complete=False`. Excluding it
    understates total return; asserting a factor anyway could invert a price.

    The result is gross of withholding tax and of FX. Both are real and both
    are the cost model's business, not this function's.
    """
    live = latest_vintages(actions, as_of)
    splits = price_factor(live, at=at, as_of=as_of)
    value = splits.value
    missing: list[str] = []
    counted = splits.n_actions

    for action in effective_after(live, at):
        if action.action_type is not ActionType.CASH_DIVIDEND:
            continue
        close = reference_close.get(action.action_id)
        assert action.gross_amount is not None  # __post_init__
        if close is None or close <= 0 or action.gross_amount >= close:
            missing.append(action.action_id)
            continue
        value *= Fraction(close - action.gross_amount) / Fraction(close)
        counted += 1

    return Factor(
        value=value,
        series=Series.TOTAL_RETURN,
        n_actions=counted,
        complete=not missing,
        missing=tuple(missing),
    )


def factor_for(
    series: Series,
    actions: Iterable[CorporateAction],
    *,
    at: date,
    as_of: datetime,
    reference_close: Mapping[str, Decimal] | None = None,
) -> Factor:
    """Dispatch on the series, so a caller names what it wants."""
    if series is Series.RAW:
        return Factor(value=Fraction(1), series=Series.RAW)
    if series is Series.SPLIT_ADJUSTED:
        return price_factor(actions, at=at, as_of=as_of)
    return total_return_factor(actions, at=at, as_of=as_of, reference_close=reference_close or {})


def apply_factor(price: Decimal, factor: Factor, *, as_of: datetime, places: int) -> AdjustedPrice:
    """Apply a factor and quantize once, explicitly.

    The arithmetic runs in a local context at `_WORKING_PRECISION` so the result
    does not depend on whatever the ambient decimal context happens to be, and
    rounds half-to-even at the end — stated here rather than inherited, because
    a rounding mode that varies between processes makes two runs of the same
    backtest disagree in the last digit and every hash over them differ.
    """
    if not isinstance(price, Decimal):
        raise FactorError(f"prices must be Decimal, got {type(price).__name__}")
    if not price.is_finite():
        raise FactorError(f"non-finite price {price!r}")

    with localcontext() as ctx:
        ctx.prec = _WORKING_PRECISION
        scaled = price * Decimal(factor.value.numerator) / Decimal(factor.value.denominator)
        quantum = Decimal(1).scaleb(-places)
        adjusted = scaled.quantize(quantum, rounding=ROUND_HALF_EVEN)

    return AdjustedPrice(
        value=adjusted,
        series=factor.series,
        factor=factor.value,
        as_of=as_of,
        complete=factor.complete,
    )


def apply_volume_factor(volume: int, factor: Factor) -> int:
    """Restate a share count onto the current split scale.

    Rounded to a whole share, because a fractional share count is not a thing
    the tape can report. The rounding is the reason the *dollar* volume
    invariant is asserted on the factors rather than on the rounded integers.
    """
    exact = Fraction(volume) * factor.value
    return round(exact)


def require_raw(price: AdjustedPrice) -> Decimal:
    """Unwrap a price, refusing anything but the raw series.

    The enforcement point for the rule that only raw prices go near execution.
    A split-adjusted price sent to a stop-placement or cross-venue-comparison
    path disagrees with the broker by the entire split ratio, and the
    disagreement check would report it as a mismapped ticker — a data problem
    misdiagnosed as an identity problem, which is the expensive direction.
    """
    if not price.admissible_for_execution:
        raise SeriesMisuseError(
            f"a {price.series.value} price reached a path that requires the raw series. "
            "Staleness checks, cross-venue comparison, stop placement, tick rounding and "
            "minimum-quantity checks all compare against what the venue is quoting now, "
            "which is the unadjusted price."
        )
    return price.value


# --------------------------------------------------------------------------
# Residual detection: actions nobody told us about
# --------------------------------------------------------------------------

# A jump has to be big before any ratio is evidence of anything. The smallest
# ratio in the candidate set below is 3-for-2, which moves the price 33%, so
# nothing under 30% can be a split signature — and a 20% overnight gap after
# bad earnings is entirely ordinary on a mid-cap. Setting this lower would flag
# every earnings miss in the universe.
MIN_SUSPICIOUS_MOVE_PCT = Decimal("30")
MAX_RATIO_TERM = 20
# How close the fit must be, in percent of the implied ratio. A split's ratio is
# exact, but the gap also contains one session of genuine price movement, so a
# few percent of slack is needed. More than this and unrelated moves start
# landing on candidates.
RATIO_TOLERANCE_PCT = Decimal("5")


def _candidate_ratios(max_term: int) -> tuple[Fraction, ...]:
    """The ratios a real split actually uses.

    A fixed candidate set rather than `Fraction.limit_denominator`. That
    function returns the *best rational approximation*, and with denominators up
    to twenty it fits almost any number: a 37% drop comes back as a convincing
    19-for-12 with a 0.02% error. A detector that can explain anything explains
    nothing, so the search is restricted to shapes that are issued in practice —
    `k`-for-1, 1-for-`k`, and 3-for-2 either way.

    Deliberately excluded: 5-for-4 and its inverse. A 5-for-4 split moves the
    price 20%, which collides with an ordinary earnings gap, and the collision
    is common enough that including it would block entries across the universe
    every reporting season. A genuine 5-for-4 is instead caught by the audit's
    unexplained-jump check, which reports the move without asserting a ratio.
    """
    candidates: list[Fraction] = [Fraction(3, 2), Fraction(2, 3)]
    for term in range(2, max_term + 1):
        candidates.append(Fraction(term, 1))
        candidates.append(Fraction(1, term))
    return tuple(candidates)


@dataclass(frozen=True, slots=True)
class SplitSuspicion:
    """An unexplained price jump that a small-integer ratio accounts for.

    This is what catches a vendor back-adjusting its cache without ever
    reporting an action — Yahoo's actual behaviour. The response is to block
    new entries in the symbol, not to invent a factor: acting on an inferred
    ratio would be guessing at the size of a correction to real money.
    """

    instrument_uid: str
    effective_date: date
    prev_close: Decimal
    next_open: Decimal
    implied_ratio: Fraction
    move_pct: Decimal
    fit_error_pct: Decimal

    @property
    def description(self) -> str:
        return (
            f"{self.instrument_uid}: {self.move_pct:.1f}% gap into {self.effective_date} "
            f"fits a {self.implied_ratio.numerator}-for-{self.implied_ratio.denominator} "
            f"split within {self.fit_error_pct:.2f}%, but no action row explains it"
        )


def detect_unexplained_split(
    *,
    instrument_uid: str,
    prev_close: Decimal,
    next_open: Decimal,
    effective_date: date,
    known_actions: Iterable[CorporateAction] = (),
    min_move_pct: Decimal = MIN_SUSPICIOUS_MOVE_PCT,
    max_ratio_term: int = MAX_RATIO_TERM,
    tolerance_pct: Decimal = RATIO_TOLERANCE_PCT,
) -> SplitSuspicion | None:
    """Look for a split that no action row accounts for.

    Runs on the raw series only — on a split-adjusted series the jump has
    already been removed, so this would never fire, which is the quiet way a
    detector ends up doing nothing.
    """
    if prev_close <= 0 or next_open <= 0:
        raise FactorError(
            f"{instrument_uid}: cannot test a gap into {effective_date} with a "
            f"non-positive price ({prev_close} -> {next_open})"
        )

    if any(
        action.effective_date == effective_date and action.action_type is ActionType.SPLIT
        for action in known_actions
    ):
        return None

    move_pct = abs(next_open - prev_close) / prev_close * Decimal(100)
    if move_pct < min_move_pct:
        return None

    implied = Fraction(prev_close) / Fraction(next_open)
    best: Fraction | None = None
    best_error = Fraction(tolerance_pct)
    for candidate in _candidate_ratios(max_ratio_term):
        error = abs(candidate - implied) / implied * 100
        if error <= best_error:
            best, best_error = candidate, error
    if best is None:
        return None

    return SplitSuspicion(
        instrument_uid=instrument_uid,
        effective_date=effective_date,
        prev_close=prev_close,
        next_open=next_open,
        implied_ratio=best,
        move_pct=move_pct,
        fit_error_pct=Decimal(best_error.numerator) / Decimal(best_error.denominator),
    )
