"""Bars in, features out. Exactly one implementation, used everywhere.

The "exactly one" is the whole design. Backtest, paper and live must compute
features with the *same object*, not with two implementations that agree today.
Two pipelines drift, and the drift is never neutral: it is discovered when live
underperforms its backtest, which is months after the divergence was
introduced and long after anyone remembers why.

So there is no `BacktestFeaturePipeline`. There is `FeaturePipeline`, it takes
a `BarWindow` — the same type the live loop will hold — and the backtester gets
no privileged access to anything. A test asserts the two call paths produce
identical snapshot hashes.

Four properties carry the correctness:

**Absence is `UNKNOWN`, and it propagates.** A 200-day moving average over 40
days of history is not a shorter average, it is a different and wrong number.
Every feature returns `UNKNOWN` rather than padding, forward-filling, or
computing over what happens to be there. `UNKNOWN` raises on arithmetic and on
truth-testing, so a strategy cannot silently treat it as zero.

**Features read `split_adjusted` prices, never raw.** Price *shape* is what a
feature is about, and a raw series has a discontinuity at every split that is
not a return. The adjustment is applied through `tb.data.adjustments` using
only actions knowable at the decision time — which is the difference between an
adjusted series and a lookahead channel.

**Every snapshot is hashed.** `feature_snapshot_hash` goes into the decision
record, so "why did it buy that" is answerable against the exact inputs months
later. Hashed over canonical JSON with Decimals as exact strings, never over a
float, because a float round trip changes the hash.

**No feature may see beyond its window.** The window physically contains
nothing later than `as_of` (see `tb.data.asof`), so this is structural rather
than enforced here. What *is* enforced here is that a feature declares its
lookback, and asking for a window shorter than the declared lookback returns
`UNKNOWN` instead of a number computed from too little.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime
from decimal import Decimal, localcontext
from itertools import pairwise
from typing import Protocol, TypeAlias

from tb.core.canonical import hash_payload
from tb.core.errors import TbError
from tb.data.adjustments import CorporateAction, Series, price_factor
from tb.data.asof import UNKNOWN, BarWindow, Unknown
from tb.data.provider import Bar

# What a feature yields. `Unknown` is not an error case to be handled once at
# the edge — it is a legitimate, common value that must survive to the caller.
FeatureValue: TypeAlias = "Decimal | Unknown"

_WORKING_PRECISION = 40
# Features are rounded to this many places before hashing. Without it, two
# runs that differ only in Decimal context would produce different snapshot
# hashes for identical inputs, which would make every snapshot hash useless
# for the thing it exists to do.
FEATURE_PLACES = 10


class FeatureError(TbError):
    """A feature could not be computed for a reason that is not mere absence.

    Distinct from `UNKNOWN`: absence of data is expected and returns a value,
    while a malformed request (a negative lookback, an unknown feature name) is
    a bug and raises.
    """


@dataclass(frozen=True, slots=True)
class FeatureSpec:
    """One named feature and the lookback it needs.

    The lookback is declared rather than inferred so the pipeline can refuse a
    short window *before* computing anything. A feature that discovers
    mid-computation that it has too little data has already had the chance to
    produce a number.
    """

    name: str
    lookback: int
    compute: Callable[[Sequence[Decimal]], FeatureValue]

    def __post_init__(self) -> None:
        if self.lookback < 1:
            raise FeatureError(f"{self.name}: lookback must be at least 1, got {self.lookback}")


@dataclass(frozen=True, slots=True)
class _Close:
    """An adjusted close that knows whether its scale is trustworthy.

    Carried alongside the number rather than tracked as a parallel index set,
    because the two are sliced together by every lookback and a slice applied to
    one but not the other is a silent misalignment.
    """

    value: Decimal
    unadjustable: bool


@dataclass(frozen=True, slots=True)
class FeatureSnapshot:
    """The feature vector at one decision time, and its hash.

    Frozen and hashed because this is the evidence for a decision. M5's replay
    (`tb replay --fill <id>`) reconstructs a trade from the ledger, and without
    a hash over the exact inputs, "the strategy saw these features" is an
    assertion rather than a check.
    """

    as_of: datetime
    instrument_uid: str
    values: Mapping[str, FeatureValue]
    snapshot_hash: str
    n_bars_seen: int
    series: Series = Series.SPLIT_ADJUSTED
    # Action ids whose factor could not be applied over this window — in
    # practice splits the residual detector only *inferred*, which are recorded
    # but never used as factors. Diagnostic, and so not hashed, for the same
    # reason `n_bars_seen` is not: the consequence is already in `values`, where
    # every feature whose lookback crosses the discontinuity is `UNKNOWN`. It is
    # here so that "why is everything unknown" has an answer at the call site
    # rather than in a log.
    unadjustable_actions: tuple[str, ...] = ()

    def get(self, name: str) -> FeatureValue:
        try:
            return self.values[name]
        except KeyError as exc:
            raise FeatureError(
                f"no feature named {name!r} in this snapshot; it has "
                f"{sorted(self.values)}. A strategy referring to a feature the "
                "pipeline does not compute is a malformed spec, not a missing value."
            ) from exc

    @property
    def complete(self) -> bool:
        """Whether every feature resolved to a number.

        Checked explicitly rather than by truth-testing the values, because
        `UNKNOWN.__bool__` raises — which is the point of it.
        """
        return all(value is not UNKNOWN for value in self.values.values())

    def unknown_features(self) -> tuple[str, ...]:
        return tuple(sorted(name for name, v in self.values.items() if v is UNKNOWN))


# --------------------------------------------------------------------------
# The feature library
# --------------------------------------------------------------------------
#
# Pure functions over a sequence of closes, oldest first. Kept separate from
# the pipeline so each is testable against a hand-built list, and kept pure so
# the DSL can reference them by name without being able to reach a clock, a
# filesystem, or a network.


def _mean(values: Sequence[Decimal]) -> Decimal:
    with localcontext() as ctx:
        ctx.prec = _WORKING_PRECISION
        return sum(values, Decimal(0)) / Decimal(len(values))


def sma(values: Sequence[Decimal]) -> FeatureValue:
    """Simple moving average over the whole window."""
    if not values:
        return UNKNOWN
    return _mean(values)


def _return_pct(values: Sequence[Decimal]) -> FeatureValue:
    """Total return across the window, as a percentage.

    Refuses a zero or negative start rather than dividing by it: a
    non-positive price is not a price, and a division that produced `inf` here
    would propagate into a signal as a very large number rather than as an
    error.
    """
    if len(values) < 2:
        return UNKNOWN
    first, last = values[0], values[-1]
    if first <= 0:
        return UNKNOWN
    with localcontext() as ctx:
        ctx.prec = _WORKING_PRECISION
        return (last - first) / first * Decimal(100)


def _stdev(values: Sequence[Decimal]) -> FeatureValue:
    """Sample standard deviation of simple returns, as a percentage.

    Sample rather than population (n-1): with a 20-bar window the difference is
    ~2.6%, and volatility feeds position sizing.
    """
    if len(values) < 3:
        return UNKNOWN
    with localcontext() as ctx:
        ctx.prec = _WORKING_PRECISION
        returns: list[Decimal] = []
        for previous, current in pairwise(values):
            if previous <= 0:
                return UNKNOWN
            returns.append((current - previous) / previous)
        avg = _mean(returns)
        variance = sum(((r - avg) ** 2 for r in returns), Decimal(0)) / Decimal(len(returns) - 1)
        return variance.sqrt() * Decimal(100)


def _zscore(values: Sequence[Decimal]) -> FeatureValue:
    """How far the last close sits from the window mean, in window sigmas.

    Returns `UNKNOWN` at zero dispersion rather than dividing. A flat window
    has no scale to measure against, and a large number here would read as a
    strong signal when it actually means "nothing has moved".
    """
    if len(values) < 3:
        return UNKNOWN
    with localcontext() as ctx:
        ctx.prec = _WORKING_PRECISION
        avg = _mean(values)
        variance = sum(((v - avg) ** 2 for v in values), Decimal(0)) / Decimal(len(values) - 1)
        if variance <= 0:
            return UNKNOWN
        return (values[-1] - avg) / variance.sqrt()


def _max_drawdown_pct(values: Sequence[Decimal]) -> FeatureValue:
    """Worst peak-to-trough fall within the window, as a positive percentage."""
    if len(values) < 2:
        return UNKNOWN
    with localcontext() as ctx:
        ctx.prec = _WORKING_PRECISION
        peak = values[0]
        worst = Decimal(0)
        for value in values:
            if value > peak:
                peak = value
            if peak > 0:
                fall = (peak - value) / peak * Decimal(100)
                worst = max(worst, fall)
        return worst


def _last(values: Sequence[Decimal]) -> FeatureValue:
    return values[-1] if values else UNKNOWN


# The named library. A DSL spec may reference these names and nothing else,
# which is what makes "no eval of generated output" implementable: a generated
# spec selects from this table rather than supplying code.
FEATURE_LIBRARY: dict[str, Callable[[Sequence[Decimal]], FeatureValue]] = {
    "last": _last,
    "sma": sma,
    "return_pct": _return_pct,
    "stdev_pct": _stdev,
    "zscore": _zscore,
    "max_drawdown_pct": _max_drawdown_pct,
}


def make_spec(kind: str, lookback: int, *, name: str | None = None) -> FeatureSpec:
    """Build a spec from a library name, refusing anything not in the library."""
    try:
        compute = FEATURE_LIBRARY[kind]
    except KeyError as exc:
        raise FeatureError(
            f"unknown feature kind {kind!r}; the library is {sorted(FEATURE_LIBRARY)}. "
            "Features are selected from a fixed table rather than supplied as code, "
            "so a generated spec cannot introduce a new one."
        ) from exc
    return FeatureSpec(name=name or f"{kind}_{lookback}", lookback=lookback, compute=compute)


class Scorer(Protocol):
    """A value computed from other features at the same instant: a model's score.

    Evaluated after every feature, from those features' values alone — no bars
    and no window — so a scorer sees exactly what a strategy reading the same
    features sees, and nothing a strategy could not. It is the only way a
    model's output enters a snapshot, which is what keeps models inside the one
    pipeline: the backtest, the loop and `tb replay` all get the score from the
    same `compute` call, hashed with everything else in the snapshot.

    `as_of` is the decision time, which a strategy also knows. A recorded model
    ignores it; the trainer's walk-forward scorer uses it to pick the model
    fitted before that instant's fold, and has none to offer before the first
    fold — `None`, which the snapshot reads as `UNKNOWN`.
    """

    @property
    def name(self) -> str: ...

    @property
    def inputs(self) -> tuple[str, ...]:
        """Feature names, in the order `score` reads them."""
        ...

    def score(self, row: Sequence[float], *, as_of: datetime) -> float | None: ...


def pipeline_for(
    requests: Iterable[tuple[str, int]], *, scorers: Sequence[Scorer] = ()
) -> FeaturePipeline:
    """The pipeline computing `(kind, lookback)` features, under their canonical names.

    `kind_lookback`, as `make_spec` names them and as the DSL's feature keys
    read them. Strategy specs and models both declare their inputs this way, so
    the name a model was fitted under and the name the loop computes for it are
    the same string by construction rather than by two callers agreeing.
    """
    return FeaturePipeline(
        specs=tuple(make_spec(kind, lookback) for kind, lookback in requests),
        scorers=tuple(scorers),
    )


# --------------------------------------------------------------------------
# The pipeline
# --------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class FeaturePipeline:
    """The one pipeline. Same object in backtest, paper and live.

    Holds no clock, no connection and no mutable state, so it cannot behave
    differently in one caller than in another — which is the property that
    makes "identical in backtest and live" checkable rather than aspirational.
    """

    specs: tuple[FeatureSpec, ...]
    series: Series = Series.SPLIT_ADJUSTED
    # Computed after `specs`, from their values. Scorers read features and
    # never each other: a score of a score is a model nobody trained.
    scorers: tuple[Scorer, ...] = ()

    @property
    def max_lookback(self) -> int:
        return max((spec.lookback for spec in self.specs), default=0)

    @property
    def names(self) -> tuple[str, ...]:
        """Every key a snapshot from this pipeline holds: features, then scores."""
        return (*(spec.name for spec in self.specs), *(scorer.name for scorer in self.scorers))

    def __post_init__(self) -> None:
        seen: set[str] = set()
        for name in self.names:
            if name in seen:
                raise FeatureError(
                    f"duplicate feature name {name!r}. Two features under one name "
                    "would make a snapshot ambiguous and its hash meaningless."
                )
            seen.add(name)
        features = {spec.name for spec in self.specs}
        for scorer in self.scorers:
            missing = [name for name in scorer.inputs if name not in features]
            if missing:
                raise FeatureError(
                    f"scorer {scorer.name!r} reads {missing}, which this pipeline does not "
                    "compute as features. A score over a missing input would be UNKNOWN at "
                    "every decision, which reads as a model that never found anything."
                )

    def compute(
        self,
        window: BarWindow,
        instrument_uid: str,
        *,
        actions: Sequence[CorporateAction] = (),
    ) -> FeatureSnapshot:
        """The feature vector for one instrument at the window's decision time.

        `actions` is the *unfiltered* action set. It is deliberately not
        pre-filtered by the caller: `price_factor` applies both filters itself —
        `latest_vintages(..., as_of)` for what was known, `effective_after(...,
        at)` for what affects the price path — and those two are the classic
        thing to conflate. Duplicating that logic out here to hand it a
        "pre-filtered" list is how one of the two ends up applied twice or not
        at all, so the module that owns the distinction keeps it.
        """
        bars = window.bars(instrument_uid)
        # This instrument's own actions only. `price_factor` does not look at
        # the instrument, so a caller handing over a mixed list — as
        # `compute_all` does — would otherwise scale every name by every other
        # name's splits.
        own = [action for action in actions if action.instrument_uid == instrument_uid]
        closes, unadjustable = self._adjusted_closes(bars, actions=own, as_of=window.as_of)

        values: dict[str, FeatureValue] = {}
        for spec in self.specs:
            if len(closes) < spec.lookback:
                # Refused before computing. A mean over 40 of a requested 200
                # is not a shorter mean, it is a different number wearing the
                # same name.
                values[spec.name] = UNKNOWN
                continue
            window_closes = closes[-spec.lookback :]
            if any(close.unadjustable for close in window_closes):
                # This spec's own slice crosses a discontinuity we were not able
                # to adjust away, so the oldest closes in it are on a different
                # scale from the newest. A 4-for-1 left unadjusted reads as a
                # 300% return, and refusing is the only answer that does not
                # invent one. Per-spec rather than per-window: a lookback
                # entirely on the far side of the split is untouched by it.
                values[spec.name] = UNKNOWN
                continue
            values[spec.name] = self._round(spec.compute([close.value for close in window_closes]))

        for scorer in self.scorers:
            values[scorer.name] = self._score(scorer, values, as_of=window.as_of)

        return FeatureSnapshot(
            as_of=window.as_of,
            instrument_uid=instrument_uid,
            values=values,
            snapshot_hash=self._hash(window.as_of, instrument_uid, values),
            n_bars_seen=len(bars),
            series=self.series,
            unadjustable_actions=unadjustable,
        )

    def compute_all(
        self,
        window: BarWindow,
        *,
        actions: Sequence[CorporateAction] = (),
    ) -> dict[str, FeatureSnapshot]:
        """One snapshot per instrument in the window."""
        return {uid: self.compute(window, uid, actions=actions) for uid in window.instruments}

    # -- internals ---------------------------------------------------------

    def _adjusted_closes(
        self,
        bars: Sequence[Bar],
        *,
        actions: Sequence[CorporateAction],
        as_of: datetime,
    ) -> tuple[tuple[_Close, ...], tuple[str, ...]]:
        """Closes in this pipeline's series, oldest first, each knowing its scale.

        `RAW` is passed straight through — it is what the cross-venue price
        check, stop placement and tick rounding need. Anything else is scaled
        by the exact `Fraction` factor from the last session that bar's price
        already reflects (`Bar.quoted_through`), so a split part-way through
        the window does not appear as a return. That is the bar's own session
        for a raw feed and the day it was fetched for a vendor-adjusted one:
        Yahoo has already divided its history by every split before the fetch,
        and scaling from the bar's session would divide again.

        A close is `unadjustable` when its factor was incomplete: some action
        effective after that bar could not be applied, so this close is on a
        different scale from the newest one and the two cannot be compared. In
        practice that means an inferred split — `price_factor` excludes those
        deliberately, because a guessed ratio must not scale a real price. The
        returned action ids are those excluded anywhere in the window.
        """
        if self.series is Series.RAW or not actions:
            return tuple(_Close(bar.close, False) for bar in bars), ()

        adjusted: list[_Close] = []
        unadjustable: list[str] = []
        for bar in bars:
            factor = price_factor(actions, at=bar.quoted_through, as_of=as_of)
            for action_id in factor.missing:
                if action_id not in unadjustable:
                    unadjustable.append(action_id)
            with localcontext() as ctx:
                ctx.prec = _WORKING_PRECISION
                # Through the exact Fraction rather than float(factor): a
                # 1-for-3 reverse split is 1/3, and the drift from a float
                # round trip would be hashed faithfully into every snapshot.
                value = (
                    bar.close * Decimal(factor.value.numerator) / Decimal(factor.value.denominator)
                )
            adjusted.append(_Close(value, not factor.complete))
        return tuple(adjusted), tuple(unadjustable)

    @staticmethod
    def _round(value: FeatureValue) -> FeatureValue:
        if value is UNKNOWN or not isinstance(value, Decimal):
            return value
        quantum = Decimal(1).scaleb(-FEATURE_PLACES)
        return value.quantize(quantum)

    def _score(
        self, scorer: Scorer, values: Mapping[str, FeatureValue], *, as_of: datetime
    ) -> FeatureValue:
        """A scorer's value, `UNKNOWN` whenever any input is.

        Not imputed: the model was never fitted on a guessed input (the dataset
        drops such rows), so a score over one would be an answer to a question
        it was never asked. A non-finite score is a broken model, not absence,
        and raises rather than reading as a cautious one.
        """
        inputs = [values[name] for name in scorer.inputs]
        row: list[float] = []
        for value in inputs:
            if not isinstance(value, Decimal):
                return UNKNOWN
            row.append(float(value))
        result = scorer.score(row, as_of=as_of)
        if result is None:
            return UNKNOWN
        if not math.isfinite(result):
            raise FeatureError(f"scorer {scorer.name!r} returned {result} for {row}")
        # Through the float's exact binary value, then the same rounding as
        # every feature: the hash must not depend on how a float prints.
        return self._round(Decimal(result))

    def _hash(
        self,
        as_of: datetime,
        instrument_uid: str,
        values: Mapping[str, FeatureValue],
    ) -> str:
        return snapshot_hash(
            as_of=as_of, instrument_uid=instrument_uid, series=self.series, values=values
        )


def snapshot_hash(
    *,
    as_of: datetime,
    instrument_uid: str,
    series: Series,
    values: Mapping[str, FeatureValue],
) -> str:
    """Hash over the canonical form, with `UNKNOWN` as an explicit marker.

    `UNKNOWN` hashes as the string "UNKNOWN" rather than as null: a feature
    that was absent and a feature that was omitted are different facts, and a
    snapshot hash that conflated them would match across two genuinely
    different decisions.

    A module function rather than only a method, so `tb replay` recomputes the
    hash from a decision's recorded features with the same code that produced
    it, rather than with a second copy that could drift.
    """
    return hash_payload(
        {
            "as_of": as_of,
            "instrument_uid": instrument_uid,
            "series": series.value,
            "features": {
                name: "UNKNOWN" if value is UNKNOWN else value
                for name, value in sorted(values.items())
            },
        }
    )


def default_pipeline() -> FeaturePipeline:
    """The pipeline the CLI and the calibration use.

    Lookbacks are deliberately unremarkable. This is not a tuned set — the
    search loop in M6 proposes its own — it is a fixed reference so that
    calibration measures the *engine* rather than a particular feature choice.
    """
    return FeaturePipeline(
        specs=(
            make_spec("last", 1, name="close"),
            make_spec("sma", 10),
            make_spec("sma", 50),
            make_spec("return_pct", 5),
            make_spec("return_pct", 20),
            make_spec("stdev_pct", 20),
            make_spec("zscore", 20),
            make_spec("max_drawdown_pct", 20),
        )
    )
