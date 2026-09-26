"""Labelled samples for a model, from a sealed vintage, through the one pipeline.

A model is only as honest as the rows it is fitted to, and there are three ways
for a row to lie. This module closes each of them by construction rather than by
care:

**Features the live loop would not compute.** Every sample's features come from
`FeaturePipeline.compute` on the window the forward-only reader returns at the
decision time, with the vintage's corporate actions — the same object and the
same call the loop makes. A training frame built any other way (a dataframe of
rolling means, say) is a second pipeline, and two pipelines drift.

**A label that is not the trade.** The label is what a strategy acting on the
decision would have realised: in at the *next* bar's open — the backtester's
fill, since the decision bar's close is gone by the time it is known — out at
the open `horizon` bars later, net of the cost model's round trip. The holding
follows any split between the two exactly as the backtester's does, so a 4-for-1
inside the horizon is no move rather than a 75% loss.

**A label from beyond the seal.** Labels are completed only as the reader walks
forward and their exit bar becomes visible. The reader a research process gets
is sealed, so a sample whose exit falls in the holdout never completes and is
dropped: no label can carry holdout prices into training, because those prices
are never in memory. Each sample's span ends when its exit bar became knowable,
which is what `cv.walk_forward_folds` purges on.

A sample with any `UNKNOWN` feature is dropped and counted rather than imputed.
The loop would have held on it — the model's score is `UNKNOWN` too — so a row
filled in with a guess would be training on a decision that could never be made.
"""

from __future__ import annotations

from bisect import bisect_right
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from datetime import datetime
from decimal import Decimal, localcontext

from tb.core.errors import TbError
from tb.data.adjustments import CorporateAction, holding_factor
from tb.data.asof import ForwardOnlyReader
from tb.data.provider import Bar
from tb.features.pipeline import FeaturePipeline, FeatureValue
from tb.strategy.ml.cv import LabelSpan

_WORKING_PRECISION = 40
_BPS = Decimal(10_000)


class DatasetError(TbError):
    """A dataset could not be built from what was asked for."""


@dataclass(frozen=True, slots=True)
class LabelDefinition:
    """What a sample's label measures: a round trip held `horizon` bars, net of cost."""

    horizon: int
    cost_bps: Decimal

    def __post_init__(self) -> None:
        if self.horizon < 1:
            raise DatasetError(f"a label horizon of {self.horizon} bars holds nothing")
        if not self.cost_bps.is_finite() or self.cost_bps < 0:
            raise DatasetError(f"a round-trip cost of {self.cost_bps}bps is not a cost")

    def describe(self) -> dict[str, str]:
        """The definition as recorded alongside a model: enough to rebuild the labels."""
        return {
            "horizon_bars": str(self.horizon),
            "cost_bps": str(self.cost_bps),
            "entry": "open of the first bar knowable after the decision",
            "exit": "open of the bar `horizon` bars after the entry bar",
            "adjusted": "holding follows splits between entry and exit",
        }


@dataclass(frozen=True, slots=True)
class Sample:
    """One decision, the features it saw, and the trade it would have been."""

    instrument_uid: str
    features: tuple[float, ...]
    gross_return: Decimal
    net_return: Decimal
    span: LabelSpan

    @property
    def decided_at(self) -> datetime:
        return self.span.decided_at

    @property
    def label(self) -> int:
        """1 when the round trip made money after its costs."""
        return 1 if self.net_return > 0 else 0


@dataclass(frozen=True, slots=True)
class Dataset:
    """Samples in decision order, with what was dropped and why."""

    feature_names: tuple[str, ...]
    label: LabelDefinition
    samples: tuple[Sample, ...]
    n_decisions: int
    n_unknown: int
    n_unlabelled: int

    @property
    def spans(self) -> tuple[LabelSpan, ...]:
        return tuple(sample.span for sample in self.samples)

    @property
    def rows(self) -> list[list[float]]:
        return [list(sample.features) for sample in self.samples]

    @property
    def labels(self) -> list[int]:
        return [sample.label for sample in self.samples]

    @property
    def base_rate(self) -> float | None:
        """The share of profitable labels: what a model with no skill converges to."""
        if not self.samples:
            return None
        return sum(self.labels) / len(self.samples)


@dataclass(slots=True)
class _Pending:
    uid: str
    decided_at: datetime
    features: tuple[float, ...]


@dataclass(slots=True)
class _Walk:
    pending: list[_Pending] = field(default_factory=list)
    samples: list[Sample] = field(default_factory=list)


def build_dataset(
    *,
    reader: ForwardOnlyReader,
    pipeline: FeaturePipeline,
    decision_times: Sequence[datetime],
    instruments: Sequence[str],
    label: LabelDefinition,
    actions: Mapping[str, Sequence[CorporateAction]] | None = None,
) -> Dataset:
    """Walk the reader forward once, computing features now and labels as they arrive."""
    if list(decision_times) != sorted(decision_times):
        raise DatasetError("decision times must be ascending: the reader cannot rewind")
    by_uid = actions or {}
    walk = _Walk()
    n_decisions = 0
    n_unknown = 0

    for moment in decision_times:
        window = reader.advance_to(moment)
        for uid in sorted(instruments):
            bars = window.bars(uid)
            _complete(walk, uid=uid, bars=bars, label=label, actions=by_uid.get(uid, ()))
            snapshot = pipeline.compute(window, uid, actions=by_uid.get(uid, ()))
            n_decisions += 1
            if not snapshot.complete:
                n_unknown += 1
                continue
            walk.pending.append(
                _Pending(uid=uid, decided_at=moment, features=_row(snapshot.values, pipeline))
            )

    samples = sorted(walk.samples, key=lambda sample: (sample.decided_at, sample.instrument_uid))
    return Dataset(
        feature_names=pipeline.names,
        label=label,
        samples=tuple(samples),
        n_decisions=n_decisions,
        n_unknown=n_unknown,
        n_unlabelled=len(walk.pending),
    )


def _row(values: Mapping[str, FeatureValue], pipeline: FeaturePipeline) -> tuple[float, ...]:
    """The snapshot's values in the pipeline's order, as the model reads them."""
    row: list[float] = []
    for name in pipeline.names:
        value = values[name]
        if not isinstance(value, Decimal):  # pragma: no cover - `complete` was checked
            raise DatasetError(f"{name} is UNKNOWN in a snapshot reported complete")
        row.append(float(value))
    return tuple(row)


def _complete(
    walk: _Walk,
    *,
    uid: str,
    bars: Sequence[Bar],
    label: LabelDefinition,
    actions: Sequence[CorporateAction],
) -> None:
    """Label every pending decision on `uid` whose exit bar is now visible."""
    still: list[_Pending] = []
    knowable = [bar.available_at_utc for bar in bars]
    for pending in walk.pending:
        if pending.uid != uid:
            still.append(pending)
            continue
        # The first bar knowable after the decision: the backtester's fill.
        entry = bisect_right(knowable, pending.decided_at)
        if entry + label.horizon >= len(bars):
            still.append(pending)
            continue
        opened, closed = bars[entry], bars[entry + label.horizon]
        walk.samples.append(
            _sample(pending, opened=opened, closed=closed, label=label, actions=actions)
        )
    walk.pending = still


def _sample(
    pending: _Pending,
    *,
    opened: Bar,
    closed: Bar,
    label: LabelDefinition,
    actions: Sequence[CorporateAction],
) -> Sample:
    shares = holding_factor(actions, quoted_through=opened.quoted_through, to=closed.quoted_through)
    with localcontext() as ctx:
        ctx.prec = _WORKING_PRECISION
        gross = closed.open * Decimal(shares.numerator) / Decimal(
            shares.denominator
        ) / opened.open - Decimal(1)
        net = gross - label.cost_bps / _BPS
    return Sample(
        instrument_uid=pending.uid,
        features=pending.features,
        gross_return=gross,
        net_return=net,
        span=LabelSpan(decided_at=pending.decided_at, known_at=closed.available_at_utc),
    )
