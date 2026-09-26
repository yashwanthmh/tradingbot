"""Training a model, and searching the specs that read it, with every trial counted.

The I/O half of M7, as `research.loop` is of M6 — and built on it rather than
beside it, so the trial log, the multiplicity stamps, registration and the
holdout boundary are the same machinery for a model's specs as for any other:

    sealed vintage -> training reader -> labelled dataset (one pipeline)
      -> purged, embargoed walk-forward folds -> one model per fold
      -> out-of-sample skill, and the same folds on shuffled labels
      -> final model on every sample, recorded by hash
      -> threshold specs pinned to it, each a trial, judged walk-forward
      -> survivors registered as candidates

Four decisions carry the weight.

**Specs pin the final model and are judged by the fold models.** The final
model has seen every label in the training window, so a backtest of it there
would be in-sample. Each candidate is instead backtested with the fold models
— every decision scored by the model fitted before its fold — which is what the
recorded spec *would have* done had it been retrained on this schedule. The
final model's own out-of-sample test is the sealed holdout, spent once, later,
by `tb research holdout`.

**The null runs with every fit, and can stop it.** The same folds and
parameters on shuffled labels must show no skill. When they do, the evaluation
is supplying the skill and nothing from it is trustworthy, so nothing is
recorded: no model, no trial, no registration — `TrainingError` with both
numbers.

**Thresholds come from out-of-sample scores, and there are few of them.** A
threshold is a quantile of the walk-forward predictions, and the declared edge
is the out-of-sample mean gross return of the samples scored above it. Both are
read from the training window alone. The grid is short on purpose: every entry
is a trial, and ADR 0002's arithmetic says a search's haircut grows with its
size.

**A lineage is an idea, not a model.** The candidates' lineage is derived from
the features and the label definition, not from the artifact, so retraining one
idea with other parameters keeps adding to one lineage's count. "Retrain until
it passes" would otherwise be a sequence of small, separately-deflated searches.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from decimal import ROUND_HALF_EVEN, Decimal

from tb.config.hard_limits import HardLimits
from tb.core.clock import now_utc
from tb.core.errors import TbError
from tb.core.ids import deterministic_id, new_id
from tb.data.asof import HoldoutViolation
from tb.data.provider import Resolution
from tb.data.snapshot import SnapshotStore
from tb.features.pipeline import FeatureError, FeaturePipeline, pipeline_for
from tb.ledger.events import Actor
from tb.ledger.store import Ledger
from tb.registry.model_store import ModelRecord, ModelStore
from tb.registry.models import AuthorKind
from tb.research.holdout import DEFAULT_HOLDOUT_FRACTION, training_reader
from tb.research.loop import (
    DAILY_HOLD_MINUTES,
    CycleReport,
    EvaluationPlan,
    ProposalContext,
    ResearchCycle,
    lookback_for,
    record_violation,
    sealed_training_data,
)
from tb.research.mutate import Proposal, SpecProposer
from tb.research.searcher import SearchBudget
from tb.strategy.dsl.schema import MAX_LOOKBACK, MIN_LOOKBACK, ModelScore, StrategySpec
from tb.strategy.ml.cv import CrossValidationError, walk_forward_folds
from tb.strategy.ml.dataset import Dataset, LabelDefinition, build_dataset
from tb.strategy.ml.model import MODEL_KIND, ModelParams, TrainedModel
from tb.strategy.ml.trainer import (
    OutOfSample,
    WalkForwardScorer,
    shuffled,
    walk_forward,
)

_BPS = Decimal(10_000)
# Thresholds are constants in a spec, which carry at most the ten places every
# feature is rounded to; four is finer than any score difference that matters.
_THRESHOLD_QUANTUM = Decimal("0.0001")
_EDGE_QUANTUM = Decimal("0.01")


class TrainingError(TbError):
    """A model could not be trained, or its training cannot be trusted."""


@dataclass(frozen=True, slots=True)
class TrainingConfig:
    """What to learn, how, and which specs to try on the result."""

    features: tuple[tuple[str, int], ...]
    label: LabelDefinition
    params: ModelParams = field(default_factory=ModelParams)
    n_folds: int = 4
    # Calendar time between the last training label's completion and a fold's
    # first decision. Rolling features make the rows either side of a fold
    # boundary near-duplicates; a week of sessions separates them.
    embargo: timedelta = timedelta(days=7)
    min_train: int = 100
    entry_quantiles: tuple[float, ...] = (0.6, 0.75, 0.9)
    exit_quantile: float = 0.5
    null_seed: int = 0

    def __post_init__(self) -> None:
        if not self.features:
            raise TrainingError("a model needs at least one feature")
        for kind, lookback in self.features:
            if not MIN_LOOKBACK <= lookback <= MAX_LOOKBACK:
                raise TrainingError(
                    f"{kind} at {lookback} bars is outside the grammar's lookbacks "
                    f"({MIN_LOOKBACK}-{MAX_LOOKBACK}): a spec reading this model must be "
                    "computable by the same pipeline a hand-written one is"
                )
        try:
            pipeline_for(self.features)
        except FeatureError as exc:
            raise TrainingError(str(exc)) from exc
        if not self.entry_quantiles:
            raise TrainingError("at least one entry threshold is needed to propose a spec")
        for q in (*self.entry_quantiles, self.exit_quantile):
            if not 0 < q < 1:
                raise TrainingError(f"quantile {q} is not strictly between 0 and 1")
        if self.exit_quantile >= min(self.entry_quantiles):
            raise TrainingError(
                f"the exit quantile {self.exit_quantile} must sit below every entry quantile: "
                "a spec that exits where it enters trades every bar and pays for each one"
            )

    @property
    def lineage_id(self) -> str:
        """One lineage per idea — features and label — whatever the parameters."""
        return deterministic_id(
            "lin",
            parts={
                "kind": MODEL_KIND,
                "features": [[kind, lookback] for kind, lookback in self.features],
                "label": self.label.describe(),
            },
            length=12,
        )


@dataclass(frozen=True, slots=True)
class Threshold:
    """One candidate: enter above `entry`, leave below `exit`."""

    quantile: float
    entry: Decimal
    exit: Decimal
    # The out-of-sample mean gross return of the samples scored above `entry`,
    # in bps — the edge a spec at this threshold declares before it is
    # backtested, clamped into the band the limits permit.
    edge_bps: Decimal
    # What was measured before the clamp, so a report can say "-12bps out of
    # sample" rather than show the floor a negative edge was raised to.
    measured_bps: Decimal
    n_above: int


@dataclass(frozen=True, slots=True)
class ThresholdProposer:
    """The trainer's proposer: the threshold specs, once, attributed to ML."""

    specs: tuple[StrategySpec, ...]
    name: str = "ml-threshold"

    def propose(
        self,
        *,
        n: int,
        rng: object,
        parents: Sequence[StrategySpec] = (),
    ) -> list[Proposal]:
        del rng, parents
        return [
            Proposal(spec=spec, author_kind=AuthorKind.ML, operator="threshold")
            for spec in self.specs[:n]
        ]


@dataclass(frozen=True, slots=True)
class TrainingReport:
    """What one training run learned, and what its specs did."""

    record: ModelRecord
    oos: OutOfSample
    null: OutOfSample
    dataset_summary: tuple[int, int, int, int]
    thresholds: tuple[Threshold, ...]
    cycle: CycleReport

    def explain(self) -> str:
        n_samples, n_decisions, n_unknown, n_unlabelled = self.dataset_summary
        lines = [
            f"{self.record.model_id}: {n_samples} labelled sample(s) from {n_decisions} "
            f"decision(s) on {self.record.vintage_id} ({n_unknown} with an unknown feature, "
            f"{n_unlabelled} unlabelled at the seal)",
            _describe("out of sample", self.oos),
            _describe("shuffled labels, same folds", self.null)
            + " — the evaluation supplies no skill of its own",
        ]
        for threshold in self.thresholds:
            lines.append(
                f"q{threshold.quantile:g}: enter above {threshold.entry}, leave below "
                f"{threshold.exit}: {threshold.measured_bps}bps gross out of sample over "
                f"{threshold.n_above} sample(s) above, declaring {threshold.edge_bps}bps"
            )
        lines.append(self.cycle.explain())
        return "\n".join(lines)


class ModelTrainer:
    """Trains one model on a sealed vintage and searches its threshold specs."""

    def __init__(
        self,
        ledger: Ledger,
        *,
        limits: HardLimits,
        snapshots: SnapshotStore,
        models: ModelStore,
        run_id: str | None = None,
        clock: Callable[[], datetime] = now_utc,
    ) -> None:
        self._ledger = ledger
        self._limits = limits
        self._snapshots = snapshots
        self._models = models
        self._run_id = run_id
        self._clock = clock

    def run(
        self,
        *,
        vintage_id: str,
        config: TrainingConfig,
        fraction: float = DEFAULT_HOLDOUT_FRACTION,
        register: bool = True,
        search_id: str | None = None,
    ) -> TrainingReport:
        search = search_id or new_id("srch", length=12)
        data = sealed_training_data(self._snapshots, vintage_id, fraction=fraction)
        features = pipeline_for(config.features)
        try:
            dataset = build_dataset(
                reader=training_reader(
                    data.source,
                    sealed_from=data.window.sealed_from,
                    resolution=Resolution.DAILY,
                    instrument_uids=data.uids,
                    # Long enough for the features and for a label's entry bar
                    # to still be held when its exit arrives.
                    lookback=lookback_for(max(features.max_lookback, config.label.horizon + 1)),
                ),
                pipeline=features,
                decision_times=data.schedule,
                instruments=data.uids,
                label=config.label,
                actions=self._snapshots.actions_of(vintage_id),
            )
        except HoldoutViolation as exc:
            raise TrainingError(
                record_violation(
                    self._ledger,
                    exc,
                    caller=f"model training {search}",
                    window=data.window,
                    run_id=self._run_id,
                )
            ) from exc

        try:
            folds = walk_forward_folds(
                dataset.spans,
                n_folds=config.n_folds,
                embargo=config.embargo,
                min_train=config.min_train,
            )
        except CrossValidationError as exc:
            raise TrainingError(
                f"{len(dataset.samples)} labelled sample(s) cannot be walked forward: {exc}"
            ) from exc

        oos, fold_models = walk_forward(dataset, folds=folds, params=config.params)
        null, _ = walk_forward(
            dataset,
            folds=folds,
            params=config.params,
            labels=shuffled(dataset.labels, seed=config.null_seed),
        )
        if null.shows_skill():
            raise TrainingError(
                f"on shuffled labels the same folds scored AUC {null.auc:.3f}, beyond the "
                f"±{(null.auc_null_se or 0.0):.3f} a sample this size explains. The evaluation "
                "is supplying skill — a fold scored on rows it was fitted on is the classic "
                "cause — so nothing from it is trustworthy, and nothing was recorded."
            )
        thresholds = self._thresholds(oos, dataset, config)
        if not thresholds:
            raise TrainingError(
                "every entry quantile fell at or below the exit quantile: the out-of-sample "
                "scores do not spread enough to hold a position between them"
            )
        # Checked here, before anything is recorded, rather than left to the
        # cycle: a model recorded by a run that then could not backtest one
        # threshold would be a model with no trials behind it.
        first_scored = fold_models[0].fold.test_start
        if sum(1 for instant in data.schedule if instant >= first_scored) < 2:
            raise TrainingError(
                f"the first fold starts at {first_scored.isoformat()}, leaving fewer than two "
                "decisions to backtest a threshold on; a fill needs the bar after a decision"
            )

        final = TrainedModel.fit(
            dataset.rows, dataset.labels, feature_names=dataset.feature_names, params=config.params
        )
        record = self._models.record(
            final,
            dataset=dataset,
            features=config.features,
            params=config.params,
            vintage_id=vintage_id,
            sealed_from=data.window.sealed_from,
            metrics={
                **oos.metrics("oos_"),
                **null.metrics("null_"),
                "n_decisions": dataset.n_decisions,
                "n_unknown": dataset.n_unknown,
                "n_unlabelled": dataset.n_unlabelled,
                "embargo_days": config.embargo.total_seconds() / 86_400,
                "min_train": config.min_train,
                "lineage_id": config.lineage_id,
            },
            search_id=search,
            actor=Actor.SEARCH,
        )

        scorer = WalkForwardScorer(
            name=record.model_id, features=config.features, fold_models=fold_models
        )
        specs = tuple(
            _spec_for(record, threshold, min_holding_minutes=DAILY_HOLD_MINUTES)
            for threshold in thresholds
        )
        cycle = ResearchCycle(
            self._ledger,
            limits=self._limits,
            snapshots=self._snapshots,
            run_id=self._run_id,
            clock=self._clock,
        ).run(
            vintage_id=vintage_id,
            budget=SearchBudget(n_trials=len(specs), n_per_generation=len(specs), n_survivors=1),
            fraction=fraction,
            register=register,
            search_id=search,
            proposer=_proposer(specs),
            evaluation=EvaluationPlan(
                pipeline=_walk_forward_pipeline(record, config, scorer),
                starts_at=scorer.starts_at,
                allow_models=True,
                lineage_id=config.lineage_id,
            ),
        )
        return TrainingReport(
            record=record,
            oos=oos,
            null=null,
            dataset_summary=(
                len(dataset.samples),
                dataset.n_decisions,
                dataset.n_unknown,
                dataset.n_unlabelled,
            ),
            thresholds=thresholds,
            cycle=cycle,
        )

    def _thresholds(
        self, oos: OutOfSample, dataset: Dataset, config: TrainingConfig
    ) -> tuple[Threshold, ...]:
        scores = sorted(oos.predictions)
        exit_ = _quantile(scores, config.exit_quantile)
        high = Decimal(str(self._limits.costs.max_expected_edge_bps))
        out: list[Threshold] = []
        seen: set[tuple[Decimal, Decimal]] = set()
        for q in sorted(config.entry_quantiles):
            entry = _quantile(scores, q)
            if entry <= exit_ or (entry, exit_) in seen:
                continue
            seen.add((entry, exit_))
            above = [
                dataset.samples[index].gross_return
                for index, score in zip(oos.indices, oos.predictions, strict=True)
                if Decimal(score) > entry
            ]
            mean = sum(above, Decimal(0)) / len(above) * _BPS if above else Decimal(0)
            # Capped at the most the limits let any spec declare, and never
            # lifted: a claim below the band stays below it, and the validator
            # refuses that spec as it should — a trial, counted.
            edge = min(max(mean, _EDGE_QUANTUM), high).quantize(_EDGE_QUANTUM)
            out.append(
                Threshold(
                    quantile=q,
                    entry=entry,
                    exit=exit_,
                    edge_bps=edge,
                    measured_bps=mean.quantize(_EDGE_QUANTUM),
                    n_above=len(above),
                )
            )
        return tuple(out)


def _quantile(sorted_scores: Sequence[float], q: float) -> Decimal:
    """The nearest-rank quantile, as a spec constant."""
    index = min(len(sorted_scores) - 1, max(0, math.floor(q * len(sorted_scores))))
    return Decimal(sorted_scores[index]).quantize(_THRESHOLD_QUANTUM, rounding=ROUND_HALF_EVEN)


def _spec_for(
    record: ModelRecord, threshold: Threshold, *, min_holding_minutes: int
) -> StrategySpec:
    term = {
        "kind": "model",
        "model_id": record.model_id,
        "artifact_sha256": record.artifact_sha256,
    }
    return StrategySpec.parse(
        {
            "name": f"{record.model_id} above its q{threshold.quantile:g} score",
            "entry": {
                "kind": "compare",
                "op": "gt",
                "left": term,
                "right": {"kind": "const", "value": str(threshold.entry)},
            },
            "exit": {
                "kind": "compare",
                "op": "lt",
                "left": term,
                "right": {"kind": "const", "value": str(threshold.exit)},
            },
            "expected_edge_bps": str(threshold.edge_bps),
            "min_holding_minutes": min_holding_minutes,
            "notes": (
                f"trained on {record.vintage_id}, labels known through "
                f"{record.trained_through.isoformat()}; entry at the {threshold.quantile:g} "
                "quantile of walk-forward scores, exit at the median"
            ),
        }
    )


def _proposer(specs: tuple[StrategySpec, ...]) -> Callable[[ProposalContext], SpecProposer]:
    def build(context: ProposalContext) -> SpecProposer:
        del context
        return ThresholdProposer(specs=specs)

    return build


def _walk_forward_pipeline(
    record: ModelRecord, config: TrainingConfig, scorer: WalkForwardScorer
) -> Callable[[StrategySpec], FeaturePipeline]:
    pinned = ModelScore(model_id=record.model_id, artifact_sha256=record.artifact_sha256)

    def build(spec: StrategySpec) -> FeaturePipeline:
        if spec.model_refs != (pinned,):
            raise TrainingError(
                f"a candidate reads {[ref.model_id for ref in spec.model_refs]}, not only "
                f"{record.model_id}: this search judges specs on one model's folds"
            )
        requests = sorted(set(spec.feature_requests) | set(config.features))
        return pipeline_for(requests, scorers=[scorer])

    return build


def _describe(label: str, oos: OutOfSample) -> str:
    auc = "n/a" if oos.auc is None else f"{oos.auc:.3f}"
    se = "" if oos.auc_null_se is None else f" (noise ±{oos.auc_null_se:.3f})"
    skill = "n/a" if oos.skill is None else f"{oos.skill:+.3f}"
    return (
        f"{label}: AUC {auc}{se} over {oos.n_scored} prediction(s) in {oos.n_folds} "
        f"fold(s); log-loss skill {skill} against each fold's base rate"
    )


__all__ = [
    "ModelTrainer",
    "Threshold",
    "ThresholdProposer",
    "TrainingConfig",
    "TrainingError",
    "TrainingReport",
]
