"""Walk-forward training, out-of-sample skill, and the null that must show none.

The part of training that is arithmetic. `tb.research.ml` gives it a sealed
dataset and records what it concludes; this module holds no ledger, no clock
and no bars, so every claim it makes is testable on samples built by hand.

**Out of sample means walk-forward.** Each fold is scored by a model fitted only
on the samples `cv.walk_forward_folds` leaves it — decided before the fold and
known an embargo ahead of it — so every prediction reported here is one a live
model could have made at that instant. The in-sample fit of the final model is
never scored: it has seen every label in the window, and its training accuracy
is a measurement of its capacity, not of the market.

**Skill is measured against the model that knows nothing.** Two numbers, both
of which a model with no information scores at zero:

* **AUC** — the chance a profitable trade was scored above an unprofitable one.
  0.5 is no skill whatever the base rate, and its standard error under that
  null is known in closed form, so "0.53" can be read against the noise a
  sample this size produces rather than against intuition.
* **Log-loss skill** — the improvement over predicting each fold's own training
  base rate. At or below zero, the model's probabilities are worth less than a
  count of how often trades used to pay.

**The null runs every time.** `shuffled` permutes the labels, which keeps every
feature and the base rate and destroys any relation between them. The same
folds and parameters on those labels must show no skill; if they do, the
evaluation — not the market — is supplying it (a fold scored on rows it was
trained on is the classic way), and every positive result from the same
machinery is suspect. The trainer runs the null beside the real fit and refuses
to record a model when the null fails, so the M7 verification is a property of
every training run rather than of one test.
"""

from __future__ import annotations

import math
import random
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import datetime

from tb.strategy.ml.cv import Fold
from tb.strategy.ml.dataset import Dataset
from tb.strategy.ml.model import ModelError, ModelParams, TrainedModel

# How far a null AUC may sit from 0.5, in standard errors, before the
# machinery is called broken. Four: at one shuffle per run, a working
# evaluation trips it about once in 16,000 runs, and a leaking one — which
# scores well above 0.9 on shuffled labels — trips it every time.
NULL_TOLERANCE_SE = 4.0

# Probabilities are clipped here before a log is taken. LightGBM never returns
# 0 or 1 exactly, but a base rate of 0 or 1 in a fold is possible.
_EPSILON = 1e-12


@dataclass(frozen=True, slots=True)
class FoldModel:
    """A fold and the model fitted on its training samples."""

    fold: Fold
    model: TrainedModel


@dataclass(frozen=True, slots=True)
class OutOfSample:
    """Every test-fold prediction, and what the set of them says about skill.

    Aligned tuples rather than a mapping so the arithmetic below is
    order-stable. Each sample is scored at most once, by the one fold whose
    test window contains it.
    """

    indices: tuple[int, ...]
    predictions: tuple[float, ...]
    labels: tuple[int, ...]
    # Per prediction: the base rate of the training set of the fold that made
    # it — the prediction a model with no information would have made instead.
    base_rates: tuple[float, ...]
    n_folds: int

    @property
    def n_scored(self) -> int:
        return len(self.predictions)

    @property
    def n_positive(self) -> int:
        return sum(self.labels)

    @property
    def auc(self) -> float | None:
        """Mann-Whitney AUC, ties counted as half. `None` with only one class."""
        positives = self.n_positive
        negatives = self.n_scored - positives
        if positives == 0 or negatives == 0:
            return None
        ranks = _average_ranks(self.predictions)
        rank_sum = sum(rank for rank, label in zip(ranks, self.labels, strict=True) if label)
        u = rank_sum - positives * (positives + 1) / 2
        return u / (positives * negatives)

    @property
    def auc_null_se(self) -> float | None:
        """The AUC's standard error when the scores carry no information."""
        positives = self.n_positive
        negatives = self.n_scored - positives
        if positives == 0 or negatives == 0:
            return None
        return math.sqrt((positives + negatives + 1) / (12 * positives * negatives))

    @property
    def log_loss(self) -> float:
        return _log_loss(self.predictions, self.labels)

    @property
    def base_log_loss(self) -> float:
        return _log_loss(self.base_rates, self.labels)

    @property
    def skill(self) -> float | None:
        """1 - log_loss / base_log_loss: zero for the know-nothing model."""
        base = self.base_log_loss
        if base <= 0:
            return None
        return 1 - self.log_loss / base

    def shows_skill(self, *, tolerance_se: float = NULL_TOLERANCE_SE) -> bool:
        """Whether the AUC sits further from 0.5 than noise of this size explains."""
        auc, se = self.auc, self.auc_null_se
        if auc is None or se is None:
            return False
        return abs(auc - 0.5) > tolerance_se * se

    def metrics(self, prefix: str = "oos_") -> dict[str, float | int | None]:
        return {
            f"{prefix}n_folds": self.n_folds,
            f"{prefix}n_scored": self.n_scored,
            f"{prefix}base_rate": _rounded(
                self.n_positive / self.n_scored if self.n_scored else None
            ),
            f"{prefix}auc": _rounded(self.auc),
            f"{prefix}auc_null_se": _rounded(self.auc_null_se),
            f"{prefix}log_loss": _rounded(self.log_loss if self.n_scored else None),
            f"{prefix}base_log_loss": _rounded(self.base_log_loss if self.n_scored else None),
            f"{prefix}skill": _rounded(self.skill if self.n_scored else None),
        }


def walk_forward(
    dataset: Dataset,
    *,
    folds: Sequence[Fold],
    params: ModelParams,
    labels: Sequence[int] | None = None,
) -> tuple[OutOfSample, tuple[FoldModel, ...]]:
    """Fit one model per fold on that fold's training samples; score its test samples.

    `labels` replaces the dataset's own, for the null. The folds are taken as
    given — purging and embargo are `cv`'s job and are tested there — but a
    test index scored twice, or a training index inside its own test set, is
    refused rather than scored: either would put in-sample predictions in an
    out-of-sample number.
    """
    targets = list(dataset.labels if labels is None else labels)
    if len(targets) != len(dataset.samples):
        raise ModelError(f"{len(targets)} labels for {len(dataset.samples)} samples")
    if not folds:
        raise ModelError("no folds to walk forward through")
    rows = dataset.rows

    indices: list[int] = []
    predictions: list[float] = []
    truths: list[int] = []
    base_rates: list[float] = []
    fold_models: list[FoldModel] = []
    scored: set[int] = set()
    for fold in folds:
        if set(fold.train) & set(fold.test):
            raise ModelError(f"fold {fold.index} trains on rows it is scored on")
        if scored & set(fold.test):
            raise ModelError(f"fold {fold.index} scores rows an earlier fold already scored")
        train_labels = [targets[i] for i in fold.train]
        model = TrainedModel.fit(
            [rows[i] for i in fold.train],
            train_labels,
            feature_names=dataset.feature_names,
            params=params,
        )
        fold_models.append(FoldModel(fold=fold, model=model))
        base_rate = sum(train_labels) / len(train_labels)
        for index, prediction in zip(
            fold.test, model.predict([rows[i] for i in fold.test]), strict=True
        ):
            indices.append(index)
            predictions.append(prediction)
            truths.append(targets[index])
            base_rates.append(base_rate)
        scored |= set(fold.test)

    return (
        OutOfSample(
            indices=tuple(indices),
            predictions=tuple(predictions),
            labels=tuple(truths),
            base_rates=tuple(base_rates),
            n_folds=len(folds),
        ),
        tuple(fold_models),
    )


def shuffled(labels: Sequence[int], *, seed: int) -> list[int]:
    """The labels permuted: every feature and the base rate kept, their relation gone."""
    out = list(labels)
    random.Random(seed).shuffle(out)  # noqa: S311 - a null draw, not a key
    return out


@dataclass(frozen=True, slots=True)
class WalkForwardScorer:
    """The fold models as one scorer: each decision scored by the model fitted before it.

    What a candidate spec is backtested with during training. The spec pins
    the final model, which has seen every label in the window, so scoring the
    training window with it would be in-sample; this scores each decision with
    the model of the fold it falls in — fitted only on labels known an embargo
    before that fold began — and the last fold's model beyond it. Before the
    first fold there is no model that has not seen the future, and the score is
    `None`: unknown, so no entry.
    """

    name: str
    features: tuple[tuple[str, int], ...]
    fold_models: tuple[FoldModel, ...]

    def __post_init__(self) -> None:
        if not self.fold_models:
            raise ModelError("a walk-forward scorer needs at least one fold model")
        starts = [fold_model.fold.test_start for fold_model in self.fold_models]
        if starts != sorted(starts) or len(set(starts)) != len(starts):
            raise ModelError("fold models must be in ascending order of their test windows")
        rebuilt = tuple(f"{kind}_{lookback}" for kind, lookback in self.features)
        for fold_model in self.fold_models:
            if fold_model.model.feature_names != rebuilt:
                raise ModelError(
                    f"fold {fold_model.fold.index}'s model reads "
                    f"{list(fold_model.model.feature_names)}, not {list(rebuilt)}"
                )

    @property
    def inputs(self) -> tuple[str, ...]:
        return self.fold_models[0].model.feature_names

    @property
    def starts_at(self) -> datetime:
        """The first instant any fold model may score."""
        return self.fold_models[0].fold.test_start

    def score(self, row: Sequence[float], *, as_of: datetime) -> float | None:
        chosen: FoldModel | None = None
        for fold_model in self.fold_models:
            if fold_model.fold.test_start > as_of:
                break
            chosen = fold_model
        return None if chosen is None else chosen.model.score(row)


def _average_ranks(values: Sequence[float]) -> list[float]:
    """1-based ranks, ties sharing the mean of the ranks they span."""
    order = sorted(range(len(values)), key=lambda i: values[i])
    ranks = [0.0] * len(values)
    position = 0
    while position < len(order):
        end = position
        while end + 1 < len(order) and values[order[end + 1]] == values[order[position]]:
            end += 1
        shared = (position + end) / 2 + 1
        for k in range(position, end + 1):
            ranks[order[k]] = shared
        position = end + 1
    return ranks


def _log_loss(probabilities: Sequence[float], labels: Sequence[int]) -> float:
    if not labels:
        return 0.0
    total = 0.0
    for probability, label in zip(probabilities, labels, strict=True):
        p = min(max(probability, _EPSILON), 1 - _EPSILON)
        total -= math.log(p) if label else math.log(1 - p)
    return total / len(labels)


def _rounded(value: float | None) -> float | None:
    return None if value is None else round(value, 6)
