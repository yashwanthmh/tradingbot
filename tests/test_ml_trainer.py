"""Walk-forward skill, and the shuffled-label null that must show none.

M7's verification, stated as properties of the arithmetic rather than of one
run: a label that *is* a feature is found out of sample (so the null is not
passing by being unable to find anything); the same folds on shuffled labels
find nothing, across seeds; and an in-sample score on shuffled labels — the
bug the null exists to catch — does show "skill", so the null has the power to
catch it. All on samples built by hand, with no bars and no ledger.
"""

from __future__ import annotations

import random
from datetime import UTC, datetime, timedelta
from decimal import Decimal

import pytest

from tb.strategy.ml.cv import Fold, LabelSpan, walk_forward_folds
from tb.strategy.ml.dataset import Dataset, LabelDefinition, Sample
from tb.strategy.ml.model import ModelError, ModelParams, TrainedModel
from tb.strategy.ml.trainer import (
    OutOfSample,
    WalkForwardScorer,
    shuffled,
    walk_forward,
)

UID = "isin:US0378331005"
BASE = datetime(2025, 1, 6, 21, tzinfo=UTC)
LABEL = LabelDefinition(horizon=2, cost_bps=Decimal(0))
PARAMS = ModelParams(n_trees=30, min_data_in_leaf=20)
NAMES = ("signal", "noise")


def dataset(
    n: int = 800,
    *,
    seed: int = 5,
    planted: bool = True,
    names: tuple[str, str] = NAMES,
) -> Dataset:
    """Daily samples with two features. Planted: the trade paid exactly when
    the first feature exceeded one half. Otherwise the outcome is a coin."""
    rng = random.Random(seed)
    samples: list[Sample] = []
    for day in range(n):
        signal, noise = rng.random(), rng.random()
        paid = signal > 0.5 if planted else rng.random() > 0.5
        move = Decimal("0.01") if paid else Decimal("-0.01")
        decided = BASE + timedelta(days=day)
        samples.append(
            Sample(
                instrument_uid=UID,
                features=(signal, noise),
                gross_return=move,
                net_return=move,
                span=LabelSpan(decided_at=decided, known_at=decided + timedelta(days=2)),
            )
        )
    return Dataset(
        feature_names=names,
        label=LABEL,
        samples=tuple(samples),
        n_decisions=n,
        n_unknown=0,
        n_unlabelled=0,
    )


def folds_of(data: Dataset) -> tuple[Fold, ...]:
    return walk_forward_folds(data.spans, n_folds=4, embargo=timedelta(days=1), min_train=50)


# --------------------------------------------------------------------------
# The positive control, and the null
# --------------------------------------------------------------------------


def test_a_planted_signal_is_found_out_of_sample() -> None:
    data = dataset()
    folds = folds_of(data)
    oos, fold_models = walk_forward(data, folds=folds, params=PARAMS)

    assert oos.auc is not None and oos.auc > 0.95
    assert oos.skill is not None and oos.skill > 0.5
    assert oos.shows_skill()
    # Every test sample scored exactly once, by the fold that holds it.
    assert sorted(oos.indices) == sorted(i for fold in folds for i in fold.test)
    assert len(fold_models) == len(folds)


@pytest.mark.parametrize("seed", range(8))
def test_shuffled_labels_show_no_skill(seed: int) -> None:
    """**The M7 verification.** The same folds, parameters and features, with
    the labels permuted: whatever the model finds now, the market did not put
    there."""
    data = dataset()
    null, _ = walk_forward(
        data, folds=folds_of(data), params=PARAMS, labels=shuffled(data.labels, seed=seed)
    )
    assert null.auc is not None and null.auc_null_se is not None
    assert not null.shows_skill(), f"shuffled labels scored AUC {null.auc:.3f}"
    assert null.skill is not None and null.skill < 0.02


def test_features_that_carry_nothing_show_no_skill() -> None:
    data = dataset(planted=False)
    oos, _ = walk_forward(data, folds=folds_of(data), params=PARAMS)
    assert not oos.shows_skill()


def test_an_in_sample_score_is_exactly_what_the_null_catches() -> None:
    """The null's power, shown on the bug it exists for: a model scored on the
    rows it was fitted on. On shuffled labels that is memorised noise, and it
    reads as skill — so a null that passes means the evaluation is not doing it."""
    data = dataset()
    labels = shuffled(data.labels, seed=0)
    memorised = TrainedModel.fit(
        data.rows, labels, feature_names=NAMES, params=ModelParams(n_trees=200, min_data_in_leaf=2)
    )
    in_sample = OutOfSample(
        indices=tuple(range(len(labels))),
        predictions=tuple(memorised.predict(data.rows)),
        labels=tuple(labels),
        base_rates=tuple([sum(labels) / len(labels)] * len(labels)),
        n_folds=1,
    )
    assert in_sample.shows_skill()


def test_a_fold_that_trains_on_its_test_rows_is_refused() -> None:
    """The in-sample bug in its plainest form is refused outright rather than
    left for the null to detect."""
    data = dataset(n=200)
    folds = folds_of(data)
    leaking = Fold(
        index=1,
        train=tuple(range(150)),
        test=tuple(range(100, 200)),
        test_start=data.samples[100].decided_at,
        test_end=data.samples[-1].decided_at,
        n_purged=0,
    )
    with pytest.raises(ModelError, match="trains on rows it is scored on"):
        walk_forward(data, folds=(leaking,), params=PARAMS)
    with pytest.raises(ModelError, match="already scored"):
        walk_forward(data, folds=(folds[-1], folds[-1]), params=PARAMS)
    with pytest.raises(ModelError, match="labels for"):
        walk_forward(data, folds=folds, params=PARAMS, labels=[1, 0])


def test_each_fold_model_is_fitted_on_its_training_rows_alone() -> None:
    """Refit from the fold's training indices only: the same bytes. Proves no
    fold model saw a row outside its purged, embargoed training set."""
    data = dataset(n=400)
    folds = folds_of(data)
    _, fold_models = walk_forward(data, folds=folds, params=PARAMS)
    labels = data.labels
    for fold_model in fold_models:
        refit = TrainedModel.fit(
            [data.rows[i] for i in fold_model.fold.train],
            [labels[i] for i in fold_model.fold.train],
            feature_names=NAMES,
            params=PARAMS,
        )
        assert refit.sha256 == fold_model.model.sha256


# --------------------------------------------------------------------------
# The arithmetic
# --------------------------------------------------------------------------


def oos(predictions: list[float], labels: list[int], base: float = 0.5) -> OutOfSample:
    return OutOfSample(
        indices=tuple(range(len(labels))),
        predictions=tuple(predictions),
        labels=tuple(labels),
        base_rates=tuple([base] * len(labels)),
        n_folds=1,
    )


def test_auc_is_the_chance_a_winner_outscores_a_loser() -> None:
    assert oos([0.1, 0.4, 0.35, 0.8], [0, 0, 1, 1]).auc == pytest.approx(0.75)
    # Ties count half.
    assert oos([0.5, 0.5], [0, 1]).auc == pytest.approx(0.5)
    assert oos([0.9, 0.1], [0, 1]).auc == pytest.approx(0.0)
    # One class: no ordering to measure.
    assert oos([0.2, 0.3], [1, 1]).auc is None
    assert not oos([0.2, 0.3], [1, 1]).shows_skill()


def test_the_base_rate_has_no_skill_by_definition() -> None:
    labels = [1, 0, 0, 1, 0, 0, 1, 0]
    rate = sum(labels) / len(labels)
    assert oos([rate] * len(labels), labels, base=rate).skill == pytest.approx(0.0)
    worse = oos([1 - rate] * len(labels), labels, base=rate)
    assert worse.skill is not None and worse.skill < 0


def test_metrics_are_recordable() -> None:
    metrics = oos([0.1, 0.4, 0.35, 0.8], [0, 0, 1, 1]).metrics("oos_")
    assert metrics["oos_auc"] == pytest.approx(0.75)
    assert metrics["oos_n_scored"] == 4
    assert set(metrics) >= {"oos_auc_null_se", "oos_log_loss", "oos_base_log_loss", "oos_skill"}


# --------------------------------------------------------------------------
# The walk-forward scorer
# --------------------------------------------------------------------------


def test_each_decision_is_scored_by_the_model_fitted_before_its_fold() -> None:
    """Nothing before the first fold — no model exists that has not seen what
    follows — then each fold's own model, and the last one beyond."""
    features = (("last", 1), ("sma", 2))
    data = dataset(n=400, names=("last_1", "sma_2"))
    _, fold_models = walk_forward(data, folds=folds_of(data), params=PARAMS)
    scorer = WalkForwardScorer(name="mdl_walk", features=features, fold_models=fold_models)
    row = [0.7, 0.2]

    first = fold_models[0].fold
    assert scorer.starts_at == first.test_start
    assert scorer.score(row, as_of=first.test_start - timedelta(seconds=1)) is None
    for fold_model in fold_models:
        inside = fold_model.fold.test_start + timedelta(hours=1)
        assert scorer.score(row, as_of=inside) == fold_model.model.score(row)
    beyond = fold_models[-1].fold.test_end + timedelta(days=30)
    assert scorer.score(row, as_of=beyond) == fold_models[-1].model.score(row)


def test_a_walk_forward_scorer_refuses_models_of_other_columns() -> None:
    data = dataset(n=400, names=("last_1", "sma_2"))
    _, fold_models = walk_forward(data, folds=folds_of(data), params=PARAMS)
    with pytest.raises(ModelError, match="reads"):
        WalkForwardScorer(
            name="mdl_walk", features=(("sma", 2), ("last", 1)), fold_models=fold_models
        )
    with pytest.raises(ModelError, match="ascending"):
        WalkForwardScorer(
            name="mdl_walk",
            features=(("last", 1), ("sma", 2)),
            fold_models=tuple(reversed(fold_models)),
        )
    with pytest.raises(ModelError, match="at least one"):
        WalkForwardScorer(name="mdl_walk", features=(("last", 1),), fold_models=())
