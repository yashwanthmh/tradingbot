"""A model's score as a DSL term: read like a feature, computed inside the one pipeline.

The claim under test is that nothing downstream needs to know a model exists.
The pipeline computes the score from its own features at the decision time and
puts it in the snapshot under the model's id; the interpreter reads it as it
reads a moving average; the backtester, the snapshot hash and the validator
all see an ordinary spec. What *is* model-specific is refused loudly: a spec
without its store, a pin the store does not hold, a scorer fed columns other
than the ones it was fitted on, and a searched spec reaching for a model at all.
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from decimal import Decimal
from pathlib import Path
from typing import Any

import pytest

from tb.backtest.costs import CostModel, Jurisdiction
from tb.backtest.engine import Backtester, InstrumentMeta
from tb.config.loader import load_hard_limits
from tb.data.asof import UNKNOWN, ForwardOnlyReader, InMemoryBarSource
from tb.data.provider import Bar, Provenance, Resolution, Session
from tb.features.pipeline import (
    FEATURE_PLACES,
    FeatureError,
    FeaturePipeline,
    pipeline_for,
    snapshot_hash,
)
from tb.ledger.store import Ledger
from tb.registry.model_store import ModelRecord, ModelStore, model_id_for
from tb.research.holdout import decisions_between
from tb.research.validate import SpecValidator
from tb.strategy.base import Action, PositionState
from tb.strategy.dsl.interpreter import check_features_available
from tb.strategy.dsl.ops import DslStrategy, pipeline_from_spec
from tb.strategy.dsl.schema import ModelScore, SpecError, StrategySpec
from tb.strategy.ml.dataset import Dataset, LabelDefinition, build_dataset
from tb.strategy.ml.model import ModelError, ModelParams, ModelScorer, TrainedModel

UID = "isin:US0378331005"
BASE = datetime(2026, 1, 5, tzinfo=UTC)
# Deliberately not in the pipeline's sorted order: the model reads
# `return_pct_2` first, the pipeline computes `last_1` first, and the scorer
# must feed the model by name, not by position in the snapshot.
FEATURES = (("return_pct", 2), ("last", 1))
PARAMS = ModelParams(n_trees=8, min_data_in_leaf=10)
LIMITS = load_hard_limits(None).limits


def daily(offset: int, price: Decimal) -> Bar:
    opened = BASE + timedelta(days=offset)
    return Bar(
        instrument_uid=UID,
        resolution=Resolution.DAILY,
        bar_open_utc=opened,
        available_at_utc=opened + timedelta(days=1),
        ingested_at_utc=opened + timedelta(days=1),
        provider="alpaca",
        provenance=Provenance.BACKFILL,
        session=Session.REGULAR,
        open=price,
        high=price,
        low=price,
        close=price,
        volume=1_000_000,
    )


def walk(days: int, *, seed: int = 7) -> list[Bar]:
    rng = random.Random(seed)
    price = Decimal(100)
    bars: list[Bar] = []
    for offset in range(days):
        step = Decimal(str(round(rng.uniform(-0.03, 0.03), 6)))
        price = (price * (1 + step)).quantize(Decimal("0.01"))
        bars.append(daily(offset, price))
    return bars


def reader(bars: list[Bar]) -> ForwardOnlyReader:
    return ForwardOnlyReader(
        source=InMemoryBarSource(bars=bars), resolution=Resolution.DAILY, instrument_uids=(UID,)
    )


@dataclass(frozen=True, slots=True)
class Trained:
    store: ModelStore
    record: ModelRecord
    model: TrainedModel
    data: Dataset
    bars: list[Bar]


def train(store: ModelStore, bars: list[Bar], params: ModelParams = PARAMS) -> Trained:
    data = build_dataset(
        reader=reader(bars),
        pipeline=pipeline_for(FEATURES),
        decision_times=decisions_between(bars),
        instruments=(UID,),
        label=LabelDefinition(horizon=2, cost_bps=Decimal(0)),
    )
    model = TrainedModel.fit(
        data.rows, data.labels, feature_names=data.feature_names, params=params
    )
    record = store.record(
        model,
        dataset=data,
        features=FEATURES,
        params=params,
        vintage_id="vin_test",
        sealed_from=None,
    )
    return Trained(store=store, record=record, model=model, data=data, bars=bars)


@pytest.fixture
def trained(ledger: Ledger, tmp_path: Path) -> Trained:
    return train(ModelStore(ledger, tmp_path / "models"), walk(120))


def term(model_id: str, sha: str) -> dict[str, Any]:
    return {"kind": "model", "model_id": model_id, "artifact_sha256": sha}


def spec_reading(model: dict[str, Any], *, enter: str = "0.5", leave: str = "0.5") -> StrategySpec:
    return StrategySpec.parse(
        {
            "name": "the model thinks it pays",
            "entry": {
                "kind": "compare",
                "op": "gt",
                "left": model,
                "right": {"kind": "const", "value": enter},
            },
            "exit": {
                "kind": "compare",
                "op": "lt",
                "left": model,
                "right": {"kind": "const", "value": leave},
            },
            "expected_edge_bps": "300",
            "min_holding_minutes": 1440,
        }
    )


def pinned(trained: Trained) -> dict[str, Any]:
    return term(trained.record.model_id, trained.record.artifact_sha256)


# --------------------------------------------------------------------------
# The term
# --------------------------------------------------------------------------


def test_a_model_is_named_by_its_own_hash() -> None:
    """The DSL's naming rule and the store's are one rule: a pair the store
    would derive always parses, and a pair it would not never does."""
    for sha in ("0" * 64, "ab" * 32, "0123456789abcdef" * 4):
        parsed = ModelScore(model_id=model_id_for(sha), artifact_sha256=sha)
        assert parsed.feature_key == model_id_for(sha)

    other = "f" * 64
    with pytest.raises(SpecError, match="named by its own hash"):
        spec_reading(term(model_id_for("0" * 64), other))
    for bad_id, bad_sha in (
        ("mdl_XYZ", "0" * 64),
        (model_id_for("0" * 64), "0" * 63),
        ("mdl_" + "0" * 16, "0" * 64 + "\n"),
        ("__import__('os')", "0" * 64),
    ):
        with pytest.raises(SpecError):
            spec_reading(term(bad_id, bad_sha))


def test_a_spec_reads_the_score_as_one_more_snapshot_key(trained: Trained) -> None:
    spec = spec_reading(pinned(trained))
    assert spec.model_refs == (
        ModelScore(
            model_id=trained.record.model_id, artifact_sha256=trained.record.artifact_sha256
        ),
    )
    assert spec.required_features == (trained.record.model_id,)
    # The spec reads no feature directly; the model's inputs are on its record.
    assert spec.feature_requests == ()
    assert spec.n_nodes == 6


# --------------------------------------------------------------------------
# The pipeline
# --------------------------------------------------------------------------


def test_a_spec_reading_a_model_is_refused_without_the_store(trained: Trained) -> None:
    with pytest.raises(SpecError, match="no model store"):
        pipeline_from_spec(spec_reading(pinned(trained)))


def test_the_score_is_the_model_on_the_columns_it_was_trained_on(trained: Trained) -> None:
    """At every decision the dataset labelled, the pipeline's score is the
    model's prediction on that dataset row — the same features, fed in the
    model's order rather than the snapshot's, so training and trading ask the
    model one question."""
    spec = spec_reading(pinned(trained))
    pipeline = pipeline_from_spec(spec, models=trained.store)
    model_id = trained.record.model_id
    assert pipeline.names == ("last_1", "return_pct_2", model_id)

    rows = {sample.decided_at: sample.features for sample in trained.data.samples}
    walker = reader(trained.bars)
    compared = 0
    for moment in decisions_between(trained.bars):
        snapshot = pipeline.compute(walker.advance_to(moment), UID)
        if moment not in rows:
            continue
        expected = Decimal(trained.model.score(rows[moment])).quantize(
            Decimal(1).scaleb(-FEATURE_PLACES)
        )
        assert snapshot.values[model_id] == expected
        check_features_available(spec, snapshot.values)
        compared += 1
    assert compared == len(trained.data.samples)


def test_an_unknown_input_is_an_unknown_score_and_no_entry(trained: Trained) -> None:
    """One bar in: a two-bar return has no value, so neither has the score, and
    the strategy holds rather than entering on a guess."""
    spec = spec_reading(pinned(trained))
    pipeline = pipeline_from_spec(spec, models=trained.store)
    first = decisions_between(trained.bars)[0]
    window = reader(trained.bars).advance_to(first)
    snapshot = pipeline.compute(window, UID)
    assert snapshot.values["return_pct_2"] is UNKNOWN
    assert snapshot.values[trained.record.model_id] is UNKNOWN

    decision = DslStrategy(spec=spec, strategy_id="stg_ml").decide(
        snapshot=snapshot, window=window, position=PositionState(instrument_uid=UID)
    )
    assert decision.action is Action.HOLD


def test_the_snapshot_hash_covers_the_score(ledger: Ledger, tmp_path: Path) -> None:
    """Two models over identical features: identical inputs, different scores,
    different hashes — so a decision's hash says which model it heard. And the
    hash recomputes from the recorded values, which is what `tb replay` does."""
    bars = walk(120)
    store = ModelStore(ledger, tmp_path / "models")
    first = train(store, bars)
    second = train(store, bars, ModelParams(n_trees=24, min_data_in_leaf=10))
    assert first.record.model_id != second.record.model_id

    moment = decisions_between(bars)[60]
    snapshots = [
        pipeline_from_spec(spec_reading(pinned(one)), models=store).compute(
            reader(bars).advance_to(moment), UID
        )
        for one in (first, second)
    ]
    assert snapshots[0].values["last_1"] == snapshots[1].values["last_1"]
    assert snapshots[0].snapshot_hash != snapshots[1].snapshot_hash
    for snapshot in snapshots:
        assert snapshot.snapshot_hash == snapshot_hash(
            as_of=snapshot.as_of,
            instrument_uid=UID,
            series=snapshot.series,
            values=snapshot.values,
        )


def test_a_pin_the_store_does_not_hold_is_refused(trained: Trained) -> None:
    """Same id, other bytes: the retrained model that would otherwise inherit
    a promotion it never earned. And an id the store never recorded at all."""
    record = trained.record
    impostor = record.artifact_sha256[:16] + "0" * 48
    with pytest.raises(ModelError, match="evidence was produced with"):
        pipeline_from_spec(spec_reading(term(record.model_id, impostor)), models=trained.store)

    unknown = "ab" * 32
    with pytest.raises(ModelError, match="not in the model store"):
        pipeline_from_spec(spec_reading(term(model_id_for(unknown), unknown)), models=trained.store)


def test_a_scorer_is_fed_only_what_it_was_fitted_on(trained: Trained) -> None:
    scorer = trained.store.scorer_for(
        trained.record.model_id, artifact_sha256=trained.record.artifact_sha256
    )
    assert scorer.inputs == ("return_pct_2", "last_1")

    with pytest.raises(FeatureError, match="does not compute as features"):
        FeaturePipeline(specs=pipeline_for([("last", 1)]).specs, scorers=(scorer,))
    with pytest.raises(FeatureError, match="duplicate"):
        FeaturePipeline(
            specs=pipeline_for(FEATURES).specs,
            scorers=(ModelScorer(name="last_1", features=FEATURES, model=trained.model),),
        )
    with pytest.raises(ModelError, match="would be fed"):
        ModelScorer(name="mdl_x", features=tuple(reversed(FEATURES)), model=trained.model)


def test_an_edited_row_cannot_change_what_a_model_is_fed(
    trained: Trained,
    ledger_path: Path,
    tamper: Any,
) -> None:
    """The artifact is untouched and still verifies; only the catalog's
    lookbacks were lengthened. The event still says two bars, and wins."""
    tamper(
        ledger_path,
        "UPDATE ml_models SET features_json = replace(features_json, '2', '20') WHERE model_id = ?",
        (trained.record.model_id,),
    )
    with pytest.raises(ModelError, match="disagrees with the event"):
        pipeline_from_spec(spec_reading(pinned(trained)), models=trained.store)


# --------------------------------------------------------------------------
# Downstream: the search refuses it, the backtester just runs it
# --------------------------------------------------------------------------


def test_the_search_may_not_propose_a_spec_reading_a_model(trained: Trained) -> None:
    spec = spec_reading(pinned(trained))
    rejection = SpecValidator(limits=LIMITS).check(spec)
    assert rejection is not None
    assert rejection.code == "reads_a_model"
    assert trained.record.model_id in rejection.observed

    allowed = SpecValidator(limits=LIMITS, allow_models=True).check(spec)
    assert allowed is None or allowed.code != "reads_a_model"


def test_the_backtester_trades_on_the_score_with_no_model_code_of_its_own(
    trained: Trained,
) -> None:
    """Thresholds at the model's median score, so it enters and exits; the
    engine is handed a pipeline and a spec like any other."""
    scores = sorted(trained.model.predict(trained.data.rows))
    median = Decimal(scores[len(scores) // 2]).quantize(Decimal("0.0001"))
    spec = spec_reading(pinned(trained), enter=str(median), leave=str(median))
    engine = Backtester(
        cost_model=CostModel(LIMITS),
        pipeline=pipeline_from_spec(spec, models=trained.store),
        instruments={
            UID: InstrumentMeta(instrument_uid=UID, currency="USD", jurisdiction=Jurisdiction.US)
        },
        min_holding_minutes=spec.min_holding_minutes,
    )
    result = engine.run(
        strategy=DslStrategy(spec=spec, strategy_id="stg_ml"),
        reader=reader(trained.bars),
        decision_times=decisions_between(trained.bars),
    )
    assert result.n_signals > 0
    assert result.metrics.n_trades > 0
