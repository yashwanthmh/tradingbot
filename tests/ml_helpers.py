"""A recorded model and a spec that reads it, for suites that need one to exist.

Not a test module: funding, replay and the CLI suites each need "a promoted
spec that reads a model", and building it in each would be three copies of
the same fixture drifting apart. The model has learned exactly one thing — a
positive five-bar return pays — so on the loop fixtures' rising bars it scores
high and the spec enters, deterministically.
"""

from __future__ import annotations

import random
from datetime import UTC, datetime, timedelta
from decimal import Decimal

from tb.registry.model_store import ModelRecord, ModelStore
from tb.strategy.dsl.schema import StrategySpec
from tb.strategy.ml.cv import LabelSpan
from tb.strategy.ml.dataset import Dataset, LabelDefinition, Sample
from tb.strategy.ml.model import ModelParams, TrainedModel

MOMENTUM_FEATURES = (("return_pct", 5),)
_PARAMS = ModelParams(n_trees=30, min_data_in_leaf=10)


def momentum_model(store: ModelStore, *, uid: str = "isin:US0378331005") -> ModelRecord:
    """Record a model whose score is high exactly when `return_pct_5` is positive."""
    rng = random.Random(3)
    base = datetime(2025, 1, 6, 21, tzinfo=UTC)
    samples: list[Sample] = []
    for day in range(400):
        move = rng.uniform(-5, 5)
        result = Decimal("0.01") if move > 0 else Decimal("-0.01")
        decided = base + timedelta(days=day)
        samples.append(
            Sample(
                instrument_uid=uid,
                features=(move,),
                gross_return=result,
                net_return=result,
                span=LabelSpan(decided_at=decided, known_at=decided + timedelta(days=2)),
            )
        )
    data = Dataset(
        feature_names=("return_pct_5",),
        label=LabelDefinition(horizon=2, cost_bps=Decimal(0)),
        samples=tuple(samples),
        n_decisions=len(samples),
        n_unknown=0,
        n_unlabelled=0,
    )
    model = TrainedModel.fit(
        data.rows, data.labels, feature_names=data.feature_names, params=_PARAMS
    )
    return store.record(
        model,
        dataset=data,
        features=MOMENTUM_FEATURES,
        params=_PARAMS,
        vintage_id="vin_fixture",
        sealed_from=None,
    )


def model_spec(record: ModelRecord, *, edge: str = "450") -> StrategySpec:
    """Enter above 0.6, leave below 0.4, pinned to `record`."""
    term = {
        "kind": "model",
        "model_id": record.model_id,
        "artifact_sha256": record.artifact_sha256,
    }
    return StrategySpec.model_validate(
        {
            "name": "the momentum model",
            "entry": {
                "kind": "compare",
                "op": "gt",
                "left": term,
                "right": {"kind": "const", "value": "0.6"},
            },
            "exit": {
                "kind": "compare",
                "op": "lt",
                "left": term,
                "right": {"kind": "const", "value": "0.4"},
            },
            "expected_edge_bps": edge,
            "min_holding_minutes": 1440,
        }
    )
