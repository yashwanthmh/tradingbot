"""A model and its store: reproducible bytes, and nothing loaded that was not admitted.

The model half pins the three properties the rest of M7 leans on — the same
rows train the same artifact, a reloaded artifact predicts exactly as the
fitted one, and a model is refused the moment it is asked about columns it was
not fitted on. The store half attacks the one thing a trading process trusts,
the file on disk: edited, deleted, planted, or re-pointed by an edited catalog
row, each is refused by name before a single tree is parsed.
"""

from __future__ import annotations

import json
import random
import sqlite3
from collections.abc import Callable
from datetime import UTC, datetime, timedelta
from decimal import Decimal
from pathlib import Path
from typing import Any

import pytest

from tb.data.asof import ForwardOnlyReader, InMemoryBarSource
from tb.data.provider import Bar, Provenance, Resolution, Session
from tb.features.pipeline import pipeline_for
from tb.ledger.events import EventType
from tb.ledger.store import Ledger
from tb.ledger.verify import verify_chain
from tb.registry.model_store import ModelRecord, ModelStore, TrainingProvenance, model_id_for
from tb.research.holdout import decisions_between
from tb.strategy.ml.dataset import Dataset, LabelDefinition, build_dataset
from tb.strategy.ml.model import ModelError, ModelParams, TrainedModel

UID = "isin:US0378331005"
BASE = datetime(2026, 1, 5, tzinfo=UTC)
FEATURES = (("return_pct", 2), ("last", 1))
FEATURE_NAMES = pipeline_for(FEATURES).names
SMALL = ModelParams(n_trees=8, min_data_in_leaf=10)


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
    """A random walk: labels that go both ways, so there is something to fit."""
    rng = random.Random(seed)
    price = Decimal(100)
    bars: list[Bar] = []
    for offset in range(days):
        step = Decimal(str(round(rng.uniform(-0.03, 0.03), 6)))
        price = (price * (1 + step)).quantize(Decimal("0.01"))
        bars.append(daily(offset, price))
    return bars


def dataset(days: int = 120, *, seed: int = 7) -> Dataset:
    bars = walk(days, seed=seed)
    return build_dataset(
        reader=ForwardOnlyReader(
            source=InMemoryBarSource(bars=bars),
            resolution=Resolution.DAILY,
            instrument_uids=(UID,),
        ),
        pipeline=pipeline_for(FEATURES),
        decision_times=decisions_between(bars),
        instruments=(UID,),
        label=LabelDefinition(horizon=2, cost_bps=Decimal(0)),
    )


def fit(data: Dataset, params: ModelParams = SMALL) -> TrainedModel:
    return TrainedModel.fit(data.rows, data.labels, feature_names=data.feature_names, params=params)


@pytest.fixture
def store(ledger: Ledger, tmp_path: Path) -> ModelStore:
    return ModelStore(ledger, tmp_path / "models")


def record(store: ModelStore, data: Dataset, model: TrainedModel, **overrides: Any) -> ModelRecord:
    arguments: dict[str, Any] = {
        "dataset": data,
        "features": FEATURES,
        "params": SMALL,
        "vintage_id": "vin_test",
        "sealed_from": None,
        "metrics": {"oos_auc": 0.5, "n_folds": 4},
    }
    arguments.update(overrides)
    return store.record(model, **arguments)


# --------------------------------------------------------------------------
# The model
# --------------------------------------------------------------------------


def test_the_same_rows_train_the_same_bytes() -> None:
    """Without this the artifact hash names a training run, not a model, and a
    spec pinned to it could never be reproduced from its trial."""
    data = dataset()
    first, second = fit(data), fit(data)
    assert first.artifact == second.artifact
    assert first.sha256 == second.sha256
    assert fit(dataset(seed=8)).sha256 != first.sha256


def test_a_reloaded_model_predicts_exactly_as_the_fitted_one() -> None:
    data = dataset()
    model = fit(data)
    reloaded = TrainedModel.from_artifact(model.artifact, feature_names=data.feature_names)
    assert reloaded.predict(data.rows) == model.predict(data.rows)
    assert reloaded.sha256 == model.sha256


def test_a_planted_signal_is_learned() -> None:
    """The positive control: a label that is a feature, and a feature that is
    noise. A model that cannot find this cannot be trusted to report that it
    found nothing elsewhere."""
    rng = random.Random(3)
    rows = [[rng.random(), rng.random()] for _ in range(600)]
    labels = [1 if signal > 0.5 else 0 for signal, _ in rows]
    model = TrainedModel.fit(rows, labels, feature_names=("signal", "noise"), params=ModelParams())
    assert model.score([0.9, 0.2]) > 0.9
    assert model.score([0.1, 0.8]) < 0.1


def test_a_model_asked_about_other_columns_refuses() -> None:
    """It reads by position: reordered names are a different question, not a
    smaller mistake than missing ones."""
    data = dataset()
    model = fit(data)
    with pytest.raises(ModelError, match="fitted on"):
        TrainedModel.from_artifact(model.artifact, feature_names=tuple(reversed(FEATURE_NAMES)))
    with pytest.raises(ModelError, match="a row of 1 value"):
        model.score([1.0])
    with pytest.raises(ModelError, match="not a model this build can read"):
        TrainedModel.from_artifact("tree\nnot a model", feature_names=FEATURE_NAMES)


def test_rows_and_labels_must_agree() -> None:
    with pytest.raises(ModelError, match="nothing to fit"):
        TrainedModel.fit([], [], feature_names=FEATURE_NAMES, params=SMALL)
    with pytest.raises(ModelError, match="2 rows against 1 labels"):
        TrainedModel.fit([[1.0, 2.0], [3.0, 4.0]], [1], feature_names=FEATURE_NAMES, params=SMALL)
    with pytest.raises(ModelError, match="width"):
        TrainedModel.fit([[1.0]], [1], feature_names=FEATURE_NAMES, params=SMALL)


# --------------------------------------------------------------------------
# The store
# --------------------------------------------------------------------------


def test_recording_writes_one_file_one_event_one_row(ledger: Ledger, store: ModelStore) -> None:
    data = dataset()
    model = fit(data)
    recorded = record(store, data, model, search_id="srch_1")

    assert recorded.model_id == model_id_for(model.sha256)
    assert recorded.feature_names == FEATURE_NAMES
    assert recorded.features == FEATURES
    assert recorded.n_samples == len(data.samples)
    assert recorded.trained_through == max(s.span.known_at for s in data.samples)
    assert recorded.label["horizon_bars"] == "2"
    assert recorded.metrics == {"oos_auc": 0.5, "n_folds": 4}
    assert recorded.search_id == "srch_1"
    assert [p.name for p in store.root.iterdir()] == [f"{model.sha256}.txt"]

    events = list(ledger.iter_events(event_type=EventType.MODEL_RECORDED))
    assert len(events) == 1
    assert json.loads(events[0]["payload_json"])["artifact_sha256"] == model.sha256
    assert events[0]["aggregate_type"] == "model"
    assert verify_chain(ledger).ok

    loaded = store.load(recorded.model_id)
    assert loaded.predict(data.rows) == model.predict(data.rows)


def test_the_same_bytes_are_one_model(ledger: Ledger, store: ModelStore) -> None:
    """Content-addressed: a second recording of an identical model — even with
    other metadata — returns the first, whose provenance stands."""
    data = dataset()
    first = record(store, data, fit(data))
    again = record(store, data, fit(data), search_id="srch_other")
    assert again == first
    assert again.search_id is None
    assert len(list(ledger.iter_events(event_type=EventType.MODEL_RECORDED))) == 1
    assert len(store.records()) == 1


def test_an_edited_artifact_is_refused(store: ModelStore) -> None:
    data = dataset()
    recorded = record(store, data, fit(data))
    path = store.root / recorded.relative_path
    # One leaf value nudged: a model that still parses, and trades differently.
    text = path.read_text()
    marker = "leaf_value="
    at = text.index(marker) + len(marker)
    path.write_text(text[:at] + "9" + text[at:])

    with pytest.raises(ModelError, match="has changed since it was admitted"):
        store.load(recorded.model_id)
    # Nor does recording the model again paper over it: the evidence stays.
    with pytest.raises(ModelError, match="has changed since it was admitted"):
        record(store, data, fit(data))


def test_a_missing_artifact_is_refused_and_restored_only_from_the_same_bytes(
    store: ModelStore,
) -> None:
    data = dataset()
    model = fit(data)
    recorded = record(store, data, model)
    (store.root / recorded.relative_path).unlink()

    with pytest.raises(ModelError, match="missing from"):
        store.load(recorded.model_id)
    record(store, data, model)
    assert store.load(recorded.model_id).sha256 == model.sha256


def test_a_file_the_ledger_never_admitted_is_not_a_model(store: ModelStore) -> None:
    """A directory listing is not a catalog: an artifact dropped next to the
    admitted ones, under the name its hash would give it, is never loaded."""
    model = fit(dataset())
    store.root.mkdir(parents=True)
    (store.root / f"{model.sha256}.txt").write_text(model.artifact)
    with pytest.raises(ModelError, match="not in the model store"):
        store.load(model_id_for(model.sha256))


def plant(store: ModelStore, *, seed: int) -> str:
    """A real, loadable model written straight into the store directory,
    under its own hash, without ever being recorded. Returns its hash."""
    planted = fit(dataset(seed=seed))
    (store.root / f"{planted.sha256}.txt").write_text(planted.artifact)
    return planted.sha256


def test_an_edited_catalog_row_cannot_repoint_a_model(
    ledger_path: Path,
    store: ModelStore,
    tamper: Callable[[Path, str, tuple[Any, ...]], None],
) -> None:
    """The row is a projection. Pointing it at a planted artifact — a model
    that parses and predicts — is caught against the event that recorded the
    original. (Pointing it at another *recorded* model is refused earlier
    still, by the uniqueness of `artifact_sha256`.)"""
    data = dataset()
    recorded = record(store, data, fit(data))
    planted = plant(store, seed=13)
    tamper(
        ledger_path,
        "UPDATE ml_models SET artifact_sha256 = ? WHERE model_id = ?",
        (planted, recorded.model_id),
    )
    with pytest.raises(ModelError, match="disagrees with the event"):
        store.load(recorded.model_id)


def test_an_edited_event_is_caught_without_walking_the_chain(
    ledger_path: Path,
    store: ModelStore,
    tamper: Callable[[Path, str, tuple[Any, ...]], None],
) -> None:
    """Editing the event and the row together is the next attack; the event no
    longer hashes to what is stored on it. (Re-signing the whole chain as well
    is what external anchors exist to catch.)"""
    data = dataset()
    recorded = record(store, data, fit(data))
    planted = plant(store, seed=13)
    tamper(
        ledger_path,
        "UPDATE event_log SET payload_json = replace(payload_json, ?, ?) WHERE seq = ?",
        (recorded.artifact_sha256, planted, recorded.recording_event_seq),
    )
    tamper(
        ledger_path,
        "UPDATE ml_models SET artifact_sha256 = ? WHERE model_id = ?",
        (planted, recorded.model_id),
    )
    with pytest.raises(ModelError, match="does not hash to its stored values"):
        store.load(recorded.model_id)


def test_a_model_that_saw_the_holdout_is_never_recorded(store: ModelStore) -> None:
    """A seal at the last label's completion: that label's exit price is the
    first holdout price, and the model has learned from it."""
    data = dataset()
    last_known = max(s.span.known_at for s in data.samples)
    with pytest.raises(ModelError, match="seen the holdout"):
        record(store, data, fit(data), sealed_from=last_known)
    assert not store.root.exists(), "refused before anything touched the disk"
    recorded = record(store, data, fit(data), sealed_from=last_known + timedelta(seconds=1))
    assert recorded.sealed_from == last_known + timedelta(seconds=1)


def test_the_provenance_is_the_datasets_own() -> None:
    data = dataset()
    provenance = TrainingProvenance.of(data, vintage_id="vin_test", sealed_from=None)
    assert provenance.window_start == data.samples[0].decided_at
    assert provenance.window_end == data.samples[-1].decided_at
    assert provenance.trained_through > provenance.window_end
    with pytest.raises(ModelError, match="no labelled samples"):
        TrainingProvenance.of(
            Dataset(
                feature_names=data.feature_names,
                label=data.label,
                samples=(),
                n_decisions=0,
                n_unknown=0,
                n_unlabelled=0,
            ),
            vintage_id="vin_test",
            sealed_from=None,
        )


def test_a_record_must_rebuild_the_columns_the_model_reads(store: ModelStore) -> None:
    data = dataset()
    model = fit(data)
    with pytest.raises(ModelError, match="rebuilds other columns"):
        record(store, data, model, features=tuple(reversed(FEATURES)))
    with pytest.raises(ModelError, match="cannot be computed"):
        record(store, data, model, features=(("clairvoyance", 2), ("last", 1)))
    assert not store.root.exists()


def test_an_undefined_metric_is_none_not_nan(store: ModelStore) -> None:
    data = dataset()
    with pytest.raises(ModelError, match="as None"):
        record(store, data, fit(data), metrics={"oos_auc": float("nan")})
    recorded = record(store, data, fit(data), metrics={"oos_auc": None})
    assert recorded.metrics == {"oos_auc": None}


def test_the_catalog_survives_a_reopen(ledger_path: Path, tmp_path: Path) -> None:
    data = dataset()
    model = fit(data)
    with Ledger(ledger_path) as ledger:
        ledger.initialise(created_by="test")
        recorded = record(ModelStore(ledger, tmp_path / "models"), data, model)
    with Ledger(ledger_path) as reopened:
        again = ModelStore(reopened, tmp_path / "models")
        assert again.get(recorded.model_id) == recorded
        assert again.load(recorded.model_id).sha256 == model.sha256


def test_no_model_table_row_without_its_event(ledger_path: Path, store: ModelStore) -> None:
    """Every catalog row names the event that wrote it, and that event names it."""
    data = dataset()
    recorded = record(store, data, fit(data))
    conn = sqlite3.connect(ledger_path)
    try:
        (event_type, aggregate_id) = conn.execute(
            "SELECT event_type, aggregate_id FROM event_log WHERE seq = ?",
            (recorded.recording_event_seq,),
        ).fetchone()
    finally:
        conn.close()
    assert (event_type, aggregate_id) == ("model.recorded", recorded.model_id)
