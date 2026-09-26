"""The model store: artifacts on disk by hash, admitted only by the ledger.

The bar store's arrangement, for the bar store's reasons. An artifact is written
to a temporary name, fsynced, renamed to `<sha256>.txt`, and only *then*
recorded — the `model.recorded` event and its catalog row in one transaction.
The event is what makes a model exist:

* A file with no event is never loaded, because loading starts from the
  ledger, never from a directory listing. A crash between the rename and the
  append leaves harmless garbage.
* An event whose file has vanished or changed is refused at load, with the two
  hashes named. The file a trading process reads is the file anyone with disk
  access could edit, and a model is code in all but name: a swapped artifact
  would trade on logic nobody reviewed, under a hash that still reads as
  approved.

**The pin is checked against the chained event, not the catalog row.** The
row is a projection and can be edited; the event is hash-chained. So a load
reads the recording event, checks that its payload and chain hashes are the
ones stored on it, and that it names the same artifact as the row. Continuity
of the whole chain is `tb ledger verify`'s job; this is the check that makes
the one fact a model load depends on local and cheap.

**The same bytes are the same model.** `model_id` is derived from the artifact
hash, so recording a model twice returns the first record — its provenance
stands — rather than inventing a second identity for identical trees. That is
the same reason the strategy registry dedupes on `spec_hash`: two names for one
thing inflate every count the multiplicity haircut divides by.

Here rather than beside the model in `tb.strategy.ml`, because this module
writes files and `tb.strategy` is the interpretation path: no module under it
may import `os`, and `test_dsl` walks every file there to keep it that way.
Computing with a model is strategy code; storing one is registry code, as it
is for specs.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
import sqlite3
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

from tb.core.clock import from_iso, now_utc, to_iso
from tb.core.ids import new_id
from tb.features.pipeline import FeatureError, pipeline_for
from tb.ledger.chain import compute_chain_hash, compute_payload_hash
from tb.ledger.events import Actor, EventType, ModelRecordedPayload
from tb.ledger.store import Ledger
from tb.strategy.ml.dataset import Dataset
from tb.strategy.ml.model import MODEL_KIND, ModelError, ModelParams, ModelScorer, TrainedModel

_SHA256 = re.compile(r"[0-9a-f]{64}")
_ARTIFACT_SUFFIX = ".txt"

# A metric is a number or a short fact about one. Anything else is a payload
# nobody can query, and a non-finite float cannot be hashed into the chain.
MetricValue = float | int | str | bool | None


def model_id_for(artifact_sha256: str) -> str:
    """The model's identity, derived from its bytes."""
    return f"mdl_{artifact_sha256[:16]}"


def default_model_root(ledger_path: Path | str) -> Path:
    """Where artifacts live unless told otherwise: beside the ledger, as bars do."""
    return Path(ledger_path).parent / "models"


@dataclass(frozen=True, slots=True)
class TrainingProvenance:
    """Where a model's training rows came from, and the latest thing they knew."""

    vintage_id: str
    window_start: datetime
    window_end: datetime
    trained_through: datetime
    n_samples: int
    base_rate: float | None
    sealed_from: datetime | None = None

    @classmethod
    def of(
        cls, dataset: Dataset, *, vintage_id: str, sealed_from: datetime | None
    ) -> TrainingProvenance:
        """A dataset's window and horizon, as they are recorded against its model.

        `trained_through` is the latest instant any label was knowable, not the
        last decision time: a label decided in March and completed in April
        carries April's prices, and it is April that has to precede the seal.
        """
        if not dataset.samples:
            raise ModelError("a dataset with no labelled samples trains nothing")
        return cls(
            vintage_id=vintage_id,
            window_start=dataset.samples[0].decided_at,
            window_end=dataset.samples[-1].decided_at,
            trained_through=max(sample.span.known_at for sample in dataset.samples),
            n_samples=len(dataset.samples),
            base_rate=dataset.base_rate,
            sealed_from=sealed_from,
        )


@dataclass(frozen=True, slots=True)
class ModelRecord:
    """A recorded model, as its catalog row and recording event describe it."""

    model_id: str
    artifact_sha256: str
    kind: str
    relative_path: str
    byte_size: int
    feature_names: tuple[str, ...]
    # `(kind, lookback)` in the order the model reads them: enough to rebuild
    # the pipeline that computes its inputs, which is what a spec reading the
    # model needs at every decision.
    features: tuple[tuple[str, int], ...]
    label: Mapping[str, str]
    params: Mapping[str, str]
    vintage_id: str
    sealed_from: datetime | None
    window_start: datetime
    window_end: datetime
    trained_through: datetime
    n_samples: int
    base_rate: float | None
    metrics: Mapping[str, Any]
    search_id: str | None
    recorded_at: datetime
    recording_event_seq: int


class ModelStore:
    """Model artifacts under `root`, catalogued in the ledger."""

    def __init__(self, ledger: Ledger, root: Path, *, run_id: str | None = None) -> None:
        self._ledger = ledger
        self._root = Path(root)
        self._run_id = run_id

    @property
    def root(self) -> Path:
        return self._root

    # -- recording ---------------------------------------------------------

    def record(
        self,
        model: TrainedModel,
        *,
        dataset: Dataset,
        features: Sequence[tuple[str, int]],
        params: ModelParams,
        vintage_id: str,
        sealed_from: datetime | None,
        metrics: Mapping[str, MetricValue] | None = None,
        search_id: str | None = None,
        actor: Actor = Actor.SYSTEM,
        at: datetime | None = None,
    ) -> ModelRecord:
        """Admit a model trained on `dataset`, or return the record of the same bytes.

        Refused before anything touches the disk if the record would describe
        a different model from the one it names: features that do not rebuild
        the columns the model was fitted on, a dataset of other columns, or
        training labels known at or after the seal the data was read under.
        The window and horizon are taken from the dataset rather than passed
        alongside it, so they cannot describe some other set of rows.
        """
        provenance = TrainingProvenance.of(dataset, vintage_id=vintage_id, sealed_from=sealed_from)
        requests = tuple((str(kind), int(lookback)) for kind, lookback in features)
        try:
            names = pipeline_for(requests).names
        except FeatureError as exc:
            raise ModelError(f"the model's features cannot be computed: {exc}") from exc
        if names != model.feature_names or dataset.feature_names != model.feature_names:
            raise ModelError(
                f"the model reads {list(model.feature_names)}, but the record would rebuild "
                f"{list(names)} from a dataset of {list(dataset.feature_names)}. A record that "
                "rebuilds other columns hands the model a different question at every decision."
            )
        _check_provenance(provenance)
        clean = _check_metrics(metrics or {})

        data = model.artifact.encode("utf-8")
        sha = hashlib.sha256(data).hexdigest()
        model_id = model_id_for(sha)

        existing = self.get(model_id)
        if existing is not None:
            self._restore(existing, data)
            return existing

        self._write(sha, data)
        moment = at or now_utc()
        payload = ModelRecordedPayload(
            model_id=model_id,
            artifact_sha256=sha,
            kind=MODEL_KIND,
            relative_path=_relative(sha),
            byte_size=len(data),
            feature_names=list(names),
            features=[{"kind": kind, "lookback": lookback} for kind, lookback in requests],
            label=dataset.label.describe(),
            params=params.describe(),
            vintage_id=provenance.vintage_id,
            sealed_from=None if provenance.sealed_from is None else to_iso(provenance.sealed_from),
            window_start=to_iso(provenance.window_start),
            window_end=to_iso(provenance.window_end),
            trained_through=to_iso(provenance.trained_through),
            n_samples=provenance.n_samples,
            base_rate=provenance.base_rate,
            metrics=dict(clean),
            search_id=search_id,
        )
        with self._ledger.transaction() as tx:
            # Re-checked inside the write lock: two processes recording the same
            # bytes at once must produce one event, not a second with no row.
            taken = tx.execute("SELECT 1 FROM ml_models WHERE model_id = ?", (model_id,))
            if taken.fetchone() is None:
                event = tx.append(
                    EventType.MODEL_RECORDED,
                    model_id,
                    payload,
                    actor=actor,
                    run_id=self._run_id,
                )
                tx.execute(
                    """
                    INSERT INTO ml_models (
                        model_id, artifact_sha256, kind, relative_path, byte_size,
                        feature_names_json, features_json, label_json, params_json,
                        vintage_id, sealed_from, window_start, window_end, trained_through,
                        n_samples, base_rate, metrics_json, search_id, recorded_at,
                        recording_event_seq
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        model_id,
                        sha,
                        payload.kind,
                        payload.relative_path,
                        payload.byte_size,
                        json.dumps(payload.feature_names),
                        json.dumps(payload.features),
                        json.dumps(payload.label, sort_keys=True),
                        json.dumps(payload.params, sort_keys=True),
                        payload.vintage_id,
                        payload.sealed_from,
                        payload.window_start,
                        payload.window_end,
                        payload.trained_through,
                        payload.n_samples,
                        payload.base_rate,
                        json.dumps(payload.metrics, sort_keys=True),
                        search_id,
                        to_iso(moment),
                        event.seq,
                    ),
                )
        recorded = self.get(model_id)
        if recorded is None:  # pragma: no cover - the row was written or already there
            raise ModelError(f"{model_id} was recorded but cannot be read back")
        return recorded

    # -- reading -----------------------------------------------------------

    def get(self, model_id: str) -> ModelRecord | None:
        row: sqlite3.Row | None = self._ledger.conn.execute(
            "SELECT * FROM ml_models WHERE model_id = ?", (model_id,)
        ).fetchone()
        return None if row is None else _row_to_record(row)

    def records(self) -> list[ModelRecord]:
        rows = self._ledger.conn.execute(
            "SELECT * FROM ml_models ORDER BY recorded_at, model_id"
        ).fetchall()
        return [_row_to_record(row) for row in rows]

    def load(self, model_id: str) -> TrainedModel:
        """The recorded model, after its bytes are checked against the ledger.

        Every check runs before the artifact is parsed: an unrecorded id, a
        catalog row that disagrees with its chained event, a missing file, and a
        file that no longer hashes to the recorded value are each refused by
        name. The format is not a pickle, so parsing runs no code — but a
        tampered model is refused for being tampered, not for failing to parse.
        """
        record = self._require(model_id)
        self._check_event(record)
        data = self._read(record)
        return TrainedModel.from_artifact(data.decode("utf-8"), feature_names=record.feature_names)

    def scorer_for(self, model_id: str, *, artifact_sha256: str) -> ModelScorer:
        """The recorded model as a pipeline scorer, if it is the one pinned.

        The pin is compared before anything is loaded. A spec is trained,
        backtested and gated against one artifact; a store that now holds a
        different one under the same id would hand the loop a model none of
        that evidence was produced with.
        """
        record = self._require(model_id)
        if record.artifact_sha256 != artifact_sha256:
            raise ModelError(
                f"the spec pins {model_id} to artifact {artifact_sha256}, but the store "
                f"records {record.artifact_sha256}. A strategy trades only the model its "
                "evidence was produced with."
            )
        return ModelScorer(name=model_id, features=record.features, model=self.load(model_id))

    # -- internals ---------------------------------------------------------

    def _require(self, model_id: str) -> ModelRecord:
        record = self.get(model_id)
        if record is None:
            raise ModelError(
                f"{model_id} is not in the model store. A model exists when the ledger "
                "records it; a file in the directory is not a model."
            )
        return record

    def _path(self, sha: str) -> Path:
        # Built from the hash rather than read from the row: the row's
        # `relative_path` is editable, and a path is where a traversal lives.
        if not _SHA256.fullmatch(sha):
            raise ModelError(f"{sha!r} is not a sha256 hex digest")
        return self._root / _relative(sha)

    def _write(self, sha: str, data: bytes) -> None:
        self._root.mkdir(parents=True, exist_ok=True)
        final = self._path(sha)
        temp = self._root / f".tmp-{new_id('mdl', length=12)}{_ARTIFACT_SUFFIX}"
        with open(temp, "wb") as handle:
            handle.write(data)
            handle.flush()
            # fsync before the rename, or a crash can leave a renamed-but-empty
            # file under a name that claims a hash it does not have.
            os.fsync(handle.fileno())
        os.replace(temp, final)
        _fsync_dir(self._root)

    def _restore(self, record: ModelRecord, data: bytes) -> None:
        """Re-materialise a recorded artifact whose file has gone missing.

        Safe because `data` hashes to the value the ledger already holds: this
        restores the admitted bytes and nothing else. A file that is present
        but *different* is not overwritten — that is evidence, and the right
        response to it is a refusal with both hashes, not a quiet repair.
        """
        path = self._path(record.artifact_sha256)
        if path.exists():
            self._read(record)
            return
        self._write(record.artifact_sha256, data)

    def _read(self, record: ModelRecord) -> bytes:
        path = self._path(record.artifact_sha256)
        if not path.exists():
            raise ModelError(
                f"{record.model_id} is recorded but its artifact {record.relative_path} is "
                f"missing from {self._root}. The ledger admits a model we do not have."
            )
        data = path.read_bytes()
        actual = hashlib.sha256(data).hexdigest()
        if actual != record.artifact_sha256:
            raise ModelError(
                f"{record.model_id}'s artifact hashes to {actual}, not the recorded "
                f"{record.artifact_sha256}: the file has changed since it was admitted. "
                "Refusing to load it — a swapped model trades on logic nobody reviewed."
            )
        return data

    def _check_event(self, record: ModelRecord) -> None:
        row = self._ledger.get(record.recording_event_seq)
        if row is None or str(row["event_type"]) != EventType.MODEL_RECORDED.value:
            raise ModelError(
                f"{record.model_id}'s catalog row points at event "
                f"{record.recording_event_seq}, which is not the event that recorded it"
            )
        payload_json = str(row["payload_json"])
        chain_hash = compute_chain_hash(
            prev_hash=str(row["prev_hash"]),
            seq=int(row["seq"]),
            ts_utc=str(row["ts_utc"]),
            event_type=str(row["event_type"]),
            aggregate_type=str(row["aggregate_type"]),
            aggregate_id=str(row["aggregate_id"]),
            actor=str(row["actor"]),
            payload_hash=str(row["payload_hash"]),
        )
        payload_intact = compute_payload_hash(payload_json) == str(row["payload_hash"])
        if not payload_intact or chain_hash != str(row["chain_hash"]):
            raise ModelError(
                f"the event recording {record.model_id} (seq {row['seq']}) does not hash to "
                "its stored values. Run `tb ledger verify`: the chain has been edited."
            )
        payload = json.loads(payload_json)
        # The features too, not only the hash: they decide what the model is
        # fed, and a row edited to a longer lookback under the same column
        # names would feed a faithful artifact a different question.
        recorded_features = tuple(
            (str(item.get("kind")), item.get("lookback")) for item in payload.get("features", ())
        )
        if (
            payload.get("model_id") != record.model_id
            or payload.get("artifact_sha256") != record.artifact_sha256
            or tuple(payload.get("feature_names", ())) != record.feature_names
            or recorded_features != record.features
        ):
            raise ModelError(
                f"{record.model_id}'s catalog row disagrees with the event that recorded it "
                f"(seq {row['seq']}): the row names artifact {record.artifact_sha256} fed "
                f"{list(record.features)}, the event {payload.get('artifact_sha256')} fed "
                f"{list(recorded_features)}. The event is the fact."
            )


def _relative(sha: str) -> str:
    return f"{sha}{_ARTIFACT_SUFFIX}"


def _check_provenance(provenance: TrainingProvenance) -> None:
    if provenance.n_samples < 1:
        raise ModelError("a model trained on no samples is not a model")
    if provenance.window_end < provenance.window_start:
        raise ModelError("a training window that ends before it starts")
    if provenance.trained_through < provenance.window_end:
        raise ModelError("labels cannot be known before the decisions they label")
    if provenance.sealed_from is not None and provenance.trained_through >= provenance.sealed_from:
        raise ModelError(
            f"training labels were known through {to_iso(provenance.trained_through)}, at or "
            f"past the seal at {to_iso(provenance.sealed_from)}. The model has seen the "
            "holdout, and would carry it into every spec that reads it."
        )


def _check_metrics(metrics: Mapping[str, MetricValue]) -> dict[str, MetricValue]:
    clean: dict[str, MetricValue] = {}
    for name, value in metrics.items():
        if value is not None and not isinstance(value, float | int | str | bool):
            raise ModelError(f"metric {name!r} is a {type(value).__name__}, not a number")
        if isinstance(value, float) and not math.isfinite(value):
            raise ModelError(
                f"metric {name!r} is {value}: record an undefined metric as None, because a "
                "non-finite number cannot be hashed into the ledger and reads as a result"
            )
        clean[str(name)] = value
    return clean


def _fsync_dir(directory: Path) -> None:
    """Durably record the rename, not just the file contents."""
    try:
        fd = os.open(directory, os.O_RDONLY)
    except OSError:  # pragma: no cover - platform dependent
        return
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def _row_to_record(row: sqlite3.Row) -> ModelRecord:
    features = json.loads(str(row["features_json"]))
    sealed = row["sealed_from"]
    return ModelRecord(
        model_id=str(row["model_id"]),
        artifact_sha256=str(row["artifact_sha256"]),
        kind=str(row["kind"]),
        relative_path=str(row["relative_path"]),
        byte_size=int(row["byte_size"]),
        feature_names=tuple(str(name) for name in json.loads(str(row["feature_names_json"]))),
        features=tuple((str(item["kind"]), int(item["lookback"])) for item in features),
        label=json.loads(str(row["label_json"])),
        params=json.loads(str(row["params_json"])),
        vintage_id=str(row["vintage_id"]),
        sealed_from=None if sealed is None else from_iso(str(sealed)),
        window_start=from_iso(str(row["window_start"])),
        window_end=from_iso(str(row["window_end"])),
        trained_through=from_iso(str(row["trained_through"])),
        n_samples=int(row["n_samples"]),
        base_rate=None if row["base_rate"] is None else float(row["base_rate"]),
        metrics=json.loads(str(row["metrics_json"] or "{}")),
        search_id=None if row["search_id"] is None else str(row["search_id"]),
        recorded_at=from_iso(str(row["recorded_at"])),
        recording_event_seq=int(row["recording_event_seq"]),
    )


__all__ = [
    "MetricValue",
    "ModelRecord",
    "ModelStore",
    "TrainingProvenance",
    "model_id_for",
]
