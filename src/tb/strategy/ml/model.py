"""A model: deterministic gradient boosting, a text artifact, nothing that runs on load.

Three properties, each chosen against a specific way ML goes wrong in a system
that has to explain itself later.

**Deterministic.** One thread, fixed seed, no bagging or feature sampling, and
LightGBM's `deterministic` mode: the same rows train the same trees, byte for
byte. Without that the artifact hash identifies a training *run* rather than a
model, and a spec pinned to one could never be reproduced from its trial.

**A text artifact.** LightGBM's own model format, parsed on load. Not a pickle:
unpickling runs code, and the file a trading process reads is exactly the file
someone with access to the disk would edit. The artifact is also hashed and the
hash ledgered (`tb.registry.model_store`), so an edited file is refused before
it is parsed at all — the format is the second line of defence, not the first.

**Read by name.** The feature names are fitted into the artifact and checked on
load. A model reads its inputs by position, so a pipeline that produced the
same features in a different order would feed it a different question and get
a confident answer to it; comparing names turns that into a refusal.

The knobs are few and deliberately unremarkable. This layer's job is to be
checkable — trained without leakage, scored out of sample, pinned by hash —
not to win a tuning contest the multiplicity haircut would then have to pay for.

LightGBM is the optional `ml` extra and is imported only where a model is
fitted or parsed, so importing this module costs nothing. The live loop imports
the model store whether or not any funded strategy reads a model, and a trading
process must not need an ML stack to trade strategies that never touch one.
"""

from __future__ import annotations

import hashlib
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from tb.core.errors import TbError

if TYPE_CHECKING:
    import lightgbm as lgb

MODEL_KIND = "lightgbm-binary"


class ModelError(TbError):
    """A model could not be trained, loaded or asked for a score."""


def _require_lightgbm() -> None:
    """A named refusal, rather than an ImportError from inside a fit."""
    try:
        import lightgbm  # noqa: F401
    except ImportError as exc:  # pragma: no cover - depends on install extras
        raise ModelError(
            "models need LightGBM, the optional `ml` extra: `uv sync --extra ml`. Strategies "
            "that read no model trade without it."
        ) from exc


@dataclass(frozen=True, slots=True)
class ModelParams:
    """Every knob, each part of the model's identity through its artifact."""

    n_trees: int = 60
    learning_rate: float = 0.05
    num_leaves: int = 7
    max_depth: int = 3
    min_data_in_leaf: int = 20
    lambda_l2: float = 1.0
    seed: int = 7

    def lightgbm(self) -> dict[str, object]:
        return {
            "objective": "binary",
            "learning_rate": self.learning_rate,
            "num_leaves": self.num_leaves,
            "max_depth": self.max_depth,
            "min_data_in_leaf": self.min_data_in_leaf,
            "lambda_l2": self.lambda_l2,
            "seed": self.seed,
            # Determinism, spelled out: one thread, no row or column sampling,
            # and LightGBM's own deterministic mode on top.
            "num_threads": 1,
            "deterministic": True,
            "force_row_wise": True,
            "bagging_fraction": 1.0,
            "feature_fraction": 1.0,
            "feature_pre_filter": False,
            "verbose": -1,
        }

    def describe(self) -> dict[str, str]:
        return {
            "kind": MODEL_KIND,
            "n_trees": str(self.n_trees),
            "learning_rate": str(self.learning_rate),
            "num_leaves": str(self.num_leaves),
            "max_depth": str(self.max_depth),
            "min_data_in_leaf": str(self.min_data_in_leaf),
            "lambda_l2": str(self.lambda_l2),
            "seed": str(self.seed),
        }


@dataclass(frozen=True, slots=True)
class TrainedModel:
    """A fitted model and the artifact that is its identity."""

    artifact: str
    feature_names: tuple[str, ...]
    _booster: lgb.Booster = field(repr=False, compare=False)

    @classmethod
    def fit(
        cls,
        rows: Sequence[Sequence[float]],
        labels: Sequence[int],
        *,
        feature_names: Sequence[str],
        params: ModelParams,
    ) -> TrainedModel:
        if not rows:
            raise ModelError("there is nothing to fit a model to")
        if len(rows) != len(labels):
            raise ModelError(f"{len(rows)} rows against {len(labels)} labels")
        width = {len(row) for row in rows}
        if width != {len(feature_names)}:
            raise ModelError(
                f"rows of width {sorted(width)} against {len(feature_names)} feature names"
            )
        _require_lightgbm()
        import lightgbm as lgb
        import numpy as np

        data = lgb.Dataset(
            np.asarray(rows, dtype=np.float64),
            label=np.asarray(labels, dtype=np.float64),
            feature_name=list(feature_names),
            params={"verbose": -1},
            free_raw_data=True,
        )
        booster = lgb.train(params.lightgbm(), data, num_boost_round=params.n_trees)
        if tuple(booster.feature_name()) != tuple(feature_names):
            # LightGBM rewrites names it cannot store. Caught here, at training,
            # because the same mismatch at load would refuse a model that was
            # already recorded — and pinned by a spec — as unloadable.
            raise ModelError(
                f"LightGBM stored the features as {booster.feature_name()}, not "
                f"{list(feature_names)}: a model that cannot name its inputs cannot be checked"
            )
        return cls(
            artifact=booster.model_to_string(),
            feature_names=tuple(feature_names),
            _booster=booster,
        )

    @classmethod
    def from_artifact(cls, artifact: str, *, feature_names: Sequence[str]) -> TrainedModel:
        """Parse an artifact, refusing one fitted to other features.

        The hash has already been checked against the ledger by the store; this
        checks the one thing a hash cannot, that the caller means the same
        columns the model was fitted on.
        """
        _require_lightgbm()
        import lightgbm as lgb

        try:
            booster = lgb.Booster(model_str=artifact)
        except lgb.basic.LightGBMError as exc:
            raise ModelError(f"the artifact is not a model this build can read: {exc}") from exc
        fitted = tuple(booster.feature_name())
        if fitted != tuple(feature_names):
            raise ModelError(
                f"this model was fitted on {list(fitted)}, not {list(feature_names)}. It reads "
                "its inputs by position, so different or reordered features are not a smaller "
                "mistake than missing ones — they are a silently different question."
            )
        return cls(artifact=artifact, feature_names=fitted, _booster=booster)

    @property
    def sha256(self) -> str:
        return hashlib.sha256(self.artifact.encode("utf-8")).hexdigest()

    def predict(self, rows: Sequence[Sequence[float]]) -> list[float]:
        """The probability each row's label is 1: a profitable trade after costs."""
        if not rows:
            return []
        import numpy as np

        predicted = self._booster.predict(np.asarray(rows, dtype=np.float64))
        return [float(value) for value in predicted]

    def score(self, row: Sequence[float]) -> float:
        if len(row) != len(self.feature_names):
            raise ModelError(
                f"a row of {len(row)} value(s) for a model of {len(self.feature_names)} feature(s)"
            )
        return self.predict([row])[0]


@dataclass(frozen=True, slots=True)
class ModelScorer:
    """A loaded model as the pipeline sees it: a named score over named features.

    `name` is the key the score is read under — the model id a spec's `model`
    term names. `features` rebuilds the inputs as `(kind, lookback)`, from the
    record, and must name exactly the columns the model was fitted on, in the
    same order; anything else is refused here rather than scored.
    """

    name: str
    features: tuple[tuple[str, int], ...]
    model: TrainedModel

    def __post_init__(self) -> None:
        rebuilt = tuple(f"{kind}_{lookback}" for kind, lookback in self.features)
        if rebuilt != self.model.feature_names:
            raise ModelError(
                f"{self.name} would be fed {list(rebuilt)} but was fitted on "
                f"{list(self.model.feature_names)}"
            )

    @property
    def inputs(self) -> tuple[str, ...]:
        return self.model.feature_names

    def score(self, row: Sequence[float]) -> float:
        return self.model.score(row)
