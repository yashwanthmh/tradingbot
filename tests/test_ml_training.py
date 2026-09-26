"""The trainer end to end: sealed vintage in, recorded model and counted trials out.

Driven through the same stores, ledger and research cycle the CLI uses, on two
vintages: the fixtures' random walk, where there is nothing to learn, and a
sawtooth — five sessions up, five down — where there plainly is. The second is
the positive control the first needs: a trainer that registered nothing on a
random walk has only shown something if it registers a spec when the pattern is
real.
"""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import timedelta
from decimal import Decimal
from pathlib import Path
from typing import Any

import pytest

from tb.config.hard_limits import HardLimits
from tb.config.loader import load_hard_limits
from tb.data.barstore import BarStore
from tb.data.provider import Bar, BarBatch, Provenance, Resolution, Session
from tb.data.snapshot import SnapshotStore
from tb.ledger.events import EventType
from tb.ledger.store import Ledger
from tb.ledger.verify import verify_chain
from tb.registry.lineage import SpecRegistry
from tb.registry.model_store import ModelStore
from tb.registry.models import AuthorKind
from tb.research import ml
from tb.research.ml import ModelTrainer, TrainingConfig, TrainingError
from tb.strategy.dsl.ops import pipeline_from_spec
from tb.strategy.ml.dataset import LabelDefinition
from tb.strategy.ml.model import ModelParams
from tests.test_cli_registry import BASE, UID, _init, _out, _run, seed_bars

CONFIG = TrainingConfig(
    features=(("return_pct", 2), ("return_pct", 4), ("zscore", 10)),
    label=LabelDefinition(horizon=2, cost_bps=Decimal("40")),
    params=ModelParams(n_trees=40, min_data_in_leaf=10),
    n_folds=3,
    min_train=60,
)


def seed_sawtooth(env: dict[str, Any], *, days: int = 500) -> None:
    """Five sessions up 1%, five down, repeated: learnable from recent returns."""
    pinned = load_hard_limits(env["limits"])
    with _open(env) as (ledger, _):
        store = BarStore(ledger, root=env["bars"], scale=pinned.limits.data.price_scale)
        price = Decimal("100.00")
        bars: list[Bar] = []
        for day in range(days):
            step = Decimal("1.01") if (day // 5) % 2 == 0 else Decimal("0.99")
            price = (price * step).quantize(Decimal("0.01"))
            opened = BASE + timedelta(days=day)
            bars.append(
                Bar(
                    instrument_uid=UID,
                    resolution=Resolution.DAILY,
                    bar_open_utc=opened,
                    available_at_utc=opened + timedelta(days=1),
                    ingested_at_utc=opened + timedelta(days=1),
                    provider="fixture",
                    provenance=Provenance.BACKFILL,
                    session=Session.REGULAR,
                    open=price,
                    high=price + Decimal("0.5"),
                    low=price - Decimal("0.5"),
                    close=price,
                    volume=1_000_000,
                )
            )
        store.ingest(
            BarBatch(
                bars=tuple(bars),
                provider="fixture",
                symbol=UID,
                resolution=Resolution.DAILY,
                requested_start=bars[0].bar_open_utc,
                requested_end=bars[-1].bar_open_utc,
            )
        )
        store.compact()


def sealed(env: dict[str, Any], *, pattern: bool) -> str:
    _init(env)
    if pattern:
        seed_sawtooth(env)
    else:
        seed_bars(env)
    result = _run(["data", "seal", "--resolution", "daily", *env["bar_args"]])
    assert result.exit_code == 0, _out(result)
    with _open(env) as (ledger, _):
        row = ledger.conn.execute("SELECT vintage_id FROM data_snapshots").fetchone()
    return str(row["vintage_id"])


@contextmanager
def _open(env: dict[str, Any]) -> Iterator[tuple[Ledger, HardLimits]]:
    pinned = load_hard_limits(env["limits"])
    with Ledger(env["db"], config_hash=pinned.config_hash).open() as ledger:
        yield ledger, pinned.limits


@dataclass(frozen=True, slots=True)
class Stores:
    ledger: Ledger
    limits: HardLimits
    snapshots: SnapshotStore
    models: ModelStore
    trainer: ModelTrainer


@contextmanager
def stores(env: dict[str, Any]) -> Iterator[Stores]:
    with _open(env) as (ledger, limits):
        snapshots = SnapshotStore(
            ledger, BarStore(ledger, root=env["bars"], scale=limits.data.price_scale)
        )
        models = ModelStore(ledger, Path(env["bars"]).parent / "models")
        yield Stores(
            ledger=ledger,
            limits=limits,
            snapshots=snapshots,
            models=models,
            trainer=ModelTrainer(ledger, limits=limits, snapshots=snapshots, models=models),
        )


def trials_of(ledger: Ledger, search_id: str) -> list[Any]:
    return list(
        ledger.conn.execute(
            "SELECT * FROM trials WHERE search_id = ? ORDER BY recorded_at", (search_id,)
        ).fetchall()
    )


# --------------------------------------------------------------------------
# The positive control
# --------------------------------------------------------------------------


def test_a_learnable_pattern_is_learned_and_a_spec_reading_it_registered(
    cli_env: dict[str, Any],
) -> None:
    vintage_id = sealed(cli_env, pattern=True)
    with stores(cli_env) as s:
        report = s.trainer.run(vintage_id=vintage_id, config=CONFIG)

        assert report.oos.auc is not None and report.oos.auc > 0.8
        assert not report.null.shows_skill()
        record = report.record
        assert record.search_id == report.cycle.search_id
        assert record.metrics["oos_auc"] == pytest.approx(report.oos.auc, abs=1e-6)
        assert "null_auc" in record.metrics
        assert record.trained_through < report.cycle.window.sealed_from

        # Every threshold is a trial, all in the idea's one lineage, all ML.
        rows = trials_of(s.ledger, report.cycle.search_id)
        assert len(rows) == len(report.thresholds) >= 1
        assert {str(row["author_kind"]) for row in rows} == {AuthorKind.ML.value}
        assert {str(row["lineage_id"]) for row in rows} == {CONFIG.lineage_id}

        # A survivor was registered, pinned to the recorded model, and its
        # pipeline builds from the store.
        assert report.cycle.registered, report.explain()
        registry = SpecRegistry(
            s.ledger, per_lineage_budget_ccy=s.limits.loss.per_lineage_budget_ccy
        )
        for registered in report.cycle.registered:
            assert registered.author_kind is AuthorKind.ML
            assert registered.lineage_id == CONFIG.lineage_id
            spec = registry.spec_of(registered.strategy_id, registered.version)
            assert spec is not None
            (ref,) = spec.model_refs
            assert (ref.model_id, ref.artifact_sha256) == (record.model_id, record.artifact_sha256)
            pipeline = pipeline_from_spec(spec, models=s.models)
            assert record.model_id in pipeline.names

        # Judged walk-forward: each evaluated trial traded, and only after the
        # first fold began.
        evaluated = [row for row in rows if str(row["outcome"]) == "evaluated"]
        assert evaluated and all(int(row["n_trades"]) > 0 for row in evaluated)
        assert verify_chain(s.ledger).ok


# --------------------------------------------------------------------------
# Nothing to learn
# --------------------------------------------------------------------------


def test_a_random_walk_teaches_nothing_and_every_trial_is_still_counted(
    cli_env: dict[str, Any],
) -> None:
    vintage_id = sealed(cli_env, pattern=False)
    with stores(cli_env) as s:
        report = s.trainer.run(vintage_id=vintage_id, config=CONFIG)

        assert not report.oos.shows_skill(), report.explain()
        assert not report.null.shows_skill()
        rows = trials_of(s.ledger, report.cycle.search_id)
        assert len(rows) == len(report.thresholds)
        for row in rows:
            assert int(row["trials_in_search_at_time"]) >= len(rows)
        assert s.models.get(report.record.model_id) == report.record


def test_a_dry_run_registers_nothing_but_counts_everything(cli_env: dict[str, Any]) -> None:
    vintage_id = sealed(cli_env, pattern=True)
    with stores(cli_env) as s:
        report = s.trainer.run(vintage_id=vintage_id, config=CONFIG, register=False)
        assert report.cycle.dry_run and not report.cycle.registered
        assert len(trials_of(s.ledger, report.cycle.search_id)) == len(report.thresholds)
        assert s.ledger.conn.execute("SELECT COUNT(*) FROM strategy_specs").fetchone()[0] == 0
        # The model is recorded regardless: the trials' specs pin it, and a
        # trial whose spec could not be rebuilt would be a count of nothing.
        assert s.models.get(report.record.model_id) is not None


def test_retraining_one_idea_accumulates_in_one_lineage(cli_env: dict[str, Any]) -> None:
    """Other parameters, same features and label: the second search's trials
    are stamped with the first's, so "retrain until it passes" is one growing
    search, not a series of small ones."""
    vintage_id = sealed(cli_env, pattern=False)
    with stores(cli_env) as s:
        first = s.trainer.run(vintage_id=vintage_id, config=CONFIG, register=False)
        retuned = TrainingConfig(
            features=CONFIG.features,
            label=CONFIG.label,
            params=ModelParams(n_trees=80, min_data_in_leaf=15),
            n_folds=CONFIG.n_folds,
            min_train=CONFIG.min_train,
        )
        assert retuned.lineage_id == CONFIG.lineage_id
        second = s.trainer.run(vintage_id=vintage_id, config=retuned, register=False)
        assert second.record.model_id != first.record.model_id

        before = len(trials_of(s.ledger, first.cycle.search_id))
        for row in trials_of(s.ledger, second.cycle.search_id):
            assert int(row["trials_in_lineage_at_time"]) > before


# --------------------------------------------------------------------------
# The null, failing
# --------------------------------------------------------------------------


def test_a_null_that_shows_skill_stops_everything(
    cli_env: dict[str, Any], monkeypatch: pytest.MonkeyPatch
) -> None:
    """Simulate an evaluation that leaks: the "shuffled" labels are the real
    ones, so on a learnable pattern the null scores like the model. Nothing
    may be recorded from machinery in that state — no model, no trial."""
    vintage_id = sealed(cli_env, pattern=True)
    monkeypatch.setattr(ml, "shuffled", lambda labels, *, seed: list(labels))
    with stores(cli_env) as s:
        with pytest.raises(TrainingError, match="evaluation is supplying skill"):
            s.trainer.run(vintage_id=vintage_id, config=CONFIG)
        assert s.models.records() == []
        assert s.ledger.conn.execute("SELECT COUNT(*) FROM trials").fetchone()[0] == 0
        events = list(s.ledger.iter_events(event_type=EventType.MODEL_RECORDED))
        assert events == []


def test_history_too_short_for_the_folds_is_refused(cli_env: dict[str, Any]) -> None:
    vintage_id = sealed(cli_env, pattern=False)
    greedy = TrainingConfig(
        features=CONFIG.features, label=CONFIG.label, n_folds=3, min_train=100_000
    )
    with stores(cli_env) as s, pytest.raises(TrainingError, match="cannot be walked forward"):
        s.trainer.run(vintage_id=vintage_id, config=greedy)


def test_a_config_that_could_not_hold_a_position_is_refused() -> None:
    label = LabelDefinition(horizon=2, cost_bps=Decimal("40"))
    with pytest.raises(TrainingError, match="below every entry"):
        TrainingConfig(features=(("sma", 5),), label=label, exit_quantile=0.7)
    with pytest.raises(TrainingError, match="outside the grammar"):
        TrainingConfig(features=(("sma", 5000),), label=label)
    with pytest.raises(TrainingError, match="unknown feature"):
        TrainingConfig(features=(("clairvoyance", 5),), label=label)
    with pytest.raises(TrainingError, match="at least one feature"):
        TrainingConfig(features=(), label=label)
