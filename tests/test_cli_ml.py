"""`tb ml`, and the commands M7 changed: driven as an operator would drive them.

A sealed sawtooth vintage is trained on through the CLI; the store is listed,
shown and verified; an edited artifact is caught; the null's failure exits 1
with nothing recorded; and the holdout evaluates a spec with its model — and
refuses one whose model learned from the very window it would be scored on.
"""

from __future__ import annotations

from typing import Any

import pytest

from tb.config.loader import load_hard_limits
from tb.ledger.store import Ledger
from tb.registry.model_store import ModelStore, default_model_root
from tb.research import ml
from tests.test_cli_registry import _out, _run
from tests.test_ml_training import sealed

FEATURES = ["--feature", "return_pct:2", "--feature", "return_pct:4", "--feature", "zscore:10"]
SMALL = ["--horizon", "2", "--cost-bps", "40", "--folds", "3", "--min-train", "60"]
TREES = ["--trees", "40", "--min-leaf", "10"]


def _train(env: dict[str, Any], vintage_id: str, *extra: str) -> Any:
    return _run(["ml", "train", vintage_id, *FEATURES, *SMALL, *TREES, *extra, *env["bar_args"]])


def _ledger(env: dict[str, Any]) -> Ledger:
    pinned = load_hard_limits(env["limits"])
    return Ledger(env["db"], config_hash=pinned.config_hash).open()


def _count(env: dict[str, Any], table: str) -> int:
    with _ledger(env) as ledger:
        return int(ledger.conn.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0])


def _args(env: dict[str, Any]) -> list[str]:
    return ["--limits", str(env["limits"]), "--db", str(env["db"])]


def test_every_ml_command_is_registered() -> None:
    out = _out(_run(["ml", "--help"]))
    for command in ("train", "models", "show", "verify", "calibrate"):
        assert command in out, f"ml {command} is not registered"


def test_calibrate_is_a_gate_that_passes_on_this_build() -> None:
    result = _run(["ml", "calibrate", "--nulls", "4"])
    assert result.exit_code == 0, _out(result)
    assert "shuffled labels show no skill" in _out(result)


def test_a_dry_run_records_the_model_and_its_trials_and_registers_nothing(
    cli_env: dict[str, Any],
) -> None:
    vintage_id = sealed(cli_env, pattern=True)
    result = _train(cli_env, vintage_id)
    assert result.exit_code == 0, _out(result)
    assert "recorded mdl_" in _out(result)
    assert "shuffled labels, same folds" in _out(result)
    assert _count(cli_env, "ml_models") == 1
    assert _count(cli_env, "trials") >= 1
    assert _count(cli_env, "strategy_specs") == 0


def test_apply_registers_and_the_store_lists_shows_and_verifies_it(
    cli_env: dict[str, Any],
) -> None:
    vintage_id = sealed(cli_env, pattern=True)
    assert _train(cli_env, vintage_id, "--apply").exit_code == 0
    assert _count(cli_env, "strategy_specs") >= 1
    with _ledger(cli_env) as ledger:
        (record,) = ModelStore(ledger, default_model_root(cli_env["db"])).records()

    listed = _run(["ml", "models", *_args(cli_env)])
    assert listed.exit_code == 0 and record.model_id in _out(listed)
    shown = _run(["ml", "show", record.model_id, *_args(cli_env)])
    assert shown.exit_code == 0, _out(shown)
    assert "hashes to its record" in _out(shown)
    verified = _run(["ml", "verify", *_args(cli_env)])
    assert verified.exit_code == 0 and "1 model(s) verified" in _out(verified)
    assert _run(["ml", "show", "mdl_0000000000000000", *_args(cli_env)]).exit_code == 2


def test_an_edited_artifact_fails_verify_and_show(cli_env: dict[str, Any]) -> None:
    vintage_id = sealed(cli_env, pattern=True)
    assert _train(cli_env, vintage_id).exit_code == 0
    with _ledger(cli_env) as ledger:
        (record,) = ModelStore(ledger, default_model_root(cli_env["db"])).records()
    artifact = default_model_root(cli_env["db"]) / record.relative_path
    artifact.write_text(artifact.read_text() + "\n")

    verified = _run(["ml", "verify", *_args(cli_env)])
    assert verified.exit_code == 1
    assert "changed since it was admitted" in _out(verified)
    assert _run(["ml", "show", record.model_id, *_args(cli_env)]).exit_code == 1


def test_a_malformed_feature_is_a_setup_error(cli_env: dict[str, Any]) -> None:
    vintage_id = sealed(cli_env, pattern=True)
    result = _run(["ml", "train", vintage_id, "--feature", "sma", *cli_env["bar_args"]])
    assert result.exit_code == 2
    assert "KIND:LOOKBACK" in _out(result)


def test_a_null_that_shows_skill_exits_1_and_records_nothing(
    cli_env: dict[str, Any], monkeypatch: pytest.MonkeyPatch
) -> None:
    vintage_id = sealed(cli_env, pattern=True)
    monkeypatch.setattr(ml, "shuffled", lambda labels, *, seed: list(labels))
    result = _train(cli_env, vintage_id)
    assert result.exit_code == 1, _out(result)
    assert "supplying skill" in _out(result)
    assert _count(cli_env, "ml_models") == 0
    assert _count(cli_env, "trials") == 0


# --------------------------------------------------------------------------
# The holdout, with a model
# --------------------------------------------------------------------------


def _survivor(env: dict[str, Any], vintage_id: str) -> str:
    assert _train(env, vintage_id, "--apply").exit_code == 0
    with _ledger(env) as ledger:
        row = ledger.conn.execute("SELECT strategy_id FROM strategy_specs").fetchone()
    return str(row["strategy_id"])


def test_the_holdout_evaluates_a_spec_with_its_model(cli_env: dict[str, Any]) -> None:
    vintage_id = sealed(cli_env, pattern=True)
    strategy_id = _survivor(cli_env, vintage_id)
    result = _run(["research", "holdout", strategy_id, vintage_id, *cli_env["bar_args"]])
    # Passed or failed on its merits — 0 or 1 — but evaluated: 2 would mean
    # the spec's model could not be loaded into the holdout's pipeline.
    assert result.exit_code in (0, 1), _out(result)
    assert "dry run" in _out(result)


def test_a_holdout_the_model_learned_from_is_refused(cli_env: dict[str, Any]) -> None:
    """The same vintage split earlier: half the window held back, while the
    model learned from labels three-quarters of the way through it. Scored on
    that "holdout", the model would be graded on prices it was fitted to."""
    vintage_id = sealed(cli_env, pattern=True)
    strategy_id = _survivor(cli_env, vintage_id)
    result = _run(
        [
            "research",
            "holdout",
            strategy_id,
            vintage_id,
            "--fraction",
            "0.5",
            "--apply",
            *cli_env["bar_args"],
        ]
    )
    assert result.exit_code == 2, _out(result)
    assert "inside this holdout" in _out(result)
    assert _count(cli_env, "holdout_evaluations") == 0
