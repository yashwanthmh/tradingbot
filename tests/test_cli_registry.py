"""`tb registry ...`, `tb promote ...`, `tb allocator ...`, `tb research ...`.

Exit codes are the contract, as everywhere else in this CLI: 2 for a setup
problem the operator must fix, 1 for a finding, 0 for clean. A promotion
refusal is a finding, so `tb promote evaluate` exits 1 — which is what lets it
sit in a script without a wrapper interpreting its output.
"""

from __future__ import annotations

import json
import random
from collections.abc import Callable
from datetime import UTC, datetime, timedelta
from decimal import Decimal
from pathlib import Path
from typing import Any

import pytest
from typer.testing import CliRunner

from tb.cli import app
from tb.config.loader import load_hard_limits
from tb.data.barstore import BarStore
from tb.data.provider import Bar, BarBatch, Provenance, Resolution, Session
from tb.ledger.store import Ledger
from tb.portfolio.allocator import Allocator, StrategyInput
from tb.registry.ladder import BreachReason, RungEvidence, SizeLadder
from tb.registry.lineage import SpecRegistry
from tb.registry.models import AuthorKind, StrategyStatus, TrialOutcome
from tb.research.holdout import HoldoutRegistry
from tb.research.trials import TrialLog

runner = CliRunner()
BASE = datetime(2024, 1, 2, tzinfo=UTC)
AS_OF = datetime(2026, 6, 1, 14, 0, tzinfo=UTC)
UID = "isin:US0378331005"

SPEC = {
    "name": "sma-cross",
    "entry": {
        "kind": "compare",
        "op": "gt",
        "left": {"kind": "feature", "name": "sma", "lookback": 20},
        "right": {"kind": "feature", "name": "sma", "lookback": 100},
    },
    "exit": {
        "kind": "compare",
        "op": "lt",
        "left": {"kind": "feature", "name": "sma", "lookback": 20},
        "right": {"kind": "feature", "name": "sma", "lookback": 100},
    },
    "expected_edge_bps": "300",
    "min_holding_minutes": 1440,
}


@pytest.fixture
def env(tmp_path: Path, write_limits: Callable[[dict[str, Any]], Path]) -> dict[str, Any]:
    run_dir = tmp_path / "run"
    run_dir.mkdir(exist_ok=True)
    limits = write_limits(
        {
            "safety": {
                "kill_switch_path": str(run_dir / "KILL"),
                "heartbeat_path": str(run_dir / "heartbeat"),
            }
        }
    )
    spec_file = tmp_path / "spec.json"
    spec_file.write_text(json.dumps(SPEC), encoding="utf-8")
    return {
        "limits": limits,
        "db": tmp_path / "ledger.db",
        "bars": tmp_path / "bars",
        "spec": spec_file,
        "args": ["--limits", str(limits), "--db", str(tmp_path / "ledger.db")],
        "bar_args": [
            "--limits",
            str(limits),
            "--db",
            str(tmp_path / "ledger.db"),
            "--bars",
            str(tmp_path / "bars"),
        ],
    }


def _run(args: list[str]) -> Any:
    result = runner.invoke(app, args)
    if result.exception and not isinstance(result.exception, SystemExit):
        raise result.exception
    return result


def _out(result: Any) -> str:
    stderr = ""
    try:
        stderr = result.stderr or ""
    except ValueError:  # pragma: no cover - depends on the click version
        stderr = ""
    return (result.stdout or "") + stderr


def _init(env: dict[str, Any]) -> None:
    _run(["init", "--limits", str(env["limits"]), "--db", str(env["db"])])


def _ledger(env: dict[str, Any]) -> Ledger:
    pinned = load_hard_limits(env["limits"])
    return Ledger(env["db"], config_hash=pinned.config_hash).open()


def _registry(env: dict[str, Any], ledger: Ledger) -> SpecRegistry:
    pinned = load_hard_limits(env["limits"])
    return SpecRegistry(ledger, per_lineage_budget_ccy=pinned.limits.loss.per_lineage_budget_ccy)


def seed_bars(env: dict[str, Any], *, days: int = 500) -> None:
    """A random walk in the store, then sealed. No drift, so no edge."""
    pinned = load_hard_limits(env["limits"])
    with _ledger(env) as ledger:
        store = BarStore(ledger, root=env["bars"], scale=pinned.limits.data.price_scale)
        walk = random.Random(77)
        price = Decimal("100.00")
        bars = []
        for day in range(days):
            opened = BASE + timedelta(days=day)
            price = max(price + Decimal(str(round(walk.gauss(0, 1.1), 4))), Decimal("1.00"))
            close = max(price + Decimal(str(round(walk.gauss(0, 0.7), 4))), Decimal("1.00"))
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
                    high=max(price, close) + Decimal("0.5"),
                    low=min(price, close) - Decimal("0.5"),
                    close=close,
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


def registered_id(env: dict[str, Any]) -> str:
    result = _run(["registry", "register", str(env["spec"]), *env["args"]])
    assert result.exit_code == 0, _out(result)
    with _ledger(env) as ledger:
        row = ledger.conn.execute("SELECT strategy_id FROM strategy_specs").fetchone()
    return str(row["strategy_id"])


# --------------------------------------------------------------------------
# Wiring
# --------------------------------------------------------------------------


def test_every_m5_command_is_registered() -> None:
    """The reachability check. A module with no command is a module nothing
    runs, and this repo has found orphans that way before."""
    groups = {
        "registry": ("register", "list", "lineage", "review"),
        "promote": ("evaluate", "history"),
        "allocator": ("explain", "ladder"),
        "research": ("trials", "holdout", "null-gate"),
    }
    for group, commands in groups.items():
        out = _out(_run([group, "--help"]))
        for command in commands:
            assert command in out, f"{group} {command} is not registered"


# --------------------------------------------------------------------------
# tb registry
# --------------------------------------------------------------------------


def test_register_writes_a_candidate(env: dict[str, Any]) -> None:
    _init(env)
    result = _run(["registry", "register", str(env["spec"]), *env["args"]])
    assert result.exit_code == 0
    out = _out(result)
    assert "registered in lineage" in out
    assert "status candidate" in out


def test_register_refuses_a_malformed_spec(env: dict[str, Any], tmp_path: Path) -> None:
    _init(env)
    bad = tmp_path / "bad.json"
    bad.write_text(json.dumps({"name": "x"}), encoding="utf-8")
    result = _run(["registry", "register", str(bad), *env["args"]])
    assert result.exit_code == 2
    assert "invalid strategy spec" in _out(result)


def test_register_refuses_an_unreadable_file(env: dict[str, Any], tmp_path: Path) -> None:
    _init(env)
    result = _run(["registry", "register", str(tmp_path / "nope.json"), *env["args"]])
    assert result.exit_code == 2
    assert "cannot read" in _out(result)


def test_register_refuses_an_unknown_author_kind(env: dict[str, Any]) -> None:
    _init(env)
    result = _run(["registry", "register", str(env["spec"]), "--author", "wizard", *env["args"]])
    assert result.exit_code == 2


def test_list_shows_status_rung_and_notional(env: dict[str, Any]) -> None:
    _init(env)
    strategy_id = registered_id(env)
    result = _run(["registry", "list", *env["args"]])
    assert result.exit_code == 0
    out = _out(result)
    assert strategy_id[:12] in out
    assert "candidate" in out


def test_list_says_so_when_nothing_is_registered(env: dict[str, Any]) -> None:
    _init(env)
    result = _run(["registry", "list", *env["args"]])
    assert result.exit_code == 0
    assert "nothing registered" in _out(result)


def test_lineage_reports_the_budget_and_exits_one_when_spent(env: dict[str, Any]) -> None:
    _init(env)
    strategy_id = registered_id(env)
    with _ledger(env) as ledger:
        registry = _registry(env, ledger)
        spec = registry.get(strategy_id)
        assert spec is not None
        lineage_id = spec.lineage_id

    ok = _run(["registry", "lineage", lineage_id, *env["args"]])
    assert ok.exit_code == 0
    assert "budget open" in _out(ok)

    with _ledger(env) as ledger:
        _registry(env, ledger).charge(lineage_id, loss_ccy=Decimal("500"), at=AS_OF)

    spent = _run(["registry", "lineage", lineage_id, *env["args"]])
    assert spent.exit_code == 1
    assert "spent its loss budget" in _out(spent)


def test_lineage_refuses_an_unknown_lineage(env: dict[str, Any]) -> None:
    _init(env)
    result = _run(["registry", "lineage", "lin_ghost", *env["args"]])
    assert result.exit_code == 2


# --------------------------------------------------------------------------
# tb promote
# --------------------------------------------------------------------------


def test_promote_refuses_an_unregistered_strategy(env: dict[str, Any]) -> None:
    _init(env)
    result = _run(["promote", "evaluate", "stg_ghost", *env["bar_args"]])
    assert result.exit_code == 2
    assert "is not registered" in _out(result)


def test_promote_prints_every_gate_and_exits_one_on_a_refusal(
    env: dict[str, Any],
) -> None:
    """A fresh candidate has no holdout, no calibration and no vintage, so it is
    refused — and the report must still name every check with its observed
    value, because that is what tells a searcher whether a lineage is worth
    continuing."""
    _init(env)
    strategy_id = registered_id(env)
    result = _run(["promote", "evaluate", strategy_id, *env["bar_args"]])
    assert result.exit_code == 1
    out = _out(result)
    for gate in (
        "sealed_vintage",
        "engine_calibrated",
        "sealed_holdout",
        "oos_trades",
        "deflated_sharpe",
        "deflated_probability",
        "pbo",
        "cost_to_edge",
        "edge_to_feed_noise",
        "holding_period",
        "declared_edge_bounds",
        "lineage_budget",
        "paper_shadow",
    ):
        assert gate in out, f"{gate} was not reported"
    assert "refuse" in out


def test_promote_is_a_dry_run_by_default(env: dict[str, Any]) -> None:
    """Promotion is the moment a generated strategy becomes eligible for real
    money, so the command that does it has to be asked explicitly."""
    _init(env)
    strategy_id = registered_id(env)
    _run(["promote", "evaluate", strategy_id, *env["bar_args"]])
    with _ledger(env) as ledger:
        rows = ledger.conn.execute("SELECT COUNT(*) FROM promotions").fetchone()
        assert rows[0] == 0


def test_promote_with_apply_records_the_decision(env: dict[str, Any]) -> None:
    _init(env)
    strategy_id = registered_id(env)
    result = _run(["promote", "evaluate", strategy_id, "--apply", *env["bar_args"]])
    assert result.exit_code == 1
    with _ledger(env) as ledger:
        row = ledger.conn.execute("SELECT * FROM promotions").fetchone()
        assert row is not None
        assert row["decision"] == "refuse"
        # Every gate's verdict, not only the failures.
        assert len(json.loads(str(row["gate_results_json"]))) >= 13


def test_promote_refuses_an_unknown_jurisdiction(env: dict[str, Any]) -> None:
    _init(env)
    strategy_id = registered_id(env)
    result = _run(
        [
            "promote",
            "evaluate",
            strategy_id,
            "--jurisdiction",
            "atlantis",
            *env["bar_args"],
        ]
    )
    assert result.exit_code == 2


def test_promote_history_is_empty_before_any_decision(env: dict[str, Any]) -> None:
    _init(env)
    strategy_id = registered_id(env)
    result = _run(["promote", "history", strategy_id, *env["args"]])
    assert result.exit_code == 0
    assert "no promotion decisions" in _out(result)


def test_promote_history_reads_back_the_recorded_report(env: dict[str, Any]) -> None:
    _init(env)
    strategy_id = registered_id(env)
    _run(["promote", "evaluate", strategy_id, "--apply", *env["bar_args"]])
    result = _run(["promote", "history", strategy_id, *env["args"]])
    assert result.exit_code == 0
    out = _out(result)
    assert "deflated_sharpe" in out
    assert "refuse" in out


# --------------------------------------------------------------------------
# tb allocator
# --------------------------------------------------------------------------


def test_allocator_explain_says_so_when_nothing_is_allocated(
    env: dict[str, Any],
) -> None:
    _init(env)
    result = _run(["allocator", "explain", *env["args"]])
    assert result.exit_code == 0
    assert "no allocations recorded" in _out(result)


def test_allocator_explain_shows_the_shrinkage(env: dict[str, Any]) -> None:
    """The point of the command: at ten trades the realised edge carries a
    quarter of the weight, so a number that looked like a measurement is mostly
    the backtest's opinion."""
    _init(env)
    strategy_id = registered_id(env)
    pinned = load_hard_limits(env["limits"])
    with _ledger(env) as ledger:
        Allocator(ledger, limits=pinned.limits).allocate(
            strategies=[
                StrategyInput(
                    strategy_id=strategy_id,
                    version=1,
                    lineage_id="lin_1",
                    rung=0,
                    prior_edge_bps=Decimal("300"),
                    realised_edge_bps=Decimal("100"),
                    n_realised_trades=10,
                )
            ],
            equity_ccy=Decimal("20000"),
            at=AS_OF,
        )
    result = _run(["allocator", "explain", *env["args"]])
    assert result.exit_code == 0
    out = _out(result)
    assert "0.250" in out
    assert "mainly on the backtest prior" in out


def test_allocator_ladder_shows_every_rung_change(env: dict[str, Any]) -> None:
    _init(env)
    strategy_id = registered_id(env)
    pinned = load_hard_limits(env["limits"])
    with _ledger(env) as ledger:
        ladder = SizeLadder(ledger, limits=pinned.limits)
        earned = RungEvidence(
            days_at_rung=10, n_trades_at_rung=8, realised_pnl_at_rung=Decimal("12")
        )
        ladder.review(strategy_id, evidence=earned, at=AS_OF)
        ladder.review(strategy_id, evidence=earned, at=AS_OF)
        ladder.breach(strategy_id, reason=BreachReason.DRAWDOWN, at=AS_OF)

    result = _run(["allocator", "ladder", strategy_id, *env["args"]])
    assert result.exit_code == 0
    out = _out(result)
    assert "0→1" in out
    assert "2→0" in out


def test_allocator_ladder_refuses_an_unregistered_strategy(env: dict[str, Any]) -> None:
    _init(env)
    result = _run(["allocator", "ladder", "stg_ghost", *env["args"]])
    assert result.exit_code == 2


# --------------------------------------------------------------------------
# tb research
# --------------------------------------------------------------------------


def test_research_trials_says_so_when_empty(env: dict[str, Any]) -> None:
    _init(env)
    result = _run(["research", "trials", *env["args"]])
    assert result.exit_code == 0
    assert "no trials recorded" in _out(result)


def test_research_trials_shows_both_multiplicity_counts(env: dict[str, Any]) -> None:
    _init(env)
    with _ledger(env) as ledger:
        log = TrialLog(ledger)
        for index in range(4):
            log.record(
                search_id="srch_1",
                lineage_id=f"lin_{index}",
                spec_hash=f"h{index}",
                author_kind=AuthorKind.SEARCH,
                outcome=TrialOutcome.EVALUATED,
                net_sharpe=0.3 * index,
                n_trades=12,
                at=AS_OF,
            )
        log.record(
            search_id="srch_1",
            lineage_id="lin_0",
            spec_hash="h_reject",
            author_kind=AuthorKind.SEARCH,
            outcome=TrialOutcome.REJECTED,
            rejection_reason="cost gate refused the declared edge",
            at=AS_OF,
        )
    result = _run(["research", "trials", *env["args"]])
    assert result.exit_code == 0
    out = _out(result)
    assert "rejected" in out
    assert "cost gate refused" in out
    assert "deflated against" in out


def test_research_holdout_refuses_an_unregistered_strategy(env: dict[str, Any]) -> None:
    _init(env)
    result = _run(["research", "holdout", "stg_ghost", "vint_x", *env["bar_args"]])
    assert result.exit_code == 2
    assert "is not registered" in _out(result)


def test_research_holdout_refuses_an_unsealed_vintage(env: dict[str, Any]) -> None:
    _init(env)
    strategy_id = registered_id(env)
    result = _run(["research", "holdout", strategy_id, "vint_ghost", *env["bar_args"]])
    assert result.exit_code == 2
    assert "not in the ledger" in _out(result)


def test_research_holdout_runs_the_evaluation_and_records_it_once(
    env: dict[str, Any],
) -> None:
    """The whole point of the command, and the refusal that follows it."""
    _init(env)
    seed_bars(env)
    strategy_id = registered_id(env)
    sealed = _run(["data", "seal", "--resolution", "daily", *env["bar_args"]])
    assert sealed.exit_code == 0, _out(sealed)
    with _ledger(env) as ledger:
        row = ledger.conn.execute("SELECT vintage_id FROM data_snapshots").fetchone()
    vintage_id = str(row["vintage_id"])

    # Exit 1: a 20/100 SMA cross over a 124-day holdout produces one trade,
    # well short of `min_oos_trades`, so it fails. That is the right answer on a
    # random walk and the assertion is specific rather than "0 or 1" — a loose
    # code here would keep passing if the command stopped evaluating at all.
    dry = _run(["research", "holdout", strategy_id, vintage_id, *env["bar_args"]])
    assert dry.exit_code == 1
    out = _out(dry)
    assert "dry run: nothing recorded" in out
    assert "trades over" in out, "the backtest summary should be printed"
    with _ledger(env) as ledger:
        assert ledger.conn.execute("SELECT COUNT(*) FROM holdout_evaluations").fetchone()[0] == 0

    applied = _run(["research", "holdout", strategy_id, vintage_id, "--apply", *env["bar_args"]])
    assert applied.exit_code == 1
    assert "failed the holdout" in _out(applied)
    assert "terminal for this version" in _out(applied)
    with _ledger(env) as ledger:
        row = ledger.conn.execute("SELECT * FROM holdout_evaluations").fetchone()
        assert row["passed"] == 0
        assert row["vintage_id"] == vintage_id

    again = _run(["research", "holdout", strategy_id, vintage_id, "--apply", *env["bar_args"]])
    assert again.exit_code == 2
    out = _out(again)
    assert "already spent its holdout evaluation" in out
    assert "multiplicity haircut" in out


def test_research_holdout_refuses_a_window_too_short_to_mean_anything(
    env: dict[str, Any],
) -> None:
    """A ten-day holdout produces out-of-sample statistics that are noise, and
    promoting on them while believing there was an independent check is the
    failure the whole mechanism exists to prevent."""
    _init(env)
    seed_bars(env, days=120)
    strategy_id = registered_id(env)
    _run(["data", "seal", "--resolution", "daily", *env["bar_args"]])
    with _ledger(env) as ledger:
        row = ledger.conn.execute("SELECT vintage_id FROM data_snapshots").fetchone()
    result = _run(
        [
            "research",
            "holdout",
            strategy_id,
            str(row["vintage_id"]),
            "--fraction",
            "0.1",
            *env["bar_args"],
        ]
    )
    assert result.exit_code == 2
    assert "Out-of-sample statistics over that are noise" in _out(result)


def test_a_promoted_strategy_reads_as_promoted_everywhere(env: dict[str, Any]) -> None:
    """One check that the registry, the ladder and the CLI agree.

    Promotion is written by the gate alone, so this drives it through the
    registry's own status row rather than through a second code path.
    """
    _init(env)
    strategy_id = registered_id(env)
    with _ledger(env) as ledger:
        ledger.conn.execute(
            "UPDATE strategy_status SET status = ?, promoted_at = ? WHERE strategy_id = ?",
            (StrategyStatus.PROMOTED.value, AS_OF.isoformat(), strategy_id),
        )
        ledger.conn.commit()
        registry = _registry(env, ledger)
        allowed, why = registry.may_trade(strategy_id)
    assert allowed, why

    listed = _run(["registry", "list", "--status", "promoted", *env["args"]])
    assert listed.exit_code == 0
    assert strategy_id[:12] in _out(listed)


def test_the_holdout_registry_and_the_cli_agree_on_what_was_spent(
    env: dict[str, Any],
) -> None:
    _init(env)
    strategy_id = registered_id(env)
    with _ledger(env) as ledger:
        registry = _registry(env, ledger)
        spec = registry.get(strategy_id)
        assert spec is not None
        HoldoutRegistry(ledger).record(
            strategy_id=strategy_id,
            version=1,
            lineage_id=spec.lineage_id,
            spec_hash=spec.spec_hash,
            vintage_id="vint_manual",
            sealed_from=AS_OF,
            passed=False,
            detail="failed by hand for the test",
        )
    result = _run(["research", "holdout", strategy_id, "vint_manual", "--apply", *env["bar_args"]])
    assert result.exit_code == 2
    assert "already spent" in _out(result)


# --------------------------------------------------------------------------
# tb registry review
# --------------------------------------------------------------------------


def test_review_says_so_when_nothing_is_promoted(env: dict[str, Any]) -> None:
    _init(env)
    registered_id(env)
    result = _run(["registry", "review", *env["args"]])
    assert result.exit_code == 0
    assert "nothing is promoted" in _out(result)


def promote_by_hand(env: dict[str, Any], strategy_id: str) -> None:
    """Set the status row directly.

    Promotion is written by the gate alone, and the gate would refuse a
    candidate with no evidence — so a review test that went through it could
    never reach a promoted strategy. Writing the row is honest about what is
    being set up.
    """
    with _ledger(env) as ledger:
        ledger.conn.execute(
            "UPDATE strategy_status SET status = ?, promoted_at = ? WHERE strategy_id = ?",
            (StrategyStatus.PROMOTED.value, AS_OF.isoformat(), strategy_id),
        )
        ledger.conn.commit()


def test_review_keeps_a_strategy_with_too_little_evidence(env: dict[str, Any]) -> None:
    """KEEP means nothing has been shown either way, not that anything works.
    Most reviews of a young strategy should return it."""
    _init(env)
    strategy_id = registered_id(env)
    promote_by_hand(env, strategy_id)
    result = _run(["registry", "review", *env["args"]])
    assert result.exit_code == 0
    out = _out(result)
    assert "KEEP" in out
    # A short fragment: rich wraps the reason at the terminal width, so a longer
    # phrase would straddle a newline and the assertion would be about the
    # console width rather than about the verdict.
    assert "nothing has been shown" in out


def test_review_kills_a_strategy_that_has_eaten_its_lineages_budget(
    env: dict[str, Any],
) -> None:
    """KILL is cheap on purpose: the cost of retiring a good strategy is a
    missed opportunity, and the cost of keeping a bad one is money."""
    _init(env)
    strategy_id = registered_id(env)
    promote_by_hand(env, strategy_id)
    with _ledger(env) as ledger:
        _registry(env, ledger).record_realised(
            strategy_id, pnl_ccy=Decimal("-60"), n_trades=12, at=AS_OF
        )
    result = _run(["registry", "review", *env["args"]])
    assert result.exit_code == 1
    out = _out(result)
    assert "KILL" in out
    assert "lineage's loss budget" in out
    assert "would be retired" in out


def test_review_is_a_dry_run_by_default(env: dict[str, Any]) -> None:
    _init(env)
    strategy_id = registered_id(env)
    promote_by_hand(env, strategy_id)
    with _ledger(env) as ledger:
        _registry(env, ledger).record_realised(
            strategy_id, pnl_ccy=Decimal("-60"), n_trades=12, at=AS_OF
        )
    _run(["registry", "review", *env["args"]])
    with _ledger(env) as ledger:
        record = _registry(env, ledger).status_of(strategy_id)
        assert record is not None
        assert record.status is StrategyStatus.PROMOTED


def test_review_with_apply_retires_and_records(env: dict[str, Any]) -> None:
    _init(env)
    strategy_id = registered_id(env)
    promote_by_hand(env, strategy_id)
    with _ledger(env) as ledger:
        _registry(env, ledger).record_realised(
            strategy_id, pnl_ccy=Decimal("-60"), n_trades=12, at=AS_OF
        )
    result = _run(["registry", "review", "--apply", *env["args"]])
    assert result.exit_code == 1
    with _ledger(env) as ledger:
        record = _registry(env, ledger).status_of(strategy_id)
        assert record is not None
        # Blocked, not retired: the lineage budget went first, and that is a
        # statement about the lineage rather than about this strategy.
        assert record.status in (StrategyStatus.RETIRED, StrategyStatus.BLOCKED)
        events = ledger.conn.execute(
            "SELECT COUNT(*) FROM event_log WHERE event_type = 'strategy.reviewed'"
        ).fetchone()
        assert events[0] == 1


def test_review_cannot_enlarge_a_position(env: dict[str, Any]) -> None:
    """SCALE is a verdict, not a size change. The rung is moved by the ladder on
    its own schedule — a review must not be a second way to grow a position."""
    _init(env)
    strategy_id = registered_id(env)
    promote_by_hand(env, strategy_id)
    pinned = load_hard_limits(env["limits"])
    with _ledger(env) as ledger:
        registry = _registry(env, ledger)
        registry.record_realised(strategy_id, pnl_ccy=Decimal("40"), n_trades=50, at=AS_OF)
        Allocator(ledger, limits=pinned.limits).allocate(
            strategies=[
                StrategyInput(
                    strategy_id=strategy_id,
                    version=1,
                    lineage_id="lin_1",
                    rung=0,
                    prior_edge_bps=Decimal("300"),
                    realised_edge_bps=Decimal("250"),
                    n_realised_trades=50,
                )
            ],
            equity_ccy=Decimal("20000"),
            at=AS_OF,
        )
        rung_before = SizeLadder(ledger, limits=pinned.limits).rung_of(strategy_id)

    result = _run(["registry", "review", "--apply", *env["args"]])
    assert result.exit_code == 0
    with _ledger(env) as ledger:
        rung_after = SizeLadder(ledger, limits=pinned.limits).rung_of(strategy_id)
    assert rung_after == rung_before


# --------------------------------------------------------------------------
# tb research null-gate
# --------------------------------------------------------------------------


@pytest.mark.slow
def test_null_gate_reports_a_rate_against_the_ceiling(env: dict[str, Any]) -> None:
    """The release gate as a command, because the thresholds it depends on live
    in a file a human edits."""
    _init(env)
    result = _run(["research", "null-gate", "--specs", "30", "--days", "300", *env["args"]])
    assert result.exit_code == 0
    out = _out(result)
    assert "within the" in out
    assert "reached the statistical checks" in out
    assert "which gates refused" in out


def test_null_gate_refuses_a_population_of_one(env: dict[str, Any]) -> None:
    """A rate over a single spec is not a rate, and reporting 0% from one would
    look like evidence."""
    _init(env)
    result = _run(["research", "null-gate", "--specs", "1", *env["args"]])
    assert result.exit_code == 2
    assert "at least 2 specs" in _out(result)


def test_null_gate_refuses_a_fixture_too_short_to_split(env: dict[str, Any]) -> None:
    _init(env)
    result = _run(["research", "null-gate", "--specs", "4", "--days", "3", *env["args"]])
    assert result.exit_code == 2
    assert "cannot be split" in _out(result)
