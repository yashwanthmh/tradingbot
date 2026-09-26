"""`tb replay --fill`: one fill, explained from the ledger alone — and checked.

The README has advertised `tb replay --fill <fill_id>` since M0 and the plan's
success condition ends on it (the ledger explains the whole chain from a fill
back to the spec), but no such command existed. The lineage was all there;
nothing read it back.

The replay here is a registered DSL spec entering on rising bars through the
real loop, settled from order history a cycle later — so every link in the
chain was written by the production path, not by the test.
"""

from __future__ import annotations

import json
from collections.abc import Callable
from decimal import Decimal
from pathlib import Path
from typing import Any

from tb.cli import app
from tb.engine.funding import explicit_book, unfunded_notional
from tb.engine.replay import replay_fill
from tb.ledger.store import Ledger
from tb.registry.lineage import SpecRegistry
from tb.registry.models import AuthorKind
from tb.strategy.dsl.ops import DslStrategy, pipeline_from_spec
from tests.test_cli_registry import _out, runner
from tests.test_funding import a_spec
from tests.test_loop import AS_OF, _broker, _rising_bars, _seed
from tests.test_protection import LATER, _cycle


def _entry_fill(env: dict[str, Any]) -> tuple[str, str]:
    """A registered spec enters; the next cycle settles the entry's fill."""
    _seed(env, _rising_bars(days=140))
    broker = _broker(price=Decimal("119.0"))
    spec = a_spec()
    with Ledger(env["db"], config_hash=env["pinned"].config_hash) as ledger:
        registry = SpecRegistry(
            ledger, per_lineage_budget_ccy=env["pinned"].limits.loss.per_lineage_budget_ccy
        )
        record = registry.register(spec, author_kind=AuthorKind.SEARCH, at=AS_OF)
        book = explicit_book(
            DslStrategy(spec=spec, strategy_id=record.strategy_id, version=record.version),
            pipeline_from_spec(spec),
            notional_ccy=unfunded_notional(env["pinned"].limits, equity_ccy=Decimal("10000.00")),
        )
        entered = _cycle(env, ledger, broker, book, at=AS_OF, run_id="run_a")
        assert entered.submitted, entered.refusals
        settled = _cycle(env, ledger, broker, book, at=LATER, run_id="run_b")
        assert settled.fills_recorded, "the entry's fill is read from history a cycle later"
        (row,) = ledger.conn.execute(
            "SELECT f.fill_id FROM fills f JOIN order_intents i ON f.intent_id = i.intent_id"
            " WHERE i.purpose = 'entry'"
        ).fetchall()
    return str(row["fill_id"]), record.strategy_id


def test_an_entry_fill_is_explained_back_to_its_spec_and_checked(env: dict[str, Any]) -> None:
    fill_id, strategy_id = _entry_fill(env)
    with Ledger(env["db"]) as ledger:
        replayed = replay_fill(ledger, fill_id)

    assert replayed.ok, [check for check in replayed.checks if not check.ok]
    assert [check.name for check in replayed.checks] == [
        "ledger",
        "features",
        "risk",
        "spec",
        "decision",
    ]
    assert replayed.intent is not None and replayed.intent["purpose"] == "entry"
    assert replayed.decision is not None
    assert replayed.decision["action"] == "enter"
    assert replayed.decision["strategy_id"] == strategy_id
    assert set(replayed.features) == {"sma_20", "sma_100"}
    assert replayed.verdicts, "every rule's verdict is part of the explanation"
    assert replayed.spec is not None and replayed.spec["strategy_id"] == strategy_id


def test_features_that_are_not_what_the_strategy_saw_are_caught(
    env: dict[str, Any], tamper: Callable[[Path, str, tuple[Any, ...]], None]
) -> None:
    """Someone with the file rewrites the features a decision records. The
    chain breaks, and the features no longer hash to what the strategy saw."""
    fill_id, _ = _entry_fill(env)
    with Ledger(env["db"]) as ledger:
        (event,) = ledger.conn.execute(
            "SELECT e.seq, e.payload_json FROM event_log e JOIN decisions d"
            " ON d.deciding_event_seq = e.seq WHERE d.action = 'enter'"
        ).fetchall()
    payload = json.loads(str(event["payload_json"]))
    payload["feature_vector"]["sma_20"] = "1.0000000000"
    tamper(
        env["db"],
        "UPDATE event_log SET payload_json = ? WHERE seq = ?",
        (json.dumps(payload), int(event["seq"])),
    )

    with Ledger(env["db"]) as ledger:
        replayed = replay_fill(ledger, fill_id)
    failed = {check.name for check in replayed.checks if not check.ok}
    assert {"ledger", "features"} <= failed


def test_a_recorded_action_its_spec_would_not_take_is_caught(env: dict[str, Any]) -> None:
    """The decision must follow from its inputs, not merely sit beside them:
    recorded as an exit, the same features make this spec hold."""
    fill_id, _ = _entry_fill(env)
    with Ledger(env["db"]) as ledger:
        ledger.conn.execute("UPDATE decisions SET action = 'exit' WHERE action = 'enter'")
        ledger.conn.commit()
        replayed = replay_fill(ledger, fill_id)

    (decision,) = [check for check in replayed.checks if check.name == "decision"]
    assert not decision.ok
    assert "decides hold" in decision.detail


def test_tb_replay_prints_the_chain_and_exits_on_what_it_found(env: dict[str, Any]) -> None:
    fill_id, strategy_id = _entry_fill(env)
    args = ["--limits", str(env["limits"]), "--db", str(env["db"])]

    found = runner.invoke(app, ["replay", "--fill", fill_id, *args])
    missing = runner.invoke(app, ["replay", "--fill", "fil_nothing", *args])

    assert found.exit_code == 0, _out(found)
    output = _out(found)
    for part in (fill_id, strategy_id, "decision", "risk", "spec", "the spec, run on"):
        assert part in output, part
    assert missing.exit_code == 2, _out(missing)
