"""The strategy DSL: schema, interpreter, and the hostile inputs.

Three things these tests exist to establish.

That a generated spec cannot cause code to run — asserted structurally by
walking the AST of the interpretation path for `eval`, `exec`, `compile`,
`__import__` and friends, not by reading the source and believing it.

That the three-valued logic is genuinely three-valued. A comparison against a
missing feature is `UNKNOWN`, not False. Collapsing those two makes a strategy
that never has enough data indistinguishable from one that looked and declined,
and only the second is a real negative result.

That the entry/exit asymmetry holds. An unevaluable entry means no entry; an
unevaluable *exit* means exit anyway, because refusing to close a position
over a missing feature converts a data problem into an unhedged position.
"""

from __future__ import annotations

import ast
import pathlib
from datetime import UTC, datetime, timedelta
from decimal import Decimal

import pytest
from pydantic import ValidationError

from tb.data.asof import UNKNOWN, BarWindow
from tb.data.provider import Bar, Provenance, Resolution, Session
from tb.features.pipeline import FeaturePipeline, make_spec
from tb.strategy.base import Action, Decision, PositionState, StrategyError
from tb.strategy.dsl import (
    MAX_DEPTH,
    MAX_NODES,
    All,
    Any_,
    BudgetExceeded,
    Comparison,
    Constant,
    DslStrategy,
    EvaluationError,
    FeatureRef,
    Not,
    SpecError,
    StrategySpec,
    check_features_available,
    evaluate,
    pipeline_from_spec,
)

UID = "isin:US0378331005"
BASE = datetime(2026, 3, 2, tzinfo=UTC)


def bar(offset: int, close: str) -> Bar:
    opened = BASE + timedelta(days=offset)
    price = Decimal(close)
    return Bar(
        instrument_uid=UID,
        resolution=Resolution.DAILY,
        bar_open_utc=opened,
        available_at_utc=opened + timedelta(days=1),
        ingested_at_utc=opened + timedelta(days=1),
        provider="fixture",
        provenance=Provenance.BACKFILL,
        session=Session.REGULAR,
        open=price,
        high=price,
        low=price,
        close=price,
        volume=1_000_000,
    )


def window(count: int, *, step: str = "1") -> BarWindow:
    bars = tuple(bar(i, str(Decimal("100") + Decimal(step) * i)) for i in range(count))
    return BarWindow(
        as_of=BASE + timedelta(days=count + 1),
        resolution=Resolution.DAILY,
        _by_uid={UID: bars},
    )


def feature(name: str, lookback: int) -> FeatureRef:
    return FeatureRef(name=name, lookback=lookback)


def const(value: str) -> Constant:
    return Constant(value=Decimal(value))


def compare(op: str, left: object, right: object) -> Comparison:
    return Comparison(op=op, left=left, right=right)  # type: ignore[arg-type]


SIMPLE = StrategySpec(
    name="close above its ten-day mean",
    entry=compare("gt", feature("last", 1), feature("sma", 10)),
    exit=compare("lt", feature("last", 1), feature("sma", 10)),
    expected_edge_bps=Decimal("150"),
    min_holding_minutes=0,
)


# --------------------------------------------------------------------------
# No code execution, asserted structurally
# --------------------------------------------------------------------------

FORBIDDEN_CALLS = {"eval", "exec", "compile", "__import__", "globals", "locals", "getattr"}
FORBIDDEN_MODULES = {"subprocess", "importlib", "ctypes", "pickle", "marshal", "os"}


def test_the_interpretation_path_contains_no_way_to_execute_code() -> None:
    """Walked, not read.

    "We do not call eval" as a code-review rule survives until someone needs a
    quick way to support arithmetic in a spec. This test is what makes it a
    property of the package.
    """
    package = pathlib.Path(__file__).resolve().parents[1] / "src" / "tb" / "strategy"
    checked = 0
    for path in sorted(package.rglob("*.py")):
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        checked += 1
        for node in ast.walk(tree):
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
                assert node.func.id not in FORBIDDEN_CALLS, (
                    f"{path.name}:{node.lineno} calls {node.func.id}() — a generated spec "
                    "must never be able to cause code to run"
                )
            if isinstance(node, ast.Import):
                for alias in node.names:
                    assert alias.name.split(".")[0] not in FORBIDDEN_MODULES, (
                        f"{path.name}:{node.lineno} imports {alias.name}"
                    )
            if isinstance(node, ast.ImportFrom) and node.module:
                assert node.module.split(".")[0] not in FORBIDDEN_MODULES, (
                    f"{path.name}:{node.lineno} imports from {node.module}"
                )
    assert checked >= 4, "the AST walk found suspiciously few files"


# --------------------------------------------------------------------------
# Schema validation
# --------------------------------------------------------------------------


def test_a_feature_outside_the_library_is_refused() -> None:
    """The mechanism that makes eval-free interpretation possible."""
    with pytest.raises(ValidationError, match="unknown feature"):
        FeatureRef(name="__import__", lookback=5)
    with pytest.raises(ValidationError, match="unknown feature"):
        FeatureRef(name="os.system", lookback=5)


def test_a_nan_constant_is_refused() -> None:
    """Otherwise every comparison against it is false.

    Which reads as a strategy that legitimately declined to trade, rather than
    as the malformed spec it is.
    """
    with pytest.raises(ValidationError, match="finite"):
        Constant(value=Decimal("NaN"))
    with pytest.raises(ValidationError, match="finite"):
        Constant(value=Decimal("Infinity"))


def test_an_empty_conjunction_is_refused() -> None:
    """An empty `all` is vacuously true, i.e. a spec that always enters."""
    with pytest.raises(ValidationError):
        All(operands=[])
    with pytest.raises(ValidationError):
        Any_(operands=[])


def test_a_lookback_outside_the_bounds_is_refused() -> None:
    """Above, because a 100,000-bar lookback is a denial of service against
    the reader rather than a strategy; below, because zero is meaningless."""
    with pytest.raises(ValidationError):
        FeatureRef(name="sma", lookback=0)
    with pytest.raises(ValidationError):
        FeatureRef(name="sma", lookback=100_000)


def test_a_tree_deeper_than_the_limit_is_refused() -> None:
    """Unbounded nesting is the obvious attack on a recursive interpreter."""
    node: object = compare("gt", feature("last", 1), const("100"))
    for _ in range(MAX_DEPTH + 2):
        node = Not(operand=node)  # type: ignore[arg-type]
    with pytest.raises(SpecError, match="deep"):
        StrategySpec.parse(
            {
                "name": "deep",
                "entry": node,
                "exit": compare("lt", feature("last", 1), const("1")),
                "expected_edge_bps": "100",
            }
        )


def test_a_wide_tree_is_refused_too() -> None:
    """Same denial of service, different shape."""
    operands = [compare("gt", feature("last", 1), const(str(i))) for i in range(MAX_NODES)]
    with pytest.raises(SpecError, match="nodes"):
        StrategySpec.parse(
            {
                "name": "wide",
                "entry": All(operands=operands),  # type: ignore[arg-type]
                "exit": compare("lt", feature("last", 1), const("1")),
                "expected_edge_bps": "100",
            }
        )


def test_an_unrecognised_key_is_refused_rather_than_ignored() -> None:
    """Ignoring it would evaluate something other than what was proposed.

    While the spec hash recorded the proposal — so the ledger would show a
    strategy that is not the one that ran.
    """
    with pytest.raises(SpecError):
        StrategySpec.parse(
            {
                "name": "sneaky",
                "entry": compare("gt", feature("last", 1), const("1")).model_dump(),
                "exit": compare("lt", feature("last", 1), const("1")).model_dump(),
                "expected_edge_bps": "100",
                "on_error": "ignore",
            }
        )


def test_a_spec_must_declare_a_positive_edge() -> None:
    for edge in ("0", "-50"):
        with pytest.raises(SpecError):
            StrategySpec.parse(
                {
                    "name": "no edge",
                    "entry": compare("gt", feature("last", 1), const("1")).model_dump(),
                    "exit": compare("lt", feature("last", 1), const("1")).model_dump(),
                    "expected_edge_bps": edge,
                }
            )


@pytest.mark.parametrize(
    "payload",
    [
        {"kind": "compare", "op": "exec", "left": {"kind": "const", "value": "1"}},
        {"kind": "eval", "operands": []},
        {
            "kind": "compare",
            "op": "gt",
            "left": "__import__('os')",
            "right": {"kind": "const", "value": "1"},
        },
        "entry = __import__('os').system('true')",
        ["all", ["gt", "sma_10", 5]],
        {"kind": "all", "operands": "not a list"},
        None,
        42,
    ],
)
def test_hostile_payloads_are_refused_without_executing_anything(payload: object) -> None:
    """Fuzzing the parser. Nothing here runs, and nothing hangs."""
    with pytest.raises(SpecError):
        StrategySpec.parse(
            {
                "name": "fuzz",
                "entry": payload,
                "exit": compare("lt", feature("last", 1), const("1")).model_dump(),
                "expected_edge_bps": "100",
            }
        )


def test_a_unicode_name_is_accepted_and_hashes_stably() -> None:
    """Not every odd input is an attack; a name is free text."""
    spec = StrategySpec(
        name="ünïcödé 策略 🎯",
        entry=compare("gt", feature("last", 1), const("1")),
        exit=compare("lt", feature("last", 1), const("1")),
        expected_edge_bps=Decimal("100"),
    )
    assert spec.spec_hash == spec.model_copy().spec_hash


# --------------------------------------------------------------------------
# Spec identity
# --------------------------------------------------------------------------


def test_the_same_tree_under_a_different_name_is_a_different_spec() -> None:
    """The name is part of the content, so renaming produces a new hash.

    Deliberate: M5 counts trials by spec hash, and a searcher that renamed a
    tree to get a second evaluation would otherwise succeed.
    """
    other = SIMPLE.model_copy(update={"name": "renamed"})
    assert other.spec_hash != SIMPLE.spec_hash


def test_a_spec_reports_the_features_it_needs() -> None:
    spec = StrategySpec(
        name="two features",
        entry=All(
            operands=[
                compare("gt", feature("last", 1), feature("sma", 20)),
                compare("lt", feature("stdev_pct", 20), const("5")),
            ]
        ),
        exit=compare("lt", feature("last", 1), feature("sma", 20)),
        expected_edge_bps=Decimal("120"),
    )
    assert spec.required_features == ("last_1", "sma_20", "stdev_pct_20")
    assert spec.max_lookback == 20


def test_the_pipeline_is_built_from_the_spec_not_hoped_to_match() -> None:
    """A mismatch would make every decision UNKNOWN — a false negative.

    And it would look exactly like a strategy that legitimately never found an
    opportunity, which is why this is derived rather than configured.
    """
    pipeline = pipeline_from_spec(SIMPLE)
    assert set(pipeline.names) == set(SIMPLE.required_features)


def test_a_spec_reading_an_absent_feature_is_rejected_at_registration() -> None:
    wrong = FeaturePipeline(specs=(make_spec("sma", 99),))
    snapshot = wrong.compute(window(120), UID)
    with pytest.raises(EvaluationError, match="does not compute"):
        check_features_available(SIMPLE, snapshot.values)


# --------------------------------------------------------------------------
# Three-valued logic
# --------------------------------------------------------------------------


def test_a_comparison_against_a_missing_feature_is_unknown_not_false() -> None:
    """The distinction the whole engine depends on.

    "Declined to enter" and "could not tell" are different facts, and only the
    first is a negative result about the strategy.
    """
    pipeline = FeaturePipeline(specs=(make_spec("sma", 50),))
    snapshot = pipeline.compute(window(5), UID)
    outcome = evaluate(compare("gt", feature("sma", 50), const("1")), snapshot)
    assert outcome.result is UNKNOWN
    assert not outcome.fired


def test_not_unknown_is_unknown() -> None:
    """Negating an unevaluable thing does not make it evaluable."""
    pipeline = FeaturePipeline(specs=(make_spec("sma", 50),))
    snapshot = pipeline.compute(window(5), UID)
    inner = compare("gt", feature("sma", 50), const("1"))
    assert evaluate(Not(operand=inner), snapshot).result is UNKNOWN


def test_a_definite_false_makes_a_conjunction_false_despite_unknowns() -> None:
    """No value of the unknown could rescue it, so False is the honest answer."""
    pipeline = FeaturePipeline(specs=(make_spec("last", 1), make_spec("sma", 50)))
    snapshot = pipeline.compute(window(5), UID)
    tree = All(
        operands=[
            compare("lt", feature("last", 1), const("0")),  # definitely false
            compare("gt", feature("sma", 50), const("1")),  # unknown
        ]
    )
    assert evaluate(tree, snapshot).result is False


def test_a_definite_true_makes_a_disjunction_true_despite_unknowns() -> None:
    pipeline = FeaturePipeline(specs=(make_spec("last", 1), make_spec("sma", 50)))
    snapshot = pipeline.compute(window(5), UID)
    tree = Any_(
        operands=[
            compare("gt", feature("last", 1), const("0")),  # definitely true
            compare("gt", feature("sma", 50), const("1")),  # unknown
        ]
    )
    assert evaluate(tree, snapshot).result is True


def test_evaluation_records_which_clause_fired() -> None:
    """Diagnosing a spec that never trades needs the blocking clause named."""
    pipeline = pipeline_from_spec(SIMPLE)
    snapshot = pipeline.compute(window(30), UID)
    outcome = evaluate(SIMPLE.entry, snapshot)
    assert outcome.evidence
    assert any("gt" in label for label in outcome.evidence)


def test_an_unvalidated_object_is_refused_rather_than_walked() -> None:
    pipeline = pipeline_from_spec(SIMPLE)
    snapshot = pipeline.compute(window(30), UID)
    with pytest.raises(EvaluationError, match="not a predicate node"):
        evaluate({"kind": "compare"}, snapshot)


def test_a_tiny_step_budget_refuses_rather_than_answering_partially() -> None:
    """If this fires in production, a spec got past validation."""
    pipeline = pipeline_from_spec(SIMPLE)
    snapshot = pipeline.compute(window(30), UID)
    with pytest.raises(BudgetExceeded, match="step budget"):
        evaluate(SIMPLE.entry, snapshot, step_budget=1)


# --------------------------------------------------------------------------
# The strategy adapter
# --------------------------------------------------------------------------


def flat() -> PositionState:
    return PositionState(instrument_uid=UID)


def held(*, entered_days_ago: int = 5, as_of: datetime | None = None) -> PositionState:
    """An open position entered `entered_days_ago` before the decision time.

    Anchored to the window's own `as_of` rather than to a fixed day: anchoring
    to a constant produced an `entry_at` *after* the decision time for short
    windows, which is how the negative-holding-period bug surfaced.
    """
    reference = as_of or (BASE + timedelta(days=31))
    return PositionState(
        instrument_uid=UID,
        quantity=Decimal("10"),
        entry_price=Decimal("100"),
        entry_at=reference - timedelta(days=entered_days_ago),
    )


def test_a_rising_series_triggers_the_entry() -> None:
    strategy = DslStrategy(spec=SIMPLE, strategy_id="strat_test")
    view = window(30)  # monotonically rising, so close > sma_10
    snapshot = pipeline_from_spec(SIMPLE).compute(view, UID)
    decision = strategy.decide(snapshot=snapshot, window=view, position=flat())
    assert decision.action is Action.ENTER
    assert decision.expected_edge_bps == SIMPLE.expected_edge_bps
    assert decision.feature_snapshot_hash == snapshot.snapshot_hash


def test_a_falling_series_does_not_trigger_the_entry() -> None:
    strategy = DslStrategy(spec=SIMPLE, strategy_id="strat_test")
    view = window(30, step="-1")
    snapshot = pipeline_from_spec(SIMPLE).compute(view, UID)
    decision = strategy.decide(snapshot=snapshot, window=view, position=flat())
    assert decision.action is Action.HOLD
    assert not decision.wants_to_trade


def test_an_unevaluable_entry_means_no_entry() -> None:
    """Acting on conditions you cannot evaluate is guessing."""
    strategy = DslStrategy(spec=SIMPLE, strategy_id="strat_test")
    view = window(3)  # too short for sma_10
    snapshot = pipeline_from_spec(SIMPLE).compute(view, UID)
    decision = strategy.decide(snapshot=snapshot, window=view, position=flat())
    assert decision.action is Action.HOLD
    assert "unevaluable" in decision.rationale


def test_an_unevaluable_exit_means_exit_anyway() -> None:
    """The asymmetry, and the most consequential default in the module.

    With no bracket orders on this venue, an open position whose exit logic
    cannot be evaluated is precisely the state the safety design exists to
    avoid. Every entry rule exists to stop the bot taking on risk; none of
    them is a reason to hold risk it has decided to shed.
    """
    strategy = DslStrategy(spec=SIMPLE, strategy_id="strat_test")
    view = window(3)  # too short for sma_10
    snapshot = pipeline_from_spec(SIMPLE).compute(view, UID)
    position = held(entered_days_ago=2, as_of=view.as_of)
    decision = strategy.decide(snapshot=snapshot, window=view, position=position)
    assert decision.action is Action.EXIT
    assert "unhedged" in decision.rationale


def test_a_position_entered_after_the_decision_time_is_refused() -> None:
    """The bug this file's fixture originally had, now a test.

    Every minimum-hold check is `held < minimum`, and a negative holding
    period is less than every minimum — so a position with a future
    `entry_at` would be refused an exit at every single decision, for as long
    as the bad timestamp stood. That is a permanently unhedged position
    arrived at through the very gate meant to prevent one, so the negative
    interval raises instead of being compared.
    """
    future = PositionState(
        instrument_uid=UID,
        quantity=Decimal("10"),
        entry_price=Decimal("100"),
        entry_at=BASE + timedelta(days=100),
    )
    with pytest.raises(StrategyError, match="after the decision time"):
        future.holding_minutes_at(BASE)


def test_the_specs_own_minimum_hold_delays_an_exit() -> None:
    spec = SIMPLE.model_copy(update={"min_holding_minutes": 10_000})
    strategy = DslStrategy(spec=spec, strategy_id="strat_test")
    view = window(30, step="-1")  # exit conditions met
    snapshot = pipeline_from_spec(spec).compute(view, UID)
    decision = strategy.decide(snapshot=snapshot, window=view, position=held(entered_days_ago=1))
    assert decision.action is Action.HOLD
    assert "minimum" in decision.rationale


def test_a_hold_still_carries_its_snapshot_hash() -> None:
    """ "Looked and declined" is different evidence from "was not consulted".

    Only the first says the loop was alive.
    """
    strategy = DslStrategy(spec=SIMPLE, strategy_id="strat_test")
    view = window(30, step="-1")
    snapshot = pipeline_from_spec(SIMPLE).compute(view, UID)
    decision = strategy.decide(snapshot=snapshot, window=view, position=flat())
    assert decision.action is Action.HOLD
    assert decision.feature_snapshot_hash == snapshot.snapshot_hash


# --------------------------------------------------------------------------
# The Decision invariant
# --------------------------------------------------------------------------


def test_a_risk_increasing_decision_must_declare_an_edge() -> None:
    """A trade that cannot be cost-gated does not happen."""
    with pytest.raises(StrategyError, match="positive expected edge"):
        Decision(
            as_of=BASE,
            instrument_uid=UID,
            action=Action.ENTER,
            expected_edge_bps=Decimal(0),
            feature_snapshot_hash="deadbeef",
            strategy_id="strat_test",
            strategy_version=1,
        )


def test_an_exit_needs_no_declared_edge() -> None:
    """Risk reduction is never gated on an edge claim."""
    decision = Decision(
        as_of=BASE,
        instrument_uid=UID,
        action=Action.EXIT,
        expected_edge_bps=Decimal(0),
        feature_snapshot_hash="deadbeef",
        strategy_id="strat_test",
        strategy_version=1,
    )
    assert decision.wants_to_trade
    assert not decision.action.is_risk_increasing


def test_there_is_no_short_action() -> None:
    """The account type does not permit it, so the type system should not.

    An unreachable enum member invites a searcher to propose specs that can
    never execute, and a reviewer to assume the capability exists.
    """
    assert {a.value for a in Action} == {"enter", "exit", "hold"}
