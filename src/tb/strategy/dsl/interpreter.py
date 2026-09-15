"""Evaluating a spec. No `eval`, no `exec`, no `compile`, no import.

The interpreter walks the validated tree and dispatches on a `Literal` type
tag through a fixed operator table. There is no path from spec content to
executed code — a spec is data all the way down, and the worst a malicious or
malformed one can do is be rejected.

Three budgets, because validation bounds the *shape* of a tree and evaluation
also has to bound its *cost*:

**Steps.** Every node visit decrements a counter. Exceeding it raises. The
schema's node cap makes this unreachable for a valid spec, which is the point:
if the step budget ever fires, something got past validation and the right
response is to stop rather than to continue with a partial evaluation.

**Wall clock.** Belt to the step budget's brace, because a step is not a
constant amount of work.

**Unknown propagation, not coercion.** A comparison against `UNKNOWN` is
neither true nor false — it is unevaluable, and the whole predicate becomes
`UNKNOWN`. It does *not* become False. That distinction is the difference
between "the strategy declined to enter" and "the strategy could not tell",
and collapsing them would make a strategy that never has enough data look
identical to one that looked and said no. The engine treats an `UNKNOWN` entry
predicate as no-entry, but records it separately.

An `UNKNOWN` *exit* predicate is different and is handled in the engine, not
here: refusing to exit because a feature is missing turns a data problem into
an unhedged position, which is the asymmetry the whole system is built around.
"""

from __future__ import annotations

import time
from collections.abc import Mapping
from dataclasses import dataclass, field
from decimal import Decimal
from typing import TypeAlias

from tb.core.errors import TbError
from tb.data.asof import UNKNOWN, Unknown
from tb.features.pipeline import FeatureSnapshot
from tb.strategy.dsl.schema import (
    All,
    Any_,
    Comparison,
    Constant,
    FeatureRef,
    Not,
    StrategySpec,
)

# Evaluation is three-valued. `Unknown` is not an error — it is the honest
# answer when a feature has no value yet, and it must survive to the caller.
Tri: TypeAlias = "bool | Unknown"

# Generous against the schema's 64-node cap. If this ever fires, a spec got
# past validation and the correct response is to refuse rather than to return
# a partial answer.
DEFAULT_STEP_BUDGET = 1_000
DEFAULT_WALL_CLOCK_SECONDS = 1.0


class EvaluationError(TbError):
    """A spec could not be evaluated. Never a signal — always a refusal."""


class BudgetExceeded(EvaluationError):
    """A spec exhausted its step or wall-clock budget."""


# The operator table. Comparisons only, on Decimals, total and side-effect
# free. This table is the entire set of things a generated spec can cause to
# happen.
_COMPARISONS: dict[str, object] = {
    "lt": lambda a, b: a < b,
    "lte": lambda a, b: a <= b,
    "gt": lambda a, b: a > b,
    "gte": lambda a, b: a >= b,
}


@dataclass(slots=True)
class _Budget:
    steps_remaining: int
    deadline: float
    steps_used: int = 0

    def spend(self) -> None:
        self.steps_used += 1
        self.steps_remaining -= 1
        if self.steps_remaining < 0:
            raise BudgetExceeded(
                f"spec exceeded its step budget after {self.steps_used} nodes. The schema "
                "caps a valid spec well below this, so reaching it means a spec got past "
                "validation — refusing rather than returning a partial evaluation."
            )
        # Checked every 32 steps: `time.monotonic()` per node would dominate
        # the cost of evaluating a tree this small.
        if self.steps_used % 32 == 0 and time.monotonic() > self.deadline:
            raise BudgetExceeded(
                f"spec exceeded its wall-clock budget after {self.steps_used} nodes"
            )


@dataclass(slots=True)
class Evaluation:
    """The outcome of evaluating one predicate tree, with its working shown."""

    result: Tri
    steps_used: int
    evidence: dict[str, str] = field(default_factory=dict)

    @property
    def fired(self) -> bool:
        """True only on a definite True.

        `UNKNOWN` is not firing. Accessing `.result` directly and truth-testing
        it would raise, which is deliberate — this property is the explicit
        way to ask.
        """
        return self.result is True


def evaluate(
    predicate: object,
    snapshot: FeatureSnapshot,
    *,
    step_budget: int = DEFAULT_STEP_BUDGET,
    wall_clock_seconds: float = DEFAULT_WALL_CLOCK_SECONDS,
) -> Evaluation:
    """Evaluate a predicate tree against one feature snapshot."""
    budget = _Budget(
        steps_remaining=step_budget,
        deadline=time.monotonic() + wall_clock_seconds,
    )
    evidence: dict[str, str] = {}
    result = _eval_predicate(predicate, snapshot, budget, evidence)
    return Evaluation(result=result, steps_used=budget.steps_used, evidence=evidence)


def _eval_predicate(
    node: object,
    snapshot: FeatureSnapshot,
    budget: _Budget,
    evidence: dict[str, str],
) -> Tri:
    budget.spend()

    if isinstance(node, Comparison):
        return _eval_comparison(node, snapshot, budget, evidence)

    if isinstance(node, Not):
        inner = _eval_predicate(node.operand, snapshot, budget, evidence)
        # `not UNKNOWN` is UNKNOWN, not True. Negating an unevaluable thing
        # does not make it evaluable.
        return UNKNOWN if inner is UNKNOWN else (not inner)

    if isinstance(node, All):
        # Three-valued conjunction: one definite False makes the whole thing
        # False even with unknowns present, because no value of the unknown
        # could rescue it. Otherwise any unknown makes the result unknown.
        # Evaluated eagerly rather than short-circuiting so the evidence map
        # records every operand — diagnosing a spec that never fires needs to
        # know which clause is the blocker, not just that one is.
        results = [
            _eval_predicate(operand, snapshot, budget, evidence) for operand in node.operands
        ]
        if any(r is False for r in results):
            return False
        return UNKNOWN if any(r is UNKNOWN for r in results) else True

    if isinstance(node, Any_):
        results = [
            _eval_predicate(operand, snapshot, budget, evidence) for operand in node.operands
        ]
        if any(r is True for r in results):
            return True
        return UNKNOWN if any(r is UNKNOWN for r in results) else False

    raise EvaluationError(
        f"not a predicate node: {type(node).__name__}. The schema discriminates on a "
        "Literal tag, so reaching this means an unvalidated object was passed in."
    )


def _eval_comparison(
    node: Comparison,
    snapshot: FeatureSnapshot,
    budget: _Budget,
    evidence: dict[str, str],
) -> Tri:
    left = _eval_term(node.left, snapshot, budget)
    right = _eval_term(node.right, snapshot, budget)

    label = f"{_describe(node.left)} {node.op} {_describe(node.right)}"
    if left is UNKNOWN or right is UNKNOWN:
        evidence[label] = "UNKNOWN"
        return UNKNOWN

    operator = _COMPARISONS[node.op]
    outcome = bool(operator(left, right))  # type: ignore[operator]
    evidence[label] = f"{left} {node.op} {right} -> {outcome}"
    return outcome


def _eval_term(node: object, snapshot: FeatureSnapshot, budget: _Budget) -> Decimal | Unknown:
    budget.spend()
    if isinstance(node, Constant):
        return node.value
    if isinstance(node, FeatureRef):
        try:
            return snapshot.values[node.feature_key]
        except KeyError:
            # A spec referring to a feature the pipeline was not built for.
            # Refused rather than treated as absent: absence means "no data
            # yet" and would make the strategy look like it was working.
            raise EvaluationError(
                f"spec reads feature {node.feature_key!r}, which this pipeline does not "
                f"compute (it has {sorted(snapshot.values)}). Build the pipeline from "
                "the spec's own feature_requests."
            ) from None
    raise EvaluationError(f"not a term node: {type(node).__name__}")


def _describe(node: object) -> str:
    if isinstance(node, Constant):
        return str(node.value)
    if isinstance(node, FeatureRef):
        return node.feature_key
    return type(node).__name__


def pipeline_for(spec: StrategySpec) -> tuple[tuple[str, int], ...]:
    """The `(kind, lookback)` pairs a pipeline must compute for this spec.

    Exposed so the engine builds the pipeline *from the spec* rather than
    hoping a default pipeline happens to contain what the spec reads. A
    mismatch there would surface as every decision being `UNKNOWN`, which
    looks like a legitimately cautious strategy.
    """
    return spec.feature_requests


def check_features_available(spec: StrategySpec, available: Mapping[str, object]) -> None:
    """Raise unless every feature the spec reads is present.

    Called at registration. Without it, a spec referring to a missing feature
    is accepted, evaluates to `UNKNOWN` forever, never trades, and is recorded
    as a strategy that found no opportunities.
    """
    missing = [key for key in spec.required_features if key not in available]
    if missing:
        raise EvaluationError(
            f"spec {spec.name!r} reads features that the pipeline does not compute: "
            f"{sorted(missing)}. Rejected at registration rather than evaluating to "
            "UNKNOWN at every decision, which would be recorded as a strategy that "
            "simply never found an opportunity."
        )
