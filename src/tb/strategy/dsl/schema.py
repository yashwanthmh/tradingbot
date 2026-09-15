"""The strategy specification language, as validated data.

A spec is a Pydantic tree, not a string of code. That is the whole point: M6's
searcher — and optionally an LLM — generates these, and there is **no
`eval`/`exec` anywhere in the interpretation path**. A generated spec selects
operators from a fixed table and feature names from a fixed library; anything
it cannot express is unreachable rather than merely discouraged.

The grammar is deliberately small. It expresses "some comparisons of features
against constants and against each other, combined with and/or/not" and
nothing else. No loops, no user-defined functions, no arithmetic on arbitrary
expressions. A richer language would search a bigger space, which sounds like
an advantage and is not: the space is already vastly larger than the evidence
available to judge candidates on, so the binding constraint is multiplicity
(see M5's deflated Sharpe), not expressiveness.

Two structural bounds, both enforced at validation rather than at evaluation:

**Depth.** A tree deeper than `MAX_DEPTH` is refused. Unbounded nesting is the
obvious fuzzing attack on a recursive interpreter, and a spec that blows the
Python stack takes the trading loop with it.

**Node count.** A wide-but-shallow tree is the same attack. Both are checked
before the interpreter ever sees the spec, so the interpreter's own step budget
is a second line rather than the only one.
"""

from __future__ import annotations

from decimal import Decimal
from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from tb.core.canonical import hash_payload
from tb.core.errors import TbError
from tb.features.pipeline import FEATURE_LIBRARY

# A tree deeper than this is refused. Deep enough for anything meaningful,
# shallow enough that recursive evaluation cannot approach Python's stack.
MAX_DEPTH = 8
# Total nodes, so a wide tree is bounded too.
MAX_NODES = 64
# Lookbacks a spec may request. Bounded above because a 100,000-bar lookback
# is not a strategy, it is a denial of service against the reader; bounded
# below because a 0-bar lookback is meaningless.
MIN_LOOKBACK = 1
MAX_LOOKBACK = 400


class SpecError(TbError):
    """A specification was malformed, or asked for something outside the grammar."""


class _Node(BaseModel):
    """Base for every grammar node.

    `extra="forbid"` for the same reason as the hard limits: this is our own
    schema, and a generated spec carrying an unrecognised key is a searcher
    bug. Silently ignoring it would mean evaluating something other than what
    was proposed, while the spec hash recorded the proposal.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)


# --------------------------------------------------------------------------
# Terms — the leaves
# --------------------------------------------------------------------------


class FeatureRef(_Node):
    """A named feature at a named lookback."""

    kind: Literal["feature"] = "feature"
    name: str = Field(min_length=1, max_length=40)
    lookback: int = Field(ge=MIN_LOOKBACK, le=MAX_LOOKBACK)

    @model_validator(mode="after")
    def _check_in_library(self) -> FeatureRef:
        if self.name not in FEATURE_LIBRARY:
            raise ValueError(
                f"unknown feature {self.name!r}; the library is "
                f"{sorted(FEATURE_LIBRARY)}. Features are selected from a fixed table, "
                "never supplied as code, which is what makes generated specs safe to "
                "interpret without eval."
            )
        return self

    @property
    def feature_key(self) -> str:
        return f"{self.name}_{self.lookback}"


class Constant(_Node):
    """A literal number.

    `Decimal` rather than float, and finite-checked. A spec carrying `NaN`
    would make every comparison against it false, producing a strategy that
    silently never trades and looks like a legitimate negative result.
    """

    kind: Literal["const"] = "const"
    value: Decimal

    @model_validator(mode="after")
    def _check_finite(self) -> Constant:
        if not self.value.is_finite():
            raise ValueError(
                f"constant must be finite, got {self.value}. A NaN constant makes every "
                "comparison against it false, which reads as a strategy that legitimately "
                "declined to trade rather than as a malformed spec."
            )
        return self


Term = Annotated[FeatureRef | Constant, Field(discriminator="kind")]


# --------------------------------------------------------------------------
# Predicates — the branches
# --------------------------------------------------------------------------


class Comparison(_Node):
    """One feature or constant compared against another."""

    kind: Literal["compare"] = "compare"
    op: Literal["lt", "lte", "gt", "gte"]
    left: Term
    right: Term


class Not(_Node):
    kind: Literal["not"] = "not"
    operand: Predicate


class All(_Node):
    """Conjunction. Empty is refused rather than treated as vacuously true.

    An empty `all` evaluating to True is a spec that always enters, which is a
    plausible-looking strategy produced by a searcher bug.
    """

    kind: Literal["all"] = "all"
    operands: list[Predicate] = Field(min_length=1, max_length=MAX_NODES)


class Any_(_Node):
    kind: Literal["any"] = "any"
    operands: list[Predicate] = Field(min_length=1, max_length=MAX_NODES)


Predicate = Annotated[Comparison | Not | All | Any_, Field(discriminator="kind")]

All.model_rebuild()
Any_.model_rebuild()
Not.model_rebuild()


# --------------------------------------------------------------------------
# The spec
# --------------------------------------------------------------------------


def _depth(node: object) -> int:
    if isinstance(node, (Comparison, FeatureRef, Constant)):
        return 1
    if isinstance(node, Not):
        return 1 + _depth(node.operand)
    if isinstance(node, (All, Any_)):
        return 1 + max(_depth(operand) for operand in node.operands)
    return 1


def _count(node: object) -> int:
    if isinstance(node, Comparison):
        return 3  # itself plus two terms
    if isinstance(node, (FeatureRef, Constant)):
        return 1
    if isinstance(node, Not):
        return 1 + _count(node.operand)
    if isinstance(node, (All, Any_)):
        return 1 + sum(_count(operand) for operand in node.operands)
    return 1


def _features_in(node: object) -> set[tuple[str, int]]:
    if isinstance(node, FeatureRef):
        return {(node.name, node.lookback)}
    if isinstance(node, Constant):
        return set()
    if isinstance(node, Comparison):
        return _features_in(node.left) | _features_in(node.right)
    if isinstance(node, Not):
        return _features_in(node.operand)
    if isinstance(node, (All, Any_)):
        found: set[tuple[str, int]] = set()
        for operand in node.operands:
            found |= _features_in(operand)
        return found
    return set()


class StrategySpec(_Node):
    """A complete, self-describing strategy.

    `expected_edge_bps` is declared on the spec rather than computed by it.
    That is a deliberate constraint on the searcher: a spec must commit to a
    claim about its own edge before it is backtested, so the claim can be
    checked against the realised result instead of fitted to it.
    """

    spec_version: Literal[1] = 1
    name: str = Field(min_length=1, max_length=80)
    entry: Predicate
    exit: Predicate
    expected_edge_bps: Decimal = Field(gt=Decimal(0))
    # Not the hard limit — that is enforced by the risk layer. This is the
    # strategy's own intent, which the risk layer then intersects.
    min_holding_minutes: int = Field(ge=0, le=100_000, default=60)
    notes: str = Field(default="", max_length=500)

    @model_validator(mode="after")
    def _check_bounds(self) -> StrategySpec:
        for label, tree in (("entry", self.entry), ("exit", self.exit)):
            depth = _depth(tree)
            if depth > MAX_DEPTH:
                raise ValueError(
                    f"{label} tree is {depth} deep, over the {MAX_DEPTH} limit. Unbounded "
                    "nesting is the obvious attack on a recursive interpreter, and a spec "
                    "that exhausts the stack takes the trading loop with it."
                )
        total = _count(self.entry) + _count(self.exit)
        if total > MAX_NODES:
            raise ValueError(
                f"spec has {total} nodes, over the {MAX_NODES} limit. A wide shallow tree "
                "is the same denial of service as a deep one."
            )
        if not self.expected_edge_bps.is_finite():
            raise ValueError("expected_edge_bps must be finite")
        return self

    @property
    def required_features(self) -> tuple[str, ...]:
        """Feature keys this spec reads, as the pipeline names them."""
        pairs = _features_in(self.entry) | _features_in(self.exit)
        return tuple(sorted(f"{name}_{lookback}" for name, lookback in pairs))

    @property
    def feature_requests(self) -> tuple[tuple[str, int], ...]:
        """`(kind, lookback)` pairs, for building the pipeline this spec needs."""
        return tuple(sorted(_features_in(self.entry) | _features_in(self.exit)))

    @property
    def max_lookback(self) -> int:
        requests = self.feature_requests
        return max((lookback for _, lookback in requests), default=0)

    @property
    def n_nodes(self) -> int:
        return _count(self.entry) + _count(self.exit)

    @property
    def spec_hash(self) -> str:
        """Content hash over the canonical form.

        The identity M5's trial accounting keys on: two searchers proposing the
        same tree under different names are one trial, not two, and counting
        them twice would inflate every deflated metric computed from the count.
        """
        return hash_payload(self.model_dump(mode="json"))

    @classmethod
    def parse(cls, payload: object) -> StrategySpec:
        """Validate an untrusted payload into a spec.

        The single entry point for anything generated. Raises `SpecError` with
        the validation detail rather than letting a Pydantic error escape,
        because the searcher loop needs to log a rejection and continue rather
        than crash on its own output.
        """
        try:
            return cls.model_validate(payload)
        except Exception as exc:
            raise SpecError(f"invalid strategy spec: {exc}") from exc
