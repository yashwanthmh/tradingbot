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
from tb.features.pipeline import FEATURE_LIBRARY, FEATURE_PLACES

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
# The largest threshold a constant may state. Above the highest share price
# any listed instrument trades at by several orders, and far below the point
# where the decimal context overflows on a multiplication — see `Constant`.
MAX_CONSTANT_MAGNITUDE = Decimal("1E+9")
# The largest edge a spec may declare: a round trip that doubles the money.
# Not the tradable band — `costs.max_expected_edge_bps` in the hash-pinned file
# is that, and the validator enforces it — but the point past which a number is
# not a claim at all. It exists for the same reason as the constant bound: the
# searcher does arithmetic on the declared edge, and `1E+999999999` overflowed
# the edge mutation the way it overflowed the constant one.
MAX_DECLARED_EDGE_BPS = Decimal("10000")


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

    **Bounded in magnitude and in precision**, and the bounds come from what a
    constant is *for*: it is a threshold a feature is compared against, and
    every feature in the library is a price, a percentage or a z-score. A
    constant of `1E+999999999` is finite, so the finiteness check alone admitted
    it — and the first arithmetic anything did on it raised `decimal.Overflow`.
    M6's fuzzing found exactly that: a mutation perturbing such a constant took
    the whole search down. Beyond `MAX_CONSTANT_MAGNITUDE`, or finer than the
    `FEATURE_PLACES` every feature is rounded to, a constant is a threshold no
    feature can cross, so it is refused as malformed rather than admitted as a
    strategy that silently never fires.
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
        if abs(self.value) > MAX_CONSTANT_MAGNITUDE:
            raise ValueError(
                f"constant {self.value} is beyond {MAX_CONSTANT_MAGNITUDE}. Every feature in "
                "the library is a price, a percentage or a z-score, so no feature can cross "
                "this threshold — and arithmetic on it overflows the decimal context."
            )
        exponent = self.value.as_tuple().exponent
        if isinstance(exponent, int) and exponent < -FEATURE_PLACES:
            raise ValueError(
                f"constant {self.value} has more decimal places than the {FEATURE_PLACES} "
                "every feature is rounded to, so the comparison cannot resolve it."
            )
        return self


class ModelScore(_Node):
    """A recorded model's score: its probability that a trade here profits after costs.

    The model is not run by the interpreter. It is a scorer inside the feature
    pipeline, fed that pipeline's own features at the same instant, and its
    output arrives in the snapshot under `model_id` like any other feature — so
    this term is read exactly as a `FeatureRef` is, and a generated spec gains
    no new capability by naming one.

    **Pinned twice.** `model_id` finds the record; `artifact_sha256` is the
    bytes the spec was trained, backtested and gated with. A store holding
    anything else under that id is refused when the pipeline is built, because
    an id alone would let a retrained model inherit a promotion it never
    earned. The id is derived from the hash (`mdl_` and its first sixteen hex
    digits), and a pair that disagrees is refused here, so one model cannot be
    named with another's hash.
    """

    kind: Literal["model"] = "model"
    model_id: str = Field(pattern=r"^mdl_[0-9a-f]{16}$")
    artifact_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")

    @model_validator(mode="after")
    def _check_id_is_the_hash(self) -> ModelScore:
        if self.model_id != f"mdl_{self.artifact_sha256[:16]}":
            raise ValueError(
                f"model id {self.model_id} is not the one artifact {self.artifact_sha256[:16]}… "
                "derives: a model is named by its own hash, never by another's"
            )
        return self

    @property
    def feature_key(self) -> str:
        return self.model_id


Term = Annotated[FeatureRef | Constant | ModelScore, Field(discriminator="kind")]


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
    if isinstance(node, (Comparison, FeatureRef, Constant, ModelScore)):
        return 1
    if isinstance(node, Not):
        return 1 + _depth(node.operand)
    if isinstance(node, (All, Any_)):
        return 1 + max(_depth(operand) for operand in node.operands)
    return 1


def _count(node: object) -> int:
    if isinstance(node, Comparison):
        return 3  # itself plus two terms
    if isinstance(node, (FeatureRef, Constant, ModelScore)):
        return 1
    if isinstance(node, Not):
        return 1 + _count(node.operand)
    if isinstance(node, (All, Any_)):
        return 1 + sum(_count(operand) for operand in node.operands)
    return 1


def _models_in(node: object) -> set[ModelScore]:
    if isinstance(node, ModelScore):
        return {node}
    if isinstance(node, Comparison):
        return _models_in(node.left) | _models_in(node.right)
    if isinstance(node, Not):
        return _models_in(node.operand)
    if isinstance(node, (All, Any_)):
        found: set[ModelScore] = set()
        for operand in node.operands:
            found |= _models_in(operand)
        return found
    return set()


def _features_in(node: object) -> set[tuple[str, int]]:
    if isinstance(node, FeatureRef):
        return {(node.name, node.lookback)}
    if isinstance(node, (Constant, ModelScore)):
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
        if self.expected_edge_bps > MAX_DECLARED_EDGE_BPS:
            raise ValueError(
                f"expected_edge_bps {self.expected_edge_bps} is beyond {MAX_DECLARED_EDGE_BPS}: "
                "a round trip that more than doubles the money is not a claim a spec can make, "
                "and arithmetic on a number that size overflows the decimal context."
            )
        return self

    @property
    def required_features(self) -> tuple[str, ...]:
        """Snapshot keys this spec reads, as the pipeline names them: its
        features and the scores of any models it reads."""
        pairs = _features_in(self.entry) | _features_in(self.exit)
        keys = {f"{name}_{lookback}" for name, lookback in pairs}
        keys |= {model.feature_key for model in self.model_refs}
        return tuple(sorted(keys))

    @property
    def feature_requests(self) -> tuple[tuple[str, int], ...]:
        """`(kind, lookback)` pairs this spec reads directly.

        Not the whole pipeline when the spec reads a model: the model's own
        inputs are on its record, and `pipeline_from_spec` adds them.
        """
        return tuple(sorted(_features_in(self.entry) | _features_in(self.exit)))

    @property
    def model_refs(self) -> tuple[ModelScore, ...]:
        """The models this spec reads, each once, pinned by artifact hash."""
        found = _models_in(self.entry) | _models_in(self.exit)
        return tuple(sorted(found, key=lambda model: model.model_id))

    @property
    def max_lookback(self) -> int:
        """The longest lookback among the features this spec reads directly.

        A spec reading a model needs its model's inputs too, which only the
        built pipeline knows; its `max_lookback` is the one to size a warm-up by.
        """
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
