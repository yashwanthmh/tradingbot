"""Proposing specs: a fixed table of mutations over a tree that is already data.

The searcher's whole vocabulary. A proposal is produced by picking one operator
from the table below and applying it to a validated parent, and the result goes
back through `StrategySpec.parse` before anything else touches it. There is no
path from a proposal to executed code — the same property the DSL has, held at
the point where the *generator* is the untrusted party rather than a file on
disk.

Four things about the operator set are load-bearing.

**Every mutation is re-validated, not trusted because its parent was valid.**
`add_clause` can push a tree past the node cap and `shift_lookback` can ask for
more history than exists. A mutation returning an invalid tree is normal and is
the schema's business; a mutation *assumed* valid is how a searcher writes a
spec the interpreter then refuses at every decision.

**The set shrinks as well as grows.** `drop_clause` and `unwrap_not` exist
because a set with only additive operators walks monotonically to the node cap:
every `add_clause` is accepted until the ceiling, and complexity is exactly what
overfits a 300-bar training window. A searcher that can only grow is a searcher
whose best candidate is always its most complex one.

**A mutation that rediscovers its parent still costs a trial.** The registry
deduplicates by `spec_hash`, so the *strategy* is not duplicated — but the
multiplicity accounting counts everything tried, and inverting a mutation is not
a free lookup. `MutationProposer` therefore reports what it proposed, including
the duplicates, and leaves the deduplication to the registry.

**The declared edge is bounded by the hash-pinned file, not by a constant
here.** The cost gate divides cost by the strategy's own claim, so a proposer
free to claim 10,000bps would defeat the one control that keeps the search out
of the fee trap. `ProposalBounds.from_limits` is the only way to build the edge
ladder, and it reads `costs.min_expected_edge_bps` and
`costs.max_expected_edge_bps`.
"""

from __future__ import annotations

import random
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from decimal import Decimal
from typing import Any, Protocol, runtime_checkable

from tb.config.hard_limits import HardLimits
from tb.core.errors import TbError
from tb.features.pipeline import FEATURE_LIBRARY
from tb.registry.models import AuthorKind
from tb.strategy.dsl.schema import (
    MAX_LOOKBACK,
    MIN_LOOKBACK,
    SpecError,
    StrategySpec,
)

# Lookbacks a proposal may ask for. A ladder rather than a range: adjacent
# lookbacks produce almost identical features, so a searcher sampling 1..400
# uniformly spends most of its trial budget distinguishing a 173-bar average
# from a 174-bar one — and every trial spent is a larger multiplicity haircut on
# whatever it eventually finds (see docs/decisions/0002).
LOOKBACK_LADDER: tuple[int, ...] = (5, 10, 20, 50, 100, 200)

# The comparisons the grammar has. Named here so an operator that swaps one
# cannot invent a fifth.
COMPARISONS: tuple[str, ...] = ("lt", "lte", "gt", "gte")

FEATURES: tuple[str, ...] = tuple(sorted(FEATURE_LIBRARY))

# Steps a proposal may move a declared edge by, as multiples. Multiplicative
# rather than additive so the step is proportionate at 150bps and at 450.
EDGE_STEPS: tuple[str, ...] = ("0.7", "0.85", "1.2", "1.5")

# How many times a proposer retries when its chosen operator does not apply —
# there is nothing to drop in a single-comparison tree, nothing to unwrap
# without a `not`. Bounded rather than looping until success: a parent that no
# operator can change is a fact worth reporting, not one worth spinning on.
MAX_OPERATOR_ATTEMPTS = 12


class ProposalError(TbError):
    """A proposer could not produce a spec."""


@dataclass(frozen=True, slots=True)
class ProposalBounds:
    """What a proposal may ask for. Built from the limits, never from constants.

    `max_lookback` is separate from the grammar's own bound because the binding
    constraint is usually the *data*: a 200-bar average over a 150-bar training
    window is `UNKNOWN` at every decision, and a spec that never evaluates is
    recorded as a strategy that found no opportunities — a false negative
    indistinguishable from a true one. The caller passes the history it actually
    has.
    """

    min_edge_bps: Decimal
    max_edge_bps: Decimal
    lookbacks: tuple[int, ...] = LOOKBACK_LADDER
    min_holding_minutes: int = 1440

    def __post_init__(self) -> None:
        if not self.lookbacks:
            raise ProposalError(
                "no lookbacks are available to propose from. A proposer with an empty "
                "ladder would return nothing and read as a search that found no ideas."
            )
        for lookback in self.lookbacks:
            if not MIN_LOOKBACK <= lookback <= MAX_LOOKBACK:
                raise ProposalError(
                    f"lookback {lookback} is outside the grammar's bounds "
                    f"[{MIN_LOOKBACK}, {MAX_LOOKBACK}]"
                )
        if self.min_edge_bps <= 0 or self.max_edge_bps < self.min_edge_bps:
            raise ProposalError(
                f"edge bounds [{self.min_edge_bps}, {self.max_edge_bps}] are not a usable "
                "band; the grammar requires a positive declared edge"
            )

    @classmethod
    def from_limits(
        cls,
        limits: HardLimits,
        *,
        max_lookback: int | None = None,
        min_holding_minutes: int = 1440,
    ) -> ProposalBounds:
        """The bounds the hash-pinned file allows.

        `costs.max_expected_edge_bps` is the one that closes a hole rather than
        tuning anything: the cost gate divides the round trip by the strategy's
        own declared edge, so an unbounded claim passes it trivially.
        """
        ceiling = MAX_LOOKBACK if max_lookback is None else max_lookback
        return cls(
            min_edge_bps=Decimal(str(limits.costs.min_expected_edge_bps)),
            max_edge_bps=Decimal(str(limits.costs.max_expected_edge_bps)),
            lookbacks=tuple(look for look in LOOKBACK_LADDER if look <= ceiling)
            or (min(LOOKBACK_LADDER),),
            min_holding_minutes=min_holding_minutes,
        )

    def clamp_edge(self, value: Decimal) -> Decimal:
        return max(self.min_edge_bps, min(self.max_edge_bps, value))


@dataclass(frozen=True, slots=True)
class Proposal:
    """One spec, and how it came to exist.

    The provenance travels with the spec because both the trial log and the
    registry need it: `author_kind` decides which actor the registration is
    attributed to, and `parent_spec_hash` is what puts a mutation in its
    parent's lineage — which is what carries an exhausted loss budget across a
    rename.
    """

    spec: StrategySpec
    author_kind: AuthorKind
    operator: str
    parent_spec_hash: str | None = None
    detail: str = ""

    @property
    def spec_hash(self) -> str:
        return self.spec.spec_hash


@runtime_checkable
class SpecProposer(Protocol):
    """Where specs come from. One interface, three implementations.

    A random draw, a mutation of what already exists, and (optionally) an LLM.
    The searcher does not know which it has, which is the point: the LLM is a
    source of proposals and nothing else, and it reaches neither the broker nor
    the gate.
    """

    @property
    def name(self) -> str: ...

    def propose(
        self,
        *,
        n: int,
        rng: random.Random,
        parents: Sequence[StrategySpec] = (),
    ) -> list[Proposal]: ...


# --------------------------------------------------------------------------
# Drawing a spec from nothing
# --------------------------------------------------------------------------


def seed_spec(rng: random.Random, *, bounds: ProposalBounds, index: int = 0) -> StrategySpec:
    """A first-generation spec: one or two comparisons a side.

    Shallow deliberately. Depth costs evaluation and buys nothing the gate reads
    differently, and a searcher that starts deep has spent its trial budget on
    trees too complex for the evidence to judge.

    Built as a payload and passed through `parse`, not constructed directly, so
    a generated spec goes through exactly the validation an LLM's output would.
    Constructing the model here would skip the discriminated-union coercion and
    let this function produce trees the real path could not.
    """
    return StrategySpec.parse(
        {
            "name": f"search-{index:04d}",
            "entry": _random_predicate(rng, bounds),
            "exit": _random_predicate(rng, bounds),
            "expected_edge_bps": str(_random_edge(rng, bounds)),
            "min_holding_minutes": bounds.min_holding_minutes,
            "notes": "proposed by the deterministic searcher",
        }
    )


def _random_edge(rng: random.Random, bounds: ProposalBounds) -> Decimal:
    """A declared edge inside the permitted band.

    Drawn from the *upper* part of the band on purpose. Below roughly 121bps a
    US round trip is unaffordable at `max_cost_to_edge_ratio`, so a population
    drawn uniformly from 5bps up would be refused by the cost gate before
    reaching any statistical check — and a search whose rejections are all
    procedural has measured nothing about its own ideas.
    """
    low = bounds.clamp_edge(bounds.max_edge_bps * Decimal("0.3"))
    high = bounds.max_edge_bps
    step = (high - low) / Decimal(8) if high > low else Decimal(0)
    return bounds.clamp_edge((low + step * Decimal(rng.randint(0, 8))).quantize(Decimal("1")))


def _random_predicate(rng: random.Random, bounds: ProposalBounds) -> dict[str, object]:
    comparison = _random_comparison(rng, bounds)
    if rng.random() < 0.35:
        return {
            "kind": rng.choice(("all", "any")),
            "operands": [comparison, _random_comparison(rng, bounds)],
        }
    return comparison


def _random_comparison(rng: random.Random, bounds: ProposalBounds) -> dict[str, object]:
    return {
        "kind": "compare",
        "op": rng.choice(COMPARISONS),
        "left": _random_feature(rng, bounds),
        "right": _random_term(rng, bounds),
    }


def _random_term(rng: random.Random, bounds: ProposalBounds) -> dict[str, object]:
    if rng.random() < 0.3:
        return {"kind": "const", "value": str(round(rng.uniform(-5, 120), 3))}
    return _random_feature(rng, bounds)


def _random_feature(rng: random.Random, bounds: ProposalBounds) -> dict[str, object]:
    return {
        "kind": "feature",
        "name": rng.choice(FEATURES),
        "lookback": rng.choice(bounds.lookbacks),
    }


# --------------------------------------------------------------------------
# Mutating a spec that exists
# --------------------------------------------------------------------------
#
# Every operator works on the spec's **JSON payload**, not on the validated
# model, and the result goes back through `StrategySpec.parse`. That is the same
# discipline the rest of the DSL path holds to: a spec is data all the way down,
# and the schema is the only thing that decides whether a tree is a tree. The
# alternative — surgery on Pydantic instances via `model_copy(update=...)` —
# skips validation at exactly the point where the *generator* is the untrusted
# party.
#
# Each operator mutates its subtree in place and returns whether it applied.
# `False` is not an error: there is nothing to drop in a single comparison and
# nothing to unwrap without a `not`, so the caller tries another operator and a
# parent that no operator can change is reported rather than spun on.

_Tree = dict[str, Any]
_Operator = Callable[[_Tree, random.Random, ProposalBounds], bool]

_GROUPS = ("all", "any")


def _walk(node: _Tree) -> list[_Tree]:
    """Every predicate node in the subtree, as the live dictionaries.

    Flattened rather than descended with a coin flip per level, because a
    per-level flip biases hard toward the root — the operators that change the
    shape of a tree would then almost never reach a leaf of a deep one.

    The dictionaries are the ones inside `node`, so mutating a returned
    dictionary mutates the tree. That is how an operator changes the root as
    easily as a leaf, which a functional rebuild needs a special case for.
    """
    found: list[_Tree] = [node]
    kind = node.get("kind")
    if kind == "not":
        found.extend(_walk(node["operand"]))
    elif kind in _GROUPS:
        for operand in node["operands"]:
            found.extend(_walk(operand))
    return found


def _comparisons(node: _Tree) -> list[_Tree]:
    return [found for found in _walk(node) if found.get("kind") == "compare"]


def _terms(node: _Tree, kind: str) -> list[_Tree]:
    """Every `feature` or `const` term under a comparison."""
    return [
        term
        for comparison in _comparisons(node)
        for term in (comparison["left"], comparison["right"])
        if term.get("kind") == kind
    ]


def _become(node: _Tree, replacement: _Tree) -> None:
    """Rewrite `node` in place to be `replacement`.

    `clear` then `update` rather than returning a new dictionary, so an operator
    can replace the root of the tree it was handed. A rebuild-and-return scheme
    needs the caller to special-case "the node I replaced was the root", and
    that special case is where a mutation silently does nothing.
    """
    inner = dict(replacement)
    node.clear()
    node.update(inner)


def _swap_operator(tree: _Tree, rng: random.Random, bounds: ProposalBounds) -> bool:
    """Change one comparison's operator. The cheapest useful move.

    `gt` to `gte` is nearly a no-op and `gt` to `lt` inverts the idea. Both are
    in the table: a search needs the small step to refine and the large one to
    escape.
    """
    candidates = _comparisons(tree)
    if not candidates:
        return False
    target = rng.choice(candidates)
    others = [op for op in COMPARISONS if op != target.get("op")]
    if not others:  # pragma: no cover - the grammar has four
        return False
    target["op"] = rng.choice(others)
    return True


def _shift_lookback(tree: _Tree, rng: random.Random, bounds: ProposalBounds) -> bool:
    """Move one feature to a neighbouring rung of the lookback ladder."""
    features = _terms(tree, "feature")
    if not features:
        return False
    target = rng.choice(features)
    ladder = list(bounds.lookbacks)
    current = int(target["lookback"])
    if current in ladder and len(ladder) > 1:
        index = ladder.index(current)
        steps = [step for step in (-1, 1) if 0 <= index + step < len(ladder)]
        lookback = ladder[index + rng.choice(steps)]
    else:
        lookback = rng.choice(ladder)
    if lookback == current:
        return False
    target["lookback"] = lookback
    return True


def _swap_feature(tree: _Tree, rng: random.Random, bounds: ProposalBounds) -> bool:
    """Change which feature one term reads."""
    features = _terms(tree, "feature")
    if not features:
        return False
    target = rng.choice(features)
    others = [name for name in FEATURES if name != target.get("name")]
    if not others:  # pragma: no cover - the library has six
        return False
    target["name"] = rng.choice(others)
    return True


def _perturb_constant(tree: _Tree, rng: random.Random, bounds: ProposalBounds) -> bool:
    """Scale one literal threshold.

    Multiplicative, so the step is proportionate whether the threshold is 0.5 or
    120 — an additive step would be a rounding error against a price and a total
    rewrite against a z-score, and the grammar lets both appear as constants.
    """
    constants = _terms(tree, "const")
    if not constants:
        return False
    target = rng.choice(constants)
    try:
        current = Decimal(str(target["value"]))
        moved = (current * Decimal(rng.choice(EDGE_STEPS))).quantize(Decimal("0.001"))
    except ArithmeticError:
        # A constant the decimal context cannot scale — the schema bounds them
        # now, but a parent read from an older registry row may predate that.
        # "Does not apply" rather than a raise: one unrepresentable parent must
        # not take the whole search down, which is what M6's fuzzing found it
        # doing.
        return False
    if moved == current:
        return False
    target["value"] = str(moved)
    return True


def _add_clause(tree: _Tree, rng: random.Random, bounds: ProposalBounds) -> bool:
    """Conjoin or disjoin a fresh comparison.

    Applied at a randomly chosen node rather than at the root, so a search can
    refine a branch instead of only stacking clauses at the top. The node and
    depth caps stay the schema's to enforce: this may return a tree that exceeds
    them and `parse` refuses it, which is one rejection rather than a private
    copy of the grammar's arithmetic here.
    """
    fresh = _random_comparison(rng, bounds)
    target = rng.choice(_walk(tree))
    if target.get("kind") in _GROUPS:
        target["operands"].append(fresh)
        return True
    _become(
        target,
        {"kind": rng.choice(_GROUPS), "operands": [dict(target), fresh]},
    )
    return True


def _drop_clause(tree: _Tree, rng: random.Random, bounds: ProposalBounds) -> bool:
    """Remove one operand from an `all` or `any`.

    The inverse of `_add_clause`, and the reason the set is not purely additive:
    without it every generation is at least as complex as the last, and
    complexity is what fits a training window rather than a market.
    """
    droppable = [
        node for node in _walk(tree) if node.get("kind") in _GROUPS and len(node["operands"]) > 1
    ]
    if not droppable:
        return False
    target = rng.choice(droppable)
    target["operands"].pop(rng.randrange(len(target["operands"])))
    if len(target["operands"]) == 1:
        # A one-operand `all` is valid but says nothing, and leaving it would let
        # a chain of drops build a tower of single-operand groups that the node
        # cap eventually refuses for no visible reason.
        _become(target, target["operands"][0])
    return True


def _negate(tree: _Tree, rng: random.Random, bounds: ProposalBounds) -> bool:
    """Wrap a node in `not`, or unwrap one that already is.

    Both directions in one operator, because a table with only the wrapping half
    would accumulate negations a search could never remove.
    """
    target = rng.choice(_walk(tree))
    if target.get("kind") == "not":
        _become(target, target["operand"])
        return True
    _become(target, {"kind": "not", "operand": dict(target)})
    return True


MUTATIONS: dict[str, _Operator] = {
    "swap_operator": _swap_operator,
    "shift_lookback": _shift_lookback,
    "swap_feature": _swap_feature,
    "perturb_constant": _perturb_constant,
    "add_clause": _add_clause,
    "drop_clause": _drop_clause,
    "negate": _negate,
}


def mutate(
    spec: StrategySpec,
    *,
    rng: random.Random,
    bounds: ProposalBounds,
    index: int = 0,
) -> Proposal | None:
    """One mutation of `spec`, or `None` if no operator applied.

    Mutates one side or the other, or the declared edge — never several at once.
    A multi-change mutation makes the search's attribution useless: the point of
    recording the operator is that "shifting this lookback helped" is a fact the
    next generation can use, and a proposal that changed four things at once
    supports no such statement.
    """
    for _ in range(MAX_OPERATOR_ATTEMPTS):
        name = rng.choice(sorted(MUTATIONS))
        if rng.random() < 0.15:
            candidate = _mutate_edge(spec, rng=rng, bounds=bounds, index=index)
            if candidate is not None:
                return candidate
            continue
        side = "entry" if rng.random() < 0.5 else "exit"
        payload = spec.model_dump(mode="json")
        if not MUTATIONS[name](payload[side], rng, bounds):
            continue
        payload["name"] = f"mut-{index:04d}-{name}"
        payload["notes"] = f"{name} on {side} of {spec.spec_hash[:10]}"
        try:
            child = StrategySpec.parse(payload)
        except SpecError:
            # The mutation broke a grammar bound — usually the node cap after an
            # `add_clause`. Not an error: the schema is what decides, and a
            # searcher that assumed its own output valid is how an unevaluable
            # spec reaches the gate.
            continue
        if child.spec_hash == spec.spec_hash:
            continue
        return Proposal(
            spec=child,
            author_kind=AuthorKind.MUTATION,
            operator=f"{name}:{side}",
            parent_spec_hash=spec.spec_hash,
            detail=f"{name} applied to the {side} tree",
        )
    return None


def _mutate_edge(
    spec: StrategySpec,
    *,
    rng: random.Random,
    bounds: ProposalBounds,
    index: int,
) -> Proposal | None:
    """Change what the strategy claims to earn.

    A mutation like any other, and one the gate reads directly: the cost gate
    divides the round trip by this number, so raising it is how a search makes a
    fast idea look affordable. It is bounded by `costs.max_expected_edge_bps`,
    which is why that ceiling exists.
    """
    factor = Decimal(rng.choice(EDGE_STEPS))
    try:
        moved = bounds.clamp_edge((spec.expected_edge_bps * factor).quantize(Decimal("1")))
    except ArithmeticError:
        # An edge the decimal context cannot scale. The schema bounds the
        # declared edge now; a seed read from a registry row that predates the
        # bound may not be, and — as with `_perturb_constant` — one such parent
        # must not take the whole search down.
        return None
    if moved == spec.expected_edge_bps:
        return None
    payload = spec.model_dump(mode="json")
    payload["expected_edge_bps"] = str(moved)
    payload["name"] = f"mut-{index:04d}-edge"
    payload["notes"] = f"declared edge {spec.expected_edge_bps} -> {moved}"
    try:
        child = StrategySpec.parse(payload)
    except SpecError:  # pragma: no cover - clamped into the permitted band above
        return None
    return Proposal(
        spec=child,
        author_kind=AuthorKind.MUTATION,
        operator="adjust_edge",
        parent_spec_hash=spec.spec_hash,
        detail=f"declared edge {spec.expected_edge_bps} -> {moved}",
    )


def crossover(
    left: StrategySpec,
    right: StrategySpec,
    *,
    rng: random.Random,
    index: int = 0,
) -> Proposal | None:
    """One parent's entry with the other's exit.

    The only recombination here, and it is the one that means something in this
    grammar: an entry rule and an exit rule are separable ideas, so pairing a
    good entry with a different exit is a hypothesis rather than a shuffle.
    Splicing subtrees *within* a predicate would mostly produce trees whose two
    halves disagree about scale — comparing a z-score against a price — which the
    random draw already generates plenty of.
    """
    if left.spec_hash == right.spec_hash:
        return None
    payload = left.model_dump(mode="json")
    payload["exit"] = right.model_dump(mode="json")["exit"]
    payload["name"] = f"cross-{index:04d}"
    payload["expected_edge_bps"] = str(min(left.expected_edge_bps, right.expected_edge_bps))
    payload["notes"] = f"entry from {left.spec_hash[:10]}, exit from {right.spec_hash[:10]}"
    try:
        child = StrategySpec.parse(payload)
    except SpecError:
        return None
    if child.spec_hash in (left.spec_hash, right.spec_hash):
        return None
    return Proposal(
        spec=child,
        author_kind=AuthorKind.MUTATION,
        operator="crossover",
        # The entry's parent, because lineage has to be single-valued: the
        # budget a lineage carries is the reason a mutation cannot choose its
        # own, and a child claiming two lineages could be funded out of the one
        # that still has budget left.
        parent_spec_hash=left.spec_hash,
        detail=f"entry from {left.spec_hash[:10]}, exit from {right.spec_hash[:10]}",
    )


# --------------------------------------------------------------------------
# The two proposers
# --------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class RandomProposer:
    """Draws from the grammar with no reference to what came before.

    Generation zero, and the fallback whenever a mutation proposer has no
    parents to work from. Kept as its own proposer rather than a branch inside
    the mutator so a search can be run purely random — which is the control a
    genetic search has to beat to be worth its extra machinery.
    """

    bounds: ProposalBounds
    name: str = "random"

    def propose(
        self,
        *,
        n: int,
        rng: random.Random,
        parents: Sequence[StrategySpec] = (),
    ) -> list[Proposal]:
        return [
            Proposal(
                spec=seed_spec(rng, bounds=self.bounds, index=index),
                author_kind=AuthorKind.SEARCH,
                operator="random",
                detail="drawn from the grammar",
            )
            for index in range(n)
        ]


@dataclass(frozen=True, slots=True)
class MutationProposer:
    """Mutates and recombines the parents it is given.

    Falls back to a random draw when it has no parents or when no operator
    applies — a generation that returned fewer specs than asked for would make
    the trial count depend on the shape of the survivors, and the trial count is
    the denominator of every deflated metric downstream.
    """

    bounds: ProposalBounds
    crossover_rate: float = 0.25
    name: str = "mutation"

    def propose(
        self,
        *,
        n: int,
        rng: random.Random,
        parents: Sequence[StrategySpec] = (),
    ) -> list[Proposal]:
        if not parents:
            return RandomProposer(bounds=self.bounds).propose(n=n, rng=rng)
        out: list[Proposal] = []
        for index in range(n):
            proposal: Proposal | None = None
            if len(parents) > 1 and rng.random() < self.crossover_rate:
                left, right = rng.sample(list(parents), 2)
                proposal = crossover(left, right, rng=rng, index=index)
            if proposal is None:
                proposal = mutate(
                    rng.choice(list(parents)), rng=rng, bounds=self.bounds, index=index
                )
            if proposal is None:
                # No operator applied to the parent this draw picked. A random
                # spec rather than a short generation, for the reason in the
                # class docstring.
                proposal = Proposal(
                    spec=seed_spec(rng, bounds=self.bounds, index=index),
                    author_kind=AuthorKind.SEARCH,
                    operator="random_fallback",
                    detail="no mutation operator applied to the chosen parent",
                )
            out.append(proposal)
        return out
