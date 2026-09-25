"""The searcher: proposals, refusals, and the budget that bounds both.

Three groups, matching the three modules:

* **mutation** — every operator applies, the result is always a valid spec, the
  tree can shrink as well as grow, and a seed reproduces a generation exactly;
* **validation** — each rejection code fires for its own reason, and a duplicate
  is a rejection rather than a silent skip, because a dropped proposal shrinks
  the denominator of every deflated metric downstream;
* **search** — the budget bounds the trial count, survivors breed, an evaluator
  that raises is counted rather than fatal, and the Sharpe a search of a given
  size must show grows with the size.

The last of those is the point of the whole milestone. `docs/decisions/0002`
measured it: the multiplicity haircut is about 1.6 Sharpe at ten trials and 3.3
at a thousand, so a search's size decides what edge it is *able* to prove. A
searcher that maximises coverage optimises against itself, and
`test_required_sharpe_grows_with_the_trial_count` is what keeps that arithmetic
from being quietly flattened.
"""

from __future__ import annotations

import random
from dataclasses import replace
from datetime import UTC, datetime
from decimal import Decimal

import pytest

from tb.backtest.engine import BacktestResult
from tb.backtest.metrics import Metrics
from tb.config.loader import load_hard_limits
from tb.data.provider import Resolution
from tb.registry.models import AuthorKind, TrialOutcome
from tb.research.mutate import (
    COMPARISONS,
    FEATURES,
    LOOKBACK_LADDER,
    MUTATIONS,
    MutationProposer,
    ProposalBounds,
    RandomProposer,
    crossover,
    mutate,
)
from tb.research.searcher import (
    MIN_TRADES_FOR_FITNESS,
    Candidate,
    SearchBudget,
    Searcher,
    required_sharpe,
)
from tb.research.validate import SpecValidator
from tb.strategy.dsl.schema import MAX_NODES, StrategySpec

LIMITS = load_hard_limits(None).limits
BOUNDS = ProposalBounds.from_limits(LIMITS, max_lookback=100)
AS_OF = datetime(2026, 6, 1, tzinfo=UTC)


def a_spec(
    *,
    name: str = "base",
    edge: str = "450",
    hold: int = 1440,
    lookback: int = 20,
    slow: int = 100,
) -> StrategySpec:
    return StrategySpec.parse(
        {
            "name": name,
            "entry": {
                "kind": "compare",
                "op": "gt",
                "left": {"kind": "feature", "name": "sma", "lookback": lookback},
                "right": {"kind": "feature", "name": "sma", "lookback": slow},
            },
            "exit": {
                "kind": "compare",
                "op": "lt",
                "left": {"kind": "feature", "name": "sma", "lookback": lookback},
                "right": {"kind": "feature", "name": "sma", "lookback": slow},
            },
            "expected_edge_bps": edge,
            "min_holding_minutes": hold,
        }
    )


# --------------------------------------------------------------------------
# Mutation
# --------------------------------------------------------------------------


def test_every_mutation_produces_a_valid_spec() -> None:
    """The invariant the whole module rests on.

    Every operator works on the spec's JSON payload and the result goes back
    through `StrategySpec.parse`, so an operator that breaks a grammar bound
    produces a rejection rather than an invalid tree. Asserted over enough draws
    to hit every operator many times.
    """
    rng = random.Random(1)
    parent = a_spec()
    produced = 0
    for index in range(400):
        proposal = mutate(parent, rng=rng, bounds=BOUNDS, index=index)
        if proposal is None:
            continue
        produced += 1
        # Re-parsing an already-valid spec is cheap and is the assertion: if the
        # mutation had produced something the grammar refuses, `mutate` would
        # have returned None rather than this.
        StrategySpec.parse(proposal.spec.model_dump(mode="json"))
        assert proposal.spec.n_nodes <= MAX_NODES
        assert proposal.spec.spec_hash != parent.spec_hash
        assert proposal.parent_spec_hash == parent.spec_hash
        assert proposal.author_kind is AuthorKind.MUTATION
    assert produced > 350, f"only {produced} of 400 mutations applied at all"


def a_rich_spec() -> StrategySpec:
    """A parent every operator can act on: a group, and a constant.

    Which operators apply is a property of the *parent*, not of the table —
    there is nothing to drop in a single comparison and no constant to perturb
    in a feature-versus-feature one. A reachability test therefore has to hand
    the table something it can work with, or it measures the fixture.
    """
    return StrategySpec.parse(
        {
            "name": "rich",
            "entry": {
                "kind": "all",
                "operands": [
                    a_spec().entry.model_dump(mode="json"),
                    {
                        "kind": "compare",
                        "op": "gt",
                        "left": {"kind": "feature", "name": "zscore", "lookback": 20},
                        "right": {"kind": "const", "value": "1.5"},
                    },
                ],
            },
            "exit": {
                "kind": "any",
                "operands": [
                    a_spec().exit.model_dump(mode="json"),
                    {
                        "kind": "compare",
                        "op": "lt",
                        "left": {"kind": "feature", "name": "zscore", "lookback": 20},
                        "right": {"kind": "const", "value": "-1.5"},
                    },
                ],
            },
            "expected_edge_bps": "450",
            "min_holding_minutes": 1440,
        }
    )


def test_every_operator_is_reachable() -> None:
    """A table with an unreachable entry is a table with a typo.

    The operator names are recorded on every proposal and read back by the
    review cycle, so an operator that never fires is a claim about the search
    space that is not true of the search.
    """
    rng = random.Random(2)
    parent = a_rich_spec()
    seen: set[str] = set()
    for index in range(600):
        proposal = mutate(parent, rng=rng, bounds=BOUNDS, index=index)
        if proposal is not None:
            seen.add(proposal.operator.split(":")[0])
    assert set(MUTATIONS) <= seen, f"never fired: {sorted(set(MUTATIONS) - seen)}"
    assert "adjust_edge" in seen


def test_which_operators_apply_is_a_property_of_the_parent() -> None:
    """And a plain parent is the common case, so it must still mutate.

    A single feature-versus-feature comparison has nothing to drop and no
    constant to perturb. Those operators returning `False` rather than raising
    is what lets `mutate` try another one instead of failing the draw.
    """
    rng = random.Random(21)
    plain = a_spec()
    applied = {
        proposal.operator.split(":")[0]
        for index in range(300)
        if (proposal := mutate(plain, rng=rng, bounds=BOUNDS, index=index)) is not None
    }
    assert "drop_clause" not in applied
    assert "perturb_constant" not in applied
    assert {"swap_operator", "shift_lookback", "swap_feature", "add_clause"} <= applied


def test_a_tree_can_shrink_as_well_as_grow() -> None:
    """**Why the operator set is not purely additive.**

    Every `add_clause` is accepted by the schema until the node cap, so a table
    with no `drop_clause` walks monotonically to the most complex tree the
    grammar allows — and complexity is exactly what fits a 300-bar training
    window rather than a market.
    """
    rng = random.Random(3)
    # A parent with room to shrink: two clauses a side.
    wide = StrategySpec.parse(
        {
            "name": "wide",
            "entry": {
                "kind": "all",
                "operands": [
                    a_spec().entry.model_dump(mode="json"),
                    a_spec(lookback=5, slow=50).entry.model_dump(mode="json"),
                ],
            },
            "exit": a_spec().exit.model_dump(mode="json"),
            "expected_edge_bps": "450",
            "min_holding_minutes": 1440,
        }
    )
    smaller = [
        proposal
        for index in range(200)
        if (proposal := mutate(wide, rng=rng, bounds=BOUNDS, index=index)) is not None
        and proposal.spec.n_nodes < wide.n_nodes
    ]
    assert smaller, "no mutation ever produced a smaller tree"


def test_a_maximal_parent_still_mutates_within_the_bounds() -> None:
    """An `add_clause` on a full tree is refused, and another operator applies.

    The case that would otherwise produce an invalid spec: the grammar's node cap
    is checked by `parse`, and a searcher that assumed its own output valid would
    hand the interpreter a tree it refuses at every decision.
    """
    rng = random.Random(4)
    clause = a_spec().entry.model_dump(mode="json")
    # Nine comparisons a side is 27 nodes each — just inside the 64-node cap, so
    # most `add_clause` mutations push it over.
    big = StrategySpec.parse(
        {
            "name": "big",
            "entry": {"kind": "all", "operands": [clause] * 9},
            "exit": {"kind": "any", "operands": [clause] * 9},
            "expected_edge_bps": "450",
            "min_holding_minutes": 1440,
        }
    )
    assert big.n_nodes <= MAX_NODES
    for index in range(100):
        proposal = mutate(big, rng=rng, bounds=BOUNDS, index=index)
        if proposal is not None:
            assert proposal.spec.n_nodes <= MAX_NODES


def test_the_same_seed_proposes_the_same_generation() -> None:
    """Reproducibility, which is what makes a failed search debuggable.

    A search is a sequence of random draws; without this, a promotion that came
    out of generation four could not be reconstructed, and the trial count that
    haircut it would be unverifiable.
    """
    parents = [a_spec(name="a"), a_spec(name="b", lookback=10)]
    first = MutationProposer(bounds=BOUNDS).propose(n=15, rng=random.Random(9), parents=parents)
    again = MutationProposer(bounds=BOUNDS).propose(n=15, rng=random.Random(9), parents=parents)
    assert [p.spec_hash for p in first] == [p.spec_hash for p in again]
    assert [p.operator for p in first] == [p.operator for p in again]


def test_a_random_draw_is_distinct_and_valid() -> None:
    proposals = RandomProposer(bounds=BOUNDS).propose(n=60, rng=random.Random(10))
    assert len({p.spec_hash for p in proposals}) == 60
    for proposal in proposals:
        assert proposal.author_kind is AuthorKind.SEARCH
        assert proposal.spec.required_features, "a drawn spec must read a feature"
        assert BOUNDS.min_edge_bps <= proposal.spec.expected_edge_bps <= BOUNDS.max_edge_bps


def test_crossover_pairs_one_entry_with_another_exit() -> None:
    """The only recombination, and the one that means something here.

    An entry rule and an exit rule are separable ideas, so pairing a good entry
    with a different exit is a hypothesis. Splicing subtrees *within* a predicate
    would mostly produce halves that disagree about scale, which the random draw
    already generates plenty of.
    """
    left = a_spec(name="left", lookback=20, slow=100, edge="400")
    right = a_spec(name="right", lookback=5, slow=50, edge="300")
    child = crossover(left, right, rng=random.Random(11))

    assert child is not None
    assert child.spec.entry == left.entry
    assert child.spec.exit == right.exit
    # The lower of the two claims. A child that inherited the larger one would
    # let a search raise its declared edge by recombination, and the cost gate
    # divides by that declaration.
    assert child.spec.expected_edge_bps == Decimal("300")
    assert child.parent_spec_hash == left.spec_hash


def test_crossover_with_itself_produces_nothing() -> None:
    spec = a_spec()
    assert crossover(spec, spec, rng=random.Random(12)) is None


def test_the_proposal_bounds_come_from_the_pinned_limits() -> None:
    """The edge ceiling closes a hole rather than tuning anything.

    The cost gate divides the round trip by the strategy's own declared edge, so
    a proposer free to claim 10,000bps defeats the one control that keeps the
    search out of the fee trap.
    """
    assert BOUNDS.max_edge_bps == Decimal(str(LIMITS.costs.max_expected_edge_bps))
    assert BOUNDS.min_edge_bps == Decimal(str(LIMITS.costs.min_expected_edge_bps))
    assert BOUNDS.clamp_edge(Decimal("100000")) == BOUNDS.max_edge_bps
    assert BOUNDS.clamp_edge(Decimal("0.0001")) == BOUNDS.min_edge_bps


def test_the_bounds_drop_lookbacks_the_history_cannot_support() -> None:
    """A 200-bar average over 150 bars is `UNKNOWN`, not a shorter average.

    Which reads as a strategy that declined to trade — a false negative
    indistinguishable from a true one — so the ladder is trimmed to what the
    window can actually evaluate.
    """
    short = ProposalBounds.from_limits(LIMITS, max_lookback=20)
    assert max(short.lookbacks) <= 20
    assert set(short.lookbacks) <= set(LOOKBACK_LADDER)
    # Never empty: an empty ladder would make the proposer return nothing, which
    # reads as a search that found no ideas.
    assert ProposalBounds.from_limits(LIMITS, max_lookback=1).lookbacks


def test_the_operator_and_feature_tables_are_the_grammar_s() -> None:
    """A fifth comparison or a seventh feature here would be a spec the
    interpreter refuses at every decision."""
    from tb.features.pipeline import FEATURE_LIBRARY
    from tb.strategy.dsl.interpreter import _COMPARISONS

    assert set(COMPARISONS) == set(_COMPARISONS)
    assert set(FEATURES) == set(FEATURE_LIBRARY)


# --------------------------------------------------------------------------
# Validation
# --------------------------------------------------------------------------


def test_a_spec_the_gate_would_accept_is_not_pre_rejected() -> None:
    """The control. A validator that refused everything would pass every other
    test in this section while measuring nothing."""
    validator = SpecValidator(limits=LIMITS, n_training_bars=300)
    assert validator.check(a_spec()) is None


def test_an_edge_the_costs_eat_is_rejected_before_the_backtest() -> None:
    """The binding constraint on this venue, applied before spending anything.

    A US round trip is about 40bps at `max_cost_to_edge_ratio` 0.33, so a spec
    needs roughly 121bps to be admissible at all. Most of a random population
    fails here, and failing before the backtest is what makes a large cycle
    affordable.
    """
    rejection = SpecValidator(limits=LIMITS).check(a_spec(edge="30"))
    assert rejection is not None
    assert rejection.code == "cost_to_edge"
    assert "fee schedule" in rejection.reason


def test_an_edge_above_the_ceiling_is_rejected() -> None:
    """The ceiling is what stops a spec claiming its way through the cost gate."""
    over = Decimal(str(LIMITS.costs.max_expected_edge_bps)) + Decimal(1)
    rejection = SpecValidator(limits=LIMITS).check(a_spec(edge=str(over)))
    assert rejection is not None
    assert rejection.code == "edge_band"


def test_a_holding_period_no_permitted_resolution_can_express_is_rejected() -> None:
    """Unexecutable rather than expensive.

    Minute resolution is refused on measured evidence, so a spec implying a
    30-minute hold cannot be acted on at all — and without this check the
    searcher breeds a whole family the loop can never trade and records every
    one as a cost failure.
    """
    rejection = SpecValidator(limits=LIMITS).check(a_spec(hold=30))
    assert rejection is not None
    assert rejection.code == "holding_period"


def test_a_lookback_longer_than_the_training_window_is_rejected() -> None:
    rejection = SpecValidator(limits=LIMITS, n_training_bars=60).check(
        a_spec(lookback=20, slow=200)
    )
    assert rejection is not None
    assert rejection.code == "insufficient_history"


def test_a_spec_that_reads_no_feature_is_rejected() -> None:
    """Always-true enters on every bar; always-false never trades.

    Both are searcher output rather than strategies, and the grammar permits
    `const > const`, so something has to refuse it.
    """
    constant = StrategySpec.parse(
        {
            "name": "constant",
            "entry": {
                "kind": "compare",
                "op": "gt",
                "left": {"kind": "const", "value": "2"},
                "right": {"kind": "const", "value": "1"},
            },
            "exit": {
                "kind": "compare",
                "op": "lt",
                "left": {"kind": "const", "value": "2"},
                "right": {"kind": "const", "value": "1"},
            },
            "expected_edge_bps": "450",
            "min_holding_minutes": 1440,
        }
    )
    rejection = SpecValidator(limits=LIMITS).check(constant)
    assert rejection is not None
    assert rejection.code == "no_feature_read"


def test_a_duplicate_is_a_rejection_not_a_silent_skip() -> None:
    """**Counting it is the point.**

    Rediscovering a parent is what mutation does — invert a mutation and you are
    back where you started. The registry deduplicates the *strategy*, but the
    trial still happened, and not counting it would let a search lower its own
    multiplicity by proposing in circles.
    """
    spec = a_spec()
    rejection = SpecValidator(limits=LIMITS).check(spec, seen={spec.spec_hash})
    assert rejection is not None
    assert rejection.code == "duplicate"
    assert "counts as a trial" in rejection.reason


def test_a_missing_feed_noise_figure_does_not_reject_the_population() -> None:
    """The asymmetry with the gate, and it is deliberate.

    At the gate an unmeasured error bar must not read as a small one, so it
    refuses. Here, refusing every proposal because no bake-off has run would
    report a search that rejected everything for a reason no spec could have
    avoided — the measurement is about the dataset, not the idea.
    """
    assert SpecValidator(limits=LIMITS, feed_noise_p95_bps=None).check(a_spec()) is None
    noisy = SpecValidator(limits=LIMITS, feed_noise_p95_bps=Decimal("200"))
    rejection = noisy.check(a_spec(edge="450"))
    assert rejection is not None
    assert rejection.code == "edge_to_feed_noise"


# --------------------------------------------------------------------------
# The search
# --------------------------------------------------------------------------


def _metrics(*, sharpe: float | None, trades: int) -> Metrics:
    return Metrics(
        n_periods=300,
        n_trades=trades,
        gross_return_pct=Decimal("5"),
        net_return_pct=Decimal("4"),
        gross_sharpe=sharpe,
        net_sharpe=sharpe,
        max_drawdown_pct=Decimal("6"),
        total_cost_ccy=Decimal("3"),
        cost_drag_bps=Decimal("40"),
        turnover=Decimal("2"),
    )


def _result(spec: StrategySpec, *, sharpe: float | None, trades: int) -> BacktestResult:
    return BacktestResult(
        backtest_id=f"bt_{spec.spec_hash[:10]}",
        strategy_id=f"stg_{spec.spec_hash[:12]}",
        strategy_version=1,
        resolution=Resolution.DAILY,
        rng_seed=0,
        window_start=AS_OF,
        window_end=AS_OF,
        metrics=_metrics(sharpe=sharpe, trades=trades),
        trades=(),
        curve=(),
        n_decisions=300,
        n_signals=trades,
        n_rejected_by_cost_gate=0,
        n_unevaluable=0,
        n_dropped_no_next_bar=0,
        starting_equity_ccy=Decimal("1000"),
    )


def _searcher(
    *,
    evaluate: object,
    budget: SearchBudget,
    n_training_bars: int | None = 300,
) -> Searcher:
    from collections.abc import Callable
    from typing import cast

    return Searcher(
        evaluate=cast(Callable[[StrategySpec], BacktestResult], evaluate),
        validator=SpecValidator(limits=LIMITS, n_training_bars=n_training_bars),
        initial_proposer=RandomProposer(bounds=BOUNDS),
        mutation_proposer=MutationProposer(bounds=BOUNDS),
        budget=budget,
        min_deflated_sharpe=LIMITS.promotion.min_oos_deflated_sharpe,
        search_id="srch_test",
    )


def test_the_budget_bounds_the_trial_count() -> None:
    """The budget is the total, not a per-generation figure.

    Expressed the other way round, a caller could add generations without
    noticing the multiplicity growing — and the multiplicity is what the haircut
    is computed from.
    """
    searcher = _searcher(
        evaluate=lambda spec: _result(spec, sharpe=1.0, trades=30),
        budget=SearchBudget(n_trials=37, n_per_generation=10, seed=1),
    )
    outcome = searcher.run()

    assert outcome.n_proposed == 37
    assert outcome.generations == 4
    assert outcome.n_evaluated + outcome.n_rejected + outcome.n_errored == 37


def test_later_generations_breed_from_the_survivors() -> None:
    """Generation one is a random draw; everything after it has a parent.

    Without this the "genetic" half of the searcher is decoration, and a
    mutation's lineage — which is what carries an exhausted loss budget across a
    rename — would never be set.
    """
    searcher = _searcher(
        evaluate=lambda spec: _result(spec, sharpe=1.0, trades=30),
        budget=SearchBudget(n_trials=40, n_per_generation=10, seed=2),
    )
    outcome = searcher.run()

    first = [c for c in outcome.candidates if c.generation == 1]
    later = [c for c in outcome.candidates if c.generation > 1]
    assert all(c.proposal.parent_spec_hash is None for c in first)
    assert later, "the budget should have run more than one generation"
    bred = [c for c in later if c.proposal.parent_spec_hash is not None]
    assert len(bred) > len(later) // 2, "most later proposals should have a parent"


def test_an_evaluator_that_raises_is_counted_not_fatal() -> None:
    """A spec that breaks the evaluator is a sample that came back empty.

    Excluding errors would let a searcher lower its own multiplicity by
    proposing specs that crash, so they are counted — and one bad spec must not
    cost the whole cycle.
    """

    def explode(spec: StrategySpec) -> BacktestResult:
        raise RuntimeError("the pipeline could not evaluate this")

    outcome = _searcher(
        evaluate=explode, budget=SearchBudget(n_trials=12, n_per_generation=6, seed=3)
    ).run()

    assert outcome.n_proposed == 12
    assert outcome.n_errored == 12
    assert outcome.n_evaluated == 0
    assert all(c.outcome is TrialOutcome.ERRORED for c in outcome.candidates)
    assert "could not evaluate" in outcome.candidates[0].error


def test_a_result_with_too_few_trades_cannot_become_a_parent() -> None:
    """A Sharpe over three trades is a statement about three trades.

    Breeding from it is how a search spends its entire budget refining noise it
    mistook for a signal in generation one. `fitness` is `None` rather than a
    low score, because a floor score still sorts above a genuinely bad result.
    """
    thin = _searcher(
        evaluate=lambda spec: _result(spec, sharpe=9.0, trades=MIN_TRADES_FOR_FITNESS - 1),
        budget=SearchBudget(n_trials=20, n_per_generation=10, seed=4),
    ).run()

    assert thin.n_evaluated == 20
    assert thin.n_measurable == 0
    assert thin.survivors == ()
    assert thin.best is None
    # With nothing to breed from, later generations fall back to random draws
    # rather than returning short.
    assert thin.n_proposed == 20


def test_survivors_are_diversified_by_feature_signature() -> None:
    """Otherwise generation three is twelve variants of one lookback.

    Ranking purely on training Sharpe collapses the population onto whichever
    idea got lucky first, and a collapsed population cannot recover because
    every child is a near-neighbour of the same parent.
    """
    searcher = _searcher(
        evaluate=lambda spec: _result(spec, sharpe=1.0, trades=30),
        budget=SearchBudget(n_trials=60, n_per_generation=20, n_survivors=4, seed=5),
    )
    outcome = searcher.run()

    signatures = [c.feature_signature for c in outcome.survivors]
    assert len(signatures) == len(set(signatures)), "two survivors read the same features"


def test_the_same_seed_runs_the_same_search() -> None:
    def evaluate(spec: StrategySpec) -> BacktestResult:
        # A deterministic pseudo-Sharpe from the spec's own hash, so ranking is
        # reproducible without being uniform.
        return _result(spec, sharpe=int(spec.spec_hash[:4], 16) / 65535, trades=30)

    first = _searcher(
        evaluate=evaluate, budget=SearchBudget(n_trials=30, n_per_generation=10, seed=6)
    ).run()
    again = _searcher(
        evaluate=evaluate, budget=SearchBudget(n_trials=30, n_per_generation=10, seed=6)
    ).run()

    assert [c.spec_hash for c in first.candidates] == [c.spec_hash for c in again.candidates]
    assert [c.spec_hash for c in first.survivors] == [c.spec_hash for c in again.survivors]


def test_the_search_reports_its_rejections_by_kind() -> None:
    """ "600 of 1,000 refused on cost" is a fact a searcher can act on.

    600 sentences are not, which is why the rejection carries a code as well as
    a reason.
    """
    # A window too short for the ladder's longer rungs, so some proposals are
    # refused for history and the rest are evaluated.
    outcome = _searcher(
        evaluate=lambda spec: _result(spec, sharpe=1.0, trades=30),
        budget=SearchBudget(n_trials=40, n_per_generation=20, seed=7),
        n_training_bars=45,
    ).run()

    assert outcome.n_rejected > 0, "a 45-bar window should refuse the longer lookbacks"
    assert "insufficient_history" in outcome.rejections_by_code
    assert sum(outcome.rejections_by_code.values()) == outcome.n_rejected
    assert "rejections by kind" in outcome.explain()


def test_required_sharpe_grows_with_the_trial_count() -> None:
    """**The arithmetic the whole milestone is shaped by.**

    The haircut is the best Sharpe a search of N would produce from noise, so it
    grows with N: an out-of-sample Sharpe of 3.0 is promotable out of a focused
    search and is not out of a thousand-spec sweep. Asserted at three points, so
    a change that flattened the curve fails here rather than by promoting
    something it should not.
    """
    ten = required_sharpe(n_trials=10, min_deflated_sharpe=0.5)
    hundred = required_sharpe(n_trials=100, min_deflated_sharpe=0.5)
    thousand = required_sharpe(n_trials=1000, min_deflated_sharpe=0.5)

    assert ten < hundred < thousand
    assert ten == pytest.approx(2.07, abs=0.05)
    assert thousand == pytest.approx(3.76, abs=0.05)
    # The consequence, stated as an assertion: a genuinely excellent daily
    # Sharpe does not survive a thousand-trial sweep.
    assert thousand > 3.0


def test_the_outcome_explains_itself() -> None:
    outcome = _searcher(
        evaluate=lambda spec: _result(spec, sharpe=1.5, trades=30),
        budget=SearchBudget(n_trials=10, n_per_generation=5, seed=8),
    ).run()
    text = outcome.explain()

    assert "10 proposed" in text
    assert "needs an out-of-sample Sharpe" in text
    assert outcome.best is not None


def test_a_budget_must_be_usable() -> None:
    from tb.research.searcher import SearchError

    with pytest.raises(SearchError):
        SearchBudget(n_trials=0)
    with pytest.raises(SearchError):
        SearchBudget(n_trials=10, n_per_generation=0)
    with pytest.raises(SearchError):
        SearchBudget(n_trials=10, n_survivors=0)


def test_a_candidate_carries_exactly_one_outcome() -> None:
    """A candidate with no outcome would be a trial the multiplicity count misses."""
    spec = a_spec()
    evaluated = Candidate(
        proposal=RandomProposer(bounds=BOUNDS).propose(n=1, rng=random.Random(0))[0],
        generation=1,
        outcome=TrialOutcome.EVALUATED,
        result=_result(spec, sharpe=1.0, trades=30),
    )
    assert evaluated.fitness == 1.0
    assert replace(evaluated, result=None).fitness is None
