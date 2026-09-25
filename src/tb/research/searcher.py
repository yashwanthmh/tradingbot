"""The deterministic searcher: generations, a trial budget, and the haircut.

Random and genetic search over the spec space, with no API key and no ongoing
cost. It holds no ledger, no clock and no bars: it takes a proposer, a validator
and an `evaluate` callable, so the search logic is testable against a fake
evaluator and the I/O lives in `research.loop`.

**The trial budget is the design, not a safety valve.** `docs/decisions/
0002-search-size-and-provable-edge.md` records the measurement: the deflated
Sharpe subtracts the best result a search of N trials would produce from noise,
and that haircut is about 1.6 Sharpe at N=10 and 3.3 at N=1,000. So an
out-of-sample Sharpe of 3.0 — an excellent real result — is promotable out of a
focused search and is *not* out of a thousand-spec sweep. A searcher that
maximises coverage is therefore optimising against itself, and `required_sharpe`
puts the number the current budget implies in front of the caller rather than
leaving it to be discovered at the gate.

**Fitness is training Sharpe, and that is the thing that overfits.** It has to
be: the holdout is sealed, so training performance is all the searcher may see.
Everything that makes selecting on it safe lives downstream — the sealed holdout
spent once, the multiplicity haircut, PBO over the trial matrix — which is also
the real argument for a small budget. Two guards live here because they are
cheap and the alternative is breeding from nothing: a candidate needs
`MIN_TRADES_FOR_FITNESS` closed trades before it can be a parent, and survivors
are diversified by feature signature so a generation cannot collapse into twelve
variants of one lookback.

**An evaluator that raises is a counted outcome, not a crash.** A spec the
pipeline cannot evaluate is a sample that came back empty; excluding errors
would let a searcher lower its own multiplicity by proposing specs that break.
"""

from __future__ import annotations

import random
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from math import ceil

from tb.backtest.engine import BacktestResult
from tb.core.errors import TbError
from tb.registry.models import TrialOutcome
from tb.research.mutate import MutationProposer, Proposal, RandomProposer, SpecProposer
from tb.research.selection import expected_max_sharpe
from tb.research.trials import FALLBACK_SHARPE_DISPERSION
from tb.research.validate import Rejection, SpecValidator
from tb.strategy.dsl.schema import StrategySpec

# Closed trades a candidate needs before its Sharpe may steer the search.
#
# The same number as `portfolio.decay.MIN_TRADES_FOR_A_POSITIVE_VERDICT`, and
# for the same reason: a Sharpe over three trades is a statement about three
# trades. Breeding from it is how a search spends its whole budget refining
# noise it mistook for a signal in generation one.
MIN_TRADES_FOR_FITNESS = 10

# Survivors carried into the next generation. Small on purpose — a wide elite
# with a small budget means every parent gets one child, which is a random
# search wearing a genetic search's clothes.
DEFAULT_SURVIVORS = 4


class SearchError(TbError):
    """A search could not be run."""


@dataclass(frozen=True, slots=True)
class SearchBudget:
    """How many draws this search may take, and how they are spent.

    `n_trials` is the *total*, because that total is what the haircut is computed
    from. Expressing the budget as "generations x population" instead would let
    a caller add generations without noticing the multiplicity growing, which is
    the mistake ADR 0002 exists to prevent.
    """

    n_trials: int
    n_per_generation: int = 25
    n_survivors: int = DEFAULT_SURVIVORS
    seed: int = 0

    def __post_init__(self) -> None:
        if self.n_trials < 1:
            raise SearchError(f"a search needs at least one trial, got {self.n_trials}")
        if self.n_per_generation < 1:
            raise SearchError(
                f"a generation needs at least one proposal, got {self.n_per_generation}"
            )
        if self.n_survivors < 1:
            raise SearchError(f"at least one survivor must carry forward, got {self.n_survivors}")

    @property
    def n_generations(self) -> int:
        return max(1, ceil(self.n_trials / self.n_per_generation))


@dataclass(frozen=True, slots=True)
class Candidate:
    """One proposal and what became of it.

    Carries the rejection *or* the result, never both and never neither: a
    candidate with no outcome recorded would be a trial the multiplicity count
    misses.
    """

    proposal: Proposal
    generation: int
    outcome: TrialOutcome
    result: BacktestResult | None = None
    rejection: Rejection | None = None
    error: str = ""

    @property
    def spec(self) -> StrategySpec:
        return self.proposal.spec

    @property
    def spec_hash(self) -> str:
        return self.proposal.spec_hash

    @property
    def net_sharpe(self) -> float | None:
        return None if self.result is None else self.result.metrics.net_sharpe

    @property
    def n_trades(self) -> int:
        return 0 if self.result is None else self.result.metrics.n_trades

    @property
    def fitness(self) -> float | None:
        """What the search ranks on, or `None` when the candidate says nothing.

        `None` rather than a low score for a candidate with too few trades. A
        floor score would still sort above a genuinely bad result, so a
        generation in which nothing traded would breed from whichever spec
        happened to trade twice.
        """
        if self.result is None or self.n_trades < MIN_TRADES_FOR_FITNESS:
            return None
        return self.result.metrics.net_sharpe

    @property
    def feature_signature(self) -> tuple[str, ...]:
        """Which features this spec reads, as the diversity key.

        Two specs reading the same features at the same lookbacks are
        near-neighbours whatever their operators, and a survivor set full of them
        is one idea with different punctuation.
        """
        return self.spec.required_features


@dataclass(frozen=True, slots=True)
class SearchOutcome:
    """Everything one search produced, including what it refused."""

    search_id: str
    candidates: tuple[Candidate, ...]
    survivors: tuple[Candidate, ...]
    generations: int
    required_sharpe: float
    rejections_by_code: dict[str, int] = field(default_factory=dict)

    @property
    def n_proposed(self) -> int:
        return len(self.candidates)

    @property
    def n_evaluated(self) -> int:
        return sum(1 for c in self.candidates if c.outcome is TrialOutcome.EVALUATED)

    @property
    def n_rejected(self) -> int:
        return sum(1 for c in self.candidates if c.outcome is TrialOutcome.REJECTED)

    @property
    def n_errored(self) -> int:
        return sum(1 for c in self.candidates if c.outcome is TrialOutcome.ERRORED)

    @property
    def n_measurable(self) -> int:
        return sum(1 for c in self.candidates if c.fitness is not None)

    def errored_examples(self, limit: int = 3) -> tuple[tuple[str, str], ...]:
        """A few of the candidates the evaluator could not run, with why.

        Shown rather than only counted. An errored candidate is either a spec the
        grammar admits and the pipeline cannot evaluate — a bug in one of them —
        or a data problem in the vintage, and a count of twelve says neither.
        """
        return tuple(
            (candidate.spec_hash, candidate.error)
            for candidate in self.candidates
            if candidate.outcome is TrialOutcome.ERRORED
        )[:limit]

    @property
    def best(self) -> Candidate | None:
        ranked = [c for c in self.candidates if c.fitness is not None]
        if not ranked:
            return None
        return max(ranked, key=lambda c: (c.fitness or 0.0, c.spec_hash))

    def explain(self) -> str:
        best = self.best
        headline = (
            f"{self.search_id}: {self.n_proposed} proposed over {self.generations} "
            f"generation(s) — {self.n_evaluated} evaluated, {self.n_rejected} rejected, "
            f"{self.n_errored} errored, {self.n_measurable} produced a usable Sharpe"
        )
        lines = [headline]
        if self.rejections_by_code:
            counts = ", ".join(
                f"{code} {count}" for code, count in sorted(self.rejections_by_code.items())
            )
            lines.append(f"rejections by kind: {counts}")
        lines.append(
            f"a search of {self.n_proposed} trials needs an out-of-sample Sharpe of about "
            f"{self.required_sharpe:.2f} to clear the deflated-Sharpe gate; the best "
            + (
                f"training Sharpe here is {best.fitness:.2f} ({best.spec_hash[:10]})"
                if best is not None and best.fitness is not None
                else "candidate produced no usable Sharpe"
            )
        )
        return "\n".join(lines)


def required_sharpe(
    *,
    n_trials: int,
    min_deflated_sharpe: float,
    sharpe_dispersion: float = FALLBACK_SHARPE_DISPERSION,
) -> float:
    """The out-of-sample Sharpe a candidate from a search this size must show.

    The haircut plus the threshold. Reported to the caller because it is the
    single most decision-relevant number about a search *before* it runs: at
    1,000 trials it is above 3.7, which no honest daily strategy on free data
    will produce, so a sweep that size has already failed and the right response
    is a smaller budget rather than a looser gate.
    """
    return (
        expected_max_sharpe(n_trials=n_trials, sharpe_dispersion=sharpe_dispersion)
        + min_deflated_sharpe
    )


class Searcher:
    """Runs generations against an evaluator. No I/O of its own.

    `evaluate` takes a spec and returns a training backtest. It is injected
    rather than built here for two reasons: the searcher stays testable without
    a bar store, and — more importantly — the caller is the one that knows to
    hand it a `training_reader`. A searcher that built its own reader could build
    one that sees past the seal, and no amount of care in this module would
    catch it.
    """

    def __init__(
        self,
        *,
        evaluate: Callable[[StrategySpec], BacktestResult],
        validator: SpecValidator,
        random_proposer: RandomProposer,
        mutation_proposer: MutationProposer,
        budget: SearchBudget,
        min_deflated_sharpe: float,
        search_id: str,
        seed_parents: Sequence[StrategySpec] = (),
    ) -> None:
        self._evaluate = evaluate
        self._validator = validator
        self._random = random_proposer
        self._mutation = mutation_proposer
        self._budget = budget
        self._min_deflated = min_deflated_sharpe
        self._search_id = search_id
        # Specs from an earlier search for generation one to mutate, instead of
        # a random draw. This is what makes "many small searches" a way to
        # refine an idea rather than only to start new ones — and it does not
        # make refinement cheaper: a mutation stays in its parent's lineage, and
        # the lineage count accumulates across every search that touches it.
        self._seed_parents = tuple(seed_parents)

    def run(self) -> SearchOutcome:
        """Propose, validate, evaluate, select — `n_generations` times."""
        rng = random.Random(self._budget.seed)  # noqa: S311 - a spec search, not a key
        candidates: list[Candidate] = []
        seen: set[str] = set()
        survivors: list[Candidate] = []
        remaining = self._budget.n_trials
        generation = 0

        while remaining > 0:
            generation += 1
            batch = min(self._budget.n_per_generation, remaining)
            remaining -= batch
            if generation == 1:
                # Seeded: mutate the parents handed in. Unseeded: draw from the
                # grammar. The mutation proposer falls back to a random draw on
                # its own when it has no parents, so this is a choice of source
                # rather than two code paths.
                proposer: SpecProposer = self._mutation if self._seed_parents else self._random
                parents: tuple[StrategySpec, ...] = self._seed_parents
            else:
                proposer = self._mutation
                parents = tuple(candidate.spec for candidate in survivors)
            for proposal in proposer.propose(n=batch, rng=rng, parents=parents):
                candidate = self._consider(proposal, generation=generation, seen=seen)
                seen.add(candidate.spec_hash)
                candidates.append(candidate)
            survivors = self._select(candidates)

        rejections: dict[str, int] = {}
        for candidate in candidates:
            if candidate.rejection is not None:
                code = candidate.rejection.code
                rejections[code] = rejections.get(code, 0) + 1

        return SearchOutcome(
            search_id=self._search_id,
            candidates=tuple(candidates),
            survivors=tuple(survivors),
            generations=generation,
            required_sharpe=required_sharpe(
                n_trials=len(candidates), min_deflated_sharpe=self._min_deflated
            ),
            rejections_by_code=rejections,
        )

    # -- one proposal ------------------------------------------------------

    def _consider(self, proposal: Proposal, *, generation: int, seen: set[str]) -> Candidate:
        rejection = self._validator.check(proposal.spec, seen=seen)
        if rejection is not None:
            return Candidate(
                proposal=proposal,
                generation=generation,
                outcome=TrialOutcome.REJECTED,
                rejection=rejection,
            )
        try:
            result = self._evaluate(proposal.spec)
        # Deliberately broad. A spec that breaks the evaluator is a sample that
        # came back empty, not a reason to abandon the search — and narrowing
        # this to the exceptions seen so far would turn the next unfamiliar one
        # into a lost cycle.
        except Exception as exc:
            return Candidate(
                proposal=proposal,
                generation=generation,
                outcome=TrialOutcome.ERRORED,
                error=f"{type(exc).__name__}: {exc}",
            )
        return Candidate(
            proposal=proposal,
            generation=generation,
            outcome=TrialOutcome.EVALUATED,
            result=result,
        )

    # -- selection ---------------------------------------------------------

    def _select(self, candidates: Sequence[Candidate]) -> list[Candidate]:
        """The parents of the next generation: best first, one per feature signature.

        Ranked over *every* candidate so far rather than only the last
        generation, so a good idea from generation one is not lost because
        generation two happened to be poor. The tie-break is the spec hash,
        which makes the whole search reproducible from its seed — sorting on
        anything unstable would make a replay pick different parents.
        """
        ranked = sorted(
            (c for c in candidates if c.fitness is not None),
            key=lambda c: (-(c.fitness or 0.0), c.spec_hash),
        )
        chosen: list[Candidate] = []
        signatures: set[tuple[str, ...]] = set()
        for candidate in ranked:
            if candidate.feature_signature in signatures:
                continue
            signatures.add(candidate.feature_signature)
            chosen.append(candidate)
            if len(chosen) >= self._budget.n_survivors:
                break
        if not chosen:
            # Nothing measurable yet. An empty parent set makes the mutation
            # proposer fall back to a random draw, which is the right behaviour:
            # a generation with nothing to breed from is generation one again.
            return []
        return chosen
