"""What a registered strategy is, and what states it can be in.

Values rather than rows, so the state machine is testable without a database
and so a caller cannot construct an incoherent combination by assigning to a
field.

**Status is the answer to one question: may this strategy trade right now.**
Five members, and the two that look redundant are the important ones:

* `RETIRED` is a decision about *this strategy* — its statistics stopped
  holding, so it was killed. The lineage may continue.
* `BLOCKED` is a decision about *its lineage* — the lineage's lifetime loss
  budget is spent, so no member of it may trade, including a child registered
  after the exhaustion. A per-strategy retirement cannot express that, and a
  searcher that responds to a retirement by proposing a child would walk
  straight through it.

`CANDIDATE` and `AWAITING_HOLDOUT` are separated for the same kind of reason.
A candidate has been backtested; one awaiting the holdout has passed
everything the training window can tell us and is queued for the single
evaluation it is allowed. Conflating them would make "has this used up its one
shot" unanswerable, and that question is the whole point of a sealed holdout.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from decimal import Decimal
from enum import StrEnum

from tb.core.errors import TbError


class RegistryError(TbError):
    """A registry operation was refused."""


class AuthorKind(StrEnum):
    """Who proposed a spec.

    Recorded because it changes how the numbers should be read, not for
    credit. A deterministic searcher proposes candidates by the thousand and a
    person proposes a handful, so the multiplicity haircut on a lineage is
    dominated by whichever produced it — and a hand-written spec with a trial
    count of one is a genuinely different claim from a generated one with a
    trial count of one.
    """

    HUMAN = "human"
    SEARCH = "search"
    LLM = "llm"
    # A spec derived from another by mutation. Distinct from SEARCH because the
    # parent's trials are part of this candidate's multiplicity: a mutation of
    # a spec found after 900 trials inherits those 900.
    MUTATION = "mutation"


class StrategyStatus(StrEnum):
    """Whether a strategy may trade, and if not, why not."""

    CANDIDATE = "candidate"
    AWAITING_HOLDOUT = "awaiting_holdout"
    PROMOTED = "promoted"
    RETIRED = "retired"
    BLOCKED = "blocked"

    @property
    def may_trade(self) -> bool:
        """Only one status permits trading.

        Written as an equality rather than as "not retired and not blocked" on
        purpose: adding a sixth status later must default to *not* trading, and
        a negative test would have silently admitted it.
        """
        return self is StrategyStatus.PROMOTED

    @property
    def is_terminal(self) -> bool:
        """Whether this status can still change by ordinary operation.

        `BLOCKED` is terminal for the strategy even though the lineage's budget
        could in principle be raised — that is a human action against the hard
        limits, not something the bot can bring about.
        """
        return self in (StrategyStatus.RETIRED, StrategyStatus.BLOCKED)


class TrialOutcome(StrEnum):
    """How a trial ended.

    `REJECTED` and `ERRORED` are separate because they say different things
    about the searcher. A rejection is the system working: a candidate was
    evaluated and did not clear a gate. An error is the system failing to
    evaluate, and a search whose error count climbs is producing specs the
    pipeline cannot process — a bug report rather than a negative result.

    All of them count toward multiplicity. A candidate that errored was still
    a draw from the search space, and excluding it would understate how many
    times the space was sampled before the survivor was found.
    """

    EVALUATED = "evaluated"
    REJECTED = "rejected"
    ERRORED = "errored"
    PASSED_GATE = "passed_gate"

    @property
    def was_evaluated(self) -> bool:
        """Whether a backtest actually ran, so its metrics mean something."""
        return self in (TrialOutcome.EVALUATED, TrialOutcome.PASSED_GATE)


@dataclass(frozen=True, slots=True)
class RegisteredSpec:
    """A spec with an identity, a lineage and an ancestry.

    `spec_hash` is the deduplication key. Two searchers proposing the same tree
    under different names are one strategy and one trial, not two — counting
    them twice would inflate the trial count, and the trial count is divided
    *out* of the Sharpe, so double-counting is the conservative direction for
    the haircut but the wrong answer for "how many distinct things were tried".
    """

    strategy_id: str
    version: int
    lineage_id: str
    spec_hash: str
    author_kind: AuthorKind
    registered_at: datetime
    parent_strategy_id: str | None = None
    expected_edge_bps: Decimal | None = None
    name: str = ""
    generation: int = 0

    @property
    def label(self) -> str:
        """How this strategy is named in reports and log lines."""
        return f"{self.strategy_id}@v{self.version}"


@dataclass(frozen=True, slots=True)
class LineageBudget:
    """A lineage's lifetime loss budget and what is left of it.

    Consumption is a positive number of currency units lost. Storing it as a
    signed P&L would invite the reading where a profitable lineage accrues
    *negative* consumption and earns itself a larger budget than a human set,
    which is the one thing the control layer exists to prevent.
    """

    lineage_id: str
    budget_ccy: Decimal
    consumed_ccy: Decimal
    n_strategies: int
    opened_at: datetime
    exhausted_at: datetime | None = None

    @property
    def remaining_ccy(self) -> Decimal:
        return max(Decimal(0), self.budget_ccy - self.consumed_ccy)

    @property
    def is_exhausted(self) -> bool:
        return self.exhausted_at is not None or self.consumed_ccy >= self.budget_ccy

    @property
    def fraction_used(self) -> Decimal:
        if self.budget_ccy <= 0:
            return Decimal(1)
        return min(Decimal(1), self.consumed_ccy / self.budget_ccy)

    def summary(self) -> str:
        state = "exhausted" if self.is_exhausted else "open"
        return (
            f"{self.lineage_id}: {state}, {self.consumed_ccy} of {self.budget_ccy} "
            f"consumed across {self.n_strategies} strategy/strategies"
        )


@dataclass(frozen=True, slots=True)
class StrategyRecord:
    """The live state of one registered strategy.

    Carries the realised record alongside the status because every consumer
    needs both: the allocator shrinks its prior by the trade count, the review
    cycle reads the P&L, and the ladder reads both. Splitting them would mean
    two queries that can disagree about which cycle they describe.
    """

    strategy_id: str
    version: int
    lineage_id: str
    status: StrategyStatus
    rung: int
    updated_at: datetime
    rung_changed_at: datetime | None = None
    promoted_at: datetime | None = None
    retired_at: datetime | None = None
    retire_reason: str = ""
    realised_pnl_ccy: Decimal = Decimal(0)
    n_realised_trades: int = 0

    @property
    def label(self) -> str:
        return f"{self.strategy_id}@v{self.version}"

    @property
    def may_trade(self) -> bool:
        return self.status.may_trade
