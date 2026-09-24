"""Which strategies may trade, what each may deploy, and who owns a position.

This module is the join between M5 and M4. Everything M5 built — the registry,
the promotion gate, the ladder, the allocator — wrote its decisions into its own
tables, and M4's loop held one hardcoded strategy and read none of them. A
promotion therefore changed nothing about what the bot traded. `funded_book` is
what closes that: it reads `strategy_status`, `strategy_specs`, the rung and the
allocation, and hands back the strategies the loop will actually ask.

Four things about it are load-bearing.

**Each funded strategy carries its own pipeline, built from its own spec.** Not
a shared default pipeline that the specs are hoped to fit. A spec reading a
feature the pipeline does not compute evaluates to `UNKNOWN` at every decision
and is recorded as a strategy that simply never found an opportunity — a false
negative indistinguishable from a true one. `pipeline_from_spec` is the only
way a pipeline gets built here.

**A missing allocation falls back to the rung, and that is not the same as the
risk rule's fail-closed `None`.** `StrategyAllocationRule` blocks when nothing
was supplied, because for a promoted strategy an absent notional on the order
path is a wiring error. Here the absence has a different meaning: the gate has
promoted a strategy and the allocator has not run yet, and the plan's whole
premise is that such a strategy goes live at floor notional immediately. The
rung notional *is* a bound — floor times two per rung, intersected with the
per-position cap and the absolute ceiling — so falling back to it funds the
strategy at the size the ladder says, rather than at no size or at no bound.

**A strategy is dropped from the book only for reasons of identity, never for
reasons of size.** An allocation of zero stays in the book with a notional of
zero: entries are then refused by the rule, while exits still happen. Dropping
it would strand its open positions, and "a strategy whose allocation fell to
zero cannot get out" is the asymmetry this system is built to avoid. What does
drop a strategy is being unpromoted, having an exhausted lineage budget, or
having a spec that no longer parses — and its open positions are then unowned,
which the loop flattens.

**Ownership is resolved from the decision lineage, not from a position table.**
The broker reports a position per instrument and knows nothing about
strategies, so "whose position is this" has to be answered by looking at which
decision's entry opened it: `order_intents` joined to `decisions`, most recent
entry first, ordered by the event sequence that committed it. That join is the
reason the ledger records `decision_id` on every intent.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime
from decimal import Decimal

from tb.config.hard_limits import HardLimits
from tb.core.clock import from_iso, now_utc, to_iso
from tb.core.errors import TbError
from tb.engine.intents import IntentState
from tb.features.pipeline import FeaturePipeline
from tb.ledger.events import Actor, BookFundedPayload, EventType
from tb.ledger.store import Ledger
from tb.portfolio.allocator import allocation_as_of
from tb.registry.ladder import notional_for
from tb.registry.lineage import SpecRegistry
from tb.strategy.base import Strategy
from tb.strategy.dsl.ops import DslStrategy, pipeline_from_spec
from tb.strategy.dsl.schema import SpecError

# The intent states in which an entry may have opened the position it names.
# A positive list rather than a list of exclusions: `PENDING_SUBMIT` and
# `SUBMITTED` are the unknown states — an order may exist for each — so they
# own the position for attribution purposes, and the loop halts on an
# unresolved unknown anyway.
_OWNING_STATES: tuple[IntentState, ...] = (
    IntentState.PENDING_SUBMIT,
    IntentState.SUBMITTED,
    IntentState.ACKNOWLEDGED,
    IntentState.RESOLVED_FILLED,
)


class FundingError(TbError):
    """A book could not be assembled."""


@dataclass(frozen=True, slots=True)
class FundedStrategy:
    """One strategy the loop may ask, with the size it may deploy.

    `strategy_id` and `version` are read off the strategy rather than stored
    beside it, so the identity on the funding record and the identity on the
    decision the strategy produces cannot disagree. A mismatch there would make
    the ownership join attribute a position to a strategy that never traded it.
    """

    strategy: Strategy
    pipeline: FeaturePipeline
    notional_ccy: Decimal
    lineage_id: str = ""
    rung: int = 0
    detail: str = ""

    @property
    def strategy_id(self) -> str:
        return self.strategy.strategy_id

    @property
    def version(self) -> int:
        return self.strategy.version

    @property
    def key(self) -> tuple[str, int]:
        return (self.strategy_id, self.version)

    @property
    def label(self) -> str:
        return f"{self.strategy_id}@v{self.version}"

    def explain(self) -> str:
        return f"{self.label}: rung {self.rung}, {self.notional_ccy} per position ({self.detail})"


@dataclass(frozen=True, slots=True)
class Book:
    """The strategies a run will trade, and the ones it will not.

    The exclusions are carried rather than dropped. "Nothing is funded" and
    "four strategies are funded" are both answers an operator acts on, but so
    is "four were promoted and all four were excluded because their lineage is
    out of budget" — and that is invisible if the book is only a list of what
    survived.
    """

    funded: tuple[FundedStrategy, ...]
    excluded: tuple[tuple[str, str], ...] = ()
    equity_ccy: Decimal | None = None
    as_of: datetime | None = None
    source: str = "registry"

    def __bool__(self) -> bool:
        return bool(self.funded)

    def __len__(self) -> int:
        return len(self.funded)

    @property
    def keys(self) -> frozenset[tuple[str, int]]:
        return frozenset(strategy.key for strategy in self.funded)

    @property
    def labels(self) -> tuple[str, ...]:
        return tuple(strategy.label for strategy in self.funded)

    @property
    def by_key(self) -> Mapping[tuple[str, int], FundedStrategy]:
        return {strategy.key: strategy for strategy in self.funded}

    def explain(self) -> str:
        lines = [strategy.explain() for strategy in self.funded]
        lines.extend(f"{label}: excluded — {reason}" for label, reason in self.excluded)
        if not lines:
            lines.append("nothing is funded")
        return "\n".join(lines)


@dataclass(frozen=True, slots=True)
class Ownership:
    """Which strategy's entry opened the position currently held in a ticker."""

    strategy_id: str
    version: int
    intent_id: str
    decision_id: str
    opened_at: datetime | None = None

    @property
    def key(self) -> tuple[str, int]:
        return (self.strategy_id, self.version)

    @property
    def label(self) -> str:
        return f"{self.strategy_id}@v{self.version}"


def unfunded_notional(limits: HardLimits, *, equity_ccy: Decimal | None = None) -> Decimal:
    """The size for a strategy the registry does not fund.

    Used by `tb run --strategy trivial`, which is an explicit opt-in for
    drilling the loop rather than the funding path. Such a strategy has no rung,
    so the honest answer is that the ladder does not apply to it and it is
    bounded by the M4 caps alone — which is what the top of the ladder reduces
    to, since `notional_for` intersects the rung size with the per-position cap
    and the absolute ceiling and the tightest wins.

    Expressed through `notional_for` rather than by recomputing the per-position
    cap here, so there is still exactly one place that knows how a position size
    is bounded.
    """
    return notional_for(limits.promotion.ratchet_max_rung, limits=limits, equity_ccy=equity_ccy)


def explicit_book(
    strategy: Strategy,
    pipeline: FeaturePipeline,
    *,
    notional_ccy: Decimal,
    detail: str = "supplied explicitly, not funded by the registry",
) -> Book:
    """A one-strategy book for a hand-written strategy.

    `notional_ccy` is required and has no default. The risk rule blocks on a
    missing allocation, and a default here would be a way to reach the order
    path without any funding decision behind it — which is the hole the rule
    exists to close.
    """
    return Book(
        funded=(
            FundedStrategy(
                strategy=strategy,
                pipeline=pipeline,
                notional_ccy=notional_ccy,
                detail=detail,
            ),
        ),
        source="explicit",
    )


def funded_book(
    ledger: Ledger,
    *,
    limits: HardLimits,
    equity_ccy: Decimal | None = None,
    run_id: str | None = None,
    at: datetime | None = None,
) -> Book:
    """The promoted strategies, each with the notional it may deploy per position.

    Ordered by label rather than by promotion time. The loop resolves two
    strategies competing for one instrument by taking the first in book order,
    so that order has to be a function of identity alone: ordering by
    `promoted_at` would let a re-promotion change which strategy wins a
    contention, and a replay would resolve it differently from the live run.
    """
    moment = at or now_utc()
    registry = SpecRegistry(
        ledger,
        per_lineage_budget_ccy=limits.loss.per_lineage_budget_ccy,
        run_id=run_id,
    )
    allocations = {
        allocation.strategy_id: allocation for allocation in allocation_as_of(ledger, moment)
    }

    funded: list[FundedStrategy] = []
    excluded: list[tuple[str, str]] = []

    # The blocked ones are read alongside the promoted ones rather than left
    # out: exhausting a lineage budget *changes the status*, so a book built
    # from `promoted()` alone would show an empty list with no reason, and
    # "all of them are out of budget" would be indistinguishable from "nothing
    # was ever promoted".
    for record in [*registry.promoted(), *registry.blocked_after_promotion()]:
        may_trade, why = registry.may_trade(record.strategy_id, record.version)
        if not may_trade:
            # `may_trade` is asked even of a row whose status is PROMOTED,
            # because the lineage budget is the one reason a promoted strategy
            # may not trade without its status having changed yet. Trusting the
            # status column alone would be a budget a rename escapes.
            excluded.append((record.label, why))
            continue

        try:
            spec = registry.spec_of(record.strategy_id, record.version)
        except SpecError as exc:
            # The stored spec no longer parses against the current grammar. One
            # strategy is unfunded; the rest of the book still runs. Raising
            # here would let one stale row stop a whole account from trading,
            # and the fix for that row is a human's.
            excluded.append(
                (
                    record.label,
                    f"its stored spec no longer validates against this build's grammar: {exc}",
                )
            )
            continue
        if spec is None:
            excluded.append(
                (
                    record.label,
                    "promoted but its spec is not in strategy_specs, so there is nothing "
                    "to evaluate. The registry is inconsistent; `tb registry list` shows "
                    "what it holds.",
                )
            )
            continue

        rung_cap = notional_for(record.rung, limits=limits, equity_ccy=equity_ccy)
        allocation = allocations.get(record.strategy_id)
        if allocation is None:
            notional = rung_cap
            detail = (
                f"no allocation round covers {to_iso(moment)}, so the ladder's rung "
                f"{record.rung} notional stands: a strategy that has cleared the gate "
                "trades at its rung size rather than waiting for an allocator run"
            )
        else:
            notional = min(allocation.notional_ccy, rung_cap)
            detail = (
                f"allocation {allocation.notional_ccy} at weight {allocation.weight} "
                f"(shrinkage {allocation.shrinkage}), rung {record.rung} cap {rung_cap}"
            )

        funded.append(
            FundedStrategy(
                strategy=DslStrategy(
                    spec=spec,
                    strategy_id=record.strategy_id,
                    version=record.version,
                ),
                pipeline=pipeline_from_spec(spec),
                # Kept even at zero. An allocation of nothing refuses entries
                # through the risk rule and leaves exits alone; dropping the
                # strategy here would strand whatever it already holds.
                notional_ccy=notional,
                lineage_id=record.lineage_id,
                rung=record.rung,
                detail=detail,
            )
        )

    funded.sort(key=lambda strategy: strategy.label)
    return Book(
        funded=tuple(funded),
        excluded=tuple(excluded),
        equity_ccy=equity_ccy,
        as_of=moment,
    )


def owner_of(ledger: Ledger, *, t212_ticker: str) -> Ownership | None:
    """Which strategy's entry opened the position held in this ticker.

    `None` when no entry in the intent log can be attributed to a decision —
    a position the bot did not open through this path, or one that predates the
    lineage. The caller must treat that as unowned rather than as unowned-by-
    nobody-in-particular: an entry that cannot be attributed cannot be added to
    either, because the add would be sized against an allocation that did not
    pay for what is already held.

    Ordered by `committing_event_seq`, the system's total order. `wal_committed_at`
    ties whenever two intents are committed in the same instant, and an intent id
    is a hash, so neither is a usable tiebreaker — and getting the order wrong
    here attributes a position to the wrong strategy.

    The state filter is applied in Python rather than in the `WHERE` clause. The
    owning states are an enum, so building an `IN (?, ?, ?, ?)` list from it means
    generating SQL, and writing the placeholders out by hand means a literal count
    that can drift from the enum. Iterating instead costs one row per rejected
    entry on the same ticker and cannot go stale.
    """
    cursor = ledger.conn.execute(
        "SELECT d.strategy_id AS strategy_id, d.strategy_version AS version, "
        "       i.intent_id AS intent_id, i.decision_id AS decision_id, "
        "       i.state AS state, i.wal_committed_at AS committed_at "
        "FROM order_intents i JOIN decisions d ON d.decision_id = i.decision_id "
        "WHERE i.t212_ticker = ? AND i.purpose = 'entry' "
        "ORDER BY i.committing_event_seq DESC",
        (t212_ticker,),
    )
    for row in cursor:
        if IntentState(str(row["state"])) not in _OWNING_STATES:
            continue
        committed = row["committed_at"]
        return Ownership(
            strategy_id=str(row["strategy_id"]),
            version=int(row["version"]),
            intent_id=str(row["intent_id"]),
            decision_id=str(row["decision_id"]),
            opened_at=None if committed is None else from_iso(str(committed)),
        )
    return None


def record_book(ledger: Ledger, book: Book, *, run_id: str) -> None:
    """Write the book to the ledger, once per run.

    The event that answers "what was this run actually trading", which is not
    answerable from the decisions alone: a strategy that was funded and signalled
    nothing leaves no decision rows for the instruments it declined, and a
    strategy that was excluded leaves none at all. Recorded at startup rather
    than per cycle, because the book only changes when a promotion, a retirement
    or an allocation round does — each of which has its own event.
    """
    ledger.append(
        EventType.BOOK_FUNDED,
        run_id,
        BookFundedPayload(
            run_id=run_id,
            as_of_utc=to_iso(book.as_of or now_utc()),
            source=book.source,
            n_funded=len(book.funded),
            n_excluded=len(book.excluded),
            equity_ccy=None if book.equity_ccy is None else str(book.equity_ccy),
            entries=[
                {
                    "strategy_id": strategy.strategy_id,
                    "version": strategy.version,
                    "lineage_id": strategy.lineage_id,
                    "rung": strategy.rung,
                    "notional_ccy": str(strategy.notional_ccy),
                    "detail": strategy.detail,
                }
                for strategy in book.funded
            ],
            excluded=[{"label": label, "reason": reason} for label, reason in book.excluded],
        ),
        actor=Actor.SYSTEM,
        run_id=run_id,
    )


def unowned_positions(
    ledger: Ledger,
    *,
    book: Book,
    held: Sequence[str],
) -> list[tuple[str, Ownership | None, str]]:
    """Held tickers no funded strategy will ever close, with the reason.

    Two causes, kept apart because they are different facts: the owner is known
    and is no longer funded (retired, blocked, or its lineage is out of budget),
    or no owner can be resolved at all. Both leave a position that nothing in the
    book will decide about, which is the unmanaged exposure the loop flattens —
    but an operator reading the ledger needs to know which one happened.
    """
    funded = book.keys
    out: list[tuple[str, Ownership | None, str]] = []
    for ticker in sorted(held):
        owner = owner_of(ledger, t212_ticker=ticker)
        if owner is not None and owner.key in funded:
            continue
        if owner is None:
            reason = (
                "no entry in the intent log can be attributed to a decision for this "
                "ticker, so no strategy in the book will ever decide to close it"
            )
        else:
            reason = (
                f"opened by {owner.label}, which is not in the funded book "
                f"({', '.join(book.labels) or 'nothing is funded'})"
            )
        out.append((ticker, owner, reason))
    return out
