"""The per-session portfolio pass: review, then rungs, then allocations.

The review, the size ladder and the allocator were built in M5 and never run:
nothing in the live path called the ladder or the allocator, so every promoted
strategy traded at the floor rung for ever, no allocation round was ever
recorded, and a verdict existed only when someone typed `tb review`. Capital
could not move — the step the plan's success condition ends on.

Once per session this pass, over every promoted strategy:

1. **Reviews it** on its realised record (`tb.portfolio.decay`), recording the
   KEEP / KILL / ITERATE / SCALE verdict. Retiring on KILL or ITERATE stays a
   human's `tb review --apply`: retirement is irreversible and flattens what
   the strategy holds, and the loop should not do that to itself on a
   judgement it made alone. The verdict still moves capital, through 2 and 3.
2. **Moves its rung** (`tb.registry.ladder`) on the evidence of its current
   rung — sessions at the rung, trades closed at it, what they realised. A KILL
   or ITERATE verdict is a breach and drops two rungs at once; a rung is climbed
   only on the ladder's own slow terms, so a SCALE verdict is not a second way
   up.
3. **Allocates** (`tb.portfolio.allocator`) deployable capital across them,
   each prior shrunk toward its realised edge. A strategy whose blended edge has
   gone to nothing is allocated nothing, which refuses its entries and leaves
   its exits alone.

The pass is recorded as `session.reviewed` under the session's date, and the
record is what makes it once a session: a `tb run` restarted every minute by a
scheduler reviews the portfolio once a day, not once a minute — which matters,
since a breach applied twice drops four rungs.

**The prior** is the declared edge scaled by the promotion's deflated Sharpe
probability — the confidence, after the multiplicity haircut, that the edge is
real. A promoted strategy with no promotion decision behind it has no prior and
is allocated nothing: `PROMOTED` without the gate's record is an inconsistency
to refuse, not a strategy to fund on its own say-so.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from datetime import date, datetime, timedelta
from decimal import Decimal

from tb.config.hard_limits import HardLimits
from tb.data.calendar import TradingCalendar
from tb.ledger.events import Actor, EventType, SessionReviewedPayload
from tb.ledger.store import Ledger
from tb.portfolio.allocator import AllocationRound, Allocator, StrategyInput
from tb.portfolio.correlation import Holding
from tb.portfolio.decay import Review, ReviewCycle, ReviewInput
from tb.registry.ladder import BreachReason, LadderMove, RungEvidence, SizeLadder
from tb.registry.lineage import SpecRegistry
from tb.registry.models import StrategyRecord
from tb.strategy.dsl.schema import SpecError


@dataclass(frozen=True, slots=True)
class SessionPass:
    """What one session's pass did. `skipped` when the session was already done."""

    session: date
    reviews: tuple[Review, ...] = ()
    moves: tuple[LadderMove, ...] = ()
    allocation: AllocationRound | None = None
    skipped: bool = False
    detail: str = ""


@dataclass(frozen=True, slots=True)
class _Record:
    """One strategy's realised record, from its charged round trips."""

    n_trades: int
    pnl_ccy: Decimal
    edge_bps: Decimal | None
    n_at_rung: int
    pnl_at_rung: Decimal


def session_reviewed(ledger: Ledger, session: date) -> bool:
    """Whether this session's pass has already run."""
    row = ledger.conn.execute(
        "SELECT 1 FROM event_log WHERE event_type = ? AND aggregate_id = ? LIMIT 1",
        (EventType.SESSION_REVIEWED.value, session.isoformat()),
    ).fetchone()
    return row is not None


def run_session_pass(
    ledger: Ledger,
    *,
    limits: HardLimits,
    equity_ccy: Decimal | None,
    at: datetime,
    holdings: Sequence[Holding] = (),
    run_id: str | None = None,
    calendar: TradingCalendar | None = None,
) -> SessionPass:
    """Review, re-rung and re-allocate every promoted strategy, once a session."""
    days = calendar or TradingCalendar()
    session = days.day_of(at).day
    if session_reviewed(ledger, session):
        return SessionPass(session=session, skipped=True, detail="already reviewed")

    registry = SpecRegistry(
        ledger, per_lineage_budget_ccy=limits.loss.per_lineage_budget_ccy, run_id=run_id
    )
    ladder = SizeLadder(ledger, limits=limits, run_id=run_id)
    promoted = registry.promoted()

    records = {record.label: _realised(ledger, record) for record in promoted}
    candidates = [_review_input(registry, record, records[record.label]) for record in promoted]
    reviews = ReviewCycle(ledger, run_id=run_id).run(candidates, at=at)

    moves: list[LadderMove] = []
    inputs: list[StrategyInput] = []
    for record, verdict in zip(promoted, reviews, strict=True):
        realised = records[record.label]
        breached = verdict.verdict.retires_the_strategy
        at_rung_since = record.rung_changed_at or record.promoted_at
        move = ladder.review(
            record.strategy_id,
            version=record.version,
            evidence=RungEvidence(
                days_at_rung=_sessions_since(days, at_rung_since, at),
                n_trades_at_rung=realised.n_at_rung,
                realised_pnl_at_rung=realised.pnl_at_rung,
                breached=breached,
                breach_reason=BreachReason.DECAY if breached else None,
            ),
            equity_ccy=equity_ccy,
            at=at,
        )
        if move is not None:
            moves.append(move)
        inputs.append(
            StrategyInput(
                strategy_id=record.strategy_id,
                version=record.version,
                lineage_id=record.lineage_id,
                rung=record.rung if move is None else move.to_rung,
                prior_edge_bps=_prior_edge(ledger, registry, record),
                realised_edge_bps=realised.edge_bps,
                n_realised_trades=realised.n_trades,
            )
        )

    allocation: AllocationRound | None = None
    detail = ""
    if not inputs:
        detail = "nothing is promoted"
    elif equity_ccy is None or equity_ccy <= 0:
        detail = "no equity observation, so nothing to allocate; rungs alone size the book"
    else:
        allocation = Allocator(ledger, limits=limits, run_id=run_id).allocate(
            strategies=inputs, equity_ccy=equity_ccy, holdings=holdings, at=at
        )

    ledger.append(
        EventType.SESSION_REVIEWED,
        session.isoformat(),
        SessionReviewedPayload(
            session_date=session.isoformat(),
            n_strategies=len(promoted),
            run_id=run_id,
            verdicts={result.label: result.verdict.value for result in reviews},
            rung_moves={
                f"{move.strategy_id}@v{move.version}": f"{move.from_rung}->{move.to_rung}"
                for move in moves
            },
            allocation_id=None if allocation is None else allocation.allocation_id,
            detail=detail,
        ),
        actor=Actor.SYSTEM,
        run_id=run_id,
    )
    return SessionPass(
        session=session,
        reviews=tuple(reviews),
        moves=tuple(moves),
        allocation=allocation,
        detail=detail,
    )


def _realised(ledger: Ledger, record: StrategyRecord) -> _Record:
    """The strategy's charged round trips, in total and since its rung changed.

    The edge is realised P&L over the capital the trips deployed — quantity at
    cost basis — in basis points: a return per unit risked, comparable across
    strategies of different sizes, where P&L alone is not.
    """
    since = record.rung_changed_at or record.promoted_at
    n_trades = 0
    pnl = Decimal(0)
    deployed = Decimal(0)
    n_at_rung = 0
    pnl_at_rung = Decimal(0)
    for row in ledger.conn.execute(
        "SELECT quantity, cost_basis, pnl_ccy, closed_at FROM round_trips"
        " WHERE strategy_id = ? AND strategy_version = ? AND charged = 1"
        " ORDER BY recording_event_seq",
        (record.strategy_id, record.version),
    ):
        trip_pnl = Decimal(str(row["pnl_ccy"]))
        n_trades += 1
        pnl += trip_pnl
        if row["cost_basis"] is not None:
            deployed += Decimal(str(row["quantity"])) * Decimal(str(row["cost_basis"]))
        closed = None if row["closed_at"] is None else datetime.fromisoformat(row["closed_at"])
        if since is None or (closed is not None and closed >= since):
            n_at_rung += 1
            pnl_at_rung += trip_pnl
    edge = None if n_trades == 0 or deployed <= 0 else pnl / deployed * Decimal(10_000)
    return _Record(n_trades, pnl, edge, n_at_rung, pnl_at_rung)


def _review_input(registry: SpecRegistry, record: StrategyRecord, realised: _Record) -> ReviewInput:
    budget = registry.budget_for(record.lineage_id)
    return ReviewInput(
        strategy_id=record.strategy_id,
        version=record.version,
        lineage_id=record.lineage_id,
        n_realised_trades=realised.n_trades,
        realised_pnl_ccy=realised.pnl_ccy,
        declared_edge_bps=_declared_edge(registry, record),
        realised_edge_bps=realised.edge_bps,
        lineage_budget_ccy=None if budget is None else budget.budget_ccy,
        lineage_consumed_ccy=None if budget is None else budget.consumed_ccy,
    )


def _declared_edge(registry: SpecRegistry, record: StrategyRecord) -> Decimal:
    try:
        spec = registry.spec_of(record.strategy_id, record.version)
    except SpecError:
        return Decimal(0)
    if spec is None or spec.expected_edge_bps is None:
        return Decimal(0)
    return spec.expected_edge_bps


def _prior_edge(ledger: Ledger, registry: SpecRegistry, record: StrategyRecord) -> Decimal:
    """The declared edge, scaled by the gate's confidence that it is real."""
    row = ledger.conn.execute(
        "SELECT deflated_sharpe_probability FROM promotions"
        " WHERE strategy_id = ? AND version = ? AND decision = 'promote'"
        " ORDER BY deciding_event_seq DESC LIMIT 1",
        (record.strategy_id, record.version),
    ).fetchone()
    if row is None or row["deflated_sharpe_probability"] is None:
        return Decimal(0)
    confidence = Decimal(str(row["deflated_sharpe_probability"]))
    return _declared_edge(registry, record) * confidence


def _sessions_since(days: TradingCalendar, since: datetime | None, at: datetime) -> int:
    """Trading sessions completed at a rung: after the one it changed in, up to now."""
    if since is None:
        return 0
    first = days.day_of(since).day + timedelta(days=1)
    today = days.day_of(at).day
    if first > today:
        return 0
    return len(days.sessions_between(first, today))
