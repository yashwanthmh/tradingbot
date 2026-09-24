"""Capital allocation: priors shrunk toward realised edge, with the weight shown.

The problem this solves is small samples. At floor size with multi-day holds a
strategy produces 10-20 trades a month, so its realised edge after a month is
one of the noisiest numbers in the system. Allocating on it would chase noise;
allocating on the backtest prior alone would ignore every fact the live account
has produced. So the allocator blends, and the blend weight is **recorded**:

    blended = (1 - w) * prior + w * realised,   w = n / (n + SHRINKAGE_PRIOR_TRADES)

`w` is stored beside the result rather than folded into it. A weight of 0.25
says the number is mostly the backtest's opinion, and a row holding only the
blended figure could not say that — it would present a prior as a measurement.

**Why the prior is the deflated edge, not the declared one.** A spec declares
its edge and the cost gate divides by that declaration, so allocating on it
would let a strategy claim its way to a larger allocation. The prior is the
edge the *evidence* supports: the declared edge scaled by how much of its
backtest Sharpe survived the multiplicity haircut.

**Order of operations matters, and it is: blend, weight, cap, size.** The
family cap is applied to the weights and not to the final notionals, because
capping notionals after sizing would let the ladder's rungs change which
family binds — a strategy climbing a rung would push its family over the cap
and silently shrink its neighbours.

**The allocator never enlarges anything.** It distributes a deployed total that
the hard limits bound, and every per-strategy result is intersected with that
strategy's rung notional. It can move capital between strategies and can hand
out less than the total; it cannot hand out more.
"""

from __future__ import annotations

import sqlite3
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime
from decimal import Decimal, localcontext

from tb.config.hard_limits import HardLimits
from tb.core.clock import now_utc, to_iso
from tb.core.errors import TbError
from tb.core.ids import new_id
from tb.ledger.events import Actor, AllocationPayload, EventType
from tb.ledger.store import Ledger
from tb.portfolio.correlation import Holding, apply_family_cap
from tb.registry.ladder import notional_for

# The trade count at which realised evidence is worth as much as the prior.
#
# 30, the same number as `promotion.min_oos_trades`, and for the same reason:
# the gate refuses to believe a backtest on fewer than 30 out-of-sample trades,
# so it would be incoherent for the allocator to believe a *live* record on
# fewer. At 10 trades the weight is 0.25 — a quarter measurement, three
# quarters prior — which is about right for a month of trading.
SHRINKAGE_PRIOR_TRADES = 30

_WORKING_PRECISION = 28


class AllocationError(TbError):
    """An allocation could not be computed."""


@dataclass(frozen=True, slots=True)
class StrategyInput:
    """One promoted strategy, and what is known about it.

    `prior_edge_bps` is supplied rather than derived here: it comes from the
    promotion record, which is where the deflation that produced it lives. An
    allocator recomputing it would be a second opinion about a number the
    ledger already holds.
    """

    strategy_id: str
    version: int
    lineage_id: str
    rung: int
    prior_edge_bps: Decimal
    realised_edge_bps: Decimal | None = None
    n_realised_trades: int = 0

    @property
    def label(self) -> str:
        return f"{self.strategy_id}@v{self.version}"


@dataclass(frozen=True, slots=True)
class Allocation:
    """What one strategy got, and the whole derivation behind it."""

    strategy_id: str
    version: int
    lineage_id: str
    prior_edge_bps: Decimal
    realised_edge_bps: Decimal | None
    shrinkage: Decimal
    blended_edge_bps: Decimal
    raw_weight: Decimal
    weight: Decimal
    rung: int
    notional_ccy: Decimal
    n_realised_trades: int
    correlation_capped: bool = False
    detail: str = ""

    @property
    def label(self) -> str:
        return f"{self.strategy_id}@v{self.version}"

    @property
    def evidence_is_mostly_prior(self) -> bool:
        """Whether this allocation rests mainly on the backtest.

        The question an operator actually asks of an allocation table, and the
        reason the shrinkage weight is a stored column rather than an
        intermediate.
        """
        return self.shrinkage < Decimal("0.5")

    def explain(self) -> str:
        realised = "none" if self.realised_edge_bps is None else f"{self.realised_edge_bps}bps"
        return (
            f"{self.label}: prior {self.prior_edge_bps}bps, realised {realised} over "
            f"{self.n_realised_trades} trade(s), shrinkage {self.shrinkage} -> blended "
            f"{self.blended_edge_bps}bps; weight {self.weight} at rung {self.rung} = "
            f"{self.notional_ccy}" + (" (family cap applied)" if self.correlation_capped else "")
        )


@dataclass(frozen=True, slots=True)
class AllocationRound:
    """One allocation decision across the whole portfolio."""

    allocation_id: str
    as_of_utc: datetime
    allocations: tuple[Allocation, ...]
    total_notional_ccy: Decimal
    deployable_ccy: Decimal
    n_correlation_capped: int = 0
    detail: str = ""

    @property
    def by_strategy(self) -> Mapping[str, Allocation]:
        return {allocation.strategy_id: allocation for allocation in self.allocations}

    def explain(self) -> str:
        lines = [allocation.explain() for allocation in self.allocations]
        lines.append(
            f"total {self.total_notional_ccy} of {self.deployable_ccy} deployable; "
            f"{self.n_correlation_capped} family/families capped"
        )
        if self.detail:
            lines.append(self.detail)
        return "\n".join(lines)


def shrinkage_weight(n_trades: int, *, prior_trades: int = SHRINKAGE_PRIOR_TRADES) -> Decimal:
    """How much of the blend the realised record deserves.

    `n / (n + prior)`. Zero trades gives zero weight — the prior alone, which
    is correct for a strategy promoted five minutes ago — and the weight
    approaches but never reaches 1, so the prior never disappears entirely.
    That tail matters: a strategy with 200 good trades still had a backtest,
    and discarding it would make the allocator forget why the strategy was
    funded.
    """
    if n_trades < 0:
        raise AllocationError(f"trade count must not be negative, got {n_trades}")
    with localcontext() as ctx:
        ctx.prec = _WORKING_PRECISION
        return Decimal(n_trades) / Decimal(n_trades + prior_trades)


def blend(
    *,
    prior_edge_bps: Decimal,
    realised_edge_bps: Decimal | None,
    n_trades: int,
    prior_trades: int = SHRINKAGE_PRIOR_TRADES,
) -> tuple[Decimal, Decimal]:
    """Blend a prior with a realised edge. Returns `(shrinkage, blended)`.

    A `None` realised edge means no admissible realised evidence exists — every
    fill so far had an inferred price, or there have been no fills. That is not
    a realised edge of zero, so the weight is forced to zero and the prior
    stands alone. Treating it as zero would penalise a strategy for the
    reconciler's inability to price its fills.
    """
    if realised_edge_bps is None:
        return Decimal(0), prior_edge_bps
    weight = shrinkage_weight(n_trades, prior_trades=prior_trades)
    with localcontext() as ctx:
        ctx.prec = _WORKING_PRECISION
        blended = (Decimal(1) - weight) * prior_edge_bps + weight * realised_edge_bps
    return weight, blended


class Allocator:
    """Turns promoted strategies into per-strategy notionals."""

    def __init__(
        self,
        ledger: Ledger,
        *,
        limits: HardLimits,
        run_id: str | None = None,
        prior_trades: int = SHRINKAGE_PRIOR_TRADES,
    ) -> None:
        self._ledger = ledger
        self._limits = limits
        self._run_id = run_id
        self._prior_trades = prior_trades

    def allocate(
        self,
        *,
        strategies: Sequence[StrategyInput],
        equity_ccy: Decimal,
        holdings: Sequence[Holding] = (),
        at: datetime | None = None,
        record: bool = True,
    ) -> AllocationRound:
        """Allocate deployable capital across promoted strategies."""
        moment = at or now_utc()
        capital = self._limits.capital
        with localcontext() as ctx:
            ctx.prec = _WORKING_PRECISION
            deployable = min(
                equity_ccy * Decimal(str(capital.max_deployed_pct)) / Decimal(100),
                capital.absolute_ceiling_ccy,
            )

        if not strategies:
            return self._empty(moment, deployable)

        blended: dict[str, tuple[Decimal, Decimal]] = {}
        for strategy in strategies:
            blended[strategy.strategy_id] = blend(
                prior_edge_bps=strategy.prior_edge_bps,
                realised_edge_bps=strategy.realised_edge_bps,
                n_trades=strategy.n_realised_trades,
                prior_trades=self._prior_trades,
            )

        raw = _weights({s.strategy_id: blended[s.strategy_id][1] for s in strategies})
        outcome = apply_family_cap(
            weights=raw,
            holdings=holdings,
            max_family_fraction=Decimal(str(capital.max_family_deployed_pct))
            / Decimal(str(capital.max_deployed_pct)),
        )

        allocations: list[Allocation] = []
        for strategy in strategies:
            shrink, blend_bps = blended[strategy.strategy_id]
            weight = outcome.weights[strategy.strategy_id]
            rung_cap = notional_for(strategy.rung, limits=self._limits, equity_ccy=equity_ccy)
            with localcontext() as ctx:
                ctx.prec = _WORKING_PRECISION
                share = (deployable * weight).quantize(Decimal("0.01"))
            allocations.append(
                Allocation(
                    strategy_id=strategy.strategy_id,
                    version=strategy.version,
                    lineage_id=strategy.lineage_id,
                    prior_edge_bps=strategy.prior_edge_bps,
                    realised_edge_bps=strategy.realised_edge_bps,
                    shrinkage=shrink,
                    blended_edge_bps=blend_bps,
                    raw_weight=raw[strategy.strategy_id],
                    weight=weight,
                    rung=strategy.rung,
                    # The rung is a ceiling, not a target. A strategy at rung 0
                    # does not get a larger position because the portfolio has
                    # room — that is what the ratchet is for.
                    notional_ccy=min(share, rung_cap),
                    n_realised_trades=strategy.n_realised_trades,
                    correlation_capped=weight != raw[strategy.strategy_id],
                )
            )

        total = sum((a.notional_ccy for a in allocations), Decimal(0))
        result = AllocationRound(
            allocation_id=new_id("alloc", length=12),
            as_of_utc=moment,
            allocations=tuple(allocations),
            total_notional_ccy=total,
            deployable_ccy=deployable,
            n_correlation_capped=outcome.n_capped,
            detail=outcome.detail,
        )
        if record:
            self._record(result)
        return result

    def _empty(self, moment: datetime, deployable: Decimal) -> AllocationRound:
        return AllocationRound(
            allocation_id=new_id("alloc", length=12),
            as_of_utc=moment,
            allocations=(),
            total_notional_ccy=Decimal(0),
            deployable_ccy=deployable,
            detail="no promoted strategies to allocate to",
        )

    def _record(self, result: AllocationRound) -> None:
        with self._ledger.transaction() as tx:
            event = tx.append(
                EventType.ALLOCATION_DECIDED,
                result.allocation_id,
                AllocationPayload(
                    allocation_id=result.allocation_id,
                    as_of_utc=to_iso(result.as_of_utc),
                    run_id=self._run_id,
                    n_strategies=len(result.allocations),
                    total_notional_ccy=str(result.total_notional_ccy),
                    entries=[
                        {
                            "strategy_id": allocation.strategy_id,
                            "version": allocation.version,
                            "shrinkage": str(allocation.shrinkage),
                            "blended_edge_bps": str(allocation.blended_edge_bps),
                            "weight": str(allocation.weight),
                            "notional_ccy": str(allocation.notional_ccy),
                            "rung": allocation.rung,
                            "correlation_capped": allocation.correlation_capped,
                        }
                        for allocation in result.allocations
                    ],
                    n_correlation_capped=result.n_correlation_capped,
                    detail=result.detail,
                ),
                actor=Actor.SYSTEM,
                run_id=self._run_id,
            )
            for allocation in result.allocations:
                tx.execute(
                    """
                    INSERT INTO allocations (
                        allocation_id, run_id, as_of_utc, strategy_id, version, lineage_id,
                        family, prior_edge_bps, realised_edge_bps, shrinkage,
                        blended_edge_bps, raw_weight, weight, rung, notional_ccy,
                        n_realised_trades, correlation_cap_applied, detail,
                        allocating_event_seq
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        f"{result.allocation_id}:{allocation.strategy_id}",
                        self._run_id,
                        to_iso(result.as_of_utc),
                        allocation.strategy_id,
                        allocation.version,
                        allocation.lineage_id,
                        None,
                        str(allocation.prior_edge_bps),
                        (
                            None
                            if allocation.realised_edge_bps is None
                            else str(allocation.realised_edge_bps)
                        ),
                        str(allocation.shrinkage),
                        str(allocation.blended_edge_bps),
                        str(allocation.raw_weight),
                        str(allocation.weight),
                        allocation.rung,
                        str(allocation.notional_ccy),
                        allocation.n_realised_trades,
                        1 if allocation.correlation_capped else 0,
                        allocation.detail,
                        event.seq,
                    ),
                )

    # -- reading -----------------------------------------------------------

    def latest(self) -> list[Allocation]:
        """The most recent round's allocations, for `tb allocator explain`.

        Keyed on the recording event's sequence rather than on `as_of_utc`. All
        the rows of one round share that sequence, and two rounds can share a
        timestamp — a caller passing an explicit `at`, or simply two rounds in
        the same second — in which case a timestamp query would splice them
        into one round that never happened.
        """
        row: sqlite3.Row | None = self._ledger.conn.execute(
            "SELECT allocating_event_seq FROM allocations "
            "ORDER BY allocating_event_seq DESC LIMIT 1"
        ).fetchone()
        if row is None:
            return []
        return self._round(int(row["allocating_event_seq"]))

    def history_for(self, strategy_id: str, *, limit: int = 20) -> list[Allocation]:
        rows = self._ledger.conn.execute(
            "SELECT * FROM allocations WHERE strategy_id = ? "
            "ORDER BY allocating_event_seq DESC LIMIT ?",
            (strategy_id, limit),
        ).fetchall()
        return [_row_to_allocation(row) for row in rows]

    def _round(self, event_seq: int) -> list[Allocation]:
        rows = self._ledger.conn.execute(
            "SELECT * FROM allocations WHERE allocating_event_seq = ? ORDER BY strategy_id",
            (event_seq,),
        ).fetchall()
        return [_row_to_allocation(entry) for entry in rows]


def _weights(edges: Mapping[str, Decimal]) -> dict[str, Decimal]:
    """Normalise blended edges into weights summing to 1.

    A non-positive blended edge gets zero weight rather than a negative one:
    this account is long-only and unlevered, so there is no way to express "I
    believe this strategy loses money" other than by not funding it.

    When every edge is non-positive the result is all zeros and nothing is
    deployed. That is the correct outcome and deliberately not an equal split —
    a portfolio where the allocator believes nothing should hold cash, not
    spread itself evenly over things it disbelieves.
    """
    positive = {key: value for key, value in edges.items() if value > 0}
    total = sum(positive.values(), Decimal(0))
    if total <= 0:
        return dict.fromkeys(edges, Decimal(0))
    with localcontext() as ctx:
        ctx.prec = _WORKING_PRECISION
        return {key: (positive.get(key, Decimal(0)) / total) for key in edges}


def _row_to_allocation(row: sqlite3.Row) -> Allocation:
    realised = row["realised_edge_bps"]
    return Allocation(
        strategy_id=str(row["strategy_id"]),
        version=int(row["version"]),
        lineage_id=str(row["lineage_id"]),
        prior_edge_bps=Decimal(str(row["prior_edge_bps"])),
        realised_edge_bps=None if realised is None else Decimal(str(realised)),
        shrinkage=Decimal(str(row["shrinkage"])),
        blended_edge_bps=Decimal(str(row["blended_edge_bps"])),
        raw_weight=Decimal(str(row["raw_weight"])),
        weight=Decimal(str(row["weight"])),
        rung=int(row["rung"]),
        notional_ccy=Decimal(str(row["notional_ccy"])),
        n_realised_trades=int(row["n_realised_trades"]),
        correlation_capped=bool(row["correlation_cap_applied"]),
        detail="" if row["detail"] is None else str(row["detail"]),
    )


def allocation_as_of(ledger: Ledger, moment: datetime) -> list[Allocation]:
    """The allocations in force at an instant.

    Point-in-time rather than "the latest", because the question a replay asks
    is what the allocator believed *then* — and answering it with today's table
    would explain a months-old position with a weight it never had.
    """
    row: sqlite3.Row | None = ledger.conn.execute(
        "SELECT allocating_event_seq FROM allocations WHERE as_of_utc <= ? "
        "ORDER BY as_of_utc DESC, allocating_event_seq DESC LIMIT 1",
        (to_iso(moment),),
    ).fetchone()
    if row is None:
        return []
    # Selected by the recording event's sequence, not by the timestamp: all the
    # rows of one round share the sequence, and two rounds sharing a timestamp
    # would otherwise be spliced into one round that never happened.
    rows = ledger.conn.execute(
        "SELECT * FROM allocations WHERE allocating_event_seq = ? ORDER BY strategy_id",
        (int(row["allocating_event_seq"]),),
    ).fetchall()
    return [_row_to_allocation(entry) for entry in rows]
