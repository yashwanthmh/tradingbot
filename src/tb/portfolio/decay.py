"""KEEP / KILL / ITERATE / SCALE, and the asymmetry that makes them honest.

The review cycle's problem is that it has almost no data. At floor size with
multi-day holds a strategy produces 10-20 trades a month, so after a month its
realised Sharpe has an error bar wider than any edge it could plausibly have.
A review that treated that number as a measurement would be noise-chasing with
real capital.

So the four verdicts are held to deliberately different standards:

**KILL is cheap.** A handful of trades is enough, because the cost of killing a
good strategy is a missed opportunity and the cost of keeping a bad one is
money. A strategy can be re-registered as a mutation and re-earn its place; the
lineage carries its trial count into the next haircut, which is the price.

**SCALE is expensive.** It needs the trade count the promotion gate demanded of
a backtest — it would be incoherent to disbelieve a backtest on 29 trades and
then believe a live record on 8 — plus a realised edge that is actually holding
up against what was declared.

**ITERATE is the honest middle.** Not working, not clearly broken: the evidence
says stop funding this shape and try a mutation. It differs from KILL in what
the searcher should do next, which is the only reason it exists as a separate
verdict.

**KEEP is the default, and it is not an endorsement.** It means nothing has
been shown either way. Most reviews of a young strategy should return KEEP, and
a review cycle that rarely does is reading noise as signal.

Every verdict carries its reasons, because "KILL" alone is unreviewable.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from datetime import datetime
from decimal import Decimal
from enum import StrEnum

from tb.core.clock import now_utc
from tb.ledger.events import Actor, EventType, StrategyReviewedPayload
from tb.ledger.store import Ledger

# Trades below which nothing but KILL may be concluded.
#
# A realised edge over four trades is a statement about four trades. KILL is
# still reachable below this because KILL does not rest on the edge estimate —
# it rests on losses, which are facts about the account rather than estimates
# about the future.
MIN_TRADES_FOR_A_POSITIVE_VERDICT = 10

# Trades before SCALE may be concluded. The same 30 as
# `promotion.min_oos_trades` and `SHRINKAGE_PRIOR_TRADES`: disbelieving a
# backtest on 29 trades and then believing a live record on 8 would be
# incoherent.
MIN_TRADES_FOR_SCALE = 30

# How much of the declared edge a strategy must still be realising to SCALE.
# Not 100%: a strategy realising its full declared edge is rarer than a bug in
# the measurement, and demanding it would mean SCALE never fires. Two thirds is
# "the edge is real and somewhat smaller than hoped", which is the ordinary
# good outcome.
SCALE_EDGE_RETENTION = Decimal("0.66")

# The fraction of a lineage's loss budget a single strategy may consume before
# it is killed regardless of anything else. Below the budget's own exhaustion,
# so one member cannot spend the whole lineage's allowance by itself.
KILL_BUDGET_FRACTION = Decimal("0.5")


class Verdict(StrEnum):
    """What to do about a strategy."""

    KEEP = "keep"
    KILL = "kill"
    ITERATE = "iterate"
    SCALE = "scale"

    @property
    def retires_the_strategy(self) -> bool:
        return self in (Verdict.KILL, Verdict.ITERATE)

    @property
    def needs_strong_evidence(self) -> bool:
        return self is Verdict.SCALE


@dataclass(frozen=True, slots=True)
class ReviewInput:
    """What the review cycle knows about one strategy."""

    strategy_id: str
    version: int
    lineage_id: str
    n_realised_trades: int
    realised_pnl_ccy: Decimal
    declared_edge_bps: Decimal
    realised_edge_bps: Decimal | None = None
    lineage_budget_ccy: Decimal | None = None
    lineage_consumed_ccy: Decimal | None = None
    days_live: int = 0
    # A strategy whose exits are being forced by the risk layer is not
    # expressing its own logic any more, whatever its P&L says.
    n_risk_blocks: int = 0

    @property
    def edge_retention(self) -> Decimal | None:
        """How much of the declared edge is actually showing up.

        `None` when there is no admissible realised edge — which is not a
        retention of zero. A strategy whose fills all had inferred prices has
        produced no measurement, and scoring it as though it earned nothing
        would punish it for the reconciler's limitations.
        """
        if self.realised_edge_bps is None or self.declared_edge_bps <= 0:
            return None
        return self.realised_edge_bps / self.declared_edge_bps


@dataclass(frozen=True, slots=True)
class Review:
    """One verdict, with the evidence that produced it."""

    strategy_id: str
    version: int
    lineage_id: str
    verdict: Verdict
    reasons: tuple[str, ...]
    n_realised_trades: int
    realised_pnl_ccy: Decimal
    realised_edge_bps: Decimal | None
    declared_edge_bps: Decimal
    evidence_sufficient: bool
    reviewed_at: datetime

    @property
    def label(self) -> str:
        return f"{self.strategy_id}@v{self.version}"

    def summary(self) -> str:
        head = (
            f"{self.label}: {self.verdict.value.upper()} on {self.n_realised_trades} "
            f"trade(s), realised {self.realised_pnl_ccy}"
        )
        return "\n".join([head, *(f"  - {reason}" for reason in self.reasons)])


def review(candidate: ReviewInput, *, at: datetime | None = None) -> Review:
    """Judge one strategy. Pure, so the asymmetry is testable in isolation.

    The order of the checks is the order of the asymmetry: the kill conditions
    run first and need little evidence, then the evidence bar, then SCALE,
    which needs the most.
    """
    moment = at or now_utc()
    reasons: list[str] = []
    enough = candidate.n_realised_trades >= MIN_TRADES_FOR_A_POSITIVE_VERDICT

    budget_spent = _budget_fraction(candidate)
    if budget_spent is not None and budget_spent >= KILL_BUDGET_FRACTION:
        reasons.append(
            f"has consumed {budget_spent:.0%} of its lineage's loss budget, past the "
            f"{KILL_BUDGET_FRACTION:.0%} at which one member is killed so it cannot "
            "spend the whole lineage's allowance by itself"
        )
        return _verdict(candidate, Verdict.KILL, reasons, enough, moment)

    if candidate.n_risk_blocks > 0 and candidate.realised_pnl_ccy < 0:
        reasons.append(
            f"{candidate.n_risk_blocks} risk block(s) alongside a realised loss of "
            f"{candidate.realised_pnl_ccy}: a strategy whose exits are being forced by "
            "the risk layer is not expressing its own logic any more"
        )
        return _verdict(candidate, Verdict.KILL, reasons, enough, moment)

    if not enough:
        reasons.append(
            f"{candidate.n_realised_trades} trade(s), below the "
            f"{MIN_TRADES_FOR_A_POSITIVE_VERDICT} a positive verdict needs. KEEP here "
            "means nothing has been shown either way, not that anything is working."
        )
        return _verdict(candidate, Verdict.KEEP, reasons, enough, moment)

    retention = candidate.edge_retention
    if candidate.realised_pnl_ccy < 0:
        reasons.append(
            f"realised {candidate.realised_pnl_ccy} over {candidate.n_realised_trades} "
            "trades with enough of them to mean something"
        )
        if retention is not None and retention > 0:
            reasons.append(
                f"realising {retention:.0%} of its declared "
                f"{candidate.declared_edge_bps}bps: the shape may be right and the "
                "parameters wrong, so a mutation is worth a try"
            )
            return _verdict(candidate, Verdict.ITERATE, reasons, enough, moment)
        return _verdict(candidate, Verdict.KILL, reasons, enough, moment)

    if candidate.n_realised_trades >= MIN_TRADES_FOR_SCALE:
        if retention is None:
            reasons.append(
                "no admissible realised edge: every fill so far had an inferred price, "
                "so there is nothing to scale on. Not a retention of zero — a "
                "measurement that does not exist."
            )
            return _verdict(candidate, Verdict.KEEP, reasons, enough, moment)
        if retention >= SCALE_EDGE_RETENTION:
            reasons.append(
                f"realising {retention:.0%} of its declared "
                f"{candidate.declared_edge_bps}bps over {candidate.n_realised_trades} "
                f"trades, past the {SCALE_EDGE_RETENTION:.0%} bar"
            )
            return _verdict(candidate, Verdict.SCALE, reasons, enough, moment)
        reasons.append(
            f"profitable but realising only {retention:.0%} of its declared edge, "
            f"short of the {SCALE_EDGE_RETENTION:.0%} SCALE needs"
        )
        return _verdict(candidate, Verdict.KEEP, reasons, enough, moment)

    reasons.append(
        f"profitable over {candidate.n_realised_trades} trades, but SCALE needs "
        f"{MIN_TRADES_FOR_SCALE} — the promotion gate refuses to believe a backtest on "
        "fewer, so believing a live record on fewer would be incoherent"
    )
    return _verdict(candidate, Verdict.KEEP, reasons, enough, moment)


def _budget_fraction(candidate: ReviewInput) -> Decimal | None:
    budget = candidate.lineage_budget_ccy
    consumed = candidate.lineage_consumed_ccy
    if budget is None or consumed is None or budget <= 0:
        return None
    return consumed / budget


def _verdict(
    candidate: ReviewInput,
    verdict: Verdict,
    reasons: list[str],
    enough: bool,
    moment: datetime,
) -> Review:
    return Review(
        strategy_id=candidate.strategy_id,
        version=candidate.version,
        lineage_id=candidate.lineage_id,
        verdict=verdict,
        reasons=tuple(reasons),
        n_realised_trades=candidate.n_realised_trades,
        realised_pnl_ccy=candidate.realised_pnl_ccy,
        realised_edge_bps=candidate.realised_edge_bps,
        declared_edge_bps=candidate.declared_edge_bps,
        evidence_sufficient=enough,
        reviewed_at=moment,
    )


class ReviewCycle:
    """Runs reviews and records them."""

    def __init__(self, ledger: Ledger, *, run_id: str | None = None) -> None:
        self._ledger = ledger
        self._run_id = run_id

    def run(
        self,
        candidates: Sequence[ReviewInput],
        *,
        at: datetime | None = None,
        record: bool = True,
    ) -> list[Review]:
        results = [review(candidate, at=at) for candidate in candidates]
        if record:
            for result in results:
                self._record(result)
        return results

    def _record(self, result: Review) -> None:
        self._ledger.append(
            EventType.STRATEGY_REVIEWED,
            result.strategy_id,
            StrategyReviewedPayload(
                strategy_id=result.strategy_id,
                version=result.version,
                lineage_id=result.lineage_id,
                verdict=result.verdict.value,
                n_realised_trades=result.n_realised_trades,
                realised_pnl_ccy=str(result.realised_pnl_ccy),
                realised_edge_bps=(
                    None if result.realised_edge_bps is None else str(result.realised_edge_bps)
                ),
                declared_edge_bps=str(result.declared_edge_bps),
                evidence_sufficient=result.evidence_sufficient,
                reasons=list(result.reasons),
            ),
            actor=Actor.SYSTEM,
            run_id=self._run_id,
        )
