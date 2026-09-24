"""The gate. With no shadow period, this is the whole safety argument.

`paper_shadow_sessions: 0` was a deliberate choice, and its consequence is that
nothing stands between a generated strategy and real money except the checks in
this module. There is no human approval step, no demo period, no second look.
So the gate is built the way the rest of this system is built — every check
reports pass/fail with its observed value beside its threshold, an unmeasurable
input is a refusal rather than a pass, and the whole set runs rather than
short-circuiting on the first failure.

**Why it does not short-circuit.** "Refused by the drawdown check" and "refused
by the drawdown check and five others" call for opposite responses from the
search loop: the first is a near miss worth mutating, the second is a lineage
worth abandoning. A gate that stopped at the first failure could not tell them
apart, and the searcher would keep breeding from candidates that were never
close.

**Why unmeasured is refused.** Three inputs can come back `None`: the deflated
probability on too short a sample, PBO on too few trials, and the feed-noise
figure when no bake-off has run. Treating any of them as a pass would mean a
candidate could clear the gate by arranging for a computation to fail, which is
easier than clearing it on merit. So `None` blocks, and the reason says which
measurement is missing.

**The holding-period check is not a duplicate of the cost gate.** The cost gate
asks whether one trade can pay for itself. This asks whether the strategy's
implied holding period is expressible at the resolutions the bot is allowed to
trade live — which, per `docs/decisions/0001-minute-resolution.md`, is daily
only. A spec declaring a 30-minute hold is not an expensive strategy, it is an
unexecutable one, and catching it here rather than per trade is what stops the
searcher breeding a family the loop can never act on.
"""

from __future__ import annotations

import json
import sqlite3
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import datetime
from decimal import Decimal
from enum import StrEnum

from tb.backtest.costs import CostModel, Jurisdiction
from tb.config.hard_limits import HardLimits
from tb.core.clock import from_iso, now_utc, to_iso
from tb.core.errors import TbError
from tb.core.ids import new_id
from tb.ledger.events import Actor, EventType, PromotionPayload
from tb.ledger.store import Ledger
from tb.registry.lineage import SpecRegistry
from tb.registry.models import LineageBudget, StrategyStatus
from tb.research.holdout import HoldoutResult
from tb.research.selection import Deflation, PboResult
from tb.research.trials import Multiplicity
from tb.strategy.dsl.schema import StrategySpec

# How stale a calibration may be before the gate stops believing it. The
# calibration says the *backtester* can be trusted — that null strategies show
# no post-cost edge — and it is evidence about a build, so a calibration from a
# different code revision is evidence about a different backtester. Days rather
# than a revision check because a rebuild with no engine change is common and
# re-running the calibration is cheap.
MAX_CALIBRATION_AGE_DAYS = 30

# The notional a promotion's cost arithmetic is quoted at. Floor notional would
# understate the cost ratio for a strategy that will ratchet up, and the
# published table is quoted at 1,000 — see the decision record. A strategy must
# clear the gate at the size it could reach, not only at the size it starts.
GATE_NOTIONAL_CCY = Decimal("1000.00")


class PromotionError(TbError):
    """A promotion could not be evaluated."""


class Decision(StrEnum):
    """What the gate concluded."""

    PROMOTE = "promote"
    REFUSE = "refuse"
    # Every check passed but a shadow period is configured and unserved. Not a
    # refusal: nothing is wrong with the candidate, it is waiting. Separate so
    # a searcher does not treat "come back later" as "this idea is dead".
    AWAIT_SHADOW = "await_shadow"

    @property
    def funds_the_strategy(self) -> bool:
        return self is Decision.PROMOTE


@dataclass(frozen=True, slots=True)
class GateResult:
    """One check, with its observed value beside its threshold.

    The pair is the point. A candidate that failed the drawdown check at 15.1%
    against a 15% limit is a different object from one that failed at 60%, and
    a record holding only "failed" cannot tell a searcher which.
    """

    name: str
    passed: bool
    observed: str
    threshold: str
    detail: str = ""
    # False for a check that is reported but cannot refuse — the shadow-period
    # check when no shadow is configured, for instance. Named rather than
    # inferred from `passed`, because a non-blocking check that fails must not
    # silently become a refusal if someone later reads the list as a conjunction.
    blocking: bool = True

    @property
    def refuses(self) -> bool:
        return self.blocking and not self.passed

    def as_dict(self) -> dict[str, object]:
        return {
            "name": self.name,
            "passed": self.passed,
            "observed": self.observed,
            "threshold": self.threshold,
            "detail": self.detail,
            "blocking": self.blocking,
        }

    def line(self) -> str:
        mark = "pass" if self.passed else "FAIL"
        suffix = "" if self.blocking else " (advisory)"
        return f"{mark:<4} {self.name:<28} {self.observed:>14} vs {self.threshold:<14}{suffix}"


@dataclass(frozen=True, slots=True)
class PromotionDecision:
    """Everything the gate concluded, and the evidence for each part."""

    promotion_id: str
    strategy_id: str
    version: int
    lineage_id: str
    spec_hash: str
    decision: Decision
    gates: tuple[GateResult, ...]
    decided_at: datetime
    deflated_sharpe: float | None = None
    deflated_sharpe_probability: float | None = None
    pbo: float | None = None
    n_trials_deflated_by: int | None = None
    vintage_id: str | None = None
    holdout_evaluation_id: str | None = None
    effective_at: datetime | None = None
    caveats: tuple[str, ...] = ()

    @property
    def label(self) -> str:
        return f"{self.strategy_id}@v{self.version}"

    @property
    def n_failed(self) -> int:
        return sum(1 for gate in self.gates if gate.refuses)

    @property
    def failures(self) -> tuple[GateResult, ...]:
        return tuple(gate for gate in self.gates if gate.refuses)

    def report(self) -> str:
        lines = [gate.line() for gate in self.gates]
        lines.append("")
        lines.append(f"{self.label}: {self.decision.value} ({self.n_failed} blocking failure(s))")
        for caveat in self.caveats:
            lines.append(f"  caveat: {caveat}")
        return "\n".join(lines)


@dataclass(frozen=True, slots=True)
class PromotionEvidence:
    """Everything the gate reads, assembled by the caller.

    A value object rather than a set of stores the gate queries, for one
    reason: the gate is the single most important piece of logic in this
    system, and it must be testable exhaustively without a database, a vintage
    on disk or a backtest run. Every hole in the gate is a strategy that should
    not have been funded, so the cost of assembling this explicitly is worth it.
    """

    spec: StrategySpec
    lineage_id: str
    # Out-of-sample results — from the holdout, not the training window. A gate
    # reading training numbers is not a gate.
    holdout: HoldoutResult | None
    multiplicity: Multiplicity | None
    deflation: Deflation | None
    pbo: PboResult | None
    vintage_id: str | None
    vintage_admissible: bool
    vintage_caveats: tuple[str, ...] = ()
    calibration_passed: bool | None = None
    calibration_age_days: float | None = None
    budget: LineageBudget | None = None
    # p95 cross-provider disagreement in bps, from `tb data bakeoff`. `None`
    # means no bake-off has produced one, which is a refusal rather than a
    # zero: an unmeasured error bar is not a small one.
    feed_noise_p95_bps: Decimal | None = None
    jurisdiction: Jurisdiction = Jurisdiction.US
    instrument_currency: str = "USD"
    shadow_sessions_served: int = 0
    shadow_trades_served: int = 0


class PromotionGate:
    """Runs every check and records the decision.

    Holds the hard limits and the cost model; everything else arrives as
    evidence. The limits are the hash-pinned file, so the thresholds this gate
    enforces cannot be changed by anything the bot does.
    """

    def __init__(
        self,
        ledger: Ledger,
        *,
        limits: HardLimits,
        cost_model: CostModel | None = None,
        run_id: str | None = None,
    ) -> None:
        self._ledger = ledger
        self._limits = limits
        self._costs = cost_model or CostModel(limits=limits)
        self._run_id = run_id

    # -- evaluation --------------------------------------------------------

    def evaluate(
        self,
        *,
        strategy_id: str,
        version: int = 1,
        evidence: PromotionEvidence,
        at: datetime | None = None,
        record: bool = True,
    ) -> PromotionDecision:
        """Run every gate. Records the decision unless asked not to.

        `record=False` exists for the null-strategy release gate, which
        evaluates a thousand candidates and would otherwise put a thousand
        events into the hash chain to say the same thing a thousand times. It
        changes nothing about the arithmetic — the same function computes the
        same verdicts either way, which is what makes that test evidence about
        the real gate rather than about a copy of it.
        """
        moment = at or now_utc()
        gates = self._run_gates(evidence)

        blocking_failures = [gate for gate in gates if gate.refuses]
        shadow_pending = any(gate.name == "paper_shadow" and not gate.passed for gate in gates)
        if blocking_failures:
            decision = Decision.REFUSE
        elif shadow_pending:
            decision = Decision.AWAIT_SHADOW
        else:
            decision = Decision.PROMOTE

        result = PromotionDecision(
            promotion_id=new_id("prom", length=12),
            strategy_id=strategy_id,
            version=version,
            lineage_id=evidence.lineage_id,
            spec_hash=evidence.spec.spec_hash,
            decision=decision,
            gates=tuple(gates),
            decided_at=moment,
            deflated_sharpe=(
                None if evidence.deflation is None else evidence.deflation.deflated_sharpe
            ),
            deflated_sharpe_probability=(
                None if evidence.deflation is None else evidence.deflation.deflated_probability
            ),
            pbo=None if evidence.pbo is None else evidence.pbo.pbo,
            n_trials_deflated_by=(
                None if evidence.multiplicity is None else evidence.multiplicity.n_trials
            ),
            vintage_id=evidence.vintage_id,
            holdout_evaluation_id=(
                None if evidence.holdout is None else evidence.holdout.evaluation_id
            ),
            effective_at=moment if decision.funds_the_strategy else None,
            caveats=_caveats(evidence),
        )

        if record:
            self._record(
                result,
                promote=decision.funds_the_strategy,
                queue_for_holdout=_awaiting_only_the_holdout(
                    result, holdout_run=evidence.holdout is not None
                ),
            )
        return result

    # -- the checks --------------------------------------------------------

    def _run_gates(self, evidence: PromotionEvidence) -> list[GateResult]:
        """Every check, in the order a reader should think about them.

        Evidence admissibility first, then the out-of-sample statistics, then
        the executability checks. A candidate that fails the first group has
        numbers that mean nothing, and reading its Sharpe would be reading
        noise — so the order is also the order in which the failures should be
        interpreted.
        """
        promotion = self._limits.promotion
        gates: list[GateResult] = []

        gates.append(self._vintage_gate(evidence))
        gates.append(self._calibration_gate(evidence))
        gates.append(self._holdout_gate(evidence))
        gates.append(self._trades_gate(evidence))
        gates.append(self._deflated_level_gate(evidence))
        gates.append(self._deflated_probability_gate(evidence))
        gates.append(self._pbo_gate(evidence))
        gates.append(self._drawdown_gate(evidence))
        gates.append(self._cost_gate(evidence))
        gates.append(self._feed_noise_gate(evidence))
        gates.append(self._holding_period_gate(evidence))
        gates.append(self._declared_edge_gate(evidence))
        gates.append(self._budget_gate(evidence))
        gates.append(self._shadow_gate(evidence, promotion.paper_shadow_sessions))
        return gates

    def _vintage_gate(self, evidence: PromotionEvidence) -> GateResult:
        """A backtest that names no sealed vintage is not evidence.

        The data it ran on has been free to change since, so the result cannot
        be reproduced or checked — which makes it unfalsifiable rather than
        merely unverified.
        """
        named = evidence.vintage_id or "(none)"
        return GateResult(
            name="sealed_vintage",
            passed=bool(evidence.vintage_id) and evidence.vintage_admissible,
            observed=named,
            threshold="sealed and non-empty",
            detail=(
                ""
                if evidence.vintage_admissible and evidence.vintage_id
                else "a backtest not pinned to a sealed vintage cannot be reproduced"
            ),
        )

    def _calibration_gate(self, evidence: PromotionEvidence) -> GateResult:
        """Whether the backtester itself has recently been shown to be honest.

        Evidence about the engine, not the strategy: null strategies must show
        a post-cost Sharpe of about the cost drag and no better. Without a
        current calibration, every number below is produced by a backtester
        nobody has checked.
        """
        age = evidence.calibration_age_days
        passed = (
            evidence.calibration_passed is True
            and age is not None
            and age <= MAX_CALIBRATION_AGE_DAYS
        )
        if evidence.calibration_passed is None:
            observed = "never run"
        elif not evidence.calibration_passed:
            observed = "failed"
        else:
            observed = f"{age:.0f}d old" if age is not None else "age unknown"
        return GateResult(
            name="engine_calibrated",
            passed=passed,
            observed=observed,
            threshold=f"passed within {MAX_CALIBRATION_AGE_DAYS}d",
            detail=(
                ""
                if passed
                else "without a current calibration these numbers come from a "
                "backtester nobody has checked for lookahead or under-charging"
            ),
        )

    def _holdout_gate(self, evidence: PromotionEvidence) -> GateResult:
        """The holdout must have been evaluated, once, and passed."""
        holdout = evidence.holdout
        if holdout is None:
            return GateResult(
                name="sealed_holdout",
                passed=False,
                observed="not evaluated",
                threshold="evaluated and passed",
                detail=(
                    "the holdout is the only evidence in this decision the search could "
                    "not have fitted to; without it the gate is reading the search's own "
                    "output back to itself"
                ),
            )
        return GateResult(
            name="sealed_holdout",
            passed=holdout.passed,
            observed="passed" if holdout.passed else "failed",
            threshold="evaluated and passed",
            detail=holdout.detail,
        )

    def _trades_gate(self, evidence: PromotionEvidence) -> GateResult:
        minimum = self._limits.promotion.min_oos_trades
        observed = None if evidence.holdout is None else evidence.holdout.n_trades
        return GateResult(
            name="oos_trades",
            passed=observed is not None and observed >= minimum,
            observed="unmeasured" if observed is None else str(observed),
            threshold=f">= {minimum}",
            detail=(
                ""
                if observed is not None and observed >= minimum
                else "a Sharpe over a handful of trades is a statement about a handful of trades"
            ),
        )

    def _deflated_level_gate(self, evidence: PromotionEvidence) -> GateResult:
        minimum = self._limits.promotion.min_oos_deflated_sharpe
        deflation = evidence.deflation
        if deflation is None:
            return GateResult(
                name="deflated_sharpe",
                passed=False,
                observed="unmeasured",
                threshold=f">= {minimum}",
                detail="no deflation was computed, so the search size is unaccounted for",
            )
        return GateResult(
            name="deflated_sharpe",
            passed=deflation.deflated_sharpe >= minimum,
            observed=f"{deflation.deflated_sharpe:.3f}",
            threshold=f">= {minimum}",
            detail=(
                f"raw {deflation.observed_sharpe:.3f} less the expected maximum of "
                f"{deflation.n_trials} trial(s) ({deflation.expected_max_sharpe:.3f})"
            ),
        )

    def _deflated_probability_gate(self, evidence: PromotionEvidence) -> GateResult:
        minimum = self._limits.promotion.min_deflated_sharpe_probability
        deflation = evidence.deflation
        probability = None if deflation is None else deflation.deflated_probability
        if probability is None:
            return GateResult(
                name="deflated_probability",
                passed=False,
                observed="unmeasured",
                threshold=f">= {minimum}",
                detail=(
                    "the sample is too short, or its higher moments too extreme, to "
                    "support a confidence figure — which is not the same as a low one"
                ),
            )
        return GateResult(
            name="deflated_probability",
            passed=probability >= minimum,
            observed=f"{probability:.3f}",
            threshold=f">= {minimum}",
            detail="probability the true Sharpe beats what this search would find by chance",
        )

    def _pbo_gate(self, evidence: PromotionEvidence) -> GateResult:
        maximum = self._limits.promotion.max_probability_of_backtest_overfitting
        pbo = evidence.pbo
        if pbo is None:
            return GateResult(
                name="pbo",
                passed=False,
                observed="unmeasured",
                threshold=f"<= {maximum}",
                detail=(
                    "PBO could not be computed — too few trials to select among, or "
                    "windows too short to cross-validate. 'We could not check whether "
                    "this was overfit' is not evidence that it was not."
                ),
            )
        return GateResult(
            name="pbo",
            passed=pbo.pbo <= maximum,
            observed=f"{pbo.pbo:.3f}",
            threshold=f"<= {maximum}",
            detail=f"over {pbo.n_combinations} train/test splits of {pbo.n_trials} trials",
        )

    def _drawdown_gate(self, evidence: PromotionEvidence) -> GateResult:
        maximum = float(self._limits.promotion.max_oos_drawdown_pct)
        observed = None if evidence.holdout is None else evidence.holdout.max_drawdown_pct
        return GateResult(
            name="oos_drawdown",
            passed=observed is not None and observed <= maximum,
            observed="unmeasured" if observed is None else f"{observed:.2f}%",
            threshold=f"<= {maximum}%",
        )

    def _cost_gate(self, evidence: PromotionEvidence) -> GateResult:
        """The binding constraint on this venue, applied before funding.

        Quoted at `GATE_NOTIONAL_CCY` rather than at floor notional. A strategy
        promoted at floor size ratchets up, and cost in bps is mildly
        notional-dependent, so gating at the smallest size it will ever trade
        would admit a strategy that stops clearing the gate as it grows.
        """
        verdict, trip = self._costs.gate_trade(
            notional_ccy=GATE_NOTIONAL_CCY,
            instrument_currency=evidence.instrument_currency,
            jurisdiction=evidence.jurisdiction,
            expected_edge_bps=evidence.spec.expected_edge_bps,
        )
        ratio = "n/a" if verdict.ratio is None else f"{verdict.ratio:.3f}"
        return GateResult(
            name="cost_to_edge",
            passed=verdict.allowed,
            observed=ratio,
            threshold=f"<= {verdict.max_ratio}",
            detail=(
                f"{trip.total_bps:.1f}bps round trip against a declared "
                f"{evidence.spec.expected_edge_bps}bps edge"
            ),
        )

    def _feed_noise_gate(self, evidence: PromotionEvidence) -> GateResult:
        """The declared edge must exceed the error bar on our own prices.

        Without this the bot would trade an edge smaller than the disagreement
        between the two feeds that produced it — which is not a strategy, it is
        a measurement of vendor noise.
        """
        ratio = Decimal(str(self._limits.data.min_edge_to_feed_noise_ratio))
        noise = evidence.feed_noise_p95_bps
        if noise is None:
            return GateResult(
                name="edge_to_feed_noise",
                passed=False,
                observed="unmeasured",
                threshold=f">= {ratio}x p95 disagreement",
                detail=(
                    "no bake-off has produced a p95 cross-provider disagreement. An "
                    "unmeasured error bar is not a small one, so this refuses rather "
                    "than assuming zero."
                ),
            )
        required = noise * ratio
        edge = evidence.spec.expected_edge_bps
        return GateResult(
            name="edge_to_feed_noise",
            passed=edge >= required,
            observed=f"{edge}bps",
            threshold=f">= {required}bps",
            detail=f"p95 feed disagreement {noise}bps x {ratio}",
        )

    def _holding_period_gate(self, evidence: PromotionEvidence) -> GateResult:
        """Can this strategy's holding period be expressed at a permitted resolution?

        Not a duplicate of the cost gate. Minute resolution is refused on
        measured evidence (docs/decisions/0001-minute-resolution.md), so a spec
        implying a 30-minute hold is not expensive, it is unexecutable — and the
        searcher would otherwise breed a whole family the loop can never act on
        and record them all as failures of the cost gate.
        """
        allowed = set(self._limits.data.allowed_live_resolutions)
        shortest = _shortest_permitted_minutes(allowed)
        risk_minimum = self._limits.execution.min_holding_minutes
        required = max(shortest, risk_minimum)
        declared = evidence.spec.min_holding_minutes
        return GateResult(
            name="holding_period",
            passed=declared >= required,
            observed=f"{declared}min",
            threshold=f">= {required}min",
            detail=(
                f"live resolutions are {sorted(allowed)}, so a position cannot change "
                f"more often than every {shortest}min; the risk layer's own minimum is "
                f"{risk_minimum}min"
            ),
        )

    def _declared_edge_gate(self, evidence: PromotionEvidence) -> GateResult:
        """The declared edge must be inside the bounds a spec may claim.

        The upper bound is not tuning — it closes a hole. The cost gate divides
        cost by the strategy's own declared edge, so a spec claiming 10,000bps
        would pass it trivially, and the one control keeping the search loop out
        of the fee trap would be defeatable by the search loop.
        """
        low = self._limits.costs.min_expected_edge_bps
        high = self._limits.costs.max_expected_edge_bps
        edge = evidence.spec.expected_edge_bps
        return GateResult(
            name="declared_edge_bounds",
            passed=low <= edge <= high,
            observed=f"{edge}bps",
            threshold=f"{low}-{high}bps",
            detail=(
                ""
                if low <= edge <= high
                else "a spec outside these bounds is malformed, not optimistic: the cost "
                "gate divides by this number"
            ),
        )

    def _budget_gate(self, evidence: PromotionEvidence) -> GateResult:
        budget = evidence.budget
        if budget is None:
            return GateResult(
                name="lineage_budget",
                passed=True,
                observed="fresh",
                threshold="not exhausted",
                detail="no losses charged to this lineage yet",
            )
        return GateResult(
            name="lineage_budget",
            passed=not budget.is_exhausted,
            observed=f"{budget.consumed_ccy} of {budget.budget_ccy}",
            threshold="not exhausted",
            detail=(
                ""
                if not budget.is_exhausted
                else "this lineage has spent its lifetime loss budget; a child inherits "
                "the exhaustion, which is the point of charging it per lineage"
            ),
        )

    def _shadow_gate(self, evidence: PromotionEvidence, required: int) -> GateResult:
        """The paper-shadow period, if one is configured.

        Configured to zero per the operator's choice, which is why every other
        check here carries the weight. Reported anyway, and as a *blocking*
        check when it is configured, so turning it back on is a config edit
        rather than a code change.
        """
        if required <= 0:
            return GateResult(
                name="paper_shadow",
                passed=True,
                observed="not required",
                threshold="0 sessions",
                blocking=False,
                detail=(
                    "paper_shadow_sessions is 0, so this gate is the only thing between "
                    "a generated strategy and real money"
                ),
            )
        trades_required = self._limits.promotion.paper_shadow_min_trades
        enough = (
            evidence.shadow_sessions_served >= required
            and evidence.shadow_trades_served >= trades_required
        )
        return GateResult(
            name="paper_shadow",
            passed=enough,
            observed=(
                f"{evidence.shadow_sessions_served} sessions / "
                f"{evidence.shadow_trades_served} trades"
            ),
            threshold=f"{required} sessions / {trades_required} trades",
            # Non-blocking so an unserved shadow reads as AWAIT_SHADOW rather
            # than as a refusal: nothing is wrong with the candidate, it is
            # waiting, and a searcher must not mutate away from a good idea
            # because it was early.
            blocking=False,
        )

    # -- recording ---------------------------------------------------------

    def _record(self, result: PromotionDecision, *, promote: bool, queue_for_holdout: bool) -> None:
        with self._ledger.transaction() as tx:
            event = tx.append(
                EventType.PROMOTION_EVALUATED,
                result.strategy_id,
                PromotionPayload(
                    promotion_id=result.promotion_id,
                    strategy_id=result.strategy_id,
                    version=result.version,
                    lineage_id=result.lineage_id,
                    spec_hash=result.spec_hash,
                    decision=result.decision.value,
                    n_gates=len(result.gates),
                    n_failed=result.n_failed,
                    gate_results=[gate.as_dict() for gate in result.gates],
                    deflated_sharpe=result.deflated_sharpe,
                    deflated_sharpe_probability=result.deflated_sharpe_probability,
                    pbo=result.pbo,
                    n_trials_deflated_by=result.n_trials_deflated_by,
                    vintage_id=result.vintage_id,
                    holdout_evaluation_id=result.holdout_evaluation_id,
                    effective_at=(
                        None if result.effective_at is None else to_iso(result.effective_at)
                    ),
                ),
                actor=Actor.SYSTEM,
                run_id=self._run_id,
            )
            tx.execute(
                """
                INSERT INTO promotions (
                    promotion_id, strategy_id, version, lineage_id, spec_hash, decision,
                    n_gates, n_failed, gate_results_json, deflated_sharpe,
                    deflated_sharpe_probability, pbo, n_trials_deflated_by, vintage_id,
                    holdout_evaluation_id, effective_at, decided_at, deciding_event_seq
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    result.promotion_id,
                    result.strategy_id,
                    result.version,
                    result.lineage_id,
                    result.spec_hash,
                    result.decision.value,
                    len(result.gates),
                    result.n_failed,
                    json.dumps([gate.as_dict() for gate in result.gates]),
                    result.deflated_sharpe,
                    result.deflated_sharpe_probability,
                    result.pbo,
                    result.n_trials_deflated_by,
                    result.vintage_id,
                    result.holdout_evaluation_id,
                    None if result.effective_at is None else to_iso(result.effective_at),
                    to_iso(result.decided_at),
                    event.seq,
                ),
            )
            if promote:
                # The single place `PROMOTED` is written. Inside the same
                # transaction as the event, so a strategy is never funded
                # without a record of the decision that funded it.
                tx.execute(
                    "UPDATE strategy_status SET status = ?, promoted_at = ?, rung = 0, "
                    "rung_changed_at = ?, updated_at = ? WHERE strategy_id = ? AND version = ?",
                    (
                        StrategyStatus.PROMOTED.value,
                        to_iso(result.decided_at),
                        to_iso(result.decided_at),
                        to_iso(result.decided_at),
                        result.strategy_id,
                        result.version,
                    ),
                )
            elif queue_for_holdout:
                # Everything checkable without spending the holdout passed, and
                # the holdout has not been run. That is not a rejection, it is a
                # queue position — and it is the only thing `AWAITING_HOLDOUT`
                # means. A refusal for any other reason leaves the status alone:
                # the gate's job is to promote or not, and quietly reclassifying
                # a candidate on the way past would lose whatever the previous
                # status recorded.
                tx.execute(
                    "UPDATE strategy_status SET status = ?, updated_at = ? "
                    "WHERE strategy_id = ? AND version = ? AND status = ?",
                    (
                        StrategyStatus.AWAITING_HOLDOUT.value,
                        to_iso(result.decided_at),
                        result.strategy_id,
                        result.version,
                        StrategyStatus.CANDIDATE.value,
                    ),
                )

    # -- reading -----------------------------------------------------------

    def history(self, strategy_id: str, version: int = 1) -> list[PromotionDecision]:
        rows = self._ledger.conn.execute(
            "SELECT * FROM promotions WHERE strategy_id = ? AND version = ? ORDER BY decided_at",
            (strategy_id, version),
        ).fetchall()
        return [_row_to_decision(row) for row in rows]

    def latest(self, strategy_id: str, version: int = 1) -> PromotionDecision | None:
        history = self.history(strategy_id, version)
        return history[-1] if history else None


# --------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------

# How often a position may change at each resolution. Daily is a *session*, not
# 24 hours: the gate asks how long a position must be held, and a position
# entered on one session's close and exited on the next is held overnight —
# about 24 hours of wall clock, which is what the number says.
_RESOLUTION_MINUTES = {"minute": 1, "hourly": 60, "daily": 1440}


# The gates that are functions of the out-of-sample result, so an absent
# holdout makes all of them fail at once. Naming the set is what lets the gate
# tell "this candidate has not been checked yet" apart from "this candidate was
# checked and is bad" — the first draft compared a failure *count* of one, and
# a test found that a missing holdout fails three checks, not one.
#
# Everything outside this set is computable before the holdout is spent: PBO
# comes from the training trial matrix, and cost, feed noise, holding period,
# edge bounds, budget, vintage and calibration are all properties of the spec
# or of the evidence around it. So "every failure is in this set" is exactly
# the pre-holdout screen: nothing we could have checked cheaply is wrong.
_HOLDOUT_DERIVED = frozenset(
    {
        "sealed_holdout",
        "oos_trades",
        "oos_drawdown",
        "deflated_sharpe",
        "deflated_probability",
    }
)


def _awaiting_only_the_holdout(result: PromotionDecision, *, holdout_run: bool) -> bool:
    """Whether the only thing missing is the holdout evaluation itself.

    Requires that the holdout has genuinely not been run. A candidate whose
    holdout ran and *failed* is refused, not queued: failing is terminal, and
    treating it as "waiting" would invite the retry the uniqueness constraint
    exists to prevent.
    """
    if holdout_run:
        return False
    failures = result.failures
    return bool(failures) and all(gate.name in _HOLDOUT_DERIVED for gate in failures)


def _shortest_permitted_minutes(allowed: set[str]) -> int:
    """The fastest a position may turn over, given the permitted resolutions.

    Unknown resolution names fall back to the slowest known one rather than
    being ignored. Ignoring an unrecognised name would mean a typo in the
    limits file silently loosened this check, and the limits file is the one
    place a typo must not do that.
    """
    known = [_RESOLUTION_MINUTES[name] for name in allowed if name in _RESOLUTION_MINUTES]
    if not known:
        return max(_RESOLUTION_MINUTES.values())
    return min(known)


def _caveats(evidence: PromotionEvidence) -> tuple[str, ...]:
    """Everything true about this decision that no gate refuses on.

    Recorded rather than dropped. A vintage flagged `survivorship: unmeasured`
    does not fail the gate — on free data it cannot be anything else — but a
    promotion made on it should say so, because the alternative is a decision
    that looks unqualified in the record.
    """
    notes = list(evidence.vintage_caveats)
    if evidence.multiplicity is not None:
        notes.extend(evidence.multiplicity.caveats)
    if evidence.deflation is not None and not evidence.deflation.dispersion_measured:
        notes.append(
            "the trial-Sharpe dispersion behind the haircut was a conservative default "
            "rather than a measurement"
        )
    return tuple(notes)


def _row_to_decision(row: sqlite3.Row) -> PromotionDecision:
    raw_gates: Sequence[dict[str, object]] = json.loads(str(row["gate_results_json"]))
    gates = tuple(
        GateResult(
            name=str(entry["name"]),
            passed=bool(entry["passed"]),
            observed=str(entry["observed"]),
            threshold=str(entry["threshold"]),
            detail=str(entry.get("detail", "")),
            blocking=bool(entry.get("blocking", True)),
        )
        for entry in raw_gates
    )
    effective = row["effective_at"]
    return PromotionDecision(
        promotion_id=str(row["promotion_id"]),
        strategy_id=str(row["strategy_id"]),
        version=int(row["version"]),
        lineage_id=str(row["lineage_id"]),
        spec_hash=str(row["spec_hash"]),
        decision=Decision(str(row["decision"])),
        gates=gates,
        decided_at=from_iso(str(row["decided_at"])),
        deflated_sharpe=(None if row["deflated_sharpe"] is None else float(row["deflated_sharpe"])),
        deflated_sharpe_probability=(
            None
            if row["deflated_sharpe_probability"] is None
            else float(row["deflated_sharpe_probability"])
        ),
        pbo=None if row["pbo"] is None else float(row["pbo"]),
        n_trials_deflated_by=(
            None if row["n_trials_deflated_by"] is None else int(row["n_trials_deflated_by"])
        ),
        vintage_id=None if row["vintage_id"] is None else str(row["vintage_id"]),
        holdout_evaluation_id=(
            None if row["holdout_evaluation_id"] is None else str(row["holdout_evaluation_id"])
        ),
        effective_at=None if effective is None else from_iso(str(effective)),
    )


@dataclass(frozen=True, slots=True)
class CalibrationStatus:
    """Whether the backtester has recently been shown to be honest."""

    passed: bool | None
    age_days: float | None
    calibration_id: str | None = None
    ran_at: datetime | None = None


def latest_calibration(ledger: Ledger, *, at: datetime | None = None) -> CalibrationStatus:
    """The most recent calibration run, and how old it is.

    Reads the most recent run whatever its verdict, rather than the most recent
    *passing* one. A failing calibration is the finding — an engine that just
    failed its honesty check must not be treated as uncalibrated-but-fine
    because an older passing run exists.
    """
    row: sqlite3.Row | None = ledger.conn.execute(
        "SELECT calibration_id, passed, ran_at FROM backtest_calibrations "
        "ORDER BY ran_at DESC LIMIT 1"
    ).fetchone()
    if row is None:
        return CalibrationStatus(passed=None, age_days=None)
    ran_at = from_iso(str(row["ran_at"]))
    moment = at or now_utc()
    return CalibrationStatus(
        passed=bool(row["passed"]),
        age_days=(moment - ran_at).total_seconds() / 86_400.0,
        calibration_id=str(row["calibration_id"]),
        ran_at=ran_at,
    )


@dataclass(slots=True)
class EvidenceBuilder:
    """Assembles a `PromotionEvidence` from the stores.

    Separate from the gate so the gate stays pure over its evidence — every
    hole in the gate is a strategy that should not have been funded, and a gate
    that queried its own inputs could not be exhaustively tested without a
    database, a vintage on disk and a backtest run.
    """

    ledger: Ledger
    registry: SpecRegistry

    def build(
        self,
        *,
        spec: StrategySpec,
        lineage_id: str,
        holdout: HoldoutResult | None,
        multiplicity: Multiplicity | None,
        deflation: Deflation | None,
        pbo: PboResult | None,
        vintage_id: str | None,
        vintage_admissible: bool,
        vintage_caveats: Sequence[str] = (),
        feed_noise_p95_bps: Decimal | None = None,
        jurisdiction: Jurisdiction = Jurisdiction.US,
        instrument_currency: str = "USD",
        at: datetime | None = None,
    ) -> PromotionEvidence:
        calibration = latest_calibration(self.ledger, at=at)
        return PromotionEvidence(
            spec=spec,
            lineage_id=lineage_id,
            holdout=holdout,
            multiplicity=multiplicity,
            deflation=deflation,
            pbo=pbo,
            vintage_id=vintage_id,
            vintage_admissible=vintage_admissible,
            vintage_caveats=tuple(vintage_caveats),
            calibration_passed=calibration.passed,
            calibration_age_days=calibration.age_days,
            budget=self.registry.budget_for(lineage_id),
            feed_noise_p95_bps=feed_noise_p95_bps,
            jurisdiction=jurisdiction,
            instrument_currency=instrument_currency,
        )
