"""The promotion gate. With no shadow period this is the whole safety argument.

Every test here is written the same way: build evidence that would promote, then
break exactly one thing and assert the gate refuses for that reason. A gate test
that only checks the happy path is a gate test that would pass against a gate
that always says yes.
"""

from __future__ import annotations

from dataclasses import replace
from datetime import UTC, datetime, timedelta
from decimal import Decimal

import pytest

from tb.backtest.costs import Jurisdiction
from tb.config.loader import PinnedLimits
from tb.ledger.events import EventType
from tb.ledger.store import Ledger
from tb.registry.lineage import SpecRegistry
from tb.registry.models import AuthorKind, StrategyStatus
from tb.registry.promotion import (
    GATE_NOTIONAL_CCY,
    MAX_CALIBRATION_AGE_DAYS,
    CalibrationStatus,
    Decision,
    EvidenceBuilder,
    PromotionEvidence,
    PromotionGate,
    latest_calibration,
)
from tb.research.holdout import HoldoutResult
from tb.research.selection import Deflation, PboResult
from tb.research.trials import Multiplicity
from tb.strategy.dsl.schema import StrategySpec

AS_OF = datetime(2026, 6, 1, 14, 0, tzinfo=UTC)
SEAL = datetime(2025, 1, 1, tzinfo=UTC)
LINEAGE = "lin_1"
STRATEGY = "stg_1"


def a_spec(*, edge: str = "300", hold: int = 1440) -> StrategySpec:
    return StrategySpec.model_validate(
        {
            "name": "cross",
            "entry": {
                "kind": "compare",
                "op": "gt",
                "left": {"kind": "feature", "name": "sma", "lookback": 20},
                "right": {"kind": "feature", "name": "sma", "lookback": 100},
            },
            "exit": {
                "kind": "compare",
                "op": "lt",
                "left": {"kind": "feature", "name": "sma", "lookback": 20},
                "right": {"kind": "feature", "name": "sma", "lookback": 100},
            },
            "expected_edge_bps": edge,
            "min_holding_minutes": hold,
        }
    )


def a_holdout(*, passed: bool = True, trades: int = 45, drawdown: float = 8.0) -> HoldoutResult:
    return HoldoutResult(
        evaluation_id="hold_1",
        strategy_id=STRATEGY,
        version=1,
        lineage_id=LINEAGE,
        spec_hash="hash_1",
        vintage_id="vint_1",
        sealed_from=SEAL,
        passed=passed,
        evaluated_at=AS_OF,
        n_trades=trades,
        net_sharpe=2.4,
        net_return_pct=18.0,
        max_drawdown_pct=drawdown,
    )


def a_deflation(*, level: float = 1.2, probability: float | None = 0.99) -> Deflation:
    return Deflation(
        observed_sharpe=2.4,
        expected_max_sharpe=1.2,
        deflated_sharpe=level,
        deflated_probability=probability,
        n_trials=40,
        sharpe_dispersion=0.6,
        n_periods=600,
        skew=-0.1,
        kurtosis=3.2,
    )


def a_pbo(*, value: float = 0.1) -> PboResult:
    return PboResult(
        pbo=value,
        n_combinations=70,
        n_trials=40,
        n_periods=600,
        n_splits=8,
        median_logit=1.4,
    )


def promotable(**overrides: object) -> PromotionEvidence:
    """Evidence that clears every gate. Each test breaks exactly one thing."""
    base = PromotionEvidence(
        spec=a_spec(),
        lineage_id=LINEAGE,
        holdout=a_holdout(),
        multiplicity=Multiplicity(
            n_trials=40,
            n_lineage_trials=40,
            n_search_trials=40,
            sharpe_dispersion=0.6,
            dispersion_measured=True,
            n_measurable=40,
        ),
        deflation=a_deflation(),
        pbo=a_pbo(),
        vintage_id="vint_1",
        vintage_admissible=True,
        calibration_passed=True,
        calibration_age_days=3.0,
        feed_noise_p95_bps=Decimal("40"),
    )
    return replace(base, **overrides)  # type: ignore[arg-type]


@pytest.fixture
def registry(ledger: Ledger) -> SpecRegistry:
    return SpecRegistry(ledger, per_lineage_budget_ccy=Decimal("100"), run_id="run_test")


@pytest.fixture
def gate(ledger: Ledger, pinned: PinnedLimits) -> PromotionGate:
    return PromotionGate(ledger, limits=pinned.limits, run_id="run_test")


@pytest.fixture
def registered(registry: SpecRegistry) -> tuple[str, str]:
    """A registered candidate, so the gate has a status row to write."""
    spec = registry.register(a_spec(), author_kind=AuthorKind.SEARCH, at=AS_OF)
    return spec.strategy_id, spec.lineage_id


def run(
    gate: PromotionGate,
    evidence: PromotionEvidence,
    *,
    strategy_id: str = STRATEGY,
    record: bool = False,
) -> object:
    return gate.evaluate(strategy_id=strategy_id, evidence=evidence, at=AS_OF, record=record)


def failing_names(decision: object) -> set[str]:
    return {gate.name for gate in decision.failures}  # type: ignore[attr-defined]


# --------------------------------------------------------------------------
# The happy path — and the check that it is not vacuous
# --------------------------------------------------------------------------


def test_a_good_candidate_is_promoted(gate: PromotionGate) -> None:
    decision = run(gate, promotable())
    assert decision.decision is Decision.PROMOTE  # type: ignore[attr-defined]
    assert decision.n_failed == 0  # type: ignore[attr-defined]


def test_every_gate_reports_its_observed_value_and_threshold(gate: PromotionGate) -> None:
    decision = run(gate, promotable())
    for result in decision.gates:  # type: ignore[attr-defined]
        assert result.observed, f"{result.name} reported no observed value"
        assert result.threshold, f"{result.name} reported no threshold"
    report = decision.report()  # type: ignore[attr-defined]
    assert "cost_to_edge" in report
    assert "deflated_sharpe" in report


def test_the_gate_does_not_short_circuit(gate: PromotionGate) -> None:
    """'Refused by one check' and 'refused by six' call for opposite responses
    from the search loop, and a gate that stopped at the first cannot say which.
    """
    broken = promotable(
        holdout=a_holdout(passed=False, trades=2, drawdown=40.0),
        deflation=a_deflation(level=-2.0, probability=0.01),
        pbo=a_pbo(value=0.9),
    )
    decision = run(gate, broken)
    assert decision.n_failed >= 5  # type: ignore[attr-defined]
    assert {"sealed_holdout", "oos_trades", "deflated_sharpe", "pbo", "oos_drawdown"} <= (
        failing_names(decision)
    )


# --------------------------------------------------------------------------
# Evidence admissibility
# --------------------------------------------------------------------------


def test_an_unsealed_vintage_refuses(gate: PromotionGate) -> None:
    decision = run(gate, promotable(vintage_id=None, vintage_admissible=False))
    assert failing_names(decision) == {"sealed_vintage"}


def test_an_empty_vintage_refuses(gate: PromotionGate) -> None:
    decision = run(gate, promotable(vintage_admissible=False))
    assert "sealed_vintage" in failing_names(decision)


def test_an_uncalibrated_engine_refuses(gate: PromotionGate) -> None:
    """Without a current calibration every number below comes from a
    backtester nobody has checked for lookahead or under-charging."""
    decision = run(gate, promotable(calibration_passed=None, calibration_age_days=None))
    assert failing_names(decision) == {"engine_calibrated"}


def test_a_failed_calibration_refuses(gate: PromotionGate) -> None:
    decision = run(gate, promotable(calibration_passed=False, calibration_age_days=1.0))
    assert "engine_calibrated" in failing_names(decision)


def test_a_stale_calibration_refuses(gate: PromotionGate) -> None:
    fresh = run(gate, promotable(calibration_age_days=float(MAX_CALIBRATION_AGE_DAYS)))
    stale = run(gate, promotable(calibration_age_days=MAX_CALIBRATION_AGE_DAYS + 1.0))
    assert "engine_calibrated" not in failing_names(fresh)
    assert "engine_calibrated" in failing_names(stale)


# --------------------------------------------------------------------------
# The holdout
# --------------------------------------------------------------------------


def test_an_unevaluated_holdout_refuses(gate: PromotionGate) -> None:
    decision = run(gate, promotable(holdout=None))
    assert "sealed_holdout" in failing_names(decision)


def test_a_failed_holdout_refuses(gate: PromotionGate) -> None:
    decision = run(gate, promotable(holdout=a_holdout(passed=False)))
    assert "sealed_holdout" in failing_names(decision)


def test_too_few_out_of_sample_trades_refuses(gate: PromotionGate, pinned: PinnedLimits) -> None:
    minimum = pinned.limits.promotion.min_oos_trades
    just_enough = run(gate, promotable(holdout=a_holdout(trades=minimum)))
    one_short = run(gate, promotable(holdout=a_holdout(trades=minimum - 1)))
    assert "oos_trades" not in failing_names(just_enough)
    assert "oos_trades" in failing_names(one_short)


# --------------------------------------------------------------------------
# The two deflation gates
# --------------------------------------------------------------------------


def test_a_deflated_sharpe_below_the_threshold_refuses(
    gate: PromotionGate, pinned: PinnedLimits
) -> None:
    minimum = pinned.limits.promotion.min_oos_deflated_sharpe
    at_threshold = run(gate, promotable(deflation=a_deflation(level=minimum)))
    below = run(gate, promotable(deflation=a_deflation(level=minimum - 0.01)))
    assert "deflated_sharpe" not in failing_names(at_threshold)
    assert "deflated_sharpe" in failing_names(below)


def test_a_strong_raw_sharpe_out_of_a_huge_search_still_refuses(
    gate: PromotionGate,
) -> None:
    """The whole reason deflation exists. A raw Sharpe of 2.4 out of a thousand
    trials is what a thousand trials produce from noise."""
    huge_search = Deflation(
        observed_sharpe=2.4,
        expected_max_sharpe=3.26,
        deflated_sharpe=-0.86,
        deflated_probability=0.02,
        n_trials=1_000,
        sharpe_dispersion=1.0,
        n_periods=600,
        skew=0.0,
        kurtosis=3.0,
    )
    decision = run(gate, promotable(deflation=huge_search))
    assert {"deflated_sharpe", "deflated_probability"} <= failing_names(decision)


def test_an_unmeasurable_probability_refuses_rather_than_passing(
    gate: PromotionGate,
) -> None:
    """Otherwise a candidate clears the gate by arranging for a computation to
    fail, which is easier than clearing it on merit."""
    decision = run(gate, promotable(deflation=a_deflation(probability=None)))
    assert "deflated_probability" in failing_names(decision)


def test_a_low_probability_refuses_even_with_a_good_level(gate: PromotionGate) -> None:
    """The pair is not redundant: a comfortable level on a short sample with
    fat tails is exactly what the probability is there to catch."""
    decision = run(gate, promotable(deflation=a_deflation(level=2.0, probability=0.6)))
    assert failing_names(decision) == {"deflated_probability"}


def test_no_deflation_at_all_refuses(gate: PromotionGate) -> None:
    decision = run(gate, promotable(deflation=None))
    assert {"deflated_sharpe", "deflated_probability"} <= failing_names(decision)


# --------------------------------------------------------------------------
# PBO
# --------------------------------------------------------------------------


def test_a_high_pbo_refuses(gate: PromotionGate, pinned: PinnedLimits) -> None:
    maximum = pinned.limits.promotion.max_probability_of_backtest_overfitting
    at_threshold = run(gate, promotable(pbo=a_pbo(value=maximum)))
    above = run(gate, promotable(pbo=a_pbo(value=maximum + 0.01)))
    assert "pbo" not in failing_names(at_threshold)
    assert "pbo" in failing_names(above)


def test_an_unmeasurable_pbo_refuses(gate: PromotionGate) -> None:
    """'We could not check whether this was overfit' is not evidence that it
    was not."""
    decision = run(gate, promotable(pbo=None))
    assert failing_names(decision) == {"pbo"}


# --------------------------------------------------------------------------
# Executability
# --------------------------------------------------------------------------


def test_too_deep_an_out_of_sample_drawdown_refuses(
    gate: PromotionGate, pinned: PinnedLimits
) -> None:
    maximum = float(pinned.limits.promotion.max_oos_drawdown_pct)
    decision = run(gate, promotable(holdout=a_holdout(drawdown=maximum + 0.1)))
    assert "oos_drawdown" in failing_names(decision)


def test_an_edge_the_costs_eat_refuses(gate: PromotionGate) -> None:
    """The binding constraint on this venue, applied before funding rather than
    per trade."""
    decision = run(gate, promotable(spec=a_spec(edge="30")))
    assert "cost_to_edge" in failing_names(decision)


def test_an_irish_issuer_needs_far_more_edge(gate: PromotionGate) -> None:
    """424bps on an Irish name against 121 on a US one, from the published
    cost table. The jurisdiction is part of the evidence for that reason."""
    us = run(gate, promotable(spec=a_spec(edge="300"), jurisdiction=Jurisdiction.US))
    irish = run(
        gate,
        promotable(
            spec=a_spec(edge="300"),
            jurisdiction=Jurisdiction.IRELAND,
            instrument_currency="EUR",
        ),
    )
    assert "cost_to_edge" not in failing_names(us)
    assert "cost_to_edge" in failing_names(irish)


def test_an_edge_smaller_than_the_feed_noise_refuses(gate: PromotionGate) -> None:
    """Trading an edge smaller than the disagreement between the feeds that
    produced it is a measurement of vendor noise, not a strategy."""
    decision = run(gate, promotable(feed_noise_p95_bps=Decimal("150")))
    assert "edge_to_feed_noise" in failing_names(decision)


def test_an_unmeasured_feed_noise_refuses(gate: PromotionGate) -> None:
    """An unmeasured error bar is not a small one."""
    decision = run(gate, promotable(feed_noise_p95_bps=None))
    assert failing_names(decision) == {"edge_to_feed_noise"}


def test_a_holding_period_shorter_than_the_resolution_permits_refuses(
    gate: PromotionGate,
) -> None:
    """Not a duplicate of the cost gate. Minute resolution is refused on
    measured evidence, so a 30-minute hold is unexecutable rather than
    expensive — and the searcher would otherwise breed a whole family the loop
    can never act on."""
    decision = run(gate, promotable(spec=a_spec(hold=30)))
    assert "holding_period" in failing_names(decision)


def test_a_daily_holding_period_is_accepted(gate: PromotionGate) -> None:
    decision = run(gate, promotable(spec=a_spec(hold=1440)))
    assert "holding_period" not in failing_names(decision)


def test_an_edge_claim_above_the_ceiling_refuses(gate: PromotionGate, pinned: PinnedLimits) -> None:
    """The hole this closes: the cost gate divides by the declared edge, so a
    spec claiming 10,000bps would clear it trivially."""
    ceiling = pinned.limits.costs.max_expected_edge_bps
    decision = run(gate, promotable(spec=a_spec(edge=str(ceiling + 1))))
    assert "declared_edge_bounds" in failing_names(decision)


def test_an_exhausted_lineage_budget_refuses(gate: PromotionGate, registry: SpecRegistry) -> None:
    registry.register(a_spec(), author_kind=AuthorKind.SEARCH, at=AS_OF)
    registry.charge(LINEAGE, loss_ccy=Decimal("500"), at=AS_OF)
    decision = run(gate, promotable(budget=registry.budget_for(LINEAGE)))
    assert "lineage_budget" in failing_names(decision)


# --------------------------------------------------------------------------
# The shadow period
# --------------------------------------------------------------------------


def test_the_shadow_gate_is_advisory_when_none_is_configured(
    gate: PromotionGate,
) -> None:
    decision = run(gate, promotable())
    shadow = next(g for g in decision.gates if g.name == "paper_shadow")  # type: ignore[attr-defined]
    assert not shadow.blocking
    assert "only thing between" in shadow.detail


def test_a_configured_shadow_period_holds_rather_than_refuses(
    ledger: Ledger, write_limits: object
) -> None:
    """A candidate waiting out a shadow period is not a bad candidate, and a
    searcher must not mutate away from a good idea because it was early."""
    from tb.config.loader import load_hard_limits

    path = write_limits({"promotion": {"paper_shadow_sessions": 30, "paper_shadow_min_trades": 20}})  # type: ignore[operator]
    limits = load_hard_limits(path).limits
    shadowed = PromotionGate(ledger, limits=limits, run_id="run_test")

    waiting = shadowed.evaluate(strategy_id=STRATEGY, evidence=promotable(), at=AS_OF, record=False)
    assert waiting.decision is Decision.AWAIT_SHADOW
    assert waiting.n_failed == 0

    served = shadowed.evaluate(
        strategy_id=STRATEGY,
        evidence=promotable(shadow_sessions_served=30, shadow_trades_served=25),
        at=AS_OF,
        record=False,
    )
    assert served.decision is Decision.PROMOTE


# --------------------------------------------------------------------------
# Recording
# --------------------------------------------------------------------------


def test_a_promotion_writes_the_status_the_event_and_the_row(
    gate: PromotionGate, registry: SpecRegistry, ledger: Ledger, registered: tuple[str, str]
) -> None:
    strategy_id, lineage_id = registered
    decision = gate.evaluate(
        strategy_id=strategy_id,
        evidence=promotable(lineage_id=lineage_id),
        at=AS_OF,
        record=True,
    )
    assert decision.decision is Decision.PROMOTE

    record = registry.status_of(strategy_id)
    assert record is not None
    assert record.status is StrategyStatus.PROMOTED
    assert record.promoted_at == AS_OF
    assert record.rung == 0

    row = ledger.conn.execute("SELECT * FROM promotions").fetchone()
    assert row["decision"] == "promote"
    assert row["n_failed"] == 0

    events = ledger.conn.execute(
        "SELECT COUNT(*) FROM event_log WHERE event_type = ?",
        (EventType.PROMOTION_EVALUATED.value,),
    ).fetchone()
    assert events[0] == 1


def test_a_refusal_is_recorded_with_every_gates_verdict(
    gate: PromotionGate, ledger: Ledger, registered: tuple[str, str]
) -> None:
    """Including the ones that passed. A record of only the failures cannot
    distinguish a near miss from a hopeless candidate."""
    strategy_id, lineage_id = registered
    gate.evaluate(
        strategy_id=strategy_id,
        evidence=promotable(lineage_id=lineage_id, pbo=a_pbo(value=0.9)),
        at=AS_OF,
        record=True,
    )
    stored = gate.latest(strategy_id)
    assert stored is not None
    assert stored.decision is Decision.REFUSE
    assert len(stored.gates) > 5
    assert any(g.passed for g in stored.gates)
    assert any(not g.passed for g in stored.gates)


def test_a_refused_candidate_is_not_promoted(
    gate: PromotionGate, registry: SpecRegistry, registered: tuple[str, str]
) -> None:
    strategy_id, lineage_id = registered
    gate.evaluate(
        strategy_id=strategy_id,
        evidence=promotable(lineage_id=lineage_id, holdout=a_holdout(passed=False)),
        at=AS_OF,
        record=True,
    )
    record = registry.status_of(strategy_id)
    assert record is not None
    assert record.status is not StrategyStatus.PROMOTED


def test_a_candidate_missing_only_the_holdout_is_queued_for_it(
    gate: PromotionGate, registry: SpecRegistry, registered: tuple[str, str]
) -> None:
    """The one thing `AWAITING_HOLDOUT` means, and the only way it is written.

    Deflation is `None` alongside the holdout, because the deflated figures are
    computed from the out-of-sample returns: a candidate with no holdout has no
    out-of-sample Sharpe to deflate. The first draft of the gate compared a
    failure *count* of one and this test found that a missing holdout fails
    five checks at once, which is why the holdout-derived set is named.
    """
    strategy_id, lineage_id = registered
    gate.evaluate(
        strategy_id=strategy_id,
        evidence=promotable(lineage_id=lineage_id, holdout=None, deflation=None),
        at=AS_OF,
        record=True,
    )
    record = registry.status_of(strategy_id)
    assert record is not None
    assert record.status is StrategyStatus.AWAITING_HOLDOUT


def test_a_failed_holdout_is_refused_rather_than_queued(
    gate: PromotionGate, registry: SpecRegistry, registered: tuple[str, str]
) -> None:
    """Failing the holdout is terminal. Treating it as 'waiting' would invite
    exactly the retry the uniqueness constraint exists to prevent."""
    strategy_id, lineage_id = registered
    gate.evaluate(
        strategy_id=strategy_id,
        evidence=promotable(lineage_id=lineage_id, holdout=a_holdout(passed=False)),
        at=AS_OF,
        record=True,
    )
    record = registry.status_of(strategy_id)
    assert record is not None
    assert record.status is StrategyStatus.CANDIDATE


def test_a_candidate_failing_more_than_the_holdout_is_not_queued(
    gate: PromotionGate, registry: SpecRegistry, registered: tuple[str, str]
) -> None:
    """Spending its single evaluation would waste the only independent evidence
    it will ever get."""
    strategy_id, lineage_id = registered
    gate.evaluate(
        strategy_id=strategy_id,
        evidence=promotable(
            lineage_id=lineage_id, holdout=None, deflation=None, pbo=a_pbo(value=0.9)
        ),
        at=AS_OF,
        record=True,
    )
    record = registry.status_of(strategy_id)
    assert record is not None
    assert record.status is StrategyStatus.CANDIDATE


def test_record_false_changes_no_arithmetic(
    gate: PromotionGate, ledger: Ledger, registered: tuple[str, str]
) -> None:
    """The release gate evaluates a thousand candidates unrecorded, so this
    must be the same function producing the same verdicts."""
    strategy_id, lineage_id = registered
    evidence = promotable(lineage_id=lineage_id)
    quiet = gate.evaluate(strategy_id=strategy_id, evidence=evidence, at=AS_OF, record=False)
    loud = gate.evaluate(strategy_id=strategy_id, evidence=evidence, at=AS_OF, record=True)
    assert quiet.decision is loud.decision
    assert [g.as_dict() for g in quiet.gates] == [g.as_dict() for g in loud.gates]

    rows = ledger.conn.execute("SELECT COUNT(*) FROM promotions").fetchone()
    assert rows[0] == 1


def test_caveats_ride_into_the_decision(gate: PromotionGate) -> None:
    """A survivorship-unmeasured vintage does not fail the gate — on free data
    it cannot be anything else — but a promotion made on it should say so."""
    decision = run(
        gate,
        promotable(
            vintage_caveats=("survivorship: unmeasured",),
            multiplicity=Multiplicity(
                n_trials=40,
                n_lineage_trials=1,
                n_search_trials=40,
                sharpe_dispersion=1.0,
                dispersion_measured=False,
                n_measurable=0,
            ),
        ),
    )
    assert any("survivorship" in note for note in decision.caveats)  # type: ignore[attr-defined]
    assert any("selection universe" in note for note in decision.caveats)  # type: ignore[attr-defined]


def test_the_gate_notional_is_not_the_floor(pinned: PinnedLimits) -> None:
    """A strategy promoted at floor size ratchets up, so gating at the smallest
    size it will ever trade would admit one that stops clearing as it grows."""
    assert pinned.limits.capital.floor_notional_ccy < GATE_NOTIONAL_CCY


# --------------------------------------------------------------------------
# The calibration lookup
# --------------------------------------------------------------------------


def test_no_calibration_reports_none_rather_than_a_default(ledger: Ledger) -> None:
    status = latest_calibration(ledger, at=AS_OF)
    assert status == CalibrationStatus(passed=None, age_days=None)


def test_the_most_recent_calibration_wins_even_when_it_failed(ledger: Ledger) -> None:
    """An engine that just failed its honesty check must not be treated as
    fine because an older passing run exists."""
    for index, (passed, days_ago) in enumerate(((1, 10), (0, 1))):
        ledger.conn.execute(
            "INSERT INTO backtest_calibrations (calibration_id, vintage_id, resolution, "
            "n_strategies, n_runs, rng_seed, tolerance, passed, ran_at, "
            "completing_event_seq) VALUES (?, 'v', 'daily', 1, 1, 0, 0.1, ?, ?, 1)",
            (f"cal_{index}", passed, (AS_OF - timedelta(days=days_ago)).isoformat()),
        )
    ledger.conn.commit()

    status = latest_calibration(ledger, at=AS_OF)
    assert status.passed is False
    assert status.age_days is not None
    assert status.age_days == pytest.approx(1.0)


def test_the_evidence_builder_reads_the_calibration_and_the_budget(
    ledger: Ledger, registry: SpecRegistry
) -> None:
    registered = registry.register(a_spec(), author_kind=AuthorKind.SEARCH, at=AS_OF)
    registry.charge(registered.lineage_id, loss_ccy=Decimal("10"), at=AS_OF)
    ledger.conn.execute(
        "INSERT INTO backtest_calibrations (calibration_id, vintage_id, resolution, "
        "n_strategies, n_runs, rng_seed, tolerance, passed, ran_at, completing_event_seq) "
        "VALUES ('cal_1', 'v', 'daily', 3, 9, 0, 0.1, 1, ?, 1)",
        ((AS_OF - timedelta(days=2)).isoformat(),),
    )
    ledger.conn.commit()

    builder = EvidenceBuilder(ledger=ledger, registry=registry)
    evidence = builder.build(
        spec=a_spec(),
        lineage_id=registered.lineage_id,
        holdout=a_holdout(),
        multiplicity=None,
        deflation=a_deflation(),
        pbo=a_pbo(),
        vintage_id="vint_1",
        vintage_admissible=True,
        feed_noise_p95_bps=Decimal("40"),
        at=AS_OF,
    )
    assert evidence.calibration_passed is True
    assert evidence.calibration_age_days == pytest.approx(2.0)
    assert evidence.budget is not None
    assert evidence.budget.consumed_ccy == Decimal("10")
