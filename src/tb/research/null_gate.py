"""The false-promotion measurement: how many specs with no edge get funded.

One function, and it is the most important measurement in the system.
`paper_shadow_sessions` is 0, so nothing stands between a generated strategy and
real money except `PromotionGate` — and the only way to know whether that gate
holds is to point a population with **no edge by construction** at it and count.

It lives here rather than in a test because two callers need exactly the same
arithmetic: the release-gate test in the suite, and `tb research null-gate`,
which an operator runs after editing a threshold. Two implementations would
drift, and the one that drifted would be the one nobody ran.

**What makes this a fair test rather than a straw man**, restated because it is
easy to erode:

* The specs come from the real grammar, the real backtester, the real deflation
  and PBO, and the real gate. Nothing here models the pipeline.
* The data is a random walk with zero drift, so any Sharpe is sampling noise.
  Measuring "no edge" on real prices would confound the gate's behaviour with
  whatever the decade did.
* The holdout is handed a **pass**. The gate's own checks must refuse this
  population; a pre-filter here would make the result about the filter.
* `n_reached_statistics` is reported alongside the rate, because a rate of zero
  from a population that never produced a measurable number says nothing. A
  caller that reads only the rate is reading half the result.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from datetime import datetime, timedelta
from decimal import Decimal
from math import sqrt

from tb.backtest.costs import CostModel, Jurisdiction
from tb.backtest.engine import Backtester, BacktestResult, InstrumentMeta
from tb.backtest.metrics import PERIODS_PER_YEAR_DAILY, returns_of
from tb.config.hard_limits import HardLimits
from tb.data.asof import ForwardOnlyReader, InMemoryBarSource
from tb.data.provider import Bar, Resolution
from tb.registry.models import AuthorKind, TrialOutcome
from tb.registry.promotion import Decision, PromotionEvidence, PromotionGate
from tb.research.holdout import HoldoutResult
from tb.research.nulls import decision_schedule, population, random_walk_bars
from tb.research.selection import (
    Deflation,
    PboResult,
    deflate,
    probability_of_backtest_overfitting,
)
from tb.research.trials import Multiplicity, Trial, returns_matrix
from tb.strategy.dsl.ops import DslStrategy, pipeline_from_spec
from tb.strategy.dsl.schema import StrategySpec

# The instrument the fixture prices. Not a real ISIN, and deliberately
# recognisable: a figure from this population must never be mistaken for one
# measured on a real name.
NULL_UID = "isin:NULLPOPULATION"

# Share of the fixture held back as the out-of-sample window, matching
# `DEFAULT_HOLDOUT_FRACTION`. The same split the real pipeline uses, so the
# out-of-sample statistics here have the same shape as the ones the gate will
# actually read.
TRAIN_FRACTION = 0.75

# Bounded so each decision's window is the lookback rather than all history.
# Without it the walk is quadratic in the number of bars — a real cost in the
# pipeline, not only here.
WALK_LOOKBACK = timedelta(days=150)


@dataclass(frozen=True, slots=True)
class NullGateResult:
    """What a null population did to the gate."""

    n_specs: int
    n_promoted: int
    n_traded: int
    n_reached_statistics: int
    multiplicity: Multiplicity
    pbo: PboResult | None
    refusals_by_gate: dict[str, int]

    @property
    def rate(self) -> float:
        if self.n_specs == 0:
            return 0.0
        return self.n_promoted / self.n_specs

    @property
    def is_informative(self) -> bool:
        """Whether the rate says anything about the gate.

        A rate of zero from a population that never produced a measurable
        statistic is a fact about the fixture. The threshold is a tenth,
        because below that the population is dominated by specs that never
        traded and the statistical checks were never reached.
        """
        return self.n_reached_statistics >= max(2, self.n_specs // 10)

    def summary(self) -> str:
        pbo = "unmeasurable" if self.pbo is None else f"{self.pbo.pbo:.3f}"
        return (
            f"{self.n_promoted} of {self.n_specs} random specs promoted "
            f"({self.rate:.2%}); {self.n_traded} traded, "
            f"{self.n_reached_statistics} reached the statistical checks with "
            f"measured numbers; PBO {pbo}, haircut from "
            f"{self.multiplicity.n_trials} trials at a Sharpe dispersion of "
            f"{self.multiplicity.sharpe_dispersion:.2f}"
        )


def measure_false_promotion_rate(
    *,
    gate: PromotionGate,
    limits: HardLimits,
    n_specs: int = 1_000,
    days: int = 400,
    seed: int = 20260601,
) -> NullGateResult:
    """Run a null population through the real gate and count the promotions.

    Never records a promotion event: the population is a measurement, not a set
    of decisions about strategies that exist. `record=False` changes no
    arithmetic — the same function computes the same verdicts either way, which
    `test_record_false_changes_no_arithmetic` asserts directly.
    """
    if n_specs < 2:
        raise ValueError(f"need at least 2 specs to measure a rate, got {n_specs}")

    bars = random_walk_bars(instrument_uid=NULL_UID, days=days, seed=seed)
    schedule = decision_schedule(bars)
    split = int(len(schedule) * TRAIN_FRACTION)
    train_times, holdout_times = schedule[:split], schedule[split:]
    if len(train_times) < 2 or len(holdout_times) < 2:
        raise ValueError(
            f"{days} bars cannot be split into a train and a holdout window that each "
            "support a backtest; a fill comes from the bar after the decision, so each "
            "side needs at least two"
        )

    costs = CostModel(limits)
    specs = population(n_specs, seed=seed)

    def walk(spec: StrategySpec, times: Sequence[datetime]) -> BacktestResult:
        engine = Backtester(
            cost_model=costs,
            pipeline=pipeline_from_spec(spec),
            instruments={NULL_UID: InstrumentMeta(NULL_UID, "USD", Jurisdiction.US)},
        )
        return engine.run(
            strategy=DslStrategy(spec=spec, strategy_id=f"stg_{spec.spec_hash[:12]}"),
            reader=ForwardOnlyReader(
                source=InMemoryBarSource(bars=bars),
                resolution=Resolution.DAILY,
                instrument_uids=(NULL_UID,),
                lookback=WALK_LOOKBACK,
            ),
            decision_times=times,
        )

    train = {spec.spec_hash: walk(spec, train_times) for spec in specs}
    # Only specs that traded in training get a holdout run, the same shape as
    # the real pipeline: the holdout is spent on candidates that got that far,
    # and a spec with no training trades has nothing to evaluate.
    holdouts = {
        spec.spec_hash: walk(spec, holdout_times)
        for spec in specs
        if train[spec.spec_hash].metrics.n_trades > 0
    }

    multiplicity = _multiplicity_of(train, n_specs=len(specs))
    pbo = probability_of_backtest_overfitting(
        returns_matrix(
            [
                _as_trial(spec_hash, result, bars, n_specs=len(specs))
                for spec_hash, result in train.items()
                if result.metrics.n_trades > 0
            ],
            seed=seed,
        )
    )

    promoted = 0
    reached = 0
    refusals: dict[str, int] = {}
    for spec in specs:
        result = holdouts.get(spec.spec_hash)
        record = None
        deflation: Deflation | None = None
        if result is not None:
            curve = _curve_returns(result)
            record = _as_holdout(spec, result, curve, bars)
            if record.net_sharpe is not None:
                deflation = deflate(
                    observed_sharpe=record.net_sharpe,
                    returns=curve,
                    n_trials=multiplicity.n_trials,
                    sharpe_dispersion=multiplicity.sharpe_dispersion,
                    periods_per_year=PERIODS_PER_YEAR_DAILY,
                    dispersion_measured=multiplicity.dispersion_measured,
                )
                if deflation.deflated_probability is not None:
                    reached += 1

        decision = gate.evaluate(
            strategy_id=f"stg_{spec.spec_hash[:12]}",
            evidence=PromotionEvidence(
                spec=spec,
                lineage_id=f"lin_{spec.spec_hash[:10]}",
                holdout=record,
                multiplicity=multiplicity,
                deflation=deflation,
                pbo=pbo,
                vintage_id="vint_null_gate",
                vintage_admissible=True,
                calibration_passed=True,
                calibration_age_days=0.0,
                feed_noise_p95_bps=Decimal("40"),
            ),
            record=False,
        )
        if decision.decision is Decision.PROMOTE:
            promoted += 1
        for failure in decision.failures:
            refusals[failure.name] = refusals.get(failure.name, 0) + 1

    return NullGateResult(
        n_specs=len(specs),
        n_promoted=promoted,
        n_traded=len(holdouts),
        n_reached_statistics=reached,
        multiplicity=multiplicity,
        pbo=pbo,
        refusals_by_gate=refusals,
    )


# --------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------


def _curve_returns(result: BacktestResult) -> list[float]:
    return [float(value) for value in returns_of([point.equity_ccy for point in result.curve])]


def _multiplicity_of(train: dict[str, BacktestResult], *, n_specs: int) -> Multiplicity:
    """The haircut this whole search implies.

    The dispersion comes from the *training* Sharpes, which is the population a
    real lineage would have. It is measured rather than defaulted whenever two
    or more specs produced one, because the fallback exists for the case where
    there is nothing to measure — using it while real data sits in the table
    would be a needlessly coarse haircut.
    """
    sharpes = [
        result.metrics.net_sharpe
        for result in train.values()
        if result.metrics.net_sharpe is not None
    ]
    if len(sharpes) < 2:
        return Multiplicity(
            n_trials=n_specs,
            n_lineage_trials=1,
            n_search_trials=n_specs,
            sharpe_dispersion=1.0,
            dispersion_measured=False,
            n_measurable=len(sharpes),
        )
    mean = sum(sharpes) / len(sharpes)
    dispersion = sqrt(sum((value - mean) ** 2 for value in sharpes) / (len(sharpes) - 1))
    return Multiplicity(
        n_trials=n_specs,
        n_lineage_trials=1,
        n_search_trials=n_specs,
        sharpe_dispersion=dispersion if dispersion > 0 else 1.0,
        dispersion_measured=dispersion > 0,
        n_measurable=len(sharpes),
    )


def _as_trial(spec_hash: str, result: BacktestResult, bars: list[Bar], *, n_specs: int) -> Trial:
    return Trial(
        trial_id=spec_hash[:10],
        search_id="srch_null_gate",
        lineage_id="lin_null_gate",
        spec_hash=spec_hash,
        author_kind=AuthorKind.SEARCH,
        outcome=TrialOutcome.EVALUATED,
        recorded_at=bars[-1].available_at_utc,
        net_sharpe=result.metrics.net_sharpe,
        returns=tuple(_curve_returns(result)),
        trials_in_lineage_at_time=1,
        trials_in_search_at_time=n_specs,
    )


def _as_holdout(
    spec: StrategySpec, result: BacktestResult, curve: list[float], bars: list[Bar]
) -> HoldoutResult:
    """The holdout record a null candidate's backtest implies.

    `passed=True` deliberately. The gate's own checks are what must refuse this
    population; failing the holdout here would make the measurement about this
    function rather than about the gate, and handing it an easy pass is the
    harder test.
    """
    at = bars[-1].available_at_utc
    return HoldoutResult(
        evaluation_id=f"hold_{spec.spec_hash[:10]}",
        strategy_id=f"stg_{spec.spec_hash[:12]}",
        version=1,
        lineage_id=f"lin_{spec.spec_hash[:10]}",
        spec_hash=spec.spec_hash,
        vintage_id="vint_null_gate",
        sealed_from=at,
        passed=True,
        evaluated_at=at,
        n_trades=result.metrics.n_trades,
        net_sharpe=result.metrics.net_sharpe,
        net_return_pct=float(result.metrics.net_return_pct),
        max_drawdown_pct=float(result.metrics.max_drawdown_pct),
        cost_drag_bps=float(result.metrics.cost_drag_bps),
        returns=tuple(curve),
    )
