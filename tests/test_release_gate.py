"""The release gate: 1,000 random specs must promote at or below the ceiling.

This is the single number that says whether autonomous promotion is safe.
`paper_shadow_sessions` is 0, so nothing stands between a generated strategy and
real money except `PromotionGate` — and the only way to know whether that gate
holds is to point a population with **no edge by construction** at it and count
how many get through.

The measurement itself lives in `tb.research.null_gate`, not here, because
`tb research null-gate` runs exactly the same arithmetic: the thresholds it
depends on are in a file a human edits, so an operator has to be able to ask
what an edit did without reading the test suite. Two implementations would
drift, and the one that drifted would be the one nobody ran.

**The vacuity checks matter as much as the ceiling.** A gate that refused
everything for a reason unrelated to the statistics — every spec making zero
trades, or every candidate missing a vintage — would pass the ceiling while
proving nothing. So this asserts that a substantial population reached the
statistical checks with real measured numbers, that the refusals were
statistical rather than procedural, and separately that a strategy with a
genuine edge still promotes. A gate that says no to everything is not a gate,
it is a switch in the off position.
"""

from __future__ import annotations

import random
from datetime import UTC, datetime
from decimal import Decimal
from itertools import pairwise
from math import sqrt

import pytest

from tb.backtest.metrics import PERIODS_PER_YEAR_DAILY
from tb.config.loader import PinnedLimits, load_hard_limits
from tb.data.provider import Bar
from tb.ledger.store import Ledger
from tb.registry.promotion import Decision, PromotionEvidence, PromotionGate
from tb.research.holdout import HoldoutResult
from tb.research.null_gate import (
    NULL_UID,
    NullGateResult,
    measure_false_promotion_rate,
)
from tb.research.nulls import population, random_walk_bars
from tb.research.selection import PboResult, deflate
from tb.research.trials import Multiplicity
from tb.strategy.dsl.schema import StrategySpec
from tests.conftest import REFERENCE_LIMITS

# The population size the plan names. Large enough that a 1% ceiling is a
# meaningful count rather than a rounding artefact.
N_SPECS = 1_000
N_BARS = 400

AS_OF = datetime(2026, 6, 1, 14, 0, tzinfo=UTC)


@pytest.fixture(scope="module")
def null_result(tmp_path_factory: pytest.TempPathFactory) -> NullGateResult:
    """Run the whole population through the real gate, once.

    Module-scoped because it is the expensive part — about a hundred backtests
    a second — and three tests read the same result.
    """
    limits = load_hard_limits(REFERENCE_LIMITS).limits
    path = tmp_path_factory.mktemp("release_gate") / "ledger.db"
    with Ledger(path) as ledger:
        ledger.initialise(created_by="release-gate")
        gate = PromotionGate(ledger, limits=limits)
        return measure_false_promotion_rate(gate=gate, limits=limits, n_specs=N_SPECS, days=N_BARS)


@pytest.fixture
def gate(ledger: Ledger, pinned: PinnedLimits) -> PromotionGate:
    return PromotionGate(ledger, limits=pinned.limits, run_id="run_release_gate")


# --------------------------------------------------------------------------
# The gate
# --------------------------------------------------------------------------


@pytest.mark.slow
def test_the_null_population_promotes_at_or_below_the_ceiling(
    null_result: NullGateResult, pinned: PinnedLimits
) -> None:
    """The number the whole milestone exists to produce.

    A 3% false-promotion rate against a searcher proposing a thousand specs a
    week puts thirty random strategies into production a week. The ceiling is
    1%, and the observed rate on this population should be far below it.
    """
    ceiling = pinned.limits.promotion.max_null_promotion_rate
    assert null_result.rate <= ceiling, (
        f"{null_result.summary()} — over the {ceiling:.1%} ceiling. With no "
        "paper-shadow period this rate is the number of noise strategies that "
        "reach real money."
    )


@pytest.mark.slow
def test_the_population_actually_reached_the_statistical_checks(
    null_result: NullGateResult,
) -> None:
    """The vacuity check, and it is not optional.

    A gate that refused everything because no spec ever traded would pass the
    ceiling while proving nothing about the statistics. This asserts that a
    substantial number of candidates arrived at the gate with *measured*
    deflation and PBO figures — so the ceiling above was tested against real
    numbers rather than against a population of `unmeasured`.
    """
    assert null_result.is_informative, null_result.summary()
    assert null_result.n_traded >= 100, (
        f"only {null_result.n_traded} of {null_result.n_specs} specs traded at all; "
        "the population is not exercising the gate"
    )
    assert null_result.n_reached_statistics >= 100, (
        f"only {null_result.n_reached_statistics} candidates produced a measurable "
        "deflated probability; the ceiling was tested against a population of "
        "'unmeasured' rather than against real statistics"
    )
    assert null_result.pbo is not None, "PBO was unmeasurable across the whole population"
    assert null_result.pbo.n_trials >= 16, "the PBO sample is too small to mean anything"
    assert null_result.multiplicity.dispersion_measured, (
        "the haircut fell back to its conservative default while a thousand real "
        "training Sharpes were available, which would make the ceiling easier to meet "
        "than the arithmetic intends"
    )


@pytest.mark.slow
def test_the_refusals_are_statistical_rather_than_procedural(
    null_result: NullGateResult,
) -> None:
    """Which gates actually did the refusing.

    If the whole population were refused by `sealed_vintage` or
    `engine_calibrated`, the ceiling would be measuring the fixture rather than
    the statistics. The statistical checks — deflation, PBO, trade count — must
    be the ones carrying it.
    """
    counts = null_result.refusals_by_gate
    assert counts.get("sealed_vintage", 0) == 0
    assert counts.get("engine_calibrated", 0) == 0
    statistical = sum(
        counts.get(name, 0)
        for name in ("deflated_sharpe", "deflated_probability", "pbo", "oos_trades")
    )
    assert statistical > 0, f"no statistical refusals at all; failures were {counts}"
    assert counts.get("deflated_sharpe", 0) >= null_result.n_reached_statistics // 2, (
        "the deflated Sharpe refused fewer candidates than reached it, which would mean "
        f"the multiplicity haircut is not binding: {counts}"
    )


# --------------------------------------------------------------------------
# The other half: a gate that says no to everything is not a gate
# --------------------------------------------------------------------------


def returns_with_sharpe(sharpe: float, *, n_periods: int = 500, seed: int = 3) -> list[float]:
    """A return series whose own annualised Sharpe really is `sharpe`.

    Worth the arithmetic rather than hand-writing a plausible-looking list. The
    deflated *probability* reads `n_periods`, the skew and the kurtosis from the
    series while reading the Sharpe from its argument, so a series that does not
    match its claimed Sharpe produces a confidence figure about a strategy that
    does not exist — which is how the first draft of this fixture asserted a
    promotable candidate the gate correctly refused.
    """
    rng = random.Random(seed)
    raw = [rng.gauss(0.0, 1.0) for _ in range(n_periods)]
    mean = sum(raw) / n_periods
    spread = sqrt(sum((value - mean) ** 2 for value in raw) / (n_periods - 1))
    target = sharpe / sqrt(PERIODS_PER_YEAR_DAILY)
    sigma = 0.01
    return [(value - mean) / spread * sigma + target * sigma for value in raw]


def a_strong_candidate(*, n_trials: int, sharpe: float, n_periods: int = 500) -> PromotionEvidence:
    """Evidence a genuinely good strategy would produce.

    Standalone rather than derived from the null population, so these tests cost
    nothing and run on every local `pytest`.

    500 periods is about two years of daily holdout, which is what it takes to
    be 95% confident about a Sharpe against a multiplicity benchmark. That is
    not a fixture convenience — it is the sample length the arithmetic demands,
    and a shorter holdout genuinely cannot support the claim.
    """
    spec = population(1, seed=11)[0]
    returns = returns_with_sharpe(sharpe, n_periods=n_periods)
    return PromotionEvidence(
        spec=spec,
        lineage_id="lin_genuine",
        holdout=HoldoutResult(
            evaluation_id="hold_genuine",
            strategy_id="stg_genuine",
            version=1,
            lineage_id="lin_genuine",
            spec_hash=spec.spec_hash,
            vintage_id="vint_null_fixture",
            sealed_from=AS_OF,
            passed=True,
            evaluated_at=AS_OF,
            n_trades=60,
            net_sharpe=sharpe,
            net_return_pct=30.0,
            max_drawdown_pct=6.0,
            returns=tuple(returns),
        ),
        multiplicity=Multiplicity(
            n_trials=n_trials,
            n_lineage_trials=n_trials,
            n_search_trials=n_trials,
            sharpe_dispersion=1.0,
            dispersion_measured=True,
            n_measurable=n_trials,
        ),
        deflation=deflate(
            observed_sharpe=sharpe,
            returns=returns,
            n_trials=n_trials,
            sharpe_dispersion=1.0,
            periods_per_year=PERIODS_PER_YEAR_DAILY,
        ),
        pbo=PboResult(
            pbo=0.05,
            n_combinations=70,
            n_trials=64,
            n_periods=300,
            n_splits=8,
            median_logit=2.0,
        ),
        vintage_id="vint_null_fixture",
        vintage_admissible=True,
        calibration_passed=True,
        calibration_age_days=1.0,
        feed_noise_p95_bps=Decimal("40"),
    )


def test_a_genuine_edge_still_promotes(gate: PromotionGate) -> None:
    """The gate is calibrated, not merely closed.

    A gate that refuses everything meets any ceiling and is worthless. This is
    the other half of the release gate: an out-of-sample Sharpe well clear of
    what the search would find by chance, on a sample long enough to be
    confident about, with a PBO showing the selection held up.
    """
    decision = gate.evaluate(
        strategy_id="stg_genuine",
        evidence=a_strong_candidate(n_trials=40, sharpe=4.0),
        at=AS_OF,
        record=False,
    )
    assert decision.decision is Decision.PROMOTE, (
        "a strategy with a genuine edge was refused, so the ceiling above is "
        "being met by a gate that says no to everything:\n" + decision.report()
    )


def test_the_search_size_decides_what_edge_is_provable(gate: PromotionGate) -> None:
    """The finding that should shape M6, asserted so it cannot be lost.

    The haircut is `dispersion x E[max of N]`, and at a trial dispersion of 1.0
    that is about 1.6 Sharpe for a search of ten and about 3.3 for a search of a
    thousand. So the *same* out-of-sample record — a Sharpe of 3.0, which is an
    excellent real result — is promotable out of a focused search and is not out
    of a thousand-spec sweep.

    That is the arithmetic working, not a threshold to loosen. Its consequence
    for the searcher is concrete: a thousand specs per lineage per window makes
    promotion effectively impossible, so M6 wants many small searches rather
    than one large one, and the trial count has to be a budget the searcher
    spends deliberately.
    """
    focused = gate.evaluate(
        strategy_id="stg_focused",
        evidence=a_strong_candidate(n_trials=10, sharpe=3.0),
        at=AS_OF,
        record=False,
    )
    sweeping = gate.evaluate(
        strategy_id="stg_sweeping",
        evidence=a_strong_candidate(n_trials=1_000, sharpe=3.0),
        at=AS_OF,
        record=False,
    )
    assert focused.decision is Decision.PROMOTE
    assert sweeping.decision is Decision.REFUSE
    assert "deflated_sharpe" in {result.name for result in sweeping.failures}


def test_the_null_population_is_drawn_from_the_real_grammar() -> None:
    """A population of deliberately silly specs would be rejected by the schema
    rather than by the gate, and would prove nothing about the gate."""
    specs = population(50, seed=1)
    assert len({spec.spec_hash for spec in specs}) == 50
    for spec in specs:
        # Re-validating through the untrusted-payload path: these are exactly
        # the trees a searcher or an LLM could produce.
        reparsed = StrategySpec.parse(spec.model_dump(mode="json"))
        assert reparsed.spec_hash == spec.spec_hash
        assert spec.required_features
        assert spec.min_holding_minutes >= 1440


def test_the_fixture_has_no_edge_by_construction() -> None:
    """A random walk with zero drift. Measuring 'no edge' on real prices would
    confound the gate's behaviour with whatever the decade did."""
    bars: list[Bar] = random_walk_bars(instrument_uid=NULL_UID, days=N_BARS, seed=20260601)
    closes = [float(bar.close) for bar in bars]
    steps = [second - first for first, second in pairwise(closes)]
    mean = sum(steps) / len(steps)
    spread = sqrt(sum((value - mean) ** 2 for value in steps) / (len(steps) - 1))
    # The drift must be small against its own standard error.
    assert abs(mean) < spread / sqrt(len(steps)) * 3


def test_a_population_of_one_cannot_measure_a_rate(ledger: Ledger) -> None:
    """A rate over a single spec is not a rate, and pretending otherwise would
    let a one-spec run report 0% and look like evidence."""
    limits = load_hard_limits(REFERENCE_LIMITS).limits
    with pytest.raises(ValueError, match="at least 2 specs"):
        measure_false_promotion_rate(
            gate=PromotionGate(ledger, limits=limits), limits=limits, n_specs=1
        )


def test_too_short_a_fixture_is_refused(ledger: Ledger) -> None:
    """Both halves need at least two decisions, since a fill comes from the bar
    after the decision."""
    limits = load_hard_limits(REFERENCE_LIMITS).limits
    with pytest.raises(ValueError, match="cannot be split"):
        measure_false_promotion_rate(
            gate=PromotionGate(ledger, limits=limits), limits=limits, n_specs=4, days=3
        )
