"""Deflating a Sharpe by the size of the search that found it."""

from __future__ import annotations

import random
from itertools import pairwise
from math import sqrt

import pytest

from tb.backtest.metrics import PERIODS_PER_YEAR_DAILY
from tb.research.selection import (
    DEFAULT_CSCV_SPLITS,
    MIN_PERIODS_PER_HALF,
    SelectionError,
    deflate,
    expected_max_sharpe,
    probability_of_backtest_overfitting,
)


def noise(n: int, *, seed: int, drift: float = 0.0, sigma: float = 0.01) -> list[float]:
    rng = random.Random(seed)
    return [rng.gauss(drift, sigma) for _ in range(n)]


def annualised_sharpe(values: list[float]) -> float:
    mean = sum(values) / len(values)
    variance = sum((v - mean) ** 2 for v in values) / (len(values) - 1)
    return mean / sqrt(variance) * sqrt(PERIODS_PER_YEAR_DAILY)


# --------------------------------------------------------------------------
# The expected maximum
# --------------------------------------------------------------------------


def test_the_expected_maximum_grows_with_the_number_of_trials() -> None:
    values = [
        expected_max_sharpe(n_trials=n, sharpe_dispersion=1.0) for n in (2, 10, 100, 1_000, 10_000)
    ]
    assert values == sorted(values)
    assert all(a < b for a, b in pairwise(values))


def test_a_single_trial_takes_no_haircut() -> None:
    """One draw has no maximum to take, and `Z(1 - 1/1)` is minus infinity —
    so this is the case that would raise if it were not handled."""
    assert expected_max_sharpe(n_trials=1, sharpe_dispersion=1.0) == 0.0
    assert expected_max_sharpe(n_trials=0, sharpe_dispersion=1.0) == 0.0


def test_a_zero_dispersion_gives_no_haircut() -> None:
    """Correct arithmetic, and the reason `trials.py` never passes zero for
    'unmeasured': there would be no haircut at all."""
    assert expected_max_sharpe(n_trials=1_000, sharpe_dispersion=0.0) == 0.0


def test_the_haircut_scales_with_the_dispersion() -> None:
    one = expected_max_sharpe(n_trials=100, sharpe_dispersion=1.0)
    two = expected_max_sharpe(n_trials=100, sharpe_dispersion=2.0)
    assert two == pytest.approx(2 * one)


def test_a_thousand_trials_needs_about_three_sharpe_to_clear_noise() -> None:
    """The number that makes the whole gate meaningful.

    At a trial dispersion of 1.0, a search of a thousand produces a best
    Sharpe over 3 from noise alone. A raw OOS Sharpe of 2 out of such a search
    is not evidence of anything, which is why the deflated figure is what the
    gate reads.
    """
    haircut = expected_max_sharpe(n_trials=1_000, sharpe_dispersion=1.0)
    assert 3.0 < haircut < 3.5


# --------------------------------------------------------------------------
# Deflation
# --------------------------------------------------------------------------


def test_a_noise_strategy_deflates_to_nothing() -> None:
    returns = noise(500, seed=7)
    result = deflate(
        observed_sharpe=annualised_sharpe(returns),
        returns=returns,
        n_trials=1_000,
        sharpe_dispersion=1.0,
        periods_per_year=PERIODS_PER_YEAR_DAILY,
    )
    assert result.deflated_sharpe < 0
    assert result.deflated_probability is not None
    assert result.deflated_probability < 0.05


def test_a_real_edge_survives_a_small_search_and_not_a_large_one() -> None:
    """The whole point, in one test. The same backtest is evidence out of ten
    trials and is not out of a thousand.

    Five thousand periods rather than five hundred so the realised Sharpe sits
    close to the drift it was drawn from: at 500 the standard error of an
    annualised Sharpe is about 0.7, which is wide enough that the test would be
    about the seed rather than about the arithmetic. The bracketing assertion
    below fails loudly if a change to the generator moves it anyway.
    """
    returns = noise(5_000, seed=7, drift=0.0015)
    observed = annualised_sharpe(returns)

    small_haircut = expected_max_sharpe(n_trials=10, sharpe_dispersion=1.0)
    large_haircut = expected_max_sharpe(n_trials=1_000, sharpe_dispersion=1.0)
    assert small_haircut < observed < large_haircut, (
        f"the sample's Sharpe ({observed:.3f}) must sit between the two haircuts "
        f"({small_haircut:.3f}, {large_haircut:.3f}) for this test to be about "
        "deflation rather than about the draw"
    )

    small = deflate(
        observed_sharpe=observed,
        returns=returns,
        n_trials=10,
        sharpe_dispersion=1.0,
        periods_per_year=PERIODS_PER_YEAR_DAILY,
    )
    large = deflate(
        observed_sharpe=observed,
        returns=returns,
        n_trials=1_000,
        sharpe_dispersion=1.0,
        periods_per_year=PERIODS_PER_YEAR_DAILY,
    )
    assert small.deflated_sharpe > 0
    assert large.deflated_sharpe < 0
    assert small.deflated_probability is not None
    assert large.deflated_probability is not None
    assert small.deflated_probability > large.deflated_probability


def test_the_probability_reflects_sample_length_where_the_level_cannot() -> None:
    """What the level cannot see: sixty observations and five hundred can give
    the same Sharpe and very different confidence in it.

    The direction depends on the sign of the gap, and stating that is the point.
    The probability is `P(true Sharpe > the haircut benchmark)`, not "confidence
    in the Sharpe" — so more data moves it *away* from 0.5 in whichever
    direction the gap points. Here the gap is positive, so the long sample is
    the more confident one; with a negative gap the long sample would be the
    lower number, and a test that assumed otherwise would be asserting a
    property this statistic does not have.
    """
    long_run = noise(500, seed=3, drift=0.0008)
    short_run = long_run[:60]
    observed = annualised_sharpe(long_run)
    assert observed > 0, "this test needs a positive gap to be about anything"

    def probability(returns: list[float]) -> float:
        result = deflate(
            observed_sharpe=observed,
            returns=returns,
            # One trial, so the benchmark is zero and the gap is the Sharpe
            # itself: the test is about sample length, not about the haircut.
            n_trials=1,
            sharpe_dispersion=1.0,
            periods_per_year=PERIODS_PER_YEAR_DAILY,
        )
        assert result.deflated_probability is not None
        return result.deflated_probability

    assert probability(short_run) < probability(long_run)


def test_too_short_a_sample_has_no_probability_rather_than_a_low_one() -> None:
    """`None` reads as 'not measured' at the gate, which refuses. Zero would
    read as 'measured, no confidence', which is a different claim."""
    result = deflate(
        observed_sharpe=1.5,
        returns=[0.01, 0.02, -0.01],
        n_trials=5,
        sharpe_dispersion=1.0,
        periods_per_year=PERIODS_PER_YEAR_DAILY,
    )
    assert result.deflated_probability is None
    assert not result.is_measured


def test_deflation_reports_the_moments_it_used() -> None:
    returns = noise(400, seed=5)
    result = deflate(
        observed_sharpe=0.5,
        returns=returns,
        n_trials=50,
        sharpe_dispersion=1.0,
        periods_per_year=PERIODS_PER_YEAR_DAILY,
    )
    assert result.n_periods == 400
    # Non-excess kurtosis: about 3 for a normal sample, never near 0.
    assert 2.0 < result.kurtosis < 4.5
    assert abs(result.skew) < 1.0
    assert "expected maximum" in result.summary()


def test_the_summary_says_unmeasurable_rather_than_printing_a_number() -> None:
    result = deflate(
        observed_sharpe=1.0,
        returns=[0.01, 0.01, 0.01],
        n_trials=2,
        sharpe_dispersion=1.0,
        periods_per_year=PERIODS_PER_YEAR_DAILY,
    )
    assert "unmeasurable" in result.summary()


def test_an_annualised_input_is_not_silently_used_as_a_per_period_sharpe() -> None:
    """The units bug this module is built to avoid.

    A probability computed from an annualised Sharpe as if it were per-period
    would be essentially 1.0 for any positive Sharpe, because the statistic
    scales by sqrt(T-1). A correct computation on a Sharpe of 0.5 annualised
    over 500 daily periods against a benchmark of 0 sits well short of
    certainty, and that gap is the check.
    """
    returns = noise(500, seed=9, drift=0.0002)
    result = deflate(
        observed_sharpe=0.5,
        returns=returns,
        n_trials=1,
        sharpe_dispersion=1.0,
        periods_per_year=PERIODS_PER_YEAR_DAILY,
    )
    assert result.deflated_probability is not None
    assert 0.5 < result.deflated_probability < 0.95


# --------------------------------------------------------------------------
# PBO
# --------------------------------------------------------------------------


def matrix_of(columns: list[list[float]]) -> list[list[float]]:
    """Transpose a list of per-trial series into period rows."""
    return [list(row) for row in zip(*columns, strict=True)]


def test_pbo_of_pure_noise_is_about_a_half() -> None:
    """The calibration. Selecting the best of N noise series on one half tells
    you nothing about the other half, so the winner lands either side of the
    median with equal probability."""
    columns = [noise(400, seed=100 + i) for i in range(20)]
    result = probability_of_backtest_overfitting(matrix_of(columns))
    assert result is not None
    assert 0.35 < result.pbo < 0.65
    assert result.n_combinations == 70
    assert result.n_splits == DEFAULT_CSCV_SPLITS


def test_pbo_of_a_genuinely_persistent_winner_is_zero() -> None:
    columns = [noise(400, seed=200 + i) for i in range(20)]
    columns[3] = noise(400, seed=999, drift=0.004)
    result = probability_of_backtest_overfitting(matrix_of(columns))
    assert result is not None
    assert result.pbo == 0.0
    assert result.median_logit > 0


def test_pbo_of_a_pure_selection_artefact_is_high() -> None:
    """Columns engineered so whichever wins in-sample loses out of sample:
    each is good on even blocks and bad on odd ones, or the reverse. Selecting
    the in-sample best therefore picks a column that inverts on the test half.
    """
    periods, n_trials, n_splits = 400, 20, DEFAULT_CSCV_SPLITS
    per_split = periods // n_splits
    rng = random.Random(3)
    columns: list[list[float]] = []
    for column in range(n_trials):
        sign = 1 if column % 2 == 0 else -1
        series: list[float] = []
        for index in range(periods):
            block = min(index // per_split, n_splits - 1)
            flip = 1 if block % 2 == 0 else -1
            series.append(rng.gauss(sign * flip * 0.01, 0.01))
        columns.append(series)
    result = probability_of_backtest_overfitting(matrix_of(columns))
    assert result is not None
    assert result.pbo > 0.6
    assert result.median_logit < 0
    assert "PBO" in result.summary


def test_pbo_is_none_with_a_single_trial() -> None:
    """Nothing to select among, so overfitting by selection is not defined.
    `None` is a refusal at the gate, which is the right direction."""
    assert probability_of_backtest_overfitting([[0.01]] * 400) is None


def test_pbo_is_none_on_too_short_a_sample() -> None:
    """The case that found the bound was too loose.

    Twenty periods over eight splits leaves eight per half. That cleared the
    original bound of 6 and produced a confident-looking 0.56 from Sharpes over
    eight observations — a number the gate would have believed.
    """
    columns = [noise(20, seed=i) for i in range(5)]
    assert probability_of_backtest_overfitting(matrix_of(columns)) is None


def test_pbo_is_none_on_an_empty_matrix() -> None:
    assert probability_of_backtest_overfitting([]) is None


def test_pbo_refuses_a_ragged_matrix() -> None:
    """Otherwise one trial's Sharpe over one window is compared against
    another's over a different window."""
    with pytest.raises(SelectionError, match="ragged"):
        probability_of_backtest_overfitting([[0.1, 0.2], [0.3]])


def test_pbo_refuses_an_odd_split_count() -> None:
    columns = [noise(400, seed=i) for i in range(5)]
    with pytest.raises(SelectionError, match="even"):
        probability_of_backtest_overfitting(matrix_of(columns), n_splits=7)


def test_pbo_keeps_the_trailing_periods() -> None:
    """A window that does not divide evenly must not be silently shortened."""
    columns = [noise(405, seed=300 + i) for i in range(6)]
    result = probability_of_backtest_overfitting(matrix_of(columns))
    assert result is not None
    assert result.n_periods == 405


def test_the_minimum_half_is_enforced() -> None:
    """Just above and just below the bound, so the constant is load-bearing
    rather than decorative."""
    per_half = MIN_PERIODS_PER_HALF
    too_short = per_half * 2 - 1
    long_enough = (per_half + 1) * 2
    few = [noise(too_short, seed=400 + i) for i in range(4)]
    enough = [noise(long_enough * 4, seed=500 + i) for i in range(4)]
    assert probability_of_backtest_overfitting(matrix_of(few), n_splits=2) is None
    assert probability_of_backtest_overfitting(matrix_of(enough), n_splits=2) is not None
