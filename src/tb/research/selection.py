"""Deflating a Sharpe ratio by the size of the search that found it.

A backtest Sharpe is an order statistic, not a measurement. Try a thousand
specs on the same data and the best of them has a good Sharpe *by
construction*, and the number says nothing about the strategy. This module
turns "how many did you try" into an arithmetic correction, three ways:

**The expected maximum.** Bailey and Lopez de Prado's approximation for the
best Sharpe a search of N independent trials produces from noise alone, given
the dispersion of the trials' Sharpes. It needs both the count and the
dispersion, which is why `tb.research.trials` stores every trial's Sharpe
rather than only a count.

**The deflated level.** The candidate's Sharpe minus that expected maximum. In
Sharpe units, so it can be compared against a Sharpe threshold. This is the
number `promotion.min_oos_deflated_sharpe` bounds.

**The deflated probability.** The probabilistic Sharpe ratio evaluated at the
expected maximum as its benchmark — the probability that the true Sharpe
exceeds what the search would have produced by chance. Unlike the level, it
accounts for sample length, skew and kurtosis, so it catches the case where the
level is comfortably positive on sixty observations with a fat left tail.

And separately, **PBO** — the probability of backtest overfitting, by
combinatorially symmetric cross-validation. Where deflation asks "is this
Sharpe bigger than the best noise would give", PBO asks a different and harder
question: "when I pick the best on one half of the data, does it stay good on
the other half?". A search can produce a high deflated Sharpe and a PBO of 0.6,
and that combination means the selection procedure is not selecting anything.

**Everything here returns `None` rather than a number when it cannot be
computed.** A DSR of 0.0 reads as "measured, no confidence"; `None` reads as
"not measured", and the promotion gate treats the second as a refusal. Zero
would have been the permissive direction, because a threshold comparison
against a fabricated zero is a comparison that can be passed by arranging for
the computation to fail.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from itertools import combinations
from math import e, log, sqrt
from statistics import NormalDist

from tb.core.errors import TbError

# Euler-Mascheroni, in the expected-maximum approximation.
EULER_MASCHERONI = 0.577_215_664_901_532_9

_NORMAL = NormalDist()

# How many disjoint sub-samples CSCV splits the period axis into. The original
# paper uses 16, giving C(16,8) = 12,870 combinations. 8 gives 70, which is
# where this sits: the estimate is coarser, and it is computed over every trial
# in a search that may hold a thousand of them, so the finer split would turn a
# release-gate test into a several-minute one. Passed rather than hardcoded at
# the call site so a specific investigation can use 16.
DEFAULT_CSCV_SPLITS = 8

# The shortest train or test half CSCV will accept. Below this a Sharpe over
# the half is dominated by its own sampling error, and a PBO assembled from such
# halves is noise about noise. Refusing is the right answer: the gate reads an
# unmeasurable PBO as a refusal, so a too-short sample cannot be promoted.
#
# 30 rather than the "enough to compute a Sharpe at all" figure of 3. The first
# draft used 6, and a test asserting that a 20-period matrix was unusable found
# that it was not: 20 periods over 8 splits leaves 8 per half, which cleared the
# bound and produced a confident-looking PBO from Sharpes over eight
# observations. At 30 a half is about six weeks of daily bars, and the matrices
# this actually runs on are training-window backtests measured in years — so the
# bound refuses only the cases that should be refused.
MIN_PERIODS_PER_HALF = 30


class SelectionError(TbError):
    """A selection statistic could not be computed from what was supplied."""


@dataclass(frozen=True, slots=True)
class Deflation:
    """A Sharpe ratio, and the same Sharpe after the search is divided out.

    Both forms are here because the gate wants both and they disagree in
    informative ways. A high level with a low probability is a short sample; a
    decent probability with a negative level is a search too large for the edge
    found.
    """

    observed_sharpe: float
    expected_max_sharpe: float
    deflated_sharpe: float
    deflated_probability: float | None
    n_trials: int
    sharpe_dispersion: float
    n_periods: int
    skew: float
    kurtosis: float
    dispersion_measured: bool = True

    @property
    def haircut(self) -> float:
        """How much of the observed Sharpe the search accounts for."""
        return self.expected_max_sharpe

    @property
    def is_measured(self) -> bool:
        """Whether the probability form could be computed at all."""
        return self.deflated_probability is not None

    def summary(self) -> str:
        probability = (
            "unmeasurable"
            if self.deflated_probability is None
            else f"{self.deflated_probability:.3f}"
        )
        return (
            f"Sharpe {self.observed_sharpe:.3f} less the expected maximum of "
            f"{self.n_trials} trial(s) ({self.expected_max_sharpe:.3f}) = "
            f"{self.deflated_sharpe:.3f}; probability {probability} over "
            f"{self.n_periods} periods"
        )


@dataclass(frozen=True, slots=True)
class PboResult:
    """The probability of backtest overfitting, and the work behind it."""

    pbo: float
    n_combinations: int
    n_trials: int
    n_periods: int
    n_splits: int
    median_logit: float

    @property
    def summary(self) -> str:
        return (
            f"PBO {self.pbo:.3f} over {self.n_combinations} train/test splits of "
            f"{self.n_trials} trials x {self.n_periods} periods "
            f"(median logit {self.median_logit:+.3f})"
        )


# --------------------------------------------------------------------------
# The expected maximum of N trials
# --------------------------------------------------------------------------


def expected_max_sharpe(*, n_trials: int, sharpe_dispersion: float) -> float:
    """The best Sharpe a search of `n_trials` yields from noise alone.

    Bailey and Lopez de Prado's approximation to the expected maximum of N
    draws from a normal with the given dispersion:

        E[max SR] ~ sigma * [ (1 - g) * Z(1 - 1/N) + g * Z(1 - 1/(N e)) ]

    with `g` the Euler-Mascheroni constant and `Z` the inverse standard normal.

    Two edge cases, both handled so the caller never has to:

    **N <= 1.** There is no maximum to take, and `Z(1 - 1/1) = Z(0)` is minus
    infinity. The expected maximum of one centred draw is zero, so the haircut
    is zero — which is the right answer: a single trial carries no selection
    bias from within its own search.

    **A non-positive dispersion.** Returns zero rather than raising, because a
    dispersion of zero genuinely implies no spread to select from. The caller is
    responsible for not *supplying* zero when the truth is "unmeasured" — see
    `FALLBACK_SHARPE_DISPERSION` in `tb.research.trials`, which exists because
    an unmeasured dispersion passed through as zero would remove the haircut
    entirely.
    """
    if n_trials <= 1 or sharpe_dispersion <= 0:
        return 0.0
    n = float(n_trials)
    first = _NORMAL.inv_cdf(1.0 - 1.0 / n)
    second = _NORMAL.inv_cdf(1.0 - 1.0 / (n * e))
    return sharpe_dispersion * ((1.0 - EULER_MASCHERONI) * first + EULER_MASCHERONI * second)


def deflate(
    *,
    observed_sharpe: float,
    returns: Sequence[float],
    n_trials: int,
    sharpe_dispersion: float,
    periods_per_year: int,
    dispersion_measured: bool = True,
) -> Deflation:
    """Deflate an annualised Sharpe by the search that produced it.

    `observed_sharpe` and `sharpe_dispersion` are both **annualised**, because
    that is how every Sharpe in this repo is reported and a mixed-units
    subtraction here would be invisible and wrong by a factor of sqrt(252).
    The probability form needs the *per-period* figures, so the conversion
    happens inside — once, in one place.

    The probability is `None` when the sample is too short to support it, or
    when the higher moments make its variance term non-positive. Both are real
    conditions rather than errors, and both must read as "not measured" at the
    gate.
    """
    haircut = expected_max_sharpe(n_trials=n_trials, sharpe_dispersion=sharpe_dispersion)
    skew = _skew(returns)
    kurt = _kurtosis(returns)
    probability = _deflated_probability(
        observed_sharpe=observed_sharpe,
        benchmark_sharpe=haircut,
        n_periods=len(returns),
        periods_per_year=periods_per_year,
        skew=skew,
        kurtosis=kurt,
    )
    return Deflation(
        observed_sharpe=observed_sharpe,
        expected_max_sharpe=haircut,
        deflated_sharpe=observed_sharpe - haircut,
        deflated_probability=probability,
        n_trials=n_trials,
        sharpe_dispersion=sharpe_dispersion,
        n_periods=len(returns),
        skew=skew,
        kurtosis=kurt,
        dispersion_measured=dispersion_measured,
    )


def _deflated_probability(
    *,
    observed_sharpe: float,
    benchmark_sharpe: float,
    n_periods: int,
    periods_per_year: int,
    skew: float,
    kurtosis: float,
) -> float | None:
    """The probabilistic Sharpe ratio at a deflated benchmark.

        PSR(SR*) = Phi[ (SR - SR*) sqrt(T - 1) / sqrt(1 - g3 SR + (g4 - 1)/4 SR^2) ]

    with every Sharpe **per period**, `g3` the skew and `g4` the (non-excess)
    kurtosis of the returns. The annualised-to-per-period conversion is why this
    is private: calling it with annualised Sharpes returns a number that looks
    reasonable and is wrong by roughly sqrt(periods_per_year), which is the kind
    of error that survives review.
    """
    if n_periods < 4 or periods_per_year <= 0:
        # Skew and kurtosis of three observations are not estimates.
        return None
    scale = sqrt(float(periods_per_year))
    per_period = observed_sharpe / scale
    benchmark = benchmark_sharpe / scale
    variance_term = 1.0 - skew * per_period + (kurtosis - 1.0) / 4.0 * per_period**2
    if variance_term <= 0:
        # Extreme higher moments. Genuinely unmeasurable rather than zero: a
        # negative variance means the approximation has left its domain, and
        # reporting a probability from it would be inventing one.
        return None
    statistic = (per_period - benchmark) * sqrt(float(n_periods - 1)) / sqrt(variance_term)
    return _NORMAL.cdf(statistic)


# --------------------------------------------------------------------------
# PBO by combinatorially symmetric cross-validation
# --------------------------------------------------------------------------


def probability_of_backtest_overfitting(
    matrix: Sequence[Sequence[float]],
    *,
    n_splits: int = DEFAULT_CSCV_SPLITS,
) -> PboResult | None:
    """PBO over a period-by-trial returns matrix.

    `matrix` is rows of periods, each row holding one value per trial — the
    orientation the cross-validation needs, since it partitions *time* and
    compares across trials.

    The procedure, from Bailey, Borwein, Lopez de Prado and Zhu: split the
    period axis into `n_splits` disjoint sub-samples; for every way of choosing
    half of them as the training set, find the trial with the best training
    Sharpe and see where it ranks on the complementary test set. PBO is the
    fraction of splits where the training winner lands below the test median.

    Returns `None` when the matrix cannot support the computation — fewer than
    two trials (nothing to select among, so overfitting by selection is not
    defined), or halves too short to carry a Sharpe. Both read as a refusal at
    the gate, which is the correct direction: "we could not check whether this
    was overfit" is not evidence that it was not.
    """
    if not matrix:
        return None
    n_trials = len(matrix[0])
    if n_trials < 2:
        return None
    if any(len(row) != n_trials for row in matrix):
        raise SelectionError(
            "the returns matrix is ragged: every period row must hold one value per "
            "trial. A ragged matrix would silently compare a trial's Sharpe over one "
            "window against another's over a different window."
        )
    if n_splits < 2 or n_splits % 2 != 0:
        raise SelectionError(
            f"n_splits must be even and at least 2, got {n_splits}. CSCV pairs each "
            "half of the sub-samples with its complement, which an odd count cannot do."
        )

    n_periods = len(matrix)
    per_split = n_periods // n_splits
    if per_split * (n_splits // 2) < MIN_PERIODS_PER_HALF:
        return None

    blocks = [list(range(index * per_split, (index + 1) * per_split)) for index in range(n_splits)]
    # Trailing periods that do not divide evenly go to the last block rather than
    # being dropped: dropping the tail would quietly shorten every backtest by up
    # to `n_splits - 1` periods, and a backtest silently shorter than the window
    # it names is the kind of discrepancy that surfaces months later.
    blocks[-1].extend(range(n_splits * per_split, n_periods))

    logits: list[float] = []
    all_indices = set(range(n_splits))
    for train_blocks in combinations(range(n_splits), n_splits // 2):
        test_blocks = sorted(all_indices - set(train_blocks))
        train_rows = [row for block in train_blocks for row in blocks[block]]
        test_rows = [row for block in test_blocks for row in blocks[block]]

        train_sharpes = _column_sharpes(matrix, train_rows, n_trials)
        test_sharpes = _column_sharpes(matrix, test_rows, n_trials)
        if train_sharpes is None or test_sharpes is None:
            continue

        best = max(range(n_trials), key=lambda column: train_sharpes[column])
        # Relative rank of the training winner among the test Sharpes, in
        # (0, 1). The `+ 1` denominator keeps the logit finite when the winner
        # is also the test best, which happens often and is not an error.
        rank = sum(1 for value in test_sharpes if value <= test_sharpes[best])
        omega = rank / (n_trials + 1)
        omega = min(max(omega, 1e-9), 1 - 1e-9)
        logits.append(log(omega / (1.0 - omega)))

    if not logits:
        return None
    overfit = sum(1 for value in logits if value < 0)
    ordered = sorted(logits)
    middle = len(ordered) // 2
    median = (
        ordered[middle] if len(ordered) % 2 == 1 else (ordered[middle - 1] + ordered[middle]) / 2.0
    )
    return PboResult(
        pbo=overfit / len(logits),
        n_combinations=len(logits),
        n_trials=n_trials,
        n_periods=n_periods,
        n_splits=n_splits,
        median_logit=median,
    )


def _column_sharpes(
    matrix: Sequence[Sequence[float]], rows: Sequence[int], n_trials: int
) -> list[float] | None:
    """Per-trial Sharpe over a subset of periods, un-annualised.

    Un-annualised because only the *ranking* matters here and annualising every
    column by the same constant cannot change a ranking. A flat column gets
    zero rather than being dropped: a strategy that did nothing over this half
    genuinely has no edge over it, and dropping it would remove a candidate
    from the comparison the rank is taken over.
    """
    if len(rows) < 3:
        return None
    out: list[float] = []
    for column in range(n_trials):
        values = [matrix[row][column] for row in rows]
        mean = sum(values) / len(values)
        variance = sum((v - mean) ** 2 for v in values) / (len(values) - 1)
        out.append(0.0 if variance <= 0 else mean / sqrt(variance))
    return out


# --------------------------------------------------------------------------
# Moments
# --------------------------------------------------------------------------


def _skew(values: Sequence[float]) -> float:
    """Population skewness. Zero on a sample too short to have one."""
    if len(values) < 3:
        return 0.0
    mean = sum(values) / len(values)
    m2 = sum((v - mean) ** 2 for v in values) / len(values)
    if m2 <= 0:
        return 0.0
    m3 = sum((v - mean) ** 3 for v in values) / len(values)
    return float(m3 / sqrt(m2**3))


def _kurtosis(values: Sequence[float]) -> float:
    """Population kurtosis, **not** excess: 3.0 for a normal sample.

    Non-excess because that is the convention the probabilistic Sharpe formula
    uses, and the difference is a whole unit in the term `(g4 - 1)/4`. Returns
    3.0 on a sample too short to estimate it, which is the normal value and
    therefore the neutral assumption.
    """
    if len(values) < 4:
        return 3.0
    mean = sum(values) / len(values)
    m2 = sum((v - mean) ** 2 for v in values) / len(values)
    if m2 <= 0:
        return 3.0
    m4 = sum((v - mean) ** 4 for v in values) / len(values)
    return m4 / m2**2
