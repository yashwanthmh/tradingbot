"""Running the null population and judging the engine by the result.

The verdict this produces is about the backtester, not about a strategy, which
is why it has its own event type and its own table. M5's promotion gate reads
the most recent calibration for the running code version: a backtest from an
uncalibrated engine is not admissible evidence, for the same reason a backtest
without a `vintage_id` is not.

The assertion is two-sided, and the second half is the part people leave out.

**No null strategy may show a materially positive net Sharpe.** That is the bug
hunt — a coin flip with positive net Sharpe means the engine is handing out
information, most likely through the fill price.

**The population's cost drag must be materially non-zero.** A run in which
nothing was charged passes the first test trivially while proving nothing. This
is the check that catches a calibration made vacuous by a fixture with no
trades, a cost model wired to zero, or a cost gate that rejected everything.

Both halves have to hold, or the calibration reports failure with the reason.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime
from decimal import Decimal

from tb.backtest.costs import CostModel
from tb.backtest.engine import Backtester, BacktestResult, InstrumentMeta
from tb.backtest.null_strategies import population
from tb.core.ids import new_id
from tb.data.asof import BarSource, ForwardOnlyReader
from tb.data.provider import Resolution
from tb.features.pipeline import FeaturePipeline, default_pipeline

# A null strategy may show this much annualised net Sharpe before it counts as
# evidence of a bug. Not zero: a finite sample of random entries has sampling
# noise, and a threshold of zero would fail on noise alone and then be raised
# by whoever was unlucky, which is worse than setting it honestly now.
DEFAULT_TOLERANCE = 0.5

# The population must pay at least this much, or the run proves nothing.
MIN_MEANINGFUL_COST_DRAG_BPS = Decimal("1")


@dataclass(frozen=True, slots=True)
class CalibrationResult:
    """Whether this engine can be trusted, and what the nulls actually did."""

    calibration_id: str
    resolution: Resolution
    rng_seed: int
    tolerance: float
    n_strategies: int
    n_runs: int
    results: tuple[BacktestResult, ...]
    failures: tuple[str, ...]

    @property
    def net_sharpes(self) -> tuple[float, ...]:
        return tuple(r.metrics.net_sharpe for r in self.results if r.metrics.net_sharpe is not None)

    @property
    def worst_net_sharpe(self) -> float | None:
        values = self.net_sharpes
        return min(values) if values else None

    @property
    def best_net_sharpe(self) -> float | None:
        """The most *dangerous* number here, despite the name.

        A high net Sharpe from a null strategy is the failure mode. "Best" is
        the arithmetic sense, not the desirable one.
        """
        values = self.net_sharpes
        return max(values) if values else None

    @property
    def mean_net_sharpe(self) -> float | None:
        values = self.net_sharpes
        return sum(values) / len(values) if values else None

    @property
    def mean_cost_drag_bps(self) -> Decimal:
        drags = [r.metrics.cost_drag_bps for r in self.results if r.metrics.n_trades]
        return sum(drags, Decimal(0)) / Decimal(len(drags)) if drags else Decimal(0)

    @property
    def total_trades(self) -> int:
        return sum(r.metrics.n_trades for r in self.results)

    @property
    def passed(self) -> bool:
        return not self.failures

    def summary(self) -> str:
        best = "n/a" if self.best_net_sharpe is None else f"{self.best_net_sharpe:.2f}"
        return (
            f"{self.n_strategies} null strategies, {self.total_trades} trades: "
            f"worst net Sharpe {self.worst_net_sharpe}, highest {best} "
            f"(tolerance {self.tolerance}), mean cost drag "
            f"{self.mean_cost_drag_bps:.1f}bps"
        )


def run_calibration(
    *,
    cost_model: CostModel,
    source: BarSource,
    instruments: Mapping[str, InstrumentMeta],
    decision_times: Sequence[datetime],
    resolution: Resolution = Resolution.DAILY,
    pipeline: FeaturePipeline | None = None,
    tolerance: float = DEFAULT_TOLERANCE,
    seed: int = 0,
    starting_equity_ccy: Decimal = Decimal("10000.00"),
    position_notional_ccy: Decimal = Decimal("1000.00"),
) -> CalibrationResult:
    """Run every null strategy and judge the engine on the results."""
    used_pipeline = pipeline or default_pipeline()
    nulls = population(seed=seed)
    results: list[BacktestResult] = []

    for strategy in nulls:
        # A fresh reader per strategy: `ForwardOnlyReader` refuses to rewind,
        # which is exactly the property that makes reuse across runs
        # impossible rather than merely wrong.
        reader = ForwardOnlyReader(
            source=source,
            resolution=resolution,
            instrument_uids=tuple(sorted(instruments)),
            lookback=None,
        )
        backtester = Backtester(
            cost_model=cost_model,
            pipeline=used_pipeline,
            instruments=instruments,
            starting_equity_ccy=starting_equity_ccy,
            position_notional_ccy=position_notional_ccy,
            rng_seed=seed,
        )
        results.append(
            backtester.run(
                strategy=strategy,  # type: ignore[arg-type]
                reader=reader,
                decision_times=decision_times,
                resolution=resolution,
            )
        )

    return CalibrationResult(
        calibration_id=new_id("calib", length=12),
        resolution=resolution,
        rng_seed=seed,
        tolerance=tolerance,
        n_strategies=len(nulls),
        n_runs=len(results),
        results=tuple(results),
        failures=tuple(_judge(results, tolerance=tolerance)),
    )


def _judge(results: Sequence[BacktestResult], *, tolerance: float) -> list[str]:
    """Both halves of the assertion."""
    failures: list[str] = []

    for result in results:
        sharpe = result.metrics.net_sharpe
        if sharpe is not None and sharpe > tolerance:
            failures.append(
                f"{result.strategy_id} shows a net Sharpe of {sharpe:.2f} against a "
                f"tolerance of {tolerance}. A strategy with no edge by construction "
                "cannot earn one: this is a fill-timing error, a mark taken from a bar "
                "the position could not see, or a cost charged on one leg — not a "
                "discovery."
            )
        # The flat control is exact, not approximate: it never trades, so any
        # movement at all is P&L with no position to attribute it to.
        if result.strategy_id == "null_always_flat":
            if result.metrics.n_trades:
                failures.append(
                    f"the always-flat control recorded {result.metrics.n_trades} trades"
                )
            if result.metrics.net_return_pct != Decimal(0):
                failures.append(
                    f"the always-flat control moved {result.metrics.net_return_pct}%. "
                    "It holds nothing, so the engine is inventing P&L with no position "
                    "to attribute it to."
                )

    traded = [r for r in results if r.metrics.n_trades]
    if not traded:
        failures.append(
            "no null strategy completed a single trade, so this run proves nothing "
            "about the engine. Check the fixture's length and the cost gate: a "
            "calibration that charges nothing passes the Sharpe test trivially."
        )
    else:
        mean_drag = sum((r.metrics.cost_drag_bps for r in traded), Decimal(0)) / Decimal(
            len(traded)
        )
        if mean_drag < MIN_MEANINGFUL_COST_DRAG_BPS:
            failures.append(
                f"mean cost drag across trading nulls is {mean_drag:.2f}bps, under the "
                f"{MIN_MEANINGFUL_COST_DRAG_BPS}bps floor. Costs are not reaching the "
                "equity curve, so the Sharpe check above is vacuous — it would pass "
                "with the fee schedule set to zero."
            )

    return failures
