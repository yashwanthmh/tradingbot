"""Turning an equity curve into the numbers a gate can read.

Every metric here is reported **gross and net**. On a venue where a round trip
costs 40-140bps, a gross Sharpe is a number about a strategy that does not
exist, and reporting it alone is the single easiest way to build something that
looks profitable and is not. The pair is what makes the cost drag visible as a
quantity rather than an argument.

Two choices worth stating:

**Sharpe is computed from per-period returns with no risk-free adjustment, and
the annualisation factor is passed in rather than assumed.** A daily strategy
and a minute strategy annualise by different constants, and hardcoding 252
would silently overstate a minute strategy's Sharpe by about eight times.

**A sample of one or two periods has no Sharpe, and this returns `None` rather
than zero.** Zero reads as "measured, no edge"; `None` reads as "not
measured". M5's promotion gate must not be able to mistake the second for the
first.
"""

from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal, localcontext
from itertools import pairwise
from math import sqrt

# Trading days in a year, and the derived factors for finer resolutions. Passed
# explicitly to `sharpe` rather than inferred, so a resolution change cannot
# silently rescale the metric.
PERIODS_PER_YEAR_DAILY = 252
PERIODS_PER_YEAR_HOURLY = 252 * 7
PERIODS_PER_YEAR_MINUTE = 252 * 390

_WORKING_PRECISION = 40
_BPS = Decimal("10000")


@dataclass(frozen=True, slots=True)
class CurvePoint:
    """One mark-to-market observation of the portfolio."""

    equity_ccy: Decimal
    gross_equity_ccy: Decimal


@dataclass(frozen=True, slots=True)
class Metrics:
    """What a backtest produced, gross and net side by side."""

    n_periods: int
    n_trades: int

    gross_return_pct: Decimal
    net_return_pct: Decimal
    gross_sharpe: float | None
    net_sharpe: float | None
    max_drawdown_pct: Decimal
    total_cost_ccy: Decimal
    cost_drag_bps: Decimal
    turnover: Decimal

    @property
    def cost_drag_pct(self) -> Decimal:
        return self.cost_drag_bps / Decimal(100)

    def summary(self) -> str:
        net = "n/a" if self.net_sharpe is None else f"{self.net_sharpe:.2f}"
        gross = "n/a" if self.gross_sharpe is None else f"{self.gross_sharpe:.2f}"
        return (
            f"{self.n_trades} trades over {self.n_periods} periods: "
            f"net {self.net_return_pct:.2f}% (Sharpe {net}) vs "
            f"gross {self.gross_return_pct:.2f}% (Sharpe {gross}), "
            f"cost drag {self.cost_drag_bps:.1f}bps, "
            f"max drawdown {self.max_drawdown_pct:.2f}%"
        )


def returns_of(values: list[Decimal]) -> list[Decimal]:
    """Simple period returns from a level series.

    A non-positive level contributes no return rather than an undefined one:
    dividing by it would produce `inf`, which propagates into a mean and comes
    out as a plausible-looking large number.
    """
    out: list[Decimal] = []
    with localcontext() as ctx:
        ctx.prec = _WORKING_PRECISION
        for previous, current in pairwise(values):
            if previous <= 0:
                continue
            out.append((current - previous) / previous)
    return out


def sharpe(period_returns: list[Decimal], *, periods_per_year: int) -> float | None:
    """Annualised Sharpe from period returns, or `None` if unmeasurable.

    `None` rather than 0.0 on a short or zero-dispersion sample. Zero means
    "measured, no edge" and `None` means "not measured", and a promotion gate
    that could not tell them apart would treat an unmeasured strategy as a
    measured flat one.
    """
    if len(period_returns) < 3:
        return None
    with localcontext() as ctx:
        ctx.prec = _WORKING_PRECISION
        n = Decimal(len(period_returns))
        mean = sum(period_returns, Decimal(0)) / n
        variance = sum(((r - mean) ** 2 for r in period_returns), Decimal(0)) / (n - 1)
    if variance <= 0:
        # A perfectly flat curve. Not an infinite Sharpe — an unmeasurable one.
        return None
    return float(mean) / sqrt(float(variance)) * sqrt(periods_per_year)


def max_drawdown_pct(values: list[Decimal]) -> Decimal:
    """Worst peak-to-trough fall in the series, as a positive percentage."""
    if len(values) < 2:
        return Decimal(0)
    with localcontext() as ctx:
        ctx.prec = _WORKING_PRECISION
        peak = values[0]
        worst = Decimal(0)
        for value in values:
            if value > peak:
                peak = value
            if peak > 0:
                worst = max(worst, (peak - value) / peak * Decimal(100))
        return worst


def compute(
    *,
    curve: list[CurvePoint],
    n_trades: int,
    total_cost_ccy: Decimal,
    traded_notional_ccy: Decimal,
    starting_equity_ccy: Decimal,
    periods_per_year: int,
) -> Metrics:
    """Assemble every metric from one equity curve.

    `cost_drag_bps` is costs over *traded notional*, not over equity. Over
    equity it would shrink simply by trading less of the account, which would
    reward a strategy for being small rather than for being cheap.
    """
    if not curve:
        return Metrics(
            n_periods=0,
            n_trades=0,
            gross_return_pct=Decimal(0),
            net_return_pct=Decimal(0),
            gross_sharpe=None,
            net_sharpe=None,
            max_drawdown_pct=Decimal(0),
            total_cost_ccy=Decimal(0),
            cost_drag_bps=Decimal(0),
            turnover=Decimal(0),
        )

    net_levels = [point.equity_ccy for point in curve]
    gross_levels = [point.gross_equity_ccy for point in curve]

    with localcontext() as ctx:
        ctx.prec = _WORKING_PRECISION
        start = starting_equity_ccy
        net_return = (net_levels[-1] - start) / start * Decimal(100) if start > 0 else Decimal(0)
        gross_return = (
            (gross_levels[-1] - start) / start * Decimal(100) if start > 0 else Decimal(0)
        )
        drag = (
            total_cost_ccy / traded_notional_ccy * _BPS if traded_notional_ccy > 0 else Decimal(0)
        )
        turnover = traded_notional_ccy / start if start > 0 else Decimal(0)

    return Metrics(
        n_periods=len(curve),
        n_trades=n_trades,
        gross_return_pct=gross_return,
        net_return_pct=net_return,
        gross_sharpe=sharpe(returns_of(gross_levels), periods_per_year=periods_per_year),
        net_sharpe=sharpe(returns_of(net_levels), periods_per_year=periods_per_year),
        max_drawdown_pct=max_drawdown_pct(net_levels),
        total_cost_ccy=total_cost_ccy,
        cost_drag_bps=drag,
        turnover=turnover,
    )
