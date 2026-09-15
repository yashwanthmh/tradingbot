"""Cost model, the non-cheating backtester, and its calibration.

The order of dependence matters: `costs` is the foundation because on this
venue the fee schedule, not the signal, decides whether a strategy exists.
"""

from tb.backtest.calibrate import CalibrationResult, run_calibration
from tb.backtest.costs import (
    CostBreakdown,
    CostError,
    CostModel,
    CostVerdict,
    Jurisdiction,
    RoundTrip,
    jurisdiction_from_isin,
)
from tb.backtest.engine import (
    Backtester,
    BacktestError,
    BacktestResult,
    InstrumentMeta,
    Trade,
)
from tb.backtest.metrics import CurvePoint, Metrics

__all__ = [
    "BacktestError",
    "BacktestResult",
    "Backtester",
    "CalibrationResult",
    "CostBreakdown",
    "CostError",
    "CostModel",
    "CostVerdict",
    "CurvePoint",
    "InstrumentMeta",
    "Jurisdiction",
    "Metrics",
    "RoundTrip",
    "Trade",
    "jurisdiction_from_isin",
    "run_calibration",
]
