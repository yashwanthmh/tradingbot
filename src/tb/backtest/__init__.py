"""Cost model, the non-cheating backtester, and its calibration.

The order of dependence matters: `costs` is the foundation because on this
venue the fee schedule, not the signal, decides whether a strategy exists.
"""

from tb.backtest.costs import (
    CostBreakdown,
    CostError,
    CostModel,
    CostVerdict,
    Jurisdiction,
    RoundTrip,
    jurisdiction_from_isin,
)

__all__ = [
    "CostBreakdown",
    "CostError",
    "CostModel",
    "CostVerdict",
    "Jurisdiction",
    "RoundTrip",
    "jurisdiction_from_isin",
]
