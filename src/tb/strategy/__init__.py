"""Strategies: one interface, three eventual implementations.

`base` defines what the risk layer sees. `dsl` is the declarative
implementation M6's searcher generates; M7's ML layer and M9's RL stub satisfy
the same protocol without the rest of the system knowing which is which.
"""

from tb.strategy.base import (
    Action,
    Decision,
    PositionState,
    Strategy,
    StrategyError,
    hold,
)

__all__ = [
    "Action",
    "Decision",
    "PositionState",
    "Strategy",
    "StrategyError",
    "hold",
]
