"""The declarative strategy language.

A spec is validated data, never code. There is no `eval`, `exec`, `compile` or
import anywhere in the interpretation path — a test walks the AST of this
package to assert that.
"""

from tb.strategy.dsl.interpreter import (
    BudgetExceeded,
    Evaluation,
    EvaluationError,
    check_features_available,
    evaluate,
)
from tb.strategy.dsl.ops import DslStrategy, pipeline_from_spec
from tb.strategy.dsl.schema import (
    MAX_DEPTH,
    MAX_LOOKBACK,
    MAX_NODES,
    All,
    Any_,
    Comparison,
    Constant,
    FeatureRef,
    ModelScore,
    Not,
    SpecError,
    StrategySpec,
)

__all__ = [
    "MAX_DEPTH",
    "MAX_LOOKBACK",
    "MAX_NODES",
    "All",
    "Any_",
    "BudgetExceeded",
    "Comparison",
    "Constant",
    "DslStrategy",
    "Evaluation",
    "EvaluationError",
    "FeatureRef",
    "ModelScore",
    "Not",
    "SpecError",
    "StrategySpec",
    "check_features_available",
    "evaluate",
    "pipeline_from_spec",
]
