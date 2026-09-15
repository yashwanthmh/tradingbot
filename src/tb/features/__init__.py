"""Feature computation.

One pipeline, used identically by backtest, paper and live. The package exists
to make that singular: there is no per-caller variant to import by mistake.
"""

from tb.features.pipeline import (
    FEATURE_LIBRARY,
    FeatureError,
    FeaturePipeline,
    FeatureSnapshot,
    FeatureSpec,
    default_pipeline,
    make_spec,
)

__all__ = [
    "FEATURE_LIBRARY",
    "FeatureError",
    "FeaturePipeline",
    "FeatureSnapshot",
    "FeatureSpec",
    "default_pipeline",
    "make_spec",
]
