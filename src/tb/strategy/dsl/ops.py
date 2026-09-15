"""A DSL spec, as a `Strategy`.

The adapter between validated data and the one interface the risk layer sees.
It holds the spec and a spec-derived feature pipeline, and it is where the
asymmetry between entry and exit lives:

**An unevaluable entry predicate means no entry.** If a feature is `UNKNOWN`
the strategy does not know whether its conditions hold, and acting on that is
guessing.

**An unevaluable exit predicate means exit anyway, once a position is open and
past its minimum hold.** This is the opposite default and it is deliberate.
Refusing to close a position because a feature is missing converts a *data*
problem into an unhedged position — and with no bracket orders on this venue,
an open position with no working exit logic is the state the entire safety
design exists to avoid. Every entry rule in this system exists to stop the bot
taking on risk; none of them is a reason to hold risk it has decided to shed.
"""

from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal

from tb.data.asof import UNKNOWN, BarWindow
from tb.features.pipeline import FeaturePipeline, FeatureSnapshot, make_spec
from tb.strategy.base import Action, Decision, PositionState, hold
from tb.strategy.dsl.interpreter import evaluate
from tb.strategy.dsl.schema import StrategySpec


def pipeline_from_spec(spec: StrategySpec) -> FeaturePipeline:
    """Build the pipeline this spec needs, from the spec itself.

    Not a default pipeline the spec is hoped to fit. A spec reading a feature
    the pipeline does not compute evaluates to `UNKNOWN` at every decision and
    is recorded as a strategy that found no opportunities — a false negative
    that looks exactly like a true one.
    """
    return FeaturePipeline(
        specs=tuple(make_spec(kind, lookback) for kind, lookback in spec.feature_requests)
    )


@dataclass(frozen=True, slots=True)
class DslStrategy:
    """A spec bound to the `Strategy` interface."""

    spec: StrategySpec
    strategy_id: str
    version: int = 1

    @property
    def required_features(self) -> tuple[str, ...]:
        return self.spec.required_features

    @property
    def expected_edge_bps(self) -> Decimal:
        return self.spec.expected_edge_bps

    def decide(
        self,
        *,
        snapshot: FeatureSnapshot,
        window: BarWindow,
        position: PositionState,
    ) -> Decision:
        if position.is_open:
            return self._decide_exit(snapshot=snapshot, position=position)
        return self._decide_entry(snapshot=snapshot)

    # -- entry -------------------------------------------------------------

    def _decide_entry(self, *, snapshot: FeatureSnapshot) -> Decision:
        outcome = evaluate(self.spec.entry, snapshot)
        if outcome.result is UNKNOWN:
            return hold(
                as_of=snapshot.as_of,
                instrument_uid=snapshot.instrument_uid,
                strategy_id=self.strategy_id,
                strategy_version=self.version,
                snapshot_hash=snapshot.snapshot_hash,
                rationale=(
                    "entry conditions unevaluable: "
                    f"{sorted(snapshot.unknown_features())} have no value yet"
                ),
            )
        if not outcome.fired:
            return hold(
                as_of=snapshot.as_of,
                instrument_uid=snapshot.instrument_uid,
                strategy_id=self.strategy_id,
                strategy_version=self.version,
                snapshot_hash=snapshot.snapshot_hash,
                rationale="entry conditions not met",
            )
        return Decision(
            as_of=snapshot.as_of,
            instrument_uid=snapshot.instrument_uid,
            action=Action.ENTER,
            expected_edge_bps=self.spec.expected_edge_bps,
            feature_snapshot_hash=snapshot.snapshot_hash,
            strategy_id=self.strategy_id,
            strategy_version=self.version,
            rationale=f"entry conditions met for {self.spec.name!r}",
            evidence=dict(outcome.evidence),
        )

    # -- exit --------------------------------------------------------------

    def _decide_exit(self, *, snapshot: FeatureSnapshot, position: PositionState) -> Decision:
        held = position.holding_minutes_at(snapshot.as_of)
        if held is not None and held < self.spec.min_holding_minutes:
            return hold(
                as_of=snapshot.as_of,
                instrument_uid=snapshot.instrument_uid,
                strategy_id=self.strategy_id,
                strategy_version=self.version,
                snapshot_hash=snapshot.snapshot_hash,
                rationale=(
                    f"held {held}min, under the spec's own "
                    f"{self.spec.min_holding_minutes}min minimum"
                ),
            )

        outcome = evaluate(self.spec.exit, snapshot)
        if outcome.result is UNKNOWN:
            # The asymmetry. A missing feature is not a reason to keep risk on.
            return Decision(
                as_of=snapshot.as_of,
                instrument_uid=snapshot.instrument_uid,
                action=Action.EXIT,
                expected_edge_bps=Decimal(0),
                feature_snapshot_hash=snapshot.snapshot_hash,
                strategy_id=self.strategy_id,
                strategy_version=self.version,
                rationale=(
                    "exit conditions unevaluable, so exiting: refusing to close a "
                    "position because a feature is missing would turn a data problem "
                    "into an unhedged one"
                ),
                evidence=dict(outcome.evidence),
            )
        if outcome.fired:
            return Decision(
                as_of=snapshot.as_of,
                instrument_uid=snapshot.instrument_uid,
                action=Action.EXIT,
                expected_edge_bps=Decimal(0),
                feature_snapshot_hash=snapshot.snapshot_hash,
                strategy_id=self.strategy_id,
                strategy_version=self.version,
                rationale=f"exit conditions met for {self.spec.name!r}",
                evidence=dict(outcome.evidence),
            )
        return hold(
            as_of=snapshot.as_of,
            instrument_uid=snapshot.instrument_uid,
            strategy_id=self.strategy_id,
            strategy_version=self.version,
            snapshot_hash=snapshot.snapshot_hash,
            rationale="exit conditions not met; holding",
        )
