"""A reinforcement-learning policy, as far as this system goes: the shape one
must take to trade through it, and nothing that trains one.

M9 is a stub by decision; docs/decisions/0003-rl-deferred.md has the reasons,
and the conditions under which to revisit them. What this module fixes is the
interface, so a policy can slot in later without a second code path, and so the
properties that make every other strategy safe hold for a policy by
construction rather than by review:

* **It observes what the snapshot hash covers, and nothing else.** A policy is
  shown the named features from the one pipeline and its own position: not
  the bar window, not the account. Every input to a decision is therefore in
  the snapshot the decision is recorded with, and `tb replay` can reproduce
  it. A policy that read raw bars would decide on inputs no hash covers.
* **It is never shown a missing value dressed as a number.** When a feature it
  observes is `UNKNOWN`, the adapter decides without asking it: flat stays
  flat, and a held position is exited. That is the DSL's asymmetry, for the
  DSL's reason — a missing feature is never a reason to keep risk on.
* **Its action is a target: flat or long.** No short, as for every strategy
  here, and no size, which is the allocator's and the risk layer's to set.
* **Its expected edge is a declared constant, not an output.** The cost gate
  divides by it. A learned policy allowed to state its edge per decision
  would learn to state whatever clears the gate.
* **It is frozen while it trades.** The interface has no reward, no update and
  no hook the loop could call. A policy that learned from live fills would be
  an optimiser running inside the risk loop, tuning itself against the
  breakers that exist to bound it. Training, if it ever exists, happens
  elsewhere and produces a new version.
* **Its features come from the fixed library**, as a spec's do, so the one
  pipeline computes a policy's inputs exactly as it computes everything else.

`PolicyStrategy` binds a policy to the `Strategy` interface, so it can be
backtested and put through `tb.strategy.rl.conformance`. Nothing builds one
into a funded book: `funded_book` constructs strategies from registered specs
only, a policy is not a spec, and a test holds that line until the decision
record's conditions are met.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from decimal import Decimal
from enum import StrEnum
from types import MappingProxyType
from typing import Protocol, runtime_checkable

from tb.data.asof import BarWindow
from tb.features.pipeline import FeaturePipeline, FeatureSnapshot, FeatureSpec
from tb.strategy.base import Action, Decision, PositionState, StrategyError, hold


class Target(StrEnum):
    """The position a policy wants in one instrument.

    No SHORT, for the reason `Action` has none; no size, because a policy that
    chose its own size would be choosing its own exposure, which the allocator
    and the risk layer exist to bound.
    """

    FLAT = "flat"
    LONG = "long"


@dataclass(frozen=True, slots=True)
class Observation:
    """Everything a policy is shown at one decision.

    `features` holds each observed feature as a number, in a read-only
    mapping: never `UNKNOWN`, because the adapter decides without the policy
    when one is missing, and never writable, because a policy that could edit
    what it was shown could leave notes for its next call. `held_minutes` is
    `None` when flat, or when the entry time is unknown.
    """

    features: Mapping[str, Decimal]
    holding: bool
    held_minutes: int | None


@runtime_checkable
class Policy(Protocol):
    """What a policy must provide. `act` is the only thing the adapter calls."""

    @property
    def policy_id(self) -> str: ...

    @property
    def version(self) -> int: ...

    @property
    def features(self) -> tuple[FeatureSpec, ...]:
        """What it observes, as specs from the feature library."""
        ...

    @property
    def expected_edge_bps(self) -> Decimal:
        """The edge it claims per round trip: a constant, bounded by the limits."""
        ...

    @property
    def min_holding_minutes(self) -> int:
        """Its own minimum hold, as a spec declares one."""
        ...

    def act(self, observation: Observation) -> Target: ...


def policy_pipeline(policy: Policy) -> FeaturePipeline:
    """The one pipeline, computing exactly the features this policy observes."""
    return FeaturePipeline(specs=tuple(policy.features))


@dataclass(frozen=True, slots=True)
class PolicyStrategy:
    """A policy bound to the `Strategy` interface."""

    policy: Policy

    @property
    def strategy_id(self) -> str:
        return self.policy.policy_id

    @property
    def version(self) -> int:
        return self.policy.version

    @property
    def required_features(self) -> tuple[str, ...]:
        return tuple(spec.name for spec in self.policy.features)

    @property
    def expected_edge_bps(self) -> Decimal:
        return self.policy.expected_edge_bps

    @property
    def label(self) -> str:
        return f"{self.strategy_id}@v{self.version}"

    def decide(
        self,
        *,
        snapshot: FeatureSnapshot,
        window: BarWindow,
        position: PositionState,
    ) -> Decision:
        """Turn the policy's target into an action, asking it only when it may.

        `window` is accepted to satisfy the interface and deliberately not
        passed on: see the module docstring.
        """
        held = position.holding_minutes_at(snapshot.as_of) if position.is_open else None
        minimum = self.policy.min_holding_minutes
        if position.is_open and held is not None and held < minimum:
            return self._hold(
                snapshot, f"held {held}min, under the policy's own {minimum}min minimum"
            )

        observed: dict[str, Decimal] = {}
        missing: list[str] = []
        for name in self.required_features:
            value = snapshot.get(name)
            if isinstance(value, Decimal):
                observed[name] = value
            else:
                missing.append(name)
        if missing:
            if position.is_open:
                return self._decide(
                    snapshot,
                    Action.EXIT,
                    rationale=(
                        f"{sorted(missing)} have no value, so exiting without asking the "
                        "policy: a missing feature is never a reason to keep risk on"
                    ),
                    evidence={"unknown": ",".join(sorted(missing))},
                )
            return self._hold(
                snapshot, f"{sorted(missing)} have no value yet, so the policy was not asked"
            )

        target = self.policy.act(
            Observation(
                features=MappingProxyType(observed), holding=position.is_open, held_minutes=held
            )
        )
        if not isinstance(target, Target):
            raise StrategyError(
                f"{self.label} returned {target!r}, not a Target. A policy's answer is flat "
                "or long; anything else is refused rather than interpreted."
            )
        evidence = {"target": target.value}
        if position.is_open:
            if target is Target.FLAT:
                return self._decide(
                    snapshot, Action.EXIT, rationale="policy wants flat", evidence=evidence
                )
            return self._hold(snapshot, "policy holds long")
        if target is Target.LONG:
            return self._decide(
                snapshot, Action.ENTER, rationale="policy wants long", evidence=evidence
            )
        return self._hold(snapshot, "policy stays flat")

    def _decide(
        self,
        snapshot: FeatureSnapshot,
        action: Action,
        *,
        rationale: str,
        evidence: Mapping[str, str],
    ) -> Decision:
        # Zero on an exit, as for a spec: the cost is sunk in the position and
        # refusing the exit would not avoid it. On an entry the declared edge,
        # which `Decision` refuses unless it is positive.
        edge = self.policy.expected_edge_bps if action is Action.ENTER else Decimal(0)
        return Decision(
            as_of=snapshot.as_of,
            instrument_uid=snapshot.instrument_uid,
            action=action,
            expected_edge_bps=edge,
            feature_snapshot_hash=snapshot.snapshot_hash,
            strategy_id=self.strategy_id,
            strategy_version=self.version,
            rationale=f"{self.label}: {rationale}",
            evidence=dict(evidence),
        )

    def _hold(self, snapshot: FeatureSnapshot, rationale: str) -> Decision:
        return hold(
            as_of=snapshot.as_of,
            instrument_uid=snapshot.instrument_uid,
            strategy_id=self.strategy_id,
            strategy_version=self.version,
            snapshot_hash=snapshot.snapshot_hash,
            rationale=f"{self.label}: {rationale}",
        )
