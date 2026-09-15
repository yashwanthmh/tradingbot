"""What a strategy is, and what it must declare to be allowed to trade.

One interface, satisfied by all three learning layers the plan calls for: a
DSL-interpreted spec, an ML model, and (deferred) an RL policy. They differ in
how a signal is produced and not at all in what the risk layer sees.

The load-bearing part is `Decision.expected_edge_bps`. It is required, not
optional, and the reason is the venue rather than tidiness: a round trip costs
40-140bps here depending on jurisdiction, so `expected_cost_bps /
expected_edge_bps` is the gate that decides whether an order exists at all. A
strategy that will not say what it expects to earn cannot be cost-gated, and
something that cannot be cost-gated does not trade.

That declared number is also the one input to the gate the strategy controls,
which is why `costs.max_expected_edge_bps` bounds it — see `tb.backtest.costs`.
A strategy cannot divide its way through its own constraint.

Long-only and unlevered, because the Trading 212 API covers Invest and Stocks
ISA only. `Action` has no SHORT. Cash is the sole defensive position, and that
bounds what the M6 search loop is permitted to invent.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from datetime import datetime
from decimal import Decimal
from enum import StrEnum
from typing import Protocol, runtime_checkable

from tb.core.errors import TbError
from tb.data.asof import BarWindow
from tb.features.pipeline import FeatureSnapshot


class StrategyError(TbError):
    """A strategy could not produce a usable decision."""


class Action(StrEnum):
    """What a strategy wants to do about one instrument.

    No SHORT. The account type does not permit it, so the type system should
    not either — an unreachable enum member is an invitation for a searcher to
    propose specs that can never be executed, and for a reviewer to assume the
    capability exists.
    """

    ENTER = "enter"
    EXIT = "exit"
    HOLD = "hold"

    @property
    def is_risk_increasing(self) -> bool:
        """Whether this action adds exposure.

        The rate governor's priority classes and the regime gate both key on
        this: an exit must never queue behind a batch of entries.
        """
        return self is Action.ENTER


@dataclass(frozen=True, slots=True)
class Decision:
    """One strategy's verdict on one instrument at one decision time.

    Carries its own evidence. `feature_snapshot_hash` and `as_of` are what
    make `tb replay --fill <id>` able to reconstruct the inputs months later,
    rather than reporting that a decision happened and leaving why unanswerable.
    """

    as_of: datetime
    instrument_uid: str
    action: Action
    expected_edge_bps: Decimal
    feature_snapshot_hash: str
    strategy_id: str
    strategy_version: int
    rationale: str = ""
    # Populated by the interpreter for a DSL spec: which predicates fired.
    # Not for display — it is how a nonsensical-looking decision gets
    # diagnosed without re-running the search that produced the spec.
    evidence: Mapping[str, str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.as_of.tzinfo is None:
            raise StrategyError(f"{self.strategy_id}: as_of is a naive datetime")
        if self.action.is_risk_increasing and self.expected_edge_bps <= 0:
            raise StrategyError(
                f"{self.strategy_id} proposed {self.action.value} on "
                f"{self.instrument_uid} with an expected edge of "
                f"{self.expected_edge_bps}bps. A risk-increasing action must declare a "
                "positive expected edge, because the cost gate divides by it — and on "
                "this venue a round trip costs 40-140bps, so an undeclared edge is not a "
                "modelling gap, it is a trade that cannot be evaluated."
            )

    @property
    def wants_to_trade(self) -> bool:
        return self.action is not Action.HOLD


def hold(
    *,
    as_of: datetime,
    instrument_uid: str,
    strategy_id: str,
    strategy_version: int,
    snapshot_hash: str,
    rationale: str,
) -> Decision:
    """A do-nothing decision.

    A helper because HOLD is the overwhelmingly common case and it still has to
    carry its snapshot hash: "the strategy looked and declined" is a different
    fact from "the strategy was not consulted", and only the first one is
    evidence that the loop was alive.
    """
    return Decision(
        as_of=as_of,
        instrument_uid=instrument_uid,
        action=Action.HOLD,
        expected_edge_bps=Decimal(0),
        feature_snapshot_hash=snapshot_hash,
        strategy_id=strategy_id,
        strategy_version=strategy_version,
        rationale=rationale,
    )


@dataclass(frozen=True, slots=True)
class PositionState:
    """What the strategy is told about its own position.

    Deliberately minimal, and deliberately not the broker's position record.
    A strategy may know whether it is in, at what price, and for how long —
    enough for an exit rule — and nothing about account equity, other
    strategies' positions, or available cash. Sizing and allocation are the
    risk layer's and the allocator's jobs, and a strategy that could see
    equity would start expressing opinions about it.
    """

    instrument_uid: str
    quantity: Decimal = Decimal(0)
    entry_price: Decimal | None = None
    entry_at: datetime | None = None

    @property
    def is_open(self) -> bool:
        return self.quantity > 0

    def holding_minutes_at(self, moment: datetime) -> int | None:
        """How long this position has been open at `moment`.

        Refuses a negative interval rather than returning one. A position whose
        `entry_at` is after the decision time is impossible in a forward-walking
        system, so it means clock skew, a reconciliation that wrote a bad
        timestamp, or a bar/position timestamp mix-up.

        Returning the negative number is the dangerous option, and not
        obviously so: every minimum-hold check is `held < minimum`, and a
        negative value is less than *every* minimum. The position would be
        refused an exit at every decision, for as long as the bad timestamp
        stood — arriving at a permanently unhedged position through exactly the
        gate that exists to prevent one.
        """
        if self.entry_at is None:
            return None
        elapsed = (moment - self.entry_at).total_seconds()
        if elapsed < 0:
            raise StrategyError(
                f"{self.instrument_uid}: position entry_at "
                f"{self.entry_at.isoformat()} is after the decision time "
                f"{moment.isoformat()}. A negative holding period is less than every "
                "minimum-hold threshold, so this would silently block the exit at every "
                "decision rather than being noticed."
            )
        return int(elapsed // 60)


@runtime_checkable
class Strategy(Protocol):
    """The one interface. A DSL spec, an ML model and an RL policy all satisfy it.

    `decide` receives a `FeatureSnapshot` and a `BarWindow`, both of which
    contain nothing later than the decision time — the window structurally so
    (see `tb.data.asof`). The strategy is handed the window as well as the
    features because an exit rule legitimately needs the raw last price, and
    forcing that through a feature would mean a strategy could not place a
    stop.
    """

    @property
    def strategy_id(self) -> str: ...

    @property
    def version(self) -> int: ...

    @property
    def required_features(self) -> tuple[str, ...]:
        """Feature names this strategy reads.

        Declared so a spec referring to a feature the pipeline does not compute
        is rejected at registration rather than producing `UNKNOWN` at every
        decision and silently never trading.
        """
        ...

    def decide(
        self,
        *,
        snapshot: FeatureSnapshot,
        window: BarWindow,
        position: PositionState,
    ) -> Decision: ...
