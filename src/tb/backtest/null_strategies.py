"""Strategies with no edge, and the calibration that uses them.

This is the most important test in the backtester and the least glamorous. A
strategy that decides by coin flip has, by construction, zero expected gross
edge. After costs it must therefore lose approximately the cost drag — and if
it does not, the engine is wrong in a way that will flatter *every* strategy
run through it.

The specific failures this catches, each of which produces a plausible small
positive Sharpe rather than an obvious error:

* **Filling at the decision bar's close.** The signal is computed from that
  close, so the fill is free information. Worth roughly the whole gross edge
  at minute resolution.
* **Marking to a bar the position could not see.** Same error, in the equity
  curve rather than the fill.
* **Charging costs on one leg.** Halves the drag, so a null strategy drifts
  toward break-even instead of bleeding.
* **Sign errors in P&L** that happen to favour the position.
* **Survivorship in the fixture**, if calibration runs on a universe selected
  for having survived.

So the assertion is two-sided. A null strategy's net Sharpe must not be
materially *positive* — that is the bug hunt — and its cost drag must be
materially non-zero, because a run where nothing was charged proves nothing
about whether charging works.

`tb backtest calibrate` is a release gate, not a diagnostic.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from decimal import Decimal

from tb.data.asof import BarWindow
from tb.features.pipeline import FeatureSnapshot
from tb.strategy.base import Action, Decision, PositionState, hold


@dataclass(slots=True)
class CoinFlipStrategy:
    """Enters or exits at random, with a fixed probability per decision.

    Seeded, and the seed is part of the recorded result: an unreproducible
    calibration cannot be compared across commits, which is the only way to
    notice that a refactor broke the fill timing.

    Declares a large `expected_edge_bps` deliberately. The point of a null run
    is to exercise the *engine*, and a null strategy that declared 5bps would
    be refused by the cost gate on every signal — measuring the gate rather
    than the fills. The declared edge is a lie, which is exactly why this is
    not a strategy anyone would promote.
    """

    strategy_id: str = "null_coinflip"
    version: int = 1
    enter_probability: float = 0.1
    exit_probability: float = 0.2
    seed: int = 0
    declared_edge_bps: Decimal = Decimal("400")
    _rng: random.Random = field(init=False)

    def __post_init__(self) -> None:
        # Seeded and deliberately not cryptographic: the whole point is that a
        # calibration is reproducible across commits, which is the only way to
        # notice that a refactor broke the fill timing. `secrets` would make
        # every run incomparable to the last.
        self._rng = random.Random(self.seed)  # noqa: S311

    @property
    def required_features(self) -> tuple[str, ...]:
        return ()

    def decide(
        self,
        *,
        snapshot: FeatureSnapshot,
        window: BarWindow,
        position: PositionState,
    ) -> Decision:
        roll = self._rng.random()
        if position.is_open:
            if roll < self.exit_probability:
                return Decision(
                    as_of=snapshot.as_of,
                    instrument_uid=snapshot.instrument_uid,
                    action=Action.EXIT,
                    expected_edge_bps=Decimal(0),
                    feature_snapshot_hash=snapshot.snapshot_hash,
                    strategy_id=self.strategy_id,
                    strategy_version=self.version,
                    rationale="coin flip: exit",
                )
            return hold(
                as_of=snapshot.as_of,
                instrument_uid=snapshot.instrument_uid,
                strategy_id=self.strategy_id,
                strategy_version=self.version,
                snapshot_hash=snapshot.snapshot_hash,
                rationale="coin flip: hold",
            )
        if roll < self.enter_probability:
            return Decision(
                as_of=snapshot.as_of,
                instrument_uid=snapshot.instrument_uid,
                action=Action.ENTER,
                expected_edge_bps=self.declared_edge_bps,
                feature_snapshot_hash=snapshot.snapshot_hash,
                strategy_id=self.strategy_id,
                strategy_version=self.version,
                rationale="coin flip: enter",
            )
        return hold(
            as_of=snapshot.as_of,
            instrument_uid=snapshot.instrument_uid,
            strategy_id=self.strategy_id,
            strategy_version=self.version,
            snapshot_hash=snapshot.snapshot_hash,
            rationale="coin flip: no entry",
        )


@dataclass(slots=True)
class AlwaysLongStrategy:
    """Buys once and never sells.

    The buy-and-hold control. Its purpose is different from the coin flip's: it
    has *one* round trip, so its cost drag is a single known quantity, and its
    net-versus-gross gap is the most direct check that costs are charged once
    per leg rather than twice or never.
    """

    strategy_id: str = "null_always_long"
    version: int = 1
    declared_edge_bps: Decimal = Decimal("400")

    @property
    def required_features(self) -> tuple[str, ...]:
        return ()

    def decide(
        self,
        *,
        snapshot: FeatureSnapshot,
        window: BarWindow,
        position: PositionState,
    ) -> Decision:
        if position.is_open:
            return hold(
                as_of=snapshot.as_of,
                instrument_uid=snapshot.instrument_uid,
                strategy_id=self.strategy_id,
                strategy_version=self.version,
                snapshot_hash=snapshot.snapshot_hash,
                rationale="always long: holding",
            )
        return Decision(
            as_of=snapshot.as_of,
            instrument_uid=snapshot.instrument_uid,
            action=Action.ENTER,
            expected_edge_bps=self.declared_edge_bps,
            feature_snapshot_hash=snapshot.snapshot_hash,
            strategy_id=self.strategy_id,
            strategy_version=self.version,
            rationale="always long: entering",
        )


@dataclass(slots=True)
class AlwaysFlatStrategy:
    """Never trades. The zero control.

    Its equity curve must be exactly flat and its cost exactly zero. If this
    one shows any return at all, the engine is inventing P&L with no position
    to attribute it to — which would otherwise be invisible under the noise of
    a strategy that does trade.
    """

    strategy_id: str = "null_always_flat"
    version: int = 1

    @property
    def required_features(self) -> tuple[str, ...]:
        return ()

    def decide(
        self,
        *,
        snapshot: FeatureSnapshot,
        window: BarWindow,
        position: PositionState,
    ) -> Decision:
        return hold(
            as_of=snapshot.as_of,
            instrument_uid=snapshot.instrument_uid,
            strategy_id=self.strategy_id,
            strategy_version=self.version,
            snapshot_hash=snapshot.snapshot_hash,
            rationale="always flat",
        )


@dataclass(slots=True)
class AlternatingStrategy:
    """Enters and exits on alternate decisions. The maximum-turnover control.

    Deterministic, so its trade count is known in advance, which makes it the
    right instrument for checking that the cost total equals the per-trade cost
    times the trade count. It also produces the worst possible cost drag, which
    is the honest upper bound on what turnover costs on this venue.
    """

    strategy_id: str = "null_alternating"
    version: int = 1
    declared_edge_bps: Decimal = Decimal("400")

    @property
    def required_features(self) -> tuple[str, ...]:
        return ()

    def decide(
        self,
        *,
        snapshot: FeatureSnapshot,
        window: BarWindow,
        position: PositionState,
    ) -> Decision:
        if position.is_open:
            return Decision(
                as_of=snapshot.as_of,
                instrument_uid=snapshot.instrument_uid,
                action=Action.EXIT,
                expected_edge_bps=Decimal(0),
                feature_snapshot_hash=snapshot.snapshot_hash,
                strategy_id=self.strategy_id,
                strategy_version=self.version,
                rationale="alternating: exit",
            )
        return Decision(
            as_of=snapshot.as_of,
            instrument_uid=snapshot.instrument_uid,
            action=Action.ENTER,
            expected_edge_bps=self.declared_edge_bps,
            feature_snapshot_hash=snapshot.snapshot_hash,
            strategy_id=self.strategy_id,
            strategy_version=self.version,
            rationale="alternating: enter",
        )


def population(seed: int = 0) -> tuple[object, ...]:
    """The null population `tb backtest calibrate` runs.

    Four kinds rather than one, because each catches a different engine bug:
    the flat one catches invented P&L, the always-long one catches
    double-charging, the alternating one catches a cost total that does not
    match its trade count, and the coin flips catch fill-timing errors that
    only show up over many entries.
    """
    return (
        AlwaysFlatStrategy(),
        AlwaysLongStrategy(),
        AlternatingStrategy(),
        *(CoinFlipStrategy(strategy_id=f"null_coinflip_{i}", seed=seed + i) for i in range(8)),
    )
