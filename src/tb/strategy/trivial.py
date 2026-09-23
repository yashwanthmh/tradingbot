"""One hand-written strategy, so the live loop has something to ask.

Deliberately not an attempt at edge. M4's subject is the *loop* — that a
decision travels through the risk engine, the intent log, the broker and back
into the ledger exactly once — and a strategy with real opinions would make
every loop test depend on whether those opinions fired.

So: a slow moving-average cross with a long minimum hold. It is the simplest
thing that is not a null strategy, and the choice of "not null" matters — a
strategy that never traded would let the whole loop pass its tests without
ever placing an order.

**It declares a deliberately modest edge.** On this venue a round trip costs
40-140bps against 5-20bps of gross minute-bar edge, so a daily-resolution
cross is the only shape that could clear the cost gate at all. The declared
`expected_edge_bps` is what the gate divides by, and inflating it to get
trades through would defeat the one control keeping the system out of the fee
trap. It is set where a genuine daily trend-following edge plausibly sits, and
the cost gate is then allowed to refuse it — which on a cheap US name it does
not, and on an Irish-domiciled one it does.

M5 replaces this with searched and gated specs. Nothing here should be
promoted, and `author_kind` records it as hand-written so the registry can
tell.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from decimal import Decimal

from tb.data.asof import UNKNOWN, BarWindow
from tb.features.pipeline import FeatureSnapshot, FeatureSpec, make_spec
from tb.strategy.base import Action, Decision, PositionState, hold

# The two lookbacks. Wide apart, because a fast/slow pair that are close
# together crosses constantly and turnover is what costs money here.
FAST_DAYS = 20
SLOW_DAYS = 100

# What this strategy claims, and the arithmetic behind the number.
#
# A 20/100 daily cross holds for weeks — it is a trend follower, not a
# scalper — and a trend it catches is a percent-level move rather than a
# handful of basis points. 250bps is the low end of what such a strategy
# targets per round trip.
#
# The first version of this file declared 60bps, which the cost gate refused:
# 36bps of round-trip cost on a US name against 60bps of edge is a ratio of
# 0.60, over the 0.33 limit. That refusal was correct and the *declaration*
# was wrong — 60bps is what a strategy holding for minutes might claim, not
# one holding for a month. Raising it to match the holding period is not
# gaming the gate; raising it to whatever number happens to clear the gate
# would be, which is what `costs.max_expected_edge_bps` exists to bound.
#
# At 250bps the gate permits this on a US issuer (36bps cost, ratio 0.14) and
# still refuses it on an Irish-domiciled one, where stamp duty pushes the
# round trip past 150bps. That asymmetry is the point: the same strategy is
# tradable in one jurisdiction and not in another, and the gate is what makes
# that visible rather than discovered in the P&L.
DECLARED_EDGE_BPS = Decimal("250")


def specs() -> tuple[FeatureSpec, ...]:
    """The features this strategy needs, for the pipeline to compute."""
    return (
        make_spec("sma", FAST_DAYS, name=f"sma_{FAST_DAYS}"),
        make_spec("sma", SLOW_DAYS, name=f"sma_{SLOW_DAYS}"),
        make_spec("last", 1, name="close"),
    )


@dataclass(frozen=True, slots=True)
class MovingAverageCross:
    """Long when the fast average is above the slow one, flat otherwise.

    Holds no state between decisions. Every input arrives in `decide`, which
    is what lets the same object be asked about a past instant in a backtest
    and the present one live without behaving differently — the property the
    single `FeaturePipeline` exists to protect, applied to the strategy too.
    """

    strategy_id: str = "trivial_ma_cross"
    version: int = 1
    min_hold_minutes: int = 60 * 24

    @property
    def required_features(self) -> tuple[str, ...]:
        return (f"sma_{FAST_DAYS}", f"sma_{SLOW_DAYS}", "close")

    def decide(
        self,
        *,
        snapshot: FeatureSnapshot,
        window: BarWindow,
        position: PositionState,
    ) -> Decision:
        """Cross up to enter, cross down to exit.

        Every `UNKNOWN` path holds rather than guessing. That is not caution
        for its own sake: `UNKNOWN` here means the window was too short, or it
        spans a split we could not adjust for, and both mean the cross this
        strategy reads does not exist in the data. Trading on it would be
        trading on an artefact.
        """
        fast = snapshot.get(f"sma_{FAST_DAYS}")
        slow = snapshot.get(f"sma_{SLOW_DAYS}")

        if fast is UNKNOWN or slow is UNKNOWN:
            return hold(
                as_of=snapshot.as_of,
                instrument_uid=snapshot.instrument_uid,
                strategy_id=self.strategy_id,
                strategy_version=self.version,
                snapshot_hash=snapshot.snapshot_hash,
                rationale=(
                    f"one of the averages is unknown over {snapshot.n_bars_seen} bars "
                    f"(needs {SLOW_DAYS}); the cross this reads does not exist in the data"
                ),
            )

        assert isinstance(fast, Decimal)  # narrowed by the UNKNOWN check above
        assert isinstance(slow, Decimal)
        above = fast > slow

        if position.is_open:
            if above:
                return hold(
                    as_of=snapshot.as_of,
                    instrument_uid=snapshot.instrument_uid,
                    strategy_id=self.strategy_id,
                    strategy_version=self.version,
                    snapshot_hash=snapshot.snapshot_hash,
                    rationale=f"holding: fast {fast} still above slow {slow}",
                )
            return Decision(
                as_of=snapshot.as_of,
                instrument_uid=snapshot.instrument_uid,
                action=Action.EXIT,
                # Zero on an exit, and permitted to be: the cost gate does not
                # apply to exits, because the cost is already sunk in the
                # position and refusing the exit does not avoid it.
                expected_edge_bps=Decimal(0),
                feature_snapshot_hash=snapshot.snapshot_hash,
                strategy_id=self.strategy_id,
                strategy_version=self.version,
                rationale=f"cross down: fast {fast} below slow {slow}",
                evidence={"fast": str(fast), "slow": str(slow)},
            )

        if not above:
            return hold(
                as_of=snapshot.as_of,
                instrument_uid=snapshot.instrument_uid,
                strategy_id=self.strategy_id,
                strategy_version=self.version,
                snapshot_hash=snapshot.snapshot_hash,
                rationale=f"flat: fast {fast} below slow {slow}",
            )

        return Decision(
            as_of=snapshot.as_of,
            instrument_uid=snapshot.instrument_uid,
            action=Action.ENTER,
            expected_edge_bps=DECLARED_EDGE_BPS,
            feature_snapshot_hash=snapshot.snapshot_hash,
            strategy_id=self.strategy_id,
            strategy_version=self.version,
            rationale=f"cross up: fast {fast} above slow {slow}",
            evidence={"fast": str(fast), "slow": str(slow)},
        )

    def wants_to_exit(self, *, position: PositionState, at: datetime) -> bool:
        """Whether the minimum hold has elapsed.

        Asked by the loop rather than folded into `decide`, because the
        minimum hold is enforced by the *risk* layer — this is the strategy's
        own view of it, and the two disagreeing is worth being able to see.
        """
        held = position.holding_minutes_at(at)
        return held is not None and held >= self.min_hold_minutes
