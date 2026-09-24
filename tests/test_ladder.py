"""The size ratchet: up slowly, down fast, never past the hard ceiling."""

from __future__ import annotations

from datetime import UTC, datetime
from decimal import Decimal

import pytest

from tb.config.loader import PinnedLimits
from tb.ledger.events import EventType
from tb.ledger.store import Ledger
from tb.registry.ladder import (
    MIN_TRADES_PER_RUNG,
    RUNG_MULTIPLIER,
    BreachReason,
    Direction,
    LadderError,
    RungEvidence,
    SizeLadder,
    notional_for,
)
from tb.registry.lineage import SpecRegistry
from tb.registry.models import AuthorKind
from tb.strategy.dsl.schema import StrategySpec

AS_OF = datetime(2026, 6, 1, 14, 0, tzinfo=UTC)
EQUITY = Decimal("20000")


def a_spec() -> StrategySpec:
    return StrategySpec.model_validate(
        {
            "name": "cross",
            "entry": {
                "kind": "compare",
                "op": "gt",
                "left": {"kind": "feature", "name": "sma", "lookback": 20},
                "right": {"kind": "feature", "name": "sma", "lookback": 100},
            },
            "exit": {
                "kind": "compare",
                "op": "lt",
                "left": {"kind": "feature", "name": "sma", "lookback": 20},
                "right": {"kind": "feature", "name": "sma", "lookback": 100},
            },
            "expected_edge_bps": "300",
            "min_holding_minutes": 1440,
        }
    )


def earned(days: int = 10, trades: int = 8, pnl: str = "12") -> RungEvidence:
    """Evidence that justifies a rung-up."""
    return RungEvidence(
        days_at_rung=days, n_trades_at_rung=trades, realised_pnl_at_rung=Decimal(pnl)
    )


@pytest.fixture
def ladder(ledger: Ledger, pinned: PinnedLimits) -> SizeLadder:
    return SizeLadder(ledger, limits=pinned.limits, run_id="run_test")


@pytest.fixture
def strategy_id(ledger: Ledger) -> str:
    registry = SpecRegistry(ledger, per_lineage_budget_ccy=Decimal("100"))
    return registry.register(a_spec(), author_kind=AuthorKind.SEARCH, at=AS_OF).strategy_id


# --------------------------------------------------------------------------
# Sizing
# --------------------------------------------------------------------------


def test_rung_zero_is_the_floor_notional(pinned: PinnedLimits) -> None:
    floor = pinned.limits.capital.floor_notional_ccy
    assert notional_for(0, limits=pinned.limits, equity_ccy=EQUITY) == floor


def test_each_rung_doubles(pinned: PinnedLimits) -> None:
    floor = pinned.limits.capital.floor_notional_ccy
    for rung in range(3):
        expected = floor * Decimal(RUNG_MULTIPLIER**rung)
        assert notional_for(rung, limits=pinned.limits, equity_ccy=EQUITY) == expected


def test_the_per_position_cap_binds_before_the_ladder_does(
    pinned: PinnedLimits,
) -> None:
    """The ladder moves a strategy toward the ceilings and can never move it
    past one. Raising a ceiling is a human edit to a hash-pinned file."""
    small_account = Decimal("2000")
    cap = small_account * Decimal(str(pinned.limits.capital.per_position_pct)) / Decimal(100)
    assert notional_for(4, limits=pinned.limits, equity_ccy=small_account) == cap


def test_the_absolute_ceiling_binds_whatever_the_equity(pinned: PinnedLimits) -> None:
    huge = notional_for(20, limits=pinned.limits, equity_ccy=Decimal("10000000"))
    assert huge == pinned.limits.capital.absolute_ceiling_ccy


def test_an_unknown_equity_is_not_treated_as_unlimited(pinned: PinnedLimits) -> None:
    """With no equity observation the percentage cap cannot be evaluated, so
    only the rung size and the absolute ceiling apply — both of which are
    already bounds."""
    without = notional_for(2, limits=pinned.limits, equity_ccy=None)
    assert without <= pinned.limits.capital.absolute_ceiling_ccy
    assert without == pinned.limits.capital.floor_notional_ccy * Decimal(RUNG_MULTIPLIER**2)


def test_a_negative_rung_is_refused(pinned: PinnedLimits) -> None:
    with pytest.raises(LadderError, match="must not be negative"):
        notional_for(-1, limits=pinned.limits)


# --------------------------------------------------------------------------
# Climbing
# --------------------------------------------------------------------------


def test_earned_evidence_climbs_one_rung(ladder: SizeLadder) -> None:
    direction, target, _ = ladder.propose(current_rung=0, evidence=earned())
    assert direction is Direction.UP
    assert target == 1


def test_days_alone_do_not_earn_a_rung(ladder: SizeLadder) -> None:
    """A strategy that signalled nothing for a fortnight has shown nothing."""
    direction, target, reason = ladder.propose(
        current_rung=0, evidence=earned(days=30, trades=MIN_TRADES_PER_RUNG - 1)
    )
    assert direction is Direction.HOLD
    assert target == 0
    assert "days alone are not evidence" in reason


def test_trades_alone_do_not_earn_a_rung(ladder: SizeLadder, pinned: PinnedLimits) -> None:
    minimum = pinned.limits.promotion.ratchet_min_days_between_promotions
    direction, _, reason = ladder.propose(
        current_rung=0, evidence=earned(days=minimum - 1, trades=50)
    )
    assert direction is Direction.HOLD
    assert "required" in reason


def test_surviving_is_not_the_same_as_holding_up(ladder: SizeLadder) -> None:
    """A rung is earned by holding up at this size, not by getting through it."""
    direction, _, reason = ladder.propose(current_rung=0, evidence=earned(pnl="0"))
    assert direction is Direction.HOLD
    assert "not by surviving it" in reason


def test_the_maximum_rung_is_a_ceiling(ladder: SizeLadder, pinned: PinnedLimits) -> None:
    top = pinned.limits.promotion.ratchet_max_rung
    direction, target, reason = ladder.propose(current_rung=top, evidence=earned())
    assert direction is Direction.HOLD
    assert target == top
    assert "maximum rung" in reason


# --------------------------------------------------------------------------
# Falling
# --------------------------------------------------------------------------


def test_a_breach_drops_more_rungs_than_a_promotion_gains(
    ladder: SizeLadder, pinned: PinnedLimits
) -> None:
    """The asymmetry, stated as a comparison rather than as two constants.

    A symmetric ladder leaves a strategy oscillating around its threshold
    sitting at maximum size half the time.
    """
    up_direction, up_target, _ = ladder.propose(current_rung=2, evidence=earned())
    down_direction, down_target, _ = ladder.propose(
        current_rung=2,
        evidence=RungEvidence(
            days_at_rung=1,
            n_trades_at_rung=1,
            realised_pnl_at_rung=Decimal("-5"),
            breached=True,
            breach_reason=BreachReason.REALISED_LOSS,
        ),
    )
    assert up_direction is Direction.UP
    assert down_direction is Direction.DOWN
    gained = up_target - 2
    lost = 2 - down_target
    assert lost > gained
    assert lost == pinned.limits.promotion.ratchet_rungs_lost_on_breach


def test_a_breach_overrides_good_evidence(ladder: SizeLadder) -> None:
    """Down is checked first on purpose: a strategy that breached this period
    does not climb on the strength of the trades before the breach."""
    evidence = RungEvidence(
        days_at_rung=30,
        n_trades_at_rung=50,
        realised_pnl_at_rung=Decimal("100"),
        breached=True,
        breach_reason=BreachReason.DRAWDOWN,
    )
    direction, target, _ = ladder.propose(current_rung=3, evidence=evidence)
    assert direction is Direction.DOWN
    assert target == 1


def test_a_breach_cannot_go_below_the_floor_rung(ladder: SizeLadder) -> None:
    direction, target, _ = ladder.propose(
        current_rung=1,
        evidence=RungEvidence(
            days_at_rung=1,
            n_trades_at_rung=1,
            realised_pnl_at_rung=Decimal("-5"),
            breached=True,
            breach_reason=BreachReason.RISK_BLOCK,
        ),
    )
    assert direction is Direction.DOWN
    assert target == 0

    at_floor = ladder.propose(
        current_rung=0,
        evidence=RungEvidence(
            days_at_rung=1,
            n_trades_at_rung=1,
            realised_pnl_at_rung=Decimal("-5"),
            breached=True,
        ),
    )
    assert at_floor[0] is Direction.HOLD
    assert "cannot demote further" in at_floor[2]


# --------------------------------------------------------------------------
# Persistence
# --------------------------------------------------------------------------


def test_a_move_writes_the_rung_the_row_and_the_event(
    ladder: SizeLadder, ledger: Ledger, strategy_id: str
) -> None:
    move = ladder.review(strategy_id, evidence=earned(), equity_ccy=EQUITY, at=AS_OF)
    assert move is not None
    assert move.from_rung == 0
    assert move.to_rung == 1
    assert move.rungs_moved == 1

    assert ladder.rung_of(strategy_id) == 1
    row = ledger.conn.execute("SELECT * FROM ladder_moves").fetchone()
    assert row["direction"] == "up"

    events = ledger.conn.execute(
        "SELECT COUNT(*) FROM event_log WHERE event_type = ?",
        (EventType.LADDER_MOVED.value,),
    ).fetchone()
    assert events[0] == 1


def test_a_hold_writes_nothing(ladder: SizeLadder, ledger: Ledger, strategy_id: str) -> None:
    assert ladder.review(strategy_id, evidence=earned(trades=1), at=AS_OF) is None
    rows = ledger.conn.execute("SELECT COUNT(*) FROM ladder_moves").fetchone()
    assert rows[0] == 0


def test_climbing_then_breaching_lands_below_where_it_started(
    ladder: SizeLadder, strategy_id: str
) -> None:
    """Two rungs up over two months, one breach, and it is back at the floor."""
    ladder.review(strategy_id, evidence=earned(), equity_ccy=EQUITY, at=AS_OF)
    ladder.review(strategy_id, evidence=earned(), equity_ccy=EQUITY, at=AS_OF)
    assert ladder.rung_of(strategy_id) == 2

    ladder.breach(strategy_id, reason=BreachReason.DRAWDOWN, equity_ccy=EQUITY, at=AS_OF)
    assert ladder.rung_of(strategy_id) == 0


def test_a_breach_at_the_floor_records_nothing(
    ladder: SizeLadder, ledger: Ledger, strategy_id: str
) -> None:
    """A move from 0 to 0 would be a meaningless row in the history operators
    read to see how often a strategy breaches."""
    assert ladder.breach(strategy_id, reason=BreachReason.DECAY, at=AS_OF) is None
    rows = ledger.conn.execute("SELECT COUNT(*) FROM ladder_moves").fetchone()
    assert rows[0] == 0


def test_the_notional_follows_the_rung(
    ladder: SizeLadder, strategy_id: str, pinned: PinnedLimits
) -> None:
    floor = pinned.limits.capital.floor_notional_ccy
    assert ladder.notional_of(strategy_id, equity_ccy=EQUITY) == floor
    ladder.review(strategy_id, evidence=earned(), equity_ccy=EQUITY, at=AS_OF)
    assert ladder.notional_of(strategy_id, equity_ccy=EQUITY) == floor * RUNG_MULTIPLIER


def test_an_unregistered_strategy_has_no_rung(ladder: SizeLadder) -> None:
    """Sizing one would mean funding something that never passed the gate."""
    with pytest.raises(LadderError, match="not in the registry"):
        ladder.rung_of("stg_ghost")


def test_the_history_records_every_move_in_order(ladder: SizeLadder, strategy_id: str) -> None:
    ladder.review(strategy_id, evidence=earned(), equity_ccy=EQUITY, at=AS_OF)
    ladder.review(strategy_id, evidence=earned(), equity_ccy=EQUITY, at=AS_OF)
    ladder.breach(strategy_id, reason=BreachReason.RISK_BLOCK, at=AS_OF)

    moves = ladder.history(strategy_id)
    assert [(m.from_rung, m.to_rung) for m in moves] == [(0, 1), (1, 2), (2, 0)]
    assert [m.direction for m in moves] == [Direction.UP, Direction.UP, Direction.DOWN]
