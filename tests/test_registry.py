"""The registry: identity, ancestry, and the budget a rename cannot escape."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from decimal import Decimal

import pytest

from tb.ledger.events import EventType
from tb.ledger.store import Ledger
from tb.registry.lineage import SpecRegistry, lineage_roots
from tb.registry.models import (
    AuthorKind,
    RegistryError,
    StrategyStatus,
)
from tb.strategy.dsl.schema import StrategySpec

AS_OF = datetime(2026, 6, 1, 14, 0, tzinfo=UTC)
BUDGET = Decimal("100.00")


def spec(name: str = "cross", *, edge: str = "250", lookback: int = 20) -> StrategySpec:
    return StrategySpec.model_validate(
        {
            "name": name,
            "entry": {
                "kind": "compare",
                "op": "gt",
                "left": {"kind": "feature", "name": "sma", "lookback": lookback},
                "right": {"kind": "feature", "name": "sma", "lookback": 100},
            },
            "exit": {
                "kind": "compare",
                "op": "lt",
                "left": {"kind": "feature", "name": "sma", "lookback": lookback},
                "right": {"kind": "feature", "name": "sma", "lookback": 100},
            },
            "expected_edge_bps": edge,
            "min_holding_minutes": 1440,
        }
    )


@pytest.fixture
def registry(ledger: Ledger) -> SpecRegistry:
    return SpecRegistry(ledger, per_lineage_budget_ccy=BUDGET, run_id="run_test")


# --------------------------------------------------------------------------
# Registration
# --------------------------------------------------------------------------


def test_registering_a_spec_writes_the_row_and_the_event(
    registry: SpecRegistry, ledger: Ledger
) -> None:
    registered = registry.register(spec(), author_kind=AuthorKind.HUMAN, at=AS_OF)

    assert registered.strategy_id.startswith("stg_")
    assert registered.version == 1
    assert registered.generation == 0

    row = ledger.conn.execute(
        "SELECT * FROM strategy_specs WHERE strategy_id = ?", (registered.strategy_id,)
    ).fetchone()
    assert row["spec_hash"] == registered.spec_hash
    assert row["author_kind"] == "human"

    events = ledger.conn.execute(
        "SELECT COUNT(*) FROM event_log WHERE event_type = ?",
        (EventType.STRATEGY_SPEC_REGISTERED.value,),
    ).fetchone()
    assert events[0] == 1


def test_the_same_tree_registers_once(registry: SpecRegistry) -> None:
    """Deduplication by spec hash, which the trial count depends on.

    The case this models is a mutation that happens to reproduce its parent —
    a searcher inverting its own change, which happens constantly. Without the
    hash check that lineage's apparent diversity would be twice its real
    diversity, and the trial count is what every deflated metric divides by.

    Note what this does *not* claim: `name` is part of the hashed spec, so two
    genuinely different names are two strategies. That is the right reading —
    the name rides into the ledger and a spec is the thing that was recorded,
    not an equivalence class over it.
    """
    first = registry.register(spec("alpha"), author_kind=AuthorKind.SEARCH, at=AS_OF)
    again = registry.register(spec("alpha"), author_kind=AuthorKind.SEARCH, at=AS_OF)

    assert again.strategy_id == first.strategy_id
    assert again.spec_hash == first.spec_hash

    renamed = registry.register(spec("beta"), author_kind=AuthorKind.SEARCH, at=AS_OF)
    assert renamed.strategy_id != first.strategy_id


def test_a_spec_is_registered_as_a_candidate_and_nothing_here_can_promote_it(
    registry: SpecRegistry,
) -> None:
    registered = registry.register(spec(), author_kind=AuthorKind.SEARCH, at=AS_OF)
    record = registry.status_of(registered.strategy_id)
    assert record is not None
    assert record.status is StrategyStatus.CANDIDATE
    assert not record.may_trade

    allowed, why = registry.may_trade(registered.strategy_id)
    assert not allowed
    assert "candidate" in why


def test_an_unregistered_strategy_may_not_trade(registry: SpecRegistry) -> None:
    """Fail-closed. An unknown strategy is a code path that skipped the gate."""
    allowed, why = registry.may_trade("stg_nonexistent")
    assert not allowed
    assert "not in the registry" in why


def test_a_child_inherits_its_parents_lineage(registry: SpecRegistry) -> None:
    parent = registry.register(spec("parent"), author_kind=AuthorKind.HUMAN, at=AS_OF)
    child = registry.register(
        spec("child", lookback=30),
        author_kind=AuthorKind.MUTATION,
        parent_strategy_id=parent.strategy_id,
        at=AS_OF,
    )
    assert child.lineage_id == parent.lineage_id
    assert child.generation == 1
    assert child.parent_strategy_id == parent.strategy_id


def test_a_grandchild_counts_two_generations(registry: SpecRegistry) -> None:
    parent = registry.register(spec("p"), author_kind=AuthorKind.HUMAN, at=AS_OF)
    child = registry.register(
        spec("c", lookback=30),
        author_kind=AuthorKind.MUTATION,
        parent_strategy_id=parent.strategy_id,
        at=AS_OF,
    )
    grandchild = registry.register(
        spec("g", lookback=40),
        author_kind=AuthorKind.MUTATION,
        parent_strategy_id=child.strategy_id,
        at=AS_OF,
    )
    assert grandchild.generation == 2
    assert grandchild.lineage_id == parent.lineage_id


def test_a_child_cannot_choose_its_own_lineage(registry: SpecRegistry) -> None:
    """The hole this closes: a mutation walking away from an exhausted budget."""
    parent = registry.register(spec("p"), author_kind=AuthorKind.HUMAN, at=AS_OF)
    with pytest.raises(RegistryError, match="cannot choose its own lineage"):
        registry.register(
            spec("c", lookback=30),
            author_kind=AuthorKind.MUTATION,
            parent_strategy_id=parent.strategy_id,
            lineage_id="lin_somewhere_else",
            at=AS_OF,
        )


def test_an_unregistered_parent_is_refused(registry: SpecRegistry) -> None:
    with pytest.raises(RegistryError, match="is not registered"):
        registry.register(
            spec(),
            author_kind=AuthorKind.MUTATION,
            parent_strategy_id="stg_ghost",
            at=AS_OF,
        )


def test_the_stored_spec_round_trips_through_validation(registry: SpecRegistry) -> None:
    original = spec("round-trip")
    registered = registry.register(original, author_kind=AuthorKind.HUMAN, at=AS_OF)
    loaded = registry.spec_of(registered.strategy_id)
    assert loaded is not None
    assert loaded.spec_hash == original.spec_hash


def test_lineage_roots_picks_the_earliest_registration(registry: SpecRegistry) -> None:
    first = registry.register(spec("a"), author_kind=AuthorKind.HUMAN, at=AS_OF)
    second = registry.register(
        spec("b", lookback=30),
        author_kind=AuthorKind.MUTATION,
        parent_strategy_id=first.strategy_id,
        at=AS_OF + timedelta(days=1),
    )
    roots = lineage_roots([second, first])
    assert roots[first.lineage_id].strategy_id == first.strategy_id


# --------------------------------------------------------------------------
# Lineage budgets
# --------------------------------------------------------------------------


def test_registering_opens_a_budget_at_the_configured_ceiling(
    registry: SpecRegistry,
) -> None:
    registered = registry.register(spec(), author_kind=AuthorKind.HUMAN, at=AS_OF)
    budget = registry.budget_for(registered.lineage_id)
    assert budget is not None
    assert budget.budget_ccy == BUDGET
    assert budget.consumed_ccy == Decimal(0)
    assert not budget.is_exhausted
    assert budget.n_strategies == 1


def test_charging_a_loss_consumes_the_budget(registry: SpecRegistry) -> None:
    registered = registry.register(spec(), author_kind=AuthorKind.HUMAN, at=AS_OF)
    budget = registry.charge(
        registered.lineage_id, loss_ccy=Decimal("30"), strategy_id=registered.strategy_id
    )
    assert budget.consumed_ccy == Decimal("30")
    assert budget.remaining_ccy == Decimal("70")
    assert budget.fraction_used == Decimal("0.3")


def test_a_negative_loss_is_refused(registry: SpecRegistry) -> None:
    """A signed P&L here would let a profitable lineage raise its own ceiling."""
    registered = registry.register(spec(), author_kind=AuthorKind.HUMAN, at=AS_OF)
    with pytest.raises(RegistryError, match="negative loss"):
        registry.charge(registered.lineage_id, loss_ccy=Decimal("-50"))


def test_exhausting_the_budget_blocks_every_member_including_a_promoted_one(
    registry: SpecRegistry, ledger: Ledger
) -> None:
    parent = registry.register(spec("p"), author_kind=AuthorKind.HUMAN, at=AS_OF)
    child = registry.register(
        spec("c", lookback=30),
        author_kind=AuthorKind.MUTATION,
        parent_strategy_id=parent.strategy_id,
        at=AS_OF,
    )
    # Promote both, the way the gate would.
    ledger.conn.execute(
        "UPDATE strategy_status SET status = ?, promoted_at = ?",
        (StrategyStatus.PROMOTED.value, AS_OF.isoformat()),
    )
    ledger.conn.commit()
    assert registry.may_trade(parent.strategy_id)[0]

    registry.charge(parent.lineage_id, loss_ccy=BUDGET, strategy_id=parent.strategy_id, at=AS_OF)

    for strategy in (parent, child):
        record = registry.status_of(strategy.strategy_id)
        assert record is not None
        assert record.status is StrategyStatus.BLOCKED
        assert not registry.may_trade(strategy.strategy_id)[0]

    events = ledger.conn.execute(
        "SELECT COUNT(*) FROM event_log WHERE event_type = ?",
        (EventType.LINEAGE_BUDGET_EXHAUSTED.value,),
    ).fetchone()
    assert events[0] == 1


def test_a_child_registered_after_exhaustion_cannot_escape_it(
    registry: SpecRegistry, ledger: Ledger
) -> None:
    """The reason the budget is per lineage rather than per strategy.

    A searcher's default response to a retirement is to propose a mutation. If
    the budget were per strategy that mutation would start fresh, and the
    lineage would keep losing money one rename at a time.

    The child is promoted here deliberately. A child left as a candidate is
    refused for being a candidate, which would make this test pass without ever
    reaching the budget check it exists to exercise.
    """
    parent = registry.register(spec("p"), author_kind=AuthorKind.HUMAN, at=AS_OF)
    registry.charge(parent.lineage_id, loss_ccy=BUDGET, at=AS_OF)

    child = registry.register(
        spec("c", lookback=30),
        author_kind=AuthorKind.MUTATION,
        parent_strategy_id=parent.strategy_id,
        at=AS_OF + timedelta(days=1),
    )
    ledger.conn.execute(
        "UPDATE strategy_status SET status = ?, promoted_at = ? WHERE strategy_id = ?",
        (StrategyStatus.PROMOTED.value, AS_OF.isoformat(), child.strategy_id),
    )
    ledger.conn.commit()

    allowed, why = registry.may_trade(child.strategy_id)
    assert not allowed
    assert "spent its loss budget" in why

    budget = registry.budget_for(child.lineage_id)
    assert budget is not None
    assert budget.is_exhausted


def test_exhaustion_is_announced_once_not_on_every_further_loss(
    registry: SpecRegistry, ledger: Ledger
) -> None:
    registered = registry.register(spec(), author_kind=AuthorKind.HUMAN, at=AS_OF)
    registry.charge(registered.lineage_id, loss_ccy=BUDGET, at=AS_OF)
    registry.charge(registered.lineage_id, loss_ccy=Decimal("10"), at=AS_OF)

    events = ledger.conn.execute(
        "SELECT COUNT(*) FROM event_log WHERE event_type = ?",
        (EventType.LINEAGE_BUDGET_EXHAUSTED.value,),
    ).fetchone()
    assert events[0] == 1


def test_recording_a_realised_loss_charges_the_lineage(registry: SpecRegistry) -> None:
    """One call, because two that can drift would leave a budget that never depletes."""
    registered = registry.register(spec(), author_kind=AuthorKind.HUMAN, at=AS_OF)
    record = registry.record_realised(
        registered.strategy_id, pnl_ccy=Decimal("-25"), n_trades=1, at=AS_OF
    )
    assert record.realised_pnl_ccy == Decimal("-25")
    assert record.n_realised_trades == 1

    budget = registry.budget_for(registered.lineage_id)
    assert budget is not None
    assert budget.consumed_ccy == Decimal("25")


def test_a_realised_profit_does_not_refund_the_budget(registry: SpecRegistry) -> None:
    registered = registry.register(spec(), author_kind=AuthorKind.HUMAN, at=AS_OF)
    registry.record_realised(registered.strategy_id, pnl_ccy=Decimal("-40"), at=AS_OF)
    registry.record_realised(registered.strategy_id, pnl_ccy=Decimal("60"), at=AS_OF)

    record = registry.status_of(registered.strategy_id)
    assert record is not None
    assert record.realised_pnl_ccy == Decimal("20")

    budget = registry.budget_for(registered.lineage_id)
    assert budget is not None
    # The profit shows on the strategy; the budget still remembers the loss.
    assert budget.consumed_ccy == Decimal("40")


# --------------------------------------------------------------------------
# Retirement
# --------------------------------------------------------------------------


def test_retiring_a_strategy_records_the_reason(registry: SpecRegistry, ledger: Ledger) -> None:
    registered = registry.register(spec(), author_kind=AuthorKind.HUMAN, at=AS_OF)
    record = registry.retire(registered.strategy_id, reason="decayed", at=AS_OF)

    assert record.status is StrategyStatus.RETIRED
    assert record.retire_reason == "decayed"
    assert record.retired_at == AS_OF

    events = ledger.conn.execute(
        "SELECT COUNT(*) FROM event_log WHERE event_type = ?",
        (EventType.STRATEGY_RETIRED.value,),
    ).fetchone()
    assert events[0] == 1


def test_retire_refuses_to_set_a_trading_status(registry: SpecRegistry) -> None:
    """The one path out must not be usable as a path in."""
    registered = registry.register(spec(), author_kind=AuthorKind.HUMAN, at=AS_OF)
    with pytest.raises(RegistryError, match="permits trading"):
        registry.retire(registered.strategy_id, reason="oops", status=StrategyStatus.PROMOTED)


def test_a_retired_strategy_is_not_re_blocked_by_a_later_exhaustion(
    registry: SpecRegistry,
) -> None:
    """Retired and blocked say different things; the first must not be overwritten."""
    registered = registry.register(spec(), author_kind=AuthorKind.HUMAN, at=AS_OF)
    registry.retire(registered.strategy_id, reason="decayed", at=AS_OF)
    registry.charge(registered.lineage_id, loss_ccy=BUDGET, at=AS_OF)

    record = registry.status_of(registered.strategy_id)
    assert record is not None
    assert record.status is StrategyStatus.RETIRED
    assert record.retire_reason == "decayed"


def test_only_promoted_permits_trading(registry: SpecRegistry) -> None:
    """Written as an equality so a sixth status defaults to not trading."""
    permitted = [status for status in StrategyStatus if status.may_trade]
    assert permitted == [StrategyStatus.PROMOTED]


def test_promoted_lists_only_live_strategies(registry: SpecRegistry, ledger: Ledger) -> None:
    live = registry.register(spec("live"), author_kind=AuthorKind.HUMAN, at=AS_OF)
    registry.register(spec("dead", lookback=30), author_kind=AuthorKind.HUMAN, at=AS_OF)
    ledger.conn.execute(
        "UPDATE strategy_status SET status = ?, promoted_at = ? WHERE strategy_id = ?",
        (StrategyStatus.PROMOTED.value, AS_OF.isoformat(), live.strategy_id),
    )
    ledger.conn.commit()

    records = registry.promoted()
    assert [r.strategy_id for r in records] == [live.strategy_id]
