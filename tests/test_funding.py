"""Funding the live loop: what a promotion actually changes.

The gap this closes is the one M5 left. The registry, the gate, the ladder and
the allocator all wrote their decisions into their own tables, and the loop held
one hardcoded strategy and read none of them — so every one of those controls
was true of the database and false of the account. The headline test here is
`test_a_promotion_changes_what_the_loop_trades`, which asserts the difference
end to end: the same loop, the same bars, the same broker, promoted versus not.

The rest divide into three groups:

* the book — who is funded, at what size, and why someone was left out;
* ownership — which strategy's entry opened a position, resolved from the
  decision lineage rather than from the broker, which knows nothing about
  strategies;
* the risk rule — the allocation as a verdict row, fail-closed on absence.

The loop-level helpers are imported from `tests.test_loop` rather than copied:
they are the same bars, the same verified symbol and the same simulated broker,
and two divergent copies of "a working universe" is how a test suite ends up
proving something about a fixture instead of about the system. The `env` fixture
they both use lives in `conftest.py` for the same reason.
"""

from __future__ import annotations

from dataclasses import replace
from datetime import datetime, timedelta
from decimal import Decimal
from typing import Any

import pytest
from typer.testing import CliRunner

from tb.broker.port import OrderPurpose, Side
from tb.broker.simulated import SimulatedBroker
from tb.cli import app
from tb.config.loader import PinnedLimits
from tb.engine.funding import (
    explicit_book,
    funded_book,
    owner_of,
    record_book,
    unfunded_notional,
    unowned_positions,
)
from tb.engine.loop import CycleResult, TradingLoop
from tb.ledger.store import Ledger
from tb.portfolio.allocator import Allocator, StrategyInput
from tb.registry.ladder import notional_for
from tb.registry.lineage import SpecRegistry
from tb.registry.models import AuthorKind, StrategyStatus
from tb.risk.rules.capital import StrategyAllocationRule
from tb.risk.state import AccountState, OrderRequest, RiskContext, Verdict
from tb.strategy.base import Action
from tb.strategy.dsl.ops import DslStrategy, pipeline_from_spec
from tb.strategy.dsl.schema import StrategySpec
from tests.test_loop import AS_OF, TICKER, UID, _broker, _loop, _own, _rising_bars, _seed

LINEAGE_BUDGET = Decimal("100.00")


def a_spec(*, name: str = "cross", edge: str = "450") -> StrategySpec:
    """A spec that fires on the rising bars the loop fixtures seed.

    The same shape as the trivial strategy's rule — fast average above slow —
    so a monotonically rising series enters deterministically. A spec whose
    entry depended on a cross happening at a particular bar would be a test of
    the fixture.
    """
    return StrategySpec.model_validate(
        {
            "name": name,
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
            "expected_edge_bps": edge,
            "min_holding_minutes": 1440,
        }
    )


def _registry(ledger: Ledger) -> SpecRegistry:
    return SpecRegistry(ledger, per_lineage_budget_ccy=LINEAGE_BUDGET, run_id="run_funding")


def _promote(
    ledger: Ledger,
    *,
    spec: StrategySpec | None = None,
    rung: int = 0,
    at: datetime = AS_OF,
) -> tuple[str, str]:
    """Register a spec and mark it promoted. Returns `(strategy_id, lineage_id)`.

    The status is written directly rather than through `PromotionGate`, and the
    reason is scope: the gate has its own suite, which breaks one piece of
    evidence at a time, and driving it from here would make every funding test
    depend on a full evidence fixture. What these tests are about is what the
    loop does with a promotion, not how one is earned — and
    `tests/test_promotion.py` already asserts that nothing but the gate writes
    PROMOTED.
    """
    registered = _registry(ledger).register(spec or a_spec(), author_kind=AuthorKind.SEARCH, at=at)
    ledger.conn.execute(
        "UPDATE strategy_status SET status = ?, rung = ?, promoted_at = ?, updated_at = ? "
        "WHERE strategy_id = ? AND version = ?",
        (
            StrategyStatus.PROMOTED.value,
            rung,
            at.isoformat(),
            at.isoformat(),
            registered.strategy_id,
            registered.version,
        ),
    )
    ledger.conn.commit()
    return registered.strategy_id, registered.lineage_id


# --------------------------------------------------------------------------
# The book
# --------------------------------------------------------------------------


def test_an_empty_registry_funds_nothing(ledger: Ledger, pinned: PinnedLimits) -> None:
    book = funded_book(ledger, limits=pinned.limits, equity_ccy=Decimal("10000"), at=AS_OF)
    assert not book
    assert book.funded == ()
    assert "nothing is funded" in book.explain()


def test_a_candidate_is_not_funded(ledger: Ledger, pinned: PinnedLimits) -> None:
    """Registration funds nothing. Only a promotion does.

    The property that makes "how did this get funded" have one answer: a
    searcher can register as many specs as it likes and none of them reaches the
    loop until the gate says so.
    """
    _registry(ledger).register(a_spec(), author_kind=AuthorKind.SEARCH, at=AS_OF)
    book = funded_book(ledger, limits=pinned.limits, equity_ccy=Decimal("10000"), at=AS_OF)
    assert not book.funded


def test_a_promoted_strategy_is_funded_at_its_rung(ledger: Ledger, pinned: PinnedLimits) -> None:
    """With no allocation round yet, the ladder's rung is the size.

    This is the plan's "live at floor notional as soon as it clears the gate":
    the rung notional is already a bound — floor doubled per rung, intersected
    with the per-position cap and the absolute ceiling — so waiting for an
    allocator run would mean a gate-cleared strategy sitting idle.
    """
    equity = Decimal("10000")
    strategy_id, lineage_id = _promote(ledger, rung=0)

    book = funded_book(ledger, limits=pinned.limits, equity_ccy=equity, at=AS_OF)

    assert [s.strategy_id for s in book.funded] == [strategy_id]
    funded = book.funded[0]
    assert funded.notional_ccy == pinned.limits.capital.floor_notional_ccy
    assert funded.rung == 0
    assert funded.lineage_id == lineage_id
    assert "rung 0" in funded.detail


def test_the_rung_is_what_makes_a_position_bigger(ledger: Ledger, pinned: PinnedLimits) -> None:
    """**The ratchet, as a number the order path reads.**

    Before this wiring a promotion at rung 0 and one at rung 3 produced the
    same order, because nothing outside `strategy_status` read the rung. Here
    the same strategy at a higher rung is funded at eight times the size — and
    still under the per-position cap, which is what stops the ladder enlarging
    the blast radius.
    """
    equity = Decimal("100000")
    strategy_id, _ = _promote(ledger, rung=0)
    floor_book = funded_book(ledger, limits=pinned.limits, equity_ccy=equity, at=AS_OF)

    ledger.conn.execute("UPDATE strategy_status SET rung = 3 WHERE strategy_id = ?", (strategy_id,))
    ledger.conn.commit()
    climbed = funded_book(ledger, limits=pinned.limits, equity_ccy=equity, at=AS_OF)

    assert floor_book.funded[0].notional_ccy == pinned.limits.capital.floor_notional_ccy
    assert climbed.funded[0].notional_ccy == pinned.limits.capital.floor_notional_ccy * 8
    assert climbed.funded[0].notional_ccy <= notional_for(
        3, limits=pinned.limits, equity_ccy=equity
    )


def test_an_allocation_below_the_rung_wins(ledger: Ledger, pinned: PinnedLimits) -> None:
    """The rung is a ceiling, the allocation is a share, and the tighter binds.

    A strategy at rung 4 whose blended edge earned it a small share of the
    portfolio gets the small share. The ladder says how large a position *may*
    be; it does not say the portfolio has to hold one that size.
    """
    equity = Decimal("100000")
    strategy_id, lineage_id = _promote(ledger, rung=4)
    Allocator(ledger, limits=pinned.limits, run_id="run_funding").allocate(
        strategies=[
            StrategyInput(
                strategy_id=strategy_id,
                version=1,
                lineage_id=lineage_id,
                rung=4,
                prior_edge_bps=Decimal("200"),
            ),
            StrategyInput(
                strategy_id="stg_other",
                version=1,
                lineage_id="lin_other",
                rung=4,
                prior_edge_bps=Decimal("1800"),
            ),
        ],
        equity_ccy=equity,
        at=AS_OF - timedelta(minutes=5),
    )

    book = funded_book(ledger, limits=pinned.limits, equity_ccy=equity, at=AS_OF)

    funded = book.funded[0]
    rung_cap = notional_for(4, limits=pinned.limits, equity_ccy=equity)
    assert funded.notional_ccy < rung_cap, "the allocator's share should bind here"
    assert "allocation" in funded.detail
    assert "weight" in funded.detail


def test_a_zero_allocation_stays_in_the_book(ledger: Ledger, pinned: PinnedLimits) -> None:
    """**A strategy is never dropped for a reason of size.**

    An allocator that believes nothing about a strategy gives it nothing, and
    the risk rule then refuses its entries. It must still be *in* the book:
    dropping it would strand whatever it already holds, and "the strategy whose
    allocation fell to zero cannot get out" is the asymmetry this system exists
    to avoid.
    """
    equity = Decimal("10000")
    strategy_id, lineage_id = _promote(ledger, rung=0)
    Allocator(ledger, limits=pinned.limits, run_id="run_funding").allocate(
        strategies=[
            StrategyInput(
                strategy_id=strategy_id,
                version=1,
                lineage_id=lineage_id,
                rung=0,
                # A non-positive blended edge earns zero weight: long-only and
                # unlevered, so there is no way to express disbelief except by
                # not funding it.
                prior_edge_bps=Decimal("-50"),
            )
        ],
        equity_ccy=equity,
        at=AS_OF - timedelta(minutes=5),
    )

    book = funded_book(ledger, limits=pinned.limits, equity_ccy=equity, at=AS_OF)

    assert [s.strategy_id for s in book.funded] == [strategy_id]
    assert book.funded[0].notional_ccy == Decimal(0)


def test_an_exhausted_lineage_budget_unfunds_a_promoted_strategy(
    ledger: Ledger, pinned: PinnedLimits
) -> None:
    """The budget a rename cannot escape, applied at funding time.

    Charged past its budget, every member of the lineage is blocked — so the
    status is no longer PROMOTED and the book says why rather than silently
    shrinking.
    """
    strategy_id, lineage_id = _promote(ledger, rung=0)
    registry = _registry(ledger)
    registry.charge(lineage_id, loss_ccy=LINEAGE_BUDGET, strategy_id=strategy_id, at=AS_OF)

    book = funded_book(ledger, limits=pinned.limits, equity_ccy=Decimal("10000"), at=AS_OF)

    assert not book.funded
    assert book.excluded, "an excluded strategy must say why it was left out"
    assert "blocked" in book.excluded[0][1]


def test_a_spec_that_no_longer_parses_unfunds_one_strategy_not_the_book(
    ledger: Ledger, pinned: PinnedLimits
) -> None:
    """One stale row must not stop a whole account from trading.

    `spec_of` re-validates on the way out rather than trusting the stored JSON,
    which is right — a spec that no longer parses against the current grammar
    must not be interpreted approximately. But raising from here would let one
    strategy written by an older build unfund every other strategy too, and the
    fix for that row is a human's.
    """
    broken, _ = _promote(ledger, spec=a_spec(name="broken"))
    healthy, _ = _promote(ledger, spec=a_spec(name="healthy"))
    ledger.conn.execute(
        "UPDATE strategy_specs SET spec_json = ? WHERE strategy_id = ?",
        ('{"name": "broken", "entry": {"kind": "nonsense"}}', broken),
    )
    ledger.conn.commit()

    book = funded_book(ledger, limits=pinned.limits, equity_ccy=Decimal("10000"), at=AS_OF)

    assert [s.strategy_id for s in book.funded] == [healthy]
    assert [label for label, _ in book.excluded] == [f"{broken}@v1"]
    assert "no longer validates" in book.excluded[0][1]


def test_the_book_is_ordered_by_identity_not_by_promotion_time(
    ledger: Ledger, pinned: PinnedLimits
) -> None:
    """Contention between two strategies is resolved by book order.

    So book order has to be a function of identity alone. Ordering by
    `promoted_at` would let a re-promotion change which strategy wins a
    contention, and a replay would then resolve it differently from the live
    run that it is supposed to reconstruct.
    """
    first, _ = _promote(ledger, spec=a_spec(name="alpha"), at=AS_OF)
    second, _ = _promote(ledger, spec=a_spec(name="beta"), at=AS_OF - timedelta(days=30))

    book = funded_book(ledger, limits=pinned.limits, equity_ccy=Decimal("10000"), at=AS_OF)

    assert list(book.labels) == sorted([f"{first}@v1", f"{second}@v1"])


def test_the_book_is_recorded_with_its_exclusions(ledger: Ledger, pinned: PinnedLimits) -> None:
    """What a run was trading is not answerable from the decisions alone.

    A funded strategy that signalled nothing leaves no rows for the instruments
    it declined, and an excluded one leaves none at all — so "four were promoted
    and all four were out of budget" would look identical to "nothing was ever
    promoted".
    """
    kept, _ = _promote(ledger, spec=a_spec(name="kept"))
    blocked, blocked_lineage = _promote(ledger, spec=a_spec(name="blocked"))
    _registry(ledger).charge(
        blocked_lineage, loss_ccy=LINEAGE_BUDGET, strategy_id=blocked, at=AS_OF
    )

    book = funded_book(ledger, limits=pinned.limits, equity_ccy=Decimal("10000"), at=AS_OF)
    record_book(ledger, book, run_id="run_funding")

    row = ledger.conn.execute(
        "SELECT payload_json FROM event_log WHERE event_type = 'book.funded'"
    ).fetchone()
    payload = str(row["payload_json"])
    assert kept in payload
    assert blocked in payload
    assert '"n_funded":1' in payload.replace(" ", "")
    assert '"n_excluded":1' in payload.replace(" ", "")


def test_an_unfunded_strategy_is_bounded_by_the_per_position_cap(
    pinned: PinnedLimits,
) -> None:
    """`--strategy trivial` has no rung, so the M4 caps bind alone.

    Deliberately *not* the floor: a hand-written strategy passed on the command
    line is a drill of the loop rather than of the funding path, and sizing it
    at a rung would imply the ladder has an opinion about something that never
    entered the registry.
    """
    equity = Decimal("10000")
    assert unfunded_notional(pinned.limits, equity_ccy=equity) == (
        equity * Decimal(str(pinned.limits.capital.per_position_pct)) / Decimal(100)
    )


def test_an_explicit_book_must_be_given_a_size() -> None:
    """No default. A default would be a way onto the order path with no
    funding decision behind it, which is the hole the rule closes."""
    with pytest.raises(TypeError):
        explicit_book(  # type: ignore[call-arg]
            DslStrategy(spec=a_spec(), strategy_id="stg_x"),
            pipeline_from_spec(a_spec()),
        )


# --------------------------------------------------------------------------
# Ownership
# --------------------------------------------------------------------------


def test_a_position_belongs_to_the_strategy_whose_entry_opened_it(
    env: dict[str, Any],
) -> None:
    """Resolved from the decision lineage, because the broker cannot say.

    Trading 212 reports a position per instrument and knows nothing about
    strategies, so ownership has to be answered by looking at which decision's
    entry opened it. That join is why every intent records a `decision_id`.
    """
    _seed(env, _rising_bars(days=140))
    _own(env, strategy_id="stg_owner")

    with Ledger(env["db"]) as ledger:
        owner = owner_of(ledger, t212_ticker=TICKER)

    assert owner is not None
    assert owner.strategy_id == "stg_owner"
    assert owner.version == 1


def test_a_position_with_no_recorded_entry_has_no_owner(env: dict[str, Any]) -> None:
    """And that is not the same as belonging to whoever asks first.

    An entry that cannot be attributed cannot be added to either: the add would
    be sized against an allocation that never paid for what is already held.
    """
    _seed(env, _rising_bars(days=140))
    with Ledger(env["db"]) as ledger:
        assert owner_of(ledger, t212_ticker=TICKER) is None


def test_the_most_recent_entry_owns_the_position(env: dict[str, Any]) -> None:
    """A ticker traded by two strategies over time belongs to the latest.

    Positions are flat between entries, so the newest entry is the one that
    opened what is held now. Ordered by the committing event's sequence rather
    than by timestamp: two intents committed in the same instant tie, and an
    intent id is a hash, so neither is a usable tiebreaker.
    """
    _seed(env, _rising_bars(days=140))
    with Ledger(env["db"], config_hash=env["pinned"].config_hash) as ledger:
        _own(env, ledger=ledger, strategy_id="stg_first", suffix="first", seq=1)
        _own(env, ledger=ledger, strategy_id="stg_second", suffix="second", seq=2)
        owner = owner_of(ledger, t212_ticker=TICKER)

    assert owner is not None
    assert owner.strategy_id == "stg_second"


def test_a_rejected_entry_owns_nothing(env: dict[str, Any]) -> None:
    """A rejected order never opened a position, so it cannot own one.

    The states that *can* own are listed positively, and the two unknown ones
    are in the list: an intent committed but never answered may have reached the
    broker, and attributing a position to it is the fail-closed reading.
    """
    _seed(env, _rising_bars(days=140))
    _own(env, strategy_id="stg_rejected")
    with Ledger(env["db"]) as ledger:
        ledger.conn.execute("UPDATE order_intents SET state = 'rejected'")
        ledger.conn.commit()
        assert owner_of(ledger, t212_ticker=TICKER) is None


def test_a_holding_whose_owner_is_unfunded_is_unowned(env: dict[str, Any]) -> None:
    _seed(env, _rising_bars(days=140))
    _own(env, strategy_id="stg_retired")

    with Ledger(env["db"]) as ledger:
        book = funded_book(ledger, limits=env["pinned"].limits, at=AS_OF)
        unowned = unowned_positions(ledger, book=book, held=[TICKER])

    assert [ticker for ticker, _, _ in unowned] == [TICKER]
    assert "stg_retired" in unowned[0][2]
    assert "not in the funded book" in unowned[0][2]


# --------------------------------------------------------------------------
# The allocation as a risk verdict
# --------------------------------------------------------------------------


def _risk_context(
    *,
    notional: Decimal | None,
    held: Decimal = Decimal(0),
    purpose: OrderPurpose = OrderPurpose.ENTRY,
    pinned: PinnedLimits,
) -> RiskContext:
    request = OrderRequest(
        t212_ticker=TICKER,
        instrument_uid=UID,
        side=Side.BUY if purpose is OrderPurpose.ENTRY else Side.SELL,
        purpose=purpose,
        action=Action.ENTER if purpose is OrderPurpose.ENTRY else Action.EXIT,
        reference_price=Decimal("100.00"),
        quantity=None if purpose is OrderPurpose.ENTRY else held,
        expected_edge_bps=Decimal("450") if purpose is OrderPurpose.ENTRY else None,
    )
    return RiskContext(
        as_of=AS_OF,
        limits=pinned.limits,
        request=request,
        account=AccountState(
            equity_ccy=Decimal("10000"),
            free_cash_ccy=Decimal("10000"),
            deployed_ccy=Decimal(0),
            currency="GBP",
        ),
        position_quantity=held,
        may_enter=True,
        regime_exposure_factor=Decimal(1),
        regime_state="risk_on",
        strategy_notional_ccy=notional,
    )


def test_a_missing_allocation_blocks(pinned: PinnedLimits) -> None:
    """**Fail-closed, and this is the case that matters.**

    For a promoted strategy an absent notional on the order path is a wiring
    error, not an unlimited budget — and the permissive reading of a wiring
    error is an order sized by nothing at all. Every other cap in this package
    fails closed on a missing input for the same reason.
    """
    verdict = StrategyAllocationRule().evaluate(_risk_context(notional=None, pinned=pinned))
    assert verdict.verdict is Verdict.BLOCK
    assert verdict.blocks
    assert "wiring error" in verdict.detail


def test_a_zero_allocation_blocks_an_entry(pinned: PinnedLimits) -> None:
    verdict = StrategyAllocationRule().evaluate(_risk_context(notional=Decimal(0), pinned=pinned))
    assert verdict.verdict is Verdict.BLOCK


def test_an_allocation_sizes_the_order(pinned: PinnedLimits) -> None:
    """A sizing opinion, not a gate: "yes, but smaller" is the usual answer."""
    verdict = StrategyAllocationRule().evaluate(
        _risk_context(notional=Decimal("50.00"), pinned=pinned)
    )
    assert verdict.verdict is Verdict.PASS
    assert verdict.max_quantity == Decimal("0.50000000")


def test_what_is_already_held_counts_against_the_allocation(pinned: PinnedLimits) -> None:
    """An add is the same position getting larger, not a new one.

    Without this the allocation would cap each *order* rather than the position,
    and a strategy could reach any size in steps.
    """
    partial = StrategyAllocationRule().evaluate(
        _risk_context(notional=Decimal("50.00"), held=Decimal("0.30"), pinned=pinned)
    )
    assert partial.verdict is Verdict.PASS
    assert partial.max_quantity == Decimal("0.20000000")

    full = StrategyAllocationRule().evaluate(
        _risk_context(notional=Decimal("50.00"), held=Decimal("0.50"), pinned=pinned)
    )
    assert full.verdict is Verdict.BLOCK
    assert "already holding" in full.detail


def test_an_exit_is_not_bounded_by_the_allocation(pinned: PinnedLimits) -> None:
    """The asymmetry again: a strategy whose allocation went to zero still gets out."""
    verdict = StrategyAllocationRule().evaluate(
        _risk_context(
            notional=Decimal(0),
            held=Decimal("0.50"),
            purpose=OrderPurpose.EXIT,
            pinned=pinned,
        )
    )
    assert verdict.verdict is Verdict.NOT_APPLICABLE
    assert not verdict.blocks


# --------------------------------------------------------------------------
# End to end: the promotion changes the account
# --------------------------------------------------------------------------


def _from_registry(env: dict[str, Any], ledger: Ledger, broker: SimulatedBroker) -> TradingLoop:
    """The shared loop fixture, with a book read from the registry."""
    return _loop(
        env,
        ledger,
        broker,
        book=funded_book(
            ledger,
            limits=env["pinned"].limits,
            equity_ccy=broker.get_cash().equity,
            at=AS_OF,
        ),
    )


def _run_one_cycle(env: dict[str, Any], *, rung: int | None) -> tuple[CycleResult, SimulatedBroker]:
    """One cycle against a book built from the registry.

    `rung=None` means nothing is promoted, which is the control: the same bars,
    the same broker and the same universe, with an empty book.
    """
    broker = _broker()
    with Ledger(env["db"], config_hash=env["pinned"].config_hash) as ledger:
        if rung is not None:
            _promote(ledger, rung=rung)
        loop = _from_registry(env, ledger, broker)
        loop.lock.acquire(at=AS_OF)
        return loop.run_cycle(), broker


def test_a_promotion_changes_what_the_loop_trades(env: dict[str, Any]) -> None:
    """**The M5b success condition.**

    The same loop, bars and broker, run twice: once with nothing promoted and
    once with a promoted spec. Before this wiring both runs were identical,
    because the loop held a hardcoded strategy and never read the registry —
    which meant the promotion gate, the ladder and the allocator were all true
    of the database and false of the account.
    """
    _seed(env, _rising_bars(days=140))
    idle, idle_broker = _run_one_cycle(env, rung=None)

    assert not idle.decisions, "an empty book must consult nobody"
    assert not idle.submitted
    assert idle_broker.get_position(TICKER) is None

    # A fresh ledger for the promoted run, so the two differ only in the
    # promotion rather than in what the first run left behind.
    env["db"] = env["db"].with_suffix(".promoted")
    _seed(env, _rising_bars(days=140))
    traded, traded_broker = _run_one_cycle(env, rung=0)

    assert traded.decisions, "a promoted strategy was not consulted"
    assert traded.decisions[0].action is Action.ENTER, traded.decisions[0].rationale
    assert traded.submitted, f"nothing was submitted: {traded.refusals}"
    position = traded_broker.get_position(TICKER)
    assert position is not None and position.quantity > 0


def _entry_quantity(env: dict[str, Any]) -> Decimal:
    with Ledger(env["db"]) as ledger:
        row = ledger.conn.execute(
            "SELECT quantity FROM order_intents WHERE purpose = 'entry'"
        ).fetchone()
    assert row is not None, "no entry intent was written"
    return Decimal(str(row["quantity"]))


def _allocation_limit(env: dict[str, Any]) -> Decimal:
    with Ledger(env["db"]) as ledger:
        row = ledger.conn.execute(
            "SELECT limit_value FROM risk_verdicts WHERE rule_name = 'strategy_allocation'"
        ).fetchone()
    assert row is not None, "the allocation cap left no verdict row"
    return Decimal(str(row["limit_value"]))


def test_the_rung_changes_the_size_the_loop_sends(env: dict[str, Any]) -> None:
    """**The ratchet, measured at the order.**

    A promotion at rung 2 sends four times the notional of one at rung 0.
    Asserted as a comparison between two runs rather than against a hardcoded
    quantity, because that difference is what a rung is *for* — and a rung
    recorded in `strategy_status` but never read would pass every other test in
    this file.

    The comparison is on the *notional* the allocation cap recorded rather than
    on the share count, because the quantity is quantised and — at rung 0, where
    the cap sits exactly on the minimum ticket — rounded up to reach the floor,
    so the two quantities are not in an exact 4:1 ratio.
    """
    floor = env["pinned"].limits.capital.floor_notional_ccy

    _seed(env, _rising_bars(days=140))
    floor_cycle, _ = _run_one_cycle(env, rung=0)
    assert floor_cycle.submitted, f"the floor run did not trade: {floor_cycle.refusals}"
    at_floor = _entry_quantity(env)
    assert _allocation_limit(env) == floor

    env["db"] = env["db"].with_suffix(".rung2")
    _seed(env, _rising_bars(days=140))
    climbed_cycle, _ = _run_one_cycle(env, rung=2)
    assert climbed_cycle.submitted, f"the rung-2 run did not trade: {climbed_cycle.refusals}"
    at_rung_2 = _entry_quantity(env)

    assert _allocation_limit(env) == floor * 4
    assert at_rung_2 > at_floor, (
        f"the rung did not change the order: {at_floor} at rung 0, {at_rung_2} at rung 2"
    )
    assert float(at_rung_2 / at_floor) == pytest.approx(4.0, abs=0.001)


def test_the_ladder_cannot_climb_past_the_per_position_cap(env: dict[str, Any]) -> None:
    """**Climbing rungs moves a strategy toward the ceilings, never past one.**

    At this equity the top rung's own size (`floor * 16`) is above
    `per_position_pct` of the account, so the cap binds and the order is smaller
    than the rung asks for. That is the property that makes the ratchet safe to
    run unattended: the agent can promote itself up the ladder and still cannot
    enlarge its own blast radius, because raising a ceiling is a human edit to a
    hash-pinned file.
    """
    limits = env["pinned"].limits
    equity = Decimal("10000.00")
    rung_size = limits.capital.floor_notional_ccy * 16
    cap = equity * Decimal(str(limits.capital.per_position_pct)) / Decimal(100)
    assert rung_size > cap, "this test needs an equity at which the cap is the tighter bound"

    _seed(env, _rising_bars(days=140))
    cycle, _ = _run_one_cycle(env, rung=limits.promotion.ratchet_max_rung)
    assert cycle.submitted, f"the top-rung run did not trade: {cycle.refusals}"

    assert _allocation_limit(env) == cap


def test_the_allocation_appears_as_a_verdict_row(env: dict[str, Any]) -> None:
    """Every cap leaves a row, and this one is no exception.

    "Why is this position small" is then answerable from the same place as "why
    was this order refused", instead of requiring a join against the allocator's
    own table.
    """
    _seed(env, _rising_bars(days=140))
    _run_one_cycle(env, rung=0)

    with Ledger(env["db"]) as ledger:
        row = ledger.conn.execute(
            "SELECT verdict, limit_value, observed_value FROM risk_verdicts"
            " WHERE rule_name = 'strategy_allocation'"
        ).fetchone()

    assert row is not None, "the allocation cap left no verdict row"
    assert row["verdict"] == "pass"
    assert Decimal(str(row["limit_value"])) == env["pinned"].limits.capital.floor_notional_ccy


def test_a_position_nobody_owns_is_flattened(env: dict[str, Any]) -> None:
    """**A held position no funded strategy will ever close.**

    Reached by retiring the owner while it holds — the searcher's normal
    outcome. Nothing in the book will produce an exit for it, and an open
    position with no bracket order behind it and nothing managing it is the
    state the whole safety design is about. So the loop closes it itself, with
    the reason recorded.
    """
    _seed(env, _rising_bars(days=140))
    broker = _broker()
    broker.seed_position(
        TICKER,
        quantity=Decimal("1"),
        average_price=Decimal("150.00"),
        entered_at=AS_OF - timedelta(days=30),
    )
    _own(env, strategy_id="stg_retired")

    with Ledger(env["db"], config_hash=env["pinned"].config_hash) as ledger:
        _promote(ledger, rung=0)
        loop = _from_registry(env, ledger, broker)
        loop.lock.acquire(at=AS_OF)
        result = loop.run_cycle()

    assert result.unowned, "the orphaned position was not detected"
    assert result.unowned[0][0] == TICKER
    assert "stg_retired" in result.unowned[0][1]
    assert broker.get_position(TICKER) is None, "the orphan was detected but not closed"

    with Ledger(env["db"]) as ledger:
        row = ledger.conn.execute(
            "SELECT payload_json FROM event_log WHERE event_type = 'position.orphaned'"
        ).fetchone()
    assert row is not None
    assert "stg_retired" in str(row["payload_json"])
    assert "flatten" in str(row["payload_json"])


def test_an_orphan_whose_flatten_is_refused_is_still_protected(env: dict[str, Any]) -> None:
    """A regression. The protection pass skipped every stood-down position,
    including one whose flatten was refused — so an orphan with no stop stayed
    without one for as long as the refusal lasted: the one holding nothing
    manages, unprotected. The way there is a symbol whose bars have stopped:
    the flatten has nothing to size from, but a stop is anchored on the price
    the venue reports the position at."""
    year = timedelta(days=365)
    only_later = [
        replace(
            bar,
            bar_open_utc=bar.bar_open_utc + year,
            available_at_utc=bar.available_at_utc + year,
        )
        for bar in _rising_bars(days=5)
    ]
    _seed(env, only_later)
    broker = _broker()
    broker.seed_position(
        TICKER,
        quantity=Decimal("1"),
        average_price=Decimal("150.00"),
        entered_at=AS_OF - timedelta(days=30),
    )
    _own(env, strategy_id="stg_retired")

    with Ledger(env["db"], config_hash=env["pinned"].config_hash) as ledger:
        _promote(ledger, rung=0)
        loop = _from_registry(env, ledger, broker)
        loop.lock.acquire(at=AS_OF)
        result = loop.run_cycle()

    assert result.unowned and result.unowned[0][0] == TICKER
    assert any("not flattened" in reason for _, reason in result.refusals), result.refusals
    assert broker.get_position(TICKER) is not None, "vacuous: the flatten was not refused"
    assert len(broker.protective_orders_for(TICKER)) == 1, "the orphan was left unprotected"


def test_an_owned_position_is_not_flattened(env: dict[str, Any]) -> None:
    """The control for the test above, and it is not a formality.

    A flatten path that fired on positions the book *does* own would close every
    position on the cycle after it opened — the most expensive possible bug
    here, and one that a test of the orphan case alone would not catch.
    """
    _seed(env, _rising_bars(days=140))
    broker = _broker()

    with Ledger(env["db"], config_hash=env["pinned"].config_hash) as ledger:
        strategy_id, _ = _promote(ledger, rung=2)
        book = funded_book(
            ledger,
            limits=env["pinned"].limits,
            equity_ccy=broker.get_cash().equity,
            at=AS_OF,
        )
        first = _loop(env, ledger, broker, book=book)
        first.lock.acquire(at=AS_OF)
        opened = first.run_cycle()
        assert opened.submitted, f"the entry did not go through: {opened.refusals}"
        assert not opened.unowned

        # A second cycle, now holding. The position is the promoted strategy's,
        # so the owner decides and the flatten path leaves it alone.
        later = _loop(env, ledger, broker, book=book)
        held = later.run_cycle()

    assert not held.unowned, f"an owned position was treated as an orphan: {held.unowned}"
    position = broker.get_position(TICKER)
    assert position is not None and position.quantity > 0
    assert held.decisions, "the owner was not asked about its own position"
    assert held.decisions[0].strategy_id == strategy_id


def test_only_the_owner_is_asked_about_a_held_position(env: dict[str, Any]) -> None:
    """Two funded strategies, one holding: the owner decides alone.

    Otherwise both would be asked, both could act, and each would be sized
    against an allocation that paid for part of one position.
    """
    _seed(env, _rising_bars(days=140))
    broker = _broker()
    broker.seed_position(
        TICKER,
        quantity=Decimal("0.1"),
        average_price=Decimal("150.00"),
        entered_at=AS_OF - timedelta(days=30),
    )

    with Ledger(env["db"], config_hash=env["pinned"].config_hash) as ledger:
        owner_id, _ = _promote(ledger, spec=a_spec(name="owner"))
        other_id, _ = _promote(ledger, spec=a_spec(name="other"))
        _own(env, strategy_id=owner_id)
        book = funded_book(
            ledger,
            limits=env["pinned"].limits,
            equity_ccy=broker.get_cash().equity,
            at=AS_OF,
        )
        assert len(book) == 2
        loop = _loop(env, ledger, broker, book=book)
        loop.lock.acquire(at=AS_OF)
        result = loop.run_cycle()

    assert not result.unowned
    consulted = {decision.strategy_id for decision in result.decisions}
    assert consulted == {owner_id}, f"{other_id} was asked about a position it does not own"


def _cli(env: dict[str, Any], *extra: str) -> Any:
    return CliRunner().invoke(
        app,
        [
            "run",
            "--limits",
            str(env["limits"]),
            "--db",
            str(env["db"]),
            "--bars",
            str(env["bars"]),
            "--mode",
            "paper",
            "--cycles",
            "1",
            *extra,
        ],
    )


def test_tb_run_refuses_to_start_with_nothing_promoted(env: dict[str, Any]) -> None:
    """A refusal, not an idle loop — and it names the command that fixes it.

    A loop that started with an empty book would write a run record, a
    heartbeat and a cycle event saying nothing was traded, which is
    indistinguishable from a working day on which nothing signalled. Exit 2:
    the operator has something to do.
    """
    _seed(env, _rising_bars(days=140))
    result = _cli(env)

    assert result.exit_code == 2, result.output
    assert "no strategy is promoted" in result.output
    assert "tb promote evaluate" in result.output


def test_tb_run_trades_the_trivial_strategy_only_when_asked(env: dict[str, Any]) -> None:
    """`--strategy trivial` is the opt-in, and it says what it is doing.

    The hand-written strategy has cleared no gate, so the run warns rather than
    presenting it as a funded one — otherwise a drill and a live promotion would
    look the same in the log.
    """
    _seed(env, _rising_bars(days=140))
    result = _cli(env, "--strategy", "trivial")

    assert result.exit_code == 0, result.output
    assert "no gate has cleared" in result.output
    with Ledger(env["db"]) as ledger:
        row = ledger.conn.execute(
            "SELECT payload_json FROM event_log WHERE event_type = 'book.funded'"
        ).fetchone()
    assert row is not None, "the drill book was not recorded"
    assert "explicit" in str(row["payload_json"])


def test_tb_run_refuses_an_unknown_strategy_name(env: dict[str, Any]) -> None:
    """Every other strategy reaches the loop by being promoted."""
    _seed(env, _rising_bars(days=140))
    result = _cli(env, "--strategy", "something-else")

    assert result.exit_code == 2, result.output
    assert "tb promote evaluate" in result.output


def test_two_strategies_cannot_both_enter_one_instrument(env: dict[str, Any]) -> None:
    """Contention resolved, with the winner named in the refusal.

    A silent skip would be indistinguishable from a strategy that simply held,
    and "why did only one of my two strategies trade" is a question the cycle
    record has to be able to answer.
    """
    _seed(env, _rising_bars(days=140))
    broker = _broker()

    with Ledger(env["db"], config_hash=env["pinned"].config_hash) as ledger:
        first, _ = _promote(ledger, spec=a_spec(name="alpha"))
        second, _ = _promote(ledger, spec=a_spec(name="beta"))
        book = funded_book(
            ledger,
            limits=env["pinned"].limits,
            equity_ccy=broker.get_cash().equity,
            at=AS_OF,
        )
        loop = _loop(env, ledger, broker, book=book)
        loop.lock.acquire(at=AS_OF)
        result = loop.run_cycle()

    assert len(result.submitted) == 1, "both strategies opened a position in one instrument"
    winner = book.funded[0]
    loser = book.funded[1]
    assert result.decisions[0].strategy_id == winner.strategy_id
    refusals = " ".join(reason for _, reason in result.refusals)
    assert loser.label in refusals
    assert winner.label in refusals
    assert {first, second} == {winner.strategy_id, loser.strategy_id}
