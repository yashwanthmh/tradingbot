"""Family caps from realised overlap, and priors shrunk toward realised edge."""

from __future__ import annotations

from datetime import UTC, date, datetime
from decimal import Decimal

import pytest

from tb.config.loader import PinnedLimits
from tb.ledger.events import EventType
from tb.ledger.store import Ledger
from tb.portfolio.allocator import (
    SHRINKAGE_PRIOR_TRADES,
    AllocationError,
    Allocator,
    StrategyInput,
    allocation_as_of,
    blend,
    shrinkage_weight,
)
from tb.portfolio.correlation import (
    FAMILY_OVERLAP_THRESHOLD,
    Holding,
    apply_family_cap,
    families,
    overlap,
)
from tb.portfolio.decay import (
    MIN_TRADES_FOR_A_POSITIVE_VERDICT,
    MIN_TRADES_FOR_SCALE,
    SCALE_EDGE_RETENTION,
    ReviewCycle,
    ReviewInput,
    Verdict,
    review,
)

AS_OF = datetime(2026, 6, 1, 14, 0, tzinfo=UTC)
EQUITY = Decimal("20000")
DAY = date(2026, 5, 1)


def holdings(strategy_id: str, names: list[str], *, days: int = 10) -> list[Holding]:
    return [
        Holding(
            strategy_id=strategy_id,
            instrument_uid=name,
            session_date=date(2026, 5, offset + 1),
        )
        for offset in range(days)
        for name in names
    ]


# --------------------------------------------------------------------------
# Overlap
# --------------------------------------------------------------------------


def test_identical_books_overlap_completely() -> None:
    a = holdings("a", ["AAPL", "MSFT"])
    b = holdings("b", ["AAPL", "MSFT"])
    assert overlap(a, b) == 1.0


def test_disjoint_books_do_not_overlap() -> None:
    assert overlap(holdings("a", ["AAPL"]), holdings("b", ["XOM"])) == 0.0


def test_overlap_is_symmetric_rather_than_a_subset_fraction() -> None:
    """The asymmetric form would report 1.0 for a strategy holding one name on
    one day inside another's year-long book, which is not the fact a
    concentration cap is after."""
    big = holdings("big", ["AAPL", "MSFT", "NVDA"], days=20)
    small = [Holding(strategy_id="small", instrument_uid="AAPL", session_date=DAY)]
    assert overlap(small, big) == overlap(big, small)
    assert overlap(small, big) < 0.1


def test_two_empty_books_do_not_overlap() -> None:
    """The conservative reading of 'neither has held anything yet' is that
    nothing has been shown, not that they are identical."""
    assert overlap([], []) == 0.0


# --------------------------------------------------------------------------
# Families
# --------------------------------------------------------------------------


def test_strategies_holding_the_same_names_are_one_family() -> None:
    grouped = families(holdings("a", ["AAPL", "MSFT"]) + holdings("b", ["AAPL", "MSFT"]))
    assert len(grouped) == 1
    assert grouped[0].strategy_ids == ("a", "b")
    assert grouped[0].max_pairwise_overlap == 1.0
    assert not grouped[0].is_singleton


def test_unrelated_strategies_are_separate_families() -> None:
    grouped = families(holdings("a", ["AAPL"]) + holdings("b", ["XOM"]))
    assert len(grouped) == 2
    assert all(family.is_singleton for family in grouped)


def test_a_chain_of_overlap_is_one_family() -> None:
    """Single-linkage on purpose. A chain concentrates as much as a clique,
    and requiring every pair to overlap would split it and cap nothing.

    Four names each with three shared puts consecutive pairs at 0.6 and the
    ends at 0.33. The first draft used three names with two shared, which lands
    on exactly 0.5 — the threshold is strict, so nothing merged and the test was
    asserting the opposite of what it meant to.
    """
    a = ["AAPL", "MSFT", "NVDA", "AMZN"]
    b = ["MSFT", "NVDA", "AMZN", "GOOGL"]
    c = ["NVDA", "AMZN", "GOOGL", "META"]
    book = holdings("a", a) + holdings("b", b) + holdings("c", c)

    ends = overlap(holdings("a", a), holdings("c", c))
    assert ends < FAMILY_OVERLAP_THRESHOLD, "the ends must not overlap directly"
    assert overlap(holdings("a", a), holdings("b", b)) > FAMILY_OVERLAP_THRESHOLD

    grouped = families(book)
    assert len(grouped) == 1
    assert grouped[0].strategy_ids == ("a", "b", "c")


def test_the_threshold_is_load_bearing() -> None:
    """Just either side of it, so the constant is doing work."""
    book = holdings("a", ["AAPL", "MSFT"]) + holdings("b", ["AAPL", "XOM"])
    score = overlap(holdings("a", ["AAPL", "MSFT"]), holdings("b", ["AAPL", "XOM"]))
    assert families(book, threshold=score - 0.01)[0].size == 2
    assert all(f.is_singleton for f in families(book, threshold=score + 0.01))


def test_a_strategy_with_no_holdings_still_gets_a_family() -> None:
    """It has an allocation, so it has to appear in the cap arithmetic."""
    outcome = apply_family_cap(
        weights={"a": Decimal("0.5"), "b": Decimal("0.5")},
        holdings=holdings("a", ["AAPL"]),
        max_family_fraction=Decimal("1"),
    )
    assert {f.strategy_ids[0] for f in outcome.families if f.is_singleton} >= {"a", "b"}


# --------------------------------------------------------------------------
# The cap
# --------------------------------------------------------------------------


def test_a_family_over_its_cap_is_scaled_proportionally() -> None:
    """The cap is a statement about the family's total, not a judgement about
    which member deserves it."""
    outcome = apply_family_cap(
        weights={"a": Decimal("0.6"), "b": Decimal("0.4")},
        holdings=holdings("a", ["AAPL", "MSFT"]) + holdings("b", ["AAPL", "MSFT"]),
        max_family_fraction=Decimal("0.5"),
    )
    assert outcome.n_capped == 1
    total = outcome.weights["a"] + outcome.weights["b"]
    assert total == pytest.approx(Decimal("0.5"))
    # Relative standing preserved: a still has 60% of the family's weight.
    assert outcome.weights["a"] / total == pytest.approx(Decimal("0.6"))


def test_an_uncorrelated_portfolio_is_untouched() -> None:
    outcome = apply_family_cap(
        weights={"a": Decimal("0.5"), "b": Decimal("0.5")},
        holdings=holdings("a", ["AAPL"]) + holdings("b", ["XOM"]),
        max_family_fraction=Decimal("0.6"),
    )
    assert outcome.n_capped == 0
    assert outcome.weights == {"a": Decimal("0.5"), "b": Decimal("0.5")}


def test_freed_weight_is_not_redistributed() -> None:
    """Handing it to the next family would push that one toward its own cap,
    and which family got the surplus would depend on iteration order."""
    outcome = apply_family_cap(
        weights={"a": Decimal("0.4"), "b": Decimal("0.4"), "c": Decimal("0.2")},
        holdings=holdings("a", ["AAPL"]) + holdings("b", ["AAPL"]) + holdings("c", ["XOM"]),
        max_family_fraction=Decimal("0.5"),
    )
    assert outcome.weights["c"] == Decimal("0.2")
    assert sum(outcome.weights.values()) < Decimal("1")


# --------------------------------------------------------------------------
# Shrinkage
# --------------------------------------------------------------------------


def test_no_trades_means_the_prior_alone() -> None:
    assert shrinkage_weight(0) == Decimal(0)


def test_the_weight_reaches_a_half_at_the_prior_trade_count() -> None:
    assert shrinkage_weight(SHRINKAGE_PRIOR_TRADES) == Decimal("0.5")


def test_the_prior_never_disappears_entirely() -> None:
    """A strategy with 200 good trades still had a backtest, and discarding it
    would make the allocator forget why the strategy was funded."""
    assert shrinkage_weight(10_000) < Decimal(1)


def test_a_negative_trade_count_is_refused() -> None:
    with pytest.raises(AllocationError, match="must not be negative"):
        shrinkage_weight(-1)


def test_a_month_of_trading_is_mostly_still_the_prior() -> None:
    """The number the whole module exists for: at 10 trades the realised edge
    gets a quarter of the weight."""
    weight, blended = blend(
        prior_edge_bps=Decimal("300"), realised_edge_bps=Decimal("100"), n_trades=10
    )
    assert weight == Decimal("0.25")
    assert blended == Decimal("250")


def test_an_absent_realised_edge_is_not_an_edge_of_zero() -> None:
    """A strategy whose fills all had inferred prices has produced no
    measurement, and scoring it as zero would punish it for the reconciler."""
    weight, blended = blend(prior_edge_bps=Decimal("300"), realised_edge_bps=None, n_trades=50)
    assert weight == Decimal(0)
    assert blended == Decimal("300")


# --------------------------------------------------------------------------
# Allocation
# --------------------------------------------------------------------------


def a_strategy(
    strategy_id: str,
    *,
    prior: str = "300",
    realised: str | None = None,
    trades: int = 0,
    rung: int = 0,
) -> StrategyInput:
    return StrategyInput(
        strategy_id=strategy_id,
        version=1,
        lineage_id=f"lin_{strategy_id}",
        rung=rung,
        prior_edge_bps=Decimal(prior),
        realised_edge_bps=None if realised is None else Decimal(realised),
        n_realised_trades=trades,
    )


@pytest.fixture
def allocator(ledger: Ledger, pinned: PinnedLimits) -> Allocator:
    return Allocator(ledger, limits=pinned.limits, run_id="run_test")


def test_deployable_capital_is_bounded_by_the_hard_limits(
    allocator: Allocator, pinned: PinnedLimits
) -> None:
    result = allocator.allocate(
        strategies=[a_strategy("a")], equity_ccy=EQUITY, at=AS_OF, record=False
    )
    percentage = EQUITY * Decimal(str(pinned.limits.capital.max_deployed_pct)) / Decimal(100)
    assert result.deployable_ccy == min(percentage, pinned.limits.capital.absolute_ceiling_ccy)


def test_the_absolute_ceiling_does_not_scale_with_the_account(
    allocator: Allocator, pinned: PinnedLimits
) -> None:
    result = allocator.allocate(
        strategies=[a_strategy("a")],
        equity_ccy=Decimal("10000000"),
        at=AS_OF,
        record=False,
    )
    assert result.deployable_ccy == pinned.limits.capital.absolute_ceiling_ccy


def test_a_stronger_blended_edge_gets_a_larger_weight(allocator: Allocator) -> None:
    result = allocator.allocate(
        strategies=[a_strategy("a", prior="400"), a_strategy("b", prior="200")],
        equity_ccy=EQUITY,
        at=AS_OF,
        record=False,
    )
    by_id = result.by_strategy
    assert by_id["a"].weight > by_id["b"].weight
    assert by_id["a"].raw_weight + by_id["b"].raw_weight == pytest.approx(Decimal(1))
    assert by_id["a"].raw_weight == pytest.approx(Decimal(2) / Decimal(3))


def test_one_family_cannot_hold_the_whole_deployable_budget(
    allocator: Allocator, pinned: PinnedLimits
) -> None:
    """Surprising and intended: a lone strategy gets 40% of the deployable
    total, not all of it.

    `max_family_deployed_pct` is 4% against a `max_deployed_pct` of 10%, and a
    single strategy is a family of one. So using the full deployable budget
    takes at least three families that do not overlap — which is exactly the
    control, since "ten uncorrelated strategies" is the claim being tested. The
    unused portion is not redistributed: there was nothing uncorrelated to put
    it into.
    """
    capital = pinned.limits.capital
    expected = Decimal(str(capital.max_family_deployed_pct)) / Decimal(
        str(capital.max_deployed_pct)
    )
    result = allocator.allocate(
        strategies=[a_strategy("a")], equity_ccy=EQUITY, at=AS_OF, record=False
    )
    allocation = result.by_strategy["a"]
    assert allocation.raw_weight == Decimal(1)
    assert allocation.weight == pytest.approx(expected)


def test_the_rung_is_a_ceiling_not_a_target(allocator: Allocator, pinned: PinnedLimits) -> None:
    """A strategy at rung 0 does not get a larger position because the
    portfolio has room — that is what the ratchet is for."""
    result = allocator.allocate(
        strategies=[a_strategy("a")], equity_ccy=EQUITY, at=AS_OF, record=False
    )
    allocation = result.by_strategy["a"]
    # Its share of the deployable budget is far above the floor notional, and
    # the rung still holds it there.
    assert result.deployable_ccy * allocation.weight > pinned.limits.capital.floor_notional_ccy
    assert allocation.notional_ccy == pinned.limits.capital.floor_notional_ccy


def test_a_higher_rung_allows_a_larger_position(allocator: Allocator) -> None:
    low = allocator.allocate(
        strategies=[a_strategy("a", rung=0)], equity_ccy=EQUITY, at=AS_OF, record=False
    )
    high = allocator.allocate(
        strategies=[a_strategy("a", rung=3)], equity_ccy=EQUITY, at=AS_OF, record=False
    )
    assert high.by_strategy["a"].notional_ccy > low.by_strategy["a"].notional_ccy


def test_a_portfolio_the_allocator_disbelieves_holds_cash(allocator: Allocator) -> None:
    """Not an equal split. A portfolio where the allocator believes nothing
    should hold cash, not spread itself over things it disbelieves."""
    result = allocator.allocate(
        strategies=[
            a_strategy("a", prior="300", realised="-500", trades=1000),
            a_strategy("b", prior="300", realised="-400", trades=1000),
        ],
        equity_ccy=EQUITY,
        at=AS_OF,
        record=False,
    )
    assert result.total_notional_ccy == Decimal(0)
    assert all(allocation.weight == Decimal(0) for allocation in result.allocations)


def test_an_empty_portfolio_allocates_nothing(allocator: Allocator) -> None:
    result = allocator.allocate(strategies=[], equity_ccy=EQUITY, at=AS_OF, record=False)
    assert result.allocations == ()
    assert "no promoted strategies" in result.detail


def test_correlated_strategies_are_capped_together(
    allocator: Allocator, pinned: PinnedLimits
) -> None:
    """The failure this prevents: ten strategies finding the same trade, and
    the portfolio quietly being one position."""
    book = (
        holdings("a", ["AAPL", "MSFT"])
        + holdings("b", ["AAPL", "MSFT"])
        + holdings("c", ["XOM", "JPM"])
    )
    result = allocator.allocate(
        strategies=[a_strategy("a"), a_strategy("b"), a_strategy("c")],
        equity_ccy=EQUITY,
        holdings=book,
        at=AS_OF,
        record=False,
    )
    assert result.n_correlation_capped == 1
    by_id = result.by_strategy
    family_weight = by_id["a"].weight + by_id["b"].weight
    expected = Decimal(str(pinned.limits.capital.max_family_deployed_pct)) / Decimal(
        str(pinned.limits.capital.max_deployed_pct)
    )
    assert family_weight == pytest.approx(expected)
    assert by_id["a"].correlation_capped
    assert not by_id["c"].correlation_capped


def test_an_allocation_records_the_shrinkage_beside_the_result(
    allocator: Allocator, ledger: Ledger
) -> None:
    """A row holding only the blended figure would present a prior as a
    measurement."""
    result = allocator.allocate(
        strategies=[a_strategy("a", prior="300", realised="100", trades=10)],
        equity_ccy=EQUITY,
        at=AS_OF,
    )
    stored = allocator.latest()
    assert len(stored) == 1
    assert stored[0].shrinkage == Decimal("0.25")
    assert stored[0].prior_edge_bps == Decimal("300")
    assert stored[0].realised_edge_bps == Decimal("100")
    assert stored[0].evidence_is_mostly_prior
    assert "shrinkage" in stored[0].explain()
    assert "shrinkage" in result.explain()

    events = ledger.conn.execute(
        "SELECT COUNT(*) FROM event_log WHERE event_type = ?",
        (EventType.ALLOCATION_DECIDED.value,),
    ).fetchone()
    assert events[0] == 1


def test_two_rounds_at_the_same_instant_are_not_spliced(
    allocator: Allocator, ledger: Ledger
) -> None:
    """All of one round's rows share the recording event's sequence, which is
    why the lookup keys on that rather than on the timestamp."""
    allocator.allocate(strategies=[a_strategy("a")], equity_ccy=EQUITY, at=AS_OF)
    allocator.allocate(strategies=[a_strategy("b"), a_strategy("c")], equity_ccy=EQUITY, at=AS_OF)
    latest = allocator.latest()
    assert [allocation.strategy_id for allocation in latest] == ["b", "c"]


def test_the_allocation_in_force_at_an_instant_is_answerable(
    allocator: Allocator, ledger: Ledger
) -> None:
    """A replay asks what the allocator believed *then*, and answering with
    today's table would explain a months-old position with a weight it never
    had."""
    earlier = AS_OF.replace(month=5)
    allocator.allocate(strategies=[a_strategy("a")], equity_ccy=EQUITY, at=earlier)
    allocator.allocate(strategies=[a_strategy("b")], equity_ccy=EQUITY, at=AS_OF)

    assert [a.strategy_id for a in allocation_as_of(ledger, earlier)] == ["a"]
    assert [a.strategy_id for a in allocation_as_of(ledger, AS_OF)] == ["b"]
    assert allocation_as_of(ledger, earlier.replace(year=2020)) == []


# --------------------------------------------------------------------------
# The review cycle
# --------------------------------------------------------------------------


def a_review(
    *,
    trades: int = 20,
    pnl: str = "10",
    declared: str = "300",
    realised: str | None = "250",
    budget: str | None = None,
    consumed: str | None = None,
    risk_blocks: int = 0,
) -> ReviewInput:
    return ReviewInput(
        strategy_id="stg_1",
        version=1,
        lineage_id="lin_1",
        n_realised_trades=trades,
        realised_pnl_ccy=Decimal(pnl),
        declared_edge_bps=Decimal(declared),
        realised_edge_bps=None if realised is None else Decimal(realised),
        lineage_budget_ccy=None if budget is None else Decimal(budget),
        lineage_consumed_ccy=None if consumed is None else Decimal(consumed),
        n_risk_blocks=risk_blocks,
    )


def test_too_little_evidence_yields_keep_not_an_endorsement() -> None:
    result = review(a_review(trades=MIN_TRADES_FOR_A_POSITIVE_VERDICT - 1), at=AS_OF)
    assert result.verdict is Verdict.KEEP
    assert not result.evidence_sufficient
    assert "nothing has been shown either way" in result.reasons[0]


def test_kill_is_reachable_on_thin_evidence() -> None:
    """KILL does not rest on the edge estimate — it rests on losses, which are
    facts about the account rather than estimates about the future."""
    result = review(a_review(trades=2, pnl="-60", budget="100", consumed="60"), at=AS_OF)
    assert result.verdict is Verdict.KILL
    assert not result.evidence_sufficient


def test_one_member_cannot_spend_the_whole_lineages_budget() -> None:
    result = review(a_review(trades=50, pnl="-55", budget="100", consumed="55"), at=AS_OF)
    assert result.verdict is Verdict.KILL
    assert "lineage's loss budget" in result.reasons[0]


def test_forced_exits_alongside_a_loss_are_a_kill() -> None:
    """A strategy whose exits are being forced by the risk layer is not
    expressing its own logic any more."""
    result = review(a_review(trades=20, pnl="-5", risk_blocks=3), at=AS_OF)
    assert result.verdict is Verdict.KILL
    assert "risk block" in result.reasons[0]


def test_a_losing_strategy_with_some_edge_left_is_iterated() -> None:
    """The shape may be right and the parameters wrong."""
    result = review(a_review(trades=20, pnl="-4", realised="60"), at=AS_OF)
    assert result.verdict is Verdict.ITERATE
    assert result.verdict.retires_the_strategy


def test_a_losing_strategy_with_no_edge_left_is_killed() -> None:
    result = review(a_review(trades=20, pnl="-4", realised="-10"), at=AS_OF)
    assert result.verdict is Verdict.KILL


def test_scale_needs_the_same_trade_count_the_gate_demanded() -> None:
    """Disbelieving a backtest on 29 trades and then believing a live record on
    8 would be incoherent."""
    just_short = review(a_review(trades=MIN_TRADES_FOR_SCALE - 1), at=AS_OF)
    enough = review(a_review(trades=MIN_TRADES_FOR_SCALE), at=AS_OF)
    assert just_short.verdict is Verdict.KEEP
    assert enough.verdict is Verdict.SCALE


def test_scale_needs_the_edge_to_be_holding_up() -> None:
    declared = Decimal("300")
    below = declared * (SCALE_EDGE_RETENTION - Decimal("0.05"))
    result = review(a_review(trades=50, realised=str(below), declared=str(declared)), at=AS_OF)
    assert result.verdict is Verdict.KEEP
    assert "short of the" in result.reasons[0]


def test_an_unmeasurable_realised_edge_cannot_be_scaled_on() -> None:
    """Not a retention of zero — a measurement that does not exist."""
    result = review(a_review(trades=50, pnl="20", realised=None), at=AS_OF)
    assert result.verdict is Verdict.KEEP
    assert "does not exist" in result.reasons[0]


def test_scale_is_the_only_verdict_needing_strong_evidence() -> None:
    needing = [verdict for verdict in Verdict if verdict.needs_strong_evidence]
    assert needing == [Verdict.SCALE]


def test_the_review_cycle_records_each_verdict(ledger: Ledger) -> None:
    cycle = ReviewCycle(ledger, run_id="run_test")
    results = cycle.run([a_review(), a_review(trades=2, pnl="-1")], at=AS_OF)
    assert len(results) == 2
    events = ledger.conn.execute(
        "SELECT COUNT(*) FROM event_log WHERE event_type = ?",
        (EventType.STRATEGY_REVIEWED.value,),
    ).fetchone()
    assert events[0] == 2
    assert "KEEP" in results[1].summary()
