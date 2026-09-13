"""The reconciler.

Every test here is a state the system could genuinely find itself in after a
crash, a manual intervention, or a partial fill — and the question in each case
is whether it notices.

Two findings are blocking on purpose and worth stating plainly:

* An **unprotected position**. Trading 212 has no bracket orders, so an entry
  fill always precedes its protective stop; a crash in that window leaves a
  naked long. This is not a hypothetical gap, it is the normal order of events
  interrupted.
* An **orphan buy order**. A leftover sell is untidy. A leftover buy silently
  re-enters a position the system believes is closed.
"""

from __future__ import annotations

from datetime import UTC, datetime
from decimal import Decimal

import pytest

from tb.broker.port import (
    AccountInfo,
    AccountSnapshot,
    BrokerOrder,
    CashBalance,
    OrderStatus,
    OrderType,
    Position,
    Side,
    TimeValidity,
)
from tb.broker.reconcile import (
    DesiredOrder,
    ReconcileFindingKind,
    Reconciler,
    ReconcileVerdict,
    Severity,
    UnresolvedIntent,
)
from tb.config.loader import PinnedLimits
from tb.data.symbols import SymbolMap
from tb.ledger.events import EventType
from tb.ledger.store import Ledger


class FakeBroker:
    """Returns a scripted snapshot. Read-only, like the real M1 client."""

    def __init__(self, snapshot: AccountSnapshot) -> None:
        self._snapshot = snapshot
        self.snapshot_calls = 0

    @property
    def environment(self) -> str:
        return "demo"

    def snapshot(self) -> AccountSnapshot:
        self.snapshot_calls += 1
        return self._snapshot


class FakeIntents:
    def __init__(
        self,
        unresolved: tuple[UnresolvedIntent, ...] = (),
        desired: tuple[DesiredOrder, ...] = (),
    ) -> None:
        self._unresolved = unresolved
        self._desired = desired

    def unresolved_intents(self) -> tuple[UnresolvedIntent, ...]:
        return self._unresolved

    def desired_orders(self) -> tuple[DesiredOrder, ...]:
        return self._desired


def make_snapshot(
    *,
    positions: tuple[Position, ...] = (),
    orders: tuple[BrokerOrder, ...] = (),
    currency: str = "GBP",
    warnings: tuple[str, ...] = (),
) -> AccountSnapshot:
    return AccountSnapshot(
        snap_id="snap_test01",
        taken_at=datetime(2026, 3, 2, 14, 30, tzinfo=UTC),
        environment="demo",
        cash=CashBalance(
            currency=currency,
            free=Decimal("850.00"),
            total=Decimal("1000.00"),
            invested=Decimal("150.00"),
        ),
        positions=positions,
        open_orders=orders,
        account=AccountInfo(account_id=42, currency_code=currency),
        staleness_warnings=warnings,
    )


def protective_stop(ticker: str, quantity: Decimal, order_id: str = "stop1") -> BrokerOrder:
    return BrokerOrder(
        broker_order_id=order_id,
        ticker=ticker,
        side=Side.SELL,
        order_type=OrderType.STOP,
        status=OrderStatus.WORKING,
        quantity=quantity,
        stop_price=Decimal("140.00"),
        time_validity=TimeValidity.GOOD_TILL_CANCEL,
    )


def _reconciler(
    ledger: Ledger,
    pinned: PinnedLimits,
    snapshot: AccountSnapshot,
    *,
    intents: FakeIntents | None = None,
    with_symbol_map: bool = False,
) -> Reconciler:
    return Reconciler(
        ledger=ledger,
        broker=FakeBroker(snapshot),  # type: ignore[arg-type]
        limits=pinned.limits,
        symbol_map=SymbolMap(ledger) if with_symbol_map else None,
        intents=intents,
        run_id="run_test",
    )


class TestCleanAccount:
    def test_an_empty_account_reconciles_clean(self, ledger: Ledger, pinned: PinnedLimits) -> None:
        report = _reconciler(ledger, pinned, make_snapshot()).run()
        assert report.verdict is ReconcileVerdict.CLEAN
        assert report.clean

    def test_a_protected_position_reconciles_clean(
        self, ledger: Ledger, pinned: PinnedLimits
    ) -> None:
        snapshot = make_snapshot(
            positions=(Position(ticker="AAPL_US_EQ", quantity=Decimal("2")),),
            orders=(protective_stop("AAPL_US_EQ", Decimal("2")),),
        )
        report = _reconciler(ledger, pinned, snapshot).run()
        assert report.verdict is ReconcileVerdict.CLEAN, report.summary()

    def test_the_snapshot_is_persisted(self, ledger: Ledger, pinned: PinnedLimits) -> None:
        snapshot = make_snapshot(
            positions=(
                Position(
                    ticker="AAPL_US_EQ",
                    quantity=Decimal("2"),
                    average_price=Decimal("150.10"),
                    current_price=Decimal("155.40"),
                ),
            ),
            orders=(protective_stop("AAPL_US_EQ", Decimal("2")),),
        )
        _reconciler(ledger, pinned, snapshot).run()

        cash = ledger.conn.execute("SELECT * FROM cash_snapshot").fetchone()
        assert cash["total"] == "1000.00"
        # Stored as text, not float: binding a Decimal as a float is how a
        # quantity acquires a rounding error on its way to disk.
        position = ledger.conn.execute("SELECT * FROM positions_snapshot").fetchone()
        assert position["quantity"] == "2"
        assert position["average_price"] == "150.10"

    def test_a_snapshot_event_is_recorded(self, ledger: Ledger, pinned: PinnedLimits) -> None:
        _reconciler(ledger, pinned, make_snapshot()).run()
        events = list(ledger.iter_events(event_type=EventType.BROKER_SNAPSHOT_TAKEN))
        assert len(events) == 1


class TestUnprotectedPositions:
    def test_a_position_with_no_stop_is_blocking(
        self, ledger: Ledger, pinned: PinnedLimits
    ) -> None:
        snapshot = make_snapshot(positions=(Position(ticker="AAPL_US_EQ", quantity=Decimal("2")),))
        report = _reconciler(ledger, pinned, snapshot).run()

        findings = [
            f for f in report.findings if f.kind is ReconcileFindingKind.UNPROTECTED_POSITION
        ]
        assert len(findings) == 1
        assert findings[0].severity is Severity.BLOCKING
        assert "no bracket orders" in findings[0].detail
        assert report.verdict is ReconcileVerdict.HALTED

    def test_the_suggested_action_follows_the_configured_policy(
        self, ledger: Ledger, pinned: PinnedLimits
    ) -> None:
        """Flatten is the default.

        Placing a stop at an unknown price after an unknown gap is a decision
        made with no information about what happened while we were down.
        """
        assert pinned.limits.safety.on_unprotected_position == "flatten"
        snapshot = make_snapshot(positions=(Position(ticker="AAPL_US_EQ", quantity=Decimal("2")),))
        report = _reconciler(ledger, pinned, snapshot).run()
        finding = next(
            f for f in report.findings if f.kind is ReconcileFindingKind.UNPROTECTED_POSITION
        )
        assert finding.suggested_action == "flatten the position"

    def test_partial_protection_is_blocking_and_names_the_uncovered_amount(
        self, ledger: Ledger, pinned: PinnedLimits
    ) -> None:
        """A stop for half the position leaves half with no floor."""
        snapshot = make_snapshot(
            positions=(Position(ticker="AAPL_US_EQ", quantity=Decimal("10")),),
            orders=(protective_stop("AAPL_US_EQ", Decimal("4")),),
        )
        report = _reconciler(ledger, pinned, snapshot).run()
        finding = next(
            f
            for f in report.findings
            if f.kind is ReconcileFindingKind.PARTIALLY_PROTECTED_POSITION
        )
        assert finding.severity is Severity.BLOCKING
        assert "cover only 4" in finding.detail
        assert "uncovered 6" in finding.detail

    def test_a_buy_order_does_not_count_as_protection(
        self, ledger: Ledger, pinned: PinnedLimits
    ) -> None:
        buy_stop = BrokerOrder(
            broker_order_id="b1",
            ticker="AAPL_US_EQ",
            side=Side.BUY,
            order_type=OrderType.STOP,
            status=OrderStatus.WORKING,
            quantity=Decimal("2"),
        )
        snapshot = make_snapshot(
            positions=(Position(ticker="AAPL_US_EQ", quantity=Decimal("2")),),
            orders=(buy_stop,),
        )
        report = _reconciler(ledger, pinned, snapshot).run()
        assert report.count(ReconcileFindingKind.UNPROTECTED_POSITION) == 1

    def test_a_limit_sell_does_not_count_as_protection(
        self, ledger: Ledger, pinned: PinnedLimits
    ) -> None:
        """A take-profit is not a floor.

        A limit sell above the market protects nothing on the way down.
        """
        take_profit = BrokerOrder(
            broker_order_id="tp1",
            ticker="AAPL_US_EQ",
            side=Side.SELL,
            order_type=OrderType.LIMIT,
            status=OrderStatus.WORKING,
            quantity=Decimal("2"),
            limit_price=Decimal("200.00"),
        )
        snapshot = make_snapshot(
            positions=(Position(ticker="AAPL_US_EQ", quantity=Decimal("2")),),
            orders=(take_profit,),
        )
        report = _reconciler(ledger, pinned, snapshot).run()
        assert report.count(ReconcileFindingKind.UNPROTECTED_POSITION) == 1

    def test_a_stop_limit_does_count_as_protection(
        self, ledger: Ledger, pinned: PinnedLimits
    ) -> None:
        stop_limit = BrokerOrder(
            broker_order_id="sl1",
            ticker="AAPL_US_EQ",
            side=Side.SELL,
            order_type=OrderType.STOP_LIMIT,
            status=OrderStatus.WORKING,
            quantity=Decimal("2"),
            stop_price=Decimal("140.00"),
            limit_price=Decimal("139.00"),
        )
        snapshot = make_snapshot(
            positions=(Position(ticker="AAPL_US_EQ", quantity=Decimal("2")),),
            orders=(stop_limit,),
        )
        report = _reconciler(ledger, pinned, snapshot).run()
        assert report.count(ReconcileFindingKind.UNPROTECTED_POSITION) == 0

    def test_a_zero_quantity_position_needs_no_protection(
        self, ledger: Ledger, pinned: PinnedLimits
    ) -> None:
        snapshot = make_snapshot(positions=(Position(ticker="AAPL_US_EQ", quantity=Decimal("0")),))
        report = _reconciler(ledger, pinned, snapshot).run()
        assert report.count(ReconcileFindingKind.UNPROTECTED_POSITION) == 0


class TestOrphanOrders:
    def test_a_leftover_buy_is_blocking(self, ledger: Ledger, pinned: PinnedLimits) -> None:
        """The dangerous orphan.

        A buy limit left behind after an exit re-enters a position the system
        believes is closed — and does it silently, with no decision record.
        """
        stray = BrokerOrder(
            broker_order_id="stray1",
            ticker="MSFT_US_EQ",
            side=Side.BUY,
            order_type=OrderType.LIMIT,
            status=OrderStatus.WORKING,
            quantity=Decimal("1"),
            limit_price=Decimal("300.00"),
        )
        intents = FakeIntents(desired=(DesiredOrder("AAPL_US_EQ", Side.SELL, "protective_stop"),))
        report = _reconciler(ledger, pinned, make_snapshot(orders=(stray,)), intents=intents).run()

        finding = next(
            f for f in report.findings if f.kind is ReconcileFindingKind.ORPHAN_BUY_ORDER
        )
        assert finding.severity is Severity.BLOCKING
        assert "re-enter a position" in finding.detail
        assert finding.suggested_action == "cancel the order"

    def test_a_leftover_sell_is_a_warning(self, ledger: Ledger, pinned: PinnedLimits) -> None:
        stray = BrokerOrder(
            broker_order_id="stray2",
            ticker="MSFT_US_EQ",
            side=Side.SELL,
            order_type=OrderType.LIMIT,
            status=OrderStatus.WORKING,
            quantity=Decimal("1"),
        )
        intents = FakeIntents(desired=(DesiredOrder("AAPL_US_EQ", Side.BUY, "entry"),))
        report = _reconciler(ledger, pinned, make_snapshot(orders=(stray,)), intents=intents).run()
        finding = next(f for f in report.findings if f.kind is ReconcileFindingKind.ORPHAN_ORDER)
        assert finding.severity is Severity.WARN

    def test_a_traceable_order_is_not_an_orphan(self, ledger: Ledger, pinned: PinnedLimits) -> None:
        order = protective_stop("AAPL_US_EQ", Decimal("2"), order_id="known1")
        intents = FakeIntents(
            desired=(
                DesiredOrder("AAPL_US_EQ", Side.SELL, "protective_stop", broker_order_id="known1"),
            )
        )
        snapshot = make_snapshot(
            positions=(Position(ticker="AAPL_US_EQ", quantity=Decimal("2")),), orders=(order,)
        )
        report = _reconciler(ledger, pinned, snapshot, intents=intents).run()
        assert report.count(ReconcileFindingKind.ORPHAN_ORDER) == 0
        assert report.count(ReconcileFindingKind.ORPHAN_BUY_ORDER) == 0

    def test_with_no_intents_nothing_is_called_an_orphan(
        self, ledger: Ledger, pinned: PinnedLimits
    ) -> None:
        """A fresh install inherits whatever the account already holds.

        Reporting every pre-existing order as an orphan would be noise, not a
        finding — the concept only means something once we have intended
        something.
        """
        stray = BrokerOrder(
            broker_order_id="pre1",
            ticker="MSFT_US_EQ",
            side=Side.BUY,
            order_type=OrderType.LIMIT,
            status=OrderStatus.WORKING,
            quantity=Decimal("1"),
        )
        report = _reconciler(ledger, pinned, make_snapshot(orders=(stray,))).run()
        assert report.count(ReconcileFindingKind.ORPHAN_BUY_ORDER) == 0


class TestUnknownIntents:
    def test_an_unresolvable_intent_is_blocking_and_not_called_failed(
        self, ledger: Ledger, pinned: PinnedLimits
    ) -> None:
        """The most expensive mistake available.

        An intent whose fate cannot be established must stay unknown. Resolving
        it to "not placed" on the strength of an absence is how a crashed
        process places the same order twice.
        """
        intents = FakeIntents(
            unresolved=(
                UnresolvedIntent(
                    intent_id="int_abc123",
                    ticker="AAPL_US_EQ",
                    side=Side.BUY,
                    quantity=Decimal("1"),
                    wal_committed_at="2026-03-02T14:29:58.000000+0000",
                ),
            )
        )
        report = _reconciler(ledger, pinned, make_snapshot(), intents=intents).run()
        finding = next(f for f in report.findings if f.kind is ReconcileFindingKind.UNKNOWN_INTENT)
        assert finding.severity is Severity.BLOCKING
        assert "unknown, not failed" in finding.detail
        assert "double the fill" in finding.detail
        assert report.verdict is ReconcileVerdict.HALTED

    def test_an_intent_matching_a_live_order_is_resolved(
        self, ledger: Ledger, pinned: PinnedLimits
    ) -> None:
        order = protective_stop("AAPL_US_EQ", Decimal("1"), order_id="live1")
        intents = FakeIntents(
            unresolved=(
                UnresolvedIntent(
                    intent_id="int_abc123",
                    ticker="AAPL_US_EQ",
                    side=Side.SELL,
                    quantity=Decimal("1"),
                    wal_committed_at="2026-03-02T14:29:58.000000+0000",
                    broker_order_id="live1",
                ),
            ),
            desired=(
                DesiredOrder("AAPL_US_EQ", Side.SELL, "protective_stop", broker_order_id="live1"),
            ),
        )
        snapshot = make_snapshot(
            positions=(Position(ticker="AAPL_US_EQ", quantity=Decimal("1")),), orders=(order,)
        )
        report = _reconciler(ledger, pinned, snapshot, intents=intents).run()
        assert report.count(ReconcileFindingKind.UNKNOWN_INTENT) == 0


class TestCurrencyAndCrowding:
    def test_a_currency_mismatch_is_blocking(self, ledger: Ledger, pinned: PinnedLimits) -> None:
        """A ceiling of 500 means something very different in JPY."""
        report = _reconciler(ledger, pinned, make_snapshot(currency="JPY")).run()
        finding = next(
            f for f in report.findings if f.kind is ReconcileFindingKind.CURRENCY_MISMATCH
        )
        assert finding.severity is Severity.BLOCKING
        assert "wrong currency" in finding.detail
        assert "currency: JPY" in (finding.suggested_action or "")

    def test_a_matching_currency_is_silent(self, ledger: Ledger, pinned: PinnedLimits) -> None:
        report = _reconciler(ledger, pinned, make_snapshot(currency="GBP")).run()
        assert report.count(ReconcileFindingKind.CURRENCY_MISMATCH) == 0

    def test_pending_order_crowding_warns_well_before_the_ceiling(
        self, ledger: Ledger, pinned: PinnedLimits
    ) -> None:
        """Trading 212 caps pending orders at 50 per ticker.

        Hitting it does not merely block a new entry: it makes a protective
        stop rejected, turning a resource leak into an unhedged position.
        """
        orders = tuple(
            BrokerOrder(
                broker_order_id=f"o{i}",
                ticker="AAPL_US_EQ",
                side=Side.BUY,
                order_type=OrderType.LIMIT,
                status=OrderStatus.WORKING,
                quantity=Decimal("1"),
            )
            for i in range(30)
        )
        report = _reconciler(ledger, pinned, make_snapshot(orders=orders)).run()
        finding = next(
            f for f in report.findings if f.kind is ReconcileFindingKind.PENDING_ORDER_CROWDING
        )
        assert finding.severity is Severity.WARN
        assert "ceiling of 50" in finding.detail

    def test_crowding_becomes_blocking_near_the_ceiling(
        self, ledger: Ledger, pinned: PinnedLimits
    ) -> None:
        orders = tuple(
            BrokerOrder(
                broker_order_id=f"o{i}",
                ticker="AAPL_US_EQ",
                side=Side.SELL,
                order_type=OrderType.LIMIT,
                status=OrderStatus.WORKING,
                quantity=Decimal("1"),
            )
            for i in range(42)
        )
        report = _reconciler(ledger, pinned, make_snapshot(orders=orders)).run()
        finding = next(
            f for f in report.findings if f.kind is ReconcileFindingKind.PENDING_ORDER_CROWDING
        )
        assert finding.severity is Severity.BLOCKING


class TestSymbolMapIntegration:
    def test_holding_an_unmapped_instrument_warns_but_permits_exit(
        self, ledger: Ledger, pinned: PinnedLimits
    ) -> None:
        snapshot = make_snapshot(
            positions=(Position(ticker="MYSTERY_EQ", quantity=Decimal("1")),),
            orders=(protective_stop("MYSTERY_EQ", Decimal("1")),),
        )
        report = _reconciler(ledger, pinned, snapshot, with_symbol_map=True).run()
        finding = next(
            f for f in report.findings if f.kind is ReconcileFindingKind.UNMAPPED_HELD_POSITION
        )
        assert finding.severity is Severity.WARN
        assert "Exits are still permitted" in finding.detail


class TestStalenessAndVerdicts:
    def test_a_staleness_warning_is_surfaced(self, ledger: Ledger, pinned: PinnedLimits) -> None:
        snapshot = make_snapshot(warnings=("the reads span 42s because of rate limits",))
        report = _reconciler(ledger, pinned, snapshot).run()
        finding = next(
            f for f in report.findings if f.kind is ReconcileFindingKind.SNAPSHOT_NOT_ATOMIC
        )
        assert finding.severity is Severity.WARN

    def test_warnings_alone_do_not_halt(self, ledger: Ledger, pinned: PinnedLimits) -> None:
        snapshot = make_snapshot(warnings=("reads not simultaneous",))
        report = _reconciler(ledger, pinned, snapshot).run()
        assert report.verdict is ReconcileVerdict.CLEAN
        assert report.findings

    def test_a_read_only_reconciler_halts_rather_than_claiming_a_repair(
        self, ledger: Ledger, pinned: PinnedLimits
    ) -> None:
        """M1 can see everything and fix nothing, which is the point."""
        snapshot = make_snapshot(positions=(Position(ticker="AAPL_US_EQ", quantity=Decimal("2")),))
        report = _reconciler(ledger, pinned, snapshot).run(dry_run=True)
        assert report.verdict is ReconcileVerdict.HALTED
        assert report.actions == []


class TestRecording:
    def test_the_run_is_recorded_with_its_findings(
        self, ledger: Ledger, pinned: PinnedLimits
    ) -> None:
        snapshot = make_snapshot(positions=(Position(ticker="AAPL_US_EQ", quantity=Decimal("2")),))
        report = _reconciler(ledger, pinned, snapshot).run()

        row = ledger.conn.execute(
            "SELECT * FROM reconciliations WHERE recon_id = ?", (report.recon_id,)
        ).fetchone()
        assert row is not None
        assert row["verdict"] == "halted"
        assert row["n_unprotected_positions"] == 1
        assert "unprotected_position" in row["findings_json"]

        events = list(ledger.iter_events(event_type=EventType.RECONCILE_COMPLETED))
        assert len(events) == 1
        assert report.recon_id in events[0]["payload_json"]

    def test_persist_false_writes_nothing(self, ledger: Ledger, pinned: PinnedLimits) -> None:
        before = ledger.count()
        _reconciler(ledger, pinned, make_snapshot()).run(persist=False)
        assert ledger.count() == before
        assert ledger.conn.execute("SELECT COUNT(*) AS n FROM reconciliations").fetchone()["n"] == 0

    def test_the_event_and_its_projection_land_together(
        self, ledger: Ledger, pinned: PinnedLimits
    ) -> None:
        report = _reconciler(ledger, pinned, make_snapshot()).run()
        events = list(ledger.iter_events(event_type=EventType.RECONCILE_COMPLETED))
        rows = ledger.conn.execute("SELECT recon_id FROM reconciliations").fetchall()
        assert len(events) == len(rows) == 1
        assert rows[0]["recon_id"] == report.recon_id


class TestMultiplePositions:
    def test_findings_are_reported_per_ticker(self, ledger: Ledger, pinned: PinnedLimits) -> None:
        snapshot = make_snapshot(
            positions=(
                Position(ticker="AAPL_US_EQ", quantity=Decimal("2")),
                Position(ticker="MSFT_US_EQ", quantity=Decimal("3")),
                Position(ticker="GOOG_US_EQ", quantity=Decimal("1")),
            ),
            orders=(protective_stop("MSFT_US_EQ", Decimal("3"), order_id="s2"),),
        )
        report = _reconciler(ledger, pinned, snapshot).run()
        unprotected = {
            f.ticker for f in report.findings if f.kind is ReconcileFindingKind.UNPROTECTED_POSITION
        }
        assert unprotected == {"AAPL_US_EQ", "GOOG_US_EQ"}

    def test_the_summary_counts_by_severity(self, ledger: Ledger, pinned: PinnedLimits) -> None:
        snapshot = make_snapshot(
            positions=(Position(ticker="AAPL_US_EQ", quantity=Decimal("2")),),
            currency="JPY",
        )
        report = _reconciler(ledger, pinned, snapshot).run()
        assert "blocking" in report.summary()
        assert len(report.blocking) >= 2


def test_a_negative_position_is_refused_by_the_domain_type() -> None:
    """The Invest API cannot hold a short.

    So a negative quantity means the response was misparsed or the account is
    not the one we think it is, and neither is a state to reconcile against.
    """
    with pytest.raises(ValueError, match="cannot hold a short"):
        Position(ticker="AAPL_US_EQ", quantity=Decimal("-1"))
