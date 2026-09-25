"""Settlement: what became of each order, and what each trade realised.

The pre-M7 audit found the learning half of the loop disconnected at its first
link. Nothing ever recorded a fill — `record_fill` had no caller — so every
intent stayed acknowledged forever, no realised result could be computed, and
nothing called `record_realised`. Every strategy showed zero trades; no loss
ever reached a lineage budget; the review could only ever say the evidence was
thin, and the allocator's realised edge was never measured.

Now each cycle opens by settling: every acknowledged order that has left the
venue's open list is read from order history, its fill recorded with the
venue's price, its intent resolved — and every closing fill is turned into a
realised result charged to the strategy whose decision stands behind it.
"""

from __future__ import annotations

import json
from dataclasses import replace
from datetime import UTC, date, datetime, timedelta
from decimal import Decimal
from typing import Any

from tb.broker.port import OrderPurpose, OrderStatus
from tb.engine.funding import Book, explicit_book, unfunded_notional
from tb.engine.intents import IntentLog, IntentState
from tb.engine.orders import OrderSubmitter
from tb.features.pipeline import FeaturePipeline
from tb.ledger.store import Ledger
from tb.portfolio.attribution import round_trip
from tb.registry.lineage import SpecRegistry
from tb.registry.models import AuthorKind
from tb.strategy.trivial import specs
from tests.test_funding import a_spec
from tests.test_loop import AS_OF, TICKER, _broker, _rising_bars, _seed
from tests.test_paper_venue import _daily, _ingest, _paper, _store
from tests.test_protection import LATER, Scripted, _cycle, _stops

LATEST = LATER + timedelta(hours=1)


def _registered(env: dict[str, Any], ledger: Ledger) -> tuple[SpecRegistry, str, int, str]:
    registry = SpecRegistry(
        ledger, per_lineage_budget_ccy=env["pinned"].limits.loss.per_lineage_budget_ccy
    )
    record = registry.register(a_spec(), author_kind=AuthorKind.SEARCH, at=AS_OF)
    return registry, record.strategy_id, record.version, record.lineage_id


def _book(env: dict[str, Any], script: list[str], *, strategy_id: str, version: int) -> Book:
    return explicit_book(
        Scripted(script, strategy_id=strategy_id, version=version),
        FeaturePipeline(specs=specs()),
        notional_ccy=unfunded_notional(env["pinned"].limits, equity_ccy=Decimal("10000.00")),
    )


def _intent_states(ledger: Ledger) -> dict[str, str]:
    return {
        str(row["purpose"]): str(row["state"])
        for row in ledger.conn.execute("SELECT purpose, state FROM order_intents")
    }


def test_a_round_trip_is_settled_and_charged_to_its_strategy(env: dict[str, Any]) -> None:
    """**The link that was missing.** Enter, exit, and one cycle later the
    strategy's record shows one trade and exactly what it realised — computed
    from the prices the venue reported for the two fills."""
    _seed(env, _rising_bars(days=140))
    broker = _broker(price=Decimal("119.0"))
    with Ledger(env["db"], config_hash=env["pinned"].config_hash) as ledger:
        registry, strategy_id, version, _ = _registered(env, ledger)
        book = _book(env, ["enter", "exit"], strategy_id=strategy_id, version=version)

        _cycle(env, ledger, broker, book, at=AS_OF, run_id="run_a")
        held = broker.get_position(TICKER)
        assert held is not None
        broker.prices[TICKER] = Decimal("125.0")

        exited = _cycle(env, ledger, broker, book, at=LATER, run_id="run_b")
        assert exited.submitted, exited.refusals
        assert len(exited.fills_recorded) == 1, "the entry's fill is read at the next cycle"

        settled = _cycle(env, ledger, broker, book, at=LATEST, run_id="run_c")
        assert len(settled.fills_recorded) == 1, "and the exit's at the one after"

        record = registry.status_of(strategy_id, version)
        assert record is not None
        assert record.n_realised_trades == 1
        assert record.realised_pnl_ccy == held.quantity * (Decimal("125.0") - Decimal("119.0"))
        assert _intent_states(ledger) == {
            "entry": IntentState.RESOLVED_FILLED.value,
            "exit": IntentState.RESOLVED_FILLED.value,
            "protective_stop": IntentState.RESOLVED_CANCELLED.value,
        }
        (trip,) = ledger.conn.execute("SELECT * FROM round_trips").fetchall()
        assert trip["charged"] == 1 and trip["strategy_id"] == strategy_id
        events = {
            row["event_type"] for row in ledger.conn.execute("SELECT event_type FROM event_log")
        }
        assert {"fill.recorded", "trade.closed"} <= events


def test_a_stop_that_fires_charges_the_loss_to_the_lineage(env: dict[str, Any]) -> None:
    """The case the lineage budget exists for. The market gaps through the
    stop; the stop's fill is charged to the strategy whose entry it protected —
    the loss the gap caused, at the price the stop actually got."""
    _seed(env, _rising_bars(days=140))
    later = datetime(2026, 4, 6, 15, 30, tzinfo=UTC)
    with Ledger(env["db"], config_hash=env["pinned"].config_hash) as ledger:
        registry, strategy_id, version, lineage_id = _registered(env, ledger)
        store = _store(env, ledger)
        broker = _paper(store)
        book = _book(env, ["enter"], strategy_id=strategy_id, version=version)
        _cycle(env, ledger, broker, book, at=AS_OF, run_id="run_a")
        held = broker.get_position(TICKER)
        assert held is not None and held.average_price is not None

        _ingest(store, _daily(date(2026, 4, 2), "90.00"))
        _cycle(env, ledger, broker, book, at=later, run_id="run_b")

        loss = held.quantity * (held.average_price - Decimal("90.00"))
        record = registry.status_of(strategy_id, version)
        assert record is not None
        assert record.n_realised_trades == 1
        assert record.realised_pnl_ccy == -loss
        budget = registry.budget_for(lineage_id)
        assert budget is not None and budget.consumed_ccy == loss


def test_an_order_history_has_not_reported_yet_is_read_again_next_cycle(
    env: dict[str, Any],
) -> None:
    """History trails fills and is rationed, so an order can leave the open
    list before history says what became of it. That absence is not evidence
    of anything: the intent stays acknowledged, and settles when history
    catches up."""
    _seed(env, _rising_bars(days=140))
    broker = _broker(price=Decimal("119.0"))
    with Ledger(env["db"], config_hash=env["pinned"].config_hash) as ledger:
        _, strategy_id, version, _ = _registered(env, ledger)
        book = _book(env, ["enter"], strategy_id=strategy_id, version=version)
        _cycle(env, ledger, broker, book, at=AS_OF, run_id="run_a")

        broker.publish_history = False
        quiet = _cycle(env, ledger, broker, book, at=LATER, run_id="run_b")
        assert quiet.fills_recorded == ()
        assert _intent_states(ledger)["entry"] == IntentState.ACKNOWLEDGED.value

        broker.publish_history = True
        caught_up = _cycle(env, ledger, broker, book, at=LATEST, run_id="run_c")
        assert len(caught_up.fills_recorded) == 1
        assert _intent_states(ledger)["entry"] == IntentState.RESOLVED_FILLED.value


def test_a_stop_cancelled_in_the_app_is_settled_and_replaced(env: dict[str, Any]) -> None:
    """Cancelled by hand, at the venue. Settlement reads it as cancelled with
    nothing traded, and the protection pass that follows puts a real stop back
    — a new order, where before the cancelled one was handed back as if it
    were still protecting the position."""
    _seed(env, _rising_bars(days=140))
    broker = _broker(price=Decimal("119.0"))
    with Ledger(env["db"], config_hash=env["pinned"].config_hash) as ledger:
        _, strategy_id, version, _ = _registered(env, ledger)
        book = _book(env, ["enter"], strategy_id=strategy_id, version=version)
        _cycle(env, ledger, broker, book, at=AS_OF, run_id="run_a")
        (stop,) = broker.protective_orders_for(TICKER)
        broker._orders[stop.broker_order_id] = replace(stop, status=OrderStatus.CANCELLED)

        later = _cycle(env, ledger, broker, book, at=LATER, run_id="run_b")
        assert later.stops_placed, later.refusals
        (replacement,) = broker.protective_orders_for(TICKER)
        assert replacement.broker_order_id != stop.broker_order_id
        position = broker.get_position(TICKER)
        assert position is not None and _stops(broker) == [position.quantity]
        states = [
            str(row["state"])
            for row in ledger.conn.execute(
                "SELECT state FROM order_intents WHERE purpose = 'protective_stop'"
                " ORDER BY committing_event_seq"
            )
        ]
        assert states == [IntentState.RESOLVED_CANCELLED.value, IntentState.ACKNOWLEDGED.value]


def test_a_fill_recorded_before_a_crash_is_not_recorded_twice(env: dict[str, Any]) -> None:
    """A pass that died between writing a fill and resolving its intent leaves
    the intent acknowledged with its fill already recorded. The next pass
    resolves it without writing a second: a doubled fill would double the trade
    in every number built on fills."""
    _seed(env, _rising_bars(days=140))
    broker = _broker(price=Decimal("119.0"))
    with Ledger(env["db"], config_hash=env["pinned"].config_hash) as ledger:
        _, strategy_id, version, _ = _registered(env, ledger)
        book = _book(env, ["enter"], strategy_id=strategy_id, version=version)
        _cycle(env, ledger, broker, book, at=AS_OF, run_id="run_a")
        log = IntentLog(ledger, run_id="run_crashed")
        (entry,) = [i for i in log.for_ticker(TICKER) if i.purpose is OrderPurpose.ENTRY]
        OrderSubmitter(ledger=ledger, broker=broker, log=log, run_id="run_crashed").record_fill(
            intent=entry, quantity=entry.quantity, price=Decimal("119.0"), source="api_history"
        )
        # ... and the pass died here, before resolving the intent.

        _cycle(env, ledger, broker, book, at=LATER, run_id="run_b")
        (count,) = ledger.conn.execute(
            "SELECT COUNT(*) FROM fills WHERE intent_id = ?", (entry.intent_id,)
        ).fetchone()
        assert count == 1
        assert _intent_states(ledger)["entry"] == IntentState.RESOLVED_FILLED.value


def test_the_acknowledgement_records_what_the_venue_said(env: dict[str, Any]) -> None:
    """A venue that fills on accept answers FILLED, and the ledger said WORKING."""
    _seed(env, _rising_bars(days=140))
    broker = _broker(price=Decimal("119.0"))
    with Ledger(env["db"], config_hash=env["pinned"].config_hash) as ledger:
        _, strategy_id, version, _ = _registered(env, ledger)
        book = _book(env, ["enter"], strategy_id=strategy_id, version=version)
        _cycle(env, ledger, broker, book, at=AS_OF, run_id="run_a")
        statuses = [
            json.loads(row["payload_json"])["status"]
            for row in ledger.conn.execute(
                "SELECT payload_json FROM event_log WHERE event_type = 'order.acknowledged'"
                " ORDER BY seq"
            )
        ]
    assert statuses == ["filled", "working"], "the entry filled; its stop is working"


# --------------------------------------------------------------------------
# The arithmetic, on fills written directly
# --------------------------------------------------------------------------


def _fill(
    ledger: Ledger,
    seq: int,
    side: str,
    quantity: str,
    price: str | None,
    *,
    fees: dict[str, str] | None = None,
) -> str:
    fill_id = f"fill_{seq}"
    ledger.conn.execute(
        "INSERT INTO fills (fill_id, intent_id, broker_order_id, t212_ticker, instrument_uid,"
        " side, quantity, price, filled_at, fees_json, fx_rate, source, confidence,"
        " admissible_for_pnl, recorded_at, recording_event_seq)"
        " VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
        (
            fill_id,
            None,
            None,
            TICKER,
            None,
            side,
            quantity,
            price,
            AS_OF.isoformat(),
            None if fees is None else json.dumps(fees),
            None,
            "api_history" if price is not None else "inferred_from_position_delta",
            "observed" if price is not None else "inferred",
            1 if price is not None else 0,
            AS_OF.isoformat(),
            seq,
        ),
    )
    return fill_id


def test_average_cost_with_charges_across_adds_and_partial_exits(ledger: Ledger) -> None:
    """Two buys and two sells. The first sell realises against the average of
    both buys, the buys' charges in the basis and its own charge deducted; the
    second sells the rest at the same average."""
    _fill(ledger, 1, "buy", "2", "100", fees={"CURRENCY_CONVERSION_FEE": "1.00"})
    _fill(ledger, 2, "buy", "2", "110")
    first = _fill(ledger, 3, "sell", "3", "120", fees={"CURRENCY_CONVERSION_FEE": "0.50"})
    second = _fill(ledger, 4, "sell", "1", "90")

    basis = Decimal("421") / 4  # (200 + 1 + 220) / 4
    trip = round_trip(ledger, fill_id=first)
    assert trip.admissible and trip.cost_basis == basis
    assert trip.pnl_ccy == 3 * (Decimal("120") - basis) - Decimal("0.50")
    rest = round_trip(ledger, fill_id=second)
    assert rest.admissible and rest.pnl_ccy == Decimal("90") - basis


def test_an_unpriced_entry_makes_its_round_trip_inadmissible(ledger: Ledger) -> None:
    """A guessed basis would teach the allocator an edge nobody earned. Once
    the holding goes flat, the next one is measured afresh."""
    _fill(ledger, 1, "buy", "1", None)
    blind = _fill(ledger, 2, "sell", "1", "120")
    _fill(ledger, 3, "buy", "1", "100")
    seen = _fill(ledger, 4, "sell", "1", "105")

    assert not round_trip(ledger, fill_id=blind).admissible
    measured = round_trip(ledger, fill_id=seen)
    assert measured.admissible and measured.pnl_ccy == Decimal("5")


def test_a_sell_beyond_the_recorded_buys_is_not_charged(ledger: Ledger) -> None:
    _fill(ledger, 1, "buy", "1", "100")
    over = _fill(ledger, 2, "sell", "2", "110")
    trip = round_trip(ledger, fill_id=over)
    assert not trip.admissible
    assert "basis" in trip.detail
