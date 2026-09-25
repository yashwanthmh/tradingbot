"""The paper venue: a market the loop can actually be wrong against.

`tb run --mode paper` used to build the simulator with no prices at all. The
pre-M7 audit drove it end to end and found what that meant:

* **every paper fill was at the simulator's 100.00 stand-in**, whatever the bars
  said — so a paper position's size and its stop level were computed from one
  price and its cost basis recorded at another;
* **every position was marked at its own fill**, and equity was a constant, so
  none of the three loss breakers could ever fire in a paper run;
* **no stop ever fired**, so protection was placed on every paper position and
  exercised on none.

An overnight paper run proved the order path and none of the numbers the risk
rules read. Now the paper venue is marked to the newest close the loop itself
can see (`BarMarks`), and its account follows that market.
"""

from __future__ import annotations

from dataclasses import replace
from datetime import UTC, date, datetime, timedelta
from decimal import Decimal
from typing import Any

from tb.broker.port import OrderStatus
from tb.broker.simulated import BarMarks, SimulatedBroker
from tb.cli_engine import _broker as broker_for_mode
from tb.cli_engine import _price_paper_venue
from tb.data.asof import InMemoryBarSource, visible_bars
from tb.data.barstore import BarStore
from tb.data.provider import Bar, BarBatch, Provenance, Resolution, Session
from tb.ledger.store import Ledger
from tb.portfolio.pnl import EquityCurve
from tests.test_loop import AS_OF, TICKER, UID, _rising_bars, _seed
from tests.test_protection import _book, _cycle

# The newest close visible at AS_OF. The fixture's last bar, for 1 April, is
# not knowable until the 2nd — a paper fill at its 119.50 would be a lookahead.
DECIDED_ON = Decimal("119.0")


def _store(env: dict[str, Any], ledger: Ledger) -> BarStore:
    return BarStore(ledger, root=env["bars"], scale=env["pinned"].limits.data.price_scale)


def _paper(store: BarStore) -> SimulatedBroker:
    """The simulator as `tb run --mode paper` builds it, clock pinned."""
    broker = SimulatedBroker(
        environment="paper",
        currency="GBP",
        equity=Decimal("10000.00"),
        free_cash=Decimal("10000.00"),
        min_trade_quantity={TICKER: Decimal("0.1")},
        clock=lambda: AS_OF,
        mark_to_market=True,
    )
    broker.price_source = BarMarks(
        bars=store,
        instruments={TICKER: UID},
        resolution=Resolution.DAILY,
        clock=lambda: broker.clock(),
    )
    return broker


def _daily(day: date, close: str, *, uid: str = UID) -> Bar:
    opened = datetime(day.year, day.month, day.day, tzinfo=UTC)
    price = Decimal(close)
    return Bar(
        instrument_uid=uid,
        resolution=Resolution.DAILY,
        bar_open_utc=opened,
        available_at_utc=opened + timedelta(days=1),
        ingested_at_utc=opened + timedelta(days=1),
        provider="alpaca",
        provenance=Provenance.BACKFILL,
        session=Session.REGULAR,
        open=price,
        high=price + 1,
        low=price - 1,
        close=price,
        volume=1_000_000,
    )


def _ingest(store: BarStore, bar: Bar) -> None:
    store.ingest(
        BarBatch(
            bars=(bar,),
            provider="alpaca",
            symbol="AAPL",
            resolution=Resolution.DAILY,
            requested_start=bar.bar_open_utc,
            requested_end=bar.bar_open_utc,
        )
    )


def test_a_paper_fill_is_at_the_close_the_loop_decided_on(env: dict[str, Any]) -> None:
    """**The same price for the decision, the size, the stop and the fill.**
    Not the 100.00 stand-in, and not the next bar, which was not knowable yet."""
    _seed(env, _rising_bars(days=140))
    with Ledger(env["db"], config_hash=env["pinned"].config_hash) as ledger:
        store = _store(env, ledger)
        assert visible_bars(store, UID, Resolution.DAILY, as_of=AS_OF)[-1].close == DECIDED_ON
        broker = _paper(store)

        entered = _cycle(env, ledger, broker, _book(env, ["enter"]), at=AS_OF, run_id="run_a")
        assert entered.submitted and entered.stops_placed, entered.refusals

    position = broker.get_position(TICKER)
    assert position is not None
    assert position.average_price == DECIDED_ON
    cash = broker.get_cash()
    assert cash.free == Decimal("10000.00") - position.quantity * DECIDED_ON
    assert cash.total == Decimal("10000.00"), "marked at its own fill, nothing has moved yet"


def test_a_paper_stop_fires_when_the_market_gaps_through_it(env: dict[str, Any]) -> None:
    """The protection every paper position carried and none ever used. The
    market gaps from 119 to 90, through a stop 15% below the entry: the stop
    sells at 90 — the gap, not the level — the account shows the loss, and the
    loop finds nothing left working afterwards."""
    _seed(env, _rising_bars(days=140))
    later = datetime(2026, 4, 6, 15, 30, tzinfo=UTC)
    with Ledger(env["db"], config_hash=env["pinned"].config_hash) as ledger:
        store = _store(env, ledger)
        broker = _paper(store)
        book = _book(env, ["enter"])
        _cycle(env, ledger, broker, book, at=AS_OF, run_id="run_a")
        held = broker.get_position(TICKER)
        assert held is not None
        (stop,) = broker.protective_orders_for(TICKER)
        assert stop.stop_price is not None and stop.stop_price > Decimal("90")

        _ingest(store, _daily(date(2026, 4, 2), "90.00"))
        after = _cycle(env, ledger, broker, book, at=later, run_id="run_b")

        assert broker.get_position(TICKER) is None, "the stop did not fire"
        assert broker.get_open_orders() == (), after.refusals
        fired = broker.get_order(stop.broker_order_id)
        assert fired is not None and fired.status is OrderStatus.FILLED
        loss = held.quantity * (DECIDED_ON - Decimal("90.00"))
        cash = broker.get_cash()
        assert cash.total == cash.free == Decimal("10000.00") - loss

        # What the breakers read: the curve marked the loss. Drawdown is a
        # magnitude below the peak; the rolling change is signed.
        reading = EquityCurve(ledger, run_id="run_b").read(at=later)
        expected = float(loss / Decimal("10000.00") * 100)
        assert reading.drawdown_from_peak_pct is not None and reading.rolling_pnl_pct is not None
        assert abs(reading.drawdown_from_peak_pct - expected) < 1e-9
        assert abs(reading.rolling_pnl_pct + expected) < 1e-9


def test_bar_marks_never_price_from_a_bar_before_it_is_knowable() -> None:
    """The lookahead guard on the venue's own price, memo included: a memo can
    be staler than a fresh read but never newer, even when the clock goes back."""
    source = InMemoryBarSource()
    # Knowable thirty seconds into the minute, so a memo read inside that
    # minute is on one side of it or the other.
    knowable = datetime(2026, 4, 1, 0, 0, 30, tzinfo=UTC)
    source.add(
        _daily(date(2026, 3, 30), "100.00"),
        replace(
            _daily(date(2026, 3, 31), "110.00"),
            available_at_utc=knowable,
            ingested_at_utc=knowable,
        ),
    )
    now = [datetime(2026, 4, 1, 0, 0, 10, tzinfo=UTC)]
    marks = BarMarks(
        bars=source, instruments={TICKER: UID}, resolution=Resolution.DAILY, clock=lambda: now[0]
    )

    assert marks(TICKER) == Decimal("100.00"), "the 31 March bar is not knowable yet"
    now[0] = datetime(2026, 4, 1, 0, 1, 5, tzinfo=UTC)
    assert marks(TICKER) == Decimal("110.00")
    now[0] = datetime(2026, 4, 1, 0, 0, 20, tzinfo=UTC)
    assert marks(TICKER) == Decimal("100.00"), "a memo newer than the clock was returned"
    assert marks("NOT_IN_THE_UNIVERSE") is None


def test_tb_run_prices_its_paper_venue_from_the_bar_store(env: dict[str, Any]) -> None:
    """The wiring: `tb run --mode paper` builds a marked-to-market venue and
    prices it from the store it trades from, through the venue's own clock."""
    _seed(env, _rising_bars(days=140))
    broker = broker_for_mode("paper", env["pinned"], equity=Decimal("10000.00"))
    assert isinstance(broker, SimulatedBroker) and broker.mark_to_market

    with Ledger(env["db"], config_hash=env["pinned"].config_hash) as ledger:
        _price_paper_venue(
            broker, store=_store(env, ledger), universe={TICKER: UID}, resolution="daily"
        )
        # Pinned after the wiring, which the marks must follow.
        broker.clock = lambda: AS_OF
        assert broker.price_source is not None
        assert broker.price_source(TICKER) == DECIDED_ON
