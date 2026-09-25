"""The protective stop's whole life: placed, kept true to the position, withdrawn.

Trading 212 has no bracket or OCO orders, so the stop behind a position is an
ordinary working order the bot has to manage itself — and every way of getting
that wrong leaves real exposure behind. The audit before M7 found two:

* **An exit left its stop working.** Nothing in the engine ever cancelled an
  order, so a closed position's stop stayed at the venue as a sell for shares
  nobody held — one more per round trip, until the venue's pending-order ceiling
  refused the next position's stop, and in the meantime waiting to close
  whatever position came next at an old price.
* **Protection depended on the entry's response saying FILLED.** That is true
  of the simulator and not of the venue: a Trading 212 market order's POST
  answers before the fill, so a live entry would never have been protected.

So protection now follows the position, every cycle, and an exit withdraws the
stop before it sells. These tests drive the loop through each case with the
loop suite's own fixture, and read the answer off the simulated venue.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timedelta
from decimal import Decimal
from typing import Any

from tb.broker.port import OrderStatus
from tb.broker.simulated import SimulatedBroker
from tb.engine.funding import Book, explicit_book, unfunded_notional
from tb.engine.intents import IntentState
from tb.engine.loop import CycleResult
from tb.features.pipeline import FeaturePipeline
from tb.ledger.store import Ledger
from tb.strategy.base import Action, Decision, PositionState
from tb.strategy.trivial import specs
from tests.test_loop import AS_OF, TICKER, _broker, _loop, _rising_bars, _seed


@dataclass
class Scripted:
    """A strategy that does what it is told, in order, whatever the market says.

    `script` is consumed one step per decision: `enter`, `exit` or `hold`. Once
    it runs out the strategy holds. Deterministic on purpose — these tests are
    about what the loop does with a decision, not about how one is reached.
    """

    script: list[str]
    strategy_id: str = "scripted"
    version: int = 1
    _step: int = field(default=0, init=False)

    @property
    def required_features(self) -> tuple[str, ...]:
        return ()

    def decide(self, *, snapshot: Any, window: Any, position: PositionState) -> Decision:
        step = self.script[self._step] if self._step < len(self.script) else "hold"
        self._step += 1
        action = {"enter": Action.ENTER, "exit": Action.EXIT, "hold": Action.HOLD}[step]
        return Decision(
            as_of=window.as_of,
            instrument_uid=position.instrument_uid,
            action=action,
            expected_edge_bps=Decimal("300"),
            feature_snapshot_hash=snapshot.snapshot_hash,
            strategy_id=self.strategy_id,
            strategy_version=self.version,
            rationale=f"scripted {step}",
        )


def _book(env: dict[str, Any], script: list[str]) -> Book:
    return explicit_book(
        Scripted(script),
        FeaturePipeline(specs=specs()),
        notional_ccy=unfunded_notional(env["pinned"].limits, equity_ccy=Decimal("10000.00")),
    )


def _cycle(
    env: dict[str, Any],
    ledger: Ledger,
    broker: SimulatedBroker,
    book: Book,
    *,
    at: datetime,
    run_id: str,
) -> CycleResult:
    broker.clock = lambda: at
    loop = _loop(env, ledger, broker, book=book, run_id=run_id, at=at)
    loop.lock.acquire(at=at)
    try:
        return loop.run_cycle()
    finally:
        loop.lock.release()


def _stops(broker: SimulatedBroker) -> list[Decimal]:
    return [order.quantity or Decimal(0) for order in broker.protective_orders_for(TICKER)]


def _held(broker: SimulatedBroker) -> Decimal:
    position = broker.get_position(TICKER)
    return position.quantity if position is not None else Decimal(0)


LATER = AS_OF + timedelta(hours=3)


def test_an_exit_withdraws_its_stop_before_it_sells(env: dict[str, Any]) -> None:
    """**The stop goes first, then the shares.** Both halves are asserted:
    the exit went through — so the stop was not in its way — and nothing is left
    working afterwards."""
    _seed(env, _rising_bars(days=140))
    broker = _broker()
    book = _book(env, ["enter", "exit"])
    with Ledger(env["db"], config_hash=env["pinned"].config_hash) as ledger:
        entered = _cycle(env, ledger, broker, book, at=AS_OF, run_id="run_a")
        assert entered.stops_placed and _stops(broker) == [_held(broker)]

        exited = _cycle(env, ledger, broker, book, at=LATER, run_id="run_b")
        assert exited.submitted, f"the exit did not go out: {exited.refusals}"
        assert _held(broker) == 0
        assert _stops(broker) == [], "a stop was left working behind a closed position"

        (stop_intent,) = [
            row
            for row in ledger.conn.execute(
                "SELECT state, resolution_note FROM order_intents WHERE purpose = ?",
                ("protective_stop",),
            )
        ]
        assert stop_intent["state"] == IntentState.RESOLVED_CANCELLED.value
        assert "exit" in stop_intent["resolution_note"]
        cancelled = ledger.conn.execute(
            "SELECT COUNT(*) FROM event_log WHERE event_type = 'order.cancelled'"
        ).fetchone()[0]
        assert cancelled == 1


def test_a_stop_with_no_position_behind_it_is_withdrawn(env: dict[str, Any]) -> None:
    """The position went away without the bot — sold by hand in the app, say —
    and its stop stayed. Next cycle the stop is withdrawn rather than left to
    sell whatever is bought next."""
    _seed(env, _rising_bars(days=140))
    broker = _broker()
    book = _book(env, ["enter"])
    with Ledger(env["db"], config_hash=env["pinned"].config_hash) as ledger:
        _cycle(env, ledger, broker, book, at=AS_OF, run_id="run_a")
        assert _stops(broker)
        broker._positions.pop(TICKER)  # sold outside the bot

        swept = _cycle(env, ledger, broker, book, at=LATER, run_id="run_b")
        assert _stops(broker) == [], swept.refusals


def test_a_position_that_fills_after_its_entry_is_protected_next_cycle(
    env: dict[str, Any],
) -> None:
    """**The venue's actual behaviour.** A market order's POST answers before
    the fill, so the entry's response never says FILLED. Before this fix no
    stop was ever placed in that case; now the next cycle sees the position and
    protects exactly what is held."""
    _seed(env, _rising_bars(days=140))
    broker = _broker()
    broker.fill_on_accept = False
    book = _book(env, ["enter"])
    with Ledger(env["db"], config_hash=env["pinned"].config_hash) as ledger:
        entered = _cycle(env, ledger, broker, book, at=AS_OF, run_id="run_a")
        assert entered.submitted and not entered.stops_placed
        (entry,) = broker.live_orders_for(TICKER)
        broker.fill(entry.broker_order_id)
        assert _held(broker) > 0 and _stops(broker) == []

        protected = _cycle(env, ledger, broker, book, at=LATER, run_id="run_b")
        assert protected.stops_placed, protected.refusals
        assert _stops(broker) == [_held(broker)]


def test_a_stop_is_resized_when_the_position_changes(env: dict[str, Any]) -> None:
    """A stop for less leaves the difference unprotected; a stop for more is a
    sell for shares nobody holds. Either way it is replaced at the size held."""
    _seed(env, _rising_bars(days=140))
    broker = _broker()
    book = _book(env, ["enter"])
    with Ledger(env["db"], config_hash=env["pinned"].config_hash) as ledger:
        _cycle(env, ledger, broker, book, at=AS_OF, run_id="run_a")
        original = _held(broker)
        broker.seed_position(
            TICKER, quantity=original * 2, average_price=Decimal("150.00"), entered_at=AS_OF
        )

        _cycle(env, ledger, broker, book, at=LATER, run_id="run_b")
        assert _stops(broker) == [original * 2]


def test_an_exit_the_venue_refuses_is_re_protected_in_the_same_cycle(
    env: dict[str, Any],
) -> None:
    """The one window withdraw-then-sell opens is a refused sell after the stop
    is gone. It is closed immediately rather than left for the next cycle."""
    _seed(env, _rising_bars(days=140))
    broker = _broker()
    book = _book(env, ["enter", "exit"])
    with Ledger(env["db"], config_hash=env["pinned"].config_hash) as ledger:
        _cycle(env, ledger, broker, book, at=AS_OF, run_id="run_a")
        held = _held(broker)
        broker.reject_once[TICKER] = ("MarketClosed", "the venue refused the exit")

        refused = _cycle(env, ledger, broker, book, at=LATER, run_id="run_b")
        assert refused.refusals, "the exit was expected to be refused"
        assert _held(broker) == held
        assert _stops(broker) == [held], "a refused exit left the position unprotected"
        assert refused.stops_placed


def test_a_refused_exit_is_re_protected_at_the_level_its_stop_had(env: dict[str, Any]) -> None:
    """The same case at a venue that fills at the close the entry was sized
    from, as the paper venue does and the real one nearly does. The stop goes
    back at exactly its old level, size and decision — every field its
    withdrawn predecessor had — and was refused as a duplicate of it, leaving
    the position with no stop. The previous test passed only because its fill
    price differed from the bar."""
    _seed(env, _rising_bars(days=140))
    broker = _broker(price=Decimal("119.0"))  # the newest close visible at AS_OF
    book = _book(env, ["enter", "exit"])
    with Ledger(env["db"], config_hash=env["pinned"].config_hash) as ledger:
        _cycle(env, ledger, broker, book, at=AS_OF, run_id="run_a")
        held = _held(broker)
        (original,) = broker.protective_orders_for(TICKER)
        broker.reject_once[TICKER] = ("MarketClosed", "the venue refused the exit")

        refused = _cycle(env, ledger, broker, book, at=LATER, run_id="run_b")
        assert refused.stops_placed, refused.refusals
        (restored,) = broker.protective_orders_for(TICKER)
        assert restored.quantity == held
        assert restored.stop_price == original.stop_price
        assert restored.broker_order_id != original.broker_order_id


def test_an_exit_in_flight_is_not_re_protected_underneath(env: dict[str, Any]) -> None:
    """While an exit is working at the venue, placing a stop would reserve the
    very shares it is selling. The protection pass leaves the instrument alone
    until the exit settles."""
    _seed(env, _rising_bars(days=140))
    broker = _broker()
    book = _book(env, ["enter", "exit"])
    with Ledger(env["db"], config_hash=env["pinned"].config_hash) as ledger:
        _cycle(env, ledger, broker, book, at=AS_OF, run_id="run_a")
        broker.fill_on_accept = False
        _cycle(env, ledger, broker, book, at=LATER, run_id="run_b")
        working = broker.live_orders_for(TICKER)
        assert [o.status for o in working] == [OrderStatus.WORKING]
        assert not working[0].is_protective

        later = _cycle(env, ledger, broker, book, at=LATER + timedelta(hours=1), run_id="run_c")
        assert not later.stops_placed
        assert _stops(broker) == []
