"""Two gates every order reads, which the loop was not feeding.

The pre-M7 audit found both by reading the context the loop builds for the
risk rules, rather than the rules themselves:

* **The halt rule never saw a halt.** `RiskContext.halted` was never set, so
  `HaltedRule` passed every order it was ever shown. The cycle's preflight does
  refuse to start a pass while the kill switch is engaged — but a pass over a
  full universe against a rate-limited venue takes minutes, and a switch thrown
  during one did not stop the entries after it. Now every order reads the state
  machine's full permission check, and the rule does what it says: nothing
  risk-increasing while halted, and every exit and stop still allowed out.

* **The session-window refusal blamed the calendar for a closed market.** Every
  entry outside the session was refused with "without a calendar we cannot
  tell an open auction from a quiet midday" — true of an unclassifiable date,
  and misleading for the case that happens every night and weekend. The
  refusal now names why no session is open.

And two things the halt tests found in the loop on the way, both about which
instruments a pass offers to the book: an instrument exited in the first pass
was offered for entry in the second, on the same bar; and an EXIT decided about
a flat instrument raised out of the risk request and took the loop down.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from decimal import Decimal
from pathlib import Path
from typing import Any

import pytest

from tb.engine.funding import Book, explicit_book, unfunded_notional
from tb.engine.loop import LoopHalted
from tb.features.pipeline import FeaturePipeline
from tb.ledger.store import Ledger
from tb.ops.killswitch import engage_kill_switch
from tb.strategy.base import Action, Decision, PositionState
from tb.strategy.trivial import specs
from tests.test_loop import AS_OF, TICKER, _broker, _loop, _rising_bars, _seed
from tests.test_protection import Scripted, _cycle

LATER = AS_OF + timedelta(hours=3)


@dataclass
class ThrowsTheSwitch:
    """Decides `action`, throwing the kill switch while it does.

    Stands in for an operator running `tb halt` while a pass is under way: the
    preflight saw a clear switch, and the order this decision leads to is the
    first thing to happen after it was thrown. Shares the `Scripted` stub's
    identity, so it is the owner of a position that stub opened.
    """

    action: Action
    kill_switch: Path
    strategy_id: str = "scripted"
    version: int = 1

    @property
    def required_features(self) -> tuple[str, ...]:
        return ()

    def decide(self, *, snapshot: Any, window: Any, position: PositionState) -> Decision:
        engage_kill_switch(self.kill_switch, engaged_by="test", reason="thrown mid-pass")
        return Decision(
            as_of=window.as_of,
            instrument_uid=position.instrument_uid,
            action=self.action,
            expected_edge_bps=Decimal("300"),
            feature_snapshot_hash=snapshot.snapshot_hash,
            strategy_id=self.strategy_id,
            strategy_version=self.version,
            rationale=f"{self.action.value}, with the switch thrown",
        )


def _book_of(env: dict[str, Any], strategy: Any) -> Book:
    return explicit_book(
        strategy,
        FeaturePipeline(specs=specs()),
        notional_ccy=unfunded_notional(env["pinned"].limits, equity_ccy=Decimal("10000.00")),
    )


def _halted_verdicts(ledger: Ledger) -> list[tuple[str, str]]:
    return [
        (row["verdict"], row["detail"])
        for row in ledger.conn.execute(
            "SELECT v.verdict, v.detail FROM risk_verdicts v"
            " JOIN decisions d ON d.decision_id = v.decision_id"
            " WHERE v.rule_name = 'halted' AND d.rationale LIKE '%switch thrown'"
        )
    ]


def test_a_kill_switch_thrown_mid_pass_stops_the_next_entry(env: dict[str, Any]) -> None:
    """**Per order, not per cycle.** The entry decided after the throw is
    refused by the halt rule, with the switch named; and the next cycle does
    not start at all."""
    _seed(env, _rising_bars(days=140))
    broker = _broker()
    kill = Path(env["pinned"].limits.safety.kill_switch_path)
    book = _book_of(env, ThrowsTheSwitch(Action.ENTER, kill))
    with Ledger(env["db"], config_hash=env["pinned"].config_hash) as ledger:
        result = _cycle(env, ledger, broker, book, at=AS_OF, run_id="run_a")

        assert not result.submitted, "an entry went out after the kill switch was thrown"
        assert broker.get_positions() == ()
        ((verdict, detail),) = _halted_verdicts(ledger)
        assert verdict == "block"
        assert "kill switch is engaged" in detail

        with pytest.raises(LoopHalted, match="kill switch"):
            _cycle(env, ledger, broker, book, at=LATER, run_id="run_b")


def test_a_kill_switch_thrown_mid_pass_still_lets_an_exit_out(env: dict[str, Any]) -> None:
    """The other half of the rule, and the reason it exists rather than a halt
    that stops everything: a halt that blocked exits would trap the exposure it
    fired to limit. The exit decided after the throw goes out, its stop
    withdrawn first, and the halt rule records that it let it through."""
    _seed(env, _rising_bars(days=140))
    broker = _broker()
    kill = Path(env["pinned"].limits.safety.kill_switch_path)
    with Ledger(env["db"], config_hash=env["pinned"].config_hash) as ledger:
        _cycle(env, ledger, broker, _book_of(env, Scripted(["enter"])), at=AS_OF, run_id="run_a")
        assert broker.get_position(TICKER) is not None

        exiting = _book_of(env, ThrowsTheSwitch(Action.EXIT, kill))
        result = _cycle(env, ledger, broker, exiting, at=LATER, run_id="run_b")

        assert result.submitted, f"the exit was trapped by the halt: {result.refusals}"
        assert broker.get_position(TICKER) is None
        assert broker.get_open_orders() == ()
        ((verdict, detail),) = _halted_verdicts(ledger)
        assert verdict == "pass"
        assert "reduces risk" in detail


@pytest.mark.parametrize(
    ("at", "says"),
    [
        # A Saturday.
        (datetime(2026, 4, 4, 15, 30, tzinfo=UTC), "the market is closed on 2026-04-04, a weekend"),
        # Good Friday, a market holiday.
        (datetime(2026, 4, 3, 15, 30, tzinfo=UTC), "the market is closed on 2026-04-03, a holiday"),
        # 02:00 UTC on Wednesday is Tuesday evening in New York, so Tuesday's
        # session is the one that has closed — not Wednesday's, which has not
        # opened. The date is the exchange's, not UTC's.
        (datetime(2026, 4, 1, 2, 0, tzinfo=UTC), "2026-03-31's session runs"),
        # Past the hand-maintained holiday lists.
        (datetime(2028, 6, 1, 15, 30, tzinfo=UTC), "past the trading calendar's range"),
    ],
)
def test_a_closed_market_is_named_as_closed(env: dict[str, Any], at: datetime, says: str) -> None:
    _seed(env, _rising_bars(days=140))
    with Ledger(env["db"], config_hash=env["pinned"].config_hash) as ledger:
        loop = _loop(env, ledger, _broker(), at=at)
        since, until, note = loop._session_position(at)
    assert since is None and until is None
    assert says in note


def test_the_session_position_inside_the_session(env: dict[str, Any]) -> None:
    """11:30 in New York: two hours after the 09:30 open, four and a half
    before the 16:00 close."""
    _seed(env, _rising_bars(days=140))
    with Ledger(env["db"], config_hash=env["pinned"].config_hash) as ledger:
        loop = _loop(env, ledger, _broker())
        assert loop._session_position(AS_OF) == (120, 270, "")


def test_an_entry_on_a_closed_day_is_refused_for_that_reason(env: dict[str, Any]) -> None:
    """Through the rule, as recorded: the refusal an operator reads on a
    Saturday says the market is closed, not that the calendar is missing."""
    _seed(env, _rising_bars(days=140))
    saturday = datetime(2026, 4, 4, 15, 30, tzinfo=UTC)
    broker = _broker()
    with Ledger(env["db"], config_hash=env["pinned"].config_hash) as ledger:
        result = _cycle(
            env, ledger, broker, _book_of(env, Scripted(["enter"])), at=saturday, run_id="run_a"
        )
        assert not result.submitted
        (detail,) = [
            row["detail"]
            for row in ledger.conn.execute(
                "SELECT detail FROM risk_verdicts WHERE rule_name = 'session_window'"
            )
        ]
    assert detail.startswith("no regular session is open: the market is closed on 2026-04-04")
    assert "without a calendar" not in detail


def test_an_instrument_exited_this_cycle_is_not_bought_back_on_the_same_bar(
    env: dict[str, Any],
) -> None:
    """A strategy whose exit and entry both fire on one snapshot sold and
    rebought: the exit filled, the instrument read flat by the entry pass, and
    a second order went out against the bar the exit had just been decided on
    — a round trip's costs for no change in view."""
    _seed(env, _rising_bars(days=140))
    broker = _broker()
    with Ledger(env["db"], config_hash=env["pinned"].config_hash) as ledger:
        _cycle(env, ledger, broker, _book_of(env, Scripted(["enter"])), at=AS_OF, run_id="run_a")
        churn = _book_of(env, Scripted(["exit", "enter"]))
        result = _cycle(env, ledger, broker, churn, at=LATER, run_id="run_b")

    assert len(result.submitted) == 1, "sold and bought back on the same bar"
    assert broker.get_position(TICKER) is None


def test_an_exit_with_nothing_held_is_refused_rather_than_crashing_the_loop(
    env: dict[str, Any],
) -> None:
    _seed(env, _rising_bars(days=140))
    broker = _broker()
    with Ledger(env["db"], config_hash=env["pinned"].config_hash) as ledger:
        book = _book_of(env, Scripted(["exit"]))
        result = _cycle(env, ledger, broker, book, at=AS_OF, run_id="run_a")

    assert not result.submitted
    assert result.refusals == ((TICKER, "exit decided with nothing held"),)
