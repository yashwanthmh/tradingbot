"""The run state machine and the trading permission gate.

The property that matters most: **only `TRADING` may place orders**, and every
blocking condition is reported at once rather than one restart at a time.
"""

from __future__ import annotations

from collections.abc import Callable
from decimal import Decimal
from pathlib import Path
from typing import Any

import pytest

from tb.config.loader import PinnedLimits, load_hard_limits
from tb.core.errors import HaltRequired, Killed, TbError
from tb.ledger.events import EventType
from tb.ledger.store import Ledger
from tb.ops.killswitch import KillSwitchState, engage_kill_switch, release_kill_switch
from tb.ops.state import InvalidTransition, RunState, StateMachine


@pytest.fixture
def machine(ledger: Ledger, pinned_in_tmp: PinnedLimits) -> StateMachine:
    return StateMachine(ledger, pinned_in_tmp, run_id="run_test")


@pytest.fixture
def pinned_in_tmp(tmp_path: Path, write_limits: Callable[[dict[str, Any]], Path]) -> PinnedLimits:
    """Limits whose kill switch and heartbeat live under tmp_path.

    Without this the tests would read the developer's real `var/run/KILL`, and a
    test suite that can be influenced by the state of the working tree is not a
    test suite.
    """
    run_dir = tmp_path / "run"
    run_dir.mkdir(exist_ok=True)
    path = write_limits(
        {
            "safety": {
                "kill_switch_path": str(run_dir / "KILL"),
                "heartbeat_path": str(run_dir / "heartbeat"),
            }
        }
    )
    return load_hard_limits(path)


class TestTransitions:
    def test_a_fresh_ledger_starts_in_boot(self, machine: StateMachine) -> None:
        assert machine.current().state is RunState.BOOT

    def test_the_happy_path_is_boot_reconciling_trading(self, machine: StateMachine) -> None:
        machine.transition_to(RunState.RECONCILING, reason="startup")
        assert machine.current().state is RunState.RECONCILING
        machine.transition_to(RunState.TRADING, reason="reconciled clean")
        assert machine.current().state is RunState.TRADING

    def test_boot_cannot_jump_straight_to_trading(self, machine: StateMachine) -> None:
        """Reconciliation is not skippable.

        Until the reconciler has established what the account actually holds,
        every order is a guess about an unknown position.
        """
        with pytest.raises(InvalidTransition, match="cannot go from boot to trading"):
            machine.transition_to(RunState.TRADING, reason="impatience")

    def test_a_halt_returns_through_reconciling_not_straight_to_trading(
        self, machine: StateMachine
    ) -> None:
        """While we were stopped, the account may have moved.

        A stop may have filled; a position may have been closed by hand.
        Resuming straight into trading would act on a stale picture.
        """
        machine.transition_to(RunState.RECONCILING, reason="startup")
        machine.transition_to(RunState.TRADING, reason="ok")
        machine.transition_to(RunState.HALTED, reason="breaker")
        with pytest.raises(InvalidTransition, match="cannot go from halted to trading"):
            machine.transition_to(RunState.TRADING, reason="carry on")
        machine.transition_to(RunState.RECONCILING, reason="cleared")

    def test_transitioning_to_the_current_state_is_a_no_op(self, machine: StateMachine) -> None:
        machine.transition_to(RunState.RECONCILING, reason="startup")
        before = machine._ledger.count()
        machine.transition_to(RunState.RECONCILING, reason="again")
        assert machine._ledger.count() == before

    def test_every_transition_is_recorded_in_the_ledger(
        self, machine: StateMachine, ledger: Ledger
    ) -> None:
        machine.transition_to(RunState.RECONCILING, reason="startup")
        machine.transition_to(RunState.TRADING, reason="clean")
        events = list(ledger.iter_events(event_type=EventType.STATE_TRANSITIONED))
        assert len(events) == 2
        assert '"to_state":"trading"' in events[-1]["payload_json"]

    def test_the_unreconciled_halt_is_a_distinct_state(self, machine: StateMachine) -> None:
        """Separate from HALTED because it needs a person.

        A normal halt clears when its condition passes. An unreconciled one
        means we cannot account for something that may be a real position, and
        guessing about that is worse than waiting.
        """
        assert RunState.HALT_UNRECONCILED.requires_human_ack
        assert not RunState.HALTED.requires_human_ack

    def test_only_trading_may_place_orders(self) -> None:
        permitted = [s for s in RunState if s.may_place_orders]
        assert permitted == [RunState.TRADING]

    def test_reconciling_may_not_place_orders_even_protective_ones(self) -> None:
        # Stated explicitly because it is the tempting exception: "surely we can
        # at least place a stop". Not before we know what position it protects.
        assert not RunState.RECONCILING.may_place_orders


class TestHalts:
    def test_raising_a_halt_stops_trading_and_records_both(
        self, machine: StateMachine, ledger: Ledger
    ) -> None:
        machine.transition_to(RunState.RECONCILING, reason="startup")
        machine.transition_to(RunState.TRADING, reason="clean")

        halt_id = machine.raise_halt(
            "daily_loss",
            "day is -2.4% against a 2.0% limit",
            observed_value=Decimal("-2.4"),
            limit_value=Decimal("2.0"),
        )

        assert machine.current().state is RunState.HALTED
        row = ledger.conn.execute("SELECT * FROM halts WHERE halt_id = ?", (halt_id,)).fetchone()
        assert row is not None
        # The numbers are worth the columns: "the breaker fired" is far less
        # useful six weeks later than "fired at -2.4% against 2.0%".
        assert row["observed_value"] == "-2.4"
        assert row["limit_value"] == "2.0"
        assert row["cleared_at"] is None

    def test_an_unreconciled_halt_lands_in_its_own_state(self, machine: StateMachine) -> None:
        machine.raise_halt("unreconciled", "intent int_abc has unknown state", unreconciled=True)
        assert machine.current().state is RunState.HALT_UNRECONCILED

    def test_open_halts_are_listed(self, machine: StateMachine) -> None:
        machine.raise_halt("daily_loss", "first")
        machine.raise_halt("anomaly", "second")
        assert {h.trigger for h in machine.open_halts()} == {"daily_loss", "anomaly"}

    def test_clearing_a_halt_does_not_resume_trading(self, machine: StateMachine) -> None:
        """Clearing acknowledges; it does not restart."""
        halt_id = machine.raise_halt("daily_loss", "-2.4%")
        machine.clear_halt(halt_id, cleared_by="alex", clear_reason="new session")
        assert machine.open_halts() == ()
        assert machine.current().state is RunState.HALTED

    def test_clearing_is_idempotent(self, machine: StateMachine) -> None:
        halt_id = machine.raise_halt("manual", "stop")
        machine.clear_halt(halt_id, cleared_by="a", clear_reason="x")
        machine.clear_halt(halt_id, cleared_by="b", clear_reason="y")
        row = machine._ledger.conn.execute(
            "SELECT cleared_by FROM halts WHERE halt_id = ?", (halt_id,)
        ).fetchone()
        assert row["cleared_by"] == "a"

    def test_clearing_an_unknown_halt_raises(self, machine: StateMachine) -> None:
        with pytest.raises(TbError, match="no such halt"):
            machine.clear_halt("halt_nope", cleared_by="a", clear_reason="x")


class TestTradingPermission:
    def test_trading_is_permitted_when_everything_is_clean(self, machine: StateMachine) -> None:
        machine.transition_to(RunState.RECONCILING, reason="startup")
        machine.transition_to(RunState.TRADING, reason="clean")
        permission = machine.check_trading_permission()
        assert permission.allowed, permission.reasons
        assert permission.kill_switch is KillSwitchState.CLEAR

    def test_boot_state_blocks_trading(self, machine: StateMachine) -> None:
        permission = machine.check_trading_permission()
        assert not permission.allowed
        assert any("run state is boot" in r for r in permission.reasons)

    def test_the_kill_switch_blocks_trading(self, machine: StateMachine) -> None:
        machine.transition_to(RunState.RECONCILING, reason="startup")
        machine.transition_to(RunState.TRADING, reason="clean")
        engage_kill_switch(machine.kill_switch_path, engaged_by="ops", reason="drill")
        try:
            permission = machine.check_trading_permission()
            assert not permission.allowed
            assert permission.kill_switch is KillSwitchState.ENGAGED
        finally:
            release_kill_switch(machine.kill_switch_path)

    def test_config_drift_blocks_trading(self, machine: StateMachine) -> None:
        """Editing the limits under a running process stops it.

        Silently adopting different caps mid-flight is how a bot ends up
        trading under limits nobody reviewed.
        """
        machine.transition_to(RunState.RECONCILING, reason="startup")
        machine.transition_to(RunState.TRADING, reason="clean")

        path = machine._pinned.source_path
        path.write_text(
            path.read_text(encoding="utf-8").replace(
                "max_orders_per_day: 40", "max_orders_per_day: 4000"
            ),
            encoding="utf-8",
        )
        permission = machine.check_trading_permission()
        assert not permission.allowed
        assert permission.config_drift is not None
        assert any("changed while running" in r for r in permission.reasons)

    def test_all_blocking_reasons_are_reported_at_once(self, machine: StateMachine) -> None:
        """Not short-circuited.

        "Why is it not trading" should be answerable on the first look, rather
        than discovered one restart at a time.
        """
        machine.raise_halt("daily_loss", "-2.4%")
        engage_kill_switch(machine.kill_switch_path, engaged_by="ops", reason="drill")
        try:
            permission = machine.check_trading_permission()
            assert not permission.allowed
            assert len(permission.reasons) >= 3
            joined = " | ".join(permission.reasons)
            assert "halted" in joined
            assert "kill switch" in joined
            assert "daily_loss" in joined
        finally:
            release_kill_switch(machine.kill_switch_path)

    def test_raise_if_blocked_prefers_the_kill_switch_reason(self, machine: StateMachine) -> None:
        engage_kill_switch(machine.kill_switch_path, engaged_by="ops", reason="drill")
        try:
            with pytest.raises(Killed):
                machine.check_trading_permission().raise_if_blocked()
        finally:
            release_kill_switch(machine.kill_switch_path)

    def test_raise_if_blocked_raises_halt_for_a_breaker(self, machine: StateMachine) -> None:
        machine.raise_halt("drawdown", "-6.2% from high water")
        with pytest.raises(HaltRequired, match="drawdown"):
            machine.check_trading_permission().raise_if_blocked()

    def test_raise_if_blocked_is_silent_when_allowed(self, machine: StateMachine) -> None:
        machine.transition_to(RunState.RECONCILING, reason="startup")
        machine.transition_to(RunState.TRADING, reason="clean")
        machine.check_trading_permission().raise_if_blocked()

    def test_an_undeterminable_kill_switch_blocks_trading(
        self,
        ledger: Ledger,
        tmp_path: Path,
        write_limits: Callable[[dict[str, Any]], Path],
    ) -> None:
        """A switch in a directory that does not exist blocks trading."""
        path = write_limits(
            {
                "safety": {
                    "kill_switch_path": str(tmp_path / "never-mounted" / "KILL"),
                    "heartbeat_path": str(tmp_path / "never-mounted" / "heartbeat"),
                }
            }
        )
        machine = StateMachine(ledger, load_hard_limits(path), run_id="run_x")
        machine.transition_to(RunState.RECONCILING, reason="startup")
        machine.transition_to(RunState.TRADING, reason="clean")

        permission = machine.check_trading_permission()
        assert not permission.allowed
        assert permission.kill_switch is KillSwitchState.UNDETERMINABLE
        with pytest.raises(Killed):
            permission.raise_if_blocked()
