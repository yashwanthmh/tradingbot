"""The `tb` control plane.

These are the commands an operator reaches for during an incident, so the
things worth asserting are: exit codes are meaningful, `status` explains *why*
it will not trade, `resume` refuses to blow past a safety halt without being
told to, and nothing ever prints a credential.
"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest
from typer.testing import CliRunner

from tb.cli import app
from tb.ops.secrets import DEMO_KEY_VAR, LIVE_KEY_VAR

runner = CliRunner()


@pytest.fixture
def env(tmp_path: Path, write_limits: Callable[[dict[str, Any]], Path]) -> dict[str, Any]:
    """An isolated control-plane environment under tmp_path."""
    run_dir = tmp_path / "run"
    run_dir.mkdir(exist_ok=True)
    limits = write_limits(
        {
            "safety": {
                "kill_switch_path": str(run_dir / "KILL"),
                "heartbeat_path": str(run_dir / "heartbeat"),
            }
        }
    )
    return {
        "limits": limits,
        "db": tmp_path / "ledger.db",
        "args": ["--limits", str(limits), "--db", str(tmp_path / "ledger.db")],
    }


def _run(args: list[str], **kwargs: Any) -> Any:
    result = runner.invoke(app, args, **kwargs)
    if result.exception and not isinstance(result.exception, SystemExit):
        raise result.exception
    return result


def _out(result: Any) -> str:
    """Combined stdout and stderr.

    Failures are written to stderr on purpose, so a human piping `tb status`
    into a file still sees errors, and scripts can separate the two. Tests read
    both.
    """
    stderr = ""
    try:
        stderr = result.stderr or ""
    except ValueError:  # pragma: no cover - depends on the click version
        stderr = ""
    return (result.stdout or "") + stderr


class TestInit:
    def test_init_creates_the_ledger_and_pins_the_limits(self, env: dict[str, Any]) -> None:
        result = _run(["init", *env["args"]])
        assert result.exit_code == 0
        assert "ledger created" in _out(result)
        assert "hard limits pinned" in _out(result)
        assert env["db"].exists()

    def test_init_is_idempotent(self, env: dict[str, Any]) -> None:
        _run(["init", *env["args"]])
        result = _run(["init", *env["args"]])
        assert result.exit_code == 0
        assert "already exists" in _out(result)

    def test_init_with_missing_limits_exits_two(self, tmp_path: Path) -> None:
        result = _run(["init", "--limits", str(tmp_path / "nope.yaml")])
        assert result.exit_code == 2
        assert "will not start without them" in _out(result)


class TestStatus:
    def test_status_on_a_fresh_ledger_explains_why_it_will_not_trade(
        self, env: dict[str, Any]
    ) -> None:
        _run(["init", *env["args"]])
        result = _run(["status", *env["args"]])
        assert result.exit_code == 0
        assert "Blocked because" in _out(result)
        assert "run state is boot" in _out(result)

    def test_status_without_a_ledger_exits_two(self, env: dict[str, Any]) -> None:
        result = _run(["status", *env["args"]])
        assert result.exit_code == 2
        assert "tb init" in _out(result)


class TestHaltAndResume:
    def test_halt_engages_the_switch_and_records_a_halt(self, env: dict[str, Any]) -> None:
        _run(["init", *env["args"]])
        result = _run(["halt", "-r", "spread looks wrong", *env["args"]])
        assert result.exit_code == 0
        assert "kill switch engaged" in _out(result)
        assert "halt recorded" in _out(result)

        status = _run(["status", *env["args"]])
        assert "kill switch is engaged" in _out(status)
        assert "spread looks wrong" in _out(status)

    def test_resume_clears_a_manual_halt(self, env: dict[str, Any]) -> None:
        _run(["init", *env["args"]])
        _run(["halt", "-r", "manual stop", *env["args"]])
        result = _run(["resume", "-r", "all good", *env["args"]])
        assert result.exit_code == 0
        assert "kill switch released" in _out(result)
        # Resuming lands in reconciling, never straight into trading.
        assert "reconciling" in _out(result)

    def test_resume_refuses_to_clear_a_safety_halt_without_force(self, env: dict[str, Any]) -> None:
        """A breaker fired for a reason.

        `tb resume` will clear a halt someone raised by hand, but not one a
        safety check raised, unless told explicitly.
        """
        from tb.config.loader import load_hard_limits
        from tb.ledger.store import Ledger
        from tb.ops.state import StateMachine

        _run(["init", *env["args"]])
        pinned = load_hard_limits(env["limits"])
        with Ledger(env["db"]) as ledger:
            StateMachine(ledger, pinned, run_id="run_x").raise_halt(
                "daily_loss", "day is -2.4% against a 2.0% limit"
            )

        result = _run(["resume", "-r", "looks fine to me", *env["args"]])
        assert result.exit_code == 1
        assert "raised by safety checks" in _out(result)
        assert "--force" in _out(result)

        forced = _run(["resume", "-r", "reviewed", "--force", *env["args"]])
        assert forced.exit_code == 0


class TestLedgerCommands:
    def test_verify_passes_and_warns_about_the_missing_anchor(self, env: dict[str, Any]) -> None:
        _run(["init", *env["args"]])
        result = _run(["ledger", "verify", *env["args"]])
        assert result.exit_code == 0
        assert "chain intact" in _out(result)
        # Honest about what an unanchored chain cannot detect.
        assert "rewrite-and-re-sign" in _out(result)

    def test_anchor_then_verify_reports_agreement(
        self, env: dict[str, Any], tmp_path: Path
    ) -> None:
        _run(["init", *env["args"]])
        anchored = _run(["ledger", "anchor", "--path", str(tmp_path / "heads.jsonl"), *env["args"]])
        assert anchored.exit_code == 0
        assert "anchored seq=" in _out(anchored)

        result = _run(["ledger", "verify", *env["args"]])
        assert "1 anchors agree" in _out(result)

    def test_verify_exits_one_on_a_tampered_chain(
        self, env: dict[str, Any], tamper: Callable[..., None]
    ) -> None:
        _run(["init", *env["args"]])
        tamper(env["db"], "UPDATE event_log SET actor = 'human' WHERE seq = 1")
        result = _run(["ledger", "verify", *env["args"]])
        assert result.exit_code == 1
        assert "CHAIN VERIFICATION FAILED" in _out(result)

    def test_tail_shows_events(self, env: dict[str, Any]) -> None:
        _run(["init", *env["args"]])
        result = _run(["ledger", "tail", "-n", "5", *env["args"]])
        assert result.exit_code == 0
        assert "ledger.genesis" in _out(result)

    def test_show_displays_one_event(self, env: dict[str, Any]) -> None:
        _run(["init", *env["args"]])
        result = _run(["ledger", "show", "1", *env["args"]])
        assert result.exit_code == 0
        assert "payload" in _out(result)

    def test_show_of_a_missing_seq_exits_one(self, env: dict[str, Any]) -> None:
        _run(["init", *env["args"]])
        result = _run(["ledger", "show", "9999", *env["args"]])
        assert result.exit_code == 1


class TestDoctor:
    def test_doctor_passes_on_a_clean_setup(
        self, env: dict[str, Any], monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.delenv(LIVE_KEY_VAR, raising=False)
        monkeypatch.setenv(DEMO_KEY_VAR, "demo-key-abcd1234")
        _run(["init", *env["args"]])
        result = _run(["doctor", *env["args"]])
        assert result.exit_code == 0
        assert "No problems" in _out(result)

    def test_doctor_fails_when_both_keys_are_present(
        self, env: dict[str, Any], monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv(DEMO_KEY_VAR, "demo-key-abcd1234")
        monkeypatch.setenv(LIVE_KEY_VAR, "live-key-wxyz5678")
        _run(["init", *env["args"]])
        result = _run(["doctor", *env["args"]])
        assert result.exit_code == 1
        assert "ambiguous" in _out(result)

    def test_doctor_never_prints_a_secret(
        self, env: dict[str, Any], monkeypatch: pytest.MonkeyPatch
    ) -> None:
        secret = "t212demo_SUPERSECRETVALUE99"
        monkeypatch.delenv(LIVE_KEY_VAR, raising=False)
        monkeypatch.setenv(DEMO_KEY_VAR, secret)
        _run(["init", *env["args"]])
        result = _run(["doctor", *env["args"]])
        assert secret not in _out(result)
        assert "SUPERSECRET" not in _out(result)

    def test_doctor_reports_an_undeterminable_kill_switch_as_a_problem(
        self,
        tmp_path: Path,
        write_limits: Callable[[dict[str, Any]], Path],
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        monkeypatch.delenv(LIVE_KEY_VAR, raising=False)
        monkeypatch.setenv(DEMO_KEY_VAR, "demo-key-abcd1234")
        limits = write_limits(
            {
                "safety": {
                    "kill_switch_path": str(tmp_path / "never-mounted" / "KILL"),
                    "heartbeat_path": str(tmp_path / "never-mounted" / "heartbeat"),
                }
            }
        )
        db = tmp_path / "ledger.db"
        args = ["--limits", str(limits), "--db", str(db)]
        _run(["init", *args])
        # `init` creates the directory, so remove it to simulate a lost mount.
        (tmp_path / "never-mounted").rmdir()

        result = _run(["doctor", *args])
        assert result.exit_code == 1
        assert "undeterminable" in _out(result)


class TestConfigCommand:
    def test_config_show_prints_the_hashes(self, env: dict[str, Any]) -> None:
        result = _run(["config", "--limits", str(env["limits"])])
        assert result.exit_code == 0
        assert "content hash" in _out(result)
        assert "canonical hash" in _out(result)

    def test_config_show_json_is_machine_readable(self, env: dict[str, Any]) -> None:
        import json

        result = _run(["config", "--json", "--limits", str(env["limits"])])
        assert result.exit_code == 0
        payload = json.loads(result.stdout)
        assert payload["values"]["capital"]["absolute_ceiling_ccy"] == "500.00"


def test_version() -> None:
    result = _run(["version"])
    assert result.exit_code == 0
    assert "tradingbot" in _out(result)
