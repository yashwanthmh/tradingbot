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
from tb.data.providers.alpaca import KEY_ID_VAR as ALPACA_KEY_ID_VAR
from tb.data.providers.alpaca import SECRET_VAR as ALPACA_SECRET_VAR
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
        data_secret = "alpaca_ALSOSUPERSECRET77"
        monkeypatch.delenv(LIVE_KEY_VAR, raising=False)
        monkeypatch.setenv(DEMO_KEY_VAR, secret)
        monkeypatch.setenv(ALPACA_KEY_ID_VAR, "alpaca-key-id-1234")
        monkeypatch.setenv(ALPACA_SECRET_VAR, data_secret)
        _run(["init", *env["args"]])
        result = _run(["doctor", *env["args"]])
        assert secret not in _out(result)
        assert data_secret not in _out(result)
        assert "SUPERSECRET" not in _out(result)

    def test_doctor_warns_but_still_passes_without_market_data_keys(
        self, env: dict[str, Any], monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Yahoo-only is a supported way to run, so this must not be fatal."""
        monkeypatch.delenv(LIVE_KEY_VAR, raising=False)
        monkeypatch.setenv(DEMO_KEY_VAR, "demo-key-abcd1234")
        for var in (ALPACA_KEY_ID_VAR, ALPACA_SECRET_VAR):
            monkeypatch.delenv(var, raising=False)
        _run(["init", *env["args"]])
        result = _run(["doctor", *env["args"]])
        assert result.exit_code == 0
        assert "yahoo only" in _out(result)

    def test_doctor_names_a_key_set_under_a_variable_nothing_reads(
        self, env: dict[str, Any], monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The whole point of the panel: name the typo, not the provider."""
        monkeypatch.delenv(LIVE_KEY_VAR, raising=False)
        monkeypatch.setenv(DEMO_KEY_VAR, "demo-key-abcd1234")
        for var in (ALPACA_KEY_ID_VAR, ALPACA_SECRET_VAR):
            monkeypatch.delenv(var, raising=False)
        monkeypatch.setenv("ALPACA_API_KEY_ID", "wrong-name-key")
        _run(["init", *env["args"]])
        result = _run(["doctor", *env["args"]])
        assert result.exit_code == 0
        output = _out(result)
        assert "ALPACA_API_KEY_ID" in output
        assert ALPACA_KEY_ID_VAR in output
        assert "wrong-name-key" not in output

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


# --------------------------------------------------------------------------
# M1: broker, symbols, reconcile
# --------------------------------------------------------------------------


class TestBrokerCommands:
    def test_limits_reports_the_table_and_the_arithmetic(self, env: dict[str, Any]) -> None:
        """The universe ceiling is derived, not chosen.

        Protective stops are limit-class orders at one per two seconds, so this
        command is where that constraint becomes visible instead of buried.
        """
        _run(["init", *env["args"]])
        result = _run(["broker", "limits", *env["args"]])
        assert result.exit_code == 0
        out = _out(result)
        assert "/equity/orders/market" in out or "orders/market" in out
        assert "not probed" in out
        assert "governor budget for protection" in out

    def test_probe_without_credentials_explains_the_practice_mode_trap(
        self, env: dict[str, Any], monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.delenv(DEMO_KEY_VAR, raising=False)
        monkeypatch.delenv(LIVE_KEY_VAR, raising=False)
        _run(["init", *env["args"]])
        result = _run(["broker", "probe", *env["args"]])
        assert result.exit_code == 2
        assert "Practice mode" in _out(result)

    def test_probe_refuses_a_live_key_by_default(
        self, env: dict[str, Any], monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """There is no reason to characterise an API with real money."""
        monkeypatch.delenv(DEMO_KEY_VAR, raising=False)
        monkeypatch.setenv(LIVE_KEY_VAR, "live-key-abcd1234")
        _run(["init", *env["args"]])
        result = _run(["broker", "probe", *env["args"]])
        assert result.exit_code == 2
        assert "real-money" in _out(result)

    def test_drift_is_empty_on_a_fresh_ledger(self, env: dict[str, Any]) -> None:
        _run(["init", *env["args"]])
        result = _run(["broker", "drift", *env["args"]])
        assert result.exit_code == 0
        assert "no unparsed responses" in _out(result)

    def test_replay_of_an_unknown_message_exits_one(self, env: dict[str, Any]) -> None:
        _run(["init", *env["args"]])
        result = _run(["broker", "replay", "msg_nope", *env["args"]])
        assert result.exit_code == 1


class TestSymbolCommands:
    def _seed(self, env: dict[str, Any]) -> None:
        """Cache a couple of instruments so the audit needs no credentials."""
        from decimal import Decimal

        from tb.broker.port import Instrument
        from tb.broker.t212.probe import cache_instruments
        from tb.config.loader import load_hard_limits
        from tb.ledger.store import Ledger

        pinned = load_hard_limits(env["limits"])
        with Ledger(env["db"], config_hash=pinned.config_hash) as ledger:
            cache_instruments(
                ledger,
                [
                    Instrument(
                        ticker="AAPL_US_EQ",
                        currency_code="USD",
                        instrument_type="STOCK",
                        min_trade_quantity=Decimal("0.1"),
                    ),
                    Instrument(ticker="VODl_EQ", currency_code="GBX", instrument_type="STOCK"),
                ],
            )

    def test_audit_reports_nothing_enterable_before_verification(self, env: dict[str, Any]) -> None:
        """The correct default for a join this dangerous, and it names the remedy.

        Derivation proposes a symbol; nothing has confirmed it. `enterable` is
        the number that matters — while it is zero the first position cannot be
        opened — so the message says which command produces the bars the
        cross-provider tier needs, rather than leaving a dead end.
        """
        _run(["init", *env["args"]])
        self._seed(env)
        result = _run(["symbols", "audit", *env["args"]])
        assert result.exit_code == 0
        out = _out(result)
        assert "cross-verified" in out
        assert "enterable" in out
        assert "nothing is enterable yet" in out
        assert "tb data backfill" in out
        assert "Exits are never gated" in out

    def test_audit_maps_a_us_listing_and_refuses_a_non_us_one_for_alpaca(
        self, env: dict[str, Any]
    ) -> None:
        _run(["init", *env["args"]])
        self._seed(env)
        _run(["symbols", "audit", "--provider", "alpaca", *env["args"]])

        shown = _run(["symbols", "show", "AAPL_US_EQ", *env["args"]])
        assert "AAPL" in _out(shown)

        unmapped = _run(["symbols", "show", "VODl_EQ", *env["args"]])
        assert unmapped.exit_code == 0
        assert "none" in _out(unmapped)

    def test_audit_maps_a_london_listing_for_yfinance(self, env: dict[str, Any]) -> None:
        _run(["init", *env["args"]])
        self._seed(env)
        _run(["symbols", "audit", "--provider", "yfinance", *env["args"]])
        shown = _run(["symbols", "show", "VODl_EQ", *env["args"]])
        assert "VOD.L" in _out(shown)

    def test_show_reports_the_asymmetric_gate(self, env: dict[str, Any]) -> None:
        _run(["init", *env["args"]])
        self._seed(env)
        _run(["symbols", "audit", *env["args"]])
        result = _run(["symbols", "show", "AAPL_US_EQ", *env["args"]])
        out = _out(result)
        assert "may open a position" in out
        assert "may close a position" in out
        assert "never gated" in out

    def test_show_of_an_unknown_ticker_exits_one(self, env: dict[str, Any]) -> None:
        _run(["init", *env["args"]])
        result = _run(["symbols", "show", "NOSUCH_EQ", *env["args"]])
        assert result.exit_code == 1
        assert "tb symbols audit" in _out(result)


class TestReconcileCommand:
    def test_reconcile_without_credentials_exits_two(
        self, env: dict[str, Any], monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.delenv(DEMO_KEY_VAR, raising=False)
        monkeypatch.delenv(LIVE_KEY_VAR, raising=False)
        _run(["init", *env["args"]])
        result = _run(["reconcile", *env["args"]])
        assert result.exit_code == 2
