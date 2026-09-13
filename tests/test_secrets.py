"""Credential hygiene.

The design decision under test: two deliberately different variable names
instead of one key plus a mode flag. With a flag, a stale shell or a copied
systemd unit sends demo-intended orders to a real account. With two names, the
base URL is *derived* from which key is present, so there is no flag to get
wrong — and "both keys present" is an error rather than something resolved by a
precedence rule nobody remembers.
"""

from __future__ import annotations

import pytest

from tb.ops.secrets import (
    DEMO_BASE_URL,
    DEMO_KEY_VAR,
    LIVE_BASE_URL,
    LIVE_KEY_VAR,
    BrokerEnvironment,
    Severity,
    inspect_secrets,
    redact,
)


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch: pytest.MonkeyPatch) -> None:
    """Start from no credentials, so the developer's real keys cannot leak in."""
    monkeypatch.delenv(DEMO_KEY_VAR, raising=False)
    monkeypatch.delenv(LIVE_KEY_VAR, raising=False)


class TestEnvironmentDerivation:
    def test_demo_key_alone_selects_the_demo_endpoint(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv(DEMO_KEY_VAR, "demo-key-1234")
        report = inspect_secrets()
        assert report.environment is BrokerEnvironment.DEMO
        assert report.environment.base_url == DEMO_BASE_URL
        assert not report.environment.is_real_money
        assert report.severity is Severity.OK
        assert report.usable

    def test_live_key_alone_selects_the_live_endpoint_and_warns(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv(LIVE_KEY_VAR, "live-key-5678")
        report = inspect_secrets()
        assert report.environment is BrokerEnvironment.LIVE
        assert report.environment.base_url == LIVE_BASE_URL
        assert report.environment.is_real_money
        assert report.severity is Severity.WARN
        assert any("REAL MONEY" in f for f in report.findings)

    def test_both_keys_present_is_a_failure_not_a_precedence_decision(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Which account receives orders must never depend on a tiebreak rule."""
        monkeypatch.setenv(DEMO_KEY_VAR, "demo-key-1234")
        monkeypatch.setenv(LIVE_KEY_VAR, "live-key-5678")
        report = inspect_secrets()
        assert report.environment is BrokerEnvironment.AMBIGUOUS
        assert report.severity is Severity.FAIL
        assert not report.usable
        assert report.environment.base_url is None

    def test_no_keys_warns_and_explains_the_practice_mode_trap(self) -> None:
        """The mistake everyone makes once.

        Trading 212 mints the key for whichever mode the app is in, so
        generating a key without switching to Practice first silently gives you
        a live one.
        """
        report = inspect_secrets()
        assert report.environment is BrokerEnvironment.NONE
        assert report.severity is Severity.WARN
        assert not report.usable
        assert any("Practice mode first" in f for f in report.findings)


class TestPlaceholders:
    @pytest.mark.parametrize("value", ["", "  ", "changeme", "TODO", "your-key-here", "xxx"])
    def test_a_placeholder_is_treated_as_unset(
        self, monkeypatch: pytest.MonkeyPatch, value: str
    ) -> None:
        """Better to say so now than to get a 400 from the broker at 09:30."""
        monkeypatch.setenv(DEMO_KEY_VAR, value)
        report = inspect_secrets()
        assert not report.demo_key_present
        assert report.environment is BrokerEnvironment.NONE

    def test_a_placeholder_live_key_does_not_make_it_ambiguous(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv(DEMO_KEY_VAR, "demo-key-1234")
        monkeypatch.setenv(LIVE_KEY_VAR, "changeme")
        report = inspect_secrets()
        assert report.environment is BrokerEnvironment.DEMO
        assert any("placeholder" in f for f in report.findings)


class TestResearchIsolation:
    def test_a_live_key_in_a_research_process_is_a_failure(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Research and backtest paths hold no trading credentials by design.

        There is no reason for a process that only reads history to be able to
        send an order, and every reason for it not to be.
        """
        monkeypatch.setenv(LIVE_KEY_VAR, "live-key-5678")
        report = inspect_secrets(require_no_live_key=True)
        assert report.severity is Severity.FAIL
        assert not report.usable
        assert any("does not trade" in f for f in report.findings)

    def test_a_demo_key_in_a_research_process_is_fine(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv(DEMO_KEY_VAR, "demo-key-1234")
        report = inspect_secrets(require_no_live_key=True)
        assert report.severity is Severity.OK


class TestRedaction:
    def test_redact_keeps_only_a_tail(self) -> None:
        assert redact("abcdefghijklmnop") == "************mnop"

    def test_redact_handles_unset(self) -> None:
        assert redact(None) == "(unset)"
        assert redact("") == "(unset)"

    def test_a_short_secret_is_fully_masked(self) -> None:
        """Keeping 4 of a 4-character secret would reveal all of it."""
        assert redact("abcd") == "****"
        assert redact("ab") == "**"

    def test_redaction_never_contains_the_leading_secret(self) -> None:
        secret = "t212_live_SUPERSECRETVALUE"
        masked = redact(secret)
        assert "SUPERSECRET" not in masked
        assert secret not in masked
