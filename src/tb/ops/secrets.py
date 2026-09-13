"""Credential hygiene checks.

Trading 212 issues a separate API key per environment, and this system reads
them from **deliberately different variable names** rather than from one key
plus a mode flag. A flag is one typo, one stale shell, one copied systemd unit
away from sending demo-intended orders to a real account. With two names, the
base URL is *derived* from which key is present, so there is no flag to get
wrong.

The rules enforced here:

* A live key present in a research or backtest process is a configuration
  error, not a convenience. Those processes have no reason to hold trading
  credentials and every reason not to.
* Both keys present at once is ambiguous, and ambiguity about which account
  gets the orders is not a state worth resolving by precedence rules.
* A key that looks like a placeholder is reported, because a 400 from the
  broker at 09:30 is a worse place to learn this.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from enum import StrEnum

DEMO_KEY_VAR = "T212_DEMO_API_KEY"
LIVE_KEY_VAR = "T212_LIVE_API_KEY"

DEMO_BASE_URL = "https://demo.trading212.com/api/v0"
LIVE_BASE_URL = "https://live.trading212.com/api/v0"

# Names that must never appear in a process that only researches or backtests.
TRADING_KEY_VARS = (DEMO_KEY_VAR, LIVE_KEY_VAR)

_PLACEHOLDERS = frozenset(
    {"", "changeme", "your-key-here", "xxx", "todo", "none", "null", "<your_api_key>"}
)


class BrokerEnvironment(StrEnum):
    DEMO = "demo"
    LIVE = "live"
    NONE = "none"
    AMBIGUOUS = "ambiguous"

    @property
    def base_url(self) -> str | None:
        if self is BrokerEnvironment.DEMO:
            return DEMO_BASE_URL
        if self is BrokerEnvironment.LIVE:
            return LIVE_BASE_URL
        return None

    @property
    def is_real_money(self) -> bool:
        return self is BrokerEnvironment.LIVE


class Severity(StrEnum):
    OK = "ok"
    WARN = "warn"
    FAIL = "fail"


@dataclass(frozen=True, slots=True)
class SecretsReport:
    environment: BrokerEnvironment
    severity: Severity
    findings: tuple[str, ...]
    demo_key_present: bool
    live_key_present: bool

    @property
    def usable(self) -> bool:
        return self.severity is not Severity.FAIL and self.environment in (
            BrokerEnvironment.DEMO,
            BrokerEnvironment.LIVE,
        )


def _looks_like_placeholder(value: str) -> bool:
    return value.strip().lower() in _PLACEHOLDERS


def _present(var: str) -> bool:
    value = os.environ.get(var)
    return value is not None and not _looks_like_placeholder(value)


def inspect_secrets(*, require_no_live_key: bool = False) -> SecretsReport:
    """Assess the credentials visible to this process.

    `require_no_live_key` is passed by research and backtest entry points: for
    them, seeing a live key at all is a failure.
    """
    findings: list[str] = []
    severity = Severity.OK

    demo_raw = os.environ.get(DEMO_KEY_VAR)
    live_raw = os.environ.get(LIVE_KEY_VAR)
    demo = _present(DEMO_KEY_VAR)
    live = _present(LIVE_KEY_VAR)

    for var, raw in ((DEMO_KEY_VAR, demo_raw), (LIVE_KEY_VAR, live_raw)):
        if raw is not None and _looks_like_placeholder(raw):
            findings.append(f"{var} is set to a placeholder value; treating it as unset")
            severity = Severity.WARN

    if demo and live:
        findings.append(
            f"both {DEMO_KEY_VAR} and {LIVE_KEY_VAR} are set. Which account receives "
            "orders must not depend on a precedence rule — unset one."
        )
        return SecretsReport(
            environment=BrokerEnvironment.AMBIGUOUS,
            severity=Severity.FAIL,
            findings=tuple(findings),
            demo_key_present=demo,
            live_key_present=live,
        )

    if require_no_live_key and live:
        findings.append(
            f"{LIVE_KEY_VAR} is present in a process that does not trade. Remove it "
            "from this environment: research and backtest paths hold no trading "
            "credentials by design."
        )
        return SecretsReport(
            environment=BrokerEnvironment.LIVE,
            severity=Severity.FAIL,
            findings=tuple(findings),
            demo_key_present=demo,
            live_key_present=live,
        )

    if live:
        findings.append(
            f"{LIVE_KEY_VAR} is set: orders from this process reach a REAL MONEY account."
        )
        return SecretsReport(
            environment=BrokerEnvironment.LIVE,
            severity=max(severity, Severity.WARN, key=_severity_rank),
            findings=tuple(findings),
            demo_key_present=demo,
            live_key_present=live,
        )

    if demo:
        return SecretsReport(
            environment=BrokerEnvironment.DEMO,
            severity=severity,
            findings=tuple(findings),
            demo_key_present=demo,
            live_key_present=live,
        )

    findings.append(
        f"no broker credentials found. Set {DEMO_KEY_VAR} (generate it with the "
        "Trading 212 app switched to Practice mode first, or you will get a live key)."
    )
    return SecretsReport(
        environment=BrokerEnvironment.NONE,
        severity=Severity.WARN,
        findings=tuple(findings),
        demo_key_present=demo,
        live_key_present=live,
    )


def _severity_rank(severity: Severity) -> int:
    return {Severity.OK: 0, Severity.WARN: 1, Severity.FAIL: 2}[severity]


def redact(value: str | None, *, keep: int = 4) -> str:
    """Render a secret safe to print or log.

    Every path that could put a credential into a log line, a ledger payload, or
    a model prompt goes through here.
    """
    if not value:
        return "(unset)"
    stripped = value.strip()
    if len(stripped) <= keep:
        return "*" * len(stripped)
    return f"{'*' * (len(stripped) - keep)}{stripped[-keep:]}"
