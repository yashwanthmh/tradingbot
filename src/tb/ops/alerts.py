"""Alerts: the ledger's warnings, delivered to a person, once, out of process.

The ledger records everything and tells no one. An operator cannot watch a
terminal all session, so `tb alerts --follow` runs beside the loop and the
watchdog, reads the ledger from a saved cursor, and turns the events a person
must act on into alerts: halts, the kill switch, watchdog trips, a stop that
failed to land, drift, a failed drill, real money armed or disarmed, and a
session judged unclean.

**Out of process and read-only.** A notifier that shared the trader's process
would go quiet exactly when the trader wedged, and one that could write the
record could hide something in it. This one reads the ledger and nothing else;
its only state is a cursor file beside the kill switch.

**At least once.** The cursor advances only after every sink has taken the
batch. A webhook that is down means the batch is offered again next pass, and
a duplicate on the console is the price of never losing a halt.

**Deduplicated.** Each alert has a key — the halt's trigger, the ticker whose
stop failed — and a key seen within the window is counted, not re-sent. A
loop restarted into the same fault ten times is one alert with a count.

**No secrets.** Payloads never carry keys, by the ledger's design. The webhook
URL, which usually embeds one, is read from `TB_ALERT_WEBHOOK_URL` only, never
taken as an argument, and never appears in an error: a transport failure is
reported without it.

**From now, not from the beginning.** A first run starts at the current head
and treats every session already judged as already told: history is not
news, and a first run that replayed a year of it would teach the operator to
ignore the channel.
"""

from __future__ import annotations

import json
import os
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import StrEnum
from pathlib import Path
from typing import Any, Protocol

from tb.config.hard_limits import LiveLimits
from tb.core.clock import from_iso, now_utc, to_iso
from tb.core.errors import TbError
from tb.core.http import Transport
from tb.ledger.events import EventType
from tb.ledger.store import Ledger
from tb.ops.sessions import TRADING_MODES, SessionVerdict, read_all_sessions

WEBHOOK_ENV = "TB_ALERT_WEBHOOK_URL"
STATE_FILE = "alerts.json"
DEFAULT_WINDOW = timedelta(hours=1)
# How long a sent key is remembered. Past the dedup window, only session keys
# matter — a session is judged once — and a month covers any restart.
_REMEMBER = timedelta(days=30)
# How often sessions are re-judged. Judging reads the whole ledger, and a
# session is judged fifteen minutes after its close anyway, so a follower
# polling every thirty seconds need not redo it every pass.
SESSION_CHECK_EVERY = timedelta(minutes=15)


class AlertError(TbError):
    """An alert could not be delivered, or the alerter could not run."""


class Severity(StrEnum):
    INFO = "info"
    WARNING = "warning"
    CRITICAL = "critical"

    @property
    def rank(self) -> int:
        return {"info": 0, "warning": 1, "critical": 2}[self.value]


@dataclass(frozen=True, slots=True)
class Alert:
    key: str
    severity: Severity
    title: str
    detail: str
    at: datetime
    seq: int | None = None
    repeats: int = 0

    def text(self) -> str:
        again = f" (and {self.repeats} more like it)" if self.repeats else ""
        where = f" [seq {self.seq}]" if self.seq is not None else ""
        return f"{self.severity.value.upper()}: {self.title}{again} — {self.detail}{where}"

    def as_dict(self) -> dict[str, Any]:
        return {
            "key": self.key,
            "severity": self.severity.value,
            "title": self.title,
            "detail": self.detail,
            "at": to_iso(self.at),
            "seq": self.seq,
            "repeats": self.repeats,
        }


# --------------------------------------------------------------------------
# Which events are news
# --------------------------------------------------------------------------

# Each rule reads a payload and returns (severity, key, title, detail), or
# None when this instance of the event is not news — a stop withdrawn on
# purpose before an exit, a run that ended cleanly.
_Rule = Callable[[dict[str, Any]], tuple[Severity, str, str, str] | None]

_PROTECTION_FAILURES = frozenset({"no_price", "risk_refused", "placement_failed"})


def _run_ended(p: dict[str, Any]) -> tuple[Severity, str, str, str] | None:
    reason = str(p.get("exit_reason", ""))
    if reason == "halted":
        return (
            Severity.CRITICAL,
            f"run_end:{p.get('run_id')}",
            f"run {p.get('run_id')} halted",
            str(p.get("error_detail") or "the loop halted"),
        )
    if reason == "error":
        return (
            Severity.CRITICAL,
            f"run_end:{p.get('run_id')}",
            f"run {p.get('run_id')} stopped on an error",
            f"{p.get('error_type')}: {p.get('error_detail')}",
        )
    return None


def _unprotected(p: dict[str, Any]) -> tuple[Severity, str, str, str] | None:
    cause = p.get("cause")
    detail = str(p.get("detail", ""))
    failed = (
        cause in _PROTECTION_FAILURES
        if cause is not None
        else not detail.startswith("stop withdrawn")
    )
    if not failed:
        return None
    ticker = p.get("t212_ticker")
    return (
        Severity.CRITICAL,
        f"unprotected:{ticker}:{cause}",
        f"{ticker} is held without a stop",
        detail,
    )


def _drill(p: dict[str, Any]) -> tuple[Severity, str, str, str]:
    if p.get("passed"):
        return (
            Severity.INFO,
            f"drill:{p.get('drill_id')}",
            f"{p.get('kind')} drill passed",
            "; ".join(str(o) for o in p.get("observations", [])),
        )
    return (
        Severity.CRITICAL,
        f"drill:{p.get('drill_id')}",
        f"{p.get('kind')} drill failed",
        "; ".join(str(f) for f in p.get("failures", [])),
    )


def _reconciled(p: dict[str, Any]) -> tuple[Severity, str, str, str] | None:
    if p.get("verdict") != "halted":
        return None
    return (
        Severity.CRITICAL,
        f"reconcile:{p.get('recon_id')}",
        "reconciliation found blocking discrepancies",
        f"{p.get('n_unknown_intents')} unknown intent(s), {p.get('n_orphan_orders')} orphan "
        f"order(s), {p.get('n_position_mismatches')} position mismatch(es), "
        f"{p.get('n_unprotected_positions')} unprotected position(s)",
    )


RULES: dict[EventType, _Rule] = {
    EventType.RUN_ENDED: _run_ended,
    EventType.HALT_RAISED: lambda p: (
        Severity.CRITICAL,
        f"halt:{p.get('trigger')}",
        f"halt raised [{p.get('trigger')}]",
        str(p.get("detail")),
    ),
    EventType.KILLSWITCH_ENGAGED: lambda p: (
        Severity.CRITICAL,
        "killswitch",
        "kill switch engaged",
        f"by {p.get('engaged_by') or 'unknown'}: {p.get('detail')}",
    ),
    EventType.WATCHDOG_TRIPPED: lambda p: (
        Severity.CRITICAL,
        "watchdog",
        "the watchdog found the trader silent",
        f"{p.get('action_taken')}: {p.get('detail')}",
    ),
    EventType.WATCHDOG_UNREACHABLE: lambda p: (
        Severity.CRITICAL,
        f"selfcheck:{p.get('direction')}",
        "the trader halted itself on its self-check",
        f"{p.get('direction')}: {p.get('detail')}",
    ),
    EventType.POSITION_UNPROTECTED: _unprotected,
    EventType.CONFIG_DRIFT_DETECTED: lambda p: (
        Severity.CRITICAL,
        "config_drift",
        "the limits changed under a running process",
        str(p.get("detail")),
    ),
    EventType.BROKER_SCHEMA_DRIFT: lambda p: (
        Severity.CRITICAL,
        f"schema_drift:{p.get('endpoint')}",
        "the broker's response changed shape",
        f"{p.get('endpoint')} {p.get('model')}: {p.get('error_detail')}",
    ),
    EventType.RECONCILE_COMPLETED: _reconciled,
    EventType.DRILL_COMPLETED: _drill,
    EventType.LIVE_ARMED: lambda p: (
        Severity.CRITICAL,
        f"live:{p.get('arming_id')}",
        "real-money trading armed",
        f"by {p.get('armed_by')} for {', '.join(str(s) for s in p.get('strategies', []))} "
        f"until {p.get('expires_at')}",
    ),
    EventType.LIVE_DISARMED: lambda p: (
        Severity.CRITICAL,
        f"live_off:{p.get('arming_id')}",
        "real-money trading disarmed",
        f"by {p.get('disarmed_by')}: {p.get('reason')}",
    ),
    EventType.ORDER_REJECTED: lambda p: (
        Severity.WARNING,
        f"rejected:{p.get('t212_ticker')}",
        f"the broker rejected an order on {p.get('t212_ticker')}",
        str(p.get("detail")) + (f" ({p['broker_message']})" if p.get("broker_message") else ""),
    ),
    EventType.POSITION_ORPHANED: lambda p: (
        Severity.WARNING,
        f"orphan:{p.get('t212_ticker')}",
        f"{p.get('t212_ticker')} is held by no funded strategy",
        f"{p.get('reason')}; {p.get('action_taken')}",
    ),
    EventType.INSTANCE_LOCK_REFUSED: lambda p: (
        Severity.WARNING,
        "second_instance",
        "a second trading instance was refused",
        f"run {p.get('run_id')} was started while {p.get('held_by_run_id')} held the lease",
    ),
    EventType.LINEAGE_BUDGET_EXHAUSTED: lambda p: (
        Severity.WARNING,
        f"budget:{p.get('lineage_id')}",
        f"lineage {p.get('lineage_id')} spent its loss budget",
        f"{p.get('consumed_ccy')} of {p.get('budget_ccy')}",
    ),
    EventType.HOLDOUT_VIOLATION_ATTEMPTED: lambda p: (
        Severity.WARNING,
        f"holdout:{p.get('strategy_id')}",
        "something asked for data past the sealed holdout",
        f"{p.get('caller')}: {p.get('detail')}",
    ),
    EventType.DATA_STALENESS_BREACH: lambda p: (
        Severity.WARNING,
        f"stale:{p.get('instrument_uid')}",
        f"no fresh bar for {p.get('instrument_uid')}",
        f"{p.get('resolution')}: {p.get('action')}",
    ),
    EventType.STRATEGY_RETIRED: lambda p: (
        Severity.INFO,
        f"retired:{p.get('strategy_id')}",
        f"{p.get('strategy_id')} v{p.get('version')} retired",
        str(p.get("reason")),
    ),
    EventType.BACKUP_RESTORE_VERIFIED: lambda p: (
        Severity.INFO,
        f"restore:{p.get('backup_id')}",
        f"backup {p.get('backup_id')} restored on {p.get('restored_host')}",
        "same host: the live gate does not count it"
        if p.get("same_host")
        else "on another machine",
    ),
}


def alerts_from_events(rows: Iterable[Any]) -> list[Alert]:
    found: list[Alert] = []
    for row in rows:
        rule = RULES.get(EventType(row["event_type"]))
        if rule is None:
            continue
        made = rule(json.loads(row["payload_json"]))
        if made is None:
            continue
        severity, key, title, detail = made
        found.append(
            Alert(
                key=key,
                severity=severity,
                title=title,
                detail=detail,
                at=from_iso(str(row["ts_utc"])),
                seq=int(row["seq"]),
            )
        )
    return found


def session_alerts(ledger: Ledger, *, limits: LiveLimits, now: datetime) -> list[Alert]:
    """Every judged session that was not clean, as a candidate alert."""
    found: list[Alert] = []
    records = read_all_sessions(ledger, limits=limits, now=now)
    for mode in TRADING_MODES:
        for session in records[mode].sessions:
            if session.verdict not in (SessionVerdict.FAULTED, SessionVerdict.INCOMPLETE):
                continue
            found.append(
                Alert(
                    key=f"session:{mode}:{session.session_date}",
                    severity=Severity.CRITICAL
                    if session.verdict is SessionVerdict.FAULTED
                    else Severity.WARNING,
                    title=f"the {mode} session of {session.session_date} was "
                    f"{session.verdict.value}",
                    detail=session.summary(),
                    at=session.close_utc,
                )
            )
    return found


# --------------------------------------------------------------------------
# Sinks
# --------------------------------------------------------------------------


class Sink(Protocol):
    @property
    def name(self) -> str: ...

    def send(self, alerts: Sequence[Alert]) -> None:
        """Deliver the batch or raise; a raise leaves the cursor where it was."""
        ...


@dataclass
class ConsoleSink:
    write: Callable[[str], None]
    name: str = "console"

    def send(self, alerts: Sequence[Alert]) -> None:
        for alert in alerts:
            self.write(alert.text())


@dataclass
class FileSink:
    """One JSON object per alert, appended, for anything that tails a file."""

    path: Path
    name: str = "file"

    def send(self, alerts: Sequence[Alert]) -> None:
        if not alerts:
            return
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("a", encoding="utf-8") as handle:
            for alert in alerts:
                handle.write(json.dumps(alert.as_dict(), sort_keys=True) + "\n")
            handle.flush()
            os.fsync(handle.fileno())


@dataclass
class WebhookSink:
    """A POST per alert, `{"text": ...}`, the shape chat webhooks accept.

    The URL is private to this object: it is never logged, printed or put in
    an error, because a webhook URL usually is its own credential.
    """

    url: str = field(repr=False)
    transport: Transport = field(repr=False)
    name: str = "webhook"
    timeout: float = 10.0

    @classmethod
    def from_env(cls, transport: Transport) -> WebhookSink:
        url = os.environ.get(WEBHOOK_ENV, "").strip()
        if not url:
            raise AlertError(
                f"{WEBHOOK_ENV} is not set. The webhook URL is read from the environment "
                "only, never from an argument, because it usually carries its own secret."
            )
        if not url.startswith("https://"):
            raise AlertError(f"{WEBHOOK_ENV} must be an https:// URL")
        return cls(url=url, transport=transport)

    def send(self, alerts: Sequence[Alert]) -> None:
        for alert in alerts:
            try:
                response = self.transport.request(
                    "POST",
                    self.url,
                    headers={"Content-Type": "application/json"},
                    json_body={"text": alert.text(), "alert": alert.as_dict()},
                    timeout=self.timeout,
                )
            except Exception as exc:
                # Deliberately without `from exc`: the transport's error names
                # the URL, and the URL is the secret.
                raise AlertError(
                    f"the webhook could not be reached ({type(exc).__name__})"
                ) from None
            if not response.ok:
                raise AlertError(f"the webhook answered {response.status_code}")


# --------------------------------------------------------------------------
# One pass
# --------------------------------------------------------------------------


@dataclass
class AlertState:
    """Where the alerter has read to, and what it has already said."""

    cursor: int = 0
    sent: dict[str, str] = field(default_factory=dict)
    suppressed: dict[str, int] = field(default_factory=dict)
    sessions_checked_at: str | None = None

    @classmethod
    def load(cls, path: Path) -> AlertState | None:
        try:
            body = json.loads(path.read_text(encoding="utf-8"))
        except FileNotFoundError:
            return None
        except (OSError, ValueError) as exc:
            raise AlertError(f"the alert state at {path} is unreadable: {exc}") from exc
        return cls(
            cursor=int(body.get("cursor", 0)),
            sent={str(k): str(v) for k, v in body.get("sent", {}).items()},
            suppressed={str(k): int(v) for k, v in body.get("suppressed", {}).items()},
            sessions_checked_at=body.get("sessions_checked_at"),
        )

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_name(f".{path.name}.tmp")
        temporary.write_text(
            json.dumps(
                {
                    "cursor": self.cursor,
                    "sent": self.sent,
                    "suppressed": self.suppressed,
                    "sessions_checked_at": self.sessions_checked_at,
                },
                sort_keys=True,
            ),
            encoding="utf-8",
        )
        os.replace(temporary, path)


@dataclass(frozen=True, slots=True)
class PassResult:
    delivered: tuple[Alert, ...]
    suppressed: int
    cursor: int
    started_fresh: bool


def run_pass(
    ledger: Ledger,
    *,
    state_path: Path,
    sinks: Sequence[Sink],
    limits: LiveLimits,
    min_severity: Severity = Severity.WARNING,
    window: timedelta = DEFAULT_WINDOW,
    now: datetime | None = None,
) -> PassResult:
    """Read what is new, decide what is news, deliver it, and move the cursor."""
    moment = now or now_utc()
    state = AlertState.load(state_path)
    head = ledger.head()
    top = 0 if head is None else head.seq
    if state is None:
        # History is not news: start at the head, with every session already
        # judged counted as told.
        state = AlertState(cursor=top, sessions_checked_at=to_iso(moment))
        for alert in session_alerts(ledger, limits=limits, now=moment):
            state.sent[alert.key] = to_iso(moment)
        state.save(state_path)
        return PassResult(delivered=(), suppressed=0, cursor=top, started_fresh=True)

    rows = ledger.iter_events(start_seq=state.cursor + 1, end_seq=top, event_types=RULES.keys())
    candidates = alerts_from_events(rows)
    due = (
        state.sessions_checked_at is None
        or moment - from_iso(state.sessions_checked_at) >= SESSION_CHECK_EVERY
    )
    if due:
        candidates += [
            a for a in session_alerts(ledger, limits=limits, now=moment) if a.key not in state.sent
        ]

    fresh: dict[str, Alert] = {}
    suppressed = 0
    for alert in candidates:
        if alert.severity.rank < min_severity.rank:
            continue
        last = state.sent.get(alert.key)
        if last is not None and moment - from_iso(last) < window:
            state.suppressed[alert.key] = state.suppressed.get(alert.key, 0) + 1
            suppressed += 1
            continue
        if alert.key in fresh:
            earlier = fresh[alert.key]
            fresh[alert.key] = Alert(
                key=earlier.key,
                severity=earlier.severity,
                title=earlier.title,
                detail=earlier.detail,
                at=earlier.at,
                seq=earlier.seq,
                repeats=earlier.repeats + 1,
            )
            continue
        fresh[alert.key] = alert
    batch = tuple(
        Alert(
            key=a.key,
            severity=a.severity,
            title=a.title,
            detail=a.detail,
            at=a.at,
            seq=a.seq,
            repeats=a.repeats + state.suppressed.pop(a.key, 0),
        )
        for a in sorted(fresh.values(), key=lambda a: (a.at, a.key))
    )

    for sink in sinks:
        # A sink that raises leaves the cursor unmoved: the batch is offered
        # again next pass, to every sink. At least once.
        sink.send(batch)

    stamp = to_iso(moment)
    for alert in batch:
        state.sent[alert.key] = stamp
    state.sent = {key: at for key, at in state.sent.items() if moment - from_iso(at) <= _REMEMBER}
    state.cursor = top
    if due:
        state.sessions_checked_at = stamp
    state.save(state_path)
    return PassResult(delivered=batch, suppressed=suppressed, cursor=top, started_fresh=False)
