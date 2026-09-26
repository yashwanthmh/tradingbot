"""Alerts: what a person must act on, delivered once, out of process.

What has to hold: a first run starts from now; events that need a person are
delivered and the rest are not (a stop withdrawn before an exit is not news);
repeats within the window are counted rather than re-sent; a sink that fails
leaves the cursor where it was, so nothing is lost; a judged session that was
not clean is alerted exactly once; and the webhook's URL — usually its own
credential — never appears in an error or in the output.
"""

from __future__ import annotations

from collections.abc import Iterator, Sequence
from datetime import datetime
from decimal import Decimal
from pathlib import Path
from typing import Any

import pytest
from typer.testing import CliRunner

from tb.cli import app
from tb.config.loader import load_hard_limits
from tb.core.errors import TransportError
from tb.core.http import HttpResponse, RecordingTransport
from tb.ledger.events import (
    EventType,
    HaltRaisedPayload,
    KillswitchPayload,
    OrderOutcomePayload,
    ProtectionPayload,
    RunEndedPayload,
    StrategyRetiredPayload,
)
from tb.ledger.store import Ledger
from tb.ops.alerts import (
    WEBHOOK_ENV,
    Alert,
    AlertError,
    ConsoleSink,
    FileSink,
    Severity,
    WebhookSink,
    run_pass,
)
from tests.conftest import REFERENCE_LIMITS
from tests.session_helpers import MON, TUE, PinnedClock, append, pin_clock, run, utc

LIVE = load_hard_limits(REFERENCE_LIMITS).limits.live
URL = "https://hooks.example/services/T000/B000/secret-token-in-the-url"


@pytest.fixture
def clock(monkeypatch: pytest.MonkeyPatch) -> PinnedClock:
    return pin_clock(monkeypatch)


@pytest.fixture
def book(ledger_path: Path, clock: PinnedClock) -> Iterator[Ledger]:
    with Ledger(ledger_path) as opened:
        opened.initialise(created_by="test")
        yield opened


class Collect:
    """A sink that keeps what it was given, and can be told to fail."""

    name = "collect"

    def __init__(self) -> None:
        self.batches: list[list[Alert]] = []
        self.fail = False

    def send(self, alerts: Sequence[Alert]) -> None:
        if self.fail:
            raise AlertError("the channel is down")
        self.batches.append(list(alerts))

    @property
    def all(self) -> list[Alert]:
        return [alert for batch in self.batches for alert in batch]


def _pass(ledger: Ledger, state: Path, sink: Collect, *, now: datetime, **kw: Any) -> list[Alert]:
    result = run_pass(ledger, state_path=state, sinks=[sink], limits=LIVE, now=now, **kw)
    return list(result.delivered)


def _halt(ledger: Ledger, clock: PinnedClock, at: datetime, trigger: str = "daily_loss") -> None:
    append(
        ledger,
        clock,
        at,
        EventType.HALT_RAISED,
        HaltRaisedPayload(halt_id=f"halt_{at:%H%M%S}", trigger=trigger, detail="-2.4% on the day"),
    )


def _ready(ledger: Ledger, state: Path, sink: Collect, now: datetime) -> None:
    started = run_pass(ledger, state_path=state, sinks=[sink], limits=LIVE, now=now)
    assert started.started_fresh and not started.delivered


# --------------------------------------------------------------------------
# What is news
# --------------------------------------------------------------------------


def test_a_first_run_starts_from_now(book: Ledger, clock: PinnedClock, tmp_path: Path) -> None:
    """History is not news: a year of it replayed would teach anyone to mute the channel."""
    _halt(book, clock, utc(MON, 14))
    sink = Collect()
    state = tmp_path / "alerts.json"
    _ready(book, state, sink, utc(MON, 15))
    assert _pass(book, state, sink, now=utc(MON, 15, 1)) == []


def test_events_that_need_a_person_are_delivered_and_the_rest_are_not(
    book: Ledger, clock: PinnedClock, tmp_path: Path
) -> None:
    sink = Collect()
    state = tmp_path / "alerts.json"
    _ready(book, state, sink, utc(MON, 13))
    protection: dict[str, Any] = {
        "t212_ticker": "AAPL_US_EQ",
        "run_id": "run_a",
        "quantity": Decimal(1),
    }
    append(
        book,
        clock,
        utc(MON, 14),
        EventType.RUN_ENDED,
        RunEndedPayload(
            run_id="run_a", exit_reason="halted", error_detail="self-check failed: kill switch"
        ),
    )
    append(
        book,
        clock,
        utc(MON, 14, 1),
        EventType.KILLSWITCH_ENGAGED,
        KillswitchPayload(path="KILL", determinable=True, detail="manual", engaged_by="ops"),
    )
    append(
        book,
        clock,
        utc(MON, 14, 2),
        EventType.POSITION_UNPROTECTED,
        ProtectionPayload(
            **protection,
            protected=False,
            cause="placement_failed",
            detail="the stop could not be placed: 503",
        ),
    )
    append(
        book,
        clock,
        utc(MON, 14, 3),
        EventType.POSITION_UNPROTECTED,
        ProtectionPayload(
            **protection, protected=False, cause="withdrawn", detail="stop withdrawn: exit"
        ),
    )
    append(
        book,
        clock,
        utc(MON, 14, 4),
        EventType.ORDER_REJECTED,
        OrderOutcomePayload(
            run_id="run_a",
            t212_ticker="AAPL_US_EQ",
            status="rejected",
            detail="insufficient funds",
        ),
    )
    append(
        book,
        clock,
        utc(MON, 14, 5),
        EventType.STRATEGY_RETIRED,
        StrategyRetiredPayload(strategy_id="s1", version=1, lineage_id="l1", reason="killed"),
    )

    delivered = _pass(book, state, sink, now=utc(MON, 15))

    assert [(a.severity, a.key) for a in delivered] == [
        (Severity.CRITICAL, "run_end:run_a"),
        (Severity.CRITICAL, "killswitch"),
        (Severity.CRITICAL, "unprotected:AAPL_US_EQ:placement_failed"),
        (Severity.WARNING, "rejected:AAPL_US_EQ"),
    ]
    assert "kill switch" in delivered[0].detail
    # Info is below the default floor, and shown when asked for.
    _halt(book, clock, utc(MON, 16))
    everything = _pass(book, state, sink, now=utc(MON, 17), min_severity=Severity.INFO)
    assert [a.key for a in everything] == ["halt:daily_loss"]


def test_a_repeat_within_the_window_is_counted_not_resent(
    book: Ledger, clock: PinnedClock, tmp_path: Path
) -> None:
    """A loop restarted into the same fault ten times is one alert with a count."""
    sink = Collect()
    state = tmp_path / "alerts.json"
    _ready(book, state, sink, utc(MON, 13))
    _halt(book, clock, utc(MON, 14))
    _halt(book, clock, utc(MON, 14, 1))

    [first] = _pass(book, state, sink, now=utc(MON, 14, 5))
    assert first.repeats == 1 and "(and 1 more like it)" in first.text()

    _halt(book, clock, utc(MON, 14, 10))
    assert _pass(book, state, sink, now=utc(MON, 14, 15)) == []

    _halt(book, clock, utc(MON, 16))
    [later] = _pass(book, state, sink, now=utc(MON, 16, 5))
    assert later.repeats == 1, "the one held back in the window is counted on the next"


def test_a_sink_that_fails_loses_nothing(book: Ledger, clock: PinnedClock, tmp_path: Path) -> None:
    sink = Collect()
    state = tmp_path / "alerts.json"
    _ready(book, state, sink, utc(MON, 13))
    _halt(book, clock, utc(MON, 14))

    sink.fail = True
    with pytest.raises(AlertError, match="channel is down"):
        _pass(book, state, sink, now=utc(MON, 14, 5))
    sink.fail = False
    [again] = _pass(book, state, sink, now=utc(MON, 14, 6))
    assert again.key == "halt:daily_loss"


def test_a_session_judged_unclean_is_alerted_once(
    book: Ledger, clock: PinnedClock, tmp_path: Path
) -> None:
    sink = Collect()
    state = tmp_path / "alerts.json"
    _ready(book, state, sink, utc(MON, 12))
    run(book, clock, run_id="run_a", day=MON, stop=(16, 0))  # stopped at noon: incomplete

    assert _pass(book, state, sink, now=utc(MON, 20, 5)) == [], "not judged before the bound"
    [session] = _pass(book, state, sink, now=utc(MON, 20, 30))
    assert session.key == f"session:demo:{MON}"
    assert session.severity is Severity.WARNING and "incomplete" in session.title
    assert _pass(book, state, sink, now=utc(TUE, 9)) == []


# --------------------------------------------------------------------------
# Sinks
# --------------------------------------------------------------------------


def _alert() -> Alert:
    return Alert(
        key="halt:x", severity=Severity.CRITICAL, title="halt raised", detail="d", at=utc(MON, 14)
    )


def test_the_webhook_posts_text_and_never_names_its_url() -> None:
    posted = RecordingTransport(default=HttpResponse(200, {}, "ok", 1.0))
    WebhookSink(url=URL, transport=posted).send([_alert()])
    [call] = posted.calls
    assert call["method"] == "POST" and call["json"]["text"].startswith("CRITICAL: halt raised")

    refused = RecordingTransport(default=HttpResponse(500, {}, "no", 1.0))
    with pytest.raises(AlertError, match="answered 500") as caught:
        WebhookSink(url=URL, transport=refused).send([_alert()])
    assert URL not in str(caught.value)

    class Down:
        def request(self, *args: Any, **kwargs: Any) -> HttpResponse:
            raise TransportError(f"connection refused to {URL}", endpoint=URL)

        def close(self) -> None:
            pass

    with pytest.raises(AlertError) as caught:
        WebhookSink(url=URL, transport=Down()).send([_alert()])
    assert URL not in str(caught.value)
    assert caught.value.__cause__ is None
    assert URL not in repr(WebhookSink(url=URL, transport=Down()))


def test_the_webhook_url_comes_from_the_environment_only(monkeypatch: pytest.MonkeyPatch) -> None:
    transport = RecordingTransport()
    monkeypatch.delenv(WEBHOOK_ENV, raising=False)
    with pytest.raises(AlertError, match="not set"):
        WebhookSink.from_env(transport)
    monkeypatch.setenv(WEBHOOK_ENV, "http://plain.example/hook")
    with pytest.raises(AlertError, match="https"):
        WebhookSink.from_env(transport)
    monkeypatch.setenv(WEBHOOK_ENV, URL)
    assert WebhookSink.from_env(transport).url == URL


def test_the_file_and_console_sinks(tmp_path: Path) -> None:
    target = tmp_path / "alerts.jsonl"
    FileSink(path=target).send([_alert(), _alert()])
    lines = target.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 2 and '"key": "halt:x"' in lines[0]

    written: list[str] = []
    ConsoleSink(write=written.append).send([_alert()])
    assert written == ["CRITICAL: halt raised — d"]


# --------------------------------------------------------------------------
# tb alerts
# --------------------------------------------------------------------------


def test_tb_alerts_starts_from_now_then_delivers_what_is_new(
    env: dict[str, Any], clock: PinnedClock, monkeypatch: pytest.MonkeyPatch
) -> None:
    with Ledger(env["db"]) as ledger:
        ledger.initialise(created_by="test")
        _halt(ledger, clock, utc(MON, 14), trigger="history")
    args = ["alerts", "--limits", str(env["limits"]), "--db", str(env["db"])]
    runner = CliRunner()

    first = runner.invoke(app, args)
    assert first.exit_code == 0, first.output
    assert "history, not news" in first.output

    with Ledger(env["db"]) as ledger:
        _halt(ledger, clock, utc(MON, 15), trigger="daily_loss")
    second = runner.invoke(app, args)
    assert second.exit_code == 0, second.output
    assert "halt raised [daily_loss]" in second.output
    assert "history" not in second.output.split("delivered")[0]
    assert "1 alert(s) delivered" in second.output

    monkeypatch.delenv(WEBHOOK_ENV, raising=False)
    assert runner.invoke(app, [*args, "--sink", "webhook"]).exit_code == 2
    assert runner.invoke(app, [*args, "--sink", "pager"]).exit_code == 2
    assert runner.invoke(app, [*args, "--min-severity", "loud"]).exit_code == 2
