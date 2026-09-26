"""The dashboard: read-only views of the ledger, and a button that only stops.

What has to hold: each view says what the ledger says, scoped to the account
asked about; nothing but the kill switch writes, and nothing here releases it;
without a token only loopback is served and the button is off; with one,
nothing under /api is served without it and it never comes back out; a Host
the dashboard was not started for is refused, which is what stops DNS
rebinding; every response carries the policy that keeps ledger text inert;
and `tb dashboard` will not listen beyond loopback without a token.
"""

from __future__ import annotations

import ast
import json
import re
import sys
from collections.abc import Iterator
from datetime import datetime, timedelta
from decimal import Decimal
from pathlib import Path
from typing import Any

import pytest
from fastapi.routing import APIRoute
from fastapi.testclient import TestClient
from starlette.routing import Mount
from starlette.staticfiles import StaticFiles
from typer.testing import CliRunner

import tb.api
from tb.api.app import SECURITY_HEADERS, STATIC_DIR, create_app
from tb.api.deps import (
    LOOPBACK_HOSTS,
    TOKEN_ENV,
    DashboardError,
    DashboardSettings,
    check_token,
)
from tb.api.routes import control, read
from tb.cli import app as cli
from tb.core.errors import LedgerError
from tb.ledger.events import EventType, HaltRaisedPayload, TradeClosedPayload
from tb.ledger.store import Ledger
from tb.ops.journal import ensure_pinned, render_page, write_page
from tb.ops.killswitch import KillSwitchState, read_kill_switch
from tb.portfolio.pnl import EquityCurve
from tb.registry.lineage import SpecRegistry
from tb.registry.models import AuthorKind
from tb.strategy.dsl.schema import StrategySpec
from tests.session_helpers import MON, TUE, PinnedClock, append, pin_clock, run, utc

NOW = utc(MON, 21)
TOKEN = "dashboard-token-for-tests-0123456789"


@pytest.fixture
def clock(monkeypatch: pytest.MonkeyPatch) -> PinnedClock:
    return pin_clock(monkeypatch)


@pytest.fixture
def book(env: dict[str, Any], clock: PinnedClock) -> Iterator[Ledger]:
    with Ledger(env["db"]) as opened:
        opened.initialise(created_by="test")
        yield opened


def _settings(
    env: dict[str, Any],
    *,
    token: str | None = None,
    hosts: tuple[str, ...] = LOOPBACK_HOSTS,
    now: datetime = NOW,
) -> DashboardSettings:
    return DashboardSettings(
        db=env["db"],
        pinned=env["pinned"],
        journal_dir=env["run_dir"] / "journal",
        token=token,
        allowed_hosts=hosts,
        clock=lambda: now,
    )


def _client(
    settings: DashboardSettings, *, peer: str = "127.0.0.1", host: str = "localhost"
) -> TestClient:
    return TestClient(create_app(settings), base_url=f"http://{host}", client=(peer, 50000))


def _auth(token: str = TOKEN) -> dict[str, str]:
    return {"Authorization": f"Bearer {token}"}


def _spec() -> StrategySpec:
    return StrategySpec.model_validate(
        {
            "name": "cross",
            "entry": {
                "kind": "compare",
                "op": "gt",
                "left": {"kind": "feature", "name": "sma", "lookback": 20},
                "right": {"kind": "feature", "name": "sma", "lookback": 100},
            },
            "exit": {
                "kind": "compare",
                "op": "lt",
                "left": {"kind": "feature", "name": "sma", "lookback": 20},
                "right": {"kind": "feature", "name": "sma", "lookback": 100},
            },
            "expected_edge_bps": "250",
            "min_holding_minutes": 1440,
        }
    )


def _closed(
    ledger: Ledger,
    clock: PinnedClock,
    at: datetime,
    *,
    fill: str,
    run_id: str,
    pnl: str | None,
    strategy: tuple[str, int] | None,
) -> None:
    append(
        ledger,
        clock,
        at,
        EventType.TRADE_CLOSED,
        TradeClosedPayload(
            closing_fill_id=fill,
            run_id=run_id,
            t212_ticker="AAPL_US_EQ",
            quantity=Decimal("0.5"),
            admissible=pnl is not None,
            charged=pnl is not None,
            strategy_id=None if strategy is None else strategy[0],
            strategy_version=None if strategy is None else strategy[1],
            pnl_ccy=None if pnl is None else Decimal(pnl),
            closed_at=at.isoformat(),
        ),
    )


def _trading_day(ledger: Ledger, clock: PinnedClock) -> tuple[str, int]:
    """A clean demo Monday with marks and four closed trades, and a paper run
    on Tuesday whose trade must never be counted as the demo account's."""
    registry = SpecRegistry(ledger, per_lineage_budget_ccy=Decimal("100"), run_id="run_a")
    registered = registry.register(_spec(), author_kind=AuthorKind.HUMAN, at=utc(MON, 12))
    strategy = (registered.strategy_id, registered.version)
    run(ledger, clock, run_id="run_a", mode="demo", day=MON)
    _closed(ledger, clock, utc(MON, 19), fill="f1", run_id="run_a", pnl="1.25", strategy=strategy)
    _closed(
        ledger, clock, utc(MON, 19, 1), fill="f2", run_id="run_a", pnl="-0.50", strategy=strategy
    )
    _closed(ledger, clock, utc(MON, 19, 2), fill="f3", run_id="run_a", pnl=None, strategy=strategy)
    _closed(ledger, clock, utc(MON, 19, 3), fill="f4", run_id="run_a", pnl="0.10", strategy=None)
    curve = EquityCurve(ledger, run_id="run_a")
    curve.mark(
        equity_ccy=Decimal("1000.00"),
        at=utc(MON, 13, 31),
        currency="GBP",
        deployed_ccy=Decimal("0"),
    )
    curve.mark(
        equity_ccy=Decimal("1001.25"),
        at=utc(MON, 19, 59),
        currency="GBP",
        deployed_ccy=Decimal("15.00"),
    )
    run(ledger, clock, run_id="run_p", mode="paper", day=TUE)
    _closed(ledger, clock, utc(TUE, 19), fill="p1", run_id="run_p", pnl="9.99", strategy=strategy)
    return strategy


# --------------------------------------------------------------------------
# The views
# --------------------------------------------------------------------------


def test_status_on_a_fresh_ledger(env: dict[str, Any], book: Ledger) -> None:
    body = _client(_settings(env)).get("/api/status").json()
    assert body["run_state"]["state"] == "boot"
    assert body["kill_switch"] == {
        "state": "clear",
        "may_trade": True,
        "detail": body["kill_switch"]["detail"],
    }
    assert body["heartbeat"]["stale"] is True, "no heartbeat file: stale, never 'not started'"
    assert body["running"] is None and body["open_halts"] == []
    assert body["live"]["armed"] is False and body["live"]["enabled_in_limits"] is False
    assert body["ledger"]["head_seq"] == 1
    assert body["limits"] == {
        "config_hash": env["pinned"].config_hash,
        "currency": env["pinned"].limits.currency,
        "drift": None,
    }
    assert body["control"] == {"kill_switch_button": False}


def test_status_names_limits_that_changed_on_disk(env: dict[str, Any], book: Ledger) -> None:
    limits: Path = env["limits"]
    limits.write_text(limits.read_text(encoding="utf-8") + "\n# edited\n", encoding="utf-8")
    drift = _client(_settings(env)).get("/api/status").json()["limits"]["drift"]
    assert drift is not None and "changed while running" in drift


def test_the_equity_view_is_the_accounts_curve(
    env: dict[str, Any], book: Ledger, clock: PinnedClock
) -> None:
    _trading_day(book, clock)
    client = _client(_settings(env))
    body = client.get("/api/equity", params={"mode": "demo"}).json()
    assert body["run_id"] == "run_a" and body["currency"] == "GBP"
    assert [p["equity"] for p in body["points"]] == ["1000.00", "1001.25"]
    assert body["reading"]["equity"] == "1001.25" and body["reading"]["peak"] == "1001.25"
    assert body["reading"]["day_pnl_pct"] == pytest.approx(0.125)
    assert body["reading"]["drawdown_from_peak_pct"] == pytest.approx(0.0)

    paper = client.get("/api/equity", params={"mode": "paper"}).json()
    assert paper["run_id"] == "run_p" and paper["points"] == [], "a paper run is its own account"
    live = client.get("/api/equity", params={"mode": "live"}).json()
    assert live == {"mode": "live", "run_id": None, "currency": None, "points": [], "reading": None}
    assert client.get("/api/equity", params={"mode": "real"}).status_code == 422


def test_a_long_curve_is_sampled_and_keeps_its_last_mark(
    env: dict[str, Any], book: Ledger, clock: PinnedClock, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr("tb.api.views.MAX_POINTS", 10)
    run(book, clock, run_id="run_a", mode="demo", day=MON)
    curve = EquityCurve(book, run_id="run_a")
    for n in range(26):
        curve.mark(
            equity_ccy=Decimal(1000 + n),
            at=utc(MON, 14) + timedelta(minutes=n),
            currency="GBP",
            deployed_ccy=Decimal("0"),
        )
    points = _client(_settings(env)).get("/api/equity").json()["points"]
    assert len(points) == 10, "every third mark, and the last one exactly"
    assert points[0]["equity"] == "1000" and points[-1]["equity"] == "1025"


def test_the_risk_view_reads_each_budget_as_its_breaker_does(
    env: dict[str, Any], book: Ledger, clock: PinnedClock
) -> None:
    _trading_day(book, clock)
    limits = env["pinned"].limits
    budgets = {
        (b["name"], b["unit"]): b
        for b in _client(_settings(env)).get("/api/risk").json()["budgets"]
    }
    day = budgets[("day's loss", "%")]
    assert day["observed"] == pytest.approx(0.125) and day["used_pct"] == 0.0, "a gain uses none"
    assert day["limit"] == float(limits.loss.daily_halt_pct)
    assert budgets[("drawdown from peak", "%")]["used_pct"] == 0.0
    share = budgets[("deployed", "% of equity")]
    assert share["observed"] == pytest.approx(15 / 1001.25 * 100)
    assert share["used_pct"] == pytest.approx(
        share["observed"] / float(limits.capital.max_deployed_pct) * 100
    )
    absolute = budgets[("deployed", limits.currency)]
    assert absolute["observed"] == "15.00" and absolute["limit"] == str(
        limits.capital.absolute_ceiling_ccy
    )
    orders = budgets[("orders today", "orders")]
    assert orders["observed"] == 0 and orders["limit"] == limits.execution.max_orders_per_day


def test_attribution_is_per_account_and_keeps_what_it_cannot_place(
    env: dict[str, Any], book: Ledger, clock: PinnedClock
) -> None:
    strategy = _trading_day(book, clock)
    client = _client(_settings(env))
    demo = client.get("/api/strategies", params={"mode": "demo"}).json()
    placed, unplaced = demo
    assert (placed["strategy_id"], placed["version"]) == strategy
    assert placed["status"] == "candidate" and placed["lineage_budget"] == "100"
    assert placed["account"] == {
        "trades": 3,
        "measured": 2,
        "wins": 1,
        "realised": "0.75",
        "last_closed": utc(MON, 19, 2).isoformat(),
    }, "the unmeasured trade is counted, never summed"
    assert unplaced["strategy_id"] is None and unplaced["status"] is None
    assert unplaced["account"]["realised"] == "0.10"

    [paper] = client.get("/api/strategies", params={"mode": "paper"}).json()
    assert paper["account"]["realised"] == "9.99" and paper["account"]["trades"] == 1


def test_the_sessions_view_is_the_record_the_gate_counts(
    env: dict[str, Any], book: Ledger, clock: PinnedClock
) -> None:
    _trading_day(book, clock)
    body = _client(_settings(env)).get("/api/sessions", params={"mode": "demo"}).json()
    assert (
        body["streak"] == 1
        and body["required"] == env["pinned"].limits.live.min_clean_demo_sessions
    )
    [monday] = body["sessions"]
    assert monday["date"] == MON.isoformat() and monday["verdict"] == "clean"
    paper = _client(_settings(env)).get("/api/sessions", params={"mode": "paper"}).json()
    assert paper["required"] is None


def test_the_event_tail_polls_from_a_cursor_and_cuts_what_is_too_long(
    env: dict[str, Any], book: Ledger, clock: PinnedClock
) -> None:
    run(book, clock, run_id="run_a", mode="demo", day=MON, stop=(14, 0))
    head = book.head()
    assert head is not None
    client = _client(_settings(env))
    tail = client.get("/api/events", params={"after": 0, "limit": 5}).json()
    assert [e["seq"] for e in tail] == list(range(head.seq - 4, head.seq + 1))
    assert tail[-1]["type"] == "run.ended" and tail[-1]["payload"]["run_id"] == "run_a"
    assert client.get("/api/events", params={"after": head.seq}).json() == []

    append(
        book,
        clock,
        utc(MON, 14, 5),
        EventType.HALT_RAISED,
        HaltRaisedPayload(halt_id="halt_long", trigger="audit", detail="x" * 5000),
    )
    [long] = client.get("/api/events", params={"after": head.seq}).json()
    assert long["truncated"] is True and long["payload"] is None
    assert client.get("/api/events", params={"limit": 501}).status_code == 422


def test_the_journal_list_and_one_page_as_text(
    env: dict[str, Any], book: Ledger, clock: PinnedClock
) -> None:
    _trading_day(book, clock)
    pinned = env["pinned"]
    ensure_pinned(book, pinned)
    directory = env["run_dir"] / "journal"
    directory.mkdir()
    page = render_page(
        book, session=MON, limits=pinned.limits, limits_hash=pinned.config_hash, as_of=NOW
    )
    write_page(book, page, directory=directory)
    (env["run_dir"] / "secret.md").write_text("not for the browser", encoding="utf-8")

    client = _client(_settings(env))
    [listed] = client.get("/api/journal").json()
    assert listed["date"] == MON.isoformat() and listed["status"] == page.header.status
    shown = client.get(f"/api/journal/{MON}")
    assert shown.status_code == 200 and shown.text == page.text
    assert shown.headers["content-type"].startswith("text/plain")
    assert client.get(f"/api/journal/{TUE}").status_code == 404
    assert client.get("/api/journal/not-a-date").status_code == 422
    escaped = client.get("/api/journal/..%2F..%2Fsecret")
    assert escaped.status_code in (404, 405, 422) and "not for the browser" not in escaped.text


def test_the_views_refuse_a_ledger_they_cannot_read(env: dict[str, Any]) -> None:
    missing = _client(_settings(env)).get("/api/status")
    assert missing.status_code == 503 and "tb init" in missing.json()["detail"]


# --------------------------------------------------------------------------
# Who may ask
# --------------------------------------------------------------------------


def test_without_a_token_only_loopback_is_served(env: dict[str, Any], book: Ledger) -> None:
    settings = _settings(env)
    assert _client(settings, peer="::1").get("/api/status").status_code == 200
    refused = _client(settings, peer="10.0.0.5").get("/api/status")
    assert refused.status_code == 403 and "loopback" in refused.json()["detail"]


def test_with_a_token_nothing_under_api_is_served_without_it(
    env: dict[str, Any], book: Ledger
) -> None:
    client = _client(_settings(env, token=TOKEN), peer="10.0.0.5")
    bare = client.get("/api/status")
    assert bare.status_code == 401 and bare.headers["www-authenticate"] == "Bearer"
    assert client.get("/api/status", headers=_auth("wrong-" + TOKEN)).status_code == 401
    assert client.get("/api/status", headers={"Authorization": TOKEN}).status_code == 401
    assert client.get("/api/status", headers=_auth()).status_code == 200
    # The page itself carries no data and must load to ask for the token.
    assert client.get("/").status_code == 200


def test_the_token_never_comes_back_out(
    env: dict[str, Any], book: Ledger, clock: PinnedClock
) -> None:
    _trading_day(book, clock)
    settings = _settings(env, token=TOKEN)
    assert TOKEN not in repr(settings)
    client = _client(settings)
    responses = [
        client.get(path, headers=_auth())
        for path in (
            "/api/status",
            "/api/equity",
            "/api/risk",
            "/api/strategies",
            "/api/sessions",
            "/api/events",
            "/api/journal",
            "/",
        )
    ]
    responses.append(client.get("/api/status", headers=_auth("wrong-" + TOKEN)))
    responses.append(client.post("/api/killswitch", headers=_auth(), json={"reason": "drill"}))
    for response in responses:
        assert TOKEN not in response.text
        assert all(TOKEN not in value for value in response.headers.values())


def test_a_host_it_was_not_started_for_is_refused(env: dict[str, Any], book: Ledger) -> None:
    """DNS rebinding: a hostile page that points its own name at 127.0.0.1
    arrives with its own name in the Host header, and gets nothing."""
    for path in ("/", "/api/status"):
        response = _client(_settings(env), host="attacker.example").get(path)
        assert response.status_code == 400
        assert (
            response.headers["content-security-policy"]
            == SECURITY_HEADERS["Content-Security-Policy"]
        )
    named = _settings(env, token=TOKEN, hosts=(*LOOPBACK_HOSTS, "tb.lan"))
    assert _client(named, host="tb.lan").get("/api/status", headers=_auth()).status_code == 200


def test_a_weak_token_is_refused_without_being_repeated() -> None:
    for weak in ("Qz7-weak", "has spaces in it, twenty-four+", "tab\there-and-long-enough-0123"):
        with pytest.raises(DashboardError) as caught:
            check_token(weak)
        assert weak not in str(caught.value)
    assert check_token(TOKEN) == TOKEN


# --------------------------------------------------------------------------
# The kill switch, and nothing else
# --------------------------------------------------------------------------


def _kill_events(ledger: Ledger) -> list[Any]:
    return list(
        ledger.conn.execute(
            "SELECT event_type, actor, payload_json FROM event_log"
            " WHERE event_type IN (?, ?) ORDER BY seq",
            (EventType.KILLSWITCH_ENGAGED.value, EventType.HALT_RAISED.value),
        ).fetchall()
    )


def _switch(env: dict[str, Any]) -> KillSwitchState:
    return read_kill_switch(env["pinned"].limits.safety.kill_switch_path).state


def test_the_button_is_off_without_a_token(env: dict[str, Any], book: Ledger) -> None:
    refused = _client(_settings(env)).post("/api/killswitch", json={"reason": "stop"})
    assert refused.status_code == 403 and "tb halt" in refused.json()["detail"]
    assert _switch(env) is KillSwitchState.CLEAR and _kill_events(book) == []


def test_the_button_engages_and_records_a_halt_once(env: dict[str, Any], book: Ledger) -> None:
    client = _client(_settings(env, token=TOKEN))
    engaged = client.post("/api/killswitch", headers=_auth(), json={"reason": "  odd fills  "})
    assert engaged.status_code == 200, engaged.text
    body = engaged.json()
    assert body["engaged"] and body["recorded"] and not body["already_engaged"]
    assert _switch(env) is KillSwitchState.ENGAGED

    switched, halted = _kill_events(book)
    assert switched["actor"] == halted["actor"] == "human"
    assert json.loads(switched["payload_json"])["engaged_by"] == "dashboard (127.0.0.1)"
    halt = json.loads(halted["payload_json"])
    assert halt["trigger"] == "manual" and halt["detail"] == "odd fills"
    assert halt["halt_id"] == body["halt_id"]

    status = client.get("/api/status", headers=_auth()).json()
    assert status["kill_switch"]["state"] == "engaged" and status["run_state"]["state"] == "halted"
    assert [h["halt_id"] for h in status["open_halts"]] == [body["halt_id"]]

    again = client.post("/api/killswitch", headers=_auth(), json={"reason": "double click"})
    assert again.status_code == 200 and again.json()["already_engaged"] is True
    assert len(_kill_events(book)) == 2, "a double click is one halt, not two"


@pytest.mark.parametrize(
    ("headers", "body", "status"),
    [
        ({}, '{"reason": "x"}', 401),
        ({"Origin": "http://attacker.example"}, '{"reason": "x"}', 403),
        ({"Sec-Fetch-Site": "cross-site"}, '{"reason": "x"}', 403),
        ({"Content-Type": "text/plain"}, '{"reason": "x"}', 415),
        ({}, '{"reason": "   "}', 422),
        ({}, '{"reason": "x", "and": "more"}', 422),
        ({}, "not json", 422),
        ({}, json.dumps({"reason": "x" * 5000}), 413),
    ],
)
def test_the_button_refuses_what_it_should(
    env: dict[str, Any], book: Ledger, headers: dict[str, str], body: str, status: int
) -> None:
    sent = {"Content-Type": "application/json", **(_auth() if status != 401 else {}), **headers}
    response = _client(_settings(env, token=TOKEN)).post(
        "/api/killswitch", headers=sent, content=body
    )
    assert response.status_code == status, response.text
    assert _switch(env) is KillSwitchState.CLEAR and _kill_events(book) == []


def test_a_same_origin_request_from_the_page_is_accepted(env: dict[str, Any], book: Ledger) -> None:
    response = _client(_settings(env, token=TOKEN)).post(
        "/api/killswitch",
        headers={**_auth(), "Origin": "http://localhost", "Sec-Fetch-Site": "same-origin"},
        json={"reason": "from the page"},
    )
    assert response.status_code == 200 and _switch(env) is KillSwitchState.ENGAGED


def test_the_switch_is_thrown_even_when_the_ledger_cannot_record_it(
    env: dict[str, Any], book: Ledger, monkeypatch: pytest.MonkeyPatch
) -> None:
    def busy(*args: Any, **kwargs: Any) -> str:
        raise LedgerError("database is locked")

    monkeypatch.setattr("tb.api.routes.control.StateMachine.raise_halt", busy)
    response = _client(_settings(env, token=TOKEN)).post(
        "/api/killswitch", headers=_auth(), json={"reason": "stop"}
    )
    assert response.status_code == 200
    body = response.json()
    assert body["engaged"] is True and body["recorded"] is False and "tb halt" in body["problem"]
    assert _switch(env) is KillSwitchState.ENGAGED


def test_a_switch_that_cannot_be_thrown_says_run_tb_halt(
    env: dict[str, Any], book: Ledger, write_limits: Any
) -> None:
    blocker = env["run_dir"] / "not-a-directory"
    blocker.write_text("", encoding="utf-8")
    from tb.config.loader import load_hard_limits

    pinned = load_hard_limits(
        write_limits(
            {
                "safety": {
                    "kill_switch_path": str(blocker / "KILL"),
                    "heartbeat_path": str(env["run_dir"] / "heartbeat"),
                }
            }
        )
    )
    settings = DashboardSettings(
        db=env["db"], pinned=pinned, journal_dir=env["run_dir"], token=TOKEN, clock=lambda: NOW
    )
    response = _client(settings).post("/api/killswitch", headers=_auth(), json={"reason": "stop"})
    assert response.status_code == 500 and "tb halt" in response.json()["detail"]


def test_there_is_no_way_to_release_it_here(env: dict[str, Any], book: Ledger) -> None:
    client = _client(_settings(env, token=TOKEN))
    client.post("/api/killswitch", headers=_auth(), json={"reason": "stop"})
    for method in ("GET", "PUT", "PATCH", "DELETE"):
        response = client.request(method, "/api/killswitch", headers=_auth())
        assert response.status_code in (404, 405), method
    for path in ("/api/killswitch/release", "/api/resume", "/api/halts/clear"):
        assert client.post(path, headers=_auth(), json={}).status_code in (404, 405)
    assert _switch(env) is KillSwitchState.ENGAGED


def test_the_kill_switch_is_the_only_route_that_is_not_a_read(env: dict[str, Any]) -> None:
    """The structural form of 'the dashboard cannot trade': one write route,
    and it is the one that stops."""
    app = create_app(_settings(env))
    operations = {
        (path, method.upper())
        for path, methods in app.openapi()["paths"].items()
        for method in methods
    }
    assert {op for op in operations if op[1] != "GET"} == {("/api/killswitch", "POST")}
    # Nothing hidden from that listing, and nothing added beside the routers.
    for router in (read.router, control.router):
        assert all(isinstance(r, APIRoute) and r.include_in_schema for r in router.routes)
    assert not [r for r in app.routes if isinstance(r, APIRoute)]
    [mount] = [r for r in app.routes if isinstance(r, Mount)]
    assert isinstance(mount.app, StaticFiles) and len(app.routes) == 3


FORBIDDEN_CALLS = {
    "release_kill_switch",
    "clear_halt",
    "transition_to",
    "place_order",
    "cancel_order",
    "arm_live",
    "disarm",
    "record_run_start",
}


def test_nothing_in_the_api_package_can_trade_release_or_arm() -> None:
    package = Path(tb.api.__file__).parent
    found: list[str] = []
    for source in package.rglob("*.py"):
        for node in ast.walk(ast.parse(source.read_text(encoding="utf-8"))):
            name = None
            if isinstance(node, ast.Attribute):
                name = node.attr
            elif isinstance(node, ast.Name):
                name = node.id
            elif isinstance(node, ast.alias):
                name = node.name.rsplit(".", 1)[-1]
            if name in FORBIDDEN_CALLS:
                found.append(f"{source.name}: {name}")
    assert found == []


# --------------------------------------------------------------------------
# The page
# --------------------------------------------------------------------------


def test_every_response_carries_the_policy(env: dict[str, Any], book: Ledger) -> None:
    client = _client(_settings(env))
    for path in ("/", "/app.js", "/style.css", "/api/status", "/api/nope"):
        response = client.get(path)
        for name, value in SECURITY_HEADERS.items():
            assert response.headers.get(name) == value, (path, name)
    # With nosniff, a script served as anything but JavaScript does not run.
    assert client.get("/app.js").headers["content-type"].startswith("text/javascript")
    assert client.get("/style.css").headers["content-type"].startswith("text/css")


def test_the_page_runs_only_its_own_script_and_never_parses_ledger_text_as_markup() -> None:
    page = (STATIC_DIR / "index.html").read_text(encoding="utf-8")
    script = (STATIC_DIR / "app.js").read_text(encoding="utf-8")
    assert re.findall(r"<script\b[^>]*>", page) == ['<script src="app.js" defer>']
    assert not re.search(r"\son[a-z]+\s*=", page), "no inline handlers"
    assert not re.search(r"(src|href)=\"(https?:)?//", page), "nothing from elsewhere"
    for sink in (
        "innerHTML",
        "outerHTML",
        "insertAdjacentHTML",
        "document.write",
        "eval(",
        "Function(",
    ):
        assert sink not in script, sink
    assert "textContent" in script and "createTextNode" in script
    # The token prompt and the page viewer are shown and hidden by attribute,
    # which a class setting `display` would silently override.
    assert "[hidden]" in (STATIC_DIR / "style.css").read_text(encoding="utf-8")


def test_the_page_and_the_docs_are_what_is_served(env: dict[str, Any], book: Ledger) -> None:
    client = _client(_settings(env))
    assert "Kill switch" in client.get("/").text
    for path in ("/docs", "/redoc", "/openapi.json"):
        assert client.get(path).status_code == 404, path


# --------------------------------------------------------------------------
# tb dashboard
# --------------------------------------------------------------------------


def _args(env: dict[str, Any], *extra: str) -> list[str]:
    return ["dashboard", "--limits", str(env["limits"]), "--db", str(env["db"]), *extra]


@pytest.fixture
def served(monkeypatch: pytest.MonkeyPatch) -> list[tuple[Any, dict[str, Any]]]:
    calls: list[tuple[Any, dict[str, Any]]] = []
    monkeypatch.setattr("uvicorn.run", lambda app, **kw: calls.append((app, kw)))
    monkeypatch.delenv(TOKEN_ENV, raising=False)
    return calls


def test_tb_dashboard_serves_loopback_with_the_button_off(
    env: dict[str, Any], book: Ledger, served: list[tuple[Any, dict[str, Any]]]
) -> None:
    result = CliRunner().invoke(cli, _args(env))
    assert result.exit_code == 0, result.output
    [(served_app, options)] = served
    assert options["host"] == "127.0.0.1" and options["port"] == 8765
    assert options["proxy_headers"] is False
    settings = served_app.state.settings
    assert settings.token is None and settings.allowed_hosts == LOOPBACK_HOSTS
    assert "button off" in result.output


def test_tb_dashboard_refuses_to_listen_beyond_loopback_without_a_token(
    env: dict[str, Any],
    book: Ledger,
    served: list[tuple[Any, dict[str, Any]]],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runner = CliRunner()
    bare = runner.invoke(cli, _args(env, "--host", "192.168.1.10"))
    assert bare.exit_code == 2 and TOKEN_ENV in bare.output

    monkeypatch.setenv(TOKEN_ENV, "too-short")
    weak = runner.invoke(cli, _args(env))
    assert weak.exit_code == 2 and "too-short" not in weak.output

    monkeypatch.setenv(TOKEN_ENV, TOKEN)
    everywhere = runner.invoke(cli, _args(env, "--host", "0.0.0.0"))
    assert everywhere.exit_code == 2 and "--allow-host" in everywhere.output
    assert served == []

    lan = runner.invoke(cli, _args(env, "--host", "192.168.1.10"))
    assert lan.exit_code == 0, lan.output
    assert TOKEN not in lan.output and "plain HTTP" in lan.output
    [(served_app, options)] = served
    assert options["host"] == "192.168.1.10"
    assert "192.168.1.10" in served_app.state.settings.allowed_hosts


def test_tb_dashboard_needs_a_ledger_and_the_api_extra(
    env: dict[str, Any],
    served: list[tuple[Any, dict[str, Any]]],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runner = CliRunner()
    missing = runner.invoke(cli, _args(env))
    assert missing.exit_code == 2 and "tb init" in missing.output
    monkeypatch.setitem(sys.modules, "uvicorn", None)
    absent = runner.invoke(cli, _args(env))
    assert absent.exit_code == 2 and "--extra api" in absent.output
    assert served == []
