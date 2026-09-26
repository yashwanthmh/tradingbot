"""What the dashboard shows, as plain data read from the ledger.

Each function takes an open ledger and returns something JSON can carry, so
the routes stay thin and every view can be tested without a server. None of
them writes. Money stays a decimal string end to end: a float on the way to a
browser is how 0.1 + 0.2 ends up on a risk panel. Percentages are floats,
because the breakers that read them are.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from datetime import datetime
from decimal import Decimal
from pathlib import Path
from typing import Any

from tb.config.loader import PinnedLimits
from tb.core.clock import from_iso, now_utc
from tb.core.errors import ConfigDriftError
from tb.engine.intents import IntentLog
from tb.ledger.events import EventType
from tb.ledger.store import Ledger
from tb.ops.arming import arming_state
from tb.ops.journal import JournalError, PageHeader, parse_header
from tb.ops.killswitch import read_heartbeat, read_kill_switch
from tb.ops.sessions import TRADING_MODES, read_sessions
from tb.ops.state import StateMachine
from tb.ops.watchdog import LOCK_NAME
from tb.portfolio.pnl import EquityCurve

# The largest event payload the tail carries whole. Anything longer — a model
# prompt on a proposal, a full audit's findings — is cut, and says so.
PAYLOAD_LIMIT = 4000
# Enough points to draw a curve at any width a screen has; a year of
# ten-minute marks is sampled down to this, keeping the last mark exact.
MAX_POINTS = 1000
MAX_MARKS = 50_000


def _decimal(value: Decimal | None) -> str | None:
    return None if value is None else str(value)


def _finite(value: float | None) -> float | None:
    """JSON has no NaN or infinity; a reading that is one is not a reading."""
    return value if value is not None and math.isfinite(value) else None


def _share(observed: float | None, limit: float) -> float | None:
    if observed is None or limit <= 0:
        return None
    return _finite(observed / limit * 100)


def _mode(mode: str) -> str:
    if mode not in TRADING_MODES:
        raise ValueError(f"unknown mode {mode!r}; one of {', '.join(TRADING_MODES)}")
    return mode


def status(ledger: Ledger, pinned: PinnedLimits, *, now: datetime | None = None) -> dict[str, Any]:
    """The one-glance panel: state, switch, heartbeat, halts, the lease, arming."""
    moment = now or now_utc()
    safety = pinned.limits.safety
    machine = StateMachine(ledger, pinned)
    reading = machine.current()
    switch = read_kill_switch(safety.kill_switch_path)
    heartbeat = read_heartbeat(
        safety.heartbeat_path, stale_after_seconds=safety.heartbeat_stale_seconds
    )
    lease = ledger.conn.execute(
        "SELECT l.run_id, l.pid, l.host, l.expires_at, l.released_at, r.mode"
        " FROM instance_locks l LEFT JOIN runs r ON r.run_id = l.run_id"
        " WHERE l.lock_name = ?",
        (LOCK_NAME,),
    ).fetchone()
    holder = None
    if (
        lease is not None
        and lease["released_at"] is None
        and from_iso(str(lease["expires_at"])) > moment
    ):
        holder = {
            "run_id": str(lease["run_id"]),
            "mode": lease["mode"],
            "pid": int(lease["pid"]),
            "host": str(lease["host"]),
            "expires_at": str(lease["expires_at"]),
        }
    try:
        pinned.verify_unchanged()
        drift = None
    except ConfigDriftError as exc:
        drift = str(exc)
    arming = arming_state(ledger, config_hash=pinned.config_hash, now=moment)
    head = ledger.head()
    return {
        "as_of": moment.isoformat(),
        "run_state": {
            "state": reading.state.value,
            "since": reading.since,
            "reason": reading.reason,
        },
        "kill_switch": {
            "state": switch.state.value,
            "may_trade": switch.may_trade,
            "detail": switch.detail,
        },
        "heartbeat": {
            "stale": heartbeat.stale,
            "age_seconds": _finite(heartbeat.age_seconds),
            "detail": heartbeat.detail,
        },
        "open_halts": [
            {
                "halt_id": h.halt_id,
                "trigger": h.trigger,
                "detail": h.detail,
                "raised_at": h.raised_at,
            }
            for h in machine.open_halts()
        ],
        "running": holder,
        "live": {
            "enabled_in_limits": pinned.limits.live.enabled,
            "armed": arming.armed,
            "reason": arming.reason,
            "strategies": list(arming.arming.strategies) if arming.arming else [],
        },
        "ledger": {
            "head_seq": None if head is None else head.seq,
            "head_chain_hash": None if head is None else head.chain_hash,
        },
        "limits": {
            "config_hash": pinned.config_hash,
            "currency": pinned.limits.currency,
            "drift": drift,
        },
    }


def _latest_run(ledger: Ledger, mode: str) -> str | None:
    """The newest run of a mode. The curve it names spans every run of the
    account, except on paper, where each run is an account of its own."""
    row = ledger.conn.execute(
        "SELECT run_id FROM runs WHERE mode = ? ORDER BY started_at DESC LIMIT 1", (mode,)
    ).fetchone()
    return None if row is None else str(row["run_id"])


def equity(ledger: Ledger, *, mode: str, now: datetime | None = None) -> dict[str, Any]:
    """The account's curve for one mode, and the three breaker readings on it."""
    moment = now or now_utc()
    run_id = _latest_run(ledger, _mode(mode))
    if run_id is None:
        return {"mode": mode, "run_id": None, "currency": None, "points": [], "reading": None}
    curve = EquityCurve(ledger, run_id=run_id)
    marks = curve.marks(limit=MAX_MARKS)
    step = max(1, math.ceil(len(marks) / MAX_POINTS))
    sampled = list(marks[::step])
    if marks and sampled[-1] is not marks[-1]:
        sampled.append(marks[-1])
    reading = curve.read(at=moment)
    return {
        "mode": mode,
        "run_id": run_id,
        "currency": marks[-1].currency if marks else None,
        "points": [
            {
                "at": mark.at.isoformat(),
                "equity": str(mark.equity_ccy),
                "deployed": _decimal(mark.deployed_ccy),
            }
            for mark in sampled
        ],
        "reading": {
            "equity": _decimal(reading.equity_ccy),
            "peak": _decimal(reading.peak_equity_ccy),
            "day_pnl_pct": _finite(reading.day_pnl_pct),
            "rolling_pnl_pct": _finite(reading.rolling_pnl_pct),
            "drawdown_from_peak_pct": _finite(reading.drawdown_from_peak_pct),
            "caveats": list(reading.caveats()),
        },
    }


def _budget(
    name: str, observed: float | str | None, limit: float | str, unit: str, used: float | None
) -> dict[str, Any]:
    return {"name": name, "observed": observed, "limit": limit, "unit": unit, "used_pct": used}


def risk(
    ledger: Ledger, pinned: PinnedLimits, *, mode: str, now: datetime | None = None
) -> dict[str, Any]:
    """How much of each loss, exposure and order budget is used, read the way
    the breakers read it — so the panel and the loop cannot disagree."""
    moment = now or now_utc()
    limits = pinned.limits
    run_id = _latest_run(ledger, _mode(mode))
    rows: list[dict[str, Any]] = []
    if run_id is not None:
        curve = EquityCurve(ledger, run_id=run_id)
        reading = curve.read(at=moment)
        latest = curve.marks(limit=1)
        deployed = latest[-1].deployed_ccy if latest else None

        # A loss is negative and its threshold positive; a gain uses none of it.
        for name, observed, limit in (
            ("day's loss", _finite(reading.day_pnl_pct), float(limits.loss.daily_halt_pct)),
            (
                "rolling five-session loss",
                _finite(reading.rolling_pnl_pct),
                float(limits.loss.rolling_5d_halt_pct),
            ),
        ):
            spent = None if observed is None else max(0.0, -observed)
            rows.append(_budget(name, observed, limit, "%", _share(spent, limit)))
        drawdown = _finite(reading.drawdown_from_peak_pct)
        cap = float(limits.loss.max_drawdown_flatten_pct)
        rows.append(_budget("drawdown from peak", drawdown, cap, "%", _share(drawdown, cap)))

        if deployed is not None and reading.equity_ccy:
            share = _finite(float(deployed / reading.equity_ccy * 100))
            ceiling = float(limits.capital.max_deployed_pct)
            rows.append(_budget("deployed", share, ceiling, "% of equity", _share(share, ceiling)))
        if deployed is not None:
            absolute = limits.capital.absolute_ceiling_ccy
            rows.append(
                _budget(
                    "deployed",
                    str(deployed),
                    str(absolute),
                    limits.currency,
                    _share(float(deployed), float(absolute)),
                )
            )
    # Counted as the order-count rule counts them: every intent committed on
    # the UTC date, whichever run committed it.
    today, _ = IntentLog(ledger, run_id="dashboard").counts_today(day=moment)
    cap_orders = limits.execution.max_orders_per_day
    rows.append(
        _budget("orders today", today, cap_orders, "orders", _share(float(today), cap_orders))
    )
    return {"mode": mode, "run_id": run_id, "budgets": rows}


@dataclass
class _Tally:
    trades: int = 0
    measured: int = 0
    wins: int = 0
    realised: Decimal = Decimal(0)
    last_closed: str | None = None


def _attribution(ledger: Ledger, mode: str) -> dict[tuple[str | None, int | None], _Tally]:
    """Closed trades in one account, by the strategy they belong to.

    Read from the events themselves, each tagged with its run, rather than
    from the strategy record — which sums every account, paper included. Only
    a measured result counts toward the P&L: an inferred price is a guess, and
    a guess in the attribution is how a strategy looks better than it is.
    """
    modes = {
        str(row["run_id"]): str(row["mode"])
        for row in ledger.conn.execute("SELECT run_id, mode FROM runs")
    }
    tallies: dict[tuple[str | None, int | None], _Tally] = {}
    for row in ledger.iter_events(event_type=EventType.TRADE_CLOSED):
        payload = json.loads(row["payload_json"])
        if modes.get(str(payload.get("run_id"))) != mode:
            continue
        key = (payload.get("strategy_id"), payload.get("strategy_version"))
        tally = tallies.setdefault(key, _Tally())
        tally.trades += 1
        tally.last_closed = payload.get("closed_at") or str(row["ts_utc"])
        pnl = payload.get("pnl_ccy")
        if payload.get("admissible") and pnl is not None:
            value = Decimal(str(pnl))
            tally.measured += 1
            tally.realised += value
            if value > 0:
                tally.wins += 1
    return tallies


def strategies(ledger: Ledger, *, mode: str) -> list[dict[str, Any]]:
    """Every strategy's standing, with what it has realised in one account.

    A strategy that traded but has no standing — the hand-written one the
    loop was proved with, or trades nobody could attribute — still gets a row:
    an attribution table that drops the trades it cannot place hides exactly
    the ones worth asking about.
    """
    tallies = _attribution(ledger, _mode(mode))
    rows = ledger.conn.execute(
        "SELECT s.strategy_id, s.version, s.status, s.rung, s.lineage_id,"
        " s.realised_pnl_ccy, s.n_realised_trades, b.budget_ccy, b.consumed_ccy"
        " FROM strategy_status s LEFT JOIN lineage_budgets b ON b.lineage_id = s.lineage_id"
        " ORDER BY s.status, s.strategy_id, s.version"
    ).fetchall()
    out: list[dict[str, Any]] = []

    def account(key: tuple[str | None, int | None]) -> dict[str, Any]:
        tally = tallies.pop(key, _Tally())
        return {
            "trades": tally.trades,
            "measured": tally.measured,
            "wins": tally.wins,
            "realised": str(tally.realised),
            "last_closed": tally.last_closed,
        }

    for row in rows:
        key = (str(row["strategy_id"]), int(row["version"]))
        out.append(
            {
                "strategy_id": key[0],
                "version": key[1],
                "status": str(row["status"]),
                "rung": int(row["rung"]),
                "lineage_id": str(row["lineage_id"]),
                "lifetime_realised": str(row["realised_pnl_ccy"]),
                "lifetime_trades": int(row["n_realised_trades"]),
                "lineage_budget": _decimal(row["budget_ccy"]),
                "lineage_consumed": _decimal(row["consumed_ccy"]),
                "account": account(key),
            }
        )
    for unplaced in sorted(tallies, key=lambda k: (k[0] or "", k[1] or 0)):
        out.append(
            {
                "strategy_id": unplaced[0],
                "version": unplaced[1],
                "status": None,
                "rung": None,
                "lineage_id": None,
                "lifetime_realised": None,
                "lifetime_trades": None,
                "lineage_budget": None,
                "lineage_consumed": None,
                "account": account(unplaced),
            }
        )
    return out


def sessions(
    ledger: Ledger, pinned: PinnedLimits, *, mode: str, limit: int = 30, now: datetime | None = None
) -> dict[str, Any]:
    """The newest sessions of one mode, judged, and the clean streak they make."""
    record = read_sessions(ledger, limits=pinned.limits.live, mode=_mode(mode), now=now)
    return {
        "mode": mode,
        "streak": len(record.streak),
        "trades_in_streak": record.trades_in_streak,
        "required": pinned.limits.live.min_clean_demo_sessions if mode == "demo" else None,
        "sessions": [
            {
                "date": s.session_date.isoformat(),
                "verdict": s.verdict.value,
                "coverage_pct": round(s.coverage_pct, 1),
                "cycles": s.n_cycles,
                "orders": s.n_orders,
                "fills": s.n_fills,
                "trades": s.n_trades_closed,
                "why": s.summary(),
            }
            for s in reversed(record.sessions[-limit:])
        ],
    }


def events(ledger: Ledger, *, after: int = 0, limit: int = 50) -> list[dict[str, Any]]:
    """Up to `limit` of the newest events after `after`, oldest first — a tail
    to poll. A reader that fell further behind than `limit` skips ahead."""
    head = ledger.head()
    if head is None or head.seq <= after:
        return []
    start = max(after + 1, head.seq - limit + 1)
    tail: list[dict[str, Any]] = []
    for row in ledger.iter_events(start_seq=start, end_seq=head.seq):
        text = str(row["payload_json"])
        cut = len(text) > PAYLOAD_LIMIT
        tail.append(
            {
                "seq": int(row["seq"]),
                "ts": str(row["ts_utc"]),
                "type": str(row["event_type"]),
                "aggregate_id": str(row["aggregate_id"]),
                "actor": str(row["actor"]),
                "run_id": row["run_id"],
                "payload": None if cut else json.loads(text),
                "truncated": cut,
            }
        )
    return tail


def journal_pages(directory: Path) -> list[dict[str, Any]]:
    """The journal's pages, newest first, with what each header claims."""
    pages: list[dict[str, Any]] = []
    if not directory.is_dir():
        return pages
    for path in sorted(directory.glob("????-??-??.md"), reverse=True):
        header: PageHeader | None
        try:
            header = parse_header(path.read_text(encoding="utf-8"))
        except (OSError, UnicodeDecodeError, JournalError):
            header = None
        pages.append(
            {
                "date": path.stem,
                "status": None if header is None else header.status,
                "through_seq": None if header is None else header.through_seq,
            }
        )
    return pages
