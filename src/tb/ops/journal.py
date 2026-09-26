"""The daily journal: one markdown page per session, written from the ledger.

A page is a pure function of the ledger read up to one event. Its front matter
names that event's sequence number and chain hash, the moment it was written,
and the hash of the limits its sessions were judged by. So a page can be
produced again byte for byte, and `tb journal verify` does exactly that:

* the chain hash at the page's seq must still be the one it names — a ledger
  rewritten beneath a committed page no longer produces it;
* the page must still be what that ledger produces — an edited page, or a
  ledger edited in a way its own chain did not catch, reads differently, and
  the check names the sections that changed.

Committed to git with the chain head beside it (`tb journal write --commit`),
a page is also an anchor: once pushed, the remote holds a head the bot cannot
take back.

**What a page covers** is the session record's window: everything recorded
after the previous session's close up to this session's close. Every event
lands on exactly one page, and a weekend's research is Monday's.

**One section is not from the chain.** The account's equity marks are
measurements the loop re-takes from the broker every cycle and deliberately
does not chain. The page shows them, says so, and verification still covers
them — an edited mark changes the page — but the chain does not vouch for
them the way it does for everything else here.

Text from outside the system — a broker's rejection message, a provider's
reason — reaches a page only escaped: a page is data, not markup.
"""

from __future__ import annotations

import json
import os
from collections import Counter
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from datetime import date, datetime, timedelta
from decimal import ROUND_HALF_UP, Decimal, InvalidOperation
from pathlib import Path
from typing import Any

from tb.config.hard_limits import HardLimits, LiveLimits
from tb.config.loader import PinnedLimits
from tb.core.clock import from_iso, now_utc, to_iso
from tb.core.errors import TbError
from tb.data.calendar import DayKind, TradingCalendar, TradingDay
from tb.data.provider import US_EASTERN
from tb.ledger.events import EventType
from tb.ledger.store import Ledger
from tb.ops.sessions import (
    TRADING_MODES,
    ModeRecord,
    SessionHealth,
    read_all_sessions,
    session_of,
    silence_bound,
)

JOURNAL_VERSION = 1
DEFAULT_DIR = Path("journal")
HEADS_FILE = "chain-heads.jsonl"

# The furthest back a page's window can start. A window opens at the previous
# session's close, and the calendar never puts that more than a long holiday
# weekend behind; this is a prefilter, and `session_of` decides exactly.
_LOOKBACK = timedelta(days=11)

_ORDER_LIFECYCLE = frozenset(
    {
        EventType.ORDER_SUBMITTED,
        EventType.ORDER_ACKNOWLEDGED,
        EventType.ORDER_REJECTED,
        EventType.ORDER_CANCELLED,
        EventType.INTENT_RESOLVED,
    }
)


class JournalError(TbError):
    """A page could not be written, read or checked."""


@dataclass(frozen=True, slots=True)
class PageHeader:
    """The front matter: what a page claims about where it came from."""

    session: date
    as_of: datetime
    through_seq: int
    chain_hash: str
    limits_hash: str
    status: str

    def render(self) -> list[str]:
        return [
            "---",
            f"tb_journal: {JOURNAL_VERSION}",
            f"session: {self.session.isoformat()}",
            f"as_of: {to_iso(self.as_of)}",
            f"through_seq: {self.through_seq}",
            f"chain_hash: {self.chain_hash}",
            f"limits: {self.limits_hash}",
            f"status: {self.status}",
            "---",
        ]


@dataclass(frozen=True, slots=True)
class JournalPage:
    header: PageHeader
    text: str

    @property
    def filename(self) -> str:
        return page_filename(self.header.session)

    @property
    def final(self) -> bool:
        return self.header.status == "final"


@dataclass(frozen=True, slots=True)
class PageCheck:
    """Whether a page is still what the ledger produces, and if not, why."""

    ok: bool
    header: PageHeader | None
    problems: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class WriteOutcome:
    session: date
    path: Path
    action: str  # written | rewritten | unchanged | refused
    detail: str = ""


def page_filename(session: date) -> str:
    return f"{session.isoformat()}.md"


# --------------------------------------------------------------------------
# Rendering
# --------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class _Ev:
    seq: int
    at: datetime
    type: str
    aggregate_id: str
    run_id: str | None
    payload: dict[str, Any]


def render_page(
    ledger: Ledger,
    *,
    session: date,
    limits: HardLimits,
    limits_hash: str,
    as_of: datetime | None = None,
    through_seq: int | None = None,
    calendar: TradingCalendar | None = None,
) -> JournalPage:
    """The page for one session, from the ledger as it stood at `through_seq`.

    `as_of` is when the page is written, and decides whether the session can
    be judged yet. It is never earlier than the last event read: a page cannot
    have been written before the ledger it was written from.
    """
    cal = calendar or TradingCalendar()
    day = cal.classify(session)
    if not day.is_trading_day or day.open_utc is None or day.close_utc is None:
        raise JournalError(
            f"{session} is not a trading session ({day.kind.value}); a page is written per "
            "session, and what happened on a closed day is on the next session's page"
        )
    head = ledger.head()
    if head is None:
        raise JournalError("the ledger is empty; there is nothing to write a page from")
    seq = head.seq if through_seq is None else through_seq
    row = ledger.get(seq)
    if row is None:
        raise JournalError(f"the ledger has no event at seq {seq}")
    last = from_iso(str(row["ts_utc"]))
    moment = max(as_of or now_utc(), last)
    final = moment >= day.close_utc + silence_bound(limits.live)
    header = PageHeader(
        session=session,
        as_of=moment,
        through_seq=seq,
        chain_hash=str(row["chain_hash"]),
        limits_hash=limits_hash,
        status="final" if final else "provisional",
    )

    window = _window(ledger, cal, day, through_seq=seq)
    modes = _run_modes(ledger, through_seq=seq)
    records = read_all_sessions(
        ledger, limits=limits.live, calendar=cal, now=moment, through_seq=seq
    )
    committed = [e for e in window if e.type == EventType.INTENT_COMMITTED.value]
    outcomes = _outcomes(ledger, {str(e.payload.get("intent_id")) for e in committed}, seq)

    lines = header.render()
    lines += _title(cal, day, window, final=final, moment=moment)
    lines += _sessions_section(records, day.day)
    lines += _account_section(ledger, cal, day, final=final, modes=modes)
    lines += _orders_section(committed, outcomes, modes)
    lines += _fills_section(window, modes)
    lines += _trades_section(window, modes)
    lines += _protection_section(window, modes)
    lines += _list_section("Strategies", window, _STRATEGY_LINES)
    lines += _research_section(window)
    lines += _list_section("Safety", window, _SAFETY_LINES)
    lines += _data_section(window)
    lines += _ledger_section(window)
    return JournalPage(header=header, text="\n".join(lines).rstrip("\n") + "\n")


def _window(
    ledger: Ledger, cal: TradingCalendar, day: TradingDay, *, through_seq: int
) -> list[_Ev]:
    """Every event this session's page covers, in sequence order."""
    assert day.close_utc is not None
    events: list[_Ev] = []
    for row in ledger.iter_events(
        end_seq=through_seq,
        after_ts=to_iso(day.close_utc - _LOOKBACK),
        until_ts=to_iso(day.close_utc),
    ):
        at = from_iso(str(row["ts_utc"]))
        home = session_of(cal, at)
        if home is None or home.day != day.day:
            continue
        payload = json.loads(row["payload_json"])
        run_id = payload.get("run_id") or row["run_id"]
        events.append(
            _Ev(
                seq=int(row["seq"]),
                at=at,
                type=str(row["event_type"]),
                aggregate_id=str(row["aggregate_id"]),
                run_id=str(run_id) if run_id else None,
                payload=payload,
            )
        )
    return events


def _run_modes(ledger: Ledger, *, through_seq: int) -> dict[str, str]:
    modes: dict[str, str] = {}
    for row in ledger.iter_events(end_seq=through_seq, event_type=EventType.RUN_STARTED):
        payload = json.loads(row["payload_json"])
        modes[str(payload["run_id"])] = str(payload.get("mode", ""))
    return modes


def _outcomes(ledger: Ledger, intent_ids: set[str], through_seq: int) -> dict[str, str]:
    """What became of each intent, as far as the ledger knew at `through_seq`."""
    outcomes: dict[str, str] = {}
    if not intent_ids:
        return outcomes
    for row in ledger.iter_events(end_seq=through_seq, event_types=_ORDER_LIFECYCLE):
        payload = json.loads(row["payload_json"])
        intent_id = payload.get("intent_id")
        if intent_id not in intent_ids:
            continue
        kind = EventType(row["event_type"])
        if kind is EventType.ORDER_SUBMITTED:
            outcomes[intent_id] = "sent, unanswered"
        elif kind is EventType.ORDER_ACKNOWLEDGED:
            outcomes[intent_id] = f"{payload.get('status')} ({payload.get('broker_order_id')})"
        elif kind is EventType.ORDER_REJECTED:
            outcomes[intent_id] = f"rejected: {_said(payload)}"
        elif kind is EventType.ORDER_CANCELLED:
            outcomes[intent_id] = "cancelled"
        else:
            final = str(payload.get("final_state", "")).removeprefix("resolved_")
            outcomes[intent_id] = f"{final} ({payload.get('resolved_by')})"
    return outcomes


def _title(
    cal: TradingCalendar, day: TradingDay, window: Sequence[_Ev], *, final: bool, moment: datetime
) -> list[str]:
    assert day.open_utc is not None and day.close_utc is not None
    opens, closes = day.open_utc.astimezone(US_EASTERN), day.close_utc.astimezone(US_EASTERN)
    kind = "Half day" if day.kind is DayKind.HALF_DAY else "Regular session"
    previous = cal.previous_session(day.day)
    since = (
        f"the previous close, {_long_date(previous.day)} at "
        f"{previous.close_utc.astimezone(US_EASTERN):%H:%M} ET"
        if previous is not None and previous.close_utc is not None
        else "the start of the calendar"
    )
    lines = [
        "",
        f"# {_long_date(day.day)}",
        "",
        f"{kind}, {opens:%H:%M} to {closes:%H:%M} ET ({day.open_utc:%H:%M} to "
        f"{day.close_utc:%H:%M} UTC). This page is everything the ledger recorded from "
        f"{since}, to this close. Times are UTC.",
    ]
    if window:
        lines += ["", f"Events seq {window[0].seq} to {window[-1].seq}: {len(window)} in all."]
    else:
        lines += ["", "The ledger recorded nothing in this window."]
    if not final:
        lines += [
            "",
            f"**Provisional.** Written at {to_iso(moment)}, before the session could be "
            "judged; a later page for this session replaces it.",
        ]
    return lines


def _sessions_section(records: dict[str, ModeRecord], day: date) -> list[str]:
    lines = ["", "## Sessions", ""]
    found: list[SessionHealth] = []
    for mode in TRADING_MODES:
        session = records[mode].session(day)
        if session is not None:
            found.append(session)
    if not found:
        return [*lines, "No trading run was up for this session."]
    lines += _table(
        (
            "mode", "verdict", "coverage", "longest gap", "cycles", "decisions", "orders",
            "fills", "trades closed", "runs",
        ),
        [
            (
                s.mode,
                s.verdict.value,
                f"{s.coverage_pct:.1f}%",
                f"{s.longest_gap_seconds:.0f}s",
                s.n_cycles,
                s.n_decisions,
                s.n_orders,
                s.n_fills,
                s.n_trades_closed,
                ", ".join(s.run_ids),
            )
            for s in found
        ],
    )  # fmt: skip
    findings = [
        (s.mode, label, finding)
        for s in found
        for label, group in (("fault", s.faults), ("note", s.notes))
        for finding in group
    ]
    if findings:
        lines.append("")
        lines += [
            f"- {mode} {label} `{f.kind}` at {f.at:%H:%M:%S} (seq {f.seq}"
            + (f", run {f.run_id}" if f.run_id else "")
            + f"): {_text(f.detail)}"
            for mode, label, f in findings
        ]
    return lines


def _account_section(
    ledger: Ledger,
    cal: TradingCalendar,
    day: TradingDay,
    *,
    final: bool,
    modes: dict[str, str],
) -> list[str]:
    assert day.close_utc is not None
    lines = [
        "",
        "## Account",
        "",
        "Equity as the broker reported it each cycle. These marks are measurements the "
        "loop re-takes rather than chained events, so this is the one section the chain "
        "does not vouch for.",
        "",
    ]
    if not final:
        # A mark carries the instant its cycle began and lands seconds later,
        # after the broker answered. Mid-session, a page could be written in
        # between and read differently the next time; past the close and the
        # silence bound, every mark of the session has long since landed.
        return [*lines, "Shown once the session is final."]
    rows = ledger.conn.execute(
        "SELECT run_id, at_utc, equity, currency, deployed FROM equity_marks"
        " WHERE at_utc > ? AND at_utc <= ? ORDER BY at_utc, mark_id",
        (to_iso(day.close_utc - _LOOKBACK), to_iso(day.close_utc)),
    ).fetchall()
    by_mode: dict[str, list[tuple[datetime, Decimal, str, str | None]]] = {}
    for row in rows:
        at = from_iso(str(row["at_utc"]))
        home = session_of(cal, at)
        if home is None or home.day != day.day:
            continue
        equity = _decimal(row["equity"])
        if equity is None:
            continue
        mode = modes.get(str(row["run_id"]), "unattributed")
        by_mode.setdefault(mode, []).append(
            (at, equity, str(row["currency"] or ""), row["deployed"])
        )
    if not by_mode:
        return [*lines, "No equity was marked in this window."]
    table: list[tuple[object, ...]] = []
    for mode in sorted(by_mode, key=_mode_order):
        marks = by_mode[mode]
        first, last = marks[0], marks[-1]
        change = last[1] - first[1]
        pct = (change / first[1] * 100).quantize(Decimal("0.01"), rounding=ROUND_HALF_UP)
        table.append(
            (
                mode,
                len(marks),
                last[2] or "—",
                f"{first[1]} at {first[0]:%H:%M:%S}",
                f"{last[1]} at {last[0]:%H:%M:%S}",
                f"{change:+} ({pct:+}%)",
                max(m[1] for m in marks),
                min(m[1] for m in marks),
                last[3],
            )
        )
    return lines + _table(
        ("mode", "marks", "currency", "first", "last", "change", "high", "low", "deployed"),
        table,
    )


def _orders_section(
    committed: Sequence[_Ev], outcomes: dict[str, str], modes: dict[str, str]
) -> list[str]:
    lines = ["", "## Orders", ""]
    if not committed:
        return [*lines, "None."]
    return lines + _table(
        ("time", "mode", "intent", "ticker", "side", "purpose", "type", "quantity", "outcome"),
        [
            (
                f"{e.at:%H:%M:%S}",
                _mode(e, modes),
                e.payload.get("intent_id"),
                e.payload.get("t212_ticker"),
                e.payload.get("side"),
                e.payload.get("purpose"),
                e.payload.get("order_type"),
                e.payload.get("quantity"),
                outcomes.get(str(e.payload.get("intent_id")), "pending (never sent)"),
            )
            for e in committed
        ],
    )


def _fills_section(window: Sequence[_Ev], modes: dict[str, str]) -> list[str]:
    fills = [e for e in window if e.type == EventType.FILL_RECORDED.value]
    lines = ["", "## Fills", ""]
    if not fills:
        return [*lines, "None."]
    return lines + _table(
        ("time", "mode", "ticker", "side", "quantity", "price", "source", "counted"),
        [
            (
                f"{e.at:%H:%M:%S}",
                _mode(e, modes),
                e.payload.get("t212_ticker"),
                e.payload.get("side"),
                e.payload.get("quantity"),
                e.payload.get("price"),
                e.payload.get("source"),
                "yes" if e.payload.get("admissible_for_pnl") else "no",
            )
            for e in fills
        ],
    )


def _trades_section(window: Sequence[_Ev], modes: dict[str, str]) -> list[str]:
    trades = [e for e in window if e.type == EventType.TRADE_CLOSED.value]
    lines = ["", "## Trades closed", ""]
    if not trades:
        return [*lines, "None."]
    lines += _table(
        ("time", "mode", "ticker", "strategy", "quantity", "exit", "P&L", "charged"),
        [
            (
                f"{e.at:%H:%M:%S}",
                _mode(e, modes),
                e.payload.get("t212_ticker"),
                _strategy(e.payload),
                e.payload.get("quantity"),
                e.payload.get("exit_price"),
                e.payload.get("pnl_ccy"),
                "yes" if e.payload.get("charged") else f"no: {e.payload.get('detail') or '—'}",
            )
            for e in trades
        ],
    )
    counted = [
        pnl
        for e in trades
        if e.payload.get("admissible") and (pnl := _decimal(e.payload.get("pnl_ccy"))) is not None
    ]
    total = sum(counted, Decimal(0))
    unpriced = len(trades) - len(counted)
    lines += [
        "",
        f"Realised {total:+} across {len(counted)} priced trade(s)"
        + (f"; {unpriced} without a reported price, not counted." if unpriced else "."),
    ]
    return lines


def _protection_section(window: Sequence[_Ev], modes: dict[str, str]) -> list[str]:
    events = [
        e
        for e in window
        if e.type in (EventType.POSITION_PROTECTED.value, EventType.POSITION_UNPROTECTED.value)
    ]
    lines = ["", "## Protection", ""]
    if not events:
        return [*lines, "None."]

    def state(e: _Ev) -> str:
        if e.payload.get("protected"):
            bare = e.payload.get("unprotected_seconds")
            return "protected" + (f" after {float(bare):.0f}s bare" if bare is not None else "")
        return f"unprotected ({e.payload.get('cause') or 'cause not recorded'})"

    return lines + _table(
        ("time", "mode", "ticker", "state", "stop", "detail"),
        [
            (
                f"{e.at:%H:%M:%S}",
                _mode(e, modes),
                e.payload.get("t212_ticker"),
                state(e),
                e.payload.get("stop_price"),
                e.payload.get("detail"),
            )
            for e in events
        ],
    )


def _list_section(
    title: str, window: Sequence[_Ev], formatters: dict[EventType, Callable[[dict[str, Any]], str]]
) -> list[str]:
    wanted = {kind.value: fmt for kind, fmt in formatters.items()}
    items = [
        f"- {e.at:%H:%M:%S} {_text(wanted[e.type](e.payload))}" for e in window if e.type in wanted
    ]
    return ["", f"## {title}", "", *(items or ["None."])]


def _research_section(window: Sequence[_Ev]) -> list[str]:
    lines = ["", "## Research", ""]
    trials = Counter(
        str(e.payload.get("outcome")) for e in window if e.type == EventType.TRIAL_RECORDED.value
    )
    counts = Counter(e.type for e in window)
    items: list[str] = []
    if trials:
        by_outcome = ", ".join(f"{outcome} {n}" for outcome, n in sorted(trials.items()))
        items.append(f"- trials recorded: {sum(trials.values())} ({by_outcome})")
    for kind, label in (
        (EventType.STRATEGY_SPEC_REGISTERED, "specs registered"),
        (EventType.BACKTEST_COMPLETED, "backtests completed"),
    ):
        if counts[kind.value]:
            items.append(f"- {label}: {counts[kind.value]}")
    wanted = {kind.value: fmt for kind, fmt in _RESEARCH_LINES.items()}
    items += [
        f"- {e.at:%H:%M:%S} {_text(wanted[e.type](e.payload))}" for e in window if e.type in wanted
    ]
    return lines + (items or ["None."])


def _data_section(window: Sequence[_Ev]) -> list[str]:
    lines = ["", "## Data", ""]
    counts = Counter(e.type for e in window)
    items: list[str] = []
    for kind, label in (
        (EventType.DATA_PARTITION_SEALED, "partitions sealed"),
        (EventType.DATA_BAR_REVISION_DETECTED, "bar revisions detected"),
        (EventType.DATA_STALENESS_BREACH, "decisions refused on stale data"),
        (EventType.DATA_ACTION_RECORDED, "corporate actions recorded"),
    ):
        if counts[kind.value]:
            items.append(f"- {label}: {counts[kind.value]}")
    wanted = {kind.value: fmt for kind, fmt in _DATA_LINES.items()}
    items += [
        f"- {e.at:%H:%M:%S} {_text(wanted[e.type](e.payload))}" for e in window if e.type in wanted
    ]
    return lines + (items or ["None."])


def _ledger_section(window: Sequence[_Ev]) -> list[str]:
    lines = ["", "## Ledger", ""]
    if not window:
        return [*lines, "Nothing recorded."]
    counts = Counter(e.type for e in window)
    lines += _table(("event", "count"), sorted(counts.items()))
    anchors = [e for e in window if e.type == EventType.CHAIN_ANCHORED.value]
    if anchors:
        lines += ["", "Chain heads published:", ""]
        lines += [
            f"- {e.at:%H:%M:%S} seq {e.payload.get('anchored_seq')} via "
            f"{e.payload.get('sink')}"
            + (
                f" ({_text(str(e.payload['external_ref']))})"
                if e.payload.get("external_ref")
                else ""
            )
            for e in anchors
        ]
    return lines


# --------------------------------------------------------------------------
# One line per event, by type
# --------------------------------------------------------------------------


def _said(p: dict[str, Any]) -> str:
    said = p.get("broker_message")
    return f"{p.get('detail')}" + (f" ({said})" if said else "")


def _strategy(p: dict[str, Any]) -> str:
    sid = p.get("strategy_id")
    if not sid:
        return "—"
    version = p.get("strategy_version", p.get("version"))
    return f"{sid} v{version}" if version is not None else str(sid)


def _book(p: dict[str, Any]) -> str:
    funded = "; ".join(
        f"{entry.get('strategy_id')} v{entry.get('version')} rung {entry.get('rung')} "
        f"at {entry.get('notional_ccy')}"
        for entry in p.get("entries", [])
    )
    excluded = len(p.get("excluded", []))
    return (
        f"book funded for run {p.get('run_id')} ({p.get('source')}): "
        f"{p.get('n_funded')} strateg{'y' if p.get('n_funded') == 1 else 'ies'}"
        + (f": {funded}" if funded else "")
        + (f"; {excluded} excluded" if excluded else "")
    )


_STRATEGY_LINES: dict[EventType, Callable[[dict[str, Any]], str]] = {
    EventType.BOOK_FUNDED: _book,
    EventType.PROMOTION_EVALUATED: lambda p: (
        f"{_strategy(p)} promotion: {p.get('decision')} "
        f"({p.get('n_failed')} of {p.get('n_gates')} gates failed)"
    ),
    EventType.STRATEGY_REVIEWED: lambda p: (
        f"{_strategy(p)} reviewed: {p.get('verdict')} on {p.get('n_realised_trades')} "
        f"realised trade(s), P&L {p.get('realised_pnl_ccy') or '—'}"
    ),
    EventType.SESSION_REVIEWED: lambda p: (
        f"session review of {p.get('n_strategies')} strateg"
        f"{'y' if p.get('n_strategies') == 1 else 'ies'}"
    ),
    EventType.LADDER_MOVED: lambda p: (
        f"{_strategy(p)} moved {p.get('direction')} from rung {p.get('from_rung')} to "
        f"{p.get('to_rung')}: {p.get('reason')}"
    ),
    EventType.ALLOCATION_DECIDED: lambda p: (
        f"allocation {p.get('allocation_id')}: {p.get('n_strategies')} strategy(ies), "
        f"{p.get('total_notional_ccy')} in all"
    ),
    EventType.STRATEGY_RETIRED: lambda p: f"{_strategy(p)} retired: {p.get('reason')}",
    EventType.LINEAGE_BUDGET_EXHAUSTED: lambda p: (
        f"lineage {p.get('lineage_id')} out of budget: {p.get('consumed_ccy')} of "
        f"{p.get('budget_ccy')} spent"
    ),
}

_RESEARCH_LINES: dict[EventType, Callable[[dict[str, Any]], str]] = {
    EventType.SEARCH_COMPLETED: lambda p: (
        f"search {p.get('search_id')}: {p.get('n_proposed')} proposed, "
        f"{p.get('n_evaluated')} evaluated, {p.get('n_rejected')} rejected, "
        f"{p.get('n_passed_gate')} passed the gate"
    ),
    EventType.SPECS_PROPOSED: lambda p: (
        f"{p.get('proposer')} ({p.get('served_model')}) proposed {p.get('n_items')} spec(s), "
        f"{p.get('n_accepted')} accepted"
    ),
    EventType.HOLDOUT_EVALUATED: lambda p: (
        f"holdout for {p.get('strategy_id')} v{p.get('version')}: "
        f"{'passed' if p.get('passed') else 'failed'}"
    ),
    EventType.HOLDOUT_VIOLATION_ATTEMPTED: lambda p: (
        f"a read past the seal ({p.get('sealed_from')}) was refused: {p.get('caller')}"
    ),
    EventType.MODEL_RECORDED: lambda p: (
        f"model {p.get('model_id')} ({p.get('kind')}) recorded from {p.get('n_samples')} "
        f"samples on vintage {p.get('vintage_id')}"
    ),
}

_SAFETY_LINES: dict[EventType, Callable[[dict[str, Any]], str]] = {
    EventType.STATE_TRANSITIONED: lambda p: (
        f"run state {p.get('from_state')} to {p.get('to_state')}: {p.get('reason')}"
    ),
    EventType.HALT_RAISED: lambda p: (
        f"halt {p.get('halt_id')} [{p.get('trigger')}]: {p.get('detail')}"
    ),
    EventType.HALT_CLEARED: lambda p: (
        f"halt {p.get('halt_id')} cleared by {p.get('cleared_by')}: {p.get('clear_reason')}"
    ),
    EventType.KILLSWITCH_ENGAGED: lambda p: (
        f"kill switch engaged by {p.get('engaged_by') or 'unknown'}: {p.get('detail')}"
    ),
    EventType.KILLSWITCH_RELEASED: lambda p: f"kill switch released: {p.get('detail')}",
    EventType.WATCHDOG_TRIPPED: lambda p: f"watchdog {p.get('action_taken')}: {p.get('detail')}",
    EventType.WATCHDOG_UNREACHABLE: lambda p: f"self-check {p.get('direction')}: {p.get('detail')}",
    EventType.CONFIG_DRIFT_DETECTED: lambda p: f"limits changed under a run: {p.get('detail')}",
    EventType.BROKER_SCHEMA_DRIFT: lambda p: (
        f"broker schema drift on {p.get('endpoint')}: {p.get('error_detail')}"
    ),
    EventType.INSTANCE_LOCK_REFUSED: lambda p: (
        f"second instance refused (run {p.get('run_id')}, held by {p.get('held_by_run_id')})"
    ),
    EventType.RECONCILE_COMPLETED: lambda p: (
        f"reconcile {p.get('verdict')}: {p.get('n_unknown_intents')} unknown intent(s), "
        f"{p.get('n_orphan_orders')} orphan order(s), {p.get('n_position_mismatches')} "
        f"position mismatch(es), {p.get('n_unprotected_positions')} unprotected"
    ),
    EventType.POSITION_ORPHANED: lambda p: (
        f"{p.get('t212_ticker')} orphaned: {p.get('reason')}; {p.get('action_taken')}"
    ),
    EventType.DRILL_STARTED: lambda p: (
        f"{p.get('kind')} drill {p.get('drill_id')} started on run {p.get('run_id')} with "
        f"{len(p.get('holdings', []))} position(s) held"
    ),
    EventType.DRILL_COMPLETED: lambda p: (
        f"{p.get('kind')} drill {p.get('drill_id')} "
        + (
            "passed"
            if p.get("passed")
            else "failed: " + "; ".join(str(f) for f in p.get("failures", []))
        )
    ),
}

_DATA_LINES: dict[EventType, Callable[[dict[str, Any]], str]] = {
    EventType.DATA_PROVIDER_DEGRADED: lambda p: (
        f"{p.get('provider')} {p.get('resolution')} degraded: {p.get('reason')}"
    ),
    EventType.DATA_SNAPSHOT_SEALED: lambda p: (
        f"vintage {p.get('vintage_id')} sealed: {p.get('n_instruments')} instrument(s), "
        f"{p.get('row_count')} rows"
    ),
    EventType.DATA_AUDIT_COMPLETED: lambda p: (
        f"audit {p.get('audit_id')}: {p.get('n_findings')} finding(s), "
        f"{p.get('n_blocking')} blocking"
    ),
    EventType.SYMBOL_BLOCKED: lambda p: f"{p.get('t212_ticker')} blocked: {p.get('reason')}",
    EventType.SYMBOL_UNBLOCKED: lambda p: f"{p.get('t212_ticker')} unblocked: {p.get('reason')}",
}


# --------------------------------------------------------------------------
# Markdown
# --------------------------------------------------------------------------


def _text(value: str) -> str:
    """Untrusted text made inert: no markup, no table breaks, one line."""
    return (
        value.replace("\\", "\\\\")
        .replace("|", "\\|")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace("\r", " ")
        .replace("\n", " ")
    )


def _cell(value: object) -> str:
    if value is None or value == "":
        return "—"
    return _text(str(value))


def _table(headers: Sequence[str], rows: Sequence[Sequence[object]]) -> list[str]:
    return [
        "| " + " | ".join(headers) + " |",
        "|" + "|".join("---" for _ in headers) + "|",
        *("| " + " | ".join(_cell(value) for value in row) + " |" for row in rows),
    ]


def _mode(event: _Ev, modes: dict[str, str]) -> str:
    return modes.get(event.run_id or "", "—")


def _mode_order(mode: str) -> tuple[int, str]:
    return (TRADING_MODES.index(mode) if mode in TRADING_MODES else len(TRADING_MODES), mode)


def _decimal(value: object) -> Decimal | None:
    if value is None:
        return None
    try:
        return Decimal(str(value))
    except InvalidOperation:
        return None


def _long_date(day: date) -> str:
    return f"{day:%A} {day.day} {day:%B %Y}"


# --------------------------------------------------------------------------
# Reading a page back
# --------------------------------------------------------------------------


def parse_header(text: str) -> PageHeader:
    lines = text.splitlines()
    if not lines or lines[0] != "---":
        raise JournalError("no front matter: this is not a journal page")
    try:
        end = lines.index("---", 1)
    except ValueError as exc:
        raise JournalError("the front matter is never closed") from exc
    fields: dict[str, str] = {}
    for line in lines[1:end]:
        key, sep, value = line.partition(": ")
        if not sep:
            raise JournalError(f"malformed front matter line: {line!r}")
        fields[key] = value
    if fields.get("tb_journal") != str(JOURNAL_VERSION):
        raise JournalError(
            f"journal version {fields.get('tb_journal')!r}; this build reads {JOURNAL_VERSION}"
        )
    try:
        return PageHeader(
            session=date.fromisoformat(fields["session"]),
            as_of=from_iso(fields["as_of"]),
            through_seq=int(fields["through_seq"]),
            chain_hash=fields["chain_hash"],
            limits_hash=fields["limits"],
            status=fields["status"],
        )
    except (KeyError, ValueError) as exc:
        raise JournalError(f"the front matter is incomplete or malformed: {exc}") from exc


def pinned_limits(ledger: Ledger, config_hash: str) -> HardLimits | None:
    """The limits a hash names, as the ledger recorded them when they were pinned."""
    for row in ledger.iter_events(event_type=EventType.CONFIG_PINNED):
        payload = json.loads(row["payload_json"])
        if payload.get("config_hash") != config_hash:
            continue
        try:
            return HardLimits.model_validate(payload["values"])
        except Exception:
            return None
    return None


def ensure_pinned(ledger: Ledger, pinned: PinnedLimits) -> bool:
    """Record the limits a page is judged by, if the ledger has never seen them.

    Without the pin a page names a limits hash nothing can turn back into
    limits, and verification could not rebuild it. Returns whether it pinned.
    """
    if pinned_limits(ledger, pinned.config_hash) is not None:
        return False
    ledger.record_config_pin(pinned.audit_record())
    return True


def verify_page(ledger: Ledger, text: str, *, calendar: TradingCalendar | None = None) -> PageCheck:
    """Whether the ledger still produces this page, byte for byte."""
    try:
        header = parse_header(text)
    except JournalError as exc:
        return PageCheck(ok=False, header=None, problems=(str(exc),))
    row = ledger.get(header.through_seq)
    if row is None:
        return PageCheck(
            ok=False,
            header=header,
            problems=(
                f"the ledger has no event at seq {header.through_seq}: the page was written "
                "from another ledger, or this one has lost its tail",
            ),
        )
    if str(row["chain_hash"]) != header.chain_hash:
        return PageCheck(
            ok=False,
            header=header,
            problems=(
                f"the chain hash at seq {header.through_seq} is {str(row['chain_hash'])[:16]}…, "
                f"the page recorded {header.chain_hash[:16]}…: the ledger was rewritten "
                "beneath the page, or the page belongs to another ledger",
            ),
        )
    limits = pinned_limits(ledger, header.limits_hash)
    if limits is None:
        return PageCheck(
            ok=False,
            header=header,
            problems=(
                f"the limits {header.limits_hash[:16]}… the page was judged by were never "
                "pinned in this ledger, so its sessions cannot be judged again",
            ),
        )
    again = render_page(
        ledger,
        session=header.session,
        limits=limits,
        limits_hash=header.limits_hash,
        as_of=header.as_of,
        through_seq=header.through_seq,
        calendar=calendar,
    )
    if again.text == text:
        return PageCheck(ok=True, header=header)
    changed = _differing_sections(again.text, text)
    return PageCheck(
        ok=False,
        header=header,
        problems=(
            "the page is not what the ledger produces; it differs in: " + ", ".join(changed),
        ),
    )


def _sections(text: str) -> dict[str, str]:
    sections: dict[str, list[str]] = {"front matter and title": []}
    current = "front matter and title"
    for line in text.splitlines():
        if line.startswith("## "):
            current = line[3:]
            sections.setdefault(current, [])
        sections[current].append(line)
    return {name: "\n".join(body) for name, body in sections.items()}


def _differing_sections(expected: str, actual: str) -> list[str]:
    want, have = _sections(expected), _sections(actual)
    names = list(dict.fromkeys([*want, *have]))
    return [name for name in names if want.get(name) != have.get(name)] or ["whitespace"]


# --------------------------------------------------------------------------
# Writing pages
# --------------------------------------------------------------------------


def latest_final_session(
    calendar: TradingCalendar, *, now: datetime, limits: LiveLimits
) -> TradingDay | None:
    """The most recent session that can be judged at `now`."""
    day = session_of(calendar, now)
    bound = silence_bound(limits)
    for _ in range(15):
        if day is None or day.close_utc is None:
            return None
        if now >= day.close_utc + bound:
            return day
        day = calendar.previous_session(day.day)
    return None


def write_page(
    ledger: Ledger,
    page: JournalPage,
    *,
    directory: Path,
    calendar: TradingCalendar | None = None,
) -> WriteOutcome:
    """Put a page on disk, never over one that is evidence of something else.

    A final page already there that the ledger still produces is left alone:
    rewriting it would change nothing but the seq it names. A provisional one
    is replaced. One the ledger no longer produces is refused — it is exactly
    what verification exists to surface, and overwriting it would destroy it.
    """
    path = directory / page.filename
    action = "written"
    if path.exists():
        existing = path.read_text(encoding="utf-8")
        check = verify_page(ledger, existing, calendar=calendar)
        if not check.ok:
            return WriteOutcome(
                session=page.header.session,
                path=path,
                action="refused",
                detail=(
                    "the page on disk is not what the ledger produces ("
                    + "; ".join(check.problems)
                    + "). It is left as it is: `tb journal verify` says why."
                ),
            )
        if check.header is not None and check.header.status == "final":
            return WriteOutcome(
                session=page.header.session,
                path=path,
                action="unchanged",
                detail=f"already written through seq {check.header.through_seq}, and verified",
            )
        action = "rewritten"
    directory.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(page.text, encoding="utf-8")
    os.replace(temporary, path)
    return WriteOutcome(
        session=page.header.session,
        path=path,
        action=action,
        detail=f"{page.header.status}, through seq {page.header.through_seq}",
    )
