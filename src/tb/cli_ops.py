"""`tb sessions` — the M8 operator's view of how the loop has been running.

    tb sessions                  the recent demo sessions, and the clean streak
    tb sessions --mode paper     the same for paper
    tb sessions --date 2026-09-25   one session in full: every fault and note

The streak it prints is the number `tb arm --live` will count, computed by the
same function from the same ledger, so what an operator reads here is what
the gate will see.
"""

from __future__ import annotations

import os
import sys
from datetime import date
from pathlib import Path
from typing import Annotated

import typer
from rich.console import Console
from rich.markup import escape
from rich.table import Table

from tb.config.loader import PinnedLimits, load_hard_limits
from tb.core.errors import TbError
from tb.ledger.store import Ledger, default_ledger_path
from tb.ops.sessions import TRADING_MODES, SessionHealth, SessionVerdict, read_sessions

_WIDTH: int | None = None if sys.stdout.isatty() else int(os.environ.get("COLUMNS") or 120)
console = Console(width=_WIDTH)
err_console = Console(stderr=True, width=_WIDTH)

OK = "[green]✓[/green]"
WARN = "[yellow]![/yellow]"
BAD = "[red]✗[/red]"

LimitsOpt = Annotated[
    Path | None, typer.Option("--limits", help="Path to hard_limits.yaml.", show_default=False)
]
DbOpt = Annotated[
    Path | None, typer.Option("--db", help="Path to the ledger database.", show_default=False)
]

_VERDICT_STYLE = {
    SessionVerdict.CLEAN: "[green]clean[/green]",
    SessionVerdict.FAULTED: "[red]faulted[/red]",
    SessionVerdict.INCOMPLETE: "[yellow]incomplete[/yellow]",
    SessionVerdict.IN_PROGRESS: "[dim]in progress[/dim]",
}


def _load(limits: Path | None) -> PinnedLimits:
    try:
        return load_hard_limits(limits)
    except TbError as exc:
        err_console.print(f"{BAD} {escape(str(exc))}", soft_wrap=True)
        raise typer.Exit(2) from exc


def _ledger(db: Path | None) -> Ledger:
    """Read-only: a report has no business writing the record it reports on."""
    path = db or default_ledger_path()
    if not path.exists():
        err_console.print(f"{BAD} no ledger at {path}. Run `tb init` first.", soft_wrap=True)
        raise typer.Exit(2)
    try:
        return Ledger(path, read_only=True).open()
    except TbError as exc:
        err_console.print(f"{BAD} {escape(str(exc))}", soft_wrap=True)
        raise typer.Exit(2) from exc


def sessions_command(
    mode: Annotated[
        str, typer.Option("--mode", help="Which mode's sessions: paper, demo or live.")
    ] = "demo",
    day: Annotated[
        str | None,
        typer.Option("--date", help="Show one session in full (YYYY-MM-DD).", show_default=False),
    ] = None,
    limit: Annotated[
        int, typer.Option("--limit", min=1, help="How many recent sessions to list.")
    ] = 20,
    limits: LimitsOpt = None,
    db: DbOpt = None,
) -> None:
    """Which sessions the loop ran cleanly, and the streak the live gate counts."""
    if mode not in TRADING_MODES:
        err_console.print(
            f"{BAD} unknown mode {mode!r}; one of {', '.join(TRADING_MODES)}.", soft_wrap=True
        )
        raise typer.Exit(2)
    wanted: date | None = None
    if day is not None:
        try:
            wanted = date.fromisoformat(day)
        except ValueError as exc:
            err_console.print(f"{BAD} --date wants YYYY-MM-DD, got {day!r}.", soft_wrap=True)
            raise typer.Exit(2) from exc

    pinned = _load(limits)
    live = pinned.limits.live
    with _ledger(db) as ledger:
        record = read_sessions(ledger, limits=live, mode=mode)

    if wanted is not None:
        session = record.session(wanted)
        if session is None:
            err_console.print(
                f"{BAD} no {mode} session on {wanted}: no {mode} run was up for it.",
                soft_wrap=True,
            )
            raise typer.Exit(1)
        _print_session(session)
        return

    if not record.sessions:
        console.print(
            f"{WARN} no {mode} sessions in the ledger. A session is recorded by "
            f"`tb run --mode {mode}` being up for a trading day."
        )
    else:
        table = Table(
            "date", "verdict", "coverage", "cycles", "orders", "fills", "trades", "why",
            title=f"{mode} sessions",
        )  # fmt: skip
        for session in record.sessions[-limit:]:
            table.add_row(
                session.session_date.isoformat() + (" ½" if session.half_day else ""),
                _VERDICT_STYLE[session.verdict],
                f"{session.coverage_pct:.1f}%",
                str(session.n_cycles),
                str(session.n_orders),
                str(session.n_fills),
                str(session.n_trades_closed),
                escape(session.summary()),
            )
        console.print(table)

    streak = record.streak
    marker = OK if streak else WARN
    since = f", since {streak[0].session_date}" if streak else ""
    console.print(
        f"\n{marker} {len(streak)} consecutive clean {mode} session(s){since}, "
        f"{record.trades_in_streak} trade(s) closed across them."
    )
    if mode == "demo":
        console.print(
            f"  `tb arm --live` needs {live.min_clean_demo_sessions} and "
            f"{live.min_demo_closed_trades}. A session is clean when the loop covered "
            f"{live.session_min_coverage_pct:g}% of it with nothing halting, crashing or "
            "leaving a position unprotected."
        )


def _print_session(session: SessionHealth) -> None:
    table = Table(show_header=False, box=None, padding=(0, 2, 0, 0))
    table.add_row("session", f"{session.session_date} ({session.mode})")
    table.add_row("verdict", _VERDICT_STYLE[session.verdict])
    table.add_row(
        "hours (UTC)",
        f"{session.open_utc:%H:%M} to {session.close_utc:%H:%M}"
        + (" (half day)" if session.half_day else ""),
    )
    table.add_row("coverage", f"{session.coverage_pct:.1f}%")
    table.add_row("longest gap", f"{session.longest_gap_seconds:.0f}s")
    table.add_row(
        "activity",
        f"{session.n_cycles} cycle(s), {session.n_decisions} decision(s), "
        f"{session.n_orders} order(s), {session.n_fills} fill(s), "
        f"{session.n_trades_closed} trade(s) closed",
    )
    table.add_row("runs", ", ".join(session.run_ids) or "—")
    table.add_row("code", ", ".join(session.code_shas) or "—")
    table.add_row("limits", ", ".join(h[:12] for h in session.config_hashes) or "—")
    console.print(table)

    for title, findings, marker in (
        ("faults", session.faults, BAD),
        ("notes", session.notes, WARN),
    ):
        if not findings:
            continue
        console.print(f"\n[bold]{title}[/bold]")
        for finding in findings:
            console.print(
                f"  {marker} {finding.at:%H:%M:%S} seq {finding.seq} "
                f"{escape(f'[{finding.kind}]')} {escape(finding.detail)}",
                soft_wrap=True,
            )
