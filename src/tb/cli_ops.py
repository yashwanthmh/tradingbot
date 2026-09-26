"""The M8 operator's commands: how the loop has been running, and its record.

    tb sessions                     the recent demo sessions, and the clean streak
    tb sessions --mode paper        the same for paper
    tb sessions --date 2026-09-25   one session in full: every fault and note

    tb journal write                the latest finished session's page
    tb journal write --commit       ...committed to git with the chain head
    tb journal verify               every page, regenerated and compared
    tb journal show --date D        one page, printed rather than written

    tb backup create                the ledger and everything it names, hashed
    tb backup verify DIR            every file, the chain, and the catalogues
    tb backup restore DIR --to T    rebuild on a clean machine, and prove it
    tb backup receipt FILE          bring that proof home, where the gate reads it
    tb backup list                  the backups made and the restores reported

    tb drill killswitch             engage the switch on a running demo loop
    tb drill watchdog               freeze the loop until the watchdog notices
    tb drill list                   every drill, and whether it passed

    tb arm                          the evidence live trading needs, and the state
    tb arm --live --strategy S      arm real money, once all of it is in the ledger
    tb disarm --reason R            end an arming; a live run halts next cycle

    tb alerts                       what is new since the last pass, delivered
    tb alerts --follow              ...every --interval seconds, beside the loop

The streak `tb sessions` prints is the number `tb arm --live` will count,
computed by the same function from the same ledger, so what an operator reads
here is what the gate will see. The journal is the same record written down:
one page per session, reproducible byte for byte from the ledger it names.
"""

from __future__ import annotations

import os
import sys
from datetime import date, timedelta
from pathlib import Path
from typing import Annotated

import typer
from rich.console import Console
from rich.markup import escape
from rich.table import Table

from tb.broker.t212.client import T212Client
from tb.broker.t212.errors import AuthError, BrokerError
from tb.broker.t212.ratelimit import RateGovernor
from tb.config.loader import PinnedLimits, load_hard_limits
from tb.core.clock import now_utc
from tb.core.errors import TbError
from tb.core.ids import new_run_id
from tb.data.calendar import TradingCalendar
from tb.ledger.anchor import GitAnchorSink, anchor_head
from tb.ledger.store import Ledger, default_ledger_path
from tb.ops.alerts import (
    STATE_FILE,
    WEBHOOK_ENV,
    AlertError,
    ConsoleSink,
    FileSink,
    Severity,
    Sink,
    WebhookSink,
    run_pass,
)
from tb.ops.arming import (
    ArmingError,
    arm_live,
    arming_state,
    disarm,
    live_evidence,
    resolve_strategies,
)
from tb.ops.backup import (
    DEFAULT_DESTINATION,
    RECEIPT,
    BackupError,
    RestoreReceipt,
    backups_made,
    create_backup,
    record_receipt,
    restore_backup,
    verified_restores,
    verify_backup,
)
from tb.ops.drills import DrillError, DrillKind, DrillResult, DrillWorld, drills, run_drill
from tb.ops.journal import (
    DEFAULT_DIR,
    HEADS_FILE,
    JournalError,
    WriteOutcome,
    ensure_pinned,
    latest_final_session,
    render_page,
    verify_page,
    write_page,
)
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
    SessionVerdict.DRILL: "[cyan]drill[/cyan]",
}


def _load(limits: Path | None) -> PinnedLimits:
    try:
        return load_hard_limits(limits)
    except TbError as exc:
        err_console.print(f"{BAD} {escape(str(exc))}", soft_wrap=True)
        raise typer.Exit(2) from exc


def _ledger(db: Path | None, *, writer: PinnedLimits | None = None) -> Ledger:
    """Read-only unless a command must write: a report has no business writing
    the record it reports on."""
    path = db or default_ledger_path()
    if not path.exists():
        err_console.print(f"{BAD} no ledger at {path}. Run `tb init` first.", soft_wrap=True)
        raise typer.Exit(2)
    try:
        if writer is None:
            return Ledger(path, read_only=True).open()
        return Ledger(path, config_hash=writer.config_hash).open()
    except TbError as exc:
        err_console.print(f"{BAD} {escape(str(exc))}", soft_wrap=True)
        raise typer.Exit(2) from exc


def _day(text: str, flag: str) -> date:
    try:
        return date.fromisoformat(text)
    except ValueError as exc:
        err_console.print(f"{BAD} {flag} wants YYYY-MM-DD, got {text!r}.", soft_wrap=True)
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
    wanted = None if day is None else _day(day, "--date")

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


# --------------------------------------------------------------------------
# tb journal
# --------------------------------------------------------------------------

journal_app = typer.Typer(
    help="The daily journal: one markdown page per session, written from the ledger.",
    no_args_is_help=True,
)

DirOpt = Annotated[Path, typer.Option("--dir", help="Where the pages are kept.")]

_ACTION_MARK = {"written": OK, "rewritten": OK, "unchanged": OK, "refused": BAD}


@journal_app.command("write")
def journal_write(
    day: Annotated[
        str | None,
        typer.Option("--date", help="The session to write (YYYY-MM-DD).", show_default=False),
    ] = None,
    since: Annotated[
        str | None,
        typer.Option(
            "--since",
            help="Write every session from this one to the latest finished (YYYY-MM-DD).",
            show_default=False,
        ),
    ] = None,
    directory: DirOpt = DEFAULT_DIR,
    commit: Annotated[
        bool,
        typer.Option(
            "--commit",
            help="Commit the pages to git with the chain head, and record the anchor.",
        ),
    ] = False,
    limits: LimitsOpt = None,
    db: DbOpt = None,
) -> None:
    """Write the latest finished session's page, or the sessions asked for.

    Safe to schedule: a final page already written and still verified is left
    alone, a provisional one is replaced, and one the ledger no longer produces
    is never overwritten — that page is the evidence.
    """
    if day is not None and since is not None:
        err_console.print(f"{BAD} --date and --since are exclusive.", soft_wrap=True)
        raise typer.Exit(2)
    pinned = _load(limits)
    calendar = TradingCalendar()
    moment = now_utc()
    latest = latest_final_session(calendar, now=moment, limits=pinned.limits.live)
    if day is not None:
        days = [_day(day, "--date")]
    elif latest is None:
        err_console.print(f"{BAD} no session has finished yet to write a page for.", soft_wrap=True)
        raise typer.Exit(2)
    elif since is not None:
        start = _day(since, "--since")
        days = (
            [s.day for s in calendar.sessions_between(start, latest.day)]
            if start <= latest.day
            else []
        )
        if not days:
            console.print(f"{WARN} no finished session from {start} on to write.")
    else:
        days = [latest.day]

    outcomes: list[WriteOutcome] = []
    with _ledger(db, writer=pinned) as ledger:
        if ensure_pinned(ledger, pinned):
            console.print(
                f"{OK} pinned the limits {pinned.config_hash[:12]} in the ledger, so the "
                "pages judged by them can be checked against them later"
            )
        for session in days:
            try:
                page = render_page(
                    ledger,
                    session=session,
                    limits=pinned.limits,
                    limits_hash=pinned.config_hash,
                    as_of=moment,
                    calendar=calendar,
                )
            except JournalError as exc:
                err_console.print(f"{BAD} {escape(str(exc))}", soft_wrap=True)
                raise typer.Exit(2) from exc
            outcome = write_page(ledger, page, directory=directory, calendar=calendar)
            outcomes.append(outcome)
            console.print(
                f"{_ACTION_MARK[outcome.action]} {outcome.session} {outcome.action}: "
                f"{outcome.path} ({escape(outcome.detail)})",
                soft_wrap=True,
            )

        changed = [o.path for o in outcomes if o.action in ("written", "rewritten")]
        if commit and changed:
            names = ", ".join(o.session.isoformat() for o in outcomes if o.path in changed)
            sink = GitAnchorSink(
                directory / HEADS_FILE,
                repo_root=directory,
                also=changed,
                message=f"journal: {names}",
            )
            try:
                anchored = anchor_head(ledger, sink)
            except (TbError, OSError) as exc:
                err_console.print(
                    f"{BAD} the pages are written but not committed: {escape(str(exc))}",
                    soft_wrap=True,
                )
                raise typer.Exit(1) from exc
            console.print(
                f"{OK} committed {len(changed)} page(s) with the chain head at seq "
                f"{anchored.seq}: {anchored.external_ref}. It is evidence once pushed."
            )
        elif commit:
            console.print(f"{OK} nothing new to commit.")

    if any(o.action == "refused" for o in outcomes):
        raise typer.Exit(1)


@journal_app.command("verify")
def journal_verify(
    pages: Annotated[
        list[Path] | None,
        typer.Argument(help="Pages to check. Default: every page in --dir.", show_default=False),
    ] = None,
    directory: DirOpt = DEFAULT_DIR,
    db: DbOpt = None,
) -> None:
    """Regenerate each page from the ledger it names, and compare byte for byte."""
    targets = list(pages or sorted(directory.glob("????-??-??.md")))
    if not targets:
        console.print(f"{WARN} no pages in {directory}.")
        return
    failed = 0
    with _ledger(db) as ledger:
        for target in targets:
            try:
                text = target.read_text(encoding="utf-8")
            except OSError as exc:
                failed += 1
                console.print(f"{BAD} {target}: {escape(str(exc))}", soft_wrap=True)
                continue
            check = verify_page(ledger, text)
            if check.ok and check.header is not None:
                console.print(
                    f"{OK} {target}: {check.header.status}, through seq {check.header.through_seq}"
                )
                continue
            failed += 1
            for problem in check.problems:
                console.print(f"{BAD} {target}: {escape(problem)}", soft_wrap=True)
    if failed:
        console.print(f"\n{BAD} {failed} of {len(targets)} page(s) failed verification.")
        raise typer.Exit(1)
    console.print(f"\n{OK} {len(targets)} page(s) are what the ledger produces.")


@journal_app.command("show")
def journal_show(
    day: Annotated[str, typer.Option("--date", help="The session to show (YYYY-MM-DD).")],
    limits: LimitsOpt = None,
    db: DbOpt = None,
) -> None:
    """Print a session's page without writing it."""
    session = _day(day, "--date")
    pinned = _load(limits)
    with _ledger(db) as ledger:
        try:
            page = render_page(
                ledger,
                session=session,
                limits=pinned.limits,
                limits_hash=pinned.config_hash,
                as_of=now_utc(),
            )
        except JournalError as exc:
            err_console.print(f"{BAD} {escape(str(exc))}", soft_wrap=True)
            raise typer.Exit(2) from exc
    typer.echo(page.text, nl=False)


# --------------------------------------------------------------------------
# tb backup
# --------------------------------------------------------------------------

backup_app = typer.Typer(
    help="Back up the ledger and what it names, restore it elsewhere, and prove it.",
    no_args_is_help=True,
)


@backup_app.command("create")
def backup_create(
    destination: Annotated[
        Path, typer.Option("--to", help="Where backups are kept. Never inside the repo.")
    ] = DEFAULT_DESTINATION,
    bars: Annotated[
        Path | None,
        typer.Option(
            "--bars", help="The bar store. Default: beside the ledger.", show_default=False
        ),
    ] = None,
    models: Annotated[
        Path | None,
        typer.Option(
            "--models", help="The model store. Default: beside the ledger.", show_default=False
        ),
    ] = None,
    limits: LimitsOpt = None,
    db: DbOpt = None,
) -> None:
    """Copy the ledger and every file it names, hash it all, and record the backup."""
    pinned = _load(limits)
    with _ledger(db, writer=pinned) as ledger:
        beside = Path(ledger.path).parent
        try:
            result = create_backup(
                ledger,
                destination=destination,
                bars_root=bars or beside / "bars",
                models_root=models or beside / "models",
                limits=pinned,
            )
        except (BackupError, OSError) as exc:
            err_console.print(f"{BAD} {escape(str(exc))}", soft_wrap=True)
            raise typer.Exit(1) from exc
    manifest = result.manifest
    kinds = ", ".join(
        f"{sum(1 for f in manifest.files if f.kind == kind)} {kind}(s)"
        for kind in ("ledger", "partition", "model", "limits")
    )
    console.print(
        f"{OK} backup {result.backup_id} at {result.path}: {kinds}, "
        f"{manifest.total_bytes:,} bytes, through seq {manifest.head_seq}"
    )
    console.print(
        f"  manifest {result.manifest_sha256}. Copy the directory to another machine and run "
        "`tb backup restore` there; it holds account data, so keep it off anything public."
    )


@backup_app.command("verify")
def backup_verify(
    path: Annotated[Path, typer.Argument(help="The backup directory.")],
) -> None:
    """Check a backup: every file, the chain, and nothing the ledger names missing."""
    check = verify_backup(path)
    if not check.ok:
        for problem in check.problems:
            console.print(f"{BAD} {escape(problem)}", soft_wrap=True)
        raise typer.Exit(1)
    assert check.manifest is not None
    console.print(f"{OK} backup {check.manifest.backup_id} is whole:")
    for line in check.checks:
        console.print(f"  {OK} {escape(line)}")


@backup_app.command("restore")
def backup_restore(
    path: Annotated[Path, typer.Argument(help="The backup directory.")],
    target: Annotated[Path, typer.Option("--to", help="An empty directory to restore into.")],
) -> None:
    """Rebuild a ledger and its stores in an empty directory, check it, and write a receipt."""
    try:
        receipt = restore_backup(path, target)
    except (BackupError, OSError) as exc:
        err_console.print(f"{BAD} {escape(str(exc))}", soft_wrap=True)
        raise typer.Exit(1) from exc
    console.print(f"{OK} restored {receipt.backup_id} into {target} on {receipt.restored_host}:")
    for line in receipt.checks:
        console.print(f"  {OK} {escape(line)}")
    console.print(
        f"\n  The receipt is {target / RECEIPT}. Take it back to the machine the backup came "
        "from and run `tb backup receipt` there: that ledger is the one the live gate reads."
    )


@backup_app.command("receipt")
def backup_receipt(
    path: Annotated[Path, typer.Argument(help="The restore-receipt.json a restore wrote.")],
    limits: LimitsOpt = None,
    db: DbOpt = None,
) -> None:
    """Record a restore done elsewhere, if this ledger made the backup it restored."""
    try:
        receipt = RestoreReceipt.parse(path.read_text(encoding="utf-8"))
    except (BackupError, OSError) as exc:
        err_console.print(f"{BAD} {escape(str(exc))}", soft_wrap=True)
        raise typer.Exit(2) from exc
    pinned = _load(limits)
    with _ledger(db, writer=pinned) as ledger:
        try:
            recorded = record_receipt(ledger, receipt)
        except BackupError as exc:
            err_console.print(f"{BAD} {escape(str(exc))}", soft_wrap=True)
            raise typer.Exit(1) from exc
    console.print(
        f"{OK} recorded the restore of {recorded.backup_id} on {recorded.restored_host} "
        f"at {recorded.restored_at.isoformat()} (seq {recorded.seq})"
    )
    if recorded.same_host:
        console.print(
            f"{WARN} it was restored on the machine that made it. That proves the files are "
            "intact, not that the state survives losing this machine, so the live gate does "
            "not count it. Restore on another machine."
        )


@backup_app.command("list")
def backup_list(db: DbOpt = None) -> None:
    """The backups this ledger made, and the restores reported back to it."""
    with _ledger(db) as ledger:
        made = backups_made(ledger)
        restores = verified_restores(ledger)
    if not made:
        console.print(f"{WARN} no backups recorded. `tb backup create` makes one.")
        return
    # Identifiers and hosts go on their own lines, never into a wrapped cell:
    # a host name broken across two rows is one nobody can read back, and a
    # backup id broken across two is one nobody can paste into `verify`.
    table = Table("backup", "made", "through seq", "files", "bytes", "restores", title="backups")
    for column in table.columns:
        column.no_wrap = True
    for backup in made:
        table.add_row(
            str(backup["backup_id"]),
            f"{backup['at']:%Y-%m-%d %H:%M}",
            str(backup["head_seq"]),
            str(backup["n_files"]),
            f"{int(backup['n_bytes']):,}",
            str(sum(1 for r in restores if r.backup_id == backup["backup_id"])),
        )
    console.print(table)
    if not restores:
        console.print(
            f"{WARN} no restore reported back yet. Restore on another machine, then run "
            "`tb backup receipt` here."
        )
    for restore in restores:
        counted = (
            "same host, so the live gate does not count it"
            if restore.same_host
            else "on another machine"
        )
        console.print(
            f"  {WARN if restore.same_host else OK} {restore.backup_id} restored on "
            f"{escape(restore.restored_host)} at {restore.restored_at:%Y-%m-%d %H:%M}: "
            f"{counted}",
            soft_wrap=True,
        )


# --------------------------------------------------------------------------
# tb drill
# --------------------------------------------------------------------------

drill_app = typer.Typer(
    help="Fire the kill switch or the watchdog on purpose, and record what happened.",
    no_args_is_help=True,
)

HaltWithinOpt = Annotated[
    int,
    typer.Option(
        "--halt-within",
        min=10,
        help="Seconds the loop has to stop once the switch is engaged.",
    ),
]


def _drill(
    kind: DrillKind,
    *,
    limits: Path | None,
    db: Path | None,
    halt_within: int,
    notice_within: int | None,
) -> None:
    pinned = _load(limits)
    safety = pinned.limits.safety
    with _ledger(db, writer=pinned) as ledger:
        state_dir = Path(safety.kill_switch_path).parent
        try:
            client = T212Client.from_env(
                governor=RateGovernor(state_path=state_dir / "ratelimit.json"),
                ledger=ledger,
                run_id=new_run_id(),
                require_demo=True,
            )
        except AuthError as exc:
            err_console.print(
                f"{BAD} {escape(str(exc))}. A drill reads positions and stops from the "
                "demo account, so it needs T212_DEMO_API_KEY.",
                soft_wrap=True,
            )
            raise typer.Exit(2) from exc
        world = DrillWorld(
            ledger=ledger,
            broker=client,
            kill_switch_path=Path(safety.kill_switch_path),
            heartbeat_path=Path(safety.heartbeat_path),
        )
        try:
            result = run_drill(
                world,
                kind,
                halt_within=timedelta(seconds=halt_within),
                notice_within=timedelta(
                    seconds=notice_within or safety.heartbeat_stale_seconds + 120
                ),
                cycling_within=timedelta(seconds=pinned.limits.live.session_max_cycle_gap_seconds),
            )
        except DrillError as exc:
            err_console.print(f"{BAD} {escape(str(exc))}", soft_wrap=True)
            raise typer.Exit(2) from exc
        except BrokerError as exc:
            err_console.print(
                f"{BAD} the demo account could not be read: {escape(str(exc))}", soft_wrap=True
            )
            raise typer.Exit(2) from exc
        finally:
            client.close()
    _print_drill(result)


def _print_drill(result: DrillResult) -> None:
    marker = OK if result.passed else BAD
    console.print(
        f"{marker} {result.kind.value} drill {result.drill_id} on run {result.run_id}: "
        + ("passed" if result.passed else "failed")
    )
    for line in result.observations:
        console.print(f"  {OK} {escape(line)}", soft_wrap=True)
    for line in result.failures:
        console.print(f"  {BAD} {escape(line)}", soft_wrap=True)
    console.print(
        "\nThe kill switch is engaged and the loop has stopped. Resuming is your call: "
        "`tb resume --reason '...'`, then `tb run --mode demo`."
    )
    if not result.passed:
        raise typer.Exit(1)


@drill_app.command("killswitch")
def drill_killswitch(
    halt_within: HaltWithinOpt = 300,
    limits: LimitsOpt = None,
    db: DbOpt = None,
) -> None:
    """Engage the kill switch on the running demo loop, with positions open.

    Passes when the loop halts, sends nothing after the switch, and every
    position still has its stop at the broker.
    """
    _drill(DrillKind.KILL_SWITCH, limits=limits, db=db, halt_within=halt_within, notice_within=None)


@drill_app.command("watchdog")
def drill_watchdog(
    halt_within: HaltWithinOpt = 300,
    notice_within: Annotated[
        int | None,
        typer.Option(
            "--notice-within",
            min=10,
            help="Seconds the watchdog has to notice. Default: its stale bound plus two minutes.",
            show_default=False,
        ),
    ] = None,
    limits: LimitsOpt = None,
    db: DbOpt = None,
) -> None:
    """Freeze the running demo loop until the watchdog notices, then thaw it.

    Needs `tb watchdog` running, and runs on the loop's own host. Passes when
    the watchdog engages the switch, the thawed loop halts on it, nothing is
    sent after the freeze, and every position still has its stop.
    """
    _drill(
        DrillKind.WATCHDOG,
        limits=limits,
        db=db,
        halt_within=halt_within,
        notice_within=notice_within,
    )


@drill_app.command("list")
def drill_list(db: DbOpt = None) -> None:
    """Every drill in the ledger, and whether it passed."""
    with _ledger(db) as ledger:
        records = drills(ledger)
    if not records:
        console.print(f"{WARN} no drills recorded. `tb drill killswitch` runs one.")
        return
    table = Table("drill", "kind", "started", "run", "mode", "held", "outcome", title="drills")
    for record in records:
        outcome = (
            "[green]passed[/green]"
            if record.passed
            else "[yellow]never completed[/yellow]"
            if record.completed_at is None
            else "[red]failed[/red]: " + escape("; ".join(record.failures))
        )
        table.add_row(
            record.drill_id,
            record.kind,
            f"{record.started_at:%Y-%m-%d %H:%M}",
            record.run_id,
            record.mode,
            str(record.n_holdings),
            outcome,
        )
    console.print(table)


# --------------------------------------------------------------------------
# tb arm / tb disarm
# --------------------------------------------------------------------------

CONFIRMATION = "arm live"


def _who() -> str:
    return os.environ.get("USER") or f"uid:{os.getuid()}"


def arm_command(
    live: Annotated[
        bool,
        typer.Option("--live", help="Arm real-money trading. Without it, only report."),
    ] = False,
    strategies: Annotated[
        list[str] | None,
        typer.Option(
            "--strategy",
            help="A promoted strategy to arm, as id or id@vN. Repeat for more, within the cap.",
            show_default=False,
        ),
    ] = None,
    limits: LimitsOpt = None,
    db: DbOpt = None,
) -> None:
    """The evidence live trading needs; with --live, arm it after confirmation.

    Every requirement is read from the ledger: the clean demo streak and the
    trades closed across it, both drills, a restore verified on another
    machine, an intact chain, and the limits file's own `live.enabled`.
    Arming asks you to type the phrase it names, and lapses on its own.
    """
    pinned = _load(limits)
    live_limits = pinned.limits.live
    with _ledger(db, writer=pinned if live else None) as ledger:
        evidence = live_evidence(ledger, limits=pinned.limits)
        state = arming_state(ledger, config_hash=pinned.config_hash)

        table = Table("requirement", "observed", "required", "", title="live trading needs")
        for requirement in evidence.requirements:
            table.add_row(
                requirement.name,
                escape(requirement.observed),
                escape(requirement.required),
                OK if requirement.met else BAD,
            )
        console.print(table)
        for note in evidence.notes:
            console.print(f"{WARN} {escape(note)}", soft_wrap=True)
        if state.arming is not None:
            console.print(
                f"{OK} armed: {', '.join(state.arming.strategies)}, {escape(state.reason)} "
                f"(arming {state.arming.arming_id} by {state.arming.armed_by})"
            )
        else:
            console.print(f"{WARN} not armed: {escape(state.reason)}", soft_wrap=True)

        if not live:
            if not evidence.ready:
                raise typer.Exit(1)
            return

        if not evidence.ready:
            err_console.print(
                f"\n{BAD} not armed: {len(evidence.unmet)} requirement(s) unmet.",
                soft_wrap=True,
            )
            raise typer.Exit(1)
        try:
            resolved = resolve_strategies(ledger, strategies or [], limits=pinned.limits)
        except ArmingError as exc:
            err_console.print(f"{BAD} {escape(str(exc))}", soft_wrap=True)
            raise typer.Exit(2) from exc

        capital = pinned.limits.capital
        console.print(
            f"\n[bold]Real money.[/bold] Trading 212 live, with {', '.join(resolved)} at no "
            f"rung above {live_limits.max_rung} (floor {capital.floor_notional_ccy} "
            f"{pinned.limits.currency} a position), never more than "
            f"{capital.absolute_ceiling_ccy} {pinned.limits.currency} in all, for "
            f"{live_limits.arming_valid_days} day(s) unless disarmed sooner. Every other "
            "limit in the file still binds."
        )
        typed = typer.prompt(f"Type '{CONFIRMATION}' to arm, anything else to stop")
        if typed.strip() != CONFIRMATION:
            console.print(f"{WARN} not armed.")
            raise typer.Exit(1)
        try:
            arming = arm_live(ledger, pinned=pinned, strategies=resolved, armed_by=_who())
        except ArmingError as exc:
            err_console.print(f"{BAD} {escape(str(exc))}", soft_wrap=True)
            raise typer.Exit(1) from exc
    console.print(
        f"{OK} armed {', '.join(arming.strategies)} until "
        f"{arming.expires_at:%Y-%m-%d %H:%M} UTC (arming {arming.arming_id}). Start it with "
        "`tb run --mode live --cycles 0` beside `tb watchdog`; `tb disarm` ends it."
    )


def disarm_command(
    reason: Annotated[str, typer.Option("--reason", "-r", help="Why live trading stops.")],
    limits: LimitsOpt = None,
    db: DbOpt = None,
) -> None:
    """End the live arming. A running live loop halts at its next cycle."""
    pinned = _load(limits)
    with _ledger(db, writer=pinned) as ledger:
        ended = disarm(ledger, disarmed_by=_who(), reason=reason)
    if ended is None:
        console.print(f"{OK} nothing was armed; the disarm is recorded anyway.")
    else:
        console.print(
            f"{OK} disarmed {ended}. A running live loop halts at its next cycle; its stops "
            "stay at the broker."
        )


# --------------------------------------------------------------------------
# tb alerts
# --------------------------------------------------------------------------


def alerts_command(
    follow: Annotated[
        bool, typer.Option("--follow", help="Keep going, one pass every --interval seconds.")
    ] = False,
    interval: Annotated[
        float, typer.Option("--interval", min=1.0, help="Seconds between passes.")
    ] = 30.0,
    sinks: Annotated[
        list[str] | None,
        typer.Option(
            "--sink",
            help=f"console, file or webhook (URL from {WEBHOOK_ENV}). Repeatable.",
            show_default=False,
        ),
    ] = None,
    file: Annotated[
        Path | None,
        typer.Option("--file", help="Where the file sink appends. Default: beside the state."),
    ] = None,
    min_severity: Annotated[
        str, typer.Option("--min-severity", help="info, warning or critical.")
    ] = "warning",
    passes: Annotated[
        int, typer.Option("--passes", min=0, help="Stop after this many passes. 0: never.")
    ] = 0,
    limits: LimitsOpt = None,
    db: DbOpt = None,
) -> None:
    """Deliver what the ledger says a person must act on: halts, faults, drills, arming.

    Read-only against the ledger. Its cursor lives beside the kill switch, and
    a first run starts from now: history is not news.
    """
    try:
        floor = Severity(min_severity)
    except ValueError as exc:
        err_console.print(f"{BAD} --min-severity is info, warning or critical.", soft_wrap=True)
        raise typer.Exit(2) from exc
    pinned = _load(limits)
    state_dir = Path(pinned.limits.safety.kill_switch_path).parent
    chosen = list(dict.fromkeys(sinks or ["console"]))
    built: list[Sink] = []
    for name in chosen:
        if name == "console":
            built.append(
                ConsoleSink(write=lambda line: console.print(escape(line), soft_wrap=True))
            )
        elif name == "file":
            built.append(FileSink(path=file or state_dir / "alerts.jsonl"))
        elif name == "webhook":
            from tb.core.http import HttpxTransport

            try:
                built.append(WebhookSink.from_env(HttpxTransport()))
            except AlertError as exc:
                err_console.print(f"{BAD} {escape(str(exc))}", soft_wrap=True)
                raise typer.Exit(2) from exc
        else:
            err_console.print(
                f"{BAD} unknown sink {name!r}: console, file or webhook.", soft_wrap=True
            )
            raise typer.Exit(2)

    import time

    done = 0
    with _ledger(db) as ledger:
        while True:
            try:
                result = run_pass(
                    ledger,
                    state_path=state_dir / STATE_FILE,
                    sinks=built,
                    limits=pinned.limits.live,
                    min_severity=floor,
                )
            except AlertError as exc:
                # Not delivered, so not marked delivered: the next pass offers
                # it again. Said on stderr, where the operator's service
                # manager keeps what went wrong.
                err_console.print(f"{BAD} {escape(str(exc))}; retrying next pass", soft_wrap=True)
                if not follow:
                    raise typer.Exit(1) from exc
            else:
                if result.started_fresh:
                    console.print(
                        f"{OK} alerting from seq {result.cursor} on; what came before is "
                        "history, not news."
                    )
                elif not follow:
                    console.print(
                        f"{OK} {len(result.delivered)} alert(s) delivered, "
                        f"{result.suppressed} repeat(s) held back, read to seq {result.cursor}."
                    )
            done += 1
            if not follow or (passes and done >= passes):
                return
            time.sleep(interval)
