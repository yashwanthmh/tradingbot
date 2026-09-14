"""The `tb` command line.

M0 surface: initialise, inspect, halt, resume, verify and anchor. Trading
commands arrive in M4 — and only after the crash drills pass.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import Annotated

import typer
from rich.console import Console
from rich.markup import escape
from rich.panel import Panel
from rich.table import Table

from tb import __version__
from tb.config.loader import PinnedLimits, load_hard_limits
from tb.core.errors import TbError
from tb.core.ids import new_run_id
from tb.data.providers.alpaca import KEY_ID_VAR as ALPACA_KEY_ID_VAR
from tb.data.providers.alpaca import SECRET_VAR as ALPACA_SECRET_VAR
from tb.data.providers.alpaca import AlpacaProvider
from tb.ledger.anchor import FileAnchorSink, GitAnchorSink, anchor_head
from tb.ledger.events import Actor
from tb.ledger.store import Ledger, default_ledger_path
from tb.ledger.verify import verify_chain
from tb.ops.killswitch import (
    KillSwitchState,
    engage_kill_switch,
    read_heartbeat,
    read_kill_switch,
    release_kill_switch,
)
from tb.ops.secrets import Severity, inspect_secrets, redact
from tb.ops.state import RunState, StateMachine

app = typer.Typer(
    name="tb",
    help="Autonomous trading bot: control plane and audit tools.",
    no_args_is_help=True,
    add_completion=False,
)
ledger_app = typer.Typer(help="Inspect and verify the audit ledger.", no_args_is_help=True)
app.add_typer(ledger_app, name="ledger")

# M1: the broker adapter, symbol map and reconciler.
from tb.cli_broker import broker_app, reconcile_command, symbols_app  # noqa: E402

app.add_typer(broker_app, name="broker")
app.add_typer(symbols_app, name="symbols")

# M2: the point-in-time data layer.
from tb.cli_data import data_app, universe_app  # noqa: E402

app.add_typer(data_app, name="data")
app.add_typer(universe_app, name="universe")


@app.command("reconcile")
def reconcile(
    limits: Annotated[Path | None, typer.Option("--limits", show_default=False)] = None,
    db: Annotated[Path | None, typer.Option("--db", show_default=False)] = None,
) -> None:
    """Establish what the account actually holds, and report what disagrees.

    Read-only in M1: it finds every discrepancy and can repair none of them.
    Repair arrives in M4, behind the risk engine.
    """
    reconcile_command(limits=limits, db=db, dry_run=True)


# Rich falls back to 80 columns when output is not a terminal, which mangles the
# things this CLI exists to print: absolute paths, hashes, dotted event names.
# Use the real terminal width when there is one, and something roomier when the
# output is a pipe or a file.
_WIDTH: int | None = None if sys.stdout.isatty() else int(os.environ.get("COLUMNS") or 120)

console = Console(width=_WIDTH)
err_console = Console(stderr=True, width=_WIDTH)

OK = "[green]✓[/green]"
WARN = "[yellow]![/yellow]"
BAD = "[red]✗[/red]"


def _fail(message: str) -> None:
    """Print an error without letting Rich break a path across lines.

    `soft_wrap` leaves wrapping to the terminal, so ``Run `tb init``` stays
    greppable and a long path stays copy-pasteable.
    """
    err_console.print(f"{BAD} {escape(message)}", soft_wrap=True)


# --------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------


def _load_pinned(limits_path: Path | None) -> PinnedLimits:
    try:
        return load_hard_limits(limits_path)
    except TbError as exc:
        _fail(str(exc))
        raise typer.Exit(2) from exc


def _open_ledger(db: Path | None, pinned: PinnedLimits, *, read_only: bool = False) -> Ledger:
    path = db or default_ledger_path()
    if read_only and not path.exists():
        _fail(f"no ledger at {path}. Run `tb init` first.")
        raise typer.Exit(2)
    ledger = Ledger(path, config_hash=pinned.config_hash, read_only=read_only)
    try:
        return ledger.open()
    except TbError as exc:
        _fail(str(exc))
        raise typer.Exit(2) from exc


LimitsOpt = Annotated[
    Path | None,
    typer.Option("--limits", help="Path to hard_limits.yaml.", show_default=False),
]
DbOpt = Annotated[
    Path | None,
    typer.Option("--db", help="Path to the ledger database.", show_default=False),
]


# --------------------------------------------------------------------------
# top-level commands
# --------------------------------------------------------------------------


@app.command()
def version() -> None:
    """Print the version."""
    console.print(f"tradingbot {__version__}")


@app.command()
def init(limits: LimitsOpt = None, db: DbOpt = None) -> None:
    """Create the runtime directories and the ledger, and pin the hard limits."""
    pinned = _load_pinned(limits)

    kill_dir = Path(pinned.limits.safety.kill_switch_path).parent
    kill_dir.mkdir(parents=True, exist_ok=True)
    console.print(f"{OK} runtime directory {kill_dir}")

    with _open_ledger(db, pinned) as ledger:
        genesis = ledger.initialise(created_by="tb init")
        if genesis is not None:
            console.print(f"{OK} ledger created at {ledger.path}, genesis at seq={genesis.seq}")
        else:
            console.print(f"{OK} ledger already exists at {ledger.path}")

        ledger.record_config_pin(pinned.audit_record())
        console.print(f"{OK} hard limits pinned ({pinned.config_hash[:12]}…)")

        if not pinned.immutability.enforced:
            console.print(f"{WARN} {pinned.immutability.explanation}")

        machine = StateMachine(ledger, pinned)
        if machine.current().state is RunState.BOOT:
            console.print(f"{OK} run state: {machine.current().state.value}")

    console.print("\nNext: [bold]tb doctor[/bold] to check the environment.")


@app.command()
def doctor(limits: LimitsOpt = None, db: DbOpt = None) -> None:
    """Check the environment before letting anything trade.

    Exits non-zero if anything would make trading unsafe.
    """
    problems = 0
    warnings = 0

    console.print(Panel.fit("[bold]Control layer[/bold]", border_style="dim"))
    pinned = _load_pinned(limits)
    lim = pinned.limits
    console.print(f"{OK} limits valid at {pinned.source_path} ({pinned.config_hash[:12]}…)")
    console.print(
        f"  currency {lim.currency}, absolute ceiling {lim.capital.absolute_ceiling_ccy}, "
        f"floor order {lim.capital.floor_notional_ccy}"
    )
    console.print(
        f"  worst-case unprotected gap loss {lim.worst_case_unprotected_loss_pct:.2f}% "
        f"vs daily halt {lim.loss.daily_halt_pct:.2f}%"
    )
    if pinned.immutability.enforced:
        console.print(f"{OK} limits are immutable to this process")
    else:
        console.print(f"{WARN} {pinned.immutability.explanation}")
        warnings += 1

    console.print(Panel.fit("[bold]Credentials[/bold]", border_style="dim"))
    secrets = inspect_secrets()
    marker = {Severity.OK: OK, Severity.WARN: WARN, Severity.FAIL: BAD}[secrets.severity]
    console.print(f"{marker} broker environment: [bold]{secrets.environment.value}[/bold]")
    if secrets.environment.base_url:
        console.print(f"  base URL {secrets.environment.base_url}")
    console.print(f"  demo key {redact(os.environ.get('T212_DEMO_API_KEY'))}")
    console.print(f"  live key {redact(os.environ.get('T212_LIVE_API_KEY'))}")
    for finding in secrets.findings:
        console.print(
            f"  {WARN if secrets.severity is not Severity.FAIL else BAD} {escape(finding)}"
        )
    if secrets.severity is Severity.FAIL:
        problems += 1
    elif secrets.findings:
        warnings += 1

    # The market-data feed, reported here and never fatal: Yahoo-only is a
    # supported way to run. What this catches is a key exported under a name
    # nothing reads, which otherwise stays silent until a backfill fails
    # complaining about a missing provider.
    data_findings = AlpacaProvider.credential_findings()
    configured = AlpacaProvider.configured()
    console.print(
        f"{OK if configured else WARN} market data: "
        f"[bold]{'alpaca + yahoo' if configured else 'yahoo only'}[/bold]"
    )
    console.print(f"  alpaca key {redact(os.environ.get(ALPACA_KEY_ID_VAR))}")
    console.print(f"  alpaca secret {redact(os.environ.get(ALPACA_SECRET_VAR))}")
    for finding in data_findings:
        console.print(f"  {WARN} {escape(finding)}")
    if data_findings:
        warnings += 1

    console.print(Panel.fit("[bold]Safety[/bold]", border_style="dim"))
    switch = read_kill_switch(lim.safety.kill_switch_path)
    if switch.state is KillSwitchState.CLEAR:
        console.print(f"{OK} kill switch clear ({switch.path})")
    elif switch.state is KillSwitchState.ENGAGED:
        console.print(f"{WARN} kill switch ENGAGED: {escape(switch.reason or switch.detail)}")
        warnings += 1
    else:
        console.print(f"{BAD} kill switch undeterminable — treated as engaged")
        console.print(f"  {switch.detail}")
        problems += 1

    beat = read_heartbeat(
        lim.safety.heartbeat_path,
        stale_after_seconds=lim.safety.heartbeat_stale_seconds,
    )
    console.print(f"{OK if not beat.stale else WARN} {beat.detail}")

    console.print(Panel.fit("[bold]Ledger[/bold]", border_style="dim"))
    path = db or default_ledger_path()
    if not path.exists():
        console.print(f"{BAD} no ledger at {path}. Run `tb init`.")
        problems += 1
    else:
        with _open_ledger(db, pinned, read_only=True) as ledger:
            head = ledger.head()
            console.print(f"{OK} ledger at {path}: {ledger.count()} events")
            if head:
                console.print(f"  head seq={head.seq} {head.chain_hash[:12]}…")
            report = verify_chain(ledger)
            if report.ok:
                console.print(f"{OK} {report.summary()}")
            else:
                console.print(f"{BAD} {report.summary()}")
                problems += 1
            if not ledger.anchors():
                console.print(
                    f"{WARN} chain head has never been anchored. Until it is published "
                    "outside this database, a rewrite of history would still verify."
                )
                warnings += 1

    console.print()
    if problems:
        err_console.print(f"[red bold]{problems} problem(s), {warnings} warning(s).[/red bold]")
        raise typer.Exit(1)
    console.print(f"[green bold]No problems. {warnings} warning(s).[/green bold]")


@app.command()
def status(limits: LimitsOpt = None, db: DbOpt = None) -> None:
    """Show run state, kill switch, open halts and the ledger head."""
    pinned = _load_pinned(limits)
    with _open_ledger(db, pinned, read_only=True) as ledger:
        machine = StateMachine(ledger, pinned)
        permission = machine.check_trading_permission()
        reading = machine.current()

        table = Table(show_header=False, box=None, padding=(0, 2, 0, 0))
        table.add_row("run state", f"[bold]{reading.state.value}[/bold]  since {reading.since}")
        table.add_row("may trade", "[green]yes[/green]" if permission.allowed else "[red]no[/red]")
        table.add_row("kill switch", permission.kill_switch.value)
        table.add_row("open halts", str(len(permission.open_halts)))
        head = ledger.head()
        table.add_row(
            "ledger",
            f"{ledger.count()} events"
            + (f", head seq={head.seq} {head.chain_hash[:12]}…" if head else " (empty)"),
        )
        anchors = ledger.anchors()
        table.add_row(
            "last anchor",
            f"seq={anchors[-1]['seq']} via {anchors[-1]['sink']}"
            if anchors
            else "[yellow]never[/yellow]",
        )
        table.add_row("limits", f"{pinned.config_hash[:12]}… at {pinned.source_path}")
        console.print(table)

        if not permission.allowed:
            console.print("\n[bold]Blocked because:[/bold]")
            for reason in permission.reasons:
                # escape(): these strings carry operator-supplied text and
                # trigger names in square brackets, which Rich would otherwise
                # read as markup tags and silently swallow.
                console.print(f"  {BAD} {escape(reason)}")

        for halt in permission.open_halts:
            console.print(
                f"\n[yellow]halt[/yellow] {halt.halt_id} "
                f"{escape(f'[{halt.trigger}]')} raised {halt.raised_at}"
                f"\n  {escape(halt.detail or '')}"
            )


@app.command()
def halt(
    reason: Annotated[str, typer.Option("--reason", "-r", help="Why you are stopping it.")],
    limits: LimitsOpt = None,
    db: DbOpt = None,
) -> None:
    """Stop trading now: engage the kill switch and record a halt."""
    pinned = _load_pinned(limits)
    who = os.environ.get("USER") or f"uid:{os.getuid()}"

    switch = engage_kill_switch(
        pinned.limits.safety.kill_switch_path, engaged_by=who, reason=reason
    )
    console.print(f"{OK} kill switch engaged at {switch.path}")

    with _open_ledger(db, pinned) as ledger:
        machine = StateMachine(ledger, pinned, run_id=new_run_id())
        machine.record_kill_switch(
            engaged=True,
            path=str(switch.path),
            determinable=switch.determinable,
            detail=switch.detail,
            engaged_by=who,
        )
        halt_id = machine.raise_halt("manual", reason, actor=Actor.HUMAN)
        console.print(f"{OK} halt recorded: {halt_id}")
        console.print(f"{OK} run state: {machine.current().state.value}")


@app.command()
def resume(
    reason: Annotated[str, typer.Option("--reason", "-r", help="Why it is safe to resume.")],
    limits: LimitsOpt = None,
    db: DbOpt = None,
    force: Annotated[
        bool,
        typer.Option("--force", help="Clear halts raised by safety checks, not just manual ones."),
    ] = False,
) -> None:
    """Release the kill switch and clear halts.

    This does not resume trading directly — it moves to RECONCILING. The
    account may have changed while we were stopped, so state is re-established
    against the broker before any order can be placed.
    """
    pinned = _load_pinned(limits)
    who = os.environ.get("USER") or f"uid:{os.getuid()}"

    with _open_ledger(db, pinned) as ledger:
        machine = StateMachine(ledger, pinned, run_id=new_run_id())
        open_halts = machine.open_halts()

        automatic = [h for h in open_halts if h.trigger != "manual"]
        if automatic and not force:
            _fail(f"{len(automatic)} halt(s) were raised by safety checks, not by hand:")
            for h in automatic:
                err_console.print(f"    [{h.trigger}] {h.detail}")
            err_console.print(
                "\nThese fired for a reason. Investigate first; pass --force to clear them."
            )
            raise typer.Exit(1)

        for h in open_halts:
            machine.clear_halt(h.halt_id, cleared_by=who, clear_reason=reason)
            console.print(f"{OK} cleared halt {h.halt_id} [{h.trigger}]")

        switch = release_kill_switch(pinned.limits.safety.kill_switch_path)
        if switch.state is KillSwitchState.CLEAR:
            console.print(f"{OK} kill switch released")
            machine.record_kill_switch(
                engaged=False,
                path=str(switch.path),
                determinable=switch.determinable,
                detail=switch.detail,
                engaged_by=who,
            )
        else:
            _fail(f"kill switch is still not clear: {switch.detail}")
            raise typer.Exit(1)

        if machine.current().state is not RunState.RECONCILING:
            machine.transition_to(
                RunState.RECONCILING, reason=f"resume: {reason}", actor=Actor.HUMAN
            )
        console.print(f"{OK} run state: {machine.current().state.value}")
        console.print(
            "\nState will be re-established against the broker before any order is placed."
        )


@app.command("config")
def config_show(
    limits: LimitsOpt = None,
    as_json: Annotated[bool, typer.Option("--json", help="Emit the values as JSON.")] = False,
) -> None:
    """Show the hard limits in force and their hashes."""
    pinned = _load_pinned(limits)
    if as_json:
        console.print_json(
            json.dumps(
                {
                    "config_hash": pinned.content_sha256,
                    "canonical_hash": pinned.canonical_sha256,
                    "source_path": str(pinned.source_path),
                    "immutability_enforced": pinned.immutability.enforced,
                    "values": pinned.limits.model_dump(mode="json"),
                }
            )
        )
        return

    console.print(f"[bold]{pinned.source_path}[/bold]")
    console.print(f"  content hash   {pinned.content_sha256}")
    console.print(f"  canonical hash {pinned.canonical_sha256}")
    console.print(
        f"  immutable      {pinned.immutability.enforced} — "
        f"{escape(pinned.immutability.explanation)}\n"
    )

    for section, values in pinned.limits.model_dump(mode="json").items():
        if not isinstance(values, dict):
            console.print(f"[dim]{section}[/dim] = {values}")
            continue
        table = Table(title=section, show_header=False, box=None, title_justify="left")
        for key, value in values.items():
            table.add_row(key, str(value))
        console.print(table)


# --------------------------------------------------------------------------
# tb ledger ...
# --------------------------------------------------------------------------


@ledger_app.command("verify")
def ledger_verify(
    limits: LimitsOpt = None,
    db: DbOpt = None,
    quick: Annotated[
        bool,
        typer.Option(
            "--quick", help="Stop at the first finding instead of walking the whole chain."
        ),
    ] = False,
) -> None:
    """Verify the hash chain and check it against every published anchor."""
    pinned = _load_pinned(limits)
    with _open_ledger(db, pinned, read_only=True) as ledger:
        report = verify_chain(ledger, stop_on_first=quick)
        if report.ok:
            console.print(f"{OK} {report.summary()}")
            if report.anchors_checked == 0:
                console.print(
                    f"{WARN} no anchors exist, so this check cannot detect a full "
                    "rewrite-and-re-sign of history. Run `tb ledger anchor`."
                )
            return

        _fail(f"{report.summary()}\n")
        table = Table("seq", "kind", "detail")
        for finding in sorted(report.findings, key=lambda f: f.seq):
            table.add_row(str(finding.seq), finding.kind.value, finding.detail)
        err_console.print(table)
        raise typer.Exit(1)


@ledger_app.command("anchor")
def ledger_anchor(
    limits: LimitsOpt = None,
    db: DbOpt = None,
    sink: Annotated[str, typer.Option("--sink", help="Where to publish: file or git.")] = "file",
    path: Annotated[Path, typer.Option("--path", help="Anchor file to append to.")] = Path(
        "journal/chain-heads.jsonl"
    ),
) -> None:
    """Publish the chain head outside the ledger.

    Without this, a process that can rewrite the ledger can also recompute every
    hash, producing a chain that verifies perfectly and is entirely fictional.
    """
    pinned = _load_pinned(limits)
    chosen = GitAnchorSink(path) if sink == "git" else FileAnchorSink(path)

    with _open_ledger(db, pinned) as ledger:
        try:
            result = anchor_head(ledger, chosen)
        except TbError as exc:
            _fail(str(exc))
            raise typer.Exit(1) from exc

    console.print(f"{OK} anchored seq={result.seq} {result.chain_hash[:12]}… via {result.sink}")
    if result.external_ref:
        console.print(f"  reference {result.external_ref}")
    marker = OK if result.crosses_trust_boundary else WARN
    console.print(f"{marker} {result.trust_boundary}")


@ledger_app.command("tail")
def ledger_tail(
    limits: LimitsOpt = None,
    db: DbOpt = None,
    number: Annotated[int, typer.Option("-n", "--number", help="How many events.")] = 20,
) -> None:
    """Show the most recent events."""
    pinned = _load_pinned(limits)
    with _open_ledger(db, pinned, read_only=True) as ledger:
        rows = ledger.tail(number)
        if not rows:
            console.print("[dim]ledger is empty[/dim]")
            return

        # The event type must never be abbreviated: `ledger.gene…` tells an
        # operator nothing. Rather than fight Rich's column allocator — which
        # shrinks every column when one wants more room than exists — the
        # payload is truncated here, so it is the only thing that ever loses
        # characters and the narrow columns always render in full.
        fixed_width = 4 + 12 + max((len(r["event_type"]) for r in rows), default=10) + 10
        payload_budget = max(20, (_WIDTH or console.width) - fixed_width - 6)

        table = Table(box=None, pad_edge=False)
        table.add_column("seq", no_wrap=True, justify="right")
        table.add_column("time", no_wrap=True)
        table.add_column("event", no_wrap=True, style="cyan")
        table.add_column("actor", no_wrap=True)
        table.add_column("payload", no_wrap=True)

        for row in rows:
            payload = row["payload_json"]
            table.add_row(
                str(row["seq"]),
                row["ts_utc"][11:23],
                row["event_type"],
                row["actor"],
                payload if len(payload) <= payload_budget else payload[: payload_budget - 1] + "…",
            )
        console.print(table)


@ledger_app.command("show")
def ledger_show(
    seq: Annotated[int, typer.Argument(help="Sequence number to display.")],
    limits: LimitsOpt = None,
    db: DbOpt = None,
) -> None:
    """Show one event in full, with its chain links."""
    pinned = _load_pinned(limits)
    with _open_ledger(db, pinned, read_only=True) as ledger:
        row = ledger.get(seq)
        if row is None:
            _fail(f"no event at seq={seq}")
            raise typer.Exit(1)
        table = Table(show_header=False, box=None, padding=(0, 2, 0, 0))
        # sqlite3.Row is not a Mapping: iterating it yields values, not keys,
        # so .keys() is required here rather than stylistic.
        for key in row.keys():  # noqa: SIM118
            if key == "payload_json":
                continue
            table.add_row(key, str(row[key]))
        console.print(table)
        console.print("\n[bold]payload[/bold]")
        console.print_json(row["payload_json"])


if __name__ == "__main__":  # pragma: no cover
    app()
