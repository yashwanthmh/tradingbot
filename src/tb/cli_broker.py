"""`tb broker ...`, `tb symbols ...` and `tb reconcile` — the M1 surface.

Read-only throughout. None of these commands can place or cancel an order;
that arrives in M4 behind a risk token.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Annotated, Any

import typer
from rich.console import Console
from rich.markup import escape
from rich.panel import Panel
from rich.table import Table

from tb.broker.reconcile import Reconciler, ReconcileVerdict, Severity
from tb.broker.t212.client import T212Client
from tb.broker.t212.endpoints import Endpoint, seconds_to_place_protective_stops
from tb.broker.t212.errors import AuthError, BrokerError, SchemaDriftError
from tb.broker.t212.probe import (
    ProbeReport,
    cache_instruments,
    cached_instruments,
    estimate_duration_seconds,
    run_probe,
)
from tb.broker.t212.ratelimit import RateGovernor
from tb.config.loader import PinnedLimits, load_hard_limits
from tb.core.errors import TbError
from tb.core.ids import new_run_id
from tb.data.symbols import Confidence, SymbolMap
from tb.ledger.store import Ledger, default_ledger_path

broker_app = typer.Typer(help="Inspect the Trading 212 account. Read-only.", no_args_is_help=True)
symbols_app = typer.Typer(
    help="Map Trading 212 tickers to market-data symbols.", no_args_is_help=True
)

# Same width policy as tb.cli: Rich falls back to 80 columns off a terminal,
# which truncates exactly the paths and hashes these commands exist to print.
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


def _load(limits: Path | None) -> PinnedLimits:
    try:
        return load_hard_limits(limits)
    except TbError as exc:
        err_console.print(f"{BAD} {escape(str(exc))}", soft_wrap=True)
        raise typer.Exit(2) from exc


def _ledger(db: Path | None, pinned: PinnedLimits) -> Ledger:
    path = db or default_ledger_path()
    if not path.exists():
        err_console.print(f"{BAD} no ledger at {path}. Run `tb init` first.", soft_wrap=True)
        raise typer.Exit(2)
    return Ledger(path, config_hash=pinned.config_hash).open()


def _client(ledger: Ledger, *, require_demo: bool, run_id: str, state_dir: Path) -> T212Client:
    try:
        return T212Client.from_env(
            governor=RateGovernor(state_path=state_dir / "ratelimit.json"),
            ledger=ledger,
            run_id=run_id,
            require_demo=require_demo,
        )
    except AuthError as exc:
        err_console.print(f"{BAD} {escape(str(exc))}", soft_wrap=True)
        err_console.print(
            "\nSet [bold]T212_DEMO_API_KEY[/bold]. Switch the Trading 212 app to "
            "Practice mode [bold]before[/bold] generating the key, or you will get a "
            "live one — the two are not interchangeable and look identical."
        )
        raise typer.Exit(2) from exc


# --------------------------------------------------------------------------
# tb broker
# --------------------------------------------------------------------------


@broker_app.command("probe")
def broker_probe(
    limits: LimitsOpt = None,
    db: DbOpt = None,
    skip_slow: Annotated[
        bool,
        typer.Option("--skip-slow", help="Omit the instruments, exchanges and account-info calls."),
    ] = False,
    allow_live: Annotated[
        bool,
        typer.Option("--allow-live", help="Permit probing a real-money account. Not advised."),
    ] = False,
) -> None:
    """Characterise the live API: shapes, limits, and auth format.

    Everything in the adapter's endpoint table and response models is a
    reconstruction — the official reference is not reachable from the build
    environment and the API is in beta. This command replaces guesses with
    observations and reports anything that disagrees.
    """
    pinned = _load(limits)
    run_id = new_run_id()
    estimate = estimate_duration_seconds(skip_slow=skip_slow)

    with _ledger(db, pinned) as ledger:
        client = _client(
            ledger,
            require_demo=not allow_live,
            run_id=run_id,
            state_dir=Path(pinned.limits.safety.kill_switch_path).parent,
        )
        ledger.record_run_start(run_id=run_id, mode="probe")

        console.print(
            Panel.fit(
                f"[bold]Probing the {client.environment} account[/bold]\n"
                f"about {estimate:.0f}s — the governor assumes its budget is spent on a\n"
                f"cold start, so the first call to a tightly limited endpoint waits a\n"
                f"full period. It is not hung.",
                border_style="dim",
            )
        )

        def progress(endpoint: Endpoint, wait: float) -> None:
            suffix = f" (waiting {wait:.0f}s for rate limit)" if wait > 1.0 else ""
            console.print(f"  … {endpoint.value}{suffix}")

        try:
            report = run_probe(client, ledger=ledger, skip_slow=skip_slow, on_progress=progress)
        except BrokerError as exc:
            err_console.print(f"{BAD} {escape(str(exc))}", soft_wrap=True)
            ledger.record_run_end(run_id=run_id, exit_reason="probe failed", error_detail=str(exc))
            raise typer.Exit(1) from exc

        _print_probe(report, pinned)

        # Cache the instrument list so later commands do not spend the
        # one-call-per-fifty-seconds budget re-reading it.
        instruments_result = next(
            (r for r in report.results if r.endpoint is Endpoint.INSTRUMENTS and r.parsed), None
        )
        if instruments_result is not None:
            try:
                cached = cache_instruments(ledger, client.get_instruments())
                console.print(f"{OK} cached {cached} instruments")
            except (BrokerError, SchemaDriftError) as exc:
                console.print(f"{WARN} could not cache instruments: {escape(str(exc))}")

        ledger.record_run_end(run_id=run_id, exit_reason="probe complete")
        client.close()

        if not report.usable:
            raise typer.Exit(1)


def _print_probe(report: ProbeReport, pinned: PinnedLimits) -> None:
    console.print()
    table = Table("endpoint", "status", "parsed", "items", "limit (obs/cfg)", "detail")
    for result in report.results:
        row_status = (
            f"[green]{result.status_code}[/green]"
            if result.ok
            else f"[red]{result.status_code or 'ERR'}[/red]"
        )
        parsed = OK if result.parsed else (BAD if result.ok else "—")
        observed = (
            f"{result.observed_limit}/{result.observed_period_seconds}s"
            if result.observed_limit is not None
            else "—"
        )
        configured = (
            f"{result.as_dict()['configured_limit']}/{result.as_dict()['configured_period_s']:g}s"
        )
        table.add_row(
            result.endpoint.value,
            row_status,
            parsed,
            "—" if result.item_count is None else str(result.item_count),
            f"{observed} / {configured}",
            escape(result.detail[:70]),
        )
    console.print(table)

    console.print(f"\n{OK} auth header shape: [bold]{report.auth_scheme}[/bold]")
    console.print(f"{OK} {report.summary()}")

    if report.base_currency:
        matches = report.base_currency.upper() == pinned.limits.currency.upper()
        marker = OK if matches else BAD
        console.print(
            f"{marker} account currency {report.base_currency} vs hard_limits "
            f"{pinned.limits.currency}"
        )
        if not matches:
            console.print(
                f"  Every *_ccy cap would be read in the wrong currency. Set "
                f"[bold]currency: {report.base_currency.upper()}[/bold] in "
                "hard_limits.yaml and re-check the absolute ceiling."
            )

    if report.disagreements:
        console.print(f"\n{WARN} [bold]rate limits differ from the assumed table:[/bold]")
        for note in report.disagreements:
            console.print(f"  • {escape(note)}")
        console.print(
            "  These are now recorded in endpoint_observations. Update "
            "src/tb/broker/t212/endpoints.py to match."
        )

    if report.unknown_fields:
        console.print(f"\n{WARN} [bold]fields present that the models do not declare:[/bold]")
        for endpoint, fields in sorted(report.unknown_fields.items()):
            console.print(f"  • {endpoint}: {', '.join(escape(f) for f in fields[:12])}")
        console.print("  Ignored by design. Listed in case any of them is worth consuming.")

    if report.unmapped_enum_values:
        console.print(f"\n{WARN} [bold]enum values with no mapping:[/bold]")
        for value in report.unmapped_enum_values:
            console.print(f"  • {escape(value)}")
        console.print("  Mapped to UNKNOWN rather than guessed at. Extend the tables.")

    for result in report.drifted:
        console.print(f"\n{BAD} [bold]{result.endpoint.value}[/bold] did not parse")
        console.print(f"  {escape(result.detail)}")
        console.print("  The raw body is in broker_messages; `tb broker drift` lists them.")

    if not report.usable:
        console.print(
            f"\n{BAD} [bold]The adapter is not usable against this account.[/bold] "
            "Equity, positions and open orders must all read and parse."
        )
    else:
        console.print(f"\n{OK} the essential endpoints read and parse")


@broker_app.command("account")
def broker_account(limits: LimitsOpt = None, db: DbOpt = None) -> None:
    """Show cash, equity and base currency."""
    pinned = _load(limits)
    with _ledger(db, pinned) as ledger:
        run_id = new_run_id()
        client = _client(
            ledger,
            require_demo=False,
            run_id=run_id,
            state_dir=Path(pinned.limits.safety.kill_switch_path).parent,
        )
        try:
            info = client.get_account_info()
            cash = client.get_cash(currency=info.currency_code)
        except (BrokerError, SchemaDriftError) as exc:
            err_console.print(f"{BAD} {escape(str(exc))}", soft_wrap=True)
            raise typer.Exit(1) from exc
        finally:
            client.close()

        banner = "[red]REAL MONEY[/red]" if client.is_real_money else "[green]demo[/green]"
        table = Table(show_header=False, box=None, padding=(0, 2, 0, 0))
        table.add_row("environment", banner)
        table.add_row("account", str(info.account_id or "—"))
        table.add_row("currency", info.currency_code or "—")
        table.add_row("free", str(cash.free))
        table.add_row("total (equity)", str(cash.total))
        table.add_row("invested", str(cash.invested))
        table.add_row("blocked by pending orders", str(cash.blocked))
        console.print(table)

        # The caps that actually bind, computed against real equity.
        if cash.total is not None:
            lim = pinned.limits.capital
            from decimal import Decimal

            pct_cap = cash.total * Decimal(str(lim.max_deployed_pct)) / Decimal(100)
            binding = min(pct_cap, lim.absolute_ceiling_ccy)
            console.print(
                f"\nmax deployable: [bold]{binding}[/bold] "
                f"({lim.max_deployed_pct}% of equity = {pct_cap:.2f}, "
                f"absolute ceiling = {lim.absolute_ceiling_ccy}; the tighter binds)"
            )


@broker_app.command("portfolio")
def broker_portfolio(limits: LimitsOpt = None, db: DbOpt = None) -> None:
    """Show open positions and open orders."""
    pinned = _load(limits)
    with _ledger(db, pinned) as ledger:
        run_id = new_run_id()
        client = _client(
            ledger,
            require_demo=False,
            run_id=run_id,
            state_dir=Path(pinned.limits.safety.kill_switch_path).parent,
        )
        try:
            snapshot = client.snapshot()
        except (BrokerError, SchemaDriftError) as exc:
            err_console.print(f"{BAD} {escape(str(exc))}", soft_wrap=True)
            raise typer.Exit(1) from exc
        finally:
            client.close()

        if snapshot.positions:
            table = Table("ticker", "qty", "avg", "last", "P/L")
            for position in snapshot.positions:
                table.add_row(
                    position.ticker,
                    str(position.quantity),
                    str(position.average_price or "—"),
                    str(position.current_price or "—"),
                    str(position.ppl or "—"),
                )
            console.print(table)
        else:
            console.print("[dim]no open positions[/dim]")

        if snapshot.open_orders:
            table = Table("order", "ticker", "side", "type", "qty", "limit", "stop", "status")
            for order in snapshot.open_orders:
                table.add_row(
                    order.broker_order_id,
                    order.ticker,
                    order.side.value if order.side else "—",
                    order.order_type.value if order.order_type else "—",
                    str(order.quantity or "—"),
                    str(order.limit_price or "—"),
                    str(order.stop_price or "—"),
                    order.status.value,
                )
            console.print(table)
        else:
            console.print("[dim]no open orders[/dim]")

        for warning in snapshot.staleness_warnings:
            console.print(f"{WARN} {escape(warning)}")


@broker_app.command("limits")
def broker_limits(limits: LimitsOpt = None, db: DbOpt = None) -> None:
    """Show the rate-limit table, and what it implies for the universe size."""
    pinned = _load(limits)
    governor = RateGovernor(
        state_path=Path(pinned.limits.safety.kill_switch_path).parent / "ratelimit.json"
    )

    # `method` is folded into the path rather than given its own column: at 80
    # columns the path is what gets truncated otherwise, and the path is the
    # content.
    table = Table(box=None, pad_edge=False)
    table.add_column("endpoint", no_wrap=True)
    table.add_column("request", no_wrap=True)
    table.add_column("configured", no_wrap=True, justify="right")
    table.add_column("observed", no_wrap=True, justify="right")
    table.add_column("ok", no_wrap=True, justify="center")

    for row in sorted(governor.observations(), key=lambda r: str(r["endpoint"])):
        observed = (
            f"{row['observed_limit']}/{row['observed_period_s']}s"
            if row["observed_limit"] is not None
            else "[dim]not probed[/dim]"
        )
        agrees = {None: "—", True: OK, False: BAD}[row["agrees"]]
        table.add_row(
            str(row["endpoint"]),
            f"{row['method']} {row['path']}",
            f"{row['configured_limit']}/{row['configured_period_s']:g}s",
            observed,
            agrees,
        )
    console.print(table)

    universe = pinned.limits.execution.max_universe_symbols
    seconds = seconds_to_place_protective_stops(universe)
    console.print(
        f"\nProtective stops are limit-class orders at one per two seconds, so "
        f"protecting a full turnover of [bold]{universe}[/bold] symbols needs "
        f"[bold]{seconds:.0f}s[/bold] of governor budget for protection alone."
    )
    if seconds > 60:
        console.print(
            f"{WARN} that exceeds one minute, so a whole-universe turn cannot complete "
            "inside the bar that triggered it. This is why the design targets "
            "minute-resolution features with hour-to-day position changes."
        )


@broker_app.command("drift")
def broker_drift(
    limits: LimitsOpt = None,
    db: DbOpt = None,
    number: Annotated[int, typer.Option("-n", "--number")] = 10,
) -> None:
    """List responses that failed to parse, for replay."""
    pinned = _load(limits)
    with _ledger(db, pinned) as ledger:
        rows = ledger.conn.execute(
            "SELECT msg_id, endpoint, url_path, status_code, parse_error, received_at "
            "FROM broker_messages WHERE parse_ok = 0 ORDER BY received_at DESC LIMIT ?",
            (number,),
        ).fetchall()
        if not rows:
            console.print(f"{OK} no unparsed responses recorded")
            return
        table = Table("msg_id", "endpoint", "status", "when", "error")
        for row in rows:
            table.add_row(
                row["msg_id"],
                row["endpoint"],
                str(row["status_code"]),
                str(row["received_at"])[:19],
                escape(str(row["parse_error"] or "")[:60]),
            )
        console.print(table)
        console.print(
            "\nThe raw bodies are stored alongside. Replay one with "
            "[bold]tb broker replay <msg_id>[/bold]."
        )


@broker_app.command("replay")
def broker_replay(
    msg_id: Annotated[str, typer.Argument(help="Archived message id.")],
    limits: LimitsOpt = None,
    db: DbOpt = None,
) -> None:
    """Print an archived response body."""
    pinned = _load(limits)
    with _ledger(db, pinned) as ledger:
        row = ledger.conn.execute(
            "SELECT * FROM broker_messages WHERE msg_id = ?", (msg_id,)
        ).fetchone()
        if row is None:
            err_console.print(f"{BAD} no archived message {msg_id}", soft_wrap=True)
            raise typer.Exit(1)
        console.print(
            f"[bold]{row['method']} {row['url_path']}[/bold] -> {row['status_code']} "
            f"at {row['received_at']}"
        )
        if row["parse_error"]:
            console.print(f"{BAD} {escape(str(row['parse_error']))}")
        body = row["raw_body"] or ""
        if body.strip().startswith(("{", "[")):
            console.print_json(body)
        else:
            console.print(escape(body))


# --------------------------------------------------------------------------
# tb symbols
# --------------------------------------------------------------------------


@symbols_app.command("audit")
def symbols_audit(
    limits: LimitsOpt = None,
    db: DbOpt = None,
    provider: Annotated[str, typer.Option("--provider", help="alpaca or yfinance.")] = "alpaca",
    refresh: Annotated[
        bool, typer.Option("--refresh", help="Re-fetch the instrument list from the broker.")
    ] = False,
) -> None:
    """Derive and report the Trading 212 -> data-provider symbol mapping.

    Derivation alone never makes an instrument tradable. Promotion to
    `verified` needs a price comparison against the broker's own quote, which
    needs the market-data layer from M2 — so on a fresh install this command
    correctly reports everything as not yet tradable.
    """
    pinned = _load(limits)
    with _ledger(db, pinned) as ledger:
        instruments = tuple(cached_instruments(ledger))

        if refresh or not instruments:
            run_id = new_run_id()
            client = _client(
                ledger,
                require_demo=False,
                run_id=run_id,
                state_dir=Path(pinned.limits.safety.kill_switch_path).parent,
            )
            console.print("fetching the instrument list (one call per ~50s, large payload)…")
            try:
                instruments = client.get_instruments()
                cache_instruments(ledger, instruments)
            except (BrokerError, SchemaDriftError) as exc:
                err_console.print(f"{BAD} {escape(str(exc))}", soft_wrap=True)
                raise typer.Exit(1) from exc
            finally:
                client.close()

        symbol_map = SymbolMap(ledger, provider=provider)
        counts = symbol_map.derive_all(instruments)
        summary = symbol_map.audit_summary(n_instruments=len(instruments))
        symbol_map.record_audit(summary)

        table = Table(show_header=False, box=None, padding=(0, 2, 0, 0))
        table.add_row("provider", provider)
        table.add_row("instruments", str(summary["n_instruments"]))
        table.add_row("mapped", str(summary["n_mapped"]))
        table.add_row("unmapped", f"[yellow]{summary['n_unmapped']}[/yellow]")
        table.add_row("verified (tradable)", f"[green]{summary['n_verified']}[/green]")
        table.add_row("derived (not tradable yet)", str(summary["n_derived"]))
        table.add_row("ambiguous", f"[yellow]{summary['n_ambiguous']}[/yellow]")
        table.add_row("blocked", f"[red]{summary['n_blocked']}[/red]")
        console.print(table)
        console.print(f"\n[dim]this run derived: {counts}[/dim]")

        unmapped = [m for m in symbol_map.all() if not m.data_symbol][:10]
        if unmapped:
            console.print(f"\n{WARN} [bold]a sample of what could not be mapped:[/bold]")
            for mapping in unmapped:
                console.print(
                    f"  • {mapping.t212_ticker} ({mapping.currency_code or '?'}) — "
                    f"{escape(mapping.derivation)}"
                )

        ambiguous = [m for m in symbol_map.all() if m.confidence is Confidence.AMBIGUOUS][:10]
        if ambiguous:
            console.print(
                f"\n{WARN} [bold]ambiguous — the bare symbol may be a different "
                "company on the provider:[/bold]"
            )
            for mapping in ambiguous:
                console.print(f"  • {mapping.t212_ticker} -> {mapping.data_symbol}")

        if summary["n_verified"] == 0:
            console.print(
                f"\n{WARN} nothing is verified, so nothing may be entered. Verification "
                "compares the broker's quote against the data feed, which needs the "
                "market-data layer (M2). Exits are never gated on this."
            )


@symbols_app.command("show")
def symbols_show(
    ticker: Annotated[str, typer.Argument(help="Trading 212 ticker, e.g. AAPL_US_EQ.")],
    limits: LimitsOpt = None,
    db: DbOpt = None,
) -> None:
    """Show one mapping and whether it permits an entry."""
    pinned = _load(limits)
    with _ledger(db, pinned) as ledger:
        symbol_map = SymbolMap(ledger)
        mapping = symbol_map.get(ticker)
        if mapping is None:
            err_console.print(
                f"{BAD} no mapping for {ticker}. Run `tb symbols audit`.", soft_wrap=True
            )
            raise typer.Exit(1)

        may_enter, why_enter = symbol_map.may_enter(ticker)
        _, why_exit = symbol_map.may_exit(ticker)

        table = Table(show_header=False, box=None, padding=(0, 2, 0, 0))
        table.add_row("t212 ticker", mapping.t212_ticker)
        table.add_row("data symbol", mapping.data_symbol or "[red]none[/red]")
        table.add_row("provider", mapping.provider)
        table.add_row("currency", mapping.currency_code or "—")
        table.add_row("confidence", mapping.confidence.value)
        table.add_row("derivation", escape(mapping.derivation))
        table.add_row("verified at", mapping.verified_at or "[yellow]never[/yellow]")
        table.add_row(
            "last disagreement",
            "—"
            if mapping.last_disagreement_bps is None
            else f"{mapping.last_disagreement_bps:.1f}bps",
        )
        table.add_row("blocked", str(mapping.blocked))
        if mapping.blocked_reason:
            table.add_row("blocked reason", escape(mapping.blocked_reason))
        console.print(table)

        console.print(f"\n{OK if may_enter else BAD} may open a position: {escape(why_enter)}")
        console.print(f"{OK} may close a position: {escape(why_exit)}")


# --------------------------------------------------------------------------
# tb reconcile
# --------------------------------------------------------------------------


def reconcile_command(
    limits: Path | None = None,
    db: Path | None = None,
    dry_run: bool = True,
) -> None:
    """Compare intents, open orders, and position/cash. Report what disagrees."""
    pinned = _load(limits)
    with _ledger(db, pinned) as ledger:
        run_id = new_run_id()
        client = _client(
            ledger,
            require_demo=False,
            run_id=run_id,
            state_dir=Path(pinned.limits.safety.kill_switch_path).parent,
        )
        ledger.record_run_start(run_id=run_id, mode="reconcile")
        try:
            reconciler = Reconciler(
                ledger=ledger,
                broker=client,
                limits=pinned.limits,
                symbol_map=SymbolMap(ledger),
                run_id=run_id,
            )
            report = reconciler.run(dry_run=dry_run)
        except (BrokerError, SchemaDriftError) as exc:
            err_console.print(f"{BAD} {escape(str(exc))}", soft_wrap=True)
            ledger.record_run_end(run_id=run_id, exit_reason="reconcile failed")
            raise typer.Exit(1) from exc
        finally:
            client.close()

        ledger.record_run_end(run_id=run_id, exit_reason=report.verdict.value)
        _print_reconcile(report)

        if report.verdict is ReconcileVerdict.HALTED:
            raise typer.Exit(1)


def _print_reconcile(report: Any) -> None:
    marker = {
        ReconcileVerdict.CLEAN: OK,
        ReconcileVerdict.REPAIRED: WARN,
        ReconcileVerdict.HALTED: BAD,
    }[report.verdict]
    console.print(f"{marker} {report.summary()}")

    if report.snapshot is not None:
        snap = report.snapshot
        console.print(
            f"  {len(snap.positions)} position(s), {len(snap.open_orders)} open order(s), "
            f"equity {snap.cash.total} {snap.cash.currency or ''}"
        )

    if not report.findings:
        return

    console.print()
    table = Table("severity", "ticker", "finding", "detail", "suggested action")
    order = {Severity.BLOCKING: 0, Severity.WARN: 1, Severity.INFO: 2}
    for finding in sorted(report.findings, key=lambda f: order[f.severity]):
        colour = {
            Severity.BLOCKING: "red",
            Severity.WARN: "yellow",
            Severity.INFO: "dim",
        }[finding.severity]
        table.add_row(
            f"[{colour}]{finding.severity.value}[/{colour}]",
            finding.ticker or "—",
            finding.kind.value,
            escape(finding.detail),
            escape(finding.suggested_action or "—"),
        )
    console.print(table)

    if report.dry_run and report.blocking:
        console.print(
            f"\n{WARN} M1 is read-only, so nothing was repaired. The suggested actions "
            "become available in M4, behind the risk engine."
        )
