"""`tb data ...` and `tb universe ...` — the M2 surface.

Read-and-write, but only into the data layer: nothing here can place an order.
The commands correspond one-to-one with the M2 verification steps, in the order
you would actually run them:

    tb universe build     pick the symbols, record a dated snapshot
    tb data backfill      fetch history for them
    tb data audit         classify what is wrong with it
    tb data bakeoff       measure whether the feed is good enough to trade
    tb data seal          freeze a vintage M3 can cite
    tb data verify        re-hash a sealed vintage

`backfill` and `bakeoff` are the only two that touch a network, and both name
which provider they are calling before they call it.
"""

from __future__ import annotations

import os
import sys
from datetime import UTC, datetime, timedelta
from decimal import Decimal
from pathlib import Path
from typing import Annotated

import typer
from rich.console import Console
from rich.markup import escape
from rich.table import Table

from tb.broker.reconcile import Severity
from tb.broker.t212.probe import cached_instruments
from tb.config.loader import PinnedLimits, load_hard_limits
from tb.core.errors import TbError
from tb.core.ids import new_run_id
from tb.data.actions import ActionStore
from tb.data.audit import DataAuditor, summarise_gaps
from tb.data.bakeoff import Bakeoff, Verdict, may_widen_live_resolutions
from tb.data.barstore import BarStore
from tb.data.calendar import TradingCalendar
from tb.data.fx import FxStore, rates_from_bars
from tb.data.provider import (
    DataError,
    MarketDataProvider,
    Provenance,
    Resolution,
    make_instrument_uid,
)
from tb.data.providers import AlpacaProvider, YahooProvider
from tb.data.snapshot import SnapshotStore
from tb.data.symbols import SymbolMap
from tb.data.universe import (
    UniverseStore,
    build_candidates,
    dollar_volume_from_bars,
    select,
)
from tb.ledger.store import Ledger, default_ledger_path

data_app = typer.Typer(help="Fetch, audit, compare and seal market data.", no_args_is_help=True)
universe_app = typer.Typer(help="Choose and record the tradable universe.", no_args_is_help=True)

# Same width policy as the rest of the CLI: Rich falls back to 80 columns off a
# terminal, which truncates exactly the vintage ids and hashes these commands
# exist to print.
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
RootOpt = Annotated[
    Path | None,
    typer.Option("--bars", help="Directory holding the Parquet bar store.", show_default=False),
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


def _store(ledger: Ledger, pinned: PinnedLimits, root: Path | None) -> BarStore:
    return BarStore(
        ledger,
        root=root or (Path(ledger.path).parent / "bars"),
        scale=pinned.limits.data.price_scale,
    )


KNOWN_PROVIDERS = ("alpaca", "yahoo")


def _check_provider(name: str) -> str:
    """Validate a provider name before doing any work.

    Checked first, so a typo fails on the spot rather than after the universe
    lookup — and there is deliberately no registry mapping arbitrary config
    strings to classes, because a typo must not be able to silently swap the
    feed underneath a strategy.
    """
    if name not in KNOWN_PROVIDERS:
        err_console.print(
            f"{BAD} unknown provider {name!r}. Known providers: {', '.join(KNOWN_PROVIDERS)}.",
            soft_wrap=True,
        )
        raise typer.Exit(2)
    return name


def _provider(name: str) -> MarketDataProvider:
    """Build a validated provider, resolving credentials where it needs them."""
    _check_provider(name)
    if name == "yahoo":
        return YahooProvider()
    try:
        return AlpacaProvider.from_env()
    except DataError as exc:
        err_console.print(f"{BAD} {escape(str(exc))}", soft_wrap=True)
        raise typer.Exit(2) from exc


def _resolution(text: str) -> Resolution:
    try:
        return Resolution(text)
    except ValueError as exc:
        err_console.print(
            f"{BAD} unknown resolution {text!r}. Use daily, hourly or minute.", soft_wrap=True
        )
        raise typer.Exit(2) from exc


# --------------------------------------------------------------------------
# tb universe build
# --------------------------------------------------------------------------


@universe_app.command("build")
def universe_build(
    limits: LimitsOpt = None,
    db: DbOpt = None,
    bars: RootOpt = None,
    provider: Annotated[
        str, typer.Option("--provider", help="Data provider the symbols must map to.")
    ] = "alpaca",
    min_dollar_volume: Annotated[
        float,
        typer.Option(
            "--min-dollar-volume",
            help="Liquidity floor. An unmeasured name counts as illiquid, not liquid.",
        ),
    ] = 0.0,
) -> None:
    """Select the tradable universe and record it as a dated snapshot.

    The cap comes from `execution.max_universe_symbols`, which is the
    rate-limit budget rather than a preference: every symbol costs poll
    capacity each cycle, so a larger universe buys older prices, not more
    breadth.

    Snapshots are append-only and dated from today forward. That does not fix
    survivorship bias on history nobody recorded — nothing can — but it stops
    the bias growing, and any backtest over a window predating the first
    snapshot is stamped `survivorship: unmeasured`.
    """
    pinned = _load(limits)
    with _ledger(db, pinned) as ledger:
        instruments = tuple(cached_instruments(ledger))
        if not instruments:
            err_console.print(
                f"{BAD} no cached instrument list. Run `tb symbols audit --refresh` first.",
                soft_wrap=True,
            )
            raise typer.Exit(2)

        symbol_map = SymbolMap(ledger, provider=provider)
        store = _store(ledger, pinned, bars)

        mapped = {
            mapping.t212_ticker: mapping.data_symbol
            for mapping in symbol_map.all()
            if mapping.data_symbol and not mapping.blocked
        }
        volumes: dict[str, Decimal] = {}
        for instrument in instruments:
            if instrument.ticker not in mapped:
                continue
            uid = make_instrument_uid(isin=instrument.isin, t212_ticker=instrument.ticker)
            measured = dollar_volume_from_bars(store.bars_for(uid, Resolution.DAILY))
            if measured is not None:
                volumes[instrument.ticker] = measured

        candidates, rejections = build_candidates(
            instruments, symbol_for=mapped, dollar_volume=volumes
        )
        snapshot = select(
            candidates,
            max_symbols=pinned.limits.execution.max_universe_symbols,
            preferred_currency=pinned.limits.currency,
            min_dollar_volume=Decimal(str(min_dollar_volume)) if min_dollar_volume else None,
        )
        recorded = UniverseStore(ledger, run_id=new_run_id()).record(snapshot)

        table = Table(show_header=False, box=None, padding=(0, 2, 0, 0))
        table.add_row("snapshot", snapshot.snapshot_id)
        table.add_row("rule", escape(snapshot.selection_rule))
        table.add_row("candidates", str(len(candidates)))
        table.add_row("members", f"[green]{len(snapshot)}[/green]")
        table.add_row("rejected", str(len(rejections) + len(snapshot.rejections)))
        table.add_row("all ISIN-keyed", str(snapshot.all_ids_stable))
        console.print(table)

        if snapshot.members:
            members = Table(box=None, padding=(0, 2, 0, 0))
            members.add_column("rank", justify="right")
            members.add_column("t212")
            members.add_column("data")
            members.add_column("$ volume", justify="right")
            for member in snapshot.members[:25]:
                members.add_row(
                    str(member.rank),
                    member.t212_ticker,
                    member.data_symbol,
                    "—" if member.dollar_volume is None else f"{member.dollar_volume:,.0f}",
                )
            console.print("")
            console.print(members)

        if not recorded:
            console.print(f"\n{OK} identical to the snapshot already recorded today")
        if not snapshot.all_ids_stable:
            console.print(
                f"\n{WARN} some members have no ISIN, so their identity is only as stable "
                "as the ticker string. A reused ticker could splice two companies' "
                "histories; the audit reports which."
            )


# --------------------------------------------------------------------------
# tb data backfill
# --------------------------------------------------------------------------


@data_app.command("backfill")
def data_backfill(
    limits: LimitsOpt = None,
    db: DbOpt = None,
    bars: RootOpt = None,
    resolution: Annotated[
        str, typer.Option("--resolution", help="daily, hourly or minute.")
    ] = "daily",
    provider: Annotated[str, typer.Option("--provider", help="alpaca or yahoo.")] = "yahoo",
    years: Annotated[
        int | None, typer.Option("--years", help="Override data.backfill_years_daily.")
    ] = None,
    days: Annotated[
        int | None, typer.Option("--days", help="Override data.backfill_days_minute.")
    ] = None,
    symbols: Annotated[
        str | None,
        typer.Option("--symbols", help="Comma-separated T212 tickers. Default: the universe."),
    ] = None,
    fx: Annotated[
        bool, typer.Option("--fx/--no-fx", help="Also fetch the account-currency rate.")
    ] = True,
    seal: Annotated[
        bool, typer.Option("--seal/--no-seal", help="Compact into Parquet when done.")
    ] = True,
) -> None:
    """Fetch history into the bar store.

    Defaults to Yahoo for daily bars deliberately. Alpaca's IEX archive starts
    around 2016, so a ten-year daily backfill has to come from the consolidated
    feed — while Alpaca is the one to use for anything recent, because it
    honours `adjustment=raw` and Yahoo back-adjusts silently.

    Backfilled bars carry `provenance=backfill`, meaning their knowledge times
    are assumed rather than measured. Any vintage built only from these is
    stamped `vendor_current_view`, which is honest rather than
    point-in-time.
    """
    pinned = _load(limits)
    res = _resolution(resolution)
    _check_provider(provider)
    with _ledger(db, pinned) as ledger:
        store = _store(ledger, pinned, bars)
        symbol_map = SymbolMap(ledger, provider=provider)
        instruments = {i.ticker: i for i in cached_instruments(ledger)}

        if symbols:
            tickers = [t.strip() for t in symbols.split(",") if t.strip()]
        else:
            snapshot = UniverseStore(ledger).latest()
            if snapshot is None:
                err_console.print(
                    f"{BAD} no universe snapshot. Run `tb universe build` first, or pass "
                    "--symbols.",
                    soft_wrap=True,
                )
                raise typer.Exit(2)
            tickers = list(snapshot.tickers)

        span = (
            timedelta(days=365 * (years or pinned.limits.data.backfill_years_daily))
            if res is Resolution.DAILY
            else timedelta(days=days or pinned.limits.data.backfill_days_minute)
        )
        end = datetime.now(UTC)
        start = end - span

        feed = _provider(provider)
        console.print(
            f"fetching {res.value} bars for {len(tickers)} symbol(s) from "
            f"[bold]{provider}[/bold], {start.date()} to {end.date()}"
        )

        total = 0
        revisions = 0
        problems: list[str] = []
        try:
            for ticker in tickers:
                mapping = symbol_map.get(ticker)
                if mapping is None or not mapping.data_symbol:
                    problems.append(f"{ticker}: no data symbol mapped")
                    continue
                instrument = instruments.get(ticker)
                uid = make_instrument_uid(
                    isin=None if instrument is None else instrument.isin, t212_ticker=ticker
                )
                try:
                    batch = feed.fetch_bars(
                        mapping.data_symbol,
                        instrument_uid=uid,
                        resolution=res,
                        start=start,
                        end=end,
                        provenance=Provenance.BACKFILL,
                    )
                except TbError as exc:
                    problems.append(f"{ticker}: {exc}")
                    continue

                result = store.ingest(batch)
                total += result.rows_written
                revisions += len(result.revisions)
                problems.extend(f"{ticker}: {w}" for w in batch.warnings)

            if fx:
                total += _backfill_fx(ledger, feed, pinned, start=start, end=end, problems=problems)
        finally:
            feed.close()

        sealed = store.compact() if seal else []

        table = Table(show_header=False, box=None, padding=(0, 2, 0, 0))
        table.add_row("rows written", str(total))
        table.add_row("revisions detected", f"[yellow]{revisions}[/yellow]" if revisions else "0")
        table.add_row("partitions sealed", str(len(sealed)))
        table.add_row("warnings", f"[yellow]{len(problems)}[/yellow]" if problems else "0")
        console.print(table)

        for problem in problems[:15]:
            console.print(f"  {WARN} {escape(problem)}")
        if len(problems) > 15:
            console.print(f"  [dim]… and {len(problems) - 15} more[/dim]")

        console.print(f"\n{OK} now run [bold]tb data audit[/bold]")


def _backfill_fx(
    ledger: Ledger,
    feed: MarketDataProvider,
    pinned: PinnedLimits,
    *,
    start: datetime,
    end: datetime,
    problems: list[str],
) -> int:
    """Fetch the account-currency rate through the same provider path.

    Reusing the bar path rather than growing a second ingest means the rate
    inherits the bar's validation, its knowledge time and its revision
    detection. A separate, weaker FX path is how one of the two ends up without
    them.
    """
    account = pinned.limits.currency
    if account == "USD":
        return 0
    symbol = f"{account}USD=X"
    try:
        batch = feed.fetch_bars(
            symbol,
            instrument_uid=f"sym:{symbol}",
            resolution=Resolution.DAILY,
            start=start,
            end=end,
            provenance=Provenance.BACKFILL,
        )
    except TbError as exc:
        problems.append(f"{symbol}: {exc}")
        return 0

    rates = rates_from_bars(batch.bars, base=account, quote="USD", provider=feed.name)
    written, revised = FxStore(ledger).record(rates)
    problems.extend(f"fx: {note}" for note in revised)
    return written


# --------------------------------------------------------------------------
# tb data audit
# --------------------------------------------------------------------------


@data_app.command("audit")
def data_audit(
    limits: LimitsOpt = None,
    db: DbOpt = None,
    bars: RootOpt = None,
    resolution: Annotated[
        str, typer.Option("--resolution", help="daily, hourly or minute.")
    ] = "daily",
) -> None:
    """Classify what is wrong with the stored data. Exit 1 if anything blocks.

    Gaps are reported by *cause*, not as a total. A missing minute on a thin
    name over IEX is normal; on a megacap it is a feed failure; a missing
    afternoon on Christmas Eve is a half-day. Only `unexplained` counts against
    `data.max_unexplained_gap_pct`, because a single uninterpretable number
    gets ignored — and then the gap that mattered is never seen.
    """
    pinned = _load(limits)
    res = _resolution(resolution)
    with _ledger(db, pinned) as ledger:
        store = _store(ledger, pinned, bars)
        auditor = DataAuditor(
            ledger,
            store,
            calendar=TradingCalendar(),
            actions=ActionStore(ledger),
            max_unexplained_gap_pct=float(pinned.limits.data.max_unexplained_gap_pct),
            max_provider_delay_seconds=pinned.limits.data.max_provider_delay_seconds,
            run_id=new_run_id(),
        )
        report = auditor.run(resolutions=[res])

        table = Table(show_header=False, box=None, padding=(0, 2, 0, 0))
        table.add_row("instruments", str(report.n_instruments))
        table.add_row("bars checked", f"{report.n_bars_checked:,}")
        table.add_row("findings", str(len(report.findings)))
        table.add_row(
            "blocking",
            f"[red]{len(report.blocking)}[/red]" if report.blocking else "[green]0[/green]",
        )
        table.add_row("gaps unexplained", f"[yellow]{report.gaps_unexplained}[/yellow]")
        table.add_row("gaps explained", str(report.gaps_explained))
        console.print(table)

        causes = summarise_gaps(report.coverage)
        if causes:
            console.print("\n[bold]gaps by cause[/bold] (only `unexplained` counts):")
            for cause, count in causes.items():
                marker = WARN if cause == "unexplained" else " "
                console.print(f"  {marker} {cause}: {count}")

        worst = report.worst_coverage()
        if worst and worst[0].unexplained:
            console.print("\n[bold]worst coverage[/bold]:")
            for coverage in worst:
                if not coverage.unexplained:
                    continue
                console.print(
                    f"  • {coverage.instrument_uid} {coverage.resolution.value}: "
                    f"{coverage.unexplained}/{coverage.expected} unexplained "
                    f"({coverage.unexplained_pct:.2f}%)"
                )

        shown = 0
        for finding in sorted(report.findings, key=lambda f: 0 if f.blocking else 1):
            if finding.severity is Severity.INFO and shown >= 10:
                continue
            mark = BAD if finding.blocking else WARN
            console.print(f"\n{mark} {escape(str(finding))}")
            if finding.suggested_action:
                console.print(f"    [dim]→ {escape(finding.suggested_action)}[/dim]")
            shown += 1
            if shown >= 20:
                console.print(f"\n[dim]… {len(report.findings) - shown} more findings[/dim]")
                break

        if report.suspicions:
            console.print(
                f"\n{BAD} {len(report.suspicions)} unexplained split(s) detected. Entries in "
                "those symbols should be blocked until a provider confirms the action — "
                "the inferred ratio is evidence, not something to adjust prices by."
            )

        if not report.clean:
            raise typer.Exit(1)
        console.print(f"\n{OK} nothing blocking")


# --------------------------------------------------------------------------
# tb data bakeoff
# --------------------------------------------------------------------------


@data_app.command("bakeoff")
def data_bakeoff(
    limits: LimitsOpt = None,
    db: DbOpt = None,
    bars: RootOpt = None,
    resolution: Annotated[
        str, typer.Option("--resolution", help="Usually minute — that is the question.")
    ] = "minute",
    primary: Annotated[str, typer.Option("--primary", help="Provider under test.")] = "alpaca",
    secondary: Annotated[str, typer.Option("--secondary", help="Reference feed.")] = "yahoo",
) -> None:
    """Measure whether the free feed is good enough to trade this resolution.

    The output is one number: the gross edge a strategy would need before the
    feed's own disagreement is small enough to trade through. Compare it to
    what a round trip costs on Trading 212 and the paid-data question answers
    itself.

    Reads bars already in the store — run `tb data backfill` for both providers
    first. Comparing only what is stored keeps the measurement reproducible;
    fetching live here would measure a different window on every run.
    """
    pinned = _load(limits)
    res = _resolution(resolution)
    with _ledger(db, pinned) as ledger:
        store = _store(ledger, pinned, bars)
        uids = list(store.instruments())
        if not uids:
            err_console.print(
                f"{BAD} the store is empty. Run `tb data backfill` for both providers first.",
                soft_wrap=True,
            )
            raise typer.Exit(2)

        primary_bars = []
        secondary_bars = []
        for uid in uids:
            for bar in store.bars_for(uid, res):
                if bar.provider == primary:
                    primary_bars.append(bar)
                elif bar.provider == secondary:
                    secondary_bars.append(bar)

        if not primary_bars or not secondary_bars:
            missing = primary if not primary_bars else secondary
            err_console.print(
                f"{BAD} no {res.value} bars from {missing} in the store. A bake-off needs "
                f"both feeds: run `tb data backfill --provider {missing} "
                f"--resolution {res.value}`.",
                soft_wrap=True,
            )
            raise typer.Exit(2)

        result = Bakeoff(
            ledger,
            min_edge_to_noise_ratio=pinned.limits.data.min_edge_to_feed_noise_ratio,
            max_delay_seconds=pinned.limits.data.max_provider_delay_seconds,
            run_id=new_run_id(),
        ).run(
            primary_bars=primary_bars,
            secondary_bars=secondary_bars,
            resolution=res,
            primary=primary,
            secondary=secondary,
        )

        table = Table(show_header=False, box=None, padding=(0, 2, 0, 0))
        table.add_row("resolution", res.value)
        table.add_row("symbols", str(result.n_symbols))
        table.add_row("comparable bars", f"{result.n_compared_bars:,}")
        table.add_row(
            "disagreement median",
            "—" if result.median_bps is None else f"{result.median_bps:.1f}bps",
        )
        table.add_row(
            "disagreement p95", "—" if result.p95_bps is None else f"{result.p95_bps:.1f}bps"
        )
        table.add_row(
            "disagreement p99", "—" if result.p99_bps is None else f"{result.p99_bps:.1f}bps"
        )
        required = result.minimum_viable_edge_bps
        table.add_row(
            "[bold]minimum viable edge[/bold]",
            "—" if required is None else f"[bold]{required:.0f}bps gross[/bold]",
        )
        for stat in result.stats:
            table.add_row(
                f"{stat.provider} missing",
                f"{stat.missing_fraction:.1%}"
                + (
                    ""
                    if stat.observed_delay_p95_s is None
                    else f", p95 delay {stat.observed_delay_p95_s:.0f}s"
                ),
            )
        if result.cycles_meeting_staleness_pct is not None:
            table.add_row("within staleness bound", f"{result.cycles_meeting_staleness_pct:.1f}%")
        console.print(table)

        mark = OK if result.verdict is Verdict.FREE_DATA_SUFFICIENT else BAD
        console.print(f"\n{mark} [bold]{result.verdict.value}[/bold]")
        console.print(escape(result.rationale), soft_wrap=True)

        allowed, why = may_widen_live_resolutions(
            result, currently_allowed=pinned.limits.data.allowed_live_resolutions
        )
        console.print(f"\n{OK if allowed else WARN} {escape(why)}", soft_wrap=True)

        if result.worst:
            console.print("\n[bold]worst disagreements[/bold] (go and look at these):")
            for pair in result.worst:
                console.print(
                    f"  • {pair.instrument_uid} {pair.bar_open_utc.isoformat()}: "
                    f"{pair.primary_close} vs {pair.secondary_close} "
                    f"({pair.disagreement_bps:.1f}bps)"
                )


# --------------------------------------------------------------------------
# tb data seal / verify / vintages
# --------------------------------------------------------------------------


@data_app.command("seal")
def data_seal(
    limits: LimitsOpt = None,
    db: DbOpt = None,
    bars: RootOpt = None,
    resolution: Annotated[
        str, typer.Option("--resolution", help="daily, hourly or minute.")
    ] = "daily",
) -> None:
    """Freeze the current dataset as a vintage M3 can cite.

    A backtest whose `vintage_id` is not in the ledger is not admissible
    evidence for promotion — because without one, "we backtested this and it
    worked" cannot be checked. Yahoo restates history routinely, so the data a
    result was produced on is genuinely free to move.

    Idempotent: the id is derived from the content, so sealing the same data
    twice returns the same vintage rather than a second indistinguishable one.
    """
    pinned = _load(limits)
    res = _resolution(resolution)
    with _ledger(db, pinned) as ledger:
        store = _store(ledger, pinned, bars)
        snapshots = SnapshotStore(ledger, store, calendar=TradingCalendar(), run_id=new_run_id())
        try:
            vintage = snapshots.seal(resolutions=[res])
        except DataError as exc:
            err_console.print(f"{BAD} {escape(str(exc))}", soft_wrap=True)
            raise typer.Exit(1) from exc

        table = Table(show_header=False, box=None, padding=(0, 2, 0, 0))
        table.add_row("vintage", f"[bold]{vintage.vintage_id}[/bold]")
        table.add_row("manifest", vintage.manifest_hash)
        table.add_row("files", str(vintage.n_files))
        table.add_row("rows", f"{vintage.row_count:,}")
        table.add_row("instruments", str(len(vintage.instrument_uids)))
        table.add_row(
            "window",
            "—"
            if vintage.window_start is None or vintage.window_end is None
            else f"{vintage.window_start.date()} → {vintage.window_end.date()}",
        )
        table.add_row("survivorship", vintage.survivorship_flag.value)
        table.add_row("point-in-time", vintage.pit_completeness_flag.value)
        table.add_row("action table", (vintage.action_table_hash or "—")[:16])
        table.add_row("fx table", (vintage.fx_table_hash or "—")[:16])
        table.add_row("universe snapshot", vintage.universe_snapshot_id or "[yellow]none[/yellow]")
        console.print(table)

        for caveat in vintage.caveats:
            console.print(f"\n{WARN} {escape(caveat)}", soft_wrap=True)

        console.print(
            f"\n{OK} cite [bold]{vintage.vintage_id}[/bold] from any backtest whose result "
            "is meant to support a promotion."
        )


@data_app.command("verify")
def data_verify(
    vintage_id: Annotated[
        str | None, typer.Argument(help="Vintage to re-hash. Default: verify the store.")
    ] = None,
    limits: LimitsOpt = None,
    db: DbOpt = None,
    bars: RootOpt = None,
) -> None:
    """Re-hash a sealed vintage, or check the whole store's provenance.

    The data-layer analogue of `tb ledger verify`, with the same asymmetry: a
    file the catalog promises and cannot produce is a hard failure, while an
    unrecorded file on disk is harmless because nothing reads it.
    """
    pinned = _load(limits)
    with _ledger(db, pinned) as ledger:
        store = _store(ledger, pinned, bars)
        snapshots = SnapshotStore(ledger, store, calendar=TradingCalendar())

        if vintage_id is None:
            findings = store.verify_partitions()
            blocking = [f for f in findings if not f.startswith("ORPHAN")]
            for finding in findings:
                console.print(
                    f"{BAD if not finding.startswith('ORPHAN') else WARN} {escape(finding)}",
                    soft_wrap=True,
                )
            if blocking:
                raise typer.Exit(1)
            console.print(f"{OK} every recorded partition is present and unchanged")
            return

        problems = snapshots.verify(vintage_id)
        for problem in problems:
            err_console.print(f"{BAD} {escape(problem)}", soft_wrap=True)
        if problems:
            err_console.print(
                "\nA vintage that no longer hashes to what was sealed is not the data any "
                "backtest citing it ran on. Every promotion resting on it needs re-running.",
                soft_wrap=True,
            )
            raise typer.Exit(1)
        console.print(f"{OK} {vintage_id} matches what was sealed")


@data_app.command("vintages")
def data_vintages(
    limits: LimitsOpt = None,
    db: DbOpt = None,
    bars: RootOpt = None,
    limit: Annotated[int, typer.Option("-n", "--limit", help="How many to show.")] = 10,
) -> None:
    """List sealed vintages, newest first."""
    pinned = _load(limits)
    with _ledger(db, pinned) as ledger:
        store = _store(ledger, pinned, bars)
        vintages = SnapshotStore(ledger, store).list_vintages(limit=limit)
        if not vintages:
            console.print(f"{WARN} no vintages sealed yet. Run `tb data seal`.")
            return

        table = Table(box=None, padding=(0, 2, 0, 0))
        table.add_column("vintage")
        table.add_column("sealed")
        table.add_column("rows", justify="right")
        table.add_column("files", justify="right")
        table.add_column("survivorship")
        table.add_column("point-in-time")
        for vintage in vintages:
            table.add_row(
                vintage.vintage_id,
                vintage.as_of_utc.date().isoformat(),
                f"{vintage.row_count:,}",
                str(vintage.n_files),
                vintage.survivorship_flag.value,
                vintage.pit_completeness_flag.value,
            )
        console.print(table)


@data_app.command("coverage")
def data_coverage(
    limits: LimitsOpt = None,
    db: DbOpt = None,
    bars: RootOpt = None,
) -> None:
    """Show the `bar_coverage` projection from the last audit.

    A projection, rebuilt from the bars each time rather than incremented — a
    counter that drifts from the files it summarises is worse than no counter,
    because it gets trusted anyway.
    """
    pinned = _load(limits)
    with _ledger(db, pinned) as ledger:
        store = _store(ledger, pinned, bars)
        auditor = DataAuditor(ledger, store)
        rows = auditor.coverage_rows()
        if not rows:
            console.print(f"{WARN} nothing audited yet. Run `tb data audit`.")
            return

        table = Table(box=None, padding=(0, 2, 0, 0))
        table.add_column("instrument")
        table.add_column("res")
        table.add_column("provider")
        table.add_column("from")
        table.add_column("to")
        table.add_column("rows", justify="right")
        table.add_column("unexplained", justify="right")
        for row in rows:
            gaps = int(row["n_gaps_unexplained"] or 0)
            table.add_row(
                str(row["instrument_uid"]),
                str(row["resolution"]),
                str(row["provider"]),
                str(row["first_bar_open"] or "—")[:10],
                str(row["last_bar_open"] or "—")[:10],
                f"{int(row['row_count'] or 0):,}",
                f"[yellow]{gaps}[/yellow]" if gaps else "0",
            )
        console.print(table)
        console.print(f"\n[dim]last audited: {auditor.last_audit_at() or 'never'}[/dim]")


@data_app.command("actions")
def data_actions(
    limits: LimitsOpt = None,
    db: DbOpt = None,
    bars: RootOpt = None,
    provider: Annotated[str, typer.Option("--provider", help="alpaca or yahoo.")] = "yahoo",
    years: Annotated[int, typer.Option("--years", help="How far back to fetch.")] = 10,
    symbols: Annotated[
        str | None, typer.Option("--symbols", help="Comma-separated T212 tickers.")
    ] = None,
) -> None:
    """Fetch corporate actions for the universe.

    Actions are stored as dated, revisable *facts*; factors are derived from
    them on demand. Never the reverse — a materialised adjustment column is how
    a table ends up containing tomorrow's split, and no schema check catches it
    because the number looks perfectly ordinary.
    """
    pinned = _load(limits)
    _check_provider(provider)
    with _ledger(db, pinned) as ledger:
        symbol_map = SymbolMap(ledger, provider=provider)
        instruments = {i.ticker: i for i in cached_instruments(ledger)}
        actions = ActionStore(ledger, run_id=new_run_id())

        if symbols:
            tickers = [t.strip() for t in symbols.split(",") if t.strip()]
        else:
            snapshot = UniverseStore(ledger).latest()
            if snapshot is None:
                err_console.print(
                    f"{BAD} no universe snapshot. Run `tb universe build` first.",
                    soft_wrap=True,
                )
                raise typer.Exit(2)
            tickers = list(snapshot.tickers)

        end = datetime.now(UTC)
        start = end - timedelta(days=365 * years)
        feed = _provider(provider)
        recorded = 0
        superseded = 0
        problems: list[str] = []
        try:
            for ticker in tickers:
                mapping = symbol_map.get(ticker)
                if mapping is None or not mapping.data_symbol:
                    continue
                instrument = instruments.get(ticker)
                uid = make_instrument_uid(
                    isin=None if instrument is None else instrument.isin, t212_ticker=ticker
                )
                try:
                    raw = feed.fetch_actions(
                        mapping.data_symbol, instrument_uid=uid, start=start, end=end
                    )
                except TbError as exc:
                    problems.append(f"{ticker}: {exc}")
                    continue
                result = actions.record_raw(raw)
                recorded += result.recorded
                superseded += len(result.superseded)
                problems.extend(result.rejected)
        finally:
            feed.close()

        table = Table(show_header=False, box=None, padding=(0, 2, 0, 0))
        table.add_row("actions recorded", str(recorded))
        table.add_row("restatements", f"[yellow]{superseded}[/yellow]" if superseded else "0")
        table.add_row("instruments with actions", str(len(actions.instruments_with_actions())))
        console.print(table)
        for problem in problems[:10]:
            console.print(f"  {WARN} {escape(problem)}")
