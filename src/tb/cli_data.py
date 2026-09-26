"""`tb data ...` and `tb universe ...` — the M2 surface.

Read-and-write, but only into the data layer: nothing here can place an order.
The commands correspond one-to-one with the M2 verification steps, in the order
you would actually run them:

    tb universe build          pick the symbols, record a dated snapshot
    tb data backfill           fetch history for them
    tb data audit              classify what is wrong with it
    tb data actions            fetch splits and dividends
    tb data reconcile-actions  check dividends against cash the broker paid
    tb data canary             re-read old history and record restatements
    tb data regime             the exposure factor, and whether it is measured
    tb data bakeoff            measure whether the feed is good enough to trade
    tb data seal               freeze a vintage M3 can cite
    tb data verify             re-hash a sealed vintage

`backfill`, `bakeoff`, `actions`, `canary` and `reconcile-actions` are the ones
that touch a network, and each names which provider it is calling before it
calls it. Everything else reads the local store.
"""

from __future__ import annotations

import os
import sys
from collections.abc import Sequence
from datetime import UTC, date, datetime, timedelta
from decimal import Decimal
from pathlib import Path
from typing import Annotated

import typer
from rich.console import Console
from rich.markup import escape
from rich.table import Table

from tb.broker.reconcile import Severity
from tb.broker.t212.client import T212Client
from tb.broker.t212.probe import cached_instruments
from tb.broker.t212.raw_archive import RawArchive
from tb.config.loader import PinnedLimits, load_hard_limits
from tb.core.errors import TbError
from tb.core.ids import new_run_id
from tb.data.actions import ActionStore
from tb.data.adjustments import SplitSuspicion
from tb.data.audit import DataAuditor, summarise_gaps
from tb.data.bakeoff import Bakeoff, Verdict, may_widen_live_resolutions
from tb.data.barstore import BarStore
from tb.data.calendar import TradingCalendar
from tb.data.canary import RevisionCanary
from tb.data.fx import FxStore, rates_from_bars
from tb.data.provider import (
    DataError,
    MarketDataProvider,
    Provenance,
    Resolution,
    make_instrument_uid,
)
from tb.data.providers import AlpacaProvider, YahooProvider
from tb.data.providers.alpaca import KEY_ID_VAR as ALPACA_KEY_ID_VAR
from tb.data.providers.alpaca import SECRET_VAR as ALPACA_SECRET_VAR
from tb.data.regime import RegimeGate, RegimeState
from tb.data.snapshot import SnapshotStore
from tb.data.symbols import SymbolMap
from tb.data.universe import (
    UniverseStore,
    build_candidates,
    dollar_volume_from_bars,
    select,
)
from tb.ledger.events import Actor, EventType, RegimeReadPayload
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


def _provider(name: str, *, archive: RawArchive | None = None) -> MarketDataProvider:
    """Build a validated provider, resolving credentials where it needs them.

    `archive` is threaded through rather than constructed here because it needs
    a ledger, and the provider factory is deliberately usable without one (for
    `tb data bakeoff --help`, for instance). Passing it is what makes the
    payloads replayable; before this was wired, `tb data backfill` archived
    nothing at all and a Yahoo shape change left no forensic record.
    """
    _check_provider(name)
    if name == "yahoo":
        return YahooProvider(archive=archive)
    try:
        return AlpacaProvider.from_env(archive=archive)
    except DataError as exc:
        err_console.print(f"{BAD} {escape(str(exc))}", soft_wrap=True)
        raise typer.Exit(2) from exc


def _broker_client(ledger: Ledger, run_id: str) -> T212Client:
    """A read-only broker client, or a refusal naming the missing key.

    Imported inside the function so every offline `tb data` command stays
    importable without broker credentials — the data layer must not require a
    broker key to audit a store.
    """
    from tb.broker.t212.client import T212Client
    from tb.broker.t212.ratelimit import RateGovernor

    return T212Client.from_env(
        governor=RateGovernor(state_path=Path(ledger.path).parent / "run" / "ratelimit.json"),
        ledger=ledger,
        run_id=run_id,
    )


def _archive(ledger: Ledger, provider: str, run_id: str | None = None) -> RawArchive:
    """A provider archive with the credential scrubbed.

    Alpaca sends its key in headers, which never reach `record()`, so this is
    defence in depth rather than the primary control: a 401 body that echoed
    the key back would otherwise be archived verbatim. Yahoo needs it more
    directly, since its query parameters can carry tokens.
    """
    secrets = tuple(
        value
        for value in (
            os.environ.get(ALPACA_KEY_ID_VAR),
            os.environ.get(ALPACA_SECRET_VAR),
        )
        if value
    )
    return RawArchive.for_provider(ledger, provider=provider, run_id=run_id, redact_values=secrets)


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
    data_symbols: Annotated[
        str | None,
        typer.Option(
            "--data-symbols",
            help="Research only: provider symbols, bypassing the symbol map. Not tradable.",
        ),
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

    `--symbols` takes *Trading 212* tickers and resolves them through the symbol
    map, which is what the trading path uses. `--data-symbols` takes provider
    symbols directly and skips the map entirely. That exists because the
    bake-off measures **data providers**, not the broker, and requiring a
    broker-derived symbol map to run it was a coupling that made the
    measurement impossible anywhere without Trading 212 credentials — including
    CI, which is the only place with unrestricted egress here.

    Bars fetched that way are keyed `sym:SYMBOL` rather than by ISIN. That is
    deliberately self-marking: the trading path resolves an instrument through
    the symbol map to an `isin:` or `t212:` uid and will never match a `sym:`
    one, so research bars cannot become trading inputs by accident, and
    `tb data audit` reports them as research-only identities.
    """
    pinned = _load(limits)
    res = _resolution(resolution)
    _check_provider(provider)
    with _ledger(db, pinned) as ledger:
        store = _store(ledger, pinned, bars)
        symbol_map = SymbolMap(ledger, provider=provider)
        instruments = {i.ticker: i for i in cached_instruments(ledger)}

        # (ticker-or-symbol, data symbol, instrument_uid). Built up front so the
        # fetch loop does not branch on which flag produced it.
        targets: list[tuple[str, str, str]] = []
        problems: list[str] = []
        if symbols and data_symbols:
            err_console.print(
                f"{BAD} pass --symbols or --data-symbols, not both: they key bars under "
                "different identities, and mixing them in one run would split an "
                "instrument's history across two uids.",
                soft_wrap=True,
            )
            raise typer.Exit(2)

        if data_symbols:
            for raw in data_symbols.split(","):
                symbol = raw.strip().upper()
                if symbol:
                    targets.append((symbol, symbol, make_instrument_uid(data_symbol=symbol)))
            console.print(
                f"{WARN} research mode: {len(targets)} symbol(s) keyed by ticker, not ISIN. "
                "Not tradable, and not point-in-time.",
                soft_wrap=True,
            )
        else:
            if symbols:
                tickers = [t.strip() for t in symbols.split(",") if t.strip()]
            else:
                snapshot = UniverseStore(ledger).latest()
                if snapshot is None:
                    err_console.print(
                        f"{BAD} no universe snapshot. Run `tb universe build` first, or pass "
                        "--symbols (Trading 212 tickers) or --data-symbols (provider symbols, "
                        "research only).",
                        soft_wrap=True,
                    )
                    raise typer.Exit(2)
                tickers = list(snapshot.tickers)
            for ticker in tickers:
                mapping = symbol_map.get(ticker)
                if mapping is None or not mapping.data_symbol:
                    problems.append(f"{ticker}: no data symbol mapped")
                    continue
                instrument = instruments.get(ticker)
                targets.append(
                    (
                        ticker,
                        mapping.data_symbol,
                        make_instrument_uid(
                            isin=None if instrument is None else instrument.isin,
                            t212_ticker=ticker,
                        ),
                    )
                )

        span = (
            timedelta(days=365 * (years or pinned.limits.data.backfill_years_daily))
            if res is Resolution.DAILY
            else timedelta(days=days or pinned.limits.data.backfill_days_minute)
        )
        end = datetime.now(UTC)
        start = end - span

        run_id = new_run_id()
        feed = _provider(provider, archive=_archive(ledger, provider, run_id))
        console.print(
            f"fetching {res.value} bars for {len(targets)} symbol(s) from "
            f"[bold]{provider}[/bold], {start.date()} to {end.date()}"
        )

        total = 0
        revisions = 0
        # Counted separately from `problems`, because a warning is not a
        # failure and the two must not be added together. `n_fetched` is the
        # number of symbols that produced a batch at all — the signal that
        # tells "the feed is unreachable" apart from "the store was already up
        # to date", which both write zero rows.
        n_fetched = 0
        first_failure = ""
        try:
            for label, data_symbol, uid in targets:
                try:
                    batch = feed.fetch_bars(
                        data_symbol,
                        instrument_uid=uid,
                        resolution=res,
                        start=start,
                        end=end,
                        provenance=Provenance.BACKFILL,
                    )
                except TbError as exc:
                    problems.append(f"{label}: {exc}")
                    first_failure = first_failure or f"{label}: {exc}"
                    continue

                n_fetched += 1
                result = store.ingest(batch)
                total += result.rows_written
                revisions += len(result.revisions)
                problems.extend(f"{label}: {w}" for w in batch.warnings)

            if fx:
                total += _backfill_fx(ledger, feed, pinned, start=start, end=end, problems=problems)
        finally:
            feed.close()

        sealed = store.compact() if seal else []

        table = Table(show_header=False, box=None, padding=(0, 2, 0, 0))
        table.add_row("symbols fetched", f"{n_fetched}/{len(targets)}")
        table.add_row("rows written", str(total))
        table.add_row("revisions detected", f"[yellow]{revisions}[/yellow]" if revisions else "0")
        table.add_row("partitions sealed", str(len(sealed)))
        table.add_row("warnings", f"[yellow]{len(problems)}[/yellow]" if problems else "0")
        console.print(table)

        for problem in problems[:15]:
            console.print(f"  {WARN} {escape(problem)}")
        if len(problems) > 15:
            console.print(f"  [dim]… and {len(problems) - 15} more[/dim]")

        # A backfill that fetched nothing at all is not a success, and saying
        # so here is what keeps a pipeline honest: the first run of the
        # bake-off workflow reported success having written zero rows, because
        # every Yahoo request was throttled and this command still exited 0.
        # The next step then failed for "no bars in the store", four steps
        # away from the actual cause.
        #
        # Zero *rows* on its own is fine — a re-run of an up-to-date store
        # writes none. Zero *symbols fetched* is not.
        if targets and n_fetched == 0:
            err_console.print(
                f"\n{BAD} no symbol could be fetched from {provider}: "
                f"all {len(targets)} request(s) failed. The store is unchanged.",
                soft_wrap=True,
            )
            if first_failure:
                err_console.print(f"  first failure: {escape(first_failure)}", soft_wrap=True)
            raise typer.Exit(2)

        if n_fetched < len(targets):
            console.print(
                f"\n{WARN} {len(targets) - n_fetched} of {len(targets)} symbol(s) returned "
                "nothing. Anything measured over this store covers the rest only.",
                soft_wrap=True,
            )

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
            # Recorded and *acted on*, not advised about. The plan's rule is
            # "halt entries on that symbol", and a report telling an operator
            # to block something by hand is not a control — the next run of the
            # loop would enter before anyone read it.
            recorded, blocked = _act_on_suspicions(ledger, report.suspicions)
            console.print(
                f"\n{BAD} {len(report.suspicions)} unexplained split(s) detected: "
                f"{recorded} recorded as inferred actions, {blocked} symbol(s) blocked "
                "from new entries until a provider confirms the action.",
                soft_wrap=True,
            )
            console.print(
                "  The inferred ratio is evidence, never something to adjust prices by: "
                "sizing a position off a guessed ratio is worse than not trading it.",
                soft_wrap=True,
            )
            if blocked < len(report.suspicions):
                console.print(
                    f"  {WARN} {len(report.suspicions) - blocked} could not be blocked — no "
                    "mapped Trading 212 ticker for that instrument. It is unreachable by "
                    "the trading path anyway, but nothing will stop it if that changes.",
                    soft_wrap=True,
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

        # The primary is the feed under test, so its absence is a setup
        # problem: there is nothing to measure. A missing *secondary* is
        # different — three of the four outputs (no-print fraction, observed
        # delay, share of cycles inside the staleness bound) are properties of
        # one feed and were already measured. Refusing outright used to
        # discard them.
        if not primary_bars:
            err_console.print(
                f"{BAD} no {res.value} bars from {primary} in the store, so there is "
                f"nothing to measure. Run `tb data backfill --provider {primary} "
                f"--resolution {res.value}`.",
                soft_wrap=True,
            )
            raise typer.Exit(2)

        single_feed = not secondary_bars
        if single_feed:
            console.print(
                f"{WARN} no {res.value} bars from [bold]{secondary}[/bold]: the cross-feed "
                "disagreement cannot be computed. Reporting what one feed establishes on "
                "its own.",
                soft_wrap=True,
            )

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

        # Exit 1 on anything but a clean pass, so this can gate a pipeline.
        # Previously it printed the verdict and exited 0 whatever it said,
        # which is the same masking as a `| tee` without `pipefail`: the
        # measurement reported a problem and the caller saw success. 1 rather
        # than 2, because the exit-code contract here reserves 2 for a setup
        # problem the operator must fix, and a feed that is not good enough is
        # a finding about the data.
        if result.verdict is not Verdict.FREE_DATA_SUFFICIENT:
            raise typer.Exit(1)


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
        actions_run_id = new_run_id()
        actions = ActionStore(ledger, run_id=actions_run_id)

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
        run_id = new_run_id()
        feed = _provider(provider, archive=_archive(ledger, provider, run_id))
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


# --------------------------------------------------------------------------
# tb data canary
# --------------------------------------------------------------------------


@data_app.command("canary")
def data_canary(
    limits: LimitsOpt = None,
    db: DbOpt = None,
    bars: RootOpt = None,
    provider: Annotated[str, typer.Option("--provider", help="alpaca or yahoo.")] = "yahoo",
    resolution: Annotated[str, typer.Option("--resolution", help="daily, hourly or minute.")] = (
        "daily"
    ),
    report: Annotated[
        bool, typer.Option("--report", help="Show coverage history without fetching.")
    ] = False,
) -> None:
    """Re-read a slice of stored history and record what the vendor now says.

    Revision detection only fires on bars that get refetched, and the
    incremental poll only refetches the last few sessions. A restatement
    twenty days back is therefore *invisible* — nothing in the system would
    ever look there — and Yahoo back-adjusts historical OHLC as a matter of
    course, including for actions it never reports.

    Windows are selected least-recently-verified first rather than at random,
    so coverage accumulates instead of re-rolling the same dice, and the oldest
    unverified history is reached first. `data.revision_canary_sample_pct` is
    the per-run budget.
    """
    pinned = _load(limits)
    res = _resolution(resolution)
    _check_provider(provider)

    with _ledger(db, pinned) as ledger:
        store = _store(ledger, pinned, bars)
        canary_run_id = new_run_id()
        canary = RevisionCanary(
            ledger,
            store,
            sample_pct=pinned.limits.data.revision_canary_sample_pct,
            run_id=canary_run_id,
        )

        if report:
            _canary_report(canary, res)
            return

        available = canary.windows(resolution=res)
        if not available:
            console.print(
                f"{WARN} nothing stored at {res.value} resolution, so there is no "
                "history to re-verify. Run `tb data backfill` first.",
                soft_wrap=True,
            )
            return

        console.print(
            f"re-verifying {canary.budget(len(available))} of {len(available)} "
            f"{res.value} window(s) via [bold]{provider}[/bold], "
            "least-recently-checked first"
        )
        feed = _provider(provider, archive=_archive(ledger, provider, canary_run_id))
        try:
            result = canary.run(feed, resolution=res)
        finally:
            feed.close()

        table = Table(show_header=False, box=None, padding=(0, 2, 0, 0))
        table.add_row("windows checked", f"{result.n_windows_checked}/{result.n_windows_available}")
        table.add_row("coverage this run", f"{result.coverage_pct:.1f}%")
        table.add_row("bars compared", str(result.n_bars_compared))
        table.add_row(
            "restatements found",
            f"[yellow]{result.n_revisions_found}[/yellow]" if result.n_revisions_found else "0",
        )
        if result.oldest_checked_age_days is not None:
            table.add_row("oldest checked", f"{result.oldest_checked_age_days:.0f} days unverified")
        console.print(table)

        for uid in result.instruments_restated():
            console.print(
                f"  {WARN} {escape(uid)} was restated. Any backtest citing the live "
                "store over that window is no longer reproducible; a sealed vintage "
                "is unaffected, which is what vintages are for.",
                soft_wrap=True,
            )
        for problem in result.problems[:10]:
            console.print(f"  {WARN} {escape(problem)}", soft_wrap=True)

        if result.clean:
            console.print(
                f"\n{OK} no restatements in the windows checked. That is a statement "
                f"about {result.coverage_pct:.0f}% of stored history, not all of it.",
                soft_wrap=True,
            )


def _canary_report(canary: RevisionCanary, res: Resolution) -> None:
    """Coverage history: which windows have been verified, and how often."""
    rows = canary.coverage(resolution=res)
    if not rows:
        console.print(
            f"{WARN} the canary has never run at {res.value} resolution. Every stored "
            "bar older than the incremental poll's reach is unverified.",
            soft_wrap=True,
        )
        return
    table = Table(show_header=True)
    table.add_column("instrument")
    table.add_column("window from")
    table.add_column("checks", justify="right")
    table.add_column("last checked")
    table.add_column("restatements", justify="right")
    for row in rows[:25]:
        found = int(row["total_revisions"] or 0)
        table.add_row(
            str(row["instrument_uid"]),
            str(row["window_start"])[:10],
            str(row["n_checks"]),
            str(row["last_checked_at"])[:19],
            f"[yellow]{found}[/yellow]" if found else "0",
        )
    console.print(table)
    console.print(f"\n[dim]last canary run: {canary.last_run_at() or 'never'}[/dim]")


# --------------------------------------------------------------------------
# tb data regime
# --------------------------------------------------------------------------


@data_app.command("regime")
def data_regime(
    limits: LimitsOpt = None,
    db: DbOpt = None,
    bars: RootOpt = None,
    as_of: Annotated[
        str | None,
        typer.Option("--as-of", help="ISO instant to read as of. Default: now."),
    ] = None,
) -> None:
    """The one factor that scales every strategy's exposure at once.

    Long-only and unlevered means every live strategy is a long-equity beta
    expression, so in a drawdown their correlation goes to one and per-strategy
    caps stop helping exactly when they are needed. This gate sits above the
    allocator: below the reference index's long moving average, all gross
    exposure is scaled by `regime.exposure_factor_below_ma`.

    Run it before arming anything. The states that matter are the two that are
    *not* a market signal — `insufficient_history` and `unavailable` — because
    both mean the gate cannot see the index, and both still reduce exposure. A
    fresh install is in one of them for roughly ten months, and this command
    exists so that is a number an operator has read rather than a surprise.

    Exit 1 when the reading is unmeasured, so a deployment check can gate on
    it. `risk_off` exits 0: the index being below its average is the gate
    working, not a fault.
    """
    pinned = _load(limits)
    moment = _instant(as_of)

    with _ledger(db, pinned) as ledger:
        store = _store(ledger, pinned, bars)
        gate = RegimeGate(limits=pinned.limits)
        reading = gate.read(
            store,
            as_of=moment,
            actions=ActionStore(ledger).actions_for(gate.instrument_uid),
        )

        # Recorded on every read, not only on a change. The factor that was in
        # force at a decision time has to be recoverable from the ledger
        # alone, and a reading only written on transitions cannot answer that
        # for the instants in between.
        with ledger.transaction() as tx:
            tx.append(
                EventType.DATA_REGIME_READ,
                reading.instrument_uid,
                RegimeReadPayload(
                    state=reading.state.value,
                    exposure_factor=reading.exposure_factor,
                    reference_symbol=reading.reference_symbol,
                    instrument_uid=reading.instrument_uid,
                    ma_days=reading.ma_days,
                    n_sessions_seen=reading.n_sessions_seen,
                    is_measured=reading.state.is_measured,
                    as_of=reading.as_of.isoformat(),
                    last_close=reading.last_close,
                    moving_average=reading.moving_average,
                    detail=reading.detail,
                ),
                actor=Actor.SYSTEM,
                run_id=new_run_id(),
            )

        table = Table(show_header=False, box=None, padding=(0, 2, 0, 0))
        table.add_row("reference", f"{reading.reference_symbol} ({reading.instrument_uid})")
        table.add_row("as of", reading.as_of.isoformat())
        colour = "green" if reading.state is RegimeState.RISK_ON else "yellow"
        table.add_row("state", f"[{colour}]{reading.state.value}[/{colour}]")
        table.add_row(
            "exposure factor",
            f"x{reading.exposure_factor}" if reading.reduced else "x1 (full)",
        )
        table.add_row("sessions seen", f"{reading.n_sessions_seen} (needs {gate.min_sessions})")
        if reading.last_close is not None:
            table.add_row("last close", str(reading.last_close))
        if reading.moving_average is not None:
            table.add_row(f"{reading.ma_days}-day average", str(reading.moving_average))
        console.print(table)
        console.print(f"\n{escape(reading.detail)}", soft_wrap=True)

        if reading.state.is_measured:
            console.print(f"\n{OK} the gate is measuring the market.")
            return

        # The actionable half. An unmeasured gate is not an error to be cleared
        # by ignoring it — it is the reason exposure is reduced, and the only
        # fix is more history for the reference series.
        console.print(
            f"\n{WARN} this reading is not a market signal. Exposure stays at "
            f"x{reading.exposure_factor} until the gate can see "
            f"{gate.min_sessions} sessions of {reading.reference_symbol}.",
            soft_wrap=True,
        )
        console.print(
            f"  backfill the reference series with: [bold]tb data backfill "
            f"--data-symbols {reading.reference_symbol} --resolution daily "
            f"--years 2[/bold]",
            soft_wrap=True,
        )
        raise typer.Exit(1)


def _instant(text: str | None) -> datetime:
    """Parse an as-of instant, refusing a naive one.

    A naive datetime here would be interpreted as UTC by arithmetic further
    down while the operator meant local time, which silently shifts the
    visibility boundary by up to a day.
    """
    if text is None:
        return datetime.now(UTC)
    try:
        parsed = datetime.fromisoformat(text)
    except ValueError as exc:
        raise typer.BadParameter(
            f"{text!r} is not an ISO instant. Try 2026-03-02T00:00:00Z."
        ) from exc
    if parsed.tzinfo is None:
        raise typer.BadParameter(
            f"{text!r} has no timezone. An as-of instant without one would be read "
            "as UTC while you meant local time, moving the visibility boundary by "
            "up to a day. Append Z or an offset."
        )
    return parsed


# --------------------------------------------------------------------------
# tb data reconcile-actions
# --------------------------------------------------------------------------


@data_app.command("reconcile-actions")
def data_reconcile_actions(
    limits: LimitsOpt = None,
    db: DbOpt = None,
    limit: Annotated[int, typer.Option("--limit", help="Broker dividend records to read.")] = 50,
) -> None:
    """Check provider dividends against cash Trading 212 actually paid.

    The highest-value identity check in the data layer, and the only one that
    can catch the failure the symbol map exists to prevent. If a provider
    reports a dividend and the broker credited no cash for a position held
    through the ex-date, either the action data is wrong *or the symbol map
    points at a different company than the one in the account*. Every other
    check in this layer compares our data against our data; this one compares
    it against money that moved.

    Needs the broker, so it needs a key. Position spans come from
    `positions_snapshot` — the reconciler's own output — because a dividend on
    a stock we never held is not expected and counting those as mismatches
    would bury the one case that matters under the whole universe.

    With no position history at all the result is **inconclusive, not clean**:
    without knowing what was held, "no credit" and "no credit expected" are
    indistinguishable, and reporting that as a pass would be the wrong answer
    in the permissive direction.
    """
    pinned = _load(limits)
    with _ledger(db, pinned) as ledger:
        instruments = {i.ticker: i for i in cached_instruments(ledger)}
        actions = ActionStore(ledger, run_id=new_run_id())
        with_actions = set(actions.instruments_with_actions())
        if not with_actions:
            console.print(
                f"{WARN} no corporate actions stored, so there is nothing to reconcile. "
                "Run [bold]tb data actions[/bold] first.",
                soft_wrap=True,
            )
            return

        spans = _held_spans(ledger)
        if not spans:
            console.print(
                f"{WARN} no position history in `positions_snapshot`, so this check is "
                "[bold]inconclusive rather than clean[/bold]: without knowing what was "
                "held, a missing credit and a credit that was never due look identical. "
                "Run [bold]tb reconcile[/bold] while holding a position first.",
                soft_wrap=True,
            )
            raise typer.Exit(1)

        try:
            client = _broker_client(ledger, new_run_id())
        except TbError as exc:
            err_console.print(f"{BAD} {escape(str(exc))}", soft_wrap=True)
            raise typer.Exit(2) from exc

        try:
            credits = client.get_dividends(limit=limit)
        except TbError as exc:
            err_console.print(
                f"{BAD} could not read broker dividends: {escape(str(exc))}", soft_wrap=True
            )
            raise typer.Exit(2) from exc
        finally:
            client.close()

        by_ticker: dict[str, list[tuple[date, Decimal]]] = {}
        undated = 0
        for record in credits:
            if not record.ticker or record.amount is None or not record.paid_on:
                # Never coerce a missing amount to zero: it would reconcile as
                # "paid nothing" and mask the exact mismatch being looked for.
                undated += 1
                continue
            try:
                paid = datetime.fromisoformat(record.paid_on.replace("Z", "+00:00")).date()
            except ValueError:
                undated += 1
                continue
            by_ticker.setdefault(record.ticker, []).append((paid, record.amount))

        matched = 0
        unexplained: list[str] = []
        skipped = 0
        for ticker, instrument in sorted(instruments.items()):
            uid = make_instrument_uid(isin=instrument.isin, t212_ticker=ticker)
            if uid not in with_actions:
                continue
            results = actions.reconcile_dividends(
                uid,
                broker_credits=by_ticker.get(ticker, []),
                held_through=spans.get(ticker, ()),
            )
            for outcome in results:
                if not outcome.matched:
                    unexplained.append(f"{ticker} {outcome.effective_date}: {outcome.detail}")
                elif "no position held" in outcome.detail:
                    skipped += 1
                else:
                    matched += 1

        table = Table(show_header=False, box=None, padding=(0, 2, 0, 0))
        table.add_row("broker credits read", str(len(credits)))
        table.add_row("unusable credit records", f"[yellow]{undated}[/yellow]" if undated else "0")
        table.add_row("dividends matched", f"[green]{matched}[/green]" if matched else "0")
        table.add_row("not expected (never held)", str(skipped))
        table.add_row("UNEXPLAINED", f"[red]{len(unexplained)}[/red]" if unexplained else "0")
        console.print(table)

        for note in unexplained[:10]:
            console.print(f"  {BAD} {escape(note)}", soft_wrap=True)

        if unexplained:
            err_console.print(
                f"\n{BAD} {len(unexplained)} dividend(s) had no matching broker credit on a "
                "position that was held through the ex-date. Either the action data is "
                "wrong or the symbol map points at a different company — check "
                "[bold]tb symbols show[/bold] for the affected tickers before trading them.",
                soft_wrap=True,
            )
            raise typer.Exit(1)
        console.print(f"\n{OK} every expected dividend matches a broker credit.")


def _held_spans(ledger: Ledger) -> dict[str, tuple[tuple[date, date], ...]]:
    """When each ticker was actually held, from the reconciler's snapshots.

    `(initial_fill_date, snapshot_ts)` per row with a positive quantity. Coarse
    on purpose: it is evidence that the position existed across that span, not
    a claim that it existed at no other time. Under-claiming is the safe
    direction — a dividend wrongly treated as "not expected" is skipped rather
    than flagged, and an over-claimed span would manufacture mismatches for
    stock that was never owned.
    """
    rows = ledger.conn.execute(
        "SELECT ticker, quantity, initial_fill_date, ts FROM positions_snapshot "
        "WHERE initial_fill_date IS NOT NULL"
    ).fetchall()
    found: dict[str, list[tuple[date, date]]] = {}
    for row in rows:
        try:
            if Decimal(str(row["quantity"])) <= 0:
                continue
            start = datetime.fromisoformat(
                str(row["initial_fill_date"]).replace("Z", "+00:00")
            ).date()
            end = datetime.fromisoformat(str(row["ts"]).replace("Z", "+00:00")).date()
        except (ValueError, ArithmeticError):
            continue
        if end >= start:
            found.setdefault(str(row["ticker"]), []).append((start, end))
    return {ticker: tuple(spans) for ticker, spans in found.items()}


def _act_on_suspicions(ledger: Ledger, suspicions: Sequence[SplitSuspicion]) -> tuple[int, int]:
    """Record each inferred split and block the symbol it belongs to.

    Two separate consequences, both required by the plan's rule. Recording
    makes the inference a durable fact the factor algebra can be asked to
    exclude; blocking is what actually stops an entry. Doing only the first
    would leave the loop free to trade a series the audit just called into
    question.

    The uid-to-ticker direction needs the instruments cache, because an
    `isin:` uid carries no ticker — which is why this lives in the CLI rather
    than in `DataAuditor`. The data audit must stay runnable without a broker.
    """
    actions = ActionStore(ledger, run_id=new_run_id())
    by_uid: dict[str, str] = {}
    for instrument in cached_instruments(ledger):
        by_uid[make_instrument_uid(isin=instrument.isin, t212_ticker=instrument.ticker)] = (
            instrument.ticker
        )

    recorded = 0
    blocked = 0
    for suspicion in suspicions:
        result = actions.record_suspicion(suspicion)
        if result.recorded:
            recorded += 1
        ticker = by_uid.get(suspicion.instrument_uid)
        if ticker is None:
            continue
        for provider in KNOWN_PROVIDERS:
            symbol_map = SymbolMap(ledger, provider=provider)
            if symbol_map.get(ticker) is None:
                continue
            symbol_map.block_for(
                ticker,
                # `description` rather than a second format string: it already
                # reads as "N-for-M split within X%", and a reason an operator
                # has to decode is a reason they will override.
                reason=f"unexplained split — {suspicion.description}",
            )
            blocked += 1
            break
    return recorded, blocked
