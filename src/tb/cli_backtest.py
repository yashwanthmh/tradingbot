"""`tb backtest ...` — run a strategy, or interrogate the engine itself.

Two commands with different subjects, which is why they are not one.

`tb backtest calibrate` asks whether the *engine* can be trusted: it runs a
population of strategies with no edge by construction and fails if any of them
earned one. It is a release gate. Nothing above it means anything if it fails,
which is why it exits non-zero and says why.

`tb backtest costs` asks what the fee schedule does to a strategy before any
strategy exists. It is the fastest way to see why this project targets
hour-to-day position changes rather than minute-by-minute ones.

Both write their verdict into the ledger. A calibration that was run but not
recorded cannot be cited by M5's promotion gate, and a gate that cannot check
whether the engine was calibrated is not a gate.
"""

from __future__ import annotations

import contextlib
import json
import os
import sys
from datetime import datetime, timedelta
from decimal import Decimal
from pathlib import Path
from typing import Annotated

import typer
from rich.console import Console
from rich.markup import escape
from rich.table import Table

from tb.backtest.calibrate import CalibrationResult, run_calibration
from tb.backtest.costs import CostModel, Jurisdiction, jurisdiction_from_isin
from tb.backtest.engine import InstrumentMeta
from tb.config.loader import PinnedLimits, load_hard_limits
from tb.core.clock import now_iso
from tb.core.errors import TbError
from tb.core.ids import new_run_id
from tb.data.asof import InMemoryBarSource
from tb.data.provider import Resolution
from tb.data.snapshot import SnapshotStore
from tb.features.pipeline import default_pipeline
from tb.ledger.events import Actor, CalibrationPayload, EventType
from tb.ledger.store import Ledger, default_ledger_path

backtest_app = typer.Typer(
    help="Run backtests, and check that the backtester can be trusted.",
    no_args_is_help=True,
)

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


def _resolution(name: str) -> Resolution:
    try:
        return Resolution(name.lower())
    except ValueError as exc:
        err_console.print(f"{BAD} unknown resolution {name!r}: daily, hourly or minute.")
        raise typer.Exit(2) from exc


# --------------------------------------------------------------------------
# tb backtest costs
# --------------------------------------------------------------------------


@backtest_app.command("costs")
def backtest_costs(
    limits: LimitsOpt = None,
    notional: Annotated[
        str, typer.Option("--notional", help="Order size in the account currency.")
    ] = "1000",
) -> None:
    """What a round trip costs, per jurisdiction, and the edge it demands.

    The whole venue analysis in one table. Run this before believing any
    backtest: if the required gross edge is larger than the edge the signal
    class actually has, the strategy was never possible and no amount of
    searching will find one.
    """
    pinned = _load(limits)
    lim = pinned.limits
    model = CostModel(lim)
    size = Decimal(notional)
    ratio = Decimal(str(lim.execution.max_cost_to_edge_ratio))

    table = Table(title=f"round trip on {size} {lim.currency}", title_justify="left")
    table.add_column("instrument")
    table.add_column("ccy")
    table.add_column("FX", justify="right")
    table.add_column("tax", justify="right")
    table.add_column("spread", justify="right")
    table.add_column("slip", justify="right")
    table.add_column("round trip", justify="right")
    table.add_column("edge needed", justify="right")

    rows = [
        ("US large-cap", "USD", Jurisdiction.US),
        ("UK share", lim.currency, Jurisdiction.UK),
        ("Irish share", "EUR", Jurisdiction.IRELAND),
        ("French share", "EUR", Jurisdiction.FRANCE),
        ("German share", "EUR", Jurisdiction.OTHER),
    ]
    for label, ccy, jurisdiction in rows:
        trip = model.round_trip(
            notional_ccy=size, instrument_currency=ccy, jurisdiction=jurisdiction
        )
        needed = trip.total_bps / ratio
        entry = trip.entry
        table.add_row(
            label,
            ccy,
            f"{entry.fx_fee_ccy}",
            f"{entry.transaction_tax_ccy}",
            f"{entry.half_spread_ccy}",
            f"{entry.slippage_ccy}",
            f"[bold]{trip.total_bps:.1f}bps[/bold]",
            f"[bold]{needed:.0f}bps[/bold]",
        )
    console.print(table)
    console.print(
        f"\nEdge needed = round trip / [bold]{ratio}[/bold] "
        f"(execution.max_cost_to_edge_ratio).\n"
        "Gross edge on minute-bar signals in liquid names is [bold]5-20bps[/bold]. "
        "That gap is why this system targets minute-resolution features with "
        "hour-to-day position changes, and it is enforced as a pre-trade "
        "rejection rather than left to judgement.",
        soft_wrap=True,
    )
    console.print(
        f"\nA strategy may declare between {lim.costs.min_expected_edge_bps}bps and "
        f"{lim.costs.max_expected_edge_bps}bps of edge. The ceiling is not tuning: the "
        "gate divides by the declared number, so an unbounded claim would defeat it.",
        soft_wrap=True,
    )


# --------------------------------------------------------------------------
# tb backtest calibrate
# --------------------------------------------------------------------------


@backtest_app.command("calibrate")
def backtest_calibrate(
    limits: LimitsOpt = None,
    db: DbOpt = None,
    bars: RootOpt = None,
    vintage: Annotated[
        str | None,
        typer.Option("--vintage", help="Sealed vintage to calibrate against."),
    ] = None,
    resolution: Annotated[str, typer.Option("--resolution", help="daily, hourly or minute.")] = (
        "daily"
    ),
    seed: Annotated[int, typer.Option("--seed", help="RNG seed for the coin flips.")] = 7,
    tolerance: Annotated[
        float, typer.Option("--tolerance", help="Net Sharpe a null may show.")
    ] = 0.5,
) -> None:
    """Prove the backtester cannot manufacture an edge.

    Runs strategies with no edge by construction and fails if any earned one.
    A positive net Sharpe from a coin flip is a fill-timing error, a mark taken
    from a bar the position could not see, or a cost charged on one leg — never
    a discovery.

    Exits 1 on failure, because nothing built on an uncalibrated engine means
    anything.
    """
    pinned = _load(limits)
    res = _resolution(resolution)

    with _ledger(db, pinned) as ledger:
        source, uids, vintage_id = _load_source(ledger, pinned, bars, vintage, res)
        if len(uids) == 0:
            err_console.print(
                f"{BAD} no instruments in the store for {res.value}. Run "
                "`tb data backfill` first — a calibration over no data would pass "
                "every check while proving nothing.",
                soft_wrap=True,
            )
            raise typer.Exit(2)

        times = _decision_times(source, uids, res)
        if len(times) < 3:
            err_console.print(
                f"{BAD} only {len(times)} decision times available. A calibration needs "
                "enough sessions for a Sharpe to exist at all.",
                soft_wrap=True,
            )
            raise typer.Exit(2)

        instruments = {
            uid: InstrumentMeta(
                instrument_uid=uid,
                currency="USD",
                # From the uid's own ISIN where it has one. A `sym:` uid has no
                # ISIN, so it is costed as US — stated here rather than hidden,
                # since a research-keyed instrument is not a trading input.
                jurisdiction=_jurisdiction_of(uid),
            )
            for uid in uids
        }

        console.print(
            f"calibrating over {len(uids)} instrument(s), {len(times)} decision times, "
            f"{res.value} bars" + (f", vintage {vintage_id}" if vintage_id else " (live store)")
        )

        result = run_calibration(
            cost_model=CostModel(pinned.limits),
            source=source,
            instruments=instruments,
            decision_times=times,
            resolution=res,
            pipeline=default_pipeline(),
            tolerance=tolerance,
            seed=seed,
        )

        _print_calibration(result)
        _record(ledger, result, vintage_id=vintage_id or "", pinned=pinned)

        if not result.passed:
            err_console.print(
                f"\n{BAD} calibration FAILED. The backtester cannot be trusted, so no "
                "result it has produced is evidence for anything.",
                soft_wrap=True,
            )
            raise typer.Exit(1)
        console.print(
            f"\n{OK} calibration passed: no null strategy earned an edge, and costs "
            "reached the equity curve."
        )


def _jurisdiction_of(uid: str) -> Jurisdiction:
    if uid.startswith("isin:"):
        found = jurisdiction_from_isin(uid.removeprefix("isin:"))
        # A research or ticker-keyed uid carries no issuer country. Treated as
        # US rather than UNKNOWN so a calibration can run at all, and reported
        # as a caveat rather than silently assumed correct.
        return Jurisdiction.US if found is Jurisdiction.UNKNOWN else found
    return Jurisdiction.US


def _load_source(
    ledger: Ledger,
    pinned: PinnedLimits,
    bars: Path | None,
    vintage: str | None,
    res: Resolution,
) -> tuple[InMemoryBarSource, tuple[str, ...], str | None]:
    """Read from a sealed vintage where one is named, else from the live store."""
    from tb.data.barstore import BarStore

    store = BarStore(
        ledger,
        root=bars or (Path(ledger.path).parent / "bars"),
        scale=pinned.limits.data.price_scale,
    )
    if vintage:
        snapshots = SnapshotStore(ledger, store)
        admissible, reason = snapshots.is_admissible(vintage)
        if not admissible:
            err_console.print(f"{BAD} {escape(reason)}", soft_wrap=True)
            raise typer.Exit(2)
        source = snapshots.source_for(vintage)
        uids = tuple(sorted({bar.instrument_uid for bar in source.bars}))
        return source, uids, vintage

    # A damaged store must be *diagnosed*, not raised through as a pyarrow
    # traceback about missing magic bytes. That error names a file and a
    # library, neither of which tells an operator that the store is corrupt or
    # what to do — and it reads like a backtester bug. `verify_partitions`
    # already answers this question, so the failure points at the store.
    try:
        everything = [bar for uid in store.instruments() for bar in store.bars_for(uid, res)]
    except Exception as exc:
        findings: list[str] = []
        # Suppressed deliberately: the store is already known unreadable, and
        # the point of this branch is to report the *first* failure usefully.
        # A second exception from the diagnostic would replace a good message
        # with a worse one.
        with contextlib.suppress(Exception):  # pragma: no branch
            findings = list(store.verify_partitions())
        err_console.print(
            f"{BAD} the bar store could not be read: {escape(str(exc))}",
            soft_wrap=True,
        )
        for finding in findings[:5]:
            err_console.print(f"  {WARN} {escape(finding)}", soft_wrap=True)
        err_console.print(
            "Run [bold]tb data audit[/bold] to see what the catalog and the files "
            "disagree about. A calibration over an unreadable store is not a "
            "statement about the engine.",
            soft_wrap=True,
        )
        raise typer.Exit(2) from exc

    source = InMemoryBarSource(bars=everything)
    return source, tuple(sorted({bar.instrument_uid for bar in everything})), None


def _decision_times(
    source: InMemoryBarSource, uids: tuple[str, ...], res: Resolution
) -> list[datetime]:
    """One decision per distinct knowledge time in the fixture.

    Derived from `available_at`, not from bar time: the decision schedule has
    to be the moments at which new information existed, or the engine would be
    asked to decide before it could see anything.
    """
    moments = sorted({bar.available_at_utc for bar in source.bars if bar.resolution is res})
    return [moment + timedelta(minutes=1) for moment in moments]


def _print_calibration(result: CalibrationResult) -> None:
    table = Table(title="null strategies", title_justify="left")
    table.add_column("strategy")
    table.add_column("trades", justify="right")
    table.add_column("gross %", justify="right")
    table.add_column("net %", justify="right")
    table.add_column("gross Sharpe", justify="right")
    table.add_column("net Sharpe", justify="right")
    table.add_column("drag bps", justify="right")

    for run in result.results:
        m = run.metrics
        net = "—" if m.net_sharpe is None else f"{m.net_sharpe:.2f}"
        gross = "—" if m.gross_sharpe is None else f"{m.gross_sharpe:.2f}"
        over = m.net_sharpe is not None and m.net_sharpe > result.tolerance
        table.add_row(
            run.strategy_id,
            str(m.n_trades),
            f"{m.gross_return_pct:.2f}",
            f"{m.net_return_pct:.2f}",
            gross,
            f"[red]{net}[/red]" if over else net,
            f"{m.cost_drag_bps:.1f}",
        )
    console.print(table)
    console.print(f"\n{result.summary()}")
    for failure in result.failures:
        console.print(f"  {BAD} {escape(failure)}", soft_wrap=True)


def _record(
    ledger: Ledger,
    result: CalibrationResult,
    *,
    vintage_id: str,
    pinned: PinnedLimits,
) -> None:
    """Write the verdict to the ledger and its projection table.

    Both in one transaction. A calibration recorded in the projection but not
    the chain would be a claim about the engine with nothing backing it.
    """
    # `tx.append` / `tx.execute`, not `ledger.append`: the latter opens its own
    # transaction, and nesting it inside this one raises "cannot start a
    # transaction within a transaction". Both writes belong in one transaction
    # so a projection row cannot exist without the chain entry that justifies it.
    with ledger.transaction() as tx:
        event = tx.append(
            EventType.BACKTEST_CALIBRATED,
            result.calibration_id,
            CalibrationPayload(
                calibration_id=result.calibration_id,
                vintage_id=vintage_id,
                resolution=result.resolution.value,
                n_strategies=result.n_strategies,
                n_runs=result.n_runs,
                rng_seed=result.rng_seed,
                worst_net_sharpe=result.worst_net_sharpe,
                best_net_sharpe=result.best_net_sharpe,
                mean_net_sharpe=result.mean_net_sharpe,
                mean_cost_drag_bps=float(result.mean_cost_drag_bps),
                tolerance=result.tolerance,
                passed=result.passed,
                failures=list(result.failures),
            ),
            actor=Actor.SYSTEM,
            run_id=new_run_id(),
        )
        tx.execute(
            """
            INSERT INTO backtest_calibrations (
                calibration_id, vintage_id, resolution, code_git_sha, n_strategies,
                n_runs, rng_seed, worst_net_sharpe, best_net_sharpe, mean_net_sharpe,
                mean_cost_drag_bps, tolerance, passed, failures_json, ran_at,
                completing_event_seq
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                result.calibration_id,
                vintage_id,
                result.resolution.value,
                pinned.config_hash,
                result.n_strategies,
                result.n_runs,
                result.rng_seed,
                result.worst_net_sharpe,
                result.best_net_sharpe,
                result.mean_net_sharpe,
                float(result.mean_cost_drag_bps),
                result.tolerance,
                int(result.passed),
                json.dumps(list(result.failures)),
                now_iso(),
                event.seq,
            ),
        )


# --------------------------------------------------------------------------
# tb backtest calibrations
# --------------------------------------------------------------------------


@backtest_app.command("calibrations")
def list_calibrations(
    limits: LimitsOpt = None,
    db: DbOpt = None,
    limit: Annotated[int, typer.Option("--limit", help="How many to show.")] = 10,
) -> None:
    """Recent calibration runs.

    M5's promotion gate reads the newest one for the running code version: a
    backtest from an uncalibrated engine is not admissible evidence, for the
    same reason a backtest with no `vintage_id` is not.
    """
    pinned = _load(limits)
    with _ledger(db, pinned) as ledger:
        rows = ledger.conn.execute(
            "SELECT * FROM backtest_calibrations ORDER BY ran_at DESC LIMIT ?", (limit,)
        ).fetchall()
        if not rows:
            console.print(
                f"{WARN} no calibrations recorded. Run [bold]tb backtest calibrate[/bold]."
            )
            return
        table = Table(show_header=True)
        table.add_column("calibration")
        table.add_column("ran")
        table.add_column("res")
        table.add_column("nulls", justify="right")
        table.add_column("highest net Sharpe", justify="right")
        table.add_column("drag bps", justify="right")
        table.add_column("verdict")
        for row in rows:
            best = row["best_net_sharpe"]
            table.add_row(
                row["calibration_id"],
                str(row["ran_at"])[:19],
                row["resolution"],
                str(row["n_strategies"]),
                "—" if best is None else f"{best:.2f}",
                f"{row['mean_cost_drag_bps']:.1f}",
                "[green]passed[/green]" if row["passed"] else "[red]FAILED[/red]",
            )
        console.print(table)
