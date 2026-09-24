"""`tb research ...` — the trial log and the single holdout evaluation.

    tb research trials     the trial log, and the multiplicity it implies
    tb research holdout    spend a strategy's one holdout evaluation
    tb research null-gate  measure the gate's false-promotion rate

`tb research holdout` is the command with consequences. A version gets exactly
one evaluation, enforced by `UNIQUE (strategy_id, version)`, so the command
refuses a second attempt rather than quietly overwriting — and it says what the
first answer was, because the caller's correct response is to treat that as
final rather than to retry.

The research surface holds no broker credential and can place no order. It is
expected to run in a process without `T212_LIVE_API_KEY` in its environment at
all.
"""

from __future__ import annotations

import os
import sys
from datetime import timedelta
from pathlib import Path
from typing import Annotated

import typer
from rich.console import Console
from rich.markup import escape
from rich.table import Table

from tb.backtest.costs import CostModel, Jurisdiction
from tb.backtest.engine import Backtester, InstrumentMeta
from tb.backtest.metrics import returns_of
from tb.config.loader import PinnedLimits, load_hard_limits
from tb.core.errors import TbError
from tb.data.barstore import BarStore
from tb.data.provider import Resolution
from tb.data.snapshot import SnapshotStore
from tb.ledger.store import Ledger, default_ledger_path
from tb.registry.lineage import SpecRegistry
from tb.registry.promotion import PromotionGate
from tb.research.holdout import (
    DEFAULT_HOLDOUT_FRACTION,
    HoldoutAlreadyEvaluated,
    HoldoutRegistry,
    evaluation_reader,
    holdout_boundary,
)
from tb.research.null_gate import measure_false_promotion_rate
from tb.research.trials import TrialLog
from tb.strategy.dsl.ops import DslStrategy, pipeline_from_spec

research_app = typer.Typer(
    help="Trials, multiplicity and the sealed holdout.", no_args_is_help=True
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


# --------------------------------------------------------------------------
# tb research trials
# --------------------------------------------------------------------------


@research_app.command("trials")
def trials(
    limits: LimitsOpt = None,
    db: DbOpt = None,
    lineage: Annotated[
        str | None, typer.Option("--lineage", help="Filter to one lineage.", show_default=False)
    ] = None,
    search: Annotated[
        str | None, typer.Option("--search", help="Filter to one search.", show_default=False)
    ] = None,
    limit: Annotated[int, typer.Option("--limit", help="How many to show.")] = 20,
) -> None:
    """The trial log, and the multiplicity haircut it implies.

    The counts are the point. Deflated Sharpe divides out the size of the
    search, so this table is the denominator of every promotion decision — and
    a searcher whose rejections were never logged would have its haircut
    computed from the survivors alone.
    """
    pinned = _load(limits)
    with _ledger(db, pinned) as ledger:
        log = TrialLog(ledger)
        if lineage is not None:
            rows = log.trials_in_lineage(lineage)[-limit:]
        elif search is not None:
            rows = log.trials_in_search(search)[-limit:]
        else:
            raw = ledger.conn.execute(
                "SELECT search_id FROM trials ORDER BY recorded_at DESC LIMIT 1"
            ).fetchone()
            if raw is None:
                console.print(f"{WARN} no trials recorded yet")
                return
            rows = log.trials_in_search(str(raw["search_id"]))[-limit:]

        if not rows:
            console.print(f"{WARN} no trials match")
            return

        table = Table(show_header=True)
        table.add_column("trial")
        table.add_column("lineage")
        table.add_column("outcome")
        table.add_column("net Sharpe", justify="right")
        table.add_column("trades", justify="right")
        table.add_column("in lineage", justify="right")
        table.add_column("in search", justify="right")
        table.add_column("why refused")
        for trial in rows:
            table.add_row(
                trial.trial_id,
                trial.lineage_id,
                _outcome_markup(trial.outcome.value),
                "—" if trial.net_sharpe is None else f"{trial.net_sharpe:.2f}",
                "—" if trial.n_trades is None else str(trial.n_trades),
                str(trial.trials_in_lineage_at_time),
                str(trial.trials_in_search_at_time),
                trial.rejection_reason or "—",
            )
        console.print(table)

        newest = rows[-1]
        multiplicity = log.multiplicity_for(newest.spec_hash)
        if multiplicity is None:
            return
        console.print(
            f"  newest trial is deflated against {multiplicity.n_trials} trial(s) "
            f"at a Sharpe dispersion of {multiplicity.sharpe_dispersion:.3f}"
        )
        for caveat in multiplicity.caveats:
            console.print(f"  {WARN} {caveat}")


def _outcome_markup(outcome: str) -> str:
    if outcome == "passed_gate":
        return "[green]passed_gate[/green]"
    if outcome == "errored":
        return "[red]errored[/red]"
    if outcome == "rejected":
        return "[yellow]rejected[/yellow]"
    return outcome


# --------------------------------------------------------------------------
# tb research holdout
# --------------------------------------------------------------------------


@research_app.command("holdout")
def holdout(
    strategy_id: Annotated[str, typer.Argument(help="The strategy to evaluate.")],
    vintage_id: Annotated[str, typer.Argument(help="The sealed vintage to evaluate against.")],
    limits: LimitsOpt = None,
    db: DbOpt = None,
    bars: RootOpt = None,
    version: Annotated[int, typer.Option("--version", help="Spec version.")] = 1,
    fraction: Annotated[
        float, typer.Option("--fraction", help="Share of the window held back.")
    ] = DEFAULT_HOLDOUT_FRACTION,
    apply: Annotated[
        bool,
        typer.Option(
            "--apply/--dry-run",
            help="Record the evaluation. A version gets exactly one.",
        ),
    ] = False,
) -> None:
    """Spend a strategy's one holdout evaluation.

    `--dry-run` by default, because the recording is irreversible: the
    uniqueness constraint means there is no second attempt, and a holdout that
    could be re-evaluated is not a holdout but a slower training set.

    Exits 1 when the strategy fails the holdout, 2 when it cannot be evaluated
    at all — an unsealed vintage, a window too short, or an evaluation already
    spent.
    """
    pinned = _load(limits)
    with _ledger(db, pinned) as ledger:
        registry = SpecRegistry(
            ledger, per_lineage_budget_ccy=pinned.limits.loss.per_lineage_budget_ccy
        )
        registered = registry.get(strategy_id, version)
        spec = registry.spec_of(strategy_id, version)
        if registered is None or spec is None:
            err_console.print(f"{BAD} {strategy_id}@v{version} is not registered")
            raise typer.Exit(2)

        holdouts = HoldoutRegistry(ledger)
        prior = holdouts.existing(strategy_id, version)
        if prior is not None:
            err_console.print(
                f"{BAD} {strategy_id}@v{version} already spent its holdout evaluation on "
                f"{prior.evaluated_at.isoformat()[:19]} and "
                f"{'passed' if prior.passed else 'failed'}. Register the change as a new "
                "strategy: its lineage carries this one's trial count into the "
                "multiplicity haircut, which is the cost of the second attempt."
            )
            raise typer.Exit(2)

        store = BarStore(
            ledger,
            root=bars or (Path(ledger.path).parent / "bars"),
            scale=pinned.limits.data.price_scale,
        )
        snapshots = SnapshotStore(ledger, store)
        admissible, why = snapshots.is_admissible(vintage_id)
        if not admissible:
            err_console.print(f"{BAD} {escape(why)}", soft_wrap=True)
            raise typer.Exit(2)
        vintage = snapshots.get(vintage_id)
        if vintage is None:  # pragma: no cover - is_admissible just found it
            err_console.print(f"{BAD} {vintage_id} vanished between checks")
            raise typer.Exit(2)

        try:
            window = holdout_boundary(vintage, fraction=fraction)
        except TbError as exc:
            err_console.print(f"{BAD} {escape(str(exc))}", soft_wrap=True)
            raise typer.Exit(2) from exc
        console.print(f"  {window.summary()}")
        if not window.is_usable:
            err_console.print(
                f"{BAD} the holdout is only {window.holdout_days} day(s) long. "
                "Out-of-sample statistics over that are noise, and promoting on them "
                "while believing there was an independent check is the failure this "
                "whole mechanism exists to prevent."
            )
            raise typer.Exit(2)

        source = snapshots.source_for(vintage_id)
        uids = tuple(vintage.instrument_uids)
        reader = evaluation_reader(
            source,
            resolution=Resolution.DAILY,
            instrument_uids=uids,
            lookback=timedelta(days=max(400, spec.max_lookback * 2)),
        )
        schedule = [
            bar.available_at_utc + timedelta(hours=1)
            for bar in sorted(
                (b for b in snapshots.bars_of(vintage_id) if b.bar_open_utc >= window.sealed_from),
                key=lambda b: b.bar_open_utc,
            )
        ]
        if len(schedule) < 2:
            err_console.print(f"{BAD} the holdout window holds fewer than two bars")
            raise typer.Exit(2)

        engine = Backtester(
            cost_model=CostModel(pinned.limits),
            pipeline=pipeline_from_spec(spec),
            instruments={uid: InstrumentMeta(uid, "USD", Jurisdiction.US) for uid in uids},
            min_holding_minutes=spec.min_holding_minutes,
        )
        result = engine.run(
            strategy=DslStrategy(spec=spec, strategy_id=strategy_id, version=version),
            reader=reader,
            decision_times=schedule,
            resolution=Resolution.DAILY,
        )

        promotion = pinned.limits.promotion
        passed = (
            result.metrics.n_trades >= promotion.min_oos_trades
            and result.metrics.net_sharpe is not None
            and result.metrics.max_drawdown_pct <= promotion.max_oos_drawdown_pct
        )
        console.print(f"  {result.metrics.summary()}")
        for caveat in (*result.caveats, *vintage.caveats):
            console.print(f"  {WARN} {caveat}")

        if not apply:
            console.print(
                f"{WARN} dry run: nothing recorded. Re-run with [bold]--apply[/bold] to "
                "spend this version's one evaluation."
            )
            raise typer.Exit(0 if passed else 1)

        try:
            recorded = holdouts.record(
                strategy_id=strategy_id,
                version=version,
                lineage_id=registered.lineage_id,
                spec_hash=registered.spec_hash,
                vintage_id=vintage_id,
                sealed_from=window.sealed_from,
                window_start=window.sealed_from,
                window_end=window.holdout_end,
                backtest_id=result.backtest_id,
                passed=passed,
                n_trades=result.metrics.n_trades,
                net_sharpe=result.metrics.net_sharpe,
                net_return_pct=float(result.metrics.net_return_pct),
                max_drawdown_pct=float(result.metrics.max_drawdown_pct),
                cost_drag_bps=float(result.metrics.cost_drag_bps),
                returns=[
                    float(value)
                    for value in returns_of([point.equity_ccy for point in result.curve])
                ],
                detail=(
                    "; ".join(result.caveats) if result.caveats else "no caveats from the backtest"
                ),
            )
        except HoldoutAlreadyEvaluated as exc:
            err_console.print(f"{BAD} {escape(str(exc))}", soft_wrap=True)
            raise typer.Exit(2) from exc

    if passed:
        console.print(f"{OK} {recorded.label} passed the holdout ({recorded.evaluation_id})")
        return
    err_console.print(
        f"{BAD} {recorded.label} failed the holdout. This is terminal for this version: "
        "a change is a new strategy, and its lineage carries this one's trial count."
    )
    raise typer.Exit(1)


# --------------------------------------------------------------------------
# tb research null-gate
# --------------------------------------------------------------------------


@research_app.command("null-gate")
def null_gate(
    limits: LimitsOpt = None,
    db: DbOpt = None,
    specs: Annotated[int, typer.Option("--specs", help="How many random specs to draw.")] = 200,
    days: Annotated[int, typer.Option("--days", help="Length of the fixture, in bars.")] = 400,
    seed: Annotated[int, typer.Option("--seed", help="Seed, so a failure reproduces.")] = 20260601,
) -> None:
    """Measure the gate's false-promotion rate against a population with no edge.

    The same measurement the release-gate test makes, through the same function
    — because the thresholds it depends on live in a file a human edits, and
    after editing one the operator should be able to ask what it did rather than
    reading the test suite. Run this after any change to `promotion.*` or
    `costs.*`.

    Exits 1 if the observed rate is over `promotion.max_null_promotion_rate`.
    With no paper-shadow period that rate is the number of noise strategies that
    reach real money.

    The default of 200 specs is coarser than the suite's 1,000 and the output
    says so rather than presenting it as the same number.
    """
    pinned = _load(limits)
    console.print(f"  backtesting {specs} random specs over {days} bars…")
    with _ledger(db, pinned) as ledger:
        gate = PromotionGate(ledger, limits=pinned.limits)
        try:
            result = measure_false_promotion_rate(
                gate=gate, limits=pinned.limits, n_specs=specs, days=days, seed=seed
            )
        except ValueError as exc:
            err_console.print(f"{BAD} {escape(str(exc))}", soft_wrap=True)
            raise typer.Exit(2) from exc

    console.print(f"  {result.summary()}")
    if result.refusals_by_gate:
        table = Table(show_header=True, title="which gates refused")
        table.add_column("gate")
        table.add_column("refusals", justify="right")
        for name, count in sorted(result.refusals_by_gate.items(), key=lambda item: -item[1]):
            table.add_row(name, str(count))
        console.print(table)

    if not result.is_informative:
        console.print(
            f"{WARN} only {result.n_reached_statistics} of {result.n_specs} candidates "
            "reached the statistical checks with measured numbers, so this rate says more "
            "about the fixture than about the gate. Lengthen --days."
        )

    ceiling = pinned.limits.promotion.max_null_promotion_rate
    if result.rate > ceiling:
        err_console.print(
            f"{BAD} {result.n_promoted} of {result.n_specs} random specs promoted "
            f"({result.rate:.2%}), over the {ceiling:.1%} ceiling."
        )
        raise typer.Exit(1)
    console.print(f"{OK} false-promotion rate {result.rate:.2%}, within the {ceiling:.1%} ceiling")
