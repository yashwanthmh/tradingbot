"""`tb registry ...`, `tb promote ...`, `tb allocator ...` — the M5 surface.

In the order the pipeline runs them:

    tb registry register    put a spec in the registry as a candidate
    tb registry list        every strategy, its status and its rung
    tb registry lineage     one lineage's ancestry and its loss budget
    tb registry review      KEEP / KILL / ITERATE / SCALE on the live book
    tb promote evaluate     run the gate, printing every check
    tb promote history      what the gate decided before, and why
    tb allocator explain    the prior/realised shrinkage behind each allocation
    tb allocator ladder     every rung change, up and down

Nothing here places an order. `tb promote evaluate` can *promote* — which makes
a strategy eligible for the loop to fund at floor notional — and that is as
close to money as this surface gets.

Exit codes follow the contract the rest of the CLI uses: **2** is a setup
problem the operator must fix, **1** is a finding in the data, **0** is clean.
So `tb promote evaluate` exits 1 on a refusal — a refusal is a finding, not a
failure of the command — and 2 when the evidence needed to decide is missing
altogether.
"""

from __future__ import annotations

import json
import os
import sys
from decimal import Decimal
from pathlib import Path
from typing import Annotated

import typer
from rich.console import Console
from rich.markup import escape
from rich.table import Table

from tb.backtest.costs import Jurisdiction
from tb.config.loader import PinnedLimits, load_hard_limits
from tb.core.errors import TbError
from tb.data.snapshot import SnapshotStore
from tb.ledger.store import Ledger, default_ledger_path
from tb.portfolio.allocator import Allocator
from tb.portfolio.decay import ReviewCycle, ReviewInput, Verdict
from tb.registry.ladder import SizeLadder, notional_for
from tb.registry.lineage import SpecRegistry
from tb.registry.models import AuthorKind, RegistryError
from tb.registry.promotion import (
    Decision,
    EvidenceBuilder,
    PromotionGate,
    latest_calibration,
)
from tb.research.holdout import HoldoutRegistry
from tb.research.selection import deflate, probability_of_backtest_overfitting
from tb.research.trials import TrialLog, returns_matrix
from tb.strategy.dsl.schema import SpecError, StrategySpec

registry_app = typer.Typer(
    help="Register strategies and inspect their lineage.", no_args_is_help=True
)
promote_app = typer.Typer(help="Run the promotion gate.", no_args_is_help=True)
allocator_app = typer.Typer(help="Explain capital allocation.", no_args_is_help=True)

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


def _registry(ledger: Ledger, pinned: PinnedLimits) -> SpecRegistry:
    return SpecRegistry(ledger, per_lineage_budget_ccy=pinned.limits.loss.per_lineage_budget_ccy)


# --------------------------------------------------------------------------
# tb registry register
# --------------------------------------------------------------------------


@registry_app.command("register")
def register(
    spec_file: Annotated[Path, typer.Argument(help="A JSON file holding one strategy spec.")],
    limits: LimitsOpt = None,
    db: DbOpt = None,
    author: Annotated[
        str, typer.Option("--author", help="human, search, llm or mutation.")
    ] = "human",
    parent: Annotated[
        str | None,
        typer.Option("--parent", help="Parent strategy id, for a mutation.", show_default=False),
    ] = None,
) -> None:
    """Register a spec as a candidate.

    Registering promotes nothing. A registered spec is a `candidate` and only
    `tb promote evaluate` can change that, which is what makes "how did this get
    funded" have exactly one answer.
    """
    pinned = _load(limits)
    try:
        payload = json.loads(spec_file.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        err_console.print(f"{BAD} cannot read {spec_file}: {escape(str(exc))}", soft_wrap=True)
        raise typer.Exit(2) from exc
    try:
        spec = StrategySpec.parse(payload)
        kind = AuthorKind(author.lower())
    except (SpecError, ValueError) as exc:
        err_console.print(f"{BAD} {escape(str(exc))}", soft_wrap=True)
        raise typer.Exit(2) from exc

    with _ledger(db, pinned) as ledger:
        try:
            registered = _registry(ledger, pinned).register(
                spec, author_kind=kind, parent_strategy_id=parent
            )
        except RegistryError as exc:
            err_console.print(f"{BAD} {escape(str(exc))}", soft_wrap=True)
            raise typer.Exit(2) from exc

    console.print(
        f"{OK} {registered.label} registered in lineage {registered.lineage_id} "
        f"(generation {registered.generation}, {kind.value})"
    )
    console.print(f"  spec hash {registered.spec_hash[:16]}…, status candidate")


# --------------------------------------------------------------------------
# tb registry list
# --------------------------------------------------------------------------


@registry_app.command("list")
def list_strategies(
    limits: LimitsOpt = None,
    db: DbOpt = None,
    status: Annotated[
        str | None,
        typer.Option("--status", help="Filter to one status.", show_default=False),
    ] = None,
) -> None:
    """Every registered strategy, with its status, rung and realised record."""
    pinned = _load(limits)
    with _ledger(db, pinned) as ledger:
        sql = "SELECT * FROM strategy_status"
        params: tuple[object, ...] = ()
        if status is not None:
            sql += " WHERE status = ?"
            params = (status.lower(),)
        rows = ledger.conn.execute(sql + " ORDER BY updated_at DESC", params).fetchall()
        if not rows:
            console.print(f"{WARN} nothing registered. Run [bold]tb registry register[/bold].")
            return

        table = Table(show_header=True)
        table.add_column("strategy")
        table.add_column("lineage")
        table.add_column("status")
        table.add_column("rung", justify="right")
        table.add_column("notional", justify="right")
        table.add_column("trades", justify="right")
        table.add_column("realised", justify="right")
        for row in rows:
            rung = int(row["rung"])
            table.add_row(
                f"{row['strategy_id']}@v{row['version']}",
                str(row["lineage_id"]),
                _status_markup(str(row["status"])),
                str(rung),
                str(notional_for(rung, limits=pinned.limits)),
                str(row["n_realised_trades"]),
                str(row["realised_pnl_ccy"]),
            )
        console.print(table)
        console.print(
            "  notional shown at the rung's own size, before the per-position cap "
            "that equity imposes at run time"
        )


def _status_markup(status: str) -> str:
    if status == "promoted":
        return "[green]promoted[/green]"
    if status in ("retired", "blocked"):
        return f"[red]{status}[/red]"
    return f"[yellow]{status}[/yellow]"


# --------------------------------------------------------------------------
# tb registry lineage
# --------------------------------------------------------------------------


@registry_app.command("lineage")
def show_lineage(
    lineage_id: Annotated[str, typer.Argument(help="The lineage to describe.")],
    limits: LimitsOpt = None,
    db: DbOpt = None,
) -> None:
    """One lineage: its members, its trial count and its loss budget.

    Exits 1 when the budget is exhausted — a finding, since every member is
    blocked and a child registered afterwards inherits it.
    """
    pinned = _load(limits)
    with _ledger(db, pinned) as ledger:
        registry = _registry(ledger, pinned)
        members = registry.in_lineage(lineage_id)
        if not members:
            err_console.print(f"{BAD} no lineage {lineage_id} in the registry")
            raise typer.Exit(2)

        table = Table(show_header=True)
        table.add_column("strategy")
        table.add_column("gen", justify="right")
        table.add_column("parent")
        table.add_column("author")
        table.add_column("edge bps", justify="right")
        table.add_column("registered")
        for spec in members:
            table.add_row(
                spec.label,
                str(spec.generation),
                spec.parent_strategy_id or "—",
                spec.author_kind.value,
                "—" if spec.expected_edge_bps is None else str(spec.expected_edge_bps),
                spec.registered_at.isoformat()[:19],
            )
        console.print(table)

        trials = TrialLog(ledger)
        console.print(f"  trials recorded in this lineage: {trials.count_in_lineage(lineage_id)}")

        budget = registry.budget_for(lineage_id)
        if budget is None:
            console.print(f"{WARN} no budget row for {lineage_id}")
            return
        console.print(f"  budget: {budget.summary()}")
        if budget.is_exhausted:
            err_console.print(
                f"{BAD} this lineage has spent its loss budget: every member is blocked, "
                "and a child registered after this point inherits the exhaustion"
            )
            raise typer.Exit(1)
        console.print(f"{OK} budget open, {budget.remaining_ccy} remaining")


# --------------------------------------------------------------------------
# tb registry review
# --------------------------------------------------------------------------


@registry_app.command("review")
def review(
    limits: LimitsOpt = None,
    db: DbOpt = None,
    apply: Annotated[
        bool,
        typer.Option("--apply/--dry-run", help="Act on the verdicts: retire and drop rungs."),
    ] = False,
) -> None:
    """KEEP / KILL / ITERATE / SCALE over every promoted strategy.

    Asymmetric by design. KILL needs little evidence, because at floor size the
    cost of retiring a good strategy is a missed opportunity and the cost of
    keeping a bad one is money. SCALE needs the trade count the promotion gate
    demanded of a backtest, because disbelieving a backtest on 29 trades and
    then believing a live record on 8 would be incoherent.

    **`realised_edge_bps` is computed against the strategy's current rung
    notional**, which understates the edge of one that has ratcheted up — the
    conservative direction for SCALE, and stated rather than hidden. A strategy
    with no allocation on record has no measurable realised edge, and the review
    reads that as "not measured" rather than as zero.

    Exits 1 when any verdict is KILL or ITERATE — a finding worth a non-zero
    code so it can sit in a scheduled job.
    """
    pinned = _load(limits)
    with _ledger(db, pinned) as ledger:
        registry = _registry(ledger, pinned)
        live = registry.promoted()
        if not live:
            console.print(f"{WARN} nothing is promoted, so there is nothing to review")
            return

        allocator = Allocator(ledger, limits=pinned.limits)
        candidates = []
        for record in live:
            spec = registry.get(record.strategy_id, record.version)
            declared = (
                spec.expected_edge_bps
                if spec is not None and spec.expected_edge_bps is not None
                else Decimal(0)
            )
            budget = registry.budget_for(record.lineage_id)
            history = allocator.history_for(record.strategy_id, limit=1)
            notional = history[0].notional_ccy if history else None
            candidates.append(
                ReviewInput(
                    strategy_id=record.strategy_id,
                    version=record.version,
                    lineage_id=record.lineage_id,
                    n_realised_trades=record.n_realised_trades,
                    realised_pnl_ccy=record.realised_pnl_ccy,
                    declared_edge_bps=declared,
                    realised_edge_bps=_realised_edge(record, notional),
                    lineage_budget_ccy=None if budget is None else budget.budget_ccy,
                    lineage_consumed_ccy=None if budget is None else budget.consumed_ccy,
                )
            )

        reviews = ReviewCycle(ledger).run(candidates, record=apply)

        table = Table(show_header=True)
        table.add_column("strategy")
        table.add_column("verdict")
        table.add_column("trades", justify="right")
        table.add_column("realised", justify="right")
        table.add_column("edge bps", justify="right")
        table.add_column("evidence")
        for result in reviews:
            table.add_row(
                result.label,
                _verdict_markup(result.verdict.value),
                str(result.n_realised_trades),
                str(result.realised_pnl_ccy),
                "—" if result.realised_edge_bps is None else f"{result.realised_edge_bps:.1f}",
                "sufficient" if result.evidence_sufficient else "thin",
            )
        console.print(table)
        for result in reviews:
            for reason in result.reasons:
                console.print(f"  {result.label}: {reason}")

        retiring = [r for r in reviews if r.verdict.retires_the_strategy]
        if apply:
            for result in retiring:
                registry.retire(
                    result.strategy_id,
                    version=result.version,
                    reason=f"{result.verdict.value}: {result.reasons[0] if result.reasons else ''}",
                )
            scaling = [r for r in reviews if r.verdict is Verdict.SCALE]
            if scaling:
                console.print(
                    f"{WARN} {len(scaling)} strategy/strategies earned SCALE. The rung is "
                    "moved by the ladder on its own schedule, not here — a review must "
                    "not be a second way to enlarge a position."
                )
        elif retiring:
            console.print(
                f"{WARN} dry run: {len(retiring)} strategy/strategies would be retired. "
                "Re-run with [bold]--apply[/bold]."
            )

    if retiring:
        raise typer.Exit(1)


def _realised_edge(record: object, notional: Decimal | None) -> Decimal | None:
    """Realised edge in bps, or `None` when it cannot be measured.

    `None` rather than zero when there is no allocation on record or no trades.
    A strategy whose edge has not been measured is not a strategy with no edge,
    and the review's SCALE path depends on the difference.
    """
    trades = getattr(record, "n_realised_trades", 0)
    pnl = getattr(record, "realised_pnl_ccy", Decimal(0))
    if notional is None or notional <= 0 or trades <= 0:
        return None
    return pnl / (Decimal(trades) * notional) * Decimal(10_000)


def _verdict_markup(verdict: str) -> str:
    if verdict == "scale":
        return "[green]SCALE[/green]"
    if verdict == "kill":
        return "[red]KILL[/red]"
    if verdict == "iterate":
        return "[yellow]ITERATE[/yellow]"
    return "KEEP"


# --------------------------------------------------------------------------
# tb promote evaluate
# --------------------------------------------------------------------------


@promote_app.command("evaluate")
def evaluate(
    strategy_id: Annotated[str, typer.Argument(help="The strategy to evaluate.")],
    limits: LimitsOpt = None,
    db: DbOpt = None,
    bars: RootOpt = None,
    version: Annotated[int, typer.Option("--version", help="Spec version.")] = 1,
    vintage: Annotated[
        str | None,
        typer.Option(
            "--vintage",
            help="Sealed vintage the evidence came from.",
            show_default=False,
        ),
    ] = None,
    feed_noise_bps: Annotated[
        float | None,
        typer.Option(
            "--feed-noise-bps",
            help="p95 cross-provider disagreement, from `tb data bakeoff`.",
            show_default=False,
        ),
    ] = None,
    jurisdiction: Annotated[
        str, typer.Option("--jurisdiction", help="Whose transaction tax applies.")
    ] = "us",
    currency: Annotated[
        str, typer.Option("--currency", help="The instrument's own currency.")
    ] = "USD",
    apply: Annotated[
        bool,
        typer.Option("--apply/--dry-run", help="Record the decision and promote on a pass."),
    ] = False,
) -> None:
    """Run every gate and print pass/fail, observed and threshold for each.

    `--dry-run` is the default. Promotion is the moment a generated strategy
    becomes eligible for real money, and with no shadow period behind it the
    command that does that should have to be asked explicitly.

    Exits 1 on a refusal, 0 on a promotion, and 2 when the evidence needed to
    decide cannot be assembled at all.
    """
    pinned = _load(limits)
    with _ledger(db, pinned) as ledger:
        registry = _registry(ledger, pinned)
        registered = registry.get(strategy_id, version)
        if registered is None:
            err_console.print(
                f"{BAD} {strategy_id}@v{version} is not registered. "
                "Run [bold]tb registry register[/bold] first."
            )
            raise typer.Exit(2)
        spec = registry.spec_of(strategy_id, version)
        if spec is None:  # pragma: no cover - a row without its spec_json
            err_console.print(f"{BAD} {strategy_id}@v{version} has no stored spec")
            raise typer.Exit(2)

        holdout = HoldoutRegistry(ledger).existing(strategy_id, version)
        trials = TrialLog(ledger)
        multiplicity = trials.multiplicity_for(registered.spec_hash)

        deflation = None
        if holdout is not None and holdout.net_sharpe is not None and multiplicity is not None:
            deflation = deflate(
                observed_sharpe=holdout.net_sharpe,
                returns=list(holdout.returns),
                n_trials=multiplicity.n_trials,
                sharpe_dispersion=multiplicity.sharpe_dispersion,
                periods_per_year=periods_per_year_for(holdout.resolution),
                dispersion_measured=multiplicity.dispersion_measured,
            )

        peers = trials.trials_in_lineage(registered.lineage_id)
        latest = trials.latest_for_spec(registered.spec_hash)
        if latest is not None and latest.trials_in_search_at_time > len(peers):
            peers = trials.trials_in_search(latest.search_id)
        pbo = probability_of_backtest_overfitting(returns_matrix(peers))

        vintage_id = vintage or (holdout.vintage_id if holdout is not None else None)
        admissible, why = _vintage_check(ledger, pinned, bars, vintage_id)

        builder = EvidenceBuilder(ledger=ledger, registry=registry)
        evidence = builder.build(
            spec=spec,
            lineage_id=registered.lineage_id,
            holdout=holdout,
            multiplicity=multiplicity,
            deflation=deflation,
            pbo=pbo,
            vintage_id=vintage_id,
            vintage_admissible=admissible,
            vintage_caveats=(() if admissible else (why,)),
            feed_noise_p95_bps=(None if feed_noise_bps is None else Decimal(str(feed_noise_bps))),
            jurisdiction=_jurisdiction(jurisdiction),
            instrument_currency=currency.upper(),
        )

        gate = PromotionGate(ledger, limits=pinned.limits)
        decision = gate.evaluate(
            strategy_id=strategy_id,
            version=version,
            evidence=evidence,
            record=apply,
        )

    console.print(decision.report())
    calibration = None
    with _ledger(db, pinned) as ledger:
        calibration = latest_calibration(ledger)
    if calibration.passed is None:
        console.print(
            f"{WARN} no calibration on record: run [bold]tb backtest calibrate[/bold], "
            "because a backtest from an unchecked engine is not admissible evidence"
        )

    if decision.decision is Decision.PROMOTE:
        if apply:
            console.print(f"{OK} {decision.label} promoted, funded at rung 0 (floor notional)")
        else:
            console.print(
                f"{OK} {decision.label} would be promoted. Re-run with [bold]--apply[/bold] "
                "to record it; that is the moment it becomes eligible for real money."
            )
        return
    err_console.print(
        f"{BAD} {decision.label}: {decision.decision.value} "
        f"({decision.n_failed} blocking failure(s))"
    )
    raise typer.Exit(1)


def _vintage_check(
    ledger: Ledger,
    pinned: PinnedLimits,
    bars: Path | None,
    vintage_id: str | None,
) -> tuple[bool, str]:
    """Whether the cited vintage is sealed and non-empty.

    A missing vintage id is not an error here — it is a refusal the gate makes,
    with its reason on the gate's own row. Answering it in the CLI would move a
    safety decision out of the gate.
    """
    if vintage_id is None:
        return False, "no vintage was cited"
    from tb.data.barstore import BarStore

    store = BarStore(
        ledger,
        root=bars or (Path(ledger.path).parent / "bars"),
        scale=pinned.limits.data.price_scale,
    )
    return SnapshotStore(ledger, store).is_admissible(vintage_id)


def periods_per_year_for(resolution: str) -> int:
    """The annualisation factor for the resolution an evaluation ran at.

    Taken from the *evaluation*, not from `allowed_live_resolutions`. The first
    draft read the config and picked the fastest permitted resolution, which is
    correct only while that list holds one entry: the moment minute were
    allowed, a daily strategy's Sharpe would be converted to a per-period one by
    a factor about eight times too large, and its deflated probability would
    collapse toward 0.5. Conservative, and wrong — and wrong in a way that would
    appear on a config edit rather than on a code change.

    An unrecognised name falls back to daily rather than raising. A stored row
    from a future build naming a resolution this one does not know is a
    reporting problem, not a reason to refuse to evaluate; daily is the slowest
    factor and therefore the one that overstates nothing.
    """
    from tb.backtest.metrics import (
        PERIODS_PER_YEAR_DAILY,
        PERIODS_PER_YEAR_HOURLY,
        PERIODS_PER_YEAR_MINUTE,
    )

    return {
        "minute": PERIODS_PER_YEAR_MINUTE,
        "hourly": PERIODS_PER_YEAR_HOURLY,
        "daily": PERIODS_PER_YEAR_DAILY,
    }.get(resolution.lower(), PERIODS_PER_YEAR_DAILY)


def _jurisdiction(name: str) -> Jurisdiction:
    try:
        return Jurisdiction(name.lower())
    except ValueError as exc:
        err_console.print(
            f"{BAD} unknown jurisdiction {name!r}: "
            f"{', '.join(sorted(j.value for j in Jurisdiction))}"
        )
        raise typer.Exit(2) from exc


# --------------------------------------------------------------------------
# tb promote history
# --------------------------------------------------------------------------


@promote_app.command("history")
def history(
    strategy_id: Annotated[str, typer.Argument(help="The strategy to show.")],
    limits: LimitsOpt = None,
    db: DbOpt = None,
    version: Annotated[int, typer.Option("--version", help="Spec version.")] = 1,
) -> None:
    """Every promotion decision ever taken about one strategy."""
    pinned = _load(limits)
    with _ledger(db, pinned) as ledger:
        decisions = PromotionGate(ledger, limits=pinned.limits).history(strategy_id, version)
    if not decisions:
        console.print(f"{WARN} no promotion decisions recorded for {strategy_id}@v{version}")
        return
    for decision in decisions:
        console.print(f"[bold]{decision.decided_at.isoformat()[:19]}[/bold]")
        console.print(decision.report())
        console.print("")


# --------------------------------------------------------------------------
# tb allocator explain
# --------------------------------------------------------------------------


@allocator_app.command("explain")
def explain(
    limits: LimitsOpt = None,
    db: DbOpt = None,
    strategy_id: Annotated[
        str | None,
        typer.Option("--strategy", help="Show one strategy's history instead.", show_default=False),
    ] = None,
) -> None:
    """The prior, the realised edge, and the shrinkage weight between them.

    The shrinkage is the point of the command. At ten trades a strategy's
    realised edge carries a quarter of the weight, so a number that looked like
    a measurement is mostly the backtest's opinion — and that is the thing an
    operator needs to see before believing an allocation.
    """
    pinned = _load(limits)
    with _ledger(db, pinned) as ledger:
        allocator = Allocator(ledger, limits=pinned.limits)
        rows = allocator.history_for(strategy_id) if strategy_id is not None else allocator.latest()
    if not rows:
        console.print(
            f"{WARN} no allocations recorded yet. `tb run` allocates at the first cycle "
            "of each trading session, once a strategy is promoted."
        )
        return

    table = Table(show_header=True)
    table.add_column("strategy")
    table.add_column("prior bps", justify="right")
    table.add_column("realised bps", justify="right")
    table.add_column("trades", justify="right")
    table.add_column("shrinkage", justify="right")
    table.add_column("blended bps", justify="right")
    table.add_column("weight", justify="right")
    table.add_column("rung", justify="right")
    table.add_column("notional", justify="right")
    table.add_column("capped")
    for row in rows:
        table.add_row(
            row.label,
            str(row.prior_edge_bps),
            "—" if row.realised_edge_bps is None else str(row.realised_edge_bps),
            str(row.n_realised_trades),
            f"{row.shrinkage:.3f}",
            f"{row.blended_edge_bps:.1f}",
            f"{row.weight:.4f}",
            str(row.rung),
            str(row.notional_ccy),
            "[yellow]family[/yellow]" if row.correlation_capped else "—",
        )
    console.print(table)

    mostly_prior = [row for row in rows if row.evidence_is_mostly_prior]
    if mostly_prior:
        console.print(
            f"{WARN} {len(mostly_prior)} of {len(rows)} allocation(s) rest mainly on the "
            "backtest prior rather than on realised trading. At floor size with "
            "multi-day holds that is expected for the first month or two."
        )


# --------------------------------------------------------------------------
# tb allocator ladder
# --------------------------------------------------------------------------


@allocator_app.command("ladder")
def ladder(
    strategy_id: Annotated[str, typer.Argument(help="The strategy to show.")],
    limits: LimitsOpt = None,
    db: DbOpt = None,
    version: Annotated[int, typer.Option("--version", help="Spec version.")] = 1,
) -> None:
    """Every rung change, up and down, with what justified it."""
    pinned = _load(limits)
    with _ledger(db, pinned) as ledger:
        size = SizeLadder(ledger, limits=pinned.limits)
        try:
            current = size.rung_of(strategy_id, version)
        except TbError as exc:
            err_console.print(f"{BAD} {escape(str(exc))}", soft_wrap=True)
            raise typer.Exit(2) from exc
        moves = size.history(strategy_id, version)

    console.print(
        f"{strategy_id}@v{version} is at rung {current} "
        f"({notional_for(current, limits=pinned.limits)} at the rung's own size)"
    )
    if not moves:
        console.print(f"{WARN} no rung changes recorded: still at the floor it was promoted to")
        return
    table = Table(show_header=True)
    table.add_column("when")
    table.add_column("move")
    table.add_column("trades", justify="right")
    table.add_column("days", justify="right")
    table.add_column("reason")
    for move in moves:
        arrow = "[green]↑[/green]" if move.direction.value == "up" else "[red]↓[/red]"
        table.add_row(
            move.moved_at.isoformat()[:19],
            f"{arrow} {move.from_rung}→{move.to_rung}",
            "—" if move.n_trades_at_move is None else str(move.n_trades_at_move),
            "—" if move.days_at_rung is None else str(move.days_at_rung),
            move.reason,
        )
    console.print(table)
