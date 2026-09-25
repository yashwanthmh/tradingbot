"""`tb research ...` — the search, the trial log and the single holdout evaluation.

    tb research cycle      search a sealed vintage; record every trial
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

import json
import os
import sys
from datetime import timedelta
from enum import StrEnum
from pathlib import Path
from typing import TYPE_CHECKING, Annotated

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
    decisions_between,
    evaluation_reader,
    holdout_boundary,
)
from tb.research.llm.adapter import DEFAULT_MODEL, LLMError
from tb.research.null_gate import measure_false_promotion_rate
from tb.research.trials import TrialLog
from tb.strategy.dsl.ops import DslStrategy, pipeline_from_spec

if TYPE_CHECKING:
    from tb.research.loop import CycleReport, ProposalContext, ProposerFactory
    from tb.research.mutate import SpecProposer

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
# tb research cycle
# --------------------------------------------------------------------------


class ProposerChoice(StrEnum):
    """Where a search's first generation comes from."""

    RANDOM = "random"
    LLM = "llm"


@research_app.command("cycle")
def cycle(
    vintage_id: Annotated[str, typer.Argument(help="The sealed vintage to search over.")],
    limits: LimitsOpt = None,
    db: DbOpt = None,
    bars: RootOpt = None,
    trials: Annotated[
        int,
        typer.Option(
            "--trials",
            help="The whole search's budget. The haircut is computed from this total.",
        ),
    ] = 50,
    per_generation: Annotated[
        int, typer.Option("--per-generation", help="Proposals per generation.")
    ] = 25,
    survivors: Annotated[
        int, typer.Option("--survivors", help="Parents carried into each next generation.")
    ] = 4,
    seed: Annotated[int, typer.Option("--seed", help="Seed, so the search reproduces.")] = 0,
    fraction: Annotated[
        float, typer.Option("--fraction", help="Share of the window held back.")
    ] = DEFAULT_HOLDOUT_FRACTION,
    parents: Annotated[
        list[str] | None,
        typer.Option(
            "--from",
            help="A registered strategy to refine instead of drawing at random. Repeatable.",
            show_default=False,
        ),
    ] = None,
    apply: Annotated[
        bool,
        typer.Option(
            "--apply/--dry-run",
            help="Register the survivors. Every trial is recorded either way.",
        ),
    ] = False,
    proposer: Annotated[
        ProposerChoice,
        typer.Option(
            "--proposer",
            help=(
                "Where the first generation comes from: `random` needs nothing; `llm` asks "
                "a Claude model (the llm extra and ANTHROPIC_API_KEY)."
            ),
        ),
    ] = ProposerChoice.RANDOM,
    model: Annotated[
        str, typer.Option("--model", help="The Claude model `--proposer llm` asks.")
    ] = DEFAULT_MODEL,
    fallback: Annotated[
        bool,
        typer.Option(
            "--fallback/--no-fallback",
            help="Let the API re-run a request its model declines on a recommended fallback.",
        ),
    ] = True,
    out: Annotated[
        Path | None,
        typer.Option(
            "--out",
            help="Write every candidate spec, with its outcome, as JSON lines.",
            show_default=False,
        ),
    ] = None,
) -> None:
    """Search a sealed vintage for strategies, recording every trial.

    Proposes, validates and backtests on the training window only — the reader
    it builds cannot reach past the holdout boundary — then registers the
    survivors as candidates. It promotes nothing and evaluates no holdout; those
    are `tb research holdout` and `tb promote evaluate`.

    `--dry-run` by default, and a dry run still **records every trial**. The
    search happened, and a search whose size went unrecorded would let its best
    result be registered by hand with no multiplicity haircut at all.

    The budget is the whole search. The output reports the out-of-sample Sharpe a
    search of that size must show to clear the deflated-Sharpe gate — about 2.1
    at ten trials and 3.8 at a thousand — which is the reason to keep it small.

    The report always says what was refused before a backtest and why, "none"
    included. `--out FILE` writes every candidate's full spec with its outcome
    as JSON lines — the trial log keeps only hashes, so this is how a dry run's
    specs can be read.

    `--proposer llm` changes where the first generation comes from and nothing
    else: the model is shown the feature dictionary, the constraints and whether
    the index is above or below its long average — never a date, a price or an
    instrument — and what it returns meets the same validator, backtest and
    selection as a random draw. Every exchange is recorded in the ledger, since
    unlike a seeded draw it cannot be replayed.
    """
    from tb.research.loop import CycleError, ResearchCycle
    from tb.research.searcher import SearchBudget, SearchError

    pinned = _load(limits)
    try:
        budget = SearchBudget(
            n_trials=trials,
            n_per_generation=per_generation,
            n_survivors=survivors,
            seed=seed,
        )
    except SearchError as exc:
        err_console.print(f"{BAD} {escape(str(exc))}", soft_wrap=True)
        raise typer.Exit(2) from exc

    factory: ProposerFactory | None = None
    if proposer is ProposerChoice.LLM:
        if parents:
            err_console.print(
                f"{BAD} `--from` refines registered strategies by mutation, so a search "
                "seeded with it never asks the model. Run one or the other.",
                soft_wrap=True,
            )
            raise typer.Exit(2)
        factory = _llm_factory(model=model, fallback=fallback)

    with _ledger(db, pinned) as ledger:
        store = BarStore(
            ledger,
            root=bars or (Path(ledger.path).parent / "bars"),
            scale=pinned.limits.data.price_scale,
        )
        console.print(f"  searching {vintage_id} with a budget of {trials} trial(s)…")
        try:
            report = ResearchCycle(
                ledger, limits=pinned.limits, snapshots=SnapshotStore(ledger, store)
            ).run(
                vintage_id=vintage_id,
                budget=budget,
                fraction=fraction,
                register=apply,
                seed_strategy_ids=tuple(parents or ()),
                proposer=factory,
            )
        except LLMError as exc:
            # The model is asked for the first generation only, so a failed call
            # always precedes the first evaluation — worth saying, because the
            # operator's next question is whether the failure cost trials.
            err_console.print(
                f"{BAD} {escape(str(exc))}\n  No trial was recorded: the model is asked "
                "before anything is evaluated, so this search did not happen.",
                soft_wrap=True,
            )
            raise typer.Exit(2) from exc
        except (CycleError, TbError) as exc:
            err_console.print(f"{BAD} {escape(str(exc))}", soft_wrap=True)
            raise typer.Exit(2) from exc

    outcome = report.outcome
    if outcome.rejections_by_code:
        table = Table(show_header=True, title="refused before a backtest")
        table.add_column("reason")
        table.add_column("count", justify="right")
        for code, count in sorted(outcome.rejections_by_code.items(), key=lambda item: -item[1]):
            table.add_row(code, str(count))
        console.print(table)
    else:
        # Said rather than left out. An absent table reads the same as a report
        # that was never produced, and "nothing was refused" is a finding: it
        # means every proposal cost a backtest.
        console.print("  refused before a backtest: none")
    if out is not None:
        _write_candidates(out, report)
        console.print(
            f"  wrote {outcome.n_proposed} candidate spec(s), with what became of each, "
            f"to {escape(str(out))}",
            soft_wrap=True,
        )

    ranked = sorted(
        (c for c in outcome.candidates if c.fitness is not None),
        key=lambda c: (-(c.fitness or 0.0), c.spec_hash),
    )[:10]
    if ranked:
        table = Table(show_header=True, title="best by training Sharpe")
        table.add_column("spec")
        table.add_column("gen", justify="right")
        table.add_column("how")
        table.add_column("train Sharpe", justify="right")
        table.add_column("trades", justify="right")
        table.add_column("lineage")
        for candidate in ranked:
            table.add_row(
                candidate.spec_hash[:12],
                str(candidate.generation),
                candidate.proposal.operator,
                f"{candidate.fitness:.2f}",
                str(candidate.n_trades),
                report.lineage_of.get(candidate.spec_hash, "—"),
            )
        console.print(table)

    for line in report.explain().splitlines():
        console.print(f"  {escape(line)}", soft_wrap=True)
    if outcome.errored_examples():
        for spec_hash, error in outcome.errored_examples():
            console.print(f"  {WARN} {spec_hash[:12]} errored: {escape(error)}", soft_wrap=True)
    console.print(
        f"{OK} {report.search_id}: {outcome.n_proposed} trial(s) in "
        f"{report.duration_seconds:.1f}s. `tb research trials --search {report.search_id}` "
        "shows every one."
    )


def _write_candidates(path: Path, report: CycleReport) -> None:
    """Every candidate of a search as a JSON line: the spec, and what became of it.

    What a dry run produces. The trial log keeps each candidate's hash and
    numbers but not its tree, so without this a dry run's specs exist only as
    hashes; with it, each one can be read, compared or registered by hand —
    and a hand registration still carries the search's trial count, because the
    trials were recorded when the search ran.
    """
    with path.open("w", encoding="utf-8") as handle:
        for candidate in report.outcome.candidates:
            rejection = candidate.rejection
            line = {
                "spec_hash": candidate.spec_hash,
                "generation": candidate.generation,
                "operator": candidate.proposal.operator,
                "author_kind": candidate.proposal.author_kind.value,
                "lineage_id": report.lineage_of.get(candidate.spec_hash),
                "outcome": candidate.outcome.value,
                "rejection": None
                if rejection is None
                else {
                    "code": rejection.code,
                    "reason": rejection.reason,
                    "observed": rejection.observed,
                    "threshold": rejection.threshold,
                },
                "error": candidate.error or None,
                "net_sharpe": candidate.net_sharpe,
                "n_trades": candidate.n_trades,
                "spec": candidate.spec.model_dump(mode="json"),
            }
            handle.write(json.dumps(line, sort_keys=True) + "\n")


def _llm_factory(*, model: str, fallback: bool) -> ProposerFactory:
    """The model-backed proposer, with its client built before any data loads.

    Built here rather than inside the cycle so a missing SDK, a missing key or a
    live broker key in the environment fails in the first second, not after the
    vintage has been read and the training window laid out.
    """
    from tb.research.llm.adapter import AnthropicClient, LLMProposer, LLMUnavailable

    try:
        client = AnthropicClient(model=model, fallbacks=fallback)
    except LLMUnavailable as exc:
        err_console.print(f"{BAD} {escape(str(exc))}", soft_wrap=True)
        raise typer.Exit(2) from exc
    console.print(
        f"  first generation from {escape(model)}, refusal fallbacks "
        + ("on" if fallback else "off")
    )

    def build(context: ProposalContext) -> SpecProposer:
        return LLMProposer(
            client=client,
            bounds=context.bounds,
            regime=context.regime,
            required_sharpe=context.required_sharpe,
            n_trials=context.n_trials,
        )

    return build


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
        # The same schedule builder the search cycle uses for training, so the
        # statistics a candidate was selected on and the ones it is judged on
        # come from schedules built identically — deduplicated across
        # instruments, and split on the decision time itself.
        schedule = decisions_between(snapshots.bars_of(vintage_id), start=window.sealed_from)
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
                resolution=Resolution.DAILY.value,
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

        if not passed:
            # Retired, not left as a candidate. Failing is terminal — the
            # evaluation cannot be re-run and the gate requires a passing one,
            # so this version can never be promoted. Leaving it a candidate
            # would say the opposite in the one table an operator reads, and
            # would invite a searcher to keep re-evaluating something that is
            # already finished. It remains available as a parent: a mutation is
            # a new strategy in the same lineage, which is what carries this
            # one's trial count into the next haircut.
            registry.retire(
                strategy_id,
                version=version,
                reason="failed the sealed holdout, which is terminal for a version",
            )

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
