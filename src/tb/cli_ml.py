"""`tb ml`: train a model, list and verify the store, and calibrate the null.

The command-line face of M7. `train` runs `tb.research.ml.ModelTrainer`, which
records the model and every threshold it tried as trials; `models`, `show` and
`verify` read the store back through the same hash checks a trading process
makes; `calibrate` is the release gate — shuffled labels must show no skill,
and a planted pattern must.
"""

from __future__ import annotations

import os
import sys
from datetime import timedelta
from decimal import Decimal
from pathlib import Path
from typing import Annotated

import typer
from rich.console import Console
from rich.markup import escape
from rich.table import Table

from tb.backtest.costs import CostModel, Jurisdiction
from tb.config.loader import PinnedLimits, load_hard_limits
from tb.core.errors import TbError
from tb.data.barstore import BarStore
from tb.data.snapshot import SnapshotStore
from tb.ledger.store import Ledger, default_ledger_path
from tb.registry.model_store import ModelRecord, ModelStore, default_model_root
from tb.research.holdout import DEFAULT_HOLDOUT_FRACTION
from tb.research.validate import VALIDATION_NOTIONAL_CCY

ml_app = typer.Typer(
    help="Train models, read and verify the model store, and calibrate the null.",
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
ModelsOpt = Annotated[
    Path | None,
    typer.Option(
        "--models",
        help="Directory holding model artifacts. Default: beside the ledger.",
        show_default=False,
    ),
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


def _store(ledger: Ledger, models: Path | None) -> ModelStore:
    return ModelStore(ledger, models or default_model_root(ledger.path))


def _features(values: list[str]) -> tuple[tuple[str, int], ...]:
    out: list[tuple[str, int]] = []
    for value in values:
        kind, _, lookback = value.partition(":")
        if not kind or not lookback.isdigit():
            err_console.print(
                f"{BAD} feature {value!r} is not KIND:LOOKBACK, e.g. return_pct:5", soft_wrap=True
            )
            raise typer.Exit(2)
        out.append((kind, int(lookback)))
    return tuple(out)


# --------------------------------------------------------------------------
# tb ml train
# --------------------------------------------------------------------------


@ml_app.command("train")
def train(
    vintage_id: Annotated[str, typer.Argument(help="The sealed vintage to train on.")],
    feature: Annotated[
        list[str],
        typer.Option("--feature", help="KIND:LOOKBACK, repeatable: the model's inputs, in order."),
    ],
    horizon: Annotated[int, typer.Option("--horizon", help="Label holding period, in bars.")] = 5,
    cost_bps: Annotated[
        float | None,
        typer.Option(
            "--cost-bps",
            help="Round trip netted from each label. Default: the cost model's US round trip.",
            show_default=False,
        ),
    ] = None,
    folds: Annotated[int, typer.Option("--folds", help="Walk-forward test folds.")] = 4,
    min_train: Annotated[
        int, typer.Option("--min-train", help="Training samples a fold needs to be scored.")
    ] = 100,
    embargo_days: Annotated[
        float, typer.Option("--embargo-days", help="Gap before each fold, in calendar days.")
    ] = 7.0,
    trees: Annotated[int, typer.Option("--trees")] = 60,
    learning_rate: Annotated[float, typer.Option("--learning-rate")] = 0.05,
    leaves: Annotated[int, typer.Option("--leaves")] = 7,
    depth: Annotated[int, typer.Option("--depth")] = 3,
    min_leaf: Annotated[int, typer.Option("--min-leaf")] = 20,
    seed: Annotated[int, typer.Option("--seed")] = 7,
    entry_quantile: Annotated[
        list[float] | None,
        typer.Option(
            "--entry-quantile",
            help="Repeatable. Each is a trial; default 0.6, 0.75 and 0.9.",
            show_default=False,
        ),
    ] = None,
    exit_quantile: Annotated[float, typer.Option("--exit-quantile")] = 0.5,
    fraction: Annotated[
        float, typer.Option("--fraction", help="Share of the vintage held back.")
    ] = DEFAULT_HOLDOUT_FRACTION,
    apply: Annotated[
        bool,
        typer.Option("--apply/--dry-run", help="Register the surviving specs as candidates."),
    ] = False,
    limits: LimitsOpt = None,
    db: DbOpt = None,
    bars: RootOpt = None,
    models: ModelsOpt = None,
) -> None:
    """Train a model on a sealed vintage and search the specs that read it.

    Walks the training window forward through purged, embargoed folds, fits
    the same folds on shuffled labels beside it, records the final model by
    hash, and backtests one threshold spec per entry quantile with the fold
    models — every one a trial, in one lineage per feature set and label.

    `--dry-run` by default; a dry run still records the model and every
    trial, because the search happened. Exits 1 when shuffled labels show
    skill (nothing is recorded then), 2 when training cannot run.
    """
    from tb.research.ml import ModelTrainer, NullShowsSkill, TrainingConfig
    from tb.strategy.ml.dataset import LabelDefinition
    from tb.strategy.ml.model import ModelParams

    pinned = _load(limits)
    if cost_bps is None:
        _, trip = CostModel(pinned.limits).gate_trade(
            notional_ccy=VALIDATION_NOTIONAL_CCY,
            instrument_currency="USD",
            jurisdiction=Jurisdiction.US,
            expected_edge_bps=Decimal(100),
        )
        label_cost = trip.total_bps.quantize(Decimal("0.01"))
    else:
        label_cost = Decimal(str(cost_bps))

    try:
        config = TrainingConfig(
            features=_features(feature),
            label=LabelDefinition(horizon=horizon, cost_bps=label_cost),
            params=ModelParams(
                n_trees=trees,
                learning_rate=learning_rate,
                num_leaves=leaves,
                max_depth=depth,
                min_data_in_leaf=min_leaf,
                seed=seed,
            ),
            n_folds=folds,
            embargo=timedelta(days=embargo_days),
            min_train=min_train,
            entry_quantiles=tuple(entry_quantile or (0.6, 0.75, 0.9)),
            exit_quantile=exit_quantile,
        )
    except TbError as exc:
        err_console.print(f"{BAD} {escape(str(exc))}", soft_wrap=True)
        raise typer.Exit(2) from exc

    with _ledger(db, pinned) as ledger:
        snapshots = SnapshotStore(
            ledger,
            BarStore(
                ledger,
                root=bars or (Path(ledger.path).parent / "bars"),
                scale=pinned.limits.data.price_scale,
            ),
        )
        console.print(
            f"  training on {vintage_id}: {', '.join(f'{k}:{n}' for k, n in config.features)}, "
            f"{horizon}-bar labels net of {label_cost}bps…"
        )
        try:
            report = ModelTrainer(
                ledger,
                limits=pinned.limits,
                snapshots=snapshots,
                models=_store(ledger, models),
            ).run(vintage_id=vintage_id, config=config, fraction=fraction, register=apply)
        except NullShowsSkill as exc:
            err_console.print(f"{BAD} {escape(str(exc))}", soft_wrap=True)
            raise typer.Exit(1) from exc
        except TbError as exc:
            err_console.print(f"{BAD} {escape(str(exc))}", soft_wrap=True)
            raise typer.Exit(2) from exc

    for line in report.explain().splitlines():
        console.print(f"  {escape(line)}", soft_wrap=True)
    console.print(f"{OK} recorded {report.record.model_id} ({report.record.artifact_sha256})")


# --------------------------------------------------------------------------
# Reading the store
# --------------------------------------------------------------------------


def _auc(record: ModelRecord, prefix: str) -> str:
    auc = record.metrics.get(f"{prefix}auc")
    se = record.metrics.get(f"{prefix}auc_null_se")
    if not isinstance(auc, int | float):
        return "n/a"
    return f"{auc:.3f}" + (f" ±{se:.3f}" if isinstance(se, int | float) else "")


@ml_app.command("models")
def list_models(
    limits: LimitsOpt = None,
    db: DbOpt = None,
    models: ModelsOpt = None,
) -> None:
    """Every recorded model: what it learned from, and how it scored out of sample."""
    pinned = _load(limits)
    with _ledger(db, pinned) as ledger:
        records = _store(ledger, models).records()
    if not records:
        console.print("  no model is recorded. `tb ml train <vintage> --feature …` records one.")
        return
    table = Table(show_header=True)
    # Identifiers never wrap: an id broken across two lines cannot be copied
    # into `tb ml show`, and it is the one thing in this table that gets copied.
    table.add_column("model", no_wrap=True)
    table.add_column("vintage", no_wrap=True)
    for column in ("samples", "through", "oos AUC", "null AUC"):
        table.add_column(column, no_wrap=True)
    table.add_column("features")
    for record in records:
        table.add_row(
            record.model_id,
            record.vintage_id,
            str(record.n_samples),
            record.trained_through.isoformat()[:10],
            _auc(record, "oos_"),
            _auc(record, "null_"),
            ", ".join(f"{kind}:{lookback}" for kind, lookback in record.features),
        )
    console.print(table)


@ml_app.command("show")
def show(
    model_id: Annotated[str, typer.Argument(help="The model to show, e.g. mdl_0123abcd…")],
    limits: LimitsOpt = None,
    db: DbOpt = None,
    models: ModelsOpt = None,
) -> None:
    """One model's record, and whether its artifact still verifies.

    Exits 1 when the artifact is missing or no longer hashes to its record, 2
    when there is no such model.
    """
    pinned = _load(limits)
    with _ledger(db, pinned) as ledger:
        store = _store(ledger, models)
        record = store.get(model_id)
        if record is None:
            err_console.print(f"{BAD} {model_id} is not in the model store")
            raise typer.Exit(2)
        for label, value in (
            ("artifact", record.artifact_sha256),
            ("kind", record.kind),
            ("vintage", record.vintage_id),
            ("sealed from", "none" if record.sealed_from is None else record.sealed_from),
            ("window", f"{record.window_start.isoformat()} to {record.window_end.isoformat()}"),
            ("trained through", record.trained_through.isoformat()),
            ("samples", record.n_samples),
            ("features", ", ".join(f"{k}:{n}" for k, n in record.features)),
            ("label", ", ".join(f"{k}={v}" for k, v in sorted(record.label.items()))),
            ("params", ", ".join(f"{k}={v}" for k, v in sorted(record.params.items()))),
            ("search", record.search_id or "none"),
            ("recorded", f"{record.recorded_at.isoformat()} (event {record.recording_event_seq})"),
        ):
            console.print(f"  {label:>15}: {escape(str(value))}", soft_wrap=True)
        for name, value in sorted(record.metrics.items()):
            console.print(f"  {name:>15}: {value}")
        try:
            store.load(model_id)
        except TbError as exc:
            err_console.print(f"{BAD} {escape(str(exc))}", soft_wrap=True)
            raise typer.Exit(1) from exc
    console.print(f"{OK} the artifact hashes to its record and parses")


@ml_app.command("verify")
def verify(
    limits: LimitsOpt = None,
    db: DbOpt = None,
    models: ModelsOpt = None,
) -> None:
    """Load every recorded model through the hash checks. Exits 1 on any refusal."""
    pinned = _load(limits)
    failures = 0
    with _ledger(db, pinned) as ledger:
        store = _store(ledger, models)
        records = store.records()
        for record in records:
            try:
                store.load(record.model_id)
            except TbError as exc:
                failures += 1
                err_console.print(f"{BAD} {escape(str(exc))}", soft_wrap=True)
            else:
                console.print(f"{OK} {record.model_id}")
    if failures:
        err_console.print(f"{BAD} {failures} of {len(records)} model(s) refused")
        raise typer.Exit(1)
    console.print(f"{OK} {len(records)} model(s) verified against the ledger")


# --------------------------------------------------------------------------
# tb ml calibrate
# --------------------------------------------------------------------------


@ml_app.command("calibrate")
def calibrate_command(
    nulls: Annotated[int, typer.Option("--nulls", help="Shuffles to run.")] = 8,
    days: Annotated[int, typer.Option("--days", help="Synthetic sessions.")] = 600,
) -> None:
    """The shuffled-label null through the real data path. A release gate.

    Synthetic bars with a learnable rhythm go through the reader, the one
    pipeline, the labels and the purged folds. The same folds on shuffled
    labels must show no skill — skill there would be the evaluation's own —
    and the real labels must show it, or the null passed only by being unable
    to find anything. Exits 1 on either failure.
    """
    from tb.research.ml import calibrate

    result = calibrate(n_nulls=nulls, days=days)
    table = Table(show_header=True, title=f"{result.n_samples} samples, walk-forward")
    for column in ("labels", "AUC", "noise ±", "skill", "verdict"):
        table.add_column(column)

    def row(label: str, oos: object, *, wants_skill: bool) -> None:
        from tb.strategy.ml.trainer import OutOfSample

        assert isinstance(oos, OutOfSample)
        found = oos.shows_skill()
        good = found if wants_skill else not found
        table.add_row(
            label,
            "n/a" if oos.auc is None else f"{oos.auc:.3f}",
            "n/a" if oos.auc_null_se is None else f"{oos.auc_null_se:.3f}",
            "n/a" if oos.skill is None else f"{oos.skill:+.3f}",
            ("[green]" if good else "[red]")
            + ("skill" if found else "no skill")
            + ("[/green]" if good else "[/red]"),
        )

    row("real (control)", result.control, wants_skill=True)
    for index, null in enumerate(result.nulls):
        row(f"shuffled #{index}", null, wants_skill=False)
    console.print(table)

    if not result.control_found_it:
        err_console.print(
            f"{BAD} the planted rhythm was not found. A null that cannot fail because the "
            "model cannot find anything is not a null."
        )
        raise typer.Exit(1)
    if not result.nulls_found_nothing:
        err_console.print(
            f"{BAD} shuffled labels scored as skill: the evaluation is supplying it, and every "
            "out-of-sample number this build reports is suspect."
        )
        raise typer.Exit(1)
    console.print(
        f"{OK} shuffled labels show no skill across {len(result.nulls)} shuffle(s); the real "
        "labels do"
    )
