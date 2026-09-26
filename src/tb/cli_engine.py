"""`tb run` and `tb watchdog` — the M4 surface.

Two commands, and they are meant to be run as two processes. That is not a
deployment preference: the watchdog exists to catch a *wedged* trader, and a
wedged process cannot supervise itself. Running them in one process would give
the appearance of supervision with none of it.

    tb watchdog          supervise a trader; engage the kill switch on a stall
    tb run --mode paper  the trading loop, against the simulated broker
    tb run --mode demo   the trading loop, against the Trading 212 demo account

There is deliberately no `--mode live`. Reaching a real-money account needs
`live_writes_armed` on the client *and* the live key present, and neither is
settable from this CLI — arming live trading should be a reviewed change, not
a flag someone can pass at 2am.
"""

from __future__ import annotations

import os
import sys
from collections.abc import Callable
from contextlib import suppress
from datetime import datetime
from decimal import Decimal
from pathlib import Path
from typing import Annotated

import typer
from rich.console import Console
from rich.markup import escape
from rich.table import Table

from tb.broker.port import Broker
from tb.config.loader import PinnedLimits, load_hard_limits
from tb.core.errors import TbError
from tb.core.ids import new_run_id
from tb.data.barstore import BarStore
from tb.data.calendar import TradingCalendar
from tb.engine.funding import (
    Book,
    explicit_book,
    funded_book,
    record_book,
    unfunded_notional,
)
from tb.engine.intents import IntentLog
from tb.engine.loop import LoopHalted, TradingLoop, build_instrument_map
from tb.engine.orders import OrderSubmitter
from tb.features.pipeline import FeaturePipeline
from tb.ledger.store import Ledger, default_ledger_path
from tb.ops.state import StateMachine
from tb.ops.watchdog import (
    WATCHDOG_STALE_SECONDS,
    InstanceLock,
    InstanceLockRefused,
    SelfCheck,
    Watchdog,
)
from tb.portfolio.pnl import EquityCurve
from tb.registry.model_store import ModelStore, default_model_root
from tb.strategy.dsl.ops import ModelSource
from tb.strategy.trivial import MovingAverageCross, specs

engine_app = typer.Typer(help="Run the trading loop and its supervisor.", no_args_is_help=True)

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


@engine_app.command("run")
def run(
    limits: LimitsOpt = None,
    db: DbOpt = None,
    bars: RootOpt = None,
    models: ModelsOpt = None,
    mode: Annotated[
        str, typer.Option("--mode", help="paper (simulated broker) or demo (Trading 212).")
    ] = "paper",
    cycles: Annotated[
        int, typer.Option("--cycles", help="Stop after this many cycles. 0 means forever.")
    ] = 1,
    interval: Annotated[float, typer.Option("--interval", help="Seconds between cycles.")] = 60.0,
    tickers: Annotated[
        str | None,
        typer.Option("--tickers", help="Comma-separated T212 tickers. Default: every verified."),
    ] = None,
    equity: Annotated[
        float,
        typer.Option("--equity", help="Starting equity for the paper broker."),
    ] = 10_000.0,
    no_watchdog: Annotated[
        bool,
        typer.Option(
            "--no-watchdog",
            help="Run without a supervisor. For a drill only — it removes a control.",
        ),
    ] = False,
    strategy: Annotated[
        str | None,
        typer.Option(
            "--strategy",
            help=(
                "Trade one hand-written strategy instead of the promoted book. "
                "Only 'trivial' is accepted. For drilling the loop, not the funding path."
            ),
            show_default=False,
        ),
    ] = None,
) -> None:
    """The trading loop: bars in, orders out, every step in the ledger.

    Refuses to start rather than starting degraded. A missing ledger, an
    unverified universe, a held lease or an unresolved unknown intent each
    stop it here with the remedy named — because a loop that started anyway
    and halted on its first cycle would have already written a run record and
    a heartbeat, which makes the failure harder to read rather than easier.

    `--cycles 1` by default, so an accidental invocation does one pass and
    stops. Pass `--cycles 0` to run until halted.

    What it trades is the *promoted book* — every strategy the registry says may
    trade, sized by its rung and its allocation. Nothing promoted is a refusal
    to start rather than an idle loop: a run that decided nothing all day and a
    run with nothing to decide with are indistinguishable from the outside, and
    only one of them means `tb promote evaluate` has never been run.
    """
    pinned = _load(limits)
    if strategy is not None and strategy != "trivial":
        err_console.print(
            f"{BAD} unknown --strategy {strategy!r}. The only hand-written strategy here is "
            "'trivial'; every other strategy reaches the loop by being promoted, which is "
            "what `tb promote evaluate` records.",
            soft_wrap=True,
        )
        raise typer.Exit(2)
    if mode not in ("paper", "demo"):
        err_console.print(
            f"{BAD} unknown mode {mode!r}. Use paper (simulated broker) or demo "
            "(Trading 212 demo account). There is no live mode here: arming a real-money "
            "account is a reviewed change, not a CLI flag.",
            soft_wrap=True,
        )
        raise typer.Exit(2)

    run_id = new_run_id()
    with _ledger(db, pinned) as ledger:
        # The broker first, and specifically before the universe check. That
        # ordering is deliberate: `_broker` is where a live key is refused,
        # and a live key present is a more serious condition than an empty
        # universe. With the checks the other way round an operator who had
        # `T212_LIVE_API_KEY` set was told "nothing is verified to trade",
        # which is true and beside the point — and a CI step asserting the
        # credential refusal passed without ever reaching it.
        broker = _broker(mode, pinned, equity=Decimal(str(equity)))

        universe = build_instrument_map(
            ledger, tickers=[t.strip() for t in tickers.split(",")] if tickers else None
        )
        if not universe:
            err_console.print(
                f"{BAD} no instrument is both mapped and tradable, so there is nothing to "
                "trade. `tb symbols verify` is what makes a first entry possible; "
                "`tb symbols audit` shows where each mapping stands.",
                soft_wrap=True,
            )
            raise typer.Exit(2)

        # The book, after the broker and the universe. Both of those refuse for
        # reasons an operator must fix first — a live key present, nothing
        # verified to trade — and reporting "nothing is promoted" ahead of
        # either would name the least urgent problem. Read-only here; it is
        # recorded once the lease is held, so an instance that loses the lease
        # leaves no book behind.
        model_store = ModelStore(ledger, models or default_model_root(ledger.path))
        book = _book(ledger, pinned, strategy=strategy, broker=broker, models=model_store)

        store = BarStore(
            ledger,
            root=bars or (Path(ledger.path).parent / "bars"),
            scale=pinned.limits.data.price_scale,
        )
        log = IntentLog(ledger, run_id=run_id)
        submitter = OrderSubmitter(ledger=ledger, broker=broker, log=log, run_id=run_id)

        run_dir = Path(pinned.limits.safety.heartbeat_path).parent
        self_check = SelfCheck(
            ledger=ledger,
            kill_switch_path=Path(pinned.limits.safety.kill_switch_path),
            liveness_path=run_dir / "watchdog",
            run_id=run_id,
            require_watchdog=not no_watchdog,
        )
        lock = InstanceLock(ledger, run_id=run_id)
        try:
            lease = lock.acquire()
        except InstanceLockRefused as exc:
            err_console.print(f"{BAD} {escape(str(exc))}", soft_wrap=True)
            raise typer.Exit(2) from exc

        # Now that this process holds the lease: that it is running, in which
        # mode, from which code and config, on which host. A trading run was
        # the one kind of run that recorded none of this, so "was that session
        # on the demo account or on paper" had no answer in the ledger — and
        # it is the question the live gate's clean-session count asks.
        ledger.record_run_start(run_id=run_id, mode=mode)

        # How the run ended, recorded however it ends, from the moment its start
        # is on record. A run whose end is never written is itself the evidence
        # of a crash — a kill -9, a power cut — so this is written for every
        # exit this process survives to see, a failure setting up included.
        ending: tuple[str, str | None, str | None] = ("error", None, None)
        try:
            # Then what it is trading. Not answerable from the decisions alone,
            # since a funded strategy that signalled nothing leaves no rows for
            # the instruments it declined and an excluded one leaves none at all.
            record_book(ledger, book, run_id=run_id)

            loop = TradingLoop(
                ledger=ledger,
                pinned=pinned,
                broker=broker,
                bars=store,
                book=book,
                submitter=submitter,
                log=log,
                run_id=run_id,
                instruments=universe,
                state=StateMachine(ledger, pinned, run_id=run_id),
                self_check=self_check,
                lock=lock,
                equity=EquityCurve(ledger, run_id=run_id),
                # The promoted book is reviewed, re-rung, re-allocated and
                # rebuilt once a session. The drill book is not: nothing funds it.
                on_new_session=(
                    None
                    if strategy is not None
                    else _session_refresh(
                        ledger,
                        pinned,
                        broker=broker,
                        universe=universe,
                        run_id=run_id,
                        models=model_store,
                    )
                ),
            )
            _price_paper_venue(broker, store=store, universe=universe, resolution=loop.resolution)

            console.print(
                f"run [bold]{run_id}[/bold] in [bold]{mode}[/bold] mode over "
                f"{len(universe)} instrument(s), lease to {lease.expires_at.isoformat()}"
            )
            for line in book.explain().splitlines():
                console.print(f"  {escape(line)}", soft_wrap=True)
            if no_watchdog:
                console.print(
                    f"{WARN} running without a supervisor. A wedged loop is the one state it "
                    "cannot detect about itself, so nothing would notice.",
                    soft_wrap=True,
                )

            results = loop.run_forever(
                interval_seconds=interval, max_cycles=None if cycles == 0 else cycles
            )
            ending = (f"completed {len(results)} cycle(s)", None, None)
        except LoopHalted as exc:
            ending = ("halted", type(exc).__name__, str(exc))
            err_console.print(f"\n{BAD} halted: {escape(str(exc))}", soft_wrap=True)
            raise typer.Exit(1) from exc
        except KeyboardInterrupt:
            # Stopped by the operator: an ending, not a fault.
            ending = ("interrupted", None, None)
            raise
        except Exception as exc:
            ending = ("error", type(exc).__name__, str(exc))
            raise
        finally:
            with suppress(Exception):
                ledger.record_run_end(
                    run_id=run_id,
                    exit_reason=ending[0],
                    error_type=ending[1],
                    error_detail=ending[2],
                )
            # Best effort: the lease expires on its own, so failing to release
            # it costs the next instance a wait rather than the account.
            with suppress(Exception):
                lock.release()

        _report(results)


@engine_app.command("watchdog")
def watchdog(
    limits: LimitsOpt = None,
    db: DbOpt = None,
    cycles: Annotated[
        int, typer.Option("--cycles", help="Stop after this many checks. 0 means forever.")
    ] = 1,
    interval: Annotated[float, typer.Option("--interval", help="Seconds between checks.")] = 15.0,
) -> None:
    """Supervise a trader. Engages the kill switch when its heartbeat stalls.

    Run as a separate process from `tb run`. The whole point is to catch a
    trader that has stopped running its own checks, and a supervisor sharing
    that process would stop with it.

    It deliberately takes the paths from the limits file but does not require
    the trader to be running: a supervisor that refused to start without a
    live trader would be unavailable at exactly the moment one had died.
    """
    import time

    pinned = _load(limits)
    run_dir = Path(pinned.limits.safety.heartbeat_path).parent
    with _ledger(db, pinned) as ledger:
        dog = Watchdog(
            heartbeat_path=Path(pinned.limits.safety.heartbeat_path),
            kill_switch_path=Path(pinned.limits.safety.kill_switch_path),
            liveness_path=run_dir / "watchdog",
            stale_after_seconds=pinned.limits.safety.heartbeat_stale_seconds,
            ledger=ledger,
        )
        console.print(
            f"supervising: heartbeat {dog.heartbeat_path}, stale after "
            f"{dog.stale_after_seconds}s (watchdog's own liveness bound is "
            f"{WATCHDOG_STALE_SECONDS}s)"
        )

        tripped = 0
        checked = 0
        # One stale episode, one trip in the ledger: the switch is re-engaged
        # every pass, but only the first pass of an episode records it.
        in_episode = False
        while cycles == 0 or checked < cycles:
            verdict = dog.check(record=not in_episode)
            in_episode = verdict.tripped
            checked += 1
            if verdict.tripped:
                tripped += 1
                err_console.print(
                    f"{BAD} {escape(verdict.detail)} — {escape(verdict.action_taken)}",
                    soft_wrap=True,
                )
            if cycles != 0 and checked >= cycles:
                break
            time.sleep(interval)

        if tripped:
            console.print(f"\n{BAD} tripped on {tripped} of {checked} check(s)")
            raise typer.Exit(1)
        console.print(f"\n{OK} {checked} check(s), heartbeat healthy throughout")


def _book(
    ledger: Ledger,
    pinned: PinnedLimits,
    *,
    strategy: str | None,
    broker: Broker,
    models: ModelSource,
) -> Book:
    """The funded book, or the one-strategy drill book `--strategy trivial` asks for.

    Equity comes from the broker rather than from `--equity`, because the rung
    and the allocation are both intersected with `per_position_pct` of the real
    account. Sizing the book against a number passed on the command line would
    let a typo fund a strategy above the cap that is actually in force.
    """
    equity = broker.get_cash().equity
    if strategy == "trivial":
        # Not funded by the registry, so the ladder does not apply to it and the
        # M4 caps bind alone — see `unfunded_notional`. Recorded like any other
        # book by the caller, because "this run was trading a hand-written
        # strategy" is exactly the thing a reader of the ledger must not have to
        # infer.
        book = explicit_book(
            MovingAverageCross(),
            FeaturePipeline(specs=specs()),
            notional_ccy=unfunded_notional(pinned.limits, equity_ccy=equity),
            detail=(
                "hand-written, supplied with --strategy trivial: it has passed no gate "
                "and is bounded by the per-position cap rather than by a rung"
            ),
        )
        console.print(
            f"{WARN} trading the hand-written trivial strategy, which no gate has cleared. "
            "The promoted book is what `tb run` trades without --strategy.",
            soft_wrap=True,
        )
        return book

    book = funded_book(ledger, limits=pinned.limits, equity_ccy=equity, models=models)
    if not book.funded:
        for label, reason in book.excluded:
            err_console.print(f"{BAD} {escape(label)}: {escape(reason)}", soft_wrap=True)
        err_console.print(
            f"{BAD} no strategy is promoted, so there is nothing to trade. "
            "`tb promote evaluate <strategy-id>` runs the gate and prints every check; "
            "`tb registry list` shows what is registered. `--strategy trivial` runs the "
            "hand-written strategy instead, which is a drill of the loop rather than of "
            "the funding path.",
            soft_wrap=True,
        )
        raise typer.Exit(2)
    return book


def _broker(mode: str, pinned: PinnedLimits, *, equity: Decimal) -> Broker:
    """The broker for this mode.

    `paper` is the simulated broker rather than the demo account. That is the
    honest naming: a paper run reaches no network at all, so it tests the loop
    and not the adapter. `demo` is a real Trading 212 account with fake money,
    which tests both.
    """
    if mode == "paper":
        from tb.broker.simulated import SimulatedBroker

        # Marked to market, so the paper account moves: fills at the price the
        # loop decided on, equity that the loss breakers can read, and stops
        # that fire. Priced once the bar store is open — see
        # `_price_paper_venue`. Until then it holds only cash, which needs no
        # price, and it refuses any market order rather than invent one.
        return SimulatedBroker(
            environment="paper",
            currency=pinned.limits.currency,
            equity=equity,
            free_cash=equity,
            mark_to_market=True,
        )

    from tb.broker.t212.client import T212Client
    from tb.broker.t212.ratelimit import RateGovernor

    run_dir = Path(pinned.limits.safety.heartbeat_path).parent
    try:
        return T212Client.from_env(
            governor=RateGovernor(state_path=run_dir / "ratelimit.json"),
            # Refuses a live key outright. `tb run` never reaches a real-money
            # account, whatever is in the environment.
            require_demo=True,
        )
    except TbError as exc:
        err_console.print(f"{BAD} {escape(str(exc))}", soft_wrap=True)
        raise typer.Exit(2) from exc


def _session_refresh(
    ledger: Ledger,
    pinned: PinnedLimits,
    *,
    broker: Broker,
    universe: dict[str, str],
    run_id: str,
    models: ModelSource,
) -> Callable[[datetime], Book | None]:
    """The loop's once-a-session hook: the portfolio pass, then a fresh book.

    The pass reviews, re-rungs and re-allocates (`tb.portfolio.session`); the
    book is then rebuilt from the registry, so a loop left running trades the
    rungs and allocations of today — and picks up a promotion, or a retirement
    applied with `tb review --apply`, at the next session rather than at the
    next restart. A session already reviewed keeps the book it has.
    """
    from tb.engine.funding import owner_of
    from tb.portfolio.correlation import Holding
    from tb.portfolio.session import run_session_pass

    def refresh(at: datetime) -> Book | None:
        equity = broker.get_cash().equity
        session = TradingCalendar().day_of(at).day
        holdings: list[Holding] = []
        for position in broker.get_positions():
            owner = owner_of(ledger, t212_ticker=position.ticker)
            if position.quantity > 0 and position.ticker in universe and owner is not None:
                holdings.append(
                    Holding(
                        strategy_id=owner.strategy_id,
                        instrument_uid=universe[position.ticker],
                        session_date=session,
                    )
                )
        passed = run_session_pass(
            ledger,
            limits=pinned.limits,
            equity_ccy=equity,
            at=at,
            holdings=holdings,
            run_id=run_id,
        )
        if passed.skipped:
            return None
        book = funded_book(
            ledger,
            limits=pinned.limits,
            equity_ccy=equity,
            run_id=run_id,
            at=at,
            models=models,
        )
        record_book(ledger, book, run_id=run_id)
        return book

    return refresh


def _price_paper_venue(
    broker: Broker, *, store: BarStore, universe: dict[str, str], resolution: str
) -> None:
    """Give the paper venue its prices: the newest close the loop can see.

    Before this existed the paper broker had no prices at all, so every paper
    fill was at its 100.00 stand-in, every position was marked at its own
    fill, equity never moved and no stop ever fired — an overnight paper run
    exercised the order path but none of the numbers the risk rules read. The
    demo broker is left alone: Trading 212 prices its own fills.
    """
    from tb.broker.simulated import BarMarks, SimulatedBroker
    from tb.data.provider import Resolution

    if isinstance(broker, SimulatedBroker):
        venue = broker
        broker.price_source = BarMarks(
            bars=store,
            instruments=universe,
            resolution=Resolution(resolution),
            # Through the venue's clock rather than a copy of it, so a mark is
            # always read at the instant the venue is being asked.
            clock=lambda: venue.clock(),
        )


def _report(results: tuple[object, ...]) -> None:
    from tb.engine.loop import CycleResult

    table = Table(show_header=True)
    table.add_column("cycle", justify="right")
    table.add_column("regime")
    table.add_column("decisions", justify="right")
    table.add_column("orders", justify="right")
    table.add_column("stops", justify="right")
    table.add_column("fills", justify="right")
    table.add_column("refused", justify="right")
    table.add_column("ms", justify="right")

    for result in results:
        assert isinstance(result, CycleResult)
        table.add_row(
            str(result.cycle),
            result.regime.state.value if result.regime else "-",
            str(len(result.decisions)),
            str(len(result.submitted)),
            str(len(result.stops_placed)),
            str(len(result.fills_recorded)),
            f"[yellow]{len(result.refusals)}[/yellow]" if result.refusals else "0",
            f"{result.duration_ms:.0f}",
        )
    console.print(table)

    # Refusals are printed rather than only counted: "the loop ran and placed
    # nothing" is the normal state of a correct system on most days, and the
    # only way to tell it from a broken one is the reason.
    seen: set[str] = set()
    for result in results:
        assert isinstance(result, CycleResult)
        for ticker, reason in result.refusals:
            line = f"{ticker}: {reason}"
            if line not in seen:
                seen.add(line)
                console.print(f"  {WARN} {escape(line)}", soft_wrap=True)
        if result.detail and result.detail not in seen:
            seen.add(result.detail)
            console.print(f"  {WARN} cycle {result.cycle}: {escape(result.detail)}", soft_wrap=True)

    total_orders = sum(len(r.submitted) for r in results if isinstance(r, CycleResult))
    console.print(
        f"\n{OK} {len(results)} cycle(s), {total_orders} order(s). "
        "`tb reconcile` checks the three axes agree; `tb ledger verify` checks the chain."
    )


def replay(
    fill: Annotated[
        str, typer.Option("--fill", help="The fill to explain, as settlement recorded it.")
    ],
    limits: LimitsOpt = None,
    db: DbOpt = None,
    models: ModelsOpt = None,
) -> None:
    """Explain one fill from the ledger alone, and check the chain behind it.

    The fill, the order intent, the decision and the features it saw, every
    risk rule's verdict, the spec, the search trials, the holdout and the
    promotion behind it, and the round trip it closed. Each link that can be
    checked is: the hash chain, the feature hash, the spec hash, the spec
    reaching the recorded action again on the recorded features, the score of
    any model it read recomputed from the recorded inputs, and no blocking risk
    failure.

    Exits 1 when a check fails, 2 when the fill is not in the ledger.
    """
    from tb.engine.replay import ReplayError, replay_fill

    pinned = _load(limits)
    with _ledger(db, pinned) as ledger:
        try:
            replayed = replay_fill(
                ledger,
                fill,
                models=ModelStore(ledger, models or default_model_root(ledger.path)),
            )
        except ReplayError as exc:
            err_console.print(f"{BAD} {escape(str(exc))}", soft_wrap=True)
            raise typer.Exit(2) from exc

    row = replayed.fill
    console.print(
        f"[bold]fill[/bold] {row['fill_id']}  {row['t212_ticker']} {row['side']} "
        f"{row['quantity']} @ {row['price'] or 'unpriced'} on {row['filled_at'] or '-'}  "
        f"({row['source']}, {'admissible' if row['admissible_for_pnl'] else 'not admissible'} "
        "for P&L)"
    )
    if row["fees_json"]:
        console.print(f"  charges {escape(str(row['fees_json']))}")

    intent = replayed.intent
    if intent is None:
        console.print(
            f"  {WARN} no order intent: history reported an order this ledger never placed"
        )
    else:
        stop = f" stop {intent['stop_price']}" if intent["stop_price"] else ""
        console.print(
            f"[bold]intent[/bold] {intent['intent_id']}  {intent['purpose']} "
            f"{intent['order_type']} {intent['side']} {intent['quantity']}{stop}, "
            f"{intent['state']}; broker order {intent['broker_order_id'] or '-'}"
        )

    decision = replayed.decision
    if intent is not None and decision is None:
        console.print(f"  {WARN} no decision: a {intent['purpose']} answers none")
    if decision is not None:
        console.print(
            f"[bold]decision[/bold] {decision['decision_id']}  "
            f"{decision['strategy_id']}@v{decision['strategy_version']} "
            f"{decision['action']} at {decision['as_of_utc']}, declared edge "
            f"{decision['expected_edge_bps']}bps, regime {decision['regime_state']} "
            f"x{decision['regime_exposure_factor']}"
        )
        console.print(f"  {escape(str(decision['rationale'] or ''))}", soft_wrap=True)
        features = "  ".join(f"{name}={value}" for name, value in sorted(replayed.features.items()))
        console.print(f"  features  {escape(features)}", soft_wrap=True)

    if replayed.verdicts:
        table = Table(title="risk", show_header=True, box=None, padding=(0, 2, 0, 0))
        table.add_column("rule")
        table.add_column("verdict")
        table.add_column("observed", justify="right")
        table.add_column("limit", justify="right")
        for verdict in replayed.verdicts:
            table.add_row(
                str(verdict["rule_name"]),
                str(verdict["verdict"]),
                str(verdict["observed_value"] or ""),
                str(verdict["limit_value"] or ""),
            )
        console.print(table)

    spec = replayed.spec
    if decision is not None and spec is None:
        console.print(
            f"  {WARN} {decision['strategy_id']} is not a registered spec (the drill "
            "strategy is built in), so there is no search, holdout or gate behind it"
        )
    if spec is not None:
        console.print(
            f"[bold]spec[/bold] {spec['strategy_id']}@v{spec['version']}  lineage "
            f"{spec['lineage_id']}, {spec['author_kind']}-authored, registered "
            f"{spec['registered_at']}"
        )
        console.print(f"  {escape(str(spec['spec_json']))}", soft_wrap=True)
        searches = sorted({str(trial["search_id"]) for trial in replayed.trials})
        console.print(
            f"  {len(replayed.trials)} trial(s) of this spec, in "
            f"{', '.join(searches) if searches else 'no recorded search'}"
        )
        holdout = replayed.holdout
        console.print(
            "  holdout "
            + (
                "never evaluated"
                if holdout is None
                else f"{'passed' if holdout['passed'] else 'failed'} on {holdout['vintage_id']}: "
                f"{holdout['n_trades']} trades, net Sharpe {holdout['net_sharpe']}"
            )
        )
        promotion = replayed.promotion
        console.print(
            "  gate "
            + (
                "never run"
                if promotion is None
                else f"{promotion['decision']} ({promotion['n_failed']} of {promotion['n_gates']} "
                f"failed), deflated Sharpe {promotion['deflated_sharpe']}, "
                f"at {promotion['decided_at']}"
            )
        )

    trip = replayed.round_trip
    if trip is not None:
        owner = f"charged to {trip['strategy_id']}" if trip["charged"] else "charged to nobody"
        console.print(
            f"[bold]round trip[/bold] closed {trip['quantity']} at {trip['exit_price']} against "
            f"a basis of {trip['cost_basis']}: {trip['pnl_ccy']}, {owner}"
        )

    console.print("")
    for check in replayed.checks:
        console.print(
            f"{OK if check.ok else BAD} {check.name}: {escape(check.detail)}", soft_wrap=True
        )
    if not replayed.ok:
        raise typer.Exit(1)
