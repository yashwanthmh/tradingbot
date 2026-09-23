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
from contextlib import suppress
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
) -> None:
    """The trading loop: bars in, orders out, every step in the ledger.

    Refuses to start rather than starting degraded. A missing ledger, an
    unverified universe, a held lease or an unresolved unknown intent each
    stop it here with the remedy named — because a loop that started anyway
    and halted on its first cycle would have already written a run record and
    a heartbeat, which makes the failure harder to read rather than easier.

    `--cycles 1` by default, so an accidental invocation does one pass and
    stops. Pass `--cycles 0` to run until halted.
    """
    pinned = _load(limits)
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

        loop = TradingLoop(
            ledger=ledger,
            pinned=pinned,
            broker=broker,
            bars=store,
            strategy=MovingAverageCross(),
            pipeline=FeaturePipeline(specs=specs()),
            submitter=submitter,
            log=log,
            run_id=run_id,
            instruments=universe,
            state=StateMachine(ledger, pinned, run_id=run_id),
            self_check=self_check,
            lock=lock,
            equity=EquityCurve(ledger, run_id=run_id),
        )

        console.print(
            f"run [bold]{run_id}[/bold] in [bold]{mode}[/bold] mode over "
            f"{len(universe)} instrument(s), lease to {lease.expires_at.isoformat()}"
        )
        if no_watchdog:
            console.print(
                f"{WARN} running without a supervisor. A wedged loop is the one state it "
                "cannot detect about itself, so nothing would notice.",
                soft_wrap=True,
            )

        try:
            results = loop.run_forever(
                interval_seconds=interval, max_cycles=None if cycles == 0 else cycles
            )
        except LoopHalted as exc:
            err_console.print(f"\n{BAD} halted: {escape(str(exc))}", soft_wrap=True)
            lock.release()
            raise typer.Exit(1) from exc
        finally:
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
        while cycles == 0 or checked < cycles:
            verdict = dog.check()
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


def _broker(mode: str, pinned: PinnedLimits, *, equity: Decimal) -> Broker:
    """The broker for this mode.

    `paper` is the simulated broker rather than the demo account. That is the
    honest naming: a paper run reaches no network at all, so it tests the loop
    and not the adapter. `demo` is a real Trading 212 account with fake money,
    which tests both.
    """
    if mode == "paper":
        from tb.broker.simulated import SimulatedBroker

        return SimulatedBroker(
            environment="paper",
            currency=pinned.limits.currency,
            equity=equity,
            free_cash=equity,
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


def _report(results: tuple[object, ...]) -> None:
    from tb.engine.loop import CycleResult

    table = Table(show_header=True)
    table.add_column("cycle", justify="right")
    table.add_column("regime")
    table.add_column("decisions", justify="right")
    table.add_column("orders", justify="right")
    table.add_column("stops", justify="right")
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

    total_orders = sum(len(r.submitted) for r in results if isinstance(r, CycleResult))
    console.print(
        f"\n{OK} {len(results)} cycle(s), {total_orders} order(s). "
        "`tb reconcile` checks the three axes agree; `tb ledger verify` checks the chain."
    )
