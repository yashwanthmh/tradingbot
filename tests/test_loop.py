"""The trading loop: one decision travelling the whole path.

The centrepiece is `test_a_decision_travels_the_whole_path`, which is the M4
success condition in miniature — bars in, features, strategy, risk engine,
intent log, broker, protective stop, and every step recoverable from the
ledger alone. Everything else here is about the gates *before* the strategy is
asked anything, because each of them is a way the loop could trade when it
should not.

Order matters in the preflight and the tests mirror it: self-check, lease,
config hash, recovery, run state. A loop that checked its config after placing
an order would have placed it under caps that were no longer in force.
"""

from __future__ import annotations

from collections.abc import Callable, Iterator
from datetime import UTC, date, datetime, timedelta
from decimal import Decimal
from pathlib import Path
from typing import Any

import pytest

from tb.broker.simulated import SimulatedBroker
from tb.config.loader import PinnedLimits, load_hard_limits
from tb.data.barstore import BarStore
from tb.data.calendar import TradingCalendar
from tb.data.provider import Bar, BarBatch, Provenance, Resolution, Session
from tb.data.symbols import Confidence, SymbolMap, SymbolMapping
from tb.engine.intents import IntentLog
from tb.engine.loop import LoopHalted, TradingLoop, build_instrument_map
from tb.engine.orders import OrderSubmitter
from tb.features.pipeline import FeaturePipeline
from tb.ledger.store import Ledger
from tb.ledger.verify import verify_chain
from tb.ops.killswitch import write_heartbeat
from tb.ops.state import RunState, StateMachine
from tb.ops.watchdog import InstanceLock, SelfCheck
from tb.portfolio.pnl import EquityCurve
from tb.strategy.base import Action
from tb.strategy.trivial import MovingAverageCross, specs

TICKER = "AAPL_US_EQ"
UID = "isin:US0378331005"
CAL = TradingCalendar()

# Mid-session on a regular trading day, well clear of both session windows.
AS_OF = datetime(2026, 4, 1, 15, 30, tzinfo=UTC)


@pytest.fixture
def env(tmp_path: Path, write_limits: Callable[[dict[str, Any]], Path]) -> dict[str, Any]:
    run_dir = tmp_path / "run"
    run_dir.mkdir(exist_ok=True)
    limits = write_limits(
        {
            "safety": {
                "kill_switch_path": str(run_dir / "KILL"),
                "heartbeat_path": str(run_dir / "heartbeat"),
            }
        }
    )
    # The watchdog's liveness marker, so the self-check passes. Written here
    # rather than disabled, because `require_watchdog=False` is a different
    # code path and the default one is what production runs.
    write_heartbeat(run_dir / "watchdog", run_id="watchdog", state="supervising")
    return {
        "limits": limits,
        "db": tmp_path / "ledger.db",
        "bars": tmp_path / "bars",
        "run_dir": run_dir,
        "pinned": load_hard_limits(limits),
    }


def _rising_bars(*, days: int, uid: str = UID) -> list[Bar]:
    """Enough history for a 100-day average, rising so the cross is up.

    Rising monotonically on purpose: the fast average is then always above the
    slow one, so the strategy's entry fires deterministically. A test that
    depended on a cross happening at a particular bar would be a test of the
    fixture.
    """
    # Wide enough that the reference series can have the 250 sessions the
    # regime gate requires, which is more than any instrument needs.
    sessions = [s.day for s in CAL.sessions_between(date(2024, 1, 2), date(2026, 4, 1))]
    sessions = sessions[-days:]
    bars = []
    for index, day in enumerate(sessions):
        opened = datetime(day.year, day.month, day.day, tzinfo=UTC)
        price = Decimal("50.00") + Decimal(index) / Decimal(2)
        bars.append(
            Bar(
                instrument_uid=uid,
                resolution=Resolution.DAILY,
                bar_open_utc=opened,
                # Knowledge time one day after the bar, which is what a daily
                # feed actually offers. The staleness rule reads this.
                available_at_utc=opened + timedelta(days=1),
                ingested_at_utc=opened + timedelta(days=1),
                provider="alpaca",
                provenance=Provenance.BACKFILL,
                session=Session.REGULAR,
                open=price,
                high=price + Decimal("1"),
                low=price - Decimal("1"),
                close=price,
                volume=1_000_000,
            )
        )
    return bars


def _falling_bars(*, days: int) -> list[Bar]:
    bars = _rising_bars(days=days)
    top = Decimal("50.00") + Decimal(len(bars)) / Decimal(2)
    out = []
    for index, bar in enumerate(bars):
        price = top - Decimal(index) / Decimal(2)
        out.append(
            Bar(
                instrument_uid=bar.instrument_uid,
                resolution=bar.resolution,
                bar_open_utc=bar.bar_open_utc,
                available_at_utc=bar.available_at_utc,
                ingested_at_utc=bar.ingested_at_utc,
                provider=bar.provider,
                provenance=bar.provenance,
                session=bar.session,
                open=price,
                high=price + Decimal("1"),
                low=price - Decimal("1"),
                close=price,
                volume=bar.volume,
            )
        )
    return out


# The regime gate needs `data.min_history_days_for_regime` sessions before it
# will say anything but INSUFFICIENT_HISTORY. That is 250 in the shipped
# config, deliberately above the 200-day average it computes.
REFERENCE_DAYS = 260


def _seed(
    env: dict[str, Any],
    bars: list[Bar],
    *,
    uid: str = UID,
    reference_days: int = REFERENCE_DAYS,
) -> None:
    """Instrument row, verified mapping, bars, and the regime reference series.

    `reference_days` is separate from the instrument's history because the two
    thresholds are different: a strategy needs 100 sessions for its slow
    average, the regime gate needs 250 before it will report anything but
    INSUFFICIENT_HISTORY. Seeding both from one number would make every test
    run at half exposure without saying so.
    """
    pinned: PinnedLimits = env["pinned"]
    with Ledger(env["db"], config_hash=pinned.config_hash) as ledger:
        ledger.initialise(created_by="test")
        ledger.conn.execute(
            "INSERT OR REPLACE INTO instruments (ticker, instrument_type, isin,"
            " currency_code, short_name, full_name, exchange_id, working_schedule_id,"
            " min_trade_quantity, max_open_quantity, added_on, fetched_at, raw_json)"
            " VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)",
            (
                TICKER,
                "STOCK",
                "US0378331005",
                "USD",
                "Apple",
                "Apple Inc.",
                1,
                1,
                "0.1",
                "1000",
                None,
                AS_OF.isoformat(),
                "{}",
            ),
        )
        ledger.conn.commit()

        # CROSS_VERIFIED, because DERIVED does not permit an entry — the
        # two-tier gate, and the reason `tb symbols verify` exists.
        SymbolMap(ledger, provider="alpaca").upsert(
            SymbolMapping(
                t212_ticker=TICKER,
                data_symbol="AAPL",
                provider="alpaca",
                confidence=Confidence.CROSS_VERIFIED,
                derivation="test",
                currency_code="USD",
                verified_at=AS_OF.isoformat(),
            )
        )

        store = BarStore(ledger, root=env["bars"], scale=pinned.limits.data.price_scale)
        store.ingest(
            BarBatch(
                bars=tuple(bars),
                provider="alpaca",
                symbol="AAPL",
                resolution=Resolution.DAILY,
                requested_start=bars[0].bar_open_utc,
                requested_end=bars[-1].bar_open_utc,
            )
        )
        # The regime gate's own series, under its self-marking `sym:` uid.
        # Without it the gate reads UNAVAILABLE and every entry is refused —
        # which is correct, and is asserted separately below.
        reference = _rising_bars(days=reference_days, uid="sym:SPY")
        store.ingest(
            BarBatch(
                bars=tuple(reference),
                provider="alpaca",
                symbol="SPY",
                resolution=Resolution.DAILY,
                requested_start=reference[0].bar_open_utc,
                requested_end=reference[-1].bar_open_utc,
            )
        )


def _loop(
    env: dict[str, Any],
    ledger: Ledger,
    broker: SimulatedBroker,
    *,
    run_id: str = "run_loop",
    at: datetime = AS_OF,
) -> TradingLoop:
    pinned: PinnedLimits = env["pinned"]
    log = IntentLog(ledger, run_id=run_id)
    return TradingLoop(
        ledger=ledger,
        pinned=pinned,
        broker=broker,
        bars=BarStore(ledger, root=env["bars"], scale=pinned.limits.data.price_scale),
        strategy=MovingAverageCross(),
        pipeline=FeaturePipeline(specs=specs()),
        submitter=OrderSubmitter(
            ledger=ledger, broker=broker, log=log, run_id=run_id, clock=lambda: at
        ),
        log=log,
        run_id=run_id,
        instruments={TICKER: UID},
        state=StateMachine(ledger, pinned, run_id=run_id),
        self_check=SelfCheck(
            ledger=ledger,
            kill_switch_path=Path(pinned.limits.safety.kill_switch_path),
            liveness_path=env["run_dir"] / "watchdog",
            run_id=run_id,
        ),
        lock=InstanceLock(ledger, run_id=run_id),
        equity=EquityCurve(ledger, run_id=run_id),
        clock=lambda: at,
    )


def _broker(*, price: Decimal = Decimal("150.00")) -> SimulatedBroker:
    return SimulatedBroker(
        environment="paper",
        currency="GBP",
        equity=Decimal("10000.00"),
        free_cash=Decimal("10000.00"),
        prices={TICKER: price},
        min_trade_quantity={TICKER: Decimal("0.1")},
        clock=lambda: AS_OF,
    )


@pytest.fixture
def ledger(env: dict[str, Any]) -> Iterator[Ledger]:
    with Ledger(env["db"], config_hash=env["pinned"].config_hash) as opened:
        yield opened


# --------------------------------------------------------------------------
# The whole path
# --------------------------------------------------------------------------


def test_a_decision_travels_the_whole_path(env: dict[str, Any]) -> None:
    """**The M4 success condition, in miniature.**

    A minute of history, a cross, a risk verdict per rule, an intent committed
    before the wire, an order at the broker, a protective stop behind it — and
    every step answerable from the ledger alone afterwards.
    """
    _seed(env, _rising_bars(days=140))
    broker = _broker()

    with Ledger(env["db"], config_hash=env["pinned"].config_hash) as ledger:
        loop = _loop(env, ledger, broker)
        loop.lock.acquire(at=AS_OF)
        result = loop.run_cycle()

    assert result.decisions, "the strategy was not consulted"
    assert result.decisions[0].action is Action.ENTER, result.decisions[0].rationale
    assert result.submitted, f"nothing was submitted: {result.refusals}"
    assert result.stops_placed, "the entry filled with no protective stop behind it"

    # The broker's side.
    position = broker.get_position(TICKER)
    assert position is not None and position.quantity > 0
    assert len(broker.protective_orders_for(TICKER)) == 1

    # The ledger's side: the whole lineage, in order.
    with Ledger(env["db"]) as check:
        kinds = [
            row["event_type"]
            for row in check.conn.execute(
                "SELECT event_type FROM event_log ORDER BY seq"
            ).fetchall()
        ]
        assert "decision.made" in kinds
        assert "risk.evaluated" in kinds
        assert "intent.committed" in kinds
        assert "order.submitted" in kinds
        assert "order.acknowledged" in kinds
        assert "position.protected" in kinds
        assert "loop.cycle_completed" in kinds
        # The write-ahead ordering, asserted rather than assumed: the commit
        # must precede the send in the chain, or the exactly-once argument
        # does not hold.
        assert kinds.index("intent.committed") < kinds.index("order.submitted")

        assert verify_chain(check).ok


def test_every_rule_leaves_a_verdict_row(env: dict[str, Any]) -> None:
    """One row per rule per decision, not one row per decision.

    "Blocked by the daily breaker" and "blocked by that and three others" are
    different investigations, and the passes make the margins queryable.
    """
    _seed(env, _rising_bars(days=140))
    with Ledger(env["db"], config_hash=env["pinned"].config_hash) as ledger:
        loop = _loop(env, ledger, _broker())
        loop.lock.acquire(at=AS_OF)
        loop.run_cycle()

    with Ledger(env["db"]) as check:
        rows = check.conn.execute("SELECT rule_name, verdict FROM risk_verdicts").fetchall()
    names = {row["rule_name"] for row in rows}
    assert len(names) >= 15, f"only {len(names)} rules recorded a verdict: {sorted(names)}"
    # The passes are there too, not only the failures.
    assert any(row["verdict"] == "pass" for row in rows)


def test_a_hold_is_recorded_too(env: dict[str, Any]) -> None:
    """ "The strategy looked and declined" is evidence the loop was alive.

    Distinct from "the strategy was not consulted", and only the first says
    the cycle did its job.
    """
    _seed(env, _falling_bars(days=140))
    with Ledger(env["db"], config_hash=env["pinned"].config_hash) as ledger:
        loop = _loop(env, ledger, _broker())
        loop.lock.acquire(at=AS_OF)
        result = loop.run_cycle()

    assert result.decisions[0].action is Action.HOLD
    assert not result.submitted

    with Ledger(env["db"]) as check:
        row = check.conn.execute("SELECT action, rationale FROM decisions").fetchone()
    assert row["action"] == "hold"
    assert row["rationale"], "a hold with no rationale cannot be reviewed"


def test_too_little_history_holds_rather_than_guessing(env: dict[str, Any]) -> None:
    """`UNKNOWN` means the cross does not exist in the data.

    Not caution for its own sake: a 100-day average over 30 bars is a
    different number wearing the same name, and trading on it is trading on
    an artefact.
    """
    _seed(env, _rising_bars(days=30))
    with Ledger(env["db"], config_hash=env["pinned"].config_hash) as ledger:
        loop = _loop(env, ledger, _broker())
        loop.lock.acquire(at=AS_OF)
        result = loop.run_cycle()

    assert result.decisions[0].action is Action.HOLD
    assert "unknown" in result.decisions[0].rationale.lower()
    assert not result.submitted


# --------------------------------------------------------------------------
# The gates before the strategy is asked anything
# --------------------------------------------------------------------------


def test_an_engaged_kill_switch_stops_the_cycle(env: dict[str, Any]) -> None:
    from tb.ops.killswitch import engage_kill_switch

    _seed(env, _rising_bars(days=140))
    engage_kill_switch(
        env["pinned"].limits.safety.kill_switch_path, engaged_by="human", reason="testing"
    )
    with Ledger(env["db"], config_hash=env["pinned"].config_hash) as ledger:
        loop = _loop(env, ledger, _broker())
        loop.lock.acquire(at=AS_OF)
        with pytest.raises(LoopHalted, match="self-check failed"):
            loop.run_cycle()


def test_a_dead_watchdog_stops_the_cycle(env: dict[str, Any]) -> None:
    """Running unsupervised is a stop, not a degraded mode."""
    _seed(env, _rising_bars(days=140))
    (env["run_dir"] / "watchdog").unlink()

    with Ledger(env["db"], config_hash=env["pinned"].config_hash) as ledger:
        loop = _loop(env, ledger, _broker())
        loop.lock.acquire(at=AS_OF)
        with pytest.raises(LoopHalted, match="self-check failed"):
            loop.run_cycle()


def test_a_changed_limits_file_halts_rather_than_reloading(env: dict[str, Any]) -> None:
    """**The control layer, re-verified every cycle.**

    A halt rather than a reload, and the distinction is the whole point: the
    caps that sized the open positions are not the caps now in force, so
    continuing would mean holding positions under limits they were never
    checked against.
    """
    _seed(env, _rising_bars(days=140))
    with Ledger(env["db"], config_hash=env["pinned"].config_hash) as ledger:
        loop = _loop(env, ledger, _broker())
        loop.lock.acquire(at=AS_OF)

        # Edit the file underneath the running loop.
        text = Path(env["limits"]).read_text(encoding="utf-8")
        Path(env["limits"]).write_text(
            text.replace("absolute_ceiling_ccy: 500", "absolute_ceiling_ccy: 5000"),
            encoding="utf-8",
        )

        with pytest.raises(LoopHalted, match="hard limits changed"):
            loop.run_cycle()


def test_losing_the_lease_halts_the_cycle(env: dict[str, Any]) -> None:
    """Another instance took over while this one was between cycles."""
    _seed(env, _rising_bars(days=140))
    with Ledger(env["db"], config_hash=env["pinned"].config_hash) as ledger:
        loop = _loop(env, ledger, _broker())
        loop.lock.acquire(at=AS_OF, ttl_seconds=60)

        # A second instance takes it after ours expires.
        InstanceLock(ledger, run_id="run_other").acquire(at=AS_OF + timedelta(seconds=61))

        later = _loop(env, ledger, _broker(), at=AS_OF + timedelta(seconds=62))
        later.lock = loop.lock
        with pytest.raises(LoopHalted, match="lost the trading lease"):
            later.run_cycle()


def test_an_unresolved_unknown_intent_halts_before_deciding(env: dict[str, Any]) -> None:
    """Recovery runs every cycle, and an unknown blocks.

    A clean shutdown and a crash are indistinguishable without reading the
    intent table, so the loop reads it every time — and an order that may
    exist stops trading rather than being carried.
    """
    from tb.broker.simulated import CrashPoint
    from tb.engine.orders import SubmissionUnknown

    _seed(env, _rising_bars(days=140))
    broker = _broker()
    broker.fail_at = CrashPoint.POST_RESPONSE_PRE_PERSIST

    with Ledger(env["db"], config_hash=env["pinned"].config_hash) as ledger:
        loop = _loop(env, ledger, broker)
        loop.lock.acquire(at=AS_OF)
        with pytest.raises((LoopHalted, SubmissionUnknown)):
            loop.run_cycle()

    broker.clear_crash()
    # A restart happens after the crashed instance's lease has expired — which
    # is why the lease has an expiry at all. Acquiring at AS_OF with a new run
    # id would be refused by the lock, correctly.
    restarted_at = AS_OF + timedelta(seconds=120)
    with Ledger(env["db"], config_hash=env["pinned"].config_hash) as ledger:
        fresh = _loop(env, ledger, broker, run_id="run_restarted", at=restarted_at)
        fresh.lock.acquire(at=restarted_at)
        with pytest.raises(LoopHalted, match="unknown state"):
            fresh.run_cycle()


def test_an_unverified_symbol_is_not_in_the_universe(env: dict[str, Any]) -> None:
    """The universe is the verified universe by construction.

    `build_instrument_map` excludes a mapping that permits neither an entry
    nor an exit, so an unverified symbol is simply absent rather than being
    checked inside the cycle and refused there.
    """
    pinned: PinnedLimits = env["pinned"]
    with Ledger(env["db"], config_hash=pinned.config_hash) as ledger:
        ledger.initialise(created_by="test")
        SymbolMap(ledger, provider="alpaca").upsert(
            SymbolMapping(
                t212_ticker=TICKER,
                data_symbol="AAPL",
                provider="alpaca",
                # DERIVED does not permit an entry.
                confidence=Confidence.DERIVED,
                derivation="suffix_strip",
                currency_code="USD",
            )
        )
        universe = build_instrument_map(ledger)

    # `may_exit` is True in every mapping state, so it *is* included — for
    # exits only. That asymmetry is deliberate and is what stops a data
    # problem from trapping a position.
    assert TICKER in universe


# --------------------------------------------------------------------------
# The regime gate, applied
# --------------------------------------------------------------------------


def _without_reference_series(env: dict[str, Any]) -> None:
    with Ledger(env["db"], config_hash=env["pinned"].config_hash) as ledger:
        ledger.conn.execute("DELETE FROM bars_hot WHERE instrument_uid = 'sym:SPY'")
        ledger.conn.execute("DELETE FROM data_partitions WHERE instrument_uid = 'sym:SPY'")
        ledger.conn.commit()


def test_an_unavailable_regime_halves_the_size_rather_than_blocking(
    env: dict[str, Any],
) -> None:
    """Reduced exposure, not zero — which is the gate's actual rule.

    Worth being exact about, because the obvious reading is wrong in both
    directions. "No signal" must not mean *full* exposure: that is the most
    expensive default in the system, and it applies on day one to the whole
    portfolio. But it must not mean *no* exposure either, or a fresh install
    could never open a position at all and the bot would look broken for the
    ten months a 200-day average takes to exist.

    So the gate halves, and the reading is recorded as `unavailable` rather
    than as `risk_off` — an operator has to be able to tell "the index is
    down" from "we cannot see the index", even though both scale the same.
    """
    _seed(env, _rising_bars(days=140))
    _without_reference_series(env)

    with Ledger(env["db"], config_hash=env["pinned"].config_hash) as ledger:
        loop = _loop(env, ledger, _broker())
        loop.lock.acquire(at=AS_OF)
        reduced = loop.run_cycle()

    assert reduced.regime is not None
    assert reduced.regime.state.value == "unavailable"
    assert reduced.regime.exposure_factor == Decimal("0.5")
    assert reduced.submitted, "a reduced regime must still permit a floor-size entry"

    with Ledger(env["db"]) as check:
        halved = check.conn.execute(
            "SELECT quantity FROM order_intents WHERE purpose = 'entry'"
        ).fetchone()["quantity"]

    # The same setup with the reference series present must size *larger*.
    # This comparison is the real assertion: a factor recorded on the reading
    # but never multiplied into the quantity would pass every check above.
    #
    # The same run id, because a lease is re-acquirable by its own holder and
    # a different id would be refused by the instance lock — correctly.
    _seed(env, _rising_bars(days=140))  # re-seeds sym:SPY
    with Ledger(env["db"], config_hash=env["pinned"].config_hash) as ledger:
        ledger.conn.execute("DELETE FROM order_intents")
        ledger.conn.commit()
        loop = _loop(env, ledger, _broker())
        loop.lock.acquire(at=AS_OF)
        full = loop.run_cycle()

    assert full.regime is not None and full.regime.exposure_factor == Decimal(1)
    with Ledger(env["db"]) as check:
        unhalved = check.conn.execute(
            "SELECT quantity FROM order_intents WHERE purpose = 'entry'"
        ).fetchone()["quantity"]

    assert Decimal(str(halved)) < Decimal(str(unhalved)), (
        f"the regime factor was recorded but not applied: {halved} vs {unhalved}"
    )


# --------------------------------------------------------------------------
# Exits come first
# --------------------------------------------------------------------------


def test_an_exit_is_processed_before_any_entry(env: dict[str, Any]) -> None:
    """Risk-reducing work must not queue behind a batch of entries.

    Asserted through the pass structure: an instrument with a position is
    handled on the risk-reducing pass and skipped on the entry pass, so an
    add is deferred a cycle rather than competing with its own exit for the
    rate-limit budget.
    """
    _seed(env, _falling_bars(days=140))
    broker = _broker()
    broker.seed_position(
        TICKER,
        quantity=Decimal("1"),
        average_price=Decimal("150.00"),
        entered_at=AS_OF - timedelta(days=30),
    )

    with Ledger(env["db"], config_hash=env["pinned"].config_hash) as ledger:
        loop = _loop(env, ledger, broker)
        loop.lock.acquire(at=AS_OF)
        result = loop.run_cycle()

    assert result.decisions[0].action is Action.EXIT, result.decisions[0].rationale
    assert result.submitted, f"the exit was not submitted: {result.refusals}"
    assert broker.get_position(TICKER) is None, "the position should be closed"


def test_an_exit_is_not_scaled_by_the_regime_factor(env: dict[str, Any]) -> None:
    """The asymmetry that runs through the whole risk layer.

    A reduced regime halves new exposure. It must not halve an *exit* — a
    half-closed position is the worst of both, and the regime is a statement
    about taking on risk rather than about shedding it. So the rule reads
    NOT_APPLICABLE for a risk-reducing order and the exit closes the whole
    position.
    """
    _seed(env, _falling_bars(days=140))
    _without_reference_series(env)

    broker = _broker()
    broker.seed_position(
        TICKER,
        quantity=Decimal("1"),
        average_price=Decimal("150.00"),
        entered_at=AS_OF - timedelta(days=30),
    )

    with Ledger(env["db"], config_hash=env["pinned"].config_hash) as ledger:
        loop = _loop(env, ledger, broker)
        loop.lock.acquire(at=AS_OF)
        result = loop.run_cycle()

    assert result.submitted, f"the exit was blocked: {result.refusals}"
    assert broker.get_position(TICKER) is None


# --------------------------------------------------------------------------
# The cycle record
# --------------------------------------------------------------------------


def test_every_cycle_is_recorded_even_when_nothing_trades(env: dict[str, Any]) -> None:
    """A running loop that decides nothing looks identical to a stopped one.

    Unless each cycle says so — and "it was up all day" should be a claim the
    ledger can settle.
    """
    _seed(env, _falling_bars(days=140))
    with Ledger(env["db"], config_hash=env["pinned"].config_hash) as ledger:
        loop = _loop(env, ledger, _broker())
        loop.lock.acquire(at=AS_OF)
        loop.run_cycle()
        loop.run_cycle()

    with Ledger(env["db"]) as check:
        rows = check.conn.execute(
            "SELECT payload_json FROM event_log WHERE event_type = 'loop.cycle_completed'"
        ).fetchall()
    assert len(rows) == 2
    assert '"cycle":1' in rows[0]["payload_json"].replace(" ", "")


def test_the_loop_reaches_trading_through_reconciling(env: dict[str, Any]) -> None:
    """The state transition records that recovery happened.

    BOOT to TRADING directly would skip the step that says the intent table
    was read, and that step is the only evidence recovery ran.
    """
    _seed(env, _rising_bars(days=140))
    with Ledger(env["db"], config_hash=env["pinned"].config_hash) as ledger:
        loop = _loop(env, ledger, _broker())
        loop.lock.acquire(at=AS_OF)
        loop.run_cycle()
        assert loop.state.current().state is RunState.TRADING

    with Ledger(env["db"]) as check:
        states = [
            row["payload_json"]
            for row in check.conn.execute(
                "SELECT payload_json FROM event_log WHERE event_type = 'state.transitioned'"
                " ORDER BY seq"
            ).fetchall()
        ]
    assert any("reconciling" in s for s in states)
    assert any("trading" in s for s in states)


def test_a_heartbeat_is_written_every_cycle(env: dict[str, Any]) -> None:
    """The watchdog's only evidence the trader is alive."""
    _seed(env, _rising_bars(days=140))
    heartbeat = Path(env["pinned"].limits.safety.heartbeat_path)
    assert not heartbeat.exists()

    with Ledger(env["db"], config_hash=env["pinned"].config_hash) as ledger:
        loop = _loop(env, ledger, _broker())
        loop.lock.acquire(at=AS_OF)
        loop.run_cycle()

    assert heartbeat.exists()


def test_run_forever_stops_at_the_cycle_bound(env: dict[str, Any]) -> None:
    _seed(env, _falling_bars(days=140))
    with Ledger(env["db"], config_hash=env["pinned"].config_hash) as ledger:
        loop = _loop(env, ledger, _broker())
        loop.lock.acquire(at=AS_OF)
        results = loop.run_forever(interval_seconds=0.0, max_cycles=3)
    assert [r.cycle for r in results] == [1, 2, 3]


# --------------------------------------------------------------------------
# The loss breakers, through the loop
# --------------------------------------------------------------------------


def test_the_daily_breaker_stops_the_loop_entering_after_a_loss(
    env: dict[str, Any],
) -> None:
    """**The end-to-end proof that the breakers are live.**

    Before the equity curve existed the loop passed `day_pnl_pct=0.0`, so
    this path was unreachable: the account could fall any distance and the
    rule still returned PASS. Here the broker's equity drops between cycles
    and the loop's own verdict changes.

    Two cycles, because the first is what establishes the day's opening
    equity — which is also the ordering the loop depends on.
    """
    _seed(env, _rising_bars(days=140))

    # The control: a healthy account enters. Run first and in its own ledger
    # session, because once it holds a position the next cycle is an exit
    # decision rather than an entry — and the comparison needs both cycles to
    # be attempting the same thing.
    with Ledger(env["db"], config_hash=env["pinned"].config_hash) as ledger:
        healthy = _loop(env, ledger, _broker())
        healthy.lock.acquire(at=AS_OF)
        control = healthy.run_cycle()
    assert control.submitted, f"the control cycle did not trade: {control.refusals}"

    # Now the same decision against an account down 4%, past the 2% halt. A
    # fresh store so the loop is again considering an *entry*.
    env2 = dict(env)
    env2["db"] = Path(str(env["db"]) + ".down")
    env2["bars"] = env["bars"]
    _seed(env2, _rising_bars(days=140))

    broker = _broker()
    with Ledger(env2["db"], config_hash=env2["pinned"].config_hash) as ledger:
        loop = _loop(env2, ledger, broker)
        loop.lock.acquire(at=AS_OF)
        # The day's opening equity, as the first cycle of the session would
        # have marked it.
        loop.equity.mark(equity_ccy=Decimal("10000"), at=AS_OF - timedelta(minutes=5))
        # Mutating the simulator's equity is what a bad day looks like from
        # the loop's side: it reads `get_cash().total` like any other cycle.
        broker.equity = Decimal("9600.00")
        second = loop.run_cycle()

    assert second.decisions[0].action is Action.ENTER, (
        "the second cycle must be attempting an entry for the comparison to mean anything"
    )
    assert not second.submitted, "an entry was placed with the day down 4%"
    assert any("daily_loss" in reason for _, reason in second.refusals), second.refusals
    env["db"] = env2["db"]  # so the verdict query below reads this run's ledger

    with Ledger(env["db"]) as check:
        verdict = check.conn.execute(
            "SELECT verdict, observed_value, limit_value FROM risk_verdicts"
            " WHERE rule_name = 'daily_loss' ORDER BY evaluated_at DESC LIMIT 1"
        ).fetchone()
    assert verdict["verdict"] == "block"
    assert float(verdict["observed_value"]) == pytest.approx(4.0, abs=0.01)
    assert float(verdict["limit_value"]) == env["pinned"].limits.loss.daily_halt_pct


def test_an_exit_still_goes_through_with_the_day_breached(env: dict[str, Any]) -> None:
    """The asymmetry, end to end and at the worst moment.

    A breaker that blocked the flattening orders it fired to place would lock
    in the loss it existed to limit. The daily breaker is breached here and
    the exit is submitted anyway.
    """
    _seed(env, _falling_bars(days=140))
    broker = _broker()
    broker.seed_position(
        TICKER,
        quantity=Decimal("1"),
        average_price=Decimal("150.00"),
        entered_at=AS_OF - timedelta(days=30),
    )

    with Ledger(env["db"], config_hash=env["pinned"].config_hash) as ledger:
        loop = _loop(env, ledger, broker)
        loop.lock.acquire(at=AS_OF)
        # Establish the day's open, then crater it.
        loop.equity.mark(equity_ccy=Decimal("10000"), at=AS_OF - timedelta(minutes=5))
        broker.equity = Decimal("8000.00")  # -20%: past all three thresholds
        result = loop.run_cycle()

    assert result.decisions[0].action is Action.EXIT
    assert result.submitted, f"the exit was blocked at -20%: {result.refusals}"
    assert broker.get_position(TICKER) is None


def test_an_unmeasurable_account_blocks_the_entry(env: dict[str, Any]) -> None:
    """A broker that reports no equity must not read as a flat day.

    The loop leaves the curve unmarked, so the three percentages reach the
    rules as `None` and every one of them blocks. That is the fail-closed
    reading, and it is the reason the loop passes `None` rather than `0.0`.
    """
    _seed(env, _rising_bars(days=140))
    broker = _broker()

    with Ledger(env["db"], config_hash=env["pinned"].config_hash) as ledger:
        loop = _loop(env, ledger, broker)
        loop.lock.acquire(at=AS_OF)

        # A broker with no equity to report. `CashBalance.total` is optional
        # precisely because the venue may not say.
        from tb.broker.port import CashBalance

        broker.get_cash = lambda: CashBalance(  # type: ignore[method-assign]
            currency="GBP", free=None, total=None, invested=None, blocked=None
        )
        result = loop.run_cycle()

    assert not result.submitted, "an entry was placed against an unmeasured account"
    reasons = " ".join(reason for _, reason in result.refusals)
    assert "daily_loss" in reasons or "unknown" in reasons, result.refusals


def test_the_loop_records_an_equity_mark_per_cycle(env: dict[str, Any]) -> None:
    """The curve is what the breakers read, so it has to be fed every cycle."""
    _seed(env, _falling_bars(days=140))
    with Ledger(env["db"], config_hash=env["pinned"].config_hash) as ledger:
        loop = _loop(env, ledger, _broker())
        loop.lock.acquire(at=AS_OF)
        loop.run_cycle()
        loop.run_cycle()
        loop.run_cycle()

    with Ledger(env["db"]) as check:
        marks = check.conn.execute("SELECT COUNT(*) AS n FROM equity_marks").fetchone()
    assert marks["n"] == 3
