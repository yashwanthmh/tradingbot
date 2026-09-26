"""The equity curve, and the loss breakers actually firing.

The gap this closes: the loop passed `day_pnl_pct=0.0`, so `DailyLossRule`
read a flat day forever. Three breakers, a verdict row each, unable to block
anything — and a ledger full of `daily_loss: pass` reads as evidence the
breaker was watching.

So the tests that matter here are the ones at the bottom:
`test_the_daily_breaker_blocks_an_entry_once_the_day_is_down` and its two
siblings drive real equity marks through the curve and assert the rules
change their verdict. Everything above them is about the arithmetic being
right and the cold start being honest rather than absent.
"""

from __future__ import annotations

from collections.abc import Iterator
from datetime import UTC, datetime, timedelta
from decimal import Decimal
from pathlib import Path

import pytest

from tb.config.loader import load_hard_limits
from tb.ledger.store import Ledger
from tb.portfolio.pnl import ROLLING_SESSIONS, EquityCurve, PnlError
from tb.risk.rules.loss import DailyLossRule, DrawdownRule, RollingLossRule
from tb.risk.state import AccountState, OrderRequest, RiskContext, Verdict

# A Wednesday, mid-session. Sessions run Mon-Fri, so stepping back by trading
# days from here crosses a weekend — which is the case a naive
# five-calendar-days window gets wrong.
AS_OF = datetime(2026, 4, 1, 15, 30, tzinfo=UTC)
LIMITS = load_hard_limits(None).limits
TICKER = "AAPL_US_EQ"
UID = "isin:US0378331005"


@pytest.fixture
def ledger(tmp_path: Path) -> Iterator[Ledger]:
    with Ledger(tmp_path / "ledger.db") as opened:
        opened.initialise(created_by="test")
        yield opened


@pytest.fixture
def curve(ledger: Ledger) -> EquityCurve:
    return EquityCurve(ledger, run_id="run_test")


# --------------------------------------------------------------------------
# The cold start is knowable, not unknown
# --------------------------------------------------------------------------


def test_no_marks_at_all_reads_as_unmeasured(curve: EquityCurve) -> None:
    """`None`, not zero — and the rules block on it.

    The distinction this module exists for. A flat day and an unmeasured day
    look identical if both are `0.0`, and only one of them is a reason to
    keep trading.
    """
    reading = curve.read(at=AS_OF)
    assert reading.equity_ccy is None
    assert reading.day_pnl_pct is None
    assert reading.rolling_pnl_pct is None
    assert reading.drawdown_from_peak_pct is None
    assert not reading.measurable
    assert "unmeasured account is not a flat one" in reading.caveat()


def test_the_first_mark_of_all_time_is_a_flat_day(curve: EquityCurve) -> None:
    """0%, and that is *true* rather than a placeholder.

    The day has not moved yet, the peak is the current value, and the bot has
    existed for one session. All three are computable from one observation,
    which is why the cold start here needs no special case.
    """
    curve.mark(equity_ccy=Decimal("10000"), at=AS_OF)
    reading = curve.read(at=AS_OF)

    assert reading.equity_ccy == Decimal("10000")
    assert reading.day_pnl_pct == 0.0
    assert reading.rolling_pnl_pct == 0.0
    assert reading.drawdown_from_peak_pct == 0.0
    assert reading.measurable
    assert reading.sessions_observed == 1


def test_a_one_day_old_bot_reports_a_one_session_rolling_window(
    curve: EquityCurve,
) -> None:
    """Honest rather than silent.

    The number is the true P&L over the bot's whole life, which is the right
    input for a breaker asking "how much has been lost". It is the wrong thing
    to quote as a five-day result, so the reading says how many sessions it
    actually spans and the caveat says why that matters.
    """
    curve.mark(equity_ccy=Decimal("10000"), at=AS_OF)
    curve.mark(equity_ccy=Decimal("9900"), at=AS_OF + timedelta(minutes=5))
    reading = curve.read(at=AS_OF + timedelta(minutes=5))

    assert reading.sessions_observed == 1
    assert not reading.rolling_window_is_full
    assert reading.rolling_pnl_pct == pytest.approx(-1.0)
    assert f"not {ROLLING_SESSIONS}" in reading.caveat()


def test_a_non_positive_equity_is_refused_rather_than_stored(
    curve: EquityCurve,
) -> None:
    """Every percentage divides by it.

    And a long-only unlevered account cannot have non-positive equity, so
    this is a misparsed response rather than a bad day — storing it would
    poison every subsequent reading.
    """
    with pytest.raises(PnlError, match="misparsed response"):
        curve.mark(equity_ccy=Decimal("0"), at=AS_OF)
    with pytest.raises(PnlError, match="misparsed response"):
        curve.mark(equity_ccy=Decimal("-100"), at=AS_OF)


# --------------------------------------------------------------------------
# The arithmetic
# --------------------------------------------------------------------------


def test_the_day_is_measured_from_this_session_s_first_mark(
    curve: EquityCurve,
) -> None:
    """**Not from yesterday's close.**

    Those differ by the overnight gap, and attributing an overnight move to
    today would fire the daily breaker on exposure the bot could not have
    avoided — exposure the unprotected-window sizing has already budgeted
    for. Double-counting it here would halt on a risk that was accounted.
    """
    yesterday = AS_OF - timedelta(days=1)
    curve.mark(equity_ccy=Decimal("10000"), at=yesterday)
    # Gaps down 5% overnight, then flat through the session.
    curve.mark(equity_ccy=Decimal("9500"), at=AS_OF)
    curve.mark(equity_ccy=Decimal("9500"), at=AS_OF + timedelta(hours=1))

    reading = curve.read(at=AS_OF + timedelta(hours=1))
    assert reading.day_pnl_pct == 0.0, (
        "the overnight gap was attributed to today; the daily breaker would fire on a "
        "move that happened while the bot was not trading"
    )
    # The rolling window and the drawdown *do* see it, which is correct —
    # those are the breakers that should notice an overnight loss.
    assert reading.rolling_pnl_pct == pytest.approx(-5.0)
    assert reading.drawdown_from_peak_pct == pytest.approx(5.0)


def test_the_rolling_window_counts_trading_sessions_not_calendar_days(
    curve: EquityCurve,
) -> None:
    """Five calendar days back from a Wednesday is the previous Friday.

    That spans three sessions, not five — so a date-arithmetic window would
    silently be shorter than it claims over every weekend, and the breaker
    would be measuring a different period than its name says.
    """
    from tb.data.calendar import TradingCalendar

    sessions = TradingCalendar().sessions_between(AS_OF.date() - timedelta(days=20), AS_OF.date())
    window = [s.day for s in sessions][-ROLLING_SESSIONS:]
    assert len(window) == ROLLING_SESSIONS

    for index, day in enumerate(window):
        curve.mark(
            equity_ccy=Decimal("10000") - Decimal(index * 100),
            at=datetime(day.year, day.month, day.day, 15, 30, tzinfo=UTC),
        )

    reading = curve.read(at=AS_OF)
    assert reading.sessions_observed == ROLLING_SESSIONS
    assert reading.rolling_window_is_full
    # 10000 -> 9600 over the window.
    assert reading.rolling_pnl_pct == pytest.approx(-4.0)


def test_the_drawdown_peak_never_ages_out(curve: EquityCurve) -> None:
    """Over every mark ever taken, not over a window.

    A drawdown measured from a rolling peak resets as the peak ages out,
    which would let a slow bleed never register — and the slow bleed is
    exactly what this breaker exists to catch.
    """
    curve.mark(equity_ccy=Decimal("12000"), at=AS_OF - timedelta(days=90))
    for day in range(30):
        curve.mark(
            equity_ccy=Decimal("11000") - Decimal(day * 10),
            at=AS_OF - timedelta(days=30 - day),
        )
    curve.mark(equity_ccy=Decimal("10700"), at=AS_OF)

    reading = curve.read(at=AS_OF)
    assert reading.peak_equity_ccy == Decimal("12000"), (
        "the peak aged out, so a long slow decline would report no drawdown"
    )
    # (12000 - 10700) / 12000 = 10.83%
    assert reading.drawdown_from_peak_pct == pytest.approx(10.833, abs=0.01)


def test_a_new_high_resets_the_drawdown_to_zero(curve: EquityCurve) -> None:
    curve.mark(equity_ccy=Decimal("10000"), at=AS_OF - timedelta(days=10))
    curve.mark(equity_ccy=Decimal("9000"), at=AS_OF - timedelta(days=5))
    curve.mark(equity_ccy=Decimal("11000"), at=AS_OF)
    assert curve.read(at=AS_OF).drawdown_from_peak_pct == 0.0


def test_the_reading_is_point_in_time(curve: EquityCurve) -> None:
    """Only marks at or before `at`.

    The same as-of discipline the data layer enforces, applied to the
    account: a replay asking about a past instant must get the numbers that
    were available then, not today's.
    """
    curve.mark(equity_ccy=Decimal("10000"), at=AS_OF)
    curve.mark(equity_ccy=Decimal("5000"), at=AS_OF + timedelta(days=1))

    earlier = curve.read(at=AS_OF)
    assert earlier.equity_ccy == Decimal("10000")
    assert earlier.drawdown_from_peak_pct == 0.0, "a future collapse leaked backwards"

    later = curve.read(at=AS_OF + timedelta(days=1))
    assert later.equity_ccy == Decimal("5000")
    assert later.drawdown_from_peak_pct == pytest.approx(50.0)


def test_an_inferred_fill_is_excluded_from_realised_pnl(ledger: Ledger, curve: EquityCurve) -> None:
    """A guessed price must not enter the series the allocator learns from.

    Equity is unaffected — it comes from the broker — but per-trade
    attribution over the window is incomplete, and the reading says so rather
    than reporting a smaller number as if it were whole.
    """
    curve.mark(equity_ccy=Decimal("10000"), at=AS_OF)
    for index, (source, admissible) in enumerate(
        (("api_history", 1), ("inferred_from_position_delta", 0))
    ):
        ledger.conn.execute(
            "INSERT INTO fills (fill_id, t212_ticker, side, quantity, price, filled_at,"
            " source, confidence, admissible_for_pnl, recorded_at, recording_event_seq)"
            " VALUES (?,?,?,?,?,?,?,?,?,?,?)",
            (
                f"fill_{index}",
                TICKER,
                "sell",
                "1",
                "100",
                AS_OF.isoformat(),
                source,
                "observed" if admissible else "inferred",
                admissible,
                AS_OF.isoformat(),
                1,
            ),
        )
    ledger.conn.commit()

    reading = curve.read(at=AS_OF)
    assert reading.n_inadmissible_fills == 1
    assert reading.realised_pnl_ccy == Decimal("100"), "only the admissible sell counted"
    # Both caveats apply here — a one-session window and an excluded fill.
    # `caveats()` returns all of them, because reporting only the first is how
    # the second gets silently dropped just as someone decides to trust the
    # number.
    assert any("inferred prices" in c for c in reading.caveats())
    assert any("session(s)" in c for c in reading.caveats())
    assert len(reading.caveats()) == 2


# --------------------------------------------------------------------------
# The breakers, firing
# --------------------------------------------------------------------------


def _context(reading_account: AccountState) -> RiskContext:
    from tb.broker.port import OrderPurpose, Side
    from tb.strategy.base import Action

    return RiskContext(
        as_of=AS_OF,
        limits=LIMITS,
        request=OrderRequest(
            t212_ticker=TICKER,
            instrument_uid=UID,
            side=Side.BUY,
            purpose=OrderPurpose.ENTRY,
            action=Action.ENTER,
            reference_price=Decimal("100"),
            expected_edge_bps=Decimal("250"),
        ),
        account=reading_account,
        may_enter=True,
        regime_exposure_factor=Decimal(1),
        regime_state="risk_on",
        bar_age_seconds=5.0,
        bar_period_seconds=86400,
        minutes_since_open=60,
        minutes_until_close=120,
    )


def _account_from(curve: EquityCurve, *, at: datetime) -> AccountState:
    """The account the loop would build, for the same instant."""
    reading = curve.read(at=at)
    return AccountState(
        equity_ccy=reading.equity_ccy,
        free_cash_ccy=reading.equity_ccy,
        deployed_ccy=Decimal(0),
        n_open_positions=0,
        currency="GBP",
        day_pnl_pct=reading.day_pnl_pct,
        rolling_5d_pnl_pct=reading.rolling_pnl_pct,
        drawdown_from_peak_pct=reading.drawdown_from_peak_pct,
    )


def test_the_daily_breaker_blocks_an_entry_once_the_day_is_down(
    curve: EquityCurve,
) -> None:
    """**The gap this closes.**

    Before the equity curve existed the loop passed `day_pnl_pct=0.0`, so
    this rule returned PASS at every loss the account could suffer. The
    assertion is the *change* in verdict across the threshold, not merely that
    a contrived number blocks.
    """
    limit = LIMITS.loss.daily_halt_pct  # 2.0

    curve.mark(equity_ccy=Decimal("10000"), at=AS_OF)
    healthy = DailyLossRule().evaluate(_context(_account_from(curve, at=AS_OF)))
    assert healthy.verdict is Verdict.PASS
    assert not healthy.blocks

    # Down past the threshold within the same session.
    down = AS_OF + timedelta(hours=1)
    curve.mark(equity_ccy=Decimal("9700"), at=down)  # -3%
    breached = DailyLossRule().evaluate(_context(_account_from(curve, at=down)))

    assert breached.verdict is Verdict.BLOCK
    assert breached.blocks, "the daily breaker did not block a 3% loss"
    assert breached.observed_value == pytest.approx(3.0)
    assert breached.limit_value == limit


def test_the_daily_breaker_warns_inside_the_last_fifth_of_the_budget(
    curve: EquityCurve,
) -> None:
    """The run that ends at -1.9% against a 2% limit is worth knowing about."""
    curve.mark(equity_ccy=Decimal("10000"), at=AS_OF)
    warn_at = AS_OF + timedelta(hours=1)
    curve.mark(equity_ccy=Decimal("9820"), at=warn_at)  # -1.8%, inside 80% of 2%

    verdict = DailyLossRule().evaluate(_context(_account_from(curve, at=warn_at)))
    assert verdict.verdict is Verdict.WARN
    assert not verdict.blocks, "a warning must not block"


def test_the_rolling_breaker_catches_a_bleed_no_single_day_would_trip(
    curve: EquityCurve,
) -> None:
    """The reason there are three thresholds rather than one larger one.

    1.2% a day never trips a 2% daily limit and is down 6% in a week.
    """
    from tb.data.calendar import TradingCalendar

    sessions = [
        s.day
        for s in TradingCalendar().sessions_between(AS_OF.date() - timedelta(days=20), AS_OF.date())
    ][-ROLLING_SESSIONS:]

    equity = Decimal("10000")
    for day in sessions:
        moment = datetime(day.year, day.month, day.day, 15, 30, tzinfo=UTC)
        # Opening mark for the session, then a 1.2% fall within it.
        curve.mark(equity_ccy=equity, at=moment)
        equity = (equity * Decimal("0.988")).quantize(Decimal("0.01"))
        curve.mark(equity_ccy=equity, at=moment + timedelta(hours=1))

    at = datetime(sessions[-1].year, sessions[-1].month, sessions[-1].day, 16, 30, tzinfo=UTC)
    account = _account_from(curve, at=at)

    daily = DailyLossRule().evaluate(_context(account))
    rolling = RollingLossRule().evaluate(_context(account))

    assert not daily.blocks, (
        f"the daily breaker fired at {account.day_pnl_pct:.2f}%, so this test is not "
        "demonstrating what the rolling breaker adds"
    )
    assert rolling.blocks, (
        f"the rolling breaker missed a bleed to {account.rolling_5d_pnl_pct:.2f}% "
        f"against a {LIMITS.loss.rolling_5d_halt_pct}% limit"
    )


def test_the_drawdown_breaker_blocks_past_its_threshold(curve: EquityCurve) -> None:
    """The outermost breaker: flatten and halt."""
    curve.mark(equity_ccy=Decimal("10000"), at=AS_OF - timedelta(days=30))
    at = AS_OF
    curve.mark(equity_ccy=Decimal("9200"), at=at)  # 8% below peak, past 6%

    verdict = DrawdownRule().evaluate(_context(_account_from(curve, at=at)))
    assert verdict.blocks
    assert verdict.observed_value == pytest.approx(8.0)
    assert verdict.limit_value == LIMITS.loss.max_drawdown_flatten_pct


def test_an_unmeasured_account_blocks_every_breaker(curve: EquityCurve) -> None:
    """`None` is not zero, and all three fail closed on it.

    The half that makes the rest safe: if the equity curve is empty — a
    broker that reported nothing, a first cycle that could not read cash —
    the breakers must refuse rather than read a flat day.
    """
    account = _account_from(curve, at=AS_OF)
    assert account.day_pnl_pct is None

    for rule in (DailyLossRule(), RollingLossRule(), DrawdownRule()):
        verdict = rule.evaluate(_context(account))
        assert verdict.blocks, f"{rule.name} passed on an unmeasured account"
        assert verdict.observed_value == "unknown"


def test_every_breaker_still_permits_an_exit_when_breached(curve: EquityCurve) -> None:
    """**The asymmetry, at the worst moment.**

    A breaker that blocked the flattening orders it fired to trigger would
    lock in exactly the loss it existed to limit. Asserted at a loss past all
    three thresholds at once, which is when it matters.
    """
    from tb.broker.port import OrderPurpose, Side
    from tb.strategy.base import Action

    curve.mark(equity_ccy=Decimal("10000"), at=AS_OF - timedelta(days=30))
    at = AS_OF
    curve.mark(equity_ccy=Decimal("10000"), at=at - timedelta(hours=2))
    curve.mark(equity_ccy=Decimal("8000"), at=at)  # -20%: past all three

    account = _account_from(curve, at=at)
    ctx = _context(account)
    exit_ctx = RiskContext(
        as_of=ctx.as_of,
        limits=ctx.limits,
        request=OrderRequest(
            t212_ticker=TICKER,
            instrument_uid=UID,
            side=Side.SELL,
            purpose=OrderPurpose.EXIT,
            action=Action.EXIT,
            reference_price=Decimal("100"),
            quantity=Decimal("1"),
        ),
        account=account,
        position_quantity=Decimal("1"),
        position_entry_at=at - timedelta(days=10),
        may_exit=True,
        bar_age_seconds=5.0,
        bar_period_seconds=86400,
    )

    # Every breaker blocks the entry...
    for rule in (DailyLossRule(), RollingLossRule(), DrawdownRule()):
        assert rule.evaluate(ctx).blocks, f"{rule.name} permitted an entry at -20%"

    # ...and none blocks the exit.
    for rule in (DailyLossRule(), RollingLossRule(), DrawdownRule()):
        verdict = rule.evaluate(exit_ctx)
        assert not verdict.blocks, (
            f"{rule.name} blocked an EXIT at -20%. A breaker that refuses the orders it "
            "fired to place locks in the loss it existed to limit."
        )


# --------------------------------------------------------------------------
# Through the loop
# --------------------------------------------------------------------------


def test_the_loop_marks_equity_before_it_decides(ledger: Ledger) -> None:
    """Ordering, asserted at the table.

    On the first cycle of a session the mark is what establishes the day's
    opening equity. A cycle that decided first would compute today's P&L
    against yesterday's close and attribute the overnight gap to today.
    """
    rows = ledger.conn.execute("SELECT COUNT(*) AS n FROM equity_marks").fetchone()
    assert rows["n"] == 0

    curve = EquityCurve(ledger, run_id="run_test")
    curve.mark(equity_ccy=Decimal("10000"), at=AS_OF)

    stored = ledger.conn.execute("SELECT session_date, equity, source FROM equity_marks").fetchone()
    assert stored["session_date"] == AS_OF.date().isoformat()
    assert stored["equity"] == "10000"
    assert stored["source"] == "broker"


def test_marks_are_not_events(ledger: Ledger) -> None:
    """One mark per cycle for a year is half a million rows.

    The hash chain is for facts that must be tamper-evident; an equity mark is
    a measurement that can be re-taken from the broker. Inflating the chain
    with them would make the events that do matter harder to read.
    """
    before = ledger.conn.execute("SELECT COUNT(*) AS n FROM event_log").fetchone()["n"]
    curve = EquityCurve(ledger, run_id="run_test")
    for minute in range(50):
        curve.mark(equity_ccy=Decimal("10000"), at=AS_OF + timedelta(minutes=minute))
    after = ledger.conn.execute("SELECT COUNT(*) AS n FROM event_log").fetchone()["n"]

    assert after == before, "fifty equity marks appended fifty events to the chain"
    assert len(curve.marks(limit=100)) == 50


# --------------------------------------------------------------------------
# One account per curve
# --------------------------------------------------------------------------


def _marked(ledger: Ledger, run_id: str, mode: str, *points: tuple[datetime, str]) -> EquityCurve:
    """A run of `mode`, recorded as `tb run` records it, and its equity marks."""
    ledger.record_run_start(run_id=run_id, mode=mode)
    curve = EquityCurve(ledger, run_id=run_id)
    for at, equity in points:
        curve.mark(equity_ccy=Decimal(equity), at=at, currency="GBP")
    return curve


def test_a_demo_run_is_not_measured_against_a_paper_account(ledger: Ledger) -> None:
    """A paper peak of 10,000 is not a demo drawdown of half: different accounts.

    Read as one curve, the demo run's first mark is a 50% drawdown and the
    drawdown breaker flattens and halts a run that has lost nothing.
    """
    _marked(ledger, "run_paper", "paper", (AS_OF - timedelta(days=2), "10000.00"))
    demo = _marked(ledger, "run_demo", "demo", (AS_OF, "5000.00"))

    reading = demo.read(at=AS_OF)
    assert reading.equity_ccy == Decimal("5000.00")
    assert reading.drawdown_from_peak_pct == 0.0
    assert reading.n_marks == 1


def test_demo_runs_share_one_account_and_its_peak(ledger: Ledger) -> None:
    """A restart is the same demo account: its drawdown must survive the restart."""
    _marked(ledger, "run_demo_1", "demo", (AS_OF - timedelta(days=1), "10000.00"))
    second = _marked(ledger, "run_demo_2", "demo", (AS_OF, "9500.00"))

    reading = second.read(at=AS_OF)
    assert reading.peak_equity_ccy == Decimal("10000.00")
    assert reading.drawdown_from_peak_pct == pytest.approx(5.0)


def test_each_paper_run_is_its_own_account(ledger: Ledger) -> None:
    """Every paper run starts a fresh simulated account at --equity."""
    _marked(ledger, "run_paper_1", "paper", (AS_OF - timedelta(days=1), "11000.00"))
    fresh = _marked(ledger, "run_paper_2", "paper", (AS_OF, "10000.00"))
    assert fresh.read(at=AS_OF).drawdown_from_peak_pct == 0.0


def test_a_live_run_is_not_measured_against_demo(ledger: Ledger) -> None:
    """Demo's larger balance would otherwise hide a real loss, or invent one."""
    _marked(ledger, "run_demo", "demo", (AS_OF - timedelta(days=1), "50000.00"))
    live = _marked(
        ledger,
        "run_live",
        "live",
        (AS_OF - timedelta(hours=2), "500.00"),
        (AS_OF, "480.00"),
    )
    reading = live.read(at=AS_OF)
    assert reading.peak_equity_ccy == Decimal("500.00")
    assert reading.drawdown_from_peak_pct == pytest.approx(4.0)


def test_a_run_with_no_recorded_mode_reads_every_mark(ledger: Ledger) -> None:
    """A ledger from before runs were recorded keeps the curve it had."""
    EquityCurve(ledger, run_id="run_old").mark(equity_ccy=Decimal("100.00"), at=AS_OF)
    unrecorded = EquityCurve(ledger, run_id="run_older")
    assert unrecorded.read(at=AS_OF).equity_ccy == Decimal("100.00")
