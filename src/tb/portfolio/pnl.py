"""The equity curve, and the three numbers the loss breakers divide by.

Until this module existed the breakers were structurally wired and could not
fire: the loop passed `day_pnl_pct=0.0` and friends, so `DailyLossRule` read a
flat day forever. Three rules, present in every verdict row, unable to block
anything. That is worse than not having them, because a ledger full of
`daily_loss: pass` reads as evidence the breaker was watching.

**Equity comes from the broker, not from the fills.** Reconstructing equity by
summing fill P&L would be a second opinion about the account balance, and the
first opinion — what Trading 212 says `total` is — is the one that matters:
it includes fees we did not model, FX we did not predict, and dividends we did
not place. A reconstruction that disagreed with the broker would leave two
numbers and no way to choose, so the broker's is definitive and the fills are
used only for attribution.

What the fills *are* used for is `admissible_for_pnl`. A fill whose price was
inferred from a position delta is a guess, and the realised series the
allocator learns from must not contain guesses — so `realised_pnl` reports
the inadmissible count alongside the number rather than silently omitting
those trades.

**The cold start is knowable, not unknown.** Every one of the three is
computable from a single equity mark:

* `day_pnl_pct` — against the first mark of the current session. On the first
  cycle that is the current mark, so the answer is 0%, which is *true*: the
  day has not moved yet.
* `drawdown_from_peak_pct` — against the highest mark seen. On the first mark
  the peak is the current value, so 0%.
* `rolling_5d_pnl_pct` — against the first mark within five sessions. With one
  day of history this is the one-day figure, which is not an approximation: a
  bot that has existed for one day genuinely has a one-day P&L, and the
  breaker is asking how much has been lost rather than how long ago.

So `None` means one thing only: **no equity observation exists at all.** Every
rule reads that as a block, which is the correct fail-closed reading — an
unmeasured account is not a flat one.

The rolling figure carries a caveat rather than a silence. `PnlReading.
sessions_observed` says how many sessions the window actually spans, so a
promotion decision in M5 can tell "down 1% over five sessions" from "down 1%
over the only session there has been".
"""

from __future__ import annotations

import sqlite3
from dataclasses import dataclass
from datetime import date, datetime, timedelta
from decimal import Decimal

from tb.core.clock import from_iso, now_utc, to_iso
from tb.core.errors import TbError
from tb.core.ids import new_id
from tb.data.calendar import TradingCalendar
from tb.ledger.store import Ledger

# How many sessions the rolling breaker looks back over. Five, matching
# `loss.rolling_5d_halt_pct`, and named here so the query and the limit cannot
# drift apart.
ROLLING_SESSIONS = 5


class PnlError(TbError):
    """An equity figure could not be established."""


@dataclass(frozen=True, slots=True)
class EquityMark:
    """One observation of account equity, at one instant."""

    mark_id: str
    at: datetime
    equity_ccy: Decimal
    currency: str | None = None
    deployed_ccy: Decimal | None = None
    free_cash_ccy: Decimal | None = None
    run_id: str | None = None
    source: str = "broker"


@dataclass(frozen=True, slots=True)
class PnlReading:
    """The three breaker inputs, and how much history is behind them.

    Every percentage is signed the way a P&L is: negative is a loss. The rules
    negate it themselves, because a threshold expressed as a positive
    percentage and a P&L expressed as a signed one is the clearer pair — the
    alternative is a config file full of negative numbers that read as
    typos.
    """

    equity_ccy: Decimal | None
    day_pnl_pct: float | None
    rolling_pnl_pct: float | None
    drawdown_from_peak_pct: float | None
    peak_equity_ccy: Decimal | None = None
    day_open_equity_ccy: Decimal | None = None
    # How many distinct sessions the rolling window actually spans. 1 on the
    # first day, which is honest rather than a defect — but a caller reporting
    # "five-day P&L" needs to be able to say so.
    sessions_observed: int = 0
    n_marks: int = 0
    # Realised P&L from admissible fills, for attribution rather than for the
    # breakers. The breakers read equity, which already contains this.
    realised_pnl_ccy: Decimal | None = None
    n_inadmissible_fills: int = 0

    @property
    def measurable(self) -> bool:
        """Whether the breakers can be evaluated at all."""
        return self.equity_ccy is not None and self.day_pnl_pct is not None

    @property
    def rolling_window_is_full(self) -> bool:
        return self.sessions_observed >= ROLLING_SESSIONS

    def caveats(self) -> tuple[str, ...]:
        """Everything a reader should know before quoting these numbers.

        All of them, not the first. Two can apply at once — a short rolling
        window *and* inadmissible fills — and returning only the first means
        the second is silently dropped at exactly the moment someone is
        deciding whether to trust the number.
        """
        out: list[str] = []
        if not self.measurable:
            # The only one that stands alone: with no equity observation the
            # others have nothing to qualify.
            return (
                "no equity observation exists, so no loss breaker can be evaluated. "
                "Every one of them blocks, which is the fail-closed reading: an "
                "unmeasured account is not a flat one.",
            )
        if not self.rolling_window_is_full:
            out.append(
                f"the rolling figure spans {self.sessions_observed} session(s), not "
                f"{ROLLING_SESSIONS}. It is the true P&L over the bot's whole life, which "
                "is the right number for the breaker and the wrong one to quote as a "
                "five-day result."
            )
        if self.n_inadmissible_fills:
            out.append(
                f"{self.n_inadmissible_fills} fill(s) have inferred prices and are "
                "excluded from realised P&L. Equity is unaffected — it comes from the "
                "broker — but per-trade attribution over this window is incomplete."
            )
        return tuple(out)

    def caveat(self) -> str:
        """The caveats as one line, for a log or a CLI table."""
        return " ".join(self.caveats())


class EquityCurve:
    """Records equity marks and derives the breaker inputs from them.

    A projection like every other table here: the marks are appended and the
    percentages are computed on read. Storing the percentages would mean a
    cached number that drifts from the marks it summarises, and a drifting
    loss breaker is worse than none.

    **One account per curve.** A ledger carries marks from paper, demo and
    live runs, and those are different accounts: a demo run read against a
    paper run's peak sees a drawdown that never happened, and is flattened for
    it; a live run read against demo's larger balance hides one that did. So a
    demo or live run reads the marks of every run of its own mode — one
    persistent account each — and a paper run reads only its own, because
    every paper run starts a fresh simulated account. A run with no recorded
    mode (a ledger from before runs were recorded, or a test) reads every
    mark, as before.
    """

    def __init__(
        self,
        ledger: Ledger,
        *,
        run_id: str | None = None,
        calendar: TradingCalendar | None = None,
    ) -> None:
        self._ledger = ledger
        self._run_id = run_id
        self._calendar = calendar or TradingCalendar()

    # -- recording ---------------------------------------------------------

    def mark(
        self,
        *,
        equity_ccy: Decimal,
        at: datetime | None = None,
        currency: str | None = None,
        deployed_ccy: Decimal | None = None,
        free_cash_ccy: Decimal | None = None,
        source: str = "broker",
    ) -> EquityMark:
        """Record one equity observation.

        Called once per cycle by the loop, before any decision. Deliberately
        *not* an event append: a mark every minute for a year is half a
        million rows, and the ledger's chain is for facts that need to be
        tamper-evident. An equity mark is a measurement we can re-take from
        the broker, and inflating the chain with them would make the events
        that do matter harder to read.

        Refuses a non-positive equity rather than storing it. A zero would
        make every percentage a division by zero, and a negative equity in a
        long-only unlevered account means the response was misparsed.
        """
        if equity_ccy <= 0:
            raise PnlError(
                f"equity of {equity_ccy} is not a usable mark. Every loss percentage "
                "divides by it, and a long-only unlevered account cannot have "
                "non-positive equity — so this is a misparsed response, not a poor day."
            )
        moment = at or now_utc()
        mark = EquityMark(
            mark_id=new_id("eqm"),
            at=moment,
            equity_ccy=equity_ccy,
            currency=currency,
            deployed_ccy=deployed_ccy,
            free_cash_ccy=free_cash_ccy,
            run_id=self._run_id,
            source=source,
        )
        self._ledger.conn.execute(
            "INSERT INTO equity_marks (mark_id, run_id, at_utc, session_date, equity,"
            " currency, deployed, free_cash, source) VALUES (?,?,?,?,?,?,?,?,?)",
            (
                mark.mark_id,
                mark.run_id,
                to_iso(moment),
                self._session_date(moment).isoformat(),
                str(equity_ccy),
                currency,
                None if deployed_ccy is None else str(deployed_ccy),
                None if free_cash_ccy is None else str(free_cash_ccy),
                source,
            ),
        )
        self._ledger.conn.commit()
        return mark

    # -- reading -----------------------------------------------------------

    def _scope(self) -> tuple[str, tuple[str, ...]]:
        """The SQL condition that keeps this curve to one account.

        Resolved on each read rather than once: the run's mode is recorded
        when it starts, which may be after the curve was built.
        """
        if self._run_id is None:
            return "1 = 1", ()
        row = self._ledger.conn.execute(
            "SELECT mode FROM runs WHERE run_id = ?", (self._run_id,)
        ).fetchone()
        mode = None if row is None else str(row["mode"])
        if mode == "paper":
            return "run_id = ?", (self._run_id,)
        if mode in ("demo", "live"):
            return "run_id IN (SELECT run_id FROM runs WHERE mode = ?)", (mode,)
        return "1 = 1", ()

    def read(self, *, at: datetime | None = None) -> PnlReading:
        """The three breaker inputs as of `at`.

        Reads only marks at or before `at`, so a backtest or a replay asking
        about a past instant gets the numbers that were available then rather
        than today's. That is the same as-of discipline the data layer
        enforces, applied to the account.
        """
        moment = at or now_utc()
        latest = self._latest(moment)
        if latest is None:
            return PnlReading(
                equity_ccy=None,
                day_pnl_pct=None,
                rolling_pnl_pct=None,
                drawdown_from_peak_pct=None,
            )

        session = self._session_date(moment)
        day_open = self._first_of_session(session, moment)
        peak = self._peak(moment)
        rolling_open, sessions = self._rolling_open(session, moment)
        realised, inadmissible = self._realised(moment)

        return PnlReading(
            equity_ccy=latest,
            day_pnl_pct=_pct_change(day_open, latest),
            rolling_pnl_pct=_pct_change(rolling_open, latest),
            # Drawdown is expressed as a positive percentage *below* the peak,
            # unlike the other two — matching `max_drawdown_flatten_pct`,
            # which is also positive. A signed drawdown would be negative for
            # a loss and zero at the peak, which reads as the same thing.
            drawdown_from_peak_pct=_drawdown_pct(peak, latest),
            peak_equity_ccy=peak,
            day_open_equity_ccy=day_open,
            sessions_observed=sessions,
            n_marks=self._count(moment),
            realised_pnl_ccy=realised,
            n_inadmissible_fills=inadmissible,
        )

    def marks(self, *, limit: int = 100) -> tuple[EquityMark, ...]:
        scope, params = self._scope()
        rows = self._ledger.conn.execute(
            f"SELECT * FROM equity_marks WHERE {scope} ORDER BY at_utc DESC LIMIT ?",  # noqa: S608
            (*params, limit),
        ).fetchall()
        return tuple(_mark_from_row(row) for row in reversed(rows))

    # -- internals ---------------------------------------------------------

    def _latest(self, moment: datetime) -> Decimal | None:
        scope, params = self._scope()
        row = self._ledger.conn.execute(
            f"SELECT equity FROM equity_marks WHERE {scope} AND at_utc <= ?"  # noqa: S608
            " ORDER BY at_utc DESC LIMIT 1",
            (*params, to_iso(moment)),
        ).fetchone()
        return None if row is None else Decimal(str(row["equity"]))

    def _first_of_session(self, session: date, moment: datetime) -> Decimal | None:
        """The day's opening equity.

        The *first mark of this session*, not the last mark of the previous
        one. Those differ by the overnight gap, and attributing an overnight
        move to today would make the daily breaker fire on a gap the bot could
        not have avoided — the unprotected-window sizing already accounts for
        that exposure, and double-counting it here would halt on a risk that
        was already budgeted.
        """
        scope, params = self._scope()
        row = self._ledger.conn.execute(
            f"SELECT equity FROM equity_marks WHERE {scope} AND session_date = ?"  # noqa: S608
            " AND at_utc <= ? ORDER BY at_utc ASC LIMIT 1",
            (*params, session.isoformat(), to_iso(moment)),
        ).fetchone()
        return None if row is None else Decimal(str(row["equity"]))

    def _peak(self, moment: datetime) -> Decimal | None:
        """The high-water mark.

        Over every mark ever taken, not over a window. A drawdown measured
        from a rolling peak resets itself as the peak ages out, which would
        let a slow bleed never register — and the slow bleed is exactly what
        this breaker exists to catch.
        """
        scope, params = self._scope()
        row = self._ledger.conn.execute(
            f"SELECT equity FROM equity_marks WHERE {scope} AND at_utc <= ?"  # noqa: S608
            " ORDER BY CAST(equity AS REAL) DESC LIMIT 1",
            (*params, to_iso(moment)),
        ).fetchone()
        return None if row is None else Decimal(str(row["equity"]))

    def _rolling_open(self, session: date, moment: datetime) -> tuple[Decimal | None, int]:
        """Equity at the start of the rolling window, and sessions spanned.

        The window is the last `ROLLING_SESSIONS` *trading* sessions, taken
        from the calendar rather than by subtracting days. Five calendar days
        back from a Monday is the previous Wednesday, which spans three
        sessions — so a date-arithmetic window would silently be shorter than
        it claims over every weekend.
        """
        sessions = self._calendar.sessions_between(
            session - timedelta(days=ROLLING_SESSIONS * 3), session
        )
        if not sessions:
            return self._first_of_session(session, moment), 1
        window = sessions[-ROLLING_SESSIONS:]
        earliest = window[0].day

        scope, params = self._scope()
        row = self._ledger.conn.execute(
            f"SELECT equity FROM equity_marks WHERE {scope} AND session_date >= ?"  # noqa: S608
            " AND at_utc <= ? ORDER BY at_utc ASC LIMIT 1",
            (*params, earliest.isoformat(), to_iso(moment)),
        ).fetchone()
        observed = self._ledger.conn.execute(
            "SELECT COUNT(DISTINCT session_date) AS n FROM equity_marks"  # noqa: S608
            f" WHERE {scope} AND session_date >= ? AND at_utc <= ?",
            (*params, earliest.isoformat(), to_iso(moment)),
        ).fetchone()
        return (
            None if row is None else Decimal(str(row["equity"])),
            int(observed["n"] or 0),
        )

    def _count(self, moment: datetime) -> int:
        scope, params = self._scope()
        row = self._ledger.conn.execute(
            f"SELECT COUNT(*) AS n FROM equity_marks WHERE {scope} AND at_utc <= ?",  # noqa: S608
            (*params, to_iso(moment)),
        ).fetchone()
        return int(row["n"] or 0)

    def _realised(self, moment: datetime) -> tuple[Decimal | None, int]:
        """Realised P&L from admissible fills, and how many were excluded.

        Attribution only — the breakers read equity, which already contains
        this and more. Returned so a caller can say how much of the account's
        movement it can actually explain, which is a different and useful
        question.
        """
        scope, params = self._scope()
        if params:
            # Fills carry no run of their own; they belong to one through the
            # intent that placed them. A fill with no intent — history for an
            # order this bot never placed — belongs to no account's curve.
            # The condition is one of `_scope`'s fixed strings; values are bound.
            scope = f"intent_id IN (SELECT intent_id FROM order_intents WHERE {scope})"  # noqa: S608
        rows = self._ledger.conn.execute(
            "SELECT side, quantity, price, admissible_for_pnl FROM fills"  # noqa: S608
            f" WHERE {scope} AND (filled_at <= ? OR filled_at IS NULL)",
            (*params, to_iso(moment)),
        ).fetchall()
        if not rows:
            return None, 0

        total = Decimal(0)
        excluded = 0
        for row in rows:
            if not row["admissible_for_pnl"] or row["price"] is None:
                excluded += 1
                continue
            value = Decimal(str(row["quantity"])) * Decimal(str(row["price"]))
            # A sell brings cash in, a buy takes it out. Not a P&L on its own
            # — it is cash flow — which is why this is attribution and the
            # breakers use equity instead.
            total += value if row["side"] == "sell" else -value
        return total, excluded

    def _session_date(self, moment: datetime) -> date:
        """Which trading session a moment belongs to.

        The UTC date, which is correct for a US-session bot: the session opens
        at 13:30 UTC and closes at 20:00, so it never crosses midnight UTC. A
        bot trading Asian hours would need the exchange's own date here, and
        this is the line that would have to change.
        """
        return moment.astimezone(tz=moment.tzinfo).date() if moment.tzinfo else moment.date()


def _pct_change(opening: Decimal | None, current: Decimal) -> float | None:
    if opening is None or opening <= 0:
        return None
    return float((current - opening) / opening * Decimal(100))


def _drawdown_pct(peak: Decimal | None, current: Decimal) -> float | None:
    if peak is None or peak <= 0:
        return None
    if current >= peak:
        return 0.0
    return float((peak - current) / peak * Decimal(100))


def _mark_from_row(row: sqlite3.Row) -> EquityMark:
    get = row.__getitem__
    return EquityMark(
        mark_id=str(get("mark_id")),
        at=from_iso(str(get("at_utc"))),
        equity_ccy=Decimal(str(get("equity"))),
        currency=None if get("currency") is None else str(get("currency")),
        deployed_ccy=None if get("deployed") is None else Decimal(str(get("deployed"))),
        free_cash_ccy=None if get("free_cash") is None else Decimal(str(get("free_cash"))),
        run_id=None if get("run_id") is None else str(get("run_id")),
        source=str(get("source") or "broker"),
    )
