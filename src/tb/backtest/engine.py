"""The backtester. Its job is to be unable to flatter a strategy.

Built on `ForwardOnlyReader` from `tb.data.asof` rather than a cursor of its
own. That reader already refuses to rewind and holds no bars later than the
decision time, and reimplementing it here would produce a second answer to
"what could I see" — which is the class of divergence this whole layer exists
to prevent.

**Where backtests actually cheat: the fill price.** A signal computed from the
bar closing at time `t` cannot be filled at that same close. The close is the
last print *of* that bar; acting on it requires having seen it, and by then it
is gone. So an order decided on bar `t` fills at the **open of bar `t+1`**, and
the engine obtains that price by advancing the reader — it is never read from a
bar the decision could see. Filling at the decision bar's close is the single
most common way a backtest manufactures returns, and it is worth roughly the
entire gross edge at minute resolution.

That has a consequence the code makes explicit: a decision on the last bar of
the window has no next bar to fill against, so it is **dropped**, not filled at
the last close. Those dropped decisions are counted and reported.

**Costs are charged on the fill, from the same model the live path uses.** The
cost gate runs *before* the order exists, so a rejected trade never appears in
the equity curve — and the count of rejections is reported, because "this
strategy had 400 signals and 3 affordable trades" is the finding, not a
footnote.

**Gross and net are tracked in parallel.** Two equity curves, identical except
that one never pays. Their difference is the cost drag, measured rather than
argued, and it is what `tb backtest calibrate` compares a null strategy's net
Sharpe against.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from datetime import date, datetime
from decimal import ROUND_HALF_EVEN, Decimal, localcontext

from tb.backtest import metrics as metrics_mod
from tb.backtest.costs import CostModel, Jurisdiction, RoundTrip
from tb.backtest.metrics import CurvePoint, Metrics
from tb.core.errors import TbError
from tb.core.ids import new_id
from tb.data.adjustments import CorporateAction, holding_factor
from tb.data.asof import UNKNOWN, BarWindow, ForwardOnlyReader
from tb.data.provider import Bar, Resolution
from tb.features.pipeline import FeaturePipeline, FeatureSnapshot
from tb.strategy.base import Action, Decision, PositionState, Strategy

_CENTS = Decimal("0.01")
_WORKING_PRECISION = 40

_PERIODS_PER_YEAR = {
    Resolution.DAILY: metrics_mod.PERIODS_PER_YEAR_DAILY,
    Resolution.HOURLY: metrics_mod.PERIODS_PER_YEAR_HOURLY,
    Resolution.MINUTE: metrics_mod.PERIODS_PER_YEAR_MINUTE,
}


class BacktestError(TbError):
    """A backtest could not run, or could not be trusted to have run correctly."""


@dataclass(frozen=True, slots=True)
class InstrumentMeta:
    """What the cost model needs to know about an instrument.

    Required rather than defaulted. A missing jurisdiction defaults to the
    zero-tax case in every implementation that defaults it, which understates
    cost by up to 100bps per entry — so the engine demands it and
    `Jurisdiction.UNKNOWN` refuses to be costed at all.
    """

    instrument_uid: str
    currency: str
    jurisdiction: Jurisdiction


@dataclass(frozen=True, slots=True)
class Trade:
    """One completed round trip, with its costs itemised."""

    trade_seq: int
    instrument_uid: str
    entry_at: datetime
    exit_at: datetime
    entry_price: Decimal
    exit_price: Decimal
    quantity: Decimal
    gross_pnl_ccy: Decimal
    net_pnl_ccy: Decimal
    cost_total_ccy: Decimal
    cost_breakdown: Mapping[str, str]
    expected_edge_bps: Decimal
    expected_cost_bps: Decimal
    holding_minutes: int
    exit_reason: str


@dataclass(frozen=True, slots=True)
class BacktestResult:
    """Everything one backtest produced, plus what it refused to do.

    The refusal counts are not diagnostics. `n_rejected_by_cost_gate` of 400
    against `n_trades` of 3 *is* the result for a minute-resolution strategy on
    this venue, and a report that showed only the 3 would be describing a
    strategy that was never possible.
    """

    backtest_id: str
    strategy_id: str
    strategy_version: int
    resolution: Resolution
    rng_seed: int
    window_start: datetime | None
    window_end: datetime | None

    metrics: Metrics
    trades: tuple[Trade, ...]
    curve: tuple[CurvePoint, ...]

    n_decisions: int
    n_signals: int
    n_rejected_by_cost_gate: int
    n_unevaluable: int
    n_dropped_no_next_bar: int
    starting_equity_ccy: Decimal

    @property
    def caveats(self) -> tuple[str, ...]:
        """Anything a reader must know before believing the numbers."""
        notes: list[str] = []
        if self.n_rejected_by_cost_gate:
            notes.append(
                f"{self.n_rejected_by_cost_gate} of {self.n_signals} entry signals were "
                "refused by the cost gate: the fee schedule, not the signal, is what "
                "bounded this strategy"
            )
        if self.n_unevaluable:
            notes.append(
                f"{self.n_unevaluable} decisions were unevaluable (a feature had no "
                "value), which is distinct from the strategy declining to trade"
            )
        if self.n_dropped_no_next_bar:
            notes.append(
                f"{self.n_dropped_no_next_bar} decisions had no following bar to fill "
                "against and were dropped rather than filled at the decision bar's close"
            )
        if self.metrics.n_trades == 0:
            notes.append("no trades: every metric here is vacuous")
        return tuple(notes)


@dataclass(slots=True)
class _Position:
    """Open position bookkeeping, in account currency.

    `quantity` and `entry_price` are on the scale of `quoted_through`, the last
    session whose splits they reflect; `_follow_splits` moves all three
    together, so their product — the entry notional — never changes.
    """

    instrument_uid: str
    quantity: Decimal
    entry_price: Decimal
    entry_at: datetime
    entry_notional_ccy: Decimal
    entry_cost_ccy: Decimal
    entry_breakdown: Mapping[str, str]
    expected_edge_bps: Decimal
    expected_cost_bps: Decimal
    quoted_through: date

    def to_state(self) -> PositionState:
        return PositionState(
            instrument_uid=self.instrument_uid,
            quantity=self.quantity,
            entry_price=self.entry_price,
            entry_at=self.entry_at,
        )


@dataclass(slots=True)
class Backtester:
    """Walks a reader forward, one decision time at a time."""

    cost_model: CostModel
    pipeline: FeaturePipeline
    instruments: Mapping[str, InstrumentMeta]
    starting_equity_ccy: Decimal = Decimal("10000.00")
    position_notional_ccy: Decimal = Decimal("1000.00")
    rng_seed: int = 0
    # Set by the caller from the spec; the risk layer's own minimum is a
    # separate and stricter check applied in M4.
    min_holding_minutes: int = 0
    # Each instrument's corporate actions, unfiltered — the pipeline applies
    # the knowledge-time filter itself, and a holding follows a split whether
    # or not anyone had recorded it (`holding_factor`). Empty is right only
    # for series that cannot split, like the calibration's random walk: over
    # real history, leaving it out turns every split into a 75% "loss" inside
    # any holding across it and a step in every feature across it.
    actions: Mapping[str, Sequence[CorporateAction]] = field(default_factory=dict)

    _trades: list[Trade] = field(default_factory=list, init=False)
    _curve: list[CurvePoint] = field(default_factory=list, init=False)

    def run(
        self,
        *,
        strategy: Strategy,
        reader: ForwardOnlyReader,
        decision_times: Sequence[datetime],
        resolution: Resolution = Resolution.DAILY,
    ) -> BacktestResult:
        """Walk the schedule, and report what happened and what did not."""
        if len(decision_times) < 2:
            raise BacktestError(
                f"a backtest needs at least two decision times, got {len(decision_times)}. "
                "A fill comes from the bar *after* the decision, so a single-step run can "
                "produce a signal but never a trade."
            )
        if list(decision_times) != sorted(decision_times):
            raise BacktestError(
                "decision times must be ascending. The reader refuses to rewind, so an "
                "out-of-order schedule would raise part-way through a run rather than "
                "before it."
            )

        self._trades = []
        self._curve = []

        cash = self.starting_equity_ccy
        gross_cash = self.starting_equity_ccy
        open_positions: dict[str, _Position] = {}

        n_decisions = 0
        n_signals = 0
        n_rejected = 0
        n_unevaluable = 0
        n_dropped = 0
        total_cost = Decimal(0)
        traded_notional = Decimal(0)

        # Decisions made at step i are filled from the window at step i+1. The
        # pending map carries them across, which is what makes the fill price
        # structurally unavailable to the decision that caused it.
        pending: list[Decision] = []

        for index, moment in enumerate(decision_times):
            window = reader.advance_to(moment)
            self._follow_splits(open_positions, window)

            # --- fill what the previous step decided ----------------------
            for decision in pending:
                filled = self._fill(
                    decision=decision,
                    window=window,
                    open_positions=open_positions,
                )
                if filled is None:
                    n_dropped += 1
                    continue
                cost, notional, trade = filled
                total_cost += cost
                traded_notional += notional
                cash -= cost
                if trade is not None:
                    self._trades.append(trade)
                    cash += trade.gross_pnl_ccy
                    gross_cash += trade.gross_pnl_ccy
            pending = []

            # --- mark to market -------------------------------------------
            self._curve.append(
                CurvePoint(
                    equity_ccy=cash + self._unrealised(open_positions, window),
                    gross_equity_ccy=gross_cash + self._unrealised(open_positions, window),
                )
            )

            # --- decide, unless this is the last step ---------------------
            if index == len(decision_times) - 1:
                # No following bar exists, so any decision here could only be
                # filled at a price the decision itself could see.
                break

            for uid in sorted(self.instruments):
                snapshot = self.pipeline.compute(window, uid, actions=self.actions.get(uid, ()))
                position = open_positions.get(uid)
                decision = strategy.decide(
                    snapshot=snapshot,
                    window=window,
                    position=position.to_state() if position else PositionState(instrument_uid=uid),
                )
                n_decisions += 1
                if "unevaluable" in decision.rationale:
                    n_unevaluable += 1
                if not decision.wants_to_trade:
                    continue
                if decision.action is Action.ENTER:
                    n_signals += 1
                    if not self._affordable(decision, snapshot):
                        n_rejected += 1
                        continue
                    if uid in open_positions:
                        continue
                pending.append(decision)

        metrics = metrics_mod.compute(
            curve=self._curve,
            n_trades=len(self._trades),
            total_cost_ccy=total_cost,
            traded_notional_ccy=traded_notional,
            starting_equity_ccy=self.starting_equity_ccy,
            periods_per_year=_PERIODS_PER_YEAR[resolution],
        )

        return BacktestResult(
            backtest_id=new_id("bt", length=12),
            strategy_id=strategy.strategy_id,
            strategy_version=strategy.version,
            resolution=resolution,
            rng_seed=self.rng_seed,
            window_start=decision_times[0],
            window_end=decision_times[-1],
            metrics=metrics,
            trades=tuple(self._trades),
            curve=tuple(self._curve),
            n_decisions=n_decisions,
            n_signals=n_signals,
            n_rejected_by_cost_gate=n_rejected,
            n_unevaluable=n_unevaluable,
            n_dropped_no_next_bar=n_dropped,
            starting_equity_ccy=self.starting_equity_ccy,
        )

    # -- the cost gate -----------------------------------------------------

    def _affordable(self, decision: Decision, snapshot: FeatureSnapshot) -> bool:
        """Run the pre-trade cost gate. A refusal means the order never exists."""
        meta = self.instruments[decision.instrument_uid]
        verdict, _ = self.cost_model.gate_trade(
            notional_ccy=self.position_notional_ccy,
            instrument_currency=meta.currency,
            jurisdiction=meta.jurisdiction,
            expected_edge_bps=decision.expected_edge_bps,
        )
        return verdict.allowed

    # -- fills -------------------------------------------------------------

    def _fill(
        self,
        *,
        decision: Decision,
        window: BarWindow,
        open_positions: dict[str, _Position],
    ) -> tuple[Decimal, Decimal, Trade | None] | None:
        """Execute a decision at the *next* bar's open.

        Returns `(cost, notional, trade_or_None)`, or `None` if there is no bar
        to fill against. The price comes from this window — the one *after* the
        decision — so the decision could not have seen it.
        """
        uid = decision.instrument_uid
        bars = window.bars(uid)
        if not bars:
            return None
        fill_bar = bars[-1]
        # On **knowledge** time, not bar time. A bar is visible at a decision
        # precisely because its `available_at` has already passed, so its
        # `bar_open_utc` is necessarily *before* the decision instant —
        # comparing bar time here rejected every fill, and the calibration's
        # zero-trade check is what caught it. The question is not "did this bar
        # start after the decision" but "could the decision have known it",
        # and that is the third time axis the data layer exists to carry.
        if fill_bar.available_at_utc <= decision.as_of:
            return None
        price = fill_bar.open
        meta = self.instruments[uid]

        if decision.action is Action.ENTER:
            return self._open(
                decision=decision,
                price=price,
                at=fill_bar.bar_open_utc,
                quoted_through=fill_bar.quoted_through,
                meta=meta,
                open_positions=open_positions,
            )
        if decision.action is Action.EXIT:
            return self._close(
                decision=decision,
                price=price,
                at=fill_bar.bar_open_utc,
                meta=meta,
                open_positions=open_positions,
            )
        return None

    def _open(
        self,
        *,
        decision: Decision,
        price: Decimal,
        at: datetime,
        quoted_through: date,
        meta: InstrumentMeta,
        open_positions: dict[str, _Position],
    ) -> tuple[Decimal, Decimal, None]:
        notional = self.position_notional_ccy
        with localcontext() as ctx:
            ctx.prec = _WORKING_PRECISION
            quantity = (notional / price).quantize(Decimal("0.000001"))
        leg = self.cost_model.leg(
            notional_ccy=notional,
            side="buy",
            instrument_currency=meta.currency,
            jurisdiction=meta.jurisdiction,
        )
        trip = self.cost_model.round_trip(
            notional_ccy=notional,
            instrument_currency=meta.currency,
            jurisdiction=meta.jurisdiction,
        )
        open_positions[decision.instrument_uid] = _Position(
            instrument_uid=decision.instrument_uid,
            quantity=quantity,
            entry_price=price,
            entry_at=at,
            entry_notional_ccy=notional,
            entry_cost_ccy=leg.total_ccy,
            entry_breakdown=leg.itemised(),
            expected_edge_bps=decision.expected_edge_bps,
            expected_cost_bps=trip.total_bps,
            quoted_through=quoted_through,
        )
        return leg.total_ccy, notional, None

    def _close(
        self,
        *,
        decision: Decision,
        price: Decimal,
        at: datetime,
        meta: InstrumentMeta,
        open_positions: dict[str, _Position],
    ) -> tuple[Decimal, Decimal, Trade | None] | None:
        position = open_positions.pop(decision.instrument_uid, None)
        if position is None:
            return None

        with localcontext() as ctx:
            ctx.prec = _WORKING_PRECISION
            exit_notional = (position.quantity * price).quantize(_CENTS, ROUND_HALF_EVEN)
            gross_pnl = (exit_notional - position.entry_notional_ccy).quantize(
                _CENTS, ROUND_HALF_EVEN
            )

        leg = self.cost_model.leg(
            notional_ccy=exit_notional,
            side="sell",
            instrument_currency=meta.currency,
            jurisdiction=meta.jurisdiction,
        )
        total_cost = position.entry_cost_ccy + leg.total_ccy
        holding = int((at - position.entry_at).total_seconds() // 60)

        trade = Trade(
            trade_seq=len(self._trades) + 1,
            instrument_uid=decision.instrument_uid,
            entry_at=position.entry_at,
            exit_at=at,
            entry_price=position.entry_price,
            exit_price=price,
            quantity=position.quantity,
            gross_pnl_ccy=gross_pnl,
            net_pnl_ccy=gross_pnl - total_cost,
            cost_total_ccy=total_cost,
            cost_breakdown={
                **{f"entry_{k}": v for k, v in position.entry_breakdown.items()},
                **{f"exit_{k}": v for k, v in leg.itemised().items()},
            },
            expected_edge_bps=position.expected_edge_bps,
            expected_cost_bps=position.expected_cost_bps,
            holding_minutes=holding,
            exit_reason=decision.rationale,
        )
        return leg.total_ccy, exit_notional, trade

    # -- splits ------------------------------------------------------------

    def _follow_splits(self, open_positions: Mapping[str, _Position], window: BarWindow) -> None:
        """Carry each open holding onto the scale of its newest visible price.

        Run as each step's window arrives, before anything fills or is marked
        against it. A 4-for-1 between entry and exit leaves four shares for each
        one bought, at a quarter of the price; without this the exit notional
        was the old share count at the new price — a 75% loss on a trade that
        made nothing, in every backtest holding across a split, which the
        search would then learn to avoid as though it were a signal. Quantity
        and entry price move in step, so the entry notional, and the cost
        already charged on it, stand.
        """
        for uid, position in open_positions.items():
            newest = window.last(uid)
            if newest is UNKNOWN or not isinstance(newest, Bar):
                continue
            to = newest.quoted_through
            if to == position.quoted_through:
                continue
            ratio = holding_factor(
                self.actions.get(uid, ()), quoted_through=position.quoted_through, to=to
            )
            if ratio != 1:
                with localcontext() as ctx:
                    ctx.prec = _WORKING_PRECISION
                    shares, per = Decimal(ratio.numerator), Decimal(ratio.denominator)
                    position.quantity = position.quantity * shares / per
                    position.entry_price = position.entry_price * per / shares
            position.quoted_through = to

    # -- marking -----------------------------------------------------------

    def _unrealised(self, open_positions: Mapping[str, _Position], window: BarWindow) -> Decimal:
        """Mark open positions at the newest visible close.

        Zero when a position has no visible bar, rather than carrying the entry
        notional forward: an equity curve that quietly held a stale mark would
        smooth exactly the drawdown a gate is looking for.
        """
        total = Decimal(0)
        for uid, position in open_positions.items():
            last = window.last(uid)
            if last is UNKNOWN or not isinstance(last, Bar):
                continue
            with localcontext() as ctx:
                ctx.prec = _WORKING_PRECISION
                marked = (position.quantity * last.close).quantize(_CENTS, ROUND_HALF_EVEN)
                total += marked - position.entry_notional_ccy
        return total


def round_trip_for(model: CostModel, meta: InstrumentMeta, notional_ccy: Decimal) -> RoundTrip:
    """Convenience for reporting: what a round trip in this name costs."""
    return model.round_trip(
        notional_ccy=notional_ccy,
        instrument_currency=meta.currency,
        jurisdiction=meta.jurisdiction,
    )
