"""The tick loop: one cycle from bars to orders, with the gates in order.

This is where every layer built so far meets, and the order is the design:

    self-check -> lease -> recover -> config hash -> regime
      -> per instrument: features -> strategy -> risk -> intent -> broker
      -> protective stop
      -> record the cycle

Five things about that order are load-bearing.

**Recovery runs before the first decision of every cycle, not only after a
known crash.** A run that ended cleanly cannot be distinguished from one that
did not without reading the intent table, so the loop reads it every time.
An unresolved unknown halts rather than being carried.

**The config hash is re-verified every cycle.** The limits file is the control
layer, and a control layer that is only checked at startup can be edited
underneath a running process. Drift is a halt, not a reload: a cap that
changed while positions were open was not in force when they were sized.

**Risk-reducing work is done first, and separately.** Exits and protective
stops are processed in their own pass before any entry is considered, so a
batch of entries cannot consume the rate-limit budget an exit needs. That is
the priority queue, and it is an ordering rather than a data structure because
the set of work per cycle is small and bounded.

**The protective stop is placed immediately after the entry fill, in the same
cycle.** The window between them is unavoidable — the venue has no bracket
orders — but leaving it until the next cycle would make it a tick long instead
of a second long, and the sizing rules assume the short version.

**A refused decision is recorded as fully as an accepted one.** Every rule's
verdict goes to the ledger whether the order was placed or not, because "why
did it stop trading" is the more common question.

The loop holds no clock and no network of its own: the clock is injected, the
broker and the provider are passed in. `run_cycle` is a single pass so a test
can drive it one tick at a time, and `run_forever` is a thin sleep-and-repeat
over it.
"""

from __future__ import annotations

import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from datetime import datetime
from decimal import Decimal

from tb.broker.port import Broker, OrderPurpose, OrderType, TimeValidity
from tb.config.loader import PinnedLimits, load_hard_limits
from tb.core.clock import now_utc
from tb.core.errors import TbError
from tb.core.ids import new_id
from tb.data.asof import BarSource, BarWindow, visible_bars
from tb.data.regime import RegimeGate, RegimeReading
from tb.data.symbols import SymbolMap
from tb.engine.intents import IntentLog
from tb.engine.orders import OrderSubmitter, SubmissionError, SubmissionUnknown
from tb.features.pipeline import FeaturePipeline, FeatureSnapshot
from tb.ledger.events import (
    Actor,
    DecisionPayload,
    EventType,
    LoopCyclePayload,
    RiskEvaluationPayload,
    RiskVerdictRow,
)
from tb.ledger.store import Ledger
from tb.ops.state import RunState, StateMachine
from tb.ops.watchdog import InstanceLock, InstanceLockRefused, SelfCheck, WatchdogError
from tb.portfolio.pnl import EquityCurve
from tb.risk.engine import Evaluation, RiskEngine, entry_request, exit_request, stop_price_for
from tb.risk.state import AccountState, RiskContext
from tb.strategy.base import Action, Decision, PositionState, Strategy


class LoopHalted(TbError):
    """The loop stopped itself. The reason is in the message and the ledger."""


@dataclass(frozen=True, slots=True)
class CycleResult:
    """What one pass did. Returned rather than logged, so a test can assert it."""

    cycle: int
    as_of: datetime
    n_considered: int
    decisions: tuple[Decision, ...] = ()
    submitted: tuple[str, ...] = ()
    refusals: tuple[tuple[str, str], ...] = ()
    stops_placed: tuple[str, ...] = ()
    regime: RegimeReading | None = None
    duration_ms: float = 0.0
    halted: bool = False
    detail: str = ""

    @property
    def traded(self) -> bool:
        return bool(self.submitted)


@dataclass
class TradingLoop:
    """One cycle per call. Holds the collaborators, no state of its own.

    `instruments` maps a broker ticker to the instrument uid its bars are
    stored under. Both are needed and neither derives from the other: the
    order goes to the ticker, the features come from the uid, and conflating
    them is the mismapping the symbol layer exists to prevent.
    """

    ledger: Ledger
    pinned: PinnedLimits
    broker: Broker
    bars: BarSource
    strategy: Strategy
    pipeline: FeaturePipeline
    submitter: OrderSubmitter
    log: IntentLog
    run_id: str
    instruments: dict[str, str]
    state: StateMachine
    self_check: SelfCheck
    lock: InstanceLock
    equity: EquityCurve
    risk: RiskEngine = field(default_factory=RiskEngine)
    clock: Callable[[], datetime] = now_utc
    resolution: str = "daily"
    _cycle: int = field(default=0, init=False)

    # -- one pass ----------------------------------------------------------

    def run_cycle(self) -> CycleResult:
        """One tick. Raises `LoopHalted` rather than trading through a problem."""
        started = time.monotonic()
        self._cycle += 1
        at = self.clock()

        self._preflight(at)
        regime = self._read_regime(at)

        decisions: list[Decision] = []
        submitted: list[str] = []
        refusals: list[tuple[str, str]] = []
        stops: list[str] = []

        # Two passes, risk-reducing first. An ordering rather than a queue
        # object: the per-cycle work is small and bounded, and the property
        # that matters is only that no entry is attempted before every exit
        # has been.
        for risk_reducing in (True, False):
            for ticker, uid in sorted(self.instruments.items()):
                outcome = self._consider(
                    ticker=ticker,
                    uid=uid,
                    at=at,
                    regime=regime,
                    risk_reducing_pass=risk_reducing,
                )
                if outcome is None:
                    continue
                decision, intent_id, refusal, stop_id = outcome
                if decision is not None:
                    decisions.append(decision)
                if intent_id is not None:
                    submitted.append(intent_id)
                if refusal is not None:
                    refusals.append(refusal)
                if stop_id is not None:
                    stops.append(stop_id)

        duration = (time.monotonic() - started) * 1000
        result = CycleResult(
            cycle=self._cycle,
            as_of=at,
            n_considered=len(self.instruments),
            decisions=tuple(decisions),
            submitted=tuple(submitted),
            refusals=tuple(refusals),
            stops_placed=tuple(stops),
            regime=regime,
            duration_ms=duration,
        )
        self._record_cycle(result)
        return result

    def run_forever(
        self, *, interval_seconds: float, max_cycles: int | None = None
    ) -> tuple[CycleResult, ...]:
        """Sleep and repeat. Stops on a halt rather than retrying through it.

        `max_cycles` exists so a test can bound it; production passes `None`.
        A halt propagates: the loop's job is to stop, and a supervisor's job is
        to decide whether to restart it. A loop that caught its own halt and
        carried on would defeat every control above.
        """
        results: list[CycleResult] = []
        while max_cycles is None or len(results) < max_cycles:
            results.append(self.run_cycle())
            if max_cycles is not None and len(results) >= max_cycles:
                break
            time.sleep(interval_seconds)
        return tuple(results)

    # -- the gates before any decision -------------------------------------

    def _preflight(self, at: datetime) -> None:
        """Everything that must be true before the strategy is asked anything."""
        # 1. The self-facing dead-man switch. First, because a trader that
        #    cannot write the ledger must not reach the code that would place
        #    an order it then cannot record.
        try:
            self.self_check.assert_alive(at=at)
        except WatchdogError as exc:
            self._halt(f"self-check failed: {exc}")

        # 2. The lease. Renewed rather than re-acquired: a renew that fails
        #    means another instance took over, which is the two-instance state
        #    the lock exists to prevent.
        try:
            self.lock.renew(at=at)
        except InstanceLockRefused as exc:
            self._halt(f"lost the trading lease: {exc}")

        # 3. The control layer, re-verified. A limits file that changed under a
        #    running process means the caps that sized the open positions are
        #    not the caps now in force, so this is a halt rather than a reload.
        current = load_hard_limits(self.pinned.source_path)
        if current.config_hash != self.pinned.config_hash:
            self._halt(
                f"hard limits changed under the running process: pinned "
                f"{self.pinned.config_hash[:12]}, now {current.config_hash[:12]}. The caps "
                "that sized the open positions are no longer the caps in force."
            )

        # 4. Recovery. Every cycle, not only after a known crash — a clean
        #    shutdown and a crash are indistinguishable without reading this.
        _, blocking = self.submitter.recover()
        if blocking:
            names = ", ".join(i.intent_id for i in blocking)
            self._halt(
                f"{len(blocking)} intent(s) are in an unknown state and could not be "
                f"resolved against the broker's open orders: {names}. An order may exist "
                "for each. Run `tb reconcile`, which reads order history, rather than "
                "trading past it."
            )

        # 5. The state machine. Only TRADING may place orders, and reaching it
        #    from BOOT goes through RECONCILING — so the transition itself
        #    records that recovery happened.
        reading = self.state.current()
        if reading.state is RunState.BOOT:
            self.state.transition_to(RunState.RECONCILING, reason="cycle preflight")
            reading = self.state.current()
        if reading.state is RunState.RECONCILING:
            self.state.transition_to(RunState.TRADING, reason="recovery clean")
            reading = self.state.current()
        if not reading.state.may_place_orders:
            self._halt(f"run state is {reading.state.value}, which may not place orders")

        self.self_check.beat(
            state=RunState.TRADING.value,
            heartbeat_path=self.pinned.limits.safety.heartbeat_path,  # type: ignore[arg-type]
        )

        # 6. The equity mark, before any decision. On the first cycle of a
        #    session this is what establishes the day's opening equity, so a
        #    cycle that decided first would compute today's P&L against
        #    yesterday's close and attribute the overnight gap to today —
        #    firing the daily breaker on exposure the unprotected-window
        #    sizing had already budgeted for.
        #
        #    A broker that reports no equity leaves the curve unmarked, and
        #    the three loss percentages then reach the rules as `None`, which
        #    every one of them blocks on. That is the fail-closed reading: an
        #    unmeasured account is not a flat one.
        cash = self.broker.get_cash()
        if cash.equity is not None and cash.equity > 0:
            positions = self.broker.get_positions()
            self.equity.mark(
                equity_ccy=cash.equity,
                at=at,
                currency=cash.currency,
                deployed_ccy=sum((p.market_value or Decimal(0) for p in positions), Decimal(0)),
                free_cash_ccy=cash.free,
            )

    def _read_regime(self, at: datetime) -> RegimeReading:
        """The portfolio-wide exposure factor.

        Read once per cycle rather than per instrument, so every decision in
        one cycle is sized against the same reading — two instruments scaled
        by different factors would not be a coherent portfolio.
        """
        return RegimeGate(limits=self.pinned.limits).read(self.bars, as_of=at)

    # -- one instrument ----------------------------------------------------

    def _consider(
        self,
        *,
        ticker: str,
        uid: str,
        at: datetime,
        regime: RegimeReading,
        risk_reducing_pass: bool,
    ) -> tuple[Decision | None, str | None, tuple[str, str] | None, str | None] | None:
        """Ask the strategy about one instrument and act on the answer.

        Returns `None` when this instrument is not this pass's business, so
        the caller can tell "considered and declined" from "not considered" —
        which matters for the cycle record.
        """
        position = self._position(ticker, uid)

        # The pass filter. An instrument with no position has nothing
        # risk-reducing to do; one with a position is handled on the first
        # pass and skipped on the second, so an add is deferred a cycle rather
        # than competing with its own exit for the budget.
        if risk_reducing_pass != position.is_open:
            return None

        window = self._window(uid, at)
        snapshot = self.pipeline.compute(window, uid)
        decision = self.strategy.decide(snapshot=snapshot, window=window, position=position)
        decision_id = self._record_decision(
            decision, ticker=ticker, snapshot=snapshot, regime=regime
        )

        if not decision.wants_to_trade:
            return decision, None, None, None

        reference = self._reference_price(window, uid)
        if reference is None:
            return decision, None, (ticker, "no usable reference price in the window"), None

        evaluation = self._evaluate(
            decision=decision,
            decision_id=decision_id,
            ticker=ticker,
            uid=uid,
            at=at,
            reference=reference,
            position=position,
            regime=regime,
        )
        self._record_risk(evaluation, decision_id=decision_id)

        if evaluation.token is None:
            return decision, None, (ticker, evaluation.refusal_summary), None

        try:
            submission = self.submitter.submit(
                evaluation.token,
                order_type=OrderType.MARKET,
                purpose=(
                    OrderPurpose.EXIT if decision.action is Action.EXIT else OrderPurpose.ENTRY
                ),
                instrument_uid=uid,
            )
        except SubmissionUnknown as exc:
            # Never retried here. The loop halts and lets recovery — which can
            # read the broker — settle it.
            self._halt(str(exc))
            raise  # pragma: no cover - _halt always raises
        except SubmissionError as exc:
            # Conclusive refusal. Recorded and skipped, not halted: an
            # ordinary rejection must not stop the whole loop.
            return decision, None, (ticker, f"rejected: {exc}"), None

        stop_id: str | None = None
        if decision.action is Action.ENTER and submission.filled:
            stop_id = self._protect(
                ticker=ticker,
                uid=uid,
                quantity=evaluation.approved_quantity or Decimal(0),
                entry_price=reference,
                at=at,
                decision_id=decision_id,
                parent_intent_id=submission.intent.intent_id,
            )

        return decision, submission.intent.intent_id, None, stop_id

    def _protect(
        self,
        *,
        ticker: str,
        uid: str,
        quantity: Decimal,
        entry_price: Decimal,
        at: datetime,
        decision_id: str,
        parent_intent_id: str,
    ) -> str | None:
        """Place the protective stop, in the same cycle as the entry fill.

        The window between the fill and this order is unavoidable on a venue
        with no bracket orders, but it is *bounded* by doing this here rather
        than next cycle. A failure to protect is recorded loudly and the
        position is left held: flattening on a failed stop would turn a data
        problem into a realised loss, and `on_unprotected_position` in the
        limits is where that policy is chosen rather than here.
        """
        stop_price = stop_price_for(entry_price=entry_price, limits=self.pinned.limits)
        request = exit_request(
            t212_ticker=ticker,
            instrument_uid=uid,
            reference_price=entry_price,
            quantity=quantity,
            purpose=OrderPurpose.PROTECTIVE_STOP,
            decision_id=decision_id,
        )
        evaluation = self.risk.evaluate(
            self._context(
                request=request,
                at=at,
                position=PositionState(instrument_uid=uid, quantity=quantity, entry_at=at),
                regime=None,
                ticker=ticker,
            ),
            run_id=self.run_id,
        )
        if evaluation.token is None:
            self.submitter.record_protection(
                t212_ticker=ticker,
                quantity=quantity,
                protected=False,
                detail=f"risk refused the protective stop: {evaluation.refusal_summary}",
            )
            return None

        try:
            stop = self.submitter.submit(
                evaluation.token,
                order_type=OrderType.STOP,
                purpose=OrderPurpose.PROTECTIVE_STOP,
                stop_price=stop_price,
                time_validity=TimeValidity.GOOD_TILL_CANCEL,
                parent_intent_id=parent_intent_id,
                instrument_uid=uid,
            )
        except (SubmissionError, SubmissionUnknown) as exc:
            # Recorded as unprotected and *not* halted on a conclusive
            # rejection: the position exists either way, and halting would
            # leave it unprotected as well as unmanaged.
            self.submitter.record_protection(
                t212_ticker=ticker,
                quantity=quantity,
                protected=False,
                detail=f"the protective stop could not be placed: {exc}",
            )
            return None

        self.submitter.record_protection(
            t212_ticker=ticker,
            quantity=quantity,
            protected=True,
            stop_intent_id=stop.intent.intent_id,
            stop_price=stop_price,
            unprotected_seconds=(self.clock() - at).total_seconds(),
        )
        return stop.intent.intent_id

    # -- assembling what the rules read ------------------------------------

    def _evaluate(
        self,
        *,
        decision: Decision,
        decision_id: str,
        ticker: str,
        uid: str,
        at: datetime,
        reference: Decimal,
        position: PositionState,
        regime: RegimeReading,
    ) -> Evaluation:
        if decision.action is Action.EXIT:
            request = exit_request(
                t212_ticker=ticker,
                instrument_uid=uid,
                reference_price=reference,
                quantity=position.quantity,
                decision_id=decision_id,
                strategy_id=decision.strategy_id,
            )
        else:
            request = entry_request(
                t212_ticker=ticker,
                instrument_uid=uid,
                reference_price=reference,
                expected_edge_bps=decision.expected_edge_bps,
                decision_id=decision_id,
                strategy_id=decision.strategy_id,
            )
        return self.risk.evaluate(
            self._context(request=request, at=at, position=position, regime=regime, ticker=ticker),
            run_id=self.run_id,
        )

    def _context(
        self,
        *,
        request: object,
        at: datetime,
        position: PositionState,
        regime: RegimeReading | None,
        ticker: str,
    ) -> RiskContext:
        """Build the snapshot every rule reads, once per decision.

        Assembled here rather than by each rule, so two rules cannot disagree
        about equity — a set of verdicts computed against different account
        states is not a coherent decision.
        """
        from tb.risk.state import OrderRequest

        assert isinstance(request, OrderRequest)
        symbols = SymbolMap(self.ledger, provider="alpaca")
        may_enter, enter_reason = symbols.may_enter(ticker)
        may_exit, exit_reason = symbols.may_exit(ticker)
        mapping = symbols.get(ticker)

        total, for_symbol = self.log.counts_today(day=at, t212_ticker=ticker)
        since_open, until_close = self._session_position(at)
        instrument = next((i for i in self.broker.get_instruments() if i.ticker == ticker), None)
        return RiskContext(
            as_of=at,
            limits=self.pinned.limits,
            request=request,
            account=self._account(total=total, for_symbol=for_symbol, at=at),
            position_quantity=position.quantity,
            position_entry_at=position.entry_at,
            may_enter=may_enter,
            may_enter_reason=enter_reason,
            may_exit=may_exit,
            may_exit_reason=exit_reason,
            permits_full_size=bool(mapping and mapping.permits_full_size),
            # `None` for a protective stop: the regime scales *new exposure*,
            # and a stop is not new exposure. The rule reads it as
            # NOT_APPLICABLE for a risk-reducing order, so passing the reading
            # would be harmless — but passing None makes the intent explicit.
            regime_exposure_factor=regime.exposure_factor if regime else None,
            regime_state=regime.state.value if regime else None,
            bar_age_seconds=self._bar_age(request.instrument_uid, at),
            bar_period_seconds=_period_seconds(self.resolution),
            minutes_since_open=since_open,
            minutes_until_close=until_close,
            min_trade_quantity=instrument.min_trade_quantity if instrument else None,
            max_open_quantity=instrument.max_open_quantity if instrument else None,
            extra={
                # The ISIN comes from the broker's cached instrument record in
                # the ledger, not from the live instrument list: `/instruments`
                # is rate-limited to about one call per fifty seconds, and the
                # cost gate needs the jurisdiction on every decision. An empty
                # ISIN means the cost model refuses to guess — a UK-domiciled
                # name costs 50bps more per entry than a US one, so a default
                # here would be a silent 50bps error.
                "isin": self._isin(ticker),
                "instrument_currency": (instrument.currency_code if instrument else "")
                or self.pinned.limits.currency,
            },
        )

    def _account(self, *, total: int, for_symbol: int, at: datetime) -> AccountState:
        """The account as the broker last described it, plus its P&L.

        Every field stays optional through to the rules. A missing equity must
        reach them as missing rather than as zero, because a percentage cap
        evaluated against zero passes trivially — and the three loss
        percentages must reach them as `None` rather than `0.0` for the same
        reason. A flat day and an unmeasured day are different facts, and only
        one of them is a reason to keep trading.

        The P&L comes from the equity curve, which the cycle marks before any
        decision. That ordering matters on the first cycle of a session: the
        mark is what establishes the day's opening equity, so a cycle that
        decided before marking would compute today's P&L against yesterday's
        close and attribute the overnight gap to today.
        """
        cash = self.broker.get_cash()
        positions = self.broker.get_positions()
        deployed = sum(
            (p.market_value or Decimal(0) for p in positions),
            Decimal(0),
        )
        pnl = self.equity.read(at=at)
        return AccountState(
            equity_ccy=cash.equity,
            free_cash_ccy=cash.free,
            blocked_cash_ccy=cash.blocked,
            deployed_ccy=deployed,
            n_open_positions=len([p for p in positions if p.quantity > 0]),
            currency=cash.currency,
            day_pnl_pct=pnl.day_pnl_pct,
            rolling_5d_pnl_pct=pnl.rolling_pnl_pct,
            drawdown_from_peak_pct=pnl.drawdown_from_peak_pct,
            orders_today=total,
            orders_today_for_symbol=for_symbol,
            orders_last_hour=total,
        )

    def _isin(self, ticker: str) -> str:
        """The instrument's ISIN, from the broker's cached record.

        The ISIN rather than the ticker suffix, because it is the
        incorporation country and therefore the right key for a transaction
        tax — a `.L` suffix means "listed in London", which is not the same
        thing and differs by 50bps on every entry.
        """
        row = self.ledger.conn.execute(
            "SELECT isin FROM instruments WHERE ticker = ?", (ticker,)
        ).fetchone()
        if row is None or not row["isin"]:
            return ""
        return str(row["isin"])

    def _position(self, ticker: str, uid: str) -> PositionState:
        held = self.broker.get_position(ticker)
        if held is None or held.quantity <= 0:
            return PositionState(instrument_uid=uid)
        entry_at: datetime | None = None
        if held.initial_fill_date:
            from contextlib import suppress

            from tb.core.clock import from_iso

            with suppress(Exception):
                entry_at = from_iso(held.initial_fill_date)
        return PositionState(
            instrument_uid=uid,
            quantity=held.quantity,
            entry_price=held.average_price,
            entry_at=entry_at,
        )

    def _window(self, uid: str, at: datetime) -> BarWindow:
        from tb.data.provider import Resolution

        resolution = Resolution(self.resolution)
        rows = visible_bars(self.bars, uid, resolution, as_of=at)
        return BarWindow(as_of=at, resolution=resolution, _by_uid={uid: rows})

    def _reference_price(self, window: BarWindow, uid: str) -> Decimal | None:
        """The raw close of the newest visible bar.

        `RAW` deliberately: this price is used for sizing, for the stop level
        and for the cost arithmetic, and all three are about the price the
        broker will actually transact at. A split-adjusted price would be a
        different number.
        """
        rows = window.bars(uid)
        return rows[-1].close if rows else None

    def _bar_age(self, uid: str, at: datetime) -> float | None:
        rows = self._window(uid, at).bars(uid)
        if not rows:
            return None
        # Against knowledge time, never bar time. Bar time would understate
        # the age by exactly the provider delay, which is the one number this
        # bound exists to catch.
        return (at - rows[-1].available_at_utc).total_seconds()

    def _session_position(self, at: datetime) -> tuple[int | None, int | None]:
        """Minutes since the open and until the close, or `(None, None)`.

        Both `None` when the market is closed or the day is outside the
        calendar's range — which the session-window rule reads as "unknown"
        and blocks on. That is the right answer: an entry outside a session
        cannot fill, and one in an unclassifiable day cannot be sized against
        a close that is not known.

        Derived from `classify` rather than from a helper on the calendar,
        because the calendar's own gate (`may_enter_now`) answers a *different*
        question — whether to enter at all — and the risk rule needs the two
        numbers so it can record how close to the boundary it was.
        """
        from tb.data.calendar import TradingCalendar

        day = TradingCalendar().classify(at.date())
        if not day.is_trading_day or day.open_utc is None or day.close_utc is None:
            return None, None
        if not day.contains(at):
            return None, None
        since = int((at - day.open_utc).total_seconds() // 60)
        until = int((day.close_utc - at).total_seconds() // 60)
        return since, until

    # -- recording ---------------------------------------------------------

    def _record_decision(
        self,
        decision: Decision,
        *,
        ticker: str,
        snapshot: FeatureSnapshot,
        regime: RegimeReading,
    ) -> str:
        """Record what the strategy saw and concluded, including a HOLD.

        HOLDs are recorded too. "The strategy looked and declined" is a
        different fact from "the strategy was not consulted", and only the
        first is evidence the loop was alive.
        """
        decision_id = new_id("dec")
        with self.ledger.transaction() as tx:
            event = tx.append(
                EventType.DECISION_MADE,
                decision.instrument_uid,
                DecisionPayload(
                    decision_id=decision_id,
                    run_id=self.run_id,
                    strategy_id=decision.strategy_id,
                    strategy_version=decision.strategy_version,
                    instrument_uid=decision.instrument_uid,
                    as_of_utc=decision.as_of.isoformat(),
                    resolution=self.resolution,
                    action=decision.action.value,
                    feature_snapshot_hash=decision.feature_snapshot_hash,
                    feature_vector={name: str(value) for name, value in snapshot.values.items()},
                    t212_ticker=ticker,
                    expected_edge_bps=float(decision.expected_edge_bps),
                    rationale=decision.rationale,
                    regime_state=regime.state.value,
                    regime_exposure_factor=regime.exposure_factor,
                ),
                actor=Actor.SYSTEM,
                run_id=self.run_id,
            )
            tx.execute(
                "INSERT INTO decisions (decision_id, run_id, strategy_id, strategy_version,"
                " instrument_uid, t212_ticker, as_of_utc, resolution,"
                " feature_snapshot_hash, feature_vector_json, action, expected_edge_bps,"
                " rationale, regime_state, regime_exposure_factor, decided_at,"
                " deciding_event_seq) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                (
                    decision_id,
                    self.run_id,
                    decision.strategy_id,
                    decision.strategy_version,
                    decision.instrument_uid,
                    ticker,
                    decision.as_of.isoformat(),
                    self.resolution,
                    decision.feature_snapshot_hash,
                    repr({k: str(v) for k, v in snapshot.values.items()}),
                    decision.action.value,
                    float(decision.expected_edge_bps),
                    decision.rationale,
                    regime.state.value,
                    str(regime.exposure_factor),
                    self.clock().isoformat(),
                    event.seq,
                ),
            )
        return decision_id

    def _record_risk(self, evaluation: Evaluation, *, decision_id: str) -> None:
        """Every rule's verdict, pass or fail, in one transaction."""
        with self.ledger.transaction() as tx:
            tx.append(
                EventType.RISK_EVALUATED,
                decision_id,
                RiskEvaluationPayload(
                    decision_id=decision_id,
                    run_id=self.run_id,
                    approved=evaluation.approved,
                    n_rules_evaluated=evaluation.n_evaluated,
                    n_blocking_failures=len(evaluation.blocking),
                    verdicts=[
                        RiskVerdictRow(
                            rule_name=v.rule_name,
                            verdict=v.verdict.value,
                            limit_value=v.limit_value,
                            observed_value=v.observed_value,
                            detail=v.detail,
                            is_blocking=v.is_blocking,
                        )
                        for v in evaluation.verdicts
                    ],
                    approved_quantity=evaluation.approved_quantity,
                    approved_notional_ccy=evaluation.approved_notional_ccy,
                    token_id=evaluation.token.token_id if evaluation.token else None,
                    refusal_summary=evaluation.refusal_summary,
                ),
                actor=Actor.RISK,
                run_id=self.run_id,
            )
            for verdict in evaluation.verdicts:
                tx.execute(
                    "INSERT OR REPLACE INTO risk_verdicts (decision_id, rule_name, verdict,"
                    " limit_value, observed_value, detail, is_blocking, evaluated_at)"
                    " VALUES (?,?,?,?,?,?,?,?)",
                    (
                        decision_id,
                        verdict.rule_name,
                        verdict.verdict.value,
                        None if verdict.limit_value is None else str(verdict.limit_value),
                        None if verdict.observed_value is None else str(verdict.observed_value),
                        verdict.detail,
                        1 if verdict.is_blocking else 0,
                        self.clock().isoformat(),
                    ),
                )

    def _record_cycle(self, result: CycleResult) -> None:
        self.ledger.append(
            EventType.LOOP_CYCLE_COMPLETED,
            self.run_id,
            LoopCyclePayload(
                run_id=self.run_id,
                cycle=result.cycle,
                as_of_utc=result.as_of.isoformat(),
                n_instruments_considered=result.n_considered,
                n_decisions=len(result.decisions),
                n_orders_submitted=len(result.submitted),
                n_risk_refusals=len(result.refusals),
                duration_ms=result.duration_ms,
                halted=result.halted,
                detail=result.detail,
            ),
            actor=Actor.SYSTEM,
            run_id=self.run_id,
        )

    def _halt(self, reason: str) -> None:
        """Move to HALTED and raise. Never returns."""
        from contextlib import suppress

        with suppress(Exception):
            self.state.transition_to(RunState.HALTED, reason=reason)
        raise LoopHalted(reason)


def _period_seconds(resolution: str) -> int:
    """One bar's span, for the staleness bound.

    A table rather than arithmetic on the enum, because the answer for
    `daily` is a calendar question with a conventional answer (86400) rather
    than a session length — a US session is 6.5 hours, but a daily bar's
    successor arrives the next day, and the bound is about when the next
    price is available.
    """
    return {"minute": 60, "hourly": 3600, "daily": 86400}.get(resolution, 60)


def build_instrument_map(
    ledger: Ledger, *, provider: str = "alpaca", tickers: Sequence[str] | None = None
) -> dict[str, str]:
    """Ticker -> instrument uid, for the instruments the loop may trade.

    Only mappings that permit an entry are included, which means the loop's
    universe is the *verified* universe by construction rather than by a check
    inside the cycle. An unverified symbol is simply absent, and `tb symbols
    audit` is where that is visible.
    """
    from tb.data.provider import make_instrument_uid

    symbols = SymbolMap(ledger, provider=provider)
    wanted = set(tickers) if tickers else None
    out: dict[str, str] = {}
    for mapping in symbols.all():
        if wanted is not None and mapping.t212_ticker not in wanted:
            continue
        if not mapping.may_enter and not mapping.may_exit:
            continue
        row = ledger.conn.execute(
            "SELECT isin FROM instruments WHERE ticker = ?", (mapping.t212_ticker,)
        ).fetchone()
        isin = None if row is None else row["isin"]
        out[mapping.t212_ticker] = make_instrument_uid(
            isin=isin, t212_ticker=mapping.t212_ticker, data_symbol=mapping.data_symbol
        )
    return out
