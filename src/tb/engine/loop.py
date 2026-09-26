"""The tick loop: one cycle from bars to orders, with the gates in order.

This is where every layer built so far meets, and the order is the design:

    self-check -> lease -> recover -> config hash -> regime
      -> flatten what no funded strategy owns
      -> per instrument, per funded strategy:
           features -> strategy -> risk -> intent -> broker
      -> protective stop
      -> record the cycle

Seven things about that order are load-bearing.

**The loop trades a funded *book*, not a strategy.** `book` comes from
`tb.engine.funding`, which reads the registry, the ladder and the allocator —
so a promotion changes what this loop asks and a retirement changes it back.
Each funded strategy carries its own spec-derived pipeline, because a shared
pipeline that happened not to compute a feature some spec reads would make that
spec evaluate to `UNKNOWN` at every decision and look like a strategy that
never found an opportunity.

**A position belongs to the strategy whose entry opened it.** Only that
strategy is asked about it, so two strategies cannot take turns deciding the
same holding — and an instrument no funded strategy owns is flattened before
anything else happens, because a position nothing will ever decide to close is
unmanaged exposure and that is the state this whole design exists to avoid.

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

**Two funded strategies wanting the same flat instrument is resolved, not
averaged.** The first in book order takes it and the second is refused with the
winner named. Book order is by label, so the resolution is a function of
identity rather than of promotion time — which is what makes a replay resolve
the contention the same way the live run did.

The loop holds no clock and no network of its own: the clock is injected, the
broker and the provider are passed in. `run_cycle` is a single pass so a test
can drive it one tick at a time, and `run_forever` is a thin sleep-and-repeat
over it.
"""

from __future__ import annotations

import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from datetime import date, datetime
from decimal import Decimal

from tb.broker.port import Broker, OrderPurpose, OrderType, Side, TimeValidity
from tb.broker.t212.errors import BrokerHttpError, RateLimited
from tb.config.loader import PinnedLimits, load_hard_limits
from tb.core.clock import now_utc
from tb.core.errors import TbError, TransportError
from tb.core.ids import new_id
from tb.data.actions import ActionStore
from tb.data.adjustments import CorporateAction
from tb.data.asof import BarSource, BarWindow, visible_bars
from tb.data.regime import RegimeGate, RegimeReading
from tb.data.symbols import SymbolMap
from tb.engine.funding import Book, FundedStrategy, Ownership, owner_of, unowned_positions
from tb.engine.intents import IntentLog
from tb.engine.orders import OrderSubmitter, SubmissionError, SubmissionUnknown
from tb.features.pipeline import FeatureSnapshot
from tb.ledger.events import (
    Actor,
    DecisionPayload,
    EventType,
    LoopCyclePayload,
    PositionOrphanedPayload,
    RiskEvaluationPayload,
    RiskVerdictRow,
)
from tb.ledger.store import Ledger
from tb.ops.state import RunState, StateMachine
from tb.ops.watchdog import InstanceLock, InstanceLockRefused, SelfCheck, WatchdogError
from tb.portfolio.attribution import attribute_closed_trades
from tb.portfolio.pnl import EquityCurve
from tb.registry.lineage import SpecRegistry
from tb.risk.engine import Evaluation, RiskEngine, entry_request, exit_request, stop_price_for
from tb.risk.state import AccountState, RiskContext
from tb.strategy.base import Action, Decision, PositionState

# What one consultation produced: the decision, the intent it submitted, a
# refusal, and the protective stop behind it. A tuple rather than a value object
# because it is assembled and consumed in this module only, and naming it here
# keeps the signatures readable.
_Outcome = tuple[Decision | None, str | None, tuple[str, str] | None, str | None]


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
    # Tickers held by no funded strategy, and what happened to each. Carried on
    # the result rather than only in the ledger because it is the one outcome an
    # operator should see immediately: a position nothing in the book will close
    # is unmanaged exposure, whether or not the flattening order got through.
    unowned: tuple[tuple[str, str], ...] = ()
    # Fills settlement read from the venue's history this cycle.
    fills_recorded: tuple[str, ...] = ()
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

    `book` is the funded set — see `tb.engine.funding`. It is passed in rather
    than read here, so the loop does not have to decide what "funded" means and
    a test can hand it one strategy at a known size.
    """

    ledger: Ledger
    pinned: PinnedLimits
    broker: Broker
    bars: BarSource
    book: Book
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
    # Called at the first cycle of each trading session, before its first
    # decision, with the cycle's instant; returns the book to trade from then
    # on, or `None` to keep the current one. `tb run` passes the per-session
    # portfolio pass and a rebuild of the promoted book, so a loop left running
    # for weeks trades the rungs and allocations of today rather than those of
    # the day it started. A test, or `--strategy trivial`, passes nothing and
    # the book it was given stands.
    on_new_session: Callable[[datetime], Book | None] | None = None
    _cycle: int = field(default=0, init=False)
    _session: date | None = field(default=None, init=False)

    # -- one pass ----------------------------------------------------------

    def run_cycle(self) -> CycleResult:
        """One tick. Raises `LoopHalted` rather than trading through a problem."""
        started = time.monotonic()
        self._cycle += 1
        at = self.clock()

        self._preflight(at)
        fills, settlement_note = self._settle(at)
        self._begin_session(at)
        regime = self._read_regime(at)

        decisions: list[Decision] = []
        submitted: list[str] = []
        refusals: list[tuple[str, str]] = []
        stops: list[str] = []

        # Before anything the book might want: positions the book will never
        # decide about. Risk-reducing, and ahead of the exits a funded strategy
        # might ask for, because an unowned position is the only holding in the
        # account that nothing is managing.
        unowned = self._flatten_unowned(at=at, submitted=submitted, refusals=refusals)
        stood_down = {ticker for ticker, _, _ in unowned}

        # Protection follows the position, not the entry's response. Every
        # cycle, each held position gets a working stop for exactly what is
        # held, and a stop with nothing behind it is withdrawn. On this venue a
        # market order fills asynchronously — the POST answers before the fill
        # — so a stop placed only "when the entry reports filled" would never
        # be placed at all, and nothing else would notice.
        stops.extend(self._maintain_protection(at=at, skip=stood_down, refusals=refusals))

        # Two passes, risk-reducing first. An ordering rather than a queue
        # object: the per-cycle work is small and bounded, and the property
        # that matters is only that no entry is attempted before every exit
        # has been.
        #
        # Instruments outer, strategies inner: the position and its owner are
        # facts about the instrument, so resolving them once per instrument is
        # what keeps two strategies from both acting on one holding.
        acted: set[str] = set()
        for risk_reducing in (True, False):
            for ticker, uid in sorted(self.instruments.items()):
                if ticker in stood_down:
                    # Flattened, or refused a flatten, this cycle. Either way
                    # nothing in the book should open a position on top of it:
                    # a flatten that has not settled would net against the
                    # entry unpredictably, and one that was refused means the
                    # unattributed holding is still there.
                    continue
                if not risk_reducing and ticker in acted:
                    # Exited in the first pass. Once that exit has filled the
                    # instrument is flat, and offering it for entry on the
                    # same bar would buy back what was just sold — a round
                    # trip's costs for no change in view. It is offered again
                    # next cycle, against a new bar.
                    continue
                outcomes = self._consider_instrument(
                    ticker=ticker,
                    uid=uid,
                    at=at,
                    regime=regime,
                    risk_reducing_pass=risk_reducing,
                )
                for decision, intent_id, refusal, stop_id in outcomes:
                    if decision is not None:
                        decisions.append(decision)
                    if intent_id is not None:
                        submitted.append(intent_id)
                        acted.add(ticker)
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
            unowned=tuple((ticker, reason) for ticker, _, reason in unowned),
            fills_recorded=fills,
            duration_ms=duration,
            detail=settlement_note,
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
        gate = RegimeGate(limits=self.pinned.limits)
        return gate.read(self.bars, as_of=at, actions=self._actions_for(gate.instrument_uid))

    def _actions_for(self, uid: str) -> tuple[CorporateAction, ...]:
        """Every corporate action recorded on one instrument, unfiltered.

        The pipeline applies the knowledge-time and effective-date filters
        itself. Read afresh each time rather than held, so a split a backfill
        records while the loop runs reaches the next decision, not the next
        restart. Before this nothing in the loop passed actions at all, so a
        split put a step the size of its ratio into every feature across it —
        a 4-for-1 read as a 75% fall by each strategy, on the day it happened.
        """
        return ActionStore(self.ledger).actions_for(uid)

    def _settle(self, at: datetime) -> tuple[tuple[str, ...], str]:
        """Read what finished since the last cycle, and charge what it realised.

        Before anything is decided, so the protection pass sees what actually
        traded — a stop that fired leaves no position, not an unprotected one —
        and a closed trade reaches its strategy's record, and its lineage's
        budget, in the cycle it settles. Without this every fill stayed
        unrecorded and every strategy showed zero trades forever.

        A venue that cannot be read this cycle is read next cycle: settlement is
        bookkeeping, a missed pass loses nothing, and halting over it would
        also stop the protection pass that follows. Drift in what the venue
        returns is not caught here — that still halts, as everywhere else.
        Attribution runs either way, since it reads only the ledger and picks up
        any fill a previous pass recorded but did not charge.

        Returns the fills recorded and a note when settlement was deferred.
        """
        fills: tuple[str, ...] = ()
        note = ""
        try:
            settlement = self.submitter.settle()
            fills = tuple(fill_id for _, fill_id in settlement.filled)
        except (TransportError, BrokerHttpError, RateLimited) as exc:
            note = f"settlement deferred to the next cycle: {exc}"
        attribute_closed_trades(
            self.ledger,
            registry=SpecRegistry(
                self.ledger,
                per_lineage_budget_ccy=self.pinned.limits.loss.per_lineage_budget_ccy,
                run_id=self.run_id,
            ),
            run_id=self.run_id,
            at=at,
        )
        return fills, note

    def _begin_session(self, at: datetime) -> None:
        """Once a trading session, before its first decision: refresh the book.

        After settlement, so the session's review and rungs see every trade
        closed so far. Only on trading days: a weekend has no session to size.
        """
        if self.on_new_session is None:
            return
        from tb.data.calendar import TradingCalendar

        day = TradingCalendar().day_of(at)
        if not day.is_trading_day or day.day == self._session:
            return
        self._session = day.day
        refreshed = self.on_new_session(at)
        if refreshed is not None:
            self.book = refreshed

    # -- one instrument ----------------------------------------------------

    def _consider_instrument(
        self,
        *,
        ticker: str,
        uid: str,
        at: datetime,
        regime: RegimeReading,
        risk_reducing_pass: bool,
    ) -> list[_Outcome]:
        """Ask whichever funded strategies may act on this instrument.

        Returns one outcome per strategy consulted, and an empty list when the
        instrument is not this pass's business — so the caller can tell
        "considered and declined" from "not considered", which matters for the
        cycle record.
        """
        position = self._position(ticker, uid)

        # The pass filter. An instrument with no position has nothing
        # risk-reducing to do; one with a position is handled on the first
        # pass and skipped on the second, so an add is deferred a cycle rather
        # than competing with its own exit for the budget.
        if risk_reducing_pass != position.is_open:
            return []

        candidates = self._candidates_for(ticker, position=position)
        outcomes: list[_Outcome] = []
        for index, funded in enumerate(candidates):
            decision, intent_id, refusal, stop_id = self._consider(
                funded=funded,
                ticker=ticker,
                uid=uid,
                at=at,
                regime=regime,
                position=position,
            )
            outcomes.append((decision, intent_id, refusal, stop_id))
            if intent_id is None:
                continue
            # An order went out for this instrument. Whoever comes next in book
            # order is refused rather than allowed to add to a position their
            # own allocation did not pay for — and refused explicitly, because a
            # silent skip would look identical to a strategy that simply held.
            for loser in candidates[index + 1 :]:
                outcomes.append(
                    (
                        None,
                        None,
                        (
                            ticker,
                            f"{loser.label}: {funded.label} placed an order on this "
                            "instrument first in this cycle. Two strategies sharing one "
                            "position would each be sized against an allocation that "
                            "paid for part of it.",
                        ),
                        None,
                    )
                )
            break
        return outcomes

    def _candidates_for(self, ticker: str, *, position: PositionState) -> list[FundedStrategy]:
        """Which funded strategies may act on this instrument, in book order.

        A held position is its owner's alone: only the strategy whose entry
        opened it is asked, so two strategies cannot take turns deciding one
        holding, and an add is charged against the allocation that already paid
        for the rest of it. A position with no owner in the book has already
        been flattened this cycle, so there is nobody to ask.

        A flat instrument is offered to the whole book in order, and the first
        strategy to get an order out takes it.
        """
        if not position.is_open:
            return list(self.book.funded)
        owner = owner_of(self.ledger, t212_ticker=ticker)
        if owner is None:
            return []
        return [funded for funded in self.book.funded if funded.key == owner.key]

    def _consider(
        self,
        *,
        funded: FundedStrategy,
        ticker: str,
        uid: str,
        at: datetime,
        regime: RegimeReading,
        position: PositionState,
    ) -> _Outcome:
        """Ask one strategy about one instrument and act on the answer."""
        window = self._window(uid, at)
        snapshot = funded.pipeline.compute(window, uid, actions=self._actions_for(uid))
        decision = funded.strategy.decide(snapshot=snapshot, window=window, position=position)
        decision_id = self._record_decision(
            decision, ticker=ticker, snapshot=snapshot, regime=regime
        )

        if not decision.wants_to_trade:
            return decision, None, None, None
        if decision.action is Action.EXIT and not position.is_open:
            # Nothing to sell. No shipped strategy says EXIT about a flat
            # instrument, but the interface does not forbid it, and an exit
            # sized at zero is not an order: sent on, it raised out of the
            # risk request and took the whole loop down with it.
            return decision, None, (ticker, "exit decided with nothing held"), None

        reference = self._reference_price(window, uid)
        if reference is None:
            return decision, None, (ticker, "no usable reference price in the window"), None

        evaluation = self._evaluate(
            decision=decision,
            decision_id=decision_id,
            funded=funded,
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

        exiting = decision.action is Action.EXIT
        if exiting:
            # After the approval, before the sell. The position's own stop has
            # every share reserved at the venue, so an exit sent past it has
            # nothing to sell; and a stop left working after the exit is a sell
            # order for shares nobody holds, waiting to close whatever position
            # comes next at an old price. Withdrawn only once the exit is
            # approved, so a refused exit never costs the position its stop.
            cleared, why = self._withdraw_protection(
                ticker, at=at, reason=f"making way for {funded.label}'s exit"
            )
            if not cleared:
                return decision, None, (ticker, f"exit deferred: {why}"), None

        try:
            submission = self.submitter.submit(
                evaluation.token,
                order_type=OrderType.MARKET,
                purpose=OrderPurpose.EXIT if exiting else OrderPurpose.ENTRY,
                instrument_uid=uid,
            )
        except SubmissionUnknown as exc:
            # Never retried here. The loop halts and lets recovery — which can
            # read the broker — settle it.
            self._halt(str(exc))
            raise  # pragma: no cover - _halt always raises
        except SubmissionError as exc:
            # Conclusive refusal. Recorded and skipped, not halted: an
            # ordinary rejection must not stop the whole loop. An exit refused
            # after its stop was withdrawn is re-protected now rather than next
            # cycle — the one window this ordering opens is closed in the same
            # breath.
            restored = self._reprotect(ticker, uid, at=at) if exiting else None
            return decision, None, (ticker, f"rejected: {exc}"), restored

        stop_id: str | None = None
        if decision.action is Action.ENTER and submission.filled:
            # Protected in the same cycle when the venue fills synchronously,
            # as the simulator does. When it does not, the next cycle's
            # protection pass sees the position and places the stop.
            stop_id = self._protect(
                notional_ccy=funded.notional_ccy,
                ticker=ticker,
                uid=uid,
                quantity=evaluation.approved_quantity or Decimal(0),
                entry_price=reference,
                at=at,
                decision_id=decision_id,
                parent_intent_id=submission.intent.intent_id,
            )

        return decision, submission.intent.intent_id, None, stop_id

    # -- positions the book does not own -----------------------------------

    def _flatten_unowned(
        self,
        *,
        at: datetime,
        submitted: list[str],
        refusals: list[tuple[str, str]],
    ) -> list[tuple[str, Ownership | None, str]]:
        """Close positions no funded strategy will ever decide about.

        Two ways to arrive here: the strategy that opened the position has been
        retired, blocked or has run its lineage out of budget, or no entry in the
        intent log can be attributed to a decision at all. Either way nothing in
        the book will produce an exit for it, and an open position that nothing
        is managing — with no bracket order behind it on this venue — is the
        state the whole safety design is about.

        Flattened through the risk engine like any other order, so the refusal
        of a flatten is as recorded as its submission. A refusal is *not* a halt:
        the minimum holding period blocks a young position's exit, and halting
        the loop over a position it will be allowed to close in an hour would
        stop it managing everything else in the meantime.

        Scoped to `instruments` — the loop's own universe — and not to the whole
        account. A holding in something this run cannot even price is not the
        loop's to close: it has no bars for it, so it could not size the order
        or check the staleness bound, and selling an instrument it knows nothing
        about is a worse answer than reporting it. `tb reconcile` is what reads
        the account as a whole.
        """
        if not self.instruments:
            return []
        # One `get_positions` rather than a `get_position` per instrument. The
        # per-ticker endpoint is a network call each, so at 25 symbols this
        # sweep would double the portfolio calls every cycle makes against an
        # endpoint the rate governor is already rationing.
        held = [
            position.ticker
            for position in self.broker.get_positions()
            if position.quantity > 0 and position.ticker in self.instruments
        ]
        unowned = unowned_positions(self.ledger, book=self.book, held=held)
        for ticker, owner, reason in unowned:
            uid = self.instruments[ticker]
            state = self._position(ticker, uid)
            intent_id, action = self._flatten(
                ticker=ticker,
                uid=uid,
                position=state,
                at=at,
                reason=reason,
            )
            self._record_orphan(
                ticker=ticker,
                uid=uid,
                position=state,
                owner=owner,
                reason=reason,
                action_taken=action,
                intent_id=intent_id,
            )
            if intent_id is not None:
                submitted.append(intent_id)
            else:
                refusals.append((ticker, f"unowned position not flattened: {action}"))
        return unowned

    def _flatten(
        self,
        *,
        ticker: str,
        uid: str,
        position: PositionState,
        at: datetime,
        reason: str,
    ) -> tuple[str | None, str]:
        """Submit a flattening order. Returns `(intent_id, what happened)`."""
        window = self._window(uid, at)
        reference = self._reference_price(window, uid)
        if reference is None:
            return None, "no usable reference price in the window, so no order could be sized"

        request = exit_request(
            t212_ticker=ticker,
            instrument_uid=uid,
            reference_price=reference,
            quantity=position.quantity,
            purpose=OrderPurpose.FLATTEN,
        )
        evaluation = self.risk.evaluate(
            self._context(
                request=request,
                at=at,
                position=position,
                regime=None,
                ticker=ticker,
                # No allocation: a flatten belongs to no strategy, which is the
                # whole reason it is happening. The allocation rule reads a
                # risk-reducing order as not applicable, so this is explicit
                # rather than load-bearing.
                notional_ccy=None,
            ),
            run_id=self.run_id,
        )
        if evaluation.token is None:
            return None, f"risk refused the flatten: {evaluation.refusal_summary}"

        # The same ordering as an exit: approved first, then the stop withdrawn
        # so the shares are free to sell, then the sell.
        cleared, why = self._withdraw_protection(
            ticker, at=at, reason="making way for flattening an unowned position"
        )
        if not cleared:
            return None, f"flatten deferred: {why}"

        try:
            submission = self.submitter.submit(
                evaluation.token,
                order_type=OrderType.MARKET,
                purpose=OrderPurpose.FLATTEN,
                instrument_uid=uid,
            )
        except SubmissionUnknown as exc:
            self._halt(str(exc))
            raise  # pragma: no cover - _halt always raises
        except SubmissionError as exc:
            self._reprotect(ticker, uid, at=at)
            return None, f"the broker rejected the flatten: {exc}"
        return submission.intent.intent_id, f"flattened {position.quantity} ({reason})"

    def _record_orphan(
        self,
        *,
        ticker: str,
        uid: str,
        position: PositionState,
        owner: Ownership | None,
        reason: str,
        action_taken: str,
        intent_id: str | None,
    ) -> None:
        self.ledger.append(
            EventType.POSITION_ORPHANED,
            ticker,
            PositionOrphanedPayload(
                t212_ticker=ticker,
                instrument_uid=uid,
                quantity=str(position.quantity),
                owner_strategy_id=None if owner is None else owner.strategy_id,
                owner_version=None if owner is None else owner.version,
                owner_intent_id=None if owner is None else owner.intent_id,
                reason=reason,
                action_taken=action_taken,
                intent_id=intent_id,
            ),
            actor=Actor.SYSTEM,
            run_id=self.run_id,
        )

    # -- protection ----------------------------------------------------------

    def _maintain_protection(
        self,
        *,
        at: datetime,
        skip: set[str],
        refusals: list[tuple[str, str]],
    ) -> list[str]:
        """Make every held position's protection match what is actually held.

        Driven by the position at the venue rather than by any response,
        because the responses do not carry the facts: a market order's POST
        answers before it fills, a partial fill changes the size, and a stop can
        be cancelled by hand in the venue's app. Three corrections, per
        instrument in the loop's universe:

        * **held, no stop** — protect it. The case an asynchronous fill leaves.
        * **held, stop for a different quantity** — withdraw and re-protect at
          the size held. A stop for less leaves the difference unprotected; a
          stop for more is a sell for shares nobody holds.
        * **not held, stop working** — withdraw it. A stop left behind by a
          closed position is a sell order waiting to close whatever position
          comes next, at an old price.

        Skipped while a non-protective sell is working for the instrument: an
        exit or flatten is in flight, and re-protecting underneath it would
        reserve the very shares it is selling.

        Returns the protective intents placed.
        """
        placed: list[str] = []
        held = {
            position.ticker: position
            for position in self.broker.get_positions()
            if position.ticker in self.instruments and position.quantity > 0
        }
        open_orders = self.broker.get_open_orders()
        for ticker, uid in sorted(self.instruments.items()):
            if ticker in skip:
                continue
            mine = [order for order in open_orders if order.ticker == ticker]
            if any(order.side is Side.SELL and not order.is_protective for order in mine):
                continue
            stops = [order for order in mine if order.is_protective]
            position = held.get(ticker)
            quantity = position.quantity if position is not None else Decimal(0)
            covered = sum((order.quantity or Decimal(0) for order in stops), Decimal(0))

            if quantity <= 0:
                if stops:
                    cleared, why = self._withdraw_protection(
                        ticker, at=at, reason="the position it protected is no longer held"
                    )
                    if not cleared:
                        refusals.append((ticker, f"stale stop not withdrawn: {why}"))
                continue
            if covered == quantity:
                continue
            if stops:
                cleared, why = self._withdraw_protection(
                    ticker,
                    at=at,
                    reason=f"the stop covers {covered} but {quantity} is held; re-protecting",
                )
                if not cleared:
                    refusals.append((ticker, f"mis-sized stop not withdrawn: {why}"))
                    continue
            stop_id = self._reprotect(ticker, uid, at=at)
            if stop_id is not None:
                placed.append(stop_id)
        return placed

    def _withdraw_protection(self, ticker: str, *, at: datetime, reason: str) -> tuple[bool, str]:
        """Withdraw every working protective stop on an instrument.

        `(True, ...)` only when none is still working afterwards — the caller is
        about to rely on the shares being free. Each withdrawal goes through the
        risk engine like any other write to the venue: a cancel needs a token,
        and the token is the one the engine issues for managing a protective
        stop of that size, so a stop is never touched by a path the engine did
        not approve.
        """
        stops = [
            order
            for order in self.broker.get_open_orders()
            if order.ticker == ticker and order.is_protective
        ]
        if not stops:
            return True, "no protective stop was working"
        uid = self.instruments.get(ticker, "")
        position = self._position(ticker, uid)
        outcomes: list[str] = []
        for order in stops:
            quantity = order.quantity or Decimal(0)
            reference = order.stop_price or self._reference_price(self._window(uid, at), uid)
            if quantity <= 0 or reference is None or reference <= 0:
                return False, (
                    f"stop {order.broker_order_id} carries no usable quantity or price, so "
                    "no withdrawal could be approved for it"
                )
            evaluation = self.risk.evaluate(
                self._context(
                    request=exit_request(
                        t212_ticker=ticker,
                        instrument_uid=uid,
                        reference_price=reference,
                        quantity=quantity,
                        purpose=OrderPurpose.PROTECTIVE_STOP,
                    ),
                    at=at,
                    position=position,
                    regime=None,
                    ticker=ticker,
                    notional_ccy=None,
                ),
                run_id=self.run_id,
            )
            if evaluation.token is None:
                return False, f"risk refused withdrawing the stop: {evaluation.refusal_summary}"
            withdrawal = self.submitter.cancel(
                evaluation.token,
                broker_order_id=order.broker_order_id,
                t212_ticker=ticker,
                reason=reason,
            )
            if not withdrawal.withdrawn:
                return False, (
                    f"stop {order.broker_order_id} is still working: {withdrawal.detail}"
                )
            what = "filled before it could be withdrawn" if withdrawal.filled_first else "withdrawn"
            outcomes.append(f"{order.broker_order_id} {what}")
        if position.is_open:
            self.submitter.record_protection(
                t212_ticker=ticker,
                quantity=position.quantity,
                protected=False,
                detail=f"stop withdrawn: {reason}",
            )
        return True, "; ".join(outcomes)

    def _reprotect(self, ticker: str, uid: str, *, at: datetime) -> str | None:
        """Protect whatever is held now, at the size held now.

        Anchored on the position's average price when the venue reports one —
        the stop's distance is the unprotected-gap assumption the sizing rules
        used, measured from the entry — and on the newest close otherwise. The
        owner's allocation rides along for the record; a protective stop is
        risk-reducing, so no cap is applied to it.
        """
        position = self._position(ticker, uid)
        if not position.is_open:
            return None
        anchor = position.entry_price or self._reference_price(self._window(uid, at), uid)
        if anchor is None or anchor <= 0:
            self.submitter.record_protection(
                t212_ticker=ticker,
                quantity=position.quantity,
                protected=False,
                detail="no entry price and no usable bar to place a stop from",
            )
            return None
        owner = owner_of(self.ledger, t212_ticker=ticker)
        funded = (
            None
            if owner is None
            else next((f for f in self.book.funded if f.key == owner.key), None)
        )
        return self._protect(
            notional_ccy=None if funded is None else funded.notional_ccy,
            ticker=ticker,
            uid=uid,
            quantity=position.quantity,
            entry_price=anchor,
            at=at,
            decision_id=None if owner is None else owner.decision_id,
            parent_intent_id=None if owner is None else owner.intent_id,
        )

    def _protect(
        self,
        *,
        notional_ccy: Decimal | None,
        ticker: str,
        uid: str,
        quantity: Decimal,
        entry_price: Decimal,
        at: datetime,
        decision_id: str | None,
        parent_intent_id: str | None,
    ) -> str | None:
        """Place a protective stop for `quantity`, `unprotected_gap` below `entry_price`.

        Called in the same cycle as a synchronous entry fill, and by the
        protection pass for everything else. The window between a fill and its
        stop is unavoidable on a venue with no bracket orders, but it is
        *bounded* — to one cycle at most. A failure to protect is recorded
        loudly and the position is left held: flattening on a failed stop would
        turn a data problem into a realised loss, and `on_unprotected_position`
        in the limits is where that policy is chosen rather than here.
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
                notional_ccy=notional_ccy,
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
        funded: FundedStrategy,
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
            self._context(
                request=request,
                at=at,
                position=position,
                regime=regime,
                ticker=ticker,
                notional_ccy=funded.notional_ccy,
            ),
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
        notional_ccy: Decimal | None,
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
        since_open, until_close, session_note = self._session_position(at)
        instrument = next((i for i in self.broker.get_instruments() if i.ticker == ticker), None)
        # Read per order, not per cycle. The preflight refuses to start a pass
        # while anything forbids trading, but a pass over a full universe
        # against a rate-limited venue takes minutes — and a kill switch thrown,
        # or `tb halt` run, during it must stop the next entry rather than the
        # next cycle. Every gate the state machine knows is consulted (run
        # state, kill switch, open halts, the limits file), and the halt rule
        # still passes risk-reducing orders, so a halt mid-pass never strands a
        # position whose exit or stop was already on its way.
        permission = self.state.check_trading_permission()
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
            session_note=session_note,
            halted=not permission.allowed,
            halt_reason="; ".join(permission.reasons),
            min_trade_quantity=instrument.min_trade_quantity if instrument else None,
            max_open_quantity=instrument.max_open_quantity if instrument else None,
            # What the ladder and the allocator gave this strategy. Passed
            # through rather than applied here, so the allocation appears as a
            # verdict row beside every other cap — "why is this position small"
            # is then answerable from the same place as "why was this order
            # refused".
            strategy_notional_ccy=notional_ccy,
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

    def _session_position(self, at: datetime) -> tuple[int | None, int | None, str]:
        """Minutes since the open and until the close, or `None`s and why.

        Both `None` when the market is closed or the day is outside the
        calendar's range, and the session-window rule blocks an entry on
        either. That is the right answer: an entry outside a session cannot
        fill until the next open, and one in an unclassifiable day cannot be
        sized against a close that is not known. The third value says which,
        so the refusal names it.

        Derived from the day's bounds rather than from the calendar's own gate,
        because `may_enter_now` answers a *different* question — whether to
        enter at all — and the risk rule needs the two numbers so it can
        record how close to the boundary it was.
        """
        from tb.data.calendar import DayKind, TradingCalendar

        day = TradingCalendar().day_of(at)
        if day.kind is DayKind.UNKNOWN:
            note = (
                f"{day.day} is past the trading calendar's range, so it is treated as "
                "closed; extend the holiday lists or fetch the broker's schedule"
            )
            return None, None, note
        if not day.is_trading_day or day.open_utc is None or day.close_utc is None:
            return None, None, f"the market is closed on {day.day}, a {day.kind.value}"
        if not day.contains(at):
            note = (
                f"the market is closed at {at.isoformat()}; {day.day}'s session runs "
                f"{day.open_utc.isoformat()} to {day.close_utc.isoformat()}"
            )
            return None, None, note
        since = int((at - day.open_utc).total_seconds() // 60)
        until = int((day.close_utc - at).total_seconds() // 60)
        return since, until, ""

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
                n_fills_recorded=len(result.fills_recorded),
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
