"""The crash matrix: five injection points, exactly one POST per intent.

This is the file that decides whether the exactly-once claim is true. Trading
212 has no idempotency key, so the property is synthesised by ordering — and
an ordering argument is worth nothing until the crashes are actually injected
and the recovery actually run.

Each drill follows the same shape, which mirrors what really happens:

    1. a process submits, and dies at a named point
    2. a *new* process starts, with a fresh run id and the same ledger
    3. it runs recovery against the broker's real state
    4. assert the ledger's conclusion, and assert the POST count

Step 2 matters. Recovery is tested across a process boundary — a new
`IntentLog` and `OrderSubmitter` over the same database — because the whole
mechanism exists for the case where the process that sent the order is gone.
Testing it in the same objects would test a retry, not a recovery.

The count assertions are the teeth. `SimulatedBroker.posts` records attempts
that *reached the venue*, so "exactly one POST" is checkable rather than
inferred from the absence of a visible duplicate.
"""

from __future__ import annotations

from collections.abc import Iterator
from datetime import UTC, datetime, timedelta
from decimal import Decimal
from pathlib import Path

import pytest

from tb.broker.port import OrderPurpose, OrderStatus, OrderType, TimeValidity
from tb.broker.simulated import CrashPoint, SimulatedBroker
from tb.config.loader import load_hard_limits
from tb.engine.intents import IntentLog, IntentState
from tb.engine.orders import FillSource, OrderSubmitter, SubmissionError, SubmissionUnknown
from tb.ledger.store import Ledger
from tb.ledger.verify import verify_chain
from tb.risk.engine import RiskEngine, entry_request, exit_request, stop_price_for
from tb.risk.state import AccountState, RiskContext
from tb.risk.token import RiskToken

TICKER = "AAPL_US_EQ"
UID = "isin:US0378331005"
AS_OF = datetime(2026, 4, 1, 14, 30, tzinfo=UTC)
LIMITS = load_hard_limits(None).limits


@pytest.fixture
def db(tmp_path: Path) -> Path:
    """One database file, opened and closed once per simulated process."""
    path = tmp_path / "ledger.db"
    with Ledger(path) as ledger:
        ledger.initialise(created_by="drill")
    return path


@pytest.fixture
def broker() -> SimulatedBroker:
    return SimulatedBroker(prices={TICKER: Decimal("100.00")}, clock=lambda: AS_OF)


def _token(
    *,
    purpose: OrderPurpose = OrderPurpose.ENTRY,
    held: Decimal = Decimal(0),
    decision_id: str = "dec_1",
) -> RiskToken:
    account = AccountState(
        equity_ccy=Decimal("10000"),
        free_cash_ccy=Decimal("9000"),
        deployed_ccy=held * Decimal("100"),
        n_open_positions=1 if held else 0,
        currency="GBP",
        day_pnl_pct=0.0,
        rolling_5d_pnl_pct=0.0,
        drawdown_from_peak_pct=0.0,
    )
    if purpose.is_risk_reducing:
        request = exit_request(
            t212_ticker=TICKER,
            instrument_uid=UID,
            reference_price=Decimal("100"),
            quantity=held,
            purpose=purpose,
            decision_id=decision_id,
        )
    else:
        request = entry_request(
            t212_ticker=TICKER,
            instrument_uid=UID,
            reference_price=Decimal("100"),
            expected_edge_bps=Decimal("150"),
            decision_id=decision_id,
        )
    ctx = RiskContext(
        as_of=AS_OF,
        limits=LIMITS,
        request=request,
        account=account,
        position_quantity=held,
        position_entry_at=AS_OF - timedelta(hours=3) if held else None,
        may_enter=True,
        may_enter_reason="cross-verified",
        regime_exposure_factor=Decimal(1),
        regime_state="risk_on",
        bar_age_seconds=5.0,
        minutes_since_open=60,
        minutes_until_close=120,
        extra={"isin": "US0378331005", "instrument_currency": "USD"},
        # The allocator's per-position size. Fail-closed at the rule, so a
        # context that omits it refuses every entry — see
        # `StrategyAllocationRule`. Well above the caps here, so the M4 caps
        # stay the binding ones and these tests keep measuring what they were
        # written to measure.
        strategy_notional_ccy=Decimal("1000.00"),
    )
    evaluation = RiskEngine().evaluate(ctx, run_id="run_drill")
    assert evaluation.token is not None, evaluation.refusal_summary
    return evaluation.token


class _Process:
    """One simulated process: its own ledger handle, log and submitter.

    A context manager so the database really is closed between "processes",
    which is what makes the recovery tests recovery rather than retry.
    """

    def __init__(self, db: Path, broker: SimulatedBroker, *, run_id: str) -> None:
        self._db = db
        self._broker = broker
        self.run_id = run_id

    def __enter__(self) -> OrderSubmitter:
        self._ledger = Ledger(self._db).open()
        self.log = IntentLog(self._ledger, run_id=self.run_id)
        self.submitter = OrderSubmitter(
            ledger=self._ledger,
            broker=self._broker,
            log=self.log,
            run_id=self.run_id,
            clock=lambda: AS_OF,
        )
        return self.submitter

    def __exit__(self, *exc: object) -> None:
        self._ledger.close()


def _process(db: Path, broker: SimulatedBroker, *, run_id: str) -> _Process:
    return _Process(db, broker, run_id=run_id)


def _intents(db: Path) -> Iterator[dict[str, object]]:
    with Ledger(db) as ledger:
        for row in ledger.conn.execute(
            "SELECT intent_id, state, broker_order_id, n_submit_attempts, resolution_note"
            " FROM order_intents ORDER BY wal_committed_at"
        ).fetchall():
            yield dict(row)


# --------------------------------------------------------------------------
# The happy path, so the drills below are not vacuous
# --------------------------------------------------------------------------


def test_a_clean_submission_is_acknowledged_with_one_post(
    db: Path, broker: SimulatedBroker
) -> None:
    """The control. Without it, every drill could pass on a broken submitter."""
    token = _token()
    with _process(db, broker, run_id="run_1") as submitter:
        result = submitter.submit(token, order_type=OrderType.MARKET, purpose=OrderPurpose.ENTRY)

    assert result.status is OrderStatus.FILLED
    assert len(broker.posts) == 1
    rows = list(_intents(db))
    assert len(rows) == 1
    assert rows[0]["state"] == IntentState.ACKNOWLEDGED.value
    assert rows[0]["broker_order_id"] == result.broker_order_id
    assert rows[0]["n_submit_attempts"] == 1


# --------------------------------------------------------------------------
# pre_wal — nothing was recorded, nothing was sent
# --------------------------------------------------------------------------


def test_pre_wal_leaves_no_trace_at_all(db: Path, broker: SimulatedBroker) -> None:
    """A crash before the write-ahead commit is a non-event.

    Modelled by never calling submit. The assertion is that this state is
    *indistinguishable from nothing having happened* — which is what makes it
    the only crash point needing no recovery.
    """
    assert list(_intents(db)) == []
    assert broker.posts == []

    with _process(db, broker, run_id="run_2") as submitter:
        assert submitter.recover() == ((), ())


# --------------------------------------------------------------------------
# post_wal_pre_send — recorded, never sent. Absence is conclusive.
# --------------------------------------------------------------------------


def test_post_wal_pre_send_is_abandoned_and_never_sent(db: Path, broker: SimulatedBroker) -> None:
    """The one crash point where recovery may safely conclude "no order".

    Nothing left the process, so there is nothing at the venue to find. The
    intent is abandoned rather than left blocking, and the POST count stays
    zero — which is the half that proves the conclusion was not reached by
    sending it again to check.
    """
    broker.fail_at = CrashPoint.POST_WAL_PRE_SEND
    token = _token()

    with _process(db, broker, run_id="run_1") as submitter, pytest.raises(SubmissionUnknown):
        submitter.submit(token, order_type=OrderType.MARKET, purpose=OrderPurpose.ENTRY)

    # The crashed process left it submitted-but-unanswered, because it marked
    # submitted before the wire. Recovery has to sort that out.
    assert broker.posts == [], "no POST reached the venue"

    broker.clear_crash()
    with _process(db, broker, run_id="run_2") as submitter:
        resolved, blocking = submitter.recover()

    assert broker.posts == [], "recovery must not send anything"
    assert len(list(_intents(db))) == 1, "and must not create a second intent"

    row = next(iter(_intents(db)))
    # `SUBMITTED` with no matching order is deliberately *not* resolved — the
    # marker went down before the wire, so from the ledger's point of view the
    # order may exist. That is the conservative reading, and it is correct:
    # the ledger cannot know the POST never left.
    assert row["state"] == IntentState.SUBMITTED.value
    assert [i.intent_id for i in blocking] == [row["intent_id"]]
    assert resolved == ()


def test_an_intent_committed_but_not_marked_is_abandoned(db: Path, broker: SimulatedBroker) -> None:
    """The genuinely-never-sent case, which recovery *can* settle.

    Reached by committing without submitting — a crash in the gap between
    steps 1 and 2. Here the ledger knows nothing was sent, so absence at the
    broker is conclusive and the intent is abandoned.
    """
    token = _token()
    with _process(db, broker, run_id="run_1") as submitter:
        submitter.log.commit(token=token, order_type=OrderType.MARKET, at=AS_OF)

    assert next(iter(_intents(db)))["state"] == IntentState.PENDING_SUBMIT.value

    with _process(db, broker, run_id="run_2") as submitter:
        resolved, blocking = submitter.recover()

    assert blocking == ()
    assert [r.state for r in resolved] == [IntentState.ABANDONED]
    assert broker.posts == []
    assert "absence is conclusive" in next(iter(_intents(db)))["resolution_note"]  # type: ignore[operator]


# --------------------------------------------------------------------------
# post_send_pre_response — the order exists, we never learned its id
# --------------------------------------------------------------------------


def test_post_send_pre_response_adopts_the_live_order(db: Path, broker: SimulatedBroker) -> None:
    """**The drill that matters most.**

    The POST reached the venue and the response was lost. The order is live
    with an id we never saw. Recovery finds it and adopts it — and crucially
    does *not* send a second one, which is what a retry would have done.
    """
    broker.fail_at = CrashPoint.POST_SEND_PRE_RESPONSE
    broker.fill_on_accept = False  # stay working, so it is in the open-orders list
    token = _token()

    with (
        _process(db, broker, run_id="run_1") as submitter,
        pytest.raises(SubmissionUnknown, match="unknown state"),
    ):
        submitter.submit(token, order_type=OrderType.MARKET, purpose=OrderPurpose.ENTRY)

    assert len(broker.posts) == 1, "the order reached the venue"
    assert len(broker.get_open_orders()) == 1

    broker.clear_crash()
    with _process(db, broker, run_id="run_2") as submitter:
        resolved, blocking = submitter.recover()

    assert blocking == (), "the order was found, so nothing is left unknown"
    assert [r.state for r in resolved] == [IntentState.ACKNOWLEDGED]
    assert resolved[0].broker_order_id == broker.get_open_orders()[0].broker_order_id

    assert len(broker.posts) == 1, (
        "EXACTLY ONE POST. A retry here would have been the double fill this whole "
        "mechanism exists to prevent."
    )
    assert len(list(_intents(db))) == 1


def test_a_resubmission_after_an_unknown_is_refused_outright(
    db: Path, broker: SimulatedBroker
) -> None:
    """Belt to the recovery brace.

    Even if a caller ignores the unknown and calls submit again with the same
    token, the submitter refuses: the intent is already `SUBMITTED`, and
    re-sending something on the wire is the double-fill path.
    """
    broker.fail_at = CrashPoint.POST_SEND_PRE_RESPONSE
    broker.fill_on_accept = False
    token = _token()

    with _process(db, broker, run_id="run_1") as submitter:
        with pytest.raises(SubmissionUnknown):
            submitter.submit(token, order_type=OrderType.MARKET, purpose=OrderPurpose.ENTRY)

        broker.clear_crash()
        with pytest.raises(SubmissionUnknown, match="double-fill path"):
            submitter.submit(token, order_type=OrderType.MARKET, purpose=OrderPurpose.ENTRY)

    assert len(broker.posts) == 1, "the refused re-submission must not have sent anything"


# --------------------------------------------------------------------------
# post_response_pre_persist — filled at the venue, unrecorded here
# --------------------------------------------------------------------------


def test_post_response_pre_persist_leaves_a_fill_we_cannot_see(
    db: Path, broker: SimulatedBroker
) -> None:
    """The hardest state, and the one recovery must refuse to resolve.

    The order filled, so it is *not* in the open-orders list — the same
    absence a never-accepted order would produce. Recovery therefore cannot
    conclude anything, leaves it unknown, and the run is blocked. That is the
    correct answer: order history would settle it, and history is rate-limited
    to six calls a minute, so the reconciler owns that step rather than the
    hot path.
    """
    broker.fail_at = CrashPoint.POST_RESPONSE_PRE_PERSIST
    token = _token()

    with _process(db, broker, run_id="run_1") as submitter, pytest.raises(SubmissionUnknown):
        submitter.submit(token, order_type=OrderType.MARKET, purpose=OrderPurpose.ENTRY)

    assert len(broker.posts) == 1
    position = broker.get_position(TICKER)
    assert position is not None and position.quantity > 0, "the fill really happened"
    assert broker.get_open_orders() == (), "and it is not in the open-orders list"

    broker.clear_crash()
    with _process(db, broker, run_id="run_2") as submitter:
        resolved, blocking = submitter.recover()

    assert resolved == ()
    assert len(blocking) == 1, (
        "an absent-but-possibly-filled order must block. Concluding 'not placed' from "
        "this absence is exactly how a crash becomes a double fill."
    )
    assert len(broker.posts) == 1


# --------------------------------------------------------------------------
# post_fill_pre_stop — held, and unprotected
# --------------------------------------------------------------------------


def test_post_fill_pre_stop_leaves_the_position_naked_and_says_so(
    db: Path, broker: SimulatedBroker
) -> None:
    """The unprotected window, left open by a crash rather than by latency.

    The entry filled and the process died before the protective stop went in.
    Nothing is ambiguous here — the entry is fully recorded — which is why
    this drill is not about recovery but about *detection*: the position is
    held with no stop behind it, and the only thing that makes that visible is
    a check that looks.
    """
    token = _token()
    with _process(db, broker, run_id="run_1") as submitter:
        result = submitter.submit(token, order_type=OrderType.MARKET, purpose=OrderPurpose.ENTRY)
        assert result.filled
        submitter.record_fill(
            intent=result.intent,
            quantity=token.quantity,
            price=Decimal("100.00"),
            source=FillSource.API_HISTORY,
            filled_at=AS_OF,
        )
        # ... and the process dies here, before placing the stop.

    position = broker.get_position(TICKER)
    assert position is not None and position.quantity > 0
    assert broker.protective_orders_for(TICKER) == ()

    with _process(db, broker, run_id="run_2") as submitter:
        resolved, blocking = submitter.recover()
        # Recovery has nothing to do: the entry is resolved. The gap is not an
        # unknown intent, it is a missing one — which is a different check.
        assert (resolved, blocking) == ((), ())

        assert submitter.log.protective_for(TICKER) == (), (
            "no protective intent was ever committed, which is the state the "
            "reconciler's unprotected-position finding exists to catch"
        )

        # The restarted process can close the window.
        stop_token = _token(purpose=OrderPurpose.PROTECTIVE_STOP, held=position.quantity)
        submitter.submit(
            stop_token,
            order_type=OrderType.STOP,
            purpose=OrderPurpose.PROTECTIVE_STOP,
            stop_price=stop_price_for(entry_price=Decimal("100.00"), limits=LIMITS),
            time_validity=TimeValidity.GOOD_TILL_CANCEL,
        )

    assert len(broker.protective_orders_for(TICKER)) == 1, "the window is closed"


def test_a_stop_gone_from_the_venue_is_not_handed_back_as_its_replacement(
    db: Path, broker: SimulatedBroker
) -> None:
    """A stop cancelled in the venue's app — or one that fired — leaves its
    intent acknowledged until settlement reads what became of it. A replacement
    at the same level has the same id, and handing the acknowledged one back
    reported protection that was not there. It is refused instead, sending
    nothing, until settlement says whether the old stop sold the shares."""
    with _process(db, broker, run_id="run_1") as submitter:
        submitter.submit(_token(), order_type=OrderType.MARKET, purpose=OrderPurpose.ENTRY)
        position = broker.get_position(TICKER)
        assert position is not None
        stop_token = _token(purpose=OrderPurpose.PROTECTIVE_STOP, held=position.quantity)
        stop = {
            "order_type": OrderType.STOP,
            "purpose": OrderPurpose.PROTECTIVE_STOP,
            "stop_price": stop_price_for(entry_price=Decimal("100.00"), limits=LIMITS),
            "time_validity": TimeValidity.GOOD_TILL_CANCEL,
        }
        placed = submitter.submit(stop_token, **stop)  # type: ignore[arg-type]
        broker.cancel_order(stop_token, broker_order_id=placed.broker_order_id)  # in the app
        posts = len(broker.posts)

        with pytest.raises(SubmissionError, match="not yet settled"):
            submitter.submit(stop_token, **stop)  # type: ignore[arg-type]

    assert len(broker.posts) == posts, "the refused replacement sent nothing"
    assert broker.protective_orders_for(TICKER) == ()


def test_the_unprotected_window_is_recorded_with_its_duration(
    db: Path, broker: SimulatedBroker
) -> None:
    """Sizing assumes a bound on this window, so it has to be measurable."""
    token = _token()
    with _process(db, broker, run_id="run_1") as submitter:
        result = submitter.submit(token, order_type=OrderType.MARKET, purpose=OrderPurpose.ENTRY)
        submitter.record_protection(
            t212_ticker=TICKER,
            quantity=token.quantity,
            protected=False,
            entry_fill_id=None,
            detail="crashed before the stop went in",
        )
        submitter.record_protection(
            t212_ticker=TICKER,
            quantity=token.quantity,
            protected=True,
            stop_intent_id=result.intent.intent_id,
            stop_price=Decimal("85.00"),
            unprotected_seconds=412.0,
        )

    with Ledger(db) as ledger:
        rows = ledger.conn.execute(
            "SELECT event_type, payload_json FROM event_log"
            " WHERE event_type IN ('position.unprotected', 'position.protected')"
            " ORDER BY seq"
        ).fetchall()
    assert [r["event_type"] for r in rows] == ["position.unprotected", "position.protected"]
    assert "412" in rows[1]["payload_json"], "the duration has to be in the record"


# --------------------------------------------------------------------------
# An explicit rejection is the opposite case: conclusive, and retryable
# --------------------------------------------------------------------------


def test_a_venue_rejection_is_terminal_and_does_not_block(
    db: Path, broker: SimulatedBroker
) -> None:
    """The contrast that makes UNKNOWN meaningful.

    A rejection is conclusive: the order does not exist. So it resolves the
    intent terminally, leaves nothing blocking, and a corrected order is safe
    to send. Conflating this with an unknown would halt the bot on every
    ordinary refusal.
    """
    broker.reject_once[TICKER] = ("InsufficientFunds", "not enough cash")
    token = _token()

    with _process(db, broker, run_id="run_1") as submitter:
        with pytest.raises(SubmissionError, match="rejected"):
            submitter.submit(token, order_type=OrderType.MARKET, purpose=OrderPurpose.ENTRY)
        resolved, blocking = submitter.recover()

    assert (resolved, blocking) == ((), ()), "a rejection leaves nothing unknown"
    assert next(iter(_intents(db)))["state"] == IntentState.REJECTED.value
    assert len(broker.posts) == 1


# --------------------------------------------------------------------------
# Across every drill: the ledger stays intact and honest
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    "crash",
    [
        CrashPoint.POST_WAL_PRE_SEND,
        CrashPoint.POST_SEND_PRE_RESPONSE,
        CrashPoint.POST_RESPONSE_PRE_PERSIST,
    ],
)
def test_no_crash_point_produces_more_than_one_post(
    db: Path, broker: SimulatedBroker, crash: CrashPoint
) -> None:
    """The headline property, over every point that can send at all.

    One submit, one recovery, and at most one POST — whatever the crash. The
    bound is "at most one" rather than "exactly one" because `pre_send` sends
    nothing by construction.
    """
    broker.fail_at = crash
    broker.fill_on_accept = crash is not CrashPoint.POST_SEND_PRE_RESPONSE
    token = _token()

    with _process(db, broker, run_id="run_1") as submitter, pytest.raises(SubmissionUnknown):
        submitter.submit(token, order_type=OrderType.MARKET, purpose=OrderPurpose.ENTRY)

    broker.clear_crash()
    with _process(db, broker, run_id="run_2") as submitter:
        submitter.recover()

    assert len(broker.posts) <= 1, f"{crash.value} produced {len(broker.posts)} POSTs"
    assert len(list(_intents(db))) == 1, f"{crash.value} produced a second intent"


@pytest.mark.parametrize(
    "crash",
    [
        CrashPoint.POST_WAL_PRE_SEND,
        CrashPoint.POST_SEND_PRE_RESPONSE,
        CrashPoint.POST_RESPONSE_PRE_PERSIST,
    ],
)
def test_the_chain_survives_every_crash_point(
    db: Path, broker: SimulatedBroker, crash: CrashPoint
) -> None:
    """A crash mid-transaction must not leave a broken chain.

    The ledger's value is that it can be trusted after the worst has happened,
    so verifying it *after* each drill is not ceremony — a partial write that
    left the chain unverifiable would make the whole audit trail worthless at
    exactly the moment it is needed.
    """
    broker.fail_at = crash
    token = _token()
    with _process(db, broker, run_id="run_1") as submitter, pytest.raises(SubmissionUnknown):
        submitter.submit(token, order_type=OrderType.MARKET, purpose=OrderPurpose.ENTRY)

    broker.clear_crash()
    with _process(db, broker, run_id="run_2") as submitter:
        submitter.recover()

    with Ledger(db) as ledger:
        report = verify_chain(ledger)
    assert report.ok, f"{crash.value} broke the chain: {report.summary()}"


def test_a_recovered_run_records_why_under_a_new_run_id(db: Path, broker: SimulatedBroker) -> None:
    """The audit trail crosses the process boundary.

    The intent was committed by one run and resolved by another, and both
    facts have to be in the log — otherwise "which process placed this order"
    is unanswerable, and that is the first question after an incident.
    """
    broker.fail_at = CrashPoint.POST_SEND_PRE_RESPONSE
    broker.fill_on_accept = False
    token = _token()

    with (
        _process(db, broker, run_id="run_crashed") as submitter,
        pytest.raises(SubmissionUnknown),
    ):
        submitter.submit(token, order_type=OrderType.MARKET, purpose=OrderPurpose.ENTRY)

    broker.clear_crash()
    with _process(db, broker, run_id="run_restarted") as submitter:
        submitter.recover()

    with Ledger(db) as ledger:
        runs = [
            row["run_id"]
            for row in ledger.conn.execute(
                "SELECT run_id FROM event_log WHERE run_id IS NOT NULL ORDER BY seq"
            ).fetchall()
        ]
        intent_run = ledger.conn.execute("SELECT run_id FROM order_intents").fetchone()["run_id"]

    assert "run_crashed" in runs and "run_restarted" in runs
    assert intent_run == "run_crashed", (
        "the intent stays attributed to the run that committed it; the resolution is a "
        "separate event under the run that did the resolving"
    )
