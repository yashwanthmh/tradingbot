"""The intent write-ahead log.

One property matters more than the rest: **`UNKNOWN` is never converted to
failed.** Trading 212 has no idempotency key, so exactly-once submission is
synthesised by committing before the wire — and the value of that is entirely
in what happens to an intent whose response was lost. Deciding it "must have
failed" and retrying is how one intention becomes two orders and a double
position.

The tests are organised around the three things that can be true after a
crash: nothing was sent (conclusive, abandon), something was sent and the
order is visible (adopt it), something was sent and nothing is visible
(**unresolvable**, halt). The third is the one that would be tempting to
resolve.
"""

from __future__ import annotations

from collections.abc import Iterator
from datetime import UTC, datetime, timedelta
from decimal import Decimal
from pathlib import Path

import pytest

from tb.broker.port import BrokerOrder, OrderPurpose, OrderStatus, OrderType, Side, TimeValidity
from tb.config.loader import load_hard_limits
from tb.engine.intents import (
    IntentError,
    IntentLog,
    IntentState,
    PriorityClass,
    compute_intent_id,
    priority_for,
)
from tb.ledger.store import Ledger
from tb.ledger.verify import verify_chain
from tb.risk.engine import RiskEngine, entry_request, exit_request
from tb.risk.state import AccountState, RiskContext
from tb.risk.token import RiskToken

TICKER = "AAPL_US_EQ"
UID = "isin:US0378331005"
AS_OF = datetime(2026, 4, 1, 14, 30, tzinfo=UTC)


@pytest.fixture
def ledger(tmp_path: Path) -> Iterator[Ledger]:
    with Ledger(tmp_path / "ledger.db") as opened:
        opened.initialise(created_by="test")
        yield opened


@pytest.fixture
def log(ledger: Ledger) -> IntentLog:
    return IntentLog(ledger, run_id="run_test")


def _token(
    *,
    quantity: Decimal = Decimal("3"),
    side: Side = Side.BUY,
    purpose: OrderPurpose = OrderPurpose.ENTRY,
    decision_id: str | None = "dec_1",
) -> RiskToken:
    """A real token, minted the only legitimate way: through the engine."""
    limits = load_hard_limits(None).limits
    account = AccountState(
        equity_ccy=Decimal("10000"),
        free_cash_ccy=Decimal("9000"),
        deployed_ccy=Decimal("0"),
        n_open_positions=0,
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
            quantity=quantity,
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
        limits=limits,
        request=request,
        account=account,
        position_quantity=quantity if purpose.is_risk_reducing else Decimal(0),
        position_entry_at=AS_OF - timedelta(hours=3) if purpose.is_risk_reducing else None,
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
    evaluation = RiskEngine().evaluate(ctx, run_id="run_test")
    assert evaluation.token is not None, evaluation.refusal_summary
    assert side is evaluation.token.side
    return evaluation.token


def _order(
    *,
    order_id: str = "ord_1",
    quantity: Decimal | None = None,
    ticker: str = TICKER,
    side: Side | None = Side.BUY,
) -> BrokerOrder:
    return BrokerOrder(
        broker_order_id=order_id,
        ticker=ticker,
        side=side,
        order_type=OrderType.MARKET,
        status=OrderStatus.WORKING,
        quantity=quantity,
    )


# --------------------------------------------------------------------------
# Determinism: the same intention computes the same id
# --------------------------------------------------------------------------


def test_the_same_logical_order_computes_the_same_id() -> None:
    """The mechanism. Without it a retry creates a second order."""
    kwargs = {
        "t212_ticker": TICKER,
        "side": Side.BUY,
        "order_type": OrderType.MARKET,
        "purpose": OrderPurpose.ENTRY,
        "quantity": Decimal("3"),
        "decision_id": "dec_1",
    }
    assert compute_intent_id(**kwargs) == compute_intent_id(**kwargs)  # type: ignore[arg-type]


def test_the_run_id_is_not_part_of_the_id() -> None:
    """Asserted through the log, because this is the subtle half.

    A crash-and-restart is a new run. If the run id fed the hash, the recovery
    attempt would compute a fresh id, insert a fresh row, and send a second
    order — defeating the whole mechanism at exactly the moment it exists for.
    """
    assert "run" not in compute_intent_id(
        t212_ticker=TICKER,
        side=Side.BUY,
        order_type=OrderType.MARKET,
        purpose=OrderPurpose.ENTRY,
        quantity=Decimal("3"),
        decision_id="dec_1",
    )


def test_two_decisions_on_one_instrument_are_two_intents() -> None:
    """Determinism must not collapse genuinely distinct orders.

    Two entries in the same name from two decisions are two orders. Without
    `decision_id` in the hash they would collide and the second would be
    silently dropped as a duplicate — a strategy that stopped trading for a
    reason nothing recorded.
    """
    base = {
        "t212_ticker": TICKER,
        "side": Side.BUY,
        "order_type": OrderType.MARKET,
        "purpose": OrderPurpose.ENTRY,
        "quantity": Decimal("3"),
    }
    first = compute_intent_id(**base, decision_id="dec_1")  # type: ignore[arg-type]
    second = compute_intent_id(**base, decision_id="dec_2")  # type: ignore[arg-type]
    assert first != second


def test_a_stop_at_a_new_level_is_a_new_intent() -> None:
    """A trailing stop moves, and each level is a distinct order."""
    base = {
        "t212_ticker": TICKER,
        "side": Side.SELL,
        "order_type": OrderType.STOP,
        "purpose": OrderPurpose.PROTECTIVE_STOP,
        "quantity": Decimal("3"),
        "decision_id": "dec_1",
    }
    low = compute_intent_id(**base, stop_price=Decimal("85.00"))  # type: ignore[arg-type]
    high = compute_intent_id(**base, stop_price=Decimal("90.00"))  # type: ignore[arg-type]
    assert low != high


# --------------------------------------------------------------------------
# Commit before the wire
# --------------------------------------------------------------------------


def test_a_committed_intent_starts_unknown(log: IntentLog) -> None:
    """`PENDING_SUBMIT` is an unknown, not a pending success.

    The row exists before anything is sent, which is what makes the question
    answerable later. It also means the row is, at that instant, a claim that
    an order *might* exist — and that is the correct reading.
    """
    intent = log.commit(token=_token(), order_type=OrderType.MARKET, at=AS_OF)
    assert intent.state is IntentState.PENDING_SUBMIT
    assert intent.is_unknown
    assert intent.n_submit_attempts == 0
    assert intent.broker_order_id is None


def test_re_committing_returns_the_same_row(log: IntentLog) -> None:
    """A re-commit is what a retry looks like, and it is not an error.

    The correct response is the row that already exists. Creating a second
    would be the duplicate this log exists to prevent.
    """
    token = _token()
    first = log.commit(token=token, order_type=OrderType.MARKET, at=AS_OF)
    second = log.commit(token=token, order_type=OrderType.MARKET, at=AS_OF)
    assert first.intent_id == second.intent_id
    assert len(log.for_ticker(TICKER)) == 1


@pytest.mark.parametrize(
    ("purpose", "decision_id"),
    [(OrderPurpose.PROTECTIVE_STOP, "dec_1"), (OrderPurpose.FLATTEN, None)],
)
def test_a_settled_stop_or_flatten_is_placed_again_under_a_new_id(
    log: IntentLog, purpose: OrderPurpose, decision_id: str | None
) -> None:
    """**Not a retry: a replacement.** A stop withdrawn for an exit the venue
    then refuses goes back at the same level, same size, same entry decision;
    every flatten of one size has no decision at all. Each field matches the
    settled predecessor, and handing that predecessor back refused the
    replacement as a duplicate — leaving the position unprotected, or the
    orphan unflattened. A predecessor still live is handed back as before."""
    token = _token(side=Side.SELL, purpose=purpose, decision_id=decision_id)
    order_type = OrderType.STOP if purpose is OrderPurpose.PROTECTIVE_STOP else OrderType.MARKET
    stop = Decimal("85.00") if purpose is OrderPurpose.PROTECTIVE_STOP else None

    def commit() -> str:
        return log.commit(token=token, order_type=order_type, stop_price=stop, at=AS_OF).intent_id

    first = commit()
    assert commit() == first, "a live predecessor is the retry, and is handed back"
    log.resolve(first, state=IntentState.RESOLVED_CANCELLED, resolved_by="test", at=AS_OF)

    second = commit()
    assert second != first
    assert commit() == second
    log.mark_rejected(second, detail="refused", at=AS_OF)

    third = commit()
    assert third not in (first, second)
    assert len(log.for_ticker(TICKER)) == 3


def test_a_settled_entry_is_never_placed_again(log: IntentLog) -> None:
    """The other side of the rule: an entry answers exactly one decision, so
    the same decision committed again after it filled is still the filled
    intent — and the submitter refuses to send it a second time."""
    token = _token()
    first = log.commit(token=token, order_type=OrderType.MARKET, at=AS_OF)
    log.resolve(first.intent_id, state=IntentState.RESOLVED_FILLED, resolved_by="test", at=AS_OF)
    again = log.commit(token=token, order_type=OrderType.MARKET, at=AS_OF)
    assert again.intent_id == first.intent_id
    assert again.state is IntentState.RESOLVED_FILLED


def test_an_intent_cannot_describe_an_unapproved_order(log: IntentLog) -> None:
    """The intent is derived from the token, not from loose parameters.

    So there is no argument through which a caller could ask for a different
    quantity than the engine approved — the signature does not offer one.
    """
    token = _token(quantity=Decimal("3"))
    intent = log.commit(token=token, order_type=OrderType.MARKET, at=AS_OF)
    assert intent.quantity == token.quantity
    assert intent.side is token.side
    assert intent.t212_ticker == token.t212_ticker
    assert intent.risk_token_id == token.token_id


def test_an_expired_token_cannot_commit_an_intent(log: IntentLog) -> None:
    from tb.risk.token import RiskTokenError

    token = _token()
    with pytest.raises(RiskTokenError, match="expired"):
        log.commit(
            token=token,
            order_type=OrderType.MARKET,
            at=token.expires_at + timedelta(seconds=1),
        )


def test_a_stop_order_without_a_level_is_refused(log: IntentLog) -> None:
    """ "A protective stop with no level is not protection."""
    with pytest.raises(IntentError, match="needs a stop price"):
        log.commit(
            token=_token(purpose=OrderPurpose.PROTECTIVE_STOP, side=Side.SELL),
            order_type=OrderType.STOP,
            at=AS_OF,
        )


def test_a_limit_order_without_a_price_is_refused(log: IntentLog) -> None:
    with pytest.raises(IntentError, match="needs a limit price"):
        log.commit(token=_token(), order_type=OrderType.LIMIT, at=AS_OF)


# --------------------------------------------------------------------------
# Exactly one POST per intent
# --------------------------------------------------------------------------


def test_an_intent_on_the_wire_cannot_be_re_sent(log: IntentLog) -> None:
    """The double-fill path, closed.

    Re-sending something already submitted is the whole failure mode. The
    refusal names the alternative, because the caller's instinct will be to
    retry.
    """
    intent = log.commit(token=_token(), order_type=OrderType.MARKET, at=AS_OF)
    log.mark_submitted(intent.intent_id, at=AS_OF)
    with pytest.raises(IntentError, match="already on the wire"):
        log.mark_submitted(intent.intent_id, at=AS_OF)


def test_submission_counts_attempts(log: IntentLog) -> None:
    """The count is the evidence the crash drills assert on."""
    intent = log.commit(token=_token(), order_type=OrderType.MARKET, at=AS_OF)
    submitted = log.mark_submitted(intent.intent_id, at=AS_OF)
    assert submitted.n_submit_attempts == 1
    assert submitted.state is IntentState.SUBMITTED
    assert submitted.is_unknown, "sent-but-unanswered is still unknown"


def test_a_terminal_intent_cannot_be_submitted(log: IntentLog) -> None:
    intent = log.commit(token=_token(), order_type=OrderType.MARKET, at=AS_OF)
    log.mark_rejected(intent.intent_id, detail="insufficient funds", at=AS_OF)
    with pytest.raises(IntentError, match="terminal state"):
        log.mark_submitted(intent.intent_id, at=AS_OF)


# --------------------------------------------------------------------------
# Recovery: the three things that can be true after a crash
# --------------------------------------------------------------------------


def test_an_intent_never_sent_is_abandoned(log: IntentLog) -> None:
    """The one case where absence *is* conclusive.

    Nothing left the process, so nothing can be at the broker.
    """
    intent = log.commit(token=_token(), order_type=OrderType.MARKET, at=AS_OF)
    resolved = log.resolve_unknown(intent, broker_orders=(), at=AS_OF)
    assert resolved.state is IntentState.ABANDONED
    assert not resolved.is_unknown
    assert "absence is conclusive" in resolved.resolution_note


def test_a_sent_intent_with_a_matching_order_is_adopted(log: IntentLog) -> None:
    """The response was lost, not the order."""
    intent = log.commit(token=_token(), order_type=OrderType.MARKET, at=AS_OF)
    log.mark_submitted(intent.intent_id, at=AS_OF)
    sent = log.get(intent.intent_id)
    assert sent is not None

    resolved = log.resolve_unknown(
        sent, broker_orders=(_order(order_id="ord_42", quantity=sent.quantity),), at=AS_OF
    )
    assert resolved.state is IntentState.ACKNOWLEDGED
    assert resolved.broker_order_id == "ord_42"
    assert not resolved.is_unknown


def test_a_sent_intent_with_no_matching_order_stays_unknown(log: IntentLog) -> None:
    """**The test that matters.** Absence proves nothing once it was sent.

    A filled-and-closed order is not in the open-orders list, so "not there"
    is consistent with both "never accepted" and "already filled". Treating it
    as failed and retrying is exactly how a double fill happens, so it stays
    unknown and the caller halts.
    """
    intent = log.commit(token=_token(), order_type=OrderType.MARKET, at=AS_OF)
    log.mark_submitted(intent.intent_id, at=AS_OF)
    sent = log.get(intent.intent_id)
    assert sent is not None

    resolved = log.resolve_unknown(sent, broker_orders=(), at=AS_OF)
    assert resolved.state is IntentState.SUBMITTED, (
        "a sent intent with no visible order must NOT be resolved: a filled-and-closed "
        "order is absent from the open-orders list too, so this is unresolvable from here"
    )
    assert resolved.is_unknown

    blocking = log.blocking_unknowns(broker_orders=())
    assert [i.intent_id for i in blocking] == [intent.intent_id]


def test_two_matching_orders_refuse_to_be_tidied_away(log: IntentLog) -> None:
    """The duplicate this whole log exists to prevent, surfaced not swallowed.

    Cancelling the wrong one of a pair is worse than leaving both visible, so
    it needs a human.
    """
    intent = log.commit(token=_token(), order_type=OrderType.MARKET, at=AS_OF)
    log.mark_submitted(intent.intent_id, at=AS_OF)
    sent = log.get(intent.intent_id)
    assert sent is not None

    with pytest.raises(IntentError, match="matches 2 live broker orders"):
        log.resolve_unknown(
            sent,
            broker_orders=(
                _order(order_id="ord_1", quantity=sent.quantity),
                _order(order_id="ord_2", quantity=sent.quantity),
            ),
            at=AS_OF,
        )


def test_recovery_sees_a_previous_run_s_intents(tmp_path: Path) -> None:
    """Across runs, which is the only case that matters.

    A crashed run's unresolved intents are the *next* run's problem, so the
    query must not be scoped to the current run id.
    """
    with Ledger(tmp_path / "ledger.db") as first:
        first.initialise(created_by="test")
        old = IntentLog(first, run_id="run_crashed")
        intent = old.commit(token=_token(), order_type=OrderType.MARKET, at=AS_OF)
        old.mark_submitted(intent.intent_id, at=AS_OF)

    with Ledger(tmp_path / "ledger.db") as second:
        fresh = IntentLog(second, run_id="run_restarted")
        unknowns = fresh.unknown()
        assert [i.intent_id for i in unknowns] == [intent.intent_id]
        assert unknowns[0].run_id == "run_crashed"


def test_an_order_for_another_ticker_is_not_adopted(log: IntentLog) -> None:
    """A false positive here adopts someone else's order as ours."""
    intent = log.commit(token=_token(), order_type=OrderType.MARKET, at=AS_OF)
    log.mark_submitted(intent.intent_id, at=AS_OF)
    sent = log.get(intent.intent_id)
    assert sent is not None
    assert not sent.matches(_order(ticker="MSFT_US_EQ"))
    assert not sent.matches(_order(side=Side.SELL))


def test_a_broker_order_with_no_quantity_still_matches(log: IntentLog) -> None:
    """The venue omits quantity on some shapes.

    Refusing to match on a field the broker did not send would orphan an order
    that is genuinely ours.
    """
    intent = log.commit(token=_token(), order_type=OrderType.MARKET, at=AS_OF)
    assert intent.matches(_order(quantity=None))


# --------------------------------------------------------------------------
# Rejection is only ever explicit
# --------------------------------------------------------------------------


def test_rejection_is_terminal_and_records_the_broker_s_words(log: IntentLog) -> None:
    """A rejection reason is the only evidence of a venue rule we did not know."""
    intent = log.commit(token=_token(), order_type=OrderType.MARKET, at=AS_OF)
    rejected = log.mark_rejected(
        intent.intent_id,
        detail="below minimum quantity",
        broker_message='{"code":"MinQuantityRequired"}',
        at=AS_OF,
    )
    assert rejected.state is IntentState.REJECTED
    assert rejected.state.is_terminal
    assert not rejected.is_unknown


def test_resolve_refuses_a_non_terminal_state(log: IntentLog) -> None:
    intent = log.commit(token=_token(), order_type=OrderType.MARKET, at=AS_OF)
    with pytest.raises(IntentError, match="not a terminal state"):
        log.resolve(
            intent.intent_id,
            state=IntentState.SUBMITTED,
            resolved_by="test",
            at=AS_OF,
        )


def test_resolution_records_how_it_was_settled(log: IntentLog) -> None:
    """Discovered and received are different facts.

    Collapsing them would make an unacknowledged order that turned out to have
    filled look like a normal one, and the unprotected window it implies would
    go unmeasured.
    """
    intent = log.commit(token=_token(), order_type=OrderType.MARKET, at=AS_OF)
    log.mark_submitted(intent.intent_id, at=AS_OF)
    log.mark_acknowledged(intent.intent_id, broker_order_id="ord_9")
    settled = log.resolve(
        intent.intent_id,
        state=IntentState.RESOLVED_FILLED,
        resolved_by="reconciler_history",
        detail="matched in order history",
        at=AS_OF,
    )
    assert settled.state is IntentState.RESOLVED_FILLED
    assert settled.broker_order_id == "ord_9"


def test_submitting_an_intent_that_was_never_committed_raises(log: IntentLog) -> None:
    """The write-ahead row is the only record the order was ever intended."""
    with pytest.raises(IntentError, match="no intent"):
        log.mark_submitted("int_nonexistent", at=AS_OF)


# --------------------------------------------------------------------------
# Priority
# --------------------------------------------------------------------------


def test_a_protective_stop_outranks_every_other_order() -> None:
    """Closing the unprotected window is the most urgent thing here.

    Ahead of a discretionary exit, which is merely important — because the
    sizing assumptions are only load-bearing during that window.
    """
    assert priority_for(OrderPurpose.PROTECTIVE_STOP) is PriorityClass.PROTECTIVE
    assert priority_for(OrderPurpose.EXIT) is PriorityClass.RISK_REDUCING
    assert priority_for(OrderPurpose.ENTRY) is PriorityClass.RISK_INCREASING

    ranks = [
        priority_for(OrderPurpose.PROTECTIVE_STOP).rank,
        priority_for(OrderPurpose.EXIT).rank,
        priority_for(OrderPurpose.ENTRY).rank,
    ]
    assert ranks == sorted(ranks), "protective must sort ahead of exit, exit ahead of entry"


def test_a_rebalance_does_not_get_the_risk_reducing_reserve() -> None:
    """It may increase a position, so it must not consume the exit reserve."""
    assert priority_for(OrderPurpose.REBALANCE) is PriorityClass.RISK_INCREASING


# --------------------------------------------------------------------------
# Counting, for the anomaly breaker
# --------------------------------------------------------------------------


def test_order_counts_come_from_intents_not_the_broker(log: IntentLog) -> None:
    """An order sent but never answered still counts against a runaway budget.

    Which is exactly the case the broker's order list omits — and the broker's
    list is rate-limited an order of magnitude below the write path anyway.
    """
    log.commit(token=_token(decision_id="dec_1"), order_type=OrderType.MARKET, at=AS_OF)
    second = log.commit(token=_token(decision_id="dec_2"), order_type=OrderType.MARKET, at=AS_OF)
    log.mark_submitted(second.intent_id, at=AS_OF)

    total, for_symbol = log.counts_today(day=AS_OF, t212_ticker=TICKER)
    assert total == 2
    assert for_symbol == 2


def test_protective_lookup_ignores_terminal_stops(log: IntentLog) -> None:
    """A cancelled stop is not protection, and a rejected one never was."""
    stop = log.commit(
        token=_token(purpose=OrderPurpose.PROTECTIVE_STOP, side=Side.SELL),
        order_type=OrderType.STOP,
        stop_price=Decimal("85.00"),
        time_validity=TimeValidity.GOOD_TILL_CANCEL,
        at=AS_OF,
    )
    assert [i.intent_id for i in log.protective_for(TICKER)] == [stop.intent_id]

    log.mark_rejected(stop.intent_id, detail="rejected by venue", at=AS_OF)
    assert log.protective_for(TICKER) == (), "a rejected stop must not count as protection"


# --------------------------------------------------------------------------
# The ledger behind it
# --------------------------------------------------------------------------


def test_every_state_change_appends_an_event(ledger: Ledger, log: IntentLog) -> None:
    """The intents table is a projection, like every other table here.

    A projection row with no chain entry behind it would be a claim about an
    order with nothing backing it.
    """
    intent = log.commit(token=_token(), order_type=OrderType.MARKET, at=AS_OF)
    log.mark_submitted(intent.intent_id, at=AS_OF)
    log.mark_acknowledged(intent.intent_id, broker_order_id="ord_1")

    rows = ledger.conn.execute(
        "SELECT event_type FROM event_log WHERE aggregate_id = ? ORDER BY seq",
        (intent.intent_id,),
    ).fetchall()
    assert [r["event_type"] for r in rows] == [
        "intent.committed",
        "order.submitted",
        "order.acknowledged",
    ]
    assert verify_chain(ledger).ok, "the intent rows and their events must sit on an intact chain"
