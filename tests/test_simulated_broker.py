"""The simulated broker, and the write half of the port.

Not tests of a mock. These assert that the simulator reproduces the venue
behaviours that produce distinct bugs — because a simulator that only does the
happy path would let the engine pass every crash drill and still fail on the
first real rejection.

The centrepiece is `test_an_entry_that_cannot_be_protected_leaves_it_naked`.
Trading 212 rejects an order below the instrument minimum, and it rejects the
*protective stop* for the same reason as the entry that preceded it — so an
engine that sizes an entry it cannot protect ends up holding an unhedged
position. That has to be reproducible in a test before anything is built on
top of it.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from decimal import Decimal

import pytest

from tb.broker.port import (
    Broker,
    OrderPurpose,
    OrderStatus,
    OrderType,
    ReadOnlyBroker,
    TimeValidity,
)
from tb.broker.simulated import (
    CrashPoint,
    SimulatedBroker,
    SimulatedBrokerError,
    SimulatedRejection,
    SimulatedTransportFailure,
)
from tb.config.loader import load_hard_limits
from tb.risk.engine import RiskEngine, entry_request, exit_request, stop_price_for
from tb.risk.state import AccountState, RiskContext
from tb.risk.token import RiskToken, RiskTokenError

TICKER = "AAPL_US_EQ"
UID = "isin:US0378331005"
AS_OF = datetime(2026, 4, 1, 14, 30, tzinfo=UTC)


def _at(*, prices: dict[str, Decimal] | None = None, **kwargs: object) -> SimulatedBroker:
    """A simulator whose clock is pinned to the tokens' decision instant.

    Without this every token would be expired: the engine mints at `AS_OF` with
    a ten-second life, and a wall-clock simulator would compare that against
    today. Pinning the clock is what makes a drill reproducible rather than
    dependent on when it ran.
    """
    return SimulatedBroker(
        prices=prices if prices is not None else {TICKER: Decimal("100.00")},
        clock=lambda: AS_OF,
        **kwargs,  # type: ignore[arg-type]
    )


@pytest.fixture
def broker() -> SimulatedBroker:
    return _at()


def _token(
    *,
    purpose: OrderPurpose = OrderPurpose.ENTRY,
    quantity: Decimal | None = None,
    ticker: str = TICKER,
    price: Decimal = Decimal("100"),
    held: Decimal = Decimal(0),
    equity: Decimal = Decimal("10000"),
) -> RiskToken:
    """A real token from the real engine. Never hand-built.

    Using the engine here rather than a fixture token is deliberate: it means
    these tests exercise the actual approved quantity, so a sizing change
    shows up here rather than being papered over by a literal.
    """
    limits = load_hard_limits(None).limits
    account = AccountState(
        equity_ccy=equity,
        free_cash_ccy=equity * Decimal("0.9"),
        deployed_ccy=held * price,
        n_open_positions=1 if held else 0,
        currency="GBP",
        day_pnl_pct=0.0,
        rolling_5d_pnl_pct=0.0,
        drawdown_from_peak_pct=0.0,
    )
    if purpose.is_risk_reducing:
        request = exit_request(
            t212_ticker=ticker,
            instrument_uid=UID,
            reference_price=price,
            quantity=quantity or held,
            purpose=purpose,
            decision_id="dec_1",
        )
    else:
        request = entry_request(
            t212_ticker=ticker,
            instrument_uid=UID,
            reference_price=price,
            expected_edge_bps=Decimal("150"),
            decision_id="dec_1",
        )
    ctx = RiskContext(
        as_of=AS_OF,
        limits=limits,
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
    )
    evaluation = RiskEngine().evaluate(ctx, run_id="run_test")
    assert evaluation.token is not None, evaluation.refusal_summary
    return evaluation.token


# --------------------------------------------------------------------------
# The port contract
# --------------------------------------------------------------------------


def test_the_simulator_satisfies_both_halves_of_the_port(broker: SimulatedBroker) -> None:
    """Structural, so a missing method is caught here rather than in the loop."""
    assert isinstance(broker, ReadOnlyBroker)
    assert isinstance(broker, Broker)


def test_placing_an_order_requires_a_token_that_authorises_it(broker: SimulatedBroker) -> None:
    """The point of the whole risk path, at the last gate before the wire."""
    token = _token()
    # The order it approved.
    broker.place_order(token, order_type=OrderType.MARKET, purpose=OrderPurpose.ENTRY)

    # A different purpose than the token carries.
    with pytest.raises((RiskTokenError, SimulatedBrokerError)):
        broker.place_order(token, order_type=OrderType.MARKET, purpose=OrderPurpose.EXIT)


def test_an_expired_token_cannot_place_an_order() -> None:
    """Checked through the broker, not just on the token.

    A token asserts something about account state at a moment — equity,
    deployed capital, the loss breakers — and that state moves. The injectable
    clock is what makes this testable without sleeping: the broker is pinned
    one second past the token's expiry, which is what a submit path stalled
    behind a rate limiter would look like.
    """
    token = _token()
    late = SimulatedBroker(
        prices={TICKER: Decimal("100.00")},
        clock=lambda: token.expires_at + timedelta(seconds=1),
    )
    with pytest.raises(RiskTokenError, match="expired"):
        late.place_order(token, order_type=OrderType.MARKET, purpose=OrderPurpose.ENTRY)
    assert late.posts == [], "an expired token must not reach the venue at all"


# --------------------------------------------------------------------------
# The venue's refusals
# --------------------------------------------------------------------------


def test_an_order_below_the_instrument_minimum_is_rejected() -> None:
    broker = _at(min_trade_quantity={TICKER: Decimal("5")})
    token = _token()  # the engine sizes ~1 share at 1% of 10,000
    with pytest.raises(SimulatedRejection, match="MinQuantityRequired"):
        broker.place_order(token, order_type=OrderType.MARKET, purpose=OrderPurpose.ENTRY)


def test_an_entry_that_cannot_be_protected_leaves_it_naked() -> None:
    """**The failure this simulator exists to make reproducible.**

    The entry clears the minimum; the protective stop is for the same quantity
    and is rejected for the same reason only if the minimum sits between them —
    so the case that matters is the one where the *entry* succeeds and the
    *stop* does not. Here the stop is refused by the pending-order ceiling
    rather than the minimum, which reaches the same state by the other route
    the venue offers.

    The position is now held with nothing behind it. Nobody gets told unless
    something checks, which is why the reconciler's unprotected-position
    finding exists.
    """
    broker = _at()
    entry = _token()
    placed = broker.place_order(entry, order_type=OrderType.MARKET, purpose=OrderPurpose.ENTRY)
    assert placed.status is OrderStatus.FILLED

    position = broker.get_position(TICKER)
    assert position is not None and position.quantity > 0
    assert broker.protective_orders_for(TICKER) == (), "no stop yet — the window is open"

    # Fill the ticker's pending-order budget, then try to protect.
    broker.MAX_PENDING_PER_TICKER = 0
    stop = _token(purpose=OrderPurpose.PROTECTIVE_STOP, held=position.quantity)
    with pytest.raises(SimulatedRejection) as caught:
        broker.place_order(
            stop,
            order_type=OrderType.STOP,
            purpose=OrderPurpose.PROTECTIVE_STOP,
            stop_price=stop_price_for(
                entry_price=Decimal("100.00"), limits=load_hard_limits(None).limits
            ),
            time_validity=TimeValidity.GOOD_TILL_CANCEL,
        )
    assert "PROTECTIVE STOP" in str(caught.value), (
        "the rejection must say the position is now unhedged — a generic "
        "'too many orders' reads as harmless"
    )

    # The state that matters: held, and unprotected.
    still_held = broker.get_position(TICKER)
    assert still_held is not None and still_held.quantity > 0
    assert broker.protective_orders_for(TICKER) == ()


def test_a_stop_order_without_a_level_is_rejected(broker: SimulatedBroker) -> None:
    token = _token(purpose=OrderPurpose.PROTECTIVE_STOP, held=Decimal("1"))
    with pytest.raises(SimulatedRejection, match="StopPriceRequired"):
        broker.place_order(token, order_type=OrderType.STOP, purpose=OrderPurpose.PROTECTIVE_STOP)


def test_a_market_order_with_a_limit_price_is_rejected(broker: SimulatedBroker) -> None:
    """There is no bracket call, so a price attached to a market order is a bug."""
    with pytest.raises(SimulatedRejection, match="UnexpectedLimitPrice"):
        broker.place_order(
            _token(),
            order_type=OrderType.MARKET,
            purpose=OrderPurpose.ENTRY,
            limit_price=Decimal("99"),
        )


def test_selling_more_than_held_is_rejected_rather_than_going_short(
    broker: SimulatedBroker,
) -> None:
    """The account cannot short, so this must be a refusal, not a negative."""
    broker.seed_position(TICKER, quantity=Decimal("1"), average_price=Decimal("100"))
    token = _token(purpose=OrderPurpose.EXIT, held=Decimal("5"), quantity=Decimal("5"))
    with pytest.raises(SimulatedRejection, match="InsufficientQuantity"):
        broker.place_order(token, order_type=OrderType.MARKET, purpose=OrderPurpose.EXIT)


def test_a_rejection_fires_once_so_a_retry_can_succeed(broker: SimulatedBroker) -> None:
    broker.reject_once[TICKER] = ("InsufficientFunds", "not enough cash")
    token = _token()
    with pytest.raises(SimulatedRejection, match="InsufficientFunds"):
        broker.place_order(token, order_type=OrderType.MARKET, purpose=OrderPurpose.ENTRY)
    placed = broker.place_order(token, order_type=OrderType.MARKET, purpose=OrderPurpose.ENTRY)
    assert placed.status is OrderStatus.FILLED


# --------------------------------------------------------------------------
# currentPrice only on held positions
# --------------------------------------------------------------------------


def test_an_unheld_instrument_reports_no_price(broker: SimulatedBroker) -> None:
    """The venue behaviour behind the whole two-tier verification design.

    `currentPrice` comes only from the portfolio, so a candidate symbol can
    never obtain a broker price — which is why the cross-venue gate cannot
    authorise a first entry and cross-provider agreement has to.
    """
    assert broker.get_position(TICKER) is None

    broker.place_order(_token(), order_type=OrderType.MARKET, purpose=OrderPurpose.ENTRY)
    held = broker.get_position(TICKER)
    assert held is not None
    assert held.current_price == Decimal("100.00"), "a held position does have a price"


# --------------------------------------------------------------------------
# Crash points, and what each leaves behind
# --------------------------------------------------------------------------


def test_a_crash_before_the_send_leaves_nothing(broker: SimulatedBroker) -> None:
    """The one case where absence is conclusive."""
    broker.fail_at = CrashPoint.POST_WAL_PRE_SEND
    with pytest.raises(SimulatedTransportFailure, match="before the send"):
        broker.place_order(_token(), order_type=OrderType.MARKET, purpose=OrderPurpose.ENTRY)

    assert broker.posts == [], "no POST reached the venue"
    assert broker.get_open_orders() == ()
    assert broker.get_position(TICKER) is None
    assert not CrashPoint.POST_WAL_PRE_SEND.order_may_exist


def test_a_crash_after_the_send_leaves_a_live_order(broker: SimulatedBroker) -> None:
    """**The case that makes UNKNOWN necessary.**

    The order exists and the caller never learned its id. Recovery has to find
    it by looking; re-sending would double the position.
    """
    broker.fail_at = CrashPoint.POST_SEND_PRE_RESPONSE
    with pytest.raises(SimulatedTransportFailure, match="exists and the caller never learned"):
        broker.place_order(_token(), order_type=OrderType.MARKET, purpose=OrderPurpose.ENTRY)

    assert len(broker.posts) == 1, "exactly one POST reached the venue"
    assert len(broker.get_open_orders()) == 1, "and it is live"
    assert CrashPoint.POST_SEND_PRE_RESPONSE.order_may_exist


def test_a_crash_after_the_response_leaves_it_filled_and_unrecorded(
    broker: SimulatedBroker,
) -> None:
    broker.fail_at = CrashPoint.POST_RESPONSE_PRE_PERSIST
    with pytest.raises(SimulatedTransportFailure, match="acknowledged at the venue"):
        broker.place_order(_token(), order_type=OrderType.MARKET, purpose=OrderPurpose.ENTRY)

    assert len(broker.posts) == 1
    position = broker.get_position(TICKER)
    assert position is not None and position.quantity > 0, (
        "the fill happened; only our record of it is missing"
    )
    # And it is *not* in the open-orders list, because it filled — which is
    # exactly why recovery cannot conclude anything from that list alone.
    assert broker.get_open_orders() == ()


def test_a_transport_failure_is_not_a_rejection(broker: SimulatedBroker) -> None:
    """The distinction the intent log is built on.

    A rejection is conclusive and a retry is safe. A transport failure is not
    and a retry is a double fill, so the two must not share a type.
    """
    assert not issubclass(SimulatedTransportFailure, SimulatedRejection)
    assert not issubclass(SimulatedRejection, SimulatedTransportFailure)


def test_clearing_the_crash_models_a_restart(broker: SimulatedBroker) -> None:
    broker.fail_at = CrashPoint.POST_SEND_PRE_RESPONSE
    with pytest.raises(SimulatedTransportFailure):
        broker.place_order(_token(), order_type=OrderType.MARKET, purpose=OrderPurpose.ENTRY)
    broker.clear_crash()

    # The restarted process can see what the crashed one left behind.
    live = broker.get_open_orders()
    assert len(live) == 1
    assert live[0].ticker == TICKER


# --------------------------------------------------------------------------
# Exactly one POST
# --------------------------------------------------------------------------


def test_every_post_that_reached_the_venue_is_recorded(broker: SimulatedBroker) -> None:
    """`posts` is the evidence the drills assert on.

    It counts attempts that *reached* the venue, so a crash before the send
    does not increment it and a crash after does — which is the distinction
    the whole matrix is about.
    """
    token = _token()
    broker.place_order(token, order_type=OrderType.MARKET, purpose=OrderPurpose.ENTRY)
    assert len(broker.posts_for_token(token.token_id)) == 1

    broker.fail_at = CrashPoint.POST_WAL_PRE_SEND
    with pytest.raises(SimulatedTransportFailure):
        broker.place_order(token, order_type=OrderType.MARKET, purpose=OrderPurpose.ENTRY)
    assert len(broker.posts_for_token(token.token_id)) == 1, (
        "a crash before the send must not count as a POST"
    )


def test_a_rejected_post_is_still_recorded_with_its_outcome(broker: SimulatedBroker) -> None:
    """A rejection reached the venue too, and counts against a rate budget."""
    broker.reject_once[TICKER] = ("InsufficientFunds", "not enough cash")
    token = _token()
    with pytest.raises(SimulatedRejection):
        broker.place_order(token, order_type=OrderType.MARKET, purpose=OrderPurpose.ENTRY)

    posts = broker.posts_for_token(token.token_id)
    assert len(posts) == 1
    assert posts[0].outcome == "rejected:InsufficientFunds"
    assert posts[0].broker_order_id is None


# --------------------------------------------------------------------------
# Fills and positions
# --------------------------------------------------------------------------


def test_a_second_entry_averages_the_basis_rather_than_replacing_it(
    broker: SimulatedBroker,
) -> None:
    """An average price that jumped to the latest fill misreports every add."""
    broker.seed_position(TICKER, quantity=Decimal("1"), average_price=Decimal("90.00"))
    broker.prices[TICKER] = Decimal("110.00")

    # Equity large enough that the 1% per-position cap permits an add: one
    # share at 100 already *is* the cap on a 10,000 account, so the engine
    # would refuse the second entry — correctly, which is why this raises the
    # account rather than reaching past the rule.
    token = _token(held=Decimal("1"), price=Decimal("110"), equity=Decimal("100000"))
    broker.place_order(token, order_type=OrderType.MARKET, purpose=OrderPurpose.ENTRY)

    position = broker.get_position(TICKER)
    assert position is not None
    assert position.average_price is not None
    assert Decimal("90") < position.average_price < Decimal("110"), (
        f"a weighted average should sit between the two fills, got {position.average_price}"
    )


def test_a_full_exit_removes_the_position(broker: SimulatedBroker) -> None:
    broker.seed_position(TICKER, quantity=Decimal("2"), average_price=Decimal("100"))
    token = _token(purpose=OrderPurpose.EXIT, held=Decimal("2"), quantity=Decimal("2"))
    broker.place_order(token, order_type=OrderType.MARKET, purpose=OrderPurpose.EXIT)
    assert broker.get_position(TICKER) is None


def test_a_limit_order_does_not_fill_on_accept(broker: SimulatedBroker) -> None:
    """A working order that has not traded is the normal state of a limit."""
    placed = broker.place_order(
        _token(),
        order_type=OrderType.LIMIT,
        purpose=OrderPurpose.ENTRY,
        limit_price=Decimal("95.00"),
    )
    assert placed.status is OrderStatus.WORKING
    assert broker.get_position(TICKER) is None
    assert len(broker.get_open_orders()) == 1


# --------------------------------------------------------------------------
# Cancellation
# --------------------------------------------------------------------------


def test_cancelling_a_working_order_closes_it(broker: SimulatedBroker) -> None:
    placed = broker.place_order(
        _token(),
        order_type=OrderType.LIMIT,
        purpose=OrderPurpose.ENTRY,
        limit_price=Decimal("95.00"),
    )
    broker.cancel_order(_token(), broker_order_id=placed.broker_order_id)
    order = broker.get_order(placed.broker_order_id)
    assert order is not None and order.status is OrderStatus.CANCELLED
    assert broker.get_open_orders() == ()


def test_cancelling_an_already_filled_order_is_refused(broker: SimulatedBroker) -> None:
    placed = broker.place_order(_token(), order_type=OrderType.MARKET, purpose=OrderPurpose.ENTRY)
    with pytest.raises(SimulatedRejection, match="OrderNotCancellable"):
        broker.cancel_order(_token(), broker_order_id=placed.broker_order_id)


def test_cancelling_an_unknown_order_is_refused(broker: SimulatedBroker) -> None:
    with pytest.raises(SimulatedRejection, match="OrderNotFound"):
        broker.cancel_order(_token(), broker_order_id="ord_nope")


# --------------------------------------------------------------------------
# The snapshot
# --------------------------------------------------------------------------


def test_the_snapshot_is_internally_consistent(broker: SimulatedBroker) -> None:
    """The three axes have to agree, because reconciliation compares them."""
    broker.place_order(_token(), order_type=OrderType.MARKET, purpose=OrderPurpose.ENTRY)
    snapshot = broker.snapshot()

    assert snapshot.environment == "demo"
    assert TICKER in snapshot.held_tickers
    position = snapshot.position_for(TICKER)
    assert position is not None
    cash = snapshot.cash
    assert cash.invested is not None
    assert cash.invested > 0, "an open position must show as invested capital"


def test_the_simulator_never_reports_a_short_position(broker: SimulatedBroker) -> None:
    """`Position` refuses a negative quantity, so this is structural.

    Asserted anyway: the simulator is what the loop is developed against, and
    a simulator that could produce an impossible state would let the loop
    handle one.
    """
    broker.seed_position(TICKER, quantity=Decimal("1"), average_price=Decimal("100"))
    token = _token(purpose=OrderPurpose.EXIT, held=Decimal("1"), quantity=Decimal("1"))
    broker.place_order(token, order_type=OrderType.MARKET, purpose=OrderPurpose.EXIT)
    assert all(p.quantity >= 0 for p in broker.get_positions())
