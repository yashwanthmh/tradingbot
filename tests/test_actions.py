"""The corporate-action store, its supersession rules, and the broker check.

`test_a_dividend_with_no_broker_credit_is_an_identity_risk` is the most
valuable test in this file and arguably in the data layer. It is the only place
in the system where the two venues can be checked against each other on a fact
neither can fudge: if a provider says a company paid a dividend and Trading 212
credited nothing for a position held through the ex-date, then either the action
data is wrong or the symbol map is pointing at a different company than the one
in the account. The second case is what the symbol map exists to prevent and is
otherwise invisible until a position behaves nothing like the backtest.
"""

from __future__ import annotations

from datetime import UTC, date, datetime, timedelta
from decimal import Decimal
from pathlib import Path

import pytest

from tb.data.actions import (
    DIVIDEND_AMOUNT_TOLERANCE_PCT,
    ActionError,
    ActionStore,
    from_raw,
    make_action_id,
)
from tb.data.adjustments import ActionType, CorporateAction, price_factor
from tb.data.provider import (
    Bar,
    Provenance,
    RawAction,
    Resolution,
    Session,
)
from tb.ledger.events import EventType
from tb.ledger.store import Ledger

AAPL = "isin:US0378331005"
NOW = datetime(2026, 1, 1, tzinfo=UTC)


@pytest.fixture
def store(ledger: Ledger) -> ActionStore:
    return ActionStore(ledger, run_id="run-test")


def split(
    *,
    num: int = 4,
    den: int = 1,
    effective: date = date(2020, 8, 31),
    known: datetime = datetime(2020, 7, 31, tzinfo=UTC),
    provider: str = "alpaca",
    action_id: str | None = None,
) -> CorporateAction:
    ratio = f"{num}/{den}"
    return CorporateAction(
        action_id=action_id
        or make_action_id(
            instrument_uid=AAPL,
            action_type="split",
            effective_date=effective.isoformat(),
            ratio=ratio,
            amount="",
        ),
        instrument_uid=AAPL,
        action_type=ActionType.SPLIT,
        effective_date=effective,
        known_at_utc=known,
        source_provider=provider,
        ratio_num=num,
        ratio_den=den,
    )


def dividend(
    *,
    amount: str = "0.205",
    effective: date = date(2021, 2, 5),
    known: datetime = datetime(2021, 1, 27, tzinfo=UTC),
) -> CorporateAction:
    return CorporateAction(
        action_id=make_action_id(
            instrument_uid=AAPL,
            action_type="cash_dividend",
            effective_date=effective.isoformat(),
            ratio="",
            amount=amount,
        ),
        instrument_uid=AAPL,
        action_type=ActionType.CASH_DIVIDEND,
        effective_date=effective,
        known_at_utc=known,
        source_provider="alpaca",
        gross_amount=Decimal(amount),
        currency="USD",
    )


def daily_bar(*, day: int, close: str, open_: str | None = None) -> Bar:
    bar_open = datetime(2020, 8, 24, tzinfo=UTC) + timedelta(days=day)
    opening = Decimal(open_ if open_ is not None else close)
    closing = Decimal(close)
    return Bar(
        instrument_uid=AAPL,
        resolution=Resolution.DAILY,
        bar_open_utc=bar_open,
        available_at_utc=bar_open + timedelta(days=1),
        ingested_at_utc=datetime(1970, 1, 1, tzinfo=UTC),
        provider="alpaca",
        provenance=Provenance.BACKFILL,
        session=Session.REGULAR,
        open=opening,
        high=max(opening, closing),
        low=min(opening, closing),
        close=closing,
        volume=1_000_000,
    )


# --------------------------------------------------------------------------
# Recording
# --------------------------------------------------------------------------


def test_recording_emits_an_event_and_a_row(store: ActionStore, ledger: Ledger) -> None:
    result = store.record([split()])
    assert result.recorded == 1

    events = list(ledger.iter_events(event_type=EventType.DATA_ACTION_RECORDED))
    assert len(events) == 1
    assert events[0]["aggregate_id"] == AAPL

    (stored,) = store.actions_for(AAPL)
    assert stored.split_ratio.numerator == 4


def test_refetching_the_same_window_is_idempotent(store: ActionStore) -> None:
    """The deterministic action_id is what makes this true.

    A backfill is re-run constantly — after a crash, on a schedule, by hand.
    With a counter-based id every run would duplicate every action, and the
    factor product would apply each split twice.
    """
    first = store.record([split()])
    second = store.record([split()])
    assert (first.recorded, second.recorded) == (1, 0)
    assert second.already_held == 1
    assert len(store.actions_for(AAPL)) == 1


def test_a_restated_ratio_supersedes_without_deleting(store: ActionStore) -> None:
    """Both vintages survive, which is what an as-of query needs.

    The correction is the truth *now*. A backtest run over an instant before the
    correction must still see the ratio that was believed then, or re-running it
    next month gives different numbers from the same `vintage_id`.
    """
    store.record([split(num=2, den=1)])
    result = store.record([split(num=4, den=1, known=datetime(2020, 9, 15, tzinfo=UTC))])
    assert result.recorded == 1
    assert len(result.superseded) == 1

    both = store.actions_for(AAPL)
    assert len(both) == 2

    early = store.actions_for(AAPL, as_of=datetime(2020, 8, 1, tzinfo=UTC))
    assert len(early) == 1
    assert early[0].split_ratio.numerator == 2

    late = store.actions_for(AAPL, as_of=NOW)
    assert len(late) == 1
    assert late[0].split_ratio.numerator == 4


def test_a_stale_vintage_cannot_retire_a_newer_fact(store: ActionStore) -> None:
    """Backfill order must not decide which ratio wins.

    A backfill of an old window runs after a live poll has already recorded the
    corrected ratio. Superseding by arrival order would let the stale value take
    over, and nothing downstream would report it.
    """
    store.record([split(num=4, den=1, known=datetime(2020, 9, 15, tzinfo=UTC))])
    result = store.record([split(num=2, den=1, known=datetime(2020, 7, 31, tzinfo=UTC))])

    assert result.recorded == 0
    assert result.superseded == ()
    assert len(result.rejected) == 1
    assert "later knowledge time" in result.rejected[0]

    live = store.actions_for(AAPL, as_of=NOW)
    assert len(live) == 1
    assert live[0].split_ratio.numerator == 4


def test_a_rejected_supersession_leaves_no_dangling_reference(store: ActionStore) -> None:
    """Two siblings, one newer: nothing may be half-superseded.

    Deciding sibling by sibling would let the loop retire the older row and then
    abort, leaving `superseded_by` pointing at an action_id that was never
    inserted.
    """
    store.record([split(num=2, den=1, known=datetime(2020, 7, 1, tzinfo=UTC))])
    store.record([split(num=3, den=1, known=datetime(2020, 9, 20, tzinfo=UTC))])
    store.record([split(num=5, den=1, known=datetime(2020, 8, 1, tzinfo=UTC))])

    stored = {action.action_id for action in store.actions_for(AAPL)}
    referenced = {
        action.superseded_by
        for action in store.actions_for(AAPL)
        if action.superseded_by is not None
    }
    assert referenced <= stored, f"dangling superseded_by: {referenced - stored}"


def test_an_unknown_action_type_is_refused_rather_than_stored(store: ActionStore) -> None:
    """Silently ignoring it at adjustment time is worse than not having it."""
    with pytest.raises(ActionError, match="unknown action type"):
        from_raw(
            RawAction(
                instrument_uid=AAPL,
                action_type="rights_issue",
                effective_date="2024-01-02",
                known_at_utc=NOW,
                provider="alpaca",
            )
        )


def test_provider_actions_convert_and_record(store: ActionStore) -> None:
    result = store.record_raw(
        [
            RawAction(
                instrument_uid=AAPL,
                action_type="split",
                effective_date="2020-08-31",
                known_at_utc=datetime(2020, 7, 31, tzinfo=UTC),
                provider="alpaca",
                ratio_num=4,
                ratio_den=1,
                declared_date="2020-07-30",
            ),
            RawAction(
                instrument_uid=AAPL,
                action_type="cash_dividend",
                effective_date="2021-02-05",
                known_at_utc=datetime(2021, 1, 27, tzinfo=UTC),
                provider="alpaca",
                gross_amount=Decimal("0.205"),
                currency="USD",
            ),
        ]
    )
    assert result.recorded == 2
    stored = store.actions_for(AAPL)
    by_type = {action.action_type: action for action in stored}
    assert by_type[ActionType.SPLIT].declared_date == date(2020, 7, 30)
    assert by_type[ActionType.CASH_DIVIDEND].gross_amount == Decimal("0.205")


def test_a_dividend_survives_the_round_trip_as_an_exact_decimal(
    store: ActionStore,
) -> None:
    """Stored as text, not REAL.

    A dividend read back as a float and multiplied into a total-return factor
    would make the factor depend on binary rounding, and the same backtest would
    hash differently on a different platform.
    """
    store.record([dividend(amount="0.20500000000001")])
    (stored,) = store.actions_for(AAPL)
    assert stored.gross_amount == Decimal("0.20500000000001")
    assert str(stored.gross_amount) == "0.20500000000001"


def test_the_store_feeds_the_factor_algebra_directly(store: ActionStore) -> None:
    """The two halves meet: rows out of SQL, arithmetic with no I/O."""
    store.record([split()])
    factor = price_factor(store.actions_for(AAPL, as_of=NOW), at=date(2020, 8, 28), as_of=NOW)
    assert factor.value.numerator == 1
    assert factor.value.denominator == 4


# --------------------------------------------------------------------------
# Broker dividend reconciliation
# --------------------------------------------------------------------------


def test_a_matching_broker_credit_reconciles(store: ActionStore, ledger: Ledger) -> None:
    store.record([dividend()])
    (match,) = store.reconcile_dividends(
        AAPL,
        broker_credits=[(date(2021, 2, 11), Decimal("0.174"))],
        held_through=[(date(2021, 1, 1), date(2021, 3, 1))],
    )
    assert match.matched
    assert match.broker_amount == Decimal("0.174")
    assert not match.is_identity_risk

    assert len(list(ledger.iter_events(event_type=EventType.DATA_ACTION_RECONCILED))) == 1


def test_a_dividend_with_no_broker_credit_is_an_identity_risk(store: ActionStore) -> None:
    """The highest-value check in the layer.

    A provider dividend with no broker credit at all, on a position held through
    the ex-date, means the action data is wrong *or* this mapping points at a
    different company than the one held. The second is exactly what the symbol
    map exists to prevent, and nothing else in the system can see it.
    """
    store.record([dividend()])
    (match,) = store.reconcile_dividends(
        AAPL,
        broker_credits=[],
        held_through=[(date(2021, 1, 1), date(2021, 3, 1))],
    )
    assert match.matched is False
    assert match.is_identity_risk
    assert "different company" in match.detail


def test_no_position_means_no_credit_is_expected(store: ActionStore) -> None:
    """Skipped, not failed.

    There is no reason to expect a credit for a stock we did not own, and
    counting those as mismatches would bury the one case that matters under
    noise from the entire universe.
    """
    store.record([dividend()])
    (match,) = store.reconcile_dividends(
        AAPL,
        broker_credits=[],
        held_through=[(date(2022, 1, 1), date(2022, 3, 1))],
    )
    assert match.matched
    assert "no position held" in match.detail


def test_withholding_tax_sized_differences_are_tolerated(store: ActionStore) -> None:
    """US withholding on a UK account is 15%; this check is not a rounding audit."""
    store.record([dividend(amount="1.00")])
    (match,) = store.reconcile_dividends(
        AAPL,
        broker_credits=[(date(2021, 2, 20), Decimal("0.85"))],
        held_through=[(date(2021, 1, 1), date(2021, 3, 1))],
    )
    assert match.matched
    assert "withholding" in match.detail


def test_a_wildly_wrong_amount_is_reported(store: ActionStore) -> None:
    store.record([dividend(amount="1.00")])
    (match,) = store.reconcile_dividends(
        AAPL,
        broker_credits=[(date(2021, 2, 20), Decimal("0.10"))],
        held_through=[(date(2021, 1, 1), date(2021, 3, 1))],
    )
    assert match.matched is False
    # Not an identity risk: cash *did* arrive, so the mapping is not the suspect.
    assert not match.is_identity_risk
    assert str(DIVIDEND_AMOUNT_TOLERANCE_PCT) in match.detail


def test_credits_match_the_nearest_dividend_not_the_first(store: ActionStore) -> None:
    """Quarterly payers, with one credit missing.

    Matching in order would pair each dividend with the wrong quarter from the
    gap onwards, turning one missing credit into four mismatches — and the
    amounts would then disagree wildly, so the report would blame the amounts.
    """
    held = [(date(2021, 1, 1), date(2022, 1, 1))]
    for month, amount in ((2, "0.205"), (5, "0.22"), (8, "0.23"), (11, "0.24")):
        store.record([dividend(amount=amount, effective=date(2021, month, 5))])

    matches = store.reconcile_dividends(
        AAPL,
        broker_credits=[
            (date(2021, 2, 11), Decimal("0.205")),
            # May's credit never arrived.
            (date(2021, 8, 12), Decimal("0.23")),
            (date(2021, 11, 11), Decimal("0.24")),
        ],
        held_through=held,
    )
    by_month = {match.effective_date.month: match for match in matches}
    assert by_month[2].matched
    assert by_month[8].matched
    assert by_month[11].matched
    assert by_month[5].matched is False
    assert by_month[5].is_identity_risk


def test_reconciliation_is_recorded_on_the_row(store: ActionStore, ledger: Ledger) -> None:
    store.record([dividend()])
    store.reconcile_dividends(
        AAPL,
        broker_credits=[(date(2021, 2, 11), Decimal("0.19"))],
        held_through=[(date(2021, 1, 1), date(2021, 3, 1))],
    )
    row = ledger.conn.execute(
        "SELECT reconciled_with_broker, reconcile_note FROM corporate_actions"
    ).fetchone()
    assert row["reconciled_with_broker"] == 1
    assert row["reconcile_note"]


# --------------------------------------------------------------------------
# Residual detection
# --------------------------------------------------------------------------


def test_an_unreported_split_in_the_price_path_is_found(store: ActionStore) -> None:
    """What a vendor back-adjusting its cache looks like from outside."""
    bars = [
        daily_bar(day=0, close="499.23"),
        daily_bar(day=1, close="127.58", open_="127.58"),
        daily_bar(day=2, close="129.04"),
    ]
    (suspicion,) = store.scan_for_unexplained_splits(AAPL, bars)
    assert suspicion.implied_ratio.numerator == 4
    assert suspicion.effective_date == date(2020, 8, 25)


def test_a_recorded_split_produces_no_suspicion(store: ActionStore) -> None:
    store.record([split(effective=date(2020, 8, 25))])
    bars = [
        daily_bar(day=0, close="499.23"),
        daily_bar(day=1, close="127.58", open_="127.58"),
    ]
    assert store.scan_for_unexplained_splits(AAPL, bars, as_of=NOW) == ()


def test_intraday_bars_are_ignored_by_the_scan(store: ActionStore) -> None:
    """Overnight gaps would drown the detector at minute resolution."""
    minute = Bar(
        instrument_uid=AAPL,
        resolution=Resolution.MINUTE,
        bar_open_utc=datetime(2020, 8, 24, 14, 30, tzinfo=UTC),
        available_at_utc=datetime(2020, 8, 24, 14, 31, tzinfo=UTC),
        ingested_at_utc=datetime(1970, 1, 1, tzinfo=UTC),
        provider="alpaca",
        provenance=Provenance.BACKFILL,
        session=Session.REGULAR,
        open=Decimal("499.00"),
        high=Decimal("499.50"),
        low=Decimal("498.50"),
        close=Decimal("499.23"),
        volume=1000,
    )
    assert store.scan_for_unexplained_splits(AAPL, [minute]) == ()


def test_an_inferred_split_is_recorded_as_inferred(store: ActionStore) -> None:
    """Recorded so the gate can see it, flagged so nothing adjusts prices by it.

    Acting on a guessed ratio would size real money off an inference. The
    correct response to an unexplained split is to stop entering that symbol
    until a provider confirms it.
    """
    bars = [
        daily_bar(day=0, close="499.23"),
        daily_bar(day=1, close="127.58", open_="127.58"),
    ]
    (suspicion,) = store.scan_for_unexplained_splits(AAPL, bars)
    result = store.record_suspicion(suspicion)
    assert result.recorded == 1

    (stored,) = store.actions_for(AAPL)
    assert stored.inferred_from_price_jump
    assert stored.source_provider == "residual_detector"


def test_the_store_reports_which_instruments_have_actions(store: ActionStore) -> None:
    assert store.instruments_with_actions() == ()
    store.record([split()])
    assert store.instruments_with_actions() == (AAPL,)


def test_recording_nothing_writes_nothing(store: ActionStore, ledger_path: Path) -> None:
    result = store.record([])
    assert result.recorded == 0
    assert not result.changed
