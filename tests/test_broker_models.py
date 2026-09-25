"""The schema drift matrix.

One asymmetry under test throughout, and it is the whole parsing policy:

* An **added** field is ignored. Trading 212's API is in beta; taking the bot
  down because the broker added a field would escalate a harmless upstream
  change into an inability to read the portfolio — which means an inability to
  reconcile or to place a protective stop.
* A **consumed** field that goes missing, goes null, or changes type is a halt.
  Continuing means acting on a number nobody sent.

And one rule that deserves its own section: a missing number never becomes
zero. A quantity of zero reads as "flat", which is the most dangerous possible
wrong answer about a position — it invites re-entering something already held,
or leaving a real holding unprotected.
"""

from __future__ import annotations

import json
from decimal import Decimal
from typing import Any

import pytest

from tb.broker.port import OrderStatus, OrderType, Side
from tb.broker.t212.errors import SchemaDriftError
from tb.broker.t212.models import (
    AccountInfoResponse,
    CashResponse,
    HistoricalOrderResponse,
    InstrumentResponse,
    OrderResponse,
    PositionResponse,
    executions_from_history,
    map_status,
    parse_many,
    parse_one,
)

CASH: dict[str, Any] = {
    "free": 850.25,
    "total": 1000.50,
    "invested": 150.25,
    "ppl": 3.75,
    "result": 1.20,
    "blocked": None,
    "pieCash": 0.0,
}

POSITION: dict[str, Any] = {
    "ticker": "AAPL_US_EQ",
    "quantity": 0.5,
    "averagePrice": 150.10,
    "currentPrice": 155.40,
    "ppl": 2.65,
    "fxPpl": -0.12,
    "initialFillDate": "2026-01-15T14:30:00.000+00:00",
    "maxBuy": 10.0,
    "maxSell": 0.5,
    "pieQuantity": 0.0,
}

ORDER: dict[str, Any] = {
    "id": 987654321,
    "ticker": "AAPL_US_EQ",
    "quantity": 1.0,
    "filledQuantity": 0.0,
    "limitPrice": 150.0,
    "stopPrice": None,
    "status": "WORKING",
    "strategy": "QUANTITY",
    "type": "LIMIT",
    "timeValidity": "GTC",
    "creationTime": "2026-01-15T14:30:00.000+00:00",
}

INSTRUMENT: dict[str, Any] = {
    "ticker": "AAPL_US_EQ",
    "type": "STOCK",
    "isin": "US0378331005",
    "currencyCode": "USD",
    "name": "Apple Inc.",
    "shortName": "AAPL",
    "workingScheduleId": 1,
    "minTradeQuantity": 0.1,
    "maxOpenQuantity": 10000.0,
    "addedOn": "2020-01-01T00:00:00.000+00:00",
}


class TestHappyPath:
    def test_cash_parses(self) -> None:
        parsed = parse_one(CashResponse, CASH, endpoint="account_cash")
        assert parsed.free == Decimal("850.25")
        assert parsed.total == Decimal("1000.50")
        assert parsed.to_domain(currency="GBP").equity == Decimal("1000.50")

    def test_position_parses_and_maps_to_the_domain(self) -> None:
        parsed = parse_one(PositionResponse, POSITION, endpoint="portfolio")
        position = parsed.to_domain()
        assert position.ticker == "AAPL_US_EQ"
        assert position.quantity == Decimal("0.5")
        assert position.market_value == Decimal("0.5") * Decimal("155.40")

    def test_order_parses_and_infers_side_from_the_quantity_sign(self) -> None:
        parsed = parse_one(OrderResponse, ORDER, endpoint="orders_list").to_domain()
        assert parsed.side is Side.BUY
        assert parsed.order_type is OrderType.LIMIT
        assert parsed.status is OrderStatus.WORKING
        assert parsed.quantity == Decimal("1.0")

    def test_a_negative_order_quantity_is_a_sell(self) -> None:
        """Trading 212 encodes direction in the sign of the order quantity.

        Note this is the order field only: a negative *position* quantity is
        impossible on an Invest account, and `Position` rejects one.
        """
        parsed = parse_one(
            OrderResponse, {**ORDER, "quantity": -2.0}, endpoint="orders_list"
        ).to_domain()
        assert parsed.side is Side.SELL
        assert parsed.quantity == Decimal("2.0")

    def test_instrument_parses(self) -> None:
        parsed = parse_one(InstrumentResponse, INSTRUMENT, endpoint="instruments").to_domain()
        assert parsed.ticker == "AAPL_US_EQ"
        assert parsed.min_trade_quantity == Decimal("0.1")
        assert parsed.is_equity_or_etf

    def test_a_list_endpoint_parses(self) -> None:
        parsed = parse_many(
            PositionResponse, [POSITION, {**POSITION, "ticker": "MSFT_US_EQ"}], endpoint="portfolio"
        )
        assert [p.ticker for p in parsed] == ["AAPL_US_EQ", "MSFT_US_EQ"]

    def test_an_empty_list_is_valid(self) -> None:
        assert parse_many(PositionResponse, [], endpoint="portfolio") == []


class TestFailOpenOnAdditions:
    """Unknown extra fields must not take the system down."""

    def test_an_added_field_is_ignored(self) -> None:
        body = {**CASH, "someNewFieldT212Added": {"nested": [1, 2, 3]}}
        parsed = parse_one(CashResponse, body, endpoint="account_cash")
        assert parsed.total == Decimal("1000.50")

    def test_many_added_fields_are_ignored(self) -> None:
        body = {**POSITION, **{f"extra{i}": i for i in range(50)}}
        assert parse_one(PositionResponse, body, endpoint="portfolio").quantity == Decimal("0.5")

    def test_an_added_field_on_a_list_element_is_ignored(self) -> None:
        parsed = parse_many(
            PositionResponse, [{**POSITION, "brandNew": True}], endpoint="portfolio"
        )
        assert len(parsed) == 1


class TestFailClosedOnConsumedFields:
    """A field the system uses, changed, is a halt."""

    @pytest.mark.parametrize("field", ["free", "total"])
    def test_a_missing_consumed_cash_field_is_drift(self, field: str) -> None:
        body = {k: v for k, v in CASH.items() if k != field}
        with pytest.raises(SchemaDriftError) as caught:
            parse_one(CashResponse, body, endpoint="account_cash")
        assert field in str(caught.value)

    def test_a_renamed_consumed_field_is_drift(self) -> None:
        body = {**CASH}
        body["totalValue"] = body.pop("total")
        with pytest.raises(SchemaDriftError, match="total"):
            parse_one(CashResponse, body, endpoint="account_cash")

    def test_a_consumed_field_turning_null_is_drift(self) -> None:
        with pytest.raises(SchemaDriftError, match="total"):
            parse_one(CashResponse, {**CASH, "total": None}, endpoint="account_cash")

    @pytest.mark.parametrize("value", ["not-a-number", {"amount": 5}, [1, 2], True])
    def test_a_consumed_field_changing_type_is_drift(self, value: Any) -> None:
        with pytest.raises(SchemaDriftError):
            parse_one(CashResponse, {**CASH, "total": value}, endpoint="account_cash")

    def test_a_missing_ticker_is_drift(self) -> None:
        body = {k: v for k, v in POSITION.items() if k != "ticker"}
        with pytest.raises(SchemaDriftError, match="ticker"):
            parse_one(PositionResponse, body, endpoint="portfolio")

    def test_an_empty_ticker_is_drift(self) -> None:
        with pytest.raises(SchemaDriftError):
            parse_one(PositionResponse, {**POSITION, "ticker": ""}, endpoint="portfolio")

    def test_a_missing_currency_code_is_drift(self) -> None:
        """It is asserted against the limits' currency at startup.

        An absolute ceiling of 500 means something very different against a JPY
        account than a GBP one, so this field is not optional.
        """
        with pytest.raises(SchemaDriftError, match="currency"):
            parse_one(AccountInfoResponse, {"id": 1}, endpoint="account_info")

    def test_the_error_names_the_offending_field(self) -> None:
        """ "3 validation errors" is useless at 09:30."""
        body = {k: v for k, v in CASH.items() if k not in ("free", "total")}
        with pytest.raises(SchemaDriftError) as caught:
            parse_one(CashResponse, body, endpoint="account_cash")
        message = str(caught.value)
        assert "free" in message
        assert "total" in message

    def test_the_error_carries_the_archive_id_for_replay(self) -> None:
        with pytest.raises(SchemaDriftError, match="msg_abc123"):
            parse_one(CashResponse, {}, endpoint="account_cash", msg_id="msg_abc123")

    def test_drift_is_a_halt_not_a_broker_error(self) -> None:
        """Subclassing HaltRequired is the point.

        The correct response to a consumed field changing shape is to stop, not
        to retry or degrade.
        """
        from tb.core.errors import HaltRequired

        with pytest.raises(HaltRequired):
            parse_one(CashResponse, {}, endpoint="account_cash")


class TestMissingNumbersNeverBecomeZero:
    """The single most dangerous coercion available."""

    def test_a_missing_quantity_is_drift_not_zero(self) -> None:
        body = {k: v for k, v in POSITION.items() if k != "quantity"}
        with pytest.raises(SchemaDriftError, match="quantity"):
            parse_one(PositionResponse, body, endpoint="portfolio")

    def test_a_null_quantity_is_drift_not_zero(self) -> None:
        with pytest.raises(SchemaDriftError, match="quantity"):
            parse_one(PositionResponse, {**POSITION, "quantity": None}, endpoint="portfolio")

    def test_an_empty_string_quantity_is_drift_not_zero(self) -> None:
        with pytest.raises(SchemaDriftError, match="quantity"):
            parse_one(PositionResponse, {**POSITION, "quantity": ""}, endpoint="portfolio")

    def test_a_genuine_zero_quantity_is_accepted(self) -> None:
        """A real zero is data; a missing value is not.

        The distinction has to survive, or the check above would be untestable.
        """
        parsed = parse_one(PositionResponse, {**POSITION, "quantity": 0}, endpoint="portfolio")
        assert parsed.quantity == Decimal("0")

    def test_a_negative_position_quantity_is_refused_at_the_domain_boundary(self) -> None:
        """The Invest API cannot hold a short.

        So a negative quantity means the response was misparsed or the account
        is not the one we think it is — either way, not something to trade on.
        """
        parsed = parse_one(PositionResponse, {**POSITION, "quantity": -1.0}, endpoint="portfolio")
        with pytest.raises(ValueError, match="cannot hold a short"):
            parsed.to_domain()


class TestUnknownEnumValues:
    """An unrecognised value becomes UNKNOWN, never the nearest guess."""

    def test_an_unknown_status_becomes_unknown(self) -> None:
        assert map_status("SOME_NEW_T212_STATUS") is OrderStatus.UNKNOWN

    def test_an_unknown_status_does_not_become_filled_or_cancelled(self) -> None:
        """The dangerous mappings, named explicitly.

        A wrong FILLED makes the system believe a position exists that does
        not; a wrong CANCELLED makes it believe an order is gone when it is
        live. UNKNOWN is handled safely everywhere downstream.
        """
        mapped = map_status("PENDING_SOMETHING_NEW")
        assert mapped is not OrderStatus.FILLED
        assert mapped is not OrderStatus.CANCELLED
        assert not mapped.is_terminal

    def test_unknown_is_not_treated_as_terminal(self) -> None:
        assert not OrderStatus.UNKNOWN.is_terminal
        assert not OrderStatus.UNKNOWN.is_open

    def test_an_unknown_status_is_recorded_for_the_probe(self) -> None:
        from tb.broker.t212.models import UNMAPPED_VALUES

        map_status("A_BRAND_NEW_STATUS")
        assert "status=A_BRAND_NEW_STATUS" in UNMAPPED_VALUES

    def test_an_order_with_an_unknown_status_still_parses(self) -> None:
        """One unrecognised enum must not make the order unreadable."""
        parsed = parse_one(
            OrderResponse, {**ORDER, "status": "NEWFANGLED"}, endpoint="orders_list"
        ).to_domain()
        assert parsed.status is OrderStatus.UNKNOWN
        assert parsed.ticker == "AAPL_US_EQ"

    def test_an_unknown_order_type_is_none_not_a_guess(self) -> None:
        parsed = parse_one(
            OrderResponse, {**ORDER, "type": "TRAILING_STOP"}, endpoint="orders_list"
        ).to_domain()
        assert parsed.order_type is None


class TestNumericFidelity:
    def test_a_json_float_does_not_inherit_binary_artefacts(self) -> None:
        """`Decimal(0.1)` is 0.1000000000000000055511151231257827.

        These numbers become position sizes, so the conversion goes through
        `str` and keeps what the broker actually sent.
        """
        parsed = parse_one(PositionResponse, {**POSITION, "quantity": 0.1}, endpoint="portfolio")
        assert parsed.quantity == Decimal("0.1")
        assert str(parsed.quantity) == "0.1"

    def test_a_string_number_is_accepted(self) -> None:
        parsed = parse_one(CashResponse, {**CASH, "total": "1000.50"}, endpoint="account_cash")
        assert parsed.total == Decimal("1000.50")

    def test_high_precision_is_preserved(self) -> None:
        parsed = parse_one(
            PositionResponse, {**POSITION, "quantity": "0.123456789"}, endpoint="portfolio"
        )
        assert parsed.quantity == Decimal("0.123456789")

    @pytest.mark.parametrize("value", ["NaN", "Infinity", "-Infinity"])
    def test_a_non_finite_number_is_drift(self, value: str) -> None:
        with pytest.raises(SchemaDriftError):
            parse_one(CashResponse, {**CASH, "total": value}, endpoint="account_cash")

    def test_a_boolean_where_a_number_belongs_is_drift(self) -> None:
        """`bool` is an `int` subclass, so a careless check lets True become 1."""
        with pytest.raises(SchemaDriftError):
            parse_one(CashResponse, {**CASH, "free": True}, endpoint="account_cash")


class TestShapeConfusion:
    def test_a_list_where_an_object_was_expected_is_drift(self) -> None:
        with pytest.raises(SchemaDriftError, match="expected a JSON object"):
            parse_one(CashResponse, [CASH], endpoint="account_cash")

    def test_an_object_where_a_list_was_expected_is_drift(self) -> None:
        with pytest.raises(SchemaDriftError, match="expected a JSON array"):
            parse_many(PositionResponse, POSITION, endpoint="portfolio")

    def test_null_where_an_object_was_expected_is_drift(self) -> None:
        with pytest.raises(SchemaDriftError):
            parse_one(CashResponse, None, endpoint="account_cash")

    def test_one_bad_element_fails_the_whole_list(self) -> None:
        """A silently dropped position is an invisible holding.

        Which is exactly the state the reconciler exists to make impossible, so
        skipping the bad element would defeat the point.
        """
        bodies = [POSITION, {"ticker": "BROKEN_EQ"}, {**POSITION, "ticker": "MSFT_US_EQ"}]
        with pytest.raises(SchemaDriftError, match="element 1"):
            parse_many(PositionResponse, bodies, endpoint="portfolio")

    def test_the_failing_element_index_is_named(self) -> None:
        bodies = [POSITION, POSITION, "not an object"]
        with pytest.raises(SchemaDriftError, match="element 2"):
            parse_many(PositionResponse, bodies, endpoint="portfolio")


class TestHistory:
    def test_a_historical_order_parses_with_everything_optional(self) -> None:
        """History rows are inspected, not acted on directly.

        So nothing here is required: a partially populated row is still worth
        recording.
        """
        parsed = parse_one(HistoricalOrderResponse, {}, endpoint="history_orders")
        assert parsed.order_id is None

    def test_a_populated_historical_order_parses(self) -> None:
        body = {
            "id": 123,
            "ticker": "AAPL_US_EQ",
            "orderedQuantity": 1.0,
            "filledQuantity": 1.0,
            "fillPrice": 155.25,
            "status": "FILLED",
            "type": "MARKET",
            "dateExecuted": "2026-01-15T14:31:00.000+00:00",
            "taxes": [{"name": "STAMP_DUTY", "quantity": 0.5}],
        }
        parsed = parse_one(HistoricalOrderResponse, body, endpoint="history_orders")
        assert parsed.fill_price == Decimal("155.25")
        assert parsed.taxes[0]["name"] == "STAMP_DUTY"

    @staticmethod
    def _history(*bodies: dict[str, Any]) -> list[HistoricalOrderResponse]:
        return parse_many(HistoricalOrderResponse, list(bodies), endpoint="history_orders")

    def test_history_becomes_one_execution_per_order(self) -> None:
        """What settlement reads. A sell's quantity is signed negative by the
        venue and read unsigned; charges are summed by name and unsigned; the
        executed time keeps its offset."""
        (execution,) = executions_from_history(
            self._history(
                {
                    "id": 7,
                    "ticker": "AAPL_US_EQ",
                    "filledQuantity": -2.0,
                    "fillPrice": 150.0,
                    "status": "FILLED",
                    "dateExecuted": "2026-01-15T14:31:00.000+00:00",
                    "taxes": [
                        {"name": "CURRENCY_CONVERSION_FEE", "quantity": -0.45},
                        {"name": "STAMP_DUTY", "quantity": "unreadable"},
                    ],
                }
            )
        )
        assert execution.broker_order_id == "7"
        assert execution.status is OrderStatus.FILLED
        assert execution.filled and execution.filled_quantity == Decimal("2")
        assert execution.fill_price == Decimal("150")
        assert execution.fees == (("CURRENCY_CONVERSION_FEE", Decimal("0.45")),)
        assert execution.executed_at is not None
        assert execution.executed_at.isoformat() == "2026-01-15T14:31:00+00:00"

    def test_an_order_filled_in_parts_is_combined(self) -> None:
        """Newest first, as the venue returns them; parts combined, the price
        volume-weighted, and an entry with no id skipped."""
        executions = executions_from_history(
            self._history(
                {"id": 9, "ticker": "X", "filledQuantity": 1, "fillPrice": 10, "status": "FILLED"},
                {"id": 8, "ticker": "X", "filledQuantity": 1, "fillPrice": 100, "status": "FILLED"},
                {"id": 8, "ticker": "X", "filledQuantity": 3, "fillPrice": 104, "status": "FILLED"},
                {"ticker": "X", "filledQuantity": 1, "fillPrice": 1, "status": "FILLED"},
            )
        )
        assert [e.broker_order_id for e in executions] == ["9", "8"]
        assert executions[1].filled_quantity == Decimal("4")
        assert executions[1].fill_price == Decimal("103")

    def test_a_fill_the_venue_did_not_price_stays_unpriced(self) -> None:
        """Traded, at a price unknown — not zero, and not an average over the
        parts that happen to carry one."""
        (execution,) = executions_from_history(
            self._history(
                {"id": 3, "ticker": "X", "filledQuantity": 1, "fillPrice": 10, "status": "FILLED"},
                {"id": 3, "ticker": "X", "filledQuantity": 1, "status": "FILLED"},
            )
        )
        assert execution.filled and execution.fill_price is None

    def test_a_cancelled_order_is_finished_with_nothing_traded(self) -> None:
        (execution,) = executions_from_history(
            self._history({"id": 4, "ticker": "X", "filledQuantity": 0, "status": "CANCELLED"})
        )
        assert execution.status.is_terminal and not execution.filled
        assert execution.executed_at is None


class TestRealWorldJson:
    def test_parsing_works_from_raw_json_text(self) -> None:
        """The client decodes text, so the models must survive the round trip."""
        body = json.loads(json.dumps([POSITION, ORDER][0:1]))
        parsed = parse_many(PositionResponse, body, endpoint="portfolio")
        assert parsed[0].ticker == "AAPL_US_EQ"
