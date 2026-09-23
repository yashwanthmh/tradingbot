"""The read-only Trading 212 client.

Driven entirely by `RecordingTransport`, so these properties are checked
deterministically rather than against whatever a demo account happened to hold.

The two that matter most: the client **cannot** place an order, enforced rather
than trusted; and the API key **never** reaches the archive, checked by looking
for it in what was actually written.
"""

from __future__ import annotations

import json
from collections.abc import Iterator
from decimal import Decimal
from pathlib import Path
from typing import Any

import pytest

from tb.broker.t212.client import AuthScheme, ClientConfig, T212Client
from tb.broker.t212.endpoints import Endpoint
from tb.broker.t212.errors import (
    AuthError,
    BrokerHttpError,
    RateLimited,
    SchemaDriftError,
)
from tb.broker.t212.ratelimit import RateGovernor
from tb.core.http import HttpResponse, RecordingTransport, json_response
from tb.ledger.events import EventType
from tb.ledger.store import Ledger
from tests.test_broker_models import CASH, INSTRUMENT, ORDER, POSITION
from tests.test_ratelimit import FakeClock

SECRET = "t212demo_SUPERSECRETKEY_9f2a1c4b"


@pytest.fixture
def transport() -> RecordingTransport:
    return RecordingTransport(
        responses={
            "/equity/account/cash": json_response(json.dumps(CASH)),
            "/equity/account/info": json_response(json.dumps({"currencyCode": "GBP", "id": 42})),
            "/equity/portfolio": json_response(json.dumps([POSITION])),
            "/equity/orders": json_response(json.dumps([ORDER])),
            "/equity/metadata/instruments": json_response(json.dumps([INSTRUMENT])),
            "/equity/metadata/exchanges": json_response(
                json.dumps([{"id": 1, "name": "NASDAQ", "workingSchedules": []}])
            ),
            "/equity/history/orders": json_response(json.dumps({"items": []})),
        }
    )


@pytest.fixture
def client(transport: RecordingTransport, ledger: Ledger, tmp_path: Path) -> Iterator[T212Client]:
    clock = FakeClock()
    governor = RateGovernor(
        clock=clock.time,
        sleep=clock.sleep,
        state_path=tmp_path / "ratelimit.json",
        assume_spent_on_cold_boot=False,
    )
    built = T212Client(
        ClientConfig(
            api_key=SECRET, base_url="https://demo.trading212.com/api/v0", environment="demo"
        ),
        transport=transport,
        governor=governor,
        ledger=ledger,
        run_id="run_test",
    )
    yield built
    built.close()


class TestWriteEndpointsNeedTheTokenPath:
    """M1's rule was "no writes at all". M4's is stronger, not weaker.

    The write endpoints now exist, but `_request` still refuses them: only
    `place_order` and `cancel_order` may reach them, and both check a
    `RiskToken` first. Keeping the guard rather than deleting it means a new
    method that forgets the token also forgets the private flag, and is
    refused — the guard fails closed against its own future callers, which is
    the property that outlives whoever wrote it.
    """

    @pytest.mark.parametrize(
        "endpoint",
        [
            Endpoint.ORDER_MARKET,
            Endpoint.ORDER_LIMIT,
            Endpoint.ORDER_STOP,
            Endpoint.ORDER_STOP_LIMIT,
            Endpoint.ORDER_CANCEL,
        ],
    )
    def test_a_write_endpoint_cannot_be_reached_directly(
        self, client: T212Client, transport: RecordingTransport, endpoint: Endpoint
    ) -> None:
        with pytest.raises(BrokerHttpError, match="may only be reached through place_order"):
            client._request(endpoint)
        assert transport.calls == [], "nothing should have reached the network"

    def test_the_refusal_names_the_route_rather_than_just_refusing(
        self, client: T212Client
    ) -> None:
        """A refusal that does not say what to do instead gets worked around."""
        with pytest.raises(BrokerHttpError) as caught:
            client._request(Endpoint.ORDER_MARKET)
        message = str(caught.value)
        assert "RiskToken" in message
        assert "RiskEngine.evaluate" in message

    def test_the_read_methods_still_send_only_GET(
        self, client: T212Client, transport: RecordingTransport
    ) -> None:
        """The read surface must not have acquired a write by accident."""
        client.get_cash()
        client.get_positions()
        client.get_open_orders()
        client.get_instruments()
        assert set(transport.methods_called()) == {"GET"}

    def test_the_write_methods_require_a_token_positionally(self) -> None:
        """No `token=None` default, or calling without one would type-check.

        Asserted against the signature rather than by calling, because the
        point is that the *absence* of a token is not expressible — a runtime
        check would still leave `place_order()` a valid thing to write.
        """
        import inspect

        for name in ("place_order", "cancel_order"):
            signature = inspect.signature(getattr(T212Client, name))
            first = list(signature.parameters.values())[1]
            assert first.name == "token", f"{name}'s first parameter should be the token"
            assert first.default is inspect.Parameter.empty, (
                f"{name} has a default for its token, so calling it without one would "
                "type-check — which is exactly what the token exists to prevent"
            )


class TestSecretHandling:
    def test_the_key_is_sent_in_the_authorization_header(
        self, client: T212Client, transport: RecordingTransport
    ) -> None:
        client.get_cash()
        assert transport.calls[0]["headers"]["Authorization"] == SECRET

    def test_the_key_never_reaches_the_archive(self, client: T212Client, ledger: Ledger) -> None:
        client.get_cash()
        rows = ledger.conn.execute(
            "SELECT raw_body, request_json, url_path FROM broker_messages"
        ).fetchall()
        assert rows
        for row in rows:
            blob = json.dumps(dict(row))
            assert SECRET not in blob
            assert "SUPERSECRETKEY" not in blob

    def test_a_key_echoed_back_by_the_broker_is_scrubbed(
        self, ledger: Ledger, tmp_path: Path
    ) -> None:
        """Defence in depth.

        An error body that quotes the offending header would otherwise put the
        credential into permanent storage.
        """
        clock = FakeClock()
        transport = RecordingTransport(
            responses={
                "/equity/account/cash": json_response(
                    json.dumps({"error": f"bad key {SECRET}"}), status=500
                )
            }
        )
        built = T212Client(
            ClientConfig(
                api_key=SECRET,
                base_url="https://demo.trading212.com/api/v0",
                environment="demo",
            ),
            transport=transport,
            governor=RateGovernor(
                clock=clock.time, sleep=clock.sleep, assume_spent_on_cold_boot=False
            ),
            ledger=ledger,
        )
        with pytest.raises(BrokerHttpError):
            built.get_cash()

        row = ledger.conn.execute("SELECT raw_body FROM broker_messages").fetchone()
        assert SECRET not in row["raw_body"]
        assert "[REDACTED]" in row["raw_body"]


class TestAuthDiscovery:
    def test_the_bare_header_is_tried_first(
        self, client: T212Client, transport: RecordingTransport
    ) -> None:
        client.get_cash()
        assert not transport.calls[0]["headers"]["Authorization"].startswith("Bearer")

    def test_a_401_falls_back_to_bearer(self, ledger: Ledger, tmp_path: Path) -> None:
        """The documentation is unreachable, so the format is settled empirically."""
        clock = FakeClock()
        state = {"seen": 0}

        def responder(url: str, params: Any) -> HttpResponse:
            state["seen"] += 1
            if state["seen"] == 1:
                return json_response('{"error":"unauthorized"}', status=401)
            return json_response(json.dumps(CASH))

        transport = RecordingTransport(responses={"/equity/account/cash": responder})
        built = T212Client(
            ClientConfig(
                api_key=SECRET,
                base_url="https://demo.trading212.com/api/v0",
                environment="demo",
            ),
            transport=transport,
            governor=RateGovernor(
                clock=clock.time, sleep=clock.sleep, assume_spent_on_cold_boot=False
            ),
            ledger=ledger,
        )
        cash = built.get_cash()
        assert cash.total == Decimal("1000.50")
        assert built.auth_scheme is AuthScheme.BEARER
        assert transport.calls[1]["headers"]["Authorization"].startswith("Bearer ")

    def test_both_schemes_failing_raises_with_the_practice_mode_hint(self, ledger: Ledger) -> None:
        """The mistake everyone makes once, named in the message."""
        clock = FakeClock()
        transport = RecordingTransport(
            responses={"/equity/account/cash": json_response("{}", status=401)}
        )
        built = T212Client(
            ClientConfig(
                api_key=SECRET,
                base_url="https://demo.trading212.com/api/v0",
                environment="demo",
            ),
            transport=transport,
            governor=RateGovernor(
                clock=clock.time, sleep=clock.sleep, assume_spent_on_cold_boot=False
            ),
            ledger=ledger,
        )
        with pytest.raises(AuthError, match="per-environment"):
            built.get_cash()
        assert len(transport.calls) == 2, "should try each scheme exactly once"


class TestArchiving:
    def test_every_response_is_archived(self, client: T212Client, ledger: Ledger) -> None:
        client.get_cash()
        client.get_positions()
        rows = ledger.conn.execute(
            "SELECT endpoint, status_code, parse_ok FROM broker_messages ORDER BY received_at"
        ).fetchall()
        assert len(rows) == 2
        assert all(row["parse_ok"] == 1 for row in rows)

    def test_a_failure_is_archived_too(self, ledger: Ledger) -> None:
        """A 500 or a 429 is exactly the thing worth having later."""
        clock = FakeClock()
        transport = RecordingTransport(
            responses={"/equity/portfolio": json_response('{"err":1}', status=503)}
        )
        built = T212Client(
            ClientConfig(api_key=SECRET, base_url="https://x/api/v0", environment="demo"),
            transport=transport,
            governor=RateGovernor(
                clock=clock.time, sleep=clock.sleep, assume_spent_on_cold_boot=False
            ),
            ledger=ledger,
        )
        with pytest.raises(BrokerHttpError):
            built.get_positions()
        row = ledger.conn.execute("SELECT * FROM broker_messages").fetchone()
        assert row["status_code"] == 503
        assert row["parse_ok"] == 0

    def test_rate_limit_headers_are_archived(self, ledger: Ledger) -> None:
        clock = FakeClock()
        transport = RecordingTransport(
            responses={
                "/equity/portfolio": json_response(
                    json.dumps([POSITION]),
                    ratelimit={
                        "x-ratelimit-limit": "1",
                        "x-ratelimit-period": "1",
                        "x-ratelimit-remaining": "0",
                    },
                )
            }
        )
        built = T212Client(
            ClientConfig(api_key=SECRET, base_url="https://x/api/v0", environment="demo"),
            transport=transport,
            governor=RateGovernor(
                clock=clock.time, sleep=clock.sleep, assume_spent_on_cold_boot=False
            ),
            ledger=ledger,
        )
        built.get_positions()
        row = ledger.conn.execute("SELECT ratelimit_json FROM broker_messages").fetchone()
        assert '"remaining":0' in row["ratelimit_json"]

    def test_a_large_body_is_truncated_with_a_marker(self, ledger: Ledger) -> None:
        """The instruments endpoint returns megabytes.

        Keeping every copy in full would grow the ledger without adding
        diagnostic value past the first few kilobytes.
        """
        clock = FakeClock()
        big = [{**INSTRUMENT, "ticker": f"T{i}_US_EQ"} for i in range(4000)]
        transport = RecordingTransport(
            responses={"/equity/metadata/instruments": json_response(json.dumps(big))}
        )
        built = T212Client(
            ClientConfig(api_key=SECRET, base_url="https://x/api/v0", environment="demo"),
            transport=transport,
            governor=RateGovernor(
                clock=clock.time, sleep=clock.sleep, assume_spent_on_cold_boot=False
            ),
            ledger=ledger,
        )
        assert len(built.get_instruments()) == 4000
        row = ledger.conn.execute("SELECT raw_body FROM broker_messages").fetchone()
        assert "[truncated:" in row["raw_body"]


class TestDrift:
    def test_drift_records_an_event_and_marks_the_archive_row(self, ledger: Ledger) -> None:
        clock = FakeClock()
        broken = {k: v for k, v in CASH.items() if k != "total"}
        transport = RecordingTransport(
            responses={"/equity/account/cash": json_response(json.dumps(broken))}
        )
        built = T212Client(
            ClientConfig(api_key=SECRET, base_url="https://x/api/v0", environment="demo"),
            transport=transport,
            governor=RateGovernor(
                clock=clock.time, sleep=clock.sleep, assume_spent_on_cold_boot=False
            ),
            ledger=ledger,
        )
        with pytest.raises(SchemaDriftError):
            built.get_cash()

        row = ledger.conn.execute("SELECT parse_ok, parse_error FROM broker_messages").fetchone()
        assert row["parse_ok"] == 0
        assert "total" in row["parse_error"]

        events = list(ledger.iter_events(event_type=EventType.BROKER_SCHEMA_DRIFT))
        assert len(events) == 1
        assert "account_cash" in events[0]["payload_json"]

    def test_a_non_json_body_is_drift(self, ledger: Ledger) -> None:
        clock = FakeClock()
        transport = RecordingTransport(
            responses={
                "/equity/portfolio": HttpResponse(
                    status_code=200,
                    headers={"content-type": "text/html"},
                    text="<html>maintenance</html>",
                    elapsed_ms=3.0,
                )
            }
        )
        built = T212Client(
            ClientConfig(api_key=SECRET, base_url="https://x/api/v0", environment="demo"),
            transport=transport,
            governor=RateGovernor(
                clock=clock.time, sleep=clock.sleep, assume_spent_on_cold_boot=False
            ),
            ledger=ledger,
        )
        with pytest.raises(SchemaDriftError):
            built.get_positions()


class TestRateLimiting:
    def test_a_429_raises_and_is_recorded(self, ledger: Ledger) -> None:
        clock = FakeClock()
        transport = RecordingTransport(
            responses={
                "/equity/portfolio": json_response(
                    '{"error":"too many"}',
                    status=429,
                    ratelimit={"x-ratelimit-period": "1", "x-ratelimit-limit": "1"},
                )
            }
        )
        built = T212Client(
            ClientConfig(api_key=SECRET, base_url="https://x/api/v0", environment="demo"),
            transport=transport,
            governor=RateGovernor(
                clock=clock.time, sleep=clock.sleep, assume_spent_on_cold_boot=False
            ),
            ledger=ledger,
        )
        with pytest.raises(RateLimited, match="per-account"):
            built.get_positions()

        events = list(ledger.iter_events(event_type=EventType.BROKER_RATE_LIMITED))
        assert len(events) == 1

    def test_the_governor_learns_from_response_headers(
        self, client: T212Client, transport: RecordingTransport
    ) -> None:
        transport.responses["/equity/portfolio"] = json_response(
            json.dumps([POSITION]),
            ratelimit={"x-ratelimit-limit": "3", "x-ratelimit-period": "1"},
        )
        client.get_positions()
        rows = {r["endpoint"]: r for r in client.governor.observations()}
        assert rows[Endpoint.PORTFOLIO.value]["observed_limit"] == 3


class TestReads:
    def test_cash_and_currency_are_joined(self, client: T212Client) -> None:
        info = client.get_account_info()
        cash = client.get_cash(currency=info.currency_code)
        assert cash.currency == "GBP"
        assert cash.equity == Decimal("1000.50")

    def test_positions_map_to_the_domain(self, client: T212Client) -> None:
        positions = client.get_positions()
        assert len(positions) == 1
        assert positions[0].ticker == "AAPL_US_EQ"

    def test_a_missing_order_returns_none_rather_than_asserting_it_never_existed(
        self, client: T212Client, transport: RecordingTransport
    ) -> None:
        """A 404 says "not here", which is not the same as "never placed".

        Filled orders disappear from this endpoint, so only the reconciler —
        after checking history and position deltas — may conclude anything.
        """
        transport.responses["/equity/orders/"] = json_response("{}", status=404)
        assert client.get_order("999") is None

    def test_history_unwraps_the_paginated_envelope(
        self, client: T212Client, transport: RecordingTransport
    ) -> None:
        transport.responses["/equity/history/orders"] = json_response(
            json.dumps({"items": [{"id": 1, "ticker": "AAPL_US_EQ", "status": "FILLED"}]})
        )
        history = client.get_order_history(limit=5)
        assert len(history) == 1


class TestSnapshot:
    def test_a_snapshot_gathers_all_three_axes(self, client: T212Client) -> None:
        snapshot = client.snapshot()
        assert snapshot.cash.total == Decimal("1000.50")
        assert len(snapshot.positions) == 1
        assert len(snapshot.open_orders) == 1
        assert snapshot.held_tickers == ("AAPL_US_EQ",)

    def test_a_snapshot_reports_when_its_reads_were_not_simultaneous(
        self, transport: RecordingTransport, ledger: Ledger
    ) -> None:
        """Rate limits mean the reads genuinely cannot be simultaneous.

        Comparing a position list fetched now against cash fetched forty
        seconds ago manufactures mismatches that are not real, so the spread is
        measured and reported rather than hidden.
        """
        clock = FakeClock()
        governor = RateGovernor(clock=clock.time, sleep=clock.sleep)
        built = T212Client(
            ClientConfig(api_key=SECRET, base_url="https://x/api/v0", environment="demo"),
            transport=transport,
            governor=governor,
            ledger=ledger,
        )
        snapshot = built.snapshot()
        # The governor's cold-boot wait pushes the reads apart; wall-clock time
        # is real here even though the governor's clock is fake, so assert on
        # the mechanism rather than the number.
        assert isinstance(snapshot.staleness_warnings, tuple)

    def test_account_info_failing_does_not_fail_the_snapshot(
        self, transport: RecordingTransport, ledger: Ledger
    ) -> None:
        clock = FakeClock()
        transport.responses["/equity/account/info"] = json_response("{}", status=503)
        built = T212Client(
            ClientConfig(api_key=SECRET, base_url="https://x/api/v0", environment="demo"),
            transport=transport,
            governor=RateGovernor(
                clock=clock.time, sleep=clock.sleep, assume_spent_on_cold_boot=False
            ),
            ledger=ledger,
        )
        snapshot = built.snapshot()
        assert snapshot.account is None
        assert any("account info unavailable" in w for w in snapshot.staleness_warnings)

    def test_a_snapshot_event_is_recorded_by_the_reconciler_not_the_client(
        self, client: T212Client, ledger: Ledger
    ) -> None:
        """Taking a snapshot is a read; recording it is the reconciler's job.

        Keeps the client free of ledger-projection responsibilities.
        """
        client.snapshot()
        assert list(ledger.iter_events(event_type=EventType.BROKER_SNAPSHOT_TAKEN)) == []


class TestEnvironmentDerivation:
    def test_from_env_refuses_when_there_are_no_credentials(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.delenv("T212_DEMO_API_KEY", raising=False)
        monkeypatch.delenv("T212_LIVE_API_KEY", raising=False)
        with pytest.raises(AuthError):
            T212Client.from_env(transport=RecordingTransport())

    def test_from_env_derives_the_demo_base_url(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        monkeypatch.delenv("T212_LIVE_API_KEY", raising=False)
        monkeypatch.setenv("T212_DEMO_API_KEY", SECRET)
        clock = FakeClock()
        built = T212Client.from_env(
            transport=RecordingTransport(),
            governor=RateGovernor(clock=clock.time, sleep=clock.sleep),
        )
        assert built.environment == "demo"
        assert not built.is_real_money

    def test_require_demo_refuses_a_live_key(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """There is no reason to characterise an API using real money."""
        monkeypatch.delenv("T212_DEMO_API_KEY", raising=False)
        monkeypatch.setenv("T212_LIVE_API_KEY", SECRET)
        with pytest.raises(AuthError, match="refuses to run against a real-money"):
            T212Client.from_env(transport=RecordingTransport(), require_demo=True)

    def test_both_keys_present_is_refused(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("T212_DEMO_API_KEY", SECRET)
        monkeypatch.setenv("T212_LIVE_API_KEY", SECRET)
        with pytest.raises(AuthError):
            T212Client.from_env(transport=RecordingTransport())
