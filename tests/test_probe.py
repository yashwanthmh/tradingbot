"""The capability probe.

The probe exists because everything in the endpoint table and the response
models is a reconstruction — the official Trading 212 reference is not reachable
from the build environment and the API is in beta. So the probe's own job is to
turn assumptions into findings, and these tests check that it reports honestly
in each case rather than only on the happy path.

The scenarios that matter: a field the venue added (must be listed, not fatal),
a consumed field the venue changed (must be reported as unusable), and a rate
limit that differs from the assumed table (must be named, because otherwise it
becomes a 429 during reconciliation weeks later).
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from tb.broker.t212.client import ClientConfig, T212Client
from tb.broker.t212.endpoints import Endpoint
from tb.broker.t212.probe import (
    cache_instruments,
    cached_instruments,
    estimate_duration_seconds,
    run_probe,
)
from tb.broker.t212.ratelimit import RateGovernor
from tb.core.http import RecordingTransport, json_response
from tb.ledger.events import EventType
from tb.ledger.store import Ledger
from tests.test_broker_models import CASH, INSTRUMENT, ORDER, POSITION
from tests.test_ratelimit import FakeClock

KEY = "t212demo_probekey_0001"


def _client(transport: RecordingTransport, ledger: Ledger, *, cold: bool = False) -> T212Client:
    clock = FakeClock()
    return T212Client(
        ClientConfig(
            api_key=KEY, base_url="https://demo.trading212.com/api/v0", environment="demo"
        ),
        transport=transport,
        governor=RateGovernor(clock=clock.time, sleep=clock.sleep, assume_spent_on_cold_boot=cold),
        ledger=ledger,
    )


def _healthy_transport(**overrides: object) -> RecordingTransport:
    responses: dict[str, object] = {
        "/equity/account/cash": json_response(
            json.dumps(CASH),
            ratelimit={"x-ratelimit-limit": "1", "x-ratelimit-period": "5"},
        ),
        "/equity/account/info": json_response(json.dumps({"currencyCode": "GBP", "id": 42})),
        "/equity/portfolio": json_response(json.dumps([POSITION])),
        "/equity/orders": json_response(json.dumps([ORDER])),
        "/equity/metadata/instruments": json_response(json.dumps([INSTRUMENT])),
        "/equity/metadata/exchanges": json_response(
            json.dumps([{"id": 1, "name": "NASDAQ", "workingSchedules": []}])
        ),
        "/equity/history/orders": json_response(json.dumps({"items": []})),
    }
    responses.update(overrides)
    return RecordingTransport(responses=responses)  # type: ignore[arg-type]


class TestHappyPath:
    def test_a_healthy_account_is_reported_usable(self, ledger: Ledger) -> None:
        report = run_probe(_client(_healthy_transport(), ledger), ledger=ledger)
        assert report.usable
        assert report.succeeded == report.probed
        assert report.drifted == []
        assert report.base_currency == "GBP"

    def test_the_probe_records_what_it_found(self, ledger: Ledger) -> None:
        run_probe(_client(_healthy_transport(), ledger), ledger=ledger)
        rows = ledger.conn.execute(
            "SELECT endpoint, observed_limit, observed_period_s, agrees "
            "FROM endpoint_observations ORDER BY endpoint"
        ).fetchall()
        assert rows
        cash = next(r for r in rows if r["endpoint"] == Endpoint.ACCOUNT_CASH.value)
        assert cash["observed_limit"] == 1
        assert cash["observed_period_s"] == 5
        assert cash["agrees"] == 1

    def test_a_probe_event_is_appended(self, ledger: Ledger) -> None:
        run_probe(_client(_healthy_transport(), ledger), ledger=ledger)
        events = list(ledger.iter_events(event_type=EventType.BROKER_PROBED))
        assert len(events) == 1
        assert '"environment":"demo"' in events[0]["payload_json"]

    def test_skip_slow_omits_the_expensive_calls(self, ledger: Ledger) -> None:
        report = run_probe(_client(_healthy_transport(), ledger), ledger=ledger, skip_slow=True)
        probed = {r.endpoint for r in report.results}
        assert Endpoint.INSTRUMENTS not in probed
        assert Endpoint.ACCOUNT_CASH in probed
        # Still usable: the essential three are not the slow ones.
        assert report.usable

    def test_progress_is_reported_per_endpoint(self, ledger: Ledger) -> None:
        """A probe that looks hung is a probe someone kills.

        The cold-boot wait on a one-call-per-fifty-seconds endpoint is
        legitimate, so the caller gets told which endpoint and how long.
        """
        seen: list[tuple[str, float]] = []
        run_probe(
            _client(_healthy_transport(), ledger, cold=True),
            ledger=ledger,
            skip_slow=True,
            on_progress=lambda endpoint, wait: seen.append((endpoint.value, wait)),
        )
        assert [name for name, _ in seen][:1] == [Endpoint.ACCOUNT_CASH.value]
        assert any(wait > 0 for _, wait in seen)


class TestUnknownFields:
    def test_an_added_field_is_listed_but_not_fatal(self, ledger: Ledger) -> None:
        """Exactly the case the models are built to survive.

        The field is ignored so the bot keeps working, and reported so a useful
        addition does not stay invisible.
        """
        transport = _healthy_transport(
            **{
                "/equity/account/cash": json_response(
                    json.dumps({**CASH, "newlyAddedByT212": 7, "anotherOne": "x"})
                )
            }
        )
        report = run_probe(_client(transport, ledger), ledger=ledger)
        assert report.usable
        extra = report.unknown_fields[Endpoint.ACCOUNT_CASH.value]
        assert "newlyAddedByT212" in extra
        assert "anotherOne" in extra

    def test_declared_fields_are_not_reported_as_unknown(self, ledger: Ledger) -> None:
        report = run_probe(_client(_healthy_transport(), ledger), ledger=ledger)
        assert Endpoint.ACCOUNT_CASH.value not in report.unknown_fields


class TestDrift:
    def test_a_changed_consumed_field_makes_the_account_unusable(self, ledger: Ledger) -> None:
        """The probe's most valuable output.

        Learning that `total` has been renamed is much better here than during
        the first sizing calculation.
        """
        broken = {k: v for k, v in CASH.items() if k != "total"}
        transport = _healthy_transport(
            **{"/equity/account/cash": json_response(json.dumps(broken))}
        )
        report = run_probe(_client(transport, ledger), ledger=ledger)

        assert not report.usable
        drifted = report.drifted
        assert len(drifted) == 1
        assert drifted[0].endpoint is Endpoint.ACCOUNT_CASH
        # Answered, but did not parse — the distinction matters for diagnosis.
        assert drifted[0].ok
        assert not drifted[0].parsed
        assert "total" in drifted[0].detail

    def test_drift_is_still_archived_for_replay(self, ledger: Ledger) -> None:
        broken = {k: v for k, v in POSITION.items() if k != "quantity"}
        transport = _healthy_transport(**{"/equity/portfolio": json_response(json.dumps([broken]))})
        run_probe(_client(transport, ledger), ledger=ledger)
        row = ledger.conn.execute(
            "SELECT raw_body, parse_error FROM broker_messages WHERE parse_ok = 0"
        ).fetchone()
        assert row is not None
        assert "quantity" in row["parse_error"]
        assert "ticker" in row["raw_body"]

    def test_an_unknown_enum_value_is_reported(self, ledger: Ledger) -> None:
        transport = _healthy_transport(
            **{"/equity/orders": json_response(json.dumps([{**ORDER, "status": "ZZZ_NEW"}]))}
        )
        report = run_probe(_client(transport, ledger), ledger=ledger)
        # Not fatal — an unrecognised status maps to UNKNOWN, which is handled
        # safely everywhere downstream.
        assert report.usable
        assert any("ZZZ_NEW" in v for v in report.unmapped_enum_values)


class TestRateLimitDisagreement:
    def test_a_differing_limit_is_named(self, ledger: Ledger) -> None:
        """A limit we assumed wrong is a 429 waiting for the worst moment."""
        transport = _healthy_transport(
            **{
                "/equity/portfolio": json_response(
                    json.dumps([POSITION]),
                    ratelimit={"x-ratelimit-limit": "4", "x-ratelimit-period": "1"},
                )
            }
        )
        report = run_probe(_client(transport, ledger), ledger=ledger)
        assert any("portfolio" in note for note in report.disagreements)
        assert any("configured 1/1s" in note for note in report.disagreements)

        row = ledger.conn.execute(
            "SELECT agrees FROM endpoint_observations WHERE endpoint = ?",
            (Endpoint.PORTFOLIO.value,),
        ).fetchone()
        assert row["agrees"] == 0

    def test_agreement_produces_no_noise(self, ledger: Ledger) -> None:
        report = run_probe(_client(_healthy_transport(), ledger), ledger=ledger)
        assert report.disagreements == []


class TestFailures:
    def test_an_auth_failure_stops_early(self, ledger: Ledger) -> None:
        """Trying more endpoints will not fix a bad key.

        And every attempt spends rate limit the next run needs.
        """
        transport = _healthy_transport(
            **{"/equity/account/cash": json_response('{"error":"nope"}', status=401)}
        )
        report = run_probe(_client(transport, ledger), ledger=ledger)
        assert not report.usable
        assert report.probed == 1
        assert report.results[0].status_code == 401

    def test_a_429_is_recorded_rather_than_raised(self, ledger: Ledger) -> None:
        transport = _healthy_transport(
            **{
                "/equity/portfolio": json_response(
                    '{"error":"slow down"}',
                    status=429,
                    ratelimit={"x-ratelimit-period": "1"},
                )
            }
        )
        report = run_probe(_client(transport, ledger), ledger=ledger)
        portfolio = next(r for r in report.results if r.endpoint is Endpoint.PORTFOLIO)
        assert portfolio.status_code == 429
        assert not report.usable

    def test_a_server_error_does_not_abort_the_whole_probe(self, ledger: Ledger) -> None:
        transport = _healthy_transport(
            **{"/equity/metadata/exchanges": json_response("{}", status=503)}
        )
        report = run_probe(_client(transport, ledger), ledger=ledger)
        exchanges = next(r for r in report.results if r.endpoint is Endpoint.EXCHANGES)
        assert exchanges.status_code == 503
        # The essential three still answered.
        assert report.usable

    def test_usable_requires_all_three_essential_endpoints(self, ledger: Ledger) -> None:
        for path in ("/equity/account/cash", "/equity/portfolio", "/equity/orders"):
            transport = _healthy_transport(**{path: json_response("{}", status=503)})
            report = run_probe(_client(transport, ledger), ledger=ledger)
            assert not report.usable, f"{path} failing should make the adapter unusable"


class TestInstrumentCache:
    def test_instruments_round_trip_through_the_cache(self, ledger: Ledger) -> None:
        """Cached because the endpoint allows one call per fifty seconds.

        Reading it on a schedule would spend the budget reconciliation needs.
        """
        client = _client(_healthy_transport(), ledger)
        instruments = client.get_instruments()
        assert cache_instruments(ledger, instruments) == 1

        restored = cached_instruments(ledger)
        assert len(restored) == 1
        assert restored[0].ticker == "AAPL_US_EQ"
        assert restored[0].min_trade_quantity is not None
        # Stored as text and restored as Decimal, so a minimum trade size does
        # not pick up a float rounding error on the way through.
        assert str(restored[0].min_trade_quantity) == "0.1"

    def test_caching_is_idempotent(self, ledger: Ledger) -> None:
        client = _client(_healthy_transport(), ledger)
        instruments = client.get_instruments()
        cache_instruments(ledger, instruments)
        cache_instruments(ledger, instruments)
        assert len(cached_instruments(ledger)) == 1


class TestEstimate:
    def test_the_estimate_is_reported_and_shrinks_when_slow_calls_are_skipped(self) -> None:
        full = estimate_duration_seconds()
        quick = estimate_duration_seconds(skip_slow=True)
        assert full > quick > 0
        # The instruments endpoint alone accounts for most of the difference.
        assert full - quick >= 50.0


class TestNoWrites:
    def test_the_probe_only_ever_reads(self, ledger: Ledger) -> None:
        transport = _healthy_transport()
        run_probe(_client(transport, ledger), ledger=ledger)
        assert set(transport.methods_called()) == {"GET"}

    def test_the_key_never_reaches_the_archive(self, ledger: Ledger) -> None:
        run_probe(_client(_healthy_transport(), ledger), ledger=ledger)
        rows = ledger.conn.execute("SELECT raw_body, request_json FROM broker_messages").fetchall()
        for row in rows:
            assert KEY not in json.dumps(dict(row))


@pytest.mark.parametrize("cold", [True, False])
def test_the_probe_works_from_either_governor_state(ledger: Ledger, cold: bool) -> None:
    report = run_probe(_client(_healthy_transport(), ledger, cold=cold), ledger=ledger)
    assert report.usable


def test_the_probe_does_not_require_a_ledger(tmp_path: Path) -> None:
    """Useful for a quick check against an account without touching the log."""
    transport = _healthy_transport()
    clock = FakeClock()
    client = T212Client(
        ClientConfig(api_key=KEY, base_url="https://x/api/v0", environment="demo"),
        transport=transport,
        governor=RateGovernor(clock=clock.time, sleep=clock.sleep, assume_spent_on_cold_boot=False),
        ledger=None,
    )
    report = run_probe(client, ledger=None)
    assert report.usable
