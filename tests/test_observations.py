"""Provider delay observations, and the archive wiring.

The point of this module is that a *declared* delay is a guess made before
anyone measured anything, and it stays authoritative forever unless something
writes the measurement down. `ProviderCapabilities.live_capable` already
prefers an observed delay when handed one, and `bakeoff.delay_p95` already
computes one — the table in between was never written, so no measurement ever
survived the run that produced it.

The load-bearing test here is
`test_a_backfilled_bar_is_never_measured`: a backfilled bar's knowledge time
was *computed from the declared delay*, so measuring it recovers the guess and
records it as evidence. That is worse than not measuring at all.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from decimal import Decimal
from pathlib import Path
from typing import Any

import pytest

from tb.broker.t212.raw_archive import RawArchive
from tb.data.observations import (
    MIN_SAMPLES_FOR_CONFIDENCE,
    DelayObservation,
    ObservationError,
    ProviderObservationStore,
    delays_of,
    measure,
)
from tb.data.provider import (
    Bar,
    Provenance,
    ProviderCapabilities,
    Resolution,
    Session,
    TimestampConvention,
)
from tb.ledger.store import Ledger

BASE = datetime(2024, 1, 2, tzinfo=UTC)
UID = "isin:US0378331005"


def bar(minute: int, *, delay_s: float, provenance: Provenance = Provenance.LIVE) -> Bar:
    opened = BASE + timedelta(minutes=minute)
    close = opened + timedelta(minutes=1)
    price = Decimal("100.00")
    return Bar(
        instrument_uid=UID,
        resolution=Resolution.MINUTE,
        bar_open_utc=opened,
        available_at_utc=close + timedelta(seconds=delay_s),
        ingested_at_utc=close + timedelta(seconds=delay_s),
        provider="alpaca",
        provenance=provenance,
        session=Session.REGULAR,
        open=price,
        high=price,
        low=price,
        close=price,
        volume=1_000,
    )


@pytest.fixture
def ledger_with_schema(tmp_path: Path) -> Any:
    db = tmp_path / "ledger.db"
    with Ledger(db) as opened:
        opened.initialise(created_by="test")
    return db


def caps(*, declared: float) -> ProviderCapabilities:
    return ProviderCapabilities(
        name="alpaca",
        resolutions=frozenset({Resolution.MINUTE}),
        timestamp_convention=TimestampConvention.BAR_OPEN,
        declared_delay_seconds={Resolution.MINUTE: declared},
        max_history_days={Resolution.MINUTE: 30},
    )


# --------------------------------------------------------------------------
# What may be measured
# --------------------------------------------------------------------------


def test_a_backfilled_bar_is_never_measured() -> None:
    """The load-bearing rule. Measuring a backfill launders a guess.

    A backfilled bar's `available_at` was *computed* from the declared delay,
    so a measurement over backfilled bars recovers the declared constant and
    then records it in the table that exists to replace it. The result would
    look like evidence and be a tautology.
    """
    backfilled = [bar(i, delay_s=30, provenance=Provenance.BACKFILL) for i in range(50)]
    assert delays_of(backfilled) == []
    assert measure(backfilled, provider="alpaca", resolution=Resolution.MINUTE) is None


def test_live_bars_are_measured_from_ingest_against_bar_close() -> None:
    bars = [bar(i, delay_s=45) for i in range(10)]
    samples = delays_of(bars)
    assert len(samples) == 10
    assert all(s == pytest.approx(45.0) for s in samples)


def test_a_mixed_batch_measures_only_the_live_bars() -> None:
    bars = [
        *(bar(i, delay_s=10) for i in range(5)),
        *(bar(100 + i, delay_s=900, provenance=Provenance.BACKFILL) for i in range(20)),
    ]
    samples = delays_of(bars)
    assert len(samples) == 5
    assert max(samples) == pytest.approx(10.0)


def test_a_bar_ingested_before_it_closed_is_not_a_delay_sample() -> None:
    """Reachable, and it matters most for the fixture provider.

    `Bar` enforces `available_at >= bar_close` but deliberately leaves
    `ingested_at` unconstrained — `CsvFixtureProvider` is clockless and stamps
    `FIXTURE_INGESTED_AT = 1970-01-01`, so tightening that relation would
    break determinism in the test provider on purpose.

    Which means a fixture bar measures to a delay of minus fifty years. A
    guard that averaged it in would report a feed as arriving decades before
    its bars closed, and `live_capable` would admit anything.
    """
    early = Bar(
        instrument_uid=UID,
        resolution=Resolution.MINUTE,
        bar_open_utc=BASE,
        # Knowable legitimately after close...
        available_at_utc=BASE + timedelta(minutes=2),
        # ...but stamped as ingested long before, the way a fixture is.
        ingested_at_utc=datetime(1970, 1, 1, tzinfo=UTC),
        provider="fixture",
        provenance=Provenance.LIVE,
        session=Session.REGULAR,
        open=Decimal("100.00"),
        high=Decimal("100.00"),
        low=Decimal("100.00"),
        close=Decimal("100.00"),
        volume=1_000,
    )
    assert (early.ingested_at_utc - early.bar_close_utc).total_seconds() < 0
    assert delays_of([early]) == []
    assert measure([early], provider="fixture", resolution=Resolution.MINUTE) is None


def test_no_live_bars_measures_to_none_rather_than_zero() -> None:
    """Zero would read as "measured, instant".

    Which would admit a feed to live trading on the strength of having never
    been observed — the exact inversion this module exists to prevent.
    """
    assert measure([], provider="alpaca", resolution=Resolution.MINUTE) is None


# --------------------------------------------------------------------------
# The summary
# --------------------------------------------------------------------------


def test_the_percentiles_are_nearest_rank() -> None:
    """Interpolation would invent a latency nobody observed."""
    bars = [bar(i, delay_s=float(i)) for i in range(101)]
    found = measure(bars, provider="alpaca", resolution=Resolution.MINUTE)
    assert found is not None
    assert found.p50_seconds == pytest.approx(50.0)
    assert found.p95_seconds == pytest.approx(95.0)
    assert found.max_seconds == pytest.approx(100.0)
    assert found.n_samples == 101


def test_a_small_sample_is_recorded_but_flagged() -> None:
    """An anecdote is still worth storing; it must not pass as a measurement."""
    found = measure(
        [bar(i, delay_s=20) for i in range(5)], provider="alpaca", resolution=Resolution.MINUTE
    )
    assert found is not None
    assert not found.confident
    assert "indicative" in found.note

    big = measure(
        [bar(i, delay_s=20) for i in range(MIN_SAMPLES_FOR_CONFIDENCE)],
        provider="alpaca",
        resolution=Resolution.MINUTE,
    )
    assert big is not None
    assert big.confident
    assert big.note == ""


def test_a_percentile_of_nothing_raises() -> None:
    from tb.data.observations import _percentile

    with pytest.raises(ObservationError, match="no samples"):
        _percentile([], 0.5)


# --------------------------------------------------------------------------
# Feeding live capability
# --------------------------------------------------------------------------


def test_a_measurement_can_take_a_resolution_out_of_live_eligibility() -> None:
    """The whole point of measuring. The declared value was optimistic.

    A provider declaring 30s and measuring 400s at the p95 is not live-capable
    against a 180s bound, and only the measurement can say so.
    """
    observation = measure(
        [bar(i, delay_s=400) for i in range(40)],
        provider="alpaca",
        resolution=Resolution.MINUTE,
    )
    assert observation is not None
    declared_ok, _ = caps(declared=30).live_capable(Resolution.MINUTE, max_delay_seconds=180)
    measured_ok, reason = observation.live_capable_against(caps(declared=30), max_delay_seconds=180)
    assert declared_ok, "the declared value should have looked fine"
    assert not measured_ok
    assert "beyond the" in reason


def test_the_p95_is_used_rather_than_the_median() -> None:
    """Fast four times in five and late once is late.

    The decision path is what suffers on the fifth, so the tail is the number
    that decides eligibility.
    """
    bars = [
        *(bar(i, delay_s=10) for i in range(80)),
        *(bar(100 + i, delay_s=900) for i in range(20)),
    ]
    observation = measure(bars, provider="alpaca", resolution=Resolution.MINUTE)
    assert observation is not None
    assert observation.p50_seconds == pytest.approx(10.0)
    assert observation.p95_seconds > 100.0
    capable, _ = observation.live_capable_against(caps(declared=10), max_delay_seconds=180)
    assert not capable


# --------------------------------------------------------------------------
# Persistence
# --------------------------------------------------------------------------


def test_an_observation_survives_the_run_that_produced_it(ledger_with_schema: Path) -> None:
    """The gap this module closes. Nothing wrote this table through all of M2."""
    with Ledger(ledger_with_schema).open() as ledger:
        store = ProviderObservationStore(ledger)
        assert store.latest("alpaca", Resolution.MINUTE) is None
        assert store.observed_delay_for("alpaca", Resolution.MINUTE) is None

        recorded = store.measure_and_record(
            [bar(i, delay_s=60) for i in range(40)],
            provider="alpaca",
            resolution=Resolution.MINUTE,
            capabilities=caps(declared=30),
            max_delay_seconds=180,
        )
        assert recorded is not None

        found = store.latest("alpaca", Resolution.MINUTE)
        assert found is not None
        assert found.p95_seconds == pytest.approx(60.0)
        assert store.observed_delay_for("alpaca", Resolution.MINUTE) == pytest.approx(60.0)


def test_an_unmeasured_provider_falls_back_to_the_conservative_constant(
    ledger_with_schema: Path,
) -> None:
    """`None`, not zero. The declared guess is the safe default, not an
    optimistic measurement nobody took."""
    with Ledger(ledger_with_schema).open() as ledger:
        store = ProviderObservationStore(ledger)
        assert store.observed_delay_for("yahoo", Resolution.DAILY) is None
        capable, reason = caps(declared=30).live_capable(
            Resolution.MINUTE,
            max_delay_seconds=180,
            observed_delay=store.observed_delay_for("yahoo", Resolution.MINUTE),
        )
        assert capable
        assert "30s" in reason


def test_the_delay_history_is_kept_not_overwritten(ledger_with_schema: Path) -> None:
    """Fast in March and slow in June is a degradation; always slow is not.

    Only the first is worth an alert, and a single overwritten current value
    cannot tell them apart.
    """
    with Ledger(ledger_with_schema).open() as ledger:
        store = ProviderObservationStore(ledger)
        for day, delay in ((1, 20.0), (2, 500.0)):
            store.record(
                DelayObservation(
                    provider="alpaca",
                    resolution=Resolution.MINUTE,
                    observed_at=f"2024-03-0{day}T00:00:00+00:00",
                    p50_seconds=delay,
                    p95_seconds=delay,
                    max_seconds=delay,
                    n_samples=50,
                )
            )
        history = store.history("alpaca")
        assert len(history) == 2
        # Newest first, so `latest` is the June reading.
        assert history[0].p95_seconds == pytest.approx(500.0)
        newest = store.latest("alpaca", Resolution.MINUTE)
        assert newest is not None
        assert newest.p95_seconds == pytest.approx(500.0)


# --------------------------------------------------------------------------
# The provider archive
# --------------------------------------------------------------------------


def test_a_provider_payload_is_archived_with_its_source(ledger_with_schema: Path) -> None:
    """Yahoo's failure mode is silently different data rather than an error.

    The archived body is the only way to tell a shape change from a guess
    about one, and before this was wired `tb data backfill` archived nothing.
    """
    with Ledger(ledger_with_schema).open() as ledger:
        archive = RawArchive.for_provider(ledger, provider="yahoo", run_id="run_x")
        msg = archive.record(
            endpoint="yahoo_chart",
            method="GET",
            url_path="/v8/finance/chart/AAPL",
            status_code=200,
            raw_body='{"chart":{"result":[]}}',
            parse_ok=True,
        )
        row = ledger.conn.execute(
            "SELECT environment, endpoint, raw_body FROM broker_messages WHERE msg_id = ?",
            (msg.msg_id,),
        ).fetchone()
        assert row["environment"] == "yahoo"
        assert row["endpoint"] == "yahoo_chart"
        assert "chart" in row["raw_body"]


def test_a_credential_is_scrubbed_from_both_the_body_and_the_params(
    ledger_with_schema: Path,
) -> None:
    """Yahoo's query parameters can carry tokens, and a 401 body can echo a key.

    An archive that leaked a credential would be worse than no archive.
    """
    secret = "SUPERSECRETTOKEN"
    with Ledger(ledger_with_schema).open() as ledger:
        archive = RawArchive.for_provider(ledger, provider="yahoo", redact_values=(secret,))
        msg = archive.record(
            endpoint="yahoo_chart",
            method="GET",
            url_path="/v8/finance/chart/AAPL",
            status_code=401,
            raw_body=f'{{"error":"bad token {secret}"}}',
            params={"crumb": secret},
            parse_ok=False,
        )
        row = ledger.conn.execute(
            "SELECT raw_body, request_json FROM broker_messages WHERE msg_id = ?",
            (msg.msg_id,),
        ).fetchone()
        assert secret not in (row["raw_body"] or "")
        assert secret not in (row["request_json"] or "")
        assert "[REDACTED]" in row["raw_body"]
        assert "[REDACTED]" in row["request_json"]
