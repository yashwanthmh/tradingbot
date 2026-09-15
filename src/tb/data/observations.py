"""Turning a provider's declared delay into a measured one.

Every provider ships a `declared_delay_seconds` table — a deliberately
conservative guess, because on day one nobody has measured anything and
guessing low would admit a feed that cannot actually drive a live decision.
`ProviderCapabilities.live_capable` already prefers an `observed_delay` over
the declared one when it is handed one, with the reasoning stated there: *the
declared value is a guess and the measurement is not.*

Both ends of that existed and the middle did not. `bakeoff.delay_p95` computes
the number and puts it in an event payload; `provider_observations` has been
in the schema since M2. Nothing wrote the table, so no measurement ever
survived the run that produced it, and the conservative constant stayed
authoritative forever.

Two rules about what may be measured:

**Only `live` bars carry a real delay.** A backfilled bar's knowledge time is
*assumed* — `available_at` is computed from the declared delay rather than
observed — so measuring a backfill would recover the guess and record it as a
measurement. That is worse than not measuring: it launders an assumption into
evidence. Backfilled samples are refused, and the refusal says why.

**A measurement is pessimistic-only against the declared value, in the
direction that matters.** A provider that measures *faster* than declared is
believed, because that is the number `live_capable` should use. A provider
that measures *slower* is also believed — and that one may take a resolution
out of live eligibility, which is the point of measuring at all.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from datetime import datetime

from tb.core.clock import now_iso
from tb.core.errors import TbError
from tb.data.provider import Bar, Provenance, ProviderCapabilities, Resolution
from tb.ledger.store import Ledger

# Below this many samples a percentile is noise. Recorded anyway, but flagged
# so a caller can tell a measurement from an anecdote.
MIN_SAMPLES_FOR_CONFIDENCE = 30


class ObservationError(TbError):
    """A delay could not be measured from the samples given."""


@dataclass(frozen=True, slots=True)
class DelayObservation:
    """What a provider's delay actually turned out to be."""

    provider: str
    resolution: Resolution
    observed_at: str
    p50_seconds: float
    p95_seconds: float
    max_seconds: float
    n_samples: int
    note: str = ""

    @property
    def confident(self) -> bool:
        """Whether the sample is large enough for the percentile to mean much."""
        return self.n_samples >= MIN_SAMPLES_FOR_CONFIDENCE

    def live_capable_against(
        self, capabilities: ProviderCapabilities, *, max_delay_seconds: float
    ) -> tuple[bool, str]:
        """Re-ask the live-capability question using the measurement.

        The p95 rather than the median: a feed that is fast four times in five
        and late once is late, and the decision path is what suffers on the
        fifth.
        """
        return capabilities.live_capable(
            self.resolution,
            max_delay_seconds=max_delay_seconds,
            observed_delay=self.p95_seconds,
        )


def _percentile(values: Sequence[float], fraction: float) -> float:
    """Nearest-rank percentile on a sorted copy.

    Nearest-rank rather than interpolated: these are delays in seconds over a
    modest sample, and an interpolated value invents a latency nobody observed.
    """
    if not values:
        raise ObservationError("cannot take a percentile of no samples")
    ordered = sorted(values)
    index = min(len(ordered) - 1, max(0, round(fraction * (len(ordered) - 1))))
    return ordered[index]


def delays_of(bars: Iterable[Bar]) -> list[float]:
    """Observed delay per bar, in seconds, from live bars only.

    `ingested_at - bar_close`: when we actually had it, against when it was
    complete. Backfilled bars are skipped rather than measured — their
    `available_at` was computed from the declared delay, so including them
    would recover the guess and stamp it as evidence.
    """
    found: list[float] = []
    for bar in bars:
        if bar.provenance is not Provenance.LIVE:
            continue
        delay = (bar.ingested_at_utc - bar.bar_close_utc).total_seconds()
        # A negative delay means the bar was ingested before it closed, i.e. a
        # partially formed bar got stored. `Bar.is_settled_at` exists to stop
        # that; if one slips through, it is not a delay sample.
        if delay >= 0:
            found.append(delay)
    return found


def measure(
    bars: Iterable[Bar],
    *,
    provider: str,
    resolution: Resolution,
    observed_at: str | None = None,
) -> DelayObservation | None:
    """Summarise observed delays, or `None` if there is nothing measurable.

    `None` rather than a zero-filled observation. A provider with no live bars
    has an *unmeasured* delay, and recording zeros would read as "measured,
    instant" — which would admit the feed to live trading on the strength of
    having never been observed.
    """
    samples = delays_of(bars)
    if not samples:
        return None
    return DelayObservation(
        provider=provider,
        resolution=resolution,
        observed_at=observed_at or now_iso(),
        p50_seconds=_percentile(samples, 0.50),
        p95_seconds=_percentile(samples, 0.95),
        max_seconds=max(samples),
        n_samples=len(samples),
        note=(
            ""
            if len(samples) >= MIN_SAMPLES_FOR_CONFIDENCE
            else f"only {len(samples)} live samples; treat the p95 as indicative"
        ),
    )


class ProviderObservationStore:
    """Reads and writes `provider_observations`.

    Append-only by `(provider, resolution, observed_at)`, so the delay history
    is visible rather than a single overwritten current value. A provider that
    was fast in March and slow in June is a different fact from one that has
    always been slow, and only the first is a degradation.
    """

    def __init__(self, ledger: Ledger) -> None:
        self._ledger = ledger

    def record(self, observation: DelayObservation, *, live_capable: bool | None = None) -> None:
        self._ledger.conn.execute(
            """
            INSERT INTO provider_observations (
                provider, resolution, observed_at, delay_p50_s, delay_p95_s,
                delay_max_s, n_samples, live_capable, note
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(provider, resolution, observed_at) DO UPDATE SET
                delay_p50_s = excluded.delay_p50_s,
                delay_p95_s = excluded.delay_p95_s,
                delay_max_s = excluded.delay_max_s,
                n_samples = excluded.n_samples,
                live_capable = excluded.live_capable,
                note = excluded.note
            """,
            (
                observation.provider,
                observation.resolution.value,
                observation.observed_at,
                observation.p50_seconds,
                observation.p95_seconds,
                observation.max_seconds,
                observation.n_samples,
                None if live_capable is None else int(live_capable),
                observation.note,
            ),
        )
        self._ledger.conn.commit()

    def latest(self, provider: str, resolution: Resolution) -> DelayObservation | None:
        row = self._ledger.conn.execute(
            "SELECT * FROM provider_observations WHERE provider = ? AND resolution = ? "
            "ORDER BY observed_at DESC LIMIT 1",
            (provider, resolution.value),
        ).fetchone()
        if row is None:
            return None
        return DelayObservation(
            provider=str(row["provider"]),
            resolution=Resolution(row["resolution"]),
            observed_at=str(row["observed_at"]),
            p50_seconds=float(row["delay_p50_s"]),
            p95_seconds=float(row["delay_p95_s"]),
            max_seconds=float(row["delay_max_s"]),
            n_samples=int(row["n_samples"]),
            note=str(row["note"] or ""),
        )

    def observed_delay_for(self, provider: str, resolution: Resolution) -> float | None:
        """The measurement to hand `live_capable`, or `None` if unmeasured.

        `None` is the honest answer for an unmeasured provider, and
        `live_capable` then falls back to the conservative declared constant
        rather than to an optimistic zero.
        """
        found = self.latest(provider, resolution)
        return None if found is None else found.p95_seconds

    def history(self, provider: str | None = None, limit: int = 50) -> list[DelayObservation]:
        sql = "SELECT * FROM provider_observations"
        params: list[object] = []
        if provider:
            sql += " WHERE provider = ?"
            params.append(provider)
        sql += " ORDER BY observed_at DESC LIMIT ?"
        params.append(limit)
        rows = self._ledger.conn.execute(sql, params).fetchall()
        return [
            DelayObservation(
                provider=str(row["provider"]),
                resolution=Resolution(row["resolution"]),
                observed_at=str(row["observed_at"]),
                p50_seconds=float(row["delay_p50_s"]),
                p95_seconds=float(row["delay_p95_s"]),
                max_seconds=float(row["delay_max_s"]),
                n_samples=int(row["n_samples"]),
                note=str(row["note"] or ""),
            )
            for row in rows
        ]

    def measure_and_record(
        self,
        bars: Iterable[Bar],
        *,
        provider: str,
        resolution: Resolution,
        capabilities: ProviderCapabilities | None = None,
        max_delay_seconds: float | None = None,
        observed_at: datetime | None = None,
    ) -> DelayObservation | None:
        """Measure, then record, resolving live capability if asked.

        The combined form, so a caller cannot measure and forget to persist —
        which is exactly how this table stayed empty through all of M2.
        """
        observation = measure(
            bars,
            provider=provider,
            resolution=resolution,
            observed_at=None if observed_at is None else observed_at.isoformat(),
        )
        if observation is None:
            return None
        capable: bool | None = None
        if capabilities is not None and max_delay_seconds is not None:
            capable, _ = observation.live_capable_against(
                capabilities, max_delay_seconds=max_delay_seconds
            )
        self.record(observation, live_capable=capable)
        return observation
