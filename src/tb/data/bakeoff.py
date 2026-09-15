"""The paid-data decision, stated as arithmetic rather than as a feeling.

The question this answers is not "can we get a bar in time". It is **"is this
the bar the market saw"**, and that reframing is the whole point of the module.

Alpaca's free tier is IEX-only: roughly 2% of US consolidated volume. So its
minute close is not the consolidated last price, its high and low are not the
session extremes, and a thin name has minutes with no print at all. The
systematic disagreement against the consolidated tape runs in the 5-20bps band
— **the same order as the entire gross edge a minute-bar strategy would be
trading**. A staleness bound of 180 seconds catches none of that, because the
bar arrives promptly and is simply the wrong number.

So the output is not a verdict someone argues about. It is a single figure:

    minimum_viable_edge_bps = p95_disagreement_bps * min_edge_to_feed_noise_ratio

That is the gross edge a strategy would need before the feed's own error is
small enough to trade through. Compare it to what a candidate strategy claims,
and the paid-data question answers itself. At a p95 of 12bps and a required
ratio of 3.0, a strategy needs 36bps gross — against a 30-55bps round-trip cost
on Trading 212, which is the arithmetic that says minute-resolution trading on
free data is not a business.

Three measurement rules that decide whether the number means anything:

**Compare the same bar period only.** Comparing one feed's 15:59 bar to
another's 15:45 measures fourteen minutes of price movement, not feed
disagreement — and on a trending name it would produce a large, confident,
meaningless number.

**Regular session both sides.** An extended-hours print can sit hundreds of
basis points from the regular close. One of those in the sample moves the p95
on its own.

**Missing and disagreeing are different measurements.** A minute where Alpaca
has no bar and Yahoo does is not a 0bps agreement and not a divergence: it is
IEX having no print, which is its own statistic (`missing_bar_fraction`) and
its own argument for paid data.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from datetime import datetime
from decimal import Decimal
from enum import StrEnum

from tb.core.clock import to_iso
from tb.core.ids import new_id
from tb.data.provider import Bar, DataError, Provenance, Resolution, Session
from tb.ledger.events import Actor, BakeoffPayload, EventType
from tb.ledger.store import Ledger


class BakeoffError(DataError):
    """The comparison could not be carried out."""


class Verdict(StrEnum):
    """What the measurement says about trading this resolution on this feed."""

    # The feed's own error is small enough that a plausible edge survives it.
    FREE_DATA_SUFFICIENT = "free_data_sufficient"
    # The error is the same size as the edge. Paid data, or a slower cadence.
    PAID_DATA_REQUIRED = "paid_data_required"
    # Not enough overlapping bars to say anything. Not a pass.
    INSUFFICIENT_SAMPLE = "insufficient_sample"

    @property
    def permits_live_resolution(self) -> bool:
        """Whether this verdict may widen `data.allowed_live_resolutions`.

        `INSUFFICIENT_SAMPLE` deliberately does not. A measurement that could
        not be taken is not a measurement that passed, and treating absence of
        evidence as a pass is how a gate becomes decoration.
        """
        return self is Verdict.FREE_DATA_SUFFICIENT


# Below this many overlapping bars the percentiles are noise. A p95 over twenty
# observations is the second-worst of twenty, which moves by a factor on one
# bad print.
MIN_COMPARABLE_BARS = 200


@dataclass(frozen=True, slots=True)
class BarComparison:
    """One bar period, seen by two providers."""

    instrument_uid: str
    bar_open_utc: datetime
    primary_close: Decimal
    secondary_close: Decimal
    disagreement_bps: float

    @property
    def mid(self) -> Decimal:
        return (self.primary_close + self.secondary_close) / Decimal(2)


@dataclass(frozen=True, slots=True)
class ProviderStats:
    """What one feed looked like over the window."""

    provider: str
    bars_present: int
    bars_expected: int
    observed_delay_p95_s: float | None = None

    @property
    def missing_fraction(self) -> float:
        if self.bars_expected <= 0:
            return 0.0
        return max(0.0, 1.0 - self.bars_present / self.bars_expected)


@dataclass(frozen=True, slots=True)
class BakeoffResult:
    """The comparison, and the number the decision turns on."""

    bakeoff_id: str
    resolution: Resolution
    window_start: datetime | None
    window_end: datetime | None
    primary: str
    secondary: str
    n_symbols: int
    n_compared_bars: int
    median_bps: float | None
    p95_bps: float | None
    p99_bps: float | None
    stats: tuple[ProviderStats, ...] = field(default_factory=tuple)
    min_edge_to_noise_ratio: float = 3.0
    max_delay_seconds: int = 180
    cycles_meeting_staleness_pct: float | None = None
    worst: tuple[BarComparison, ...] = field(default_factory=tuple)

    @property
    def minimum_viable_edge_bps(self) -> float | None:
        """The gross edge a strategy needs before the feed's error is tolerable.

        The headline number, and the one that makes the paid-data decision
        arithmetic: below this, a strategy is trading inside its own data's
        error bar and its backtested edge is indistinguishable from feed noise.
        """
        if self.p95_bps is None:
            return None
        return self.p95_bps * self.min_edge_to_noise_ratio

    @property
    def verdict(self) -> Verdict:
        if self.n_compared_bars < MIN_COMPARABLE_BARS or self.p95_bps is None:
            return Verdict.INSUFFICIENT_SAMPLE
        # The round-trip cost on Trading 212 is 30-55bps. A required gross edge
        # beyond that leaves nothing, so the feed is not good enough for this
        # resolution regardless of how clever the strategy is.
        required = self.minimum_viable_edge_bps
        if required is not None and required >= float(ROUND_TRIP_COST_BPS):
            return Verdict.PAID_DATA_REQUIRED
        return Verdict.FREE_DATA_SUFFICIENT

    @property
    def rationale(self) -> str:
        if self.verdict is Verdict.INSUFFICIENT_SAMPLE:
            return (
                f"only {self.n_compared_bars} comparable {self.resolution.value} bars "
                f"(need {MIN_COMPARABLE_BARS}). A p95 over this few observations moves by "
                "a factor on one bad print, so nothing can be concluded — which is not "
                "the same as concluding the feed is fine."
            )
        required = self.minimum_viable_edge_bps
        assert required is not None and self.p95_bps is not None
        missing = ", ".join(
            f"{stat.provider} missing {stat.missing_fraction:.1%}" for stat in self.stats
        )
        if self.verdict is Verdict.PAID_DATA_REQUIRED:
            return (
                f"{self.primary} and {self.secondary} disagree by {self.p95_bps:.1f}bps at "
                f"the 95th percentile over {self.n_compared_bars} bars, so a strategy would "
                f"need {required:.0f}bps of gross edge to clear the feed's own error by "
                f"{self.min_edge_to_noise_ratio:.1f}x. A Trading 212 round trip already "
                f"costs {ROUND_TRIP_COST_BPS}bps, which leaves nothing. "
                f"{self.resolution.value} trading on this feed is not a business; pay for "
                f"consolidated data or trade a slower cadence. ({missing})"
            )
        return (
            f"{self.primary} and {self.secondary} agree to {self.p95_bps:.1f}bps at the "
            f"95th percentile over {self.n_compared_bars} bars, so {required:.0f}bps of "
            f"gross edge clears the feed's error by {self.min_edge_to_noise_ratio:.1f}x. "
            f"({missing})"
        )

    def as_dict(self) -> dict[str, object]:
        return {
            "bakeoff_id": self.bakeoff_id,
            "resolution": self.resolution.value,
            "primary": self.primary,
            "secondary": self.secondary,
            "n_symbols": self.n_symbols,
            "n_compared_bars": self.n_compared_bars,
            "median_bps": self.median_bps,
            "p95_bps": self.p95_bps,
            "p99_bps": self.p99_bps,
            "minimum_viable_edge_bps": self.minimum_viable_edge_bps,
            "verdict": self.verdict.value,
            "missing_fraction": {stat.provider: stat.missing_fraction for stat in self.stats},
        }


# A US round trip from a GBP account: 0.15% FX each way plus spread. The figure
# the required-edge number is compared against, stated here rather than buried
# in a branch, because it is the reason a 12bps p95 is fatal rather than merely
# annoying.
ROUND_TRIP_COST_BPS = Decimal("30")


def percentile(values: Sequence[float], fraction: float) -> float | None:
    """Nearest-rank percentile. No interpolation, no numpy.

    Nearest-rank on purpose: an interpolated p95 invents a value that no bar
    actually disagreed by, and the whole number is meant to be traceable to an
    observation someone can go and look at.
    """
    if not values:
        return None
    ordered = sorted(values)
    index = min(len(ordered) - 1, max(0, round(fraction * (len(ordered) - 1))))
    return ordered[index]


def disagreement_bps(primary: Decimal, secondary: Decimal) -> float:
    """Relative disagreement in basis points, against the midpoint.

    Against the midpoint rather than either side, so the measure does not
    depend on which feed is called primary — otherwise swapping the arguments
    would change the reported number and the p95 with it.
    """
    if primary <= 0 or secondary <= 0:
        raise BakeoffError(f"non-positive price in a comparison ({primary}, {secondary})")
    mid = (primary + secondary) / Decimal(2)
    return float(abs(primary - secondary) / mid) * 10_000.0


def compare_feeds(
    primary_bars: Iterable[Bar],
    secondary_bars: Iterable[Bar],
    *,
    include_extended: bool = False,
) -> tuple[tuple[BarComparison, ...], int, int]:
    """Pair bars by instrument and bar period. Returns `(pairs, only_primary, only_secondary)`.

    Pairing on `(instrument_uid, bar_open_utc)` is the measurement rule that
    makes the result mean anything. Pairing by position, or by nearest
    timestamp, would measure the price movement between two different minutes —
    a large, confident and completely meaningless number on any trending name.

    Bars present on one side only are *counted*, not compared. A minute where
    IEX had no print is not a 0bps agreement; it is its own statistic, and on a
    2%-of-volume feed it is a large one.
    """

    def index(bars: Iterable[Bar]) -> dict[tuple[str, datetime], Bar]:
        return {
            (bar.instrument_uid, bar.bar_open_utc): bar
            for bar in bars
            if include_extended or bar.session is Session.REGULAR
        }

    left, right = index(primary_bars), index(secondary_bars)
    shared = left.keys() & right.keys()

    pairs = tuple(
        BarComparison(
            instrument_uid=uid,
            bar_open_utc=moment,
            primary_close=left[(uid, moment)].close,
            secondary_close=right[(uid, moment)].close,
            disagreement_bps=disagreement_bps(
                left[(uid, moment)].close, right[(uid, moment)].close
            ),
        )
        for uid, moment in sorted(shared, key=lambda key: (key[0], key[1]))
    )
    return pairs, len(left.keys() - right.keys()), len(right.keys() - left.keys())


def delay_p95(bars: Iterable[Bar]) -> float | None:
    """The 95th-percentile observed delay, from live bars only.

    Backfilled bars have `available_at` set to bar close, so a delay of zero.
    Including them would report a delayed feed as instantaneous, which is
    exactly the number that makes it look live-capable.
    """
    delays = [bar.delay_seconds for bar in bars if bar.provenance is Provenance.LIVE]
    return percentile(delays, 0.95)


class Bakeoff:
    """Runs the comparison and records the verdict."""

    def __init__(
        self,
        ledger: Ledger,
        *,
        min_edge_to_noise_ratio: float = 3.0,
        max_delay_seconds: int = 180,
        run_id: str | None = None,
    ) -> None:
        self._ledger = ledger
        self._ratio = min_edge_to_noise_ratio
        self._max_delay = max_delay_seconds
        self._run_id = run_id

    def run(
        self,
        *,
        primary_bars: Sequence[Bar],
        secondary_bars: Sequence[Bar],
        resolution: Resolution,
        primary: str,
        secondary: str,
        expected_bars: int | None = None,
        include_extended: bool = False,
        emit_event: bool = True,
    ) -> BakeoffResult:
        """Compare two feeds over the same window and state the verdict.

        `expected_bars` is the calendar's count of session bars, used as the
        denominator for "how often did this feed have no print". Falling back to
        the larger feed's own count would make the better-covered feed look
        perfect by definition.
        """
        if primary == secondary:
            raise BakeoffError(
                f"cannot bake off {primary} against itself; a feed agrees with itself "
                "by construction and the result would be a 0bps p95"
            )

        pairs, only_primary, only_secondary = compare_feeds(
            primary_bars, secondary_bars, include_extended=include_extended
        )
        spreads = [pair.disagreement_bps for pair in pairs]
        denominator = expected_bars or max(len(pairs) + only_primary, len(pairs) + only_secondary)

        result = BakeoffResult(
            bakeoff_id=new_id("bake", length=12),
            resolution=resolution,
            window_start=min((b.bar_open_utc for b in primary_bars), default=None),
            window_end=max((b.bar_open_utc for b in primary_bars), default=None),
            primary=primary,
            secondary=secondary,
            n_symbols=len({bar.instrument_uid for bar in primary_bars}),
            n_compared_bars=len(pairs),
            median_bps=percentile(spreads, 0.50),
            p95_bps=percentile(spreads, 0.95),
            p99_bps=percentile(spreads, 0.99),
            stats=(
                ProviderStats(
                    provider=primary,
                    bars_present=len(pairs) + only_primary,
                    bars_expected=denominator,
                    observed_delay_p95_s=delay_p95(primary_bars),
                ),
                ProviderStats(
                    provider=secondary,
                    bars_present=len(pairs) + only_secondary,
                    bars_expected=denominator,
                    observed_delay_p95_s=delay_p95(secondary_bars),
                ),
            ),
            min_edge_to_noise_ratio=self._ratio,
            max_delay_seconds=self._max_delay,
            cycles_meeting_staleness_pct=self._staleness_coverage(primary_bars),
            # The worst handful, so the headline number can be traced to bars
            # somebody can go and look at.
            worst=tuple(sorted(pairs, key=lambda p: -p.disagreement_bps)[:5]),
        )

        # Persist the delay measurement, not just report it. Until this was
        # wired the p95 went into the event payload and nowhere else, so the
        # conservative declared constant stayed authoritative forever and
        # `live_capable`'s preference for an observed delay was unreachable.
        # `measure` refuses backfilled bars, so a history-only run records
        # nothing rather than laundering the declared guess into the table
        # meant to replace it.
        self._record_observations(
            {primary: primary_bars, secondary: secondary_bars}, resolution=resolution
        )

        if emit_event:
            self._emit(result)
        return result

    def _record_observations(
        self, by_provider: dict[str, Sequence[Bar]], *, resolution: Resolution
    ) -> None:
        from tb.data.observations import ProviderObservationStore

        store = ProviderObservationStore(self._ledger)
        for provider, bars in by_provider.items():
            store.measure_and_record(bars, provider=provider, resolution=resolution)

    def _staleness_coverage(self, bars: Sequence[Bar]) -> float | None:
        """What fraction of live bars arrived inside the staleness bound.

        Measured on `available_at` minus bar close, never on bar time: a
        fifteen-minute-delayed feed produces bars whose *timestamps* look
        current, and measuring the wrong axis is what makes such a feed appear
        usable.
        """
        live = [bar for bar in bars if bar.provenance is Provenance.LIVE]
        if not live:
            return None
        inside = sum(1 for bar in live if bar.delay_seconds <= self._max_delay)
        return inside / len(live) * 100.0

    def _emit(self, result: BakeoffResult) -> None:
        self._ledger.append(
            EventType.DATA_BAKEOFF_COMPLETED,
            result.bakeoff_id,
            BakeoffPayload(
                bakeoff_id=result.bakeoff_id,
                window_start=("" if result.window_start is None else to_iso(result.window_start)),
                window_end="" if result.window_end is None else to_iso(result.window_end),
                resolution=result.resolution.value,
                providers=[result.primary, result.secondary],
                n_symbols=result.n_symbols,
                n_compared_bars=result.n_compared_bars,
                disagreement_median_bps=result.median_bps,
                disagreement_p95_bps=result.p95_bps,
                disagreement_p99_bps=result.p99_bps,
                missing_bar_fraction={
                    stat.provider: stat.missing_fraction for stat in result.stats
                },
                observed_delay_p95_s={
                    stat.provider: stat.observed_delay_p95_s
                    for stat in result.stats
                    if stat.observed_delay_p95_s is not None
                },
                cycles_meeting_staleness_bound_pct=result.cycles_meeting_staleness_pct,
                verdict=result.verdict.value,
                rationale=result.rationale,
            ),
            actor=Actor.SYSTEM,
            run_id=self._run_id,
        )

    def latest(self, resolution: Resolution) -> dict[str, object] | None:
        """The most recent recorded verdict for a resolution.

        Read by the gate that decides whether `allowed_live_resolutions` may be
        widened, so the answer comes from the ledger rather than from whatever
        someone remembers the last run said.
        """
        import json

        for row in self._ledger.iter_events(event_type=EventType.DATA_BAKEOFF_COMPLETED):
            payload = json.loads(str(row["payload_json"]))
            if payload.get("resolution") == resolution.value:
                return dict(payload)
        return None


def may_widen_live_resolutions(
    result: BakeoffResult, *, currently_allowed: Sequence[str]
) -> tuple[bool, str]:
    """Whether this measurement justifies letting the bot trade a resolution live.

    Separate from the verdict so that widening a hard limit stays a deliberate,
    reviewable edit rather than something a passing measurement does by itself.
    This function says whether the *evidence* supports the edit; a human still
    makes it.
    """
    if result.resolution.value in currently_allowed:
        return True, f"{result.resolution.value} is already permitted"
    if not result.verdict.permits_live_resolution:
        return False, (
            f"the measurement does not support it: {result.verdict.value}. {result.rationale}"
        )
    return True, (
        f"the measurement supports adding {result.resolution.value} to "
        f"data.allowed_live_resolutions: {result.rationale} Edit the limits file "
        "deliberately — this function does not widen anything by itself."
    )
