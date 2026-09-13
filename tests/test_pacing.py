"""Provider pacing.

Small surface, but two of these tests exist because of bugs found by running
the provider suite rather than by reading the code: a token bucket on a
floating-point clock can leave itself a rounding error short of a full token,
and the wait it then computes can be smaller than the clock's own resolution.
The result was an infinite busy loop inside the data layer — a hang with no
diagnostic, in a process that is supposed to be watching a market.
"""

from __future__ import annotations

import pytest

from tb.data.pacing import (
    ALPACA_PACING,
    DEFAULT_BACKOFF_SECONDS,
    MAX_ACQUIRE_ITERATIONS,
    YAHOO_PACING,
    PacerStalled,
    PacingSpec,
    ProviderPacer,
)


class VirtualClock:
    """A clock that only moves when something sleeps."""

    def __init__(self, start: float = 0.0) -> None:
        self.now = start
        self.slept: list[float] = []

    def __call__(self) -> float:
        return self.now

    def sleep(self, seconds: float) -> None:
        self.slept.append(seconds)
        self.now += seconds


def build(spec: PacingSpec, *, start: float = 0.0) -> tuple[ProviderPacer, VirtualClock]:
    clock = VirtualClock(start)
    return ProviderPacer(spec=spec, clock=clock, sleeper=clock.sleep), clock


def test_a_cold_pacer_does_not_burst() -> None:
    """Restarting must not hand back a full bucket.

    A process that crashes just after exhausting its allowance and comes straight
    back would otherwise fire a full period's worth of requests immediately, and
    earn the throttle it was trying to avoid.
    """
    pacer, clock = build(PacingSpec(requests=2, period_seconds=1.0, margin_seconds=0.0))
    pacer.acquire()
    assert clock.slept, "a cold start granted a token without waiting for it"


def test_the_long_run_rate_matches_the_spec() -> None:
    pacer, clock = build(PacingSpec(requests=10, period_seconds=1.0, margin_seconds=0.0))
    for _ in range(20):
        pacer.acquire()
    # Twenty requests at ten per second cannot take less than about two seconds.
    assert clock.now >= 1.9


def test_a_clock_at_wall_time_magnitude_still_makes_progress() -> None:
    """The bug the provider suite found.

    At a real epoch value (~1.7e9) a double's spacing is ~2.4e-7 seconds, so a
    refill lands at 0.99999998 tokens instead of 1.0 and the shortfall implies a
    wait the clock cannot represent. Without the token epsilon and the minimum
    sleep, this spins forever.
    """
    pacer, clock = build(
        PacingSpec(requests=1000, period_seconds=1.0, margin_seconds=0.0),
        start=1_774_000_000.0,
    )
    for _ in range(50):
        pacer.acquire()
    assert clock.now > 1_774_000_000.0


def test_a_frozen_clock_raises_rather_than_hanging() -> None:
    """Fail loudly. A hang here stalls the trading loop with nothing in the log."""
    pacer = ProviderPacer(
        spec=PacingSpec(requests=1, period_seconds=60.0, margin_seconds=0.0),
        clock=lambda: 1_000.0,
        sleeper=lambda _seconds: None,
    )
    with pytest.raises(PacerStalled, match="clock is not moving"):
        pacer.acquire()


def test_the_stall_guard_leaves_room_for_a_genuine_long_backoff() -> None:
    """A ten-minute Retry-After must not be mistaken for a stalled clock."""
    pacer, clock = build(PacingSpec(requests=1, period_seconds=1.0, margin_seconds=0.0))
    pacer.note_throttled(retry_after_seconds=600.0)
    pacer.acquire()
    assert clock.now >= 600.0
    assert len(clock.slept) < MAX_ACQUIRE_ITERATIONS


def test_a_backwards_clock_does_not_stall_the_pacer() -> None:
    """NTP correction and VM migration both step the wall clock backwards.

    Crediting the negative elapsed time would *remove* tokens and stall for the
    size of the step, which on a one-second correction is survivable and on a
    misconfigured host is not.
    """
    pacer, clock = build(PacingSpec(requests=10, period_seconds=1.0, margin_seconds=0.0))
    pacer.acquire()
    clock.now -= 3600.0
    pacer.acquire()
    assert clock.now > -3600.0


def test_a_429_empties_the_bucket_as_well_as_blocking() -> None:
    """The local model was wrong, so resuming on its token count earns another 429."""
    pacer, clock = build(PacingSpec(requests=100, period_seconds=1.0, margin_seconds=0.0))
    for _ in range(3):
        pacer.acquire()
    backoff = pacer.note_throttled(retry_after_seconds=5.0)
    assert backoff == 5.0
    assert pacer.blocked_for == pytest.approx(5.0)
    pacer.acquire()
    assert clock.now >= 5.0


def test_a_429_without_retry_after_uses_the_default_backoff() -> None:
    pacer, _ = build(PacingSpec(requests=1, period_seconds=1.0))
    assert pacer.note_throttled() == DEFAULT_BACKOFF_SECONDS


def test_the_shipped_specs_are_conservative() -> None:
    """Both budgets are politeness, not entitlement.

    Yahoo's endpoint is unofficial and publishes no limit at all; Alpaca's free
    tier documents roughly 200 requests a minute. Neither number should drift
    upward without someone deciding to.
    """
    assert YAHOO_PACING.interval >= 1.0
    assert ALPACA_PACING.requests <= 200
    assert ALPACA_PACING.period_seconds == 60.0
