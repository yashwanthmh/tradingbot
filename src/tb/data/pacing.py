"""Request pacing for market-data providers.

Deliberately **not** the Trading 212 `RateGovernor`. That class carries
machinery this does not need and cannot use: a reserve withheld for
risk-reducing orders, priority among waiters so an exit never queues behind a
batch of entries, and persistence keyed to the T212 endpoint enum. All of it
exists because placing an order is irreversible and a 429 during
reconciliation is dangerous. Fetching a price is neither: the worst case is
waiting, or a gap in history that the audit will report.

So this is a second, much smaller implementation, and that duplication is a
deliberate choice rather than an oversight. Generalising the governor to
arbitrary bucket keys was the alternative; it would have meant refactoring the
most safety-critical tested code in the repo to serve a caller with none of
its constraints.

What it does need: a rate cap per provider (Yahoo's unofficial endpoint will
throttle or captcha; Alpaca's free tier caps around 200 requests a minute),
and backoff that respects `Retry-After` when a vendor sends one.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field

from tb.core.errors import TbError


class PacerStalled(TbError):
    """The pacer could not make progress because its clock is not advancing."""


DEFAULT_BACKOFF_SECONDS = 30.0

# `acquire` sleeps and re-checks, so it needs a clock that advances. If one does
# not — a frozen test double, a wedged monotonic source, a container with a
# broken timer — the loop spins forever, and a hang inside the data layer stalls
# the trading loop with no diagnostic at all. Bounded instead, with an error
# that names the cause. A genuine wait takes one or two iterations even when the
# backoff is ten minutes, so the ceiling can be low enough to be obvious.
MAX_ACQUIRE_ITERATIONS = 64

# The two floors that keep the loop above making progress, both of them about
# floating-point reality rather than pacing policy.
#
# A wall clock reads ~1.7e9, where a double's spacing is ~2.4e-7 seconds. A
# refill therefore lands the bucket at 0.99999998 tokens rather than 1.0, and
# the wait computed from that shortfall is smaller than the clock can express —
# so sleeping it does not move the clock, the next refill credits nothing, and
# the loop spins on a rounding error. Neither floor changes the long-run rate:
# the epsilon is a few parts per billion of a token, and the minimum sleep is
# shorter than any real request.
TOKEN_EPSILON = 1e-6
MIN_SLEEP_SECONDS = 1e-3


@dataclass(slots=True)
class PacingSpec:
    """How often a provider may be called."""

    requests: int
    period_seconds: float
    # Applied on top of the computed wait, to absorb clock granularity rather
    # than arriving a millisecond early and having the vendor count it.
    margin_seconds: float = 0.05

    @property
    def interval(self) -> float:
        return self.period_seconds / self.requests


@dataclass(slots=True)
class ProviderPacer:
    """A token bucket per provider, with vendor-directed backoff."""

    spec: PacingSpec
    clock: object = field(default=time.time)
    sleeper: object = field(default=time.sleep)
    _tokens: float = 0.0
    _last_refill: float | None = None
    _blocked_until: float = 0.0
    waits: int = 0
    total_waited: float = 0.0

    def _now(self) -> float:
        return float(self.clock())  # type: ignore[operator]

    def _sleep(self, seconds: float) -> None:
        self.sleeper(seconds)  # type: ignore[operator]

    def _refill(self, now: float) -> None:
        if self._last_refill is None:
            # Cold start assumes the budget is spent, for the same reason the
            # broker governor does: a process that restarts immediately after
            # exhausting its allowance should not burst.
            self._tokens = 0.0
            self._last_refill = now
            return
        elapsed = now - self._last_refill
        if elapsed <= 0:
            # Wall clock stepped backwards. Crediting the negative would remove
            # tokens and stall for the size of the step.
            self._last_refill = now
            return
        rate = self.spec.requests / self.spec.period_seconds
        self._tokens = min(float(self.spec.requests), self._tokens + elapsed * rate)
        self._last_refill = now

    def acquire(self) -> float:
        """Block until a request is permitted. Returns seconds waited."""
        started = self._now()
        for _ in range(MAX_ACQUIRE_ITERATIONS):
            now = self._now()
            self._refill(now)
            if now < self._blocked_until:
                wait = self._blocked_until - now
            elif self._tokens >= 1.0 - TOKEN_EPSILON:
                self._tokens = max(0.0, self._tokens - 1.0)
                waited = max(0.0, now - started)
                if waited > 0:
                    self.waits += 1
                    self.total_waited += waited
                return waited
            else:
                wait = (1.0 - self._tokens) / (self.spec.requests / self.spec.period_seconds)
            self._sleep(max(wait, MIN_SLEEP_SECONDS) + self.spec.margin_seconds)

        raise PacerStalled(
            f"waited {MAX_ACQUIRE_ITERATIONS} times without acquiring a token and the clock "
            f"has not advanced past {started}. The pacer's clock is not moving — a frozen "
            "test double or a wedged timer — so this would otherwise spin forever."
        )

    def note_throttled(self, *, retry_after_seconds: float | None = None) -> float:
        """Record a 429 and stop calling for a while.

        The bucket is emptied as well as blocked: a 429 means the local model
        was wrong, and resuming with a stale token count just earns another.
        """
        now = self._now()
        backoff = retry_after_seconds if retry_after_seconds else DEFAULT_BACKOFF_SECONDS
        self._blocked_until = now + backoff
        self._tokens = 0.0
        self._last_refill = now
        return backoff

    @property
    def blocked_for(self) -> float:
        remaining = self._blocked_until - self._now()
        return max(0.0, remaining)


# Conservative defaults. Yahoo publishes no limit at all — the endpoint is
# unofficial — so this is a politeness budget rather than a documented one.
YAHOO_PACING = PacingSpec(requests=1, period_seconds=1.5)
# Alpaca's free tier documents roughly 200 requests/minute; stay well under.
ALPACA_PACING = PacingSpec(requests=150, period_seconds=60.0)
