"""The rate governor.

Trading 212's limits are per-account, so two processes sharing an account share
a budget, and a 429 during reconciliation is the worst possible time to
discover that. Three mechanisms, each for a failure this system will actually
meet:

**Priority among waiters.** Most limits are one call per period, so there is no
bucket to subdivide. When several callers want the same endpoint, the
risk-reducing one goes first — an exit must not queue behind a batch of
entries. This is why waiting is done on a condition variable with an explicit
priority check rather than a plain lock: a `threading.Lock` is FIFO-ish at best
and would, on the one day it matters, spend the budget opening positions while
the exits wait.

**A reserve on multi-token buckets.** Market orders get 50 per minute. A
risk-increasing call may not drain that below the reserve, so a flatten is
still possible after a runaway loop has eaten the rest.

**Pessimistic persistence.** Token state is written to disk *when a token is
consumed*, not on a timer. Losing the record of a refill is safe — we simply
wait longer than necessary. Losing the record of a spend is not: the next boot
would think it had budget it does not have and burst straight into a 429. A
cold boot with no state at all assumes the budget is fully spent.

The server's `x-ratelimit-*` headers override the local estimate on every
response. Our count is an inference; theirs is the one that returns 429.
"""

from __future__ import annotations

import contextlib
import json
import os
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from tb.broker.t212.endpoints import Endpoint, EndpointSpec, spec_for
from tb.core.errors import TbError

# Added to every computed wait. Clock granularity and network jitter mean
# sleeping the exact refill interval sometimes arrives a millisecond early,
# which the server counts and we do not.
SPACING_MARGIN_SECONDS = 0.05

# A 429 with no usable reset header. Long enough to clear any of the documented
# periods rather than guessing at which bucket was hit.
DEFAULT_BACKOFF_SECONDS = 60.0


class RateLimitTimeout(TbError):
    """The caller's deadline passed before a token was available.

    Not an error condition in itself: the M4 dispatcher uses a deadline to drop
    an entry intent whose originating bar has gone stale, which is the correct
    outcome rather than executing late.
    """

    def __init__(self, endpoint: Endpoint, waited: float, needed: float) -> None:
        super().__init__(
            f"rate limit timeout on {endpoint.value}: waited {waited:.2f}s, "
            f"needed about {needed:.2f}s more"
        )
        self.endpoint = endpoint
        self.waited = waited
        self.needed = needed


@dataclass(slots=True)
class RateLimitHeaders:
    """The server's own view of the budget."""

    limit: int | None = None
    period_seconds: int | None = None
    remaining: int | None = None
    reset_epoch: float | None = None

    @property
    def informative(self) -> bool:
        return self.limit is not None or self.remaining is not None

    def as_dict(self) -> dict[str, Any]:
        return {
            "limit": self.limit,
            "period_seconds": self.period_seconds,
            "remaining": self.remaining,
            "reset_epoch": self.reset_epoch,
        }

    @classmethod
    def from_response_headers(cls, headers: dict[str, str]) -> RateLimitHeaders:
        """Parse `x-ratelimit-*`, tolerating absence and nonsense.

        Header names are matched case-insensitively, and an unparseable value is
        treated as absent — a malformed header must not be able to crash the
        client mid-session.
        """
        lowered = {k.lower(): v for k, v in headers.items()}

        def _int(name: str) -> int | None:
            raw = lowered.get(name)
            if raw is None:
                return None
            try:
                return int(float(raw.strip()))
            except (ValueError, AttributeError):
                return None

        def _float(name: str) -> float | None:
            raw = lowered.get(name)
            if raw is None:
                return None
            try:
                return float(raw.strip())
            except (ValueError, AttributeError):
                return None

        return cls(
            limit=_int("x-ratelimit-limit"),
            period_seconds=_int("x-ratelimit-period"),
            remaining=_int("x-ratelimit-remaining"),
            reset_epoch=_float("x-ratelimit-reset"),
        )


@dataclass(slots=True)
class _Bucket:
    """Token state for one endpoint.

    Refill is computed from wall-clock time rather than a monotonic reading,
    because the state has to survive a restart and `time.monotonic()` has no
    meaning across processes. Clock skew is handled by refusing to credit
    negative elapsed time.
    """

    spec: EndpointSpec
    tokens: float
    last_refill_epoch: float
    blocked_until_epoch: float = 0.0
    # Recorded from response headers, so the probe can report disagreement
    # between what we assumed and what the venue enforces.
    observed_limit: int | None = None
    observed_period_seconds: int | None = None

    def refill(self, now: float) -> None:
        elapsed = now - self.last_refill_epoch
        if elapsed <= 0:
            # The clock moved backwards (NTP step, VM migration). Crediting the
            # negative would *remove* tokens; crediting zero and resetting the
            # marker is the conservative reading.
            self.last_refill_epoch = now
            return
        rate = self.spec.capacity / self.spec.period_seconds
        self.tokens = min(float(self.spec.capacity), self.tokens + elapsed * rate)
        self.last_refill_epoch = now

    def floor_for(self, *, risk_reducing: bool) -> float:
        """How many tokens must remain untouched by this caller."""
        if risk_reducing:
            return 0.0
        return float(self.spec.reserve_for_risk_reducing)

    def wait_needed(self, now: float, *, risk_reducing: bool) -> float:
        """Seconds until this caller may take a token. 0.0 means now."""
        if now < self.blocked_until_epoch:
            return self.blocked_until_epoch - now
        floor = self.floor_for(risk_reducing=risk_reducing)
        if self.tokens >= floor + 1.0:
            return 0.0
        rate = self.spec.capacity / self.spec.period_seconds
        deficit = (floor + 1.0) - self.tokens
        return deficit / rate

    def to_state(self) -> dict[str, Any]:
        return {
            "tokens": self.tokens,
            "last_refill_epoch": self.last_refill_epoch,
            "blocked_until_epoch": self.blocked_until_epoch,
            "observed_limit": self.observed_limit,
            "observed_period_seconds": self.observed_period_seconds,
        }


class RateGovernor:
    """Enforces Trading 212's per-account rate limits."""

    def __init__(
        self,
        *,
        state_path: Path | None = None,
        clock: Any = time.time,
        sleep: Any = time.sleep,
        assume_spent_on_cold_boot: bool = True,
    ) -> None:
        self._clock = clock
        self._sleep = sleep
        self._state_path = state_path
        self._lock = threading.RLock()
        # Woken whenever tokens are returned or a block expires, so waiters can
        # re-evaluate in priority order rather than by arrival.
        self._wakeup = threading.Condition(self._lock)
        # Count of risk-reducing callers currently waiting. A risk-increasing
        # caller yields while this is non-zero.
        self._risk_reducing_waiters = 0
        self._buckets: dict[Endpoint, _Bucket] = {}
        self._assume_spent = assume_spent_on_cold_boot
        self._init_buckets()

    # -- setup -------------------------------------------------------------

    def _init_buckets(self) -> None:
        now = float(self._clock())
        persisted = self._load_state()

        for endpoint in Endpoint:
            spec = spec_for(endpoint)
            saved = persisted.get(endpoint.value)
            if saved is not None:
                bucket = _Bucket(
                    spec=spec,
                    tokens=float(saved.get("tokens", 0.0)),
                    last_refill_epoch=float(saved.get("last_refill_epoch", now)),
                    blocked_until_epoch=float(saved.get("blocked_until_epoch", 0.0)),
                    observed_limit=saved.get("observed_limit"),
                    observed_period_seconds=saved.get("observed_period_seconds"),
                )
                # Credit whatever time passed while the process was down.
                bucket.refill(now)
            else:
                # No record of this endpoint. If the previous process spent its
                # budget a moment before dying, starting full would burst into a
                # 429 during reconciliation — the worst possible moment.
                bucket = _Bucket(
                    spec=spec,
                    tokens=0.0 if self._assume_spent else float(spec.capacity),
                    last_refill_epoch=now,
                )
            self._buckets[endpoint] = bucket

    # -- persistence -------------------------------------------------------

    def _load_state(self) -> dict[str, dict[str, Any]]:
        if self._state_path is None or not self._state_path.exists():
            return {}
        try:
            body = json.loads(self._state_path.read_text(encoding="utf-8"))
        except (OSError, ValueError, UnicodeDecodeError):
            # Unreadable state is the same as no state: assume spent.
            return {}
        if not isinstance(body, dict):
            return {}
        buckets = body.get("buckets")
        return buckets if isinstance(buckets, dict) else {}

    def _persist(self) -> None:
        """Write token state. Called after a token is consumed.

        Best-effort by design: if this fails the next boot falls back to
        assuming the budget is spent, which is the safe direction. Raising here
        would turn a full disk into an inability to trade.
        """
        if self._state_path is None:
            return
        payload = {
            "written_at_epoch": float(self._clock()),
            "buckets": {e.value: b.to_state() for e, b in self._buckets.items()},
        }
        try:
            self._state_path.parent.mkdir(parents=True, exist_ok=True)
            tmp = self._state_path.with_suffix(self._state_path.suffix + ".tmp")
            tmp.write_text(json.dumps(payload, sort_keys=True), encoding="utf-8")
            os.replace(tmp, self._state_path)
        except OSError:
            pass

    # -- acquiring ---------------------------------------------------------

    def acquire(
        self,
        endpoint: Endpoint,
        *,
        risk_reducing: bool = False,
        timeout: float | None = None,
    ) -> float:
        """Block until a call to `endpoint` is permitted. Returns seconds waited.

        `risk_reducing=True` marks a call that can only decrease exposure — an
        exit, a protective stop, a cancel, a flatten. Such calls may consume the
        reserve and are served ahead of waiting risk-increasing calls.

        Raises `RateLimitTimeout` if `timeout` elapses first, which the M4
        dispatcher uses to drop an intent whose bar has gone stale rather than
        execute it late.
        """
        started = float(self._clock())
        deadline = None if timeout is None else started + timeout

        with self._wakeup:
            if risk_reducing:
                self._risk_reducing_waiters += 1
            try:
                while True:
                    now = float(self._clock())
                    bucket = self._buckets[endpoint]
                    bucket.refill(now)

                    wait = bucket.wait_needed(now, risk_reducing=risk_reducing)

                    # Yield to any risk-reducing caller already waiting on this
                    # endpoint, even when a token is free: letting an entry take
                    # the last token while an exit waits is the exact inversion
                    # the reserve exists to prevent, and on capacity-1 buckets
                    # the reserve cannot express it.
                    if wait <= 0.0 and not risk_reducing and self._risk_reducing_waiters > 0:
                        wait = max(SPACING_MARGIN_SECONDS, bucket.spec.period_seconds / 4.0)

                    if wait <= 0.0:
                        bucket.tokens -= 1.0
                        # Persist the spend immediately — see the class docstring.
                        self._persist()
                        return max(0.0, now - started)

                    if deadline is not None and now + wait > deadline:
                        raise RateLimitTimeout(endpoint, now - started, wait)

                    sleep_for = wait + SPACING_MARGIN_SECONDS
                    if deadline is not None:
                        sleep_for = min(sleep_for, max(0.0, deadline - now))

                    # Released while sleeping so a risk-reducing caller can
                    # arrive, register itself, and be seen on the next pass.
                    self._wakeup.release()
                    try:
                        self._sleep(sleep_for)
                    finally:
                        self._wakeup.acquire()
            finally:
                if risk_reducing:
                    self._risk_reducing_waiters -= 1
                    self._wakeup.notify_all()

    def try_acquire(self, endpoint: Endpoint, *, risk_reducing: bool = False) -> bool:
        """Take a token if one is free right now. Never blocks."""
        with self._lock:
            now = float(self._clock())
            bucket = self._buckets[endpoint]
            bucket.refill(now)
            if bucket.wait_needed(now, risk_reducing=risk_reducing) > 0.0:
                return False
            bucket.tokens -= 1.0
            self._persist()
            return True

    def wait_estimate(self, endpoint: Endpoint, *, risk_reducing: bool = False) -> float:
        """How long a call would have to wait, without taking a token."""
        with self._lock:
            now = float(self._clock())
            bucket = self._buckets[endpoint]
            bucket.refill(now)
            return bucket.wait_needed(now, risk_reducing=risk_reducing)

    # -- learning from the server -----------------------------------------

    def observe_headers(self, endpoint: Endpoint, headers: RateLimitHeaders) -> None:
        """Adopt the server's count. It is authoritative; ours is an estimate."""
        if not headers.informative:
            return
        with self._lock:
            bucket = self._buckets[endpoint]
            if headers.limit is not None:
                bucket.observed_limit = headers.limit
            if headers.period_seconds is not None:
                bucket.observed_period_seconds = headers.period_seconds
            if headers.remaining is not None:
                # Take the more pessimistic of the two views. If the server says
                # we have fewer left than we thought, believe it; if it says
                # more, our own accounting may be tracking a request in flight
                # that its counter has not yet reflected.
                bucket.tokens = min(bucket.tokens, float(headers.remaining))
                bucket.last_refill_epoch = float(self._clock())
            self._persist()
            self._wakeup_all()

    def note_rate_limited(self, endpoint: Endpoint, headers: RateLimitHeaders) -> float:
        """Record a 429 and block the endpoint until its reset.

        Returns the backoff applied. A 429 means the local model was wrong, so
        the bucket is emptied as well as blocked — resuming with a stale token
        count would immediately earn another.
        """
        with self._lock:
            now = float(self._clock())
            bucket = self._buckets[endpoint]
            if headers.reset_epoch is not None and headers.reset_epoch > now:
                backoff = headers.reset_epoch - now
            elif headers.period_seconds is not None:
                backoff = float(headers.period_seconds)
            else:
                backoff = DEFAULT_BACKOFF_SECONDS
            bucket.blocked_until_epoch = now + backoff
            bucket.tokens = 0.0
            bucket.last_refill_epoch = now
            self._persist()
            return backoff

    def _wakeup_all(self) -> None:
        # Only raises if the lock is not held, which callers always hold.
        with contextlib.suppress(RuntimeError):
            self._wakeup.notify_all()

    # -- reporting ---------------------------------------------------------

    def observations(self) -> list[dict[str, Any]]:
        """Configured versus observed limits, for the probe report."""
        with self._lock:
            rows = []
            for endpoint, bucket in self._buckets.items():
                spec = bucket.spec
                agrees: bool | None = None
                if bucket.observed_limit is not None and bucket.observed_period_seconds:
                    configured_rate = spec.capacity / spec.period_seconds
                    observed_rate = bucket.observed_limit / bucket.observed_period_seconds
                    agrees = abs(configured_rate - observed_rate) < 1e-9
                rows.append(
                    {
                        "endpoint": endpoint.value,
                        "method": spec.method,
                        "path": spec.path,
                        "configured_limit": spec.capacity,
                        "configured_period_s": spec.period_seconds,
                        "observed_limit": bucket.observed_limit,
                        "observed_period_s": bucket.observed_period_seconds,
                        "agrees": agrees,
                    }
                )
            return rows

    def disagreements(self) -> list[str]:
        """Human-readable notes where the venue disagrees with the table.

        The point of the probe: a limit we assumed wrong is a 429 waiting to
        happen at the worst moment, and it should be a line in a report rather
        than a surprise during reconciliation.
        """
        notes = []
        for row in self.observations():
            if row["agrees"] is False:
                notes.append(
                    f"{row['endpoint']}: configured {row['configured_limit']}"
                    f"/{row['configured_period_s']:g}s but the venue reports "
                    f"{row['observed_limit']}/{row['observed_period_s']}s"
                )
        return notes


@dataclass(slots=True)
class NullGovernor:
    """A governor that never waits. Tests and backtests only.

    Deliberately not the default anywhere: a governor that can be switched off
    by a config flag will eventually be switched off in production.
    """

    calls: list[tuple[Endpoint, bool]] = field(default_factory=list)

    def acquire(
        self,
        endpoint: Endpoint,
        *,
        risk_reducing: bool = False,
        timeout: float | None = None,
    ) -> float:
        self.calls.append((endpoint, risk_reducing))
        return 0.0

    def try_acquire(self, endpoint: Endpoint, *, risk_reducing: bool = False) -> bool:
        self.calls.append((endpoint, risk_reducing))
        return True

    def wait_estimate(self, endpoint: Endpoint, *, risk_reducing: bool = False) -> float:
        return 0.0

    def observe_headers(self, endpoint: Endpoint, headers: RateLimitHeaders) -> None:
        return None

    def note_rate_limited(self, endpoint: Endpoint, headers: RateLimitHeaders) -> float:
        return 0.0

    def observations(self) -> list[dict[str, Any]]:
        return []

    def disagreements(self) -> list[str]:
        return []
