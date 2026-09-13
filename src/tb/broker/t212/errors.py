"""Broker error taxonomy.

The distinctions here are not cosmetic — each one leads somewhere different:

* `TransportError` is retryable. Nothing about the account changed.
* `AuthError` is fatal at startup and never retried. Retrying a bad key just
  burns rate limit.
* `RateLimited` feeds the governor's backoff.
* `SchemaDriftError` is a **halt**. A field the system consumes changed shape,
  so any number we produce from this response is a guess.
* `UnknownOrderState` is the one that matters most, and it exists to stop the
  most expensive mistake available: an order whose fate cannot be established
  must never be reported as failed. Treating unknown as failed is how a crashed
  process places the same order twice.
"""

from __future__ import annotations

from tb.core.errors import HaltRequired, TbError


class BrokerError(TbError):
    """Base for everything the broker adapter raises."""


class TransportError(BrokerError):
    """The request did not complete. Retryable; the account is unchanged."""

    def __init__(self, detail: str, *, endpoint: str | None = None) -> None:
        super().__init__(f"transport failure{f' on {endpoint}' if endpoint else ''}: {detail}")
        self.endpoint = endpoint


class AuthError(BrokerError):
    """The key was rejected.

    Fatal and never retried. The most common cause is worth naming in the
    message: Trading 212 mints a key for whichever mode the app was in, so a
    key generated without switching to Practice mode first is a live key and
    will not authenticate against the demo host.
    """

    def __init__(self, status_code: int, environment: str, detail: str = "") -> None:
        super().__init__(
            f"Trading 212 rejected the {environment} API key (HTTP {status_code})"
            f"{f': {detail}' if detail else ''}. Keys are per-environment — a key "
            "generated while the app was in real-money mode will not work against "
            "the demo host, and vice versa."
        )
        self.status_code = status_code
        self.environment = environment


class RateLimited(BrokerError):
    """HTTP 429. The local rate model was wrong."""

    def __init__(self, endpoint: str, backoff_seconds: float) -> None:
        super().__init__(
            f"rate limited on {endpoint}; backing off {backoff_seconds:.1f}s. "
            "Limits are per-account, so another process sharing this account "
            "may be spending the same budget."
        )
        self.endpoint = endpoint
        self.backoff_seconds = backoff_seconds


class BrokerHttpError(BrokerError):
    """A non-2xx response that is not auth or rate limiting."""

    def __init__(self, endpoint: str, status_code: int, body: str) -> None:
        super().__init__(f"{endpoint} returned HTTP {status_code}: {body[:400]}")
        self.endpoint = endpoint
        self.status_code = status_code
        self.body = body


class SchemaDriftError(HaltRequired):
    """A consumed field changed shape. Always a halt.

    Subclasses `HaltRequired` rather than `BrokerError` because the correct
    response is to stop trading, not to retry or degrade. The asymmetry is
    deliberate: unknown *extra* fields are ignored, since taking the bot down
    because the broker added a field would be worse than not reading it. A
    field we actually use going missing, going null, or changing type is
    different — continuing means acting on a number we invented.
    """

    def __init__(
        self, endpoint: str, model: str, detail: str, *, msg_id: str | None = None
    ) -> None:
        super().__init__(
            "broker_schema_drift",
            f"{endpoint} no longer parses as {model}: {detail}. "
            f"The raw response is archived{f' as {msg_id}' if msg_id else ''} for replay.",
        )
        self.endpoint = endpoint
        self.model = model
        self.detail = detail
        self.msg_id = msg_id


class UnknownOrderState(BrokerError):
    """An order's fate could not be established.

    Never resolve this to "not placed" on the strength of an absence. The
    broker's open-orders list drops filled orders, and the history endpoint is
    limited to six calls a minute, so "I cannot find it" is a routine reading
    that is entirely consistent with a filled order.
    """

    def __init__(self, identifier: str, detail: str) -> None:
        super().__init__(
            f"cannot establish the state of {identifier}: {detail}. "
            "Treating this as unknown, not as failed."
        )
        self.identifier = identifier
