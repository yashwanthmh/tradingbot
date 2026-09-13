"""HTTP transport, behind a protocol so the client is testable without a network.

Keeping this separate is not ceremony. The entire M1 test suite — the rate
governor under adversarial bursts, the schema-drift matrix, the reconciler's
findings — runs against `RecordingTransport`, which means those properties are
verified deterministically rather than against whatever a live demo account
happened to hold that morning.

`httpx` lives in the `broker` extra and is imported lazily, so the core install
and the M0 control plane stay dependency-light.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any, Protocol, runtime_checkable


@dataclass(frozen=True, slots=True)
class HttpResponse:
    status_code: int
    headers: dict[str, str]
    text: str
    elapsed_ms: float

    @property
    def ok(self) -> bool:
        return 200 <= self.status_code < 300


@runtime_checkable
class Transport(Protocol):
    def request(
        self,
        method: str,
        url: str,
        *,
        headers: dict[str, str],
        params: dict[str, Any] | None = None,
        json_body: Any = None,
        timeout: float = 20.0,
    ) -> HttpResponse: ...

    def close(self) -> None: ...


class HttpxTransport:
    """The real transport."""

    def __init__(self, *, verify: bool | str = True) -> None:
        try:
            import httpx
        except ImportError as exc:  # pragma: no cover - depends on install extras
            raise ImportError(
                "the broker adapter needs httpx. Install it with "
                "`uv sync --extra broker` (or `--all-extras`)."
            ) from exc

        self._httpx = httpx
        # A connection pool, so the governor's pacing is not confused by TLS
        # handshake time on every call.
        self._client = httpx.Client(
            verify=verify,
            follow_redirects=False,
            headers={"User-Agent": "tradingbot/0.1 (+https://github.com/yashwanthmh/tradingbot)"},
        )

    def request(
        self,
        method: str,
        url: str,
        *,
        headers: dict[str, str],
        params: dict[str, Any] | None = None,
        json_body: Any = None,
        timeout: float = 20.0,
    ) -> HttpResponse:
        from tb.broker.t212.errors import TransportError

        try:
            response = self._client.request(
                method,
                url,
                headers=headers,
                params=params,
                json=json_body,
                timeout=timeout,
            )
        except self._httpx.TimeoutException as exc:
            # Deliberately distinct from other transport failures. A timeout on
            # a POST is the dangerous case: the order may well have been
            # accepted, so M4 must treat the intent as UNKNOWN rather than
            # failed. The message says so where whoever reads the log will see it.
            raise TransportError(
                f"timed out after {timeout}s ({exc}). If this was a write, the "
                "request may still have been accepted — treat the intent as unknown.",
                endpoint=url,
            ) from exc
        except self._httpx.HTTPError as exc:
            raise TransportError(str(exc), endpoint=url) from exc

        return HttpResponse(
            status_code=response.status_code,
            headers=dict(response.headers),
            text=response.text,
            elapsed_ms=response.elapsed.total_seconds() * 1000.0,
        )

    def close(self) -> None:
        self._client.close()


@dataclass(slots=True)
class RecordingTransport:
    """A scripted transport for tests.

    `responses` maps a path suffix to either a response or a callable taking the
    request. A callable lets a test model statefulness — a 429 on the third
    call, a field that vanishes mid-session — which is how the drift and
    rate-limit behaviour gets exercised.
    """

    responses: dict[str, HttpResponse | Callable[[str, dict[str, Any] | None], HttpResponse]] = (
        field(default_factory=dict)
    )
    calls: list[dict[str, Any]] = field(default_factory=list)
    default: HttpResponse | None = None

    def request(
        self,
        method: str,
        url: str,
        *,
        headers: dict[str, str],
        params: dict[str, Any] | None = None,
        json_body: Any = None,
        timeout: float = 20.0,
    ) -> HttpResponse:
        self.calls.append(
            {
                "method": method,
                "url": url,
                "params": params,
                "json": json_body,
                # Recorded so a test can assert the key never appears anywhere
                # it should not, and that the auth header is shaped as expected.
                "headers": dict(headers),
            }
        )

        # Longest match wins. Substring matching is otherwise ambiguous in
        # exactly the case that matters: `/equity/orders` is a prefix of
        # `/equity/orders/{id}`, so a shorter key would shadow the specific one
        # and the test would silently exercise the wrong endpoint.
        matches = [
            (suffix, scripted)
            for suffix, scripted in self.responses.items()
            if url.endswith(suffix) or suffix in url
        ]
        if matches:
            _, scripted = max(matches, key=lambda pair: len(pair[0]))
            if callable(scripted):
                return scripted(url, params)
            return scripted

        if self.default is not None:
            return self.default

        return HttpResponse(
            status_code=404,
            headers={},
            text=f'{{"error":"RecordingTransport has no script for {url}"}}',
            elapsed_ms=1.0,
        )

    def close(self) -> None:
        return None

    def paths_called(self) -> list[str]:
        return [call["url"] for call in self.calls]

    def methods_called(self) -> list[str]:
        return [call["method"] for call in self.calls]


def json_response(
    body: str, *, status: int = 200, ratelimit: dict[str, str] | None = None
) -> HttpResponse:
    """Build a scripted JSON response, optionally with rate-limit headers."""
    headers = {"content-type": "application/json"}
    if ratelimit:
        headers.update(ratelimit)
    return HttpResponse(status_code=status, headers=headers, text=body, elapsed_ms=5.0)
