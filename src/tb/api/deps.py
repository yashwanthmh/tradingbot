"""What every dashboard request needs: its settings, who is asking, a ledger.

Kept apart from the app so the routes can depend on it without importing the
app that includes them.
"""

from __future__ import annotations

import hmac
import ipaddress
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import cast

from fastapi import HTTPException, Request

from tb.config.loader import PinnedLimits
from tb.core.clock import now_utc
from tb.core.errors import TbError
from tb.ledger.schema import LEDGER_SCHEMA_VERSION, schema_version
from tb.ledger.store import Ledger

TOKEN_ENV = "TB_DASHBOARD_TOKEN"  # noqa: S105 - the variable's name, not a token
# Long enough that guessing it over HTTP is not a plan: `secrets.token_urlsafe(32)`
# gives 43 characters.
MIN_TOKEN_LENGTH = 24
LOOPBACK_HOSTS: tuple[str, ...] = ("localhost", "127.0.0.1", "[::1]")


class DashboardError(TbError):
    """The dashboard cannot start as asked."""


@dataclass(frozen=True)
class DashboardSettings:
    """Everything a request needs, fixed when the server starts."""

    db: Path
    pinned: PinnedLimits
    journal_dir: Path
    token: str | None = field(default=None, repr=False)
    allowed_hosts: tuple[str, ...] = LOOPBACK_HOSTS
    clock: Callable[[], datetime] = now_utc

    @property
    def button_enabled(self) -> bool:
        return self.token is not None


def check_token(token: str) -> str:
    """Refuse a token too weak to stand between the network and the switch.

    The message never contains the token: a refusal is printed, and a token
    in a terminal's scrollback is a token in whatever records that terminal.
    """
    if len(token) < MIN_TOKEN_LENGTH:
        raise DashboardError(
            f"{TOKEN_ENV} is shorter than {MIN_TOKEN_LENGTH} characters. Generate one with "
            "`python -c 'import secrets; print(secrets.token_urlsafe(32))'`."
        )
    if not token.isascii() or not token.isprintable() or any(c.isspace() for c in token):
        raise DashboardError(f"{TOKEN_ENV} must be printable ASCII with no whitespace.")
    return token


def is_loopback(host: str | None) -> bool:
    if host is None:
        return False
    if host == "localhost":
        return True
    try:
        return ipaddress.ip_address(host.strip("[]")).is_loopback
    except ValueError:
        return False


def settings_of(request: Request) -> DashboardSettings:
    return cast(DashboardSettings, request.app.state.settings)


def _bearer_matches(request: Request, token: str) -> bool:
    scheme, _, given = request.headers.get("authorization", "").partition(" ")
    # Compared as bytes in constant time; a mismatch in length says only that.
    return scheme.lower() == "bearer" and hmac.compare_digest(
        given.strip().encode("utf-8"), token.encode("utf-8")
    )


def _unauthorised() -> HTTPException:
    return HTTPException(
        status_code=401,
        detail=f"this dashboard needs the token from {TOKEN_ENV} as a bearer token",
        headers={"WWW-Authenticate": "Bearer"},
    )


def require_reader(request: Request) -> DashboardSettings:
    """Anyone with the token; without one configured, loopback peers only."""
    settings = settings_of(request)
    if settings.token is None:
        peer = request.client.host if request.client else None
        if not is_loopback(peer):
            raise HTTPException(
                status_code=403,
                detail=f"without {TOKEN_ENV} this dashboard answers on loopback only",
            )
        return settings
    if not _bearer_matches(request, settings.token):
        raise _unauthorised()
    return settings


def require_operator(request: Request) -> DashboardSettings:
    """The token, always: the one control that writes answers to a credential."""
    settings = settings_of(request)
    if settings.token is None:
        raise HTTPException(
            status_code=403,
            detail=(
                f"the kill switch button is off: start the dashboard with {TOKEN_ENV} set "
                "to enable it. `tb halt --reason ...` works without it."
            ),
        )
    if not _bearer_matches(request, settings.token):
        raise _unauthorised()
    return settings


@contextmanager
def reading(settings: DashboardSettings) -> Iterator[Ledger]:
    """A read-only ledger for one request, opened and closed on its thread.

    Opened per request rather than shared: SQLite connections belong to the
    thread that made them, and a read-only connection in WAL mode is cheap
    and never blocks the trading loop's writes.
    """
    if not settings.db.exists():
        raise HTTPException(status_code=503, detail="there is no ledger yet: run `tb init`")
    try:
        ledger = Ledger(settings.db, read_only=True).open()
    except TbError as exc:
        raise HTTPException(status_code=503, detail="the ledger cannot be opened") from exc
    try:
        found = schema_version(ledger.conn)
        if found != LEDGER_SCHEMA_VERSION:
            raise HTTPException(
                status_code=503,
                detail=(
                    f"the ledger is at schema {found} and this build reads {LEDGER_SCHEMA_VERSION}"
                ),
            )
        yield ledger
    finally:
        ledger.close()
