"""Time. Always timezone-aware, always UTC at rest.

A trading system that lets a naive datetime into its data model will eventually
compare a naive local timestamp to an aware exchange timestamp and be wrong by
the size of an offset. Every timestamp that reaches the ledger goes through
here, and `ruff`'s DTZ rules are enabled to keep `datetime.now()` out of the
codebase entirely.
"""

from __future__ import annotations

import time
from datetime import UTC, datetime

# Microsecond-precision ISO 8601 with an explicit offset. Sorts lexicographically
# in the same order it sorts chronologically, which is what makes it safe to use
# as a stored key and in the hash chain.
ISO_FORMAT = "%Y-%m-%dT%H:%M:%S.%f%z"


def now_utc() -> datetime:
    """Current wall-clock time, timezone-aware UTC."""
    return datetime.now(UTC)


def now_iso() -> str:
    """Current wall-clock time as a canonical ISO 8601 UTC string."""
    return to_iso(now_utc())


def monotonic() -> float:
    """A monotonic reading, for ordering events within a single run.

    Wall-clock time can step backwards (NTP correction, VM migration). When two
    events carry the same `ts_utc`, `ts_mono` breaks the tie honestly.
    """
    return time.monotonic()


def to_iso(dt: datetime) -> str:
    """Serialise an aware datetime to canonical ISO 8601 UTC.

    Raises on a naive datetime rather than guessing a zone.
    """
    if dt.tzinfo is None:
        raise ValueError(f"refusing to serialise a naive datetime: {dt!r}")
    return dt.astimezone(UTC).strftime(ISO_FORMAT)


def from_iso(text: str) -> datetime:
    """Parse an ISO 8601 timestamp, requiring an explicit offset."""
    dt = datetime.fromisoformat(text)
    if dt.tzinfo is None:
        raise ValueError(f"timestamp has no timezone offset: {text!r}")
    return dt.astimezone(UTC)
