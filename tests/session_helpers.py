"""Writing a ledger at chosen instants: runs, cycles and events on a pinned clock.

The session record, the journal, the drills and the arming gate all read time
off the ledger, so their tests need events that land at the instant a scenario
says — a run that starts before the open, a crash at two o'clock, a kill switch
on a Tuesday. Pinning the ledger's clock gives exactly that while every event
still goes through the real append path, chain and all.

The week used throughout has no holiday in it, and US Eastern is UTC-4 across
it, so the regular session is 13:30 to 20:00 UTC every day.
"""

from __future__ import annotations

from datetime import UTC, date, datetime, timedelta
from typing import Any

import pytest

from tb.core.clock import to_iso
from tb.ledger.events import EventType, LoopCyclePayload
from tb.ledger.store import Ledger

MON = date(2026, 9, 21)
TUE = date(2026, 9, 22)
WED = date(2026, 9, 23)
THU = date(2026, 9, 24)
FRI = date(2026, 9, 25)
SAT = date(2026, 9, 26)
NEXT_MON = date(2026, 9, 28)

EVERY = timedelta(minutes=10)


def utc(day: date, hour: int, minute: int = 0) -> datetime:
    return datetime(day.year, day.month, day.day, hour, minute, tzinfo=UTC)


class PinnedClock:
    """The ledger's clock, pinned, so an event lands at the instant a test says."""

    def __init__(self) -> None:
        self.at = utc(MON, 12)

    def iso(self) -> str:
        return to_iso(self.at)


def pin_clock(monkeypatch: pytest.MonkeyPatch) -> PinnedClock:
    pinned = PinnedClock()
    monkeypatch.setattr("tb.ledger.store.now_iso", pinned.iso)
    return pinned


def append(
    ledger: Ledger, clock: PinnedClock, at: datetime, event: EventType, payload: Any
) -> None:
    clock.at = at
    ledger.append(event, getattr(payload, "run_id", None) or "subject", payload)


def cycle(
    ledger: Ledger, clock: PinnedClock, run_id: str, at: datetime, n: int, **extra: Any
) -> None:
    append(
        ledger,
        clock,
        at,
        EventType.LOOP_CYCLE_COMPLETED,
        LoopCyclePayload(
            run_id=run_id,
            cycle=n,
            as_of_utc=at.isoformat(),
            n_instruments_considered=1,
            n_decisions=extra.get("decisions", 0),
            n_orders_submitted=extra.get("orders", 0),
            n_risk_refusals=0,
            duration_ms=5.0,
            detail=extra.get("detail", ""),
            n_fills_recorded=extra.get("fills", 0),
        ),
    )


def run(
    ledger: Ledger,
    clock: PinnedClock,
    *,
    run_id: str,
    mode: str = "demo",
    day: date = MON,
    start: tuple[int, int] = (13, 0),
    stop: tuple[int, int] = (20, 0),
    skip: tuple[datetime, datetime] | None = None,
    end: str | None = "completed",
    error_type: str | None = None,
    error_detail: str | None = None,
) -> datetime:
    """A run of the loop: its start, a cycle every ten minutes, and its end.

    Returns the instant of its last cycle.
    """
    at = utc(day, *start)
    clock.at = at
    ledger.record_run_start(run_id=run_id, mode=mode)
    last = at
    n = 0
    while at <= utc(day, *stop):
        if skip is None or not skip[0] <= at < skip[1]:
            n += 1
            cycle(ledger, clock, run_id, at, n)
            last = at
        at += EVERY
    if end is not None:
        clock.at = last + timedelta(seconds=30)
        ledger.record_run_end(
            run_id=run_id, exit_reason=end, error_type=error_type, error_detail=error_detail
        )
    return last
