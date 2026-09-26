"""The read routes. Every one is a GET, and every one opens the ledger read-only."""

from __future__ import annotations

from datetime import date
from typing import Annotated, Any, Literal

from fastapi import APIRouter, Depends, HTTPException, Query
from fastapi.responses import PlainTextResponse

from tb.api import views
from tb.api.deps import DashboardSettings, reading, require_reader

router = APIRouter(prefix="/api")

Reader = Annotated[DashboardSettings, Depends(require_reader)]
Mode = Annotated[Literal["paper", "demo", "live"], Query()]


@router.get("/status")
def get_status(settings: Reader) -> dict[str, Any]:
    with reading(settings) as ledger:
        body = views.status(ledger, settings.pinned, now=settings.clock())
    body["control"] = {"kill_switch_button": settings.button_enabled}
    return body


@router.get("/equity")
def get_equity(settings: Reader, mode: Mode = "demo") -> dict[str, Any]:
    with reading(settings) as ledger:
        return views.equity(ledger, mode=mode, now=settings.clock())


@router.get("/risk")
def get_risk(settings: Reader, mode: Mode = "demo") -> dict[str, Any]:
    with reading(settings) as ledger:
        return views.risk(ledger, settings.pinned, mode=mode, now=settings.clock())


@router.get("/strategies")
def get_strategies(settings: Reader, mode: Mode = "demo") -> list[dict[str, Any]]:
    with reading(settings) as ledger:
        return views.strategies(ledger, mode=mode)


@router.get("/sessions")
def get_sessions(
    settings: Reader,
    mode: Mode = "demo",
    limit: Annotated[int, Query(ge=1, le=366)] = 30,
) -> dict[str, Any]:
    with reading(settings) as ledger:
        return views.sessions(ledger, settings.pinned, mode=mode, limit=limit, now=settings.clock())


@router.get("/events")
def get_events(
    settings: Reader,
    after: Annotated[int, Query(ge=0)] = 0,
    limit: Annotated[int, Query(ge=1, le=500)] = 50,
) -> list[dict[str, Any]]:
    with reading(settings) as ledger:
        return views.events(ledger, after=after, limit=limit)


@router.get("/journal")
def get_journal(settings: Reader) -> list[dict[str, Any]]:
    return views.journal_pages(settings.journal_dir)


@router.get("/journal/{day}", response_class=PlainTextResponse)
def get_journal_page(settings: Reader, day: date) -> str:
    """One page, as the text on disk.

    The file name is built from the parsed date, never from the request's
    text, so no path the client writes reaches the filesystem. It goes back as
    plain text for the page to show as text: rendering the markdown to HTML
    would turn a ticker or a model's words recorded in the ledger into markup.
    """
    path = settings.journal_dir / f"{day.isoformat()}.md"
    try:
        return path.read_text(encoding="utf-8")
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=f"no journal page for {day}") from exc
