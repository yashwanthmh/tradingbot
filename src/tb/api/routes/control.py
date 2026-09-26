"""The one control: engage the kill switch. There is no route that releases it.

Engaging is the direction that cannot lose money — it stops new trading, and
the loop's halt still lets protective stops and exits through — so it is the
one write a browser may make. Releasing it is a judgement about why trading
stopped and whether that is over, and it stays with `tb resume` at a terminal,
with a reason on the record.

The request is checked in the order that leaks least: who is asking, then
whether a browser on another site sent it, then what it says. The switch file
is thrown before the ledger is touched, because the file is what stops the
loop; a ledger too busy to record the moment must not delay it.
"""

from __future__ import annotations

import sqlite3
from typing import Any

from fastapi import APIRouter, HTTPException, Request
from fastapi.concurrency import run_in_threadpool
from pydantic import BaseModel, ConfigDict, Field, ValidationError

from tb.api.deps import DashboardSettings, require_operator
from tb.core.errors import TbError
from tb.core.ids import new_run_id
from tb.ledger.events import Actor
from tb.ledger.schema import SchemaDriftError
from tb.ledger.store import Ledger
from tb.ops.killswitch import KillSwitchState, engage_kill_switch, read_kill_switch
from tb.ops.state import StateMachine

router = APIRouter(prefix="/api")

MAX_BODY_BYTES = 4096


class EngageRequest(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    reason: str = Field(min_length=1, max_length=500)


def _same_origin(request: Request) -> None:
    """Refuse what a page on another site sent.

    The token already stops a forged request — a browser will not attach it
    across origins without a preflight nobody here answers — so this is the
    second lock on the same door, and it costs nothing.
    """
    site = request.headers.get("sec-fetch-site")
    if site is not None and site not in ("same-origin", "none"):
        raise HTTPException(
            status_code=403, detail="the kill switch takes same-origin requests only"
        )
    origin = request.headers.get("origin")
    if origin is not None and origin not in (
        f"http://{request.headers.get('host', '')}",
        f"https://{request.headers.get('host', '')}",
    ):
        raise HTTPException(
            status_code=403, detail="the kill switch takes same-origin requests only"
        )


def engage(settings: DashboardSettings, *, reason: str, engaged_by: str) -> dict[str, Any]:
    """Throw the switch, then record it as `tb halt` does.

    Idempotent: a switch already engaged is left as it is and nothing new is
    recorded, so a double click is one halt, not two.
    """
    path = settings.pinned.limits.safety.kill_switch_path
    before = read_kill_switch(path)
    if before.state is KillSwitchState.ENGAGED:
        return {
            "engaged": True,
            "already_engaged": True,
            "recorded": False,
            "halt_id": None,
            "detail": before.detail,
            "problem": None,
        }
    try:
        switch = engage_kill_switch(path, engaged_by=engaged_by, reason=reason)
    except OSError as exc:
        raise HTTPException(
            status_code=500,
            detail="the kill switch could not be engaged from here: run `tb halt` now",
        ) from exc
    if switch.state is not KillSwitchState.ENGAGED:
        raise HTTPException(
            status_code=500,
            detail=f"the kill switch reads {switch.state.value} after engaging: run `tb halt` now",
        )
    halt_id: str | None = None
    problem: str | None = None
    try:
        with Ledger(settings.db, config_hash=settings.pinned.config_hash) as ledger:
            machine = StateMachine(ledger, settings.pinned, run_id=new_run_id())
            machine.record_kill_switch(
                engaged=True,
                path=str(switch.path),
                determinable=switch.determinable,
                detail=switch.detail,
                engaged_by=engaged_by,
            )
            halt_id = machine.raise_halt("manual", reason, actor=Actor.HUMAN)
    except (TbError, sqlite3.Error, SchemaDriftError) as exc:
        # Opening a writable ledger applies the schema, which can fail as
        # sqlite or as drift before any of this code's own errors could.
        problem = (
            f"the switch is engaged, but recording it in the ledger failed "
            f"({type(exc).__name__}); run `tb halt` with the same reason to put it on the record"
        )
    return {
        "engaged": True,
        "already_engaged": False,
        "recorded": halt_id is not None,
        "halt_id": halt_id,
        "detail": switch.detail,
        "problem": problem,
    }


@router.post("/killswitch")
async def post_killswitch(request: Request) -> dict[str, Any]:
    settings = require_operator(request)
    _same_origin(request)
    kind = request.headers.get("content-type", "").split(";")[0].strip().lower()
    if kind != "application/json":
        raise HTTPException(status_code=415, detail="send the reason as application/json")
    raw = await request.body()
    if len(raw) > MAX_BODY_BYTES:
        raise HTTPException(status_code=413, detail="the request is too large")
    try:
        body = EngageRequest.model_validate_json(raw)
    except ValidationError as exc:
        raise HTTPException(
            status_code=422, detail='send {"reason": "..."}: why you are stopping it'
        ) from exc
    peer = request.client.host if request.client else "unknown"
    return await run_in_threadpool(
        engage, settings, reason=body.reason, engaged_by=f"dashboard ({peer})"
    )
