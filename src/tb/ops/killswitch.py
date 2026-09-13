"""The kill switch, and the heartbeat that backs it.

**Fail-closed.** There are three readings, not two: clear, engaged, and
*undeterminable*. Undeterminable is treated as engaged. A switch we cannot read
is not permission to trade.

Two details here are easy to get wrong and both fail open:

* `pathlib.Path.exists()` swallows `OSError` and returns `False`. So a
  permission error on the switch file — exactly what a misconfigured mount
  produces — would read as "no kill file, carry on". This module calls
  `os.stat` and handles each error explicitly instead.

* A **missing parent directory** is undeterminable, not clear. If `var/run/`
  is not there, whoever tries to engage the switch by touching the file will
  fail, so the mechanism is broken. Reading that as "switch not engaged" means
  the one operation that must always work, silently cannot.
"""

from __future__ import annotations

import contextlib
import json
import os
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path

from tb.core.clock import now_iso, now_utc


class KillSwitchState(StrEnum):
    CLEAR = "clear"
    ENGAGED = "engaged"
    UNDETERMINABLE = "undeterminable"


@dataclass(frozen=True, slots=True)
class KillSwitchReading:
    state: KillSwitchState
    path: Path
    detail: str
    engaged_by: str | None = None
    engaged_at: str | None = None
    reason: str | None = None

    @property
    def may_trade(self) -> bool:
        """Only an affirmatively clear switch permits trading."""
        return self.state is KillSwitchState.CLEAR

    @property
    def determinable(self) -> bool:
        return self.state is not KillSwitchState.UNDETERMINABLE


def read_kill_switch(path: str | Path) -> KillSwitchReading:
    """Read the switch, resolving every error explicitly toward "engaged"."""
    path = Path(path)
    parent = path.parent

    # The mechanism has to be usable for its absence to mean anything.
    try:
        parent_stat = os.stat(parent)
    except FileNotFoundError:
        return KillSwitchReading(
            state=KillSwitchState.UNDETERMINABLE,
            path=path,
            detail=(
                f"kill switch directory {parent} does not exist, so the switch cannot be "
                "engaged by anyone. Treating as engaged: a switch that cannot be thrown "
                "is not a switch. Run `tb init` to create it."
            ),
        )
    except OSError as exc:
        return KillSwitchReading(
            state=KillSwitchState.UNDETERMINABLE,
            path=path,
            detail=f"cannot stat kill switch directory {parent}: {exc}",
        )

    if not os.path.isdir(parent):
        return KillSwitchReading(
            state=KillSwitchState.UNDETERMINABLE,
            path=path,
            detail=f"kill switch parent {parent} is not a directory (mode {parent_stat.st_mode:o})",
        )

    try:
        os.stat(path)
    except FileNotFoundError:
        return KillSwitchReading(
            state=KillSwitchState.CLEAR,
            path=path,
            detail=f"no kill file at {path}",
        )
    except PermissionError as exc:
        return KillSwitchReading(
            state=KillSwitchState.UNDETERMINABLE,
            path=path,
            detail=(
                f"permission denied reading the kill switch at {path}: {exc}. Treating as engaged."
            ),
        )
    except OSError as exc:
        return KillSwitchReading(
            state=KillSwitchState.UNDETERMINABLE,
            path=path,
            detail=f"cannot read the kill switch at {path}: {exc}. Treating as engaged.",
        )

    # Present. Its contents are informational only — an empty or unparseable
    # kill file still kills. Requiring valid JSON to honour a stop would be an
    # absurd way to fail open.
    engaged_by: str | None = None
    engaged_at: str | None = None
    reason: str | None = None
    try:
        body = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(body, dict):
            engaged_by = body.get("engaged_by")
            engaged_at = body.get("engaged_at")
            reason = body.get("reason")
    except (OSError, ValueError, UnicodeDecodeError):
        pass

    return KillSwitchReading(
        state=KillSwitchState.ENGAGED,
        path=path,
        detail=f"kill switch is engaged at {path}",
        engaged_by=engaged_by,
        engaged_at=engaged_at,
        reason=reason,
    )


def engage_kill_switch(path: str | Path, *, engaged_by: str, reason: str) -> KillSwitchReading:
    """Throw the switch. Idempotent — engaging an engaged switch is a no-op."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    existing = read_kill_switch(path)
    if existing.state is KillSwitchState.ENGAGED:
        return existing

    payload = {"engaged_by": engaged_by, "engaged_at": now_iso(), "reason": reason}
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    # Atomic rename: a half-written kill file must never be a readable one.
    os.replace(tmp, path)
    return read_kill_switch(path)


def release_kill_switch(path: str | Path) -> KillSwitchReading:
    """Release the switch. Idempotent."""
    path = Path(path)
    with contextlib.suppress(FileNotFoundError):
        os.unlink(path)
    return read_kill_switch(path)


# --------------------------------------------------------------------------
# Heartbeat — the other half of the dead-man switch
# --------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class HeartbeatReading:
    path: Path
    exists: bool
    age_seconds: float | None
    stale: bool
    detail: str


def write_heartbeat(path: str | Path, *, run_id: str, state: str) -> None:
    """Record liveness.

    Failure to write propagates. The trader halting itself when it cannot prove
    it is alive is the self-facing half of the dead-man switch: without it, only
    the watchdog notices a stall, and a watchdog can die too.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "run_id": run_id,
        "state": state,
        "pid": os.getpid(),
        "beat_at": now_iso(),
    }
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, sort_keys=True), encoding="utf-8")
    os.replace(tmp, path)


def read_heartbeat(path: str | Path, *, stale_after_seconds: int) -> HeartbeatReading:
    """How long since the trader last proved it was alive.

    A missing heartbeat counts as stale, not as "not started yet". The watchdog
    cannot distinguish the two, and the safe reading of the ambiguity is stale.
    """
    path = Path(path)
    try:
        stat = os.stat(path)
    except FileNotFoundError:
        return HeartbeatReading(
            path=path,
            exists=False,
            age_seconds=None,
            stale=True,
            detail=f"no heartbeat at {path}; treating as stale",
        )
    except OSError as exc:
        return HeartbeatReading(
            path=path,
            exists=False,
            age_seconds=None,
            stale=True,
            detail=f"cannot read heartbeat at {path}: {exc}; treating as stale",
        )

    age = now_utc().timestamp() - stat.st_mtime
    stale = age > stale_after_seconds
    return HeartbeatReading(
        path=path,
        exists=True,
        age_seconds=age,
        stale=stale,
        detail=(
            f"heartbeat is {age:.1f}s old (threshold {stale_after_seconds}s)"
            + (" — STALE" if stale else "")
        ),
    )
