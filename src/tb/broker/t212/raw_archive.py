"""Archiving every broker response.

Three reasons this is worth the disk, all of them things that will actually
happen with a beta API whose documentation is not reliably reachable:

1. **Diagnosis.** When a field changes shape, the archived body says exactly
   what arrived. Without it, "the portfolio stopped parsing" is where the
   investigation begins and ends.
2. **Replay and repair.** A drift can be fixed against real payloads, and the
   schema-drift test matrix replays archived bodies rather than fixtures
   somebody invented.
3. **Attribution.** When a position exists that no intent explains, the archive
   is the record of what the broker actually told us and when.

Rows are written for every response including failures — a 429 or a 500 is
exactly the kind of thing worth having later. Events are *not* written per
response; that would bury the ledger under thousands of routine reads. Only
notable things (drift, rate limiting, snapshots) become events.

Nothing here ever stores the API key. Request bodies and query parameters are
scrubbed of it before writing, because an archive that leaks a credential is
worse than no archive.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from tb.core.canonical import canonical_json
from tb.core.clock import now_iso
from tb.core.ids import new_id
from tb.ledger.events import Actor, BrokerSchemaDriftPayload, EventType
from tb.ledger.store import Ledger

# Bodies larger than this are truncated with a marker. The instruments endpoint
# returns megabytes; keeping every copy would grow the ledger without adding
# diagnostic value beyond the first few kilobytes.
MAX_BODY_CHARS = 64_000


@dataclass(frozen=True, slots=True)
class ArchivedMessage:
    msg_id: str
    endpoint: str
    status_code: int | None
    parse_ok: bool


class RawArchive:
    """Writes broker responses into `broker_messages`."""

    def __init__(
        self,
        ledger: Ledger | None,
        *,
        environment: str,
        run_id: str | None = None,
        redact_values: tuple[str, ...] = (),
    ) -> None:
        self._ledger = ledger
        self._environment = environment
        self._run_id = run_id
        # The API key, so it can be scrubbed if it ever appears in a body.
        self._redact = tuple(v for v in redact_values if v)

    def _scrub(self, text: str | None) -> str | None:
        if text is None:
            return None
        for secret in self._redact:
            text = text.replace(secret, "[REDACTED]")
        return text

    def record(
        self,
        *,
        endpoint: str,
        method: str,
        url_path: str,
        status_code: int | None,
        raw_body: str | None,
        ratelimit: dict[str, Any] | None = None,
        request_body: Any = None,
        params: Any = None,
        duration_ms: float | None = None,
        parse_ok: bool = True,
        parse_error: str | None = None,
        intent_id: str | None = None,
    ) -> ArchivedMessage:
        msg_id = new_id("msg", length=16)

        if self._ledger is not None:
            body = self._scrub(raw_body)
            if body is not None and len(body) > MAX_BODY_CHARS:
                body = (
                    body[:MAX_BODY_CHARS]
                    + f"\n[truncated: {len(body) - MAX_BODY_CHARS} more characters]"
                )

            request_repr: str | None = None
            if request_body is not None or params is not None:
                request_repr = self._scrub(canonical_json({"body": request_body, "params": params}))

            self._ledger.conn.execute(
                """
                INSERT INTO broker_messages (
                    msg_id, run_id, intent_id, environment, endpoint, method, url_path,
                    status_code, ratelimit_json, request_json, raw_body, received_at,
                    duration_ms, parse_ok, parse_error
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    msg_id,
                    self._run_id,
                    intent_id,
                    self._environment,
                    endpoint,
                    method,
                    url_path,
                    status_code,
                    None if ratelimit is None else canonical_json(ratelimit),
                    request_repr,
                    body,
                    now_iso(),
                    duration_ms,
                    int(parse_ok),
                    self._scrub(parse_error),
                ),
            )
            self._ledger.conn.commit()

        return ArchivedMessage(
            msg_id=msg_id,
            endpoint=endpoint,
            status_code=status_code,
            parse_ok=parse_ok,
        )

    def record_drift(
        self,
        *,
        endpoint: str,
        url_path: str,
        msg_id: str,
        model: str,
        detail: str,
        status_code: int | None = None,
    ) -> None:
        """Emit a drift event.

        A consumed field changing shape is rare and serious, so unlike routine
        responses it earns a place in the event chain rather than only the
        archive.
        """
        if self._ledger is None:
            return
        self._ledger.append(
            EventType.BROKER_SCHEMA_DRIFT,
            endpoint,
            BrokerSchemaDriftPayload(
                endpoint=endpoint,
                url_path=url_path,
                msg_id=msg_id,
                model=model,
                error_detail=detail,
                status_code=status_code,
            ),
            actor=Actor.BROKER,
        )

    def recent_drift(self, limit: int = 20) -> list[dict[str, Any]]:
        """Responses that failed to parse, newest first."""
        if self._ledger is None:
            return []
        rows = self._ledger.conn.execute(
            "SELECT msg_id, endpoint, url_path, status_code, parse_error, received_at "
            "FROM broker_messages WHERE parse_ok = 0 ORDER BY received_at DESC LIMIT ?",
            (limit,),
        ).fetchall()
        return [dict(row) for row in rows]

    def body_for(self, msg_id: str) -> str | None:
        """The archived body, for replay."""
        if self._ledger is None:
            return None
        row = self._ledger.conn.execute(
            "SELECT raw_body FROM broker_messages WHERE msg_id = ?", (msg_id,)
        ).fetchone()
        return None if row is None else row["raw_body"]
