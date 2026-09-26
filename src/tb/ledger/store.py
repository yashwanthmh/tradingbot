"""The append-only event store.

Usage is deliberately transactional. An event and the projection it implies must
land together or not at all — a `halt.raised` event with no row in `halts`, or a
row in `halts` with no event, are both states that make the audit trail lie:

    with ledger.transaction() as tx:
        event = tx.append(EventType.HALT_RAISED, halt_id, payload, actor=Actor.RISK)
        tx.execute("INSERT INTO halts (...) VALUES (...)", params)

`append()` on the ledger itself is a convenience for the single-event case.
"""

from __future__ import annotations

import os
import socket
import sqlite3
import subprocess
import sys
from collections.abc import Collection, Iterator, Sequence
from dataclasses import dataclass
from pathlib import Path
from types import TracebackType
from typing import Any, Self

from tb import __version__
from tb.core.canonical import canonical_json
from tb.core.clock import monotonic, now_iso
from tb.core.errors import ChainIntegrityError, LedgerError, LedgerUnwritableError
from tb.ledger.chain import GENESIS_HASH, compute_chain_hash, compute_payload_hash
from tb.ledger.events import (
    Actor,
    AggregateType,
    EventPayload,
    EventType,
    GenesisPayload,
    default_aggregate,
    payload_model,
)
from tb.ledger.schema import (
    LEDGER_SCHEMA_VERSION,
    apply_schema,
    connect,
    schema_version,
)

DEFAULT_LEDGER_PATH = Path("var/ledger.db")
ENV_VAR_DIR = "TB_VAR_DIR"


@dataclass(frozen=True, slots=True)
class ChainHead:
    """The tip of the chain."""

    seq: int
    chain_hash: str


@dataclass(frozen=True, slots=True)
class AppendedEvent:
    """What was written, as written."""

    seq: int
    ts_utc: str
    event_type: EventType
    aggregate_type: str
    aggregate_id: str
    actor: str
    payload_json: str
    payload_hash: str
    prev_hash: str
    chain_hash: str


def default_ledger_path() -> Path:
    """`$TB_VAR_DIR/ledger.db`, or `var/ledger.db`."""
    var_dir = os.environ.get(ENV_VAR_DIR)
    return Path(var_dir) / "ledger.db" if var_dir else DEFAULT_LEDGER_PATH


def code_git_sha() -> str | None:
    """The current commit, for pinning behaviour to code.

    Best-effort: returns None outside a git checkout rather than failing. A
    dirty tree is reported with a `-dirty` suffix, because "which code produced
    this decision" is only answerable if uncommitted changes are visible.
    """
    try:
        sha = subprocess.run(
            ["git", "rev-parse", "HEAD"],  # noqa: S607 - git resolved from PATH by design
            capture_output=True,
            text=True,
            timeout=5,
            check=False,
        )
        if sha.returncode != 0:
            return None
        head = sha.stdout.strip()
        dirty = subprocess.run(
            ["git", "status", "--porcelain"],  # noqa: S607
            capture_output=True,
            text=True,
            timeout=5,
            check=False,
        )
        if dirty.returncode == 0 and dirty.stdout.strip():
            return f"{head}-dirty"
        return head
    except (OSError, subprocess.SubprocessError):
        return None


class LedgerTransaction:
    """A single write transaction: events plus their projections, atomically."""

    def __init__(self, ledger: Ledger, cursor: sqlite3.Cursor) -> None:
        self._ledger = ledger
        self._cursor = cursor

    def append(
        self,
        event_type: EventType,
        aggregate_id: str,
        payload: EventPayload | dict[str, Any],
        *,
        actor: Actor = Actor.SYSTEM,
        aggregate_type: AggregateType | None = None,
        run_id: str | None = None,
        config_hash: str | None = None,
    ) -> AppendedEvent:
        """Validate, hash, and append one event."""
        model = payload_model(event_type)

        if isinstance(payload, EventPayload):
            if not isinstance(payload, model):
                raise LedgerError(
                    f"{event_type} expects payload {model.__name__}, got {type(payload).__name__}"
                )
            validated: EventPayload = payload
        else:
            try:
                validated = model.model_validate(payload)
            except Exception as exc:
                raise LedgerError(f"invalid payload for {event_type}: {exc}") from exc

        # mode="json" reduces Decimals to strings and datetimes to ISO text, so
        # the dumped dict is already JSON primitives and canonical_json becomes a
        # pure re-serialisation with nothing left to interpret.
        payload_json = canonical_json(validated.model_dump(mode="json"))
        payload_hash = compute_payload_hash(payload_json)

        head = self._ledger._head_locked(self._cursor)
        prev_hash = head.chain_hash if head else GENESIS_HASH
        seq = (head.seq + 1) if head else 1

        ts_utc = now_iso()
        agg_type = (aggregate_type or default_aggregate(event_type)).value

        chain_hash = compute_chain_hash(
            prev_hash=prev_hash,
            seq=seq,
            ts_utc=ts_utc,
            event_type=event_type.value,
            aggregate_type=agg_type,
            aggregate_id=aggregate_id,
            actor=actor.value,
            payload_hash=payload_hash,
        )

        try:
            self._cursor.execute(
                """
                INSERT INTO event_log (
                    seq, ts_utc, ts_mono, event_type, aggregate_type, aggregate_id,
                    actor, payload_json, payload_hash, prev_hash, chain_hash,
                    schema_version, run_id, code_git_sha, config_hash
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    seq,
                    ts_utc,
                    monotonic(),
                    event_type.value,
                    agg_type,
                    aggregate_id,
                    actor.value,
                    payload_json,
                    payload_hash,
                    prev_hash,
                    chain_hash,
                    LEDGER_SCHEMA_VERSION,
                    run_id or self._ledger.run_id,
                    self._ledger.git_sha,
                    config_hash or self._ledger.config_hash,
                ),
            )
        except sqlite3.IntegrityError as exc:
            # The chain-link and contiguity triggers raise here. That is the
            # database refusing to store a broken chain, which is a bug in the
            # writer, not a transient failure to retry.
            raise ChainIntegrityError(
                f"ledger refused an append for {event_type}: {exc}", seq=seq
            ) from exc
        except sqlite3.OperationalError as exc:
            raise LedgerUnwritableError(
                f"cannot append {event_type} to the ledger: {exc}. "
                "Trading must stop: an unrecorded order is a position no intent explains."
            ) from exc

        return AppendedEvent(
            seq=seq,
            ts_utc=ts_utc,
            event_type=event_type,
            aggregate_type=agg_type,
            aggregate_id=aggregate_id,
            actor=actor.value,
            payload_json=payload_json,
            payload_hash=payload_hash,
            prev_hash=prev_hash,
            chain_hash=chain_hash,
        )

    def execute(self, sql: str, params: Sequence[Any] = ()) -> sqlite3.Cursor:
        """Run a projection write inside this transaction."""
        self._cursor.execute(sql, params)
        return self._cursor


class Ledger:
    """The audit ledger."""

    def __init__(
        self,
        path: str | Path | None = None,
        *,
        run_id: str | None = None,
        config_hash: str | None = None,
        read_only: bool = False,
    ) -> None:
        self.path = Path(path) if path is not None else default_ledger_path()
        self.run_id = run_id
        self.config_hash = config_hash
        self.read_only = read_only
        self.git_sha = code_git_sha()
        self._conn: sqlite3.Connection | None = None

    # -- lifecycle ---------------------------------------------------------

    def open(self) -> Self:
        if self._conn is None:
            if self.read_only and not self.path.exists():
                raise LedgerError(f"no ledger at {self.path}")
            try:
                self._conn = connect(self.path, read_only=self.read_only)
            except sqlite3.OperationalError as exc:
                raise LedgerUnwritableError(f"cannot open ledger {self.path}: {exc}") from exc
            if not self.read_only:
                apply_schema(self._conn)
        return self

    def close(self) -> None:
        if self._conn is not None:
            self._conn.close()
            self._conn = None

    def __enter__(self) -> Self:
        return self.open()

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        self.close()

    @property
    def conn(self) -> sqlite3.Connection:
        if self._conn is None:
            raise LedgerError("ledger is not open; use `with Ledger(...) as ledger:`")
        return self._conn

    def exists(self) -> bool:
        return self.path.exists()

    def schema_version(self) -> int | None:
        return schema_version(self.conn)

    # -- writing -----------------------------------------------------------

    def transaction(self) -> _TransactionContext:
        """An exclusive write transaction.

        BEGIN IMMEDIATE takes the write lock up front, so reading the chain head
        and appending to it cannot interleave with another writer. Without it,
        two processes could read the same head and race to claim one sequence
        number — which the contiguity trigger would catch, but only after one of
        them believed it had recorded an order.
        """
        return _TransactionContext(self)

    def append(
        self,
        event_type: EventType,
        aggregate_id: str,
        payload: EventPayload | dict[str, Any],
        *,
        actor: Actor = Actor.SYSTEM,
        aggregate_type: AggregateType | None = None,
        run_id: str | None = None,
        config_hash: str | None = None,
    ) -> AppendedEvent:
        """Append a single event in its own transaction."""
        with self.transaction() as tx:
            return tx.append(
                event_type,
                aggregate_id,
                payload,
                actor=actor,
                aggregate_type=aggregate_type,
                run_id=run_id,
                config_hash=config_hash,
            )

    def initialise(self, *, created_by: str) -> AppendedEvent | None:
        """Write the genesis event if the chain is empty.

        Returns None if the ledger already has events, so this is safe to call
        on every startup.
        """
        if self.head() is not None:
            return None
        return self.append(
            EventType.LEDGER_GENESIS,
            "chain",
            GenesisPayload(
                ledger_schema_version=LEDGER_SCHEMA_VERSION,
                created_by=created_by,
                code_version=__version__,
            ),
        )

    def record_run_start(self, *, run_id: str, mode: str, rng_seed_root: int | None = None) -> None:
        """Open a run: one event plus its `runs` projection, atomically."""
        from tb.ledger.events import RunStartedPayload

        payload = RunStartedPayload(
            run_id=run_id,
            mode=mode,
            code_git_sha=self.git_sha,
            code_version=__version__,
            config_hash=self.config_hash,
            rng_seed_root=rng_seed_root,
            host=socket.gethostname(),
            pid=os.getpid(),
            python_version=sys.version.split()[0],
            argv=list(sys.argv),
        )
        with self.transaction() as tx:
            event = tx.append(EventType.RUN_STARTED, run_id, payload, run_id=run_id)
            tx.execute(
                """
                INSERT INTO runs (
                    run_id, mode, started_at, code_git_sha, code_version,
                    config_hash, rng_seed_root, host, pid
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(run_id) DO NOTHING
                """,
                (
                    run_id,
                    mode,
                    event.ts_utc,
                    self.git_sha,
                    __version__,
                    self.config_hash,
                    rng_seed_root,
                    socket.gethostname(),
                    os.getpid(),
                ),
            )

    def record_run_end(
        self,
        *,
        run_id: str,
        exit_reason: str,
        error_type: str | None = None,
        error_detail: str | None = None,
    ) -> None:
        from tb.ledger.events import RunEndedPayload

        payload = RunEndedPayload(
            run_id=run_id,
            exit_reason=exit_reason,
            error_type=error_type,
            error_detail=error_detail,
        )
        with self.transaction() as tx:
            event = tx.append(EventType.RUN_ENDED, run_id, payload, run_id=run_id)
            tx.execute(
                "UPDATE runs SET ended_at = ?, exit_reason = ? WHERE run_id = ?",
                (event.ts_utc, exit_reason, run_id),
            )

    def record_config_pin(self, audit: dict[str, Any]) -> AppendedEvent:
        """Record the limits in force, with their full values."""
        from tb.ledger.events import ConfigPinnedPayload

        payload = ConfigPinnedPayload(
            config_hash=audit["config_hash"],
            canonical_hash=audit["canonical_hash"],
            kind=audit["kind"],
            source_path=audit["source_path"],
            schema_version=audit["schema_version"],
            currency=audit["currency"],
            immutability_enforced=audit["immutability_enforced"],
            immutability_detail=audit["immutability_detail"],
            values=audit["values"],
        )
        with self.transaction() as tx:
            event = tx.append(
                EventType.CONFIG_PINNED,
                audit["config_hash"],
                payload,
                config_hash=audit["config_hash"],
            )
            tx.execute(
                """
                INSERT INTO config_versions (
                    config_hash, canonical_hash, kind, source_path, schema_version,
                    currency, immutability_enforced, immutability_detail,
                    first_seen_at, values_json
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(config_hash) DO NOTHING
                """,
                (
                    audit["config_hash"],
                    audit["canonical_hash"],
                    audit["kind"],
                    audit["source_path"],
                    audit["schema_version"],
                    audit["currency"],
                    int(audit["immutability_enforced"]),
                    audit["immutability_detail"],
                    event.ts_utc,
                    canonical_json(audit["values"]),
                ),
            )
            return event

    # -- reading -----------------------------------------------------------

    def _head_locked(self, cursor: sqlite3.Cursor) -> ChainHead | None:
        row = cursor.execute(
            "SELECT seq, chain_hash FROM event_log ORDER BY seq DESC LIMIT 1"
        ).fetchone()
        return ChainHead(seq=int(row["seq"]), chain_hash=row["chain_hash"]) if row else None

    def head(self) -> ChainHead | None:
        """The current chain tip, or None for an empty chain."""
        row = self.conn.execute(
            "SELECT seq, chain_hash FROM event_log ORDER BY seq DESC LIMIT 1"
        ).fetchone()
        return ChainHead(seq=int(row["seq"]), chain_hash=row["chain_hash"]) if row else None

    def count(self) -> int:
        row = self.conn.execute("SELECT COUNT(*) AS n FROM event_log").fetchone()
        return int(row["n"])

    def get(self, seq: int) -> sqlite3.Row | None:
        row: sqlite3.Row | None = self.conn.execute(
            "SELECT * FROM event_log WHERE seq = ?", (seq,)
        ).fetchone()
        return row

    def iter_events(
        self,
        *,
        start_seq: int = 1,
        end_seq: int | None = None,
        event_type: EventType | None = None,
        event_types: Collection[EventType] | None = None,
        batch_size: int = 1000,
    ) -> Iterator[sqlite3.Row]:
        """Stream events in sequence order.

        Batched rather than `fetchall`, because verification has to walk a
        ledger that will eventually be far larger than memory. `event_types`
        narrows to several types in one ordered pass, for a reader that needs
        how they interleave.
        """
        sql = "SELECT * FROM event_log WHERE seq >= ?"
        params: list[Any] = [start_seq]
        if end_seq is not None:
            sql += " AND seq <= ?"
            params.append(end_seq)
        if event_type is not None:
            sql += " AND event_type = ?"
            params.append(event_type.value)
        if event_types is not None:
            wanted = sorted({t.value for t in event_types})
            if not wanted:
                return
            sql += f" AND event_type IN ({', '.join('?' for _ in wanted)})"
            params.extend(wanted)
        sql += " ORDER BY seq ASC"

        cursor = self.conn.execute(sql, params)
        try:
            while True:
                rows = cursor.fetchmany(batch_size)
                if not rows:
                    return
                yield from rows
        finally:
            cursor.close()

    def tail(self, limit: int = 20) -> list[sqlite3.Row]:
        rows = self.conn.execute(
            "SELECT * FROM event_log ORDER BY seq DESC LIMIT ?", (limit,)
        ).fetchall()
        return list(reversed(rows))

    def anchors(self) -> list[sqlite3.Row]:
        return list(self.conn.execute("SELECT * FROM chain_anchor ORDER BY seq ASC").fetchall())


class _TransactionContext:
    """Context manager wrapping BEGIN IMMEDIATE / COMMIT / ROLLBACK."""

    def __init__(self, ledger: Ledger) -> None:
        self._ledger = ledger
        self._cursor: sqlite3.Cursor | None = None

    def __enter__(self) -> LedgerTransaction:
        if self._ledger.read_only:
            raise LedgerUnwritableError("ledger was opened read-only")
        cursor = self._ledger.conn.cursor()
        try:
            cursor.execute("BEGIN IMMEDIATE")
        except sqlite3.OperationalError as exc:
            cursor.close()
            raise LedgerUnwritableError(
                f"cannot acquire the ledger write lock: {exc}. Another writer may be "
                "holding it, or the filesystem is read-only."
            ) from exc
        self._cursor = cursor
        return LedgerTransaction(self._ledger, cursor)

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        cursor = self._cursor
        if cursor is None:  # pragma: no cover - __enter__ always sets it
            return
        try:
            if exc_type is None:
                cursor.execute("COMMIT")
            else:
                cursor.execute("ROLLBACK")
        finally:
            cursor.close()
            self._cursor = None
