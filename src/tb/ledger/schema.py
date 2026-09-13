"""Ledger DDL.

Hand-written SQL against stdlib `sqlite3` rather than an ORM. For an
append-only log whose integrity is the product, generated SQL is the wrong
trade: the triggers below *are* the safety property, and they should be
readable by anyone auditing this system without knowing an ORM's dialect
quirks. Statements are plain enough to port to Postgres when the ledger
outgrows a single file.

Three triggers do the load-bearing work:

* `event_log_no_update` / `event_log_no_delete` make the table append-only at
  the storage layer, so a bug — or an agent editing its own history — cannot
  quietly rewrite the past even with a live connection.
* `event_log_seq_contiguous` forbids gaps, which means a deleted row cannot be
  papered over by re-inserting at the same sequence number.
* `event_log_chain_link` refuses an insert whose `prev_hash` is not the current
  head. The database itself will not accept a broken chain, so a writer bug
  fails loudly at the point of the mistake instead of producing a log that only
  fails verification weeks later.
"""

from __future__ import annotations

import sqlite3
from pathlib import Path

from tb.core.canonical import GENESIS_HASH

LEDGER_SCHEMA_VERSION = 1

# --------------------------------------------------------------------------
# Tables
# --------------------------------------------------------------------------

_TABLES: tuple[str, ...] = (
    # The single source of truth. Every other table is a projection of this one.
    """
    CREATE TABLE IF NOT EXISTS event_log (
        seq             INTEGER PRIMARY KEY,
        ts_utc          TEXT    NOT NULL,
        ts_mono         REAL,
        event_type      TEXT    NOT NULL,
        aggregate_type  TEXT    NOT NULL,
        aggregate_id    TEXT    NOT NULL,
        actor           TEXT    NOT NULL,
        payload_json    TEXT    NOT NULL,
        payload_hash    TEXT    NOT NULL,
        prev_hash       TEXT    NOT NULL,
        chain_hash      TEXT    NOT NULL UNIQUE,
        schema_version  INTEGER NOT NULL,
        run_id          TEXT,
        code_git_sha    TEXT,
        config_hash     TEXT
    )
    """,
    # The external trust boundary. Without these rows the chain proves nothing
    # against the process that wrote it.
    """
    CREATE TABLE IF NOT EXISTS chain_anchor (
        anchor_id     TEXT    PRIMARY KEY,
        seq           INTEGER NOT NULL,
        chain_hash    TEXT    NOT NULL,
        anchored_at   TEXT    NOT NULL,
        sink          TEXT    NOT NULL,
        external_ref  TEXT,
        note          TEXT
    )
    """,
    # Full limit values, not just the hash: "what were the caps when this trade
    # happened" must be answerable without the git history of a file.
    """
    CREATE TABLE IF NOT EXISTS config_versions (
        config_hash            TEXT    PRIMARY KEY,
        canonical_hash         TEXT    NOT NULL,
        kind                   TEXT    NOT NULL,
        source_path            TEXT    NOT NULL,
        schema_version         INTEGER NOT NULL,
        currency               TEXT,
        immutability_enforced  INTEGER NOT NULL,
        immutability_detail    TEXT,
        first_seen_at          TEXT    NOT NULL,
        values_json            TEXT    NOT NULL
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS runs (
        run_id        TEXT    PRIMARY KEY,
        mode          TEXT    NOT NULL,
        started_at    TEXT    NOT NULL,
        ended_at      TEXT,
        code_git_sha  TEXT,
        code_version  TEXT,
        config_hash   TEXT,
        rng_seed_root INTEGER,
        host          TEXT,
        pid           INTEGER,
        exit_reason   TEXT
    )
    """,
    # Current state. Mutable by design — it is a cache of the latest
    # state.transitioned event, and is rebuildable from event_log.
    """
    CREATE TABLE IF NOT EXISTS run_state (
        id          INTEGER PRIMARY KEY CHECK (id = 1),
        state       TEXT    NOT NULL,
        since       TEXT    NOT NULL,
        reason      TEXT,
        run_id      TEXT,
        event_seq   INTEGER
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS halts (
        halt_id        TEXT    PRIMARY KEY,
        raised_at      TEXT    NOT NULL,
        trigger        TEXT    NOT NULL,
        detail         TEXT,
        run_id         TEXT,
        observed_value TEXT,
        limit_value    TEXT,
        cleared_at     TEXT,
        cleared_by     TEXT,
        clear_reason   TEXT
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS schema_meta (
        key   TEXT PRIMARY KEY,
        value TEXT NOT NULL
    )
    """,
)

# --------------------------------------------------------------------------
# Triggers — the append-only guarantee
# --------------------------------------------------------------------------

_TRIGGERS: tuple[str, ...] = (
    """
    CREATE TRIGGER IF NOT EXISTS event_log_no_update
    BEFORE UPDATE ON event_log
    BEGIN
        SELECT RAISE(ABORT, 'event_log is append-only: UPDATE is forbidden');
    END
    """,
    """
    CREATE TRIGGER IF NOT EXISTS event_log_no_delete
    BEFORE DELETE ON event_log
    BEGIN
        SELECT RAISE(ABORT, 'event_log is append-only: DELETE is forbidden');
    END
    """,
    """
    CREATE TRIGGER IF NOT EXISTS event_log_seq_contiguous
    BEFORE INSERT ON event_log
    BEGIN
        SELECT RAISE(ABORT, 'event_log seq must be exactly one past the head')
        WHERE NEW.seq IS NOT COALESCE((SELECT MAX(seq) FROM event_log), 0) + 1;
    END
    """,
    # GENESIS_HASH is a module constant (64 zeros), not caller input.
    f"""
    CREATE TRIGGER IF NOT EXISTS event_log_chain_link
    BEFORE INSERT ON event_log
    BEGIN
        SELECT RAISE(ABORT, 'event_log chain break: prev_hash is not the current head')
        WHERE NEW.prev_hash IS NOT COALESCE(
            (SELECT chain_hash FROM event_log ORDER BY seq DESC LIMIT 1),
            '{GENESIS_HASH}'
        );
    END
    """,  # noqa: S608 - interpolates a module constant, never user input
)

_INDEXES: tuple[str, ...] = (
    "CREATE INDEX IF NOT EXISTS ix_event_log_type ON event_log (event_type)",
    "CREATE INDEX IF NOT EXISTS ix_event_log_aggregate ON event_log (aggregate_type, aggregate_id)",
    "CREATE INDEX IF NOT EXISTS ix_event_log_run ON event_log (run_id)",
    "CREATE INDEX IF NOT EXISTS ix_event_log_ts ON event_log (ts_utc)",
    "CREATE INDEX IF NOT EXISTS ix_halts_open ON halts (cleared_at)",
    "CREATE INDEX IF NOT EXISTS ix_chain_anchor_seq ON chain_anchor (seq)",
)


def connect(
    path: str | Path, *, read_only: bool = False, timeout: float = 30.0
) -> sqlite3.Connection:
    """Open a ledger connection with the pragmas this system depends on.

    `synchronous=FULL` is deliberate and not a default worth relaxing. An order
    that was recorded and then lost to a power failure is the single worst state
    this system can be in: a real position in the market that no intent in the
    log accounts for. Durability is worth more here than write throughput.
    """
    path = Path(path)
    if read_only:
        uri = f"file:{path}?mode=ro"
        conn = sqlite3.connect(uri, uri=True, timeout=timeout)
    else:
        path.parent.mkdir(parents=True, exist_ok=True)
        conn = sqlite3.connect(path, timeout=timeout)

    conn.row_factory = sqlite3.Row
    # Explicit transaction control: we need BEGIN IMMEDIATE for the read-head/
    # append sequence, and Python's implicit commit behaviour would get in the
    # way of that.
    conn.isolation_level = None

    cur = conn.cursor()
    if not read_only:
        cur.execute("PRAGMA journal_mode=WAL")
        cur.execute("PRAGMA synchronous=FULL")
    cur.execute("PRAGMA foreign_keys=ON")
    cur.execute(f"PRAGMA busy_timeout={int(timeout * 1000)}")
    cur.close()
    return conn


def apply_schema(conn: sqlite3.Connection) -> None:
    """Create tables, triggers and indexes. Idempotent."""
    cur = conn.cursor()
    cur.execute("BEGIN IMMEDIATE")
    try:
        for statement in (*_TABLES, *_TRIGGERS, *_INDEXES):
            cur.execute(statement)
        cur.execute(
            "INSERT INTO schema_meta (key, value) VALUES ('ledger_schema_version', ?) "
            "ON CONFLICT(key) DO UPDATE SET value = excluded.value",
            (str(LEDGER_SCHEMA_VERSION),),
        )
        cur.execute("COMMIT")
    except Exception:
        cur.execute("ROLLBACK")
        raise
    finally:
        cur.close()


def schema_version(conn: sqlite3.Connection) -> int | None:
    """The ledger schema version recorded in this database, if any."""
    try:
        row = conn.execute(
            "SELECT value FROM schema_meta WHERE key = 'ledger_schema_version'"
        ).fetchone()
    except sqlite3.OperationalError:
        return None
    return int(row["value"]) if row else None
