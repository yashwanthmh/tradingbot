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

LEDGER_SCHEMA_VERSION = 2

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
    # ---------------------------------------------------------------- v2 (M1)
    # Every broker response, raw. Trading 212's API is in beta and its docs are
    # not reliably reachable, so the archive is how a shape change six weeks
    # from now gets diagnosed and replayed instead of guessed at. `parse_error`
    # is populated on drift, which makes this table the forensic record.
    """
    CREATE TABLE IF NOT EXISTS broker_messages (
        msg_id             TEXT    PRIMARY KEY,
        run_id             TEXT,
        intent_id          TEXT,
        environment        TEXT    NOT NULL,
        endpoint           TEXT    NOT NULL,
        method             TEXT    NOT NULL,
        url_path           TEXT    NOT NULL,
        status_code        INTEGER,
        ratelimit_json     TEXT,
        request_json       TEXT,
        raw_body           TEXT,
        received_at        TEXT    NOT NULL,
        duration_ms        REAL,
        parse_ok           INTEGER NOT NULL,
        parse_error        TEXT
    )
    """,
    # Instrument metadata, cached because /instruments is rate limited to about
    # one call per fifty seconds and returns a very large payload.
    """
    CREATE TABLE IF NOT EXISTS instruments (
        ticker             TEXT    PRIMARY KEY,
        instrument_type    TEXT,
        isin               TEXT,
        currency_code      TEXT,
        short_name         TEXT,
        full_name          TEXT,
        exchange_id        INTEGER,
        working_schedule_id INTEGER,
        min_trade_quantity TEXT,
        max_open_quantity  TEXT,
        added_on           TEXT,
        fetched_at         TEXT    NOT NULL,
        raw_json           TEXT
    )
    """,
    # The bridge between the execution venue and the data venue. A mismapping
    # here produces a perfectly valid-looking signal filled on the wrong
    # instrument, so the mapping is stored, audited and versioned rather than
    # derived on the fly from a string transform.
    """
    CREATE TABLE IF NOT EXISTS symbol_map (
        t212_ticker        TEXT    PRIMARY KEY,
        data_symbol        TEXT    NOT NULL,
        provider           TEXT    NOT NULL,
        currency_code      TEXT,
        exchange_hint      TEXT,
        confidence         TEXT    NOT NULL,
        derivation         TEXT    NOT NULL,
        verified_at        TEXT,
        last_disagreement_bps REAL,
        last_checked_at    TEXT,
        blocked            INTEGER NOT NULL DEFAULT 0,
        blocked_reason     TEXT
    )
    """,
    # Both the broker's view and ours, side by side, with the divergence
    # between the data feed and the broker's own quote recorded per symbol.
    """
    CREATE TABLE IF NOT EXISTS positions_snapshot (
        snap_id            TEXT    NOT NULL,
        run_id             TEXT,
        ts                 TEXT    NOT NULL,
        source             TEXT    NOT NULL,
        ticker             TEXT    NOT NULL,
        quantity           TEXT    NOT NULL,
        average_price      TEXT,
        current_price_broker TEXT,
        current_price_data TEXT,
        price_disagreement_bps REAL,
        ppl                TEXT,
        initial_fill_date  TEXT,
        raw_json           TEXT,
        PRIMARY KEY (snap_id, ticker)
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS cash_snapshot (
        snap_id   TEXT PRIMARY KEY,
        run_id    TEXT,
        ts        TEXT NOT NULL,
        currency  TEXT,
        free      TEXT,
        total     TEXT,
        invested  TEXT,
        ppl       TEXT,
        result    TEXT,
        blocked   TEXT,
        pie_cash  TEXT,
        raw_json  TEXT
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS reconciliations (
        recon_id                TEXT PRIMARY KEY,
        run_id                  TEXT,
        started_at              TEXT NOT NULL,
        finished_at             TEXT,
        verdict                 TEXT,
        n_unknown_intents       INTEGER NOT NULL DEFAULT 0,
        n_orphan_orders         INTEGER NOT NULL DEFAULT 0,
        n_position_mismatches   INTEGER NOT NULL DEFAULT 0,
        n_unprotected_positions INTEGER NOT NULL DEFAULT 0,
        n_price_disagreements   INTEGER NOT NULL DEFAULT 0,
        findings_json           TEXT,
        actions_json            TEXT
    )
    """,
    # What the probe actually observed, per endpoint. The rate limits in the
    # code are conservative guesses until this table says otherwise.
    """
    CREATE TABLE IF NOT EXISTS endpoint_observations (
        endpoint          TEXT PRIMARY KEY,
        environment       TEXT NOT NULL,
        observed_limit    INTEGER,
        observed_period_s INTEGER,
        configured_limit  INTEGER,
        configured_period_s INTEGER,
        agrees            INTEGER,
        last_status       INTEGER,
        last_seen_at      TEXT NOT NULL,
        note              TEXT
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
    # v2 (M1)
    "CREATE INDEX IF NOT EXISTS ix_broker_messages_endpoint "
    "ON broker_messages (endpoint, received_at)",
    "CREATE INDEX IF NOT EXISTS ix_broker_messages_intent ON broker_messages (intent_id)",
    "CREATE INDEX IF NOT EXISTS ix_broker_messages_drift "
    "ON broker_messages (parse_ok, received_at)",
    "CREATE INDEX IF NOT EXISTS ix_positions_snapshot_ts ON positions_snapshot (ts)",
    "CREATE INDEX IF NOT EXISTS ix_symbol_map_blocked ON symbol_map (blocked)",
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
