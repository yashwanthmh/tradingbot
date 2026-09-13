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

LEDGER_SCHEMA_VERSION = 3

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
    # ---------------------------------------------------------------- v3 (M2)
    # The catalog. A Parquet file is part of the dataset only because a row here
    # (and the event that wrote it) says so — never because it happens to be on
    # disk. That inversion is what makes orphan files ignorable garbage and a
    # missing file a loud integrity failure instead of silent data loss.
    """
    CREATE TABLE IF NOT EXISTS data_partitions (
        file_sha256       TEXT    PRIMARY KEY,
        relative_path     TEXT    NOT NULL,
        instrument_uid    TEXT    NOT NULL,
        resolution        TEXT    NOT NULL,
        provider          TEXT    NOT NULL,
        first_bar_open    TEXT    NOT NULL,
        last_bar_open     TEXT    NOT NULL,
        row_count         INTEGER NOT NULL,
        byte_size         INTEGER NOT NULL,
        rows_hash         TEXT    NOT NULL,
        sealed_at         TEXT    NOT NULL,
        sealing_event_seq INTEGER NOT NULL,
        superseded_by     TEXT,
        schema_version    INTEGER NOT NULL
    )
    """,
    # Incremental bars land here, inside the same transaction as their
    # provenance event, and are compacted into Parquet at session close. Writing
    # a Parquet file per poll would produce thousands of tiny files; writing to
    # the ledger keeps the event and the data atomic, which is the property that
    # matters more than either.
    #
    # Note the three time axes. `available_at_utc` is stored rather than derived
    # because the provider's delay is itself a revisable fact, and the whole
    # point is to know what was knowable *then*, not what we would compute now.
    """
    CREATE TABLE IF NOT EXISTS bars_hot (
        instrument_uid    TEXT    NOT NULL,
        resolution        TEXT    NOT NULL,
        bar_open_utc      TEXT    NOT NULL,
        available_at_utc  TEXT    NOT NULL,
        ingested_at_utc   TEXT    NOT NULL,
        provider          TEXT    NOT NULL,
        provenance        TEXT    NOT NULL,
        session           TEXT    NOT NULL,
        open_scaled       INTEGER NOT NULL,
        high_scaled       INTEGER NOT NULL,
        low_scaled        INTEGER NOT NULL,
        close_scaled      INTEGER NOT NULL,
        volume            INTEGER,
        price_scale       INTEGER NOT NULL,
        currency          TEXT,
        row_hash          TEXT    NOT NULL,
        sealed_into       TEXT,
        PRIMARY KEY (instrument_uid, resolution, bar_open_utc, ingested_at_utc, provider)
    )
    """,
    # Every observed restatement, kept forever. Yahoo silently back-adjusts
    # history, so a bar fetched today can differ from the same bar fetched
    # tomorrow; without this table that difference is invisible and every
    # feature hash computed over the old value becomes a quiet lie.
    """
    CREATE TABLE IF NOT EXISTS bar_revisions (
        revision_id       TEXT    PRIMARY KEY,
        instrument_uid    TEXT    NOT NULL,
        resolution        TEXT    NOT NULL,
        bar_open_utc      TEXT    NOT NULL,
        provider          TEXT    NOT NULL,
        kind              TEXT    NOT NULL,
        first_seen_at     TEXT    NOT NULL,
        first_values_json TEXT,
        revised_at        TEXT    NOT NULL,
        new_values_json   TEXT,
        delta_bps         REAL,
        detected_by       TEXT    NOT NULL
    )
    """,
    # Dated, revisable facts. Factors are DERIVED from these and never stored:
    # a materialised factor column is how a table ends up silently containing
    # tomorrow's split.
    """
    CREATE TABLE IF NOT EXISTS corporate_actions (
        action_id         TEXT    PRIMARY KEY,
        instrument_uid    TEXT    NOT NULL,
        action_type       TEXT    NOT NULL,
        effective_date    TEXT    NOT NULL,
        known_at_utc      TEXT    NOT NULL,
        declared_date     TEXT,
        ratio_num         INTEGER,
        ratio_den         INTEGER,
        gross_amount      TEXT,
        currency          TEXT,
        new_symbol        TEXT,
        source_provider   TEXT    NOT NULL,
        payload_hash      TEXT,
        superseded_by     TEXT,
        reconciled_with_broker INTEGER NOT NULL DEFAULT 0,
        reconcile_note    TEXT
    )
    """,
    # Append-only dated membership. Survivorship bias cannot be fixed
    # retroactively on free data, only stopped from growing: without these rows
    # every backtest silently runs over the names that still exist today.
    """
    CREATE TABLE IF NOT EXISTS universe_snapshots (
        snapshot_id       TEXT    NOT NULL,
        taken_at          TEXT    NOT NULL,
        instrument_uid    TEXT    NOT NULL,
        t212_ticker       TEXT,
        data_symbol       TEXT,
        rank              INTEGER,
        selection_reason  TEXT,
        dollar_volume     TEXT,
        currency          TEXT,
        PRIMARY KEY (snapshot_id, instrument_uid)
    )
    """,
    # Limits are GBP; the universe is USD. Without a point-in-time rate there is
    # no sizing against absolute_ceiling_ccy and no P&L in the account currency.
    """
    CREATE TABLE IF NOT EXISTS fx_rates (
        pair              TEXT    NOT NULL,
        as_of_date        TEXT    NOT NULL,
        available_at_utc  TEXT    NOT NULL,
        ingested_at_utc   TEXT    NOT NULL,
        rate              TEXT    NOT NULL,
        provider          TEXT    NOT NULL,
        provenance        TEXT    NOT NULL,
        PRIMARY KEY (pair, as_of_date, ingested_at_utc)
    )
    """,
    # A sealed vintage is M3's entire input. A backtest whose vintage_id is not
    # here is not admissible evidence for promotion.
    """
    CREATE TABLE IF NOT EXISTS data_snapshots (
        vintage_id              TEXT PRIMARY KEY,
        as_of_utc               TEXT NOT NULL,
        manifest_hash           TEXT NOT NULL,
        window_start            TEXT,
        window_end              TEXT,
        resolutions             TEXT NOT NULL,
        instrument_uids         TEXT NOT NULL,
        file_sha256s            TEXT NOT NULL,
        row_count               INTEGER NOT NULL,
        calendar_hash           TEXT,
        action_table_hash       TEXT,
        fx_table_hash           TEXT,
        universe_snapshot_id    TEXT,
        provider_delays_json    TEXT,
        sealed_from             TEXT,
        first_live_observation_at TEXT,
        survivorship_flag       TEXT NOT NULL,
        pit_completeness_flag   TEXT NOT NULL,
        sealing_event_seq       INTEGER NOT NULL
    )
    """,
    # A rebuildable projection, stated as such. If it ever disagrees with what
    # the events say, the events win and this gets rebuilt.
    """
    CREATE TABLE IF NOT EXISTS bar_coverage (
        instrument_uid    TEXT    NOT NULL,
        resolution        TEXT    NOT NULL,
        provider          TEXT    NOT NULL,
        first_bar_open    TEXT,
        last_bar_open     TEXT,
        row_count         INTEGER NOT NULL,
        n_gaps_unexplained INTEGER NOT NULL DEFAULT 0,
        last_audited_at   TEXT,
        PRIMARY KEY (instrument_uid, resolution, provider)
    )
    """,
    # What each provider's delay and coverage actually turned out to be. The
    # delay in the code is a conservative default; this is the observation.
    """
    CREATE TABLE IF NOT EXISTS provider_observations (
        provider          TEXT    NOT NULL,
        resolution        TEXT    NOT NULL,
        observed_at       TEXT    NOT NULL,
        delay_p50_s       REAL,
        delay_p95_s       REAL,
        delay_max_s       REAL,
        n_samples         INTEGER NOT NULL,
        live_capable      INTEGER,
        note              TEXT,
        PRIMARY KEY (provider, resolution, observed_at)
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
    # v3 (M2)
    "CREATE INDEX IF NOT EXISTS ix_bars_hot_lookup "
    "ON bars_hot (instrument_uid, resolution, bar_open_utc)",
    "CREATE INDEX IF NOT EXISTS ix_bars_hot_unsealed ON bars_hot (sealed_into)",
    "CREATE INDEX IF NOT EXISTS ix_bars_hot_available ON bars_hot (available_at_utc)",
    "CREATE INDEX IF NOT EXISTS ix_data_partitions_lookup "
    "ON data_partitions (instrument_uid, resolution, first_bar_open)",
    "CREATE INDEX IF NOT EXISTS ix_data_partitions_live ON data_partitions (superseded_by)",
    "CREATE INDEX IF NOT EXISTS ix_actions_lookup "
    "ON corporate_actions (instrument_uid, effective_date)",
    "CREATE INDEX IF NOT EXISTS ix_actions_known ON corporate_actions (known_at_utc)",
    "CREATE INDEX IF NOT EXISTS ix_bar_revisions_bar "
    "ON bar_revisions (instrument_uid, resolution, bar_open_utc)",
    "CREATE INDEX IF NOT EXISTS ix_universe_snapshots_taken ON universe_snapshots (taken_at)",
    "CREATE INDEX IF NOT EXISTS ix_fx_lookup ON fx_rates (pair, as_of_date)",
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
