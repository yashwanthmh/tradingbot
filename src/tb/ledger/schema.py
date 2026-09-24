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

LEDGER_SCHEMA_VERSION = 8

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
    # Which provider symbol each instrument was fetched under. NOT `symbol_map`,
    # which maps a Trading 212 ticker to a data symbol and is a risk control
    # with confidence tiers — this is a plain provenance fact: "we fetched uid X
    # from provider P under symbol S".
    #
    # Nothing recorded it before, and the consequence was concrete: an
    # ISIN-keyed partition could not be refetched, because an ISIN does not
    # tell you the ticker and guessing one is the mismapping `symbol_map`
    # exists to prevent. The revision canary needs exactly this, and so will
    # any later "re-read this instrument" path.
    """
    CREATE TABLE IF NOT EXISTS instrument_symbols (
        instrument_uid  TEXT    NOT NULL,
        provider        TEXT    NOT NULL,
        symbol          TEXT    NOT NULL,
        first_seen_at   TEXT    NOT NULL,
        last_seen_at    TEXT    NOT NULL,
        PRIMARY KEY (instrument_uid, provider)
    )
    """,
    # Coverage history for the revision canary. Append-only rather than an
    # upsert: "checked four times, never restated" is a different and more
    # useful fact than "last checked on Tuesday", and an upsert throws the
    # first away. This table is also what makes the selection deterministic —
    # least-recently-verified ordering needs a record of what was verified.
    """
    CREATE TABLE IF NOT EXISTS canary_checks (
        check_id           TEXT    PRIMARY KEY,
        instrument_uid     TEXT    NOT NULL,
        resolution         TEXT    NOT NULL,
        provider           TEXT    NOT NULL,
        window_start       TEXT    NOT NULL,
        window_end         TEXT    NOT NULL,
        checked_at         TEXT    NOT NULL,
        outcome            TEXT    NOT NULL,
        n_bars_compared    INTEGER NOT NULL DEFAULT 0,
        n_revisions_found  INTEGER NOT NULL DEFAULT 0,
        run_id             TEXT
    )
    """,
    # ---- strategy specs and backtests (M3) -------------------------------
    # The spec is stored as submitted, with a content hash over its canonical
    # form. `UNIQUE(strategy_id, version)` is load-bearing beyond tidiness:
    # M5's sealed holdout may be evaluated exactly once per version, and a
    # uniqueness constraint is what makes a second evaluation impossible
    # rather than merely discouraged.
    """
    CREATE TABLE IF NOT EXISTS strategy_specs (
        strategy_id        TEXT    NOT NULL,
        version            INTEGER NOT NULL,
        lineage_id         TEXT    NOT NULL,
        parent_strategy_id TEXT,
        spec_json          TEXT    NOT NULL,
        spec_hash          TEXT    NOT NULL,
        author_kind        TEXT    NOT NULL,
        expected_edge_bps  REAL,
        registered_at      TEXT    NOT NULL,
        registering_event_seq INTEGER NOT NULL,
        PRIMARY KEY (strategy_id, version)
    )
    """,
    # Every backtest, including the ones that failed their gates. M5 computes
    # deflated Sharpe from the trial count in a lineage, and a table holding
    # only the successes would understate that count — which inflates every
    # deflated metric computed from it, in the permissive direction.
    """
    CREATE TABLE IF NOT EXISTS backtests (
        backtest_id        TEXT    PRIMARY KEY,
        strategy_id        TEXT    NOT NULL,
        strategy_version   INTEGER NOT NULL,
        spec_hash          TEXT    NOT NULL,
        vintage_id         TEXT    NOT NULL,
        resolution         TEXT    NOT NULL,
        window_start       TEXT,
        window_end         TEXT,
        rng_seed           INTEGER NOT NULL,
        code_git_sha       TEXT,
        config_hash        TEXT,
        n_decisions        INTEGER NOT NULL,
        n_trades           INTEGER NOT NULL,
        n_rejected_by_cost_gate INTEGER NOT NULL DEFAULT 0,
        gross_return_pct   REAL,
        net_return_pct     REAL,
        gross_sharpe       REAL,
        net_sharpe         REAL,
        max_drawdown_pct   REAL,
        cost_drag_bps      REAL,
        turnover           REAL,
        admissible         INTEGER NOT NULL DEFAULT 0,
        caveats_json       TEXT,
        ran_at             TEXT    NOT NULL,
        completing_event_seq INTEGER NOT NULL
    )
    """,
    # Per-trade detail, so a backtest result can be interrogated rather than
    # only believed. Costs are broken out by component because "it lost money
    # after costs" and "it lost money to stamp duty specifically" lead to
    # different decisions.
    """
    CREATE TABLE IF NOT EXISTS backtest_trades (
        backtest_id       TEXT    NOT NULL,
        trade_seq         INTEGER NOT NULL,
        instrument_uid    TEXT    NOT NULL,
        entry_at          TEXT    NOT NULL,
        exit_at           TEXT,
        entry_price       TEXT    NOT NULL,
        exit_price        TEXT,
        quantity          TEXT    NOT NULL,
        gross_pnl_ccy     TEXT,
        net_pnl_ccy       TEXT,
        cost_total_ccy    TEXT,
        cost_fx_ccy       TEXT,
        cost_stamp_ccy    TEXT,
        cost_spread_ccy   TEXT,
        cost_slippage_ccy TEXT,
        expected_edge_bps REAL,
        expected_cost_bps REAL,
        holding_minutes   INTEGER,
        exit_reason       TEXT,
        PRIMARY KEY (backtest_id, trade_seq)
    )
    """,
    # Calibration runs: whether the engine itself can be trusted. Kept as its
    # own table rather than a flag on `backtests` because the subject is the
    # engine, and M5 needs to find the most recent run for this code version.
    """
    CREATE TABLE IF NOT EXISTS backtest_calibrations (
        calibration_id    TEXT    PRIMARY KEY,
        vintage_id        TEXT    NOT NULL,
        resolution        TEXT    NOT NULL,
        code_git_sha      TEXT,
        n_strategies      INTEGER NOT NULL,
        n_runs            INTEGER NOT NULL,
        rng_seed          INTEGER NOT NULL,
        worst_net_sharpe  REAL,
        best_net_sharpe   REAL,
        mean_net_sharpe   REAL,
        mean_cost_drag_bps REAL,
        tolerance         REAL    NOT NULL,
        passed            INTEGER NOT NULL,
        failures_json     TEXT,
        ran_at            TEXT    NOT NULL,
        completing_event_seq INTEGER NOT NULL
    )
    """,
    # ----------------------------------------------------------------------
    # v6 (M4) — the decision lineage. decisions -> risk_verdicts ->
    # order_intents -> fills, each row pointing back at the one before it, so
    # a fill months later traces to the feature vector and the spec version
    # that produced it. This chain is the answer to "why did it buy that".
    # ----------------------------------------------------------------------
    #
    # What the strategy saw and what it concluded, before any risk opinion.
    # `feature_snapshot_hash` and `feature_vector_json` are both stored: the
    # hash makes "the strategy saw these features" checkable, the vector makes
    # it readable without re-running the pipeline against a store that may
    # have been revised since.
    """
    CREATE TABLE IF NOT EXISTS decisions (
        decision_id        TEXT    PRIMARY KEY,
        run_id             TEXT    NOT NULL,
        strategy_id        TEXT    NOT NULL,
        strategy_version   INTEGER NOT NULL,
        spec_hash          TEXT,
        model_version      TEXT,
        instrument_uid     TEXT    NOT NULL,
        t212_ticker        TEXT,
        as_of_utc          TEXT    NOT NULL,
        bar_open_utc       TEXT,
        resolution         TEXT    NOT NULL,
        data_snapshot_id   TEXT,
        feature_snapshot_hash TEXT NOT NULL,
        feature_vector_json   TEXT NOT NULL,
        action             TEXT    NOT NULL,
        expected_edge_bps  REAL,
        expected_cost_bps  REAL,
        rationale          TEXT,
        rng_seed           INTEGER,
        regime_state       TEXT,
        regime_exposure_factor TEXT,
        decided_at         TEXT    NOT NULL,
        deciding_event_seq INTEGER NOT NULL
    )
    """,
    # One row per rule per decision, not one row per decision. A refused order
    # has to record what every rule said, including the ones that passed:
    # "blocked by the daily loss breaker" is a different investigation from
    # "blocked by the daily loss breaker and three other things", and a table
    # holding only the first failure cannot tell them apart. `limit_value` and
    # `observed_value` are stored side by side so the margin is readable —
    # a rule that passed at 99% of its limit is a different fact from one that
    # passed at 10%, and only the pair shows it.
    """
    CREATE TABLE IF NOT EXISTS risk_verdicts (
        decision_id       TEXT    NOT NULL,
        rule_name         TEXT    NOT NULL,
        verdict           TEXT    NOT NULL,
        limit_value       TEXT,
        observed_value    TEXT,
        detail            TEXT,
        is_blocking       INTEGER NOT NULL DEFAULT 1,
        evaluated_at      TEXT    NOT NULL,
        PRIMARY KEY (decision_id, rule_name)
    )
    """,
    # The write-ahead log that synthesises exactly-once submission on a venue
    # with no idempotency tokens.
    #
    # `wal_committed_at` is set BEFORE the socket write and is the whole point
    # of the table: an intent present here with no `broker_order_id` after a
    # crash is `UNKNOWN`, never `FAILED`. Treating unknown as failed and
    # retrying is precisely how a double fill happens, so `state` has a
    # distinct member for it and recovery must resolve it by *looking*, never
    # by assuming.
    #
    # `intent_id` is deterministic over (run, decision, purpose, side,
    # instrument, quantity) so a retry of the same logical order computes the
    # same id and collides with the existing row instead of creating a second
    # one.
    """
    CREATE TABLE IF NOT EXISTS order_intents (
        intent_id         TEXT    PRIMARY KEY,
        decision_id       TEXT,
        run_id            TEXT    NOT NULL,
        parent_intent_id  TEXT,
        instrument_uid    TEXT,
        t212_ticker       TEXT    NOT NULL,
        side              TEXT    NOT NULL,
        order_type        TEXT    NOT NULL,
        purpose           TEXT    NOT NULL,
        priority_class    TEXT    NOT NULL,
        quantity          TEXT    NOT NULL,
        limit_price       TEXT,
        stop_price        TEXT,
        time_validity     TEXT,
        expected_cost_bps REAL,
        risk_token_id     TEXT    NOT NULL,
        state             TEXT    NOT NULL,
        wal_committed_at  TEXT    NOT NULL,
        submitted_at      TEXT,
        broker_order_id   TEXT,
        resolved_at       TEXT,
        resolution_note   TEXT,
        n_submit_attempts INTEGER NOT NULL DEFAULT 0,
        committing_event_seq INTEGER NOT NULL
    )
    """,
    # Fills, with an explicit honesty flag on the price.
    #
    # Trading 212's history endpoints are rate-limited well below the write
    # path, so history *will* fall behind, and a fill price cannot be
    # reconstructed from the data feed (different venue). `source`
    # distinguishes `api_history` from `inferred_from_position_delta`, and an
    # inferred price must never enter the realised-PnL series the allocator
    # learns from — a made-up entry price would teach it a made-up edge.
    """
    CREATE TABLE IF NOT EXISTS fills (
        fill_id           TEXT    PRIMARY KEY,
        intent_id         TEXT,
        broker_order_id   TEXT,
        t212_ticker       TEXT    NOT NULL,
        instrument_uid    TEXT,
        side              TEXT    NOT NULL,
        quantity          TEXT    NOT NULL,
        price             TEXT,
        filled_at         TEXT,
        fees_json         TEXT,
        fx_rate           TEXT,
        source            TEXT    NOT NULL,
        confidence        TEXT    NOT NULL,
        admissible_for_pnl INTEGER NOT NULL DEFAULT 0,
        recorded_at       TEXT    NOT NULL,
        recording_event_seq INTEGER NOT NULL
    )
    """,
    # Single-instance enforcement. Two trading loops against one account
    # double every position and reconcile to nonsense, and the failure is
    # silent: both processes look healthy. A row here is a lease — holder plus
    # expiry — rather than a boolean, so a crashed instance's lock expires
    # instead of locking the account out permanently.
    """
    CREATE TABLE IF NOT EXISTS instance_locks (
        lock_name         TEXT    PRIMARY KEY,
        run_id            TEXT    NOT NULL,
        host              TEXT    NOT NULL,
        pid               INTEGER NOT NULL,
        acquired_at       TEXT    NOT NULL,
        renewed_at        TEXT    NOT NULL,
        expires_at        TEXT    NOT NULL,
        released_at       TEXT
    )
    """,
    # ----------------------------------------------------------------------
    # v7 (M4a) — the equity curve the loss breakers divide by.
    # ----------------------------------------------------------------------
    #
    # Deliberately not events. One mark per cycle for a year is half a million
    # rows, and the hash chain is for facts that must be tamper-evident; an
    # equity mark is a measurement that can be re-taken from the broker.
    # Inflating the chain with them would make the events that do matter
    # harder to read.
    #
    # `session_date` is stored rather than derived at read time, because the
    # day's opening equity is looked up by it on every decision and the
    # mapping from instant to session is a calendar question, not a substring
    # of the timestamp.
    """
    CREATE TABLE IF NOT EXISTS equity_marks (
        mark_id      TEXT    PRIMARY KEY,
        run_id       TEXT,
        at_utc       TEXT    NOT NULL,
        session_date TEXT    NOT NULL,
        equity       TEXT    NOT NULL,
        currency     TEXT,
        deployed     TEXT,
        free_cash    TEXT,
        source       TEXT    NOT NULL DEFAULT 'broker'
    )
    """,
    # ----------------------------------------------------------------------
    # v8 (M5) — the promotion pipeline. With `paper_shadow_sessions: 0` a
    # strategy goes from this gate straight to real money at floor size, so
    # these tables carry the weight a shadow period would otherwise carry.
    # ----------------------------------------------------------------------
    #
    # Every trial, and rejections above all. Deflated Sharpe divides out the
    # multiplicity of the search that produced a candidate, and multiplicity is
    # measured by counting — so a table holding only the survivors would report
    # a search of three where a thousand happened, and every deflated number
    # computed from it would be wrong in the permissive direction.
    #
    # `search_id` is stored beside `lineage_id` deliberately. Counting only
    # within a lineage leaves an evasion open that a searcher would find by
    # accident: give every candidate its own lineage and each one is a search
    # of size one, with no haircut at all. The selection universe is the
    # search, so the gate deflates on the larger of the two counts.
    """
    CREATE TABLE IF NOT EXISTS trials (
        trial_id          TEXT    PRIMARY KEY,
        search_id         TEXT    NOT NULL,
        lineage_id        TEXT    NOT NULL,
        strategy_id       TEXT,
        strategy_version  INTEGER,
        spec_hash         TEXT    NOT NULL,
        author_kind       TEXT    NOT NULL,
        parent_strategy_id TEXT,
        generation        INTEGER NOT NULL DEFAULT 0,
        outcome           TEXT    NOT NULL,
        rejection_reason  TEXT,
        backtest_id       TEXT,
        vintage_id        TEXT,
        net_sharpe        REAL,
        net_return_pct    REAL,
        max_drawdown_pct  REAL,
        n_trades          INTEGER,
        cost_drag_bps     REAL,
        returns_json      TEXT,
        trials_in_lineage_at_time INTEGER NOT NULL DEFAULT 0,
        trials_in_search_at_time  INTEGER NOT NULL DEFAULT 0,
        recorded_at       TEXT    NOT NULL,
        recording_event_seq INTEGER
    )
    """,
    # The sealed holdout, evaluated once and only once.
    #
    # `UNIQUE (strategy_id, version)` is the whole mechanism. A holdout a
    # strategy may be re-evaluated against is not a holdout — it is a slower
    # training set, because "failed, tweak, try again" is fitting to it one bit
    # at a time. The constraint makes the second evaluation raise at the
    # database rather than be caught by a reviewer noticing.
    """
    CREATE TABLE IF NOT EXISTS holdout_evaluations (
        evaluation_id     TEXT    PRIMARY KEY,
        strategy_id       TEXT    NOT NULL,
        version           INTEGER NOT NULL,
        lineage_id        TEXT    NOT NULL,
        spec_hash         TEXT    NOT NULL,
        vintage_id        TEXT    NOT NULL,
        sealed_from       TEXT    NOT NULL,
        window_start      TEXT,
        window_end        TEXT,
        backtest_id       TEXT,
        n_trades          INTEGER,
        net_sharpe        REAL,
        net_return_pct    REAL,
        max_drawdown_pct  REAL,
        cost_drag_bps     REAL,
        returns_json      TEXT,
        passed            INTEGER NOT NULL,
        detail            TEXT,
        evaluated_at      TEXT    NOT NULL,
        evaluating_event_seq INTEGER NOT NULL,
        UNIQUE (strategy_id, version)
    )
    """,
    # One row per promotion decision, carrying every gate's verdict rather than
    # the first failure. "Refused" and "refused by six independent checks" are
    # different facts about a candidate, and only the full set distinguishes a
    # near miss from noise — which is exactly the judgement the search loop
    # needs when deciding whether a lineage is worth continuing.
    """
    CREATE TABLE IF NOT EXISTS promotions (
        promotion_id      TEXT    PRIMARY KEY,
        strategy_id       TEXT    NOT NULL,
        version           INTEGER NOT NULL,
        lineage_id        TEXT    NOT NULL,
        spec_hash         TEXT    NOT NULL,
        decision          TEXT    NOT NULL,
        n_gates           INTEGER NOT NULL,
        n_failed          INTEGER NOT NULL,
        gate_results_json TEXT    NOT NULL,
        deflated_sharpe   REAL,
        deflated_sharpe_probability REAL,
        pbo               REAL,
        n_trials_deflated_by INTEGER,
        vintage_id        TEXT,
        holdout_evaluation_id TEXT,
        effective_at      TEXT,
        decided_at        TEXT    NOT NULL,
        deciding_event_seq INTEGER NOT NULL
    )
    """,
    # Where a strategy stands right now. A projection of the events above, kept
    # as its own row because the live loop reads it on every cycle and walking
    # the event log per cycle to answer "may this strategy trade" would put the
    # chain on the hot path.
    """
    CREATE TABLE IF NOT EXISTS strategy_status (
        strategy_id       TEXT    NOT NULL,
        version           INTEGER NOT NULL,
        lineage_id        TEXT    NOT NULL,
        status            TEXT    NOT NULL,
        rung              INTEGER NOT NULL DEFAULT 0,
        rung_changed_at   TEXT,
        promoted_at       TEXT,
        retired_at        TEXT,
        retire_reason     TEXT,
        realised_pnl_ccy  TEXT    NOT NULL DEFAULT '0',
        n_realised_trades INTEGER NOT NULL DEFAULT 0,
        updated_at        TEXT    NOT NULL,
        PRIMARY KEY (strategy_id, version)
    )
    """,
    # Lifetime loss budgets, per lineage rather than per strategy.
    #
    # Per strategy alone is defeated by renaming: a lineage that has lost its
    # budget produces a child with a fresh one and keeps going, which is the
    # failure mode an autonomous searcher arrives at without anybody intending
    # it. The budget is charged against the lineage, so the child inherits the
    # exhaustion.
    """
    CREATE TABLE IF NOT EXISTS lineage_budgets (
        lineage_id        TEXT    PRIMARY KEY,
        budget_ccy        TEXT    NOT NULL,
        consumed_ccy      TEXT    NOT NULL DEFAULT '0',
        n_strategies      INTEGER NOT NULL DEFAULT 0,
        exhausted_at      TEXT,
        opened_at         TEXT    NOT NULL,
        updated_at        TEXT    NOT NULL
    )
    """,
    # Every rung change, up and down, with the evidence that justified it. The
    # ratchet is asymmetric by design, and an append-only history is what makes
    # "it went up three times in a week" a query rather than an impression.
    """
    CREATE TABLE IF NOT EXISTS ladder_moves (
        move_id           TEXT    PRIMARY KEY,
        strategy_id       TEXT    NOT NULL,
        version           INTEGER NOT NULL,
        from_rung         INTEGER NOT NULL,
        to_rung           INTEGER NOT NULL,
        direction         TEXT    NOT NULL,
        reason            TEXT    NOT NULL,
        n_trades_at_move  INTEGER,
        days_at_rung      INTEGER,
        notional_ccy      TEXT,
        moved_at          TEXT    NOT NULL,
        moving_event_seq  INTEGER NOT NULL
    )
    """,
    # What the allocator decided and, more usefully, why. `prior_weight`,
    # `realised_weight` and `shrinkage` are stored separately rather than as a
    # single blended number, because the blend is the whole judgement: at ten
    # trades a strategy's realised edge is noise, and a row that recorded only
    # the result could not show whether the allocator knew that.
    """
    CREATE TABLE IF NOT EXISTS allocations (
        allocation_id     TEXT    PRIMARY KEY,
        run_id            TEXT,
        as_of_utc         TEXT    NOT NULL,
        strategy_id       TEXT    NOT NULL,
        version           INTEGER NOT NULL,
        lineage_id        TEXT    NOT NULL,
        family            TEXT,
        prior_edge_bps    TEXT,
        realised_edge_bps TEXT,
        shrinkage         TEXT    NOT NULL,
        blended_edge_bps  TEXT,
        raw_weight        TEXT,
        weight            TEXT    NOT NULL,
        rung              INTEGER NOT NULL DEFAULT 0,
        notional_ccy      TEXT    NOT NULL,
        n_realised_trades INTEGER NOT NULL DEFAULT 0,
        correlation_cap_applied INTEGER NOT NULL DEFAULT 0,
        detail            TEXT,
        allocating_event_seq INTEGER
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
    "CREATE INDEX IF NOT EXISTS ix_canary_checks_window "
    "ON canary_checks (resolution, instrument_uid, window_start)",
    "CREATE INDEX IF NOT EXISTS ix_canary_checks_when ON canary_checks (checked_at)",
    # v4 (M3)
    "CREATE INDEX IF NOT EXISTS ix_strategy_specs_lineage ON strategy_specs (lineage_id)",
    "CREATE INDEX IF NOT EXISTS ix_strategy_specs_hash ON strategy_specs (spec_hash)",
    # The lineage trial count M5's deflated Sharpe divides by. Indexed because
    # it is read on every promotion evaluation, over every trial ever run.
    "CREATE INDEX IF NOT EXISTS ix_backtests_strategy ON backtests (strategy_id, strategy_version)",
    "CREATE INDEX IF NOT EXISTS ix_backtests_vintage ON backtests (vintage_id)",
    "CREATE INDEX IF NOT EXISTS ix_backtest_trades_instrument "
    "ON backtest_trades (backtest_id, instrument_uid)",
    "CREATE INDEX IF NOT EXISTS ix_calibrations_ran ON backtest_calibrations (ran_at)",
    # v6 (M4)
    "CREATE INDEX IF NOT EXISTS ix_decisions_run ON decisions (run_id, as_of_utc)",
    "CREATE INDEX IF NOT EXISTS ix_decisions_instrument ON decisions (instrument_uid, as_of_utc)",
    "CREATE INDEX IF NOT EXISTS ix_decisions_strategy ON decisions (strategy_id, strategy_version)",
    # The recovery query, and the one that must be fast: every startup and
    # every reconcile asks "which intents are unresolved" before anything
    # else is allowed to happen.
    "CREATE INDEX IF NOT EXISTS ix_intents_unresolved ON order_intents (state, wal_committed_at)",
    "CREATE INDEX IF NOT EXISTS ix_intents_run ON order_intents (run_id, wal_committed_at)",
    "CREATE INDEX IF NOT EXISTS ix_intents_broker_order ON order_intents (broker_order_id)",
    "CREATE INDEX IF NOT EXISTS ix_intents_ticker ON order_intents (t212_ticker, purpose)",
    "CREATE INDEX IF NOT EXISTS ix_fills_intent ON fills (intent_id)",
    "CREATE INDEX IF NOT EXISTS ix_fills_ticker ON fills (t212_ticker, filled_at)",
    # The allocator reads only admissible fills. Indexed so "the realised
    # series" never accidentally becomes "the realised series including the
    # inferred prices" for performance reasons.
    "CREATE INDEX IF NOT EXISTS ix_fills_admissible ON fills (admissible_for_pnl, filled_at)",
    "CREATE INDEX IF NOT EXISTS ix_locks_live ON instance_locks (lock_name, released_at)",
    # v7 (M4a). Both indexes are on the hot path: the day's opening equity
    # and the rolling window are looked up on every decision.
    "CREATE INDEX IF NOT EXISTS ix_equity_marks_at ON equity_marks (at_utc)",
    "CREATE INDEX IF NOT EXISTS ix_equity_marks_session ON equity_marks (session_date, at_utc)",
    # v8 (M5). The first two are the multiplicity counts, read once per
    # promotion evaluation over every trial ever run — the one query whose
    # cost grows with the search rather than with the portfolio.
    "CREATE INDEX IF NOT EXISTS ix_trials_lineage ON trials (lineage_id, recorded_at)",
    "CREATE INDEX IF NOT EXISTS ix_trials_search ON trials (search_id, recorded_at)",
    "CREATE INDEX IF NOT EXISTS ix_trials_spec ON trials (spec_hash)",
    "CREATE INDEX IF NOT EXISTS ix_trials_outcome ON trials (outcome)",
    "CREATE INDEX IF NOT EXISTS ix_holdout_lineage ON holdout_evaluations (lineage_id)",
    "CREATE INDEX IF NOT EXISTS ix_promotions_strategy ON promotions (strategy_id, version)",
    "CREATE INDEX IF NOT EXISTS ix_promotions_lineage ON promotions (lineage_id, decided_at)",
    "CREATE INDEX IF NOT EXISTS ix_strategy_status_live ON strategy_status (status)",
    "CREATE INDEX IF NOT EXISTS ix_strategy_status_lineage ON strategy_status (lineage_id)",
    "CREATE INDEX IF NOT EXISTS ix_ladder_moves_strategy "
    "ON ladder_moves (strategy_id, version, moved_at)",
    "CREATE INDEX IF NOT EXISTS ix_allocations_as_of ON allocations (as_of_utc)",
    "CREATE INDEX IF NOT EXISTS ix_allocations_strategy ON allocations (strategy_id, version)",
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
