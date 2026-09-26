"""The event vocabulary.

Every row in `event_log` has a registered `EventType` and a Pydantic payload
model. There is no path for appending an ad-hoc dictionary: an event type with
no registered model is rejected at the append call.

That strictness is the whole point of the ledger. "Keeps track of every move in
a tidy manner" fails the moment two code paths record the same fact under
different key names, because then no query can find both. Adding an event type
is a deliberate edit to this file.
"""

from __future__ import annotations

from decimal import Decimal
from enum import StrEnum
from typing import Any

from pydantic import BaseModel, ConfigDict, Field


class Actor(StrEnum):
    """Who caused an event.

    Worth recording separately from the event type: "position flattened" means
    something very different when the actor is HUMAN than when it is WATCHDOG.
    """

    SYSTEM = "system"
    HUMAN = "human"
    SEARCH = "search"
    LLM = "llm"
    WATCHDOG = "watchdog"
    RECONCILER = "reconciler"
    RISK = "risk"
    BROKER = "broker"


class AggregateType(StrEnum):
    """What kind of thing an event is about."""

    LEDGER = "ledger"
    RUN = "run"
    CONFIG = "config"
    SAFETY = "safety"
    BROKER = "broker"
    STRATEGY = "strategy"
    DECISION = "decision"
    ORDER = "order"
    POSITION = "position"
    DATA = "data"
    MODEL = "model"


class EventType(StrEnum):
    """Every fact the system can record.

    Dotted names so the log can be filtered by prefix.
    """

    # --- ledger itself (M0) ---
    LEDGER_GENESIS = "ledger.genesis"
    CHAIN_ANCHORED = "chain.anchored"

    # --- process lifecycle (M0) ---
    RUN_STARTED = "run.started"
    RUN_ENDED = "run.ended"

    # --- control layer (M0) ---
    CONFIG_PINNED = "config.pinned"
    CONFIG_DRIFT_DETECTED = "config.drift_detected"

    # --- safety (M0) ---
    STATE_TRANSITIONED = "state.transitioned"
    HALT_RAISED = "halt.raised"
    HALT_CLEARED = "halt.cleared"
    KILLSWITCH_ENGAGED = "killswitch.engaged"
    KILLSWITCH_RELEASED = "killswitch.released"
    HEARTBEAT_STALE = "heartbeat.stale"

    # --- broker (M1) ---
    BROKER_PROBED = "broker.probed"
    BROKER_SCHEMA_DRIFT = "broker.schema_drift"
    BROKER_RATE_LIMITED = "broker.rate_limited"
    BROKER_SNAPSHOT_TAKEN = "broker.snapshot_taken"
    RECONCILE_COMPLETED = "reconcile.completed"

    # --- symbol mapping / cross-venue (M1) ---
    SYMBOLS_AUDITED = "symbols.audited"
    SYMBOL_BLOCKED = "symbol.blocked"
    SYMBOL_UNBLOCKED = "symbol.unblocked"

    # --- data layer (M2) ---
    DATA_PARTITION_SEALED = "data.partition_sealed"
    DATA_BAR_REVISION_DETECTED = "data.bar_revision_detected"
    DATA_SNAPSHOT_SEALED = "data.snapshot_sealed"
    DATA_AUDIT_COMPLETED = "data.audit_completed"
    DATA_PROVIDER_DEGRADED = "data.provider_degraded"
    DATA_STALENESS_BREACH = "data.staleness_breach"
    DATA_ACTION_RECORDED = "data.action_recorded"
    DATA_ACTION_RECONCILED = "data.action_reconciled"
    DATA_UNIVERSE_SNAPSHOT_TAKEN = "data.universe_snapshot_taken"
    DATA_BAKEOFF_COMPLETED = "data.bakeoff_completed"
    # Distinct from `data.bar_revision_detected`, which fires per restated
    # bar: this is the *coverage* statement — how much of stored history was
    # deliberately re-read, which is what says whether "we found nothing"
    # means anything at all.
    DATA_REVISION_CANARY_COMPLETED = "data.revision_canary_completed"
    DATA_REGIME_READ = "data.regime_read"

    # --- strategy specs and backtests (M3) ---
    STRATEGY_SPEC_REGISTERED = "strategy.spec_registered"
    BACKTEST_COMPLETED = "backtest.completed"
    # The null-strategy calibration. Its own event type rather than a flag on
    # a backtest, because it is evidence about the *engine* rather than about a
    # strategy, and M5's promotion gate needs to find the most recent one.
    BACKTEST_CALIBRATED = "backtest.calibrated"

    # --- the live path (M4) ---
    #
    # The decision lineage, in the order it happens. Every member exists
    # because something reads it back: the reconciler, the recovery pass, or
    # M8's replay.
    DECISION_MADE = "decision.made"
    # Emitted for a refusal too, with every rule's verdict attached. A ledger
    # that recorded only the orders it placed could not answer "why did it
    # stop trading", which is the more common question.
    RISK_EVALUATED = "risk.evaluated"
    # S105 reads `..._TOKEN_... = "str"` as a hardcoded credential. A
    # `RiskToken` is an in-process authorisation object that never leaves this
    # machine and is never a secret. Suppressed on the line rather than
    # disabled in config: S105 catching a real key in a literal is worth far
    # more than the noise it makes here.
    RISK_TOKEN_ISSUED = "risk.token_issued"  # noqa: S105
    # The write-ahead commit, appended *before* the socket write. This event
    # existing with no `order.submitted` after it is the UNKNOWN state the
    # recovery pass exists to resolve.
    INTENT_COMMITTED = "intent.committed"
    ORDER_SUBMITTED = "order.submitted"
    ORDER_ACKNOWLEDGED = "order.acknowledged"
    ORDER_REJECTED = "order.rejected"
    ORDER_CANCELLED = "order.cancelled"
    # Distinct from `order.acknowledged`: an intent can be resolved by
    # *discovery* during recovery rather than by a response we received, and
    # conflating the two would lose the fact that we never saw the ack.
    INTENT_RESOLVED = "intent.resolved"
    FILL_RECORDED = "fill.recorded"
    # A sell fill closed some or all of a position: its realised result, the
    # strategy it belongs to, and whether it was charged to that strategy's
    # record. The realised series the review, the allocator and the lineage
    # budgets read is rebuildable from these alone.
    TRADE_CLOSED = "trade.closed"
    # The protective stop landing behind an entry closes the unprotected
    # window. Its own event because the window's duration is a number worth
    # being able to query, not just a state worth checking.
    POSITION_PROTECTED = "position.protected"
    POSITION_UNPROTECTED = "position.unprotected"
    # The bidirectional dead-man switch, both directions. Separate types
    # because "the watchdog halted the trader" and "the trader could not reach
    # the watchdog" are different faults with different fixes.
    WATCHDOG_TRIPPED = "watchdog.tripped"
    WATCHDOG_UNREACHABLE = "watchdog.unreachable"
    INSTANCE_LOCK_ACQUIRED = "instance.lock_acquired"
    INSTANCE_LOCK_REFUSED = "instance.lock_refused"
    INSTANCE_LOCK_RELEASED = "instance.lock_released"
    LOOP_CYCLE_COMPLETED = "loop.cycle_completed"

    # --- the promotion pipeline (M5) ---
    #
    # A trial is recorded whether it was evaluated, rejected before evaluation
    # or errored. The rejections are the load-bearing ones: deflated Sharpe
    # divides out the multiplicity of the search, and multiplicity is a count
    # of everything tried, not of what survived.
    TRIAL_RECORDED = "trial.recorded"
    # One per search session, carrying the totals. Separate from the per-trial
    # events so "how big was the search that produced this" is answerable
    # without scanning every trial in it.
    SEARCH_COMPLETED = "search.completed"
    # What a language model was asked and what it returned (M6). The one
    # proposer that cannot be replayed from the search's seed, so the exchange
    # itself is the record.
    SPECS_PROPOSED = "search.specs_proposed"
    # The sealed holdout, evaluated once. A second evaluation of the same
    # version is refused by a uniqueness constraint, so this event appearing
    # twice for one version is impossible rather than merely unexpected.
    HOLDOUT_EVALUATED = "holdout.evaluated"
    HOLDOUT_VIOLATION_ATTEMPTED = "holdout.violation_attempted"
    # Every promotion decision, refusals included and with every gate's
    # verdict attached. With no paper-shadow period this event is the moment a
    # generated strategy becomes eligible for real money.
    PROMOTION_EVALUATED = "promotion.evaluated"
    STRATEGY_RETIRED = "strategy.retired"
    LINEAGE_BUDGET_EXHAUSTED = "lineage.budget_exhausted"
    # Rung changes, up and down. Up is slow and needs evidence; down is fast
    # and needs almost none, which is the asymmetry the ladder exists for.
    LADDER_MOVED = "ladder.moved"
    ALLOCATION_DECIDED = "allocation.decided"
    STRATEGY_REVIEWED = "strategy.reviewed"
    # The once-per-session portfolio pass: the review's verdicts, the ladder's
    # moves and the allocation round, for one session. Also the marker that
    # keeps the pass to once a session however often `tb run` is started.
    SESSION_REVIEWED = "session.reviewed"

    # --- funding the live loop (M5b) ---
    #
    # What a run was actually trading. Not answerable from the decisions alone:
    # a funded strategy that signalled nothing leaves no rows for the
    # instruments it declined, and an excluded one leaves none at all — so
    # "four were promoted and all four were out of budget" would otherwise look
    # identical to "nothing was ever promoted".
    BOOK_FUNDED = "book.funded"
    # A held position no funded strategy will ever close, and what was done
    # about it. Its own type rather than a note on the flattening order,
    # because the *detection* is the fact worth querying: a strategy retired
    # while holding is how a position ends up unmanaged, and the count of
    # these is how you notice it happening regularly.
    POSITION_ORPHANED = "position.orphaned"

    # --- the ML signal layer (M7) ---
    #
    # A model artifact admitted to the store. The event, not the file, is what
    # makes a model exist: a file on disk with no event is never loaded, and an
    # event whose file has changed or vanished is refused at load.
    MODEL_RECORDED = "model.recorded"

    # --- operating it (M8) ---
    #
    # A backup made, a backup restored (written into the restored ledger, on
    # the machine that restored it), and that restore reported back to the
    # ledger the backup came from — the evidence the live gate reads.
    BACKUP_CREATED = "backup.created"
    BACKUP_RESTORED = "backup.restored"
    BACKUP_RESTORE_VERIFIED = "backup.restore_verified"


class EventPayload(BaseModel):
    """Base for every payload.

    `extra="forbid"` for the same reason as the hard limits: this is our own
    data model, and a typo'd key that silently lands in the JSON blob is a fact
    nobody will ever be able to query for.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)


# --------------------------------------------------------------------------
# Ledger
# --------------------------------------------------------------------------


class GenesisPayload(EventPayload):
    """First event in a chain. Records what created it, so an empty ledger is
    still attributable."""

    ledger_schema_version: int
    created_by: str
    code_version: str
    note: str = "genesis"


class ChainAnchoredPayload(EventPayload):
    """The chain head was published outside this database.

    A hash chain is only tamper-evident against an adversary who cannot rewrite
    the file, and the process writing it can. Anchoring the head across a trust
    boundary is what turns the chain from theatre into evidence.
    """

    anchor_id: str
    anchored_seq: int
    anchored_chain_hash: str
    sink: str
    external_ref: str | None = None


# --------------------------------------------------------------------------
# Process lifecycle
# --------------------------------------------------------------------------


class RunStartedPayload(EventPayload):
    run_id: str
    mode: str
    code_git_sha: str | None = None
    code_version: str
    config_hash: str | None = None
    rng_seed_root: int | None = None
    host: str
    pid: int
    python_version: str
    argv: list[str] = Field(default_factory=list)


class RunEndedPayload(EventPayload):
    run_id: str
    exit_reason: str
    error_type: str | None = None
    error_detail: str | None = None


# --------------------------------------------------------------------------
# Control layer
# --------------------------------------------------------------------------


class ConfigPinnedPayload(EventPayload):
    """The hard limits in force for this run, in full.

    The complete values are stored, not just the hash: months later, "what were
    the caps when this trade happened" must be answerable from the ledger alone,
    without needing the git history of a file that may have moved.
    """

    config_hash: str
    canonical_hash: str
    kind: str
    source_path: str
    schema_version: int
    currency: str
    immutability_enforced: bool
    immutability_detail: str
    values: dict[str, Any]


class ConfigDriftPayload(EventPayload):
    """The limits file changed underneath a running process. Always a halt."""

    source_path: str
    pinned_hash: str
    observed_hash: str | None
    detail: str


# --------------------------------------------------------------------------
# Safety
# --------------------------------------------------------------------------


class StateTransitionPayload(EventPayload):
    from_state: str
    to_state: str
    reason: str
    run_id: str | None = None


class HaltRaisedPayload(EventPayload):
    halt_id: str
    trigger: str
    detail: str
    run_id: str | None = None
    # Populated where the breaker is numeric, so the log shows not just that a
    # limit fired but by how much it was exceeded.
    observed_value: Decimal | float | None = None
    limit_value: Decimal | float | None = None


class HaltClearedPayload(EventPayload):
    halt_id: str
    cleared_by: str
    clear_reason: str


class KillswitchPayload(EventPayload):
    """Engaged or released.

    `determinable` is false when the switch state could not be read at all. The
    system treats that as engaged — undeterminable is not permission to trade —
    and recording the distinction is what lets you tell a deliberate stop from
    a broken mount.
    """

    path: str
    determinable: bool
    detail: str
    engaged_by: str | None = None


class HeartbeatStalePayload(EventPayload):
    path: str
    age_seconds: float
    threshold_seconds: int
    action: str


# --------------------------------------------------------------------------
# Broker
# --------------------------------------------------------------------------


class BrokerProbedPayload(EventPayload):
    """What the capability probe actually found.

    The rate limits and auth format in the code are conservative guesses —
    Trading 212's API is in beta and its docs are not reliably reachable. This
    event is the record of what the live endpoint really does, so the adapter
    is built against observed reality rather than a reconstruction.
    """

    environment: str
    auth_scheme: str
    endpoints_probed: int
    endpoints_ok: int
    base_currency: str | None = None
    observations: list[dict[str, Any]] = Field(default_factory=list)
    disagreements: list[str] = Field(default_factory=list)


class BrokerSchemaDriftPayload(EventPayload):
    """A field the system actually consumes changed shape.

    Unknown *extra* fields are ignored by design — taking the bot down because
    the broker added a field would be worse than not reading it. But a consumed
    field that went missing, went null, or changed type is a halt: the
    alternative is trading on a number we guessed at.
    """

    endpoint: str
    url_path: str
    msg_id: str
    model: str
    error_detail: str
    status_code: int | None = None


class BrokerRateLimitedPayload(EventPayload):
    endpoint: str
    status_code: int
    retry_after_seconds: float | None = None
    observed_limit: int | None = None
    observed_period_seconds: int | None = None
    configured_limit: int | None = None
    configured_period_seconds: int | None = None


class BrokerSnapshotPayload(EventPayload):
    """A point-in-time read of what the account actually holds."""

    snap_id: str
    environment: str
    n_positions: int
    n_open_orders: int
    currency: str | None = None
    free_cash: Decimal | None = None
    total_value: Decimal | None = None
    invested: Decimal | None = None
    max_price_disagreement_bps: float | None = None


class ReconcileCompletedPayload(EventPayload):
    recon_id: str
    verdict: str
    n_unknown_intents: int
    n_orphan_orders: int
    n_position_mismatches: int
    n_unprotected_positions: int
    n_price_disagreements: int
    findings: list[dict[str, Any]] = Field(default_factory=list)
    actions: list[str] = Field(default_factory=list)
    dry_run: bool = False


# --------------------------------------------------------------------------
# Symbol mapping
# --------------------------------------------------------------------------


class SymbolsAuditedPayload(EventPayload):
    provider: str
    n_instruments: int
    n_mapped: int
    n_unmapped: int
    n_low_confidence: int
    n_blocked: int
    unmapped_sample: list[str] = Field(default_factory=list)


class SymbolBlockedPayload(EventPayload):
    """A symbol stopped accepting new entries.

    The execution venue is not the data venue, so a signal can be computed on
    one venue's price and filled at another's. Beyond the configured band that
    divergence means something is wrong — a stale feed, a corporate action, or
    a mismapped ticker — and none of those are conditions to open a position in.
    """

    t212_ticker: str
    data_symbol: str
    reason: str
    disagreement_bps: float | None = None
    limit_bps: float | None = None
    broker_price: Decimal | None = None
    data_price: Decimal | None = None


# --------------------------------------------------------------------------
# Data layer
# --------------------------------------------------------------------------


class PartitionSealedPayload(EventPayload):
    """A Parquet file became part of the dataset.

    This event is what *makes* it part of the dataset. A file on disk with no
    sealing event is ignorable garbage; an event naming a missing file is a
    loud integrity failure. Defining membership by the ledger rather than by a
    directory listing is what keeps the two stores from silently diverging.
    """

    file_sha256: str
    relative_path: str
    instrument_uid: str
    resolution: str
    provider: str
    first_bar_open: str
    last_bar_open: str
    row_count: int
    byte_size: int
    rows_hash: str
    supersedes: list[str] = Field(default_factory=list)


class BarRevisionPayload(EventPayload):
    """A bar we had already stored came back different, or vanished.

    Yahoo silently back-adjusts history, so this is expected rather than
    exceptional — and it is the evidence when a live result diverges from the
    backtest that supposedly validated it.
    """

    revision_id: str
    instrument_uid: str
    resolution: str
    bar_open_utc: str
    provider: str
    # `changed` | `deleted` | `late_insert` — deletions and late arrivals are
    # invisible to a row-by-row diff, so the comparison runs over the set.
    kind: str
    delta_bps: float | None = None
    detected_by: str
    first_values: dict[str, Any] | None = None
    new_values: dict[str, Any] | None = None


class SnapshotSealedPayload(EventPayload):
    """A dataset vintage, frozen and hashed.

    M3's entire input. A backtest whose `vintage_id` is not in this log is not
    admissible evidence for promotion, which is what stops a strategy being
    validated against data that has since been restated underneath it.
    """

    vintage_id: str
    as_of_utc: str
    manifest_hash: str
    window_start: str | None = None
    window_end: str | None = None
    resolutions: list[str] = Field(default_factory=list)
    n_instruments: int
    n_files: int
    row_count: int
    calendar_hash: str | None = None
    action_table_hash: str | None = None
    fx_table_hash: str | None = None
    universe_snapshot_id: str | None = None
    sealed_from: str | None = None
    first_live_observation_at: str | None = None
    # `unmeasured` until dated universe snapshots span the backtest window.
    survivorship_flag: str
    # `vendor_current_view` until live observation covers the window — i.e. the
    # as-of machinery is inert over backfilled history and says so.
    pit_completeness_flag: str
    provider_delays: dict[str, float] = Field(default_factory=dict)


class DataAuditPayload(EventPayload):
    audit_id: str
    n_instruments: int
    n_bars_checked: int
    n_findings: int
    n_blocking: int
    # Only `unexplained` gaps count against a feed; a missing minute on a thin
    # name over IEX is normal, and lumping them together is uninterpretable.
    gaps_unexplained: int = 0
    gaps_explained: int = 0
    findings: list[dict[str, Any]] = Field(default_factory=list)


class ProviderDegradedPayload(EventPayload):
    """A provider stopped being usable for its declared purpose."""

    provider: str
    resolution: str
    reason: str
    observed_delay_p95_s: float | None = None
    max_allowed_delay_s: float | None = None
    live_capable: bool


class StalenessBreachPayload(EventPayload):
    """No provider could supply a fresh enough bar to decide on.

    Recorded rather than silently skipped: how often this fires is the
    empirical answer to whether the free feed can support the cadence.
    """

    instrument_uid: str
    resolution: str
    newest_available_at: str | None = None
    decision_time: str
    age_seconds: float | None = None
    limit_seconds: int
    action: str


class ActionRecordedPayload(EventPayload):
    action_id: str
    instrument_uid: str
    action_type: str
    effective_date: str
    known_at_utc: str
    ratio_num: int | None = None
    ratio_den: int | None = None
    gross_amount: Decimal | None = None
    currency: str | None = None
    source_provider: str
    # True when the residual detector inferred it from a price jump with no
    # corresponding action row — i.e. an unannounced split.
    inferred_from_price_jump: bool = False


class ActionReconciledPayload(EventPayload):
    """A provider dividend checked against cash the broker actually credited.

    The highest-value check in the data layer: a dividend with no matching
    credit means either the action data is wrong, or we are tracking a
    different company than the one we hold.
    """

    action_id: str
    instrument_uid: str
    matched: bool
    provider_amount: Decimal | None = None
    broker_amount: Decimal | None = None
    detail: str


class UniverseSnapshotPayload(EventPayload):
    snapshot_id: str
    n_members: int
    n_candidates_considered: int
    selection_rule: str
    members: list[str] = Field(default_factory=list)
    currency: str | None = None


class BakeoffPayload(EventPayload):
    """The paid-data verdict, as arithmetic.

    The headline is not freshness but representativeness: IEX is a few percent
    of consolidated volume, so its systematic disagreement sits in the same
    5-20bps band as the entire gross edge being traded.
    """

    bakeoff_id: str
    window_start: str
    window_end: str
    resolution: str
    providers: list[str] = Field(default_factory=list)
    n_symbols: int
    n_compared_bars: int
    disagreement_median_bps: float | None = None
    disagreement_p95_bps: float | None = None
    disagreement_p99_bps: float | None = None
    missing_bar_fraction: dict[str, float] = Field(default_factory=dict)
    observed_delay_p95_s: dict[str, float] = Field(default_factory=dict)
    cycles_meeting_staleness_bound_pct: float | None = None
    verdict: str
    rationale: str


class RevisionCanaryPayload(EventPayload):
    """One deliberate re-read of stored history.

    `n_windows_checked` against `n_windows_available` is the point. A run that
    checked 2 of 400 windows and found nothing has established almost nothing,
    and a payload carrying only the finding count would read as reassurance.
    """

    canary_id: str
    resolution: str
    n_windows_available: int
    n_windows_checked: int
    n_bars_compared: int
    n_revisions_found: int
    coverage_pct: float
    oldest_checked_age_days: float | None = None
    instruments_restated: list[str] = Field(default_factory=list)
    problems: list[str] = Field(default_factory=list)


class RegimeReadPayload(EventPayload):
    """One reading of the regime gate, and why it said what it said.

    `state` is carried separately from `exposure_factor` because the two
    failure states produce the *same* reduced factor as a genuine `RISK_OFF`.
    Reading the factor alone months later could not tell "the index was below
    its average" from "we could not see the index", and those call for opposite
    responses: one is the gate working, the other is the feed broken.
    """

    state: str
    exposure_factor: Decimal
    reference_symbol: str
    instrument_uid: str
    ma_days: int
    n_sessions_seen: int
    is_measured: bool
    as_of: str
    last_close: Decimal | None = None
    moving_average: Decimal | None = None
    detail: str = ""


# --------------------------------------------------------------------------
# The live path (M4)
# --------------------------------------------------------------------------


class DecisionPayload(EventPayload):
    """What the strategy saw and what it concluded.

    The feature vector is carried in full alongside its hash. The hash makes
    "the strategy saw exactly these inputs" checkable; the vector makes it
    *readable* months later without re-running the pipeline against a store
    that may have been revised since — which for a revised store would produce
    different numbers and no way to tell which set was the real one.
    """

    decision_id: str
    run_id: str
    strategy_id: str
    strategy_version: int
    instrument_uid: str
    as_of_utc: str
    resolution: str
    action: str
    feature_snapshot_hash: str
    feature_vector: dict[str, Any] = Field(default_factory=dict)
    spec_hash: str | None = None
    model_version: str | None = None
    t212_ticker: str | None = None
    bar_open_utc: str | None = None
    data_snapshot_id: str | None = None
    expected_edge_bps: float | None = None
    expected_cost_bps: float | None = None
    rationale: str = ""
    rng_seed: int | None = None
    regime_state: str | None = None
    regime_exposure_factor: Decimal | None = None


class RiskVerdictRow(BaseModel):
    """One rule's opinion. Not an `EventPayload` — it nests inside one."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    rule_name: str
    verdict: str
    limit_value: Decimal | float | str | None = None
    observed_value: Decimal | float | str | None = None
    detail: str = ""
    is_blocking: bool = True


class RiskEvaluationPayload(EventPayload):
    """Every rule's verdict for one decision, pass or fail.

    `verdicts` carries all of them rather than only the failures. A refusal
    recorded as "blocked by the daily loss breaker" cannot be told apart later
    from one blocked by that breaker *and* three other rules, and the two lead
    to different investigations. Recording the passes also makes the margins
    queryable: a rule that passed at 99% of its limit is a warning that a rule
    passing at 10% is not.
    """

    decision_id: str
    run_id: str
    approved: bool
    n_rules_evaluated: int
    n_blocking_failures: int
    verdicts: list[RiskVerdictRow] = Field(default_factory=list)
    # Present only on approval. The size the engine actually authorised, which
    # may be well below what the strategy asked for.
    approved_quantity: Decimal | None = None
    approved_notional_ccy: Decimal | None = None
    token_id: str | None = None
    refusal_summary: str = ""


class RiskTokenPayload(EventPayload):
    """A token was issued, and for exactly what.

    The bound order parameters are recorded so a token cannot later be claimed
    to have authorised something else: the ledger says the engine approved a
    BUY of this quantity of this ticker, and `place_order` refuses a token
    whose parameters do not match the order in hand.
    """

    token_id: str
    decision_id: str | None = None
    run_id: str
    t212_ticker: str
    side: str
    purpose: str
    quantity: Decimal
    max_notional_ccy: Decimal | None = None
    issued_at: str
    expires_at: str


class IntentCommittedPayload(EventPayload):
    """The write-ahead commit, appended before the socket write.

    This event with no `order.submitted` behind it *is* the UNKNOWN state. It
    is deliberately recorded before anything is sent, because the alternative
    ordering — send, then record — loses the intent entirely on a crash
    mid-flight and leaves an order at the broker that nothing in this system
    knows about.
    """

    intent_id: str
    run_id: str
    decision_id: str | None = None
    parent_intent_id: str | None = None
    t212_ticker: str
    side: str
    order_type: str
    purpose: str
    priority_class: str
    quantity: Decimal
    risk_token_id: str
    limit_price: Decimal | None = None
    stop_price: Decimal | None = None
    time_validity: str | None = None
    expected_cost_bps: float | None = None


class OrderSubmittedPayload(EventPayload):
    """A POST left the process.

    `attempt` is recorded because "exactly one POST per intent_id" is the
    property the crash drills assert, and asserting it needs the count to be
    visible rather than inferred from the absence of duplicates.
    """

    intent_id: str
    run_id: str
    t212_ticker: str
    attempt: int
    sent_at: str


class OrderOutcomePayload(EventPayload):
    """An acknowledgement, rejection or cancellation from the broker."""

    intent_id: str | None = None
    run_id: str
    t212_ticker: str
    broker_order_id: str | None = None
    status: str
    detail: str = ""
    # The broker's own words, kept because a rejection reason is the only
    # evidence of a venue rule we did not know about.
    broker_message: str | None = None


class IntentResolvedPayload(EventPayload):
    """How an unresolved intent was settled.

    `resolved_by` distinguishes a response we received from a state we
    *discovered* during recovery. Losing that distinction would make an
    unacknowledged order that turned out to have filled look like a normal
    acknowledged one, and the unprotected window it implies would go
    unmeasured.
    """

    intent_id: str
    run_id: str
    final_state: str
    resolved_by: str
    broker_order_id: str | None = None
    detail: str = ""


class FillPayload(EventPayload):
    """A fill, with the honesty of its price attached.

    `source` is `api_history` or `inferred_from_position_delta`, and
    `admissible_for_pnl` follows from it. An inferred price is a guess derived
    from a position change; letting one into the realised series would teach
    the allocator an edge that was never earned.
    """

    fill_id: str
    run_id: str
    t212_ticker: str
    side: str
    quantity: Decimal
    source: str
    confidence: str
    admissible_for_pnl: bool
    intent_id: str | None = None
    broker_order_id: str | None = None
    instrument_uid: str | None = None
    price: Decimal | None = None
    filled_at: str | None = None
    fees: dict[str, Any] = Field(default_factory=dict)
    fx_rate: Decimal | None = None


class TradeClosedPayload(EventPayload):
    """The realised result of one closing fill, and who it belongs to.

    `admissible` is false when any fill the result depends on has no reported
    price — the basis or the exit would be a guess — and then nothing is
    charged: an inferred number must not teach the allocator an edge, nor spend
    a lineage's budget on a loss nobody measured. `charged` says whether it
    reached the strategy's record, and `detail` why not when it did not.
    """

    closing_fill_id: str
    run_id: str
    t212_ticker: str
    quantity: Decimal
    admissible: bool
    charged: bool
    strategy_id: str | None = None
    strategy_version: int | None = None
    exit_price: Decimal | None = None
    cost_basis: Decimal | None = None
    pnl_ccy: Decimal | None = None
    closed_at: str | None = None
    detail: str = ""


class ProtectionPayload(EventPayload):
    """A position gained or lost its protective stop.

    `unprotected_seconds` is the number this event exists for. The window
    between an entry fill and its stop is unavoidable on a venue with no
    bracket orders, and sizing assumes a bound on it — so the actual duration
    has to be measurable rather than assumed.
    """

    t212_ticker: str
    run_id: str
    quantity: Decimal
    protected: bool
    stop_intent_id: str | None = None
    stop_price: Decimal | None = None
    entry_fill_id: str | None = None
    unprotected_seconds: float | None = None
    # Why a position lost its stop, for the one distinction that matters: a
    # stop withdrawn on purpose ahead of an exit ("withdrawn") against a failure
    # to protect — "no_price", "risk_refused", "placement_failed". Absent on
    # events written before it existed, where the detail text still says.
    cause: str | None = None
    detail: str = ""


class WatchdogPayload(EventPayload):
    """One direction of the dead-man switch firing."""

    direction: str
    run_id: str | None = None
    observed_age_seconds: float | None = None
    limit_seconds: float | None = None
    action_taken: str = ""
    detail: str = ""


class InstanceLockPayload(EventPayload):
    """An attempt to become the single trading instance.

    A refusal is recorded as loudly as an acquisition: two loops against one
    account is a silent fault — both processes look healthy while every
    position is doubled — so the second one's refusal is the only trace that
    it was ever started.
    """

    lock_name: str
    run_id: str
    host: str
    pid: int
    acquired: bool
    expires_at: str | None = None
    held_by_run_id: str | None = None
    held_by_pid: int | None = None
    detail: str = ""


class LoopCyclePayload(EventPayload):
    """One pass of the tick loop.

    Cheap to append and worth appending: a loop that is running but deciding
    nothing looks identical to a stopped loop unless each cycle says so, and
    "it was up all day" is a claim the ledger should be able to settle.
    """

    run_id: str
    cycle: int
    as_of_utc: str
    n_instruments_considered: int
    n_decisions: int
    n_orders_submitted: int
    n_risk_refusals: int
    duration_ms: float
    halted: bool = False
    detail: str = ""
    n_fills_recorded: int = 0


class StrategySpecPayload(EventPayload):
    """A strategy specification entering the registry.

    `author_kind` distinguishes a hand-written spec from a searcher's and from
    an LLM's. It matters for M5's multiplicity accounting: a deterministic
    searcher generates orders of magnitude more candidates than a person does,
    so the deflated-Sharpe trial count is dominated by whichever produced the
    lineage.
    """

    strategy_id: str
    lineage_id: str
    version: int
    spec_hash: str
    author_kind: str
    parent_strategy_id: str | None = None
    expected_edge_bps: float | None = None
    n_operators: int | None = None


class BacktestPayload(EventPayload):
    """One backtest, and the evidence needed to judge whether it means anything.

    `vintage_id` is not decoration. A backtest that does not name a sealed
    vintage ran against a store whose contents have since been free to change,
    so its numbers cannot be reproduced and cannot support a decision to put
    money behind them. `rng_seed` and `code_git_sha` complete the triple that
    makes a re-run comparable.
    """

    backtest_id: str
    strategy_id: str
    strategy_version: int
    spec_hash: str
    vintage_id: str
    resolution: str
    window_start: str
    window_end: str
    rng_seed: int

    n_decisions: int
    n_trades: int
    n_rejected_by_cost_gate: int = 0

    gross_return_pct: float | None = None
    net_return_pct: float | None = None
    # Both, always. A gross Sharpe on this venue is a number about a strategy
    # that does not exist, since the fee schedule is the binding constraint.
    gross_sharpe: float | None = None
    net_sharpe: float | None = None
    max_drawdown_pct: float | None = None
    cost_drag_bps: float | None = None
    turnover: float | None = None
    admissible: bool = False
    caveats: list[str] = Field(default_factory=list)


class CalibrationPayload(EventPayload):
    """Whether the backtester itself can be trusted.

    Evidence about the engine, not about a strategy. Null strategies — random
    entries, always-long, coin-flip — must show a post-cost Sharpe that is
    approximately the cost drag and no better. A coin-flip strategy showing
    positive net Sharpe is not a discovery, it is a lookahead bug, a fill
    price taken from the wrong bar, or a cost model charging too little.
    """

    calibration_id: str
    vintage_id: str
    resolution: str
    n_strategies: int
    n_runs: int
    rng_seed: int
    worst_net_sharpe: float | None = None
    best_net_sharpe: float | None = None
    mean_net_sharpe: float | None = None
    mean_cost_drag_bps: float | None = None
    tolerance: float
    passed: bool
    failures: list[str] = Field(default_factory=list)


# --------------------------------------------------------------------------
# The promotion pipeline (M5)
# --------------------------------------------------------------------------


class TrialPayload(EventPayload):
    """One candidate evaluated, or refused before evaluation.

    Both counts are carried because each closes a different hole. The lineage
    count is the obvious one. The *search* count is the one that matters more:
    a searcher that gives every candidate its own lineage makes each one a
    search of size one and takes no multiplicity haircut at all — not by
    cheating, just by naming things. Recording both at the moment of the trial
    means the count cannot be recomputed later against a table that has grown.
    """

    trial_id: str
    search_id: str
    lineage_id: str
    spec_hash: str
    author_kind: str
    outcome: str
    strategy_id: str | None = None
    strategy_version: int | None = None
    parent_strategy_id: str | None = None
    generation: int = 0
    rejection_reason: str | None = None
    backtest_id: str | None = None
    vintage_id: str | None = None
    net_sharpe: float | None = None
    n_trades: int | None = None
    trials_in_lineage_at_time: int = 0
    trials_in_search_at_time: int = 0


class SearchCompletedPayload(EventPayload):
    """The totals for one search session.

    `n_proposed` against `n_promoted` is the number that says whether the gate
    is doing its job. A search that promotes a tenth of what it proposes has
    either found a market inefficiency or a bug in this repo, and the second is
    overwhelmingly more likely.
    """

    search_id: str
    lineage_ids: list[str] = Field(default_factory=list)
    n_proposed: int
    n_evaluated: int
    n_rejected: int
    n_errored: int = 0
    n_passed_gate: int = 0
    duration_seconds: float | None = None
    detail: str = ""


class SpecsProposedPayload(EventPayload):
    """What a language model was asked for specs, and what came back.

    Recorded because a model is the one proposer that cannot be replayed: the
    random and mutation proposers regenerate every candidate from the search's
    seed, and a model given the same prompt twice writes two different replies.
    So the exchange is the record:

    * **the exact prompts**, which is also how anyone can check after the fact
      that the model was shown no date, price or instrument — the claim the
      price-blind prompt builder makes, made auditable;
    * **a hash of the reply** rather than the reply, since the reply is
      untrusted text and everything in it that became a spec is below;
    * **every accepted spec in full**, so a candidate that was never registered
      can still be reconstructed from the hash on its trial row.
    """

    search_id: str
    proposer: str
    requested_model: str
    served_model: str
    fell_back: bool = False
    stop_reason: str | None = None
    n_requested: int
    n_items: int
    n_accepted: int
    n_refused: int = 0
    n_duplicates: int = 0
    refused: list[str] = Field(default_factory=list)
    ignored_keys: list[str] = Field(default_factory=list)
    stopped: str = ""
    system_prompt: str
    user_prompt: str
    reply_sha256: str
    reply_chars: int
    specs: list[dict[str, Any]] = Field(default_factory=list)


class HoldoutEvaluatedPayload(EventPayload):
    """The sealed holdout, evaluated — once.

    `sealed_from` is recorded on the event rather than looked up later, because
    the seal is the claim being made: that no part of the process that produced
    this strategy could see data past this instant. A boundary read back from
    current config would be a boundary that moved.
    """

    evaluation_id: str
    strategy_id: str
    version: int
    lineage_id: str
    spec_hash: str
    vintage_id: str
    sealed_from: str
    passed: bool
    n_trades: int | None = None
    net_sharpe: float | None = None
    net_return_pct: float | None = None
    max_drawdown_pct: float | None = None
    detail: str = ""


class HoldoutViolationPayload(EventPayload):
    """Something asked for data past the seal.

    Recorded rather than only raised. A raise stops one process; the event is
    what makes a *pattern* of attempts visible, and a searcher repeatedly
    reaching past the boundary is a finding about the searcher rather than an
    accident.
    """

    strategy_id: str | None = None
    lineage_id: str | None = None
    sealed_from: str
    requested_at: str
    caller: str = ""
    detail: str = ""


class PromotionPayload(EventPayload):
    """A promotion decision, with every gate's verdict.

    `gate_results` holds one entry per gate including the ones that passed. A
    record of only the failures cannot distinguish "refused by one gate at 99%
    of its threshold" from "refused by six", and those call for opposite
    responses from the search loop.
    """

    promotion_id: str
    strategy_id: str
    version: int
    lineage_id: str
    spec_hash: str
    decision: str
    n_gates: int
    n_failed: int
    gate_results: list[dict[str, Any]] = Field(default_factory=list)
    deflated_sharpe: float | None = None
    deflated_sharpe_probability: float | None = None
    pbo: float | None = None
    n_trials_deflated_by: int | None = None
    vintage_id: str | None = None
    holdout_evaluation_id: str | None = None
    effective_at: str | None = None


class StrategyRetiredPayload(EventPayload):
    """A strategy stopped trading, and why.

    Retirement is cheap on purpose. KILL requires little evidence and SCALE
    requires a lot, because at ten trades a month the cost of retiring a good
    strategy is a missed opportunity and the cost of keeping a bad one is money.
    """

    strategy_id: str
    version: int
    lineage_id: str
    reason: str
    realised_pnl_ccy: str | None = None
    n_realised_trades: int = 0
    detail: str = ""


class LineageBudgetPayload(EventPayload):
    """A lineage's lifetime loss budget is spent.

    Blocks the lineage, not the strategy. A per-strategy budget is defeated by
    producing a child, which is what an autonomous searcher does by default
    rather than by intent.
    """

    lineage_id: str
    budget_ccy: str
    consumed_ccy: str
    n_strategies: int
    triggering_strategy_id: str | None = None
    detail: str = ""


class LadderMovePayload(EventPayload):
    """A size rung change, up or down."""

    move_id: str
    strategy_id: str
    version: int
    from_rung: int
    to_rung: int
    direction: str
    reason: str
    n_trades_at_move: int | None = None
    days_at_rung: int | None = None
    notional_ccy: str | None = None


class AllocationPayload(EventPayload):
    """What the allocator gave a strategy, and the blend behind it.

    The shrinkage weight is on the event because it is the claim being made
    about how much the realised record is worth believing. At floor size with
    multi-day holds a strategy produces 10-20 trades a month, and a number
    derived mostly from the prior should say so rather than presenting itself
    as a measurement.
    """

    allocation_id: str
    as_of_utc: str
    run_id: str | None = None
    n_strategies: int
    total_notional_ccy: str
    entries: list[dict[str, Any]] = Field(default_factory=list)
    n_correlation_capped: int = 0
    detail: str = ""


class StrategyReviewedPayload(EventPayload):
    """KEEP / KILL / ITERATE / SCALE, with the evidence behind the verdict."""

    strategy_id: str
    version: int
    lineage_id: str
    verdict: str
    n_realised_trades: int
    realised_pnl_ccy: str | None = None
    realised_edge_bps: str | None = None
    declared_edge_bps: str | None = None
    evidence_sufficient: bool = False
    reasons: list[str] = Field(default_factory=list)


class SessionReviewedPayload(EventPayload):
    """One session's portfolio pass, summarised; the detail is in its own events."""

    session_date: str
    n_strategies: int
    run_id: str | None = None
    verdicts: dict[str, str] = Field(default_factory=dict)
    rung_moves: dict[str, str] = Field(default_factory=dict)
    allocation_id: str | None = None
    detail: str = ""


# --------------------------------------------------------------------------
# Funding the live loop
# --------------------------------------------------------------------------


class BookFundedPayload(EventPayload):
    """The strategies a run will trade, with the size each may deploy.

    `excluded` is on the payload for the same reason the gate records its
    failures: a run that traded nothing because four promoted strategies were
    all out of lineage budget is a different fact from a run that traded nothing
    because nothing was ever promoted, and only one of them is a reason to look
    at the searcher.
    """

    run_id: str
    as_of_utc: str
    source: str = "registry"
    n_funded: int
    n_excluded: int = 0
    equity_ccy: str | None = None
    entries: list[dict[str, Any]] = Field(default_factory=list)
    excluded: list[dict[str, str]] = Field(default_factory=list)
    detail: str = ""


class PositionOrphanedPayload(EventPayload):
    """A held position no funded strategy will close, and what was done.

    `owner_strategy_id` is optional and its absence is meaningful: a position
    whose owner is known but retired is the bot's own, while one that cannot be
    attributed at all came from somewhere this lineage does not cover. The
    action taken is recorded either way, because flattening a position is not a
    thing to have to infer from a later order.
    """

    t212_ticker: str
    instrument_uid: str | None = None
    quantity: str
    owner_strategy_id: str | None = None
    owner_version: int | None = None
    owner_intent_id: str | None = None
    reason: str
    action_taken: str
    intent_id: str | None = None


# --------------------------------------------------------------------------
# The ML signal layer (M7)
# --------------------------------------------------------------------------


class ModelRecordedPayload(EventPayload):
    """A model artifact, the hash that is its identity, and how it was made.

    Everything needed to rebuild the model from the data is on the event
    rather than only on the projection row: the features in the order the model
    reads them, the label definition, the parameters and the vintage. The row
    is a cache and can be edited; the event is chained. `trained_through` is
    the latest instant any training label was knowable — the claim a holdout
    evaluation checks, because a model that saw prices past the seal carries
    them into every spec that reads it.
    """

    model_id: str
    artifact_sha256: str
    kind: str
    relative_path: str
    byte_size: int
    feature_names: list[str]
    features: list[dict[str, Any]]
    label: dict[str, str]
    params: dict[str, str]
    vintage_id: str
    sealed_from: str | None = None
    window_start: str
    window_end: str
    trained_through: str
    n_samples: int
    base_rate: float | None = None
    metrics: dict[str, Any] = Field(default_factory=dict)
    search_id: str | None = None


# --------------------------------------------------------------------------
# Operating it (M8)
# --------------------------------------------------------------------------


class BackupCreatedPayload(EventPayload):
    """A copy of everything a restore needs, and the manifest that names it.

    The manifest's hash is the backup's identity. A restore elsewhere reports
    it back, and a receipt naming a manifest this ledger never recorded is not
    evidence of anything.
    """

    backup_id: str
    manifest_sha256: str
    head_seq: int
    head_chain_hash: str
    n_files: int
    n_bytes: int
    host: str
    destination: str


class BackupRestoredPayload(EventPayload):
    """A backup restored and checked, written into the restored ledger itself.

    `checks` names what was checked rather than only that something passed:
    a restore whose replay step had no fills to replay proved less than one
    that replayed three, and the record should say which it was.
    """

    backup_id: str
    manifest_sha256: str
    source_host: str
    restored_host: str
    n_files: int
    checks: list[str] = Field(default_factory=list)
    n_fills_replayed: int = 0


class BackupRestoreVerifiedPayload(EventPayload):
    """A restore on another machine, reported back to the ledger it came from.

    `same_host` is on the event because a restore onto the machine that made
    the backup proves the files are intact and nothing about surviving the
    loss of that machine, which is the point of the drill.
    """

    backup_id: str
    manifest_sha256: str
    source_host: str
    restored_host: str
    restored_at: str
    same_host: bool
    restored_head_seq: int
    restored_head_chain_hash: str
    checks: list[str] = Field(default_factory=list)
    n_fills_replayed: int = 0


# --------------------------------------------------------------------------
# Registry
# --------------------------------------------------------------------------

EVENT_PAYLOADS: dict[EventType, type[EventPayload]] = {
    EventType.LEDGER_GENESIS: GenesisPayload,
    EventType.CHAIN_ANCHORED: ChainAnchoredPayload,
    EventType.RUN_STARTED: RunStartedPayload,
    EventType.RUN_ENDED: RunEndedPayload,
    EventType.CONFIG_PINNED: ConfigPinnedPayload,
    EventType.CONFIG_DRIFT_DETECTED: ConfigDriftPayload,
    EventType.STATE_TRANSITIONED: StateTransitionPayload,
    EventType.HALT_RAISED: HaltRaisedPayload,
    EventType.HALT_CLEARED: HaltClearedPayload,
    EventType.KILLSWITCH_ENGAGED: KillswitchPayload,
    EventType.KILLSWITCH_RELEASED: KillswitchPayload,
    EventType.HEARTBEAT_STALE: HeartbeatStalePayload,
    EventType.BROKER_PROBED: BrokerProbedPayload,
    EventType.BROKER_SCHEMA_DRIFT: BrokerSchemaDriftPayload,
    EventType.BROKER_RATE_LIMITED: BrokerRateLimitedPayload,
    EventType.BROKER_SNAPSHOT_TAKEN: BrokerSnapshotPayload,
    EventType.RECONCILE_COMPLETED: ReconcileCompletedPayload,
    EventType.SYMBOLS_AUDITED: SymbolsAuditedPayload,
    EventType.SYMBOL_BLOCKED: SymbolBlockedPayload,
    EventType.SYMBOL_UNBLOCKED: SymbolBlockedPayload,
    EventType.DATA_PARTITION_SEALED: PartitionSealedPayload,
    EventType.DATA_BAR_REVISION_DETECTED: BarRevisionPayload,
    EventType.DATA_SNAPSHOT_SEALED: SnapshotSealedPayload,
    EventType.DATA_AUDIT_COMPLETED: DataAuditPayload,
    EventType.DATA_PROVIDER_DEGRADED: ProviderDegradedPayload,
    EventType.DATA_STALENESS_BREACH: StalenessBreachPayload,
    EventType.DATA_ACTION_RECORDED: ActionRecordedPayload,
    EventType.DATA_ACTION_RECONCILED: ActionReconciledPayload,
    EventType.DATA_UNIVERSE_SNAPSHOT_TAKEN: UniverseSnapshotPayload,
    EventType.DATA_BAKEOFF_COMPLETED: BakeoffPayload,
    EventType.DATA_REVISION_CANARY_COMPLETED: RevisionCanaryPayload,
    EventType.DATA_REGIME_READ: RegimeReadPayload,
    EventType.STRATEGY_SPEC_REGISTERED: StrategySpecPayload,
    EventType.BACKTEST_COMPLETED: BacktestPayload,
    EventType.BACKTEST_CALIBRATED: CalibrationPayload,
    # M4
    EventType.DECISION_MADE: DecisionPayload,
    EventType.RISK_EVALUATED: RiskEvaluationPayload,
    EventType.RISK_TOKEN_ISSUED: RiskTokenPayload,
    EventType.INTENT_COMMITTED: IntentCommittedPayload,
    EventType.ORDER_SUBMITTED: OrderSubmittedPayload,
    EventType.ORDER_ACKNOWLEDGED: OrderOutcomePayload,
    EventType.ORDER_REJECTED: OrderOutcomePayload,
    EventType.ORDER_CANCELLED: OrderOutcomePayload,
    EventType.INTENT_RESOLVED: IntentResolvedPayload,
    EventType.FILL_RECORDED: FillPayload,
    EventType.TRADE_CLOSED: TradeClosedPayload,
    EventType.POSITION_PROTECTED: ProtectionPayload,
    EventType.POSITION_UNPROTECTED: ProtectionPayload,
    EventType.WATCHDOG_TRIPPED: WatchdogPayload,
    EventType.WATCHDOG_UNREACHABLE: WatchdogPayload,
    EventType.INSTANCE_LOCK_ACQUIRED: InstanceLockPayload,
    EventType.INSTANCE_LOCK_REFUSED: InstanceLockPayload,
    EventType.INSTANCE_LOCK_RELEASED: InstanceLockPayload,
    EventType.LOOP_CYCLE_COMPLETED: LoopCyclePayload,
    EventType.TRIAL_RECORDED: TrialPayload,
    EventType.SEARCH_COMPLETED: SearchCompletedPayload,
    EventType.SPECS_PROPOSED: SpecsProposedPayload,
    EventType.HOLDOUT_EVALUATED: HoldoutEvaluatedPayload,
    EventType.HOLDOUT_VIOLATION_ATTEMPTED: HoldoutViolationPayload,
    EventType.PROMOTION_EVALUATED: PromotionPayload,
    EventType.STRATEGY_RETIRED: StrategyRetiredPayload,
    EventType.LINEAGE_BUDGET_EXHAUSTED: LineageBudgetPayload,
    EventType.LADDER_MOVED: LadderMovePayload,
    EventType.ALLOCATION_DECIDED: AllocationPayload,
    EventType.STRATEGY_REVIEWED: StrategyReviewedPayload,
    EventType.SESSION_REVIEWED: SessionReviewedPayload,
    # M5b
    EventType.BOOK_FUNDED: BookFundedPayload,
    EventType.POSITION_ORPHANED: PositionOrphanedPayload,
    # M7
    EventType.MODEL_RECORDED: ModelRecordedPayload,
    # M8
    EventType.BACKUP_CREATED: BackupCreatedPayload,
    EventType.BACKUP_RESTORED: BackupRestoredPayload,
    EventType.BACKUP_RESTORE_VERIFIED: BackupRestoreVerifiedPayload,
}

# The default aggregate each event type is filed under, so callers do not have
# to remember and cannot disagree with each other.
EVENT_AGGREGATES: dict[EventType, AggregateType] = {
    EventType.LEDGER_GENESIS: AggregateType.LEDGER,
    EventType.CHAIN_ANCHORED: AggregateType.LEDGER,
    EventType.RUN_STARTED: AggregateType.RUN,
    EventType.RUN_ENDED: AggregateType.RUN,
    EventType.CONFIG_PINNED: AggregateType.CONFIG,
    EventType.CONFIG_DRIFT_DETECTED: AggregateType.CONFIG,
    EventType.STATE_TRANSITIONED: AggregateType.SAFETY,
    EventType.HALT_RAISED: AggregateType.SAFETY,
    EventType.HALT_CLEARED: AggregateType.SAFETY,
    EventType.KILLSWITCH_ENGAGED: AggregateType.SAFETY,
    EventType.KILLSWITCH_RELEASED: AggregateType.SAFETY,
    EventType.HEARTBEAT_STALE: AggregateType.SAFETY,
    EventType.BROKER_PROBED: AggregateType.BROKER,
    EventType.BROKER_SCHEMA_DRIFT: AggregateType.BROKER,
    EventType.BROKER_RATE_LIMITED: AggregateType.BROKER,
    EventType.BROKER_SNAPSHOT_TAKEN: AggregateType.BROKER,
    EventType.RECONCILE_COMPLETED: AggregateType.BROKER,
    EventType.SYMBOLS_AUDITED: AggregateType.DATA,
    EventType.SYMBOL_BLOCKED: AggregateType.DATA,
    EventType.SYMBOL_UNBLOCKED: AggregateType.DATA,
    EventType.DATA_PARTITION_SEALED: AggregateType.DATA,
    EventType.DATA_BAR_REVISION_DETECTED: AggregateType.DATA,
    EventType.DATA_SNAPSHOT_SEALED: AggregateType.DATA,
    EventType.DATA_AUDIT_COMPLETED: AggregateType.DATA,
    EventType.DATA_PROVIDER_DEGRADED: AggregateType.DATA,
    EventType.DATA_STALENESS_BREACH: AggregateType.DATA,
    EventType.DATA_ACTION_RECORDED: AggregateType.DATA,
    EventType.DATA_ACTION_RECONCILED: AggregateType.DATA,
    EventType.DATA_UNIVERSE_SNAPSHOT_TAKEN: AggregateType.DATA,
    EventType.DATA_BAKEOFF_COMPLETED: AggregateType.DATA,
    EventType.DATA_REVISION_CANARY_COMPLETED: AggregateType.DATA,
    EventType.DATA_REGIME_READ: AggregateType.DATA,
    EventType.STRATEGY_SPEC_REGISTERED: AggregateType.STRATEGY,
    EventType.BACKTEST_COMPLETED: AggregateType.STRATEGY,
    # Filed under RUN, not STRATEGY: a calibration is a statement about this
    # build of the engine, and filing it under a strategy would imply it says
    # something about one.
    EventType.BACKTEST_CALIBRATED: AggregateType.RUN,
    # M4. The aggregate is what an event is *about*, which is why a fill is
    # filed under POSITION rather than ORDER: the order is how it happened,
    # the position is the thing that changed.
    EventType.DECISION_MADE: AggregateType.DECISION,
    EventType.RISK_EVALUATED: AggregateType.DECISION,
    EventType.RISK_TOKEN_ISSUED: AggregateType.DECISION,
    EventType.INTENT_COMMITTED: AggregateType.ORDER,
    EventType.ORDER_SUBMITTED: AggregateType.ORDER,
    EventType.ORDER_ACKNOWLEDGED: AggregateType.ORDER,
    EventType.ORDER_REJECTED: AggregateType.ORDER,
    EventType.ORDER_CANCELLED: AggregateType.ORDER,
    EventType.INTENT_RESOLVED: AggregateType.ORDER,
    EventType.FILL_RECORDED: AggregateType.POSITION,
    EventType.TRADE_CLOSED: AggregateType.STRATEGY,
    EventType.POSITION_PROTECTED: AggregateType.POSITION,
    EventType.POSITION_UNPROTECTED: AggregateType.POSITION,
    EventType.WATCHDOG_TRIPPED: AggregateType.SAFETY,
    EventType.WATCHDOG_UNREACHABLE: AggregateType.SAFETY,
    EventType.INSTANCE_LOCK_ACQUIRED: AggregateType.SAFETY,
    EventType.INSTANCE_LOCK_REFUSED: AggregateType.SAFETY,
    EventType.INSTANCE_LOCK_RELEASED: AggregateType.SAFETY,
    EventType.LOOP_CYCLE_COMPLETED: AggregateType.RUN,
    # M5. A trial and a search are facts about the *search* rather than about
    # any one strategy — filing a rejected candidate under STRATEGY would
    # create an aggregate per discarded idea, which is the opposite of what an
    # aggregate is for. Everything downstream of the gate is about a strategy.
    EventType.TRIAL_RECORDED: AggregateType.RUN,
    EventType.SEARCH_COMPLETED: AggregateType.RUN,
    EventType.SPECS_PROPOSED: AggregateType.RUN,
    EventType.HOLDOUT_EVALUATED: AggregateType.STRATEGY,
    # Filed under SAFETY rather than STRATEGY: an attempt to read past the
    # seal is a fault in the process, and it is the kind of thing an operator
    # should find by filtering for safety events rather than by knowing which
    # strategy to look under.
    EventType.HOLDOUT_VIOLATION_ATTEMPTED: AggregateType.SAFETY,
    EventType.PROMOTION_EVALUATED: AggregateType.STRATEGY,
    EventType.STRATEGY_RETIRED: AggregateType.STRATEGY,
    EventType.LINEAGE_BUDGET_EXHAUSTED: AggregateType.STRATEGY,
    EventType.LADDER_MOVED: AggregateType.STRATEGY,
    EventType.ALLOCATION_DECIDED: AggregateType.RUN,
    EventType.STRATEGY_REVIEWED: AggregateType.STRATEGY,
    EventType.SESSION_REVIEWED: AggregateType.RUN,
    # M5b. The book is a fact about the run: the same strategies funded under a
    # different allocation are a different run's book, and filing it under one
    # of the strategies would hide the ones that were excluded.
    EventType.BOOK_FUNDED: AggregateType.RUN,
    EventType.POSITION_ORPHANED: AggregateType.POSITION,
    # M7. A model is its own aggregate rather than a strategy's: one model can
    # be read by many specs, and filing it under any one of them would hide it
    # from the others' histories.
    EventType.MODEL_RECORDED: AggregateType.MODEL,
    # M8. Backups are filed with the ledger they copy.
    EventType.BACKUP_CREATED: AggregateType.LEDGER,
    EventType.BACKUP_RESTORED: AggregateType.LEDGER,
    EventType.BACKUP_RESTORE_VERIFIED: AggregateType.LEDGER,
}


def payload_model(event_type: EventType) -> type[EventPayload]:
    """The payload model registered for `event_type`."""
    try:
        return EVENT_PAYLOADS[event_type]
    except KeyError as exc:  # pragma: no cover - guarded by a completeness test
        raise KeyError(
            f"{event_type} has no registered payload model. Add one in "
            "tb/ledger/events.py rather than appending an untyped dictionary."
        ) from exc


def default_aggregate(event_type: EventType) -> AggregateType:
    """The aggregate type `event_type` is filed under."""
    try:
        return EVENT_AGGREGATES[event_type]
    except KeyError as exc:  # pragma: no cover - guarded by a completeness test
        raise KeyError(f"{event_type} has no registered aggregate type") from exc
