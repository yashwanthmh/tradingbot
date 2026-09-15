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
