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
