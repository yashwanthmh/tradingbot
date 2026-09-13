"""The audit ledger: an append-only, hash-chained record of every move.

`event_log` is the single source of truth. Every other table in this database is
a projection of it, which is what makes the strongest test in the suite possible:
rebuild every derived table from the event log and assert it comes out identical.
"""

from tb.ledger.anchor import AnchorResult, FileAnchorSink, GitAnchorSink, anchor_head
from tb.ledger.chain import GENESIS_HASH, compute_chain_hash, compute_payload_hash
from tb.ledger.events import Actor, AggregateType, EventType
from tb.ledger.store import AppendedEvent, ChainHead, Ledger, default_ledger_path
from tb.ledger.verify import Finding, FindingKind, VerificationReport, verify_chain

__all__ = [
    "GENESIS_HASH",
    "Actor",
    "AggregateType",
    "AnchorResult",
    "AppendedEvent",
    "ChainHead",
    "EventType",
    "FileAnchorSink",
    "Finding",
    "FindingKind",
    "GitAnchorSink",
    "Ledger",
    "VerificationReport",
    "anchor_head",
    "compute_chain_hash",
    "compute_payload_hash",
    "default_ledger_path",
    "verify_chain",
]
