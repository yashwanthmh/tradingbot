"""The hash chain.

Each event's `chain_hash` covers its own identifying fields *and* the previous
event's `chain_hash`, so altering any historical row invalidates every row after
it. The first event chains to `GENESIS_HASH` rather than to NULL, so "this is
the start of the chain" is itself an assertion that can be checked.

The link is hashed over **canonical JSON with named keys**, not over
concatenated strings. That is not stylistic: concatenating `"ab" + "c"` and
`"a" + "bc"` yields identical bytes, so a naive `prev + seq + type + ...` chain
lets an attacker shift content across field boundaries without changing the
hash. Named keys make each field's extent unambiguous.
"""

from __future__ import annotations

from tb.core.canonical import GENESIS_HASH, hash_payload, sha256_hex

__all__ = ["GENESIS_HASH", "compute_chain_hash", "compute_payload_hash"]


def compute_payload_hash(payload_json: str) -> str:
    """SHA-256 over the exact canonical payload string that gets stored.

    Taking the already-serialised string (rather than re-serialising the object)
    is what makes verification honest: the verifier re-hashes the bytes in the
    row, so a round-trip bug in serialisation cannot hide behind a matching hash.
    """
    return sha256_hex(payload_json)


def compute_chain_hash(
    *,
    prev_hash: str,
    seq: int,
    ts_utc: str,
    event_type: str,
    aggregate_type: str,
    aggregate_id: str,
    actor: str,
    payload_hash: str,
) -> str:
    """The chain hash for one event.

    Every argument is keyword-only. A positional call site that silently swapped
    `aggregate_type` and `aggregate_id` would produce a chain that verifies
    perfectly and describes the wrong thing.
    """
    return hash_payload(
        {
            "prev_hash": prev_hash,
            "seq": seq,
            "ts_utc": ts_utc,
            "event_type": event_type,
            "aggregate_type": aggregate_type,
            "aggregate_id": aggregate_id,
            "actor": actor,
            "payload_hash": payload_hash,
        }
    )
