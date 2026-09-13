"""Chain verification.

Four independent checks, because each catches a different attack:

1. **Contiguity** — sequence numbers run 1..N with no gaps. Catches a deleted row.
2. **Payload integrity** — re-hashing the stored `payload_json` reproduces the
   stored `payload_hash`. Catches an edited payload.
3. **Link integrity** — recomputing `chain_hash` from the row's own fields and
   its predecessor's hash reproduces the stored `chain_hash`. Catches an edited
   timestamp, actor, or event type, and catches reordering.
4. **Anchor agreement** — every externally published head still matches the
   chain at that sequence number. This is the only check that catches a *whole
   chain re-signing*: an adversary with write access can rewrite history and
   recompute every hash so that checks 1-3 all pass. What they cannot do is
   change the head hash you already published somewhere they do not control.

Failures name the exact sequence number. "The ledger is corrupt" is not an
actionable message; "payload at seq=4173 was edited" is.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum

from tb.ledger.chain import GENESIS_HASH, compute_chain_hash, compute_payload_hash
from tb.ledger.store import Ledger


class FindingKind(StrEnum):
    SEQUENCE_GAP = "sequence_gap"
    PAYLOAD_TAMPERED = "payload_tampered"
    CHAIN_HASH_MISMATCH = "chain_hash_mismatch"
    BROKEN_LINK = "broken_link"
    ANCHOR_MISMATCH = "anchor_mismatch"
    ANCHOR_BEYOND_CHAIN = "anchor_beyond_chain"


@dataclass(frozen=True, slots=True)
class Finding:
    kind: FindingKind
    seq: int
    detail: str

    def __str__(self) -> str:
        return f"seq={self.seq} [{self.kind}] {self.detail}"


@dataclass(slots=True)
class VerificationReport:
    events_checked: int = 0
    anchors_checked: int = 0
    head_seq: int | None = None
    head_hash: str | None = None
    findings: list[Finding] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return not self.findings

    @property
    def first_failure(self) -> Finding | None:
        """The earliest problem, which is the one worth investigating.

        Later findings are usually consequences: tampering at seq=100 makes
        every subsequent link mismatch too.
        """
        return min(self.findings, key=lambda f: f.seq) if self.findings else None

    def summary(self) -> str:
        if self.ok:
            return (
                f"chain intact: {self.events_checked} events verified, "
                f"{self.anchors_checked} anchors agree, "
                f"head seq={self.head_seq} {(self.head_hash or '')[:12]}…"
            )
        first = self.first_failure
        assert first is not None
        return (
            f"CHAIN VERIFICATION FAILED: {len(self.findings)} finding(s) over "
            f"{self.events_checked} events. Earliest: {first}"
        )


def verify_chain(
    ledger: Ledger,
    *,
    start_seq: int = 1,
    stop_on_first: bool = False,
    check_anchors: bool = True,
) -> VerificationReport:
    """Walk the chain and check it.

    `stop_on_first` is for the hot path (a startup check that only needs a
    yes/no). The default walks the whole chain, because when investigating you
    want to know whether one row was touched or the last thousand.
    """
    report = VerificationReport()

    expected_seq = start_seq
    # Verifying a suffix cannot know its predecessor's hash from thin air, so
    # for start_seq > 1 the previous row's stored hash seeds the walk. That
    # makes a suffix check weaker than a full one by construction, which is why
    # the default is a full walk.
    prev_hash: str | None = None
    if start_seq > 1:
        previous = ledger.get(start_seq - 1)
        prev_hash = previous["chain_hash"] if previous else None
    else:
        prev_hash = GENESIS_HASH

    last_seq: int | None = None
    last_hash: str | None = None

    for row in ledger.iter_events(start_seq=start_seq):
        seq = int(row["seq"])

        if seq != expected_seq:
            report.findings.append(
                Finding(
                    kind=FindingKind.SEQUENCE_GAP,
                    seq=expected_seq,
                    detail=(
                        f"expected seq={expected_seq} but found seq={seq}; "
                        f"{seq - expected_seq} event(s) are missing from the log"
                    ),
                )
            )
            if stop_on_first:
                break
            # Resync so the rest of the walk still reports useful findings.
            expected_seq = seq

        recomputed_payload = compute_payload_hash(row["payload_json"])
        if recomputed_payload != row["payload_hash"]:
            report.findings.append(
                Finding(
                    kind=FindingKind.PAYLOAD_TAMPERED,
                    seq=seq,
                    detail=(
                        f"payload_json does not hash to the stored payload_hash "
                        f"(stored {row['payload_hash'][:12]}…, "
                        f"recomputed {recomputed_payload[:12]}…)"
                    ),
                )
            )
            if stop_on_first:
                break

        if prev_hash is not None and row["prev_hash"] != prev_hash:
            report.findings.append(
                Finding(
                    kind=FindingKind.BROKEN_LINK,
                    seq=seq,
                    detail=(
                        f"prev_hash {row['prev_hash'][:12]}… does not match the "
                        f"preceding event's chain_hash {prev_hash[:12]}…"
                    ),
                )
            )
            if stop_on_first:
                break

        recomputed_chain = compute_chain_hash(
            prev_hash=row["prev_hash"],
            seq=seq,
            ts_utc=row["ts_utc"],
            event_type=row["event_type"],
            aggregate_type=row["aggregate_type"],
            aggregate_id=row["aggregate_id"],
            actor=row["actor"],
            # The row's own stored payload_hash, so this check isolates the link
            # fields; payload edits are caught by the check above.
            payload_hash=row["payload_hash"],
        )
        if recomputed_chain != row["chain_hash"]:
            report.findings.append(
                Finding(
                    kind=FindingKind.CHAIN_HASH_MISMATCH,
                    seq=seq,
                    detail=(
                        "one of ts_utc / event_type / aggregate_type / aggregate_id / "
                        "actor was altered: the row's fields do not reproduce its "
                        f"stored chain_hash {row['chain_hash'][:12]}… "
                        f"(recomputed {recomputed_chain[:12]}…)"
                    ),
                )
            )
            if stop_on_first:
                break

        report.events_checked += 1
        prev_hash = row["chain_hash"]
        last_seq, last_hash = seq, row["chain_hash"]
        expected_seq = seq + 1

    report.head_seq = last_seq
    report.head_hash = last_hash

    if check_anchors:
        _verify_anchors(ledger, report)

    return report


def _verify_anchors(ledger: Ledger, report: VerificationReport) -> None:
    """Compare every published head against the chain as it stands now."""
    for anchor in ledger.anchors():
        seq = int(anchor["seq"])
        report.anchors_checked += 1
        row = ledger.get(seq)
        if row is None:
            report.findings.append(
                Finding(
                    kind=FindingKind.ANCHOR_BEYOND_CHAIN,
                    seq=seq,
                    detail=(
                        f"anchor {anchor['anchor_id']} (sink={anchor['sink']}) was "
                        f"published at seq={seq}, but the chain now ends before it — "
                        "the log has been truncated since that head was published"
                    ),
                )
            )
            continue
        if row["chain_hash"] != anchor["chain_hash"]:
            report.findings.append(
                Finding(
                    kind=FindingKind.ANCHOR_MISMATCH,
                    seq=seq,
                    detail=(
                        f"anchor {anchor['anchor_id']} (sink={anchor['sink']}, "
                        f"ref={anchor['external_ref']}) published "
                        f"{anchor['chain_hash'][:12]}… at seq={seq}, but the chain now "
                        f"holds {row['chain_hash'][:12]}…. History was rewritten and "
                        "re-signed after that head was published."
                    ),
                )
            )
