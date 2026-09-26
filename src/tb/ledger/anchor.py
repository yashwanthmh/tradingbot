"""Publishing the chain head outside the database.

The uncomfortable truth about a hash-chained log: it is tamper-evident only
against an adversary who cannot rewrite the file. The process writing the
ledger *can* rewrite it, and a process capable of rewriting history is also
capable of recomputing every hash so the chain verifies perfectly. Internal
verification would pass. The log would be a clean, consistent fiction.

The fix is cheap: periodically publish `(seq, chain_hash)` somewhere the bot
cannot retroactively edit. Then rewriting history still produces a valid chain,
but one that disagrees with a head you already published — and `verify_chain`
checks exactly that.

Sinks differ enormously in how much they are worth, so each one reports its own
trust boundary honestly rather than letting a file on the same disk masquerade
as evidence.
"""

from __future__ import annotations

import json
import subprocess
import uuid
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol, runtime_checkable

from tb.core.clock import now_iso
from tb.core.errors import LedgerError
from tb.ledger.events import ChainAnchoredPayload, EventType
from tb.ledger.store import Ledger


@dataclass(frozen=True, slots=True)
class AnchorResult:
    anchor_id: str
    seq: int
    chain_hash: str
    sink: str
    external_ref: str | None
    trust_boundary: str
    crosses_trust_boundary: bool


@runtime_checkable
class AnchorSink(Protocol):
    """Somewhere a chain head can be published."""

    @property
    def name(self) -> str: ...

    @property
    def trust_boundary(self) -> str:
        """Plain-English description of what this sink actually protects against."""
        ...

    @property
    def crosses_trust_boundary(self) -> bool:
        """True only if this sink is genuinely beyond the bot's reach."""
        ...

    def publish(self, *, seq: int, chain_hash: str, anchored_at: str) -> str | None:
        """Publish the head. Returns an external reference, if the sink has one."""
        ...


class FileAnchorSink:
    """Append the head to a JSONL file.

    Weak on its own: a process that can rewrite the ledger can usually rewrite a
    file next to it. It becomes real when the file lives in version control and
    is pushed to a remote — at which point the remote, not the file, is the
    boundary. The default path is inside `journal/`, which is committed, for
    exactly that reason.
    """

    def __init__(self, path: str | Path = "journal/chain-heads.jsonl") -> None:
        self.path = Path(path)

    @property
    def name(self) -> str:
        return "file"

    @property
    def trust_boundary(self) -> str:
        return (
            f"local file {self.path}. This is only evidence once it is committed and "
            "pushed to a remote the bot cannot force-push; on its own, a process that "
            "can rewrite the ledger can rewrite this too."
        )

    @property
    def crosses_trust_boundary(self) -> bool:
        return False

    def publish(self, *, seq: int, chain_hash: str, anchored_at: str) -> str | None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        record = {"seq": seq, "chain_hash": chain_hash, "anchored_at": anchored_at}
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, sort_keys=True) + "\n")
        return f"{self.path}#{seq}"


class GitAnchorSink:
    """Append the head to a file and commit it.

    Genuinely useful once the branch is pushed: the commit hash covers the
    anchor content, and rewriting it on a remote you do not control is not
    something the bot can do. Requires a git checkout with a usable identity.

    `also` commits other files in the same commit — the journal's pages, so a
    page and the head it was written from arrive on the remote together.
    """

    def __init__(
        self,
        path: str | Path = "journal/chain-heads.jsonl",
        *,
        repo_root: str | Path = ".",
        also: Sequence[str | Path] = (),
        message: str | None = None,
    ) -> None:
        self.path = Path(path)
        self.repo_root = Path(repo_root)
        self.also = tuple(Path(extra) for extra in also)
        self.message = message

    @property
    def name(self) -> str:
        return "git"

    @property
    def trust_boundary(self) -> str:
        return (
            "a git commit. Real evidence once pushed to a remote that rejects "
            "force-pushes; until it is pushed, it is local history the bot could rewrite."
        )

    @property
    def crosses_trust_boundary(self) -> bool:
        return True

    def _git(self, *args: str) -> subprocess.CompletedProcess[str]:
        return subprocess.run(  # noqa: S603 - fixed argv, no shell
            ["git", "-C", str(self.repo_root), *args],  # noqa: S607
            capture_output=True,
            text=True,
            timeout=30,
            check=False,
        )

    def publish(self, *, seq: int, chain_hash: str, anchored_at: str) -> str | None:
        FileAnchorSink(self.path).publish(seq=seq, chain_hash=chain_hash, anchored_at=anchored_at)

        # Absolute, so the paths mean the same thing whichever directory git
        # is pointed at.
        paths = [str(p.resolve()) for p in (self.path, *self.also)]
        added = self._git("add", "--", *paths)
        if added.returncode != 0:
            raise LedgerError(f"git add failed while anchoring: {added.stderr.strip()}")

        committed = self._git(
            "commit",
            "--no-verify",
            "-m",
            self.message or f"ledger: anchor chain head at seq={seq}",
            "--",
            *paths,
        )
        if committed.returncode != 0:
            raise LedgerError(f"git commit failed while anchoring: {committed.stderr.strip()}")

        rev = self._git("rev-parse", "HEAD")
        return rev.stdout.strip() if rev.returncode == 0 else None


def anchor_head(ledger: Ledger, sink: AnchorSink | None = None) -> AnchorResult:
    """Publish the current chain head and record that it was published.

    Order matters: publish externally *first*, then record. Recording first and
    failing to publish would leave a phantom anchor that makes verification fail
    against a head nobody can produce. The reverse — published but unrecorded —
    is harmless: verification simply does not know to check it.
    """
    sink = sink or FileAnchorSink()

    head = ledger.head()
    if head is None:
        raise LedgerError("nothing to anchor: the chain is empty")

    anchor_id = f"anc_{uuid.uuid4().hex[:16]}"
    anchored_at = now_iso()

    external_ref = sink.publish(seq=head.seq, chain_hash=head.chain_hash, anchored_at=anchored_at)

    payload = ChainAnchoredPayload(
        anchor_id=anchor_id,
        anchored_seq=head.seq,
        anchored_chain_hash=head.chain_hash,
        sink=sink.name,
        external_ref=external_ref,
    )
    with ledger.transaction() as tx:
        tx.append(EventType.CHAIN_ANCHORED, anchor_id, payload)
        tx.execute(
            """
            INSERT INTO chain_anchor (
                anchor_id, seq, chain_hash, anchored_at, sink, external_ref, note
            ) VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (
                anchor_id,
                head.seq,
                head.chain_hash,
                anchored_at,
                sink.name,
                external_ref,
                sink.trust_boundary,
            ),
        )

    return AnchorResult(
        anchor_id=anchor_id,
        seq=head.seq,
        chain_hash=head.chain_hash,
        sink=sink.name,
        external_ref=external_ref,
        trust_boundary=sink.trust_boundary,
        crosses_trust_boundary=sink.crosses_trust_boundary,
    )


SINKS: dict[str, type[FileAnchorSink] | type[GitAnchorSink]] = {
    "file": FileAnchorSink,
    "git": GitAnchorSink,
}
