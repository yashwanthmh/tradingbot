"""Backups a clean machine can restore, and the evidence that one did.

A backup is everything a restore needs and nothing it must not carry:

* the ledger, copied with SQLite's online backup API, so a loop writing to it
  throughout gets a consistent copy rather than a torn one;
* every Parquet file the ledger's partition catalogue names and every model
  artifact its model store names. The copied ledger says which, so a stray file
  is never carried and a missing or altered one fails the backup, now, rather
  than the restore, later;
* the limits file, for reference;
* and a manifest hashing all of it. The manifest's own hash is the backup's
  identity.

Never secrets. Keys live in the environment or the keyring, never in the
ledger, and the raw archive inside it was scrubbed of them on the way in.

**Restore** refuses anything but an empty target — restoring over a ledger
would merge two histories — checks the backup before copying and the copy
after, walks the restored chain, and replays the most recent fills from the
restored files alone: the strongest evidence a restore can give is that the
rebuilt machine explains the last trades the old one made. What it checked
is written into the restored ledger and into a receipt.

**The receipt** goes back to the machine the backup came from, where `tb backup
receipt` accepts it only for a backup that ledger recorded making, and records
the restore there. That event, not the restore itself, is what `tb arm --live`
reads: the machine that will trade is the one that has to know its state can
be rebuilt without it.
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import socket
import sqlite3
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

from tb.config.loader import PinnedLimits
from tb.core.clock import from_iso, now_utc, to_iso
from tb.core.errors import TbError
from tb.engine.replay import ReplayError, replay_fill
from tb.ledger.events import (
    Actor,
    BackupCreatedPayload,
    BackupRestoredPayload,
    BackupRestoreVerifiedPayload,
    EventType,
)
from tb.ledger.store import Ledger, code_git_sha
from tb.ledger.verify import verify_chain
from tb.registry.model_store import ModelStore

BACKUP_VERSION = 1
MANIFEST = "MANIFEST.json"
RECEIPT = "restore-receipt.json"
LEDGER_FILE = "ledger.db"
LIMITS_FILE = "hard_limits.yaml"
BARS_DIR = "bars"
MODELS_DIR = "models"
DEFAULT_DESTINATION = Path("var/backups")

# How many of the most recent fills a restore replays. Enough to show the
# rebuilt machine can explain what the old one did; a full replay is `tb
# replay`, run fill by fill, for as long as anyone wants.
REPLAY_FILLS = 3

_CHUNK = 1 << 20


class BackupError(TbError):
    """A backup could not be made, read, restored or accepted."""


@dataclass(frozen=True, slots=True)
class BackupFile:
    path: str
    sha256: str
    size: int
    kind: str

    def as_dict(self) -> dict[str, Any]:
        return {"path": self.path, "sha256": self.sha256, "bytes": self.size, "kind": self.kind}


@dataclass(frozen=True, slots=True)
class Manifest:
    backup_id: str
    created_at: datetime
    host: str
    code_git_sha: str | None
    head_seq: int
    head_chain_hash: str
    ledger_schema_version: int | None
    limits_config_hash: str
    files: tuple[BackupFile, ...]

    @property
    def total_bytes(self) -> int:
        return sum(f.size for f in self.files)

    def render(self) -> str:
        body = {
            "tb_backup": BACKUP_VERSION,
            "backup_id": self.backup_id,
            "created_at": to_iso(self.created_at),
            "host": self.host,
            "code_git_sha": self.code_git_sha,
            "ledger": {
                "head_seq": self.head_seq,
                "head_chain_hash": self.head_chain_hash,
                "schema_version": self.ledger_schema_version,
            },
            "limits_config_hash": self.limits_config_hash,
            "files": [f.as_dict() for f in self.files],
        }
        return json.dumps(body, indent=2, sort_keys=True) + "\n"

    @classmethod
    def parse(cls, text: str) -> Manifest:
        try:
            body = json.loads(text)
            if body.get("tb_backup") != BACKUP_VERSION:
                raise BackupError(
                    f"backup format {body.get('tb_backup')!r}; this build reads {BACKUP_VERSION}"
                )
            return cls(
                backup_id=str(body["backup_id"]),
                created_at=from_iso(str(body["created_at"])),
                host=str(body["host"]),
                code_git_sha=body.get("code_git_sha"),
                head_seq=int(body["ledger"]["head_seq"]),
                head_chain_hash=str(body["ledger"]["head_chain_hash"]),
                ledger_schema_version=body["ledger"].get("schema_version"),
                limits_config_hash=str(body["limits_config_hash"]),
                files=tuple(
                    BackupFile(
                        path=str(f["path"]),
                        sha256=str(f["sha256"]),
                        size=int(f["bytes"]),
                        kind=str(f["kind"]),
                    )
                    for f in body["files"]
                ),
            )
        except BackupError:
            raise
        except (ValueError, KeyError, TypeError) as exc:
            raise BackupError(f"the manifest is malformed: {exc}") from exc


@dataclass(frozen=True, slots=True)
class BackupResult:
    backup_id: str
    path: Path
    manifest_sha256: str
    manifest: Manifest


@dataclass(frozen=True, slots=True)
class BackupCheck:
    """Whether a backup is whole: every file, the chain, and the catalogues."""

    ok: bool
    manifest: Manifest | None
    manifest_sha256: str | None
    problems: tuple[str, ...] = ()
    checks: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class RestoreReceipt:
    """What a restore checked, carried back to the ledger the backup came from."""

    backup_id: str
    manifest_sha256: str
    source_host: str
    restored_host: str
    restored_at: datetime
    code_git_sha: str | None
    restored_head_seq: int
    restored_head_chain_hash: str
    checks: tuple[str, ...]
    n_fills_replayed: int

    def render(self) -> str:
        body = {
            "tb_restore_receipt": BACKUP_VERSION,
            "backup_id": self.backup_id,
            "manifest_sha256": self.manifest_sha256,
            "source_host": self.source_host,
            "restored_host": self.restored_host,
            "restored_at": to_iso(self.restored_at),
            "code_git_sha": self.code_git_sha,
            "restored_head": {
                "seq": self.restored_head_seq,
                "chain_hash": self.restored_head_chain_hash,
            },
            "checks": list(self.checks),
            "n_fills_replayed": self.n_fills_replayed,
        }
        return json.dumps(body, indent=2, sort_keys=True) + "\n"

    @classmethod
    def parse(cls, text: str) -> RestoreReceipt:
        try:
            body = json.loads(text)
            if body.get("tb_restore_receipt") != BACKUP_VERSION:
                raise BackupError("this is not a restore receipt this build can read")
            return cls(
                backup_id=str(body["backup_id"]),
                manifest_sha256=str(body["manifest_sha256"]),
                source_host=str(body["source_host"]),
                restored_host=str(body["restored_host"]),
                restored_at=from_iso(str(body["restored_at"])),
                code_git_sha=body.get("code_git_sha"),
                restored_head_seq=int(body["restored_head"]["seq"]),
                restored_head_chain_hash=str(body["restored_head"]["chain_hash"]),
                checks=tuple(str(c) for c in body.get("checks", [])),
                n_fills_replayed=int(body.get("n_fills_replayed", 0)),
            )
        except BackupError:
            raise
        except (ValueError, KeyError, TypeError) as exc:
            raise BackupError(f"the receipt is malformed: {exc}") from exc


@dataclass(frozen=True, slots=True)
class VerifiedRestore:
    """A restore the ledger has on record, as the live gate reads it."""

    seq: int
    backup_id: str
    restored_host: str
    restored_at: datetime
    same_host: bool
    n_fills_replayed: int
    checks: tuple[str, ...] = field(default_factory=tuple)


# --------------------------------------------------------------------------
# Making one
# --------------------------------------------------------------------------


def create_backup(
    ledger: Ledger,
    *,
    destination: Path,
    bars_root: Path,
    models_root: Path,
    limits: PinnedLimits,
    at: datetime | None = None,
) -> BackupResult:
    """Copy the ledger and everything it names, hash it all, and record that it was done."""
    moment = at or now_utc()
    backup_id = f"bkp_{moment:%Y%m%dT%H%M%S}_{uuid.uuid4().hex[:8]}"
    root = destination / backup_id
    root.mkdir(parents=True, exist_ok=False)
    try:
        files = [_copy_ledger(ledger, root / LEDGER_FILE)]
        with Ledger(root / LEDGER_FILE, read_only=True) as copy:
            head = copy.head()
            if head is None:
                raise BackupError("the ledger is empty; there is nothing to back up")
            schema_version = copy.schema_version()
            for sha, relative in _catalogue(copy, "data_partitions", "file_sha256"):
                files.append(
                    _copy_named(bars_root / relative, root / BARS_DIR / relative, sha, "partition")
                )
            for sha, relative in _catalogue(copy, "ml_models", "artifact_sha256"):
                files.append(
                    _copy_named(models_root / relative, root / MODELS_DIR / relative, sha, "model")
                )
        files.append(
            _copy_named(limits.source_path, root / LIMITS_FILE, limits.content_sha256, "limits")
        )
        manifest = Manifest(
            backup_id=backup_id,
            created_at=moment,
            host=socket.gethostname(),
            code_git_sha=code_git_sha(),
            head_seq=head.seq,
            head_chain_hash=head.chain_hash,
            ledger_schema_version=schema_version,
            limits_config_hash=limits.config_hash,
            files=tuple(_relative(f, root) for f in files),
        )
        text = manifest.render()
        (root / MANIFEST).write_text(text, encoding="utf-8")
    except BaseException:
        # Half a backup is worse than none: it looks like one.
        shutil.rmtree(root, ignore_errors=True)
        raise

    manifest_sha256 = hashlib.sha256(text.encode("utf-8")).hexdigest()
    ledger.append(
        EventType.BACKUP_CREATED,
        backup_id,
        BackupCreatedPayload(
            backup_id=backup_id,
            manifest_sha256=manifest_sha256,
            head_seq=manifest.head_seq,
            head_chain_hash=manifest.head_chain_hash,
            n_files=len(manifest.files),
            n_bytes=manifest.total_bytes,
            host=manifest.host,
            destination=str(root),
        ),
        actor=Actor.HUMAN,
    )
    return BackupResult(
        backup_id=backup_id, path=root, manifest_sha256=manifest_sha256, manifest=manifest
    )


def _copy_ledger(ledger: Ledger, target: Path) -> BackupFile:
    """A consistent copy of a ledger that may be written to throughout."""
    copy = sqlite3.connect(target)
    try:
        ledger.conn.backup(copy)
        # One self-contained file, not a database and a write-ahead log.
        copy.execute("PRAGMA journal_mode=DELETE")
    finally:
        copy.close()
    sha, size = _hash(target)
    return BackupFile(path=str(target), sha256=sha, size=size, kind="ledger")


def _catalogue(ledger: Ledger, table: str, column: str) -> list[tuple[str, str]]:
    rows = ledger.conn.execute(
        f"SELECT {column} AS sha, relative_path FROM {table} ORDER BY relative_path"  # noqa: S608
    ).fetchall()
    return [(str(row["sha"]), str(row["relative_path"])) for row in rows]


def _copy_named(source: Path, target: Path, sha256: str, kind: str) -> BackupFile:
    """Copy a file the ledger names, refusing one that is not what it names."""
    if not source.is_file():
        raise BackupError(
            f"the ledger names {kind} {source}, and it is not there. Backing up a ledger "
            "that names files it does not have would hand the restore the same hole."
        )
    target.parent.mkdir(parents=True, exist_ok=True)
    digest = hashlib.sha256()
    size = 0
    with source.open("rb") as src, target.open("wb") as dst:
        while chunk := src.read(_CHUNK):
            digest.update(chunk)
            size += len(chunk)
            dst.write(chunk)
        dst.flush()
        os.fsync(dst.fileno())
    if digest.hexdigest() != sha256:
        raise BackupError(
            f"{kind} {source} does not match the hash the ledger recorded for it "
            f"({digest.hexdigest()[:12]}…, recorded {sha256[:12]}…). It changed after it "
            "was recorded; a backup of it would preserve the damage."
        )
    return BackupFile(path=str(target), sha256=sha256, size=size, kind=kind)


def _relative(entry: BackupFile, root: Path) -> BackupFile:
    return BackupFile(
        path=Path(entry.path).relative_to(root).as_posix(),
        sha256=entry.sha256,
        size=entry.size,
        kind=entry.kind,
    )


def _hash(path: Path) -> tuple[str, int]:
    digest = hashlib.sha256()
    size = 0
    with path.open("rb") as handle:
        while chunk := handle.read(_CHUNK):
            digest.update(chunk)
            size += len(chunk)
    return digest.hexdigest(), size


# --------------------------------------------------------------------------
# Checking one
# --------------------------------------------------------------------------


def verify_backup(root: Path) -> BackupCheck:
    """Every file present and unaltered, the chain intact, and nothing it names missing."""
    try:
        text = (root / MANIFEST).read_text(encoding="utf-8")
    except OSError as exc:
        return BackupCheck(False, None, None, (f"no readable manifest in {root}: {exc}",))
    try:
        manifest = Manifest.parse(text)
    except BackupError as exc:
        return BackupCheck(False, None, None, (str(exc),))
    manifest_sha256 = hashlib.sha256(text.encode("utf-8")).hexdigest()
    problems = _check_files(root, manifest)
    checks = [f"{len(manifest.files)} file(s) present, each matching its hash"]
    if not problems:
        chain_problems, chain_checks = _check_ledger(root / LEDGER_FILE, manifest)
        problems += chain_problems
        checks += chain_checks
    return BackupCheck(
        ok=not problems,
        manifest=manifest,
        manifest_sha256=manifest_sha256,
        problems=tuple(problems),
        checks=tuple(checks) if not problems else (),
    )


def _check_files(root: Path, manifest: Manifest) -> list[str]:
    problems: list[str] = []
    if not any(f.kind == "ledger" and f.path == LEDGER_FILE for f in manifest.files):
        problems.append("the manifest names no ledger")
    for entry in manifest.files:
        path = root / entry.path
        if not path.resolve().is_relative_to(root.resolve()):
            problems.append(f"{entry.path} points outside the backup")
            continue
        if not path.is_file():
            problems.append(f"{entry.path} is missing")
            continue
        sha, size = _hash(path)
        if sha != entry.sha256 or size != entry.size:
            problems.append(f"{entry.path} does not match its hash: it changed after the backup")
    return problems


def _check_ledger(path: Path, manifest: Manifest) -> tuple[list[str], list[str]]:
    """The copied chain intact, its head the one the manifest names, its catalogues whole."""
    problems: list[str] = []
    with Ledger(path, read_only=True) as copy:
        report = verify_chain(copy)
        if not report.ok:
            return [f"the backed-up chain does not verify: {report.summary()}"], []
        head = copy.head()
        if head is None or (head.seq, head.chain_hash) != (
            manifest.head_seq,
            manifest.head_chain_hash,
        ):
            problems.append("the ledger's head is not the one the manifest names")
        named = {(f.kind, f.sha256) for f in manifest.files}
        for kind, table, column in (
            ("partition", "data_partitions", "file_sha256"),
            ("model", "ml_models", "artifact_sha256"),
        ):
            for sha, relative in _catalogue(copy, table, column):
                if (kind, sha) not in named:
                    problems.append(
                        f"the ledger names {kind} {relative}, which is not in the backup"
                    )
    checks = [
        f"chain verified through seq {manifest.head_seq}",
        "every partition and model the ledger names is in the backup",
    ]
    return problems, checks


# --------------------------------------------------------------------------
# Restoring one
# --------------------------------------------------------------------------


def restore_backup(root: Path, target: Path, *, at: datetime | None = None) -> RestoreReceipt:
    """Rebuild a ledger and its stores in an empty directory, and prove it.

    Writes the receipt beside the restored files as well as returning it.
    """
    if target.exists() and any(target.iterdir()):
        raise BackupError(
            f"{target} is not empty. Restore into an empty directory: restoring over an "
            "existing ledger would merge two histories into one that neither machine wrote."
        )
    check = verify_backup(root)
    if not check.ok or check.manifest is None or check.manifest_sha256 is None:
        raise BackupError("the backup does not verify: " + "; ".join(check.problems))
    manifest = check.manifest

    target.mkdir(parents=True, exist_ok=True)
    for entry in manifest.files:
        destination = target / entry.path
        destination.parent.mkdir(parents=True, exist_ok=True)
        _copy_named(root / entry.path, destination, entry.sha256, entry.kind)
    restored = verify_backup_files_at(target, manifest)
    if restored:
        raise BackupError("the restored copy does not match the backup: " + "; ".join(restored))

    checks = list(check.checks)
    moment = at or now_utc()
    with Ledger(target / LEDGER_FILE) as ledger:
        report = verify_chain(ledger)
        if not report.ok:
            raise BackupError(f"the restored chain does not verify: {report.summary()}")
        replayed, replay_checks = _replay_recent(ledger, target / MODELS_DIR)
        checks += [f"restored chain verified through seq {manifest.head_seq}", *replay_checks]
        ledger.append(
            EventType.BACKUP_RESTORED,
            manifest.backup_id,
            BackupRestoredPayload(
                backup_id=manifest.backup_id,
                manifest_sha256=check.manifest_sha256,
                source_host=manifest.host,
                restored_host=socket.gethostname(),
                n_files=len(manifest.files),
                checks=checks,
                n_fills_replayed=replayed,
            ),
            actor=Actor.HUMAN,
        )
        head = ledger.head()
        assert head is not None

    receipt = RestoreReceipt(
        backup_id=manifest.backup_id,
        manifest_sha256=check.manifest_sha256,
        source_host=manifest.host,
        restored_host=socket.gethostname(),
        restored_at=moment,
        code_git_sha=code_git_sha(),
        restored_head_seq=head.seq,
        restored_head_chain_hash=head.chain_hash,
        checks=tuple(checks),
        n_fills_replayed=replayed,
    )
    (target / RECEIPT).write_text(receipt.render(), encoding="utf-8")
    return receipt


def verify_backup_files_at(root: Path, manifest: Manifest) -> list[str]:
    """The manifest's files, checked where they now are."""
    return _check_files(root, manifest)


def _replay_recent(ledger: Ledger, models_root: Path) -> tuple[int, list[str]]:
    """Replay the latest fills that came from a decision, from the restored files alone."""
    rows = ledger.conn.execute(
        "SELECT f.fill_id FROM fills f JOIN order_intents i ON i.intent_id = f.intent_id"
        " WHERE i.decision_id IS NOT NULL ORDER BY f.rowid DESC LIMIT ?",
        (REPLAY_FILLS,),
    ).fetchall()
    if not rows:
        return 0, ["no fill from a decision to replay yet"]
    models = ModelStore(ledger, models_root)
    replayed = 0
    for row in rows:
        fill_id = str(row["fill_id"])
        try:
            outcome = replay_fill(ledger, fill_id, models=models)
        except ReplayError as exc:
            raise BackupError(f"the restored files cannot replay fill {fill_id}: {exc}") from exc
        if not outcome.ok:
            failed = "; ".join(c.detail for c in outcome.checks if not c.ok)
            raise BackupError(f"the restored files do not explain fill {fill_id}: {failed}")
        replayed += 1
    return replayed, [f"replayed the latest {replayed} fill(s) from the restored files"]


# --------------------------------------------------------------------------
# The receipt, home again
# --------------------------------------------------------------------------


def record_receipt(ledger: Ledger, receipt: RestoreReceipt) -> VerifiedRestore:
    """Accept a restore done elsewhere, if this ledger made the backup it restored."""
    created = _created(ledger, receipt.backup_id)
    if created is None:
        raise BackupError(
            f"this ledger never made backup {receipt.backup_id}; a receipt for it says "
            "nothing about whether this ledger's state can be rebuilt"
        )
    if created["manifest_sha256"] != receipt.manifest_sha256:
        raise BackupError(
            f"the receipt restored a manifest ({receipt.manifest_sha256[:12]}…) other than "
            f"the one this ledger recorded for {receipt.backup_id} "
            f"({str(created['manifest_sha256'])[:12]}…)"
        )
    for existing in verified_restores(ledger):
        if existing.backup_id == receipt.backup_id and existing.restored_at == receipt.restored_at:
            return existing
    same_host = receipt.restored_host == created["host"]
    event = ledger.append(
        EventType.BACKUP_RESTORE_VERIFIED,
        receipt.backup_id,
        BackupRestoreVerifiedPayload(
            backup_id=receipt.backup_id,
            manifest_sha256=receipt.manifest_sha256,
            source_host=str(created["host"]),
            restored_host=receipt.restored_host,
            restored_at=to_iso(receipt.restored_at),
            same_host=same_host,
            restored_head_seq=receipt.restored_head_seq,
            restored_head_chain_hash=receipt.restored_head_chain_hash,
            checks=list(receipt.checks),
            n_fills_replayed=receipt.n_fills_replayed,
        ),
        actor=Actor.HUMAN,
    )
    return VerifiedRestore(
        seq=event.seq,
        backup_id=receipt.backup_id,
        restored_host=receipt.restored_host,
        restored_at=receipt.restored_at,
        same_host=same_host,
        n_fills_replayed=receipt.n_fills_replayed,
        checks=receipt.checks,
    )


def _created(ledger: Ledger, backup_id: str) -> dict[str, Any] | None:
    for row in ledger.iter_events(event_type=EventType.BACKUP_CREATED):
        payload: dict[str, Any] = json.loads(row["payload_json"])
        if payload.get("backup_id") == backup_id:
            return payload
    return None


def verified_restores(
    ledger: Ledger, *, within: timedelta | None = None, now: datetime | None = None
) -> list[VerifiedRestore]:
    """Every restore reported back to this ledger, newest last."""
    moment = now or now_utc()
    found: list[VerifiedRestore] = []
    for row in ledger.iter_events(event_type=EventType.BACKUP_RESTORE_VERIFIED):
        payload = json.loads(row["payload_json"])
        restored_at = from_iso(str(payload["restored_at"]))
        if within is not None and moment - restored_at > within:
            continue
        found.append(
            VerifiedRestore(
                seq=int(row["seq"]),
                backup_id=str(payload["backup_id"]),
                restored_host=str(payload["restored_host"]),
                restored_at=restored_at,
                same_host=bool(payload["same_host"]),
                n_fills_replayed=int(payload.get("n_fills_replayed", 0)),
                checks=tuple(str(c) for c in payload.get("checks", [])),
            )
        )
    return found


def backups_made(ledger: Ledger) -> list[dict[str, Any]]:
    """Every backup this ledger recorded, with when it was made."""
    made: list[dict[str, Any]] = []
    for row in ledger.iter_events(event_type=EventType.BACKUP_CREATED):
        payload: dict[str, Any] = json.loads(row["payload_json"])
        made.append({**payload, "at": from_iso(str(row["ts_utc"])), "seq": int(row["seq"])})
    return made
