"""Backup and restore: a copy a clean machine can rebuild from, and the proof.

The ledger here is a real one: a spec reading a recorded model entered through
the loop and its fill settled, and the bar store was compacted into sealed
Parquet. So a backup has every kind of file in it, and a restore has a fill
to replay from the restored files alone.
"""

from __future__ import annotations

import hashlib
import json
from datetime import timedelta
from pathlib import Path
from typing import Any

import pytest
from typer.testing import CliRunner

from tb.cli import app
from tb.core.clock import now_utc
from tb.data.barstore import BarStore
from tb.ledger.store import Ledger
from tb.ledger.verify import verify_chain
from tb.ops.backup import (
    LEDGER_FILE,
    MANIFEST,
    RECEIPT,
    BackupError,
    BackupResult,
    Manifest,
    RestoreReceipt,
    backups_made,
    create_backup,
    record_receipt,
    restore_backup,
    verified_restores,
    verify_backup,
)
from tb.registry.model_store import default_model_root
from tests.test_replay import _model_entry_fill


@pytest.fixture
def traded(env: dict[str, Any]) -> dict[str, Any]:
    """A ledger with a model-decided fill, and its bars sealed into Parquet."""
    fill_id, _ = _model_entry_fill(env)
    with Ledger(env["db"], config_hash=env["pinned"].config_hash) as ledger:
        store = BarStore(ledger, root=env["bars"], scale=env["pinned"].limits.data.price_scale)
        store.compact()
        n = ledger.conn.execute("SELECT COUNT(*) FROM data_partitions").fetchone()[0]
    assert n > 0, "the fixture needs sealed partitions to back up"
    return {**env, "fill_id": fill_id, "models": default_model_root(env["db"])}


def _backup(env: dict[str, Any], destination: Path) -> BackupResult:
    with Ledger(env["db"], config_hash=env["pinned"].config_hash) as ledger:
        return create_backup(
            ledger,
            destination=destination,
            bars_root=env["bars"],
            models_root=env["models"],
            limits=env["pinned"],
        )


def _file(result: BackupResult, kind: str) -> Path:
    entry = next(f for f in result.manifest.files if f.kind == kind)
    return Path(result.path) / entry.path


# --------------------------------------------------------------------------
# Making one
# --------------------------------------------------------------------------


def test_a_backup_holds_the_ledger_and_everything_it_names(
    traded: dict[str, Any], tmp_path: Path
) -> None:
    with Ledger(traded["db"]) as ledger:
        head_before = ledger.head()
    result = _backup(traded, tmp_path / "backups")

    kinds = [f.kind for f in result.manifest.files]
    assert kinds.count("ledger") == 1 and kinds.count("model") == 1
    assert kinds.count("limits") == 1 and kinds.count("partition") >= 1
    assert result.manifest.head_seq == head_before.seq  # type: ignore[union-attr]
    manifest_bytes = (result.path / MANIFEST).read_bytes()
    assert hashlib.sha256(manifest_bytes).hexdigest() == result.manifest_sha256
    assert verify_backup(result.path).ok

    with Ledger(traded["db"]) as ledger:
        [made] = backups_made(ledger)
    assert made["backup_id"] == result.backup_id
    assert made["manifest_sha256"] == result.manifest_sha256
    # One self-contained file, not a database and its write-ahead log.
    assert not (result.path / f"{LEDGER_FILE}-wal").exists()


def test_a_backup_refuses_a_file_that_is_not_what_the_ledger_names(
    traded: dict[str, Any], tmp_path: Path
) -> None:
    """Better to fail the backup now than hand the restore the damage later."""
    partition = next(Path(traded["bars"]).rglob("*.parquet"))
    partition.write_bytes(partition.read_bytes() + b"\0")
    destination = tmp_path / "backups"

    with pytest.raises(BackupError, match="does not match the hash"):
        _backup(traded, destination)
    assert not any(destination.iterdir()), "half a backup was left behind"


def test_a_backup_refuses_a_file_the_ledger_names_and_does_not_have(
    traded: dict[str, Any], tmp_path: Path
) -> None:
    next(Path(traded["models"]).glob("*.txt")).unlink()
    with pytest.raises(BackupError, match="it is not there"):
        _backup(traded, tmp_path / "backups")


# --------------------------------------------------------------------------
# Checking one
# --------------------------------------------------------------------------


def test_verify_names_what_is_wrong_with_a_backup(traded: dict[str, Any], tmp_path: Path) -> None:
    result = _backup(traded, tmp_path / "backups")

    partition = _file(result, "partition")
    original = partition.read_bytes()
    partition.write_bytes(original + b"\0")
    assert "changed after the backup" in verify_backup(result.path).problems[0]
    partition.write_bytes(original)

    model = _file(result, "model")
    kept = model.read_bytes()
    model.unlink()
    assert "is missing" in verify_backup(result.path).problems[0]
    model.write_bytes(kept)
    assert verify_backup(result.path).ok

    body = json.loads((result.path / MANIFEST).read_text(encoding="utf-8"))
    body["files"] = [f for f in body["files"] if f["kind"] != "model"]
    (result.path / MANIFEST).write_text(json.dumps(body), encoding="utf-8")
    check = verify_backup(result.path)
    assert any("which is not in the backup" in p for p in check.problems)

    body["files"].append({"path": "../escape", "sha256": "0" * 64, "bytes": 0, "kind": "model"})
    (result.path / MANIFEST).write_text(json.dumps(body), encoding="utf-8")
    assert any("outside the backup" in p for p in verify_backup(result.path).problems)

    assert "no readable manifest" in verify_backup(tmp_path / "nowhere").problems[0]


def test_a_backup_with_a_rewritten_chain_does_not_verify(
    traded: dict[str, Any], tmp_path: Path, tamper: Any
) -> None:
    result = _backup(traded, tmp_path / "backups")
    copy = result.path / LEDGER_FILE
    tamper(copy, "UPDATE event_log SET payload_json = payload_json || ' ' WHERE seq = 2")
    # Re-hash the file so the manifest still matches: the chain must catch it.
    body = json.loads((result.path / MANIFEST).read_text(encoding="utf-8"))
    for entry in body["files"]:
        if entry["kind"] == "ledger":
            entry["sha256"] = hashlib.sha256(copy.read_bytes()).hexdigest()
            entry["bytes"] = copy.stat().st_size
    (result.path / MANIFEST).write_text(json.dumps(body), encoding="utf-8")

    check = verify_backup(result.path)
    assert not check.ok
    assert "chain does not verify" in check.problems[0]


# --------------------------------------------------------------------------
# Restoring one
# --------------------------------------------------------------------------


def test_a_restore_rebuilds_the_ledger_and_replays_its_fills(
    traded: dict[str, Any], tmp_path: Path
) -> None:
    result = _backup(traded, tmp_path / "backups")
    target = tmp_path / "clean-machine"

    receipt = restore_backup(result.path, target)

    assert receipt.backup_id == result.backup_id
    assert receipt.manifest_sha256 == result.manifest_sha256
    assert receipt.n_fills_replayed == 1
    assert any("replayed the latest 1 fill(s)" in c for c in receipt.checks)
    assert RestoreReceipt.parse((target / RECEIPT).read_text(encoding="utf-8")) == receipt

    with Ledger(target / LEDGER_FILE) as restored:
        assert verify_chain(restored).ok
        head = restored.head()
        assert head is not None
        assert (head.seq, head.chain_hash) == (
            receipt.restored_head_seq,
            receipt.restored_head_chain_hash,
        )
        [event] = restored.conn.execute(
            "SELECT payload_json FROM event_log WHERE event_type = 'backup.restored'"
        ).fetchall()
    assert json.loads(event["payload_json"])["n_fills_replayed"] == 1
    for kind in ("partition", "model"):
        entry = next(f for f in result.manifest.files if f.kind == kind)
        assert (target / entry.path).is_file()


def test_a_restore_refuses_a_directory_that_is_not_empty(
    traded: dict[str, Any], tmp_path: Path
) -> None:
    result = _backup(traded, tmp_path / "backups")
    target = tmp_path / "occupied"
    target.mkdir()
    (target / LEDGER_FILE).write_text("someone else's history", encoding="utf-8")

    with pytest.raises(BackupError, match="not empty"):
        restore_backup(result.path, target)


def test_a_restore_refuses_a_backup_that_does_not_verify(
    traded: dict[str, Any], tmp_path: Path
) -> None:
    result = _backup(traded, tmp_path / "backups")
    partition = _file(result, "partition")
    partition.write_bytes(partition.read_bytes() + b"\0")
    target = tmp_path / "clean-machine"

    with pytest.raises(BackupError, match="does not verify"):
        restore_backup(result.path, target)
    assert not target.exists()


# --------------------------------------------------------------------------
# The receipt, home again
# --------------------------------------------------------------------------


def _elsewhere(receipt: RestoreReceipt) -> RestoreReceipt:
    """The same receipt as a different machine would have written it."""
    body = json.loads(receipt.render())
    body["restored_host"] = "a-clean-machine"
    return RestoreReceipt.parse(json.dumps(body))


def test_a_receipt_is_accepted_only_for_a_backup_this_ledger_made(
    traded: dict[str, Any], tmp_path: Path
) -> None:
    result = _backup(traded, tmp_path / "backups")
    receipt = _elsewhere(restore_backup(result.path, tmp_path / "clean-machine"))

    with Ledger(traded["db"], config_hash=traded["pinned"].config_hash) as ledger:
        recorded = record_receipt(ledger, receipt)
        assert recorded.backup_id == result.backup_id
        assert recorded.same_host is False
        assert recorded.n_fills_replayed == 1
        # The same receipt twice is one restore, not two.
        assert record_receipt(ledger, receipt).seq == recorded.seq
        assert len(verified_restores(ledger)) == 1

        body = json.loads(receipt.render())
        body["backup_id"] = "bkp_never_made"
        with pytest.raises(BackupError, match="never made backup"):
            record_receipt(ledger, RestoreReceipt.parse(json.dumps(body)))

        body = json.loads(receipt.render())
        body["manifest_sha256"] = "0" * 64
        with pytest.raises(BackupError, match="other than the one"):
            record_receipt(ledger, RestoreReceipt.parse(json.dumps(body)))


def test_a_restore_on_the_same_machine_is_recorded_as_such(
    traded: dict[str, Any], tmp_path: Path
) -> None:
    """It proves the files are intact, not that the state survives losing the machine."""
    result = _backup(traded, tmp_path / "backups")
    receipt = restore_backup(result.path, tmp_path / "same-machine")

    with Ledger(traded["db"], config_hash=traded["pinned"].config_hash) as ledger:
        assert record_receipt(ledger, receipt).same_host is True


def test_recent_restores_are_read_by_age(traded: dict[str, Any], tmp_path: Path) -> None:
    result = _backup(traded, tmp_path / "backups")
    receipt = _elsewhere(restore_backup(result.path, tmp_path / "clean-machine"))
    with Ledger(traded["db"], config_hash=traded["pinned"].config_hash) as ledger:
        record_receipt(ledger, receipt)
        later = now_utc() + timedelta(days=100)
        assert len(verified_restores(ledger, within=timedelta(days=90))) == 1
        assert verified_restores(ledger, within=timedelta(days=90), now=later) == []


def test_malformed_manifests_and_receipts_are_refused() -> None:
    with pytest.raises(BackupError, match="malformed"):
        Manifest.parse(json.dumps({"tb_backup": 1}))
    with pytest.raises(BackupError, match="format"):
        Manifest.parse(json.dumps({"tb_backup": 99}))
    with pytest.raises(BackupError, match="not a restore receipt"):
        RestoreReceipt.parse("{}")


# --------------------------------------------------------------------------
# tb backup
# --------------------------------------------------------------------------


def test_tb_backup_end_to_end(traded: dict[str, Any], tmp_path: Path) -> None:
    runner = CliRunner()
    common = ["--limits", str(traded["limits"]), "--db", str(traded["db"])]
    destination = tmp_path / "backups"

    made = runner.invoke(app, ["backup", "create", "--to", str(destination), *common])
    assert made.exit_code == 0, made.output
    [path] = list(destination.iterdir())
    assert path.name in made.output

    verified = runner.invoke(app, ["backup", "verify", str(path)])
    assert verified.exit_code == 0, verified.output
    assert "is whole" in verified.output

    target = tmp_path / "clean-machine"
    restored = runner.invoke(app, ["backup", "restore", str(path), "--to", str(target)])
    assert restored.exit_code == 0, restored.output
    assert "replayed the latest 1 fill(s)" in restored.output
    again = runner.invoke(app, ["backup", "restore", str(path), "--to", str(target)])
    assert again.exit_code == 1
    assert "not empty" in again.output

    recorded = runner.invoke(app, ["backup", "receipt", str(target / RECEIPT), *common])
    assert recorded.exit_code == 0, recorded.output
    assert "restored on the machine that made it" in recorded.output
    # Two machines that share a host name are one to the gate; the operator
    # who did restore elsewhere is told why it did not count.
    assert "told apart by host name" in recorded.output

    listed = runner.invoke(app, ["backup", "list", "--db", str(traded["db"])])
    assert listed.exit_code == 0, listed.output
    assert path.name in listed.output
    assert "same host, so the live gate does not count it" in listed.output

    partition = next((path / "bars").rglob("*.parquet"))
    partition.write_bytes(partition.read_bytes() + b"\0")
    broken = runner.invoke(app, ["backup", "verify", str(path)])
    assert broken.exit_code == 1
    assert "changed after the backup" in broken.output

    unreadable = runner.invoke(app, ["backup", "receipt", str(tmp_path / "none.json"), *common])
    assert unreadable.exit_code == 2
