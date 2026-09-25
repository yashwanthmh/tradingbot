"""Shared fixtures.

Note `tamper` and `resign_chain`: both bypass the append-only triggers by
dropping them first. That is deliberate — it models the realistic adversary,
which is not someone politely issuing an UPDATE the trigger blocks, but someone
(or some buggy agent) with full SQL access to the file. Tests that only exercise
the triggers would prove the lock works while never checking the alarm.
"""

from __future__ import annotations

import os
import shutil
import sqlite3
from collections.abc import Callable, Iterator
from pathlib import Path
from typing import Any

import pytest
import yaml

from tb.config.loader import PinnedLimits, load_hard_limits
from tb.ledger.chain import GENESIS_HASH, compute_chain_hash, compute_payload_hash
from tb.ledger.store import Ledger
from tb.ops.killswitch import write_heartbeat

REPO_ROOT = Path(__file__).resolve().parents[1]
REFERENCE_LIMITS = REPO_ROOT / "config" / "hard_limits.yaml"

_TRIGGERS = (
    "event_log_no_update",
    "event_log_no_delete",
    "event_log_chain_link",
    "event_log_seq_contiguous",
)

# Permission-based tests are meaningless as root: root bypasses mode bits, so a
# chmod 000 file is still readable and the check under test cannot fire.
running_as_root = pytest.mark.skipif(
    os.getuid() == 0,
    reason="root bypasses file permissions, so this check cannot be exercised",
)


@pytest.fixture
def limits_file(tmp_path: Path) -> Path:
    """A copy of the repository's real limits file.

    A copy of the shipped file rather than a minimal fixture, so the tests fail
    if the file we actually deploy stops validating.
    """
    target = tmp_path / "hard_limits.yaml"
    shutil.copy(REFERENCE_LIMITS, target)
    return target


@pytest.fixture
def write_limits(tmp_path: Path) -> Callable[[dict[str, Any]], Path]:
    """Write a limits file with overrides applied to the real one."""

    def _write(overrides: dict[str, Any]) -> Path:
        base = yaml.safe_load(REFERENCE_LIMITS.read_text(encoding="utf-8"))
        for section, values in overrides.items():
            if isinstance(values, dict) and isinstance(base.get(section), dict):
                base[section].update(values)
            else:
                base[section] = values
        target = tmp_path / "overridden_limits.yaml"
        target.write_text(yaml.safe_dump(base, sort_keys=True), encoding="utf-8")
        return target

    return _write


@pytest.fixture
def pinned(limits_file: Path) -> PinnedLimits:
    return load_hard_limits(limits_file)


@pytest.fixture
def env(tmp_path: Path, write_limits: Callable[[dict[str, Any]], Path]) -> dict[str, Any]:
    """A run directory a trading loop can actually start in.

    Here rather than in one test module because both `test_loop` and
    `test_funding` drive the loop, and two copies of "a working run directory"
    is how a suite ends up proving something about a fixture rather than about
    the system.
    """
    run_dir = tmp_path / "run"
    run_dir.mkdir(exist_ok=True)
    limits = write_limits(
        {
            "safety": {
                "kill_switch_path": str(run_dir / "KILL"),
                "heartbeat_path": str(run_dir / "heartbeat"),
            }
        }
    )
    # The watchdog's liveness marker, so the self-check passes. Written here
    # rather than disabled, because `require_watchdog=False` is a different
    # code path and the default one is what production runs.
    write_heartbeat(run_dir / "watchdog", run_id="watchdog", state="supervising")
    return {
        "limits": limits,
        "db": tmp_path / "ledger.db",
        "bars": tmp_path / "bars",
        "run_dir": run_dir,
        "pinned": load_hard_limits(limits),
    }


@pytest.fixture
def cli_env(tmp_path: Path, write_limits: Callable[[dict[str, Any]], Path]) -> dict[str, Any]:
    """A ledger, a bar store and a limits file, addressed the way the CLI takes them.

    Shared by the research-cycle suites, deterministic and model-backed alike,
    so both drive the cycle through the same directory layout.
    """
    run_dir = tmp_path / "run"
    run_dir.mkdir(exist_ok=True)
    limits = write_limits(
        {
            "safety": {
                "kill_switch_path": str(run_dir / "KILL"),
                "heartbeat_path": str(run_dir / "heartbeat"),
            }
        }
    )
    return {
        "limits": limits,
        "db": tmp_path / "ledger.db",
        "bars": tmp_path / "bars",
        "bar_args": [
            "--limits",
            str(limits),
            "--db",
            str(tmp_path / "ledger.db"),
            "--bars",
            str(tmp_path / "bars"),
        ],
    }


@pytest.fixture
def ledger_path(tmp_path: Path) -> Path:
    return tmp_path / "ledger.db"


@pytest.fixture
def ledger(ledger_path: Path) -> Iterator[Ledger]:
    with Ledger(ledger_path) as opened:
        opened.initialise(created_by="test")
        yield opened


@pytest.fixture
def tamper() -> Callable[[Path, str, tuple[Any, ...]], None]:
    """Run arbitrary SQL against a ledger with the guard triggers removed."""

    def _tamper(path: Path, sql: str, params: tuple[Any, ...] = ()) -> None:
        conn = sqlite3.connect(path)
        try:
            for trigger in _TRIGGERS:
                conn.execute(f"DROP TRIGGER IF EXISTS {trigger}")
            conn.execute(sql, params)
            conn.commit()
        finally:
            conn.close()

    return _tamper


@pytest.fixture
def resign_chain() -> Callable[[Path], None]:
    """Recompute every hash so the chain is internally perfect again.

    This is the strongest attack available to anyone with write access: rewrite
    history, then re-sign it so contiguity, payload hashes and links all agree.
    Internal verification cannot catch it — only an anchor published outside the
    database can. Tests use this to prove the anchor is load-bearing rather than
    decorative.
    """

    def _resign(path: Path) -> None:
        conn = sqlite3.connect(path)
        conn.row_factory = sqlite3.Row
        try:
            for trigger in _TRIGGERS:
                conn.execute(f"DROP TRIGGER IF EXISTS {trigger}")
            rows = conn.execute("SELECT * FROM event_log ORDER BY seq ASC").fetchall()
            prev = GENESIS_HASH
            for index, row in enumerate(rows, start=1):
                payload_hash = compute_payload_hash(row["payload_json"])
                chain_hash = compute_chain_hash(
                    prev_hash=prev,
                    seq=index,
                    ts_utc=row["ts_utc"],
                    event_type=row["event_type"],
                    aggregate_type=row["aggregate_type"],
                    aggregate_id=row["aggregate_id"],
                    actor=row["actor"],
                    payload_hash=payload_hash,
                )
                conn.execute(
                    "UPDATE event_log SET seq = ?, payload_hash = ?, prev_hash = ?, "
                    "chain_hash = ? WHERE seq = ?",
                    (index, payload_hash, prev, chain_hash, row["seq"]),
                )
                prev = chain_hash
            conn.commit()
        finally:
            conn.close()

    return _resign
