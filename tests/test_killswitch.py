"""The kill switch and heartbeat.

Everything here tests one property from different angles: **the ambiguous
reading resolves to "stop"**. A switch that cannot be read, a directory that is
not there, a heartbeat that never appeared — each of those is a state where the
honest answer is "I do not know", and the only safe interpretation of "I do not
know" in a system that moves money is "do not trade".
"""

from __future__ import annotations

import json
import os
import time
from pathlib import Path

import pytest

from tb.ops.killswitch import (
    KillSwitchState,
    engage_kill_switch,
    read_heartbeat,
    read_kill_switch,
    release_kill_switch,
    write_heartbeat,
)
from tests.conftest import running_as_root


class TestReading:
    def test_absent_file_in_an_existing_directory_is_clear(self, tmp_path: Path) -> None:
        reading = read_kill_switch(tmp_path / "KILL")
        assert reading.state is KillSwitchState.CLEAR
        assert reading.may_trade

    def test_present_file_is_engaged(self, tmp_path: Path) -> None:
        path = tmp_path / "KILL"
        path.write_text("{}", encoding="utf-8")
        reading = read_kill_switch(path)
        assert reading.state is KillSwitchState.ENGAGED
        assert not reading.may_trade

    def test_a_missing_parent_directory_is_undeterminable_not_clear(self, tmp_path: Path) -> None:
        """The subtle fail-open this avoids.

        If `var/run/` is not mounted, nobody can engage the switch by touching
        the file — the touch itself would fail. Reading that absence as "switch
        not engaged" means the one operation that must always work silently
        cannot, and the bot trades on through it.
        """
        reading = read_kill_switch(tmp_path / "not-mounted" / "KILL")
        assert reading.state is KillSwitchState.UNDETERMINABLE
        assert not reading.may_trade
        assert "cannot be engaged by anyone" in reading.detail

    def test_a_parent_that_is_a_file_is_undeterminable(self, tmp_path: Path) -> None:
        not_a_dir = tmp_path / "run"
        not_a_dir.write_text("i am a file", encoding="utf-8")
        reading = read_kill_switch(not_a_dir / "KILL")
        assert reading.state is KillSwitchState.UNDETERMINABLE

    @running_as_root
    def test_permission_denied_is_undeterminable_not_clear(self, tmp_path: Path) -> None:
        """`Path.exists()` would have returned False here.

        It swallows OSError, so a permission error on the switch file — exactly
        what a misconfigured mount produces — reads as "no kill file, carry on".
        This module uses `os.stat` and resolves the error explicitly instead.
        """
        directory = tmp_path / "run"
        directory.mkdir()
        path = directory / "KILL"
        path.write_text("{}", encoding="utf-8")
        directory.chmod(0o000)
        try:
            reading = read_kill_switch(path)
            assert reading.state is KillSwitchState.UNDETERMINABLE
            assert not reading.may_trade
        finally:
            directory.chmod(0o755)

    def test_an_unparseable_kill_file_still_kills(self, tmp_path: Path) -> None:
        """Requiring valid JSON to honour a stop would be an absurd fail-open."""
        path = tmp_path / "KILL"
        path.write_text("this is not json {{{", encoding="utf-8")
        reading = read_kill_switch(path)
        assert reading.state is KillSwitchState.ENGAGED
        assert reading.engaged_by is None

    def test_an_empty_kill_file_still_kills(self, tmp_path: Path) -> None:
        path = tmp_path / "KILL"
        path.touch()
        assert read_kill_switch(path).state is KillSwitchState.ENGAGED

    def test_a_json_list_kill_file_still_kills(self, tmp_path: Path) -> None:
        path = tmp_path / "KILL"
        path.write_text("[1, 2, 3]", encoding="utf-8")
        assert read_kill_switch(path).state is KillSwitchState.ENGAGED

    def test_only_clear_permits_trading(self) -> None:
        # Stated as an explicit invariant so a future refactor that adds a
        # fourth state has to decide about it rather than default to permissive.
        permitting = [s for s in KillSwitchState if s is KillSwitchState.CLEAR]
        assert permitting == [KillSwitchState.CLEAR]


class TestEngaging:
    def test_engage_then_read_reports_who_and_why(self, tmp_path: Path) -> None:
        path = tmp_path / "run" / "KILL"
        engage_kill_switch(path, engaged_by="alex", reason="spread looks wrong")
        reading = read_kill_switch(path)
        assert reading.state is KillSwitchState.ENGAGED
        assert reading.engaged_by == "alex"
        assert reading.reason == "spread looks wrong"
        assert reading.engaged_at is not None

    def test_engage_creates_the_directory(self, tmp_path: Path) -> None:
        path = tmp_path / "deep" / "nested" / "KILL"
        engage_kill_switch(path, engaged_by="ops", reason="drill")
        assert path.exists()

    def test_engage_is_idempotent_and_preserves_the_original_reason(self, tmp_path: Path) -> None:
        """Re-engaging must not overwrite why it was first stopped."""
        path = tmp_path / "KILL"
        engage_kill_switch(path, engaged_by="first", reason="the real reason")
        engage_kill_switch(path, engaged_by="second", reason="a later guess")
        reading = read_kill_switch(path)
        assert reading.engaged_by == "first"
        assert reading.reason == "the real reason"

    def test_no_partial_file_is_left_behind(self, tmp_path: Path) -> None:
        """Written to a temp name then renamed.

        A half-written kill file must never be a readable one, and a `.tmp`
        left lying around would be a confusing artefact in an incident.
        """
        path = tmp_path / "KILL"
        engage_kill_switch(path, engaged_by="ops", reason="drill")
        assert not list(tmp_path.glob("*.tmp"))
        assert json.loads(path.read_text(encoding="utf-8"))["engaged_by"] == "ops"

    def test_release_makes_it_clear(self, tmp_path: Path) -> None:
        path = tmp_path / "KILL"
        engage_kill_switch(path, engaged_by="ops", reason="drill")
        reading = release_kill_switch(path)
        assert reading.state is KillSwitchState.CLEAR

    def test_release_is_idempotent(self, tmp_path: Path) -> None:
        path = tmp_path / "KILL"
        assert release_kill_switch(path).state is KillSwitchState.CLEAR
        assert release_kill_switch(path).state is KillSwitchState.CLEAR


class TestHeartbeat:
    def test_a_missing_heartbeat_is_stale_not_pending(self, tmp_path: Path) -> None:
        """The watchdog cannot tell "not started" from "died at startup".

        The safe reading of that ambiguity is stale.
        """
        reading = read_heartbeat(tmp_path / "heartbeat", stale_after_seconds=60)
        assert reading.stale
        assert not reading.exists
        assert reading.age_seconds is None

    def test_a_fresh_heartbeat_is_not_stale(self, tmp_path: Path) -> None:
        path = tmp_path / "heartbeat"
        write_heartbeat(path, run_id="run_1", state="trading")
        reading = read_heartbeat(path, stale_after_seconds=60)
        assert not reading.stale
        assert reading.exists
        assert reading.age_seconds is not None
        assert reading.age_seconds < 5

    def test_an_old_heartbeat_is_stale(self, tmp_path: Path) -> None:
        path = tmp_path / "heartbeat"
        write_heartbeat(path, run_id="run_1", state="trading")
        old = time.time() - 600
        os.utime(path, (old, old))
        reading = read_heartbeat(path, stale_after_seconds=120)
        assert reading.stale
        assert reading.age_seconds is not None
        assert reading.age_seconds > 120
        assert "STALE" in reading.detail

    def test_heartbeat_records_the_run_and_state(self, tmp_path: Path) -> None:
        path = tmp_path / "heartbeat"
        write_heartbeat(path, run_id="run_7", state="reconciling")
        body = json.loads(path.read_text(encoding="utf-8"))
        assert body["run_id"] == "run_7"
        assert body["state"] == "reconciling"
        assert body["pid"] == os.getpid()

    def test_write_failure_propagates(self, tmp_path: Path) -> None:
        """The trader must halt itself if it cannot prove it is alive.

        Swallowing this error would leave only the watchdog noticing a stall,
        and a watchdog can die too.
        """
        blocker = tmp_path / "blocked"
        blocker.write_text("i am a file, not a directory", encoding="utf-8")
        with pytest.raises(OSError):
            write_heartbeat(blocker / "heartbeat", run_id="run_1", state="trading")
