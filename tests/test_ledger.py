"""The audit ledger.

Grouped by the property being defended:

* `TestAppendOnlyStorage` — the storage layer refuses to let history change.
* `TestChainVerification` — when someone bypasses that layer, the alarm names
  the exact row.
* `TestAnchoring` — the two attacks that produce an internally *valid* chain,
  and which only an external anchor can catch.
* `TestEventVocabulary` — events cannot be recorded off-schema.
* `TestAtomicity` — an event and its projection land together or not at all.
"""

from __future__ import annotations

import itertools
import sqlite3
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest

from tb.core.errors import LedgerError
from tb.ledger.anchor import FileAnchorSink, anchor_head
from tb.ledger.chain import GENESIS_HASH, compute_chain_hash
from tb.ledger.events import (
    EVENT_AGGREGATES,
    EVENT_PAYLOADS,
    EventType,
    GenesisPayload,
    RunStartedPayload,
)
from tb.ledger.schema import LEDGER_SCHEMA_VERSION
from tb.ledger.store import Ledger
from tb.ledger.verify import FindingKind, verify_chain


def _add_events(ledger: Ledger, count: int = 4) -> None:
    for index in range(count):
        ledger.record_run_start(run_id=f"run_{index}", mode="paper")


class TestAppendOnlyStorage:
    """The storage layer must refuse to let recorded history change."""

    def test_genesis_chains_to_a_known_constant(self, ledger: Ledger) -> None:
        # Chaining to GENESIS_HASH rather than NULL makes "this is the start of
        # the chain" a checkable assertion rather than an absence.
        first = ledger.get(1)
        assert first is not None
        assert first["prev_hash"] == GENESIS_HASH
        assert first["event_type"] == EventType.LEDGER_GENESIS.value

    def test_initialise_is_idempotent(self, ledger: Ledger) -> None:
        assert ledger.initialise(created_by="again") is None
        assert ledger.count() == 1

    def test_each_event_links_to_its_predecessor(self, ledger: Ledger) -> None:
        _add_events(ledger, 3)
        rows = list(ledger.iter_events())
        for previous, current in itertools.pairwise(rows):
            assert current["prev_hash"] == previous["chain_hash"]
            assert current["seq"] == previous["seq"] + 1

    def test_update_is_blocked(self, ledger: Ledger) -> None:
        with pytest.raises(sqlite3.IntegrityError, match="append-only"):
            ledger.conn.execute("UPDATE event_log SET actor = 'human' WHERE seq = 1")

    def test_delete_is_blocked(self, ledger: Ledger) -> None:
        with pytest.raises(sqlite3.IntegrityError, match="append-only"):
            ledger.conn.execute("DELETE FROM event_log WHERE seq = 1")

    def test_a_broken_link_is_refused_at_insert(self, ledger: Ledger) -> None:
        """The database will not store a broken chain.

        Defence in depth: a writer bug fails loudly where the mistake happens,
        rather than producing a log that only fails verification weeks later.
        """
        with pytest.raises(sqlite3.IntegrityError, match="chain break"):
            ledger.conn.execute(
                "INSERT INTO event_log (seq, ts_utc, event_type, aggregate_type, "
                "aggregate_id, actor, payload_json, payload_hash, prev_hash, "
                "chain_hash, schema_version) VALUES "
                "(2, '2026-01-01T00:00:00.000000+0000', 'x', 'y', 'z', 'system', "
                "'{}', 'h', 'not-the-head', 'c', 1)"
            )

    def test_a_sequence_gap_is_refused_at_insert(self, ledger: Ledger) -> None:
        """Gaps are impossible, so a deleted row cannot be papered over."""
        head = ledger.head()
        assert head is not None
        with pytest.raises(sqlite3.IntegrityError, match="exactly one past the head"):
            ledger.conn.execute(
                "INSERT INTO event_log (seq, ts_utc, event_type, aggregate_type, "
                "aggregate_id, actor, payload_json, payload_hash, prev_hash, "
                "chain_hash, schema_version) VALUES "
                "(99, '2026-01-01T00:00:00.000000+0000', 'x', 'y', 'z', 'system', "
                "'{}', 'h', ?, 'c2', 1)",
                (head.chain_hash,),
            )

    def test_duplicate_chain_hash_is_refused(self, ledger: Ledger) -> None:
        row = ledger.get(1)
        assert row is not None
        with pytest.raises(sqlite3.IntegrityError):
            ledger.conn.execute(
                "INSERT INTO event_log (seq, ts_utc, event_type, aggregate_type, "
                "aggregate_id, actor, payload_json, payload_hash, prev_hash, "
                "chain_hash, schema_version) VALUES "
                "(2, '2026-01-01T00:00:00.000000+0000', 'x', 'y', 'z', 'system', "
                "'{}', 'h', ?, ?, 1)",
                (row["chain_hash"], row["chain_hash"]),
            )


class TestChainVerification:
    """When the storage guards are bypassed, the alarm must name the row."""

    def test_a_clean_chain_verifies(self, ledger: Ledger) -> None:
        _add_events(ledger, 5)
        report = verify_chain(ledger)
        assert report.ok, report.summary()
        assert report.events_checked == ledger.count()
        assert report.first_failure is None

    def test_an_edited_payload_is_caught_at_the_exact_seq(
        self, ledger: Ledger, ledger_path: Path, tamper: Callable[..., None]
    ) -> None:
        _add_events(ledger, 5)
        ledger.close()

        tamper(
            ledger_path,
            "UPDATE event_log SET payload_json = replace(payload_json, 'paper', 'live') "
            "WHERE seq = 3",
        )

        with Ledger(ledger_path) as reopened:
            report = verify_chain(reopened)
        assert not report.ok
        failure = report.first_failure
        assert failure is not None
        assert failure.seq == 3
        assert failure.kind is FindingKind.PAYLOAD_TAMPERED

    @pytest.mark.parametrize(
        "column,value",
        [
            ("actor", "'human'"),
            ("event_type", "'run.ended'"),
            ("ts_utc", "'2020-01-01T00:00:00.000000+0000'"),
            ("aggregate_id", "'someone_else'"),
        ],
    )
    def test_an_edited_link_field_is_caught(
        self,
        ledger: Ledger,
        ledger_path: Path,
        tamper: Callable[..., None],
        column: str,
        value: str,
    ) -> None:
        """Every field covered by the chain hash must be protected.

        Backdating `ts_utc` is the interesting one: it is how you would make a
        trade look like it happened before the news that justified it.
        """
        _add_events(ledger, 4)
        ledger.close()

        tamper(ledger_path, f"UPDATE event_log SET {column} = {value} WHERE seq = 3")

        with Ledger(ledger_path) as reopened:
            report = verify_chain(reopened)
        failure = report.first_failure
        assert failure is not None
        assert failure.seq == 3
        assert failure.kind is FindingKind.CHAIN_HASH_MISMATCH

    def test_a_deleted_row_is_caught_as_a_gap(
        self, ledger: Ledger, ledger_path: Path, tamper: Callable[..., None]
    ) -> None:
        _add_events(ledger, 5)
        ledger.close()

        tamper(ledger_path, "DELETE FROM event_log WHERE seq = 3")

        with Ledger(ledger_path) as reopened:
            report = verify_chain(reopened)
        failure = report.first_failure
        assert failure is not None
        assert failure.seq == 3
        assert failure.kind is FindingKind.SEQUENCE_GAP
        # And the following row's link no longer resolves.
        assert any(f.kind is FindingKind.BROKEN_LINK for f in report.findings)

    def test_reordering_two_events_is_caught(
        self, ledger: Ledger, ledger_path: Path, tamper: Callable[..., None]
    ) -> None:
        _add_events(ledger, 5)
        ledger.close()

        # Swap seq 3 and 4 without touching anything else.
        tamper(ledger_path, "UPDATE event_log SET seq = 999 WHERE seq = 3")
        tamper(ledger_path, "UPDATE event_log SET seq = 3 WHERE seq = 4")
        tamper(ledger_path, "UPDATE event_log SET seq = 4 WHERE seq = 999")

        with Ledger(ledger_path) as reopened:
            report = verify_chain(reopened)
        assert not report.ok

    def test_quick_mode_stops_at_the_first_finding(
        self, ledger: Ledger, ledger_path: Path, tamper: Callable[..., None]
    ) -> None:
        _add_events(ledger, 6)
        ledger.close()
        tamper(ledger_path, "UPDATE event_log SET actor = 'human' WHERE seq = 2")

        with Ledger(ledger_path) as reopened:
            quick = verify_chain(reopened, stop_on_first=True, check_anchors=False)
            full = verify_chain(reopened, check_anchors=False)
        assert len(quick.findings) == 1
        assert len(full.findings) >= len(quick.findings)


class TestAnchoring:
    """The two attacks that produce an internally valid chain."""

    def test_anchor_records_the_head_and_agrees_with_it(
        self, ledger: Ledger, tmp_path: Path
    ) -> None:
        _add_events(ledger, 3)
        result = anchor_head(ledger, FileAnchorSink(tmp_path / "heads.jsonl"))
        head_before_event = result.seq

        report = verify_chain(ledger)
        assert report.ok
        assert report.anchors_checked == 1
        # The anchoring event itself extends the chain, so the anchored head is
        # one behind the new tip. That is expected, not a bug.
        assert head_before_event == ledger.count() - 1

    def test_anchoring_an_empty_chain_is_refused(self, ledger_path: Path, tmp_path: Path) -> None:
        with Ledger(ledger_path) as empty, pytest.raises(LedgerError, match="chain is empty"):
            anchor_head(empty, FileAnchorSink(tmp_path / "heads.jsonl"))

    def test_truncation_past_an_anchor_is_caught_only_by_the_anchor(
        self, ledger: Ledger, ledger_path: Path, tmp_path: Path, tamper: Callable[..., None]
    ) -> None:
        """Truncation leaves a perfect chain behind it.

        Drop the tail of the log and rows 1..n still hash correctly, still link
        correctly, still have no gaps. Nothing inside the database can tell.
        The published head can.
        """
        _add_events(ledger, 5)
        anchor_head(ledger, FileAnchorSink(tmp_path / "heads.jsonl"))
        anchored_seq = ledger.count() - 1
        ledger.close()

        tamper(ledger_path, f"DELETE FROM event_log WHERE seq > {anchored_seq - 1}")

        with Ledger(ledger_path) as reopened:
            without_anchors = verify_chain(reopened, check_anchors=False)
            with_anchors = verify_chain(reopened, check_anchors=True)

        assert without_anchors.ok, "internal checks cannot see a clean truncation"
        assert not with_anchors.ok
        assert any(f.kind is FindingKind.ANCHOR_BEYOND_CHAIN for f in with_anchors.findings), (
            with_anchors.summary()
        )

    def test_a_full_rewrite_and_resign_is_caught_only_by_the_anchor(
        self,
        ledger: Ledger,
        ledger_path: Path,
        tmp_path: Path,
        tamper: Callable[..., None],
        resign_chain: Callable[[Path], None],
    ) -> None:
        """The strongest attack, and the reason anchoring exists at all.

        An adversary with write access — a compromised host, or an agent editing
        its own history — rewrites a payload and then recomputes every hash
        downstream. Contiguity, payload hashes and links all agree afterwards.
        Internal verification passes on a log that is entirely fictional.

        What they cannot change is a head hash already published somewhere they
        do not control.
        """
        _add_events(ledger, 5)
        anchor_head(ledger, FileAnchorSink(tmp_path / "heads.jsonl"))
        ledger.close()

        tamper(
            ledger_path,
            "UPDATE event_log SET payload_json = replace(payload_json, 'paper', 'live') "
            "WHERE seq = 3",
        )
        resign_chain(ledger_path)

        with Ledger(ledger_path) as reopened:
            without_anchors = verify_chain(reopened, check_anchors=False)
            with_anchors = verify_chain(reopened, check_anchors=True)

        assert without_anchors.ok, (
            "a re-signed chain is internally perfect — this is exactly why a "
            "hash chain alone is not evidence against its own writer"
        )
        assert not with_anchors.ok
        assert any(f.kind is FindingKind.ANCHOR_MISMATCH for f in with_anchors.findings)
        failure = with_anchors.first_failure
        assert failure is not None
        assert "re-signed" in failure.detail

    def test_file_sink_reports_its_trust_boundary_honestly(self, tmp_path: Path) -> None:
        """A file beside the database is not a trust boundary, and says so."""
        sink = FileAnchorSink(tmp_path / "heads.jsonl")
        assert not sink.crosses_trust_boundary
        assert "only evidence once" in sink.trust_boundary


class TestEventVocabulary:
    """Events cannot be recorded off-schema."""

    def test_every_event_type_has_a_payload_model(self) -> None:
        missing = [e for e in EventType if e not in EVENT_PAYLOADS]
        assert not missing, f"event types with no payload model: {missing}"

    def test_every_event_type_has_an_aggregate(self) -> None:
        missing = [e for e in EventType if e not in EVENT_AGGREGATES]
        assert not missing, f"event types with no aggregate type: {missing}"

    def test_a_mismatched_payload_model_is_refused(self, ledger: Ledger) -> None:
        with pytest.raises(LedgerError, match="expects payload"):
            ledger.append(
                EventType.RUN_STARTED,
                "run_x",
                GenesisPayload(ledger_schema_version=1, created_by="x", code_version="0"),
            )

    def test_an_unknown_payload_field_is_refused(self, ledger: Ledger) -> None:
        with pytest.raises(LedgerError, match="invalid payload"):
            ledger.append(
                EventType.RUN_ENDED,
                "run_x",
                {"run_id": "run_x", "exit_reason": "ok", "surprise": 1},
            )

    def test_a_missing_required_field_is_refused(self, ledger: Ledger) -> None:
        with pytest.raises(LedgerError, match="invalid payload"):
            ledger.append(EventType.RUN_ENDED, "run_x", {"run_id": "run_x"})

    def test_a_dict_payload_is_validated_and_normalised(self, ledger: Ledger) -> None:
        event = ledger.append(
            EventType.RUN_ENDED, "run_x", {"run_id": "run_x", "exit_reason": "ok"}
        )
        # Canonical form: sorted keys, no incidental whitespace, defaults filled.
        assert event.payload_json == (
            '{"error_detail":null,"error_type":null,"exit_reason":"ok","run_id":"run_x"}'
        )


class TestAtomicity:
    """An event and the projection it implies must land together."""

    def test_a_failed_projection_rolls_back_the_event(self, ledger: Ledger) -> None:
        """A `halt.raised` event with no row in `halts` makes the log lie.

        So would a row in `halts` with no event. Both go in one transaction.
        """
        before = ledger.count()
        with pytest.raises(sqlite3.OperationalError), ledger.transaction() as tx:
            tx.append(
                EventType.RUN_STARTED,
                "run_bad",
                RunStartedPayload(
                    run_id="run_bad",
                    mode="paper",
                    code_version="0",
                    host="h",
                    pid=1,
                    python_version="3.11",
                ),
            )
            tx.execute("INSERT INTO table_that_does_not_exist VALUES (1)")

        assert ledger.count() == before

    def test_run_start_writes_both_the_event_and_the_projection(self, ledger: Ledger) -> None:
        ledger.record_run_start(run_id="run_1", mode="paper")
        row = ledger.conn.execute("SELECT * FROM runs WHERE run_id = 'run_1'").fetchone()
        assert row is not None
        assert row["mode"] == "paper"
        events = list(ledger.iter_events(event_type=EventType.RUN_STARTED))
        assert len(events) == 1

    def test_config_pin_writes_the_full_values_to_its_projection(
        self, ledger: Ledger, pinned: Any
    ) -> None:
        ledger.record_config_pin(pinned.audit_record())
        row = ledger.conn.execute("SELECT * FROM config_versions").fetchone()
        assert row is not None
        assert row["config_hash"] == pinned.config_hash
        assert "absolute_ceiling_ccy" in row["values_json"]


class TestChainHashing:
    def test_chain_hash_requires_keyword_arguments(self) -> None:
        """Positional args would let a call site swap two fields silently.

        A swap of `aggregate_type` and `aggregate_id` produces a chain that
        verifies perfectly and describes the wrong thing.
        """
        with pytest.raises(TypeError):
            compute_chain_hash("a", 1, "b", "c", "d", "e", "f", "g")  # type: ignore[call-arg]

    def test_swapping_two_fields_changes_the_hash(self) -> None:
        base: dict[str, Any] = {
            "prev_hash": GENESIS_HASH,
            "seq": 1,
            "ts_utc": "2026-01-01T00:00:00.000000+0000",
            "event_type": "run.started",
            "aggregate_type": "run",
            "aggregate_id": "run_1",
            "actor": "system",
            "payload_hash": "abc",
        }
        swapped = {**base, "aggregate_type": "run_1", "aggregate_id": "run"}
        assert compute_chain_hash(**base) != compute_chain_hash(**swapped)

    def test_dropping_the_guard_triggers_is_not_a_durable_bypass(
        self, ledger: Ledger, ledger_path: Path, tamper: Callable[..., None]
    ) -> None:
        """Reopening the ledger restores the append-only guards.

        `apply_schema` runs on every open, so an attacker who drops the triggers
        has to drop them again before each write. It does not make tampering
        impossible — nothing at this layer can, against someone with the file —
        but it does mean the guards cannot be removed once and forgotten, and
        any write through the normal code path is protected again.
        """
        _add_events(ledger, 2)
        ledger.close()

        tamper(ledger_path, "UPDATE event_log SET actor = 'human' WHERE seq = 2")

        conn = sqlite3.connect(ledger_path)
        present = {
            row[0]
            for row in conn.execute(
                "SELECT name FROM sqlite_master WHERE type = 'trigger'"
            ).fetchall()
        }
        conn.close()
        assert "event_log_no_update" not in present, "fixture should have dropped the guards"

        with Ledger(ledger_path) as reopened:
            restored = {
                row["name"]
                for row in reopened.conn.execute(
                    "SELECT name FROM sqlite_master WHERE type = 'trigger'"
                ).fetchall()
            }
            assert {
                "event_log_no_update",
                "event_log_no_delete",
                "event_log_chain_link",
                "event_log_seq_contiguous",
            } <= restored

            with pytest.raises(sqlite3.IntegrityError, match="append-only"):
                reopened.conn.execute("UPDATE event_log SET actor = 'llm' WHERE seq = 1")

            # And the tamper that happened while the guards were down is still
            # visible to verification.
            assert not verify_chain(reopened).ok


class TestReading:
    def test_iter_events_streams_in_sequence_order(self, ledger: Ledger) -> None:
        _add_events(ledger, 7)
        seqs = [int(r["seq"]) for r in ledger.iter_events(batch_size=2)]
        assert seqs == sorted(seqs)
        assert seqs == list(range(1, ledger.count() + 1))

    def test_iter_events_filters_by_type(self, ledger: Ledger) -> None:
        _add_events(ledger, 3)
        rows = list(ledger.iter_events(event_type=EventType.RUN_STARTED))
        assert len(rows) == 3
        assert {r["event_type"] for r in rows} == {EventType.RUN_STARTED.value}

    def test_tail_returns_the_most_recent_in_ascending_order(self, ledger: Ledger) -> None:
        _add_events(ledger, 5)
        rows = ledger.tail(3)
        seqs = [int(r["seq"]) for r in rows]
        assert seqs == sorted(seqs)
        assert seqs[-1] == ledger.count()

    def test_events_carry_the_code_and_config_pins(self, ledger_path: Path, pinned: Any) -> None:
        """Every event records which code and which limits produced it."""
        with Ledger(ledger_path, config_hash=pinned.config_hash) as led:
            led.initialise(created_by="test")
            row = led.get(1)
        assert row is not None
        assert row["config_hash"] == pinned.config_hash
        assert row["schema_version"] == LEDGER_SCHEMA_VERSION

    def test_read_only_ledger_refuses_writes(self, ledger: Ledger, ledger_path: Path) -> None:
        ledger.close()
        with Ledger(ledger_path, read_only=True) as ro:
            assert ro.count() >= 1
            with pytest.raises(LedgerError, match="read-only"), ro.transaction():
                pass

    def test_opening_a_missing_read_only_ledger_fails(self, tmp_path: Path) -> None:
        with pytest.raises(LedgerError, match="no ledger at"):
            Ledger(tmp_path / "absent.db", read_only=True).open()


def test_schema_drift_is_named_rather_than_silently_ignored(tmp_path: Path) -> None:
    """The failure `CREATE TABLE IF NOT EXISTS` cannot prevent.

    Adding a table is free; adding a column to an existing one is silently
    ignored, the statement succeeds, and the first read of the new column
    raises an IndexError from inside a row mapper with nothing naming the
    cause. This drops a column from a live table and asserts that opening the
    ledger refuses, names the table and the column, and says what to do.
    """
    import sqlite3

    from tb.ledger.schema import SchemaDriftError, apply_schema, check_schema_drift

    path = tmp_path / "drifted.db"
    with Ledger(path) as ledger:
        ledger.initialise(created_by="test")

    conn = sqlite3.connect(path)
    try:
        conn.execute("ALTER TABLE holdout_evaluations DROP COLUMN resolution")
        conn.commit()
        assert any("resolution" in note for note in check_schema_drift(conn))
    finally:
        conn.close()

    with pytest.raises(SchemaDriftError) as caught, Ledger(path) as ledger:
        apply_schema(ledger.conn)
    message = str(caught.value)
    assert "holdout_evaluations is missing resolution" in message
    assert "start a fresh ledger" in message


def test_an_intact_ledger_reports_no_drift(ledger: Ledger) -> None:
    """The other half: the check must not fire on a healthy database, or it
    would refuse every open and nobody would keep it."""
    from tb.ledger.schema import check_schema_drift

    assert check_schema_drift(ledger.conn) == []
