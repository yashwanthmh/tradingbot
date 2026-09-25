"""One search cycle, end to end: the vintage, the seal, the trials, the registry.

Everything here runs the real pipeline on a sealed random-walk vintage — the
same fixture the holdout command's tests use — so there is no edge to find and
nothing should survive the gate. That is fine: this suite is about what the
cycle *records*, and the properties that matter are all about the record.

* **Every candidate is a trial,** refused and broken ones included, and a dry
  run still counts: the search happened.
* **Every candidate is stamped with the whole batch.** A batch selects after all
  N have run, so a survivor recorded tenth was still chosen from all N.
* **The search never reads past the seal.** Asserted on the decision times the
  backtester was actually handed, not inferred from the absence of an error.
* **Lineage carries across searches.** A second search seeded from a registered
  winner stays in its lineage, and the lineage count keeps growing — which is
  what stops "many small searches" of one idea each taking a small haircut.
"""

from __future__ import annotations

from collections.abc import Callable
from datetime import UTC, datetime, timedelta
from decimal import Decimal
from pathlib import Path
from typing import Any

import pytest

from tb.backtest.engine import Backtester
from tb.config.loader import load_hard_limits
from tb.data.provider import Bar, Provenance, Resolution, Session
from tb.ledger.store import Ledger
from tb.registry.models import StrategyStatus
from tb.research.holdout import DECISION_OFFSET, decisions_between
from tb.research.trials import TrialLog
from tests.test_cli_registry import _init, _out, _run, seed_bars


@pytest.fixture
def cli_env(tmp_path: Path, write_limits: Callable[[dict[str, Any]], Path]) -> dict[str, Any]:
    """A ledger, a bar store and a limits file, addressed the way the CLI takes them."""
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


def _ledger(env: dict[str, Any]) -> Ledger:
    pinned = load_hard_limits(env["limits"])
    return Ledger(env["db"], config_hash=pinned.config_hash).open()


def _sealed_vintage(env: dict[str, Any]) -> str:
    _init(env)
    seed_bars(env)
    sealed = _run(["data", "seal", "--resolution", "daily", *env["bar_args"]])
    assert sealed.exit_code == 0, _out(sealed)
    with _ledger(env) as ledger:
        row = ledger.conn.execute("SELECT vintage_id FROM data_snapshots").fetchone()
    return str(row["vintage_id"])


def _cycle(env: dict[str, Any], vintage_id: str, *extra: str, trials: int = 16) -> Any:
    return _run(
        [
            "research",
            "cycle",
            vintage_id,
            "--trials",
            str(trials),
            "--per-generation",
            "8",
            "--seed",
            "3",
            *extra,
            *env["bar_args"],
        ]
    )


# --------------------------------------------------------------------------
# The record
# --------------------------------------------------------------------------


def test_a_cycle_records_every_trial_including_the_refused(cli_env: dict[str, Any]) -> None:
    """**The denominator.** Proposed equals recorded, whatever became of each.

    A cycle that recorded only what it evaluated would report a search of eight
    where sixteen happened — and the deflated Sharpe's haircut is computed from
    that count.
    """
    vintage_id = _sealed_vintage(cli_env)
    result = _cycle(cli_env, vintage_id)
    assert result.exit_code == 0, _out(result)

    with _ledger(cli_env) as ledger:
        rows = ledger.conn.execute("SELECT outcome FROM trials").fetchall()
        completed = ledger.conn.execute(
            "SELECT payload_json FROM event_log WHERE event_type = 'search.completed'"
        ).fetchone()
    assert len(rows) == 16
    assert completed is not None, "the search was never closed with its totals"
    assert '"n_proposed":16' in str(completed["payload_json"]).replace(" ", "")


def test_a_dry_run_registers_nothing_and_still_counts(cli_env: dict[str, Any]) -> None:
    """The search happened, so its trials are recorded either way.

    Otherwise ten dry runs and a hand-registration of the best result is a search
    of hundreds whose size nobody ever wrote down.
    """
    vintage_id = _sealed_vintage(cli_env)
    result = _cycle(cli_env, vintage_id)
    assert result.exit_code == 0, _out(result)
    assert "dry run: nothing registered" in _out(result)

    with _ledger(cli_env) as ledger:
        assert ledger.conn.execute("SELECT COUNT(*) FROM strategy_specs").fetchone()[0] == 0
        assert ledger.conn.execute("SELECT COUNT(*) FROM trials").fetchone()[0] == 16


def test_apply_registers_survivors_as_candidates_and_promotes_nothing(
    cli_env: dict[str, Any],
) -> None:
    """Registration is not promotion.

    The cycle chooses; the holdout and the gate check the choice. A cycle that
    did both would be a process marking its own homework, which is the
    arrangement the sealed holdout exists to prevent.
    """
    vintage_id = _sealed_vintage(cli_env)
    result = _cycle(cli_env, vintage_id, "--apply", trials=24)
    assert result.exit_code == 0, _out(result)

    with _ledger(cli_env) as ledger:
        statuses = [
            str(row["status"])
            for row in ledger.conn.execute("SELECT status FROM strategy_status").fetchall()
        ]
        trial_lineages = {
            str(row["spec_hash"]): str(row["lineage_id"])
            for row in ledger.conn.execute("SELECT spec_hash, lineage_id FROM trials").fetchall()
        }
        registered = ledger.conn.execute(
            "SELECT spec_hash, lineage_id FROM strategy_specs"
        ).fetchall()

    assert statuses, "a random walk over 24 trials should leave something rankable"
    assert all(status == StrategyStatus.CANDIDATE.value for status in statuses)
    # The lineage a survivor was registered under is the one its trials were
    # counted against. If they differed, its haircut would be computed from a
    # lineage nobody reads.
    for row in registered:
        assert trial_lineages[str(row["spec_hash"])] == str(row["lineage_id"])


def test_every_candidate_is_stamped_with_the_whole_batch(cli_env: dict[str, Any]) -> None:
    """**A batch selects after all N have run.**

    The trial log's default stamp is the running count, which is right for a
    search that decides as it goes. Here a survivor recorded tenth was still
    chosen from all sixteen, and stamping it with ten would deflate it against a
    fraction of the search that produced it — the permissive direction.
    """
    vintage_id = _sealed_vintage(cli_env)
    result = _cycle(cli_env, vintage_id, "--apply")
    assert result.exit_code == 0, _out(result)

    with _ledger(cli_env) as ledger:
        stamps = [
            int(row["trials_in_search_at_time"])
            for row in ledger.conn.execute("SELECT trials_in_search_at_time FROM trials").fetchall()
        ]
        survivors = [
            str(row["spec_hash"])
            for row in ledger.conn.execute("SELECT spec_hash FROM strategy_specs").fetchall()
        ]
        log = TrialLog(ledger)
        multiplicities = [log.multiplicity_for(spec_hash) for spec_hash in survivors]

    assert stamps == [16] * 16, stamps
    for multiplicity in multiplicities:
        assert multiplicity is not None
        assert multiplicity.n_trials >= 16


def test_the_search_never_decides_at_or_past_the_seal(
    cli_env: dict[str, Any], monkeypatch: pytest.MonkeyPatch
) -> None:
    """**Asserted on what the backtester was handed**, not on a missing error.

    The sealed reader would raise if the schedule crossed the boundary, so a
    cycle that ran at all has not crossed it — but "it did not crash" is an
    argument from absence. This records every decision time the search actually
    requested and checks each one against the boundary.
    """
    vintage_id = _sealed_vintage(cli_env)
    seen: list[datetime] = []
    original = Backtester.run

    def recording(self: Backtester, **kwargs: Any) -> Any:
        seen.extend(kwargs["decision_times"])
        return original(self, **kwargs)

    monkeypatch.setattr(Backtester, "run", recording)
    result = _cycle(cli_env, vintage_id)
    assert result.exit_code == 0, _out(result)
    assert seen, "no backtest ran, so the property was never exercised"

    from tb.data.barstore import BarStore
    from tb.data.snapshot import SnapshotStore
    from tb.research.holdout import holdout_boundary

    pinned = load_hard_limits(cli_env["limits"])
    with _ledger(cli_env) as ledger:
        snapshots = SnapshotStore(
            ledger, BarStore(ledger, root=cli_env["bars"], scale=pinned.limits.data.price_scale)
        )
        vintage = snapshots.get(vintage_id)
    assert vintage is not None
    boundary = holdout_boundary(vintage).sealed_from
    assert max(seen) < boundary, f"a search decision at {max(seen)} reached the seal"


def test_a_second_search_seeded_from_a_winner_stays_in_its_lineage(
    cli_env: dict[str, Any],
) -> None:
    """**Refinement accumulates the lineage count.**

    ADR 0002's consequence is "many small searches", and the obvious worry is
    that it is an evasion: ten searches of ten, each taking a haircut of ten. It
    is not, because a search seeded from a registered strategy stays in that
    strategy's lineage, and the lineage count grows across every search that
    touches it. A new idea gets a small N; refining an old one does not.
    """
    vintage_id = _sealed_vintage(cli_env)
    first = _cycle(cli_env, vintage_id, "--apply", trials=24)
    assert first.exit_code == 0, _out(first)

    with _ledger(cli_env) as ledger:
        row = ledger.conn.execute(
            "SELECT strategy_id, lineage_id FROM strategy_specs ORDER BY registered_at LIMIT 1"
        ).fetchone()
        assert row is not None, "the first search registered nothing to seed from"
        seed_id, lineage = str(row["strategy_id"]), str(row["lineage_id"])
        before = TrialLog(ledger).count_in_lineage(lineage)

    second = _run(
        [
            "research",
            "cycle",
            vintage_id,
            "--trials",
            "8",
            "--per-generation",
            "8",
            "--seed",
            "5",
            "--from",
            seed_id,
            *cli_env["bar_args"],
        ]
    )
    assert second.exit_code == 0, _out(second)

    with _ledger(cli_env) as ledger:
        search = ledger.conn.execute(
            "SELECT search_id FROM trials ORDER BY recorded_at DESC LIMIT 1"
        ).fetchone()["search_id"]
        rows = ledger.conn.execute(
            "SELECT lineage_id, trials_in_lineage_at_time FROM trials WHERE search_id = ?",
            (search,),
        ).fetchall()

    in_lineage = [row for row in rows if str(row["lineage_id"]) == lineage]
    assert in_lineage, "no mutation of the seed was recorded in the seed's lineage"
    # The stamp includes the first search's trials in this lineage, not only
    # the second search's.
    assert all(int(row["trials_in_lineage_at_time"]) > before for row in in_lineage)


def test_an_unregistered_seed_is_a_setup_error(cli_env: dict[str, Any]) -> None:
    vintage_id = _sealed_vintage(cli_env)
    result = _cycle(cli_env, vintage_id, "--from", "stg_ghost")
    assert result.exit_code == 2
    assert "not registered" in _out(result)


def test_an_unsealed_vintage_is_a_setup_error(cli_env: dict[str, Any]) -> None:
    _init(cli_env)
    result = _cycle(cli_env, "vint_ghost")
    assert result.exit_code == 2


def test_an_unusable_budget_is_a_setup_error(cli_env: dict[str, Any]) -> None:
    _init(cli_env)
    result = _cycle(cli_env, "vint_ghost", trials=0)
    assert result.exit_code == 2
    assert "at least one trial" in _out(result)


def test_the_report_names_the_sharpe_the_search_size_requires(
    cli_env: dict[str, Any],
) -> None:
    """The most decision-relevant number about a search, printed with it."""
    vintage_id = _sealed_vintage(cli_env)
    result = _cycle(cli_env, vintage_id)
    assert result.exit_code == 0, _out(result)
    assert "needs an out-of-sample Sharpe" in _out(result)


# --------------------------------------------------------------------------
# The schedule and the stamp, as units
# --------------------------------------------------------------------------


def _bar(uid: str, opened: datetime) -> Bar:
    return Bar(
        instrument_uid=uid,
        resolution=Resolution.DAILY,
        bar_open_utc=opened,
        available_at_utc=opened + timedelta(days=1),
        ingested_at_utc=opened + timedelta(days=1),
        provider="fixture",
        provenance=Provenance.BACKFILL,
        session=Session.REGULAR,
        open=Decimal("10"),
        high=Decimal("11"),
        low=Decimal("9"),
        close=Decimal("10"),
        volume=1,
    )


def test_the_schedule_is_deduplicated_across_instruments() -> None:
    """Twenty-five instruments must not mean twenty-five decisions per session.

    An undeduplicated schedule decides at the same instant once per instrument,
    producing an equity curve with a run of zero returns per day — which
    deflates measured volatility and inflates every Sharpe read from it.
    """
    start = datetime(2025, 1, 1, tzinfo=UTC)
    bars = [
        _bar(uid, start + timedelta(days=day))
        for day in range(10)
        for uid in ("isin:A", "isin:B", "isin:C")
    ]
    schedule = decisions_between(bars)
    assert len(schedule) == 10
    assert schedule == sorted(set(schedule))


def test_the_schedule_splits_on_the_decision_time_with_no_gap() -> None:
    """Filtered on the decision, not on the bar.

    A bar opened the day before the seal is knowable the day after it, so a
    training schedule filtered on `bar_open` ends with a decision past the seal
    — which the sealed reader refuses, killing the search on its last bar.
    Split on the decision itself, training and holdout partition time exactly.
    """
    start = datetime(2025, 1, 1, tzinfo=UTC)
    bars = [_bar("isin:A", start + timedelta(days=day)) for day in range(20)]
    seal = start + timedelta(days=10, hours=12)

    training = decisions_between(bars, end=seal)
    holdout = decisions_between(bars, start=seal)

    assert max(training) < seal <= min(holdout)
    assert sorted(training + holdout) == decisions_between(bars), "a decision fell in a gap"
    assert all(t - DECISION_OFFSET in {b.available_at_utc for b in bars} for t in training)


def test_a_batch_floor_can_raise_a_stamp_and_never_lower_one(ledger: Ledger) -> None:
    """The floor exists to count a selection correctly, never to shrink a haircut."""
    from tb.registry.models import AuthorKind, TrialOutcome

    log = TrialLog(ledger)
    for index in range(5):
        log.record(
            search_id="srch_floor",
            lineage_id="lin_floor",
            spec_hash=f"hash_{index}",
            author_kind=AuthorKind.SEARCH,
            outcome=TrialOutcome.EVALUATED,
        )
    raised = log.record(
        search_id="srch_floor",
        lineage_id="lin_floor",
        spec_hash="hash_raised",
        author_kind=AuthorKind.SEARCH,
        outcome=TrialOutcome.EVALUATED,
        selected_from_search=50,
        selected_from_lineage=80,
    )
    assert raised.trials_in_search_at_time == 50
    assert raised.trials_in_lineage_at_time == 80

    not_lowered = log.record(
        search_id="srch_floor",
        lineage_id="lin_floor",
        spec_hash="hash_low",
        author_kind=AuthorKind.SEARCH,
        outcome=TrialOutcome.EVALUATED,
        selected_from_search=1,
        selected_from_lineage=1,
    )
    # Seven rows now exist in each count; a floor of 1 must not pull that down.
    assert not_lowered.trials_in_search_at_time == 7
    assert not_lowered.trials_in_lineage_at_time == 7
