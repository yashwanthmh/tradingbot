"""The trial log — the denominator every deflated metric divides by."""

from __future__ import annotations

from datetime import UTC, datetime

import pytest

from tb.ledger.events import EventType
from tb.ledger.store import Ledger
from tb.registry.models import AuthorKind, TrialOutcome
from tb.research.trials import (
    FALLBACK_SHARPE_DISPERSION,
    MAX_STORED_RETURNS,
    SearchSession,
    Trial,
    TrialError,
    TrialLog,
    multiplicity_of,
    returns_matrix,
)

AS_OF = datetime(2026, 6, 1, 14, 0, tzinfo=UTC)


@pytest.fixture
def log(ledger: Ledger) -> TrialLog:
    return TrialLog(ledger, run_id="run_test")


def a_trial(
    *,
    lineage: int = 1,
    search: int = 1,
    sharpe: float | None = 1.0,
    outcome: TrialOutcome = TrialOutcome.EVALUATED,
) -> Trial:
    return Trial(
        trial_id="trial_x",
        search_id="srch_x",
        lineage_id="lin_x",
        spec_hash="h",
        author_kind=AuthorKind.SEARCH,
        outcome=outcome,
        recorded_at=AS_OF,
        net_sharpe=sharpe,
        trials_in_lineage_at_time=lineage,
        trials_in_search_at_time=search,
    )


# --------------------------------------------------------------------------
# Recording
# --------------------------------------------------------------------------


def test_a_trial_writes_a_row_and_an_event(log: TrialLog, ledger: Ledger) -> None:
    trial = log.record(
        search_id="srch_1",
        lineage_id="lin_1",
        spec_hash="hash_1",
        author_kind=AuthorKind.SEARCH,
        outcome=TrialOutcome.EVALUATED,
        net_sharpe=0.8,
        n_trades=41,
        at=AS_OF,
    )
    row = ledger.conn.execute(
        "SELECT * FROM trials WHERE trial_id = ?", (trial.trial_id,)
    ).fetchone()
    assert row["outcome"] == "evaluated"
    assert row["net_sharpe"] == 0.8

    events = ledger.conn.execute(
        "SELECT COUNT(*) FROM event_log WHERE event_type = ?",
        (EventType.TRIAL_RECORDED.value,),
    ).fetchone()
    assert events[0] == 1


def test_rejections_are_recorded_and_counted(log: TrialLog) -> None:
    """The load-bearing case. A search of a thousand that keeps three is a
    search of a thousand, and the haircut has to know that."""
    for index in range(5):
        log.record(
            search_id="srch_1",
            lineage_id="lin_1",
            spec_hash=f"hash_{index}",
            author_kind=AuthorKind.SEARCH,
            outcome=TrialOutcome.REJECTED,
            rejection_reason="cost gate",
            at=AS_OF,
        )
    assert log.count_in_search("srch_1") == 5
    assert log.count_in_lineage("lin_1") == 5


def test_errors_count_toward_multiplicity_too(log: TrialLog) -> None:
    """Otherwise a searcher lowers its own haircut by proposing specs that crash."""
    log.record(
        search_id="srch_1",
        lineage_id="lin_1",
        spec_hash="h1",
        author_kind=AuthorKind.SEARCH,
        outcome=TrialOutcome.ERRORED,
        at=AS_OF,
    )
    assert log.count_in_search("srch_1") == 1
    summary = log.summarise_search("srch_1")
    assert summary.n_errored == 1
    assert summary.n_proposed == 1


def test_a_rejection_must_carry_a_reason(log: TrialLog) -> None:
    with pytest.raises(TrialError, match="must carry a reason"):
        log.record(
            search_id="srch_1",
            lineage_id="lin_1",
            spec_hash="h",
            author_kind=AuthorKind.SEARCH,
            outcome=TrialOutcome.REJECTED,
            at=AS_OF,
        )


def test_the_counts_are_stamped_as_they_stood(log: TrialLog) -> None:
    """A count recomputed later describes a different search.

    The first trial of a search was drawn against a search of one; the
    hundredth against a search of a hundred. Recomputing at promotion time
    would deflate the first by the size the search eventually reached, which
    answers a question nobody asked.
    """
    first = log.record(
        search_id="srch_1",
        lineage_id="lin_1",
        spec_hash="h1",
        author_kind=AuthorKind.SEARCH,
        outcome=TrialOutcome.EVALUATED,
        at=AS_OF,
    )
    for index in range(2, 11):
        log.record(
            search_id="srch_1",
            lineage_id="lin_1",
            spec_hash=f"h{index}",
            author_kind=AuthorKind.SEARCH,
            outcome=TrialOutcome.EVALUATED,
            at=AS_OF,
        )
    assert first.trials_in_search_at_time == 1
    reread = log.latest_for_spec("h1")
    assert reread is not None
    assert reread.trials_in_search_at_time == 1

    tenth = log.latest_for_spec("h10")
    assert tenth is not None
    assert tenth.trials_in_search_at_time == 10


def test_the_event_can_be_suppressed_but_the_row_cannot(log: TrialLog, ledger: Ledger) -> None:
    """A thousand-spec calibration must not write a thousand chain events —
    but the count is what the gate divides by, so the row is never optional."""
    log.record(
        search_id="srch_1",
        lineage_id="lin_1",
        spec_hash="h1",
        author_kind=AuthorKind.SEARCH,
        outcome=TrialOutcome.EVALUATED,
        at=AS_OF,
        emit_event=False,
    )
    assert log.count_in_search("srch_1") == 1
    events = ledger.conn.execute(
        "SELECT COUNT(*) FROM event_log WHERE event_type = ?",
        (EventType.TRIAL_RECORDED.value,),
    ).fetchone()
    assert events[0] == 0


def test_stored_returns_are_bounded(log: TrialLog) -> None:
    trial = log.record(
        search_id="srch_1",
        lineage_id="lin_1",
        spec_hash="h",
        author_kind=AuthorKind.SEARCH,
        outcome=TrialOutcome.EVALUATED,
        returns=[0.001] * (MAX_STORED_RETURNS + 500),
        at=AS_OF,
    )
    assert len(trial.returns) == MAX_STORED_RETURNS


def test_completing_a_search_derives_its_totals_from_the_rows(
    log: TrialLog, ledger: Ledger
) -> None:
    log.record(
        search_id="s",
        lineage_id="l",
        spec_hash="a",
        author_kind=AuthorKind.SEARCH,
        outcome=TrialOutcome.PASSED_GATE,
        at=AS_OF,
    )
    log.record(
        search_id="s",
        lineage_id="l",
        spec_hash="b",
        author_kind=AuthorKind.SEARCH,
        outcome=TrialOutcome.REJECTED,
        rejection_reason="drawdown",
        at=AS_OF,
    )
    summary = log.complete_search("s", duration_seconds=1.5)
    assert summary.n_proposed == 2
    assert summary.n_passed_gate == 1
    assert summary.n_rejected == 1
    assert summary.promotion_rate == 0.5
    assert "cleared the gate" in summary.summary()

    events = ledger.conn.execute(
        "SELECT COUNT(*) FROM event_log WHERE event_type = ?",
        (EventType.SEARCH_COMPLETED.value,),
    ).fetchone()
    assert events[0] == 1


# --------------------------------------------------------------------------
# Multiplicity
# --------------------------------------------------------------------------


def test_multiplicity_takes_the_larger_of_the_two_counts(log: TrialLog) -> None:
    """The lineage-splitting evasion, closed.

    One trial per lineage makes every lineage a search of size one. Deflating
    on the lineage count alone would give each of a thousand candidates no
    haircut at all.
    """
    for index in range(20):
        log.record(
            search_id="srch_big",
            lineage_id=f"lin_{index}",
            spec_hash=f"h{index}",
            author_kind=AuthorKind.SEARCH,
            outcome=TrialOutcome.EVALUATED,
            net_sharpe=0.1 * index,
            at=AS_OF,
        )
    multiplicity = log.multiplicity_for("h19")
    assert multiplicity is not None
    assert multiplicity.n_lineage_trials == 1
    assert multiplicity.n_search_trials == 20
    assert multiplicity.n_trials == 20
    assert any("selection universe is the search" in note for note in multiplicity.caveats)


def test_multiplicity_uses_the_search_peers_when_the_search_is_larger(
    log: TrialLog,
) -> None:
    """The dispersion must come from the same population as the count.

    Counting 20 search trials while measuring dispersion over 1 lineage trial
    would pair a large N with an unmeasurable spread, and fall back to the
    conservative default while real data was sitting in the table.
    """
    for index in range(20):
        log.record(
            search_id="srch_big",
            lineage_id=f"lin_{index}",
            spec_hash=f"h{index}",
            author_kind=AuthorKind.SEARCH,
            outcome=TrialOutcome.EVALUATED,
            net_sharpe=0.1 * index,
            at=AS_OF,
        )
    multiplicity = log.multiplicity_for("h19")
    assert multiplicity is not None
    assert multiplicity.n_measurable == 20
    assert multiplicity.dispersion_measured


def test_an_unmeasurable_dispersion_falls_back_conservatively() -> None:
    """Zero would remove the haircut entirely, which is the wrong direction."""
    trial = a_trial(lineage=1, search=1)
    multiplicity = multiplicity_of(trial, [trial])
    assert not multiplicity.dispersion_measured
    assert multiplicity.sharpe_dispersion == FALLBACK_SHARPE_DISPERSION
    assert any("conservative default" in note for note in multiplicity.caveats)


def test_identical_trial_sharpes_are_not_a_measured_dispersion() -> None:
    """Three trials all at 1.0 have no spread to select from, which is not the
    same as a measured spread of zero."""
    peers = [a_trial(sharpe=1.0) for _ in range(3)]
    multiplicity = multiplicity_of(peers[0], peers)
    assert not multiplicity.dispersion_measured
    assert multiplicity.sharpe_dispersion == FALLBACK_SHARPE_DISPERSION


def test_unevaluated_trials_do_not_contribute_a_sharpe() -> None:
    peers = [
        a_trial(sharpe=1.0),
        a_trial(sharpe=None, outcome=TrialOutcome.REJECTED),
        a_trial(sharpe=3.0),
    ]
    multiplicity = multiplicity_of(peers[0], peers)
    assert multiplicity.n_measurable == 2
    assert multiplicity.dispersion_measured


def test_the_multiplicity_is_never_below_one() -> None:
    """A trial row is evidence that a search happened."""
    assert a_trial(lineage=0, search=0).n_trials_for_deflation == 1


def test_multiplicity_is_none_for_an_unseen_spec(log: TrialLog) -> None:
    assert log.multiplicity_for("never_recorded") is None


# --------------------------------------------------------------------------
# The returns matrix
# --------------------------------------------------------------------------


def test_the_returns_matrix_truncates_rather_than_pads() -> None:
    """Padding with zeros invents flat periods, which lowers dispersion and
    raises every Sharpe in the matrix."""
    trials = [
        Trial(
            trial_id="a",
            search_id="s",
            lineage_id="l",
            spec_hash="a",
            author_kind=AuthorKind.SEARCH,
            outcome=TrialOutcome.EVALUATED,
            recorded_at=AS_OF,
            returns=(0.1, 0.2, 0.3, 0.4),
        ),
        Trial(
            trial_id="b",
            search_id="s",
            lineage_id="l",
            spec_hash="b",
            author_kind=AuthorKind.SEARCH,
            outcome=TrialOutcome.EVALUATED,
            recorded_at=AS_OF,
            returns=(0.5, 0.6),
        ),
    ]
    matrix = returns_matrix(trials)
    assert matrix == ((0.3, 0.4), (0.5, 0.6))


def test_the_returns_matrix_is_empty_when_nothing_has_returns() -> None:
    assert returns_matrix([a_trial()]) == ()


# --------------------------------------------------------------------------
# The session wrapper
# --------------------------------------------------------------------------


def test_a_search_session_keeps_one_id(log: TrialLog) -> None:
    session = SearchSession(log=log)
    for index in range(3):
        session.record(
            lineage_id="lin_1",
            spec_hash=f"h{index}",
            author_kind=AuthorKind.SEARCH,
            outcome=TrialOutcome.EVALUATED,
            at=AS_OF,
        )
    assert log.count_in_search(session.search_id) == 3


def test_a_search_session_refuses_a_second_search_id(log: TrialLog) -> None:
    """Splitting one search in two would halve the haircut on each half."""
    session = SearchSession(log=log)
    with pytest.raises(TrialError, match="owns its search_id"):
        session.record(
            search_id="srch_other",
            lineage_id="lin_1",
            spec_hash="h",
            author_kind=AuthorKind.SEARCH,
            outcome=TrialOutcome.EVALUATED,
        )
