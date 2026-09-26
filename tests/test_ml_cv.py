"""Purged, embargoed walk-forward folds: the properties that make ML scores honest.

A label spans time — decided today, known a week from now — so the ordinary
cross-validation shuffle leaks: training rows carry labels computed from the
prices a test fold is scored on. Every property here is asserted over random
spans rather than hand-picked ones, because the leak lives in the cases nobody
thinks to write: several instruments at one instant, a horizon longer than a
fold, an embargo that swallows a fold's whole history.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from itertools import pairwise

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from tb.strategy.ml.cv import CrossValidationError, LabelSpan, walk_forward_folds

BASE = datetime(2025, 1, 6, 21, tzinfo=UTC)


def span(day: int, horizon: int) -> LabelSpan:
    decided = BASE + timedelta(days=day)
    return LabelSpan(decided_at=decided, known_at=decided + timedelta(days=horizon))


@given(
    samples=st.lists(
        st.tuples(st.integers(min_value=0, max_value=120), st.integers(min_value=0, max_value=12)),
        min_size=2,
        max_size=150,
    ),
    n_folds=st.integers(min_value=1, max_value=6),
    embargo_days=st.integers(min_value=0, max_value=6),
)
@settings(max_examples=300, deadline=None)
def test_no_fold_trains_on_a_label_it_could_not_have_known(
    samples: list[tuple[int, int]], n_folds: int, embargo_days: int
) -> None:
    spans = [span(day, horizon) for day, horizon in samples]
    embargo = timedelta(days=embargo_days)
    try:
        folds = walk_forward_folds(spans, n_folds=n_folds, embargo=embargo)
    except CrossValidationError:
        return

    seen_test: set[int] = set()
    for fold in folds:
        test_times = {spans[i].decided_at for i in fold.test}
        for i in fold.train:
            # Walk-forward, purged and embargoed, in one line each.
            assert spans[i].decided_at < fold.test_start
            assert spans[i].known_at + embargo <= fold.test_start
            # So no instant is ever on both sides of a fold.
            assert spans[i].decided_at not in test_times
        for i in fold.test:
            assert fold.test_start <= spans[i].decided_at <= fold.test_end
        # Purging removes exactly what it must and nothing more.
        eligible = [
            i
            for i, candidate in enumerate(spans)
            if candidate.decided_at < fold.test_start
            and candidate.known_at + embargo <= fold.test_start
        ]
        assert list(fold.train) == eligible
        before = sum(1 for candidate in spans if candidate.decided_at < fold.test_start)
        assert fold.n_purged == before - len(eligible)
        assert not seen_test & set(fold.test), "a sample was scored twice"
        seen_test |= set(fold.test)

    for earlier, later in pairwise(folds):
        assert earlier.test_end < later.test_start


def test_the_leak_purging_exists_to_stop() -> None:
    """Daily samples with five-session labels: the four sessions before each
    fold are decided in the past but labelled from prices inside the fold. A
    plain time split would train on them."""
    spans = [span(day, 5) for day in range(60)]
    folds = walk_forward_folds(spans, n_folds=3, embargo=timedelta(0))

    for fold in folds:
        naive = [i for i, candidate in enumerate(spans) if candidate.decided_at < fold.test_start]
        leaked = [i for i in naive if spans[i].known_at > fold.test_start]
        assert len(leaked) == 4
        assert not set(leaked) & set(fold.train)
        assert fold.n_purged == 4


def test_an_embargo_widens_the_gap_before_each_fold() -> None:
    spans = [span(day, 5) for day in range(60)]
    plain = walk_forward_folds(spans, n_folds=3, embargo=timedelta(0))
    embargoed = walk_forward_folds(spans, n_folds=3, embargo=timedelta(days=3))

    for without, with_gap in zip(plain, embargoed, strict=True):
        assert with_gap.n_purged == without.n_purged + 3
        assert set(with_gap.train) < set(without.train)


def test_instruments_sharing_an_instant_stay_on_one_side_of_every_boundary() -> None:
    """Three instruments decided together each day: a boundary through the
    middle of one instant would put that moment in training and in test."""
    spans = [span(day, 2) for day in range(30) for _ in range(3)]
    for fold in walk_forward_folds(spans, n_folds=4, embargo=timedelta(days=1)):
        assert len(fold.test) % 3 == 0
        test_times = {spans[i].decided_at for i in fold.test}
        assert not {spans[i].decided_at for i in fold.train} & test_times


def test_a_fold_with_too_little_history_is_dropped_not_scored() -> None:
    spans = [span(day, 10) for day in range(20)]
    folds = walk_forward_folds(spans, n_folds=4, embargo=timedelta(0), min_train=3)
    assert folds, "the later folds have enough history"
    assert all(len(fold.train) >= 3 for fold in folds)
    assert folds[0].index > 1, "the first fold had too little history to score"


def test_history_too_short_for_the_horizon_is_refused() -> None:
    spans = [span(day, 30) for day in range(10)]
    with pytest.raises(CrossValidationError, match="too short for the label horizon"):
        walk_forward_folds(spans, n_folds=2, embargo=timedelta(0))
    with pytest.raises(CrossValidationError, match="distinct decision time"):
        walk_forward_folds(spans[:2], n_folds=2, embargo=timedelta(0))
    with pytest.raises(CrossValidationError, match="negative embargo"):
        walk_forward_folds(spans, n_folds=2, embargo=timedelta(days=-1))


def test_a_label_known_before_its_decision_is_refused() -> None:
    with pytest.raises(CrossValidationError, match="label from the future"):
        LabelSpan(decided_at=BASE, known_at=BASE - timedelta(seconds=1))
