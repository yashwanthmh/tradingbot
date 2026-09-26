"""Labelled samples: features the loop would compute, labels that are the trade.

Every row a model is fitted to must be one the live system could have produced
at that instant, labelled with what acting on it would actually have realised.
These tests pin both halves against the machinery the rest of the system uses
— the forward-only reader, the one pipeline, the backtester's fill and its
split handling — rather than against a second implementation of any of them.
"""

from __future__ import annotations

from datetime import UTC, date, datetime, timedelta
from decimal import Decimal
from fractions import Fraction

import pytest

from tb.data.adjustments import ActionType, CorporateAction
from tb.data.asof import ForwardOnlyReader, InMemoryBarSource
from tb.data.provider import Bar, Provenance, Resolution, Session
from tb.features.pipeline import FeaturePipeline, make_spec
from tb.research.holdout import decisions_between, training_reader
from tb.strategy.ml.dataset import DatasetError, LabelDefinition, build_dataset

UID = "isin:US0378331005"
OTHER = "isin:US5949181045"
BASE = datetime(2026, 1, 5, tzinfo=UTC)
COST = Decimal("40")


def daily(offset: int, price: str, *, uid: str = UID, close: str | None = None) -> Bar:
    opened = BASE + timedelta(days=offset)
    first, last = Decimal(price), Decimal(close or price)
    return Bar(
        instrument_uid=uid,
        resolution=Resolution.DAILY,
        bar_open_utc=opened,
        available_at_utc=opened + timedelta(days=1),
        ingested_at_utc=opened + timedelta(days=1),
        provider="alpaca",
        provenance=Provenance.BACKFILL,
        session=Session.REGULAR,
        open=first,
        high=max(first, last),
        low=min(first, last),
        close=last,
        volume=1_000_000,
    )


def ramp(days: int, *, uid: str = UID) -> list[Bar]:
    """Opens 100, 101, 102…; each close half a point above its open."""
    return [
        daily(i, str(Decimal(100 + i)), uid=uid, close=str(Decimal(100 + i) + Decimal("0.5")))
        for i in range(days)
    ]


def reader(bars: list[Bar], uids: tuple[str, ...] = (UID,)) -> ForwardOnlyReader:
    return ForwardOnlyReader(
        source=InMemoryBarSource(bars=bars), resolution=Resolution.DAILY, instrument_uids=uids
    )


PIPELINE = FeaturePipeline(specs=(make_spec("return_pct", 2), make_spec("last", 1)))


def test_a_label_is_the_round_trip_a_trade_would_have_made() -> None:
    """In at the next bar's open, out at the open three bars on, less costs."""
    bars = ramp(30)
    label = LabelDefinition(horizon=3, cost_bps=COST)
    data = build_dataset(
        reader=reader(bars),
        pipeline=PIPELINE,
        decision_times=decisions_between(bars),
        instruments=(UID,),
        label=label,
    )

    decided = bars[5].available_at_utc + timedelta(hours=1)
    (sample,) = [s for s in data.samples if s.decided_at == decided]
    entry, exit_ = bars[6], bars[9]
    assert sample.gross_return == pytest.approx(exit_.open / entry.open - 1, abs=Decimal("1e-20"))
    assert sample.net_return == pytest.approx(
        sample.gross_return - COST / Decimal(10_000), abs=Decimal("1e-20")
    )
    assert sample.label == 1
    assert sample.span.known_at == exit_.available_at_utc
    # The last decisions have no exit bar yet: dropped, never guessed.
    assert data.n_unlabelled == 3 + 1


def test_no_label_is_completed_from_beyond_the_seal() -> None:
    """Through the reader a research process gets, a label whose exit lies in
    the holdout never completes — the prices it would need are not in memory."""
    bars = ramp(40)
    sealed_from = bars[30].available_at_utc
    schedule = decisions_between(bars, end=sealed_from)
    data = build_dataset(
        reader=training_reader(
            InMemoryBarSource(bars=bars),
            sealed_from=sealed_from,
            resolution=Resolution.DAILY,
            instrument_uids=(UID,),
        ),
        pipeline=PIPELINE,
        decision_times=schedule,
        instruments=(UID,),
        label=LabelDefinition(horizon=4, cost_bps=COST),
    )

    assert data.samples
    assert all(sample.span.known_at < sealed_from for sample in data.samples)
    assert data.n_unlabelled == 4 + 1


def test_the_features_are_the_ones_the_live_pipeline_computes() -> None:
    bars = ramp(20)
    data = build_dataset(
        reader=reader(bars),
        pipeline=PIPELINE,
        decision_times=decisions_between(bars),
        instruments=(UID,),
        label=LabelDefinition(horizon=2, cost_bps=COST),
    )

    replayed = reader(bars)
    by_time = {sample.decided_at: sample for sample in data.samples}
    for moment in decisions_between(bars):
        window = replayed.advance_to(moment)
        if moment not in by_time:
            continue
        snapshot = PIPELINE.compute(window, UID)
        assert by_time[moment].features == tuple(
            float(snapshot.values[name]) for name in PIPELINE.names
        )
    assert data.feature_names == PIPELINE.names


def test_a_split_inside_the_horizon_is_no_move() -> None:
    """A 4-for-1 between entry and exit: four shares for each one bought, so
    100 in and 25 out is exactly nothing, as the backtester books it."""
    bars = [daily(i, "100" if i < 10 else "25") for i in range(20)]
    split = CorporateAction(
        action_id="act_split",
        instrument_uid=UID,
        action_type=ActionType.SPLIT,
        effective_date=date(2026, 1, 15),
        known_at_utc=BASE - timedelta(days=30),
        source_provider="alpaca",
        ratio_num=4,
        ratio_den=1,
    )
    label = LabelDefinition(horizon=3, cost_bps=Decimal(0))

    def gross(actions: dict[str, list[CorporateAction]]) -> Decimal:
        data = build_dataset(
            reader=reader(bars),
            pipeline=PIPELINE,
            decision_times=decisions_between(bars),
            instruments=(UID,),
            label=label,
            actions=actions,
        )
        decided = bars[7].available_at_utc + timedelta(hours=1)
        (sample,) = [s for s in data.samples if s.decided_at == decided]
        return sample.gross_return

    assert split.split_ratio == Fraction(4, 1)
    assert gross({UID: [split]}) == 0
    assert gross({}) == Decimal("-0.75")


def test_a_row_with_an_unknown_feature_is_dropped_and_counted() -> None:
    """A ten-bar average has no value for the first nine decisions: the loop
    would have held on each, so none of them is a row to learn from."""
    bars = ramp(25)
    data = build_dataset(
        reader=reader(bars),
        pipeline=FeaturePipeline(specs=(make_spec("sma", 10),)),
        decision_times=decisions_between(bars),
        instruments=(UID,),
        label=LabelDefinition(horizon=2, cost_bps=COST),
    )
    assert data.n_unknown == 9
    assert data.n_decisions == 25
    assert len(data.samples) + data.n_unknown + data.n_unlabelled == data.n_decisions


def test_each_instrument_is_labelled_from_its_own_prices() -> None:
    falling = [daily(i, str(200 - i), uid=OTHER) for i in range(20)]
    bars = [*ramp(20), *falling]
    data = build_dataset(
        reader=reader(bars, (UID, OTHER)),
        pipeline=PIPELINE,
        decision_times=decisions_between(bars),
        instruments=(UID, OTHER),
        label=LabelDefinition(horizon=2, cost_bps=Decimal(0)),
    )
    rising = [s for s in data.samples if s.instrument_uid == UID]
    sinking = [s for s in data.samples if s.instrument_uid == OTHER]
    assert rising and sinking
    assert all(s.label == 1 for s in rising)
    assert all(s.label == 0 for s in sinking)
    assert data.base_rate == pytest.approx(0.5)


@pytest.mark.parametrize(
    ("lookback_days", "every", "horizon"),
    [
        # Daily decisions: the label could never complete, and would have been
        # dropped as "unlabelled" with nothing to say why.
        (3, 1, 5),
        # Every tenth session: at the next decision the window holds six later
        # bars, so the label *would* complete — from the wrong entry bar.
        (6, 10, 2),
    ],
)
def test_a_lookback_shorter_than_the_label_is_refused_not_mislabelled(
    lookback_days: int, every: int, horizon: int
) -> None:
    """Enough history for every feature at the decision, and too little to
    still hold the decision's bars when its exit arrives."""
    bars = ramp(60)
    short = ForwardOnlyReader(
        source=InMemoryBarSource(bars=bars),
        resolution=Resolution.DAILY,
        instrument_uids=(UID,),
        lookback=timedelta(days=lookback_days),
    )
    with pytest.raises(DatasetError, match="lookback is shorter"):
        build_dataset(
            reader=short,
            pipeline=PIPELINE,
            decision_times=decisions_between(bars)[::every],
            instruments=(UID,),
            label=LabelDefinition(horizon=horizon, cost_bps=COST),
        )


def test_decision_times_must_run_forward() -> None:
    bars = ramp(10)
    with pytest.raises(DatasetError, match="ascending"):
        build_dataset(
            reader=reader(bars),
            pipeline=PIPELINE,
            decision_times=list(reversed(decisions_between(bars))),
            instruments=(UID,),
            label=LabelDefinition(horizon=2, cost_bps=COST),
        )
    with pytest.raises(DatasetError, match="holds nothing"):
        LabelDefinition(horizon=0, cost_bps=COST)
