"""The feature pipeline.

Two things these tests are really about. That absence stays absent — every
feature returns UNKNOWN rather than a number computed over too little data,
because a 50-bar average over 10 bars is not a shorter average but a different
number wearing the same name. And that a split part-way through a window does
not read as a return, which is the whole reason features run on the
split-adjusted series rather than the raw one.
"""

from __future__ import annotations

from datetime import UTC, date, datetime, timedelta
from decimal import Decimal
from fractions import Fraction

import pytest

from tb.data.adjustments import RESIDUAL_DETECTOR, ActionType, CorporateAction, Series
from tb.data.asof import UNKNOWN, BarWindow, UnknownValueError
from tb.data.provider import Bar, DataError, Provenance, Resolution, Session
from tb.features.pipeline import (
    FEATURE_LIBRARY,
    FeatureError,
    FeaturePipeline,
    FeatureSpec,
    default_pipeline,
    make_spec,
)

UID = "isin:US0378331005"


BASE = datetime(2026, 3, 2, tzinfo=UTC)


def bar(offset: int, close: str) -> Bar:
    """A daily bar `offset` days after the base date.

    Offsets rather than day-of-month numbers: the sequences here run past 30
    bars, and day arithmetic would walk off the end of March.
    """
    opened = BASE + timedelta(days=offset)
    price = Decimal(close)
    return Bar(
        instrument_uid=UID,
        resolution=Resolution.DAILY,
        bar_open_utc=opened,
        available_at_utc=opened + timedelta(days=1),
        ingested_at_utc=opened + timedelta(days=1),
        provider="fixture",
        provenance=Provenance.BACKFILL,
        session=Session.REGULAR,
        open=price,
        high=price,
        low=price,
        close=price,
        volume=1_000_000,
    )


def window(*bars: Bar, as_of: datetime | None = None) -> BarWindow:
    latest = max(b.available_at_utc for b in bars) if bars else datetime(2026, 4, 1, tzinfo=UTC)
    return BarWindow(
        as_of=as_of or latest,
        resolution=Resolution.DAILY,
        _by_uid={UID: tuple(bars)},
    )


def closes(count: int, *, step: str = "1") -> tuple[Bar, ...]:
    return tuple(bar(i, str(Decimal("100") + Decimal(step) * i)) for i in range(count))


# --------------------------------------------------------------------------
# Absence
# --------------------------------------------------------------------------


def test_a_short_window_yields_unknown_not_a_shorter_average() -> None:
    """The single most important property here.

    A 50-bar mean over 10 bars is not a 10-bar mean, it is a wrong number with
    a confident name. Padding or forward-filling would produce exactly that.
    """
    pipeline = FeaturePipeline(specs=(make_spec("sma", 50),))
    snapshot = pipeline.compute(window(*closes(10)), UID)
    assert snapshot.get("sma_50") is UNKNOWN
    assert not snapshot.complete
    assert snapshot.unknown_features() == ("sma_50",)


def test_an_unknown_feature_raises_on_arithmetic_rather_than_becoming_zero() -> None:
    """UNKNOWN is hostile on purpose. `or 0` must not work on it."""
    pipeline = FeaturePipeline(specs=(make_spec("sma", 50),))
    value = pipeline.compute(window(*closes(3)), UID).get("sma_50")
    # The specific refusal, not a blind Exception: this asserts the sentinel's
    # own guard fired rather than some unrelated TypeError.
    with pytest.raises(UnknownValueError, match="UNKNOWN"):
        _ = value + Decimal(1)
    with pytest.raises(UnknownValueError):
        bool(value)
    # Both directions: `Decimal(1) + UNKNOWN` must refuse too, or a feature
    # could slip into an accumulator from the right-hand side.
    with pytest.raises(UnknownValueError):
        _ = Decimal(1) + value


def test_an_empty_window_makes_every_feature_unknown() -> None:
    pipeline = default_pipeline()
    snapshot = pipeline.compute(window(), UID)
    assert snapshot.n_bars_seen == 0
    assert set(snapshot.unknown_features()) == set(pipeline.names)


def test_zero_dispersion_yields_unknown_rather_than_a_huge_zscore() -> None:
    """A flat window has no scale, so there is nothing to measure against.

    Dividing anyway would produce a very large number that reads as a strong
    signal while actually meaning "nothing has moved".
    """
    flat = tuple(bar(i, "100.00") for i in range(20))
    pipeline = FeaturePipeline(specs=(make_spec("zscore", 20),))
    assert pipeline.compute(window(*flat), UID).get("zscore_20") is UNKNOWN


def test_a_non_positive_price_yields_unknown_rather_than_dividing_by_it() -> None:
    """Tested at the function, because `Bar` makes it unreachable above.

    `Bar.__post_init__` already refuses a non-positive low, so a zero close
    cannot arrive through the pipeline. The guard in the library functions is
    therefore belt to that brace — worth keeping, since these are pure
    functions over a bare sequence that the DSL and M7's trainer will also
    call, and `inf` propagating into a signal reads as a very large number
    rather than as an error.
    """
    with pytest.raises(DataError, match="non-positive price"):
        bar(0, "0.00")

    for name in ("return_pct", "stdev_pct"):
        assert FEATURE_LIBRARY[name]((Decimal(0), Decimal(100), Decimal(110))) is UNKNOWN


# --------------------------------------------------------------------------
# The library
# --------------------------------------------------------------------------


def test_the_features_compute_what_they_say() -> None:
    bars = closes(5)  # 100, 101, 102, 103, 104
    pipeline = FeaturePipeline(
        specs=(
            make_spec("last", 1, name="close"),
            make_spec("sma", 5),
            make_spec("return_pct", 5),
            make_spec("max_drawdown_pct", 5),
        )
    )
    snapshot = pipeline.compute(window(*bars), UID)
    assert snapshot.get("close") == Decimal("104")
    assert snapshot.get("sma_5") == Decimal("102")
    # 100 -> 104 is +4%.
    assert snapshot.get("return_pct_5") == Decimal("4")
    # Monotonically rising, so no drawdown.
    assert snapshot.get("max_drawdown_pct_5") == Decimal("0")


def test_max_drawdown_finds_the_worst_peak_to_trough() -> None:
    bars = (bar(0, "100"), bar(1, "120"), bar(2, "90"), bar(3, "110"))
    pipeline = FeaturePipeline(specs=(make_spec("max_drawdown_pct", 4),))
    # Peak 120 down to 90 is a 25% fall.
    assert pipeline.compute(window(*bars), UID).get("max_drawdown_pct_4") == Decimal("25")


def test_a_feature_kind_outside_the_library_is_refused() -> None:
    """Features are selected from a fixed table, never supplied as code.

    This is what makes "no eval of generated output" implementable: an M6
    searcher proposes a *name*, and an unknown name is a malformed spec.
    """
    with pytest.raises(FeatureError, match="unknown feature kind"):
        make_spec("__import__", 5)
    with pytest.raises(FeatureError, match="unknown feature kind"):
        make_spec("os.system", 5)


def test_every_library_feature_survives_a_window_of_ones() -> None:
    """A degenerate but legal input must not raise from any library function."""
    values = (Decimal(1),) * 30
    for name, fn in FEATURE_LIBRARY.items():
        result = fn(values)
        assert result is UNKNOWN or isinstance(result, Decimal), name


@pytest.mark.parametrize("lookback", [0, -1])
def test_a_non_positive_lookback_is_a_bug_not_a_value(lookback: int) -> None:
    with pytest.raises(FeatureError, match="lookback must be at least 1"):
        FeatureSpec(name="x", lookback=lookback, compute=FEATURE_LIBRARY["sma"])


def test_duplicate_feature_names_are_refused() -> None:
    """Two features under one name make a snapshot hash meaningless."""
    with pytest.raises(FeatureError, match="duplicate feature name"):
        FeaturePipeline(specs=(make_spec("sma", 10), make_spec("sma", 10)))


def test_asking_for_a_feature_the_pipeline_does_not_compute_raises() -> None:
    snapshot = default_pipeline().compute(window(*closes(60)), UID)
    with pytest.raises(FeatureError, match="no feature named"):
        snapshot.get("rsi_14")


# --------------------------------------------------------------------------
# Corporate actions
# --------------------------------------------------------------------------


def split(effective: date, *, known: datetime, ratio: Fraction) -> CorporateAction:
    return CorporateAction(
        action_id=f"act_{effective}_{ratio.numerator}_{ratio.denominator}",
        instrument_uid=UID,
        action_type=ActionType.SPLIT,
        effective_date=effective,
        known_at_utc=known,
        ratio_num=ratio.numerator,
        ratio_den=ratio.denominator,
        source_provider="fixture",
    )


def test_a_split_inside_the_window_is_not_a_return() -> None:
    """Why features run on split_adjusted rather than raw.

    A 2-for-1 halves the quoted price overnight. On the raw series that is a
    -50% return, and a mean-reversion feature would read it as the strongest
    buy signal it has ever seen.
    """
    pre = (bar(0, "200"), bar(1, "200"))
    post = (bar(2, "100"), bar(3, "100"))
    action = split(date(2026, 3, 4), known=datetime(2026, 3, 1, tzinfo=UTC), ratio=Fraction(2, 1))

    pipeline = FeaturePipeline(specs=(make_spec("return_pct", 4),))
    raw = FeaturePipeline(specs=(make_spec("return_pct", 4),), series=Series.RAW)

    view = window(*pre, *post)
    adjusted = pipeline.compute(view, UID, actions=[action]).get("return_pct_4")
    unadjusted = raw.compute(view, UID, actions=[action]).get("return_pct_4")

    assert unadjusted == Decimal("-50"), "the raw series should show the split as a crash"
    assert adjusted == Decimal("0"), "the adjusted series should show no move at all"


def test_a_split_not_yet_announced_does_not_adjust_anything() -> None:
    """The as-of filter, which `price_factor` owns rather than the caller.

    Adjusting for a split that had not been announced at the decision time
    means future corporate-action information has leaked past every schema
    gate, and the resulting series looks perfectly ordinary.
    """
    bars = (bar(0, "200"), bar(1, "200"), bar(2, "100"), bar(3, "100"))
    # Announced AFTER the decision time.
    action = split(date(2026, 3, 4), known=datetime(2026, 4, 1, tzinfo=UTC), ratio=Fraction(2, 1))
    view = window(*bars, as_of=datetime(2026, 3, 6, tzinfo=UTC))

    pipeline = FeaturePipeline(specs=(make_spec("return_pct", 4),))
    value = pipeline.compute(view, UID, actions=[action]).get("return_pct_4")
    assert value == Decimal("-50"), (
        "a split knowable only later must not adjust a decision-time series"
    )


def test_a_reverse_split_uses_an_exact_fraction() -> None:
    """1-for-3 is Fraction(1, 3), never 0.333...

    A float round trip would be hashed faithfully into every snapshot, so two
    runs over identical data could disagree on the hash.
    """
    bars = (bar(0, "10"), bar(1, "10"), bar(2, "30"), bar(3, "30"))
    action = split(date(2026, 3, 4), known=datetime(2026, 3, 1, tzinfo=UTC), ratio=Fraction(1, 3))
    pipeline = FeaturePipeline(specs=(make_spec("return_pct", 4),))
    value = pipeline.compute(window(*bars), UID, actions=[action]).get("return_pct_4")
    assert value == Decimal("0")


# --------------------------------------------------------------------------
# Inferred splits: recorded, never applied, and never silently ignored
# --------------------------------------------------------------------------


def inferred_split(effective: date, *, known: datetime, ratio: Fraction) -> CorporateAction:
    """What the residual detector records for an unexplained overnight jump.

    `source_provider` is the stored discriminator — `inferred_from_price_jump`
    has no column — so the constant is used here rather than a literal, which
    keeps this test honest about what production actually reads.
    """
    return CorporateAction(
        action_id=f"act_inferred_{effective}",
        instrument_uid=UID,
        action_type=ActionType.SPLIT,
        effective_date=effective,
        known_at_utc=known,
        ratio_num=ratio.numerator,
        ratio_den=ratio.denominator,
        source_provider=RESIDUAL_DETECTOR,
        inferred_from_price_jump=True,
    )


def test_a_guessed_ratio_never_adjusts_a_price() -> None:
    """A feature is UNKNOWN across a suspected split, not a number.

    The residual detector guesses a ratio from an unexplained gap. Applying it
    would turn a real 75% loss into a flat series if the guess is wrong; *not*
    applying it and returning a number anyway reports a -75% return as genuine
    price action. Both are answers the strategy would act on, so the only
    honest one is to refuse — which is what `UNKNOWN` is for, and it raises on
    arithmetic rather than reading as zero.
    """
    bars = (bar(0, "200"), bar(1, "200"), bar(2, "50"), bar(3, "50"))
    action = inferred_split(
        date(2026, 3, 4), known=datetime(2026, 3, 1, tzinfo=UTC), ratio=Fraction(4, 1)
    )
    pipeline = FeaturePipeline(specs=(make_spec("return_pct", 4),))
    snapshot = pipeline.compute(window(*bars), UID, actions=[action])

    assert snapshot.get("return_pct_4") is UNKNOWN
    assert not snapshot.complete
    # And the reason is on the snapshot, so "why is this unknown" does not
    # require re-deriving the factor at the call site.
    assert snapshot.unadjustable_actions == (action.action_id,)


def test_a_confirmed_split_still_adjusts_normally() -> None:
    """The vacuity check: the refusal is about the inference, not about splits.

    Without this, a bug that refused *every* split would pass the test above
    while making the whole adjusted series unusable.
    """
    bars = (bar(0, "200"), bar(1, "200"), bar(2, "50"), bar(3, "50"))
    action = split(date(2026, 3, 4), known=datetime(2026, 3, 1, tzinfo=UTC), ratio=Fraction(4, 1))
    pipeline = FeaturePipeline(specs=(make_spec("return_pct", 4),))
    snapshot = pipeline.compute(window(*bars), UID, actions=[action])

    assert snapshot.get("return_pct_4") == Decimal("0")
    assert snapshot.complete
    assert snapshot.unadjustable_actions == ()


def test_only_the_lookbacks_that_cross_the_jump_are_refused() -> None:
    """Per-spec, not per-window. A short lookback after it is unaffected.

    Refusing the whole snapshot would blind the strategy to a symbol for as
    long as the suspicion stands — including for the exit decision, where
    having no features is worse than the data problem itself.
    """
    bars = (bar(0, "200"), bar(1, "200"), bar(2, "50"), bar(3, "50"))
    action = inferred_split(
        date(2026, 3, 4), known=datetime(2026, 3, 1, tzinfo=UTC), ratio=Fraction(4, 1)
    )
    pipeline = FeaturePipeline(
        specs=(make_spec("return_pct", 4), make_spec("return_pct", 2, name="recent"))
    )
    snapshot = pipeline.compute(window(*bars), UID, actions=[action])

    assert snapshot.get("return_pct_4") is UNKNOWN, "the span across the jump is not comparable"
    assert snapshot.get("recent") == Decimal("0"), "both bars post-jump: nothing to adjust"


def test_the_raw_series_is_untouched_by_an_inferred_split() -> None:
    """`RAW` is what the broker and the stop see, and it is already right.

    It is the quoted price, adjusted by nothing by definition, so an inferred
    action cannot make it less trustworthy — and refusing it would remove the
    one series execution is allowed to use.
    """
    bars = (bar(0, "200"), bar(1, "200"), bar(2, "50"), bar(3, "50"))
    action = inferred_split(
        date(2026, 3, 4), known=datetime(2026, 3, 1, tzinfo=UTC), ratio=Fraction(4, 1)
    )
    pipeline = FeaturePipeline(specs=(make_spec("last", 1),), series=Series.RAW)
    snapshot = pipeline.compute(window(*bars), UID, actions=[action])
    assert snapshot.get("last_1") == Decimal("50")
    assert snapshot.complete


# --------------------------------------------------------------------------
# Snapshot hashing
# --------------------------------------------------------------------------


def test_identical_inputs_hash_identically() -> None:
    """The property that makes a snapshot hash worth storing."""
    pipeline = default_pipeline()
    bars = closes(60)
    first = pipeline.compute(window(*bars), UID)
    second = pipeline.compute(window(*bars), UID)
    assert first.snapshot_hash == second.snapshot_hash


def test_a_different_close_changes_the_hash() -> None:
    pipeline = default_pipeline()
    base = list(closes(60))
    changed = [*base[:-1], bar(base[-1].bar_open_utc.day, "999.00")]
    assert (
        pipeline.compute(window(*base), UID).snapshot_hash
        != pipeline.compute(window(*changed), UID).snapshot_hash
    )


def test_absent_and_omitted_are_different_facts_in_the_hash() -> None:
    """UNKNOWN hashes as a marker, not as null.

    A feature that was absent and a feature that was never requested are
    different, and a hash conflating them would match across two genuinely
    different decisions.
    """
    short = FeaturePipeline(specs=(make_spec("sma", 10), make_spec("sma", 50)))
    only_one = FeaturePipeline(specs=(make_spec("sma", 10),))
    bars = closes(20)  # enough for sma_10, not for sma_50
    with_unknown = short.compute(window(*bars), UID)
    without = only_one.compute(window(*bars), UID)
    assert with_unknown.get("sma_50") is UNKNOWN
    assert with_unknown.snapshot_hash != without.snapshot_hash


def test_the_series_is_part_of_the_hash() -> None:
    """A raw and an adjusted snapshot of the same bars are different evidence."""
    bars = closes(20)
    adjusted = FeaturePipeline(specs=(make_spec("sma", 10),))
    raw = FeaturePipeline(specs=(make_spec("sma", 10),), series=Series.RAW)
    assert (
        adjusted.compute(window(*bars), UID).snapshot_hash
        != raw.compute(window(*bars), UID).snapshot_hash
    )


def test_a_snapshot_records_which_series_it_is_on() -> None:
    """So a consumer cannot mistake an adjusted price for an executable one."""
    snapshot = default_pipeline().compute(window(*closes(20)), UID)
    assert snapshot.series is Series.SPLIT_ADJUSTED


# --------------------------------------------------------------------------
# One pipeline
# --------------------------------------------------------------------------


def test_compute_all_covers_every_instrument_in_the_window() -> None:
    other = "isin:US5949181045"
    view = BarWindow(
        as_of=datetime(2026, 4, 1, tzinfo=UTC),
        resolution=Resolution.DAILY,
        _by_uid={UID: closes(20), other: ()},
    )
    snapshots = default_pipeline().compute_all(view)
    assert set(snapshots) == {UID, other}
    assert snapshots[other].n_bars_seen == 0


def test_the_pipeline_holds_no_state_between_calls() -> None:
    """Frozen and I/O-free, so it cannot behave differently per caller.

    That is the property making "identical in backtest and live" checkable
    rather than aspirational: there is nothing for one caller to configure
    differently.
    """
    pipeline = default_pipeline()
    bars = closes(60)
    first = pipeline.compute(window(*bars), UID)
    _ = pipeline.compute(window(*closes(5)), UID)
    again = pipeline.compute(window(*bars), UID)
    assert first.snapshot_hash == again.snapshot_hash


def test_max_lookback_reports_what_the_pipeline_needs() -> None:
    """The engine uses this to size the reader's lookback window."""
    assert default_pipeline().max_lookback == 50
    assert FeaturePipeline(specs=(make_spec("sma", 7),)).max_lookback == 7
