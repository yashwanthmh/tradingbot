"""Factor algebra: the properties, not the cases.

Two of these are worth more than the rest.

`test_dollar_volume_is_invariant_across_a_split` is what stops M3's slippage
model inheriting a 4x liquidity error on every name that has ever split, and it
only passes because the volume factor is the *exact* inverse of the price
factor rather than an approximation of it.

`test_an_unannounced_split_does_not_leak_backwards` is the lookahead channel
that no schema check catches: adjust a price for a split whose announcement had
not happened yet and the pre-ex-date series silently contains information from
the future, while every number in it still looks perfectly ordinary.
"""

from __future__ import annotations

from datetime import UTC, date, datetime
from decimal import Decimal
from fractions import Fraction

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from tb.data.adjustments import (
    ActionType,
    CorporateAction,
    FactorError,
    Series,
    SeriesMisuseError,
    apply_factor,
    apply_volume_factor,
    detect_unexplained_split,
    effective_after,
    effective_in,
    factor_for,
    known_by,
    latest_vintages,
    price_factor,
    require_raw,
    total_return_factor,
    volume_factor,
)
from tb.data.provider import DataError

AAPL = "isin:US0378331005"
NVDA = "isin:US67066G1040"

# Real histories, so the arithmetic is checked against events that happened
# rather than against numbers chosen to make it pass.
AAPL_4_FOR_1 = CorporateAction(
    action_id="act_aapl_2020",
    instrument_uid=AAPL,
    action_type=ActionType.SPLIT,
    effective_date=date(2020, 8, 31),
    known_at_utc=datetime(2020, 7, 31, tzinfo=UTC),
    source_provider="alpaca",
    ratio_num=4,
    ratio_den=1,
    declared_date=date(2020, 7, 30),
)
NVDA_10_FOR_1 = CorporateAction(
    action_id="act_nvda_2024",
    instrument_uid=NVDA,
    action_type=ActionType.SPLIT,
    effective_date=date(2024, 6, 10),
    known_at_utc=datetime(2024, 5, 22, tzinfo=UTC),
    source_provider="alpaca",
    ratio_num=10,
    ratio_den=1,
)
AAPL_7_FOR_1 = CorporateAction(
    action_id="act_aapl_2014",
    instrument_uid=AAPL,
    action_type=ActionType.SPLIT,
    effective_date=date(2014, 6, 9),
    known_at_utc=datetime(2014, 4, 23, tzinfo=UTC),
    source_provider="alpaca",
    ratio_num=7,
    ratio_den=1,
)
AAPL_DIVIDEND = CorporateAction(
    action_id="act_aapl_div",
    instrument_uid=AAPL,
    action_type=ActionType.CASH_DIVIDEND,
    effective_date=date(2021, 2, 5),
    known_at_utc=datetime(2021, 1, 27, tzinfo=UTC),
    source_provider="alpaca",
    gross_amount=Decimal("0.205"),
    currency="USD",
)

NOW = datetime(2026, 1, 1, tzinfo=UTC)


# --------------------------------------------------------------------------
# The two filters, which must never be conflated
# --------------------------------------------------------------------------


def test_as_of_visibility_hides_an_unannounced_action() -> None:
    before = datetime(2020, 7, 1, tzinfo=UTC)
    assert known_by([AAPL_4_FOR_1], before) == ()
    assert known_by([AAPL_4_FOR_1], NOW) == (AAPL_4_FOR_1,)


def test_the_effective_filter_is_strict_on_the_ex_date() -> None:
    """A bar dated on the ex-date is already quoted post-split.

    Adjusting it would apply the ratio twice, which shows up as a one-day
    250% gap in the adjusted series — visible if anyone looks, and nobody does.
    """
    assert effective_after([AAPL_4_FOR_1], date(2020, 8, 30)) == (AAPL_4_FOR_1,)
    assert effective_after([AAPL_4_FOR_1], date(2020, 8, 31)) == ()


def test_the_span_filter_is_half_open() -> None:
    assert effective_in([AAPL_4_FOR_1], after=date(2020, 8, 30), through=date(2020, 8, 31)) == (
        AAPL_4_FOR_1,
    )
    assert effective_in([AAPL_4_FOR_1], after=date(2020, 8, 31), through=date(2020, 9, 30)) == ()


def test_a_backwards_span_is_an_error_not_an_empty_result() -> None:
    with pytest.raises(FactorError, match="runs backwards"):
        effective_in([AAPL_4_FOR_1], after=date(2020, 9, 1), through=date(2020, 8, 1))


def test_a_naive_as_of_is_refused() -> None:
    with pytest.raises(DataError, match="timezone-aware"):
        known_by([AAPL_4_FOR_1], datetime(2020, 9, 1))  # noqa: DTZ001


# --------------------------------------------------------------------------
# Price and volume factors
# --------------------------------------------------------------------------


def test_a_four_for_one_split_quarters_a_prior_price() -> None:
    factor = price_factor([AAPL_4_FOR_1], at=date(2020, 8, 28), as_of=NOW)
    assert factor.value == Fraction(1, 4)
    adjusted = apply_factor(Decimal("499.23"), factor, as_of=NOW, places=6)
    assert adjusted.value == Decimal("124.807500")
    assert adjusted.series is Series.SPLIT_ADJUSTED


def test_factors_compose_exactly_across_two_splits() -> None:
    """7-for-1 in 2014 then 4-for-1 in 2020: a 1:28 total.

    Exact because the ratios are Fractions. Done in float, twenty years of
    products drift in the last digits, and `canonical_json` hashes the drift
    into every vintage that touches the series.
    """
    factor = price_factor([AAPL_7_FOR_1, AAPL_4_FOR_1], at=date(2014, 1, 2), as_of=NOW)
    assert factor.value == Fraction(1, 28)
    assert factor.n_actions == 2


def test_a_reverse_split_multiplies_rather_than_divides() -> None:
    reverse = CorporateAction(
        action_id="act_reverse",
        instrument_uid=AAPL,
        action_type=ActionType.SPLIT,
        effective_date=date(2021, 1, 5),
        known_at_utc=datetime(2020, 12, 1, tzinfo=UTC),
        source_provider="alpaca",
        ratio_num=1,
        ratio_den=10,
    )
    factor = price_factor([reverse], at=date(2020, 12, 31), as_of=NOW)
    # One old share becomes a tenth of a share, so the price multiplies by ten.
    assert factor.value == Fraction(10, 1)


def test_the_volume_factor_is_the_exact_inverse() -> None:
    at, actions = date(2014, 1, 2), [AAPL_7_FOR_1, AAPL_4_FOR_1]
    prices = price_factor(actions, at=at, as_of=NOW)
    volumes = volume_factor(actions, at=at, as_of=NOW)
    assert prices.value * volumes.value == 1
    assert volumes.value == Fraction(28, 1)


@given(
    numerators=st.lists(st.integers(min_value=1, max_value=50), min_size=1, max_size=5),
    denominators=st.lists(st.integers(min_value=1, max_value=50), min_size=1, max_size=5),
    price_cents=st.integers(min_value=1, max_value=10_000_000),
    volume=st.integers(min_value=1, max_value=5_000_000_000),
)
@settings(max_examples=200, deadline=None)
def test_dollar_volume_is_invariant_across_a_split(
    numerators: list[int],
    denominators: list[int],
    price_cents: int,
    volume: int,
) -> None:
    """The property that keeps ADV from step-changing at every split.

    A split moves shares and price in exactly opposite directions, so their
    product is untouched. If this ever fails, every liquidity-derived number —
    position sizing, slippage, the participation cap — is wrong by the split
    ratio on any name that has split.
    """
    actions = [
        CorporateAction(
            action_id=f"act_{index}",
            instrument_uid=AAPL,
            action_type=ActionType.SPLIT,
            effective_date=date(2020, 1, 1 + index),
            known_at_utc=datetime(2019, 1, 1, tzinfo=UTC),
            source_provider="test",
            ratio_num=num,
            ratio_den=den,
        )
        for index, (num, den) in enumerate(zip(numerators, denominators, strict=False))
    ]
    at = date(2019, 12, 31)
    prices = price_factor(actions, at=at, as_of=NOW)
    volumes = volume_factor(actions, at=at, as_of=NOW)

    price = Fraction(price_cents, 100)
    assert (price * prices.value) * (Fraction(volume) * volumes.value) == price * volume


def test_no_relevant_action_gives_the_identity_factor() -> None:
    factor = price_factor([AAPL_4_FOR_1], at=date(2021, 1, 1), as_of=NOW)
    assert factor.is_identity
    assert factor.n_actions == 0


def test_a_dividend_does_not_move_the_split_factor() -> None:
    """Kept separate on purpose: a dividend does not change the share count.

    Folding dividends into the split factor is how a "price" series ends up
    being a total-return series, and then Sharpe is computed twice over the
    same distributions.
    """
    factor = price_factor([AAPL_4_FOR_1, AAPL_DIVIDEND], at=date(2020, 8, 28), as_of=NOW)
    assert factor.value == Fraction(1, 4)
    assert factor.n_actions == 1


def test_apply_volume_factor_returns_whole_shares() -> None:
    factor = volume_factor([AAPL_4_FOR_1], at=date(2020, 8, 28), as_of=NOW)
    assert apply_volume_factor(1_000_001, factor) == 4_000_004


# --------------------------------------------------------------------------
# Total return
# --------------------------------------------------------------------------


def test_total_return_includes_the_dividend_and_the_split() -> None:
    factor = total_return_factor(
        [AAPL_4_FOR_1, AAPL_DIVIDEND],
        at=date(2020, 8, 28),
        as_of=NOW,
        reference_close={AAPL_DIVIDEND.action_id: Decimal("136.76")},
    )
    expected = (
        Fraction(1, 4)
        * Fraction(Decimal("136.76") - Decimal("0.205"))
        / Fraction(Decimal("136.76"))
    )
    assert factor.value == expected
    assert factor.series is Series.TOTAL_RETURN
    assert factor.complete


def test_a_dividend_with_no_reference_close_is_named_not_silently_dropped() -> None:
    """The bias runs downward, which is the direction nobody investigates.

    Leaving a dividend out understates total return for exactly the names that
    pay them. `complete=False` plus the action id is what makes that visible
    instead of showing up years later as "our dividend payers underperform".
    """
    factor = total_return_factor(
        [AAPL_DIVIDEND], at=date(2020, 8, 28), as_of=NOW, reference_close={}
    )
    assert factor.is_identity
    assert factor.complete is False
    assert factor.missing == (AAPL_DIVIDEND.action_id,)


def test_a_dividend_larger_than_the_price_is_excluded_rather_than_inverting_it() -> None:
    factor = total_return_factor(
        [AAPL_DIVIDEND],
        at=date(2020, 8, 28),
        as_of=NOW,
        reference_close={AAPL_DIVIDEND.action_id: Decimal("0.10")},
    )
    assert factor.is_identity
    assert factor.missing == (AAPL_DIVIDEND.action_id,)


def test_total_return_is_never_negative_or_zero() -> None:
    with pytest.raises(FactorError, match="non-positive"):
        from tb.data.adjustments import Factor

        Factor(value=Fraction(0), series=Series.TOTAL_RETURN)


# --------------------------------------------------------------------------
# Restatements and as-of correctness
# --------------------------------------------------------------------------


def test_a_restated_ratio_does_not_rewrite_the_earlier_belief() -> None:
    """The whole reason the table is append-only.

    A vendor that corrects a ratio must not change what a backtest over an
    earlier instant sees. Otherwise re-running the same backtest next month
    produces different numbers from the same `vintage_id`, and nothing in the
    ledger explains why.
    """
    wrong = CorporateAction(
        action_id="act_wrong",
        instrument_uid=AAPL,
        action_type=ActionType.SPLIT,
        effective_date=date(2020, 8, 31),
        known_at_utc=datetime(2020, 7, 31, tzinfo=UTC),
        source_provider="yahoo",
        ratio_num=2,
        ratio_den=1,
    )
    corrected = CorporateAction(
        action_id="act_corrected",
        instrument_uid=AAPL,
        action_type=ActionType.SPLIT,
        effective_date=date(2020, 8, 31),
        known_at_utc=datetime(2020, 9, 15, tzinfo=UTC),
        source_provider="alpaca",
        ratio_num=4,
        ratio_den=1,
    )
    both = [wrong, corrected]

    before = price_factor(both, at=date(2020, 8, 28), as_of=datetime(2020, 8, 1, tzinfo=UTC))
    after = price_factor(both, at=date(2020, 8, 28), as_of=NOW)
    assert before.value == Fraction(1, 2)
    assert after.value == Fraction(1, 4)


def test_latest_vintages_keeps_one_row_per_action_identity() -> None:
    older = AAPL_4_FOR_1
    newer = CorporateAction(
        action_id="act_newer",
        instrument_uid=AAPL,
        action_type=ActionType.SPLIT,
        effective_date=date(2020, 8, 31),
        known_at_utc=datetime(2021, 1, 1, tzinfo=UTC),
        source_provider="alpaca",
        ratio_num=4,
        ratio_den=1,
    )
    assert latest_vintages([older, newer], NOW) == (newer,)
    assert latest_vintages([newer, older], NOW) == (newer,)


def test_an_unannounced_split_does_not_leak_backwards() -> None:
    """The lookahead channel no schema check catches.

    Apple's 4-for-1 was announced on 2020-07-30. A backtest deciding on
    2020-07-01 could not have known it, so the price series it sees must be
    unadjusted — and every number in an adjusted one would still look ordinary.
    """
    as_of = datetime(2020, 7, 1, tzinfo=UTC)
    assert price_factor([AAPL_4_FOR_1], at=date(2020, 6, 30), as_of=as_of).is_identity
    # The day after the announcement, it is knowable.
    known = datetime(2020, 8, 1, tzinfo=UTC)
    assert price_factor([AAPL_4_FOR_1], at=date(2020, 6, 30), as_of=known).value == Fraction(1, 4)


@given(as_of_days=st.integers(min_value=0, max_value=4000))
@settings(max_examples=200, deadline=None)
def test_a_factor_only_ever_incorporates_more_actions_as_time_passes(
    as_of_days: int,
) -> None:
    """Monotonicity in knowledge time.

    Later can only know more. If a factor ever *loses* an action as `as_of`
    advances, some filter is comparing the wrong pair of axes.
    """
    actions = [AAPL_7_FOR_1, AAPL_4_FOR_1, NVDA_10_FOR_1]
    base = datetime(2014, 1, 1, tzinfo=UTC)
    earlier = base + (datetime(2014, 1, 2, tzinfo=UTC) - base) * as_of_days
    later = earlier + (datetime(2014, 1, 2, tzinfo=UTC) - base) * 30
    at = date(2013, 1, 1)
    assert (
        price_factor(actions, at=at, as_of=earlier).n_actions
        <= price_factor(actions, at=at, as_of=later).n_actions
    )


# --------------------------------------------------------------------------
# Series discipline
# --------------------------------------------------------------------------


def test_only_the_raw_series_may_reach_execution() -> None:
    """Enforced by a raise, not by a comment.

    A split-adjusted price compared against the broker's quote disagrees by the
    entire split ratio, and the cross-venue check reads that as a mismapped
    ticker — a data problem misdiagnosed as an identity problem, which sends
    the investigation in exactly the wrong direction.
    """
    raw = apply_factor(
        Decimal("124.81"),
        factor_for(Series.RAW, [AAPL_4_FOR_1], at=date(2020, 8, 28), as_of=NOW),
        as_of=NOW,
        places=6,
    )
    assert require_raw(raw) == Decimal("124.810000")

    adjusted = apply_factor(
        Decimal("499.23"),
        factor_for(Series.SPLIT_ADJUSTED, [AAPL_4_FOR_1], at=date(2020, 8, 28), as_of=NOW),
        as_of=NOW,
        places=6,
    )
    with pytest.raises(SeriesMisuseError, match="raw series"):
        require_raw(adjusted)


def test_the_raw_series_is_always_the_identity_factor() -> None:
    factor = factor_for(Series.RAW, [AAPL_7_FOR_1, AAPL_4_FOR_1], at=date(2013, 1, 1), as_of=NOW)
    assert factor.is_identity
    assert factor.series is Series.RAW


def test_rounding_is_half_even_and_stated() -> None:
    """Pinned so two processes cannot disagree in the last digit.

    The decimal context is process-global mutable state; inheriting it would
    make the same backtest hash differently depending on what else the process
    had imported.
    """
    from tb.data.adjustments import Factor

    half = Factor(value=Fraction(1, 2), series=Series.SPLIT_ADJUSTED)
    assert apply_factor(Decimal("0.05"), half, as_of=NOW, places=2).value == Decimal("0.02")
    assert apply_factor(Decimal("0.15"), half, as_of=NOW, places=2).value == Decimal("0.08")


def test_a_float_price_is_refused() -> None:
    from tb.data.adjustments import Factor

    with pytest.raises(FactorError, match="must be Decimal"):
        apply_factor(
            499.23,  # type: ignore[arg-type]
            Factor(value=Fraction(1, 4), series=Series.SPLIT_ADJUSTED),
            as_of=NOW,
            places=6,
        )


# --------------------------------------------------------------------------
# Validation at construction
# --------------------------------------------------------------------------


def test_a_split_without_a_ratio_is_refused() -> None:
    with pytest.raises(DataError, match="needs both ratio"):
        CorporateAction(
            action_id="act_bad",
            instrument_uid=AAPL,
            action_type=ActionType.SPLIT,
            effective_date=date(2020, 8, 31),
            known_at_utc=NOW,
            source_provider="test",
        )


def test_a_dividend_without_an_amount_is_refused() -> None:
    with pytest.raises(DataError, match="needs a gross_amount"):
        CorporateAction(
            action_id="act_bad",
            instrument_uid=AAPL,
            action_type=ActionType.CASH_DIVIDEND,
            effective_date=date(2020, 8, 31),
            known_at_utc=NOW,
            source_provider="test",
        )


def test_a_naive_known_at_is_refused() -> None:
    with pytest.raises(DataError, match="naive datetime"):
        CorporateAction(
            action_id="act_bad",
            instrument_uid=AAPL,
            action_type=ActionType.DELISTING,
            effective_date=date(2020, 8, 31),
            known_at_utc=datetime(2020, 8, 31),  # noqa: DTZ001
            source_provider="test",
        )


def test_asking_a_dividend_for_its_split_ratio_is_an_error() -> None:
    with pytest.raises(FactorError, match="not a split"):
        _ = AAPL_DIVIDEND.split_ratio


# --------------------------------------------------------------------------
# The residual detector
# --------------------------------------------------------------------------


def test_an_unreported_four_for_one_is_detected() -> None:
    suspicion = detect_unexplained_split(
        instrument_uid=AAPL,
        prev_close=Decimal("499.23"),
        next_open=Decimal("127.58"),
        effective_date=date(2020, 8, 31),
    )
    assert suspicion is not None
    assert suspicion.implied_ratio == Fraction(4, 1)
    assert "no action row explains it" in suspicion.description


def test_a_reported_split_is_not_flagged() -> None:
    assert (
        detect_unexplained_split(
            instrument_uid=AAPL,
            prev_close=Decimal("499.23"),
            next_open=Decimal("127.58"),
            effective_date=date(2020, 8, 31),
            known_actions=[AAPL_4_FOR_1],
        )
        is None
    )


@pytest.mark.parametrize("move_pct", ["0.5", "3", "10", "18", "20", "25", "28"])
def test_an_ordinary_move_is_never_a_split(move_pct: str) -> None:
    """A detector that fires on ordinary sessions blocks the whole universe.

    Below the threshold no ratio is evidence of anything: with a large enough
    denominator a rational fits almost any move, which is how a detector becomes
    a random number generator.
    """
    prev = Decimal("100.00")
    assert (
        detect_unexplained_split(
            instrument_uid=AAPL,
            prev_close=prev,
            next_open=prev * (1 - Decimal(move_pct) / 100),
            effective_date=date(2024, 3, 4),
        )
        is None
    )


def test_a_large_move_that_fits_no_real_ratio_is_not_flagged() -> None:
    """A 42% crash is a crash.

    With `Fraction.limit_denominator` this came back as a convincing 19-for-12
    with a 0.02% error — which is why the search is a fixed set of ratios that
    issuers actually use, not a best-rational-approximation.
    """
    assert (
        detect_unexplained_split(
            instrument_uid=AAPL,
            prev_close=Decimal("100.00"),
            next_open=Decimal("58.00"),
            effective_date=date(2024, 3, 4),
        )
        is None
    )


def test_a_five_for_four_split_is_deliberately_not_flagged() -> None:
    """Its 20% signature collides with an ordinary earnings gap.

    Including it would block new entries across the universe every reporting
    season. The audit's unexplained-jump check reports the move instead,
    without asserting a ratio nobody can confirm.
    """
    assert (
        detect_unexplained_split(
            instrument_uid=AAPL,
            prev_close=Decimal("100.00"),
            next_open=Decimal("80.00"),
            effective_date=date(2024, 3, 4),
        )
        is None
    )


def test_a_three_for_two_split_is_flagged() -> None:
    suspicion = detect_unexplained_split(
        instrument_uid=AAPL,
        prev_close=Decimal("150.00"),
        next_open=Decimal("100.00"),
        effective_date=date(2024, 3, 4),
    )
    assert suspicion is not None
    assert suspicion.implied_ratio == Fraction(3, 2)


def test_an_unreported_reverse_split_is_detected() -> None:
    suspicion = detect_unexplained_split(
        instrument_uid=AAPL,
        prev_close=Decimal("1.02"),
        next_open=Decimal("10.15"),
        effective_date=date(2024, 3, 4),
    )
    assert suspicion is not None
    assert suspicion.implied_ratio == Fraction(1, 10)


def test_the_detector_refuses_a_non_positive_price() -> None:
    with pytest.raises(FactorError, match="non-positive price"):
        detect_unexplained_split(
            instrument_uid=AAPL,
            prev_close=Decimal("0"),
            next_open=Decimal("10"),
            effective_date=date(2024, 3, 4),
        )


@given(
    ratio_num=st.integers(min_value=2, max_value=20),
    price_cents=st.integers(min_value=500, max_value=10_000_000),
)
@settings(max_examples=150, deadline=None)
def test_every_simple_forward_split_is_caught(ratio_num: int, price_cents: int) -> None:
    """No split ratio between 2 and 20 slips past the detector."""
    prev = Decimal(price_cents) / 100
    suspicion = detect_unexplained_split(
        instrument_uid=AAPL,
        prev_close=prev,
        next_open=prev / ratio_num,
        effective_date=date(2024, 3, 4),
    )
    assert suspicion is not None
    assert suspicion.implied_ratio == Fraction(ratio_num, 1)
