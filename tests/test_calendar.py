"""Sessions, and the one predicate that gates entries.

The two tests worth reading first are
`test_an_unknown_date_is_closed_not_open` and
`test_exits_are_allowed_right_up_to_the_bell`. They are the two halves of the
same asymmetry: the calendar fails closed about *taking on* risk and stays open
about *shedding* it. A calendar problem must never become an unhedged position.
"""

from __future__ import annotations

from datetime import UTC, date, datetime, timedelta

import pytest

from tb.data.calendar import (
    CALENDAR_THROUGH,
    NO_ENTRY_BEFORE_CLOSE,
    CalendarError,
    DayKind,
    TradingCalendar,
    broker_days_from_exchanges,
)
from tb.data.provider import US_EASTERN, Session


@pytest.fixture
def cal() -> TradingCalendar:
    return TradingCalendar()


def eastern(day: date, hour: int, minute: int = 0) -> datetime:
    return datetime(day.year, day.month, day.day, hour, minute, tzinfo=US_EASTERN).astimezone(UTC)


# --------------------------------------------------------------------------
# Classification
# --------------------------------------------------------------------------


def test_an_ordinary_weekday_is_a_full_session(cal: TradingCalendar) -> None:
    day = cal.classify(date(2026, 3, 4))
    assert day.kind is DayKind.REGULAR
    assert day.expected_minute_bars == 390


def test_a_half_day_is_shorter_and_says_so(cal: TradingCalendar) -> None:
    """The distinction that keeps the audit honest.

    Without it, every early close reads as 180 missing minutes and the
    unexplained-gap percentage is dominated by days when nothing was wrong.
    """
    day = cal.classify(date(2026, 11, 27))
    assert day.kind is DayKind.HALF_DAY
    assert day.expected_minute_bars == 210


def test_weekends_and_holidays_are_not_trading_days(cal: TradingCalendar) -> None:
    assert cal.classify(date(2026, 3, 7)).kind is DayKind.WEEKEND
    assert cal.classify(date(2026, 12, 25)).kind is DayKind.HOLIDAY
    assert not cal.classify(date(2026, 12, 25)).is_trading_day


def test_an_unknown_date_is_closed_not_open(cal: TradingCalendar) -> None:
    """Fail-closed, and the bound is explicit.

    The holiday list is hand-maintained and will run out. Treating 2031 as a
    normal trading year because nobody extended it would send orders into
    markets that are shut; on Trading 212 those queue to the next open and fill
    at a price nothing in the decision was based on.
    """
    beyond = CALENDAR_THROUGH + timedelta(days=1)
    day = cal.classify(beyond)
    assert day.kind is DayKind.UNKNOWN
    assert not day.is_trading_day
    assert day.open_utc is None


def test_session_bounds_track_eastern_dst_not_utc(cal: TradingCalendar) -> None:
    """09:30 Eastern is 14:30 UTC in winter and 13:30 in summer.

    A hardcoded UTC offset is right for four months of the year. This is the
    same reason `classify_us_session` compares in Eastern.
    """
    winter = cal.classify(date(2026, 3, 4))
    summer = cal.classify(date(2026, 7, 1))
    assert winter.open_utc is not None and summer.open_utc is not None
    assert winter.open_utc.hour == 14
    assert summer.open_utc.hour == 13


def test_sessions_between_counts_trading_days_only(cal: TradingCalendar) -> None:
    """The denominator for a coverage check.

    Comparing bars held against calendar days would report every weekend as
    missing data, and the real gaps would be lost in the noise.
    """
    days = cal.sessions_between(date(2026, 3, 2), date(2026, 3, 8))
    assert len(days) == 5
    assert all(day.is_trading_day for day in days)


def test_a_backwards_range_is_an_error(cal: TradingCalendar) -> None:
    with pytest.raises(CalendarError, match="runs backwards"):
        cal.sessions_between(date(2026, 3, 8), date(2026, 3, 2))


def test_previous_session_skips_weekends_and_holidays(cal: TradingCalendar) -> None:
    """ "Yesterday" is wrong across a weekend, and wrong in a way that matters.

    The residual split detector compares `prev_close` to `next_open`; pointing
    `prev_close` at a day the market was shut changes which gap looks
    unexplained.
    """
    monday = cal.previous_session(date(2026, 3, 9))
    assert monday is not None
    assert monday.day == date(2026, 3, 6)

    after_christmas = cal.previous_session(date(2026, 12, 28))
    assert after_christmas is not None
    assert after_christmas.day == date(2026, 12, 24)


def test_next_session_skips_a_holiday(cal: TradingCalendar) -> None:
    nxt = cal.next_session(date(2026, 12, 24))
    assert nxt is not None
    assert nxt.day == date(2026, 12, 28)


# --------------------------------------------------------------------------
# may_enter_now
# --------------------------------------------------------------------------


def test_mid_session_entries_are_allowed(cal: TradingCalendar) -> None:
    verdict = cal.may_enter_now(Session.REGULAR, eastern(date(2026, 3, 4), 11))
    assert verdict.allowed
    assert verdict.detail == "within the regular session"


def test_no_entry_in_the_closing_minutes(cal: TradingCalendar) -> None:
    """An entry filled at 15:59 has no time for a stop behind it.

    The position then carries the full overnight gap unprotected, which is
    precisely the exposure the sizing rule is built to bound — so the rule that
    bounds it must not be undermined by the clock.
    """
    verdict = cal.may_enter_now(Session.REGULAR, eastern(date(2026, 3, 4), 15, 50))
    assert not verdict.allowed
    assert any("protective stop" in reason for reason in verdict.reasons)


def test_the_closing_window_follows_an_early_close(cal: TradingCalendar) -> None:
    """12:55 on a half-day is the closing window, not mid-session.

    A fixed 15:45 cutoff would authorise an entry five minutes before a 13:00
    bell. The window has to be measured from the day's actual close.
    """
    half_day = date(2026, 11, 27)
    blocked = cal.may_enter_now(Session.REGULAR, eastern(half_day, 12, 55))
    assert not blocked.allowed
    assert any("half_day close" in reason for reason in blocked.reasons)

    # The same clock time on a full session is fine.
    assert cal.may_enter_now(Session.REGULAR, eastern(date(2026, 11, 30), 12, 55)).allowed


def test_no_entry_in_the_opening_minutes(cal: TradingCalendar) -> None:
    verdict = cal.may_enter_now(Session.REGULAR, eastern(date(2026, 3, 4), 9, 32))
    assert not verdict.allowed
    assert any("opening auction" in reason for reason in verdict.reasons)


def test_an_extended_hours_bar_cannot_authorise_an_entry(cal: TradingCalendar) -> None:
    """The bar's session, not the current one.

    An extended-hours print on an earnings night can be hundreds of basis
    points from the regular-session price. A decision computed on one must not
    become an order in the other.
    """
    verdict = cal.may_enter_now(Session.EXTENDED, eastern(date(2026, 3, 4), 11))
    assert not verdict.allowed
    assert any("extended-session print" in reason for reason in verdict.reasons)


def test_an_unknown_session_is_also_refused(cal: TradingCalendar) -> None:
    verdict = cal.may_enter_now(Session.UNKNOWN, eastern(date(2026, 3, 4), 11))
    assert not verdict.allowed


def test_every_blocking_reason_is_reported_at_once(cal: TradingCalendar) -> None:
    """ "Why is the bot not trading" must be answerable without a debugger.

    A verdict that stops at the first failure sends whoever is asking round the
    loop once per cause.
    """
    verdict = cal.may_enter_now(Session.EXTENDED, eastern(date(2026, 3, 7), 11))
    assert not verdict.allowed
    assert len(verdict.reasons) == 2


def test_a_naive_now_is_refused(cal: TradingCalendar) -> None:
    with pytest.raises(CalendarError, match="timezone-aware"):
        cal.may_enter_now(Session.REGULAR, datetime(2026, 3, 4, 15))  # noqa: DTZ001


def test_an_out_of_range_date_names_the_remedy(cal: TradingCalendar) -> None:
    verdict = cal.may_enter_now(
        Session.REGULAR, eastern(CALENDAR_THROUGH + timedelta(days=400), 11)
    )
    assert not verdict.allowed
    assert any("US_HOLIDAYS" in reason for reason in verdict.reasons)


# --------------------------------------------------------------------------
# may_exit_now — deliberately asymmetric
# --------------------------------------------------------------------------


def test_exits_are_allowed_right_up_to_the_bell(cal: TradingCalendar) -> None:
    """Every entry rule above exists to stop taking on risk.

    None of them is a reason to hold a position the bot has decided to close.
    Refusing an exit converts a data or calendar problem into an unhedged
    position, which is strictly worse than the problem itself.
    """
    last_minute = eastern(date(2026, 3, 4), 15, 59)
    assert not cal.may_enter_now(Session.REGULAR, last_minute).allowed
    assert cal.may_exit_now(last_minute).allowed


def test_exits_are_allowed_in_the_opening_minutes(cal: TradingCalendar) -> None:
    opening = eastern(date(2026, 3, 4), 9, 31)
    assert not cal.may_enter_now(Session.REGULAR, opening).allowed
    assert cal.may_exit_now(opening).allowed


def test_an_exit_still_needs_an_open_market(cal: TradingCalendar) -> None:
    verdict = cal.may_exit_now(eastern(date(2026, 3, 7), 11))
    assert not verdict.allowed
    assert "not open" in verdict.detail


# --------------------------------------------------------------------------
# The broker's own schedule
# --------------------------------------------------------------------------


class _Event:
    def __init__(self, moment: str, kind: str) -> None:
        self.date = moment
        self.event_type = kind


class _Schedule:
    def __init__(self, events: list[_Event]) -> None:
        self.time_events = events


class _Exchange:
    def __init__(self, exchange_id: int, schedules: list[_Schedule]) -> None:
        self.exchange_id = exchange_id
        self.working_schedules = schedules


def test_the_brokers_schedule_overrides_the_builtin_lists() -> None:
    """Trading 212 fills the orders, so its opinion is the one that counts.

    A third-party calendar disagreeing with the venue is a warning about the
    mapping, not grounds to override the venue.
    """
    days = broker_days_from_exchanges(
        [
            _Exchange(
                1,
                [
                    _Schedule(
                        [
                            # A Saturday session, which the built-in list would
                            # call a weekend.
                            _Event("2026-03-07T14:30:00Z", "OPEN"),
                            _Event("2026-03-07T21:00:00Z", "CLOSE"),
                        ]
                    )
                ],
            )
        ]
    )
    cal = TradingCalendar(broker_days=days)
    assert cal.uses_broker_schedule
    assert cal.classify(date(2026, 3, 7)).kind is DayKind.REGULAR
    assert cal.may_enter_now(Session.REGULAR, eastern(date(2026, 3, 7), 11)).allowed


def test_a_short_broker_session_is_recognised_as_a_half_day() -> None:
    days = broker_days_from_exchanges(
        [
            _Exchange(
                1,
                [
                    _Schedule(
                        [
                            _Event("2026-03-04T14:30:00Z", "OPEN"),
                            _Event("2026-03-04T18:00:00Z", "CLOSE"),
                        ]
                    )
                ],
            )
        ]
    )
    assert TradingCalendar(broker_days=days).classify(date(2026, 3, 4)).kind is DayKind.HALF_DAY


def test_unpaired_or_unparseable_events_are_skipped_not_guessed() -> None:
    """A fabricated close time is worse than no schedule.

    With one, the calendar would confidently authorise entries in a window
    nobody verified.
    """
    days = broker_days_from_exchanges(
        [
            _Exchange(
                1,
                [
                    _Schedule(
                        [
                            _Event("2026-03-04T14:30:00Z", "OPEN"),
                            # No close for this day.
                            _Event("not-a-timestamp", "CLOSE"),
                            _Event("2026-03-05T14:30:00Z", "OPEN"),
                            _Event("2026-03-05T21:00:00Z", "CLOSE"),
                        ]
                    )
                ],
            )
        ]
    )
    assert set(days) == {date(2026, 3, 5)}


def test_a_broker_schedule_can_be_filtered_to_one_exchange() -> None:
    exchanges = [
        _Exchange(
            1,
            [
                _Schedule(
                    [
                        _Event("2026-03-04T14:30:00Z", "OPEN"),
                        _Event("2026-03-04T21:00:00Z", "CLOSE"),
                    ]
                )
            ],
        ),
        _Exchange(
            2,
            [
                _Schedule(
                    [
                        _Event("2026-03-05T08:00:00Z", "OPEN"),
                        _Event("2026-03-05T16:30:00Z", "CLOSE"),
                    ]
                )
            ],
        ),
    ]
    assert set(broker_days_from_exchanges(exchanges, exchange_id=2)) == {date(2026, 3, 5)}


def test_the_entry_windows_are_the_documented_widths() -> None:
    """Pinned so a tuning edit is a visible change rather than a silent one."""
    assert NO_ENTRY_BEFORE_CLOSE.total_seconds() == 15 * 60
