"""When the market is open, and when this bot may act.

Two separate questions that look like one. "Is the exchange open" is a calendar
fact. "May the bot enter a position right now" is a policy decision that
depends on the calendar plus several things that have nothing to do with it —
whether the bar driving the decision was a regular-session print, how close the
close is, and whether today ends early. Conflating them is how a system ends up
placing a market order into the last thirty seconds of a half-day.

So there is exactly one predicate, `may_enter_now`, and it returns a reason
along with the verdict. Every entry path calls it. A second opinion computed
somewhere else would diverge, and the divergence would be in the permissive
direction, because that is the direction nobody investigates.

**Trading 212's own working schedules are the authority.** It serves no market
data, but it does publish `GET /equity/metadata/exchanges` with time events per
exchange — and since Trading 212 is the venue that fills the orders, *its*
opinion about whether a market is open is the one that matters. A third-party
calendar that disagrees is a warning about the mapping, not an override.

The fallback calendar here is the regular US schedule with the fixed holidays
and the early closes, used when the broker's schedule has not been fetched.
It is **fail-closed**: an unknown date is treated as closed rather than open.
A wrongly-closed day costs one session of opportunity; a wrongly-open day
sends orders into a market that cannot fill them, and on Trading 212 an order
placed while a market is closed queues until the open, arriving at a price
nothing in the decision was based on.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from datetime import UTC, date, datetime, time, timedelta
from enum import StrEnum

from tb.data.provider import US_EASTERN, DataError, Session

REGULAR_OPEN = time(9, 30)
REGULAR_CLOSE = time(16, 0)
HALF_DAY_CLOSE = time(13, 0)

# No entries in the last minutes of a session. Two reasons, and the second is
# the one that bites: liquidity thins into the close so the slippage estimate
# the cost gate used is no longer the slippage that will happen, and an entry
# filled at 15:59 cannot have a protective stop behind it before the bell —
# leaving the position unprotected across the whole overnight gap, which is
# exactly the exposure the sizing rule is built to bound.
NO_ENTRY_BEFORE_CLOSE = timedelta(minutes=15)

# Nor in the first minutes. The opening auction's prints are not comparable to
# the continuous session, and a feed's first minute bar is the least
# representative bar of the day — on IEX it may be a handful of trades.
NO_ENTRY_AFTER_OPEN = timedelta(minutes=5)


class CalendarError(DataError):
    """The calendar could not answer, so the answer is no."""


class DayKind(StrEnum):
    REGULAR = "regular"
    HALF_DAY = "half_day"
    WEEKEND = "weekend"
    HOLIDAY = "holiday"
    # Nothing in the calendar covers this date. Fail-closed: treated as closed.
    UNKNOWN = "unknown"

    @property
    def is_trading_day(self) -> bool:
        return self in (DayKind.REGULAR, DayKind.HALF_DAY)


# US market holidays, as observed. Hand-maintained rather than computed from
# rules, because the rules have exceptions — a holiday falling on a Saturday is
# observed on the preceding Friday, Good Friday moves every year, and a
# national day of mourning closes the market with no rule at all. A list that
# runs out is honest; a rule that is subtly wrong is not.
US_HOLIDAYS: frozenset[date] = frozenset(
    {
        # 2024
        date(2024, 1, 1), date(2024, 1, 15), date(2024, 2, 19), date(2024, 3, 29),
        date(2024, 5, 27), date(2024, 6, 19), date(2024, 7, 4), date(2024, 9, 2),
        date(2024, 11, 28), date(2024, 12, 25),
        # 2025
        date(2025, 1, 1), date(2025, 1, 9), date(2025, 1, 20), date(2025, 2, 17),
        date(2025, 4, 18), date(2025, 5, 26), date(2025, 6, 19), date(2025, 7, 4),
        date(2025, 9, 1), date(2025, 11, 27), date(2025, 12, 25),
        # 2026
        date(2026, 1, 1), date(2026, 1, 19), date(2026, 2, 16), date(2026, 4, 3),
        date(2026, 5, 25), date(2026, 6, 19), date(2026, 7, 3), date(2026, 9, 7),
        date(2026, 11, 26), date(2026, 12, 25),
        # 2027
        date(2027, 1, 1), date(2027, 1, 18), date(2027, 2, 15), date(2027, 3, 26),
        date(2027, 5, 31), date(2027, 6, 18), date(2027, 7, 5), date(2027, 9, 6),
        date(2027, 11, 25), date(2027, 12, 24),
    }
)  # fmt: skip

# Early closes at 13:00 Eastern: the day after Thanksgiving, Christmas Eve when
# it falls on a weekday, and 3 July when Independence Day is observed on the
# 4th. A half-day matters twice over — a coverage check would call the missing
# afternoon a gap, and an entry at 12:55 has no time for a stop behind it.
US_HALF_DAYS: frozenset[date] = frozenset(
    {
        date(2024, 7, 3), date(2024, 11, 29), date(2024, 12, 24),
        date(2025, 7, 3), date(2025, 11, 28), date(2025, 12, 24),
        date(2026, 11, 27), date(2026, 12, 24),
        date(2027, 11, 26),
    }
)  # fmt: skip

# The span the hand-maintained lists actually cover. A date outside it is
# `UNKNOWN`, not `REGULAR`: silently treating 2031 as a normal trading year
# because nobody extended the list is the failure this bound exists to prevent.
CALENDAR_FROM = date(2024, 1, 1)
CALENDAR_THROUGH = date(2027, 12, 31)


@dataclass(frozen=True, slots=True)
class TradingDay:
    """One date, classified, with its session bounds in UTC."""

    day: date
    kind: DayKind
    open_utc: datetime | None
    close_utc: datetime | None

    @property
    def is_trading_day(self) -> bool:
        return self.kind.is_trading_day

    def contains(self, moment: datetime) -> bool:
        if self.open_utc is None or self.close_utc is None:
            return False
        return self.open_utc <= moment < self.close_utc

    @property
    def duration(self) -> timedelta:
        if self.open_utc is None or self.close_utc is None:
            return timedelta(0)
        return self.close_utc - self.open_utc

    @property
    def expected_minute_bars(self) -> int:
        """How many regular-session minute bars this day should produce.

        What separates a real coverage gap from a half-day. Without it, every
        early close reads as 180 missing minutes and the audit's unexplained-gap
        percentage is dominated by days when nothing was wrong.
        """
        return int(self.duration.total_seconds() // 60)


@dataclass(frozen=True, slots=True)
class EntryVerdict:
    """Whether an entry may be placed now, and why not if not.

    Carries every blocking reason rather than the first. "The bot is not
    trading" has to be answerable without attaching a debugger, and a verdict
    that stops at the first failure hides the other three.
    """

    allowed: bool
    day: TradingDay
    reasons: tuple[str, ...] = field(default_factory=tuple)

    @property
    def detail(self) -> str:
        return "; ".join(self.reasons) if self.reasons else "within the regular session"


class TradingCalendar:
    """US equity sessions, from the broker's schedule where available.

    `broker_days` is the authority when present, keyed by date. It comes from
    `GET /equity/metadata/exchanges` — the venue that fills the orders deciding
    whether the market is open. The built-in lists are the fallback, and
    anything outside their range is `UNKNOWN`, which means closed.
    """

    def __init__(
        self,
        *,
        broker_days: Mapping[date, tuple[datetime, datetime]] | None = None,
        holidays: Iterable[date] = US_HOLIDAYS,
        half_days: Iterable[date] = US_HALF_DAYS,
        covers_from: date = CALENDAR_FROM,
        covers_through: date = CALENDAR_THROUGH,
    ) -> None:
        self._broker = dict(broker_days or {})
        self._holidays = frozenset(holidays)
        self._half_days = frozenset(half_days)
        self._from = covers_from
        self._through = covers_through

    @property
    def uses_broker_schedule(self) -> bool:
        return bool(self._broker)

    @property
    def coverage(self) -> tuple[date, date]:
        return (self._from, self._through)

    def classify(self, day: date) -> TradingDay:
        if day in self._broker:
            opened, closed = self._broker[day]
            kind = DayKind.HALF_DAY if closed - opened < timedelta(hours=6) else DayKind.REGULAR
            return TradingDay(day=day, kind=kind, open_utc=opened, close_utc=closed)

        if not self._from <= day <= self._through:
            # Fail-closed. A wrongly-closed day costs one session; a wrongly-open
            # day sends an order into a market that cannot fill it, and on
            # Trading 212 it queues to the next open at a price nothing in the
            # decision was based on.
            return TradingDay(day=day, kind=DayKind.UNKNOWN, open_utc=None, close_utc=None)

        if day.weekday() >= 5:
            return TradingDay(day=day, kind=DayKind.WEEKEND, open_utc=None, close_utc=None)
        if day in self._holidays:
            return TradingDay(day=day, kind=DayKind.HOLIDAY, open_utc=None, close_utc=None)

        half = day in self._half_days
        return TradingDay(
            day=day,
            kind=DayKind.HALF_DAY if half else DayKind.REGULAR,
            open_utc=_eastern(day, REGULAR_OPEN),
            close_utc=_eastern(day, HALF_DAY_CLOSE if half else REGULAR_CLOSE),
        )

    def day_of(self, moment: datetime) -> TradingDay:
        """The session day `moment` falls in, by the exchange's own date.

        Eastern rather than UTC: 01:00 UTC on a Tuesday is Monday evening in
        New York, and naming Tuesday's session for it would say the wrong day
        was closed.
        """
        return self.classify(_eastern_date(moment))

    def is_open_at(self, moment: datetime) -> bool:
        return self.day_of(moment).contains(moment)

    def sessions_between(self, start: date, end: date) -> tuple[TradingDay, ...]:
        """Every trading day in `[start, end]`, inclusive.

        The denominator for a coverage check: comparing bars held against
        calendar days rather than trading days would report every weekend as
        missing data.
        """
        if end < start:
            raise CalendarError(f"date range runs backwards: {start} to {end}")
        days: list[TradingDay] = []
        cursor = start
        while cursor <= end:
            classified = self.classify(cursor)
            if classified.is_trading_day:
                days.append(classified)
            cursor += timedelta(days=1)
        return tuple(days)

    def previous_session(self, day: date, *, limit: int = 10) -> TradingDay | None:
        """The last trading day strictly before `day`.

        Needed for the dividend reference close and for the residual detector's
        `prev_close`: "yesterday" is wrong across a weekend or a holiday, and
        wrong in a way that changes which gap looks unexplained.
        """
        cursor = day - timedelta(days=1)
        for _ in range(limit):
            classified = self.classify(cursor)
            if classified.is_trading_day:
                return classified
            cursor -= timedelta(days=1)
        return None

    def next_session(self, day: date, *, limit: int = 10) -> TradingDay | None:
        cursor = day + timedelta(days=1)
        for _ in range(limit):
            classified = self.classify(cursor)
            if classified.is_trading_day:
                return classified
            cursor += timedelta(days=1)
        return None

    # -- the one predicate -------------------------------------------------

    def may_enter_now(
        self,
        session: Session,
        now: datetime,
        *,
        no_entry_after_open: timedelta = NO_ENTRY_AFTER_OPEN,
        no_entry_before_close: timedelta = NO_ENTRY_BEFORE_CLOSE,
    ) -> EntryVerdict:
        """The single gate every entry path calls.

        `session` is the session the *deciding bar* belongs to, not the current
        one. An extended-hours bar is not comparable to a regular-session bar —
        its price may be hundreds of basis points away on an earnings night —
        so a decision computed on one must not become an order in the other.
        """
        if now.tzinfo is None:
            raise CalendarError("now must be timezone-aware")

        day = self.classify(_eastern_date(now))
        reasons: list[str] = []

        if day.kind is DayKind.UNKNOWN:
            reasons.append(
                f"{day.day} is outside the calendar's range "
                f"{self._from}..{self._through}, so it is treated as closed. Extend "
                "US_HOLIDAYS and US_HALF_DAYS, or fetch the broker's working schedules."
            )
        elif not day.is_trading_day:
            reasons.append(f"{day.day} is a {day.kind.value}")
        elif not day.contains(now):
            assert day.open_utc is not None and day.close_utc is not None
            reasons.append(
                f"{now.isoformat()} is outside the {day.kind.value} session "
                f"({day.open_utc.isoformat()} to {day.close_utc.isoformat()})"
            )
        else:
            assert day.open_utc is not None and day.close_utc is not None
            if now - day.open_utc < no_entry_after_open:
                reasons.append(
                    f"within {no_entry_after_open} of the open; the opening auction's "
                    "prints are not comparable to the continuous session"
                )
            if day.close_utc - now < no_entry_before_close:
                reasons.append(
                    f"within {no_entry_before_close} of the {day.kind.value} close; an "
                    "entry filled now cannot have a protective stop behind it before the "
                    "bell, leaving the position exposed across the overnight gap"
                )

        if session is not Session.REGULAR:
            reasons.append(
                f"the deciding bar is a {session.value}-session print, which is not "
                "comparable to a regular-session price"
            )

        return EntryVerdict(allowed=not reasons, day=day, reasons=tuple(reasons))

    def may_exit_now(self, now: datetime) -> EntryVerdict:
        """Exits are permitted through the whole session, right to the bell.

        Deliberately asymmetric with `may_enter_now`. Every rule above exists to
        stop the bot *taking on* risk in a bad moment; none of them is a reason
        to hold a position it has decided to close. A calendar problem must
        never convert into an unhedged position, so the only thing that blocks
        an exit is the market being shut.
        """
        if now.tzinfo is None:
            raise CalendarError("now must be timezone-aware")
        day = self.classify(_eastern_date(now))
        if day.contains(now):
            return EntryVerdict(allowed=True, day=day)
        return EntryVerdict(
            allowed=False,
            day=day,
            reasons=(f"the market is not open at {now.isoformat()} ({day.kind.value})",),
        )


# --------------------------------------------------------------------------
# Broker schedules
# --------------------------------------------------------------------------


def broker_days_from_exchanges(
    exchanges: Iterable[object], *, exchange_id: int | None = None
) -> dict[date, tuple[datetime, datetime]]:
    """Turn `GET /equity/metadata/exchanges` time events into session bounds.

    Trading 212 publishes a list of `{date, type}` events per working schedule.
    Pairing an `OPEN` with the next `CLOSE` on the same Eastern date gives the
    session. Events that do not pair are skipped rather than guessed at: a
    fabricated close time would be worse than no schedule, because the calendar
    would then confidently authorise entries at a time nobody verified.
    """
    sessions: dict[date, tuple[datetime, datetime]] = {}
    for exchange in exchanges:
        eid = getattr(exchange, "exchange_id", None)
        if exchange_id is not None and eid != exchange_id:
            continue
        for schedule in getattr(exchange, "working_schedules", ()) or ():
            opens: dict[date, datetime] = {}
            for event in getattr(schedule, "time_events", ()) or ():
                raw_date = getattr(event, "date", None)
                raw_type = (getattr(event, "event_type", None) or "").upper()
                if not raw_date or raw_type not in ("OPEN", "CLOSE"):
                    continue
                try:
                    moment = _parse_broker_time(str(raw_date))
                except CalendarError:
                    continue
                local_day = _eastern_date(moment)
                if raw_type == "OPEN":
                    opens[local_day] = moment
                elif local_day in opens and moment > opens[local_day]:
                    sessions[local_day] = (opens.pop(local_day), moment)
    return sessions


def _parse_broker_time(text: str) -> datetime:
    normalised = text.strip()
    if normalised.endswith("Z"):
        normalised = f"{normalised[:-1]}+00:00"
    try:
        parsed = datetime.fromisoformat(normalised)
    except ValueError as exc:
        raise CalendarError(f"unparseable schedule timestamp {text!r}") from exc
    if parsed.tzinfo is None:
        raise CalendarError(f"schedule timestamp {text!r} carries no offset")
    return parsed.astimezone(UTC)


def _eastern(day: date, clock: time) -> datetime:
    """A US session boundary, built in Eastern and returned in UTC.

    Built in Eastern rather than as a fixed UTC offset because the session
    moves with Eastern DST, not with UTC. A hardcoded 14:30 is right for four
    months of the year and an hour wrong for the rest.
    """
    local = datetime.combine(day, clock, tzinfo=US_EASTERN)
    return local.astimezone(UTC)


def _eastern_date(moment: datetime) -> date:
    if moment.tzinfo is None:
        raise CalendarError("moment must be timezone-aware")
    return moment.astimezone(US_EASTERN).date()
