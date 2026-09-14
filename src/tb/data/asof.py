"""Point-in-time visibility: what was knowable, when.

This is the module that decides whether the backtester can cheat, so it is
deliberately small, pure, and testable without touching a filesystem.

Three ideas do the work:

**One visibility function.** `visible_bars(as_of=...)` is used by the live loop
and by the backtester alike. Two code paths that answer "which bars may I see"
differently will diverge, and the divergence is always in the profitable
direction because that is the direction people stop investigating.

**Future rows are not in memory.** `ForwardOnlyReader.advance_to(t)` hands back
a `BarWindow` containing only bars with `available_at <= t`, and there is no
method on it that returns anything later. A full-history frame plus a "don't
peek" convention is not a defence: that convention is broken by every
`.shift(-1)`, every `bfill`, every `rolling(center=True)`, and every
`scaler.fit(X_full)`. The only structural defence is absence.

**Missing is `UNKNOWN`, never `None` or `NaN`.** An uncovered span returns a
sentinel that raises on arithmetic and on truth-testing, so feature code cannot
silently treat "no data" as zero or as last-value. `if not window.last(uid)`
raises rather than quietly taking the false branch.
"""

from __future__ import annotations

from collections.abc import Iterable, Iterator, Mapping, Sequence
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from decimal import Decimal
from enum import StrEnum
from typing import Any, Protocol, TypeAlias, runtime_checkable

from tb.data.provider import Bar, DataError, Resolution, Session, dedupe_latest


class LookaheadError(DataError):
    """Something asked for data it could not have had.

    Raised rather than answered. This is the canary: a strategy that reaches
    past its decision time must crash, because a strategy that silently gets
    `UNKNOWN` and handles it badly produces a plausible-looking backtest.
    """


class UnknownValueError(DataError):
    """An `UNKNOWN` value was used as if it were a number or a boolean."""


class _Unknown:
    """Absence of data, as a value that refuses to be mistaken for one.

    Deliberately hostile. `None` gets `or 0`-ed, `NaN` propagates silently
    through pandas and comes out the far end as a plausible number, and both
    turn "we have no price" into "the price is zero" somewhere downstream. This
    raises on every arithmetic and truth operation, so the only way past it is
    to handle it explicitly.
    """

    __slots__ = ()
    _MESSAGE = (
        "this value is UNKNOWN — the data layer has no coverage for that span. "
        "Handle it explicitly (`is UNKNOWN`) rather than letting it become a "
        "number: treating missing data as zero or as last-value is how a "
        "backtest invents returns nobody could have earned."
    )

    def __repr__(self) -> str:
        return "UNKNOWN"

    def __bool__(self) -> bool:
        raise UnknownValueError(self._MESSAGE)

    def _refuse(self, *_args: object, **_kwargs: object) -> Any:
        raise UnknownValueError(self._MESSAGE)

    __add__ = __radd__ = _refuse
    __sub__ = __rsub__ = _refuse
    __mul__ = __rmul__ = _refuse
    __truediv__ = __rtruediv__ = _refuse
    __floordiv__ = __rfloordiv__ = _refuse
    __lt__ = __le__ = __gt__ = __ge__ = _refuse
    __float__ = __int__ = _refuse
    __iter__ = _refuse
    __len__ = _refuse


UNKNOWN = _Unknown()

# What a lookup returns: a value, or the refusal above. (PEP 695 `type`
# aliases need 3.12; this project targets 3.11.)
MaybeBar: TypeAlias = "Bar | _Unknown"
MaybePrice: TypeAlias = "Decimal | _Unknown"

# The sentinel's type, under a public name. Other modules need it to spell
# `Something | Unknown` in a signature — `_Unknown` is private and importing it
# across modules would be the wrong shape of dependency.
Unknown: TypeAlias = _Unknown


# --------------------------------------------------------------------------
# Bar sources
# --------------------------------------------------------------------------


@runtime_checkable
class BarSource(Protocol):
    """Where bars come from.

    Keeping this a protocol is what lets every visibility property be tested
    against an in-memory list, with no Parquet, no DuckDB and no clock.
    """

    def bars_for(
        self,
        instrument_uid: str,
        resolution: Resolution,
        *,
        start: datetime | None = None,
        end: datetime | None = None,
    ) -> Iterable[Bar]: ...

    def instruments(self) -> Iterable[str]: ...


@dataclass(slots=True)
class InMemoryBarSource:
    """A bar source backed by a list. Tests and backtests over a sealed vintage."""

    bars: list[Bar] = field(default_factory=list)

    def add(self, *bars: Bar) -> None:
        self.bars.extend(bars)

    def bars_for(
        self,
        instrument_uid: str,
        resolution: Resolution,
        *,
        start: datetime | None = None,
        end: datetime | None = None,
    ) -> Iterable[Bar]:
        for bar in self.bars:
            if bar.instrument_uid != instrument_uid or bar.resolution is not resolution:
                continue
            if start is not None and bar.bar_open_utc < start:
                continue
            if end is not None and bar.bar_open_utc > end:
                continue
            yield bar

    def instruments(self) -> Iterable[str]:
        return sorted({bar.instrument_uid for bar in self.bars})


# --------------------------------------------------------------------------
# Coverage
# --------------------------------------------------------------------------


class CoverageStatus(StrEnum):
    COVERED = "covered"
    PARTIAL = "partial"
    UNCOVERED = "uncovered"


@dataclass(frozen=True, slots=True)
class Coverage:
    """Whether we have data for a span — distinct from whether bars printed.

    "No bar printed in that minute" and "we have no coverage for that minute"
    are different facts with different consequences, and a feature computed
    over the second one is not a feature. Keeping them distinguishable is why
    `UNKNOWN` exists.
    """

    status: CoverageStatus
    instrument_uid: str
    resolution: Resolution
    requested_start: datetime
    requested_end: datetime
    present_bars: int
    first_present: datetime | None = None
    last_present: datetime | None = None

    @property
    def usable(self) -> bool:
        return self.status is CoverageStatus.COVERED


# --------------------------------------------------------------------------
# Staleness
# --------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class StalenessVerdict:
    """Whether the newest knowable bar is fresh enough to act on.

    Measured against `available_at`, never against bar time. A fifteen-minute
    delayed feed produces bars whose *timestamps* look current while the data
    was knowable a quarter of an hour ago; measuring the wrong axis is what
    makes such a feed appear usable.
    """

    fresh: bool
    age_seconds: float | None
    limit_seconds: int
    reason: str

    def raise_if_stale(self) -> None:
        if not self.fresh:
            raise LookaheadError(self.reason)


def assess_staleness(bar: MaybeBar, *, now: datetime, limit_seconds: int) -> StalenessVerdict:
    if bar is UNKNOWN or not isinstance(bar, Bar):
        return StalenessVerdict(
            fresh=False,
            age_seconds=None,
            limit_seconds=limit_seconds,
            reason="no bar is knowable yet, so there is nothing fresh to act on",
        )
    age = bar.age_seconds_at(now)
    if age <= limit_seconds:
        return StalenessVerdict(
            fresh=True,
            age_seconds=age,
            limit_seconds=limit_seconds,
            reason=f"newest bar became knowable {age:.0f}s ago (limit {limit_seconds}s)",
        )
    return StalenessVerdict(
        fresh=False,
        age_seconds=age,
        limit_seconds=limit_seconds,
        reason=(
            f"{bar.instrument_uid}: newest knowable bar is {age:.0f}s old against a "
            f"{limit_seconds}s bound. Acting on it would mean trading on a price the "
            "market has already moved past."
        ),
    )


# --------------------------------------------------------------------------
# The as-of view
# --------------------------------------------------------------------------


def visible_bars(
    source: BarSource,
    instrument_uid: str,
    resolution: Resolution,
    *,
    as_of: datetime,
    lookback: timedelta | None = None,
    include_extended: bool = False,
) -> tuple[Bar, ...]:
    """Every bar that was knowable at `as_of`, oldest first.

    The one visibility function. Filters on `available_at <= as_of` — not on
    bar open, not on bar close — and collapses revisions to the version that
    had been ingested by then, so an as-of query returns the data as it stood,
    not as it has since been restated.
    """
    if as_of.tzinfo is None:
        raise DataError("as_of must be timezone-aware")

    start = None if lookback is None else as_of - lookback
    candidates = [
        bar
        for bar in source.bars_for(instrument_uid, resolution, start=start)
        # Three conditions, none redundant: the bar must have been knowable
        # by `as_of`, the *revision* must also have been ingested by then (or an
        # as-of query would return a restatement that had not happened yet), and
        # extended-hours prints are excluded unless asked for, because comparing
        # a post-market print against a regular-hours close is not a comparison.
        if bar.is_visible_at(as_of)
        and bar.ingested_at_utc <= as_of
        and (include_extended or bar.session is not Session.EXTENDED)
    ]
    return dedupe_latest(candidates)


@dataclass(slots=True)
class AsOfView:
    """A read of the store frozen at one instant."""

    source: BarSource
    as_of: datetime
    include_extended: bool = False

    def bars(
        self,
        instrument_uid: str,
        resolution: Resolution,
        *,
        lookback: timedelta | None = None,
    ) -> tuple[Bar, ...]:
        return visible_bars(
            self.source,
            instrument_uid,
            resolution,
            as_of=self.as_of,
            lookback=lookback,
            include_extended=self.include_extended,
        )

    def latest(self, instrument_uid: str, resolution: Resolution) -> MaybeBar:
        found = self.bars(instrument_uid, resolution)
        return found[-1] if found else UNKNOWN

    def coverage(
        self,
        instrument_uid: str,
        resolution: Resolution,
        *,
        start: datetime,
        end: datetime,
        expected_bars: int | None = None,
    ) -> Coverage:
        present = [
            bar for bar in self.bars(instrument_uid, resolution) if start <= bar.bar_open_utc <= end
        ]
        if not present:
            status = CoverageStatus.UNCOVERED
        elif expected_bars is not None and len(present) < expected_bars:
            status = CoverageStatus.PARTIAL
        else:
            status = CoverageStatus.COVERED
        return Coverage(
            status=status,
            instrument_uid=instrument_uid,
            resolution=resolution,
            requested_start=start,
            requested_end=end,
            present_bars=len(present),
            first_present=present[0].bar_open_utc if present else None,
            last_present=present[-1].bar_open_utc if present else None,
        )


# --------------------------------------------------------------------------
# The window handed to a strategy
# --------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class BarWindow:
    """Bars visible at one decision time. Contains nothing later.

    There is no method here that returns a future bar, and no attribute holding
    one. That absence is the lookahead defence — everything else is convention,
    and conventions get refactored away.
    """

    as_of: datetime
    resolution: Resolution
    _by_uid: Mapping[str, tuple[Bar, ...]]

    @property
    def instruments(self) -> tuple[str, ...]:
        return tuple(sorted(self._by_uid))

    def bars(self, instrument_uid: str) -> tuple[Bar, ...]:
        return self._by_uid.get(instrument_uid, ())

    def last(self, instrument_uid: str) -> MaybeBar:
        found = self.bars(instrument_uid)
        return found[-1] if found else UNKNOWN

    def closes(self, instrument_uid: str, count: int) -> tuple[Decimal, ...] | _Unknown:
        """The last `count` closes, or `UNKNOWN` if there are not that many.

        Refusing a short window rather than padding it is the point: a 200-day
        moving average computed over 40 days is not a shorter moving average,
        it is a different and wrong number.
        """
        if count <= 0:
            raise ValueError("count must be positive")
        found = self.bars(instrument_uid)
        if len(found) < count:
            return UNKNOWN
        return tuple(bar.close for bar in found[-count:])

    def require_visible(self, timestamp: datetime) -> None:
        """The canary. Raises if `timestamp` is past this window's as-of.

        A strategy that reaches forward calls this (directly or via a helper)
        and crashes, instead of quietly receiving `UNKNOWN` and producing a
        backtest nobody questions.
        """
        if timestamp > self.as_of:
            raise LookaheadError(
                f"asked for data at {timestamp.isoformat()} from a window as of "
                f"{self.as_of.isoformat()}. That information did not exist yet."
            )

    def staleness(
        self, instrument_uid: str, *, now: datetime | None = None, limit_seconds: int
    ) -> StalenessVerdict:
        return assess_staleness(
            self.last(instrument_uid),
            now=now or self.as_of,
            limit_seconds=limit_seconds,
        )

    def __len__(self) -> int:
        return sum(len(v) for v in self._by_uid.values())


# --------------------------------------------------------------------------
# The forward-only reader
# --------------------------------------------------------------------------


@dataclass(slots=True)
class ForwardOnlyReader:
    """Walks time forward. Cannot be rewound, cannot look ahead.

    `advance_to` refuses to go backwards. That is not tidiness: a backtester
    that can re-read an earlier moment can fit a parameter on data it then
    "predicts", and the refusal makes that inexpressible rather than merely
    discouraged.
    """

    source: BarSource
    resolution: Resolution
    instrument_uids: tuple[str, ...]
    lookback: timedelta | None = None
    include_extended: bool = False
    _current: datetime | None = None
    _advances: int = 0

    @property
    def current_time(self) -> datetime | None:
        return self._current

    @property
    def advances(self) -> int:
        return self._advances

    def advance_to(self, decision_time: datetime) -> BarWindow:
        """Move to `decision_time` and return what was knowable then."""
        if decision_time.tzinfo is None:
            raise DataError("decision_time must be timezone-aware")
        if self._current is not None and decision_time < self._current:
            raise LookaheadError(
                f"cannot rewind from {self._current.isoformat()} to "
                f"{decision_time.isoformat()}. A reader that can revisit an earlier "
                "moment can fit on data it then predicts."
            )
        self._current = decision_time
        self._advances += 1

        window: dict[str, tuple[Bar, ...]] = {}
        for uid in self.instrument_uids:
            window[uid] = visible_bars(
                self.source,
                uid,
                self.resolution,
                as_of=decision_time,
                lookback=self.lookback,
                include_extended=self.include_extended,
            )
        return BarWindow(as_of=decision_time, resolution=self.resolution, _by_uid=window)

    def walk(self, decision_times: Sequence[datetime]) -> Iterator[BarWindow]:
        """Advance through a schedule of decision times in order."""
        for moment in decision_times:
            yield self.advance_to(moment)
