"""Pure predicates over bars. Every one returns a typed finding, or nothing.

No I/O, no ledger, no clock beyond what is passed in. That is what makes each
check exhaustively testable against hand-built bar sequences, and it is why the
orchestration lives next door in `audit.py`: a predicate that reads a database
can only be tested against a database, and then nobody tests the interesting
cases.

**Gaps are classified, not counted.** This is the single most important thing
in the module. A missing minute on a thin name over IEX is completely normal —
IEX is ~2% of consolidated volume, so a small-cap simply has minutes with no
print. A missing minute on a megacap is a feed failure. A missing afternoon on
Christmas Eve is a half-day. Lumping them into one "gaps: 4,312" number is
uninterpretable, and an uninterpretable number gets ignored, which means the
one gap that mattered was never seen. So every gap gets a `GapCause`, and only
`UNEXPLAINED` counts against `data.max_unexplained_gap_pct`.

The checks that earn their place, roughly in order of how much they have caught
in systems like this:

* **Frozen feed across symbols.** Every symbol's latest bar stuck at the same
  instant means the *provider* stopped, not the market. Per-symbol staleness
  checks miss it because each symbol looks individually plausible.
* **Stale repeats.** The same OHLCV repeated bar after bar is a feed echoing
  its last value. Distinguishable from a genuinely flat minute only by the run
  length, which is why the threshold exists.
* **Jumps, with an explanation attempted.** A large gap that a split explains
  is not a data problem; the same gap with no action row is either an
  unannounced split or a mismapped ticker.
* **Knowledge-time sanity.** A bar claiming to be knowable before it closed is
  a lookahead channel. `Bar.__post_init__` already refuses to build one, so
  this is the belt to that braces — it catches rows that entered the store
  through some other path, which is exactly the case the constructor cannot.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from datetime import UTC, date, datetime, timedelta
from decimal import Decimal
from enum import StrEnum
from typing import Any

from tb.broker.reconcile import Severity
from tb.core.clock import from_iso
from tb.data.adjustments import CorporateAction, SplitSuspicion, detect_unexplained_split
from tb.data.calendar import DayKind, TradingCalendar
from tb.data.provider import US_EASTERN, Bar, Provenance, Resolution

# A run of identical bars longer than this is a feed repeating itself rather
# than a market standing still. Three flat minutes on a thin name is ordinary;
# thirty is not, and on a megacap even three is worth a look — which is why the
# threshold is a parameter and the caller scales it by liquidity.
MAX_IDENTICAL_RUN = 12

# A single-bar move beyond this is worth explaining. Well above an ordinary
# session so the detector does not fire on every earnings day, and well below a
# 2-for-1 split so it does fire on the thing it is looking for.
JUMP_PCT = Decimal("25")

# Two prices this far apart in a single bar mean the high/low are not from the
# same instrument as the open/close, or the tick size is wrong.
MAX_INTRABAR_RANGE_PCT = Decimal("50")


class CheckKind(StrEnum):
    COVERAGE_GAP = "coverage_gap"
    STALE_REPEAT = "stale_repeat"
    FROZEN_FEED = "frozen_feed"
    UNEXPLAINED_JUMP = "unexplained_jump"
    EXPLAINED_JUMP = "explained_jump"
    IMPLAUSIBLE_RANGE = "implausible_range"
    ZERO_VOLUME = "zero_volume"
    KNOWLEDGE_TIME_INVALID = "knowledge_time_invalid"
    DUPLICATE_BAR = "duplicate_bar"
    OUT_OF_ORDER = "out_of_order"
    REVISION_CLUSTER = "revision_cluster"
    BACKFILL_ONLY = "backfill_only"
    PARTITION_MISSING = "partition_missing"
    PARTITION_ALTERED = "partition_altered"
    PARTITION_ORPHAN = "partition_orphan"
    RESEARCH_ONLY_IDENTITY = "research_only_identity"


class GapCause(StrEnum):
    """Why a bar is missing. Only the last one counts against a feed.

    The distinction is the whole point of the coverage check. Without it, a
    feed that is working perfectly on a universe containing one thin name
    reports thousands of "gaps" and the number becomes noise.
    """

    MARKET_CLOSED = "market_closed"
    HALF_DAY = "half_day"
    KNOWN_HALT = "known_halt"
    THIN_NAME_NO_PRINT = "thin_name_no_print"
    OUTSIDE_COVERAGE = "outside_coverage"
    UNEXPLAINED = "unexplained"

    @property
    def counts_against_feed(self) -> bool:
        return self is GapCause.UNEXPLAINED


@dataclass(frozen=True, slots=True)
class Finding:
    """One thing wrong, or one thing explained.

    Deliberately the same shape as `broker.reconcile.ReconcileFinding` — same
    `Severity`, same `as_dict`, same `__str__` — so the two reports read alike
    and an operator does not have to learn two vocabularies during an incident.
    """

    kind: CheckKind
    severity: Severity
    detail: str
    instrument_uid: str | None = None
    resolution: Resolution | None = None
    at: datetime | None = None
    cause: GapCause | None = None
    observed: str | None = None
    suggested_action: str | None = None

    def as_dict(self) -> dict[str, Any]:
        return {
            "kind": self.kind.value,
            "severity": self.severity.value,
            "instrument_uid": self.instrument_uid,
            "resolution": None if self.resolution is None else self.resolution.value,
            "at": None if self.at is None else self.at.isoformat(),
            "cause": None if self.cause is None else self.cause.value,
            "detail": self.detail,
            "observed": self.observed,
            "suggested_action": self.suggested_action,
        }

    def __str__(self) -> str:
        where = f" [{self.instrument_uid}]" if self.instrument_uid else ""
        when = f" @{self.at.isoformat()}" if self.at else ""
        return f"{self.severity.value.upper()}{where}{when} {self.kind.value}: {self.detail}"

    @property
    def blocking(self) -> bool:
        return self.severity is Severity.BLOCKING


# --------------------------------------------------------------------------
# Coverage
# --------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class CoverageReport:
    """What a window should have contained against what it did.

    `unexplained_pct` is the number `data.max_unexplained_gap_pct` bounds, and
    it is deliberately *not* `missing / expected`: a thin name with no IEX
    prints would fail a threshold it has no way to meet, and the only available
    fix would be to loosen the threshold for everyone.
    """

    instrument_uid: str
    resolution: Resolution
    expected: int
    present: int
    by_cause: dict[GapCause, int] = field(default_factory=dict)

    @property
    def missing(self) -> int:
        return max(0, self.expected - self.present)

    @property
    def unexplained(self) -> int:
        return self.by_cause.get(GapCause.UNEXPLAINED, 0)

    @property
    def unexplained_pct(self) -> float:
        if self.expected <= 0:
            return 0.0
        return self.unexplained / self.expected * 100.0

    @property
    def explained(self) -> int:
        return sum(count for cause, count in self.by_cause.items() if not cause.counts_against_feed)


def classify_gap(
    missing_at: datetime,
    *,
    calendar: TradingCalendar,
    resolution: Resolution,
    halted_dates: Iterable[date] = (),
    thin_name: bool = False,
) -> GapCause:
    """Say why one bar is absent.

    Two axes have to be read differently, and getting it wrong silently
    reclassifies every gap. A **daily** bar is anchored at midnight UTC of its
    session date (see `normalise_bar_open`), so its session date is the *UTC*
    date and it has no position inside the session at all. An **intraday** bar
    carries a real instant, so its session date is the *Eastern* date and its
    position within the session is meaningful. Reading a daily bar's date in
    Eastern puts midnight UTC at 19:00 the previous evening, which lands on the
    wrong session and then falls outside it — turning every genuinely missing
    session into a reassuring "market closed".

    Order matters too: a date outside the calendar's range must not be reported
    as a market closure, because "we do not know" and "the market was shut" are
    different claims and only the first needs someone to extend a list.
    """
    daily = resolution is Resolution.DAILY
    session_date = (
        missing_at.astimezone(UTC).date() if daily else missing_at.astimezone(US_EASTERN).date()
    )
    day = calendar.classify(session_date)
    if day.kind is DayKind.UNKNOWN:
        return GapCause.OUTSIDE_COVERAGE
    if not day.is_trading_day:
        return GapCause.MARKET_CLOSED
    if session_date in set(halted_dates):
        return GapCause.KNOWN_HALT
    if not daily and not day.contains(missing_at):
        # Inside a trading day but outside its session: on a half-day that is
        # the missing afternoon, which is not a gap at all.
        return GapCause.HALF_DAY if day.kind is DayKind.HALF_DAY else GapCause.MARKET_CLOSED
    if thin_name and resolution.is_intraday:
        # IEX is ~2% of consolidated volume. A small-cap genuinely has minutes
        # with no print, and counting those against the feed makes the metric
        # unusable for the whole universe. Not extended to daily bars: every
        # listed name prints *something* over a whole session, so a missing
        # daily bar is missing however thin the name.
        return GapCause.THIN_NAME_NO_PRINT
    return GapCause.UNEXPLAINED


def check_coverage(
    bars: Sequence[Bar],
    *,
    instrument_uid: str,
    resolution: Resolution,
    start: date,
    end: date,
    calendar: TradingCalendar,
    max_unexplained_pct: float,
    halted_dates: Iterable[date] = (),
    thin_name: bool = False,
) -> tuple[CoverageReport, tuple[Finding, ...]]:
    """Compare bars held against bars the calendar says should exist.

    The denominator is *trading* sessions, never calendar days. Using calendar
    days would report every weekend as missing data and bury the real gaps.
    """
    sessions = calendar.sessions_between(start, end)
    present_opens = {bar.bar_open_utc for bar in bars}

    expected_opens: list[datetime] = []
    for session in sessions:
        if resolution is Resolution.DAILY:
            expected_opens.append(
                datetime(session.day.year, session.day.month, session.day.day, tzinfo=UTC)
            )
            continue
        if session.open_utc is None:
            continue
        step = resolution.duration
        cursor = session.open_utc
        while session.close_utc is not None and cursor < session.close_utc:
            expected_opens.append(cursor)
            cursor += step

    by_cause: dict[GapCause, int] = {}
    unexplained_moments: list[datetime] = []
    for moment in expected_opens:
        if moment in present_opens:
            continue
        cause = classify_gap(
            moment,
            calendar=calendar,
            resolution=resolution,
            halted_dates=halted_dates,
            thin_name=thin_name,
        )
        by_cause[cause] = by_cause.get(cause, 0) + 1
        if cause.counts_against_feed:
            unexplained_moments.append(moment)

    report = CoverageReport(
        instrument_uid=instrument_uid,
        resolution=resolution,
        expected=len(expected_opens),
        present=sum(1 for moment in expected_opens if moment in present_opens),
        by_cause=by_cause,
    )

    findings: list[Finding] = []
    if report.unexplained_pct > max_unexplained_pct:
        findings.append(
            Finding(
                kind=CheckKind.COVERAGE_GAP,
                # Blocking: a feature computed over a window with holes in it is
                # not a shorter feature, it is a different and wrong number.
                severity=Severity.BLOCKING,
                instrument_uid=instrument_uid,
                resolution=resolution,
                at=unexplained_moments[0] if unexplained_moments else None,
                cause=GapCause.UNEXPLAINED,
                detail=(
                    f"{report.unexplained} of {report.expected} expected "
                    f"{resolution.value} bars are missing with no explanation "
                    f"({report.unexplained_pct:.2f}% against a "
                    f"{max_unexplained_pct:.2f}% bound). "
                    f"{report.explained} further gaps are explained "
                    f"({_cause_summary(by_cause)})."
                ),
                observed=f"{report.unexplained_pct:.2f}%",
                suggested_action=(
                    "re-run the backfill for this window; if the gaps persist the feed "
                    "does not have the history this resolution needs"
                ),
            )
        )
    elif unexplained_moments:
        findings.append(
            Finding(
                kind=CheckKind.COVERAGE_GAP,
                severity=Severity.INFO,
                instrument_uid=instrument_uid,
                resolution=resolution,
                at=unexplained_moments[0],
                cause=GapCause.UNEXPLAINED,
                detail=(
                    f"{report.unexplained} unexplained gap(s) within the "
                    f"{max_unexplained_pct:.2f}% bound ({_cause_summary(by_cause)})"
                ),
                observed=f"{report.unexplained_pct:.2f}%",
            )
        )
    return report, tuple(findings)


def _cause_summary(by_cause: dict[GapCause, int]) -> str:
    if not by_cause:
        return "none"
    return ", ".join(f"{cause.value}={count}" for cause, count in sorted(by_cause.items(), key=str))


# --------------------------------------------------------------------------
# Per-series checks
# --------------------------------------------------------------------------


def check_ordering(
    bars: Sequence[Bar], *, instrument_uid: str, resolution: Resolution
) -> tuple[Finding, ...]:
    """Bars must arrive in time order with no duplicate periods.

    Expects **one provider's** series. Two providers' bars for the same period
    are different observations, both legitimate, so a duplicate is keyed on
    `(bar_open, provider)` — and a multi-provider series interleaved by bar
    time is not a time series at all, which is why `audit.py` groups by
    provider before calling this.

    A genuine duplicate — same period, same provider, same vintage — means the
    read path collapsed vintages wrongly, and every feature over the window is
    then computed on a doubled bar.
    """
    findings: list[Finding] = []
    seen: dict[tuple[datetime, str], Bar] = {}
    previous: Bar | None = None
    for bar in bars:
        if previous is not None and bar.bar_open_utc < previous.bar_open_utc:
            findings.append(
                Finding(
                    kind=CheckKind.OUT_OF_ORDER,
                    severity=Severity.BLOCKING,
                    instrument_uid=instrument_uid,
                    resolution=resolution,
                    at=bar.bar_open_utc,
                    detail=(
                        f"bar at {bar.bar_open_utc.isoformat()} follows "
                        f"{previous.bar_open_utc.isoformat()}. Anything that walks this "
                        "series forward would see time run backwards."
                    ),
                )
            )
        key = (bar.bar_open_utc, bar.provider)
        existing = seen.get(key)
        if existing is not None and existing.ingested_at_utc == bar.ingested_at_utc:
            findings.append(
                Finding(
                    kind=CheckKind.DUPLICATE_BAR,
                    severity=Severity.BLOCKING,
                    instrument_uid=instrument_uid,
                    resolution=resolution,
                    at=bar.bar_open_utc,
                    detail=(
                        f"two bars for {bar.bar_open_utc.isoformat()} at the same vintage "
                        f"from {bar.provider} — a vintage collapse, which doubles this "
                        "bar's weight in every feature over the window"
                    ),
                )
            )
        seen[key] = bar
        previous = bar
    return tuple(findings)


def check_knowledge_times(
    bars: Sequence[Bar], *, instrument_uid: str, resolution: Resolution
) -> tuple[Finding, ...]:
    """No bar may claim to have been knowable before it closed.

    `Bar.__post_init__` refuses to construct one, so this exists to catch rows
    that reached the store some other way — a hand-edited Parquet file, a
    migration, a future writer. Precisely the case the constructor cannot see.
    """
    findings: list[Finding] = []
    for bar in bars:
        if bar.available_at_utc < bar.bar_close_utc:
            findings.append(
                Finding(
                    kind=CheckKind.KNOWLEDGE_TIME_INVALID,
                    severity=Severity.BLOCKING,
                    instrument_uid=instrument_uid,
                    resolution=resolution,
                    at=bar.bar_open_utc,
                    detail=(
                        f"available_at {bar.available_at_utc.isoformat()} precedes bar "
                        f"close {bar.bar_close_utc.isoformat()}. Any backtest over this "
                        "row sees the bar before it existed."
                    ),
                    suggested_action="delete the partition and refetch; do not adjust the row",
                )
            )
    return tuple(findings)


def check_stale_repeats(
    bars: Sequence[Bar],
    *,
    instrument_uid: str,
    resolution: Resolution,
    max_run: int = MAX_IDENTICAL_RUN,
) -> tuple[Finding, ...]:
    """A run of identical bars is a feed echoing its last value.

    Volume is part of the comparison and does most of the work: a genuinely
    flat minute still has *some* volume, so identical OHLC *and* identical
    volume is a repeat rather than a quiet market.
    """
    findings: list[Finding] = []
    run_start: Bar | None = None
    run_length = 0

    def signature(bar: Bar) -> tuple[Decimal, Decimal, Decimal, Decimal, int | None]:
        return (bar.open, bar.high, bar.low, bar.close, bar.volume)

    previous: Bar | None = None
    for bar in bars:
        if previous is not None and signature(bar) == signature(previous):
            run_length += 1
            run_start = run_start or previous
        else:
            if run_start is not None and run_length >= max_run:
                findings.append(_stale_finding(run_start, run_length, instrument_uid, resolution))
            run_start, run_length = None, 0
        previous = bar
    if run_start is not None and run_length >= max_run:
        findings.append(_stale_finding(run_start, run_length, instrument_uid, resolution))
    return tuple(findings)


def _stale_finding(start: Bar, length: int, instrument_uid: str, resolution: Resolution) -> Finding:
    return Finding(
        kind=CheckKind.STALE_REPEAT,
        severity=Severity.WARN,
        instrument_uid=instrument_uid,
        resolution=resolution,
        at=start.bar_open_utc,
        detail=(
            f"{length + 1} consecutive bars with identical OHLC and volume from "
            f"{start.bar_open_utc.isoformat()}. A flat market still prints varying "
            "volume, so this is the feed repeating its last value."
        ),
        observed=f"{length + 1} bars",
    )


def check_intrabar_range(
    bars: Sequence[Bar],
    *,
    instrument_uid: str,
    resolution: Resolution,
    max_range_pct: Decimal = MAX_INTRABAR_RANGE_PCT,
) -> tuple[Finding, ...]:
    """A single bar spanning an implausible range.

    OHLC *ordering* is already enforced at construction; this catches the case
    where the ordering is fine but the high and low cannot have come from the
    same instrument as the open and close — a bad merge, or a tick size wrong
    by a factor.
    """
    findings: list[Finding] = []
    for bar in bars:
        span = (bar.high - bar.low) / bar.low * Decimal(100)
        if span > max_range_pct:
            findings.append(
                Finding(
                    kind=CheckKind.IMPLAUSIBLE_RANGE,
                    severity=Severity.WARN,
                    instrument_uid=instrument_uid,
                    resolution=resolution,
                    at=bar.bar_open_utc,
                    detail=(
                        f"one {resolution.value} bar spans {span:.1f}% "
                        f"(h={bar.high} l={bar.low}). Either the extremes are not from "
                        "this instrument, or the tick size is wrong."
                    ),
                    observed=f"{span:.1f}%",
                )
            )
    return tuple(findings)


def check_jumps(
    bars: Sequence[Bar],
    *,
    instrument_uid: str,
    resolution: Resolution,
    known_actions: Sequence[CorporateAction] = (),
    jump_pct: Decimal = JUMP_PCT,
) -> tuple[tuple[Finding, ...], tuple[SplitSuspicion, ...]]:
    """Large bar-to-bar moves, with an explanation attempted for each.

    The attempt is what makes this useful rather than noisy. A 50% gap that a
    recorded 2-for-1 split explains is not a data problem and is reported as
    `EXPLAINED_JUMP` at INFO. The same gap with no action behind it is either an
    unannounced split — which means the stored history has a discontinuity
    every feature will inherit — or a mismapped ticker. Both block entries.
    """
    findings: list[Finding] = []
    suspicions: list[SplitSuspicion] = []
    action_dates = {
        action.effective_date for action in known_actions if action.ratio_num is not None
    }

    previous: Bar | None = None
    for bar in bars:
        if previous is None:
            previous = bar
            continue
        move = abs(bar.open - previous.close) / previous.close * Decimal(100)
        if move < jump_pct:
            previous = bar
            continue

        session_date = bar.session_date
        if session_date in action_dates:
            findings.append(
                Finding(
                    kind=CheckKind.EXPLAINED_JUMP,
                    severity=Severity.INFO,
                    instrument_uid=instrument_uid,
                    resolution=resolution,
                    at=bar.bar_open_utc,
                    detail=(
                        f"{move:.1f}% gap into {session_date}, explained by a recorded "
                        "corporate action on that date"
                    ),
                    observed=f"{move:.1f}%",
                )
            )
            previous = bar
            continue

        suspicion = detect_unexplained_split(
            instrument_uid=instrument_uid,
            prev_close=previous.close,
            next_open=bar.open,
            effective_date=session_date,
            known_actions=known_actions,
        )
        if suspicion is not None:
            suspicions.append(suspicion)
            findings.append(
                Finding(
                    kind=CheckKind.UNEXPLAINED_JUMP,
                    severity=Severity.BLOCKING,
                    instrument_uid=instrument_uid,
                    resolution=resolution,
                    at=bar.bar_open_utc,
                    detail=suspicion.description,
                    observed=f"{move:.1f}%",
                    suggested_action=(
                        "block new entries in this symbol until a provider confirms the "
                        "action; do not adjust prices by the inferred ratio"
                    ),
                )
            )
        else:
            findings.append(
                Finding(
                    kind=CheckKind.UNEXPLAINED_JUMP,
                    severity=Severity.WARN,
                    instrument_uid=instrument_uid,
                    resolution=resolution,
                    at=bar.bar_open_utc,
                    detail=(
                        f"{move:.1f}% gap into {session_date} with no corporate action and "
                        "no simple split ratio that fits. Most likely a real move, but "
                        "large enough to be worth confirming against a second feed."
                    ),
                    observed=f"{move:.1f}%",
                )
            )
        previous = bar
    return tuple(findings), tuple(suspicions)


def check_provenance(
    bars: Sequence[Bar], *, instrument_uid: str, resolution: Resolution
) -> tuple[Finding, ...]:
    """Whether any bar in the series was ever observed live.

    A series entirely of backfilled rows has no real knowledge times — the
    as-of machinery is inert across it, because `available_at` was set to bar
    close rather than measured. That does not make the data unusable; it makes
    any backtest over it a vendor-current-view backtest, and the flag has to
    ride into the vintage rather than being discovered later.
    """
    if not bars:
        return ()
    if any(bar.provenance is Provenance.LIVE for bar in bars):
        return ()
    return (
        Finding(
            kind=CheckKind.BACKFILL_ONLY,
            severity=Severity.INFO,
            instrument_uid=instrument_uid,
            resolution=resolution,
            at=bars[0].bar_open_utc,
            detail=(
                f"all {len(bars)} bars are backfilled, so their knowledge times are "
                "assumed rather than observed. The as-of machinery is inert over this "
                "span: any backtest on it is a vendor-current-view backtest."
            ),
            suggested_action=("recorded on the vintage as pit_completeness=vendor_current_view"),
        ),
    )


# --------------------------------------------------------------------------
# Cross-symbol checks
# --------------------------------------------------------------------------


def check_frozen_feed(
    latest_by_uid: dict[str, Bar],
    *,
    now: datetime,
    limit_seconds: int,
    min_symbols: int = 3,
) -> tuple[Finding, ...]:
    """Every symbol's newest bar stuck at the same instant.

    The check per-symbol staleness cannot make. Each symbol individually looks
    like one stale name, which is unremarkable; all of them stale at the *same*
    timestamp means the provider stopped publishing. Distinguishing the two
    matters because the responses differ: one symbol is a data gap, the whole
    feed is a halt.
    """
    if len(latest_by_uid) < min_symbols:
        return ()

    stale = {
        uid: bar for uid, bar in latest_by_uid.items() if bar.age_seconds_at(now) > limit_seconds
    }
    if len(stale) != len(latest_by_uid):
        return ()

    instants = {bar.available_at_utc for bar in stale.values()}
    if len(instants) > 1:
        return ()

    frozen_at = next(iter(instants))
    age = (now - frozen_at).total_seconds()
    return (
        Finding(
            kind=CheckKind.FROZEN_FEED,
            severity=Severity.BLOCKING,
            at=frozen_at,
            detail=(
                f"all {len(stale)} symbols have the same newest knowledge time "
                f"{frozen_at.isoformat()}, {age:.0f}s ago. That is the provider having "
                "stopped, not the market — per-symbol staleness reads this as a set of "
                "individually plausible stale names."
            ),
            observed=f"{age:.0f}s",
            suggested_action="halt new entries until the feed resumes; exits are unaffected",
        ),
    )


def check_zero_volume_sessions(
    bars: Sequence[Bar],
    *,
    instrument_uid: str,
    resolution: Resolution,
    max_zero_fraction: float = 0.5,
) -> tuple[Finding, ...]:
    """A series where most bars have no volume at all.

    One zero-volume minute on a thin name is normal. Most of them means either
    the instrument does not really trade — in which case no slippage estimate
    over it is meaningful — or the feed is not reporting volume, in which case
    every liquidity-derived number is wrong.
    """
    measured = [bar for bar in bars if bar.volume is not None]
    if len(measured) < 20:
        return ()
    zero = sum(1 for bar in measured if bar.volume == 0)
    fraction = zero / len(measured)
    if fraction <= max_zero_fraction:
        return ()
    return (
        Finding(
            kind=CheckKind.ZERO_VOLUME,
            severity=Severity.WARN,
            instrument_uid=instrument_uid,
            resolution=resolution,
            detail=(
                f"{zero} of {len(measured)} bars have zero volume ({fraction:.0%}). "
                "Either this instrument barely trades, or the feed is not reporting "
                "volume — and every liquidity-derived number depends on knowing which."
            ),
            observed=f"{fraction:.0%}",
        ),
    )


def check_revision_clusters(
    revisions: Sequence[dict[str, Any]],
    *,
    window: timedelta = timedelta(days=1),
    min_cluster: int = 20,
) -> tuple[Finding, ...]:
    """Many revisions to one instrument at once.

    A trickle of revisions is expected — Yahoo back-adjusts history, and that is
    why `bar_revisions` exists. A *cluster* is different: it means a vendor
    rewrote a span wholesale, and every backtest that ran against the old values
    was validating a series that no longer exists.
    """
    by_uid: dict[str, list[datetime]] = {}
    for row in revisions:
        uid = str(row.get("instrument_uid") or "")
        revised = row.get("revised_at")
        if not uid or not revised:
            continue
        try:
            by_uid.setdefault(uid, []).append(from_iso(str(revised)))
        except ValueError:
            continue

    findings: list[Finding] = []
    for uid, moments in by_uid.items():
        moments.sort()
        for index, start in enumerate(moments):
            within = sum(1 for moment in moments[index:] if moment - start <= window)
            if within >= min_cluster:
                findings.append(
                    Finding(
                        kind=CheckKind.REVISION_CLUSTER,
                        severity=Severity.WARN,
                        instrument_uid=uid,
                        at=start,
                        detail=(
                            f"{within} revisions to {uid} within {window}. A trickle is "
                            "expected from a vendor that back-adjusts; a cluster means a "
                            "span was rewritten wholesale, so any backtest against the "
                            "old values validated a series that no longer exists."
                        ),
                        observed=f"{within} revisions",
                        suggested_action="re-seal the affected vintage and re-run promotion",
                    )
                )
                break
    return tuple(findings)
