"""Running the checks, writing the findings, emitting the event.

All the I/O that `checks.py` deliberately does not do. The split is what makes
the predicates testable against hand-built sequences instead of only against a
populated database — and the interesting cases (a half-day, a frozen feed, a
split nobody reported) are exactly the ones nobody builds a database for.

Three things this module is careful about.

**The coverage denominator is trading sessions, not calendar days.** Comparing
bars held against elapsed days reports every weekend as missing data, and the
real gaps disappear into the noise. `TradingCalendar.sessions_between` supplies
the denominator, and `classify_gap` explains each hole.

**`bar_coverage` is a projection, not a source.** It is rebuilt from the bars
on every audit rather than incremented. A counter that drifts from the files it
claims to summarise is worse than no counter, because it is consulted with
confidence.

**Store provenance is checked in both directions**, mirroring `tb ledger
verify`: a catalog row whose file is missing or whose hash has changed is a
hard failure, while an unrecorded file on disk is reported and harmless
because nothing reads it. The ledger defines the dataset; the directory does
not.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta
from decimal import Decimal
from typing import Any

from tb.broker.reconcile import Severity
from tb.core.clock import now_iso, now_utc
from tb.core.ids import new_id
from tb.data.actions import ActionStore
from tb.data.adjustments import SplitSuspicion
from tb.data.barstore import BarStore
from tb.data.calendar import TradingCalendar
from tb.data.checks import (
    CheckKind,
    CoverageReport,
    Finding,
    GapCause,
    check_coverage,
    check_frozen_feed,
    check_intrabar_range,
    check_jumps,
    check_knowledge_times,
    check_ordering,
    check_provenance,
    check_revision_clusters,
    check_stale_repeats,
    check_zero_volume_sessions,
)
from tb.data.provider import Bar, Resolution, dedupe_latest
from tb.ledger.events import Actor, DataAuditPayload, EventType
from tb.ledger.store import Ledger

# Below this average daily dollar volume a name is "thin" for gap-classification
# purposes: on a 2%-of-volume feed it genuinely has minutes with no print, and
# counting those against the feed makes the metric useless for the whole
# universe. Above it, a missing minute is a feed failure.
THIN_NAME_DOLLAR_VOLUME = Decimal(50_000_000)

# How many findings ride into the event payload. The full set goes to the
# report; the event carries a sample, because an audit over ten years of
# minute bars can produce thousands and burying the chain under them defeats
# the purpose of having a chain.
MAX_EVENT_FINDINGS = 50


@dataclass(slots=True)
class AuditReport:
    """Everything one audit found."""

    audit_id: str
    started_at: str
    resolutions: tuple[Resolution, ...] = field(default_factory=tuple)
    findings: list[Finding] = field(default_factory=list)
    coverage: list[CoverageReport] = field(default_factory=list)
    suspicions: list[SplitSuspicion] = field(default_factory=list)
    n_bars_checked: int = 0
    n_instruments: int = 0

    @property
    def blocking(self) -> list[Finding]:
        return [finding for finding in self.findings if finding.blocking]

    @property
    def clean(self) -> bool:
        """Whether anything found would stop trading.

        Warnings do not make an audit dirty. An audit that reports "not clean"
        for a thin name's zero-volume minutes would be ignored within a week,
        and then the blocking findings would be ignored with it.
        """
        return not self.blocking

    @property
    def gaps_unexplained(self) -> int:
        return sum(report.unexplained for report in self.coverage)

    @property
    def gaps_explained(self) -> int:
        return sum(report.explained for report in self.coverage)

    def by_kind(self) -> dict[str, int]:
        counts: dict[str, int] = {}
        for finding in self.findings:
            counts[finding.kind.value] = counts.get(finding.kind.value, 0) + 1
        return counts

    def worst_coverage(self, limit: int = 5) -> list[CoverageReport]:
        return sorted(self.coverage, key=lambda r: -r.unexplained_pct)[:limit]


class DataAuditor:
    """Runs every check over the store and records the outcome."""

    def __init__(
        self,
        ledger: Ledger,
        store: BarStore,
        *,
        calendar: TradingCalendar | None = None,
        actions: ActionStore | None = None,
        max_unexplained_gap_pct: float = 1.0,
        max_provider_delay_seconds: int = 180,
        run_id: str | None = None,
    ) -> None:
        self._ledger = ledger
        self._store = store
        self._calendar = calendar or TradingCalendar()
        self._actions = actions or ActionStore(ledger)
        self._max_gap_pct = max_unexplained_gap_pct
        self._max_delay = max_provider_delay_seconds
        self._run_id = run_id

    def run(
        self,
        *,
        resolutions: Sequence[Resolution] = (Resolution.DAILY,),
        instrument_uids: Sequence[str] | None = None,
        start: date | None = None,
        end: date | None = None,
        dollar_volume: dict[str, Decimal] | None = None,
        halted_dates: dict[str, Iterable[date]] | None = None,
        now: datetime | None = None,
        emit_event: bool = True,
    ) -> AuditReport:
        """Audit the store and record what it found.

        `start`/`end` default to the span the store actually holds rather than
        to a fixed window: auditing a window the store was never asked to cover
        would report the whole of it as missing, which is a true statement that
        says nothing about the feed.
        """
        moment = now or now_utc()
        report = AuditReport(
            audit_id=new_id("audit", length=12),
            started_at=now_iso(),
            resolutions=tuple(resolutions),
        )

        uids = list(instrument_uids) if instrument_uids else list(self._store.instruments())
        report.n_instruments = len(uids)
        volumes = dollar_volume or {}
        halts = halted_dates or {}

        for resolution in resolutions:
            latest_by_uid: dict[str, Bar] = {}
            for uid in uids:
                bars = dedupe_latest(self._store.bars_for(uid, resolution))
                if not bars:
                    continue
                report.n_bars_checked += len(bars)
                if bars:
                    latest_by_uid[uid] = bars[-1]

                window_start = start or bars[0].bar_open_utc.date()
                window_end = end or bars[-1].bar_open_utc.date()
                thin = volumes.get(uid, Decimal(0)) < THIN_NAME_DOLLAR_VOLUME

                coverage, coverage_findings = check_coverage(
                    bars,
                    instrument_uid=uid,
                    resolution=resolution,
                    start=window_start,
                    end=window_end,
                    calendar=self._calendar,
                    max_unexplained_pct=self._max_gap_pct,
                    halted_dates=halts.get(uid, ()),
                    thin_name=thin,
                )
                report.coverage.append(coverage)
                report.findings.extend(coverage_findings)

                report.findings.extend(
                    check_ordering(bars, instrument_uid=uid, resolution=resolution)
                )
                report.findings.extend(
                    check_knowledge_times(bars, instrument_uid=uid, resolution=resolution)
                )
                report.findings.extend(
                    check_stale_repeats(bars, instrument_uid=uid, resolution=resolution)
                )
                report.findings.extend(
                    check_intrabar_range(bars, instrument_uid=uid, resolution=resolution)
                )
                report.findings.extend(
                    check_zero_volume_sessions(bars, instrument_uid=uid, resolution=resolution)
                )
                report.findings.extend(
                    check_provenance(bars, instrument_uid=uid, resolution=resolution)
                )

                if resolution is Resolution.DAILY:
                    # The jump detector runs on daily bars only. At minute
                    # resolution every overnight gap is a "jump" and the signal
                    # drowns; and on an adjusted series the jump has already
                    # been removed, so a detector that never fires reads as
                    # evidence of absence.
                    jump_findings, suspicions = check_jumps(
                        bars,
                        instrument_uid=uid,
                        resolution=resolution,
                        known_actions=self._actions.actions_for(uid, as_of=moment),
                    )
                    report.findings.extend(jump_findings)
                    report.suspicions.extend(suspicions)

                self._write_coverage(coverage, bars)

            report.findings.extend(
                check_frozen_feed(latest_by_uid, now=moment, limit_seconds=self._max_delay)
            )

        report.findings.extend(check_revision_clusters(self._recent_revisions()))
        report.findings.extend(self._check_partitions())

        if emit_event:
            self._emit(report)
        return report

    # -- store provenance --------------------------------------------------

    def _check_partitions(self) -> tuple[Finding, ...]:
        """Catalog against filesystem, both directions.

        The data-layer analogue of `tb ledger verify`, with the same asymmetric
        reading. `BarStore.verify_partitions` returns text findings; they are
        classified here so a missing file and a stray file get different
        severities — one is data the ledger promised and cannot produce, the
        other is garbage nothing reads.
        """
        findings: list[Finding] = []
        for text in self._store.verify_partitions():
            lowered = text.lower()
            if "missing" in lowered:
                kind, severity = CheckKind.PARTITION_MISSING, Severity.BLOCKING
            elif "altered" in lowered or "hash" in lowered:
                kind, severity = CheckKind.PARTITION_ALTERED, Severity.BLOCKING
            else:
                kind, severity = CheckKind.PARTITION_ORPHAN, Severity.INFO
            findings.append(
                Finding(
                    kind=kind,
                    severity=severity,
                    detail=text,
                    suggested_action=(
                        "the ledger names a file the store cannot produce; restore from "
                        "backup or re-seal the window"
                        if severity is Severity.BLOCKING
                        else "an unrecorded file; nothing reads it, so it is safe to delete"
                    ),
                )
            )
        return tuple(findings)

    def _recent_revisions(self, limit: int = 5000) -> list[dict[str, Any]]:
        rows = self._ledger.conn.execute(
            "SELECT instrument_uid, resolution, bar_open_utc, kind, revised_at "
            "FROM bar_revisions ORDER BY revised_at DESC LIMIT ?",
            (limit,),
        ).fetchall()
        return [{key: row[key] for key in row.keys()} for row in rows]  # noqa: SIM118

    # -- the coverage projection ------------------------------------------

    def _write_coverage(self, coverage: CoverageReport, bars: Sequence[Bar]) -> None:
        """Rebuild this instrument's coverage row from the bars themselves.

        Replaced wholesale rather than incremented. `bar_coverage` is declared a
        projection precisely so it can be rebuilt; a counter that drifts from
        the files it summarises is worse than no counter, because it gets
        trusted.
        """
        providers = {bar.provider for bar in bars}
        for provider in providers:
            subset = [bar for bar in bars if bar.provider == provider]
            self._ledger.conn.execute(
                """
                INSERT INTO bar_coverage (
                    instrument_uid, resolution, provider, first_bar_open, last_bar_open,
                    row_count, n_gaps_unexplained, last_audited_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(instrument_uid, resolution, provider) DO UPDATE SET
                    first_bar_open = excluded.first_bar_open,
                    last_bar_open = excluded.last_bar_open,
                    row_count = excluded.row_count,
                    n_gaps_unexplained = excluded.n_gaps_unexplained,
                    last_audited_at = excluded.last_audited_at
                """,
                (
                    coverage.instrument_uid,
                    coverage.resolution.value,
                    provider,
                    subset[0].bar_open_utc.isoformat(),
                    subset[-1].bar_open_utc.isoformat(),
                    len(subset),
                    coverage.unexplained,
                    now_iso(),
                ),
            )
        self._ledger.conn.commit()

    # -- the event ---------------------------------------------------------

    def _emit(self, report: AuditReport) -> None:
        self._ledger.append(
            EventType.DATA_AUDIT_COMPLETED,
            report.audit_id,
            DataAuditPayload(
                audit_id=report.audit_id,
                n_instruments=report.n_instruments,
                n_bars_checked=report.n_bars_checked,
                n_findings=len(report.findings),
                n_blocking=len(report.blocking),
                gaps_unexplained=report.gaps_unexplained,
                gaps_explained=report.gaps_explained,
                # Blocking findings first, so a truncated sample never drops the
                # ones that stop trading in favour of a thin name's flat minutes.
                findings=[
                    finding.as_dict()
                    for finding in sorted(report.findings, key=lambda f: 0 if f.blocking else 1)[
                        :MAX_EVENT_FINDINGS
                    ]
                ],
            ),
            actor=Actor.SYSTEM,
            run_id=self._run_id,
        )

    # -- reading the projection back --------------------------------------

    def coverage_rows(self) -> list[dict[str, Any]]:
        rows = self._ledger.conn.execute(
            "SELECT * FROM bar_coverage ORDER BY instrument_uid, resolution, provider"
        ).fetchall()
        return [{key: row[key] for key in row.keys()} for row in rows]  # noqa: SIM118

    def last_audit_at(self) -> str | None:
        row = self._ledger.conn.execute(
            "SELECT MAX(last_audited_at) AS latest FROM bar_coverage"
        ).fetchone()
        return None if row is None or row["latest"] is None else str(row["latest"])


def summarise_gaps(reports: Iterable[CoverageReport]) -> dict[str, int]:
    """Total gaps by cause across a set of coverage reports.

    The shape `tb data audit` prints. Presented per cause rather than as a
    total because the total is the number that gets ignored: only `unexplained`
    is a statement about the feed.
    """
    totals: dict[str, int] = {}
    for report in reports:
        for cause, count in report.by_cause.items():
            totals[cause.value] = totals.get(cause.value, 0) + count
    return dict(sorted(totals.items()))


def staleness_window(resolution: Resolution, *, limit_seconds: int) -> timedelta:
    """How far back a live decision may look for this resolution.

    Named here rather than inlined because both the audit and the live loop
    need the same number, and two copies of a staleness bound diverge in the
    permissive direction.
    """
    return timedelta(seconds=max(limit_seconds, resolution.seconds))


def gap_cause_of(finding: Finding) -> GapCause | None:
    return finding.cause
