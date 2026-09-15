"""Deliberately looking at old history, because nothing else ever does.

Revision detection in `BarStore.ingest` only fires on bars that get *refetched*.
The incremental poll refetches the last few sessions, so a restatement there is
caught within a day. A restatement twenty days back is invisible — not
unreported, invisible. Nothing in the system would ever re-read that window, so
the stored bar and the vendor's current answer can diverge indefinitely and no
check would notice.

That is not a hypothetical. Yahoo back-adjusts historical OHLC as a matter of
course, including for corporate actions it never reports as actions, which is
precisely why `returns_raw_prices=False` on that provider. The failure mode is
a backtest that cannot be reproduced next month and gives no reason.

So this resamples stored history on purpose and re-compares.

**Selection is least-recently-verified, not random.** A random sample tells you
nothing about coverage: successive runs re-roll the same dice, some windows are
checked five times and others never, and "we sampled and found nothing" cannot
be distinguished from "we never looked there". Ordering by last-checked instead
gives three things a sample does not — every window is eventually reached, the
oldest unverified history is reached first (which is exactly where the
invisible restatements are), and the whole thing is deterministic and
auditable without a seed.

`data.revision_canary_sample_pct` is therefore a **budget**, not a probability:
what fraction of live partitions to spend provider quota on per run.

**The comparison reuses `ingest`.** The diff, the `bar_revisions` rows and the
delta arithmetic already exist and are tested; a second implementation here
would be a second answer to "did this bar change". The only difference is the
`detected_by` column, which the schema already anticipated — `"ingest"` versus
`"canary"` is how you tell a restatement someone happened to walk into from
one that was hunted.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any

from tb.core.clock import now_utc
from tb.core.errors import TbError
from tb.core.ids import new_id
from tb.data.barstore import BarStore, PartitionInfo, Revision
from tb.data.provider import MarketDataProvider, Provenance, Resolution
from tb.ledger.events import Actor, EventType, RevisionCanaryPayload
from tb.ledger.store import Ledger


class CanaryError(TbError):
    """The canary could not run, which is different from finding nothing."""


@dataclass(frozen=True, slots=True)
class CanaryWindow:
    """One span of stored history due for re-verification."""

    instrument_uid: str
    resolution: Resolution
    symbol: str
    provider: str
    start: datetime
    end: datetime
    row_count: int
    last_checked_at: datetime | None = None

    @property
    def never_checked(self) -> bool:
        return self.last_checked_at is None

    def age_days_at(self, now: datetime) -> float:
        """How long since this window was verified, or since it was sealed."""
        reference = self.last_checked_at or self.end
        return (now - reference).total_seconds() / 86400.0


@dataclass(frozen=True, slots=True)
class CanaryResult:
    """What one canary run looked at and what it found.

    `n_windows_available` against `n_windows_checked` is the honest coverage
    statement: a run that checked 2 of 400 windows found nothing about the
    other 398, and reporting only the finding count would imply otherwise.
    """

    canary_id: str
    resolution: Resolution
    n_windows_available: int
    n_windows_checked: int
    n_bars_compared: int
    revisions: tuple[Revision, ...] = field(default_factory=tuple)
    problems: tuple[str, ...] = field(default_factory=tuple)
    oldest_checked_age_days: float | None = None

    @property
    def n_revisions_found(self) -> int:
        return len(self.revisions)

    @property
    def clean(self) -> bool:
        """No restatements found. Says nothing about windows not checked."""
        return not self.revisions

    @property
    def coverage_pct(self) -> float:
        if not self.n_windows_available:
            return 0.0
        return self.n_windows_checked / self.n_windows_available * 100.0

    def instruments_restated(self) -> tuple[str, ...]:
        return tuple(sorted({r.instrument_uid for r in self.revisions}))

    def summary(self) -> str:
        return (
            f"checked {self.n_windows_checked}/{self.n_windows_available} windows "
            f"({self.coverage_pct:.0f}% of stored history), compared "
            f"{self.n_bars_compared} bars, found {self.n_revisions_found} "
            f"restatement(s)"
        )


class RevisionCanary:
    """Resamples stored history and records what the vendor now says.

    Takes a provider per call rather than holding one, so the selection logic
    is testable without a network and so a caller can canary one provider's
    history without constructing the others.
    """

    def __init__(
        self,
        ledger: Ledger,
        store: BarStore,
        *,
        sample_pct: float,
        run_id: str | None = None,
    ) -> None:
        if not 0.0 <= sample_pct <= 100.0:
            raise CanaryError(f"sample_pct must be a percentage, got {sample_pct}")
        self._ledger = ledger
        self._store = store
        self._sample_pct = sample_pct
        self._run_id = run_id

    # -- selection ---------------------------------------------------------

    def budget(self, n_available: int) -> int:
        """How many windows this run may spend quota on.

        Rounded *up* to at least one whenever anything is stored and the
        percentage is non-zero. A budget that rounded to zero on a small store
        would mean the canary silently never ran on exactly the deployments
        where history is shortest and a restatement matters most.
        """
        if n_available == 0 or self._sample_pct == 0.0:
            return 0
        return max(1, round(n_available * self._sample_pct / 100.0))

    def windows(self, *, resolution: Resolution) -> list[CanaryWindow]:
        """Every live partition, ordered least-recently-verified first.

        The ordering is the whole design — see the module docstring. Ties
        (typically "never checked") break on the *oldest* history, since that
        is where a restatement has had longest to go unnoticed.
        """
        checked = self._last_checked(resolution)
        found: list[CanaryWindow] = []
        for partition in self._store.live_partitions(resolution=resolution):
            key = (partition.instrument_uid, partition.first_bar_open.isoformat())
            found.append(
                CanaryWindow(
                    instrument_uid=partition.instrument_uid,
                    resolution=partition.resolution,
                    symbol=self._symbol_for(partition),
                    provider=partition.provider,
                    start=partition.first_bar_open,
                    end=partition.last_bar_open,
                    row_count=partition.row_count,
                    last_checked_at=checked.get(key),
                )
            )
        # None sorts first via the sentinel, then oldest-checked, then oldest
        # history. Fully deterministic: no seed, and two runs over the same
        # store pick the same windows.
        found.sort(
            key=lambda w: (
                w.last_checked_at or datetime.min.replace(tzinfo=UTC),
                w.start,
                w.instrument_uid,
            )
        )
        return found

    def select(self, *, resolution: Resolution) -> list[CanaryWindow]:
        """The windows this run will actually re-verify."""
        available = self.windows(resolution=resolution)
        return available[: self.budget(len(available))]

    # -- the run -----------------------------------------------------------

    def run(
        self,
        feed: MarketDataProvider,
        *,
        resolution: Resolution = Resolution.DAILY,
        now: datetime | None = None,
        emit_event: bool = True,
    ) -> CanaryResult:
        """Refetch the selected windows and record any divergence.

        Every fetch failure is collected as a problem rather than raised: a
        provider being briefly unreachable is not a reason to abandon the
        windows that did come back, and a canary that aborted on the first
        error would in practice never complete a run.
        """
        moment = now or now_utc()
        available = self.windows(resolution=resolution)
        chosen = available[: self.budget(len(available))]

        revisions: list[Revision] = []
        problems: list[str] = []
        n_bars = 0

        for window in chosen:
            if not window.symbol:
                problems.append(
                    f"{window.instrument_uid}: no symbol recorded for this partition, so "
                    "it cannot be refetched. Its history is unverifiable."
                )
                self._record_check(window, moment, outcome="unverifiable", n_bars=0, found=0)
                continue
            try:
                batch = feed.fetch_bars(
                    window.symbol,
                    instrument_uid=window.instrument_uid,
                    resolution=window.resolution,
                    start=window.start,
                    end=window.end,
                    # `BACKFILL`, because that is what it is: a re-read of old
                    # history, not a live observation. Marking it `LIVE` would
                    # put a fabricated knowledge time on a bar that was
                    # actually first seen months ago.
                    provenance=Provenance.BACKFILL,
                )
            except Exception as exc:
                problems.append(f"{window.symbol} {window.start.date()}: {exc}")
                self._record_check(window, moment, outcome="fetch_failed", n_bars=0, found=0)
                continue

            result = self._store.ingest(batch)
            n_bars += len(batch.bars)
            revisions.extend(result.revisions)
            self._record_check(
                window,
                moment,
                outcome="clean" if not result.revisions else "restated",
                n_bars=len(batch.bars),
                found=len(result.revisions),
            )

        outcome = CanaryResult(
            canary_id=new_id("canary", length=12),
            resolution=resolution,
            n_windows_available=len(available),
            n_windows_checked=len(chosen),
            n_bars_compared=n_bars,
            revisions=tuple(revisions),
            problems=tuple(problems),
            oldest_checked_age_days=(
                max((w.age_days_at(moment) for w in chosen), default=None) if chosen else None
            ),
        )
        if emit_event:
            self._emit(outcome)
        return outcome

    # -- persistence -------------------------------------------------------

    def _last_checked(self, resolution: Resolution) -> dict[tuple[str, str], datetime]:
        rows = self._ledger.conn.execute(
            "SELECT instrument_uid, window_start, MAX(checked_at) AS checked_at "
            "FROM canary_checks WHERE resolution = ? "
            "GROUP BY instrument_uid, window_start",
            (resolution.value,),
        ).fetchall()
        found: dict[tuple[str, str], datetime] = {}
        for row in rows:
            try:
                found[(row["instrument_uid"], row["window_start"])] = datetime.fromisoformat(
                    row["checked_at"]
                )
            except (TypeError, ValueError):  # pragma: no cover - defensive
                continue
        return found

    def _symbol_for(self, partition: PartitionInfo) -> str:
        """The provider symbol this partition was fetched under.

        From `instrument_symbols`, which the store records on ingest. Not
        derived from the uid: an `isin:` uid does not carry a ticker, and
        guessing one is the mismapping the symbol map exists to prevent — a
        refetch of the wrong company's prices would be recorded as a
        restatement of this one.
        """
        found = self._store.symbol_for(partition.instrument_uid, partition.provider)
        if found:
            return found
        if partition.instrument_uid.startswith("sym:"):
            return partition.instrument_uid.removeprefix("sym:")
        return ""

    def _record_check(
        self,
        window: CanaryWindow,
        moment: datetime,
        *,
        outcome: str,
        n_bars: int,
        found: int,
    ) -> None:
        """Append-only, so coverage history is not overwritten by the newest run.

        "This window has been checked four times and never restated" is a
        different and more useful fact than "last checked on Tuesday", and an
        upsert would throw the first away.
        """
        self._ledger.conn.execute(
            """
            INSERT INTO canary_checks (
                check_id, instrument_uid, resolution, provider, window_start,
                window_end, checked_at, outcome, n_bars_compared, n_revisions_found,
                run_id
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                new_id("cchk", length=12),
                window.instrument_uid,
                window.resolution.value,
                window.provider,
                window.start.isoformat(),
                window.end.isoformat(),
                moment.isoformat(),
                outcome,
                n_bars,
                found,
                self._run_id,
            ),
        )
        self._ledger.conn.commit()

    def _emit(self, result: CanaryResult) -> None:
        self._ledger.append(
            EventType.DATA_REVISION_CANARY_COMPLETED,
            result.canary_id,
            RevisionCanaryPayload(
                canary_id=result.canary_id,
                resolution=result.resolution.value,
                n_windows_available=result.n_windows_available,
                n_windows_checked=result.n_windows_checked,
                n_bars_compared=result.n_bars_compared,
                n_revisions_found=result.n_revisions_found,
                coverage_pct=result.coverage_pct,
                oldest_checked_age_days=result.oldest_checked_age_days,
                instruments_restated=list(result.instruments_restated()),
                problems=list(result.problems[:20]),
            ),
            actor=Actor.SYSTEM,
            run_id=self._run_id,
        )

    # -- reporting ---------------------------------------------------------

    def coverage(self, *, resolution: Resolution) -> list[dict[str, Any]]:
        """Per-window verification history, for `tb data canary --report`."""
        rows = self._ledger.conn.execute(
            """
            SELECT instrument_uid, window_start, window_end,
                   COUNT(*) AS n_checks,
                   MAX(checked_at) AS last_checked_at,
                   SUM(n_revisions_found) AS total_revisions,
                   SUM(n_bars_compared) AS total_bars
            FROM canary_checks WHERE resolution = ?
            GROUP BY instrument_uid, window_start
            ORDER BY last_checked_at DESC
            """,
            (resolution.value,),
        ).fetchall()
        return [{key: row[key] for key in row.keys()} for row in rows]  # noqa: SIM118

    def last_run_at(self) -> str | None:
        row = self._ledger.conn.execute(
            "SELECT MAX(checked_at) AS last FROM canary_checks"
        ).fetchone()
        return None if row is None else row["last"]


def unverifiable_windows(
    windows: Sequence[CanaryWindow], *, now: datetime, stale_after_days: float
) -> tuple[CanaryWindow, ...]:
    """Windows nobody has looked at in too long.

    The number that says whether the canary is keeping up. At a 2% budget and
    one run a day, a 400-partition store takes fifty days to cover once — so a
    window unverified for months means the budget is too small for the store
    it is guarding, not that the data is fine.
    """
    return tuple(w for w in windows if w.age_days_at(now) > stale_after_days)
