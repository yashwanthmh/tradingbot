"""Sealed dataset vintages: what M3 is handed, and nothing else.

M3 does not get the store. It gets a `DatasetVintage` — a named, hashed set of
Parquet file hashes plus the hashes of everything else a backtest silently
depends on — and a forward-only reader over exactly those files.

**The rule that gives this teeth: a backtest whose `vintage_id` is not in the
ledger is not admissible evidence for promotion.** Without it, "we backtested
this and it worked" is unfalsifiable, because the data it worked on has since
been restated underneath it. Yahoo back-adjusts history as a matter of course,
so this is not a hypothetical — re-running the same backtest next month against
"the store" legitimately produces different numbers, and nothing records why.

Four things a vintage pins that are easy to forget, each of which changes a
backtest's results on its own:

* **The bar files**, by content hash. Loading a vintage verifies every hash and
  refuses if one differs. That refusal is the immutability guarantee.
* **The corporate-action table.** A restated split ratio changes every adjusted
  price before its ex-date.
* **The FX table.** The limits are GBP and the universe is USD, so a revised
  rate changes every position size and every P&L figure.
* **The calendar and the universe snapshot.** Which days existed, and which
  names were selectable.

And two flags that ride into M5's promotion record rather than being discovered
afterwards:

* `survivorship_flag` — whether dated universe snapshots actually cover the
  window, or whether it is today's survivors backtested over history.
* `pit_completeness_flag` — whether knowledge times were *observed* or assumed.
  A vintage built entirely from backfill has an inert as-of machinery: the
  values are the vendor's current view, and `available_at` was set to bar close
  rather than measured. The backtest is still useful; calling it point-in-time
  would be a lie.

**Sealing compacts first.** A vintage references only sealed Parquet, so hot
rows staged in the ledger have to be sealed before the manifest is taken —
otherwise the snapshot is silently missing the newest data while the catalog
looks complete.
"""

from __future__ import annotations

import json
import sqlite3
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime
from enum import StrEnum

from tb.core.canonical import hash_payload
from tb.core.clock import from_iso, now_utc, to_iso
from tb.data.actions import ActionStore
from tb.data.adjustments import CorporateAction
from tb.data.asof import ForwardOnlyReader, InMemoryBarSource
from tb.data.barstore import BarStore, PartitionInfo
from tb.data.calendar import TradingCalendar
from tb.data.fx import FxStore
from tb.data.provider import Bar, DataError, Provenance, Resolution
from tb.data.universe import SurvivorshipFlag, UniverseStore
from tb.ledger.events import Actor, EventType, SnapshotSealedPayload
from tb.ledger.store import Ledger


class SnapshotError(DataError):
    """A vintage could not be sealed or loaded."""


class VintageAlteredError(SnapshotError):
    """A sealed vintage's files no longer hash to what was recorded.

    Fatal and never worked around. The whole value of a vintage is that it is
    the same data it was when the backtest ran; a vintage that loads with
    different bytes is not that data, and silently continuing would make every
    promotion decision resting on it unfalsifiable.
    """


class PitCompleteness(StrEnum):
    """How much of a vintage's knowledge time was measured rather than assumed.

    Three values because the difference is the difference between a real
    point-in-time backtest and a plausible one, and only a label keeps them
    apart once the numbers are in a spreadsheet.
    """

    # Every bar was observed arriving: `available_at` is a measurement.
    POINT_IN_TIME = "point_in_time"
    # Some live observation, but the window predates most of it.
    PARTIAL = "partial"
    # Entirely backfilled: the vendor's *current* view, with knowledge times
    # set to bar close because nothing was there to measure them.
    VENDOR_CURRENT_VIEW = "vendor_current_view"

    @property
    def is_point_in_time(self) -> bool:
        return self is PitCompleteness.POINT_IN_TIME


@dataclass(frozen=True, slots=True)
class DatasetVintage:
    """An immutable, named dataset. M3's entire input.

    `manifest_hash` covers the file hashes *and* the action, FX, calendar and
    universe hashes together, so two vintages with the same bars but a restated
    split ratio are different vintages — which they are, because they produce
    different adjusted prices.
    """

    vintage_id: str
    as_of_utc: datetime
    manifest_hash: str
    file_sha256s: tuple[str, ...]
    instrument_uids: tuple[str, ...]
    resolutions: tuple[Resolution, ...]
    window_start: datetime | None
    window_end: datetime | None
    row_count: int
    calendar_hash: str | None = None
    action_table_hash: str | None = None
    fx_table_hash: str | None = None
    universe_snapshot_id: str | None = None
    provider_delays: dict[str, float] = field(default_factory=dict)
    sealed_from: datetime | None = None
    first_live_observation_at: datetime | None = None
    survivorship_flag: SurvivorshipFlag = SurvivorshipFlag.UNMEASURED
    pit_completeness_flag: PitCompleteness = PitCompleteness.VENDOR_CURRENT_VIEW
    survivorship_note: str = ""

    @property
    def n_files(self) -> int:
        return len(self.file_sha256s)

    @property
    def admissible_for_promotion(self) -> bool:
        """Whether this vintage alone supports a promotion decision.

        Deliberately permissive about `pit_completeness`: on free data a
        ten-year window is *necessarily* a vendor-current-view backtest, so
        refusing those would refuse the only evidence available. What is not
        permitted is a vintage with no rows, which is a backtest over nothing.
        """
        return self.row_count > 0

    @property
    def caveats(self) -> tuple[str, ...]:
        """Everything a promotion decision should be weighed against.

        Assembled here so M5 cannot forget one: the flags are on the object, but
        a caller reading `survivorship_flag` and not `pit_completeness_flag` has
        seen half the problem.
        """
        notes: list[str] = []
        if self.survivorship_flag is not SurvivorshipFlag.MEASURED:
            notes.append(f"survivorship: {self.survivorship_flag.value} — {self.survivorship_note}")
        if not self.pit_completeness_flag.is_point_in_time:
            notes.append(
                f"point-in-time: {self.pit_completeness_flag.value} — knowledge times over "
                "this window are assumed rather than observed, so the as-of machinery is "
                "inert across it and the values are the vendor's current view"
            )
        if self.fx_table_hash is None:
            notes.append(
                "no FX table was pinned: any figure in the account currency depends on "
                "rates that are free to change underneath this vintage"
            )
        return tuple(notes)

    def manifest(self) -> dict[str, object]:
        """The canonical dict the manifest hash is taken over.

        Sorted, so the hash does not depend on the order the catalog happened
        to return rows in — a manifest hash that varied with SQL ordering would
        make the same dataset seal to a different vintage each time.
        """
        return {
            "file_sha256s": sorted(self.file_sha256s),
            "instrument_uids": sorted(self.instrument_uids),
            "resolutions": sorted(r.value for r in self.resolutions),
            "window_start": None if self.window_start is None else to_iso(self.window_start),
            "window_end": None if self.window_end is None else to_iso(self.window_end),
            "row_count": self.row_count,
            "calendar_hash": self.calendar_hash,
            "action_table_hash": self.action_table_hash,
            "fx_table_hash": self.fx_table_hash,
            "universe_snapshot_id": self.universe_snapshot_id,
            "provider_delays": dict(sorted(self.provider_delays.items())),
            "survivorship_flag": self.survivorship_flag.value,
            "pit_completeness_flag": self.pit_completeness_flag.value,
        }


def calendar_hash(calendar: TradingCalendar) -> str:
    """A hash of the calendar's own content.

    Which days existed changes a backtest's results, so a calendar edit has to
    produce a different vintage. Covers the coverage bounds too: the same
    holiday list over a different range answers differently about 2031.
    """
    start, end = calendar.coverage
    return hash_payload(
        {
            "coverage": [start.isoformat(), end.isoformat()],
            "uses_broker_schedule": calendar.uses_broker_schedule,
            "sessions": [day.day.isoformat() for day in calendar.sessions_between(start, end)],
            "half_days": [
                day.day.isoformat()
                for day in calendar.sessions_between(start, end)
                if day.kind.value == "half_day"
            ],
        }
    )


class SnapshotStore:
    """Seals vintages and loads them back."""

    def __init__(
        self,
        ledger: Ledger,
        store: BarStore,
        *,
        calendar: TradingCalendar | None = None,
        actions: ActionStore | None = None,
        fx: FxStore | None = None,
        universe: UniverseStore | None = None,
        run_id: str | None = None,
    ) -> None:
        self._ledger = ledger
        self._store = store
        self._calendar = calendar or TradingCalendar()
        self._actions = actions or ActionStore(ledger)
        self._fx = fx or FxStore(ledger)
        self._universe = universe or UniverseStore(ledger)
        self._run_id = run_id

    # -- sealing -----------------------------------------------------------

    def seal(
        self,
        *,
        resolutions: Sequence[Resolution] = (Resolution.DAILY,),
        instrument_uids: Sequence[str] | None = None,
        provider_delays: dict[str, float] | None = None,
        compact_first: bool = True,
        as_of: datetime | None = None,
    ) -> DatasetVintage:
        """Freeze the current dataset as a named, hashed vintage.

        `compact_first` defaults to True and should stay that way. A vintage
        names sealed Parquet files; rows still staged in `bars_hot` are not in
        any file, so sealing without compacting produces a snapshot missing the
        newest data — silently, because the catalog looks complete and the row
        count looks plausible.
        """
        if compact_first:
            self._store.compact()
        elif self._store.has_hot_rows():
            raise SnapshotError(
                "there are staged rows in bars_hot and compact_first=False. A vintage "
                "references only sealed Parquet, so sealing now would omit the newest "
                "bars without saying so. Compact first, or accept the default."
            )

        moment = as_of or now_utc()
        wanted = set(instrument_uids) if instrument_uids else None

        partitions: list[PartitionInfo] = []
        for resolution in resolutions:
            for info in self._store.live_partitions(resolution=resolution):
                if wanted is not None and info.instrument_uid not in wanted:
                    continue
                partitions.append(info)

        if not partitions:
            raise SnapshotError(
                "nothing to seal: no live partitions match those resolutions and "
                "instruments. Run `tb data backfill` first — an empty vintage would be "
                "admissible-looking evidence for a backtest over no data."
            )

        uids = sorted({info.instrument_uid for info in partitions})
        window_start = min(info.first_bar_open for info in partitions)
        window_end = max(info.last_bar_open for info in partitions)
        row_count = sum(info.row_count for info in partitions)

        completeness, first_live = self._assess_pit(partitions)
        survivorship, note = self._universe.survivorship_for(
            window_start=window_start, window_end=window_end
        )
        snapshot = self._universe.as_of(moment)

        vintage = DatasetVintage(
            # Filled in below, once the manifest hash exists: the id is derived
            # from the content, so two seals of identical data are the same
            # vintage rather than two indistinguishable ones.
            vintage_id="",
            as_of_utc=moment,
            manifest_hash="",
            file_sha256s=tuple(sorted(info.file_sha256 for info in partitions)),
            instrument_uids=tuple(uids),
            resolutions=tuple(resolutions),
            window_start=window_start,
            window_end=window_end,
            row_count=row_count,
            calendar_hash=calendar_hash(self._calendar),
            action_table_hash=self._action_table_hash(),
            fx_table_hash=self._fx.table_hash(),
            universe_snapshot_id=None if snapshot is None else snapshot.snapshot_id,
            provider_delays=dict(provider_delays or {}),
            sealed_from=moment,
            first_live_observation_at=first_live,
            survivorship_flag=survivorship,
            pit_completeness_flag=completeness,
            survivorship_note=note,
        )
        manifest_hash = hash_payload(vintage.manifest())
        vintage = _with_identity(vintage, manifest_hash=manifest_hash)

        existing = self.get(vintage.vintage_id)
        if existing is not None:
            # Identical content: the same vintage, not a second one. Returning
            # it keeps `tb data seal` idempotent, which matters because it is
            # the command someone runs twice when unsure whether it worked.
            return existing

        self._record(vintage)
        return vintage

    def _assess_pit(
        self, partitions: Sequence[PartitionInfo]
    ) -> tuple[PitCompleteness, datetime | None]:
        """How much of this data was observed arriving, rather than backfilled.

        Read from the bars themselves rather than from a config flag: a vintage
        that *claims* to be point-in-time because someone set a flag is exactly
        the artefact this whole layer exists to prevent.
        """
        live_total = 0
        row_total = 0
        first_live: datetime | None = None
        for info in partitions:
            for bar in self._store.read_partition(info.relative_path):
                row_total += 1
                if bar.provenance is Provenance.LIVE:
                    live_total += 1
                    if first_live is None or bar.available_at_utc < first_live:
                        first_live = bar.available_at_utc

        if row_total == 0 or live_total == 0:
            return PitCompleteness.VENDOR_CURRENT_VIEW, None
        if live_total == row_total:
            return PitCompleteness.POINT_IN_TIME, first_live
        return PitCompleteness.PARTIAL, first_live

    def _action_table_hash(self) -> str:
        rows = self._ledger.conn.execute(
            "SELECT action_id, instrument_uid, action_type, effective_date, known_at_utc, "
            "ratio_num, ratio_den, gross_amount, superseded_by FROM corporate_actions "
            "ORDER BY action_id"
        ).fetchall()
        return hash_payload(
            [
                {key: (None if row[key] is None else str(row[key])) for key in row.keys()}  # noqa: SIM118
                for row in rows
            ]
        )

    def _record(self, vintage: DatasetVintage) -> None:
        with self._ledger.transaction() as tx:
            event = tx.append(
                EventType.DATA_SNAPSHOT_SEALED,
                vintage.vintage_id,
                SnapshotSealedPayload(
                    vintage_id=vintage.vintage_id,
                    as_of_utc=to_iso(vintage.as_of_utc),
                    manifest_hash=vintage.manifest_hash,
                    window_start=(
                        None if vintage.window_start is None else to_iso(vintage.window_start)
                    ),
                    window_end=(None if vintage.window_end is None else to_iso(vintage.window_end)),
                    resolutions=[r.value for r in vintage.resolutions],
                    n_instruments=len(vintage.instrument_uids),
                    n_files=vintage.n_files,
                    row_count=vintage.row_count,
                    calendar_hash=vintage.calendar_hash,
                    action_table_hash=vintage.action_table_hash,
                    fx_table_hash=vintage.fx_table_hash,
                    universe_snapshot_id=vintage.universe_snapshot_id,
                    sealed_from=(
                        None if vintage.sealed_from is None else to_iso(vintage.sealed_from)
                    ),
                    first_live_observation_at=(
                        None
                        if vintage.first_live_observation_at is None
                        else to_iso(vintage.first_live_observation_at)
                    ),
                    survivorship_flag=vintage.survivorship_flag.value,
                    pit_completeness_flag=vintage.pit_completeness_flag.value,
                    provider_delays=dict(vintage.provider_delays),
                ),
                actor=Actor.SYSTEM,
                run_id=self._run_id,
            )
            tx.execute(
                """
                INSERT INTO data_snapshots (
                    vintage_id, as_of_utc, manifest_hash, window_start, window_end,
                    resolutions, instrument_uids, file_sha256s, row_count, calendar_hash,
                    action_table_hash, fx_table_hash, universe_snapshot_id,
                    provider_delays_json, sealed_from, first_live_observation_at,
                    survivorship_flag, pit_completeness_flag, sealing_event_seq
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    vintage.vintage_id,
                    to_iso(vintage.as_of_utc),
                    vintage.manifest_hash,
                    None if vintage.window_start is None else to_iso(vintage.window_start),
                    None if vintage.window_end is None else to_iso(vintage.window_end),
                    json.dumps([r.value for r in vintage.resolutions]),
                    json.dumps(list(vintage.instrument_uids)),
                    json.dumps(list(vintage.file_sha256s)),
                    vintage.row_count,
                    vintage.calendar_hash,
                    vintage.action_table_hash,
                    vintage.fx_table_hash,
                    vintage.universe_snapshot_id,
                    json.dumps(vintage.provider_delays),
                    None if vintage.sealed_from is None else to_iso(vintage.sealed_from),
                    (
                        None
                        if vintage.first_live_observation_at is None
                        else to_iso(vintage.first_live_observation_at)
                    ),
                    vintage.survivorship_flag.value,
                    vintage.pit_completeness_flag.value,
                    event.seq,
                ),
            )

    # -- loading -----------------------------------------------------------

    def get(self, vintage_id: str) -> DatasetVintage | None:
        row: sqlite3.Row | None = self._ledger.conn.execute(
            "SELECT * FROM data_snapshots WHERE vintage_id = ?", (vintage_id,)
        ).fetchone()
        return None if row is None else _row_to_vintage(row)

    def list_vintages(self, limit: int = 20) -> list[DatasetVintage]:
        rows = self._ledger.conn.execute(
            "SELECT * FROM data_snapshots ORDER BY as_of_utc DESC LIMIT ?", (limit,)
        ).fetchall()
        return [_row_to_vintage(row) for row in rows]

    def is_admissible(self, vintage_id: str) -> tuple[bool, str]:
        """Whether a backtest citing this vintage is admissible evidence.

        The check M5's promotion gate calls. A `vintage_id` absent from the
        ledger means the backtest ran against "the store", whose contents are
        free to change — so the result is unreproducible and cannot support a
        decision to put money behind it.
        """
        vintage = self.get(vintage_id)
        if vintage is None:
            return False, (
                f"{vintage_id} is not in the ledger. A backtest that does not name a "
                "sealed vintage ran against data that has since been free to change, so "
                "its result cannot be reproduced or checked."
            )
        if not vintage.admissible_for_promotion:
            return False, f"{vintage_id} contains no rows: a backtest over nothing"
        return True, f"{vintage_id} is sealed with {vintage.n_files} file(s)"

    def verify(self, vintage_id: str) -> list[str]:
        """Re-hash every file a vintage names. Empty list means intact.

        This is the immutability guarantee, actually checked. Yahoo restates
        history routinely, so "the file we sealed" and "the file on disk under
        that path" can genuinely diverge — and a backtest reading the second
        while citing the first is the failure the whole mechanism exists to
        make impossible.
        """
        vintage = self.get(vintage_id)
        if vintage is None:
            return [f"{vintage_id} is not in the ledger"]

        problems: list[str] = []
        for file_hash in vintage.file_sha256s:
            info = self._store.partition_by_hash(file_hash)
            if info is None:
                problems.append(
                    f"{file_hash[:12]}… is named by {vintage_id} but is not in the "
                    "partition catalog at all"
                )
                continue
            actual = self._store.file_hash_on_disk(info.relative_path)
            if actual is None:
                problems.append(
                    f"{info.relative_path} is named by {vintage_id} but is missing from disk"
                )
            elif actual != file_hash:
                problems.append(
                    f"{info.relative_path} hashes to {actual[:12]}… but {vintage_id} "
                    f"recorded {file_hash[:12]}…. The file changed after sealing."
                )
        return problems

    def bars_of(self, vintage_id: str, *, verify: bool = True) -> list[Bar]:
        """Every bar in a vintage, read from exactly the files it names.

        Not "every bar the catalog currently has for these instruments". That
        distinction is the entire point: later ingestion, compaction and
        revisions change the catalog, and a vintage that followed the catalog
        would not be a vintage.
        """
        vintage = self.get(vintage_id)
        if vintage is None:
            raise SnapshotError(f"{vintage_id} is not in the ledger")

        if verify:
            problems = self.verify(vintage_id)
            if problems:
                raise VintageAlteredError(
                    f"{vintage_id} no longer matches what was sealed:\n  " + "\n  ".join(problems)
                )

        bars: list[Bar] = []
        for file_hash in vintage.file_sha256s:
            info = self._store.partition_by_hash(file_hash)
            if info is None:
                raise SnapshotError(f"{file_hash[:12]}… is not in the partition catalog")
            bars.extend(self._store.read_partition(info.relative_path))
        return bars

    def source_for(self, vintage_id: str, *, verify: bool = True) -> InMemoryBarSource:
        """A `BarSource` restricted to one vintage's files.

        In memory because the restriction has to be structural. Handing M3 the
        `BarStore` with a note saying "only read these files" is a convention,
        and a convention is broken by the first helper that takes a shortcut.
        """
        return InMemoryBarSource(bars=self.bars_of(vintage_id, verify=verify))

    def actions_of(self, vintage_id: str) -> dict[str, tuple[CorporateAction, ...]]:
        """The corporate actions this vintage was sealed with, per instrument.

        Those recorded by the event that sealed it and no others, since its
        `action_table_hash` pins the table as it then stood. The knowledge-time
        filter cannot do this job: a later backfill records an action with the
        time it was *public*, which can be years before the seal, so without
        this a re-run of the same vintage would adjust by a split it had never
        seen, and two backtests citing one vintage would disagree.
        """
        row = self._ledger.conn.execute(
            "SELECT sealing_event_seq FROM data_snapshots WHERE vintage_id = ?", (vintage_id,)
        ).fetchone()
        if row is None:
            raise SnapshotError(f"{vintage_id} is not in the ledger")
        recorded: set[str] = set()
        for event in self._ledger.conn.execute(
            "SELECT payload_json FROM event_log WHERE event_type = ? AND seq <= ?",
            (EventType.DATA_ACTION_RECORDED.value, int(row["sealing_event_seq"])),
        ):
            recorded.add(str(json.loads(str(event["payload_json"]))["action_id"]))
        vintage = self.get(vintage_id)
        uids = () if vintage is None else vintage.instrument_uids
        return {
            uid: tuple(
                action for action in self._actions.actions_for(uid) if action.action_id in recorded
            )
            for uid in uids
        }

    def reader_for(
        self,
        vintage_id: str,
        *,
        resolution: Resolution = Resolution.DAILY,
        instrument_uids: Sequence[str] | None = None,
        lookback: object = None,
        verify: bool = True,
    ) -> ForwardOnlyReader:
        """The forward-only reader M3 walks. Contains nothing later than `t`.

        Two structural defences compose here: the source holds only this
        vintage's files, and the reader refuses to rewind or to return a bar
        whose `available_at` is past the decision time. Neither is a
        convention, which matters because the conventional version of this is
        broken by every `.shift(-1)` and every `scaler.fit(X_full)`.
        """
        vintage = self.get(vintage_id)
        if vintage is None:
            raise SnapshotError(f"{vintage_id} is not in the ledger")
        source = self.source_for(vintage_id, verify=verify)
        uids = tuple(instrument_uids) if instrument_uids else vintage.instrument_uids
        return ForwardOnlyReader(
            source=source,
            resolution=resolution,
            instrument_uids=uids,
            lookback=lookback,  # type: ignore[arg-type]
        )


# --------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------


def _with_identity(vintage: DatasetVintage, *, manifest_hash: str) -> DatasetVintage:
    """Give a vintage its content-derived id.

    Derived from the manifest hash rather than from a counter, so sealing the
    same dataset twice yields the same `vintage_id` — which makes `tb data seal`
    idempotent, and makes two backtests citing the same id provably the same
    data rather than merely the same label.
    """
    stamp = vintage.as_of_utc.astimezone(UTC).date().isoformat()
    from dataclasses import replace

    return replace(
        vintage,
        manifest_hash=manifest_hash,
        vintage_id=f"vint_{stamp}_{manifest_hash[:12]}",
    )


def _row_to_vintage(row: sqlite3.Row) -> DatasetVintage:
    def parsed(key: str) -> list[str]:
        raw = row[key]
        return [] if raw is None else list(json.loads(str(raw)))

    delays_raw = row["provider_delays_json"]
    return DatasetVintage(
        vintage_id=str(row["vintage_id"]),
        as_of_utc=from_iso(str(row["as_of_utc"])),
        manifest_hash=str(row["manifest_hash"]),
        file_sha256s=tuple(parsed("file_sha256s")),
        instrument_uids=tuple(parsed("instrument_uids")),
        resolutions=tuple(Resolution(value) for value in parsed("resolutions")),
        window_start=(None if row["window_start"] is None else from_iso(str(row["window_start"]))),
        window_end=None if row["window_end"] is None else from_iso(str(row["window_end"])),
        row_count=int(row["row_count"]),
        calendar_hash=None if row["calendar_hash"] is None else str(row["calendar_hash"]),
        action_table_hash=(
            None if row["action_table_hash"] is None else str(row["action_table_hash"])
        ),
        fx_table_hash=None if row["fx_table_hash"] is None else str(row["fx_table_hash"]),
        universe_snapshot_id=(
            None if row["universe_snapshot_id"] is None else str(row["universe_snapshot_id"])
        ),
        provider_delays=({} if delays_raw is None else dict(json.loads(str(delays_raw)))),
        sealed_from=None if row["sealed_from"] is None else from_iso(str(row["sealed_from"])),
        first_live_observation_at=(
            None
            if row["first_live_observation_at"] is None
            else from_iso(str(row["first_live_observation_at"]))
        ),
        survivorship_flag=SurvivorshipFlag(str(row["survivorship_flag"])),
        pit_completeness_flag=PitCompleteness(str(row["pit_completeness_flag"])),
    )


def observed_delays(bars: Iterable[Bar]) -> dict[str, float]:
    """Median observed delay per provider, from live bars only.

    Backfilled bars have `available_at` set to bar close, so including them
    would report a delay of zero for every provider — which is precisely the
    number that makes a delayed feed look live-capable.
    """
    by_provider: dict[str, list[float]] = {}
    for bar in bars:
        if bar.provenance is not Provenance.LIVE:
            continue
        by_provider.setdefault(bar.provider, []).append(bar.delay_seconds)
    return {
        provider: sorted(values)[len(values) // 2]
        for provider, values in by_provider.items()
        if values
    }
