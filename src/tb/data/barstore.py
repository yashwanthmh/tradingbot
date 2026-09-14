"""The bar store: Parquet for the bytes, the ledger for what counts as data.

The arrangement is deliberate and the ordering rule is the whole design:

**A Parquet file is part of the dataset only because the ledger says so.** Bar
bytes live in content-addressed Parquet files; membership is a `data_partitions`
row written together with a `data.partition_sealed` event. Files are written to
a temporary name, fsynced, renamed into place, and *then* recorded. So a crash
either leaves an unrecorded file — ignorable garbage, because nothing reads by
directory listing — or nothing at all. The inverse ordering would leave an event
naming a file that does not exist, which is unrecoverable rather than merely
untidy. There is no two-phase commit here because there is no coordinator to
run one.

**Writes are two-tier.** Incremental polls go to `bars_hot` inside the ledger
transaction that records them, because a Parquet file per poll would mean
thousands of tiny files and, worse, would split the data from its provenance.
Bulk backfill seals straight to Parquet, because millions of rows should not
pass through a `synchronous=FULL` sqlite connection. Compaction at session close
turns hot rows into one sealed partition.

**Revisions are detected over the set, not row by row.** A vendor can change a
bar, *delete* one (dropping a bad print), or insert one late. A row-wise diff
sees only the first. Yahoo does all three, silently, which is why every
observation is kept rather than overwritten.
"""

from __future__ import annotations

import os
import sqlite3
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from tb.core.canonical import hash_payload
from tb.core.clock import from_iso, now_iso
from tb.core.ids import new_id
from tb.data.provider import (
    Bar,
    BarBatch,
    DataError,
    Provenance,
    Resolution,
    Session,
    dedupe_vintages,
    from_scaled,
)
from tb.ledger.events import Actor, BarRevisionPayload, EventType, PartitionSealedPayload
from tb.ledger.store import Ledger

# Bumped when the on-disk column set changes. Written into each file's metadata
# so a reader can refuse a file it does not understand rather than
# misinterpreting one.
PARQUET_SCHEMA_VERSION = 1

_COLUMNS: tuple[str, ...] = (
    "instrument_uid",
    "resolution",
    "bar_open_utc",
    "available_at_utc",
    "ingested_at_utc",
    "provider",
    "provenance",
    "session",
    "open_scaled",
    "high_scaled",
    "low_scaled",
    "close_scaled",
    "volume",
    "price_scale",
    "currency",
)


class RevisionKind:
    CHANGED = "changed"
    DELETED = "deleted"
    LATE_INSERT = "late_insert"


@dataclass(frozen=True, slots=True)
class PartitionInfo:
    file_sha256: str
    relative_path: str
    instrument_uid: str
    resolution: Resolution
    provider: str
    first_bar_open: datetime
    last_bar_open: datetime
    row_count: int
    byte_size: int
    rows_hash: str


@dataclass(frozen=True, slots=True)
class Revision:
    kind: str
    instrument_uid: str
    resolution: Resolution
    bar_open_utc: datetime
    provider: str
    delta_bps: float | None
    first_values: dict[str, Any] | None
    new_values: dict[str, Any] | None


@dataclass(slots=True)
class IngestResult:
    rows_written: int = 0
    rows_unchanged: int = 0
    revisions: list[Revision] = field(default_factory=list)

    @property
    def had_revisions(self) -> bool:
        return bool(self.revisions)


def _pyarrow() -> Any:
    """Import pyarrow lazily.

    Kept out of module import so the live trading loop, which only appends to
    `bars_hot`, never pays for arrow or pandas. Columnar machinery is read-time
    only.
    """
    try:
        import pyarrow as pa
        import pyarrow.parquet as pq
    except ImportError as exc:  # pragma: no cover - depends on install extras
        raise DataError(
            "the bar store needs pyarrow. Install it with `uv sync --extra data`."
        ) from exc
    return pa, pq


def _arrow_schema() -> Any:
    pa, _ = _pyarrow()
    return pa.schema(
        [
            ("instrument_uid", pa.string()),
            ("resolution", pa.string()),
            # Microsecond UTC timestamps, declared. Leaving arrow to infer the
            # unit is how a nanosecond column silently becomes a different
            # value on a round trip through a different pandas version.
            ("bar_open_utc", pa.timestamp("us", tz="UTC")),
            ("available_at_utc", pa.timestamp("us", tz="UTC")),
            ("ingested_at_utc", pa.timestamp("us", tz="UTC")),
            ("provider", pa.string()),
            ("provenance", pa.string()),
            ("session", pa.string()),
            # Scaled integers, never floats: a float round trip changes the
            # bits, which changes every hash computed over the row.
            ("open_scaled", pa.int64()),
            ("high_scaled", pa.int64()),
            ("low_scaled", pa.int64()),
            ("close_scaled", pa.int64()),
            ("volume", pa.int64()),
            ("price_scale", pa.int32()),
            ("currency", pa.string()),
        ],
        metadata={b"tb_schema_version": str(PARQUET_SCHEMA_VERSION).encode()},
    )


def rows_hash(rows: Sequence[dict[str, Any]]) -> str:
    """Content hash over canonical row dicts, order-independent.

    Hashed over the rows rather than over the file bytes, so the same logical
    data seals to the same `rows_hash` regardless of arrow's compression or
    row-group layout. The file's own sha256 is tracked separately and names it
    on disk.
    """
    return hash_payload(sorted(hash_payload(row) for row in rows))


class BarStore:
    """Ingests, seals and serves bars. Implements `asof.BarSource`."""

    def __init__(
        self,
        ledger: Ledger,
        *,
        root: Path | str = "data/bars",
        scale: int = 6,
        run_id: str | None = None,
    ) -> None:
        self._ledger = ledger
        self._root = Path(root)
        self._scale = scale
        self._run_id = run_id

    @property
    def root(self) -> Path:
        return self._root

    @property
    def scale(self) -> int:
        return self._scale

    # -- write path: incremental ------------------------------------------

    def ingest(self, batch: BarBatch, *, detect_revisions: bool = True) -> IngestResult:
        """Stage bars in `bars_hot`, atomically with their provenance event.

        Re-ingesting identical data is a no-op: unchanged rows are counted and
        skipped rather than written again. Without that, every poll would look
        like a revision and `bar_revisions` would fill with noise until the real
        signal was undetectable.
        """
        result = IngestResult()
        if not batch.bars:
            return result

        ordered = batch.sorted_bars()
        existing = self._existing_rows(
            ordered[0].instrument_uid,
            ordered[0].resolution,
            batch.provider,
            start=ordered[0].bar_open_utc,
            end=ordered[-1].bar_open_utc,
        )

        if detect_revisions:
            result.revisions = self._diff(existing, ordered, batch)

        to_write: list[Bar] = []
        for bar in ordered:
            key = self._identity_key(bar)
            prior = existing.get(key)
            if prior is not None and prior["row_hash"] == bar.row_hash(scale=self._scale):
                result.rows_unchanged += 1
                continue
            to_write.append(bar)

        if not to_write and not result.revisions:
            return result

        with self._ledger.transaction() as tx:
            for bar in to_write:
                row = bar.row_values(scale=self._scale)
                tx.execute(
                    """
                    INSERT INTO bars_hot (
                        instrument_uid, resolution, bar_open_utc, available_at_utc,
                        ingested_at_utc, provider, provenance, session,
                        open_scaled, high_scaled, low_scaled, close_scaled,
                        volume, price_scale, currency, row_hash
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    ON CONFLICT DO NOTHING
                    """,
                    (
                        bar.instrument_uid,
                        bar.resolution.value,
                        _iso(bar.bar_open_utc),
                        _iso(bar.available_at_utc),
                        _iso(bar.ingested_at_utc),
                        bar.provider,
                        bar.provenance.value,
                        bar.session.value,
                        row["open_scaled"],
                        row["high_scaled"],
                        row["low_scaled"],
                        row["close_scaled"],
                        bar.volume,
                        self._scale,
                        bar.currency,
                        bar.row_hash(scale=self._scale),
                    ),
                )
                result.rows_written += 1

            for revision in result.revisions:
                revision_id = new_id("rev", length=16)
                tx.append(
                    EventType.DATA_BAR_REVISION_DETECTED,
                    revision_id,
                    BarRevisionPayload(
                        revision_id=revision_id,
                        instrument_uid=revision.instrument_uid,
                        resolution=revision.resolution.value,
                        bar_open_utc=_iso(revision.bar_open_utc),
                        provider=revision.provider,
                        kind=revision.kind,
                        delta_bps=revision.delta_bps,
                        detected_by="ingest",
                        first_values=revision.first_values,
                        new_values=revision.new_values,
                    ),
                    actor=Actor.SYSTEM,
                )
                tx.execute(
                    """
                    INSERT INTO bar_revisions (
                        revision_id, instrument_uid, resolution, bar_open_utc, provider,
                        kind, first_seen_at, first_values_json, revised_at,
                        new_values_json, delta_bps, detected_by
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        revision_id,
                        revision.instrument_uid,
                        revision.resolution.value,
                        _iso(revision.bar_open_utc),
                        revision.provider,
                        revision.kind,
                        (revision.first_values or {}).get("ingested_at_utc", now_iso()),
                        _json_or_none(revision.first_values),
                        now_iso(),
                        _json_or_none(revision.new_values),
                        revision.delta_bps,
                        "ingest",
                    ),
                )
        return result

    def _diff(
        self,
        existing: dict[tuple[str, str, str, str], sqlite3.Row | dict[str, Any]],
        incoming: Sequence[Bar],
        batch: BarBatch,
    ) -> list[Revision]:
        """Compare over the set, so deletions and late inserts are visible too."""
        revisions: list[Revision] = []
        incoming_by_key = {self._identity_key(bar): bar for bar in incoming}

        for key, bar in incoming_by_key.items():
            prior = existing.get(key)
            if prior is None:
                # Only a late insert if it lands inside a range we already had.
                # A bar newer than everything stored is just the next bar.
                if existing and bar.bar_open_utc < max(
                    from_iso(str(r["bar_open_utc"])) for r in existing.values()
                ):
                    revisions.append(
                        Revision(
                            kind=RevisionKind.LATE_INSERT,
                            instrument_uid=bar.instrument_uid,
                            resolution=bar.resolution,
                            bar_open_utc=bar.bar_open_utc,
                            provider=bar.provider,
                            delta_bps=None,
                            first_values=None,
                            new_values=bar.row_values(scale=self._scale),
                        )
                    )
                continue

            new_hash = bar.row_hash(scale=self._scale)
            if prior["row_hash"] == new_hash:
                continue

            old_close = int(prior["close_scaled"])
            new_values = bar.row_values(scale=self._scale)
            new_close = int(str(new_values["close_scaled"]))
            delta_bps = abs(new_close - old_close) / old_close * 10_000.0 if old_close else None
            revisions.append(
                Revision(
                    kind=RevisionKind.CHANGED,
                    instrument_uid=bar.instrument_uid,
                    resolution=bar.resolution,
                    bar_open_utc=bar.bar_open_utc,
                    provider=bar.provider,
                    delta_bps=delta_bps,
                    first_values=_row_to_values(prior),
                    new_values=new_values,
                )
            )

        # Deletions: a bar we had, inside the window the provider just answered
        # for, that has now vanished. Invisible to a row-wise diff.
        #
        # The window comes from what was *requested* where the provider tells
        # us, falling back to the span of what came back. That distinction
        # matters for sparse data: on a thin name, "the provider answered for
        # 14:00-15:00 and no longer has 14:37" is a restatement worth logging,
        # whereas inferring the window from returned bars alone would call
        # every quiet minute a deletion.
        if batch.bars:
            window_start = batch.requested_start or incoming[0].bar_open_utc
            window_end = batch.requested_end or incoming[-1].bar_open_utc
            for key, prior in existing.items():
                if key in incoming_by_key:
                    continue
                prior_open = from_iso(str(prior["bar_open_utc"]))
                if not (window_start <= prior_open <= window_end):
                    continue
                revisions.append(
                    Revision(
                        kind=RevisionKind.DELETED,
                        instrument_uid=str(prior["instrument_uid"]),
                        resolution=Resolution(str(prior["resolution"])),
                        bar_open_utc=prior_open,
                        provider=str(prior["provider"]),
                        delta_bps=None,
                        first_values=_row_to_values(prior),
                        new_values=None,
                    )
                )
        return revisions

    # -- write path: sealing ----------------------------------------------

    def seal(
        self, bars: Sequence[Bar], *, provider: str, supersedes: Sequence[str] = ()
    ) -> PartitionInfo:
        """Write bars to a content-addressed Parquet file and record it.

        Ordering is the safety property: temp write, fsync, rename, *then* the
        ledger transaction. A crash before the transaction leaves an unrecorded
        file, which nothing will ever read because reads go through the catalog.
        A crash after it is impossible — the rename already happened.
        """
        if not bars:
            raise DataError("nothing to seal")

        # Keeps every vintage: collapsing here would discard the value an
        # as-of query over an earlier instant must return.
        ordered = dedupe_vintages(bars)
        uids = {bar.instrument_uid for bar in ordered}
        resolutions = {bar.resolution for bar in ordered}
        providers = {bar.provider for bar in ordered}
        if len(uids) != 1 or len(resolutions) != 1:
            raise DataError(
                "a partition holds one instrument at one resolution; got "
                f"{len(uids)} instrument(s) and {len(resolutions)} resolution(s)"
            )
        if providers != {provider}:
            # Asserted, not tolerated. A partition labelled with one provider
            # but containing another's bars makes the catalog lie about where
            # its rows came from — and the cross-provider check, which is the
            # only defence against a mismapped ticker, would then be comparing
            # a feed against itself.
            raise DataError(
                f"a partition holds one provider's bars; sealing as {provider!r} but the "
                f"bars come from {sorted(providers)}. Two providers' observations of the "
                "same period are different data and belong in different partitions."
            )
        uid = ordered[0].instrument_uid
        resolution = ordered[0].resolution

        rows = [bar.storage_row(scale=self._scale) for bar in ordered]
        # Hash over the value rows, not the storage rows: including vintage
        # time would make an identical re-seal produce a different hash.
        content_hash = rows_hash([bar.row_values(scale=self._scale) for bar in ordered])

        directory = self._partition_dir(uid, resolution, ordered[0].bar_open_utc)
        directory.mkdir(parents=True, exist_ok=True)
        temp_path = directory / f".tmp-{new_id('part', length=12)}.parquet"

        self._write_parquet(temp_path, rows)
        file_hash = _file_sha256(temp_path)
        final_path = directory / f"{file_hash}.parquet"
        os.replace(temp_path, final_path)
        _fsync_dir(directory)

        info = PartitionInfo(
            file_sha256=file_hash,
            relative_path=str(final_path.relative_to(self._root)),
            instrument_uid=uid,
            resolution=resolution,
            provider=provider,
            first_bar_open=min(b.bar_open_utc for b in ordered),
            last_bar_open=max(b.bar_open_utc for b in ordered),
            row_count=len(ordered),
            byte_size=final_path.stat().st_size,
            rows_hash=content_hash,
        )

        with self._ledger.transaction() as tx:
            event = tx.append(
                EventType.DATA_PARTITION_SEALED,
                file_hash,
                PartitionSealedPayload(
                    file_sha256=file_hash,
                    relative_path=info.relative_path,
                    instrument_uid=uid,
                    resolution=resolution.value,
                    provider=provider,
                    first_bar_open=_iso(info.first_bar_open),
                    last_bar_open=_iso(info.last_bar_open),
                    row_count=info.row_count,
                    byte_size=info.byte_size,
                    rows_hash=content_hash,
                    supersedes=list(supersedes),
                ),
                actor=Actor.SYSTEM,
            )
            tx.execute(
                """
                INSERT INTO data_partitions (
                    file_sha256, relative_path, instrument_uid, resolution, provider,
                    first_bar_open, last_bar_open, row_count, byte_size, rows_hash,
                    sealed_at, sealing_event_seq, schema_version
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(file_sha256) DO NOTHING
                """,
                (
                    file_hash,
                    info.relative_path,
                    uid,
                    resolution.value,
                    provider,
                    _iso(info.first_bar_open),
                    _iso(info.last_bar_open),
                    info.row_count,
                    info.byte_size,
                    content_hash,
                    event.ts_utc,
                    event.seq,
                    PARQUET_SCHEMA_VERSION,
                ),
            )
            for superseded in supersedes:
                tx.execute(
                    "UPDATE data_partitions SET superseded_by = ? WHERE file_sha256 = ?",
                    (file_hash, superseded),
                )
        return info

    def compact(
        self, *, instrument_uid: str | None = None, resolution: Resolution | None = None
    ) -> list[PartitionInfo]:
        """Seal staged hot rows into Parquet and clear them.

        Hot rows are dropped only after the file is durably renamed and the
        sealing event committed, and in the same transaction as the delete — so
        a failure leaves the rows staged and the file an ignorable orphan,
        never data with no home.
        """
        sealed: list[PartitionInfo] = []
        for uid, res, provider in self._hot_groups(instrument_uid, resolution):
            bars = list(self._hot_bars(uid, res, provider))
            if not bars:
                continue
            existing = self._sealed_partitions(uid, res, provider=provider)
            supersedes = [
                str(row["file_sha256"])
                for row in existing
                if _overlaps(row, bars[0].bar_open_utc, bars[-1].bar_open_utc)
            ]
            if supersedes:
                # Re-seal the union so the new partition fully replaces the old
                # ones. Leaving both live would make the as-of view depend on
                # file iteration order.
                for row in existing:
                    if str(row["file_sha256"]) in supersedes:
                        bars.extend(self._read_partition(str(row["relative_path"])))
                bars = list(dedupe_vintages(bars))

            info = self.seal(bars, provider=provider, supersedes=supersedes)
            self._ledger.conn.execute(
                "DELETE FROM bars_hot WHERE instrument_uid = ? AND resolution = ? AND provider = ?",
                (uid, res.value, provider),
            )
            self._ledger.conn.commit()
            sealed.append(info)
        return sealed

    # -- read path (BarSource) --------------------------------------------

    def bars_for(
        self,
        instrument_uid: str,
        resolution: Resolution,
        *,
        start: datetime | None = None,
        end: datetime | None = None,
    ) -> Iterable[Bar]:
        """Every observation we hold, sealed and staged, deduped by vintage.

        Deliberately returns *all* vintages of a bar rather than the newest:
        collapsing here would make an as-of query unable to see the value that
        was current at the time. `visible_bars` does the collapsing, after
        filtering on knowledge time.
        """
        seen: list[Bar] = []
        for row in self._sealed_partitions(instrument_uid, resolution):
            if start is not None and from_iso(str(row["last_bar_open"])) < start:
                continue
            if end is not None and from_iso(str(row["first_bar_open"])) > end:
                continue
            seen.extend(self._read_partition(str(row["relative_path"])))
        seen.extend(self._hot_bars(instrument_uid, resolution, provider=None))

        for bar in seen:
            if start is not None and bar.bar_open_utc < start:
                continue
            if end is not None and bar.bar_open_utc > end:
                continue
            yield bar

    def instruments(self) -> Iterable[str]:
        rows = self._ledger.conn.execute(
            "SELECT DISTINCT instrument_uid FROM data_partitions WHERE superseded_by IS NULL "
            "UNION SELECT DISTINCT instrument_uid FROM bars_hot"
        ).fetchall()
        return sorted(str(row["instrument_uid"]) for row in rows)

    # -- the catalog, for sealing a vintage --------------------------------

    def live_partitions(
        self,
        *,
        instrument_uid: str | None = None,
        resolution: Resolution | None = None,
    ) -> list[PartitionInfo]:
        """Every catalog row that is still part of the dataset.

        `superseded_by IS NULL` is the whole filter: a compaction re-seals the
        union of overlapping partitions and retires the originals, so including
        them would count the same bars twice in a vintage's row total and make
        its manifest depend on compaction history.
        """
        sql = "SELECT * FROM data_partitions WHERE superseded_by IS NULL"
        params: list[Any] = []
        if instrument_uid is not None:
            sql += " AND instrument_uid = ?"
            params.append(instrument_uid)
        if resolution is not None:
            sql += " AND resolution = ?"
            params.append(resolution.value)
        sql += " ORDER BY instrument_uid, resolution, first_bar_open"
        return [_row_to_partition(row) for row in self._ledger.conn.execute(sql, params)]

    def partition_by_hash(self, file_sha256: str) -> PartitionInfo | None:
        row = self._ledger.conn.execute(
            "SELECT * FROM data_partitions WHERE file_sha256 = ?", (file_sha256,)
        ).fetchone()
        return None if row is None else _row_to_partition(row)

    def read_partition(self, relative_path: str) -> list[Bar]:
        """Read one named partition file.

        Public because a sealed vintage is defined as a *set of file hashes*,
        and reading it means reading exactly those files — not whatever the
        catalog currently says about an instrument. That distinction is what
        makes a vintage immutable under later ingestion.
        """
        return self._read_partition(relative_path)

    def file_hash_on_disk(self, relative_path: str) -> str | None:
        path = self._root / relative_path
        return _file_sha256(path) if path.exists() else None

    def has_hot_rows(self) -> bool:
        """Whether any bars are still staged rather than sealed.

        A vintage must reference only immutable Parquet, so sealing one while
        rows sit in `bars_hot` would produce a snapshot missing the newest
        data — silently, because the catalog would look complete.
        """
        row = self._ledger.conn.execute("SELECT 1 FROM bars_hot LIMIT 1").fetchone()
        return row is not None

    # -- integrity ---------------------------------------------------------

    def verify_partitions(self) -> list[str]:
        """Check the catalog against the filesystem, both directions.

        The data-layer analogue of `tb ledger verify`, with the same fail-closed
        reading: a recorded file that is missing or whose hash has changed is a
        hard failure; an unrecorded file on disk is reported but harmless,
        because nothing reads it.
        """
        findings: list[str] = []
        recorded: set[str] = set()

        for row in self._ledger.conn.execute(
            "SELECT * FROM data_partitions WHERE superseded_by IS NULL"
        ).fetchall():
            path = self._root / str(row["relative_path"])
            recorded.add(str(path.resolve()))
            if not path.exists():
                findings.append(
                    f"MISSING: partition {row['file_sha256'][:12]}… is recorded in the "
                    f"catalog but {path} does not exist. Bars the ledger claims we hold "
                    "are gone."
                )
                continue
            actual = _file_sha256(path)
            if actual != row["file_sha256"]:
                findings.append(
                    f"ALTERED: {path} hashes to {actual[:12]}… but the catalog recorded "
                    f"{str(row['file_sha256'])[:12]}…. The file changed after sealing."
                )

        if self._root.exists():
            for path in sorted(self._root.rglob("*.parquet")):
                if str(path.resolve()) not in recorded:
                    findings.append(
                        f"ORPHAN: {path} is not named by any sealing event, so it is not "
                        "part of the dataset and will never be read. Safe to delete."
                    )
        return findings

    # -- internals ---------------------------------------------------------

    def _identity_key(self, bar: Bar) -> tuple[str, str, str, str]:
        return (
            bar.instrument_uid,
            bar.resolution.value,
            _iso(bar.bar_open_utc),
            bar.provider,
        )

    def _existing_rows(
        self,
        instrument_uid: str,
        resolution: Resolution,
        provider: str,
        *,
        start: datetime,
        end: datetime,
    ) -> dict[tuple[str, str, str, str], Any]:
        """The newest stored observation per bar, across hot rows and sealed files."""
        out: dict[tuple[str, str, str, str], Any] = {}

        for bar in self.bars_for(instrument_uid, resolution, start=start, end=end):
            if bar.provider != provider:
                continue
            key = self._identity_key(bar)
            row = {
                **bar.row_values(scale=self._scale),
                "row_hash": bar.row_hash(scale=self._scale),
                "ingested_at_utc": _iso(bar.ingested_at_utc),
            }
            prior = out.get(key)
            if prior is None or str(prior["ingested_at_utc"]) <= str(row["ingested_at_utc"]):
                out[key] = row
        return out

    def _partition_dir(
        self, instrument_uid: str, resolution: Resolution, first_open: datetime
    ) -> Path:
        # Symbol first: every read is symbol-scoped, so a date-first layout
        # would touch every instrument's files to answer one instrument's query.
        safe_uid = instrument_uid.replace(":", "_").replace("/", "_")
        return (
            self._root
            / f"resolution={resolution.value}"
            / f"instrument_uid={safe_uid}"
            / f"year={first_open.year:04d}"
            / f"month={first_open.month:02d}"
        )

    def _write_parquet(self, path: Path, rows: Sequence[dict[str, Any]]) -> None:
        pa, pq = _pyarrow()
        schema = _arrow_schema()
        columns: dict[str, list[Any]] = {name: [] for name in _COLUMNS}
        for row in rows:
            for name in _COLUMNS:
                value = row[name]
                if name.endswith("_utc"):
                    value = from_iso(str(value))
                columns[name].append(value)
        table = pa.table(columns, schema=schema)
        pq.write_table(table, path, compression="zstd", version="2.6")
        # fsync the file itself before the rename, or a crash can leave a
        # renamed-but-empty file, which is worse than no file at all.
        with open(path, "rb") as handle:
            os.fsync(handle.fileno())

    def _read_partition(self, relative_path: str) -> list[Bar]:
        _, pq = _pyarrow()
        path = self._root / relative_path
        if not path.exists():
            raise DataError(
                f"partition {relative_path} is in the catalog but missing from disk. "
                "Run `tb data verify`: the ledger claims bars we do not have."
            )
        table = pq.read_table(path)
        declared = (table.schema.metadata or {}).get(b"tb_schema_version")
        if declared is not None and int(declared) != PARQUET_SCHEMA_VERSION:
            raise DataError(
                f"{relative_path} was written with bar schema v{int(declared)}; this build "
                f"reads v{PARQUET_SCHEMA_VERSION}. Refusing to guess at the difference."
            )
        return [_dict_to_bar(row) for row in table.to_pylist()]

    def _sealed_partitions(
        self,
        instrument_uid: str,
        resolution: Resolution,
        *,
        provider: str | None = None,
    ) -> list[sqlite3.Row]:
        """Live partitions for an instrument, optionally for one provider only.

        `provider` matters for supersession and not for reading. Two providers'
        bars for the same instrument and period are *different observations*,
        both legitimate, and each lives in its own partition — the catalog has a
        `provider` column for exactly that reason. Without the filter,
        compaction treated an overlapping Alpaca partition as something a Yahoo
        seal should replace, merged both providers' rows into one file, and
        labelled the result with whichever provider happened to compact second.
        The cross-provider check then had nothing left to compare.
        """
        sql = (
            "SELECT * FROM data_partitions WHERE instrument_uid = ? AND resolution = ? "
            "AND superseded_by IS NULL"
        )
        params: list[Any] = [instrument_uid, resolution.value]
        if provider is not None:
            sql += " AND provider = ?"
            params.append(provider)
        return list(self._ledger.conn.execute(sql + " ORDER BY first_bar_open", params))

    def _hot_groups(
        self, instrument_uid: str | None, resolution: Resolution | None
    ) -> list[tuple[str, Resolution, str]]:
        sql = "SELECT DISTINCT instrument_uid, resolution, provider FROM bars_hot"
        params: list[Any] = []
        clauses = []
        if instrument_uid is not None:
            clauses.append("instrument_uid = ?")
            params.append(instrument_uid)
        if resolution is not None:
            clauses.append("resolution = ?")
            params.append(resolution.value)
        if clauses:
            sql += " WHERE " + " AND ".join(clauses)
        return [
            (str(r["instrument_uid"]), Resolution(str(r["resolution"])), str(r["provider"]))
            for r in self._ledger.conn.execute(sql, params).fetchall()
        ]

    def _hot_bars(
        self, instrument_uid: str, resolution: Resolution, provider: str | None
    ) -> Iterable[Bar]:
        sql = "SELECT * FROM bars_hot WHERE instrument_uid = ? AND resolution = ?"
        params: list[Any] = [instrument_uid, resolution.value]
        if provider is not None:
            sql += " AND provider = ?"
            params.append(provider)
        sql += " ORDER BY bar_open_utc, ingested_at_utc"
        for row in self._ledger.conn.execute(sql, params).fetchall():
            yield _sqlite_row_to_bar(row)


# --------------------------------------------------------------------------
# Row conversion
# --------------------------------------------------------------------------


def _iso(value: datetime) -> str:
    return value.astimezone(UTC).isoformat()


def _json_or_none(values: dict[str, Any] | None) -> str | None:
    if values is None:
        return None
    from tb.core.canonical import canonical_json

    return canonical_json(values)


def _row_to_values(row: Any) -> dict[str, Any]:
    return {name: row[name] for name in _COLUMNS if _has_key(row, name)}


def _has_key(row: Any, name: str) -> bool:
    if isinstance(row, dict):
        return name in row
    try:
        row[name]
    except (IndexError, KeyError):
        return False
    return True


def _bar_from_parts(
    *,
    instrument_uid: str,
    resolution: str,
    bar_open: datetime,
    available_at: datetime,
    ingested_at: datetime,
    provider: str,
    provenance: str,
    session: str,
    open_scaled: int,
    high_scaled: int,
    low_scaled: int,
    close_scaled: int,
    volume: int | None,
    price_scale: int,
    currency: str | None,
) -> Bar:
    return Bar(
        instrument_uid=instrument_uid,
        resolution=Resolution(resolution),
        bar_open_utc=bar_open,
        available_at_utc=available_at,
        ingested_at_utc=ingested_at,
        provider=provider,
        provenance=Provenance(provenance),
        session=Session(session),
        open=from_scaled(int(open_scaled), int(price_scale)),
        high=from_scaled(int(high_scaled), int(price_scale)),
        low=from_scaled(int(low_scaled), int(price_scale)),
        close=from_scaled(int(close_scaled), int(price_scale)),
        volume=None if volume is None else int(volume),
        currency=currency,
    )


def _sqlite_row_to_bar(row: sqlite3.Row) -> Bar:
    return _bar_from_parts(
        instrument_uid=str(row["instrument_uid"]),
        resolution=str(row["resolution"]),
        bar_open=from_iso(str(row["bar_open_utc"])),
        available_at=from_iso(str(row["available_at_utc"])),
        ingested_at=from_iso(str(row["ingested_at_utc"])),
        provider=str(row["provider"]),
        provenance=str(row["provenance"]),
        session=str(row["session"]),
        open_scaled=int(row["open_scaled"]),
        high_scaled=int(row["high_scaled"]),
        low_scaled=int(row["low_scaled"]),
        close_scaled=int(row["close_scaled"]),
        volume=row["volume"],
        price_scale=int(row["price_scale"]),
        currency=row["currency"],
    )


def _dict_to_bar(row: dict[str, Any]) -> Bar:
    def _aware(value: Any) -> datetime:
        stamp: datetime = value
        return stamp if stamp.tzinfo is not None else stamp.replace(tzinfo=UTC)

    return _bar_from_parts(
        instrument_uid=str(row["instrument_uid"]),
        resolution=str(row["resolution"]),
        bar_open=_aware(row["bar_open_utc"]),
        available_at=_aware(row["available_at_utc"]),
        ingested_at=_aware(row["ingested_at_utc"]),
        provider=str(row["provider"]),
        provenance=str(row["provenance"]),
        session=str(row["session"]),
        open_scaled=int(row["open_scaled"]),
        high_scaled=int(row["high_scaled"]),
        low_scaled=int(row["low_scaled"]),
        close_scaled=int(row["close_scaled"]),
        volume=row["volume"],
        price_scale=int(row["price_scale"]),
        currency=row["currency"],
    )


def _file_sha256(path: Path) -> str:
    import hashlib

    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _fsync_dir(directory: Path) -> None:
    """Durably record the rename, not just the file contents."""
    try:
        fd = os.open(directory, os.O_RDONLY)
    except OSError:  # pragma: no cover - platform dependent
        return
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def _overlaps(row: Any, start: datetime, end: datetime) -> bool:
    first = from_iso(str(row["first_bar_open"]))
    last = from_iso(str(row["last_bar_open"]))
    return not (last < start or first > end)


__all__ = [
    "PARQUET_SCHEMA_VERSION",
    "BarStore",
    "IngestResult",
    "PartitionInfo",
    "Revision",
    "RevisionKind",
    "rows_hash",
]


def _row_to_partition(row: sqlite3.Row) -> PartitionInfo:
    return PartitionInfo(
        file_sha256=str(row["file_sha256"]),
        relative_path=str(row["relative_path"]),
        instrument_uid=str(row["instrument_uid"]),
        resolution=Resolution(str(row["resolution"])),
        provider=str(row["provider"]),
        first_bar_open=from_iso(str(row["first_bar_open"])),
        last_bar_open=from_iso(str(row["last_bar_open"])),
        row_count=int(row["row_count"]),
        byte_size=int(row["byte_size"]),
        rows_hash=str(row["rows_hash"]),
    )
