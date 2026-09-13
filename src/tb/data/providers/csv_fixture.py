"""A deterministic provider backed by CSV files or in-memory bars.

Its purpose is to make the rest of the layer testable, and one design choice
carries all of that: **it has no clock**. `ingested_at` comes from a constructor
argument with a fixed default, so the same fixture produces byte-identical bars
today and next month. Without that, every hash in the conformance suite and
every sealed-vintage assertion would differ per run, and the vintage
immutability property — the single most important thing the store claims —
could not be tested at all.

Two further uses beyond tests:

* **Backtesting from a file.** A CSV of bars is a legitimate dataset; it just
  carries `provenance=BACKFILL` and no knowledge time, exactly like any other
  backfill, so the as-of machinery treats it the same way.
* **Reproducing a bug.** An archived provider payload can be reduced to a CSV
  and replayed through the identical ingest path.

It does *not* pretend to be a market. `capabilities` declares
`consolidated_tape=False` and a note saying so, and `live_capable` will let it
drive a live decision — deliberately, because it is used to drive the
deterministic live-loop drills in M4. That is only safe because constructing
one requires naming a fixture directory, which no production path does.
"""

from __future__ import annotations

import csv
from collections.abc import Iterable, Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from tb.core.clock import from_iso
from tb.data.provider import (
    Bar,
    BarBatch,
    DataError,
    Provenance,
    ProviderCapabilities,
    RawAction,
    Resolution,
    Session,
    TimestampConvention,
    classify_us_session,
    knowledge_time,
    parse_price,
)

# A fixed instant, chosen to be before any plausible fixture bar so that
# `knowledge_time`'s "we cannot have known it before it arrived" floor never
# engages and the result stays independent of the fixture's dates.
FIXTURE_INGESTED_AT = datetime(1970, 1, 1, tzinfo=UTC)

BAR_COLUMNS = ("bar_open_utc", "open", "high", "low", "close", "volume")
ACTION_COLUMNS = ("action_type", "effective_date")


class CsvFixtureProvider:
    """Bars from CSV files under a directory, or from a list of bars.

    Files are named `{symbol}_{resolution}.csv`, e.g. `AAPL_daily.csv`, with a
    header row. `bar_open_utc` must carry an explicit offset; a naive timestamp
    is refused rather than assumed to be UTC, for the same reason
    `tb.core.clock` refuses one.
    """

    name = "csv_fixture"

    def __init__(
        self,
        root: Path | str | None = None,
        *,
        bars: Iterable[Bar] = (),
        actions: Iterable[RawAction] = (),
        ingested_at: datetime = FIXTURE_INGESTED_AT,
        delay_seconds: float = 0.0,
        provider_name: str | None = None,
    ) -> None:
        self._root = Path(root) if root is not None else None
        self._ingested_at = ingested_at
        self._delay_seconds = delay_seconds
        # Lets a test stand this in for another provider's bars — the bake-off
        # and cross-check both need two named sources that disagree, without a
        # network.
        self._provider_name = provider_name or self.name
        self._bars: list[Bar] = list(bars)
        self._actions: list[RawAction] = list(actions)

        if ingested_at.tzinfo is None:
            raise DataError("ingested_at must be timezone-aware")

    @property
    def provider_name(self) -> str:
        return self._provider_name

    @property
    def capabilities(self) -> ProviderCapabilities:
        return ProviderCapabilities(
            name=self._provider_name,
            resolutions=frozenset(Resolution),
            timestamp_convention=TimestampConvention.BAR_OPEN,
            declared_delay_seconds=dict.fromkeys(Resolution, self._delay_seconds),
            max_history_days=dict.fromkeys(Resolution, 36500),
            supports_extended_hours=True,
            supports_corporate_actions=True,
            consolidated_tape=False,
            returns_raw_prices=True,
            note=(
                "a fixture, not a market. Deterministic by construction: no clock, so the "
                "same input yields byte-identical bars on every run."
            ),
        )

    def close(self) -> None:
        return None

    # -- ingestion of fixture content --------------------------------------

    def add_bars(self, bars: Iterable[Bar]) -> None:
        self._bars.extend(bars)

    def add_actions(self, actions: Iterable[RawAction]) -> None:
        self._actions.extend(actions)

    # -- bars --------------------------------------------------------------

    def fetch_bars(
        self,
        symbol: str,
        *,
        instrument_uid: str,
        resolution: Resolution,
        start: datetime,
        end: datetime,
        provenance: Provenance = Provenance.BACKFILL,
        include_extended: bool = False,
    ) -> BarBatch:
        if start.tzinfo is None or end.tzinfo is None:
            raise DataError("fetch window bounds must be timezone-aware")

        warnings: list[str] = []
        candidates = [
            bar
            for bar in self._bars
            if bar.instrument_uid == instrument_uid and bar.resolution is resolution
        ]
        if self._root is not None:
            loaded, file_warnings = self._load_csv(
                symbol,
                instrument_uid=instrument_uid,
                resolution=resolution,
                provenance=provenance,
            )
            candidates.extend(loaded)
            warnings.extend(file_warnings)

        # The window is half-open on the right: `end` is the decision instant in
        # every caller, and a bar opening exactly at it has not happened yet.
        selected = [
            bar
            for bar in candidates
            if start <= bar.bar_open_utc < end
            and (include_extended or bar.session is not Session.EXTENDED)
        ]
        return BarBatch(
            bars=tuple(sorted(selected, key=lambda b: (b.bar_open_utc, b.ingested_at_utc))),
            provider=self._provider_name,
            symbol=symbol,
            resolution=resolution,
            requested_start=start,
            requested_end=end,
            warnings=tuple(warnings),
        )

    def latest_bar(self, symbol: str, *, instrument_uid: str, resolution: Resolution) -> Bar | None:
        batch = self.fetch_bars(
            symbol,
            instrument_uid=instrument_uid,
            resolution=resolution,
            start=datetime(1900, 1, 1, tzinfo=UTC),
            end=datetime(2999, 1, 1, tzinfo=UTC),
            include_extended=True,
        )
        ordered = batch.sorted_bars()
        return ordered[-1] if ordered else None

    def _csv_path(self, symbol: str, resolution: Resolution) -> Path | None:
        if self._root is None:
            return None
        path = self._root / f"{symbol}_{resolution.value}.csv"
        return path if path.exists() else None

    def _load_csv(
        self,
        symbol: str,
        *,
        instrument_uid: str,
        resolution: Resolution,
        provenance: Provenance,
    ) -> tuple[list[Bar], list[str]]:
        path = self._csv_path(symbol, resolution)
        if path is None:
            return [], [f"no fixture file for {symbol} {resolution.value}"]

        bars: list[Bar] = []
        warnings: list[str] = []
        with path.open(newline="", encoding="utf-8") as handle:
            reader = csv.DictReader(handle)
            _require_columns(reader.fieldnames, BAR_COLUMNS, path)
            for line_no, row in enumerate(reader, start=2):
                try:
                    bars.append(
                        self._bar_from_row(
                            row,
                            instrument_uid=instrument_uid,
                            resolution=resolution,
                            provenance=provenance,
                        )
                    )
                except (DataError, ValueError) as exc:
                    # Reported with the line number, because "the fixture is
                    # broken" is not an actionable message either.
                    warnings.append(f"{path.name}:{line_no}: dropped row ({exc})")
        return bars, warnings

    def _bar_from_row(
        self,
        row: dict[str, str],
        *,
        instrument_uid: str,
        resolution: Resolution,
        provenance: Provenance,
    ) -> Bar:
        bar_open = from_iso(row["bar_open_utc"])
        declared = (row.get("session") or "").strip()
        session = Session(declared) if declared else classify_us_session(bar_open, resolution)
        raw_volume = (row.get("volume") or "").strip()
        return Bar(
            instrument_uid=instrument_uid,
            resolution=resolution,
            bar_open_utc=bar_open,
            available_at_utc=knowledge_time(
                bar_open=bar_open,
                resolution=resolution,
                provenance=provenance,
                delay_seconds=self._delay_seconds,
            ),
            ingested_at_utc=self._ingested_at,
            provider=self._provider_name,
            provenance=provenance,
            session=session,
            open=parse_price(row["open"]),
            high=parse_price(row["high"]),
            low=parse_price(row["low"]),
            close=parse_price(row["close"]),
            volume=None if not raw_volume else int(raw_volume),
            currency=(row.get("currency") or "").strip() or None,
        )

    # -- corporate actions -------------------------------------------------

    def fetch_actions(
        self, symbol: str, *, instrument_uid: str, start: datetime, end: datetime
    ) -> tuple[RawAction, ...]:
        start_date = start.astimezone(UTC).date().isoformat()
        end_date = end.astimezone(UTC).date().isoformat()
        actions = [
            action
            for action in self._actions
            if action.instrument_uid == instrument_uid
            and start_date <= action.effective_date <= end_date
        ]
        if self._root is not None:
            actions.extend(
                action
                for action in self._load_action_csv(symbol, instrument_uid=instrument_uid)
                if start_date <= action.effective_date <= end_date
            )
        return tuple(sorted(actions, key=lambda a: (a.effective_date, a.action_type)))

    def _load_action_csv(self, symbol: str, *, instrument_uid: str) -> list[RawAction]:
        if self._root is None:
            return []
        path = self._root / f"{symbol}_actions.csv"
        if not path.exists():
            return []

        actions: list[RawAction] = []
        with path.open(newline="", encoding="utf-8") as handle:
            reader = csv.DictReader(handle)
            _require_columns(reader.fieldnames, ACTION_COLUMNS, path)
            for row in reader:
                known_at = (row.get("known_at_utc") or "").strip()
                gross = (row.get("gross_amount") or "").strip()
                actions.append(
                    RawAction(
                        instrument_uid=instrument_uid,
                        action_type=row["action_type"].strip(),
                        effective_date=row["effective_date"].strip(),
                        # A fixture action with no stated knowledge time gets
                        # the fixed ingest instant, not "now": an action whose
                        # known_at drifts forward every run would make the
                        # as-of factor tests pass or fail by the calendar.
                        known_at_utc=from_iso(known_at) if known_at else self._ingested_at,
                        provider=self._provider_name,
                        ratio_num=_optional_int(row.get("ratio_num")),
                        ratio_den=_optional_int(row.get("ratio_den")),
                        gross_amount=parse_price(gross) if gross else None,
                        currency=(row.get("currency") or "").strip() or None,
                        new_symbol=(row.get("new_symbol") or "").strip() or None,
                        declared_date=(row.get("declared_date") or "").strip() or None,
                    )
                )
        return actions


# --------------------------------------------------------------------------
# Writing fixtures
# --------------------------------------------------------------------------


def write_bar_csv(path: Path, bars: Sequence[Bar]) -> None:
    """Write bars as a fixture file this provider can read back.

    The round trip is a test in its own right: a bar written and reloaded must
    hash identically, which it only does because prices go out as Decimal
    strings rather than through `float`.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow([*BAR_COLUMNS, "session", "currency"])
        for bar in sorted(bars, key=lambda b: b.bar_open_utc):
            writer.writerow(
                [
                    bar.bar_open_utc.astimezone(UTC).isoformat(),
                    str(bar.open),
                    str(bar.high),
                    str(bar.low),
                    str(bar.close),
                    "" if bar.volume is None else str(bar.volume),
                    bar.session.value,
                    bar.currency or "",
                ]
            )


def _require_columns(fieldnames: Sequence[str] | None, required: Sequence[str], path: Path) -> None:
    present = set(fieldnames or ())
    missing = [name for name in required if name not in present]
    if missing:
        raise DataError(f"{path.name} is missing required column(s): {', '.join(missing)}")


def _optional_int(value: Any) -> int | None:
    text = str(value or "").strip()
    return int(text) if text else None
