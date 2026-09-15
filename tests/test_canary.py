"""The revision canary.

What this file is really establishing is that the canary's *coverage* claim is
honest. Finding a restatement is the easy half; the hard half is that "we
looked and found nothing" has to mean something, and it only does if selection
accumulates coverage rather than re-rolling the same dice every run.

So the tests here are mostly about ordering and budget, not about detection —
detection is `BarStore.ingest`'s job and is already tested there, which is
exactly why the canary reuses it rather than implementing a second diff.
"""

from __future__ import annotations

from collections.abc import Callable
from datetime import UTC, datetime, timedelta
from decimal import Decimal
from pathlib import Path
from typing import Any

import pytest

from tb.config.loader import load_hard_limits
from tb.data.barstore import BarStore
from tb.data.canary import (
    CanaryError,
    CanaryWindow,
    RevisionCanary,
    unverifiable_windows,
)
from tb.data.provider import (
    Bar,
    BarBatch,
    Provenance,
    Resolution,
    Session,
)
from tb.data.providers import CsvFixtureProvider
from tb.ledger.store import Ledger

BASE = datetime(2024, 1, 2, tzinfo=UTC)
UID_A = "isin:US0378331005"
UID_B = "isin:US5949181045"


def bar(uid: str, day: int, close: str, *, symbol_month: int | None = None) -> Bar:
    opened = BASE + timedelta(days=day)
    price = Decimal(close)
    return Bar(
        instrument_uid=uid,
        resolution=Resolution.DAILY,
        bar_open_utc=opened,
        available_at_utc=opened + timedelta(days=1),
        ingested_at_utc=opened + timedelta(days=1),
        provider="fixture",
        provenance=Provenance.BACKFILL,
        session=Session.REGULAR,
        open=price,
        high=price + Decimal("1"),
        low=price - Decimal("1"),
        close=price,
        volume=1_000_000,
    )


@pytest.fixture
def env(tmp_path: Path, write_limits: Callable[[dict[str, Any]], Path]) -> dict[str, Any]:
    limits = write_limits({})
    db = tmp_path / "ledger.db"
    with Ledger(db) as ledger:
        ledger.initialise(created_by="test")
    return {"limits": limits, "db": db, "bars": tmp_path / "bars"}


def opened_store(env: dict[str, Any]) -> tuple[Ledger, BarStore]:
    pinned = load_hard_limits(env["limits"])
    ledger = Ledger(env["db"], config_hash=pinned.config_hash).open()
    store = BarStore(ledger, root=env["bars"], scale=pinned.limits.data.price_scale)
    return ledger, store


def seal(store: BarStore, uid: str, *, days: range, symbol: str, base_price: int = 100) -> None:
    bars = tuple(bar(uid, d, str(base_price + d)) for d in days)
    store.ingest(
        BarBatch(
            bars=bars,
            provider="fixture",
            symbol=symbol,
            resolution=Resolution.DAILY,
            requested_start=bars[0].bar_open_utc,
            requested_end=bars[-1].bar_open_utc,
        )
    )
    store.compact()


def canary_for(
    env: dict[str, Any], ledger: Ledger, store: BarStore, *, pct: float = 50.0
) -> RevisionCanary:
    return RevisionCanary(ledger, store, sample_pct=pct, run_id="run_test")


# --------------------------------------------------------------------------
# Budget
# --------------------------------------------------------------------------


def test_the_budget_never_rounds_down_to_nothing(env: dict[str, Any]) -> None:
    """A 2% budget on a 10-window store must still check one.

    Rounding to zero would mean the canary silently never runs on exactly the
    deployments where history is shortest — which is where a restatement does
    the most damage, because there is least other data to notice it against.
    """
    ledger, store = opened_store(env)
    with ledger:
        canary = canary_for(env, ledger, store, pct=2.0)
        assert canary.budget(10) == 1
        assert canary.budget(1) == 1
        assert canary.budget(400) == 8


def test_a_zero_budget_checks_nothing(env: dict[str, Any]) -> None:
    """Explicitly disabling the canary is a legitimate choice."""
    ledger, store = opened_store(env)
    with ledger:
        assert canary_for(env, ledger, store, pct=0.0).budget(100) == 0


def test_an_empty_store_has_no_budget(env: dict[str, Any]) -> None:
    ledger, store = opened_store(env)
    with ledger:
        assert canary_for(env, ledger, store).budget(0) == 0


def test_a_percentage_outside_the_range_is_refused(env: dict[str, Any]) -> None:
    ledger, store = opened_store(env)
    with ledger:
        for bad in (-1.0, 101.0):
            with pytest.raises(CanaryError, match="percentage"):
                RevisionCanary(ledger, store, sample_pct=bad)


# --------------------------------------------------------------------------
# Selection: the honest-coverage property
# --------------------------------------------------------------------------


def test_never_checked_windows_come_first(env: dict[str, Any]) -> None:
    """The oldest unverified history is where an invisible restatement lives."""
    ledger, store = opened_store(env)
    with ledger:
        seal(store, UID_A, days=range(0, 20), symbol="AAPL")
        seal(store, UID_B, days=range(0, 20), symbol="MSFT")
        canary = canary_for(env, ledger, store)
        windows = canary.windows(resolution=Resolution.DAILY)
        assert windows
        assert all(w.never_checked for w in windows)
        # Deterministic: no seed, and the same store yields the same order.
        assert [w.instrument_uid for w in windows] == [
            w.instrument_uid for w in canary.windows(resolution=Resolution.DAILY)
        ]


def test_coverage_accumulates_instead_of_re_rolling_the_dice(env: dict[str, Any]) -> None:
    """The central property. Successive runs must reach *different* windows.

    A random sample would re-check some windows repeatedly and never reach
    others, which makes "we sampled and found nothing" unfalsifiable.
    """
    ledger, store = opened_store(env)
    with ledger:
        seal(store, UID_A, days=range(0, 20), symbol="AAPL")
        seal(store, UID_B, days=range(0, 20), symbol="MSFT")

        bars = list(store.bars_for(UID_A, Resolution.DAILY)) + list(
            store.bars_for(UID_B, Resolution.DAILY)
        )
        feed = CsvFixtureProvider(bars=bars, provider_name="fixture")

        # One window per run.
        canary = canary_for(env, ledger, store, pct=1.0)
        first = canary.run(feed, resolution=Resolution.DAILY)
        second = canary.run(feed, resolution=Resolution.DAILY)

        assert first.n_windows_checked == 1
        assert second.n_windows_checked == 1

        history = canary.coverage(resolution=Resolution.DAILY)
        # Two *different* windows now have a check recorded, rather than one
        # window having two.
        assert len(history) == 2, (
            "the second run re-checked the same window; selection is not accumulating coverage"
        )


def test_a_checked_window_moves_to_the_back_of_the_queue(env: dict[str, Any]) -> None:
    ledger, store = opened_store(env)
    with ledger:
        seal(store, UID_A, days=range(0, 20), symbol="AAPL")
        seal(store, UID_B, days=range(0, 20), symbol="MSFT")
        canary = canary_for(env, ledger, store, pct=1.0)

        before = canary.windows(resolution=Resolution.DAILY)
        first_uid = before[0].instrument_uid

        bars = list(store.bars_for(UID_A, Resolution.DAILY)) + list(
            store.bars_for(UID_B, Resolution.DAILY)
        )
        canary.run(
            CsvFixtureProvider(bars=bars, provider_name="fixture"),
            resolution=Resolution.DAILY,
        )

        after = canary.windows(resolution=Resolution.DAILY)
        assert after[-1].instrument_uid == first_uid
        assert not after[-1].never_checked


def test_the_check_history_is_append_only(env: dict[str, Any]) -> None:
    """ "Checked four times, never restated" beats "last checked Tuesday"."""
    ledger, store = opened_store(env)
    with ledger:
        seal(store, UID_A, days=range(0, 20), symbol="AAPL")
        bars = list(store.bars_for(UID_A, Resolution.DAILY))
        feed = CsvFixtureProvider(bars=bars, provider_name="fixture")
        canary = canary_for(env, ledger, store, pct=100.0)
        canary.run(feed, resolution=Resolution.DAILY)
        canary.run(feed, resolution=Resolution.DAILY)

        rows = canary.coverage(resolution=Resolution.DAILY)
        assert len(rows) == 1
        assert rows[0]["n_checks"] == 2, "an upsert would have thrown the first check away"


# --------------------------------------------------------------------------
# Detection and reporting
# --------------------------------------------------------------------------


def test_a_restated_bar_is_found_and_attributed(env: dict[str, Any]) -> None:
    """The thing nothing else would ever notice.

    The vendor now answers differently for a window no incremental poll would
    re-read. Detection reuses `BarStore.ingest`, so what is asserted here is
    that the canary *reached* the window and surfaced the result.
    """
    ledger, store = opened_store(env)
    with ledger:
        seal(store, UID_A, days=range(0, 20), symbol="AAPL")

        # The vendor's new answer: one old bar restated.
        restated = [bar(UID_A, d, "999" if d == 3 else str(100 + d)) for d in range(0, 20)]
        feed = CsvFixtureProvider(bars=restated, provider_name="fixture")

        result = canary_for(env, ledger, store, pct=100.0).run(feed, resolution=Resolution.DAILY)
        assert not result.clean
        assert result.n_revisions_found >= 1
        assert UID_A in result.instruments_restated()
        assert result.n_bars_compared >= 19


def test_an_unchanged_window_is_clean(env: dict[str, Any]) -> None:
    ledger, store = opened_store(env)
    with ledger:
        seal(store, UID_A, days=range(0, 20), symbol="AAPL")
        bars = list(store.bars_for(UID_A, Resolution.DAILY))
        result = canary_for(env, ledger, store, pct=100.0).run(
            CsvFixtureProvider(bars=bars, provider_name="fixture"),
            resolution=Resolution.DAILY,
        )
        assert result.clean
        assert result.n_revisions_found == 0


def test_the_result_reports_coverage_not_just_findings(env: dict[str, Any]) -> None:
    """A run that checked 1 of 10 windows established almost nothing.

    Reporting only the finding count would read as reassurance about history
    the run never touched.
    """
    ledger, store = opened_store(env)
    with ledger:
        for index in range(4):
            seal(store, f"sym:TEST{index}", days=range(0, 10), symbol=f"TEST{index}")
        canary = canary_for(env, ledger, store, pct=25.0)
        bars = [
            b for index in range(4) for b in store.bars_for(f"sym:TEST{index}", Resolution.DAILY)
        ]
        result = canary.run(
            CsvFixtureProvider(bars=bars, provider_name="fixture"),
            resolution=Resolution.DAILY,
        )
        assert result.n_windows_available == 4
        assert result.n_windows_checked == 1
        assert result.coverage_pct == 25.0
        assert "1/4 windows" in result.summary()
        assert "25% of stored history" in result.summary()


def test_a_fetch_failure_is_collected_not_raised(env: dict[str, Any]) -> None:
    """A briefly unreachable provider must not abandon the windows that worked.

    A canary that aborted on the first error would in practice never complete
    a run, and then its coverage claim would be permanently vacuous.
    """
    ledger, store = opened_store(env)
    with ledger:
        seal(store, UID_A, days=range(0, 20), symbol="AAPL")

        class Broken(CsvFixtureProvider):
            def fetch_bars(self, *args: Any, **kwargs: Any) -> BarBatch:
                raise RuntimeError("provider unreachable")

        result = canary_for(env, ledger, store, pct=100.0).run(
            Broken(bars=[], provider_name="fixture"), resolution=Resolution.DAILY
        )
        assert result.problems
        assert "unreachable" in result.problems[0]
        # And the attempt is still recorded, so the window is not silently
        # treated as verified.
        rows = canary_for(env, ledger, store).coverage(resolution=Resolution.DAILY)
        assert rows and rows[0]["n_checks"] == 1


def test_the_run_emits_its_coverage_as_an_event(env: dict[str, Any]) -> None:
    ledger, store = opened_store(env)
    with ledger:
        seal(store, UID_A, days=range(0, 20), symbol="AAPL")
        bars = list(store.bars_for(UID_A, Resolution.DAILY))
        canary_for(env, ledger, store, pct=100.0).run(
            CsvFixtureProvider(bars=bars, provider_name="fixture"),
            resolution=Resolution.DAILY,
        )
        rows = ledger.conn.execute(
            "SELECT payload_json FROM event_log WHERE event_type = ?",
            ("data.revision_canary_completed",),
        ).fetchall()
        assert len(rows) == 1
        assert "coverage_pct" in rows[0]["payload_json"]


# --------------------------------------------------------------------------
# Keeping up
# --------------------------------------------------------------------------


def test_unverifiable_windows_names_what_the_budget_is_not_reaching() -> None:
    """The number that says the budget is too small for the store it guards.

    At a 2% budget and one run a day, a 400-partition store takes fifty days
    to cover once. A window unverified for months means the canary is not
    keeping up — not that the data is fine.
    """
    now = datetime(2024, 6, 1, tzinfo=UTC)
    fresh = CanaryWindow(
        instrument_uid=UID_A,
        resolution=Resolution.DAILY,
        symbol="AAPL",
        provider="fixture",
        start=now - timedelta(days=100),
        end=now - timedelta(days=90),
        row_count=10,
        last_checked_at=now - timedelta(days=2),
    )
    stale = CanaryWindow(
        instrument_uid=UID_B,
        resolution=Resolution.DAILY,
        symbol="MSFT",
        provider="fixture",
        start=now - timedelta(days=400),
        end=now - timedelta(days=390),
        row_count=10,
        last_checked_at=None,
    )
    found = unverifiable_windows([fresh, stale], now=now, stale_after_days=60)
    assert found == (stale,)


def test_an_unchecked_window_ages_from_its_own_end_date() -> None:
    """So a never-checked window is not reported as zero days old."""
    now = datetime(2024, 6, 1, tzinfo=UTC)
    window = CanaryWindow(
        instrument_uid=UID_A,
        resolution=Resolution.DAILY,
        symbol="AAPL",
        provider="fixture",
        start=now - timedelta(days=40),
        end=now - timedelta(days=30),
        row_count=10,
    )
    assert window.age_days_at(now) == pytest.approx(30.0)
