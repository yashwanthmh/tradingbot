"""`tb backtest ...`.

Exit codes are the contract, same as the data CLI: 2 for a setup problem the
operator must fix, 1 for a finding (a calibration that failed), 0 for clean.
The calibration exiting non-zero is what lets it sit in CI as a release gate
rather than as a report nobody reads.
"""

from __future__ import annotations

import random
from collections.abc import Callable
from datetime import UTC, datetime, timedelta
from decimal import Decimal
from pathlib import Path
from typing import Any

import pytest
from typer.testing import CliRunner

from tb.cli import app
from tb.config.loader import load_hard_limits
from tb.data.barstore import BarStore
from tb.data.provider import Bar, BarBatch, Provenance, Resolution, Session
from tb.ledger.store import Ledger

runner = CliRunner()
BASE = datetime(2024, 1, 2, tzinfo=UTC)


@pytest.fixture
def env(tmp_path: Path, write_limits: Callable[[dict[str, Any]], Path]) -> dict[str, Any]:
    run_dir = tmp_path / "run"
    run_dir.mkdir(exist_ok=True)
    limits = write_limits(
        {
            "safety": {
                "kill_switch_path": str(run_dir / "KILL"),
                "heartbeat_path": str(run_dir / "heartbeat"),
            }
        }
    )
    return {
        "limits": limits,
        "db": tmp_path / "ledger.db",
        "bars": tmp_path / "bars",
        # `calibrations` reads no bars, so it takes no --bars; passing one would
        # be a user error, and accepting an option it ignores would be worse.
        "ledger_args": [
            "--limits",
            str(limits),
            "--db",
            str(tmp_path / "ledger.db"),
        ],
        "args": [
            "--limits",
            str(limits),
            "--db",
            str(tmp_path / "ledger.db"),
            "--bars",
            str(tmp_path / "bars"),
        ],
    }


def _run(args: list[str]) -> Any:
    result = runner.invoke(app, args)
    if result.exception and not isinstance(result.exception, SystemExit):
        raise result.exception
    return result


def _out(result: Any) -> str:
    stderr = ""
    try:
        stderr = result.stderr or ""
    except ValueError:  # pragma: no cover - depends on the click version
        stderr = ""
    return (result.stdout or "") + stderr


def _init(env: dict[str, Any]) -> None:
    _run(["init", "--limits", str(env["limits"]), "--db", str(env["db"])])


def seed(env: dict[str, Any], *, days: int = 400, uids: tuple[str, ...] = ()) -> None:
    """A random walk in the store: no drift, so no strategy can have an edge."""
    instruments = uids or ("isin:US0378331005", "isin:GB00B03MLX29")
    pinned = load_hard_limits(env["limits"])
    with Ledger(env["db"], config_hash=pinned.config_hash) as ledger:
        store = BarStore(ledger, root=env["bars"], scale=pinned.limits.data.price_scale)
        for index, uid in enumerate(instruments):
            walk = random.Random(500 + index)
            price = Decimal("100.00")
            bars = []
            for day in range(days):
                opened = BASE + timedelta(days=day)
                price = max(price + Decimal(str(round(walk.gauss(0, 1.1), 4))), Decimal("1.00"))
                close = max(price + Decimal(str(round(walk.gauss(0, 0.7), 4))), Decimal("1.00"))
                bars.append(
                    Bar(
                        instrument_uid=uid,
                        resolution=Resolution.DAILY,
                        bar_open_utc=opened,
                        available_at_utc=opened + timedelta(days=1),
                        ingested_at_utc=opened + timedelta(days=1),
                        provider="fixture",
                        provenance=Provenance.BACKFILL,
                        session=Session.REGULAR,
                        open=price,
                        high=max(price, close) + Decimal("0.5"),
                        low=min(price, close) - Decimal("0.5"),
                        close=close,
                        volume=1_000_000,
                    )
                )
            store.ingest(
                BarBatch(
                    bars=tuple(bars),
                    provider="fixture",
                    symbol=uid,
                    resolution=Resolution.DAILY,
                    requested_start=bars[0].bar_open_utc,
                    requested_end=bars[-1].bar_open_utc,
                )
            )
        store.compact()


# --------------------------------------------------------------------------
# Wiring
# --------------------------------------------------------------------------


def test_the_backtest_commands_are_registered() -> None:
    out = _out(_run(["backtest", "--help"]))
    for command in ("costs", "calibrate", "calibrations"):
        assert command in out, f"backtest {command} is not registered"


def test_costs_needs_no_ledger_and_prints_the_venue_arithmetic(env: dict[str, Any]) -> None:
    """Runnable before anything exists, because it is a property of the venue.

    The number it prints is why the design targets hour-to-day position
    changes, so it should be reachable on a fresh clone.
    """
    result = _run(["backtest", "costs", "--limits", str(env["limits"])])
    assert result.exit_code == 0
    out = _out(result)
    assert "round trip" in out
    assert "edge needed" in out
    # Both the cheap and the expensive jurisdiction, so the stamp-duty
    # difference is visible rather than implied.
    assert "US large-cap" in out
    assert "Irish share" in out
    assert "5-20bps" in out


# --------------------------------------------------------------------------
# tb backtest calibrate
# --------------------------------------------------------------------------


def test_calibrate_needs_a_ledger(env: dict[str, Any]) -> None:
    result = _run(["backtest", "calibrate", *env["args"]])
    assert result.exit_code == 2
    assert "tb init" in _out(result)


def test_calibrate_refuses_an_empty_store(env: dict[str, Any]) -> None:
    """A calibration over no data would pass every check while proving nothing."""
    _init(env)
    result = _run(["backtest", "calibrate", *env["args"]])
    assert result.exit_code == 2
    out = _out(result)
    assert "no instruments" in out
    assert "proving nothing" in out or "tb data backfill" in out


def test_calibrate_passes_on_a_random_walk(env: dict[str, Any]) -> None:
    """The release gate, end to end through the CLI.

    A random walk has no drift, so no null strategy can earn a gross edge, and
    costs must push every net Sharpe at or below the tolerance.
    """
    _init(env)
    seed(env)
    result = _run(["backtest", "calibrate", *env["args"]])
    assert result.exit_code == 0, _out(result)
    out = _out(result)
    assert "calibration passed" in out
    assert "null_always_flat" in out
    assert "null_alternating" in out


def test_calibrate_records_its_verdict_in_the_ledger(env: dict[str, Any]) -> None:
    """A calibration that was run but not recorded cannot be cited by M5's gate."""
    _init(env)
    seed(env)
    _run(["backtest", "calibrate", *env["args"]])

    listed = _run(["backtest", "calibrations", *env["ledger_args"]])
    assert listed.exit_code == 0
    assert "passed" in _out(listed)

    # And the chain is intact over the event it just wrote.
    verified = _run(["ledger", "verify", *env["ledger_args"]])
    assert verified.exit_code == 0


def test_calibrations_reports_when_none_have_run(env: dict[str, Any]) -> None:
    _init(env)
    result = _run(["backtest", "calibrations", *env["ledger_args"]])
    assert result.exit_code == 0
    assert "no calibrations recorded" in _out(result)


def test_an_unreadable_store_is_diagnosed_rather_than_traced(env: dict[str, Any]) -> None:
    """The CI failure this test exists because of.

    The vintage-immutability drill corrupts every Parquet file on purpose, and
    a step running after it inherited the wreckage. The raw pyarrow error names
    a file and a library — it reads like a backtester bug and says nothing
    about what to do. This asserts the store is named as the problem.
    """
    _init(env)
    seed(env, days=40, uids=("isin:US0378331005",))
    for path in Path(env["bars"]).rglob("*.parquet"):
        path.write_text("tampered")

    result = _run(["backtest", "calibrate", *env["args"]])
    assert result.exit_code == 2
    out = _out(result)
    assert "bar store could not be read" in out
    # The catalog disagreement, with the hashes, and the command to run next.
    assert "ALTERED" in out
    assert "tb data audit" in out
    assert "not a statement about the engine" in out


def test_an_unknown_resolution_is_refused(env: dict[str, Any]) -> None:
    _init(env)
    result = _run(["backtest", "calibrate", "--resolution", "weekly", *env["args"]])
    assert result.exit_code == 2
    assert "daily, hourly or minute" in _out(result)


def test_a_missing_vintage_is_refused_by_name(env: dict[str, Any]) -> None:
    """Citing a vintage the ledger does not know is a setup error, not a finding."""
    _init(env)
    seed(env)
    result = _run(["backtest", "calibrate", "--vintage", "vint_nope", *env["args"]])
    assert result.exit_code == 2
    assert "not in the ledger" in _out(result)


def test_calibration_is_reproducible_from_its_seed(env: dict[str, Any]) -> None:
    """Otherwise a calibration cannot be compared across commits.

    Which is the only way to notice that a refactor broke the fill timing.
    """
    _init(env)
    seed(env)
    first = _out(_run(["backtest", "calibrate", "--seed", "11", *env["args"]]))
    second = _out(_run(["backtest", "calibrate", "--seed", "11", *env["args"]]))
    # The per-strategy trade counts and returns must match; the calibration id
    # is content-free and differs, so compare the table body rather than all.
    assert "null_alternating" in first
    for line in first.splitlines():
        if "null_coinflip_0" in line:
            assert line in second
            break
    else:  # pragma: no cover - the fixture always produces this row
        pytest.fail("the coin-flip row was not in the output")
