"""`tb symbols verify` — the command that makes a first position possible.

`Confidence.CROSS_VERIFIED` is reachable only through `SymbolVerifier`, and
`SymbolVerifier` had no production caller. So the tier was structurally
unreachable, `may_enter` was False for every symbol, and `tb symbols audit`
reported "0 cross-verified" while telling the operator an entry was possible.
The two-tier gate fixed the deadlock in the type system and left it in the
pipeline.

`test_verification_makes_the_first_position_possible` is the one that matters:
it asserts `enterable` goes from 0 to 1. Everything else here is about not
verifying something that should not be.
"""

from __future__ import annotations

from collections.abc import Callable
from datetime import UTC, datetime, timedelta
from decimal import Decimal
from pathlib import Path
from typing import Any

import pytest
from typer.testing import CliRunner

from tb.cli import app
from tb.config.loader import load_hard_limits
from tb.core.clock import now_iso
from tb.data.barstore import BarStore
from tb.data.provider import Bar, BarBatch, Provenance, Resolution, Session
from tb.data.symbols import Confidence, SymbolMap, SymbolMapping
from tb.ledger.store import Ledger

runner = CliRunner()
BASE = datetime(2026, 3, 2, tzinfo=UTC)
UID = "isin:US0378331005"
TICKER = "AAPL_US_EQ"


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
        "args": [
            "--limits",
            str(limits),
            "--db",
            str(tmp_path / "ledger.db"),
            "--bars",
            str(tmp_path / "bars"),
        ],
        "ledger_args": ["--limits", str(limits), "--db", str(tmp_path / "ledger.db")],
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
    _run(["init", *env["ledger_args"]])


def seed(
    env: dict[str, Any],
    *,
    providers: tuple[tuple[str, str], ...] = (("alpaca", "0"), ("yahoo", "0.02")),
    session: Session = Session.REGULAR,
    offset_days: int = 0,
    currency: str | None = "USD",
    instrument: bool = True,
    mapping: bool = True,
) -> None:
    """An instrument, a derived mapping, and bars from the given providers.

    `offset_days` shifts the *second* provider's bar periods, which is how the
    "different bar period" refusal is exercised: comparing one feed's Tuesday
    to another's Wednesday measures the price move, not the mapping.
    """
    pinned = load_hard_limits(env["limits"])
    with Ledger(env["db"], config_hash=pinned.config_hash) as ledger:
        if instrument:
            ledger.conn.execute(
                "INSERT INTO instruments (ticker, instrument_type, isin, currency_code,"
                " short_name, full_name, exchange_id, working_schedule_id,"
                " min_trade_quantity, max_open_quantity, added_on, fetched_at, raw_json)"
                " VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)",
                (
                    TICKER,
                    "STOCK",
                    "US0378331005",
                    "USD",
                    "Apple",
                    "Apple Inc.",
                    1,
                    1,
                    "0.1",
                    "1000",
                    None,
                    now_iso(),
                    "{}",
                ),
            )
            ledger.conn.commit()

        if mapping:
            SymbolMap(ledger, provider="alpaca").upsert(
                SymbolMapping(
                    t212_ticker=TICKER,
                    data_symbol="AAPL",
                    provider="alpaca",
                    confidence=Confidence.DERIVED,
                    derivation="suffix_strip",
                    currency_code="USD",
                )
            )

        store = BarStore(ledger, root=env["bars"], scale=pinned.limits.data.price_scale)
        for index, (provider, nudge) in enumerate(providers):
            shift = offset_days if index else 0
            bars = []
            for day in range(20):
                opened = BASE + timedelta(days=day + shift)
                price = Decimal("100.00") + Decimal(day) + Decimal(nudge)
                bars.append(
                    Bar(
                        instrument_uid=UID,
                        resolution=Resolution.DAILY,
                        bar_open_utc=opened,
                        available_at_utc=opened + timedelta(days=1),
                        ingested_at_utc=opened + timedelta(days=1),
                        provider=provider,
                        provenance=Provenance.BACKFILL,
                        session=session if index else Session.REGULAR,
                        open=price,
                        high=price + Decimal("1"),
                        low=price - Decimal("1"),
                        close=price,
                        volume=1_000_000,
                        currency=currency,
                    )
                )
            store.ingest(
                BarBatch(
                    bars=tuple(bars),
                    provider=provider,
                    symbol="AAPL",
                    resolution=Resolution.DAILY,
                    requested_start=bars[0].bar_open_utc,
                    requested_end=bars[-1].bar_open_utc,
                )
            )


def enterable(env: dict[str, Any]) -> int:
    pinned = load_hard_limits(env["limits"])
    with Ledger(env["db"], config_hash=pinned.config_hash) as ledger:
        rows = SymbolMap(ledger, provider="alpaca").all()
        return sum(1 for m in rows if m.may_enter)


# --------------------------------------------------------------------------
# The whole point
# --------------------------------------------------------------------------


def test_verification_makes_the_first_position_possible(env: dict[str, Any]) -> None:
    """0 enterable before, 1 after. The deadlock, actually broken.

    Before this command existed, `CROSS_VERIFIED` was unreachable in
    production: nothing called the only code path that sets it. A symbol map
    full of `DERIVED` mappings means nothing is tradable, on a fresh install,
    forever.
    """
    _init(env)
    seed(env)
    assert enterable(env) == 0, "the fixture should start untradable"

    result = _run(["symbols", "verify", *env["args"]])
    assert result.exit_code == 0, _out(result)
    out = _out(result)
    assert "newly cross-verified" in out
    assert "floor-notional entry is now possible" in out

    assert enterable(env) == 1, "verification did not make anything enterable"


def test_the_audit_stops_contradicting_itself(env: dict[str, Any]) -> None:
    """It used to say "0 cross-verified, so an entry is possible"."""
    _init(env)
    seed(env)
    _run(["symbols", "verify", *env["args"]])
    out = _out(_run(["symbols", "audit", *env["ledger_args"]]))
    assert "1 symbol(s) are cross-verified" in out


def test_the_weak_tier_does_not_grant_full_size(env: dict[str, Any]) -> None:
    """Two feeds agreeing says they describe the same instrument.

    It does not say the broker agrees that the ticker we are about to send
    maps to it, which is what full size would require.
    """
    _init(env)
    seed(env)
    _run(["symbols", "verify", *env["args"]])
    pinned = load_hard_limits(env["limits"])
    with Ledger(env["db"], config_hash=pinned.config_hash) as ledger:
        mapping = SymbolMap(ledger, provider="alpaca").get(TICKER)
        assert mapping is not None
        assert mapping.confidence is Confidence.CROSS_VERIFIED
        assert mapping.may_enter
        assert not mapping.permits_full_size


# --------------------------------------------------------------------------
# What must not verify
# --------------------------------------------------------------------------


def test_a_feed_cannot_corroborate_itself(env: dict[str, Any]) -> None:
    """Otherwise every symbol in the universe verifies on no evidence."""
    _init(env)
    seed(env)
    result = _run(
        ["symbols", "verify", "--primary", "alpaca", "--secondary", "alpaca", *env["args"]]
    )
    assert result.exit_code == 2
    assert "agreeing with itself" in _out(result)
    assert enterable(env) == 0


def test_a_price_disagreement_beyond_the_band_does_not_verify(env: dict[str, Any]) -> None:
    """A mismapped ticker is the failure this gate exists to catch."""
    _init(env)
    # Second feed 30% away: a different company, not a rounding difference.
    seed(env, providers=(("alpaca", "0"), ("yahoo", "40")))
    result = _run(["symbols", "verify", *env["args"]])
    assert result.exit_code == 0
    assert enterable(env) == 0, "a 40-point disagreement must not verify"
    assert "refused" in _out(result)


def test_extended_hours_bars_are_not_paired(env: dict[str, Any]) -> None:
    """An after-hours print can sit hundreds of bps from the regular close.

    Pairing across sessions fails for a reason that has nothing to do with the
    mapping, which would block half the universe every earnings night.
    """
    _init(env)
    seed(env, session=Session.EXTENDED)
    result = _run(["symbols", "verify", *env["args"]])
    assert result.exit_code == 0
    out = _out(result)
    assert "no comparable bar pair" in out
    assert enterable(env) == 0


def test_bars_from_different_periods_are_not_paired(env: dict[str, Any]) -> None:
    """Comparing Tuesday to Wednesday measures the price move, not the mapping."""
    _init(env)
    seed(env, offset_days=60)
    result = _run(["symbols", "verify", *env["args"]])
    assert result.exit_code == 0
    assert "no comparable bar pair" in _out(result)
    assert enterable(env) == 0


def test_a_single_feed_degrades_rather_than_deadlocks(env: dict[str, Any]) -> None:
    """One unreachable feed must not verify anything — and must say why.

    The failure mode to avoid is a silent zero: the operator needs to be told
    that cross-verification needs a second backfill, not left looking at an
    empty table.
    """
    _init(env)
    seed(env, providers=(("alpaca", "0"),))
    result = _run(["symbols", "verify", *env["args"]])
    assert result.exit_code == 0
    out = _out(result)
    assert enterable(env) == 0
    assert "two independent" in out
    assert "tb data backfill" in out


def test_a_missing_instrument_record_is_refused_by_name(env: dict[str, Any]) -> None:
    """Reference cross-matching needs the broker's own record of the instrument."""
    _init(env)
    seed(env, instrument=False)
    result = _run(["symbols", "verify", *env["args"]])
    assert result.exit_code == 0
    assert "no cached instrument record" in _out(result)
    assert enterable(env) == 0


# --------------------------------------------------------------------------
# Setup refusals
# --------------------------------------------------------------------------


def test_verify_needs_a_ledger(env: dict[str, Any]) -> None:
    result = _run(["symbols", "verify", *env["args"]])
    assert result.exit_code == 2
    assert "tb init" in _out(result)


def test_verify_needs_mapped_symbols(env: dict[str, Any]) -> None:
    _init(env)
    result = _run(["symbols", "verify", *env["args"]])
    assert result.exit_code == 2
    assert "tb symbols audit" in _out(result)


def test_an_unknown_resolution_is_refused(env: dict[str, Any]) -> None:
    _init(env)
    seed(env)
    result = _run(["symbols", "verify", "--resolution", "weekly", *env["args"]])
    assert result.exit_code == 2
    assert "daily, hourly or minute" in _out(result)


def test_verify_is_idempotent(env: dict[str, Any]) -> None:
    """Safe to run repeatedly: it reads the store and reaches no network."""
    _init(env)
    seed(env)
    _run(["symbols", "verify", *env["args"]])
    first = enterable(env)
    _run(["symbols", "verify", *env["args"]])
    assert enterable(env) == first == 1
