"""`tb data ...` and `tb universe ...`.

Exit codes are the contract here: 2 for a setup problem the operator has to
fix, 1 for a finding the data itself has, 0 for clean. `tb data audit` exiting
1 on a blocking finding is what lets it sit in a pipeline in front of anything
that trades.

Every test runs without a network. `backfill` and `bakeoff` are the only
commands that would reach out, and both are exercised on their refusal paths —
the happy path needs a provider, which the conformance suite covers.
"""

from __future__ import annotations

from collections.abc import Callable
from datetime import UTC, date, datetime, timedelta
from decimal import Decimal
from pathlib import Path
from typing import Any

import pytest
from typer.testing import CliRunner

from tb.cli import app
from tb.config.loader import load_hard_limits
from tb.data.barstore import BarStore
from tb.data.calendar import TradingCalendar
from tb.data.provider import Bar, BarBatch, Provenance, Resolution, Session
from tb.ledger.store import Ledger

runner = CliRunner()
CAL = TradingCalendar()
UID = "isin:US0378331005"


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


def daily(day: date, *, close: str = "100.00", provider: str = "alpaca", uid: str = UID) -> Bar:
    bar_open = datetime(day.year, day.month, day.day, tzinfo=UTC)
    price = Decimal(close)
    return Bar(
        instrument_uid=uid,
        resolution=Resolution.DAILY,
        bar_open_utc=bar_open,
        available_at_utc=bar_open + timedelta(days=1),
        ingested_at_utc=bar_open + timedelta(days=1),
        provider=provider,
        provenance=Provenance.BACKFILL,
        session=Session.REGULAR,
        open=price,
        high=price,
        low=price,
        close=price,
        volume=1_000_000,
    )


def seed(
    env: dict[str, Any],
    *,
    providers: tuple[str, ...] = ("alpaca",),
    skip: date | None = None,
    end: date = date(2026, 3, 31),
) -> None:
    """Put bars in the store without a network."""
    pinned = load_hard_limits(env["limits"])
    days = [s.day for s in CAL.sessions_between(date(2026, 3, 2), end)]
    with Ledger(env["db"], config_hash=pinned.config_hash) as ledger:
        store = BarStore(ledger, root=env["bars"], scale=pinned.limits.data.price_scale)
        for index, provider in enumerate(providers):
            bars = [
                daily(day, close=f"{100 + index * 0.02 + i / 10:.2f}", provider=provider)
                for i, day in enumerate(days)
                if day != skip
            ]
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


# --------------------------------------------------------------------------
# Wiring
# --------------------------------------------------------------------------


def test_the_data_and_universe_commands_are_registered() -> None:
    for group, expected in (
        ("data", ("backfill", "audit", "bakeoff", "seal", "verify", "vintages")),
        ("universe", ("build",)),
    ):
        out = _out(_run([group, "--help"]))
        for command in expected:
            assert command in out, f"{group} {command} is not registered"


def test_every_data_command_needs_a_ledger(env: dict[str, Any]) -> None:
    """Exit 2, naming `tb init`. A setup problem, not a data problem."""
    for command in (["data", "audit"], ["data", "vintages"], ["universe", "build"]):
        result = _run([*command, *env["args"]])
        assert result.exit_code == 2
        assert "tb init" in _out(result)


def test_an_unknown_provider_is_refused_by_name(env: dict[str, Any]) -> None:
    """No registry mapping arbitrary strings to classes.

    A typo must not be able to silently swap the feed underneath a strategy.
    """
    _init(env)
    result = _run(["data", "backfill", "--provider", "polygon", *env["args"]])
    assert result.exit_code == 2
    assert "unknown provider" in _out(result)
    assert "alpaca, yahoo" in _out(result)


def test_an_unknown_resolution_is_refused(env: dict[str, Any]) -> None:
    _init(env)
    result = _run(["data", "audit", "--resolution", "weekly", *env["args"]])
    assert result.exit_code == 2
    assert "daily, hourly or minute" in _out(result)


# --------------------------------------------------------------------------
# tb data audit
# --------------------------------------------------------------------------


def test_audit_on_an_empty_store_is_clean(env: dict[str, Any]) -> None:
    _init(env)
    result = _run(["data", "audit", *env["args"]])
    assert result.exit_code == 0
    assert "nothing blocking" in _out(result)


def test_audit_reports_gaps_by_cause_and_exits_one_when_blocking(
    env: dict[str, Any],
) -> None:
    """The exit code is what lets this sit in front of anything that trades."""
    _init(env)
    seed(env, skip=date(2026, 3, 10))
    result = _run(["data", "audit", *env["args"]])
    assert result.exit_code == 1
    out = _out(result)
    assert "gaps by cause" in out
    assert "unexplained" in out
    assert "worst coverage" in out


def test_audit_over_two_providers_reports_no_duplicates(env: dict[str, Any]) -> None:
    """The regression an end-to-end run found.

    Two feeds interleaved by bar time reported a `duplicate_bar` for every
    date — alarming, and meaningless. The checks run per provider now.
    """
    _init(env)
    seed(env, providers=("alpaca", "yahoo"))
    result = _run(["data", "audit", *env["args"]])
    assert result.exit_code == 0
    assert "duplicate_bar" not in _out(result)


def test_audit_writes_the_coverage_projection(env: dict[str, Any]) -> None:
    _init(env)
    seed(env, providers=("alpaca", "yahoo"))
    _run(["data", "audit", *env["args"]])

    result = _run(["data", "coverage", *env["args"]])
    assert result.exit_code == 0
    out = _out(result)
    assert "alpaca" in out
    assert "yahoo" in out
    assert "last audited" in out


def test_coverage_before_an_audit_says_so(env: dict[str, Any]) -> None:
    _init(env)
    result = _run(["data", "coverage", *env["args"]])
    assert result.exit_code == 0
    assert "tb data audit" in _out(result)


# --------------------------------------------------------------------------
# tb data seal / verify / vintages
# --------------------------------------------------------------------------


def test_sealing_an_empty_store_exits_one(env: dict[str, Any]) -> None:
    """An empty vintage would be admissible-looking evidence for nothing."""
    _init(env)
    result = _run(["data", "seal", *env["args"]])
    assert result.exit_code == 1
    assert "nothing to seal" in _out(result)


def test_seal_prints_the_vintage_and_its_caveats(env: dict[str, Any]) -> None:
    """The caveats are the point: a labelled biased backtest is usable."""
    _init(env)
    seed(env)
    result = _run(["data", "seal", *env["args"]])
    assert result.exit_code == 0
    out = _out(result)
    assert "vint_" in out
    assert "survivorship" in out
    assert "unmeasured" in out
    assert "vendor_current_view" in out
    assert "cite" in out


def test_seal_keeps_providers_in_separate_partitions(env: dict[str, Any]) -> None:
    _init(env)
    seed(env, providers=("alpaca", "yahoo"))
    result = _run(["data", "seal", *env["args"]])
    assert result.exit_code == 0
    # Two partitions, one per provider, rather than one merged file.
    assert "files" in _out(result)

    listed = _run(["data", "vintages", *env["args"]])
    assert listed.exit_code == 0
    assert "vint_" in _out(listed)


def test_verify_passes_on_an_intact_store(env: dict[str, Any]) -> None:
    _init(env)
    seed(env)
    _run(["data", "seal", *env["args"]])
    result = _run(["data", "verify", *env["args"]])
    assert result.exit_code == 0
    assert "present and unchanged" in _out(result)


def test_verify_exits_one_on_an_altered_partition(env: dict[str, Any]) -> None:
    """The data-layer analogue of `tb ledger verify`, same fail-closed reading."""
    _init(env)
    seed(env)
    _run(["data", "seal", *env["args"]])

    for path in env["bars"].rglob("*.parquet"):
        path.write_bytes(b"tampered")

    result = _run(["data", "verify", *env["args"]])
    assert result.exit_code == 1
    assert "ALTERED" in _out(result)


def test_verify_of_a_named_vintage_reports_what_changed(env: dict[str, Any]) -> None:
    _init(env)
    seed(env)
    sealed = _out(_run(["data", "seal", *env["args"]]))
    vintage_id = next(token for token in sealed.split() if token.startswith("vint_"))

    clean = _run(["data", "verify", vintage_id, *env["args"]])
    assert clean.exit_code == 0
    assert "matches what was sealed" in _out(clean)

    for path in env["bars"].rglob("*.parquet"):
        path.unlink()
    broken = _run(["data", "verify", vintage_id, *env["args"]])
    assert broken.exit_code == 1
    assert "needs re-running" in _out(broken)


def test_verify_of_an_unknown_vintage_exits_one(env: dict[str, Any]) -> None:
    _init(env)
    result = _run(["data", "verify", "vint_nope", *env["args"]])
    assert result.exit_code == 1
    assert "not in the ledger" in _out(result)


def test_vintages_before_sealing_says_so(env: dict[str, Any]) -> None:
    _init(env)
    result = _run(["data", "vintages", *env["args"]])
    assert result.exit_code == 0
    assert "tb data seal" in _out(result)


# --------------------------------------------------------------------------
# tb data bakeoff
# --------------------------------------------------------------------------


def test_bakeoff_on_an_empty_store_names_the_remedy(env: dict[str, Any]) -> None:
    _init(env)
    result = _run(["data", "bakeoff", *env["args"]])
    assert result.exit_code == 2
    assert "tb data backfill" in _out(result)


def test_bakeoff_with_only_one_feed_names_the_missing_one(env: dict[str, Any]) -> None:
    """A feed cannot be baked off against nothing, and the message says which."""
    _init(env)
    seed(env, providers=("alpaca",))
    result = _run(["data", "bakeoff", "--resolution", "daily", *env["args"]])
    assert result.exit_code == 2
    out = _out(result)
    assert "no daily bars from yahoo" in out
    assert "--provider yahoo" in out


def test_bakeoff_reports_the_minimum_viable_edge(env: dict[str, Any]) -> None:
    """The headline number, and it is honest about a short sample."""
    _init(env)
    seed(env, providers=("alpaca", "yahoo"))
    result = _run(["data", "bakeoff", "--resolution", "daily", *env["args"]])
    assert result.exit_code == 0
    out = _out(result)
    assert "minimum viable edge" in out
    assert "worst disagreements" in out
    # 22 sessions is nowhere near enough to conclude anything.
    assert "insufficient_sample" in out
    assert "daily is already permitted" in out


# --------------------------------------------------------------------------
# tb universe build
# --------------------------------------------------------------------------


def test_universe_build_without_instruments_names_the_remedy(
    env: dict[str, Any],
) -> None:
    _init(env)
    result = _run(["universe", "build", *env["args"]])
    assert result.exit_code == 2
    assert "tb symbols audit --refresh" in _out(result)


def test_universe_build_records_a_dated_snapshot(env: dict[str, Any]) -> None:
    from tb.broker.port import Instrument
    from tb.broker.t212.probe import cache_instruments
    from tb.data.symbols import SymbolMap

    _init(env)
    pinned = load_hard_limits(env["limits"])
    with Ledger(env["db"], config_hash=pinned.config_hash) as ledger:
        instruments = (
            Instrument(
                ticker="AAPL_US_EQ",
                instrument_type="STOCK",
                isin="US0378331005",
                currency_code="USD",
                short_name="AAPL",
            ),
            Instrument(
                ticker="MSFT_US_EQ",
                instrument_type="STOCK",
                isin="US5949181045",
                currency_code="USD",
                short_name="MSFT",
            ),
        )
        cache_instruments(ledger, instruments)
        SymbolMap(ledger, provider="alpaca").derive_all(instruments)

    result = _run(["universe", "build", *env["args"]])
    assert result.exit_code == 0
    out = _out(result)
    assert "uni_" in out
    assert "AAPL_US_EQ" in out
    assert "members" in out


def test_universe_build_is_idempotent_within_a_day(env: dict[str, Any]) -> None:
    from tb.broker.port import Instrument
    from tb.broker.t212.probe import cache_instruments
    from tb.data.symbols import SymbolMap

    _init(env)
    pinned = load_hard_limits(env["limits"])
    with Ledger(env["db"], config_hash=pinned.config_hash) as ledger:
        instruments = (
            Instrument(
                ticker="AAPL_US_EQ",
                instrument_type="STOCK",
                isin="US0378331005",
                currency_code="USD",
            ),
        )
        cache_instruments(ledger, instruments)
        SymbolMap(ledger, provider="alpaca").derive_all(instruments)

    _run(["universe", "build", *env["args"]])
    again = _run(["universe", "build", *env["args"]])
    assert again.exit_code == 0
    assert "identical to the snapshot already recorded today" in _out(again)


def test_backfill_without_a_universe_names_the_remedy(env: dict[str, Any]) -> None:
    """It refuses before touching a network, which is the right order."""
    _init(env)
    result = _run(["data", "backfill", "--provider", "yahoo", *env["args"]])
    assert result.exit_code == 2
    assert "tb universe build" in _out(result)


def test_actions_without_a_universe_names_the_remedy(env: dict[str, Any]) -> None:
    _init(env)
    result = _run(["data", "actions", *env["args"]])
    assert result.exit_code == 2
    assert "tb universe build" in _out(result)


def test_backfill_refuses_both_symbol_flags_at_once(env: dict[str, Any]) -> None:
    """They key bars under different identities, so mixing them splits a series."""
    _init(env)
    result = _run(
        [
            "data",
            "backfill",
            "--symbols",
            "AAPL_US_EQ",
            "--data-symbols",
            "AAPL",
            *env["args"],
        ]
    )
    assert result.exit_code == 2
    assert "not both" in _out(result)


def test_backfill_by_data_symbol_needs_no_universe_or_symbol_map(
    env: dict[str, Any], monkeypatch: pytest.MonkeyPatch
) -> None:
    """The bake-off measures data providers, so it must not need the broker.

    Requiring a T212-derived symbol map to fetch a bar made the measurement
    impossible anywhere without broker credentials — including CI, which is the
    only environment here with egress to either feed.
    """
    import tb.cli_data as cli_data
    from tb.data.providers import CsvFixtureProvider

    _init(env)
    days = [s.day for s in CAL.sessions_between(date(2026, 3, 2), date(2026, 3, 6))]
    bars = [
        Bar(
            instrument_uid="sym:AAPL",
            resolution=Resolution.DAILY,
            bar_open_utc=datetime(day.year, day.month, day.day, tzinfo=UTC),
            available_at_utc=datetime(day.year, day.month, day.day, tzinfo=UTC) + timedelta(days=1),
            ingested_at_utc=datetime(day.year, day.month, day.day, tzinfo=UTC) + timedelta(days=1),
            provider="yahoo",
            provenance=Provenance.BACKFILL,
            session=Session.REGULAR,
            open=Decimal("100.00") + Decimal(index),
            high=Decimal("101.00") + Decimal(index),
            low=Decimal("99.00") + Decimal(index),
            close=Decimal("100.50") + Decimal(index),
            volume=1_000_000,
        )
        for index, day in enumerate(days)
    ]
    monkeypatch.setattr(
        cli_data,
        "_provider",
        # Accepts `archive=` because the real factory threads one through:
        # `tb data backfill` archives provider payloads so a Yahoo shape
        # change has a forensic record. The fixture ignores it.
        lambda _name, archive=None: CsvFixtureProvider(bars=bars, provider_name="yahoo"),
    )

    result = _run(
        ["data", "backfill", "--provider", "yahoo", "--data-symbols", "aapl", *env["args"]]
    )
    assert result.exit_code == 0, _out(result)
    output = _out(result)
    assert "research mode" in output
    # Lower-cased on the way in, keyed upper-case: a uid that varied by how the
    # flag was typed would split one instrument's history across two series.
    assert "1 symbol(s)" in output

    with Ledger(env["db"]) as ledger:
        store = BarStore(ledger, root=env["bars"])
        assert "sym:AAPL" in set(store.instruments())


def test_audit_reports_research_keyed_instruments_without_blocking(
    env: dict[str, Any],
) -> None:
    """INFO, not BLOCKING: the bars are real, they just are not ISIN-keyed.

    Worth a line because a sealed vintage includes them, so a backtest could
    cite one without noticing part of its universe was research fixtures.
    """
    _init(env)
    day = CAL.sessions_between(date(2026, 3, 2), date(2026, 3, 3))[0].day
    opened = datetime(day.year, day.month, day.day, tzinfo=UTC)
    with Ledger(env["db"]) as ledger:
        store = BarStore(ledger, root=env["bars"])
        store.ingest(
            BarBatch(
                bars=(
                    Bar(
                        instrument_uid="sym:AAPL",
                        resolution=Resolution.DAILY,
                        bar_open_utc=opened,
                        available_at_utc=opened + timedelta(days=1),
                        ingested_at_utc=opened + timedelta(days=1),
                        provider="yahoo",
                        provenance=Provenance.BACKFILL,
                        session=Session.REGULAR,
                        open=Decimal("100"),
                        high=Decimal("100"),
                        low=Decimal("100"),
                        close=Decimal("100"),
                        volume=1_000_000,
                    ),
                ),
                provider="yahoo",
                symbol="AAPL",
                resolution=Resolution.DAILY,
                requested_start=opened,
                requested_end=opened,
            )
        )

    result = _run(["data", "audit", *env["args"]])
    assert result.exit_code == 0, _out(result)
    assert "keyed by ticker rather than ISIN" in _out(result)


def test_no_command_prints_a_credential(
    env: dict[str, Any], monkeypatch: pytest.MonkeyPatch
) -> None:
    """The standing rule, re-checked on the new surface.

    Alpaca's keys never reach a query string and never reach stdout; this
    asserts the CLI cannot leak one by accident on its refusal paths either.
    """
    monkeypatch.setenv("ALPACA_DATA_KEY_ID", "SECRET-KEY-ID")
    monkeypatch.setenv("ALPACA_DATA_SECRET_KEY", "SECRET-SECRET")
    _init(env)
    seed(env, providers=("alpaca", "yahoo"))

    for command in (
        ["data", "audit"],
        ["data", "seal"],
        ["data", "verify"],
        ["data", "vintages"],
        ["data", "coverage"],
        ["data", "bakeoff", "--resolution", "daily"],
    ):
        out = _out(_run([*command, *env["args"]]))
        assert "SECRET-KEY-ID" not in out
        assert "SECRET-SECRET" not in out
