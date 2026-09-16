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
        (
            "data",
            # Every command, not a sample. Two M2 modules shipped with passing
            # tests and no production caller at all, and the thing that would
            # have caught both is an assertion that the CLI actually reaches
            # them — `regime` and `canary` are here for that reason.
            (
                "backfill",
                "audit",
                "bakeoff",
                "seal",
                "verify",
                "vintages",
                "coverage",
                "actions",
                "canary",
                "regime",
                "reconcile-actions",
            ),
        ),
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


# --------------------------------------------------------------------------
# Unexplained splits: the audit acts rather than advises
# --------------------------------------------------------------------------


def _mapped_instrument(env: dict[str, Any], *, confidence: Any = None) -> None:
    """One instrument, tradable, mapped for alpaca."""
    from tb.core.clock import now_iso
    from tb.data.symbols import Confidence, SymbolMap, SymbolMapping

    pinned = load_hard_limits(env["limits"])
    with Ledger(env["db"], config_hash=pinned.config_hash) as ledger:
        ledger.conn.execute(
            "INSERT INTO instruments (ticker, instrument_type, isin, currency_code,"
            " short_name, full_name, exchange_id, working_schedule_id,"
            " min_trade_quantity, max_open_quantity, added_on, fetched_at, raw_json)"
            " VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)",
            (
                "AAPL_US_EQ",
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
        SymbolMap(ledger, provider="alpaca").upsert(
            SymbolMapping(
                t212_ticker="AAPL_US_EQ",
                data_symbol="AAPL",
                provider="alpaca",
                confidence=confidence or Confidence.CROSS_VERIFIED,
                derivation="suffix_strip",
                currency_code="USD",
            )
        )


def _seed_split(env: dict[str, Any]) -> None:
    """A clean 4-for-1 jump that no action row explains."""
    pinned = load_hard_limits(env["limits"])
    days = [s.day for s in CAL.sessions_between(date(2026, 3, 2), date(2026, 3, 31))]
    with Ledger(env["db"], config_hash=pinned.config_hash) as ledger:
        store = BarStore(ledger, root=env["bars"], scale=pinned.limits.data.price_scale)
        bars = [
            daily(day, close="400.00" if index < len(days) // 2 else "100.00")
            for index, day in enumerate(days)
        ]
        store.ingest(
            BarBatch(
                bars=tuple(bars),
                provider="alpaca",
                symbol="AAPL",
                resolution=Resolution.DAILY,
                requested_start=bars[0].bar_open_utc,
                requested_end=bars[-1].bar_open_utc,
            )
        )


def _mapping(env: dict[str, Any]) -> Any:
    from tb.data.symbols import SymbolMap

    pinned = load_hard_limits(env["limits"])
    with Ledger(env["db"], config_hash=pinned.config_hash) as ledger:
        return SymbolMap(ledger, provider="alpaca").get("AAPL_US_EQ")


def test_an_unexplained_split_actually_blocks_the_symbol(env: dict[str, Any]) -> None:
    """Acted on, not advised about.

    The audit used to print "entries in those symbols should be blocked",
    which is not a control: the next turn of the loop would enter before
    anyone read it. Yahoo back-adjusting its cache without reporting an action
    is its documented behaviour, so this path is the one that fires in
    practice.
    """
    _init(env)
    _mapped_instrument(env)
    before = _mapping(env)
    assert before is not None
    assert before.may_enter, "the fixture should start tradable"

    _seed_split(env)
    result = _run(["data", "audit", *env["args"]])
    assert "unexplained split" in _out(result)
    assert "blocked from new entries" in _out(result)

    after = _mapping(env)
    assert after is not None
    assert after.blocked
    assert not after.may_enter, "the symbol is still enterable after an unexplained split"
    assert "split" in (after.blocked_reason or "")
    # The reason names the ratio and the date, so an operator does not have to
    # decode it — a reason nobody can read is a reason they override.
    assert "for-" in (after.blocked_reason or "")


def test_the_inferred_split_is_recorded_as_a_flagged_action(env: dict[str, Any]) -> None:
    """Recorded, and read back flagged — asserted through the real read path.

    `inferred_from_price_jump` has no column: `corporate_actions` predates the
    flag and `apply_schema` cannot add columns, so `source_provider` is the
    stored discriminator and `from_row` derives the flag from it. Asserting the
    derivation rather than a column is what pins that coupling — the failure to
    catch would be a rename on one side turning the flag silently False.
    """
    _init(env)
    _mapped_instrument(env)
    _seed_split(env)
    _run(["data", "audit", *env["args"]])

    from tb.data.actions import ActionStore
    from tb.data.adjustments import RESIDUAL_DETECTOR, ActionType

    pinned = load_hard_limits(env["limits"])
    with Ledger(env["db"], config_hash=pinned.config_hash) as ledger:
        stored = ActionStore(ledger).actions_for(UID)
        assert stored, "the suspicion was not recorded"
        assert all(a.action_type is ActionType.SPLIT for a in stored)
        assert all(a.inferred_from_price_jump for a in stored)
        assert all(a.source_provider == RESIDUAL_DETECTOR for a in stored)


def test_an_inferred_split_never_adjusts_a_price(env: dict[str, Any]) -> None:
    """The consequence of the flag, which is the only reason to store it.

    A guessed ratio reaching the factor algebra would quadruple every feature
    computed across the jump and size a position off the inference. Excluded —
    but `complete=False`, because the discontinuity is still in the series and
    silently returning the identity factor would claim a clean window.
    """
    _init(env)
    _mapped_instrument(env)
    _seed_split(env)
    _run(["data", "audit", *env["args"]])

    from tb.data.actions import ActionStore
    from tb.data.adjustments import price_factor, volume_factor

    pinned = load_hard_limits(env["limits"])
    with Ledger(env["db"], config_hash=pinned.config_hash) as ledger:
        stored = ActionStore(ledger).actions_for(UID)

    before = min(a.effective_date for a in stored) - timedelta(days=1)
    as_of = datetime.now(UTC) + timedelta(days=1)
    factor = price_factor(stored, at=before, as_of=as_of)
    assert factor.is_identity, "an inferred ratio adjusted a real price"
    assert not factor.complete, "the excluded split was not reported"
    assert factor.missing == tuple(a.action_id for a in stored)
    # The inverse of the identity is the identity, so the volume factor must
    # not pick up a 4x liquidity error from the same guess.
    assert volume_factor(stored, at=before, as_of=as_of).is_identity


def test_an_unmapped_instrument_is_reported_rather_than_silently_skipped(
    env: dict[str, Any],
) -> None:
    """It is unreachable by the trading path — but say so.

    "0 blocked" against "1 detected" would otherwise read as a control that
    ran and found nothing to do.
    """
    _init(env)
    _seed_split(env)  # bars, but no instrument and no mapping
    result = _run(["data", "audit", *env["args"]])
    out = _out(result)
    assert "unexplained split" in out
    assert "could not be blocked" in out


def test_blocking_never_prevents_an_exit(env: dict[str, Any]) -> None:
    """The asymmetry. Refusing to sell over a data problem is worse than it."""
    _init(env)
    _mapped_instrument(env)
    _seed_split(env)
    _run(["data", "audit", *env["args"]])

    from tb.data.symbols import SymbolMap

    pinned = load_hard_limits(env["limits"])
    with Ledger(env["db"], config_hash=pinned.config_hash) as ledger:
        allowed, reason = SymbolMap(ledger, provider="alpaca").may_exit("AAPL_US_EQ")
        assert allowed, f"a blocked symbol refused an exit: {reason}"


# --------------------------------------------------------------------------
# tb data reconcile-actions
# --------------------------------------------------------------------------


def test_reconcile_actions_needs_stored_actions(env: dict[str, Any]) -> None:
    _init(env)
    result = _run(
        ["data", "reconcile-actions", "--limits", str(env["limits"]), "--db", str(env["db"])]
    )
    assert result.exit_code == 0
    assert "tb data actions" in _out(result)


def test_reconcile_actions_is_inconclusive_without_position_history(
    env: dict[str, Any],
) -> None:
    """Exit 1, not 0. "No credit" and "no credit expected" look identical.

    Reporting that as clean would be the wrong answer in the permissive
    direction — it would clear a symbol the check never actually examined.
    """
    _init(env)
    pinned = load_hard_limits(env["limits"])
    with Ledger(env["db"], config_hash=pinned.config_hash) as ledger:
        from tb.data.actions import ActionStore
        from tb.data.adjustments import ActionType, CorporateAction

        ActionStore(ledger).record(
            [
                CorporateAction(
                    action_id="act_div_1",
                    instrument_uid=UID,
                    action_type=ActionType.CASH_DIVIDEND,
                    effective_date=date(2026, 3, 10),
                    known_at_utc=datetime(2026, 3, 1, tzinfo=UTC),
                    source_provider="fixture",
                    gross_amount=Decimal("0.24"),
                    currency="USD",
                )
            ]
        )

    result = _run(
        ["data", "reconcile-actions", "--limits", str(env["limits"]), "--db", str(env["db"])]
    )
    assert result.exit_code == 1
    out = _out(result)
    assert "inconclusive rather than clean" in out
    assert "tb reconcile" in out


# --------------------------------------------------------------------------
# tb data regime
# --------------------------------------------------------------------------
#
# The gate had no production caller at all: `tb.data.regime` was imported only
# by its own test, so the exposure factor existed, was correct, and was applied
# to nothing. These tests are about the command being the caller — and about the
# cold start, which is the expensive half: "no signal" must mean reduced
# exposure, never full.


def seed_reference(env: dict[str, Any], *, sessions: int, rising: bool = True) -> None:
    """Bars for the reference series under its own `sym:` uid.

    `sym:SPY`, not an ISIN: the reference series is not traded, so there is no
    broker instrument record to take an ISIN from, and it is deliberately
    outside the symbol map's tradability gate.
    """
    pinned = load_hard_limits(env["limits"])
    days = [s.day for s in CAL.sessions_between(date(2024, 1, 2), date(2026, 3, 31))][-sessions:]
    with Ledger(env["db"], config_hash=pinned.config_hash) as ledger:
        store = BarStore(ledger, root=env["bars"], scale=pinned.limits.data.price_scale)
        bars = [
            daily(
                day,
                # Rising ends above its own trailing average; falling ends below.
                close=f"{100 + (i if rising else sessions - i) / 10:.2f}",
                provider="alpaca",
                uid="sym:SPY",
            )
            for i, day in enumerate(days)
        ]
        store.ingest(
            BarBatch(
                bars=tuple(bars),
                provider="alpaca",
                symbol="SPY",
                resolution=Resolution.DAILY,
                requested_start=bars[0].bar_open_utc,
                requested_end=bars[-1].bar_open_utc,
            )
        )


def _regime(env: dict[str, Any], *extra: str) -> Any:
    return _run(["data", "regime", *env["args"], *extra])


def test_an_empty_store_reduces_exposure_rather_than_permitting_it(env: dict[str, Any]) -> None:
    """The most expensive default in the system, asserted.

    On a fresh install there are no reference bars. "No signal, so full
    exposure" would apply on day one, to the whole portfolio, at exactly the
    moment a new deployment is least likely to be right about anything.
    """
    _init(env)
    result = _regime(env)
    assert result.exit_code == 1, "an unmeasured gate must not report success"
    out = _out(result)
    assert "unavailable" in out
    assert "x0.5" in out
    assert "not a market signal" in out
    # And it says how to fix it, naming the reference series.
    assert "tb data backfill --data-symbols SPY" in out


def test_too_little_history_is_not_the_same_fact_as_risk_off(env: dict[str, Any]) -> None:
    """Four states, not two. An operator must be able to tell them apart.

    Both reduce exposure, but one means the index is down and the other means
    we cannot see it — and only the second is fixed by backfilling.
    """
    _init(env)
    seed_reference(env, sessions=30)
    result = _regime(env, "--as-of", "2026-04-01T00:00:00Z")
    assert result.exit_code == 1
    out = _out(result)
    assert "insufficient_history" in out
    assert "30 sessions" in out
    assert "x0.5" in out


def test_enough_rising_history_permits_full_exposure(env: dict[str, Any]) -> None:
    """The vacuity check, and the only test here that reaches x1.

    Without it every assertion above would still pass against a gate that
    always answered "unavailable" — proving the fail-closed half while leaving
    the gate useless. Exit 0, because a measured reading is the gate working
    whichever way it points.
    """
    _init(env)
    seed_reference(env, sessions=300, rising=True)
    result = _regime(env, "--as-of", "2026-04-01T00:00:00Z")
    assert result.exit_code == 0, _out(result)
    out = _out(result)
    assert "risk_on" in out
    assert "x1 (full)" in out
    assert "measuring the market" in out


def test_enough_falling_history_reduces_exposure_and_still_exits_zero(
    env: dict[str, Any],
) -> None:
    """`risk_off` is the gate working, not a fault.

    Exiting non-zero here would make a deployment check fail for the whole of
    a bear market — which is when the check matters most and when an operator
    is most likely to start ignoring it.
    """
    _init(env)
    seed_reference(env, sessions=300, rising=False)
    result = _regime(env, "--as-of", "2026-04-01T00:00:00Z")
    assert result.exit_code == 0, _out(result)
    out = _out(result)
    assert "risk_off" in out
    assert "x0.5" in out


def test_a_reading_is_recorded_in_the_ledger_on_every_read(env: dict[str, Any]) -> None:
    """Every read, not only transitions.

    The factor in force at a decision time has to be recoverable from the
    ledger alone. A reading written only on a change cannot answer that for the
    instants in between, which is most of them.
    """
    _init(env)
    seed_reference(env, sessions=30)
    _regime(env, "--as-of", "2026-04-01T00:00:00Z")

    pinned = load_hard_limits(env["limits"])
    with Ledger(env["db"], config_hash=pinned.config_hash) as ledger:
        rows = ledger.conn.execute(
            "SELECT payload_json FROM event_log WHERE event_type = 'data.regime_read'"
        ).fetchall()
        assert len(rows) == 1
        payload = rows[0]["payload_json"]
        # The state is stored alongside the factor: reading x0.5 back on its own
        # could not tell a genuine risk_off from a broken feed.
        assert "insufficient_history" in payload
        assert "sym:SPY" in payload
        assert '"is_measured":false' in payload.replace(" ", "")


def test_the_reading_is_point_in_time(env: dict[str, Any]) -> None:
    """An as-of instant before any bar was knowable sees nothing.

    Same visibility rule as every other read. Without it, a regime reading in a
    backtest would be the one computed from today's data rather than the one
    available then — which is the whole lookahead channel this layer exists to
    close.
    """
    _init(env)
    seed_reference(env, sessions=30)
    result = _regime(env, "--as-of", "2020-01-01T00:00:00Z")
    assert result.exit_code == 1
    assert "no SPY bars knowable" in _out(result)


def test_a_naive_as_of_is_refused(env: dict[str, Any]) -> None:
    """It would be read as UTC while the operator meant local time."""
    _init(env)
    result = _regime(env, "--as-of", "2026-04-01T00:00:00")
    assert result.exit_code == 2
    assert "no timezone" in _out(result)


def test_an_unparseable_as_of_is_refused(env: dict[str, Any]) -> None:
    _init(env)
    result = _regime(env, "--as-of", "last tuesday")
    assert result.exit_code == 2
    assert "not an ISO instant" in _out(result)


def test_regime_needs_a_ledger(env: dict[str, Any]) -> None:
    result = _regime(env)
    assert result.exit_code == 2
    assert "tb init" in _out(result)


# --------------------------------------------------------------------------
# A backfill that fetched nothing is not a success
# --------------------------------------------------------------------------
#
# The first real bake-off run reported success having written zero rows: Yahoo
# throttled all ten requests, `tb data backfill` exited 0 anyway, and the
# failure surfaced four steps later as "no bars in the store" — pointing at
# the store rather than at the feed.


class _RefusingProvider:
    """A provider that fails every fetch, the way a throttled feed does."""

    def __init__(self, *, message: str = "throttled the request") -> None:
        self.message = message
        self.calls = 0

    @property
    def name(self) -> str:
        return "yahoo"

    def fetch_bars(self, symbol: str, **_: Any) -> Any:
        from tb.data.provider import DataError

        self.calls += 1
        raise DataError(f"transport failure on {symbol}: {self.message}")

    def close(self) -> None:
        return None


def test_a_backfill_that_fetched_nothing_exits_non_zero(
    env: dict[str, Any], monkeypatch: pytest.MonkeyPatch
) -> None:
    """Exit 2, naming the feed — not exit 0 with a warning nobody reads.

    This is the honesty that keeps a pipeline diagnosable: the failure has to
    be reported by the step that caused it, in the words of the cause.
    """
    import tb.cli_data as cli_data

    _init(env)
    feed = _RefusingProvider()
    monkeypatch.setattr(cli_data, "_provider", lambda _name, archive=None: feed)

    result = _run(
        [
            "data",
            "backfill",
            "--provider",
            "yahoo",
            "--data-symbols",
            "aapl,msft,nvda",
            # `--no-fx` so the call count is the symbol count: the FX leg
            # fetches through the same provider and would add one.
            "--no-fx",
            *env["args"],
        ]
    )
    assert result.exit_code == 2, _out(result)
    out = _out(result)
    assert "no symbol could be fetched from yahoo" in out
    assert "all 3 request(s) failed" in out
    # The cause, not just the count: a throttle and a 401 need different fixes.
    assert "throttled" in out
    assert feed.calls == 3, "every symbol should still have been attempted"


def test_a_partial_backfill_says_what_it_missed(
    env: dict[str, Any], monkeypatch: pytest.MonkeyPatch
) -> None:
    """Exit 0, because some data arrived — but the gap is stated.

    Anything measured over this store covers the symbols that arrived, and a
    measurement that silently ran over 1 of 3 symbols while reporting on "the
    universe" is the quiet version of the same bug.
    """
    import tb.cli_data as cli_data
    from tb.data.provider import DataError
    from tb.data.providers import CsvFixtureProvider

    _init(env)
    days = [s.day for s in CAL.sessions_between(date(2026, 3, 2), date(2026, 3, 6))]
    bars = [daily(day, provider="yahoo", uid="sym:AAPL") for day in days]

    real = CsvFixtureProvider(bars=bars, provider_name="yahoo")

    class _Flaky:
        @property
        def name(self) -> str:
            return "yahoo"

        def fetch_bars(self, symbol: str, **kwargs: Any) -> Any:
            if symbol.upper() != "AAPL":
                raise DataError(f"transport failure on {symbol}: throttled")
            return real.fetch_bars(symbol, **kwargs)

        def close(self) -> None:
            return None

    monkeypatch.setattr(cli_data, "_provider", lambda _name, archive=None: _Flaky())

    result = _run(
        [
            "data",
            "backfill",
            "--provider",
            "yahoo",
            "--data-symbols",
            "aapl,msft,nvda",
            "--no-fx",
            *env["args"],
        ]
    )
    assert result.exit_code == 0, _out(result)
    out = _out(result)
    assert "symbols fetched" in out
    assert "2 of 3 symbol(s) returned nothing" in out


def test_an_up_to_date_backfill_is_still_a_success(
    env: dict[str, Any], monkeypatch: pytest.MonkeyPatch
) -> None:
    """Zero *rows* is fine; zero *symbols fetched* is not.

    The vacuity guard on the rule above. Re-running a backfill over a store
    that already holds every bar writes nothing, and that must stay exit 0 —
    otherwise the refusal would fire on the most ordinary case there is.
    """
    import tb.cli_data as cli_data
    from tb.data.providers import CsvFixtureProvider

    _init(env)
    days = [s.day for s in CAL.sessions_between(date(2026, 3, 2), date(2026, 3, 6))]
    bars = [daily(day, provider="yahoo", uid="sym:AAPL") for day in days]
    monkeypatch.setattr(
        cli_data,
        "_provider",
        lambda _name, archive=None: CsvFixtureProvider(bars=bars, provider_name="yahoo"),
    )
    args = ["data", "backfill", "--provider", "yahoo", "--data-symbols", "aapl", *env["args"]]

    assert _run(args).exit_code == 0
    second = _run(args)
    assert second.exit_code == 0, _out(second)
    out = _out(second)
    assert "symbols fetched" in out
    assert "1/1" in out
