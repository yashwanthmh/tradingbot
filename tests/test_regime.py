"""The regime gate.

The tests that matter here are the negative ones. `RISK_OFF` working is
ordinary; what this file is really for is establishing that **no absence of
information can produce full exposure** — not an empty store, not a short
history, not a stale bar, and not a state added later by someone who did not
read the module docstring.

That default applies on day one, to the entire portfolio, at the moment a
deployment is least likely to be right about anything. It is the most
expensive single default in the system, which is why it is asserted by
enumerating the enum rather than by testing the paths someone thought of.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from decimal import Decimal
from pathlib import Path

import pytest
import yaml

from tb.config.hard_limits import HardLimits
from tb.config.loader import load_hard_limits
from tb.data.asof import InMemoryBarSource
from tb.data.provider import Bar, Provenance, Resolution, Session
from tb.data.regime import (
    RegimeError,
    RegimeGate,
    RegimeReading,
    RegimeState,
)

BASE = datetime(2024, 1, 2, tzinfo=UTC)
REFERENCE_LIMITS = Path("config/hard_limits.yaml")


@pytest.fixture
def gate(limits_file: Path) -> RegimeGate:
    return RegimeGate(load_hard_limits(limits_file).limits)


def spy_bar(offset: int, close: str, *, uid: str) -> Bar:
    opened = BASE + timedelta(days=offset)
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
        high=price + Decimal("0.5"),
        low=price - Decimal("0.5"),
        close=price,
        volume=50_000_000,
    )


def series(gate: RegimeGate, closes: list[str]) -> InMemoryBarSource:
    return InMemoryBarSource(
        bars=[spy_bar(i, c, uid=gate.instrument_uid) for i, c in enumerate(closes)]
    )


def rising(gate: RegimeGate, n: int) -> InMemoryBarSource:
    """A long uptrend: the last close sits well above its own average."""
    return series(gate, [str(Decimal("100") + Decimal(i)) for i in range(n)])


def falling(gate: RegimeGate, n: int) -> InMemoryBarSource:
    return series(gate, [str(Decimal("1000") - Decimal(i)) for i in range(n)])


def as_of_after(n: int) -> datetime:
    """An instant at which all `n` bars are knowable, and none is stale."""
    return BASE + timedelta(days=n, hours=2)


# --------------------------------------------------------------------------
# The default that matters
# --------------------------------------------------------------------------


def test_only_risk_on_permits_full_exposure() -> None:
    """Asserted by enumeration, so a fifth state cannot inherit full exposure.

    Testing the paths someone thought of is exactly how a new enum member
    ships with the wrong default.
    """
    permitting = [s for s in RegimeState if s.permits_full_exposure]
    assert permitting == [RegimeState.RISK_ON]


def test_a_non_risk_on_reading_cannot_be_constructed_with_full_exposure() -> None:
    """The invariant lives in `__post_init__`, so every path goes through it.

    A future branch that computed the state correctly and forgot to reduce the
    factor raises here rather than quietly authorising full size.
    """
    for state in RegimeState:
        if state.permits_full_exposure:
            continue
        with pytest.raises(RegimeError, match="full-exposure factor"):
            RegimeReading(
                as_of=BASE,
                state=state,
                exposure_factor=Decimal(1),
                reference_symbol="SPY",
                instrument_uid="sym:SPY",
                ma_days=200,
                n_sessions_seen=0,
            )


def test_an_exposure_factor_above_one_is_refused() -> None:
    """A factor over 1 would *increase* exposure past what the allocator sized."""
    with pytest.raises(RegimeError, match=r"outside \[0, 1\]"):
        RegimeReading(
            as_of=BASE,
            state=RegimeState.RISK_ON,
            exposure_factor=Decimal("1.5"),
            reference_symbol="SPY",
            instrument_uid="sym:SPY",
            ma_days=200,
            n_sessions_seen=300,
        )


def test_an_empty_store_reduces_exposure_rather_than_permitting_it(
    gate: RegimeGate,
) -> None:
    """Day one. Not seeing the index is not the same as the index being fine."""
    reading = gate.read(InMemoryBarSource(bars=[]), as_of=BASE)
    assert reading.state is RegimeState.UNAVAILABLE
    assert reading.exposure_factor == gate.reduced_factor
    assert reading.reduced
    assert not reading.state.is_measured
    assert "not the same as" in reading.detail


def test_a_short_history_reduces_exposure(gate: RegimeGate) -> None:
    """The cold start the plan calls the highest-leverage number in the system.

    A 200-day average needs ~10 months. Before that, "insufficient history"
    must mean reduced exposure — never "no signal, so go ahead".
    """
    n = gate.min_sessions // 2
    reading = gate.read(rising(gate, n), as_of=as_of_after(n))
    assert reading.state is RegimeState.INSUFFICIENT_HISTORY
    assert reading.exposure_factor == gate.reduced_factor
    assert reading.n_sessions_seen == n
    assert "must never read as" in reading.detail


def test_a_stale_reference_bar_reduces_exposure(gate: RegimeGate) -> None:
    """A regime reading from a months-old bar is not a regime reading."""
    n = gate.min_sessions + 10
    reading = gate.read(rising(gate, n), as_of=as_of_after(n) + timedelta(days=30))
    assert reading.state is RegimeState.UNAVAILABLE
    assert reading.exposure_factor == gate.reduced_factor
    assert "stale" in reading.detail


def test_a_long_weekend_does_not_make_the_reference_stale(gate: RegimeGate) -> None:
    """The bound is generous in wall time and tight in sessions.

    Four days covers a Friday close read on the Tuesday after a Monday
    holiday. Tighter, and the gate would halve exposure every long weekend.
    """
    n = gate.min_sessions + 10
    reading = gate.read(rising(gate, n), as_of=as_of_after(n) + timedelta(days=3))
    assert reading.state.is_measured


# --------------------------------------------------------------------------
# The measured states
# --------------------------------------------------------------------------


def test_an_index_above_its_average_permits_full_exposure(gate: RegimeGate) -> None:
    n = gate.min_sessions + 60
    reading = gate.read(rising(gate, n), as_of=as_of_after(n))
    assert reading.state is RegimeState.RISK_ON
    assert reading.exposure_factor == Decimal(1)
    assert not reading.reduced
    assert reading.last_close is not None
    assert reading.moving_average is not None
    assert reading.last_close > reading.moving_average


def test_an_index_below_its_average_reduces_exposure(gate: RegimeGate) -> None:
    n = gate.min_sessions + 60
    reading = gate.read(falling(gate, n), as_of=as_of_after(n))
    assert reading.state is RegimeState.RISK_OFF
    assert reading.exposure_factor == gate.reduced_factor
    assert reading.last_close is not None
    assert reading.moving_average is not None
    assert reading.last_close < reading.moving_average


def test_the_factor_scales_a_sized_notional(gate: RegimeGate) -> None:
    n = gate.min_sessions + 60
    risk_off = gate.read(falling(gate, n), as_of=as_of_after(n))
    assert risk_off.scale(Decimal("1000")) == Decimal("1000") * gate.reduced_factor

    risk_on = gate.read(rising(gate, n), as_of=as_of_after(n))
    assert risk_on.scale(Decimal("1000")) == Decimal("1000")


# --------------------------------------------------------------------------
# Identity and point-in-time behaviour
# --------------------------------------------------------------------------


def test_the_reference_series_has_its_own_ticker_keyed_identity(gate: RegimeGate) -> None:
    """It cannot live in symbol_map and can never be broker-verified.

    `symbol_map` is keyed on `t212_ticker`, and verification needs a held
    position — which the bot never takes in its own reference index. The `sym:`
    uid is self-marking, so an audit finds it and the trading path never
    resolves to it.
    """
    assert gate.instrument_uid == "sym:SPY"
    assert gate.instrument_uid.startswith("sym:")
    assert gate.reference_symbol == "SPY"


def test_the_reading_is_point_in_time(gate: RegimeGate) -> None:
    """The reading available *then*, not the one computed from today's data.

    A backtest whose regime gate saw the future would have had its exposure
    halved on exactly the days it mattered, and doubled on the others.
    """
    n = gate.min_sessions + 100
    # A series that rises then collapses below its average.
    closes = [str(Decimal("100") + Decimal(i)) for i in range(n)]
    closes += [str(Decimal("50"))] * 40
    source = series(gate, closes)

    early = gate.read(source, as_of=as_of_after(n))
    late = gate.read(source, as_of=as_of_after(len(closes)))
    assert early.state is RegimeState.RISK_ON
    assert late.state is RegimeState.RISK_OFF
    # The early reading saw fewer bars, because the later ones were not
    # knowable yet.
    assert early.n_sessions_seen < late.n_sessions_seen


def test_a_naive_as_of_is_a_programming_error(gate: RegimeGate) -> None:
    """The one thing this raises on. Everything else returns a reduced reading."""
    with pytest.raises(RegimeError, match="timezone-aware"):
        gate.read(InMemoryBarSource(bars=[]), as_of=datetime(2024, 6, 1))  # noqa: DTZ001


def test_read_bars_matches_read(gate: RegimeGate) -> None:
    n = gate.min_sessions + 60
    source = rising(gate, n)
    moment = as_of_after(n)
    assert (
        gate.read(source, as_of=moment).state
        == gate.read_bars(list(source.bars), as_of=moment).state
    )


# --------------------------------------------------------------------------
# Config coupling
# --------------------------------------------------------------------------


def test_the_gate_reads_the_pinned_config_not_loose_numbers(gate: RegimeGate) -> None:
    """So the factor cannot drift from the hash-pinned limits."""
    limits = gate.limits
    assert gate.ma_days == limits.regime.exposure_ma_days
    assert gate.reduced_factor == Decimal(str(limits.regime.exposure_factor_below_ma))
    assert gate.min_sessions == limits.data.min_history_days_for_regime


def test_a_gate_permitted_less_history_than_its_own_average_is_refused() -> None:
    """Already enforced by the config validator; asserted here as the reason.

    A gate running on history shorter than its moving average produces a
    number that looks like a signal and is not.
    """
    payload = yaml.safe_load(REFERENCE_LIMITS.read_text(encoding="utf-8"))
    payload["data"]["min_history_days_for_regime"] = 50
    payload["regime"]["exposure_ma_days"] = 200
    with pytest.raises(Exception, match=r"below regime\.exposure_ma_days"):
        HardLimits.model_validate(payload)


def test_a_zero_exposure_factor_is_permitted(limits_file: Path) -> None:
    """Flattening entirely below the average is a legitimate policy.

    Zero is in range; the refusal is only for factors above 1.
    """
    payload = yaml.safe_load(Path(limits_file).read_text(encoding="utf-8"))
    payload["regime"]["exposure_factor_below_ma"] = 0.0
    target = Path(limits_file).parent / "zero_regime.yaml"
    target.write_text(yaml.safe_dump(payload), encoding="utf-8")

    gate = RegimeGate(load_hard_limits(target).limits)
    n = gate.min_sessions + 60
    reading = gate.read(falling(gate, n), as_of=as_of_after(n))
    assert reading.exposure_factor == Decimal(0)
    assert reading.scale(Decimal("1000")) == Decimal(0)
