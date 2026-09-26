"""The RL interface: a policy trades through the one strategy path or not at all.

What has to hold: a policy satisfies the one `Strategy` interface; it sees the
features the snapshot hash covers and its own position, and nothing else — not
the window, not the account, no reward; it is never asked about a missing
feature, and a held position with one is exited; its target maps onto enter,
exit and hold exactly as a spec's conditions do, with its declared edge on an
entry; the conformance check passes a well-formed policy and names each way a
broken one fails; a policy can be backtested through the real engine; and no
module outside the RL package imports it, so nothing can fund one.
"""

from __future__ import annotations

import ast
import dataclasses
import math
import secrets
import time
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from decimal import Decimal
from pathlib import Path
from typing import Any

import pytest

import tb
from tb.backtest.costs import CostModel, Jurisdiction
from tb.backtest.engine import Backtester, InstrumentMeta
from tb.config.loader import load_hard_limits
from tb.data.asof import BarWindow, ForwardOnlyReader, InMemoryBarSource
from tb.data.provider import Bar, Provenance, Resolution, Session
from tb.features.pipeline import FeatureSnapshot, FeatureSpec, make_spec
from tb.research.selection import expected_max_sharpe
from tb.strategy.base import Action, PositionState, Strategy, StrategyError
from tb.strategy.rl.conformance import check_policy
from tb.strategy.rl.policy import Observation, Policy, PolicyStrategy, Target, policy_pipeline
from tests.conftest import REFERENCE_LIMITS

UID = "isin:US0378331005"
BASE = datetime(2024, 1, 2, tzinfo=UTC)
LIMITS = load_hard_limits(REFERENCE_LIMITS).limits
MAX_EDGE = Decimal(str(LIMITS.costs.max_expected_edge_bps))


def _bar(day: int, close: Decimal) -> Bar:
    opened = BASE + timedelta(days=day)
    return Bar(
        instrument_uid=UID,
        resolution=Resolution.DAILY,
        bar_open_utc=opened,
        available_at_utc=opened + timedelta(days=1),
        ingested_at_utc=opened + timedelta(days=1),
        provider="fixture",
        provenance=Provenance.BACKFILL,
        session=Session.REGULAR,
        open=close,
        high=close + Decimal("0.50"),
        low=close - Decimal("0.50"),
        close=close,
        volume=1_000_000,
    )


def _bars(days: int) -> list[Bar]:
    """A slow swing, so a close crosses its ten-day mean both ways."""
    return [
        _bar(day, Decimal("100") + Decimal(str(round(8 * math.sin(day / 6), 4))))
        for day in range(days)
    ]


def _times(days: int) -> list[datetime]:
    return [BASE + timedelta(days=day + 1, hours=1) for day in range(days)]


def _reader(days: int) -> ForwardOnlyReader:
    return ForwardOnlyReader(
        source=InMemoryBarSource(bars=_bars(days)),
        resolution=Resolution.DAILY,
        instrument_uids=(UID,),
    )


def _windows(days: int = 60) -> list[BarWindow]:
    reader = _reader(days)
    return [reader.advance_to(at) for at in _times(days)]


# --------------------------------------------------------------------------
# Reference policies: one well-formed, and one per way to be malformed
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class AboveAverage:
    """Long while the close is above its ten-day mean. A shape, not a claim."""

    policy_id: str = "ref_above_average"
    version: int = 1
    expected_edge_bps: Decimal = Decimal("150")
    min_holding_minutes: int = 0

    @property
    def features(self) -> tuple[FeatureSpec, ...]:
        return (make_spec("last", 1, name="close"), make_spec("sma", 10, name="sma_10"))

    def act(self, observation: Observation) -> Target:
        close, mean = observation.features["close"], observation.features["sma_10"]
        return Target.LONG if close > mean else Target.FLAT


@dataclass(frozen=True)
class Learning(AboveAverage):
    """Changes its mind about an identical input: hidden state across calls."""

    calls: list[int] = field(default_factory=list)

    def act(self, observation: Observation) -> Target:
        self.calls.append(1)
        return Target.LONG if len(self.calls) % 2 else Target.FLAT


class Unseeded(AboveAverage):
    def act(self, observation: Observation) -> Target:
        return Target.LONG if secrets.randbelow(2) else Target.FLAT


class Slow(AboveAverage):
    def act(self, observation: Observation) -> Target:
        time.sleep(0.005)
        return super().act(observation)


class Raising(AboveAverage):
    def act(self, observation: Observation) -> Target:
        raise ValueError("the policy failed")


class Writer(AboveAverage):
    """Tries to leave a note in what it was shown."""

    def act(self, observation: Observation) -> Target:
        observation.features["close"] = Decimal(0)  # type: ignore[index]
        return Target.FLAT


class Stringly(AboveAverage):
    def act(self, observation: Observation) -> Target:
        return "long"  # type: ignore[return-value]


@dataclass(frozen=True)
class HandRolled(AboveAverage):
    """A feature whose computation is the policy's own code."""

    @property
    def features(self) -> tuple[FeatureSpec, ...]:
        return (FeatureSpec(name="close", lookback=1, compute=lambda closes: closes[-1]),)


@dataclass(frozen=True)
class Blind(AboveAverage):
    """Observes a lookback no test window is long enough to fill."""

    @property
    def features(self) -> tuple[FeatureSpec, ...]:
        return (make_spec("sma", 500, name="sma_500"),)


@dataclass(frozen=True)
class Counting(AboveAverage):
    """Well-formed, and counts how often it was asked."""

    asked: list[Observation] = field(default_factory=list)

    def act(self, observation: Observation) -> Target:
        self.asked.append(observation)
        return super().act(observation)


def _check(policy: Policy, **kw: Any) -> Any:
    return check_policy(policy, windows=_windows(), instrument_uid=UID, max_edge_bps=MAX_EDGE, **kw)


# --------------------------------------------------------------------------
# The adapter: a policy is a strategy, and decides as a spec would
# --------------------------------------------------------------------------


def _snapshot(policy: Policy, window: BarWindow) -> FeatureSnapshot:
    return policy_pipeline(policy).compute(window, UID)


def _held(window: BarWindow, minutes: int) -> PositionState:
    return PositionState(
        instrument_uid=UID, quantity=Decimal(1), entry_at=window.as_of - timedelta(minutes=minutes)
    )


def _window_where(policy: AboveAverage, target: Target) -> BarWindow:
    for window in _windows():
        snapshot = _snapshot(policy, window)
        if snapshot.complete:
            values = {name: snapshot.get(name) for name in ("close", "sma_10")}
            observation = Observation(features=values, holding=False, held_minutes=None)  # type: ignore[arg-type]
            if policy.act(observation) is target:
                return window
    raise AssertionError(f"no window where the reference policy wants {target}")


def test_a_policy_is_a_strategy() -> None:
    strategy = PolicyStrategy(policy=AboveAverage())
    assert isinstance(strategy, Strategy)
    assert isinstance(AboveAverage(), Policy)
    assert strategy.required_features == ("close", "sma_10")
    assert (strategy.strategy_id, strategy.version) == ("ref_above_average", 1)


def test_a_target_becomes_the_action_a_spec_would_take() -> None:
    policy = AboveAverage()
    strategy = PolicyStrategy(policy=policy)
    flat = PositionState(instrument_uid=UID)
    long_window = _window_where(policy, Target.LONG)
    flat_window = _window_where(policy, Target.FLAT)

    enter = strategy.decide(
        snapshot=_snapshot(policy, long_window), window=long_window, position=flat
    )
    assert enter.action is Action.ENTER and enter.expected_edge_bps == policy.expected_edge_bps
    assert enter.feature_snapshot_hash == _snapshot(policy, long_window).snapshot_hash
    assert enter.evidence == {"target": "long"}

    stay_flat = strategy.decide(
        snapshot=_snapshot(policy, flat_window), window=flat_window, position=flat
    )
    assert stay_flat.action is Action.HOLD

    leave = strategy.decide(
        snapshot=_snapshot(policy, flat_window),
        window=flat_window,
        position=_held(flat_window, 60),
    )
    assert leave.action is Action.EXIT and leave.expected_edge_bps == 0

    stay_long = strategy.decide(
        snapshot=_snapshot(policy, long_window),
        window=long_window,
        position=_held(long_window, 60),
    )
    assert stay_long.action is Action.HOLD, "no adds: long while long is a hold"


def test_the_policy_is_not_asked_inside_its_own_minimum_hold() -> None:
    policy = Raising(min_holding_minutes=120)
    window = _windows()[-1]
    decision = PolicyStrategy(policy=policy).decide(
        snapshot=_snapshot(policy, window), window=window, position=_held(window, 30)
    )
    assert decision.action is Action.HOLD and "minimum" in decision.rationale


def test_a_missing_feature_is_never_shown_and_never_keeps_risk_on() -> None:
    policy = Counting()
    strategy = PolicyStrategy(policy=policy)
    early = _windows()[3]  # too short for a ten-day mean
    snapshot = _snapshot(policy, early)
    assert not snapshot.complete

    flat = strategy.decide(snapshot=snapshot, window=early, position=PositionState(UID))
    assert flat.action is Action.HOLD
    held = strategy.decide(snapshot=snapshot, window=early, position=_held(early, 60))
    assert held.action is Action.EXIT and "sma_10" in held.evidence["unknown"]
    assert policy.asked == [], "the policy was never shown the gap"


def test_an_entry_needs_a_declared_edge_and_an_answer_needs_to_be_a_target() -> None:
    window = _window_where(AboveAverage(), Target.LONG)
    free = AboveAverage(expected_edge_bps=Decimal(0))
    with pytest.raises(StrategyError, match="positive expected edge"):
        PolicyStrategy(policy=free).decide(
            snapshot=_snapshot(free, window), window=window, position=PositionState(UID)
        )
    with pytest.raises(StrategyError, match="not a Target"):
        PolicyStrategy(policy=Stringly()).decide(
            snapshot=_snapshot(Stringly(), window), window=window, position=PositionState(UID)
        )


def test_what_a_policy_sees_is_features_and_its_own_position_and_nothing_else() -> None:
    """Structural, so widening it is a deliberate act: a field for the window
    would let a policy decide on inputs no snapshot hash covers; one for the
    account would let it size itself; one for a reward would let it learn."""
    assert [f.name for f in dataclasses.fields(Observation)] == [
        "features",
        "holding",
        "held_minutes",
    ]
    members = {
        name
        for name in vars(Policy)
        if not name.startswith("_") and name not in {"register", "__init__"}
    }
    assert members == {
        "policy_id",
        "version",
        "features",
        "expected_edge_bps",
        "min_holding_minutes",
        "act",
    }, "a policy has no update, reward or training hook: it is frozen while it trades"


def test_what_a_policy_is_shown_is_read_only() -> None:
    counted = Counting()
    window = _window_where(AboveAverage(), Target.LONG)
    PolicyStrategy(policy=counted).decide(
        snapshot=_snapshot(counted, window), window=window, position=PositionState(UID)
    )
    [observation] = counted.asked
    with pytest.raises(TypeError):
        observation.features["close"] = Decimal(0)  # type: ignore[index]
    assert (observation.holding, observation.held_minutes) == (False, None)


# --------------------------------------------------------------------------
# The conformance check
# --------------------------------------------------------------------------


def test_a_well_formed_policy_conforms() -> None:
    report = _check(AboveAverage())
    assert report.conforms, report.summary()
    assert report.cases == 120 and 0 < report.asked <= 120
    assert "conforms" in report.summary()


@pytest.mark.parametrize(
    ("policy", "rule"),
    [
        (Learning(), "deterministic"),
        (Unseeded(), "deterministic"),
        (Raising(), "raised"),
        (Writer(), "raised"),
        (Stringly(), "action_space"),
        (HandRolled(), "features"),
        (Blind(), "vacuous"),
        (AboveAverage(expected_edge_bps=Decimal(0)), "declared_edge"),
        (AboveAverage(expected_edge_bps=MAX_EDGE + 1), "declared_edge"),
        (AboveAverage(min_holding_minutes=-1), "min_hold"),
        (AboveAverage(policy_id="Not An Id"), "identity"),
        (AboveAverage(version=0), "identity"),
    ],
    ids=lambda value: value if isinstance(value, str) else type(value).__name__,
)
def test_a_broken_policy_is_named_for_what_it_broke(policy: Policy, rule: str) -> None:
    report = _check(policy)
    assert not report.conforms
    assert report.rules_broken == (rule,), report.summary()


def test_a_slow_policy_misses_its_time_budget() -> None:
    report = _check(Slow(), time_budget=timedelta(milliseconds=1))
    assert report.rules_broken == ("time_budget",), report.summary()


def test_one_fault_seen_many_times_is_reported_once_with_a_count() -> None:
    report = _check(Raising())
    lines = [v for v in report.violations if v.rule == "raised"]
    assert len(lines) == 4 and "more like it" in lines[-1].detail


def test_structure_is_checked_before_behaviour() -> None:
    """A policy with a malformed edge is not run: every decision it made would
    report the same fault again, as something else."""
    report = _check(Raising(expected_edge_bps=Decimal(0)))
    assert report.rules_broken == ("declared_edge",) and report.cases == 0


# --------------------------------------------------------------------------
# The research path, and the closed door to the book
# --------------------------------------------------------------------------


def test_a_policy_backtests_through_the_real_engine(limits_file: Path) -> None:
    policy = AboveAverage()
    engine = Backtester(
        cost_model=CostModel(load_hard_limits(limits_file).limits),
        pipeline=policy_pipeline(policy),
        instruments={
            UID: InstrumentMeta(instrument_uid=UID, currency="USD", jurisdiction=Jurisdiction.US)
        },
    )
    result = engine.run(
        strategy=PolicyStrategy(policy=policy), reader=_reader(90), decision_times=_times(90)
    )
    assert result.metrics.n_trades > 0, "the policy traded through the same engine as a spec"


def test_nothing_outside_the_rl_package_imports_it() -> None:
    """The closed door to the funded book, held structurally.

    `funded_book` builds strategies from registered specs, and a policy is not
    a spec; this keeps it so. Opening the door is a deliberate change to this
    test, made once docs/decisions/0003-rl-deferred.md's conditions are met.
    """
    root = Path(tb.__file__).resolve().parent
    rl = root / "strategy" / "rl"
    importers: list[str] = []
    for source in root.rglob("*.py"):
        if rl in source.parents:
            continue
        for node in ast.walk(ast.parse(source.read_text(encoding="utf-8"))):
            names: list[str] = []
            if isinstance(node, ast.ImportFrom) and node.module:
                names = [node.module, *(f"{node.module}.{a.name}" for a in node.names)]
            elif isinstance(node, ast.Import):
                names = [a.name for a in node.names]
            if any(
                name == "tb.strategy.rl" or name.startswith("tb.strategy.rl.") for name in names
            ):
                importers.append(str(source.relative_to(root)))
    assert importers == []


def test_the_decision_records_arithmetic_is_the_gates() -> None:
    """0003's first reason is a table; it must stay the gate's own numbers."""
    record = (
        Path(__file__).resolve().parents[1] / "docs" / "decisions" / "0003-rl-deferred.md"
    ).read_text(encoding="utf-8")
    floor = float(LIMITS.promotion.min_oos_deflated_sharpe)
    for trials in (10, 1_000, 5_000, 50_000):
        haircut = expected_max_sharpe(n_trials=trials, sharpe_dispersion=1.0)
        assert f"| {trials:,} | {haircut:.2f} | {haircut + floor:.2f} |" in record, trials
