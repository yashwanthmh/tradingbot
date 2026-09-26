"""Whether a policy fits: the check it must pass before anything could trade it.

M9 builds no policy. It fixes the contract one must meet, as a function that
runs a policy through the real pipeline and the real adapter, over windows from
the forward-only reader, and names every way the policy falls short. When RL
is revisited, "does this policy fit the system" is then a check's output rather
than a reviewer's impression, and it is the same check the reference policies
are held to in the tests.

What each rule protects:

* **identity** — an id the registry could hold, and a version from 1. The
  decision record, the ownership join and replay all key on the pair.
* **features** — every observed feature comes from the library, under a
  unique name. A hand-written compute function would be code the one pipeline
  runs on the policy's behalf, outside the table that makes features safe.
* **declared_edge** — positive, and within `costs.max_expected_edge_bps`, the
  bound that stops a strategy dividing its way through the cost gate.
* **min_hold** — a whole number of minutes, not negative.
* **action_space** — every answer is a `Target`.
* **raised** — no exception on a valid observation. A policy that raises takes
  the cycle down with it, and one that raises when handed a read-only mapping
  was trying to write to what it was shown.
* **deterministic** — the same observation gets the same answer, whenever it
  is asked and whatever was asked before. A policy that changes its mind about
  an identical input is learning in the loop or drawing unseeded randomness,
  and either way its decisions cannot be replayed.
* **time_budget** — every answer within the budget. The loop asks each funded
  strategy about each instrument every cycle, against a rate-limited broker.
* **vacuous** — at least one window in which every observed feature had a
  value. A check in which the policy was never asked proves nothing, and
  saying so is what stops it passing as though it had.

Structure is checked before behaviour: a policy whose identity, features, edge
or minimum hold is wrong is not run, because every decision it made would
report the same fault again as something else.
"""

from __future__ import annotations

import re
import time
from collections.abc import Sequence
from dataclasses import dataclass, field
from datetime import timedelta
from decimal import Decimal

from tb.data.asof import BarWindow
from tb.features.pipeline import FEATURE_LIBRARY, FeatureError, FeaturePipeline, FeatureSpec
from tb.strategy.base import Action, PositionState
from tb.strategy.rl.policy import Observation, Policy, PolicyStrategy, Target, policy_pipeline

# Fifty milliseconds an answer: a universe of 25 instruments costs a policy
# about a second a cycle, against cycles minutes apart. A DSL spec answers in
# microseconds, so this is generous and still rules out a policy that searches
# at decision time.
DEFAULT_TIME_BUDGET = timedelta(milliseconds=50)
POLICY_ID = re.compile(r"[a-z][a-z0-9_]{2,63}")
# Each rule reports its first few cases and counts the rest: one fault seen in
# a hundred windows is one fault, not a hundred lines.
_SHOWN_PER_RULE = 3
# What the probe holds while the policy has not answered: distinct from every
# answer, `None` included, so a policy that raised is never read as one that
# answered with the wrong type.
_UNANSWERED = object()


@dataclass(frozen=True, slots=True)
class Violation:
    rule: str
    detail: str


@dataclass(frozen=True, slots=True)
class ConformanceReport:
    """What the check found. `asked` counts the cases the policy was consulted on."""

    policy: str
    cases: int
    asked: int
    slowest: timedelta | None
    violations: tuple[Violation, ...]

    @property
    def conforms(self) -> bool:
        return not self.violations

    @property
    def rules_broken(self) -> tuple[str, ...]:
        return tuple(sorted({v.rule for v in self.violations}))

    def summary(self) -> str:
        if self.conforms:
            return f"{self.policy} conforms: asked on {self.asked} of {self.cases} cases"
        lines = [f"{self.policy} does not conform:"]
        lines.extend(f"  [{v.rule}] {v.detail}" for v in self.violations)
        return "\n".join(lines)


@dataclass
class _Findings:
    violations: list[Violation] = field(default_factory=list)
    counts: dict[str, int] = field(default_factory=dict)

    def add(self, rule: str, detail: str) -> None:
        seen = self.counts.get(rule, 0)
        self.counts[rule] = seen + 1
        if seen < _SHOWN_PER_RULE:
            self.violations.append(Violation(rule, detail))

    def closed(self) -> tuple[Violation, ...]:
        extra = [
            Violation(rule, f"and {count - _SHOWN_PER_RULE} more like it")
            for rule, count in sorted(self.counts.items())
            if count > _SHOWN_PER_RULE
        ]
        return (*self.violations, *extra)


@dataclass
class _Probe:
    """The policy under test, timed, with its raw answers kept.

    Delegates everything, so the adapter treats it exactly as the policy; it
    only watches, so a non-`Target` answer still reaches the adapter's own
    refusal, which is part of what is being checked.
    """

    inner: Policy
    asked: int = 0
    slowest: float = 0.0
    answer: object = _UNANSWERED

    @property
    def policy_id(self) -> str:
        return self.inner.policy_id

    @property
    def version(self) -> int:
        return self.inner.version

    @property
    def features(self) -> tuple[FeatureSpec, ...]:
        return self.inner.features

    @property
    def expected_edge_bps(self) -> Decimal:
        return self.inner.expected_edge_bps

    @property
    def min_holding_minutes(self) -> int:
        return self.inner.min_holding_minutes

    def act(self, observation: Observation) -> Target:
        self.asked += 1
        self.answer = _UNANSWERED
        started = time.perf_counter()
        try:
            self.answer = self.inner.act(observation)
        finally:
            self.slowest = max(self.slowest, time.perf_counter() - started)
        return self.answer


@dataclass(frozen=True, slots=True)
class _Case:
    label: str
    window: BarWindow
    position: PositionState


def _structure(policy: Policy, max_edge_bps: Decimal, found: _Findings) -> FeaturePipeline | None:
    policy_id, version = policy.policy_id, policy.version
    if not isinstance(policy_id, str) or not POLICY_ID.fullmatch(policy_id):
        found.add(
            "identity",
            f"policy id {policy_id!r} is not 3-64 lowercase letters, digits or underscores "
            "starting with a letter",
        )
    if not isinstance(version, int) or isinstance(version, bool) or version < 1:
        found.add("identity", f"version {version!r} is not a whole number from 1")

    specs = tuple(policy.features)
    if not specs:
        found.add("features", "observes nothing, so it decides on nothing a snapshot records")
    library = tuple(FEATURE_LIBRARY.values())
    names: set[str] = set()
    for spec in specs:
        if not any(spec.compute is known for known in library):
            found.add(
                "features",
                f"{spec.name!r} is computed by a function outside the feature library; "
                "build it with make_spec so the pipeline runs only code from the table",
            )
        if spec.name in names:
            found.add("features", f"{spec.name!r} is observed twice")
        names.add(spec.name)
    pipeline: FeaturePipeline | None = None
    try:
        pipeline = policy_pipeline(policy)
    except FeatureError as exc:
        found.add("features", str(exc))

    edge = policy.expected_edge_bps
    if not isinstance(edge, Decimal) or not edge.is_finite():
        found.add("declared_edge", f"{edge!r} is not a finite Decimal")
    elif edge <= 0:
        found.add(
            "declared_edge",
            f"{edge}bps: an entry must declare a positive edge, or the cost gate has "
            "nothing to divide by",
        )
    elif edge > max_edge_bps:
        found.add(
            "declared_edge",
            f"{edge}bps is above costs.max_expected_edge_bps ({max_edge_bps}); the bound "
            "exists so a strategy cannot divide its way through the cost gate",
        )

    minimum = policy.min_holding_minutes
    if not isinstance(minimum, int) or isinstance(minimum, bool) or minimum < 0:
        found.add("min_hold", f"{minimum!r} is not a whole number of minutes from 0")
    return pipeline


def _cases(
    pipeline: FeaturePipeline, windows: Sequence[BarWindow], uid: str, minimum: int
) -> list[_Case]:
    cases: list[_Case] = []
    for window in windows:
        entered = window.as_of - timedelta(minutes=minimum + 1)
        when = window.as_of.isoformat()
        cases.append(_Case(f"{when}, flat", window, PositionState(instrument_uid=uid)))
        cases.append(
            _Case(
                f"{when}, holding",
                window,
                PositionState(instrument_uid=uid, quantity=Decimal(1), entry_at=entered),
            )
        )
    return cases


def check_policy(
    policy: Policy,
    *,
    windows: Sequence[BarWindow],
    instrument_uid: str,
    max_edge_bps: Decimal | float,
    time_budget: timedelta = DEFAULT_TIME_BUDGET,
) -> ConformanceReport:
    """Run `policy` through the pipeline and the adapter, and report every fault.

    Each window is asked about twice — flat, and holding past the policy's
    own minimum — then the whole set is asked again in reverse order, so an
    answer that depends on what was asked before shows up as two answers to
    one question.
    """
    found = _Findings()
    label = f"{policy.policy_id}@v{policy.version}"
    bound = max_edge_bps if isinstance(max_edge_bps, Decimal) else Decimal(str(max_edge_bps))
    pipeline = _structure(policy, bound, found)
    if found.violations or pipeline is None:
        return ConformanceReport(label, 0, 0, None, found.closed())

    probe = _Probe(policy)
    strategy = PolicyStrategy(policy=probe)
    cases = _cases(pipeline, windows, instrument_uid, policy.min_holding_minutes)
    snapshots = {id(window): pipeline.compute(window, instrument_uid) for window in windows}

    def ask(case: _Case) -> Action | None:
        before = probe.asked
        try:
            decision = strategy.decide(
                snapshot=snapshots[id(case.window)], window=case.window, position=case.position
            )
        except Exception as exc:
            # Every failure is a finding. An answer that arrived but was not a
            # Target is the adapter refusing it; anything else is the policy's.
            answered = probe.asked > before and probe.answer is not _UNANSWERED
            if answered and not isinstance(probe.answer, Target):
                found.add("action_space", f"at {case.label}: answered {probe.answer!r}")
            else:
                found.add("raised", f"at {case.label}: {type(exc).__name__}: {exc}")
            return None
        return decision.action

    first = [ask(case) for case in cases]
    asked = probe.asked
    second = [ask(case) for case in reversed(cases)][::-1]
    for case, one, two in zip(cases, first, second, strict=True):
        if one is not None and two is not None and one is not two:
            found.add(
                "deterministic",
                f"at {case.label}: {one.value} when first asked, {two.value} when asked again",
            )
    slowest = timedelta(seconds=probe.slowest) if probe.asked else None
    if slowest is not None and slowest > time_budget:
        found.add(
            "time_budget",
            f"slowest answer {slowest.total_seconds() * 1000:.1f}ms against a budget of "
            f"{time_budget.total_seconds() * 1000:.0f}ms",
        )
    if asked == 0:
        found.add(
            "vacuous",
            f"not asked in any of {len(cases)} cases: no window had a value for every "
            "observed feature, so nothing about the policy was checked",
        )
    return ConformanceReport(label, len(cases), asked, slowest, found.closed())
