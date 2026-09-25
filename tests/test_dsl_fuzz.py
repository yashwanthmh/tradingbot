"""Fuzzing the spec path: nothing executes, nothing hangs, nothing escapes.

M6 is where the DSL's input stops being hand-written. A searcher proposes specs,
mutation rewrites them, and an optional LLM can emit them — so the parser and the
interpreter now face a generator, and every property the M3 tests asserted on a
handful of hostile payloads has to hold for *anything*.

Three properties, each asserted as strongly as the runtime allows:

**Nothing executes.** Not inferred from the source (the M3 AST walk does that)
but observed at runtime: a `sys.addaudithook` hook records every `exec`,
`compile`, process spawn, `os.system`, `ctypes` load and unpickle while a fuzzed
payload is parsed and evaluated, and the test fails if any fired. `eval` and
`exec` both raise the `exec` audit event, so a code path that reached either —
however indirectly — is caught here.

**Nothing escapes.** `StrategySpec.parse` either returns a valid spec or raises
`SpecError`. Any other exception is a crash a searcher's loop would not catch,
and a crash on generated input is a denial of service against the research
process.

**Nothing hangs.** Hypothesis runs each example under a deadline, and evaluation
runs under the interpreter's own step and wall-clock budgets.

And one regression this suite found and the schema now refuses: a constant of
`1E+999999999` is finite, so the finiteness check admitted it, and the first
mutation that scaled it raised `decimal.Overflow` out of the proposer — taking
the whole search down with it.
"""

from __future__ import annotations

import random
import sys
import threading
from collections.abc import Iterator
from contextlib import contextmanager
from datetime import UTC, datetime
from decimal import Decimal
from typing import Any

import pytest
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

from tb.config.loader import load_hard_limits
from tb.data.asof import UNKNOWN
from tb.features.pipeline import FEATURE_LIBRARY, FeatureSnapshot
from tb.research.mutate import MUTATIONS, ProposalBounds, mutate, seed_spec
from tb.strategy.dsl.interpreter import EvaluationError, evaluate
from tb.strategy.dsl.schema import (
    MAX_CONSTANT_MAGNITUDE,
    MAX_DECLARED_EDGE_BPS,
    MAX_DEPTH,
    SpecError,
    StrategySpec,
)

LIMITS = load_hard_limits(None).limits
BOUNDS = ProposalBounds.from_limits(LIMITS)
AS_OF = datetime(2026, 6, 1, tzinfo=UTC)

# --------------------------------------------------------------------------
# The audit hook: runtime proof that nothing executed
# --------------------------------------------------------------------------
#
# Audit hooks cannot be removed once installed, so there is exactly one, and it
# only records while armed. Scoped to the current thread as well, so a pytest
# plugin doing something legitimate on another thread cannot fail a test here.

_FORBIDDEN_EVENTS = frozenset(
    {
        "exec",
        "compile",
        "os.system",
        "os.exec",
        "os.posix_spawn",
        "os.spawn",
        "os.fork",
        "subprocess.Popen",
        "ctypes.dlopen",
        "ctypes.dlsym",
        "pickle.find_class",
        "marshal.loads",
        "code.__new__",
        "function.__new__",
    }
)
_armed: dict[str, Any] = {"thread": None, "events": []}


def _audit(event: str, args: tuple[Any, ...]) -> None:
    # `!=`, not `is not`: thread ids are large ints, which are not interned, so
    # an identity test is false for two equal ids and the hook would silently
    # record nothing. The control test below exists because of exactly that.
    if _armed["thread"] != threading.get_ident():
        return
    if event in _FORBIDDEN_EVENTS:
        _armed["events"].append(event)


sys.addaudithook(_audit)


@contextmanager
def nothing_executes() -> Iterator[None]:
    _armed["events"] = []
    _armed["thread"] = threading.get_ident()
    try:
        yield
    finally:
        _armed["thread"] = None
    assert not _armed["events"], f"code execution observed: {sorted(set(_armed['events']))}"


def test_the_audit_hook_would_catch_an_eval() -> None:
    """The control. A hook that recorded nothing would pass every test below."""
    _armed["events"] = []
    _armed["thread"] = threading.get_ident()
    try:
        eval("1 + 1")
    finally:
        _armed["thread"] = None
    assert "exec" in _armed["events"] or "compile" in _armed["events"]


# --------------------------------------------------------------------------
# Payloads
# --------------------------------------------------------------------------

_HOSTILE_STRINGS = [
    "__import__('os').system('true')",
    "eval('1')",
    "exec('x=1')",
    "lambda: 0",
    "().__class__.__bases__[0].__subclasses__()",
    "{{7*7}}",
    "${jndi:ldap://x}",
    "sma\u0430",  # Cyrillic 'a': a homoglyph of a library name
    "sma\x00",
    "sma_10",
    "__builtins__",
    "open('/etc/passwd')",
    "1e999999999",
    "NaN",
    "Infinity",
    "-0",
    "9" * 400,
    "‮⁦gt",
]

_KINDS = ["compare", "not", "all", "any", "feature", "const", "eval", "exec", "call", "attr"]
_OPS = ["gt", "gte", "lt", "lte", "eq", "ne", "exec", "__gt__", "is", "in"]
_NAMES = [*sorted(FEATURE_LIBRARY), "__import__", "eval", "os", "sys", "open"]

_scalars = st.one_of(
    st.none(),
    st.booleans(),
    st.integers(min_value=-(10**30), max_value=10**30),
    st.floats(allow_nan=True, allow_infinity=True),
    st.text(max_size=40),
    st.sampled_from(_HOSTILE_STRINGS),
)


def _node(children: st.SearchStrategy[Any]) -> st.SearchStrategy[Any]:
    """A dictionary shaped *like* a grammar node, which is the dangerous kind.

    Random JSON is refused at the first key. Payloads that use the grammar's own
    vocabulary with the wrong types, hostile strings in the slots, and nesting
    in the wrong places get much further into validation, which is where a
    crash would hide.
    """
    return st.fixed_dictionaries(
        {"kind": st.sampled_from(_KINDS)},
        optional={
            "op": st.one_of(st.sampled_from(_OPS), _scalars),
            "left": children,
            "right": children,
            "operand": children,
            "operands": st.lists(children, max_size=4),
            "name": st.one_of(st.sampled_from(_NAMES), _scalars),
            "lookback": st.one_of(st.integers(min_value=-5, max_value=10**6), _scalars),
            "value": st.one_of(
                _scalars,
                st.decimals(allow_nan=True, allow_infinity=True),
                st.sampled_from(_HOSTILE_STRINGS),
            ),
            "__class__": _scalars,
        },
    )


_payloads = st.recursive(_scalars | st.lists(_scalars, max_size=3), _node, max_leaves=25)


def _spec_payload(entry: Any, exit_: Any, edge: Any = "450") -> dict[str, Any]:
    return {
        "name": "fuzz",
        "entry": entry,
        "exit": exit_,
        "expected_edge_bps": edge,
        "min_holding_minutes": 1440,
    }


# --------------------------------------------------------------------------
# The properties
# --------------------------------------------------------------------------


@settings(
    max_examples=400,
    deadline=2000,
    suppress_health_check=[HealthCheck.too_slow, HealthCheck.data_too_large],
)
@given(entry=_payloads, exit_=_payloads, edge=_scalars)
def test_any_payload_is_a_spec_or_a_spec_error_and_nothing_runs(
    entry: Any, exit_: Any, edge: Any
) -> None:
    """**The parser's whole contract, under a generator.**

    A valid spec or `SpecError`, never anything else, and never a side effect.
    A `TypeError` or `RecursionError` escaping here would be a crash in the
    searcher's inner loop on its own output.
    """
    with nothing_executes():
        try:
            spec = StrategySpec.parse(_spec_payload(entry, exit_, edge))
        except SpecError:
            return
    # Anything accepted must be genuinely valid — a spec the interpreter can
    # walk — not merely something the parser let through.
    assert spec.n_nodes >= 2
    assert spec.required_features is not None


@settings(max_examples=150, deadline=3000, suppress_health_check=[HealthCheck.too_slow])
@given(seed=st.integers(min_value=0, max_value=2**32 - 1))
def test_generated_and_mutated_specs_evaluate_within_budget(seed: int) -> None:
    """A searcher's own output, fed back through mutation and the interpreter.

    Twenty mutations deep from a random draw, each evaluated against a snapshot
    holding every feature it reads. The answer must be a three-valued result —
    True, False or UNKNOWN — reached inside the budgets, with no code executed
    and no exception but the interpreter's own refusal.
    """
    rng = random.Random(seed)
    spec = seed_spec(rng, bounds=BOUNDS)
    with nothing_executes():
        for step in range(20):
            proposal = mutate(spec, rng=rng, bounds=BOUNDS, index=step)
            if proposal is None:
                continue
            spec = proposal.spec
            snapshot = _snapshot_for(spec, rng)
            for tree in (spec.entry, spec.exit):
                try:
                    outcome = evaluate(tree, snapshot)
                except EvaluationError:  # pragma: no cover - budgets are generous
                    continue
                assert (
                    outcome.result is True or outcome.result is False or (outcome.result is UNKNOWN)
                )


def _snapshot_for(spec: StrategySpec, rng: random.Random) -> FeatureSnapshot:
    """Every feature the spec reads, some of them UNKNOWN."""
    values: dict[str, Any] = {}
    for key in spec.required_features:
        values[key] = (
            UNKNOWN if rng.random() < 0.15 else Decimal(str(round(rng.uniform(-50, 200), 4)))
        )
    return FeatureSnapshot(
        as_of=AS_OF,
        instrument_uid="isin:FUZZ",
        values=values,
        snapshot_hash="fuzz",
        n_bars_seen=300,
    )


# --------------------------------------------------------------------------
# The named cases from the plan
# --------------------------------------------------------------------------


def _deep(depth: int) -> dict[str, Any]:
    """Built iteratively, so the fixture cannot itself hit the recursion limit."""
    node: dict[str, Any] = {
        "kind": "compare",
        "op": "gt",
        "left": {"kind": "feature", "name": "sma", "lookback": 5},
        "right": {"kind": "const", "value": "1"},
    }
    for _ in range(depth):
        node = {"kind": "not", "operand": node}
    return node


@pytest.mark.parametrize("depth", [MAX_DEPTH, 200, 5_000, 50_000])
def test_deep_nesting_is_refused_quickly_and_cleanly(depth: int) -> None:
    """Unbounded nesting is the obvious attack on a recursive walker.

    Refused as a `SpecError` — never a `RecursionError` escaping, and never a
    segfault in the validator — and refused fast, because a parser that took
    seconds per payload is a denial of service at a thousand proposals a cycle.
    """
    import time

    started = time.monotonic()
    with nothing_executes(), pytest.raises(SpecError):
        StrategySpec.parse(_spec_payload(_deep(depth), _deep(1)))
    assert time.monotonic() - started < 2.0


@pytest.mark.parametrize(
    "value",
    [
        "1E+999999999",
        "-1E+999999999",
        "1E+28",
        "9" * 5_000,
        str(MAX_CONSTANT_MAGNITUDE + 1),
        "1E-999999999",
        "0.000000000001",
    ],
)
def test_a_constant_no_feature_can_cross_is_refused(value: str) -> None:
    """**The regression this suite found.**

    `1E+999999999` is finite, so the finiteness check alone admitted it, and the
    first mutation that scaled it raised `decimal.Overflow` out of the proposer.
    Every feature in the library is a price, a percentage or a z-score, so a
    threshold beyond ±10^9 — or finer than the ten places a feature carries —
    cannot be crossed, and is refused as malformed.
    """
    constant = {
        "kind": "compare",
        "op": "gt",
        "left": {"kind": "feature", "name": "zscore", "lookback": 20},
        "right": {"kind": "const", "value": value},
    }
    with nothing_executes(), pytest.raises(SpecError):
        StrategySpec.parse(_spec_payload(constant, _deep(0)))


@pytest.mark.parametrize("value", ["1E+9", "-1E+9", "0.0000000001", "123456.789", "0"])
def test_a_constant_at_the_bounds_is_still_accepted(value: str) -> None:
    """The bound refuses nonsense, not the edge of the legitimate range."""
    constant = {
        "kind": "compare",
        "op": "gt",
        "left": {"kind": "feature", "name": "sma", "lookback": 20},
        "right": {"kind": "const", "value": value},
    }
    StrategySpec.parse(_spec_payload(constant, _deep(0)))


def test_perturbing_an_unrepresentable_constant_does_not_crash_the_proposer() -> None:
    """Defence in depth behind the schema.

    A parent read from a registry row written before the constant bound existed
    could still carry one. The operator treats an unscalable constant as "does
    not apply" and the proposer moves on, rather than raising out of the search.
    """
    legacy = StrategySpec.model_construct(
        **StrategySpec.parse(
            _spec_payload(
                {
                    "kind": "compare",
                    "op": "gt",
                    "left": {"kind": "feature", "name": "zscore", "lookback": 20},
                    "right": {"kind": "const", "value": "1.5"},
                },
                _deep(0),
            )
        ).__dict__
    )
    payload = legacy.model_dump(mode="json")
    payload["entry"]["right"]["value"] = "1E+999999999"

    tree = payload["entry"]
    rng = random.Random(0)
    for _ in range(50):
        # Called directly: the operator is what must not raise.
        MUTATIONS["perturb_constant"](tree, rng, BOUNDS)


@pytest.mark.parametrize(
    "edge", ["1E+999999999", "1E+28", "9" * 5_000, str(MAX_DECLARED_EDGE_BPS + 1)]
)
def test_a_declared_edge_that_is_not_a_claim_is_refused(edge: str) -> None:
    """**The same regression, one field over.**

    Found while writing the LLM proposer's hostile-reply tests: the declared
    edge was finite-checked and nothing more, so `1E+999999999` parsed \u2014 and the
    edge mutation's `quantize` raised `decimal.Overflow` on it. The tradable band is
    the validator's (`costs.max_expected_edge_bps`); this bound is where a
    number stops being a claim at all.
    """
    payload = _spec_payload(_deep(0), _deep(0))
    payload["expected_edge_bps"] = edge
    with nothing_executes(), pytest.raises(SpecError):
        StrategySpec.parse(payload)


def test_a_declared_edge_at_the_bound_is_still_a_spec() -> None:
    payload = _spec_payload(_deep(0), _deep(0))
    payload["expected_edge_bps"] = str(MAX_DECLARED_EDGE_BPS)
    StrategySpec.parse(payload)


def test_mutating_an_unrepresentable_edge_does_not_crash_the_proposer() -> None:
    """Defence in depth, as for constants: a seed read from a registry row that
    predates the bound is skipped by the edge mutation rather than raised out of
    the search."""
    legacy = StrategySpec.model_construct(
        **{
            **StrategySpec.parse(_spec_payload(_deep(0), _deep(0))).__dict__,
            "expected_edge_bps": Decimal("1E+999999999"),
        }
    )
    rng = random.Random(0)
    for index in range(50):
        proposal = mutate(legacy, rng=rng, bounds=BOUNDS, index=index)
        assert proposal is None or proposal.spec.expected_edge_bps <= MAX_DECLARED_EDGE_BPS


@pytest.mark.parametrize("name", ["sma\u0430", "SMA", "sma ", "sma\x00", "__import__", "open"])
def test_a_lookalike_feature_name_is_refused(name: str) -> None:
    """Features come from a fixed table; a homoglyph is not in it."""
    lookalike = {
        "kind": "compare",
        "op": "gt",
        "left": {"kind": "feature", "name": name, "lookback": 20},
        "right": {"kind": "const", "value": "1"},
    }
    with nothing_executes(), pytest.raises(SpecError):
        StrategySpec.parse(_spec_payload(lookalike, _deep(0)))


def test_a_python_expression_in_any_slot_is_refused() -> None:
    """Every string-typed slot in the grammar, filled with code."""
    expression = "__import__('os').system('touch /tmp/pwned')"
    slots = [
        {"kind": expression},
        {"kind": "compare", "op": expression, "left": _deep(0)["left"], "right": _deep(0)["right"]},
        {"kind": "compare", "op": "gt", "left": expression, "right": _deep(0)["right"]},
        {
            "kind": "compare",
            "op": "gt",
            "left": {"kind": "feature", "name": expression, "lookback": 5},
            "right": _deep(0)["right"],
        },
        {
            "kind": "compare",
            "op": "gt",
            "left": _deep(0)["left"],
            "right": {"kind": "const", "value": expression},
        },
    ]
    for slot in slots:
        with nothing_executes(), pytest.raises(SpecError):
            StrategySpec.parse(_spec_payload(slot, _deep(0)))
