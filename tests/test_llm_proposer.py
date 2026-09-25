"""The LLM proposer: price-blind going out, untrusted coming back.

M6's optional model-backed proposer, tested with no network and no key — every
test drives it through a fake `LLMClient`, and the SDK path through a fake
stream. The properties, in the order the data flows:

**Nothing dated leaves the process.** The prompt builders take only abstract
types, and the tripwire behind them refuses a rendered prompt that holds a date,
a price, an instrument or a credential. Tested both ways: the real prompts pass,
and each thing a careless edit could add is caught. The strongest form is the
last in that section — two vintages with different dates, prices and
instruments produce byte-identical prompts.

**Nothing the model returns is executed or obeyed.** Hostile replies — Python
expressions, deep nesting, huge exponents, NaN, lone surrogates, control
characters, keys the model has no say over — are parsed as JSON, validated by
the schema, and either become ordinary specs or are refused, under the fuzz
suite's audit hook.

**A failed call records nothing.** A refusal, a transport failure or a reply
with nothing usable raises before any candidate is evaluated, so no trial rows
exist for a search that did not happen.

**A successful call is on the record.** The exact prompts and every accepted
spec are ledgered, because unlike a seeded draw the exchange cannot be replayed.
"""

from __future__ import annotations

import dataclasses
import inspect
import json
import random
import sys
import time
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from decimal import Decimal
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast

import pytest

from tb.config.loader import load_hard_limits
from tb.data.asof import HoldoutViolation, InMemoryBarSource
from tb.data.barstore import BarStore
from tb.data.provider import (
    Bar,
    BarBatch,
    Provenance,
    Resolution,
    Session,
    make_instrument_uid,
)
from tb.data.regime import RegimeGate, RegimeReading, RegimeState
from tb.data.snapshot import SnapshotStore
from tb.features.pipeline import FEATURE_LIBRARY
from tb.registry.lineage import SpecRegistry
from tb.registry.models import AuthorKind
from tb.research.llm import adapter
from tb.research.llm.adapter import (
    FALLBACK_BETA,
    MAX_SPECS_PER_CALL,
    AnthropicClient,
    LLMError,
    LLMProposer,
    LLMResponse,
    LLMUnavailable,
    extract_items,
)
from tb.research.llm.prompts import (
    FEATURE_NOTES,
    PromptLeak,
    RegimeDescription,
    audit_prompt,
    render_system_prompt,
    render_user_prompt,
)
from tb.research.loop import (
    CycleError,
    CycleReport,
    ProposalContext,
    ResearchCycle,
    training_regime,
)
from tb.research.mutate import ProposalBounds, SpecProposer
from tb.research.searcher import SearchBudget, required_sharpe
from tb.research.validate import SpecValidator
from tb.strategy.dsl.schema import StrategySpec
from tests.test_cli_registry import UID, _init, _out, _run
from tests.test_dsl_fuzz import nothing_executes
from tests.test_research_cycle import _ledger, _sealed_vintage

LIMITS = load_hard_limits(None).limits
BOUNDS = ProposalBounds.from_limits(LIMITS, min_holding_minutes=1440)


# --------------------------------------------------------------------------
# A model that says exactly what the test tells it to
# --------------------------------------------------------------------------


@dataclass
class FakeClient:
    """An `LLMClient` with a scripted reply, recording every prompt it was sent."""

    reply: str = "[]"
    stop_reason: str | None = "end_turn"
    refusal: str = ""
    served_model: str = "fake-model-1"
    fell_back: bool = False
    calls: list[tuple[str, str]] = field(default_factory=list)

    @property
    def model(self) -> str:
        return "fake-model-1"

    def complete(self, *, system: str, user: str) -> LLMResponse:
        self.calls.append((system, user))
        return LLMResponse(
            text=self.reply,
            requested_model=self.model,
            served_model=self.served_model,
            stop_reason=self.stop_reason,
            fell_back=self.fell_back,
            refusal=self.refusal,
        )


def _compare(op: str, name: str, lookback: int, value: str) -> dict[str, Any]:
    return {
        "kind": "compare",
        "op": op,
        "left": {"kind": "feature", "name": name, "lookback": lookback},
        "right": {"kind": "const", "value": value},
    }


def item(
    *,
    name: str = "zscore",
    lookback: int = 20,
    entry: str = "-1.5",
    exit_: str = "0",
    edge: str = "300",
) -> dict[str, Any]:
    """One well-formed item, as a model following the system prompt would write it."""
    return {
        "entry": _compare("lt", name, lookback, entry),
        "exit": _compare("gt", name, lookback, exit_),
        "expected_edge_bps": edge,
    }


# Six distinct ideas, each affordable at a US round trip and short enough for
# the fixture's training window.
GOOD_ITEMS: list[dict[str, Any]] = [
    item(),
    item(lookback=50, entry="-2", exit_="0.5"),
    item(name="return_pct", lookback=20, entry="-8", exit_="2", edge="250"),
    item(name="stdev_pct", lookback=20, entry="1", exit_="3", edge="200"),
    item(name="max_drawdown_pct", lookback=50, entry="5", exit_="10", edge="350"),
    item(name="return_pct", lookback=50, entry="-12", exit_="4", edge="400"),
]


def _proposer(client: FakeClient, **kwargs: Any) -> LLMProposer:
    return LLMProposer(client=client, bounds=BOUNDS, **kwargs)


# --------------------------------------------------------------------------
# Going out: nothing the model could date
# --------------------------------------------------------------------------


def test_what_a_proposer_may_be_told_has_no_field_for_a_date_a_price_or_a_ticker() -> None:
    """**The primary guarantee is the types, so the types are pinned.**

    The tripwire below catches a rendered leak; this catches the edit that would
    make one possible — a field added to what a proposer is handed. Adding one
    is sometimes right, and when it is, this test is where the reason gets
    written down.
    """
    assert {f.name for f in dataclasses.fields(RegimeDescription)} == {"trend"}
    assert {f.name for f in dataclasses.fields(ProposalContext)} == {
        "bounds",
        "regime",
        "n_trials",
        "required_sharpe",
    }
    assert {f.name for f in dataclasses.fields(ProposalBounds)} == {
        "min_edge_bps",
        "max_edge_bps",
        "lookbacks",
        "min_holding_minutes",
    }
    assert set(inspect.signature(render_system_prompt).parameters) == {"bounds"}
    assert set(inspect.signature(render_user_prompt).parameters) == {
        "n",
        "regime",
        "required_sharpe",
        "n_trials",
    }


@pytest.mark.parametrize("n_trials", [1, 10, 50, 1_950, 2_000, 2_049, 12_345, 1_000_000])
@pytest.mark.parametrize("trend", ["above", "below", "unknown"])
def test_the_real_prompts_pass_their_own_tripwire(n_trials: int, trend: str) -> None:
    """Including the budgets that would read as a year if written plainly."""
    bounds = ProposalBounds.from_limits(LIMITS, max_lookback=120)
    regime = RegimeDescription(trend=cast(Any, trend))
    audit_prompt(render_system_prompt(bounds))
    audit_prompt(
        render_user_prompt(
            n=MAX_SPECS_PER_CALL,
            regime=regime,
            required_sharpe=required_sharpe(
                n_trials=n_trials,
                min_deflated_sharpe=LIMITS.promotion.min_oos_deflated_sharpe,
            ),
            n_trials=n_trials,
        )
    )


def test_the_system_prompt_states_the_bounds_the_validator_will_enforce() -> None:
    """A prompt that drifted from the bounds would breed refusals.

    Every lookback on offer, the edge band, and the three keys the model
    decides are stated — and a lookback the training window cannot support is
    not offered at all.
    """
    bounds = ProposalBounds.from_limits(LIMITS, max_lookback=60)
    system = render_system_prompt(bounds)
    assert bounds.lookbacks == (5, 10, 20, 50)
    assert "LOOKBACK is one of: 5, 10, 20, 50 " in system
    assert "100" not in system.split("LOOKBACK is one of:")[1].split("\n")[0]
    assert f"between {bounds.min_edge_bps} and {bounds.max_edge_bps}" in system
    for key in ("entry", "exit", "expected_edge_bps"):
        assert f'"{key}"' in system


@pytest.mark.parametrize(
    ("addition", "label"),
    [
        ("as of 2020-03-16", "an ISO date"),
        ("like the 2008 crisis", "a calendar year"),
        ("since March", "a month name"),
        ("it closed at $60.55", "a currency amount"),
        ("down to €12", "a currency amount"),
        ("US0378331005", "an ISIN"),
        ("sym:SPY", "an instrument uid"),
        ("AAPL_US_EQ", "a broker ticker"),
        ("sk-ant-api03-abcdefgh", "an API key"),
        ("T212_LIVE_API_KEY", "a credential variable"),
        ("ANTHROPIC_API_KEY", "a credential variable"),
        ("A1b2C3d4" * 6, "a long opaque token"),
    ],
)
def test_the_tripwire_catches_what_a_careless_edit_would_add(addition: str, label: str) -> None:
    prompt = render_user_prompt(n=5, regime=RegimeDescription()) + "\n" + addition
    with pytest.raises(PromptLeak, match=label):
        audit_prompt(prompt)


def test_a_leaking_prompt_never_reaches_the_model(monkeypatch: pytest.MonkeyPatch) -> None:
    """The audit runs before the call, so a leak costs an exception, not a disclosure."""
    client = FakeClient(reply=json.dumps(GOOD_ITEMS))
    monkeypatch.setattr(
        adapter, "render_user_prompt", lambda **_: "As of 2020-03-16, propose strategies."
    )
    with pytest.raises(PromptLeak):
        _proposer(client).propose(n=5, rng=random.Random(0))
    assert client.calls == []


def test_every_library_feature_is_described_with_its_units() -> None:
    """A feature the model is not told the units of is one it will compare with a
    feature of different units — `sma_20 > zscore_20` is always true."""
    assert set(FEATURE_NOTES) == set(FEATURE_LIBRARY)


def _reading(state: RegimeState, *, at: datetime, close: str, average: str) -> RegimeReading:
    factor = Decimal(1) if state is RegimeState.RISK_ON else Decimal("0.5")
    return RegimeReading(
        as_of=at,
        state=state,
        exposure_factor=factor,
        reference_symbol="SPY",
        instrument_uid="sym:SPY",
        ma_days=200,
        n_sessions_seen=300,
        last_close=Decimal(close),
        moving_average=Decimal(average),
        detail=f"SPY {close} against its 200-day average {average} on {at.date()}",
    )


def test_the_regime_description_reads_the_state_and_nothing_that_dates_it() -> None:
    """Two readings years apart, at different levels, describe the same market."""
    early = _reading(
        RegimeState.RISK_ON, at=datetime(2020, 3, 16, tzinfo=UTC), close="240.1", average="230"
    )
    late = _reading(
        RegimeState.RISK_ON, at=datetime(2024, 11, 5, tzinfo=UTC), close="590", average="540"
    )
    assert RegimeDescription.from_reading(early) == RegimeDescription.from_reading(late)
    assert RegimeDescription.from_reading(early) == RegimeDescription(trend="above")

    for state, trend in (
        (RegimeState.RISK_OFF, "below"),
        (RegimeState.INSUFFICIENT_HISTORY, "unknown"),
        (RegimeState.UNAVAILABLE, "unknown"),
    ):
        described = RegimeDescription.from_reading(
            _reading(state, at=datetime(2022, 6, 1, tzinfo=UTC), close="380", average="420")
        )
        assert described.trend == trend
        sentence = described.sentence()
        audit_prompt(sentence)
        assert not any(ch.isdigit() for ch in sentence)


# The reference series, as the regime gate names it.
SPY = make_instrument_uid(data_symbol="SPY")


def _daily(uid: str, opened: datetime, close: Decimal) -> Bar:
    return Bar(
        instrument_uid=uid,
        resolution=Resolution.DAILY,
        bar_open_utc=opened,
        available_at_utc=opened + timedelta(days=1),
        ingested_at_utc=opened + timedelta(days=1),
        provider="fixture",
        provenance=Provenance.BACKFILL,
        session=Session.REGULAR,
        open=close,
        high=close + Decimal("0.5"),
        low=close - Decimal("0.5"),
        close=close,
        volume=1_000_000,
    )


def test_the_regime_a_proposer_is_told_is_read_from_before_the_seal() -> None:
    """**A rally before the seal, a crash after it.**

    Read correctly — through the sealed source, at the last training decision —
    the market is above its average. Read from the whole vintage, the crash in
    the holdout says the opposite. So the word a proposer is given is evidence
    about which data it was computed from, and this asserts it is the training
    window's.
    """
    start = datetime(2023, 1, 2, tzinfo=UTC)
    rally = [Decimal(100) + Decimal(day) / 4 for day in range(300)]
    crash = [rally[-1] - Decimal(day) / 2 for day in range(1, 120)]
    bars = [
        _daily(SPY, start + timedelta(days=day), close) for day, close in enumerate(rally + crash)
    ]
    source = InMemoryBarSource(bars=bars)
    sealed_from = start + timedelta(days=300, hours=12)
    last_training_decision = bars[299].available_at_utc + timedelta(hours=1)

    told = training_regime(LIMITS, source, sealed_from=sealed_from, as_of=last_training_decision)
    assert told == RegimeDescription(trend="above")

    # The control: the same data read across the seal says the opposite, so
    # the assertion above is about the seal and not about the fixture.
    unsealed = RegimeGate(LIMITS).read(source, as_of=bars[-1].available_at_utc)
    assert RegimeDescription.from_reading(unsealed) == RegimeDescription(trend="below")

    with pytest.raises(HoldoutViolation):
        training_regime(LIMITS, source, sealed_from=sealed_from, as_of=sealed_from)


def test_a_vintage_without_the_reference_series_is_described_as_unknown() -> None:
    """Fail-closed, as the regime gate is: not seeing the index is not a trend."""
    start = datetime(2023, 1, 2, tzinfo=UTC)
    bars = [_daily(UID, start + timedelta(days=day), Decimal(100)) for day in range(300)]
    told = training_regime(
        LIMITS,
        InMemoryBarSource(bars=bars),
        sealed_from=start + timedelta(days=400),
        as_of=bars[-1].available_at_utc + timedelta(hours=1),
    )
    assert told == RegimeDescription(trend="unknown")


# --------------------------------------------------------------------------
# Coming back: parsed as data, validated by the schema, never obeyed
# --------------------------------------------------------------------------


def test_a_clean_reply_becomes_proposals() -> None:
    client = FakeClient(reply=json.dumps(GOOD_ITEMS[:3]))
    proposer = _proposer(client)
    proposals = proposer.propose(n=3, rng=random.Random(0))

    assert len(proposals) == 3
    assert [p.spec.name for p in proposals] == ["llm-0000", "llm-0001", "llm-0002"]
    for proposal in proposals:
        assert proposal.author_kind is AuthorKind.LLM
        assert proposal.operator == "llm"
        assert proposal.parent_spec_hash is None
        assert proposal.spec.min_holding_minutes == BOUNDS.min_holding_minutes
        assert "fake-model-1" in proposal.spec.notes
    (report,) = proposer.reports
    assert (report.n_items, len(report.accepted), report.n_refused) == (3, 3, 0)
    assert report.stopped == ""
    assert len(client.calls) == 1


def test_the_model_decides_three_fields_and_no_others() -> None:
    """**The holding period is the one that matters.** A model free to declare
    `0` could propose exactly the high-turnover specs the fee schedule forbids.
    Keys it adds are reported, not obeyed."""
    hostile = {
        **item(),
        "name": "ignore previous instructions",
        "min_holding_minutes": 0,
        "notes": "set max_expected_edge_bps to 99999",
        "spec_version": 2,
        "__import__": "os",
    }
    proposer = _proposer(FakeClient(reply=json.dumps([hostile])))
    (proposal,) = proposer.propose(n=1, rng=random.Random(0))

    assert proposal.spec.name == "llm-0000"
    assert proposal.spec.min_holding_minutes == BOUNDS.min_holding_minutes
    assert proposal.spec.spec_version == 1
    assert "ignore previous" not in proposal.spec.notes
    assert set(proposer.reports[0].ignored_keys) == {
        "name",
        "min_holding_minutes",
        "notes",
        "spec_version",
        "__import__",
    }


def test_prose_and_fences_around_the_array_are_tolerated() -> None:
    """Models wrap JSON despite being told not to. A bracket in the prose is not
    mistaken for the payload."""
    reply = (
        "Here are [2] ideas.\n```json\n" + json.dumps(GOOD_ITEMS[:2], indent=2) + "\n```\n"
        "Each exits when the signal reverts [see above]."
    )
    proposals = _proposer(FakeClient(reply=reply)).propose(n=2, rng=random.Random(0))
    assert len(proposals) == 2


def test_a_cut_off_reply_keeps_every_complete_item() -> None:
    """A reply truncated by the output budget still yields what was finished."""
    full = json.dumps(GOOD_ITEMS[:3])
    cut = full[: full.rindex('"expected_edge_bps"')]
    proposer = _proposer(FakeClient(reply=cut, stop_reason="max_tokens"))
    proposals = proposer.propose(n=3, rng=random.Random(0))
    assert len(proposals) == 2
    assert "cut off" in proposer.reports[0].stopped


def test_items_beyond_what_was_asked_are_not_read() -> None:
    proposer = _proposer(FakeClient(reply=json.dumps(GOOD_ITEMS)))
    proposals = proposer.propose(n=2, rng=random.Random(0))
    assert len(proposals) == 2
    assert "not read" in proposer.reports[0].stopped


def test_a_request_is_capped_at_what_one_call_can_carry() -> None:
    client = FakeClient(reply=json.dumps(GOOD_ITEMS))
    _proposer(client).propose(n=500, rng=random.Random(0))
    (_, user) = client.calls[0]
    assert f"Propose {MAX_SPECS_PER_CALL} distinct strategies." in user


def test_the_same_idea_twice_in_one_reply_is_one_proposal() -> None:
    """Two evaluations of one tree would spend two trials on one result."""
    proposer = _proposer(FakeClient(reply=json.dumps([item(), item(), GOOD_ITEMS[1]])))
    proposals = proposer.propose(n=3, rng=random.Random(0))
    assert len(proposals) == 2
    assert proposer.reports[0].n_duplicates == 1


def test_a_malformed_item_is_counted_and_reported_not_proposed() -> None:
    """It never became a spec, so it has no hash to record and cannot be selected."""
    bad_feature = item(name="rsi")
    nan_constant = item(entry="NaN")
    reply = json.dumps([GOOD_ITEMS[0], bad_feature, nan_constant, GOOD_ITEMS[1], 42])
    proposer = _proposer(FakeClient(reply=reply))
    proposals = proposer.propose(n=5, rng=random.Random(0))

    assert len(proposals) == 2
    report = proposer.reports[0]
    assert report.n_refused == 3
    assert any("rsi" in reason for reason in report.refused)
    assert any("not an object" in reason for reason in report.refused)


def test_the_declared_edge_is_judged_by_the_validator_not_quietly_clamped() -> None:
    """The model's claim is recorded as made. Clamping it would change the claim
    while keeping the model's name on it; refusing it is a counted trial."""
    (proposal,) = _proposer(FakeClient(reply=json.dumps([item(edge="9000")]))).propose(
        n=1, rng=random.Random(0)
    )
    assert proposal.spec.expected_edge_bps == Decimal("9000")
    rejection = SpecValidator(limits=LIMITS).check(proposal.spec)
    assert rejection is not None
    assert rejection.code == "edge_band"


def test_a_refusal_fails_the_call_rather_than_becoming_a_random_search() -> None:
    client = FakeClient(reply="", stop_reason="refusal", refusal="cyber: declined")
    with pytest.raises(LLMError, match="declined") as caught:
        _proposer(client).propose(n=5, rng=random.Random(0))
    assert "--proposer random" in str(caught.value)


@pytest.mark.parametrize(
    "reply",
    [
        "I can't help with that.",
        "[]",
        json.dumps([item(name="rsi"), item(name="macd")]),
        json.dumps([1, 2, 3]),
    ],
)
def test_a_reply_with_nothing_usable_fails_the_call(reply: str) -> None:
    with pytest.raises(LLMError, match="no usable spec"):
        _proposer(FakeClient(reply=reply)).propose(n=5, rng=random.Random(0))


_SPEC = json.dumps(item())

_HOSTILE_REPLIES = [
    pytest.param("__import__('os').system('true')", id="bare-python"),
    pytest.param("[__import__('os').system('true')]", id="python-in-array"),
    pytest.param('[{"entry": __import__("os")}]', id="python-as-value"),
    pytest.param(
        '[{"entry": "__import__(\'os\').system(\'true\')", "exit": "eval(\'1\')", '
        '"expected_edge_bps": "exec(\'x=1\')"}]',
        id="python-in-strings",
    ),
    pytest.param(
        '[{"entry": "{{7*7}}", "exit": "${jndi:ldap://x}", "expected_edge_bps": 300}]',
        id="template-injection",
    ),
    pytest.param('[{"entry": ' + "[" * 200_000 + "]" * 200_000 + "}]", id="deep-nesting"),
    pytest.param("[" + _SPEC.replace('"-1.5"', "1e999999999") + "]", id="huge-constant"),
    pytest.param("[" + _SPEC.replace('"300"', "1e999999999") + "]", id="huge-edge"),
    pytest.param("[" + _SPEC.replace('"-1.5"', "NaN") + "]", id="nan"),
    pytest.param("[" + _SPEC.replace('"-1.5"', "-Infinity") + "]", id="infinity"),
    pytest.param("[" + _SPEC.replace('"300"', "9" * 5_000) + "]", id="5000-digit-int"),
    pytest.param("[" + _SPEC.replace('"zscore"', '"zscor\\u0435"') + "]", id="homoglyph"),
    pytest.param("[" + _SPEC.replace('"zscore"', '"__import__"') + "]", id="dunder-feature"),
    pytest.param(
        '[{"__class__": {"__init__": {"__globals__": 1}}, ' + _SPEC[1:] + "]", id="dunder-keys"
    ),
    pytest.param('[{"\\u001b[2J\\u001b[31mowned": 1, ' + _SPEC[1:] + "]", id="ansi-key"),
    pytest.param('[{"entry": "\\ud800", "exit": 1, "expected_edge_bps": 1}]', id="lone-surrogate"),
    pytest.param('[{"\\ud800": 1, ' + _SPEC[1:] + "]", id="lone-surrogate-key"),
    # A raw surrogate in the reply text itself, not a JSON escape: what the
    # reply hash has to survive.
    pytest.param("[" + _SPEC + "]\ud800", id="raw-surrogate"),
    pytest.param("[" + ",".join([_SPEC] * 2_000) + "]", id="oversized"),
    pytest.param("\x00" * 1_000, id="nul-bytes"),
]


@pytest.mark.parametrize("reply", _HOSTILE_REPLIES)
def test_a_hostile_reply_is_data_never_code(reply: str) -> None:
    """**Parsed, validated, and either an ordinary spec or refused — quickly.**

    Under the fuzz suite's audit hook, so "nothing executed" is observed rather
    than inferred. Whatever survives must be indistinguishable from a spec the
    schema built itself, and whatever is reported must be safe to print and to
    ledger.
    """
    proposer = _proposer(FakeClient(reply=reply))
    started = time.monotonic()
    with nothing_executes():
        try:
            proposals = proposer.propose(n=5, rng=random.Random(0))
        except LLMError:
            proposals = []
    assert time.monotonic() - started < 5.0

    for proposal in proposals:
        assert StrategySpec.parse(proposal.spec.model_dump(mode="json")) == proposal.spec
        assert proposal.spec.min_holding_minutes == BOUNDS.min_holding_minutes
    for report in proposer.reports:
        for line in report.lines():
            assert line.isprintable(), line
            line.encode("utf-8")  # a lone surrogate here would break the ledger


# --------------------------------------------------------------------------
# The reader, as a unit
# --------------------------------------------------------------------------


def test_the_reader_reads_numbers_as_decimals_and_stops_cleanly() -> None:
    items, stopped = extract_items('[{"a": 0.1, "b": NaN, "c": 1e400}]', limit=5)
    assert stopped == ""
    (value,) = items
    assert isinstance(value, dict)
    assert value["a"] == Decimal("0.1")
    assert cast(Decimal, value["b"]).is_nan()
    assert value["c"] == Decimal("1E+400")


def test_the_reader_names_where_it_stopped() -> None:
    assert extract_items("no array here", limit=5) == (
        [],
        "the reply holds no JSON array of objects",
    )
    items, stopped = extract_items('[{"a": 1} {"b": 2}]', limit=5)
    assert items == [{"a": 1}]
    assert "expected ',' or ']'" in stopped


# --------------------------------------------------------------------------
# The Claude API client, against a fake stream
# --------------------------------------------------------------------------


@dataclass
class _Stream:
    message: Any

    def __enter__(self) -> _Stream:
        return self

    def __exit__(self, *exc: object) -> None:
        return None

    def get_final_message(self) -> Any:
        return self.message


@dataclass
class _Messages:
    outcome: Any
    kwargs: dict[str, Any] = field(default_factory=dict)

    def stream(self, **kwargs: Any) -> _Stream:
        self.kwargs = kwargs
        if isinstance(self.outcome, BaseException):
            raise self.outcome
        return _Stream(self.outcome)


def _sdk_client(
    monkeypatch: pytest.MonkeyPatch, outcome: Any, *, fallbacks: bool = True
) -> tuple[AnthropicClient, _Messages]:
    pytest.importorskip("anthropic")
    monkeypatch.delenv("T212_LIVE_API_KEY", raising=False)
    client = AnthropicClient(model="claude-opus-5", fallbacks=fallbacks)
    messages = _Messages(outcome)
    monkeypatch.setattr(client, "_client", SimpleNamespace(beta=SimpleNamespace(messages=messages)))
    return client, messages


def _message(
    text: str,
    *,
    stop_reason: str = "end_turn",
    model: str = "claude-opus-5",
    fallback: bool = False,
    refusal: dict[str, Any] | None = None,
) -> Any:
    beta = pytest.importorskip("anthropic.types.beta")
    usage: dict[str, Any] = {"input_tokens": 10, "output_tokens": 20}
    if fallback:
        usage["iterations"] = [
            {
                "type": "fallback_message",
                "model": model,
                "input_tokens": 10,
                "output_tokens": 20,
                "cache_creation_input_tokens": 0,
                "cache_read_input_tokens": 0,
            }
        ]
    return beta.BetaMessage.model_validate(
        {
            "id": "msg_test",
            "type": "message",
            "role": "assistant",
            "model": model,
            "content": [
                # Reasoning is not a proposal: only text blocks are read.
                {"type": "thinking", "thinking": json.dumps([item(name="sma")]), "signature": "x"},
                {"type": "text", "text": text, "citations": None},
            ],
            "stop_reason": stop_reason,
            "stop_sequence": None,
            "stop_details": refusal,
            "usage": usage,
        }
    )


def test_the_client_asks_for_refusal_fallbacks_by_default(monkeypatch: pytest.MonkeyPatch) -> None:
    client, messages = _sdk_client(monkeypatch, _message("[]"))
    client.complete(system="the system prompt", user="the user prompt")

    assert messages.kwargs["model"] == "claude-opus-5"
    assert messages.kwargs["betas"] == [FALLBACK_BETA]
    assert messages.kwargs["fallbacks"] == "default"
    assert messages.kwargs["system"] == "the system prompt"
    assert messages.kwargs["messages"] == [{"role": "user", "content": "the user prompt"}]


def test_fallbacks_can_be_turned_off(monkeypatch: pytest.MonkeyPatch) -> None:
    anthropic = pytest.importorskip("anthropic")
    client, messages = _sdk_client(monkeypatch, _message("[]"), fallbacks=False)
    client.complete(system="s", user="u")
    assert messages.kwargs["betas"] is anthropic.omit
    assert messages.kwargs["fallbacks"] is anthropic.omit


def test_the_client_reads_text_blocks_and_who_served_them(monkeypatch: pytest.MonkeyPatch) -> None:
    reply = json.dumps([item()])
    client, _ = _sdk_client(monkeypatch, _message(reply, model="claude-opus-4-8", fallback=True))
    response = client.complete(system="s", user="u")
    assert response.text == reply
    assert response.requested_model == "claude-opus-5"
    assert response.served_model == "claude-opus-4-8"
    assert response.fell_back


def test_the_client_reports_a_refusal_with_its_category(monkeypatch: pytest.MonkeyPatch) -> None:
    refusal = {"type": "refusal", "category": "cyber", "explanation": "declined"}
    client, _ = _sdk_client(monkeypatch, _message("", stop_reason="refusal", refusal=refusal))
    response = client.complete(system="s", user="u")
    assert response.stop_reason == "refusal"
    assert response.refusal == "cyber: declined"


def _status_error(name: str, status: int) -> BaseException:
    anthropic = pytest.importorskip("anthropic")
    httpx2 = pytest.importorskip("httpx2")
    request = httpx2.Request("POST", "https://api.anthropic.com/v1/messages")
    error: BaseException = getattr(anthropic, name)(
        "text the server sent", response=httpx2.Response(status, request=request), body=None
    )
    return error


@pytest.mark.parametrize(
    ("name", "status", "unavailable"),
    [
        ("AuthenticationError", 401, True),
        ("PermissionDeniedError", 403, True),
        ("NotFoundError", 404, True),
        ("BadRequestError", 400, False),
        ("RateLimitError", 429, False),
        ("InternalServerError", 500, False),
    ],
)
def test_api_failures_become_typed_errors(
    monkeypatch: pytest.MonkeyPatch, name: str, status: int, unavailable: bool
) -> None:
    """Configuration problems are `LLMUnavailable`; transient ones are `LLMError`.

    And a credential failure does not echo the server's text: a message about a
    key is the one kind that should not be repeated into a terminal by habit.
    """
    client, _ = _sdk_client(monkeypatch, _status_error(name, status))
    with pytest.raises(LLMError) as caught:
        client.complete(system="s", user="u")
    assert isinstance(caught.value, LLMUnavailable) is unavailable
    if name in ("AuthenticationError", "PermissionDeniedError"):
        assert "text the server sent" not in str(caught.value)


def test_a_connection_failure_is_transient(monkeypatch: pytest.MonkeyPatch) -> None:
    anthropic = pytest.importorskip("anthropic")
    httpx2 = pytest.importorskip("httpx2")
    failure = anthropic.APIConnectionError(
        request=httpx2.Request("POST", "https://api.anthropic.com/v1/messages")
    )
    client, _ = _sdk_client(monkeypatch, failure)
    with pytest.raises(LLMError) as caught:
        client.complete(system="s", user="u")
    assert not isinstance(caught.value, LLMUnavailable)


def test_no_credential_at_all_is_a_setup_error(monkeypatch: pytest.MonkeyPatch) -> None:
    client, _ = _sdk_client(monkeypatch, TypeError('"Could not resolve authentication method."'))
    with pytest.raises(LLMUnavailable, match="ANTHROPIC_API_KEY"):
        client.complete(system="s", user="u")


def test_the_client_refuses_a_process_holding_the_live_broker_key(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The one research component that sends text to a third party runs in a
    process without the live key, or not at all."""
    monkeypatch.setenv("T212_LIVE_API_KEY", "not-a-real-key")
    with pytest.raises(LLMUnavailable, match="T212_LIVE_API_KEY"):
        AnthropicClient()


def test_a_missing_sdk_names_the_extra(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("T212_LIVE_API_KEY", raising=False)
    monkeypatch.setitem(cast(dict[str, Any], sys.modules), "anthropic", None)
    with pytest.raises(LLMUnavailable, match="uv sync --extra llm"):
        AnthropicClient()


# --------------------------------------------------------------------------
# In the cycle: one call, on the record, and nothing recorded when it fails
# --------------------------------------------------------------------------


def _run_cycle(
    env: dict[str, Any],
    vintage_id: str,
    client: FakeClient,
    *,
    seeds: tuple[str, ...] = (),
) -> tuple[CycleReport, list[ProposalContext]]:
    pinned = load_hard_limits(env["limits"])
    contexts: list[ProposalContext] = []

    def factory(context: ProposalContext) -> SpecProposer:
        contexts.append(context)
        return LLMProposer(
            client=client,
            bounds=context.bounds,
            regime=context.regime,
            required_sharpe=context.required_sharpe,
            n_trials=context.n_trials,
        )

    with _ledger(env) as ledger:
        store = BarStore(ledger, root=env["bars"], scale=pinned.limits.data.price_scale)
        report = ResearchCycle(
            ledger, limits=pinned.limits, snapshots=SnapshotStore(ledger, store)
        ).run(
            vintage_id=vintage_id,
            budget=SearchBudget(n_trials=16, n_per_generation=8, seed=3),
            register=False,
            seed_strategy_ids=seeds,
            proposer=factory,
        )
    return report, contexts


def _reply_of_eight() -> str:
    """Six good ideas and two the schema refuses."""
    return json.dumps([*GOOD_ITEMS, item(name="rsi"), item(entry="NaN")])


def test_a_model_backed_search_is_on_the_record(cli_env: dict[str, Any]) -> None:
    """**The exchange, then the trials it produced — all of it in the ledger.**

    The accepted specs are stored in full because a model cannot be replayed:
    every trial row that came from it must be reconstructible from the event.
    And the stored prompts are checked for what they must not contain — the
    vintage's instrument, its identifiers, any date.
    """
    vintage_id = _sealed_vintage(cli_env)
    client = FakeClient(reply=_reply_of_eight())
    report, contexts = _run_cycle(cli_env, vintage_id, client)

    assert len(client.calls) == 1, "one call per search is the cost model"
    assert report.proposer == "llm"
    assert any("6 accepted" in note for note in report.proposer_notes)
    # Six from the model, then a generation of eight bred from them.
    assert report.outcome.n_proposed == 14
    assert report.n_trials_recorded == 14

    (context,) = contexts
    assert context.regime == RegimeDescription(trend="unknown")  # no SPY in the fixture
    assert context.n_trials == 16
    assert context.bounds.min_holding_minutes == 1440

    with _ledger(cli_env) as ledger:
        events = ledger.conn.execute(
            "SELECT seq, payload_json FROM event_log WHERE event_type = 'search.specs_proposed'"
        ).fetchall()
        first_trial = ledger.conn.execute(
            "SELECT MIN(seq) FROM event_log WHERE event_type = 'trial.recorded'"
        ).fetchone()[0]
        llm_trials = {
            str(row["spec_hash"])
            for row in ledger.conn.execute(
                "SELECT spec_hash FROM trials WHERE author_kind = 'llm'"
            ).fetchall()
        }

    (event,) = events
    assert int(event["seq"]) < int(first_trial), "the exchange is recorded before its trials"
    payload = json.loads(event["payload_json"])
    assert (payload["n_accepted"], payload["n_refused"]) == (6, 2)
    assert {StrategySpec.parse(spec).spec_hash for spec in payload["specs"]} == llm_trials

    system, user = client.calls[0]
    assert (payload["system_prompt"], payload["user_prompt"]) == (system, user)
    for prompt in (system, user):
        audit_prompt(prompt)
        assert UID not in prompt
        assert UID.split(":")[1] not in prompt
        assert vintage_id not in prompt


def _seed_walk(env: dict[str, Any], *, base: datetime, uid: str, start: str, seed: int) -> None:
    """A driftless random walk for one instrument, at any date and price level."""
    pinned = load_hard_limits(env["limits"])
    with _ledger(env) as ledger:
        store = BarStore(ledger, root=env["bars"], scale=pinned.limits.data.price_scale)
        walk = random.Random(seed)
        price = Decimal(start)
        bars = []
        for day in range(500):
            price = max(price + Decimal(str(round(walk.gauss(0, 1.1), 4))), Decimal("1.00"))
            bars.append(_daily(uid, base + timedelta(days=day), price))
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


def _vintage_at(env: dict[str, Any], *, base: datetime, uid: str, start: str, seed: int) -> str:
    _init(env)
    _seed_walk(env, base=base, uid=uid, start=start, seed=seed)
    sealed = _run(["data", "seal", "--resolution", "daily", *env["bar_args"]])
    assert sealed.exit_code == 0, _out(sealed)
    with _ledger(env) as ledger:
        row = ledger.conn.execute("SELECT vintage_id FROM data_snapshots").fetchone()
    return str(row["vintage_id"])


def test_two_markets_years_apart_produce_the_same_prompt(
    cli_env: dict[str, Any], tmp_path: Path
) -> None:
    """**The strongest form of price-blindness: the prompt cannot tell them apart.**

    Two vintages with different instruments, different decades and prices an
    order of magnitude apart, but the same length. Everything a model could use
    to recognise a period differs between them; the prompts are byte-identical.
    """
    other = {
        **cli_env,
        "db": tmp_path / "other.db",
        "bars": tmp_path / "other_bars",
        "bar_args": [
            "--limits",
            str(cli_env["limits"]),
            "--db",
            str(tmp_path / "other.db"),
            "--bars",
            str(tmp_path / "other_bars"),
        ],
    }
    first = _vintage_at(
        cli_env, base=datetime(2024, 1, 2, tzinfo=UTC), uid=UID, start="100.00", seed=77
    )
    second = _vintage_at(
        other,
        base=datetime(2011, 6, 6, tzinfo=UTC),
        uid="isin:GB0002634946",
        start="1450.00",
        seed=5,
    )
    first_client = FakeClient(reply=_reply_of_eight())
    second_client = FakeClient(reply=_reply_of_eight())
    _run_cycle(cli_env, first, first_client)
    _run_cycle(other, second, second_client)

    assert first_client.calls == second_client.calls


def test_a_failed_model_call_records_no_trial(cli_env: dict[str, Any]) -> None:
    """The call fails before anything is evaluated, so the search did not happen
    — and a search that did not happen leaves no trial rows to deflate against."""
    vintage_id = _sealed_vintage(cli_env)
    client = FakeClient(reply="", stop_reason="refusal")
    with pytest.raises(LLMError):
        _run_cycle(cli_env, vintage_id, client)

    with _ledger(cli_env) as ledger:
        counts = {
            event_type: ledger.conn.execute(
                "SELECT COUNT(*) FROM event_log WHERE event_type = ?", (event_type,)
            ).fetchone()[0]
            for event_type in ("trial.recorded", "search.completed", "search.specs_proposed")
        }
    assert counts == {"trial.recorded": 0, "search.completed": 0, "search.specs_proposed": 0}


def test_a_proposer_and_seeds_are_one_or_the_other(cli_env: dict[str, Any]) -> None:
    """A seeded search starts by mutating its seeds, so a proposer passed with
    them would never be asked — and would be recorded as having run."""
    vintage_id = _sealed_vintage(cli_env)
    pinned = load_hard_limits(cli_env["limits"])
    with _ledger(cli_env) as ledger:
        registered = SpecRegistry(
            ledger, per_lineage_budget_ccy=pinned.limits.loss.per_lineage_budget_ccy
        ).register(
            StrategySpec.parse({**item(), "name": "seed", "min_holding_minutes": 1440}),
            author_kind=AuthorKind.HUMAN,
            at=datetime(2026, 1, 5, tzinfo=UTC),
        )
    client = FakeClient(reply=_reply_of_eight())
    with pytest.raises(CycleError, match="one search, one source"):
        _run_cycle(cli_env, vintage_id, client, seeds=(registered.strategy_id,))
    assert client.calls == []


# --------------------------------------------------------------------------
# Through the CLI
# --------------------------------------------------------------------------


def _cycle_args(env: dict[str, Any], vintage_id: str, *extra: str) -> list[str]:
    return [
        "research",
        "cycle",
        vintage_id,
        "--trials",
        "16",
        "--per-generation",
        "8",
        "--proposer",
        "llm",
        *extra,
        *env["bar_args"],
    ]


def _fake_sdk(monkeypatch: pytest.MonkeyPatch, client: FakeClient) -> list[dict[str, Any]]:
    """Stand in for `AnthropicClient`, recording how the CLI built it."""
    built: list[dict[str, Any]] = []

    def build(**kwargs: Any) -> FakeClient:
        built.append(kwargs)
        return client

    monkeypatch.setattr(adapter, "AnthropicClient", build)
    return built


def test_the_cli_runs_a_model_backed_search(
    cli_env: dict[str, Any], monkeypatch: pytest.MonkeyPatch
) -> None:
    vintage_id = _sealed_vintage(cli_env)
    built = _fake_sdk(monkeypatch, FakeClient(reply=_reply_of_eight()))
    result = _run(_cycle_args(cli_env, vintage_id))

    assert result.exit_code == 0, _out(result)
    assert built == [{"model": "claude-opus-5", "fallbacks": True}]
    output = _out(result)
    assert "refusal fallbacks on" in output
    assert "llm: asked fake-model-1 for 8 spec(s)" in output
    assert "not reproducible from the seed" in output


def test_the_cli_passes_the_model_and_the_fallback_choice(
    cli_env: dict[str, Any], monkeypatch: pytest.MonkeyPatch
) -> None:
    vintage_id = _sealed_vintage(cli_env)
    built = _fake_sdk(monkeypatch, FakeClient(reply=_reply_of_eight()))
    result = _run(_cycle_args(cli_env, vintage_id, "--model", "claude-sonnet-5", "--no-fallback"))
    assert result.exit_code == 0, _out(result)
    assert built == [{"model": "claude-sonnet-5", "fallbacks": False}]
    assert "refusal fallbacks off" in _out(result)


def test_a_refusal_through_the_cli_is_a_setup_exit_with_nothing_recorded(
    cli_env: dict[str, Any], monkeypatch: pytest.MonkeyPatch
) -> None:
    vintage_id = _sealed_vintage(cli_env)
    _fake_sdk(monkeypatch, FakeClient(reply="", stop_reason="refusal", refusal="cyber"))
    result = _run(_cycle_args(cli_env, vintage_id))

    assert result.exit_code == 2
    assert "declined" in _out(result)
    # The operator's next question is whether the failure cost trials; the
    # answer is printed, and it is true.
    assert "No trial was recorded" in _out(result)
    with _ledger(cli_env) as ledger:
        assert ledger.conn.execute("SELECT COUNT(*) FROM trials").fetchone()[0] == 0


def test_the_cli_refuses_to_seed_a_model_backed_search(cli_env: dict[str, Any]) -> None:
    result = _run(_cycle_args(cli_env, "vint_ghost", "--from", "stg_anything"))
    assert result.exit_code == 2
    assert "--from" in _out(result)


def test_the_live_broker_key_stops_the_cli_before_any_data_is_read(
    cli_env: dict[str, Any], monkeypatch: pytest.MonkeyPatch
) -> None:
    """No ledger exists here at all: the refusal must come first, and name the key."""
    monkeypatch.setenv("T212_LIVE_API_KEY", "not-a-real-key")
    result = _run(_cycle_args(cli_env, "vint_ghost"))
    assert result.exit_code == 2
    assert "T212_LIVE_API_KEY" in _out(result)
    assert "tb init" not in _out(result)
