"""`tb replay --fill`: one fill, explained from the ledger alone.

The plan's success condition ends on it — the ledger can explain the whole
chain from a fill back to the spec — and the README has advertised the command
since M0. The lineage tables have carried the chain since M4 (fills → intents →
decisions → risk verdicts, and on through the registry to the spec, the search
that proposed it, its holdout and the gate that promoted it), but nothing read
it back, so "why did it buy that, months ago" was answerable only by hand in
SQL.

This reads the chain and **checks** it rather than only printing it:

* the hash chain is intact, since every other answer here rests on it;
* the feature snapshot hash recomputes from the features recorded with the
  decision, so those are the features the strategy saw;
* the spec hash recomputes from the stored spec;
* the spec, run on those features, reaches the recorded action — the decision
  follows from its inputs rather than merely sitting beside them;
* for a spec that reads a model, the model pinned by the spec, loaded through
  the store's hash checks and fed the recorded inputs, gives the recorded score
  — so "the model said 0.73" is checked, not quoted;
* no blocking risk rule failed on the decision the order answered.

The features are read from the `decision.made` event, not the `decisions`
projection: the event is the source of truth, and the projection's column was
written as a Python repr until it was found here.

Read-only. Nothing in this module writes to the ledger.
"""

from __future__ import annotations

import json
import sqlite3
from collections.abc import Mapping
from dataclasses import dataclass, field
from datetime import datetime
from decimal import Decimal, InvalidOperation
from typing import Any

from tb.core.errors import TbError
from tb.data.adjustments import Series
from tb.data.asof import UNKNOWN, BarWindow
from tb.data.provider import Resolution
from tb.features.pipeline import (
    FeatureSnapshot,
    FeatureValue,
    quantize_feature,
    snapshot_hash,
)
from tb.ledger.store import Ledger
from tb.ledger.verify import verify_chain
from tb.strategy.base import Action, PositionState
from tb.strategy.dsl.ops import DslStrategy, ModelSource
from tb.strategy.dsl.schema import SpecError, StrategySpec

Row = Mapping[str, Any]


class ReplayError(TbError):
    """The fill asked for is not in the ledger."""


@dataclass(frozen=True, slots=True)
class Check:
    """One property of the chain, verified rather than asserted."""

    name: str
    ok: bool
    detail: str


@dataclass(frozen=True, slots=True)
class FillReplay:
    """Everything the ledger records behind one fill, and what was verified."""

    fill: Row
    intent: Row | None = None
    decision: Row | None = None
    features: Mapping[str, str] = field(default_factory=dict)
    verdicts: tuple[Row, ...] = ()
    spec: Row | None = None
    trials: tuple[Row, ...] = ()
    holdout: Row | None = None
    promotion: Row | None = None
    round_trip: Row | None = None
    checks: tuple[Check, ...] = ()

    @property
    def ok(self) -> bool:
        return all(check.ok for check in self.checks)


def replay_fill(ledger: Ledger, fill_id: str, *, models: ModelSource | None = None) -> FillReplay:
    """Walk one fill back to its spec, checking each link that can be checked.

    `models` is where a spec's model terms are loaded to check their scores;
    without it a model-reading decision's score is reported unchecked, which
    fails the replay rather than passing it on trust.
    """
    fill = _one(ledger, "SELECT * FROM fills WHERE fill_id = ?", fill_id)
    if fill is None:
        raise ReplayError(f"there is no fill {fill_id!r} in this ledger")
    checks = [_chain(ledger)]

    intent = (
        None
        if fill["intent_id"] is None
        else _one(ledger, "SELECT * FROM order_intents WHERE intent_id = ?", fill["intent_id"])
    )
    decision = (
        None
        if intent is None or intent["decision_id"] is None
        else _one(ledger, "SELECT * FROM decisions WHERE decision_id = ?", intent["decision_id"])
    )
    round_trip = _one(ledger, "SELECT * FROM round_trips WHERE closing_fill_id = ?", fill_id)
    if intent is None or decision is None:
        # A flatten answers no decision, and a fill history reported for an
        # order this process never placed has no intent. Both are recorded
        # facts rather than breaks in the chain.
        return FillReplay(fill=fill, intent=intent, round_trip=round_trip, checks=tuple(checks))

    features = _recorded_features(ledger, decision)
    values = _values(features)
    checks.append(_feature_check(decision, values))
    verdicts = tuple(
        _all(
            ledger,
            "SELECT * FROM risk_verdicts WHERE decision_id = ? ORDER BY rule_name",
            decision["decision_id"],
        )
    )
    checks.append(_risk_check(intent, verdicts))

    key = (decision["strategy_id"], decision["strategy_version"])
    spec_row = _one(
        ledger, "SELECT * FROM strategy_specs WHERE strategy_id = ? AND version = ?", *key
    )
    trials: tuple[Row, ...] = ()
    holdout = promotion = None
    if spec_row is not None:
        spec, spec_check = _spec_check(spec_row)
        checks.append(spec_check)
        if spec is not None and values is not None:
            checks.append(_redecision_check(spec, decision, values))
            if spec.model_refs:
                checks.append(_model_check(spec, decision, values, models))
        trials = tuple(
            _all(
                ledger,
                "SELECT * FROM trials WHERE spec_hash = ? ORDER BY recorded_at",
                spec_row["spec_hash"],
            )
        )
        holdout = _one(
            ledger,
            "SELECT * FROM holdout_evaluations WHERE strategy_id = ? AND version = ?",
            *key,
        )
        promotion = _one(
            ledger,
            "SELECT * FROM promotions WHERE strategy_id = ? AND version = ?"
            " ORDER BY deciding_event_seq DESC LIMIT 1",
            *key,
        )

    return FillReplay(
        fill=fill,
        intent=intent,
        decision=decision,
        features=features,
        verdicts=verdicts,
        spec=spec_row,
        trials=trials,
        holdout=holdout,
        promotion=promotion,
        round_trip=round_trip,
        checks=tuple(checks),
    )


# --------------------------------------------------------------------------
# The checks
# --------------------------------------------------------------------------


def _chain(ledger: Ledger) -> Check:
    report = verify_chain(ledger, stop_on_first=True)
    return Check("ledger", report.ok, report.summary())


def _feature_check(decision: Row, values: dict[str, FeatureValue] | None) -> Check:
    """The recorded features hash to the snapshot hash the decision carries.

    Tried against each series, since the series is part of the hash and not a
    column of its own; whichever matches is named.
    """
    recorded = str(decision["feature_snapshot_hash"])
    if values is None:
        return Check("features", False, "the recorded feature vector does not parse")
    as_of = datetime.fromisoformat(str(decision["as_of_utc"]))
    uid = str(decision["instrument_uid"])
    for series in Series:
        if snapshot_hash(as_of=as_of, instrument_uid=uid, series=series, values=values) == (
            recorded
        ):
            return Check(
                "features",
                True,
                f"{len(values)} recorded feature(s) hash to {recorded[:12]}… "
                f"({series.value} series)",
            )
    return Check(
        "features",
        False,
        f"the recorded features do not hash to {recorded[:12]}…: what was recorded is "
        "not what the strategy saw",
    )


def _risk_check(intent: Row, verdicts: tuple[Row, ...]) -> Check:
    if not verdicts:
        return Check(
            "risk",
            False,
            f"no risk verdict is recorded for the decision behind {intent['intent_id']}, "
            f"yet it carries risk token {intent['risk_token_id']}",
        )
    blocked = [
        str(row["rule_name"])
        for row in verdicts
        if int(row["is_blocking"]) and str(row["verdict"]) == "block"
    ]
    if blocked:
        return Check(
            "risk",
            False,
            f"blocking rule(s) {', '.join(blocked)} refused this decision, yet an order "
            f"was placed under token {intent['risk_token_id']}",
        )
    return Check(
        "risk",
        True,
        f"{len(verdicts)} rule(s) evaluated, none blocking; token {intent['risk_token_id']}",
    )


def _spec_check(row: Row) -> tuple[StrategySpec | None, Check]:
    try:
        spec = StrategySpec.parse(json.loads(str(row["spec_json"])))
    except (SpecError, ValueError) as exc:
        return None, Check("spec", False, f"the stored spec no longer parses: {exc}")
    recorded = str(row["spec_hash"])
    if spec.spec_hash != recorded:
        return spec, Check(
            "spec",
            False,
            f"the stored spec hashes to {spec.spec_hash[:12]}…, not the "
            f"{recorded[:12]}… it was registered under",
        )
    return spec, Check("spec", True, f"the stored spec hashes to {recorded[:12]}…")


def _redecision_check(spec: StrategySpec, decision: Row, values: dict[str, FeatureValue]) -> Check:
    """Run the spec on the recorded features and compare the action.

    Flat for an entry and holding for an exit, with no entry time, so the
    spec's own minimum hold — a matter of the clock rather than the features
    — does not decide it.
    """
    recorded = Action(str(decision["action"]))
    as_of = datetime.fromisoformat(str(decision["as_of_utc"]))
    uid = str(decision["instrument_uid"])
    snapshot = FeatureSnapshot(
        as_of=as_of,
        instrument_uid=uid,
        values=values,
        snapshot_hash=str(decision["feature_snapshot_hash"]),
        n_bars_seen=0,
    )
    held = Decimal(1) if recorded is Action.EXIT else Decimal(0)
    replayed = DslStrategy(
        spec=spec,
        strategy_id=str(decision["strategy_id"]),
        version=int(decision["strategy_version"]),
    ).decide(
        snapshot=snapshot,
        window=BarWindow(
            as_of=as_of, resolution=Resolution(str(decision["resolution"])), _by_uid={}
        ),
        position=PositionState(instrument_uid=uid, quantity=held),
    )
    if replayed.action is recorded:
        return Check(
            "decision",
            True,
            f"the spec, run on the recorded features, decides {recorded.value} again",
        )
    return Check(
        "decision",
        False,
        f"the spec, run on the recorded features, decides {replayed.action.value}, "
        f"not the {recorded.value} that was recorded ({replayed.rationale})",
    )


def _model_check(
    spec: StrategySpec,
    decision: Row,
    values: dict[str, FeatureValue],
    models: ModelSource | None,
) -> Check:
    """Each model the spec reads, re-scored from the recorded inputs.

    Through the store, so the artifact is hash-checked against the ledger
    before it is parsed; and rounded by the pipeline's own rule, so an exact
    comparison is the right one.
    """
    if models is None:
        return Check(
            "model",
            False,
            f"the spec reads {[ref.model_id for ref in spec.model_refs]}, and no model store "
            "was given to check the score it recorded",
        )
    as_of = datetime.fromisoformat(str(decision["as_of_utc"]))
    lines: list[str] = []
    for ref in spec.model_refs:
        try:
            scorer = models.scorer_for(ref.model_id, artifact_sha256=ref.artifact_sha256)
        except TbError as exc:
            return Check("model", False, f"{ref.model_id} cannot be loaded as pinned: {exc}")
        recorded = values.get(ref.model_id)
        inputs = [values.get(name) for name in scorer.inputs]
        if any(not isinstance(value, Decimal) for value in inputs):
            ok = recorded is UNKNOWN
            detail = f"{ref.model_id} had an unknown input, so its score was UNKNOWN"
        else:
            row = [float(value) for value in inputs if isinstance(value, Decimal)]
            score = scorer.score(row, as_of=as_of)
            expected = UNKNOWN if score is None else quantize_feature(Decimal(score))
            ok = recorded == expected
            detail = (
                f"{ref.model_id}, fed the recorded {', '.join(scorer.inputs)}, scores "
                f"{expected}; recorded {recorded}"
            )
        if not ok:
            return Check("model", False, f"{detail} — the recorded score is not this model's")
        lines.append(detail)
    return Check("model", True, "; ".join(lines))


# --------------------------------------------------------------------------
# Reading
# --------------------------------------------------------------------------


def _recorded_features(ledger: Ledger, decision: Row) -> dict[str, str]:
    event = ledger.get(int(decision["deciding_event_seq"]))
    if event is not None:
        vector = json.loads(str(event["payload_json"])).get("feature_vector")
        if isinstance(vector, dict):
            return {str(name): str(value) for name, value in vector.items()}
    return {}


def _values(features: Mapping[str, str]) -> dict[str, FeatureValue] | None:
    values: dict[str, FeatureValue] = {}
    for name, text in features.items():
        if text == "UNKNOWN":
            values[name] = UNKNOWN
            continue
        try:
            values[name] = Decimal(text)
        except InvalidOperation:
            return None
    return values


def _one(ledger: Ledger, sql: str, *params: object) -> Row | None:
    row: sqlite3.Row | None = ledger.conn.execute(sql, params).fetchone()
    return None if row is None else dict(row)


def _all(ledger: Ledger, sql: str, *params: object) -> list[Row]:
    return [dict(row) for row in ledger.conn.execute(sql, params).fetchall()]
