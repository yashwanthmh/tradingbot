"""The sealed holdout: enforced by the data layer, evaluated exactly once.

With `paper_shadow_sessions: 0` a strategy that clears the gate is funded with
real money and no human looks at it. The holdout is the one piece of evidence in
that decision that the search could not have fitted to, so two properties have
to be structural rather than procedural:

**The research process cannot see it.** Not "should not" — cannot. Two defences
compose. `SealedBarSource` holds no bar past the boundary, so post-boundary data
is not in memory; and `ForwardOnlyReader.sealed_from` (in `tb.data.asof`, the
one function every read goes through) refuses to advance to the boundary at all.
The first makes a leak return nothing; the second makes the attempt fatal.

**It can be evaluated once per version.** `UNIQUE (strategy_id, version)` on
`holdout_evaluations` is the mechanism. A holdout that can be re-evaluated is
not a holdout: "failed, tweak, resubmit" fits to it one bit per attempt, and a
few dozen attempts is enough to fit it thoroughly. Failing is therefore
**terminal** — the strategy is retired, and a mutation of it is a new strategy
with its own single attempt, whose lineage carries the parent's trial count into
the multiplicity haircut.

**Where the boundary comes from.** A fraction of the vintage's window, computed
from the vintage itself, so it is a property of the data rather than of whoever
is evaluating. Recomputed identically by every caller from the same vintage —
`holdout_boundary` is a pure function, and its inputs are all hashed into the
vintage id.
"""

from __future__ import annotations

import json
import sqlite3
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass
from datetime import datetime, timedelta
from decimal import Decimal

from tb.core.clock import from_iso, now_utc, to_iso
from tb.core.errors import TbError
from tb.core.ids import new_id
from tb.data.asof import BarSource, ForwardOnlyReader, HoldoutViolation
from tb.data.provider import Bar, Resolution
from tb.data.snapshot import DatasetVintage
from tb.ledger.events import (
    Actor,
    EventType,
    HoldoutEvaluatedPayload,
    HoldoutViolationPayload,
)
from tb.ledger.store import Ledger

# How much of a vintage's window is held back. A quarter is the usual choice and
# the reason is arithmetic rather than convention: the holdout has to be long
# enough for `min_oos_trades` to be reachable on a strategy holding positions
# for days, and short enough to leave the search something to work with. At ten
# years of daily bars a quarter is about 630 sessions, which at a multi-day
# holding period is a few dozen trades — just enough, and the gate's trade
# minimum is what refuses the cases where it is not.
DEFAULT_HOLDOUT_FRACTION = 0.25

# The shortest holdout worth evaluating against. Below this the out-of-sample
# statistics are noise, and a gate that read them would be promoting on noise
# while believing it had an independent check.
MIN_HOLDOUT_DAYS = 60


class HoldoutError(TbError):
    """A holdout could not be constructed or evaluated."""


class HoldoutAlreadyEvaluated(HoldoutError):
    """This version has already spent its single evaluation.

    Its own type because the caller's correct response is specific: not retry,
    not widen the window, but treat the recorded result as final. A searcher
    that caught a generic error here and retried would be doing exactly the
    thing the uniqueness constraint exists to prevent.
    """


@dataclass(frozen=True, slots=True)
class HoldoutWindow:
    """Where the training window ends and the holdout begins."""

    vintage_id: str
    sealed_from: datetime
    train_start: datetime
    holdout_end: datetime

    @property
    def train_days(self) -> int:
        return max(0, (self.sealed_from - self.train_start).days)

    @property
    def holdout_days(self) -> int:
        return max(0, (self.holdout_end - self.sealed_from).days)

    @property
    def is_usable(self) -> bool:
        return self.holdout_days >= MIN_HOLDOUT_DAYS and self.train_days > 0

    def summary(self) -> str:
        return (
            f"{self.vintage_id}: train {self.train_start.date()} to "
            f"{self.sealed_from.date()} ({self.train_days}d), holdout "
            f"{self.sealed_from.date()} to {self.holdout_end.date()} "
            f"({self.holdout_days}d)"
        )


@dataclass(frozen=True, slots=True)
class HoldoutResult:
    """One holdout evaluation, recorded and unrepeatable."""

    evaluation_id: str
    strategy_id: str
    version: int
    lineage_id: str
    spec_hash: str
    vintage_id: str
    sealed_from: datetime
    passed: bool
    evaluated_at: datetime
    # The bar resolution the evaluation ran at. Stored rather than assumed,
    # because the deflated *probability* converts an annualised Sharpe to a
    # per-period one and the factor differs by about eight times between daily
    # and minute. Reading it from config at gate time would annualise a daily
    # strategy by a minute factor the moment someone widened
    # `allowed_live_resolutions`.
    resolution: str = "daily"
    n_trades: int | None = None
    net_sharpe: float | None = None
    net_return_pct: float | None = None
    max_drawdown_pct: float | None = None
    cost_drag_bps: float | None = None
    returns: tuple[float, ...] = ()
    detail: str = ""

    @property
    def label(self) -> str:
        return f"{self.strategy_id}@v{self.version}"


def holdout_boundary(
    vintage: DatasetVintage,
    *,
    fraction: float = DEFAULT_HOLDOUT_FRACTION,
) -> HoldoutWindow:
    """Where this vintage's holdout starts. A pure function of the vintage.

    Pure and derived so every caller computes the same boundary from the same
    data. A boundary passed in by the caller would be a boundary the caller
    could move after seeing a result, which is the same failure as re-evaluating
    — slower, and harder to notice.
    """
    if not 0.0 < fraction < 1.0:
        raise HoldoutError(
            f"holdout fraction must be strictly between 0 and 1, got {fraction}. "
            "A fraction of 0 is no holdout and a fraction of 1 is no training data."
        )
    if vintage.window_start is None or vintage.window_end is None:
        raise HoldoutError(
            f"{vintage.vintage_id} has no window, so it cannot be split. An empty "
            "vintage is not admissible evidence for anything."
        )
    span = vintage.window_end - vintage.window_start
    if span <= timedelta(0):
        raise HoldoutError(
            f"{vintage.vintage_id} spans {span}: there is nothing to split. A "
            "single-instant vintage cannot support a train/holdout separation."
        )
    sealed_from = vintage.window_start + timedelta(seconds=span.total_seconds() * (1.0 - fraction))
    return HoldoutWindow(
        vintage_id=vintage.vintage_id,
        sealed_from=sealed_from,
        train_start=vintage.window_start,
        holdout_end=vintage.window_end,
    )


@dataclass(slots=True)
class SealedBarSource:
    """A bar source with the holdout removed from it.

    Two behaviours, and both matter:

    **A request that explicitly names the holdout raises.** That is the "window
    provider that raises past SEALED_FROM" the plan asks for, and it catches a
    caller reading the source directly rather than through a reader.

    **Bars past the boundary are not yielded.** So even a caller who somehow
    gets a raw reader over this source finds nothing there. A filter alone would
    be a silent shortening; a raise alone would leave the filter's job to the
    caller. Together, the holdout is neither visible nor quietly absent.

    The filter tests **both** time axes. A bar whose `bar_open` precedes the
    boundary but whose `available_at` follows it was not knowable before the
    boundary, so including it would leak knowledge time even with event time
    respected — which is the more subtle half of the same mistake.
    """

    inner: BarSource
    sealed_from: datetime
    # Called with the requested instant and which bound named it, so an
    # attempted read is recorded and not only raised. A raise stops one process;
    # the event is what makes a *pattern* of attempts visible, and a searcher
    # repeatedly reaching past the boundary is a finding about the searcher.
    on_violation: Callable[[datetime, str], None] | None = None

    def bars_for(
        self,
        instrument_uid: str,
        resolution: Resolution,
        *,
        start: datetime | None = None,
        end: datetime | None = None,
    ) -> Iterable[Bar]:
        if start is not None and start >= self.sealed_from:
            self._refuse(start, "start")
        if end is not None and end > self.sealed_from:
            self._refuse(end, "end")
        for bar in self.inner.bars_for(instrument_uid, resolution, start=start, end=end):
            if bar.bar_open_utc >= self.sealed_from:
                continue
            if bar.available_at_utc >= self.sealed_from:
                continue
            yield bar

    def instruments(self) -> Iterable[str]:
        return self.inner.instruments()

    def _refuse(self, requested: datetime, which: str) -> None:
        if self.on_violation is not None:
            self.on_violation(requested, which)
        raise HoldoutViolation(
            f"a research read named {which}={requested.isoformat()}, which is at or past "
            f"the holdout boundary {self.sealed_from.isoformat()}. The holdout is the "
            "only check on this process's own output that the process did not produce, "
            "so reaching into it removes the evidence rather than adding any."
        )


# A decision is taken this long after the bar it acts on became knowable. The
# same offset the null population uses, so a candidate's training statistics and
# its holdout statistics are computed on schedules with the same shape.
DECISION_OFFSET = timedelta(hours=1)


def decisions_between(
    bars: Iterable[Bar],
    *,
    start: datetime | None = None,
    end: datetime | None = None,
) -> list[datetime]:
    """One decision per distinct knowable instant, in `[start, end)`.

    Used by both the search cycle and the single holdout evaluation, so the
    statistics a candidate was selected on and the ones it is judged on come
    from schedules built the same way. Two properties matter:

    **Filtered on the decision time, not on the bar time.** A daily bar opened
    the day before the boundary is knowable the day after it, so filtering
    training decisions by `bar_open < sealed_from` produces a last decision
    *past* the seal — which the sealed reader refuses, correctly, and the whole
    search dies on its final bar. Filtering on the decision itself partitions
    time cleanly: every training decision is strictly before the seal and every
    holdout decision is at or after it, with no gap between them.

    **Deduplicated.** A vintage of twenty-five instruments has twenty-five bars
    per session, all knowable at the same instant, and an undeduplicated
    schedule decides twenty-five times at each — an equity curve with twenty-four
    zero returns per day, which deflates measured volatility and inflates every
    Sharpe read from it.
    """
    instants = {bar.available_at_utc + DECISION_OFFSET for bar in bars}
    return sorted(
        instant
        for instant in instants
        if (start is None or instant >= start) and (end is None or instant < end)
    )


def training_reader(
    source: BarSource,
    *,
    sealed_from: datetime,
    resolution: Resolution,
    instrument_uids: Sequence[str],
    lookback: timedelta | None = None,
) -> ForwardOnlyReader:
    """A reader a research process may hold. Cannot reach the holdout.

    Both defences at once: the source is sealed, and the reader knows the
    boundary so an attempt to advance into it raises rather than returning an
    empty window. The empty window is the dangerous version — a strategy that
    sees no bars concludes it has no signal, and a searcher records a negative
    result about a window it never actually tested.
    """
    return ForwardOnlyReader(
        source=SealedBarSource(inner=source, sealed_from=sealed_from),
        resolution=resolution,
        instrument_uids=tuple(instrument_uids),
        lookback=lookback,
        sealed_from=sealed_from,
    )


def evaluation_reader(
    source: BarSource,
    *,
    resolution: Resolution,
    instrument_uids: Sequence[str],
    lookback: timedelta | None = None,
) -> ForwardOnlyReader:
    """A reader over the whole vintage, for the single holdout evaluation.

    Deliberately *not* sealed, and deliberately a separate function with its own
    name. The holdout evaluation is the one process that must see the holdout,
    and its lookback reaches back into the training window — a feature needing
    100 bars on the first holdout decision has to get them from somewhere, and
    they are training data, which is fine: the boundary is about what may inform
    the *choice* of strategy, not about what a fixed strategy may compute from.

    Separating the two functions is the point. A single function with a
    `sealed: bool` parameter would default to something, and a caller that
    omitted the argument would get a reader whose safety depended on which
    default was chosen.
    """
    return ForwardOnlyReader(
        source=source,
        resolution=resolution,
        instrument_uids=tuple(instrument_uids),
        lookback=lookback,
    )


class HoldoutRegistry:
    """Records holdout evaluations, and refuses a second one."""

    def __init__(self, ledger: Ledger, *, run_id: str | None = None) -> None:
        self._ledger = ledger
        self._run_id = run_id

    def existing(self, strategy_id: str, version: int = 1) -> HoldoutResult | None:
        row: sqlite3.Row | None = self._ledger.conn.execute(
            "SELECT * FROM holdout_evaluations WHERE strategy_id = ? AND version = ?",
            (strategy_id, version),
        ).fetchone()
        return None if row is None else _row_to_result(row)

    def has_been_evaluated(self, strategy_id: str, version: int = 1) -> bool:
        return self.existing(strategy_id, version) is not None

    def record(
        self,
        *,
        strategy_id: str,
        version: int,
        lineage_id: str,
        spec_hash: str,
        vintage_id: str,
        sealed_from: datetime,
        passed: bool,
        resolution: str = "daily",
        window_start: datetime | None = None,
        window_end: datetime | None = None,
        backtest_id: str | None = None,
        n_trades: int | None = None,
        net_sharpe: float | None = None,
        net_return_pct: float | None = None,
        max_drawdown_pct: float | None = None,
        cost_drag_bps: float | None = None,
        returns: Sequence[Decimal] | Sequence[float] = (),
        detail: str = "",
        at: datetime | None = None,
    ) -> HoldoutResult:
        """Record the one evaluation this version gets.

        Refuses a second, by checking *and* by the uniqueness constraint behind
        it. The explicit check exists to raise `HoldoutAlreadyEvaluated` with
        the first result attached, so the caller learns what the answer was
        rather than only that it asked twice; the constraint exists because the
        check can be raced and the constraint cannot.
        """
        prior = self.existing(strategy_id, version)
        if prior is not None:
            raise HoldoutAlreadyEvaluated(
                f"{strategy_id}@v{version} was already evaluated against the holdout on "
                f"{prior.evaluated_at.isoformat()} and "
                f"{'passed' if prior.passed else 'failed'}. A version gets one "
                "evaluation: 'failed, tweak, resubmit' fits the holdout a bit at a time, "
                "and a few dozen rounds fits it completely. Register the tweak as a new "
                "strategy — its lineage carries this one's trial count into the "
                "multiplicity haircut, which is the cost of the second attempt."
            )

        moment = at or now_utc()
        stored = tuple(float(r) for r in returns)
        result = HoldoutResult(
            evaluation_id=new_id("hold", length=12),
            strategy_id=strategy_id,
            version=version,
            lineage_id=lineage_id,
            spec_hash=spec_hash,
            vintage_id=vintage_id,
            sealed_from=sealed_from,
            passed=passed,
            evaluated_at=moment,
            resolution=resolution,
            n_trades=n_trades,
            net_sharpe=net_sharpe,
            net_return_pct=net_return_pct,
            max_drawdown_pct=max_drawdown_pct,
            cost_drag_bps=cost_drag_bps,
            returns=stored,
            detail=detail,
        )

        try:
            with self._ledger.transaction() as tx:
                event = tx.append(
                    EventType.HOLDOUT_EVALUATED,
                    strategy_id,
                    HoldoutEvaluatedPayload(
                        evaluation_id=result.evaluation_id,
                        strategy_id=strategy_id,
                        version=version,
                        lineage_id=lineage_id,
                        spec_hash=spec_hash,
                        vintage_id=vintage_id,
                        sealed_from=to_iso(sealed_from),
                        passed=passed,
                        n_trades=n_trades,
                        net_sharpe=net_sharpe,
                        net_return_pct=net_return_pct,
                        max_drawdown_pct=max_drawdown_pct,
                        detail=detail,
                    ),
                    actor=Actor.SYSTEM,
                    run_id=self._run_id,
                )
                tx.execute(
                    """
                    INSERT INTO holdout_evaluations (
                        evaluation_id, strategy_id, version, lineage_id, spec_hash,
                        vintage_id, resolution, sealed_from, window_start, window_end,
                        backtest_id, n_trades, net_sharpe, net_return_pct,
                        max_drawdown_pct, cost_drag_bps, returns_json, passed, detail,
                        evaluated_at, evaluating_event_seq
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        result.evaluation_id,
                        strategy_id,
                        version,
                        lineage_id,
                        spec_hash,
                        vintage_id,
                        resolution,
                        to_iso(sealed_from),
                        None if window_start is None else to_iso(window_start),
                        None if window_end is None else to_iso(window_end),
                        backtest_id,
                        n_trades,
                        net_sharpe,
                        net_return_pct,
                        max_drawdown_pct,
                        cost_drag_bps,
                        json.dumps(list(stored)) if stored else None,
                        1 if passed else 0,
                        detail,
                        to_iso(moment),
                        event.seq,
                    ),
                )
        except sqlite3.IntegrityError as exc:
            # The constraint, not the check. Reached only if two processes
            # evaluated the same version concurrently — which is exactly the
            # case the check cannot cover, and the reason the constraint is
            # there rather than the check being trusted.
            raise HoldoutAlreadyEvaluated(
                f"{strategy_id}@v{version} already has a holdout evaluation; the "
                f"database refused a second ({exc}). Two processes evaluated the same "
                "version at once, and only one of them counts."
            ) from exc

        return result

    def record_violation(
        self,
        *,
        sealed_from: datetime,
        requested_at: datetime,
        strategy_id: str | None = None,
        lineage_id: str | None = None,
        caller: str = "",
        detail: str = "",
    ) -> None:
        """Record that something tried to read past the boundary."""
        self._ledger.append(
            EventType.HOLDOUT_VIOLATION_ATTEMPTED,
            strategy_id or "holdout",
            HoldoutViolationPayload(
                strategy_id=strategy_id,
                lineage_id=lineage_id,
                sealed_from=to_iso(sealed_from),
                requested_at=to_iso(requested_at),
                caller=caller,
                detail=detail,
            ),
            actor=Actor.SEARCH,
            run_id=self._run_id,
        )

    def in_lineage(self, lineage_id: str) -> list[HoldoutResult]:
        rows = self._ledger.conn.execute(
            "SELECT * FROM holdout_evaluations WHERE lineage_id = ? ORDER BY evaluated_at",
            (lineage_id,),
        ).fetchall()
        return [_row_to_result(row) for row in rows]


def _row_to_result(row: sqlite3.Row) -> HoldoutResult:
    raw = row["returns_json"]
    return HoldoutResult(
        evaluation_id=str(row["evaluation_id"]),
        strategy_id=str(row["strategy_id"]),
        version=int(row["version"]),
        lineage_id=str(row["lineage_id"]),
        spec_hash=str(row["spec_hash"]),
        vintage_id=str(row["vintage_id"]),
        sealed_from=from_iso(str(row["sealed_from"])),
        passed=bool(row["passed"]),
        evaluated_at=from_iso(str(row["evaluated_at"])),
        resolution=str(row["resolution"]),
        n_trades=None if row["n_trades"] is None else int(row["n_trades"]),
        net_sharpe=None if row["net_sharpe"] is None else float(row["net_sharpe"]),
        net_return_pct=None if row["net_return_pct"] is None else float(row["net_return_pct"]),
        max_drawdown_pct=(
            None if row["max_drawdown_pct"] is None else float(row["max_drawdown_pct"])
        ),
        cost_drag_bps=None if row["cost_drag_bps"] is None else float(row["cost_drag_bps"]),
        returns=() if raw is None else tuple(json.loads(str(raw))),
        detail="" if row["detail"] is None else str(row["detail"]),
    )
