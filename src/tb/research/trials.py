"""Every trial, including every rejection. The denominator of the whole gate.

A deflated Sharpe is an ordinary Sharpe with the best-of-N expectation
subtracted, and N is a count of things tried. So this table is not
record-keeping — it is the denominator, and an undercount here makes every
downstream number wrong in the permissive direction: too small an N means too
small a haircut means a noise strategy clears the gate.

Three things that are easy to get wrong, all of them handled here rather than
left to the caller:

**Rejections count.** A candidate refused before its backtest ran was still a
draw from the search space. Recording only the ones that produced numbers would
report a search of three where a thousand happened.

**Errors count too.** A spec the pipeline could not evaluate is not a
non-event; it is a sample that came back empty. Excluding errors would let a
searcher lower its own apparent multiplicity by proposing specs that crash.

**The search is the selection universe, not the lineage.** Counting only within
a lineage leaves an evasion that a searcher finds without trying: give every
candidate a fresh lineage and each is a search of size one, with no haircut at
all. `n_trials_for_deflation` therefore takes the **larger** of the two counts,
and both are stamped onto the trial row at the moment it happens so a later
read cannot be fooled by a table that has since grown.

**The counts are stamped, not recomputed.** `trials_in_lineage_at_time` is the
count *as it stood* when the trial ran. Recomputing it at promotion time would
answer a different question — how big the search became afterwards — and would
make the same promotion decision produce different numbers on a re-read.
"""

from __future__ import annotations

import json
import random
import sqlite3
from collections.abc import Sequence
from dataclasses import dataclass, field
from datetime import datetime
from decimal import Decimal
from math import sqrt

from tb.core.clock import from_iso, now_utc, to_iso
from tb.core.errors import TbError
from tb.core.ids import new_id
from tb.ledger.events import Actor, EventType, SearchCompletedPayload, TrialPayload
from tb.ledger.store import Ledger
from tb.registry.models import AuthorKind, TrialOutcome

# How many per-period returns to keep on a trial row. PBO needs the returns
# matrix across a lineage's trials, not just their Sharpes, and a bounded slice
# is the difference between a table that can support the computation and one
# that cannot. 2,000 daily returns is about eight years — past the point where
# a longer series changes a cross-validated answer, and small enough that a
# thousand-trial search stores tens of megabytes rather than gigabytes.
MAX_STORED_RETURNS = 2_000

# How many trials the PBO matrix may be built from.
#
# CSCV costs `combinations x columns x periods` arithmetic in pure Python, so a
# thousand-trial search takes minutes rather than seconds — a real constraint on
# the pipeline, not only on its tests. 64 columns over 70 combinations is a few
# seconds and gives an estimate precise enough for a threshold of 0.25.
MAX_PBO_TRIALS = 64


class TrialError(TbError):
    """A trial could not be recorded."""


@dataclass(frozen=True, slots=True)
class Trial:
    """One candidate, and what became of it."""

    trial_id: str
    search_id: str
    lineage_id: str
    spec_hash: str
    author_kind: AuthorKind
    outcome: TrialOutcome
    recorded_at: datetime
    strategy_id: str | None = None
    strategy_version: int | None = None
    parent_strategy_id: str | None = None
    generation: int = 0
    rejection_reason: str = ""
    backtest_id: str | None = None
    vintage_id: str | None = None
    net_sharpe: float | None = None
    net_return_pct: float | None = None
    max_drawdown_pct: float | None = None
    n_trades: int | None = None
    cost_drag_bps: float | None = None
    returns: tuple[float, ...] = ()
    trials_in_lineage_at_time: int = 0
    trials_in_search_at_time: int = 0

    @property
    def n_trials_for_deflation(self) -> int:
        """The multiplicity this trial was drawn against.

        The larger of the two counts, and never below 1. Below 1 would mean "no
        search happened", and this row is evidence that one did.
        """
        return max(1, self.trials_in_lineage_at_time, self.trials_in_search_at_time)

    @property
    def is_measurable(self) -> bool:
        """Whether this trial produced a Sharpe worth putting in a dispersion."""
        return self.outcome.was_evaluated and self.net_sharpe is not None


@dataclass(frozen=True, slots=True)
class SearchSummary:
    """The totals for one search session.

    `n_passed_gate` against `n_proposed` is the number that says whether the
    gate works. A search promoting a tenth of what it proposes has found a
    market inefficiency or a bug in this repo, and the second is much more
    likely.
    """

    search_id: str
    n_proposed: int
    n_evaluated: int
    n_rejected: int
    n_errored: int
    n_passed_gate: int
    lineage_ids: tuple[str, ...] = ()

    @property
    def promotion_rate(self) -> float:
        if self.n_proposed == 0:
            return 0.0
        return self.n_passed_gate / self.n_proposed

    def summary(self) -> str:
        return (
            f"{self.search_id}: {self.n_proposed} proposed, {self.n_evaluated} evaluated, "
            f"{self.n_rejected} rejected, {self.n_errored} errored, "
            f"{self.n_passed_gate} cleared the gate "
            f"({self.promotion_rate * 100:.2f}%) across {len(self.lineage_ids)} lineage(s)"
        )


@dataclass(frozen=True, slots=True)
class Multiplicity:
    """How many times the space was sampled before this candidate survived.

    Carries the dispersion of the trial Sharpes alongside the count, because
    the expected maximum of N draws depends on both and a count alone cannot
    produce a haircut. `dispersion_measured` says whether that number came from
    the trials or from a conservative fallback — a distinction the gate must be
    able to see, since an unmeasured dispersion of zero would mean no haircut.
    """

    n_trials: int
    n_lineage_trials: int
    n_search_trials: int
    sharpe_dispersion: float
    dispersion_measured: bool
    n_measurable: int

    @property
    def caveats(self) -> tuple[str, ...]:
        notes: list[str] = []
        if not self.dispersion_measured:
            notes.append(
                f"the dispersion of trial Sharpes could not be measured from "
                f"{self.n_measurable} measurable trial(s), so a conservative default "
                "was used instead of zero: an unmeasured dispersion of zero would mean "
                "no multiplicity haircut at all"
            )
        if self.n_search_trials > self.n_lineage_trials:
            notes.append(
                f"deflated against the search's {self.n_search_trials} trials rather "
                f"than the lineage's {self.n_lineage_trials}: the selection universe is "
                "the search, and counting only within a lineage would let one candidate "
                "per lineage escape deflation entirely"
            )
        return tuple(notes)


class TrialLog:
    """Records trials and answers multiplicity questions about them."""

    def __init__(self, ledger: Ledger, *, run_id: str | None = None) -> None:
        self._ledger = ledger
        self._run_id = run_id

    # -- recording ---------------------------------------------------------

    def record(
        self,
        *,
        search_id: str,
        lineage_id: str,
        spec_hash: str,
        author_kind: AuthorKind,
        outcome: TrialOutcome,
        strategy_id: str | None = None,
        strategy_version: int | None = None,
        parent_strategy_id: str | None = None,
        generation: int = 0,
        rejection_reason: str = "",
        backtest_id: str | None = None,
        vintage_id: str | None = None,
        net_sharpe: float | None = None,
        net_return_pct: float | None = None,
        max_drawdown_pct: float | None = None,
        n_trades: int | None = None,
        cost_drag_bps: float | None = None,
        returns: Sequence[Decimal] | Sequence[float] = (),
        at: datetime | None = None,
        emit_event: bool = True,
    ) -> Trial:
        """Record one trial, stamping the multiplicity as it stands now.

        `emit_event` exists for one legitimate case: a thousand-spec calibration
        run would otherwise put a thousand events into the hash chain to say
        something the search summary already says. The row is always written —
        the count is what the gate divides by, and it is never optional. Only
        the per-trial *event* is suppressible, and the search summary event
        still records the totals.
        """
        moment = at or now_utc()
        if outcome is TrialOutcome.REJECTED and not rejection_reason:
            raise TrialError(
                "a rejected trial must carry a reason. 'Rejected' with no reason is a "
                "row that counts toward multiplicity and explains nothing, and the "
                "rejection reasons are how a searcher learns which part of the space "
                "is not worth sampling."
            )

        # Counts *before* this row, plus one for this row: the multiplicity this
        # candidate was drawn against includes itself.
        in_lineage = self.count_in_lineage(lineage_id) + 1
        in_search = self.count_in_search(search_id) + 1

        trial = Trial(
            trial_id=new_id("trial", length=12),
            search_id=search_id,
            lineage_id=lineage_id,
            spec_hash=spec_hash,
            author_kind=author_kind,
            outcome=outcome,
            recorded_at=moment,
            strategy_id=strategy_id,
            strategy_version=strategy_version,
            parent_strategy_id=parent_strategy_id,
            generation=generation,
            rejection_reason=rejection_reason,
            backtest_id=backtest_id,
            vintage_id=vintage_id,
            net_sharpe=net_sharpe,
            net_return_pct=net_return_pct,
            max_drawdown_pct=max_drawdown_pct,
            n_trades=n_trades,
            cost_drag_bps=cost_drag_bps,
            returns=tuple(float(r) for r in returns[-MAX_STORED_RETURNS:]),
            trials_in_lineage_at_time=in_lineage,
            trials_in_search_at_time=in_search,
        )

        with self._ledger.transaction() as tx:
            seq: int | None = None
            if emit_event:
                seq = tx.append(
                    EventType.TRIAL_RECORDED,
                    trial.trial_id,
                    TrialPayload(
                        trial_id=trial.trial_id,
                        search_id=search_id,
                        lineage_id=lineage_id,
                        spec_hash=spec_hash,
                        author_kind=author_kind.value,
                        outcome=outcome.value,
                        strategy_id=strategy_id,
                        strategy_version=strategy_version,
                        parent_strategy_id=parent_strategy_id,
                        generation=generation,
                        rejection_reason=rejection_reason or None,
                        backtest_id=backtest_id,
                        vintage_id=vintage_id,
                        net_sharpe=net_sharpe,
                        n_trades=n_trades,
                        trials_in_lineage_at_time=in_lineage,
                        trials_in_search_at_time=in_search,
                    ),
                    actor=Actor.SEARCH,
                    run_id=self._run_id,
                ).seq
            tx.execute(
                """
                INSERT INTO trials (
                    trial_id, search_id, lineage_id, strategy_id, strategy_version,
                    spec_hash, author_kind, parent_strategy_id, generation, outcome,
                    rejection_reason, backtest_id, vintage_id, net_sharpe,
                    net_return_pct, max_drawdown_pct, n_trades, cost_drag_bps,
                    returns_json, trials_in_lineage_at_time, trials_in_search_at_time,
                    recorded_at, recording_event_seq
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    trial.trial_id,
                    search_id,
                    lineage_id,
                    strategy_id,
                    strategy_version,
                    spec_hash,
                    author_kind.value,
                    parent_strategy_id,
                    generation,
                    outcome.value,
                    rejection_reason or None,
                    backtest_id,
                    vintage_id,
                    net_sharpe,
                    net_return_pct,
                    max_drawdown_pct,
                    n_trades,
                    cost_drag_bps,
                    json.dumps(list(trial.returns)) if trial.returns else None,
                    in_lineage,
                    in_search,
                    to_iso(moment),
                    seq,
                ),
            )
        return trial

    def complete_search(
        self,
        search_id: str,
        *,
        duration_seconds: float | None = None,
        detail: str = "",
    ) -> SearchSummary:
        """Close a search and record its totals.

        Derived from the rows rather than from counters the caller kept, so the
        summary cannot disagree with the trials it summarises.
        """
        summary = self.summarise_search(search_id)
        self._ledger.append(
            EventType.SEARCH_COMPLETED,
            search_id,
            SearchCompletedPayload(
                search_id=search_id,
                lineage_ids=list(summary.lineage_ids),
                n_proposed=summary.n_proposed,
                n_evaluated=summary.n_evaluated,
                n_rejected=summary.n_rejected,
                n_errored=summary.n_errored,
                n_passed_gate=summary.n_passed_gate,
                duration_seconds=duration_seconds,
                detail=detail,
            ),
            actor=Actor.SEARCH,
            run_id=self._run_id,
        )
        return summary

    # -- counting ----------------------------------------------------------

    def count_in_lineage(self, lineage_id: str, *, before: datetime | None = None) -> int:
        sql = "SELECT COUNT(*) FROM trials WHERE lineage_id = ?"
        params: list[object] = [lineage_id]
        if before is not None:
            sql += " AND recorded_at <= ?"
            params.append(to_iso(before))
        row = self._ledger.conn.execute(sql, params).fetchone()
        return 0 if row is None else int(row[0])

    def count_in_search(self, search_id: str, *, before: datetime | None = None) -> int:
        sql = "SELECT COUNT(*) FROM trials WHERE search_id = ?"
        params: list[object] = [search_id]
        if before is not None:
            sql += " AND recorded_at <= ?"
            params.append(to_iso(before))
        row = self._ledger.conn.execute(sql, params).fetchone()
        return 0 if row is None else int(row[0])

    def summarise_search(self, search_id: str) -> SearchSummary:
        rows = self._ledger.conn.execute(
            "SELECT outcome, lineage_id FROM trials WHERE search_id = ?", (search_id,)
        ).fetchall()
        counts: dict[str, int] = {}
        lineages: set[str] = set()
        for row in rows:
            counts[str(row["outcome"])] = counts.get(str(row["outcome"]), 0) + 1
            lineages.add(str(row["lineage_id"]))
        return SearchSummary(
            search_id=search_id,
            n_proposed=len(rows),
            n_evaluated=counts.get(TrialOutcome.EVALUATED.value, 0)
            + counts.get(TrialOutcome.PASSED_GATE.value, 0),
            n_rejected=counts.get(TrialOutcome.REJECTED.value, 0),
            n_errored=counts.get(TrialOutcome.ERRORED.value, 0),
            n_passed_gate=counts.get(TrialOutcome.PASSED_GATE.value, 0),
            lineage_ids=tuple(sorted(lineages)),
        )

    # -- reading -----------------------------------------------------------

    def trials_in_lineage(self, lineage_id: str) -> list[Trial]:
        rows = self._ledger.conn.execute(
            "SELECT * FROM trials WHERE lineage_id = ? ORDER BY recorded_at", (lineage_id,)
        ).fetchall()
        return [_row_to_trial(row) for row in rows]

    def trials_in_search(self, search_id: str) -> list[Trial]:
        rows = self._ledger.conn.execute(
            "SELECT * FROM trials WHERE search_id = ? ORDER BY recorded_at", (search_id,)
        ).fetchall()
        return [_row_to_trial(row) for row in rows]

    def latest_for_spec(self, spec_hash: str) -> Trial | None:
        row: sqlite3.Row | None = self._ledger.conn.execute(
            "SELECT * FROM trials WHERE spec_hash = ? ORDER BY recorded_at DESC LIMIT 1",
            (spec_hash,),
        ).fetchone()
        return None if row is None else _row_to_trial(row)

    def multiplicity_for(self, spec_hash: str) -> Multiplicity | None:
        """The multiplicity the gate should deflate this candidate against.

        Read from the trial's own stamped counts, not recomputed. The candidate
        was selected out of the search as it stood when it ran, and a count
        taken later describes a different search.
        """
        trial = self.latest_for_spec(spec_hash)
        if trial is None:
            return None
        peers = self.trials_in_lineage(trial.lineage_id)
        if trial.trials_in_search_at_time > trial.trials_in_lineage_at_time:
            peers = self.trials_in_search(trial.search_id)
        return multiplicity_of(trial, peers)


# --------------------------------------------------------------------------
# Pure helpers
# --------------------------------------------------------------------------

# Used when the trial Sharpes give no usable dispersion — fewer than two
# measurable trials, or all of them identical. Not zero: a dispersion of zero
# makes the expected maximum of N draws equal to zero too, so the haircut
# vanishes and the deflated Sharpe becomes the raw one. That is the permissive
# direction, and the whole point of the haircut is to be conservative where the
# evidence is thin. 1.0 is roughly the dispersion of annualised Sharpe across
# random specs on daily bars over a few years, which is the situation this
# fallback covers.
FALLBACK_SHARPE_DISPERSION = 1.0


def multiplicity_of(trial: Trial, peers: Sequence[Trial]) -> Multiplicity:
    """Assemble the multiplicity for one trial from its peers.

    Pure, so the property that matters — that a thin or degenerate peer set
    produces a conservative dispersion rather than none — is testable without a
    database.
    """
    sharpes = [t.net_sharpe for t in peers if t.is_measurable and t.net_sharpe is not None]
    dispersion = _stdev(sharpes)
    measured = dispersion is not None and dispersion > 0
    return Multiplicity(
        n_trials=trial.n_trials_for_deflation,
        n_lineage_trials=trial.trials_in_lineage_at_time,
        n_search_trials=trial.trials_in_search_at_time,
        sharpe_dispersion=(
            dispersion if measured and dispersion is not None else FALLBACK_SHARPE_DISPERSION
        ),
        dispersion_measured=measured,
        n_measurable=len(sharpes),
    )


def returns_matrix(
    trials: Sequence[Trial],
    *,
    max_trials: int = MAX_PBO_TRIALS,
    seed: int = 0,
) -> tuple[tuple[float, ...], ...]:
    """The trials' return series as period rows, ready for cross-validation.

    Rows are periods and each row holds one value per trial, which is the
    orientation PBO wants: it partitions *time* and compares across trials.

    **Truncated to the shortest series, never padded.** Padding with zeros
    would invent flat periods, and a flat period lowers dispersion and raises
    every Sharpe in the matrix.

    **Capped at `max_trials` columns, by a seeded random sample.** CSCV costs
    `combinations x columns x periods` in Python, so a thousand-trial search
    would take minutes — in the real pipeline as much as in a test. The sample
    is random rather than the top-N by Sharpe: taking the best would change
    what the statistic measures, since CSCV asks whether the in-sample winner
    of a candidate set holds up, and a set containing only winners has a
    different answer. A smaller sample makes the estimate noisier without
    moving its expectation.
    """
    series = [t.returns for t in trials if t.returns]
    if not series:
        return ()
    if len(series) > max_trials:
        series = random.Random(seed).sample(series, max_trials)  # noqa: S311
    length = min(len(s) for s in series)
    if length == 0:
        return ()
    columns = [s[-length:] for s in series]
    return tuple(tuple(column[row] for column in columns) for row in range(length))


def _stdev(values: Sequence[float]) -> float | None:
    """Sample standard deviation, or `None` when there is nothing to measure."""
    if len(values) < 2:
        return None
    mean = sum(values) / len(values)
    variance = sum((v - mean) ** 2 for v in values) / (len(values) - 1)
    return sqrt(variance)


def _row_to_trial(row: sqlite3.Row) -> Trial:
    raw_returns = row["returns_json"]
    return Trial(
        trial_id=str(row["trial_id"]),
        search_id=str(row["search_id"]),
        lineage_id=str(row["lineage_id"]),
        spec_hash=str(row["spec_hash"]),
        author_kind=AuthorKind(str(row["author_kind"])),
        outcome=TrialOutcome(str(row["outcome"])),
        recorded_at=from_iso(str(row["recorded_at"])),
        strategy_id=None if row["strategy_id"] is None else str(row["strategy_id"]),
        strategy_version=(
            None if row["strategy_version"] is None else int(row["strategy_version"])
        ),
        parent_strategy_id=(
            None if row["parent_strategy_id"] is None else str(row["parent_strategy_id"])
        ),
        generation=int(row["generation"]),
        rejection_reason="" if row["rejection_reason"] is None else str(row["rejection_reason"]),
        backtest_id=None if row["backtest_id"] is None else str(row["backtest_id"]),
        vintage_id=None if row["vintage_id"] is None else str(row["vintage_id"]),
        net_sharpe=None if row["net_sharpe"] is None else float(row["net_sharpe"]),
        net_return_pct=None if row["net_return_pct"] is None else float(row["net_return_pct"]),
        max_drawdown_pct=(
            None if row["max_drawdown_pct"] is None else float(row["max_drawdown_pct"])
        ),
        n_trades=None if row["n_trades"] is None else int(row["n_trades"]),
        cost_drag_bps=None if row["cost_drag_bps"] is None else float(row["cost_drag_bps"]),
        returns=() if raw_returns is None else tuple(json.loads(str(raw_returns))),
        trials_in_lineage_at_time=int(row["trials_in_lineage_at_time"]),
        trials_in_search_at_time=int(row["trials_in_search_at_time"]),
    )


@dataclass(slots=True)
class SearchSession:
    """A convenience wrapper that keeps one `search_id` across many records.

    Exists because the alternative is a caller threading `search_id` through
    every call site, and the first one that forgets creates a search of size
    one — which takes no multiplicity haircut. Making the id hard to omit is
    worth a small class.
    """

    log: TrialLog
    search_id: str = field(default_factory=lambda: new_id("srch", length=10))

    def record(self, **kwargs: object) -> Trial:
        if "search_id" in kwargs:
            raise TrialError(
                "a SearchSession owns its search_id; passing another one would split "
                "one search into two and halve the multiplicity haircut on each half"
            )
        return self.log.record(search_id=self.search_id, **kwargs)  # type: ignore[arg-type]

    def complete(self, **kwargs: object) -> SearchSummary:
        return self.log.complete_search(self.search_id, **kwargs)  # type: ignore[arg-type]
