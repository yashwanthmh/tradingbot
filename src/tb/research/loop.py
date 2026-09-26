"""One search cycle, end to end, with the holdout untouched.

The I/O half of M6. `research.searcher` is pure; this module gives it data it may
see, records everything it did, and registers what survived:

    sealed vintage -> holdout boundary -> training reader
      -> search (propose, validate, backtest, select)
      -> every candidate recorded as a trial
      -> survivors registered as candidates
      -> search closed, totals in one event

Nothing here promotes anything, and nothing here evaluates the holdout. Both are
separate steps with their own commands — `tb research holdout` spends the one
evaluation, `tb promote evaluate` runs the gate — because a cycle that did either
would be a process that both chooses a strategy and checks the choice, which is
the arrangement the sealed holdout exists to prevent.

Four things are load-bearing.

**The evaluator gets a `training_reader`, built here, and nothing else.** The
searcher takes its evaluator by injection precisely so that the one place a
reader is constructed is a place that knows where the seal is. Both defences of
`research.holdout` apply: the source holds no bar past the boundary, and the
reader refuses to advance to it. The training schedule is split on the decision
time itself, so the last training decision is strictly before the seal.

**Every candidate is a trial, including the refused and the broken.** The
denominator of every deflated metric is a count of draws, and a cycle that
recorded only what it evaluated would report a search of fifty where a hundred
happened. The rows are written whether or not anything is registered — so a dry
run still counts, because the search it ran happened. A human who dry-ran ten
times and hand-registered the best result would otherwise have run a search whose
size was never recorded.

**A batch selects after every candidate has run, so every candidate is stamped
with the whole batch.** `TrialLog` stamps the running count by default, which is
right for a search that decides as it goes. Here the survivors are chosen from
all N, and stamping the tenth-recorded one with ten would deflate it against a
fifth of the search that produced it. `selected_from_search` and
`selected_from_lineage` carry the batch totals; they are floors, never overrides.

**Lineage is inherited, and registered lineage wins.** A mutation joins its
parent's lineage; a fresh draw starts one. But a spec the registry already holds
keeps the lineage it was registered under — otherwise a random draw that happened
to rediscover a registered strategy would record its trials against a lineage
nobody reads, and that strategy's lineage count, which is what its haircut is
computed from, would stop growing.

**A proposer is told the search's shape and nothing dated.** The first
generation comes from a random draw unless a `ProposerFactory` is passed, and
the factory receives a `ProposalContext`: the bounds, the budget, and the
market's state as one of three words, read through the sealed source at the
last training decision. No bar, no price and no date crosses into it, so a
proposer backed by a model that has read market history has nothing to
recognise the period by.
"""

from __future__ import annotations

import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime, timedelta

from tb.backtest.costs import CostModel, Jurisdiction
from tb.backtest.engine import Backtester, BacktestResult, InstrumentMeta
from tb.backtest.metrics import returns_of
from tb.config.hard_limits import HardLimits
from tb.core.clock import now_utc
from tb.core.errors import TbError
from tb.core.ids import new_id
from tb.data.adjustments import CorporateAction
from tb.data.asof import BarSource, HoldoutViolation
from tb.data.provider import Resolution
from tb.data.regime import RegimeGate
from tb.data.snapshot import SnapshotStore
from tb.ledger.events import Actor, EventType, SpecsProposedPayload
from tb.ledger.store import Ledger
from tb.registry.lineage import SpecRegistry, strategy_id_for
from tb.registry.models import AuthorKind, RegisteredSpec, TrialOutcome
from tb.research.holdout import (
    DEFAULT_HOLDOUT_FRACTION,
    HoldoutRegistry,
    HoldoutWindow,
    SealedBarSource,
    decisions_between,
    holdout_boundary,
    training_reader,
)
from tb.research.llm.adapter import LLMProposer
from tb.research.llm.prompts import RegimeDescription
from tb.research.mutate import (
    MutationProposer,
    ProposalBounds,
    RandomProposer,
    SpecProposer,
)
from tb.research.searcher import (
    Candidate,
    SearchBudget,
    Searcher,
    SearchOutcome,
    required_sharpe,
)
from tb.research.trials import TrialLog
from tb.research.validate import HISTORY_MARGIN_BARS, SpecValidator
from tb.strategy.dsl.ops import DslStrategy, pipeline_from_spec
from tb.strategy.dsl.schema import StrategySpec

# Minutes in a daily bar, and therefore the shortest hold a daily strategy can
# express. The cycle proposes daily specs because daily is the only resolution
# the bot may trade live (see docs/decisions/0001-minute-resolution.md); a search
# breeding minute-scale specs would be breeding a family the gate refuses whole.
DAILY_HOLD_MINUTES = 1440


class CycleError(TbError):
    """A search cycle could not be run."""


@dataclass(frozen=True, slots=True)
class ProposalContext:
    """Everything a proposer factory may know about the search it proposes for.

    Short and abstract on purpose. The bounds come from the limits and the
    training window's length; the budget and the Sharpe it implies come from
    the caller's numbers alone; the regime is one of three words. None of it is
    a bar, a price or a date, which is what lets a model-backed proposer be
    handed all of it.
    """

    bounds: ProposalBounds
    regime: RegimeDescription
    n_trials: int
    required_sharpe: float


ProposerFactory = Callable[[ProposalContext], SpecProposer]


@dataclass(frozen=True, slots=True)
class CycleReport:
    """What one cycle did, in the shape an operator reads."""

    search_id: str
    vintage_id: str
    window: HoldoutWindow
    outcome: SearchOutcome
    registered: tuple[RegisteredSpec, ...]
    n_trials_recorded: int
    dry_run: bool
    duration_seconds: float
    lineage_of: Mapping[str, str]
    proposer: str = "random"
    proposer_notes: tuple[str, ...] = ()

    def explain(self) -> str:
        lines = [self.window.summary()]
        lines.extend(f"{self.proposer}: {note}" for note in self.proposer_notes)
        lines.append(self.outcome.explain())
        lines.append(f"{self.n_trials_recorded} trial(s) recorded under {self.search_id}")
        if self.dry_run:
            lines.append(
                "dry run: nothing registered. The trials are recorded anyway — the search "
                "happened, and a search whose size went unrecorded would let its best result "
                "be registered by hand with no haircut at all"
            )
        elif self.registered:
            labels = ", ".join(spec.strategy_id for spec in self.registered)
            lines.append(
                f"registered {len(self.registered)} survivor(s) as candidates: {labels}. "
                "Each gets one holdout evaluation (`tb research holdout`) and then the "
                "gate (`tb promote evaluate`); registration promotes nothing"
            )
        else:
            lines.append(
                "nothing survived to register: no candidate produced a Sharpe over enough "
                "trades to rank"
            )
        return "\n".join(lines)


class ResearchCycle:
    """Runs one search over a sealed vintage and records all of it."""

    def __init__(
        self,
        ledger: Ledger,
        *,
        limits: HardLimits,
        snapshots: SnapshotStore,
        run_id: str | None = None,
        clock: Callable[[], datetime] = now_utc,
    ) -> None:
        self._ledger = ledger
        self._limits = limits
        self._snapshots = snapshots
        self._run_id = run_id
        self._clock = clock
        self._registry = SpecRegistry(
            ledger,
            per_lineage_budget_ccy=limits.loss.per_lineage_budget_ccy,
            run_id=run_id,
        )
        self._trials = TrialLog(ledger, run_id=run_id)

    def run(
        self,
        *,
        vintage_id: str,
        budget: SearchBudget,
        fraction: float = DEFAULT_HOLDOUT_FRACTION,
        register: bool = True,
        seed_strategy_ids: Sequence[str] = (),
        search_id: str | None = None,
        proposer: ProposerFactory | None = None,
    ) -> CycleReport:
        """Search, record every trial, register the survivors unless dry-running.

        `proposer` builds the first generation's source; a random draw when it
        is `None`. It cannot be combined with seeds: a seeded search starts by
        mutating the seeds, so a proposer passed alongside them would never be
        asked, and a search that silently ignored the one it was given would be
        recorded as something it was not.
        """
        started = time.monotonic()
        moment = self._clock()
        search = search_id or new_id("srch", length=12)

        window, source, uids, schedule, n_training_bars = self._training_data(
            vintage_id, fraction=fraction
        )
        seeds = self._seeds(seed_strategy_ids)
        if proposer is not None and seeds:
            raise CycleError(
                "a seeded search starts by mutating its seeds, so a proposer passed with "
                "them would never be asked. Refine registered strategies with `--from`, or "
                "start fresh ideas from a proposer — one search, one source."
            )

        costs = CostModel(self._limits)
        meta = {uid: InstrumentMeta(uid, "USD", Jurisdiction.US) for uid in uids}
        # The actions the vintage was sealed with: every candidate is judged on
        # the same adjusted history, and a re-run of the search sees it again.
        actions = self._snapshots.actions_of(vintage_id)

        def evaluate(spec: StrategySpec) -> BacktestResult:
            # A fresh reader per spec: the reader is forward-only and stateful,
            # so sharing one would hand the second spec a reader already
            # advanced to the end of the window.
            engine = Backtester(
                cost_model=costs,
                pipeline=pipeline_from_spec(spec),
                instruments=meta,
                min_holding_minutes=spec.min_holding_minutes,
                actions=actions,
            )
            return engine.run(
                strategy=DslStrategy(spec=spec, strategy_id=strategy_id_for(spec.spec_hash)),
                reader=training_reader(
                    source,
                    sealed_from=window.sealed_from,
                    resolution=Resolution.DAILY,
                    instrument_uids=uids,
                    lookback=_lookback_for(spec),
                ),
                decision_times=schedule,
                resolution=Resolution.DAILY,
            )

        bounds = ProposalBounds.from_limits(
            self._limits,
            max_lookback=max(1, n_training_bars - HISTORY_MARGIN_BARS),
            min_holding_minutes=DAILY_HOLD_MINUTES,
        )
        min_deflated = self._limits.promotion.min_oos_deflated_sharpe
        initial: SpecProposer = RandomProposer(bounds=bounds)
        try:
            if proposer is not None:
                initial = proposer(
                    ProposalContext(
                        bounds=bounds,
                        regime=training_regime(
                            self._limits,
                            source,
                            sealed_from=window.sealed_from,
                            as_of=schedule[-1],
                            actions=actions.get(RegimeGate(self._limits).instrument_uid, ()),
                        ),
                        n_trials=budget.n_trials,
                        required_sharpe=required_sharpe(
                            n_trials=budget.n_trials, min_deflated_sharpe=min_deflated
                        ),
                    )
                )
            searcher = Searcher(
                evaluate=evaluate,
                validator=SpecValidator(limits=self._limits, n_training_bars=n_training_bars),
                initial_proposer=initial,
                mutation_proposer=MutationProposer(bounds=bounds),
                budget=budget,
                min_deflated_sharpe=min_deflated,
                search_id=search,
                seed_parents=tuple(spec for _, spec in seeds),
            )
            outcome = searcher.run()
        except HoldoutViolation as exc:
            raise self._violation(exc, search_id=search, window=window) from exc

        # The exchange first, then the trials it produced — the order in which
        # they happened, and the order a reader of the log expects.
        notes = self._record_exchanges(initial, search_id=search)
        lineage_of = self._lineages(outcome.candidates, seeds=seeds)
        registered = self._register(outcome.survivors, lineage_of=lineage_of) if register else ()
        n_recorded = self._record(
            outcome.candidates,
            search_id=search,
            vintage_id=vintage_id,
            lineage_of=lineage_of,
            at=moment,
        )
        duration = time.monotonic() - started
        self._trials.complete_search(
            search,
            duration_seconds=duration,
            detail=(
                f"vintage {vintage_id}, {outcome.generations} generation(s), "
                f"{'dry run' if not register else f'{len(registered)} registered'}; a "
                f"search this size needs an out-of-sample Sharpe near "
                f"{outcome.required_sharpe:.2f}"
            ),
        )
        return CycleReport(
            search_id=search,
            vintage_id=vintage_id,
            window=window,
            outcome=outcome,
            registered=registered,
            n_trials_recorded=n_recorded,
            dry_run=not register,
            duration_seconds=duration,
            lineage_of=lineage_of,
            proposer="mutation" if seeds else initial.name,
            proposer_notes=notes,
        )

    def _violation(
        self, exc: HoldoutViolation, *, search_id: str, window: HoldoutWindow
    ) -> CycleError:
        """Record a read past the seal, and the error that ends the search.

        Fatal, as `HoldoutViolation` promises: every result this search produced
        came from machinery that has just reached past its own boundary, so
        nothing is registered and no trial is recorded — a re-run once the cause
        is fixed records its own. Recorded, because a raise stops one process
        and the event is what makes a pattern of attempts visible. Before this
        the searcher filed the violation as one more errored trial and the
        cycle went on to register survivors, and `record_violation` had no
        caller at all.
        """
        HoldoutRegistry(self._ledger, run_id=self._run_id).record_violation(
            sealed_from=exc.sealed_from or window.sealed_from,
            requested_at=exc.requested_at or window.sealed_from,
            caller=f"research cycle {search_id}",
            detail=str(exc),
        )
        return CycleError(
            f"search {search_id} stopped: it reached past the holdout boundary "
            f"{window.sealed_from.isoformat()}, so nothing it produced can be trusted. "
            "Nothing was registered and no trial recorded; the attempt is in the ledger "
            f"as {EventType.HOLDOUT_VIOLATION_ATTEMPTED.value}. {exc}"
        )

    # -- what a proposer told us -----------------------------------------------

    def _record_exchanges(self, initial: SpecProposer, *, search_id: str) -> tuple[str, ...]:
        """Ledger each model call, and return what the operator should read about it.

        Only a model-backed proposer has anything to record: the random and
        mutation proposers replay from the seed, so their exchanges are
        reconstructible and recording them would be noise.
        """
        if not isinstance(initial, LLMProposer):
            return ()
        lines: list[str] = []
        for report in initial.reports:
            self._ledger.append(
                EventType.SPECS_PROPOSED,
                search_id,
                SpecsProposedPayload(
                    search_id=search_id,
                    proposer=initial.name,
                    requested_model=report.requested_model,
                    served_model=report.served_model,
                    fell_back=report.fell_back,
                    stop_reason=report.stop_reason,
                    n_requested=report.n_requested,
                    n_items=report.n_items,
                    n_accepted=len(report.accepted),
                    n_refused=report.n_refused,
                    n_duplicates=report.n_duplicates,
                    refused=list(report.refused),
                    ignored_keys=list(report.ignored_keys),
                    stopped=report.stopped,
                    system_prompt=report.system_prompt,
                    user_prompt=report.user_prompt,
                    reply_sha256=report.reply_sha256,
                    reply_chars=report.reply_chars,
                    specs=[proposal.spec.model_dump(mode="json") for proposal in report.accepted],
                ),
                actor=Actor.LLM,
                run_id=self._run_id,
            )
            lines.extend(report.lines())
        return tuple(lines)

    # -- the data the search may see ----------------------------------------

    def _training_data(
        self, vintage_id: str, *, fraction: float
    ) -> tuple[HoldoutWindow, BarSource, tuple[str, ...], list[datetime], int]:
        """The sealed source, the boundary, and the schedule strictly before it."""
        admissible, why = self._snapshots.is_admissible(vintage_id)
        if not admissible:
            raise CycleError(why)
        vintage = self._snapshots.get(vintage_id)
        if vintage is None:  # pragma: no cover - is_admissible just found it
            raise CycleError(f"{vintage_id} vanished between checks")

        window = holdout_boundary(vintage, fraction=fraction)
        if not window.is_usable:
            raise CycleError(
                f"{window.summary()}. The holdout is too short to judge anything this "
                "search produces, so running the search would spend trials on candidates "
                "that can never be evaluated."
            )

        bars = self._snapshots.bars_of(vintage_id)
        schedule = decisions_between(bars, end=window.sealed_from)
        if len(schedule) < 2:
            raise CycleError(
                f"the training window of {vintage_id} holds {len(schedule)} decision "
                "time(s); a backtest needs at least two, since a fill comes from the bar "
                "after the decision"
            )
        uids = tuple(vintage.instrument_uids)
        # The shortest instrument's history is the binding one: a lookback that
        # fits the longest series but not the shortest evaluates to UNKNOWN on
        # the shortest at every decision.
        per_uid = {uid: 0 for uid in uids}
        for bar in bars:
            if bar.available_at_utc < window.sealed_from and bar.instrument_uid in per_uid:
                per_uid[bar.instrument_uid] += 1
        n_training_bars = min(per_uid.values()) if per_uid else 0
        return window, self._snapshots.source_for(vintage_id), uids, schedule, n_training_bars

    def _seeds(self, strategy_ids: Sequence[str]) -> list[tuple[RegisteredSpec, StrategySpec]]:
        out: list[tuple[RegisteredSpec, StrategySpec]] = []
        for strategy_id in strategy_ids:
            registered = self._registry.latest_version_of(strategy_id)
            spec = (
                None
                if registered is None
                else self._registry.spec_of(strategy_id, registered.version)
            )
            if registered is None or spec is None:
                raise CycleError(
                    f"seed {strategy_id} is not registered. A seed is how a search refines "
                    "an earlier result, and its lineage is what carries that result's trial "
                    "count forward — an unregistered seed has neither."
                )
            out.append((registered, spec))
        return out

    # -- lineage -------------------------------------------------------------

    def _lineages(
        self,
        candidates: Sequence[Candidate],
        *,
        seeds: Sequence[tuple[RegisteredSpec, StrategySpec]],
    ) -> dict[str, str]:
        """Which lineage each candidate belongs to.

        Candidates arrive in generation order, so a parent is always resolved
        before its children. Three sources, in order of authority: the registry
        (a spec it holds keeps its lineage), the parent (a mutation joins it),
        and the spec itself (a fresh draw starts a lineage named for its hash,
        so the same idea drawn in two searches is one lineage rather than two).
        """
        lineage_of: dict[str, str] = {
            registered.spec_hash: registered.lineage_id for registered, _ in seeds
        }
        for candidate in candidates:
            spec_hash = candidate.spec_hash
            if spec_hash in lineage_of:
                continue
            known = self._registry.by_hash(spec_hash)
            if known is not None:
                lineage_of[spec_hash] = known.lineage_id
                continue
            parent = candidate.proposal.parent_spec_hash
            if parent is not None and parent in lineage_of:
                lineage_of[spec_hash] = lineage_of[parent]
                continue
            lineage_of[spec_hash] = f"lin_{spec_hash[:12]}"
        return lineage_of

    # -- registering the survivors -------------------------------------------

    def _register(
        self, survivors: Sequence[Candidate], *, lineage_of: Mapping[str, str]
    ) -> tuple[RegisteredSpec, ...]:
        """Register what survived, as candidates.

        Named after its parent when the parent is registered, so the registry
        derives and checks the lineage itself — a mutation cannot choose its own,
        which is what stops an exhausted budget being left behind. When the parent
        was only ever an in-search candidate, the lineage the search assigned is
        passed explicitly; it is the parent's, by construction.
        """
        out: list[RegisteredSpec] = []
        for candidate in survivors:
            parent_hash = candidate.proposal.parent_spec_hash
            parent = None if parent_hash is None else self._registry.by_hash(parent_hash)
            author = candidate.proposal.author_kind
            if parent is not None:
                registered = self._registry.register(
                    candidate.spec,
                    author_kind=author,
                    parent_strategy_id=parent.strategy_id,
                    at=self._clock(),
                )
            else:
                registered = self._registry.register(
                    candidate.spec,
                    author_kind=author if author is not AuthorKind.MUTATION else AuthorKind.SEARCH,
                    lineage_id=lineage_of[candidate.spec_hash],
                    at=self._clock(),
                )
            out.append(registered)
        return tuple(out)

    # -- recording every trial -------------------------------------------------

    def _record(
        self,
        candidates: Sequence[Candidate],
        *,
        search_id: str,
        vintage_id: str,
        lineage_of: Mapping[str, str],
        at: datetime,
    ) -> int:
        """Write every candidate to the trial log, stamped with the whole batch."""
        per_lineage: dict[str, int] = {}
        for candidate in candidates:
            lineage = lineage_of[candidate.spec_hash]
            per_lineage[lineage] = per_lineage.get(lineage, 0) + 1
        # Prior counts first, so the floor includes the lineage's history across
        # earlier searches as well as this batch — that accumulation is what
        # stops "many small searches" of one idea from each taking a small
        # haircut.
        lineage_totals = {
            lineage: self._trials.count_in_lineage(lineage) + count
            for lineage, count in per_lineage.items()
        }
        batch_total = self._trials.count_in_search(search_id) + len(candidates)

        for candidate in candidates:
            registered = self._registry.by_hash(candidate.spec_hash)
            parent_hash = candidate.proposal.parent_spec_hash
            parent = None if parent_hash is None else self._registry.by_hash(parent_hash)
            result = candidate.result
            reason = ""
            if candidate.rejection is not None:
                reason = candidate.rejection.describe()
            elif candidate.outcome is TrialOutcome.ERRORED:
                reason = candidate.error or "the evaluator raised"
            self._trials.record(
                search_id=search_id,
                lineage_id=lineage_of[candidate.spec_hash],
                spec_hash=candidate.spec_hash,
                author_kind=candidate.proposal.author_kind,
                outcome=candidate.outcome,
                strategy_id=None if registered is None else registered.strategy_id,
                strategy_version=None if registered is None else registered.version,
                parent_strategy_id=None if parent is None else parent.strategy_id,
                generation=candidate.generation,
                rejection_reason=reason,
                backtest_id=None if result is None else result.backtest_id,
                vintage_id=vintage_id,
                net_sharpe=None if result is None else result.metrics.net_sharpe,
                net_return_pct=None if result is None else float(result.metrics.net_return_pct),
                max_drawdown_pct=(
                    None if result is None else float(result.metrics.max_drawdown_pct)
                ),
                n_trades=None if result is None else result.metrics.n_trades,
                cost_drag_bps=None if result is None else float(result.metrics.cost_drag_bps),
                returns=(
                    ()
                    if result is None
                    else [float(v) for v in returns_of([p.equity_ccy for p in result.curve])]
                ),
                at=at,
                selected_from_search=batch_total,
                selected_from_lineage=lineage_totals[lineage_of[candidate.spec_hash]],
            )
        return len(candidates)


def training_regime(
    limits: HardLimits,
    source: BarSource,
    *,
    sealed_from: datetime,
    as_of: datetime,
    actions: Sequence[CorporateAction] = (),
) -> RegimeDescription:
    """The market's state at a training instant, reduced to one word.

    Two defences, the same pair the training reader has. An instant at or past
    the boundary raises: it is a caller's mistake, and describing "the market
    now" from a point inside the holdout is exactly the leak the seal exists to
    prevent. And the bars are read through a `SealedBarSource`, so even a
    correct instant sees nothing from after the boundary if a later edit to
    `visible_bars` widens what it asks its source for.

    A vintage without the reference series reads as `unknown` — the regime
    gate's own fail-closed answer — rather than as a guess.
    """
    if as_of >= sealed_from:
        raise HoldoutViolation(
            f"a regime reading for a proposer was asked for at {as_of.isoformat()}, at or "
            f"past the holdout boundary {sealed_from.isoformat()}. What a proposer is told "
            "must come from the training window like everything else the search sees.",
            sealed_from=sealed_from,
            requested_at=as_of,
        )
    reading = RegimeGate(limits).read(
        SealedBarSource(inner=source, sealed_from=sealed_from),
        as_of=as_of,
        actions=actions,
    )
    return RegimeDescription.from_reading(reading)


def _lookback_for(spec: StrategySpec) -> timedelta:
    """The calendar window a spec's longest feature needs.

    Bars are sessions and the window is calendar days: about 1.45 calendar days
    per trading session, plus holidays. `1.6x + 30` covers that with room, and a
    window larger than needed changes no feature value — each feature reads only
    its own last `lookback` closes — so erring long costs time, not correctness.
    """
    return timedelta(days=int(spec.max_lookback * 1.6) + 30)
