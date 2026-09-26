"""Refusing a proposal before it costs a backtest — and counting the refusal.

Two jobs, and the second is the one that is easy to skip. The obvious one is
economy: a spec declaring 40bps of edge on a venue whose round trip is 40bps
will be refused by the promotion gate whatever its backtest says, so running the
backtest is a few hundred milliseconds spent to learn something already known.
At a thousand proposals a cycle that is the difference between a search that
finishes and one that does not.

The second job is that **a rejection is still a trial**. It was a draw from the
search space, and the deflated Sharpe's `N` is a count of draws. A validator
that quietly dropped a proposal would shrink the denominator of every
downstream statistic, which moves the haircut in the permissive direction —
fewer trials means a smaller best-of-N expectation means a noise strategy clears
the gate. So `check` returns a `Rejection` carrying a reason rather than a bare
bool, and the caller records it in the trial log with `TrialOutcome.REJECTED`.

**Every check here mirrors one the gate makes, and none of them replaces it.**
This is a pre-filter over cheap facts about a spec: its declared edge, its
lookback against the history available, its holding period against the
resolutions the bot may trade. The gate's own version still runs on whatever
survives, because a pre-filter that the gate trusted would be a gate that could
be bypassed by changing a pre-filter.

**A duplicate is a rejection, not a silent skip.** Proposing a tree that already
exists is the normal consequence of mutation — invert a mutation and you have
rediscovered the parent — and it costs a trial even though it costs no backtest.
Counting it is what stops a searcher lowering its own multiplicity by proposing
in circles.
"""

from __future__ import annotations

from collections.abc import Container
from dataclasses import dataclass
from decimal import Decimal

from tb.backtest.costs import CostModel, Jurisdiction
from tb.config.hard_limits import HardLimits
from tb.strategy.dsl.schema import StrategySpec

# The notional a pre-trade cost check is quoted at. The same figure the
# promotion gate uses, and for the same reason: cost in bps is mildly
# notional-dependent, so pre-filtering at the floor while the gate quotes at
# £1,000 would pass specs to the gate that it then refuses — and the searcher
# would learn the wrong lesson about which part of the space is affordable.
VALIDATION_NOTIONAL_CCY = Decimal("1000.00")

# Minutes per bar, for the holding-period check. Duplicated from the promotion
# gate's own table deliberately: importing it would make this module depend on
# the gate, and the gate is the thing this one must not be able to weaken.
_RESOLUTION_MINUTES: dict[str, int] = {"minute": 1, "hourly": 60, "daily": 1440}

# How much history a lookback needs before its feature is a number rather than
# `UNKNOWN`. Exactly the lookback — the pipeline refuses a window shorter than
# the declared lookback rather than computing over what happens to be there —
# plus a margin, because a spec whose longest feature is exactly the window
# length produces one evaluable decision and no trades.
HISTORY_MARGIN_BARS = 20


@dataclass(frozen=True, slots=True)
class Rejection:
    """Why a proposal was refused, in the shape the trial log records.

    `code` is a short slug so a cycle can report rejections *by kind* — "600 of
    1,000 refused on cost" is a fact a searcher can act on, where 600 sentences
    are not.
    """

    code: str
    reason: str
    observed: str = ""
    threshold: str = ""

    def describe(self) -> str:
        if not self.observed and not self.threshold:
            return f"{self.code}: {self.reason}"
        return f"{self.code}: {self.reason} (observed {self.observed}, needs {self.threshold})"


@dataclass(frozen=True, slots=True)
class SpecValidator:
    """The cheap checks, in one place, built from the hash-pinned limits.

    `n_training_bars` and `feed_noise_p95_bps` are optional and behave
    differently when absent, which is deliberate:

    * no bar count means the history check is skipped — the caller genuinely may
      not know yet, and a spec is not wrong for having a long lookback;
    * no feed-noise figure means the *gate* will refuse on it later, so
      pre-filtering on it here would refuse the whole population for a reason
      that has nothing to do with the specs. The gate's fail-closed reading
      stays where it is, at the gate.
    """

    limits: HardLimits
    n_training_bars: int | None = None
    feed_noise_p95_bps: Decimal | None = None
    jurisdiction: Jurisdiction = Jurisdiction.US
    instrument_currency: str = "USD"
    # Off for the searcher, on only for the trainer that recorded the model:
    # see `_reads_a_model`.
    allow_models: bool = False

    def check(
        self,
        spec: StrategySpec,
        *,
        seen: Container[str] = frozenset(),
    ) -> Rejection | None:
        """The first reason to refuse this spec, or `None` to evaluate it.

        Returns the *first* reason rather than all of them, unlike the promotion
        gate, which reports every check. The difference is what each is for: the
        gate's verdict is read by a human deciding whether a strategy is fundable
        and "refused by one check at 99% of threshold" matters there, while this
        runs a thousand times a cycle and the caller wants a count by kind.
        """
        for rejection in (
            self._duplicate(spec, seen=seen),
            self._reads_a_model(spec),
            self._reads_a_feature(spec),
            self._edge_band(spec),
            self._holding_period(spec),
            self._history(spec),
            self._cost(spec),
            self._feed_noise(spec),
        ):
            if rejection is not None:
                return rejection
        return None

    # -- the checks --------------------------------------------------------

    @staticmethod
    def _duplicate(spec: StrategySpec, *, seen: Container[str]) -> Rejection | None:
        if spec.spec_hash in seen:
            return Rejection(
                code="duplicate",
                reason=(
                    "this tree has already been proposed. It still counts as a trial — "
                    "rediscovering a parent is what mutation does, and not counting it "
                    "would let a search lower its own multiplicity by going in circles"
                ),
                observed=spec.spec_hash[:12],
            )
        return None

    def _reads_a_model(self, spec: StrategySpec) -> Rejection | None:
        """A searched spec may not read a model; only the trainer's may.

        A model is the product of its own search — every fold, every candidate
        threshold — and its trials are counted where it was trained. A searcher
        that composed models into specs would run a second search on top of the
        first and count only the second, which is the multiplicity leak this
        module exists to close, one level up. It also cannot know which
        artifacts exist, so any model term it proposes was copied or invented:
        crossover grafting one from a parent is the innocent way in, a proposer
        naming one it never trained is the other.
        """
        if self.allow_models or not spec.model_refs:
            return None
        return Rejection(
            code="reads_a_model",
            reason=(
                "only the trainer that recorded a model may propose specs reading it: the "
                "model's trials are counted where it was trained, and a search composing "
                "models would stack a second search on them uncounted"
            ),
            observed=", ".join(ref.model_id for ref in spec.model_refs),
        )

    @staticmethod
    def _reads_a_feature(spec: StrategySpec) -> Rejection | None:
        """An entry rule that reads no feature is a constant, not a strategy.

        The grammar permits `const > const`, which is either always true or
        always false. Always-true is the dangerous one: a spec that enters on
        every bar is a plausible-looking strategy with a turnover the cost gate
        will refuse for the wrong reason, and its backtest would be a
        measurement of the fee schedule.
        """
        if not spec.required_features:
            return Rejection(
                code="no_feature_read",
                reason=(
                    "neither the entry nor the exit rule reads a feature, so the spec is "
                    "a constant. Always-true enters on every bar and always-false never "
                    "trades; both are searcher output rather than strategies"
                ),
            )
        return None

    def _edge_band(self, spec: StrategySpec) -> Rejection | None:
        """The declared edge must sit inside the band the limits permit.

        The ceiling is the one that closes a hole: the cost gate divides the
        round trip by this number, so a spec free to claim 10,000bps passes it
        trivially and the control keeping the search out of the fee trap becomes
        defeatable by the search.
        """
        low = Decimal(str(self.limits.costs.min_expected_edge_bps))
        high = Decimal(str(self.limits.costs.max_expected_edge_bps))
        edge = spec.expected_edge_bps
        if edge < low or edge > high:
            return Rejection(
                code="edge_band",
                reason=(
                    "the declared edge is outside the band the limits permit. The upper "
                    "bound is not tuning: the cost gate divides by this number, so an "
                    "unbounded claim would pass it trivially"
                ),
                observed=f"{edge}bps",
                threshold=f"{low}-{high}bps",
            )
        return None

    def _holding_period(self, spec: StrategySpec) -> Rejection | None:
        """Can this spec's holding period be expressed at a permitted resolution?

        Not a duplicate of the cost check. Minute resolution is refused on
        measured evidence (`docs/decisions/0001-minute-resolution.md`), so a spec
        implying a 30-minute hold is not expensive, it is unexecutable — and
        without this the searcher breeds a whole family the loop can never act
        on and records every one of them as a cost failure.
        """
        allowed = set(self.limits.data.allowed_live_resolutions)
        known = [_RESOLUTION_MINUTES[name] for name in allowed if name in _RESOLUTION_MINUTES]
        # An unrecognised resolution name falls back to the slowest known one
        # rather than being ignored, so a typo in the limits file cannot loosen
        # this check.
        shortest = min(known) if known else max(_RESOLUTION_MINUTES.values())
        required = max(shortest, self.limits.execution.min_holding_minutes)
        if spec.min_holding_minutes < required:
            return Rejection(
                code="holding_period",
                reason=(
                    "the spec's own minimum hold is shorter than one bar at any permitted "
                    "live resolution, or shorter than the risk layer's minimum. Such a "
                    "strategy is unexecutable rather than expensive"
                ),
                observed=f"{spec.min_holding_minutes}min",
                threshold=f">= {required}min",
            )
        return None

    def _history(self, spec: StrategySpec) -> Rejection | None:
        """The longest lookback must fit inside the training window.

        A 200-bar average over 150 bars of history is not a shorter average —
        the pipeline returns `UNKNOWN` rather than computing over what happens
        to be there — so the spec holds at every decision and is recorded as a
        strategy that found no opportunities. That is a false negative
        indistinguishable from a true one, and it wastes a trial on a question
        about the fixture rather than about the idea.
        """
        if self.n_training_bars is None:
            return None
        needed = spec.max_lookback + HISTORY_MARGIN_BARS
        if needed > self.n_training_bars:
            return Rejection(
                code="insufficient_history",
                reason=(
                    "the longest lookback does not fit inside the training window with "
                    "room to trade. Every feature would be UNKNOWN, which reads as a "
                    "strategy that declined rather than one that could not be evaluated"
                ),
                observed=f"{spec.max_lookback} bars + {HISTORY_MARGIN_BARS} margin",
                threshold=f"<= {self.n_training_bars} bars",
            )
        return None

    def _cost(self, spec: StrategySpec) -> Rejection | None:
        """The binding constraint on this venue, applied before the backtest.

        A US round trip is about 40bps and `max_cost_to_edge_ratio` is 0.33, so
        a spec needs roughly 121bps of declared edge to be admissible at all —
        424bps for an Irish issuer. Most of a random population fails here, and
        failing before the backtest is what makes a thousand-proposal cycle
        affordable.
        """
        verdict, trip = CostModel(self.limits).gate_trade(
            notional_ccy=VALIDATION_NOTIONAL_CCY,
            instrument_currency=self.instrument_currency,
            jurisdiction=self.jurisdiction,
            expected_edge_bps=spec.expected_edge_bps,
        )
        if verdict.allowed:
            return None
        ratio = "n/a" if verdict.ratio is None else f"{verdict.ratio:.3f}"
        return Rejection(
            code="cost_to_edge",
            reason=(
                f"a {trip.total_bps:.1f}bps round trip against a declared "
                f"{spec.expected_edge_bps}bps edge leaves too little of the edge. The "
                "fee schedule is the binding constraint here, not the alpha"
            ),
            observed=ratio,
            threshold=f"<= {verdict.max_ratio}",
        )

    def _feed_noise(self, spec: StrategySpec) -> Rejection | None:
        """The declared edge must exceed the error bar on our own prices.

        Skipped when no bake-off figure exists, unlike the gate, which refuses.
        The asymmetry is deliberate: at the gate an unmeasured error bar must not
        read as a small one, while here refusing every proposal for a missing
        *dataset* measurement would report a search that rejected everything for
        a reason no spec could have avoided.
        """
        if self.feed_noise_p95_bps is None:
            return None
        ratio = Decimal(str(self.limits.data.min_edge_to_feed_noise_ratio))
        required = self.feed_noise_p95_bps * ratio
        if spec.expected_edge_bps < required:
            return Rejection(
                code="edge_to_feed_noise",
                reason=(
                    "the declared edge is smaller than the disagreement between the feeds "
                    "that would produce it. That is not a strategy, it is a measurement "
                    "of vendor noise"
                ),
                observed=f"{spec.expected_edge_bps}bps",
                threshold=f">= {required}bps",
            )
        return None
