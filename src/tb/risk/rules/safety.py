"""Rules that answer "should anything trade at all right now".

These run first and they are the cheapest to get wrong in the permissive
direction, so each is written so that *absence of information blocks*. A halt
we cannot read is a halt. A kill switch we cannot read is engaged. A regime
factor that was never computed is not full exposure.

The asymmetry with exits is deliberate and runs through the whole file: a halt
stops new exposure, it does not trap existing exposure. Refusing to sell over a
data problem is worse than the data problem — it converts something
recoverable into an unhedged position that nothing can close.
"""

from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal

from tb.risk.state import RiskContext, RuleVerdict, Verdict


@dataclass(frozen=True, slots=True)
class HaltedRule:
    """Nothing risk-increasing while halted or killed.

    Exits are explicitly still permitted. The drawdown breaker's whole purpose
    is to *flatten*, which it cannot do if the halt it raises also blocks the
    flattening orders — a breaker that locked in the loss it fired to limit.
    """

    name: str = "halted"

    def evaluate(self, ctx: RiskContext) -> RuleVerdict:
        if not ctx.halted:
            return RuleVerdict(self.name, Verdict.PASS, detail="not halted")
        if not ctx.request.is_risk_increasing:
            return RuleVerdict(
                self.name,
                Verdict.PASS,
                detail=(
                    f"halted ({ctx.halt_reason}) but this order reduces risk. A halt "
                    "that blocked exits would trap the exposure it fired to limit."
                ),
            )
        return RuleVerdict(
            self.name,
            Verdict.BLOCK,
            detail=f"halted: {ctx.halt_reason or 'no reason recorded'}",
            observed_value="halted",
            limit_value="running",
        )


@dataclass(frozen=True, slots=True)
class SymbolTradableRule:
    """The symbol map's two-tier gate, applied to the side being traded.

    Reads `may_enter` for an entry and `may_exit` for an exit, never one for
    the other. The symbol map guarantees `may_exit` is True in every reachable
    state; this rule is where that guarantee is spent.
    """

    name: str = "symbol_tradable"

    def evaluate(self, ctx: RiskContext) -> RuleVerdict:
        if ctx.request.is_risk_increasing:
            if ctx.may_enter:
                return RuleVerdict(
                    self.name,
                    Verdict.PASS,
                    detail=ctx.may_enter_reason or "mapping permits an entry",
                )
            return RuleVerdict(
                self.name,
                Verdict.BLOCK,
                detail=(
                    f"{ctx.request.t212_ticker} may not be entered: "
                    f"{ctx.may_enter_reason or 'no verified mapping'}"
                ),
                observed_value="blocked",
                limit_value="cross_verified",
            )
        if ctx.may_exit:
            return RuleVerdict(self.name, Verdict.PASS, detail="mapping permits an exit")
        # Reachable only through a bug: `SymbolMap.may_exit` is total. Recorded
        # as a verdict anyway rather than asserted, because the useful output
        # of this situation is a ledger row naming it.
        return RuleVerdict(
            self.name,
            Verdict.BLOCK,
            detail=(
                f"{ctx.request.t212_ticker} refused an EXIT: {ctx.may_exit_reason}. This "
                "should be unreachable — every mapping state permits an exit — so it "
                "means the symbol map has a state it should not have."
            ),
            observed_value="exit_blocked",
            limit_value="always_exitable",
        )


@dataclass(frozen=True, slots=True)
class RegimeRule:
    """The portfolio-wide exposure factor, as a sizing opinion.

    Does not block: a reduced regime means smaller, not nothing. It blocks
    only when the factor is *missing*, which means the gate was never
    consulted — and a missing factor treated as 1 would be the exact
    "absence reads as permission" failure the gate exists to prevent.
    """

    name: str = "regime"

    def evaluate(self, ctx: RiskContext) -> RuleVerdict:
        if not ctx.request.is_risk_increasing:
            return RuleVerdict(
                self.name, Verdict.NOT_APPLICABLE, detail="regime scales exposure, not exits"
            )
        factor = ctx.regime_exposure_factor
        if factor is None:
            return RuleVerdict(
                self.name,
                Verdict.BLOCK,
                detail=(
                    "no regime reading was supplied. The engine will not assume full "
                    "exposure from a missing factor — that is the most expensive default "
                    "available here. Run `tb data regime` to see what the gate says."
                ),
                observed_value="unknown",
                limit_value="a factor in [0, 1]",
            )
        if factor <= 0:
            return RuleVerdict(
                self.name,
                Verdict.BLOCK,
                detail=f"regime exposure factor is {factor} ({ctx.regime_state})",
                observed_value=factor,
                limit_value="> 0",
            )
        verdict = Verdict.PASS if factor >= 1 else Verdict.WARN
        return RuleVerdict(
            self.name,
            verdict,
            detail=f"regime {ctx.regime_state or 'unknown'}: exposure x{factor}",
            observed_value=factor,
            limit_value=Decimal(1),
            is_blocking=False,
        )


@dataclass(frozen=True, slots=True)
class BarFreshnessRule:
    """A decision may not be made on a stale price.

    Evaluated against the bar's *knowledge* time, which the data layer already
    computed — the age passed in here is `as_of - available_at`, never
    `as_of - bar_open`. Using bar time would understate the age by exactly the
    provider delay, which is the one number this bound exists to catch.

    **The bound is `bar_period + max_bar_staleness_seconds`, not the config
    value alone.** That is a correction rather than a relaxation, and the
    reason matters: `max_bar_staleness_seconds` is 180, and a daily bar is by
    definition many hours old the moment it becomes knowable. Comparing a
    daily bar's age against 180 seconds refuses every decision at the only
    resolution `allowed_live_resolutions` currently permits — so the two
    settings contradicted each other, and the loop simply never traded.

    What the config value actually bounds is *lateness*: how far past the end
    of its own period a bar may be before the price it carries is not the
    price now. Adding the period makes that explicit and keeps the minute case
    at 240 seconds, which is what was intended. The regime gate reached the
    same conclusion independently and hardcoded four days; this is the general
    form of that.
    """

    name: str = "bar_freshness"

    def evaluate(self, ctx: RiskContext) -> RuleVerdict:
        limit = ctx.limits.execution.max_bar_staleness_seconds + ctx.bar_period_seconds
        if not ctx.request.is_risk_increasing:
            # An exit priced off a slightly stale bar still beats not exiting.
            # Recorded rather than skipped so the staleness is visible if the
            # fill later looks wrong.
            return RuleVerdict(
                self.name,
                Verdict.NOT_APPLICABLE,
                detail=(
                    f"exit: staleness ({ctx.bar_age_seconds}s) recorded but not enforced, "
                    "because refusing to exit on a stale price leaves the position open"
                ),
                observed_value=ctx.bar_age_seconds,
                limit_value=limit,
            )
        age = ctx.bar_age_seconds
        if age is None:
            return RuleVerdict(
                self.name,
                Verdict.BLOCK,
                detail=(
                    "bar age is unknown, so freshness cannot be established. Treated as "
                    "stale: an unmeasured price is not a fresh one."
                ),
                observed_value="unknown",
                limit_value=limit,
            )
        if age > limit:
            return RuleVerdict(
                self.name,
                Verdict.BLOCK,
                detail=f"newest bar is {age:.0f}s old, past the {limit}s bound",
                observed_value=age,
                limit_value=limit,
            )
        return RuleVerdict(
            self.name,
            Verdict.PASS,
            detail=f"bar is {age:.0f}s old",
            observed_value=age,
            limit_value=limit,
        )


@dataclass(frozen=True, slots=True)
class CrossVenueAgreementRule:
    """The data feed and the broker must agree about the price.

    Trading 212 serves no market data, so a signal computed on one venue is
    executed against another. This is the gate on that divergence, and it only
    has an opinion when a broker price exists at all — which, because
    `currentPrice` is reported only for held positions, means it can speak for
    an add or an exit but not for a first entry. The two-tier symbol
    verification covers that case instead.
    """

    name: str = "cross_venue_agreement"

    def evaluate(self, ctx: RiskContext) -> RuleVerdict:
        limit = ctx.limits.execution.max_cross_venue_disagreement_bps
        observed = ctx.cross_venue_disagreement_bps
        if observed is None:
            return RuleVerdict(
                self.name,
                Verdict.NOT_APPLICABLE,
                detail=(
                    "no broker price to compare against. Trading 212 reports "
                    "currentPrice only for held positions, so a first entry cannot be "
                    "checked this way — that is what cross-provider verification is for."
                ),
                limit_value=limit,
            )
        if abs(observed) > limit:
            if not ctx.request.is_risk_increasing:
                return RuleVerdict(
                    self.name,
                    Verdict.WARN,
                    detail=(
                        f"{observed:.1f}bps disagreement exceeds {limit}bps, but this is an "
                        "exit. The position is already on; a mispriced exit is better than "
                        "an unmanaged holding."
                    ),
                    observed_value=observed,
                    limit_value=limit,
                    is_blocking=False,
                )
            return RuleVerdict(
                self.name,
                Verdict.BLOCK,
                detail=(
                    f"data feed and broker disagree by {observed:.1f}bps, past {limit}bps. "
                    "Either the mapping is wrong or one feed is broken; both mean the "
                    "signal was computed on a price this order will not get."
                ),
                observed_value=observed,
                limit_value=limit,
            )
        return RuleVerdict(
            self.name,
            Verdict.PASS,
            detail=f"venues agree within {observed:.1f}bps",
            observed_value=observed,
            limit_value=limit,
        )
