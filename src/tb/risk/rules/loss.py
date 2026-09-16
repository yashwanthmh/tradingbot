"""The circuit breakers: daily, rolling and drawdown.

Three thresholds in a deliberate order, which the config validator enforces:
`daily_halt_pct <= rolling_5d_halt_pct <= max_drawdown_flatten_pct`. Each is a
different statement. A bad day is normal. A bad week is a signal. A drawdown
from peak past the outermost threshold is "stop and flatten", because at that
point the thing being doubted is not a strategy but the system's own model of
its edge.

The first two **stop opening**. The third **flattens**, which is why it must
never block a risk-reducing order: a breaker that fired to close positions and
then blocked the closing orders would lock in exactly the loss it existed to
limit. That is the single most important line in this file.
"""

from __future__ import annotations

from dataclasses import dataclass

from tb.risk.state import RiskContext, RuleVerdict, Verdict


def _exit_is_always_allowed(name: str, ctx: RiskContext, *, threshold: float) -> RuleVerdict:
    return RuleVerdict(
        name,
        Verdict.PASS,
        detail=(
            "loss breakers stop new exposure; they never block a risk-reducing order. "
            "Blocking one would lock in the loss the breaker fired to limit."
        ),
        observed_value=ctx.account.day_pnl_pct,
        limit_value=threshold,
    )


@dataclass(frozen=True, slots=True)
class DailyLossRule:
    """Stop opening once the day's loss passes the threshold."""

    name: str = "daily_loss"

    def evaluate(self, ctx: RiskContext) -> RuleVerdict:
        threshold = ctx.limits.loss.daily_halt_pct
        if not ctx.request.is_risk_increasing:
            return _exit_is_always_allowed(self.name, ctx, threshold=threshold)
        observed = ctx.account.day_pnl_pct
        if observed is None:
            return RuleVerdict(
                self.name,
                Verdict.BLOCK,
                detail=(
                    "the day's P&L is unknown, so the daily breaker cannot be evaluated. "
                    "Blocked rather than assumed flat: this is the breaker most likely to "
                    "be the one that should have fired."
                ),
                observed_value="unknown",
                limit_value=threshold,
            )
        # `day_pnl_pct` is signed; a loss is negative, the threshold positive.
        loss_pct = -observed
        if loss_pct >= threshold:
            return RuleVerdict(
                self.name,
                Verdict.BLOCK,
                detail=f"down {loss_pct:.2f}% today, at or past the {threshold}% daily halt",
                observed_value=loss_pct,
                limit_value=threshold,
            )
        # Warn inside the last fifth of the budget. Not decoration: the run
        # that ends the day at -1.9% against a 2% limit is the one where
        # someone wants to know the breaker was nearly reached.
        if loss_pct >= threshold * 0.8:
            return RuleVerdict(
                self.name,
                Verdict.WARN,
                detail=f"down {loss_pct:.2f}%, inside the last fifth of the {threshold}% budget",
                observed_value=loss_pct,
                limit_value=threshold,
                is_blocking=False,
            )
        return RuleVerdict(
            self.name,
            Verdict.PASS,
            detail=f"day P&L {observed:+.2f}% against a {threshold}% halt",
            observed_value=loss_pct,
            limit_value=threshold,
        )


@dataclass(frozen=True, slots=True)
class RollingLossRule:
    """Five-day loss. Catches the slow bleed a daily breaker never sees.

    A strategy losing 1.5% a day against a 2% daily limit never trips it and
    is down 7% in a week.
    """

    name: str = "rolling_5d_loss"

    def evaluate(self, ctx: RiskContext) -> RuleVerdict:
        threshold = ctx.limits.loss.rolling_5d_halt_pct
        if not ctx.request.is_risk_increasing:
            return _exit_is_always_allowed(self.name, ctx, threshold=threshold)
        observed = ctx.account.rolling_5d_pnl_pct
        if observed is None:
            return RuleVerdict(
                self.name,
                Verdict.BLOCK,
                detail="the rolling 5-day P&L is unknown, so the breaker cannot be evaluated",
                observed_value="unknown",
                limit_value=threshold,
            )
        loss_pct = -observed
        if loss_pct >= threshold:
            return RuleVerdict(
                self.name,
                Verdict.BLOCK,
                detail=(
                    f"down {loss_pct:.2f}% over five sessions, past the {threshold}% limit. "
                    "This is the breaker that catches a bleed no single day would trip."
                ),
                observed_value=loss_pct,
                limit_value=threshold,
            )
        return RuleVerdict(
            self.name,
            Verdict.PASS,
            detail=f"5-day P&L {observed:+.2f}% against a {threshold}% halt",
            observed_value=loss_pct,
            limit_value=threshold,
        )


@dataclass(frozen=True, slots=True)
class DrawdownRule:
    """Drawdown from peak equity. The outermost breaker: flatten and halt.

    Measured from the high-water mark rather than from the day's open, because
    a series of small losses that never trips a daily or weekly breaker still
    ends up here — which is the point of having a third threshold rather than
    a larger version of the first.
    """

    name: str = "max_drawdown"

    def evaluate(self, ctx: RiskContext) -> RuleVerdict:
        threshold = ctx.limits.loss.max_drawdown_flatten_pct
        if not ctx.request.is_risk_increasing:
            return RuleVerdict(
                self.name,
                Verdict.PASS,
                detail=(
                    "this breaker *flattens*, so it must permit the flattening orders. "
                    "Blocking them would be a breaker that locks in the loss it fired to "
                    "limit — the worst available failure in this file."
                ),
                observed_value=ctx.account.drawdown_from_peak_pct,
                limit_value=threshold,
            )
        observed = ctx.account.drawdown_from_peak_pct
        if observed is None:
            return RuleVerdict(
                self.name,
                Verdict.BLOCK,
                detail="drawdown from peak is unknown, so the flatten breaker cannot be evaluated",
                observed_value="unknown",
                limit_value=threshold,
            )
        if observed >= threshold:
            return RuleVerdict(
                self.name,
                Verdict.BLOCK,
                detail=(
                    f"{observed:.2f}% below peak equity, past the {threshold}% flatten "
                    "threshold. Past here the thing in doubt is the system's model of its "
                    "own edge, not one strategy."
                ),
                observed_value=observed,
                limit_value=threshold,
            )
        return RuleVerdict(
            self.name,
            Verdict.PASS,
            detail=f"{observed:.2f}% below peak against a {threshold}% flatten threshold",
            observed_value=observed,
            limit_value=threshold,
        )
