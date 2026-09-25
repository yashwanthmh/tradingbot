"""Execution discipline: cost, holding period, session windows, runaway containment.

The cost gate is the most consequential rule in the system, because on this
venue cost rather than alpha is the binding constraint. A US round trip from a
GBP account is ~40bps before spread; a UK one ~60bps with stamp duty; minute-bar
gross edge on liquid names is 5-20bps. So the gate is not a tuning parameter —
it is the rule that keeps the search loop out of a fee trap it would otherwise
breed strategies to fall into.

The anomaly rules are the cheapest insurance here and worth more than the whole
alpha stack: most catastrophic losses from automated trading are not bad
predictions, they are a loop that placed ten thousand orders in four minutes.
"""

from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal

from tb.backtest.costs import CostModel, jurisdiction_from_isin
from tb.risk.state import RiskContext, RuleVerdict, Verdict


@dataclass(frozen=True, slots=True)
class CostToEdgeRule:
    """`expected_cost_bps / expected_edge_bps` must clear the ratio.

    Reuses `tb.backtest.costs.CostModel` rather than reimplementing the
    arithmetic. That is deliberate and load-bearing: if the live gate computed
    costs differently from the backtest gate, every promotion decision would
    have been made against numbers the live path does not honour, and the
    divergence would only show up as live underperformance months later.

    The jurisdiction comes from the ISIN, which is the incorporation country
    and therefore the right key for a transaction tax — a `.L` ticker suffix
    means "listed in London", which is not the same thing and differs by 50bps
    on every entry.
    """

    name: str = "cost_to_edge"

    def evaluate(self, ctx: RiskContext) -> RuleVerdict:
        max_ratio = ctx.limits.execution.max_cost_to_edge_ratio
        if not ctx.request.is_risk_increasing:
            return RuleVerdict(
                self.name,
                Verdict.NOT_APPLICABLE,
                detail=(
                    "an exit is not gated on cost. The cost is already sunk in the "
                    "position, and refusing the exit does not avoid it — it just keeps "
                    "the risk."
                ),
                limit_value=max_ratio,
            )
        edge = ctx.request.expected_edge_bps
        if edge is None:
            return RuleVerdict(
                self.name,
                Verdict.BLOCK,
                detail=(
                    "no expected edge declared, and the gate divides by it. On a venue "
                    "where a round trip costs 40-140bps, an undeclared edge is not a "
                    "modelling gap — it is a trade that cannot be evaluated."
                ),
                observed_value="undeclared",
                limit_value=max_ratio,
            )
        quantity = ctx.request.quantity
        if quantity is None:
            return RuleVerdict(
                self.name,
                Verdict.NOT_APPLICABLE,
                detail="no quantity yet; re-checked after sizing",
                limit_value=max_ratio,
            )
        notional = quantity * ctx.request.reference_price
        model = CostModel(limits=ctx.limits)
        isin = ctx.extra.get("isin")
        verdict, round_trip = model.gate_trade(
            notional_ccy=notional,
            instrument_currency=ctx.extra.get("instrument_currency") or ctx.limits.currency,
            jurisdiction=jurisdiction_from_isin(isin),
            expected_edge_bps=edge,
        )
        if not verdict.allowed:
            return RuleVerdict(
                self.name,
                Verdict.BLOCK,
                detail=verdict.reason,
                observed_value=verdict.expected_cost_bps,
                limit_value=max_ratio,
            )
        return RuleVerdict(
            self.name,
            Verdict.PASS,
            detail=(
                f"{round_trip.total_bps:.1f}bps round trip against a {edge}bps declared "
                f"edge (ratio {verdict.ratio}, limit {max_ratio})"
            ),
            observed_value=verdict.expected_cost_bps,
            limit_value=max_ratio,
        )


@dataclass(frozen=True, slots=True)
class MinHoldingPeriodRule:
    """A position may not be closed before the minimum holding period.

    A risk rule rather than a strategy parameter, because its purpose is to
    bound turnover — and therefore cost — regardless of what any strategy
    believes. A strategy that could set its own minimum would set it to zero.

    The one exception is a protective stop, which must be able to fire at any
    time. A minimum hold that applied to stops would be a rule that forbids
    the loss-limiting mechanism from working for the first hour of every
    position, which is when a gap is most likely to matter.
    """

    name: str = "min_holding_period"

    def evaluate(self, ctx: RiskContext) -> RuleVerdict:
        minimum = ctx.limits.execution.min_holding_minutes
        if ctx.request.is_risk_increasing:
            return RuleVerdict(self.name, Verdict.NOT_APPLICABLE, detail="entry")
        if ctx.request.purpose.is_risk_reducing and _is_protective(ctx):
            return RuleVerdict(
                self.name,
                Verdict.PASS,
                detail=(
                    "a protective stop may fire at any time. A minimum hold applied to "
                    "stops would disable loss-limiting for the first "
                    f"{minimum} minutes of every position."
                ),
                limit_value=minimum,
            )
        if minimum == 0:
            return RuleVerdict(
                self.name, Verdict.PASS, detail="no minimum holding period configured"
            )
        entry_at = ctx.position_entry_at
        if entry_at is None:
            # No entry time means no evidence the position is young. Passing is
            # the safe direction here, unlike everywhere else in this package:
            # the failure being avoided is a position that cannot be closed.
            return RuleVerdict(
                self.name,
                Verdict.WARN,
                detail=(
                    "no recorded entry time, so the holding period is unknown. Permitted: "
                    "an unclosable position is a worse outcome than an early exit."
                ),
                observed_value="unknown",
                limit_value=minimum,
                is_blocking=False,
            )
        held_minutes = (ctx.as_of - entry_at).total_seconds() / 60
        if held_minutes < 0:
            # Mirrors PositionState.holding_minutes_at: a negative interval is
            # less than every minimum, so treating it as a number would block
            # the exit at every decision, forever.
            return RuleVerdict(
                self.name,
                Verdict.WARN,
                detail=(
                    f"position entry {entry_at.isoformat()} is after the decision time "
                    f"{ctx.as_of.isoformat()}: clock skew or a bad reconciliation. "
                    "Permitted rather than treated as 'held for negative minutes', which "
                    "would refuse the exit at every decision from now on."
                ),
                observed_value=held_minutes,
                limit_value=minimum,
                is_blocking=False,
            )
        if held_minutes < minimum:
            return RuleVerdict(
                self.name,
                Verdict.BLOCK,
                detail=(
                    f"held {held_minutes:.0f} of a required {minimum} minutes. This bounds "
                    "turnover, which on this venue bounds cost."
                ),
                observed_value=held_minutes,
                limit_value=minimum,
            )
        return RuleVerdict(
            self.name,
            Verdict.PASS,
            detail=f"held {held_minutes:.0f} minutes, past the {minimum} minimum",
            observed_value=held_minutes,
            limit_value=minimum,
        )


def _is_protective(ctx: RiskContext) -> bool:
    from tb.broker.port import OrderPurpose

    return ctx.request.purpose in (OrderPurpose.PROTECTIVE_STOP, OrderPurpose.FLATTEN)


@dataclass(frozen=True, slots=True)
class SessionWindowRule:
    """No entries in the first or last minutes of a session.

    Both ends are hostile for the same underlying reason — the price is not yet
    or no longer a consensus — but the consequences differ. The open is where
    the overnight auction imbalance clears, so spreads are widest and a market
    order pays for it. The close is where a position that cannot be exited
    becomes an overnight gap, which is the exposure the unprotected-window
    sizing assumes away.
    """

    name: str = "session_window"

    def evaluate(self, ctx: RiskContext) -> RuleVerdict:
        first = ctx.limits.execution.no_entry_first_minutes
        last = ctx.limits.execution.no_entry_last_minutes
        if not ctx.request.is_risk_increasing:
            return RuleVerdict(
                self.name,
                Verdict.NOT_APPLICABLE,
                detail="exits are permitted throughout the session, including at the close",
            )
        since_open = ctx.minutes_since_open
        until_close = ctx.minutes_until_close
        if since_open is None or until_close is None:
            # Almost always because the market is shut — overnight, a weekend,
            # a holiday — and said so, since that refusal is routine. An order
            # sent now would queue for the next open and fill at a price this
            # decision never saw. The note also carries the rarer case, a date
            # past the calendar's range, which needs a human.
            return RuleVerdict(
                self.name,
                Verdict.BLOCK,
                detail=(
                    f"no regular session is open: "
                    f"{ctx.session_note or 'the calendar could not place this instant'}. "
                    "An entry now would wait for the next open and fill at a price this "
                    "decision never saw."
                ),
                observed_value="closed",
                limit_value=f"{first}/{last}",
            )
        if since_open < first:
            return RuleVerdict(
                self.name,
                Verdict.BLOCK,
                detail=(
                    f"{since_open} minutes into the session, inside the {first}-minute "
                    "opening window where the overnight imbalance is still clearing and "
                    "the spread is widest"
                ),
                observed_value=since_open,
                limit_value=first,
            )
        if until_close < last:
            return RuleVerdict(
                self.name,
                Verdict.BLOCK,
                detail=(
                    f"{until_close} minutes to the close, inside the {last}-minute window. "
                    "An entry here may not be exitable today, which turns it into the "
                    "overnight gap the sizing rules assume away."
                ),
                observed_value=until_close,
                limit_value=last,
            )
        return RuleVerdict(
            self.name,
            Verdict.PASS,
            detail=f"{since_open} minutes in, {until_close} to the close",
            observed_value=min(since_open, until_close),
            limit_value=max(first, last),
        )


@dataclass(frozen=True, slots=True)
class OrderCountRule:
    """Per-day and per-symbol order counts.

    Counted from our own intent log rather than from the broker's order list.
    Two reasons: `GET /equity/orders` is rate-limited to one call per five
    seconds while market orders POST at fifty per minute, so the broker's view
    is always behind; and an order we sent but never got a response for still
    counts against a runaway-loop budget, which is exactly the case the
    broker's list omits.
    """

    name: str = "order_count"

    def evaluate(self, ctx: RiskContext) -> RuleVerdict:
        daily = ctx.limits.execution.max_orders_per_day
        per_symbol = ctx.limits.execution.max_orders_per_symbol_per_day
        today = ctx.account.orders_today
        for_symbol = ctx.account.orders_today_for_symbol

        # Risk-reducing orders are counted but not capped. A day that has
        # exhausted its order budget must still be able to close positions,
        # and the budget exists to bound cost and runaway loops — neither of
        # which is served by trapping exposure overnight.
        if not ctx.request.is_risk_increasing:
            return RuleVerdict(
                self.name,
                Verdict.PASS,
                detail=(
                    f"{today} orders today; risk-reducing orders are counted but not "
                    "capped, or an exhausted budget would trap positions"
                ),
                observed_value=today,
                limit_value=daily,
            )
        if today >= daily:
            return RuleVerdict(
                self.name,
                Verdict.BLOCK,
                detail=f"{today} orders placed today, at the cap of {daily}",
                observed_value=today,
                limit_value=daily,
            )
        if for_symbol >= per_symbol:
            return RuleVerdict(
                self.name,
                Verdict.BLOCK,
                detail=(
                    f"{for_symbol} orders on {ctx.request.t212_ticker} today, at the "
                    f"per-symbol cap of {per_symbol}. This one also protects the venue's "
                    "50-pending-per-ticker ceiling, which if exhausted would cause a "
                    "protective stop to be rejected."
                ),
                observed_value=for_symbol,
                limit_value=per_symbol,
            )
        return RuleVerdict(
            self.name,
            Verdict.PASS,
            detail=f"{today}/{daily} today, {for_symbol}/{per_symbol} on this symbol",
            observed_value=today,
            limit_value=daily,
        )


@dataclass(frozen=True, slots=True)
class AnomalyRule:
    """Runaway-loop containment. The cheapest gate in the system.

    Two bounds, and the second exists because the first is useless on day one.
    A percentile comparison needs history; a brand-new deployment has none,
    which is precisely when a fresh bug is most likely to be running. So
    `absolute_max_orders_per_hour` is a flat backstop that applies regardless.
    """

    name: str = "anomaly"

    def evaluate(self, ctx: RiskContext) -> RuleVerdict:
        hourly_cap = ctx.limits.anomaly.absolute_max_orders_per_hour
        multiple = ctx.limits.anomaly.halt_if_daily_orders_exceed_p95_by
        last_hour = ctx.account.orders_last_hour

        # Applies to risk-reducing orders too, unlike every other cap here. A
        # runaway loop emitting ten thousand *exits* is still a runaway loop,
        # and this rule is about the loop rather than about the exposure.
        if last_hour >= hourly_cap:
            return RuleVerdict(
                self.name,
                Verdict.BLOCK,
                detail=(
                    f"{last_hour} orders in the last hour, at the absolute ceiling of "
                    f"{hourly_cap}. This is the cold-start backstop: it applies before "
                    "any percentile history exists, and to exits as well as entries, "
                    "because a runaway loop is a loop whichever way it trades."
                ),
                observed_value=last_hour,
                limit_value=hourly_cap,
            )
        p95 = ctx.account.p95_orders_per_day
        if p95 is not None and p95 > 0:
            threshold = p95 * multiple
            if ctx.account.orders_today > threshold:
                return RuleVerdict(
                    self.name,
                    Verdict.BLOCK,
                    detail=(
                        f"{ctx.account.orders_today} orders today is more than {multiple}x "
                        f"the p95 of {p95:.0f}. Either the universe changed or something "
                        "is looping."
                    ),
                    observed_value=ctx.account.orders_today,
                    limit_value=threshold,
                )
        return RuleVerdict(
            self.name,
            Verdict.PASS,
            detail=f"{last_hour} orders in the last hour, {ctx.account.orders_today} today",
            observed_value=last_hour,
            limit_value=hourly_cap,
        )


@dataclass(frozen=True, slots=True)
class UnprotectedGapRule:
    """Size must survive a gap through the unprotected window.

    Trading 212 has no bracket or OCO orders, so an entry fill *always*
    precedes its protective stop. The window is real and cannot be designed
    away — only sized around. This rule is where that sizing rule becomes an
    actual constraint on an actual order rather than an inequality the config
    validator checked once.

    The arithmetic: a position of `notional` suffering
    `unprotected_gap_pct_assumption` loses `notional * gap%`, and that has to
    fit inside the remaining daily loss budget — the *remaining* budget, not
    the whole one, because a day already down 1.5% of a 2% allowance has far
    less room than the config validator's worst case assumed.
    """

    name: str = "unprotected_gap"

    def evaluate(self, ctx: RiskContext) -> RuleVerdict:
        gap_pct = Decimal(str(ctx.limits.execution.unprotected_gap_pct_assumption))
        daily_pct = Decimal(str(ctx.limits.loss.daily_halt_pct))
        if not ctx.request.is_risk_increasing:
            return RuleVerdict(
                self.name, Verdict.NOT_APPLICABLE, detail="an exit closes the window"
            )
        equity = ctx.account.equity_ccy
        if equity is None or equity <= 0:
            return RuleVerdict(
                self.name,
                Verdict.BLOCK,
                detail="equity is unknown, so the survivable gap cannot be computed",
                observed_value="unknown",
                limit_value=gap_pct,
            )
        spent_pct = Decimal(str(-(ctx.account.day_pnl_pct or 0.0)))
        remaining_pct = daily_pct - max(spent_pct, Decimal(0))
        if remaining_pct <= 0:
            return RuleVerdict(
                self.name,
                Verdict.BLOCK,
                detail=(
                    f"the daily loss budget is exhausted ({spent_pct:.2f}% of {daily_pct}%), "
                    "so no unprotected window is survivable"
                ),
                observed_value=spent_pct,
                limit_value=daily_pct,
            )
        budget_ccy = equity * remaining_pct / Decimal(100)
        # notional * gap% <= budget  =>  notional <= budget / gap%
        max_notional = (budget_ccy * Decimal(100) / gap_pct).quantize(Decimal("0.01"))
        max_quantity = (max_notional / ctx.request.reference_price).quantize(Decimal("0.00000001"))
        if max_quantity <= 0:
            return RuleVerdict(
                self.name,
                Verdict.BLOCK,
                detail=(
                    f"no size survives a {gap_pct}% gap inside the remaining "
                    f"{remaining_pct:.2f}% daily budget"
                ),
                observed_value=max_notional,
                limit_value=budget_ccy,
            )
        return RuleVerdict(
            self.name,
            Verdict.PASS,
            detail=(
                f"up to {max_notional} survives a {gap_pct}% unprotected gap within the "
                f"remaining {remaining_pct:.2f}% of the daily budget"
            ),
            observed_value=max_notional,
            limit_value=budget_ccy,
            max_quantity=max_quantity,
        )
