"""Capital caps: how much may be at risk, in total and per position.

Every rule here fails closed on a missing input. `AccountState` makes equity
optional precisely so these rules can tell "deployed 4%" from "we do not know
what is deployed", and the second must block — a percentage cap evaluated
against an equity of zero passes trivially and authorises an unbounded order.

Two of these are sizing rules rather than gates: they return a `max_quantity`
and let the engine take the minimum. A cap that only ever said no would make
the bot untradeable at any size, when the correct answer is almost always
"yes, but smaller".
"""

from __future__ import annotations

from dataclasses import dataclass
from decimal import ROUND_DOWN, Decimal

from tb.risk.state import RiskContext, RuleVerdict, Verdict


def _quantity_for(notional: Decimal, price: Decimal) -> Decimal:
    """Shares affordable at `notional`, rounded *down*.

    Down, always. Rounding up to reach a whole share is how a 1% cap becomes
    1.4% on a high-priced instrument, and the cap is the blast radius rather
    than a preference. Trading 212 supports fractional quantities, so the
    rounding is to a fine quantum rather than to an integer.
    """
    if price <= 0:  # pragma: no cover - OrderRequest refuses it
        return Decimal(0)
    return (notional / price).quantize(Decimal("0.00000001"), rounding=ROUND_DOWN)


@dataclass(frozen=True, slots=True)
class AbsoluteCeilingRule:
    """The hard currency ceiling. The one cap that does not scale with equity.

    Percentage caps grow as the account grows, which is correct for risk and
    wrong for a blast radius: the agent must not be able to enlarge the
    consequences of its own bug by making money. This is the number a human
    raises by hand.
    """

    name: str = "absolute_ceiling"

    def evaluate(self, ctx: RiskContext) -> RuleVerdict:
        ceiling = ctx.limits.capital.absolute_ceiling_ccy
        if not ctx.request.is_risk_increasing:
            return RuleVerdict(
                self.name, Verdict.NOT_APPLICABLE, detail="exits reduce deployed capital"
            )
        deployed = ctx.account.deployed_ccy
        if deployed is None:
            return RuleVerdict(
                self.name,
                Verdict.BLOCK,
                detail=(
                    "deployed capital is unknown, so the ceiling cannot be checked. "
                    "Blocked rather than assumed: an unmeasured exposure is not a small one."
                ),
                observed_value="unknown",
                limit_value=ceiling,
            )
        headroom = ceiling - deployed
        if headroom <= 0:
            return RuleVerdict(
                self.name,
                Verdict.BLOCK,
                detail=(
                    f"{deployed} already deployed against a {ceiling} absolute ceiling. "
                    "This limit is raised by a human, never by the bot."
                ),
                observed_value=deployed,
                limit_value=ceiling,
            )
        return RuleVerdict(
            self.name,
            Verdict.PASS,
            detail=f"{headroom} of {ceiling} headroom remains",
            observed_value=deployed,
            limit_value=ceiling,
            max_quantity=_quantity_for(headroom, ctx.request.reference_price),
        )


@dataclass(frozen=True, slots=True)
class PerPositionCapRule:
    """No single position beyond `per_position_pct` of equity.

    Also the rule that carries the unprotected-gap arithmetic: the config
    validator has already proved that a position of this size suffering the
    assumed gap fits inside the daily loss budget, so respecting this cap is
    what makes that proof true at runtime rather than only on paper.
    """

    name: str = "per_position_cap"

    def evaluate(self, ctx: RiskContext) -> RuleVerdict:
        pct = ctx.limits.capital.per_position_pct
        if not ctx.request.is_risk_increasing:
            return RuleVerdict(self.name, Verdict.NOT_APPLICABLE, detail="exit")
        equity = ctx.account.equity_ccy
        if equity is None or equity <= 0:
            return RuleVerdict(
                self.name,
                Verdict.BLOCK,
                detail=(
                    "account equity is unknown, so a percentage cap has nothing to apply "
                    "to. A cap evaluated against zero equity would pass every order."
                ),
                observed_value="unknown",
                limit_value=pct,
            )
        cap_ccy = (equity * Decimal(str(pct)) / Decimal(100)).quantize(Decimal("0.01"))
        # Existing exposure in this instrument counts against the same cap: an
        # add is the same position getting larger, not a new one.
        held_ccy = ctx.position_quantity * ctx.request.reference_price
        headroom = cap_ccy - held_ccy
        if headroom <= 0:
            return RuleVerdict(
                self.name,
                Verdict.BLOCK,
                detail=(
                    f"already holding {held_ccy} of {ctx.request.t212_ticker}, at or past "
                    f"the {pct}% per-position cap ({cap_ccy})"
                ),
                observed_value=held_ccy,
                limit_value=cap_ccy,
            )
        return RuleVerdict(
            self.name,
            Verdict.PASS,
            detail=f"{headroom} of {cap_ccy} per-position headroom",
            observed_value=held_ccy,
            limit_value=cap_ccy,
            max_quantity=_quantity_for(headroom, ctx.request.reference_price),
        )


@dataclass(frozen=True, slots=True)
class DeployedCapRule:
    """Total deployed capital as a percentage of equity."""

    name: str = "deployed_cap"

    def evaluate(self, ctx: RiskContext) -> RuleVerdict:
        pct = ctx.limits.capital.max_deployed_pct
        if not ctx.request.is_risk_increasing:
            return RuleVerdict(self.name, Verdict.NOT_APPLICABLE, detail="exit")
        equity = ctx.account.equity_ccy
        deployed = ctx.account.deployed_ccy
        if equity is None or deployed is None or equity <= 0:
            return RuleVerdict(
                self.name,
                Verdict.BLOCK,
                detail="equity or deployed capital is unknown; the total cap cannot be checked",
                observed_value="unknown",
                limit_value=pct,
            )
        cap_ccy = (equity * Decimal(str(pct)) / Decimal(100)).quantize(Decimal("0.01"))
        headroom = cap_ccy - deployed
        observed_pct = float(deployed / equity * 100)
        if headroom <= 0:
            return RuleVerdict(
                self.name,
                Verdict.BLOCK,
                detail=f"{observed_pct:.2f}% deployed, at or past the {pct}% cap",
                observed_value=observed_pct,
                limit_value=pct,
            )
        return RuleVerdict(
            self.name,
            Verdict.PASS,
            detail=f"{observed_pct:.2f}% deployed of {pct}% permitted",
            observed_value=observed_pct,
            limit_value=pct,
            max_quantity=_quantity_for(headroom, ctx.request.reference_price),
        )


@dataclass(frozen=True, slots=True)
class MaxPositionsRule:
    """A cap on how many positions may be open at once.

    Separate from the deployed cap because they fail differently: twenty-five
    positions at 0.4% each satisfies the notional cap and still means
    twenty-five protective stops to place, twenty-five reconciliations per
    cycle, and a rate-limit budget that cannot service them.
    """

    name: str = "max_positions"

    def evaluate(self, ctx: RiskContext) -> RuleVerdict:
        limit = ctx.limits.capital.max_positions
        if not ctx.request.is_risk_increasing:
            return RuleVerdict(self.name, Verdict.NOT_APPLICABLE, detail="exit")
        open_now = ctx.account.n_open_positions
        # Adding to a position already held does not open a new one.
        if ctx.has_position:
            return RuleVerdict(
                self.name,
                Verdict.PASS,
                detail=f"already holding {ctx.request.t212_ticker}; not a new position",
                observed_value=open_now,
                limit_value=limit,
            )
        if open_now >= limit:
            return RuleVerdict(
                self.name,
                Verdict.BLOCK,
                detail=f"{open_now} positions open, at the cap of {limit}",
                observed_value=open_now,
                limit_value=limit,
            )
        return RuleVerdict(
            self.name,
            Verdict.PASS,
            detail=f"{open_now} of {limit} positions open",
            observed_value=open_now,
            limit_value=limit,
        )


@dataclass(frozen=True, slots=True)
class FloorNotionalRule:
    """An order must be worth placing at all.

    Below the floor, fixed costs dominate: a £3 order paying 40bps round-trip
    plus a minimum FX charge is a rounding error with a fee attached. This is
    also the rule that catches a sizing bug that has quantised down to almost
    nothing — an order for 0.00001 shares is not a small position, it is a
    mistake that would otherwise be submitted.

    It never blocks an exit. A position that is somehow below the floor still
    has to be closeable, or the floor becomes a trap.
    """

    name: str = "floor_notional"

    def evaluate(self, ctx: RiskContext) -> RuleVerdict:
        floor = ctx.limits.capital.floor_notional_ccy
        if not ctx.request.is_risk_increasing:
            return RuleVerdict(
                self.name,
                Verdict.NOT_APPLICABLE,
                detail="an exit below the floor must still be permitted, or the floor traps it",
            )
        quantity = ctx.request.quantity
        if quantity is None:
            # Sizing has not happened yet; the engine re-checks the floor
            # against the size it settles on.
            return RuleVerdict(
                self.name,
                Verdict.NOT_APPLICABLE,
                detail="no quantity yet; re-checked after sizing",
                limit_value=floor,
            )
        notional = quantity * ctx.request.reference_price
        if notional < floor:
            return RuleVerdict(
                self.name,
                Verdict.BLOCK,
                detail=(
                    f"{notional} is below the {floor} floor. At this size the fixed costs "
                    "exceed any plausible edge, so the order is not worth placing."
                ),
                observed_value=notional,
                limit_value=floor,
            )
        return RuleVerdict(
            self.name,
            Verdict.PASS,
            detail=f"{notional} clears the {floor} floor",
            observed_value=notional,
            limit_value=floor,
        )


@dataclass(frozen=True, slots=True)
class BrokerQuantityRule:
    """The instrument's own minimum and maximum, as the broker declares them.

    The maximum is a sizing opinion; the minimum is a gate, and it is the one
    with teeth. An order below `min_trade_quantity` is rejected by the venue —
    and a *protective stop* rejected for being below the minimum leaves a
    naked position. So an entry that could not be protected at the same size
    must not be placed.
    """

    name: str = "broker_quantity"

    def evaluate(self, ctx: RiskContext) -> RuleVerdict:
        minimum = ctx.min_trade_quantity
        maximum = ctx.max_open_quantity
        quantity = ctx.request.quantity
        if minimum is None and maximum is None:
            return RuleVerdict(
                self.name,
                Verdict.NOT_APPLICABLE,
                detail="no cached instrument record declares quantity bounds",
            )
        if quantity is None:
            return RuleVerdict(
                self.name,
                Verdict.NOT_APPLICABLE,
                detail="no quantity yet; re-checked after sizing",
                limit_value=minimum,
            )
        if minimum is not None and quantity < minimum:
            return RuleVerdict(
                self.name,
                Verdict.BLOCK,
                detail=(
                    f"{quantity} is below the broker's minimum of {minimum}. The venue "
                    "would reject it — and it would reject the protective stop for the "
                    "same reason, which is how an entry becomes a naked position."
                ),
                observed_value=quantity,
                limit_value=minimum,
            )
        if maximum is not None:
            total = ctx.position_quantity + quantity if ctx.request.is_risk_increasing else quantity
            if total > maximum:
                return RuleVerdict(
                    self.name,
                    Verdict.BLOCK,
                    detail=f"{total} would exceed the broker's maximum open quantity {maximum}",
                    observed_value=total,
                    limit_value=maximum,
                    max_quantity=max(maximum - ctx.position_quantity, Decimal(0)),
                )
        return RuleVerdict(
            self.name,
            Verdict.PASS,
            detail=f"{quantity} within the broker's bounds",
            observed_value=quantity,
            limit_value=minimum,
        )
