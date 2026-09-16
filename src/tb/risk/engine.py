"""The risk engine. The only module that may construct a `RiskToken`.

Three properties matter here, and each is a choice that could reasonably have
gone the other way:

**Every rule is evaluated, always. No short-circuit.** Stopping at the first
block would be faster and would lose the thing the ledger is for. "Blocked by
the daily loss breaker" and "blocked by the daily loss breaker and three other
rules" are different situations calling for different responses, and a
first-failure-only record cannot tell them apart. Recording the passes matters
too: it makes margins queryable, so "the caps were never close" can be
distinguished from "the caps were holding it back all week".

**Sizing is the minimum of every opinion.** Several rules return a
`max_quantity` — the absolute ceiling's headroom, the per-position cap, the
deployed cap, the survivable unprotected gap, the broker's own maximum. The
engine takes the smallest and then *re-evaluates* the quantity-dependent rules
against the size it settled on, because the cost gate and the floor cannot be
judged before a size exists. Sizing before gating would be backwards; gating
without re-checking would let a size shrink below the floor unnoticed.

**A refusal is a recorded outcome, not an exception.** `evaluate` returns an
`Evaluation` either way. An exception would leave no verdict rows, and "why
did it stop trading" is the more common question than "why did it trade".
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import timedelta
from decimal import ROUND_DOWN, Decimal

from tb.broker.port import OrderPurpose, Side
from tb.config.hard_limits import HardLimits
from tb.core.clock import now_utc
from tb.core.ids import new_id
from tb.risk.rules.capital import (
    AbsoluteCeilingRule,
    BrokerQuantityRule,
    DeployedCapRule,
    FloorNotionalRule,
    MaxPositionsRule,
    PerPositionCapRule,
)
from tb.risk.rules.execution import (
    AnomalyRule,
    CostToEdgeRule,
    MinHoldingPeriodRule,
    OrderCountRule,
    SessionWindowRule,
    UnprotectedGapRule,
)
from tb.risk.rules.loss import DailyLossRule, DrawdownRule, RollingLossRule
from tb.risk.rules.safety import (
    BarFreshnessRule,
    CrossVenueAgreementRule,
    HaltedRule,
    RegimeRule,
    SymbolTradableRule,
)
from tb.risk.state import (
    OrderRequest,
    RiskContext,
    RiskError,
    RiskRule,
    RuleVerdict,
    Verdict,
)
from tb.risk.token import TOKEN_TTL_SECONDS, RiskToken, _mint
from tb.strategy.base import Action

# The rule set, in evaluation order. Safety first, then loss breakers, then
# capital, then execution — so a reader of a refusal sees the most fundamental
# reason first even though every rule ran.
#
# Order does not affect the outcome (no short-circuit), which is what makes it
# safe to order this list for legibility rather than for logic.
DEFAULT_RULES: tuple[RiskRule, ...] = (
    HaltedRule(),
    SymbolTradableRule(),
    BarFreshnessRule(),
    CrossVenueAgreementRule(),
    DailyLossRule(),
    RollingLossRule(),
    DrawdownRule(),
    AbsoluteCeilingRule(),
    PerPositionCapRule(),
    DeployedCapRule(),
    MaxPositionsRule(),
    RegimeRule(),
    UnprotectedGapRule(),
    SessionWindowRule(),
    MinHoldingPeriodRule(),
    OrderCountRule(),
    # The runaway-loop breaker. Listed here rather than omitted by accident:
    # it was imported and left out of this tuple on the first write, which the
    # linter caught as an unused import. A rule absent from this tuple does
    # not run, and this is the one the plan calls worth more than the whole
    # alpha stack.
    AnomalyRule(),
    CostToEdgeRule(),
    FloorNotionalRule(),
    BrokerQuantityRule(),
)

# Rules whose verdict depends on the order's quantity, so they cannot be
# judged until sizing has happened. Named explicitly rather than inferred,
# because a rule silently omitted from a second pass is a rule that does not
# run — the M2 lesson, applied here.
QUANTITY_DEPENDENT: frozenset[str] = frozenset(
    {"cost_to_edge", "floor_notional", "broker_quantity"}
)


@dataclass(frozen=True, slots=True)
class Evaluation:
    """The engine's complete answer for one request."""

    approved: bool
    verdicts: tuple[RuleVerdict, ...]
    token: RiskToken | None = None
    approved_quantity: Decimal | None = None
    approved_notional_ccy: Decimal | None = None
    refusal_summary: str = ""
    sizing_notes: tuple[str, ...] = field(default_factory=tuple)

    @property
    def blocking(self) -> tuple[RuleVerdict, ...]:
        return tuple(v for v in self.verdicts if v.blocks)

    @property
    def warnings(self) -> tuple[RuleVerdict, ...]:
        return tuple(v for v in self.verdicts if v.verdict is Verdict.WARN)

    @property
    def n_evaluated(self) -> int:
        """Rules that actually had an opinion.

        Excludes `NOT_APPLICABLE`, so "17 of 17 passed" cannot be claimed for a
        set where six rules declined to apply.
        """
        return sum(1 for v in self.verdicts if v.verdict is not Verdict.NOT_APPLICABLE)

    def verdict_for(self, rule_name: str) -> RuleVerdict | None:
        return next((v for v in self.verdicts if v.rule_name == rule_name), None)

    def summary(self) -> str:
        if self.approved:
            return (
                f"approved {self.approved_quantity} "
                f"({self.n_evaluated} rules evaluated, {len(self.warnings)} warning(s))"
            )
        return self.refusal_summary or "refused"


@dataclass(frozen=True, slots=True)
class RiskEngine:
    """Evaluates a request against every rule and issues a token or a refusal.

    Holds the rule list and nothing else — no clock, no broker, no ledger. The
    caller assembles a `RiskContext` and persists the result, which keeps the
    engine a pure function and makes every rule combination testable without
    a database.
    """

    rules: tuple[RiskRule, ...] = DEFAULT_RULES
    token_ttl_seconds: int = TOKEN_TTL_SECONDS

    def __post_init__(self) -> None:
        names = [rule.name for rule in self.rules]
        duplicates = {n for n in names if names.count(n) > 1}
        if duplicates:
            raise RiskError(
                f"duplicate risk rule name(s) {sorted(duplicates)}. Verdicts are keyed by "
                "rule name in the ledger, so two rules sharing one would overwrite each "
                "other and the record would show fewer checks than ran."
            )
        unknown = QUANTITY_DEPENDENT - set(names)
        if unknown:
            # Guards the second pass against a rename. A quantity-dependent
            # rule that no longer matches its name here would silently stop
            # being re-evaluated after sizing, which is the failure mode this
            # whole codebase keeps finding.
            raise RiskError(
                f"QUANTITY_DEPENDENT names {sorted(unknown)} match no rule in the set. "
                "These rules must be re-evaluated after sizing; a stale name here means "
                "one of them silently never runs on the final quantity."
            )

    # -- the entry point ---------------------------------------------------

    def evaluate(self, ctx: RiskContext, *, run_id: str) -> Evaluation:
        """Evaluate, size, re-evaluate, and issue a token or refuse."""
        first_pass = tuple(self._run_rule(rule, ctx) for rule in self.rules)

        blocking = tuple(v for v in first_pass if v.blocks)
        if blocking:
            return self._refuse(first_pass)

        quantity, notes = self._size(ctx, first_pass)
        if quantity is None or quantity <= 0:
            refusal = RuleVerdict(
                "sizing",
                Verdict.BLOCK,
                detail=(
                    "every rule passed but no positive size survived the caps. This is a "
                    "refusal, not an error: the account has room for nothing here."
                ),
                observed_value=quantity,
                limit_value="> 0",
            )
            return self._refuse((*first_pass, refusal), sizing_notes=notes)

        # Second pass. Only the quantity-dependent rules are re-run; the rest
        # cannot have changed, because nothing in the context moved.
        sized_ctx = _with_quantity(ctx, quantity)
        second_pass = {
            rule.name: self._run_rule(rule, sized_ctx)
            for rule in self.rules
            if rule.name in QUANTITY_DEPENDENT
        }
        final = tuple(second_pass.get(v.rule_name, v) for v in first_pass)

        blocking = tuple(v for v in final if v.blocks)
        if blocking:
            return self._refuse(final, sizing_notes=notes)

        notional = (quantity * ctx.request.reference_price).quantize(Decimal("0.01"))
        token = self._mint_token(
            ctx,
            run_id=run_id,
            quantity=quantity,
            notional=notional,
            rules_passed=tuple(v.rule_name for v in final if v.verdict is Verdict.PASS),
        )
        return Evaluation(
            approved=True,
            verdicts=final,
            token=token,
            approved_quantity=quantity,
            approved_notional_ccy=notional,
            sizing_notes=notes,
        )

    # -- internals ---------------------------------------------------------

    def _run_rule(self, rule: RiskRule, ctx: RiskContext) -> RuleVerdict:
        """Evaluate one rule, converting a crash into a blocking verdict.

        A rule that raises must not take the process down *and* must not be
        skipped. Both would be wrong in the permissive direction: a crash in
        the daily-loss rule that aborted the evaluation would leave the caller
        to decide, and a caller deciding is how "the risk engine errored so we
        traded anyway" happens.
        """
        try:
            verdict = rule.evaluate(ctx)
        # Deliberately broad: any failure in any rule must become a block.
        except Exception as exc:
            return RuleVerdict(
                rule.name,
                Verdict.BLOCK,
                detail=(
                    f"the rule raised {type(exc).__name__}: {exc}. A rule that cannot "
                    "reach a verdict blocks — an unevaluated limit is not a satisfied one."
                ),
                observed_value="error",
            )
        if verdict.rule_name != rule.name:
            # Cheap, and catches a copy-paste between rule modules that would
            # otherwise file one rule's verdict under another's name.
            return RuleVerdict(
                rule.name,
                Verdict.BLOCK,
                detail=(
                    f"rule {rule.name!r} returned a verdict named {verdict.rule_name!r}. "
                    "Verdicts are keyed by name in the ledger, so this would file the "
                    "result under the wrong rule."
                ),
                observed_value=verdict.rule_name,
                limit_value=rule.name,
            )
        return verdict

    def _size(
        self, ctx: RiskContext, verdicts: tuple[RuleVerdict, ...]
    ) -> tuple[Decimal | None, tuple[str, ...]]:
        """The smallest quantity any rule will permit, times the regime factor.

        An explicitly requested quantity is honoured as a *ceiling*, never as a
        floor: an exit asks for the exact position size and must get it, while
        an entry that asked for more than the caps allow gets the caps.
        """
        notes: list[str] = []
        # An explicit loop rather than a comprehension: a comprehension's `if`
        # does not narrow `max_quantity` inside the tuple it builds, so the
        # result would be `Decimal | None` and every downstream multiplication
        # would need a cast.
        opinions: list[tuple[str, Decimal]] = []
        for verdict in verdicts:
            ceiling = verdict.max_quantity
            if ceiling is not None:
                opinions.append((verdict.rule_name, ceiling))

        if not ctx.request.is_risk_increasing:
            # Exits are not sized by the caps. Closing a position means closing
            # the position; a cap that reduced an exit would leave a remainder
            # nothing would ever come back for.
            exit_quantity = ctx.request.quantity or ctx.position_quantity
            notes.append(
                f"exit: taking the position size {exit_quantity} rather than a capped size"
            )
            return (exit_quantity if exit_quantity > 0 else None), tuple(notes)

        if not opinions:
            # No rule offered a ceiling. That means the cap rules all declined
            # to apply, which for an entry should be impossible — so refuse
            # rather than fall back to the requested size.
            notes.append("no rule offered a size ceiling for a risk-increasing order")
            return None, tuple(notes)

        limiting_rule, quantity = min(opinions, key=lambda pair: pair[1])
        notes.append(f"{limiting_rule} is the binding constraint at {quantity}")

        factor = ctx.regime_exposure_factor
        if factor is not None and factor < 1:
            quantity = (quantity * factor).quantize(Decimal("0.00000001"))
            notes.append(f"regime factor x{factor} applied -> {quantity}")

        requested = ctx.request.quantity
        if requested is not None and requested < quantity:
            quantity = requested
            notes.append(f"the request asked for less ({requested}) than the caps allow")

        return quantity, tuple(notes)

    def _refuse(
        self,
        verdicts: tuple[RuleVerdict, ...],
        *,
        sizing_notes: tuple[str, ...] = (),
    ) -> Evaluation:
        blocking = tuple(v for v in verdicts if v.blocks)
        summary = "; ".join(f"{v.rule_name}: {v.detail}" for v in blocking) or "refused"
        return Evaluation(
            approved=False,
            verdicts=verdicts,
            refusal_summary=summary,
            sizing_notes=sizing_notes,
        )

    def _mint_token(
        self,
        ctx: RiskContext,
        *,
        run_id: str,
        quantity: Decimal,
        notional: Decimal,
        rules_passed: tuple[str, ...],
    ) -> RiskToken:
        """The one `RiskToken(...)` construction site in the codebase.

        `tests/test_risk_token.py` parses the whole tree and asserts that. The
        guard in `RiskToken.__post_init__` sees this module as the caller,
        which is why the construction is inline here rather than delegated to
        a helper in `tb.risk.token` — a helper would report *that* module as
        the caller and be refused by its own check.
        """
        issued = ctx.as_of if ctx.as_of.tzinfo else now_utc()
        return RiskToken(
            token_id=new_id("rtok"),
            run_id=run_id,
            t212_ticker=ctx.request.t212_ticker,
            side=ctx.request.side,
            purpose=ctx.request.purpose,
            quantity=quantity,
            issued_at=issued,
            expires_at=issued + timedelta(seconds=self.token_ttl_seconds),
            decision_id=ctx.request.decision_id,
            max_notional_ccy=notional,
            rules_passed=rules_passed,
            mint=_mint,
        )


def _with_quantity(ctx: RiskContext, quantity: Decimal) -> RiskContext:
    """A copy of the context whose request carries the sized quantity.

    Built by hand rather than with `dataclasses.replace` on the request, to
    keep the construction of the nested frozen types explicit — and because
    the quantity-dependent rules are the ones being re-run, so exactly what
    changed between passes should be readable here.
    """
    request = ctx.request
    sized = OrderRequest(
        t212_ticker=request.t212_ticker,
        instrument_uid=request.instrument_uid,
        side=request.side,
        purpose=request.purpose,
        action=request.action,
        reference_price=request.reference_price,
        quantity=quantity,
        expected_edge_bps=request.expected_edge_bps,
        decision_id=request.decision_id,
        strategy_id=request.strategy_id,
    )
    return RiskContext(
        as_of=ctx.as_of,
        limits=ctx.limits,
        request=sized,
        account=ctx.account,
        position_quantity=ctx.position_quantity,
        position_entry_at=ctx.position_entry_at,
        may_enter=ctx.may_enter,
        may_enter_reason=ctx.may_enter_reason,
        may_exit=ctx.may_exit,
        may_exit_reason=ctx.may_exit_reason,
        permits_full_size=ctx.permits_full_size,
        regime_exposure_factor=ctx.regime_exposure_factor,
        regime_state=ctx.regime_state,
        bar_age_seconds=ctx.bar_age_seconds,
        cross_venue_disagreement_bps=ctx.cross_venue_disagreement_bps,
        minutes_since_open=ctx.minutes_since_open,
        minutes_until_close=ctx.minutes_until_close,
        halted=ctx.halted,
        halt_reason=ctx.halt_reason,
        min_trade_quantity=ctx.min_trade_quantity,
        max_open_quantity=ctx.max_open_quantity,
        extra=dict(ctx.extra),
    )


def stop_price_for(
    *,
    entry_price: Decimal,
    limits: HardLimits,
    tick: Decimal = Decimal("0.01"),
) -> Decimal:
    """Where the protective stop goes behind an entry fill.

    Placed at the assumed unprotected gap below the entry, which ties the stop
    to the same number the sizing rules used: size assumes a survivable gap of
    `unprotected_gap_pct_assumption`, so a stop further away than that would
    make the sizing assumption false.

    Rounded *down* to the tick, so rounding can only move the stop further
    from the entry and never closer — a stop rounded up could sit inside the
    spread and fire immediately on a quote blip.
    """
    gap = Decimal(str(limits.execution.unprotected_gap_pct_assumption))
    raw = entry_price * (Decimal(100) - gap) / Decimal(100)
    stops = (raw / tick).to_integral_value(rounding=ROUND_DOWN) * tick
    if stops <= 0:
        raise RiskError(
            f"a {gap}% stop below {entry_price} rounds to {stops}, which is not a price. "
            "A stop at or below zero would never fire."
        )
    return stops


def entry_request(
    *,
    t212_ticker: str,
    instrument_uid: str,
    reference_price: Decimal,
    expected_edge_bps: Decimal,
    decision_id: str | None = None,
    strategy_id: str | None = None,
) -> OrderRequest:
    """An unsized entry request. The engine derives the quantity."""
    return OrderRequest(
        t212_ticker=t212_ticker,
        instrument_uid=instrument_uid,
        side=Side.BUY,
        purpose=OrderPurpose.ENTRY,
        action=Action.ENTER,
        reference_price=reference_price,
        expected_edge_bps=expected_edge_bps,
        decision_id=decision_id,
        strategy_id=strategy_id,
    )


def exit_request(
    *,
    t212_ticker: str,
    instrument_uid: str,
    reference_price: Decimal,
    quantity: Decimal,
    purpose: OrderPurpose = OrderPurpose.EXIT,
    decision_id: str | None = None,
    strategy_id: str | None = None,
) -> OrderRequest:
    """A sized exit request. The quantity is the position, not a cap."""
    return OrderRequest(
        t212_ticker=t212_ticker,
        instrument_uid=instrument_uid,
        side=Side.SELL,
        purpose=purpose,
        action=Action.EXIT,
        reference_price=reference_price,
        quantity=quantity,
        decision_id=decision_id,
        strategy_id=strategy_id,
    )
