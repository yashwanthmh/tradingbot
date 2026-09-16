"""What a risk rule is allowed to see, and the vocabulary every rule speaks.

Split from the engine so each rule is a pure function of a value object and is
testable against a hand-built one — no broker, no ledger, no clock. The engine
assembles the state once per decision and hands the same immutable snapshot to
every rule, which matters for a reason beyond tidiness: rules that each fetched
their own view of equity could disagree about it, and a set of verdicts
computed against different account states is not a coherent decision.

`RiskContext` is deliberately a *snapshot*, not a live accessor. Two rules
reading `deployed_pct` must get the same number even if a fill lands between
them.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from decimal import Decimal
from enum import StrEnum
from typing import Protocol, runtime_checkable

from tb.broker.port import OrderPurpose, Side
from tb.config.hard_limits import HardLimits
from tb.core.errors import TbError
from tb.strategy.base import Action


class RiskError(TbError):
    """A risk evaluation could not be completed."""


class Verdict(StrEnum):
    """One rule's answer.

    `PASS` and `BLOCK` are obvious. `NOT_APPLICABLE` is the one that earns its
    place: a rule that does not apply to this order (a holding-period check on
    an entry, say) must say so rather than returning `PASS`. A ledger full of
    passes from rules that never ran would read as far more scrutiny than
    actually happened, and "17 of 17 rules passed" is a claim someone will
    eventually rely on.

    `WARN` never blocks. It exists so a rule can record a margin worth looking
    at — approaching a cap, an unusual order count — without either failing the
    order or being invisible.
    """

    # S105 reads `PASS = "..."` as a hardcoded password. It is a verdict.
    PASS = "pass"  # noqa: S105
    WARN = "warn"
    BLOCK = "block"
    NOT_APPLICABLE = "not_applicable"

    @property
    def blocks(self) -> bool:
        return self is Verdict.BLOCK


@dataclass(frozen=True, slots=True)
class RuleVerdict:
    """A rule's opinion, with the numbers that produced it.

    `limit_value` and `observed_value` are stored as a pair even on a pass.
    A rule that passed at 99% of its limit is a different fact from one that
    passed at 10%, and only the pair can tell them apart months later — which
    is the difference between "the caps were never close" and "the caps were
    holding it back all week".
    """

    rule_name: str
    verdict: Verdict
    detail: str = ""
    limit_value: Decimal | float | str | None = None
    observed_value: Decimal | float | str | None = None
    # A rule may declare itself advisory. Kept on the verdict rather than on
    # the rule so a rule can block in one circumstance and merely warn in
    # another without needing two rules.
    is_blocking: bool = True
    # Set by a sizing rule that will accept the order at a smaller quantity.
    # `None` means "no opinion on size"; the engine takes the minimum of every
    # opinion offered.
    max_quantity: Decimal | None = None

    @property
    def blocks(self) -> bool:
        return self.verdict.blocks and self.is_blocking

    def summary(self) -> str:
        if self.limit_value is None and self.observed_value is None:
            return f"{self.rule_name}: {self.verdict.value}"
        return (
            f"{self.rule_name}: {self.verdict.value} "
            f"(observed {self.observed_value}, limit {self.limit_value})"
        )


@dataclass(frozen=True, slots=True)
class OrderRequest:
    """The order being evaluated, before the engine has sized it.

    `quantity` is what the caller *wants*; sizing rules may reduce it. It is
    optional because an entry usually arrives with no quantity at all — the
    engine derives one from the caps — while an exit arrives with the exact
    position size, and those are genuinely different requests.
    """

    t212_ticker: str
    instrument_uid: str
    side: Side
    purpose: OrderPurpose
    action: Action
    reference_price: Decimal
    quantity: Decimal | None = None
    expected_edge_bps: Decimal | None = None
    decision_id: str | None = None
    strategy_id: str | None = None

    def __post_init__(self) -> None:
        if self.reference_price <= 0:
            raise RiskError(
                f"{self.t212_ticker}: non-positive reference price {self.reference_price}. "
                "Every cap here is a notional, and a notional computed from a "
                "non-positive price would size an unbounded order."
            )
        if self.quantity is not None and self.quantity <= 0:
            raise RiskError(
                f"{self.t212_ticker}: requested quantity {self.quantity} is not positive. "
                "Pass None to have the engine size it; zero is not a request."
            )

    @property
    def is_risk_increasing(self) -> bool:
        """Whether this order can add exposure.

        Keyed on `purpose` rather than on `side`, because the mapping is not
        one-to-one: a BUY is risk-increasing, but a SELL that is a rebalance
        can be too, and `OrderPurpose.is_risk_reducing` already encodes the
        venue-specific answer.
        """
        return not self.purpose.is_risk_reducing


@dataclass(frozen=True, slots=True)
class AccountState:
    """The account as the reconciler last saw it.

    Every field is optional because the broker genuinely may not tell us. That
    is the whole reason this type exists rather than passing an
    `AccountSnapshot` around: a rule needs to distinguish "deployed 4%" from
    "we do not know what is deployed", and the second must fail closed. A
    missing equity value with a `0` default would make every percentage cap
    evaluate against zero and pass trivially.
    """

    equity_ccy: Decimal | None = None
    free_cash_ccy: Decimal | None = None
    blocked_cash_ccy: Decimal | None = None
    deployed_ccy: Decimal | None = None
    n_open_positions: int = 0
    currency: str | None = None
    # Realised plus unrealised, as a percentage of the day's opening equity.
    # Negative is a loss.
    day_pnl_pct: float | None = None
    rolling_5d_pnl_pct: float | None = None
    drawdown_from_peak_pct: float | None = None
    # Populated from the intents table, not from the broker: what *we* sent is
    # the thing the anomaly breaker is about, and the broker's order list is
    # both rate-limited and incomplete.
    orders_today: int = 0
    orders_today_for_symbol: int = 0
    orders_last_hour: int = 0
    notional_today_ccy: Decimal = Decimal(0)
    # Percentile history for the anomaly rules. `None` on a cold start, which
    # is exactly when a new bug is most likely to be running — so the rules
    # fall back to the absolute bound rather than skipping.
    p95_orders_per_day: float | None = None
    p95_notional_per_day_ccy: Decimal | None = None

    @property
    def deployed_pct(self) -> float | None:
        if self.equity_ccy is None or self.deployed_ccy is None or self.equity_ccy <= 0:
            return None
        return float(self.deployed_ccy / self.equity_ccy * 100)


@dataclass(frozen=True, slots=True)
class RiskContext:
    """Everything a rule may read, fixed at one instant.

    Assembled once per decision by the engine. A rule receives this and
    returns a verdict; it cannot reach a broker, a clock or the ledger, which
    is what makes the rule set testable and what stops a rule from having a
    different opinion on its second evaluation.
    """

    as_of: datetime
    limits: HardLimits
    request: OrderRequest
    account: AccountState
    # Existing position in this instrument, if any.
    position_quantity: Decimal = Decimal(0)
    position_entry_at: datetime | None = None
    # From the symbol map. Two separate answers, because they are genuinely
    # asymmetric: a data problem must be able to block an entry without ever
    # blocking an exit.
    may_enter: bool = False
    may_enter_reason: str = ""
    may_exit: bool = True
    may_exit_reason: str = ""
    permits_full_size: bool = False
    # From the regime gate. `None` means the gate was never consulted, which
    # is a programming error rather than a market state — the engine refuses
    # it rather than assuming full exposure.
    regime_exposure_factor: Decimal | None = None
    regime_state: str | None = None
    # From the data layer.
    bar_age_seconds: float | None = None
    cross_venue_disagreement_bps: float | None = None
    # From the calendar: minutes since the session opened and until it closes.
    minutes_since_open: int | None = None
    minutes_until_close: int | None = None
    # Set when the run is halted or the kill switch is engaged. Carried in the
    # context so the refusal is recorded as a verdict like any other, rather
    # than as an exception that leaves no row.
    halted: bool = False
    halt_reason: str = ""
    # Broker-declared instrument constraints. A protective stop rejected for
    # being under the minimum leaves a naked position, so sizing has to know.
    min_trade_quantity: Decimal | None = None
    max_open_quantity: Decimal | None = None
    extra: dict[str, str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.as_of.tzinfo is None:
            raise RiskError("RiskContext.as_of must be timezone-aware")

    @property
    def is_entry(self) -> bool:
        return self.request.is_risk_increasing

    @property
    def has_position(self) -> bool:
        return self.position_quantity > 0


@runtime_checkable
class RiskRule(Protocol):
    """One rule. Pure function of the context.

    A protocol rather than a base class so a rule can be a plain function,
    and so the engine's rule list is data rather than a class hierarchy.
    """

    @property
    def name(self) -> str: ...

    def evaluate(self, ctx: RiskContext) -> RuleVerdict: ...
