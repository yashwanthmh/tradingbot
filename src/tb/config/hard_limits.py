"""Typed model of `config/hard_limits.yaml`.

Note the asymmetry against the broker adapter built in M1, which deliberately
ignores unknown fields: here every model is `extra="forbid"` and `frozen=True`.

That is the point. An unknown field arriving from a beta API is normal and must
not take the system down. An unknown field in *our own* limits file means
someone wrote `max_orders_per_day_` or `daily_halt_percent` and believes a cap
is in force that is not. Fail-closed on our own config; fail-open on unknown
fields from someone else's.
"""

from __future__ import annotations

from decimal import ROUND_HALF_UP, Decimal, InvalidOperation
from typing import Annotated, Literal

from pydantic import BaseModel, BeforeValidator, ConfigDict, Field, model_validator

_CENTS = Decimal("0.01")


def _to_money(value: object) -> Decimal:
    """Coerce a YAML scalar to a 2dp Decimal.

    Routed through `str` so a YAML float never contributes its binary
    representation to a monetary limit.
    """
    if isinstance(value, Decimal):
        amount = value
    elif isinstance(value, (int, float, str)):
        try:
            amount = Decimal(str(value))
        except InvalidOperation as exc:
            raise ValueError(f"not a valid monetary amount: {value!r}") from exc
    else:
        raise ValueError(f"not a valid monetary amount: {value!r}")

    if not amount.is_finite():
        raise ValueError(f"monetary amount must be finite: {value!r}")
    return amount.quantize(_CENTS, rounding=ROUND_HALF_UP)


Money = Annotated[Decimal, BeforeValidator(_to_money), Field(ge=Decimal("0"))]
Percent = Annotated[float, Field(gt=0.0, le=100.0)]
Factor = Annotated[float, Field(ge=0.0, le=1.0)]


class _Section(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class CapitalLimits(_Section):
    """How much money the bot may have at risk, in total and per position."""

    # Independent of account equity. Percentages scale with the account; this
    # does not, which is what makes it a blast radius rather than a preference.
    absolute_ceiling_ccy: Money
    max_deployed_pct: Percent
    per_position_pct: Percent
    max_positions: int = Field(ge=1, le=100)
    floor_notional_ccy: Money

    @model_validator(mode="after")
    def _check_coherent(self) -> CapitalLimits:
        if self.per_position_pct > self.max_deployed_pct:
            raise ValueError(
                f"per_position_pct ({self.per_position_pct}) exceeds max_deployed_pct "
                f"({self.max_deployed_pct}): a single position could breach the total cap"
            )
        if self.floor_notional_ccy > self.absolute_ceiling_ccy:
            raise ValueError(
                f"floor_notional_ccy ({self.floor_notional_ccy}) exceeds "
                f"absolute_ceiling_ccy ({self.absolute_ceiling_ccy}): the smallest "
                "permitted order would breach the total ceiling, so nothing could trade"
            )
        return self


class LossLimits(_Section):
    """The circuit breakers, and the lifetime budgets that retire a strategy."""

    daily_halt_pct: Percent
    rolling_5d_halt_pct: Percent
    max_drawdown_flatten_pct: Percent
    per_strategy_budget_ccy: Money
    per_lineage_budget_ccy: Money

    @model_validator(mode="after")
    def _check_ordering(self) -> LossLimits:
        if self.daily_halt_pct > self.rolling_5d_halt_pct:
            raise ValueError(
                f"daily_halt_pct ({self.daily_halt_pct}) exceeds rolling_5d_halt_pct "
                f"({self.rolling_5d_halt_pct}): the rolling breaker could never fire first"
            )
        if self.rolling_5d_halt_pct > self.max_drawdown_flatten_pct:
            raise ValueError(
                f"rolling_5d_halt_pct ({self.rolling_5d_halt_pct}) exceeds "
                f"max_drawdown_flatten_pct ({self.max_drawdown_flatten_pct}): the "
                "flatten-everything breaker must sit outside the stop-opening breakers"
            )
        if self.per_strategy_budget_ccy > self.per_lineage_budget_ccy:
            raise ValueError(
                f"per_strategy_budget_ccy ({self.per_strategy_budget_ccy}) exceeds "
                f"per_lineage_budget_ccy ({self.per_lineage_budget_ccy}): a single "
                "strategy could outlive the budget of the lineage that spawned it"
            )
        return self


class ExecutionLimits(_Section):
    """Cost discipline, runaway containment, and cross-venue sanity."""

    min_holding_minutes: int = Field(ge=0)
    max_cost_to_edge_ratio: float = Field(gt=0.0, le=1.0)
    max_orders_per_day: int = Field(ge=1)
    max_orders_per_symbol_per_day: int = Field(ge=1)
    max_universe_symbols: int = Field(ge=1, le=200)
    unprotected_gap_pct_assumption: Percent
    max_cross_venue_disagreement_bps: float = Field(gt=0.0)
    max_bar_staleness_seconds: int = Field(ge=1)
    no_entry_first_minutes: int = Field(ge=0, le=120)
    no_entry_last_minutes: int = Field(ge=0, le=120)

    @model_validator(mode="after")
    def _check_coherent(self) -> ExecutionLimits:
        if self.max_orders_per_symbol_per_day > self.max_orders_per_day:
            raise ValueError(
                f"max_orders_per_symbol_per_day ({self.max_orders_per_symbol_per_day}) "
                f"exceeds max_orders_per_day ({self.max_orders_per_day})"
            )
        # Trading 212 caps pending orders at 50 per ticker per account. Staying
        # well under that is not optional: hitting it means a protective stop
        # gets rejected, which turns a cap breach into an unhedged position.
        if self.max_orders_per_symbol_per_day > 25:
            raise ValueError(
                f"max_orders_per_symbol_per_day ({self.max_orders_per_symbol_per_day}) "
                "is too close to Trading 212's 50-pending-orders-per-ticker ceiling; "
                "exhausting it would cause protective stops to be rejected"
            )
        return self


class AnomalyLimits(_Section):
    """Runaway-loop containment.

    Most catastrophic losses from automated trading are not bad predictions,
    they are a loop that placed ten thousand orders in four minutes.
    """

    halt_if_daily_orders_exceed_p95_by: float = Field(gt=1.0)
    halt_if_daily_notional_exceeds_p95_by: float = Field(gt=1.0)
    # Cold-start backstop: on day one there is no percentile history to compare
    # against, which is precisely when a new bug is most likely to be running.
    absolute_max_orders_per_hour: int = Field(ge=1)


class RegimeLimits(_Section):
    """Portfolio-wide exposure scaling.

    Long-only means every strategy is a long-equity beta expression, so in a
    drawdown their correlation goes to one and per-strategy caps stop helping.
    This sits above the allocator for that reason: a strategy can be killed, a
    regime cannot.
    """

    reference_symbol: str = Field(min_length=1, max_length=20)
    exposure_ma_days: int = Field(ge=20, le=400)
    exposure_factor_below_ma: Factor


class PromotionLimits(_Section):
    """The gate between a generated strategy and real money."""

    paper_shadow_sessions: int = Field(ge=0)
    paper_shadow_min_trades: int = Field(ge=0)

    min_oos_deflated_sharpe: float
    max_oos_drawdown_pct: Percent
    min_oos_trades: int = Field(ge=1)
    max_probability_of_backtest_overfitting: float = Field(gt=0.0, le=1.0)
    max_null_promotion_rate: float = Field(ge=0.0, le=1.0)

    ratchet_min_days_between_promotions: int = Field(ge=1)
    ratchet_rungs_lost_on_breach: int = Field(ge=1)
    ratchet_max_rung: int = Field(ge=1, le=20)


class DataLimits(_Section):
    """What the bot is allowed to believe about its own prices.

    Trading 212 serves no market data, so the feed is a separate vendor with
    separate failure modes — and two of them are easy to miss. A *delayed* feed
    creates a lookahead exactly the size of the delay, because the earliest
    actionable moment for a bar is `bar_close + delay + ingest` rather than
    bar close. An *unrepresentative* feed is worse and quieter: Alpaca's free
    tier is IEX only, a few percent of consolidated volume, so its systematic
    disagreement with the consolidated tape sits in the same 5-20bps band as
    the entire gross edge being traded. The first is bounded here by
    `max_provider_delay_seconds`; the second by `min_edge_to_feed_noise_ratio`
    and by `allowed_live_resolutions`.
    """

    max_provider_delay_seconds: int = Field(ge=1)
    # Daily only until the bake-off produces evidence. Kept as a list so
    # widening it is a visible, reviewable edit rather than a flag flip.
    allowed_live_resolutions: list[str] = Field(min_length=1)
    min_edge_to_feed_noise_ratio: float = Field(gt=0.0)
    min_history_days_for_regime: int = Field(ge=1)
    max_unexplained_gap_pct: Percent
    # Prices are stored as integers scaled by 10^price_scale. Floats would
    # round-trip through Parquet and change every hash computed over them.
    price_scale: int = Field(ge=2, le=12)
    backfill_years_daily: int = Field(ge=1, le=50)
    backfill_days_minute: int = Field(ge=1, le=3650)
    revision_canary_sample_pct: float = Field(ge=0.0, le=100.0)

    @model_validator(mode="after")
    def _check_resolutions(self) -> DataLimits:
        known = {"daily", "hourly", "minute"}
        unknown = [r for r in self.allowed_live_resolutions if r not in known]
        if unknown:
            raise ValueError(
                f"allowed_live_resolutions contains unknown values {unknown}; "
                f"known resolutions are {sorted(known)}"
            )
        return self

    def permits_live(self, resolution: str) -> bool:
        return resolution in self.allowed_live_resolutions


class SafetyLimits(_Section):
    """Kill switch, heartbeat, and what to do about state we cannot explain."""

    kill_switch_path: str = Field(min_length=1)
    heartbeat_path: str = Field(min_length=1)
    heartbeat_stale_seconds: int = Field(ge=5)
    on_unprotected_position: Literal["flatten", "protect"]
    halt_on_unreconciled: bool


class HardLimits(_Section):
    """The whole control layer, as validated values."""

    schema_version: int = Field(ge=1)
    # Asserted against the base currency the broker reports for the account. A
    # ceiling of 500 means something very different in GBP than in JPY.
    currency: str = Field(pattern=r"^[A-Z]{3}$")

    capital: CapitalLimits
    loss: LossLimits
    execution: ExecutionLimits
    anomaly: AnomalyLimits
    regime: RegimeLimits
    promotion: PromotionLimits
    data: DataLimits
    safety: SafetyLimits

    @model_validator(mode="after")
    def _check_cross_section(self) -> HardLimits:
        # The unprotected window is unavoidable: Trading 212 has no bracket
        # orders, so an entry fill always precedes its protective stop. Size
        # must therefore be small enough that a gap across that window fits
        # inside the daily loss budget.
        gap = self.execution.unprotected_gap_pct_assumption
        worst_case_loss_pct = self.capital.per_position_pct * gap / 100.0
        if worst_case_loss_pct > self.loss.daily_halt_pct:
            raise ValueError(
                f"a position of {self.capital.per_position_pct}% of equity suffering the "
                f"assumed {gap}% unprotected gap would lose {worst_case_loss_pct:.2f}% of "
                f"equity, which exceeds daily_halt_pct ({self.loss.daily_halt_pct}%). "
                "Trading 212 has no bracket orders, so this window is real: reduce "
                "per_position_pct, or raise the daily limit deliberately."
            )
        return self

    @property
    def worst_case_unprotected_loss_pct(self) -> float:
        """Equity lost if one position gaps through its unprotected window."""
        return self.capital.per_position_pct * self.execution.unprotected_gap_pct_assumption / 100.0

    @model_validator(mode="after")
    def _check_data_coherence(self) -> HardLimits:
        # The staleness bound and the provider delay bound describe the same
        # window from two sides. A data layer that admitted a feed slower than
        # the decision layer will act on would fetch bars it is then forbidden
        # to use — a system that looks alive and never trades.
        if self.data.max_provider_delay_seconds > self.execution.max_bar_staleness_seconds:
            raise ValueError(
                f"data.max_provider_delay_seconds ({self.data.max_provider_delay_seconds}s) "
                f"exceeds execution.max_bar_staleness_seconds "
                f"({self.execution.max_bar_staleness_seconds}s): a bar admitted by the data "
                "layer would then be refused by the decision layer, so the bot would fetch "
                "prices it can never act on"
            )
        # A 200-day moving average needs roughly 200 sessions. If the regime
        # gate could run on less, "insufficient history" would quietly become
        # "no signal" — which reads as full exposure, the most expensive
        # possible default.
        if self.data.min_history_days_for_regime < self.regime.exposure_ma_days:
            raise ValueError(
                f"data.min_history_days_for_regime ({self.data.min_history_days_for_regime}) is "
                f"below regime.exposure_ma_days ({self.regime.exposure_ma_days}): the regime "
                "gate would be permitted to run on history too short to compute its own "
                "moving average"
            )
        return self
