"""What a round trip actually costs, and whether any edge survives it.

The single most consequential module in the backtester, because on this venue
cost is the binding constraint rather than alpha. Trading 212 is
commission-free, which is not the same as free:

| charge | when | rate |
|---|---|---|
| FX conversion | every currency crossing, both directions | 0.15% |
| UK stamp duty | purchases of UK-incorporated shares only | 0.50% |
| Irish stamp duty | purchases of Irish-incorporated shares | 1.00% |
| French FTT | purchases of large French issuers | 0.30% |
| half-spread | entry and exit | assumption |
| slippage | entry and exit | assumption |

A GBP account buying a US name pays FX twice plus two half-spreads plus two
slippages: roughly 30bps before the market has moved at all. The same account
buying a UK name skips the FX and pays 50bps of stamp duty on the way in:
roughly 55bps. Gross edge on minute-bar signals in liquid names is 5-20bps.

That arithmetic — not a preference about turnover — is why this system targets
minute-resolution *features* with hour-to-day position changes, and why
`expected_cost_bps / expected_edge_bps` is a pre-trade rejection rather than a
post-hoc report.

Three decisions worth stating, because each is a way the model could flatter a
strategy:

**Stamp duty is charged on the buy only, and never on a sale.** Charging it
symmetrically would overstate cost by 50bps a round trip on UK names, which
sounds conservative but is not: it would push the search loop away from the
cheaper venue for a GBP account and towards paying FX instead.

**The published rates live here, as code.** They are facts about the world, not
policy choices, so they are not in `hard_limits.yaml` where an operator could
edit one. What *is* in the limits file is the part nobody publishes — the
spread and slippage a market order really pays — because that is a judgement
call, and understating it is the direction that makes trading look free.

**A computed cost below `min_round_trip_cost_bps` is raised to it.** A backstop
against this module's own arithmetic. Every other error here is recoverable;
"the model said it was free" is how a strategy gets funded.

Costs are `Decimal` throughout. A basis point of a £15 floor order is
£0.00015, and float accumulation over ten thousand backtested trades is the
kind of error that shows up as a small positive Sharpe.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from decimal import ROUND_HALF_EVEN, Decimal, localcontext
from enum import StrEnum

from tb.config.hard_limits import HardLimits
from tb.core.errors import TbError
from tb.data.fx import FX_FEE_RATE

# Published transaction taxes, by the jurisdiction of the *issuer* rather than
# the venue: an Irish company traded in London still attracts Irish duty.
# Buys only — none of these is charged on a sale.
UK_STAMP_DUTY_RATE = Decimal("0.005")
IRISH_STAMP_DUTY_RATE = Decimal("0.010")
FRENCH_FTT_RATE = Decimal("0.003")

# Working precision for intermediate products. Wide enough that a basis point
# of a floor-size order does not vanish before it is summed.
_WORKING_PRECISION = 40
_CENTS = Decimal("0.01")
_BPS = Decimal("10000")


class CostError(TbError):
    """A cost could not be computed, so no order may be sized from it."""


class Jurisdiction(StrEnum):
    """Whose transaction tax applies to a purchase.

    Explicit rather than inferred from a ticker suffix at the point of use.
    A `.L` suffix means "listed in London", which is not the same as
    "UK-incorporated" — and the difference is 50bps on every entry.
    """

    UK = "uk"
    IRELAND = "ireland"
    FRANCE = "france"
    US = "us"
    OTHER = "other"
    # Refused rather than assumed. Guessing `OTHER` for an unknown issuer
    # assumes the cheapest possible tax treatment, which is the one direction
    # a cost model must never guess in.
    UNKNOWN = "unknown"

    @property
    def buy_tax_rate(self) -> Decimal:
        if self is Jurisdiction.UK:
            return UK_STAMP_DUTY_RATE
        if self is Jurisdiction.IRELAND:
            return IRISH_STAMP_DUTY_RATE
        if self is Jurisdiction.FRANCE:
            return FRENCH_FTT_RATE
        return Decimal(0)


def jurisdiction_from_isin(isin: str | None) -> Jurisdiction:
    """Map an ISIN's country prefix to a tax jurisdiction.

    The ISIN prefix is the incorporation country, which is exactly the right
    key for a transaction tax — unlike the trading venue, which is what a
    ticker suffix tells you. An absent or malformed ISIN yields `UNKNOWN`
    rather than a cheap default.
    """
    if not isin or len(isin) < 2:
        return Jurisdiction.UNKNOWN
    prefix = isin[:2].upper()
    if not prefix.isalpha():
        return Jurisdiction.UNKNOWN
    return {
        "GB": Jurisdiction.UK,
        "IE": Jurisdiction.IRELAND,
        "FR": Jurisdiction.FRANCE,
        "US": Jurisdiction.US,
    }.get(prefix, Jurisdiction.OTHER)


@dataclass(frozen=True, slots=True)
class CostBreakdown:
    """Every charge on one side of a trade, itemised.

    Itemised rather than totalled because the decisions differ: "this strategy
    dies to stamp duty" points at the universe, "this strategy dies to the
    spread" points at the holding period, and a single number points at
    neither.
    """

    notional_ccy: Decimal
    fx_fee_ccy: Decimal = Decimal(0)
    transaction_tax_ccy: Decimal = Decimal(0)
    half_spread_ccy: Decimal = Decimal(0)
    slippage_ccy: Decimal = Decimal(0)
    jurisdiction: Jurisdiction = Jurisdiction.OTHER
    crossed_currency: bool = False

    @property
    def total_ccy(self) -> Decimal:
        return self.fx_fee_ccy + self.transaction_tax_ccy + self.half_spread_ccy + self.slippage_ccy

    @property
    def total_bps(self) -> Decimal:
        if self.notional_ccy == 0:
            raise CostError("cannot express a cost in bps against a zero notional")
        return (self.total_ccy / self.notional_ccy) * _BPS

    def itemised(self) -> dict[str, str]:
        """For the ledger. Strings, so no float ever enters a stored payload."""
        return {
            "fx_fee": str(self.fx_fee_ccy),
            "transaction_tax": str(self.transaction_tax_ccy),
            "half_spread": str(self.half_spread_ccy),
            "slippage": str(self.slippage_ccy),
            "total": str(self.total_ccy),
            "jurisdiction": self.jurisdiction.value,
        }


@dataclass(frozen=True, slots=True)
class RoundTrip:
    """Entry plus exit, which is the only unit a strategy can be judged on.

    Quoting a one-way cost invites halving it in the head. Every gate in this
    system consumes the round trip.
    """

    entry: CostBreakdown
    exit: CostBreakdown
    floored_to_bps: Decimal | None = None

    @property
    def total_ccy(self) -> Decimal:
        return self.entry.total_ccy + self.exit.total_ccy

    @property
    def total_bps(self) -> Decimal:
        """Cost as bps of the entry notional.

        Against the *entry* notional specifically: that is the capital actually
        committed, and normalising each leg against its own notional would make
        a profitable trade look cheaper per bp than a losing one.
        """
        if self.floored_to_bps is not None:
            return self.floored_to_bps
        if self.entry.notional_ccy == 0:
            raise CostError("cannot express a cost in bps against a zero notional")
        return (self.total_ccy / self.entry.notional_ccy) * _BPS


@dataclass(frozen=True, slots=True)
class CostVerdict:
    """Whether a trade may proceed on cost grounds, and the arithmetic behind it."""

    allowed: bool
    expected_cost_bps: Decimal
    expected_edge_bps: Decimal
    ratio: Decimal | None
    max_ratio: Decimal
    reason: str

    def raise_if_refused(self) -> None:
        if not self.allowed:
            raise CostError(self.reason)


@dataclass(frozen=True, slots=True)
class CostModel:
    """The venue's fee schedule, applied.

    Constructed from `HardLimits` so the assumptions cannot drift from the
    pinned config, and holds no I/O: an FX *rate* is passed in per call rather
    than looked up, because the rate is point-in-time data and this module
    must stay usable from a backtest walking 2019 without a live rate table.
    """

    limits: HardLimits
    _half_spread_rate: Decimal = field(init=False)
    _slippage_rate: Decimal = field(init=False)

    def __post_init__(self) -> None:
        costs = self.limits.costs
        object.__setattr__(
            self, "_half_spread_rate", Decimal(str(costs.assumed_half_spread_bps)) / _BPS
        )
        object.__setattr__(self, "_slippage_rate", Decimal(str(costs.assumed_slippage_bps)) / _BPS)

    @property
    def account_currency(self) -> str:
        return self.limits.currency

    # -- one side ----------------------------------------------------------

    def leg(
        self,
        *,
        notional_ccy: Decimal,
        side: str,
        instrument_currency: str,
        jurisdiction: Jurisdiction,
    ) -> CostBreakdown:
        """Cost of one fill, in the account currency.

        `notional_ccy` is already in the account currency; conversion of the
        price itself belongs to the caller, which holds the point-in-time rate.
        """
        if notional_ccy <= 0:
            raise CostError(f"notional must be positive, got {notional_ccy}")
        if side not in ("buy", "sell"):
            raise CostError(f"side must be 'buy' or 'sell', got {side!r}")
        if jurisdiction is Jurisdiction.UNKNOWN:
            raise CostError(
                "cannot cost a trade in an instrument whose jurisdiction is unknown: "
                "the transaction tax ranges from 0 to 100bps, and assuming the low end "
                "is the one direction a cost model must not guess in. Record the ISIN."
            )

        crossed = instrument_currency.upper() != self.account_currency.upper()

        with localcontext() as ctx:
            ctx.prec = _WORKING_PRECISION
            fx_fee = notional_ccy * FX_FEE_RATE if crossed else Decimal(0)
            # Buys only. Charging it on the exit too would overstate a UK round
            # trip by 50bps and push the search away from the venue that is
            # cheaper for a GBP account.
            tax = notional_ccy * jurisdiction.buy_tax_rate if side == "buy" else Decimal(0)
            spread = notional_ccy * self._half_spread_rate
            slippage = notional_ccy * self._slippage_rate

            quantum = _CENTS
            fx_fee = fx_fee.quantize(quantum, rounding=ROUND_HALF_EVEN)
            tax = tax.quantize(quantum, rounding=ROUND_HALF_EVEN)
            spread = spread.quantize(quantum, rounding=ROUND_HALF_EVEN)
            slippage = slippage.quantize(quantum, rounding=ROUND_HALF_EVEN)

        return CostBreakdown(
            notional_ccy=notional_ccy,
            fx_fee_ccy=fx_fee,
            transaction_tax_ccy=tax,
            half_spread_ccy=spread,
            slippage_ccy=slippage,
            jurisdiction=jurisdiction,
            crossed_currency=crossed,
        )

    # -- both sides --------------------------------------------------------

    def round_trip(
        self,
        *,
        notional_ccy: Decimal,
        instrument_currency: str,
        jurisdiction: Jurisdiction,
        exit_notional_ccy: Decimal | None = None,
    ) -> RoundTrip:
        """Entry plus exit. `exit_notional_ccy` defaults to the entry notional.

        Passing the exit notional matters for a position that moved: the exit
        leg's spread, slippage and FX are charged on what is actually sold, and
        assuming they equal the entry understates cost on a winner.
        """
        entry = self.leg(
            notional_ccy=notional_ccy,
            side="buy",
            instrument_currency=instrument_currency,
            jurisdiction=jurisdiction,
        )
        exit_leg = self.leg(
            notional_ccy=exit_notional_ccy if exit_notional_ccy is not None else notional_ccy,
            side="sell",
            instrument_currency=instrument_currency,
            jurisdiction=jurisdiction,
        )

        trip = RoundTrip(entry=entry, exit=exit_leg)
        floor = Decimal(str(self.limits.costs.min_round_trip_cost_bps))
        if trip.total_bps < floor:
            # The backstop. Not a correction to the arithmetic above — a refusal
            # to believe it when it says trading is cheaper than the floor an
            # operator set.
            return RoundTrip(entry=entry, exit=exit_leg, floored_to_bps=floor)
        return trip

    # -- the gate ----------------------------------------------------------

    def gate(
        self,
        *,
        expected_cost_bps: Decimal,
        expected_edge_bps: Decimal,
    ) -> CostVerdict:
        """The pre-trade cost rejection.

        Three refusals, not one. An edge outside the declarable band is a
        malformed spec and is refused *before* the ratio is computed —
        otherwise a strategy declaring an absurd edge would divide its way
        through the only control that keeps the search loop out of the fee
        trap.
        """
        costs = self.limits.costs
        max_ratio = Decimal(str(self.limits.execution.max_cost_to_edge_ratio))
        floor = Decimal(str(costs.min_expected_edge_bps))
        ceiling = Decimal(str(costs.max_expected_edge_bps))

        if expected_edge_bps <= 0:
            return CostVerdict(
                allowed=False,
                expected_cost_bps=expected_cost_bps,
                expected_edge_bps=expected_edge_bps,
                ratio=None,
                max_ratio=max_ratio,
                reason=(
                    f"declared edge is {expected_edge_bps}bps. A strategy that will not "
                    "declare a positive expected edge cannot be cost-gated, and a trade "
                    "that cannot be cost-gated does not happen."
                ),
            )
        if expected_edge_bps < floor:
            return CostVerdict(
                allowed=False,
                expected_cost_bps=expected_cost_bps,
                expected_edge_bps=expected_edge_bps,
                ratio=None,
                max_ratio=max_ratio,
                reason=(
                    f"declared edge {expected_edge_bps}bps is below "
                    f"costs.min_expected_edge_bps ({floor}bps)"
                ),
            )
        if expected_edge_bps > ceiling:
            return CostVerdict(
                allowed=False,
                expected_cost_bps=expected_cost_bps,
                expected_edge_bps=expected_edge_bps,
                ratio=None,
                max_ratio=max_ratio,
                reason=(
                    f"declared edge {expected_edge_bps}bps exceeds "
                    f"costs.max_expected_edge_bps ({ceiling}bps). The cost gate divides by "
                    "this number, so an unbounded claim would defeat it — an edge this "
                    "large is a malformed spec, not an opportunity."
                ),
            )

        with localcontext() as ctx:
            ctx.prec = _WORKING_PRECISION
            ratio = expected_cost_bps / expected_edge_bps

        if ratio > max_ratio:
            return CostVerdict(
                allowed=False,
                expected_cost_bps=expected_cost_bps,
                expected_edge_bps=expected_edge_bps,
                ratio=ratio,
                max_ratio=max_ratio,
                reason=(
                    f"cost {expected_cost_bps:.1f}bps against declared edge "
                    f"{expected_edge_bps:.1f}bps is a ratio of {ratio:.2f}, over the "
                    f"{max_ratio} limit. The fee schedule eats this trade."
                ),
            )
        return CostVerdict(
            allowed=True,
            expected_cost_bps=expected_cost_bps,
            expected_edge_bps=expected_edge_bps,
            ratio=ratio,
            max_ratio=max_ratio,
            reason=(
                f"cost {expected_cost_bps:.1f}bps / edge {expected_edge_bps:.1f}bps = "
                f"{ratio:.2f}, within {max_ratio}"
            ),
        )

    def gate_trade(
        self,
        *,
        notional_ccy: Decimal,
        instrument_currency: str,
        jurisdiction: Jurisdiction,
        expected_edge_bps: Decimal,
    ) -> tuple[CostVerdict, RoundTrip]:
        """Cost a prospective trade and gate it in one call.

        The form the decision path uses, so the cost that was charged and the
        cost that was gated on cannot differ.
        """
        trip = self.round_trip(
            notional_ccy=notional_ccy,
            instrument_currency=instrument_currency,
            jurisdiction=jurisdiction,
        )
        verdict = self.gate(
            expected_cost_bps=trip.total_bps,
            expected_edge_bps=expected_edge_bps,
        )
        return verdict, trip
