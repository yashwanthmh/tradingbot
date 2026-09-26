"""The regime gate: one number that scales every strategy's exposure at once.

Long-only and unlevered means every live strategy is a long-equity beta
expression. In a drawdown their correlation goes to one, so per-strategy caps
stop helping precisely when they are needed — a strategy can be killed, but a
regime cannot. Hence a single factor applied above the allocator: when the
reference index is below its long moving average, every strategy's gross
exposure is scaled by `regime.exposure_factor_below_ma`.

**The property that matters is the cold start, and it is inverted from the
obvious one.** A 200-day moving average needs roughly ten months of sessions.
Before that exists there is no signal — and "no signal" must mean *reduced*
exposure, never full. Reading absence as permission is the most expensive
default available anywhere in this system: it applies on day one, to the whole
portfolio, at exactly the moment a new deployment is least likely to be
correct about anything.

So `RegimeState` has four members, not two, and only one of them permits full
exposure:

| state | exposure |
|---|---|
| `RISK_ON` — index above its MA | full |
| `RISK_OFF` — index below its MA | reduced |
| `INSUFFICIENT_HISTORY` — fewer sessions than the MA needs | **reduced** |
| `UNAVAILABLE` — no data, or the newest bar is stale | **reduced** |

`exposure_factor` is `1` in exactly one of those four cases, and a test
asserts that by enumeration rather than by inspection.

**The reference series gets its own identity.** It cannot live in `symbol_map`,
whose primary key is `t212_ticker`, because it is not something the bot trades
— and it can never be broker-verified, because verification requires holding a
position in it. So it is keyed by the self-marking `sym:` uid, and it is
deliberately *not* subject to the symbol map's tradability gate: a reference
series that had to be tradable to be readable would make the regime gate
unavailable on a fresh install, which by the rule above means permanently
halved exposure.

Point-in-time throughout. The moving average is computed from
`visible_bars(as_of=...)` through the same `FeaturePipeline` every strategy
uses, so a regime reading in a backtest is the reading that was actually
available then — not the one computed from today's data.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from datetime import datetime
from decimal import Decimal
from enum import StrEnum

from tb.config.hard_limits import HardLimits
from tb.core.errors import TbError
from tb.data.adjustments import CorporateAction
from tb.data.asof import (
    UNKNOWN,
    BarSource,
    BarWindow,
    Unknown,
    assess_staleness,
    visible_bars,
)
from tb.data.provider import Bar, Resolution, make_instrument_uid
from tb.features.pipeline import FeaturePipeline, make_spec


class RegimeError(TbError):
    """A regime reading could not be produced, so exposure cannot be scaled."""


class RegimeState(StrEnum):
    """Why exposure is what it is.

    Four members rather than two, because the two failure modes are not the
    same fact as `RISK_OFF` and conflating them would hide a data outage
    behind a market signal. Both still reduce exposure — see the module
    docstring — but an operator needs to be able to tell "the index is down"
    from "we cannot see the index".
    """

    RISK_ON = "risk_on"
    RISK_OFF = "risk_off"
    INSUFFICIENT_HISTORY = "insufficient_history"
    UNAVAILABLE = "unavailable"

    @property
    def permits_full_exposure(self) -> bool:
        """Only one state does.

        Written as an explicit identity check rather than `self is not
        RISK_OFF` so that adding a fifth state defaults it to *reduced*. A new
        member that silently inherited full exposure is the bug this file
        exists to prevent.
        """
        return self is RegimeState.RISK_ON

    @property
    def is_measured(self) -> bool:
        """Whether this reading reflects the market rather than our own blindness."""
        return self in (RegimeState.RISK_ON, RegimeState.RISK_OFF)


@dataclass(frozen=True, slots=True)
class RegimeReading:
    """The factor, and everything needed to explain it later."""

    as_of: datetime
    state: RegimeState
    exposure_factor: Decimal
    reference_symbol: str
    instrument_uid: str
    ma_days: int
    n_sessions_seen: int
    last_close: Decimal | None = None
    moving_average: Decimal | None = None
    detail: str = ""

    def __post_init__(self) -> None:
        # The invariant, enforced at construction rather than trusted at the
        # call sites. Every path that builds a reading goes through here, so a
        # future branch that forgot to reduce cannot produce a full-exposure
        # reading from a non-RISK_ON state.
        if self.exposure_factor == Decimal(1) and not self.state.permits_full_exposure:
            raise RegimeError(
                f"{self.state.value} produced a full-exposure factor. Only "
                "RISK_ON may, and reading absence of signal as permission to "
                "take full exposure is the most expensive default in this system."
            )
        if not (Decimal(0) <= self.exposure_factor <= Decimal(1)):
            raise RegimeError(
                f"exposure factor {self.exposure_factor} is outside [0, 1]: a factor "
                "above 1 would *increase* exposure above what the allocator sized."
            )

    @property
    def reduced(self) -> bool:
        return self.exposure_factor < Decimal(1)

    def scale(self, notional: Decimal) -> Decimal:
        """Apply the factor to a sized notional."""
        return notional * self.exposure_factor

    def summary(self) -> str:
        return (
            f"{self.reference_symbol} {self.state.value}: exposure x"
            f"{self.exposure_factor} ({self.detail})"
        )


@dataclass(frozen=True, slots=True)
class RegimeGate:
    """Reads the reference series and produces the exposure factor.

    Holds the limits rather than loose numbers, so the factor cannot drift from
    the pinned config, and holds no clock: `as_of` is always passed in, which
    is what lets a backtest ask for the reading that was available at a past
    instant rather than the one computed from today's data.
    """

    limits: HardLimits
    resolution: Resolution = Resolution.DAILY

    @property
    def reference_symbol(self) -> str:
        return self.limits.regime.reference_symbol

    @property
    def instrument_uid(self) -> str:
        """The reference series' own identity.

        `sym:` rather than `isin:` deliberately. The reference series is not
        traded, so it has no broker instrument record to take an ISIN from, and
        `make_instrument_uid`'s `sym:` fallback is already self-marking for
        exactly this case — an audit looking for ticker-keyed identities finds
        it, and the trading path never resolves to it.
        """
        return make_instrument_uid(data_symbol=self.reference_symbol)

    @property
    def ma_days(self) -> int:
        return self.limits.regime.exposure_ma_days

    @property
    def reduced_factor(self) -> Decimal:
        return Decimal(str(self.limits.regime.exposure_factor_below_ma))

    @property
    def min_sessions(self) -> int:
        """Sessions required before the gate will say RISK_ON at all.

        `data.min_history_days_for_regime`, which the config validator already
        refuses to let fall below `exposure_ma_days` — a gate permitted to run
        on history shorter than its own moving average would produce a number
        that looks like a signal and is not.
        """
        return self.limits.data.min_history_days_for_regime

    # -- the reading -------------------------------------------------------

    def read(
        self,
        source: BarSource,
        *,
        as_of: datetime,
        check_staleness: bool = True,
        actions: Sequence[CorporateAction] = (),
    ) -> RegimeReading:
        """The exposure factor as of one instant.

        Every failure path below returns a *reduced* reading rather than
        raising, because raising would leave the caller to decide — and the
        caller deciding is how "the regime gate was unavailable so we traded
        full size" happens. The one thing this raises on is a naive datetime,
        which is a programming error rather than a data condition.

        `actions` are the reference series' corporate actions, unfiltered, as
        `FeaturePipeline.compute` takes them. Without them a split in the index
        fund puts its 200-day average four times above the price for most of a
        year: RISK_OFF, and half exposure, on a number that is not a signal.
        """
        if as_of.tzinfo is None:
            raise RegimeError("as_of must be timezone-aware")

        bars = visible_bars(
            source,
            self.instrument_uid,
            self.resolution,
            as_of=as_of,
        )
        if not bars:
            return self._unavailable(
                as_of=as_of,
                n_seen=0,
                detail=(
                    f"no {self.reference_symbol} bars knowable at "
                    f"{as_of.isoformat()}. Exposure is reduced rather than full: "
                    "not seeing the index is not the same as the index being fine."
                ),
            )

        if check_staleness:
            verdict = assess_staleness(
                bars[-1],
                now=as_of,
                limit_seconds=self._staleness_limit_seconds(),
            )
            if not verdict.fresh:
                return self._unavailable(
                    as_of=as_of,
                    n_seen=len(bars),
                    detail=(
                        f"newest {self.reference_symbol} bar is stale: {verdict.reason}. "
                        "A regime reading from an old bar is not a regime reading."
                    ),
                    last_close=bars[-1].close,
                )

        if len(bars) < self.min_sessions:
            return RegimeReading(
                as_of=as_of,
                state=RegimeState.INSUFFICIENT_HISTORY,
                exposure_factor=self.reduced_factor,
                reference_symbol=self.reference_symbol,
                instrument_uid=self.instrument_uid,
                ma_days=self.ma_days,
                n_sessions_seen=len(bars),
                last_close=bars[-1].close,
                detail=(
                    f"{len(bars)} sessions of {self.reference_symbol}, under the "
                    f"{self.min_sessions} a {self.ma_days}-day average needs. "
                    "Exposure stays reduced: 'not enough data' must never read as "
                    "'no signal, so go ahead'."
                ),
            )

        window = BarWindow(
            as_of=as_of,
            resolution=self.resolution,
            _by_uid={self.instrument_uid: bars},
        )
        pipeline = self._pipeline()
        snapshot = pipeline.compute(window, self.instrument_uid, actions=actions)
        average = snapshot.get(f"sma_{self.ma_days}")
        last = snapshot.get("close")

        if average is UNKNOWN or last is UNKNOWN:
            # Reachable when the session count clears `min_sessions` but the
            # pipeline still refuses — belt to the length check above rather
            # than dead code, since the two thresholds are separate config
            # values and could be edited apart.
            return self._unavailable(
                as_of=as_of,
                n_seen=len(bars),
                detail=(f"the {self.ma_days}-day average is unavailable over {len(bars)} sessions"),
                last_close=bars[-1].close,
            )

        return self._measured(
            as_of=as_of,
            n_seen=len(bars),
            last=_as_decimal(last),
            average=_as_decimal(average),
        )

    def read_bars(
        self,
        bars: list[Bar],
        *,
        as_of: datetime,
        check_staleness: bool = True,
        actions: Sequence[CorporateAction] = (),
    ) -> RegimeReading:
        """Convenience for callers that already hold the reference bars."""
        from tb.data.asof import InMemoryBarSource

        return self.read(
            InMemoryBarSource(bars=bars),
            as_of=as_of,
            check_staleness=check_staleness,
            actions=actions,
        )

    # -- internals ---------------------------------------------------------

    def _pipeline(self) -> FeaturePipeline:
        """One spec for the average, one for the latest close.

        Through the shared `FeaturePipeline` rather than a private mean: that
        gets the refuse-a-short-window behaviour, the Decimal arithmetic and
        the split adjustment for free, and it means the regime gate cannot
        disagree with a strategy about what a 200-day average is.
        """
        return FeaturePipeline(
            specs=(
                make_spec("last", 1, name="close"),
                make_spec("sma", self.ma_days),
            )
        )

    def _staleness_limit_seconds(self) -> int:
        """How old the newest reference bar may be.

        Not `execution.max_bar_staleness_seconds` — that bound is for the
        decision path, where a three-minute-old price matters. A daily regime
        signal is legitimately a day old, so the bound is generous in wall
        time and tight in *sessions*: four days covers a long weekend plus a
        holiday, and beyond that the index we are reading is not the index the
        market is trading.
        """
        return 4 * 24 * 60 * 60

    def _measured(
        self, *, as_of: datetime, n_seen: int, last: Decimal, average: Decimal
    ) -> RegimeReading:
        above = last > average
        state = RegimeState.RISK_ON if above else RegimeState.RISK_OFF
        return RegimeReading(
            as_of=as_of,
            state=state,
            exposure_factor=Decimal(1) if above else self.reduced_factor,
            reference_symbol=self.reference_symbol,
            instrument_uid=self.instrument_uid,
            ma_days=self.ma_days,
            n_sessions_seen=n_seen,
            last_close=last,
            moving_average=average,
            detail=(
                f"{self.reference_symbol} {last} "
                f"{'above' if above else 'below'} its {self.ma_days}-day average "
                f"{average}"
            ),
        )

    def _unavailable(
        self,
        *,
        as_of: datetime,
        n_seen: int,
        detail: str,
        last_close: Decimal | None = None,
    ) -> RegimeReading:
        return RegimeReading(
            as_of=as_of,
            state=RegimeState.UNAVAILABLE,
            exposure_factor=self.reduced_factor,
            reference_symbol=self.reference_symbol,
            instrument_uid=self.instrument_uid,
            ma_days=self.ma_days,
            n_sessions_seen=n_seen,
            last_close=last_close,
            detail=detail,
        )


def _as_decimal(value: Decimal | Unknown) -> Decimal:
    """Narrow a feature value that has already been checked for UNKNOWN."""
    if not isinstance(value, Decimal):  # pragma: no cover - guarded by the caller
        raise RegimeError("expected a Decimal feature value")
    return value
