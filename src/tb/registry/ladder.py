"""The size ratchet: up slowly, down fast, and never past the hard ceiling.

A promoted strategy starts at floor notional and earns its way up one rung at a
time. The asymmetry is the whole design:

**Up is slow.** A rung costs at least `ratchet_min_days_between_promotions`
trading days *and* a minimum number of closed trades *and* a non-negative
realised result over them. At floor size with multi-day holds a strategy
produces 10-20 trades a month, so a rung is roughly a month of evidence.

**Down is fast.** A breach drops `ratchet_rungs_lost_on_breach` rungs — two,
not one, and the config refuses to express one. A symmetric ladder would leave
a strategy oscillating around its threshold sitting at maximum size half the
time, which is the opposite of what a ratchet is for.

**The ladder cannot enlarge the blast radius.** `notional_for` intersects the
rung's size with the per-position cap and the absolute currency ceiling, and
the tightest wins. Climbing rungs moves a strategy toward those ceilings and
can never move it past one — raising a ceiling is a human edit to a hash-pinned
file, and nothing here is a way around that.

**Why rungs double rather than step linearly.** A linear ladder from 15 to the
per-position cap would need dozens of rungs to cover the range, and each rung
costs a month. Doubling reaches the cap in four, which is what
`ratchet_max_rung: 4` is sized for: a strategy that keeps working is at full
size in about four months, and one that stops working is back at the floor
after two breaches.
"""

from __future__ import annotations

import sqlite3
from dataclasses import dataclass
from datetime import datetime
from decimal import Decimal
from enum import StrEnum

from tb.config.hard_limits import HardLimits
from tb.core.clock import from_iso, now_utc, to_iso
from tb.core.errors import TbError
from tb.core.ids import new_id
from tb.ledger.events import Actor, EventType, LadderMovePayload
from tb.ledger.store import Ledger

# Closed trades a rung needs before it can be climbed. Days alone are not
# evidence: a strategy that signalled nothing for a fortnight has been at its
# rung for a fortnight and has shown nothing, and a ladder counting only days
# would promote it on the strength of having existed.
MIN_TRADES_PER_RUNG = 5

# Each rung doubles. See the module note: the alternative is a ladder too fine
# to climb in a useful time, given that each rung costs about a month.
RUNG_MULTIPLIER = 2


class LadderError(TbError):
    """A ladder move was refused."""


class Direction(StrEnum):
    UP = "up"
    DOWN = "down"
    HOLD = "hold"


class BreachReason(StrEnum):
    """Why a strategy lost rungs.

    Enumerated rather than free text because the review cycle reads them back:
    a strategy that lost rungs to `RISK_BLOCK` three times is misbehaving in a
    way a losing strategy is not, and the two call for different responses.
    """

    REALISED_LOSS = "realised_loss"
    DRAWDOWN = "drawdown"
    RISK_BLOCK = "risk_block"
    DECAY = "decay"
    MANUAL = "manual"


@dataclass(frozen=True, slots=True)
class RungEvidence:
    """What is known about a strategy's time at its current rung.

    Scoped to the rung, not to the strategy's lifetime. A strategy that made
    money for six months and has lost money since its last promotion should not
    climb on the strength of the six months — the question a rung-up answers is
    "has it held up at *this* size".
    """

    days_at_rung: int
    n_trades_at_rung: int
    realised_pnl_at_rung: Decimal
    breached: bool = False
    breach_reason: BreachReason | None = None

    @property
    def is_profitable(self) -> bool:
        return self.realised_pnl_at_rung > 0


@dataclass(frozen=True, slots=True)
class LadderMove:
    """One rung change, with what justified it."""

    move_id: str
    strategy_id: str
    version: int
    from_rung: int
    to_rung: int
    direction: Direction
    reason: str
    moved_at: datetime
    n_trades_at_move: int | None = None
    days_at_rung: int | None = None
    notional_ccy: Decimal | None = None

    @property
    def rungs_moved(self) -> int:
        return self.to_rung - self.from_rung


def notional_for(rung: int, *, limits: HardLimits, equity_ccy: Decimal | None = None) -> Decimal:
    """What one position at this rung may be worth, in the account currency.

    Three ceilings intersected, tightest wins:

    * the rung's own size, `floor * 2^rung`;
    * `per_position_pct` of account equity, when equity is known;
    * `absolute_ceiling_ccy`, which does not scale with the account and is the
      blast radius rather than a preference.

    Equity is optional and its absence is handled the safe way: with no equity
    observation the percentage cap cannot be evaluated, so only the rung size
    and the absolute ceiling apply — both of which are already bounds. It is
    never treated as unlimited.
    """
    if rung < 0:
        raise LadderError(f"rung must not be negative, got {rung}")
    capital = limits.capital
    size = capital.floor_notional_ccy * Decimal(RUNG_MULTIPLIER**rung)
    ceilings = [size, capital.absolute_ceiling_ccy]
    if equity_ccy is not None and equity_ccy > 0:
        ceilings.append(equity_ccy * Decimal(str(capital.per_position_pct)) / Decimal(100))
    return min(ceilings)


class SizeLadder:
    """Decides rung changes and records them."""

    def __init__(
        self,
        ledger: Ledger,
        *,
        limits: HardLimits,
        run_id: str | None = None,
    ) -> None:
        self._ledger = ledger
        self._limits = limits
        self._run_id = run_id

    # -- deciding ----------------------------------------------------------

    def propose(self, *, current_rung: int, evidence: RungEvidence) -> tuple[Direction, int, str]:
        """Where this strategy's rung should be, and why.

        Pure, so the asymmetry is testable without a database. Returns the
        direction, the target rung and the reason — the reason travels with the
        decision rather than being reconstructed by the caller, because a rung
        change with no recorded justification is the one thing an operator will
        want explained months later.
        """
        promotion = self._limits.promotion

        if evidence.breached:
            target = max(0, current_rung - promotion.ratchet_rungs_lost_on_breach)
            if target == current_rung:
                return (
                    Direction.HOLD,
                    current_rung,
                    "already at the floor rung; a breach cannot demote further",
                )
            reason = (
                evidence.breach_reason.value
                if evidence.breach_reason is not None
                else BreachReason.MANUAL.value
            )
            return (
                Direction.DOWN,
                target,
                f"breach ({reason}): down {current_rung - target} rung(s), "
                "which is deliberately more than a promotion gains",
            )

        if current_rung >= promotion.ratchet_max_rung:
            return (
                Direction.HOLD,
                current_rung,
                f"at the maximum rung ({promotion.ratchet_max_rung})",
            )
        if evidence.days_at_rung < promotion.ratchet_min_days_between_promotions:
            return (
                Direction.HOLD,
                current_rung,
                f"{evidence.days_at_rung} day(s) at this rung, "
                f"{promotion.ratchet_min_days_between_promotions} required",
            )
        if evidence.n_trades_at_rung < MIN_TRADES_PER_RUNG:
            return (
                Direction.HOLD,
                current_rung,
                f"{evidence.n_trades_at_rung} trade(s) at this rung, "
                f"{MIN_TRADES_PER_RUNG} required — days alone are not evidence, since a "
                "strategy that signalled nothing has shown nothing",
            )
        if not evidence.is_profitable:
            return (
                Direction.HOLD,
                current_rung,
                f"realised {evidence.realised_pnl_at_rung} at this rung: a rung is "
                "earned by holding up at this size, not by surviving it",
            )
        return (
            Direction.UP,
            current_rung + 1,
            f"{evidence.n_trades_at_rung} trade(s) over {evidence.days_at_rung} day(s) "
            f"returning {evidence.realised_pnl_at_rung}",
        )

    def review(
        self,
        strategy_id: str,
        *,
        version: int = 1,
        evidence: RungEvidence,
        equity_ccy: Decimal | None = None,
        at: datetime | None = None,
    ) -> LadderMove | None:
        """Apply `propose` to the stored rung. `None` when nothing moves."""
        current = self.rung_of(strategy_id, version)
        direction, target, reason = self.propose(current_rung=current, evidence=evidence)
        if direction is Direction.HOLD:
            return None
        return self._move(
            strategy_id=strategy_id,
            version=version,
            from_rung=current,
            to_rung=target,
            direction=direction,
            reason=reason,
            evidence=evidence,
            equity_ccy=equity_ccy,
            at=at,
        )

    def breach(
        self,
        strategy_id: str,
        *,
        version: int = 1,
        reason: BreachReason,
        detail: str = "",
        equity_ccy: Decimal | None = None,
        at: datetime | None = None,
    ) -> LadderMove | None:
        """Drop rungs immediately, without waiting for a review cycle.

        The fast half of the asymmetry, and the reason it is its own method: a
        breach must not have to wait for whatever schedule reviews run on. A
        strategy already at the floor rung returns `None` — there is nowhere
        further down, and recording a move from 0 to 0 would put a meaningless
        row in the history operators read to see how often a strategy breaches.
        """
        return self.review(
            strategy_id,
            version=version,
            evidence=RungEvidence(
                days_at_rung=0,
                n_trades_at_rung=0,
                realised_pnl_at_rung=Decimal(0),
                breached=True,
                breach_reason=reason,
            ),
            equity_ccy=equity_ccy,
            at=at,
        )

    # -- reading -----------------------------------------------------------

    def rung_of(self, strategy_id: str, version: int = 1) -> int:
        row: sqlite3.Row | None = self._ledger.conn.execute(
            "SELECT rung FROM strategy_status WHERE strategy_id = ? AND version = ?",
            (strategy_id, version),
        ).fetchone()
        if row is None:
            raise LadderError(
                f"{strategy_id}@v{version} is not in the registry, so it has no rung. A "
                "strategy the registry has never heard of has not been promoted, and "
                "sizing one would mean funding something that never passed the gate."
            )
        return int(row["rung"])

    def notional_of(
        self, strategy_id: str, *, version: int = 1, equity_ccy: Decimal | None = None
    ) -> Decimal:
        return notional_for(
            self.rung_of(strategy_id, version), limits=self._limits, equity_ccy=equity_ccy
        )

    def history(self, strategy_id: str, version: int = 1) -> list[LadderMove]:
        """Every rung change, in the order it happened.

        Ordered by the recording event's sequence number, not by `moved_at`.
        Timestamps tie — several moves can share an instant, and a caller
        passing an explicit `at` makes that routine — and the first draft broke
        the tie on `move_id`, which is random, so a strategy's history came back
        shuffled. The event log's `seq` is the system's total order and is the
        only tiebreaker that means anything.
        """
        rows = self._ledger.conn.execute(
            "SELECT * FROM ladder_moves WHERE strategy_id = ? AND version = ? "
            "ORDER BY moving_event_seq",
            (strategy_id, version),
        ).fetchall()
        return [_row_to_move(row) for row in rows]

    # -- writing -----------------------------------------------------------

    def _move(
        self,
        *,
        strategy_id: str,
        version: int,
        from_rung: int,
        to_rung: int,
        direction: Direction,
        reason: str,
        evidence: RungEvidence,
        equity_ccy: Decimal | None,
        at: datetime | None,
    ) -> LadderMove:
        moment = at or now_utc()
        notional = notional_for(to_rung, limits=self._limits, equity_ccy=equity_ccy)
        move = LadderMove(
            move_id=new_id("rung", length=12),
            strategy_id=strategy_id,
            version=version,
            from_rung=from_rung,
            to_rung=to_rung,
            direction=direction,
            reason=reason,
            moved_at=moment,
            n_trades_at_move=evidence.n_trades_at_rung,
            days_at_rung=evidence.days_at_rung,
            notional_ccy=notional,
        )

        with self._ledger.transaction() as tx:
            event = tx.append(
                EventType.LADDER_MOVED,
                strategy_id,
                LadderMovePayload(
                    move_id=move.move_id,
                    strategy_id=strategy_id,
                    version=version,
                    from_rung=from_rung,
                    to_rung=to_rung,
                    direction=direction.value,
                    reason=reason,
                    n_trades_at_move=evidence.n_trades_at_rung,
                    days_at_rung=evidence.days_at_rung,
                    notional_ccy=str(notional),
                ),
                actor=Actor.SYSTEM,
                run_id=self._run_id,
            )
            tx.execute(
                """
                INSERT INTO ladder_moves (
                    move_id, strategy_id, version, from_rung, to_rung, direction,
                    reason, n_trades_at_move, days_at_rung, notional_ccy, moved_at,
                    moving_event_seq
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    move.move_id,
                    strategy_id,
                    version,
                    from_rung,
                    to_rung,
                    direction.value,
                    reason,
                    evidence.n_trades_at_rung,
                    evidence.days_at_rung,
                    str(notional),
                    to_iso(moment),
                    event.seq,
                ),
            )
            tx.execute(
                "UPDATE strategy_status SET rung = ?, rung_changed_at = ?, updated_at = ? "
                "WHERE strategy_id = ? AND version = ?",
                (to_rung, to_iso(moment), to_iso(moment), strategy_id, version),
            )
        return move


def _row_to_move(row: sqlite3.Row) -> LadderMove:
    notional = row["notional_ccy"]
    return LadderMove(
        move_id=str(row["move_id"]),
        strategy_id=str(row["strategy_id"]),
        version=int(row["version"]),
        from_rung=int(row["from_rung"]),
        to_rung=int(row["to_rung"]),
        direction=Direction(str(row["direction"])),
        reason=str(row["reason"]),
        moved_at=from_iso(str(row["moved_at"])),
        n_trades_at_move=(
            None if row["n_trades_at_move"] is None else int(row["n_trades_at_move"])
        ),
        days_at_rung=None if row["days_at_rung"] is None else int(row["days_at_rung"]),
        notional_ccy=None if notional is None else Decimal(str(notional)),
    )
