"""Realised results, trade by trade, and the strategy each belongs to.

Settlement records fills; this turns each closing fill into a realised result
and charges it to the strategy that traded it — the number the review's
KEEP/KILL/SCALE, the allocator's realised edge and the lineage loss budgets all
read. Until this existed nothing called `record_realised`: every strategy showed
zero trades forever, no loss ever reached a lineage budget, and the review could
only ever say the evidence was thin.

**Average cost, from our own fills.** A sell realises `quantity * (price -
average cost)` less its own charges, and a buy's charges are part of its cost.
Walked over the instrument's fills in the order they were recorded, starting
afresh each time the holding returns to flat — the venue's own average-price
arithmetic, over prices the venue reported for fills we can name.

**Only measured numbers are charged.** If a fill the result depends on has no
reported price — the entry's or the exit's — the result is recorded as
inadmissible and charged to nobody: a guessed basis would teach the allocator an
edge nobody earned, or spend a lineage's budget on a loss nobody measured. So is
a sell for more than the recorded buys cover, whose basis is unknown.

**The strategy is the closing order's, not whoever holds the ticker now.** An
exit carries the decision that asked for it and a protective stop the entry
decision it protects; only a flatten has neither, and it belongs to the entry it
closes. Asking "who owns this ticker" at settlement instead would charge the
wrong strategy whenever history lagged past a new entry.

**Charged, then recorded.** The strategy's record is updated before the round
trip is written, so a crash between the two re-attributes the fill next cycle
and charges it twice rather than never. Twice exhausts a lineage early; never
lets it trade past its budget — and only one of those is the safe direction.

Prices are in the instrument's currency and charges in the account's, and no
conversion is applied — the same simplification the sizing path makes. Exact
for an account and universe in one currency; for a GBP account trading USD
shares the result is in dollars, labelled as account currency.
"""

from __future__ import annotations

import json
import sqlite3
from dataclasses import dataclass, replace
from datetime import datetime
from decimal import Decimal, InvalidOperation

from tb.core.clock import to_iso
from tb.ledger.events import Actor, EventType, TradeClosedPayload
from tb.ledger.store import Ledger
from tb.registry.lineage import SpecRegistry


@dataclass(frozen=True, slots=True)
class RoundTrip:
    """The realised result of one closing fill."""

    closing_fill_id: str
    t212_ticker: str
    quantity: Decimal
    admissible: bool
    strategy_id: str | None = None
    strategy_version: int | None = None
    exit_price: Decimal | None = None
    # Per share, charges on the buys included.
    cost_basis: Decimal | None = None
    pnl_ccy: Decimal | None = None
    closed_at: str | None = None
    detail: str = ""
    charged: bool = False

    @property
    def label(self) -> str:
        if self.strategy_id is None:
            return "no strategy"
        return f"{self.strategy_id}@v{self.strategy_version}"


def unattributed_closing_fills(ledger: Ledger) -> tuple[str, ...]:
    """Sell fills with no round trip recorded yet, oldest first."""
    rows = ledger.conn.execute(
        "SELECT f.fill_id AS fill_id FROM fills f"
        " LEFT JOIN round_trips r ON r.closing_fill_id = f.fill_id"
        " WHERE f.side = 'sell' AND r.closing_fill_id IS NULL"
        " ORDER BY f.recording_event_seq"
    ).fetchall()
    return tuple(str(row["fill_id"]) for row in rows)


def round_trip(ledger: Ledger, *, fill_id: str) -> RoundTrip:
    """What one sell fill realised, from the fills recorded before it."""
    target = ledger.conn.execute("SELECT * FROM fills WHERE fill_id = ?", (fill_id,)).fetchone()
    if target is None:
        raise ValueError(f"no fill {fill_id!r}")
    ticker = str(target["t212_ticker"])
    quantity = Decimal(str(target["quantity"]))
    exit_price = _price(target)
    owner = _owner(ledger, target["intent_id"])

    def result(
        *, admissible: bool, detail: str, basis: Decimal | None = None, pnl: Decimal | None = None
    ) -> RoundTrip:
        return RoundTrip(
            closing_fill_id=fill_id,
            t212_ticker=ticker,
            quantity=quantity,
            admissible=admissible,
            strategy_id=None if owner is None else owner[0],
            strategy_version=None if owner is None else owner[1],
            exit_price=exit_price,
            cost_basis=basis,
            pnl_ccy=pnl,
            closed_at=None if target["filled_at"] is None else str(target["filled_at"]),
            detail=detail,
        )

    held = Decimal(0)
    cost = Decimal(0)
    measured = True
    for row in ledger.conn.execute(
        "SELECT * FROM fills WHERE t212_ticker = ? AND recording_event_seq < ?"
        " ORDER BY recording_event_seq",
        (ticker, target["recording_event_seq"]),
    ):
        amount = Decimal(str(row["quantity"]))
        if row["side"] == "buy":
            if held <= 0:
                held, cost, measured = Decimal(0), Decimal(0), True
            held += amount
            price = _price(row)
            if price is None:
                measured = False
            else:
                cost += amount * price + _charges(row["fees_json"])
            continue
        if held <= 0:
            continue
        cost -= cost * min(amount, held) / held
        held -= amount
        if held <= 0:
            held, cost, measured = Decimal(0), Decimal(0), True

    if held <= 0 or quantity > held:
        return result(
            admissible=False,
            detail=(
                f"sold {quantity} with {max(held, Decimal(0))} recorded as bought: the basis "
                "of the rest is unknown, so nothing is charged"
            ),
        )
    if not measured:
        return result(
            admissible=False, detail="an entry has no reported price, so its basis is a guess"
        )
    basis = cost / held
    if exit_price is None:
        return result(
            admissible=False,
            basis=basis,
            detail="the exit has no reported price, so the result would be a guess",
        )
    pnl = quantity * (exit_price - basis) - _charges(target["fees_json"])
    return result(
        admissible=True,
        basis=basis,
        pnl=pnl,
        detail=f"{quantity} at {exit_price} against an average cost of {basis:.6f}",
    )


def attribute_closed_trades(
    ledger: Ledger, *, registry: SpecRegistry, run_id: str, at: datetime
) -> tuple[RoundTrip, ...]:
    """Charge every unattributed closing fill to its strategy, and record it.

    Idempotent by construction: a fill with a round trip recorded is not
    reached again. A strategy that is not in the registry — a hand-written one
    run with `--strategy trivial` — has no record to charge, and the round trip
    says so.
    """
    recorded: list[RoundTrip] = []
    for fill_id in unattributed_closing_fills(ledger):
        trip = round_trip(ledger, fill_id=fill_id)
        charged = False
        detail = trip.detail
        if trip.admissible and trip.pnl_ccy is not None:
            if trip.strategy_id is None or trip.strategy_version is None:
                detail = "no strategy's decision stands behind this order, so nobody is charged"
            elif registry.status_of(trip.strategy_id, trip.strategy_version) is None:
                detail = f"{trip.label} is not in the registry, so it has no record to charge"
            else:
                registry.record_realised(
                    trip.strategy_id,
                    version=trip.strategy_version,
                    pnl_ccy=trip.pnl_ccy,
                    n_trades=1,
                    at=at,
                )
                charged = True
        settled = replace(trip, detail=detail, charged=charged)
        _record(ledger, settled, run_id=run_id, at=at)
        recorded.append(settled)
    return tuple(recorded)


def _record(ledger: Ledger, trip: RoundTrip, *, run_id: str, at: datetime) -> None:
    with ledger.transaction() as tx:
        event = tx.append(
            EventType.TRADE_CLOSED,
            trip.strategy_id or trip.t212_ticker,
            TradeClosedPayload(
                closing_fill_id=trip.closing_fill_id,
                run_id=run_id,
                t212_ticker=trip.t212_ticker,
                quantity=trip.quantity,
                admissible=trip.admissible,
                charged=trip.charged,
                strategy_id=trip.strategy_id,
                strategy_version=trip.strategy_version,
                exit_price=trip.exit_price,
                cost_basis=trip.cost_basis,
                pnl_ccy=trip.pnl_ccy,
                closed_at=trip.closed_at,
                detail=trip.detail,
            ),
            actor=Actor.SYSTEM,
            run_id=run_id,
        )
        tx.execute(
            "INSERT INTO round_trips (closing_fill_id, t212_ticker, strategy_id,"
            " strategy_version, quantity, exit_price, cost_basis, pnl_ccy, admissible,"
            " charged, detail, closed_at, recording_event_seq)"
            " VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)",
            (
                trip.closing_fill_id,
                trip.t212_ticker,
                trip.strategy_id,
                trip.strategy_version,
                str(trip.quantity),
                None if trip.exit_price is None else str(trip.exit_price),
                None if trip.cost_basis is None else str(trip.cost_basis),
                None if trip.pnl_ccy is None else str(trip.pnl_ccy),
                1 if trip.admissible else 0,
                1 if trip.charged else 0,
                trip.detail,
                trip.closed_at or to_iso(at),
                event.seq,
            ),
        )


def _owner(ledger: Ledger, intent_id: object) -> tuple[str, int] | None:
    """The strategy whose decision stands behind a closing order."""
    if intent_id is None:
        return None
    intent = ledger.conn.execute(
        "SELECT decision_id, t212_ticker, committing_event_seq FROM order_intents"
        " WHERE intent_id = ?",
        (str(intent_id),),
    ).fetchone()
    if intent is None:
        return None
    decision = intent["decision_id"]
    if decision is None:
        # A flatten answers no decision. It closes the most recent entry that
        # may have opened what it sold.
        entry = ledger.conn.execute(
            "SELECT decision_id FROM order_intents WHERE t212_ticker = ? AND purpose = 'entry'"
            " AND committing_event_seq < ? AND state IN ('acknowledged', 'resolved_filled')"
            " ORDER BY committing_event_seq DESC LIMIT 1",
            (intent["t212_ticker"], intent["committing_event_seq"]),
        ).fetchone()
        decision = None if entry is None else entry["decision_id"]
    if decision is None:
        return None
    row = ledger.conn.execute(
        "SELECT strategy_id, strategy_version FROM decisions WHERE decision_id = ?",
        (str(decision),),
    ).fetchone()
    return None if row is None else (str(row["strategy_id"]), int(row["strategy_version"]))


def _price(row: sqlite3.Row) -> Decimal | None:
    """A fill's price, or `None` when it is not one realised P&L may use."""
    if not row["admissible_for_pnl"] or row["price"] is None:
        return None
    return Decimal(str(row["price"]))


def _charges(fees_json: object) -> Decimal:
    """The venue's charges on one fill, summed. Unreadable entries count as none."""
    if not fees_json:
        return Decimal(0)
    try:
        fees = json.loads(str(fees_json))
    except ValueError:
        return Decimal(0)
    total = Decimal(0)
    for amount in fees.values() if isinstance(fees, dict) else ():
        try:
            total += abs(Decimal(str(amount)))
        except InvalidOperation:
            continue
    return total
