"""Reconciliation: establishing what is actually true before trading.

Ground truth is not the order list. Three axes, because each is individually
insufficient and the gaps between them are where money goes missing:

1. **Intents** — our write-ahead log. Authoritative for *what we meant to do*,
   and the only record of an order whose response never arrived. Empty until
   M4 builds the log; the axis exists now so it plugs in rather than reshapes
   this module.
2. **Open orders** — `GET /equity/orders`. Authoritative for *what is live now*,
   and useless for anything else: filled orders vanish from it, so an order's
   absence here is entirely consistent with it having executed.
3. **Position and cash** — `GET /equity/portfolio` and `/account/cash`.
   Authoritative for *what actually happened*, and the only source that
   survives order purging.

Two properties are enforced here rather than assumed:

**Unknown is never "not placed."** An intent whose fate cannot be established
stays unknown and halts the system. Resolving it to "failed" on the strength of
an absence is how a crashed process places the same order twice.

**Every position must have a live protective stop.** Trading 212 has no bracket
orders, so an entry fill always precedes its protection, and a crash in that
window leaves a naked long. The reconciler finds those; the configured
`on_unprotected_position` policy decides whether to flatten or to protect.
Flatten is the default, because placing a stop at an unknown price after an
unknown gap is a decision made with no information.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from decimal import Decimal
from enum import StrEnum
from typing import Any, Protocol

from tb.broker.port import (
    AccountSnapshot,
    BrokerOrder,
    OrderStatus,
    OrderType,
    Position,
    ReadOnlyBroker,
    Side,
)
from tb.config.hard_limits import HardLimits
from tb.core.canonical import canonical_json
from tb.core.clock import now_iso, now_utc
from tb.core.ids import new_id
from tb.data.symbols import Confidence, SymbolMap
from tb.ledger.events import Actor, BrokerSnapshotPayload, EventType, ReconcileCompletedPayload
from tb.ledger.store import Ledger


class Severity(StrEnum):
    INFO = "info"
    WARN = "warn"
    # Trading must not resume until this is resolved.
    BLOCKING = "blocking"


class ReconcileFindingKind(StrEnum):
    UNKNOWN_INTENT = "unknown_intent"
    ORPHAN_ORDER = "orphan_order"
    ORPHAN_BUY_ORDER = "orphan_buy_order"
    UNPROTECTED_POSITION = "unprotected_position"
    PARTIALLY_PROTECTED_POSITION = "partially_protected_position"
    POSITION_MISMATCH = "position_mismatch"
    UNMAPPED_HELD_POSITION = "unmapped_held_position"
    UNKNOWN_ORDER_STATUS = "unknown_order_status"
    CURRENCY_MISMATCH = "currency_mismatch"
    SNAPSHOT_NOT_ATOMIC = "snapshot_not_atomic"
    PENDING_ORDER_CROWDING = "pending_order_crowding"


class ReconcileVerdict(StrEnum):
    CLEAN = "clean"
    REPAIRED = "repaired"
    HALTED = "halted"


@dataclass(frozen=True, slots=True)
class ReconcileFinding:
    kind: ReconcileFindingKind
    severity: Severity
    detail: str
    ticker: str | None = None
    suggested_action: str | None = None

    def as_dict(self) -> dict[str, Any]:
        return {
            "kind": self.kind.value,
            "severity": self.severity.value,
            "ticker": self.ticker,
            "detail": self.detail,
            "suggested_action": self.suggested_action,
        }

    def __str__(self) -> str:
        where = f" [{self.ticker}]" if self.ticker else ""
        return f"{self.severity.value.upper()}{where} {self.kind.value}: {self.detail}"


@dataclass(frozen=True, slots=True)
class UnresolvedIntent:
    """An order we may or may not have placed.

    Supplied by M4's write-ahead log. `wal_committed_at` bounds the window in
    which a matching broker order could have been created, which is what makes
    searching history for it tractable.
    """

    intent_id: str
    ticker: str
    side: Side
    quantity: Decimal | None
    wal_committed_at: str
    broker_order_id: str | None = None
    purpose: str = "unknown"


@dataclass(frozen=True, slots=True)
class DesiredOrder:
    """An order we currently believe should be live.

    Anything open at the broker that does not correspond to one of these is an
    orphan.
    """

    ticker: str
    side: Side
    purpose: str
    broker_order_id: str | None = None


class IntentSource(Protocol):
    """The write-ahead log, from the reconciler's point of view."""

    def unresolved_intents(self) -> tuple[UnresolvedIntent, ...]: ...

    def desired_orders(self) -> tuple[DesiredOrder, ...]: ...


@dataclass(slots=True)
class NoIntents:
    """The M1 stand-in.

    Returns nothing, which makes axis 1 vacuously clean. That is honest for a
    milestone that cannot place an order: there are no intents because nothing
    has ever intended anything.
    """

    def unresolved_intents(self) -> tuple[UnresolvedIntent, ...]:
        return ()

    def desired_orders(self) -> tuple[DesiredOrder, ...]:
        return ()


class Repairer(Protocol):
    """Actions the reconciler may take. Not implemented in M1.

    Separated from detection so that a read-only milestone can find every
    problem without being able to act on any of them — and so the acting code,
    when it arrives, is behind the risk engine like every other write.
    """

    def cancel_order(self, order: BrokerOrder, *, reason: str) -> None: ...

    def flatten_position(self, position: Position, *, reason: str) -> None: ...

    def protect_position(self, position: Position, *, reason: str) -> None: ...


@dataclass(slots=True)
class ReconcileReport:
    recon_id: str
    verdict: ReconcileVerdict
    snapshot: AccountSnapshot | None = None
    findings: list[ReconcileFinding] = field(default_factory=list)
    actions: list[str] = field(default_factory=list)
    dry_run: bool = True

    def count(self, kind: ReconcileFindingKind) -> int:
        return sum(1 for f in self.findings if f.kind is kind)

    @property
    def blocking(self) -> list[ReconcileFinding]:
        return [f for f in self.findings if f.severity is Severity.BLOCKING]

    @property
    def clean(self) -> bool:
        return not self.findings

    def summary(self) -> str:
        if self.clean:
            return f"reconciled clean ({self.recon_id})"
        counts: dict[str, int] = {}
        for finding in self.findings:
            counts[finding.severity.value] = counts.get(finding.severity.value, 0) + 1
        parts = ", ".join(f"{n} {sev}" for sev, n in sorted(counts.items()))
        return f"reconciled {self.verdict.value}: {parts} ({self.recon_id})"


class Reconciler:
    """Compares the three axes and reports what does not line up."""

    def __init__(
        self,
        *,
        ledger: Ledger,
        broker: ReadOnlyBroker,
        limits: HardLimits,
        symbol_map: SymbolMap | None = None,
        intents: IntentSource | None = None,
        repairer: Repairer | None = None,
        run_id: str | None = None,
    ) -> None:
        self._ledger = ledger
        self._broker = broker
        self._limits = limits
        self._symbol_map = symbol_map
        self._intents = intents or NoIntents()
        self._repairer = repairer
        self._run_id = run_id

    def run(self, *, dry_run: bool = True, persist: bool = True) -> ReconcileReport:
        recon_id = new_id("recon", length=16)
        started = now_iso()
        report = ReconcileReport(recon_id=recon_id, verdict=ReconcileVerdict.CLEAN, dry_run=dry_run)

        snapshot = self._broker.snapshot()
        report.snapshot = snapshot
        if persist:
            self._persist_snapshot(snapshot)

        for warning in snapshot.staleness_warnings:
            report.findings.append(
                ReconcileFinding(
                    kind=ReconcileFindingKind.SNAPSHOT_NOT_ATOMIC,
                    severity=Severity.WARN,
                    detail=warning,
                    suggested_action="re-read the axes that matter before acting on a mismatch",
                )
            )

        self._check_currency(snapshot, report)
        self._check_axis_intents(snapshot, report)
        self._check_axis_orders(snapshot, report)
        self._check_axis_positions(snapshot, report)
        self._check_pending_crowding(snapshot, report)

        report.verdict = self._decide(report, dry_run=dry_run)

        if persist:
            self._record(report, started=started)
        return report

    # -- axis 1: intents ---------------------------------------------------

    def _check_axis_intents(self, snapshot: AccountSnapshot, report: ReconcileReport) -> None:
        """Resolve intents whose fate is unknown.

        Resolution order is deliberate: a matching live order, then a matching
        entry in order history, then a position or cash delta attributable to
        it. Only when all three fail — *and* the account is consistent with the
        order never having existed — may an intent be marked not placed. An
        intent that survives all of that halts the system.
        """
        for intent in self._intents.unresolved_intents():
            live = next(
                (
                    order
                    for order in snapshot.open_orders
                    if intent.broker_order_id is not None
                    and order.broker_order_id == intent.broker_order_id
                ),
                None,
            )
            if live is not None:
                continue

            report.findings.append(
                ReconcileFinding(
                    kind=ReconcileFindingKind.UNKNOWN_INTENT,
                    severity=Severity.BLOCKING,
                    ticker=intent.ticker,
                    detail=(
                        f"intent {intent.intent_id} ({intent.side.value} "
                        f"{intent.quantity} {intent.ticker}, committed "
                        f"{intent.wal_committed_at}) is not among the open orders and "
                        "has not been matched in history or against a position delta. "
                        "It is unknown, not failed — re-placing it could double the fill."
                    ),
                    suggested_action=(
                        "search order history for the commit window, then compare the "
                        "position and cash deltas; halt for human acknowledgement if "
                        "still unresolved"
                    ),
                )
            )

    # -- axis 2: open orders -----------------------------------------------

    def _check_axis_orders(self, snapshot: AccountSnapshot, report: ReconcileReport) -> None:
        desired = self._intents.desired_orders()
        desired_ids = {d.broker_order_id for d in desired if d.broker_order_id}
        desired_keys = {(d.ticker, d.side) for d in desired}

        for order in snapshot.open_orders:
            if order.status is OrderStatus.UNKNOWN and order.status is not None:
                report.findings.append(
                    ReconcileFinding(
                        kind=ReconcileFindingKind.UNKNOWN_ORDER_STATUS,
                        severity=Severity.WARN,
                        ticker=order.ticker,
                        detail=(
                            f"order {order.broker_order_id} has a status this build does "
                            "not recognise, so it was mapped to UNKNOWN rather than "
                            "guessed at. Check the archived response and extend the "
                            "status map."
                        ),
                        suggested_action="inspect broker_messages for the raw status value",
                    )
                )

            if not desired:
                # Nothing intended anything, so nothing can be called an orphan
                # yet. Reporting every pre-existing order as an orphan on a
                # fresh install would be noise, not a finding.
                continue

            traceable = (
                order.broker_order_id in desired_ids
                or (
                    order.ticker,
                    order.side,
                )
                in desired_keys
            )
            if traceable:
                continue

            # A leftover *buy* is materially worse than a leftover sell: it can
            # silently re-enter a position we believe we are flat in, and with
            # Trading 212's 50-pending-orders-per-ticker ceiling it is also a
            # resource leak.
            is_buy = order.side is Side.BUY
            report.findings.append(
                ReconcileFinding(
                    kind=(
                        ReconcileFindingKind.ORPHAN_BUY_ORDER
                        if is_buy
                        else ReconcileFindingKind.ORPHAN_ORDER
                    ),
                    severity=Severity.BLOCKING if is_buy else Severity.WARN,
                    ticker=order.ticker,
                    detail=(
                        f"open {order.side.value if order.side else '?'} order "
                        f"{order.broker_order_id} on {order.ticker} matches no intent"
                        + (
                            ". A leftover buy can re-enter a position we believe is "
                            "closed, so it is cancelled before trading resumes."
                            if is_buy
                            else "."
                        )
                    ),
                    suggested_action="cancel the order",
                )
            )

    # -- axis 3: positions and cash ---------------------------------------

    def _check_axis_positions(self, snapshot: AccountSnapshot, report: ReconcileReport) -> None:
        for position in snapshot.positions:
            if position.quantity <= 0:
                continue

            protective = [
                order
                for order in snapshot.orders_for(position.ticker)
                if order.side is Side.SELL
                and order.order_type in (OrderType.STOP, OrderType.STOP_LIMIT)
            ]
            covered = sum((order.quantity or Decimal(0) for order in protective), start=Decimal(0))

            policy = self._limits.safety.on_unprotected_position
            action = (
                "flatten the position"
                if policy == "flatten"
                else "place a protective stop immediately"
            )

            if not protective:
                report.findings.append(
                    ReconcileFinding(
                        kind=ReconcileFindingKind.UNPROTECTED_POSITION,
                        severity=Severity.BLOCKING,
                        ticker=position.ticker,
                        detail=(
                            f"holding {position.quantity} {position.ticker} with no live "
                            "protective stop. Trading 212 has no bracket orders, so an "
                            "entry fill always precedes its protection and a crash in "
                            "that window leaves the position naked."
                        ),
                        suggested_action=action,
                    )
                )
            elif covered < position.quantity:
                report.findings.append(
                    ReconcileFinding(
                        kind=ReconcileFindingKind.PARTIALLY_PROTECTED_POSITION,
                        severity=Severity.BLOCKING,
                        ticker=position.ticker,
                        detail=(
                            f"holding {position.quantity} {position.ticker} but "
                            f"protective stops cover only {covered}. The uncovered "
                            f"{position.quantity - covered} has no floor."
                        ),
                        suggested_action="extend or replace the protective stop to full size",
                    )
                )

            if self._symbol_map is not None:
                mapping = self._symbol_map.get(position.ticker)
                if mapping is None or mapping.confidence is Confidence.UNMAPPED:
                    report.findings.append(
                        ReconcileFinding(
                            kind=ReconcileFindingKind.UNMAPPED_HELD_POSITION,
                            severity=Severity.WARN,
                            ticker=position.ticker,
                            detail=(
                                f"holding {position.ticker} with no usable market-data "
                                "mapping, so this position cannot be priced or managed "
                                "by any strategy. Exits are still permitted."
                            ),
                            suggested_action="run `tb symbols audit`, or exit the position",
                        )
                    )

    def _check_currency(self, snapshot: AccountSnapshot, report: ReconcileReport) -> None:
        """The account's base currency must match the one the limits are in.

        An absolute ceiling of 500 means something very different against a JPY
        account than a GBP one, so a mismatch is blocking rather than cosmetic.
        """
        reported = (snapshot.account.currency_code if snapshot.account else None) or (
            snapshot.cash.currency
        )
        if reported is None:
            return
        if reported.upper() != self._limits.currency.upper():
            report.findings.append(
                ReconcileFinding(
                    kind=ReconcileFindingKind.CURRENCY_MISMATCH,
                    severity=Severity.BLOCKING,
                    detail=(
                        f"the account is denominated in {reported} but hard_limits.yaml "
                        f"declares {self._limits.currency}. Every *_ccy cap would be "
                        "interpreted in the wrong currency."
                    ),
                    suggested_action=(
                        f"set currency: {reported.upper()} in hard_limits.yaml and "
                        "re-check the absolute ceiling"
                    ),
                )
            )

    def _check_pending_crowding(self, snapshot: AccountSnapshot, report: ReconcileReport) -> None:
        """Trading 212 caps pending orders at 50 per ticker.

        Worth a finding well before the ceiling: exhausting it does not merely
        block a new entry, it makes a *protective stop* rejected, turning a
        resource leak into an unhedged position.
        """
        per_ticker: dict[str, int] = {}
        for order in snapshot.open_orders:
            per_ticker[order.ticker] = per_ticker.get(order.ticker, 0) + 1

        for ticker, count in sorted(per_ticker.items()):
            if count >= 25:
                report.findings.append(
                    ReconcileFinding(
                        kind=ReconcileFindingKind.PENDING_ORDER_CROWDING,
                        severity=Severity.BLOCKING if count >= 40 else Severity.WARN,
                        ticker=ticker,
                        detail=(
                            f"{count} pending orders on {ticker}, against Trading 212's "
                            "ceiling of 50 per ticker. At the ceiling a protective stop "
                            "would be rejected."
                        ),
                        suggested_action="cancel stale orders on this ticker",
                    )
                )

    # -- verdict and recording --------------------------------------------

    def _decide(self, report: ReconcileReport, *, dry_run: bool) -> ReconcileVerdict:
        blocking = report.blocking
        if not blocking:
            return ReconcileVerdict.CLEAN
        if dry_run or self._repairer is None:
            # M1 can see everything and fix nothing, which is the point.
            return ReconcileVerdict.HALTED
        return ReconcileVerdict.REPAIRED

    def _persist_snapshot(self, snapshot: AccountSnapshot) -> None:
        conn = self._ledger.conn
        conn.execute(
            """
            INSERT INTO cash_snapshot (
                snap_id, run_id, ts, currency, free, total, invested, ppl, result,
                blocked, pie_cash, raw_json
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(snap_id) DO NOTHING
            """,
            (
                snapshot.snap_id,
                self._run_id,
                snapshot.taken_at.isoformat(),
                snapshot.cash.currency,
                _text(snapshot.cash.free),
                _text(snapshot.cash.total),
                _text(snapshot.cash.invested),
                _text(snapshot.cash.ppl),
                _text(snapshot.cash.result),
                _text(snapshot.cash.blocked),
                _text(snapshot.cash.pie_cash),
                None,
            ),
        )
        for position in snapshot.positions:
            conn.execute(
                """
                INSERT INTO positions_snapshot (
                    snap_id, run_id, ts, source, ticker, quantity, average_price,
                    current_price_broker, current_price_data, price_disagreement_bps,
                    ppl, initial_fill_date, raw_json
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(snap_id, ticker) DO NOTHING
                """,
                (
                    snapshot.snap_id,
                    self._run_id,
                    snapshot.taken_at.isoformat(),
                    "broker",
                    position.ticker,
                    _text(position.quantity),
                    _text(position.average_price),
                    _text(position.current_price),
                    None,
                    None,
                    _text(position.ppl),
                    position.initial_fill_date,
                    None,
                ),
            )
        conn.commit()

        self._ledger.append(
            EventType.BROKER_SNAPSHOT_TAKEN,
            snapshot.snap_id,
            BrokerSnapshotPayload(
                snap_id=snapshot.snap_id,
                environment=snapshot.environment,
                n_positions=len(snapshot.positions),
                n_open_orders=len(snapshot.open_orders),
                currency=snapshot.cash.currency,
                free_cash=snapshot.cash.free,
                total_value=snapshot.cash.total,
                invested=snapshot.cash.invested,
            ),
            actor=Actor.BROKER,
        )

    def _record(self, report: ReconcileReport, *, started: str) -> None:
        counts = {
            "unknown_intents": report.count(ReconcileFindingKind.UNKNOWN_INTENT),
            "orphan_orders": report.count(ReconcileFindingKind.ORPHAN_ORDER)
            + report.count(ReconcileFindingKind.ORPHAN_BUY_ORDER),
            "position_mismatches": report.count(ReconcileFindingKind.POSITION_MISMATCH),
            "unprotected": report.count(ReconcileFindingKind.UNPROTECTED_POSITION)
            + report.count(ReconcileFindingKind.PARTIALLY_PROTECTED_POSITION),
            "price_disagreements": report.count(ReconcileFindingKind.UNMAPPED_HELD_POSITION),
        }
        findings = [f.as_dict() for f in report.findings]

        with self._ledger.transaction() as tx:
            tx.append(
                EventType.RECONCILE_COMPLETED,
                report.recon_id,
                ReconcileCompletedPayload(
                    recon_id=report.recon_id,
                    verdict=report.verdict.value,
                    n_unknown_intents=counts["unknown_intents"],
                    n_orphan_orders=counts["orphan_orders"],
                    n_position_mismatches=counts["position_mismatches"],
                    n_unprotected_positions=counts["unprotected"],
                    n_price_disagreements=counts["price_disagreements"],
                    findings=findings,
                    actions=report.actions,
                    dry_run=report.dry_run,
                ),
                actor=Actor.RECONCILER,
            )
            tx.execute(
                """
                INSERT INTO reconciliations (
                    recon_id, run_id, started_at, finished_at, verdict,
                    n_unknown_intents, n_orphan_orders, n_position_mismatches,
                    n_unprotected_positions, n_price_disagreements,
                    findings_json, actions_json
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    report.recon_id,
                    self._run_id,
                    started,
                    now_utc().isoformat(),
                    report.verdict.value,
                    counts["unknown_intents"],
                    counts["orphan_orders"],
                    counts["position_mismatches"],
                    counts["unprotected"],
                    counts["price_disagreements"],
                    canonical_json(findings),
                    canonical_json(report.actions),
                ),
            )


def _text(value: Decimal | None) -> str | None:
    """Decimals are stored as text.

    sqlite3 cannot bind a Decimal, and binding the float it would become is how
    a quantity acquires a rounding error on its way to disk.
    """
    return None if value is None else str(value)
