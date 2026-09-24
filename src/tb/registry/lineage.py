"""Identity, ancestry, and the budget that a rename cannot escape.

Three jobs, together because they share one key. A strategy's identity is its
`(strategy_id, version)`; its ancestry is a `lineage_id` shared with every
mutation of the same original idea; and the lifetime loss budget is charged
against the **lineage**, not the strategy.

**Why the budget has to be per lineage.** A per-strategy budget is defeated by
producing a child, and producing a child is what a searcher does by default
rather than by intent: a strategy exhausts its budget, is retired, and the
next cycle proposes a mutation of it that starts fresh. The lineage carries
the exhaustion, so the child inherits it and `StrategyStatus.BLOCKED` is
reachable for a strategy that has personally lost nothing.

**Deduplication is by `spec_hash`, and it is load-bearing.** Registering the
same tree twice returns the existing registration rather than a second one.
Not for tidiness: the trial count is what every deflated metric divides by, so
one spec proposed under two names must be one entry. The place this shows up
is mutation — a searcher that mutates a spec and happens to invert the
mutation has rediscovered the parent, and without the hash check that lineage's
apparent diversity would be twice its real diversity.

**Registration never promotes.** A registered spec is a `CANDIDATE`, and
nothing in this module can make it anything else; only
`tb.registry.promotion` writes `PROMOTED`. Keeping the two apart is what makes
"how did this get funded" have exactly one answer.
"""

from __future__ import annotations

import json
import sqlite3
from collections.abc import Sequence
from datetime import datetime
from decimal import Decimal
from typing import Any, Protocol

from tb.core.clock import from_iso, now_utc, to_iso
from tb.core.ids import deterministic_id, new_id
from tb.ledger.events import (
    Actor,
    EventType,
    LineageBudgetPayload,
    StrategyRetiredPayload,
    StrategySpecPayload,
)
from tb.ledger.store import Ledger
from tb.registry.models import (
    AuthorKind,
    LineageBudget,
    RegisteredSpec,
    RegistryError,
    StrategyRecord,
    StrategyStatus,
)
from tb.strategy.dsl.schema import StrategySpec


class _Writer(Protocol):
    """Just the part of a ledger transaction a projection write needs.

    Narrower than the transaction context on purpose: a helper that only writes
    projections should not be able to append an event, because an event
    appended from a helper is an event whose reasoning lives nowhere.
    """

    def execute(self, sql: str, params: Sequence[Any] = ()) -> object: ...


class SpecRegistry:
    """The strategy registry, over the ledger.

    `per_lineage_budget_ccy` is passed in from the hard limits rather than read
    from a module constant, because it is a ceiling on losses and every ceiling
    in this system lives in the hash-pinned file.
    """

    def __init__(
        self,
        ledger: Ledger,
        *,
        per_lineage_budget_ccy: Decimal,
        run_id: str | None = None,
    ) -> None:
        self._ledger = ledger
        self._budget = per_lineage_budget_ccy
        self._run_id = run_id

    # -- registration ------------------------------------------------------

    def register(
        self,
        spec: StrategySpec,
        *,
        author_kind: AuthorKind,
        parent_strategy_id: str | None = None,
        lineage_id: str | None = None,
        at: datetime | None = None,
    ) -> RegisteredSpec:
        """Register a spec, or return the existing registration of the same tree.

        `lineage_id` is derived rather than generated when a parent is named:
        a mutation belongs to its parent's lineage, and letting the caller pass
        an arbitrary one would make the budget escapable by the code that is
        supposed to be bound by it.
        """
        moment = at or now_utc()
        spec_hash = spec.spec_hash

        existing = self.by_hash(spec_hash)
        if existing is not None:
            # The same tree. One strategy, one trial — see the module note.
            return existing

        if parent_strategy_id is not None:
            parent = self.latest_version_of(parent_strategy_id)
            if parent is None:
                raise RegistryError(
                    f"parent {parent_strategy_id} is not registered, so the lineage this "
                    "spec claims to belong to cannot be checked. A child whose lineage is "
                    "taken on trust is a child that escapes its lineage's loss budget."
                )
            derived_lineage = parent.lineage_id
            generation = parent.generation + 1
            if lineage_id is not None and lineage_id != derived_lineage:
                raise RegistryError(
                    f"spec names parent {parent_strategy_id} (lineage "
                    f"{derived_lineage}) but asks for lineage {lineage_id}. A mutation "
                    "cannot choose its own lineage: that is exactly how an exhausted "
                    "loss budget gets left behind."
                )
        else:
            derived_lineage = lineage_id or new_id("lin")
            generation = 0

        strategy_id = deterministic_id("stg", parts={"spec_hash": spec_hash}, length=12)
        version = 1

        registered = RegisteredSpec(
            strategy_id=strategy_id,
            version=version,
            lineage_id=derived_lineage,
            spec_hash=spec_hash,
            author_kind=author_kind,
            registered_at=moment,
            parent_strategy_id=parent_strategy_id,
            expected_edge_bps=spec.expected_edge_bps,
            name=spec.name,
            generation=generation,
        )

        with self._ledger.transaction() as tx:
            event = tx.append(
                EventType.STRATEGY_SPEC_REGISTERED,
                strategy_id,
                StrategySpecPayload(
                    strategy_id=strategy_id,
                    lineage_id=derived_lineage,
                    version=version,
                    spec_hash=spec_hash,
                    author_kind=author_kind.value,
                    parent_strategy_id=parent_strategy_id,
                    expected_edge_bps=float(spec.expected_edge_bps),
                    n_operators=spec.n_nodes,
                ),
                actor=_actor_for(author_kind),
                run_id=self._run_id,
            )
            tx.execute(
                """
                INSERT INTO strategy_specs (
                    strategy_id, version, lineage_id, parent_strategy_id, spec_json,
                    spec_hash, author_kind, expected_edge_bps, registered_at,
                    registering_event_seq
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    strategy_id,
                    version,
                    derived_lineage,
                    parent_strategy_id,
                    spec.model_dump_json(),
                    spec_hash,
                    author_kind.value,
                    float(spec.expected_edge_bps),
                    to_iso(moment),
                    event.seq,
                ),
            )
            tx.execute(
                """
                INSERT INTO strategy_status (
                    strategy_id, version, lineage_id, status, rung, updated_at
                ) VALUES (?, ?, ?, ?, 0, ?)
                """,
                (
                    strategy_id,
                    version,
                    derived_lineage,
                    # A freshly registered spec is a candidate, whatever else is
                    # true about it. Nothing in this module can write PROMOTED.
                    StrategyStatus.CANDIDATE.value,
                    to_iso(moment),
                ),
            )
            self._touch_budget(tx, derived_lineage, moment)

        return registered

    # -- reading -----------------------------------------------------------

    def get(self, strategy_id: str, version: int = 1) -> RegisteredSpec | None:
        row: sqlite3.Row | None = self._ledger.conn.execute(
            "SELECT * FROM strategy_specs WHERE strategy_id = ? AND version = ?",
            (strategy_id, version),
        ).fetchone()
        return None if row is None else _row_to_spec(row, self._generation_of(row))

    def by_hash(self, spec_hash: str) -> RegisteredSpec | None:
        row: sqlite3.Row | None = self._ledger.conn.execute(
            "SELECT * FROM strategy_specs WHERE spec_hash = ? ORDER BY version DESC LIMIT 1",
            (spec_hash,),
        ).fetchone()
        return None if row is None else _row_to_spec(row, self._generation_of(row))

    def latest_version_of(self, strategy_id: str) -> RegisteredSpec | None:
        row: sqlite3.Row | None = self._ledger.conn.execute(
            "SELECT * FROM strategy_specs WHERE strategy_id = ? ORDER BY version DESC LIMIT 1",
            (strategy_id,),
        ).fetchone()
        return None if row is None else _row_to_spec(row, self._generation_of(row))

    def spec_of(self, strategy_id: str, version: int = 1) -> StrategySpec | None:
        """The stored spec, re-validated on the way out.

        Re-validated rather than trusted: the row was written by an earlier
        build, and a spec that no longer parses against the current grammar
        must fail loudly here rather than be interpreted approximately.
        """
        row: sqlite3.Row | None = self._ledger.conn.execute(
            "SELECT spec_json FROM strategy_specs WHERE strategy_id = ? AND version = ?",
            (strategy_id, version),
        ).fetchone()
        if row is None:
            return None
        return StrategySpec.parse(json.loads(str(row["spec_json"])))

    def in_lineage(self, lineage_id: str) -> list[RegisteredSpec]:
        rows = self._ledger.conn.execute(
            "SELECT * FROM strategy_specs WHERE lineage_id = ? ORDER BY registered_at",
            (lineage_id,),
        ).fetchall()
        return [_row_to_spec(row, self._generation_of(row)) for row in rows]

    def _generation_of(self, row: sqlite3.Row) -> int:
        """How many mutations deep this spec is.

        Walked from the parent chain rather than stored, because a stored
        generation can disagree with the chain it describes and the chain is
        the fact. The walk is bounded by a visited set: a cycle in the parent
        chain is impossible to create through `register`, but a hand-edited
        table should hang nothing.
        """
        generation = 0
        seen = {str(row["strategy_id"])}
        parent = row["parent_strategy_id"]
        while parent is not None and str(parent) not in seen:
            seen.add(str(parent))
            generation += 1
            next_row: sqlite3.Row | None = self._ledger.conn.execute(
                "SELECT parent_strategy_id FROM strategy_specs WHERE strategy_id = ? "
                "ORDER BY version DESC LIMIT 1",
                (str(parent),),
            ).fetchone()
            if next_row is None:
                break
            parent = next_row["parent_strategy_id"]
        return generation

    # -- status ------------------------------------------------------------

    def status_of(self, strategy_id: str, version: int = 1) -> StrategyRecord | None:
        row: sqlite3.Row | None = self._ledger.conn.execute(
            "SELECT * FROM strategy_status WHERE strategy_id = ? AND version = ?",
            (strategy_id, version),
        ).fetchone()
        return None if row is None else _row_to_record(row)

    def promoted(self) -> list[StrategyRecord]:
        """Every strategy currently permitted to trade.

        Read by the live loop, so it filters on the stored status rather than
        re-deriving eligibility. The derivation happens once, at the gate.
        """
        rows = self._ledger.conn.execute(
            "SELECT * FROM strategy_status WHERE status = ? ORDER BY promoted_at",
            (StrategyStatus.PROMOTED.value,),
        ).fetchall()
        return [_row_to_record(row) for row in rows]

    def blocked_after_promotion(self) -> list[StrategyRecord]:
        """Strategies that were funded once and are blocked now.

        Read beside `promoted()` when assembling the live book, because a
        strategy blocked by its lineage's loss budget has already left
        `promoted()` — so a book built from that alone would show an empty list
        and no reason. "Four were promoted and all four are out of budget" and
        "nothing was ever promoted" are the same picture otherwise, and only one
        of them is about the searcher.

        Retired strategies are deliberately not here. Retirement is the normal
        way out and listing every past strategy on every run would bury the one
        row that means something.
        """
        rows = self._ledger.conn.execute(
            "SELECT * FROM strategy_status WHERE status = ? AND promoted_at IS NOT NULL "
            "ORDER BY promoted_at",
            (StrategyStatus.BLOCKED.value,),
        ).fetchall()
        return [_row_to_record(row) for row in rows]

    def may_trade(self, strategy_id: str, version: int = 1) -> tuple[bool, str]:
        """Whether this strategy may trade, and why not if it may not.

        Fail-closed on an unregistered strategy. A strategy the registry has
        never heard of is not a new strategy with no history — it is a code
        path that reached the live loop without passing the gate, and the only
        safe reading is no.
        """
        record = self.status_of(strategy_id, version)
        if record is None:
            return False, (
                f"{strategy_id}@v{version} is not in the registry. Nothing may trade "
                "that has not been registered and promoted, because the registry is "
                "where the promotion gate records its decision."
            )
        if not record.status.may_trade:
            # The stored reason when there is one. "blocked" alone does not say
            # whether a lineage ran out of budget or a review killed it, and
            # `retire_reason` already holds the sentence that does — a caller
            # reporting only the status would send an operator looking for it.
            if record.retire_reason:
                return False, f"{record.label} is {record.status.value}: {record.retire_reason}"
            return False, f"{record.label} is {record.status.value}"
        budget = self.budget_for(record.lineage_id)
        if budget is not None and budget.is_exhausted:
            return False, (
                f"{record.label} is promoted but its lineage {record.lineage_id} has "
                f"spent its loss budget ({budget.consumed_ccy} of {budget.budget_ccy})"
            )
        return True, f"{record.label} is promoted"

    def retire(
        self,
        strategy_id: str,
        *,
        version: int = 1,
        reason: str,
        at: datetime | None = None,
        status: StrategyStatus = StrategyStatus.RETIRED,
    ) -> StrategyRecord:
        """Stop a strategy trading.

        Cheap on purpose. At floor size with multi-day holds a strategy
        produces 10-20 trades a month, so the evidence for KILL is necessarily
        thin — and the asymmetry the plan asks for is that KILL needs little
        evidence while SCALE needs a lot. Making retirement expensive would
        invert that.
        """
        if status.may_trade:
            raise RegistryError(
                f"retire() was asked to set status {status.value}, which permits "
                "trading. Retirement is the path out; promotion is a separate module "
                "and a separate gate."
            )
        record = self.status_of(strategy_id, version)
        if record is None:
            raise RegistryError(f"{strategy_id}@v{version} is not in the registry")
        moment = at or now_utc()

        with self._ledger.transaction() as tx:
            tx.append(
                EventType.STRATEGY_RETIRED,
                strategy_id,
                StrategyRetiredPayload(
                    strategy_id=strategy_id,
                    version=version,
                    lineage_id=record.lineage_id,
                    reason=reason,
                    realised_pnl_ccy=str(record.realised_pnl_ccy),
                    n_realised_trades=record.n_realised_trades,
                    detail=status.value,
                ),
                actor=Actor.SYSTEM,
                run_id=self._run_id,
            )
            tx.execute(
                "UPDATE strategy_status SET status = ?, retired_at = ?, retire_reason = ?, "
                "updated_at = ? WHERE strategy_id = ? AND version = ?",
                (status.value, to_iso(moment), reason, to_iso(moment), strategy_id, version),
            )

        updated = self.status_of(strategy_id, version)
        if updated is None:  # pragma: no cover - written in the transaction above
            raise RegistryError(f"{strategy_id}@v{version} vanished mid-retirement")
        return updated

    # -- budgets -----------------------------------------------------------

    def budget_for(self, lineage_id: str) -> LineageBudget | None:
        row: sqlite3.Row | None = self._ledger.conn.execute(
            "SELECT * FROM lineage_budgets WHERE lineage_id = ?", (lineage_id,)
        ).fetchone()
        return None if row is None else _row_to_budget(row)

    def charge(
        self,
        lineage_id: str,
        *,
        loss_ccy: Decimal,
        strategy_id: str | None = None,
        at: datetime | None = None,
    ) -> LineageBudget:
        """Charge a realised loss against a lineage's budget.

        Takes a **loss**, as a positive number, and refuses a negative one. A
        signed P&L would mean a profitable lineage accrued negative consumption
        and granted itself a budget larger than the hash-pinned file allows —
        the single thing the control layer exists to make impossible. Profits
        are recorded on the strategy's own realised P&L, where they belong.

        Exhaustion blocks every promoted member of the lineage in the same
        transaction as the event, so there is no window in which the budget is
        spent and a strategy is still trading against it.
        """
        if loss_ccy < 0:
            raise RegistryError(
                f"charge() was given {loss_ccy}, a negative loss. This takes losses as "
                "positive numbers: a signed P&L here would let a profitable lineage "
                "enlarge its own loss budget beyond what a human set, which is the one "
                "thing the hash-pinned limits exist to prevent."
            )
        moment = at or now_utc()
        existing = self.budget_for(lineage_id)
        if existing is None:
            with self._ledger.transaction() as tx:
                self._touch_budget(tx, lineage_id, moment)
            existing = self.budget_for(lineage_id)
            if existing is None:  # pragma: no cover - the insert above just ran
                raise RegistryError(f"could not open a budget for {lineage_id}")

        consumed = existing.consumed_ccy + loss_ccy
        newly_exhausted = consumed >= existing.budget_ccy and existing.exhausted_at is None

        with self._ledger.transaction() as tx:
            tx.execute(
                "UPDATE lineage_budgets SET consumed_ccy = ?, updated_at = ?, "
                "exhausted_at = COALESCE(exhausted_at, ?) WHERE lineage_id = ?",
                (
                    str(consumed),
                    to_iso(moment),
                    to_iso(moment) if consumed >= existing.budget_ccy else None,
                    lineage_id,
                ),
            )
            if newly_exhausted:
                tx.append(
                    EventType.LINEAGE_BUDGET_EXHAUSTED,
                    lineage_id,
                    LineageBudgetPayload(
                        lineage_id=lineage_id,
                        budget_ccy=str(existing.budget_ccy),
                        consumed_ccy=str(consumed),
                        n_strategies=existing.n_strategies,
                        triggering_strategy_id=strategy_id,
                        detail=(
                            "every member of this lineage is blocked, including any "
                            "child registered after this point: a per-strategy budget "
                            "would be escaped by proposing one"
                        ),
                    ),
                    actor=Actor.SYSTEM,
                    run_id=self._run_id,
                )
                # Blocked, not retired. The distinction is the reason this is a
                # lineage budget: the strategies themselves may have been fine.
                tx.execute(
                    "UPDATE strategy_status SET status = ?, retire_reason = ?, "
                    "updated_at = ? WHERE lineage_id = ? AND status NOT IN (?, ?)",
                    (
                        StrategyStatus.BLOCKED.value,
                        f"lineage {lineage_id} exhausted its loss budget",
                        to_iso(moment),
                        lineage_id,
                        StrategyStatus.RETIRED.value,
                        StrategyStatus.BLOCKED.value,
                    ),
                )

        updated = self.budget_for(lineage_id)
        if updated is None:  # pragma: no cover - written above
            raise RegistryError(f"budget for {lineage_id} vanished mid-charge")
        return updated

    def record_realised(
        self,
        strategy_id: str,
        *,
        version: int = 1,
        pnl_ccy: Decimal,
        n_trades: int = 1,
        at: datetime | None = None,
    ) -> StrategyRecord:
        """Add a realised result to a strategy's record, charging any loss.

        One call rather than two because the two must not drift: a realised
        loss recorded against the strategy but not charged to the lineage is a
        budget that never depletes, and the failure would look like a control
        that is in force.
        """
        record = self.status_of(strategy_id, version)
        if record is None:
            raise RegistryError(f"{strategy_id}@v{version} is not in the registry")
        moment = at or now_utc()
        total = record.realised_pnl_ccy + pnl_ccy

        with self._ledger.transaction() as tx:
            tx.execute(
                "UPDATE strategy_status SET realised_pnl_ccy = ?, n_realised_trades = ?, "
                "updated_at = ? WHERE strategy_id = ? AND version = ?",
                (
                    str(total),
                    record.n_realised_trades + n_trades,
                    to_iso(moment),
                    strategy_id,
                    version,
                ),
            )

        if pnl_ccy < 0:
            self.charge(
                record.lineage_id,
                loss_ccy=-pnl_ccy,
                strategy_id=strategy_id,
                at=moment,
            )

        updated = self.status_of(strategy_id, version)
        if updated is None:  # pragma: no cover - written above
            raise RegistryError(f"{strategy_id}@v{version} vanished mid-update")
        return updated

    def _touch_budget(self, tx: _Writer, lineage_id: str, moment: datetime) -> None:
        """Open a lineage's budget if it has none, and count its members.

        `INSERT OR IGNORE` then a recount, rather than an upsert on the count:
        the number of strategies in a lineage is a fact about `strategy_specs`,
        and deriving it there means the two cannot disagree.
        """
        tx.execute(
            "INSERT OR IGNORE INTO lineage_budgets "
            "(lineage_id, budget_ccy, consumed_ccy, n_strategies, opened_at, updated_at) "
            "VALUES (?, ?, '0', 0, ?, ?)",
            (lineage_id, str(self._budget), to_iso(moment), to_iso(moment)),
        )
        tx.execute(
            "UPDATE lineage_budgets SET n_strategies = "
            "(SELECT COUNT(*) FROM strategy_specs WHERE lineage_id = ?), updated_at = ? "
            "WHERE lineage_id = ?",
            (lineage_id, to_iso(moment), lineage_id),
        )


# --------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------


def _actor_for(author_kind: AuthorKind) -> Actor:
    """Which actor a registration is attributed to.

    A mutation is attributed to SEARCH rather than to a fourth actor: the
    ledger's `Actor` vocabulary answers "who caused this", and a mutation is
    caused by the searcher running. The distinction between a fresh spec and a
    mutation is on the payload, where it is a property of the spec.
    """
    if author_kind is AuthorKind.HUMAN:
        return Actor.HUMAN
    if author_kind is AuthorKind.LLM:
        return Actor.LLM
    return Actor.SEARCH


def _row_to_spec(row: sqlite3.Row, generation: int) -> RegisteredSpec:
    edge = row["expected_edge_bps"]
    return RegisteredSpec(
        strategy_id=str(row["strategy_id"]),
        version=int(row["version"]),
        lineage_id=str(row["lineage_id"]),
        spec_hash=str(row["spec_hash"]),
        author_kind=AuthorKind(str(row["author_kind"])),
        registered_at=from_iso(str(row["registered_at"])),
        parent_strategy_id=(
            None if row["parent_strategy_id"] is None else str(row["parent_strategy_id"])
        ),
        expected_edge_bps=None if edge is None else Decimal(str(edge)),
        generation=generation,
    )


def _row_to_record(row: sqlite3.Row) -> StrategyRecord:
    def when(key: str) -> datetime | None:
        value = row[key]
        return None if value is None else from_iso(str(value))

    return StrategyRecord(
        strategy_id=str(row["strategy_id"]),
        version=int(row["version"]),
        lineage_id=str(row["lineage_id"]),
        status=StrategyStatus(str(row["status"])),
        rung=int(row["rung"]),
        updated_at=from_iso(str(row["updated_at"])),
        rung_changed_at=when("rung_changed_at"),
        promoted_at=when("promoted_at"),
        retired_at=when("retired_at"),
        retire_reason="" if row["retire_reason"] is None else str(row["retire_reason"]),
        realised_pnl_ccy=Decimal(str(row["realised_pnl_ccy"])),
        n_realised_trades=int(row["n_realised_trades"]),
    )


def _row_to_budget(row: sqlite3.Row) -> LineageBudget:
    exhausted = row["exhausted_at"]
    return LineageBudget(
        lineage_id=str(row["lineage_id"]),
        budget_ccy=Decimal(str(row["budget_ccy"])),
        consumed_ccy=Decimal(str(row["consumed_ccy"])),
        n_strategies=int(row["n_strategies"]),
        opened_at=from_iso(str(row["opened_at"])),
        exhausted_at=None if exhausted is None else from_iso(str(exhausted)),
    )


def lineage_roots(specs: Sequence[RegisteredSpec]) -> dict[str, RegisteredSpec]:
    """The earliest registration in each lineage.

    Pure, so the lineage view can be assembled and tested without a database.
    """
    roots: dict[str, RegisteredSpec] = {}
    for spec in sorted(specs, key=lambda s: (s.registered_at, s.strategy_id)):
        roots.setdefault(spec.lineage_id, spec)
    return roots
