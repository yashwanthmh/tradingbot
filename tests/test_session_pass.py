"""The per-session portfolio pass: review, rungs, allocations — and the book.

The review, the size ladder and the allocator existed and had no production
caller. Every promoted strategy traded at the floor rung for ever, no
allocation round was ever recorded, and a running loop kept the book it
started with however the registry changed. Capital could not move.

Now the loop's first cycle of each trading session reviews every promoted
strategy, moves its rung on the evidence of that rung, allocates, and hands the
loop a book rebuilt from all of it.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from decimal import Decimal
from typing import Any

from tb.config.loader import PinnedLimits
from tb.core.clock import to_iso
from tb.engine.funding import Book, explicit_book, funded_book, unfunded_notional
from tb.features.pipeline import FeaturePipeline
from tb.ledger.store import Ledger
from tb.portfolio.session import run_session_pass
from tb.registry.ladder import notional_for
from tb.registry.lineage import SpecRegistry
from tb.registry.model_store import ModelStore, default_model_root
from tb.strategy.trivial import specs
from tests.test_funding import _promote, a_spec
from tests.test_loop import AS_OF, _broker, _loop, _rising_bars, _seed
from tests.test_protection import Scripted

PROMOTED = AS_OF - timedelta(days=20)
EQUITY = Decimal("10000.00")


def _registry(env: dict[str, Any], ledger: Ledger) -> SpecRegistry:
    pinned: PinnedLimits = env["pinned"]
    return SpecRegistry(ledger, per_lineage_budget_ccy=pinned.limits.loss.per_lineage_budget_ccy)


def _promotion_record(ledger: Ledger, strategy_id: str, *, probability: float) -> None:
    """What the gate writes when it promotes, reduced to what the pass reads."""
    ledger.conn.execute(
        "INSERT INTO promotions (promotion_id, strategy_id, version, lineage_id, spec_hash,"
        " decision, n_gates, n_failed, gate_results_json, deflated_sharpe,"
        " deflated_sharpe_probability, decided_at, deciding_event_seq)"
        " VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)",
        (
            f"prm_{strategy_id}",
            strategy_id,
            1,
            "lin",
            "hash",
            "promote",
            0,
            0,
            "[]",
            0.8,
            probability,
            to_iso(PROMOTED),
            0,
        ),
    )
    ledger.conn.commit()


def _trip(ledger: Ledger, strategy_id: str, *, n: int, pnl: str, closed: datetime) -> None:
    """A charged round trip: one share bought at 100, closed for `pnl`."""
    ledger.conn.execute(
        "INSERT INTO round_trips (closing_fill_id, t212_ticker, strategy_id,"
        " strategy_version, quantity, exit_price, cost_basis, pnl_ccy, admissible,"
        " charged, detail, closed_at, recording_event_seq)"
        " VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)",
        (
            f"fill_{strategy_id}_{n}",
            "AAPL_US_EQ",
            strategy_id,
            1,
            "1",
            str(Decimal("100") + Decimal(pnl)),
            "100",
            pnl,
            1,
            1,
            "",
            to_iso(closed),
            n,
        ),
    )
    ledger.conn.commit()


def _pass(env: dict[str, Any], ledger: Ledger, *, at: datetime = AS_OF) -> Any:
    return run_session_pass(ledger, limits=env["pinned"].limits, equity_ccy=EQUITY, at=at)


def test_a_strategy_that_has_earned_a_rung_climbs_it(env: dict[str, Any]) -> None:
    """Twenty days at the floor, five profitable trades: up one rung, and the
    allocation that session is sized at the new rung."""
    _seed(env, _rising_bars(days=140))
    with Ledger(env["db"], config_hash=env["pinned"].config_hash) as ledger:
        strategy_id, _ = _promote(ledger, at=PROMOTED)
        _promotion_record(ledger, strategy_id, probability=0.97)
        for n in range(5):
            _trip(ledger, strategy_id, n=n, pnl="2.00", closed=PROMOTED + timedelta(days=n + 1))

        passed = _pass(env, ledger)
        record = _registry(env, ledger).status_of(strategy_id, 1)

    assert [(m.from_rung, m.to_rung) for m in passed.moves] == [(0, 1)]
    assert record is not None and record.rung == 1
    assert passed.allocation is not None
    (allocation,) = passed.allocation.allocations
    assert allocation.rung == 1
    assert allocation.notional_ccy == notional_for(
        1, limits=env["pinned"].limits, equity_ccy=EQUITY
    )


def test_a_kill_verdict_drops_two_rungs_at_once(env: dict[str, Any]) -> None:
    """The fast half of the ratchet. A strategy that has spent most of its
    lineage's loss budget is judged KILL, and the pass demotes it two rungs in
    the same session — retirement itself stays a human's `tb review --apply`."""
    _seed(env, _rising_bars(days=140))
    with Ledger(env["db"], config_hash=env["pinned"].config_hash) as ledger:
        strategy_id, lineage_id = _promote(ledger, at=PROMOTED)
        ledger.conn.execute(
            "UPDATE strategy_status SET rung = 3, rung_changed_at = ? WHERE strategy_id = ?",
            (to_iso(PROMOTED), strategy_id),
        )
        ledger.conn.commit()
        registry = _registry(env, ledger)
        budget = env["pinned"].limits.loss.per_lineage_budget_ccy
        registry.charge(lineage_id, loss_ccy=budget * Decimal("0.6"), strategy_id=strategy_id)

        passed = _pass(env, ledger)
        record = registry.status_of(strategy_id, 1)

    assert [r.verdict.value for r in passed.reviews] == ["kill"]
    assert [(m.from_rung, m.to_rung) for m in passed.moves] == [(3, 1)]
    assert record is not None and record.rung == 1


def test_the_pass_runs_once_a_session(env: dict[str, Any]) -> None:
    """However often the loop is started. A breach applied twice would drop four
    rungs, so the second call the same session is a recorded no-op."""
    _seed(env, _rising_bars(days=140))
    with Ledger(env["db"], config_hash=env["pinned"].config_hash) as ledger:
        strategy_id, _ = _promote(ledger, at=PROMOTED)
        _promotion_record(ledger, strategy_id, probability=0.97)

        first = _pass(env, ledger)
        again = _pass(env, ledger, at=AS_OF + timedelta(hours=2))
        tomorrow = _pass(env, ledger, at=AS_OF + timedelta(days=1))
        (count,) = ledger.conn.execute(
            "SELECT COUNT(*) FROM event_log WHERE event_type = 'session.reviewed'"
        ).fetchone()

    assert not first.skipped and again.skipped and not tomorrow.skipped
    assert count == 2


def test_a_promotion_with_no_gate_record_is_allocated_nothing(env: dict[str, Any]) -> None:
    """`PROMOTED` without the gate's decision behind it has no deflated prior.
    It is refused capital rather than funded on its own say-so — and the book
    built afterwards carries that allocation, which refuses its entries."""
    _seed(env, _rising_bars(days=140))
    with Ledger(env["db"], config_hash=env["pinned"].config_hash) as ledger:
        gated, _ = _promote(ledger, spec=a_spec(name="gated"), at=PROMOTED)
        _promotion_record(ledger, gated, probability=0.97)
        ungated, _ = _promote(ledger, spec=a_spec(name="ungated"), at=PROMOTED)
        passed = _pass(env, ledger)
        book = funded_book(
            ledger,
            limits=env["pinned"].limits,
            equity_ccy=EQUITY,
            at=AS_OF + timedelta(minutes=1),
        )

    assert gated != ungated
    assert passed.allocation is not None
    notionals = {a.strategy_id: a.notional_ccy for a in passed.allocation.allocations}
    assert notionals[gated] > 0
    assert notionals[ungated] == 0
    sizes = {funded.strategy_id: funded.notional_ccy for funded in book.funded}
    assert sizes[ungated] == 0


@dataclass
class _Hook:
    """Counts the sessions the loop opened, and hands back a book to adopt."""

    book: Book
    calls: list[datetime]

    def __call__(self, at: datetime) -> Book:
        self.calls.append(at)
        return self.book


def _holding_book(env: dict[str, Any]) -> Book:
    return explicit_book(
        Scripted([]),
        FeaturePipeline(specs=specs()),
        notional_ccy=unfunded_notional(env["pinned"].limits, equity_ccy=EQUITY),
    )


def test_the_loop_opens_each_trading_session_once_and_adopts_its_book(
    env: dict[str, Any],
) -> None:
    """Twice on Wednesday is one session; Good Friday and the Saturday after
    have none; Thursday and Monday each open one. The book the hook returns is
    the book the loop trades from then on."""
    _seed(env, _rising_bars(days=140))
    broker = _broker()
    refreshed = _holding_book(env)
    hook = _Hook(book=refreshed, calls=[])
    instants = [
        AS_OF,
        AS_OF + timedelta(hours=1),
        datetime(2026, 4, 2, 15, 30, tzinfo=UTC),
        datetime(2026, 4, 3, 15, 30, tzinfo=UTC),  # Good Friday
        datetime(2026, 4, 4, 15, 30, tzinfo=UTC),  # Saturday
        datetime(2026, 4, 6, 15, 30, tzinfo=UTC),
    ]
    with Ledger(env["db"], config_hash=env["pinned"].config_hash) as ledger:
        loop = _loop(env, ledger, broker, book=_holding_book(env))
        loop.on_new_session = hook
        loop.lock.acquire(at=AS_OF)
        for moment in instants:
            loop.clock = lambda moment=moment: moment  # type: ignore[misc]
            loop.run_cycle()

    assert [call.date().isoformat() for call in hook.calls] == [
        "2026-04-01",
        "2026-04-02",
        "2026-04-06",
    ]
    assert loop.book is refreshed


def test_tb_run_s_hook_reviews_rebuilds_and_records_the_book(env: dict[str, Any]) -> None:
    """The wiring `tb run` uses: the pass, then the promoted book rebuilt at the
    new rung and recorded; the same session again keeps the book it has."""
    from tb.cli_engine import _broker as broker_for_mode
    from tb.cli_engine import _session_refresh

    _seed(env, _rising_bars(days=140))
    broker = broker_for_mode("paper", env["pinned"], equity=EQUITY)
    with Ledger(env["db"], config_hash=env["pinned"].config_hash) as ledger:
        strategy_id, _ = _promote(ledger, at=PROMOTED)
        _promotion_record(ledger, strategy_id, probability=0.97)
        for n in range(5):
            _trip(ledger, strategy_id, n=n, pnl="2.00", closed=PROMOTED + timedelta(days=n + 1))
        hook = _session_refresh(
            ledger,
            env["pinned"],
            broker=broker,
            universe={},
            run_id="run_hook",
            models=ModelStore(ledger, default_model_root(env["db"])),
        )

        book = hook(AS_OF)
        again = hook(AS_OF + timedelta(hours=1))
        (recorded,) = ledger.conn.execute(
            "SELECT COUNT(*) FROM event_log WHERE event_type = 'book.funded'"
        ).fetchone()

    assert book is not None and again is None
    ((funded),) = book.funded
    assert funded.strategy_id == strategy_id and funded.rung == 1
    assert recorded == 1
