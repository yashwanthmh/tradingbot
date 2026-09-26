"""Arming real money: nothing short of the recorded evidence and a person's yes.

What has to hold: every requirement is read from the ledger and each gap is
named; evidence that is stale, from paper, off-hours or from the same machine
does not count; an arming names promoted strategies within the cap, lapses,
ends on a disarm, and is bound to the limits it was judged under; the live
book is only what was armed, at the capped rung; the loop halts when its
permit is withdrawn; an unarmed real-money client refuses every write,
cancels included; and `tb run --mode live` refuses at each missing piece.
"""

from __future__ import annotations

from collections.abc import Callable, Iterator
from datetime import datetime, timedelta
from decimal import Decimal
from pathlib import Path
from typing import Any

import pytest
from typer.testing import CliRunner

from tb.broker.port import OrderPurpose, OrderType
from tb.broker.t212.client import ClientConfig, T212Client
from tb.broker.t212.errors import BrokerError
from tb.cli import app
from tb.config.loader import PinnedLimits, load_hard_limits
from tb.core.http import RecordingTransport
from tb.engine.funding import funded_book
from tb.engine.loop import LoopHalted
from tb.ledger.events import (
    BackupCreatedPayload,
    BackupRestoreVerifiedPayload,
    DrillCompletedPayload,
    DrillStartedPayload,
    EventType,
    TradeClosedPayload,
)
from tb.ledger.store import Ledger
from tb.ops.arming import (
    ArmingError,
    arm_live,
    arming_state,
    disarm,
    live_book,
    live_evidence,
    live_permit,
    resolve_strategies,
)
from tb.registry.ladder import notional_for
from tests.conftest import REFERENCE_LIMITS
from tests.session_helpers import SAT, TUE, WED, PinnedClock, append, pin_clock, run, utc
from tests.test_crash_drills import _token
from tests.test_funding import _promote, a_spec

NOW = utc(SAT, 12)
LIVE_OVERRIDES = {"enabled": True, "min_clean_demo_sessions": 2, "min_demo_closed_trades": 1}


@pytest.fixture
def pinned(write_limits: Callable[[dict[str, Any]], Path]) -> PinnedLimits:
    return load_hard_limits(write_limits({"live": LIVE_OVERRIDES}))


@pytest.fixture
def clock(monkeypatch: pytest.MonkeyPatch) -> PinnedClock:
    return pin_clock(monkeypatch)


@pytest.fixture
def book(ledger_path: Path, clock: PinnedClock, pinned: PinnedLimits) -> Iterator[Ledger]:
    with Ledger(ledger_path, config_hash=pinned.config_hash) as opened:
        opened.initialise(created_by="test")
        yield opened


def _drill(ledger: Ledger, clock: PinnedClock, kind: str, **overrides: Any) -> None:
    fields: dict[str, Any] = {"mode": "demo", "market_open": True, "passed": True} | overrides
    drill_id = f"drl_{kind}"
    append(
        ledger,
        clock,
        utc(SAT, 10),
        EventType.DRILL_STARTED,
        DrillStartedPayload(
            drill_id=drill_id,
            kind=kind,
            run_id="run_drilled",
            mode=fields["mode"],
            pid=1,
            host="h",
            session_date="2026-09-25",
            market_open=fields["market_open"],
            holdings=[{"ticker": "AAPL_US_EQ", "quantity": "1", "covered": "1"}],
        ),
    )
    append(
        ledger,
        clock,
        utc(SAT, 10, 5),
        EventType.DRILL_COMPLETED,
        DrillCompletedPayload(
            drill_id=drill_id, kind=kind, run_id="run_drilled", passed=fields["passed"]
        ),
    )


def _restore(ledger: Ledger, clock: PinnedClock, *, same_host: bool = False) -> None:
    append(
        ledger,
        clock,
        utc(SAT, 9),
        EventType.BACKUP_CREATED,
        BackupCreatedPayload(
            backup_id="bkp_1",
            manifest_sha256="a" * 64,
            head_seq=1,
            head_chain_hash="b" * 64,
            n_files=3,
            n_bytes=100,
            host="trading-box",
            destination="var/backups/bkp_1",
        ),
    )
    append(
        ledger,
        clock,
        utc(SAT, 11),
        EventType.BACKUP_RESTORE_VERIFIED,
        BackupRestoreVerifiedPayload(
            backup_id="bkp_1",
            manifest_sha256="a" * 64,
            source_host="trading-box",
            restored_host="trading-box" if same_host else "spare-laptop",
            restored_at=utc(SAT, 11).isoformat(),
            same_host=same_host,
            restored_head_seq=2,
            restored_head_chain_hash="c" * 64,
        ),
    )


def _evidence(ledger: Ledger, clock: PinnedClock, *, drills: bool = True) -> str:
    """Two clean demo sessions with a trade closed, both drills, an off-machine restore.

    Returns the promoted strategy's id.
    """
    run(ledger, clock, run_id="run_tue", day=TUE)
    run(ledger, clock, run_id="run_wed", day=WED)
    append(
        ledger,
        clock,
        utc(WED, 18),
        EventType.TRADE_CLOSED,
        TradeClosedPayload(
            closing_fill_id="fill_1",
            run_id="run_wed",
            t212_ticker="AAPL_US_EQ",
            quantity=Decimal(1),
            admissible=True,
            charged=True,
        ),
    )
    if drills:
        _drill(ledger, clock, "kill_switch")
        _drill(ledger, clock, "watchdog")
    _restore(ledger, clock)
    clock.at = NOW
    strategy_id, _ = _promote(ledger)
    return strategy_id


def _unmet(ledger: Ledger, pinned: PinnedLimits, now: datetime = NOW) -> list[str]:
    return [r.name for r in live_evidence(ledger, limits=pinned.limits, now=now).unmet]


# --------------------------------------------------------------------------
# The evidence
# --------------------------------------------------------------------------


def test_each_missing_piece_of_evidence_is_named(book: Ledger, pinned: PinnedLimits) -> None:
    assert _unmet(book, pinned) == [
        "clean demo sessions in a row",
        "trades closed across that streak",
        "kill-switch drill passed",
        "watchdog drill passed",
        "a restore verified on another machine",
    ]
    shipped = load_hard_limits(REFERENCE_LIMITS)
    assert "live trading enabled in the limits file" in _unmet(book, shipped)


def test_with_the_evidence_recorded_the_gate_is_ready(
    book: Ledger, clock: PinnedClock, pinned: PinnedLimits
) -> None:
    _evidence(book, clock)
    evidence = live_evidence(book, limits=pinned.limits, now=NOW)
    assert evidence.ready, evidence.unmet
    observed = {r.name: r.observed for r in evidence.requirements}
    assert observed["clean demo sessions in a row"] == "2, since 2026-09-22"
    assert observed["trades closed across that streak"] == "1"


@pytest.mark.parametrize(
    ("overrides", "gap"),
    [
        ({"mode": "paper"}, "kill-switch drill passed"),
        ({"market_open": False}, "kill-switch drill passed"),
        ({"passed": False}, "kill-switch drill passed"),
    ],
)
def test_a_drill_counts_only_on_demo_in_market_hours_and_passed(
    book: Ledger, clock: PinnedClock, pinned: PinnedLimits, overrides: dict[str, Any], gap: str
) -> None:
    _evidence(book, clock, drills=False)
    _drill(book, clock, "kill_switch", **overrides)
    _drill(book, clock, "watchdog")
    assert _unmet(book, pinned) == [gap]


def test_a_restore_on_the_same_machine_does_not_count(
    book: Ledger, clock: PinnedClock, pinned: PinnedLimits
) -> None:
    _evidence(book, clock)
    _restore(book, clock, same_host=True)  # a later same-host restore changes nothing
    assert _unmet(book, pinned) == []

    with Ledger(book.path.with_name("other.db"), config_hash=pinned.config_hash) as other:
        other.initialise(created_by="test")
        _restore(other, clock, same_host=True)
        assert "a restore verified on another machine" in _unmet(other, pinned)


def test_evidence_goes_stale(book: Ledger, clock: PinnedClock, pinned: PinnedLimits) -> None:
    _evidence(book, clock)
    month_on = NOW + timedelta(days=31)
    assert _unmet(book, pinned, month_on) == ["kill-switch drill passed", "watchdog drill passed"]
    assert "a restore verified on another machine" in _unmet(book, pinned, NOW + timedelta(91))


# --------------------------------------------------------------------------
# Arming
# --------------------------------------------------------------------------


def test_arming_refuses_without_the_evidence(book: Ledger, pinned: PinnedLimits) -> None:
    strategy_id, _ = _promote(book)
    with pytest.raises(ArmingError, match="not armed; unmet"):
        arm_live(book, pinned=pinned, strategies=[strategy_id], armed_by="ops", now=NOW)
    assert not arming_state(book, config_hash=pinned.config_hash, now=NOW).armed


def test_an_arming_names_its_strategies_and_holds_until_it_lapses(
    book: Ledger, clock: PinnedClock, pinned: PinnedLimits
) -> None:
    strategy_id = _evidence(book, clock)
    arming = arm_live(book, pinned=pinned, strategies=[strategy_id], armed_by="ops", now=NOW)

    assert arming.strategies == (f"{strategy_id}@v1",)
    assert arming.covers(strategy_id, 1)
    assert arming.max_rung == 0
    assert arming.expires_at == NOW + timedelta(days=7)
    state = arming_state(book, config_hash=pinned.config_hash, now=NOW + timedelta(days=1))
    assert state.armed

    lapsed = arming_state(book, config_hash=pinned.config_hash, now=arming.expires_at)
    assert not lapsed.armed and "lapsed" in lapsed.reason
    moved = arming_state(book, config_hash="0" * 64, now=NOW)
    assert not moved.armed and "limits changed since arming" in moved.reason

    assert disarm(book, disarmed_by="ops", reason="done for the week") == arming.arming_id
    after = arming_state(book, config_hash=pinned.config_hash, now=NOW + timedelta(hours=1))
    assert not after.armed and "done for the week" in after.reason


def test_only_promoted_strategies_within_the_cap_can_be_armed(
    book: Ledger, pinned: PinnedLimits
) -> None:
    first, _ = _promote(book)
    second, _ = _promote(book, spec=a_spec(name="second"))
    assert resolve_strategies(book, [first], limits=pinned.limits) == [f"{first}@v1"]
    with pytest.raises(ArmingError, match="no promoted version"):
        resolve_strategies(book, ["strat_nobody"], limits=pinned.limits)
    with pytest.raises(ArmingError, match="allow 1"):
        resolve_strategies(book, [first, second], limits=pinned.limits)
    with pytest.raises(ArmingError, match="not in the registry"):
        resolve_strategies(book, [f"{first}@v9"], limits=pinned.limits)
    with pytest.raises(ArmingError, match="name the strategy"):
        resolve_strategies(book, [], limits=pinned.limits)


def test_the_live_book_is_what_was_armed_at_the_capped_rung(
    book: Ledger, clock: PinnedClock, pinned: PinnedLimits
) -> None:
    armed_id = _evidence(book, clock)
    other_id, _ = _promote(book, spec=a_spec(name="other"), rung=2)
    book.conn.execute("UPDATE strategy_status SET rung = 2 WHERE strategy_id = ?", (armed_id,))
    arming = arm_live(book, pinned=pinned, strategies=[armed_id], armed_by="ops", now=NOW)
    equity = Decimal("10000")
    funded = funded_book(book, limits=pinned.limits, equity_ccy=equity, at=NOW)
    assert len(funded) == 2

    live = live_book(funded, arming, limits=pinned.limits, equity_ccy=equity)

    [only] = live.funded
    assert only.strategy_id == armed_id
    assert only.rung == 0
    assert only.notional_ccy <= notional_for(0, limits=pinned.limits, equity_ccy=equity)
    assert (f"{other_id}@v1", f"not armed for live ({arming.arming_id})") in live.excluded


# --------------------------------------------------------------------------
# While trading
# --------------------------------------------------------------------------


def test_the_permit_follows_the_arming(
    book: Ledger, clock: PinnedClock, pinned: PinnedLimits
) -> None:
    strategy_id = _evidence(book, clock)
    permit = live_permit(book, config_hash=pinned.config_hash)
    assert permit(NOW) is not None and "never been armed" in str(permit(NOW))
    arm_live(book, pinned=pinned, strategies=[strategy_id], armed_by="ops", now=NOW)
    assert permit(NOW) is None
    disarm(book, disarmed_by="ops", reason="stop")
    assert "stop" in str(permit(NOW + timedelta(minutes=1)))


def test_the_loop_halts_when_its_permit_is_withdrawn(env: dict[str, Any]) -> None:
    from tests.test_loop import _broker, _loop, _rising_bars, _seed

    _seed(env, _rising_bars(days=140))
    with Ledger(env["db"], config_hash=env["pinned"].config_hash) as ledger:
        loop = _loop(env, ledger, _broker())
        loop.lock.acquire()
        loop.permit = lambda at: "live trading is not armed: disarmed by ops"
        with pytest.raises(LoopHalted, match="disarmed by ops"):
            loop.run_cycle()


def _live_client(transport: RecordingTransport) -> T212Client:
    return T212Client(
        ClientConfig(api_key="k", base_url="https://live.example", environment="live"),
        transport=transport,
    )


def test_an_unarmed_real_money_client_refuses_every_write(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Cancels included: cancelling a protective stop adds risk as an entry does."""
    token = _token()
    # Inside the token's short life, so the refusal tested is the arming's.
    monkeypatch.setattr(
        "tb.broker.t212.client.now_utc", lambda: token.expires_at - timedelta(seconds=1)
    )
    transport = RecordingTransport(responses={})
    client = _live_client(transport)
    with pytest.raises(BrokerError, match="has not been armed"):
        client.place_order(token, order_type=OrderType.MARKET, purpose=OrderPurpose.ENTRY)
    with pytest.raises(BrokerError, match="has not been armed"):
        client.cancel_order(token, broker_order_id="42")
    assert transport.calls == [], "nothing may reach the wire unarmed"


# --------------------------------------------------------------------------
# The commands
# --------------------------------------------------------------------------


def _cli(*args: str, input: str | None = None) -> Any:
    return CliRunner().invoke(app, list(args), input=input)


def test_tb_arm_reports_arms_on_the_phrase_and_disarms(
    book: Ledger, clock: PinnedClock, pinned: PinnedLimits
) -> None:
    strategy_id = _evidence(book, clock)
    common = ["--limits", str(pinned.source_path), "--db", str(book.path)]

    reported = _cli("arm", *common)
    assert reported.exit_code == 0, reported.output
    assert "clean demo sessions in a row" in reported.output
    assert "not armed" in reported.output

    refused = _cli("arm", "--live", "--strategy", strategy_id, *common, input="yes\n")
    assert refused.exit_code == 1, refused.output
    assert arming_state(book, config_hash=pinned.config_hash).arming is None

    armed = _cli("arm", "--live", "--strategy", strategy_id, *common, input="arm live\n")
    assert armed.exit_code == 0, armed.output
    assert "Real money." in armed.output and f"armed {strategy_id}@v1" in armed.output
    assert arming_state(book, config_hash=pinned.config_hash).armed

    stopped = _cli("disarm", "--reason", "end of the trial", *common)
    assert stopped.exit_code == 0, stopped.output
    assert not arming_state(book, config_hash=pinned.config_hash).armed


def test_tb_arm_without_the_evidence_exits_1(book: Ledger, pinned: PinnedLimits) -> None:
    common = ["--limits", str(pinned.source_path), "--db", str(book.path)]
    assert _cli("arm", *common).exit_code == 1
    live = _cli("arm", "--live", "--strategy", "x", *common, input="arm live\n")
    assert live.exit_code == 1
    assert "requirement(s) unmet" in live.output


def test_tb_run_live_refuses_at_each_missing_piece(
    book: Ledger,
    clock: PinnedClock,
    pinned: PinnedLimits,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    common = ["--limits", str(pinned.source_path), "--db", str(book.path), "--mode", "live"]

    shipped = _cli(
        "run", "--limits", str(REFERENCE_LIMITS), "--db", str(book.path), "--mode", "live"
    )
    assert shipped.exit_code == 2 and "live trading is disabled" in shipped.output

    unarmed = _cli("run", *common)
    assert unarmed.exit_code == 2 and "never been armed" in unarmed.output

    assert "for drilling the loop" in _cli("run", *common, "--strategy", "trivial").output
    assert "never runs unsupervised" in _cli("run", *common, "--no-watchdog").output

    strategy_id = _evidence(book, clock)
    arm_live(book, pinned=pinned, strategies=[strategy_id], armed_by="ops", now=NOW)
    clock.at = NOW + timedelta(minutes=1)
    monkeypatch.setattr("tb.ops.arming.now_utc", lambda: NOW + timedelta(minutes=1))
    monkeypatch.delenv("T212_LIVE_API_KEY", raising=False)
    monkeypatch.setenv("T212_DEMO_API_KEY", "demo-key-for-the-test")
    wrong_key = _cli("run", *common)
    assert wrong_key.exit_code == 2, wrong_key.output
    assert "needs T212_LIVE_API_KEY" in wrong_key.output
