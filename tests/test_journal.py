"""The daily journal: a page per session, reproducible from the ledger it names.

What has to hold: a page shows what the ledger recorded for its session and
nothing from any other; the same ledger produces the same bytes; every event
lands on exactly one page; verification catches an edited page and a ledger
rewritten beneath one, and names what changed; a page written before its
session could be judged says so and is replaced, and one that no longer
verifies is never overwritten; text from outside the system arrives inert.
"""

from __future__ import annotations

import json
import re
import subprocess
from collections.abc import Callable, Iterator
from datetime import datetime, timedelta
from decimal import Decimal
from pathlib import Path
from typing import Any

import pytest
from typer.testing import CliRunner

from tb.cli import app
from tb.config.loader import load_hard_limits
from tb.data.calendar import TradingCalendar
from tb.ledger.events import (
    EventType,
    FillPayload,
    IntentCommittedPayload,
    OrderOutcomePayload,
    ProtectionPayload,
    TradeClosedPayload,
)
from tb.ledger.store import Ledger
from tb.ops.journal import (
    JournalError,
    JournalPage,
    ensure_pinned,
    latest_final_session,
    parse_header,
    render_page,
    verify_page,
    write_page,
)
from tb.portfolio.pnl import EquityCurve
from tests.conftest import REFERENCE_LIMITS
from tests.session_helpers import (
    FRI,
    MON,
    NEXT_MON,
    SAT,
    THU,
    TUE,
    WED,
    PinnedClock,
    append,
    cycle,
    pin_clock,
    run,
    utc,
)

PINNED = load_hard_limits(REFERENCE_LIMITS)


@pytest.fixture
def clock(monkeypatch: pytest.MonkeyPatch) -> PinnedClock:
    return pin_clock(monkeypatch)


@pytest.fixture
def book(ledger_path: Path, clock: PinnedClock) -> Iterator[Ledger]:
    with Ledger(ledger_path) as opened:
        opened.initialise(created_by="test")
        ensure_pinned(opened, PINNED)
        yield opened


def _page(
    ledger: Ledger,
    day: Any = MON,
    *,
    as_of: datetime | None = None,
    through_seq: int | None = None,
    limits_hash: str | None = None,
) -> JournalPage:
    return render_page(
        ledger,
        session=day,
        limits=PINNED.limits,
        limits_hash=limits_hash or PINNED.config_hash,
        as_of=as_of or utc(day, 21),
        through_seq=through_seq,
    )


def _body(text: str) -> str:
    """Everything after the front matter."""
    return text.split("\n---\n", 1)[1]


def _trading_day(ledger: Ledger, clock: PinnedClock) -> None:
    """A demo session with one order, its fill, its stop, a closed trade and marks."""
    run(ledger, clock, run_id="run_a")
    common: dict[str, Any] = {"run_id": "run_a", "t212_ticker": "AAPL_US_EQ"}
    append(
        ledger,
        clock,
        utc(MON, 15),
        EventType.INTENT_COMMITTED,
        IntentCommittedPayload(
            **common,
            intent_id="int_entry",
            side="buy",
            order_type="market",
            purpose="entry",
            priority_class="risk_increasing",
            quantity=Decimal("0.5"),
            risk_token_id="tok_1",
        ),
    )
    append(
        ledger,
        clock,
        utc(MON, 15) + timedelta(seconds=1),
        EventType.ORDER_ACKNOWLEDGED,
        OrderOutcomePayload(
            **common, intent_id="int_entry", broker_order_id="9001", status="working"
        ),
    )
    append(
        ledger,
        clock,
        utc(MON, 15, 1),
        EventType.FILL_RECORDED,
        FillPayload(
            **common,
            fill_id="fill_1",
            side="buy",
            quantity=Decimal("0.5"),
            source="api_history",
            confidence="reported",
            admissible_for_pnl=True,
            intent_id="int_entry",
            price=Decimal("150.00"),
        ),
    )
    append(
        ledger,
        clock,
        utc(MON, 15, 2),
        EventType.POSITION_PROTECTED,
        ProtectionPayload(
            **common,
            quantity=Decimal("0.5"),
            protected=True,
            stop_price=Decimal("135.00"),
            unprotected_seconds=60.0,
        ),
    )
    append(
        ledger,
        clock,
        utc(MON, 19),
        EventType.TRADE_CLOSED,
        TradeClosedPayload(
            **common,
            closing_fill_id="fill_2",
            quantity=Decimal("0.5"),
            admissible=True,
            charged=True,
            strategy_id="strat_x",
            strategy_version=1,
            exit_price=Decimal("152.50"),
            pnl_ccy=Decimal("1.25"),
        ),
    )
    curve = EquityCurve(ledger, run_id="run_a")
    curve.mark(
        equity_ccy=Decimal("1000.00"),
        at=utc(MON, 13, 31),
        currency="GBP",
        deployed_ccy=Decimal("0"),
    )
    curve.mark(
        equity_ccy=Decimal("1001.25"),
        at=utc(MON, 19, 59),
        currency="GBP",
        deployed_ccy=Decimal("0"),
    )


# --------------------------------------------------------------------------
# What a page says
# --------------------------------------------------------------------------


def test_a_page_is_the_session_as_the_ledger_recorded_it(book: Ledger, clock: PinnedClock) -> None:
    _trading_day(book, clock)
    page = _page(book)
    text = page.text

    header = parse_header(text)
    assert header.session == MON
    assert header.status == "final" and page.final
    assert header.through_seq == book.head().seq  # type: ignore[union-attr]
    assert header.limits_hash == PINNED.config_hash
    assert page.filename == "2026-09-21.md"

    assert "# Monday 21 September 2026" in text
    assert "| demo | clean | 100.0% |" in text
    assert "| int_entry | AAPL_US_EQ | buy | entry | market | 0.5 | working (9001) |" in text
    assert "| AAPL_US_EQ | buy | 0.5 | 150.00 | api_history | yes |" in text
    assert "| strat_x v1 | 0.5 | 152.50 | 1.25 | yes |" in text
    assert "Realised +1.25 across 1 priced trade(s)." in text
    assert "protected after 60s bare" in text
    assert "1000.00 at 13:31:00" in text and "+1.25 (+0.13%)" in text
    assert "| loop.cycle_completed | 43 |" in text


def test_a_quiet_session_still_gets_a_page_that_says_so(book: Ledger, clock: PinnedClock) -> None:
    page = _page(book, TUE)
    assert "No trading run was up for this session." in page.text
    assert "The ledger recorded nothing in this window." in page.text
    assert "No equity was marked in this window." in page.text


def test_a_page_is_only_written_for_a_session(book: Ledger) -> None:
    with pytest.raises(JournalError, match="not a trading session"):
        _page(book, SAT)


# --------------------------------------------------------------------------
# Reproducible
# --------------------------------------------------------------------------


def test_a_page_is_reproducible_byte_for_byte(book: Ledger, clock: PinnedClock) -> None:
    _trading_day(book, clock)
    first = _page(book)
    assert _page(book, through_seq=first.header.through_seq).text == first.text

    # The next session's events move the head but not this page's content.
    run(book, clock, run_id="run_b", day=TUE)
    later = _page(book, as_of=utc(TUE, 21))
    assert later.header.through_seq > first.header.through_seq
    assert _body(later.text) == _body(first.text)


def test_every_event_is_on_exactly_one_page(book: Ledger, clock: PinnedClock) -> None:
    """Overnight and weekend events belong to the session they were waiting for."""
    run(book, clock, run_id="run_mon", day=MON, stop=(23, 50))
    run(book, clock, run_id="run_wed", day=WED)
    cycle(book, clock, "run_wed", utc(SAT, 10), 99)
    days = [MON, TUE, WED, THU, FRI, NEXT_MON]

    counted = 0
    for day in days:
        found = re.search(r": (\d+) in all\.", _page(book, day, as_of=utc(NEXT_MON, 21)).text)
        counted += int(found.group(1)) if found else 0
    assert counted == book.count()


# --------------------------------------------------------------------------
# Verification
# --------------------------------------------------------------------------


def test_verification_accepts_a_page_and_names_what_was_edited(
    book: Ledger, clock: PinnedClock
) -> None:
    _trading_day(book, clock)
    page = _page(book)
    assert verify_page(book, page.text).ok

    edited = page.text.replace("working (9001)", "cancelled")
    check = verify_page(book, edited)
    assert not check.ok
    assert "Orders" in check.problems[0]

    forged = page.text.replace(page.header.chain_hash, "0" * 64)
    assert "rewritten beneath the page" in verify_page(book, forged).problems[0]

    assert "not a journal page" in verify_page(book, "# notes\n").problems[0]


def test_verification_catches_a_ledger_rewritten_beneath_a_page(
    book: Ledger,
    clock: PinnedClock,
    ledger_path: Path,
    tamper: Callable[..., None],
    resign_chain: Callable[[Path], None],
) -> None:
    """Rewritten and re-signed: internally perfect, and still not the ledger the page saw."""
    _trading_day(book, clock)
    page = _page(book)
    tamper(
        ledger_path,
        "UPDATE event_log SET payload_json = replace(payload_json, '1.25', '9.25')"
        " WHERE event_type = 'trade.closed'",
    )
    resign_chain(ledger_path)

    check = verify_page(book, page.text)
    assert not check.ok
    assert "rewritten beneath the page" in check.problems[0]


def test_verification_needs_the_limits_the_page_was_judged_by(
    book: Ledger, clock: PinnedClock
) -> None:
    run(book, clock, run_id="run_a")
    page = _page(book, limits_hash="f" * 64)
    check = verify_page(book, page.text)
    assert not check.ok
    assert "never pinned" in check.problems[0]


# --------------------------------------------------------------------------
# Writing
# --------------------------------------------------------------------------


def test_a_provisional_page_is_replaced_and_a_final_one_is_kept(
    book: Ledger, clock: PinnedClock, tmp_path: Path
) -> None:
    run(book, clock, run_id="run_a", end=None)
    early = _page(book, as_of=utc(MON, 20, 5))
    assert early.header.status == "provisional"
    assert "**Provisional.**" in early.text
    # Mid-session a mark may still be landing; the account waits for the close.
    assert "Shown once the session is final." in early.text

    assert write_page(book, early, directory=tmp_path).action == "written"
    final = _page(book, as_of=utc(MON, 21))
    assert write_page(book, final, directory=tmp_path).action == "rewritten"
    assert (tmp_path / final.filename).read_text(encoding="utf-8") == final.text
    again = _page(book, as_of=utc(MON, 22))
    assert write_page(book, again, directory=tmp_path).action == "unchanged"


def test_a_page_the_ledger_no_longer_produces_is_never_overwritten(
    book: Ledger, clock: PinnedClock, tmp_path: Path
) -> None:
    run(book, clock, run_id="run_a")
    page = _page(book)
    write_page(book, page, directory=tmp_path)
    target = tmp_path / page.filename
    edited = page.text.replace("| demo | clean |", "| demo | faulted |")
    target.write_text(edited, encoding="utf-8")

    outcome = write_page(book, _page(book), directory=tmp_path)
    assert outcome.action == "refused"
    assert target.read_text(encoding="utf-8") == edited


def test_text_from_outside_arrives_inert(book: Ledger, clock: PinnedClock) -> None:
    run(book, clock, run_id="run_a")
    append(
        book,
        clock,
        utc(MON, 14),
        EventType.INTENT_COMMITTED,
        IntentCommittedPayload(
            intent_id="int_r",
            run_id="run_a",
            t212_ticker="AAPL_US_EQ",
            side="buy",
            order_type="market",
            purpose="entry",
            priority_class="risk_increasing",
            quantity=Decimal(1),
            risk_token_id="tok",
        ),
    )
    append(
        book,
        clock,
        utc(MON, 14, 1),
        EventType.ORDER_REJECTED,
        OrderOutcomePayload(
            intent_id="int_r",
            run_id="run_a",
            t212_ticker="AAPL_US_EQ",
            status="rejected",
            detail="refused",
            broker_message="<script>alert(1)</script> | x",
        ),
    )
    text = _page(book).text
    assert "<script>" not in text
    assert "&lt;script&gt;alert(1)&lt;/script&gt; \\| x" in text


def test_the_latest_final_session_waits_for_the_silence_bound() -> None:
    calendar = TradingCalendar()
    live = PINNED.limits.live

    def latest(now: datetime) -> Any:
        found = latest_final_session(calendar, now=now, limits=live)
        return None if found is None else found.day

    assert latest(utc(FRI, 20, 10)) == THU
    assert latest(utc(FRI, 20, 20)) == FRI
    assert latest(utc(SAT, 10)) == FRI
    assert latest(utc(NEXT_MON, 12)) == FRI


# --------------------------------------------------------------------------
# tb journal
# --------------------------------------------------------------------------


def _cli(ledger_path: Path, *args: str) -> Any:
    return CliRunner().invoke(app, ["journal", *args, "--db", str(ledger_path)])


def _write_args(directory: Path, *extra: str) -> list[str]:
    return ["write", "--limits", str(REFERENCE_LIMITS), "--dir", str(directory), *extra]


@pytest.fixture
def written(
    ledger_path: Path, clock: PinnedClock, monkeypatch: pytest.MonkeyPatch
) -> Iterator[Path]:
    """A ledger with Monday's session in it, and `now` on Tuesday morning."""
    with Ledger(ledger_path) as opened:
        opened.initialise(created_by="test")
        _trading_day(opened, clock)
    monkeypatch.setattr("tb.cli_ops.now_utc", lambda: utc(TUE, 12))
    yield ledger_path


def test_tb_journal_writes_verifies_and_shows(written: Path, tmp_path: Path) -> None:
    pages = tmp_path / "journal"
    first = _cli(written, *_write_args(pages))
    assert first.exit_code == 0, first.output
    assert "pinned the limits" in first.output
    assert "2026-09-21 written" in first.output
    assert (pages / "2026-09-21.md").exists()

    second = _cli(written, *_write_args(pages))
    assert second.exit_code == 0, second.output
    assert "2026-09-21 unchanged" in second.output

    verified = _cli(written, "verify", "--dir", str(pages))
    assert verified.exit_code == 0, verified.output
    assert "1 page(s) are what the ledger produces" in verified.output

    target = pages / "2026-09-21.md"
    target.write_text(target.read_text(encoding="utf-8").replace("1.25", "9.25"), "utf-8")
    tampered = _cli(written, "verify", "--dir", str(pages))
    assert tampered.exit_code == 1
    assert "Trades closed" in tampered.output

    refused = _cli(written, *_write_args(pages))
    assert refused.exit_code == 1
    assert "refused" in refused.output

    shown = CliRunner().invoke(
        app,
        [
            "journal",
            "show",
            "--date",
            "2026-09-21",
            "--limits",
            str(REFERENCE_LIMITS),
            "--db",
            str(written),
        ],
    )
    assert shown.exit_code == 0, shown.output
    assert shown.output.startswith("---\ntb_journal: 1\n")
    assert "## Sessions" in shown.output


def test_tb_journal_write_catches_up_and_refuses_bad_input(written: Path, tmp_path: Path) -> None:
    pages = tmp_path / "journal"
    caught_up = _cli(written, *_write_args(pages, "--since", "2026-09-17"))
    assert caught_up.exit_code == 0, caught_up.output
    assert sorted(p.name for p in pages.glob("*.md")) == [
        "2026-09-17.md",
        "2026-09-18.md",
        "2026-09-21.md",
    ]

    ahead = _cli(written, *_write_args(pages, "--since", "2026-09-22"))
    assert ahead.exit_code == 0, ahead.output
    assert "no finished session from 2026-09-22" in ahead.output

    assert _cli(written, *_write_args(pages, "--date", "2026-09-26")).exit_code == 2
    assert _cli(written, *_write_args(pages, "--date", "monday")).exit_code == 2
    both = _write_args(pages, "--date", "2026-09-21", "--since", "2026-09-17")
    assert _cli(written, *both).exit_code == 2


def test_tb_journal_commit_anchors_the_pages_in_git(written: Path, tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    repo.mkdir()
    for argv in (
        ["init", "-q"],
        ["config", "user.email", "journal@example.invalid"],
        ["config", "user.name", "journal"],
    ):
        subprocess.run(["git", "-C", str(repo), *argv], check=True)
    pages = repo / "journal"

    result = _cli(written, *_write_args(pages, "--commit"))
    assert result.exit_code == 0, result.output
    assert "committed 1 page(s)" in result.output

    log = subprocess.run(
        ["git", "-C", str(repo), "log", "--format=%H %s", "--name-only"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    assert "journal: 2026-09-21" in log
    assert "journal/2026-09-21.md" in log and "journal/chain-heads.jsonl" in log
    head = log.split()[0]

    with Ledger(written) as ledger:
        [anchor] = ledger.conn.execute(
            "SELECT payload_json FROM event_log WHERE event_type = 'chain.anchored'"
        ).fetchall()
    assert json.loads(anchor["payload_json"])["external_ref"] == head

    again = _cli(written, *_write_args(pages, "--commit"))
    assert again.exit_code == 0, again.output
    assert "nothing new to commit" in again.output
