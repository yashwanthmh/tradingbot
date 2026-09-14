"""Which instruments the bot may consider, recorded as dated snapshots.

This module exists because of a bias that cannot be fixed, only stopped from
growing and stopped from lying.

**Survivorship.** Build a universe today from the names that exist today, then
backtest it over ten years, and the backtest never sees a single company that
was delisted, acquired, or went to zero. Every surviving name survived. The
resulting equity curve is not optimistic by a little — on a ten-year window it
is optimistic by more than most strategies' entire claimed edge, and no amount
of out-of-sample discipline detects it, because the bias is in the *sample*.

Free data cannot fix this: neither Alpaca nor Yahoo will tell us what the
S&P 500 contained in 2016. So the three things this module actually does are:

1. **Append-only dated snapshots from install forward.** From today on, the
   record of what was selectable on each date is real. Ten years from now this
   is a survivorship-free universe; today it is a start.
2. **Keying on `instrument_uid`, not a ticker.** Tickers get reused. A backtest
   keyed on a ticker string will happily splice a dead issuer's history onto a
   new one's and produce a continuous-looking series belonging to two different
   businesses. An ISIN does not move.
3. **Stamping the bias where it cannot be missed.** Any backtest window that
   predates the first snapshot is marked `survivorship: unmeasured`, and that
   flag rides into the sealed vintage and into M5's promotion record. A biased
   backtest that is *labelled* biased is usable evidence; an unlabelled one is
   a trap.

Selection itself is deliberately dull: liquid US large-caps, ranked by dollar
volume, capped at `execution.max_universe_symbols`. The cap is arithmetic
rather than taste — it comes from the rate-limit budget, since every symbol in
the universe costs poll capacity every cycle, and a universe that cannot be
polled inside one decision interval is a universe whose prices are stale by
construction.
"""

from __future__ import annotations

import sqlite3
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime
from decimal import Decimal
from enum import StrEnum

from tb.broker.port import Instrument
from tb.core.canonical import hash_payload
from tb.core.clock import from_iso, now_utc, to_iso
from tb.data.provider import DataError, make_instrument_uid, uid_is_stable
from tb.ledger.events import Actor, EventType, UniverseSnapshotPayload
from tb.ledger.store import Ledger

# Instrument types this bot will hold. Long-only cash equity and plain ETFs
# only: everything else on Trading 212's list carries either leverage, an
# expiry, or a swap counterparty, and none of those are things a bot sized
# against a GBP 500 ceiling should be discovering by accident.
TRADABLE_TYPES = frozenset({"STOCK", "ETF"})

EXCLUDED_TYPE_REASON = (
    "instrument type is not plain equity or ETF — leveraged, dated and "
    "swap-based products are out of scope by design"
)


class UniverseError(DataError):
    """A universe could not be built or recorded."""


class Exclusion(StrEnum):
    """Why a candidate did not make the universe.

    Recorded per candidate rather than summarised. "Why is this name not being
    traded" is asked constantly, and a count of rejections does not answer it.
    """

    WRONG_TYPE = "wrong_type"
    NO_STABLE_ID = "no_stable_id"
    WRONG_CURRENCY = "wrong_currency"
    NOT_US = "not_us"
    UNMAPPED = "unmapped"
    ILLIQUID = "illiquid"
    BELOW_RANK_CAP = "below_rank_cap"
    BLOCKED = "blocked"


@dataclass(frozen=True, slots=True)
class Candidate:
    """An instrument considered for the universe, with its liquidity measure."""

    instrument_uid: str
    t212_ticker: str
    data_symbol: str
    currency: str | None
    instrument_type: str | None
    dollar_volume: Decimal | None = None
    short_name: str | None = None

    @property
    def has_stable_id(self) -> bool:
        return uid_is_stable(self.instrument_uid)


@dataclass(frozen=True, slots=True)
class Member:
    """One instrument in one snapshot."""

    instrument_uid: str
    t212_ticker: str
    data_symbol: str
    rank: int
    selection_reason: str
    dollar_volume: Decimal | None
    currency: str | None


@dataclass(frozen=True, slots=True)
class Rejection:
    instrument_uid: str
    t212_ticker: str
    reason: Exclusion
    detail: str


@dataclass(frozen=True, slots=True)
class UniverseSnapshot:
    """The selectable set, as of one instant.

    Immutable by construction. A snapshot that could be edited would destroy
    the only thing it is for: being able to say what was selectable on a date
    without today's knowledge leaking back into it.
    """

    snapshot_id: str
    taken_at: datetime
    members: tuple[Member, ...]
    rejections: tuple[Rejection, ...] = field(default_factory=tuple)
    selection_rule: str = ""
    currency: str | None = None
    n_candidates: int = 0

    def __len__(self) -> int:
        return len(self.members)

    @property
    def uids(self) -> tuple[str, ...]:
        return tuple(member.instrument_uid for member in self.members)

    @property
    def tickers(self) -> tuple[str, ...]:
        return tuple(member.t212_ticker for member in self.members)

    def contains(self, instrument_uid: str) -> bool:
        return instrument_uid in set(self.uids)

    @property
    def all_ids_stable(self) -> bool:
        """Whether every member is keyed by ISIN.

        A member keyed only by ticker is a place where two companies' histories
        could be spliced, so the count rides into the vintage rather than being
        assumed away.
        """
        return all(uid_is_stable(uid) for uid in self.uids)

    def content_hash(self) -> str:
        return hash_payload(
            {
                "members": [
                    {
                        "instrument_uid": member.instrument_uid,
                        "t212_ticker": member.t212_ticker,
                        "data_symbol": member.data_symbol,
                        "rank": member.rank,
                    }
                    for member in self.members
                ],
                "selection_rule": self.selection_rule,
                "currency": self.currency,
            }
        )


class SurvivorshipFlag(StrEnum):
    """How trustworthy a window's membership record is.

    Three values rather than a boolean, because "we have snapshots covering
    this window" and "we have no idea what this window contained" are different
    claims and only one of them is a reason to believe a backtest.
    """

    # Dated snapshots cover the whole window.
    MEASURED = "measured"
    # Snapshots start inside the window: the earlier part is today's names.
    PARTIAL = "partial"
    # The window predates every snapshot. Today's survivors, backtested.
    UNMEASURED = "unmeasured"

    @property
    def admissible_for_promotion(self) -> bool:
        """Whether a backtest over this window may support promotion on its own.

        `UNMEASURED` is not forbidden — on free data it is the only option for
        the ten-year window the promotion gate needs. It is *labelled*, and M5
        weighs it accordingly. The unacceptable outcome is not a biased
        backtest; it is a biased backtest nobody knew was biased.
        """
        return self is not SurvivorshipFlag.UNMEASURED


def build_candidates(
    instruments: Iterable[Instrument],
    *,
    symbol_for: dict[str, str],
    dollar_volume: dict[str, Decimal] | None = None,
) -> tuple[tuple[Candidate, ...], tuple[Rejection, ...]]:
    """Turn broker instruments into candidates, keeping every rejection.

    `symbol_for` maps a T212 ticker to a data-provider symbol — the output of
    `SymbolMap`. An instrument with no mapping is rejected here rather than
    later: a universe member the data layer cannot fetch prices for would sit in
    the universe consuming a poll slot and never producing a decision.
    """
    volumes = dollar_volume or {}
    candidates: list[Candidate] = []
    rejections: list[Rejection] = []

    for instrument in instruments:
        uid = make_instrument_uid(isin=instrument.isin, t212_ticker=instrument.ticker)
        kind = (instrument.instrument_type or "").upper()

        if kind not in TRADABLE_TYPES:
            rejections.append(
                Rejection(
                    instrument_uid=uid,
                    t212_ticker=instrument.ticker,
                    reason=Exclusion.WRONG_TYPE,
                    detail=f"{instrument.instrument_type or 'unknown type'}: "
                    f"{EXCLUDED_TYPE_REASON}",
                )
            )
            continue

        symbol = symbol_for.get(instrument.ticker)
        if not symbol:
            rejections.append(
                Rejection(
                    instrument_uid=uid,
                    t212_ticker=instrument.ticker,
                    reason=Exclusion.UNMAPPED,
                    detail="no verified data-provider symbol, so no prices can be fetched",
                )
            )
            continue

        if not uid_is_stable(uid):
            # Not fatal. A ticker-keyed member is allowed, because ISIN is
            # occasionally absent from the broker's own list, but it is recorded
            # as a place where a reused ticker could splice two histories.
            rejections.append(
                Rejection(
                    instrument_uid=uid,
                    t212_ticker=instrument.ticker,
                    reason=Exclusion.NO_STABLE_ID,
                    detail=(
                        "no ISIN, so identity is only as stable as the ticker string; "
                        "included but flagged"
                    ),
                )
            )

        candidates.append(
            Candidate(
                instrument_uid=uid,
                t212_ticker=instrument.ticker,
                data_symbol=symbol,
                currency=instrument.currency_code,
                instrument_type=instrument.instrument_type,
                dollar_volume=volumes.get(instrument.ticker),
                short_name=instrument.short_name,
            )
        )

    return tuple(candidates), tuple(rejections)


def select(
    candidates: Sequence[Candidate],
    *,
    max_symbols: int,
    preferred_currency: str | None = None,
    min_dollar_volume: Decimal | None = None,
    blocked: Iterable[str] = (),
    taken_at: datetime | None = None,
) -> UniverseSnapshot:
    """Rank candidates and take the top `max_symbols`.

    `max_symbols` is the rate-limit budget, not a preference: every symbol costs
    poll capacity every cycle, and a universe too large to poll inside one
    decision interval has stale prices by construction. Passing a larger number
    does not buy more breadth, it buys older data.

    `preferred_currency` is the account's own. Trading 212 charges 0.15% per
    conversion, so an instrument already denominated in the account currency is
    ~30bps cheaper to round-trip than an identical one that is not — which, at
    the 5-20bps edges this system is hunting, is the difference between a viable
    strategy and a fee-collection scheme. The preference is a tiebreak rather
    than a filter, because on a US-large-cap universe from a GBP account
    nothing would survive a hard filter.
    """
    if max_symbols < 1:
        raise UniverseError(f"max_universe_symbols must be at least 1, got {max_symbols}")

    blocked_set = set(blocked)
    stamp = taken_at or now_utc()
    rejections: list[Rejection] = []
    eligible: list[Candidate] = []

    for candidate in candidates:
        if candidate.t212_ticker in blocked_set or candidate.instrument_uid in blocked_set:
            rejections.append(
                Rejection(
                    instrument_uid=candidate.instrument_uid,
                    t212_ticker=candidate.t212_ticker,
                    reason=Exclusion.BLOCKED,
                    detail="blocked by the symbol map or an operator",
                )
            )
            continue
        if min_dollar_volume is not None:
            if candidate.dollar_volume is None:
                rejections.append(
                    Rejection(
                        instrument_uid=candidate.instrument_uid,
                        t212_ticker=candidate.t212_ticker,
                        reason=Exclusion.ILLIQUID,
                        detail=(
                            "no dollar-volume measurement. Treated as illiquid rather than "
                            "as liquid: an unmeasured name is one whose slippage cannot be "
                            "estimated, and the cost gate would be guessing."
                        ),
                    )
                )
                continue
            if candidate.dollar_volume < min_dollar_volume:
                rejections.append(
                    Rejection(
                        instrument_uid=candidate.instrument_uid,
                        t212_ticker=candidate.t212_ticker,
                        reason=Exclusion.ILLIQUID,
                        detail=(
                            f"dollar volume {candidate.dollar_volume} is below the "
                            f"{min_dollar_volume} floor"
                        ),
                    )
                )
                continue
        eligible.append(candidate)

    wanted = (preferred_currency or "").upper()
    ordered = sorted(
        eligible,
        key=lambda c: (
            -(c.dollar_volume or Decimal(0)),
            # Same-currency first among equals, then ticker so the ordering is
            # total. A non-deterministic universe would make two runs of the
            # same backtest select different names.
            0 if wanted and (c.currency or "").upper() == wanted else 1,
            c.t212_ticker,
        ),
    )

    members = tuple(
        Member(
            instrument_uid=candidate.instrument_uid,
            t212_ticker=candidate.t212_ticker,
            data_symbol=candidate.data_symbol,
            rank=index,
            selection_reason=(
                f"rank {index} by dollar volume"
                + (
                    " (account currency)"
                    if wanted and (candidate.currency or "").upper() == wanted
                    else ""
                )
            ),
            dollar_volume=candidate.dollar_volume,
            currency=candidate.currency,
        )
        for index, candidate in enumerate(ordered[:max_symbols], start=1)
    )

    for candidate in ordered[max_symbols:]:
        rejections.append(
            Rejection(
                instrument_uid=candidate.instrument_uid,
                t212_ticker=candidate.t212_ticker,
                reason=Exclusion.BELOW_RANK_CAP,
                detail=(
                    f"outside the top {max_symbols} by dollar volume. The cap is the "
                    "rate-limit budget: more symbols means older prices, not more breadth."
                ),
            )
        )

    rule = (
        f"top {max_symbols} by dollar volume"
        + (f", preferring {wanted}" if wanted else "")
        + (f", min dollar volume {min_dollar_volume}" if min_dollar_volume else "")
    )
    return UniverseSnapshot(
        snapshot_id=_snapshot_id(stamp, members),
        taken_at=stamp,
        members=members,
        rejections=tuple(rejections),
        selection_rule=rule,
        currency=wanted or None,
        n_candidates=len(candidates),
    )


class UniverseStore:
    """Append-only dated membership, in `universe_snapshots`."""

    def __init__(self, ledger: Ledger, *, run_id: str | None = None) -> None:
        self._ledger = ledger
        self._run_id = run_id

    def record(self, snapshot: UniverseSnapshot) -> bool:
        """Store a snapshot and emit its event. Returns False if already held.

        Rows and event go in one transaction, so a snapshot can never exist
        without the event that admits it — the same rule the bar store uses for
        partitions, and for the same reason: the ledger defines the dataset.
        """
        existing = self._ledger.conn.execute(
            "SELECT 1 FROM universe_snapshots WHERE snapshot_id = ? LIMIT 1",
            (snapshot.snapshot_id,),
        ).fetchone()
        if existing is not None:
            return False

        with self._ledger.transaction() as tx:
            for member in snapshot.members:
                tx.execute(
                    "INSERT INTO universe_snapshots (snapshot_id, taken_at, instrument_uid, "
                    "t212_ticker, data_symbol, rank, selection_reason, dollar_volume, currency) "
                    "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
                    (
                        snapshot.snapshot_id,
                        to_iso(snapshot.taken_at),
                        member.instrument_uid,
                        member.t212_ticker,
                        member.data_symbol,
                        member.rank,
                        member.selection_reason,
                        None if member.dollar_volume is None else str(member.dollar_volume),
                        member.currency,
                    ),
                )
            tx.append(
                EventType.DATA_UNIVERSE_SNAPSHOT_TAKEN,
                snapshot.snapshot_id,
                UniverseSnapshotPayload(
                    snapshot_id=snapshot.snapshot_id,
                    n_members=len(snapshot.members),
                    n_candidates_considered=snapshot.n_candidates,
                    selection_rule=snapshot.selection_rule,
                    members=[member.instrument_uid for member in snapshot.members],
                    currency=snapshot.currency,
                ),
                actor=Actor.SYSTEM,
                run_id=self._run_id,
            )
        return True

    def latest(self) -> UniverseSnapshot | None:
        row = self._ledger.conn.execute(
            "SELECT snapshot_id FROM universe_snapshots ORDER BY taken_at DESC LIMIT 1"
        ).fetchone()
        return None if row is None else self.get(str(row["snapshot_id"]))

    def as_of(self, moment: datetime) -> UniverseSnapshot | None:
        """The snapshot in force at `moment`.

        The point-in-time membership query. Note what it does *not* do: fall
        back to the newest snapshot when `moment` predates them all. That
        fallback is the survivorship bug in one line — it would answer a
        question about 2019 with today's survivors and look like a successful
        lookup.
        """
        if moment.tzinfo is None:
            raise UniverseError("moment must be timezone-aware")
        row = self._ledger.conn.execute(
            "SELECT snapshot_id FROM universe_snapshots WHERE taken_at <= ? "
            "ORDER BY taken_at DESC LIMIT 1",
            (to_iso(moment),),
        ).fetchone()
        return None if row is None else self.get(str(row["snapshot_id"]))

    def get(self, snapshot_id: str) -> UniverseSnapshot | None:
        rows: list[sqlite3.Row] = self._ledger.conn.execute(
            "SELECT * FROM universe_snapshots WHERE snapshot_id = ? ORDER BY rank ASC",
            (snapshot_id,),
        ).fetchall()
        if not rows:
            return None
        return UniverseSnapshot(
            snapshot_id=snapshot_id,
            taken_at=from_iso(str(rows[0]["taken_at"])),
            members=tuple(_member_from_row(row) for row in rows),
        )

    def first_snapshot_at(self) -> datetime | None:
        """When membership started being recorded.

        The boundary between a point-in-time universe and today's survivors.
        Everything before it is `UNMEASURED`.
        """
        row = self._ledger.conn.execute(
            "SELECT MIN(taken_at) AS earliest FROM universe_snapshots"
        ).fetchone()
        earliest = None if row is None else row["earliest"]
        return None if earliest is None else from_iso(str(earliest))

    def survivorship_for(
        self, *, window_start: datetime, window_end: datetime
    ) -> tuple[SurvivorshipFlag, str]:
        """Classify a backtest window's membership record.

        Returns the flag and the sentence that goes into the vintage, because
        this is the caveat a promotion decision is weighed against and it should
        not have to be reconstructed from a flag name.
        """
        if window_end < window_start:
            raise UniverseError("window runs backwards")
        first = self.first_snapshot_at()
        if first is None:
            return (
                SurvivorshipFlag.UNMEASURED,
                "no universe snapshots exist, so this window is today's surviving names "
                "backtested over history: every member survived, by construction",
            )
        if first <= window_start:
            return (
                SurvivorshipFlag.MEASURED,
                f"dated snapshots cover the whole window from {first.date()}",
            )
        if first <= window_end:
            return (
                SurvivorshipFlag.PARTIAL,
                f"snapshots begin {first.date()}, inside the window; everything before that "
                "is today's surviving names and is survivorship-biased",
            )
        return (
            SurvivorshipFlag.UNMEASURED,
            f"the window ends {window_end.date()}, before the first snapshot {first.date()}: "
            "membership is entirely today's survivors",
        )

    def changes_between(self, earlier: str, later: str) -> tuple[tuple[str, ...], tuple[str, ...]]:
        """`(added, removed)` between two snapshots, by `instrument_uid`.

        A name that leaves the universe is the interesting direction: it may
        have been delisted, acquired, or simply fallen below the liquidity
        floor, and only the first two mean a held position needs attention.
        """
        before, after = self.get(earlier), self.get(later)
        if before is None or after is None:
            raise UniverseError(f"unknown snapshot: {earlier if before is None else later}")
        prior, current = set(before.uids), set(after.uids)
        return tuple(sorted(current - prior)), tuple(sorted(prior - current))

    def table_hash(self) -> str:
        """Content hash of all membership, for a sealed vintage."""
        rows = self._ledger.conn.execute(
            "SELECT snapshot_id, taken_at, instrument_uid, rank FROM universe_snapshots "
            "ORDER BY snapshot_id, rank"
        ).fetchall()
        return hash_payload(
            [
                {key: (None if row[key] is None else str(row[key])) for key in row.keys()}  # noqa: SIM118
                for row in rows
            ]
        )


def _snapshot_id(taken_at: datetime, members: Sequence[Member]) -> str:
    """A deterministic id from the date and the membership.

    Content-derived, so re-running the selector on the same day with the same
    result is idempotent rather than appending a near-duplicate snapshot every
    time the command is invoked.
    """
    digest = hash_payload(
        {
            "date": taken_at.astimezone(UTC).date().isoformat(),
            "members": [member.instrument_uid for member in members],
        }
    )
    return f"uni_{taken_at.astimezone(UTC).date().isoformat()}_{digest[:12]}"


def _member_from_row(row: sqlite3.Row) -> Member:
    volume = row["dollar_volume"]
    return Member(
        instrument_uid=str(row["instrument_uid"]),
        t212_ticker=str(row["t212_ticker"] or ""),
        data_symbol=str(row["data_symbol"] or ""),
        rank=int(row["rank"] or 0),
        selection_reason=str(row["selection_reason"] or ""),
        dollar_volume=None if volume is None else Decimal(str(volume)),
        currency=None if row["currency"] is None else str(row["currency"]),
    )


def dollar_volume_from_bars(bars: Iterable[object], *, days: int = 20) -> Decimal | None:
    """Average daily dollar volume over the most recent `days` bars.

    `close * volume`, not volume alone: a 5-dollar stock trading ten million
    shares is less liquid than a 400-dollar stock trading a million, and
    position sizing is in currency. Returns None rather than zero when there is
    nothing to measure — `select` treats an unmeasured name as illiquid, which
    is the conservative direction.
    """
    priced: list[tuple[datetime, Decimal]] = []
    for bar in bars:
        volume = getattr(bar, "volume", None)
        close = getattr(bar, "close", None)
        opened = getattr(bar, "bar_open_utc", None)
        if volume is None or close is None or opened is None:
            continue
        priced.append((opened, Decimal(volume) * close))
    if not priced:
        return None
    recent = [value for _, value in sorted(priced, key=lambda pair: pair[0])[-days:]]
    return sum(recent, Decimal(0)) / Decimal(len(recent))
