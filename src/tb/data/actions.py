"""Corporate actions as dated, revisable facts.

The store side of `adjustments.py`: this module writes and reads rows, that one
does arithmetic and touches nothing. The split is so the factor properties can
be tested exhaustively without a database, and so the only way a factor can
reach a price is through a function whose inputs are entirely visible.

Three jobs live here, in ascending order of how much they are worth.

**Recording.** Provider actions arrive as `RawAction`; they become
`corporate_actions` rows with a deterministic `action_id`, so re-fetching the
same window is idempotent. A vendor that *changes* a ratio produces a new row
rather than an update, and the old one is marked superseded — an as-of query
over an earlier instant still sees what was believed then, which is the entire
reason the table is append-only.

**Broker dividend reconciliation.** The highest-value check in the data layer,
and the argument for it is short: if a provider says a company paid a dividend
and Trading 212 credited no cash for a position we held through the ex-date,
then either the action data is wrong or **we are tracking a different company
than the one we own**. That second case is the exact failure the symbol map
exists to prevent, and this is the only place in the system where the two
venues can be checked against each other on a fact neither of them can fudge.

**Residual detection.** Each session, look at `prev_close / next_open` per
instrument. A large gap that a small-integer ratio explains, with no action row
behind it, is an unrecorded split — which is what a vendor back-adjusting its
cache without reporting an action looks like from the outside, and is Yahoo's
actual behaviour. The response is to **block new entries in that symbol**, not
to invent a factor: acting on a guessed ratio would be sizing real money off an
inference.
"""

from __future__ import annotations

import itertools
import sqlite3
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from datetime import UTC, date, datetime
from decimal import Decimal

from tb.core.canonical import hash_payload
from tb.core.clock import from_iso, now_utc, to_iso
from tb.data.adjustments import (
    ActionType,
    CorporateAction,
    SplitSuspicion,
    detect_unexplained_split,
    latest_vintages,
)
from tb.data.provider import Bar, DataError, RawAction, Resolution
from tb.ledger.events import (
    ActionReconciledPayload,
    ActionRecordedPayload,
    Actor,
    EventType,
)
from tb.ledger.store import Ledger, LedgerTransaction

# How far either side of the ex-date a broker credit may land and still count as
# the same dividend. Trading 212 pays on the payable date, which is typically a
# few weeks after the ex-date and is not reported per-action, so the match is by
# instrument and amount within a window rather than by date equality.
DIVIDEND_MATCH_WINDOW_DAYS = 45

# Broker and provider amounts will not agree to the penny: withholding tax,
# fractional-share rounding and FX all sit between them. The tolerance is
# generous on purpose — this check is looking for "no credit at all", which is
# the signal that means something is badly wrong, not for a rounding audit.
DIVIDEND_AMOUNT_TOLERANCE_PCT = Decimal("35")


class ActionError(DataError):
    """A corporate action could not be recorded or reconciled."""


def make_action_id(
    *, instrument_uid: str, action_type: str, effective_date: str, ratio: str, amount: str
) -> str:
    """A deterministic id, so re-fetching a window cannot duplicate a row.

    Derived from the action's *content* rather than from a counter: two runs
    that see the same split produce the same id, and a vendor that restates the
    ratio produces a different one — which is precisely the distinction between
    "already have this" and "this is a revision".
    """
    digest = hash_payload(
        {
            "instrument_uid": instrument_uid,
            "action_type": action_type,
            "effective_date": effective_date,
            "ratio": ratio,
            "amount": amount,
        }
    )
    return f"act_{digest[:24]}"


def from_raw(raw: RawAction) -> CorporateAction:
    """Convert a provider action into the stored form, with its id."""
    try:
        action_type = ActionType(raw.action_type)
    except ValueError as exc:
        raise ActionError(
            f"unknown action type {raw.action_type!r} from {raw.provider}. Refusing to store "
            "it: an action the factor algebra does not understand would be silently ignored "
            "at adjustment time, which is worse than not having it."
        ) from exc

    ratio = (
        f"{raw.ratio_num}/{raw.ratio_den}"
        if raw.ratio_num is not None and raw.ratio_den is not None
        else ""
    )
    amount = "" if raw.gross_amount is None else str(raw.gross_amount)
    effective = _as_date(raw.effective_date)

    return CorporateAction(
        action_id=make_action_id(
            instrument_uid=raw.instrument_uid,
            action_type=raw.action_type,
            effective_date=effective.isoformat(),
            ratio=ratio,
            amount=amount,
        ),
        instrument_uid=raw.instrument_uid,
        action_type=action_type,
        effective_date=effective,
        known_at_utc=raw.known_at_utc,
        source_provider=raw.provider,
        ratio_num=raw.ratio_num,
        ratio_den=raw.ratio_den,
        gross_amount=raw.gross_amount,
        currency=raw.currency,
        new_symbol=raw.new_symbol,
        declared_date=None if raw.declared_date is None else _as_date(raw.declared_date),
    )


@dataclass(slots=True)
class RecordResult:
    recorded: int = 0
    already_held: int = 0
    superseded: tuple[str, ...] = field(default_factory=tuple)
    rejected: tuple[str, ...] = field(default_factory=tuple)

    @property
    def changed(self) -> bool:
        return bool(self.recorded or self.superseded)


@dataclass(frozen=True, slots=True)
class DividendMatch:
    """One provider dividend checked against broker cash."""

    action_id: str
    instrument_uid: str
    effective_date: date
    matched: bool
    provider_amount: Decimal | None
    broker_amount: Decimal | None
    detail: str

    @property
    def is_identity_risk(self) -> bool:
        """Whether this mismatch could mean we hold a different company.

        A provider dividend with *no* broker credit at all on a position held
        through the ex-date is the shape that matters. An amount that disagrees
        is ordinary — withholding tax alone is 15-30%.
        """
        return not self.matched and self.broker_amount is None


class ActionStore:
    """Reads and writes `corporate_actions`, and reconciles them."""

    def __init__(self, ledger: Ledger, *, run_id: str | None = None) -> None:
        self._ledger = ledger
        self._run_id = run_id

    # -- recording ---------------------------------------------------------

    def record(self, actions: Iterable[CorporateAction]) -> RecordResult:
        """Store actions, one transaction per batch, emitting an event each.

        Supersession is resolved here rather than at read time: a new row whose
        identity matches an existing one with a *different* id is a restatement,
        so the old row's `superseded_by` is set. The old row stays, because an
        as-of query over an instant before the restatement must still return the
        ratio that was believed then.
        """
        result = RecordResult()
        pending = list(actions)
        if not pending:
            return result

        rejected: list[str] = []
        superseded: list[str] = []

        with self._ledger.transaction() as tx:
            for action in pending:
                existing = tx.execute(
                    "SELECT action_id FROM corporate_actions WHERE action_id = ?",
                    (action.action_id,),
                ).fetchone()
                if existing is not None:
                    result.already_held += 1
                    continue

                siblings = tx.execute(
                    "SELECT action_id, known_at_utc FROM corporate_actions "
                    "WHERE instrument_uid = ? AND action_type = ? AND effective_date = ? "
                    "AND superseded_by IS NULL",
                    (
                        action.instrument_uid,
                        action.action_type.value,
                        action.effective_date.isoformat(),
                    ),
                ).fetchall()

                # Every sibling is examined before anything is written. Deciding
                # row by row would let a batch supersede one vintage and then
                # abort on the next, leaving a `superseded_by` pointing at an
                # action_id that was never inserted.
                newer = [
                    str(row["action_id"])
                    for row in siblings
                    if from_iso(str(row["known_at_utc"])) > action.known_at_utc
                ]
                if newer:
                    # What we hold was learned *later* than this. Backfilling an
                    # older vintage must not retire a newer fact.
                    rejected.append(
                        f"{action.action_id}: {', '.join(newer)} already records the same "
                        f"{action.action_type.value} on {action.effective_date} with a later "
                        "knowledge time; not superseding it"
                    )
                    continue

                for row in siblings:
                    sibling_id = str(row["action_id"])
                    tx.execute(
                        "UPDATE corporate_actions SET superseded_by = ? WHERE action_id = ?",
                        (action.action_id, sibling_id),
                    )
                    superseded.append(sibling_id)

                _insert(tx, action)
                tx.append(
                    EventType.DATA_ACTION_RECORDED,
                    action.instrument_uid,
                    ActionRecordedPayload(
                        action_id=action.action_id,
                        instrument_uid=action.instrument_uid,
                        action_type=action.action_type.value,
                        effective_date=action.effective_date.isoformat(),
                        known_at_utc=to_iso(action.known_at_utc),
                        ratio_num=action.ratio_num,
                        ratio_den=action.ratio_den,
                        gross_amount=action.gross_amount,
                        currency=action.currency,
                        source_provider=action.source_provider,
                        inferred_from_price_jump=action.inferred_from_price_jump,
                    ),
                    actor=Actor.SYSTEM,
                    run_id=self._run_id,
                )
                result.recorded += 1

        result.rejected = tuple(rejected)
        result.superseded = tuple(superseded)
        return result

    def record_raw(self, raw: Iterable[RawAction]) -> RecordResult:
        """Convert and record provider actions in one call."""
        return self.record(from_raw(item) for item in raw)

    # -- reading -----------------------------------------------------------

    def actions_for(
        self,
        instrument_uid: str,
        *,
        as_of: datetime | None = None,
        include_superseded: bool = True,
    ) -> tuple[CorporateAction, ...]:
        """Every action on an instrument, optionally resolved as of an instant.

        `include_superseded` defaults to True and the as-of resolution is done
        in `latest_vintages`, not in SQL. A `WHERE superseded_by IS NULL` filter
        would answer "what do we believe now", which is the wrong question for a
        backtest: the row that was current at the as-of instant is exactly the
        one that flag retires.
        """
        # Two fixed statements rather than a concatenated clause: nothing here
        # is interpolated, so there is no query to build.
        sql = (
            "SELECT * FROM corporate_actions WHERE instrument_uid = ?"
            if include_superseded
            else "SELECT * FROM corporate_actions WHERE instrument_uid = ? "
            "AND superseded_by IS NULL"
        )
        rows = self._ledger.conn.execute(sql, (instrument_uid,)).fetchall()
        actions = tuple(_from_row(row) for row in rows)
        if as_of is None:
            return tuple(sorted(actions, key=lambda a: (a.effective_date, a.known_at_utc)))
        return latest_vintages(actions, as_of)

    def instruments_with_actions(self) -> tuple[str, ...]:
        rows = self._ledger.conn.execute(
            "SELECT DISTINCT instrument_uid FROM corporate_actions"
        ).fetchall()
        return tuple(sorted(str(row["instrument_uid"]) for row in rows))

    # -- broker reconciliation --------------------------------------------

    def reconcile_dividends(
        self,
        instrument_uid: str,
        *,
        broker_credits: Sequence[tuple[date, Decimal]],
        held_through: Sequence[tuple[date, date]] = (),
        as_of: datetime | None = None,
        emit_events: bool = True,
    ) -> tuple[DividendMatch, ...]:
        """Check provider dividends against cash the broker actually paid.

        `broker_credits` is `(paid_on, amount)` from `Endpoint.HISTORY_DIVIDENDS`.
        `held_through` is the spans during which a position existed; a dividend
        whose ex-date falls outside all of them is **skipped, not failed** —
        there is no reason to expect a credit for a stock we did not own, and
        counting those as mismatches would bury the one case that matters under
        noise from the whole universe.

        A dividend with no credit at all, on a position held through its
        ex-date, is flagged as an identity risk: the symbol map may be pointing
        at a different company than the one in the account.
        """
        matches: list[DividendMatch] = []
        unclaimed = list(broker_credits)

        dividends = [
            action
            for action in self.actions_for(instrument_uid, as_of=as_of)
            if action.action_type is ActionType.CASH_DIVIDEND
        ]

        for action in dividends:
            if held_through and not any(
                start <= action.effective_date <= end for start, end in held_through
            ):
                matches.append(
                    DividendMatch(
                        action_id=action.action_id,
                        instrument_uid=instrument_uid,
                        effective_date=action.effective_date,
                        matched=True,
                        provider_amount=action.gross_amount,
                        broker_amount=None,
                        detail="no position held through the ex-date, so no credit is expected",
                    )
                )
                continue

            candidate = _closest_credit(action.effective_date, unclaimed)
            if candidate is None:
                matches.append(
                    DividendMatch(
                        action_id=action.action_id,
                        instrument_uid=instrument_uid,
                        effective_date=action.effective_date,
                        matched=False,
                        provider_amount=action.gross_amount,
                        broker_amount=None,
                        detail=(
                            "the provider reports a dividend but the broker credited no cash "
                            "within "
                            f"{DIVIDEND_MATCH_WINDOW_DAYS} days of the ex-date, on a position "
                            "held through it. Either the action data is wrong or this mapping "
                            "points at a different company than the one held."
                        ),
                    )
                )
                continue

            unclaimed.remove(candidate)
            _, amount = candidate
            assert action.gross_amount is not None  # CorporateAction.__post_init__
            drift = abs(amount - action.gross_amount) / action.gross_amount * Decimal(100)
            within = drift <= DIVIDEND_AMOUNT_TOLERANCE_PCT
            matches.append(
                DividendMatch(
                    action_id=action.action_id,
                    instrument_uid=instrument_uid,
                    effective_date=action.effective_date,
                    matched=within,
                    provider_amount=action.gross_amount,
                    broker_amount=amount,
                    detail=(
                        f"broker credit differs by {drift:.1f}%"
                        + (
                            " — within tolerance for withholding tax, fractional shares and FX"
                            if within
                            else f", beyond the {DIVIDEND_AMOUNT_TOLERANCE_PCT}% tolerance; "
                            "too large to be withholding alone"
                        )
                    ),
                )
            )

        if emit_events:
            self._emit_reconciliations(matches)
        return tuple(matches)

    def _emit_reconciliations(self, matches: Iterable[DividendMatch]) -> None:
        pending = list(matches)
        if not pending:
            return
        with self._ledger.transaction() as tx:
            for match in pending:
                tx.execute(
                    "UPDATE corporate_actions SET reconciled_with_broker = ?, "
                    "reconcile_note = ? WHERE action_id = ?",
                    (int(match.matched), match.detail, match.action_id),
                )
                tx.append(
                    EventType.DATA_ACTION_RECONCILED,
                    match.instrument_uid,
                    ActionReconciledPayload(
                        action_id=match.action_id,
                        instrument_uid=match.instrument_uid,
                        matched=match.matched,
                        provider_amount=match.provider_amount,
                        broker_amount=match.broker_amount,
                        detail=match.detail,
                    ),
                    actor=Actor.SYSTEM,
                    run_id=self._run_id,
                )

    # -- residual detection ------------------------------------------------

    def scan_for_unexplained_splits(
        self,
        instrument_uid: str,
        bars: Iterable[Bar],
        *,
        as_of: datetime | None = None,
    ) -> tuple[SplitSuspicion, ...]:
        """Look for splits nobody reported, from the price path alone.

        Runs on consecutive *daily* bars of the raw series. Intraday bars would
        drown the detector in overnight gaps, and an adjusted series has the
        jump already removed — a detector that can never fire is worse than
        none, because it reads as evidence of absence.
        """
        known = self.actions_for(instrument_uid, as_of=as_of)
        daily = sorted(
            (bar for bar in bars if bar.resolution is Resolution.DAILY),
            key=lambda b: b.bar_open_utc,
        )
        suspicions: list[SplitSuspicion] = []
        for previous, current in itertools.pairwise(daily):
            suspicion = detect_unexplained_split(
                instrument_uid=instrument_uid,
                prev_close=previous.close,
                next_open=current.open,
                effective_date=current.bar_open_utc.astimezone(UTC).date(),
                known_actions=known,
            )
            if suspicion is not None:
                suspicions.append(suspicion)
        return tuple(suspicions)

    def record_suspicion(self, suspicion: SplitSuspicion) -> RecordResult:
        """Record an inferred split as a fact, flagged as inferred.

        Stored so the audit and the symbol gate can see it, and marked
        `inferred_from_price_jump` so the factor algebra can be asked to exclude
        it. The ratio is *recorded*, never acted on: adjusting real prices by a
        guessed ratio would size positions off an inference, and the correct
        response to an unexplained split is to stop entering that symbol until
        a provider confirms it.
        """
        ratio = f"{suspicion.implied_ratio.numerator}/{suspicion.implied_ratio.denominator}"
        return self.record(
            [
                CorporateAction(
                    action_id=make_action_id(
                        instrument_uid=suspicion.instrument_uid,
                        action_type="split",
                        effective_date=suspicion.effective_date.isoformat(),
                        ratio=ratio,
                        amount="inferred",
                    ),
                    instrument_uid=suspicion.instrument_uid,
                    action_type=ActionType.SPLIT,
                    effective_date=suspicion.effective_date,
                    known_at_utc=now_utc(),
                    source_provider="residual_detector",
                    ratio_num=suspicion.implied_ratio.numerator,
                    ratio_den=suspicion.implied_ratio.denominator,
                    inferred_from_price_jump=True,
                )
            ]
        )


# --------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------


def _insert(tx: LedgerTransaction, action: CorporateAction) -> None:
    tx.execute(
        """
        INSERT INTO corporate_actions (
            action_id, instrument_uid, action_type, effective_date, known_at_utc,
            declared_date, ratio_num, ratio_den, gross_amount, currency, new_symbol,
            source_provider, payload_hash, superseded_by, reconciled_with_broker
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, NULL, 0)
        """,
        (
            action.action_id,
            action.instrument_uid,
            action.action_type.value,
            action.effective_date.isoformat(),
            to_iso(action.known_at_utc),
            None if action.declared_date is None else action.declared_date.isoformat(),
            action.ratio_num,
            action.ratio_den,
            # Stored as text, not REAL: a dividend read back as a float and
            # multiplied into a total-return factor would make the factor
            # depend on binary rounding.
            None if action.gross_amount is None else str(action.gross_amount),
            action.currency,
            action.new_symbol,
            action.source_provider,
            _payload_hash(action),
        ),
    )


def _as_date(text: str | date) -> date:
    if isinstance(text, date):
        return text
    try:
        return date.fromisoformat(text.strip()[:10])
    except ValueError as exc:
        raise ActionError(f"unparseable action date {text!r}") from exc


def _payload_hash(action: CorporateAction) -> str:
    return hash_payload(
        {
            "instrument_uid": action.instrument_uid,
            "action_type": action.action_type.value,
            "effective_date": action.effective_date.isoformat(),
            "known_at_utc": to_iso(action.known_at_utc),
            "ratio_num": action.ratio_num,
            "ratio_den": action.ratio_den,
            "gross_amount": None if action.gross_amount is None else str(action.gross_amount),
            "currency": action.currency,
            "new_symbol": action.new_symbol,
            "declared_date": (
                None if action.declared_date is None else action.declared_date.isoformat()
            ),
            "source_provider": action.source_provider,
            "inferred_from_price_jump": action.inferred_from_price_jump,
        }
    )


def _closest_credit(
    ex_date: date, credits: Sequence[tuple[date, Decimal]]
) -> tuple[date, Decimal] | None:
    """The broker credit nearest the ex-date, within the matching window.

    Nearest rather than first: a name paying quarterly has four credits a year,
    and matching by order would pair each dividend with the wrong quarter as
    soon as one is missing — turning a single gap into four mismatches.
    """
    best: tuple[date, Decimal] | None = None
    best_gap = DIVIDEND_MATCH_WINDOW_DAYS + 1
    for paid_on, amount in credits:
        gap = abs((paid_on - ex_date).days)
        if gap <= DIVIDEND_MATCH_WINDOW_DAYS and gap < best_gap:
            best, best_gap = (paid_on, amount), gap
    return best


def _from_row(row: sqlite3.Row) -> CorporateAction:
    get = row.__getitem__
    gross = get("gross_amount")
    declared = get("declared_date")
    return CorporateAction(
        action_id=str(get("action_id")),
        instrument_uid=str(get("instrument_uid")),
        action_type=ActionType(str(get("action_type"))),
        effective_date=_as_date(str(get("effective_date"))),
        known_at_utc=from_iso(str(get("known_at_utc"))),
        source_provider=str(get("source_provider")),
        ratio_num=None if get("ratio_num") is None else int(get("ratio_num")),
        ratio_den=None if get("ratio_den") is None else int(get("ratio_den")),
        gross_amount=None if gross is None else Decimal(str(gross)),
        currency=None if get("currency") is None else str(get("currency")),
        new_symbol=None if get("new_symbol") is None else str(get("new_symbol")),
        declared_date=None if declared is None else _as_date(str(declared)),
        superseded_by=(None if get("superseded_by") is None else str(get("superseded_by"))),
        inferred_from_price_jump=str(get("source_provider")) == "residual_detector",
    )
