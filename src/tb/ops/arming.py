"""Arming real-money trading: a person's decision, made against recorded evidence.

Three things stand between this system and a real trade, and no one of them
is enough alone:

1. **The limits file enables it.** `live.enabled` is false as shipped. The file
   is mounted read-only under another user and its hash is pinned, so turning
   it on is an edit a person makes and every run records.
2. **A person arms it here,** against evidence read from the ledger — never
   typed in, never asserted:

   * a streak of clean demo sessions (`tb sessions`) at least
     `min_clean_demo_sessions` long, with at least `min_demo_closed_trades`
     trades closed across it: the loop ran the whole session, repeatedly,
     and exercised the order path doing it;
   * a passed kill-switch drill and a passed watchdog drill (`tb drill`),
     each on the demo account, in market hours, with a position held, within
     `drills_valid_days`;
   * a restore verified on another machine (`tb backup`) within
     `restore_valid_days`;
   * a ledger whose chain verifies.

   The arming names its strategies — at most `max_armed_strategies`, each
   promoted by the gate — and lapses after `arming_valid_days`. It is bound
   to the limits it was judged under: change the limits and it no longer
   holds.
3. **The live key, and only it.** `tb run --mode live` needs
   `T212_LIVE_API_KEY` with no demo key beside it, and is the one caller that
   builds a client armed for real-money writes.

While a live run trades, the loop re-reads the arming every cycle and halts
the moment it has lapsed, been disarmed, or stopped matching the limits in
force. It trades only the armed strategies, at no rung above `max_rung`.
"""

from __future__ import annotations

import json
import socket
import uuid
from collections.abc import Callable, Sequence
from dataclasses import dataclass, replace
from datetime import datetime, timedelta
from decimal import Decimal
from typing import Any

from tb.config.hard_limits import HardLimits
from tb.config.loader import PinnedLimits
from tb.core.clock import from_iso, now_utc, to_iso
from tb.core.errors import TbError
from tb.data.calendar import TradingCalendar
from tb.engine.funding import Book
from tb.ledger.events import Actor, EventType, LiveArmedPayload, LiveDisarmedPayload
from tb.ledger.store import Ledger, code_git_sha
from tb.ledger.verify import verify_chain
from tb.ops.backup import verified_restores
from tb.ops.drills import DrillKind, drills
from tb.ops.sessions import read_sessions
from tb.registry.ladder import notional_for
from tb.registry.lineage import SpecRegistry


class ArmingError(TbError):
    """Live trading could not be armed, or is not armed."""


@dataclass(frozen=True, slots=True)
class Requirement:
    """One condition for arming, with what was observed against what is required."""

    name: str
    met: bool
    observed: str
    required: str


@dataclass(frozen=True, slots=True)
class Evidence:
    requirements: tuple[Requirement, ...]
    notes: tuple[str, ...] = ()

    @property
    def ready(self) -> bool:
        return all(r.met for r in self.requirements)

    @property
    def unmet(self) -> tuple[Requirement, ...]:
        return tuple(r for r in self.requirements if not r.met)

    def as_rows(self) -> list[dict[str, Any]]:
        return [
            {"name": r.name, "met": r.met, "observed": r.observed, "required": r.required}
            for r in self.requirements
        ]


@dataclass(frozen=True, slots=True)
class Arming:
    """A live arming as the ledger holds it."""

    arming_id: str
    seq: int
    armed_at: datetime
    expires_at: datetime
    strategies: tuple[str, ...]
    config_hash: str
    max_rung: int
    armed_by: str

    def covers(self, strategy_id: str, version: int) -> bool:
        return f"{strategy_id}@v{version}" in self.strategies


@dataclass(frozen=True, slots=True)
class ArmingState:
    """Whether live trading is armed right now, and if not, exactly why not."""

    arming: Arming | None
    reason: str

    @property
    def armed(self) -> bool:
        return self.arming is not None


# --------------------------------------------------------------------------
# The evidence
# --------------------------------------------------------------------------


def live_evidence(
    ledger: Ledger,
    *,
    limits: HardLimits,
    now: datetime | None = None,
    calendar: TradingCalendar | None = None,
) -> Evidence:
    """Every requirement for arming, read from the ledger."""
    moment = now or now_utc()
    live = limits.live
    requirements: list[Requirement] = [
        Requirement(
            "live trading enabled in the limits file",
            live.enabled,
            "enabled" if live.enabled else "disabled",
            "live.enabled: true, a person's edit to hard_limits.yaml",
        )
    ]

    chain = verify_chain(ledger, stop_on_first=True)
    requirements.append(
        Requirement(
            "the ledger's chain verifies",
            chain.ok,
            "intact" if chain.ok else chain.summary(),
            "intact",
        )
    )

    record = read_sessions(ledger, limits=live, mode="demo", now=moment, calendar=calendar)
    streak = record.streak
    since = f", since {streak[0].session_date}" if streak else ""
    requirements.append(
        Requirement(
            "clean demo sessions in a row",
            len(streak) >= live.min_clean_demo_sessions,
            f"{len(streak)}{since}",
            f"at least {live.min_clean_demo_sessions}",
        )
    )
    requirements.append(
        Requirement(
            "trades closed across that streak",
            record.trades_in_streak >= live.min_demo_closed_trades,
            str(record.trades_in_streak),
            f"at least {live.min_demo_closed_trades}",
        )
    )

    window = timedelta(days=live.drills_valid_days)
    recorded = drills(ledger)
    for kind in DrillKind:
        good = [
            d
            for d in recorded
            if d.kind == kind.value
            and d.passed
            and d.mode == "demo"
            and d.market_open
            and d.n_holdings > 0
            and d.completed_at is not None
            and moment - d.completed_at <= window
        ]
        latest = good[-1] if good else None
        requirements.append(
            Requirement(
                f"{kind.value.replace('_', '-')} drill passed",
                latest is not None,
                "none"
                if latest is None or latest.completed_at is None
                else f"{latest.drill_id}, {latest.completed_at:%Y-%m-%d}",
                f"on demo, in market hours, with a position held, within "
                f"{live.drills_valid_days} days",
            )
        )

    restores = [
        r
        for r in verified_restores(
            ledger, within=timedelta(days=live.restore_valid_days), now=moment
        )
        if not r.same_host
    ]
    latest_restore = restores[-1] if restores else None
    requirements.append(
        Requirement(
            "a restore verified on another machine",
            latest_restore is not None,
            "none"
            if latest_restore is None
            else f"{latest_restore.backup_id} on {latest_restore.restored_host}, "
            f"{latest_restore.restored_at:%Y-%m-%d}",
            f"within {live.restore_valid_days} days",
        )
    )

    notes: list[str] = []
    current = code_git_sha()
    ran_on = sorted({sha for session in streak for sha in session.code_shas})
    if ran_on and current not in ran_on:
        notes.append(
            f"the demo streak ran on code {', '.join(s[:12] for s in ran_on)}; this is "
            f"{(current or 'an unknown version')[:12]}. The evidence is about the code "
            "that produced it."
        )
    return Evidence(requirements=tuple(requirements), notes=tuple(notes))


# --------------------------------------------------------------------------
# Arming and disarming
# --------------------------------------------------------------------------


def resolve_strategies(ledger: Ledger, names: Sequence[str], *, limits: HardLimits) -> list[str]:
    """Each name as `id@vN`, each promoted and tradable, within the armable count."""
    if not names:
        raise ArmingError("name the strategy to arm with --strategy; live starts with one")
    if len(set(names)) > limits.live.max_armed_strategies:
        raise ArmingError(
            f"{len(set(names))} strategies named; the limits allow "
            f"{limits.live.max_armed_strategies} to be armed at once. Raising that is a "
            "reviewed change to the limits file."
        )
    registry = SpecRegistry(ledger, per_lineage_budget_ccy=limits.loss.per_lineage_budget_ccy)
    resolved: list[str] = []
    for name in dict.fromkeys(names):
        strategy_id, _, version_text = name.partition("@v")
        if version_text:
            try:
                version = int(version_text)
            except ValueError as exc:
                raise ArmingError(f"{name!r} is not `id` or `id@vN`") from exc
        else:
            versions = [r.version for r in registry.promoted() if r.strategy_id == strategy_id]
            if not versions:
                raise ArmingError(
                    f"{strategy_id} has no promoted version. Only a strategy the gate "
                    "cleared can be armed."
                )
            version = max(versions)
        ok, why = registry.may_trade(strategy_id, version)
        if not ok:
            raise ArmingError(f"{why}. Only a strategy the gate cleared can be armed.")
        resolved.append(f"{strategy_id}@v{version}")
    return resolved


def arm_live(
    ledger: Ledger,
    *,
    pinned: PinnedLimits,
    strategies: Sequence[str],
    armed_by: str,
    now: datetime | None = None,
    calendar: TradingCalendar | None = None,
) -> Arming:
    """Record an arming, if every requirement is met. Refuses otherwise."""
    moment = now or now_utc()
    evidence = live_evidence(ledger, limits=pinned.limits, now=moment, calendar=calendar)
    if not evidence.ready:
        raise ArmingError(
            "not armed; unmet: "
            + "; ".join(f"{r.name} ({r.observed}, need {r.required})" for r in evidence.unmet)
        )
    resolved = resolve_strategies(ledger, strategies, limits=pinned.limits)
    live = pinned.limits.live
    arming_id = f"arm_{uuid.uuid4().hex[:12]}"
    expires = moment + timedelta(days=live.arming_valid_days)
    event = ledger.append(
        EventType.LIVE_ARMED,
        arming_id,
        LiveArmedPayload(
            arming_id=arming_id,
            strategies=resolved,
            expires_at=to_iso(expires),
            config_hash=pinned.config_hash,
            max_rung=live.max_rung,
            evidence=evidence.as_rows(),
            armed_by=armed_by,
            host=socket.gethostname(),
            code_git_sha=code_git_sha(),
        ),
        actor=Actor.HUMAN,
    )
    return Arming(
        arming_id=arming_id,
        seq=event.seq,
        armed_at=from_iso(event.ts_utc),
        expires_at=expires,
        strategies=tuple(resolved),
        config_hash=pinned.config_hash,
        max_rung=live.max_rung,
        armed_by=armed_by,
    )


def disarm(ledger: Ledger, *, disarmed_by: str, reason: str) -> str | None:
    """Record a disarm. Always allowed; returns the arming it ended, if any."""
    latest = _latest(ledger)
    arming_id = latest.arming_id if isinstance(latest, Arming) else None
    ledger.append(
        EventType.LIVE_DISARMED,
        arming_id or "live",
        LiveDisarmedPayload(arming_id=arming_id, disarmed_by=disarmed_by, reason=reason),
        actor=Actor.HUMAN,
    )
    return arming_id


def arming_state(ledger: Ledger, *, config_hash: str, now: datetime | None = None) -> ArmingState:
    """Whether a live run may trade now, under these limits."""
    moment = now or now_utc()
    latest = _latest(ledger)
    if latest is None:
        return ArmingState(
            None,
            "live trading has never been armed. `tb arm --live` records an arming once the "
            "evidence is in the ledger.",
        )
    if isinstance(latest, str):
        return ArmingState(None, latest)
    if moment >= latest.expires_at:
        return ArmingState(
            None,
            f"the arming {latest.arming_id} of {latest.armed_at:%Y-%m-%d} lapsed at "
            f"{latest.expires_at:%Y-%m-%d %H:%M} UTC; re-arming re-reads the evidence.",
        )
    if latest.config_hash != config_hash:
        return ArmingState(
            None,
            f"armed under limits {latest.config_hash[:12]}, running under {config_hash[:12]}: "
            "the limits changed since arming, so the evidence was judged under other rules. "
            "Re-arm.",
        )
    return ArmingState(latest, f"armed until {latest.expires_at:%Y-%m-%d %H:%M} UTC")


def _latest(ledger: Ledger) -> Arming | str | None:
    """The last arming, or why there is none now: a disarm after it."""
    latest: Arming | str | None = None
    for row in ledger.iter_events(event_types=(EventType.LIVE_ARMED, EventType.LIVE_DISARMED)):
        payload = json.loads(row["payload_json"])
        at = from_iso(str(row["ts_utc"]))
        if row["event_type"] == EventType.LIVE_ARMED.value:
            latest = Arming(
                arming_id=str(payload["arming_id"]),
                seq=int(row["seq"]),
                armed_at=at,
                expires_at=from_iso(str(payload["expires_at"])),
                strategies=tuple(str(s) for s in payload["strategies"]),
                config_hash=str(payload["config_hash"]),
                max_rung=int(payload["max_rung"]),
                armed_by=str(payload["armed_by"]),
            )
        else:
            latest = (
                f"disarmed at {at:%Y-%m-%d %H:%M} UTC by {payload.get('disarmed_by')}: "
                f"{payload.get('reason')}"
            )
    return latest


def live_permit(ledger: Ledger, *, config_hash: str) -> Callable[[datetime], str | None]:
    """What the loop asks every cycle: may this run still trade real money?"""

    def permit(at: datetime) -> str | None:
        state = arming_state(ledger, config_hash=config_hash, now=at)
        return None if state.armed else f"live trading is not armed: {state.reason}"

    return permit


# --------------------------------------------------------------------------
# The live book
# --------------------------------------------------------------------------


def live_book(
    book: Book, arming: Arming, *, limits: HardLimits, equity_ccy: Decimal | None
) -> Book:
    """The funded book narrowed to what was armed, at no rung above the cap.

    A promoted strategy that was not armed is excluded by name rather than
    dropped, so "funded but not armed" is as visible as any other exclusion.
    """
    funded = []
    excluded = list(book.excluded)
    for strategy in book.funded:
        if not arming.covers(strategy.strategy_id, strategy.version):
            excluded.append((strategy.label, f"not armed for live ({arming.arming_id})"))
            continue
        if strategy.rung > arming.max_rung:
            capped = notional_for(arming.max_rung, limits=limits, equity_ccy=equity_ccy)
            strategy = replace(
                strategy,
                rung=arming.max_rung,
                notional_ccy=min(strategy.notional_ccy, capped),
                detail=(
                    f"{strategy.detail}; live caps rung {strategy.rung} at {arming.max_rung}"
                ).lstrip("; "),
            )
        funded.append(strategy)
    return replace(book, funded=tuple(funded), excluded=tuple(excluded))
