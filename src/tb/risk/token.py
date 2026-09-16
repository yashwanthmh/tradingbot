"""`RiskToken` — the only thing `place_order` will accept.

The requirement is "every order passes the risk engine". As a *convention*
that is worth nothing: it holds until the first well-meaning refactor adds a
second call site, and nothing anywhere fails when it does. So it is encoded in
the type system instead — `Broker.place_order` takes a `RiskToken`, and a
`RiskToken` cannot be constructed outside `tb.risk.engine`.

Three mechanisms, because no one of them is sufficient:

**A capability only the engine holds.** The constructor requires `_mint`, a
module-private object of a private type. It cannot be forged from a literal,
and the single import that would grant another module this power is a
greppable line that a reviewer sees.

**A construction guard.** `__post_init__` identifies the module that called
the constructor and refuses anything but the engine. This is what closes the
hole the capability alone leaves: `dataclasses.replace(token, quantity=300)`
passes the *existing* mint through to a new instance, so a valid approval for
three shares would otherwise mint a valid approval for three hundred. The
frame depth is asserted by a test rather than assumed, because it depends on
dataclass codegen — `replace` is caught because it reports the `dataclasses`
module as the caller.

**An AST test over the whole tree.** `tests/test_risk_token.py` parses every
module and asserts the only `RiskToken(...)` call site is in the engine. This
is the mechanism that actually holds, because it fails in CI rather than at
runtime: a bypass cannot be merged, never mind executed.

The token is also **bound to the order it authorises**. It carries side,
ticker, quantity and purpose, and `authorises()` compares all four against the
order in hand. A bare "approved: yes" token would let an approval for 3 shares
be replayed to buy 300 — the same class of bug as a signed cheque with the
amount left blank.
"""

from __future__ import annotations

import inspect
from dataclasses import dataclass
from datetime import datetime
from decimal import Decimal

from tb.broker.port import OrderPurpose, Side
from tb.core.clock import now_utc
from tb.core.errors import TbError

# The module permitted to mint tokens. Stated once, because the runtime guard
# and the AST test both need this answer and two copies would drift.
ENGINE_MODULE = "tb.risk.engine"

# How long an approval stays valid. Short on purpose: a token is an assertion
# about account state — equity, deployed capital, the loss breakers — and that
# state moves. Ten seconds is comfortably longer than the submit path and far
# shorter than the interval over which a loss breaker could newly trip.
TOKEN_TTL_SECONDS = 10

# Frames between `_calling_module` and the code that wrote `RiskToken(...)`:
# its own frame, then `__post_init__`, then the dataclass-generated
# `__init__`, then the caller. Asserted in the tests rather than trusted,
# since it is a property of dataclass codegen rather than of this file.
_CALLER_DEPTH = 3


class RiskTokenError(TbError):
    """A token was constructed, used or presented in a way that is not allowed."""


class _Mint:
    """The capability to construct a token.

    A private type rather than a magic string, so it cannot be guessed or
    rebuilt from a literal — obtaining one means importing a leading-underscore
    name from this module, which is a visible line in a diff.
    """

    __slots__ = ()


_mint = _Mint()


@dataclass(frozen=True, slots=True)
class RiskToken:
    """Authorisation for one specific order, issued by the risk engine.

    Frozen, because an approval that could be edited after issue is not an
    approval. `slots`, so an attribute cannot be monkey-patched onto an
    instance.
    """

    token_id: str
    run_id: str
    t212_ticker: str
    side: Side
    purpose: OrderPurpose
    quantity: Decimal
    issued_at: datetime
    expires_at: datetime
    decision_id: str | None = None
    max_notional_ccy: Decimal | None = None
    # The engine's own reasoning, on the artefact it produced rather than only
    # in the ledger row beside it.
    rules_passed: tuple[str, ...] = ()
    mint: _Mint | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.mint, _Mint):
            raise RiskTokenError(
                "a RiskToken cannot be constructed without the mint held by "
                f"{ENGINE_MODULE}. Every order goes through the risk engine, and that "
                "is enforced here rather than trusted: a second construction site "
                "would otherwise be one refactor away, with nothing failing."
            )
        caller = _calling_module(depth=_CALLER_DEPTH)
        if caller != ENGINE_MODULE:
            raise RiskTokenError(
                f"RiskToken constructed from {caller!r}, but only {ENGINE_MODULE} may "
                "issue one. If this is new code that needs to place an order, route it "
                "through RiskEngine.evaluate rather than widening this check. (A "
                "`dataclasses.replace` of an existing token lands here too, which is "
                "deliberate: it would otherwise turn an approval for one size into an "
                "approval for another.)"
            )
        if self.quantity <= 0:
            raise RiskTokenError(
                f"{self.token_id}: non-positive approved quantity {self.quantity}. A "
                "token authorising zero shares is not a refusal — it would pass every "
                "check below it and submit a malformed order."
            )
        if self.issued_at.tzinfo is None or self.expires_at.tzinfo is None:
            raise RiskTokenError(f"{self.token_id}: token timestamps must be timezone-aware")
        if self.expires_at <= self.issued_at:
            raise RiskTokenError(
                f"{self.token_id}: expires_at is not after issued_at, so the token is "
                "dead on arrival and the submit path would read it as expired."
            )

    # -- validity ----------------------------------------------------------

    def is_expired(self, *, at: datetime | None = None) -> bool:
        return (at or now_utc()) >= self.expires_at

    def authorises(
        self,
        *,
        t212_ticker: str,
        side: Side,
        quantity: Decimal,
        purpose: OrderPurpose,
        at: datetime | None = None,
    ) -> None:
        """Raise unless this token authorises exactly this order.

        Quantity is compared as *equal*, not as a ceiling. "Approved for up to
        N" invites the caller to pick the number, and the engine sized this
        order for a reason: the cap arithmetic, the regime factor and the
        unprotected-gap bound all fed into it. A caller wanting a different
        size needs a different approval.
        """
        if self.is_expired(at=at):
            raise RiskTokenError(
                f"{self.token_id} expired at {self.expires_at.isoformat()}. A token is "
                "an assertion about account state at a moment; re-evaluate rather than "
                "extending it."
            )
        mismatches: list[str] = []
        if t212_ticker != self.t212_ticker:
            mismatches.append(f"ticker {t212_ticker!r} != approved {self.t212_ticker!r}")
        if side is not self.side:
            mismatches.append(f"side {side.value} != approved {self.side.value}")
        if purpose is not self.purpose:
            mismatches.append(f"purpose {purpose.value} != approved {self.purpose.value}")
        if quantity != self.quantity:
            mismatches.append(f"quantity {quantity} != approved {self.quantity}")
        if mismatches:
            raise RiskTokenError(
                f"{self.token_id} does not authorise this order: {'; '.join(mismatches)}. "
                "A token is bound to the order the engine evaluated — an approval for "
                "one order must not be reusable for another."
            )

    def summary(self) -> str:
        return (
            f"{self.token_id}: {self.side.value} {self.quantity} {self.t212_ticker} "
            f"({self.purpose.value}), valid until {self.expires_at.isoformat()}"
        )


def _calling_module(*, depth: int) -> str:
    """The `__name__` of the module `depth` frames up, or `<unknown>`.

    Defensive about a missing frame: `currentframe` returns `None` on
    implementations without frame support, and a guard that raised
    `AttributeError` there would read as a bug rather than as a refusal.
    `<unknown>` is not the engine, so it is refused — the fail-closed
    direction.
    """
    frame = inspect.currentframe()
    for _ in range(depth):
        if frame is None:
            return "<unknown>"
        frame = frame.f_back
    if frame is None:
        return "<unknown>"
    return str(frame.f_globals.get("__name__", "<unknown>"))
