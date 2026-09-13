"""Error hierarchy.

The distinction that matters operationally: a `HaltRequired` is not a bug, it is
the system correctly refusing to continue. It must surface as a halt with an
alert, never as a stack trace that a retry loop swallows.
"""

from __future__ import annotations


class TbError(Exception):
    """Base class for everything this package raises deliberately."""


class ConfigError(TbError):
    """The hard limits or runtime config is missing, malformed, or unusable."""


class ConfigDriftError(ConfigError):
    """The hard-limits file changed underneath a running process.

    Raised by the per-cycle re-verification. Always a halt: the limits the
    process validated its behaviour against are no longer the limits on disk.
    """


class LedgerError(TbError):
    """The audit ledger could not be read, written, or trusted."""


class ChainIntegrityError(LedgerError):
    """The hash chain does not verify.

    Carries the exact sequence number where verification failed, because
    "the ledger is corrupt" is not an actionable message.
    """

    def __init__(self, message: str, *, seq: int) -> None:
        super().__init__(f"{message} (at seq={seq})")
        self.seq = seq


class LedgerUnwritableError(LedgerError):
    """The ledger cannot be written to.

    A halt, not a warning. An unrecorded order is worse than no order: it is a
    position the reconciler cannot attribute to any intent.
    """


class HaltRequired(TbError):
    """The system must stop trading now.

    Raised by safety checks rather than by failures. `trigger` matches the
    `halts.trigger` column so the reason survives into the ledger.
    """

    def __init__(self, trigger: str, detail: str) -> None:
        super().__init__(f"HALT [{trigger}]: {detail}")
        self.trigger = trigger
        self.detail = detail


class Killed(HaltRequired):
    """The kill switch is engaged, or its state could not be determined.

    Fail-closed: an unreadable switch is a engaged switch.
    """

    def __init__(self, detail: str) -> None:
        super().__init__("kill_switch", detail)
