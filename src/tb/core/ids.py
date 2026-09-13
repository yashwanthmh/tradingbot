"""Identifier generation.

Two flavours, and the distinction matters:

* `new_id` is random. Correct for things that happen once — a run, a halt, an
  anchor.
* `deterministic_id` is derived from the facts that define the thing. Correct
  for order intents, where a retry of the *same* decision must produce the
  *same* id. Minting a fresh random id on retry is precisely how a crashed
  process double-fills: two ids, two orders, one intention.
"""

from __future__ import annotations

import uuid

from tb.core.canonical import hash_payload

_ID_HEX_LEN = 12


def new_id(prefix: str, *, length: int = _ID_HEX_LEN) -> str:
    """A fresh random identifier, e.g. `run_9f2a1c4b7e05`."""
    return f"{prefix}_{uuid.uuid4().hex[:length]}"


def new_run_id() -> str:
    return new_id("run")


def deterministic_id(prefix: str, *, parts: dict[str, object], length: int = 16) -> str:
    """An identifier derived from the facts that define the thing.

    Used for order intents in M4. Because it is a pure function of its inputs,
    re-deriving it after a crash yields the id already in the write-ahead log,
    so the reconciler can recognise the attempt rather than duplicate it.
    """
    return f"{prefix}_{hash_payload(parts)[:length]}"
