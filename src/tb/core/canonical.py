"""Canonical serialisation and hashing.

Every hash in this system — the event chain, feature snapshots, strategy specs,
config pins — is computed over the output of `canonical_json`. If that function
is not perfectly deterministic then `feature_snapshot_hash` is a lie and the
whole replay-and-explain property collapses, so the rules are strict and
deliberately unforgiving:

* keys sorted, tightest separators, no incidental whitespace
* `Decimal` serialises as a string, never through a float — money must not
  round-trip through binary floating point
* aware datetimes serialise as canonical ISO 8601 UTC; naive ones raise
* NaN and infinity raise rather than serialise. A NaN feature value that
  silently hashes is exactly the bug this system is built to make impossible
* output is encoded UTF-8 before hashing, stated explicitly rather than left to
  a platform default
* unknown types raise. Adding a type here is a deliberate act, because a
  `__str__` fallback would make two different objects hash identically
"""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Mapping, Sequence
from datetime import datetime
from decimal import Decimal
from enum import Enum
from typing import Any
from uuid import UUID

from tb.core.clock import to_iso

# The all-zero hash. Used as `prev_hash` for the first event in a chain, so the
# genesis record is chained to a well-known constant rather than to NULL.
GENESIS_HASH = "0" * 64

ENCODING = "utf-8"


def _normalise(value: Any) -> Any:
    """Reduce an arbitrary object to JSON-serialisable primitives, strictly."""
    # bool before int: bool is a subclass of int and must stay a JSON boolean.
    if value is None or isinstance(value, (bool, str)):
        return value

    if isinstance(value, int):
        return value

    if isinstance(value, float):
        if math.isnan(value) or math.isinf(value):
            raise ValueError(
                f"refusing to hash a non-finite float ({value!r}); "
                "a NaN or infinity here means an upstream calculation is broken"
            )
        return value

    if isinstance(value, Decimal):
        if not value.is_finite():
            raise ValueError(f"refusing to hash a non-finite Decimal ({value!r})")
        # `str` on a Decimal is exact and round-trippable, unlike float().
        return str(value)

    if isinstance(value, datetime):
        # Raises on naive datetimes — see tb.core.clock.to_iso.
        return to_iso(value)

    if isinstance(value, UUID):
        return str(value)

    if isinstance(value, Enum):
        return _normalise(value.value)

    if isinstance(value, Mapping):
        out: dict[str, Any] = {}
        for key, item in value.items():
            if not isinstance(key, str):
                raise TypeError(
                    f"canonical JSON requires string keys, got {type(key).__name__}: {key!r}"
                )
            out[key] = _normalise(item)
        return out

    # str is a Sequence, but it was handled above.
    if isinstance(value, (list, tuple)) or (
        isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray))
    ):
        return [_normalise(item) for item in value]

    if isinstance(value, (set, frozenset)):
        raise TypeError(
            "refusing to hash a set: iteration order is not guaranteed across "
            "processes. Convert to a sorted list at the call site so the "
            "ordering is an explicit decision."
        )

    # Pydantic models and dataclasses: let the caller dump them first, so the
    # dump mode (python vs json, by_alias, exclude) is explicit and visible.
    raise TypeError(
        f"no canonical form defined for {type(value).__name__}. "
        "Dump it to primitives at the call site, or add a rule here."
    )


def canonical_json(payload: Any) -> str:
    """Serialise `payload` to its one canonical JSON representation."""
    return json.dumps(
        _normalise(payload),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    )


def canonical_bytes(payload: Any) -> bytes:
    """Canonical JSON, UTF-8 encoded — the exact bytes that get hashed."""
    return canonical_json(payload).encode(ENCODING)


def sha256_hex(data: bytes | str) -> str:
    """Lowercase hex SHA-256."""
    if isinstance(data, str):
        data = data.encode(ENCODING)
    return hashlib.sha256(data).hexdigest()


def hash_payload(payload: Any) -> str:
    """SHA-256 over the canonical form of `payload`."""
    return sha256_hex(canonical_bytes(payload))


def hash_file(path: str) -> str:
    """SHA-256 over a file's raw bytes.

    Raw bytes, not parsed content: for tamper detection we want a comment edit
    or a whitespace change to register too. The semantic hash of the parsed
    content is tracked separately.
    """
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(65536), b""):
            digest.update(block)
    return digest.hexdigest()
