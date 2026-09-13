"""Canonical serialisation.

Every hash in this system is computed over `canonical_json` output. If it is not
perfectly deterministic then `feature_snapshot_hash` is a lie and no decision
can be replayed, so these tests are foundational rather than incidental.
"""

from __future__ import annotations

import math
from datetime import UTC, datetime, timedelta, timezone
from decimal import Decimal
from enum import Enum

import pytest
from hypothesis import given
from hypothesis import strategies as st

from tb.core.canonical import (
    GENESIS_HASH,
    canonical_bytes,
    canonical_json,
    hash_payload,
    sha256_hex,
)


def test_genesis_hash_is_64_zeros() -> None:
    assert GENESIS_HASH == "0" * 64
    assert len(GENESIS_HASH) == len(sha256_hex(b""))


def test_key_order_does_not_change_the_hash() -> None:
    assert hash_payload({"a": 1, "b": 2}) == hash_payload({"b": 2, "a": 1})


def test_nesting_is_canonicalised_at_every_depth() -> None:
    left = {"outer": {"z": [{"b": 1, "a": 2}]}}
    right = {"outer": {"z": [{"a": 2, "b": 1}]}}
    assert hash_payload(left) == hash_payload(right)


def test_decimal_serialises_exactly_and_not_through_float() -> None:
    # 0.1 as a float is 0.1000000000000000055511151231257827. A monetary limit
    # must never inherit that, so Decimals go out as their exact string form.
    assert canonical_json({"amount": Decimal("0.10")}) == '{"amount":"0.10"}'
    assert canonical_json({"amount": Decimal("500.00")}) == '{"amount":"500.00"}'
    # Trailing zeros are significant in a Decimal and are preserved, so 500.00
    # and 500.0 are distinguishable in the audit trail.
    assert hash_payload(Decimal("500.00")) != hash_payload(Decimal("500.0"))


def test_int_and_float_of_equal_value_hash_differently() -> None:
    # They are different values, and a quantity of 1 share is not the same fact
    # as a quantity of 1.0 shares arrived at by division.
    assert hash_payload({"q": 1}) != hash_payload({"q": 1.0})


@pytest.mark.parametrize("value", [float("nan"), float("inf"), float("-inf")])
def test_non_finite_floats_are_refused(value: float) -> None:
    with pytest.raises(ValueError, match="non-finite"):
        canonical_json({"feature": value})


def test_non_finite_decimal_is_refused() -> None:
    with pytest.raises(ValueError, match="non-finite"):
        canonical_json({"amount": Decimal("NaN")})


def test_naive_datetime_is_refused() -> None:
    with pytest.raises(ValueError, match="naive"):
        canonical_json({"at": datetime(2026, 1, 1, 12, 0, 0)})  # noqa: DTZ001


def test_equivalent_instants_in_different_zones_hash_identically() -> None:
    utc = datetime(2026, 1, 1, 12, 0, tzinfo=UTC)
    plus_two = datetime(2026, 1, 1, 14, 0, tzinfo=timezone(timedelta(hours=2)))
    assert hash_payload({"at": utc}) == hash_payload({"at": plus_two})


def test_sets_are_refused_because_their_order_is_not_defined() -> None:
    with pytest.raises(TypeError, match="iteration order"):
        canonical_json({"symbols": {"AAPL", "MSFT"}})


def test_unknown_types_are_refused_rather_than_stringified() -> None:
    class Opaque:
        def __str__(self) -> str:
            return "same"

    with pytest.raises(TypeError, match="no canonical form"):
        canonical_json({"thing": Opaque()})


def test_non_string_keys_are_refused() -> None:
    with pytest.raises(TypeError, match="string keys"):
        canonical_json({1: "one"})


def test_bool_stays_a_boolean_not_an_int() -> None:
    # bool subclasses int, so a careless isinstance order turns True into 1.
    assert canonical_json({"flag": True}) == '{"flag":true}'
    assert hash_payload({"flag": True}) != hash_payload({"flag": 1})


def test_enum_serialises_by_value() -> None:
    class Side(Enum):
        BUY = "buy"

    assert canonical_json({"side": Side.BUY}) == '{"side":"buy"}'


def test_tuples_and_lists_are_the_same_sequence() -> None:
    assert hash_payload({"xs": (1, 2)}) == hash_payload({"xs": [1, 2]})


def test_list_order_is_significant() -> None:
    assert hash_payload({"xs": [1, 2]}) != hash_payload({"xs": [2, 1]})


def test_encoding_is_utf8_and_not_escaped() -> None:
    payload = {"note": "café ☕"}
    assert canonical_json(payload) == '{"note":"café ☕"}'
    assert canonical_bytes(payload) == '{"note":"café ☕"}'.encode()


def test_no_field_boundary_ambiguity() -> None:
    """Named keys must make each field's extent unambiguous.

    A hash built by concatenating values would give `"ab" + "c"` and
    `"a" + "bc"` identical input, letting an attacker move content across field
    boundaries without changing the digest. This is the reason the chain hash is
    computed over a keyed mapping rather than a joined string.
    """
    assert hash_payload({"x": "ab", "y": "c"}) != hash_payload({"x": "a", "y": "bc"})


@given(
    st.dictionaries(
        st.text(min_size=1, max_size=8),
        st.one_of(
            st.integers(),
            st.booleans(),
            st.text(max_size=20),
            st.none(),
            st.floats(allow_nan=False, allow_infinity=False),
        ),
        max_size=6,
    )
)
def test_serialisation_is_stable_across_repeated_calls(payload: dict[str, object]) -> None:
    assert canonical_json(payload) == canonical_json(payload)
    assert hash_payload(payload) == hash_payload(payload)


@given(st.floats(allow_nan=False, allow_infinity=False))
def test_finite_floats_round_trip_through_canonical_json(value: float) -> None:
    import json

    restored = json.loads(canonical_json({"v": value}))["v"]
    # Round-trip fidelity matters: a feature value that changes when hashed
    # would make the snapshot hash disagree with the value actually used.
    assert restored == value or (math.isclose(restored, value, rel_tol=0, abs_tol=0))
