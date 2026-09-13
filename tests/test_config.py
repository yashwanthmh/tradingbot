"""The control layer.

Two properties carry the weight: an invalid or incoherent limits file must stop
the bot from starting, and a limits file that changes under a running process
must halt it. Anything softer than that turns "the agent cannot modify its own
constraints" into a comment rather than a control.
"""

from __future__ import annotations

from collections.abc import Callable
from decimal import Decimal
from pathlib import Path
from typing import Any

import pytest
import yaml

from tb.config.hard_limits import HardLimits
from tb.config.loader import load_hard_limits
from tb.core.errors import ConfigDriftError, ConfigError
from tests.conftest import REFERENCE_LIMITS, running_as_root


def test_the_shipped_limits_file_validates() -> None:
    """The file we actually deploy must load. Not a fixture — the real one."""
    pinned = load_hard_limits(REFERENCE_LIMITS)
    assert pinned.limits.currency == "GBP"
    assert pinned.limits.capital.absolute_ceiling_ccy == Decimal("500.00")


def test_money_is_decimal_not_float() -> None:
    pinned = load_hard_limits(REFERENCE_LIMITS)
    amount = pinned.limits.capital.floor_notional_ccy
    assert isinstance(amount, Decimal)
    assert str(amount) == "15.00"


def test_missing_file_is_fatal() -> None:
    with pytest.raises(ConfigError, match="will not start without them"):
        load_hard_limits(Path("/nonexistent/hard_limits.yaml"))


def test_malformed_yaml_is_fatal(tmp_path: Path) -> None:
    bad = tmp_path / "bad.yaml"
    bad.write_text("capital: [unclosed", encoding="utf-8")
    with pytest.raises(ConfigError, match="not valid YAML"):
        load_hard_limits(bad)


def test_non_mapping_is_fatal(tmp_path: Path) -> None:
    bad = tmp_path / "bad.yaml"
    bad.write_text("- a list, not a mapping", encoding="utf-8")
    with pytest.raises(ConfigError, match="must be a YAML mapping"):
        load_hard_limits(bad)


def test_unknown_schema_version_is_refused(write_limits: Callable[[dict[str, Any]], Path]) -> None:
    path = write_limits({"schema_version": 99})
    with pytest.raises(ConfigError, match="Refusing to guess"):
        load_hard_limits(path)


def test_a_typo_in_a_limit_key_is_an_error_not_a_silent_default(tmp_path: Path) -> None:
    """The asymmetry that matters.

    Unknown fields from a beta broker API are ignored, because taking the system
    down over a new field the broker added is worse than not reading it. Unknown
    fields in *our own* limits file are fatal, because `max_orders_per_day_`
    means a cap someone believes is in force is not.
    """
    base = yaml.safe_load(REFERENCE_LIMITS.read_text(encoding="utf-8"))
    base["execution"]["max_orders_per_day_typo"] = 5
    path = tmp_path / "typo.yaml"
    path.write_text(yaml.safe_dump(base), encoding="utf-8")

    with pytest.raises(ConfigError, match="max_orders_per_day_typo"):
        load_hard_limits(path)


def test_limits_are_frozen() -> None:
    pinned = load_hard_limits(REFERENCE_LIMITS)
    with pytest.raises(Exception, match=r"frozen|immutable"):
        pinned.limits.capital.absolute_ceiling_ccy = Decimal("100000")


# --------------------------------------------------------------------------
# Coherence checks — limits that validate individually but contradict together
# --------------------------------------------------------------------------


def test_position_size_larger_than_total_deployment_is_refused(
    write_limits: Callable[[dict[str, Any]], Path],
) -> None:
    path = write_limits({"capital": {"per_position_pct": 20.0, "max_deployed_pct": 10.0}})
    with pytest.raises(ConfigError, match="could breach the total cap"):
        load_hard_limits(path)


def test_floor_order_above_the_absolute_ceiling_is_refused(
    write_limits: Callable[[dict[str, Any]], Path],
) -> None:
    path = write_limits({"capital": {"floor_notional_ccy": 1000.0, "absolute_ceiling_ccy": 500.0}})
    with pytest.raises(ConfigError, match="nothing could trade"):
        load_hard_limits(path)


def test_loss_breakers_must_be_ordered(write_limits: Callable[[dict[str, Any]], Path]) -> None:
    path = write_limits({"loss": {"daily_halt_pct": 9.0, "rolling_5d_halt_pct": 4.0}})
    with pytest.raises(ConfigError, match="could never fire first"):
        load_hard_limits(path)


def test_flatten_breaker_must_sit_outside_the_stop_opening_breakers(
    write_limits: Callable[[dict[str, Any]], Path],
) -> None:
    path = write_limits({"loss": {"rolling_5d_halt_pct": 8.0, "max_drawdown_flatten_pct": 6.0}})
    with pytest.raises(ConfigError, match="must sit outside"):
        load_hard_limits(path)


def test_strategy_budget_cannot_exceed_its_lineage_budget(
    write_limits: Callable[[dict[str, Any]], Path],
) -> None:
    path = write_limits(
        {"loss": {"per_strategy_budget_ccy": 200.0, "per_lineage_budget_ccy": 100.0}}
    )
    with pytest.raises(ConfigError, match="outlive the budget of the lineage"):
        load_hard_limits(path)


def test_unprotected_gap_risk_must_fit_inside_the_daily_budget(
    write_limits: Callable[[dict[str, Any]], Path],
) -> None:
    """The coupling that only shows up if you think about crash recovery.

    Trading 212 has no bracket orders, so an entry fill always precedes its
    protective stop. If the process dies in that window the position is naked.
    A 5% position taking the assumed 15% gap loses 0.75% of equity, which is
    fine; a 30% position loses 4.5%, which blows through a 2% daily limit before
    any breaker can fire. The config refuses to express that combination.
    """
    path = write_limits(
        {
            "capital": {"per_position_pct": 30.0, "max_deployed_pct": 40.0},
            "execution": {"unprotected_gap_pct_assumption": 15.0},
            "loss": {"daily_halt_pct": 2.0},
        }
    )
    with pytest.raises(ConfigError, match="Trading 212 has no bracket orders"):
        load_hard_limits(path)


def test_pending_order_headroom_against_the_broker_ceiling(
    write_limits: Callable[[dict[str, Any]], Path],
) -> None:
    """Trading 212 caps pending orders at 50 per ticker.

    Exhausting that ceiling does not merely block a new entry — it makes a
    protective stop rejected, converting a cap breach into an unhedged position.
    """
    path = write_limits(
        {"execution": {"max_orders_per_symbol_per_day": 40, "max_orders_per_day": 80}}
    )
    with pytest.raises(ConfigError, match="protective stops to be rejected"):
        load_hard_limits(path)


def test_worst_case_unprotected_loss_is_computed_consistently() -> None:
    limits = load_hard_limits(REFERENCE_LIMITS).limits
    expected = (
        limits.capital.per_position_pct * limits.execution.unprotected_gap_pct_assumption / 100.0
    )
    assert limits.worst_case_unprotected_loss_pct == pytest.approx(expected)
    assert limits.worst_case_unprotected_loss_pct <= limits.loss.daily_halt_pct


# --------------------------------------------------------------------------
# Hash pinning and drift
# --------------------------------------------------------------------------


def test_content_hash_changes_when_the_file_changes(limits_file: Path) -> None:
    before = load_hard_limits(limits_file)
    limits_file.write_text(
        limits_file.read_text(encoding="utf-8").replace(
            "absolute_ceiling_ccy: 500.00", "absolute_ceiling_ccy: 50000.00"
        ),
        encoding="utf-8",
    )
    after = load_hard_limits(limits_file)
    assert before.content_sha256 != after.content_sha256
    assert before.canonical_sha256 != after.canonical_sha256


def test_a_comment_change_moves_the_content_hash_but_not_the_canonical_hash(
    limits_file: Path,
) -> None:
    """Why two hashes exist.

    The content hash is the tamper pin and moves on any byte. The canonical hash
    answers "did the effective limits change", so the audit trail can tell a
    reformat apart from a loosened cap.
    """
    before = load_hard_limits(limits_file)
    limits_file.write_text(
        limits_file.read_text(encoding="utf-8") + "\n# a trailing comment\n",
        encoding="utf-8",
    )
    after = load_hard_limits(limits_file)
    assert before.content_sha256 != after.content_sha256
    assert before.canonical_sha256 == after.canonical_sha256


def test_verify_unchanged_passes_when_untouched(limits_file: Path) -> None:
    load_hard_limits(limits_file).verify_unchanged()


def test_verify_unchanged_detects_an_edit_under_a_running_process(limits_file: Path) -> None:
    pinned = load_hard_limits(limits_file)
    limits_file.write_text(
        limits_file.read_text(encoding="utf-8").replace(
            "max_orders_per_day: 40", "max_orders_per_day: 4000"
        ),
        encoding="utf-8",
    )
    with pytest.raises(ConfigDriftError, match="changed while running"):
        pinned.verify_unchanged()


def test_verify_unchanged_treats_an_unreadable_file_as_drift(limits_file: Path) -> None:
    """Unreadable is as disqualifying as changed.

    Either way the process can no longer demonstrate it is operating under the
    caps that were reviewed.
    """
    pinned = load_hard_limits(limits_file)
    limits_file.unlink()
    with pytest.raises(ConfigDriftError, match="unreadable"):
        pinned.verify_unchanged()


def test_audit_record_carries_the_full_values(limits_file: Path) -> None:
    """The ledger stores the values, not just the hash.

    Months later, "what were the caps when this trade happened" has to be
    answerable from the ledger alone — not from the git history of a file that
    may have been moved, rewritten, or lost.
    """
    record = load_hard_limits(limits_file).audit_record()
    assert record["kind"] == "hard_limits"
    assert record["values"]["capital"]["absolute_ceiling_ccy"] == "500.00"
    assert record["values"]["loss"]["daily_halt_pct"] == 2.0
    assert "immutability_enforced" in record


@running_as_root
def test_immutability_is_reported_honestly_when_unenforced(limits_file: Path) -> None:
    pinned = load_hard_limits(limits_file)
    assert pinned.immutability.writable_by_process
    assert not pinned.immutability.enforced
    assert "read-only" in pinned.immutability.explanation


def test_root_is_reported_as_unenforced(limits_file: Path) -> None:
    """Running as root means mode bits cannot constrain us, and we say so.

    `os.access` returns True for root regardless of permissions, so a naive
    check would claim the limits were immutable while root could rewrite them.
    """
    import os

    pinned = load_hard_limits(limits_file)
    if os.getuid() == 0:
        assert pinned.immutability.running_as_root
        assert not pinned.immutability.enforced
        assert "root" in pinned.immutability.explanation


def test_hard_limits_model_rejects_out_of_range_percentages() -> None:
    payload = yaml.safe_load(REFERENCE_LIMITS.read_text(encoding="utf-8"))
    payload["capital"]["per_position_pct"] = 0.0
    with pytest.raises(Exception, match="greater than 0"):
        HardLimits.model_validate(payload)
