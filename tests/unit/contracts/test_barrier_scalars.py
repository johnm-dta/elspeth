"""Unit tests for BarrierScalars and sub-types.

Covers:
 - Round-trip serialization (to_dict / from_dict identity)
 - __post_init__ validation (negative offsets, type guards)
 - Empty-state semantics (has_state == False, minimal dict)
 - Hostile coalesce key characters
 - Unknown-key rejection and wrong-version rejection in from_dict
"""

from __future__ import annotations

from types import MappingProxyType
from typing import Any

import pytest

from elspeth.contracts.barrier_scalars import (
    AggregationNodeScalars,
    BarrierScalars,
    CoalescePendingScalars,
)
from elspeth.contracts.errors import AuditIntegrityError

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _agg(
    count_fire_offset: float | None = 1.5,
    condition_fire_offset: float | None = None,
) -> AggregationNodeScalars:
    return AggregationNodeScalars(
        count_fire_offset=count_fire_offset,
        condition_fire_offset=condition_fire_offset,
    )


def _coalesce(lost_branches: dict[str, str] | None = None) -> CoalescePendingScalars:
    return CoalescePendingScalars(lost_branches=lost_branches or {})


# ---------------------------------------------------------------------------
# Required tests (from plan)
# ---------------------------------------------------------------------------


def test_round_trip() -> None:
    """Full round-trip: construct → to_dict → from_dict must reproduce the original."""
    s = BarrierScalars(
        aggregation={"agg-1": AggregationNodeScalars(count_fire_offset=1.5, condition_fire_offset=None)},
        coalesce={("merge", "row-7"): CoalescePendingScalars(lost_branches={"b2": "branch lost: transform failed"})},
    )
    assert BarrierScalars.from_dict(s.to_dict()) == s


def test_empty_state_is_falsy_and_serializes_minimal() -> None:
    """Empty BarrierScalars: has_state is False; round-trips cleanly."""
    empty = BarrierScalars(aggregation={}, coalesce={})
    assert not empty.has_state
    assert BarrierScalars.from_dict(empty.to_dict()) == empty


def test_coalesce_key_with_hostile_characters_round_trips() -> None:
    """Coalesce names and row_ids with special chars must survive to_dict/from_dict.

    coalesce names have no charset constraint (CoalesceSettings.name in
    core/config.py) and row_id is operator-influenced — keys must survive
    arbitrary strings.
    """
    s = BarrierScalars(aggregation={}, coalesce={("we::ird", 'row"7'): CoalescePendingScalars(lost_branches={})})
    assert BarrierScalars.from_dict(s.to_dict()) == s


def test_negative_offset_rejected() -> None:
    """Negative count_fire_offset must raise ValueError."""
    with pytest.raises(ValueError):
        AggregationNodeScalars(count_fire_offset=-0.1, condition_fire_offset=None)


# ---------------------------------------------------------------------------
# AggregationNodeScalars validation
# ---------------------------------------------------------------------------


class TestAggregationNodeScalarsValidation:
    """__post_init__ guards on AggregationNodeScalars."""

    def test_both_none_is_valid(self) -> None:
        """Both offsets may be None (node not yet triggered)."""
        s = AggregationNodeScalars(count_fire_offset=None, condition_fire_offset=None)
        assert s.count_fire_offset is None
        assert s.condition_fire_offset is None

    def test_both_set_is_valid(self) -> None:
        """Both offsets may be simultaneously set."""
        s = AggregationNodeScalars(count_fire_offset=0.0, condition_fire_offset=2.5)
        assert s.count_fire_offset == 0.0
        assert s.condition_fire_offset == 2.5

    def test_zero_offset_is_valid(self) -> None:
        """Exactly zero is a valid (non-negative) offset."""
        s = AggregationNodeScalars(count_fire_offset=0.0, condition_fire_offset=0.0)
        assert s.count_fire_offset == 0.0

    def test_negative_count_fire_offset_rejected(self) -> None:
        with pytest.raises(ValueError, match="count_fire_offset"):
            AggregationNodeScalars(count_fire_offset=-0.001, condition_fire_offset=None)

    def test_negative_condition_fire_offset_rejected(self) -> None:
        with pytest.raises(ValueError, match="condition_fire_offset"):
            AggregationNodeScalars(count_fire_offset=None, condition_fire_offset=-1.0)

    def test_nan_count_fire_offset_rejected(self) -> None:
        with pytest.raises(ValueError, match="count_fire_offset"):
            AggregationNodeScalars(count_fire_offset=float("nan"), condition_fire_offset=None)

    def test_inf_condition_fire_offset_rejected(self) -> None:
        with pytest.raises(ValueError, match="condition_fire_offset"):
            AggregationNodeScalars(count_fire_offset=None, condition_fire_offset=float("inf"))

    def test_round_trip_both_set(self) -> None:
        s = AggregationNodeScalars(count_fire_offset=3.14, condition_fire_offset=0.5)
        d = s.to_dict()
        assert AggregationNodeScalars.from_dict(d) == s

    def test_round_trip_both_none(self) -> None:
        s = AggregationNodeScalars(count_fire_offset=None, condition_fire_offset=None)
        assert AggregationNodeScalars.from_dict(s.to_dict()) == s

    def test_from_dict_rejects_unknown_key(self) -> None:
        d = _agg().to_dict()
        d["_extra"] = "should_fail"
        with pytest.raises(AuditIntegrityError, match="unknown"):
            AggregationNodeScalars.from_dict(d)

    def test_from_dict_rejects_wrong_version(self) -> None:
        d = _agg().to_dict()
        d["_version"] = "9.9"
        with pytest.raises(AuditIntegrityError, match="version"):
            AggregationNodeScalars.from_dict(d)

    def test_from_dict_rejects_missing_required_key(self) -> None:
        d = _agg().to_dict()
        del d["count_fire_offset"]
        with pytest.raises(AuditIntegrityError, match="count_fire_offset"):
            AggregationNodeScalars.from_dict(d)

    @pytest.mark.parametrize("field", ["count_fire_offset", "condition_fire_offset"])
    def test_from_dict_rejects_string_offset(self, field: str) -> None:
        """A string offset arriving via from_dict is corruption, not a ValueError."""
        d = _agg().to_dict()
        d[field] = "1.5"
        with pytest.raises(AuditIntegrityError, match=field):
            AggregationNodeScalars.from_dict(d)

    @pytest.mark.parametrize("field", ["count_fire_offset", "condition_fire_offset"])
    def test_from_dict_rejects_bool_offset(self, field: str) -> None:
        """bool is an int subclass and math.isfinite(True) passes — must be excluded."""
        d = _agg().to_dict()
        d[field] = True
        with pytest.raises(AuditIntegrityError, match=field):
            AggregationNodeScalars.from_dict(d)

    @pytest.mark.parametrize("field", ["count_fire_offset", "condition_fire_offset"])
    def test_from_dict_rejects_negative_offset(self, field: str) -> None:
        """A negative offset arriving via from_dict raises AuditIntegrityError (not ValueError)."""
        d = _agg().to_dict()
        d[field] = -0.5
        with pytest.raises(AuditIntegrityError, match=field):
            AggregationNodeScalars.from_dict(d)

    def test_from_dict_rejects_nan_offset(self) -> None:
        d = _agg().to_dict()
        d["count_fire_offset"] = float("nan")
        with pytest.raises(AuditIntegrityError, match="count_fire_offset"):
            AggregationNodeScalars.from_dict(d)


# ---------------------------------------------------------------------------
# CoalescePendingScalars validation
# ---------------------------------------------------------------------------


class TestCoalescePendingScalarsValidation:
    """__post_init__ guards on CoalescePendingScalars."""

    def test_empty_lost_branches_is_valid(self) -> None:
        s = CoalescePendingScalars(lost_branches={})
        assert s.lost_branches == {}

    def test_nonempty_lost_branches_is_valid(self) -> None:
        s = CoalescePendingScalars(lost_branches={"arm_a": "timeout"})
        assert "arm_a" in s.lost_branches

    def test_round_trip_empty(self) -> None:
        s = CoalescePendingScalars(lost_branches={})
        assert CoalescePendingScalars.from_dict(s.to_dict()) == s

    def test_round_trip_nonempty(self) -> None:
        s = CoalescePendingScalars(lost_branches={"b1": "reason 1", "b2": "reason 2"})
        assert CoalescePendingScalars.from_dict(s.to_dict()) == s

    def test_from_dict_rejects_unknown_key(self) -> None:
        d = _coalesce().to_dict()
        d["unexpected"] = "fail"
        with pytest.raises(AuditIntegrityError, match="unknown"):
            CoalescePendingScalars.from_dict(d)

    def test_from_dict_rejects_wrong_version(self) -> None:
        d = _coalesce().to_dict()
        d["_version"] = "0.0"
        with pytest.raises(AuditIntegrityError, match="version"):
            CoalescePendingScalars.from_dict(d)

    def test_from_dict_rejects_missing_lost_branches(self) -> None:
        d = _coalesce().to_dict()
        del d["lost_branches"]
        with pytest.raises(AuditIntegrityError, match="lost_branches"):
            CoalescePendingScalars.from_dict(d)

    def test_lost_branches_is_frozen_after_construction(self) -> None:
        """lost_branches mapping must be read-only after construction."""
        s = CoalescePendingScalars(lost_branches={"k": "v"})
        assert isinstance(s.lost_branches, MappingProxyType)

    def test_from_dict_rejects_int_lost_branches_value(self) -> None:
        d: dict[str, Any] = {"_version": "1.0", "lost_branches": {"b1": 42}}
        with pytest.raises(AuditIntegrityError, match="lost_branches"):
            CoalescePendingScalars.from_dict(d)

    def test_from_dict_rejects_none_lost_branches_value(self) -> None:
        d: dict[str, Any] = {"_version": "1.0", "lost_branches": {"b1": None}}
        with pytest.raises(AuditIntegrityError, match="lost_branches"):
            CoalescePendingScalars.from_dict(d)

    def test_from_dict_rejects_nested_dict_lost_branches_value(self) -> None:
        """A nested dict would silently re-serialize and propagate — must crash."""
        d: dict[str, Any] = {"_version": "1.0", "lost_branches": {"b1": {"reason": "deep"}}}
        with pytest.raises(AuditIntegrityError, match="lost_branches"):
            CoalescePendingScalars.from_dict(d)

    def test_from_dict_rejects_non_str_lost_branches_key(self) -> None:
        d: dict[str, Any] = {"_version": "1.0", "lost_branches": {7: "reason"}}
        with pytest.raises(AuditIntegrityError, match="lost_branches"):
            CoalescePendingScalars.from_dict(d)


# ---------------------------------------------------------------------------
# BarrierScalars validation and serialization
# ---------------------------------------------------------------------------


class TestBarrierScalarsRoundTrips:
    """to_dict / from_dict identity for all interesting shapes."""

    def test_aggregation_only(self) -> None:
        s = BarrierScalars(
            aggregation={"n1": _agg(1.0, 2.0), "n2": _agg(None, None)},
            coalesce={},
        )
        assert BarrierScalars.from_dict(s.to_dict()) == s
        assert s.has_state

    def test_coalesce_only(self) -> None:
        s = BarrierScalars(
            aggregation={},
            coalesce={("join", "row-x"): _coalesce({"branch": "lost"})},
        )
        assert BarrierScalars.from_dict(s.to_dict()) == s
        assert s.has_state

    def test_both_populated(self) -> None:
        s = BarrierScalars(
            aggregation={"agg-1": _agg(0.5, None)},
            coalesce={("c", "r"): _coalesce()},
        )
        assert BarrierScalars.from_dict(s.to_dict()) == s
        assert s.has_state

    def test_multiple_coalesce_keys(self) -> None:
        s = BarrierScalars(
            aggregation={},
            coalesce={
                ("c1", "r1"): _coalesce({"b": "reason"}),
                ("c2", "r2"): _coalesce(),
            },
        )
        assert BarrierScalars.from_dict(s.to_dict()) == s

    def test_has_state_false_when_empty(self) -> None:
        assert not BarrierScalars(aggregation={}, coalesce={}).has_state

    def test_has_state_true_when_aggregation_nonempty(self) -> None:
        s = BarrierScalars(aggregation={"x": _agg()}, coalesce={})
        assert s.has_state

    def test_has_state_true_when_coalesce_nonempty(self) -> None:
        s = BarrierScalars(aggregation={}, coalesce={("a", "b"): _coalesce()})
        assert s.has_state

    def test_container_fields_are_frozen_after_construction(self) -> None:
        s = BarrierScalars(
            aggregation={"agg": _agg()},
            coalesce={("merge", "row"): _coalesce({"branch": "lost"})},
        )

        assert isinstance(s.aggregation, MappingProxyType)
        assert isinstance(s.coalesce, MappingProxyType)


class TestBarrierScalarsFromDictErrors:
    """from_dict error path coverage."""

    def test_rejects_missing_version(self) -> None:
        d = BarrierScalars(aggregation={}, coalesce={}).to_dict()
        del d["_version"]
        with pytest.raises(AuditIntegrityError, match="_version"):
            BarrierScalars.from_dict(d)

    def test_rejects_wrong_version(self) -> None:
        d = BarrierScalars(aggregation={}, coalesce={}).to_dict()
        d["_version"] = "99.0"
        with pytest.raises(AuditIntegrityError, match="version"):
            BarrierScalars.from_dict(d)

    def test_rejects_unknown_top_level_key(self) -> None:
        d = BarrierScalars(aggregation={}, coalesce={}).to_dict()
        d["_mystery"] = "oops"
        with pytest.raises(AuditIntegrityError, match="unknown"):
            BarrierScalars.from_dict(d)

    def test_rejects_non_dict_top_level(self) -> None:
        with pytest.raises(AuditIntegrityError):
            BarrierScalars.from_dict("not-a-dict")  # type: ignore[arg-type]

    def test_rejects_malformed_coalesce_entry_not_list_of_two(self) -> None:
        d: dict[str, Any] = {"_version": "1.0", "aggregation": {}, "coalesce": [[1, 2, 3], {}]}
        with pytest.raises(AuditIntegrityError):
            BarrierScalars.from_dict(d)

    def test_coalesce_entry_key_must_be_two_strings(self) -> None:
        d: dict[str, Any] = {
            "_version": "1.0",
            "aggregation": {},
            "coalesce": [[[1, "r"], {"_version": "1.0", "lost_branches": {}}]],
        }
        with pytest.raises(AuditIntegrityError):
            BarrierScalars.from_dict(d)

    def test_rejects_duplicate_coalesce_key(self) -> None:
        """to_dict can never emit a duplicate key — a duplicate is corruption, not last-wins."""
        d: dict[str, Any] = {
            "_version": "1.0",
            "aggregation": {},
            "coalesce": [
                [["merge", "row-1"], {"_version": "1.0", "lost_branches": {}}],
                [["merge", "row-1"], {"_version": "1.0", "lost_branches": {"b": "lost"}}],
            ],
        }
        with pytest.raises(AuditIntegrityError, match="duplicate"):
            BarrierScalars.from_dict(d)
