# tests/unit/contracts/test_coalesce_checkpoint.py
"""Unit tests for CoalesceTokenCheckpoint and CoalescePendingCheckpoint validation.

Covers __post_init__ invariant enforcement and from_dict error paths.
"""

from __future__ import annotations

from collections import OrderedDict
from types import MappingProxyType
from typing import Any

import pytest

from elspeth.contracts.coalesce_checkpoint import (
    CoalesceCheckpointState,
    CoalescePendingCheckpoint,
    CoalesceTokenCheckpoint,
)
from elspeth.contracts.errors import AuditIntegrityError


def _valid_token_kwargs() -> dict[str, Any]:
    """Minimal valid kwargs for CoalesceTokenCheckpoint."""
    return {
        "token_id": "tok-1",
        "row_id": "row-1",
        "branch_name": "path_a",
        "fork_group_id": None,
        "join_group_id": None,
        "expand_group_id": None,
        "row_data": {"field": "value"},
        "contract": {"schema_version": "1"},
        "state_id": "state-1",
        "arrival_offset_seconds": 0.5,
    }


def _valid_token() -> CoalesceTokenCheckpoint:
    return CoalesceTokenCheckpoint(**_valid_token_kwargs())


class TestCoalesceTokenCheckpointPostInit:
    """__post_init__ validation for CoalesceTokenCheckpoint."""

    def test_valid_construction(self) -> None:
        """Valid kwargs produce a valid object."""
        token = _valid_token()
        assert token.token_id == "tok-1"

    @pytest.mark.parametrize("field", ["token_id", "row_id", "branch_name", "state_id"])
    def test_rejects_empty_string(self, field: str) -> None:
        """Required string fields must be non-empty."""
        kwargs = _valid_token_kwargs()
        kwargs[field] = ""
        with pytest.raises(ValueError, match=field):
            CoalesceTokenCheckpoint(**kwargs)

    @pytest.mark.parametrize("field", ["token_id", "row_id", "branch_name", "state_id"])
    def test_rejects_non_string(self, field: str) -> None:
        """Required string fields reject non-string types (e.g. corrupted JSON with int)."""
        kwargs = _valid_token_kwargs()
        kwargs[field] = 42
        with pytest.raises(TypeError, match=field):
            CoalesceTokenCheckpoint(**kwargs)

    def test_rejects_negative_arrival_offset(self) -> None:
        """arrival_offset_seconds must be non-negative."""
        kwargs = _valid_token_kwargs()
        kwargs["arrival_offset_seconds"] = -1.0
        with pytest.raises(ValueError, match="arrival_offset_seconds"):
            CoalesceTokenCheckpoint(**kwargs)

    def test_rejects_nan_arrival_offset(self) -> None:
        """NaN must not bypass the non-negative check."""
        kwargs = _valid_token_kwargs()
        kwargs["arrival_offset_seconds"] = float("nan")
        with pytest.raises(ValueError, match="arrival_offset_seconds"):
            CoalesceTokenCheckpoint(**kwargs)

    def test_rejects_inf_arrival_offset(self) -> None:
        """Infinity must not bypass the non-negative check."""
        kwargs = _valid_token_kwargs()
        kwargs["arrival_offset_seconds"] = float("inf")
        with pytest.raises(ValueError, match="arrival_offset_seconds"):
            CoalesceTokenCheckpoint(**kwargs)

    def test_accepts_ordered_dict_mapping(self) -> None:
        """OrderedDict is a Mapping subtype — must be accepted (not just dict)."""
        kwargs = _valid_token_kwargs()
        kwargs["row_data"] = OrderedDict({"x": 1})
        kwargs["contract"] = OrderedDict({"mode": "observed"})
        token = CoalesceTokenCheckpoint(**kwargs)
        assert isinstance(token.row_data, MappingProxyType)
        assert token.row_data["x"] == 1

    def test_rejects_non_dict_row_data(self) -> None:
        """row_data must be a Mapping."""
        kwargs = _valid_token_kwargs()
        kwargs["row_data"] = "not a dict"
        with pytest.raises(TypeError, match="row_data must be a Mapping"):
            CoalesceTokenCheckpoint(**kwargs)

    def test_rejects_non_dict_contract(self) -> None:
        """contract must be a Mapping."""
        kwargs = _valid_token_kwargs()
        kwargs["contract"] = ["not", "a", "dict"]
        with pytest.raises(TypeError, match="contract must be a Mapping"):
            CoalesceTokenCheckpoint(**kwargs)


class TestCoalescePendingCheckpointPostInit:
    """__post_init__ validation for CoalescePendingCheckpoint."""

    def test_valid_construction(self) -> None:
        """Valid kwargs produce a valid object."""
        pending = CoalescePendingCheckpoint(
            coalesce_name="merge_1",
            row_id="row-1",
            elapsed_age_seconds=1.5,
            branches={"path_a": _valid_token()},
            lost_branches={},
        )
        assert pending.coalesce_name == "merge_1"

    @pytest.mark.parametrize("field", ["coalesce_name", "row_id"])
    def test_rejects_empty_string(self, field: str) -> None:
        """Required string fields must be non-empty."""
        kwargs: dict[str, Any] = {
            "coalesce_name": "merge_1",
            "row_id": "row-1",
            "elapsed_age_seconds": 1.5,
            "branches": {"path_a": _valid_token()},
            "lost_branches": {},
        }
        kwargs[field] = ""
        with pytest.raises(ValueError, match=field):
            CoalescePendingCheckpoint(**kwargs)

    def test_rejects_negative_elapsed_age(self) -> None:
        """elapsed_age_seconds must be non-negative."""
        with pytest.raises(ValueError, match="elapsed_age_seconds"):
            CoalescePendingCheckpoint(
                coalesce_name="merge_1",
                row_id="row-1",
                elapsed_age_seconds=-0.1,
                branches={},
                lost_branches={},
            )

    def test_rejects_nan_elapsed_age(self) -> None:
        """NaN must not bypass the non-negative check."""
        with pytest.raises(ValueError, match="elapsed_age_seconds"):
            CoalescePendingCheckpoint(
                coalesce_name="merge_1",
                row_id="row-1",
                elapsed_age_seconds=float("nan"),
                branches={},
                lost_branches={},
            )

    def test_rejects_inf_elapsed_age(self) -> None:
        """Infinity must not bypass the non-negative check."""
        with pytest.raises(ValueError, match="elapsed_age_seconds"):
            CoalescePendingCheckpoint(
                coalesce_name="merge_1",
                row_id="row-1",
                elapsed_age_seconds=float("inf"),
                branches={},
                lost_branches={},
            )

    def test_rejects_overlapping_branches_and_lost_branches(self) -> None:
        """branches and lost_branches must be disjoint."""
        token = _valid_token()
        with pytest.raises(ValueError, match="overlap"):
            CoalescePendingCheckpoint(
                coalesce_name="merge_1",
                row_id="row-1",
                elapsed_age_seconds=1.0,
                branches={"path_a": token},
                lost_branches={"path_a": "timed_out"},
            )

    def test_rejects_branch_key_mismatch(self) -> None:
        """Branch key must match embedded token's branch_name.

        Regression: elspeth-1d5cc9c4a2 — dual-encoded branch identity must agree.
        """
        token = _valid_token()  # branch_name="path_a"
        with pytest.raises(AuditIntegrityError, match="does not match"):
            CoalescePendingCheckpoint(
                coalesce_name="merge_1",
                row_id="row-1",
                elapsed_age_seconds=1.0,
                branches={"WRONG_KEY": token},  # key != token.branch_name
                lost_branches={},
            )

    def test_rejects_empty_lost_branches_value(self) -> None:
        """lost_branches values must be non-empty strings."""
        valid = _valid_token()
        with pytest.raises(ValueError, match=r"lost_branches.*non-empty"):
            CoalescePendingCheckpoint(
                coalesce_name="merge_1",
                row_id="row-1",
                elapsed_age_seconds=1.0,
                branches={"path_a": valid},
                lost_branches={"path_b": ""},
            )


class TestCoalesceTokenCheckpointFromDict:
    """from_dict validation for CoalesceTokenCheckpoint."""

    def test_rejects_missing_fields(self) -> None:
        """from_dict raises AuditIntegrityError on missing required fields."""
        with pytest.raises(AuditIntegrityError, match="missing required fields"):
            CoalesceTokenCheckpoint.from_dict({"token_id": "tok-1"})

    def test_roundtrip(self) -> None:
        """to_dict → from_dict roundtrip preserves data."""
        original = _valid_token()
        restored = CoalesceTokenCheckpoint.from_dict(original.to_dict())
        assert restored == original


class TestCoalescePendingCheckpointFromDict:
    """from_dict validation for CoalescePendingCheckpoint."""

    def test_rejects_missing_fields(self) -> None:
        """from_dict raises AuditIntegrityError on missing required fields."""
        with pytest.raises(AuditIntegrityError, match="missing required fields"):
            CoalescePendingCheckpoint.from_dict({"coalesce_name": "merge_1"})

    def test_roundtrip(self) -> None:
        """to_dict → from_dict roundtrip preserves data."""
        token = _valid_token()
        original = CoalescePendingCheckpoint(
            coalesce_name="merge_1",
            row_id="row-1",
            elapsed_age_seconds=1.5,
            branches={"path_a": token},
            lost_branches={"path_b": "timed_out"},
        )
        restored = CoalescePendingCheckpoint.from_dict(original.to_dict())
        assert restored == original


class TestCoalesceCheckpointStatePostInit:
    """__post_init__ validation for CoalesceCheckpointState."""

    def test_valid_construction(self) -> None:
        state = CoalesceCheckpointState(
            version="1.0",
            pending=(),
            completed_keys=(("merge_1", "row-1"),),
        )
        assert state.version == "1.0"

    def test_rejects_empty_version(self) -> None:
        with pytest.raises(ValueError, match="version"):
            CoalesceCheckpointState(version="", pending=(), completed_keys=())

    def test_rejects_non_string_version(self) -> None:
        with pytest.raises(TypeError, match="version"):
            CoalesceCheckpointState(version=42, pending=(), completed_keys=())  # type: ignore[arg-type]

    def test_rejects_wrong_length_completed_key(self) -> None:
        with pytest.raises(ValueError, match="completed_keys"):
            CoalesceCheckpointState(
                version="1.0",
                pending=(),
                completed_keys=(("a", "b", "c"),),  # type: ignore[arg-type]
            )

    def test_rejects_non_string_completed_key_elements(self) -> None:
        with pytest.raises(ValueError, match="completed_keys"):
            CoalesceCheckpointState(
                version="1.0",
                pending=(),
                completed_keys=((1, 2),),  # type: ignore[arg-type]
            )


class TestCoalescePendingCheckpointFromDictTypeValidation:
    """from_dict type validation for branches and lost_branches."""

    def test_rejects_non_dict_branches(self) -> None:
        """branches must be a dict — catches corruption where JSON decoded to list."""
        data = {
            "coalesce_name": "merge_1",
            "row_id": "row-1",
            "elapsed_age_seconds": 1.5,
            "branches": ["not", "a", "dict"],
            "lost_branches": {},
        }
        with pytest.raises(AuditIntegrityError, match=r"branches.*must be a dict"):
            CoalescePendingCheckpoint.from_dict(data)

    def test_rejects_non_dict_lost_branches(self) -> None:
        """lost_branches must be a dict — catches corruption where JSON decoded to list."""
        data = {
            "coalesce_name": "merge_1",
            "row_id": "row-1",
            "elapsed_age_seconds": 1.5,
            "branches": {},
            "lost_branches": "not_a_dict",
        }
        with pytest.raises(AuditIntegrityError, match=r"lost_branches.*must be a dict"):
            CoalescePendingCheckpoint.from_dict(data)


class TestCoalesceCheckpointStateFromDict:
    """from_dict validation for CoalesceCheckpointState."""

    def test_roundtrip(self) -> None:
        token = _valid_token()
        pending = CoalescePendingCheckpoint(
            coalesce_name="merge_1",
            row_id="row-1",
            elapsed_age_seconds=1.5,
            branches={"path_a": token},
            lost_branches={},
        )
        original = CoalesceCheckpointState(
            version="1.0",
            pending=(pending,),
            completed_keys=(("merge_2", "row-2"),),
        )
        restored = CoalesceCheckpointState.from_dict(original.to_dict())
        assert restored == original

    def test_rejects_missing_version(self) -> None:
        with pytest.raises(AuditIntegrityError, match="_version"):
            CoalesceCheckpointState.from_dict({"pending": [], "completed_keys": []})

    def test_rejects_missing_pending(self) -> None:
        """Missing 'pending' key in checkpoint data is corruption."""
        with pytest.raises(AuditIntegrityError, match="pending"):
            CoalesceCheckpointState.from_dict({"_version": "1.0", "completed_keys": []})

    def test_rejects_pending_not_a_list(self) -> None:
        """'pending' must be a list — catches corruption where value is wrong type."""
        with pytest.raises(AuditIntegrityError, match=r"pending.*must be a list"):
            CoalesceCheckpointState.from_dict({"_version": "1.0", "pending": "not_a_list", "completed_keys": []})

    def test_rejects_missing_completed_keys(self) -> None:
        """Missing 'completed_keys' key in checkpoint data is corruption."""
        with pytest.raises(AuditIntegrityError, match="completed_keys"):
            CoalesceCheckpointState.from_dict({"_version": "1.0", "pending": []})

    def test_rejects_completed_keys_not_a_list(self) -> None:
        """'completed_keys' must be a list — catches corruption where value is wrong type."""
        with pytest.raises(AuditIntegrityError, match=r"completed_keys.*must be a list"):
            CoalesceCheckpointState.from_dict({"_version": "1.0", "pending": [], "completed_keys": "not_a_list"})

    def test_rejects_corrupt_completed_key_wrong_length(self) -> None:
        with pytest.raises(AuditIntegrityError, match="completed_keys"):
            CoalesceCheckpointState.from_dict(
                {
                    "_version": "1.0",
                    "pending": [],
                    "completed_keys": [[1, 2, 3]],
                }
            )

    def test_rejects_corrupt_completed_key_non_string(self) -> None:
        with pytest.raises(AuditIntegrityError, match="completed_keys"):
            CoalesceCheckpointState.from_dict(
                {
                    "_version": "1.0",
                    "pending": [],
                    "completed_keys": [[1, 2]],
                }
            )

    def test_json_round_trip_preserves_tuple_types(self) -> None:
        """Round-trip through JSON must restore tuples, not lists.

        json.dumps converts tuple → list.  json.loads always returns list.
        from_dict must reconstruct tuples so equality holds.
        This is the real persistence path: to_dict → json.dumps → json.loads → from_dict.
        """
        import json

        token = _valid_token()
        pending = CoalescePendingCheckpoint(
            coalesce_name="merge_1",
            row_id="row-1",
            elapsed_age_seconds=1.5,
            branches={"path_a": token},
            lost_branches={"path_b": "timed_out"},
        )
        original = CoalesceCheckpointState(
            version="1.0",
            pending=(pending,),
            completed_keys=(("merge_2", "row-2"), ("merge_3", "row-3")),
        )

        serialized = json.dumps(original.to_dict())
        deserialized = json.loads(serialized)
        restored = CoalesceCheckpointState.from_dict(deserialized)

        assert restored == original
        # Verify pending is a tuple, not a list
        assert isinstance(restored.pending, tuple), f"Expected tuple, got {type(restored.pending).__name__}"
        # Verify completed_keys are tuples of tuples, not lists of lists
        assert isinstance(restored.completed_keys, tuple), f"Expected tuple, got {type(restored.completed_keys).__name__}"
        for key in restored.completed_keys:
            assert isinstance(key, tuple), f"Expected tuple, got {type(key).__name__}: {key}"

    def test_json_round_trip_empty_state(self) -> None:
        """JSON round-trip with empty pending and completed_keys."""
        import json

        original = CoalesceCheckpointState(
            version="1.0",
            pending=(),
            completed_keys=(),
        )

        serialized = json.dumps(original.to_dict())
        deserialized = json.loads(serialized)
        restored = CoalesceCheckpointState.from_dict(deserialized)

        assert restored == original
        assert isinstance(restored.pending, tuple)
        assert isinstance(restored.completed_keys, tuple)

    def test_full_roundtrip_with_all_optional_fields(self) -> None:
        """Full round-trip with non-None group IDs, multiple pending entries, and completed keys.

        Exercises the to_dict list serialization → from_dict tuple reconstruction
        path for completed_keys and the nested branch token serialization with
        all optional group ID fields populated.
        """
        token_a = CoalesceTokenCheckpoint(
            token_id="tok-a",
            row_id="row-1",
            branch_name="path_a",
            fork_group_id="fork-1",
            join_group_id="join-1",
            expand_group_id="expand-1",
            row_data={"nested": {"deep": [1, 2]}},
            contract={"mode": "OBSERVED", "locked": True, "fields": ["x"]},
            state_id="state-a",
            arrival_offset_seconds=1.25,
        )
        token_b = CoalesceTokenCheckpoint(
            token_id="tok-b",
            row_id="row-1",
            branch_name="path_b",
            fork_group_id="fork-1",
            join_group_id=None,
            expand_group_id=None,
            row_data={"value": "hello"},
            contract={"mode": "FLEXIBLE"},
            state_id="state-b",
            arrival_offset_seconds=0.0,
        )
        pending_1 = CoalescePendingCheckpoint(
            coalesce_name="merge_1",
            row_id="row-1",
            elapsed_age_seconds=3.5,
            branches={"path_a": token_a, "path_b": token_b},
            lost_branches={"path_c": "timed_out"},
        )
        pending_2 = CoalescePendingCheckpoint(
            coalesce_name="merge_2",
            row_id="row-2",
            elapsed_age_seconds=0.0,
            branches={},
            lost_branches={},
        )
        original = CoalesceCheckpointState(
            version="2.0",
            pending=(pending_1, pending_2),
            completed_keys=(("merge_3", "row-3"), ("merge_4", "row-4")),
        )

        serialized = original.to_dict()

        # Verify serialization format: completed_keys become lists (JSON arrays)
        assert isinstance(serialized["completed_keys"][0], list)

        restored = CoalesceCheckpointState.from_dict(serialized)
        assert restored == original
        # Verify completed_keys are tuples after deserialization
        assert isinstance(restored.completed_keys[0], tuple)


class TestHasResumableState:
    """Tests for has_resumable_state property."""

    def test_empty_state_is_not_resumable(self) -> None:
        state = CoalesceCheckpointState(version="1.0", pending=(), completed_keys=())
        assert state.has_resumable_state is False

    def test_pending_only_is_resumable(self) -> None:
        pending = CoalescePendingCheckpoint(
            coalesce_name="merge_1",
            row_id="row-1",
            elapsed_age_seconds=0.0,
            branches={},
            lost_branches={},
        )
        state = CoalesceCheckpointState(version="1.0", pending=(pending,), completed_keys=())
        assert state.has_resumable_state is True

    def test_completed_keys_only_is_resumable(self) -> None:
        """Regression: completed_keys without pending must still be resumable."""
        state = CoalesceCheckpointState(
            version="1.0",
            pending=(),
            completed_keys=(("merge_1", "row-1"),),
        )
        assert state.has_resumable_state is True

    def test_both_pending_and_completed_keys_is_resumable(self) -> None:
        pending = CoalescePendingCheckpoint(
            coalesce_name="merge_1",
            row_id="row-1",
            elapsed_age_seconds=0.0,
            branches={},
            lost_branches={},
        )
        state = CoalesceCheckpointState(
            version="1.0",
            pending=(pending,),
            completed_keys=(("merge_2", "row-2"),),
        )
        assert state.has_resumable_state is True
