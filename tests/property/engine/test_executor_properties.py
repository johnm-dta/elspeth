# tests/property/engine/test_executor_properties.py
"""Property-based tests for executor routing behavior.

These tests verify CRITICAL audit trail integrity properties:
- Transform results are preserved unchanged through executors
- Gate routing decisions are deterministic
- RoutingAction invariants are enforced
- RoutingKind round-trips through string value

Per executors.py architecture:
- TransformExecutor wraps transform.process() with audit recording
- GateExecutor wraps gate.evaluate() with routing event recording
- Results must be preserved without corruption for audit integrity

These property tests prove the executor contracts hold for ALL possible
inputs, not just specific test cases.
"""

from __future__ import annotations

from typing import Any

from hypothesis import given, settings
from hypothesis import strategies as st

from elspeth.contracts import RoutingAction, TransformErrorReason, TransformResult
from elspeth.contracts.enums import RoutingKind, RoutingMode
from tests.property.conftest import (
    branch_names,
    dict_keys,
    json_primitives,
    multiple_branches,
    row_data,
)

# =============================================================================
# Strategies for RoutingAction and RoutingKind
# =============================================================================

# All RoutingKind values
all_routing_kinds = st.sampled_from(list(RoutingKind))

# All RoutingMode values
all_routing_modes = st.sampled_from(list(RoutingMode))

# ConfigGateReason: condition + result (from config-driven gates)
config_gate_reasons = st.fixed_dictionaries(
    {
        "condition": st.text(min_size=1, max_size=100),
        "result": st.text(min_size=1, max_size=30),
    }
)

# PluginGateReason: rule + matched_value + optional threshold fields
plugin_gate_reasons = st.fixed_dictionaries(
    {
        "rule": st.text(min_size=1, max_size=100),
        "matched_value": json_primitives,
    },
    optional={
        "threshold": st.floats(allow_nan=False, allow_infinity=False),
        "field": st.text(min_size=1, max_size=50, alphabet="abcdefghijklmnopqrstuvwxyz_"),
        "comparison": st.sampled_from([">", "<", ">=", "<=", "==", "!="]),
    },
)

# RoutingReason is ConfigGateReason | PluginGateReason
routing_reasons = st.one_of(
    st.none(),
    config_gate_reasons,
    plugin_gate_reasons,
)

# Valid route labels (non-empty strings for destinations)
route_labels = st.text(
    min_size=1,
    max_size=30,
    alphabet="abcdefghijklmnopqrstuvwxyz_0123456789",
).filter(lambda s: s[0].isalpha())

# Error reason dictionaries for TransformResult.error()
# Valid TransformErrorReason requires "reason" field with Literal-typed value
# Use a subset of common error categories for property testing
_test_error_categories = [
    "api_error",
    "missing_field",
    "validation_failed",
    "test_error",
    "property_test_error",
]

error_reasons = st.fixed_dictionaries(
    {"reason": st.sampled_from(_test_error_categories)},
    optional={
        "error": st.text(min_size=1, max_size=100),
        "field": st.text(min_size=1, max_size=50, alphabet="abcdefghijklmnopqrstuvwxyz_"),
        "status_code": st.integers(min_value=100, max_value=599),
        "query": st.text(min_size=1, max_size=50),
    },
)


# =============================================================================
# TransformResult Property Tests
# =============================================================================


class TestTransformResultProperties:
    """Property tests for TransformResult integrity."""

    @given(data=row_data)
    @settings(max_examples=100)
    def test_success_result_preserves_row_data(self, data: dict[str, Any]) -> None:
        """Property: TransformResult.success() preserves row data unchanged.

        When a transform returns success, the row data must be exactly what
        was passed in. This is critical for audit integrity - any corruption
        would break lineage queries.
        """
        result = TransformResult.success(data, success_reason={"action": "test"})

        assert result.status == "success"
        assert result.row == data
        assert result.reason is None
        assert result.rows is None
        assert not result.is_multi_row
        assert result.has_output_data

    @given(data=row_data)
    @settings(max_examples=100)
    def test_success_result_row_is_same_object(self, data: dict[str, Any]) -> None:
        """Property: success() does not copy the row data.

        The row dict is passed through by reference for efficiency.
        This means mutations to the original dict would affect the result.
        Plugins must not mutate input data after returning.
        """
        result = TransformResult.success(data, success_reason={"action": "test"})

        # Same object identity
        assert result.row is data

    @given(rows=st.lists(row_data, min_size=1, max_size=5))
    @settings(max_examples=100)
    def test_success_multi_result_preserves_rows(self, rows: list[dict[str, Any]]) -> None:
        """Property: TransformResult.success_multi() preserves all rows unchanged.

        Multi-row output must preserve each row exactly as provided.
        """
        result = TransformResult.success_multi(rows, success_reason={"action": "test"})

        assert result.status == "success"
        assert result.row is None
        assert result.rows == rows
        assert result.reason is None
        assert result.is_multi_row
        assert result.has_output_data

    @given(reason=error_reasons, retryable=st.booleans())
    @settings(max_examples=100)
    def test_error_result_preserves_reason_intact(self, reason: TransformErrorReason, retryable: bool) -> None:
        """Property: TransformResult.error() preserves reason dict unchanged.

        Error reasons must be preserved exactly for audit trail integrity.
        The reason dict contains diagnostic information that operators and
        auditors need to understand what went wrong.
        """
        result = TransformResult.error(reason, retryable=retryable)

        assert result.status == "error"
        assert result.row is None
        assert result.rows is None
        assert result.reason == reason
        assert result.retryable == retryable
        assert not result.has_output_data

    @given(reason=error_reasons)
    @settings(max_examples=100)
    def test_error_result_reason_is_same_object(self, reason: TransformErrorReason) -> None:
        """Property: error() does not copy the reason dict.

        Like success(), error() passes the reason by reference.
        """
        result = TransformResult.error(reason)

        # Same object identity
        assert result.reason is reason

    def test_success_multi_empty_list_raises(self) -> None:
        """Property: success_multi() rejects empty row list.

        Empty multi-row output is invalid - if a transform has no output,
        it should return success with a single row or an error.
        """
        import pytest

        with pytest.raises(ValueError, match="at least one row"):
            TransformResult.success_multi([], success_reason={"action": "test"})


# =============================================================================
# RoutingKind Enum Property Tests
# =============================================================================


class TestRoutingKindEnumProperties:
    """Property tests for RoutingKind enum integrity."""

    @given(kind=all_routing_kinds)
    @settings(max_examples=50)
    def test_routing_kind_name_to_value_round_trip(self, kind: RoutingKind) -> None:
        """Property: RoutingKind[name].value == original.value for all kinds.

        This verifies that looking up by name returns the same enum member,
        which is critical for deserialization paths that use names.
        """
        recovered = RoutingKind[kind.name]
        assert recovered.value == kind.value
        assert recovered is kind

    @given(kind=all_routing_kinds)
    @settings(max_examples=50)
    def test_routing_kind_value_to_enum_round_trip(self, kind: RoutingKind) -> None:
        """Property: RoutingKind(value).name == original.name for all kinds.

        This verifies that looking up by value returns the same enum member,
        which is critical for database deserialization that stores values.
        """
        recovered = RoutingKind(kind.value)
        assert recovered.name == kind.name
        assert recovered is kind

    @given(kind=all_routing_kinds)
    @settings(max_examples=50)
    def test_routing_kind_value_is_lowercase_name(self, kind: RoutingKind) -> None:
        """Property: For (str, Enum), value equals lowercase name.

        ELSPETH convention: enum values are lowercase versions of names.
        """
        assert kind.value == kind.name.lower()

    @given(kind=all_routing_kinds)
    @settings(max_examples=50)
    def test_routing_kind_is_string_subclass(self, kind: RoutingKind) -> None:
        """Property: RoutingKind instances ARE strings for (str, Enum).

        Since RoutingKind inherits from (str, Enum), the enum member IS a
        string and can be compared directly to string values.
        """
        assert isinstance(kind, str)
        assert kind == kind.value

    def test_routing_kind_no_duplicate_values(self) -> None:
        """Property: All RoutingKind values are unique.

        Duplicate values would cause ambiguous deserialization.
        """
        values = [k.value for k in RoutingKind]
        assert len(values) == len(set(values))

    def test_routing_kind_expected_members(self) -> None:
        """Property: RoutingKind has exactly the expected members.

        This is a canary test - if new routing kinds are added,
        this test documents what needs to be considered.
        """
        expected = {"CONTINUE", "ROUTE", "FORK_TO_PATHS"}
        actual = {k.name for k in RoutingKind}
        assert actual == expected


# =============================================================================
# RoutingAction Property Tests
# =============================================================================


class TestRoutingActionProperties:
    """Property tests for RoutingAction invariants."""

    @given(reason=routing_reasons)
    @settings(max_examples=100)
    def test_continue_action_has_no_destination(self, reason: dict[str, Any] | None) -> None:
        """Property: CONTINUE action always has empty destinations.

        Continue means "proceed to next node in pipeline" - there's no
        explicit destination because it follows the default path.
        """
        action = RoutingAction.continue_(reason=reason)

        assert action.kind == RoutingKind.CONTINUE
        assert action.destinations == ()
        assert len(action.destinations) == 0
        assert action.mode == RoutingMode.MOVE

    @given(label=route_labels, reason=routing_reasons)
    @settings(max_examples=100)
    def test_route_action_contains_correct_sink_name(self, label: str, reason: dict[str, Any] | None) -> None:
        """Property: ROUTE action contains exactly the specified sink name.

        Route labels are semantic identifiers (e.g., "above", "below") that
        the executor resolves to actual sink names via the routes config.
        """
        action = RoutingAction.route(label, reason=reason)

        assert action.kind == RoutingKind.ROUTE
        assert action.destinations == (label,)
        assert len(action.destinations) == 1
        assert action.destinations[0] == label
        assert action.mode == RoutingMode.MOVE

    @given(paths=multiple_branches, reason=routing_reasons)
    @settings(max_examples=100)
    def test_fork_action_contains_all_branch_names(self, paths: list[str], reason: dict[str, Any] | None) -> None:
        """Property: FORK_TO_PATHS action contains all specified branch names.

        Fork operations create child tokens for each branch. All branch names
        must be preserved in the action for correct routing.
        """
        action = RoutingAction.fork_to_paths(paths, reason=reason)

        assert action.kind == RoutingKind.FORK_TO_PATHS
        assert action.destinations == tuple(paths)
        assert len(action.destinations) == len(paths)
        for path in paths:
            assert path in action.destinations
        assert action.mode == RoutingMode.COPY

    @given(reason=routing_reasons)
    @settings(max_examples=50)
    def test_continue_uses_move_mode(self, reason: dict[str, Any] | None) -> None:
        """Property: CONTINUE always uses MOVE mode.

        Continue is semantically a move - the token proceeds on its current
        path without copying.
        """
        action = RoutingAction.continue_(reason=reason)
        assert action.mode == RoutingMode.MOVE

    @given(paths=multiple_branches)
    @settings(max_examples=50)
    def test_fork_always_uses_copy_mode(self, paths: list[str]) -> None:
        """Property: FORK_TO_PATHS always uses COPY mode.

        Fork creates child tokens - this is inherently a copy operation.
        The invariant is enforced by __post_init__.
        """
        action = RoutingAction.fork_to_paths(paths)
        assert action.mode == RoutingMode.COPY

    @given(label=route_labels)
    @settings(max_examples=50)
    def test_route_rejects_copy_mode(self, label: str) -> None:
        """Property: ROUTE cannot use COPY mode.

        COPY with ROUTE would require dual terminal states (ROUTED + COMPLETED),
        which violates ELSPETH's single terminal state per token model.
        Use FORK_TO_PATHS to achieve "route to sink and continue".
        """
        import pytest

        with pytest.raises(ValueError, match="COPY mode not supported"):
            RoutingAction.route(label, mode=RoutingMode.COPY)

    def test_fork_rejects_empty_paths(self) -> None:
        """Property: FORK_TO_PATHS rejects empty path list.

        Forking to zero branches is meaningless and likely a bug.
        """
        import pytest

        with pytest.raises(ValueError, match="at least one destination"):
            RoutingAction.fork_to_paths([])

    @given(path=branch_names)
    @settings(max_examples=50)
    def test_fork_rejects_duplicate_paths(self, path: str) -> None:
        """Property: FORK_TO_PATHS rejects duplicate path names.

        Duplicate paths would create ambiguous routing - which child token
        gets which branch? The invariant ensures unique branch names.
        """
        import pytest

        # Create a list with duplicates
        with pytest.raises(ValueError, match="unique path names"):
            RoutingAction.fork_to_paths([path, path])

    @given(label=route_labels)
    @settings(max_examples=50)
    def test_route_rejects_multiple_destinations(self, label: str) -> None:
        """Property: ROUTE must have exactly one destination.

        ROUTE is for single-destination routing. For multiple destinations,
        use FORK_TO_PATHS.
        """
        import pytest

        # Direct construction with multiple destinations should fail validation
        with pytest.raises(ValueError, match="exactly one destination"):
            RoutingAction(
                kind=RoutingKind.ROUTE,
                destinations=(label, label + "_other"),
                mode=RoutingMode.MOVE,
            )


class TestRoutingActionReasonImmutability:
    """Property tests for RoutingAction reason immutability.

    RoutingAction uses deep copy to prevent external mutation. The frozen
    dataclass prevents reassignment of the reason field, but the dict
    itself is a regular dict (not MappingProxyType) for TypedDict compatibility.
    """

    @given(reason=st.dictionaries(dict_keys, json_primitives, min_size=1, max_size=3))
    @settings(max_examples=100)
    def test_reason_preserves_type(self, reason: dict[str, Any]) -> None:
        """Property: Reason dict is a dict after construction.

        RoutingAction.reason is a deep-copied dict (for TypedDict compatibility).
        The frozen dataclass prevents reassignment; deep copy prevents
        external mutation via retained references.
        """
        action = RoutingAction.continue_(reason=reason)

        # Reason is a dict (TypedDict compatible)
        assert isinstance(action.reason, dict)
        assert action.reason == reason

    @given(reason=st.dictionaries(dict_keys, json_primitives, min_size=1, max_size=3))
    @settings(max_examples=100)
    def test_reason_is_deep_copied(self, reason: dict[str, Any]) -> None:
        """Property: Reason dict is deep-copied to prevent external mutation.

        Modifying the original dict after creating the action must not
        affect the action's reason. This prevents subtle bugs where
        shared mutable state corrupts audit records.
        """
        original_reason = dict(reason)
        action = RoutingAction.continue_(reason=reason)

        # Mutate the original
        reason["__after_creation__"] = True

        # Action's reason should not be affected
        assert "__after_creation__" not in action.reason
        assert action.reason == original_reason


# =============================================================================
# RoutingMode Enum Property Tests
# =============================================================================


class TestRoutingModeEnumProperties:
    """Property tests for RoutingMode enum integrity."""

    @given(mode=all_routing_modes)
    @settings(max_examples=50)
    def test_routing_mode_round_trip(self, mode: RoutingMode) -> None:
        """Property: RoutingMode round-trips through name and value."""
        # Via name
        assert RoutingMode[mode.name] is mode
        # Via value
        assert RoutingMode(mode.value) is mode

    @given(mode=all_routing_modes)
    @settings(max_examples=50)
    def test_routing_mode_is_string_subclass(self, mode: RoutingMode) -> None:
        """Property: RoutingMode instances ARE strings for (str, Enum)."""
        assert isinstance(mode, str)
        assert mode == mode.value

    def test_routing_mode_expected_members(self) -> None:
        """Property: RoutingMode has exactly MOVE and COPY."""
        expected = {"MOVE", "COPY"}
        actual = {m.name for m in RoutingMode}
        assert actual == expected
