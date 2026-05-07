"""Tests for error/reason schema contracts.

Tests for:
- ExecutionError frozen dataclass (exception, exception_type, traceback, phase)
- CoalesceFailureReason frozen dataclass (failure_reason, expected_branches, etc.)
- RoutingReason TypedDict (rule, matched_value, threshold fields)
- TransformReason TypedDict (action, fields_modified fields)
"""

import dataclasses

import pytest


class TestExecutionError:
    """Tests for ExecutionError frozen dataclass — construction, immutability, serialization."""

    def test_execution_error_is_frozen_dataclass(self) -> None:
        """ExecutionError is a frozen dataclass (immutable after construction)."""
        from elspeth.contracts import ExecutionError

        assert dataclasses.is_dataclass(ExecutionError)
        error = ExecutionError(exception="test", exception_type="ValueError")
        with pytest.raises(dataclasses.FrozenInstanceError):
            error.exception = "modified"  # type: ignore[misc]

    def test_execution_error_to_dict_required_only(self) -> None:
        """to_dict() serializes exception_type as 'type' and omits None fields."""
        from elspeth.contracts import ExecutionError

        error = ExecutionError(exception="boom", exception_type="RuntimeError")
        d = error.to_dict()
        assert d == {"exception": "boom", "type": "RuntimeError"}
        assert "traceback" not in d
        assert "phase" not in d

    def test_execution_error_to_dict_with_optionals(self) -> None:
        """to_dict() includes optional fields when set."""
        from elspeth.contracts import ExecutionError

        error = ExecutionError(
            exception="boom",
            exception_type="RuntimeError",
            traceback="Traceback ...",
            phase="flush",
        )
        d = error.to_dict()
        assert d == {
            "exception": "boom",
            "type": "RuntimeError",
            "traceback": "Traceback ...",
            "phase": "flush",
        }


class TestRoutingReasonSchema:
    """Tests for RoutingReason union type schema introspection."""

    pass


class TestRoutingReason:
    """Tests for RoutingReason union type usage."""

    pass


class TestTransformSuccessReason:
    """Tests for TransformSuccessReason TypedDict — construction and Literal values."""

    pass


class TestRoutingReasonUsage:
    """Tests for constructing valid RoutingReason variants."""

    pass


class TestTransformErrorReasonContract:
    """Tests for TransformErrorReason TypedDict contract — Literal values and optional fields."""

    pass


class TestTransformErrorReasonUsage:
    """Tests for constructing valid TransformErrorReason values."""

    pass


class TestNestedTypeDicts:
    """Tests for nested TypedDict structures."""

    pass


class TestQueryFailureDetailUsage:
    """Tests for constructing valid QueryFailureDetail values."""

    pass


class TestErrorDetailUsage:
    """Tests for constructing valid ErrorDetail values."""

    pass


class TestFailedQueriesFieldType:
    """Tests for failed_queries field with union type."""

    pass


class TestErrorsFieldType:
    """Tests for errors field with union type."""

    pass


class TestCoalesceFailureReasonSchema:
    """Tests for CoalesceFailureReason frozen dataclass schema."""

    def test_is_frozen_dataclass(self) -> None:
        """CoalesceFailureReason is a frozen dataclass (immutable after construction)."""
        from elspeth.contracts import CoalesceFailureReason

        assert dataclasses.is_dataclass(CoalesceFailureReason)
        error = CoalesceFailureReason(
            failure_reason="quorum_not_met",
            expected_branches=("a", "b"),
            branches_arrived=("a",),
            merge_policy="union",
        )
        with pytest.raises(dataclasses.FrozenInstanceError):
            error.failure_reason = "modified"  # type: ignore[misc]

    def test_has_slots(self) -> None:
        """CoalesceFailureReason uses __slots__ for memory efficiency — no instance __dict__."""
        from elspeth.contracts import CoalesceFailureReason

        instance = CoalesceFailureReason(
            failure_reason="quorum_not_met",
            expected_branches=("a", "b"),
            branches_arrived=("a",),
            merge_policy="union",
        )
        assert not hasattr(instance, "__dict__"), "Slots dataclass should not have __dict__"

    def test_to_dict_required_only(self) -> None:
        """to_dict() omits None-valued optional fields."""
        from elspeth.contracts import CoalesceFailureReason

        error = CoalesceFailureReason(
            failure_reason="incomplete_branches",
            expected_branches=("path_a", "path_b"),
            branches_arrived=("path_a",),
            merge_policy="union",
        )
        d = error.to_dict()
        assert d == {
            "failure_reason": "incomplete_branches",
            "expected_branches": ["path_a", "path_b"],
            "branches_arrived": ["path_a"],
            "merge_policy": "union",
        }
        assert "timeout_ms" not in d
        assert "select_branch" not in d

    def test_to_dict_with_timeout(self) -> None:
        """to_dict() includes timeout_ms when set."""
        from elspeth.contracts import CoalesceFailureReason

        error = CoalesceFailureReason(
            failure_reason="quorum_not_met_at_timeout",
            expected_branches=("a", "b", "c"),
            branches_arrived=("a",),
            merge_policy="nested",
            timeout_ms=30000,
        )
        d = error.to_dict()
        assert d["timeout_ms"] == 30000
        assert "select_branch" not in d

    def test_to_dict_with_select_branch(self) -> None:
        """to_dict() includes select_branch when set."""
        from elspeth.contracts import CoalesceFailureReason

        error = CoalesceFailureReason(
            failure_reason="select_branch_not_arrived",
            expected_branches=("fast", "slow"),
            branches_arrived=("slow",),
            merge_policy="select",
            select_branch="fast",
        )
        d = error.to_dict()
        assert d["select_branch"] == "fast"
        assert "timeout_ms" not in d

    def test_to_dict_with_all_optionals(self) -> None:
        """to_dict() includes all fields when all are set."""
        from elspeth.contracts import CoalesceFailureReason

        error = CoalesceFailureReason(
            failure_reason="select_branch_not_arrived",
            expected_branches=("a", "b"),
            branches_arrived=("b",),
            merge_policy="select",
            timeout_ms=5000,
            select_branch="a",
        )
        d = error.to_dict()
        assert d == {
            "failure_reason": "select_branch_not_arrived",
            "expected_branches": ["a", "b"],
            "branches_arrived": ["b"],
            "merge_policy": "select",
            "timeout_ms": 5000,
            "select_branch": "a",
        }

    def test_late_arrival_has_empty_branches_arrived(self) -> None:
        """Late arrival failures have empty branches_arrived list."""
        from elspeth.contracts import CoalesceFailureReason

        error = CoalesceFailureReason(
            failure_reason="late_arrival_after_merge",
            expected_branches=("a", "b"),
            branches_arrived=(),
            merge_policy="union",
        )
        assert error.branches_arrived == ()
        assert error.to_dict()["branches_arrived"] == []


class TestExecutionErrorPostInit:
    """Tests for ExecutionError __post_init__ validation."""

    def test_rejects_empty_exception(self) -> None:
        from elspeth.contracts import ExecutionError

        with pytest.raises(ValueError, match="exception must not be empty"):
            ExecutionError(exception="", exception_type="ValueError")

    def test_rejects_empty_exception_type(self) -> None:
        from elspeth.contracts import ExecutionError

        with pytest.raises(ValueError, match="exception_type must not be empty"):
            ExecutionError(exception="boom", exception_type="")

    def test_accepts_valid_construction(self) -> None:
        from elspeth.contracts import ExecutionError

        error = ExecutionError(exception="boom", exception_type="RuntimeError")
        assert error.exception == "boom"

    def test_rejects_non_mapping_context(self) -> None:
        from elspeth.contracts import ExecutionError

        with pytest.raises(TypeError, match="context must be a mapping, got list"):
            ExecutionError(
                exception="boom",
                exception_type="RuntimeError",
                context=["bad"],  # type: ignore[arg-type]
            )

    def test_rejects_context_with_non_string_keys(self) -> None:
        from elspeth.contracts import ExecutionError

        with pytest.raises(TypeError, match="context keys must be strings"):
            ExecutionError(
                exception="boom",
                exception_type="RuntimeError",
                context={1: "bad"},  # type: ignore[arg-type]
            )


class TestCoalesceFailureReasonPostInit:
    """Tests for CoalesceFailureReason __post_init__ validation."""

    def test_rejects_empty_failure_reason(self) -> None:
        from elspeth.contracts import CoalesceFailureReason

        with pytest.raises(ValueError, match="failure_reason must not be empty"):
            CoalesceFailureReason(
                failure_reason="",
                expected_branches=("a",),
                branches_arrived=(),
                merge_policy="union",
            )

    def test_rejects_empty_merge_policy(self) -> None:
        from elspeth.contracts import CoalesceFailureReason

        with pytest.raises(ValueError, match="merge_policy must not be empty"):
            CoalesceFailureReason(
                failure_reason="quorum_not_met",
                expected_branches=("a",),
                branches_arrived=(),
                merge_policy="",
            )

    def test_rejects_empty_expected_branches(self) -> None:
        from elspeth.contracts import CoalesceFailureReason

        with pytest.raises(ValueError, match="expected_branches must not be empty"):
            CoalesceFailureReason(
                failure_reason="quorum_not_met",
                expected_branches=(),
                branches_arrived=(),
                merge_policy="union",
            )

    def test_rejects_negative_timeout_ms(self) -> None:
        from elspeth.contracts import CoalesceFailureReason

        with pytest.raises(ValueError, match="timeout_ms must be non-negative"):
            CoalesceFailureReason(
                failure_reason="timeout",
                expected_branches=("a",),
                branches_arrived=(),
                merge_policy="union",
                timeout_ms=-1,
            )

    def test_to_dict_serializes_tuples_as_lists(self) -> None:
        """to_dict() converts tuple fields to lists for JSON compatibility."""
        from elspeth.contracts import CoalesceFailureReason

        error = CoalesceFailureReason(
            failure_reason="quorum_not_met",
            expected_branches=("a", "b"),
            branches_arrived=("a",),
            merge_policy="union",
        )
        d = error.to_dict()
        assert isinstance(d["expected_branches"], list)
        assert isinstance(d["branches_arrived"], list)
        assert d["expected_branches"] == ["a", "b"]
        assert d["branches_arrived"] == ["a"]


class TestCoalesceFailureReasonDeepFreeze:
    """Branch fields must be deeply frozen on direct construction."""

    def test_expected_branches_frozen(self) -> None:
        from elspeth.contracts import CoalesceFailureReason

        branches: list[str] = ["a", "b"]
        reason = CoalesceFailureReason(
            failure_reason="quorum_not_met",
            expected_branches=branches,  # type: ignore[arg-type]
            branches_arrived=("a",),
            merge_policy="union",
        )
        branches.append("mutated")
        assert isinstance(reason.expected_branches, tuple)
        assert "mutated" not in reason.expected_branches

    def test_branches_arrived_frozen(self) -> None:
        from elspeth.contracts import CoalesceFailureReason

        arrived: list[str] = ["a"]
        reason = CoalesceFailureReason(
            failure_reason="quorum_not_met",
            expected_branches=("a", "b"),
            branches_arrived=arrived,  # type: ignore[arg-type]
            merge_policy="union",
        )
        arrived.append("mutated")
        assert isinstance(reason.branches_arrived, tuple)
        assert "mutated" not in reason.branches_arrived
