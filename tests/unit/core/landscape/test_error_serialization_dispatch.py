"""Tests for the isinstance dispatch in _node_state_recording.py:199.

This module covers the critical audit boundary where error DTOs are serialized
before being written to the Landscape database. Three error types flow through
this path:

1. ExecutionError (frozen dataclass) -- serialized via to_dict()
2. CoalesceFailureReason (frozen dataclass) -- serialized via to_dict()
3. TransformErrorReason (TypedDict, already a dict) -- passes through directly

The dispatch logic:
    error_data = error.to_dict() if isinstance(error, (ExecutionError, CoalesceFailureReason)) else error
    error_json = canonical_json(error_data)
"""

from __future__ import annotations

import json

import pytest

from elspeth.contracts import NodeStateFailed, NodeStateStatus, NodeType
from elspeth.contracts.errors import CoalesceFailureReason, ExecutionError, TransformErrorReason
from elspeth.core.canonical import canonical_json, stable_hash
from elspeth.core.landscape import LandscapeDB
from elspeth.core.landscape.factory import RecorderFactory
from tests.fixtures.landscape import make_recorder_with_run, register_test_node

# ---------------------------------------------------------------------------
# Helper: replicate the dispatch from _node_state_recording.py:199
# ---------------------------------------------------------------------------


def _serialize_error(
    error: ExecutionError | TransformErrorReason | CoalesceFailureReason,
) -> str:
    """Replicate the isinstance dispatch from _node_state_recording.py:199."""
    error_data = error.to_dict() if isinstance(error, (ExecutionError, CoalesceFailureReason)) else error
    return canonical_json(error_data)


# ---------------------------------------------------------------------------
# Helper: set up Landscape DB + factory with a run, node, row, and token
# ---------------------------------------------------------------------------


def _setup_with_token(
    *,
    run_id: str = "run-1",
) -> tuple[LandscapeDB, RecorderFactory, str, str]:
    setup = make_recorder_with_run(run_id=run_id, source_node_id="source-0")
    db, factory, run_id = setup.db, setup.factory, setup.run_id
    register_test_node(factory.data_flow, run_id, "transform-1", node_type=NodeType.TRANSFORM, plugin_name="transform")
    row = factory.data_flow.create_row(run_id, "source-0", 0, {"name": "test"}, row_id="row-1")
    token = factory.data_flow.create_token("row-1", token_id="tok-1")
    return db, factory, row.row_id, token.token_id


# =============================================================================
# Unit tests: dispatch logic in isolation
# =============================================================================


class TestExecutionErrorSerialization:
    """ExecutionError (frozen dataclass) is serialized via to_dict()."""

    def test_to_dict_produces_correct_keys(self):
        error = ExecutionError(exception="boom", exception_type="ValueError")
        d = error.to_dict()
        assert "exception" in d
        assert "type" in d
        # The field is named exception_type on the dataclass, but serialized as "type"
        assert "exception_type" not in d

    def test_to_dict_values_match(self):
        error = ExecutionError(exception="division by zero", exception_type="ZeroDivisionError")
        d = error.to_dict()
        assert d["exception"] == "division by zero"
        assert d["type"] == "ZeroDivisionError"

    def test_to_dict_omits_none_optional_fields(self):
        error = ExecutionError(exception="err", exception_type="RuntimeError")
        d = error.to_dict()
        assert "traceback" not in d
        assert "phase" not in d

    def test_to_dict_includes_optional_fields_when_set(self):
        error = ExecutionError(
            exception="err",
            exception_type="RuntimeError",
            traceback="Traceback (most recent call last):\n  ...",
            phase="flush",
        )
        d = error.to_dict()
        assert d["traceback"] == "Traceback (most recent call last):\n  ..."
        assert d["phase"] == "flush"

    def test_dispatch_selects_to_dict(self):
        error = ExecutionError(exception="boom", exception_type="ValueError")
        result_json = _serialize_error(error)
        parsed = json.loads(result_json)
        # Should have been serialized via to_dict(), producing "type" not "exception_type"
        assert parsed["type"] == "ValueError"
        assert "exception_type" not in parsed

    def test_canonical_json_is_valid(self):
        error = ExecutionError(exception="timeout", exception_type="TimeoutError")
        result_json = _serialize_error(error)
        # canonical_json must produce valid JSON
        parsed = json.loads(result_json)
        assert isinstance(parsed, dict)


class TestCoalesceFailureReasonSerialization:
    """CoalesceFailureReason (frozen dataclass) is serialized via to_dict()."""

    def test_to_dict_produces_correct_keys(self):
        reason = CoalesceFailureReason(
            failure_reason="quorum_not_met",
            expected_branches=["path_a", "path_b"],
            branches_arrived=["path_a"],
            merge_policy="all_or_fail",
        )
        d = reason.to_dict()
        assert d["failure_reason"] == "quorum_not_met"
        assert d["expected_branches"] == ["path_a", "path_b"]
        assert d["branches_arrived"] == ["path_a"]
        assert d["merge_policy"] == "all_or_fail"

    def test_to_dict_omits_none_optional_fields(self):
        reason = CoalesceFailureReason(
            failure_reason="timeout",
            expected_branches=["a", "b"],
            branches_arrived=["a"],
            merge_policy="wait_all",
        )
        d = reason.to_dict()
        assert "timeout_ms" not in d
        assert "select_branch" not in d

    def test_to_dict_includes_optional_fields_when_set(self):
        reason = CoalesceFailureReason(
            failure_reason="select_branch_missing",
            expected_branches=["a", "b", "c"],
            branches_arrived=["a", "c"],
            merge_policy="select",
            timeout_ms=5000,
            select_branch="b",
        )
        d = reason.to_dict()
        assert d["timeout_ms"] == 5000
        assert d["select_branch"] == "b"

    def test_dispatch_selects_to_dict(self):
        reason = CoalesceFailureReason(
            failure_reason="quorum_not_met",
            expected_branches=["x", "y"],
            branches_arrived=["x"],
            merge_policy="all_or_fail",
        )
        result_json = _serialize_error(reason)
        parsed = json.loads(result_json)
        assert parsed["failure_reason"] == "quorum_not_met"
        assert parsed["expected_branches"] == ["x", "y"]

    def test_canonical_json_is_valid(self):
        reason = CoalesceFailureReason(
            failure_reason="timeout",
            expected_branches=["a"],
            branches_arrived=[],
            merge_policy="wait_all",
            timeout_ms=3000,
        )
        result_json = _serialize_error(reason)
        parsed = json.loads(result_json)
        assert isinstance(parsed, dict)


class TestTransformErrorReasonPassthrough:
    """TransformErrorReason (TypedDict, already a dict) passes through directly."""

    def test_is_a_dict(self):
        reason: TransformErrorReason = {"reason": "api_error", "error": "connection refused"}
        assert isinstance(reason, dict)

    def test_dispatch_does_not_call_to_dict(self):
        reason: TransformErrorReason = {"reason": "missing_field", "field": "customer_id"}
        # TransformErrorReason is a dict -- isinstance check against
        # (ExecutionError, CoalesceFailureReason) must be False
        assert not isinstance(reason, (ExecutionError, CoalesceFailureReason))

    def test_dispatch_passes_dict_through(self):
        reason: TransformErrorReason = {"reason": "json_parse_failed", "error": "Expecting value"}
        result_json = _serialize_error(reason)
        parsed = json.loads(result_json)
        assert parsed["reason"] == "json_parse_failed"
        assert parsed["error"] == "Expecting value"

    def test_canonical_json_is_valid(self):
        reason: TransformErrorReason = {
            "reason": "template_rendering_failed",
            "error": "undefined variable 'x'",
            "template_hash": "abc123",
        }
        result_json = _serialize_error(reason)
        parsed = json.loads(result_json)
        assert isinstance(parsed, dict)


# =============================================================================
# Unit tests: hash stability
# =============================================================================


class TestHashStability:
    """Canonical JSON from frozen dataclass to_dict() must match hand-built dict."""

    def test_execution_error_hash_matches_hand_built_dict(self):
        error = ExecutionError(exception="boom", exception_type="ValueError")
        hand_built = {"exception": "boom", "type": "ValueError"}

        json_from_dto = canonical_json(error.to_dict())
        json_from_dict = canonical_json(hand_built)
        assert json_from_dto == json_from_dict

        hash_from_dto = stable_hash(error.to_dict())
        hash_from_dict = stable_hash(hand_built)
        assert hash_from_dto == hash_from_dict

    def test_execution_error_with_optionals_hash_matches(self):
        error = ExecutionError(
            exception="err",
            exception_type="RuntimeError",
            traceback="line 42",
            phase="flush",
        )
        hand_built = {
            "exception": "err",
            "type": "RuntimeError",
            "traceback": "line 42",
            "phase": "flush",
        }

        assert canonical_json(error.to_dict()) == canonical_json(hand_built)
        assert stable_hash(error.to_dict()) == stable_hash(hand_built)

    def test_coalesce_failure_hash_matches_hand_built_dict(self):
        reason = CoalesceFailureReason(
            failure_reason="quorum_not_met",
            expected_branches=["a", "b"],
            branches_arrived=["a"],
            merge_policy="all_or_fail",
        )
        hand_built = {
            "failure_reason": "quorum_not_met",
            "expected_branches": ["a", "b"],
            "branches_arrived": ["a"],
            "merge_policy": "all_or_fail",
        }

        json_from_dto = canonical_json(reason.to_dict())
        json_from_dict = canonical_json(hand_built)
        assert json_from_dto == json_from_dict

        hash_from_dto = stable_hash(reason.to_dict())
        hash_from_dict = stable_hash(hand_built)
        assert hash_from_dto == hash_from_dict

    def test_coalesce_failure_with_optionals_hash_matches(self):
        reason = CoalesceFailureReason(
            failure_reason="select_branch_missing",
            expected_branches=["x", "y", "z"],
            branches_arrived=["x", "z"],
            merge_policy="select",
            timeout_ms=10000,
            select_branch="y",
        )
        hand_built = {
            "failure_reason": "select_branch_missing",
            "expected_branches": ["x", "y", "z"],
            "branches_arrived": ["x", "z"],
            "merge_policy": "select",
            "timeout_ms": 10000,
            "select_branch": "y",
        }

        assert canonical_json(reason.to_dict()) == canonical_json(hand_built)
        assert stable_hash(reason.to_dict()) == stable_hash(hand_built)

    def test_transform_error_reason_hash_is_itself(self):
        """TransformErrorReason is already a dict, so JSON is identical."""
        reason: TransformErrorReason = {"reason": "api_error", "error": "timeout"}
        hand_built = {"reason": "api_error", "error": "timeout"}

        assert canonical_json(reason) == canonical_json(hand_built)
        assert stable_hash(reason) == stable_hash(hand_built)


# =============================================================================
# Integration tests: full path through ExecutionRepository.complete_node_state()
# =============================================================================


class TestCompleteNodeStateWithExecutionError:
    """Exercise the full path: ExecutionError -> complete_node_state -> DB."""

    def test_execution_error_stored_in_db(self):
        _db, factory, _row_id, token_id = _setup_with_token()

        state = factory.execution.begin_node_state(
            token_id=token_id,
            node_id="source-0",
            run_id="run-1",
            step_index=0,
            input_data={"x": 1},
        )

        error = ExecutionError(exception="division by zero", exception_type="ZeroDivisionError")
        result = factory.execution.complete_node_state(
            state.state_id,
            NodeStateStatus.FAILED,
            error=error,
            duration_ms=42.0,
        )

        assert isinstance(result, NodeStateFailed)
        assert result.error_json is not None

        parsed = json.loads(result.error_json)
        assert parsed["exception"] == "division by zero"
        assert parsed["type"] == "ZeroDivisionError"
        # Must NOT contain the dataclass field name
        assert "exception_type" not in parsed

    def test_execution_error_with_traceback_stored(self):
        _db, factory, _row_id, token_id = _setup_with_token()

        state = factory.execution.begin_node_state(
            token_id=token_id,
            node_id="source-0",
            run_id="run-1",
            step_index=0,
            input_data={"a": 1},
        )

        error = ExecutionError(
            exception="key 'missing'",
            exception_type="KeyError",
            traceback="File transform.py, line 10\n  raise KeyError('missing')",
            phase="flush",
        )
        result = factory.execution.complete_node_state(
            state.state_id,
            NodeStateStatus.FAILED,
            error=error,
            duration_ms=15.0,
        )

        assert isinstance(result, NodeStateFailed)
        parsed = json.loads(result.error_json)
        assert parsed["exception"] == "key 'missing'"
        assert parsed["type"] == "KeyError"
        assert "line 10" in parsed["traceback"]
        assert parsed["phase"] == "flush"

    def test_execution_error_json_is_canonical(self):
        """The error_json must be canonical (deterministic) for audit integrity."""
        _db, factory, _row_id, token_id = _setup_with_token()

        state = factory.execution.begin_node_state(
            token_id=token_id,
            node_id="source-0",
            run_id="run-1",
            step_index=0,
            input_data={"z": 1},
        )

        error = ExecutionError(exception="boom", exception_type="RuntimeError")
        factory.execution.complete_node_state(
            state.state_id,
            NodeStateStatus.FAILED,
            error=error,
            duration_ms=5.0,
        )

        fetched = factory.execution.get_node_state(state.state_id)
        assert isinstance(fetched, NodeStateFailed)
        # The stored JSON must match what canonical_json produces
        expected_json = canonical_json(error.to_dict())
        assert fetched.error_json == expected_json


class TestCompleteNodeStateWithCoalesceFailureReason:
    """Exercise the full path: CoalesceFailureReason -> complete_node_state -> DB."""

    def test_coalesce_failure_stored_in_db(self):
        _db, factory, _row_id, token_id = _setup_with_token()

        state = factory.execution.begin_node_state(
            token_id=token_id,
            node_id="source-0",
            run_id="run-1",
            step_index=0,
            input_data={"x": 1},
        )

        reason = CoalesceFailureReason(
            failure_reason="quorum_not_met",
            expected_branches=["path_a", "path_b"],
            branches_arrived=["path_a"],
            merge_policy="all_or_fail",
        )
        result = factory.execution.complete_node_state(
            state.state_id,
            NodeStateStatus.FAILED,
            error=reason,
            duration_ms=100.0,
        )

        assert isinstance(result, NodeStateFailed)
        assert result.error_json is not None

        parsed = json.loads(result.error_json)
        assert parsed["failure_reason"] == "quorum_not_met"
        assert parsed["expected_branches"] == ["path_a", "path_b"]
        assert parsed["branches_arrived"] == ["path_a"]
        assert parsed["merge_policy"] == "all_or_fail"

    def test_coalesce_failure_with_optionals_stored(self):
        _db, factory, _row_id, token_id = _setup_with_token()

        state = factory.execution.begin_node_state(
            token_id=token_id,
            node_id="source-0",
            run_id="run-1",
            step_index=0,
            input_data={"y": 2},
        )

        reason = CoalesceFailureReason(
            failure_reason="select_branch_missing",
            expected_branches=["a", "b", "c"],
            branches_arrived=["a", "c"],
            merge_policy="select",
            timeout_ms=5000,
            select_branch="b",
        )
        result = factory.execution.complete_node_state(
            state.state_id,
            NodeStateStatus.FAILED,
            error=reason,
            duration_ms=5001.0,
        )

        assert isinstance(result, NodeStateFailed)
        parsed = json.loads(result.error_json)
        assert parsed["timeout_ms"] == 5000
        assert parsed["select_branch"] == "b"

    def test_coalesce_failure_json_is_canonical(self):
        _db, factory, _row_id, token_id = _setup_with_token()

        state = factory.execution.begin_node_state(
            token_id=token_id,
            node_id="source-0",
            run_id="run-1",
            step_index=0,
            input_data={"q": 9},
        )

        reason = CoalesceFailureReason(
            failure_reason="timeout",
            expected_branches=["x"],
            branches_arrived=[],
            merge_policy="wait_all",
            timeout_ms=3000,
        )
        factory.execution.complete_node_state(
            state.state_id,
            NodeStateStatus.FAILED,
            error=reason,
            duration_ms=3001.0,
        )

        fetched = factory.execution.get_node_state(state.state_id)
        assert isinstance(fetched, NodeStateFailed)
        expected_json = canonical_json(reason.to_dict())
        assert fetched.error_json == expected_json


class TestCompleteNodeStateWithTransformErrorReason:
    """Exercise the full path: TransformErrorReason -> complete_node_state -> DB."""

    def test_transform_error_reason_stored_in_db(self):
        _db, factory, _row_id, token_id = _setup_with_token()

        state = factory.execution.begin_node_state(
            token_id=token_id,
            node_id="source-0",
            run_id="run-1",
            step_index=0,
            input_data={"x": 1},
        )

        reason: TransformErrorReason = {
            "reason": "api_error",
            "error": "connection refused",
            "error_type": "network_error",
        }
        result = factory.execution.complete_node_state(
            state.state_id,
            NodeStateStatus.FAILED,
            error=reason,
            duration_ms=1500.0,
        )

        assert isinstance(result, NodeStateFailed)
        assert result.error_json is not None

        parsed = json.loads(result.error_json)
        assert parsed["reason"] == "api_error"
        assert parsed["error"] == "connection refused"
        assert parsed["error_type"] == "network_error"

    def test_transform_error_reason_with_llm_context(self):
        _db, factory, _row_id, token_id = _setup_with_token()

        state = factory.execution.begin_node_state(
            token_id=token_id,
            node_id="source-0",
            run_id="run-1",
            step_index=0,
            input_data={"w": 3},
        )

        reason: TransformErrorReason = {
            "reason": "response_truncated",
            "error": "Response was truncated at 1000 tokens",
            "query": "sentiment",
            "max_tokens": 1000,
            "completion_tokens": 1000,
        }
        result = factory.execution.complete_node_state(
            state.state_id,
            NodeStateStatus.FAILED,
            error=reason,
            duration_ms=800.0,
        )

        assert isinstance(result, NodeStateFailed)
        parsed = json.loads(result.error_json)
        assert parsed["reason"] == "response_truncated"
        assert parsed["max_tokens"] == 1000

    def test_transform_error_reason_json_is_canonical(self):
        _db, factory, _row_id, token_id = _setup_with_token()

        state = factory.execution.begin_node_state(
            token_id=token_id,
            node_id="source-0",
            run_id="run-1",
            step_index=0,
            input_data={"v": 7},
        )

        reason: TransformErrorReason = {
            "reason": "missing_field",
            "field": "customer_id",
        }
        factory.execution.complete_node_state(
            state.state_id,
            NodeStateStatus.FAILED,
            error=reason,
            duration_ms=2.0,
        )

        fetched = factory.execution.get_node_state(state.state_id)
        assert isinstance(fetched, NodeStateFailed)
        expected_json = canonical_json(reason)
        assert fetched.error_json == expected_json


# =============================================================================
# Cross-type consistency: all three error types produce valid canonical JSON
# =============================================================================


class TestCrossTypeConsistency:
    """All error types must produce well-formed canonical JSON via the dispatch."""

    @pytest.mark.parametrize(
        "error",
        [
            pytest.param(
                ExecutionError(exception="boom", exception_type="ValueError"),
                id="execution_error",
            ),
            pytest.param(
                CoalesceFailureReason(
                    failure_reason="quorum_not_met",
                    expected_branches=("a", "b"),
                    branches_arrived=("a",),
                    merge_policy="all_or_fail",
                ),
                id="coalesce_failure_reason",
            ),
        ],
    )
    def test_frozen_dataclass_dispatch_produces_dict(self, error: ExecutionError | CoalesceFailureReason):
        """Both frozen dataclass types are dispatched via to_dict()."""
        assert isinstance(error, (ExecutionError, CoalesceFailureReason))
        result_json = _serialize_error(error)
        parsed = json.loads(result_json)
        assert isinstance(parsed, dict)

    def test_transform_error_reason_dispatch_passes_dict_through(self):
        reason: TransformErrorReason = {"reason": "test_error"}
        assert not isinstance(reason, (ExecutionError, CoalesceFailureReason))
        result_json = _serialize_error(reason)
        parsed = json.loads(result_json)
        assert parsed["reason"] == "test_error"

    def test_all_three_types_produce_deterministic_json(self):
        """Serialize each type twice and verify identical output."""
        errors: list[ExecutionError | TransformErrorReason | CoalesceFailureReason] = [
            ExecutionError(exception="err", exception_type="RuntimeError"),
            CoalesceFailureReason(
                failure_reason="timeout",
                expected_branches=["a"],
                branches_arrived=[],
                merge_policy="wait_all",
            ),
            {"reason": "api_error", "error": "timeout"},
        ]
        for error in errors:
            json1 = _serialize_error(error)
            json2 = _serialize_error(error)
            assert json1 == json2, f"Non-deterministic JSON for {type(error).__name__}"
