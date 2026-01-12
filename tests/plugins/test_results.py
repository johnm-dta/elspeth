# tests/plugins/test_results.py
"""Tests for plugin result types."""

from dataclasses import FrozenInstanceError

import pytest


class TestRowOutcome:
    """Terminal states for rows."""

    def test_all_terminal_states_exist(self) -> None:
        from elspeth.plugins.results import RowOutcome

        # Every row must reach exactly one terminal state
        assert RowOutcome.COMPLETED.value == "completed"
        assert RowOutcome.ROUTED.value == "routed"
        assert RowOutcome.FORKED.value == "forked"
        assert RowOutcome.CONSUMED_IN_BATCH.value == "consumed_in_batch"
        assert RowOutcome.COALESCED.value == "coalesced"
        assert RowOutcome.QUARANTINED.value == "quarantined"
        assert RowOutcome.FAILED.value == "failed"

    def test_outcome_is_enum(self) -> None:
        from enum import Enum

        from elspeth.plugins.results import RowOutcome

        assert issubclass(RowOutcome, Enum)


class TestRoutingAction:
    """Routing decisions from gates."""

    def test_continue_action(self) -> None:
        from elspeth.plugins.results import RoutingAction

        action = RoutingAction.continue_()
        assert action.kind == "continue"
        assert action.destinations == ()  # Tuple, not list
        assert action.mode == "move"

    def test_route_to_sink(self) -> None:
        from elspeth.plugins.results import RoutingAction

        action = RoutingAction.route_to_sink("flagged", reason={"confidence": 0.95})
        assert action.kind == "route_to_sink"
        assert action.destinations == ("flagged",)  # Tuple, not list
        assert action.reason["confidence"] == 0.95  # Access via mapping

    def test_fork_to_paths(self) -> None:
        from elspeth.plugins.results import RoutingAction

        action = RoutingAction.fork_to_paths(["stats", "classifier", "archive"])
        assert action.kind == "fork_to_paths"
        assert action.destinations == ("stats", "classifier", "archive")  # Tuple
        assert action.mode == "copy"

    def test_immutable(self) -> None:
        from elspeth.plugins.results import RoutingAction

        action = RoutingAction.continue_()
        with pytest.raises(FrozenInstanceError):
            action.kind = "route_to_sink"

    def test_reason_is_immutable(self) -> None:
        """Reason dict should be wrapped as immutable mapping."""
        from elspeth.plugins.results import RoutingAction

        action = RoutingAction.route_to_sink("flagged", reason={"score": 0.9})
        # Should not be able to modify reason
        with pytest.raises(TypeError):
            action.reason["score"] = 0.5


class TestTransformResult:
    """Results from transform operations."""

    def test_success_result(self) -> None:
        from elspeth.plugins.results import TransformResult

        result = TransformResult.success({"value": 42})
        assert result.status == "success"
        assert result.row == {"value": 42}
        assert result.retryable is False

    def test_error_result(self) -> None:
        from elspeth.plugins.results import TransformResult

        result = TransformResult.error(
            reason={"error": "validation failed"},
            retryable=True,
        )
        assert result.status == "error"
        assert result.row is None
        assert result.retryable is True

    def test_has_audit_fields(self) -> None:
        """Phase 3 integration: audit fields must exist."""
        from elspeth.plugins.results import TransformResult

        result = TransformResult.success({"x": 1})
        # These fields are set by the engine in Phase 3
        assert hasattr(result, "input_hash")
        assert hasattr(result, "output_hash")
        assert hasattr(result, "duration_ms")
        assert result.input_hash is None  # Not set yet


class TestGateResult:
    """Results from gate transforms."""

    def test_gate_result_with_continue(self) -> None:
        from elspeth.plugins.results import GateResult, RoutingAction

        result = GateResult(
            row={"value": 42},
            action=RoutingAction.continue_(),
        )
        assert result.row == {"value": 42}
        assert result.action.kind == "continue"

    def test_gate_result_with_route(self) -> None:
        from elspeth.plugins.results import GateResult, RoutingAction

        result = GateResult(
            row={"value": 42, "flagged": True},
            action=RoutingAction.route_to_sink("review", reason={"score": 0.9}),
        )
        assert result.action.kind == "route_to_sink"
        assert result.action.destinations == ("review",)

    def test_has_audit_fields(self) -> None:
        """Phase 3 integration: audit fields must exist."""
        from elspeth.plugins.results import GateResult, RoutingAction

        result = GateResult(
            row={"x": 1},
            action=RoutingAction.continue_(),
        )
        assert hasattr(result, "input_hash")
        assert hasattr(result, "output_hash")
        assert hasattr(result, "duration_ms")


class TestAcceptResult:
    """Results from aggregation accept()."""

    def test_accepted_no_trigger(self) -> None:
        from elspeth.plugins.results import AcceptResult

        result = AcceptResult(accepted=True, trigger=False)
        assert result.accepted is True
        assert result.trigger is False

    def test_accepted_with_trigger(self) -> None:
        from elspeth.plugins.results import AcceptResult

        result = AcceptResult(accepted=True, trigger=True)
        assert result.trigger is True

    def test_has_batch_id_field(self) -> None:
        """Phase 3 integration: batch_id for Landscape."""
        from elspeth.plugins.results import AcceptResult

        result = AcceptResult(accepted=True, trigger=False)
        assert hasattr(result, "batch_id")
        assert result.batch_id is None  # Set by engine in Phase 3
