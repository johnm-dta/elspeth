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

    def test_route(self) -> None:
        from elspeth.plugins.results import RoutingAction

        action = RoutingAction.route("suspicious", reason={"confidence": 0.95})
        assert action.kind == "route"
        assert action.destinations == (
            "suspicious",
        )  # Tuple - route label, not sink name
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

        action = RoutingAction.route("suspicious", reason={"score": 0.9})
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
            action=RoutingAction.route("suspicious", reason={"score": 0.9}),
        )
        assert result.action.kind == "route"
        assert result.action.destinations == (
            "suspicious",
        )  # Route label, not sink name

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


class TestRoutingActionEnums:
    """RoutingAction uses enum types for kind and mode."""

    def test_continue_uses_routing_kind_enum(self) -> None:
        """continue_() returns RoutingKind enum value."""
        from elspeth.contracts import RoutingKind
        from elspeth.plugins.results import RoutingAction

        action = RoutingAction.continue_()

        assert action.kind == RoutingKind.CONTINUE
        assert isinstance(action.kind, RoutingKind)

    def test_route_uses_enums(self) -> None:
        """route() uses enum types."""
        from elspeth.contracts import RoutingKind, RoutingMode
        from elspeth.plugins.results import RoutingAction

        action = RoutingAction.route("suspicious", mode=RoutingMode.COPY)

        assert action.kind == RoutingKind.ROUTE
        assert action.mode == RoutingMode.COPY
        assert isinstance(action.kind, RoutingKind)
        assert isinstance(action.mode, RoutingMode)

    def test_fork_to_paths_uses_enums(self) -> None:
        """fork_to_paths() uses enum types."""
        from elspeth.contracts import RoutingKind, RoutingMode
        from elspeth.plugins.results import RoutingAction

        action = RoutingAction.fork_to_paths(["path_a", "path_b"])

        assert action.kind == RoutingKind.FORK_TO_PATHS
        assert action.mode == RoutingMode.COPY


class TestFreezeDictDefensiveCopy:
    """_freeze_dict makes defensive copy to prevent mutation."""

    def test_original_dict_mutation_not_visible(self) -> None:
        """Mutating original dict doesn't affect frozen result."""
        from elspeth.plugins.results import RoutingAction

        reason = {"key": "original"}
        action = RoutingAction.continue_(reason=reason)

        # Mutate original
        reason["key"] = "mutated"
        reason["new_key"] = "added"

        # Frozen reason should be unchanged
        assert action.reason["key"] == "original"
        assert "new_key" not in action.reason

    def test_nested_dict_mutation_not_visible(self) -> None:
        """Nested dict mutation doesn't affect frozen result."""
        from elspeth.plugins.results import RoutingAction

        reason = {"nested": {"value": 1}}
        action = RoutingAction.continue_(reason=reason)

        # Mutate nested original
        reason["nested"]["value"] = 999

        # Frozen reason should be unchanged
        assert action.reason["nested"]["value"] == 1


class TestPluginsPublicAPI:
    """Public API exports from elspeth.plugins."""

    def test_results_importable(self) -> None:
        from elspeth.plugins import (
            AcceptResult,
            GateResult,
            RoutingAction,
            RowOutcome,
            TransformResult,
        )

        assert AcceptResult is not None
        assert GateResult is not None
        assert RoutingAction is not None
        assert RowOutcome is not None
        assert TransformResult is not None

    def test_context_importable(self) -> None:
        from elspeth.plugins import PluginContext

        assert PluginContext is not None

    def test_schemas_importable(self) -> None:
        from elspeth.plugins import PluginSchema, check_compatibility

        assert PluginSchema is not None
        assert check_compatibility is not None

    def test_protocols_importable(self) -> None:
        from elspeth.plugins import (
            AggregationProtocol,
            CoalescePolicy,
            CoalesceProtocol,
            GateProtocol,
            SinkProtocol,
            SourceProtocol,
            TransformProtocol,
        )

        assert AggregationProtocol is not None
        assert CoalescePolicy is not None
        assert CoalesceProtocol is not None
        assert GateProtocol is not None
        assert SinkProtocol is not None
        assert SourceProtocol is not None
        assert TransformProtocol is not None

    def test_base_classes_importable(self) -> None:
        from elspeth.plugins import (
            BaseAggregation,
            BaseGate,
            BaseSink,
            BaseSource,
            BaseTransform,
        )

        assert BaseAggregation is not None
        assert BaseGate is not None
        assert BaseSink is not None
        assert BaseSource is not None
        assert BaseTransform is not None

    def test_manager_importable(self) -> None:
        from elspeth.plugins import PluginManager

        assert PluginManager is not None

    def test_hookspecs_importable(self) -> None:
        from elspeth.plugins import hookimpl, hookspec

        assert hookspec is not None
        assert hookimpl is not None
