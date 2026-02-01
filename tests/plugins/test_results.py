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

        action = RoutingAction.route("suspicious", reason={"rule": "confidence_check", "matched_value": 0.95})
        assert action.kind == "route"
        assert action.destinations == ("suspicious",)  # Tuple - route label, not sink name
        assert action.reason is not None
        assert action.reason["matched_value"] == 0.95  # type: ignore[index,typeddict-item]  # Access via mapping

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
            action.kind = "route_to_sink"  # type: ignore[misc,assignment]  # Testing frozen

    def test_reason_mutation_protected_by_deep_copy(self) -> None:
        """Mutating original dict should not affect stored reason (deep copy)."""
        from typing import Any

        from elspeth.plugins.results import RoutingAction

        original: dict[str, Any] = {"rule": "score_check", "matched_value": 0.9}
        action = RoutingAction.route("suspicious", reason=original)  # type: ignore[arg-type]

        # Mutate original - should not affect action.reason (deep copy protection)
        original["matched_value"] = 0.5
        assert action.reason is not None
        assert action.reason["matched_value"] == 0.9  # type: ignore[typeddict-item]


class TestTransformResult:
    """Results from transform operations."""

    def test_success_result(self) -> None:
        from elspeth.plugins.results import TransformResult

        result = TransformResult.success({"value": 42}, success_reason={"action": "test"})
        assert result.status == "success"
        assert result.row == {"value": 42}
        assert result.retryable is False

    def test_error_result(self) -> None:
        from elspeth.plugins.results import TransformResult

        result = TransformResult.error(
            reason={"reason": "validation_failed"},
            retryable=True,
        )
        assert result.status == "error"
        assert result.row is None
        assert result.retryable is True

    def test_has_audit_fields(self) -> None:
        """Phase 3 integration: audit fields must exist."""
        from elspeth.plugins.results import TransformResult

        result = TransformResult.success({"x": 1}, success_reason={"action": "test"})
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
            action=RoutingAction.route("suspicious", reason={"rule": "score_check", "matched_value": 0.9}),
        )
        assert result.action.kind == "route"
        assert result.action.destinations == ("suspicious",)  # Route label, not sink name

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


class TestAcceptResultDeleted:
    """Verify AcceptResult was deleted in aggregation structural cleanup."""

    def test_accept_result_deleted_from_plugins_results(self) -> None:
        """AcceptResult should be deleted from plugins.results."""
        import elspeth.plugins.results as results

        assert not hasattr(results, "AcceptResult"), "AcceptResult should be deleted - aggregation is structural"

    def test_accept_result_not_exported_from_plugins(self) -> None:
        """AcceptResult should NOT be exported from elspeth.plugins."""
        import elspeth.plugins as plugins

        assert not hasattr(plugins, "AcceptResult"), "AcceptResult should not be exported - aggregation is structural"


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

        action = RoutingAction.route("suspicious", mode=RoutingMode.MOVE)

        assert action.kind == RoutingKind.ROUTE
        assert action.mode == RoutingMode.MOVE
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
        from typing import Any

        from elspeth.plugins.results import RoutingAction

        reason: dict[str, Any] = {"rule": "original_rule", "matched_value": "original_value"}
        action = RoutingAction.continue_(reason=reason)  # type: ignore[arg-type]

        # Mutate original
        reason["rule"] = "mutated"
        reason["new_key"] = "added"

        # Frozen reason should be unchanged
        assert action.reason is not None
        assert action.reason["rule"] == "original_rule"  # type: ignore[index,typeddict-item]
        assert "new_key" not in action.reason  # type: ignore[operator]

    def test_nested_dict_mutation_not_visible(self) -> None:
        """Nested dict mutation doesn't affect frozen result."""
        from typing import Any

        from elspeth.plugins.results import RoutingAction

        reason: dict[str, Any] = {"rule": "nested_test", "matched_value": {"value": 1}}
        action = RoutingAction.continue_(reason=reason)  # type: ignore[arg-type]

        # Mutate nested original
        reason["matched_value"]["value"] = 999

        # Frozen reason should be unchanged
        assert action.reason is not None
        assert action.reason["matched_value"]["value"] == 1  # type: ignore[index,typeddict-item]


class TestSourceRow:
    """Results from source loading - valid or quarantined."""

    def test_quarantined_factory(self) -> None:
        """quarantined() creates a quarantined row with error info."""
        from elspeth.plugins.results import SourceRow

        result = SourceRow.quarantined(
            row={"id": 1, "value": "bad"},
            error="validation failed: value must be int",
            destination="quarantine_sink",
        )
        assert result.is_quarantined is True
        assert result.row == {"id": 1, "value": "bad"}
        assert result.quarantine_error == "validation failed: value must be int"
        assert result.quarantine_destination == "quarantine_sink"

    def test_quarantined_preserves_original_row(self) -> None:
        """Quarantined rows preserve the original (invalid) data."""
        from elspeth.plugins.results import SourceRow

        original = {"score": "not-a-number", "name": "test"}
        result = SourceRow.quarantined(
            row=original,
            error="score must be int",
            destination="bad_data",
        )
        # Original value preserved for audit/debugging
        assert result.row["score"] == "not-a-number"

    def test_is_dataclass(self) -> None:
        """SourceRow is a dataclass."""
        from dataclasses import is_dataclass

        from elspeth.plugins.results import SourceRow

        assert is_dataclass(SourceRow)

    def test_importable_from_contracts(self) -> None:
        """SourceRow is exported from elspeth.contracts."""
        from elspeth.contracts import SourceRow

        assert SourceRow is not None


class TestPluginsPublicAPI:
    """Public API exports from elspeth.plugins."""

    def test_results_importable(self) -> None:
        from elspeth.plugins import (
            GateResult,
            RoutingAction,
            RowOutcome,
            SourceRow,
            TransformResult,
        )

        # NOTE: AcceptResult deleted in aggregation structural cleanup
        assert GateResult is not None
        assert RoutingAction is not None
        assert RowOutcome is not None
        assert SourceRow is not None
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
            CoalescePolicy,
            CoalesceProtocol,
            GateProtocol,
            SinkProtocol,
            SourceProtocol,
            TransformProtocol,
        )

        # NOTE: AggregationProtocol deleted in aggregation structural cleanup
        assert CoalescePolicy is not None
        assert CoalesceProtocol is not None
        assert GateProtocol is not None
        assert SinkProtocol is not None
        assert SourceProtocol is not None
        assert TransformProtocol is not None

    def test_aggregation_protocol_not_exported(self) -> None:
        """AggregationProtocol should NOT be exported (aggregation is structural)."""
        import elspeth.plugins as plugins

        assert not hasattr(plugins, "AggregationProtocol"), "AggregationProtocol should be deleted"

    def test_base_classes_importable(self) -> None:
        from elspeth.plugins import (
            BaseGate,
            BaseSink,
            BaseSource,
            BaseTransform,
        )

        # NOTE: BaseAggregation deleted in aggregation structural cleanup
        assert BaseGate is not None
        assert BaseSink is not None
        assert BaseSource is not None
        assert BaseTransform is not None

    def test_base_aggregation_not_exported(self) -> None:
        """BaseAggregation should NOT be exported (aggregation is structural)."""
        import elspeth.plugins as plugins

        assert not hasattr(plugins, "BaseAggregation"), "BaseAggregation should be deleted"

    def test_manager_importable(self) -> None:
        from elspeth.plugins import PluginManager

        assert PluginManager is not None

    def test_hookspecs_importable(self) -> None:
        from elspeth.plugins import hookimpl, hookspec

        assert hookspec is not None
        assert hookimpl is not None
