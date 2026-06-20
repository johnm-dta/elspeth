# tests/plugins/test_results.py
"""Tests for plugin result types."""

from dataclasses import FrozenInstanceError

import pytest

from elspeth.testing import make_pipeline_row


class TestTerminalPairEnums:
    """ADR-019 terminal pair enums are part of the plugin result API."""

    def test_all_terminal_states_exist(self) -> None:
        from elspeth.plugins.infrastructure.results import TerminalOutcome, TerminalPath

        assert TerminalOutcome.SUCCESS.value == "success"
        assert TerminalOutcome.FAILURE.value == "failure"
        assert TerminalOutcome.TRANSIENT.value == "transient"
        assert TerminalPath.DEFAULT_FLOW.value == "default_flow"
        assert TerminalPath.GATE_ROUTED.value == "gate_routed"
        assert TerminalPath.BATCH_CONSUMED.value == "batch_consumed"
        assert TerminalPath.BUFFERED.value == "buffered"

    def test_outcome_is_enum(self) -> None:
        from enum import Enum

        from elspeth.plugins.infrastructure.results import TerminalOutcome, TerminalPath

        assert issubclass(TerminalOutcome, Enum)
        assert issubclass(TerminalPath, Enum)


class TestRoutingAction:
    """Routing decisions from gates."""

    def test_continue_action(self) -> None:
        from elspeth.plugins.infrastructure.results import RoutingAction

        action = RoutingAction.continue_()
        assert action.kind == "continue"
        assert action.destinations == ()  # Tuple, not list
        assert action.mode == "move"

    def test_route(self) -> None:
        from elspeth.contracts.errors import ConfigGateReason
        from elspeth.plugins.infrastructure.results import RoutingAction

        reason = ConfigGateReason(condition="confidence_check", result="suspicious")
        action = RoutingAction.route("suspicious", reason=reason)
        assert action.kind == "route"
        assert action.destinations == ("suspicious",)  # Tuple - route label, not sink name
        assert action.reason is not None
        assert action.reason["condition"] == "confidence_check"  # type: ignore[typeddict-item]

    def test_fork_to_paths(self) -> None:
        from elspeth.plugins.infrastructure.results import RoutingAction

        action = RoutingAction.fork_to_paths(["stats", "classifier", "archive"])
        assert action.kind == "fork_to_paths"
        assert action.destinations == ("stats", "classifier", "archive")  # Tuple
        assert action.mode == "copy"

    def test_immutable(self) -> None:
        from elspeth.plugins.infrastructure.results import RoutingAction

        action = RoutingAction.continue_()
        with pytest.raises(FrozenInstanceError):
            action.kind = "route_to_sink"  # type: ignore[misc,assignment]  # Testing frozen

    def test_reason_mutation_protected_by_deep_copy(self) -> None:
        """Mutating original dict should not affect stored reason (deep copy)."""
        from typing import Any

        from elspeth.plugins.infrastructure.results import RoutingAction

        original: dict[str, Any] = {"rule": "score_check", "matched_value": 0.9}
        action = RoutingAction.route("suspicious", reason=original)  # type: ignore[arg-type]

        # Mutate original - should not affect action.reason (deep copy protection)
        original["matched_value"] = 0.5
        assert action.reason is not None
        assert action.reason["matched_value"] == 0.9  # type: ignore[typeddict-item]


class TestTransformResult:
    """Results from transform operations."""

    def test_success_result(self) -> None:
        from elspeth.plugins.infrastructure.results import TransformResult

        result = TransformResult.success(make_pipeline_row({"value": 42}), success_reason={"action": "test"})
        assert result.status == "success"
        assert result.row is not None
        assert result.row.to_dict() == {"value": 42}
        assert result.retryable is False

    def test_error_result(self) -> None:
        from elspeth.plugins.infrastructure.results import TransformResult

        result = TransformResult.error(
            reason={"reason": "validation_failed"},
            retryable=True,
        )
        assert result.status == "error"
        assert result.row is None
        assert result.retryable is True

    def test_has_audit_fields(self) -> None:
        """Phase 3 integration: audit fields must exist."""
        from elspeth.plugins.infrastructure.results import TransformResult

        result = TransformResult.success(make_pipeline_row({"x": 1}), success_reason={"action": "test"})
        # These fields are set by the engine in Phase 3
        assert hasattr(result, "input_hash")
        assert hasattr(result, "output_hash")
        assert hasattr(result, "duration_ms")
        assert result.input_hash is None  # Not set yet


class TestGateResult:
    """Results from config-driven gates (not plugins — gates are engine-owned)."""

    def test_gate_result_with_continue(self) -> None:
        from elspeth.contracts import GateResult, RoutingAction

        result = GateResult(
            row={"value": 42},
            action=RoutingAction.continue_(),
        )
        assert result.row == {"value": 42}
        assert result.action.kind == "continue"

    def test_gate_result_with_route(self) -> None:
        from elspeth.contracts import GateResult, RoutingAction
        from elspeth.contracts.errors import ConfigGateReason

        reason = ConfigGateReason(condition="score_check", result="suspicious")
        result = GateResult(
            row={"value": 42, "flagged": True},
            action=RoutingAction.route("suspicious", reason=reason),
        )
        assert result.action.kind == "route"
        assert result.action.destinations == ("suspicious",)  # Route label, not sink name

    def test_has_audit_fields(self) -> None:
        """Phase 3 integration: audit fields must exist."""
        from elspeth.contracts import GateResult, RoutingAction

        result = GateResult(
            row={"x": 1},
            action=RoutingAction.continue_(),
        )
        assert hasattr(result, "input_hash")
        assert hasattr(result, "output_hash")
        assert hasattr(result, "duration_ms")


class TestAcceptResultDeleted:
    """Guard against AcceptResult reintroduction.

    AcceptResult was removed as part of the aggregation structural cleanup.
    These tests exist per the no-legacy-code policy: if someone accidentally
    re-adds AcceptResult, these tests will fail and surface the violation.
    """

    def test_accept_result_deleted_from_plugins_results(self) -> None:
        """AcceptResult must not exist in plugins.infrastructure.results."""
        import elspeth.plugins.infrastructure.results as results

        assert "AcceptResult" not in dir(results), "AcceptResult should be deleted - aggregation is structural"

    def test_accept_result_not_exported_from_plugins(self) -> None:
        """AcceptResult must not be exported from elspeth.plugins."""
        import elspeth.plugins as plugins

        assert "AcceptResult" not in dir(plugins), "AcceptResult should not be exported - aggregation is structural"


class TestGateResultNotInPluginAPI:
    """Guard against GateResult reintroduction to plugin public API.

    GateResult was removed from plugins.infrastructure.results because gates
    are config-driven engine operations, not plugins. GateResult lives in
    elspeth.contracts and engine code imports it directly from there.
    """

    def test_gate_result_not_in_plugin_results_all(self) -> None:
        """GateResult must not be in plugins.infrastructure.results.__all__."""
        import elspeth.plugins.infrastructure.results as results

        assert "GateResult" not in results.__all__, "GateResult should not be in plugin public API — gates are not plugins"

    def test_gate_result_importable_from_contracts(self) -> None:
        """GateResult must be importable from elspeth.contracts."""
        from elspeth.contracts import GateResult

        assert GateResult is not None


class TestRoutingActionEnums:
    """RoutingAction uses enum types for kind and mode."""

    def test_continue_uses_routing_kind_enum(self) -> None:
        """continue_() returns RoutingKind enum value."""
        from elspeth.contracts import RoutingKind
        from elspeth.plugins.infrastructure.results import RoutingAction

        action = RoutingAction.continue_()

        assert action.kind == RoutingKind.CONTINUE
        assert isinstance(action.kind, RoutingKind)

    def test_route_uses_enums(self) -> None:
        """route() uses enum types."""
        from elspeth.contracts import RoutingKind, RoutingMode
        from elspeth.plugins.infrastructure.results import RoutingAction

        action = RoutingAction.route("suspicious", mode=RoutingMode.MOVE)

        assert action.kind == RoutingKind.ROUTE
        assert action.mode == RoutingMode.MOVE
        assert isinstance(action.kind, RoutingKind)
        assert isinstance(action.mode, RoutingMode)

    def test_fork_to_paths_uses_enums(self) -> None:
        """fork_to_paths() uses enum types."""
        from elspeth.contracts import RoutingKind, RoutingMode
        from elspeth.plugins.infrastructure.results import RoutingAction

        action = RoutingAction.fork_to_paths(["path_a", "path_b"])

        assert action.kind == RoutingKind.FORK_TO_PATHS
        assert action.mode == RoutingMode.COPY


class TestFreezeDictDefensiveCopy:
    """_freeze_dict makes defensive copy to prevent mutation."""

    def test_original_dict_mutation_not_visible(self) -> None:
        """Mutating original dict doesn't affect frozen result."""
        from typing import Any

        from elspeth.plugins.infrastructure.results import RoutingAction

        reason: dict[str, Any] = {"rule": "original_rule", "matched_value": "original_value"}
        action = RoutingAction.continue_(reason=reason)  # type: ignore[arg-type]

        # Mutate original
        reason["rule"] = "mutated"
        reason["new_key"] = "added"

        # Frozen reason should be unchanged
        assert action.reason is not None
        assert action.reason["rule"] == "original_rule"  # type: ignore[typeddict-item]
        assert "new_key" not in action.reason

    def test_nested_dict_mutation_not_visible(self) -> None:
        """Nested dict mutation doesn't affect frozen result."""
        from typing import Any

        from elspeth.plugins.infrastructure.results import RoutingAction

        reason: dict[str, Any] = {"rule": "nested_test", "matched_value": {"value": 1}}
        action = RoutingAction.continue_(reason=reason)  # type: ignore[arg-type]

        # Mutate nested original
        reason["matched_value"]["value"] = 999

        # Frozen reason should be unchanged
        assert action.reason is not None
        assert action.reason["matched_value"]["value"] == 1  # type: ignore[typeddict-item]


class TestSourceRow:
    """Results from source loading - valid or quarantined."""

    def test_quarantined_factory(self) -> None:
        """quarantined() creates a quarantined row with error info."""
        from elspeth.plugins.infrastructure.results import SourceRow

        result = SourceRow.quarantined(
            row={"id": 1, "value": "bad"},
            error="validation failed: value must be int",
            destination="quarantine_sink",
            source_row_index=0,
        )
        assert result.is_quarantined is True
        assert result.row == {"id": 1, "value": "bad"}
        assert result.quarantine_error == "validation failed: value must be int"
        assert result.quarantine_destination == "quarantine_sink"

    def test_quarantined_preserves_original_row(self) -> None:
        """Quarantined rows preserve the original (invalid) data."""
        from elspeth.plugins.infrastructure.results import SourceRow

        original = {"score": "not-a-number", "name": "test"}
        result = SourceRow.quarantined(
            row=original,
            error="score must be int",
            destination="bad_data",
            source_row_index=0,
        )
        # Original value preserved for audit/debugging
        assert result.row["score"] == "not-a-number"

    def test_is_dataclass(self) -> None:
        """SourceRow is a dataclass."""
        from dataclasses import is_dataclass

        from elspeth.plugins.infrastructure.results import SourceRow

        assert is_dataclass(SourceRow)

    def test_importable_from_contracts(self) -> None:
        """SourceRow is exported from elspeth.contracts."""
        from elspeth.contracts import SourceRow

        assert SourceRow is not None

    def test_quarantined_without_error_raises(self) -> None:
        """Quarantined row without error message violates invariant."""
        from elspeth.contracts import SourceRow

        with pytest.raises(ValueError, match="quarantine_error"):
            SourceRow(row={"x": 1}, is_quarantined=True, quarantine_destination="bad_sink", source_row_index=0)

    def test_quarantined_without_destination_raises(self) -> None:
        """Quarantined row without destination violates invariant."""
        from elspeth.contracts import SourceRow

        with pytest.raises(ValueError, match="quarantine_destination"):
            SourceRow(row={"x": 1}, is_quarantined=True, quarantine_error="bad data", source_row_index=0)

    def test_non_quarantined_with_error_raises(self) -> None:
        """Non-quarantined row with quarantine_error set violates invariant."""
        from elspeth.contracts import SourceRow

        with pytest.raises(ValueError, match="quarantine_error"):
            SourceRow(row={"x": 1}, is_quarantined=False, quarantine_error="stale error", source_row_index=0)

    def test_non_quarantined_with_destination_raises(self) -> None:
        """Non-quarantined row with quarantine_destination set violates invariant."""
        from elspeth.contracts import SourceRow

        with pytest.raises(ValueError, match="quarantine_destination"):
            SourceRow(row={"x": 1}, is_quarantined=False, quarantine_destination="stale_sink", source_row_index=0)

    def test_valid_factory_passes_post_init(self) -> None:
        """SourceRow.valid() produces a row that passes __post_init__ validation."""
        from elspeth.contracts import SourceRow
        from elspeth.testing import make_contract

        row = SourceRow.valid({"a": 1}, contract=make_contract({"a": 1}), source_row_index=0)
        assert not row.is_quarantined
        assert row.quarantine_error is None
        assert row.quarantine_destination is None

    def test_quarantined_factory_passes_post_init(self) -> None:
        """SourceRow.quarantined() produces a row that passes __post_init__ validation."""
        from elspeth.contracts import SourceRow

        row = SourceRow.quarantined(row={"a": "bad"}, error="bad val", destination="quarantine", source_row_index=0)
        assert row.is_quarantined
        assert row.quarantine_error == "bad val"
        assert row.quarantine_destination == "quarantine"


class TestPluginsPublicAPI:
    """Public API exports from elspeth.plugins."""

    def test_results_importable(self) -> None:
        from elspeth.plugins.infrastructure.results import (
            RoutingAction,
            SourceRow,
            TerminalOutcome,
            TerminalPath,
            TransformResult,
        )

        # NOTE: AcceptResult deleted in aggregation structural cleanup
        # NOTE: GateResult removed — gates are config-driven, not plugins
        assert RoutingAction is not None
        assert SourceRow is not None
        assert TerminalOutcome is not None
        assert TerminalPath is not None
        assert TransformResult is not None

    def test_context_importable(self) -> None:
        from elspeth.contracts.plugin_context import PluginContext

        assert PluginContext is not None

    def test_schemas_importable(self) -> None:
        from elspeth.contracts import PluginSchema, check_compatibility

        assert PluginSchema is not None
        assert check_compatibility is not None

    def test_protocols_importable(self) -> None:
        from elspeth.contracts import (
            SinkProtocol,
            SourceProtocol,
            TransformProtocol,
        )

        assert SinkProtocol is not None
        assert SourceProtocol is not None
        assert TransformProtocol is not None

    def test_deleted_protocols_not_exported(self) -> None:
        """Deleted protocols should NOT be exported."""
        import elspeth.plugins as plugins

        assert not hasattr(plugins, "AggregationProtocol"), "AggregationProtocol should be deleted"
        assert not hasattr(plugins, "CoalesceProtocol"), "CoalesceProtocol should be deleted"
        assert not hasattr(plugins, "CoalescePolicy"), "CoalescePolicy should be deleted"

    def test_base_classes_importable(self) -> None:
        from elspeth.plugins.infrastructure.base import (
            BaseSink,
            BaseSource,
            BaseTransform,
        )

        assert BaseSink is not None
        assert BaseSource is not None
        assert BaseTransform is not None

    def test_base_aggregation_not_exported(self) -> None:
        """BaseAggregation should NOT be exported (aggregation is structural)."""
        import elspeth.plugins as plugins

        assert not hasattr(plugins, "BaseAggregation"), "BaseAggregation should be deleted"

    def test_manager_importable(self) -> None:
        from elspeth.plugins.infrastructure.manager import PluginManager

        assert PluginManager is not None

    def test_hookspecs_importable(self) -> None:
        from elspeth.plugins.infrastructure.hookspecs import hookimpl, hookspec

        assert hookspec is not None
        assert hookimpl is not None
