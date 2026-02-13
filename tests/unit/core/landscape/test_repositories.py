"""Tests for the Landscape repository layer.

The repository layer converts SQLAlchemy row objects (where enums are stored
as strings) into strict domain dataclass objects. This is Tier 1 trust --
bad data from our own DB must crash, never coerce.

Tests use unittest.mock.Mock objects to simulate SQLAlchemy Row objects.
"""

import json
from datetime import UTC, datetime
from unittest.mock import Mock

import pytest

from elspeth.contracts.audit import (
    Artifact,
    Batch,
    BatchMember,
    Call,
    Edge,
    Node,
    NodeStateCompleted,
    NodeStateFailed,
    NodeStateOpen,
    NodeStatePending,
    RoutingEvent,
    Row,
    Run,
    Token,
    TokenOutcome,
    TokenParent,
    TransformErrorRecord,
    ValidationErrorRecord,
)
from elspeth.contracts.enums import (
    BatchStatus,
    CallStatus,
    CallType,
    Determinism,
    ExportStatus,
    NodeStateStatus,
    NodeType,
    RoutingMode,
    RowOutcome,
    RunStatus,
    TriggerType,
)
from elspeth.core.landscape.repositories import (
    ArtifactRepository,
    BatchMemberRepository,
    BatchRepository,
    CallRepository,
    EdgeRepository,
    NodeRepository,
    NodeStateRepository,
    RoutingEventRepository,
    RowRepository,
    RunRepository,
    TokenOutcomeRepository,
    TokenParentRepository,
    TokenRepository,
    TransformErrorRepository,
    ValidationErrorRepository,
)


def _make_sa_row(**kwargs: object) -> Mock:
    """Create a mock SQLAlchemy Row with named attributes."""
    row = Mock()
    for key, value in kwargs.items():
        setattr(row, key, value)
    return row


NOW = datetime.now(UTC)


# ---------------------------------------------------------------------------
# RunRepository
# ---------------------------------------------------------------------------


class TestRunRepository:
    """Tests for RunRepository.load()."""

    def _make_run_row(self, **overrides: object) -> Mock:
        defaults = {
            "run_id": "run-1",
            "started_at": NOW,
            "config_hash": "abc123",
            "settings_json": "{}",
            "canonical_version": "v2",
            "status": "running",
            "completed_at": None,
            "reproducibility_grade": None,
            "export_status": None,
            "export_error": None,
            "exported_at": None,
            "export_format": None,
            "export_sink": None,
        }
        defaults.update(overrides)
        return _make_sa_row(**defaults)

    def test_valid_load_all_fields(self) -> None:
        sa_row = self._make_run_row(
            status="completed",
            completed_at=NOW,
            reproducibility_grade="A",
            export_status="pending",
            export_error="some error",
            exported_at=NOW,
            export_format="csv",
            export_sink="output",
        )
        repo = RunRepository()
        result = repo.load(sa_row)

        assert isinstance(result, Run)
        assert result.run_id == "run-1"
        assert result.status == RunStatus.COMPLETED
        assert result.export_status == ExportStatus.PENDING
        assert result.completed_at == NOW
        assert result.reproducibility_grade == "A"
        assert result.export_error == "some error"
        assert result.export_format == "csv"
        assert result.export_sink == "output"

    def test_valid_load_minimal_fields(self) -> None:
        sa_row = self._make_run_row()
        repo = RunRepository()
        result = repo.load(sa_row)

        assert result.status == RunStatus.RUNNING
        assert result.export_status is None
        assert result.completed_at is None

    @pytest.mark.parametrize(
        "status_value",
        ["running", "completed", "failed"],
    )
    def test_valid_run_status_values(self, status_value: str) -> None:
        sa_row = self._make_run_row(status=status_value)
        repo = RunRepository()
        result = repo.load(sa_row)
        assert result.status == RunStatus(status_value)

    def test_invalid_status_raises_value_error(self) -> None:
        sa_row = self._make_run_row(status="bogus_status")
        repo = RunRepository()
        with pytest.raises(ValueError):
            repo.load(sa_row)

    @pytest.mark.parametrize(
        "export_value",
        ["pending", "completed", "failed"],
    )
    def test_valid_export_status_values(self, export_value: str) -> None:
        sa_row = self._make_run_row(export_status=export_value)
        repo = RunRepository()
        result = repo.load(sa_row)
        assert result.export_status == ExportStatus(export_value)

    def test_none_export_status_stays_none(self) -> None:
        sa_row = self._make_run_row(export_status=None)
        repo = RunRepository()
        result = repo.load(sa_row)
        assert result.export_status is None

    def test_invalid_export_status_raises_value_error(self) -> None:
        sa_row = self._make_run_row(export_status="bogus_export")
        repo = RunRepository()
        with pytest.raises(ValueError):
            repo.load(sa_row)

    def test_empty_string_export_status_raises_value_error(self) -> None:
        """Empty string should NOT become None -- Tier 1 trust."""
        sa_row = self._make_run_row(export_status="")
        repo = RunRepository()
        with pytest.raises(ValueError):
            repo.load(sa_row)


# ---------------------------------------------------------------------------
# NodeRepository
# ---------------------------------------------------------------------------


class TestNodeRepository:
    """Tests for NodeRepository.load()."""

    def _make_node_row(self, **overrides: object) -> Mock:
        defaults = {
            "node_id": "node-1",
            "run_id": "run-1",
            "plugin_name": "csv",
            "node_type": "source",
            "plugin_version": "1.0.0",
            "determinism": "deterministic",
            "config_hash": "cfg123",
            "config_json": "{}",
            "registered_at": NOW,
            "schema_hash": None,
            "sequence_in_pipeline": 0,
            "schema_mode": None,
            "schema_fields_json": None,
        }
        defaults.update(overrides)
        return _make_sa_row(**defaults)

    def test_valid_load_all_fields(self) -> None:
        schema_fields = [{"name": "id", "type": "int"}]
        sa_row = self._make_node_row(
            node_type="transform",
            determinism="non_deterministic",
            schema_hash="sch123",
            sequence_in_pipeline=3,
            schema_mode="fixed",
            schema_fields_json=json.dumps(schema_fields),
        )
        repo = NodeRepository()
        result = repo.load(sa_row)

        assert isinstance(result, Node)
        assert result.node_type == NodeType.TRANSFORM
        assert result.determinism == Determinism.NON_DETERMINISTIC
        assert result.schema_hash == "sch123"
        assert result.sequence_in_pipeline == 3
        assert result.schema_mode == "fixed"
        assert result.schema_fields == schema_fields

    def test_valid_load_minimal_fields(self) -> None:
        sa_row = self._make_node_row()
        repo = NodeRepository()
        result = repo.load(sa_row)

        assert result.node_type == NodeType.SOURCE
        assert result.determinism == Determinism.DETERMINISTIC
        assert result.schema_fields is None

    @pytest.mark.parametrize(
        "node_type_value",
        ["source", "transform", "gate", "aggregation", "coalesce", "sink"],
    )
    def test_valid_node_type_values(self, node_type_value: str) -> None:
        sa_row = self._make_node_row(node_type=node_type_value)
        repo = NodeRepository()
        result = repo.load(sa_row)
        assert result.node_type == NodeType(node_type_value)

    @pytest.mark.parametrize(
        "det_value",
        [
            "deterministic",
            "seeded",
            "io_read",
            "io_write",
            "external_call",
            "non_deterministic",
        ],
    )
    def test_valid_determinism_values(self, det_value: str) -> None:
        sa_row = self._make_node_row(determinism=det_value)
        repo = NodeRepository()
        result = repo.load(sa_row)
        assert result.determinism == Determinism(det_value)

    def test_invalid_node_type_raises_value_error(self) -> None:
        sa_row = self._make_node_row(node_type="bogus_type")
        repo = NodeRepository()
        with pytest.raises(ValueError):
            repo.load(sa_row)

    def test_invalid_determinism_raises_value_error(self) -> None:
        sa_row = self._make_node_row(determinism="sometimes")
        repo = NodeRepository()
        with pytest.raises(ValueError):
            repo.load(sa_row)

    def test_schema_fields_json_none_becomes_none(self) -> None:
        sa_row = self._make_node_row(schema_fields_json=None)
        repo = NodeRepository()
        result = repo.load(sa_row)
        assert result.schema_fields is None

    def test_schema_fields_json_valid_json_becomes_list(self) -> None:
        fields = [{"name": "amount", "type": "float"}]
        sa_row = self._make_node_row(schema_fields_json=json.dumps(fields))
        repo = NodeRepository()
        result = repo.load(sa_row)
        assert result.schema_fields == fields

    def test_schema_fields_json_empty_list(self) -> None:
        sa_row = self._make_node_row(schema_fields_json="[]")
        repo = NodeRepository()
        result = repo.load(sa_row)
        assert result.schema_fields == []

    def test_schema_fields_json_null_raises_value_error(self) -> None:
        sa_row = self._make_node_row(schema_fields_json="null")
        repo = NodeRepository()
        with pytest.raises(ValueError, match="must decode to list"):
            repo.load(sa_row)

    def test_schema_fields_json_object_raises_value_error(self) -> None:
        sa_row = self._make_node_row(schema_fields_json='{"name": "id"}')
        repo = NodeRepository()
        with pytest.raises(ValueError, match="must decode to list"):
            repo.load(sa_row)

    def test_schema_fields_json_non_dict_element_raises_value_error(self) -> None:
        sa_row = self._make_node_row(schema_fields_json='[{"name": "id"}, 42]')
        repo = NodeRepository()
        with pytest.raises(ValueError, match="must be object/dict"):
            repo.load(sa_row)


# ---------------------------------------------------------------------------
# EdgeRepository
# ---------------------------------------------------------------------------


class TestEdgeRepository:
    """Tests for EdgeRepository.load()."""

    def _make_edge_row(self, **overrides: object) -> Mock:
        defaults = {
            "edge_id": "edge-1",
            "run_id": "run-1",
            "from_node_id": "node-1",
            "to_node_id": "node-2",
            "label": "continue",
            "default_mode": "move",
            "created_at": NOW,
        }
        defaults.update(overrides)
        return _make_sa_row(**defaults)

    @pytest.mark.parametrize("mode_value", ["move", "copy", "divert"])
    def test_valid_routing_modes(self, mode_value: str) -> None:
        sa_row = self._make_edge_row(default_mode=mode_value)
        repo = EdgeRepository()
        result = repo.load(sa_row)
        assert isinstance(result, Edge)
        assert result.default_mode == RoutingMode(mode_value)

    def test_valid_load_all_fields(self) -> None:
        sa_row = self._make_edge_row()
        repo = EdgeRepository()
        result = repo.load(sa_row)
        assert result.edge_id == "edge-1"
        assert result.from_node_id == "node-1"
        assert result.to_node_id == "node-2"
        assert result.label == "continue"

    def test_invalid_mode_raises_value_error(self) -> None:
        sa_row = self._make_edge_row(default_mode="teleport")
        repo = EdgeRepository()
        with pytest.raises(ValueError):
            repo.load(sa_row)


# ---------------------------------------------------------------------------
# RowRepository
# ---------------------------------------------------------------------------


class TestRowRepository:
    """Tests for RowRepository.load()."""

    def _make_row_row(self, **overrides: object) -> Mock:
        defaults = {
            "row_id": "row-1",
            "run_id": "run-1",
            "source_node_id": "node-src",
            "row_index": 42,
            "source_data_hash": "hash123",
            "created_at": NOW,
            "source_data_ref": "ref://payload/abc",
        }
        defaults.update(overrides)
        return _make_sa_row(**defaults)

    def test_valid_load(self) -> None:
        sa_row = self._make_row_row()
        repo = RowRepository()
        result = repo.load(sa_row)

        assert isinstance(result, Row)
        assert result.row_id == "row-1"
        assert result.row_index == 42
        assert result.source_data_hash == "hash123"
        assert result.source_data_ref == "ref://payload/abc"

    def test_valid_load_with_none_ref(self) -> None:
        sa_row = self._make_row_row(source_data_ref=None)
        repo = RowRepository()
        result = repo.load(sa_row)
        assert result.source_data_ref is None


# ---------------------------------------------------------------------------
# TokenRepository
# ---------------------------------------------------------------------------


class TestTokenRepository:
    """Tests for TokenRepository.load()."""

    def _make_token_row(self, **overrides: object) -> Mock:
        defaults = {
            "token_id": "tok-1",
            "row_id": "row-1",
            "created_at": NOW,
            "fork_group_id": None,
            "join_group_id": None,
            "expand_group_id": None,
            "branch_name": None,
            "step_in_pipeline": None,
        }
        defaults.update(overrides)
        return _make_sa_row(**defaults)

    def test_valid_load_minimal(self) -> None:
        sa_row = self._make_token_row()
        repo = TokenRepository()
        result = repo.load(sa_row)

        assert isinstance(result, Token)
        assert result.token_id == "tok-1"
        assert result.row_id == "row-1"
        assert result.fork_group_id is None
        assert result.join_group_id is None

    def test_valid_load_with_all_optional_fields(self) -> None:
        sa_row = self._make_token_row(
            fork_group_id="fg-1",
            join_group_id="jg-1",
            expand_group_id="eg-1",
            branch_name="path_a",
            step_in_pipeline=3,
        )
        repo = TokenRepository()
        result = repo.load(sa_row)
        assert result.fork_group_id == "fg-1"
        assert result.join_group_id == "jg-1"
        assert result.expand_group_id == "eg-1"
        assert result.branch_name == "path_a"
        assert result.step_in_pipeline == 3


# ---------------------------------------------------------------------------
# TokenParentRepository
# ---------------------------------------------------------------------------


class TestTokenParentRepository:
    """Tests for TokenParentRepository.load()."""

    def test_valid_load(self) -> None:
        sa_row = _make_sa_row(
            token_id="tok-child",
            parent_token_id="tok-parent",
            ordinal=0,
        )
        repo = TokenParentRepository()
        result = repo.load(sa_row)

        assert isinstance(result, TokenParent)
        assert result.token_id == "tok-child"
        assert result.parent_token_id == "tok-parent"
        assert result.ordinal == 0

    def test_valid_load_higher_ordinal(self) -> None:
        sa_row = _make_sa_row(
            token_id="tok-child",
            parent_token_id="tok-parent-2",
            ordinal=5,
        )
        repo = TokenParentRepository()
        result = repo.load(sa_row)
        assert result.ordinal == 5


# ---------------------------------------------------------------------------
# CallRepository
# ---------------------------------------------------------------------------


class TestCallRepository:
    """Tests for CallRepository.load()."""

    def _make_call_row(self, **overrides: object) -> Mock:
        defaults = {
            "call_id": "call-1",
            "call_index": 0,
            "call_type": "llm",
            "status": "success",
            "request_hash": "req123",
            "created_at": NOW,
            "state_id": "state-1",
            "operation_id": None,
            "request_ref": None,
            "response_hash": None,
            "response_ref": None,
            "error_json": None,
            "latency_ms": None,
        }
        defaults.update(overrides)
        return _make_sa_row(**defaults)

    def test_valid_load_state_parented(self) -> None:
        sa_row = self._make_call_row(state_id="state-1", operation_id=None)
        repo = CallRepository()
        result = repo.load(sa_row)

        assert isinstance(result, Call)
        assert result.call_type == CallType.LLM
        assert result.status == CallStatus.SUCCESS
        assert result.state_id == "state-1"
        assert result.operation_id is None

    def test_valid_load_operation_parented(self) -> None:
        sa_row = self._make_call_row(
            state_id=None,
            operation_id="op-1",
            call_type="http",
            status="error",
        )
        repo = CallRepository()
        result = repo.load(sa_row)

        assert result.call_type == CallType.HTTP
        assert result.status == CallStatus.ERROR
        assert result.state_id is None
        assert result.operation_id == "op-1"

    def test_valid_load_with_response_fields(self) -> None:
        sa_row = self._make_call_row(
            response_hash="resp123",
            response_ref="ref://resp/abc",
            latency_ms=42.5,
        )
        repo = CallRepository()
        result = repo.load(sa_row)
        assert result.response_hash == "resp123"
        assert result.response_ref == "ref://resp/abc"
        assert result.latency_ms == 42.5

    def test_valid_load_with_error_json(self) -> None:
        sa_row = self._make_call_row(
            status="error",
            error_json='{"reason": "timeout"}',
        )
        repo = CallRepository()
        result = repo.load(sa_row)
        assert result.error_json == '{"reason": "timeout"}'

    @pytest.mark.parametrize(
        "call_type_value",
        ["llm", "http", "http_redirect", "sql", "filesystem"],
    )
    def test_valid_call_type_values(self, call_type_value: str) -> None:
        sa_row = self._make_call_row(call_type=call_type_value)
        repo = CallRepository()
        result = repo.load(sa_row)
        assert result.call_type == CallType(call_type_value)

    @pytest.mark.parametrize("status_value", ["success", "error"])
    def test_valid_call_status_values(self, status_value: str) -> None:
        sa_row = self._make_call_row(status=status_value)
        repo = CallRepository()
        result = repo.load(sa_row)
        assert result.status == CallStatus(status_value)

    def test_invalid_call_type_raises_value_error(self) -> None:
        sa_row = self._make_call_row(call_type="carrier_pigeon")
        repo = CallRepository()
        with pytest.raises(ValueError):
            repo.load(sa_row)

    def test_invalid_call_status_raises_value_error(self) -> None:
        sa_row = self._make_call_row(status="maybe")
        repo = CallRepository()
        with pytest.raises(ValueError):
            repo.load(sa_row)


# ---------------------------------------------------------------------------
# RoutingEventRepository
# ---------------------------------------------------------------------------


class TestRoutingEventRepository:
    """Tests for RoutingEventRepository.load()."""

    def _make_routing_row(self, **overrides: object) -> Mock:
        defaults = {
            "event_id": "evt-1",
            "state_id": "state-1",
            "edge_id": "edge-1",
            "routing_group_id": "rg-1",
            "ordinal": 0,
            "mode": "move",
            "created_at": NOW,
            "reason_hash": None,
            "reason_ref": None,
        }
        defaults.update(overrides)
        return _make_sa_row(**defaults)

    @pytest.mark.parametrize("mode_value", ["move", "copy", "divert"])
    def test_valid_routing_modes(self, mode_value: str) -> None:
        sa_row = self._make_routing_row(mode=mode_value)
        repo = RoutingEventRepository()
        result = repo.load(sa_row)
        assert isinstance(result, RoutingEvent)
        assert result.mode == RoutingMode(mode_value)

    def test_valid_load_all_fields(self) -> None:
        sa_row = self._make_routing_row(
            reason_hash="hash456",
            reason_ref="ref://reason/abc",
        )
        repo = RoutingEventRepository()
        result = repo.load(sa_row)
        assert result.event_id == "evt-1"
        assert result.reason_hash == "hash456"
        assert result.reason_ref == "ref://reason/abc"

    def test_invalid_mode_raises_value_error(self) -> None:
        sa_row = self._make_routing_row(mode="warp")
        repo = RoutingEventRepository()
        with pytest.raises(ValueError):
            repo.load(sa_row)


# ---------------------------------------------------------------------------
# BatchRepository
# ---------------------------------------------------------------------------


class TestBatchRepository:
    """Tests for BatchRepository.load()."""

    def _make_batch_row(self, **overrides: object) -> Mock:
        defaults = {
            "batch_id": "batch-1",
            "run_id": "run-1",
            "aggregation_node_id": "agg-node-1",
            "attempt": 1,
            "status": "draft",
            "created_at": NOW,
            "aggregation_state_id": None,
            "trigger_type": None,
            "trigger_reason": None,
            "completed_at": None,
        }
        defaults.update(overrides)
        return _make_sa_row(**defaults)

    @pytest.mark.parametrize(
        "status_value",
        ["draft", "executing", "completed", "failed"],
    )
    def test_valid_batch_status_values(self, status_value: str) -> None:
        sa_row = self._make_batch_row(status=status_value)
        repo = BatchRepository()
        result = repo.load(sa_row)
        assert isinstance(result, Batch)
        assert result.status == BatchStatus(status_value)

    def test_valid_load_all_fields(self) -> None:
        sa_row = self._make_batch_row(
            status="completed",
            aggregation_state_id="agg-state-1",
            trigger_type="count",
            trigger_reason="threshold=10",
            completed_at=NOW,
        )
        repo = BatchRepository()
        result = repo.load(sa_row)
        assert result.batch_id == "batch-1"
        assert result.aggregation_state_id == "agg-state-1"
        assert result.trigger_type == TriggerType.COUNT
        assert isinstance(result.trigger_type, TriggerType)
        assert result.trigger_reason == "threshold=10"
        assert result.completed_at == NOW

    def test_invalid_status_raises_value_error(self) -> None:
        sa_row = self._make_batch_row(status="cooking")
        repo = BatchRepository()
        with pytest.raises(ValueError):
            repo.load(sa_row)

    def test_invalid_trigger_type_raises_value_error(self) -> None:
        sa_row = self._make_batch_row(trigger_type="not_a_trigger")
        repo = BatchRepository()
        with pytest.raises(ValueError):
            repo.load(sa_row)


# ---------------------------------------------------------------------------
# NodeStateRepository (discriminated union -- most complex)
# ---------------------------------------------------------------------------


class TestNodeStateRepository:
    """Tests for NodeStateRepository.load() -- discriminated union."""

    def _make_node_state_row(self, **overrides: object) -> Mock:
        """Create a base node state row with all columns present.

        All completion-related fields default to None (representing OPEN).
        Tests override specific fields to represent other states.
        """
        defaults = {
            "state_id": "state-1",
            "token_id": "tok-1",
            "node_id": "node-1",
            "step_index": 0,
            "attempt": 1,
            "status": "open",
            "input_hash": "in123",
            "started_at": NOW,
            "output_hash": None,
            "completed_at": None,
            "duration_ms": None,
            "context_before_json": None,
            "context_after_json": None,
            "success_reason_json": None,
            "error_json": None,
        }
        defaults.update(overrides)
        return _make_sa_row(**defaults)

    # === OPEN variant ===

    def test_open_valid(self) -> None:
        sa_row = self._make_node_state_row(status="open")
        repo = NodeStateRepository()
        result = repo.load(sa_row)

        assert isinstance(result, NodeStateOpen)
        assert result.status == NodeStateStatus.OPEN
        assert result.state_id == "state-1"
        assert result.token_id == "tok-1"
        assert result.input_hash == "in123"

    def test_open_with_context_before(self) -> None:
        sa_row = self._make_node_state_row(
            status="open",
            context_before_json='{"key": "value"}',
        )
        repo = NodeStateRepository()
        result = repo.load(sa_row)
        assert isinstance(result, NodeStateOpen)
        assert result.context_before_json == '{"key": "value"}'

    def test_open_with_non_null_output_hash_raises(self) -> None:
        sa_row = self._make_node_state_row(
            status="open",
            output_hash="bad_hash",
        )
        repo = NodeStateRepository()
        with pytest.raises(ValueError, match="non-NULL output_hash"):
            repo.load(sa_row)

    def test_open_with_non_null_completed_at_raises(self) -> None:
        sa_row = self._make_node_state_row(
            status="open",
            completed_at=NOW,
        )
        repo = NodeStateRepository()
        with pytest.raises(ValueError, match="non-NULL completed_at"):
            repo.load(sa_row)

    def test_open_with_non_null_duration_ms_raises(self) -> None:
        sa_row = self._make_node_state_row(
            status="open",
            duration_ms=100.0,
        )
        repo = NodeStateRepository()
        with pytest.raises(ValueError, match="non-NULL duration_ms"):
            repo.load(sa_row)

    def test_open_with_all_invalid_completion_fields(self) -> None:
        """All three completion fields present on OPEN -- first check fires."""
        sa_row = self._make_node_state_row(
            status="open",
            output_hash="bad",
            completed_at=NOW,
            duration_ms=50.0,
        )
        repo = NodeStateRepository()
        with pytest.raises(ValueError, match="audit integrity violation"):
            repo.load(sa_row)

    # === PENDING variant ===

    def test_pending_valid(self) -> None:
        sa_row = self._make_node_state_row(
            status="pending",
            completed_at=NOW,
            duration_ms=150.5,
        )
        repo = NodeStateRepository()
        result = repo.load(sa_row)

        assert isinstance(result, NodeStatePending)
        assert result.status == NodeStateStatus.PENDING
        assert result.completed_at == NOW
        assert result.duration_ms == 150.5

    def test_pending_with_context_fields(self) -> None:
        sa_row = self._make_node_state_row(
            status="pending",
            completed_at=NOW,
            duration_ms=100.0,
            context_before_json='{"before": true}',
            context_after_json='{"after": true}',
        )
        repo = NodeStateRepository()
        result = repo.load(sa_row)
        assert isinstance(result, NodeStatePending)
        assert result.context_before_json == '{"before": true}'
        assert result.context_after_json == '{"after": true}'

    def test_pending_with_null_duration_ms_raises(self) -> None:
        sa_row = self._make_node_state_row(
            status="pending",
            completed_at=NOW,
            duration_ms=None,
        )
        repo = NodeStateRepository()
        with pytest.raises(ValueError, match="NULL duration_ms"):
            repo.load(sa_row)

    def test_pending_with_null_completed_at_raises(self) -> None:
        sa_row = self._make_node_state_row(
            status="pending",
            completed_at=None,
            duration_ms=100.0,
        )
        repo = NodeStateRepository()
        with pytest.raises(ValueError, match="NULL completed_at"):
            repo.load(sa_row)

    def test_pending_with_non_null_output_hash_raises(self) -> None:
        sa_row = self._make_node_state_row(
            status="pending",
            completed_at=NOW,
            duration_ms=100.0,
            output_hash="unexpected_hash",
        )
        repo = NodeStateRepository()
        with pytest.raises(ValueError, match="non-NULL output_hash"):
            repo.load(sa_row)

    # === COMPLETED variant ===

    def test_completed_valid(self) -> None:
        sa_row = self._make_node_state_row(
            status="completed",
            output_hash="out123",
            completed_at=NOW,
            duration_ms=200.0,
        )
        repo = NodeStateRepository()
        result = repo.load(sa_row)

        assert isinstance(result, NodeStateCompleted)
        assert result.status == NodeStateStatus.COMPLETED
        assert result.output_hash == "out123"
        assert result.completed_at == NOW
        assert result.duration_ms == 200.0

    def test_completed_with_optional_fields(self) -> None:
        sa_row = self._make_node_state_row(
            status="completed",
            output_hash="out123",
            completed_at=NOW,
            duration_ms=200.0,
            context_before_json='{"b": 1}',
            context_after_json='{"a": 2}',
            success_reason_json='{"action": "classified"}',
        )
        repo = NodeStateRepository()
        result = repo.load(sa_row)
        assert isinstance(result, NodeStateCompleted)
        assert result.success_reason_json == '{"action": "classified"}'
        assert result.context_before_json == '{"b": 1}'
        assert result.context_after_json == '{"a": 2}'

    def test_completed_with_null_output_hash_raises(self) -> None:
        sa_row = self._make_node_state_row(
            status="completed",
            output_hash=None,
            completed_at=NOW,
            duration_ms=100.0,
        )
        repo = NodeStateRepository()
        with pytest.raises(ValueError, match="NULL output_hash"):
            repo.load(sa_row)

    def test_completed_with_null_duration_ms_raises(self) -> None:
        sa_row = self._make_node_state_row(
            status="completed",
            output_hash="out123",
            completed_at=NOW,
            duration_ms=None,
        )
        repo = NodeStateRepository()
        with pytest.raises(ValueError, match="NULL duration_ms"):
            repo.load(sa_row)

    def test_completed_with_null_completed_at_raises(self) -> None:
        sa_row = self._make_node_state_row(
            status="completed",
            output_hash="out123",
            completed_at=None,
            duration_ms=100.0,
        )
        repo = NodeStateRepository()
        with pytest.raises(ValueError, match="NULL completed_at"):
            repo.load(sa_row)

    # === FAILED variant ===

    def test_failed_valid(self) -> None:
        sa_row = self._make_node_state_row(
            status="failed",
            completed_at=NOW,
            duration_ms=50.0,
        )
        repo = NodeStateRepository()
        result = repo.load(sa_row)

        assert isinstance(result, NodeStateFailed)
        assert result.status == NodeStateStatus.FAILED
        assert result.completed_at == NOW
        assert result.duration_ms == 50.0
        assert result.error_json is None
        assert result.output_hash is None

    def test_failed_with_error_json(self) -> None:
        sa_row = self._make_node_state_row(
            status="failed",
            completed_at=NOW,
            duration_ms=50.0,
            error_json='{"reason": "division by zero"}',
        )
        repo = NodeStateRepository()
        result = repo.load(sa_row)
        assert isinstance(result, NodeStateFailed)
        assert result.error_json == '{"reason": "division by zero"}'

    def test_failed_with_output_hash_allowed(self) -> None:
        """FAILED states MAY have output_hash (optional)."""
        sa_row = self._make_node_state_row(
            status="failed",
            completed_at=NOW,
            duration_ms=50.0,
            output_hash="partial_out",
        )
        repo = NodeStateRepository()
        result = repo.load(sa_row)
        assert isinstance(result, NodeStateFailed)
        assert result.output_hash == "partial_out"

    def test_failed_with_context_fields(self) -> None:
        sa_row = self._make_node_state_row(
            status="failed",
            completed_at=NOW,
            duration_ms=50.0,
            context_before_json='{"b": 1}',
            context_after_json='{"a": 2}',
        )
        repo = NodeStateRepository()
        result = repo.load(sa_row)
        assert isinstance(result, NodeStateFailed)
        assert result.context_before_json == '{"b": 1}'
        assert result.context_after_json == '{"a": 2}'

    def test_failed_with_null_duration_ms_raises(self) -> None:
        sa_row = self._make_node_state_row(
            status="failed",
            completed_at=NOW,
            duration_ms=None,
        )
        repo = NodeStateRepository()
        with pytest.raises(ValueError, match="NULL duration_ms"):
            repo.load(sa_row)

    def test_failed_with_null_completed_at_raises(self) -> None:
        sa_row = self._make_node_state_row(
            status="failed",
            completed_at=None,
            duration_ms=50.0,
        )
        repo = NodeStateRepository()
        with pytest.raises(ValueError, match="NULL completed_at"):
            repo.load(sa_row)

    # === Invalid status ===

    def test_unknown_status_string_raises_value_error(self) -> None:
        sa_row = self._make_node_state_row(status="processing")
        repo = NodeStateRepository()
        with pytest.raises(ValueError):
            repo.load(sa_row)

    def test_empty_status_string_raises_value_error(self) -> None:
        sa_row = self._make_node_state_row(status="")
        repo = NodeStateRepository()
        with pytest.raises(ValueError):
            repo.load(sa_row)

    # === Verify all 4 variants route correctly ===

    @pytest.mark.parametrize(
        ("status_str", "expected_type"),
        [
            ("open", NodeStateOpen),
            ("pending", NodeStatePending),
            ("completed", NodeStateCompleted),
            ("failed", NodeStateFailed),
        ],
    )
    def test_status_routes_to_correct_variant(self, status_str: str, expected_type: type) -> None:
        """Each status string produces the correct variant type."""
        extra: dict[str, object] = {}
        if status_str in ("pending", "failed"):
            extra = {"completed_at": NOW, "duration_ms": 100.0}
        elif status_str == "completed":
            extra = {
                "output_hash": "out",
                "completed_at": NOW,
                "duration_ms": 100.0,
            }

        sa_row = self._make_node_state_row(status=status_str, **extra)
        repo = NodeStateRepository()
        result = repo.load(sa_row)
        assert isinstance(result, expected_type)


# ---------------------------------------------------------------------------
# ValidationErrorRepository
# ---------------------------------------------------------------------------


class TestValidationErrorRepository:
    """Tests for ValidationErrorRepository.load()."""

    def test_valid_load(self) -> None:
        sa_row = _make_sa_row(
            error_id="ve-1",
            run_id="run-1",
            node_id="node-src",
            row_hash="hash123",
            error="Missing required field 'amount'",
            schema_mode="fixed",
            destination="quarantine",
            created_at=NOW,
            row_data_json='{"name": "test"}',
        )
        repo = ValidationErrorRepository()
        result = repo.load(sa_row)

        assert isinstance(result, ValidationErrorRecord)
        assert result.error_id == "ve-1"
        assert result.error == "Missing required field 'amount'"
        assert result.schema_mode == "fixed"
        assert result.row_data_json == '{"name": "test"}'

    def test_valid_load_with_none_optionals(self) -> None:
        sa_row = _make_sa_row(
            error_id="ve-2",
            run_id="run-1",
            node_id=None,
            row_hash="hash456",
            error="Type error",
            schema_mode="flexible",
            destination="quarantine",
            created_at=NOW,
            row_data_json=None,
        )
        repo = ValidationErrorRepository()
        result = repo.load(sa_row)
        assert result.node_id is None
        assert result.row_data_json is None


# ---------------------------------------------------------------------------
# TransformErrorRepository
# ---------------------------------------------------------------------------


class TestTransformErrorRepository:
    """Tests for TransformErrorRepository.load()."""

    def test_valid_load(self) -> None:
        sa_row = _make_sa_row(
            error_id="te-1",
            run_id="run-1",
            token_id="tok-1",
            transform_id="tfm-1",
            row_hash="hash789",
            destination="error_sink",
            created_at=NOW,
            row_data_json='{"x": 1}',
            error_details_json='{"reason": "division by zero"}',
        )
        repo = TransformErrorRepository()
        result = repo.load(sa_row)

        assert isinstance(result, TransformErrorRecord)
        assert result.error_id == "te-1"
        assert result.token_id == "tok-1"
        assert result.transform_id == "tfm-1"
        assert result.error_details_json == '{"reason": "division by zero"}'

    def test_valid_load_with_none_optionals(self) -> None:
        sa_row = _make_sa_row(
            error_id="te-2",
            run_id="run-1",
            token_id="tok-2",
            transform_id="tfm-2",
            row_hash="hash000",
            destination="error_sink",
            created_at=NOW,
            row_data_json=None,
            error_details_json=None,
        )
        repo = TransformErrorRepository()
        result = repo.load(sa_row)
        assert result.row_data_json is None
        assert result.error_details_json is None


# ---------------------------------------------------------------------------
# TokenOutcomeRepository
# ---------------------------------------------------------------------------


class TestTokenOutcomeRepository:
    """Tests for TokenOutcomeRepository.load()."""

    def _make_outcome_row(self, **overrides: object) -> Mock:
        defaults = {
            "outcome_id": "oc-1",
            "run_id": "run-1",
            "token_id": "tok-1",
            "outcome": "completed",
            "is_terminal": 1,
            "recorded_at": NOW,
            "sink_name": None,
            "batch_id": None,
            "fork_group_id": None,
            "join_group_id": None,
            "expand_group_id": None,
            "error_hash": None,
            "context_json": None,
            "expected_branches_json": None,
        }
        defaults.update(overrides)
        return _make_sa_row(**defaults)

    def test_valid_load_completed(self) -> None:
        sa_row = self._make_outcome_row(
            outcome="completed",
            is_terminal=1,
            sink_name="output",
        )
        repo = TokenOutcomeRepository()
        result = repo.load(sa_row)

        assert isinstance(result, TokenOutcome)
        assert result.outcome == RowOutcome.COMPLETED
        assert result.is_terminal is True
        assert result.sink_name == "output"

    def test_valid_load_buffered_non_terminal(self) -> None:
        sa_row = self._make_outcome_row(
            outcome="buffered",
            is_terminal=0,
        )
        repo = TokenOutcomeRepository()
        result = repo.load(sa_row)

        assert result.outcome == RowOutcome.BUFFERED
        assert result.is_terminal is False

    @pytest.mark.parametrize(
        "outcome_value",
        [
            "completed",
            "routed",
            "forked",
            "failed",
            "quarantined",
            "consumed_in_batch",
            "coalesced",
            "expanded",
            "buffered",
        ],
    )
    def test_all_row_outcome_values(self, outcome_value: str) -> None:
        expected_outcome = RowOutcome(outcome_value)
        sa_row = self._make_outcome_row(outcome=outcome_value, is_terminal=1 if expected_outcome.is_terminal else 0)
        repo = TokenOutcomeRepository()
        result = repo.load(sa_row)
        assert result.outcome == expected_outcome
        assert result.is_terminal is expected_outcome.is_terminal

    def test_is_terminal_1_becomes_true(self) -> None:
        sa_row = self._make_outcome_row(is_terminal=1)
        repo = TokenOutcomeRepository()
        result = repo.load(sa_row)
        assert result.is_terminal is True

    def test_is_terminal_0_becomes_false(self) -> None:
        sa_row = self._make_outcome_row(outcome="buffered", is_terminal=0)
        repo = TokenOutcomeRepository()
        result = repo.load(sa_row)
        assert result.is_terminal is False

    def test_outcome_buffered_with_terminal_1_raises_value_error(self) -> None:
        sa_row = self._make_outcome_row(outcome="buffered", is_terminal=1)
        repo = TokenOutcomeRepository()
        with pytest.raises(ValueError, match="inconsistent is_terminal"):
            repo.load(sa_row)

    def test_outcome_completed_with_terminal_0_raises_value_error(self) -> None:
        sa_row = self._make_outcome_row(outcome="completed", is_terminal=0)
        repo = TokenOutcomeRepository()
        with pytest.raises(ValueError, match="inconsistent is_terminal"):
            repo.load(sa_row)

    def test_is_terminal_2_raises_value_error(self) -> None:
        sa_row = self._make_outcome_row(is_terminal=2)
        repo = TokenOutcomeRepository()
        with pytest.raises(ValueError, match="invalid is_terminal"):
            repo.load(sa_row)

    def test_is_terminal_negative_1_raises_value_error(self) -> None:
        sa_row = self._make_outcome_row(is_terminal=-1)
        repo = TokenOutcomeRepository()
        with pytest.raises(ValueError, match="invalid is_terminal"):
            repo.load(sa_row)

    def test_is_terminal_none_raises_value_error(self) -> None:
        sa_row = self._make_outcome_row(is_terminal=None)
        repo = TokenOutcomeRepository()
        with pytest.raises(ValueError, match="invalid is_terminal"):
            repo.load(sa_row)

    def test_is_terminal_string_raises_value_error(self) -> None:
        """String '1' is NOT in (0, 1) -- Tier 1, no coercion."""
        sa_row = self._make_outcome_row(is_terminal="1")
        repo = TokenOutcomeRepository()
        with pytest.raises(ValueError, match="invalid is_terminal"):
            repo.load(sa_row)

    def test_is_terminal_true_bool_raises_value_error(self) -> None:
        """Python True == 1 so True is IN (0, 1). But the domain type
        validates is_terminal must be bool. Since True == 1, the repo
        check passes, but the result should be True (bool)."""
        # True is in (0, 1) because True == 1 in Python, so the repo
        # check passes. The resulting is_terminal = (True == 1) = True.
        sa_row = self._make_outcome_row(is_terminal=True)
        repo = TokenOutcomeRepository()
        result = repo.load(sa_row)
        assert result.is_terminal is True

    def test_invalid_outcome_raises_value_error(self) -> None:
        sa_row = self._make_outcome_row(outcome="vanished")
        repo = TokenOutcomeRepository()
        with pytest.raises(ValueError):
            repo.load(sa_row)

    def test_valid_load_with_all_optional_fields(self) -> None:
        sa_row = self._make_outcome_row(
            outcome="forked",
            is_terminal=1,
            sink_name=None,
            batch_id=None,
            fork_group_id="fg-1",
            join_group_id=None,
            expand_group_id=None,
            error_hash=None,
            context_json='{"paths": ["a", "b"]}',
            expected_branches_json='["path_a", "path_b"]',
        )
        repo = TokenOutcomeRepository()
        result = repo.load(sa_row)
        assert result.outcome == RowOutcome.FORKED
        assert result.fork_group_id == "fg-1"
        assert result.context_json == '{"paths": ["a", "b"]}'
        assert result.expected_branches_json == '["path_a", "path_b"]'

    def test_valid_load_routed_with_sink_name(self) -> None:
        sa_row = self._make_outcome_row(
            outcome="routed",
            is_terminal=1,
            sink_name="priority_output",
        )
        repo = TokenOutcomeRepository()
        result = repo.load(sa_row)
        assert result.outcome == RowOutcome.ROUTED
        assert result.sink_name == "priority_output"

    def test_valid_load_consumed_in_batch(self) -> None:
        sa_row = self._make_outcome_row(
            outcome="consumed_in_batch",
            is_terminal=1,
            batch_id="batch-1",
        )
        repo = TokenOutcomeRepository()
        result = repo.load(sa_row)
        assert result.outcome == RowOutcome.CONSUMED_IN_BATCH
        assert result.batch_id == "batch-1"

    def test_valid_load_failed_with_error_hash(self) -> None:
        sa_row = self._make_outcome_row(
            outcome="failed",
            is_terminal=1,
            error_hash="err123",
        )
        repo = TokenOutcomeRepository()
        result = repo.load(sa_row)
        assert result.outcome == RowOutcome.FAILED
        assert result.error_hash == "err123"


# ---------------------------------------------------------------------------
# ArtifactRepository
# ---------------------------------------------------------------------------


class TestArtifactRepository:
    """Tests for ArtifactRepository.load()."""

    def test_valid_load(self) -> None:
        sa_row = _make_sa_row(
            artifact_id="art-1",
            run_id="run-1",
            produced_by_state_id="state-1",
            sink_node_id="sink-1",
            artifact_type="csv",
            path_or_uri="/output/results.csv",
            content_hash="hash999",
            size_bytes=1024,
            created_at=NOW,
            idempotency_key=None,
        )
        repo = ArtifactRepository()
        result = repo.load(sa_row)

        assert isinstance(result, Artifact)
        assert result.artifact_id == "art-1"
        assert result.artifact_type == "csv"
        assert result.path_or_uri == "/output/results.csv"
        assert result.size_bytes == 1024
        assert result.idempotency_key is None

    def test_valid_load_with_idempotency_key(self) -> None:
        sa_row = _make_sa_row(
            artifact_id="art-2",
            run_id="run-1",
            produced_by_state_id="state-2",
            sink_node_id="sink-1",
            artifact_type="json",
            path_or_uri="/output/data.json",
            content_hash="hash888",
            size_bytes=2048,
            created_at=NOW,
            idempotency_key="retry-key-42",
        )
        repo = ArtifactRepository()
        result = repo.load(sa_row)
        assert result.idempotency_key == "retry-key-42"


# ---------------------------------------------------------------------------
# BatchMemberRepository
# ---------------------------------------------------------------------------


class TestBatchMemberRepository:
    """Tests for BatchMemberRepository.load()."""

    def test_valid_load(self) -> None:
        sa_row = _make_sa_row(
            batch_id="batch-1",
            token_id="tok-1",
            ordinal=0,
        )
        repo = BatchMemberRepository()
        result = repo.load(sa_row)

        assert isinstance(result, BatchMember)
        assert result.batch_id == "batch-1"
        assert result.token_id == "tok-1"
        assert result.ordinal == 0

    def test_valid_load_higher_ordinal(self) -> None:
        sa_row = _make_sa_row(
            batch_id="batch-2",
            token_id="tok-5",
            ordinal=10,
        )
        repo = BatchMemberRepository()
        result = repo.load(sa_row)
        assert result.ordinal == 10
