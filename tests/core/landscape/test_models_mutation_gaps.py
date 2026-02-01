# tests/core/landscape/test_models_mutation_gaps.py
"""Tests specifically targeting mutation testing gaps in models.py.

These tests were written to kill surviving mutants found during mutation testing.
Each test targets specific lines where mutations survived, indicating weak coverage.

Mutation testing run: 2026-01-23 (partial, 59% complete)
Survivors in models.py: 36 unique lines

The mutations here are primarily:
- Changing `= None` defaults to other values
- Removing default values from optional fields
- Changing field types
"""

from __future__ import annotations

from datetime import UTC, datetime

import pytest

from elspeth.contracts import (
    Artifact,
    Batch,
    BatchMember,
    BatchOutput,
    BatchStatus,
    Call,
    CallStatus,
    CallType,
    Checkpoint,
    Determinism,
    Edge,
    ExportStatus,
    Node,
    NodeState,
    NodeStateCompleted,
    NodeStateFailed,
    NodeStateOpen,
    NodeStatePending,
    NodeStateStatus,
    NodeType,
    RoutingEvent,
    RoutingMode,
    Row,
    RowLineage,
    Run,
    RunStatus,
    Token,
    TokenParent,
)

# =============================================================================
# Tests for Run dataclass (lines 38-46)
# Mutations: changing field types, default values
# =============================================================================


class TestRunDataclass:
    """Verify Run dataclass field defaults and requirements."""

    @pytest.fixture
    def minimal_run(self) -> Run:
        """Create Run with only required fields."""
        return Run(
            run_id="run-001",
            started_at=datetime.now(UTC),
            config_hash="abc123",
            settings_json="{}",
            canonical_version="sha256-rfc8785-v1",
            status=RunStatus.RUNNING,
        )

    def test_status_is_required_run_status_enum(self, minimal_run: Run) -> None:
        """Line 38: status must be RunStatus enum, not string."""
        assert isinstance(minimal_run.status, RunStatus)
        assert minimal_run.status == RunStatus.RUNNING

    def test_completed_at_defaults_to_none(self, minimal_run: Run) -> None:
        """Line 39: completed_at must default to None, not empty string or 0."""
        assert minimal_run.completed_at is None
        assert not isinstance(minimal_run.completed_at, str)

    def test_reproducibility_grade_defaults_to_none(self, minimal_run: Run) -> None:
        """Line 40: reproducibility_grade must default to None."""
        assert minimal_run.reproducibility_grade is None

    def test_export_status_defaults_to_none(self, minimal_run: Run) -> None:
        """Line 42: export_status must default to None."""
        assert minimal_run.export_status is None

    def test_export_error_defaults_to_none(self, minimal_run: Run) -> None:
        """Line 43: export_error must default to None."""
        assert minimal_run.export_error is None

    def test_exported_at_defaults_to_none(self, minimal_run: Run) -> None:
        """Line 44: exported_at must default to None."""
        assert minimal_run.exported_at is None

    def test_export_format_defaults_to_none(self, minimal_run: Run) -> None:
        """Line 45: export_format must default to None."""
        assert minimal_run.export_format is None

    def test_export_sink_defaults_to_none(self, minimal_run: Run) -> None:
        """Line 46: export_sink must default to None."""
        assert minimal_run.export_sink is None

    def test_run_with_all_optional_fields_set(self) -> None:
        """Verify all optional fields can be set explicitly."""
        now = datetime.now(UTC)
        run = Run(
            run_id="run-002",
            started_at=now,
            config_hash="abc123",
            settings_json="{}",
            canonical_version="sha256-rfc8785-v1",
            status=RunStatus.COMPLETED,
            completed_at=now,
            reproducibility_grade="FULL_REPRODUCIBLE",
            export_status=ExportStatus.COMPLETED,
            export_error=None,
            exported_at=now,
            export_format="csv",
            export_sink="output",
        )
        assert run.completed_at == now
        assert run.reproducibility_grade == "FULL_REPRODUCIBLE"
        assert run.export_status == ExportStatus.COMPLETED
        assert run.export_format == "csv"


# =============================================================================
# Tests for Node dataclass (lines 61-63)
# Mutations: changing required/optional, default values
# =============================================================================


class TestNodeDataclass:
    """Verify Node dataclass field defaults and requirements."""

    @pytest.fixture
    def minimal_node(self) -> Node:
        """Create Node with only required fields."""
        return Node(
            node_id="node-001",
            run_id="run-001",
            plugin_name="test_plugin",
            node_type=NodeType.TRANSFORM,
            plugin_version="1.0.0",
            determinism=Determinism.DETERMINISTIC,
            config_hash="xyz",
            config_json="{}",
            registered_at=datetime.now(UTC),
        )

    def test_registered_at_is_required(self) -> None:
        """Line 61: registered_at is required (no default)."""
        # This should raise TypeError if registered_at has a default
        with pytest.raises(TypeError):
            Node(  # type: ignore[call-arg]
                node_id="node-001",
                run_id="run-001",
                plugin_name="test",
                node_type=NodeType.SOURCE,
                plugin_version="1.0",
                determinism=Determinism.DETERMINISTIC,
                config_hash="x",
                config_json="{}",
                # registered_at missing
            )

    def test_schema_hash_defaults_to_none(self, minimal_node: Node) -> None:
        """Line 62: schema_hash must default to None."""
        assert minimal_node.schema_hash is None

    def test_sequence_in_pipeline_defaults_to_none(self, minimal_node: Node) -> None:
        """Line 63: sequence_in_pipeline must default to None."""
        assert minimal_node.sequence_in_pipeline is None


# =============================================================================
# Tests for Row dataclass (lines 88-89)
# =============================================================================


class TestRowDataclass:
    """Verify Row dataclass field defaults and requirements."""

    @pytest.fixture
    def minimal_row(self) -> Row:
        """Create Row with only required fields."""
        return Row(
            row_id="row-001",
            run_id="run-001",
            source_node_id="source-node",
            row_index=0,
            source_data_hash="hash123",
            created_at=datetime.now(UTC),
        )

    def test_created_at_is_required(self) -> None:
        """Line 88: created_at is required (no default)."""
        with pytest.raises(TypeError):
            Row(  # type: ignore[call-arg]
                row_id="row-001",
                run_id="run-001",
                source_node_id="source-node",
                row_index=0,
                source_data_hash="hash123",
                # created_at missing
            )

    def test_source_data_ref_defaults_to_none(self, minimal_row: Row) -> None:
        """Line 89: source_data_ref must default to None."""
        assert minimal_row.source_data_ref is None


# =============================================================================
# Tests for Token dataclass (lines 98-103)
# =============================================================================


class TestTokenDataclass:
    """Verify Token dataclass field defaults and requirements."""

    @pytest.fixture
    def minimal_token(self) -> Token:
        """Create Token with only required fields."""
        return Token(
            token_id="tok-001",
            row_id="row-001",
            created_at=datetime.now(UTC),
        )

    def test_created_at_is_required(self) -> None:
        """Line 98: created_at is required (no default)."""
        with pytest.raises(TypeError):
            Token(  # type: ignore[call-arg]
                token_id="tok-001",
                row_id="row-001",
                # created_at missing
            )

    def test_fork_group_id_defaults_to_none(self, minimal_token: Token) -> None:
        """Line 99: fork_group_id must default to None."""
        assert minimal_token.fork_group_id is None

    def test_join_group_id_defaults_to_none(self, minimal_token: Token) -> None:
        """Line 100: join_group_id must default to None."""
        assert minimal_token.join_group_id is None

    def test_expand_group_id_defaults_to_none(self, minimal_token: Token) -> None:
        """Line 101: expand_group_id must default to None."""
        assert minimal_token.expand_group_id is None

    def test_branch_name_defaults_to_none(self, minimal_token: Token) -> None:
        """Line 102: branch_name must default to None."""
        assert minimal_token.branch_name is None

    def test_step_in_pipeline_defaults_to_none(self, minimal_token: Token) -> None:
        """Line 103: step_in_pipeline must default to None."""
        assert minimal_token.step_in_pipeline is None


# =============================================================================
# Tests for NodeStateOpen dataclass (lines 135-138)
# =============================================================================


class TestNodeStateOpenDataclass:
    """Verify NodeStateOpen dataclass field defaults and requirements."""

    @pytest.fixture
    def minimal_open_state(self) -> NodeStateOpen:
        """Create NodeStateOpen with only required fields."""
        return NodeStateOpen(
            state_id="state-001",
            token_id="tok-001",
            node_id="node-001",
            step_index=0,
            attempt=1,
            status=NodeStateStatus.OPEN,
            input_hash="input123",
            started_at=datetime.now(UTC),
        )

    def test_status_is_literal_open(self, minimal_open_state: NodeStateOpen) -> None:
        """Line 135: status must be NodeStateStatus.OPEN."""
        assert minimal_open_state.status == NodeStateStatus.OPEN

    def test_started_at_is_required(self) -> None:
        """Line 137: started_at is required (no default)."""
        with pytest.raises(TypeError):
            NodeStateOpen(  # type: ignore[call-arg]
                state_id="state-001",
                token_id="tok-001",
                node_id="node-001",
                step_index=0,
                attempt=1,
                status=NodeStateStatus.OPEN,
                input_hash="input123",
                # started_at missing
            )

    def test_context_before_json_defaults_to_none(self, minimal_open_state: NodeStateOpen) -> None:
        """Line 138: context_before_json must default to None."""
        assert minimal_open_state.context_before_json is None


# =============================================================================
# Tests for NodeStateCompleted dataclass (lines 162-166)
# =============================================================================


class TestNodeStateCompletedDataclass:
    """Verify NodeStateCompleted dataclass field defaults and requirements."""

    @pytest.fixture
    def minimal_completed_state(self) -> NodeStateCompleted:
        """Create NodeStateCompleted with only required fields."""
        now = datetime.now(UTC)
        return NodeStateCompleted(
            state_id="state-001",
            token_id="tok-001",
            node_id="node-001",
            step_index=0,
            attempt=1,
            status=NodeStateStatus.COMPLETED,
            input_hash="input123",
            started_at=now,
            output_hash="output456",
            completed_at=now,
            duration_ms=100.5,
        )

    def test_duration_ms_is_required(self) -> None:
        """Line 164: duration_ms is required (no default)."""
        now = datetime.now(UTC)
        with pytest.raises(TypeError):
            NodeStateCompleted(  # type: ignore[call-arg]
                state_id="state-001",
                token_id="tok-001",
                node_id="node-001",
                step_index=0,
                attempt=1,
                status=NodeStateStatus.COMPLETED,
                input_hash="input123",
                started_at=now,
                output_hash="output456",
                completed_at=now,
                # duration_ms missing
            )

    def test_context_before_json_defaults_to_none(self, minimal_completed_state: NodeStateCompleted) -> None:
        """Line 165: context_before_json must default to None."""
        assert minimal_completed_state.context_before_json is None

    def test_context_after_json_defaults_to_none(self, minimal_completed_state: NodeStateCompleted) -> None:
        """Line 166: context_after_json must default to None."""
        assert minimal_completed_state.context_after_json is None


# =============================================================================
# Tests for NodeStateFailed dataclass (lines 190-195)
# =============================================================================


class TestNodeStateFailedDataclass:
    """Verify NodeStateFailed dataclass field defaults and requirements."""

    @pytest.fixture
    def minimal_failed_state(self) -> NodeStateFailed:
        """Create NodeStateFailed with only required fields."""
        now = datetime.now(UTC)
        return NodeStateFailed(
            state_id="state-001",
            token_id="tok-001",
            node_id="node-001",
            step_index=0,
            attempt=1,
            status=NodeStateStatus.FAILED,
            input_hash="input123",
            started_at=now,
            completed_at=now,
            duration_ms=50.0,
        )

    def test_duration_ms_is_required(self) -> None:
        """Line 191: duration_ms is required (no default)."""
        now = datetime.now(UTC)
        with pytest.raises(TypeError):
            NodeStateFailed(  # type: ignore[call-arg]
                state_id="state-001",
                token_id="tok-001",
                node_id="node-001",
                step_index=0,
                attempt=1,
                status=NodeStateStatus.FAILED,
                input_hash="input123",
                started_at=now,
                completed_at=now,
                # duration_ms missing
            )

    def test_error_json_defaults_to_none(self, minimal_failed_state: NodeStateFailed) -> None:
        """Line 192: error_json must default to None."""
        assert minimal_failed_state.error_json is None

    def test_output_hash_defaults_to_none(self, minimal_failed_state: NodeStateFailed) -> None:
        """Line 193: output_hash must default to None."""
        assert minimal_failed_state.output_hash is None

    def test_context_before_json_defaults_to_none(self, minimal_failed_state: NodeStateFailed) -> None:
        """Line 194: context_before_json must default to None."""
        assert minimal_failed_state.context_before_json is None

    def test_context_after_json_defaults_to_none(self, minimal_failed_state: NodeStateFailed) -> None:
        """Line 195: context_after_json must default to None."""
        assert minimal_failed_state.context_after_json is None


# =============================================================================
# Tests for NodeState union type (lines 198-209)
# =============================================================================


class TestNodeStateUnion:
    """Verify NodeState discriminated union works correctly."""

    def test_node_state_open_is_node_state(self) -> None:
        """NodeStateOpen is a valid NodeState."""
        state: NodeState = NodeStateOpen(
            state_id="s1",
            token_id="t1",
            node_id="n1",
            step_index=0,
            attempt=1,
            status=NodeStateStatus.OPEN,
            input_hash="h1",
            started_at=datetime.now(UTC),
        )
        assert isinstance(state, NodeStateOpen)

    def test_node_state_completed_is_node_state(self) -> None:
        """NodeStateCompleted is a valid NodeState."""
        now = datetime.now(UTC)
        state: NodeState = NodeStateCompleted(
            state_id="s1",
            token_id="t1",
            node_id="n1",
            step_index=0,
            attempt=1,
            status=NodeStateStatus.COMPLETED,
            input_hash="h1",
            started_at=now,
            output_hash="o1",
            completed_at=now,
            duration_ms=10.0,
        )
        assert isinstance(state, NodeStateCompleted)

    def test_node_state_failed_is_node_state(self) -> None:
        """NodeStateFailed is a valid NodeState."""
        now = datetime.now(UTC)
        state: NodeState = NodeStateFailed(
            state_id="s1",
            token_id="t1",
            node_id="n1",
            step_index=0,
            attempt=1,
            status=NodeStateStatus.FAILED,
            input_hash="h1",
            started_at=now,
            completed_at=now,
            duration_ms=10.0,
        )
        assert isinstance(state, NodeStateFailed)

    def test_node_state_pending_is_node_state(self) -> None:
        """P1: NodeStatePending is a valid NodeState (async batch flows)."""
        now = datetime.now(UTC)
        state: NodeState = NodeStatePending(
            state_id="s1",
            token_id="t1",
            node_id="n1",
            step_index=0,
            attempt=1,
            status=NodeStateStatus.PENDING,
            input_hash="h1",
            started_at=now,
            completed_at=now,
            duration_ms=10.0,
        )
        assert isinstance(state, NodeStatePending)


# =============================================================================
# Tests for NodeStatePending dataclass (P1: missing from union coverage)
# =============================================================================


class TestNodeStatePendingDataclass:
    """P1: Verify NodeStatePending dataclass field defaults and requirements.

    NodeStatePending was missing from mutation-gap suite. It is a core audit
    state for async operations (batch submission) where processing completed
    but output is pending.
    """

    @pytest.fixture
    def minimal_pending_state(self) -> NodeStatePending:
        """Create NodeStatePending with only required fields."""
        now = datetime.now(UTC)
        return NodeStatePending(
            state_id="state-001",
            token_id="tok-001",
            node_id="node-001",
            step_index=0,
            attempt=1,
            status=NodeStateStatus.PENDING,
            input_hash="input123",
            started_at=now,
            completed_at=now,
            duration_ms=50.0,
        )

    def test_completed_at_is_required(self) -> None:
        """P1: completed_at is required for NodeStatePending (no default)."""
        now = datetime.now(UTC)
        with pytest.raises(TypeError):
            NodeStatePending(  # type: ignore[call-arg]
                state_id="state-001",
                token_id="tok-001",
                node_id="node-001",
                step_index=0,
                attempt=1,
                status=NodeStateStatus.PENDING,
                input_hash="input123",
                started_at=now,
                # completed_at missing
                duration_ms=50.0,
            )

    def test_duration_ms_is_required(self) -> None:
        """P1: duration_ms is required for NodeStatePending (no default)."""
        now = datetime.now(UTC)
        with pytest.raises(TypeError):
            NodeStatePending(  # type: ignore[call-arg]
                state_id="state-001",
                token_id="tok-001",
                node_id="node-001",
                step_index=0,
                attempt=1,
                status=NodeStateStatus.PENDING,
                input_hash="input123",
                started_at=now,
                completed_at=now,
                # duration_ms missing
            )

    def test_context_before_json_defaults_to_none(self, minimal_pending_state: NodeStatePending) -> None:
        """P1: context_before_json must default to None."""
        assert minimal_pending_state.context_before_json is None

    def test_context_after_json_defaults_to_none(self, minimal_pending_state: NodeStatePending) -> None:
        """P1: context_after_json must default to None."""
        assert minimal_pending_state.context_after_json is None

    def test_status_is_literal_pending(self, minimal_pending_state: NodeStatePending) -> None:
        """P1: status must be NodeStateStatus.PENDING."""
        assert minimal_pending_state.status == NodeStateStatus.PENDING


# =============================================================================
# Tests for Call dataclass (lines 222-227)
# =============================================================================


class TestCallDataclass:
    """Verify Call dataclass field defaults and requirements."""

    @pytest.fixture
    def minimal_call(self) -> Call:
        """Create Call with only required fields."""
        return Call(
            call_id="call-001",
            state_id="state-001",
            call_index=0,
            call_type=CallType.HTTP,
            status=CallStatus.SUCCESS,
            request_hash="req123",
            created_at=datetime.now(UTC),
        )

    def test_created_at_is_required(self) -> None:
        """Line 222: created_at is required (no default)."""
        with pytest.raises(TypeError):
            Call(  # type: ignore[call-arg]
                call_id="call-001",
                state_id="state-001",
                call_index=0,
                call_type=CallType.HTTP,
                status=CallStatus.SUCCESS,
                request_hash="req123",
                # created_at missing
            )

    def test_request_ref_defaults_to_none(self, minimal_call: Call) -> None:
        """Line 223: request_ref must default to None."""
        assert minimal_call.request_ref is None

    def test_response_hash_defaults_to_none(self, minimal_call: Call) -> None:
        """Line 224: response_hash must default to None."""
        assert minimal_call.response_hash is None

    def test_response_ref_defaults_to_none(self, minimal_call: Call) -> None:
        """Line 225: response_ref must default to None."""
        assert minimal_call.response_ref is None

    def test_error_json_defaults_to_none(self, minimal_call: Call) -> None:
        """Line 226: error_json must default to None."""
        assert minimal_call.error_json is None

    def test_latency_ms_defaults_to_none(self, minimal_call: Call) -> None:
        """Line 227: latency_ms must default to None."""
        assert minimal_call.latency_ms is None


# =============================================================================
# Tests for Artifact dataclass (line 243)
# =============================================================================


class TestArtifactDataclass:
    """Verify Artifact dataclass field defaults and requirements."""

    @pytest.fixture
    def minimal_artifact(self) -> Artifact:
        """Create Artifact with only required fields."""
        return Artifact(
            artifact_id="art-001",
            run_id="run-001",
            produced_by_state_id="state-001",
            sink_node_id="sink-001",
            artifact_type="csv",
            path_or_uri="/path/to/file.csv",
            content_hash="content123",
            size_bytes=1024,
            created_at=datetime.now(UTC),
        )

    def test_idempotency_key_defaults_to_none(self, minimal_artifact: Artifact) -> None:
        """Line 243: idempotency_key must default to None."""
        assert minimal_artifact.idempotency_key is None


# =============================================================================
# Tests for RoutingEvent dataclass (lines 256-258)
# =============================================================================


class TestRoutingEventDataclass:
    """Verify RoutingEvent dataclass field defaults and requirements."""

    @pytest.fixture
    def minimal_routing_event(self) -> RoutingEvent:
        """Create RoutingEvent with only required fields."""
        return RoutingEvent(
            event_id="evt-001",
            state_id="state-001",
            edge_id="edge-001",
            routing_group_id="rg-001",
            ordinal=0,
            mode=RoutingMode.MOVE,
            created_at=datetime.now(UTC),
        )

    def test_created_at_is_required(self) -> None:
        """Line 256: created_at is required (no default)."""
        with pytest.raises(TypeError):
            RoutingEvent(  # type: ignore[call-arg]
                event_id="evt-001",
                state_id="state-001",
                edge_id="edge-001",
                routing_group_id="rg-001",
                ordinal=0,
                mode=RoutingMode.MOVE,
                # created_at missing
            )

    def test_reason_hash_defaults_to_none(self, minimal_routing_event: RoutingEvent) -> None:
        """Line 257: reason_hash must default to None."""
        assert minimal_routing_event.reason_hash is None

    def test_reason_ref_defaults_to_none(self, minimal_routing_event: RoutingEvent) -> None:
        """Line 258: reason_ref must default to None."""
        assert minimal_routing_event.reason_ref is None


# =============================================================================
# Tests for Batch dataclass (lines 270-274)
# =============================================================================


class TestBatchDataclass:
    """Verify Batch dataclass field defaults and requirements."""

    @pytest.fixture
    def minimal_batch(self) -> Batch:
        """Create Batch with only required fields."""
        return Batch(
            batch_id="batch-001",
            run_id="run-001",
            aggregation_node_id="agg-node-001",
            attempt=1,
            status=BatchStatus.DRAFT,
            created_at=datetime.now(UTC),
        )

    def test_created_at_is_required(self) -> None:
        """Line 270: created_at is required (no default)."""
        with pytest.raises(TypeError):
            Batch(  # type: ignore[call-arg]
                batch_id="batch-001",
                run_id="run-001",
                aggregation_node_id="agg-node-001",
                attempt=1,
                status=BatchStatus.DRAFT,
                # created_at missing
            )

    def test_aggregation_state_id_defaults_to_none(self, minimal_batch: Batch) -> None:
        """Line 271: aggregation_state_id must default to None."""
        assert minimal_batch.aggregation_state_id is None

    def test_trigger_reason_defaults_to_none(self, minimal_batch: Batch) -> None:
        """Line 272: trigger_reason must default to None."""
        assert minimal_batch.trigger_reason is None

    def test_trigger_type_defaults_to_none(self, minimal_batch: Batch) -> None:
        """Line 273: trigger_type must default to None."""
        assert minimal_batch.trigger_type is None

    def test_completed_at_defaults_to_none(self, minimal_batch: Batch) -> None:
        """Line 274: completed_at must default to None."""
        assert minimal_batch.completed_at is None


# =============================================================================
# Tests for Checkpoint dataclass (lines 308-309)
# =============================================================================


class TestCheckpointDataclass:
    """Verify Checkpoint dataclass field defaults and requirements."""

    @pytest.fixture
    def minimal_checkpoint(self) -> Checkpoint:
        """Create Checkpoint with only required fields."""
        return Checkpoint(
            checkpoint_id="cp-001",
            run_id="run-001",
            token_id="tok-001",
            node_id="node-001",
            sequence_number=1,
            created_at=datetime.now(UTC),
            upstream_topology_hash="a" * 64,
            checkpoint_node_config_hash="b" * 64,
        )

    def test_aggregation_state_json_defaults_to_none(self, minimal_checkpoint: Checkpoint) -> None:
        """Line 309: aggregation_state_json must default to None."""
        assert minimal_checkpoint.aggregation_state_json is None

    def test_checkpoint_with_aggregation_state(self) -> None:
        """Checkpoint can have aggregation_state_json set."""
        cp = Checkpoint(
            checkpoint_id="cp-002",
            run_id="run-001",
            token_id="tok-001",
            node_id="node-001",
            sequence_number=2,
            created_at=datetime.now(UTC),
            upstream_topology_hash="a" * 64,
            checkpoint_node_config_hash="b" * 64,
            aggregation_state_json='{"count": 42}',
        )
        assert cp.aggregation_state_json == '{"count": 42}'


# =============================================================================
# Tests for Edge dataclass (lines 74-76)
# =============================================================================


class TestEdgeDataclass:
    """Verify Edge dataclass field requirements."""

    def test_created_at_is_required(self) -> None:
        """Line 76: created_at is required (no default)."""
        with pytest.raises(TypeError):
            Edge(  # type: ignore[call-arg]
                edge_id="edge-001",
                run_id="run-001",
                from_node_id="node-001",
                to_node_id="node-002",
                label="continue",
                default_mode=RoutingMode.MOVE,
                # created_at missing
            )

    def test_edge_with_all_fields(self) -> None:
        """Edge can be created with all fields."""
        edge = Edge(
            edge_id="edge-001",
            run_id="run-001",
            from_node_id="node-001",
            to_node_id="node-002",
            label="continue",
            default_mode=RoutingMode.MOVE,
            created_at=datetime.now(UTC),
        )
        assert edge.label == "continue"
        assert edge.default_mode == RoutingMode.MOVE


# =============================================================================
# Tests for RowLineage dataclass (lines 330-334)
# =============================================================================


class TestRowLineageDataclass:
    """Verify RowLineage dataclass field defaults."""

    def test_row_lineage_with_payload_available(self) -> None:
        """RowLineage with source_data available."""
        lineage = RowLineage(
            row_id="row-001",
            run_id="run-001",
            source_node_id="source-001",
            row_index=0,
            source_data_hash="hash123",
            created_at=datetime.now(UTC),
            source_data={"field": "value"},
            payload_available=True,
        )
        assert lineage.source_data == {"field": "value"}
        assert lineage.payload_available is True

    def test_row_lineage_with_payload_purged(self) -> None:
        """RowLineage with source_data purged."""
        lineage = RowLineage(
            row_id="row-001",
            run_id="run-001",
            source_node_id="source-001",
            row_index=0,
            source_data_hash="hash123",
            created_at=datetime.now(UTC),
            source_data=None,
            payload_available=False,
        )
        assert lineage.source_data is None
        assert lineage.payload_available is False


# =============================================================================
# Tests for TokenParent and BatchMember/BatchOutput (simple dataclasses)
# =============================================================================


class TestSimpleDataclasses:
    """Test simple dataclasses without optional fields."""

    def test_token_parent_all_fields_required(self) -> None:
        """TokenParent has no optional fields."""
        tp = TokenParent(
            token_id="tok-001",
            parent_token_id="tok-000",
            ordinal=0,
        )
        assert tp.token_id == "tok-001"
        assert tp.parent_token_id == "tok-000"
        assert tp.ordinal == 0

    def test_batch_member_all_fields_required(self) -> None:
        """BatchMember has no optional fields."""
        bm = BatchMember(
            batch_id="batch-001",
            token_id="tok-001",
            ordinal=0,
        )
        assert bm.batch_id == "batch-001"
        assert bm.token_id == "tok-001"
        assert bm.ordinal == 0

    def test_batch_output_all_fields_required(self) -> None:
        """BatchOutput has no optional fields."""
        bo = BatchOutput(
            batch_id="batch-001",
            output_type="token",
            output_id="tok-002",
        )
        assert bo.batch_id == "batch-001"
        assert bo.output_type == "token"
        assert bo.output_id == "tok-002"
