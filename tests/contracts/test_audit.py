"""Tests for audit trail contracts."""

from datetime import UTC, datetime

from elspeth.contracts import (
    Artifact,
    Batch,
    BatchMember,
    BatchOutput,
    BatchStatus,
    Call,
    CallStatus,
    CallType,
    Determinism,
    Edge,
    ExportStatus,
    Node,
    NodeState,
    NodeStateCompleted,
    NodeStateFailed,
    NodeStateOpen,
    NodeStateStatus,
    NodeType,
    RoutingEvent,
    RoutingMode,
    Row,
    Run,
    RunStatus,
    Token,
    TokenParent,
)


class TestRun:
    """Tests for Run audit model."""

    def test_create_run_with_required_fields(self) -> None:
        """Can create Run with required fields and RunStatus enum."""
        now = datetime.now(UTC)
        run = Run(
            run_id="run-123",
            started_at=now,
            config_hash="abc123",
            settings_json="{}",
            canonical_version="1.0.0",
            status=RunStatus.RUNNING,
        )

        assert run.run_id == "run-123"
        assert run.started_at == now
        assert run.status == RunStatus.RUNNING
        assert run.completed_at is None
        assert run.export_status is None

    def test_run_status_must_be_enum(self) -> None:
        """Run.status must be RunStatus enum, not string."""
        run = Run(
            run_id="run-123",
            started_at=datetime.now(UTC),
            config_hash="abc123",
            settings_json="{}",
            canonical_version="1.0.0",
            status=RunStatus.COMPLETED,
        )

        # Status is enum type, not just string
        assert run.status == RunStatus.COMPLETED
        assert isinstance(run.status, RunStatus)
        assert run.status.value == "completed"

    def test_run_with_export_status(self) -> None:
        """Run can have ExportStatus enum."""
        run = Run(
            run_id="run-123",
            started_at=datetime.now(UTC),
            config_hash="abc123",
            settings_json="{}",
            canonical_version="1.0.0",
            status=RunStatus.COMPLETED,
            export_status=ExportStatus.PENDING,
        )

        assert run.export_status == ExportStatus.PENDING
        assert run.export_status.value == "pending"


class TestNode:
    """Tests for Node audit model."""

    def test_create_node_with_enum_fields(self) -> None:
        """Node requires NodeType and Determinism enums."""
        now = datetime.now(UTC)
        node = Node(
            node_id="node-123",
            run_id="run-456",
            plugin_name="csv_source",
            node_type=NodeType.SOURCE,
            plugin_version="1.0.0",
            determinism=Determinism.IO_READ,
            config_hash="abc123",
            config_json="{}",
            registered_at=now,
        )

        assert node.node_id == "node-123"
        assert node.node_type == NodeType.SOURCE
        assert node.determinism == Determinism.IO_READ

    def test_node_type_is_enum(self) -> None:
        """Node.node_type must be NodeType enum."""
        node = Node(
            node_id="node-123",
            run_id="run-456",
            plugin_name="gate",
            node_type=NodeType.GATE,
            plugin_version="1.0.0",
            determinism=Determinism.DETERMINISTIC,
            config_hash="abc123",
            config_json="{}",
            registered_at=datetime.now(UTC),
        )

        assert node.node_type == NodeType.GATE
        assert isinstance(node.node_type, NodeType)
        assert node.node_type.value == "gate"

    def test_determinism_is_enum(self) -> None:
        """Node.determinism must be Determinism enum."""
        node = Node(
            node_id="node-123",
            run_id="run-456",
            plugin_name="llm_transform",
            node_type=NodeType.TRANSFORM,
            plugin_version="1.0.0",
            determinism=Determinism.EXTERNAL_CALL,
            config_hash="abc123",
            config_json="{}",
            registered_at=datetime.now(UTC),
        )

        assert node.determinism == Determinism.EXTERNAL_CALL
        assert isinstance(node.determinism, Determinism)
        assert node.determinism.value == "external_call"


class TestEdge:
    """Tests for Edge audit model."""

    def test_create_edge_with_routing_mode(self) -> None:
        """Edge requires RoutingMode enum."""
        now = datetime.now(UTC)
        edge = Edge(
            edge_id="edge-123",
            run_id="run-456",
            from_node_id="node-1",
            to_node_id="node-2",
            label="continue",
            default_mode=RoutingMode.MOVE,
            created_at=now,
        )

        assert edge.edge_id == "edge-123"
        assert edge.default_mode == RoutingMode.MOVE

    def test_default_mode_is_enum(self) -> None:
        """Edge.default_mode must be RoutingMode enum."""
        edge = Edge(
            edge_id="edge-123",
            run_id="run-456",
            from_node_id="node-1",
            to_node_id="node-2",
            label="fork",
            default_mode=RoutingMode.COPY,
            created_at=datetime.now(UTC),
        )

        assert edge.default_mode == RoutingMode.COPY
        assert isinstance(edge.default_mode, RoutingMode)
        assert edge.default_mode.value == "copy"


class TestRow:
    """Tests for Row audit model."""

    def test_create_row(self) -> None:
        """Can create Row with all primitive fields."""
        now = datetime.now(UTC)
        row = Row(
            row_id="row-123",
            run_id="run-456",
            source_node_id="node-1",
            row_index=0,
            source_data_hash="abc123",
            created_at=now,
        )

        assert row.row_id == "row-123"
        assert row.row_index == 0
        assert row.source_data_ref is None

    def test_row_with_payload_ref(self) -> None:
        """Row can have source_data_ref for payload store."""
        row = Row(
            row_id="row-123",
            run_id="run-456",
            source_node_id="node-1",
            row_index=0,
            source_data_hash="abc123",
            created_at=datetime.now(UTC),
            source_data_ref="payload://abc123",
        )

        assert row.source_data_ref == "payload://abc123"


class TestToken:
    """Tests for Token audit model."""

    def test_create_token(self) -> None:
        """Can create Token with required fields."""
        now = datetime.now(UTC)
        token = Token(
            token_id="tok-123",
            row_id="row-456",
            created_at=now,
        )

        assert token.token_id == "tok-123"
        assert token.row_id == "row-456"
        assert token.fork_group_id is None
        assert token.branch_name is None

    def test_token_with_fork_fields(self) -> None:
        """Token can have fork/join metadata."""
        token = Token(
            token_id="tok-123",
            row_id="row-456",
            created_at=datetime.now(UTC),
            fork_group_id="fork-789",
            branch_name="sentiment",
            step_in_pipeline=3,
        )

        assert token.fork_group_id == "fork-789"
        assert token.branch_name == "sentiment"
        assert token.step_in_pipeline == 3


class TestTokenParent:
    """Tests for TokenParent audit model."""

    def test_create_token_parent(self) -> None:
        """Can create TokenParent for lineage tracking."""
        parent = TokenParent(
            token_id="tok-child",
            parent_token_id="tok-parent",
            ordinal=0,
        )

        assert parent.token_id == "tok-child"
        assert parent.parent_token_id == "tok-parent"
        assert parent.ordinal == 0

    def test_multi_parent_ordinal(self) -> None:
        """Ordinal supports multi-parent joins."""
        parent1 = TokenParent(
            token_id="tok-joined",
            parent_token_id="tok-a",
            ordinal=0,
        )
        parent2 = TokenParent(
            token_id="tok-joined",
            parent_token_id="tok-b",
            ordinal=1,
        )

        assert parent1.ordinal == 0
        assert parent2.ordinal == 1


class TestNodeStateVariants:
    """Tests for NodeState discriminated union."""

    def test_open_state_has_literal_status(self) -> None:
        """NodeStateOpen.status is Literal[OPEN]."""
        state = NodeStateOpen(
            state_id="state-1",
            token_id="token-1",
            node_id="node-1",
            step_index=0,
            attempt=1,
            status=NodeStateStatus.OPEN,
            input_hash="abc123",
            started_at=datetime.now(UTC),
        )
        assert state.status == NodeStateStatus.OPEN

    def test_completed_state_requires_output(self) -> None:
        """NodeStateCompleted requires output_hash and completed_at."""
        now = datetime.now(UTC)
        state = NodeStateCompleted(
            state_id="state-1",
            token_id="token-1",
            node_id="node-1",
            step_index=0,
            attempt=1,
            status=NodeStateStatus.COMPLETED,
            input_hash="abc123",
            started_at=now,
            output_hash="def456",  # Required
            completed_at=now,  # Required
            duration_ms=100.0,  # Required
        )
        assert state.output_hash == "def456"

    def test_failed_state_has_error_fields(self) -> None:
        """NodeStateFailed can have error_json."""
        now = datetime.now(UTC)
        state = NodeStateFailed(
            state_id="state-1",
            token_id="token-1",
            node_id="node-1",
            step_index=0,
            attempt=1,
            status=NodeStateStatus.FAILED,
            input_hash="abc123",
            started_at=now,
            completed_at=now,
            duration_ms=50.0,
            error_json='{"error": "something went wrong"}',
        )
        assert state.status == NodeStateStatus.FAILED
        assert state.error_json == '{"error": "something went wrong"}'

    def test_union_type_annotation(self) -> None:
        """NodeState is union of all variants."""
        # Type checker accepts any variant
        state: NodeState = NodeStateOpen(
            state_id="state-1",
            token_id="token-1",
            node_id="node-1",
            step_index=0,
            attempt=1,
            status=NodeStateStatus.OPEN,
            input_hash="abc123",
            started_at=datetime.now(UTC),
        )
        assert state is not None

    def test_frozen_dataclass_immutable(self) -> None:
        """NodeState variants are frozen (immutable)."""
        import dataclasses

        state = NodeStateOpen(
            state_id="state-1",
            token_id="token-1",
            node_id="node-1",
            step_index=0,
            attempt=1,
            status=NodeStateStatus.OPEN,
            input_hash="abc123",
            started_at=datetime.now(UTC),
        )
        # Frozen dataclass should raise FrozenInstanceError on mutation
        with __import__("pytest").raises(dataclasses.FrozenInstanceError):
            state.state_id = "modified"  # type: ignore[misc]


class TestCall:
    """Tests for Call audit model."""

    def test_create_call_with_required_fields(self) -> None:
        """Can create Call with required fields and enum types."""
        now = datetime.now(UTC)
        call = Call(
            call_id="call-123",
            state_id="state-456",
            call_index=0,
            call_type=CallType.LLM,
            status=CallStatus.SUCCESS,
            request_hash="abc123",
            created_at=now,
        )

        assert call.call_id == "call-123"
        assert call.call_type == CallType.LLM
        assert call.status == CallStatus.SUCCESS
        assert call.response_hash is None
        assert call.latency_ms is None

    def test_call_type_must_be_enum(self) -> None:
        """Call.call_type must be CallType enum, not string."""
        call = Call(
            call_id="call-123",
            state_id="state-456",
            call_index=0,
            call_type=CallType.HTTP,
            status=CallStatus.ERROR,
            request_hash="abc123",
            created_at=datetime.now(UTC),
        )

        assert call.call_type == CallType.HTTP
        assert isinstance(call.call_type, CallType)
        assert call.call_type.value == "http"

    def test_call_status_must_be_enum(self) -> None:
        """Call.status must be CallStatus enum, not string."""
        call = Call(
            call_id="call-123",
            state_id="state-456",
            call_index=0,
            call_type=CallType.SQL,
            status=CallStatus.SUCCESS,
            request_hash="abc123",
            created_at=datetime.now(UTC),
        )

        assert call.status == CallStatus.SUCCESS
        assert isinstance(call.status, CallStatus)
        assert call.status.value == "success"

    def test_call_with_optional_fields(self) -> None:
        """Call can have optional response and error fields."""
        call = Call(
            call_id="call-123",
            state_id="state-456",
            call_index=0,
            call_type=CallType.LLM,
            status=CallStatus.SUCCESS,
            request_hash="abc123",
            created_at=datetime.now(UTC),
            request_ref="payload://req",
            response_hash="def456",
            response_ref="payload://resp",
            latency_ms=150.5,
        )

        assert call.response_hash == "def456"
        assert call.latency_ms == 150.5


class TestArtifact:
    """Tests for Artifact audit model."""

    def test_create_artifact(self) -> None:
        """Can create Artifact with all fields."""
        now = datetime.now(UTC)
        artifact = Artifact(
            artifact_id="art-123",
            run_id="run-456",
            produced_by_state_id="state-789",
            sink_node_id="node-1",
            artifact_type="csv",
            path_or_uri="/output/results.csv",
            content_hash="abc123",
            size_bytes=1024,
            created_at=now,
        )

        assert artifact.artifact_id == "art-123"
        assert artifact.artifact_type == "csv"
        assert artifact.size_bytes == 1024

    def test_artifact_type_is_string_not_enum(self) -> None:
        """Artifact.artifact_type is user-defined string, not enum."""
        artifact = Artifact(
            artifact_id="art-123",
            run_id="run-456",
            produced_by_state_id="state-789",
            sink_node_id="node-1",
            artifact_type="custom_format",
            path_or_uri="s3://bucket/results",
            content_hash="abc123",
            size_bytes=2048,
            created_at=datetime.now(UTC),
        )

        # User-defined type, accepts any string
        assert artifact.artifact_type == "custom_format"
        assert isinstance(artifact.artifact_type, str)


class TestRoutingEvent:
    """Tests for RoutingEvent audit model."""

    def test_create_routing_event_with_required_fields(self) -> None:
        """Can create RoutingEvent with required fields and RoutingMode enum."""
        now = datetime.now(UTC)
        event = RoutingEvent(
            event_id="evt-123",
            state_id="state-456",
            edge_id="edge-789",
            routing_group_id="group-1",
            ordinal=0,
            mode=RoutingMode.MOVE,
            created_at=now,
        )

        assert event.event_id == "evt-123"
        assert event.mode == RoutingMode.MOVE
        assert event.reason_hash is None

    def test_routing_mode_must_be_enum(self) -> None:
        """RoutingEvent.mode must be RoutingMode enum, not string."""
        event = RoutingEvent(
            event_id="evt-123",
            state_id="state-456",
            edge_id="edge-789",
            routing_group_id="group-1",
            ordinal=0,
            mode=RoutingMode.COPY,
            created_at=datetime.now(UTC),
        )

        assert event.mode == RoutingMode.COPY
        assert isinstance(event.mode, RoutingMode)
        assert event.mode.value == "copy"

    def test_routing_event_with_reason(self) -> None:
        """RoutingEvent can have reason_hash and reason_ref."""
        event = RoutingEvent(
            event_id="evt-123",
            state_id="state-456",
            edge_id="edge-789",
            routing_group_id="group-1",
            ordinal=0,
            mode=RoutingMode.MOVE,
            created_at=datetime.now(UTC),
            reason_hash="reason123",
            reason_ref="payload://reason",
        )

        assert event.reason_hash == "reason123"
        assert event.reason_ref == "payload://reason"


class TestBatch:
    """Tests for Batch audit model."""

    def test_create_batch_with_required_fields(self) -> None:
        """Can create Batch with required fields and BatchStatus enum."""
        now = datetime.now(UTC)
        batch = Batch(
            batch_id="batch-123",
            run_id="run-456",
            aggregation_node_id="node-1",
            attempt=1,
            status=BatchStatus.DRAFT,
            created_at=now,
        )

        assert batch.batch_id == "batch-123"
        assert batch.status == BatchStatus.DRAFT
        assert batch.aggregation_state_id is None
        assert batch.completed_at is None

    def test_batch_status_must_be_enum(self) -> None:
        """Batch.status must be BatchStatus enum, not string."""
        batch = Batch(
            batch_id="batch-123",
            run_id="run-456",
            aggregation_node_id="node-1",
            attempt=1,
            status=BatchStatus.EXECUTING,
            created_at=datetime.now(UTC),
        )

        assert batch.status == BatchStatus.EXECUTING
        assert isinstance(batch.status, BatchStatus)
        assert batch.status.value == "executing"

    def test_batch_with_completion(self) -> None:
        """Batch can have completion fields."""
        now = datetime.now(UTC)
        batch = Batch(
            batch_id="batch-123",
            run_id="run-456",
            aggregation_node_id="node-1",
            attempt=1,
            status=BatchStatus.COMPLETED,
            created_at=now,
            aggregation_state_id="state-789",
            trigger_reason="count_threshold",
            completed_at=now,
        )

        assert batch.status == BatchStatus.COMPLETED
        assert batch.aggregation_state_id == "state-789"
        assert batch.trigger_reason == "count_threshold"


class TestBatchMember:
    """Tests for BatchMember audit model."""

    def test_create_batch_member(self) -> None:
        """Can create BatchMember with all fields."""
        member = BatchMember(
            batch_id="batch-123",
            token_id="token-456",
            ordinal=0,
        )

        assert member.batch_id == "batch-123"
        assert member.token_id == "token-456"
        assert member.ordinal == 0

    def test_batch_member_ordinals(self) -> None:
        """Ordinal tracks member order in batch."""
        member1 = BatchMember(batch_id="batch-1", token_id="tok-a", ordinal=0)
        member2 = BatchMember(batch_id="batch-1", token_id="tok-b", ordinal=1)
        member3 = BatchMember(batch_id="batch-1", token_id="tok-c", ordinal=2)

        assert member1.ordinal == 0
        assert member2.ordinal == 1
        assert member3.ordinal == 2


class TestBatchOutput:
    """Tests for BatchOutput audit model."""

    def test_create_batch_output_token(self) -> None:
        """Can create BatchOutput for token output."""
        output = BatchOutput(
            batch_id="batch-123",
            output_type="token",
            output_id="token-456",
        )

        assert output.batch_id == "batch-123"
        assert output.output_type == "token"
        assert output.output_id == "token-456"

    def test_create_batch_output_artifact(self) -> None:
        """Can create BatchOutput for artifact output."""
        output = BatchOutput(
            batch_id="batch-123",
            output_type="artifact",
            output_id="artifact-789",
        )

        assert output.output_type == "artifact"
        assert output.output_id == "artifact-789"
