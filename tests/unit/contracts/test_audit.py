"""Tests for audit trail contracts."""

from datetime import UTC, datetime
from typing import Any

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

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
    Node,
    NodeState,
    NodeStateCompleted,
    NodeStateFailed,
    NodeStateOpen,
    NodeStatePending,
    NodeStateStatus,
    NodeType,
    NonCanonicalMetadata,
    Operation,
    RoutingEvent,
    RoutingMode,
    Row,
    RowLineage,
    Run,
    RunStatus,
    SecretResolution,
    SecretResolutionInput,
    Token,
    TokenOutcome,
    TransformErrorRecord,
    TriggerType,
    ValidationErrorRecord,
)
from elspeth.contracts.enums import _LEGAL_TERMINAL_PAIRS, TerminalOutcome, TerminalPath


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

    # NOTE: Frozen dataclass immutability is tested in TestFrozenDataclassImmutability
    # (parametrized test covering all frozen dataclasses - see line ~1372)


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

    def test_llm_call_accepts_resolved_prompt_template_hash(self) -> None:
        """LLM calls may carry the cross-DB resolved-prompt hash anchor."""
        call = Call(
            call_id="call-123",
            state_id="state-456",
            call_index=0,
            call_type=CallType.LLM,
            status=CallStatus.SUCCESS,
            request_hash="abc123",
            created_at=datetime.now(UTC),
            resolved_prompt_template_hash="a" * 64,
        )

        assert call.resolved_prompt_template_hash == "a" * 64

    def test_non_llm_call_rejects_resolved_prompt_template_hash(self) -> None:
        """Only LLM calls may carry the resolved-prompt hash anchor."""
        with pytest.raises(ValueError, match=r"resolved_prompt_template_hash.*CallType.LLM"):
            Call(
                call_id="call-123",
                state_id="state-456",
                call_index=0,
                call_type=CallType.HTTP,
                status=CallStatus.SUCCESS,
                request_hash="abc123",
                created_at=datetime.now(UTC),
                resolved_prompt_template_hash="a" * 64,
            )

    @pytest.mark.parametrize(
        "resolved_prompt_template_hash",
        [
            "a" * 63,
            "g" * 64,
        ],
    )
    def test_resolved_prompt_template_hash_must_be_64_char_hex(
        self,
        resolved_prompt_template_hash: str,
    ) -> None:
        """Non-null resolved_prompt_template_hash is a 64-character hex digest."""
        with pytest.raises(ValueError, match=r"resolved_prompt_template_hash.*64.*hex"):
            Call(
                call_id="call-123",
                state_id="state-456",
                call_index=0,
                call_type=CallType.LLM,
                status=CallStatus.SUCCESS,
                request_hash="abc123",
                created_at=datetime.now(UTC),
                resolved_prompt_template_hash=resolved_prompt_template_hash,
            )


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

    def test_batch_trigger_type_must_be_enum(self) -> None:
        """Batch.trigger_type must be TriggerType enum, not string."""
        batch = Batch(
            batch_id="batch-123",
            run_id="run-456",
            aggregation_node_id="node-1",
            attempt=1,
            status=BatchStatus.EXECUTING,
            created_at=datetime.now(UTC),
            trigger_type=TriggerType.COUNT,
        )

        assert batch.trigger_type == TriggerType.COUNT
        assert isinstance(batch.trigger_type, TriggerType)
        assert batch.trigger_type.value == "count"

    def test_batch_invalid_trigger_type_raises_type_error(self) -> None:
        """Invalid trigger_type type must crash immediately (Tier 1)."""
        with pytest.raises(TypeError, match="trigger_type must be TriggerType"):
            Batch(
                batch_id="batch-123",
                run_id="run-456",
                aggregation_node_id="node-1",
                attempt=1,
                status=BatchStatus.EXECUTING,
                created_at=datetime.now(UTC),
                trigger_type="count",  # type: ignore[arg-type]
            )


class TestBatchMember:
    """Tests for BatchMember audit model."""

    def test_create_batch_member(self) -> None:
        """Can create BatchMember with all fields."""
        member = BatchMember(
            batch_id="batch-123",
            run_id="run-789",
            token_id="token-456",
            ordinal=0,
        )

        assert member.batch_id == "batch-123"
        assert member.run_id == "run-789"
        assert member.token_id == "token-456"
        assert member.ordinal == 0

    def test_batch_member_ordinals(self) -> None:
        """Ordinal tracks member order in batch."""
        member1 = BatchMember(batch_id="batch-1", run_id="run-1", token_id="tok-a", ordinal=0)
        member2 = BatchMember(batch_id="batch-1", run_id="run-1", token_id="tok-b", ordinal=1)
        member3 = BatchMember(batch_id="batch-1", run_id="run-1", token_id="tok-c", ordinal=2)

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


class TestCheckpoint:
    """Tests for Checkpoint audit model."""

    def test_create_checkpoint_with_required_fields(self) -> None:
        """Can create Checkpoint with required fields."""

        now = datetime.now(UTC)
        checkpoint = Checkpoint(
            checkpoint_id="cp-123",
            run_id="run-456",
            sequence_number=42,
            created_at=now,
            upstream_topology_hash="a" * 64,
        )

        assert checkpoint.checkpoint_id == "cp-123"
        assert checkpoint.run_id == "run-456"
        assert checkpoint.sequence_number == 42
        assert checkpoint.created_at == now
        assert checkpoint.aggregation_state_json is None

    def test_checkpoint_with_aggregation_state(self) -> None:
        """Checkpoint can have aggregation_state_json for stateful nodes."""

        checkpoint = Checkpoint(
            checkpoint_id="cp-123",
            run_id="run-456",
            sequence_number=42,
            created_at=datetime.now(UTC),
            upstream_topology_hash="a" * 64,
            aggregation_state_json='{"count": 10, "sum": 500}',
        )

        assert checkpoint.aggregation_state_json == '{"count": 10, "sum": 500}'

    def test_checkpoint_created_at_required(self) -> None:
        """Checkpoint.created_at is required (Tier 1 audit data).

        Per Data Manifesto: Audit trail data must be 100% pristine.
        created_at is enforced as NOT NULL in the schema.
        """

        now = datetime.now(UTC)
        checkpoint = Checkpoint(
            checkpoint_id="cp-123",
            run_id="run-456",
            sequence_number=0,
            created_at=now,
            upstream_topology_hash="a" * 64,
        )

        # created_at is always present and valid for audit integrity
        assert checkpoint.created_at == now
        assert isinstance(checkpoint.created_at, datetime)


class TestRowLineage:
    """Tests for RowLineage audit model."""

    def test_create_row_lineage_with_all_fields(self) -> None:
        """Can create RowLineage with all fields."""

        now = datetime.now(UTC)
        lineage = RowLineage(
            row_id="row-123",
            run_id="run-456",
            source_node_id="node-src",
            row_index=42,
            source_row_index=7,
            ingest_sequence=42,
            source_data_hash="abc123def456",
            created_at=now,
            source_data={"id": 1, "name": "test"},
            payload_available=True,
        )

        assert lineage.row_id == "row-123"
        assert lineage.run_id == "run-456"
        assert lineage.source_node_id == "node-src"
        assert lineage.row_index == 42
        assert lineage.source_data_hash == "abc123def456"
        assert lineage.created_at == now
        assert lineage.source_data == {"id": 1, "name": "test"}
        assert lineage.payload_available is True

    def test_row_lineage_with_purged_payload(self) -> None:
        """RowLineage supports graceful payload degradation."""

        lineage = RowLineage(
            row_id="row-123",
            run_id="run-456",
            source_node_id="node-src",
            row_index=0,
            source_row_index=0,
            ingest_sequence=0,
            source_data_hash="abc123def456",
            created_at=datetime.now(UTC),
            source_data=None,  # Payload was purged
            payload_available=False,
        )

        # Hash is still available even though payload is purged
        assert lineage.source_data_hash == "abc123def456"
        assert lineage.source_data is None
        assert lineage.payload_available is False

    def test_row_lineage_hash_always_present(self) -> None:
        """RowLineage always has source_data_hash for audit integrity."""

        # Even with purged payload, hash is required and present
        lineage = RowLineage(
            row_id="row-123",
            run_id="run-456",
            source_node_id="node-src",
            row_index=0,
            source_row_index=0,
            ingest_sequence=0,
            source_data_hash="required_hash_value",
            created_at=datetime.now(UTC),
            source_data=None,
            payload_available=False,
        )

        assert lineage.source_data_hash == "required_hash_value"

    def test_row_lineage_requires_source_scoped_identity(self) -> None:
        """RowLineage must expose persisted source-local and ingest identity.

        source_row_index is a required `int` field; passing None (the legacy
        "absent identity" shape) is rejected by the Tier-1 require_int guard.
        """
        with pytest.raises(TypeError, match="source_row_index must be int"):
            RowLineage(
                row_id="row-123",
                run_id="run-456",
                source_node_id="node-src",
                row_index=0,
                source_row_index=None,  # type: ignore[arg-type]
                ingest_sequence=0,
                source_data_hash="required_hash_value",
                created_at=datetime.now(UTC),
                source_data=None,
                payload_available=False,
            )


# =============================================================================
# MISSING COVERAGE - P0 FIXES FROM QUALITY AUDIT
# =============================================================================


class TestNodeStatePending:
    """Tests for NodeStatePending audit model.

    NodeStatePending represents async operations where processing completed
    but output is pending (e.g., batch submission, deferred results).

    Invariants:
    - No output_hash (result not available yet)
    - Has completed_at (operation finished)
    - Has duration_ms (timing complete)
    - Status is Literal[NodeStateStatus.PENDING]
    """

    def test_pending_state_has_literal_status(self) -> None:
        """NodeStatePending.status is Literal[PENDING]."""
        now = datetime.now(UTC)
        state = NodeStatePending(
            state_id="state-1",
            token_id="token-1",
            node_id="node-1",
            step_index=0,
            attempt=1,
            status=NodeStateStatus.PENDING,
            input_hash="abc123",
            started_at=now,
            completed_at=now,
            duration_ms=100.0,
        )
        assert state.status == NodeStateStatus.PENDING

    def test_pending_state_has_timing_fields(self) -> None:
        """NodeStatePending has completed_at and duration_ms (unlike OPEN)."""
        started = datetime.now(UTC)
        completed = datetime.now(UTC)
        state = NodeStatePending(
            state_id="state-1",
            token_id="token-1",
            node_id="node-1",
            step_index=0,
            attempt=1,
            status=NodeStateStatus.PENDING,
            input_hash="abc123",
            started_at=started,
            completed_at=completed,
            duration_ms=150.5,
        )
        # Timing fields are present and accurate
        assert state.started_at == started
        assert state.completed_at == completed
        assert state.duration_ms == 150.5

    def test_pending_state_no_output_hash(self) -> None:
        """NodeStatePending does not have output_hash (result pending).

        This is the key distinction from NodeStateCompleted.
        The dataclass simply does not define an output_hash field.
        """
        import dataclasses

        now = datetime.now(UTC)
        state = NodeStatePending(
            state_id="state-1",
            token_id="token-1",
            node_id="node-1",
            step_index=0,
            attempt=1,
            status=NodeStateStatus.PENDING,
            input_hash="abc123",
            started_at=now,
            completed_at=now,
            duration_ms=100.0,
        )
        field_names = {f.name for f in dataclasses.fields(state)}
        assert "output_hash" not in field_names

    # NOTE: Frozen immutability tested in TestFrozenDataclassImmutability

    def test_pending_state_with_context_fields(self) -> None:
        """NodeStatePending can have optional context_before/after_json."""
        now = datetime.now(UTC)
        state = NodeStatePending(
            state_id="state-1",
            token_id="token-1",
            node_id="node-1",
            step_index=0,
            attempt=1,
            status=NodeStateStatus.PENDING,
            input_hash="abc123",
            started_at=now,
            completed_at=now,
            duration_ms=100.0,
            context_before_json='{"batch_id": "batch-123"}',
            context_after_json='{"submitted": true}',
        )
        assert state.context_before_json == '{"batch_id": "batch-123"}'
        assert state.context_after_json == '{"submitted": true}'

    def test_pending_state_in_node_state_union(self) -> None:
        """NodeStatePending is part of NodeState discriminated union."""
        now = datetime.now(UTC)
        state: NodeState = NodeStatePending(
            state_id="state-1",
            token_id="token-1",
            node_id="node-1",
            step_index=0,
            attempt=1,
            status=NodeStateStatus.PENDING,
            input_hash="abc123",
            started_at=now,
            completed_at=now,
            duration_ms=100.0,
        )
        # Type checker accepts assignment to NodeState union
        assert state is not None
        assert state.status == NodeStateStatus.PENDING


class TestTokenOutcome:
    """Tests for TokenOutcome audit model.

    TokenOutcome records the terminal state for a token, part of AUD-001
    audit integrity. Every token must have an explicit outcome recorded.
    """

    def test_create_token_outcome_with_required_fields(self) -> None:
        """TokenOutcome captures terminal state determination."""
        now = datetime.now(UTC)
        outcome = TokenOutcome(
            outcome_id="out-1",
            run_id="run-1",
            token_id="tok-1",
            outcome=TerminalOutcome.SUCCESS,
            path=TerminalPath.DEFAULT_FLOW,
            completed=True,
            recorded_at=now,
            sink_name="output",
        )
        assert outcome.outcome_id == "out-1"
        assert outcome.run_id == "run-1"
        assert outcome.token_id == "tok-1"
        assert outcome.outcome == TerminalOutcome.SUCCESS
        assert outcome.path == TerminalPath.DEFAULT_FLOW
        assert outcome.completed is True
        assert outcome.recorded_at == now

    def test_token_outcome_completed_for_success(self) -> None:
        """SUCCESS/DEFAULT_FLOW outcome is completed (no further processing)."""
        outcome = TokenOutcome(
            outcome_id="out-1",
            run_id="run-1",
            token_id="tok-1",
            outcome=TerminalOutcome.SUCCESS,
            path=TerminalPath.DEFAULT_FLOW,
            completed=True,
            recorded_at=datetime.now(UTC),
            sink_name="output",
        )
        assert outcome.completed is True

    def test_token_outcome_buffered_is_not_completed(self) -> None:
        """BUFFERED path is NOT completed (waiting for aggregation)."""
        outcome = TokenOutcome(
            outcome_id="out-1",
            run_id="run-1",
            token_id="tok-1",
            outcome=None,
            path=TerminalPath.BUFFERED,
            completed=False,
            recorded_at=datetime.now(UTC),
            batch_id="batch-123",  # BUFFERED has batch context
        )
        assert outcome.completed is False
        assert outcome.batch_id == "batch-123"

    def test_token_outcome_routed_with_sink_name(self) -> None:
        """GATE_ROUTED path includes sink_name for traceability."""
        outcome = TokenOutcome(
            outcome_id="out-1",
            run_id="run-1",
            token_id="tok-1",
            outcome=TerminalOutcome.SUCCESS,
            path=TerminalPath.GATE_ROUTED,
            completed=True,
            recorded_at=datetime.now(UTC),
            sink_name="quarantine_sink",
        )
        assert outcome.outcome == TerminalOutcome.SUCCESS
        assert outcome.path == TerminalPath.GATE_ROUTED
        assert outcome.sink_name == "quarantine_sink"

    def test_token_outcome_forked_with_fork_group(self) -> None:
        """FORK_PARENT path includes fork_group_id for lineage tracking."""
        outcome = TokenOutcome(
            outcome_id="out-1",
            run_id="run-1",
            token_id="tok-1",
            outcome=TerminalOutcome.TRANSIENT,
            path=TerminalPath.FORK_PARENT,
            completed=True,
            recorded_at=datetime.now(UTC),
            fork_group_id="fork-456",
        )
        assert outcome.outcome == TerminalOutcome.TRANSIENT
        assert outcome.path == TerminalPath.FORK_PARENT
        assert outcome.fork_group_id == "fork-456"

    def test_token_outcome_error_with_hash(self) -> None:
        """QUARANTINED_AT_SOURCE outcomes can have error_hash."""
        outcome = TokenOutcome(
            outcome_id="out-1",
            run_id="run-1",
            token_id="tok-1",
            outcome=TerminalOutcome.FAILURE,
            path=TerminalPath.QUARANTINED_AT_SOURCE,
            completed=True,
            recorded_at=datetime.now(UTC),
            error_hash="err_abc123",
            context_json='{"reason": "validation_failed"}',
        )
        assert outcome.outcome == TerminalOutcome.FAILURE
        assert outcome.path == TerminalPath.QUARANTINED_AT_SOURCE
        assert outcome.error_hash == "err_abc123"
        assert outcome.context_json == '{"reason": "validation_failed"}'

    # NOTE: Frozen immutability tested in TestFrozenDataclassImmutability


class TestTokenOutcomeTwoAxis:
    """ADR-019 Phase 1: TokenOutcome carries (outcome, path, completed)."""

    def test_completed_outcome_has_outcome_path_completed(self) -> None:
        """A completed-state TokenOutcome has all three two-axis fields."""
        record = TokenOutcome(
            outcome_id="out_test_01",
            run_id="run_001",
            token_id="tok_001",
            outcome=TerminalOutcome.SUCCESS,
            path=TerminalPath.DEFAULT_FLOW,
            completed=True,
            recorded_at=datetime.now(UTC),
            sink_name="primary",
        )
        assert record.outcome == TerminalOutcome.SUCCESS
        assert record.path == TerminalPath.DEFAULT_FLOW
        assert record.completed is True

    def test_buffered_outcome_has_null_outcome_buffered_path(self) -> None:
        """A non-terminal TokenOutcome has outcome=None, path=BUFFERED, completed=False."""
        record = TokenOutcome(
            outcome_id="out_test_02",
            run_id="run_001",
            token_id="tok_001",
            outcome=None,
            path=TerminalPath.BUFFERED,
            completed=False,
            recorded_at=datetime.now(UTC),
            batch_id="batch_001",
        )
        assert record.outcome is None
        assert record.path == TerminalPath.BUFFERED
        assert record.completed is False

    def test_completed_xor_outcome_invariant_completed_true_outcome_none(self) -> None:
        """Tier 1: completed=True with outcome=None is an invariant violation."""
        with pytest.raises(ValueError, match="completed"):
            TokenOutcome(
                outcome_id="out_test_03",
                run_id="run_001",
                token_id="tok_001",
                outcome=None,
                path=TerminalPath.DEFAULT_FLOW,
                completed=True,
                recorded_at=datetime.now(UTC),
            )

    def test_completed_xor_outcome_invariant_completed_false_outcome_set(self) -> None:
        """Tier 1: completed=False with outcome=SUCCESS is an invariant violation."""
        with pytest.raises(ValueError, match="completed"):
            TokenOutcome(
                outcome_id="out_test_04",
                run_id="run_001",
                token_id="tok_001",
                outcome=TerminalOutcome.SUCCESS,
                path=TerminalPath.BUFFERED,
                completed=False,
                recorded_at=datetime.now(UTC),
            )

    def test_legal_pair_required(self) -> None:
        """Tier 1: an unknown (outcome, path) pair is an invariant violation."""
        with pytest.raises(ValueError, match="_LEGAL_TERMINAL_PAIRS"):
            TokenOutcome(
                outcome_id="out_test_05",
                run_id="run_001",
                token_id="tok_001",
                outcome=TerminalOutcome.SUCCESS,
                path=TerminalPath.UNROUTED,
                completed=True,
                recorded_at=datetime.now(UTC),
            )

    @given(
        outcome=st.sampled_from(list(TerminalOutcome)),
        path=st.sampled_from(list(TerminalPath)),
    )
    def test_all_illegal_completed_pairs_rejected(
        self,
        outcome: TerminalOutcome,
        path: TerminalPath,
    ) -> None:
        """Tier 1: every pair outside _LEGAL_TERMINAL_PAIRS is rejected."""
        if (outcome, path) in _LEGAL_TERMINAL_PAIRS:
            return

        with pytest.raises(ValueError):
            TokenOutcome(
                outcome_id="out_prop_illegal",
                run_id="run_001",
                token_id="tok_001",
                outcome=outcome,
                path=path,
                completed=True,
                recorded_at=datetime.now(UTC),
            )

    @given(pair=st.sampled_from(list(_LEGAL_TERMINAL_PAIRS)))
    def test_all_legal_pairs_accepted(
        self,
        pair: tuple[TerminalOutcome, TerminalPath],
    ) -> None:
        """Tier 1: every closed-set legal pair is accepted at the shape layer."""
        outcome, path = pair
        TokenOutcome(
            outcome_id="out_prop_legal",
            run_id="run_001",
            token_id="tok_001",
            outcome=outcome,
            path=path,
            completed=True,
            recorded_at=datetime.now(UTC),
        )


class TestNonCanonicalMetadata:
    """Tests for NonCanonicalMetadata audit model.

    Captures metadata for data that cannot be canonically serialized
    (NaN, Infinity, non-dict types). Part of Tier-3 boundary handling.
    """

    def test_create_non_canonical_metadata(self) -> None:
        """NonCanonicalMetadata captures why canonicalization failed."""
        meta = NonCanonicalMetadata(
            repr_value="{'value': nan}",
            type_name="dict",
            canonical_error="NaN not JSON serializable",
        )
        assert meta.repr_value == "{'value': nan}"
        assert meta.type_name == "dict"
        assert meta.canonical_error == "NaN not JSON serializable"

    def test_to_dict_produces_expected_keys(self) -> None:
        """to_dict() produces dict with __ prefixed keys for storage."""
        meta = NonCanonicalMetadata(
            repr_value="{'x': inf}",
            type_name="dict",
            canonical_error="Infinity not allowed in canonical JSON",
        )
        d = meta.to_dict()
        assert d["__repr__"] == "{'x': inf}"
        assert d["__type__"] == "dict"
        assert d["__canonical_error__"] == "Infinity not allowed in canonical JSON"

    def test_from_error_factory_method(self) -> None:
        """from_error() creates metadata from exception context."""
        data = {"value": float("nan")}
        error = ValueError("NaN is not allowed in canonical JSON")
        meta = NonCanonicalMetadata.from_error(data, error)

        # repr captures the data structure
        assert "nan" in meta.repr_value.lower()
        # type_name is correct
        assert meta.type_name == "dict"
        # error message is captured
        assert "not allowed" in meta.canonical_error

    def test_from_error_with_non_dict_type(self) -> None:
        """from_error() handles non-dict types correctly."""
        data = [1, 2, float("inf")]
        error = TypeError("list with Infinity cannot be canonical")
        meta = NonCanonicalMetadata.from_error(data, error)

        assert meta.type_name == "list"
        assert "inf" in meta.repr_value.lower()

    # NOTE: Frozen immutability tested in TestFrozenDataclassImmutability


class TestValidationErrorRecord:
    """Tests for ValidationErrorRecord audit model.

    Created when a source row fails schema validation.
    These are operational errors (bad user data), not system bugs.
    """

    def test_create_validation_error_record(self) -> None:
        """ValidationErrorRecord captures source validation failures."""
        now = datetime.now(UTC)
        record = ValidationErrorRecord(
            error_id="err-123",
            run_id="run-456",
            node_id="source-node-1",
            row_hash="abc123",
            error="Field 'amount' expected int, got str",
            schema_mode="fixed",
            destination="quarantine",
            created_at=now,
        )
        assert record.error_id == "err-123"
        assert record.run_id == "run-456"
        assert record.node_id == "source-node-1"
        assert record.row_hash == "abc123"
        assert record.error == "Field 'amount' expected int, got str"
        assert record.schema_mode == "fixed"
        assert record.destination == "quarantine"
        assert record.created_at == now

    def test_validation_error_with_row_data(self) -> None:
        """ValidationErrorRecord can include raw row data for debugging."""
        record = ValidationErrorRecord(
            error_id="err-123",
            run_id="run-456",
            node_id="source-node-1",
            row_hash="abc123",
            error="Missing required field 'id'",
            schema_mode="fixed",
            destination="quarantine",
            created_at=datetime.now(UTC),
            row_data_json='{"name": "test", "amount": "invalid"}',
        )
        assert record.row_data_json == '{"name": "test", "amount": "invalid"}'

    def test_validation_error_node_id_nullable(self) -> None:
        """ValidationErrorRecord.node_id can be None (pre-node validation)."""
        record = ValidationErrorRecord(
            error_id="err-123",
            run_id="run-456",
            node_id=None,  # Error before node assignment
            row_hash="abc123",
            error="Invalid CSV format",
            schema_mode="observed",
            destination="quarantine",
            created_at=datetime.now(UTC),
        )
        assert record.node_id is None


class TestTransformErrorRecord:
    """Tests for TransformErrorRecord audit model.

    Created when a transform returns TransformResult.error().
    These are operational errors (bad data values), not transform bugs.
    """

    def test_create_transform_error_record(self) -> None:
        """TransformErrorRecord captures transform processing errors."""
        now = datetime.now(UTC)
        record = TransformErrorRecord(
            error_id="terr-123",
            run_id="run-456",
            token_id="tok-789",
            transform_id="transform-node-1",
            row_hash="abc123",
            destination="error_sink",
            created_at=now,
        )
        assert record.error_id == "terr-123"
        assert record.run_id == "run-456"
        assert record.token_id == "tok-789"
        assert record.transform_id == "transform-node-1"
        assert record.row_hash == "abc123"
        assert record.destination == "error_sink"
        assert record.created_at == now

    def test_transform_error_with_details(self) -> None:
        """TransformErrorRecord can include error details and row data."""
        record = TransformErrorRecord(
            error_id="terr-123",
            run_id="run-456",
            token_id="tok-789",
            transform_id="transform-node-1",
            row_hash="abc123",
            destination="error_sink",
            created_at=datetime.now(UTC),
            row_data_json='{"id": 1, "amount": 0}',
            error_details_json='{"reason": "division_by_zero", "field": "amount"}',
        )
        assert record.row_data_json == '{"id": 1, "amount": 0}'
        assert record.error_details_json == '{"reason": "division_by_zero", "field": "amount"}'


# =============================================================================
# HASH VALIDATION TESTS - P0 FIXES FROM QUALITY AUDIT
# =============================================================================


class TestHashFields:
    """Tests for hash field format and integrity across audit contracts.

    Hash fields are critical for audit integrity - they must follow consistent
    format and be verifiable. All hashes are SHA-256 (64 hex characters).
    """

    # Valid SHA-256 hash examples (64 lowercase hex chars)
    VALID_SHA256 = "a" * 64
    VALID_SHA256_REAL = "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"

    # Invalid hash formats
    INVALID_TOO_SHORT = "abc123"
    INVALID_UPPERCASE = "A" * 64
    INVALID_NON_HEX = "g" * 64  # 'g' not in hex

    def test_run_config_hash_accepts_valid_format(self) -> None:
        """Run.config_hash accepts valid SHA-256 format."""
        run = Run(
            run_id="run-1",
            started_at=datetime.now(UTC),
            config_hash=self.VALID_SHA256_REAL,
            settings_json="{}",
            canonical_version="1.0",
            status=RunStatus.RUNNING,
        )
        assert len(run.config_hash) == 64
        assert run.config_hash == self.VALID_SHA256_REAL

    def test_row_source_data_hash_present(self) -> None:
        """Row.source_data_hash is required for audit integrity."""
        row = Row(
            row_id="row-1",
            run_id="run-1",
            source_node_id="node-1",
            row_index=0,
            source_row_index=0,
            ingest_sequence=0,
            source_data_hash=self.VALID_SHA256,
            created_at=datetime.now(UTC),
        )
        assert len(row.source_data_hash) == 64

    def test_node_hashes_consistent_format(self) -> None:
        """Node config_hash and schema_hash follow consistent format."""
        node = Node(
            node_id="node-1",
            run_id="run-1",
            plugin_name="csv",
            node_type=NodeType.SOURCE,
            plugin_version="1.0",
            determinism=Determinism.DETERMINISTIC,
            config_hash=self.VALID_SHA256,
            config_json="{}",
            registered_at=datetime.now(UTC),
            schema_hash=self.VALID_SHA256_REAL,
        )
        assert len(node.config_hash) == 64
        assert len(node.schema_hash) == 64  # type: ignore[arg-type] # schema_hash may be None

    def test_node_state_input_hash_present(self) -> None:
        """NodeState variants require input_hash for traceability."""
        now = datetime.now(UTC)

        # OPEN state
        open_state = NodeStateOpen(
            state_id="state-1",
            token_id="tok-1",
            node_id="node-1",
            step_index=0,
            attempt=1,
            status=NodeStateStatus.OPEN,
            input_hash=self.VALID_SHA256,
            started_at=now,
        )
        assert len(open_state.input_hash) == 64

        # COMPLETED state - also has output_hash
        completed_state = NodeStateCompleted(
            state_id="state-2",
            token_id="tok-1",
            node_id="node-1",
            step_index=0,
            attempt=1,
            status=NodeStateStatus.COMPLETED,
            input_hash=self.VALID_SHA256,
            output_hash=self.VALID_SHA256_REAL,
            started_at=now,
            completed_at=now,
            duration_ms=100.0,
        )
        assert len(completed_state.input_hash) == 64
        assert len(completed_state.output_hash) == 64

    def test_call_request_response_hashes(self) -> None:
        """Call.request_hash and response_hash follow hash format."""
        call = Call(
            call_id="call-1",
            state_id="state-1",
            call_index=0,
            call_type=CallType.LLM,
            status=CallStatus.SUCCESS,
            request_hash=self.VALID_SHA256,
            created_at=datetime.now(UTC),
            response_hash=self.VALID_SHA256_REAL,
        )
        assert len(call.request_hash) == 64
        assert len(call.response_hash) == 64  # type: ignore[arg-type]

    def test_artifact_content_hash_required(self) -> None:
        """Artifact.content_hash is required for integrity verification."""
        artifact = Artifact(
            artifact_id="art-1",
            run_id="run-1",
            produced_by_state_id="state-1",
            sink_node_id="sink-1",
            artifact_type="csv",
            path_or_uri="/tmp/output.csv",
            content_hash=self.VALID_SHA256,
            size_bytes=1024,
            created_at=datetime.now(UTC),
        )
        assert len(artifact.content_hash) == 64

    def test_error_records_have_row_hash(self) -> None:
        """Error records capture row hash for traceability."""
        validation_err = ValidationErrorRecord(
            error_id="err-1",
            run_id="run-1",
            node_id="node-1",
            row_hash=self.VALID_SHA256,
            error="test error",
            schema_mode="fixed",
            destination="quarantine",
            created_at=datetime.now(UTC),
        )
        assert len(validation_err.row_hash) == 64

        transform_err = TransformErrorRecord(
            error_id="terr-1",
            run_id="run-1",
            token_id="tok-1",
            transform_id="transform-1",
            row_hash=self.VALID_SHA256_REAL,
            destination="error_sink",
            created_at=datetime.now(UTC),
        )
        assert len(transform_err.row_hash) == 64


# =============================================================================
# NEGATIVE VALIDATION TESTS - P0 FIXES FROM QUALITY AUDIT
# =============================================================================


class TestRequiredFieldValidation:
    """Tests that verify required fields are enforced.

    Per audit findings: 98% of tests had no negative cases.
    These tests verify that missing required fields raise TypeError.
    """

    def test_run_requires_run_id(self) -> None:
        """Run.run_id is required - TypeError on missing."""
        with pytest.raises(TypeError, match="run_id"):
            Run(  # type: ignore[call-arg]
                # run_id missing
                started_at=datetime.now(UTC),
                config_hash="a" * 64,
                settings_json="{}",
                canonical_version="1.0",
                status=RunStatus.RUNNING,
            )

    def test_run_requires_started_at(self) -> None:
        """Run.started_at is required - TypeError on missing."""
        with pytest.raises(TypeError, match="started_at"):
            Run(  # type: ignore[call-arg]
                run_id="run-1",
                # started_at missing
                config_hash="a" * 64,
                settings_json="{}",
                canonical_version="1.0",
                status=RunStatus.RUNNING,
            )

    def test_run_requires_status(self) -> None:
        """Run.status is required - TypeError on missing."""
        with pytest.raises(TypeError, match="status"):
            Run(  # type: ignore[call-arg]
                run_id="run-1",
                started_at=datetime.now(UTC),
                config_hash="a" * 64,
                settings_json="{}",
                canonical_version="1.0",
                # status missing
            )

    def test_node_requires_node_type(self) -> None:
        """Node.node_type is required - TypeError on missing."""
        with pytest.raises(TypeError, match="node_type"):
            Node(  # type: ignore[call-arg]
                node_id="n1",
                run_id="r1",
                plugin_name="test",
                # node_type missing
                plugin_version="1.0",
                determinism=Determinism.DETERMINISTIC,
                config_hash="a" * 64,
                config_json="{}",
                registered_at=datetime.now(UTC),
            )

    def test_row_requires_source_data_hash(self) -> None:
        """Row.source_data_hash is required for audit integrity."""
        with pytest.raises(TypeError, match="source_data_hash"):
            Row(  # type: ignore[call-arg]
                row_id="row-1",
                run_id="run-1",
                source_node_id="node-1",
                row_index=0,
                # source_data_hash missing - critical for audit!
                created_at=datetime.now(UTC),
            )

    def test_token_requires_token_id(self) -> None:
        """Token.token_id is required."""
        with pytest.raises(TypeError, match="token_id"):
            Token(  # type: ignore[call-arg]
                # token_id missing
                row_id="row-1",
                created_at=datetime.now(UTC),
                run_id="run-1",
            )

    def test_node_state_open_requires_input_hash(self) -> None:
        """NodeStateOpen.input_hash is required for traceability."""
        with pytest.raises(TypeError, match="input_hash"):
            NodeStateOpen(  # type: ignore[call-arg]
                state_id="s1",
                token_id="t1",
                node_id="n1",
                step_index=0,
                attempt=1,
                status=NodeStateStatus.OPEN,
                # input_hash missing - critical for audit!
                started_at=datetime.now(UTC),
            )

    def test_node_state_completed_requires_output_hash(self) -> None:
        """NodeStateCompleted.output_hash is required."""
        now = datetime.now(UTC)
        with pytest.raises(TypeError, match="output_hash"):
            NodeStateCompleted(  # type: ignore[call-arg]
                state_id="s1",
                token_id="t1",
                node_id="n1",
                step_index=0,
                attempt=1,
                status=NodeStateStatus.COMPLETED,
                input_hash="a" * 64,
                # output_hash missing - required for COMPLETED state
                started_at=now,
                completed_at=now,
                duration_ms=100.0,
            )

    def test_token_outcome_requires_outcome_enum(self) -> None:
        """TokenOutcome.outcome is required."""
        with pytest.raises(TypeError, match="outcome"):
            TokenOutcome(  # type: ignore[call-arg]
                outcome_id="o1",
                run_id="r1",
                token_id="t1",
                # outcome missing
                path=TerminalPath.DEFAULT_FLOW,
                completed=True,
                recorded_at=datetime.now(UTC),
            )

    def test_artifact_requires_content_hash(self) -> None:
        """Artifact.content_hash is required for integrity."""
        with pytest.raises(TypeError, match="content_hash"):
            Artifact(  # type: ignore[call-arg]
                artifact_id="a1",
                run_id="r1",
                produced_by_state_id="s1",
                sink_node_id="sink-1",
                artifact_type="csv",
                path_or_uri="/tmp/out.csv",
                # content_hash missing - critical for audit!
                size_bytes=1024,
                created_at=datetime.now(UTC),
            )

    def test_call_requires_request_hash(self) -> None:
        """Call.request_hash is required for traceability."""
        with pytest.raises(TypeError, match="request_hash"):
            Call(  # type: ignore[call-arg]
                call_id="c1",
                state_id="s1",
                call_index=0,
                call_type=CallType.LLM,
                status=CallStatus.SUCCESS,
                # request_hash missing
                created_at=datetime.now(UTC),
            )

    def test_non_canonical_metadata_requires_all_fields(self) -> None:
        """NonCanonicalMetadata requires repr_value, type_name, canonical_error."""
        with pytest.raises(TypeError, match="repr_value"):
            NonCanonicalMetadata(  # type: ignore[call-arg]
                # repr_value missing
                type_name="dict",
                canonical_error="error",
            )

        with pytest.raises(TypeError, match="type_name"):
            NonCanonicalMetadata(  # type: ignore[call-arg]
                repr_value="{}",
                # type_name missing
                canonical_error="error",
            )

        with pytest.raises(TypeError, match="canonical_error"):
            NonCanonicalMetadata(  # type: ignore[call-arg]
                repr_value="{}",
                type_name="dict",
                # canonical_error missing
            )


# =============================================================================
# PROPERTY-BASED TESTS USING HYPOTHESIS - P0 FIXES FROM QUALITY AUDIT
# =============================================================================

# Custom strategies for common audit field types
sha256_hashes = st.text(
    alphabet="0123456789abcdef",
    min_size=64,
    max_size=64,
)

valid_ids = st.text(
    alphabet="abcdefghijklmnopqrstuvwxyz0123456789-_",
    min_size=1,
    max_size=50,
)

valid_json = st.one_of(
    st.just("{}"),
    st.just('{"key": "value"}'),
    st.just('{"nested": {"a": 1}}'),
)

positive_ints = st.integers(min_value=0, max_value=1_000_000)
positive_floats = st.floats(min_value=0.0, max_value=1_000_000.0, allow_nan=False, allow_infinity=False)


class TestPropertyBasedAuditContracts:
    """Property-based tests for audit contracts using Hypothesis.

    These tests verify invariants hold across a wide range of inputs,
    catching edge cases that example-based tests might miss.
    """

    @given(
        run_id=valid_ids,
        config_hash=sha256_hashes,
        settings_json=valid_json,
        canonical_version=st.text(min_size=1, max_size=10),
    )
    @settings(max_examples=50)
    def test_run_accepts_valid_inputs(
        self,
        run_id: str,
        config_hash: str,
        settings_json: str,
        canonical_version: str,
    ) -> None:
        """Run accepts any valid combination of inputs."""
        run = Run(
            run_id=run_id,
            started_at=datetime.now(UTC),
            config_hash=config_hash,
            settings_json=settings_json,
            canonical_version=canonical_version,
            status=RunStatus.RUNNING,
        )
        assert run.run_id == run_id
        assert len(run.config_hash) == 64

    @given(
        row_id=valid_ids,
        run_id=valid_ids,
        source_node_id=valid_ids,
        row_index=positive_ints,
        source_row_index=positive_ints,
        ingest_sequence=positive_ints,
        source_data_hash=sha256_hashes,
    )
    @settings(max_examples=50)
    def test_row_accepts_valid_inputs(
        self,
        row_id: str,
        run_id: str,
        source_node_id: str,
        row_index: int,
        source_row_index: int,
        ingest_sequence: int,
        source_data_hash: str,
    ) -> None:
        """Row accepts any valid combination of inputs."""
        row = Row(
            row_id=row_id,
            run_id=run_id,
            source_node_id=source_node_id,
            row_index=row_index,
            source_row_index=source_row_index,
            ingest_sequence=ingest_sequence,
            source_data_hash=source_data_hash,
            created_at=datetime.now(UTC),
        )
        assert row.row_id == row_id
        assert row.row_index == row_index
        assert len(row.source_data_hash) == 64

    @given(
        token_id=valid_ids,
        row_id=valid_ids,
    )
    @settings(max_examples=50)
    def test_token_accepts_valid_inputs(
        self,
        token_id: str,
        row_id: str,
    ) -> None:
        """Token accepts any valid combination of inputs."""
        token = Token(
            token_id=token_id,
            row_id=row_id,
            created_at=datetime.now(UTC),
            run_id="run-hyp",
        )
        assert token.token_id == token_id
        assert token.row_id == row_id

    @given(
        state_id=valid_ids,
        token_id=valid_ids,
        node_id=valid_ids,
        step_index=positive_ints,
        attempt=st.integers(min_value=1, max_value=100),
        input_hash=sha256_hashes,
        output_hash=sha256_hashes,
        duration_ms=positive_floats,
    )
    @settings(max_examples=50)
    def test_node_state_completed_accepts_valid_inputs(
        self,
        state_id: str,
        token_id: str,
        node_id: str,
        step_index: int,
        attempt: int,
        input_hash: str,
        output_hash: str,
        duration_ms: float,
    ) -> None:
        """NodeStateCompleted accepts valid combinations."""
        now = datetime.now(UTC)
        state = NodeStateCompleted(
            state_id=state_id,
            token_id=token_id,
            node_id=node_id,
            step_index=step_index,
            attempt=attempt,
            status=NodeStateStatus.COMPLETED,
            input_hash=input_hash,
            output_hash=output_hash,
            started_at=now,
            completed_at=now,
            duration_ms=duration_ms,
        )
        assert state.state_id == state_id
        assert len(state.input_hash) == 64
        assert len(state.output_hash) == 64

    @given(
        outcome_id=valid_ids,
        run_id=valid_ids,
        token_id=valid_ids,
        pair=st.sampled_from(list(_LEGAL_TERMINAL_PAIRS)),
    )
    @settings(max_examples=50)
    def test_token_outcome_accepts_valid_inputs(
        self,
        outcome_id: str,
        run_id: str,
        token_id: str,
        pair: tuple[TerminalOutcome, TerminalPath],
    ) -> None:
        """TokenOutcome accepts valid combinations."""
        outcome, path = pair
        token_outcome = TokenOutcome(
            outcome_id=outcome_id,
            run_id=run_id,
            token_id=token_id,
            outcome=outcome,
            path=path,
            completed=True,
            recorded_at=datetime.now(UTC),
        )
        assert token_outcome.outcome == outcome
        assert token_outcome.path == path
        assert token_outcome.completed is True

    @given(
        repr_value=st.text(min_size=1, max_size=200),
        type_name=st.sampled_from(["dict", "list", "str", "int", "float", "NoneType"]),
        canonical_error=st.text(min_size=1, max_size=200),
    )
    @settings(max_examples=50)
    def test_non_canonical_metadata_round_trip(
        self,
        repr_value: str,
        type_name: str,
        canonical_error: str,
    ) -> None:
        """NonCanonicalMetadata to_dict() preserves all fields."""
        meta = NonCanonicalMetadata(
            repr_value=repr_value,
            type_name=type_name,
            canonical_error=canonical_error,
        )
        d = meta.to_dict()
        assert d["__repr__"] == repr_value
        assert d["__type__"] == type_name
        assert d["__canonical_error__"] == canonical_error

    @given(
        artifact_id=valid_ids,
        run_id=valid_ids,
        state_id=valid_ids,
        sink_node_id=valid_ids,
        artifact_type=st.sampled_from(["csv", "json", "parquet", "pickle"]),
        path_or_uri=st.text(min_size=1, max_size=200),
        content_hash=sha256_hashes,
        size_bytes=positive_ints,
    )
    @settings(max_examples=50)
    def test_artifact_accepts_valid_inputs(
        self,
        artifact_id: str,
        run_id: str,
        state_id: str,
        sink_node_id: str,
        artifact_type: str,
        path_or_uri: str,
        content_hash: str,
        size_bytes: int,
    ) -> None:
        """Artifact accepts valid combinations."""
        artifact = Artifact(
            artifact_id=artifact_id,
            run_id=run_id,
            produced_by_state_id=state_id,
            sink_node_id=sink_node_id,
            artifact_type=artifact_type,
            path_or_uri=path_or_uri,
            content_hash=content_hash,
            size_bytes=size_bytes,
            created_at=datetime.now(UTC),
        )
        assert artifact.artifact_id == artifact_id
        assert len(artifact.content_hash) == 64
        assert artifact.size_bytes >= 0


class TestPropertyBasedHashInvariants:
    """Property-based tests for hash field invariants."""

    @given(hash_value=sha256_hashes)
    @settings(max_examples=100)
    def test_hash_is_always_64_chars(self, hash_value: str) -> None:
        """All generated hashes are exactly 64 characters."""
        assert len(hash_value) == 64

    @given(hash_value=sha256_hashes)
    @settings(max_examples=100)
    def test_hash_is_always_lowercase_hex(self, hash_value: str) -> None:
        """All generated hashes contain only lowercase hex characters."""
        assert all(c in "0123456789abcdef" for c in hash_value)

    @given(
        hash1=sha256_hashes,
        hash2=sha256_hashes,
    )
    @settings(max_examples=50)
    def test_different_hashes_are_distinguishable(
        self,
        hash1: str,
        hash2: str,
    ) -> None:
        """Different hash values can be compared."""
        # Even if they happen to be equal, comparison works
        result = hash1 == hash2
        assert isinstance(result, bool)


class TestOperation:
    """Tests for Operation audit model validation."""

    def test_create_operation_with_valid_values(self) -> None:
        """Operation accepts valid operation_type and status literals."""
        operation = Operation(
            operation_id="op-123",
            run_id="run-123",
            node_id="node-123",
            operation_type="source_load",
            started_at=datetime.now(UTC),
            status="open",
        )

        assert operation.operation_type == "source_load"
        assert operation.status == "open"

    def test_rejects_invalid_operation_type(self) -> None:
        """Operation crashes on invalid operation_type values."""
        with pytest.raises(ValueError, match="operation_type"):
            Operation(
                operation_id="op-123",
                run_id="run-123",
                node_id="node-123",
                operation_type="bad_type",  # type: ignore[arg-type]
                started_at=datetime.now(UTC),
                status="open",
            )

    def test_rejects_invalid_status(self) -> None:
        """Operation crashes on invalid status values."""
        with pytest.raises(ValueError, match="status"):
            Operation(
                operation_id="op-123",
                run_id="run-123",
                node_id="node-123",
                operation_type="sink_write",
                started_at=datetime.now(UTC),
                status="oops",  # type: ignore[arg-type]
            )

    # --- Lifecycle invariant tests (status-dependent field validation) ---

    def test_open_rejects_completed_at(self) -> None:
        """Status 'open' must not have completed_at set."""
        with pytest.raises(ValueError, match="status='open' but completed_at is set"):
            Operation(
                operation_id="op-1",
                run_id="run-1",
                node_id="node-1",
                operation_type="source_load",
                started_at=datetime.now(UTC),
                status="open",
                completed_at=datetime.now(UTC),
            )

    def test_open_rejects_duration_ms(self) -> None:
        """Status 'open' must not have duration_ms set."""
        with pytest.raises(ValueError, match="status='open' but duration_ms is set"):
            Operation(
                operation_id="op-1",
                run_id="run-1",
                node_id="node-1",
                operation_type="source_load",
                started_at=datetime.now(UTC),
                status="open",
                duration_ms=42.0,
            )

    def test_open_rejects_error_message(self) -> None:
        """Status 'open' must not have error_message set."""
        with pytest.raises(ValueError, match="status='open' but error_message is set"):
            Operation(
                operation_id="op-1",
                run_id="run-1",
                node_id="node-1",
                operation_type="source_load",
                started_at=datetime.now(UTC),
                status="open",
                error_message="something went wrong",
            )

    @pytest.mark.parametrize("status", ["completed", "failed", "pending"])
    def test_terminal_status_requires_completed_at(self, status: str) -> None:
        """Terminal statuses must have completed_at set."""
        kwargs: dict[str, Any] = {
            "operation_id": "op-1",
            "run_id": "run-1",
            "node_id": "node-1",
            "operation_type": "sink_write",
            "started_at": datetime.now(UTC),
            "status": status,
            "completed_at": None,
            "duration_ms": 100.0,
        }
        if status == "failed":
            kwargs["error_message"] = "error"
        with pytest.raises(ValueError, match="completed_at is None"):
            Operation(**kwargs)

    @pytest.mark.parametrize("status", ["completed", "failed", "pending"])
    def test_terminal_status_requires_duration_ms(self, status: str) -> None:
        """Terminal statuses must have duration_ms set."""
        kwargs: dict[str, Any] = {
            "operation_id": "op-1",
            "run_id": "run-1",
            "node_id": "node-1",
            "operation_type": "sink_write",
            "started_at": datetime.now(UTC),
            "status": status,
            "completed_at": datetime.now(UTC),
            "duration_ms": None,
        }
        if status == "failed":
            kwargs["error_message"] = "error"
        with pytest.raises(ValueError, match="duration_ms is None"):
            Operation(**kwargs)

    def test_failed_requires_error_message(self) -> None:
        """Status 'failed' must have error_message set."""
        with pytest.raises(ValueError, match="status='failed' but error_message is None"):
            Operation(
                operation_id="op-1",
                run_id="run-1",
                node_id="node-1",
                operation_type="source_load",
                started_at=datetime.now(UTC),
                status="failed",
                completed_at=datetime.now(UTC),
                duration_ms=100.0,
                error_message=None,
            )

    def test_failed_rejects_empty_error_message(self) -> None:
        """Status 'failed' must not accept blank error messages."""
        with pytest.raises(ValueError, match="error_message must not be empty"):
            Operation(
                operation_id="op-1",
                run_id="run-1",
                node_id="node-1",
                operation_type="source_load",
                started_at=datetime.now(UTC),
                status="failed",
                completed_at=datetime.now(UTC),
                duration_ms=100.0,
                error_message="",
            )

    def test_completed_rejects_error_message(self) -> None:
        """Status 'completed' must not have error_message set."""
        with pytest.raises(ValueError, match="status='completed' but error_message is set"):
            Operation(
                operation_id="op-1",
                run_id="run-1",
                node_id="node-1",
                operation_type="sink_write",
                started_at=datetime.now(UTC),
                status="completed",
                completed_at=datetime.now(UTC),
                duration_ms=50.0,
                error_message="should not be here",
            )

    def test_valid_completed_operation(self) -> None:
        """A properly formed 'completed' operation passes all invariants."""
        op = Operation(
            operation_id="op-1",
            run_id="run-1",
            node_id="node-1",
            operation_type="sink_write",
            started_at=datetime.now(UTC),
            status="completed",
            completed_at=datetime.now(UTC),
            duration_ms=50.0,
        )
        assert op.status == "completed"
        assert op.error_message is None

    def test_valid_failed_operation(self) -> None:
        """A properly formed 'failed' operation passes all invariants."""
        op = Operation(
            operation_id="op-1",
            run_id="run-1",
            node_id="node-1",
            operation_type="source_load",
            started_at=datetime.now(UTC),
            status="failed",
            completed_at=datetime.now(UTC),
            duration_ms=200.0,
            error_message="connection refused",
        )
        assert op.status == "failed"
        assert op.error_message == "connection refused"

    def test_valid_pending_operation(self) -> None:
        """A properly formed 'pending' operation passes all invariants."""
        op = Operation(
            operation_id="op-1",
            run_id="run-1",
            node_id="node-1",
            operation_type="sink_write",
            started_at=datetime.now(UTC),
            status="pending",
            completed_at=datetime.now(UTC),
            duration_ms=300.0,
        )
        assert op.status == "pending"


# ---------------------------------------------------------------------------
# SecretResolutionInput — write-side validation (__post_init__)
# ---------------------------------------------------------------------------


class TestSecretResolutionInputValidation:
    """Tests for SecretResolutionInput __post_init__ write-side validation.

    SecretResolutionInput is the write-side DTO for Key Vault audit records.
    Validation at construction ensures only well-formed records enter the
    audit trail (Tier 1 crash-on-anomaly principle).
    """

    @staticmethod
    def _valid_kwargs(**overrides: object) -> dict[str, object]:
        """Build valid SecretResolutionInput kwargs with optional overrides."""
        defaults: dict[str, object] = {
            "env_var_name": "API_KEY",
            "source": "keyvault",
            "vault_url": "https://vault.example.com",
            "secret_name": "api-key",
            "timestamp": 1709100000.0,
            "resolution_latency_ms": 42.5,
            "fingerprint": "a" * 64,
        }
        defaults.update(overrides)
        return defaults

    def test_valid_construction_succeeds(self) -> None:
        """Happy path: valid inputs create the dataclass without error."""
        sri = SecretResolutionInput(**self._valid_kwargs())  # type: ignore[arg-type]
        assert sri.env_var_name == "API_KEY"
        assert sri.fingerprint == "a" * 64

    def test_empty_env_var_name_rejected(self) -> None:
        with pytest.raises(ValueError, match="env_var_name is required"):
            SecretResolutionInput(**self._valid_kwargs(env_var_name=""))  # type: ignore[arg-type]

    def test_invalid_source_rejected(self) -> None:
        with pytest.raises(ValueError, match="source must be one of"):
            SecretResolutionInput(**self._valid_kwargs(source="s3"))  # type: ignore[arg-type]

    def test_empty_source_rejected(self) -> None:
        with pytest.raises(ValueError, match="source must be one of"):
            SecretResolutionInput(**self._valid_kwargs(source=""))  # type: ignore[arg-type]

    def test_short_fingerprint_rejected(self) -> None:
        with pytest.raises(ValueError, match="64-char lowercase hex"):
            SecretResolutionInput(**self._valid_kwargs(fingerprint="abc123"))  # type: ignore[arg-type]

    def test_uppercase_fingerprint_rejected(self) -> None:
        with pytest.raises(ValueError, match="64-char lowercase hex"):
            SecretResolutionInput(**self._valid_kwargs(fingerprint="A" * 64))  # type: ignore[arg-type]

    def test_non_hex_fingerprint_rejected(self) -> None:
        with pytest.raises(ValueError, match="64-char lowercase hex"):
            SecretResolutionInput(**self._valid_kwargs(fingerprint="g" * 64))  # type: ignore[arg-type]

    def test_negative_latency_rejected(self) -> None:
        with pytest.raises(ValueError, match="non-negative"):
            SecretResolutionInput(**self._valid_kwargs(resolution_latency_ms=-1.0))  # type: ignore[arg-type]

    def test_zero_latency_accepted(self) -> None:
        """Zero latency is valid (e.g., cached resolution)."""
        sri = SecretResolutionInput(**self._valid_kwargs(resolution_latency_ms=0.0))  # type: ignore[arg-type]
        assert sri.resolution_latency_ms == 0.0

    def test_all_hex_digits_accepted(self) -> None:
        """Fingerprint using all valid hex chars should pass."""
        # Use all 16 hex digits repeated to fill 64 chars
        fingerprint = "0123456789abcdef" * 4
        assert len(fingerprint) == 64
        sri = SecretResolutionInput(**self._valid_kwargs(fingerprint=fingerprint))  # type: ignore[arg-type]
        assert sri.fingerprint == fingerprint

    # --- Extended sources: "env" and "user" ---

    def test_env_source_accepted(self) -> None:
        """SecretResolutionInput accepts source='env' with no vault fields."""
        sri = SecretResolutionInput(**self._valid_kwargs(source="env", vault_url=None, secret_name=None))  # type: ignore[arg-type]
        assert sri.source == "env"
        assert sri.vault_url is None
        assert sri.secret_name is None

    def test_user_source_accepted(self) -> None:
        """SecretResolutionInput accepts source='user' with no vault fields."""
        sri = SecretResolutionInput(**self._valid_kwargs(source="user", vault_url=None, secret_name=None))  # type: ignore[arg-type]
        assert sri.source == "user"
        assert sri.vault_url is None
        assert sri.secret_name is None


class TestSecretResolutionReadSideValidation:
    """Tests for SecretResolution (read-side) __post_init__ validation.

    SecretResolution is the read-side contract for secret provenance records.
    Tier 1 audit data: crash immediately on any anomaly.
    """

    @staticmethod
    def _valid_kwargs(**overrides: object) -> dict[str, object]:
        """Build valid SecretResolution kwargs with optional overrides."""
        defaults: dict[str, object] = {
            "resolution_id": "res-001",
            "run_id": "run-001",
            "timestamp": 1709100000.0,
            "env_var_name": "API_KEY",
            "source": "keyvault",
            "fingerprint": "a" * 64,
            "vault_url": "https://vault.example.com",
            "secret_name": "api-key",
            "resolution_latency_ms": 42.5,
        }
        defaults.update(overrides)
        return defaults

    def test_keyvault_source_valid(self) -> None:
        """Keyvault source with vault_url and secret_name succeeds."""
        sr = SecretResolution(**self._valid_kwargs())  # type: ignore[arg-type]
        assert sr.source == "keyvault"
        assert sr.vault_url == "https://vault.example.com"
        assert sr.secret_name == "api-key"

    def test_keyvault_requires_vault_url(self) -> None:
        """Keyvault source without vault_url crashes (Tier 1)."""
        with pytest.raises(ValueError, match="vault_url is required"):
            SecretResolution(**self._valid_kwargs(vault_url=None))  # type: ignore[arg-type]

    def test_keyvault_requires_secret_name(self) -> None:
        """Keyvault source without secret_name crashes (Tier 1)."""
        with pytest.raises(ValueError, match="secret_name is required"):
            SecretResolution(**self._valid_kwargs(secret_name=None))  # type: ignore[arg-type]

    def test_env_source_accepted(self) -> None:
        """SecretResolution accepts source='env' with no vault fields."""
        sr = SecretResolution(**self._valid_kwargs(source="env", vault_url=None, secret_name=None))  # type: ignore[arg-type]
        assert sr.source == "env"
        assert sr.vault_url is None
        assert sr.secret_name is None

    def test_user_source_accepted(self) -> None:
        """SecretResolution accepts source='user' with no vault fields."""
        sr = SecretResolution(**self._valid_kwargs(source="user", vault_url=None, secret_name=None))  # type: ignore[arg-type]
        assert sr.source == "user"
        assert sr.vault_url is None
        assert sr.secret_name is None

    def test_invalid_source_rejected(self) -> None:
        """Unknown source value crashes (Tier 1)."""
        with pytest.raises(ValueError, match="source must be one of"):
            SecretResolution(**self._valid_kwargs(source="s3"))  # type: ignore[arg-type]

    def test_empty_source_rejected(self) -> None:
        """Empty source crashes (Tier 1)."""
        with pytest.raises(ValueError, match="source is required"):
            SecretResolution(**self._valid_kwargs(source=""))  # type: ignore[arg-type]


class TestRequireIntValidation:
    """Tests for require_int validation across audit dataclasses.

    Covers representative patterns:
    1. Class with no prior __post_init__ (Row)
    2. Class with existing __post_init__ (Call)
    3. Optional int field (Token.step_in_pipeline)
    4. min_value=1 field (NodeStateOpen.attempt)
    """

    # --- Row: no prior __post_init__, min_value=0 ---

    def test_row_rejects_bool_row_index(self) -> None:
        """Row.row_index must be int, not bool."""
        with pytest.raises(TypeError, match="row_index must be int"):
            Row(
                row_id="row-1",
                run_id="run-1",
                source_node_id="node-1",
                row_index=True,
                source_row_index=0,
                ingest_sequence=0,
                source_data_hash="abc123",
                created_at=datetime.now(UTC),
            )

    def test_row_rejects_str_row_index(self) -> None:
        """Row.row_index must be int, not str."""
        with pytest.raises(TypeError, match="row_index must be int"):
            Row(
                row_id="row-1",
                run_id="run-1",
                source_node_id="node-1",
                row_index="0",  # type: ignore[arg-type]
                source_row_index=0,
                ingest_sequence=0,
                source_data_hash="abc123",
                created_at=datetime.now(UTC),
            )

    def test_row_rejects_negative_row_index(self) -> None:
        """Row.row_index must be >= 0."""
        with pytest.raises(ValueError, match="row_index must be >= 0"):
            Row(
                row_id="row-1",
                run_id="run-1",
                source_node_id="node-1",
                row_index=-1,
                source_row_index=0,
                ingest_sequence=0,
                source_data_hash="abc123",
                created_at=datetime.now(UTC),
            )

    def test_row_accepts_zero_row_index(self) -> None:
        """Row.row_index accepts 0 (valid minimum)."""
        row = Row(
            row_id="row-1",
            run_id="run-1",
            source_node_id="node-1",
            row_index=0,
            source_row_index=0,
            ingest_sequence=0,
            source_data_hash="abc123",
            created_at=datetime.now(UTC),
        )
        assert row.row_index == 0

    def test_row_requires_source_scoped_identity(self) -> None:
        """Row identity must not be fabricated from legacy row_index.

        source_row_index is a required `int` field; passing None (the legacy
        "absent identity" shape) is rejected by the Tier-1 require_int guard.
        """
        with pytest.raises(TypeError, match="source_row_index must be int"):
            Row(
                row_id="row-1",
                run_id="run-1",
                source_node_id="node-1",
                row_index=0,
                source_row_index=None,  # type: ignore[arg-type]
                ingest_sequence=0,
                source_data_hash="abc123",
                created_at=datetime.now(UTC),
            )

    # --- Call: existing __post_init__, require_int added before enum validation ---

    def test_call_rejects_bool_call_index(self) -> None:
        """Call.call_index must be int, not bool."""
        with pytest.raises(TypeError, match="call_index must be int"):
            Call(
                call_id="call-1",
                state_id="state-1",
                call_index=False,
                call_type=CallType.LLM,
                status=CallStatus.SUCCESS,
                request_hash="abc123",
                created_at=datetime.now(UTC),
            )

    def test_call_rejects_negative_call_index(self) -> None:
        """Call.call_index must be >= 0."""
        with pytest.raises(ValueError, match="call_index must be >= 0"):
            Call(
                call_id="call-1",
                state_id="state-1",
                call_index=-1,
                call_type=CallType.LLM,
                status=CallStatus.SUCCESS,
                request_hash="abc123",
                created_at=datetime.now(UTC),
            )

    # --- Token.step_in_pipeline: optional int ---

    def test_token_step_in_pipeline_accepts_none(self) -> None:
        """Token.step_in_pipeline accepts None (optional)."""
        token = Token(
            token_id="tok-1",
            row_id="row-1",
            created_at=datetime.now(UTC),
            run_id="run-1",
            step_in_pipeline=None,
        )
        assert token.step_in_pipeline is None

    def test_token_step_in_pipeline_rejects_bool(self) -> None:
        """Token.step_in_pipeline must be int or None, not bool."""
        with pytest.raises(TypeError, match="step_in_pipeline must be int"):
            Token(
                token_id="tok-1",
                row_id="row-1",
                created_at=datetime.now(UTC),
                run_id="run-1",
                step_in_pipeline=True,
            )

    def test_token_step_in_pipeline_accepts_zero(self) -> None:
        """Token.step_in_pipeline accepts 0."""
        token = Token(
            token_id="tok-1",
            row_id="row-1",
            created_at=datetime.now(UTC),
            run_id="run-1",
            step_in_pipeline=0,
        )
        assert token.step_in_pipeline == 0

    # --- NodeStateOpen.attempt: min_value=0 (0-indexed: first attempt is 0) ---

    def test_node_state_open_rejects_negative_attempt(self) -> None:
        """NodeStateOpen.attempt must be >= 0 (0-indexed)."""
        with pytest.raises(ValueError, match="attempt must be >= 0"):
            NodeStateOpen(
                state_id="state-1",
                token_id="tok-1",
                node_id="node-1",
                step_index=0,
                attempt=-1,
                status=NodeStateStatus.OPEN,
                input_hash="abc123",
                started_at=datetime.now(UTC),
            )

    def test_node_state_open_accepts_zero_attempt(self) -> None:
        """NodeStateOpen.attempt accepts 0 (first attempt, 0-indexed)."""
        state = NodeStateOpen(
            state_id="state-1",
            token_id="tok-1",
            node_id="node-1",
            step_index=0,
            attempt=0,
            status=NodeStateStatus.OPEN,
            input_hash="abc123",
            started_at=datetime.now(UTC),
        )
        assert state.attempt == 0

    def test_node_state_open_rejects_bool_attempt(self) -> None:
        """NodeStateOpen.attempt must be int, not bool."""
        with pytest.raises(TypeError, match="attempt must be int"):
            NodeStateOpen(
                state_id="state-1",
                token_id="tok-1",
                node_id="node-1",
                step_index=0,
                attempt=True,
                status=NodeStateStatus.OPEN,
                input_hash="abc123",
                started_at=datetime.now(UTC),
            )
