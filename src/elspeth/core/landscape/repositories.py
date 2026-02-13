"""Repository layer for Landscape audit models.

Handles the seam between SQLAlchemy rows (strings) and domain objects
(strict enum types). This is NOT a trust boundary - if the database
has bad data, we crash. That's intentional per Data Manifesto.

Per Data Manifesto: The audit database is OUR data. Bad data = crash.
"""

from typing import Any

from sqlalchemy.engine import Row as SARow

from elspeth.contracts.audit import (
    Artifact,
    Batch,
    BatchMember,
    Call,
    Edge,
    Node,
    NodeState,
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


class RunRepository:
    """Repository for Run records."""

    def load(self, row: SARow[Any]) -> Run:
        """Load Run from database row.

        Converts string fields to enums. Crashes on invalid data.
        """
        return Run(
            run_id=row.run_id,
            started_at=row.started_at,
            config_hash=row.config_hash,
            settings_json=row.settings_json,
            canonical_version=row.canonical_version,
            status=RunStatus(row.status),  # Convert HERE
            completed_at=row.completed_at,
            reproducibility_grade=row.reproducibility_grade,
            # Use explicit is not None check - empty string should raise, not become None (Tier 1)
            export_status=ExportStatus(row.export_status) if row.export_status is not None else None,
            export_error=row.export_error,
            exported_at=row.exported_at,
            export_format=row.export_format,
            export_sink=row.export_sink,
        )


class NodeRepository:
    """Repository for Node records."""

    def load(self, row: SARow[Any]) -> Node:
        """Load Node from database row.

        Converts node_type and determinism strings to enums.
        Parses schema_fields_json back to list.
        """
        import json

        # Parse schema_fields_json back to list[dict[str, object]]
        schema_fields: list[dict[str, object]] | None = None
        if row.schema_fields_json is not None:
            parsed_fields = json.loads(row.schema_fields_json)
            if type(parsed_fields) is not list:
                raise ValueError(f"schema_fields_json must decode to list[dict], got {type(parsed_fields).__name__}")
            for idx, item in enumerate(parsed_fields):
                if type(item) is not dict:
                    raise ValueError(f"schema_fields_json[{idx}] must be object/dict, got {type(item).__name__}")
            schema_fields = parsed_fields

        return Node(
            node_id=row.node_id,
            run_id=row.run_id,
            plugin_name=row.plugin_name,
            node_type=NodeType(row.node_type),  # Convert HERE
            plugin_version=row.plugin_version,
            determinism=Determinism(row.determinism),  # Convert HERE
            config_hash=row.config_hash,
            config_json=row.config_json,
            registered_at=row.registered_at,
            schema_hash=row.schema_hash,
            sequence_in_pipeline=row.sequence_in_pipeline,
            schema_mode=row.schema_mode,
            schema_fields=schema_fields,
        )


class EdgeRepository:
    """Repository for Edge records."""

    def load(self, row: SARow[Any]) -> Edge:
        """Load Edge from database row.

        Converts default_mode string to RoutingMode enum.
        """
        return Edge(
            edge_id=row.edge_id,
            run_id=row.run_id,
            from_node_id=row.from_node_id,
            to_node_id=row.to_node_id,
            label=row.label,
            default_mode=RoutingMode(row.default_mode),  # Convert HERE
            created_at=row.created_at,
        )


class RowRepository:
    """Repository for Row records."""

    def load(self, row: SARow[Any]) -> Row:
        """Load Row from database row.

        No enum conversion needed - all fields are primitives.
        """
        return Row(
            row_id=row.row_id,
            run_id=row.run_id,
            source_node_id=row.source_node_id,
            row_index=row.row_index,
            source_data_hash=row.source_data_hash,
            created_at=row.created_at,
            source_data_ref=row.source_data_ref,
        )


class TokenRepository:
    """Repository for Token records."""

    def load(self, row: SARow[Any]) -> Token:
        """Load Token from database row.

        No enum conversion needed - all fields are primitives.
        """
        return Token(
            token_id=row.token_id,
            row_id=row.row_id,
            created_at=row.created_at,
            fork_group_id=row.fork_group_id,
            join_group_id=row.join_group_id,
            expand_group_id=row.expand_group_id,
            branch_name=row.branch_name,
            step_in_pipeline=row.step_in_pipeline,
        )


class TokenParentRepository:
    """Repository for TokenParent records."""

    def load(self, row: SARow[Any]) -> TokenParent:
        """Load TokenParent from database row."""
        return TokenParent(
            token_id=row.token_id,
            parent_token_id=row.parent_token_id,
            ordinal=row.ordinal,
        )


class CallRepository:
    """Repository for Call records."""

    def load(self, row: SARow[Any]) -> Call:
        """Load Call from database row.

        Handles both state-parented calls (transform processing) and
        operation-parented calls (source/sink I/O).
        """
        return Call(
            call_id=row.call_id,
            call_index=row.call_index,
            call_type=CallType(row.call_type),  # Convert HERE
            status=CallStatus(row.status),  # Convert HERE
            request_hash=row.request_hash,
            created_at=row.created_at,
            state_id=row.state_id,  # NULL for operation calls
            operation_id=row.operation_id,  # NULL for state calls
            request_ref=row.request_ref,
            response_hash=row.response_hash,
            response_ref=row.response_ref,
            error_json=row.error_json,
            latency_ms=row.latency_ms,
        )


class RoutingEventRepository:
    """Repository for RoutingEvent records."""

    def load(self, row: SARow[Any]) -> RoutingEvent:
        """Load RoutingEvent from database row."""
        return RoutingEvent(
            event_id=row.event_id,
            state_id=row.state_id,
            edge_id=row.edge_id,
            routing_group_id=row.routing_group_id,
            ordinal=row.ordinal,
            mode=RoutingMode(row.mode),  # Convert HERE
            created_at=row.created_at,
            reason_hash=row.reason_hash,
            reason_ref=row.reason_ref,
        )


class BatchRepository:
    """Repository for Batch records."""

    def load(self, row: SARow[Any]) -> Batch:
        """Load Batch from database row."""
        return Batch(
            batch_id=row.batch_id,
            run_id=row.run_id,
            aggregation_node_id=row.aggregation_node_id,
            attempt=row.attempt,
            status=BatchStatus(row.status),  # Convert HERE
            created_at=row.created_at,
            aggregation_state_id=row.aggregation_state_id,
            trigger_type=TriggerType(row.trigger_type) if row.trigger_type is not None else None,
            trigger_reason=row.trigger_reason,
            completed_at=row.completed_at,
        )


class NodeStateRepository:
    """Repository for NodeState records (discriminated union).

    NodeState is a discriminated union with 4 variants based on status:
    - NodeStateOpen: Just started, no output yet
    - NodeStatePending: In progress (e.g., waiting for async result)
    - NodeStateCompleted: Finished successfully with output
    - NodeStateFailed: Finished with error

    Each variant has different required fields. This repository validates
    these invariants per the Tier 1 trust model - if invariants are violated,
    we crash immediately (audit integrity violation).
    """

    def load(self, row: SARow[Any]) -> NodeState:
        """Load NodeState from database row.

        Converts status string to enum and returns the appropriate
        NodeState variant based on the discriminator (status field).

        Args:
            row: Database row from node_states table

        Returns:
            NodeStateOpen, NodeStatePending, NodeStateCompleted,
            or NodeStateFailed depending on status

        Raises:
            ValueError: If status is invalid or invariants are violated
                       (Tier 1 audit integrity violation - crash required)
        """
        status = NodeStateStatus(row.status)

        if status == NodeStateStatus.OPEN:
            # BUG #6: OPEN states must have NULL completion fields
            # Operations haven't finished yet, so output_hash, completed_at, and duration_ms
            # must all be NULL. Non-NULL values indicate corrupted audit data.
            if row.output_hash is not None:
                raise ValueError(f"OPEN state {row.state_id} has non-NULL output_hash - audit integrity violation")
            if row.completed_at is not None:
                raise ValueError(f"OPEN state {row.state_id} has non-NULL completed_at - audit integrity violation")
            if row.duration_ms is not None:
                raise ValueError(f"OPEN state {row.state_id} has non-NULL duration_ms - audit integrity violation")

            return NodeStateOpen(
                state_id=row.state_id,
                token_id=row.token_id,
                node_id=row.node_id,
                step_index=row.step_index,
                attempt=row.attempt,
                status=NodeStateStatus.OPEN,
                input_hash=row.input_hash,
                started_at=row.started_at,
                context_before_json=row.context_before_json,
            )

        elif status == NodeStateStatus.PENDING:
            # PENDING states must have completed_at, duration_ms (but no output_hash yet)
            # Validate required fields - None indicates audit integrity violation
            if row.duration_ms is None:
                raise ValueError(f"PENDING state {row.state_id} has NULL duration_ms - audit integrity violation")
            if row.completed_at is None:
                raise ValueError(f"PENDING state {row.state_id} has NULL completed_at - audit integrity violation")

            # BUG #6: PENDING states must have NULL output_hash
            # Batch processing is in progress, no output available yet.
            # Non-NULL output_hash contradicts PENDING status.
            if row.output_hash is not None:
                raise ValueError(f"PENDING state {row.state_id} has non-NULL output_hash - audit integrity violation")

            return NodeStatePending(
                state_id=row.state_id,
                token_id=row.token_id,
                node_id=row.node_id,
                step_index=row.step_index,
                attempt=row.attempt,
                status=NodeStateStatus.PENDING,
                input_hash=row.input_hash,
                started_at=row.started_at,
                completed_at=row.completed_at,
                duration_ms=row.duration_ms,
                context_before_json=row.context_before_json,
                context_after_json=row.context_after_json,
            )

        elif status == NodeStateStatus.COMPLETED:
            # COMPLETED states must have output_hash, completed_at, duration_ms
            # Validate required fields - None indicates audit integrity violation
            if row.output_hash is None:
                raise ValueError(f"COMPLETED state {row.state_id} has NULL output_hash - audit integrity violation")
            if row.duration_ms is None:
                raise ValueError(f"COMPLETED state {row.state_id} has NULL duration_ms - audit integrity violation")
            if row.completed_at is None:
                raise ValueError(f"COMPLETED state {row.state_id} has NULL completed_at - audit integrity violation")
            return NodeStateCompleted(
                state_id=row.state_id,
                token_id=row.token_id,
                node_id=row.node_id,
                step_index=row.step_index,
                attempt=row.attempt,
                status=NodeStateStatus.COMPLETED,
                input_hash=row.input_hash,
                started_at=row.started_at,
                output_hash=row.output_hash,
                completed_at=row.completed_at,
                duration_ms=row.duration_ms,
                context_before_json=row.context_before_json,
                context_after_json=row.context_after_json,
                success_reason_json=row.success_reason_json,
            )

        elif status == NodeStateStatus.FAILED:
            # FAILED states must have completed_at, duration_ms
            # error_json and output_hash are optional
            # Validate required fields - None indicates audit integrity violation
            if row.duration_ms is None:
                raise ValueError(f"FAILED state {row.state_id} has NULL duration_ms - audit integrity violation")
            if row.completed_at is None:
                raise ValueError(f"FAILED state {row.state_id} has NULL completed_at - audit integrity violation")
            return NodeStateFailed(
                state_id=row.state_id,
                token_id=row.token_id,
                node_id=row.node_id,
                step_index=row.step_index,
                attempt=row.attempt,
                status=NodeStateStatus.FAILED,
                input_hash=row.input_hash,
                started_at=row.started_at,
                completed_at=row.completed_at,
                duration_ms=row.duration_ms,
                error_json=row.error_json,
                output_hash=row.output_hash,
                context_before_json=row.context_before_json,
                context_after_json=row.context_after_json,
            )

        else:
            # This branch should be unreachable if NodeStateStatus enum is correct
            # But we include it for defensive completeness - crash on unknown status
            raise ValueError(f"Unknown status {row.status} for state {row.state_id}")


class ValidationErrorRepository:
    """Repository for ValidationErrorRecord records.

    Handles source validation errors (quarantined rows).
    No enum conversion needed - all fields are primitives or strings.
    """

    def load(self, row: SARow[Any]) -> ValidationErrorRecord:
        """Load ValidationErrorRecord from database row.

        Args:
            row: Database row from validation_errors table

        Returns:
            ValidationErrorRecord with all fields mapped
        """
        return ValidationErrorRecord(
            error_id=row.error_id,
            run_id=row.run_id,
            node_id=row.node_id,
            row_hash=row.row_hash,
            error=row.error,
            schema_mode=row.schema_mode,
            destination=row.destination,
            created_at=row.created_at,
            row_data_json=row.row_data_json,
        )


class TransformErrorRepository:
    """Repository for TransformErrorRecord records.

    Handles transform processing errors.
    No enum conversion needed - all fields are primitives or strings.
    """

    def load(self, row: SARow[Any]) -> TransformErrorRecord:
        """Load TransformErrorRecord from database row.

        Args:
            row: Database row from transform_errors table

        Returns:
            TransformErrorRecord with all fields mapped
        """
        return TransformErrorRecord(
            error_id=row.error_id,
            run_id=row.run_id,
            token_id=row.token_id,
            transform_id=row.transform_id,
            row_hash=row.row_hash,
            destination=row.destination,
            created_at=row.created_at,
            row_data_json=row.row_data_json,
            error_details_json=row.error_details_json,
        )


class TokenOutcomeRepository:
    """Repository for TokenOutcome records.

    Handles terminal token states. Converts outcome string to RowOutcome enum.
    """

    def load(self, row: SARow[Any]) -> TokenOutcome:
        """Load TokenOutcome from database row.

        Converts outcome string to RowOutcome enum.
        Converts is_terminal from DB integer (0/1) to bool.

        Args:
            row: Database row from token_outcomes table

        Returns:
            TokenOutcome with outcome converted to RowOutcome enum

        Raises:
            ValueError: If is_terminal is not 0 or 1 (Tier 1 audit integrity violation)
            ValueError: If outcome and is_terminal disagree (Tier 1 audit integrity violation)
        """
        # Tier 1 validation: is_terminal must be exactly 0 or 1
        # Per Data Manifesto: audit DB is OUR data - crash on any anomaly
        if row.is_terminal not in (0, 1):
            raise ValueError(
                f"TokenOutcome {row.outcome_id} has invalid is_terminal={row.is_terminal!r} (expected 0 or 1) - audit integrity violation"
            )
        outcome = RowOutcome(row.outcome)
        is_terminal = row.is_terminal == 1
        if is_terminal != outcome.is_terminal:
            raise ValueError(
                f"TokenOutcome {row.outcome_id} has inconsistent is_terminal={row.is_terminal!r} for outcome={outcome.value!r} "
                f"(expected {1 if outcome.is_terminal else 0}) - audit integrity violation"
            )
        return TokenOutcome(
            outcome_id=row.outcome_id,
            run_id=row.run_id,
            token_id=row.token_id,
            outcome=outcome,
            is_terminal=is_terminal,  # DB stores as Integer, now fully validated
            recorded_at=row.recorded_at,
            sink_name=row.sink_name,
            batch_id=row.batch_id,
            fork_group_id=row.fork_group_id,
            join_group_id=row.join_group_id,
            expand_group_id=row.expand_group_id,
            error_hash=row.error_hash,
            context_json=row.context_json,
            expected_branches_json=row.expected_branches_json,
        )


class ArtifactRepository:
    """Repository for Artifact records.

    Handles sink output artifacts with content hashes.
    No enum conversion needed - artifact_type is user-defined string.
    """

    def load(self, row: SARow[Any]) -> Artifact:
        """Load Artifact from database row.

        Args:
            row: Database row from artifacts table

        Returns:
            Artifact with all fields mapped
        """
        return Artifact(
            artifact_id=row.artifact_id,
            run_id=row.run_id,
            produced_by_state_id=row.produced_by_state_id,
            sink_node_id=row.sink_node_id,
            artifact_type=row.artifact_type,
            path_or_uri=row.path_or_uri,
            content_hash=row.content_hash,
            size_bytes=row.size_bytes,
            created_at=row.created_at,
            idempotency_key=row.idempotency_key,
        )


class BatchMemberRepository:
    """Repository for BatchMember records.

    Handles batch membership records for aggregation tracking.
    No enum conversion needed - all fields are primitives.
    """

    def load(self, row: SARow[Any]) -> BatchMember:
        """Load BatchMember from database row.

        Args:
            row: Database row from batch_members table

        Returns:
            BatchMember with all fields mapped
        """
        return BatchMember(
            batch_id=row.batch_id,
            token_id=row.token_id,
            ordinal=row.ordinal,
        )
