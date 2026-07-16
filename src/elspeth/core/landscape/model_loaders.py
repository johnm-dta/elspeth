"""Model loaders for Landscape audit records.

Handles the seam between SQLAlchemy rows (strings) and domain objects
(strict enum types). This is NOT a trust boundary - if the database
has bad data, we crash. That's intentional per Data Manifesto.

Per Data Manifesto: The audit database is OUR data. Bad data = crash.
"""

from typing import Any

from sqlalchemy.engine import Row as SARow

from elspeth.contracts.audit import (
    Artifact,
    AuditExportSnapshot,
    AuditExportSnapshotChunk,
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
    Operation,
    RoutingEvent,
    Row,
    Run,
    SinkEffect,
    SinkEffectAttempt,
    SinkEffectExportSnapshotAssociation,
    SinkEffectMemberRecord,
    SinkEffectStream,
    Token,
    TokenOutcome,
    TokenParent,
    TransformErrorRecord,
    ValidationErrorRecord,
    validate_node_state_persisted_fields,
    validate_token_outcome_persisted_fields,
)
from elspeth.contracts.enums import (
    BatchStatus,
    CallStatus,
    CallType,
    Determinism,
    ExportStatus,
    NodeStateStatus,
    NodeType,
    ReproducibilityGrade,
    RoutingMode,
    RunStatus,
    TerminalOutcome,
    TerminalPath,
    TriggerType,
)
from elspeth.contracts.errors import AuditIntegrityError
from elspeth.contracts.scheduler import SchedulerEvent, SchedulerEventType, TokenWorkStatus
from elspeth.contracts.sink_effects import (
    AuditExportFormat,
    AuditExportSigningMode,
    SinkEffectAttemptAction,
    SinkEffectAttemptState,
    SinkEffectDescriptorMode,
    SinkEffectInputKind,
    SinkEffectInspectionMode,
    SinkEffectReconcileKind,
    SinkEffectRole,
    SinkEffectState,
)

_COMPLETED_AT_REQUIRED_RUN_STATUSES = frozenset(
    {
        RunStatus.COMPLETED,
        RunStatus.COMPLETED_WITH_FAILURES,
        RunStatus.EMPTY,
    }
)


def validate_run_lifecycle_row(run_id: str, status: RunStatus, completed_at: object | None) -> None:
    """Validate status-dependent run lifecycle invariants for persisted rows."""
    if status == RunStatus.RUNNING:
        if completed_at is not None:
            raise AuditIntegrityError(
                f"Run {run_id} has status='running' but completed_at is set — "
                f"audit integrity violation (running runs must not have completed_at)"
            )
    elif status in _COMPLETED_AT_REQUIRED_RUN_STATUSES and completed_at is None:
        raise AuditIntegrityError(
            f"Run {run_id} has status={status.value!r} but completed_at is NULL — "
            f"audit integrity violation ({status.value!r} runs must have completed_at)"
        )


class RunLoader:
    """Loader for Run records."""

    def load(self, row: SARow[Any]) -> Run:
        """Load Run from database row.

        Converts string fields to enums. Validates status-dependent
        invariants before construction.

        Raises:
            AuditIntegrityError: If status/completed_at are inconsistent (Tier 1)
        """
        status = RunStatus(row.status)

        # Tier 1: status-dependent lifecycle invariants
        validate_run_lifecycle_row(row.run_id, status, row.completed_at)
        # FAILED and INTERRUPTED: completed_at may or may not be set.
        # complete_run() sets it; update_run_status() (recovery path) does not.

        return Run(
            run_id=row.run_id,
            started_at=row.started_at,
            config_hash=row.config_hash,
            settings_json=row.settings_json,
            canonical_version=row.canonical_version,
            status=status,
            completed_at=row.completed_at,
            # Validate reproducibility_grade on read — crash on invalid values (Tier 1)
            reproducibility_grade=ReproducibilityGrade(row.reproducibility_grade) if row.reproducibility_grade is not None else None,
            # Use explicit is not None check - empty string should raise, not become None (Tier 1)
            export_status=ExportStatus(row.export_status) if row.export_status is not None else None,
            export_error=row.export_error,
            exported_at=row.exported_at,
            export_format=row.export_format,
            export_sink=row.export_sink,
            llm_call_count=row.llm_call_count,
            seeded_from_cache=bool(row.seeded_from_cache),
            cache_key=row.cache_key,
        )


class NodeLoader:
    """Loader for Node records."""

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
                raise AuditIntegrityError(f"schema_fields_json must decode to list[dict], got {type(parsed_fields).__name__}")
            for idx, item in enumerate(parsed_fields):
                if type(item) is not dict:
                    raise AuditIntegrityError(f"schema_fields_json[{idx}] must be object/dict, got {type(item).__name__}")
            schema_fields = parsed_fields

        return Node(
            node_id=row.node_id,
            run_id=row.run_id,
            plugin_name=row.plugin_name,
            node_type=NodeType(row.node_type),  # Convert HERE
            plugin_version=row.plugin_version,
            source_file_hash=row.source_file_hash,
            determinism=Determinism(row.determinism),  # Convert HERE
            config_hash=row.config_hash,
            config_json=row.config_json,
            registered_at=row.registered_at,
            schema_hash=row.schema_hash,
            sequence_in_pipeline=row.sequence_in_pipeline,
            schema_mode=row.schema_mode,
            schema_fields=schema_fields,
        )


class EdgeLoader:
    """Loader for Edge records."""

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


class RowLoader:
    """Loader for Row records."""

    def load(self, row: SARow[Any]) -> Row:
        """Load Row from database row.

        No enum conversion needed - all fields are primitives.
        """
        return Row(
            row_id=row.row_id,
            run_id=row.run_id,
            source_node_id=row.source_node_id,
            row_index=row.row_index,
            source_row_index=row.source_row_index,
            ingest_sequence=row.ingest_sequence,
            source_data_hash=row.source_data_hash,
            created_at=row.created_at,
            source_data_ref=row.source_data_ref,
        )


class TokenLoader:
    """Loader for Token records."""

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
            run_id=row.run_id,
            # Epoch-11 column (tokens.token_data_ref): per-token payload ref for
            # expand children + post-coalesce merged tokens. Read directly from the
            # persisted column — omitting it fabricated None for a Tier-1 value the
            # audit trail actually recorded.
            token_data_ref=row.token_data_ref,
        )


class TokenParentLoader:
    """Loader for TokenParent records."""

    def load(self, row: SARow[Any]) -> TokenParent:
        """Load TokenParent from database row."""
        return TokenParent(
            token_id=row.token_id,
            parent_token_id=row.parent_token_id,
            ordinal=row.ordinal,
        )


class SchedulerEventLoader:
    """Loader for durable scheduler transition events."""

    def load(self, row: SARow[Any]) -> SchedulerEvent:
        return SchedulerEvent(
            event_id=row.event_id,
            run_id=row.run_id,
            token_id=row.token_id,
            work_item_id=row.work_item_id,
            node_id=row.node_id,
            event_type=SchedulerEventType(row.event_type),
            from_status=TokenWorkStatus(row.from_status) if row.from_status is not None else None,
            to_status=TokenWorkStatus(row.to_status),
            from_lease_owner=row.from_lease_owner,
            to_lease_owner=row.to_lease_owner,
            from_lease_expires_at=row.from_lease_expires_at,
            to_lease_expires_at=row.to_lease_expires_at,
            from_attempt=row.from_attempt,
            to_attempt=row.to_attempt,
            recorded_at=row.recorded_at,
            caller_owner=row.caller_owner,
            context_json=row.context_json,
        )


class CallLoader:
    """Loader for Call records."""

    def load(self, row: SARow[Any]) -> Call:
        """Load Call from database row.

        Handles both state-parented calls (transform processing) and
        operation-parented calls (source/sink I/O). Validates XOR
        constraint before construction.

        Raises:
            AuditIntegrityError: If state_id/operation_id XOR violated (Tier 1)
        """
        # Tier 1: XOR constraint — exactly one parent context
        has_state = row.state_id is not None
        has_operation = row.operation_id is not None
        if has_state == has_operation:
            raise AuditIntegrityError(
                f"Call {row.call_id} requires exactly one of state_id or operation_id. "
                f"Got state_id={row.state_id!r}, operation_id={row.operation_id!r} — "
                f"audit integrity violation"
            )

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
            resolved_prompt_template_hash=row.resolved_prompt_template_hash,
        )


class RoutingEventLoader:
    """Loader for RoutingEvent records."""

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


class BatchLoader:
    """Loader for Batch records."""

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
            retry_of_batch_id=row.retry_of_batch_id,
            trigger_type=TriggerType(row.trigger_type) if row.trigger_type is not None else None,
            trigger_reason=row.trigger_reason,
            completed_at=row.completed_at,
        )


class NodeStateLoader:
    """Loader for NodeState records (discriminated union).

    NodeState is a discriminated union with 4 variants based on status:
    - NodeStateOpen: Just started, no output yet
    - NodeStatePending: In progress (e.g., waiting for async result)
    - NodeStateCompleted: Finished successfully with output
    - NodeStateFailed: Finished with error

    Each variant has different required fields. This loader validates
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
            AuditIntegrityError: If status is invalid or invariants are violated
                                (Tier 1 audit integrity violation - crash required)
        """
        status = NodeStateStatus(row.status)
        try:
            validate_node_state_persisted_fields(
                row.state_id,
                status,
                output_hash=row.output_hash,
                completed_at=row.completed_at,
                duration_ms=row.duration_ms,
                context_after_json=row.context_after_json,
                error_json=row.error_json,
                success_reason_json=row.success_reason_json,
            )
        except ValueError as exc:
            raise AuditIntegrityError(str(exc)) from exc

        if status == NodeStateStatus.OPEN:
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
            # Exhaustive match: all 4 NodeStateStatus variants are handled above.
            # If we reach here, the enum was extended without updating this loader.
            raise AuditIntegrityError(f"Unknown status {row.status} for state {row.state_id}")


class ValidationErrorLoader:
    """Loader for ValidationErrorRecord records.

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
            row_id=row.row_id,
            row_data_json=row.row_data_json,
            violation_type=row.violation_type,
            original_field_name=row.original_field_name,
            normalized_field_name=row.normalized_field_name,
            expected_type=row.expected_type,
            actual_type=row.actual_type,
        )


class TransformErrorLoader:
    """Loader for TransformErrorRecord records.

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


class TokenOutcomeLoader:
    """Loader for TokenOutcome records.

    Handles token outcomes under the ADR-019 two-axis terminal model.
    """

    def load(self, row: SARow[Any]) -> TokenOutcome:
        """Load a TokenOutcome row with Tier 1 ADR-019 cross-checks."""
        oid = row.outcome_id

        if type(row.completed) is not int or row.completed not in (0, 1):
            raise AuditIntegrityError(
                f"TokenOutcome {oid}: invalid completed={row.completed!r} (expected int 0 or 1) — audit integrity violation"
            )
        completed = row.completed == 1

        if row.outcome is None:
            outcome: TerminalOutcome | None = None
        else:
            try:
                outcome = TerminalOutcome(row.outcome)
            except ValueError as exc:
                raise AuditIntegrityError(
                    f"TokenOutcome {oid}: invalid outcome={row.outcome!r} not in TerminalOutcome — audit integrity violation"
                ) from exc

        if row.path is None:
            raise AuditIntegrityError(
                f"TokenOutcome {oid}: path is NULL — audit integrity violation (path is always populated under ADR-019)"
            )
        try:
            path = TerminalPath(row.path)
        except ValueError as exc:
            raise AuditIntegrityError(
                f"TokenOutcome {oid}: invalid path={row.path!r} not in TerminalPath — audit integrity violation"
            ) from exc

        try:
            validate_token_outcome_persisted_fields(
                oid,
                outcome,
                path,
                completed,
                sink_name=row.sink_name,
                batch_id=row.batch_id,
                fork_group_id=row.fork_group_id,
                join_group_id=row.join_group_id,
                expand_group_id=row.expand_group_id,
                error_hash=row.error_hash,
            )
        except ValueError as exc:
            raise AuditIntegrityError(str(exc)) from exc

        return TokenOutcome(
            outcome_id=oid,
            run_id=row.run_id,
            token_id=row.token_id,
            outcome=outcome,
            path=path,
            completed=completed,
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


class SinkEffectStreamLoader:
    """Strict row decoder for a replacing-target stream."""

    def load(self, row: SARow[Any]) -> SinkEffectStream:
        return SinkEffectStream(
            stream_id=row.stream_id,
            run_id=row.run_id,
            sink_node_id=row.sink_node_id,
            role=SinkEffectRole(row.role),
            requested_target_hash=row.requested_target_hash,
            resolved_target=row.resolved_target,
            next_sequence=row.next_sequence,
            tail_effect_id=row.tail_effect_id,
            head_effect_id=row.head_effect_id,
            head_descriptor_hash=row.head_descriptor_hash,
        )


class SinkEffectLoader:
    """Strict row decoder for the durable effect ledger."""

    def load(self, row: SARow[Any]) -> SinkEffect:
        return SinkEffect(
            effect_id=row.effect_id,
            run_id=row.run_id,
            sink_node_id=row.sink_node_id,
            role=SinkEffectRole(row.role),
            state=SinkEffectState(row.state),
            protocol_version=row.protocol_version,
            input_kind=SinkEffectInputKind(row.input_kind),
            required_member_ordinal=row.required_member_ordinal,
            required_snapshot_slot=row.required_snapshot_slot,
            config_hash=row.config_hash,
            membership_or_manifest_hash=row.membership_or_manifest_hash,
            group_payload_hash=row.group_payload_hash,
            artifact_id=row.artifact_id,
            artifact_idempotency_key=row.artifact_idempotency_key,
            target_json=row.target_json,
            inspection_mode=SinkEffectInspectionMode(row.inspection_mode) if row.inspection_mode is not None else None,
            inspection_attempt_id=row.inspection_attempt_id,
            plan_json=row.plan_json,
            plan_hash=row.plan_hash,
            descriptor_mode=SinkEffectDescriptorMode(row.descriptor_mode) if row.descriptor_mode is not None else None,
            expected_descriptor_hash=row.expected_descriptor_hash,
            precondition_hash=row.precondition_hash,
            prepared_at=row.prepared_at,
            lease_owner=row.lease_owner,
            generation=row.generation,
            lease_expires_at=row.lease_expires_at,
            lease_heartbeat_at=row.lease_heartbeat_at,
            reconcile_kind=SinkEffectReconcileKind(row.reconcile_kind) if row.reconcile_kind is not None else None,
            reconcile_evidence_hash=row.reconcile_evidence_hash,
            result_descriptor_hash=row.result_descriptor_hash,
            publication_performed=row.publication_performed,
            publication_evidence_kind=row.publication_evidence_kind,
            primary_effect_id=row.primary_effect_id,
            stream_id=row.stream_id,
            stream_sequence=row.stream_sequence,
            predecessor_effect_id=row.predecessor_effect_id,
            created_at=row.created_at,
            updated_at=row.updated_at,
            finalized_at=row.finalized_at,
        )


class SinkEffectMemberLoader:
    """Strict row decoder for canonical effect membership."""

    def load(self, row: SARow[Any]) -> SinkEffectMemberRecord:
        return SinkEffectMemberRecord(
            effect_id=row.effect_id,
            input_kind=SinkEffectInputKind(row.input_kind),
            ordinal=row.ordinal,
            run_id=row.run_id,
            sink_node_id=row.sink_node_id,
            role=SinkEffectRole(row.role),
            token_id=row.token_id,
            row_id=row.row_id,
            ingest_sequence=row.ingest_sequence,
            lineage_json=row.lineage_json,
            lineage_hash=row.lineage_hash,
            payload_hash=row.payload_hash,
            prepared_disposition=row.prepared_disposition,
            reason_hash=row.reason_hash,
            member_effect_id=row.member_effect_id,
            member_state=SinkEffectState(row.member_state) if row.member_state is not None else None,
            descriptor_hash=row.descriptor_hash,
            evidence_hash=row.evidence_hash,
        )


class AuditExportSnapshotChunkLoader:
    """Strict internal decoder for a sealed snapshot chunk descriptor."""

    def load(self, row: SARow[Any]) -> AuditExportSnapshotChunk:
        return AuditExportSnapshotChunk(
            snapshot_id=row.snapshot_id,
            ordinal=row.ordinal,
            content_ref=row.content_ref,
            content_hash=row.content_hash,
            size_bytes=row.size_bytes,
            record_count=row.record_count,
            predecessor_seal_hash=row.predecessor_seal_hash,
            cumulative_records=row.cumulative_records,
            cumulative_bytes=row.cumulative_bytes,
            chunk_seal_hash=row.chunk_seal_hash,
        )


class _AuditExportSnapshotRowLoader:
    """Internal registry decoder; deliberately not a public content loader.

    Returning verified bytes requires the winner's durable-store resolver and
    belongs to the snapshot capability factory. Adapters must never use this
    decoder as a substitute for that verification boundary.
    """

    def load(self, row: SARow[Any]) -> AuditExportSnapshot:
        return AuditExportSnapshot(
            snapshot_id=row.snapshot_id,
            source_run_id=row.source_run_id,
            source_status=RunStatus(row.source_status),
            source_completed_at=row.source_completed_at,
            exported_at=row.exported_at,
            registry_key_hash=row.registry_key_hash,
            exporter_version=row.exporter_version,
            serialization_version=row.serialization_version,
            export_format=AuditExportFormat(row.export_format),
            signing_mode=AuditExportSigningMode(row.signing_mode),
            signer_key_id=row.signer_key_id,
            derivation_version=row.derivation_version,
            public_export_config_hash=row.public_export_config_hash,
            chunking_algorithm_version=row.chunking_algorithm_version,
            per_chunk_record_limit=row.per_chunk_record_limit,
            per_chunk_byte_limit=row.per_chunk_byte_limit,
            record_count=row.record_count,
            total_bytes=row.total_bytes,
            chunk_count=row.chunk_count,
            terminal_chunk_ordinal=row.terminal_chunk_ordinal,
            content_store_id=row.content_store_id,
            manifest_hash=row.manifest_hash,
            last_chunk_seal_hash=row.last_chunk_seal_hash,
            snapshot_hash=row.snapshot_hash,
            snapshot_seal_hash=row.snapshot_seal_hash,
            signature_hex=row.signature_hex,
            record_chain_algorithm=row.record_chain_algorithm,
            final_hash=row.final_hash,
            signed_manifest_schema=row.signed_manifest_schema,
            signed_manifest_hash=row.signed_manifest_hash,
            signed_manifest_ref=row.signed_manifest_ref,
            signed_manifest_size_bytes=row.signed_manifest_size_bytes,
        )


class SinkEffectExportSnapshotAssociationLoader:
    """Strict row decoder for the export effect's single snapshot input."""

    def load(self, row: SARow[Any]) -> SinkEffectExportSnapshotAssociation:
        return SinkEffectExportSnapshotAssociation(
            effect_id=row.effect_id,
            input_kind=SinkEffectInputKind(row.input_kind),
            slot=row.slot,
            snapshot_id=row.snapshot_id,
        )


class SinkEffectAttemptLoader:
    """Strict row decoder for external effect attempts."""

    def load(self, row: SARow[Any]) -> SinkEffectAttempt:
        return SinkEffectAttempt(
            attempt_id=row.attempt_id,
            effect_id=row.effect_id,
            member_ordinal=row.member_ordinal,
            generation=row.generation,
            action=SinkEffectAttemptAction(row.action),
            call_kind=row.call_kind,
            request_hash=row.request_hash,
            state=SinkEffectAttemptState(row.state),
            evidence_json=row.evidence_json,
            evidence_hash=row.evidence_hash,
            started_at=row.started_at,
            completed_at=row.completed_at,
            latency_ms=row.latency_ms,
        )


class ArtifactLoader:
    """Loader for Artifact records.

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
            sink_effect_id=row.sink_effect_id,
            sink_node_id=row.sink_node_id,
            artifact_type=row.artifact_type,
            path_or_uri=row.path_or_uri,
            content_hash=row.content_hash,
            size_bytes=row.size_bytes,
            created_at=row.created_at,
            idempotency_key=row.idempotency_key,
            publication_performed=row.publication_performed,
            publication_evidence_kind=row.publication_evidence_kind,
        )


class BatchMemberLoader:
    """Loader for BatchMember records.

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
            run_id=row.run_id,
            token_id=row.token_id,
            ordinal=row.ordinal,
        )


class OperationLoader:
    """Loader for Operation records (source/sink I/O operations).

    No enum conversion needed — Operation uses Literal types validated
    by __post_init__(). This loader centralizes the row→dataclass mapping
    that was previously duplicated in get_operation() and get_operations_for_run().
    """

    def load(self, row: SARow[Any]) -> Operation:
        """Load Operation from database row.

        Validates operation_type, status, and status-dependent lifecycle
        invariants before construction.

        Raises:
            AuditIntegrityError: If operation_type/status invalid or lifecycle
                invariants violated (Tier 1)
        """
        oid = row.operation_id

        # Tier 1: validate constrained literal fields
        allowed_types = ("runtime_preflight", "source_load", "sink_write")
        if row.operation_type not in allowed_types:
            raise AuditIntegrityError(
                f"Operation {oid} has invalid operation_type={row.operation_type!r} "
                f"(expected one of {allowed_types}) — audit integrity violation"
            )

        allowed_statuses = ("open", "completed", "failed", "pending")
        if row.status not in allowed_statuses:
            raise AuditIntegrityError(
                f"Operation {oid} has invalid status={row.status!r} (expected one of {allowed_statuses}) — audit integrity violation"
            )

        # Tier 1: status-dependent lifecycle invariants
        if row.status == "open":
            if row.completed_at is not None:
                raise AuditIntegrityError(f"Operation {oid}: status='open' but completed_at is set — audit integrity violation")
            if row.duration_ms is not None:
                raise AuditIntegrityError(f"Operation {oid}: status='open' but duration_ms is set — audit integrity violation")
            if row.error_message is not None:
                raise AuditIntegrityError(f"Operation {oid}: status='open' but error_message is set — audit integrity violation")
        elif row.status in ("completed", "failed", "pending"):
            if row.completed_at is None:
                raise AuditIntegrityError(f"Operation {oid}: status={row.status!r} but completed_at is NULL — audit integrity violation")
            if row.duration_ms is None:
                raise AuditIntegrityError(f"Operation {oid}: status={row.status!r} but duration_ms is NULL — audit integrity violation")
            if row.status == "failed" and row.error_message is None:
                raise AuditIntegrityError(f"Operation {oid}: status='failed' but error_message is NULL — audit integrity violation")
            if row.status == "failed" and row.error_message == "":
                raise AuditIntegrityError(f"Operation {oid}: status='failed' but error_message is empty — audit integrity violation")
            if row.status == "completed" and row.error_message is not None:
                raise AuditIntegrityError(f"Operation {oid}: status='completed' but error_message is set — audit integrity violation")

        return Operation(
            operation_id=oid,
            run_id=row.run_id,
            node_id=row.node_id,
            operation_type=row.operation_type,
            started_at=row.started_at,
            completed_at=row.completed_at,
            status=row.status,
            sink_effect_id=row.sink_effect_id,
            input_data_ref=row.input_data_ref,
            input_data_hash=row.input_data_hash,
            output_data_ref=row.output_data_ref,
            output_data_hash=row.output_data_hash,
            error_message=row.error_message,
            duration_ms=row.duration_ms,
        )
