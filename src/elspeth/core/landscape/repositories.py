"""Repository layer for Landscape audit models.

Handles the seam between SQLAlchemy rows (strings) and domain objects
(strict enum types). This is NOT a trust boundary - if the database
has bad data, we crash. That's intentional per Data Manifesto.

Per Data Manifesto: The audit database is OUR data. Bad data = crash.
"""

from typing import Any

from elspeth.contracts.audit import (
    Batch,
    Call,
    Edge,
    Node,
    RoutingEvent,
    Row,
    Run,
    Token,
    TokenParent,
)
from elspeth.contracts.enums import (
    BatchStatus,
    CallStatus,
    CallType,
    Determinism,
    ExportStatus,
    NodeType,
    RoutingMode,
    RunStatus,
)


class RunRepository:
    """Repository for Run records."""

    def __init__(self, session: Any) -> None:
        self.session = session

    def load(self, row: Any) -> Run:
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
            export_status=ExportStatus(row.export_status)
            if row.export_status
            else None,
            export_error=row.export_error,
            exported_at=row.exported_at,
            export_format=row.export_format,
            export_sink=row.export_sink,
        )


class NodeRepository:
    """Repository for Node records."""

    def __init__(self, session: Any) -> None:
        self.session = session

    def load(self, row: Any) -> Node:
        """Load Node from database row.

        Converts node_type and determinism strings to enums.
        """
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
        )


class EdgeRepository:
    """Repository for Edge records."""

    def __init__(self, session: Any) -> None:
        self.session = session

    def load(self, row: Any) -> Edge:
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

    def __init__(self, session: Any) -> None:
        self.session = session

    def load(self, row: Any) -> Row:
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

    def __init__(self, session: Any) -> None:
        self.session = session

    def load(self, row: Any) -> Token:
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

    def __init__(self, session: Any) -> None:
        self.session = session

    def load(self, row: Any) -> TokenParent:
        """Load TokenParent from database row."""
        return TokenParent(
            token_id=row.token_id,
            parent_token_id=row.parent_token_id,
            ordinal=row.ordinal,
        )


class CallRepository:
    """Repository for Call records."""

    def __init__(self, session: Any) -> None:
        self.session = session

    def load(self, row: Any) -> Call:
        """Load Call from database row."""
        return Call(
            call_id=row.call_id,
            state_id=row.state_id,
            call_index=row.call_index,
            call_type=CallType(row.call_type),  # Convert HERE
            status=CallStatus(row.status),  # Convert HERE
            request_hash=row.request_hash,
            created_at=row.created_at,
            request_ref=row.request_ref,
            response_hash=row.response_hash,
            response_ref=row.response_ref,
            error_json=row.error_json,
            latency_ms=row.latency_ms,
        )


class RoutingEventRepository:
    """Repository for RoutingEvent records."""

    def __init__(self, session: Any) -> None:
        self.session = session

    def load(self, row: Any) -> RoutingEvent:
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

    def __init__(self, session: Any) -> None:
        self.session = session

    def load(self, row: Any) -> Batch:
        """Load Batch from database row."""
        return Batch(
            batch_id=row.batch_id,
            run_id=row.run_id,
            aggregation_node_id=row.aggregation_node_id,
            attempt=row.attempt,
            status=BatchStatus(row.status),  # Convert HERE
            created_at=row.created_at,
            aggregation_state_id=row.aggregation_state_id,
            trigger_reason=row.trigger_reason,
            completed_at=row.completed_at,
        )
