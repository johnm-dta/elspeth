"""Repository layer for Landscape audit models.

Handles the seam between SQLAlchemy rows (strings) and domain objects
(strict enum types). This is NOT a trust boundary - if the database
has bad data, we crash. That's intentional per Data Manifesto.

Per Data Manifesto: The audit database is OUR data. Bad data = crash.
"""

from typing import Any

from elspeth.contracts.audit import (
    Edge,
    Node,
    Row,
    Run,
    Token,
    TokenParent,
)
from elspeth.contracts.enums import (
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
