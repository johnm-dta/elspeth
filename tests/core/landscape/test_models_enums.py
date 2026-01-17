"""Tests for enum-typed model fields."""

from datetime import UTC, datetime

from elspeth.contracts.enums import (
    Determinism,
    ExportStatus,
    NodeType,
    RoutingMode,
    RunStatus,
)
from elspeth.core.landscape.models import Edge, Node, Run


class TestModelEnumTypes:
    """Verify model fields use enum types, not strings."""

    def test_run_export_status_accepts_enum(self) -> None:
        """Run.export_status should accept ExportStatus enum."""
        run = Run(
            run_id="run-1",
            started_at=datetime.now(UTC),
            config_hash="abc",
            settings_json="{}",
            canonical_version="v1",
            status=RunStatus.COMPLETED,
            export_status=ExportStatus.PENDING,
        )
        assert run.export_status == ExportStatus.PENDING

    def test_node_type_accepts_enum(self) -> None:
        """Node.node_type should accept NodeType enum."""
        node = Node(
            node_id="node-1",
            run_id="run-1",
            plugin_name="test",
            node_type=NodeType.TRANSFORM,
            plugin_version="1.0.0",
            determinism=Determinism.DETERMINISTIC,
            config_hash="abc",
            config_json="{}",
            registered_at=datetime.now(UTC),
        )
        assert node.node_type == NodeType.TRANSFORM

    def test_node_determinism_accepts_enum(self) -> None:
        """Node.determinism should accept Determinism enum."""
        node = Node(
            node_id="node-1",
            run_id="run-1",
            plugin_name="test",
            node_type=NodeType.SOURCE,
            plugin_version="1.0.0",
            determinism=Determinism.IO_READ,
            config_hash="abc",
            config_json="{}",
            registered_at=datetime.now(UTC),
        )
        assert node.determinism == Determinism.IO_READ

    def test_edge_default_mode_accepts_enum(self) -> None:
        """Edge.default_mode should accept RoutingMode enum."""
        edge = Edge(
            edge_id="edge-1",
            run_id="run-1",
            from_node_id="node-1",
            to_node_id="node-2",
            label="continue",
            default_mode=RoutingMode.MOVE,
            created_at=datetime.now(UTC),
        )
        assert edge.default_mode == RoutingMode.MOVE
