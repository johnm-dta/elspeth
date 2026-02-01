"""Tests for enum-typed model fields."""

from datetime import UTC, datetime

from elspeth.contracts import (
    Determinism,
    Edge,
    ExportStatus,
    Node,
    NodeType,
    RoutingMode,
    Run,
    RunStatus,
)


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

    def test_enum_type_verified_not_just_value(self) -> None:
        """P2: Verify fields are actual enum types, not just matching values.

        Regressions that convert enum fields to strings would still pass
        value comparisons. This test verifies runtime type.
        """
        run = Run(
            run_id="run-1",
            started_at=datetime.now(UTC),
            config_hash="abc",
            settings_json="{}",
            canonical_version="v1",
            status=RunStatus.COMPLETED,
            export_status=ExportStatus.PENDING,
        )
        # Must be actual enum instances, not strings
        assert isinstance(run.status, RunStatus), f"status should be RunStatus enum, got {type(run.status)}"
        assert isinstance(run.export_status, ExportStatus), f"export_status should be ExportStatus enum, got {type(run.export_status)}"
        assert run.status.value == "completed"  # Verify .value accessor works


class TestModelEnumTier1Rejection:
    """P1: Tier 1 corruption tests - invalid enum values must crash.

    Per the Three-Tier Trust Model, invalid data in audit models (Tier 1)
    must crash immediately - no coercion, no defaults, no silent handling.
    """

    def test_node_type_rejects_string(self) -> None:
        """Node.node_type must reject string 'transform' (wants NodeType.TRANSFORM)."""
        import pytest

        with pytest.raises(TypeError, match=r"node_type must be NodeType, got str: 'transform'"):
            Node(
                node_id="node-1",
                run_id="run-1",
                plugin_name="test",
                node_type="transform",  # type: ignore[arg-type]
                plugin_version="1.0.0",
                determinism=Determinism.DETERMINISTIC,
                config_hash="abc",
                config_json="{}",
                registered_at=datetime.now(UTC),
            )

    def test_node_type_rejects_integer(self) -> None:
        """Node.node_type must reject integer values."""
        import pytest

        with pytest.raises(TypeError, match=r"node_type must be NodeType, got int: 1"):
            Node(
                node_id="node-1",
                run_id="run-1",
                plugin_name="test",
                node_type=1,  # type: ignore[arg-type]
                plugin_version="1.0.0",
                determinism=Determinism.DETERMINISTIC,
                config_hash="abc",
                config_json="{}",
                registered_at=datetime.now(UTC),
            )

    def test_determinism_rejects_string(self) -> None:
        """Node.determinism must reject string 'deterministic'."""
        import pytest

        with pytest.raises(TypeError, match=r"determinism must be Determinism, got str: 'deterministic'"):
            Node(
                node_id="node-1",
                run_id="run-1",
                plugin_name="test",
                node_type=NodeType.TRANSFORM,
                plugin_version="1.0.0",
                determinism="deterministic",  # type: ignore[arg-type]
                config_hash="abc",
                config_json="{}",
                registered_at=datetime.now(UTC),
            )

    def test_run_status_rejects_string(self) -> None:
        """Run.status must reject string 'completed'."""
        import pytest

        with pytest.raises(TypeError, match=r"status must be RunStatus, got str: 'completed'"):
            Run(
                run_id="run-1",
                started_at=datetime.now(UTC),
                config_hash="abc",
                settings_json="{}",
                canonical_version="v1",
                status="completed",  # type: ignore[arg-type]
            )

    def test_edge_routing_mode_rejects_string(self) -> None:
        """Edge.default_mode must reject string 'move'."""
        import pytest

        with pytest.raises(TypeError, match=r"default_mode must be RoutingMode, got str: 'move'"):
            Edge(
                edge_id="edge-1",
                run_id="run-1",
                from_node_id="node-1",
                to_node_id="node-2",
                label="continue",
                default_mode="move",  # type: ignore[arg-type]
                created_at=datetime.now(UTC),
            )
