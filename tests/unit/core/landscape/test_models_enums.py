"""Tests for enum-typed model fields."""

from datetime import UTC, datetime

from elspeth.contracts import (
    Determinism,
    Edge,
    Node,
    NodeType,
    Run,
)


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
