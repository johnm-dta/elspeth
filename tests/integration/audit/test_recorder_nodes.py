# tests/core/landscape/test_recorder_nodes.py
"""Tests for RecorderFactory node, edge, and schema operations."""

from __future__ import annotations

import pytest

from elspeth.contracts import NodeType, RoutingMode
from elspeth.contracts.schema import SchemaConfig
from elspeth.core.landscape.database import LandscapeDB
from elspeth.core.landscape.factory import RecorderFactory

# Dynamic schema for tests that don't care about specific fields
DYNAMIC_SCHEMA = SchemaConfig.from_dict({"mode": "observed"})


class TestRecorderFactoryNodes:
    """Node and edge registration."""

    def test_register_node(self) -> None:
        db = LandscapeDB.in_memory()
        factory = RecorderFactory(db)
        run = factory.run_lifecycle.begin_run(config={}, canonical_version="v1")

        node = factory.data_flow.register_node(
            run_id=run.run_id,
            plugin_name="csv_source",
            node_type=NodeType.SOURCE,
            plugin_version="1.0.0",
            config={"path": "data.csv"},
            sequence=0,
            schema_config=DYNAMIC_SCHEMA,
        )

        assert node.node_id is not None
        assert node.plugin_name == "csv_source"
        assert node.node_type == "source"

    def test_register_node_with_enum(self) -> None:
        """Test that NodeType enum is accepted and coerced."""
        db = LandscapeDB.in_memory()
        factory = RecorderFactory(db)
        run = factory.run_lifecycle.begin_run(config={}, canonical_version="v1")

        # Both enum and string should work
        node_from_enum = factory.data_flow.register_node(
            run_id=run.run_id,
            plugin_name="transform1",
            node_type=NodeType.TRANSFORM,  # Enum
            plugin_version="1.0.0",
            config={},
            schema_config=DYNAMIC_SCHEMA,
        )
        node_from_str = factory.data_flow.register_node(
            run_id=run.run_id,
            plugin_name="transform2",
            node_type=NodeType.TRANSFORM,  # Also enum now
            plugin_version="1.0.0",
            config={},
            schema_config=DYNAMIC_SCHEMA,
        )

        # Both should store the same string value
        assert node_from_enum.node_type == "transform"
        assert node_from_str.node_type == "transform"

    def test_register_node_invalid_type_raises(self) -> None:
        """Test that invalid node_type string raises TypeError."""
        db = LandscapeDB.in_memory()
        factory = RecorderFactory(db)
        run = factory.run_lifecycle.begin_run(config={}, canonical_version="v1")

        with pytest.raises(TypeError, match=r"node_type must be NodeType, got str: 'transfom'"):
            factory.data_flow.register_node(
                run_id=run.run_id,
                plugin_name="bad",
                node_type="transfom",  # type: ignore[arg-type]  # Intentionally wrong type to test validation
                plugin_version="1.0.0",
                config={},
                schema_config=DYNAMIC_SCHEMA,
            )

    def test_register_edge(self) -> None:
        db = LandscapeDB.in_memory()
        factory = RecorderFactory(db)
        run = factory.run_lifecycle.begin_run(config={}, canonical_version="v1")

        source = factory.data_flow.register_node(
            run_id=run.run_id,
            plugin_name="source",
            node_type=NodeType.SOURCE,
            plugin_version="1.0",
            config={},
            schema_config=DYNAMIC_SCHEMA,
        )
        transform = factory.data_flow.register_node(
            run_id=run.run_id,
            plugin_name="transform",
            node_type=NodeType.TRANSFORM,
            plugin_version="1.0",
            config={},
            schema_config=DYNAMIC_SCHEMA,
        )

        edge = factory.data_flow.register_edge(
            run_id=run.run_id,
            from_node_id=source.node_id,
            to_node_id=transform.node_id,
            label="continue",
            mode=RoutingMode.MOVE,
        )

        assert edge.edge_id is not None
        assert edge.label == "continue"

    def test_get_nodes_for_run(self) -> None:
        db = LandscapeDB.in_memory()
        factory = RecorderFactory(db)
        run = factory.run_lifecycle.begin_run(config={}, canonical_version="v1")

        factory.data_flow.register_node(
            run_id=run.run_id,
            plugin_name="source",
            node_type=NodeType.SOURCE,
            plugin_version="1.0",
            config={},
            schema_config=DYNAMIC_SCHEMA,
        )
        factory.data_flow.register_node(
            run_id=run.run_id,
            plugin_name="sink",
            node_type=NodeType.SINK,
            plugin_version="1.0",
            config={},
            schema_config=DYNAMIC_SCHEMA,
        )

        nodes = factory.data_flow.get_nodes(run.run_id)
        assert len(nodes) == 2


class TestRecorderFactoryEdges:
    """Edge query methods."""

    def test_get_edges_returns_all_edges_for_run(self) -> None:
        """get_edges should return all edges registered for a run."""
        db = LandscapeDB.in_memory()
        factory = RecorderFactory(db)

        run = factory.run_lifecycle.begin_run(config={}, canonical_version="v1")

        # Register nodes
        factory.data_flow.register_node(
            run_id=run.run_id,
            node_id="source_1",
            plugin_name="csv",
            node_type=NodeType.SOURCE,
            plugin_version="1.0.0",
            config={},
            schema_config=DYNAMIC_SCHEMA,
        )
        factory.data_flow.register_node(
            run_id=run.run_id,
            node_id="sink_1",
            plugin_name="csv",
            node_type=NodeType.SINK,
            plugin_version="1.0.0",
            config={},
            schema_config=DYNAMIC_SCHEMA,
        )

        # Register edge
        edge = factory.data_flow.register_edge(
            run_id=run.run_id,
            from_node_id="source_1",
            to_node_id="sink_1",
            label="continue",
            mode=RoutingMode.MOVE,
        )

        # Query edges
        edges = factory.data_flow.get_edges(run.run_id)

        assert len(edges) == 1
        assert edges[0].edge_id == edge.edge_id
        assert edges[0].from_node_id == "source_1"
        assert edges[0].to_node_id == "sink_1"
        assert edges[0].default_mode == "move"

    def test_get_edges_returns_empty_list_for_run_with_no_edges(self) -> None:
        """get_edges should return empty list when no edges exist."""
        db = LandscapeDB.in_memory()
        factory = RecorderFactory(db)

        run = factory.run_lifecycle.begin_run(config={}, canonical_version="v1")

        edges = factory.data_flow.get_edges(run.run_id)

        assert edges == []

    def test_get_edges_returns_multiple_edges(self) -> None:
        """get_edges should return all edges when multiple exist."""
        db = LandscapeDB.in_memory()
        factory = RecorderFactory(db)

        run = factory.run_lifecycle.begin_run(config={}, canonical_version="v1")

        # Register nodes
        factory.data_flow.register_node(
            run_id=run.run_id,
            node_id="source",
            plugin_name="csv",
            node_type=NodeType.SOURCE,
            plugin_version="1.0.0",
            config={},
            schema_config=DYNAMIC_SCHEMA,
        )
        factory.data_flow.register_node(
            run_id=run.run_id,
            node_id="gate",
            plugin_name="threshold",
            node_type=NodeType.GATE,
            plugin_version="1.0.0",
            config={},
            schema_config=DYNAMIC_SCHEMA,
        )
        factory.data_flow.register_node(
            run_id=run.run_id,
            node_id="sink_high",
            plugin_name="csv",
            node_type=NodeType.SINK,
            plugin_version="1.0.0",
            config={},
            schema_config=DYNAMIC_SCHEMA,
        )
        factory.data_flow.register_node(
            run_id=run.run_id,
            node_id="sink_low",
            plugin_name="csv",
            node_type=NodeType.SINK,
            plugin_version="1.0.0",
            config={},
            schema_config=DYNAMIC_SCHEMA,
        )

        # Register edges
        factory.data_flow.register_edge(
            run_id=run.run_id,
            from_node_id="source",
            to_node_id="gate",
            label="continue",
            mode=RoutingMode.MOVE,
        )
        factory.data_flow.register_edge(
            run_id=run.run_id,
            from_node_id="gate",
            to_node_id="sink_high",
            label="high",
            mode=RoutingMode.MOVE,
        )
        factory.data_flow.register_edge(
            run_id=run.run_id,
            from_node_id="gate",
            to_node_id="sink_low",
            label="low",
            mode=RoutingMode.MOVE,
        )

        # Query edges
        edges = factory.data_flow.get_edges(run.run_id)

        assert len(edges) == 3


class TestSchemaRecording:
    """Tests for schema configuration recording in audit trail."""

    def test_register_node_with_dynamic_schema(self) -> None:
        """Dynamic schema recorded in node registration."""
        db = LandscapeDB.in_memory()
        factory = RecorderFactory(db)
        run = factory.run_lifecycle.begin_run(config={}, canonical_version="v1")

        schema_config = SchemaConfig.from_dict({"mode": "observed"})

        node = factory.data_flow.register_node(
            run_id=run.run_id,
            plugin_name="csv_source",
            node_type=NodeType.SOURCE,
            plugin_version="1.0.0",
            config={"path": "data.csv"},
            schema_config=schema_config,
        )

        retrieved = factory.data_flow.get_node(node.node_id, run.run_id)
        assert retrieved is not None
        assert retrieved.schema_mode == "observed"
        assert retrieved.schema_fields is None

    def test_register_node_with_explicit_schema(self) -> None:
        """Explicit schema fields recorded in node registration."""
        db = LandscapeDB.in_memory()
        factory = RecorderFactory(db)
        run = factory.run_lifecycle.begin_run(config={}, canonical_version="v1")

        schema_config = SchemaConfig.from_dict(
            {
                "mode": "fixed",
                "fields": ["id: int", "name: str"],
            }
        )

        node = factory.data_flow.register_node(
            run_id=run.run_id,
            plugin_name="csv_source",
            node_type=NodeType.SOURCE,
            plugin_version="1.0.0",
            config={"path": "data.csv"},
            schema_config=schema_config,
        )

        retrieved = factory.data_flow.get_node(node.node_id, run.run_id)
        assert retrieved is not None
        assert retrieved.schema_mode == "fixed"
        assert retrieved.schema_fields is not None
        assert len(retrieved.schema_fields) == 2
        assert retrieved.schema_fields[0]["name"] == "id"
        assert retrieved.schema_fields[1]["name"] == "name"

    def test_register_node_with_free_schema(self) -> None:
        """Free schema (at least these fields) recorded in node registration."""
        db = LandscapeDB.in_memory()
        factory = RecorderFactory(db)
        run = factory.run_lifecycle.begin_run(config={}, canonical_version="v1")

        schema_config = SchemaConfig.from_dict(
            {
                "mode": "flexible",
                "fields": ["id: int", "name: str", "score: float?"],
            }
        )

        node = factory.data_flow.register_node(
            run_id=run.run_id,
            plugin_name="csv_source",
            node_type=NodeType.SOURCE,
            plugin_version="1.0.0",
            config={"path": "data.csv"},
            schema_config=schema_config,
        )

        retrieved = factory.data_flow.get_node(node.node_id, run.run_id)
        assert retrieved is not None
        assert retrieved.schema_mode == "flexible"
        assert retrieved.schema_fields is not None
        assert len(retrieved.schema_fields) == 3
        # Verify optional field is marked correctly
        assert retrieved.schema_fields[2]["name"] == "score"
        assert retrieved.schema_fields[2]["required"] is False

    def test_get_nodes_includes_schema_info(self) -> None:
        """get_nodes() returns nodes with schema information."""
        db = LandscapeDB.in_memory()
        factory = RecorderFactory(db)
        run = factory.run_lifecycle.begin_run(config={}, canonical_version="v1")

        # Register one node with explicit schema
        schema_config = SchemaConfig.from_dict(
            {
                "mode": "fixed",
                "fields": ["value: int"],
            }
        )
        factory.data_flow.register_node(
            run_id=run.run_id,
            plugin_name="csv_source",
            node_type=NodeType.SOURCE,
            plugin_version="1.0.0",
            config={},
            schema_config=schema_config,
        )

        nodes = factory.data_flow.get_nodes(run.run_id)
        assert len(nodes) == 1
        assert nodes[0].schema_mode == "fixed"
        assert nodes[0].schema_fields is not None
        assert len(nodes[0].schema_fields) == 1
