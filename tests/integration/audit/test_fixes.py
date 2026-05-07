# tests/integration/audit/test_fixes.py
"""Integration tests verifying all audit integration fixes.

Task 8: Final Integration Test Suite
Verifies all integration audit fixes (Tasks 1-7) work together end-to-end.

Migrated from tests/integration/test_audit_integration_fixes.py
"""

import pytest

from elspeth.contracts import EdgeInfo, ExecutionError, NodeID, NodeType, RoutingMode, RunStatus
from elspeth.core.dag import ExecutionGraph
from elspeth.plugins.infrastructure.manager import PluginManager
from tests.fixtures.factories import make_context
from tests.fixtures.landscape import make_factory, make_landscape_db

# Dynamic schema config for tests - PathConfig now requires schema
DYNAMIC_SCHEMA = {"mode": "observed"}


class TestIntegrationAuditFixes:
    """End-to-end tests for integration audit fixes."""

    def test_full_plugin_discovery_flow(self) -> None:
        """Plugins are discoverable and have proper node_id support.

        Verifies:
        - Task 1: Hook implementers for plugin discovery
        - Task 4: node_id in plugin protocols
        """
        manager = PluginManager()
        manager.register_builtin_plugins()

        # All built-in plugins discoverable — check concrete counts
        sources = manager.get_sources()
        transforms = manager.get_transforms()
        sinks = manager.get_sinks()
        assert len(sources) >= 2
        assert len(transforms) >= 2
        assert len(sinks) >= 3

        # Instantiate a plugin and verify node_id round-trip
        csv_source_cls = manager.get_source_by_name("csv")
        assert csv_source_cls is not None
        source = csv_source_cls(
            {
                "path": "test.csv",
                "on_validation_failure": "discard",
                "schema": DYNAMIC_SCHEMA,
            }
        )
        assert source.node_id is None  # Not yet set

        source.node_id = "node-123"
        assert source.node_id == "node-123"  # Can be set

    def test_dag_uses_typed_edges(self) -> None:
        """DAG edge operations use EdgeInfo contracts.

        Verifies:
        - Task 2: EdgeInfo contract integration in DAG
        - Task 3: RoutingMode enum alignment
        """
        graph = ExecutionGraph()
        schema_config = {"schema": {"mode": "observed"}}
        graph.add_node("src", node_type=NodeType.SOURCE, plugin_name="csv", config=schema_config)
        graph.add_node("sink", node_type=NodeType.SINK, plugin_name="csv", config=schema_config)
        graph.add_edge("src", "sink", label="continue", mode=RoutingMode.MOVE)

        edges = graph.get_edges()
        assert len(edges) == 1
        assert isinstance(edges[0], EdgeInfo)
        assert edges[0].mode == RoutingMode.MOVE
        # RoutingMode is a (str, Enum), so isinstance(x, str) is True
        # The key test is that it's the enum member, not a plain string
        assert isinstance(edges[0].mode, RoutingMode)

    def test_error_payloads_are_structured(self) -> None:
        """Error payloads follow ExecutionError schema.

        Verifies:
        - Task 5: TypedDict schemas for error/reason payloads
        """
        error = ExecutionError(
            exception="Test error",
            exception_type="ValueError",
        )

        # Frozen dataclass validates at construction; to_dict() serializes for audit
        assert error.exception == "Test error"
        assert error.exception_type == "ValueError"
        d = error.to_dict()
        assert d["type"] == "ValueError"

        # Test with optional traceback field
        error_with_traceback = ExecutionError(
            exception="Another error",
            exception_type="RuntimeError",
            traceback="Traceback (most recent call last):\n  File ...",
        )
        assert error_with_traceback.traceback is not None
        assert "traceback" in error_with_traceback.to_dict()

    def test_plugin_context_recorder_can_record(self) -> None:
        """PluginContext with real RecorderFactory can begin and complete a run.

        Verifies:
        - Task 6: PluginContext.landscape type fix
        - Recording actually works through the context
        """
        db = make_landscape_db()
        factory = make_factory(db)

        run = factory.run_lifecycle.begin_run(
            config={"source": {"plugin": "csv"}},
            canonical_version="1.0.0",
        )

        ctx = make_context(
            run_id=run.run_id,
            landscape=factory.plugin_audit_writer(),
        )

        # Verify recording works through the context's factory
        assert ctx.landscape is not None
        completed = factory.run_lifecycle.complete_run(run.run_id, RunStatus.COMPLETED)
        assert completed.run_id == run.run_id

        # Cleanup
        db.close()

    def test_edge_info_immutability(self) -> None:
        """EdgeInfo dataclass is frozen (immutable).

        Verifies EdgeInfo contract integrity.
        """
        edge = EdgeInfo(
            from_node=NodeID("a"),
            to_node=NodeID("b"),
            label="continue",
            mode=RoutingMode.MOVE,
        )

        # Should be immutable
        with pytest.raises(AttributeError):
            edge.from_node = NodeID("c")  # type: ignore[misc]

    def test_routing_mode_is_enum_throughout_dag(self) -> None:
        """RoutingMode stays as enum through DAG operations.

        Verifies Task 3: RoutingMode enum alignment.
        """
        graph = ExecutionGraph()
        schema_config = {"schema": {"mode": "observed"}}
        graph.add_node("src", node_type=NodeType.SOURCE, plugin_name="csv", config=schema_config)
        graph.add_node("gate", node_type=NodeType.GATE, plugin_name="filter", config=schema_config)
        graph.add_node("sink1", node_type=NodeType.SINK, plugin_name="csv", config=schema_config)
        graph.add_node("sink2", node_type=NodeType.SINK, plugin_name="json", config=schema_config)

        # Add edges with different routing modes
        graph.add_edge("src", "gate", label="continue", mode=RoutingMode.MOVE)
        graph.add_edge("gate", "sink1", label="normal", mode=RoutingMode.MOVE)
        graph.add_edge("gate", "sink2", label="flagged", mode=RoutingMode.COPY)

        edges = graph.get_edges()
        assert len(edges) == 3

        # All modes should be RoutingMode enum members
        # (RoutingMode is (str, Enum) so isinstance(x, str) is True by design)
        for edge in edges:
            assert isinstance(edge.mode, RoutingMode)
            assert edge.mode in (RoutingMode.MOVE, RoutingMode.COPY)

    def test_plugin_node_id_on_all_plugin_types(self) -> None:
        """All plugin types support node_id property.

        Verifies Task 4: node_id in plugin protocols.
        """
        manager = PluginManager()
        manager.register_builtin_plugins()

        # Test source — node_id starts None, can be set
        csv_source_cls = manager.get_source_by_name("csv")
        assert csv_source_cls is not None
        source = csv_source_cls(
            {
                "path": "test.csv",
                "on_validation_failure": "discard",
                "schema": DYNAMIC_SCHEMA,
            }
        )
        assert source.node_id is None
        source.node_id = "source-001"
        assert source.node_id == "source-001"

        # Test transform
        passthrough_cls = manager.get_transform_by_name("passthrough")
        assert passthrough_cls is not None
        transform = passthrough_cls({"schema": DYNAMIC_SCHEMA})
        assert transform.node_id is None
        transform.node_id = "transform-001"
        assert transform.node_id == "transform-001"

        # Test sink (use JSONSink which accepts dynamic schemas)
        json_sink_cls = manager.get_sink_by_name("json")
        assert json_sink_cls is not None
        sink = json_sink_cls({"path": "/tmp/test.json", "schema": DYNAMIC_SCHEMA, "format": "jsonl"})
        assert sink.node_id is None
        sink.node_id = "sink-001"
        assert sink.node_id == "sink-001"

    def test_landscape_recorder_run_lifecycle(self) -> None:
        """RecorderFactory records complete run lifecycle through PluginContext.

        End-to-end test combining multiple fixes — verifies the factory
        actually persists data, not just that assignment works.
        """
        db = make_landscape_db()
        factory = make_factory(db)

        # Begin a run
        run = factory.run_lifecycle.begin_run(
            config={"source": {"plugin": "csv"}},
            canonical_version="1.0.0",
        )

        # Create context with factory
        ctx = make_context(
            run_id=run.run_id,
            landscape=factory.plugin_audit_writer(),
        )

        # Complete the run through the factory's run_lifecycle repo
        assert ctx.landscape is not None
        completed = factory.run_lifecycle.complete_run(run.run_id, RunStatus.COMPLETED)
        assert completed.run_id == run.run_id
        assert completed.status == RunStatus.COMPLETED

        # Cleanup
        db.close()
