# tests/integration/test_audit_integration_fixes.py
"""Integration tests verifying all audit integration fixes.

Task 8: Final Integration Test Suite
Verifies all integration audit fixes (Tasks 1-7) work together end-to-end.
"""

import pytest

from elspeth.contracts import EdgeInfo, ExecutionError, RoutingMode
from elspeth.core.dag import ExecutionGraph
from elspeth.core.landscape import LandscapeDB
from elspeth.core.landscape.recorder import LandscapeRecorder
from elspeth.plugins.context import PluginContext
from elspeth.plugins.manager import PluginManager


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

        # All built-in plugins discoverable
        assert len(manager.get_sources()) >= 2
        assert len(manager.get_transforms()) >= 2
        assert len(manager.get_gates()) >= 3
        assert len(manager.get_sinks()) >= 3

        # Instantiate a plugin and verify node_id
        csv_source_cls = manager.get_source_by_name("csv")
        assert csv_source_cls is not None
        # Protocols don't define __init__ but concrete classes do
        source = csv_source_cls({"path": "test.csv"})  # type: ignore[call-arg]
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
        graph.add_node("src", node_type="source", plugin_name="csv")
        graph.add_node("sink", node_type="sink", plugin_name="csv")
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
        error: ExecutionError = {
            "exception": "Test error",
            "type": "ValueError",
        }

        # Type checker validates this structure
        assert error["exception"] == "Test error"
        assert error["type"] == "ValueError"

        # Test with optional traceback field
        error_with_traceback: ExecutionError = {
            "exception": "Another error",
            "type": "RuntimeError",
            "traceback": "Traceback (most recent call last):\n  File ...",
        }
        assert "traceback" in error_with_traceback

    def test_plugin_context_accepts_real_recorder(self) -> None:
        """PluginContext accepts LandscapeRecorder without type issues.

        Verifies:
        - Task 6: PluginContext.landscape type fix
        """
        db = LandscapeDB.in_memory()
        recorder = LandscapeRecorder(db)

        ctx = PluginContext(
            run_id="test-run",
            config={},
            landscape=recorder,
        )

        assert ctx.landscape is recorder
        assert ctx.run_id == "test-run"

        # Cleanup
        db.close()

    def test_edge_info_immutability(self) -> None:
        """EdgeInfo dataclass is frozen (immutable).

        Verifies EdgeInfo contract integrity.
        """
        edge = EdgeInfo(
            from_node="a",
            to_node="b",
            label="continue",
            mode=RoutingMode.MOVE,
        )

        # Should be immutable
        with pytest.raises(AttributeError):
            edge.from_node = "c"  # type: ignore[misc]

    def test_routing_mode_is_enum_throughout_dag(self) -> None:
        """RoutingMode stays as enum through DAG operations.

        Verifies Task 3: RoutingMode enum alignment.
        """
        graph = ExecutionGraph()
        graph.add_node("src", node_type="source", plugin_name="csv")
        graph.add_node("gate", node_type="gate", plugin_name="filter")
        graph.add_node("sink1", node_type="sink", plugin_name="csv")
        graph.add_node("sink2", node_type="sink", plugin_name="json")

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

        # Test source
        csv_source_cls = manager.get_source_by_name("csv")
        assert csv_source_cls is not None
        # Protocols don't define __init__ but concrete classes do
        source = csv_source_cls({"path": "test.csv"})  # type: ignore[call-arg]
        assert hasattr(source, "node_id")
        source.node_id = "source-001"
        assert source.node_id == "source-001"

        # Test transform
        passthrough_cls = manager.get_transform_by_name("passthrough")
        assert passthrough_cls is not None
        # Protocols don't define __init__ but concrete classes do
        transform = passthrough_cls({})  # type: ignore[call-arg]
        assert hasattr(transform, "node_id")
        transform.node_id = "transform-001"
        assert transform.node_id == "transform-001"

        # Test gate
        filter_gate_cls = manager.get_gate_by_name("filter_gate")
        assert filter_gate_cls is not None
        # Protocols don't define __init__ but concrete classes do
        gate = filter_gate_cls({"field": "score", "greater_than": 50})  # type: ignore[call-arg]
        assert hasattr(gate, "node_id")
        gate.node_id = "gate-001"
        assert gate.node_id == "gate-001"

        # Test sink
        csv_sink_cls = manager.get_sink_by_name("csv")
        assert csv_sink_cls is not None
        # Protocols don't define __init__ but concrete classes do
        sink = csv_sink_cls({"path": "/tmp/test.csv"})  # type: ignore[call-arg]
        assert hasattr(sink, "node_id")
        sink.node_id = "sink-001"
        assert sink.node_id == "sink-001"

    def test_landscape_recorder_integration(self) -> None:
        """LandscapeRecorder works with PluginContext in realistic scenario.

        End-to-end test combining multiple fixes.
        """
        db = LandscapeDB.in_memory()
        recorder = LandscapeRecorder(db)

        # Begin a run
        run = recorder.begin_run(
            config={"datasource": {"plugin": "csv"}},
            canonical_version="1.0.0",
        )

        # Create context with recorder
        ctx = PluginContext(
            run_id=run.run_id,
            config={},
            landscape=recorder,
        )

        # Verify context has the recorder
        assert ctx.landscape is recorder
        assert ctx.run_id == run.run_id

        # Complete the run
        completed = recorder.complete_run(run.run_id, "completed")
        assert completed.run_id == run.run_id

        # Cleanup
        db.close()
