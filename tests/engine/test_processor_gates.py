# tests/engine/test_processor_gates.py
"""Tests for RowProcessor gate handling.

Gate routing, forking, and nested fork tests extracted from test_processor.py.
Test plugins inherit from base classes (BaseTransform) because the processor
uses isinstance() for type-safe plugin detection. Gates are config-driven using
GateSettings.
"""

from typing import Any

from elspeth.contracts import NodeType, RoutingMode
from elspeth.contracts.types import GateName, NodeID
from tests.engine.conftest import DYNAMIC_SCHEMA, _TestSchema


class TestRowProcessorGates:
    """Gate handling in RowProcessor."""

    def test_gate_continue_proceeds(self) -> None:
        """Gate returning continue proceeds to completion."""
        from elspeth.core.config import GateSettings
        from elspeth.core.landscape import LandscapeDB, LandscapeRecorder
        from elspeth.engine.processor import RowProcessor
        from elspeth.engine.spans import SpanFactory
        from elspeth.plugins.base import BaseTransform
        from elspeth.plugins.context import PluginContext
        from elspeth.plugins.results import RowOutcome, TransformResult

        db = LandscapeDB.in_memory()
        recorder = LandscapeRecorder(db)
        run = recorder.begin_run(config={}, canonical_version="v1")

        source = recorder.register_node(
            run_id=run.run_id,
            plugin_name="source",
            node_type=NodeType.SOURCE,
            plugin_version="1.0",
            config={},
            schema_config=DYNAMIC_SCHEMA,
        )
        transform = recorder.register_node(
            run_id=run.run_id,
            plugin_name="final",
            node_type=NodeType.TRANSFORM,
            plugin_version="1.0",
            config={},
            schema_config=DYNAMIC_SCHEMA,
        )
        gate = recorder.register_node(
            run_id=run.run_id,
            plugin_name="pass_gate",
            node_type=NodeType.GATE,
            plugin_version="1.0",
            config={},
            schema_config=DYNAMIC_SCHEMA,
        )
        # AUD-002: Register continue edge for audit completeness
        continue_edge = recorder.register_edge(
            run_id=run.run_id,
            from_node_id=gate.node_id,
            to_node_id=transform.node_id,  # Gate continues to transform
            label="continue",
            mode=RoutingMode.MOVE,
        )

        class FinalTransform(BaseTransform):
            name = "final"
            input_schema = _TestSchema
            output_schema = _TestSchema

            def __init__(self, node_id: str) -> None:
                super().__init__({"schema": {"fields": "dynamic"}})
                self.node_id = node_id

            def process(self, row: dict[str, Any], ctx: PluginContext) -> TransformResult:
                return TransformResult.success({**row, "final": True}, success_reason={"action": "test"})

        # Config-driven gate: always continues
        pass_gate = GateSettings(
            name="pass_gate",
            condition="True",
            routes={"true": "continue", "false": "continue"},
        )

        ctx = PluginContext(run_id=run.run_id, config={})
        processor = RowProcessor(
            recorder=recorder,
            span_factory=SpanFactory(),
            run_id=run.run_id,
            source_node_id=NodeID(source.node_id),
            config_gates=[pass_gate],
            config_gate_id_map={GateName("pass_gate"): NodeID(gate.node_id)},
            edge_map={
                (NodeID(gate.node_id), "continue"): continue_edge.edge_id,
            },  # AUD-002: Required for continue routing events
        )

        results = processor.process_row(
            row_index=0,
            row_data={"value": 42},
            transforms=[FinalTransform(transform.node_id)],
            ctx=ctx,
        )

        # Single result - no forks
        assert len(results) == 1
        result = results[0]

        assert result.final_data == {"value": 42, "final": True}
        assert result.outcome == RowOutcome.COMPLETED

    def test_gate_route_to_sink(self) -> None:
        """Gate routing via route label returns routed outcome with sink name."""
        from elspeth.core.config import GateSettings
        from elspeth.core.landscape import LandscapeDB, LandscapeRecorder
        from elspeth.engine.processor import RowProcessor
        from elspeth.engine.spans import SpanFactory
        from elspeth.plugins.context import PluginContext
        from elspeth.plugins.results import RowOutcome

        db = LandscapeDB.in_memory()
        recorder = LandscapeRecorder(db)
        run = recorder.begin_run(config={}, canonical_version="v1")

        source = recorder.register_node(
            run_id=run.run_id,
            plugin_name="source",
            node_type=NodeType.SOURCE,
            plugin_version="1.0",
            config={},
            schema_config=DYNAMIC_SCHEMA,
        )
        gate = recorder.register_node(
            run_id=run.run_id,
            plugin_name="router",
            node_type=NodeType.GATE,
            plugin_version="1.0",
            config={},
            schema_config=DYNAMIC_SCHEMA,
        )
        sink = recorder.register_node(
            run_id=run.run_id,
            plugin_name="high_values",
            node_type=NodeType.SINK,
            plugin_version="1.0",
            config={},
            schema_config=DYNAMIC_SCHEMA,
        )

        # Register edge using route label
        edge = recorder.register_edge(
            run_id=run.run_id,
            from_node_id=gate.node_id,
            to_node_id=sink.node_id,
            label="true",  # Route label for true condition
            mode=RoutingMode.MOVE,
        )

        # Config-driven gate: routes values > 100 to sink, else continues
        router_gate = GateSettings(
            name="router",
            condition="row['value'] > 100",
            routes={"true": "high_values", "false": "continue"},
        )

        ctx = PluginContext(run_id=run.run_id, config={})
        edge_map = {(NodeID(gate.node_id), "true"): edge.edge_id}
        # Route resolution map: label -> sink_name
        route_resolution_map = {(NodeID(gate.node_id), "true"): "high_values"}
        processor = RowProcessor(
            recorder=recorder,
            span_factory=SpanFactory(),
            run_id=run.run_id,
            source_node_id=NodeID(source.node_id),
            edge_map=edge_map,
            route_resolution_map=route_resolution_map,
            config_gates=[router_gate],
            config_gate_id_map={GateName("router"): NodeID(gate.node_id)},
        )

        results = processor.process_row(
            row_index=0,
            row_data={"value": 150},
            transforms=[],
            ctx=ctx,
        )

        # Single result - routed to sink
        assert len(results) == 1
        result = results[0]

        assert result.outcome == RowOutcome.ROUTED
        assert result.sink_name == "high_values"
        assert result.final_data == {"value": 150}

    def test_gate_fork_returns_forked(self) -> None:
        """Gate forking returns forked outcome (linear pipeline mode)."""
        from elspeth.core.config import GateSettings
        from elspeth.core.landscape import LandscapeDB, LandscapeRecorder
        from elspeth.engine.processor import RowProcessor
        from elspeth.engine.spans import SpanFactory
        from elspeth.plugins.context import PluginContext
        from elspeth.plugins.results import RowOutcome

        db = LandscapeDB.in_memory()
        recorder = LandscapeRecorder(db)
        run = recorder.begin_run(config={}, canonical_version="v1")

        source = recorder.register_node(
            run_id=run.run_id,
            plugin_name="source",
            node_type=NodeType.SOURCE,
            plugin_version="1.0",
            config={},
            schema_config=DYNAMIC_SCHEMA,
        )
        gate = recorder.register_node(
            run_id=run.run_id,
            plugin_name="splitter",
            node_type=NodeType.GATE,
            plugin_version="1.0",
            config={},
            schema_config=DYNAMIC_SCHEMA,
        )
        path_a = recorder.register_node(
            run_id=run.run_id,
            plugin_name="path_a",
            node_type=NodeType.TRANSFORM,
            plugin_version="1.0",
            config={},
            schema_config=DYNAMIC_SCHEMA,
        )
        path_b = recorder.register_node(
            run_id=run.run_id,
            plugin_name="path_b",
            node_type=NodeType.TRANSFORM,
            plugin_version="1.0",
            config={},
            schema_config=DYNAMIC_SCHEMA,
        )

        # Register edges for fork
        edge_a = recorder.register_edge(
            run_id=run.run_id,
            from_node_id=gate.node_id,
            to_node_id=path_a.node_id,
            label="path_a",
            mode=RoutingMode.COPY,
        )
        edge_b = recorder.register_edge(
            run_id=run.run_id,
            from_node_id=gate.node_id,
            to_node_id=path_b.node_id,
            label="path_b",
            mode=RoutingMode.COPY,
        )

        # Config-driven fork gate: always forks to path_a and path_b
        splitter_gate = GateSettings(
            name="splitter",
            condition="True",
            routes={"true": "fork", "false": "continue"},
            fork_to=["path_a", "path_b"],
        )

        ctx = PluginContext(run_id=run.run_id, config={})
        edge_map = {
            (NodeID(gate.node_id), "path_a"): edge_a.edge_id,
            (NodeID(gate.node_id), "path_b"): edge_b.edge_id,
        }
        processor = RowProcessor(
            recorder=recorder,
            span_factory=SpanFactory(),
            run_id=run.run_id,
            source_node_id=NodeID(source.node_id),
            edge_map=edge_map,
            config_gates=[splitter_gate],
            config_gate_id_map={GateName("splitter"): NodeID(gate.node_id)},
        )

        results = processor.process_row(
            row_index=0,
            row_data={"value": 42},
            transforms=[],
            ctx=ctx,
        )

        # Fork creates 3 results: parent (FORKED) + 2 children (COMPLETED)
        # Children have no remaining transforms, so they reach COMPLETED
        assert len(results) == 3

        forked_results = [r for r in results if r.outcome == RowOutcome.FORKED]
        completed_results = [r for r in results if r.outcome == RowOutcome.COMPLETED]

        assert len(forked_results) == 1
        assert len(completed_results) == 2

        # Parent has FORKED outcome
        parent = forked_results[0]
        assert parent.final_data == {"value": 42}

        # Children completed with original data (no transforms after fork)
        for child in completed_results:
            assert child.final_data == {"value": 42}
            assert child.token.branch_name in ("path_a", "path_b")

        # === P1: Audit trail verification for FORKED ===
        # Verify FORKED outcome for parent (processor records this)
        parent_outcome = recorder.get_token_outcome(parent.token.token_id)
        assert parent_outcome is not None, "Parent token outcome should be recorded"
        assert parent_outcome.outcome == RowOutcome.FORKED, "Parent should be FORKED"
        assert parent_outcome.fork_group_id is not None, "Fork group ID should be set"
        assert parent_outcome.is_terminal is True, "FORKED is terminal"

        # Verify children have correct lineage via get_token_parents
        # Note: COMPLETED token_outcomes for children are recorded by orchestrator at sink,
        # but parent relationships (fork lineage) are recorded by processor via TokenManager
        for child in completed_results:
            parents = recorder.get_token_parents(child.token.token_id)
            assert len(parents) == 1, "Each child should have exactly 1 parent"
            assert parents[0].parent_token_id == parent.token.token_id, "Parent should be the forked token"


class TestRowProcessorNestedForks:
    """Nested fork tests for work queue execution."""

    def test_nested_forks_all_children_executed(self) -> None:
        """Nested forks should execute all descendants.

        Pipeline: source -> transform -> gate1 (fork 2) -> gate2 (fork 2)

        Expected token tree:
        - 1 parent FORKED at gate1 (with count=1 from transform)
        - 2 children FORKED at gate2 (inherit count=1)
        - 4 grandchildren COMPLETED (inherit count=1)
        Total: 7 results
        """
        from elspeth.core.config import GateSettings
        from elspeth.core.landscape import LandscapeDB, LandscapeRecorder
        from elspeth.engine.processor import RowProcessor
        from elspeth.engine.spans import SpanFactory
        from elspeth.plugins.base import BaseTransform
        from elspeth.plugins.context import PluginContext
        from elspeth.plugins.results import RowOutcome, TransformResult

        db = LandscapeDB.in_memory()
        recorder = LandscapeRecorder(db)
        run = recorder.begin_run(config={}, canonical_version="v1")

        # Setup nodes for: source -> transform -> gate1 (fork 2) -> gate2 (fork 2)
        source_node = recorder.register_node(
            run_id=run.run_id,
            plugin_name="test_source",
            node_type=NodeType.SOURCE,
            plugin_version="1.0",
            config={},
            schema_config=DYNAMIC_SCHEMA,
        )
        transform_node = recorder.register_node(
            run_id=run.run_id,
            plugin_name="marker",
            node_type=NodeType.TRANSFORM,
            plugin_version="1.0",
            config={},
            schema_config=DYNAMIC_SCHEMA,
        )
        gate1_node = recorder.register_node(
            run_id=run.run_id,
            plugin_name="fork_gate_1",
            node_type=NodeType.GATE,
            plugin_version="1.0",
            config={},
            schema_config=DYNAMIC_SCHEMA,
        )
        gate2_node = recorder.register_node(
            run_id=run.run_id,
            plugin_name="fork_gate_2",
            node_type=NodeType.GATE,
            plugin_version="1.0",
            config={},
            schema_config=DYNAMIC_SCHEMA,
        )

        # Register edges for both fork paths at each gate
        edge1a = recorder.register_edge(
            run_id=run.run_id,
            from_node_id=gate1_node.node_id,
            to_node_id=gate2_node.node_id,
            label="left",
            mode=RoutingMode.COPY,
        )
        edge1b = recorder.register_edge(
            run_id=run.run_id,
            from_node_id=gate1_node.node_id,
            to_node_id=gate2_node.node_id,
            label="right",
            mode=RoutingMode.COPY,
        )
        edge2a = recorder.register_edge(
            run_id=run.run_id,
            from_node_id=gate2_node.node_id,
            to_node_id=transform_node.node_id,
            label="left",
            mode=RoutingMode.COPY,
        )
        edge2b = recorder.register_edge(
            run_id=run.run_id,
            from_node_id=gate2_node.node_id,
            to_node_id=transform_node.node_id,
            label="right",
            mode=RoutingMode.COPY,
        )

        class MarkerTransform(BaseTransform):
            name = "marker"
            input_schema = _TestSchema
            output_schema = _TestSchema

            def __init__(self, node_id: str) -> None:
                super().__init__({"schema": {"fields": "dynamic"}})
                self.node_id = node_id

            def process(self, row: dict[str, Any], ctx: PluginContext) -> TransformResult:
                # Note: .get() is allowed here - this is row data (their data, Tier 2)
                return TransformResult.success({**row, "count": row.get("count", 0) + 1}, success_reason={"action": "count"})

        # Config-driven fork gates
        gate1_config = GateSettings(
            name="fork_gate_1",
            condition="True",
            routes={"true": "fork", "false": "continue"},
            fork_to=["left", "right"],
        )
        gate2_config = GateSettings(
            name="fork_gate_2",
            condition="True",
            routes={"true": "fork", "false": "continue"},
            fork_to=["left", "right"],
        )

        transform = MarkerTransform(transform_node.node_id)

        processor = RowProcessor(
            recorder=recorder,
            span_factory=SpanFactory(),
            run_id=run.run_id,
            source_node_id=NodeID(source_node.node_id),
            edge_map={
                (NodeID(gate1_node.node_id), "left"): edge1a.edge_id,
                (NodeID(gate1_node.node_id), "right"): edge1b.edge_id,
                (NodeID(gate2_node.node_id), "left"): edge2a.edge_id,
                (NodeID(gate2_node.node_id), "right"): edge2b.edge_id,
            },
            config_gates=[gate1_config, gate2_config],
            config_gate_id_map={
                GateName("fork_gate_1"): NodeID(gate1_node.node_id),
                GateName("fork_gate_2"): NodeID(gate2_node.node_id),
            },
        )

        ctx = PluginContext(run_id=run.run_id, config={})
        results = processor.process_row(
            row_index=0,
            row_data={"value": 42},
            transforms=[transform],
            ctx=ctx,
        )

        # Expected: 1 parent FORKED + 2 children FORKED + 4 grandchildren COMPLETED = 7
        assert len(results) == 7

        forked_count = sum(1 for r in results if r.outcome == RowOutcome.FORKED)
        completed_count = sum(1 for r in results if r.outcome == RowOutcome.COMPLETED)

        assert forked_count == 3  # Parent + 2 first-level children
        assert completed_count == 4  # 4 grandchildren

        # All tokens should have count=1 (transform runs first, data inherited through forks)
        for result in results:
            # .get() allowed on row data (their data, Tier 2)
            assert result.final_data.get("count") == 1
