# tests/engine/test_integration.py
"""Integration tests for engine module.

These tests verify:
1. All components can be imported from elspeth.engine
2. Full pipeline execution with audit trail verification
3. "Audit spine" tests proving every token reaches terminal state
4. "No silent audit loss" tests proving errors raise, not skip

Transform plugins inherit from BaseTransform. Gates use config-driven
GateSettings which are processed by the engine's ExpressionParser.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import pytest

from elspeth.contracts import RoutingMode, SourceRow
from elspeth.core.config import GateSettings
from elspeth.plugins.base import BaseTransform
from tests.conftest import (
    _TestSchema,
    _TestSinkBase,
    _TestSourceBase,
    as_source,
    as_transform,
)

if TYPE_CHECKING:
    from elspeth.contracts.results import ArtifactDescriptor, TransformResult
    from elspeth.core.dag import ExecutionGraph
    from elspeth.engine.orchestrator import PipelineConfig


def _build_test_graph(config: PipelineConfig) -> ExecutionGraph:
    """Build a simple graph for testing (temporary until from_config is wired).

    Creates a linear graph matching the PipelineConfig structure:
    source -> transforms... -> config_gates... -> sinks

    Config-driven gates (GateSettings in config.gates) can route to sinks.
    Route labels use sink names for simplicity in tests.
    """
    from elspeth.core.dag import ExecutionGraph

    graph = ExecutionGraph()

    # Add source
    graph.add_node("source", node_type="source", plugin_name=config.source.name)

    # Add transforms and populate transform_id_map
    transform_ids: dict[int, str] = {}
    prev = "source"
    for i, t in enumerate(config.transforms):
        node_id = f"transform_{i}"
        transform_ids[i] = node_id
        graph.add_node(
            node_id,
            node_type="transform",
            plugin_name=t.name,
        )
        graph.add_edge(prev, node_id, label="continue", mode=RoutingMode.MOVE)
        prev = node_id

    # Add sinks first (needed for config gate edges)
    sink_ids: dict[str, str] = {}
    for sink_name, sink in config.sinks.items():
        node_id = f"sink_{sink_name}"
        sink_ids[sink_name] = node_id
        graph.add_node(node_id, node_type="sink", plugin_name=sink.name)

    # Add config gates
    config_gate_ids: dict[str, str] = {}
    route_resolution_map: dict[tuple[str, str], str] = {}

    for gate_config in config.gates:
        node_id = f"config_gate_{gate_config.name}"
        config_gate_ids[gate_config.name] = node_id
        graph.add_node(
            node_id,
            node_type="gate",
            plugin_name=f"config_gate:{gate_config.name}",
            config={
                "condition": gate_config.condition,
                "routes": dict(gate_config.routes),
            },
        )
        graph.add_edge(prev, node_id, label="continue", mode=RoutingMode.MOVE)

        # Add route edges and resolution map
        for route_label, target in gate_config.routes.items():
            route_resolution_map[(node_id, route_label)] = target
            if target not in ("continue", "fork") and target in sink_ids:
                graph.add_edge(node_id, sink_ids[target], label=route_label, mode=RoutingMode.MOVE)

        # Handle fork paths
        if gate_config.fork_to:
            for path in gate_config.fork_to:
                route_resolution_map[(node_id, path)] = "fork"
                # Fork paths need edges to next step (or sink if no next step)
                # For fork tests, we add edges to a pseudo-node or reuse sink

        prev = node_id

    # Edge from last node to output sink
    output_sink = "default" if "default" in sink_ids else next(iter(sink_ids))
    graph.add_edge(prev, sink_ids[output_sink], label="continue", mode=RoutingMode.MOVE)

    # Populate internal ID maps
    graph._sink_id_map = sink_ids
    graph._transform_id_map = transform_ids
    graph._config_gate_id_map = config_gate_ids
    graph._route_resolution_map = route_resolution_map
    graph._output_sink = output_sink

    return graph


class TestEngineIntegration:
    """Test engine module integration and imports."""

    def test_can_import_all_components(self) -> None:
        """All public components should be importable from elspeth.engine."""
        from elspeth.engine import (
            AggregationExecutor,
            GateExecutor,
            MaxRetriesExceeded,
            MissingEdgeError,
            Orchestrator,
            PipelineConfig,
            RetryConfig,
            RetryManager,
            RowProcessor,
            RowResult,
            RunResult,
            SinkExecutor,
            SpanFactory,
            TokenInfo,
            TokenManager,
            TransformExecutor,
        )

        # Verify they are the actual classes, not None
        assert Orchestrator is not None
        assert PipelineConfig is not None
        assert RunResult is not None
        assert RowProcessor is not None
        assert RowResult is not None
        assert TokenManager is not None
        assert TokenInfo is not None
        assert TransformExecutor is not None
        assert GateExecutor is not None
        assert AggregationExecutor is not None
        assert SinkExecutor is not None
        assert MissingEdgeError is not None
        assert RetryManager is not None
        assert RetryConfig is not None
        assert MaxRetriesExceeded is not None
        assert SpanFactory is not None

    def test_full_pipeline_with_audit(self) -> None:
        """Full pipeline execution with audit trail verification."""
        from elspeth.contracts import PluginSchema
        from elspeth.core.landscape import LandscapeDB, LandscapeRecorder
        from elspeth.engine import Orchestrator, PipelineConfig
        from elspeth.engine.artifacts import ArtifactDescriptor
        from elspeth.plugins.results import TransformResult

        db = LandscapeDB.in_memory()

        class ValueSchema(PluginSchema):
            value: int

        class OutputSchema(PluginSchema):
            value: int
            processed: bool

        class ListSource(_TestSourceBase):
            name = "test_source"
            output_schema = ValueSchema

            def __init__(self, data: list[dict[str, Any]]) -> None:
                self._data = data

            def on_start(self, ctx: Any) -> None:
                pass

            def load(self, ctx: Any) -> Any:
                for _row in self._data:
                    yield SourceRow.valid(_row)

            def close(self) -> None:
                pass

        class MarkProcessedTransform(BaseTransform):
            name = "mark_processed"
            input_schema = ValueSchema
            output_schema = OutputSchema

            def __init__(self) -> None:
                super().__init__({})

            def process(self, row: Any, ctx: Any) -> TransformResult:
                return TransformResult.success(
                    {
                        "value": row["value"],
                        "processed": True,
                    }
                )

        class CollectSink(_TestSinkBase):
            name = "output_sink"

            def __init__(self) -> None:
                self.results: list[dict[str, Any]] = []

            def on_start(self, ctx: Any) -> None:
                pass

            def on_complete(self, ctx: Any) -> None:
                pass

            def write(self, rows: Any, ctx: Any) -> ArtifactDescriptor:
                self.results.extend(rows)
                return ArtifactDescriptor.for_file(path="memory://output", size_bytes=100, content_hash="abc123")

            def close(self) -> None:
                pass

        # Run pipeline
        source = ListSource([{"value": 1}, {"value": 2}, {"value": 3}])
        transform = MarkProcessedTransform()
        sink = CollectSink()

        config = PipelineConfig(
            source=as_source(source),
            transforms=[transform],
            sinks={"default": sink},
        )

        orchestrator = Orchestrator(db)
        result = orchestrator.run(config, graph=_build_test_graph(config))

        # Verify run result
        assert result.status == "completed"
        assert result.rows_processed == 3
        assert result.rows_succeeded == 3

        # Verify sink received all rows
        assert len(sink.results) == 3
        assert all(r["processed"] for r in sink.results)

        # Verify audit trail
        from elspeth.contracts import RunStatus

        recorder = LandscapeRecorder(db)
        run = recorder.get_run(result.run_id)
        assert run is not None
        assert run.status == RunStatus.COMPLETED

        # Verify nodes registered
        nodes = recorder.get_nodes(result.run_id)
        assert len(nodes) == 3  # source, transform, sink

        # Verify rows recorded
        rows = recorder.get_rows(result.run_id)
        assert len(rows) == 3

        # Verify artifacts
        artifacts = recorder.get_artifacts(result.run_id)
        assert len(artifacts) == 1
        assert artifacts[0].content_hash == "abc123"

    def test_audit_spine_intact(self) -> None:
        """THE audit spine test: proves chassis doesn't wobble.

        For every row:
        - At least one token exists
        - Every token has node_states at transform AND sink
        - All node_states are "completed"
        - Artifacts are recorded for sinks
        """
        from elspeth.contracts import NodeType, PluginSchema
        from elspeth.core.landscape import LandscapeDB, LandscapeRecorder
        from elspeth.engine import Orchestrator, PipelineConfig
        from elspeth.engine.artifacts import ArtifactDescriptor
        from elspeth.plugins.results import TransformResult

        db = LandscapeDB.in_memory()

        class NumberSchema(PluginSchema):
            n: int

        class ListSource(_TestSourceBase):
            name = "numbers"
            output_schema = NumberSchema

            def __init__(self, data: list[dict[str, Any]]) -> None:
                self._data = data

            def on_start(self, ctx: Any) -> None:
                pass

            def load(self, ctx: Any) -> Any:
                for _row in self._data:
                    yield SourceRow.valid(_row)

            def close(self) -> None:
                pass

        class DoubleTransform(BaseTransform):
            name = "double"
            input_schema = NumberSchema
            output_schema = NumberSchema

            def __init__(self) -> None:
                super().__init__({})

            def process(self, row: Any, ctx: Any) -> TransformResult:
                return TransformResult.success({"n": row["n"] * 2})

        class AddTenTransform(BaseTransform):
            name = "add_ten"
            input_schema = NumberSchema
            output_schema = NumberSchema

            def __init__(self) -> None:
                super().__init__({})

            def process(self, row: Any, ctx: Any) -> TransformResult:
                return TransformResult.success({"n": row["n"] + 10})

        class CollectSink(_TestSinkBase):
            name = "collector"

            def __init__(self) -> None:
                self.results: list[dict[str, Any]] = []

            def on_start(self, ctx: Any) -> None:
                pass

            def on_complete(self, ctx: Any) -> None:
                pass

            def write(self, rows: Any, ctx: Any) -> ArtifactDescriptor:
                self.results.extend(rows)
                return ArtifactDescriptor.for_file(path="memory://out", size_bytes=len(rows), content_hash="hash")

            def close(self) -> None:
                pass

        # Pipeline with multiple transforms
        source = ListSource([{"n": 1}, {"n": 2}, {"n": 3}, {"n": 4}, {"n": 5}])
        t1 = DoubleTransform()
        t2 = AddTenTransform()
        sink = CollectSink()

        config = PipelineConfig(
            source=as_source(source),
            transforms=[t1, t2],
            sinks={"default": sink},
        )

        orchestrator = Orchestrator(db)
        result = orchestrator.run(config, graph=_build_test_graph(config))

        assert result.status == "completed"
        assert result.rows_processed == 5

        # Now verify the audit spine
        recorder = LandscapeRecorder(db)

        # Get all nodes to identify transforms and sinks
        nodes = recorder.get_nodes(result.run_id)
        transform_node_ids = {n.node_id for n in nodes if n.node_type in (NodeType.TRANSFORM.value, "transform")}
        sink_node_ids = {n.node_id for n in nodes if n.node_type in (NodeType.SINK.value, "sink")}

        # Get all rows
        rows = recorder.get_rows(result.run_id)
        assert len(rows) == 5, "All source rows must be recorded"

        for row in rows:
            # Every row must have at least one token
            tokens = recorder.get_tokens(row.row_id)
            assert len(tokens) >= 1, f"Row {row.row_id} has no tokens - audit spine broken"

            for token in tokens:
                # Every token must have node_states
                states = recorder.get_node_states_for_token(token.token_id)
                assert len(states) > 0, f"Token {token.token_id} has no node_states - audit spine broken"

                # Verify token has states at BOTH transforms
                state_node_ids = {s.node_id for s in states}
                for transform_id in transform_node_ids:
                    assert transform_id in state_node_ids, f"Token {token.token_id} missing state at transform {transform_id}"

                # Verify token has state at sink
                sink_states = [s for s in states if s.node_id in sink_node_ids]
                assert len(sink_states) >= 1, f"Token {token.token_id} never reached a sink - audit spine broken"

                # All states must be completed
                for state in states:
                    assert state.status == "completed", f"Token {token.token_id} has non-completed state: {state.status}"

        # Verify artifacts exist
        artifacts = recorder.get_artifacts(result.run_id)
        assert len(artifacts) >= 1, "No artifacts recorded - audit spine broken"

    def test_audit_spine_with_routing(self) -> None:
        """Audit spine test with gate routing.

        Verifies:
        - Routing events exist for routed tokens
        - Routed tokens reach correct sink
        - All tokens still have complete audit trail
        """
        from elspeth.contracts import NodeType, PluginSchema
        from elspeth.core.landscape import LandscapeDB, LandscapeRecorder
        from elspeth.engine import Orchestrator, PipelineConfig
        from elspeth.engine.artifacts import ArtifactDescriptor

        db = LandscapeDB.in_memory()

        class NumberSchema(PluginSchema):
            value: int

        class ListSource(_TestSourceBase):
            name = "numbers"
            output_schema = NumberSchema

            def __init__(self, data: list[dict[str, Any]]) -> None:
                self._data = data

            def on_start(self, ctx: Any) -> None:
                pass

            def load(self, ctx: Any) -> Any:
                for _row in self._data:
                    yield SourceRow.valid(_row)

            def close(self) -> None:
                pass

        class CollectSink(_TestSinkBase):
            name: str

            def __init__(self, sink_name: str):
                self.name = sink_name
                self.results: list[dict[str, Any]] = []

            def on_start(self, ctx: Any) -> None:
                pass

            def on_complete(self, ctx: Any) -> None:
                pass

            def write(self, rows: Any, ctx: Any) -> ArtifactDescriptor:
                self.results.extend(rows)
                return ArtifactDescriptor.for_file(
                    path=f"memory://{self.name}",
                    size_bytes=len(rows),
                    content_hash=f"hash_{self.name}",
                )

            def close(self) -> None:
                pass

        # Config-driven gate: routes even numbers to "even" sink, odd continue
        even_odd_gate = GateSettings(
            name="even_odd_gate",
            condition="row['value'] % 2 == 0",
            routes={"true": "even", "false": "continue"},
        )

        # Pipeline with routing gate
        source = ListSource([{"value": 1}, {"value": 2}, {"value": 3}, {"value": 4}])
        default_sink = CollectSink("default_sink")
        even_sink = CollectSink("even_sink")

        config = PipelineConfig(
            source=as_source(source),
            transforms=[],
            sinks={"default": default_sink, "even": even_sink},
            gates=[even_odd_gate],
        )

        orchestrator = Orchestrator(db)
        result = orchestrator.run(config, graph=_build_test_graph(config))

        assert result.status == "completed"
        assert result.rows_processed == 4

        # Verify routing: 2 odd to default, 2 even to even sink
        assert len(default_sink.results) == 2  # 1, 3
        assert len(even_sink.results) == 2  # 2, 4

        # Verify audit spine with routing
        recorder = LandscapeRecorder(db)

        # Get gate node
        nodes = recorder.get_nodes(result.run_id)
        gate_nodes = [n for n in nodes if n.node_type in (NodeType.GATE.value, "gate")]
        assert len(gate_nodes) == 1
        gate_node = gate_nodes[0]

        sink_node_ids = {n.node_id for n in nodes if n.node_type in (NodeType.SINK.value, "sink")}

        # Check every row/token
        rows = recorder.get_rows(result.run_id)
        routed_count = 0

        for row in rows:
            tokens = recorder.get_tokens(row.row_id)
            assert len(tokens) >= 1

            for token in tokens:
                states = recorder.get_node_states_for_token(token.token_id)
                assert len(states) > 0

                # Find gate state and check for routing event
                gate_states = [s for s in states if s.node_id == gate_node.node_id]
                assert len(gate_states) == 1

                # Check routing events
                routing_events = recorder.get_routing_events(gate_states[0].state_id)
                if routing_events:
                    routed_count += 1

                # Token must reach a sink
                sink_states = [s for s in states if s.node_id in sink_node_ids]
                assert len(sink_states) >= 1, f"Token {token.token_id} never reached sink"

        # AUD-002: All 4 tokens now have routing events recorded
        # (2 route to even_sink, 2 continue to default_sink)
        assert routed_count == 4


class TestNoSilentAuditLoss:
    """Tests that ensure audit errors raise, never skip silently."""

    def test_missing_edge_raises_not_skips(self) -> None:
        """Critical: RouteValidationError must raise, not silently count.

        This test ensures that when a config-driven gate routes to a sink that
        doesn't exist, the error is raised immediately at pipeline initialization
        (fail-fast) rather than being silently counted as a failure.

        Note: Config-driven gates are validated at startup via RouteValidationError,
        which is better than MissingEdgeError at runtime because it catches config
        errors before any rows are processed.
        """
        from elspeth.contracts import PluginSchema
        from elspeth.core.landscape import LandscapeDB
        from elspeth.engine import Orchestrator, PipelineConfig
        from elspeth.engine.artifacts import ArtifactDescriptor
        from elspeth.engine.orchestrator import RouteValidationError

        db = LandscapeDB.in_memory()

        class RowSchema(PluginSchema):
            value: int

        class ListSource(_TestSourceBase):
            name = "source"
            output_schema = RowSchema

            def __init__(self, data: list[dict[str, Any]]) -> None:
                self._data = data

            def on_start(self, ctx: Any) -> None:
                pass

            def load(self, ctx: Any) -> Any:
                for _row in self._data:
                    yield SourceRow.valid(_row)

            def close(self) -> None:
                pass

        class CollectSink(_TestSinkBase):
            name = "default_sink"

            def __init__(self) -> None:
                self.results: list[dict[str, Any]] = []

            def on_start(self, ctx: Any) -> None:
                pass

            def on_complete(self, ctx: Any) -> None:
                pass

            def write(self, rows: Any, ctx: Any) -> ArtifactDescriptor:
                self.results.extend(rows)
                return ArtifactDescriptor.for_file(path="memory", size_bytes=0, content_hash="")

            def close(self) -> None:
                pass

        # Config-driven gate that always routes to "phantom" (nonexistent sink)
        misrouting_gate = GateSettings(
            name="misrouting_gate",
            condition="True",  # Always routes
            routes={
                "true": "phantom",
                "false": "continue",
            },  # Route to nonexistent sink
        )

        source = ListSource([{"value": 42}])
        sink = CollectSink()

        config = PipelineConfig(
            source=as_source(source),
            transforms=[],
            sinks={"default": sink},  # Note: "phantom" is NOT configured
            gates=[misrouting_gate],
        )

        orchestrator = Orchestrator(db)

        # This MUST raise RouteValidationError at startup, not silently fail
        with pytest.raises(RouteValidationError) as exc_info:
            orchestrator.run(config, graph=_build_test_graph(config))

        # Verify error message includes the missing sink name
        assert "phantom" in str(exc_info.value)

    def test_missing_edge_error_is_not_catchable_silently(self) -> None:
        """MissingEdgeError inherits from Exception, not a special base.

        This ensures it cannot be silently swallowed by overly broad
        exception handlers without explicit intent.
        """
        from elspeth.engine import MissingEdgeError

        # MissingEdgeError should inherit from Exception
        assert issubclass(MissingEdgeError, Exception)

        # But NOT from a special "audit" base that could be caught separately
        # If we had an AuditError base class, we'd test that here
        # For now, just verify it's a plain Exception subclass

        # Create an instance and verify attributes
        error = MissingEdgeError(node_id="gate_1", label="nonexistent")
        assert error.node_id == "gate_1"
        assert error.label == "nonexistent"
        assert "gate_1" in str(error)
        assert "nonexistent" in str(error)
        assert "Audit trail would be incomplete" in str(error)

    def test_transform_exception_propagates(self) -> None:
        """Transform exceptions must propagate, not be silently caught."""
        from elspeth.contracts import PluginSchema
        from elspeth.core.landscape import LandscapeDB, LandscapeRecorder
        from elspeth.engine import Orchestrator, PipelineConfig
        from elspeth.engine.artifacts import ArtifactDescriptor

        db = LandscapeDB.in_memory()

        class ValueSchema(PluginSchema):
            value: int

        class ListSource(_TestSourceBase):
            name = "source"
            output_schema = ValueSchema

            def __init__(self, data: list[dict[str, Any]]) -> None:
                self._data = data

            def on_start(self, ctx: Any) -> None:
                pass

            def load(self, ctx: Any) -> Any:
                for _row in self._data:
                    yield SourceRow.valid(_row)

            def close(self) -> None:
                pass

        class ExplodingTransform(BaseTransform):
            name = "exploder"
            input_schema = ValueSchema
            output_schema = ValueSchema

            def __init__(self) -> None:
                super().__init__({})

            def process(self, row: Any, ctx: Any) -> TransformResult:
                raise RuntimeError("Intentional explosion")

        class CollectSink(_TestSinkBase):
            name = "sink"

            def __init__(self) -> None:
                self.results: list[dict[str, Any]] = []

            def on_start(self, ctx: Any) -> None:
                pass

            def on_complete(self, ctx: Any) -> None:
                pass

            def write(self, rows: Any, ctx: Any) -> ArtifactDescriptor:
                self.results.extend(rows)
                return ArtifactDescriptor.for_file(path="memory", size_bytes=0, content_hash="")

            def close(self) -> None:
                pass

        source = ListSource([{"value": 1}])
        transform = ExplodingTransform()
        sink = CollectSink()

        config = PipelineConfig(
            source=as_source(source),
            transforms=[transform],
            sinks={"default": sink},
        )

        orchestrator = Orchestrator(db)

        # Exception must propagate
        with pytest.raises(RuntimeError, match="Intentional explosion"):
            orchestrator.run(config, graph=_build_test_graph(config))

        # Run must be marked as failed in audit trail
        from elspeth.contracts import RunStatus

        recorder = LandscapeRecorder(db)
        runs = recorder.list_runs()
        assert len(runs) == 1
        assert runs[0].status == RunStatus.FAILED

    def test_sink_exception_propagates(self) -> None:
        """Sink exceptions must propagate, not be silently caught."""
        from elspeth.contracts import PluginSchema
        from elspeth.core.landscape import LandscapeDB, LandscapeRecorder
        from elspeth.engine import Orchestrator, PipelineConfig
        from elspeth.plugins.results import TransformResult

        db = LandscapeDB.in_memory()

        class ValueSchema(PluginSchema):
            value: int

        class ListSource(_TestSourceBase):
            name = "source"
            output_schema = ValueSchema

            def __init__(self, data: list[dict[str, Any]]) -> None:
                self._data = data

            def on_start(self, ctx: Any) -> None:
                pass

            def load(self, ctx: Any) -> Any:
                for _row in self._data:
                    yield SourceRow.valid(_row)

            def close(self) -> None:
                pass

        class IdentityTransform(BaseTransform):
            name = "identity"
            input_schema = ValueSchema
            output_schema = ValueSchema

            def __init__(self) -> None:
                super().__init__({})

            def process(self, row: Any, ctx: Any) -> TransformResult:
                return TransformResult.success(row)

        class ExplodingSink(_TestSinkBase):
            name = "exploding_sink"

            def on_start(self, ctx: Any) -> None:
                pass

            def on_complete(self, ctx: Any) -> None:
                pass

            def write(self, rows: Any, ctx: Any) -> ArtifactDescriptor:
                raise OSError("Sink explosion")

            def close(self) -> None:
                pass

        source = ListSource([{"value": 1}])
        transform = IdentityTransform()
        sink = ExplodingSink()

        config = PipelineConfig(
            source=as_source(source),
            transforms=[transform],
            sinks={"default": sink},
        )

        orchestrator = Orchestrator(db)

        # Exception must propagate
        with pytest.raises(OSError, match="Sink explosion"):
            orchestrator.run(config, graph=_build_test_graph(config))

        # Run must be marked as failed in audit trail
        from elspeth.contracts import RunStatus

        recorder = LandscapeRecorder(db)
        runs = recorder.list_runs()
        assert len(runs) == 1
        assert runs[0].status == RunStatus.FAILED


class TestAuditTrailCompleteness:
    """Tests verifying complete audit trail for complex scenarios."""

    def test_empty_source_still_records_run(self) -> None:
        """Even with no rows, run must be recorded in audit trail."""
        from elspeth.contracts import PluginSchema
        from elspeth.core.landscape import LandscapeDB, LandscapeRecorder
        from elspeth.engine import Orchestrator, PipelineConfig
        from elspeth.engine.artifacts import ArtifactDescriptor
        from elspeth.plugins.results import TransformResult

        db = LandscapeDB.in_memory()

        class ValueSchema(PluginSchema):
            value: int

        class EmptySource(_TestSourceBase):
            name = "empty"
            output_schema = ValueSchema

            def on_start(self, ctx: Any) -> None:
                pass

            def load(self, ctx: Any) -> Any:
                return iter([])

            def close(self) -> None:
                pass

        class IdentityTransform(BaseTransform):
            name = "identity"
            input_schema = ValueSchema
            output_schema = ValueSchema

            def __init__(self) -> None:
                super().__init__({})

            def process(self, row: Any, ctx: Any) -> TransformResult:
                return TransformResult.success(row)

        class CollectSink(_TestSinkBase):
            name = "sink"

            def __init__(self) -> None:
                self.results: list[dict[str, Any]] = []

            def on_start(self, ctx: Any) -> None:
                pass

            def on_complete(self, ctx: Any) -> None:
                pass

            def write(self, rows: Any, ctx: Any) -> ArtifactDescriptor:
                self.results.extend(rows)
                return ArtifactDescriptor.for_file(path="memory", size_bytes=0, content_hash="")

            def close(self) -> None:
                pass

        source = EmptySource()
        transform = IdentityTransform()
        sink = CollectSink()

        config = PipelineConfig(
            source=as_source(source),
            transforms=[transform],
            sinks={"default": sink},
        )

        orchestrator = Orchestrator(db)
        result = orchestrator.run(config, graph=_build_test_graph(config))

        assert result.status == "completed"
        assert result.rows_processed == 0

        # Even with no rows, run and nodes must be recorded
        recorder = LandscapeRecorder(db)
        run = recorder.get_run(result.run_id)
        assert run is not None
        assert run.status == "completed"

        nodes = recorder.get_nodes(result.run_id)
        assert len(nodes) == 3  # source, transform, sink

    def test_multiple_sinks_all_record_artifacts(self) -> None:
        """When multiple sinks receive data, all must record artifacts."""
        from elspeth.contracts import PluginSchema
        from elspeth.core.landscape import LandscapeDB, LandscapeRecorder
        from elspeth.engine import Orchestrator, PipelineConfig
        from elspeth.engine.artifacts import ArtifactDescriptor

        db = LandscapeDB.in_memory()

        class ValueSchema(PluginSchema):
            value: int

        class ListSource(_TestSourceBase):
            name = "source"
            output_schema = ValueSchema

            def __init__(self, data: list[dict[str, Any]]) -> None:
                self._data = data

            def on_start(self, ctx: Any) -> None:
                pass

            def load(self, ctx: Any) -> Any:
                for _row in self._data:
                    yield SourceRow.valid(_row)

            def close(self) -> None:
                pass

        class CollectSink(_TestSinkBase):
            def __init__(self, sink_name: str):
                self.name = sink_name
                self.results: list[dict[str, Any]] = []

            def on_start(self, ctx: Any) -> None:
                pass

            def on_complete(self, ctx: Any) -> None:
                pass

            def write(self, rows: Any, ctx: Any) -> ArtifactDescriptor:
                self.results.extend(rows)
                return ArtifactDescriptor.for_file(
                    path=f"memory://{self.name}",
                    size_bytes=len(rows) * 10,
                    content_hash=f"{self.name}_hash",
                )

            def close(self) -> None:
                pass

        # Config-driven gate: routes values > 50 to "high" sink, otherwise continue
        split_gate = GateSettings(
            name="split_gate",
            condition="row['value'] > 50",
            routes={"true": "high", "false": "continue"},
        )

        source = ListSource([{"value": 10}, {"value": 60}, {"value": 30}, {"value": 90}])
        default_sink = CollectSink("default_output")
        high_sink = CollectSink("high_output")

        config = PipelineConfig(
            source=as_source(source),
            transforms=[],
            sinks={"default": default_sink, "high": high_sink},
            gates=[split_gate],
        )

        orchestrator = Orchestrator(db)
        result = orchestrator.run(config, graph=_build_test_graph(config))

        assert result.status == "completed"

        # Both sinks received data
        assert len(default_sink.results) == 2  # 10, 30
        assert len(high_sink.results) == 2  # 60, 90

        # Both sinks have artifacts
        recorder = LandscapeRecorder(db)
        artifacts = recorder.get_artifacts(result.run_id)
        assert len(artifacts) == 2

        artifact_hashes = {a.content_hash for a in artifacts}
        assert "default_output_hash" in artifact_hashes
        assert "high_output_hash" in artifact_hashes


class TestForkIntegration:
    """Integration tests for fork execution through full pipeline.

    Note on DiGraph limitation: NetworkX DiGraph doesn't support multiple edges
    between the same node pair. For fork operations where multiple children go
    to the same destination, we manually register edges with the LandscapeRecorder
    rather than relying on graph-based edge registration.
    """

    def test_full_pipeline_with_fork_writes_all_children_to_sink(self) -> None:
        """Full pipeline should write all fork children to sink.

        This test verifies end-to-end fork behavior:
        - 2 rows from source
        - Each row forks into 2 children via config-driven ForkGate
        - All 4 children (2 rows x 2 forks) continue processing and reach sink

        Fork behavior: Fork creates child tokens that continue processing through
        the remaining transforms. If no more transforms exist, children reach
        COMPLETED state and go to the output sink.

        Implementation note: Uses RowProcessor directly with manually registered
        edges to work around DiGraph's single-edge limitation between node pairs.
        """
        from elspeth.contracts import RowOutcome
        from elspeth.contracts.schema import SchemaConfig
        from elspeth.core.landscape import LandscapeDB, LandscapeRecorder
        from elspeth.engine.processor import RowProcessor
        from elspeth.engine.spans import SpanFactory
        from elspeth.plugins.context import PluginContext

        db = LandscapeDB.in_memory()
        recorder = LandscapeRecorder(db)

        # Start a run
        run = recorder.begin_run(config={}, canonical_version="v1")
        run_id = run.run_id

        # Register nodes
        source_node = recorder.register_node(
            run_id=run_id,
            plugin_name="list_source",
            node_type="source",
            plugin_version="1.0",
            config={},
            schema_config=SchemaConfig.from_dict({"fields": "dynamic"}),
        )

        gate_node = recorder.register_node(
            run_id=run_id,
            plugin_name="config_gate:fork_gate",
            node_type="gate",
            plugin_version="1.0",
            config={},
            schema_config=SchemaConfig.from_dict({"fields": "dynamic"}),
        )

        # Register path nodes for fork destinations
        path_a_node = recorder.register_node(
            run_id=run_id,
            plugin_name="path_a",
            node_type="transform",
            plugin_version="1.0",
            config={},
            schema_config=SchemaConfig.from_dict({"fields": "dynamic"}),
        )
        path_b_node = recorder.register_node(
            run_id=run_id,
            plugin_name="path_b",
            node_type="transform",
            plugin_version="1.0",
            config={},
            schema_config=SchemaConfig.from_dict({"fields": "dynamic"}),
        )

        # Register sink node (not used in processor but required for complete graph)
        recorder.register_node(
            run_id=run_id,
            plugin_name="collect_sink",
            node_type="sink",
            plugin_version="1.0",
            config={},
            schema_config=SchemaConfig.from_dict({"fields": "dynamic"}),
        )

        # Register edges (including fork paths to distinct intermediate nodes)
        edge_a = recorder.register_edge(
            run_id=run_id,
            from_node_id=gate_node.node_id,
            to_node_id=path_a_node.node_id,
            label="path_a",
            mode=RoutingMode.COPY,
        )
        edge_b = recorder.register_edge(
            run_id=run_id,
            from_node_id=gate_node.node_id,
            to_node_id=path_b_node.node_id,
            label="path_b",
            mode=RoutingMode.COPY,
        )

        # Build edge_map for GateExecutor
        edge_map = {
            (gate_node.node_id, "path_a"): edge_a.edge_id,
            (gate_node.node_id, "path_b"): edge_b.edge_id,
        }

        # Route resolution map: fork paths resolve to "fork"
        route_resolution_map: dict[tuple[str, str], str] = {
            (gate_node.node_id, "path_a"): "fork",
            (gate_node.node_id, "path_b"): "fork",
        }

        # Config-driven fork gate: forks every row into two parallel paths
        fork_gate = GateSettings(
            name="fork_gate",
            condition="True",  # Always fork
            routes={"true": "fork", "false": "continue"},
            fork_to=["path_a", "path_b"],
        )

        # Create processor with config gate
        processor = RowProcessor(
            recorder=recorder,
            span_factory=SpanFactory(),
            run_id=run_id,
            source_node_id=source_node.node_id,
            edge_map=edge_map,
            route_resolution_map=route_resolution_map,
            config_gates=[fork_gate],
            config_gate_id_map={"fork_gate": gate_node.node_id},
        )

        # Create context
        ctx = PluginContext(run_id=run_id, config={})

        # Process 2 rows from source
        source_rows = [{"value": 1}, {"value": 2}]
        all_results = []
        for row_index, row_data in enumerate(source_rows):
            results = processor.process_row(
                row_index=row_index,
                row_data=row_data,
                transforms=[],  # No plugin transforms, only config gate
                ctx=ctx,
            )
            all_results.extend(results)

        # Count outcomes
        completed_count = sum(1 for r in all_results if r.outcome == RowOutcome.COMPLETED)
        forked_count = sum(1 for r in all_results if r.outcome == RowOutcome.FORKED)

        # Verify:
        # - 2 parent tokens with FORKED outcome
        # - 4 child tokens with COMPLETED outcome
        assert forked_count == 2, f"Expected 2 FORKED, got {forked_count}"
        assert completed_count == 4, f"Expected 4 COMPLETED, got {completed_count}"

        # Collect the COMPLETED tokens (these are what would go to sink)
        completed_tokens = [r.token for r in all_results if r.outcome == RowOutcome.COMPLETED]
        assert len(completed_tokens) == 4

        # Verify correct values: each source value appears twice (once per fork path)
        values = [t.row_data["value"] for t in completed_tokens]
        assert values.count(1) == 2, f"Expected value 1 to appear 2 times, got {values.count(1)}"
        assert values.count(2) == 2, f"Expected value 2 to appear 2 times, got {values.count(2)}"

        # Verify audit trail completeness
        rows = recorder.get_rows(run_id)
        assert len(rows) == 2, f"Expected 2 source rows, got {len(rows)}"

        # Verify tokens: 2 parent + 4 children = 6 total
        total_tokens = 0
        for row in rows:
            tokens = recorder.get_tokens(row.row_id)
            total_tokens += len(tokens)
        assert total_tokens == 6, f"Expected 6 tokens (2 parents + 4 children), got {total_tokens}"

        # Verify routing events were recorded for fork operations
        routing_event_count = 0
        for row in rows:
            tokens = recorder.get_tokens(row.row_id)
            for token in tokens:
                states = recorder.get_node_states_for_token(token.token_id)
                for state in states:
                    events = recorder.get_routing_events(state.state_id)
                    routing_event_count += len(events)

        # Each fork creates 2 routing events (one per path), 2 rows x 2 events = 4
        assert routing_event_count == 4, f"Expected 4 routing events, got {routing_event_count}"

        # Complete the run
        recorder.complete_run(run_id, status="completed")


class TestAggregationIntegrationDeleted:
    """Verify old aggregation integration tests are deleted.

    OLD: TestAggregationIntegration tested full pipeline with BaseAggregation
         plugins using accept()/flush() interface.
    NEW: Aggregation is engine-controlled via batch-aware transforms
         (is_batch_aware=True) using buffer_row()/execute_flush().

    Integration tests for new batch-aware transforms should be added to
    a new TestBatchAwareTransformIntegration class when ready.
    """

    def test_base_aggregation_deleted(self) -> None:
        """BaseAggregation should be deleted (aggregation is structural)."""
        import elspeth.plugins.base as base

        assert not hasattr(base, "BaseAggregation"), "BaseAggregation should be deleted - use is_batch_aware=True on BaseTransform"

    def test_aggregation_protocol_deleted(self) -> None:
        """AggregationProtocol should be deleted (aggregation is structural)."""
        import elspeth.plugins.protocols as protocols

        assert not hasattr(protocols, "AggregationProtocol"), "AggregationProtocol should be deleted - aggregation is structural"


class TestForkCoalescePipelineIntegration:
    """End-to-end fork -> coalesce -> sink tests.

    Verifies the complete pipeline flow:
    - Source emits rows
    - Fork gate splits rows to parallel branches
    - Each branch processes independently
    - Coalesce merges results
    - Sink receives merged data (not fork children separately)
    - Artifacts recorded with content hashes
    """

    def test_fork_coalesce_writes_merged_to_sink(self) -> None:
        """Complete pipeline: source -> fork -> process -> coalesce -> sink.

        Verifies:
        - Sink receives merged data (not 2 fork children separately)
        - Only 1 row written to sink per source row
        - Sink artifact has correct content hash
        """
        from elspeth.contracts import NodeType, TokenInfo
        from elspeth.contracts.schema import SchemaConfig
        from elspeth.core.config import CoalesceSettings
        from elspeth.core.landscape import LandscapeDB, LandscapeRecorder
        from elspeth.engine.artifacts import ArtifactDescriptor
        from elspeth.engine.coalesce_executor import CoalesceExecutor
        from elspeth.engine.executors import SinkExecutor
        from elspeth.engine.spans import SpanFactory
        from elspeth.engine.tokens import TokenManager
        from elspeth.plugins.context import PluginContext
        from elspeth.plugins.results import TransformResult

        db = LandscapeDB.in_memory()
        recorder = LandscapeRecorder(db)

        # Start a run
        run = recorder.begin_run(config={}, canonical_version="v1")
        run_id = run.run_id

        DYNAMIC_SCHEMA = SchemaConfig.from_dict({"fields": "dynamic"})

        # Register nodes
        source_node = recorder.register_node(
            run_id=run_id,
            node_id="source_1",
            plugin_name="test_source",
            node_type=NodeType.SOURCE,
            plugin_version="1.0.0",
            config={},
            schema_config=DYNAMIC_SCHEMA,
        )

        gate_node = recorder.register_node(
            run_id=run_id,
            node_id="fork_gate",
            plugin_name="config_gate:fork_gate",
            node_type=NodeType.GATE,
            plugin_version="1.0.0",
            config={},
            schema_config=DYNAMIC_SCHEMA,
        )

        # Register branch transform nodes (simulate processing on each branch)
        sentiment_node = recorder.register_node(
            run_id=run_id,
            node_id="sentiment_transform",
            plugin_name="sentiment_analyzer",
            node_type=NodeType.TRANSFORM,
            plugin_version="1.0.0",
            config={},
            schema_config=DYNAMIC_SCHEMA,
        )

        entity_node = recorder.register_node(
            run_id=run_id,
            node_id="entity_transform",
            plugin_name="entity_extractor",
            node_type=NodeType.TRANSFORM,
            plugin_version="1.0.0",
            config={},
            schema_config=DYNAMIC_SCHEMA,
        )

        coalesce_node = recorder.register_node(
            run_id=run_id,
            node_id="merge_coalesce",
            plugin_name="merge_results",
            node_type=NodeType.COALESCE,
            plugin_version="1.0.0",
            config={},
            schema_config=DYNAMIC_SCHEMA,
        )

        sink_node = recorder.register_node(
            run_id=run_id,
            node_id="output_sink",
            plugin_name="test_sink",
            node_type=NodeType.SINK,
            plugin_version="1.0.0",
            config={},
            schema_config=DYNAMIC_SCHEMA,
        )

        # Register edges for fork paths
        edge_sentiment = recorder.register_edge(
            run_id=run_id,
            from_node_id=gate_node.node_id,
            to_node_id=sentiment_node.node_id,
            label="sentiment",
            mode=RoutingMode.COPY,
        )
        edge_entity = recorder.register_edge(
            run_id=run_id,
            from_node_id=gate_node.node_id,
            to_node_id=entity_node.node_id,
            label="entities",
            mode=RoutingMode.COPY,
        )

        # Build edge_map for GateExecutor
        edge_map = {
            (gate_node.node_id, "sentiment"): edge_sentiment.edge_id,
            (gate_node.node_id, "entities"): edge_entity.edge_id,
        }

        # Route resolution map: fork paths resolve to "fork"
        route_resolution_map: dict[tuple[str, str], str] = {
            (gate_node.node_id, "sentiment"): "fork",
            (gate_node.node_id, "entities"): "fork",
        }

        # Config-driven fork gate: forks every row into sentiment and entity branches
        fork_gate_config = GateSettings(
            name="fork_gate",
            condition="True",  # Always fork
            routes={"true": "fork", "false": "continue"},
            fork_to=["sentiment", "entities"],
        )

        class SentimentTransform(BaseTransform):
            """Simulates sentiment analysis."""

            name = "sentiment_analyzer"
            input_schema = _TestSchema
            output_schema = _TestSchema

            def __init__(self, node_id: str) -> None:
                super().__init__({})
                self.node_id = node_id

            def process(self, row: Any, ctx: Any) -> TransformResult:
                return TransformResult.success({"sentiment": "positive"})

        class EntityTransform(BaseTransform):
            """Simulates entity extraction."""

            name = "entity_extractor"
            input_schema = _TestSchema
            output_schema = _TestSchema

            def __init__(self, node_id: str) -> None:
                super().__init__({})
                self.node_id = node_id

            def process(self, row: Any, ctx: Any) -> TransformResult:
                return TransformResult.success({"entities": ["ACME"]})

        class CollectSink(_TestSinkBase):
            """Test sink that collects written rows."""

            name = "test_sink"

            def __init__(self, node_id: str) -> None:
                self.node_id = node_id
                self.rows_written: list[dict[str, Any]] = []
                self._artifact_counter = 0

            def on_start(self, ctx: Any) -> None:
                pass

            def on_complete(self, ctx: Any) -> None:
                pass

            def write(self, rows: Any, ctx: Any) -> ArtifactDescriptor:
                self.rows_written.extend(rows)
                self._artifact_counter += 1
                content_hash = f"hash_{self._artifact_counter}"
                return ArtifactDescriptor.for_file(
                    path=f"memory://output_{self._artifact_counter}",
                    size_bytes=len(str(rows)),
                    content_hash=content_hash,
                )

            def close(self) -> None:
                pass

        # Create components
        span_factory = SpanFactory()
        token_manager = TokenManager(recorder)

        # Configure coalesce
        coalesce_settings = CoalesceSettings(
            name="merge_results",
            branches=["sentiment", "entities"],
            policy="require_all",
            merge="union",
        )

        coalesce_executor = CoalesceExecutor(
            recorder=recorder,
            span_factory=span_factory,
            token_manager=token_manager,
            run_id=run_id,
        )
        coalesce_executor.register_coalesce(coalesce_settings, coalesce_node.node_id)

        # Create plugins
        sentiment = SentimentTransform(sentiment_node.node_id)
        entity = EntityTransform(entity_node.node_id)
        sink = CollectSink(sink_node.node_id)

        ctx = PluginContext(run_id=run_id, config={})

        # Process a single source row through the pipeline
        # The flow: source -> fork gate -> [sentiment branch, entity branch] -> coalesce
        source_row = {"text": "ACME reported great earnings"}

        # Step 1: Process through gate (fork)
        initial_token = token_manager.create_initial_token(
            run_id=run_id,
            source_node_id=source_node.node_id,
            row_index=0,
            row_data=source_row,
        )

        # Execute the config-driven fork gate
        from elspeth.engine.executors import GateExecutor

        gate_executor = GateExecutor(
            recorder=recorder,
            span_factory=span_factory,
            edge_map=edge_map,
            route_resolution_map=route_resolution_map,
        )
        gate_outcome = gate_executor.execute_config_gate(
            gate_config=fork_gate_config,
            node_id=gate_node.node_id,
            token=initial_token,
            ctx=ctx,
            step_in_pipeline=1,
            token_manager=token_manager,
        )

        # Verify fork created 2 children
        assert len(gate_outcome.child_tokens) == 2
        sentiment_token = next(t for t in gate_outcome.child_tokens if t.branch_name == "sentiment")
        entity_token = next(t for t in gate_outcome.child_tokens if t.branch_name == "entities")

        # Step 2: Process each branch through its transform
        from elspeth.engine.executors import TransformExecutor

        transform_executor = TransformExecutor(recorder, span_factory)

        # Process sentiment branch
        sentiment_result, sentiment_token_updated, _ = transform_executor.execute_transform(
            transform=as_transform(sentiment),
            token=sentiment_token,
            ctx=ctx,
            step_in_pipeline=2,
        )
        assert sentiment_result.status == "success"
        # Update token with transformed data while preserving branch_name
        assert sentiment_result.row is not None
        sentiment_token_processed = TokenInfo(
            row_id=sentiment_token_updated.row_id,
            token_id=sentiment_token_updated.token_id,
            row_data=sentiment_result.row,
            branch_name="sentiment",
        )

        # Process entity branch
        entity_result, entity_token_updated, _ = transform_executor.execute_transform(
            transform=as_transform(entity),
            token=entity_token,
            ctx=ctx,
            step_in_pipeline=2,
        )
        assert entity_result.status == "success"
        # Update token with transformed data while preserving branch_name
        assert entity_result.row is not None
        entity_token_processed = TokenInfo(
            row_id=entity_token_updated.row_id,
            token_id=entity_token_updated.token_id,
            row_data=entity_result.row,
            branch_name="entities",
        )

        # Step 3: Coalesce the branches
        outcome1 = coalesce_executor.accept(
            token=sentiment_token_processed,
            coalesce_name="merge_results",
            step_in_pipeline=3,
        )
        assert outcome1.held is True  # Waiting for other branch

        outcome2 = coalesce_executor.accept(
            token=entity_token_processed,
            coalesce_name="merge_results",
            step_in_pipeline=3,
        )
        assert outcome2.held is False  # All arrived, merged
        assert outcome2.merged_token is not None

        # Verify merged data contains both sentiment and entities
        merged_data = outcome2.merged_token.row_data
        assert "sentiment" in merged_data
        assert "entities" in merged_data
        assert merged_data["sentiment"] == "positive"
        assert merged_data["entities"] == ["ACME"]

        # Step 4: Write merged token to sink
        sink_executor = SinkExecutor(recorder, span_factory, run_id)
        artifact = sink_executor.write(
            sink=sink,
            tokens=[outcome2.merged_token],
            ctx=ctx,
            step_in_pipeline=4,
        )

        # CRITICAL VERIFICATION: Sink received 1 merged row, not 2 fork children
        assert len(sink.rows_written) == 1, (
            f"Expected 1 merged row, got {len(sink.rows_written)}. Fork children should be coalesced before sink."
        )

        # Verify the single row contains merged data
        written_row = sink.rows_written[0]
        assert written_row["sentiment"] == "positive"
        assert written_row["entities"] == ["ACME"]

        # Verify artifact recorded with content hash
        assert artifact is not None
        assert artifact.content_hash is not None
        assert artifact.content_hash.startswith("hash_")

        # Verify artifact in landscape
        artifacts = recorder.get_artifacts(run_id)
        assert len(artifacts) == 1
        assert artifacts[0].content_hash == artifact.content_hash

        # Complete the run
        recorder.complete_run(run_id, status="completed")

        # Verify audit trail completeness
        # We should have:
        # - 1 source row
        # - 3 tokens: 1 parent (forked) + 2 children (coalesced) -> 1 merged
        # Actually: 1 parent + 2 children + 1 merged = 4 tokens total
        rows = recorder.get_rows(run_id)
        assert len(rows) == 1

        all_tokens = recorder.get_tokens(rows[0].row_id)
        # Parent token, 2 fork children, 1 merged token = 4 total
        assert len(all_tokens) == 4, f"Expected 4 tokens, got {len(all_tokens)}"

    def test_multiple_rows_coalesce_correctly(self) -> None:
        """Multiple source rows each fork and coalesce independently.

        Verifies:
        - Each source row's fork children merge with correct siblings
        - No cross-contamination between rows
        - Sink receives correct number of merged rows
        """
        from elspeth.contracts import NodeType, TokenInfo
        from elspeth.contracts.schema import SchemaConfig
        from elspeth.core.config import CoalesceSettings
        from elspeth.core.landscape import LandscapeDB, LandscapeRecorder
        from elspeth.engine.artifacts import ArtifactDescriptor
        from elspeth.engine.coalesce_executor import CoalesceExecutor
        from elspeth.engine.executors import (
            GateExecutor,
            SinkExecutor,
        )
        from elspeth.engine.spans import SpanFactory
        from elspeth.engine.tokens import TokenManager
        from elspeth.plugins.context import PluginContext

        db = LandscapeDB.in_memory()
        recorder = LandscapeRecorder(db)

        run = recorder.begin_run(config={}, canonical_version="v1")
        run_id = run.run_id

        DYNAMIC_SCHEMA = SchemaConfig.from_dict({"fields": "dynamic"})

        # Register nodes
        source_node = recorder.register_node(
            run_id=run_id,
            node_id="source",
            plugin_name="test_source",
            node_type=NodeType.SOURCE,
            plugin_version="1.0.0",
            config={},
            schema_config=DYNAMIC_SCHEMA,
        )
        gate_node = recorder.register_node(
            run_id=run_id,
            node_id="gate",
            plugin_name="config_gate:fork_gate",
            node_type=NodeType.GATE,
            plugin_version="1.0.0",
            config={},
            schema_config=DYNAMIC_SCHEMA,
        )
        recorder.register_node(
            run_id=run_id,
            node_id="coalesce",
            plugin_name="merge",
            node_type=NodeType.COALESCE,
            plugin_version="1.0.0",
            config={},
            schema_config=DYNAMIC_SCHEMA,
        )
        sink_node = recorder.register_node(
            run_id=run_id,
            node_id="sink",
            plugin_name="test_sink",
            node_type=NodeType.SINK,
            plugin_version="1.0.0",
            config={},
            schema_config=DYNAMIC_SCHEMA,
        )

        # Register edges
        edge_a = recorder.register_edge(
            run_id=run_id,
            from_node_id=gate_node.node_id,
            to_node_id="coalesce",
            label="path_a",
            mode=RoutingMode.COPY,
        )
        edge_b = recorder.register_edge(
            run_id=run_id,
            from_node_id=gate_node.node_id,
            to_node_id="coalesce",
            label="path_b",
            mode=RoutingMode.COPY,
        )

        edge_map = {
            (gate_node.node_id, "path_a"): edge_a.edge_id,
            (gate_node.node_id, "path_b"): edge_b.edge_id,
        }
        route_resolution_map = {
            (gate_node.node_id, "path_a"): "fork",
            (gate_node.node_id, "path_b"): "fork",
        }

        # Config-driven fork gate: forks every row into path_a and path_b
        fork_gate_config = GateSettings(
            name="fork_gate",
            condition="True",  # Always fork
            routes={"true": "fork", "false": "continue"},
            fork_to=["path_a", "path_b"],
        )

        class CollectSink(_TestSinkBase):
            name = "test_sink"

            def __init__(self, node_id: str) -> None:
                self.node_id = node_id
                self.rows_written: list[dict[str, Any]] = []

            def on_start(self, ctx: Any) -> None:
                pass

            def on_complete(self, ctx: Any) -> None:
                pass

            def write(self, rows: Any, ctx: Any) -> ArtifactDescriptor:
                self.rows_written.extend(rows)
                return ArtifactDescriptor.for_file(
                    path="memory://output",
                    size_bytes=100,
                    content_hash="test_hash",
                )

            def close(self) -> None:
                pass

        # Setup executors
        span_factory = SpanFactory()
        token_manager = TokenManager(recorder)

        coalesce_settings = CoalesceSettings(
            name="merge",
            branches=["path_a", "path_b"],
            policy="require_all",
            merge="union",
        )

        coalesce_executor = CoalesceExecutor(
            recorder=recorder,
            span_factory=span_factory,
            token_manager=token_manager,
            run_id=run_id,
        )
        coalesce_executor.register_coalesce(coalesce_settings, "coalesce")

        gate_executor = GateExecutor(
            recorder=recorder,
            span_factory=span_factory,
            edge_map=edge_map,
            route_resolution_map=route_resolution_map,
        )

        sink = CollectSink(sink_node.node_id)
        ctx = PluginContext(run_id=run_id, config={})

        # Process 3 source rows
        source_rows = [{"id": 1}, {"id": 2}, {"id": 3}]
        merged_tokens: list[TokenInfo] = []

        for idx, source_row in enumerate(source_rows):
            # Create initial token
            initial_token = token_manager.create_initial_token(
                run_id=run_id,
                source_node_id=source_node.node_id,
                row_index=idx,
                row_data=source_row,
            )

            # Fork using config-driven gate
            gate_outcome = gate_executor.execute_config_gate(
                gate_config=fork_gate_config,
                node_id=gate_node.node_id,
                token=initial_token,
                ctx=ctx,
                step_in_pipeline=1,
                token_manager=token_manager,
            )

            # Simulate branch processing - each branch adds its identifier
            for child in gate_outcome.child_tokens:
                processed_data = child.row_data.copy()
                processed_data[f"from_{child.branch_name}"] = True

                processed_token = TokenInfo(
                    row_id=child.row_id,
                    token_id=child.token_id,
                    row_data=processed_data,
                    branch_name=child.branch_name,
                )

                # Submit to coalesce
                outcome = coalesce_executor.accept(
                    token=processed_token,
                    coalesce_name="merge",
                    step_in_pipeline=2,
                )

                if not outcome.held and outcome.merged_token is not None:
                    merged_tokens.append(outcome.merged_token)

        # All 3 rows should have merged
        assert len(merged_tokens) == 3, f"Expected 3 merged tokens, got {len(merged_tokens)}"

        # Write to sink
        sink_executor = SinkExecutor(recorder, span_factory, run_id)
        sink_executor.write(
            sink=sink,
            tokens=merged_tokens,
            ctx=ctx,
            step_in_pipeline=3,
        )

        # Verify sink received exactly 3 merged rows
        assert len(sink.rows_written) == 3

        # Verify each row has data from both branches and correct ID
        for idx, row in enumerate(sink.rows_written):
            expected_id = idx + 1
            assert row["id"] == expected_id, f"Wrong ID in row {idx}"
            assert row["from_path_a"] is True, f"Missing path_a data in row {idx}"
            assert row["from_path_b"] is True, f"Missing path_b data in row {idx}"

        recorder.complete_run(run_id, status="completed")


class TestComplexDAGIntegration:
    """Tests for complex DAG patterns combining multiple features.

    These tests verify:
    - Diamond DAG: source -> fork -> parallel transforms -> coalesce -> sink
    - Nested fork/coalesce patterns
    - Mixed routing with aggregation
    """

    def test_diamond_dag_fork_transform_coalesce(self) -> None:
        """Diamond DAG pattern: source -> fork -> [transform_A, transform_B] -> coalesce -> sink.

        Pipeline flow:
        1. Source emits row with text
        2. Gate forks to sentiment_path and entity_path
        3. SentimentTransform adds sentiment field
        4. EntityTransform adds entities field
        5. Coalesce merges A+B results
        6. Sink receives merged row with BOTH sentiment AND entities

        Verifies:
        - Both transforms execute independently
        - Coalesce merges results from both branches
        - Sink receives single merged row per source row
        - Audit trail captures all node traversals
        """
        from elspeth.contracts import NodeType, TokenInfo
        from elspeth.contracts.schema import SchemaConfig
        from elspeth.core.config import CoalesceSettings
        from elspeth.core.landscape import LandscapeDB, LandscapeRecorder
        from elspeth.engine.artifacts import ArtifactDescriptor
        from elspeth.engine.coalesce_executor import CoalesceExecutor
        from elspeth.engine.executors import (
            GateExecutor,
            SinkExecutor,
            TransformExecutor,
        )
        from elspeth.engine.spans import SpanFactory
        from elspeth.engine.tokens import TokenManager
        from elspeth.plugins.context import PluginContext
        from elspeth.plugins.results import TransformResult

        db = LandscapeDB.in_memory()
        recorder = LandscapeRecorder(db)

        # Start a run
        run = recorder.begin_run(config={}, canonical_version="v1")
        run_id = run.run_id

        DYNAMIC_SCHEMA = SchemaConfig.from_dict({"fields": "dynamic"})

        # Register nodes for the diamond DAG
        source_node = recorder.register_node(
            run_id=run_id,
            node_id="source",
            plugin_name="test_source",
            node_type=NodeType.SOURCE,
            plugin_version="1.0.0",
            config={},
            schema_config=DYNAMIC_SCHEMA,
        )

        fork_gate_node = recorder.register_node(
            run_id=run_id,
            node_id="fork_gate",
            plugin_name="config_gate:fork_gate",
            node_type=NodeType.GATE,
            plugin_version="1.0.0",
            config={},
            schema_config=DYNAMIC_SCHEMA,
        )

        sentiment_transform_node = recorder.register_node(
            run_id=run_id,
            node_id="sentiment_transform",
            plugin_name="sentiment_analyzer",
            node_type=NodeType.TRANSFORM,
            plugin_version="1.0.0",
            config={},
            schema_config=DYNAMIC_SCHEMA,
        )

        entity_transform_node = recorder.register_node(
            run_id=run_id,
            node_id="entity_transform",
            plugin_name="entity_extractor",
            node_type=NodeType.TRANSFORM,
            plugin_version="1.0.0",
            config={},
            schema_config=DYNAMIC_SCHEMA,
        )

        merger_node = recorder.register_node(
            run_id=run_id,
            node_id="merger",
            plugin_name="coalesce:merger",
            node_type=NodeType.COALESCE,
            plugin_version="1.0.0",
            config={},
            schema_config=DYNAMIC_SCHEMA,
        )

        output_sink_node = recorder.register_node(
            run_id=run_id,
            node_id="output_sink",
            plugin_name="test_sink",
            node_type=NodeType.SINK,
            plugin_version="1.0.0",
            config={},
            schema_config=DYNAMIC_SCHEMA,
        )

        # Register edges for the diamond pattern
        edge_to_sentiment = recorder.register_edge(
            run_id=run_id,
            from_node_id=fork_gate_node.node_id,
            to_node_id=sentiment_transform_node.node_id,
            label="sentiment_path",
            mode=RoutingMode.COPY,
        )

        edge_to_entity = recorder.register_edge(
            run_id=run_id,
            from_node_id=fork_gate_node.node_id,
            to_node_id=entity_transform_node.node_id,
            label="entity_path",
            mode=RoutingMode.COPY,
        )

        # Edge map for gate executor
        edge_map = {
            (fork_gate_node.node_id, "sentiment_path"): edge_to_sentiment.edge_id,
            (fork_gate_node.node_id, "entity_path"): edge_to_entity.edge_id,
        }

        # Route resolution map for fork
        route_resolution_map: dict[tuple[str, str], str] = {
            (fork_gate_node.node_id, "sentiment_path"): "fork",
            (fork_gate_node.node_id, "entity_path"): "fork",
        }

        # Config-driven fork gate
        fork_gate = GateSettings(
            name="fork_gate",
            condition="True",  # Always fork
            routes={"true": "fork", "false": "continue"},
            fork_to=["sentiment_path", "entity_path"],
        )

        # Coalesce settings for merging
        coalesce_settings = CoalesceSettings(
            name="merger",
            branches=["sentiment_path", "entity_path"],
            policy="require_all",
            merge="union",
        )

        # Test transforms
        class SentimentTransform(BaseTransform):
            """Adds sentiment field based on text content."""

            name = "sentiment_analyzer"
            input_schema = _TestSchema
            output_schema = _TestSchema

            def __init__(self, node_id: str) -> None:
                super().__init__({})
                self.node_id = node_id

            def process(self, row: Any, ctx: Any) -> TransformResult:
                # Simple sentiment: "good" in text means positive
                text = row["text"]
                sentiment = "positive" if "good" in text.lower() else "neutral"
                return TransformResult.success({**row, "sentiment": sentiment})

        class EntityTransform(BaseTransform):
            """Extracts entities from text."""

            name = "entity_extractor"
            input_schema = _TestSchema
            output_schema = _TestSchema

            def __init__(self, node_id: str) -> None:
                super().__init__({})
                self.node_id = node_id

            def process(self, row: Any, ctx: Any) -> TransformResult:
                # Simple entity extraction: uppercase words are entities
                text = row["text"]
                entities = [word for word in text.split() if word.isupper()]
                return TransformResult.success({**row, "entities": entities})

        class CollectSink(_TestSinkBase):
            """Collects written rows for verification."""

            name = "test_sink"

            def __init__(self, node_id: str) -> None:
                self.node_id = node_id
                self.rows_written: list[dict[str, Any]] = []
                self._artifact_counter = 0

            def on_start(self, ctx: Any) -> None:
                pass

            def on_complete(self, ctx: Any) -> None:
                pass

            def write(self, rows: Any, ctx: Any) -> ArtifactDescriptor:
                self.rows_written.extend(rows)
                self._artifact_counter += 1
                return ArtifactDescriptor.for_file(
                    path=f"memory://diamond_output_{self._artifact_counter}",
                    size_bytes=len(str(rows)),
                    content_hash=f"diamond_hash_{self._artifact_counter}",
                )

            def close(self) -> None:
                pass

        # Create components
        span_factory = SpanFactory()
        token_manager = TokenManager(recorder)

        sentiment_transform = SentimentTransform(sentiment_transform_node.node_id)
        entity_transform = EntityTransform(entity_transform_node.node_id)
        sink = CollectSink(output_sink_node.node_id)

        gate_executor = GateExecutor(
            recorder=recorder,
            span_factory=span_factory,
            edge_map=edge_map,
            route_resolution_map=route_resolution_map,
        )

        transform_executor = TransformExecutor(recorder, span_factory)

        coalesce_executor = CoalesceExecutor(
            recorder=recorder,
            span_factory=span_factory,
            token_manager=token_manager,
            run_id=run_id,
        )
        coalesce_executor.register_coalesce(coalesce_settings, merger_node.node_id)

        sink_executor = SinkExecutor(recorder, span_factory, run_id)

        ctx = PluginContext(run_id=run_id, config={})

        # Test data: row with text
        source_row = {"text": "ACME reported good earnings"}

        # Step 1: Create initial token from source
        initial_token = token_manager.create_initial_token(
            run_id=run_id,
            source_node_id=source_node.node_id,
            row_index=0,
            row_data=source_row,
        )

        # Step 2: Execute fork gate
        gate_outcome = gate_executor.execute_config_gate(
            gate_config=fork_gate,
            node_id=fork_gate_node.node_id,
            token=initial_token,
            ctx=ctx,
            step_in_pipeline=1,
            token_manager=token_manager,
        )

        # Verify fork created 2 child tokens
        assert len(gate_outcome.child_tokens) == 2, f"Fork should create 2 children, got {len(gate_outcome.child_tokens)}"

        sentiment_token = next(t for t in gate_outcome.child_tokens if t.branch_name == "sentiment_path")
        entity_token = next(t for t in gate_outcome.child_tokens if t.branch_name == "entity_path")

        # Step 3: Execute transforms on each branch
        # Sentiment branch
        sentiment_result, sentiment_token_updated, _ = transform_executor.execute_transform(
            transform=as_transform(sentiment_transform),
            token=sentiment_token,
            ctx=ctx,
            step_in_pipeline=2,
        )
        assert sentiment_result.status == "success"
        assert sentiment_result.row is not None
        sentiment_token_processed = TokenInfo(
            row_id=sentiment_token_updated.row_id,
            token_id=sentiment_token_updated.token_id,
            row_data=sentiment_result.row,
            branch_name="sentiment_path",
        )

        # Entity branch
        entity_result, entity_token_updated, _ = transform_executor.execute_transform(
            transform=as_transform(entity_transform),
            token=entity_token,
            ctx=ctx,
            step_in_pipeline=2,
        )
        assert entity_result.status == "success"
        assert entity_result.row is not None
        entity_token_processed = TokenInfo(
            row_id=entity_token_updated.row_id,
            token_id=entity_token_updated.token_id,
            row_data=entity_result.row,
            branch_name="entity_path",
        )

        # Verify each transform added its respective field
        assert sentiment_token_processed.row_data["sentiment"] == "positive"
        assert entity_token_processed.row_data["entities"] == ["ACME"]

        # Step 4: Coalesce the branches
        outcome1 = coalesce_executor.accept(
            token=sentiment_token_processed,
            coalesce_name="merger",
            step_in_pipeline=3,
        )
        assert outcome1.held is True, "First branch should be held waiting for second"

        outcome2 = coalesce_executor.accept(
            token=entity_token_processed,
            coalesce_name="merger",
            step_in_pipeline=3,
        )
        assert outcome2.held is False, "Second branch should trigger merge"
        assert outcome2.merged_token is not None, "Merge should produce a token"

        # Verify merged data contains fields from BOTH branches
        merged_data = outcome2.merged_token.row_data
        assert "text" in merged_data, "Merged data should preserve original text"
        assert "sentiment" in merged_data, "Merged data should have sentiment from branch A"
        assert "entities" in merged_data, "Merged data should have entities from branch B"
        assert merged_data["text"] == "ACME reported good earnings"
        assert merged_data["sentiment"] == "positive"
        assert merged_data["entities"] == ["ACME"]

        # Step 5: Write merged token to sink
        artifact = sink_executor.write(
            sink=sink,
            tokens=[outcome2.merged_token],
            ctx=ctx,
            step_in_pipeline=4,
        )

        # Verify sink received exactly 1 merged row (not 2 separate branch outputs)
        assert len(sink.rows_written) == 1, (
            f"Expected 1 merged row, got {len(sink.rows_written)}. Diamond DAG should merge branches before sink."
        )

        # Verify the written row contains the merged fields
        written_row = sink.rows_written[0]
        assert written_row["text"] == "ACME reported good earnings"
        assert written_row["sentiment"] == "positive"
        assert written_row["entities"] == ["ACME"]

        # Verify artifact was recorded
        assert artifact is not None
        assert artifact.content_hash == "diamond_hash_1"

        # Complete the run
        recorder.complete_run(run_id, status="completed")

        # Verify audit trail completeness
        rows = recorder.get_rows(run_id)
        assert len(rows) == 1, "Should have exactly 1 source row"

        all_tokens = recorder.get_tokens(rows[0].row_id)
        # 1 initial + 2 fork children + 1 merged = 4 tokens
        assert len(all_tokens) == 4, f"Expected 4 tokens (initial + 2 fork + 1 merged), got {len(all_tokens)}"

        # Verify all nodes were visited
        nodes = recorder.get_nodes(run_id)
        node_ids = {n.node_id for n in nodes}
        expected_nodes = {
            "source",
            "fork_gate",
            "sentiment_transform",
            "entity_transform",
            "merger",
            "output_sink",
        }
        assert expected_nodes <= node_ids, f"Missing nodes: {expected_nodes - node_ids}"

    def test_full_feature_pipeline_deleted(self) -> None:
        """Test deleted - used old BaseAggregation interface.

        OLD: Tested full pipeline with gate routing + fork + coalesce + aggregation
             using BaseAggregation plugins (accept/flush interface).
        NEW: When batch-aware transforms are integrated into pipeline config,
             a new test should be added for complete DAG execution.
        """
        import elspeth.plugins.base as base

        assert not hasattr(base, "BaseAggregation"), "BaseAggregation should be deleted - use is_batch_aware=True on BaseTransform"

    def test_run_result_captures_all_metrics(self) -> None:
        """RunResult captures metrics for all pipeline operations.

        This test verifies that RunResult includes all expected metrics
        and that they are consistent with what actually happened in the pipeline.

        Verifies:
        - rows_processed: Total rows from source
        - rows_succeeded: Rows that completed successfully
        - rows_failed: Rows that failed processing
        - rows_quarantined: Rows quarantined due to errors
        - rows_routed: Rows routed to named sinks by gates
        - rows_forked: Rows that were forked to multiple paths

        Consistency checks:
        - All metrics >= 0
        - rows_succeeded + rows_quarantined <= rows_processed
        """
        from elspeth.contracts import PluginSchema, RunStatus
        from elspeth.core.config import GateSettings
        from elspeth.core.landscape import LandscapeDB
        from elspeth.engine import Orchestrator, PipelineConfig
        from elspeth.engine.artifacts import ArtifactDescriptor
        from elspeth.plugins.results import TransformResult

        db = LandscapeDB.in_memory()

        class ValueSchema(PluginSchema):
            value: int

        class ListSource(_TestSourceBase):
            """Source emitting test data."""

            name = "test_source"
            output_schema = ValueSchema

            def __init__(self, data: list[dict[str, Any]]) -> None:
                self._data = data

            def on_start(self, ctx: Any) -> None:
                pass

            def on_complete(self, ctx: Any) -> None:
                pass

            def load(self, ctx: Any) -> Any:
                for _row in self._data:
                    yield SourceRow.valid(_row)

            def close(self) -> None:
                pass

        class SelectiveTransform(BaseTransform):
            """Transform that fails on specific values.

            - Values divisible by 3 fail (will be quarantined)
            - Other values succeed

            Has _on_error="discard" so errors become QUARANTINED.
            """

            name = "selective_transform"
            input_schema = ValueSchema
            output_schema = ValueSchema
            _on_error = "discard"  # Required for TransformResult.error() to work

            def __init__(self) -> None:
                super().__init__({})

            def process(self, row: dict[str, Any], ctx: Any) -> TransformResult:
                if row["value"] % 3 == 0:
                    return TransformResult.error({"reason": "divisible_by_3", "value": row["value"]})
                return TransformResult.success({"value": row["value"], "processed": True})

        class CollectSink(_TestSinkBase):
            """Sink that collects written rows."""

            name = "output_sink"

            def __init__(self) -> None:
                self.results: list[dict[str, Any]] = []

            def on_start(self, ctx: Any) -> None:
                pass

            def on_complete(self, ctx: Any) -> None:
                pass

            def write(self, rows: Any, ctx: Any) -> ArtifactDescriptor:
                self.results.extend(rows)
                return ArtifactDescriptor.for_file(
                    path="memory://output",
                    size_bytes=len(str(rows)),
                    content_hash="metrics_test",
                )

            def close(self) -> None:
                pass

        class RoutedSink(_TestSinkBase):
            """Sink for routed rows."""

            name = "routed_sink"

            def __init__(self) -> None:
                self.results: list[dict[str, Any]] = []

            def on_start(self, ctx: Any) -> None:
                pass

            def on_complete(self, ctx: Any) -> None:
                pass

            def write(self, rows: Any, ctx: Any) -> ArtifactDescriptor:
                self.results.extend(rows)
                return ArtifactDescriptor.for_file(
                    path="memory://routed",
                    size_bytes=len(str(rows)),
                    content_hash="routed_metrics",
                )

            def close(self) -> None:
                pass

        # Create pipeline with:
        # - 10 rows (values 1-10)
        # - Gate routes values >= 8 to "routed" sink
        # - Transform fails on divisible by 3 (3, 6, 9)
        # - Remaining go to default sink
        source = ListSource([{"value": i} for i in range(1, 11)])  # 1-10
        transform = SelectiveTransform()
        default_sink = CollectSink()
        routed_sink = RoutedSink()

        # Gate: route values >= 8 to routed sink
        routing_gate = GateSettings(
            name="routing_gate",
            condition="row['value'] >= 8",
            routes={"true": "routed", "false": "continue"},
        )

        config = PipelineConfig(
            source=as_source(source),
            transforms=[transform],
            sinks={"default": default_sink, "routed": routed_sink},
            gates=[routing_gate],
        )

        orchestrator = Orchestrator(db)
        result = orchestrator.run(config, graph=_build_test_graph(config))

        # ================================================================
        # Verify all metrics are non-negative
        # ================================================================
        assert result.rows_processed >= 0, "rows_processed must be >= 0"
        assert result.rows_succeeded >= 0, "rows_succeeded must be >= 0"
        assert result.rows_failed >= 0, "rows_failed must be >= 0"
        assert result.rows_quarantined >= 0, "rows_quarantined must be >= 0"
        assert result.rows_routed >= 0, "rows_routed must be >= 0"
        assert result.rows_forked >= 0, "rows_forked must be >= 0"

        # ================================================================
        # Verify run completed
        # ================================================================
        assert result.status == RunStatus.COMPLETED

        # ================================================================
        # Verify expected metric values
        # ================================================================
        # 10 rows processed (values 1-10)
        assert result.rows_processed == 10, f"Expected 10 rows_processed, got {result.rows_processed}"

        # Pipeline flow:
        # 1. Transform processes all 10 rows FIRST
        #    - Values 3, 6, 9 fail (divisible by 3) -> QUARANTINED
        #    - Values 1, 2, 4, 5, 7, 8, 10 succeed (7 rows continue)
        # 2. Gate routes the 7 surviving rows:
        #    - Values >= 8: 8, 10 -> ROUTED to "routed" sink
        #    - Values < 8: 1, 2, 4, 5, 7 -> SUCCEEDED to "default" sink

        # Quarantined: 3, 6, 9 (3 rows failed at transform)
        assert result.rows_quarantined == 3, f"Expected 3 rows_quarantined (3, 6, 9), got {result.rows_quarantined}"

        # Routed: 8, 10 (2 rows that passed transform and got routed by gate)
        assert result.rows_routed == 2, f"Expected 2 rows_routed (8, 10), got {result.rows_routed}"

        # Succeeded: 1, 2, 4, 5, 7 (5 rows that reached default sink)
        assert result.rows_succeeded == 5, f"Expected 5 rows_succeeded (1, 2, 4, 5, 7), got {result.rows_succeeded}"

        # Failed: 0 (no exceptions, only TransformResult.error with discard)
        assert result.rows_failed == 0, f"Expected 0 rows_failed (errors become quarantined), got {result.rows_failed}"

        # ================================================================
        # Verify consistency: succeeded + quarantined <= processed
        # ================================================================
        total_terminal = result.rows_succeeded + result.rows_quarantined
        assert total_terminal <= result.rows_processed, (
            f"Consistency check failed: "
            f"rows_succeeded ({result.rows_succeeded}) + "
            f"rows_quarantined ({result.rows_quarantined}) = {total_terminal} "
            f"should be <= rows_processed ({result.rows_processed})"
        )

        # ================================================================
        # Verify sink outputs match metrics
        # ================================================================
        # Default sink receives rows that:
        # 1. Pass the transform (not divisible by 3)
        # 2. Are not routed by gate (value < 8)
        # Expected: 1, 2, 4, 5, 7 (5 rows)
        default_values = [r["value"] for r in default_sink.results]
        assert sorted(default_values) == [
            1,
            2,
            4,
            5,
            7,
        ], f"Default sink expected [1, 2, 4, 5, 7], got {sorted(default_values)}"

        # Routed sink receives rows that:
        # 1. Pass the transform (not divisible by 3)
        # 2. Are routed by gate (value >= 8)
        # Expected: 8, 10 (2 rows - 9 failed at transform before reaching gate)
        routed_values = [r["value"] for r in routed_sink.results]
        assert sorted(routed_values) == [
            8,
            10,
        ], f"Routed sink expected [8, 10], got {sorted(routed_values)}"

        # ================================================================
        # Verify no forking in this pipeline (rows_forked should be 0)
        # ================================================================
        assert result.rows_forked == 0, f"Expected 0 rows_forked (no fork gates), got {result.rows_forked}"


class TestRetryIntegration:
    """End-to-end retry behavior tests.

    These tests verify retry integration with the full pipeline:
    - Transient failures that succeed after retries
    - Permanent failures that exhaust retries and get quarantined
    - Audit trail records all attempts
    """

    def test_transient_failure_retries_and_succeeds(self) -> None:
        """Transform that fails twice then succeeds should complete.

        Pipeline: source -> flaky_transform (fails 2x, succeeds 3rd) -> sink

        Verifies:
        - RetryManager triggers retries on ConnectionError
        - Audit trail records all attempts (attempt 0, 1, 2)
        - Final success reaches sink
        - All rows complete successfully
        """
        from collections import defaultdict

        from sqlalchemy import select

        from elspeth.contracts import PluginSchema, RunStatus
        from elspeth.core.config import ElspethSettings
        from elspeth.core.landscape import LandscapeDB, LandscapeRecorder
        from elspeth.core.landscape.schema import node_states_table
        from elspeth.engine import Orchestrator, PipelineConfig
        from elspeth.engine.artifacts import ArtifactDescriptor
        from elspeth.plugins.results import TransformResult

        db = LandscapeDB.in_memory()

        class ValueSchema(PluginSchema):
            value: int

        class ListSource(_TestSourceBase):
            name = "test_source"
            output_schema = ValueSchema

            def __init__(self, data: list[dict[str, Any]]) -> None:
                self._data = data

            def on_start(self, ctx: Any) -> None:
                pass

            def on_complete(self, ctx: Any) -> None:
                pass

            def load(self, ctx: Any) -> Any:
                for _row in self._data:
                    yield SourceRow.valid(_row)

            def close(self) -> None:
                pass

        # Track attempt counts per row
        attempt_counts: dict[int, int] = defaultdict(int)

        class FlakyTransform(BaseTransform):
            """Transform that fails first 2 times with ConnectionError, succeeds on 3rd."""

            name = "flaky_transform"
            input_schema = ValueSchema
            output_schema = ValueSchema

            def __init__(self) -> None:
                super().__init__({})

            def process(self, row: dict[str, Any], ctx: Any) -> TransformResult:
                row_value = row["value"]
                attempt_counts[row_value] += 1

                if attempt_counts[row_value] < 3:
                    # Raise ConnectionError - this is retryable
                    raise ConnectionError(f"Transient failure attempt {attempt_counts[row_value]}")

                return TransformResult.success({"value": row_value, "processed": True})

        class CollectSink(_TestSinkBase):
            name = "output_sink"

            def __init__(self) -> None:
                self.results: list[dict[str, Any]] = []

            def on_start(self, ctx: Any) -> None:
                pass

            def on_complete(self, ctx: Any) -> None:
                pass

            def write(self, rows: Any, ctx: Any) -> ArtifactDescriptor:
                self.results.extend(rows)
                return ArtifactDescriptor.for_file(path="memory://output", size_bytes=100, content_hash="retry_test")

            def close(self) -> None:
                pass

        # Create pipeline
        source = ListSource([{"value": 1}, {"value": 2}])
        transform = FlakyTransform()
        sink = CollectSink()

        config = PipelineConfig(
            source=as_source(source),
            transforms=[transform],
            sinks={"default": sink},
        )

        # Create settings with retry enabled (fast delays for testing)
        # ElspethSettings requires datasource, sinks, output_sink but
        # those are not used when orchestrator.run() already has PipelineConfig
        settings = ElspethSettings(
            datasource={"plugin": "memory"},
            sinks={"default": {"plugin": "memory"}},
            output_sink="default",
            retry={
                "max_attempts": 5,
                "initial_delay_seconds": 0.001,
                "max_delay_seconds": 0.01,
            },
        )

        orchestrator = Orchestrator(db)
        result = orchestrator.run(config, graph=_build_test_graph(config), settings=settings)

        # Verify run completed successfully
        assert result.status == RunStatus.COMPLETED
        assert result.rows_processed == 2
        assert result.rows_succeeded == 2

        # Both rows should have been retried (3 attempts each)
        assert attempt_counts[1] == 3
        assert attempt_counts[2] == 3

        # Sink received both rows
        assert len(sink.results) == 2
        assert all(r["processed"] for r in sink.results)

        # Verify audit trail records all attempts
        recorder = LandscapeRecorder(db)
        nodes = recorder.get_nodes(result.run_id)
        transform_node = next(n for n in nodes if n.plugin_name == "flaky_transform")

        with db.engine.connect() as conn:
            stmt = (
                select(node_states_table)
                .where(node_states_table.c.node_id == transform_node.node_id)
                .order_by(node_states_table.c.token_id, node_states_table.c.attempt)
            )
            states = list(conn.execute(stmt))

        # Should have 3 attempts per row = 6 node_states
        assert len(states) == 6, f"Expected 6 node_states (3 per row), got {len(states)}"

        # Verify attempt numbers: each row has attempts 0, 1, 2
        # Group by token_id and check attempts
        attempts_by_token: dict[str, list[int]] = defaultdict(list)
        for state in states:
            attempts_by_token[state.token_id].append(state.attempt)

        for token_id, attempts in attempts_by_token.items():
            assert sorted(attempts) == [0, 1, 2], f"Token {token_id} should have attempts [0, 1, 2], got {sorted(attempts)}"

        # Verify final attempts are successful
        final_states = [s for s in states if s.attempt == 2]
        assert all(s.status == "completed" for s in final_states), "All final attempts should be 'completed'"

    def test_permanent_failure_quarantines_after_max_retries(self) -> None:
        """Transform that always fails should quarantine row after max retries.

        Pipeline: source -> always_fail_transform -> sink

        Verifies:
        - RetryManager exhausts retries (max_attempts=3)
        - Failed row is FAILED, not lost
        - Other rows (from different source) still succeed
        - Audit trail shows all failed attempts
        """
        from collections import defaultdict

        from sqlalchemy import select

        from elspeth.contracts import PluginSchema, RunStatus
        from elspeth.core.config import ElspethSettings
        from elspeth.core.landscape import LandscapeDB, LandscapeRecorder
        from elspeth.core.landscape.schema import node_states_table
        from elspeth.engine import Orchestrator, PipelineConfig
        from elspeth.engine.artifacts import ArtifactDescriptor
        from elspeth.plugins.results import TransformResult

        db = LandscapeDB.in_memory()

        class ValueSchema(PluginSchema):
            value: int

        class ListSource(_TestSourceBase):
            name = "test_source"
            output_schema = ValueSchema

            def __init__(self, data: list[dict[str, Any]]) -> None:
                self._data = data

            def on_start(self, ctx: Any) -> None:
                pass

            def on_complete(self, ctx: Any) -> None:
                pass

            def load(self, ctx: Any) -> Any:
                for _row in self._data:
                    yield SourceRow.valid(_row)

            def close(self) -> None:
                pass

        # Track attempt counts per row
        attempt_counts: dict[int, int] = defaultdict(int)

        class SelectiveFailTransform(BaseTransform):
            """Transform that always fails for value=1 but succeeds for value=2."""

            name = "selective_fail"
            input_schema = ValueSchema
            output_schema = ValueSchema

            def __init__(self) -> None:
                super().__init__({})

            def process(self, row: dict[str, Any], ctx: Any) -> TransformResult:
                row_value = row["value"]
                attempt_counts[row_value] += 1

                if row_value == 1:
                    # Always fail with ConnectionError (retryable)
                    raise ConnectionError(f"Permanent failure for value=1, attempt {attempt_counts[row_value]}")

                return TransformResult.success({"value": row_value, "processed": True})

        class CollectSink(_TestSinkBase):
            name = "output_sink"

            def __init__(self) -> None:
                self.results: list[dict[str, Any]] = []

            def on_start(self, ctx: Any) -> None:
                pass

            def on_complete(self, ctx: Any) -> None:
                pass

            def write(self, rows: Any, ctx: Any) -> ArtifactDescriptor:
                self.results.extend(rows)
                return ArtifactDescriptor.for_file(
                    path="memory://output",
                    size_bytes=100,
                    content_hash="quarantine_test",
                )

            def close(self) -> None:
                pass

        # Create pipeline with 2 rows: value=1 (fails) and value=2 (succeeds)
        source = ListSource([{"value": 1}, {"value": 2}])
        transform = SelectiveFailTransform()
        sink = CollectSink()

        config = PipelineConfig(
            source=as_source(source),
            transforms=[transform],
            sinks={"default": sink},
        )

        # Create settings with retry enabled (max 3 attempts, fast delays)
        # ElspethSettings requires datasource, sinks, output_sink but
        # those are not used when orchestrator.run() already has PipelineConfig
        settings = ElspethSettings(
            datasource={"plugin": "memory"},
            sinks={"default": {"plugin": "memory"}},
            output_sink="default",
            retry={
                "max_attempts": 3,
                "initial_delay_seconds": 0.001,
                "max_delay_seconds": 0.01,
            },
        )

        orchestrator = Orchestrator(db)
        result = orchestrator.run(config, graph=_build_test_graph(config), settings=settings)

        # Run completes (partial success)
        assert result.status == RunStatus.COMPLETED
        assert result.rows_processed == 2

        # Row with value=1 failed after max retries (3 attempts)
        # Row with value=2 succeeded
        assert result.rows_failed == 1, f"Expected 1 failed row, got {result.rows_failed}"
        assert result.rows_succeeded == 1, f"Expected 1 succeeded row, got {result.rows_succeeded}"

        # Verify attempt counts
        assert attempt_counts[1] == 3, f"value=1 should have 3 attempts, got {attempt_counts[1]}"
        assert attempt_counts[2] == 1, f"value=2 should have 1 attempt, got {attempt_counts[2]}"

        # Sink only received the successful row
        assert len(sink.results) == 1
        assert sink.results[0]["value"] == 2
        assert sink.results[0]["processed"] is True

        # Verify audit trail for failed row shows all attempts
        recorder = LandscapeRecorder(db)
        nodes = recorder.get_nodes(result.run_id)
        transform_node = next(n for n in nodes if n.plugin_name == "selective_fail")

        with db.engine.connect() as conn:
            stmt = (
                select(node_states_table)
                .where(node_states_table.c.node_id == transform_node.node_id)
                .order_by(node_states_table.c.token_id, node_states_table.c.attempt)
            )
            states = list(conn.execute(stmt))

        # Should have: 3 failed attempts for value=1 + 1 success for value=2 = 4
        assert len(states) == 4, f"Expected 4 node_states, got {len(states)}"

        # Verify statuses: 3 failed + 1 completed
        statuses = [s.status for s in states]
        assert statuses.count("failed") == 3, f"Expected 3 failed, got {statuses}"
        assert statuses.count("completed") == 1, f"Expected 1 completed, got {statuses}"

        # Verify all failed attempts have error_json populated
        failed_states = [s for s in states if s.status == "failed"]
        for failed in failed_states:
            assert failed.error_json is not None, "Failed states must have error_json"


class TestExplainQuery:
    """Tests verifying explain() query functionality for audit trail traceability.

    The explain() function is the primary query interface for answering
    "what happened to this row?" - critical for audit and debugging.
    """

    def test_explain_traces_output_to_source(self) -> None:
        """Explain traces any output back to its source.

        For any token that reached a sink, explain() should provide:
        - Which source row it came from
        - Every transform it passed through (via node_states)
        - The final sink that received it

        This is THE fundamental audit query.
        """
        from elspeth.contracts import NodeType
        from elspeth.contracts.schema import SchemaConfig
        from elspeth.core.landscape import LandscapeDB, LandscapeRecorder
        from elspeth.core.landscape.lineage import explain
        from elspeth.engine.tokens import TokenManager

        db = LandscapeDB.in_memory()
        recorder = LandscapeRecorder(db)

        DYNAMIC_SCHEMA = SchemaConfig.from_dict({"fields": "dynamic"})

        # Start a run
        run = recorder.begin_run(config={}, canonical_version="v1")
        run_id = run.run_id

        # Register nodes for a 3-step pipeline: source -> transform -> sink
        source_node = recorder.register_node(
            run_id=run_id,
            node_id="source",
            plugin_name="test_source",
            node_type=NodeType.SOURCE,
            plugin_version="1.0.0",
            config={},
            schema_config=DYNAMIC_SCHEMA,
        )
        transform_node = recorder.register_node(
            run_id=run_id,
            node_id="transform",
            plugin_name="test_transform",
            node_type=NodeType.TRANSFORM,
            plugin_version="1.0.0",
            config={},
            sequence=1,
            schema_config=DYNAMIC_SCHEMA,
        )
        sink_node = recorder.register_node(
            run_id=run_id,
            node_id="sink",
            plugin_name="test_sink",
            node_type=NodeType.SINK,
            plugin_version="1.0.0",
            config={},
            sequence=2,
            schema_config=DYNAMIC_SCHEMA,
        )

        # Register edges
        recorder.register_edge(
            run_id=run_id,
            from_node_id=source_node.node_id,
            to_node_id=transform_node.node_id,
            label="continue",
            mode="move",
        )
        recorder.register_edge(
            run_id=run_id,
            from_node_id=transform_node.node_id,
            to_node_id=sink_node.node_id,
            label="continue",
            mode="move",
        )

        # Create a row and token
        token_manager = TokenManager(recorder)

        source_data = {"value": 42, "name": "test_row"}
        token = token_manager.create_initial_token(
            run_id=run_id,
            source_node_id=source_node.node_id,
            row_index=0,
            row_data=source_data,
        )

        # Record transform processing
        state1 = recorder.begin_node_state(
            token_id=token.token_id,
            node_id=transform_node.node_id,
            step_index=0,
            input_data=source_data,
        )
        transformed_data = {"value": 84, "name": "test_row", "doubled": True}
        recorder.complete_node_state(
            state_id=state1.state_id,
            status="completed",
            output_data=transformed_data,
            duration_ms=10.0,
        )

        # Record sink processing
        state2 = recorder.begin_node_state(
            token_id=token.token_id,
            node_id=sink_node.node_id,
            step_index=1,
            input_data=transformed_data,
        )
        recorder.complete_node_state(
            state_id=state2.state_id,
            status="completed",
            output_data=transformed_data,
            duration_ms=5.0,
        )

        recorder.complete_run(run_id, status="completed")

        # Now test explain()
        lineage = explain(recorder, run_id, token_id=token.token_id)

        # Lineage must exist
        assert lineage is not None, "explain() returned None for valid token"

        # Must link back to source row
        assert lineage.source_row is not None, "source_row must be present"
        assert lineage.source_row.row_id == token.row_id, "source_row.row_id must match token's row_id"
        assert lineage.source_row.source_node_id == source_node.node_id, "source_row must track source node"
        assert lineage.source_row.row_index == 0, "source_row.row_index must match original"

        # Must have node_states covering transform AND sink
        assert len(lineage.node_states) == 2, f"Expected 2 node_states (transform + sink), got {len(lineage.node_states)}"

        node_ids_in_states = {s.node_id for s in lineage.node_states}
        assert transform_node.node_id in node_ids_in_states, "node_states must include transform"
        assert sink_node.node_id in node_ids_in_states, "node_states must include sink"

        # All states must be completed
        for state in lineage.node_states:
            assert state.status == "completed", f"All states should be completed, got {state.status}"

        # Verify states are in step_index order
        step_indices = [s.step_index for s in lineage.node_states]
        assert step_indices == sorted(step_indices), "node_states should be ordered by step_index"

    def test_explain_for_aggregated_row(self) -> None:
        """Explain traces aggregated output back to all input rows.

        When aggregation produces one output from N inputs:
        - explain() should trace back to all N source rows via batch_members table
        - The batch_members link the aggregation output to all contributing tokens
        """
        from elspeth.contracts import NodeType
        from elspeth.contracts.schema import SchemaConfig
        from elspeth.core.landscape import LandscapeDB, LandscapeRecorder
        from elspeth.engine.tokens import TokenManager

        db = LandscapeDB.in_memory()
        recorder = LandscapeRecorder(db)

        DYNAMIC_SCHEMA = SchemaConfig.from_dict({"fields": "dynamic"})

        # Start a run
        run = recorder.begin_run(config={}, canonical_version="v1")
        run_id = run.run_id

        # Register nodes: source -> aggregation -> sink
        source_node = recorder.register_node(
            run_id=run_id,
            node_id="source",
            plugin_name="test_source",
            node_type=NodeType.SOURCE,
            plugin_version="1.0.0",
            config={},
            schema_config=DYNAMIC_SCHEMA,
        )
        agg_node = recorder.register_node(
            run_id=run_id,
            node_id="aggregation",
            plugin_name="sum_aggregation",
            node_type=NodeType.AGGREGATION,
            plugin_version="1.0.0",
            config={},
            sequence=1,
            schema_config=DYNAMIC_SCHEMA,
        )
        # Note: sink node not needed for this test - we're testing batch_members traceability

        # Create 3 input tokens (simulating 3 source rows being aggregated)
        token_manager = TokenManager(recorder)
        input_tokens = []
        for i in range(3):
            token = token_manager.create_initial_token(
                run_id=run_id,
                source_node_id=source_node.node_id,
                row_index=i,
                row_data={"value": (i + 1) * 10},  # 10, 20, 30
            )
            input_tokens.append(token)

            # Record aggregation acceptance (CONSUMED_IN_BATCH state)
            state = recorder.begin_node_state(
                token_id=token.token_id,
                node_id=agg_node.node_id,
                step_index=0,
                input_data={"value": (i + 1) * 10},
            )
            recorder.complete_node_state(
                state_id=state.state_id,
                status="completed",
                output_data={"accepted": True},
                duration_ms=1.0,
            )

        # Create a batch linking all 3 tokens
        batch = recorder.create_batch(
            run_id=run_id,
            aggregation_node_id=agg_node.node_id,
        )

        # Add all tokens as batch members
        for ordinal, token in enumerate(input_tokens):
            recorder.add_batch_member(
                batch_id=batch.batch_id,
                token_id=token.token_id,
                ordinal=ordinal,
            )

        # Complete the batch
        recorder.complete_batch(
            batch_id=batch.batch_id,
            status="completed",
            trigger_reason="count_reached",
        )

        recorder.complete_run(run_id, status="completed")

        # Verify batch_members links aggregation to all input tokens
        batch_members = recorder.get_batch_members(batch.batch_id)
        assert len(batch_members) == 3, f"Expected 3 batch members, got {len(batch_members)}"

        # Verify each batch member maps to a distinct token
        member_token_ids = {m.token_id for m in batch_members}
        input_token_ids = {t.token_id for t in input_tokens}
        assert member_token_ids == input_token_ids, (
            f"batch_members should link to all input tokens. Expected {input_token_ids}, got {member_token_ids}"
        )

        # Verify we can trace from each batch member back to its source row
        for member in batch_members:
            token_info = recorder.get_token(member.token_id)
            assert token_info is not None, f"Token {member.token_id} should exist"

            row = recorder.get_row(token_info.row_id)
            assert row is not None, f"Row {token_info.row_id} should exist"
            assert row.source_node_id == source_node.node_id, "Row should trace to source node"

    def test_explain_for_coalesced_row(self) -> None:
        """Explain traces coalesced output back through branches to fork point.

        When coalesce merges N branch outputs:
        - explain() should trace through all branches back to fork point
        - Should show parent token that forked
        - Uses parent_token_id to trace lineage
        """
        from elspeth.contracts import NodeType
        from elspeth.contracts.schema import SchemaConfig
        from elspeth.core.landscape import LandscapeDB, LandscapeRecorder
        from elspeth.core.landscape.lineage import explain
        from elspeth.engine.tokens import TokenManager

        db = LandscapeDB.in_memory()
        recorder = LandscapeRecorder(db)

        DYNAMIC_SCHEMA = SchemaConfig.from_dict({"fields": "dynamic"})

        # Start a run
        run = recorder.begin_run(config={}, canonical_version="v1")
        run_id = run.run_id

        # Register nodes: source -> fork gate -> [path_a, path_b] -> coalesce -> sink
        source_node = recorder.register_node(
            run_id=run_id,
            node_id="source",
            plugin_name="test_source",
            node_type=NodeType.SOURCE,
            plugin_version="1.0.0",
            config={},
            schema_config=DYNAMIC_SCHEMA,
        )
        gate_node = recorder.register_node(
            run_id=run_id,
            node_id="fork_gate",
            plugin_name="fork_gate",
            node_type=NodeType.GATE,
            plugin_version="1.0.0",
            config={},
            sequence=1,
            schema_config=DYNAMIC_SCHEMA,
        )
        coalesce_node = recorder.register_node(
            run_id=run_id,
            node_id="coalesce",
            plugin_name="merge_coalesce",
            node_type=NodeType.COALESCE,
            plugin_version="1.0.0",
            config={},
            sequence=2,
            schema_config=DYNAMIC_SCHEMA,
        )
        sink_node = recorder.register_node(
            run_id=run_id,
            node_id="sink",
            plugin_name="test_sink",
            node_type=NodeType.SINK,
            plugin_version="1.0.0",
            config={},
            sequence=3,
            schema_config=DYNAMIC_SCHEMA,
        )
        # Register branch transform nodes for FK integrity
        recorder.register_node(
            run_id=run_id,
            node_id="transform_path_a",
            plugin_name="branch_transform",
            node_type=NodeType.TRANSFORM,
            plugin_version="1.0.0",
            config={},
            schema_config=DYNAMIC_SCHEMA,
        )
        recorder.register_node(
            run_id=run_id,
            node_id="transform_path_b",
            plugin_name="branch_transform",
            node_type=NodeType.TRANSFORM,
            plugin_version="1.0.0",
            config={},
            schema_config=DYNAMIC_SCHEMA,
        )

        # Create initial token (before fork)
        token_manager = TokenManager(recorder)
        source_data = {"value": 100, "name": "original"}
        parent_token = token_manager.create_initial_token(
            run_id=run_id,
            source_node_id=source_node.node_id,
            row_index=0,
            row_data=source_data,
        )

        # Record parent token processing at gate (fork point)
        gate_state = recorder.begin_node_state(
            token_id=parent_token.token_id,
            node_id=gate_node.node_id,
            step_index=0,
            input_data=source_data,
        )
        recorder.complete_node_state(
            state_id=gate_state.state_id,
            status="completed",
            output_data={"forked_to": ["path_a", "path_b"]},
            duration_ms=1.0,
        )

        # Fork into two children using token_manager (records parent relationships)
        child_tokens = token_manager.fork_token(
            parent_token=parent_token,
            branches=["path_a", "path_b"],
            step_in_pipeline=1,
        )
        token_a = child_tokens[0]
        token_b = child_tokens[1]

        # Record child token processing (each branch does different work)
        for child in child_tokens:
            branch_state = recorder.begin_node_state(
                token_id=child.token_id,
                node_id=f"transform_{child.branch_name}",  # Pseudo-node
                step_index=1,
                input_data=source_data,
            )
            recorder.complete_node_state(
                state_id=branch_state.state_id,
                status="completed",
                output_data={f"{child.branch_name}_result": True},
                duration_ms=5.0,
            )

        # Coalesce: merge the two child tokens into one
        # Use recorder's coalesce_tokens to properly record parent relationships
        merged_token = recorder.coalesce_tokens(
            parent_token_ids=[token_a.token_id, token_b.token_id],
            row_id=parent_token.row_id,  # Same source row
            step_in_pipeline=2,
        )

        # Record coalesce processing
        coalesce_state = recorder.begin_node_state(
            token_id=merged_token.token_id,
            node_id=coalesce_node.node_id,
            step_index=2,
            input_data={"path_a_result": True, "path_b_result": True},
        )
        recorder.complete_node_state(
            state_id=coalesce_state.state_id,
            status="completed",
            output_data={"merged": True, "path_a_result": True, "path_b_result": True},
            duration_ms=2.0,
        )

        # Record sink processing
        sink_state = recorder.begin_node_state(
            token_id=merged_token.token_id,
            node_id=sink_node.node_id,
            step_index=3,
            input_data={"merged": True},
        )
        recorder.complete_node_state(
            state_id=sink_state.state_id,
            status="completed",
            output_data={"written": True},
            duration_ms=1.0,
        )

        recorder.complete_run(run_id, status="completed")

        # Test explain() on the coalesced token
        lineage = explain(recorder, run_id, token_id=merged_token.token_id)

        # Lineage must exist
        assert lineage is not None, "explain() returned None for coalesced token"

        # Must trace to original source row
        assert lineage.source_row is not None, "source_row must be present"
        assert lineage.source_row.row_id == parent_token.row_id, "source_row should trace to original row before fork"
        assert lineage.source_row.source_node_id == source_node.node_id, "source_row should trace to source node"

        # Must have parent tokens (the forked children that were merged)
        assert len(lineage.parent_tokens) == 2, f"Coalesced token should have 2 parent tokens, got {len(lineage.parent_tokens)}"

        parent_token_ids = {p.token_id for p in lineage.parent_tokens}
        expected_parent_ids = {token_a.token_id, token_b.token_id}
        assert parent_token_ids == expected_parent_ids, (
            f"parent_tokens should be the forked children. Expected {expected_parent_ids}, got {parent_token_ids}"
        )

        # Verify we can trace further back from parent tokens to the original fork
        for parent in lineage.parent_tokens:
            # Get the parent's parent (should be the original pre-fork token)
            grandparents = recorder.get_token_parents(parent.token_id)
            assert len(grandparents) == 1, f"Forked child should have exactly 1 parent, got {len(grandparents)}"
            assert grandparents[0].parent_token_id == parent_token.token_id, "Forked child's parent should be the original token"

        # Verify node_states show the path through coalesce and sink
        state_node_ids = {s.node_id for s in lineage.node_states}
        assert coalesce_node.node_id in state_node_ids, "node_states should include coalesce"
        assert sink_node.node_id in state_node_ids, "node_states should include sink"


class TestErrorRecovery:
    """Tests for error handling and recovery scenarios.

    These tests verify that:
    1. Pipelines can complete with partial success (some rows fail, others succeed)
    2. Failed/quarantined rows have complete audit trails
    3. The orchestrator correctly tracks failed vs quarantined counts

    Key distinction:
    - FAILED: Transform raised exception, retries exhausted -> rows_failed
    - QUARANTINED: Transform returned TransformResult.error() with _on_error="discard"
      -> rows_quarantined

    For transforms that return TransformResult.error(), they MUST have _on_error
    configured (otherwise RuntimeError is raised). Setting _on_error="discard"
    causes the row to be quarantined.
    """

    def test_partial_success_continues_processing(self) -> None:
        """Some rows fail via TransformResult.error(), others succeed.

        Pipeline should:
        - Process all rows
        - Quarantine failures (via _on_error="discard")
        - Complete successfully with partial results
        - Sink only receives successful rows
        """

        from elspeth.contracts import PluginSchema, RunStatus
        from elspeth.core.landscape import LandscapeDB
        from elspeth.engine.artifacts import ArtifactDescriptor
        from elspeth.engine.orchestrator import Orchestrator, PipelineConfig
        from elspeth.plugins.base import BaseTransform
        from elspeth.plugins.results import TransformResult

        db = LandscapeDB.in_memory()

        class ValueSchema(PluginSchema):
            value: int

        class ListSource(_TestSourceBase):
            """Source that yields rows with integer values."""

            name = "list_source"
            output_schema = ValueSchema

            def __init__(self, data: list[dict[str, Any]]) -> None:
                self._data = data

            def on_start(self, ctx: Any) -> None:
                pass

            def on_complete(self, ctx: Any) -> None:
                pass

            def load(self, ctx: Any) -> Any:
                for _row in self._data:
                    yield SourceRow.valid(_row)

            def close(self) -> None:
                pass

        class SelectiveFailTransform(BaseTransform):
            """Fails on even values via TransformResult.error(), succeeds on odd.

            Has _on_error="discard" so errors become QUARANTINED.
            """

            name = "selective_fail"
            input_schema = ValueSchema
            output_schema = ValueSchema
            _on_error = "discard"  # Required for TransformResult.error() to work

            def __init__(self) -> None:
                super().__init__({})

            def process(self, row: dict[str, Any], ctx: Any) -> TransformResult:
                if row["value"] % 2 == 0:
                    return TransformResult.error({"message": "Even values fail", "value": row["value"]})
                return TransformResult.success(row)

        class CollectSink(_TestSinkBase):
            """Sink that collects rows in memory."""

            name = "output_sink"

            def __init__(self) -> None:
                self.results: list[dict[str, Any]] = []

            def on_start(self, ctx: Any) -> None:
                pass

            def on_complete(self, ctx: Any) -> None:
                pass

            def write(self, rows: Any, ctx: Any) -> ArtifactDescriptor:
                self.results.extend(rows)
                return ArtifactDescriptor.for_file(
                    path="memory://output",
                    size_bytes=100,
                    content_hash="partial_success_test",
                )

            def close(self) -> None:
                pass

        # Create 10 rows: values 0-9
        # Even values (0, 2, 4, 6, 8) will fail -> 5 quarantined
        # Odd values (1, 3, 5, 7, 9) will succeed -> 5 succeeded
        source = ListSource([{"value": i} for i in range(10)])
        transform = SelectiveFailTransform()
        sink = CollectSink()

        config = PipelineConfig(
            source=as_source(source),
            transforms=[transform],
            sinks={"default": sink},
        )

        orchestrator = Orchestrator(db)
        result = orchestrator.run(config, graph=_build_test_graph(config))

        # Pipeline completes despite failures
        assert result.status == RunStatus.COMPLETED, f"Expected COMPLETED, got {result.status}"
        assert result.rows_processed == 10, f"Expected 10 rows processed, got {result.rows_processed}"
        assert result.rows_succeeded == 5, f"Expected 5 succeeded (odd values), got {result.rows_succeeded}"
        assert result.rows_quarantined == 5, f"Expected 5 quarantined (even values), got {result.rows_quarantined}"
        assert result.rows_failed == 0, f"Expected 0 failed (errors are quarantined), got {result.rows_failed}"

        # Sink received only successful rows (odd values)
        assert len(sink.results) == 5, f"Expected 5 results in sink, got {len(sink.results)}"
        result_values = {r["value"] for r in sink.results}
        expected_values = {1, 3, 5, 7, 9}
        assert result_values == expected_values, f"Expected odd values {expected_values}, got {result_values}"

    def test_quarantined_rows_have_audit_trail(self) -> None:
        """Quarantined rows must have complete audit trail.

        Even failed rows must be traceable - we need to know:
        - What source row failed (row_id linkage)
        - At which transform (node_id in node_state)
        - What the error was (error_json in node_state)
        """
        import json

        from sqlalchemy import select

        from elspeth.contracts import PluginSchema, RunStatus
        from elspeth.core.landscape import LandscapeDB, LandscapeRecorder
        from elspeth.core.landscape.schema import node_states_table
        from elspeth.engine.artifacts import ArtifactDescriptor
        from elspeth.engine.orchestrator import Orchestrator, PipelineConfig
        from elspeth.plugins.base import BaseTransform
        from elspeth.plugins.results import TransformResult

        db = LandscapeDB.in_memory()

        class ValueSchema(PluginSchema):
            value: int

        class ListSource(_TestSourceBase):
            """Source that yields rows with integer values."""

            name = "list_source"
            output_schema = ValueSchema

            def __init__(self, data: list[dict[str, Any]]) -> None:
                self._data = data

            def on_start(self, ctx: Any) -> None:
                pass

            def on_complete(self, ctx: Any) -> None:
                pass

            def load(self, ctx: Any) -> Any:
                for _row in self._data:
                    yield SourceRow.valid(_row)

            def close(self) -> None:
                pass

        class SelectiveFailTransform(BaseTransform):
            """Fails on even values via TransformResult.error(), succeeds on odd.

            Has _on_error="discard" so errors become QUARANTINED.
            """

            name = "selective_fail"
            input_schema = ValueSchema
            output_schema = ValueSchema
            _on_error = "discard"

            def __init__(self) -> None:
                super().__init__({})

            def process(self, row: dict[str, Any], ctx: Any) -> TransformResult:
                if row["value"] % 2 == 0:
                    return TransformResult.error({"message": "Even values fail", "value": row["value"]})
                return TransformResult.success(row)

        class CollectSink(_TestSinkBase):
            """Sink that collects rows in memory."""

            name = "output_sink"

            def __init__(self) -> None:
                self.results: list[dict[str, Any]] = []

            def on_start(self, ctx: Any) -> None:
                pass

            def on_complete(self, ctx: Any) -> None:
                pass

            def write(self, rows: Any, ctx: Any) -> ArtifactDescriptor:
                self.results.extend(rows)
                return ArtifactDescriptor.for_file(
                    path="memory://output",
                    size_bytes=100,
                    content_hash="audit_trail_test",
                )

            def close(self) -> None:
                pass

        # Create 4 rows: values 0-3
        # Even values (0, 2) will fail -> 2 quarantined with audit trail
        # Odd values (1, 3) will succeed -> 2 succeeded
        source = ListSource([{"value": i} for i in range(4)])
        transform = SelectiveFailTransform()
        sink = CollectSink()

        config = PipelineConfig(
            source=as_source(source),
            transforms=[transform],
            sinks={"default": sink},
        )

        orchestrator = Orchestrator(db)
        result = orchestrator.run(config, graph=_build_test_graph(config))

        # Basic verification
        assert result.status == RunStatus.COMPLETED
        assert result.rows_quarantined == 2

        # Now verify audit trail for quarantined rows
        recorder = LandscapeRecorder(db)
        nodes = recorder.get_nodes(result.run_id)
        transform_node = next(n for n in nodes if n.plugin_name == "selective_fail")

        # Query node_states for the transform
        with db.engine.connect() as conn:
            stmt = (
                select(node_states_table)
                .where(node_states_table.c.node_id == transform_node.node_id)
                .order_by(node_states_table.c.started_at)
            )
            states = list(conn.execute(stmt))

        # Should have 4 node_states: 2 failed (quarantined) + 2 completed (succeeded)
        assert len(states) == 4, f"Expected 4 node_states, got {len(states)}"

        # Count by status
        failed_states = [s for s in states if s.status == "failed"]
        completed_states = [s for s in states if s.status == "completed"]

        assert len(failed_states) == 2, f"Expected 2 failed states (quarantined rows), got {len(failed_states)}"
        assert len(completed_states) == 2, f"Expected 2 completed states (succeeded rows), got {len(completed_states)}"

        # Verify each failed state has complete audit trail
        for failed_state in failed_states:
            # Must have error_json populated
            assert failed_state.error_json is not None, f"Failed state for token {failed_state.token_id} missing error_json"

            # error_json should be valid JSON with the error details we provided
            error_data = json.loads(failed_state.error_json)
            assert "message" in error_data, f"error_json missing 'message' field: {error_data}"
            assert error_data["message"] == "Even values fail", f"Unexpected error message: {error_data['message']}"
            assert "value" in error_data, f"error_json missing 'value' field: {error_data}"
            # The value should be even (0 or 2)
            assert error_data["value"] % 2 == 0, f"Expected even value in error, got {error_data['value']}"

            # Must identify which node (transform) failed
            assert failed_state.node_id == transform_node.node_id, (
                f"Failed state node_id mismatch: {failed_state.node_id} != {transform_node.node_id}"
            )

            # Must have token_id linking back to the row
            assert failed_state.token_id is not None, "Failed state missing token_id"

            # Must have duration_ms (processing was attempted)
            assert failed_state.duration_ms is not None, "Failed state missing duration_ms"
