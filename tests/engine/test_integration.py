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

from elspeth.contracts import Determinism, PluginSchema, RoutingMode
from elspeth.core.config import GateSettings
from elspeth.plugins.base import BaseTransform

if TYPE_CHECKING:
    from elspeth.contracts.results import ArtifactDescriptor, TransformResult
    from elspeth.core.dag import ExecutionGraph
    from elspeth.engine.orchestrator import PipelineConfig


# ============================================================================
# Test Fixture Base Classes
# ============================================================================
# These provide the required protocol attributes so inline test classes
# don't need to repeat them.


class _TestSchema(PluginSchema):
    """Minimal schema for test fixtures."""

    pass


class _TestSourceBase:
    """Base class providing SourceProtocol required attributes.

    Note: output_schema is NOT provided here because child classes override it
    with their own schemas, and mypy's type invariance would flag that as a conflict.
    Each test class must provide its own output_schema.
    """

    node_id: str | None = None
    determinism = Determinism.DETERMINISTIC
    plugin_version = "1.0.0"


class _TestSinkBase:
    """Base class providing SinkProtocol required attributes."""

    input_schema = _TestSchema  # Required by protocol
    idempotent = True
    node_id: str | None = None
    determinism = Determinism.DETERMINISTIC
    plugin_version = "1.0"

    def flush(self) -> None:
        pass


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
                graph.add_edge(
                    node_id, sink_ids[target], label=route_label, mode=RoutingMode.MOVE
                )

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
                yield from self._data

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
                return ArtifactDescriptor.for_file(
                    path="memory://output", size_bytes=100, content_hash="abc123"
                )

            def close(self) -> None:
                pass

        # Run pipeline
        source = ListSource([{"value": 1}, {"value": 2}, {"value": 3}])
        transform = MarkProcessedTransform()
        sink = CollectSink()

        config = PipelineConfig(
            source=source,
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
                yield from self._data

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
                return ArtifactDescriptor.for_file(
                    path="memory://out", size_bytes=len(rows), content_hash="hash"
                )

            def close(self) -> None:
                pass

        # Pipeline with multiple transforms
        source = ListSource([{"n": 1}, {"n": 2}, {"n": 3}, {"n": 4}, {"n": 5}])
        t1 = DoubleTransform()
        t2 = AddTenTransform()
        sink = CollectSink()

        config = PipelineConfig(
            source=source,
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
        transform_node_ids = {
            n.node_id
            for n in nodes
            if n.node_type in (NodeType.TRANSFORM.value, "transform")
        }
        sink_node_ids = {
            n.node_id for n in nodes if n.node_type in (NodeType.SINK.value, "sink")
        }

        # Get all rows
        rows = recorder.get_rows(result.run_id)
        assert len(rows) == 5, "All source rows must be recorded"

        for row in rows:
            # Every row must have at least one token
            tokens = recorder.get_tokens(row.row_id)
            assert (
                len(tokens) >= 1
            ), f"Row {row.row_id} has no tokens - audit spine broken"

            for token in tokens:
                # Every token must have node_states
                states = recorder.get_node_states_for_token(token.token_id)
                assert (
                    len(states) > 0
                ), f"Token {token.token_id} has no node_states - audit spine broken"

                # Verify token has states at BOTH transforms
                state_node_ids = {s.node_id for s in states}
                for transform_id in transform_node_ids:
                    assert (
                        transform_id in state_node_ids
                    ), f"Token {token.token_id} missing state at transform {transform_id}"

                # Verify token has state at sink
                sink_states = [s for s in states if s.node_id in sink_node_ids]
                assert (
                    len(sink_states) >= 1
                ), f"Token {token.token_id} never reached a sink - audit spine broken"

                # All states must be completed
                for state in states:
                    assert (
                        state.status == "completed"
                    ), f"Token {token.token_id} has non-completed state: {state.status}"

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
                yield from self._data

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
            source=source,
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

        sink_node_ids = {
            n.node_id for n in nodes if n.node_type in (NodeType.SINK.value, "sink")
        }

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
                assert (
                    len(sink_states) >= 1
                ), f"Token {token.token_id} never reached sink"

        # 2 tokens were routed (even numbers)
        assert routed_count == 2


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
                yield from self._data

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
                return ArtifactDescriptor.for_file(
                    path="memory", size_bytes=0, content_hash=""
                )

            def close(self) -> None:
                pass

        # Config-driven gate that always routes to "phantom" (nonexistent sink)
        misrouting_gate = GateSettings(
            name="misrouting_gate",
            condition="True",  # Always routes
            routes={"true": "phantom"},  # Route to nonexistent sink
        )

        source = ListSource([{"value": 42}])
        sink = CollectSink()

        config = PipelineConfig(
            source=source,
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
                yield from self._data

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
                return ArtifactDescriptor.for_file(
                    path="memory", size_bytes=0, content_hash=""
                )

            def close(self) -> None:
                pass

        source = ListSource([{"value": 1}])
        transform = ExplodingTransform()
        sink = CollectSink()

        config = PipelineConfig(
            source=source,
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
                yield from self._data

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
            source=source,
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
                return ArtifactDescriptor.for_file(
                    path="memory", size_bytes=0, content_hash=""
                )

            def close(self) -> None:
                pass

        source = EmptySource()
        transform = IdentityTransform()
        sink = CollectSink()

        config = PipelineConfig(
            source=source,
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
                yield from self._data

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

        source = ListSource(
            [{"value": 10}, {"value": 60}, {"value": 30}, {"value": 90}]
        )
        default_sink = CollectSink("default_output")
        high_sink = CollectSink("high_output")

        config = PipelineConfig(
            source=source,
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
            routes={"true": "fork"},
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
        completed_count = sum(
            1 for r in all_results if r.outcome == RowOutcome.COMPLETED
        )
        forked_count = sum(1 for r in all_results if r.outcome == RowOutcome.FORKED)

        # Verify:
        # - 2 parent tokens with FORKED outcome
        # - 4 child tokens with COMPLETED outcome
        assert forked_count == 2, f"Expected 2 FORKED, got {forked_count}"
        assert completed_count == 4, f"Expected 4 COMPLETED, got {completed_count}"

        # Collect the COMPLETED tokens (these are what would go to sink)
        completed_tokens = [
            r.token for r in all_results if r.outcome == RowOutcome.COMPLETED
        ]
        assert len(completed_tokens) == 4

        # Verify correct values: each source value appears twice (once per fork path)
        values = [t.row_data["value"] for t in completed_tokens]
        assert (
            values.count(1) == 2
        ), f"Expected value 1 to appear 2 times, got {values.count(1)}"
        assert (
            values.count(2) == 2
        ), f"Expected value 2 to appear 2 times, got {values.count(2)}"

        # Verify audit trail completeness
        rows = recorder.get_rows(run_id)
        assert len(rows) == 2, f"Expected 2 source rows, got {len(rows)}"

        # Verify tokens: 2 parent + 4 children = 6 total
        total_tokens = 0
        for row in rows:
            tokens = recorder.get_tokens(row.row_id)
            total_tokens += len(tokens)
        assert (
            total_tokens == 6
        ), f"Expected 6 tokens (2 parents + 4 children), got {total_tokens}"

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
        assert (
            routing_event_count == 4
        ), f"Expected 4 routing events, got {routing_event_count}"

        # Complete the run
        recorder.complete_run(run_id, status="completed")


class TestAggregationIntegration:
    """Integration tests for aggregation through full pipeline."""

    def test_aggregation_output_mode_single(self) -> None:
        """output_mode=single: batch produces one aggregated result.

        Pipeline: source (3 rows) -> aggregation (count=3) -> sink
        Result: sink receives 1 row with aggregated data

        This test uses lower-level components (AggregationExecutor, SinkExecutor)
        to verify output_mode=single behavior. PipelineConfig aggregation support
        is WIP - this test validates the underlying engine behavior.
        """
        from elspeth.contracts import NodeType, PluginSchema, TokenInfo
        from elspeth.contracts.schema import SchemaConfig
        from elspeth.core.config import AggregationSettings, TriggerConfig
        from elspeth.core.landscape import LandscapeDB, LandscapeRecorder
        from elspeth.engine.artifacts import ArtifactDescriptor
        from elspeth.engine.executors import AggregationExecutor, SinkExecutor
        from elspeth.engine.spans import SpanFactory
        from elspeth.engine.tokens import TokenManager
        from elspeth.plugins.base import BaseAggregation
        from elspeth.plugins.context import PluginContext
        from elspeth.plugins.results import AcceptResult

        db = LandscapeDB.in_memory()
        recorder = LandscapeRecorder(db)

        # Start a run
        run = recorder.begin_run(config={}, canonical_version="v1")
        run_id = run.run_id

        DYNAMIC_SCHEMA = SchemaConfig.from_dict({"fields": "dynamic"})

        class RowSchema(PluginSchema):
            value: int

        # Register nodes
        source_node = recorder.register_node(
            run_id=run_id,
            node_id="source",
            plugin_name="list_source",
            node_type=NodeType.SOURCE,
            plugin_version="1.0.0",
            config={},
            schema_config=DYNAMIC_SCHEMA,
        )

        agg_node = recorder.register_node(
            run_id=run_id,
            node_id="sum_agg",
            plugin_name="sum_agg",
            node_type=NodeType.AGGREGATION,
            plugin_version="1.0.0",
            config={},
            schema_config=DYNAMIC_SCHEMA,
        )

        sink_node = recorder.register_node(
            run_id=run_id,
            node_id="output_sink",
            plugin_name="collect_sink",
            node_type=NodeType.SINK,
            plugin_version="1.0.0",
            config={},
            schema_config=DYNAMIC_SCHEMA,
        )

        class SumAggregation(BaseAggregation):
            """Aggregation that sums values."""

            name = "sum_agg"
            input_schema = RowSchema
            output_schema = RowSchema
            plugin_version = "1.0.0"

            def __init__(self) -> None:
                super().__init__({})
                self._values: list[int] = []

            def accept(self, row: dict[str, Any], ctx: PluginContext) -> AcceptResult:
                self._values.append(row["value"])
                return AcceptResult(accepted=True)

            def flush(self, ctx: PluginContext) -> list[dict[str, Any]]:
                total = sum(self._values)
                count = len(self._values)
                self._values = []
                # output_mode=single means we return exactly ONE aggregated result
                return [{"value": total, "count": count}]

        class CollectSink(_TestSinkBase):
            name = "collect"

            def __init__(self) -> None:
                self.results: list[dict[str, Any]] = []

            def on_start(self, ctx: Any) -> None:
                pass

            def on_complete(self, ctx: Any) -> None:
                pass

            def write(self, rows: Any, ctx: Any) -> ArtifactDescriptor:
                self.results.extend(rows)
                return ArtifactDescriptor.for_file(
                    path="memory://output", size_bytes=0, content_hash="test_hash"
                )

            def close(self) -> None:
                pass

        # Configure aggregation: flush after 3 rows, output_mode=single
        agg_settings = AggregationSettings(
            name="sum_agg",
            plugin="sum_agg",
            trigger=TriggerConfig(count=3),
            output_mode="single",
        )

        # Create aggregation instance and assign node_id
        aggregation = SumAggregation()
        aggregation.node_id = agg_node.node_id

        # Create sink instance and assign node_id
        sink = CollectSink()
        sink.node_id = sink_node.node_id

        # Create executors
        span_factory = SpanFactory()
        token_manager = TokenManager(recorder)

        agg_executor = AggregationExecutor(
            recorder=recorder,
            span_factory=span_factory,
            run_id=run_id,
            aggregation_settings={agg_node.node_id: agg_settings},
        )

        sink_executor = SinkExecutor(recorder, span_factory, run_id)

        ctx = PluginContext(run_id=run_id, config={})

        # Process 3 source rows through aggregation
        source_rows = [{"value": 10}, {"value": 20}, {"value": 30}]

        for row_index, row_data in enumerate(source_rows):
            # Create token for each source row
            token = token_manager.create_initial_token(
                run_id=run_id,
                source_node_id=source_node.node_id,
                row_index=row_index,
                row_data=row_data,
            )

            # Accept row into aggregation
            result = agg_executor.accept(
                aggregation=aggregation,
                token=token,
                ctx=ctx,
                step_in_pipeline=1,
            )
            assert result.accepted is True

        # Verify trigger condition is met after 3 rows
        assert agg_executor.should_flush(agg_node.node_id) is True

        # Flush the batch
        output_rows = agg_executor.flush(
            aggregation=aggregation,
            ctx=ctx,
            trigger_reason="count_reached",
            step_in_pipeline=2,
        )

        # output_mode=single: batch produces ONE aggregated result
        assert len(output_rows) == 1
        assert output_rows[0]["value"] == 60  # 10 + 20 + 30
        assert output_rows[0]["count"] == 3

        # Create token for aggregation output and write to sink
        # For aggregation outputs, we create a synthetic token
        agg_output_token = TokenInfo(
            row_id="agg_output_0",
            token_id="agg_token_0",
            row_data=output_rows[0],
        )

        # Write aggregated result to sink
        sink_executor.write(
            sink=sink,
            tokens=[agg_output_token],
            ctx=ctx,
            step_in_pipeline=3,
        )

        # CRITICAL VERIFICATION: Sink received 1 row (aggregated), not 3 (source rows)
        assert len(sink.results) == 1, (
            f"Expected 1 aggregated row, got {len(sink.results)}. "
            "output_mode=single should produce exactly one result."
        )
        assert sink.results[0]["value"] == 60
        assert sink.results[0]["count"] == 3

        # Complete the run
        recorder.complete_run(run_id, status="completed")

        # Verify audit trail completeness
        # - 3 source rows recorded
        # - Each row has token with node_state at aggregation
        # - Aggregation batch exists with 3 members
        rows = recorder.get_rows(run_id)
        assert len(rows) == 3, f"Expected 3 source rows, got {len(rows)}"

        # Verify batch was created and completed
        batches = recorder.get_batches(run_id)
        assert len(batches) == 1
        assert batches[0].status == "completed"

        # Verify batch has 3 members
        members = recorder.get_batch_members(batches[0].batch_id)
        assert len(members) == 3

        # Verify artifact was recorded
        artifacts = recorder.get_artifacts(run_id)
        assert len(artifacts) == 1
        assert artifacts[0].content_hash == "test_hash"

    def test_aggregation_output_mode_passthrough(self) -> None:
        """output_mode=passthrough: batch releases all rows unchanged.

        Pipeline: source (5 rows) -> aggregation (count=3) -> sink
        Result: sink receives all 5 rows (3 from first batch, 2 from end_of_source)

        This test verifies that passthrough mode:
        - Buffers rows until trigger condition
        - Releases all buffered rows unchanged (no aggregation)
        - end_of_source flushes remaining partial batch
        """
        from elspeth.contracts import NodeType, PluginSchema, TokenInfo
        from elspeth.contracts.schema import SchemaConfig
        from elspeth.core.config import AggregationSettings, TriggerConfig
        from elspeth.core.landscape import LandscapeDB, LandscapeRecorder
        from elspeth.engine.artifacts import ArtifactDescriptor
        from elspeth.engine.executors import AggregationExecutor, SinkExecutor
        from elspeth.engine.spans import SpanFactory
        from elspeth.engine.tokens import TokenManager
        from elspeth.plugins.base import BaseAggregation
        from elspeth.plugins.context import PluginContext
        from elspeth.plugins.results import AcceptResult

        db = LandscapeDB.in_memory()
        recorder = LandscapeRecorder(db)

        # Start a run
        run = recorder.begin_run(config={}, canonical_version="v1")
        run_id = run.run_id

        DYNAMIC_SCHEMA = SchemaConfig.from_dict({"fields": "dynamic"})

        class RowSchema(PluginSchema):
            value: int

        # Register nodes
        source_node = recorder.register_node(
            run_id=run_id,
            node_id="source",
            plugin_name="list_source",
            node_type=NodeType.SOURCE,
            plugin_version="1.0.0",
            config={},
            schema_config=DYNAMIC_SCHEMA,
        )

        agg_node = recorder.register_node(
            run_id=run_id,
            node_id="buffer_agg",
            plugin_name="buffer_agg",
            node_type=NodeType.AGGREGATION,
            plugin_version="1.0.0",
            config={},
            schema_config=DYNAMIC_SCHEMA,
        )

        sink_node = recorder.register_node(
            run_id=run_id,
            node_id="output_sink",
            plugin_name="collect_sink",
            node_type=NodeType.SINK,
            plugin_version="1.0.0",
            config={},
            schema_config=DYNAMIC_SCHEMA,
        )

        class BufferAggregation(BaseAggregation):
            """Aggregation that buffers and releases rows unchanged."""

            name = "buffer_agg"
            input_schema = RowSchema
            output_schema = RowSchema
            plugin_version = "1.0.0"

            def __init__(self) -> None:
                super().__init__({})
                self._buffer: list[dict[str, Any]] = []

            def accept(self, row: dict[str, Any], ctx: PluginContext) -> AcceptResult:
                self._buffer.append(row)
                return AcceptResult(accepted=True)

            def flush(self, ctx: PluginContext) -> list[dict[str, Any]]:
                # Passthrough mode: return all buffered rows unchanged
                result = list(self._buffer)
                self._buffer = []
                return result

        class CollectSink(_TestSinkBase):
            name = "collect"

            def __init__(self) -> None:
                self.results: list[dict[str, Any]] = []

            def on_start(self, ctx: Any) -> None:
                pass

            def on_complete(self, ctx: Any) -> None:
                pass

            def write(self, rows: Any, ctx: Any) -> ArtifactDescriptor:
                self.results.extend(rows)
                return ArtifactDescriptor.for_file(
                    path="memory://output", size_bytes=0, content_hash="test_hash"
                )

            def close(self) -> None:
                pass

        # Configure aggregation: flush after 3 rows, output_mode=passthrough
        agg_settings = AggregationSettings(
            name="buffer_agg",
            plugin="buffer_agg",
            trigger=TriggerConfig(count=3),
            output_mode="passthrough",
        )

        # Create aggregation instance and assign node_id
        aggregation = BufferAggregation()
        aggregation.node_id = agg_node.node_id

        # Create sink instance and assign node_id
        sink = CollectSink()
        sink.node_id = sink_node.node_id

        # Create executors
        span_factory = SpanFactory()
        token_manager = TokenManager(recorder)

        agg_executor = AggregationExecutor(
            recorder=recorder,
            span_factory=span_factory,
            run_id=run_id,
            aggregation_settings={agg_node.node_id: agg_settings},
        )

        sink_executor = SinkExecutor(recorder, span_factory, run_id)

        ctx = PluginContext(run_id=run_id, config={})

        # Source with 5 rows to test partial batch at end_of_source
        source_rows = [{"value": i} for i in range(1, 6)]  # 1, 2, 3, 4, 5

        # Track all tokens and output rows
        all_tokens: list[TokenInfo] = []
        all_output_rows: list[dict[str, Any]] = []

        for row_index, row_data in enumerate(source_rows):
            # Create token for each source row
            token = token_manager.create_initial_token(
                run_id=run_id,
                source_node_id=source_node.node_id,
                row_index=row_index,
                row_data=row_data,
            )
            all_tokens.append(token)

            # Accept row into aggregation
            result = agg_executor.accept(
                aggregation=aggregation,
                token=token,
                ctx=ctx,
                step_in_pipeline=1,
            )
            assert result.accepted is True

            # Check if trigger condition is met (after 3 rows)
            if agg_executor.should_flush(agg_node.node_id):
                # Flush the batch
                output_rows = agg_executor.flush(
                    aggregation=aggregation,
                    ctx=ctx,
                    trigger_reason="count_reached",
                    step_in_pipeline=2,
                )
                all_output_rows.extend(output_rows)

        # After processing all 5 rows:
        # - First batch (3 rows) was flushed when count reached
        # - 2 rows remain in buffer

        # Simulate end_of_source: flush remaining rows
        remaining_rows = agg_executor.flush(
            aggregation=aggregation,
            ctx=ctx,
            trigger_reason="end_of_source",
            step_in_pipeline=2,
        )
        all_output_rows.extend(remaining_rows)

        # output_mode=passthrough: all 5 rows should be output unchanged
        assert len(all_output_rows) == 5, (
            f"Expected 5 rows (passthrough), got {len(all_output_rows)}. "
            "Passthrough mode should release all buffered rows unchanged."
        )

        # Verify order: first batch (1,2,3), then end_of_source flush (4,5)
        expected_values = [1, 2, 3, 4, 5]
        actual_values = [r["value"] for r in all_output_rows]
        assert actual_values == expected_values, (
            f"Expected values {expected_values}, got {actual_values}. "
            "Rows should maintain original order through passthrough."
        )

        # Create tokens for output rows and write to sink
        for idx, output_row in enumerate(all_output_rows):
            output_token = TokenInfo(
                row_id=f"passthrough_output_{idx}",
                token_id=f"passthrough_token_{idx}",
                row_data=output_row,
            )
            sink_executor.write(
                sink=sink,
                tokens=[output_token],
                ctx=ctx,
                step_in_pipeline=3,
            )

        # CRITICAL VERIFICATION: Sink received all 5 rows unchanged
        assert len(sink.results) == 5, (
            f"Expected 5 rows in sink, got {len(sink.results)}. "
            "Passthrough mode should pass all rows to sink unchanged."
        )

        # Verify values are unchanged
        sink_values = [r["value"] for r in sink.results]
        assert sink_values == [1, 2, 3, 4, 5], (
            f"Expected [1, 2, 3, 4, 5], got {sink_values}. "
            "Passthrough should not modify row values."
        )

        # Complete the run
        recorder.complete_run(run_id, status="completed")

        # Verify audit trail completeness
        rows = recorder.get_rows(run_id)
        assert len(rows) == 5, f"Expected 5 source rows, got {len(rows)}"

        # Verify batches were created:
        # - 1 batch for first 3 rows (count trigger)
        # - 1 batch for remaining 2 rows (end_of_source)
        batches = recorder.get_batches(run_id)
        assert len(batches) == 2, f"Expected 2 batches, got {len(batches)}"

        # Verify batch member counts
        batch_sizes = []
        for batch in batches:
            members = recorder.get_batch_members(batch.batch_id)
            batch_sizes.append(len(members))

        # First batch has 3 members, second has 2
        assert sorted(batch_sizes) == [2, 3], (
            f"Expected batch sizes [2, 3], got {sorted(batch_sizes)}. "
            "First batch should have 3 rows, end_of_source batch should have 2."
        )

    def test_aggregation_flushes_on_source_exhaustion(self) -> None:
        """Aggregation flushes remaining rows when source exhausts.

        Pipeline: source (5 rows) -> aggregation (count=100) -> sink
        - count=100 never triggers (only 5 rows)
        - end_of_source implicit trigger flushes the 5 accumulated rows

        This is critical for not losing data: even when the explicit trigger
        (count=100) never fires, the aggregation MUST flush at end_of_source.
        """
        from elspeth.contracts import NodeType, PluginSchema, TokenInfo
        from elspeth.contracts.schema import SchemaConfig
        from elspeth.core.config import AggregationSettings, TriggerConfig
        from elspeth.core.landscape import LandscapeDB, LandscapeRecorder
        from elspeth.engine.artifacts import ArtifactDescriptor
        from elspeth.engine.executors import AggregationExecutor, SinkExecutor
        from elspeth.engine.spans import SpanFactory
        from elspeth.engine.tokens import TokenManager
        from elspeth.plugins.base import BaseAggregation
        from elspeth.plugins.context import PluginContext
        from elspeth.plugins.results import AcceptResult

        db = LandscapeDB.in_memory()
        recorder = LandscapeRecorder(db)

        # Start a run
        run = recorder.begin_run(config={}, canonical_version="v1")
        run_id = run.run_id

        DYNAMIC_SCHEMA = SchemaConfig.from_dict({"fields": "dynamic"})

        class RowSchema(PluginSchema):
            value: int

        # Register nodes
        source_node = recorder.register_node(
            run_id=run_id,
            node_id="source",
            plugin_name="list_source",
            node_type=NodeType.SOURCE,
            plugin_version="1.0.0",
            config={},
            schema_config=DYNAMIC_SCHEMA,
        )

        agg_node = recorder.register_node(
            run_id=run_id,
            node_id="buffer_agg",
            plugin_name="buffer_agg",
            node_type=NodeType.AGGREGATION,
            plugin_version="1.0.0",
            config={},
            schema_config=DYNAMIC_SCHEMA,
        )

        sink_node = recorder.register_node(
            run_id=run_id,
            node_id="output_sink",
            plugin_name="collect_sink",
            node_type=NodeType.SINK,
            plugin_version="1.0.0",
            config={},
            schema_config=DYNAMIC_SCHEMA,
        )

        class BufferAggregation(BaseAggregation):
            """Aggregation that buffers and releases rows unchanged."""

            name = "buffer_agg"
            input_schema = RowSchema
            output_schema = RowSchema
            plugin_version = "1.0.0"

            def __init__(self) -> None:
                super().__init__({})
                self._buffer: list[dict[str, Any]] = []

            def accept(self, row: dict[str, Any], ctx: PluginContext) -> AcceptResult:
                self._buffer.append(row)
                return AcceptResult(accepted=True)

            def flush(self, ctx: PluginContext) -> list[dict[str, Any]]:
                result = list(self._buffer)
                self._buffer = []
                return result

        class CollectSink(_TestSinkBase):
            name = "collect"

            def __init__(self) -> None:
                self.results: list[dict[str, Any]] = []

            def on_start(self, ctx: Any) -> None:
                pass

            def on_complete(self, ctx: Any) -> None:
                pass

            def write(self, rows: Any, ctx: Any) -> ArtifactDescriptor:
                self.results.extend(rows)
                return ArtifactDescriptor.for_file(
                    path="memory://output", size_bytes=0, content_hash="test_hash"
                )

            def close(self) -> None:
                pass

        # CRITICAL: Configure aggregation with count=100 trigger
        # This will NEVER fire for only 5 rows - we rely on end_of_source
        agg_settings = AggregationSettings(
            name="buffer_agg",
            plugin="buffer_agg",
            trigger=TriggerConfig(count=100),  # Will never fire!
            output_mode="passthrough",
        )

        # Create aggregation instance and assign node_id
        aggregation = BufferAggregation()
        aggregation.node_id = agg_node.node_id

        # Create sink instance and assign node_id
        sink = CollectSink()
        sink.node_id = sink_node.node_id

        # Create executors
        span_factory = SpanFactory()
        token_manager = TokenManager(recorder)

        agg_executor = AggregationExecutor(
            recorder=recorder,
            span_factory=span_factory,
            run_id=run_id,
            aggregation_settings={agg_node.node_id: agg_settings},
        )

        sink_executor = SinkExecutor(recorder, span_factory, run_id)

        ctx = PluginContext(run_id=run_id, config={})

        # Source provides only 5 rows
        source_rows = [{"value": i} for i in range(1, 6)]  # 1, 2, 3, 4, 5

        # Process all 5 rows - count trigger will NEVER fire (need 100)
        for row_index, row_data in enumerate(source_rows):
            token = token_manager.create_initial_token(
                run_id=run_id,
                source_node_id=source_node.node_id,
                row_index=row_index,
                row_data=row_data,
            )

            result = agg_executor.accept(
                aggregation=aggregation,
                token=token,
                ctx=ctx,
                step_in_pipeline=1,
            )
            assert result.accepted is True

            # Verify count trigger has NOT fired (only 5 rows, need 100)
            assert (
                agg_executor.should_flush(agg_node.node_id) is False
            ), f"Count trigger should not fire with only {row_index + 1} rows (need 100)"

        # At this point:
        # - 5 rows have been accepted
        # - count=100 trigger has NOT fired
        # - Without end_of_source flush, data would be LOST

        # Simulate SOURCE EXHAUSTION: flush with end_of_source trigger
        # This is the implicit trigger that always fires when source ends
        output_rows = agg_executor.flush(
            aggregation=aggregation,
            ctx=ctx,
            trigger_reason="end_of_source",  # The implicit trigger
            step_in_pipeline=2,
        )

        # CRITICAL VERIFICATION: All 5 rows were flushed
        assert len(output_rows) == 5, (
            f"Expected 5 rows from end_of_source flush, got {len(output_rows)}. "
            "end_of_source must flush ALL accumulated rows regardless of count trigger."
        )

        # Verify row values are preserved
        flushed_values = [r["value"] for r in output_rows]
        assert flushed_values == [
            1,
            2,
            3,
            4,
            5,
        ], f"Expected values [1, 2, 3, 4, 5], got {flushed_values}"

        # Write flushed rows to sink
        for idx, output_row in enumerate(output_rows):
            output_token = TokenInfo(
                row_id=f"eos_output_{idx}",
                token_id=f"eos_token_{idx}",
                row_data=output_row,
            )
            sink_executor.write(
                sink=sink,
                tokens=[output_token],
                ctx=ctx,
                step_in_pipeline=3,
            )

        # CRITICAL VERIFICATION: Sink received all 5 rows
        assert len(sink.results) == 5, (
            f"Expected 5 rows in sink, got {len(sink.results)}. "
            "All rows must reach sink via end_of_source flush."
        )

        # Verify values in sink
        sink_values = [r["value"] for r in sink.results]
        assert sink_values == [
            1,
            2,
            3,
            4,
            5,
        ], f"Expected [1, 2, 3, 4, 5] in sink, got {sink_values}"

        # Complete the run
        recorder.complete_run(run_id, status="completed")

        # Verify audit trail completeness
        rows = recorder.get_rows(run_id)
        assert len(rows) == 5, f"Expected 5 source rows, got {len(rows)}"

        # Verify exactly 1 batch was created (end_of_source flush)
        batches = recorder.get_batches(run_id)
        assert len(batches) == 1, (
            f"Expected 1 batch (from end_of_source), got {len(batches)}. "
            "Count trigger should never have fired."
        )

        # Verify the single batch has all 5 members
        members = recorder.get_batch_members(batches[0].batch_id)
        assert len(members) == 5, (
            f"Expected 5 batch members, got {len(members)}. "
            "All rows should be in the end_of_source batch."
        )

        # Verify batch was triggered by end_of_source
        assert (
            batches[0].trigger_reason == "end_of_source"
        ), f"Expected trigger_reason='end_of_source', got '{batches[0].trigger_reason}'"

    def test_aggregation_timeout_trigger(self) -> None:
        """Timeout trigger fires before count is reached.

        Pipeline: source (10 rows) -> slow_transform (20ms delay) -> aggregation -> sink
        - count=1000 (will never fire with only 10 rows)
        - timeout_seconds=0.05 (50ms, fires after ~2-3 rows with 20ms delay each)

        Verifies that timeout trigger fires, not count trigger.
        """
        import time

        from elspeth.contracts import NodeType, PluginSchema, TokenInfo
        from elspeth.contracts.enums import TriggerType
        from elspeth.contracts.schema import SchemaConfig
        from elspeth.core.config import AggregationSettings, TriggerConfig
        from elspeth.core.landscape import LandscapeDB, LandscapeRecorder
        from elspeth.engine.artifacts import ArtifactDescriptor
        from elspeth.engine.executors import AggregationExecutor, SinkExecutor
        from elspeth.engine.spans import SpanFactory
        from elspeth.engine.tokens import TokenManager
        from elspeth.plugins.base import BaseAggregation, BaseTransform
        from elspeth.plugins.context import PluginContext
        from elspeth.plugins.results import AcceptResult, TransformResult

        db = LandscapeDB.in_memory()
        recorder = LandscapeRecorder(db)

        # Start a run
        run = recorder.begin_run(config={}, canonical_version="v1")
        run_id = run.run_id

        DYNAMIC_SCHEMA = SchemaConfig.from_dict({"fields": "dynamic"})

        class RowSchema(PluginSchema):
            value: int

        # Register nodes
        source_node = recorder.register_node(
            run_id=run_id,
            node_id="source",
            plugin_name="list_source",
            node_type=NodeType.SOURCE,
            plugin_version="1.0.0",
            config={},
            schema_config=DYNAMIC_SCHEMA,
        )

        transform_node = recorder.register_node(
            run_id=run_id,
            node_id="slow_transform",
            plugin_name="slow_transform",
            node_type=NodeType.TRANSFORM,
            plugin_version="1.0.0",
            config={},
            schema_config=DYNAMIC_SCHEMA,
        )

        agg_node = recorder.register_node(
            run_id=run_id,
            node_id="buffer_agg",
            plugin_name="buffer_agg",
            node_type=NodeType.AGGREGATION,
            plugin_version="1.0.0",
            config={},
            schema_config=DYNAMIC_SCHEMA,
        )

        sink_node = recorder.register_node(
            run_id=run_id,
            node_id="output_sink",
            plugin_name="collect_sink",
            node_type=NodeType.SINK,
            plugin_version="1.0.0",
            config={},
            schema_config=DYNAMIC_SCHEMA,
        )

        class SlowTransform(BaseTransform):
            """Transform that adds 20ms delay per row."""

            name = "slow_transform"
            input_schema = RowSchema
            output_schema = RowSchema
            plugin_version = "1.0.0"

            def __init__(self) -> None:
                super().__init__({})

            def process(
                self, row: dict[str, Any], ctx: PluginContext
            ) -> TransformResult:
                time.sleep(0.02)  # 20ms delay
                return TransformResult.success(row)

        class BufferAggregation(BaseAggregation):
            """Aggregation that buffers and releases rows unchanged."""

            name = "buffer_agg"
            input_schema = RowSchema
            output_schema = RowSchema
            plugin_version = "1.0.0"

            def __init__(self) -> None:
                super().__init__({})
                self._buffer: list[dict[str, Any]] = []

            def accept(self, row: dict[str, Any], ctx: PluginContext) -> AcceptResult:
                self._buffer.append(row)
                return AcceptResult(accepted=True)

            def flush(self, ctx: PluginContext) -> list[dict[str, Any]]:
                result = list(self._buffer)
                self._buffer = []
                return result

        class CollectSink(_TestSinkBase):
            name = "collect"

            def __init__(self) -> None:
                self.results: list[dict[str, Any]] = []

            def on_start(self, ctx: Any) -> None:
                pass

            def on_complete(self, ctx: Any) -> None:
                pass

            def write(self, rows: Any, ctx: Any) -> ArtifactDescriptor:
                self.results.extend(rows)
                return ArtifactDescriptor.for_file(
                    path="memory://output", size_bytes=0, content_hash="test_hash"
                )

            def close(self) -> None:
                pass

        # Configure aggregation: count=1000 (won't fire), timeout=50ms (will fire)
        agg_settings = AggregationSettings(
            name="buffer_agg",
            plugin="buffer_agg",
            trigger=TriggerConfig(count=1000, timeout_seconds=0.05),
            output_mode="passthrough",
        )

        # Create instances and assign node_ids
        transform = SlowTransform()
        transform.node_id = transform_node.node_id

        aggregation = BufferAggregation()
        aggregation.node_id = agg_node.node_id

        sink = CollectSink()
        sink.node_id = sink_node.node_id

        # Create executors
        span_factory = SpanFactory()
        token_manager = TokenManager(recorder)

        agg_executor = AggregationExecutor(
            recorder=recorder,
            span_factory=span_factory,
            run_id=run_id,
            aggregation_settings={agg_node.node_id: agg_settings},
        )

        sink_executor = SinkExecutor(recorder, span_factory, run_id)

        ctx = PluginContext(run_id=run_id, config={})

        # Source with 10 rows
        source_rows = [{"value": i} for i in range(1, 11)]

        # Track when timeout triggers
        timeout_triggered = False
        rows_processed_before_timeout = 0

        for row_index, row_data in enumerate(source_rows):
            # Create token for each source row
            token = token_manager.create_initial_token(
                run_id=run_id,
                source_node_id=source_node.node_id,
                row_index=row_index,
                row_data=row_data,
            )

            # Process through slow transform (adds 20ms delay)
            transform_result = transform.process(row_data, ctx)
            transformed_data = transform_result.row

            # Update token with transformed data
            token = TokenInfo(
                row_id=token.row_id,
                token_id=token.token_id,
                row_data=transformed_data,
            )

            # Accept into aggregation
            result = agg_executor.accept(
                aggregation=aggregation,
                token=token,
                ctx=ctx,
                step_in_pipeline=2,
            )
            assert result.accepted is True

            # Check if timeout trigger fired
            if agg_executor.should_flush(agg_node.node_id):
                trigger_type = agg_executor.get_trigger_type(agg_node.node_id)

                # CRITICAL VERIFICATION: Timeout triggered, not count
                assert trigger_type == TriggerType.TIMEOUT, (
                    f"Expected TriggerType.TIMEOUT, got {trigger_type}. "
                    f"Timeout should fire before count with {row_index + 1} rows."
                )

                timeout_triggered = True
                rows_processed_before_timeout = row_index + 1
                break

        # CRITICAL: Timeout MUST have triggered
        assert timeout_triggered, (
            "Timeout trigger should have fired. "
            f"Processed {len(source_rows)} rows with 20ms delay each, "
            "but 50ms timeout never triggered."
        )

        # Timeout should fire after 2-4 rows (50ms / 20ms = 2.5 rows)
        # Allow some timing variance
        assert rows_processed_before_timeout <= 5, (
            f"Timeout should fire after 2-4 rows, but fired after {rows_processed_before_timeout}. "
            "Timing may be off or trigger logic is incorrect."
        )

        # Flush the batch
        output_rows = agg_executor.flush(
            aggregation=aggregation,
            ctx=ctx,
            trigger_reason="timeout",
            step_in_pipeline=3,
        )

        # Verify flushed rows match processed rows
        assert len(output_rows) == rows_processed_before_timeout, (
            f"Expected {rows_processed_before_timeout} rows from flush, "
            f"got {len(output_rows)}."
        )

        # Write to sink
        for idx, output_row in enumerate(output_rows):
            output_token = TokenInfo(
                row_id=f"timeout_output_{idx}",
                token_id=f"timeout_token_{idx}",
                row_data=output_row,
            )
            sink_executor.write(
                sink=sink,
                tokens=[output_token],
                ctx=ctx,
                step_in_pipeline=4,
            )

        # Verify sink received the rows
        assert len(sink.results) == rows_processed_before_timeout

        # Complete the run
        recorder.complete_run(run_id, status="completed")

        # Verify batch was created and triggered by timeout
        batches = recorder.get_batches(run_id)
        assert len(batches) == 1
        assert (
            batches[0].trigger_reason == "timeout"
        ), f"Expected trigger_reason='timeout', got '{batches[0].trigger_reason}'"

    def test_multiple_aggregations_in_pipeline(self) -> None:
        """Two sequential aggregations in the same pipeline.

        Pipeline: source (5 rows) -> agg1 (count=2) -> agg2 (count=3) -> sink
        - agg1 groups by 2: [1,2] -> sum=3, [3,4] -> sum=7, [5] -> sum=5 (end_of_source)
        - agg2 groups by 3: [3,7,5] -> sum=15
        - Final result: sink receives 1 row with value=15

        Verifies that aggregation outputs can feed into another aggregation,
        with proper token creation at each stage.
        """
        from elspeth.contracts import NodeType, PluginSchema, TokenInfo
        from elspeth.contracts.schema import SchemaConfig
        from elspeth.core.config import AggregationSettings, TriggerConfig
        from elspeth.core.landscape import LandscapeDB, LandscapeRecorder
        from elspeth.engine.artifacts import ArtifactDescriptor
        from elspeth.engine.executors import AggregationExecutor, SinkExecutor
        from elspeth.engine.spans import SpanFactory
        from elspeth.engine.tokens import TokenManager
        from elspeth.plugins.base import BaseAggregation
        from elspeth.plugins.context import PluginContext
        from elspeth.plugins.results import AcceptResult

        db = LandscapeDB.in_memory()
        recorder = LandscapeRecorder(db)

        # Start a run
        run = recorder.begin_run(config={}, canonical_version="v1")
        run_id = run.run_id

        DYNAMIC_SCHEMA = SchemaConfig.from_dict({"fields": "dynamic"})

        class RowSchema(PluginSchema):
            value: int

        # Register nodes
        source_node = recorder.register_node(
            run_id=run_id,
            node_id="source",
            plugin_name="list_source",
            node_type=NodeType.SOURCE,
            plugin_version="1.0.0",
            config={},
            schema_config=DYNAMIC_SCHEMA,
        )

        agg1_node = recorder.register_node(
            run_id=run_id,
            node_id="sum_agg_1",
            plugin_name="sum_agg",
            node_type=NodeType.AGGREGATION,
            plugin_version="1.0.0",
            config={},
            schema_config=DYNAMIC_SCHEMA,
        )

        agg2_node = recorder.register_node(
            run_id=run_id,
            node_id="sum_agg_2",
            plugin_name="sum_agg",
            node_type=NodeType.AGGREGATION,
            plugin_version="1.0.0",
            config={},
            schema_config=DYNAMIC_SCHEMA,
        )

        sink_node = recorder.register_node(
            run_id=run_id,
            node_id="output_sink",
            plugin_name="collect_sink",
            node_type=NodeType.SINK,
            plugin_version="1.0.0",
            config={},
            schema_config=DYNAMIC_SCHEMA,
        )

        class SumAggregation(BaseAggregation):
            """Aggregation that sums values."""

            name = "sum_agg"
            input_schema = RowSchema
            output_schema = RowSchema
            plugin_version = "1.0.0"

            def __init__(self) -> None:
                super().__init__({})
                self._values: list[int] = []

            def accept(self, row: dict[str, Any], ctx: PluginContext) -> AcceptResult:
                self._values.append(row["value"])
                return AcceptResult(accepted=True)

            def flush(self, ctx: PluginContext) -> list[dict[str, Any]]:
                total = sum(self._values)
                self._values = []
                # output_mode=single means we return exactly ONE aggregated result
                return [{"value": total}]

        class CollectSink(_TestSinkBase):
            name = "collect"

            def __init__(self) -> None:
                self.results: list[dict[str, Any]] = []

            def on_start(self, ctx: Any) -> None:
                pass

            def on_complete(self, ctx: Any) -> None:
                pass

            def write(self, rows: Any, ctx: Any) -> ArtifactDescriptor:
                self.results.extend(rows)
                return ArtifactDescriptor.for_file(
                    path="memory://output", size_bytes=0, content_hash="test_hash"
                )

            def close(self) -> None:
                pass

        # Configure aggregation 1: flush after 2 rows, output_mode=single
        agg1_settings = AggregationSettings(
            name="sum_agg_1",
            plugin="sum_agg",
            trigger=TriggerConfig(count=2),
            output_mode="single",
        )

        # Configure aggregation 2: flush after 3 rows, output_mode=single
        agg2_settings = AggregationSettings(
            name="sum_agg_2",
            plugin="sum_agg",
            trigger=TriggerConfig(count=3),
            output_mode="single",
        )

        # Create aggregation instances and assign node_ids
        aggregation1 = SumAggregation()
        aggregation1.node_id = agg1_node.node_id

        aggregation2 = SumAggregation()
        aggregation2.node_id = agg2_node.node_id

        # Create sink instance and assign node_id
        sink = CollectSink()
        sink.node_id = sink_node.node_id

        # Create executors
        span_factory = SpanFactory()
        token_manager = TokenManager(recorder)

        agg1_executor = AggregationExecutor(
            recorder=recorder,
            span_factory=span_factory,
            run_id=run_id,
            aggregation_settings={agg1_node.node_id: agg1_settings},
        )

        agg2_executor = AggregationExecutor(
            recorder=recorder,
            span_factory=span_factory,
            run_id=run_id,
            aggregation_settings={agg2_node.node_id: agg2_settings},
        )

        sink_executor = SinkExecutor(recorder, span_factory, run_id)

        ctx = PluginContext(run_id=run_id, config={})

        # Source: 5 rows with values 1, 2, 3, 4, 5
        source_rows = [{"value": i} for i in range(1, 6)]

        # Track intermediate outputs from agg1
        agg1_outputs: list[dict[str, Any]] = []

        # Process source rows through agg1
        for row_index, row_data in enumerate(source_rows):
            # Create token for each source row
            token = token_manager.create_initial_token(
                run_id=run_id,
                source_node_id=source_node.node_id,
                row_index=row_index,
                row_data=row_data,
            )

            # Accept row into aggregation 1
            result = agg1_executor.accept(
                aggregation=aggregation1,
                token=token,
                ctx=ctx,
                step_in_pipeline=1,
            )
            assert result.accepted is True

            # Check if trigger condition is met (after 2 rows)
            if agg1_executor.should_flush(agg1_node.node_id):
                output_rows = agg1_executor.flush(
                    aggregation=aggregation1,
                    ctx=ctx,
                    trigger_reason="count_reached",
                    step_in_pipeline=2,
                )
                agg1_outputs.extend(output_rows)

        # Flush remaining rows from agg1 (end_of_source)
        remaining_rows = agg1_executor.flush(
            aggregation=aggregation1,
            ctx=ctx,
            trigger_reason="end_of_source",
            step_in_pipeline=2,
        )
        agg1_outputs.extend(remaining_rows)

        # Verify agg1 output: [3, 7, 5]
        # - Batch 1: 1+2 = 3
        # - Batch 2: 3+4 = 7
        # - Batch 3 (end_of_source): 5
        assert len(agg1_outputs) == 3, (
            f"Expected 3 outputs from agg1, got {len(agg1_outputs)}. "
            "agg1 should produce 3 sums: [1+2=3, 3+4=7, 5]."
        )
        agg1_values = [r["value"] for r in agg1_outputs]
        assert agg1_values == [3, 7, 5], (
            f"Expected agg1 values [3, 7, 5], got {agg1_values}. "
            "agg1 groups by 2 with end_of_source flush for the last row."
        )

        # Feed agg1 outputs into agg2
        for idx, agg1_output in enumerate(agg1_outputs):
            # Create new token for agg1 output (intermediate token)
            agg1_output_token = TokenInfo(
                row_id=f"agg1_output_{idx}",
                token_id=f"agg1_token_{idx}",
                row_data=agg1_output,
            )

            # Accept into aggregation 2
            result = agg2_executor.accept(
                aggregation=aggregation2,
                token=agg1_output_token,
                ctx=ctx,
                step_in_pipeline=3,
            )
            assert result.accepted is True

            # Check if trigger condition is met (after 3 rows)
            if agg2_executor.should_flush(agg2_node.node_id):
                output_rows = agg2_executor.flush(
                    aggregation=aggregation2,
                    ctx=ctx,
                    trigger_reason="count_reached",
                    step_in_pipeline=4,
                )

                # Write agg2 output to sink
                for sink_idx, sink_row in enumerate(output_rows):
                    sink_output_token = TokenInfo(
                        row_id=f"agg2_output_{sink_idx}",
                        token_id=f"agg2_token_{sink_idx}",
                        row_data=sink_row,
                    )
                    sink_executor.write(
                        sink=sink,
                        tokens=[sink_output_token],
                        ctx=ctx,
                        step_in_pipeline=5,
                    )

        # agg2 should have flushed exactly when count=3 was reached
        # (after receiving all 3 outputs from agg1)

        # CRITICAL VERIFICATION: Sink received 1 row with value=15
        assert len(sink.results) == 1, (
            f"Expected 1 final aggregated row, got {len(sink.results)}. "
            "agg2 should sum the 3 outputs from agg1 into one row."
        )
        assert sink.results[0]["value"] == 15, (
            f"Expected final value 15 (3+7+5), got {sink.results[0]['value']}. "
            "agg2 should sum all agg1 outputs: 3+7+5=15."
        )

        # Complete the run
        recorder.complete_run(run_id, status="completed")

        # Verify audit trail completeness
        # Source rows
        rows = recorder.get_rows(run_id)
        assert len(rows) == 5, f"Expected 5 source rows, got {len(rows)}"

        # Verify batches were created:
        # agg1: 3 batches (2 count-triggered, 1 end_of_source)
        # agg2: 1 batch (count-triggered when reaching 3)
        batches = recorder.get_batches(run_id)
        assert (
            len(batches) == 4
        ), f"Expected 4 batches total (3 from agg1, 1 from agg2), got {len(batches)}"

        # Verify artifact was recorded for sink
        artifacts = recorder.get_artifacts(run_id)
        assert len(artifacts) == 1
        assert artifacts[0].content_hash == "test_hash"

    def test_aggregation_after_gate_routing(self) -> None:
        """Rows routed by gate aggregate correctly in separate paths.

        Pipeline: source -> gate -> aggregation(s) -> sink(s)

        Rows are routed by value:
        - High-value rows (>50) go to high_agg -> high_sink
        - Low-value rows (<=50) go to low_agg -> low_sink

        Source data:
        - [10, 20, 60, 70]

        Expected results:
        - High path: 60 + 70 = 130 -> high_sink
        - Low path: 10 + 20 = 30 -> low_sink
        """
        from elspeth.contracts import NodeType, PluginSchema, TokenInfo
        from elspeth.contracts.schema import SchemaConfig
        from elspeth.core.config import AggregationSettings, GateSettings, TriggerConfig
        from elspeth.core.landscape import LandscapeDB, LandscapeRecorder
        from elspeth.engine.artifacts import ArtifactDescriptor
        from elspeth.engine.executors import (
            AggregationExecutor,
            GateExecutor,
            SinkExecutor,
        )
        from elspeth.engine.spans import SpanFactory
        from elspeth.engine.tokens import TokenManager
        from elspeth.plugins.base import BaseAggregation
        from elspeth.plugins.context import PluginContext
        from elspeth.plugins.results import AcceptResult

        db = LandscapeDB.in_memory()
        recorder = LandscapeRecorder(db)

        # Start a run
        run = recorder.begin_run(config={}, canonical_version="v1")
        run_id = run.run_id

        DYNAMIC_SCHEMA = SchemaConfig.from_dict({"fields": "dynamic"})

        class RowSchema(PluginSchema):
            value: int

        # Register nodes: source, gate, high_agg, low_agg, high_sink, low_sink
        source_node = recorder.register_node(
            run_id=run_id,
            node_id="source",
            plugin_name="list_source",
            node_type=NodeType.SOURCE,
            plugin_version="1.0.0",
            config={},
            schema_config=DYNAMIC_SCHEMA,
        )

        gate_node = recorder.register_node(
            run_id=run_id,
            node_id="value_router",
            plugin_name="config_gate:value_router",
            node_type=NodeType.GATE,
            plugin_version="1.0.0",
            config={},
            schema_config=DYNAMIC_SCHEMA,
        )

        high_agg_node = recorder.register_node(
            run_id=run_id,
            node_id="high_agg",
            plugin_name="sum_agg",
            node_type=NodeType.AGGREGATION,
            plugin_version="1.0.0",
            config={},
            schema_config=DYNAMIC_SCHEMA,
        )

        low_agg_node = recorder.register_node(
            run_id=run_id,
            node_id="low_agg",
            plugin_name="sum_agg",
            node_type=NodeType.AGGREGATION,
            plugin_version="1.0.0",
            config={},
            schema_config=DYNAMIC_SCHEMA,
        )

        high_sink_node = recorder.register_node(
            run_id=run_id,
            node_id="high_sink",
            plugin_name="collect_sink",
            node_type=NodeType.SINK,
            plugin_version="1.0.0",
            config={},
            schema_config=DYNAMIC_SCHEMA,
        )

        low_sink_node = recorder.register_node(
            run_id=run_id,
            node_id="low_sink",
            plugin_name="collect_sink",
            node_type=NodeType.SINK,
            plugin_version="1.0.0",
            config={},
            schema_config=DYNAMIC_SCHEMA,
        )

        # Register edges for gate routing
        edge_high = recorder.register_edge(
            run_id=run_id,
            from_node_id=gate_node.node_id,
            to_node_id=high_agg_node.node_id,
            label="true",
            mode=RoutingMode.MOVE,
        )
        edge_low = recorder.register_edge(
            run_id=run_id,
            from_node_id=gate_node.node_id,
            to_node_id=low_agg_node.node_id,
            label="false",
            mode=RoutingMode.MOVE,
        )

        # Build edge_map for GateExecutor
        edge_map = {
            (gate_node.node_id, "true"): edge_high.edge_id,
            (gate_node.node_id, "false"): edge_low.edge_id,
        }

        # Route resolution map for config gate: true -> high_path, false -> low_path
        # Note: For this test we're just using route labels directly, not routing to sinks
        route_resolution_map: dict[tuple[str, str], str] = {
            (gate_node.node_id, "true"): "high_path",
            (gate_node.node_id, "false"): "low_path",
        }

        class SumAggregation(BaseAggregation):
            """Aggregation that sums values."""

            name = "sum_agg"
            input_schema = RowSchema
            output_schema = RowSchema
            plugin_version = "1.0.0"

            def __init__(self) -> None:
                super().__init__({})
                self._values: list[int] = []

            def accept(self, row: dict[str, Any], ctx: PluginContext) -> AcceptResult:
                self._values.append(row["value"])
                return AcceptResult(accepted=True)

            def flush(self, ctx: PluginContext) -> list[dict[str, Any]]:
                total = sum(self._values)
                self._values = []
                return [{"value": total}]

        class CollectSink(_TestSinkBase):
            name = "collect"

            def __init__(self) -> None:
                self.results: list[dict[str, Any]] = []

            def on_start(self, ctx: Any) -> None:
                pass

            def on_complete(self, ctx: Any) -> None:
                pass

            def write(self, rows: Any, ctx: Any) -> ArtifactDescriptor:
                self.results.extend(rows)
                return ArtifactDescriptor.for_file(
                    path="memory://output", size_bytes=0, content_hash="test_hash"
                )

            def close(self) -> None:
                pass

        # Configure gate: routes high values (>50) vs low values (<=50)
        gate_settings = GateSettings(
            name="value_router",
            condition="row['value'] > 50",
            routes={"true": "high_path", "false": "low_path"},
        )

        # Configure aggregations: flush when source exhausts (use high count trigger)
        high_agg_settings = AggregationSettings(
            name="high_agg",
            plugin="sum_agg",
            trigger=TriggerConfig(count=1000),  # Won't fire - rely on end_of_source
            output_mode="single",
        )

        low_agg_settings = AggregationSettings(
            name="low_agg",
            plugin="sum_agg",
            trigger=TriggerConfig(count=1000),  # Won't fire - rely on end_of_source
            output_mode="single",
        )

        # Create plugin instances and assign node_ids
        high_agg = SumAggregation()
        high_agg.node_id = high_agg_node.node_id

        low_agg = SumAggregation()
        low_agg.node_id = low_agg_node.node_id

        high_sink = CollectSink()
        high_sink.node_id = high_sink_node.node_id

        low_sink = CollectSink()
        low_sink.node_id = low_sink_node.node_id

        # Create executors
        span_factory = SpanFactory()
        token_manager = TokenManager(recorder)

        gate_executor = GateExecutor(
            recorder=recorder,
            span_factory=span_factory,
            edge_map=edge_map,
            route_resolution_map=route_resolution_map,
        )

        high_agg_executor = AggregationExecutor(
            recorder=recorder,
            span_factory=span_factory,
            run_id=run_id,
            aggregation_settings={high_agg_node.node_id: high_agg_settings},
        )

        low_agg_executor = AggregationExecutor(
            recorder=recorder,
            span_factory=span_factory,
            run_id=run_id,
            aggregation_settings={low_agg_node.node_id: low_agg_settings},
        )

        high_sink_executor = SinkExecutor(recorder, span_factory, run_id)
        low_sink_executor = SinkExecutor(recorder, span_factory, run_id)

        ctx = PluginContext(run_id=run_id, config={})

        # Source data: mix of high and low values
        source_rows = [
            {"value": 10},  # low
            {"value": 20},  # low
            {"value": 60},  # high
            {"value": 70},  # high
        ]

        # Track tokens routed to each path
        high_tokens: list[TokenInfo] = []
        low_tokens: list[TokenInfo] = []

        # Process each source row through the gate
        for row_index, row_data in enumerate(source_rows):
            # Create token for each source row
            token = token_manager.create_initial_token(
                run_id=run_id,
                source_node_id=source_node.node_id,
                row_index=row_index,
                row_data=row_data,
            )

            # Evaluate gate to determine routing
            outcome = gate_executor.execute_config_gate(
                gate_config=gate_settings,
                node_id=gate_node.node_id,
                token=token,
                ctx=ctx,
                step_in_pipeline=1,
            )

            # Route token based on gate result
            # GateResult action contains routing info
            action = outcome.result.action
            route_label = action.destinations[0] if action.destinations else None

            if route_label == "true":
                # High value path
                high_tokens.append(outcome.updated_token)
            else:
                # Low value path
                low_tokens.append(outcome.updated_token)

        # Verify routing: 2 high (60, 70), 2 low (10, 20)
        assert len(high_tokens) == 2, f"Expected 2 high tokens, got {len(high_tokens)}"
        assert len(low_tokens) == 2, f"Expected 2 low tokens, got {len(low_tokens)}"

        high_values = [t.row_data["value"] for t in high_tokens]
        low_values = [t.row_data["value"] for t in low_tokens]
        assert set(high_values) == {
            60,
            70,
        }, f"Expected high values {{60, 70}}, got {set(high_values)}"
        assert set(low_values) == {
            10,
            20,
        }, f"Expected low values {{10, 20}}, got {set(low_values)}"

        # Process high tokens through high aggregation
        for token in high_tokens:
            result = high_agg_executor.accept(
                aggregation=high_agg,
                token=token,
                ctx=ctx,
                step_in_pipeline=2,
            )
            assert result.accepted is True

        # Process low tokens through low aggregation
        for token in low_tokens:
            result = low_agg_executor.accept(
                aggregation=low_agg,
                token=token,
                ctx=ctx,
                step_in_pipeline=2,
            )
            assert result.accepted is True

        # Flush high aggregation (end_of_source)
        high_output = high_agg_executor.flush(
            aggregation=high_agg,
            ctx=ctx,
            trigger_reason="end_of_source",
            step_in_pipeline=3,
        )

        # Flush low aggregation (end_of_source)
        low_output = low_agg_executor.flush(
            aggregation=low_agg,
            ctx=ctx,
            trigger_reason="end_of_source",
            step_in_pipeline=3,
        )

        # CRITICAL VERIFICATION: Aggregation results are correct
        assert (
            len(high_output) == 1
        ), f"Expected 1 high aggregation result, got {len(high_output)}"
        assert (
            high_output[0]["value"] == 130
        ), f"Expected high sum 130 (60+70), got {high_output[0]['value']}"

        assert (
            len(low_output) == 1
        ), f"Expected 1 low aggregation result, got {len(low_output)}"
        assert (
            low_output[0]["value"] == 30
        ), f"Expected low sum 30 (10+20), got {low_output[0]['value']}"

        # Write aggregation outputs to respective sinks
        high_output_token = TokenInfo(
            row_id="high_agg_output_0",
            token_id="high_agg_token_0",
            row_data=high_output[0],
        )
        high_sink_executor.write(
            sink=high_sink,
            tokens=[high_output_token],
            ctx=ctx,
            step_in_pipeline=4,
        )

        low_output_token = TokenInfo(
            row_id="low_agg_output_0",
            token_id="low_agg_token_0",
            row_data=low_output[0],
        )
        low_sink_executor.write(
            sink=low_sink,
            tokens=[low_output_token],
            ctx=ctx,
            step_in_pipeline=4,
        )

        # CRITICAL VERIFICATION: Sinks received correct aggregated values
        assert (
            len(high_sink.results) == 1
        ), f"Expected 1 row in high_sink, got {len(high_sink.results)}"
        assert (
            high_sink.results[0]["value"] == 130
        ), f"Expected high_sink value 130, got {high_sink.results[0]['value']}"

        assert (
            len(low_sink.results) == 1
        ), f"Expected 1 row in low_sink, got {len(low_sink.results)}"
        assert (
            low_sink.results[0]["value"] == 30
        ), f"Expected low_sink value 30, got {low_sink.results[0]['value']}"

        # Complete the run
        recorder.complete_run(run_id, status="completed")

        # Verify audit trail completeness
        rows = recorder.get_rows(run_id)
        assert len(rows) == 4, f"Expected 4 source rows, got {len(rows)}"

        # Verify batches were created (2 batches: high_agg, low_agg)
        batches = recorder.get_batches(run_id)
        assert (
            len(batches) == 2
        ), f"Expected 2 batches (high and low), got {len(batches)}"

        # Verify each batch has correct member count
        batch_member_counts = {}
        for batch in batches:
            members = recorder.get_batch_members(batch.batch_id)
            batch_member_counts[batch.aggregation_node_id] = len(members)

        assert (
            batch_member_counts[high_agg_node.node_id] == 2
        ), f"Expected 2 members in high_agg batch, got {batch_member_counts[high_agg_node.node_id]}"
        assert (
            batch_member_counts[low_agg_node.node_id] == 2
        ), f"Expected 2 members in low_agg batch, got {batch_member_counts[low_agg_node.node_id]}"

        # Verify artifacts were recorded (2 artifacts: one for each sink)
        artifacts = recorder.get_artifacts(run_id)
        assert len(artifacts) == 2, f"Expected 2 artifacts, got {len(artifacts)}"

    def test_aggregation_condition_trigger(self) -> None:
        """Condition trigger fires when both batch_count and batch_age_seconds conditions are met.

        Pipeline: source (10 rows) -> delay_transform (5ms delay) -> aggregation -> sink
        - condition: row['batch_count'] >= 3 and row['batch_age_seconds'] > 0.01

        The condition requires BOTH:
        - At least 3 rows in the batch
        - At least 10ms elapsed since batch start

        With 5ms delay per row, after 3 rows we have:
        - batch_count = 3 (meets >= 3)
        - batch_age_seconds >= 15ms (meets > 10ms)

        Verifies that condition trigger fires when both conditions are satisfied.
        """
        import time

        from elspeth.contracts import NodeType, PluginSchema, TokenInfo
        from elspeth.contracts.enums import TriggerType
        from elspeth.contracts.schema import SchemaConfig
        from elspeth.core.config import AggregationSettings, TriggerConfig
        from elspeth.core.landscape import LandscapeDB, LandscapeRecorder
        from elspeth.engine.artifacts import ArtifactDescriptor
        from elspeth.engine.executors import AggregationExecutor, SinkExecutor
        from elspeth.engine.spans import SpanFactory
        from elspeth.engine.tokens import TokenManager
        from elspeth.plugins.base import BaseAggregation, BaseTransform
        from elspeth.plugins.context import PluginContext
        from elspeth.plugins.results import AcceptResult, TransformResult

        db = LandscapeDB.in_memory()
        recorder = LandscapeRecorder(db)

        # Start a run
        run = recorder.begin_run(config={}, canonical_version="v1")
        run_id = run.run_id

        DYNAMIC_SCHEMA = SchemaConfig.from_dict({"fields": "dynamic"})

        class RowSchema(PluginSchema):
            value: int

        # Register nodes
        source_node = recorder.register_node(
            run_id=run_id,
            node_id="source",
            plugin_name="list_source",
            node_type=NodeType.SOURCE,
            plugin_version="1.0.0",
            config={},
            schema_config=DYNAMIC_SCHEMA,
        )

        transform_node = recorder.register_node(
            run_id=run_id,
            node_id="delay_transform",
            plugin_name="delay_transform",
            node_type=NodeType.TRANSFORM,
            plugin_version="1.0.0",
            config={},
            schema_config=DYNAMIC_SCHEMA,
        )

        agg_node = recorder.register_node(
            run_id=run_id,
            node_id="buffer_agg",
            plugin_name="buffer_agg",
            node_type=NodeType.AGGREGATION,
            plugin_version="1.0.0",
            config={},
            schema_config=DYNAMIC_SCHEMA,
        )

        sink_node = recorder.register_node(
            run_id=run_id,
            node_id="output_sink",
            plugin_name="collect_sink",
            node_type=NodeType.SINK,
            plugin_version="1.0.0",
            config={},
            schema_config=DYNAMIC_SCHEMA,
        )

        class DelayTransform(BaseTransform):
            """Transform that adds 5ms delay per row."""

            name = "delay_transform"
            input_schema = RowSchema
            output_schema = RowSchema
            plugin_version = "1.0.0"

            def __init__(self) -> None:
                super().__init__({})

            def process(
                self, row: dict[str, Any], ctx: PluginContext
            ) -> TransformResult:
                time.sleep(0.005)  # 5ms delay
                return TransformResult.success(row)

        class BufferAggregation(BaseAggregation):
            """Aggregation that buffers and releases rows unchanged."""

            name = "buffer_agg"
            input_schema = RowSchema
            output_schema = RowSchema
            plugin_version = "1.0.0"

            def __init__(self) -> None:
                super().__init__({})
                self._buffer: list[dict[str, Any]] = []

            def accept(self, row: dict[str, Any], ctx: PluginContext) -> AcceptResult:
                self._buffer.append(row)
                return AcceptResult(accepted=True)

            def flush(self, ctx: PluginContext) -> list[dict[str, Any]]:
                result = list(self._buffer)
                self._buffer = []
                return result

        class CollectSink(_TestSinkBase):
            name = "collect"

            def __init__(self) -> None:
                self.results: list[dict[str, Any]] = []

            def on_start(self, ctx: Any) -> None:
                pass

            def on_complete(self, ctx: Any) -> None:
                pass

            def write(self, rows: Any, ctx: Any) -> ArtifactDescriptor:
                self.results.extend(rows)
                return ArtifactDescriptor.for_file(
                    path="memory://output", size_bytes=0, content_hash="test_hash"
                )

            def close(self) -> None:
                pass

        # Configure aggregation: condition trigger only (no count/timeout)
        # Requires BOTH: batch_count >= 3 AND batch_age_seconds > 0.01 (10ms)
        agg_settings = AggregationSettings(
            name="buffer_agg",
            plugin="buffer_agg",
            trigger=TriggerConfig(
                condition="row['batch_count'] >= 3 and row['batch_age_seconds'] > 0.01"
            ),
            output_mode="passthrough",
        )

        # Create instances and assign node_ids
        transform = DelayTransform()
        transform.node_id = transform_node.node_id

        aggregation = BufferAggregation()
        aggregation.node_id = agg_node.node_id

        sink = CollectSink()
        sink.node_id = sink_node.node_id

        # Create executors
        span_factory = SpanFactory()
        token_manager = TokenManager(recorder)

        agg_executor = AggregationExecutor(
            recorder=recorder,
            span_factory=span_factory,
            run_id=run_id,
            aggregation_settings={agg_node.node_id: agg_settings},
        )

        sink_executor = SinkExecutor(recorder, span_factory, run_id)

        ctx = PluginContext(run_id=run_id, config={})

        # Source with 10 rows
        source_rows = [{"value": i} for i in range(1, 11)]

        # Track when condition triggers
        condition_triggered = False
        rows_processed_before_condition = 0

        for row_index, row_data in enumerate(source_rows):
            # Create token for each source row
            token = token_manager.create_initial_token(
                run_id=run_id,
                source_node_id=source_node.node_id,
                row_index=row_index,
                row_data=row_data,
            )

            # Process through delay transform (adds 5ms delay)
            transform_result = transform.process(row_data, ctx)
            transformed_data = transform_result.row

            # Update token with transformed data
            token = TokenInfo(
                row_id=token.row_id,
                token_id=token.token_id,
                row_data=transformed_data,
            )

            # Accept into aggregation
            result = agg_executor.accept(
                aggregation=aggregation,
                token=token,
                ctx=ctx,
                step_in_pipeline=2,
            )
            assert result.accepted is True

            # Check if condition trigger fired
            if agg_executor.should_flush(agg_node.node_id):
                trigger_type = agg_executor.get_trigger_type(agg_node.node_id)

                # CRITICAL VERIFICATION: Condition triggered
                assert trigger_type == TriggerType.CONDITION, (
                    f"Expected TriggerType.CONDITION, got {trigger_type}. "
                    f"Condition should fire after {row_index + 1} rows."
                )

                condition_triggered = True
                rows_processed_before_condition = row_index + 1
                break

        # CRITICAL: Condition MUST have triggered
        assert condition_triggered, (
            "Condition trigger should have fired. "
            f"Processed {len(source_rows)} rows with 5ms delay each, "
            "but condition (batch_count >= 3 and batch_age_seconds > 0.01) never triggered."
        )

        # Condition requires batch_count >= 3, so it must fire on row 3 or later
        # With 5ms per row, by row 3 we have ~15ms elapsed (> 10ms requirement)
        assert rows_processed_before_condition >= 3, (
            f"Condition should fire after at least 3 rows, but fired after {rows_processed_before_condition}. "
            "batch_count >= 3 requires at least 3 rows."
        )

        # Should fire relatively soon after 3 rows (once time condition is also met)
        assert rows_processed_before_condition <= 5, (
            f"Condition should fire around row 3-4, but fired after {rows_processed_before_condition}. "
            "With 5ms delay per row, time condition (> 10ms) should be met by row 3."
        )

        # Flush the batch
        output_rows = agg_executor.flush(
            aggregation=aggregation,
            ctx=ctx,
            trigger_reason="condition",
            step_in_pipeline=3,
        )

        # Verify flushed rows match processed rows
        assert len(output_rows) == rows_processed_before_condition, (
            f"Expected {rows_processed_before_condition} rows from flush, "
            f"got {len(output_rows)}."
        )

        # Write to sink
        for idx, output_row in enumerate(output_rows):
            output_token = TokenInfo(
                row_id=f"condition_output_{idx}",
                token_id=f"condition_token_{idx}",
                row_data=output_row,
            )
            sink_executor.write(
                sink=sink,
                tokens=[output_token],
                ctx=ctx,
                step_in_pipeline=4,
            )

        # Verify sink received the rows
        assert len(sink.results) == rows_processed_before_condition

        # Complete the run
        recorder.complete_run(run_id, status="completed")

        # Verify batch was created and triggered by condition
        batches = recorder.get_batches(run_id)
        assert len(batches) == 1
        assert (
            batches[0].trigger_reason == "condition"
        ), f"Expected trigger_reason='condition', got '{batches[0].trigger_reason}'"

        # Verify batch has correct member count
        members = recorder.get_batch_members(batches[0].batch_id)
        assert (
            len(members) == rows_processed_before_condition
        ), f"Expected {rows_processed_before_condition} batch members, got {len(members)}"


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
            routes={"true": "fork"},
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
        sentiment_token = next(
            t for t in gate_outcome.child_tokens if t.branch_name == "sentiment"
        )
        entity_token = next(
            t for t in gate_outcome.child_tokens if t.branch_name == "entities"
        )

        # Step 2: Process each branch through its transform
        from elspeth.engine.executors import TransformExecutor

        transform_executor = TransformExecutor(recorder, span_factory)

        # Process sentiment branch
        sentiment_result, sentiment_token_updated, _ = (
            transform_executor.execute_transform(
                transform=sentiment,
                token=sentiment_token,
                ctx=ctx,
                step_in_pipeline=2,
            )
        )
        assert sentiment_result.status == "success"
        # Update token with transformed data while preserving branch_name
        sentiment_token_processed = TokenInfo(
            row_id=sentiment_token_updated.row_id,
            token_id=sentiment_token_updated.token_id,
            row_data=sentiment_result.row,
            branch_name="sentiment",
        )

        # Process entity branch
        entity_result, entity_token_updated, _ = transform_executor.execute_transform(
            transform=entity,
            token=entity_token,
            ctx=ctx,
            step_in_pipeline=2,
        )
        assert entity_result.status == "success"
        # Update token with transformed data while preserving branch_name
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
            f"Expected 1 merged row, got {len(sink.rows_written)}. "
            "Fork children should be coalesced before sink."
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
            routes={"true": "fork"},
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
        assert (
            len(merged_tokens) == 3
        ), f"Expected 3 merged tokens, got {len(merged_tokens)}"

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
            routes={"true": "fork"},
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
        assert (
            len(gate_outcome.child_tokens) == 2
        ), f"Fork should create 2 children, got {len(gate_outcome.child_tokens)}"

        sentiment_token = next(
            t for t in gate_outcome.child_tokens if t.branch_name == "sentiment_path"
        )
        entity_token = next(
            t for t in gate_outcome.child_tokens if t.branch_name == "entity_path"
        )

        # Step 3: Execute transforms on each branch
        # Sentiment branch
        sentiment_result, sentiment_token_updated, _ = (
            transform_executor.execute_transform(
                transform=sentiment_transform,
                token=sentiment_token,
                ctx=ctx,
                step_in_pipeline=2,
            )
        )
        assert sentiment_result.status == "success"
        sentiment_token_processed = TokenInfo(
            row_id=sentiment_token_updated.row_id,
            token_id=sentiment_token_updated.token_id,
            row_data=sentiment_result.row,
            branch_name="sentiment_path",
        )

        # Entity branch
        entity_result, entity_token_updated, _ = transform_executor.execute_transform(
            transform=entity_transform,
            token=entity_token,
            ctx=ctx,
            step_in_pipeline=2,
        )
        assert entity_result.status == "success"
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
        assert (
            "sentiment" in merged_data
        ), "Merged data should have sentiment from branch A"
        assert (
            "entities" in merged_data
        ), "Merged data should have entities from branch B"
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
            f"Expected 1 merged row, got {len(sink.rows_written)}. "
            "Diamond DAG should merge branches before sink."
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
        assert (
            len(all_tokens) == 4
        ), f"Expected 4 tokens (initial + 2 fork + 1 merged), got {len(all_tokens)}"

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

    def test_full_feature_pipeline(self) -> None:
        """Pipeline using ALL features: gate routing + fork + coalesce + aggregation.

        Pipeline structure:
        source (10 rows, values 0-9)
        -> threshold_gate (routes high/low based on value >= 5)
            -> high path (values 5-9):
                -> fork_gate (splits to path_a and path_b)
                    -> transform_A (adds field_a)
                    -> transform_B (adds field_b)
                -> coalesce (merges A+B)
                -> aggregation (batches by 2, output_mode=single)
                -> high_sink
            -> low path (values 0-4):
                -> transform_C (adds field_c)
                -> low_sink

        This exercises:
        - Config gate routing (threshold)
        - Fork execution
        - Coalesce merge
        - Aggregation triggers

        Expected results:
        - Low path: 5 rows (0-4) -> low_sink receives 5 rows
        - High path: 5 rows (5-9) -> forked -> coalesced -> aggregated by 2
          -> high_sink receives 3 outputs (2 batches of 2 + 1 end_of_source flush)
        """
        from elspeth.contracts import NodeType, TokenInfo
        from elspeth.contracts.schema import SchemaConfig
        from elspeth.core.config import (
            AggregationSettings,
            CoalesceSettings,
            TriggerConfig,
        )
        from elspeth.core.landscape import LandscapeDB, LandscapeRecorder
        from elspeth.engine.artifacts import ArtifactDescriptor
        from elspeth.engine.coalesce_executor import CoalesceExecutor
        from elspeth.engine.executors import (
            AggregationExecutor,
            GateExecutor,
            SinkExecutor,
            TransformExecutor,
        )
        from elspeth.engine.spans import SpanFactory
        from elspeth.engine.tokens import TokenManager
        from elspeth.plugins.base import BaseAggregation
        from elspeth.plugins.context import PluginContext
        from elspeth.plugins.results import AcceptResult, TransformResult

        db = LandscapeDB.in_memory()
        recorder = LandscapeRecorder(db)

        # Start a run
        run = recorder.begin_run(config={}, canonical_version="v1")
        run_id = run.run_id

        DYNAMIC_SCHEMA = SchemaConfig.from_dict({"fields": "dynamic"})

        # ====================================================================
        # Register all nodes for the complex DAG
        # ====================================================================
        source_node = recorder.register_node(
            run_id=run_id,
            node_id="source",
            plugin_name="test_source",
            node_type=NodeType.SOURCE,
            plugin_version="1.0.0",
            config={},
            schema_config=DYNAMIC_SCHEMA,
        )

        threshold_gate_node = recorder.register_node(
            run_id=run_id,
            node_id="threshold_gate",
            plugin_name="config_gate:threshold_gate",
            node_type=NodeType.GATE,
            plugin_version="1.0.0",
            config={},
            schema_config=DYNAMIC_SCHEMA,
        )

        # Low path nodes
        transform_c_node = recorder.register_node(
            run_id=run_id,
            node_id="transform_c",
            plugin_name="transform_c",
            node_type=NodeType.TRANSFORM,
            plugin_version="1.0.0",
            config={},
            schema_config=DYNAMIC_SCHEMA,
        )

        low_sink_node = recorder.register_node(
            run_id=run_id,
            node_id="low_sink",
            plugin_name="low_sink",
            node_type=NodeType.SINK,
            plugin_version="1.0.0",
            config={},
            schema_config=DYNAMIC_SCHEMA,
        )

        # High path nodes
        fork_gate_node = recorder.register_node(
            run_id=run_id,
            node_id="fork_gate",
            plugin_name="config_gate:fork_gate",
            node_type=NodeType.GATE,
            plugin_version="1.0.0",
            config={},
            schema_config=DYNAMIC_SCHEMA,
        )

        transform_a_node = recorder.register_node(
            run_id=run_id,
            node_id="transform_a",
            plugin_name="transform_a",
            node_type=NodeType.TRANSFORM,
            plugin_version="1.0.0",
            config={},
            schema_config=DYNAMIC_SCHEMA,
        )

        transform_b_node = recorder.register_node(
            run_id=run_id,
            node_id="transform_b",
            plugin_name="transform_b",
            node_type=NodeType.TRANSFORM,
            plugin_version="1.0.0",
            config={},
            schema_config=DYNAMIC_SCHEMA,
        )

        coalesce_node = recorder.register_node(
            run_id=run_id,
            node_id="high_coalesce",
            plugin_name="coalesce:high_coalesce",
            node_type=NodeType.COALESCE,
            plugin_version="1.0.0",
            config={},
            schema_config=DYNAMIC_SCHEMA,
        )

        aggregation_node = recorder.register_node(
            run_id=run_id,
            node_id="high_agg",
            plugin_name="batch_aggregator",
            node_type=NodeType.AGGREGATION,
            plugin_version="1.0.0",
            config={},
            schema_config=DYNAMIC_SCHEMA,
        )

        high_sink_node = recorder.register_node(
            run_id=run_id,
            node_id="high_sink",
            plugin_name="high_sink",
            node_type=NodeType.SINK,
            plugin_version="1.0.0",
            config={},
            schema_config=DYNAMIC_SCHEMA,
        )

        # ====================================================================
        # Register edges
        # ====================================================================
        # Threshold gate -> low path (edge label matches route result "false")
        edge_to_low = recorder.register_edge(
            run_id=run_id,
            from_node_id=threshold_gate_node.node_id,
            to_node_id=transform_c_node.node_id,
            label="false",  # Route result when value < 5
            mode=RoutingMode.MOVE,
        )

        # Threshold gate -> high path (edge label matches route result "true")
        edge_to_high = recorder.register_edge(
            run_id=run_id,
            from_node_id=threshold_gate_node.node_id,
            to_node_id=fork_gate_node.node_id,
            label="true",  # Route result when value >= 5
            mode=RoutingMode.MOVE,
        )

        # Fork gate -> path_a (transform_a)
        edge_to_a = recorder.register_edge(
            run_id=run_id,
            from_node_id=fork_gate_node.node_id,
            to_node_id=transform_a_node.node_id,
            label="path_a",
            mode=RoutingMode.COPY,
        )

        # Fork gate -> path_b (transform_b)
        edge_to_b = recorder.register_edge(
            run_id=run_id,
            from_node_id=fork_gate_node.node_id,
            to_node_id=transform_b_node.node_id,
            label="path_b",
            mode=RoutingMode.COPY,
        )

        # Edge maps for gate executor - use route_label (true/false) as keys
        threshold_edge_map = {
            (threshold_gate_node.node_id, "false"): edge_to_low.edge_id,
            (threshold_gate_node.node_id, "true"): edge_to_high.edge_id,
        }

        fork_edge_map = {
            (fork_gate_node.node_id, "path_a"): edge_to_a.edge_id,
            (fork_gate_node.node_id, "path_b"): edge_to_b.edge_id,
        }

        combined_edge_map = {**threshold_edge_map, **fork_edge_map}

        # Route resolution maps - for non-fork routing, destination is the sink
        # but in this test, we manually handle routing to transforms
        threshold_route_resolution = {
            (threshold_gate_node.node_id, "false"): "low_path",  # Routes to transform_c
            (threshold_gate_node.node_id, "true"): "high_path",  # Routes to fork_gate
        }

        fork_route_resolution = {
            (fork_gate_node.node_id, "path_a"): "fork",
            (fork_gate_node.node_id, "path_b"): "fork",
        }

        combined_route_resolution = {
            **threshold_route_resolution,
            **fork_route_resolution,
        }

        # ====================================================================
        # Config-driven gates
        # ====================================================================
        # Threshold gate: routes based on value >= 5
        # NOTE: The route values "high_path"/"low_path" are used as sink names
        # by the executor, but we manually handle the routing in the test
        threshold_gate = GateSettings(
            name="threshold_gate",
            condition="row['value'] >= 5",
            routes={"true": "high_path", "false": "low_path"},
        )

        fork_gate = GateSettings(
            name="fork_gate",
            condition="True",  # Always fork
            routes={"true": "fork"},
            fork_to=["path_a", "path_b"],
        )

        # Coalesce settings
        coalesce_settings = CoalesceSettings(
            name="high_coalesce",
            branches=["path_a", "path_b"],
            policy="require_all",
            merge="union",
        )

        # Aggregation settings (batch by 2)
        agg_settings = AggregationSettings(
            name="high_agg",
            plugin="batch_aggregator",
            trigger=TriggerConfig(count=2),
            output_mode="single",
        )

        # ====================================================================
        # Test transforms and aggregation
        # ====================================================================
        class TransformA(BaseTransform):
            """Adds field_a."""

            name = "transform_a"
            input_schema = _TestSchema
            output_schema = _TestSchema

            def __init__(self, node_id: str) -> None:
                super().__init__({})
                self.node_id = node_id

            def process(self, row: Any, ctx: Any) -> TransformResult:
                return TransformResult.success({**row, "field_a": f"a_{row['value']}"})

        class TransformB(BaseTransform):
            """Adds field_b."""

            name = "transform_b"
            input_schema = _TestSchema
            output_schema = _TestSchema

            def __init__(self, node_id: str) -> None:
                super().__init__({})
                self.node_id = node_id

            def process(self, row: Any, ctx: Any) -> TransformResult:
                return TransformResult.success({**row, "field_b": f"b_{row['value']}"})

        class TransformC(BaseTransform):
            """Adds field_c for low path."""

            name = "transform_c"
            input_schema = _TestSchema
            output_schema = _TestSchema

            def __init__(self, node_id: str) -> None:
                super().__init__({})
                self.node_id = node_id

            def process(self, row: Any, ctx: Any) -> TransformResult:
                return TransformResult.success({**row, "field_c": f"c_{row['value']}"})

        class BatchAggregation(BaseAggregation):
            """Batches rows, returns batch summary on flush."""

            name = "batch_aggregator"
            input_schema = _TestSchema
            output_schema = _TestSchema

            def __init__(self, node_id: str) -> None:
                super().__init__({})
                self.node_id = node_id
                self._batch: list[dict[str, Any]] = []

            def accept(self, row: dict[str, Any], ctx: Any) -> AcceptResult:
                self._batch.append(row)
                return AcceptResult(accepted=True)

            def flush(self, ctx: Any) -> list[dict[str, Any]]:
                values = [r["value"] for r in self._batch]
                result = [{"batch_values": values, "batch_size": len(self._batch)}]
                self._batch = []
                return result

        class CollectSink(_TestSinkBase):
            """Collects written rows."""

            name = "collect_sink"

            def __init__(self, node_id: str, sink_name: str) -> None:
                self.node_id = node_id
                self._sink_name = sink_name
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
                    path=f"memory://{self._sink_name}_{self._artifact_counter}",
                    size_bytes=len(str(rows)),
                    content_hash=f"{self._sink_name}_hash_{self._artifact_counter}",
                )

            def close(self) -> None:
                pass

        # ====================================================================
        # Create components
        # ====================================================================
        span_factory = SpanFactory()
        token_manager = TokenManager(recorder)

        transform_a = TransformA(transform_a_node.node_id)
        transform_b = TransformB(transform_b_node.node_id)
        transform_c = TransformC(transform_c_node.node_id)
        aggregation = BatchAggregation(aggregation_node.node_id)
        low_sink = CollectSink(low_sink_node.node_id, "low")
        high_sink = CollectSink(high_sink_node.node_id, "high")

        gate_executor = GateExecutor(
            recorder=recorder,
            span_factory=span_factory,
            edge_map=combined_edge_map,
            route_resolution_map=combined_route_resolution,
        )

        transform_executor = TransformExecutor(recorder, span_factory)

        coalesce_executor = CoalesceExecutor(
            recorder=recorder,
            span_factory=span_factory,
            token_manager=token_manager,
            run_id=run_id,
        )
        coalesce_executor.register_coalesce(coalesce_settings, coalesce_node.node_id)

        agg_executor = AggregationExecutor(
            recorder,
            span_factory,
            run_id,
            aggregation_settings={aggregation_node.node_id: agg_settings},
        )

        sink_executor = SinkExecutor(recorder, span_factory, run_id)

        ctx = PluginContext(run_id=run_id, config={})

        # ====================================================================
        # Process 10 rows (values 0-9)
        # ====================================================================
        low_path_tokens: list[TokenInfo] = []
        high_path_coalesced: list[TokenInfo] = []

        for i in range(10):
            source_row = {"value": i}

            # Create initial token from source
            initial_token = token_manager.create_initial_token(
                run_id=run_id,
                source_node_id=source_node.node_id,
                row_index=i,
                row_data=source_row,
            )

            # Execute threshold gate
            gate_outcome = gate_executor.execute_config_gate(
                gate_config=threshold_gate,
                node_id=threshold_gate_node.node_id,
                token=initial_token,
                ctx=ctx,
                step_in_pipeline=1,
                token_manager=token_manager,
            )

            if i < 5:
                # Low path: value 0-4
                # Route token through transform_c to low_sink
                result_c, token_c, _ = transform_executor.execute_transform(
                    transform=transform_c,
                    token=gate_outcome.updated_token,
                    ctx=ctx,
                    step_in_pipeline=2,
                )
                assert result_c.status == "success"
                low_path_tokens.append(
                    TokenInfo(
                        row_id=token_c.row_id,
                        token_id=token_c.token_id,
                        row_data=result_c.row,
                    )
                )
            else:
                # High path: value 5-9
                # Execute fork gate
                fork_outcome = gate_executor.execute_config_gate(
                    gate_config=fork_gate,
                    node_id=fork_gate_node.node_id,
                    token=gate_outcome.updated_token,
                    ctx=ctx,
                    step_in_pipeline=2,
                    token_manager=token_manager,
                )

                assert (
                    len(fork_outcome.child_tokens) == 2
                ), f"Fork should create 2 children, got {len(fork_outcome.child_tokens)}"

                # Get tokens for each path
                token_path_a = next(
                    t for t in fork_outcome.child_tokens if t.branch_name == "path_a"
                )
                token_path_b = next(
                    t for t in fork_outcome.child_tokens if t.branch_name == "path_b"
                )

                # Transform A
                result_a, token_a_updated, _ = transform_executor.execute_transform(
                    transform=transform_a,
                    token=token_path_a,
                    ctx=ctx,
                    step_in_pipeline=3,
                )
                assert result_a.status == "success"

                # Transform B
                result_b, token_b_updated, _ = transform_executor.execute_transform(
                    transform=transform_b,
                    token=token_path_b,
                    ctx=ctx,
                    step_in_pipeline=3,
                )
                assert result_b.status == "success"

                # Coalesce the branches
                processed_a = TokenInfo(
                    row_id=token_a_updated.row_id,
                    token_id=token_a_updated.token_id,
                    row_data=result_a.row,
                    branch_name="path_a",
                )
                processed_b = TokenInfo(
                    row_id=token_b_updated.row_id,
                    token_id=token_b_updated.token_id,
                    row_data=result_b.row,
                    branch_name="path_b",
                )

                outcome_a = coalesce_executor.accept(
                    token=processed_a,
                    coalesce_name="high_coalesce",
                    step_in_pipeline=4,
                )
                assert outcome_a.held is True, "First branch should be held"

                outcome_b = coalesce_executor.accept(
                    token=processed_b,
                    coalesce_name="high_coalesce",
                    step_in_pipeline=4,
                )
                assert outcome_b.held is False, "Second branch should trigger merge"
                assert outcome_b.merged_token is not None

                # Verify merged data has fields from both branches
                merged_data = outcome_b.merged_token.row_data
                assert (
                    "field_a" in merged_data
                ), f"Missing field_a in merged data: {merged_data}"
                assert (
                    "field_b" in merged_data
                ), f"Missing field_b in merged data: {merged_data}"

                high_path_coalesced.append(outcome_b.merged_token)

        # ====================================================================
        # Write low path rows to low_sink
        # ====================================================================
        assert (
            len(low_path_tokens) == 5
        ), f"Expected 5 low path tokens, got {len(low_path_tokens)}"
        sink_executor.write(
            sink=low_sink,
            tokens=low_path_tokens,
            ctx=ctx,
            step_in_pipeline=3,
        )

        # ====================================================================
        # Process aggregation and write to high_sink
        # ====================================================================
        assert (
            len(high_path_coalesced) == 5
        ), f"Expected 5 coalesced tokens, got {len(high_path_coalesced)}"

        high_sink_outputs: list[dict[str, Any]] = []

        for token in high_path_coalesced:
            # Accept into aggregation
            agg_executor.accept(
                aggregation=aggregation,
                token=token,
                ctx=ctx,
                step_in_pipeline=5,
            )

            # Check if should flush (count trigger)
            if agg_executor.should_flush(aggregation_node.node_id):
                outputs = agg_executor.flush(
                    aggregation=aggregation,
                    ctx=ctx,
                    trigger_reason="count_reached",
                    step_in_pipeline=5,
                )
                high_sink_outputs.extend(outputs)

        # Flush remaining rows at end of source
        if agg_executor.get_batch_id(aggregation_node.node_id) is not None:
            outputs = agg_executor.flush(
                aggregation=aggregation,
                ctx=ctx,
                trigger_reason="end_of_source",
                step_in_pipeline=5,
            )
            high_sink_outputs.extend(outputs)

        # Write aggregated outputs to high_sink
        # Create tokens for the aggregated outputs
        # Use row indices starting after the source rows (10+)
        agg_output_tokens = []
        for idx, output in enumerate(high_sink_outputs):
            agg_row_index = (
                10 + idx
            )  # Offset to avoid collision with source row indices
            agg_token = TokenInfo(
                row_id=f"agg_output_{idx}",
                token_id=f"agg_token_{idx}",
                row_data=output,
            )
            # Register row and token for audit
            row = recorder.create_row(
                run_id=run_id,
                source_node_id=aggregation_node.node_id,
                row_index=agg_row_index,
                data=output,
                row_id=agg_token.row_id,
            )
            recorder.create_token(row_id=row.row_id, token_id=agg_token.token_id)
            agg_output_tokens.append(agg_token)

        if agg_output_tokens:
            sink_executor.write(
                sink=high_sink,
                tokens=agg_output_tokens,
                ctx=ctx,
                step_in_pipeline=6,
            )

        # ====================================================================
        # Verify results
        # ====================================================================
        # Complete the run
        recorder.complete_run(run_id, status="completed")

        # Low sink: 5 rows with field_c
        assert (
            len(low_sink.rows_written) == 5
        ), f"Low sink should have 5 rows, got {len(low_sink.rows_written)}"
        for row in low_sink.rows_written:
            assert "field_c" in row, f"Low path row missing field_c: {row}"
            assert row["value"] < 5, f"Low path got high value: {row['value']}"

        # High sink: 3 aggregated outputs (2 batches of 2 + 1 end_of_source)
        # 5 high rows -> batched by 2 -> 2 full batches + 1 partial (1 row)
        assert (
            len(high_sink.rows_written) == 3
        ), f"High sink should have 3 aggregated outputs, got {len(high_sink.rows_written)}"

        # Verify aggregation batching
        batch_sizes = [row["batch_size"] for row in high_sink.rows_written]
        assert batch_sizes == [
            2,
            2,
            1,
        ], f"Expected batch sizes [2, 2, 1], got {batch_sizes}"

        # Verify all high values were processed
        all_values = []
        for row in high_sink.rows_written:
            all_values.extend(row["batch_values"])
        assert sorted(all_values) == [
            5,
            6,
            7,
            8,
            9,
        ], f"Expected values [5,6,7,8,9], got {sorted(all_values)}"

        # Verify audit trail
        rows = recorder.get_rows(run_id)
        # 10 source rows + 3 aggregation output rows = 13
        assert len(rows) >= 10, f"Expected at least 10 rows, got {len(rows)}"

        # Verify nodes were registered
        nodes = recorder.get_nodes(run_id)
        node_ids = {n.node_id for n in nodes}
        expected_nodes = {
            "source",
            "threshold_gate",
            "transform_c",
            "low_sink",
            "fork_gate",
            "transform_a",
            "transform_b",
            "high_coalesce",
            "high_agg",
            "high_sink",
        }
        assert expected_nodes <= node_ids, f"Missing nodes: {expected_nodes - node_ids}"
