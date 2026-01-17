# tests/engine/test_integration.py
"""Integration tests for engine module.

These tests verify:
1. All components can be imported from elspeth.engine
2. Full pipeline execution with audit trail verification
3. "Audit spine" tests proving every token reaches terminal state
4. "No silent audit loss" tests proving errors raise, not skip

All test plugins inherit from base classes (BaseTransform, BaseGate)
because the processor uses isinstance() for type-safe plugin detection.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import pytest

from elspeth.contracts import Determinism, PluginSchema, RoutingMode
from elspeth.plugins.base import BaseGate, BaseTransform

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
    source -> transforms... -> sinks

    For gates, creates additional edges to all sinks (gates can route anywhere).
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
        is_gate = isinstance(t, BaseGate)
        graph.add_node(
            node_id,
            node_type="gate" if is_gate else "transform",
            plugin_name=t.name,
        )
        graph.add_edge(prev, node_id, label="continue", mode=RoutingMode.MOVE)
        prev = node_id

    # Add sinks and populate sink_id_map
    sink_ids: dict[str, str] = {}
    for sink_name, sink in config.sinks.items():
        node_id = f"sink_{sink_name}"
        sink_ids[sink_name] = node_id
        graph.add_node(node_id, node_type="sink", plugin_name=sink.name)
        graph.add_edge(prev, node_id, label=sink_name, mode=RoutingMode.MOVE)

        # Gates can route to any sink, so add edges from all gates
        for i, t in enumerate(config.transforms):
            if isinstance(t, BaseGate):
                gate_id = f"transform_{i}"
                if gate_id != prev:  # Don't duplicate edge
                    graph.add_edge(
                        gate_id, node_id, label=sink_name, mode=RoutingMode.MOVE
                    )

    # Populate internal ID maps so get_sink_id_map() and get_transform_id_map() work
    graph._sink_id_map = sink_ids
    graph._transform_id_map = transform_ids

    # Populate route resolution map: (gate_id, label) -> sink_name
    # In test graphs, route labels = sink names for simplicity
    route_resolution_map: dict[tuple[str, str], str] = {}
    for i, t in enumerate(config.transforms):
        if isinstance(t, BaseGate):  # It's a gate
            gate_id = f"transform_{i}"
            for sink_name in sink_ids:
                route_resolution_map[(gate_id, sink_name)] = sink_name
    graph._route_resolution_map = route_resolution_map

    # Set output_sink - use "default" if present, otherwise first sink
    if "default" in sink_ids:
        graph._output_sink = "default"
    elif sink_ids:
        graph._output_sink = next(iter(sink_ids))

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
        from elspeth.plugins.results import GateResult, RoutingAction

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

        class EvenOddGate(BaseGate):
            name = "even_odd_gate"
            input_schema = NumberSchema
            output_schema = NumberSchema

            def __init__(self) -> None:
                super().__init__({})

            def evaluate(self, row: Any, ctx: Any) -> GateResult:
                if row["value"] % 2 == 0:
                    return GateResult(
                        row=row,
                        action=RoutingAction.route(
                            "even"
                        ),  # Route label (same as sink name in test)
                    )
                return GateResult(row=row, action=RoutingAction.continue_())

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

        # Pipeline with routing gate
        source = ListSource([{"value": 1}, {"value": 2}, {"value": 3}, {"value": 4}])
        gate = EvenOddGate()
        default_sink = CollectSink("default_sink")
        even_sink = CollectSink("even_sink")

        config = PipelineConfig(
            source=source,
            transforms=[gate],
            sinks={"default": default_sink, "even": even_sink},
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
        """Critical: MissingEdgeError must raise, not silently count.

        This test ensures that when a gate routes to a sink that doesn't
        have a registered edge, the error is raised immediately rather
        than being silently counted as a failure.
        """
        from elspeth.contracts import PluginSchema
        from elspeth.core.landscape import LandscapeDB
        from elspeth.engine import MissingEdgeError, Orchestrator, PipelineConfig
        from elspeth.engine.artifacts import ArtifactDescriptor
        from elspeth.plugins.results import GateResult, RoutingAction

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

        class MisroutingGate(BaseGate):
            name = "misrouting_gate"
            input_schema = RowSchema
            output_schema = RowSchema

            def __init__(self) -> None:
                super().__init__({})

            def evaluate(self, row: Any, ctx: Any) -> GateResult:
                # Route to "phantom" which is not in route_resolution_map
                return GateResult(
                    row=row,
                    action=RoutingAction.route("phantom"),
                )

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

        source = ListSource([{"value": 42}])
        gate = MisroutingGate()
        sink = CollectSink()

        config = PipelineConfig(
            source=source,
            transforms=[gate],
            sinks={"default": sink},  # Note: "phantom" is NOT configured
        )

        orchestrator = Orchestrator(db)

        # This MUST raise MissingEdgeError, not silently fail
        with pytest.raises(MissingEdgeError) as exc_info:
            orchestrator.run(config, graph=_build_test_graph(config))

        # Verify error message includes the missing edge label
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
        from elspeth.plugins.results import GateResult, RoutingAction

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

        class SplitGate(BaseGate):
            name = "split_gate"
            input_schema = ValueSchema
            output_schema = ValueSchema

            def __init__(self) -> None:
                super().__init__({})

            def evaluate(self, row: Any, ctx: Any) -> GateResult:
                if row["value"] > 50:
                    return GateResult(
                        row=row, action=RoutingAction.route("high")
                    )  # Route label
                return GateResult(row=row, action=RoutingAction.continue_())

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

        source = ListSource(
            [{"value": 10}, {"value": 60}, {"value": 30}, {"value": 90}]
        )
        gate = SplitGate()
        default_sink = CollectSink("default_output")
        high_sink = CollectSink("high_output")

        config = PipelineConfig(
            source=source,
            transforms=[gate],
            sinks={"default": default_sink, "high": high_sink},
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
