# tests/engine/test_processor.py
"""Tests for RowProcessor."""

from typing import Any


class TestRowProcessor:
    """Row processing through pipeline."""

    def test_process_through_transforms(self) -> None:
        from elspeth.core.landscape import LandscapeDB, LandscapeRecorder
        from elspeth.engine.processor import RowProcessor
        from elspeth.engine.spans import SpanFactory
        from elspeth.plugins.context import PluginContext
        from elspeth.plugins.results import TransformResult

        db = LandscapeDB.in_memory()
        recorder = LandscapeRecorder(db)
        run = recorder.begin_run(config={}, canonical_version="v1")

        # Register nodes
        source = recorder.register_node(
            run_id=run.run_id,
            plugin_name="source",
            node_type="source",
            plugin_version="1.0",
            config={},
        )
        transform1 = recorder.register_node(
            run_id=run.run_id,
            plugin_name="double",
            node_type="transform",
            plugin_version="1.0",
            config={},
        )
        transform2 = recorder.register_node(
            run_id=run.run_id,
            plugin_name="add_one",
            node_type="transform",
            plugin_version="1.0",
            config={},
        )

        class DoubleTransform:
            name = "double"
            node_id = transform1.node_id

            def process(self, row: dict[str, Any], ctx: PluginContext) -> TransformResult:
                return TransformResult.success({"value": row["value"] * 2})

        class AddOneTransform:
            name = "add_one"
            node_id = transform2.node_id

            def process(self, row: dict[str, Any], ctx: PluginContext) -> TransformResult:
                return TransformResult.success({"value": row["value"] + 1})

        ctx = PluginContext(run_id=run.run_id, config={})
        processor = RowProcessor(
            recorder=recorder,
            span_factory=SpanFactory(),
            run_id=run.run_id,
            source_node_id=source.node_id,
        )

        result = processor.process_row(
            row_index=0,
            row_data={"value": 10},
            transforms=[DoubleTransform(), AddOneTransform()],
            ctx=ctx,
        )

        # 10 * 2 = 20, 20 + 1 = 21
        assert result.final_data == {"value": 21}
        assert result.outcome == "completed"

    def test_process_single_transform(self) -> None:
        """Single transform processes correctly."""
        from elspeth.core.landscape import LandscapeDB, LandscapeRecorder
        from elspeth.engine.processor import RowProcessor
        from elspeth.engine.spans import SpanFactory
        from elspeth.plugins.context import PluginContext
        from elspeth.plugins.results import TransformResult

        db = LandscapeDB.in_memory()
        recorder = LandscapeRecorder(db)
        run = recorder.begin_run(config={}, canonical_version="v1")

        source = recorder.register_node(
            run_id=run.run_id,
            plugin_name="source",
            node_type="source",
            plugin_version="1.0",
            config={},
        )
        transform = recorder.register_node(
            run_id=run.run_id,
            plugin_name="enricher",
            node_type="transform",
            plugin_version="1.0",
            config={},
        )

        class EnricherTransform:
            name = "enricher"
            node_id = transform.node_id

            def process(self, row: dict[str, Any], ctx: PluginContext) -> TransformResult:
                return TransformResult.success({**row, "enriched": True})

        ctx = PluginContext(run_id=run.run_id, config={})
        processor = RowProcessor(
            recorder=recorder,
            span_factory=SpanFactory(),
            run_id=run.run_id,
            source_node_id=source.node_id,
        )

        result = processor.process_row(
            row_index=0,
            row_data={"name": "test"},
            transforms=[EnricherTransform()],
            ctx=ctx,
        )

        assert result.final_data == {"name": "test", "enriched": True}
        assert result.outcome == "completed"
        # Check identity preserved
        assert result.token_id is not None
        assert result.row_id is not None

    def test_process_no_transforms(self) -> None:
        """No transforms passes through data unchanged."""
        from elspeth.core.landscape import LandscapeDB, LandscapeRecorder
        from elspeth.engine.processor import RowProcessor
        from elspeth.engine.spans import SpanFactory
        from elspeth.plugins.context import PluginContext

        db = LandscapeDB.in_memory()
        recorder = LandscapeRecorder(db)
        run = recorder.begin_run(config={}, canonical_version="v1")

        source = recorder.register_node(
            run_id=run.run_id,
            plugin_name="source",
            node_type="source",
            plugin_version="1.0",
            config={},
        )

        ctx = PluginContext(run_id=run.run_id, config={})
        processor = RowProcessor(
            recorder=recorder,
            span_factory=SpanFactory(),
            run_id=run.run_id,
            source_node_id=source.node_id,
        )

        result = processor.process_row(
            row_index=0,
            row_data={"passthrough": True},
            transforms=[],
            ctx=ctx,
        )

        assert result.final_data == {"passthrough": True}
        assert result.outcome == "completed"

    def test_transform_error_returns_failed(self) -> None:
        """Transform returning error status causes failed outcome."""
        from elspeth.core.landscape import LandscapeDB, LandscapeRecorder
        from elspeth.engine.processor import RowProcessor
        from elspeth.engine.spans import SpanFactory
        from elspeth.plugins.context import PluginContext
        from elspeth.plugins.results import TransformResult

        db = LandscapeDB.in_memory()
        recorder = LandscapeRecorder(db)
        run = recorder.begin_run(config={}, canonical_version="v1")

        source = recorder.register_node(
            run_id=run.run_id,
            plugin_name="source",
            node_type="source",
            plugin_version="1.0",
            config={},
        )
        transform = recorder.register_node(
            run_id=run.run_id,
            plugin_name="validator",
            node_type="transform",
            plugin_version="1.0",
            config={},
        )

        class ValidatorTransform:
            name = "validator"
            node_id = transform.node_id

            def process(self, row: dict[str, Any], ctx: PluginContext) -> TransformResult:
                if row.get("value", 0) < 0:
                    return TransformResult.error({"message": "negative values not allowed"})
                return TransformResult.success(row)

        ctx = PluginContext(run_id=run.run_id, config={})
        processor = RowProcessor(
            recorder=recorder,
            span_factory=SpanFactory(),
            run_id=run.run_id,
            source_node_id=source.node_id,
        )

        result = processor.process_row(
            row_index=0,
            row_data={"value": -5},
            transforms=[ValidatorTransform()],
            ctx=ctx,
        )

        assert result.outcome == "failed"
        # Original data preserved on failure
        assert result.final_data == {"value": -5}


class TestRowProcessorGates:
    """Gate handling in RowProcessor."""

    def test_gate_continue_proceeds(self) -> None:
        """Gate returning continue proceeds to next transform."""
        from elspeth.core.landscape import LandscapeDB, LandscapeRecorder
        from elspeth.engine.processor import RowProcessor
        from elspeth.engine.spans import SpanFactory
        from elspeth.plugins.context import PluginContext
        from elspeth.plugins.results import GateResult, RoutingAction, TransformResult

        db = LandscapeDB.in_memory()
        recorder = LandscapeRecorder(db)
        run = recorder.begin_run(config={}, canonical_version="v1")

        source = recorder.register_node(
            run_id=run.run_id,
            plugin_name="source",
            node_type="source",
            plugin_version="1.0",
            config={},
        )
        gate = recorder.register_node(
            run_id=run.run_id,
            plugin_name="pass_gate",
            node_type="gate",
            plugin_version="1.0",
            config={},
        )
        transform = recorder.register_node(
            run_id=run.run_id,
            plugin_name="final",
            node_type="transform",
            plugin_version="1.0",
            config={},
        )

        class PassGate:
            name = "pass_gate"
            node_id = gate.node_id

            def evaluate(self, row: dict[str, Any], ctx: PluginContext) -> GateResult:
                return GateResult(row=row, action=RoutingAction.continue_())

        class FinalTransform:
            name = "final"
            node_id = transform.node_id

            def process(self, row: dict[str, Any], ctx: PluginContext) -> TransformResult:
                return TransformResult.success({**row, "final": True})

        ctx = PluginContext(run_id=run.run_id, config={})
        processor = RowProcessor(
            recorder=recorder,
            span_factory=SpanFactory(),
            run_id=run.run_id,
            source_node_id=source.node_id,
        )

        result = processor.process_row(
            row_index=0,
            row_data={"value": 42},
            transforms=[PassGate(), FinalTransform()],
            ctx=ctx,
        )

        assert result.final_data == {"value": 42, "final": True}
        assert result.outcome == "completed"

    def test_gate_route_to_sink(self) -> None:
        """Gate routing via route label returns routed outcome with sink name."""
        from elspeth.core.landscape import LandscapeDB, LandscapeRecorder
        from elspeth.engine.processor import RowProcessor
        from elspeth.engine.spans import SpanFactory
        from elspeth.plugins.context import PluginContext
        from elspeth.plugins.results import GateResult, RoutingAction

        db = LandscapeDB.in_memory()
        recorder = LandscapeRecorder(db)
        run = recorder.begin_run(config={}, canonical_version="v1")

        source = recorder.register_node(
            run_id=run.run_id,
            plugin_name="source",
            node_type="source",
            plugin_version="1.0",
            config={},
        )
        gate = recorder.register_node(
            run_id=run.run_id,
            plugin_name="router",
            node_type="gate",
            plugin_version="1.0",
            config={},
        )
        sink = recorder.register_node(
            run_id=run.run_id,
            plugin_name="high_values",
            node_type="sink",
            plugin_version="1.0",
            config={},
        )

        # Register edge using route label
        edge = recorder.register_edge(
            run_id=run.run_id,
            from_node_id=gate.node_id,
            to_node_id=sink.node_id,
            label="above",  # Route label, not sink name
            mode="move",
        )

        class RouterGate:
            name = "router"
            node_id = gate.node_id

            def evaluate(self, row: dict[str, Any], ctx: PluginContext) -> GateResult:
                if row.get("value", 0) > 100:
                    return GateResult(
                        row=row,
                        action=RoutingAction.route("above"),  # Route label
                    )
                return GateResult(row=row, action=RoutingAction.continue_())

        ctx = PluginContext(run_id=run.run_id, config={})
        edge_map = {(gate.node_id, "above"): edge.edge_id}
        # Route resolution map: label -> sink_name
        route_resolution_map = {(gate.node_id, "above"): "high_values"}
        processor = RowProcessor(
            recorder=recorder,
            span_factory=SpanFactory(),
            run_id=run.run_id,
            source_node_id=source.node_id,
            edge_map=edge_map,
            route_resolution_map=route_resolution_map,
        )

        result = processor.process_row(
            row_index=0,
            row_data={"value": 150},
            transforms=[RouterGate()],
            ctx=ctx,
        )

        assert result.outcome == "routed"
        assert result.sink_name == "high_values"
        assert result.final_data == {"value": 150}

    def test_gate_fork_returns_forked(self) -> None:
        """Gate forking returns forked outcome (linear pipeline mode)."""
        from elspeth.core.landscape import LandscapeDB, LandscapeRecorder
        from elspeth.engine.processor import RowProcessor
        from elspeth.engine.spans import SpanFactory
        from elspeth.plugins.context import PluginContext
        from elspeth.plugins.results import GateResult, RoutingAction

        db = LandscapeDB.in_memory()
        recorder = LandscapeRecorder(db)
        run = recorder.begin_run(config={}, canonical_version="v1")

        source = recorder.register_node(
            run_id=run.run_id,
            plugin_name="source",
            node_type="source",
            plugin_version="1.0",
            config={},
        )
        gate = recorder.register_node(
            run_id=run.run_id,
            plugin_name="splitter",
            node_type="gate",
            plugin_version="1.0",
            config={},
        )
        path_a = recorder.register_node(
            run_id=run.run_id,
            plugin_name="path_a",
            node_type="transform",
            plugin_version="1.0",
            config={},
        )
        path_b = recorder.register_node(
            run_id=run.run_id,
            plugin_name="path_b",
            node_type="transform",
            plugin_version="1.0",
            config={},
        )

        # Register edges for fork
        edge_a = recorder.register_edge(
            run_id=run.run_id,
            from_node_id=gate.node_id,
            to_node_id=path_a.node_id,
            label="path_a",
            mode="copy",
        )
        edge_b = recorder.register_edge(
            run_id=run.run_id,
            from_node_id=gate.node_id,
            to_node_id=path_b.node_id,
            label="path_b",
            mode="copy",
        )

        class SplitterGate:
            name = "splitter"
            node_id = gate.node_id

            def evaluate(self, row: dict[str, Any], ctx: PluginContext) -> GateResult:
                return GateResult(
                    row=row,
                    action=RoutingAction.fork_to_paths(["path_a", "path_b"]),
                )

        ctx = PluginContext(run_id=run.run_id, config={})
        edge_map = {
            (gate.node_id, "path_a"): edge_a.edge_id,
            (gate.node_id, "path_b"): edge_b.edge_id,
        }
        processor = RowProcessor(
            recorder=recorder,
            span_factory=SpanFactory(),
            run_id=run.run_id,
            source_node_id=source.node_id,
            edge_map=edge_map,
        )

        result = processor.process_row(
            row_index=0,
            row_data={"value": 42},
            transforms=[SplitterGate()],
            ctx=ctx,
        )

        # In linear pipeline mode, fork returns "forked" - caller handles children
        assert result.outcome == "forked"
        assert result.final_data == {"value": 42}


class TestRowProcessorAggregation:
    """Aggregation handling in RowProcessor."""

    def test_aggregation_consumes_row(self) -> None:
        """Row accepted by aggregation returns consumed outcome."""
        from elspeth.core.landscape import LandscapeDB, LandscapeRecorder
        from elspeth.engine.processor import RowProcessor
        from elspeth.engine.spans import SpanFactory
        from elspeth.plugins.context import PluginContext
        from elspeth.plugins.results import AcceptResult

        db = LandscapeDB.in_memory()
        recorder = LandscapeRecorder(db)
        run = recorder.begin_run(config={}, canonical_version="v1")

        source = recorder.register_node(
            run_id=run.run_id,
            plugin_name="source",
            node_type="source",
            plugin_version="1.0",
            config={},
        )
        agg = recorder.register_node(
            run_id=run.run_id,
            plugin_name="counter",
            node_type="aggregation",
            plugin_version="1.0",
            config={},
        )

        class CounterAggregation:
            name = "counter"
            node_id = agg.node_id
            _batch_id: str | None = None
            _count: int = 0

            def accept(self, row: dict[str, Any], ctx: PluginContext) -> AcceptResult:
                self._count += 1
                return AcceptResult(accepted=True, trigger=False)

            def flush(self, ctx: PluginContext) -> list[dict[str, Any]]:
                result = [{"count": self._count}]
                self._count = 0
                return result

        ctx = PluginContext(run_id=run.run_id, config={})
        processor = RowProcessor(
            recorder=recorder,
            span_factory=SpanFactory(),
            run_id=run.run_id,
            source_node_id=source.node_id,
        )

        result = processor.process_row(
            row_index=0,
            row_data={"value": 1},
            transforms=[CounterAggregation()],
            ctx=ctx,
        )

        assert result.outcome == "consumed"

    def test_aggregation_trigger_flushes(self) -> None:
        """Aggregation trigger causes flush to be called."""
        from elspeth.core.landscape import LandscapeDB, LandscapeRecorder
        from elspeth.engine.processor import RowProcessor
        from elspeth.engine.spans import SpanFactory
        from elspeth.plugins.context import PluginContext
        from elspeth.plugins.results import AcceptResult

        db = LandscapeDB.in_memory()
        recorder = LandscapeRecorder(db)
        run = recorder.begin_run(config={}, canonical_version="v1")

        source = recorder.register_node(
            run_id=run.run_id,
            plugin_name="source",
            node_type="source",
            plugin_version="1.0",
            config={},
        )
        agg = recorder.register_node(
            run_id=run.run_id,
            plugin_name="threshold_agg",
            node_type="aggregation",
            plugin_version="1.0",
            config={},
        )

        flush_called = []

        class ThresholdAggregation:
            name = "threshold_agg"
            node_id = agg.node_id
            _batch_id: str | None = None
            _values: list[int]

            def __init__(self) -> None:
                self._values = []

            def accept(self, row: dict[str, Any], ctx: PluginContext) -> AcceptResult:
                self._values.append(row["value"])
                # Trigger when we have 2 values
                return AcceptResult(accepted=True, trigger=len(self._values) >= 2)

            def flush(self, ctx: PluginContext) -> list[dict[str, Any]]:
                flush_called.append(True)
                result = [{"sum": sum(self._values)}]
                self._values = []
                return result

        ctx = PluginContext(run_id=run.run_id, config={})
        aggregation = ThresholdAggregation()
        processor = RowProcessor(
            recorder=recorder,
            span_factory=SpanFactory(),
            run_id=run.run_id,
            source_node_id=source.node_id,
        )

        # First row - no trigger
        result1 = processor.process_row(
            row_index=0,
            row_data={"value": 10},
            transforms=[aggregation],
            ctx=ctx,
        )
        assert result1.outcome == "consumed"
        assert len(flush_called) == 0

        # Second row - triggers
        result2 = processor.process_row(
            row_index=1,
            row_data={"value": 20},
            transforms=[aggregation],
            ctx=ctx,
        )
        assert result2.outcome == "consumed"
        assert len(flush_called) == 1  # Flush was called


class TestRowProcessorTokenIdentity:
    """Token identity is preserved and accessible."""

    def test_token_accessible_on_result(self) -> None:
        """RowResult provides access to full token info."""
        from elspeth.core.landscape import LandscapeDB, LandscapeRecorder
        from elspeth.engine.processor import RowProcessor
        from elspeth.engine.spans import SpanFactory
        from elspeth.plugins.context import PluginContext

        db = LandscapeDB.in_memory()
        recorder = LandscapeRecorder(db)
        run = recorder.begin_run(config={}, canonical_version="v1")

        source = recorder.register_node(
            run_id=run.run_id,
            plugin_name="source",
            node_type="source",
            plugin_version="1.0",
            config={},
        )

        ctx = PluginContext(run_id=run.run_id, config={})
        processor = RowProcessor(
            recorder=recorder,
            span_factory=SpanFactory(),
            run_id=run.run_id,
            source_node_id=source.node_id,
        )

        result = processor.process_row(
            row_index=0,
            row_data={"test": "data"},
            transforms=[],
            ctx=ctx,
        )

        # Can access token identity
        assert result.token is not None
        assert result.token.token_id is not None
        assert result.token.row_id is not None
        assert result.token.row_data == {"test": "data"}

        # Convenience properties work
        assert result.token_id == result.token.token_id
        assert result.row_id == result.token.row_id

    def test_step_counting_correct(self) -> None:
        """Step position is tracked correctly through pipeline."""
        from elspeth.core.landscape import LandscapeDB, LandscapeRecorder
        from elspeth.engine.processor import RowProcessor
        from elspeth.engine.spans import SpanFactory
        from elspeth.plugins.context import PluginContext
        from elspeth.plugins.results import TransformResult

        db = LandscapeDB.in_memory()
        recorder = LandscapeRecorder(db)
        run = recorder.begin_run(config={}, canonical_version="v1")

        source = recorder.register_node(
            run_id=run.run_id,
            plugin_name="source",
            node_type="source",
            plugin_version="1.0",
            config={},
        )
        transform1 = recorder.register_node(
            run_id=run.run_id,
            plugin_name="t1",
            node_type="transform",
            plugin_version="1.0",
            config={},
        )
        transform2 = recorder.register_node(
            run_id=run.run_id,
            plugin_name="t2",
            node_type="transform",
            plugin_version="1.0",
            config={},
        )

        class IdentityTransform:
            def __init__(self, name: str, node_id: str) -> None:
                self.name = name
                self.node_id = node_id

            def process(self, row: dict[str, Any], ctx: PluginContext) -> TransformResult:
                return TransformResult.success(row)

        ctx = PluginContext(run_id=run.run_id, config={})
        processor = RowProcessor(
            recorder=recorder,
            span_factory=SpanFactory(),
            run_id=run.run_id,
            source_node_id=source.node_id,
        )

        result = processor.process_row(
            row_index=0,
            row_data={"value": 1},
            transforms=[
                IdentityTransform("t1", transform1.node_id),
                IdentityTransform("t2", transform2.node_id),
            ],
            ctx=ctx,
        )

        assert result.outcome == "completed"

        # Verify node states recorded with correct step indices
        states = recorder.get_node_states_for_token(result.token_id)
        assert len(states) == 2
        # Steps should be 1 and 2 (source is 0, transforms start at 1)
        step_indices = {s.step_index for s in states}
        assert step_indices == {1, 2}
