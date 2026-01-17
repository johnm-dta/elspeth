# tests/engine/test_processor.py
"""Tests for RowProcessor.

All test plugins inherit from base classes (BaseTransform, BaseGate, BaseAggregation)
because the processor uses isinstance() for type-safe plugin detection.
"""

from typing import Any

from elspeth.contracts import PluginSchema, RoutingMode
from elspeth.contracts.schema import SchemaConfig
from elspeth.plugins.base import BaseAggregation, BaseGate, BaseTransform
from elspeth.plugins.context import PluginContext
from elspeth.plugins.results import (
    AcceptResult,
    GateResult,
    RoutingAction,
    RowOutcome,
    TransformResult,
)

# Dynamic schema for tests that don't care about specific fields
DYNAMIC_SCHEMA = SchemaConfig.from_dict({"fields": "dynamic"})


# Shared schema for test plugins
class _TestSchema(PluginSchema):
    """Dynamic schema for test plugins."""

    model_config = {"extra": "allow"}  # noqa: RUF012


class TestRowProcessor:
    """Row processing through pipeline."""

    def test_process_through_transforms(self) -> None:
        from elspeth.core.landscape import LandscapeDB, LandscapeRecorder
        from elspeth.engine.processor import RowProcessor
        from elspeth.engine.spans import SpanFactory

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
            schema_config=DYNAMIC_SCHEMA,
        )
        transform1 = recorder.register_node(
            run_id=run.run_id,
            plugin_name="double",
            node_type="transform",
            plugin_version="1.0",
            config={},
            schema_config=DYNAMIC_SCHEMA,
        )
        transform2 = recorder.register_node(
            run_id=run.run_id,
            plugin_name="add_one",
            node_type="transform",
            plugin_version="1.0",
            config={},
            schema_config=DYNAMIC_SCHEMA,
        )

        class DoubleTransform(BaseTransform):
            name = "double"
            input_schema = _TestSchema
            output_schema = _TestSchema

            def __init__(self, node_id: str) -> None:
                super().__init__({})
                self.node_id = node_id

            def process(
                self, row: dict[str, Any], ctx: PluginContext
            ) -> TransformResult:
                return TransformResult.success({"value": row["value"] * 2})

        class AddOneTransform(BaseTransform):
            name = "add_one"
            input_schema = _TestSchema
            output_schema = _TestSchema

            def __init__(self, node_id: str) -> None:
                super().__init__({})
                self.node_id = node_id

            def process(
                self, row: dict[str, Any], ctx: PluginContext
            ) -> TransformResult:
                return TransformResult.success({"value": row["value"] + 1})

        ctx = PluginContext(run_id=run.run_id, config={})
        processor = RowProcessor(
            recorder=recorder,
            span_factory=SpanFactory(),
            run_id=run.run_id,
            source_node_id=source.node_id,
        )

        results = processor.process_row(
            row_index=0,
            row_data={"value": 10},
            transforms=[
                DoubleTransform(transform1.node_id),
                AddOneTransform(transform2.node_id),
            ],
            ctx=ctx,
        )

        # Single result - no forks
        assert len(results) == 1
        result = results[0]

        # 10 * 2 = 20, 20 + 1 = 21
        assert result.final_data == {"value": 21}
        assert result.outcome == RowOutcome.COMPLETED

    def test_process_single_transform(self) -> None:
        """Single transform processes correctly."""
        from elspeth.core.landscape import LandscapeDB, LandscapeRecorder
        from elspeth.engine.processor import RowProcessor
        from elspeth.engine.spans import SpanFactory

        db = LandscapeDB.in_memory()
        recorder = LandscapeRecorder(db)
        run = recorder.begin_run(config={}, canonical_version="v1")

        source = recorder.register_node(
            run_id=run.run_id,
            plugin_name="source",
            node_type="source",
            plugin_version="1.0",
            config={},
            schema_config=DYNAMIC_SCHEMA,
        )
        transform = recorder.register_node(
            run_id=run.run_id,
            plugin_name="enricher",
            node_type="transform",
            plugin_version="1.0",
            config={},
            schema_config=DYNAMIC_SCHEMA,
        )

        class EnricherTransform(BaseTransform):
            name = "enricher"
            input_schema = _TestSchema
            output_schema = _TestSchema

            def __init__(self, node_id: str) -> None:
                super().__init__({})
                self.node_id = node_id

            def process(
                self, row: dict[str, Any], ctx: PluginContext
            ) -> TransformResult:
                return TransformResult.success({**row, "enriched": True})

        ctx = PluginContext(run_id=run.run_id, config={})
        processor = RowProcessor(
            recorder=recorder,
            span_factory=SpanFactory(),
            run_id=run.run_id,
            source_node_id=source.node_id,
        )

        results = processor.process_row(
            row_index=0,
            row_data={"name": "test"},
            transforms=[EnricherTransform(transform.node_id)],
            ctx=ctx,
        )

        # Single result - no forks
        assert len(results) == 1
        result = results[0]

        assert result.final_data == {"name": "test", "enriched": True}
        assert result.outcome == RowOutcome.COMPLETED
        # Check identity preserved
        assert result.token_id is not None
        assert result.row_id is not None

    def test_process_no_transforms(self) -> None:
        """No transforms passes through data unchanged."""
        from elspeth.core.landscape import LandscapeDB, LandscapeRecorder
        from elspeth.engine.processor import RowProcessor
        from elspeth.engine.spans import SpanFactory

        db = LandscapeDB.in_memory()
        recorder = LandscapeRecorder(db)
        run = recorder.begin_run(config={}, canonical_version="v1")

        source = recorder.register_node(
            run_id=run.run_id,
            plugin_name="source",
            node_type="source",
            plugin_version="1.0",
            config={},
            schema_config=DYNAMIC_SCHEMA,
        )

        ctx = PluginContext(run_id=run.run_id, config={})
        processor = RowProcessor(
            recorder=recorder,
            span_factory=SpanFactory(),
            run_id=run.run_id,
            source_node_id=source.node_id,
        )

        results = processor.process_row(
            row_index=0,
            row_data={"passthrough": True},
            transforms=[],
            ctx=ctx,
        )

        # Single result - no forks
        assert len(results) == 1
        result = results[0]

        assert result.final_data == {"passthrough": True}
        assert result.outcome == RowOutcome.COMPLETED

    def test_transform_error_returns_failed(self) -> None:
        """Transform returning error status causes failed outcome."""
        from elspeth.core.landscape import LandscapeDB, LandscapeRecorder
        from elspeth.engine.processor import RowProcessor
        from elspeth.engine.spans import SpanFactory

        db = LandscapeDB.in_memory()
        recorder = LandscapeRecorder(db)
        run = recorder.begin_run(config={}, canonical_version="v1")

        source = recorder.register_node(
            run_id=run.run_id,
            plugin_name="source",
            node_type="source",
            plugin_version="1.0",
            config={},
            schema_config=DYNAMIC_SCHEMA,
        )
        transform = recorder.register_node(
            run_id=run.run_id,
            plugin_name="validator",
            node_type="transform",
            plugin_version="1.0",
            config={},
            schema_config=DYNAMIC_SCHEMA,
        )

        class ValidatorTransform(BaseTransform):
            name = "validator"
            input_schema = _TestSchema
            output_schema = _TestSchema
            _on_error = "discard"  # Required for transforms that return errors

            def __init__(self, node_id: str) -> None:
                super().__init__({})
                self.node_id = node_id

            def process(
                self, row: dict[str, Any], ctx: PluginContext
            ) -> TransformResult:
                if row.get("value", 0) < 0:
                    return TransformResult.error(
                        {"message": "negative values not allowed"}
                    )
                return TransformResult.success(row)

        ctx = PluginContext(run_id=run.run_id, config={}, landscape=recorder)
        processor = RowProcessor(
            recorder=recorder,
            span_factory=SpanFactory(),
            run_id=run.run_id,
            source_node_id=source.node_id,
        )

        results = processor.process_row(
            row_index=0,
            row_data={"value": -5},
            transforms=[ValidatorTransform(transform.node_id)],
            ctx=ctx,
        )

        # Single result - no forks
        assert len(results) == 1
        result = results[0]

        assert result.outcome == RowOutcome.FAILED
        # Original data preserved on failure
        assert result.final_data == {"value": -5}


class TestRowProcessorGates:
    """Gate handling in RowProcessor."""

    def test_gate_continue_proceeds(self) -> None:
        """Gate returning continue proceeds to next transform."""
        from elspeth.core.landscape import LandscapeDB, LandscapeRecorder
        from elspeth.engine.processor import RowProcessor
        from elspeth.engine.spans import SpanFactory

        db = LandscapeDB.in_memory()
        recorder = LandscapeRecorder(db)
        run = recorder.begin_run(config={}, canonical_version="v1")

        source = recorder.register_node(
            run_id=run.run_id,
            plugin_name="source",
            node_type="source",
            plugin_version="1.0",
            config={},
            schema_config=DYNAMIC_SCHEMA,
        )
        gate = recorder.register_node(
            run_id=run.run_id,
            plugin_name="pass_gate",
            node_type="gate",
            plugin_version="1.0",
            config={},
            schema_config=DYNAMIC_SCHEMA,
        )
        transform = recorder.register_node(
            run_id=run.run_id,
            plugin_name="final",
            node_type="transform",
            plugin_version="1.0",
            config={},
            schema_config=DYNAMIC_SCHEMA,
        )

        class PassGate(BaseGate):
            name = "pass_gate"
            input_schema = _TestSchema
            output_schema = _TestSchema

            def __init__(self, node_id: str) -> None:
                super().__init__({})
                self.node_id = node_id

            def evaluate(self, row: dict[str, Any], ctx: PluginContext) -> GateResult:
                return GateResult(row=row, action=RoutingAction.continue_())

        class FinalTransform(BaseTransform):
            name = "final"
            input_schema = _TestSchema
            output_schema = _TestSchema

            def __init__(self, node_id: str) -> None:
                super().__init__({})
                self.node_id = node_id

            def process(
                self, row: dict[str, Any], ctx: PluginContext
            ) -> TransformResult:
                return TransformResult.success({**row, "final": True})

        ctx = PluginContext(run_id=run.run_id, config={})
        processor = RowProcessor(
            recorder=recorder,
            span_factory=SpanFactory(),
            run_id=run.run_id,
            source_node_id=source.node_id,
        )

        results = processor.process_row(
            row_index=0,
            row_data={"value": 42},
            transforms=[PassGate(gate.node_id), FinalTransform(transform.node_id)],
            ctx=ctx,
        )

        # Single result - no forks
        assert len(results) == 1
        result = results[0]

        assert result.final_data == {"value": 42, "final": True}
        assert result.outcome == RowOutcome.COMPLETED

    def test_gate_route_to_sink(self) -> None:
        """Gate routing via route label returns routed outcome with sink name."""
        from elspeth.core.landscape import LandscapeDB, LandscapeRecorder
        from elspeth.engine.processor import RowProcessor
        from elspeth.engine.spans import SpanFactory

        db = LandscapeDB.in_memory()
        recorder = LandscapeRecorder(db)
        run = recorder.begin_run(config={}, canonical_version="v1")

        source = recorder.register_node(
            run_id=run.run_id,
            plugin_name="source",
            node_type="source",
            plugin_version="1.0",
            config={},
            schema_config=DYNAMIC_SCHEMA,
        )
        gate = recorder.register_node(
            run_id=run.run_id,
            plugin_name="router",
            node_type="gate",
            plugin_version="1.0",
            config={},
            schema_config=DYNAMIC_SCHEMA,
        )
        sink = recorder.register_node(
            run_id=run.run_id,
            plugin_name="high_values",
            node_type="sink",
            plugin_version="1.0",
            config={},
            schema_config=DYNAMIC_SCHEMA,
        )

        # Register edge using route label
        edge = recorder.register_edge(
            run_id=run.run_id,
            from_node_id=gate.node_id,
            to_node_id=sink.node_id,
            label="above",  # Route label, not sink name
            mode=RoutingMode.MOVE,
        )

        class RouterGate(BaseGate):
            name = "router"
            input_schema = _TestSchema
            output_schema = _TestSchema

            def __init__(self, node_id: str) -> None:
                super().__init__({})
                self.node_id = node_id

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

        results = processor.process_row(
            row_index=0,
            row_data={"value": 150},
            transforms=[RouterGate(gate.node_id)],
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
        from elspeth.core.landscape import LandscapeDB, LandscapeRecorder
        from elspeth.engine.processor import RowProcessor
        from elspeth.engine.spans import SpanFactory

        db = LandscapeDB.in_memory()
        recorder = LandscapeRecorder(db)
        run = recorder.begin_run(config={}, canonical_version="v1")

        source = recorder.register_node(
            run_id=run.run_id,
            plugin_name="source",
            node_type="source",
            plugin_version="1.0",
            config={},
            schema_config=DYNAMIC_SCHEMA,
        )
        gate = recorder.register_node(
            run_id=run.run_id,
            plugin_name="splitter",
            node_type="gate",
            plugin_version="1.0",
            config={},
            schema_config=DYNAMIC_SCHEMA,
        )
        path_a = recorder.register_node(
            run_id=run.run_id,
            plugin_name="path_a",
            node_type="transform",
            plugin_version="1.0",
            config={},
            schema_config=DYNAMIC_SCHEMA,
        )
        path_b = recorder.register_node(
            run_id=run.run_id,
            plugin_name="path_b",
            node_type="transform",
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

        class SplitterGate(BaseGate):
            name = "splitter"
            input_schema = _TestSchema
            output_schema = _TestSchema

            def __init__(self, node_id: str) -> None:
                super().__init__({})
                self.node_id = node_id

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

        results = processor.process_row(
            row_index=0,
            row_data={"value": 42},
            transforms=[SplitterGate(gate.node_id)],
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


class TestRowProcessorAggregation:
    """Aggregation handling in RowProcessor."""

    def test_aggregation_consumes_row(self) -> None:
        """Row accepted by aggregation returns consumed outcome."""
        from elspeth.core.landscape import LandscapeDB, LandscapeRecorder
        from elspeth.engine.processor import RowProcessor
        from elspeth.engine.spans import SpanFactory

        db = LandscapeDB.in_memory()
        recorder = LandscapeRecorder(db)
        run = recorder.begin_run(config={}, canonical_version="v1")

        source = recorder.register_node(
            run_id=run.run_id,
            plugin_name="source",
            node_type="source",
            plugin_version="1.0",
            config={},
            schema_config=DYNAMIC_SCHEMA,
        )
        agg = recorder.register_node(
            run_id=run.run_id,
            plugin_name="counter",
            node_type="aggregation",
            plugin_version="1.0",
            config={},
            schema_config=DYNAMIC_SCHEMA,
        )

        class CounterAggregation(BaseAggregation):
            name = "counter"
            input_schema = _TestSchema
            output_schema = _TestSchema

            def __init__(self, node_id: str) -> None:
                super().__init__({})
                self.node_id = node_id
                self._count: int = 0

            def accept(self, row: dict[str, Any], ctx: PluginContext) -> AcceptResult:
                self._count += 1
                return AcceptResult(accepted=True, trigger=False)

            def should_trigger(self) -> bool:
                return False

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

        results = processor.process_row(
            row_index=0,
            row_data={"value": 1},
            transforms=[CounterAggregation(agg.node_id)],
            ctx=ctx,
        )

        # Single result - aggregated
        assert len(results) == 1
        result = results[0]

        assert result.outcome == RowOutcome.CONSUMED_IN_BATCH

    def test_aggregation_trigger_flushes(self) -> None:
        """Aggregation trigger causes flush to be called."""
        from elspeth.core.landscape import LandscapeDB, LandscapeRecorder
        from elspeth.engine.processor import RowProcessor
        from elspeth.engine.spans import SpanFactory

        db = LandscapeDB.in_memory()
        recorder = LandscapeRecorder(db)
        run = recorder.begin_run(config={}, canonical_version="v1")

        source = recorder.register_node(
            run_id=run.run_id,
            plugin_name="source",
            node_type="source",
            plugin_version="1.0",
            config={},
            schema_config=DYNAMIC_SCHEMA,
        )
        agg = recorder.register_node(
            run_id=run.run_id,
            plugin_name="threshold_agg",
            node_type="aggregation",
            plugin_version="1.0",
            config={},
            schema_config=DYNAMIC_SCHEMA,
        )

        flush_called: list[bool] = []

        class ThresholdAggregation(BaseAggregation):
            name = "threshold_agg"
            input_schema = _TestSchema
            output_schema = _TestSchema

            def __init__(self, node_id: str) -> None:
                super().__init__({})
                self.node_id = node_id
                self._values: list[int] = []

            def accept(self, row: dict[str, Any], ctx: PluginContext) -> AcceptResult:
                self._values.append(row["value"])
                # Trigger when we have 2 values
                return AcceptResult(accepted=True, trigger=len(self._values) >= 2)

            def should_trigger(self) -> bool:
                return len(self._values) >= 2

            def flush(self, ctx: PluginContext) -> list[dict[str, Any]]:
                flush_called.append(True)
                result = [{"sum": sum(self._values)}]
                self._values = []
                return result

        ctx = PluginContext(run_id=run.run_id, config={})
        aggregation = ThresholdAggregation(agg.node_id)
        processor = RowProcessor(
            recorder=recorder,
            span_factory=SpanFactory(),
            run_id=run.run_id,
            source_node_id=source.node_id,
        )

        # First row - no trigger
        results1 = processor.process_row(
            row_index=0,
            row_data={"value": 10},
            transforms=[aggregation],
            ctx=ctx,
        )
        assert len(results1) == 1
        assert results1[0].outcome == RowOutcome.CONSUMED_IN_BATCH
        assert len(flush_called) == 0

        # Second row - triggers
        results2 = processor.process_row(
            row_index=1,
            row_data={"value": 20},
            transforms=[aggregation],
            ctx=ctx,
        )
        assert len(results2) == 1
        assert results2[0].outcome == RowOutcome.CONSUMED_IN_BATCH
        assert len(flush_called) == 1  # Flush was called


class TestRowProcessorTokenIdentity:
    """Token identity is preserved and accessible."""

    def test_token_accessible_on_result(self) -> None:
        """RowResult provides access to full token info."""
        from elspeth.core.landscape import LandscapeDB, LandscapeRecorder
        from elspeth.engine.processor import RowProcessor
        from elspeth.engine.spans import SpanFactory

        db = LandscapeDB.in_memory()
        recorder = LandscapeRecorder(db)
        run = recorder.begin_run(config={}, canonical_version="v1")

        source = recorder.register_node(
            run_id=run.run_id,
            plugin_name="source",
            node_type="source",
            plugin_version="1.0",
            config={},
            schema_config=DYNAMIC_SCHEMA,
        )

        ctx = PluginContext(run_id=run.run_id, config={})
        processor = RowProcessor(
            recorder=recorder,
            span_factory=SpanFactory(),
            run_id=run.run_id,
            source_node_id=source.node_id,
        )

        results = processor.process_row(
            row_index=0,
            row_data={"test": "data"},
            transforms=[],
            ctx=ctx,
        )

        # Single result - no forks
        assert len(results) == 1
        result = results[0]

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

        db = LandscapeDB.in_memory()
        recorder = LandscapeRecorder(db)
        run = recorder.begin_run(config={}, canonical_version="v1")

        source = recorder.register_node(
            run_id=run.run_id,
            plugin_name="source",
            node_type="source",
            plugin_version="1.0",
            config={},
            schema_config=DYNAMIC_SCHEMA,
        )
        transform1 = recorder.register_node(
            run_id=run.run_id,
            plugin_name="t1",
            node_type="transform",
            plugin_version="1.0",
            config={},
            schema_config=DYNAMIC_SCHEMA,
        )
        transform2 = recorder.register_node(
            run_id=run.run_id,
            plugin_name="t2",
            node_type="transform",
            plugin_version="1.0",
            config={},
            schema_config=DYNAMIC_SCHEMA,
        )

        class IdentityTransform(BaseTransform):
            input_schema = _TestSchema
            output_schema = _TestSchema

            def __init__(self, name: str, node_id: str) -> None:
                super().__init__({})
                self.name = name  # type: ignore[misc]
                self.node_id = node_id

            def process(
                self, row: dict[str, Any], ctx: PluginContext
            ) -> TransformResult:
                return TransformResult.success(row)

        ctx = PluginContext(run_id=run.run_id, config={})
        processor = RowProcessor(
            recorder=recorder,
            span_factory=SpanFactory(),
            run_id=run.run_id,
            source_node_id=source.node_id,
        )

        results = processor.process_row(
            row_index=0,
            row_data={"value": 1},
            transforms=[
                IdentityTransform("t1", transform1.node_id),
                IdentityTransform("t2", transform2.node_id),
            ],
            ctx=ctx,
        )

        # Single result - no forks
        assert len(results) == 1
        result = results[0]

        assert result.outcome == RowOutcome.COMPLETED

        # Verify node states recorded with correct step indices
        states = recorder.get_node_states_for_token(result.token_id)
        assert len(states) == 2
        # Steps should be 1 and 2 (source is 0, transforms start at 1)
        step_indices = {s.step_index for s in states}
        assert step_indices == {1, 2}


class TestRowProcessorUnknownType:
    """Test handling of unknown plugin types."""

    def test_unknown_type_raises_type_error(self) -> None:
        """Unknown plugin types raise TypeError with helpful message."""
        import pytest

        from elspeth.core.landscape import LandscapeDB, LandscapeRecorder
        from elspeth.engine.processor import RowProcessor
        from elspeth.engine.spans import SpanFactory

        db = LandscapeDB.in_memory()
        recorder = LandscapeRecorder(db)
        run = recorder.begin_run(config={}, canonical_version="v1")

        source = recorder.register_node(
            run_id=run.run_id,
            plugin_name="source",
            node_type="source",
            plugin_version="1.0",
            config={},
            schema_config=DYNAMIC_SCHEMA,
        )

        class NotAPlugin:
            """A class that doesn't inherit from any base class."""

            name = "fake"
            node_id = "fake_id"

            def process(self, row: dict[str, Any], ctx: PluginContext) -> None:
                pass

        ctx = PluginContext(run_id=run.run_id, config={})
        processor = RowProcessor(
            recorder=recorder,
            span_factory=SpanFactory(),
            run_id=run.run_id,
            source_node_id=source.node_id,
        )

        with pytest.raises(TypeError) as exc_info:
            processor.process_row(
                row_index=0,
                row_data={"value": 1},
                transforms=[NotAPlugin()],
                ctx=ctx,
            )

        assert "Unknown transform type: NotAPlugin" in str(exc_info.value)
        assert "BaseTransform" in str(exc_info.value)
        assert "BaseGate" in str(exc_info.value)
        assert "BaseAggregation" in str(exc_info.value)


class TestRowProcessorWorkQueue:
    """Work queue tests for fork child execution."""

    def test_fork_children_are_executed_through_work_queue(self) -> None:
        """Fork child tokens should be processed, not orphaned."""
        from elspeth.core.landscape import LandscapeDB, LandscapeRecorder
        from elspeth.engine.processor import RowProcessor
        from elspeth.engine.spans import SpanFactory

        db = LandscapeDB.in_memory()
        recorder = LandscapeRecorder(db)
        run = recorder.begin_run(config={}, canonical_version="v1")

        # Register nodes
        source_node = recorder.register_node(
            run_id=run.run_id,
            plugin_name="test_source",
            node_type="source",
            plugin_version="1.0",
            config={},
            schema_config=DYNAMIC_SCHEMA,
        )
        gate_node = recorder.register_node(
            run_id=run.run_id,
            plugin_name="splitter",
            node_type="gate",
            plugin_version="1.0",
            config={},
            schema_config=DYNAMIC_SCHEMA,
        )
        transform_node = recorder.register_node(
            run_id=run.run_id,
            plugin_name="enricher",
            node_type="transform",
            plugin_version="1.0",
            config={},
            schema_config=DYNAMIC_SCHEMA,
        )

        # Register edges for fork paths
        edge_a = recorder.register_edge(
            run_id=run.run_id,
            from_node_id=gate_node.node_id,
            to_node_id=transform_node.node_id,
            label="path_a",
            mode=RoutingMode.COPY,
        )
        edge_b = recorder.register_edge(
            run_id=run.run_id,
            from_node_id=gate_node.node_id,
            to_node_id=transform_node.node_id,
            label="path_b",
            mode=RoutingMode.COPY,
        )

        # Create gate that forks
        class SplitterGate(BaseGate):
            name = "splitter"
            input_schema = _TestSchema
            output_schema = _TestSchema

            def __init__(self, node_id: str) -> None:
                super().__init__({})
                self.node_id = node_id

            def evaluate(self, row: dict[str, Any], ctx: PluginContext) -> GateResult:
                return GateResult(
                    row=row,
                    action=RoutingAction.fork_to_paths(["path_a", "path_b"]),
                )

        # Create transform that marks execution
        class MarkerTransform(BaseTransform):
            name = "enricher"
            input_schema = _TestSchema
            output_schema = _TestSchema

            def __init__(self, node_id: str) -> None:
                super().__init__({})
                self.node_id = node_id

            def process(
                self, row: dict[str, Any], ctx: PluginContext
            ) -> TransformResult:
                return TransformResult.success({**row, "processed": True})

        gate = SplitterGate(gate_node.node_id)
        transform = MarkerTransform(transform_node.node_id)

        processor = RowProcessor(
            recorder=recorder,
            span_factory=SpanFactory(),
            run_id=run.run_id,
            source_node_id=source_node.node_id,
            edge_map={
                (gate_node.node_id, "path_a"): edge_a.edge_id,
                (gate_node.node_id, "path_b"): edge_b.edge_id,
            },
        )

        ctx = PluginContext(run_id=run.run_id, config={})

        # Process row - should return multiple results (parent + children)
        results = processor.process_row(
            row_index=0,
            row_data={"value": 42},
            transforms=[gate, transform],
            ctx=ctx,
        )

        # Should have 3 results: parent (FORKED) + 2 children (COMPLETED)
        assert isinstance(results, list)
        assert len(results) == 3

        # Parent should be FORKED
        forked_results = [r for r in results if r.outcome == RowOutcome.FORKED]
        assert len(forked_results) == 1

        # Children should be COMPLETED and processed
        completed_results = [r for r in results if r.outcome == RowOutcome.COMPLETED]
        assert len(completed_results) == 2
        for result in completed_results:
            # Direct access - we know the field exists because we just set it
            assert result.final_data["processed"] is True
            assert result.token.branch_name in ("path_a", "path_b")
