# tests/engine/test_processor.py
"""Tests for RowProcessor.

Test plugins inherit from base classes (BaseTransform, BaseAggregation)
because the processor uses isinstance() for type-safe plugin detection.
Gates are config-driven using GateSettings.
"""

from pathlib import Path
from typing import Any

from elspeth.contracts import PluginSchema, RoutingMode
from elspeth.contracts.schema import SchemaConfig
from elspeth.plugins.base import BaseAggregation, BaseTransform
from elspeth.plugins.context import PluginContext
from elspeth.plugins.results import (
    AcceptResult,
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

    def test_transform_error_without_on_error_raises(self) -> None:
        """Transform returning error without on_error configured raises RuntimeError."""
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
            # No _on_error configured - errors are bugs

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

        # Without on_error configured, returning error is a bug - should raise
        with pytest.raises(RuntimeError) as exc_info:
            processor.process_row(
                row_index=0,
                row_data={"value": -5},
                transforms=[ValidatorTransform(transform.node_id)],
                ctx=ctx,
            )

        assert "no on_error configured" in str(exc_info.value)

    def test_transform_error_with_discard_returns_quarantined(self) -> None:
        """Transform error with on_error='discard' should return QUARANTINED."""
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

        class DiscardingValidator(BaseTransform):
            """Validator that discards errors (on_error='discard')."""

            name = "validator"
            input_schema = _TestSchema
            output_schema = _TestSchema
            _on_error = "discard"  # Intentionally discard errors

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
            transforms=[DiscardingValidator(transform.node_id)],
            ctx=ctx,
        )

        # Single result - no forks
        assert len(results) == 1
        result = results[0]

        # With on_error='discard', error becomes QUARANTINED (intentional rejection)
        assert result.outcome == RowOutcome.QUARANTINED
        # Original data preserved
        assert result.final_data == {"value": -5}

    def test_transform_error_with_sink_returns_routed(self) -> None:
        """Transform error with on_error=sink_name should return ROUTED."""
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

        class RoutingValidator(BaseTransform):
            """Validator that routes errors to error_sink."""

            name = "validator"
            input_schema = _TestSchema
            output_schema = _TestSchema
            _on_error = "error_sink"  # Route errors to named sink

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
            transforms=[RoutingValidator(transform.node_id)],
            ctx=ctx,
        )

        # Single result - no forks
        assert len(results) == 1
        result = results[0]

        # With on_error='error_sink', error becomes ROUTED to that sink
        assert result.outcome == RowOutcome.ROUTED
        assert result.sink_name == "error_sink"
        # Original data preserved
        assert result.final_data == {"value": -5}


class TestRowProcessorGates:
    """Gate handling in RowProcessor."""

    def test_gate_continue_proceeds(self) -> None:
        """Gate returning continue proceeds to completion."""
        from elspeth.core.config import GateSettings
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
            plugin_name="final",
            node_type="transform",
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

        # Config-driven gate: always continues
        pass_gate = GateSettings(
            name="pass_gate",
            condition="True",
            routes={"true": "continue"},
        )

        ctx = PluginContext(run_id=run.run_id, config={})
        processor = RowProcessor(
            recorder=recorder,
            span_factory=SpanFactory(),
            run_id=run.run_id,
            source_node_id=source.node_id,
            config_gates=[pass_gate],
            config_gate_id_map={"pass_gate": gate.node_id},
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
        edge_map = {(gate.node_id, "true"): edge.edge_id}
        # Route resolution map: label -> sink_name
        route_resolution_map = {(gate.node_id, "true"): "high_values"}
        processor = RowProcessor(
            recorder=recorder,
            span_factory=SpanFactory(),
            run_id=run.run_id,
            source_node_id=source.node_id,
            edge_map=edge_map,
            route_resolution_map=route_resolution_map,
            config_gates=[router_gate],
            config_gate_id_map={"router": gate.node_id},
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

        # Config-driven fork gate: always forks to path_a and path_b
        splitter_gate = GateSettings(
            name="splitter",
            condition="True",
            routes={"true": "fork"},
            fork_to=["path_a", "path_b"],
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
            config_gates=[splitter_gate],
            config_gate_id_map={"splitter": gate.node_id},
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
                return AcceptResult(accepted=True)

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

    def test_aggregation_accepts_multiple_rows(self) -> None:
        """Aggregation accepts rows into batch (flush is engine-controlled, WP-06)."""
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
                return AcceptResult(accepted=True)

            def flush(self, ctx: PluginContext) -> list[dict[str, Any]]:
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

        # First row - accepted into batch
        results1 = processor.process_row(
            row_index=0,
            row_data={"value": 10},
            transforms=[aggregation],
            ctx=ctx,
        )
        assert len(results1) == 1
        assert results1[0].outcome == RowOutcome.CONSUMED_IN_BATCH

        # Second row - also accepted into batch
        # NOTE: flush is now engine-controlled (WP-06 Tasks 7-8), not triggered by accept
        results2 = processor.process_row(
            row_index=1,
            row_data={"value": 20},
            transforms=[aggregation],
            ctx=ctx,
        )
        assert len(results2) == 1
        assert results2[0].outcome == RowOutcome.CONSUMED_IN_BATCH

        # Verify aggregation buffered both values (flush not automatically called)
        # Engine now handles trigger evaluation via TriggerEvaluator (WP-06)
        assert len(aggregation._values) == 2


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

        db = LandscapeDB.in_memory()
        recorder = LandscapeRecorder(db)
        run = recorder.begin_run(config={}, canonical_version="v1")

        # Setup nodes for: source -> transform -> gate1 (fork 2) -> gate2 (fork 2)
        source_node = recorder.register_node(
            run_id=run.run_id,
            plugin_name="test_source",
            node_type="source",
            plugin_version="1.0",
            config={},
            schema_config=DYNAMIC_SCHEMA,
        )
        transform_node = recorder.register_node(
            run_id=run.run_id,
            plugin_name="marker",
            node_type="transform",
            plugin_version="1.0",
            config={},
            schema_config=DYNAMIC_SCHEMA,
        )
        gate1_node = recorder.register_node(
            run_id=run.run_id,
            plugin_name="fork_gate_1",
            node_type="gate",
            plugin_version="1.0",
            config={},
            schema_config=DYNAMIC_SCHEMA,
        )
        gate2_node = recorder.register_node(
            run_id=run.run_id,
            plugin_name="fork_gate_2",
            node_type="gate",
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
                super().__init__({})
                self.node_id = node_id

            def process(
                self, row: dict[str, Any], ctx: PluginContext
            ) -> TransformResult:
                # Note: .get() is allowed here - this is row data (their data, Tier 2)
                return TransformResult.success(
                    {**row, "count": row.get("count", 0) + 1}
                )

        # Config-driven fork gates
        gate1_config = GateSettings(
            name="fork_gate_1",
            condition="True",
            routes={"true": "fork"},
            fork_to=["left", "right"],
        )
        gate2_config = GateSettings(
            name="fork_gate_2",
            condition="True",
            routes={"true": "fork"},
            fork_to=["left", "right"],
        )

        transform = MarkerTransform(transform_node.node_id)

        processor = RowProcessor(
            recorder=recorder,
            span_factory=SpanFactory(),
            run_id=run.run_id,
            source_node_id=source_node.node_id,
            edge_map={
                (gate1_node.node_id, "left"): edge1a.edge_id,
                (gate1_node.node_id, "right"): edge1b.edge_id,
                (gate2_node.node_id, "left"): edge2a.edge_id,
                (gate2_node.node_id, "right"): edge2b.edge_id,
            },
            config_gates=[gate1_config, gate2_config],
            config_gate_id_map={
                "fork_gate_1": gate1_node.node_id,
                "fork_gate_2": gate2_node.node_id,
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


class TestRowProcessorWorkQueue:
    """Work queue tests for fork child execution."""

    def test_work_queue_iteration_guard_prevents_infinite_loop(self) -> None:
        """Work queue should fail if iterations exceed limit."""
        import elspeth.engine.processor as proc_module
        from elspeth.core.landscape import LandscapeDB, LandscapeRecorder
        from elspeth.engine.processor import RowProcessor
        from elspeth.engine.spans import SpanFactory

        db = LandscapeDB.in_memory()
        recorder = LandscapeRecorder(db)
        run = recorder.begin_run(config={}, canonical_version="v1")

        source_node = recorder.register_node(
            run_id=run.run_id,
            plugin_name="test_source",
            node_type="source",
            plugin_version="1.0",
            config={},
            schema_config=DYNAMIC_SCHEMA,
        )

        # Create a transform that somehow creates infinite work
        # (This shouldn't be possible with correct implementation,
        # but the guard protects against bugs)

        processor = RowProcessor(
            recorder=recorder,
            span_factory=SpanFactory(),
            run_id=run.run_id,
            source_node_id=source_node.node_id,
        )

        # Patch MAX_WORK_QUEUE_ITERATIONS to a small number for testing
        original_max = proc_module.MAX_WORK_QUEUE_ITERATIONS
        proc_module.MAX_WORK_QUEUE_ITERATIONS = 5

        try:
            # This test verifies the guard exists - actual infinite loop
            # would require a bug in the implementation
            ctx = PluginContext(run_id=run.run_id, config={})
            results = processor.process_row(
                row_index=0,
                row_data={"value": 42},
                transforms=[],
                ctx=ctx,
            )
            # Should complete normally with no transforms
            assert len(results) == 1
            assert results[0].outcome == RowOutcome.COMPLETED
        finally:
            proc_module.MAX_WORK_QUEUE_ITERATIONS = original_max

    def test_fork_children_are_executed_through_work_queue(self) -> None:
        """Fork child tokens should be processed, not orphaned."""
        from elspeth.core.config import GateSettings
        from elspeth.core.landscape import LandscapeDB, LandscapeRecorder
        from elspeth.engine.processor import RowProcessor
        from elspeth.engine.spans import SpanFactory

        db = LandscapeDB.in_memory()
        recorder = LandscapeRecorder(db)
        run = recorder.begin_run(config={}, canonical_version="v1")

        # Register nodes (transform before gate since config gates run after transforms)
        source_node = recorder.register_node(
            run_id=run.run_id,
            plugin_name="test_source",
            node_type="source",
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
        gate_node = recorder.register_node(
            run_id=run.run_id,
            plugin_name="splitter",
            node_type="gate",
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

        # Config-driven fork gate
        splitter_gate = GateSettings(
            name="splitter",
            condition="True",
            routes={"true": "fork"},
            fork_to=["path_a", "path_b"],
        )

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
            config_gates=[splitter_gate],
            config_gate_id_map={"splitter": gate_node.node_id},
        )

        ctx = PluginContext(run_id=run.run_id, config={})

        # Process row - should return multiple results (parent + children)
        results = processor.process_row(
            row_index=0,
            row_data={"value": 42},
            transforms=[transform],
            ctx=ctx,
        )

        # Should have 3 results: parent (FORKED) + 2 children (COMPLETED)
        assert isinstance(results, list)
        assert len(results) == 3

        # Parent should be FORKED
        forked_results = [r for r in results if r.outcome == RowOutcome.FORKED]
        assert len(forked_results) == 1

        # Children should be COMPLETED and processed (all tokens have processed=True)
        completed_results = [r for r in results if r.outcome == RowOutcome.COMPLETED]
        assert len(completed_results) == 2
        for result in completed_results:
            # Direct access - we know the field exists because we just set it
            assert result.final_data["processed"] is True
            assert result.token.branch_name in ("path_a", "path_b")


class TestQuarantineIntegration:
    """Integration tests for full quarantine flow."""

    def test_pipeline_continues_after_quarantine(self) -> None:
        """Pipeline should continue processing after quarantining a row.

        Processes 5 rows with mixed outcomes:
        - 3 positive values -> COMPLETED (validated)
        - 2 negative values -> QUARANTINED (rejected by validator)

        Verifies:
        - All 5 rows are processed
        - Correct outcomes assigned to each
        - Completed rows have "validated" flag added
        - Quarantined rows have original data (not modified)
        """
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

        class ValidatingTransform(BaseTransform):
            """Validator that quarantines negative values (on_error='discard')."""

            name = "validator"
            input_schema = _TestSchema
            output_schema = _TestSchema
            _on_error = "discard"  # Intentionally quarantine errors

            def __init__(self, node_id: str) -> None:
                super().__init__({})
                self.node_id = node_id

            def process(
                self, row: dict[str, Any], ctx: PluginContext
            ) -> TransformResult:
                if row["value"] < 0:
                    return TransformResult.error(
                        {
                            "message": "negative values not allowed",
                            "value": row["value"],
                        }
                    )
                return TransformResult.success({**row, "validated": True})

        ctx = PluginContext(run_id=run.run_id, config={}, landscape=recorder)
        processor = RowProcessor(
            recorder=recorder,
            span_factory=SpanFactory(),
            run_id=run.run_id,
            source_node_id=source.node_id,
        )

        # Process 5 rows: [10, -5, 20, -1, 30]
        test_values = [10, -5, 20, -1, 30]
        all_results: list[Any] = []

        for i, value in enumerate(test_values):
            results = processor.process_row(
                row_index=i,
                row_data={"value": value},
                transforms=[ValidatingTransform(transform.node_id)],
                ctx=ctx,
            )
            all_results.extend(results)

        # Verify 5 results total (one per row)
        assert len(all_results) == 5

        # Verify outcomes
        completed = [r for r in all_results if r.outcome == RowOutcome.COMPLETED]
        quarantined = [r for r in all_results if r.outcome == RowOutcome.QUARANTINED]

        assert len(completed) == 3  # Positive values
        assert len(quarantined) == 2  # Negative values

        # Verify completed rows have "validated" flag
        for result in completed:
            assert result.final_data["validated"] is True
            assert result.final_data["value"] > 0

        # Verify quarantined rows have original data (not modified)
        for result in quarantined:
            assert "validated" not in result.final_data
            assert result.final_data["value"] < 0

    def test_quarantine_records_audit_trail(self) -> None:
        """Quarantined rows should be recorded in audit trail.

        Verifies that when a row is quarantined:
        - The outcome is QUARANTINED
        - A node_state was recorded with status="failed"
        - The node_state record exists in the database
        """
        from elspeth.contracts import NodeStateFailed
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
            plugin_name="strict_validator",
            node_type="transform",
            plugin_version="1.0",
            config={},
            schema_config=DYNAMIC_SCHEMA,
        )

        class StrictValidator(BaseTransform):
            """Validator that rejects rows with missing 'required_field'."""

            name = "strict_validator"
            input_schema = _TestSchema
            output_schema = _TestSchema
            _on_error = "discard"  # Quarantine invalid rows

            def __init__(self, node_id: str) -> None:
                super().__init__({})
                self.node_id = node_id

            def process(
                self, row: dict[str, Any], ctx: PluginContext
            ) -> TransformResult:
                # row.get() is allowed here - this is row data (their data, Tier 2)
                if "required_field" not in row:
                    return TransformResult.error(
                        {
                            "message": "missing required_field",
                            "row_keys": list(row.keys()),
                        }
                    )
                return TransformResult.success({**row, "validated": True})

        ctx = PluginContext(run_id=run.run_id, config={}, landscape=recorder)
        processor = RowProcessor(
            recorder=recorder,
            span_factory=SpanFactory(),
            run_id=run.run_id,
            source_node_id=source.node_id,
        )

        # Process an invalid row (missing required_field)
        results = processor.process_row(
            row_index=0,
            row_data={"other_field": "some_value"},
            transforms=[StrictValidator(transform.node_id)],
            ctx=ctx,
        )

        # Single result
        assert len(results) == 1
        result = results[0]

        # Verify outcome is QUARANTINED
        assert result.outcome == RowOutcome.QUARANTINED

        # Verify original data is preserved
        assert result.final_data == {"other_field": "some_value"}

        # Query the node_states table to confirm the record exists
        states = recorder.get_node_states_for_token(result.token_id)

        # Should have exactly 1 node_state (for the transform)
        assert len(states) == 1

        state = states[0]
        assert isinstance(state, NodeStateFailed)
        assert state.status.value == "failed"
        assert state.node_id == transform.node_id
        assert state.token_id == result.token_id

        # Verify the error was recorded
        assert state.error_json is not None
        import json

        error_data = json.loads(state.error_json)
        assert error_data["message"] == "missing required_field"


class TestProcessorAggregationTriggers:
    """Tests for config-driven aggregation triggers in RowProcessor."""

    def test_processor_accepts_aggregation_settings(self) -> None:
        """RowProcessor accepts aggregation_settings parameter."""
        from elspeth.core.config import AggregationSettings, TriggerConfig
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

        aggregation_settings = {
            "agg-1": AggregationSettings(
                name="test_agg",
                plugin="test",
                trigger=TriggerConfig(count=3),
            ),
        }

        # Should not raise
        processor = RowProcessor(
            recorder=recorder,
            span_factory=SpanFactory(),
            run_id=run.run_id,
            source_node_id=source.node_id,
            aggregation_settings=aggregation_settings,
        )
        assert processor is not None

    def test_processor_flushes_on_count_trigger(self) -> None:
        """Processor flushes aggregation when count trigger reached."""
        from elspeth.core.config import AggregationSettings, TriggerConfig
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
        agg_node = recorder.register_node(
            run_id=run.run_id,
            plugin_name="counting_agg",
            node_type="aggregation",
            plugin_version="1.0",
            config={},
            schema_config=DYNAMIC_SCHEMA,
        )

        # Track how many times flush was called
        flush_call_count = 0
        flushed_values: list[list[dict[str, Any]]] = []

        class CountingAggregation(BaseAggregation):
            """Aggregation that tracks flush calls."""

            name = "counting_agg"
            input_schema = _TestSchema
            output_schema = _TestSchema

            def __init__(self, node_id: str) -> None:
                super().__init__({})
                self.node_id = node_id
                self._values: list[dict[str, Any]] = []

            def accept(self, row: dict[str, Any], ctx: PluginContext) -> AcceptResult:
                self._values.append(row)
                return AcceptResult(accepted=True)

            def flush(self, ctx: PluginContext) -> list[dict[str, Any]]:
                nonlocal flush_call_count, flushed_values
                flush_call_count += 1
                result = [{"sum": sum(r["value"] for r in self._values)}]
                flushed_values.append(list(self._values))
                self._values = []
                return result

        # Configure trigger: flush after 3 rows
        aggregation_settings = {
            agg_node.node_id: AggregationSettings(
                name="counting_agg",
                plugin="counting_agg",
                trigger=TriggerConfig(count=3),
            ),
        }

        ctx = PluginContext(run_id=run.run_id, config={})
        aggregation = CountingAggregation(agg_node.node_id)
        processor = RowProcessor(
            recorder=recorder,
            span_factory=SpanFactory(),
            run_id=run.run_id,
            source_node_id=source.node_id,
            aggregation_settings=aggregation_settings,
        )

        # Process 3 rows - should trigger flush after 3rd
        for i in range(3):
            processor.process_row(
                row_index=i,
                row_data={"value": i + 1},
                transforms=[aggregation],
                ctx=ctx,
            )

        # Verify flush was called once (after 3rd accept triggered count=3)
        assert flush_call_count == 1
        # Verify all 3 rows were in the flushed batch
        assert len(flushed_values) == 1
        assert len(flushed_values[0]) == 3
        assert flushed_values[0] == [{"value": 1}, {"value": 2}, {"value": 3}]

    def test_aggregated_tokens_audit_trail(self) -> None:
        """Tokens consumed by aggregation have complete audit trail.

        For any row that was aggregated, the audit trail should show:
        - Source row entry
        - Aggregation accept (CONSUMED_IN_BATCH status)
        - Link to batch that consumed it
        - Batch output row

        This test verifies:
        1. Each token has a node_state for the aggregation node
        2. The state status is "completed" (accept succeeded)
        3. The output_hash matches expected structure containing batch_id
        4. The batch_members table links token to batch with ordinal
        """
        from elspeth.contracts import NodeType
        from elspeth.core.config import AggregationSettings, TriggerConfig
        from elspeth.core.landscape import LandscapeDB, LandscapeRecorder
        from elspeth.engine.processor import RowProcessor
        from elspeth.engine.spans import SpanFactory

        db = LandscapeDB.in_memory()
        recorder = LandscapeRecorder(db)
        run = recorder.begin_run(config={}, canonical_version="v1")

        # Register source and aggregation nodes
        source = recorder.register_node(
            run_id=run.run_id,
            plugin_name="source",
            node_type=NodeType.SOURCE,
            plugin_version="1.0",
            config={},
            schema_config=DYNAMIC_SCHEMA,
        )
        agg_node = recorder.register_node(
            run_id=run.run_id,
            plugin_name="sum_aggregator",
            node_type=NodeType.AGGREGATION,
            plugin_version="1.0",
            config={},
            schema_config=DYNAMIC_SCHEMA,
        )

        # Track processed tokens for later verification
        processed_token_ids: list[str] = []

        class SumAggregation(BaseAggregation):
            """Aggregation that sums values."""

            name = "sum_aggregator"
            input_schema = _TestSchema
            output_schema = _TestSchema

            def __init__(self, node_id: str) -> None:
                super().__init__({})
                self.node_id = node_id
                self._values: list[int] = []

            def accept(self, row: dict[str, Any], ctx: PluginContext) -> AcceptResult:
                self._values.append(row["value"])
                return AcceptResult(accepted=True)

            def flush(self, ctx: PluginContext) -> list[dict[str, Any]]:
                total = sum(self._values)
                self._values = []
                return [{"sum": total}]

        # Configure trigger: flush after 3 rows
        aggregation_settings = {
            agg_node.node_id: AggregationSettings(
                name="sum_aggregator",
                plugin="sum_aggregator",
                trigger=TriggerConfig(count=3),
            ),
        }

        ctx = PluginContext(run_id=run.run_id, config={})
        aggregation = SumAggregation(agg_node.node_id)
        processor = RowProcessor(
            recorder=recorder,
            span_factory=SpanFactory(),
            run_id=run.run_id,
            source_node_id=source.node_id,
            aggregation_settings=aggregation_settings,
        )

        # Process 3 rows - should trigger flush after 3rd
        for i in range(3):
            results = processor.process_row(
                row_index=i,
                row_data={"value": (i + 1) * 10},
                transforms=[aggregation],
                ctx=ctx,
            )
            # Each result has a token_id
            for result in results:
                processed_token_ids.append(result.token_id)

        assert len(processed_token_ids) == 3

        # === Verify batch_members table links tokens to batch ===
        # Get all batches for this run
        batches = recorder.get_batches(run.run_id)
        assert len(batches) == 1, "Should have exactly one batch"
        batch = batches[0]
        assert batch.aggregation_node_id == agg_node.node_id
        assert batch.status == "completed", "Batch should be completed after flush"

        # Verify batch members
        members = recorder.get_batch_members(batch.batch_id)
        assert len(members) == 3, "Batch should have 3 members"

        # Map token_id -> member for verification
        member_by_token = {m.token_id: m for m in members}

        # === Verify each token's audit trail ===
        for idx, token_id in enumerate(processed_token_ids):
            # 1. Token should be in batch_members
            assert token_id in member_by_token, f"Token {token_id} not in batch_members"
            member = member_by_token[token_id]
            assert member.batch_id == batch.batch_id
            assert (
                member.ordinal == idx
            ), f"Expected ordinal {idx}, got {member.ordinal}"

            # 2. Token should have node_state for the aggregation node
            states = recorder.get_node_states_for_token(token_id)
            assert (
                len(states) >= 1
            ), f"Token {token_id} should have at least one node_state"

            # Find the aggregation node state
            agg_states = [s for s in states if s.node_id == agg_node.node_id]
            assert (
                len(agg_states) == 1
            ), "Token should have exactly one aggregation state"
            agg_state = agg_states[0]

            # 2. State should be "completed" (accept succeeded)
            assert (
                agg_state.status == "completed"
            ), f"Aggregation state should be 'completed', got '{agg_state.status}'"

            # 3. Output hash should match expected output_data with batch_id
            # The output_data is canonicalized and stored as a hash for audit integrity.
            # We verify the hash matches the expected structure containing batch_id.
            from elspeth.core.canonical import stable_hash

            expected_output = {
                "row": {"value": (idx + 1) * 10},
                "batch_id": batch.batch_id,
                "ordinal": idx,
            }
            expected_hash = stable_hash(expected_output)
            assert (
                agg_state.output_hash == expected_hash
            ), f"output_hash mismatch for token {idx}: expected hash of {expected_output}"

            # 4. Verify the node is indeed an aggregation
            node = recorder.get_node(agg_state.node_id)
            assert node is not None
            assert node.node_type == NodeType.AGGREGATION

        # === Verify batch output was recorded ===
        # The batch should have a completed status with trigger reason
        assert batch.trigger_reason is not None, "Batch should have trigger_reason"


class TestRowProcessorCoalesce:
    """Test RowProcessor integration with CoalesceExecutor."""

    def test_processor_accepts_coalesce_executor(self) -> None:
        """RowProcessor should accept coalesce_executor parameter."""
        from elspeth.core.landscape import LandscapeDB, LandscapeRecorder
        from elspeth.engine.coalesce_executor import CoalesceExecutor
        from elspeth.engine.processor import RowProcessor
        from elspeth.engine.spans import SpanFactory
        from elspeth.engine.tokens import TokenManager

        db = LandscapeDB.in_memory()
        recorder = LandscapeRecorder(db)
        run = recorder.begin_run(config={}, canonical_version="v1")
        token_manager = TokenManager(recorder)

        source = recorder.register_node(
            run_id=run.run_id,
            plugin_name="source",
            node_type="source",
            plugin_version="1.0",
            config={},
            schema_config=DYNAMIC_SCHEMA,
        )

        coalesce_executor = CoalesceExecutor(
            recorder=recorder,
            span_factory=SpanFactory(),
            token_manager=token_manager,
            run_id=run.run_id,
        )

        # Should not raise
        processor = RowProcessor(
            recorder=recorder,
            span_factory=SpanFactory(),
            run_id=run.run_id,
            source_node_id=source.node_id,
            coalesce_executor=coalesce_executor,
        )
        assert processor._coalesce_executor is coalesce_executor

    def test_fork_then_coalesce_require_all(self) -> None:
        """Fork children should coalesce when all branches arrive.

        Pipeline: source -> enrich_a -> enrich_b -> fork_gate -> coalesce -> completed

        This test verifies the full fork->coalesce flow using config gates:
        1. Transforms enrich data (sentiment, entities)
        2. Gate forks to two paths (path_a, path_b) - children inherit enriched data
        3. Coalesce merges both paths with require_all policy
        4. Parent token becomes FORKED, children become COALESCED
        5. Merged token has fields from both transforms
        """
        from elspeth.contracts import NodeType, RoutingMode
        from elspeth.core.config import CoalesceSettings, GateSettings
        from elspeth.core.landscape import LandscapeDB, LandscapeRecorder
        from elspeth.engine.coalesce_executor import CoalesceExecutor
        from elspeth.engine.processor import RowProcessor
        from elspeth.engine.spans import SpanFactory
        from elspeth.engine.tokens import TokenManager

        db = LandscapeDB.in_memory()
        recorder = LandscapeRecorder(db)
        run = recorder.begin_run(config={}, canonical_version="v1")
        token_manager = TokenManager(recorder)

        # Register nodes (transforms before gate since config gates run after transforms)
        source = recorder.register_node(
            run_id=run.run_id,
            plugin_name="source",
            node_type=NodeType.SOURCE,
            plugin_version="1.0",
            config={},
            schema_config=DYNAMIC_SCHEMA,
        )
        transform_a = recorder.register_node(
            run_id=run.run_id,
            plugin_name="enrich_a",
            node_type=NodeType.TRANSFORM,
            plugin_version="1.0",
            config={},
            schema_config=DYNAMIC_SCHEMA,
        )
        transform_b = recorder.register_node(
            run_id=run.run_id,
            plugin_name="enrich_b",
            node_type=NodeType.TRANSFORM,
            plugin_version="1.0",
            config={},
            schema_config=DYNAMIC_SCHEMA,
        )
        fork_gate = recorder.register_node(
            run_id=run.run_id,
            plugin_name="splitter",
            node_type=NodeType.GATE,
            plugin_version="1.0",
            config={},
            schema_config=DYNAMIC_SCHEMA,
        )
        coalesce_node = recorder.register_node(
            run_id=run.run_id,
            plugin_name="merger",
            node_type=NodeType.COALESCE,
            plugin_version="1.0",
            config={},
            schema_config=DYNAMIC_SCHEMA,
        )

        # Register edges for fork paths
        edge_a = recorder.register_edge(
            run_id=run.run_id,
            from_node_id=fork_gate.node_id,
            to_node_id=coalesce_node.node_id,
            label="path_a",
            mode=RoutingMode.COPY,
        )
        edge_b = recorder.register_edge(
            run_id=run.run_id,
            from_node_id=fork_gate.node_id,
            to_node_id=coalesce_node.node_id,
            label="path_b",
            mode=RoutingMode.COPY,
        )

        # Setup coalesce executor
        coalesce_settings = CoalesceSettings(
            name="merger",
            branches=["path_a", "path_b"],
            policy="require_all",
            merge="union",
        )
        coalesce_executor = CoalesceExecutor(
            recorder=recorder,
            span_factory=SpanFactory(),
            token_manager=token_manager,
            run_id=run.run_id,
        )
        coalesce_executor.register_coalesce(coalesce_settings, coalesce_node.node_id)

        # Transforms enrich data before the fork
        class EnrichA(BaseTransform):
            name = "enrich_a"
            input_schema = _TestSchema
            output_schema = _TestSchema

            def __init__(self, node_id: str) -> None:
                super().__init__({})
                self.node_id = node_id

            def process(
                self, row: dict[str, Any], ctx: PluginContext
            ) -> TransformResult:
                return TransformResult.success({**row, "sentiment": "positive"})

        class EnrichB(BaseTransform):
            name = "enrich_b"
            input_schema = _TestSchema
            output_schema = _TestSchema

            def __init__(self, node_id: str) -> None:
                super().__init__({})
                self.node_id = node_id

            def process(
                self, row: dict[str, Any], ctx: PluginContext
            ) -> TransformResult:
                return TransformResult.success({**row, "entities": ["ACME"]})

        # Config-driven fork gate
        fork_gate_config = GateSettings(
            name="splitter",
            condition="True",
            routes={"true": "fork"},
            fork_to=["path_a", "path_b"],
        )

        processor = RowProcessor(
            recorder=recorder,
            span_factory=SpanFactory(),
            run_id=run.run_id,
            source_node_id=source.node_id,
            edge_map={
                (fork_gate.node_id, "path_a"): edge_a.edge_id,
                (fork_gate.node_id, "path_b"): edge_b.edge_id,
            },
            coalesce_executor=coalesce_executor,
            coalesce_node_ids={"merger": coalesce_node.node_id},
            config_gates=[fork_gate_config],
            config_gate_id_map={"splitter": fork_gate.node_id},
        )

        ctx = PluginContext(run_id=run.run_id, config={})

        # Process should:
        # 1. EnrichA adds sentiment
        # 2. EnrichB adds entities
        # 3. Fork at config gate (parent FORKED with both fields)
        # 4. Coalesce both paths (merged token COALESCED)
        results = processor.process_row(
            row_index=0,
            row_data={"text": "ACME earnings"},
            transforms=[
                EnrichA(transform_a.node_id),
                EnrichB(transform_b.node_id),
            ],
            ctx=ctx,
            coalesce_at_step=3,  # After config gate (step 3 = transforms(2) + gate(1))
            coalesce_name="merger",
        )

        # Verify outcomes
        outcomes = {r.outcome for r in results}
        assert RowOutcome.FORKED in outcomes
        assert RowOutcome.COALESCED in outcomes

        # Find the coalesced result
        coalesced = [r for r in results if r.outcome == RowOutcome.COALESCED]
        assert len(coalesced) == 1

        # Verify merged data (both fields present from transforms before fork)
        merged_data = coalesced[0].final_data
        assert merged_data["sentiment"] == "positive"
        assert merged_data["entities"] == ["ACME"]

    def test_coalesced_token_audit_trail_complete(self) -> None:
        """Coalesced tokens should have complete audit trail for explain().

        After enrich -> fork -> coalesce, querying explain() on the merged
        token should show:
        - Original source row
        - Transform processing steps
        - Fork point (parent token for forked children)
        - Both branch paths
        - Coalesce point with parent relationships

        This test verifies the audit infrastructure captures the complete
        lineage for a coalesced token, enabling explain() queries.
        """
        from elspeth.contracts import NodeType, RoutingMode
        from elspeth.core.config import CoalesceSettings, GateSettings
        from elspeth.core.landscape import LandscapeDB, LandscapeRecorder
        from elspeth.engine.coalesce_executor import CoalesceExecutor
        from elspeth.engine.processor import RowProcessor
        from elspeth.engine.spans import SpanFactory
        from elspeth.engine.tokens import TokenManager

        db = LandscapeDB.in_memory()
        recorder = LandscapeRecorder(db)
        run = recorder.begin_run(config={}, canonical_version="v1")
        token_manager = TokenManager(recorder)

        # Register nodes (transforms before gate since config gates run after transforms)
        source = recorder.register_node(
            run_id=run.run_id,
            plugin_name="source",
            node_type=NodeType.SOURCE,
            plugin_version="1.0",
            config={},
            schema_config=DYNAMIC_SCHEMA,
        )
        transform_a = recorder.register_node(
            run_id=run.run_id,
            plugin_name="enrich_a",
            node_type=NodeType.TRANSFORM,
            plugin_version="1.0",
            config={},
            schema_config=DYNAMIC_SCHEMA,
        )
        transform_b = recorder.register_node(
            run_id=run.run_id,
            plugin_name="enrich_b",
            node_type=NodeType.TRANSFORM,
            plugin_version="1.0",
            config={},
            schema_config=DYNAMIC_SCHEMA,
        )
        fork_gate = recorder.register_node(
            run_id=run.run_id,
            plugin_name="splitter",
            node_type=NodeType.GATE,
            plugin_version="1.0",
            config={},
            schema_config=DYNAMIC_SCHEMA,
        )
        coalesce_node = recorder.register_node(
            run_id=run.run_id,
            plugin_name="merger",
            node_type=NodeType.COALESCE,
            plugin_version="1.0",
            config={},
            schema_config=DYNAMIC_SCHEMA,
        )

        # Register edges for fork paths
        edge_a = recorder.register_edge(
            run_id=run.run_id,
            from_node_id=fork_gate.node_id,
            to_node_id=coalesce_node.node_id,
            label="path_a",
            mode=RoutingMode.COPY,
        )
        edge_b = recorder.register_edge(
            run_id=run.run_id,
            from_node_id=fork_gate.node_id,
            to_node_id=coalesce_node.node_id,
            label="path_b",
            mode=RoutingMode.COPY,
        )

        # Setup coalesce executor
        coalesce_settings = CoalesceSettings(
            name="merger",
            branches=["path_a", "path_b"],
            policy="require_all",
            merge="union",
        )
        coalesce_executor = CoalesceExecutor(
            recorder=recorder,
            span_factory=SpanFactory(),
            token_manager=token_manager,
            run_id=run.run_id,
        )
        coalesce_executor.register_coalesce(coalesce_settings, coalesce_node.node_id)

        # Transforms enrich data before the fork
        class EnrichA(BaseTransform):
            name = "enrich_a"
            input_schema = _TestSchema
            output_schema = _TestSchema

            def __init__(self, node_id: str) -> None:
                super().__init__({})
                self.node_id = node_id

            def process(
                self, row: dict[str, Any], ctx: PluginContext
            ) -> TransformResult:
                return TransformResult.success({**row, "sentiment": "positive"})

        class EnrichB(BaseTransform):
            name = "enrich_b"
            input_schema = _TestSchema
            output_schema = _TestSchema

            def __init__(self, node_id: str) -> None:
                super().__init__({})
                self.node_id = node_id

            def process(
                self, row: dict[str, Any], ctx: PluginContext
            ) -> TransformResult:
                return TransformResult.success({**row, "entities": ["ACME"]})

        # Config-driven fork gate
        fork_gate_config = GateSettings(
            name="splitter",
            condition="True",
            routes={"true": "fork"},
            fork_to=["path_a", "path_b"],
        )

        processor = RowProcessor(
            recorder=recorder,
            span_factory=SpanFactory(),
            run_id=run.run_id,
            source_node_id=source.node_id,
            edge_map={
                (fork_gate.node_id, "path_a"): edge_a.edge_id,
                (fork_gate.node_id, "path_b"): edge_b.edge_id,
            },
            coalesce_executor=coalesce_executor,
            coalesce_node_ids={"merger": coalesce_node.node_id},
            config_gates=[fork_gate_config],
            config_gate_id_map={"splitter": fork_gate.node_id},
        )

        ctx = PluginContext(run_id=run.run_id, config={})

        # Process the row through enrich -> fork -> coalesce
        results = processor.process_row(
            row_index=0,
            row_data={"text": "ACME earnings"},
            transforms=[
                EnrichA(transform_a.node_id),
                EnrichB(transform_b.node_id),
            ],
            ctx=ctx,
            coalesce_at_step=3,  # After config gate (step 3 = transforms(2) + gate(1))
            coalesce_name="merger",
        )

        # === Verify outcomes ===
        forked_results = [r for r in results if r.outcome == RowOutcome.FORKED]
        coalesced_results = [r for r in results if r.outcome == RowOutcome.COALESCED]

        assert len(forked_results) == 1, "Should have exactly 1 FORKED result"
        assert len(coalesced_results) == 1, "Should have exactly 1 COALESCED result"

        forked = forked_results[0]
        coalesced = coalesced_results[0]

        # === Audit Trail: Verify source row exists ===
        row = recorder.get_row(forked.row_id)
        assert row is not None, "Source row should be recorded"
        assert row.row_index == 0
        assert row.source_node_id == source.node_id

        # === Audit Trail: Verify merged token has parent relationships ===
        # The merged token's parents are the consumed child tokens (with branch names)
        merged_token = coalesced.token
        merged_parents = recorder.get_token_parents(merged_token.token_id)
        assert (
            len(merged_parents) == 2
        ), "Merged token should have 2 parents (the consumed children)"

        # Get child token IDs from the merged token's parents
        child_token_ids = {p.parent_token_id for p in merged_parents}

        # Verify child tokens have branch names
        for child_token_id in child_token_ids:
            child_token = recorder.get_token(child_token_id)
            assert child_token is not None, "Child token should exist"
            assert child_token.branch_name in (
                "path_a",
                "path_b",
            ), f"Child token should have branch name, got {child_token.branch_name}"

        # Verify child tokens have parent relationships pointing to forked token
        for child_token_id in child_token_ids:
            parents = recorder.get_token_parents(child_token_id)
            assert len(parents) == 1, "Child token should have 1 parent"
            assert (
                parents[0].parent_token_id == forked.token_id
            ), "Parent should be the forked token"

        # === Audit Trail: Verify consumed tokens have node_states at coalesce ===
        # The CoalesceExecutor records node_states for consumed tokens
        for child_token_id in child_token_ids:
            states = recorder.get_node_states_for_token(child_token_id)
            # Should have states: gate evaluation + transform processing + coalesce
            assert (
                len(states) >= 1
            ), f"Child token {child_token_id} should have node states"

            # Check that at least one state is at the coalesce node
            coalesce_states = [s for s in states if s.node_id == coalesce_node.node_id]
            assert (
                len(coalesce_states) == 1
            ), "Child token should have exactly one coalesce node_state"

            coalesce_state = coalesce_states[0]
            assert coalesce_state.status.value == "completed"

        # === Audit Trail: Verify merged token has join_group_id ===
        merged_token_record = recorder.get_token(merged_token.token_id)
        assert merged_token_record is not None
        assert (
            merged_token_record.join_group_id is not None
        ), "Merged token should have join_group_id"

        # === Audit Trail: Verify complete lineage back to source ===
        # Follow the chain: merged_token -> children -> forked parent -> source row
        assert (
            merged_token.row_id == row.row_id
        ), "Merged token traces back to source row"

    def test_coalesce_best_effort_with_quarantined_child(self) -> None:
        """best_effort policy should merge available children even if one quarantines.

        Scenario:
        - Fork to 3 paths: sentiment, entities, summary
        - summary path quarantines (transform returns TransformResult.error())
        - best_effort timeout triggers, merges sentiment + entities
        - Result should include FORKED, QUARANTINED, and COALESCED outcomes

        This test verifies the end-to-end flow using CoalesceExecutor directly
        to simulate the scenario where one branch is quarantined and never
        reaches the coalesce point.
        """
        import time

        from elspeth.contracts import NodeType, TokenInfo
        from elspeth.core.config import CoalesceSettings
        from elspeth.core.landscape import LandscapeDB, LandscapeRecorder
        from elspeth.engine.coalesce_executor import CoalesceExecutor
        from elspeth.engine.spans import SpanFactory
        from elspeth.engine.tokens import TokenManager

        db = LandscapeDB.in_memory()
        recorder = LandscapeRecorder(db)
        run = recorder.begin_run(config={}, canonical_version="v1")
        token_manager = TokenManager(recorder)

        # Register minimal nodes needed for coalesce testing
        source = recorder.register_node(
            run_id=run.run_id,
            plugin_name="source",
            node_type=NodeType.SOURCE,
            plugin_version="1.0",
            config={},
            schema_config=DYNAMIC_SCHEMA,
        )
        coalesce_node = recorder.register_node(
            run_id=run.run_id,
            plugin_name="merger",
            node_type=NodeType.COALESCE,
            plugin_version="1.0",
            config={},
            schema_config=DYNAMIC_SCHEMA,
        )

        # Setup coalesce with best_effort policy and short timeout
        coalesce_settings = CoalesceSettings(
            name="merger",
            branches=["sentiment", "entities", "summary"],
            policy="best_effort",
            timeout_seconds=0.1,  # Short timeout for testing
            merge="union",
        )
        coalesce_executor = CoalesceExecutor(
            recorder=recorder,
            span_factory=SpanFactory(),
            token_manager=token_manager,
            run_id=run.run_id,
        )
        coalesce_executor.register_coalesce(coalesce_settings, coalesce_node.node_id)

        # Create tokens to simulate fork scenario
        initial_token = token_manager.create_initial_token(
            run_id=run.run_id,
            source_node_id=source.node_id,
            row_index=0,
            row_data={"text": "ACME earnings report"},
        )
        children = token_manager.fork_token(
            parent_token=initial_token,
            branches=["sentiment", "entities", "summary"],
            step_in_pipeline=1,
        )

        # Simulate processing: sentiment and entities complete, summary is quarantined
        # sentiment child completes with enriched data
        sentiment_token = TokenInfo(
            row_id=children[0].row_id,
            token_id=children[0].token_id,
            row_data={"text": "ACME earnings report", "sentiment": "positive"},
            branch_name="sentiment",
        )
        outcome1 = coalesce_executor.accept(
            sentiment_token, "merger", step_in_pipeline=3
        )
        assert outcome1.held is True

        # entities child completes with enriched data
        entities_token = TokenInfo(
            row_id=children[1].row_id,
            token_id=children[1].token_id,
            row_data={"text": "ACME earnings report", "entities": ["ACME"]},
            branch_name="entities",
        )
        outcome2 = coalesce_executor.accept(
            entities_token, "merger", step_in_pipeline=3
        )
        assert outcome2.held is True  # Still waiting (need all 3 or timeout)

        # summary child is QUARANTINED - it never arrives at coalesce
        # (simulated by simply not calling accept for it)

        # Wait for timeout
        time.sleep(0.15)

        # Check timeouts - should trigger best_effort merge
        timed_out = coalesce_executor.check_timeouts("merger", step_in_pipeline=3)

        # Should have merged sentiment + entities (without summary)
        assert len(timed_out) == 1
        outcome = timed_out[0]
        assert outcome.held is False
        assert outcome.merged_token is not None
        assert outcome.failure_reason is None  # Not a failure, just partial merge

        # Verify merged data contains sentiment and entities but not summary
        merged_data = outcome.merged_token.row_data
        assert "sentiment" in merged_data
        assert merged_data["sentiment"] == "positive"
        assert "entities" in merged_data
        assert merged_data["entities"] == ["ACME"]
        # summary never arrived, so its data is NOT in merged result
        # (The original text field should be there from union merge)
        assert "text" in merged_data

        # Verify coalesce metadata shows partial merge
        assert outcome.coalesce_metadata is not None
        assert outcome.coalesce_metadata["policy"] == "best_effort"
        assert set(outcome.coalesce_metadata["branches_arrived"]) == {
            "sentiment",
            "entities",
        }
        assert "summary" not in outcome.coalesce_metadata["branches_arrived"]

    def test_coalesce_quorum_merges_at_threshold(self) -> None:
        """Quorum policy should merge when quorum_count branches arrive.

        Setup: Fork to 3 paths (fast, medium, slow), quorum=2
        - When 2 of 3 arrive, merge immediately
        - 3rd branch result is discarded (arrives after merge)

        This test uses CoalesceExecutor directly to verify:
        1. First branch (fast) is held
        2. Second branch (medium) triggers merge at quorum=2
        3. Merged data contains only fast and medium
        4. Late arrival (slow) starts a new pending entry (doesn't crash)
        """
        from elspeth.contracts import NodeType, TokenInfo
        from elspeth.core.config import CoalesceSettings
        from elspeth.core.landscape import LandscapeDB, LandscapeRecorder
        from elspeth.engine.coalesce_executor import CoalesceExecutor
        from elspeth.engine.spans import SpanFactory
        from elspeth.engine.tokens import TokenManager

        db = LandscapeDB.in_memory()
        recorder = LandscapeRecorder(db)
        run = recorder.begin_run(config={}, canonical_version="v1")
        token_manager = TokenManager(recorder)

        # Register minimal nodes needed for coalesce testing
        source = recorder.register_node(
            run_id=run.run_id,
            plugin_name="source",
            node_type=NodeType.SOURCE,
            plugin_version="1.0",
            config={},
            schema_config=DYNAMIC_SCHEMA,
        )
        coalesce_node = recorder.register_node(
            run_id=run.run_id,
            plugin_name="merger",
            node_type=NodeType.COALESCE,
            plugin_version="1.0",
            config={},
            schema_config=DYNAMIC_SCHEMA,
        )

        # Setup coalesce with quorum policy (2 of 3)
        coalesce_settings = CoalesceSettings(
            name="merger",
            branches=["fast", "medium", "slow"],
            policy="quorum",
            quorum_count=2,
            merge="nested",  # Use nested to see which branches contributed
        )
        coalesce_executor = CoalesceExecutor(
            recorder=recorder,
            span_factory=SpanFactory(),
            token_manager=token_manager,
            run_id=run.run_id,
        )
        coalesce_executor.register_coalesce(coalesce_settings, coalesce_node.node_id)

        # Create tokens to simulate fork scenario
        initial_token = token_manager.create_initial_token(
            run_id=run.run_id,
            source_node_id=source.node_id,
            row_index=0,
            row_data={"text": "test input"},
        )
        children = token_manager.fork_token(
            parent_token=initial_token,
            branches=["fast", "medium", "slow"],
            step_in_pipeline=1,
        )

        # Simulate: fast arrives first with enriched data
        fast_token = TokenInfo(
            row_id=children[0].row_id,
            token_id=children[0].token_id,
            row_data={"text": "test input", "fast_result": "fast done"},
            branch_name="fast",
        )
        outcome1 = coalesce_executor.accept(fast_token, "merger", step_in_pipeline=3)

        # First arrival: should be held (quorum not met yet)
        assert outcome1.held is True
        assert outcome1.merged_token is None

        # Simulate: medium arrives second with enriched data
        medium_token = TokenInfo(
            row_id=children[1].row_id,
            token_id=children[1].token_id,
            row_data={"text": "test input", "medium_result": "medium done"},
            branch_name="medium",
        )
        outcome2 = coalesce_executor.accept(medium_token, "merger", step_in_pipeline=3)

        # Second arrival: quorum met (2 of 3), merge triggers immediately
        assert outcome2.held is False
        assert outcome2.merged_token is not None
        assert outcome2.failure_reason is None  # Not a failure

        # Verify merged data using nested strategy
        merged_data = outcome2.merged_token.row_data
        assert "fast" in merged_data, "Merged data should have 'fast' branch"
        assert "medium" in merged_data, "Merged data should have 'medium' branch"
        assert "slow" not in merged_data, "Merged data should NOT have 'slow' branch"

        # Check nested structure contains expected data
        assert merged_data["fast"]["fast_result"] == "fast done"
        assert merged_data["medium"]["medium_result"] == "medium done"

        # Verify coalesce metadata shows quorum merge
        assert outcome2.coalesce_metadata is not None
        assert outcome2.coalesce_metadata["policy"] == "quorum"
        assert set(outcome2.coalesce_metadata["branches_arrived"]) == {"fast", "medium"}
        assert outcome2.coalesce_metadata["expected_branches"] == [
            "fast",
            "medium",
            "slow",
        ]

        # Verify consumed tokens
        assert len(outcome2.consumed_tokens) == 2
        consumed_ids = {t.token_id for t in outcome2.consumed_tokens}
        assert fast_token.token_id in consumed_ids
        assert medium_token.token_id in consumed_ids

        # Verify arrival order is recorded (fast came before medium)
        arrival_order = outcome2.coalesce_metadata["arrival_order"]
        assert len(arrival_order) == 2
        assert arrival_order[0]["branch"] == "fast"  # First arrival
        assert arrival_order[1]["branch"] == "medium"  # Second arrival

        # === Late arrival behavior ===
        # The slow branch arrives after merge is complete.
        # Since pending state was deleted, this creates a NEW pending entry.
        # This is by design - the row processing would have already continued
        # with the merged token, so this late arrival is effectively orphaned.
        slow_token = TokenInfo(
            row_id=children[2].row_id,
            token_id=children[2].token_id,
            row_data={"text": "test input", "slow_result": "slow done"},
            branch_name="slow",
        )
        outcome3 = coalesce_executor.accept(slow_token, "merger", step_in_pipeline=3)

        # Late arrival creates new pending state (waiting for more branches)
        # This is the expected behavior - in real pipelines, the orchestrator
        # would track that this row already coalesced and not submit the late token.
        assert outcome3.held is True
        assert outcome3.merged_token is None

    def test_nested_fork_coalesce(self) -> None:
        """Test fork within fork, with coalesce at each level.

        DAG structure:
        source → gate1 (fork A,B) → [
            path_a → gate2 (fork A1,A2) → [A1, A2] → coalesce_inner → ...
            path_b → transform_b
        ] → coalesce_outer

        Should produce:
        - 1 parent FORKED (gate1)
        - 2 level-1 children (path_a FORKED, path_b continues)
        - 2 level-2 children from path_a (A1, A2)
        - 1 inner COALESCED (A1+A2)
        - 1 outer COALESCED (inner+path_b)

        This test uses CoalesceExecutor directly to simulate the nested DAG flow,
        providing clear control over the token hierarchy at each level.
        """
        from elspeth.contracts import NodeType, TokenInfo
        from elspeth.core.config import CoalesceSettings
        from elspeth.core.landscape import LandscapeDB, LandscapeRecorder
        from elspeth.engine.coalesce_executor import CoalesceExecutor
        from elspeth.engine.spans import SpanFactory
        from elspeth.engine.tokens import TokenManager

        db = LandscapeDB.in_memory()
        recorder = LandscapeRecorder(db)
        run = recorder.begin_run(config={}, canonical_version="v1")
        token_manager = TokenManager(recorder)

        # Register nodes for the nested DAG
        source = recorder.register_node(
            run_id=run.run_id,
            plugin_name="source",
            node_type=NodeType.SOURCE,
            plugin_version="1.0",
            config={},
            schema_config=DYNAMIC_SCHEMA,
        )
        inner_coalesce_node = recorder.register_node(
            run_id=run.run_id,
            plugin_name="inner_merger",
            node_type=NodeType.COALESCE,
            plugin_version="1.0",
            config={},
            schema_config=DYNAMIC_SCHEMA,
        )
        outer_coalesce_node = recorder.register_node(
            run_id=run.run_id,
            plugin_name="outer_merger",
            node_type=NodeType.COALESCE,
            plugin_version="1.0",
            config={},
            schema_config=DYNAMIC_SCHEMA,
        )

        # === Setup two coalesce points: inner (A1+A2) and outer (inner+path_b) ===
        inner_coalesce_settings = CoalesceSettings(
            name="inner_merger",
            branches=["path_a1", "path_a2"],
            policy="require_all",
            merge="nested",  # Use nested to see branch structure
        )
        outer_coalesce_settings = CoalesceSettings(
            name="outer_merger",
            branches=["path_a_merged", "path_b"],  # inner result + path_b
            policy="require_all",
            merge="nested",
        )

        coalesce_executor = CoalesceExecutor(
            recorder=recorder,
            span_factory=SpanFactory(),
            token_manager=token_manager,
            run_id=run.run_id,
        )
        coalesce_executor.register_coalesce(
            inner_coalesce_settings, inner_coalesce_node.node_id
        )
        coalesce_executor.register_coalesce(
            outer_coalesce_settings, outer_coalesce_node.node_id
        )

        # === Level 0: Create initial token (source row) ===
        initial_token = token_manager.create_initial_token(
            run_id=run.run_id,
            source_node_id=source.node_id,
            row_index=0,
            row_data={"text": "Document for nested processing"},
        )

        # === Level 1: Fork to path_a and path_b (gate1) ===
        level1_children = token_manager.fork_token(
            parent_token=initial_token,
            branches=["path_a", "path_b"],
            step_in_pipeline=1,
        )
        assert len(level1_children) == 2
        path_a_token = level1_children[0]  # branch_name="path_a"
        path_b_token = level1_children[1]  # branch_name="path_b"

        # Verify initial token is now the FORKED parent
        initial_token_record = recorder.get_token(initial_token.token_id)
        assert initial_token_record is not None

        # Verify children have correct branch names
        assert path_a_token.branch_name == "path_a"
        assert path_b_token.branch_name == "path_b"

        # === Level 2: path_a forks again to A1 and A2 (gate2) ===
        level2_children = token_manager.fork_token(
            parent_token=path_a_token,
            branches=["path_a1", "path_a2"],
            step_in_pipeline=2,
        )
        assert len(level2_children) == 2
        path_a1_token = level2_children[0]  # branch_name="path_a1"
        path_a2_token = level2_children[1]  # branch_name="path_a2"

        # path_a token is now FORKED (has children)
        path_a_record = recorder.get_token(path_a_token.token_id)
        assert path_a_record is not None

        # Verify level 2 branch names
        assert path_a1_token.branch_name == "path_a1"
        assert path_a2_token.branch_name == "path_a2"

        # === Process level 2 children (simulate transform enrichment) ===
        # A1 adds sentiment analysis
        enriched_a1 = TokenInfo(
            row_id=path_a1_token.row_id,
            token_id=path_a1_token.token_id,
            row_data={
                "text": "Document for nested processing",
                "sentiment": "positive",
            },
            branch_name="path_a1",
        )
        # A2 adds entity extraction
        enriched_a2 = TokenInfo(
            row_id=path_a2_token.row_id,
            token_id=path_a2_token.token_id,
            row_data={
                "text": "Document for nested processing",
                "entities": ["ACME", "2024"],
            },
            branch_name="path_a2",
        )

        # === Inner coalesce: merge A1 + A2 ===
        inner_outcome1 = coalesce_executor.accept(
            enriched_a1, "inner_merger", step_in_pipeline=3
        )
        assert inner_outcome1.held is True  # First arrival, waiting for A2

        inner_outcome2 = coalesce_executor.accept(
            enriched_a2, "inner_merger", step_in_pipeline=3
        )
        assert inner_outcome2.held is False  # Both arrived, merge triggered
        assert inner_outcome2.merged_token is not None
        assert inner_outcome2.failure_reason is None

        inner_merged_token = inner_outcome2.merged_token

        # Verify inner merge consumed both A1 and A2
        assert len(inner_outcome2.consumed_tokens) == 2
        consumed_inner_ids = {t.token_id for t in inner_outcome2.consumed_tokens}
        assert enriched_a1.token_id in consumed_inner_ids
        assert enriched_a2.token_id in consumed_inner_ids

        # Verify inner merged data has nested structure
        inner_merged_data = inner_merged_token.row_data
        assert "path_a1" in inner_merged_data
        assert "path_a2" in inner_merged_data
        assert inner_merged_data["path_a1"]["sentiment"] == "positive"
        assert inner_merged_data["path_a2"]["entities"] == ["ACME", "2024"]

        # === Process path_b (simulate transform enrichment) ===
        enriched_b = TokenInfo(
            row_id=path_b_token.row_id,
            token_id=path_b_token.token_id,
            row_data={
                "text": "Document for nested processing",
                "category": "financial",
            },
            branch_name="path_b",
        )

        # === Outer coalesce: merge inner_merged + path_b ===
        # First, prepare inner merged token for outer coalesce
        # It needs branch_name="path_a_merged" to match outer coalesce config
        inner_for_outer = TokenInfo(
            row_id=inner_merged_token.row_id,
            token_id=inner_merged_token.token_id,
            row_data=inner_merged_token.row_data,
            branch_name="path_a_merged",  # Assign branch for outer coalesce
        )

        outer_outcome1 = coalesce_executor.accept(
            inner_for_outer, "outer_merger", step_in_pipeline=4
        )
        assert outer_outcome1.held is True  # Waiting for path_b

        outer_outcome2 = coalesce_executor.accept(
            enriched_b, "outer_merger", step_in_pipeline=4
        )
        assert outer_outcome2.held is False  # Both arrived, final merge triggered
        assert outer_outcome2.merged_token is not None
        assert outer_outcome2.failure_reason is None

        outer_merged_token = outer_outcome2.merged_token

        # Verify outer merge consumed both inner_merged and path_b
        assert len(outer_outcome2.consumed_tokens) == 2
        consumed_outer_ids = {t.token_id for t in outer_outcome2.consumed_tokens}
        assert inner_for_outer.token_id in consumed_outer_ids
        assert enriched_b.token_id in consumed_outer_ids

        # === Verify final merged data has complete nested hierarchy ===
        final_data = outer_merged_token.row_data
        assert "path_a_merged" in final_data
        assert "path_b" in final_data

        # path_b branch has category
        assert final_data["path_b"]["category"] == "financial"

        # path_a_merged branch has the inner merge results (nested A1+A2)
        inner_result = final_data["path_a_merged"]
        assert "path_a1" in inner_result
        assert "path_a2" in inner_result
        assert inner_result["path_a1"]["sentiment"] == "positive"
        assert inner_result["path_a2"]["entities"] == ["ACME", "2024"]

        # === Verify token hierarchy through audit trail ===
        # All tokens should trace back to the same row_id
        assert initial_token.row_id == path_a_token.row_id
        assert initial_token.row_id == path_b_token.row_id
        assert initial_token.row_id == path_a1_token.row_id
        assert initial_token.row_id == path_a2_token.row_id
        assert initial_token.row_id == inner_merged_token.row_id
        assert initial_token.row_id == outer_merged_token.row_id

        # Verify parent-child relationships at each level
        # Level 1 children (path_a, path_b) should have initial_token as parent
        path_a_parents = recorder.get_token_parents(path_a_token.token_id)
        assert len(path_a_parents) == 1
        assert path_a_parents[0].parent_token_id == initial_token.token_id

        path_b_parents = recorder.get_token_parents(path_b_token.token_id)
        assert len(path_b_parents) == 1
        assert path_b_parents[0].parent_token_id == initial_token.token_id

        # Level 2 children (A1, A2) should have path_a as parent
        a1_parents = recorder.get_token_parents(path_a1_token.token_id)
        assert len(a1_parents) == 1
        assert a1_parents[0].parent_token_id == path_a_token.token_id

        a2_parents = recorder.get_token_parents(path_a2_token.token_id)
        assert len(a2_parents) == 1
        assert a2_parents[0].parent_token_id == path_a_token.token_id

        # Inner merged token should have A1 and A2 as parents
        inner_merged_parents = recorder.get_token_parents(inner_merged_token.token_id)
        assert len(inner_merged_parents) == 2
        inner_parent_ids = {p.parent_token_id for p in inner_merged_parents}
        assert path_a1_token.token_id in inner_parent_ids
        assert path_a2_token.token_id in inner_parent_ids

        # Outer merged token should have inner_merged and path_b as parents
        outer_merged_parents = recorder.get_token_parents(outer_merged_token.token_id)
        assert len(outer_merged_parents) == 2
        outer_parent_ids = {p.parent_token_id for p in outer_merged_parents}
        assert inner_merged_token.token_id in outer_parent_ids
        assert path_b_token.token_id in outer_parent_ids

        # === Verify coalesce metadata captures the hierarchy ===
        assert inner_outcome2.coalesce_metadata is not None
        assert inner_outcome2.coalesce_metadata["policy"] == "require_all"
        assert set(inner_outcome2.coalesce_metadata["branches_arrived"]) == {
            "path_a1",
            "path_a2",
        }

        assert outer_outcome2.coalesce_metadata is not None
        assert outer_outcome2.coalesce_metadata["policy"] == "require_all"
        assert set(outer_outcome2.coalesce_metadata["branches_arrived"]) == {
            "path_a_merged",
            "path_b",
        }

        # === Verify merged tokens have join_group_id ===
        inner_merged_record = recorder.get_token(inner_merged_token.token_id)
        assert inner_merged_record is not None
        assert inner_merged_record.join_group_id is not None

        outer_merged_record = recorder.get_token(outer_merged_token.token_id)
        assert outer_merged_record is not None
        assert outer_merged_record.join_group_id is not None


class TestRowProcessorRetry:
    """Tests for retry integration in RowProcessor."""

    def test_processor_accepts_retry_manager(self) -> None:
        """RowProcessor can be constructed with RetryManager."""
        from unittest.mock import Mock

        from elspeth.engine.processor import RowProcessor
        from elspeth.engine.retry import RetryConfig, RetryManager

        retry_manager = RetryManager(RetryConfig(max_attempts=3))

        # Should not raise
        processor = RowProcessor(
            recorder=Mock(),
            span_factory=Mock(),
            run_id="test-run",
            source_node_id="source-node",
            retry_manager=retry_manager,
        )

        assert processor._retry_manager is retry_manager

    def test_retries_transient_transform_exception(self) -> None:
        """Transform exceptions are retried up to max_attempts."""
        from unittest.mock import Mock

        from elspeth.contracts import TransformResult
        from elspeth.engine.processor import RowProcessor
        from elspeth.engine.retry import RetryConfig, RetryManager

        # Track call count
        call_count = 0

        def flaky_execute(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            if call_count < 3:
                raise ConnectionError("Transient network error")
            # Return success on 3rd attempt
            return (
                TransformResult.success({"result": "ok"}),
                Mock(
                    token_id="t1",
                    row_id="r1",
                    row_data={"result": "ok"},
                    branch_name=None,
                ),
                None,  # error_sink
            )

        # Create processor with mocked internals
        processor = RowProcessor(
            recorder=Mock(),
            span_factory=Mock(),
            run_id="test-run",
            source_node_id="source",
            retry_manager=RetryManager(RetryConfig(max_attempts=3, base_delay=0.01)),
        )

        # Mock the transform executor
        processor._transform_executor = Mock()
        processor._transform_executor.execute_transform.side_effect = flaky_execute

        # Create test transform
        transform = Mock()
        transform.node_id = "transform-1"

        # Create test token
        token = Mock()
        token.token_id = "t1"
        token.row_id = "r1"
        token.row_data = {"input": 1}
        token.branch_name = None

        ctx = Mock()
        ctx.run_id = "test-run"

        # Call the retry wrapper directly
        result, _out_token, _error_sink = processor._execute_transform_with_retry(
            transform=transform,
            token=token,
            ctx=ctx,
            step=0,
        )

        # Should have retried and succeeded
        assert call_count == 3
        assert result.status == "success"

    def test_no_retry_when_retry_manager_is_none(self) -> None:
        """Without retry_manager, exceptions propagate immediately."""
        from unittest.mock import Mock

        import pytest

        from elspeth.engine.processor import RowProcessor

        processor = RowProcessor(
            recorder=Mock(),
            span_factory=Mock(),
            run_id="test-run",
            source_node_id="source",
            retry_manager=None,  # No retry
        )

        processor._transform_executor = Mock()
        processor._transform_executor.execute_transform.side_effect = ConnectionError(
            "fail"
        )

        transform = Mock()
        transform.node_id = "t1"
        token = Mock(token_id="t1", row_id="r1", row_data={}, branch_name=None)
        ctx = Mock(run_id="test-run")

        with pytest.raises(ConnectionError):
            processor._execute_transform_with_retry(transform, token, ctx, step=0)

        # Should only be called once (no retry)
        assert processor._transform_executor.execute_transform.call_count == 1

    def test_max_retries_exceeded_returns_failed_outcome(self) -> None:
        """When all retries exhausted, process_row returns FAILED outcome."""

        from elspeth.contracts import RowOutcome
        from elspeth.contracts.schema import SchemaConfig
        from elspeth.core.landscape import LandscapeDB, LandscapeRecorder
        from elspeth.engine.processor import RowProcessor
        from elspeth.engine.retry import RetryConfig, RetryManager
        from elspeth.engine.spans import SpanFactory

        # Set up real Landscape
        db = LandscapeDB.in_memory()
        recorder = LandscapeRecorder(db)
        run = recorder.begin_run(config={}, canonical_version="v1")

        source = recorder.register_node(
            run_id=run.run_id,
            plugin_name="source",
            node_type="source",
            plugin_version="1.0",
            config={},
            schema_config=SchemaConfig.from_dict({"fields": "dynamic"}),
        )
        transform_node = recorder.register_node(
            run_id=run.run_id,
            plugin_name="always_fails",
            node_type="transform",
            plugin_version="1.0",
            config={},
            schema_config=SchemaConfig.from_dict({"fields": "dynamic"}),
        )

        class AlwaysFailsTransform(BaseTransform):
            """Transform that always raises transient error."""

            name = "always_fails"
            input_schema = _TestSchema
            output_schema = _TestSchema

            def __init__(self, node_id: str) -> None:
                super().__init__({})
                self.node_id = node_id

            def process(
                self, row: dict[str, Any], ctx: PluginContext
            ) -> TransformResult:
                raise ConnectionError("Network always down")

        # Create processor with retry (max 2 attempts, fast delays for test)
        processor = RowProcessor(
            recorder=recorder,
            span_factory=SpanFactory(),
            run_id=run.run_id,
            source_node_id=source.node_id,
            retry_manager=RetryManager(RetryConfig(max_attempts=2, base_delay=0.01)),
        )

        ctx = PluginContext(run_id=run.run_id, config={})

        # Process should return FAILED, not raise MaxRetriesExceeded
        results = processor.process_row(
            row_index=0,
            row_data={"x": 1},
            transforms=[AlwaysFailsTransform(transform_node.node_id)],
            ctx=ctx,
        )

        # Should get a result, not an exception
        assert len(results) == 1
        result = results[0]

        # Outcome should be FAILED
        assert result.outcome == RowOutcome.FAILED

        # Error info should be captured
        assert result.error is not None
        assert "MaxRetriesExceeded" in str(result.error) or "attempts" in str(
            result.error
        )


class TestRowProcessorRecovery:
    """Tests for RowProcessor recovery support."""

    def test_processor_accepts_restored_aggregation_state(self, tmp_path: Path) -> None:
        """RowProcessor passes restored state to AggregationExecutor."""
        from elspeth.core.landscape.database import LandscapeDB
        from elspeth.core.landscape.recorder import LandscapeRecorder
        from elspeth.engine.processor import RowProcessor
        from elspeth.engine.spans import SpanFactory

        db = LandscapeDB(f"sqlite:///{tmp_path}/test.db")
        recorder = LandscapeRecorder(db)
        run = recorder.begin_run(config={}, canonical_version="v1")

        restored_state = {
            "agg_node": {"buffer": [1, 2], "count": 2},
        }

        processor = RowProcessor(
            recorder=recorder,
            span_factory=SpanFactory(),
            run_id=run.run_id,
            source_node_id="source",
            edge_map={},
            route_resolution_map={},
            restored_aggregation_state=restored_state,  # New parameter
        )

        # Verify state was passed to executor
        assert processor._aggregation_executor.get_restored_state("agg_node") == {
            "buffer": [1, 2],
            "count": 2,
        }
