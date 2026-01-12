# tests/plugins/test_integration.py
"""Integration tests for the plugin system."""

from collections.abc import Iterator
from typing import Any, ClassVar


class TestPluginSystemIntegration:
    """End-to-end plugin system tests."""

    def test_full_plugin_workflow(self) -> None:
        """Test source -> transform -> gate -> sink workflow."""
        from elspeth.plugins import (
            BaseGate,
            BaseSink,
            BaseSource,
            BaseTransform,
            GateResult,
            PluginContext,
            PluginManager,
            PluginSchema,
            RoutingAction,
            TransformResult,
            hookimpl,
        )

        # Define schemas
        class InputSchema(PluginSchema):
            value: int

        class EnrichedSchema(PluginSchema):
            value: int
            doubled: int

        # Define plugins
        class ListSource(BaseSource):
            name = "list"
            output_schema = InputSchema

            def load(self, ctx: PluginContext) -> Iterator[dict[str, Any]]:
                for v in self.config["values"]:
                    yield {"value": v}

            def close(self) -> None:
                pass

        class DoubleTransform(BaseTransform):
            name = "double"
            input_schema = InputSchema
            output_schema = EnrichedSchema

            def process(self, row: dict[str, Any], ctx: PluginContext) -> TransformResult:
                return TransformResult.success({
                    "value": row["value"],
                    "doubled": row["value"] * 2,
                })

        class ThresholdGate(BaseGate):
            name = "threshold"
            input_schema = EnrichedSchema
            output_schema = EnrichedSchema

            def evaluate(self, row: dict[str, Any], ctx: PluginContext) -> GateResult:
                if row["doubled"] > self.config["threshold"]:
                    return GateResult(
                        row=row,
                        action=RoutingAction.route_to_sink("high"),
                    )
                return GateResult(row=row, action=RoutingAction.continue_())

        class MemorySink(BaseSink):
            name = "memory"
            input_schema = EnrichedSchema
            collected: ClassVar[list[dict[str, Any]]] = []

            def write(self, row: dict[str, Any], ctx: PluginContext) -> None:
                MemorySink.collected.append(row)

            def flush(self) -> None:
                pass

            def close(self) -> None:
                pass

        # Register plugins
        class TestPlugin:
            @hookimpl
            def elspeth_get_source(self) -> list[type[BaseSource]]:
                return [ListSource]

            @hookimpl
            def elspeth_get_transforms(self) -> list[type[BaseTransform]]:
                return [DoubleTransform]

            @hookimpl
            def elspeth_get_gates(self) -> list[type[BaseGate]]:
                return [ThresholdGate]

            @hookimpl
            def elspeth_get_sinks(self) -> list[type[BaseSink]]:
                return [MemorySink]

        manager = PluginManager()
        manager.register(TestPlugin())

        # Verify registration
        assert len(manager.get_sources()) == 1
        assert len(manager.get_transforms()) == 1
        assert len(manager.get_gates()) == 1
        assert len(manager.get_sinks()) == 1

        # Create instances and process
        ctx = PluginContext(run_id="test-001", config={})

        source = manager.get_source_by_name("list")({"values": [10, 50, 100]})
        transform = manager.get_transform_by_name("double")({})
        gate = manager.get_gate_by_name("threshold")({"threshold": 100})
        sink = manager.get_sink_by_name("memory")({})

        MemorySink.collected = []  # Reset

        for row in source.load(ctx):
            result = transform.process(row, ctx)
            assert result.status == "success"

            gate_result = gate.evaluate(result.row, ctx)

            if gate_result.action.kind == "continue":
                sink.write(gate_result.row, ctx)

        # Verify results
        # Values: 10*2=20, 50*2=100, 100*2=200
        # Threshold 100: 20 continues, 100 continues, 200 routed
        assert len(MemorySink.collected) == 2
        assert MemorySink.collected[0]["doubled"] == 20
        assert MemorySink.collected[1]["doubled"] == 100

    def test_schema_validation_in_pipeline(self) -> None:
        """Test that schema compatibility is checked."""
        from elspeth.plugins import PluginSchema, check_compatibility

        class SourceOutput(PluginSchema):
            a: int
            b: str

        class TransformInput(PluginSchema):
            a: int
            b: str
            c: float  # Not provided by source!

        result = check_compatibility(SourceOutput, TransformInput)
        assert result.compatible is False
        assert "c" in result.missing_fields

    def test_aggregation_workflow(self) -> None:
        """Test aggregation batching behavior."""
        from elspeth.plugins import (
            AcceptResult,
            BaseAggregation,
            PluginContext,
            PluginSchema,
        )

        class InputSchema(PluginSchema):
            value: int

        class OutputSchema(PluginSchema):
            total: int
            count: int

        class SumAggregation(BaseAggregation):
            name = "sum"
            input_schema = InputSchema
            output_schema = OutputSchema

            def __init__(self, config: dict[str, Any]) -> None:
                super().__init__(config)
                self._values: list[int] = []

            def accept(self, row: dict[str, Any], ctx: PluginContext) -> AcceptResult:
                self._values.append(row["value"])
                trigger = len(self._values) >= self.config["batch_size"]
                return AcceptResult(accepted=True, trigger=trigger)

            def should_trigger(self) -> bool:
                return len(self._values) >= self.config["batch_size"]

            def flush(self, ctx: PluginContext) -> list[dict[str, Any]]:
                result = {
                    "total": sum(self._values),
                    "count": len(self._values),
                }
                self._values = []
                return [result]

        agg = SumAggregation({"batch_size": 3})
        ctx = PluginContext(run_id="test", config={})

        # Add 5 values: should trigger at 3, then have 2 remaining
        values = [10, 20, 30, 40, 50]
        outputs = []

        for v in values:
            result = agg.accept({"value": v}, ctx)
            if result.trigger:
                outputs.extend(agg.flush(ctx))

        # Force final flush
        if agg.should_trigger() or len(agg._values) > 0:
            outputs.extend(agg.flush(ctx))

        # First batch: 10+20+30=60, Second batch: 40+50=90
        assert len(outputs) == 2
        assert outputs[0] == {"total": 60, "count": 3}
        assert outputs[1] == {"total": 90, "count": 2}
