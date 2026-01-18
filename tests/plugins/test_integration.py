# tests/plugins/test_integration.py
"""Integration tests for the plugin system."""

from collections.abc import Iterator
from typing import Any, ClassVar


class TestPluginSystemIntegration:
    """End-to-end plugin system tests."""

    def test_full_plugin_workflow(self) -> None:
        """Test source -> transform -> sink workflow."""
        from elspeth.plugins import (
            BaseSink,
            BaseSource,
            BaseTransform,
            PluginContext,
            PluginManager,
            PluginSchema,
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

            def process(
                self, row: dict[str, Any], ctx: PluginContext
            ) -> TransformResult:
                return TransformResult.success(
                    {
                        "value": row["value"],
                        "doubled": row["value"] * 2,
                    }
                )

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
            def elspeth_get_sinks(self) -> list[type[BaseSink]]:
                return [MemorySink]

        manager = PluginManager()
        manager.register(TestPlugin())

        # Verify registration
        assert len(manager.get_sources()) == 1
        assert len(manager.get_transforms()) == 1
        assert len(manager.get_sinks()) == 1

        # Create instances and process
        ctx = PluginContext(run_id="test-001", config={})

        source_cls = manager.get_source_by_name("list")
        transform_cls = manager.get_transform_by_name("double")
        sink_cls = manager.get_sink_by_name("memory")

        assert source_cls is not None
        assert transform_cls is not None
        assert sink_cls is not None

        # Protocols don't define __init__ but concrete classes do
        source = source_cls({"values": [10, 50, 100]})  # type: ignore[call-arg]
        transform = transform_cls({})  # type: ignore[call-arg]
        sink = sink_cls({})  # type: ignore[call-arg]

        MemorySink.collected = []  # Reset

        for row in source.load(ctx):
            result = transform.process(row, ctx)
            assert result.status == "success"
            assert result.row is not None  # Success always has row
            sink.write(result.row, ctx)

        # Verify results
        # Values: 10*2=20, 50*2=100, 100*2=200
        assert len(MemorySink.collected) == 3
        assert MemorySink.collected[0]["doubled"] == 20
        assert MemorySink.collected[1]["doubled"] == 100
        assert MemorySink.collected[2]["doubled"] == 200

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
        """Test aggregation batching behavior.

        Note: Trigger evaluation is now engine-controlled via TriggerEvaluator (WP-06).
        This test demonstrates the aggregation plugin's accept/flush contract.
        """
        from elspeth.core.config import TriggerConfig
        from elspeth.engine.triggers import TriggerEvaluator
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
                return AcceptResult(accepted=True)

            def flush(self, ctx: PluginContext) -> list[dict[str, Any]]:
                if not self._values:
                    return []
                result = {
                    "total": sum(self._values),
                    "count": len(self._values),
                }
                self._values = []
                return [result]

        agg = SumAggregation({})
        ctx = PluginContext(run_id="test", config={})
        # Engine uses TriggerEvaluator to manage trigger conditions (WP-06)
        trigger_evaluator = TriggerEvaluator(TriggerConfig(count=3))

        # Add 5 values: trigger fires at count=3, then again at end-of-source
        values = [10, 20, 30, 40, 50]
        outputs = []

        for v in values:
            result = agg.accept({"value": v}, ctx)
            if result.accepted:
                trigger_evaluator.record_accept()
                if trigger_evaluator.should_trigger():
                    outputs.extend(agg.flush(ctx))
                    trigger_evaluator.reset()

        # Force final flush for any remaining items (end-of-source)
        outputs.extend(agg.flush(ctx))

        # First batch: 10+20+30=60, Second batch: 40+50=90
        assert len(outputs) == 2
        assert outputs[0] == {"total": 60, "count": 3}
        assert outputs[1] == {"total": 90, "count": 2}
