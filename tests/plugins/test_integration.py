# tests/plugins/test_integration.py
"""Integration tests for the plugin system."""

from collections.abc import Iterator
from typing import Any, ClassVar


class TestPluginSystemIntegration:
    """End-to-end plugin system tests."""

    def test_full_plugin_workflow(self) -> None:
        """Test source -> transform -> sink workflow."""
        from elspeth.contracts import ArtifactDescriptor, SourceRow
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

            def load(self, ctx: PluginContext) -> Iterator[SourceRow]:
                for v in self.config["values"]:
                    yield SourceRow.valid({"value": v})

            def close(self) -> None:
                pass

        class DoubleTransform(BaseTransform):
            name = "double"
            input_schema = InputSchema
            output_schema = EnrichedSchema

            def process(self, row: dict[str, Any], ctx: PluginContext) -> TransformResult:
                return TransformResult.success(
                    {
                        "value": row["value"],
                        "doubled": row["value"] * 2,
                    },
                    success_reason={"action": "double"},
                )

        class MemorySink(BaseSink):
            name = "memory"
            input_schema = EnrichedSchema
            collected: ClassVar[list[dict[str, Any]]] = []

            def write(self, rows: list[dict[str, Any]], ctx: PluginContext) -> ArtifactDescriptor:
                MemorySink.collected.extend(rows)
                return ArtifactDescriptor.for_file(path="memory://collected", content_hash="test", size_bytes=0)

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

        for source_row in source.load(ctx):
            # Extract row data from SourceRow for transform
            assert source_row.row is not None
            result = transform.process(source_row.row, ctx)
            assert result.status == "success"
            assert result.row is not None  # Success always has row
            sink.write([result.row], ctx)  # write() takes list of rows

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

    def test_aggregation_workflow_deleted(self) -> None:
        """Verify BaseAggregation deleted (aggregation is now structural).

        OLD: Tested accept/flush contract with BaseAggregation plugin.
        NEW: Aggregation is engine-controlled via batch-aware transforms
             with is_batch_aware=True, no plugin-level aggregation interface.
        """
        import elspeth.plugins.base as base

        assert not hasattr(base, "BaseAggregation"), "BaseAggregation should be deleted - use is_batch_aware=True on BaseTransform"
