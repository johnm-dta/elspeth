# tests/plugins/test_integration.py
"""Integration tests for the plugin system."""

from collections.abc import Iterator
from typing import Any, ClassVar

from elspeth.contracts.schema_contract import PipelineRow
from elspeth.testing import make_field, make_pipeline_row
from tests.fixtures.factories import make_context
from tests.fixtures.landscape import make_factory


class TestPluginSystemIntegration:
    """End-to-end plugin system tests."""

    def test_full_plugin_workflow(self) -> None:
        """Test source -> transform -> sink workflow."""
        from elspeth.contracts import ArtifactDescriptor, PluginSchema, SourceRow
        from elspeth.contracts.contexts import SinkContext, SourceContext, TransformContext
        from elspeth.contracts.diversion import SinkWriteResult
        from elspeth.contracts.schema_contract import SchemaContract
        from elspeth.plugins.infrastructure.base import BaseSink, BaseSource, BaseTransform
        from elspeth.plugins.infrastructure.hookspecs import hookimpl
        from elspeth.plugins.infrastructure.manager import PluginManager
        from elspeth.plugins.infrastructure.results import TransformResult

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

            def load(self, ctx: SourceContext) -> Iterator[SourceRow]:
                # Create schema contract for output
                contract = SchemaContract(
                    mode="FIXED",
                    fields=(make_field("value", int, original_name="value", required=True, source="declared"),),
                    locked=True,
                )
                for v in self.config["values"]:
                    yield SourceRow.valid({"value": v}, contract=contract)

            def close(self) -> None:
                pass

        class DoubleTransform(BaseTransform):
            name = "double"
            input_schema = InputSchema
            output_schema = EnrichedSchema

            def process(self, row: PipelineRow, ctx: TransformContext) -> TransformResult:
                row_dict = row.to_dict()
                return TransformResult.success(
                    make_pipeline_row(
                        {
                            "value": row_dict["value"],
                            "doubled": row_dict["value"] * 2,
                        }
                    ),
                    success_reason={"action": "double"},
                )

        class MemorySink(BaseSink):
            name = "memory"
            input_schema = EnrichedSchema
            _on_write_failure: str | None = "discard"
            collected: ClassVar[list[dict[str, Any]]] = []

            def write(self, rows: list[dict[str, Any]], ctx: SinkContext) -> SinkWriteResult:
                MemorySink.collected.extend(rows)
                return SinkWriteResult(artifact=ArtifactDescriptor.for_file(path="memory://collected", content_hash="test", size_bytes=0))

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
        factory = make_factory()
        ctx = make_context(run_id="test-001", landscape=factory.plugin_audit_writer())

        source_cls = manager.get_source_by_name("list")
        transform_cls = manager.get_transform_by_name("double")
        sink_cls = manager.get_sink_by_name("memory")

        assert source_cls is not None
        assert transform_cls is not None
        assert sink_cls is not None

        # Protocols don't define __init__ but concrete classes do
        source = source_cls({"values": [10, 50, 100]})
        transform = transform_cls({})
        sink = sink_cls({})

        MemorySink.collected = []  # Reset

        for source_row in source.load(ctx):
            # Convert SourceRow to PipelineRow for transform processing
            assert source_row.row is not None
            pipeline_row = source_row.to_pipeline_row()
            result = transform.process(pipeline_row, ctx)
            assert result.status == "success"
            assert result.row is not None  # Success always has row
            row_data = result.row.to_dict() if isinstance(result.row, PipelineRow) else result.row
            sink.write([row_data], ctx)  # write() takes list of rows

        # Verify results
        # Values: 10*2=20, 50*2=100, 100*2=200
        assert len(MemorySink.collected) == 3
        assert MemorySink.collected[0]["doubled"] == 20
        assert MemorySink.collected[1]["doubled"] == 100
        assert MemorySink.collected[2]["doubled"] == 200

    def test_schema_validation_in_pipeline(self) -> None:
        """Test that schema compatibility is checked."""
        from elspeth.contracts import PluginSchema, check_compatibility

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
        import elspeth.plugins.infrastructure.base as base

        assert not hasattr(base, "BaseAggregation"), "BaseAggregation should be deleted - use is_batch_aware=True on BaseTransform"
