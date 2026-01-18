# tests/plugins/test_base.py
"""Tests for plugin base classes."""

from typing import Any

import pytest


class TestBaseTransform:
    """Base class for transforms."""

    def test_base_transform_abstract(self) -> None:
        from elspeth.plugins.base import BaseTransform

        # Should not be instantiable directly
        with pytest.raises(TypeError):
            BaseTransform({})  # type: ignore[abstract]

    def test_subclass_implementation(self) -> None:
        from elspeth.contracts import PluginSchema
        from elspeth.plugins.base import BaseTransform
        from elspeth.plugins.context import PluginContext
        from elspeth.plugins.results import TransformResult

        class InputSchema(PluginSchema):
            x: int

        class OutputSchema(PluginSchema):
            x: int
            doubled: int

        class DoubleTransform(BaseTransform):
            name = "double"
            input_schema = InputSchema
            output_schema = OutputSchema

            def process(
                self, row: dict[str, Any], ctx: PluginContext
            ) -> TransformResult:
                return TransformResult.success(
                    {
                        "x": row["x"],
                        "doubled": row["x"] * 2,
                    }
                )

        transform = DoubleTransform({"some": "config"})
        ctx = PluginContext(run_id="test", config={})

        result = transform.process({"x": 21}, ctx)
        assert result.row == {"x": 21, "doubled": 42}

    def test_lifecycle_hooks_exist(self) -> None:
        from elspeth.plugins.base import BaseTransform

        # These should exist as no-op methods
        assert hasattr(BaseTransform, "on_register")
        assert hasattr(BaseTransform, "on_start")
        assert hasattr(BaseTransform, "on_complete")


class TestBaseAggregation:
    """Base class for aggregations."""

    def test_base_aggregation_implementation(self) -> None:
        from elspeth.contracts import PluginSchema
        from elspeth.plugins.base import BaseAggregation
        from elspeth.plugins.context import PluginContext
        from elspeth.plugins.results import AcceptResult

        class InputSchema(PluginSchema):
            value: int

        class OutputSchema(PluginSchema):
            total: int

        class SumAggregation(BaseAggregation):
            name = "sum"
            input_schema = InputSchema
            output_schema = OutputSchema

            def __init__(self, config: dict[str, Any]) -> None:
                super().__init__(config)
                self._values: list[int] = []
                self._batch_size: int = config["batch_size"]

            def accept(self, row: dict[str, Any], ctx: PluginContext) -> AcceptResult:
                self._values.append(row["value"])
                return AcceptResult(accepted=True)

            def should_trigger(self) -> bool:
                return len(self._values) >= self._batch_size

            def flush(self, ctx: PluginContext) -> list[dict[str, Any]]:
                result = {"total": sum(self._values)}
                self._values = []
                return [result]

        agg = SumAggregation({"batch_size": 2})
        ctx = PluginContext(run_id="test", config={})

        agg.accept({"value": 10}, ctx)
        agg.accept({"value": 20}, ctx)
        assert agg.should_trigger() is True

        outputs = agg.flush(ctx)
        assert outputs == [{"total": 30}]


class TestBaseSink:
    """Base class for sinks."""

    def test_base_sink_implementation(self) -> None:
        from elspeth.contracts import ArtifactDescriptor, PluginSchema
        from elspeth.plugins.base import BaseSink
        from elspeth.plugins.context import PluginContext

        class InputSchema(PluginSchema):
            value: int

        class MemorySink(BaseSink):
            name = "memory"
            input_schema = InputSchema
            idempotent = True

            def __init__(self, config: dict[str, Any]) -> None:
                super().__init__(config)
                self.rows: list[dict[str, Any]] = []

            def write(
                self, rows: list[dict[str, Any]], ctx: PluginContext
            ) -> ArtifactDescriptor:
                self.rows.extend(rows)
                return ArtifactDescriptor.for_file(
                    path="/tmp/memory",
                    content_hash="test",
                    size_bytes=len(str(rows)),
                )

            def flush(self) -> None:
                pass

            def close(self) -> None:
                pass

        sink = MemorySink({})
        ctx = PluginContext(run_id="test", config={})

        artifact = sink.write([{"value": 1}, {"value": 2}], ctx)

        assert len(sink.rows) == 2
        assert sink.rows[0] == {"value": 1}
        assert isinstance(artifact, ArtifactDescriptor)

    def test_base_sink_batch_write_signature(self) -> None:
        """BaseSink.write() accepts batch and returns ArtifactDescriptor."""
        import inspect

        from elspeth.plugins.base import BaseSink

        sig = inspect.signature(BaseSink.write)
        params = list(sig.parameters.keys())

        assert "rows" in params, "write() should accept 'rows' (batch)"
        assert "row" not in params, "write() should NOT have 'row' parameter"

    def test_base_sink_batch_implementation(self) -> None:
        """Test BaseSink subclass with batch write."""
        from elspeth.contracts import ArtifactDescriptor, PluginSchema
        from elspeth.plugins.base import BaseSink
        from elspeth.plugins.context import PluginContext

        class InputSchema(PluginSchema):
            value: int

        class BatchMemorySink(BaseSink):
            name = "batch_memory"
            input_schema = InputSchema
            idempotent = True

            def __init__(self, config: dict[str, Any]) -> None:
                super().__init__(config)
                self.rows: list[dict[str, Any]] = []

            def write(
                self, rows: list[dict[str, Any]], ctx: PluginContext
            ) -> ArtifactDescriptor:
                self.rows.extend(rows)
                return ArtifactDescriptor.for_file(
                    path="/tmp/batch",
                    content_hash="hash123",
                    size_bytes=100,
                )

            def flush(self) -> None:
                pass

            def close(self) -> None:
                pass

        sink = BatchMemorySink({})
        ctx = PluginContext(run_id="test", config={})

        artifact = sink.write([{"value": 1}, {"value": 2}, {"value": 3}], ctx)

        assert len(sink.rows) == 3
        assert isinstance(artifact, ArtifactDescriptor)
        assert artifact.content_hash == "hash123"

    def test_base_sink_has_io_write_determinism(self) -> None:
        """BaseSink should have IO_WRITE determinism by default."""
        from elspeth.contracts import Determinism
        from elspeth.plugins.base import BaseSink

        assert BaseSink.determinism == Determinism.IO_WRITE


class TestBaseSource:
    """Base class for sources."""

    def test_base_source_implementation(self) -> None:
        from collections.abc import Iterator

        from elspeth.contracts import PluginSchema
        from elspeth.plugins.base import BaseSource
        from elspeth.plugins.context import PluginContext

        class OutputSchema(PluginSchema):
            value: int

        class ListSource(BaseSource):
            name = "list"
            output_schema = OutputSchema

            def __init__(self, config: dict[str, Any]) -> None:
                super().__init__(config)
                self._data = config["data"]

            def load(self, ctx: PluginContext) -> Iterator[dict[str, Any]]:
                yield from self._data

            def close(self) -> None:
                pass

        source = ListSource({"data": [{"value": 1}, {"value": 2}]})
        ctx = PluginContext(run_id="test", config={})

        rows = list(source.load(ctx))
        assert len(rows) == 2
        assert rows[0] == {"value": 1}

    def test_base_source_has_metadata_attributes(self) -> None:
        from elspeth.contracts import Determinism
        from elspeth.plugins.base import BaseSource

        # Direct attribute access - will fail with AttributeError if missing
        assert BaseSource.determinism == Determinism.IO_READ
        assert BaseSource.plugin_version == "0.0.0"

    def test_subclass_can_override_metadata(self) -> None:
        from collections.abc import Iterator

        from elspeth.contracts import Determinism, PluginSchema
        from elspeth.plugins.base import BaseSource
        from elspeth.plugins.context import PluginContext

        class OutputSchema(PluginSchema):
            value: int

        class CustomSource(BaseSource):
            name = "custom"
            output_schema = OutputSchema
            determinism = Determinism.DETERMINISTIC
            plugin_version = "2.0.0"

            def load(self, ctx: PluginContext) -> Iterator[dict[str, Any]]:
                yield {"value": 1}

            def close(self) -> None:
                pass

        source = CustomSource({})
        assert source.determinism == Determinism.DETERMINISTIC
        assert source.plugin_version == "2.0.0"
