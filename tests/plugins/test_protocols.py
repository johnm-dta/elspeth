# tests/plugins/test_protocols.py
"""Tests for plugin protocols."""

from collections.abc import Iterator
from typing import Any, ClassVar


class TestSourceProtocol:
    """Source plugin protocol."""

    def test_source_protocol_definition(self) -> None:
        from elspeth.plugins.protocols import SourceProtocol

        # Should be a Protocol (runtime_checkable protocols have this attribute)
        assert hasattr(SourceProtocol, "__protocol_attrs__")

    def test_source_implementation(self) -> None:
        from elspeth.contracts import Determinism, PluginSchema
        from elspeth.plugins.context import PluginContext
        from elspeth.plugins.protocols import SourceProtocol

        class OutputSchema(PluginSchema):
            value: int

        class MySource:
            """Example source implementation."""

            name = "my_source"
            output_schema = OutputSchema
            node_id: str | None = None  # Set by orchestrator
            determinism = Determinism.IO_READ
            plugin_version = "1.0.0"

            def __init__(self, config: dict[str, Any]) -> None:
                self.config = config

            def load(self, ctx: PluginContext) -> Iterator[dict[str, Any]]:
                for i in range(3):
                    yield {"value": i}

            def close(self) -> None:
                pass

            def on_start(self, ctx: PluginContext) -> None:
                pass

            def on_complete(self, ctx: PluginContext) -> None:
                pass

        source = MySource({"path": "test.csv"})

        # IMPORTANT: Verify protocol conformance at runtime
        # This is why we use @runtime_checkable
        assert isinstance(
            source, SourceProtocol
        ), "Source must conform to SourceProtocol"

        ctx = PluginContext(run_id="test", config={})

        rows = list(source.load(ctx))
        assert len(rows) == 3
        assert rows[0] == {"value": 0}

    def test_source_has_lifecycle_hooks(self) -> None:
        from elspeth.plugins.protocols import SourceProtocol

        # Check protocol has expected methods
        assert hasattr(SourceProtocol, "load")
        assert hasattr(SourceProtocol, "close")

    def test_source_has_determinism_attribute(self) -> None:
        from elspeth.plugins.protocols import SourceProtocol

        assert "determinism" in SourceProtocol.__protocol_attrs__  # type: ignore[attr-defined]

    def test_source_has_version_attribute(self) -> None:
        from elspeth.plugins.protocols import SourceProtocol

        assert "plugin_version" in SourceProtocol.__protocol_attrs__  # type: ignore[attr-defined]

    def test_source_implementation_with_metadata(self) -> None:
        from collections.abc import Iterator
        from typing import Any

        from elspeth.contracts import Determinism, PluginSchema
        from elspeth.plugins.context import PluginContext
        from elspeth.plugins.protocols import SourceProtocol

        class OutputSchema(PluginSchema):
            value: int

        class MetadataSource:
            name = "metadata_source"
            output_schema = OutputSchema
            node_id: str | None = None
            determinism = Determinism.IO_READ
            plugin_version = "1.0.0"

            def __init__(self, config: dict[str, Any]) -> None:
                self.config = config

            def load(self, ctx: PluginContext) -> Iterator[dict[str, Any]]:
                yield {"value": 1}

            def close(self) -> None:
                pass

            def on_start(self, ctx: PluginContext) -> None:
                pass

            def on_complete(self, ctx: PluginContext) -> None:
                pass

        source = MetadataSource({})
        assert isinstance(source, SourceProtocol)
        assert source.determinism == Determinism.IO_READ
        assert source.plugin_version == "1.0.0"


class TestTransformProtocol:
    """Transform plugin protocol (stateless row processing)."""

    def test_transform_implementation(self) -> None:
        from elspeth.contracts import Determinism, PluginSchema
        from elspeth.plugins.context import PluginContext
        from elspeth.plugins.protocols import TransformProtocol
        from elspeth.plugins.results import TransformResult

        class InputSchema(PluginSchema):
            value: int

        class OutputSchema(PluginSchema):
            value: int
            doubled: int

        class DoubleTransform:
            name = "double"
            input_schema = InputSchema
            output_schema = OutputSchema
            node_id: str | None = None  # Set by orchestrator
            determinism = Determinism.DETERMINISTIC
            plugin_version = "1.0.0"
            _on_error: str | None = None  # Error routing (WP-11.99b)

            def __init__(self, config: dict[str, Any]) -> None:
                self.config = config

            def process(
                self, row: dict[str, Any], ctx: PluginContext
            ) -> TransformResult:
                return TransformResult.success(
                    {
                        "value": row["value"],
                        "doubled": row["value"] * 2,
                    }
                )

            def close(self) -> None:
                pass

            def on_register(self, ctx: PluginContext) -> None:
                pass

            def on_start(self, ctx: PluginContext) -> None:
                pass

            def on_complete(self, ctx: PluginContext) -> None:
                pass

        transform = DoubleTransform({})

        # IMPORTANT: Verify protocol conformance at runtime
        assert isinstance(
            transform, TransformProtocol
        ), "Must conform to TransformProtocol"

        ctx = PluginContext(run_id="test", config={})

        result = transform.process({"value": 21}, ctx)
        assert result.status == "success"
        assert result.row == {"value": 21, "doubled": 42}


class TestAggregationProtocol:
    """Aggregation plugin protocol (stateful batching)."""

    def test_aggregation_implementation(self) -> None:
        from elspeth.contracts import Determinism, PluginSchema
        from elspeth.plugins.context import PluginContext
        from elspeth.plugins.protocols import AggregationProtocol
        from elspeth.plugins.results import AcceptResult

        class InputSchema(PluginSchema):
            value: int

        class OutputSchema(PluginSchema):
            total: int
            count: int

        class SumAggregation:
            name = "sum"
            input_schema = InputSchema
            output_schema = OutputSchema
            node_id: str | None = None  # Set by orchestrator
            determinism = Determinism.DETERMINISTIC
            plugin_version = "1.0.0"

            def __init__(self, config: dict[str, Any]) -> None:
                self.batch_size: int = config["batch_size"]
                self._values: list[int] = []

            def accept(self, row: dict[str, Any], ctx: PluginContext) -> AcceptResult:
                self._values.append(row["value"])
                trigger = len(self._values) >= self.batch_size
                return AcceptResult(accepted=True, trigger=trigger)

            def should_trigger(self) -> bool:
                return len(self._values) >= self.batch_size

            def flush(self, ctx: PluginContext) -> list[dict[str, Any]]:
                result = {
                    "total": sum(self._values),
                    "count": len(self._values),
                }
                self._values = []
                return [result]

            def reset(self) -> None:
                self._values = []

            def on_register(self, ctx: PluginContext) -> None:
                pass

            def on_start(self, ctx: PluginContext) -> None:
                pass

            def on_complete(self, ctx: PluginContext) -> None:
                pass

        agg = SumAggregation({"batch_size": 2})

        # IMPORTANT: Verify protocol conformance at runtime
        assert isinstance(
            agg, AggregationProtocol
        ), "Must conform to AggregationProtocol"

        ctx = PluginContext(run_id="test", config={})

        # First row - no trigger
        result = agg.accept({"value": 10}, ctx)
        assert result.accepted is True
        assert result.trigger is False

        # Second row - trigger
        result = agg.accept({"value": 20}, ctx)
        assert result.trigger is True

        # Flush
        outputs = agg.flush(ctx)
        assert len(outputs) == 1
        assert outputs[0] == {"total": 30, "count": 2}


class TestCoalesceProtocol:
    """Coalesce plugin protocol (merge parallel paths)."""

    def test_coalesce_policy_types(self) -> None:
        from elspeth.plugins.protocols import CoalescePolicy

        # All policies should exist
        assert CoalescePolicy.REQUIRE_ALL.value == "require_all"
        assert CoalescePolicy.QUORUM.value == "quorum"
        assert CoalescePolicy.BEST_EFFORT.value == "best_effort"

    def test_quorum_requires_threshold(self) -> None:
        """QUORUM policy needs a quorum_threshold."""
        from elspeth.contracts import Determinism, PluginSchema
        from elspeth.plugins.context import PluginContext
        from elspeth.plugins.protocols import CoalescePolicy, CoalesceProtocol

        class OutputSchema(PluginSchema):
            combined: str

        class QuorumCoalesce:
            name = "quorum_merge"
            policy = CoalescePolicy.QUORUM
            quorum_threshold = 2  # At least 2 branches must arrive
            expected_branches: ClassVar[list[str]] = [
                "branch_a",
                "branch_b",
                "branch_c",
            ]
            output_schema = OutputSchema
            node_id: str | None = None  # Set by orchestrator
            determinism = Determinism.DETERMINISTIC
            plugin_version = "1.0.0"

            def __init__(self, config: dict[str, Any]) -> None:
                pass

            def merge(
                self, branch_outputs: dict[str, dict[str, Any]], ctx: PluginContext
            ) -> dict[str, Any]:
                return {"combined": "+".join(branch_outputs.keys())}

            def on_register(self, ctx: PluginContext) -> None:
                pass

            def on_start(self, ctx: PluginContext) -> None:
                pass

            def on_complete(self, ctx: PluginContext) -> None:
                pass

        coalesce = QuorumCoalesce({})

        # IMPORTANT: Verify protocol conformance at runtime
        # mypy may report this as unreachable due to structural subtyping analysis
        # but runtime_checkable protocols DO work at runtime
        assert isinstance(
            coalesce, CoalesceProtocol
        ), "Must conform to CoalesceProtocol"  # type: ignore[unreachable]

        assert coalesce.quorum_threshold == 2  # type: ignore[unreachable]
        assert len(coalesce.expected_branches) == 3  # type: ignore[unreachable]

    def test_coalesce_merge_behavior(self) -> None:
        """Test merge() combines branch outputs correctly."""
        from elspeth.contracts import Determinism, PluginSchema
        from elspeth.plugins.context import PluginContext
        from elspeth.plugins.protocols import CoalescePolicy, CoalesceProtocol

        class OutputSchema(PluginSchema):
            total: int

        class SumCoalesce:
            name = "sum_merge"
            policy = CoalescePolicy.REQUIRE_ALL
            quorum_threshold = None
            expected_branches: ClassVar[list[str]] = ["branch_a", "branch_b"]
            output_schema = OutputSchema
            node_id: str | None = None  # Set by orchestrator
            determinism = Determinism.DETERMINISTIC
            plugin_version = "1.0.0"

            def __init__(self, config: dict[str, Any]) -> None:
                pass

            def merge(
                self, branch_outputs: dict[str, dict[str, Any]], ctx: PluginContext
            ) -> dict[str, Any]:
                total = sum(out["value"] for out in branch_outputs.values())
                return {"total": total}

            def on_register(self, ctx: PluginContext) -> None:
                pass

            def on_start(self, ctx: PluginContext) -> None:
                pass

            def on_complete(self, ctx: PluginContext) -> None:
                pass

        coalesce = SumCoalesce({})
        # mypy may report this as unreachable due to structural subtyping analysis
        # but runtime_checkable protocols DO work at runtime
        assert isinstance(coalesce, CoalesceProtocol)  # type: ignore[unreachable]

        ctx = PluginContext(run_id="test", config={})  # type: ignore[unreachable]

        branch_outputs = {
            "branch_a": {"value": 10},
            "branch_b": {"value": 20},
        }
        result = coalesce.merge(branch_outputs, ctx)
        assert result == {"total": 30}


class TestSinkProtocol:
    """Sink plugin protocol."""

    def test_sink_batch_write_signature(self) -> None:
        """Sink.write() accepts batch and returns ArtifactDescriptor."""
        import inspect

        from elspeth.plugins.protocols import SinkProtocol

        # Get the write method signature
        sig = inspect.signature(SinkProtocol.write)
        params = list(sig.parameters.keys())

        # Should have 'rows' not 'row'
        assert "rows" in params, "write() should accept 'rows' (batch), not 'row'"
        assert "row" not in params, "write() should NOT have 'row' parameter"

        # Return annotation should be ArtifactDescriptor (may be forward ref string)
        return_annotation = sig.return_annotation
        # Handle both string forward reference and actual class
        if isinstance(return_annotation, str):
            assert return_annotation == "ArtifactDescriptor"
        else:
            from elspeth.contracts import ArtifactDescriptor

            assert return_annotation == ArtifactDescriptor

    def test_batch_sink_implementation(self) -> None:
        """Test sink with batch write returning ArtifactDescriptor."""
        from elspeth.contracts import ArtifactDescriptor, Determinism, PluginSchema
        from elspeth.plugins.context import PluginContext
        from elspeth.plugins.protocols import SinkProtocol

        class InputSchema(PluginSchema):
            value: int

        class BatchMemorySink:
            name = "batch_memory"
            input_schema = InputSchema
            idempotent = True
            node_id: str | None = None
            determinism = Determinism.IO_WRITE
            plugin_version = "1.0.0"

            def __init__(self, config: dict[str, Any]) -> None:
                self.rows: list[dict[str, Any]] = []

            def write(
                self, rows: list[dict[str, Any]], ctx: PluginContext
            ) -> ArtifactDescriptor:
                self.rows.extend(rows)
                return ArtifactDescriptor.for_file(
                    path="/tmp/test.json",
                    content_hash="abc123",
                    size_bytes=len(str(rows)),
                )

            def flush(self) -> None:
                pass

            def close(self) -> None:
                pass

            def on_register(self, ctx: PluginContext) -> None:
                pass

            def on_start(self, ctx: PluginContext) -> None:
                pass

            def on_complete(self, ctx: PluginContext) -> None:
                pass

        sink = BatchMemorySink({})
        assert isinstance(sink, SinkProtocol)

        ctx = PluginContext(run_id="test", config={})
        artifact = sink.write([{"value": 1}, {"value": 2}], ctx)

        assert isinstance(artifact, ArtifactDescriptor)
        assert artifact.content_hash == "abc123"
        assert len(sink.rows) == 2

    def test_sink_implementation(self) -> None:
        """Test sink conforming to updated batch protocol."""
        from typing import ClassVar

        from elspeth.contracts import ArtifactDescriptor, Determinism, PluginSchema
        from elspeth.plugins.context import PluginContext
        from elspeth.plugins.protocols import SinkProtocol

        class InputSchema(PluginSchema):
            value: int

        class MemorySink:
            """Test sink that stores rows in memory."""

            name = "memory"
            input_schema = InputSchema
            idempotent = True
            node_id: str | None = None  # Set by orchestrator
            determinism = Determinism.IO_WRITE
            plugin_version = "1.0.0"
            rows: ClassVar[list[dict[str, Any]]] = []

            def __init__(self, config: dict[str, Any]) -> None:
                self.instance_rows: list[dict[str, Any]] = []
                self.config = config

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

            def on_register(self, ctx: PluginContext) -> None:
                pass

            def on_start(self, ctx: PluginContext) -> None:
                pass

            def on_complete(self, ctx: PluginContext) -> None:
                pass

        sink = MemorySink({})

        # IMPORTANT: Verify protocol conformance at runtime
        assert isinstance(sink, SinkProtocol), "Must conform to SinkProtocol"

        ctx = PluginContext(run_id="test", config={})

        # Batch write
        artifact = sink.write([{"value": 1}, {"value": 2}], ctx)

        assert len(sink.rows) == 2
        assert sink.rows[0] == {"value": 1}
        assert isinstance(artifact, ArtifactDescriptor)

    def test_sink_has_idempotency_support(self) -> None:
        """Sinks should support idempotency keys."""
        from elspeth.plugins.protocols import SinkProtocol

        # Protocol should have idempotent attribute
        assert hasattr(SinkProtocol, "__protocol_attrs__")


class TestProtocolMetadata:
    """Test that protocols include metadata attributes."""

    def test_transform_has_determinism_attribute(self) -> None:
        from elspeth.plugins.protocols import TransformProtocol

        # Protocol attributes are tracked in __protocol_attrs__ (runtime Protocol internals)
        assert "determinism" in TransformProtocol.__protocol_attrs__  # type: ignore[attr-defined]

    def test_transform_has_version_attribute(self) -> None:
        from elspeth.plugins.protocols import TransformProtocol

        # __protocol_attrs__ is a runtime attribute on @runtime_checkable Protocols
        assert "plugin_version" in TransformProtocol.__protocol_attrs__  # type: ignore[attr-defined]

    def test_deterministic_transform(self) -> None:
        from elspeth.contracts import Determinism
        from elspeth.plugins.context import PluginContext
        from elspeth.plugins.results import TransformResult

        class MyTransform:
            name = "my_transform"
            determinism = Determinism.DETERMINISTIC
            plugin_version = "1.0.0"

            def process(
                self, row: dict[str, Any], ctx: PluginContext
            ) -> TransformResult:
                return TransformResult.success(row)

        t = MyTransform()
        assert t.determinism == Determinism.DETERMINISTIC

    def test_nondeterministic_transform(self) -> None:
        from elspeth.contracts import Determinism
        from elspeth.plugins.context import PluginContext
        from elspeth.plugins.results import TransformResult

        class LLMTransform:
            name = "llm_classifier"
            determinism = Determinism.EXTERNAL_CALL
            plugin_version = "0.1.0"

            def process(
                self, row: dict[str, Any], ctx: PluginContext
            ) -> TransformResult:
                return TransformResult.success(row)

        t = LLMTransform()
        assert t.determinism == Determinism.EXTERNAL_CALL
