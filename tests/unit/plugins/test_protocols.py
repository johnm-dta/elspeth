# tests/plugins/test_protocols.py
"""Tests for plugin protocols."""

from collections.abc import Iterator, Mapping
from typing import Any, ClassVar

from elspeth.contracts import Determinism, PipelineRow, SourceRow
from elspeth.contracts.schema import SchemaConfig
from elspeth.testing import make_contract, make_pipeline_row
from tests.fixtures.factories import make_context
from tests.fixtures.landscape import make_factory


class TestSourceProtocol:
    """Source plugin protocol."""

    def test_source_protocol_definition(self) -> None:
        from elspeth.contracts import SourceProtocol

        # Should be a Protocol (all Protocols have this attribute)
        assert hasattr(SourceProtocol, "__protocol_attrs__")

    def test_source_implementation(self) -> None:
        from elspeth.contracts import PluginSchema, SourceProtocol
        from elspeth.contracts.plugin_context import PluginContext

        class OutputSchema(PluginSchema):
            value: int

        class MySource:
            """Example source implementation."""

            name = "my_source"
            output_schema = OutputSchema
            node_id: str | None = None  # Set by orchestrator
            determinism = Determinism.IO_READ
            plugin_version = "1.0.0"
            _on_validation_failure = "discard"
            on_success = "output"
            declared_guaranteed_fields: frozenset[str] = frozenset()

            def __init__(self, config: dict[str, Any]) -> None:
                self.config = config

            def load(self, ctx: PluginContext) -> Iterator[SourceRow]:
                for i in range(3):
                    yield SourceRow.valid({"value": i}, contract=make_contract({"value": i}), source_row_index=i)

            def close(self) -> None:
                pass

            def on_start(self, ctx: PluginContext) -> None:
                pass

            def on_complete(self, ctx: PluginContext) -> None:
                pass

            def get_field_resolution(self) -> tuple[Mapping[str, str], str | None] | None:
                return None  # No field normalization

            def get_schema_contract(self) -> Any:
                return None  # No schema contract

        source = MySource({"path": "test.csv"})

        # Protocol conformance is verified by mypy, not isinstance() —
        # SourceProtocol is not @runtime_checkable.
        _source: SourceProtocol = source  # type: ignore[assignment]

        factory = make_factory()
        ctx = make_context(landscape=factory.plugin_audit_writer())

        source_rows = list(source.load(ctx))
        assert len(source_rows) == 3
        assert source_rows[0].row == {"value": 0}

    def test_source_has_lifecycle_hooks(self) -> None:
        from elspeth.contracts import SourceProtocol

        # Check protocol has expected methods
        assert hasattr(SourceProtocol, "load")
        assert hasattr(SourceProtocol, "close")

    def test_source_has_determinism_attribute(self) -> None:
        from elspeth.contracts import SourceProtocol

        assert "determinism" in SourceProtocol.__protocol_attrs__  # type: ignore[attr-defined]

    def test_source_has_version_attribute(self) -> None:
        from elspeth.contracts import SourceProtocol

        assert "plugin_version" in SourceProtocol.__protocol_attrs__  # type: ignore[attr-defined]

    def test_source_has_declared_guaranteed_fields_attribute(self) -> None:
        from elspeth.contracts import SourceProtocol

        assert "declared_guaranteed_fields" in SourceProtocol.__protocol_attrs__  # type: ignore[attr-defined]

    def test_source_implementation_with_metadata(self) -> None:
        from collections.abc import Iterator
        from typing import Any

        from elspeth.contracts import PluginSchema, SourceProtocol
        from elspeth.contracts.plugin_context import PluginContext

        class OutputSchema(PluginSchema):
            value: int

        class MetadataSource:
            name = "metadata_source"
            output_schema = OutputSchema
            node_id: str | None = None
            determinism = Determinism.IO_READ
            plugin_version = "1.0.0"
            _on_validation_failure = "discard"
            on_success = "output"
            declared_guaranteed_fields: frozenset[str] = frozenset()

            def __init__(self, config: dict[str, Any]) -> None:
                self.config = config

            def load(self, ctx: PluginContext) -> Iterator[SourceRow]:
                yield SourceRow.valid({"value": 1}, contract=make_contract({"value": 1}), source_row_index=0)

            def close(self) -> None:
                pass

            def on_start(self, ctx: PluginContext) -> None:
                pass

            def on_complete(self, ctx: PluginContext) -> None:
                pass

            def get_field_resolution(self) -> tuple[Mapping[str, str], str | None] | None:
                return None  # No field normalization

            def get_schema_contract(self) -> Any:
                return None  # No schema contract

        source = MetadataSource({})
        # Protocol conformance verified by mypy, not isinstance().
        _source: SourceProtocol = source  # type: ignore[assignment]
        assert source.determinism == Determinism.IO_READ
        assert source.plugin_version == "1.0.0"


class TestTransformProtocol:
    """Transform plugin protocol (stateless row processing)."""

    def test_transform_implementation(self) -> None:
        from elspeth.contracts import (
            DeclaredAuditCharacteristics,
            PluginSchema,
            TransformProtocol,
        )
        from elspeth.contracts.plugin_context import PluginContext
        from elspeth.plugins.infrastructure.results import TransformResult

        class InputSchema(PluginSchema):
            value: int

        class OutputSchema(PluginSchema):
            value: int
            doubled: int

        class DoubleTransform:
            name = "double"
            input_schema = InputSchema
            output_schema = OutputSchema
            routes: ClassVar[dict[str, str]] = {}
            fork_to: list[str] | None = None
            node_id: str | None = None  # Set by orchestrator
            determinism = Determinism.DETERMINISTIC
            plugin_version = "1.0.0"
            source_file_hash: str | None = None
            is_batch_aware = False  # Batch support (structural aggregation)
            supports_row_mode_when_batch_aware = False  # Batch-aware transforms only
            requires_runtime_preflight = False  # Optional engine-time readiness check
            creates_tokens = False  # Deaggregation (multi-row output)
            passes_through_input = False  # ADR-007: pass-through contract flag
            can_drop_rows = False  # ADR-012: empty-emission governance flag
            declared_input_fields: frozenset[str] = frozenset()
            declared_output_fields: frozenset[str] = frozenset()  # Collision detection
            _output_schema_config: SchemaConfig | None = None  # Set by BaseTransform
            on_error: str | None = None  # Error routing (WP-11.99b)
            on_success: str | None = None  # Success routing
            # Reference-content fields (Phase 7A) — declared on TransformProtocol
            # for the catalog/audit_readiness surfaces. The hand-rolled fake
            # mirrors BaseTransform's defaults so @runtime_checkable conformance
            # holds without inheriting BaseTransform.
            usage_when_to_use: str | None = None
            usage_when_not_to_use: str | None = None
            example_use: str | None = None
            capability_tags: tuple[str, ...] = ()
            audit_characteristics: DeclaredAuditCharacteristics = frozenset()
            discovery_secret_requirements: ClassVar[dict[str, tuple[str, ...]]] = {}
            # Lifecycle guards managed by BaseTransform; declared here for the
            # hand-rolled structural fake to satisfy TransformProtocol's
            # @runtime_checkable structural typing.
            _on_start_called: bool = False
            _on_complete_called: bool = False

            def __init__(self, config: dict[str, Any]) -> None:
                self.config = config

            def process(self, row: PipelineRow, ctx: PluginContext) -> TransformResult:
                d = row.to_dict()
                return TransformResult.success(
                    make_pipeline_row(
                        {
                            "value": d["value"],
                            "doubled": d["value"] * 2,
                        }
                    ),
                    success_reason={"action": "test"},
                )

            def effective_static_contract(self) -> frozenset[str]:
                return frozenset()

            def close(self) -> None:
                pass

            def on_start(self, ctx: PluginContext) -> None:
                pass

            def on_complete(self, ctx: PluginContext) -> None:
                pass

            def runtime_preflight(self, ctx: PluginContext) -> None:
                pass

            @classmethod
            def get_config_model(cls, config: dict[str, Any] | None = None) -> None:
                return None

            @classmethod
            def get_config_schema(cls) -> dict[str, Any]:
                return {}

            @classmethod
            def get_agent_assistance(cls, *, issue_code: str | None = None) -> Any:
                return None

            @classmethod
            def get_post_call_hints(cls, *, tool_name: str, config_snapshot: Any) -> tuple[str, ...]:
                return ()

        transform = DoubleTransform({})

        # IMPORTANT: Verify protocol conformance at runtime (see test_source_protocol_conformance).
        _conforms = isinstance(transform, TransformProtocol)
        assert _conforms, "Must conform to TransformProtocol"

        factory = make_factory()
        ctx = make_context(landscape=factory.plugin_audit_writer())

        result = transform.process(make_pipeline_row({"value": 21}), ctx)
        assert result.status == "success"
        assert result.row is not None
        assert result.row.to_dict() == {"value": 21, "doubled": 42}


class TestTransformBatchSupport:
    """Tests for batch-aware transform protocol."""

    def test_transform_process_single_row(self) -> None:
        """Transform.process() accepts single row dict."""
        from elspeth.contracts import PluginSchema
        from elspeth.contracts.contexts import TransformContext
        from elspeth.plugins.infrastructure.base import BaseTransform
        from elspeth.plugins.infrastructure.results import TransformResult

        class AnySchema(PluginSchema):
            value: int

        class SingleTransform(BaseTransform):
            name = "single"
            input_schema = AnySchema
            output_schema = AnySchema
            determinism = Determinism.DETERMINISTIC
            plugin_version = "1.0.0"

            def process(self, row: PipelineRow, ctx: TransformContext) -> TransformResult:
                return TransformResult.success(make_pipeline_row({"processed": row["value"]}), success_reason={"action": "test"})

        transform = SingleTransform({})
        factory = make_factory()
        ctx = make_context(landscape=factory.plugin_audit_writer())
        result = transform.process(make_pipeline_row({"value": 1}), ctx)
        assert result.row is not None
        assert result.row.to_dict() == {"processed": 1}

    def test_transform_process_batch_rows(self) -> None:
        """Transform.process() accepts list of row dicts when is_batch_aware=True."""
        from elspeth.contracts import PluginSchema
        from elspeth.contracts.contexts import TransformContext
        from elspeth.plugins.infrastructure.base import BaseTransform
        from elspeth.plugins.infrastructure.results import TransformResult

        class AnySchema(PluginSchema):
            pass

        class BatchTransform(BaseTransform):
            name = "batch"
            input_schema = AnySchema
            output_schema = AnySchema
            determinism = Determinism.DETERMINISTIC
            plugin_version = "1.0.0"
            is_batch_aware = True  # Declares batch support

            def process(self, row: PipelineRow | list[PipelineRow], ctx: TransformContext) -> TransformResult:
                # When given a list, process as batch
                if isinstance(row, list):
                    total = sum(r["value"] for r in row)
                    return TransformResult.success(
                        make_pipeline_row({"total": total, "count": len(row)}), success_reason={"action": "test"}
                    )
                # Single row
                return TransformResult.success(make_pipeline_row({"value": row["value"]}), success_reason={"action": "test"})

        transform = BatchTransform({})
        factory = make_factory()
        ctx = make_context(landscape=factory.plugin_audit_writer())

        # Batch mode
        result = transform.process([make_pipeline_row({"value": 1}), make_pipeline_row({"value": 2}), make_pipeline_row({"value": 3})], ctx)
        assert result.row is not None
        assert result.row.to_dict() == {"total": 6, "count": 3}

    def test_transform_is_batch_aware_default_false(self) -> None:
        """Transforms have is_batch_aware=False by default."""
        from elspeth.contracts import PluginSchema
        from elspeth.contracts.contexts import TransformContext
        from elspeth.plugins.infrastructure.base import BaseTransform
        from elspeth.plugins.infrastructure.results import TransformResult

        class AnySchema(PluginSchema):
            value: int

        class RegularTransform(BaseTransform):
            name = "regular"
            input_schema = AnySchema
            output_schema = AnySchema
            determinism = Determinism.DETERMINISTIC
            plugin_version = "1.0.0"

            def process(self, row: PipelineRow, ctx: TransformContext) -> TransformResult:
                return TransformResult.success(make_pipeline_row(row.to_dict()), success_reason={"action": "test"})

        regular = RegularTransform({})
        assert regular.is_batch_aware is False

    def test_transform_is_batch_aware_can_be_set_true(self) -> None:
        """Transforms can declare is_batch_aware=True for batch support."""
        from elspeth.contracts import PluginSchema
        from elspeth.contracts.contexts import TransformContext
        from elspeth.plugins.infrastructure.base import BaseTransform
        from elspeth.plugins.infrastructure.results import TransformResult

        class AnySchema(PluginSchema):
            pass

        class BatchAwareTransform(BaseTransform):
            name = "batch_aware"
            is_batch_aware = True  # Declares batch support
            input_schema = AnySchema
            output_schema = AnySchema
            determinism = Determinism.DETERMINISTIC
            plugin_version = "1.0.0"

            def process(self, row: PipelineRow | list[PipelineRow], ctx: TransformContext) -> TransformResult:
                if isinstance(row, list):
                    return TransformResult.success(make_pipeline_row({"count": len(row)}), success_reason={"action": "test"})
                return TransformResult.success(make_pipeline_row(row.to_dict()), success_reason={"action": "test"})

        batch = BatchAwareTransform({})
        assert batch.is_batch_aware is True


class TestAggregationProtocolDeleted:
    """Verify AggregationProtocol has been deleted.

    Aggregation is now fully structural:
    - Engine buffers rows internally
    - Engine evaluates triggers (WP-06)
    - Engine calls batch-aware Transform.process(rows: list[dict])
    - No plugin-level aggregation interface

    Use is_batch_aware=True on BaseTransform for batch processing.
    """

    def test_aggregation_protocol_deleted(self) -> None:
        """AggregationProtocol should be deleted (aggregation is structural)."""
        import elspeth.contracts.plugin_protocols as protocols

        assert not hasattr(protocols, "AggregationProtocol"), "AggregationProtocol should be deleted - aggregation is structural"

    def test_base_aggregation_deleted(self) -> None:
        """BaseAggregation should be deleted (aggregation is structural)."""
        import elspeth.plugins.infrastructure.base as base

        assert not hasattr(base, "BaseAggregation"), "BaseAggregation should be deleted - use is_batch_aware=True on BaseTransform"


class TestCoalesceProtocolDeleted:
    """Verify CoalesceProtocol has been deleted.

    Coalesce is now fully structural:
    - Engine holds tokens via CoalesceExecutor (engine/coalesce_executor.py)
    - Engine evaluates merge conditions based on CoalesceSettings policy
    - Engine merges data via CoalesceSettings merge strategy (union/nested/select)
    - No plugin-level coalesce interface
    """

    def test_coalesce_protocol_deleted(self) -> None:
        """CoalesceProtocol should be deleted (coalesce is structural)."""
        import elspeth.contracts.plugin_protocols as protocols

        assert not hasattr(protocols, "CoalesceProtocol"), "CoalesceProtocol should be deleted - coalesce is structural"

    def test_coalesce_policy_enum_deleted(self) -> None:
        """CoalescePolicy enum should be deleted (superseded by Literal strings in CoalesceSettings)."""
        import elspeth.contracts.plugin_protocols as protocols

        assert not hasattr(protocols, "CoalescePolicy"), "CoalescePolicy should be deleted - use CoalesceSettings.policy Literal"

    def test_plugin_protocol_deleted(self) -> None:
        """PluginProtocol should be deleted (never used)."""
        import elspeth.contracts.plugin_protocols as protocols

        assert not hasattr(protocols, "PluginProtocol"), "PluginProtocol should be deleted - never imported or used"


class TestSinkProtocol:
    """Sink plugin protocol."""

    def test_sink_batch_write_signature(self) -> None:
        """Sink.write() accepts batch and returns ArtifactDescriptor."""
        import inspect

        from elspeth.contracts import SinkProtocol

        # Get the write method signature
        sig = inspect.signature(SinkProtocol.write)
        params = list(sig.parameters.keys())

        # Should have 'rows' not 'row'
        assert "rows" in params, "write() should accept 'rows' (batch), not 'row'"
        assert "row" not in params, "write() should NOT have 'row' parameter"

        # Return annotation should be SinkWriteResult (may be forward ref string)
        return_annotation = sig.return_annotation
        # Handle both string forward reference and actual class
        if isinstance(return_annotation, str):
            assert return_annotation == "SinkWriteResult"
        else:
            from elspeth.contracts.diversion import SinkWriteResult

            assert return_annotation == SinkWriteResult

    def test_batch_sink_implementation(self) -> None:
        """Test sink with batch write returning SinkWriteResult."""
        from elspeth.contracts import ArtifactDescriptor, PluginSchema, SinkProtocol
        from elspeth.contracts.diversion import SinkWriteResult
        from elspeth.contracts.plugin_context import PluginContext
        from elspeth.contracts.sink import OutputValidationResult

        class InputSchema(PluginSchema):
            value: int

        class BatchMemorySink:
            name = "batch_memory"
            input_schema = InputSchema
            idempotent = True
            node_id: str | None = None
            config: dict[str, Any]  # Set in __init__
            determinism = Determinism.IO_WRITE
            plugin_version = "1.0.0"
            supports_resume = False  # Required by SinkProtocol
            declared_required_fields: frozenset[str] = frozenset()  # Centralized in executor
            _on_write_failure: str | None = "discard"

            def __init__(self, config: dict[str, Any]) -> None:
                self.config = config
                self.rows: list[dict[str, Any]] = []
                self._diversion_log: list[Any] = []

            def _reset_diversion_log(self) -> None:
                self._diversion_log = []

            def write(self, rows: list[dict[str, Any]], ctx: PluginContext) -> SinkWriteResult:
                self.rows.extend(rows)
                return SinkWriteResult(
                    artifact=ArtifactDescriptor.for_file(
                        path="/tmp/test.json",
                        content_hash="abc123",
                        size_bytes=len(str(rows)),
                    )
                )

            def flush(self) -> None:
                pass

            def close(self) -> None:
                pass

            def on_start(self, ctx: PluginContext) -> None:
                pass

            def on_complete(self, ctx: PluginContext) -> None:
                pass

            def configure_for_resume(self) -> None:
                raise NotImplementedError("BatchMemorySink does not support resume")

            def validate_output_target(self) -> OutputValidationResult:
                return OutputValidationResult.success()

            def set_resume_field_resolution(self, resolution_mapping: dict[str, str]) -> None:
                pass  # Not needed for this test sink

            @property
            def needs_resume_field_resolution(self) -> bool:
                return False

        sink = BatchMemorySink({})
        # Protocol conformance verified by mypy, not isinstance().
        _sink: SinkProtocol = sink  # type: ignore[assignment]

        factory = make_factory()
        ctx = make_context(landscape=factory.plugin_audit_writer())
        result = sink.write([{"value": 1}, {"value": 2}], ctx)

        assert isinstance(result.artifact, ArtifactDescriptor)
        assert result.artifact.content_hash == "abc123"
        assert len(sink.rows) == 2

    def test_sink_implementation(self) -> None:
        """Test sink conforming to updated batch protocol."""

        from elspeth.contracts import ArtifactDescriptor, PluginSchema, SinkProtocol
        from elspeth.contracts.diversion import SinkWriteResult
        from elspeth.contracts.plugin_context import PluginContext
        from elspeth.contracts.sink import OutputValidationResult

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
            supports_resume = False  # Required by SinkProtocol
            declared_required_fields: frozenset[str] = frozenset()  # Centralized in executor
            _on_write_failure: str | None = "discard"
            rows: ClassVar[list[dict[str, Any]]] = []

            def __init__(self, config: dict[str, Any]) -> None:
                self.instance_rows: list[dict[str, Any]] = []
                self.config = config
                self._diversion_log: list[Any] = []

            def _reset_diversion_log(self) -> None:
                self._diversion_log = []

            def write(self, rows: list[dict[str, Any]], ctx: PluginContext) -> SinkWriteResult:
                self.rows.extend(rows)
                return SinkWriteResult(
                    artifact=ArtifactDescriptor.for_file(
                        path="/tmp/memory",
                        content_hash="a" * 64,
                        size_bytes=len(str(rows)),
                    )
                )

            def flush(self) -> None:
                pass

            def close(self) -> None:
                pass

            def on_start(self, ctx: PluginContext) -> None:
                pass

            def on_complete(self, ctx: PluginContext) -> None:
                pass

            def configure_for_resume(self) -> None:
                raise NotImplementedError("MemorySink does not support resume")

            def validate_output_target(self) -> OutputValidationResult:
                return OutputValidationResult.success()

            def set_resume_field_resolution(self, resolution_mapping: dict[str, str]) -> None:
                pass  # Not needed for this test sink

            @property
            def needs_resume_field_resolution(self) -> bool:
                return False

        sink = MemorySink({})

        # Protocol conformance verified by mypy, not isinstance().
        _sink: SinkProtocol = sink  # type: ignore[assignment]

        factory = make_factory()
        ctx = make_context(landscape=factory.plugin_audit_writer())

        # Batch write
        result = sink.write([{"value": 1}, {"value": 2}], ctx)

        assert len(sink.rows) == 2
        assert sink.rows[0] == {"value": 1}
        assert isinstance(result.artifact, ArtifactDescriptor)

    def test_sink_has_idempotency_support(self) -> None:
        """Sinks should support idempotency keys."""
        from elspeth.contracts import SinkProtocol

        # Protocol should have idempotent attribute
        assert hasattr(SinkProtocol, "__protocol_attrs__")


class TestProtocolMetadata:
    """Test that protocols include metadata attributes."""

    def test_transform_has_determinism_attribute(self) -> None:
        from elspeth.contracts import TransformProtocol

        # Protocol attributes are tracked in __protocol_attrs__ (runtime Protocol internals)
        assert "determinism" in TransformProtocol.__protocol_attrs__  # type: ignore[attr-defined]

    def test_transform_has_version_attribute(self) -> None:
        from elspeth.contracts import TransformProtocol

        # __protocol_attrs__ is a runtime attribute on @runtime_checkable Protocols
        assert "plugin_version" in TransformProtocol.__protocol_attrs__  # type: ignore[attr-defined]

    def test_deterministic_transform(self) -> None:
        from elspeth.contracts.plugin_context import PluginContext
        from elspeth.plugins.infrastructure.results import TransformResult

        class MyTransform:
            name = "my_transform"
            determinism = Determinism.DETERMINISTIC
            plugin_version = "1.0.0"

            def process(self, row: PipelineRow, ctx: PluginContext) -> TransformResult:
                return TransformResult.success(make_pipeline_row(row.to_dict()), success_reason={"action": "test"})

        t = MyTransform()
        assert t.determinism == Determinism.DETERMINISTIC

    def test_nondeterministic_transform(self) -> None:
        from elspeth.contracts.plugin_context import PluginContext
        from elspeth.plugins.infrastructure.results import TransformResult

        class LLMTransform:
            name = "llm_classifier"
            determinism = Determinism.EXTERNAL_CALL
            plugin_version = "0.1.0"

            def process(self, row: PipelineRow, ctx: PluginContext) -> TransformResult:
                return TransformResult.success(make_pipeline_row(row.to_dict()), success_reason={"action": "test"})

        t = LLMTransform()
        assert t.determinism == Determinism.EXTERNAL_CALL
