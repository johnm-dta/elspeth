# tests/engine/test_orchestrator_validation.py
"""Tests for Orchestrator transform error sink validation.

These tests verify that _validate_transform_error_sinks() properly validates
that transform on_error settings reference existing sinks at startup time,
before any rows are processed.
"""

from __future__ import annotations

from collections.abc import Iterator
from typing import TYPE_CHECKING, Any

import pytest

from elspeth.contracts import PluginSchema, SourceRow
from elspeth.plugins.base import BaseTransform
from elspeth.plugins.results import TransformResult
from tests.conftest import (
    _TestSinkBase,
    _TestSourceBase,
    as_sink,
    as_source,
    as_transform,
)
from tests.engine.orchestrator_test_helpers import build_production_graph

if TYPE_CHECKING:
    from elspeth.core.landscape import LandscapeDB


# ============================================================================
# Test Classes
# ============================================================================


class TestTransformErrorSinkValidation:
    """Tests for transform on_error sink validation at startup."""

    def test_invalid_on_error_sink_fails_at_startup(self, landscape_db: LandscapeDB, payload_store) -> None:
        """Transform with on_error pointing to non-existent sink fails before processing."""
        from elspeth.contracts import ArtifactDescriptor
        from elspeth.engine.orchestrator import (
            Orchestrator,
            PipelineConfig,
            RouteValidationError,
        )

        class InputSchema(PluginSchema):
            value: int

        class TrackingSource(_TestSourceBase):
            """Source that tracks whether load() was called."""

            name = "tracking_source"
            output_schema = InputSchema

            def __init__(self, data: list[dict[str, Any]]) -> None:
                self._data = data
                self.load_called = False

            def load(self, ctx: Any) -> Iterator[SourceRow]:
                self.load_called = True
                for row in self._data:
                    yield SourceRow.valid(row)

        class TransformWithInvalidOnError(BaseTransform):
            """Transform configured to route errors to non-existent sink."""

            name = "transform_with_invalid_on_error"
            input_schema = InputSchema
            output_schema = InputSchema
            _on_error = "nonexistent_error_sink"  # Does not exist

            def __init__(self) -> None:
                super().__init__({"schema": {"fields": "dynamic"}})

            def process(self, row: dict[str, Any], ctx: Any) -> TransformResult:
                return TransformResult.success(row, success_reason={"action": "test"})

        class CollectSink(_TestSinkBase):
            name = "collect"

            def __init__(self) -> None:
                self.results: list[dict[str, Any]] = []

            def on_start(self, ctx: Any) -> None:
                pass

            def on_complete(self, ctx: Any) -> None:
                pass

            def write(self, rows: Any, ctx: Any) -> ArtifactDescriptor:
                self.results.extend(rows)
                return ArtifactDescriptor.for_file(path="memory", size_bytes=0, content_hash="")

            def close(self) -> None:
                pass

        source = TrackingSource([{"value": 1}, {"value": 2}])
        transform = TransformWithInvalidOnError()
        sink = CollectSink()

        config = PipelineConfig(
            source=as_source(source),
            transforms=[as_transform(transform)],
            sinks={"default": as_sink(sink)},
        )

        orchestrator = Orchestrator(landscape_db)

        # Must raise RouteValidationError at startup
        with pytest.raises(RouteValidationError):
            orchestrator.run(config, graph=build_production_graph(config), payload_store=payload_store)

        # Verify source.load was NOT called (validation happens before processing)
        assert not source.load_called

    def test_error_message_includes_transform_name_and_sinks(self, landscape_db: LandscapeDB, payload_store) -> None:
        """Error message includes transform name and available sinks."""
        from elspeth.contracts import ArtifactDescriptor
        from elspeth.engine.orchestrator import (
            Orchestrator,
            PipelineConfig,
            RouteValidationError,
        )

        class InputSchema(PluginSchema):
            value: int

        class ListSource(_TestSourceBase):
            name = "list_source"
            output_schema = InputSchema

            def __init__(self, data: list[dict[str, Any]]) -> None:
                self._data = data

            def load(self, ctx: Any) -> Iterator[SourceRow]:
                for row in self._data:
                    yield SourceRow.valid(row)

        class MyBadTransform(BaseTransform):
            """Transform with bad on_error config."""

            name = "my_bad_transform"
            input_schema = InputSchema
            output_schema = InputSchema
            _on_error = "phantom_sink"  # Does not exist

            def __init__(self) -> None:
                super().__init__({"schema": {"fields": "dynamic"}})

            def process(self, row: dict[str, Any], ctx: Any) -> TransformResult:
                return TransformResult.success(row, success_reason={"action": "test"})

        class CollectSink(_TestSinkBase):
            name = "collect"

            def __init__(self) -> None:
                self.results: list[dict[str, Any]] = []

            def on_start(self, ctx: Any) -> None:
                pass

            def on_complete(self, ctx: Any) -> None:
                pass

            def write(self, rows: Any, ctx: Any) -> ArtifactDescriptor:
                self.results.extend(rows)
                return ArtifactDescriptor.for_file(path="memory", size_bytes=0, content_hash="")

            def close(self) -> None:
                pass

        source = ListSource([{"value": 1}])
        transform = MyBadTransform()
        sink = CollectSink()

        config = PipelineConfig(
            source=as_source(source),
            transforms=[as_transform(transform)],
            sinks={
                "default": as_sink(sink),
                "error_archive": as_sink(sink),
            },  # Two sinks available
        )

        orchestrator = Orchestrator(landscape_db)

        with pytest.raises(RouteValidationError) as exc_info:
            orchestrator.run(config, graph=build_production_graph(config), payload_store=payload_store)

        error_msg = str(exc_info.value)
        # Error should mention the transform name
        assert "my_bad_transform" in error_msg
        # Error should mention the invalid sink name
        assert "phantom_sink" in error_msg
        # Error should list available sinks
        assert "default" in error_msg
        assert "error_archive" in error_msg

    def test_on_error_discard_passes_validation(self, landscape_db: LandscapeDB, payload_store) -> None:
        """on_error: 'discard' passes validation (special value, not a sink)."""
        from elspeth.contracts import ArtifactDescriptor
        from elspeth.engine.orchestrator import Orchestrator, PipelineConfig

        class InputSchema(PluginSchema):
            value: int

        class ListSource(_TestSourceBase):
            name = "list_source"
            output_schema = InputSchema

            def __init__(self, data: list[dict[str, Any]]) -> None:
                self._data = data

            def load(self, ctx: Any) -> Iterator[SourceRow]:
                for row in self._data:
                    yield SourceRow.valid(row)

        class DiscardTransform(BaseTransform):
            """Transform that discards errors."""

            name = "discard_transform"
            input_schema = InputSchema
            output_schema = InputSchema
            _on_error = "discard"  # Special value - should pass validation

            def __init__(self) -> None:
                super().__init__({"schema": {"fields": "dynamic"}})

            def process(self, row: dict[str, Any], ctx: Any) -> TransformResult:
                return TransformResult.success(row, success_reason={"action": "test"})

        class CollectSink(_TestSinkBase):
            name = "collect"

            def __init__(self) -> None:
                self.results: list[dict[str, Any]] = []

            def on_start(self, ctx: Any) -> None:
                pass

            def on_complete(self, ctx: Any) -> None:
                pass

            def write(self, rows: Any, ctx: Any) -> ArtifactDescriptor:
                self.results.extend(rows)
                return ArtifactDescriptor.for_file(path="memory", size_bytes=0, content_hash="")

            def close(self) -> None:
                pass

        source = ListSource([{"value": 1}])
        transform = DiscardTransform()
        sink = CollectSink()

        config = PipelineConfig(
            source=as_source(source),
            transforms=[as_transform(transform)],
            sinks={"default": as_sink(sink)},
        )

        orchestrator = Orchestrator(landscape_db)

        # Should NOT raise - "discard" is valid
        result = orchestrator.run(config, graph=build_production_graph(config), payload_store=payload_store)
        assert result.status == "completed"

    def test_on_error_none_passes_validation(self, landscape_db: LandscapeDB, payload_store) -> None:
        """on_error: null (not set) passes validation."""
        from elspeth.contracts import ArtifactDescriptor
        from elspeth.engine.orchestrator import Orchestrator, PipelineConfig

        class InputSchema(PluginSchema):
            value: int

        class ListSource(_TestSourceBase):
            name = "list_source"
            output_schema = InputSchema

            def __init__(self, data: list[dict[str, Any]]) -> None:
                self._data = data

            def load(self, ctx: Any) -> Iterator[SourceRow]:
                for row in self._data:
                    yield SourceRow.valid(row)

        class NormalTransform(BaseTransform):
            """Transform with no error routing (default)."""

            name = "normal_transform"
            input_schema = InputSchema
            output_schema = InputSchema
            # _on_error is None by default from BaseTransform

            def __init__(self) -> None:
                super().__init__({"schema": {"fields": "dynamic"}})

            def process(self, row: dict[str, Any], ctx: Any) -> TransformResult:
                return TransformResult.success(row, success_reason={"action": "test"})

        class CollectSink(_TestSinkBase):
            name = "collect"

            def __init__(self) -> None:
                self.results: list[dict[str, Any]] = []

            def on_start(self, ctx: Any) -> None:
                pass

            def on_complete(self, ctx: Any) -> None:
                pass

            def write(self, rows: Any, ctx: Any) -> ArtifactDescriptor:
                self.results.extend(rows)
                return ArtifactDescriptor.for_file(path="memory", size_bytes=0, content_hash="")

            def close(self) -> None:
                pass

        source = ListSource([{"value": 1}])
        transform = NormalTransform()
        sink = CollectSink()

        config = PipelineConfig(
            source=as_source(source),
            transforms=[as_transform(transform)],
            sinks={"default": as_sink(sink)},
        )

        orchestrator = Orchestrator(landscape_db)

        # Should NOT raise - None is valid (no error routing configured)
        result = orchestrator.run(config, graph=build_production_graph(config), payload_store=payload_store)
        assert result.status == "completed"

    def test_valid_on_error_sink_passes_validation(self, landscape_db: LandscapeDB, payload_store) -> None:
        """Valid on_error sink name passes validation."""
        from elspeth.contracts import ArtifactDescriptor
        from elspeth.engine.orchestrator import Orchestrator, PipelineConfig

        class InputSchema(PluginSchema):
            value: int

        class ListSource(_TestSourceBase):
            name = "list_source"
            output_schema = InputSchema

            def __init__(self, data: list[dict[str, Any]]) -> None:
                self._data = data

            def load(self, ctx: Any) -> Iterator[SourceRow]:
                for row in self._data:
                    yield SourceRow.valid(row)

        class ErrorRoutingTransform(BaseTransform):
            """Transform that routes errors to a valid sink."""

            name = "error_routing_transform"
            input_schema = InputSchema
            output_schema = InputSchema
            _on_error = "error_sink"  # This sink exists in config

            def __init__(self) -> None:
                super().__init__({"schema": {"fields": "dynamic"}})

            def process(self, row: dict[str, Any], ctx: Any) -> TransformResult:
                return TransformResult.success(row, success_reason={"action": "test"})

        class CollectSink(_TestSinkBase):
            name = "collect"

            def __init__(self) -> None:
                self.results: list[dict[str, Any]] = []

            def on_start(self, ctx: Any) -> None:
                pass

            def on_complete(self, ctx: Any) -> None:
                pass

            def write(self, rows: Any, ctx: Any) -> ArtifactDescriptor:
                self.results.extend(rows)
                return ArtifactDescriptor.for_file(path="memory", size_bytes=0, content_hash="")

            def close(self) -> None:
                pass

        source = ListSource([{"value": 1}])
        transform = ErrorRoutingTransform()
        default_sink = CollectSink()
        error_sink = CollectSink()

        config = PipelineConfig(
            source=as_source(source),
            transforms=[as_transform(transform)],
            sinks={
                "default": as_sink(default_sink),
                "error_sink": as_sink(error_sink),  # Valid target for on_error
            },
        )

        orchestrator = Orchestrator(landscape_db)

        # Should NOT raise - "error_sink" exists
        result = orchestrator.run(config, graph=build_production_graph(config), payload_store=payload_store)
        assert result.status == "completed"

    def test_validation_occurs_before_row_processing(self, landscape_db: LandscapeDB, payload_store) -> None:
        """Error occurs BEFORE any rows are processed (source.load should not be called)."""
        from elspeth.contracts import ArtifactDescriptor
        from elspeth.engine.orchestrator import (
            Orchestrator,
            PipelineConfig,
            RouteValidationError,
        )

        class InputSchema(PluginSchema):
            value: int

        call_tracking: dict[str, bool] = {
            "source_load_called": False,
            "transform_process_called": False,
            "sink_write_called": False,
        }

        class TrackingSource(_TestSourceBase):
            """Source that tracks all method calls."""

            name = "tracking_source"
            output_schema = InputSchema

            def __init__(self, data: list[dict[str, Any]]) -> None:
                self._data = data

            def load(self, ctx: Any) -> Iterator[SourceRow]:
                call_tracking["source_load_called"] = True
                for row in self._data:
                    yield SourceRow.valid(row)

        class TrackingTransform(BaseTransform):
            """Transform that tracks process() calls and has invalid on_error."""

            name = "tracking_transform"
            input_schema = InputSchema
            output_schema = InputSchema
            _on_error = "invalid_sink_name"  # Does not exist

            def __init__(self) -> None:
                super().__init__({"schema": {"fields": "dynamic"}})

            def process(self, row: dict[str, Any], ctx: Any) -> TransformResult:
                call_tracking["transform_process_called"] = True
                return TransformResult.success(row, success_reason={"action": "test"})

        class TrackingSink(_TestSinkBase):
            """Sink that tracks write() calls."""

            name = "tracking_sink"

            def __init__(self) -> None:
                self.results: list[dict[str, Any]] = []

            def on_start(self, ctx: Any) -> None:
                pass

            def on_complete(self, ctx: Any) -> None:
                pass

            def write(self, rows: Any, ctx: Any) -> ArtifactDescriptor:
                call_tracking["sink_write_called"] = True
                self.results.extend(rows)
                return ArtifactDescriptor.for_file(path="memory", size_bytes=0, content_hash="")

            def close(self) -> None:
                pass

        source = TrackingSource([{"value": 1}, {"value": 2}, {"value": 3}])
        transform = TrackingTransform()
        sink = TrackingSink()

        config = PipelineConfig(
            source=as_source(source),
            transforms=[as_transform(transform)],
            sinks={"default": as_sink(sink)},
        )

        orchestrator = Orchestrator(landscape_db)

        # Must raise RouteValidationError at startup
        with pytest.raises(RouteValidationError):
            orchestrator.run(config, graph=build_production_graph(config), payload_store=payload_store)

        # Verify NOTHING was processed - validation caught error before processing started
        assert not call_tracking["source_load_called"], "source.load() should NOT be called"
        assert not call_tracking["transform_process_called"], "transform.process() should NOT be called"
        assert not call_tracking["sink_write_called"], "sink.write() should NOT be called"
