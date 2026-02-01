# tests/engine/test_orchestrator_cleanup.py
"""Tests for transform/gate cleanup in orchestrator."""

from __future__ import annotations

from typing import Any

import pytest

from elspeth.contracts import ArtifactDescriptor, PluginSchema, SourceRow
from elspeth.core.dag import ExecutionGraph
from elspeth.core.landscape import LandscapeDB
from elspeth.engine.orchestrator import Orchestrator, PipelineConfig
from elspeth.plugins.base import BaseTransform
from elspeth.plugins.results import TransformResult
from tests.conftest import _TestSinkBase, _TestSourceBase, as_sink, as_source, as_transform


class ValueSchema(PluginSchema):
    """Simple schema for test rows."""

    value: int


class ListSource(_TestSourceBase):
    """Test source that yields from a list."""

    name = "list_source"
    output_schema = ValueSchema

    def __init__(self, data: list[dict[str, Any]]) -> None:
        super().__init__()
        self._data = data

    def on_start(self, ctx: Any) -> None:
        pass

    def on_complete(self, ctx: Any) -> None:
        pass

    def load(self, ctx: Any) -> Any:
        for _row in self._data:
            yield SourceRow.valid(_row)

    def close(self) -> None:
        pass


class FailingSource(ListSource):
    """Test source that raises an exception during load."""

    name = "failing_source"

    def load(self, ctx: Any) -> Any:
        raise RuntimeError("Source failed intentionally")


class CollectSink(_TestSinkBase):
    """Test sink that collects results in memory."""

    name = "collect"
    input_schema = ValueSchema

    def __init__(self) -> None:
        super().__init__()
        self.results: list[dict[str, Any]] = []

    def on_start(self, ctx: Any) -> None:
        pass

    def on_complete(self, ctx: Any) -> None:
        pass

    def write(self, rows: Any, ctx: Any) -> ArtifactDescriptor:
        self.results.extend(rows)
        return ArtifactDescriptor.for_file(path="memory", size_bytes=0, content_hash="")

    def flush(self) -> None:
        pass

    def close(self) -> None:
        pass


class TrackingTransform(BaseTransform):
    """Transform that tracks whether close() was called."""

    input_schema = ValueSchema
    output_schema = ValueSchema

    def __init__(self, name: str = "tracking") -> None:
        super().__init__({"schema": {"fields": "dynamic"}})
        self.name = name  # type: ignore[misc]
        self.close_called = False
        self.close_call_count = 0

    def on_start(self, ctx: Any) -> None:
        pass

    def on_complete(self, ctx: Any) -> None:
        pass

    def process(self, row: Any, ctx: Any) -> TransformResult:
        return TransformResult.success(row, success_reason={"action": "test"})

    def close(self) -> None:
        self.close_called = True
        self.close_call_count += 1


class FailingCloseTransform(TrackingTransform):
    """Transform whose close() raises an error."""

    def close(self) -> None:
        self.close_called = True
        self.close_call_count += 1
        raise RuntimeError("Close failed!")


class TestOrchestratorCleanup:
    """Tests for Orchestrator calling close() on plugins."""

    def test_transforms_closed_on_success(self, real_landscape_db: LandscapeDB, payload_store) -> None:
        """All transforms should have close() called after successful run."""
        transform_1 = TrackingTransform("transform_1")
        transform_2 = TrackingTransform("transform_2")

        source = ListSource([{"value": 1}, {"value": 2}])
        sink = CollectSink()

        # P2 Fix: Use from_plugin_instances instead of private field mutation
        graph = ExecutionGraph.from_plugin_instances(
            source=as_source(source),
            transforms=[as_transform(transform_1), as_transform(transform_2)],
            sinks={"default": as_sink(sink)},
            aggregations={},
            gates=[],
            default_sink="default",
        )

        config = PipelineConfig(
            source=as_source(source),
            transforms=[as_transform(transform_1), as_transform(transform_2)],
            sinks={"default": as_sink(sink)},
        )

        orchestrator = Orchestrator(real_landscape_db)
        orchestrator.run(config, graph=graph, payload_store=payload_store)

        # Verify close() was called on all transforms
        assert transform_1.close_called, "transform_1.close() was not called"
        assert transform_1.close_call_count == 1, "transform_1.close() called multiple times"
        assert transform_2.close_called, "transform_2.close() was not called"
        assert transform_2.close_call_count == 1, "transform_2.close() called multiple times"

    def test_transforms_closed_on_failure(self, real_landscape_db: LandscapeDB, payload_store) -> None:
        """All transforms should have close() called even if run fails."""
        transform_1 = TrackingTransform("transform_1")
        transform_2 = TrackingTransform("transform_2")

        # Use failing source
        source = FailingSource([{"value": 1}])
        sink = CollectSink()

        # P2 Fix: Use from_plugin_instances instead of private field mutation
        graph = ExecutionGraph.from_plugin_instances(
            source=as_source(source),
            transforms=[as_transform(transform_1), as_transform(transform_2)],
            sinks={"default": as_sink(sink)},
            aggregations={},
            gates=[],
            default_sink="default",
        )

        config = PipelineConfig(
            source=as_source(source),
            transforms=[as_transform(transform_1), as_transform(transform_2)],
            sinks={"default": as_sink(sink)},
        )

        orchestrator = Orchestrator(real_landscape_db)

        with pytest.raises(RuntimeError, match="Source failed intentionally"):
            orchestrator.run(config, graph=graph, payload_store=payload_store)

        # Verify close() was called on all transforms even though run failed
        assert transform_1.close_called, "transform_1.close() was not called after failure"
        assert transform_2.close_called, "transform_2.close() was not called after failure"

    def test_cleanup_handles_missing_close_method(self, real_landscape_db: LandscapeDB, payload_store) -> None:
        """Cleanup should handle transforms that use default close() method.

        BaseTransform provides a default no-op close() method, so transforms
        that don't override it still satisfy the protocol. This test verifies
        the cleanup process works correctly with the default implementation.
        """

        # Transform using BaseTransform's default close() implementation
        class MinimalTransform(BaseTransform):
            name = "minimal"
            input_schema = ValueSchema
            output_schema = ValueSchema

            def __init__(self) -> None:
                super().__init__({"schema": {"fields": "dynamic"}})

            def process(self, row: Any, ctx: Any) -> TransformResult:
                return TransformResult.success(row, success_reason={"action": "test"})

            # Uses default close() from BaseTransform (no-op)

        source = ListSource([{"value": 1}])
        transform = MinimalTransform()
        sink = CollectSink()

        # P2 Fix: Use from_plugin_instances instead of private field mutation
        graph = ExecutionGraph.from_plugin_instances(
            source=as_source(source),
            transforms=[as_transform(transform)],
            sinks={"default": as_sink(sink)},
            aggregations={},
            gates=[],
            default_sink="default",
        )

        config = PipelineConfig(
            source=as_source(source),
            transforms=[as_transform(transform)],
            sinks={"default": as_sink(sink)},
        )

        orchestrator = Orchestrator(real_landscape_db)
        # Should not raise even though transform has no close()
        result = orchestrator.run(config, graph=graph, payload_store=payload_store)

        assert result.status == "completed"

    def test_cleanup_continues_if_one_close_fails(self, real_landscape_db: LandscapeDB, payload_store) -> None:
        """If one transform's close() fails, others should still be closed.

        Cleanup should be best-effort - one plugin failure shouldn't prevent
        cleanup of other plugins. After attempting all cleanups, the error
        is raised to surface the bug (plugins are system code per CLAUDE.md).
        """
        # First transform: close() raises an error
        transform_1 = FailingCloseTransform("failing_close")

        # Second transform: close() works normally
        transform_2 = TrackingTransform("normal_close")

        source = ListSource([{"value": 1}])
        sink = CollectSink()

        # P2 Fix: Use from_plugin_instances instead of private field mutation
        graph = ExecutionGraph.from_plugin_instances(
            source=as_source(source),
            transforms=[as_transform(transform_1), as_transform(transform_2)],
            sinks={"default": as_sink(sink)},
            aggregations={},
            gates=[],
            default_sink="default",
        )

        config = PipelineConfig(
            source=as_source(source),
            transforms=[as_transform(transform_1), as_transform(transform_2)],
            sinks={"default": as_sink(sink)},
        )

        orchestrator = Orchestrator(real_landscape_db)
        # Should raise RuntimeError after attempting all cleanups (per CLAUDE.md, plugins are system code)
        with pytest.raises(RuntimeError, match="Plugin cleanup failed"):
            orchestrator.run(config, graph=graph, payload_store=payload_store)

        # Both close() methods should have been called (cleanup attempts all before failing)
        assert transform_1.close_called, "failing transform's close() was not called"
        assert transform_2.close_called, "second transform's close() was not called despite first failing"
