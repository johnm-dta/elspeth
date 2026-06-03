# tests/integration/pipeline/orchestrator/test_orchestrator_cleanup.py
"""Tests for transform/gate cleanup in orchestrator.

Migrated from tests/engine/test_orchestrator_cleanup.py.
Uses v2 fixtures with function-scoped databases.
"""

from __future__ import annotations

from typing import Any

import pytest

from elspeth.contracts import Determinism, PipelineRow, PluginSchema
from elspeth.core.landscape import LandscapeDB
from elspeth.engine.orchestrator import Orchestrator, PipelineConfig
from elspeth.plugins.infrastructure.base import BaseTransform
from elspeth.plugins.infrastructure.results import TransformResult
from tests.fixtures.base_classes import as_sink, as_source, as_transform
from tests.fixtures.pipeline import build_production_graph
from tests.fixtures.plugins import CollectSink, FailingSource, ListSource


class ValueSchema(PluginSchema):
    """Simple schema for test rows."""

    value: int


class TrackingTransform(BaseTransform):
    """Transform that tracks whether close() was called."""

    determinism = Determinism.DETERMINISTIC

    input_schema = ValueSchema
    output_schema = ValueSchema

    def __init__(self, name: str = "tracking") -> None:
        super().__init__({"schema": {"mode": "observed"}})
        self.name = name
        self.close_called = False
        self.close_call_count = 0

    def on_start(self, ctx: Any) -> None:
        super().on_start(ctx)

    def on_complete(self, ctx: Any) -> None:
        pass

    def process(self, row: PipelineRow, ctx: Any) -> TransformResult:
        return TransformResult.success(row, success_reason={"action": "test"})

    def close(self) -> None:
        self.close_called = True
        self.close_call_count += 1


class FailingCloseTransform(TrackingTransform):
    """Transform whose close() raises an error."""

    determinism = Determinism.DETERMINISTIC

    def close(self) -> None:
        self.close_called = True
        self.close_call_count += 1
        raise RuntimeError("Close failed!")


class TestOrchestratorCleanup:
    """Tests for Orchestrator calling close() on plugins."""

    def test_transforms_closed_on_success(self, landscape_db: LandscapeDB, payload_store) -> None:
        """All transforms should have close() called after successful run."""
        transform_1 = TrackingTransform("transform_1")
        transform_2 = TrackingTransform("transform_2")
        transform_2.on_success = "default"

        source = ListSource([{"value": 1}, {"value": 2}])
        sink = CollectSink()

        config = PipelineConfig(
            source=as_source(source),
            transforms=[as_transform(transform_1), as_transform(transform_2)],
            sinks={"default": as_sink(sink)},
        )
        graph = build_production_graph(config)

        orchestrator = Orchestrator(landscape_db)
        orchestrator.run(config, graph=graph, payload_store=payload_store)

        assert transform_1.close_called, "transform_1.close() was not called"
        assert transform_1.close_call_count == 1, "transform_1.close() called multiple times"
        assert transform_2.close_called, "transform_2.close() was not called"
        assert transform_2.close_call_count == 1, "transform_2.close() called multiple times"

    def test_transforms_closed_on_failure(self, landscape_db: LandscapeDB, payload_store) -> None:
        """All transforms should have close() called even if run fails."""
        transform_1 = TrackingTransform("transform_1")
        transform_2 = TrackingTransform("transform_2")
        transform_2.on_success = "default"

        source = FailingSource()
        sink = CollectSink()

        config = PipelineConfig(
            source=as_source(source),
            transforms=[as_transform(transform_1), as_transform(transform_2)],
            sinks={"default": as_sink(sink)},
        )
        graph = build_production_graph(config)

        orchestrator = Orchestrator(landscape_db)

        with pytest.raises(RuntimeError, match="Source failed intentionally"):
            orchestrator.run(config, graph=graph, payload_store=payload_store)

        assert transform_1.close_called, "transform_1.close() was not called after failure"
        assert transform_2.close_called, "transform_2.close() was not called after failure"

    def test_cleanup_handles_missing_close_method(self, landscape_db: LandscapeDB, payload_store) -> None:
        """Cleanup should handle transforms that use default close() method."""

        class MinimalTransform(BaseTransform):
            name = "minimal"
            determinism = Determinism.DETERMINISTIC
            input_schema = ValueSchema
            output_schema = ValueSchema

            def __init__(self) -> None:
                super().__init__({"schema": {"mode": "observed"}})

            def process(self, row: PipelineRow, ctx: Any) -> TransformResult:
                return TransformResult.success(row, success_reason={"action": "test"})

        source = ListSource([{"value": 1}])
        transform = MinimalTransform()
        transform.on_success = "default"
        sink = CollectSink()

        config = PipelineConfig(
            source=as_source(source),
            transforms=[as_transform(transform)],
            sinks={"default": as_sink(sink)},
        )
        graph = build_production_graph(config)

        orchestrator = Orchestrator(landscape_db)
        result = orchestrator.run(config, graph=graph, payload_store=payload_store)

        assert result.status == "completed"

    def test_cleanup_continues_if_one_close_fails(self, landscape_db: LandscapeDB, payload_store) -> None:
        """If one transform's close() fails, others should still be closed."""
        transform_1 = FailingCloseTransform("failing_close")
        transform_2 = TrackingTransform("normal_close")
        transform_2.on_success = "default"

        source = ListSource([{"value": 1}])
        sink = CollectSink()

        config = PipelineConfig(
            source=as_source(source),
            transforms=[as_transform(transform_1), as_transform(transform_2)],
            sinks={"default": as_sink(sink)},
        )
        graph = build_production_graph(config)

        orchestrator = Orchestrator(landscape_db)
        with pytest.raises(RuntimeError, match="Plugin cleanup failed"):
            orchestrator.run(config, graph=graph, payload_store=payload_store)

        assert transform_1.close_called, "failing transform's close() was not called"
        assert transform_2.close_called, "second transform's close() was not called despite first failing"
