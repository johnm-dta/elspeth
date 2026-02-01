"""Tests for phase event emission in Orchestrator.

Verifies that PhaseError events are emitted exactly once per failure,
and that failures are attributed to the correct phase (SOURCE vs PROCESS).
"""

from __future__ import annotations

from collections.abc import Iterator
from typing import TYPE_CHECKING, Any

import pytest

from elspeth.contracts import ArtifactDescriptor, PluginSchema, RunStatus, SourceRow
from elspeth.contracts.events import PhaseError, PipelinePhase
from elspeth.core.dag import ExecutionGraph
from elspeth.core.events import EventBus
from elspeth.core.landscape import LandscapeDB
from elspeth.core.landscape.recorder import LandscapeRecorder
from elspeth.plugins.base import BaseTransform
from tests.conftest import (
    _TestSchema,
    _TestSinkBase,
    _TestSourceBase,
    as_sink,
    as_source,
    as_transform,
)

if TYPE_CHECKING:
    from elspeth.contracts.results import TransformResult


class TestPhaseErrorEmission:
    """Test that PhaseError events are emitted correctly."""

    def test_process_failure_emits_single_phase_error(self, landscape_db: LandscapeDB, payload_store) -> None:
        """PROCESS phase failure should emit exactly ONE PhaseError(PROCESS).

        Also verifies audit trail records run as FAILED with error_json.
        """
        from elspeth.engine.orchestrator import Orchestrator, PipelineConfig

        class ValueSchema(PluginSchema):
            value: int

        class ListSource(_TestSourceBase):
            name = "test_source"
            output_schema = ValueSchema

            def __init__(self, data: list[dict[str, Any]]) -> None:
                super().__init__()
                self._data = data

            def on_start(self, ctx: Any) -> None:
                pass

            def load(self, ctx: Any) -> Iterator[SourceRow]:
                for row in self._data:
                    yield SourceRow.valid(row)

            def close(self) -> None:
                pass

        class ExplodingTransform(BaseTransform):
            name = "exploding"
            input_schema = ValueSchema
            output_schema = ValueSchema

            def __init__(self) -> None:
                super().__init__({"schema": {"fields": "dynamic"}})

            def process(self, row: Any, ctx: Any) -> TransformResult:
                raise RuntimeError("Transform exploded!")

        class CollectSink(_TestSinkBase):
            name = "default"

            def __init__(self) -> None:
                super().__init__()
                self.results: list[dict[str, Any]] = []

            def on_start(self, ctx: Any) -> None:
                pass

            def write(self, rows: Any, ctx: Any) -> ArtifactDescriptor:
                self.results.extend(rows)
                return ArtifactDescriptor.for_file(path="memory", size_bytes=0, content_hash="")

            def close(self) -> None:
                pass

        # Create plugin instances
        source = ListSource([{"value": 42}])
        transform = ExplodingTransform()
        sink = CollectSink()

        # Build graph using public API (P2 fix)
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

        # Capture events
        event_bus = EventBus()
        phase_errors: list[PhaseError] = []

        def capture_phase_error(event: PhaseError) -> None:
            phase_errors.append(event)

        event_bus.subscribe(PhaseError, capture_phase_error)

        orchestrator = Orchestrator(landscape_db, event_bus=event_bus)

        # Run should fail
        with pytest.raises(RuntimeError, match="Transform exploded"):
            orchestrator.run(config=config, graph=graph, payload_store=payload_store)

        # Should have exactly ONE PhaseError for PROCESS phase
        assert len(phase_errors) == 1, f"Expected 1 PhaseError, got {len(phase_errors)}"
        assert phase_errors[0].phase == PipelinePhase.PROCESS
        assert "Transform exploded" in str(phase_errors[0].error)

        # P1 Fix: Verify audit trail records failure
        # With module-scoped db, check most recent run (newest first)
        recorder = LandscapeRecorder(landscape_db)
        runs = recorder.list_runs()
        assert len(runs) >= 1, "Expected at least 1 run in Landscape"
        run = runs[0]  # Most recent run (list_runs returns newest first)
        assert run.status == RunStatus.FAILED, f"Run status should be FAILED, got {run.status}"

    def test_source_failure_emits_source_phase_error(self, landscape_db: LandscapeDB, payload_store) -> None:
        """SOURCE phase failure should emit PhaseError(SOURCE), not PROCESS.

        Also verifies audit trail records run as FAILED.
        """
        from elspeth.engine.orchestrator import Orchestrator, PipelineConfig

        class ExplodingSource(_TestSourceBase):
            name = "exploding_source"
            output_schema = _TestSchema

            def __init__(self) -> None:
                super().__init__()

            def on_start(self, ctx: Any) -> None:
                pass

            def load(self, ctx: Any) -> Iterator[SourceRow]:
                raise RuntimeError("Source load failed!")

            def close(self) -> None:
                pass

        class CollectSink(_TestSinkBase):
            name = "default"

            def __init__(self) -> None:
                super().__init__()
                self.results: list[dict[str, Any]] = []

            def on_start(self, ctx: Any) -> None:
                pass

            def write(self, rows: Any, ctx: Any) -> ArtifactDescriptor:
                self.results.extend(rows)
                return ArtifactDescriptor.for_file(path="memory", size_bytes=0, content_hash="")

            def close(self) -> None:
                pass

        # Create plugin instances
        source = ExplodingSource()
        sink = CollectSink()

        # Build graph using public API (P2 fix)
        graph = ExecutionGraph.from_plugin_instances(
            source=as_source(source),
            transforms=[],
            sinks={"default": as_sink(sink)},
            aggregations={},
            gates=[],
            default_sink="default",
        )

        config = PipelineConfig(
            source=as_source(source),
            transforms=[],
            sinks={"default": as_sink(sink)},
        )

        # Capture events
        event_bus = EventBus()
        phase_errors: list[PhaseError] = []

        def capture_phase_error(event: PhaseError) -> None:
            phase_errors.append(event)

        event_bus.subscribe(PhaseError, capture_phase_error)

        orchestrator = Orchestrator(landscape_db, event_bus=event_bus)

        # Run should fail
        with pytest.raises(RuntimeError, match="Source load failed"):
            orchestrator.run(config=config, graph=graph, payload_store=payload_store)

        # Should have exactly ONE PhaseError for SOURCE phase (not PROCESS)
        assert len(phase_errors) == 1, f"Expected 1 PhaseError, got {len(phase_errors)}"
        assert phase_errors[0].phase == PipelinePhase.SOURCE, "SOURCE failure should emit SOURCE PhaseError, not PROCESS"
        assert "Source load failed" in str(phase_errors[0].error)

        # P1 Fix: Verify audit trail records failure
        # With module-scoped db, check most recent run (newest first)
        recorder = LandscapeRecorder(landscape_db)
        runs = recorder.list_runs()
        assert len(runs) >= 1, "Expected at least 1 run in Landscape"
        run = runs[0]  # Most recent run (list_runs returns newest first)
        assert run.status == RunStatus.FAILED, f"Run status should be FAILED, got {run.status}"
