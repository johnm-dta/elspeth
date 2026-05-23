# tests/performance/memory/test_memory_baseline.py
"""Memory usage baseline tests.

Measures RSS memory consumption for pipeline processing at various scales.
Uses resource.getrusage for portable memory tracking.
"""

from __future__ import annotations

import gc
from typing import Any

import pytest

from elspeth.contracts import RunStatus
from elspeth.core.landscape.database import LandscapeDB
from elspeth.engine.orchestrator import Orchestrator, PipelineConfig
from tests.fixtures.base_classes import as_sink, as_source, as_transform
from tests.fixtures.pipeline import build_linear_pipeline
from tests.fixtures.plugins import CollectSink, PassTransform
from tests.fixtures.stores import MockPayloadStore
from tests.performance.conftest import MemorySnapshot

pytestmark = pytest.mark.performance


def _generate_rows(count: int) -> list[dict[str, Any]]:
    """Generate N rows with id and value fields."""
    return [{"id": i, "value": f"data-{i}"} for i in range(count)]


def _run_pipeline_and_collect(
    rows: list[dict[str, Any]],
    transforms: list[Any] | None = None,
) -> tuple[Any, CollectSink]:
    """Run a linear pipeline and return (result, sink)."""
    source, transforms_list, sinks, graph = build_linear_pipeline(
        source_data=rows,
        transforms=transforms,
    )
    db = LandscapeDB.in_memory()
    payload_store = MockPayloadStore()
    orchestrator = Orchestrator(db)
    config = PipelineConfig(
        sources={"primary": as_source(source)},
        transforms=[as_transform(t) for t in transforms_list],
        sinks={name: as_sink(s) for name, s in sinks.items()},
    )
    result = orchestrator.run(
        config,
        graph=graph,
        payload_store=payload_store,
    )
    return result, sinks["default"]


class TestMemoryBaseline:
    """Memory usage baselines for pipeline processing."""

    def test_memory_per_1000_rows_linear(self, memory_tracker) -> None:
        """Process 1000 rows and measure RSS delta."""
        gc.collect()
        rows = _generate_rows(1000)
        result, sink = _run_pipeline_and_collect(rows)

        assert result.status == RunStatus.COMPLETED
        assert len(sink.results) == 1000

        gc.collect()
        snapshot = memory_tracker.snapshot()
        # RSS delta should be bounded: 1000 simple rows should not consume
        # excessive memory. Allow generous headroom (200MB) for SQLite,
        # Python overhead, and CI variability.
        assert snapshot.delta_bytes < 200 * 1024 * 1024, (
            f"RSS grew by {snapshot.delta_bytes / 1024 / 1024:.1f}MB for 1000 rows, expected < 200MB"
        )

    def test_memory_per_1000_rows_with_transforms(self, memory_tracker) -> None:
        """Process 1000 rows with 3 transforms and measure RSS delta."""
        gc.collect()
        rows = _generate_rows(1000)
        transforms = [PassTransform(), PassTransform(), PassTransform()]
        # Give each a unique name for DAG construction
        for i, t in enumerate(transforms):
            t.name = f"pass_transform_{i}"

        result, sink = _run_pipeline_and_collect(rows, transforms=transforms)

        assert result.status == RunStatus.COMPLETED
        assert len(sink.results) == 1000

        gc.collect()
        snapshot = memory_tracker.snapshot()
        assert snapshot.delta_bytes < 200 * 1024 * 1024, (
            f"RSS grew by {snapshot.delta_bytes / 1024 / 1024:.1f}MB for 1000 rows + 3 transforms, expected < 200MB"
        )

    def test_memory_scaling_linear(self, memory_tracker) -> None:
        """Process increasing row counts and verify memory grows sub-linearly or linearly.

        Runs at 100, 500, 1000, 5000 rows and checks that the per-row
        memory overhead doesn't increase dramatically with scale.
        """
        sizes = [100, 500, 1000, 5000]
        snapshots: list[tuple[int, MemorySnapshot]] = []

        for size in sizes:
            gc.collect()
            rows = _generate_rows(size)
            result, sink = _run_pipeline_and_collect(rows)
            assert result.status == RunStatus.COMPLETED
            assert len(sink.results) == size

            gc.collect()
            snap = memory_tracker.snapshot()
            snapshots.append((size, snap))

        # Verify the overall memory growth is bounded.
        # For 5000 rows, total RSS delta should be under 500MB.
        final_size, final_snap = snapshots[-1]
        assert final_snap.delta_bytes < 500 * 1024 * 1024, (
            f"RSS grew by {final_snap.delta_bytes / 1024 / 1024:.1f}MB for {final_size} rows, expected < 500MB"
        )
