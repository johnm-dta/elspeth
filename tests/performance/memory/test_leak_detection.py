# tests/performance/memory/test_leak_detection.py
"""Memory leak detection tests.

Runs pipelines repeatedly to detect unbounded memory growth.
Uses RSS tracking to catch leaks in LandscapeDB, orchestrator,
or plugin lifecycle management.
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
from tests.fixtures.stores import MockPayloadStore
from tests.performance.conftest import MemorySnapshot

pytestmark = pytest.mark.performance


def _generate_rows(count: int) -> list[dict[str, Any]]:
    """Generate N rows with id and value fields."""
    return [{"id": i, "value": f"data-{i}"} for i in range(count)]


def _run_disposable_pipeline(rows: list[dict[str, Any]]) -> None:
    """Run a pipeline and discard all references.

    Creates fresh DB, orchestrator, and plugins each time to ensure
    no cross-run state leaks.
    """
    source, transforms_list, sinks, graph = build_linear_pipeline(
        source_data=rows,
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
    assert result.status == RunStatus.COMPLETED


class TestLeakDetection:
    """Repeated pipeline runs to detect memory leaks."""

    def test_no_leak_repeated_pipeline_runs(self, memory_tracker) -> None:
        """Run pipeline multiple times, check RSS does not grow unboundedly.

        Runs a 100-row pipeline 50 times, taking memory snapshots every
        10 iterations. Verifies the total memory growth is bounded.
        """
        rows = _generate_rows(100)
        snapshots: list[MemorySnapshot] = []

        for iteration in range(50):
            _run_disposable_pipeline(rows)

            if iteration % 10 == 0:
                gc.collect()
                snapshots.append(memory_tracker.snapshot())

        # Final snapshot after all iterations
        gc.collect()
        snapshots.append(memory_tracker.snapshot())

        # Check that memory growth between first and last snapshot is bounded.
        # Allow some growth (caches, SQLAlchemy metadata, etc.) but not
        # proportional to iteration count.
        growth_bytes = snapshots[-1].delta_bytes - snapshots[0].delta_bytes
        max_allowed_growth = 50 * 1024 * 1024  # 50MB max growth for 50 iterations
        assert growth_bytes < max_allowed_growth, (
            f"Memory grew by {growth_bytes / 1024 / 1024:.1f}MB over 50 iterations, "
            f"expected < {max_allowed_growth / 1024 / 1024:.0f}MB. "
            f"Possible memory leak."
        )

    def test_no_leak_landscape_recording(self, memory_tracker) -> None:
        """Record many rows to in-memory DB, verify cleanup is possible.

        Creates a single LandscapeDB, records 10K rows across multiple runs,
        then verifies memory is bounded.
        """
        gc.collect()

        # Run 10 pipelines of 1000 rows each, all into the same DB
        db = LandscapeDB.in_memory()
        payload_store = MockPayloadStore()

        for _run_idx in range(10):
            rows = _generate_rows(1000)
            source, transforms_list, sinks, graph = build_linear_pipeline(
                source_data=rows,
            )
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
            assert result.status == RunStatus.COMPLETED
            assert result.rows_processed == 1000

        gc.collect()
        snapshot = memory_tracker.snapshot()

        # 10K rows across 10 runs into in-memory SQLite should not consume
        # excessive memory. Allow generous headroom for SQLite page cache.
        max_allowed = 300 * 1024 * 1024  # 300MB
        assert snapshot.delta_bytes < max_allowed, (
            f"RSS grew by {snapshot.delta_bytes / 1024 / 1024:.1f}MB after recording 10K rows, expected < {max_allowed / 1024 / 1024:.0f}MB"
        )
