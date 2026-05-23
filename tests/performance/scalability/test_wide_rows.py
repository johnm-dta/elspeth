# tests/performance/scalability/test_wide_rows.py
"""Scalability tests for rows with many fields.

Verifies pipeline can process rows with hundreds or thousands of fields
without degradation or failure.
"""

from __future__ import annotations

from typing import Any

import pytest

from elspeth.contracts import RunStatus
from elspeth.core.landscape.database import LandscapeDB
from elspeth.engine.orchestrator import Orchestrator, PipelineConfig
from tests.fixtures.base_classes import as_sink, as_source, as_transform
from tests.fixtures.pipeline import build_linear_pipeline
from tests.fixtures.plugins import CollectSink
from tests.fixtures.stores import MockPayloadStore
from tests.performance.conftest import benchmark_timer

pytestmark = pytest.mark.performance


def _generate_wide_rows(width: int, count: int = 10) -> list[dict[str, Any]]:
    """Generate rows with the given number of fields."""
    return [{f"field_{j}": f"value_{i}_{j}" for j in range(width)} for i in range(count)]


def _run_wide_pipeline(
    rows: list[dict[str, Any]],
    transforms: list[Any] | None = None,
) -> tuple[Any, CollectSink]:
    """Run a linear pipeline with wide rows."""
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
    default_sink = sinks["default"]
    return result, default_sink


class TestWideRows:
    """Pipeline processing with wide rows (many fields per row)."""

    def test_wide_row_100_fields(self) -> None:
        """Rows with 100 fields through pipeline."""
        rows = _generate_wide_rows(width=100, count=50)
        with benchmark_timer() as _timing:
            result, sink = _run_wide_pipeline(rows)

        assert result.status == RunStatus.COMPLETED
        assert result.rows_processed == 50
        assert result.rows_succeeded == 50
        assert len(sink.results) == 50
        # Verify field count preserved
        assert len(sink.results[0]) == 100

    def test_wide_row_500_fields(self) -> None:
        """Rows with 500 fields through pipeline."""
        rows = _generate_wide_rows(width=500, count=20)
        with benchmark_timer() as _timing:
            result, sink = _run_wide_pipeline(rows)

        assert result.status == RunStatus.COMPLETED
        assert result.rows_processed == 20
        assert result.rows_succeeded == 20
        assert len(sink.results) == 20
        assert len(sink.results[0]) == 500

    def test_wide_row_1000_fields(self) -> None:
        """Rows with 1000 fields through pipeline.

        Larger field count may be slower due to hashing and serialization.
        Timing is recorded but not strictly asserted (varies by hardware).
        """
        rows = _generate_wide_rows(width=1000, count=10)
        with benchmark_timer() as _timing:
            result, sink = _run_wide_pipeline(rows)

        assert result.status == RunStatus.COMPLETED
        assert result.rows_processed == 10
        assert result.rows_succeeded == 10
        assert len(sink.results) == 10
        assert len(sink.results[0]) == 1000
        # Sanity check: should complete within 120 seconds even on slow CI
        assert _timing.wall_seconds < 120, f"1000-field rows took {_timing.wall_seconds:.1f}s, expected < 120s"
