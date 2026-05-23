# tests/performance/scalability/test_large_datasets.py
"""Scalability tests for large dataset processing.

Verifies pipeline can handle thousands of rows without failures,
using production assembly path (ExecutionGraph.from_plugin_instances).
"""

from __future__ import annotations

from typing import Any

import pytest

from elspeth.contracts import RunStatus
from elspeth.core.config import GateSettings
from elspeth.core.landscape.database import LandscapeDB
from elspeth.engine.orchestrator import Orchestrator, PipelineConfig
from tests.fixtures.base_classes import as_sink, as_source, as_transform
from tests.fixtures.pipeline import build_linear_pipeline, build_production_graph
from tests.fixtures.plugins import CollectSink, ListSource
from tests.fixtures.stores import MockPayloadStore
from tests.performance.conftest import benchmark_timer

pytestmark = pytest.mark.performance


def _generate_rows(count: int) -> list[dict[str, Any]]:
    """Generate N rows with id and value fields."""
    return [{"id": i, "value": f"data-{i}"} for i in range(count)]


def _run_linear_pipeline(
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
    default_sink = sinks["default"]
    return result, default_sink


class TestLargeDatasets:
    """Pipeline processing with large row counts."""

    def test_pipeline_1000_rows(self) -> None:
        """1000 rows through linear pipeline complete successfully."""
        rows = _generate_rows(1000)
        with benchmark_timer() as _timing:
            result, sink = _run_linear_pipeline(rows)

        assert result.status == RunStatus.COMPLETED
        assert result.rows_processed == 1000
        assert result.rows_succeeded == 1000
        assert len(sink.results) == 1000

    def test_pipeline_10000_rows(self) -> None:
        """10,000 rows through linear pipeline."""
        rows = _generate_rows(10_000)
        with benchmark_timer() as _timing:
            result, sink = _run_linear_pipeline(rows)

        assert result.status == RunStatus.COMPLETED
        assert result.rows_processed == 10_000
        assert result.rows_succeeded == 10_000
        assert len(sink.results) == 10_000

    def test_pipeline_with_gate_1000_rows(self) -> None:
        """1000 rows through gate routing pipeline."""
        # Half the rows go to "high" sink, half continue to default
        rows = [{"id": i, "value": i} for i in range(1000)]

        source = ListSource(rows, on_success="gate_in")
        default_sink = CollectSink(name="default")
        high_sink = CollectSink(name="high")

        gate = GateSettings(
            name="threshold",
            input="gate_in",
            condition="row['value'] >= 500",
            routes={"true": "high", "false": "default"},
        )

        config = PipelineConfig(
            sources={"primary": as_source(source)},
            transforms=[],
            sinks={
                "default": as_sink(default_sink),
                "high": as_sink(high_sink),
            },
            gates=[gate],
        )

        db = LandscapeDB.in_memory()
        payload_store = MockPayloadStore()
        orchestrator = Orchestrator(db)

        with benchmark_timer() as _timing:
            result = orchestrator.run(
                config,
                graph=build_production_graph(config),
                payload_store=payload_store,
            )

        assert result.status == RunStatus.COMPLETED
        assert result.rows_processed == 1000
        # Rows 0-499 go to default, rows 500-999 go to high
        assert len(default_sink.results) == 500
        assert len(high_sink.results) == 500
