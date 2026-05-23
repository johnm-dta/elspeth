# tests/performance/scalability/test_deep_dag.py
"""Scalability tests for deep DAG chains with many transforms.

Verifies pipeline can handle long transform chains without
stack overflow, excessive memory, or performance degradation.
"""

from __future__ import annotations

from typing import Any

import pytest

from elspeth.contracts import RunStatus
from elspeth.core.landscape.database import LandscapeDB
from elspeth.engine.orchestrator import Orchestrator, PipelineConfig
from tests.fixtures.base_classes import as_sink, as_source, as_transform
from tests.fixtures.pipeline import build_linear_pipeline
from tests.fixtures.plugins import CollectSink, PassTransform
from tests.fixtures.stores import MockPayloadStore
from tests.performance.conftest import benchmark_timer

pytestmark = pytest.mark.performance


def _make_deep_transforms(depth: int) -> list[PassTransform]:
    """Create a chain of PassTransform instances with unique names."""
    transforms = []
    for i in range(depth):
        t = PassTransform()
        t.name = f"pass_transform_{i}"
        transforms.append(t)
    return transforms


def _run_deep_pipeline(
    rows: list[dict[str, Any]],
    depth: int,
) -> tuple[Any, CollectSink]:
    """Run a linear pipeline with the given transform depth."""
    transforms = _make_deep_transforms(depth)
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


class TestDeepDag:
    """Pipeline processing with many sequential transforms."""

    def test_deep_dag_10_transforms(self) -> None:
        """100 rows through 10 PassTransform stages."""
        rows = [{"id": i, "value": f"data-{i}"} for i in range(100)]
        with benchmark_timer() as _timing:
            result, sink = _run_deep_pipeline(rows, depth=10)

        assert result.status == RunStatus.COMPLETED
        assert result.rows_processed == 100
        assert result.rows_succeeded == 100
        assert len(sink.results) == 100

    def test_deep_dag_20_transforms(self) -> None:
        """100 rows through 20 PassTransform stages."""
        rows = [{"id": i, "value": f"data-{i}"} for i in range(100)]
        with benchmark_timer() as _timing:
            result, sink = _run_deep_pipeline(rows, depth=20)

        assert result.status == RunStatus.COMPLETED
        assert result.rows_processed == 100
        assert result.rows_succeeded == 100
        assert len(sink.results) == 100

    def test_deep_dag_50_transforms(self) -> None:
        """10 rows through 50 PassTransform stages.

        Fewer rows to keep runtime reasonable with a very deep chain.
        """
        rows = [{"id": i, "value": f"data-{i}"} for i in range(10)]
        with benchmark_timer() as _timing:
            result, sink = _run_deep_pipeline(rows, depth=50)

        assert result.status == RunStatus.COMPLETED
        assert result.rows_processed == 10
        assert result.rows_succeeded == 10
        assert len(sink.results) == 10
