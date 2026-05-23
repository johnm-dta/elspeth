# tests/performance/scalability/test_many_sinks.py
"""Scalability tests for routing to many named sinks.

Verifies pipeline can route rows to many distinct sink destinations
via config-driven gates, using ExecutionGraph.from_plugin_instances().
"""

from __future__ import annotations

from typing import Any

import pytest

from elspeth.contracts import RunStatus
from elspeth.core.config import GateSettings
from elspeth.core.landscape.database import LandscapeDB
from elspeth.engine.orchestrator import Orchestrator, PipelineConfig
from tests.fixtures.base_classes import as_sink, as_source
from tests.fixtures.pipeline import build_production_graph
from tests.fixtures.plugins import CollectSink, ListSource
from tests.fixtures.stores import MockPayloadStore
from tests.performance.conftest import benchmark_timer

pytestmark = pytest.mark.performance


def _build_many_sinks_pipeline(
    num_sinks: int,
    rows_per_sink: int = 2,
) -> tuple[PipelineConfig, dict[str, CollectSink], list[dict[str, Any]]]:
    """Build a pipeline that routes to N named sinks.

    Creates rows with a 'bucket' field (0..num_sinks-1).
    Each bucket value is routed to its corresponding sink via a gate chain.
    Remaining rows (if any) fall through to the default sink.

    Returns:
        (config, sinks_dict, source_rows)
    """
    # Generate source rows: each row gets a bucket assignment
    rows: list[dict[str, Any]] = []
    for sink_idx in range(num_sinks):
        for row_idx in range(rows_per_sink):
            rows.append({"id": sink_idx * rows_per_sink + row_idx, "bucket": sink_idx})

    # Create sinks
    sinks: dict[str, CollectSink] = {"default": CollectSink(name="default")}
    for i in range(num_sinks):
        sinks[f"sink_{i}"] = CollectSink(name=f"sink_{i}")

    # Create gate chain: each gate routes one bucket value to its sink.
    # Rows not matching continue to the next gate via connection names.
    # Last gate routes false to the default sink.
    gates: list[GateSettings] = []
    for i in range(num_sinks):
        gate_input = "gate_chain_in" if i == 0 else f"gate_{i}_in"
        if i < num_sinks - 1:
            false_target = f"gate_{i + 1}_in"
        else:
            false_target = "default"
        gates.append(
            GateSettings(
                name=f"route_{i}",
                input=gate_input,
                condition=f"row['bucket'] == {i}",
                routes={"true": f"sink_{i}", "false": false_target},
            )
        )

    source = ListSource(rows, on_success="gate_chain_in")

    config = PipelineConfig(
        sources={"primary": as_source(source)},
        transforms=[],
        sinks={name: as_sink(s) for name, s in sinks.items()},
        gates=gates,
    )

    return config, sinks, rows


class TestManySinks:
    """Pipeline routing to many named sink destinations."""

    def test_10_named_sinks(self) -> None:
        """Route rows to 10 different sinks via gate chain."""
        config, sinks, _rows = _build_many_sinks_pipeline(num_sinks=10, rows_per_sink=5)

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
        assert result.rows_processed == 50  # 10 sinks * 5 rows each

        # Verify each sink received its rows
        for i in range(10):
            sink = sinks[f"sink_{i}"]
            assert len(sink.results) == 5, f"sink_{i} expected 5 rows, got {len(sink.results)}"

        # Default sink should have no rows (all matched gates)
        assert len(sinks["default"].results) == 0

    def test_50_named_sinks(self) -> None:
        """Route rows to 50 different sinks."""
        config, sinks, _rows = _build_many_sinks_pipeline(num_sinks=50, rows_per_sink=2)

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
        assert result.rows_processed == 100  # 50 sinks * 2 rows each

        # Verify each sink received its rows
        for i in range(50):
            sink = sinks[f"sink_{i}"]
            assert len(sink.results) == 2, f"sink_{i} expected 2 rows, got {len(sink.results)}"

        assert len(sinks["default"].results) == 0

    def test_100_named_sinks(self) -> None:
        """Route rows to 100 different sinks."""
        config, sinks, _rows = _build_many_sinks_pipeline(num_sinks=100, rows_per_sink=1)

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
        assert result.rows_processed == 100  # 100 sinks * 1 row each

        # Verify each sink received exactly one row
        for i in range(100):
            sink = sinks[f"sink_{i}"]
            assert len(sink.results) == 1, f"sink_{i} expected 1 row, got {len(sink.results)}"

        assert len(sinks["default"].results) == 0
