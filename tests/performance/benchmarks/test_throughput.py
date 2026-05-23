"""Pipeline throughput benchmarks.

Measures rows/second for realistic pipeline configurations using
the full Orchestrator.run() code path with in-memory LandscapeDB.
"""

from __future__ import annotations

from typing import Any

import pytest

from elspeth.core.landscape.database import LandscapeDB
from elspeth.engine.orchestrator import Orchestrator, PipelineConfig
from tests.fixtures.base_classes import as_sink, as_source, as_transform
from tests.fixtures.pipeline import build_production_graph
from tests.fixtures.plugins import CollectSink, ListSource, PassTransform
from tests.fixtures.stores import MockPayloadStore
from tests.performance.conftest import benchmark_timer


def _make_rows(n: int) -> list[dict[str, Any]]:
    """Generate N simple rows for throughput testing."""
    return [{"id": i, "value": f"row_{i}", "amount": i * 1.5} for i in range(n)]


def _run_pipeline(
    source_data: list[dict[str, Any]],
    transforms: list[Any],
    sink: CollectSink,
    *,
    sink_name: str = "default",
) -> None:
    """Run a linear pipeline through the full Orchestrator path."""
    db = LandscapeDB.in_memory()
    payload_store = MockPayloadStore()

    source = ListSource(source_data, on_success=sink_name)
    sinks = {sink_name: as_sink(sink)}

    # Set on_success on terminal transform if any
    if transforms:
        transforms[-1].on_success = sink_name

    config = PipelineConfig(
        sources={"primary": as_source(source)},
        transforms=[as_transform(t) for t in transforms],
        sinks=sinks,
    )

    graph = build_production_graph(config)
    orchestrator = Orchestrator(db)
    orchestrator.run(config, graph=graph, payload_store=payload_store)


@pytest.mark.performance
def test_throughput_100_rows_passthrough() -> None:
    """Throughput: 100 rows through ListSource -> PassTransform -> CollectSink.

    Establishes baseline for minimal pipeline overhead per row.
    """
    rows = _make_rows(100)
    sink = CollectSink()
    transform = PassTransform()

    with benchmark_timer() as timing:
        _run_pipeline(rows, [transform], sink)

    assert len(sink.results) == 100
    rows_per_sec = 100 / timing.wall_seconds

    # Baseline: 100 rows should process in < 5s (>20 rows/sec)
    # In-memory DB so overhead is mainly Landscape recording
    assert timing.wall_seconds < 5.0, (
        f"100-row pipeline took {timing.wall_seconds:.2f}s ({rows_per_sec:.0f} rows/sec, expected > 20 rows/sec)"
    )


@pytest.mark.performance
def test_throughput_1000_rows_passthrough() -> None:
    """Throughput: 1000 rows through ListSource -> PassTransform -> CollectSink.

    Tests scaling behavior with higher row counts.
    """
    rows = _make_rows(1000)
    sink = CollectSink()
    transform = PassTransform()

    with benchmark_timer() as timing:
        _run_pipeline(rows, [transform], sink)

    assert len(sink.results) == 1000
    rows_per_sec = 1000 / timing.wall_seconds

    # Baseline: 1000 rows should process in < 30s (>33 rows/sec)
    assert timing.wall_seconds < 30.0, (
        f"1000-row pipeline took {timing.wall_seconds:.2f}s ({rows_per_sec:.0f} rows/sec, expected > 33 rows/sec)"
    )


@pytest.mark.performance
def test_throughput_with_multiple_transforms() -> None:
    """Throughput: 100 rows through 5 PassTransform stages.

    Measures per-transform overhead when chaining multiple transforms.
    """
    rows = _make_rows(100)
    sink = CollectSink()
    transforms = [PassTransform() for _ in range(5)]

    with benchmark_timer() as timing:
        _run_pipeline(rows, transforms, sink)

    assert len(sink.results) == 100
    rows_per_sec = 100 / timing.wall_seconds

    # Baseline: 100 rows through 5 transforms should process in < 10s
    # Each transform adds node state recording overhead
    assert timing.wall_seconds < 10.0, (
        f"100-row 5-transform pipeline took {timing.wall_seconds:.2f}s ({rows_per_sec:.0f} rows/sec, expected > 10 rows/sec)"
    )


@pytest.mark.performance
def test_throughput_fork_pipeline() -> None:
    """Throughput: 100 rows through a gate that routes to 2 sinks.

    Measures overhead of gate evaluation and multi-sink routing.
    Uses config-driven gates for production-path fidelity.
    """
    from elspeth.core.config import GateSettings

    # Half the rows route to "high" sink, half continue to "default"
    rows = [{"id": i, "value": i, "amount": i * 1.5} for i in range(100)]

    default_sink = CollectSink(name="default")
    high_sink = CollectSink(name="high")

    threshold_gate = GateSettings(
        name="threshold",
        input="gate_in",
        condition="row['value'] >= 50",
        routes={"true": "high", "false": "default"},
    )

    db = LandscapeDB.in_memory()
    payload_store = MockPayloadStore()

    source = ListSource(rows, on_success="gate_in")
    config = PipelineConfig(
        sources={"primary": as_source(source)},
        transforms=[],
        sinks={"default": as_sink(default_sink), "high": as_sink(high_sink)},
        gates=[threshold_gate],
    )

    graph = build_production_graph(config)

    with benchmark_timer() as timing:
        orchestrator = Orchestrator(db)
        orchestrator.run(config, graph=graph, payload_store=payload_store)

    total_routed = len(default_sink.results) + len(high_sink.results)
    assert total_routed == 100
    rows_per_sec = 100 / timing.wall_seconds

    # Baseline: 100 rows with gate routing should process in < 10s
    assert timing.wall_seconds < 10.0, (
        f"100-row fork pipeline took {timing.wall_seconds:.2f}s ({rows_per_sec:.0f} rows/sec, expected > 10 rows/sec)"
    )
