"""Landscape DB write operation benchmarks.

Measures latency and throughput of core audit recording operations
using in-memory SQLite. These benchmarks establish baselines for
the overhead that Landscape recording adds to pipeline execution.
"""

from __future__ import annotations

import pytest

from elspeth.contracts import NodeType
from elspeth.contracts.audit import TokenRef
from elspeth.contracts.enums import TerminalOutcome, TerminalPath
from elspeth.contracts.schema import SchemaConfig
from elspeth.core.canonical import CANONICAL_VERSION
from elspeth.core.landscape.database import LandscapeDB
from elspeth.core.landscape.factory import RecorderFactory
from tests.performance.conftest import benchmark_timer


def _make_factory() -> tuple[LandscapeDB, RecorderFactory]:
    """Create a fresh in-memory DB and factory."""
    db = LandscapeDB.in_memory()
    factory = RecorderFactory(db)
    return db, factory


def _begin_run(factory: RecorderFactory) -> str:
    """Begin a run and return its run_id."""
    run = factory.run_lifecycle.begin_run(
        config={"source": "test", "transforms": []},
        canonical_version=CANONICAL_VERSION,
    )
    return run.run_id


def _register_source_node(factory: RecorderFactory, run_id: str) -> str:
    """Register a source node and return its node_id."""
    node = factory.data_flow.register_node(
        run_id=run_id,
        plugin_name="list_source",
        node_type=NodeType.SOURCE,
        plugin_version="1.0.0",
        config={"schema": {"mode": "observed"}},
        schema_config=SchemaConfig(mode="observed", fields=None),
    )
    return node.node_id


def _register_transform_node(factory: RecorderFactory, run_id: str) -> str:
    """Register a transform node and return its node_id."""
    node = factory.data_flow.register_node(
        run_id=run_id,
        plugin_name="passthrough",
        node_type=NodeType.TRANSFORM,
        plugin_version="1.0.0",
        config={"schema": {"mode": "observed"}},
        schema_config=SchemaConfig(mode="observed", fields=None),
    )
    return node.node_id


@pytest.mark.performance
def test_begin_run_latency() -> None:
    """Time to begin a run (single operation latency).

    Measures the overhead of creating a run record including
    canonical JSON serialization and config hashing.
    """
    iterations = 100

    with benchmark_timer() as timing:
        for _ in range(iterations):
            db = LandscapeDB.in_memory()
            factory = RecorderFactory(db)
            factory.run_lifecycle.begin_run(
                config={"source": "csv", "transforms": ["passthrough"]},
                canonical_version=CANONICAL_VERSION,
            )

    ms_per_run = (timing.wall_seconds / iterations) * 1000

    # Baseline: begin_run should complete in < 50ms each
    # Includes DB creation + table setup + run record insert
    assert ms_per_run < 50, f"begin_run latency: {ms_per_run:.2f}ms (expected < 50ms)"


@pytest.mark.performance
def test_create_row_throughput() -> None:
    """Create 1000 rows, measure rows/sec.

    Measures the overhead of source row recording including
    data hashing and database inserts.
    """
    _, factory = _make_factory()
    run_id = _begin_run(factory)
    source_node_id = _register_source_node(factory, run_id)
    iterations = 1000

    with benchmark_timer() as timing:
        for i in range(iterations):
            factory.data_flow.create_row(
                run_id=run_id,
                source_node_id=source_node_id,
                row_index=i,
                data={"id": i, "value": f"row_{i}", "amount": i * 1.5},
                source_row_index=i,
                ingest_sequence=i,
            )

    rows_per_sec = iterations / timing.wall_seconds

    # Baseline: row creation should achieve > 500 rows/sec
    # Each row involves canonical hashing + DB insert
    assert rows_per_sec > 500, f"Row creation: {rows_per_sec:.0f} rows/sec (expected > 500)"


@pytest.mark.performance
def test_create_token_throughput() -> None:
    """Create 1000 tokens, measure tokens/sec.

    Measures token creation overhead. Each token links to a row
    and represents an instance in a specific DAG path.
    """
    _, factory = _make_factory()
    run_id = _begin_run(factory)
    source_node_id = _register_source_node(factory, run_id)

    # Pre-create rows that tokens will reference
    row_ids = []
    for i in range(1000):
        row = factory.data_flow.create_row(
            run_id=run_id,
            source_node_id=source_node_id,
            row_index=i,
            data={"id": i},
            source_row_index=i,
            ingest_sequence=i,
        )
        row_ids.append(row.row_id)

    iterations = 1000

    with benchmark_timer() as timing:
        for i in range(iterations):
            factory.data_flow.create_token(row_id=row_ids[i])

    tokens_per_sec = iterations / timing.wall_seconds

    # Baseline: token creation should achieve > 1000 tokens/sec
    # Simpler than row creation (no data hashing)
    assert tokens_per_sec > 1000, f"Token creation: {tokens_per_sec:.0f} tokens/sec (expected > 1000)"


@pytest.mark.performance
def test_begin_node_state_throughput() -> None:
    """Create 1000 node states, measure states/sec.

    Measures the overhead of recording node processing entries.
    Each node state captures input hash, timestamp, and step info.
    """
    _, factory = _make_factory()
    run_id = _begin_run(factory)
    source_node_id = _register_source_node(factory, run_id)
    transform_node_id = _register_transform_node(factory, run_id)

    # Pre-create rows and tokens
    token_ids = []
    for i in range(1000):
        row = factory.data_flow.create_row(
            run_id=run_id,
            source_node_id=source_node_id,
            row_index=i,
            data={"id": i},
            source_row_index=i,
            ingest_sequence=i,
        )
        token = factory.data_flow.create_token(row_id=row.row_id)
        token_ids.append(token.token_id)

    iterations = 1000

    with benchmark_timer() as timing:
        for i in range(iterations):
            factory.execution.begin_node_state(
                token_id=token_ids[i],
                node_id=transform_node_id,
                run_id=run_id,
                step_index=0,
                input_data={"id": i, "value": f"row_{i}"},
            )

    states_per_sec = iterations / timing.wall_seconds

    # Baseline: node state creation should achieve > 500 states/sec
    # Each state involves input data hashing + DB insert
    assert states_per_sec > 500, f"Node state creation: {states_per_sec:.0f} states/sec (expected > 500)"


@pytest.mark.performance
def test_record_outcome_throughput() -> None:
    """Record 1000 token outcomes, measure outcomes/sec.

    Measures the overhead of recording terminal row states.
    Each outcome records the final disposition of a token.
    """
    _, factory = _make_factory()
    run_id = _begin_run(factory)
    source_node_id = _register_source_node(factory, run_id)

    # Pre-create rows and tokens
    token_ids = []
    for i in range(1000):
        row = factory.data_flow.create_row(
            run_id=run_id,
            source_node_id=source_node_id,
            row_index=i,
            data={"id": i},
            source_row_index=i,
            ingest_sequence=i,
        )
        token = factory.data_flow.create_token(row_id=row.row_id)
        token_ids.append(token.token_id)

    iterations = 1000

    with benchmark_timer() as timing:
        for i in range(iterations):
            factory.data_flow.record_token_outcome(
                ref=TokenRef(token_id=token_ids[i], run_id=run_id),
                outcome=TerminalOutcome.SUCCESS,
                path=TerminalPath.DEFAULT_FLOW,
                sink_name="default",
            )

    outcomes_per_sec = iterations / timing.wall_seconds

    # Baseline: outcome recording should achieve > 1000 outcomes/sec
    assert outcomes_per_sec > 1000, f"Outcome recording: {outcomes_per_sec:.0f} outcomes/sec (expected > 1000)"
