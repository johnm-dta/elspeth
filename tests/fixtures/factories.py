# tests/fixtures/factories.py
"""Test-only factories and re-exports from elspeth.testing.

Layer 1 (elspeth.testing): Production-type factories — no mocks, no fakes.
Layer 2 (this file):        Test infrastructure — mocks, graph builders, DB population.

Usage:
    from tests.fixtures.factories import make_row, make_context, make_graph_linear
    # make_row comes from elspeth.testing (re-exported)
    # make_context and make_graph_linear are test-only (defined here)
"""

from __future__ import annotations

from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any, Literal
from unittest.mock import Mock

from elspeth.contracts.coalesce_enums import CoalescePolicy, MergeStrategy
from elspeth.contracts.coalesce_metadata import CoalesceMetadata
from elspeth.contracts.node_state_context import (
    PoolConfigSnapshot,
    PoolExecutionContext,
    PoolStatsSnapshot,
    QueryOrderEntry,
)

if TYPE_CHECKING:
    from elspeth.contracts import TransformProtocol
    from elspeth.contracts.plugin_context import PluginContext
    from elspeth.core.dag import ExecutionGraph, WiredTransform

__all__ = [
    "make_artifact",
    "make_batch_checkpoint",
    "make_coalesce_metadata",
    "make_context",
    "make_contract",
    "make_contract_audit_record",
    "make_error",
    "make_error_reason",
    "make_exception_result",
    "make_execution_counters",
    "make_external_call_completed",
    "make_failure_info",
    "make_field",
    "make_flush_result",
    "make_gate_continue",
    "make_gate_fork",
    "make_gate_route",
    "make_graph_fork",
    "make_graph_linear",
    "make_operation_context",
    "make_phase_completed",
    "make_phase_started",
    "make_pipeline_config",
    "make_pipeline_row",
    "make_pool_execution_context",
    "make_row",
    "make_row_result",
    "make_run_result",
    "make_run_summary",
    "make_source_context",
    "make_source_row",
    "make_source_row_quarantined",
    "make_success",
    "make_success_multi",
    "make_success_reason",
    "make_token_completed",
    "make_token_info",
    "make_transform_completed",
    "make_transform_error_token",
    "make_validation_error_token",
    "wire_transforms",
]

# --- Re-export all production factories for single-import convenience ---
from elspeth.testing import (
    make_artifact,
    make_contract,
    make_contract_audit_record,
    make_error,
    make_error_reason,
    make_exception_result,
    make_execution_counters,
    make_external_call_completed,
    make_failure_info,
    make_field,
    make_flush_result,
    make_gate_continue,
    make_gate_fork,
    make_gate_route,
    make_phase_completed,
    make_phase_started,
    make_pipeline_config,
    make_pipeline_row,
    make_row,
    make_row_result,
    make_run_result,
    make_run_summary,
    make_source_row,
    make_source_row_quarantined,
    make_success,
    make_success_multi,
    make_success_reason,
    make_token_completed,
    make_token_info,
    make_transform_completed,
    make_transform_error_token,
    make_validation_error_token,
)

# =============================================================================
# PluginContext — Uses Mock(), so test-only
# =============================================================================


def make_context(
    *,
    run_id: str = "test-run",
    state_id: str = "state-123",
    token: Any | None = None,
    config: dict[str, Any] | None = None,
    landscape: Any | None = None,
    node_id: str | None = None,
) -> PluginContext:
    """Build a PluginContext with sensible test defaults.

    Usage:
        ctx = make_context()                            # Minimal (mock landscape)
        ctx = make_context(state_id="state-retry-3")    # Custom state_id
        ctx = make_context(landscape=recorder)           # Real landscape recorder
        ctx = make_context(node_id="source")            # With explicit node_id
    """
    from elspeth.contracts.plugin_context import PluginContext

    if token is None:
        token = make_token_info()

    if landscape is None:
        from elspeth.contracts.audit_protocols import PluginAuditWriter

        landscape = Mock(spec=PluginAuditWriter)
        # Configure get_node_state() to return a mock with matching token_id
        # so that PluginContext.record_call() token consistency checks pass.
        node_state_mock = Mock()
        node_state_mock.token_id = token.token_id
        landscape.get_node_state.return_value = node_state_mock

    return PluginContext(
        run_id=run_id,
        landscape=landscape,
        state_id=state_id,
        config=config or {},
        token=token,
        node_id=node_id,
    )


def make_source_context(
    *,
    run_id: str = "test-run",
    node_id: str = "source",
    plugin_name: str = "csv",
) -> PluginContext:
    """Build a PluginContext with run and node records for validation error recording.

    Internally delegates to make_recorder_with_run() for DB/recorder/run/node setup.

    For testing source plugins that call ctx.record_validation_error().
    Creates the FK chain: run → node → PluginContext.

    Use make_operation_context() instead when the plugin also makes external
    calls and records them via ctx.record_call().

    Usage:
        ctx = make_source_context()                         # CSV source default
        ctx = make_source_context(plugin_name="json")       # JSON source
    """
    from elspeth.contracts.plugin_context import PluginContext
    from tests.fixtures.landscape import make_recorder_with_run

    setup = make_recorder_with_run(
        run_id=run_id,
        source_node_id=node_id,
        source_plugin_name=plugin_name,
    )
    return PluginContext(
        run_id=setup.run_id,
        node_id=setup.source_node_id,
        config={},
        landscape=setup.factory.plugin_audit_writer(),
    )


def make_operation_context(
    *,
    run_id: str = "test-run",
    node_id: str = "source",
    plugin_name: str = "azure_blob",
    node_type: str = "SOURCE",
    operation_type: Literal["source_load", "sink_write"] = "source_load",
) -> PluginContext:
    """Build a PluginContext with real landscape and operation records.

    Internally delegates to make_recorder_with_run() for DB/recorder/run setup.

    For testing source/sink plugins that call ctx.record_call().
    Creates the full FK chain: run → node → operation → PluginContext.

    Use this instead of make_context() when the plugin under test makes
    external calls and records them via ctx.record_call().

    Usage:
        ctx = make_operation_context()                                  # Source default
        ctx = make_operation_context(operation_type="sink_write",       # Sink context
                                     node_id="sink", node_type="SINK")
    """
    from elspeth.contracts import NodeType
    from elspeth.contracts.plugin_context import PluginContext
    from tests.fixtures.landscape import make_recorder_with_run, register_test_node

    if node_type == "SOURCE":
        # Source node: delegate directly — make_recorder_with_run creates a SOURCE node
        setup = make_recorder_with_run(
            run_id=run_id,
            source_node_id=node_id,
            source_plugin_name=plugin_name,
        )
        actual_node_id = setup.source_node_id
    else:
        # Non-source node (SINK, TRANSFORM, etc.): create throwaway source,
        # then register the actual node type needed
        setup = make_recorder_with_run(run_id=run_id)
        register_test_node(
            setup.data_flow,
            setup.run_id,
            node_id,
            node_type=NodeType[node_type],
            plugin_name=plugin_name,
        )
        actual_node_id = node_id

    op = setup.execution.begin_operation(setup.run_id, actual_node_id, operation_type)
    return PluginContext(
        run_id=setup.run_id,
        node_id=actual_node_id,
        config={},
        landscape=setup.factory.plugin_audit_writer(),
        operation_id=op.operation_id,
    )


# =============================================================================
# ExecutionGraph — Manual construction for unit tests ONLY
# =============================================================================
#
# TIER RULES (BUG-LINEAGE-01 prevention):
#
#   unit/         → make_graph_linear(), make_graph_fork() are OK.
#                   These test graph algorithms in isolation (cycle detection,
#                   topo sort, visualization) where fake plugin names are fine.
#
#   property/     → make_graph_linear(), make_graph_fork() are OK.
#                   Property tests verify graph invariants (acyclicity, single
#                   source) and don't need real plugin wiring.
#
#   integration/  → MUST use ExecutionGraph.from_plugin_instances().
#   e2e/            These tiers test the real pipeline assembly path. Manual
#   performance/    construction would hide mapping bugs (BUG-LINEAGE-01).
#
# If you're writing an integration test and tempted to use make_graph_linear(),
# that's a sign your test setup should go through the full plugin instantiation
# path instead. See fixtures/pipeline.py for helpers that do this correctly.
# =============================================================================


def make_graph_linear(
    *node_names: str,
    source_plugin: str = "test-source",
    sink_plugin: str = "test-sink",
    transform_plugin: str = "test-transform",
) -> ExecutionGraph:
    """Build a linear ExecutionGraph: source -> t1 -> t2 -> ... -> sink.

    WARNING: For unit/property tests only. Integration+ tests MUST use
    ExecutionGraph.from_plugin_instances() to exercise the real assembly path.

    Usage:
        graph = make_graph_linear()                        # source -> sink
        graph = make_graph_linear("enrich", "classify")    # source -> t1 -> t2 -> sink
    """
    from elspeth.contracts.enums import NodeType
    from elspeth.core.dag import ExecutionGraph

    graph = ExecutionGraph()
    source = "source-node"
    sink = "sink-node"

    graph.add_node(source, node_type=NodeType.SOURCE, plugin_name=source_plugin, config={})

    prev = source
    for name in node_names:
        graph.add_node(name, node_type=NodeType.TRANSFORM, plugin_name=transform_plugin, config={})
        graph.add_edge(prev, name, label="continue")
        prev = name

    graph.add_node(sink, node_type=NodeType.SINK, plugin_name=sink_plugin, config={})
    graph.add_edge(prev, sink, label="continue")

    return graph


def make_graph_fork(
    branches: dict[str, list[str]],
    *,
    gate_name: str = "gate-node",
    coalesce_name: str = "coalesce-node",
) -> ExecutionGraph:
    """Build a fork/join ExecutionGraph.

    WARNING: For unit/property tests only. Integration+ tests MUST use
    ExecutionGraph.from_plugin_instances() to exercise the real assembly path.

    Usage:
        graph = make_graph_fork({
            "path_a": ["transform_a1", "transform_a2"],
            "path_b": ["transform_b1"],
        })
    """
    from elspeth.contracts.enums import NodeType
    from elspeth.core.dag import ExecutionGraph

    graph = ExecutionGraph()
    source = "source-node"
    sink = "sink-node"

    graph.add_node(source, node_type=NodeType.SOURCE, plugin_name="test-source", config={})
    graph.add_node(gate_name, node_type=NodeType.GATE, plugin_name="test-gate", config={})
    graph.add_edge(source, gate_name, label="continue")

    # Add coalesce node once (before the loop to avoid duplicate adds)
    graph.add_node(coalesce_name, node_type=NodeType.COALESCE, plugin_name="coalesce", config={})

    for branch_label, transforms in branches.items():
        prev = gate_name
        for t_name in transforms:
            graph.add_node(t_name, node_type=NodeType.TRANSFORM, plugin_name="test-transform", config={})
            graph.add_edge(prev, t_name, label=branch_label if prev == gate_name else "continue")
            prev = t_name
        graph.add_edge(prev, coalesce_name, label="continue")

    graph.add_node(sink, node_type=NodeType.SINK, plugin_name="test-sink", config={})
    graph.add_edge(coalesce_name, sink, label="continue")

    return graph


def _set_transform_routing(
    transform: Any,
    *,
    on_success: str | None,
    on_error: str | None,
) -> None:
    """Set routing fields on test transforms."""
    transform.on_success = on_success
    transform.on_error = on_error


def wire_transforms(
    transforms: list[TransformProtocol],
    *,
    source_connection: str = "source_out",
    final_sink: str = "output",
    names: list[str] | None = None,
) -> list[WiredTransform]:
    """Create WiredTransform entries with deterministic sequential wiring.

    This is a test helper that mirrors production config-driven routing:
    source_connection -> t0 -> t1 -> ... -> tN -> final_sink.
    """
    from elspeth.core.config import TransformSettings
    from elspeth.core.dag import WiredTransform

    if names is not None and len(names) != len(transforms):
        raise ValueError(f"names length ({len(names)}) must match transforms length ({len(transforms)})")

    wired: list[WiredTransform] = []
    total = len(transforms)
    for index, transform in enumerate(transforms):
        input_connection = source_connection if index == 0 else f"conn_{index - 1}_{index}"
        on_success = final_sink if index == total - 1 else f"conn_{index}_{index + 1}"
        node_name = names[index] if names is not None else f"{transform.name}_{index}"
        on_error = getattr(transform, "on_error", None) or "discard"

        settings = TransformSettings(
            name=node_name,
            plugin=transform.name,
            input=input_connection,
            on_success=on_success,
            on_error=on_error,
            options={},
        )
        _set_transform_routing(transform, on_success=on_success, on_error=on_error)
        wired.append(WiredTransform(plugin=transform, settings=settings))

    return wired


# =============================================================================
# Run/Landscape Setup — Eliminate begin_run()/complete_run() boilerplate
# =============================================================================


# =============================================================================
# Coalesce / Batch Checkpoint — Structural dict factories
# =============================================================================


def make_coalesce_metadata(
    *,
    policy: CoalescePolicy = CoalescePolicy.REQUIRE_ALL,
    merge_strategy: MergeStrategy = MergeStrategy.UNION,
    expected_branches: list[str] | None = None,
    branches_arrived: list[str] | None = None,
    wait_duration_ms: float = 150.0,
) -> CoalesceMetadata:
    """Build a CoalesceMetadata instance for test convenience.

    Returns the typed dataclass directly (not a dict wrapper).
    """
    from elspeth.contracts.coalesce_metadata import ArrivalOrderEntry

    branches = expected_branches or ["a", "b"]
    arrived = branches_arrived or branches
    return CoalesceMetadata.for_merge(
        policy=policy,
        merge_strategy=merge_strategy,
        expected_branches=branches,
        branches_arrived=arrived,
        branches_lost={},
        arrival_order=[ArrivalOrderEntry(branch=b, arrival_offset_ms=float(i * 50)) for i, b in enumerate(arrived)],
        wait_duration_ms=wait_duration_ms,
    )


def make_pool_execution_context(
    *,
    pool_size: int = 4,
    num_queries: int = 2,
) -> PoolExecutionContext:
    """Build a PoolExecutionContext instance for test convenience."""
    return PoolExecutionContext(
        pool_config=PoolConfigSnapshot(
            pool_size=pool_size,
            max_capacity_retry_seconds=30.0,
            dispatch_delay_at_completion_ms=10.0,
        ),
        pool_stats=PoolStatsSnapshot(
            capacity_retries=0,
            successes=num_queries,
            peak_delay_ms=15.0,
            current_delay_ms=10.0,
            total_throttle_time_ms=0.0,
            max_concurrent_reached=min(num_queries, pool_size),
        ),
        query_ordering=tuple(
            QueryOrderEntry(
                submit_index=i,
                complete_index=i,
                buffer_wait_ms=0.0,
            )
            for i in range(num_queries)
        ),
    )


def make_batch_checkpoint(
    *,
    batch_id: str = "batch-123",
    row_count: int = 3,
) -> dict[str, Any]:
    """Build a batch checkpoint dict.

    Matches the implicit schema in azure_batch.py:160-180.
    """
    return {
        "batch_id": batch_id,
        "submitted_at": datetime.now(UTC).isoformat(),
        "row_mapping": {f"custom_{i}": {"id": i, "text": f"row-{i}"} for i in range(row_count)},
        "template_errors": [],
        "requests": {f"custom_{i}": {"model": "gpt-4", "messages": []} for i in range(row_count)},
    }
