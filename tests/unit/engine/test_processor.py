# tests/unit/engine/test_processor.py
"""Unit tests for RowProcessor — the DAG execution state machine.

processor.py orchestrates row processing through transforms, gates, and
aggregation. It is the largest file in the engine (~2,000 lines) and the
most critical path for correctness.

Test strategy:
- Use LandscapeDB.in_memory() for real audit recording (no mock recorder)
- Use real SpanFactory (no tracer — no-op spans)
- Mock transforms/gates to control routing outcomes
- Verify outcomes via RowResult assertions

This avoids the anti-pattern of testing mocks instead of behavior.
"""

from __future__ import annotations

from typing import Any
from unittest.mock import Mock, patch

import pytest

# For node registration
from elspeth.contracts import NodeType, RouteDestination, RowOutcome, SourceRow, TokenInfo, TransformResult
from elspeth.contracts.enums import (
    NodeStateStatus,
    RoutingKind,
    TriggerType,
)
from elspeth.contracts.errors import OrchestrationInvariantError
from elspeth.contracts.plugin_context import PluginContext
from elspeth.contracts.results import GateResult
from elspeth.contracts.routing import RoutingAction
from elspeth.contracts.schema import SchemaConfig
from elspeth.contracts.schema_contract import SchemaContract
from elspeth.contracts.types import BranchName, CoalesceName, GateName, NodeID, SinkName
from elspeth.core.config import AggregationSettings, GateSettings
from elspeth.core.landscape import LandscapeDB, LandscapeRecorder
from elspeth.engine.dag_navigator import WorkItem
from elspeth.engine.executors import GateOutcome
from elspeth.engine.processor import (
    MAX_WORK_QUEUE_ITERATIONS,
    DAGTraversalContext,
    RowProcessor,
)
from elspeth.engine.retry import MaxRetriesExceeded, RetryManager
from elspeth.engine.spans import SpanFactory
from elspeth.plugins.clients.llm import LLMClientError
from elspeth.plugins.pooling import CapacityError
from elspeth.plugins.protocols import TransformProtocol
from elspeth.testing import make_contract, make_row, make_source_row, make_token_info

# =============================================================================
# Helpers
# =============================================================================

_DYNAMIC_SCHEMA = SchemaConfig.from_dict({"mode": "observed"})


def _make_contract() -> SchemaContract:
    """Create a minimal schema contract for testing."""
    return make_contract(fields={"value": int}, mode="OBSERVED")


def _make_source_row(data: dict[str, Any] | None = None) -> SourceRow:
    """Create a valid SourceRow with contract."""
    contract = _make_contract()
    return make_source_row(data or {"value": 42}, contract=contract)


def _make_recorder(
    *,
    run_id: str = "test-run",
    source_node_id: str = "source-0",
) -> tuple[LandscapeDB, LandscapeRecorder]:
    """Create an in-memory LandscapeDB with run and source node registered.

    This satisfies FK constraints: rows table references nodes(node_id, run_id).
    """
    db = LandscapeDB.in_memory()
    recorder = LandscapeRecorder(db)
    recorder.begin_run(config={}, canonical_version="v1", run_id=run_id)
    recorder.register_node(
        run_id=run_id,
        plugin_name="test-source",
        node_type=NodeType.SOURCE,
        plugin_version="1.0",
        config={},
        node_id=source_node_id,
        schema_config=_DYNAMIC_SCHEMA,
    )
    return db, recorder


def _make_processor(
    recorder: LandscapeRecorder,
    *,
    run_id: str = "test-run",
    source_node_id: str = "source-0",
    source_on_success: str = "default",
    edge_map: dict[tuple[NodeID, str], str] | None = None,
    route_resolution_map: dict[tuple[NodeID, str], RouteDestination] | None = None,
    config_gates: list[GateSettings] | None = None,
    config_gate_id_map: dict[GateName, NodeID] | None = None,
    aggregation_settings: dict[NodeID, AggregationSettings] | None = None,
    retry_manager: RetryManager | None = None,
    coalesce_executor: Any = None,
    coalesce_node_ids: dict[CoalesceName, NodeID] | None = None,
    branch_to_coalesce: dict[BranchName, CoalesceName] | None = None,
    branch_to_sink: dict[BranchName, str] | None = None,
    node_step_map: dict[NodeID, int] | None = None,
    coalesce_on_success_map: dict[CoalesceName, str] | None = None,
    node_to_next: dict[NodeID, NodeID | None] | None = None,
    first_transform_node_id: NodeID | None = None,
    node_to_plugin: dict[NodeID, Any] | None = None,
    restored_aggregation_state: dict[NodeID, dict[str, Any]] | None = None,
    telemetry_manager: Any = None,
    sink_names: frozenset[str] | None = None,
) -> RowProcessor:
    """Create a RowProcessor with sensible defaults."""
    coalesce_nodes = dict(coalesce_node_ids or {})
    traversal_steps = dict(node_step_map or {})
    source_node = NodeID(source_node_id)
    traversal_steps.setdefault(source_node, 0)
    for idx, coalesce_node in enumerate(coalesce_nodes.values(), start=1):
        traversal_steps.setdefault(coalesce_node, idx)

    traversal_node_to_plugin = (
        dict(node_to_plugin)
        if node_to_plugin is not None
        else {
            config_gate_id_map[GateName(gate.name)]: gate
            for gate in (config_gates or [])
            if config_gate_id_map and GateName(gate.name) in config_gate_id_map
        }
    )
    traversal_next = dict(node_to_next or {})
    traversal_next.setdefault(source_node, None)
    for coalesce_node in coalesce_nodes.values():
        traversal_next.setdefault(coalesce_node, None)

    traversal = DAGTraversalContext(
        node_step_map=traversal_steps,
        node_to_plugin=traversal_node_to_plugin,
        first_transform_node_id=first_transform_node_id,
        node_to_next=traversal_next,
        coalesce_node_map=coalesce_nodes,
    )

    return RowProcessor(
        recorder=recorder,
        span_factory=SpanFactory(),  # No tracer — no-op spans
        run_id=run_id,
        source_node_id=NodeID(source_node_id),
        source_on_success=source_on_success,
        traversal=traversal,
        edge_map=edge_map,
        route_resolution_map=route_resolution_map,
        aggregation_settings=aggregation_settings,
        retry_manager=retry_manager,
        coalesce_executor=coalesce_executor,
        branch_to_coalesce=branch_to_coalesce,
        branch_to_sink={BranchName(k): SinkName(v) for k, v in (branch_to_sink or {}).items()},
        coalesce_on_success_map=coalesce_on_success_map,
        restored_aggregation_state=restored_aggregation_state,
        telemetry_manager=telemetry_manager,
        sink_names=sink_names,
    )


def _make_mock_transform(
    *,
    node_id: str = "transform-1",
    name: str = "test-transform",
    on_error: str | None = "discard",
    on_success: str | None = None,
    is_batch_aware: bool = False,
    creates_tokens: bool = False,
    result: TransformResult | None = None,
) -> Mock:
    """Create a mock transform satisfying TransformProtocol."""
    transform = Mock(spec=TransformProtocol)
    transform.node_id = node_id
    transform.name = name
    transform.on_error = on_error
    transform.on_success = on_success
    transform.is_batch_aware = is_batch_aware
    transform.creates_tokens = creates_tokens
    if result is not None:
        transform.process.return_value = result
    return transform


# =============================================================================
# Constructor: Error Edge Map
# =============================================================================


class TestConstructorErrorEdgeMap:
    """Tests for error edge map construction in __init__."""

    def test_extracts_error_edges_from_edge_map(self) -> None:
        """Error edges (labels like __error_0__) are extracted into error_edge_ids."""
        _, recorder = _make_recorder()
        edge_map = {
            (NodeID("t1"), "continue"): "edge-1",
            (NodeID("t1"), "__error_0__"): "error-edge-1",
            (NodeID("t2"), "continue"): "edge-2",
            (NodeID("t2"), "__error_1__"): "error-edge-2",
        }
        processor = _make_processor(recorder, edge_map=edge_map)
        assert processor._error_edge_ids == {
            NodeID("t1"): "error-edge-1",
            NodeID("t2"): "error-edge-2",
        }

    def test_empty_edge_map_produces_no_error_edges(self) -> None:
        """No edge_map means no error edges."""
        _, recorder = _make_recorder()
        processor = _make_processor(recorder)
        assert processor._error_edge_ids == {}

    def test_non_error_labels_ignored(self) -> None:
        """Only __error_N__ labels are extracted; other labels are ignored."""
        _, recorder = _make_recorder()
        edge_map = {
            (NodeID("t1"), "continue"): "edge-1",
            (NodeID("t1"), "route_to_sink"): "edge-2",
            (NodeID("t1"), "fork_path_a"): "edge-3",
        }
        processor = _make_processor(recorder, edge_map=edge_map)
        assert processor._error_edge_ids == {}

    def test_restores_aggregation_state(self) -> None:
        """Restored aggregation state is passed to AggregationExecutor."""
        _, recorder = _make_recorder()
        # We can't easily verify internal state restoration without exposing
        # internals, but we can verify the constructor doesn't crash with
        # restoration data
        processor = _make_processor(
            recorder,
            aggregation_settings={
                NodeID("agg-1"): AggregationSettings(
                    name="test-agg",
                    plugin="test-plugin",
                    input="agg_in",
                    trigger={"count": 3},
                ),
            },
            restored_aggregation_state={
                NodeID("agg-1"): {"buffer": [], "trigger_state": {}},
            },
        )
        assert processor is not None


class TestTraversalNextNodeInvariants:
    """Tests for strict Tier-1 traversal next-node invariants."""

    def test_resolve_next_node_missing_entry_raises_invariant(self) -> None:
        """Missing traversal next-node entry must crash, not silently return None."""
        _, recorder = _make_recorder()
        processor = _make_processor(
            recorder,
            node_to_next={NodeID("source-0"): None},
        )

        with pytest.raises(OrchestrationInvariantError, match="missing from traversal next-node map"):
            processor._nav.resolve_next_node(NodeID("missing-node"))

    def test_process_row_raises_when_transform_missing_next_node_entry(self) -> None:
        """Processing nodes must have explicit next-node entries (None for terminal)."""
        _db, recorder = _make_recorder()
        source_row = _make_source_row()
        ctx = PluginContext(run_id="test-run", config={})
        transform = _make_mock_transform()
        source_node = NodeID("source-0")
        transform_node = NodeID(transform.node_id)
        processor = _make_processor(
            recorder,
            node_step_map={source_node: 0, transform_node: 1},
            node_to_next={source_node: transform_node},
            first_transform_node_id=transform_node,
            node_to_plugin={transform_node: transform},
        )

        with pytest.raises(OrchestrationInvariantError, match="missing from traversal next-node map"):
            processor.process_row(
                row_index=0,
                source_row=source_row,
                transforms=[transform],
                ctx=ctx,
            )


# =============================================================================
# _resolve_audit_step_for_node invariants
# =============================================================================


class TestAuditStepResolutionInvariants:
    """Tests for strict Tier-1 audit step resolution invariants.

    _resolve_audit_step_for_node has three branches:
    1. node_id in step_map → return mapped step
    2. node_id == source_node_id (not in map) → return 0
    3. unknown node_id → raise OrchestrationInvariantError
    """

    def test_known_node_returns_mapped_step(self) -> None:
        """Nodes in the step map return their assigned step value."""
        _, recorder = _make_recorder()
        transform_node = NodeID("transform-1")
        processor = _make_processor(
            recorder,
            node_step_map={NodeID("source-0"): 0, transform_node: 3},
            node_to_next={NodeID("source-0"): None, transform_node: None},
        )

        assert processor._resolve_audit_step_for_node(transform_node) == 3

    def test_source_node_returns_zero(self) -> None:
        """Source node resolves to step 0 even when not in the step map.

        This is the convention distinguishing source-originated audit records
        from transform-originated ones. The _make_processor helper auto-adds
        the source to the step map, so we construct the processor directly
        to test the explicit fallback branch.
        """
        _, recorder = _make_recorder()
        source_node = NodeID("source-0")

        # Build traversal WITHOUT source in the step map
        traversal = DAGTraversalContext(
            node_step_map={},
            node_to_plugin={},
            first_transform_node_id=None,
            node_to_next={source_node: None},
            coalesce_node_map={},
        )
        processor = RowProcessor(
            recorder=recorder,
            span_factory=SpanFactory(),
            run_id="test-run",
            source_node_id=source_node,
            source_on_success="default",
            traversal=traversal,
        )

        assert processor._resolve_audit_step_for_node(source_node) == 0

    def test_unknown_node_raises_invariant_error(self) -> None:
        """Unknown node IDs must crash, not silently return a default step."""
        _, recorder = _make_recorder()
        processor = _make_processor(
            recorder,
            node_to_next={NodeID("source-0"): None},
        )

        with pytest.raises(OrchestrationInvariantError, match="missing from traversal step map"):
            processor._resolve_audit_step_for_node(NodeID("nonexistent-node"))

    def test_unknown_node_includes_node_id_in_error(self) -> None:
        """Error message includes the offending node ID for debugging."""
        _, recorder = _make_recorder()
        processor = _make_processor(
            recorder,
            node_to_next={NodeID("source-0"): None},
        )

        with pytest.raises(OrchestrationInvariantError, match="phantom-node-42"):
            processor._resolve_audit_step_for_node(NodeID("phantom-node-42"))


# =============================================================================
# _get_gate_destinations
# =============================================================================


class TestGetGateDestinations:
    """Tests for the gate destination extraction helper."""

    def test_routed_to_sink(self) -> None:
        """Gate routed to a named sink returns that sink name."""
        _, recorder = _make_recorder()
        processor = _make_processor(recorder)
        outcome = Mock()
        outcome.sink_name = "error-sink"
        assert processor._get_gate_destinations(outcome) == ("error-sink",)

    def test_fork_to_paths(self) -> None:
        """Fork returns branch names of child tokens."""
        _, recorder = _make_recorder()
        processor = _make_processor(recorder)
        child_a = Mock()
        child_a.branch_name = "path_a"
        child_b = Mock()
        child_b.branch_name = "path_b"
        outcome = Mock()
        outcome.sink_name = None
        outcome.result.action.kind = RoutingKind.FORK_TO_PATHS
        outcome.child_tokens = [child_a, child_b]
        assert processor._get_gate_destinations(outcome) == ("path_a", "path_b")

    def test_continue_routing(self) -> None:
        """Continue routing returns ("continue",)."""
        _, recorder = _make_recorder()
        processor = _make_processor(recorder)
        outcome = Mock()
        outcome.sink_name = None
        outcome.result.action.kind = RoutingKind.CONTINUE
        outcome.next_node_id = None
        assert processor._get_gate_destinations(outcome) == ("continue",)

    def test_route_to_processing_uses_route_label(self) -> None:
        """Route-label branch to processing node reports chosen route label."""
        _, recorder = _make_recorder()
        processor = _make_processor(recorder)
        outcome = Mock()
        outcome.sink_name = None
        outcome.next_node_id = NodeID("transform-2")
        outcome.result.action.kind = RoutingKind.ROUTE
        outcome.result.action.destinations = ("high",)
        assert processor._get_gate_destinations(outcome) == ("high",)


# =============================================================================
# process_row: Linear pipeline (no transforms)
# =============================================================================


class TestProcessRowNoTransforms:
    """Tests for process_row with an empty transform list."""

    def test_empty_pipeline_returns_completed(self) -> None:
        """Row through empty pipeline gets COMPLETED outcome."""
        _, recorder = _make_recorder()

        processor = _make_processor(recorder)
        source_row = _make_source_row()
        ctx = PluginContext(run_id="test-run", config={})

        results = processor.process_row(
            row_index=0,
            source_row=source_row,
            transforms=[],
            ctx=ctx,
        )

        assert len(results) == 1
        assert results[0].outcome == RowOutcome.COMPLETED
        assert results[0].sink_name == "default"
        assert results[0].token.row_data["value"] == 42

    def test_creates_row_and_token_records(self) -> None:
        """process_row creates row record and token in audit trail."""
        _db, recorder = _make_recorder()

        processor = _make_processor(recorder)
        source_row = _make_source_row()
        ctx = PluginContext(run_id="test-run", config={})

        results = processor.process_row(
            row_index=0,
            source_row=source_row,
            transforms=[],
            ctx=ctx,
        )

        # Verify token was created (result has a valid token_id)
        token = results[0].token
        assert token.token_id is not None
        assert token.row_id is not None

    def test_records_source_node_state(self) -> None:
        """process_row records a source node_state with COMPLETED status."""
        db, recorder = _make_recorder()

        processor = _make_processor(recorder)
        source_row = _make_source_row()
        ctx = PluginContext(run_id="test-run", config={})

        processor.process_row(
            row_index=0,
            source_row=source_row,
            transforms=[],
            ctx=ctx,
        )

        # Check that source node_state was recorded
        from sqlalchemy import select

        from elspeth.core.landscape.schema import node_states_table

        with db.connection() as conn:
            states = conn.execute(select(node_states_table).where(node_states_table.c.run_id == "test-run")).fetchall()

        # At minimum, the source node_state should exist
        assert len(states) >= 1
        source_state = states[0]
        assert source_state.status == NodeStateStatus.COMPLETED.value


# =============================================================================
# process_row: Single transform
# =============================================================================


class TestProcessRowSingleTransform:
    """Tests for process_row with a single transform."""

    def _setup(self, transform: Any) -> tuple[LandscapeDB, LandscapeRecorder, RowProcessor]:
        db, recorder = _make_recorder()
        source_node = NodeID("source-0")
        transform_node = NodeID(transform.node_id)

        processor = _make_processor(
            recorder,
            node_step_map={source_node: 0, transform_node: 1},
            node_to_next={source_node: transform_node, transform_node: None},
            first_transform_node_id=transform_node,
            node_to_plugin={transform_node: transform},
        )
        return db, recorder, processor

    def test_successful_transform_returns_completed(self) -> None:
        """Row passes through transform → COMPLETED."""
        transform = _make_mock_transform()
        _db, _recorder, processor = self._setup(transform)
        source_row = _make_source_row({"value": 10})
        ctx = PluginContext(run_id="test-run", config={})

        output_data = make_row({"value": 10, "enriched": True})
        success_result = TransformResult.success(
            output_data,
            success_reason={"action": "test"},
        )

        # side_effect receives the real token and returns it with the desired result
        def executor_side_effect(*, transform, token, ctx, attempt=0):
            return (success_result, token, None)

        with patch.object(
            processor._transform_executor,
            "execute_transform",
            side_effect=executor_side_effect,
        ):
            results = processor.process_row(
                row_index=0,
                source_row=source_row,
                transforms=[transform],
                ctx=ctx,
            )

        assert len(results) == 1
        assert results[0].outcome == RowOutcome.COMPLETED
        assert results[0].sink_name == "default"

    def test_transform_error_with_discard_returns_quarantined(self) -> None:
        """Transform error with on_error='discard' → QUARANTINED."""
        transform = _make_mock_transform(on_error="discard")
        _db, _recorder, processor = self._setup(transform)
        source_row = _make_source_row()
        ctx = PluginContext(run_id="test-run", config={})

        error_result = TransformResult.error(
            {"reason": "test_error"},
            retryable=False,
        )

        def executor_side_effect(*, transform, token, ctx, attempt=0):
            return (error_result, token, "discard")

        with patch.object(
            processor._transform_executor,
            "execute_transform",
            side_effect=executor_side_effect,
        ):
            results = processor.process_row(
                row_index=0,
                source_row=source_row,
                transforms=[transform],
                ctx=ctx,
            )

        assert len(results) == 1
        assert results[0].outcome == RowOutcome.QUARANTINED

    def test_transform_error_with_named_sink_returns_routed(self) -> None:
        """Transform error with on_error='errors' → ROUTED to error sink."""
        transform = _make_mock_transform(on_error="errors")
        _db, _recorder, processor = self._setup(transform)
        source_row = _make_source_row()
        ctx = PluginContext(run_id="test-run", config={})

        error_result = TransformResult.error(
            {"reason": "test_error"},
            retryable=False,
        )

        def executor_side_effect(*, transform, token, ctx, attempt=0):
            return (error_result, token, "errors")

        with patch.object(
            processor._transform_executor,
            "execute_transform",
            side_effect=executor_side_effect,
        ):
            results = processor.process_row(
                row_index=0,
                source_row=source_row,
                transforms=[transform],
                ctx=ctx,
            )

        assert len(results) == 1
        assert results[0].outcome == RowOutcome.ROUTED
        assert results[0].sink_name == "errors"

    def test_max_retries_exceeded_returns_failed(self) -> None:
        """MaxRetriesExceeded → FAILED outcome."""
        transform = _make_mock_transform()
        _db, _recorder, processor = self._setup(transform)
        source_row = _make_source_row()
        ctx = PluginContext(run_id="test-run", config={})

        with patch.object(
            processor._transform_executor,
            "execute_transform",
            side_effect=MaxRetriesExceeded(3, Exception("boom")),
        ):
            results = processor.process_row(
                row_index=0,
                source_row=source_row,
                transforms=[transform],
                ctx=ctx,
            )

        assert len(results) == 1
        assert results[0].outcome == RowOutcome.FAILED
        assert results[0].error is not None


class TestAggregationFailureMatrix:
    """Focused aggregation failure/regression matrix coverage."""

    def _setup_batch_processor(
        self,
        *,
        output_mode: str,
        node_to_next: dict[NodeID, NodeID | None] | None = None,
        transform_on_success: str | None = "agg_sink",
    ) -> tuple[LandscapeDB, LandscapeRecorder, RowProcessor, Mock, NodeID]:
        """Create a RowProcessor configured for a single batch-aware aggregation node."""
        db, recorder = _make_recorder()
        source_node = NodeID("source-0")
        agg_node = NodeID("agg-1")

        transform = _make_mock_transform(
            node_id=str(agg_node),
            name="agg-transform",
            is_batch_aware=True,
            on_success=transform_on_success,
        )

        traversal_next = (
            dict(node_to_next)
            if node_to_next is not None
            else {
                source_node: agg_node,
                agg_node: None,
            }
        )

        # Ensure source node has an explicit next mapping.
        traversal_next.setdefault(source_node, agg_node)

        processor = _make_processor(
            recorder,
            node_step_map={source_node: 0, agg_node: 1, NodeID("downstream-2"): 2},
            node_to_next=traversal_next,
            first_transform_node_id=agg_node,
            node_to_plugin={agg_node: transform},
            aggregation_settings={
                agg_node: AggregationSettings(
                    name="batch_agg",
                    plugin="agg-transform",
                    input="default",
                    trigger={"count": 1},
                    output_mode=output_mode,
                ),
            },
        )
        return db, recorder, processor, transform, agg_node

    def test_flush_failure_passthrough_records_failed_outcomes(self) -> None:
        """Passthrough flush failure records FAILED terminal outcomes for buffered tokens."""
        _db, recorder, processor, transform, _agg_node = self._setup_batch_processor(output_mode="passthrough")
        source_row = _make_source_row({"value": 10})
        ctx = PluginContext(run_id="test-run", config={})
        captured: dict[str, TokenInfo] = {}

        def buffer_row_side_effect(node_id: NodeID, token: TokenInfo) -> None:
            captured["token"] = token

        def execute_flush_side_effect(*, node_id, transform, ctx, trigger_type):
            return (
                TransformResult.error({"reason": "flush_failed"}, retryable=False),
                [captured["token"]],
                "batch-1",
            )

        with (
            patch.object(processor._aggregation_executor, "buffer_row", side_effect=buffer_row_side_effect),
            patch.object(processor._aggregation_executor, "should_flush", return_value=True),
            patch.object(processor._aggregation_executor, "get_trigger_type", return_value=TriggerType.COUNT),
            patch.object(processor._aggregation_executor, "execute_flush", side_effect=execute_flush_side_effect),
            patch.object(recorder, "record_token_outcome") as record_outcome,
        ):
            results = processor.process_row(
                row_index=0,
                source_row=source_row,
                transforms=[transform],
                ctx=ctx,
            )

        assert len(results) == 1
        assert results[0].outcome == RowOutcome.FAILED
        assert [call.kwargs["outcome"] for call in record_outcome.call_args_list] == [RowOutcome.FAILED]

    def test_flush_failure_transform_keeps_consumed_in_batch_terminal_semantics(self) -> None:
        """Transform-mode flush failure keeps terminal outcome as CONSUMED_IN_BATCH (not FAILED)."""
        _db, recorder, processor, transform, _agg_node = self._setup_batch_processor(output_mode="transform")
        source_row = _make_source_row({"value": 10})
        ctx = PluginContext(run_id="test-run", config={})
        captured: dict[str, TokenInfo] = {}

        def buffer_row_side_effect(node_id: NodeID, token: TokenInfo) -> None:
            captured["token"] = token

        def execute_flush_side_effect(*, node_id, transform, ctx, trigger_type):
            return (
                TransformResult.error({"reason": "flush_failed"}, retryable=False),
                [captured["token"]],
                "batch-1",
            )

        with (
            patch.object(processor._aggregation_executor, "buffer_row", side_effect=buffer_row_side_effect),
            patch.object(processor._aggregation_executor, "should_flush", return_value=True),
            patch.object(processor._aggregation_executor, "get_trigger_type", return_value=TriggerType.COUNT),
            patch.object(processor._aggregation_executor, "execute_flush", side_effect=execute_flush_side_effect),
            patch.object(recorder, "record_token_outcome") as record_outcome,
        ):
            results = processor.process_row(
                row_index=0,
                source_row=source_row,
                transforms=[transform],
                ctx=ctx,
            )

        assert len(results) == 1
        assert results[0].outcome == RowOutcome.FAILED
        outcomes = [call.kwargs["outcome"] for call in record_outcome.call_args_list]
        assert outcomes == [RowOutcome.CONSUMED_IN_BATCH]
        assert record_outcome.call_args_list[0].kwargs["batch_id"] == "batch-1"
        assert RowOutcome.FAILED not in outcomes

    def test_passthrough_success_with_rows_none_raises(self) -> None:
        """Passthrough flush requires rows list; rows=None is an invariant violation."""
        _db, _recorder, processor, transform, _agg_node = self._setup_batch_processor(output_mode="passthrough")
        source_row = _make_source_row({"value": 10})
        ctx = PluginContext(run_id="test-run", config={})
        captured: dict[str, TokenInfo] = {}

        bad_result = Mock()
        bad_result.status = "success"
        bad_result.is_multi_row = True
        bad_result.rows = None

        def buffer_row_side_effect(node_id: NodeID, token: TokenInfo) -> None:
            captured["token"] = token

        def execute_flush_side_effect(*, node_id, transform, ctx, trigger_type):
            return bad_result, [captured["token"]], "batch-1"

        with (
            patch.object(processor._aggregation_executor, "buffer_row", side_effect=buffer_row_side_effect),
            patch.object(processor._aggregation_executor, "should_flush", return_value=True),
            patch.object(processor._aggregation_executor, "get_trigger_type", return_value=TriggerType.COUNT),
            patch.object(processor._aggregation_executor, "execute_flush", side_effect=execute_flush_side_effect),
            patch.object(processor, "_emit_transform_completed"),
            pytest.raises(RuntimeError, match="rows=None"),
        ):
            processor.process_row(
                row_index=0,
                source_row=source_row,
                transforms=[transform],
                ctx=ctx,
            )

    def test_passthrough_success_with_output_count_mismatch_raises(self) -> None:
        """Passthrough flush must return one output row per buffered input token."""
        _db, _recorder, processor, transform, _agg_node = self._setup_batch_processor(output_mode="passthrough")
        source_row = _make_source_row({"value": 10})
        ctx = PluginContext(run_id="test-run", config={})
        captured: dict[str, TokenInfo] = {}

        mismatch_result = TransformResult.success_multi(
            [make_row({"value": 100}, contract=_make_contract())],
            success_reason={"action": "mismatch"},
        )

        def buffer_row_side_effect(node_id: NodeID, token: TokenInfo) -> None:
            captured["token"] = token

        def execute_flush_side_effect(*, node_id, transform, ctx, trigger_type):
            other_token = make_token_info(data={"value": 20})
            return mismatch_result, [captured["token"], other_token], "batch-1"

        with (
            patch.object(processor._aggregation_executor, "buffer_row", side_effect=buffer_row_side_effect),
            patch.object(processor._aggregation_executor, "should_flush", return_value=True),
            patch.object(processor._aggregation_executor, "get_trigger_type", return_value=TriggerType.COUNT),
            patch.object(processor._aggregation_executor, "execute_flush", side_effect=execute_flush_side_effect),
            patch.object(processor, "_emit_transform_completed"),
            pytest.raises(ValueError, match="same number of output rows"),
        ):
            processor.process_row(
                row_index=0,
                source_row=source_row,
                transforms=[transform],
                ctx=ctx,
            )

    def test_timeout_flush_passthrough_with_downstream_returns_continuation_work(self) -> None:
        """Timeout flush routes passthrough tokens into child work when downstream exists."""
        downstream_node = NodeID("downstream-2")
        agg_node = NodeID("agg-1")
        _db, _recorder, processor, transform, agg_node = self._setup_batch_processor(
            output_mode="passthrough",
            node_to_next={NodeID("source-0"): agg_node, agg_node: downstream_node, downstream_node: None},
        )
        ctx = PluginContext(run_id="test-run", config={})

        result = TransformResult.success_multi(
            [make_row({"value": 11}, contract=_make_contract())],
            success_reason={"action": "passthrough"},
        )
        buffered_token = make_token_info(data={"value": 10})

        with patch.object(
            processor._aggregation_executor,
            "execute_flush",
            return_value=(result, [buffered_token], "batch-1"),
        ):
            results, child_items = processor.handle_timeout_flush(
                node_id=agg_node,
                transform=transform,
                ctx=ctx,
                trigger_type=TriggerType.TIMEOUT,
            )

        assert results == []
        assert len(child_items) == 1
        assert child_items[0].current_node_id == downstream_node

    def test_timeout_flush_passthrough_terminal_returns_completed(self) -> None:
        """Timeout flush returns terminal COMPLETED results when no downstream/coalesce exists."""
        _db, _recorder, processor, transform, agg_node = self._setup_batch_processor(output_mode="passthrough")
        ctx = PluginContext(run_id="test-run", config={})

        result = TransformResult.success_multi(
            [make_row({"value": 11}, contract=_make_contract())],
            success_reason={"action": "passthrough"},
        )
        buffered_token = make_token_info(data={"value": 10})

        with patch.object(
            processor._aggregation_executor,
            "execute_flush",
            return_value=(result, [buffered_token], "batch-1"),
        ):
            results, child_items = processor.handle_timeout_flush(
                node_id=agg_node,
                transform=transform,
                ctx=ctx,
                trigger_type=TriggerType.TIMEOUT,
            )

        assert child_items == []
        assert len(results) == 1
        assert results[0].outcome == RowOutcome.COMPLETED
        assert results[0].sink_name == "agg_sink"


class TestProcessRowGateBranching:
    """Tests for non-linear gate branching through next_node_id."""

    def test_config_gate_processing_node_jump_preloads_subchain_sink_for_expanded_children(self) -> None:
        """Config gate PROCESSING_NODE jumps should refresh inherited sink from jumped subchain."""
        _db, recorder = _make_recorder()
        source_row = _make_source_row({"value": 10})
        ctx = PluginContext(run_id="test-run", config={})

        gate_node = NodeID("cfg-gate-1")
        expander_node = NodeID("expander-2")
        terminal_node = NodeID("terminal-3")
        source_node = NodeID("source-0")

        config_gate = GateSettings(
            name="cfg_router",
            input="in_conn",
            condition="'branch_a'",
            routes={"branch_a": "branch_conn"},
        )
        expander = _make_mock_transform(
            node_id=str(expander_node),
            name="expander",
            creates_tokens=True,
            on_success=None,
        )
        terminal = _make_mock_transform(
            node_id=str(terminal_node),
            name="terminal",
            on_success="branch_sink",
        )

        processor = _make_processor(
            recorder,
            source_on_success="source_sink",
            node_step_map={
                source_node: 0,
                gate_node: 1,
                expander_node: 2,
                terminal_node: 3,
            },
            node_to_next={
                source_node: gate_node,
                gate_node: None,
                expander_node: terminal_node,
                terminal_node: None,
            },
            first_transform_node_id=gate_node,
            node_to_plugin={
                gate_node: config_gate,
                expander_node: expander,
                terminal_node: terminal,
            },
        )

        gate_contract = _make_contract()
        gate_result = GateResult(
            row={"value": 10},
            action=RoutingAction.route("branch_a"),
            contract=gate_contract,
        )
        expand_result = TransformResult.success_multi(
            [
                make_row({"value": 10, "idx": 1}, contract=gate_contract),
                make_row({"value": 10, "idx": 2}, contract=gate_contract),
            ],
            success_reason={"action": "expand"},
        )

        def config_gate_side_effect(*, gate_config, node_id, token, ctx, token_manager=None):
            return GateOutcome(
                result=gate_result,
                updated_token=token,
                next_node_id=expander_node,
            )

        def transform_side_effect(*, transform, token, ctx, attempt=0):
            if transform.name == "expander":
                return (expand_result, token, None)
            raise AssertionError("terminal transform should not execute in this regression harness")

        inherited_sinks: list[str | None] = []

        def continuation_side_effect(*, token, current_node_id, coalesce_name=None, on_success_sink=None):
            inherited_sinks.append(on_success_sink)
            return WorkItem(
                token=token,
                current_node_id=None,
                coalesce_node_id=None,
                coalesce_name=coalesce_name,
                on_success_sink=on_success_sink,
            )

        with (
            patch.object(processor._gate_executor, "execute_config_gate", side_effect=config_gate_side_effect),
            patch.object(processor._transform_executor, "execute_transform", side_effect=transform_side_effect),
            patch.object(processor._nav, "create_continuation_work_item", side_effect=continuation_side_effect),
        ):
            results = processor.process_row(
                row_index=0,
                source_row=source_row,
                transforms=[expander, terminal],
                ctx=ctx,
            )

        completed = [r for r in results if r.outcome == RowOutcome.COMPLETED]
        assert len(completed) == 2
        assert inherited_sinks == ["branch_sink", "branch_sink"]
        assert all(r.sink_name == "branch_sink" for r in completed)

    def test_jump_target_terminal_coalesce_missing_on_success_mapping_raises(self) -> None:
        """Terminal coalesce reached via jump must have an on_success sink mapping."""
        _db, recorder = _make_recorder()

        source_node = NodeID("source-0")
        router_node = NodeID("router-1")
        coalesce_node = NodeID("coalesce::merge")

        processor = _make_processor(
            recorder,
            source_on_success="source_sink",
            node_step_map={
                source_node: 0,
                router_node: 1,
                coalesce_node: 2,
            },
            node_to_next={
                source_node: router_node,
                router_node: coalesce_node,
                coalesce_node: None,
            },
            first_transform_node_id=router_node,
            coalesce_node_ids={CoalesceName("merge"): coalesce_node},
            # Intentionally omit coalesce_on_success_map
        )

        with pytest.raises(OrchestrationInvariantError, match="Coalesce 'merge' not in on_success map"):
            processor._nav.resolve_jump_target_sink(router_node)

    def test_jump_target_resolution_raises_when_no_sink_and_no_gate(self) -> None:
        """Jump path with only transforms and no terminal sink must fail closed."""
        _db, recorder = _make_recorder()

        source_node = NodeID("source-0")
        jump_start_node = NodeID("branch-transform-1")
        downstream_transform_node = NodeID("branch-transform-2")

        branch_transform1 = _make_mock_transform(
            node_id=str(jump_start_node),
            name="branch_transform1",
            on_success="branch_conn",
        )
        branch_transform2 = _make_mock_transform(
            node_id=str(downstream_transform_node),
            name="branch_transform2",
            on_success="nonexistent_conn",
        )

        processor = _make_processor(
            recorder,
            source_on_success="source_sink",
            sink_names=frozenset({"source_sink", "branch_sink"}),
            node_step_map={
                source_node: 0,
                jump_start_node: 1,
                downstream_transform_node: 2,
            },
            node_to_next={
                source_node: jump_start_node,
                jump_start_node: downstream_transform_node,
                downstream_transform_node: None,
            },
            first_transform_node_id=jump_start_node,
            node_to_plugin={
                jump_start_node: branch_transform1,
                downstream_transform_node: branch_transform2,
            },
        )

        with pytest.raises(OrchestrationInvariantError, match="no sink"):
            processor._nav.resolve_jump_target_sink(jump_start_node)

    def test_branch_to_sink_routing_applies_for_terminal_fork_children(self) -> None:
        """Branch-routed tokens bypassing coalesce should resolve sink via branch_to_sink."""
        _db, recorder = _make_recorder()
        ctx = PluginContext(run_id="test-run", config={})
        token = TokenInfo(
            row_id="row-1",
            token_id="token-branch-1",
            row_data=make_row({"value": 1}),
            branch_name="path_a",
        )

        processor = _make_processor(
            recorder,
            source_on_success="source_sink",
            branch_to_sink={"path_a": "branch_sink"},
            sink_names=frozenset({"source_sink", "branch_sink"}),
            node_step_map={NodeID("source-0"): 0},
            node_to_next={NodeID("source-0"): None},
            first_transform_node_id=None,
        )

        results = processor.process_token(
            token=token,
            ctx=ctx,
            current_node_id=None,
        )

        assert len(results) == 1
        assert results[0].outcome == RowOutcome.COMPLETED
        assert results[0].sink_name == "branch_sink"

    def test_fork_to_sink_children_bypass_gate_continuation_successor(self) -> None:
        """Regression: fork children in _branch_to_sink must not traverse downstream nodes.

        Topology: gate-1 → transform-1 (gate's structural successor)
        Gate forks to branches sink_a and sink_b (both in _branch_to_sink).

        Without fix: children get current_node_id=transform-1 and execute the transform.
        With fix: children get current_node_id=None, skip the loop, resolve via _branch_to_sink.
        """
        _db, recorder = _make_recorder()
        ctx = PluginContext(run_id="test-run", config={})

        gate_node = NodeID("gate-1")
        transform_node = NodeID("transform-1")

        # Register nodes for FK constraints
        recorder.register_node(
            run_id="test-run",
            plugin_name="fork-gate",
            node_type=NodeType.GATE,
            plugin_version="1.0",
            config={},
            node_id="gate-1",
            schema_config=_DYNAMIC_SCHEMA,
        )
        recorder.register_node(
            run_id="test-run",
            plugin_name="downstream-transform",
            node_type=NodeType.TRANSFORM,
            plugin_version="1.0",
            config={},
            node_id="transform-1",
            schema_config=_DYNAMIC_SCHEMA,
        )

        # Config gate: forks on "true" (always fires)
        gate_config = GateSettings(
            name="fork-gate",
            input="source_out",
            condition="True",
            routes={"true": "fork", "false": "sink_c"},
            fork_to=["sink_a", "sink_b"],
        )

        # Transform is the gate's continuation successor — must NOT execute for fork children
        transform = _make_mock_transform(
            node_id="transform-1",
            result=TransformResult.success({"value": 99, "transformed": True}, success_reason={"action": "test"}),
        )

        processor = _make_processor(
            recorder,
            source_on_success="sink_c",
            branch_to_sink={"sink_a": "sink_a", "sink_b": "sink_b"},
            sink_names=frozenset({"sink_a", "sink_b", "sink_c"}),
            node_step_map={gate_node: 1, transform_node: 2},
            node_to_next={gate_node: transform_node, transform_node: None},
            first_transform_node_id=gate_node,
            node_to_plugin={gate_node: gate_config, transform_node: transform},
        )

        # Mock gate executor to return FORK outcome with two child tokens.
        # This isolates the fork routing logic from gate execution infrastructure.
        def mock_execute_config_gate(gate_config, node_id, token, ctx, token_manager=None):
            child_a = TokenInfo(
                row_id=token.row_id,
                token_id="token-fork-a",
                row_data=token.row_data,
                branch_name="sink_a",
            )
            child_b = TokenInfo(
                row_id=token.row_id,
                token_id="token-fork-b",
                row_data=token.row_data,
                branch_name="sink_b",
            )
            fork_action = RoutingAction.fork_to_paths(["sink_a", "sink_b"])
            fork_result = GateResult(
                row=token.row_data.to_dict(),
                action=fork_action,
                contract=token.row_data.contract,
            )
            fork_result.input_hash = "test-hash"
            fork_result.output_hash = "test-hash"
            fork_result.duration_ms = 0.1
            return GateOutcome(
                result=fork_result,
                updated_token=token,
                child_tokens=[child_a, child_b],
            )

        processor._gate_executor.execute_config_gate = mock_execute_config_gate

        source_row = _make_source_row()
        results = processor.process_row(
            row_index=0,
            source_row=source_row,
            transforms=[],
            ctx=ctx,
        )

        # Parent should be FORKED
        forked = [r for r in results if r.outcome == RowOutcome.FORKED]
        assert len(forked) == 1

        # Fork children should complete at their branch sinks
        completed = [r for r in results if r.outcome == RowOutcome.COMPLETED]
        assert len(completed) == 2
        sink_names = sorted(r.sink_name for r in completed)
        assert sink_names == ["sink_a", "sink_b"]

        # Downstream transform must NOT have been called for fork children
        transform.process.assert_not_called()

    def test_overlapping_branch_to_coalesce_and_branch_to_sink_raises(self) -> None:
        """A branch name in both branch_to_coalesce and branch_to_sink is an invariant violation."""
        _db, recorder = _make_recorder()
        with pytest.raises(OrchestrationInvariantError, match="both branch_to_coalesce and branch_to_sink"):
            _make_processor(
                recorder,
                source_on_success="output",
                branch_to_coalesce={"path_a": CoalesceName("merge_point")},
                branch_to_sink={"path_a": "direct_sink"},
                coalesce_node_ids={CoalesceName("merge_point"): NodeID("coalesce-0")},
                sink_names=frozenset({"output", "direct_sink"}),
                node_step_map={NodeID("source-0"): 0, NodeID("coalesce-0"): 1},
                node_to_next={NodeID("source-0"): None},
            )


# =============================================================================
# process_row: Multi-row output (deaggregation)
# =============================================================================


class TestProcessRowMultiRowOutput:
    """Tests for deaggregation (1→N) in regular transforms."""

    def test_multi_row_with_creates_tokens_returns_expanded(self) -> None:
        """Transform with creates_tokens=True returning multi-row → EXPANDED."""
        _db, recorder = _make_recorder()
        source_row = _make_source_row()
        ctx = PluginContext(run_id="test-run", config={})

        contract = _make_contract()
        output_rows = [
            make_row({"value": 1}, contract=contract),
            make_row({"value": 2}, contract=contract),
        ]
        multi_result = TransformResult.success_multi(
            output_rows,
            success_reason={"action": "expand"},
        )

        transform = _make_mock_transform(creates_tokens=True)
        source_node = NodeID("source-0")
        transform_node = NodeID(transform.node_id)
        processor = _make_processor(
            recorder,
            node_step_map={source_node: 0, transform_node: 1},
            node_to_next={source_node: transform_node, transform_node: None},
            first_transform_node_id=transform_node,
            node_to_plugin={transform_node: transform},
        )

        def executor_side_effect(*, transform, token, ctx, attempt=0):
            return (multi_result, token, None)

        with patch.object(
            processor._transform_executor,
            "execute_transform",
            side_effect=executor_side_effect,
        ):
            results = processor.process_row(
                row_index=0,
                source_row=source_row,
                transforms=[transform],
                ctx=ctx,
            )

        # Parent should be EXPANDED, children should be COMPLETED
        outcomes = {r.outcome for r in results}
        assert RowOutcome.EXPANDED in outcomes
        assert RowOutcome.COMPLETED in outcomes

    def test_multi_row_without_creates_tokens_raises(self) -> None:
        """Transform returning multi-row without creates_tokens=True → RuntimeError."""
        _db, recorder = _make_recorder()
        source_row = _make_source_row()
        ctx = PluginContext(run_id="test-run", config={})

        contract = _make_contract()
        output_rows = [
            make_row({"value": 1}, contract=contract),
            make_row({"value": 2}, contract=contract),
        ]
        multi_result = TransformResult.success_multi(
            output_rows,
            success_reason={"action": "expand"},
        )

        transform = _make_mock_transform(creates_tokens=False)
        source_node = NodeID("source-0")
        transform_node = NodeID(transform.node_id)
        processor = _make_processor(
            recorder,
            node_step_map={source_node: 0, transform_node: 1},
            node_to_next={source_node: transform_node, transform_node: None},
            first_transform_node_id=transform_node,
            node_to_plugin={transform_node: transform},
        )

        def executor_side_effect(*, transform, token, ctx, attempt=0):
            return (multi_result, token, None)

        with (
            patch.object(
                processor._transform_executor,
                "execute_transform",
                side_effect=executor_side_effect,
            ),
            pytest.raises(RuntimeError, match="creates_tokens=False"),
        ):
            processor.process_row(
                row_index=0,
                source_row=source_row,
                transforms=[transform],
                ctx=ctx,
            )

    def test_multi_row_with_inconsistent_contracts_raises(self) -> None:
        """Multi-row result with mixed contracts must crash at construction (plugin bug).

        Validation lives in TransformResult.success_multi() — it fires before
        the result can reach any consumer in the processor.
        """
        from elspeth.contracts.errors import PluginContractViolation

        contract_a = make_contract(fields={"value": int}, mode="OBSERVED")
        contract_b = make_contract(fields={"other": str}, mode="OBSERVED")
        output_rows = [
            make_row({"value": 1}, contract=contract_a),
            make_row({"other": "x"}, contract=contract_b),
        ]
        with pytest.raises(PluginContractViolation, match="inconsistent contracts"):
            TransformResult.success_multi(
                output_rows,
                success_reason={"action": "expand"},
            )


# =============================================================================
# process_existing_row (resume path)
# =============================================================================


class TestProcessExistingRow:
    """Tests for process_existing_row (resume after crash)."""

    def test_does_not_create_new_row_record(self) -> None:
        """process_existing_row creates token but NOT a new row."""
        _db, recorder = _make_recorder()

        processor = _make_processor(recorder)

        contract = _make_contract()
        row_data = make_row({"value": 42}, contract=contract)
        ctx = PluginContext(run_id="test-run", config={})

        # We need a pre-existing row. Create one via process_row first.
        source_row = _make_source_row({"value": 42})
        first_results = processor.process_row(
            row_index=0,
            source_row=source_row,
            transforms=[],
            ctx=ctx,
        )
        existing_row_id = first_results[0].token.row_id

        # Now process_existing_row for the same row
        results = processor.process_existing_row(
            row_id=existing_row_id,
            row_data=row_data,
            transforms=[],
            ctx=ctx,
        )

        assert len(results) == 1
        assert results[0].outcome == RowOutcome.COMPLETED
        assert results[0].sink_name == "default"
        # The row_id should match the existing row
        assert results[0].token.row_id == existing_row_id


# =============================================================================
# process_token (mid-pipeline entry)
# =============================================================================


class TestProcessToken:
    """Tests for process_token (used for coalesce merge continuations)."""

    def test_process_token_from_midpoint(self) -> None:
        """process_token starts processing from a given step."""
        _db, recorder = _make_recorder()

        processor = _make_processor(recorder)
        ctx = PluginContext(run_id="test-run", config={})

        # Create a token to process
        token = make_token_info(data={"value": 42})

        results = processor.process_token(
            token=token,
            ctx=ctx,
            current_node_id=NodeID("source-0"),
        )

        assert len(results) == 1
        assert results[0].outcome == RowOutcome.COMPLETED
        assert results[0].sink_name == "default"

    def test_terminal_coalesce_continuation_uses_coalesce_on_success_sink(self) -> None:
        """Merged token resumed at terminal coalesce must route to coalesce sink, not source sink."""
        _db, recorder = _make_recorder()

        processor = _make_processor(
            recorder,
            source_on_success="source_sink",
            coalesce_node_ids={CoalesceName("merge"): NodeID("coalesce::merge")},
            coalesce_on_success_map={CoalesceName("merge"): "coalesce_sink"},
            node_step_map={NodeID("coalesce::merge"): 1},
            node_to_next={NodeID("coalesce::merge"): None},
        )
        ctx = PluginContext(run_id="test-run", config={})
        token = make_token_info(data={"value": 42})

        results = processor.process_token(
            token=token,
            ctx=ctx,
            current_node_id=NodeID("coalesce::merge"),
            coalesce_node_id=NodeID("coalesce::merge"),
            coalesce_name=CoalesceName("merge"),
        )

        assert len(results) == 1
        assert results[0].outcome == RowOutcome.COMPLETED
        assert results[0].sink_name == "coalesce_sink"

    def test_non_terminal_coalesce_continuation_uses_downstream_sink(self) -> None:
        """Coalesce sink must not override downstream terminal transform routing."""
        _db, recorder = _make_recorder()
        transform = _make_mock_transform(node_id="transform-1")
        transform.on_success = "downstream_sink"

        processor = _make_processor(
            recorder,
            source_on_success="source_sink",
            coalesce_node_ids={CoalesceName("merge"): NodeID("coalesce::merge")},
            coalesce_on_success_map={CoalesceName("merge"): "coalesce_sink"},
            node_step_map={NodeID("coalesce::merge"): 1, NodeID("transform-1"): 2},
            node_to_next={NodeID("coalesce::merge"): NodeID("transform-1"), NodeID("transform-1"): None},
            first_transform_node_id=NodeID("transform-1"),
            node_to_plugin={NodeID("transform-1"): transform},
        )
        ctx = PluginContext(run_id="test-run", config={})
        token = make_token_info(data={"value": 42})

        with patch.object(
            processor,
            "_execute_transform_with_retry",
            return_value=(
                TransformResult.success(make_row({"value": 42}), success_reason={"action": "noop"}),
                token,
                None,
            ),
        ):
            results = processor.process_token(
                token=token,
                ctx=ctx,
                current_node_id=NodeID("coalesce::merge"),
                coalesce_node_id=NodeID("coalesce::merge"),
                coalesce_name=CoalesceName("merge"),
            )

        assert len(results) == 1
        assert results[0].outcome == RowOutcome.COMPLETED
        assert results[0].sink_name == "downstream_sink"


# =============================================================================
# _drain_work_queue: Iteration Guard
# =============================================================================


class TestDrainWorkQueueIterationGuard:
    """Tests for the MAX_WORK_QUEUE_ITERATIONS safety limit."""

    def test_iteration_guard_prevents_infinite_loop(self) -> None:
        """Work queue exceeding MAX_WORK_QUEUE_ITERATIONS raises RuntimeError."""
        _db, recorder = _make_recorder()

        processor = _make_processor(recorder)
        ctx = PluginContext(run_id="test-run", config={})
        token = make_token_info(data={"value": 1})

        # Mock _process_single_token to always produce more work
        def infinite_loop_producer(token, ctx, current_node_id, **kwargs):
            new_token = make_token_info(data={"value": 1})
            return (None, [WorkItem(token=new_token, current_node_id=NodeID("source-0"))])

        with (
            patch.object(processor, "_process_single_token", side_effect=infinite_loop_producer),
            pytest.raises(RuntimeError, match=r"exceeded.*iterations"),
        ):
            processor._drain_work_queue(
                WorkItem(token=token, current_node_id=NodeID("source-0")),
                ctx=ctx,
            )

    def test_max_iterations_constant_is_reasonable(self) -> None:
        """MAX_WORK_QUEUE_ITERATIONS should be at least 1000."""
        assert MAX_WORK_QUEUE_ITERATIONS >= 1000


# =============================================================================
# _execute_transform_with_retry: No retry manager
# =============================================================================


class TestExecuteTransformNoRetry:
    """Tests for _execute_transform_with_retry when no retry_manager is set."""

    def _setup(self) -> tuple[LandscapeDB, LandscapeRecorder, RowProcessor]:
        db, recorder = _make_recorder()

        processor = _make_processor(recorder, retry_manager=None)
        return db, recorder, processor

    def test_single_attempt_success(self) -> None:
        """Without retry_manager, executes single attempt."""
        _, _, processor = self._setup()
        transform = _make_mock_transform()
        token = make_token_info(data={"value": 42})
        ctx = PluginContext(run_id="test-run", config={})
        expected_result = TransformResult.success(
            make_row({"value": 42}),
            success_reason={"action": "test"},
        )

        with patch.object(
            processor._transform_executor,
            "execute_transform",
            return_value=(expected_result, token, None),
        ) as mock_exec:
            result, _out_token, error_sink = processor._execute_transform_with_retry(
                transform=transform,
                token=token,
                ctx=ctx,
            )
            mock_exec.assert_called_once()
            assert result.status == "success"
            assert error_sink is None

    def test_retryable_llm_error_with_on_error_discard(self) -> None:
        """Retryable LLMClientError with on_error='discard' returns error result (no re-raise)."""
        _, _, processor = self._setup()
        transform = _make_mock_transform(node_id="t1", on_error="discard")
        token = make_token_info(data={"value": 42})
        ctx = PluginContext(run_id="test-run", config={})

        llm_error = LLMClientError("rate limited", retryable=True)
        with patch.object(
            processor._transform_executor,
            "execute_transform",
            side_effect=llm_error,
        ):
            result, _out_token, error_sink = processor._execute_transform_with_retry(
                transform=transform,
                token=token,
                ctx=ctx,
            )

        assert result.status == "error"
        assert error_sink == "discard"

    def test_retryable_llm_error_on_error_is_always_set(self) -> None:
        """on_error is now required at config time — None no longer reaches runtime.

        Previously on_error=None would raise RuntimeError. Now TransformSettings
        requires on_error, so every transform has a valid error route. This test
        documents the invariant by verifying 'discard' (minimum valid value) works.
        """
        _, _, processor = self._setup()
        transform = _make_mock_transform(node_id="t1", on_error="discard")
        token = make_token_info(data={"value": 42})
        ctx = PluginContext(run_id="test-run", config={})

        llm_error = LLMClientError("rate limited", retryable=True)
        with patch.object(
            processor._transform_executor,
            "execute_transform",
            side_effect=llm_error,
        ):
            result, _out_token, error_sink = processor._execute_transform_with_retry(
                transform=transform,
                token=token,
                ctx=ctx,
            )

        assert result.status == "error"
        assert error_sink == "discard"

    def test_non_retryable_llm_error_reraises(self) -> None:
        """Non-retryable LLMClientError is re-raised directly."""
        _, _, processor = self._setup()
        transform = _make_mock_transform(node_id="t1", on_error="discard")
        token = make_token_info(data={"value": 42})
        ctx = PluginContext(run_id="test-run", config={})

        llm_error = LLMClientError("content policy", retryable=False)
        with (
            patch.object(
                processor._transform_executor,
                "execute_transform",
                side_effect=llm_error,
            ),
            pytest.raises(LLMClientError, match="content policy"),
        ):
            processor._execute_transform_with_retry(
                transform=transform,
                token=token,
                ctx=ctx,
            )

    def test_transient_connection_error_with_on_error(self) -> None:
        """ConnectionError with on_error returns error result (no re-raise)."""
        _, _, processor = self._setup()
        transform = _make_mock_transform(node_id="t1", on_error="discard")
        token = make_token_info(data={"value": 42})
        ctx = PluginContext(run_id="test-run", config={})

        with patch.object(
            processor._transform_executor,
            "execute_transform",
            side_effect=ConnectionError("connection reset"),
        ):
            result, _out_token, error_sink = processor._execute_transform_with_retry(
                transform=transform,
                token=token,
                ctx=ctx,
            )

        assert result.status == "error"
        assert error_sink == "discard"

    def test_transient_timeout_error_with_on_error(self) -> None:
        """TimeoutError with on_error returns error result."""
        _, _, processor = self._setup()
        transform = _make_mock_transform(node_id="t1", on_error="discard")
        token = make_token_info(data={"value": 42})
        ctx = PluginContext(run_id="test-run", config={})

        with patch.object(
            processor._transform_executor,
            "execute_transform",
            side_effect=TimeoutError("timed out"),
        ):
            result, _out_token, error_sink = processor._execute_transform_with_retry(
                transform=transform,
                token=token,
                ctx=ctx,
            )

        assert result.status == "error"
        assert error_sink == "discard"

    def test_capacity_error_with_on_error_returns_row_scoped_error(self) -> None:
        """CapacityError with no retry manager returns retryable row error."""
        _, _, processor = self._setup()
        transform = _make_mock_transform(node_id="t1", on_error="discard")
        token = make_token_info(data={"value": 42})
        ctx = PluginContext(run_id="test-run", config={})

        with patch.object(
            processor._transform_executor,
            "execute_transform",
            side_effect=CapacityError(429, "rate limited"),
        ):
            result, _out_token, error_sink = processor._execute_transform_with_retry(
                transform=transform,
                token=token,
                ctx=ctx,
            )

        assert result.status == "error"
        assert result.retryable is True
        assert error_sink == "discard"

    def test_transient_error_on_error_is_always_set(self) -> None:
        """on_error is now required at config time — None no longer reaches runtime.

        Previously on_error=None would raise RuntimeError on transient errors.
        Now TransformSettings requires on_error, so every transform has a valid
        error route. This test documents the invariant.
        """
        _, _, processor = self._setup()
        transform = _make_mock_transform(node_id="t1", on_error="discard")
        token = make_token_info(data={"value": 42})
        ctx = PluginContext(run_id="test-run", config={})

        with patch.object(
            processor._transform_executor,
            "execute_transform",
            side_effect=ConnectionError("connection reset"),
        ):
            result, _out_token, error_sink = processor._execute_transform_with_retry(
                transform=transform,
                token=token,
                ctx=ctx,
            )

        assert result.status == "error"
        assert error_sink == "discard"

    def test_retryable_llm_error_with_named_error_sink_returns_error_sink(self) -> None:
        """Retryable LLMClientError with on_error pointing to real sink returns that sink."""
        _, recorder, processor = self._setup()
        # Set up error edge so the routing path is reachable
        processor._error_edge_ids = {NodeID("t1"): "error-edge-1"}

        transform = _make_mock_transform(node_id="t1", on_error="error-sink")
        token = make_token_info(data={"value": 42})
        ctx = PluginContext(run_id="test-run", config={})
        ctx.state_id = "state-123"

        llm_error = LLMClientError("rate limited", retryable=True)
        # Mock record_routing_event since we don't have a real state_id in DB
        with (
            patch.object(processor._transform_executor, "execute_transform", side_effect=llm_error),
            patch.object(recorder, "record_routing_event"),
        ):
            result, _out_token, error_sink = processor._execute_transform_with_retry(
                transform=transform,
                token=token,
                ctx=ctx,
            )

        assert result.status == "error"
        assert error_sink == "error-sink"

    def test_retryable_llm_error_with_missing_error_edge_raises(self) -> None:
        """Retryable error with named sink but no DIVERT edge → OrchestrationInvariantError."""
        _, _, processor = self._setup()
        # No error edges configured
        processor._error_edge_ids = {}

        transform = _make_mock_transform(node_id="t1", on_error="error-sink")
        token = make_token_info(data={"value": 42})
        ctx = PluginContext(run_id="test-run", config={})
        ctx.state_id = "state-123"

        llm_error = LLMClientError("rate limited", retryable=True)
        with (
            patch.object(
                processor._transform_executor,
                "execute_transform",
                side_effect=llm_error,
            ),
            pytest.raises(OrchestrationInvariantError, match="no DIVERT edge"),
        ):
            processor._execute_transform_with_retry(
                transform=transform,
                token=token,
                ctx=ctx,
            )


# =============================================================================
# _execute_transform_with_retry: With retry manager
# =============================================================================


class TestExecuteTransformWithRetry:
    """Tests for _execute_transform_with_retry when retry_manager IS configured."""

    def test_delegates_to_retry_manager(self) -> None:
        """With retry_manager, delegates to execute_with_retry."""
        _, recorder = _make_recorder()

        retry_manager = Mock(spec=RetryManager)
        processor = _make_processor(recorder, retry_manager=retry_manager)

        expected = (
            TransformResult.success(make_row({"value": 42}), success_reason={"action": "test"}),
            make_token_info(data={"value": 42}),
            None,
        )
        retry_manager.execute_with_retry.return_value = expected

        transform = _make_mock_transform()
        token = make_token_info(data={"value": 42})
        ctx = PluginContext(run_id="test-run", config={})

        result = processor._execute_transform_with_retry(
            transform=transform,
            token=token,
            ctx=ctx,
        )

        retry_manager.execute_with_retry.assert_called_once()
        assert result == expected

    def test_is_retryable_accepts_retryable_llm_error(self) -> None:
        """is_retryable callback returns True for retryable LLMClientError."""
        _, recorder = _make_recorder()

        retry_manager = Mock(spec=RetryManager)
        processor = _make_processor(recorder, retry_manager=retry_manager)

        # Capture the is_retryable callback
        retry_manager.execute_with_retry.return_value = (
            TransformResult.success(make_row({}), success_reason={"action": "t"}),
            make_token_info(),
            None,
        )

        transform = _make_mock_transform()
        token = make_token_info()
        ctx = PluginContext(run_id="test-run", config={})

        processor._execute_transform_with_retry(
            transform=transform,
            token=token,
            ctx=ctx,
        )

        # Extract the is_retryable callback from the call
        call_kwargs = retry_manager.execute_with_retry.call_args
        is_retryable = call_kwargs.kwargs.get("is_retryable") or call_kwargs[1].get("is_retryable")
        if is_retryable is None:
            # Might be positional
            is_retryable = call_kwargs[0][1] if len(call_kwargs[0]) > 1 else call_kwargs.kwargs["is_retryable"]

        # Verify retryable logic
        assert is_retryable(LLMClientError("rate limit", retryable=True)) is True
        assert is_retryable(LLMClientError("content policy", retryable=False)) is False
        assert is_retryable(ConnectionError("conn reset")) is True
        assert is_retryable(TimeoutError("timeout")) is True
        assert is_retryable(CapacityError(429, "rate limited")) is True
        assert is_retryable(AttributeError("bug")) is False
        assert is_retryable(TypeError("bug")) is False


# =============================================================================
# _maybe_coalesce_token
# =============================================================================


class TestMaybeCoalesceToken:
    """Tests for _maybe_coalesce_token branch routing."""

    def test_no_coalesce_executor_returns_not_handled(self) -> None:
        """Without coalesce_executor, always returns (False, None)."""
        _, recorder = _make_recorder()
        processor = _make_processor(recorder, coalesce_executor=None)
        token = make_token_info()

        handled, result = processor._maybe_coalesce_token(
            token,
            current_node_id=NodeID("coalesce::merge"),
            coalesce_node_id=NodeID("coalesce::merge"),
            coalesce_name=CoalesceName("merge"),
            child_items=[],
        )

        assert handled is False
        assert result is None

    def test_token_without_branch_returns_not_handled(self) -> None:
        """Token without branch_name is not a fork child, skip coalesce."""
        _, recorder = _make_recorder()
        coalesce = Mock()
        processor = _make_processor(
            recorder,
            coalesce_executor=coalesce,
            coalesce_node_ids={CoalesceName("merge"): NodeID("coalesce::merge")},
            node_step_map={NodeID("coalesce::merge"): 1},
        )
        token = make_token_info()
        # Ensure branch_name is None
        token = TokenInfo(
            row_id=token.row_id,
            token_id=token.token_id,
            row_data=token.row_data,
            branch_name=None,
        )

        handled, _result = processor._maybe_coalesce_token(
            token,
            current_node_id=NodeID("coalesce::merge"),
            coalesce_node_id=NodeID("coalesce::merge"),
            coalesce_name=CoalesceName("merge"),
            child_items=[],
        )

        assert handled is False

    def test_current_node_not_coalesce_node_returns_not_handled(self) -> None:
        """Coalesce is only triggered when traversal reaches the coalesce node."""
        _, recorder = _make_recorder()
        coalesce = Mock()
        processor = _make_processor(
            recorder,
            coalesce_executor=coalesce,
            coalesce_node_ids={CoalesceName("merge"): NodeID("coalesce::merge")},
            node_step_map={NodeID("coalesce::merge"): 5},
        )
        token = TokenInfo(
            row_id="row-1",
            token_id="token-1",
            row_data=make_row({}),
            branch_name="path_a",
        )

        handled, _result = processor._maybe_coalesce_token(
            token,
            current_node_id=NodeID("transform-3"),
            coalesce_node_id=NodeID("coalesce::merge"),
            coalesce_name=CoalesceName("merge"),
            child_items=[],
        )

        assert handled is False

    def test_coalesce_held_returns_handled_with_none(self) -> None:
        """Token accepted but not all branches arrived → handled=True, result=None."""
        _, recorder = _make_recorder()
        coalesce = Mock()
        coalesce.accept.return_value = Mock(held=True, merged_token=None)
        processor = _make_processor(
            recorder,
            coalesce_executor=coalesce,
            coalesce_node_ids={CoalesceName("merge"): NodeID("coalesce::merge")},
            node_step_map={NodeID("coalesce::merge"): 2},
        )
        token = TokenInfo(
            row_id="row-1",
            token_id="token-1",
            row_data=make_row({}),
            branch_name="path_a",
        )

        handled, result = processor._maybe_coalesce_token(
            token,
            current_node_id=NodeID("coalesce::merge"),
            coalesce_node_id=NodeID("coalesce::merge"),
            coalesce_name=CoalesceName("merge"),
            child_items=[],
        )

        assert handled is True
        assert result is None

    def test_coalesce_failure_with_outcomes_recorded_does_not_duplicate_recording(self) -> None:
        """When executor already recorded FAILED outcome, processor must not record again."""
        _, recorder = _make_recorder()
        coalesce = Mock()
        coalesce.accept.return_value = Mock(
            held=False,
            merged_token=None,
            failure_reason="late_arrival_after_merge",
            outcomes_recorded=True,
        )
        processor = _make_processor(
            recorder,
            coalesce_executor=coalesce,
            coalesce_node_ids={CoalesceName("merge"): NodeID("coalesce::merge")},
            node_step_map={NodeID("coalesce::merge"): 2},
        )
        token = TokenInfo(
            row_id="row-1",
            token_id="token-1",
            row_data=make_row({}),
            branch_name="path_a",
        )

        with (
            patch.object(recorder, "record_token_outcome") as record_outcome,
            patch.object(processor, "_emit_token_completed") as emit_token_completed,
        ):
            handled, result = processor._maybe_coalesce_token(
                token,
                current_node_id=NodeID("coalesce::merge"),
                coalesce_node_id=NodeID("coalesce::merge"),
                coalesce_name=CoalesceName("merge"),
                child_items=[],
            )

        assert handled is True
        assert result is not None
        assert result.outcome == RowOutcome.FAILED
        record_outcome.assert_not_called()
        emit_token_completed.assert_called_once()

    def test_coalesce_merged_at_terminal_returns_coalesced_result(self) -> None:
        """All branches arrived at terminal coalesce → COALESCED result."""
        _, recorder = _make_recorder()
        merged_token = make_token_info(data={"merged": True})
        coalesce = Mock()
        coalesce.accept.return_value = Mock(held=False, merged_token=merged_token)
        processor = _make_processor(
            recorder,
            coalesce_executor=coalesce,
            coalesce_node_ids={CoalesceName("merge"): NodeID("coalesce::merge")},
            node_step_map={NodeID("coalesce::merge"): 3},
            coalesce_on_success_map={CoalesceName("merge"): "output"},
        )
        token = TokenInfo(
            row_id="row-1",
            token_id="token-1",
            row_data=make_row({}),
            branch_name="path_a",
        )

        handled, result = processor._maybe_coalesce_token(
            token,
            current_node_id=NodeID("coalesce::merge"),
            coalesce_node_id=NodeID("coalesce::merge"),
            coalesce_name=CoalesceName("merge"),
            child_items=[],
        )

        assert handled is True
        assert result is not None
        assert result.outcome == RowOutcome.COALESCED
        assert result.sink_name == "output"

    def test_coalesce_merged_at_terminal_missing_sink_mapping_raises(self) -> None:
        """Terminal coalesce merge without sink mapping is an internal bug."""
        _, recorder = _make_recorder()
        merged_token = make_token_info(data={"merged": True})
        coalesce = Mock()
        coalesce.accept.return_value = Mock(held=False, merged_token=merged_token)
        processor = _make_processor(
            recorder,
            coalesce_executor=coalesce,
            coalesce_node_ids={CoalesceName("merge"): NodeID("coalesce::merge")},
            node_step_map={NodeID("coalesce::merge"): 3},
            # Intentionally omit coalesce_on_success_map
        )
        token = TokenInfo(
            row_id="row-1",
            token_id="token-1",
            row_data=make_row({}),
            branch_name="path_a",
        )

        with pytest.raises(OrchestrationInvariantError, match="Coalesce 'merge' not in on_success map"):
            processor._maybe_coalesce_token(
                token,
                current_node_id=NodeID("coalesce::merge"),
                coalesce_node_id=NodeID("coalesce::merge"),
                coalesce_name=CoalesceName("merge"),
                child_items=[],
            )

    def test_coalesce_merged_at_non_terminal_queues_work(self) -> None:
        """Merged at non-terminal step → child work item added, no result."""
        _, recorder = _make_recorder()
        merged_token = make_token_info(data={"merged": True})
        coalesce = Mock()
        coalesce.accept.return_value = Mock(held=False, merged_token=merged_token)
        processor = _make_processor(
            recorder,
            coalesce_executor=coalesce,
            coalesce_node_ids={CoalesceName("merge"): NodeID("coalesce::merge")},
            node_step_map={NodeID("coalesce::merge"): 2},
            node_to_next={NodeID("coalesce::merge"): NodeID("transform-5")},
        )
        token = TokenInfo(
            row_id="row-1",
            token_id="token-1",
            row_data=make_row({}),
            branch_name="path_a",
        )
        child_items: list[WorkItem] = []

        handled, result = processor._maybe_coalesce_token(
            token,
            current_node_id=NodeID("coalesce::merge"),
            coalesce_node_id=NodeID("coalesce::merge"),
            coalesce_name=CoalesceName("merge"),
            child_items=child_items,
        )

        assert handled is True
        assert result is None
        assert len(child_items) == 1
        assert child_items[0].current_node_id == NodeID("coalesce::merge")

    def test_invalid_coalesce_outcome_state_raises_invariant(self) -> None:
        """CoalesceOutcome must be held, merged, or failed; empty state is invalid."""
        _db, recorder = _make_recorder()
        coalesce = Mock()
        coalesce.accept.return_value = Mock(
            held=False,
            merged_token=None,
            failure_reason=None,
            outcomes_recorded=False,
        )
        processor = _make_processor(
            recorder,
            coalesce_executor=coalesce,
            coalesce_node_ids={CoalesceName("merge"): NodeID("coalesce::merge")},
            node_step_map={NodeID("coalesce::merge"): 2},
        )
        token = TokenInfo(
            row_id="row-1",
            token_id="token-1",
            row_data=make_row({}),
            branch_name="path_a",
        )

        with pytest.raises(OrchestrationInvariantError, match="invalid state"):
            processor._maybe_coalesce_token(
                token,
                current_node_id=NodeID("coalesce::merge"),
                coalesce_node_id=NodeID("coalesce::merge"),
                coalesce_name=CoalesceName("merge"),
                child_items=[],
            )


# =============================================================================
# _notify_coalesce_of_lost_branch
# =============================================================================


class TestNotifyCoalesceOfLostBranch:
    """Tests for the branch loss notification to coalesce executor."""

    def test_no_coalesce_executor_returns_empty(self) -> None:
        """Without coalesce_executor, returns empty list."""
        _, recorder = _make_recorder()
        processor = _make_processor(recorder, coalesce_executor=None)
        token = TokenInfo(
            row_id="row-1",
            token_id="token-1",
            row_data=make_row({}),
            branch_name="path_a",
        )

        results = processor._notify_coalesce_of_lost_branch(
            token,
            "quarantined:bad_value",
            [],
        )

        assert results == []

    def test_token_without_branch_returns_empty(self) -> None:
        """Non-forked token → no coalesce notification needed."""
        _, recorder = _make_recorder()
        coalesce = Mock()
        processor = _make_processor(recorder, coalesce_executor=coalesce)
        token = TokenInfo(
            row_id="row-1",
            token_id="token-1",
            row_data=make_row({}),
            branch_name=None,
        )

        results = processor._notify_coalesce_of_lost_branch(
            token,
            "quarantined:bad_value",
            [],
        )

        assert results == []

    def test_branch_not_in_coalesce_map_returns_empty(self) -> None:
        """Branch without coalesce mapping → no notification."""
        _, recorder = _make_recorder()
        coalesce = Mock()
        processor = _make_processor(
            recorder,
            coalesce_executor=coalesce,
            branch_to_coalesce={},  # Empty map
        )
        token = TokenInfo(
            row_id="row-1",
            token_id="token-1",
            row_data=make_row({}),
            branch_name="unmapped_branch",
        )

        results = processor._notify_coalesce_of_lost_branch(
            token,
            "quarantined:bad_value",
            [],
        )

        assert results == []

    def test_lost_branch_with_failure_returns_sibling_results(self) -> None:
        """Branch loss causing coalesce failure returns FAILED sibling results."""
        _, recorder = _make_recorder()
        coalesce = Mock()
        sibling_token = make_token_info(data={"value": 99})
        coalesce.notify_branch_lost.return_value = Mock(
            merged_token=None,
            failure_reason="not enough branches",
            consumed_tokens=[sibling_token],
        )
        processor = _make_processor(
            recorder,
            coalesce_executor=coalesce,
            branch_to_coalesce={BranchName("path_a"): CoalesceName("merge")},
            coalesce_node_ids={CoalesceName("merge"): NodeID("coalesce::merge")},
            node_step_map={NodeID("coalesce::merge"): 3},
        )
        token = TokenInfo(
            row_id="row-1",
            token_id="token-1",
            row_data=make_row({}),
            branch_name="path_a",
        )

        results = processor._notify_coalesce_of_lost_branch(
            token,
            "quarantined:bad_value",
            [],
        )

        assert len(results) == 1
        assert results[0].outcome == RowOutcome.FAILED
        assert results[0].error is not None
        assert "not enough branches" in results[0].error.message

    def test_lost_branch_triggers_terminal_merge(self) -> None:
        """Branch loss triggers merge at terminal coalesce → COALESCED result."""
        _, recorder = _make_recorder()
        merged_token = make_token_info(data={"merged": True})
        coalesce = Mock()
        coalesce.notify_branch_lost.return_value = Mock(
            merged_token=merged_token,
            failure_reason=None,
            consumed_tokens=[],
        )
        processor = _make_processor(
            recorder,
            coalesce_executor=coalesce,
            branch_to_coalesce={BranchName("path_a"): CoalesceName("merge")},
            coalesce_node_ids={CoalesceName("merge"): NodeID("coalesce::merge")},
            node_step_map={NodeID("coalesce::merge"): 5},
            coalesce_on_success_map={CoalesceName("merge"): "output"},
        )
        token = TokenInfo(
            row_id="row-1",
            token_id="token-1",
            row_data=make_row({}),
            branch_name="path_a",
        )

        results = processor._notify_coalesce_of_lost_branch(
            token,
            "quarantined:bad_value",
            [],
        )

        assert len(results) == 1
        assert results[0].outcome == RowOutcome.COALESCED
        assert results[0].sink_name == "output"

    def test_lost_branch_terminal_merge_missing_sink_mapping_raises(self) -> None:
        """Terminal coalesce merge from branch loss must have sink mapping."""
        _, recorder = _make_recorder()
        merged_token = make_token_info(data={"merged": True})
        coalesce = Mock()
        coalesce.notify_branch_lost.return_value = Mock(
            merged_token=merged_token,
            failure_reason=None,
            consumed_tokens=[],
        )
        processor = _make_processor(
            recorder,
            coalesce_executor=coalesce,
            branch_to_coalesce={BranchName("path_a"): CoalesceName("merge")},
            coalesce_node_ids={CoalesceName("merge"): NodeID("coalesce::merge")},
            node_step_map={NodeID("coalesce::merge"): 5},
            # Intentionally omit coalesce_on_success_map
        )
        token = TokenInfo(
            row_id="row-1",
            token_id="token-1",
            row_data=make_row({}),
            branch_name="path_a",
        )

        with pytest.raises(OrchestrationInvariantError, match="Coalesce 'merge' not in on_success map"):
            processor._notify_coalesce_of_lost_branch(
                token,
                "quarantined:bad_value",
                [],
            )

    def test_lost_branch_triggers_nonterminal_merge_queues_work(self) -> None:
        """Branch loss triggers merge at non-terminal step → queues work."""
        _, recorder = _make_recorder()
        merged_token = make_token_info(data={"merged": True})
        coalesce = Mock()
        coalesce.notify_branch_lost.return_value = Mock(
            merged_token=merged_token,
            failure_reason=None,
            consumed_tokens=[],
        )
        child_items: list[WorkItem] = []
        processor = _make_processor(
            recorder,
            coalesce_executor=coalesce,
            branch_to_coalesce={BranchName("path_a"): CoalesceName("merge")},
            coalesce_node_ids={CoalesceName("merge"): NodeID("coalesce::merge")},
            node_step_map={NodeID("coalesce::merge"): 3},
            node_to_next={NodeID("coalesce::merge"): NodeID("transform-4")},
        )
        token = TokenInfo(
            row_id="row-1",
            token_id="token-1",
            row_data=make_row({}),
            branch_name="path_a",
        )

        results = processor._notify_coalesce_of_lost_branch(
            token,
            "quarantined:bad_value",
            child_items,
        )

        assert results == []
        assert len(child_items) == 1
        assert child_items[0].current_node_id == NodeID("coalesce::merge")


# =============================================================================
# Unknown transform type
# =============================================================================


class TestUnknownTransformType:
    """Tests for the TypeError guard on unknown transform types."""

    def test_unknown_type_raises_type_error(self) -> None:
        """Transform that is neither TransformProtocol nor GateSettings raises TypeError."""
        _db, recorder = _make_recorder()
        source_row = _make_source_row()
        ctx = PluginContext(run_id="test-run", config={})

        # Create an object that is NOT a transform or gate
        class FakePlugin:
            node_id = "fake-node"

        fake_plugin = FakePlugin()
        source_node = NodeID("source-0")
        fake_node = NodeID(fake_plugin.node_id)
        processor = _make_processor(
            recorder,
            node_step_map={source_node: 0, fake_node: 1},
            node_to_next={source_node: fake_node, fake_node: None},
            first_transform_node_id=fake_node,
            node_to_plugin={fake_node: fake_plugin},
        )

        with pytest.raises(TypeError, match="Unknown transform type"):
            processor.process_row(
                row_index=0,
                source_row=source_row,
                transforms=[fake_plugin],
                ctx=ctx,
            )


class TestRoutingInvariantFailures:
    """Regression tests for strict fail-closed routing invariants."""

    def test_unhandled_config_gate_routing_kind_raises(self) -> None:
        """Config gate branch must fail closed when CONTINUE invariants are violated."""
        _db, recorder = _make_recorder()
        source_row = _make_source_row({"value": 10})
        ctx = PluginContext(run_id="test-run", config={})

        source_node = NodeID("source-0")
        gate_node = NodeID("cfg-gate-1")
        config_gate = GateSettings(
            name="cfg_router",
            input="default",
            condition="True",
            routes={"true": "default", "false": "default"},
        )
        processor = _make_processor(
            recorder,
            source_on_success="source_sink",
            node_step_map={source_node: 0, gate_node: 1},
            node_to_next={source_node: gate_node, gate_node: None},
            first_transform_node_id=gate_node,
            node_to_plugin={gate_node: config_gate},
        )

        bad_outcome = GateOutcome(
            result=GateResult(
                row={"value": 10},
                action=RoutingAction.route("branch_a"),
                contract=_make_contract(),
            ),
            updated_token=make_token_info(data={"value": 10}),
            sink_name=None,
            next_node_id=None,
            child_tokens=[],
        )

        with (
            patch.object(processor._gate_executor, "execute_config_gate", return_value=bad_outcome),
            pytest.raises(OrchestrationInvariantError, match="Unhandled config gate routing kind"),
        ):
            processor.process_row(
                row_index=0,
                source_row=source_row,
                transforms=[],
                ctx=ctx,
            )

    def test_missing_effective_sink_raises_invariant(self) -> None:
        """Terminal completion must not fall back when no sink can be resolved."""
        _db, recorder = _make_recorder()
        source_row = _make_source_row({"value": 10})
        ctx = PluginContext(run_id="test-run", config={})

        processor = _make_processor(
            recorder,
            source_on_success="   ",
            node_step_map={NodeID("source-0"): 0},
            node_to_next={NodeID("source-0"): None},
            first_transform_node_id=None,
        )

        with pytest.raises(OrchestrationInvariantError, match="No effective sink for token"):
            processor.process_row(
                row_index=0,
                source_row=source_row,
                transforms=[],
                ctx=ctx,
            )


class TestWorkItemCoalesceInvariant:
    """WorkItem must carry complete coalesce metadata together."""

    def test_missing_coalesce_name_with_coalesce_node_id_raises(self) -> None:
        """Coalesce node without coalesce name is an invariant violation."""
        token = make_token_info(data={"value": 1})
        with pytest.raises(OrchestrationInvariantError, match="coalesce fields must be both set or both None"):
            WorkItem(
                token=token,
                current_node_id=NodeID("coalesce::merge"),
                coalesce_node_id=NodeID("coalesce::merge"),
                coalesce_name=None,
            )


# =============================================================================
# Aggregation facades (thin delegation tests)
# =============================================================================


class TestAggregationFacades:
    """Tests for the aggregation public facade methods."""

    def test_check_aggregation_timeout_delegates(self) -> None:
        """check_aggregation_timeout delegates to aggregation_executor."""
        _, recorder = _make_recorder()
        processor = _make_processor(recorder)

        with patch.object(
            processor._aggregation_executor,
            "check_flush_status",
            return_value=(True, TriggerType.TIMEOUT),
        ):
            should_flush, trigger = processor.check_aggregation_timeout(NodeID("agg-1"))

        assert should_flush is True
        assert trigger == TriggerType.TIMEOUT

    def test_get_aggregation_buffer_count_delegates(self) -> None:
        """get_aggregation_buffer_count delegates to aggregation_executor."""
        _, recorder = _make_recorder()
        processor = _make_processor(recorder)

        with patch.object(
            processor._aggregation_executor,
            "get_buffer_count",
            return_value=5,
        ):
            count = processor.get_aggregation_buffer_count(NodeID("agg-1"))

        assert count == 5

    def test_get_aggregation_checkpoint_state_delegates(self) -> None:
        """get_aggregation_checkpoint_state delegates to aggregation_executor."""
        _, recorder = _make_recorder()
        processor = _make_processor(recorder)

        checkpoint: dict[str, Any] = {"agg-1": {"buffer": [], "trigger_state": {}}}
        with patch.object(
            processor._aggregation_executor,
            "get_checkpoint_state",
            return_value=checkpoint,
        ):
            result = processor.get_aggregation_checkpoint_state()

        assert result == checkpoint


# =============================================================================
# Telemetry emission (optional)
# =============================================================================


class TestTelemetryEmission:
    """Tests for telemetry emission behavior."""

    def test_no_telemetry_manager_does_not_crash(self) -> None:
        """Without telemetry_manager, _emit_telemetry is a no-op."""
        _, recorder = _make_recorder()
        processor = _make_processor(recorder)
        # Should not raise
        processor._emit_telemetry(Mock())

    def test_telemetry_manager_receives_events(self) -> None:
        """With telemetry_manager, events are forwarded."""
        _, recorder = _make_recorder()

        telemetry = Mock()
        processor = _make_processor(recorder, telemetry_manager=telemetry)

        event = Mock()
        processor._emit_telemetry(event)
        telemetry.handle_event.assert_called_once_with(event)


# =============================================================================
# Regression: hscm.1 — Terminal deaggregation children inherit correct sink
# =============================================================================


class TestTerminalDeaggregationSinkRouting:
    """Regression tests for hscm.1: terminal deagg children must inherit the
    terminal transform's on_success sink, not source_on_success."""

    def test_terminal_deagg_children_use_transform_on_success_not_source(self) -> None:
        """Children of a terminal multi-row transform must route to transform's on_success."""
        _db, recorder = _make_recorder()
        source_row = _make_source_row()
        ctx = PluginContext(run_id="test-run", config={})

        contract = _make_contract()
        output_rows = [
            make_row({"value": 1}, contract=contract),
            make_row({"value": 2}, contract=contract),
        ]
        multi_result = TransformResult.success_multi(
            output_rows,
            success_reason={"action": "expand"},
        )

        # Key: transform on_success != source_on_success
        transform = _make_mock_transform(
            creates_tokens=True,
            on_success="transform_sink",
        )
        source_node = NodeID("source-0")
        transform_node = NodeID(transform.node_id)
        processor = _make_processor(
            recorder,
            source_on_success="source_sink",
            node_step_map={source_node: 0, transform_node: 1},
            node_to_next={source_node: transform_node, transform_node: None},
            first_transform_node_id=transform_node,
            node_to_plugin={transform_node: transform},
        )

        def executor_side_effect(*, transform, token, ctx, attempt=0):
            return (multi_result, token, None)

        with patch.object(
            processor._transform_executor,
            "execute_transform",
            side_effect=executor_side_effect,
        ):
            results = processor.process_row(
                row_index=0,
                source_row=source_row,
                transforms=[transform],
                ctx=ctx,
            )

        # Parent should be EXPANDED (no sink_name)
        expanded = [r for r in results if r.outcome == RowOutcome.EXPANDED]
        assert len(expanded) == 1

        # Children should be COMPLETED with transform's on_success sink
        completed = [r for r in results if r.outcome == RowOutcome.COMPLETED]
        assert len(completed) == 2
        for r in completed:
            assert r.sink_name == "transform_sink", (
                f"Expected 'transform_sink' but got '{r.sink_name}'. "
                f"Terminal deagg children must inherit the transform's on_success, "
                f"not source_on_success."
            )

    def test_mid_chain_deagg_children_process_through_remaining_transforms(self) -> None:
        """Mid-chain multi-row expansion: children continue to downstream transforms."""
        _db, recorder = _make_recorder()
        source_row = _make_source_row()
        ctx = PluginContext(run_id="test-run", config={})

        contract = _make_contract()
        output_rows = [
            make_row({"value": 10}, contract=contract),
            make_row({"value": 20}, contract=contract),
        ]
        multi_result = TransformResult.success_multi(
            output_rows,
            success_reason={"action": "expand"},
        )
        single_result = TransformResult.success(
            make_row({"value": 99}, contract=contract),
            success_reason={"action": "passthrough"},
        )

        # First transform expands, second is terminal
        expander = _make_mock_transform(
            node_id="expander-1",
            name="expander",
            creates_tokens=True,
            on_success=None,  # mid-chain, no on_success needed
        )
        terminal = _make_mock_transform(
            node_id="terminal-2",
            name="terminal",
            creates_tokens=False,
            on_success="final_sink",
        )

        source_node = NodeID("source-0")
        expander_node = NodeID("expander-1")
        terminal_node = NodeID("terminal-2")

        processor = _make_processor(
            recorder,
            source_on_success="source_sink",
            node_step_map={source_node: 0, expander_node: 1, terminal_node: 2},
            node_to_next={source_node: expander_node, expander_node: terminal_node, terminal_node: None},
            first_transform_node_id=expander_node,
            node_to_plugin={expander_node: expander, terminal_node: terminal},
        )

        call_count = 0

        def executor_side_effect(*, transform, token, ctx, attempt=0):
            nonlocal call_count
            call_count += 1
            if transform.name == "expander":
                return (multi_result, token, None)
            return (single_result, token, None)

        with patch.object(
            processor._transform_executor,
            "execute_transform",
            side_effect=executor_side_effect,
        ):
            results = processor.process_row(
                row_index=0,
                source_row=source_row,
                transforms=[expander, terminal],
                ctx=ctx,
            )

        # Should have 1 EXPANDED + 2 COMPLETED (children processed through terminal)
        completed = [r for r in results if r.outcome == RowOutcome.COMPLETED]
        assert len(completed) == 2
        for r in completed:
            assert r.sink_name == "final_sink"


# =============================================================================
# Regression: hscm.2 — Coalesce traversal invariant check
# =============================================================================


class TestCoalesceTraversalInvariant:
    """Regression tests for hscm.2: tokens with coalesce metadata must not
    start processing downstream of their coalesce point."""

    def test_work_item_downstream_of_coalesce_raises_invariant_error(self) -> None:
        """A work item starting past the coalesce node must raise OrchestrationInvariantError."""
        _db, recorder = _make_recorder()
        ctx = PluginContext(run_id="test-run", config={})

        # Build DAG: source → transform → coalesce → downstream
        source_node = NodeID("source-0")
        transform_node = NodeID("transform-1")
        coalesce_node = NodeID("coalesce-2")
        downstream_node = NodeID("downstream-3")

        # Register coalesce node for FK constraints
        recorder.register_node(
            run_id="test-run",
            plugin_name="coalesce",
            node_type=NodeType.COALESCE,
            plugin_version="1.0",
            config={},
            node_id="coalesce-2",
            schema_config=_DYNAMIC_SCHEMA,
        )
        recorder.register_node(
            run_id="test-run",
            plugin_name="downstream",
            node_type=NodeType.TRANSFORM,
            plugin_version="1.0",
            config={},
            node_id="downstream-3",
            schema_config=_DYNAMIC_SCHEMA,
        )

        transform = _make_mock_transform(
            node_id="downstream-3",
            name="downstream",
            on_success="output",
        )

        processor = _make_processor(
            recorder,
            node_step_map={
                source_node: 0,
                transform_node: 1,
                coalesce_node: 2,
                downstream_node: 3,
            },
            node_to_next={
                source_node: transform_node,
                transform_node: coalesce_node,
                coalesce_node: downstream_node,
                downstream_node: None,
            },
            first_transform_node_id=transform_node,
            node_to_plugin={downstream_node: transform},
            coalesce_node_ids={CoalesceName("merge"): coalesce_node},
            coalesce_on_success_map={CoalesceName("merge"): "output"},
        )

        # Create a malformed work item starting PAST the coalesce node
        token = TokenInfo(
            row_id="row-1",
            token_id="tok-1",
            row_data=make_row({"value": 1}),
            branch_name="path_a",
        )
        with pytest.raises(OrchestrationInvariantError, match="downstream of coalesce"):
            processor._process_single_token(
                token=token,
                ctx=ctx,
                current_node_id=downstream_node,  # step 3 > coalesce step 2
                coalesce_node_id=coalesce_node,
                coalesce_name=CoalesceName("merge"),
            )

    def test_work_item_at_coalesce_does_not_raise(self) -> None:
        """A work item starting exactly at the coalesce node should not raise."""
        _db, recorder = _make_recorder()
        ctx = PluginContext(run_id="test-run", config={})

        source_node = NodeID("source-0")
        coalesce_node = NodeID("coalesce-1")

        recorder.register_node(
            run_id="test-run",
            plugin_name="coalesce",
            node_type=NodeType.COALESCE,
            plugin_version="1.0",
            config={},
            node_id="coalesce-1",
            schema_config=_DYNAMIC_SCHEMA,
        )

        processor = _make_processor(
            recorder,
            source_on_success="output",
            node_step_map={source_node: 0, coalesce_node: 1},
            node_to_next={source_node: coalesce_node, coalesce_node: None},
            first_transform_node_id=coalesce_node,
            coalesce_node_ids={CoalesceName("merge"): coalesce_node},
            coalesce_on_success_map={CoalesceName("merge"): "output"},
        )

        token = TokenInfo(
            row_id="row-1",
            token_id="tok-1",
            row_data=make_row({"value": 1}),
            branch_name="path_a",
        )
        # Should not raise — at coalesce node, not past it.
        # Without coalesce_executor, coalesce handling is skipped (returns False, None)
        # and the token completes normally. The invariant check only validates
        # traversal ordering, not coalesce executor presence.
        result, _ = processor._process_single_token(
            token=token,
            ctx=ctx,
            current_node_id=coalesce_node,
            coalesce_node_id=coalesce_node,
            coalesce_name=CoalesceName("merge"),
        )
        assert result is not None


class TestTerminalWorkItemInvariant:
    """Tests for current_node_id=None work-item validation."""

    def test_none_current_node_without_sink_context_raises(self) -> None:
        """None current_node_id must not default to source_on_success silently."""
        _db, recorder = _make_recorder()
        ctx = PluginContext(run_id="test-run", config={})
        processor = _make_processor(recorder, source_on_success="source_sink")
        token = make_token_info(data={"value": 1})

        with pytest.raises(OrchestrationInvariantError, match="current_node_id=None"):
            processor._process_single_token(
                token=token,
                ctx=ctx,
                current_node_id=None,
            )

    def test_none_current_node_with_inherited_sink_is_allowed(self) -> None:
        """Explicit on_success_sink context allows terminal completion with None node."""
        _db, recorder = _make_recorder()
        ctx = PluginContext(run_id="test-run", config={})
        processor = _make_processor(recorder, source_on_success="source_sink")
        token = make_token_info(data={"value": 1})

        result, _child_items = processor._process_single_token(
            token=token,
            ctx=ctx,
            current_node_id=None,
            on_success_sink="terminal_sink",
        )

        assert result is not None
        assert not isinstance(result, list)
        assert result.outcome == RowOutcome.COMPLETED
        assert result.sink_name == "terminal_sink"
