# tests/unit/engine/test_processor.py
"""Unit tests for RowProcessor — the DAG execution state machine.

processor.py orchestrates row processing through transforms, gates, and
aggregation. It is the largest file in the engine (~2,000 lines) and the
most critical path for correctness.

Test strategy:
- Use LandscapeDB.in_memory() for real audit recording (no mock factory)
- Use real SpanFactory (no tracer — no-op spans)
- Mock transforms/gates to control routing outcomes
- Verify outcomes via RowResult assertions

This avoids the anti-pattern of testing mocks instead of behavior.
"""

from __future__ import annotations

import hashlib
from contextlib import nullcontext
from datetime import UTC, datetime, timedelta
from types import SimpleNamespace
from typing import Any
from unittest.mock import Mock, create_autospec, patch

import pytest

# For node registration
from elspeth.contracts import NodeType, RouteDestination, RowResult, SourceRow, TokenInfo, TransformProtocol, TransformResult
from elspeth.contracts.audit import TokenRef
from elspeth.contracts.declaration_contracts import _attach_contract_name_from_dispatcher
from elspeth.contracts.enums import (
    BatchStatus,
    NodeStateStatus,
    TerminalOutcome,
    TerminalPath,
    TriggerType,
)
from elspeth.contracts.errors import (
    AuditIntegrityError,
    CapacityError,
    FrameworkBugError,
    MaxRetriesExceeded,
    OrchestrationInvariantError,
    SourceGuaranteedFieldsViolation,
)
from elspeth.contracts.results import FailureInfo, GateResult
from elspeth.contracts.routing import RoutingAction
from elspeth.contracts.schema import SchemaConfig
from elspeth.contracts.schema_contract import SchemaContract
from elspeth.contracts.types import BranchName, CoalesceName, GateName, NodeID, SinkName
from elspeth.core.checkpoint.recovery import IncompleteTokenSpec
from elspeth.core.config import AggregationSettings, GateSettings
from elspeth.core.landscape import LandscapeDB
from elspeth.core.landscape.errors import LandscapeRecordError
from elspeth.core.landscape.factory import RecorderFactory
from elspeth.engine.clock import MockClock
from elspeth.engine.coalesce_executor import CoalesceExecutor, CoalesceOutcome
from elspeth.engine.dag_navigator import WorkItem
from elspeth.engine.executors import GateOutcome
from elspeth.engine.executors.transform import TransformExecutor
from elspeth.engine.orchestrator.types import TelemetryManagerProtocol
from elspeth.engine.processor import (
    MAX_WORK_QUEUE_ITERATIONS,
    SCHEDULER_MAINTENANCE_INTERVAL,
    BarrierJournalRestoreContext,
    DAGTraversalContext,
    RowProcessor,
    _FlushContext,
    _LiveBarrierHold,
)
from elspeth.engine.retry import RetryManager
from elspeth.engine.spans import SpanFactory
from elspeth.plugins.infrastructure.clients.llm import LLMClientError
from elspeth.plugins.transforms.batch_replicate import BatchReplicateConfig
from elspeth.testing import make_contract, make_pipeline_row, make_row, make_source_row, make_token_info
from tests.fixtures.factories import make_context
from tests.fixtures.landscape import leader_coordination_token, make_recorder_with_run

# =============================================================================
# Helpers
# =============================================================================

_DYNAMIC_SCHEMA = SchemaConfig.from_dict({"mode": "observed"})
_DEFAULT_SCHEDULER = object()
_TEST_LEADER_WORKER_ID = "seeder"


def _make_contract() -> SchemaContract:
    """Create a minimal schema contract for testing."""
    return make_contract(fields={"value": int}, mode="OBSERVED")


def _make_source_row(data: dict[str, Any] | None = None) -> SourceRow:
    """Create a valid SourceRow with contract."""
    contract = _make_contract()
    return make_source_row(data or {"value": 42}, contract=contract)


def test_aggregation_restore_plan_freezes_sequence_fields() -> None:
    from elspeth.contracts.barrier_scalars import AggregationNodeScalars
    from elspeth.engine.barrier_coordination import _AggregationRestorePlan

    plan = _AggregationRestorePlan(
        node_id=NodeID("agg"),
        items=[],
        member_order=["left", "right"],
        batch_id="batch-1",
        accepted_count_total=2,
        completed_flush_count=0,
        scalars=AggregationNodeScalars(count_fire_offset=None, condition_fire_offset=None),
    )

    assert plan.items == ()
    assert plan.member_order == ("left", "right")


def _persist_token_for_scheduler(
    factory: RecorderFactory,
    token: TokenInfo,
    *,
    run_id: str = "test-run",
    source_node_id: str = "source-0",
    row_index: int | None = None,
    source_row_index: int | None = None,
    ingest_sequence: int = 0,
) -> None:
    """Persist a fabricated unit-test token before scheduler-backed processing."""
    try:
        factory.data_flow.resolve_row_ingest_sequence(token.row_id)
    except AuditIntegrityError:
        factory.data_flow.create_row(
            run_id=run_id,
            source_node_id=source_node_id,
            row_index=ingest_sequence if row_index is None else row_index,
            source_row_index=ingest_sequence if source_row_index is None else source_row_index,
            ingest_sequence=ingest_sequence,
            row_id=token.row_id,
            data=token.row_data.to_dict(),
        )
    if factory.query.get_token(token.token_id) is None:
        factory.data_flow.create_token(
            token.row_id,
            token_id=token.token_id,
            branch_name=token.branch_name,
            fork_group_id=token.fork_group_id or (f"test-fork:{token.row_id}" if token.branch_name is not None else None),
            join_group_id=token.join_group_id,
        )


def _persist_blocked_scheduler_work(
    factory: RecorderFactory,
    processor: RowProcessor,
    token: TokenInfo,
    *,
    node_id: NodeID,
    barrier_key: str,
    ingest_sequence: int = 0,
    adopted: bool = True,
    coalesce_name: str | None = None,
) -> None:
    """Persist BLOCKED scheduler work matching a fabricated buffered token.

    Uses the production journal verbs (enqueue+claim, then mark_blocked) so
    the BLOCKED row carries ``barrier_blocked_at`` exactly as a live barrier
    hold would (F1: the journal is the only barrier-buffer truth).

    ``adopted=True`` (the default) stamps ``barrier_adopted_epoch`` — the
    post-adoption journal image (ADR-030 §E.2). Tests that fabricate executor
    memory directly need the adopted image so the journal-first intake does
    not try to re-adopt (and re-feed) the row; pass ``adopted=False`` to
    fabricate an intake-pending deposit instead.
    """
    _persist_token_for_scheduler(factory, token, ingest_sequence=ingest_sequence)
    now = processor._clock.now_utc()
    item = processor._scheduler.enqueue_ready_claimed(
        run_id=processor.run_id,
        token_id=token.token_id,
        row_id=token.row_id,
        node_id=str(node_id),
        step_index=processor.resolve_node_step(node_id),
        ingest_sequence=factory.data_flow.resolve_row_ingest_sequence(token.row_id),
        row_payload_json=processor._scheduler.serialize_row_payload(token.row_data),
        available_at=now,
        lease_owner="test-harness",
        lease_seconds=60,
        now=now,
        branch_name=token.branch_name,
        fork_group_id=token.fork_group_id,
        join_group_id=token.join_group_id,
        expand_group_id=token.expand_group_id,
        coalesce_name=coalesce_name,
    )
    processor._scheduler.mark_blocked(
        work_item_id=item.work_item_id,
        queue_key=None,
        barrier_key=barrier_key,
        now=now,
        expected_lease_owner="test-harness",
    )
    if adopted:
        _stamp_blocked_rows_adopted(processor._scheduler, work_item_id=item.work_item_id)


def _stamp_blocked_rows_adopted(
    scheduler: Any,
    *,
    run_id: str | None = None,
    work_item_id: str | None = None,
    epoch: int = 1,
) -> None:
    """Stamp ``barrier_adopted_epoch`` on fabricated BLOCKED rows.

    ADR-030 §E.2 (slice 3): tests that fabricate the post-adoption journal
    image directly (BLOCKED rows + batch_members + BUFFERED outcomes, or
    coalesce holds + node_states) must carry the adoption marker — the
    journal restore restores ONLY adopted rows (intake-pending rows have no
    audit writes to restore and are left for the journal-first intake).
    """
    from sqlalchemy import update as _sa_update

    from elspeth.core.landscape.schema import token_work_items_table as _twi

    stmt = _sa_update(_twi).values(barrier_adopted_epoch=epoch)
    if work_item_id is not None:
        stmt = stmt.where(_twi.c.work_item_id == work_item_id)
    if run_id is not None:
        stmt = stmt.where(_twi.c.run_id == run_id)
    stmt = stmt.where(_twi.c.status == "blocked")
    with scheduler._engine.begin() as conn:
        conn.execute(stmt)


def _assert_outcome_pair(
    result: RowResult,
    outcome: TerminalOutcome | None,
    path: TerminalPath,
) -> None:
    assert result.outcome == outcome
    assert result.path == path


def _make_factory(
    *,
    run_id: str = "test-run",
    source_node_id: str = "source-0",
) -> tuple[LandscapeDB, RecorderFactory]:
    """Create an in-memory LandscapeDB with run and source node registered.

    This satisfies FK constraints: rows table references nodes(node_id, run_id).
    """
    setup = make_recorder_with_run(
        run_id=run_id,
        source_node_id=source_node_id,
        source_plugin_name="test-source",
        leader_worker_id=_TEST_LEADER_WORKER_ID,
    )
    return setup.db, setup.factory


def _register_test_worker(
    factory: RecorderFactory,
    worker_id: str,
    *,
    run_id: str = "test-run",
    heartbeat_expires_at: datetime | None = None,
) -> None:
    """Register a scheduler lease owner used by direct test claims."""
    from sqlalchemy import insert, select

    from elspeth.core.landscape.schema import run_workers_table

    with factory._db.engine.begin() as conn:
        exists = conn.execute(
            select(run_workers_table.c.worker_id)
            .where(run_workers_table.c.worker_id == worker_id)
            .where(run_workers_table.c.run_id == run_id)
        ).first()
        if exists is not None:
            return
        registered_at = datetime.now(UTC)
        conn.execute(
            insert(run_workers_table).values(
                worker_id=worker_id,
                run_id=run_id,
                role="follower",
                status="active",
                registered_at=registered_at,
                heartbeat_expires_at=heartbeat_expires_at or registered_at + timedelta(hours=1),
            )
        )


def _make_processor(
    factory: RecorderFactory,
    *,
    run_id: str = "test-run",
    source_node_id: str = "source-0",
    source_on_success: str = "default",
    source_plugin: Any | None = None,
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
    barrier_restore: BarrierJournalRestoreContext | None = None,
    telemetry_manager: Any = None,
    sink_names: frozenset[str] | None = None,
    scheduler: Any = _DEFAULT_SCHEDULER,
    scheduler_lease_owner: str | None = None,
    clock: Any = None,
    stamp_blocked_rows_adopted: bool = True,
    structural_node_ids: frozenset[NodeID] | None = None,
) -> RowProcessor:
    """Create a RowProcessor with sensible defaults."""
    if scheduler_lease_owner is not None:
        _register_test_worker(factory, scheduler_lease_owner, run_id=run_id)

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

    node_ids_to_register: set[NodeID] = set(traversal_steps)
    node_ids_to_register.update(traversal_node_to_plugin)
    node_ids_to_register.update(traversal_next)
    node_ids_to_register.update(node_id for node_id in traversal_next.values() if node_id is not None)
    node_ids_to_register.update(coalesce_nodes.values())
    node_ids_to_register.update((aggregation_settings or {}).keys())

    for node_id in sorted(node_ids_to_register, key=str):
        if node_id == source_node:
            continue
        if factory.data_flow.get_node(str(node_id), run_id) is not None:
            continue
        plugin = traversal_node_to_plugin.get(node_id)
        if node_id in coalesce_nodes.values():
            node_type = NodeType.COALESCE
            plugin_name = "coalesce"
        elif node_id in (aggregation_settings or {}):
            node_type = NodeType.AGGREGATION
            plugin_name = "aggregation"
        elif isinstance(plugin, GateSettings):
            node_type = NodeType.GATE
            plugin_name = "gate"
        else:
            node_type = NodeType.TRANSFORM
            plugin_name = getattr(plugin, "name", "transform")
        factory.data_flow.register_node(
            run_id=run_id,
            plugin_name=str(plugin_name),
            node_type=node_type,
            plugin_version="1.0",
            config={},
            node_id=str(node_id),
            schema_config=_DYNAMIC_SCHEMA,
        )

    # Sources are structural in production (graph_wiring's type-based
    # allowlist); coalesce nodes are auto-unioned by the context itself.
    traversal_structural = structural_node_ids if structural_node_ids is not None else frozenset({NodeID(source_node_id)})
    traversal = DAGTraversalContext(
        node_step_map=traversal_steps,
        node_to_plugin=traversal_node_to_plugin,
        node_to_next=traversal_next,
        coalesce_node_map=coalesce_nodes,
        structural_node_ids=traversal_structural,
    )

    if barrier_restore is not None and stamp_blocked_rows_adopted:
        # ADR-030 §E.2: restore-shaped tests fabricate the post-adoption
        # journal image (BLOCKED rows + the audit writes adoption owns), so
        # stamp the adoption marker — the journal restore restores ONLY
        # adopted rows. Tests exercising intake-pending restores (rows the
        # dead leader deposited but never adopted) pass
        # ``stamp_blocked_rows_adopted=False`` and stamp selectively.
        _stamp_blocked_rows_adopted(factory.scheduler, run_id=run_id)

    return RowProcessor(
        execution=factory.execution,
        data_flow=factory.data_flow,
        span_factory=SpanFactory(),  # No tracer — no-op spans
        run_id=run_id,
        # Slice 3 (ADR-030 §E.2): the journal-first barrier intake's adoption
        # verbs are leader-fenced; bind the run's own epoch-1 seat token.
        coordination_token=leader_coordination_token(factory, run_id),
        source_node_id=NodeID(source_node_id),
        source_on_success=source_on_success,
        source_plugin=source_plugin,
        traversal=traversal,
        edge_map=edge_map,
        route_resolution_map=route_resolution_map,
        aggregation_settings=aggregation_settings,
        retry_manager=retry_manager,
        coalesce_executor=coalesce_executor,
        branch_to_coalesce=branch_to_coalesce,
        branch_to_sink={BranchName(k): SinkName(v) for k, v in (branch_to_sink or {}).items()},
        coalesce_on_success_map=coalesce_on_success_map,
        barrier_restore=barrier_restore,
        telemetry_manager=telemetry_manager,
        sink_names=sink_names,
        scheduler=factory.scheduler if scheduler is _DEFAULT_SCHEDULER else scheduler,
        scheduler_lease_owner=scheduler_lease_owner,
        clock=clock,
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
    transform.declared_output_fields = frozenset()
    transform.declared_input_fields = frozenset()
    # ADR-009 §Clause 2: cross-check is gated on this flag. Default False
    # matches BaseTransform's default; tests opting into pass-through coverage
    # set it explicitly after construction.
    transform.passes_through_input = False
    transform.can_drop_rows = False
    transform._output_schema_config = None
    transform.effective_static_contract.return_value = frozenset()
    if result is not None:
        transform.process.return_value = result
    return transform


# =============================================================================
# Constructor: Error Edge Map
# =============================================================================


class TestConstructorErrorEdgeMap:
    """Tests for error edge map construction in __init__."""

    def test_scheduler_repository_is_required(self) -> None:
        """RowProcessor must not expose the legacy in-memory drain path."""
        _, factory = _make_factory()

        with pytest.raises(TypeError, match="scheduler"):
            _make_processor(factory, scheduler=None)

    def test_default_scheduler_lease_owner_uses_coordination_token(self) -> None:
        """A processor with a coordination token uses its registered worker identity."""
        _, factory = _make_factory()

        first = _make_processor(factory, scheduler=factory.scheduler)
        second = _make_processor(factory, scheduler=factory.scheduler)

        assert first._scheduler_lease_owner == _TEST_LEADER_WORKER_ID
        assert second._scheduler_lease_owner == _TEST_LEADER_WORKER_ID

    def test_explicit_scheduler_lease_owner_is_honored(self) -> None:
        """Tests and controlled workers can still provide a stable lease-owner identity."""
        _, factory = _make_factory()

        processor = _make_processor(factory, scheduler=factory.scheduler, scheduler_lease_owner="worker-a")

        assert processor._scheduler_lease_owner == "worker-a"

    def test_navigator_is_constructed_from_traversal_context_factory(self) -> None:
        """RowProcessor should not re-derive DAGNavigator internals at the call site."""
        _, factory = _make_factory()
        nav = SimpleNamespace(create_work_item=Mock(name="create_work_item"))
        coalesce_on_success = {CoalesceName("merge"): "out"}
        sink_names = frozenset({"out", "error"})

        with patch("elspeth.engine.processor.DAGNavigator.from_traversal_context", return_value=nav) as from_traversal:
            processor = _make_processor(
                factory,
                scheduler=factory.scheduler,
                coalesce_on_success_map=coalesce_on_success,
                sink_names=sink_names,
            )

        assert processor._nav is nav
        from_traversal.assert_called_once()
        assert from_traversal.call_args.args == (processor._traversal,)
        assert from_traversal.call_args.kwargs == {
            "coalesce_on_success_map": coalesce_on_success,
            "sink_names": sink_names,
        }

    def test_fresh_row_drains_throttle_empty_scheduler_maintenance(self) -> None:
        """Fresh source-row drains should not run empty recovery sweeps per row."""
        _, factory = _make_factory()
        processor = _make_processor(factory, scheduler=factory.scheduler)
        ctx = make_context(landscape=factory.plugin_audit_writer())

        with (
            # Slice 3 re-pin (token-bound fixture): a coordination-token-bound
            # processor ingests rows via the fenced composed verb (ADR-030
            # §C.4 row 9), not the unfenced enqueue_ready_claimed arm.
            patch.object(
                factory.scheduler, "ingest_row_with_initial_claim", wraps=factory.scheduler.ingest_row_with_initial_claim
            ) as fenced_ingest,
            patch.object(factory.scheduler, "claim_ready", wraps=factory.scheduler.claim_ready) as claim_ready,
            patch.object(factory.scheduler, "recover_expired_leases", wraps=factory.scheduler.recover_expired_leases) as recover_expired,
        ):
            for idx in range(SCHEDULER_MAINTENANCE_INTERVAL - 1):
                processor.process_row(
                    row_index=idx,
                    source_row=_make_source_row({"value": idx}),
                    transforms=[],
                    ctx=ctx,
                    source_row_index=idx,
                    ingest_sequence=idx,
                )

            assert recover_expired.call_count == 0

            processor.process_row(
                row_index=SCHEDULER_MAINTENANCE_INTERVAL - 1,
                source_row=_make_source_row({"value": SCHEDULER_MAINTENANCE_INTERVAL - 1}),
                transforms=[],
                ctx=ctx,
                source_row_index=SCHEDULER_MAINTENANCE_INTERVAL - 1,
                ingest_sequence=SCHEDULER_MAINTENANCE_INTERVAL - 1,
            )

        assert recover_expired.call_count == 1
        assert fenced_ingest.call_count == SCHEDULER_MAINTENANCE_INTERVAL
        assert claim_ready.call_count == 0

    def test_lease_recovered_work_offsets_transform_attempt_identity(self) -> None:
        """Recovered scheduler attempts must not replay node state attempt 0."""
        _, factory = _make_factory()
        processor = _make_processor(factory)
        transform = _make_mock_transform()
        token = make_token_info(row_id="row-1", token_id="token-1", data={"value": 1})
        ctx = make_context(landscape=factory.plugin_audit_writer())
        executor = create_autospec(TransformExecutor, instance=True)
        executor.execute_transform.return_value = (
            TransformResult.success(make_pipeline_row({"value": 1}), success_reason={"action": "test"}),
            token,
            None,
        )
        processor._transform_executor = executor

        processor._execute_transform_with_retry(transform, token, ctx, attempt_offset=1)

        assert executor.execute_transform.call_args.kwargs["attempt"] == 1

    def test_extracts_error_edges_from_edge_map(self) -> None:
        """Error edges (labels like __error_0__) are extracted into error_edge_ids."""
        _, factory = _make_factory()
        edge_map = {
            (NodeID("t1"), "continue"): "edge-1",
            (NodeID("t1"), "__error_0__"): "error-edge-1",
            (NodeID("t2"), "continue"): "edge-2",
            (NodeID("t2"), "__error_1__"): "error-edge-2",
        }
        processor = _make_processor(factory, edge_map=edge_map)
        assert processor._error_edge_ids == {
            NodeID("t1"): "error-edge-1",
            NodeID("t2"): "error-edge-2",
        }

    def test_empty_edge_map_produces_no_error_edges(self) -> None:
        """No edge_map means no error edges."""
        _, factory = _make_factory()
        processor = _make_processor(factory)
        assert processor._error_edge_ids == {}

    def test_non_error_labels_ignored(self) -> None:
        """Only __error_N__ labels are extracted; other labels are ignored."""
        _, factory = _make_factory()
        edge_map = {
            (NodeID("t1"), "continue"): "edge-1",
            (NodeID("t1"), "route_to_sink"): "edge-2",
            (NodeID("t1"), "fork_path_a"): "edge-3",
        }
        processor = _make_processor(factory, edge_map=edge_map)
        assert processor._error_edge_ids == {}

    def test_resume_restores_aggregation_buffers_from_blocked_rows(self) -> None:
        """F1 restore inversion: resume rebuilds aggregation buffers FROM journal BLOCKED rows.

        Seeds the journal (2 BLOCKED rows under barrier_key "agg-1", stamped
        via the production claim->mark_blocked path) and the audit trail
        (batch + members, BUFFERED token_outcomes, node_states at attempt 0),
        then builds a resuming processor and asserts the buffers, batch id,
        counters, trigger latch and attempt offsets all derive correctly —
        with NO new journal rows created (the BLOCKED rows are reused as-is).
        """
        from sqlalchemy import func, select

        from elspeth.contracts.barrier_scalars import AggregationNodeScalars, BarrierScalars
        from elspeth.core.landscape.schema import token_work_items_table

        db, factory = _make_factory()
        agg_node = NodeID("agg-1")
        factory.data_flow.register_node(
            run_id="test-run",
            plugin_name="agg-transform",
            node_type=NodeType.AGGREGATION,
            plugin_version="1.0",
            config={},
            node_id=str(agg_node),
            schema_config=_DYNAMIC_SCHEMA,
        )
        now = datetime.now(UTC)
        batch = factory.execution.create_batch(run_id="test-run", aggregation_node_id=str(agg_node))
        tokens: list[TokenInfo] = []
        for ordinal, token_id in enumerate(["t1", "t2"]):
            payload = make_row({"value": ordinal})
            token = TokenInfo(row_id=f"row-{ordinal}", token_id=token_id, row_data=payload)
            _persist_token_for_scheduler(factory, token, ingest_sequence=ordinal)
            tokens.append(token)
            # Audit: membership + BUFFERED outcome + a node_state at attempt 0.
            factory.execution.add_batch_member(batch_id=batch.batch_id, token_id=token_id, ordinal=ordinal)
            factory.data_flow.record_token_outcome(
                ref=TokenRef(token_id=token_id, run_id="test-run"),
                outcome=None,
                path=TerminalPath.BUFFERED,
                batch_id=batch.batch_id,
            )
            factory.execution.begin_node_state(
                token_id=token_id,
                node_id=str(agg_node),
                run_id="test-run",
                step_index=1,
                input_data=payload.to_dict(),
                attempt=0,
            )
            # Journal: production-shaped BLOCKED row (mark_blocked stamps
            # barrier_blocked_at; the old ensure_blocked path did not).
            factory.scheduler.enqueue_ready(
                run_id="test-run",
                token_id=token_id,
                row_id=token.row_id,
                node_id=str(agg_node),
                step_index=1,
                ingest_sequence=ordinal,
                row_payload_json=factory.scheduler.serialize_row_payload(payload),
                available_at=now,
            )
            claimed = factory.scheduler.claim_ready(run_id="test-run", lease_owner="seeder", lease_seconds=60, now=now)
            assert claimed is not None and claimed.token_id == token_id
            factory.scheduler.mark_blocked(
                work_item_id=claimed.work_item_id,
                queue_key=None,
                barrier_key=str(agg_node),
                now=now,
                expected_lease_owner="seeder",
            )

        with db.connection() as conn:
            rows_before = conn.execute(select(func.count()).select_from(token_work_items_table)).scalar_one()

        processor = _make_processor(
            factory,
            aggregation_settings={
                agg_node: AggregationSettings(
                    name="test-agg",
                    plugin="test-plugin",
                    input="agg_in",
                    on_error="discard",
                    trigger={"count": 3},
                ),
            },
            barrier_restore=BarrierJournalRestoreContext(
                resume_checkpoint_id="ckpt-resume-1",
                barrier_scalars=BarrierScalars(
                    aggregation={str(agg_node): AggregationNodeScalars(count_fire_offset=1.5, condition_fire_offset=None)},
                    coalesce={},
                ),
                batch_id_remap={},
            ),
        )

        node = processor._aggregation_executor._nodes[agg_node]
        assert [t.token_id for t in node.tokens] == ["t1", "t2"]
        assert node.batch_id == batch.batch_id
        assert [t.row_data.to_dict() for t in node.tokens] == [{"value": 0}, {"value": 1}]
        assert node.accepted_count_total == 2
        assert node.completed_flush_count == 0
        assert all(t.resume_attempt_offset == 1 for t in node.tokens)  # max_attempt(0)+1
        assert all(t.resume_checkpoint_id == "ckpt-resume-1" for t in node.tokens)
        # Checkpoint scalars (trigger latch) round-trip through the restore.
        assert node.trigger.get_count_fire_offset() == pytest.approx(1.5)
        # NO new journal rows: the BLOCKED rows are reused as-is.
        with db.connection() as conn:
            rows_after = conn.execute(select(func.count()).select_from(token_work_items_table)).scalar_one()
        assert rows_after == rows_before

    def test_resume_restore_raises_on_orphan_barrier_key(self) -> None:
        """A BLOCKED row whose barrier_key matches no coalesce or aggregation refuses resume."""
        _, factory = _make_factory()
        payload = make_row({"value": 7})
        token = TokenInfo(row_id="row-ghost", token_id="tok-ghost", row_data=payload)
        _persist_token_for_scheduler(factory, token, ingest_sequence=0)
        ghost_now = datetime.now(UTC)
        ghost_item = factory.scheduler.enqueue_ready_claimed(
            run_id="test-run",
            token_id="tok-ghost",
            row_id="row-ghost",
            node_id=None,
            step_index=0,
            ingest_sequence=0,
            row_payload_json=factory.scheduler.serialize_row_payload(payload),
            available_at=ghost_now,
            lease_owner="test-harness",
            lease_seconds=60,
            now=ghost_now,
        )
        factory.scheduler.mark_blocked(
            work_item_id=ghost_item.work_item_id,
            queue_key=None,
            barrier_key="ghost-barrier",
            now=ghost_now,
            expected_lease_owner="test-harness",
        )

        with pytest.raises(AuditIntegrityError, match="orphan barrier_key 'ghost-barrier'"):
            _make_processor(
                factory,
                barrier_restore=BarrierJournalRestoreContext(
                    resume_checkpoint_id="ckpt-resume-1",
                    barrier_scalars=None,
                    batch_id_remap={},
                ),
            )

    def test_resume_restore_ignores_adr028_queue_hold_rows(self) -> None:
        """An ADR-028 queue-hold (BLOCKED, queue_key set, barrier_key NULL) is not a barrier member.

        The queue resume path is untouched by F1: the row must be neither
        restored into a barrier buffer nor counted anywhere, and must stay
        BLOCKED on its queue_key.
        """
        from sqlalchemy import select

        from elspeth.core.landscape.schema import token_work_items_table

        db, factory = _make_factory()
        agg_node = NodeID("agg-1")
        now = datetime.now(UTC)
        payload = make_row({"value": 9})
        token = TokenInfo(row_id="row-q", token_id="tok-q", row_data=payload)
        _persist_token_for_scheduler(factory, token, ingest_sequence=0)
        factory.scheduler.enqueue_ready(
            run_id="test-run",
            token_id="tok-q",
            row_id="row-q",
            node_id=None,
            step_index=0,
            ingest_sequence=0,
            row_payload_json=factory.scheduler.serialize_row_payload(payload),
            available_at=now,
        )
        claimed = factory.scheduler.claim_ready(run_id="test-run", lease_owner="seeder", lease_seconds=60, now=now)
        assert claimed is not None
        factory.scheduler.mark_blocked(
            work_item_id=claimed.work_item_id,
            queue_key="queue-1",
            barrier_key=None,
            now=now,
            expected_lease_owner="seeder",
        )

        processor = _make_processor(
            factory,
            aggregation_settings={
                agg_node: AggregationSettings(
                    name="test-agg",
                    plugin="test-plugin",
                    input="agg_in",
                    on_error="discard",
                    trigger={"count": 3},
                ),
            },
            barrier_restore=BarrierJournalRestoreContext(
                resume_checkpoint_id="ckpt-resume-1",
                barrier_scalars=None,
                batch_id_remap={},
            ),
        )

        node = processor._aggregation_executor._nodes[agg_node]
        assert node.tokens == []
        assert node.batch_id is None
        assert node.accepted_count_total == 0
        assert node.completed_flush_count == 0
        with db.connection() as conn:
            status, queue_key, barrier_key = conn.execute(
                select(
                    token_work_items_table.c.status,
                    token_work_items_table.c.queue_key,
                    token_work_items_table.c.barrier_key,
                ).where(token_work_items_table.c.token_id == "tok-q")
            ).one()
        assert (status, queue_key, barrier_key) == ("blocked", "queue-1", None)

    # -- shared seeding for the batch_id-derivation arms ---------------------

    @staticmethod
    def _agg_settings(agg_node: NodeID) -> dict[NodeID, AggregationSettings]:
        return {
            agg_node: AggregationSettings(
                name="test-agg",
                plugin="test-plugin",
                input="agg_in",
                on_error="discard",
                trigger={"count": 3},
            ),
        }

    @staticmethod
    def _restore_ctx(batch_id_remap: dict[str, str] | None = None) -> BarrierJournalRestoreContext:
        return BarrierJournalRestoreContext(
            resume_checkpoint_id="ckpt-resume-1",
            barrier_scalars=None,
            batch_id_remap=batch_id_remap or {},
        )

    @staticmethod
    def _register_aggregation_node(factory: RecorderFactory, agg_node: NodeID) -> None:
        factory.data_flow.register_node(
            run_id="test-run",
            plugin_name="agg-transform",
            node_type=NodeType.AGGREGATION,
            plugin_version="1.0",
            config={},
            node_id=str(agg_node),
            schema_config=_DYNAMIC_SCHEMA,
        )

    @staticmethod
    def _seed_blocked_agg_row(factory: RecorderFactory, *, token_id: str, ordinal: int, agg_node: NodeID) -> TokenInfo:
        """Token + production-shaped BLOCKED journal row under the agg barrier.

        Goes through enqueue -> claim -> mark_blocked so barrier_blocked_at is
        stamped exactly as the live buffering path stamps it.
        """
        payload = make_row({"value": ordinal})
        token = TokenInfo(row_id=f"row-{ordinal}", token_id=token_id, row_data=payload)
        _persist_token_for_scheduler(factory, token, ingest_sequence=ordinal)
        now = datetime.now(UTC)
        factory.scheduler.enqueue_ready(
            run_id="test-run",
            token_id=token_id,
            row_id=token.row_id,
            node_id=str(agg_node),
            step_index=1,
            ingest_sequence=ordinal,
            row_payload_json=factory.scheduler.serialize_row_payload(payload),
            available_at=now,
        )
        claimed = factory.scheduler.claim_ready(run_id="test-run", lease_owner="seeder", lease_seconds=60, now=now)
        assert claimed is not None and claimed.token_id == token_id
        factory.scheduler.mark_blocked(
            work_item_id=claimed.work_item_id,
            queue_key=None,
            barrier_key=str(agg_node),
            now=now,
            expected_lease_owner="seeder",
        )
        return token

    def _seed_buffered_member(
        self,
        factory: RecorderFactory,
        *,
        token_id: str,
        ordinal: int,
        agg_node: NodeID,
        batch_id: str,
    ) -> TokenInfo:
        """BLOCKED journal row + batch membership + BUFFERED outcome carrying batch_id."""
        token = self._seed_blocked_agg_row(factory, token_id=token_id, ordinal=ordinal, agg_node=agg_node)
        factory.execution.add_batch_member(batch_id=batch_id, token_id=token_id, ordinal=ordinal)
        factory.data_flow.record_token_outcome(
            ref=TokenRef(token_id=token_id, run_id="test-run"),
            outcome=None,
            path=TerminalPath.BUFFERED,
            batch_id=batch_id,
        )
        return token

    def test_resume_restore_reads_batch_id_through_incomplete_batch_remap(self) -> None:
        """A flush-interrupting crash: BUFFERED outcomes carry the DEAD batch id.

        handle_incomplete_batches retries the failed batch (members copied) and
        returns the old->retry mapping; the restore must read each group's
        batch_id THROUGH that remap, land on the retry batch, and count
        accepted tokens DISTINCT-ly (the retry copies would double-count).
        """
        _db, factory = _make_factory()
        agg_node = NodeID("agg-1")
        self._register_aggregation_node(factory, agg_node)
        old_batch = factory.execution.create_batch(run_id="test-run", aggregation_node_id=str(agg_node))
        for ordinal, token_id in enumerate(["t1", "t2"]):
            self._seed_buffered_member(factory, token_id=token_id, ordinal=ordinal, agg_node=agg_node, batch_id=old_batch.batch_id)
        # Crash shape: flush died -> batch FAILED; resume's
        # handle_incomplete_batches creates the retry batch (members COPIED).
        factory.execution.update_batch_status(old_batch.batch_id, BatchStatus.FAILED)
        retry_batch = factory.execution.retry_batch(old_batch.batch_id)
        assert retry_batch.batch_id != old_batch.batch_id

        processor = _make_processor(
            factory,
            aggregation_settings=self._agg_settings(agg_node),
            barrier_restore=self._restore_ctx(batch_id_remap={old_batch.batch_id: retry_batch.batch_id}),
        )

        node = processor._aggregation_executor._nodes[agg_node]
        # Headline: the BUFFERED outcomes still say old_batch; the restored
        # in-progress batch is the RETRY batch via the remap.
        assert node.batch_id == retry_batch.batch_id
        assert [t.token_id for t in node.tokens] == ["t1", "t2"]
        # DISTINCT discipline: members exist in BOTH the dead original and the
        # retry copy; a raw count would say 4.
        assert node.accepted_count_total == 2
        assert node.completed_flush_count == 0

    def test_resume_restore_preserves_counters_when_all_flushes_failed(self) -> None:
        """A node whose flushes all FAILED is still counter-only restored.

        Crash right after a failed flush consumed the batch: accepted rows > 0
        but zero COMPLETED batches, empty buffer, no BLOCKED rows, no scalars
        entry. The accepted counter must survive the resume (it drives
        AggregationBatchContext pagination metadata), not silently reset to 0.
        """
        _db, factory = _make_factory()
        agg_node = NodeID("agg-1")
        self._register_aggregation_node(factory, agg_node)
        failed_batch = factory.execution.create_batch(run_id="test-run", aggregation_node_id=str(agg_node))
        for ordinal, token_id in enumerate(["t1", "t2"]):
            payload = make_row({"value": ordinal})
            token = TokenInfo(row_id=f"row-{ordinal}", token_id=token_id, row_data=payload)
            _persist_token_for_scheduler(factory, token, ingest_sequence=ordinal)
            factory.execution.add_batch_member(batch_id=failed_batch.batch_id, token_id=token_id, ordinal=ordinal)
        factory.execution.update_batch_status(failed_batch.batch_id, BatchStatus.FAILED)

        processor = _make_processor(
            factory,
            aggregation_settings=self._agg_settings(agg_node),
            barrier_restore=self._restore_ctx(),
        )

        node = processor._aggregation_executor._nodes[agg_node]
        assert node.tokens == []
        assert node.batch_id is None
        assert node.accepted_count_total == 2  # pagination metadata survives
        assert node.completed_flush_count == 0

    def test_resume_restore_rejects_barrier_group_split_across_batches(self) -> None:
        """Two BLOCKED tokens in one barrier group resolving DIFFERENT batch ids is corruption."""
        _db, factory = _make_factory()
        agg_node = NodeID("agg-1")
        self._register_aggregation_node(factory, agg_node)
        batch_a = factory.execution.create_batch(run_id="test-run", aggregation_node_id=str(agg_node))
        batch_b = factory.execution.create_batch(run_id="test-run", aggregation_node_id=str(agg_node))
        self._seed_buffered_member(factory, token_id="t1", ordinal=0, agg_node=agg_node, batch_id=batch_a.batch_id)
        self._seed_buffered_member(factory, token_id="t2", ordinal=1, agg_node=agg_node, batch_id=batch_b.batch_id)

        with pytest.raises(AuditIntegrityError, match="split across batches"):
            _make_processor(
                factory,
                aggregation_settings=self._agg_settings(agg_node),
                barrier_restore=self._restore_ctx(),
            )

    def test_resume_restore_rejects_blocked_row_with_terminal_outcome(self) -> None:
        """A BLOCKED row whose token already flushed (terminal outcome) is journal/audit divergence.

        Flushed-before-crash shape: the flush recorded BUFFERED -> BATCH_CONSUMED
        but died before terminalizing the journal row. Restoring it would
        double-process the token, so the restore refuses.
        """
        _db, factory = _make_factory()
        agg_node = NodeID("agg-1")
        self._register_aggregation_node(factory, agg_node)
        batch = factory.execution.create_batch(run_id="test-run", aggregation_node_id=str(agg_node))
        self._seed_buffered_member(factory, token_id="t1", ordinal=0, agg_node=agg_node, batch_id=batch.batch_id)
        factory.data_flow.record_token_outcome(
            ref=TokenRef(token_id="t1", run_id="test-run"),
            outcome=TerminalOutcome.TRANSIENT,
            path=TerminalPath.BATCH_CONSUMED,
            batch_id=batch.batch_id,
        )

        with pytest.raises(AuditIntegrityError, match="disagree about this token being buffered"):
            _make_processor(
                factory,
                aggregation_settings=self._agg_settings(agg_node),
                barrier_restore=self._restore_ctx(),
            )

    def test_resume_restore_rejects_remapped_batch_without_batches_row(self) -> None:
        """A remap target with no batches row is audit corruption, not a default."""
        _db, factory = _make_factory()
        agg_node = NodeID("agg-1")
        self._register_aggregation_node(factory, agg_node)
        old_batch = factory.execution.create_batch(run_id="test-run", aggregation_node_id=str(agg_node))
        self._seed_buffered_member(factory, token_id="t1", ordinal=0, agg_node=agg_node, batch_id=old_batch.batch_id)

        with pytest.raises(AuditIntegrityError, match="has no batches row"):
            _make_processor(
                factory,
                aggregation_settings=self._agg_settings(agg_node),
                barrier_restore=self._restore_ctx(batch_id_remap={old_batch.batch_id: "batch-that-does-not-exist"}),
            )

    def test_resume_restore_rejects_batch_owned_by_foreign_aggregation_node(self) -> None:
        """A group's resolved batch must belong to the barrier_key's own aggregation node."""
        _db, factory = _make_factory()
        agg_node = NodeID("agg-1")
        other_node = NodeID("agg-2")
        self._register_aggregation_node(factory, agg_node)
        self._register_aggregation_node(factory, other_node)
        own_batch = factory.execution.create_batch(run_id="test-run", aggregation_node_id=str(agg_node))
        foreign_batch = factory.execution.create_batch(run_id="test-run", aggregation_node_id=str(other_node))
        self._seed_buffered_member(factory, token_id="t1", ordinal=0, agg_node=agg_node, batch_id=own_batch.batch_id)

        with pytest.raises(AuditIntegrityError, match="belongs to aggregation node"):
            _make_processor(
                factory,
                aggregation_settings=self._agg_settings(agg_node),
                barrier_restore=self._restore_ctx(batch_id_remap={own_batch.batch_id: foreign_batch.batch_id}),
            )


class TestTraversalNextNodeInvariants:
    """Tests for strict Tier-1 traversal next-node invariants."""

    def test_resolve_next_node_missing_entry_raises_invariant(self) -> None:
        """Missing traversal next-node entry must crash, not silently return None."""
        _, factory = _make_factory()
        processor = _make_processor(
            factory,
            node_to_next={NodeID("source-0"): None},
        )

        with pytest.raises(OrchestrationInvariantError, match="missing from traversal next-node map"):
            processor._nav.resolve_next_node(NodeID("missing-node"))

    def test_process_row_raises_when_transform_missing_next_node_entry(self) -> None:
        """Processing nodes must have explicit next-node entries (None for terminal)."""
        _db, factory = _make_factory()
        source_row = _make_source_row()
        ctx = make_context(landscape=factory.plugin_audit_writer())
        transform = _make_mock_transform()
        source_node = NodeID("source-0")
        transform_node = NodeID(transform.node_id)
        processor = _make_processor(
            factory,
            node_step_map={source_node: 0, transform_node: 1},
            node_to_next={source_node: transform_node},
            node_to_plugin={transform_node: transform},
        )

        with pytest.raises(OrchestrationInvariantError, match="missing from traversal next-node map"):
            processor.process_row(
                row_index=0,
                source_row=source_row,
                transforms=[transform],
                ctx=ctx,
                source_row_index=0,
                ingest_sequence=0,
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
        _, factory = _make_factory()
        transform_node = NodeID("transform-1")
        processor = _make_processor(
            factory,
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
        _, factory = _make_factory()
        source_node = NodeID("source-0")

        # Build traversal WITHOUT source in the step map
        traversal = DAGTraversalContext(
            node_step_map={},
            node_to_plugin={},
            node_to_next={source_node: None},
            coalesce_node_map={},
        )
        processor = RowProcessor(
            execution=factory.execution,
            data_flow=factory.data_flow,
            span_factory=SpanFactory(),
            run_id="test-run",
            source_node_id=source_node,
            source_on_success="default",
            traversal=traversal,
            scheduler=factory.scheduler,
        )

        assert processor._resolve_audit_step_for_node(source_node) == 0

    def test_unknown_node_raises_invariant_error(self) -> None:
        """Unknown node IDs must crash, not silently return a default step."""
        _, factory = _make_factory()
        processor = _make_processor(
            factory,
            node_to_next={NodeID("source-0"): None},
        )

        with pytest.raises(OrchestrationInvariantError, match="missing from traversal step map"):
            processor._resolve_audit_step_for_node(NodeID("nonexistent-node"))

    def test_unknown_node_includes_node_id_in_error(self) -> None:
        """Error message includes the offending node ID for debugging."""
        _, factory = _make_factory()
        processor = _make_processor(
            factory,
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
        _, factory = _make_factory()
        processor = _make_processor(factory)
        outcome = SimpleNamespace(sink_name="error-sink")
        assert processor._get_gate_destinations(outcome) == ("error-sink",)

    def test_fork_to_paths(self) -> None:
        """Fork returns branch names of child tokens."""
        _, factory = _make_factory()
        processor = _make_processor(factory)
        child_a = SimpleNamespace(branch_name="path_a")
        child_b = SimpleNamespace(branch_name="path_b")
        outcome = SimpleNamespace(
            sink_name=None,
            discarded=False,
            result=SimpleNamespace(action=RoutingAction.fork_to_paths(["path_a", "path_b"])),
            child_tokens=[child_a, child_b],
        )
        assert processor._get_gate_destinations(outcome) == ("path_a", "path_b")

    def test_continue_routing(self) -> None:
        """Continue routing returns ("continue",)."""
        _, factory = _make_factory()
        processor = _make_processor(factory)
        outcome = SimpleNamespace(
            sink_name=None,
            discarded=False,
            result=SimpleNamespace(action=RoutingAction.continue_()),
            next_node_id=None,
        )
        assert processor._get_gate_destinations(outcome) == ("continue",)

    def test_route_to_processing_uses_route_label(self) -> None:
        """Route-label branch to processing node reports chosen route label."""
        _, factory = _make_factory()
        processor = _make_processor(factory)
        outcome = SimpleNamespace(
            sink_name=None,
            discarded=False,
            next_node_id=NodeID("transform-2"),
            result=SimpleNamespace(action=RoutingAction.route("high")),
        )
        assert processor._get_gate_destinations(outcome) == ("high",)

    def test_gate_outcome_false_discarded_does_not_mask_fork_to_paths(self) -> None:
        """GateOutcome with discarded=False reports fork paths."""
        _, factory = _make_factory()
        processor = _make_processor(factory)
        child_a = make_token_info(data={"value": 1}, branch_name="path_a")
        child_b = make_token_info(data={"value": 2}, branch_name="path_b")
        outcome = GateOutcome(
            result=GateResult(row={"value": 1}, action=RoutingAction.fork_to_paths(["path_a", "path_b"])),
            updated_token=make_token_info(data={"value": 1}),
            child_tokens=(child_a, child_b),
        )
        assert processor._get_gate_destinations(outcome) == ("path_a", "path_b")

    def test_gate_outcome_false_discarded_does_not_mask_continue_routing(self) -> None:
        """GateOutcome with discarded=False reports continue."""
        _, factory = _make_factory()
        processor = _make_processor(factory)
        outcome = GateOutcome(
            result=GateResult(row={"value": 1}, action=RoutingAction.continue_()),
            updated_token=make_token_info(data={"value": 1}),
        )
        assert processor._get_gate_destinations(outcome) == ("continue",)

    def test_gate_outcome_false_discarded_does_not_mask_processing_route_label(self) -> None:
        """GateOutcome with discarded=False reports route labels."""
        _, factory = _make_factory()
        processor = _make_processor(factory)
        outcome = GateOutcome(
            next_node_id=NodeID("transform-2"),
            result=GateResult(row={"value": 1}, action=RoutingAction.route("high")),
            updated_token=make_token_info(data={"value": 1}),
        )
        assert processor._get_gate_destinations(outcome) == ("high",)


# =============================================================================
# process_row: Linear pipeline (no transforms)
# =============================================================================


class TestProcessRowNoTransforms:
    """Tests for process_row with an empty transform list."""

    @staticmethod
    def _source_plugin(*, declared_guaranteed_fields: frozenset[str]) -> Any:
        plugin = type("ProcessorSourcePlugin", (), {})()
        plugin.name = "processor-source"
        plugin.node_id = "source-0"
        plugin.declared_guaranteed_fields = declared_guaranteed_fields
        return plugin

    def test_empty_pipeline_returns_completed(self) -> None:
        """Row through empty pipeline gets COMPLETED outcome."""
        _, factory = _make_factory()

        processor = _make_processor(factory)
        source_row = _make_source_row()
        ctx = make_context(landscape=factory.plugin_audit_writer())

        results = processor.process_row(
            row_index=0,
            source_row=source_row,
            transforms=[],
            ctx=ctx,
            source_row_index=0,
            ingest_sequence=0,
        )

        assert len(results) == 1
        _assert_outcome_pair(results[0], TerminalOutcome.SUCCESS, TerminalPath.DEFAULT_FLOW)
        assert results[0].sink_name == "default"
        assert results[0].token.row_data["value"] == 42

    def test_creates_row_and_token_records(self) -> None:
        """process_row creates row record and token in audit trail."""
        _db, factory = _make_factory()

        processor = _make_processor(factory)
        source_row = _make_source_row()
        ctx = make_context(landscape=factory.plugin_audit_writer())

        results = processor.process_row(
            row_index=0,
            source_row=source_row,
            transforms=[],
            ctx=ctx,
            source_row_index=0,
            ingest_sequence=0,
        )

        # Verify token was created (result has a valid token_id)
        token = results[0].token
        assert token.token_id is not None
        assert token.row_id is not None

    def test_records_source_node_state(self) -> None:
        """process_row records a source node_state with COMPLETED status."""
        db, factory = _make_factory()

        processor = _make_processor(factory)
        source_row = _make_source_row()
        ctx = make_context(landscape=factory.plugin_audit_writer())

        with (
            patch.object(
                factory.execution, "record_completed_node_state", wraps=factory.execution.record_completed_node_state
            ) as completed,
            patch.object(factory.execution, "begin_node_state", wraps=factory.execution.begin_node_state) as begin,
            patch.object(factory.execution, "complete_node_state", wraps=factory.execution.complete_node_state) as complete,
        ):
            processor.process_row(
                row_index=0,
                source_row=source_row,
                transforms=[],
                ctx=ctx,
                source_row_index=0,
                ingest_sequence=0,
            )

        assert completed.call_count == 1
        assert begin.call_count == 0
        assert complete.call_count == 0

        # Check that source node_state was recorded
        from sqlalchemy import select

        from elspeth.core.landscape.schema import node_states_table

        with db.connection() as conn:
            states = conn.execute(
                select(node_states_table).where(
                    node_states_table.c.run_id == "test-run",
                    node_states_table.c.node_id == "source-0",
                )
            ).fetchall()

        assert len(states) == 1
        [source_state] = states
        assert source_state.node_id == "source-0"
        assert source_state.status == NodeStateStatus.COMPLETED

    def test_source_boundary_violation_records_failed_outcome_and_failed_source_state(self) -> None:
        """Tier-1 source boundary failures must still leave terminal audit evidence."""
        db, factory = _make_factory()
        processor = _make_processor(
            factory,
            source_plugin=self._source_plugin(declared_guaranteed_fields=frozenset({"customer_id"})),
        )
        source_row = _make_source_row({"value": 42})
        ctx = make_context(landscape=factory.plugin_audit_writer())

        def _raise_source_boundary(*args: Any, **kwargs: Any) -> None:
            violation = SourceGuaranteedFieldsViolation(
                plugin="processor-source",
                node_id="source-0",
                run_id="test-run",
                row_id="row_0",
                token_id="token_0",
                payload={
                    "declared": ["customer_id"],
                    "runtime_observed": [],
                    "missing": ["customer_id"],
                },
                message="source boundary failed",
            )
            _attach_contract_name_from_dispatcher(violation, "source_guaranteed_fields")
            raise violation

        with (
            patch("elspeth.engine.processor.run_boundary_checks", side_effect=_raise_source_boundary),
            pytest.raises(SourceGuaranteedFieldsViolation),
        ):
            processor.process_row(
                row_index=0,
                source_row=source_row,
                transforms=[],
                ctx=ctx,
                source_row_index=0,
                ingest_sequence=0,
            )

        from sqlalchemy import select

        from elspeth.core.landscape.schema import node_states_table, token_outcomes_table

        with db.connection() as conn:
            states = conn.execute(
                select(node_states_table).where(
                    node_states_table.c.run_id == "test-run",
                    node_states_table.c.node_id == "source-0",
                )
            ).fetchall()
            outcomes = conn.execute(select(token_outcomes_table).where(token_outcomes_table.c.run_id == "test-run")).fetchall()

        assert len(states) == 1
        [source_state] = states
        assert source_state.node_id == "source-0"
        assert source_state.status == NodeStateStatus.FAILED
        assert len(outcomes) == 1
        _assert_outcome_pair(outcomes[0], TerminalOutcome.FAILURE, TerminalPath.UNROUTED)

    def test_source_boundary_violation_on_secondary_source_uses_secondary_source_identity(self) -> None:
        """Multi-source boundary failures must be attributed to the row's source root."""
        db, factory = _make_factory()
        factory.data_flow.register_node(
            run_id="test-run",
            plugin_name="secondary-source",
            node_type=NodeType.SOURCE,
            plugin_version="1.0",
            config={},
            node_id="source-1",
            schema_config=_DYNAMIC_SCHEMA,
        )
        processor = _make_processor(
            factory,
            source_plugin=self._source_plugin(declared_guaranteed_fields=frozenset({"customer_id"})),
        )
        source_row = _make_source_row({"value": 42})
        ctx = make_context(landscape=factory.plugin_audit_writer())

        def _raise_source_boundary(*args: Any, **kwargs: Any) -> None:
            violation = SourceGuaranteedFieldsViolation(
                plugin="secondary-source",
                node_id="source-1",
                run_id="test-run",
                row_id="row_0",
                token_id="token_0",
                payload={
                    "declared": ["customer_id"],
                    "runtime_observed": [],
                    "missing": ["customer_id"],
                },
                message="secondary source boundary failed",
            )
            _attach_contract_name_from_dispatcher(violation, "source_guaranteed_fields")
            raise violation

        with (
            patch("elspeth.engine.processor.run_boundary_checks", side_effect=_raise_source_boundary),
            patch("elspeth.engine.processor.best_effort", return_value=nullcontext()) as best_effort_context,
            pytest.raises(SourceGuaranteedFieldsViolation, match="secondary source boundary failed"),
        ):
            processor.process_row(
                row_index=0,
                source_row=source_row,
                transforms=[],
                ctx=ctx,
                source_node_id=NodeID("source-1"),
                source_row_index=0,
                ingest_sequence=0,
            )

        from sqlalchemy import select

        from elspeth.core.landscape.schema import node_states_table, rows_table, token_outcomes_table, tokens_table

        with db.connection() as conn:
            failed_states = conn.execute(
                select(node_states_table).where(
                    node_states_table.c.run_id == "test-run",
                    node_states_table.c.status == NodeStateStatus.FAILED,
                )
            ).fetchall()
            outcomes = conn.execute(select(token_outcomes_table).where(token_outcomes_table.c.run_id == "test-run")).fetchall()
            row_source_node_id = conn.execute(
                select(rows_table.c.source_node_id)
                .join(tokens_table, rows_table.c.row_id == tokens_table.c.row_id)
                .where(tokens_table.c.run_id == "test-run")
            ).scalar_one()

        assert row_source_node_id == "source-1"
        assert len(failed_states) == 1
        [source_state] = failed_states
        assert source_state.node_id == "source-1"
        assert len(outcomes) == 1
        expected_hash = hashlib.sha256(b"SourceGuaranteedFieldsViolation:source-1").hexdigest()[:16]
        assert outcomes[0].error_hash == expected_hash
        best_effort_context.assert_called_once()
        assert best_effort_context.call_args.kwargs["source_node_id"] == NodeID("source-1")

    def test_source_boundary_landscape_record_error_raises_audit_integrity_error(self) -> None:
        """Narrow recorder failures outrank the original boundary violation."""
        _db, factory = _make_factory()
        processor = _make_processor(
            factory,
            source_plugin=self._source_plugin(declared_guaranteed_fields=frozenset({"customer_id"})),
        )
        source_row = _make_source_row({"value": 42})
        ctx = make_context(landscape=factory.plugin_audit_writer())

        def _raise_source_boundary(*args: Any, **kwargs: Any) -> None:
            violation = SourceGuaranteedFieldsViolation(
                plugin="processor-source",
                node_id="source-0",
                run_id="test-run",
                row_id="row_0",
                token_id="token_0",
                payload={
                    "declared": ["customer_id"],
                    "runtime_observed": [],
                    "missing": ["customer_id"],
                },
                message="source boundary failed",
            )
            _attach_contract_name_from_dispatcher(violation, "source_guaranteed_fields")
            raise violation

        with (
            patch("elspeth.engine.processor.run_boundary_checks", side_effect=_raise_source_boundary),
            patch.object(factory.data_flow, "record_token_outcome", side_effect=LandscapeRecordError("token outcome DB down")),
            pytest.raises(
                AuditIntegrityError,
                match=r"Failed to record SourceGuaranteedFieldsViolation FAILED outcome for token .* on source boundary",
            ) as exc_info,
        ):
            processor.process_row(
                row_index=0,
                source_row=source_row,
                transforms=[],
                ctx=ctx,
                source_row_index=0,
                ingest_sequence=0,
            )

        assert isinstance(exc_info.value.__cause__, LandscapeRecordError)
        assert "Original violation: source boundary failed" in str(exc_info.value)

    def test_source_boundary_value_error_from_token_outcome_propagates_plainly(self) -> None:
        """Programmer bugs in token-outcome recording must not be reclassified."""
        _db, factory = _make_factory()
        processor = _make_processor(
            factory,
            source_plugin=self._source_plugin(declared_guaranteed_fields=frozenset({"customer_id"})),
        )
        source_row = _make_source_row({"value": 42})
        ctx = make_context(landscape=factory.plugin_audit_writer())

        def _raise_source_boundary(*args: Any, **kwargs: Any) -> None:
            violation = SourceGuaranteedFieldsViolation(
                plugin="processor-source",
                node_id="source-0",
                run_id="test-run",
                row_id="row_0",
                token_id="token_0",
                payload={
                    "declared": ["customer_id"],
                    "runtime_observed": [],
                    "missing": ["customer_id"],
                },
                message="source boundary failed",
            )
            _attach_contract_name_from_dispatcher(violation, "source_guaranteed_fields")
            raise violation

        with (
            patch("elspeth.engine.processor.run_boundary_checks", side_effect=_raise_source_boundary),
            patch.object(factory.data_flow, "record_token_outcome", side_effect=ValueError("recorder bug")),
            pytest.raises(ValueError, match="recorder bug"),
        ):
            processor.process_row(
                row_index=0,
                source_row=source_row,
                transforms=[],
                ctx=ctx,
                source_row_index=0,
                ingest_sequence=0,
            )

    def test_source_boundary_state_landscape_record_error_raises_audit_integrity_error(self) -> None:
        """FAILED source-state recorder failures outrank the original violation."""
        _db, factory = _make_factory()
        processor = _make_processor(
            factory,
            source_plugin=self._source_plugin(declared_guaranteed_fields=frozenset({"customer_id"})),
        )
        source_row = _make_source_row({"value": 42})
        ctx = make_context(landscape=factory.plugin_audit_writer())

        def _raise_source_boundary(*args: Any, **kwargs: Any) -> None:
            violation = SourceGuaranteedFieldsViolation(
                plugin="processor-source",
                node_id="source-0",
                run_id="test-run",
                row_id="row_0",
                token_id="token_0",
                payload={
                    "declared": ["customer_id"],
                    "runtime_observed": [],
                    "missing": ["customer_id"],
                },
                message="source boundary failed",
            )
            _attach_contract_name_from_dispatcher(violation, "source_guaranteed_fields")
            raise violation

        with (
            patch("elspeth.engine.processor.run_boundary_checks", side_effect=_raise_source_boundary),
            patch.object(factory.execution, "complete_node_state", side_effect=LandscapeRecordError("source state DB down")),
            pytest.raises(
                AuditIntegrityError,
                match=r"Failed to record FAILED source node state for token .* on source boundary",
            ) as exc_info,
        ):
            processor.process_row(
                row_index=0,
                source_row=source_row,
                transforms=[],
                ctx=ctx,
                source_row_index=0,
                ingest_sequence=0,
            )

        assert isinstance(exc_info.value.__cause__, LandscapeRecordError)
        assert "Original violation: source boundary failed" in str(exc_info.value)

    def test_source_boundary_runtime_error_from_node_state_recording_propagates_plainly(self) -> None:
        """Programmer bugs in source-state recording must not be reclassified."""
        _db, factory = _make_factory()
        processor = _make_processor(
            factory,
            source_plugin=self._source_plugin(declared_guaranteed_fields=frozenset({"customer_id"})),
        )
        source_row = _make_source_row({"value": 42})
        ctx = make_context(landscape=factory.plugin_audit_writer())

        def _raise_source_boundary(*args: Any, **kwargs: Any) -> None:
            violation = SourceGuaranteedFieldsViolation(
                plugin="processor-source",
                node_id="source-0",
                run_id="test-run",
                row_id="row_0",
                token_id="token_0",
                payload={
                    "declared": ["customer_id"],
                    "runtime_observed": [],
                    "missing": ["customer_id"],
                },
                message="source boundary failed",
            )
            _attach_contract_name_from_dispatcher(violation, "source_guaranteed_fields")
            raise violation

        with (
            patch("elspeth.engine.processor.run_boundary_checks", side_effect=_raise_source_boundary),
            patch.object(factory.execution, "complete_node_state", side_effect=RuntimeError("source state bug")),
            pytest.raises(RuntimeError, match="source state bug"),
        ):
            processor.process_row(
                row_index=0,
                source_row=source_row,
                transforms=[],
                ctx=ctx,
                source_row_index=0,
                ingest_sequence=0,
            )

    def test_source_boundary_token_completed_telemetry_failure_does_not_interrupt_audit_recording(self) -> None:
        """Best-effort telemetry must not break the FAILED-audit pair on source violations."""
        db, factory = _make_factory()
        processor = _make_processor(
            factory,
            source_plugin=self._source_plugin(declared_guaranteed_fields=frozenset({"customer_id"})),
        )
        source_row = _make_source_row({"value": 42})
        ctx = make_context(landscape=factory.plugin_audit_writer())

        def _raise_source_boundary(*args: Any, **kwargs: Any) -> None:
            violation = SourceGuaranteedFieldsViolation(
                plugin="processor-source",
                node_id="source-0",
                run_id="test-run",
                row_id="row_0",
                token_id="token_0",
                payload={
                    "declared": ["customer_id"],
                    "runtime_observed": [],
                    "missing": ["customer_id"],
                },
                message="source boundary failed",
            )
            _attach_contract_name_from_dispatcher(violation, "source_guaranteed_fields")
            raise violation

        with (
            patch("elspeth.engine.processor.run_boundary_checks", side_effect=_raise_source_boundary),
            patch.object(processor, "_emit_token_completed", side_effect=RuntimeError("telemetry down")),
            pytest.raises(SourceGuaranteedFieldsViolation, match="source boundary failed"),
        ):
            processor.process_row(
                row_index=0,
                source_row=source_row,
                transforms=[],
                ctx=ctx,
                source_row_index=0,
                ingest_sequence=0,
            )

        from sqlalchemy import select

        from elspeth.core.landscape.schema import node_states_table, token_outcomes_table

        with db.connection() as conn:
            states = conn.execute(
                select(node_states_table).where(
                    node_states_table.c.run_id == "test-run",
                    node_states_table.c.node_id == "source-0",
                )
            ).fetchall()
            outcomes = conn.execute(select(token_outcomes_table).where(token_outcomes_table.c.run_id == "test-run")).fetchall()

        assert len(states) == 1
        [source_state] = states
        assert source_state.node_id == "source-0"
        assert source_state.status == NodeStateStatus.FAILED
        assert len(outcomes) == 1
        _assert_outcome_pair(outcomes[0], TerminalOutcome.FAILURE, TerminalPath.UNROUTED)

    def test_source_boundary_framework_bug_records_failed_outcome_and_failed_source_state(self) -> None:
        """Tier-1 framework bugs after token creation must still leave terminal audit evidence."""
        db, factory = _make_factory()
        processor = _make_processor(
            factory,
            source_plugin=self._source_plugin(declared_guaranteed_fields=frozenset({"customer_id"})),
        )
        source_row = _make_source_row({"value": 42})
        ctx = make_context(landscape=factory.plugin_audit_writer())

        with (
            patch(
                "elspeth.engine.processor.run_boundary_checks",
                side_effect=FrameworkBugError("source row crossed boundary without a schema contract"),
            ),
            pytest.raises(FrameworkBugError, match="without a schema contract"),
        ):
            processor.process_row(
                row_index=0,
                source_row=source_row,
                transforms=[],
                ctx=ctx,
                source_row_index=0,
                ingest_sequence=0,
            )

        from sqlalchemy import select

        from elspeth.core.landscape.schema import node_states_table, token_outcomes_table

        with db.connection() as conn:
            states = conn.execute(
                select(node_states_table).where(
                    node_states_table.c.run_id == "test-run",
                    node_states_table.c.node_id == "source-0",
                )
            ).fetchall()
            outcomes = conn.execute(select(token_outcomes_table).where(token_outcomes_table.c.run_id == "test-run")).fetchall()

        assert len(states) == 1
        [source_state] = states
        assert source_state.node_id == "source-0"
        assert source_state.status == NodeStateStatus.FAILED
        assert len(outcomes) == 1
        _assert_outcome_pair(outcomes[0], TerminalOutcome.FAILURE, TerminalPath.UNROUTED)

    def test_batch_flush_token_completed_telemetry_failure_does_not_interrupt_failed_audit_recording(self) -> None:
        """Best-effort telemetry must not interrupt per-token FAILED batch-flush audit writes."""
        _db, factory = _make_factory()
        processor = _make_processor(factory)
        transform = _make_mock_transform(node_id="aggregate-1", name="batch-transform")
        token_a = make_token_info(row_id="row-a", token_id="token-a", data={"value": 1})
        token_b = make_token_info(row_id="row-b", token_id="token-b", data={"value": 2})
        violation = SourceGuaranteedFieldsViolation(
            plugin="batch-transform",
            node_id="aggregate-1",
            run_id="test-run",
            row_id="row-a",
            token_id="token-a",
            payload={
                "declared": ["customer_id"],
                "runtime_observed": [],
                "missing": ["customer_id"],
            },
            message="batch flush failed",
        )
        _attach_contract_name_from_dispatcher(violation, "source_guaranteed_fields")
        fctx = _FlushContext(
            node_id=NodeID("aggregate-1"),
            transform=transform,
            settings=AggregationSettings(
                name="agg",
                plugin="batch-plugin",
                input="source",
                on_error="discard",
                trigger={"count": 2},
            ),
            buffered_tokens=(token_a, token_b),
            batch_id="batch-1",
            error_msg="batch flush failed",
            expand_parent_token=token_a,
            triggering_token=token_b,
            coalesce_node_id=None,
            coalesce_name=None,
        )

        with (
            patch.object(factory.data_flow, "record_token_outcome") as mock_record_token_outcome,
            patch.object(processor, "_emit_token_completed", side_effect=RuntimeError("telemetry down")),
        ):
            processor._record_flush_violation(fctx, violation)

        assert mock_record_token_outcome.call_count == 2
        recorded_refs = {call.kwargs["ref"].token_id for call in mock_record_token_outcome.call_args_list}
        assert recorded_refs == {"token-a", "token-b"}

    def test_buffered_scheduler_barrier_key_requires_live_hold_stash(self) -> None:
        """A BUFFERED claim result without a live barrier-hold stash is a processor bug.

        Slice 3 re-pin (ADR-030 §E.2): the barrier_key for the drain's
        mark_blocked no longer derives from the (now nonexistent) in-claim
        BUFFERED outcome — the producer stashes the live hold; a missing
        stash entry is refused loudly.
        """
        _db, factory = _make_factory()
        processor = _make_processor(factory)

        with pytest.raises(AuditIntegrityError, match="no live barrier hold stash"):
            processor._barrier_key_for_live_hold("token-a")

        processor._live_barrier_holds["token-a"] = _LiveBarrierHold(
            token=make_token_info(row_id="row-a", token_id="token-a", data={"value": 1}),
            barrier_key="aggregation_a",
        )
        assert processor._barrier_key_for_live_hold("token-a") == "aggregation_a"

    def test_handle_flush_error_telemetry_failure_does_not_interrupt_failed_outcomes(self) -> None:
        """Batch-flush failure terminalization must continue after telemetry errors."""
        _db, factory = _make_factory()
        processor = _make_processor(factory)
        transform = _make_mock_transform(node_id="aggregate-1", name="batch-transform")
        token_a = make_token_info(row_id="row-a", token_id="token-a", data={"value": 1})
        token_b = make_token_info(row_id="row-b", token_id="token-b", data={"value": 2})
        token_c = make_token_info(row_id="row-c", token_id="token-c", data={"value": 3})
        fctx = _FlushContext(
            node_id=NodeID("aggregate-1"),
            transform=transform,
            settings=AggregationSettings(
                name="agg",
                plugin="batch-plugin",
                input="source",
                on_error="discard",
                trigger={"count": 3},
            ),
            buffered_tokens=(token_a, token_b, token_c),
            batch_id="batch-1",
            error_msg="batch flush failed",
            expand_parent_token=token_a,
            triggering_token=token_c,
            coalesce_node_id=None,
            coalesce_name=None,
        )

        with (
            patch.object(factory.data_flow, "record_token_outcome") as mock_record_token_outcome,
            patch.object(
                processor,
                "_emit_token_completed",
                side_effect=[None, RuntimeError("telemetry down"), None],
            ),
        ):
            results = processor._handle_flush_error(fctx)

        assert mock_record_token_outcome.call_count == 3
        recorded_refs = [call.kwargs["ref"].token_id for call in mock_record_token_outcome.call_args_list]
        assert recorded_refs == ["token-a", "token-b", "token-c"]
        assert tuple(result.token.token_id for result in results) == ("token-a", "token-b", "token-c")
        assert tuple((result.outcome, result.path) for result in results) == (
            (TerminalOutcome.FAILURE, TerminalPath.UNROUTED),
            (TerminalOutcome.FAILURE, TerminalPath.UNROUTED),
            (TerminalOutcome.FAILURE, TerminalPath.UNROUTED),
        )

    def test_handle_flush_error_recorder_failure_raises_audit_integrity_error(self) -> None:
        """Recorder failure during batch-flush terminalization must crash loudly."""
        _db, factory = _make_factory()
        processor = _make_processor(factory)
        transform = _make_mock_transform(node_id="aggregate-1", name="batch-transform")
        token_a = make_token_info(row_id="row-a", token_id="token-a", data={"value": 1})
        token_b = make_token_info(row_id="row-b", token_id="token-b", data={"value": 2})
        token_c = make_token_info(row_id="row-c", token_id="token-c", data={"value": 3})
        fctx = _FlushContext(
            node_id=NodeID("aggregate-1"),
            transform=transform,
            settings=AggregationSettings(
                name="agg",
                plugin="batch-plugin",
                input="source",
                on_error="discard",
                trigger={"count": 3},
            ),
            buffered_tokens=(token_a, token_b, token_c),
            batch_id="batch-1",
            error_msg="batch flush failed",
            expand_parent_token=token_a,
            triggering_token=token_c,
            coalesce_node_id=None,
            coalesce_name=None,
        )

        attempted_refs: list[str] = []

        def fail_on_second_record(*args: Any, **kwargs: Any) -> None:
            token_id = kwargs["ref"].token_id
            attempted_refs.append(token_id)
            if token_id == "token-b":
                raise LandscapeRecordError("audit DB down")

        with (
            patch.object(factory.data_flow, "record_token_outcome", side_effect=fail_on_second_record),
            pytest.raises(
                AuditIntegrityError,
                match=r"Failed to record FAILED outcome for token 'token-b'",
            ) as exc_info,
        ):
            processor._handle_flush_error(fctx)

        assert attempted_refs == ["token-a", "token-b"]
        assert isinstance(exc_info.value.__cause__, LandscapeRecordError)

    def test_empty_batch_flush_telemetry_failure_does_not_interrupt_dropped_outcomes(self) -> None:
        """Zero-row batch flush must still terminalize every buffered token if telemetry fails."""
        _db, factory = _make_factory()
        telemetry_manager = create_autospec(TelemetryManagerProtocol, instance=True)
        telemetry_manager.handle_event.side_effect = RuntimeError("telemetry down")
        processor = _make_processor(factory, telemetry_manager=telemetry_manager)
        transform = _make_mock_transform(node_id="aggregate-1", name="batch-transform")
        token_a = make_token_info(row_id="row-a", token_id="token-a", data={"value": 1})
        token_b = make_token_info(row_id="row-b", token_id="token-b", data={"value": 2})
        token_c = make_token_info(row_id="row-c", token_id="token-c", data={"value": 3})
        fctx = _FlushContext(
            node_id=NodeID("aggregate-1"),
            transform=transform,
            settings=AggregationSettings(
                name="agg",
                plugin="batch-plugin",
                input="source",
                on_error="discard",
                trigger={"count": 3},
            ),
            buffered_tokens=(token_a, token_b, token_c),
            batch_id="batch-1",
            error_msg="batch flush dropped rows",
            expand_parent_token=token_a,
            triggering_token=token_c,
            coalesce_node_id=None,
            coalesce_name=None,
        )

        with patch.object(factory.data_flow, "record_token_outcome") as mock_record_token_outcome:
            results, child_items = processor._route_empty_emission_results(fctx)

        assert mock_record_token_outcome.call_count == 3
        recorded_refs = {call.kwargs["ref"].token_id for call in mock_record_token_outcome.call_args_list}
        assert recorded_refs == {"token-a", "token-b", "token-c"}
        assert telemetry_manager.handle_event.call_count == 3
        assert child_items == []
        assert tuple((result.outcome, result.path) for result in results) == (
            (TerminalOutcome.SUCCESS, TerminalPath.FILTER_DROPPED),
            (TerminalOutcome.SUCCESS, TerminalPath.FILTER_DROPPED),
            (TerminalOutcome.SUCCESS, TerminalPath.FILTER_DROPPED),
        )

    def test_empty_batch_flush_recorder_failure_raises_audit_integrity_error(self) -> None:
        """Typed recorder failures during zero-row batch terminalization must outrank success."""
        _db, factory = _make_factory()
        processor = _make_processor(factory)
        transform = _make_mock_transform(node_id="aggregate-1", name="batch-transform")
        token_a = make_token_info(row_id="row-a", token_id="token-a", data={"value": 1})
        token_b = make_token_info(row_id="row-b", token_id="token-b", data={"value": 2})
        fctx = _FlushContext(
            node_id=NodeID("aggregate-1"),
            transform=transform,
            settings=AggregationSettings(
                name="agg",
                plugin="batch-plugin",
                input="source",
                on_error="discard",
                trigger={"count": 2},
            ),
            buffered_tokens=(token_a, token_b),
            batch_id="batch-1",
            error_msg="batch flush dropped rows",
            expand_parent_token=token_a,
            triggering_token=token_b,
            coalesce_node_id=None,
            coalesce_name=None,
        )

        with (
            patch.object(factory.data_flow, "record_token_outcome", side_effect=LandscapeRecordError("audit DB down")),
            pytest.raises(
                AuditIntegrityError,
                match=r"Failed to record DROPPED_BY_FILTER outcome for token 'token-a'",
            ) as exc_info,
        ):
            processor._route_empty_emission_results(fctx)

        assert isinstance(exc_info.value.__cause__, LandscapeRecordError)

    def test_success_empty_recorder_failure_raises_audit_integrity_error(self) -> None:
        """Typed recorder failures during single-row success_empty terminalization must outrank success."""
        _db, factory = _make_factory()
        processor = _make_processor(factory)
        transform = _make_mock_transform(node_id="filter-1", name="dropper")
        token = make_token_info(row_id="row-drop", token_id="token-drop", data={"value": 1})
        ctx = make_context(landscape=factory.plugin_audit_writer())

        with (
            patch.object(
                processor,
                "_execute_transform_with_retry",
                return_value=(TransformResult.success_empty(success_reason={"action": "filtered"}), token, None),
            ),
            patch.object(factory.data_flow, "record_token_outcome", side_effect=LandscapeRecordError("audit DB down")),
            pytest.raises(
                AuditIntegrityError,
                match=r"Failed to record DROPPED_BY_FILTER outcome for token 'token-drop'",
            ) as exc_info,
        ):
            processor._handle_transform_node(
                transform=transform,
                current_token=token,
                ctx=ctx,
                node_id=NodeID("filter-1"),
                child_items=[],
                coalesce_node_id=None,
                coalesce_name=None,
                current_on_success_sink="default",
            )

        assert isinstance(exc_info.value.__cause__, LandscapeRecordError)

    def test_batch_flush_non_recorder_recording_bug_propagates_unmodified(self) -> None:
        """Only recorder failures become AuditIntegrityError on batch-flush auto-fail."""
        _db, factory = _make_factory()
        processor = _make_processor(factory)
        transform = _make_mock_transform(node_id="aggregate-1", name="batch-transform")
        token_a = make_token_info(row_id="row-a", token_id="token-a", data={"value": 1})
        token_b = make_token_info(row_id="row-b", token_id="token-b", data={"value": 2})
        violation = SourceGuaranteedFieldsViolation(
            plugin="batch-transform",
            node_id="aggregate-1",
            run_id="test-run",
            row_id="row-a",
            token_id="token-a",
            payload={
                "declared": ["customer_id"],
                "runtime_observed": [],
                "missing": ["customer_id"],
            },
            message="batch flush failed",
        )
        _attach_contract_name_from_dispatcher(violation, "source_guaranteed_fields")
        fctx = _FlushContext(
            node_id=NodeID("aggregate-1"),
            transform=transform,
            settings=AggregationSettings(
                name="agg",
                plugin="batch-plugin",
                input="source",
                on_error="discard",
                trigger={"count": 2},
            ),
            buffered_tokens=(token_a, token_b),
            batch_id="batch-1",
            error_msg="batch flush failed",
            expand_parent_token=token_a,
            triggering_token=token_b,
            coalesce_node_id=None,
            coalesce_name=None,
        )
        invariant_failure = RuntimeError("wrong-run token ownership")

        with (
            patch.object(factory.data_flow, "record_token_outcome", side_effect=invariant_failure),
            patch.object(processor, "_emit_token_completed") as mock_emit,
            pytest.raises(RuntimeError, match="wrong-run token ownership") as exc_info,
        ):
            processor._record_flush_violation(fctx, violation)

        assert exc_info.value is invariant_failure
        mock_emit.assert_not_called()


# =============================================================================
# process_row: Single transform
# =============================================================================


class TestProcessRowSingleTransform:
    """Tests for process_row with a single transform."""

    def _setup(self, transform: Any) -> tuple[LandscapeDB, RecorderFactory, RowProcessor]:
        db, factory = _make_factory()
        source_node = NodeID("source-0")
        transform_node = NodeID(transform.node_id)

        processor = _make_processor(
            factory,
            node_step_map={source_node: 0, transform_node: 1},
            node_to_next={source_node: transform_node, transform_node: None},
            node_to_plugin={transform_node: transform},
        )
        return db, factory, processor

    def test_successful_transform_returns_completed(self) -> None:
        """Row passes through transform → COMPLETED."""
        transform = _make_mock_transform()
        _db, factory, processor = self._setup(transform)
        source_row = _make_source_row({"value": 10})
        ctx = make_context(landscape=factory.plugin_audit_writer())

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
                source_row_index=0,
                ingest_sequence=0,
            )

        assert len(results) == 1
        _assert_outcome_pair(results[0], TerminalOutcome.SUCCESS, TerminalPath.DEFAULT_FLOW)
        assert results[0].sink_name == "default"

    def test_transform_error_with_discard_returns_quarantined(self) -> None:
        """Transform error with on_error='discard' → QUARANTINED."""
        transform = _make_mock_transform(on_error="discard")
        _db, factory, processor = self._setup(transform)
        source_row = _make_source_row()
        ctx = make_context(landscape=factory.plugin_audit_writer())

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
                source_row_index=0,
                ingest_sequence=0,
            )

        assert len(results) == 1
        _assert_outcome_pair(results[0], TerminalOutcome.FAILURE, TerminalPath.QUARANTINED_AT_SOURCE)

    def test_transform_error_with_named_sink_returns_routed_on_error(self) -> None:
        """Transform error with on_error='errors' → ROUTED_ON_ERROR (DIVERT) to error sink.

        elspeth-5069612f3c: transform on_error path emits the dedicated
        ROUTED_ON_ERROR outcome (rows_routed_failure), distinct from the
        gate route_to_sink MOVE which uses ROUTED (rows_routed_success).
        """
        transform = _make_mock_transform(on_error="errors")
        _db, factory, processor = self._setup(transform)
        source_row = _make_source_row()
        ctx = make_context(landscape=factory.plugin_audit_writer())

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
                source_row_index=0,
                ingest_sequence=0,
            )

        assert len(results) == 1
        _assert_outcome_pair(results[0], TerminalOutcome.FAILURE, TerminalPath.ON_ERROR_ROUTED)
        assert results[0].sink_name == "errors"

    def test_max_retries_exceeded_returns_failed(self) -> None:
        """MaxRetriesExceeded → FAILED outcome."""
        transform = _make_mock_transform()
        _db, factory, processor = self._setup(transform)
        source_row = _make_source_row()
        ctx = make_context(landscape=factory.plugin_audit_writer())

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
                source_row_index=0,
                ingest_sequence=0,
            )

        assert len(results) == 1
        _assert_outcome_pair(results[0], TerminalOutcome.FAILURE, TerminalPath.UNROUTED)
        assert results[0].error is not None


class TestAggregationFailureMatrix:
    """Focused aggregation failure/regression matrix coverage."""

    def _setup_batch_processor(
        self,
        *,
        output_mode: str,
        node_to_next: dict[NodeID, NodeID | None] | None = None,
        transform_on_success: str | None = "agg_sink",
    ) -> tuple[LandscapeDB, RecorderFactory, RowProcessor, Mock, NodeID]:
        """Create a RowProcessor configured for a single batch-aware aggregation node."""
        db, factory = _make_factory()
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
            factory,
            node_step_map={source_node: 0, agg_node: 1, NodeID("downstream-2"): 2},
            node_to_next=traversal_next,
            node_to_plugin={agg_node: transform},
            aggregation_settings={
                agg_node: AggregationSettings(
                    name="batch_agg",
                    plugin="agg-transform",
                    input="default",
                    on_error="discard",
                    trigger={"count": 1},
                    output_mode=output_mode,
                ),
            },
        )
        return db, factory, processor, transform, agg_node

    def test_flush_failure_passthrough_records_failed_outcomes(self) -> None:
        """Passthrough flush failure records FAILED terminal outcomes for buffered tokens.

        Slice 3 re-pin (ADR-030 §E.2): the arrival returns a real
        (None, BUFFERED) RowResult and the count flush fires from the NEXT
        drain iteration's journal-first intake; the BUFFERED audit record is
        written by the fenced adoption verb (not record_token_outcome), so
        only the flush-failure FAILED record goes through the repository
        method.
        """
        _db, factory, processor, transform, _agg_node = self._setup_batch_processor(output_mode="passthrough")
        source_row = _make_source_row({"value": 10})
        ctx = make_context(landscape=factory.plugin_audit_writer())
        captured: dict[str, TokenInfo] = {}

        def accept_side_effect(node_id: NodeID, token: TokenInfo, *, accept_time: float | None = None) -> None:
            captured["token"] = token

        def execute_flush_side_effect(*, node_id, transform, ctx, trigger_type):
            return (
                TransformResult.error({"reason": "flush_failed"}, retryable=False),
                [captured["token"]],
                "batch-1",
            )

        with (
            patch.object(processor._aggregation_executor, "accept_adopted_row", side_effect=accept_side_effect),
            patch.object(processor._aggregation_executor, "check_flush_status", return_value=(True, TriggerType.COUNT)),
            patch.object(processor._aggregation_executor, "execute_flush", side_effect=execute_flush_side_effect),
            patch.object(factory.data_flow, "record_token_outcome") as record_outcome,
        ):
            results = processor.process_row(
                row_index=0,
                source_row=source_row,
                transforms=[transform],
                ctx=ctx,
                source_row_index=0,
                ingest_sequence=0,
            )

        assert len(results) == 2
        _assert_outcome_pair(results[0], None, TerminalPath.BUFFERED)
        _assert_outcome_pair(results[1], TerminalOutcome.FAILURE, TerminalPath.UNROUTED)
        # The intake adoption verb wrote the BUFFERED record durably inside
        # its fenced transaction; the repository method sees only the flush
        # failure's FAILED record.
        assert [(call.kwargs["outcome"], call.kwargs["path"]) for call in record_outcome.call_args_list] == [
            (TerminalOutcome.FAILURE, TerminalPath.UNROUTED),
        ]

    def test_flush_failure_transform_records_failed_for_buffered_tokens(self) -> None:
        """T26: Transform-mode flush failure records FAILED for BUFFERED tokens.

        Before T26, transform-mode buffer time recorded CONSUMED_IN_BATCH (terminal),
        so flush failures couldn't record FAILED. Now tokens are BUFFERED (non-terminal)
        at buffer time, allowing FAILED to be recorded on flush error.
        """
        _db, factory, processor, transform, _agg_node = self._setup_batch_processor(output_mode="transform")
        source_row = _make_source_row({"value": 10})
        ctx = make_context(landscape=factory.plugin_audit_writer())
        captured: dict[str, TokenInfo] = {}

        def accept_side_effect(node_id: NodeID, token: TokenInfo, *, accept_time: float | None = None) -> None:
            captured["token"] = token

        def execute_flush_side_effect(*, node_id, transform, ctx, trigger_type):
            return (
                TransformResult.error({"reason": "flush_failed"}, retryable=False),
                [captured["token"]],
                "batch-1",
            )

        with (
            patch.object(processor._aggregation_executor, "accept_adopted_row", side_effect=accept_side_effect),
            patch.object(processor._aggregation_executor, "check_flush_status", return_value=(True, TriggerType.COUNT)),
            patch.object(processor._aggregation_executor, "execute_flush", side_effect=execute_flush_side_effect),
            patch.object(factory.data_flow, "record_token_outcome") as record_outcome,
        ):
            results = processor.process_row(
                row_index=0,
                source_row=source_row,
                transforms=[transform],
                ctx=ctx,
                source_row_index=0,
                ingest_sequence=0,
            )

        # Slice 3 re-pin (ADR-030 §E.2): the arrival returns a real BUFFERED
        # RowResult; the count flush fires from the next iteration's intake.
        assert len(results) == 2
        _assert_outcome_pair(results[0], None, TerminalPath.BUFFERED)
        _assert_outcome_pair(results[1], TerminalOutcome.FAILURE, TerminalPath.UNROUTED)
        outcomes = [(call.kwargs["outcome"], call.kwargs["path"]) for call in record_outcome.call_args_list]
        # The intake adoption verb wrote the BUFFERED record durably inside
        # its fenced transaction; only the flush failure's FAILED record goes
        # through the repository method.
        assert outcomes == [(TerminalOutcome.FAILURE, TerminalPath.UNROUTED)]

    def test_passthrough_success_with_rows_none_raises(self) -> None:
        """Passthrough flush requires rows list; rows=None is an invariant violation."""
        _db, factory, processor, transform, _agg_node = self._setup_batch_processor(output_mode="passthrough")
        source_row = _make_source_row({"value": 10})
        ctx = make_context(landscape=factory.plugin_audit_writer())
        captured: dict[str, TokenInfo] = {}

        bad_result = SimpleNamespace(status="success", is_multi_row=True, rows=None)

        def accept_side_effect(node_id: NodeID, token: TokenInfo, *, accept_time: float | None = None) -> None:
            captured["token"] = token

        def execute_flush_side_effect(*, node_id, transform, ctx, trigger_type):
            return bad_result, [captured["token"]], "batch-1"

        with (
            patch.object(processor._aggregation_executor, "accept_adopted_row", side_effect=accept_side_effect),
            patch.object(processor._aggregation_executor, "check_flush_status", return_value=(True, TriggerType.COUNT)),
            patch.object(processor._aggregation_executor, "execute_flush", side_effect=execute_flush_side_effect),
            patch.object(processor._data_flow, "record_token_outcome"),
            patch.object(processor, "_emit_transform_completed"),
            pytest.raises(RuntimeError, match="rows=None"),
        ):
            processor.process_row(
                row_index=0,
                source_row=source_row,
                transforms=[transform],
                ctx=ctx,
                source_row_index=0,
                ingest_sequence=0,
            )

    def test_passthrough_success_with_output_count_mismatch_raises(self) -> None:
        """Passthrough flush must return one output row per buffered input token."""
        _db, factory, processor, transform, _agg_node = self._setup_batch_processor(output_mode="passthrough")
        source_row = _make_source_row({"value": 10})
        ctx = make_context(landscape=factory.plugin_audit_writer())
        captured: dict[str, TokenInfo] = {}

        mismatch_result = TransformResult.success_multi(
            [make_row({"value": 100}, contract=_make_contract())],
            success_reason={"action": "mismatch"},
        )

        def accept_side_effect(node_id: NodeID, token: TokenInfo, *, accept_time: float | None = None) -> None:
            captured["token"] = token

        def execute_flush_side_effect(*, node_id, transform, ctx, trigger_type):
            other_token = make_token_info(data={"value": 20})
            return mismatch_result, [captured["token"], other_token], "batch-1"

        with (
            patch.object(processor._aggregation_executor, "accept_adopted_row", side_effect=accept_side_effect),
            patch.object(processor._aggregation_executor, "check_flush_status", return_value=(True, TriggerType.COUNT)),
            patch.object(processor._aggregation_executor, "execute_flush", side_effect=execute_flush_side_effect),
            patch.object(processor._data_flow, "record_token_outcome"),
            patch.object(processor, "_emit_transform_completed"),
            pytest.raises(OrchestrationInvariantError, match="same number of output rows"),
        ):
            processor.process_row(
                row_index=0,
                source_row=source_row,
                transforms=[transform],
                ctx=ctx,
                source_row_index=0,
                ingest_sequence=0,
            )

    def test_timeout_flush_passthrough_with_downstream_returns_continuation_work(self) -> None:
        """Timeout flush routes passthrough tokens into child work when downstream exists."""
        downstream_node = NodeID("downstream-2")
        agg_node = NodeID("agg-1")
        _db, factory, processor, transform, agg_node = self._setup_batch_processor(
            output_mode="passthrough",
            node_to_next={NodeID("source-0"): agg_node, agg_node: downstream_node, downstream_node: None},
        )
        ctx = make_context(landscape=factory.plugin_audit_writer())

        result = TransformResult.success_multi(
            [make_row({"value": 11}, contract=_make_contract())],
            success_reason={"action": "passthrough"},
        )
        buffered_token = make_token_info(data={"value": 10})
        _persist_blocked_scheduler_work(factory, processor, buffered_token, node_id=agg_node, barrier_key=str(agg_node))

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

        assert results == ()
        assert len(child_items) == 1
        assert child_items[0].current_node_id == downstream_node

    def test_timeout_flush_passthrough_terminal_returns_completed(self) -> None:
        """Timeout flush returns terminal COMPLETED results when no downstream/coalesce exists."""
        _db, factory, processor, transform, agg_node = self._setup_batch_processor(output_mode="passthrough")
        ctx = make_context(landscape=factory.plugin_audit_writer())

        result = TransformResult.success_multi(
            [make_row({"value": 11}, contract=_make_contract())],
            success_reason={"action": "passthrough"},
        )
        buffered_token = make_token_info(data={"value": 10})
        _persist_blocked_scheduler_work(factory, processor, buffered_token, node_id=agg_node, barrier_key=str(agg_node))

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
        _assert_outcome_pair(results[0], TerminalOutcome.SUCCESS, TerminalPath.DEFAULT_FLOW)
        assert results[0].sink_name == "agg_sink"

    def test_transform_mode_triggering_token_quarantined_outcome(self) -> None:
        """Triggering token's RowResult must be QUARANTINED when quarantined_indices includes it.

        Regression: the triggering token (count-triggered flush) was unconditionally
        returned as CONSUMED_IN_BATCH in its RowResult, even when data_flow had
        already recorded it as QUARANTINED. This caused outcome disagreement between
        the audit trail and in-memory control flow.
        """
        _db, factory, processor, transform, _agg_node = self._setup_batch_processor(output_mode="transform")
        source_row = _make_source_row({"value": 10})
        ctx = make_context(landscape=factory.plugin_audit_writer())
        captured: dict[str, TokenInfo] = {}
        valid_buffered_token = make_token_info(row_id="row-a", token_id="token-valid", data={"value": 1})
        _persist_blocked_scheduler_work(factory, processor, valid_buffered_token, node_id=NodeID("agg-1"), barrier_key="agg-1")

        # The triggering token is the second buffered token. Index 1 in
        # quarantined_indices means the triggering token IS quarantined while
        # the batch still has a valid parent for emitted output.
        flush_result = TransformResult.success(
            make_row({"value": 999}, contract=_make_contract()),
            success_reason={
                "action": "batch_processed",
                "metadata": {"quarantined_indices": [1]},
            },
        )

        def accept_side_effect(node_id: NodeID, token: TokenInfo, *, accept_time: float | None = None) -> None:
            captured["token"] = token

        def execute_flush_side_effect(*, node_id, transform, ctx, trigger_type):
            return flush_result, [valid_buffered_token, captured["token"]], "batch-1"

        with (
            patch.object(processor._aggregation_executor, "accept_adopted_row", side_effect=accept_side_effect),
            patch.object(processor._aggregation_executor, "check_flush_status", return_value=(True, TriggerType.COUNT)),
            patch.object(processor._aggregation_executor, "execute_flush", side_effect=execute_flush_side_effect),
            patch.object(factory.data_flow, "record_token_outcome") as record_outcome,
            patch.object(processor, "_emit_transform_completed"),
            patch.object(processor, "_emit_token_completed"),
            patch.object(processor._token_manager, "expand_token", return_value=([], "expand-group-1")),
        ):
            results = processor.process_row(
                row_index=1,
                source_row=source_row,
                transforms=[transform],
                ctx=ctx,
                source_row_index=1,
                ingest_sequence=1,
            )

        # The triggering token must be returned as QUARANTINED, matching the data_flow.
        triggering_results = [r for r in results if (r.outcome, r.path) == (TerminalOutcome.FAILURE, TerminalPath.QUARANTINED_AT_SOURCE)]
        assert len(triggering_results) == 1, (
            f"Expected triggering token RowResult with QUARANTINED outcome, got outcomes: {[r.outcome for r in results]}"
        )

        # Recorder must also have recorded the triggering token as QUARANTINED,
        # not CONSUMED_IN_BATCH. Other valid buffered tokens are still consumed.
        triggering_token_id = captured["token"].token_id
        recorded = [
            (
                call.kwargs["ref"].token_id,
                call.kwargs["outcome"],
                call.kwargs["path"],
            )
            for call in record_outcome.call_args_list
        ]
        assert (triggering_token_id, TerminalOutcome.FAILURE, TerminalPath.QUARANTINED_AT_SOURCE) in recorded
        assert (triggering_token_id, TerminalOutcome.TRANSIENT, TerminalPath.BATCH_CONSUMED) not in recorded

    def test_transform_mode_triggering_token_consumed_when_not_quarantined(self) -> None:
        """Triggering token's RowResult is CONSUMED_IN_BATCH when not in quarantined_indices.

        Companion to the quarantine test — verifies the normal (non-quarantined) path
        still works correctly after the fix.
        """
        _db, factory, processor, transform, _agg_node = self._setup_batch_processor(output_mode="transform")
        source_row = _make_source_row({"value": 10})
        ctx = make_context(landscape=factory.plugin_audit_writer())
        captured: dict[str, TokenInfo] = {}

        # No quarantined_indices — all tokens are consumed normally.
        flush_result = TransformResult.success(
            make_row({"value": 999}, contract=_make_contract()),
            success_reason={"action": "batch_processed"},
        )

        def accept_side_effect(node_id: NodeID, token: TokenInfo, *, accept_time: float | None = None) -> None:
            captured["token"] = token

        def execute_flush_side_effect(*, node_id, transform, ctx, trigger_type):
            return flush_result, [captured["token"]], "batch-1"

        with (
            patch.object(processor._aggregation_executor, "accept_adopted_row", side_effect=accept_side_effect),
            patch.object(processor._aggregation_executor, "check_flush_status", return_value=(True, TriggerType.COUNT)),
            patch.object(processor._aggregation_executor, "execute_flush", side_effect=execute_flush_side_effect),
            patch.object(factory.data_flow, "record_token_outcome") as record_outcome,
            patch.object(processor, "_emit_transform_completed"),
            patch.object(processor, "_emit_token_completed"),
            patch.object(processor._token_manager, "expand_token", return_value=([], "expand-group-1")),
        ):
            results = processor.process_row(
                row_index=0,
                source_row=source_row,
                transforms=[transform],
                ctx=ctx,
                source_row_index=0,
                ingest_sequence=0,
            )

        # Slice 3 re-pin (ADR-030 §E.2): the trigger arrival surfaces as a
        # real (None, BUFFERED) RowResult and the intake-fired flush takes the
        # out-of-claim shape (triggering_token=None) — no extra
        # CONSUMED_IN_BATCH RowResult is emitted for the trigger member. Its
        # consumption lives in the audit trail.
        assert [(r.outcome, r.path) for r in results] == [(None, TerminalPath.BUFFERED)]

        # Recorder should have CONSUMED_IN_BATCH, not QUARANTINED.
        recorded_pairs = [(call.kwargs["outcome"], call.kwargs["path"]) for call in record_outcome.call_args_list]
        assert (TerminalOutcome.TRANSIENT, TerminalPath.BATCH_CONSUMED) in recorded_pairs
        assert (TerminalOutcome.FAILURE, TerminalPath.QUARANTINED_AT_SOURCE) not in recorded_pairs

    def test_transform_mode_count_flush_expands_from_non_quarantined_parent(self) -> None:
        """Count-triggered batch output children must not use a quarantined triggering token as parent."""
        _db, factory, processor, transform, _agg_node = self._setup_batch_processor(output_mode="transform")
        source_row = _make_source_row({"value": 10})
        ctx = make_context(landscape=factory.plugin_audit_writer())
        captured: dict[str, TokenInfo] = {}
        valid_buffered_token = make_token_info(row_id="row-a", token_id="token-valid", data={"value": 1})
        _persist_blocked_scheduler_work(factory, processor, valid_buffered_token, node_id=NodeID("agg-1"), barrier_key="agg-1")

        flush_result = TransformResult.success(
            make_row({"value": 999}, contract=_make_contract()),
            success_reason={
                "action": "batch_processed",
                "metadata": {"quarantined_indices": [1]},
            },
        )

        def accept_side_effect(node_id: NodeID, token: TokenInfo, *, accept_time: float | None = None) -> None:
            captured["token"] = token

        def execute_flush_side_effect(*, node_id, transform, ctx, trigger_type):
            return flush_result, [valid_buffered_token, captured["token"]], "batch-1"

        with (
            patch.object(processor._aggregation_executor, "accept_adopted_row", side_effect=accept_side_effect),
            patch.object(processor._aggregation_executor, "check_flush_status", return_value=(True, TriggerType.COUNT)),
            patch.object(processor._aggregation_executor, "execute_flush", side_effect=execute_flush_side_effect),
            patch.object(factory.data_flow, "record_token_outcome"),
            patch.object(processor, "_emit_transform_completed"),
            patch.object(processor, "_emit_token_completed"),
            patch.object(processor._token_manager, "expand_token", return_value=([], "expand-group-1")) as expand_token,
        ):
            processor.process_row(
                row_index=1,
                source_row=source_row,
                transforms=[transform],
                ctx=ctx,
                source_row_index=1,
                ingest_sequence=1,
            )

        assert expand_token.call_args is not None
        assert expand_token.call_args.kwargs["parent_token"] == valid_buffered_token

    def test_transform_mode_timeout_flush_expands_from_non_quarantined_parent(self) -> None:
        """Timeout/end-of-source batch output children must not use a quarantined first buffered token as parent."""
        _db, factory, processor, transform, agg_node = self._setup_batch_processor(output_mode="transform")
        ctx = make_context(landscape=factory.plugin_audit_writer())
        quarantined_token = make_token_info(row_id="row-a", token_id="token-quarantined", data={"value": 1})
        valid_token = make_token_info(row_id="row-b", token_id="token-valid", data={"value": 2})
        _persist_blocked_scheduler_work(
            factory, processor, quarantined_token, node_id=agg_node, barrier_key=str(agg_node), ingest_sequence=0
        )
        _persist_blocked_scheduler_work(factory, processor, valid_token, node_id=agg_node, barrier_key=str(agg_node), ingest_sequence=1)

        flush_result = TransformResult.success(
            make_row({"value": 999}, contract=_make_contract()),
            success_reason={
                "action": "batch_processed",
                "metadata": {"quarantined_indices": [0]},
            },
        )

        with (
            patch.object(
                processor._aggregation_executor,
                "execute_flush",
                return_value=(flush_result, [quarantined_token, valid_token], "batch-1"),
            ),
            patch.object(factory.data_flow, "record_token_outcome"),
            patch.object(processor, "_emit_transform_completed"),
            patch.object(processor, "_emit_token_completed"),
            patch.object(processor._token_manager, "expand_token", return_value=([], "expand-group-1")) as expand_token,
        ):
            processor.handle_timeout_flush(
                node_id=agg_node,
                transform=transform,
                ctx=ctx,
                trigger_type=TriggerType.TIMEOUT,
            )

        assert expand_token.call_args is not None
        assert expand_token.call_args.kwargs["parent_token"] == valid_token

    def test_transform_mode_out_of_range_quarantined_index_fails_before_expansion(self) -> None:
        """Malformed batch quarantine metadata must fail before child tokens are created."""
        _db, _factory, processor, transform, _agg_node = self._setup_batch_processor(output_mode="transform")
        token = make_token_info(row_id="row-a", token_id="token-a", data={"value": 1})
        fctx = _FlushContext(
            node_id=NodeID("agg-1"),
            transform=transform,
            settings=AggregationSettings(
                name="batch_agg",
                plugin="agg-transform",
                input="default",
                on_error="discard",
                trigger={"count": 1},
                output_mode="transform",
            ),
            buffered_tokens=(token,),
            batch_id="batch-1",
            error_msg="Batch transform failed",
            expand_parent_token=token,
            triggering_token=token,
            coalesce_node_id=None,
            coalesce_name=None,
        )
        flush_result = TransformResult.success(
            make_row({"value": 999}, contract=_make_contract()),
            success_reason={
                "action": "batch_processed",
                "metadata": {"quarantined_indices": [1]},
            },
        )

        with (
            patch.object(processor._token_manager, "expand_token") as expand_token,
            pytest.raises(OrchestrationInvariantError, match="quarantined_indices"),
        ):
            processor._route_transform_results(fctx, flush_result)

        expand_token.assert_not_called()


class TestTransformModeOutcomeOrdering:
    """Regression tests for P0: parent outcomes must NOT be recorded before downstream validation.

    Bug: _route_transform_results recorded CONSUMED_IN_BATCH/QUARANTINED for parent
    tokens BEFORE validating expected_output_count and before expand_token(). If either
    of those failed, the audit trail showed terminal outcomes for parents with no child
    token — recovery would skip the row (silent data loss).

    Fix: outcome recording must happen AFTER both cardinality validation and expand_token
    succeed. On failure, parent tokens must remain BUFFERED (non-terminal) so recovery
    can retry them.
    """

    def _setup_batch_processor(
        self,
        *,
        expected_output_count: int | None = None,
    ) -> tuple[LandscapeDB, RecorderFactory, RowProcessor, Mock, NodeID]:
        """Create a transform-mode batch processor with optional expected_output_count."""
        db, factory = _make_factory()
        source_node = NodeID("source-0")
        agg_node = NodeID("agg-1")

        transform = _make_mock_transform(
            node_id=str(agg_node),
            name="agg-transform",
            is_batch_aware=True,
            on_success="agg_sink",
        )

        processor = _make_processor(
            factory,
            node_step_map={source_node: 0, agg_node: 1},
            node_to_next={source_node: agg_node, agg_node: None},
            node_to_plugin={agg_node: transform},
            aggregation_settings={
                agg_node: AggregationSettings(
                    name="batch_agg",
                    plugin="agg-transform",
                    input="default",
                    on_error="discard",
                    trigger={"count": 1},
                    output_mode="transform",
                    expected_output_count=expected_output_count,
                ),
            },
        )
        return db, factory, processor, transform, agg_node

    def test_cardinality_mismatch_does_not_record_parent_terminal_outcome(self) -> None:
        """Parent tokens must NOT be CONSUMED_IN_BATCH when expected_output_count fails.

        If the cardinality check raises after outcomes are recorded, recovery
        sees terminal parents with no children — silent data loss.
        """
        _db, factory, processor, transform, _agg_node = self._setup_batch_processor(
            expected_output_count=5,  # Will mismatch: transform returns 1 row
        )
        source_row = _make_source_row({"value": 10})
        ctx = make_context(landscape=factory.plugin_audit_writer())
        captured: dict[str, TokenInfo] = {}

        # Transform returns 1 output row but expected_output_count=5 → RuntimeError
        flush_result = TransformResult.success(
            make_row({"value": 999}, contract=_make_contract()),
            success_reason={"action": "batch_processed"},
        )

        def accept_side_effect(node_id: NodeID, token: TokenInfo, *, accept_time: float | None = None) -> None:
            captured["token"] = token

        def execute_flush_side_effect(*, node_id, transform, ctx, trigger_type):
            return flush_result, [captured["token"]], "batch-1"

        with (
            patch.object(processor._aggregation_executor, "accept_adopted_row", side_effect=accept_side_effect),
            patch.object(processor._aggregation_executor, "check_flush_status", return_value=(True, TriggerType.COUNT)),
            patch.object(processor._aggregation_executor, "execute_flush", side_effect=execute_flush_side_effect),
            patch.object(factory.data_flow, "record_token_outcome") as record_outcome,
            patch.object(processor, "_emit_transform_completed"),
            patch.object(processor, "_emit_token_completed"),
            pytest.raises(RuntimeError, match="expected_output_count=5"),
        ):
            processor.process_row(
                row_index=0,
                source_row=source_row,
                transforms=[transform],
                ctx=ctx,
                source_row_index=0,
                ingest_sequence=0,
            )

        # Parent tokens must NOT have been recorded as terminal before the error.
        # Only BUFFERED is acceptable (non-terminal, allows recovery retry).
        recorded_pairs = [(call.kwargs["outcome"], call.kwargs["path"]) for call in record_outcome.call_args_list]
        assert (TerminalOutcome.TRANSIENT, TerminalPath.BATCH_CONSUMED) not in recorded_pairs, (
            f"CONSUMED_IN_BATCH was recorded before cardinality check failed — "
            f"recovery would skip this row. Recorded pairs: {recorded_pairs}"
        )

    def test_expand_token_failure_does_not_record_parent_terminal_outcome(self) -> None:
        """Parent tokens must NOT be CONSUMED_IN_BATCH when expand_token raises.

        If expand_token fails after outcomes are recorded, the audit trail shows
        terminal parents with no expanded children — silent data loss on recovery.
        """
        _db, factory, processor, transform, _agg_node = self._setup_batch_processor()
        source_row = _make_source_row({"value": 10})
        ctx = make_context(landscape=factory.plugin_audit_writer())
        captured: dict[str, TokenInfo] = {}

        flush_result = TransformResult.success(
            make_row({"value": 999}, contract=_make_contract()),
            success_reason={"action": "batch_processed"},
        )

        def accept_side_effect(node_id: NodeID, token: TokenInfo, *, accept_time: float | None = None) -> None:
            captured["token"] = token

        def execute_flush_side_effect(*, node_id, transform, ctx, trigger_type):
            return flush_result, [captured["token"]], "batch-1"

        with (
            patch.object(processor._aggregation_executor, "accept_adopted_row", side_effect=accept_side_effect),
            patch.object(processor._aggregation_executor, "check_flush_status", return_value=(True, TriggerType.COUNT)),
            patch.object(processor._aggregation_executor, "execute_flush", side_effect=execute_flush_side_effect),
            patch.object(factory.data_flow, "record_token_outcome") as record_outcome,
            patch.object(processor, "_emit_transform_completed"),
            patch.object(processor, "_emit_token_completed"),
            patch.object(
                processor._token_manager,
                "expand_token",
                side_effect=RuntimeError("expand_token DB integrity error"),
            ),
            pytest.raises(RuntimeError, match="expand_token DB integrity error"),
        ):
            processor.process_row(
                row_index=0,
                source_row=source_row,
                transforms=[transform],
                ctx=ctx,
                source_row_index=0,
                ingest_sequence=0,
            )

        # Parent tokens must NOT have terminal outcomes recorded before expand_token.
        recorded_pairs = [(call.kwargs["outcome"], call.kwargs["path"]) for call in record_outcome.call_args_list]
        assert (TerminalOutcome.TRANSIENT, TerminalPath.BATCH_CONSUMED) not in recorded_pairs, (
            f"CONSUMED_IN_BATCH was recorded before expand_token failed — recovery would skip this row. Recorded pairs: {recorded_pairs}"
        )

    def test_parent_terminal_outcome_recorder_failure_raises_audit_integrity_error(self) -> None:
        """Recorder failure after expansion must surface as audit corruption, not a raw DB error."""
        _db, factory, processor, transform, agg_node = self._setup_batch_processor()
        first_token = make_token_info(row_id="row-a", token_id="token-a", data={"value": 10})
        second_token = make_token_info(row_id="row-b", token_id="token-b", data={"value": 20})
        child_token = make_token_info(row_id="row-child", token_id="token-child", data={"value": 999})
        fctx = _FlushContext(
            node_id=agg_node,
            transform=transform,
            settings=AggregationSettings(
                name="batch_agg",
                plugin="agg-transform",
                input="default",
                on_error="discard",
                trigger={"count": 2},
                output_mode="transform",
            ),
            buffered_tokens=(first_token, second_token),
            batch_id="batch-1",
            error_msg="Batch transform failed",
            expand_parent_token=first_token,
            triggering_token=second_token,
            coalesce_node_id=None,
            coalesce_name=None,
        )
        flush_result = TransformResult.success(
            make_row({"value": 999}, contract=_make_contract()),
            success_reason={
                "action": "batch_processed",
                "metadata": {"quarantined_indices": [1]},
            },
        )

        with (
            patch.object(
                processor._token_manager,
                "expand_token",
                return_value=([child_token], "expand-group-1"),
            ),
            patch.object(
                factory.data_flow,
                "record_token_outcome",
                side_effect=[None, LandscapeRecordError("audit DB down")],
            ),
            patch.object(processor, "_emit_token_completed"),
            pytest.raises(AuditIntegrityError, match="Failed to record batch parent terminal outcome") as exc_info,
        ):
            processor._route_transform_results(fctx, flush_result)

        assert isinstance(exc_info.value.__cause__, LandscapeRecordError)


class TestProcessRowGateBranching:
    """Tests for non-linear gate branching through next_node_id."""

    def test_config_gate_processing_node_jump_preloads_subchain_sink_for_expanded_children(self) -> None:
        """Config gate PROCESSING_NODE jumps should refresh inherited sink from jumped subchain."""
        _db, factory = _make_factory()
        source_row = _make_source_row({"value": 10})
        ctx = make_context(landscape=factory.plugin_audit_writer())

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
            factory,
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
                source_row_index=0,
                ingest_sequence=0,
            )

        completed = [r for r in results if (r.outcome, r.path) == (TerminalOutcome.SUCCESS, TerminalPath.DEFAULT_FLOW)]
        assert len(completed) == 2
        assert inherited_sinks == ["branch_sink", "branch_sink"]
        assert all(r.sink_name == "branch_sink" for r in completed)

    def test_jump_target_terminal_coalesce_missing_on_success_mapping_raises(self) -> None:
        """Terminal coalesce reached via jump must have an on_success sink mapping."""
        _db, factory = _make_factory()

        source_node = NodeID("source-0")
        router_node = NodeID("router-1")
        coalesce_node = NodeID("coalesce::merge")

        processor = _make_processor(
            factory,
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
            coalesce_node_ids={CoalesceName("merge"): coalesce_node},
            # Intentionally omit coalesce_on_success_map
            # router-1 is plugin-less walk topology — declare it structural so
            # the jump walk reaches the coalesce on_success invariant under test.
            structural_node_ids=frozenset({source_node, router_node}),
        )

        with pytest.raises(OrchestrationInvariantError, match="Coalesce 'merge' not in on_success map"):
            processor._nav.resolve_jump_target_sink(router_node)

    def test_jump_target_resolution_raises_when_no_sink_and_no_gate(self) -> None:
        """Jump path with only transforms and no terminal sink must fail closed."""
        _db, factory = _make_factory()

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
            factory,
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
            node_to_plugin={
                jump_start_node: branch_transform1,
                downstream_transform_node: branch_transform2,
            },
        )

        with pytest.raises(OrchestrationInvariantError, match="no sink"):
            processor._nav.resolve_jump_target_sink(jump_start_node)

    def test_branch_to_sink_routing_applies_for_terminal_fork_children(self) -> None:
        """Branch-routed tokens bypassing coalesce should resolve sink via branch_to_sink."""
        _db, factory = _make_factory()
        ctx = make_context(landscape=factory.plugin_audit_writer())
        token = TokenInfo(
            row_id="row-1",
            token_id="token-branch-1",
            row_data=make_row({"value": 1}),
            branch_name="path_a",
        )

        processor = _make_processor(
            factory,
            source_on_success="source_sink",
            branch_to_sink={BranchName("path_a"): "branch_sink"},
            sink_names=frozenset({"source_sink", "branch_sink"}),
            node_step_map={NodeID("source-0"): 0},
            node_to_next={NodeID("source-0"): None},
        )
        _persist_token_for_scheduler(factory, token)

        results = processor.process_token(
            token=token,
            ctx=ctx,
            current_node_id=None,  # type: ignore[arg-type]  # Intentional: tests branch routing when fork child has no starting node
        )

        assert len(results) == 1
        _assert_outcome_pair(results[0], TerminalOutcome.SUCCESS, TerminalPath.DEFAULT_FLOW)
        assert results[0].sink_name == "branch_sink"

    def test_fork_to_sink_children_bypass_gate_continuation_successor(self) -> None:
        """Regression: fork children in _branch_to_sink must not traverse downstream nodes.

        Topology: gate-1 → transform-1 (gate's structural successor)
        Gate forks to branches sink_a and sink_b (both in _branch_to_sink).

        Without fix: children get current_node_id=transform-1 and execute the transform.
        With fix: children get current_node_id=None, skip the loop, resolve via _branch_to_sink.
        """
        _db, factory = _make_factory()
        ctx = make_context(landscape=factory.plugin_audit_writer())

        gate_node = NodeID("gate-1")
        transform_node = NodeID("transform-1")
        source_node = NodeID("source-0")

        # Register nodes for FK constraints
        factory.data_flow.register_node(
            run_id="test-run",
            plugin_name="fork-gate",
            node_type=NodeType.GATE,
            plugin_version="1.0",
            config={},
            node_id="gate-1",
            schema_config=_DYNAMIC_SCHEMA,
        )
        factory.data_flow.register_node(
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
            result=TransformResult.success(make_pipeline_row({"value": 99, "transformed": True}), success_reason={"action": "test"}),
        )

        processor = _make_processor(
            factory,
            source_on_success="sink_c",
            branch_to_sink={BranchName("sink_a"): "sink_a", BranchName("sink_b"): "sink_b"},
            sink_names=frozenset({"sink_a", "sink_b", "sink_c"}),
            node_step_map={source_node: 0, gate_node: 1, transform_node: 2},
            node_to_next={source_node: gate_node, gate_node: transform_node, transform_node: None},
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
            _persist_token_for_scheduler(factory, child_a)
            _persist_token_for_scheduler(factory, child_b)
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

        processor._gate_executor.execute_config_gate = mock_execute_config_gate  # type: ignore[method-assign]

        source_row = _make_source_row()
        results = processor.process_row(
            row_index=0,
            source_row=source_row,
            transforms=[],
            ctx=ctx,
            source_row_index=0,
            ingest_sequence=0,
        )

        # Parent should be FORKED
        forked = [r for r in results if (r.outcome, r.path) == (TerminalOutcome.TRANSIENT, TerminalPath.FORK_PARENT)]
        assert len(forked) == 1

        # Fork children should complete at their branch sinks
        completed = [r for r in results if (r.outcome, r.path) == (TerminalOutcome.SUCCESS, TerminalPath.DEFAULT_FLOW)]
        assert len(completed) == 2
        sink_names = sorted(r.sink_name for r in completed if r.sink_name is not None)
        assert sink_names == ["sink_a", "sink_b"]

        # Downstream transform must NOT have been called for fork children
        transform.process.assert_not_called()

    def test_overlapping_branch_to_coalesce_and_branch_to_sink_raises(self) -> None:
        """A branch name in both branch_to_coalesce and branch_to_sink is an invariant violation."""
        _db, factory = _make_factory()
        with pytest.raises(OrchestrationInvariantError, match="both branch_to_coalesce and branch_to_sink"):
            _make_processor(
                factory,
                source_on_success="output",
                branch_to_coalesce={BranchName("path_a"): CoalesceName("merge_point")},
                branch_to_sink={BranchName("path_a"): "direct_sink"},
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
        _db, factory = _make_factory()
        source_row = _make_source_row()
        ctx = make_context(landscape=factory.plugin_audit_writer())

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
            factory,
            node_step_map={source_node: 0, transform_node: 1},
            node_to_next={source_node: transform_node, transform_node: None},
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
                source_row_index=0,
                ingest_sequence=0,
            )

        # Parent should be EXPANDED, children should be COMPLETED
        pairs = {(r.outcome, r.path) for r in results}
        assert (TerminalOutcome.TRANSIENT, TerminalPath.EXPAND_PARENT) in pairs
        assert (TerminalOutcome.SUCCESS, TerminalPath.DEFAULT_FLOW) in pairs

    def test_multi_row_without_creates_tokens_raises(self) -> None:
        """Transform returning multi-row without creates_tokens=True → RuntimeError."""
        _db, factory = _make_factory()
        source_row = _make_source_row()
        ctx = make_context(landscape=factory.plugin_audit_writer())

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
            factory,
            node_step_map={source_node: 0, transform_node: 1},
            node_to_next={source_node: transform_node, transform_node: None},
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
                source_row_index=0,
                ingest_sequence=0,
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
        _db, factory = _make_factory()

        processor = _make_processor(factory)

        contract = _make_contract()
        row_data = make_row({"value": 42}, contract=contract)
        ctx = make_context(landscape=factory.plugin_audit_writer())

        # We need a pre-existing row. Create one via process_row first.
        source_row = _make_source_row({"value": 42})
        first_results = processor.process_row(
            row_index=0,
            source_row=source_row,
            transforms=[],
            ctx=ctx,
            source_row_index=0,
            ingest_sequence=0,
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
        _assert_outcome_pair(results[0], TerminalOutcome.SUCCESS, TerminalPath.DEFAULT_FLOW)
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
        _db, factory = _make_factory()

        processor = _make_processor(factory)
        ctx = make_context(landscape=factory.plugin_audit_writer())

        # Create a token to process
        token = make_token_info(data={"value": 42})
        _persist_token_for_scheduler(factory, token)

        results = processor.process_token(
            token=token,
            ctx=ctx,
            current_node_id=NodeID("source-0"),
        )

        assert len(results) == 1
        _assert_outcome_pair(results[0], TerminalOutcome.SUCCESS, TerminalPath.DEFAULT_FLOW)
        assert results[0].sink_name == "default"

    def test_terminal_coalesce_continuation_uses_coalesce_on_success_sink(self) -> None:
        """Merged token resumed at terminal coalesce must route to coalesce sink, not source sink."""
        _db, factory = _make_factory()

        processor = _make_processor(
            factory,
            source_on_success="source_sink",
            coalesce_node_ids={CoalesceName("merge"): NodeID("coalesce::merge")},
            coalesce_on_success_map={CoalesceName("merge"): "coalesce_sink"},
            node_step_map={NodeID("coalesce::merge"): 1},
            node_to_next={NodeID("coalesce::merge"): None},
        )
        ctx = make_context(landscape=factory.plugin_audit_writer())
        token = make_token_info(data={"value": 42})
        _persist_token_for_scheduler(factory, token)

        results = processor.process_token(
            token=token,
            ctx=ctx,
            current_node_id=NodeID("coalesce::merge"),
            coalesce_node_id=NodeID("coalesce::merge"),
            coalesce_name=CoalesceName("merge"),
        )

        assert len(results) == 1
        _assert_outcome_pair(results[0], TerminalOutcome.SUCCESS, TerminalPath.DEFAULT_FLOW)
        assert results[0].sink_name == "coalesce_sink"

    def test_non_terminal_coalesce_continuation_uses_downstream_sink(self) -> None:
        """Coalesce sink must not override downstream terminal transform routing."""
        _db, factory = _make_factory()
        transform = _make_mock_transform(node_id="transform-1")
        transform.on_success = "downstream_sink"

        processor = _make_processor(
            factory,
            source_on_success="source_sink",
            coalesce_node_ids={CoalesceName("merge"): NodeID("coalesce::merge")},
            coalesce_on_success_map={CoalesceName("merge"): "coalesce_sink"},
            node_step_map={NodeID("coalesce::merge"): 1, NodeID("transform-1"): 2},
            node_to_next={NodeID("coalesce::merge"): NodeID("transform-1"), NodeID("transform-1"): None},
            node_to_plugin={NodeID("transform-1"): transform},
        )
        ctx = make_context(landscape=factory.plugin_audit_writer())
        token = make_token_info(data={"value": 42})
        _persist_token_for_scheduler(factory, token)

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
        _assert_outcome_pair(results[0], TerminalOutcome.SUCCESS, TerminalPath.DEFAULT_FLOW)
        assert results[0].sink_name == "downstream_sink"


# =============================================================================
# _drain_work_queue: Iteration Guard
# =============================================================================


class TestDrainWorkQueueIterationGuard:
    """Tests for the MAX_WORK_QUEUE_ITERATIONS safety limit."""

    def test_iteration_guard_prevents_infinite_loop(self) -> None:
        """Work queue exceeding MAX_WORK_QUEUE_ITERATIONS raises RuntimeError."""
        _db, factory = _make_factory()

        processor = _make_processor(factory)
        ctx = make_context(landscape=factory.plugin_audit_writer())
        token = make_token_info(data={"value": 1})
        _persist_token_for_scheduler(factory, token)
        produced = 0

        # Mock _process_single_token to always produce more work
        def infinite_loop_producer(token, ctx, current_node_id, **kwargs):
            nonlocal produced
            produced += 1
            new_token = make_token_info(row_id=token.row_id, token_id=f"loop-token-{produced}", data={"value": 1})
            _persist_token_for_scheduler(factory, new_token)
            return (None, [WorkItem(token=new_token, current_node_id=NodeID("source-0"))])

        with (
            patch("elspeth.engine.processor.MAX_WORK_QUEUE_ITERATIONS", 3),
            patch.object(processor, "_process_single_token", side_effect=infinite_loop_producer),
            pytest.raises(RuntimeError, match=r"exceeded.*iterations"),
        ):
            processor._drain_work_queue(
                WorkItem(token=token, current_node_id=NodeID("source-0")),
                ctx=ctx,
            )

    def test_iteration_guard_allows_supported_batch_replicate_fanout(self) -> None:
        """The outer queue guard must not reject the largest supported legal fan-out."""
        _db, factory = _make_factory()
        processor = _make_processor(factory)
        ctx = make_context(landscape=factory.plugin_audit_writer())
        token = make_token_info(data={"value": 1})
        _persist_token_for_scheduler(factory, token)
        supported_copies = 3
        remaining_children = supported_copies

        def supported_fanout_producer(token, ctx, current_node_id, **kwargs):
            nonlocal remaining_children
            if remaining_children == 0:
                return (None, [])

            remaining_children -= 1
            child = make_token_info(
                row_id=token.row_id, token_id=f"fanout-token-{remaining_children}", data={"remaining": remaining_children}
            )
            _persist_token_for_scheduler(factory, child)
            return (None, [WorkItem(token=child, current_node_id=NodeID("source-0"))])

        with (
            patch("elspeth.engine.processor.MAX_WORK_QUEUE_ITERATIONS", supported_copies + 2),
            patch.object(processor, "_process_single_token", side_effect=supported_fanout_producer) as mock_process_single_token,
        ):
            results = processor._drain_work_queue(
                WorkItem(token=token, current_node_id=NodeID("source-0")),
                ctx=ctx,
            )

        assert results == []
        assert mock_process_single_token.call_count == supported_copies + 1

    def test_max_iterations_constant_is_reasonable(self) -> None:
        """MAX_WORK_QUEUE_ITERATIONS should be at least 1000."""
        assert MAX_WORK_QUEUE_ITERATIONS >= 1000
        max_supported_copies = BatchReplicateConfig.model_json_schema()["properties"]["max_copies"]["maximum"]
        assert max_supported_copies < MAX_WORK_QUEUE_ITERATIONS


class TestDurableSchedulerResumeDrain:
    """Tests for draining scheduler work created before this processor existed."""

    def test_drains_persisted_ready_work_without_source_replay(self) -> None:
        """A fresh processor rehydrates READY scheduler work from durable row payloads."""
        db, factory = _make_factory()
        transform_node = NodeID("transform-1")
        factory.data_flow.register_node(
            run_id="test-run",
            plugin_name="resume-transform",
            node_type=NodeType.TRANSFORM,
            plugin_version="1.0",
            config={},
            node_id=str(transform_node),
            schema_config=_DYNAMIC_SCHEMA,
        )
        source_payload = make_row({"value": 42})
        row = factory.data_flow.create_row(
            run_id="test-run",
            source_node_id="source-0",
            row_index=0,
            source_row_index=0,
            ingest_sequence=0,
            data=source_payload.to_dict(),
        )
        token = factory.data_flow.create_token(row.row_id, token_id="token-ready")
        factory.scheduler.enqueue_ready(
            run_id="test-run",
            token_id=token.token_id,
            row_id=row.row_id,
            node_id=str(transform_node),
            step_index=1,
            ingest_sequence=0,
            row_payload_json=factory.scheduler.serialize_row_payload(source_payload),
            available_at=datetime.now(UTC),
        )

        transform = _make_mock_transform(node_id=str(transform_node), on_success="default")
        processor = _make_processor(
            factory,
            node_step_map={NodeID("source-0"): 0, transform_node: 1},
            node_to_next={NodeID("source-0"): transform_node, transform_node: None},
            node_to_plugin={transform_node: transform},
            scheduler=factory.scheduler,
        )
        ctx = make_context(landscape=factory.plugin_audit_writer())
        success_result = TransformResult.success(
            make_row({"value": 42, "resumed": True}),
            success_reason={"action": "resume_drain"},
        )

        def executor_side_effect(*, transform, token, ctx, attempt=0):
            assert token.token_id == "token-ready"
            assert token.row_data.to_dict() == {"value": 42}
            return (success_result, token, None)

        with patch.object(processor._transform_executor, "execute_transform", side_effect=executor_side_effect):
            results = processor.drain_scheduled_work(ctx)

        assert len(results) == 1
        _assert_outcome_pair(results[0], TerminalOutcome.SUCCESS, TerminalPath.DEFAULT_FLOW)
        assert results[0].token.token_id == "token-ready"
        from sqlalchemy import select

        from elspeth.core.landscape.schema import token_work_items_table

        with db.connection() as conn:
            statuses = (
                conn.execute(select(token_work_items_table.c.status).where(token_work_items_table.c.run_id == "test-run")).scalars().all()
            )
        assert statuses == ["pending_sink"]

    def _seed_parked_on_error_pending_sink(self, *, error_hash: str | None) -> tuple[Any, Any, Any]:
        """Seed a durable ON_ERROR_ROUTED PENDING_SINK row via production verbs.

        Returns (db, factory, processor) where processor is a FRESH resume
        processor ("resume-worker") that has not yet drained. Both identities
        are registered run_workers (the factory's run already registers its
        leader, so unregistered claimants would be membership-fenced).
        """
        db, factory = _make_factory()
        transform_node = NodeID("transform-1")
        factory.data_flow.register_node(
            run_id="test-run",
            plugin_name="resume-transform",
            node_type=NodeType.TRANSFORM,
            plugin_version="1.0",
            config={},
            node_id=str(transform_node),
            schema_config=_DYNAMIC_SCHEMA,
        )
        source_payload = make_row({"value": 42})
        row = factory.data_flow.create_row(
            run_id="test-run",
            source_node_id="source-0",
            row_index=0,
            source_row_index=0,
            ingest_sequence=0,
            data=source_payload.to_dict(),
        )
        token = factory.data_flow.create_token(row.row_id, token_id="token-parked")
        factory.scheduler.enqueue_ready(
            run_id="test-run",
            token_id=token.token_id,
            row_id=row.row_id,
            node_id=str(transform_node),
            step_index=1,
            ingest_sequence=0,
            row_payload_json=factory.scheduler.serialize_row_payload(source_payload),
            available_at=datetime.now(UTC),
        )
        _register_test_worker(factory, "crashed-worker")
        claimed = factory.scheduler.claim_ready(run_id="test-run", lease_owner="crashed-worker", lease_seconds=300, now=datetime.now(UTC))
        assert claimed is not None
        factory.scheduler.mark_pending_sink(
            work_item_id=claimed.work_item_id,
            row_payload_json=factory.scheduler.serialize_row_payload(source_payload),
            sink_name="error_sink",
            outcome=TerminalOutcome.FAILURE.value,
            path=TerminalPath.ON_ERROR_ROUTED.value,
            error_hash=error_hash,
            # EMPTY original message: the class where a recomputed replay
            # hash diverges from the originally-audited one.
            error_message="",
            now=datetime.now(UTC),
            expected_lease_owner="crashed-worker",
        )
        processor = _make_processor(
            factory,
            node_step_map={NodeID("source-0"): 0, transform_node: 1},
            node_to_next={NodeID("source-0"): transform_node, transform_node: None},
            scheduler=factory.scheduler,
            scheduler_lease_owner="resume-worker",
        )
        return db, factory, processor

    def test_pending_sink_replay_preserves_persisted_error_hash(self) -> None:
        """ON_ERROR_ROUTED replay carries the PERSISTED pending_error_hash
        (elspeth-d74d19f901) instead of letting the accumulator recompute one
        from the synthetic ResumedPendingSink failure evidence."""
        from elspeth.engine.orchestrator.outcomes import accumulate_row_outcomes
        from elspeth.engine.orchestrator.types import ExecutionCounters

        stored_hash = "deadbeefdeadbeef"
        _db, _factory, processor = self._seed_parked_on_error_pending_sink(error_hash=stored_hash)
        ctx = make_context(landscape=_factory.plugin_audit_writer())

        results = processor.drain_scheduled_work(ctx)

        assert len(results) == 1
        assert results[0].authoritative_error_hash == stored_hash
        # And the accumulator prefers it end-to-end: the audit PendingOutcome
        # carries the stored hash, not a recompute of 'ResumedPendingSink'/''.
        counters = ExecutionCounters()
        pending: dict[str, list[Any]] = {"error_sink": []}
        accumulate_row_outcomes(results, counters, pending)
        pending_outcome = pending["error_sink"][0][1]
        assert pending_outcome is not None
        assert pending_outcome.error_hash == stored_hash

    def test_pending_sink_replay_fails_closed_on_missing_error_hash(self) -> None:
        """An ON_ERROR_ROUTED pending sink with no persisted error hash is
        audit corruption — replay must refuse rather than fabricate a hash."""
        _db, _factory, processor = self._seed_parked_on_error_pending_sink(error_hash=None)
        ctx = make_context(landscape=_factory.plugin_audit_writer())

        with pytest.raises(AuditIntegrityError, match="pending_error_hash"):
            processor.drain_scheduled_work(ctx)

    def test_evicted_worker_disposition_is_refused_mid_drain(self) -> None:
        """ADR-030 §G parity (elspeth-ba7b2cc25d): a registered worker whose
        run_workers row is evicted MID-PROCESSING cannot commit the claimed
        item's disposition — the drain raises RunWorkerEvictedError and the
        item stays LEASED (abandoned for the reaper) instead of landing in
        pending_sink under an evicted owner."""
        from sqlalchemy import select, update

        from elspeth.contracts.errors import RunWorkerEvictedError
        from elspeth.core.landscape.schema import run_workers_table, token_work_items_table

        db, factory = _make_factory()
        transform_node = NodeID("transform-1")
        factory.data_flow.register_node(
            run_id="test-run",
            plugin_name="resume-transform",
            node_type=NodeType.TRANSFORM,
            plugin_version="1.0",
            config={},
            node_id=str(transform_node),
            schema_config=_DYNAMIC_SCHEMA,
        )
        source_payload = make_row({"value": 42})
        row = factory.data_flow.create_row(
            run_id="test-run",
            source_node_id="source-0",
            row_index=0,
            source_row_index=0,
            ingest_sequence=0,
            data=source_payload.to_dict(),
        )
        token = factory.data_flow.create_token(row.row_id, token_id="token-ready")
        factory.scheduler.enqueue_ready(
            run_id="test-run",
            token_id=token.token_id,
            row_id=row.row_id,
            node_id=str(transform_node),
            step_index=1,
            ingest_sequence=0,
            row_payload_json=factory.scheduler.serialize_row_payload(source_payload),
            available_at=datetime.now(UTC),
        )
        transform = _make_mock_transform(node_id=str(transform_node), on_success="default")
        processor = _make_processor(
            factory,
            node_step_map={NodeID("source-0"): 0, transform_node: 1},
            node_to_next={NodeID("source-0"): transform_node, transform_node: None},
            node_to_plugin={transform_node: transform},
            scheduler=factory.scheduler,
            scheduler_lease_owner="worker-evictable",
        )
        ctx = make_context(landscape=factory.plugin_audit_writer())
        success_result = TransformResult.success(
            make_row({"value": 42, "resumed": True}),
            success_reason={"action": "resume_drain"},
        )

        def executor_side_effect(*, transform, token, ctx, attempt=0):
            # Evict the worker AFTER its claim succeeded but BEFORE the
            # drain's disposition write — the ticket's exact window.
            with db.engine.begin() as conn:
                conn.execute(
                    update(run_workers_table)
                    .where(run_workers_table.c.worker_id == "worker-evictable")
                    .values(status="evicted", evicted_at=datetime.now(UTC))
                )
            return (success_result, token, None)

        with (
            patch.object(processor._transform_executor, "execute_transform", side_effect=executor_side_effect),
            pytest.raises(RunWorkerEvictedError),
        ):
            processor.drain_scheduled_work(ctx)

        with db.connection() as conn:
            item_row = conn.execute(
                select(token_work_items_table.c.status, token_work_items_table.c.lease_owner).where(
                    token_work_items_table.c.run_id == "test-run"
                )
            ).one()
        assert item_row.status == "leased", "evicted worker's disposition must not commit"
        assert item_row.lease_owner == "worker-evictable"

    def test_sink_bound_scheduler_work_terminalizes_only_after_sink_callback(self) -> None:
        """Sink-bound work remains durable until sink outcome recording completes."""
        db, factory = _make_factory()
        transform_node = NodeID("transform-1")
        factory.data_flow.register_node(
            run_id="test-run",
            plugin_name="resume-transform",
            node_type=NodeType.TRANSFORM,
            plugin_version="1.0",
            config={},
            node_id=str(transform_node),
            schema_config=_DYNAMIC_SCHEMA,
        )
        source_payload = make_row({"value": 42})
        row = factory.data_flow.create_row(
            run_id="test-run",
            source_node_id="source-0",
            row_index=0,
            source_row_index=0,
            ingest_sequence=0,
            data=source_payload.to_dict(),
        )
        token = factory.data_flow.create_token(row.row_id, token_id="token-ready")
        factory.scheduler.enqueue_ready(
            run_id="test-run",
            token_id=token.token_id,
            row_id=row.row_id,
            node_id=str(transform_node),
            step_index=1,
            ingest_sequence=0,
            row_payload_json=factory.scheduler.serialize_row_payload(source_payload),
            available_at=datetime.now(UTC),
        )
        transform = _make_mock_transform(node_id=str(transform_node), on_success="default")
        processor = _make_processor(
            factory,
            node_step_map={NodeID("source-0"): 0, transform_node: 1},
            node_to_next={NodeID("source-0"): transform_node, transform_node: None},
            node_to_plugin={transform_node: transform},
            scheduler=factory.scheduler,
        )
        ctx = make_context(landscape=factory.plugin_audit_writer())
        success_result = TransformResult.success(
            make_row({"value": 42, "resumed": True}),
            success_reason={"action": "resume_drain"},
        )

        def executor_side_effect(*, transform, token, ctx, attempt=0):
            return (success_result, token, None)

        with patch.object(processor._transform_executor, "execute_transform", side_effect=executor_side_effect):
            results = processor.drain_scheduled_work(ctx)

        assert len(results) == 1
        processor.mark_sink_bound_scheduler_terminal("token-ready")

        from sqlalchemy import select

        from elspeth.core.landscape.schema import token_work_items_table

        with db.connection() as conn:
            status, row_payload_json = conn.execute(
                select(token_work_items_table.c.status, token_work_items_table.c.row_payload_json).where(
                    token_work_items_table.c.token_id == "token-ready"
                )
            ).one()
        assert status == "terminal"
        assert "resumed" not in row_payload_json
        assert "purged" in row_payload_json

    def test_single_sink_bound_scheduler_terminalization_requires_matching_pending_sink(self) -> None:
        """Single-token callback path must not silently ignore missing scheduler rows."""
        _db, factory = _make_factory()
        processor = _make_processor(factory, scheduler=factory.scheduler)

        with pytest.raises(AuditIntegrityError, match="terminalized 0 rows"):
            processor.mark_sink_bound_scheduler_terminal("missing-token")

    def test_pending_sink_resume_does_not_rerun_transform(self) -> None:
        """A crash after transform success resumes at sink handoff, not at the transform."""
        db, factory = _make_factory()
        transform_node = NodeID("transform-1")
        factory.data_flow.register_node(
            run_id="test-run",
            plugin_name="resume-transform",
            node_type=NodeType.TRANSFORM,
            plugin_version="1.0",
            config={},
            node_id=str(transform_node),
            schema_config=_DYNAMIC_SCHEMA,
        )
        source_payload = make_row({"value": 42})
        row = factory.data_flow.create_row(
            run_id="test-run",
            source_node_id="source-0",
            row_index=0,
            source_row_index=0,
            ingest_sequence=0,
            data=source_payload.to_dict(),
        )
        token = factory.data_flow.create_token(row.row_id, token_id="token-pending")
        factory.scheduler.enqueue_ready(
            run_id="test-run",
            token_id=token.token_id,
            row_id=row.row_id,
            node_id=str(transform_node),
            step_index=1,
            ingest_sequence=0,
            row_payload_json=factory.scheduler.serialize_row_payload(source_payload),
            available_at=datetime.now(UTC),
        )
        transform = _make_mock_transform(node_id=str(transform_node), on_success="default")
        first_processor = _make_processor(
            factory,
            node_step_map={NodeID("source-0"): 0, transform_node: 1},
            node_to_next={NodeID("source-0"): transform_node, transform_node: None},
            node_to_plugin={transform_node: transform},
            scheduler=factory.scheduler,
        )
        ctx = make_context(landscape=factory.plugin_audit_writer())
        final_token = TokenInfo(
            row_id=row.row_id,
            token_id=token.token_id,
            row_data=make_row({"value": 42, "resumed": True}),
        )
        sink_bound_result = RowResult(
            token=final_token,
            final_data=final_token.row_data,
            outcome=TerminalOutcome.SUCCESS,
            path=TerminalPath.DEFAULT_FLOW,
            sink_name="default",
        )

        with patch.object(first_processor, "_process_single_token", return_value=(sink_bound_result, [])):
            first_processor.drain_scheduled_work(ctx)

        resumed_processor = _make_processor(
            factory,
            node_step_map={NodeID("source-0"): 0, transform_node: 1},
            node_to_next={NodeID("source-0"): transform_node, transform_node: None},
            node_to_plugin={transform_node: transform},
            scheduler=factory.scheduler,
            scheduler_lease_owner="resume-worker",
        )
        with patch.object(resumed_processor._transform_executor, "execute_transform", side_effect=AssertionError("transform replayed")):
            results = resumed_processor.drain_scheduled_work(ctx)

        assert len(results) == 1
        assert results[0].token.token_id == "token-pending"
        assert results[0].token.row_data.to_dict() == {"value": 42, "resumed": True}
        _assert_outcome_pair(results[0], TerminalOutcome.SUCCESS, TerminalPath.DEFAULT_FLOW)

        from sqlalchemy import select

        from elspeth.core.landscape.schema import token_work_items_table

        with db.connection() as conn:
            status, lease_owner = conn.execute(
                select(token_work_items_table.c.status, token_work_items_table.c.lease_owner).where(
                    token_work_items_table.c.token_id == "token-pending"
                )
            ).one()
        assert status == "leased"
        assert lease_owner == "resume-worker"

    def test_pending_sink_resume_repairs_already_outcomed_row_without_reemitting_sink(self) -> None:
        """A terminal token outcome is the resume witness; do not emit the sink externally again."""
        db, factory = _make_factory()
        transform_node = NodeID("transform-1")
        factory.data_flow.register_node(
            run_id="test-run",
            plugin_name="resume-transform",
            node_type=NodeType.TRANSFORM,
            plugin_version="1.0",
            config={},
            node_id=str(transform_node),
            schema_config=_DYNAMIC_SCHEMA,
        )
        source_payload = make_row({"value": 42})
        row = factory.data_flow.create_row(
            run_id="test-run",
            source_node_id="source-0",
            row_index=0,
            source_row_index=0,
            ingest_sequence=0,
            data=source_payload.to_dict(),
        )
        token = factory.data_flow.create_token(row.row_id, token_id="token-outcomed-pending")
        factory.scheduler.enqueue_ready(
            run_id="test-run",
            token_id=token.token_id,
            row_id=row.row_id,
            node_id=str(transform_node),
            step_index=1,
            ingest_sequence=0,
            row_payload_json=factory.scheduler.serialize_row_payload(source_payload),
            available_at=datetime.now(UTC),
        )
        transform = _make_mock_transform(node_id=str(transform_node), on_success="default")
        crashed_processor = _make_processor(
            factory,
            node_step_map={NodeID("source-0"): 0, transform_node: 1},
            node_to_next={NodeID("source-0"): transform_node, transform_node: None},
            node_to_plugin={transform_node: transform},
            scheduler=factory.scheduler,
            scheduler_lease_owner="crashed-worker",
        )
        ctx = make_context(landscape=factory.plugin_audit_writer())
        final_token = TokenInfo(
            row_id=row.row_id,
            token_id=token.token_id,
            row_data=make_row({"value": 42, "resumed": True}),
        )
        sink_bound_result = RowResult(
            token=final_token,
            final_data=final_token.row_data,
            outcome=TerminalOutcome.SUCCESS,
            path=TerminalPath.DEFAULT_FLOW,
            sink_name="default",
        )
        with patch.object(crashed_processor, "_process_single_token", return_value=(sink_bound_result, [])):
            crashed_processor.drain_scheduled_work(ctx)
        factory.data_flow.record_token_outcome(
            ref=TokenRef(token_id=token.token_id, run_id="test-run"),
            outcome=TerminalOutcome.SUCCESS,
            path=TerminalPath.DEFAULT_FLOW,
            sink_name="default",
        )

        resumed_processor = _make_processor(
            factory,
            node_step_map={NodeID("source-0"): 0, transform_node: 1},
            node_to_next={NodeID("source-0"): transform_node, transform_node: None},
            node_to_plugin={transform_node: transform},
            scheduler=factory.scheduler,
            scheduler_lease_owner="resume-worker",
        )
        with patch.object(resumed_processor._transform_executor, "execute_transform", side_effect=AssertionError("transform replayed")):
            results = resumed_processor.drain_scheduled_work(ctx)

        assert results == []
        from sqlalchemy import select

        from elspeth.core.landscape.schema import scheduler_events_table, token_work_items_table

        with db.connection() as conn:
            status, lease_owner = conn.execute(
                select(token_work_items_table.c.status, token_work_items_table.c.lease_owner).where(
                    token_work_items_table.c.token_id == "token-outcomed-pending"
                )
            ).one()
            terminal_event_owner = conn.execute(
                select(scheduler_events_table.c.caller_owner)
                .where(scheduler_events_table.c.token_id == "token-outcomed-pending")
                .where(scheduler_events_table.c.event_type == "mark_pending_sink_terminal")
            ).scalar_one()
        assert status == "terminal"
        assert lease_owner is None
        assert terminal_event_owner == "resume-worker"

    def test_resume_drains_all_pending_sink_rows_in_single_call(self) -> None:
        """Multiple pre-existing PENDING_SINK rows must all drain in a single resume call.

        Regression for elspeth-5c5e88b071 (G3): the ``created_pending_sink_this_drain``
        gate previously short-circuited the ``claim_pending_sink`` fallback after
        the first pending-sink emission. When a prior crashed worker left N
        sink-bound rows durable, only one was recovered per ``drain_scheduled_work``
        call, leaving the rest stranded and tripping
        ``OrchestrationInvariantError`` at run completion.
        """
        db, factory = _make_factory()
        transform_node = NodeID("transform-1")
        factory.data_flow.register_node(
            run_id="test-run",
            plugin_name="resume-transform",
            node_type=NodeType.TRANSFORM,
            plugin_version="1.0",
            config={},
            node_id=str(transform_node),
            schema_config=_DYNAMIC_SCHEMA,
        )

        pending_token_ids: list[str] = []
        for idx in range(3):
            source_payload = make_row({"value": idx})
            row = factory.data_flow.create_row(
                run_id="test-run",
                source_node_id="source-0",
                row_index=idx,
                source_row_index=idx,
                ingest_sequence=idx,
                data=source_payload.to_dict(),
            )
            token_id = f"token-pending-{idx}"
            pending_token_ids.append(token_id)
            token = factory.data_flow.create_token(row.row_id, token_id=token_id)
            factory.scheduler.enqueue_ready(
                run_id="test-run",
                token_id=token.token_id,
                row_id=row.row_id,
                node_id=str(transform_node),
                step_index=1,
                ingest_sequence=idx,
                row_payload_json=factory.scheduler.serialize_row_payload(source_payload),
                available_at=datetime.now(UTC),
            )

        # Stage 1: simulate the crashed worker that drove the transform to
        # success on every token. Each token ends in PENDING_SINK because
        # sink durability never completed.
        transform = _make_mock_transform(node_id=str(transform_node), on_success="default")
        crashed_processor = _make_processor(
            factory,
            node_step_map={NodeID("source-0"): 0, transform_node: 1},
            node_to_next={NodeID("source-0"): transform_node, transform_node: None},
            node_to_plugin={transform_node: transform},
            scheduler=factory.scheduler,
            scheduler_lease_owner="crashed-worker",
        )
        ctx = make_context(landscape=factory.plugin_audit_writer())

        def crashed_executor(*, transform, token, ctx, attempt=0):
            return (
                TransformResult.success(
                    make_row({**token.row_data.to_dict(), "resumed": True}),
                    success_reason={"action": "crashed_worker_transform"},
                ),
                token,
                None,
            )

        with patch.object(crashed_processor._transform_executor, "execute_transform", side_effect=crashed_executor):
            crashed_processor.drain_scheduled_work(ctx)

        from sqlalchemy import select

        from elspeth.core.landscape.schema import token_work_items_table

        with db.connection() as conn:
            statuses = (
                conn.execute(select(token_work_items_table.c.status).where(token_work_items_table.c.run_id == "test-run")).scalars().all()
            )
        assert sorted(statuses) == ["pending_sink"] * 3, (
            f"Expected three PENDING_SINK rows after the crashed worker drained transform work; got {sorted(statuses)}."
        )

        # Stage 2: fresh resume worker. The bug: only one PENDING_SINK row
        # emits a RowResult; the other two stay PENDING_SINK because the gate
        # blocks the recovery branch after the first claim.
        resume_processor = _make_processor(
            factory,
            node_step_map={NodeID("source-0"): 0, transform_node: 1},
            node_to_next={NodeID("source-0"): transform_node, transform_node: None},
            node_to_plugin={transform_node: transform},
            scheduler=factory.scheduler,
            scheduler_lease_owner="resume-worker",
        )
        with patch.object(
            resume_processor._transform_executor,
            "execute_transform",
            side_effect=AssertionError("transform replayed during pending-sink recovery"),
        ):
            results = resume_processor.drain_scheduled_work(ctx)

        recovered_token_ids = sorted(result.token.token_id for result in results)
        assert recovered_token_ids == sorted(pending_token_ids), (
            f"All pre-existing PENDING_SINK rows must be re-emitted in a single drain call; got {recovered_token_ids}."
        )

        with db.connection() as conn:
            statuses_after = (
                conn.execute(select(token_work_items_table.c.status).where(token_work_items_table.c.run_id == "test-run")).scalars().all()
            )
        # Every row is now LEASED by the resume worker awaiting the sink
        # callback; none remain in PENDING_SINK status.
        assert sorted(statuses_after) == ["leased"] * 3

    def test_drain_emits_pending_sink_recovery_without_duplicating_fresh_work(self) -> None:
        """Mixed pre-existing PENDING_SINK + new READY work must not double-emit.

        Discriminating regression for elspeth-5c5e88b071 (G3): the naive fix —
        deleting the gate and putting ``claim_pending_sink`` inside the main
        loop — would re-claim PENDING_SINK rows that the current drain just
        produced via ``mark_pending_sink`` and emit a duplicate RowResult via
        ``_row_result_from_pending_sink`` for work already represented by the
        original sink-bound RowResult. Pre-existing pending-sinks must be
        drained up front; newly-marked pending-sinks must only be terminalized
        by the sink callback, never re-claimed inside the same drain.
        """
        db, factory = _make_factory()
        transform_node = NodeID("transform-1")
        factory.data_flow.register_node(
            run_id="test-run",
            plugin_name="resume-transform",
            node_type=NodeType.TRANSFORM,
            plugin_version="1.0",
            config={},
            node_id=str(transform_node),
            schema_config=_DYNAMIC_SCHEMA,
        )

        # Two pre-existing pending-sink rows from a prior crashed worker.
        pre_existing_token_ids: list[str] = []
        for idx in range(2):
            source_payload = make_row({"value": idx})
            row = factory.data_flow.create_row(
                run_id="test-run",
                source_node_id="source-0",
                row_index=idx,
                source_row_index=idx,
                ingest_sequence=idx,
                data=source_payload.to_dict(),
            )
            token_id = f"token-pre-{idx}"
            pre_existing_token_ids.append(token_id)
            token = factory.data_flow.create_token(row.row_id, token_id=token_id)
            factory.scheduler.enqueue_ready(
                run_id="test-run",
                token_id=token.token_id,
                row_id=row.row_id,
                node_id=str(transform_node),
                step_index=1,
                ingest_sequence=idx,
                row_payload_json=factory.scheduler.serialize_row_payload(source_payload),
                available_at=datetime.now(UTC),
            )

        transform = _make_mock_transform(node_id=str(transform_node), on_success="default")
        ctx = make_context(landscape=factory.plugin_audit_writer())

        # Stage 1: crashed worker pushes the pre-existing rows to PENDING_SINK.
        crashed_processor = _make_processor(
            factory,
            node_step_map={NodeID("source-0"): 0, transform_node: 1},
            node_to_next={NodeID("source-0"): transform_node, transform_node: None},
            node_to_plugin={transform_node: transform},
            scheduler=factory.scheduler,
            scheduler_lease_owner="crashed-worker",
        )

        def crashed_executor(*, transform, token, ctx, attempt=0):
            return (
                TransformResult.success(
                    make_row({**token.row_data.to_dict(), "resumed": True}),
                    success_reason={"action": "crashed_worker_transform"},
                ),
                token,
                None,
            )

        with patch.object(crashed_processor._transform_executor, "execute_transform", side_effect=crashed_executor):
            crashed_processor.drain_scheduled_work(ctx)

        # Stage 2: enqueue a brand new READY token that the resume worker
        # will process to completion (ending in PENDING_SINK durably and
        # emitting one sink-bound RowResult).
        fresh_payload = make_row({"value": 99})
        fresh_row = factory.data_flow.create_row(
            run_id="test-run",
            source_node_id="source-0",
            row_index=99,
            source_row_index=99,
            ingest_sequence=99,
            data=fresh_payload.to_dict(),
        )
        fresh_token_id = "token-fresh"
        fresh_token = factory.data_flow.create_token(fresh_row.row_id, token_id=fresh_token_id)
        factory.scheduler.enqueue_ready(
            run_id="test-run",
            token_id=fresh_token.token_id,
            row_id=fresh_row.row_id,
            node_id=str(transform_node),
            step_index=1,
            ingest_sequence=99,
            row_payload_json=factory.scheduler.serialize_row_payload(fresh_payload),
            available_at=datetime.now(UTC),
        )

        # Stage 3: resume worker drains everything in one call. Pre-existing
        # rows recover via _row_result_from_pending_sink; the fresh row
        # processes through the transform and emits the sink-bound RowResult
        # directly. Neither path may emit the same token twice.
        resume_processor = _make_processor(
            factory,
            node_step_map={NodeID("source-0"): 0, transform_node: 1},
            node_to_next={NodeID("source-0"): transform_node, transform_node: None},
            node_to_plugin={transform_node: transform},
            scheduler=factory.scheduler,
            scheduler_lease_owner="resume-worker",
        )

        def resume_executor(*, transform, token, ctx, attempt=0):
            # Pre-existing tokens must NOT re-enter the transform (their
            # transform work is durable from the crashed worker).
            assert token.token_id == fresh_token_id, f"transform replayed for {token.token_id!r} during recovery drain"
            return (
                TransformResult.success(
                    make_row({**token.row_data.to_dict(), "fresh_processed": True}),
                    success_reason={"action": "resume_worker_transform"},
                ),
                token,
                None,
            )

        with patch.object(resume_processor._transform_executor, "execute_transform", side_effect=resume_executor):
            results = resume_processor.drain_scheduled_work(ctx)

        emitted_token_ids = [result.token.token_id for result in results]
        assert len(emitted_token_ids) == len(set(emitted_token_ids)), (
            f"Duplicate RowResults emitted for the same token_id: {emitted_token_ids}. "
            "claim_pending_sink must not re-claim rows produced by mark_pending_sink "
            "within the same drain call."
        )
        assert sorted(emitted_token_ids) == sorted([*pre_existing_token_ids, fresh_token_id])

        from sqlalchemy import select

        from elspeth.core.landscape.schema import token_work_items_table

        with db.connection() as conn:
            statuses_after = (
                conn.execute(select(token_work_items_table.c.status).where(token_work_items_table.c.run_id == "test-run")).scalars().all()
            )
        # All three rows now sit in non-terminal states awaiting the sink
        # callback. The pre-existing two are LEASED (claim_pending_sink
        # transition); the fresh one is PENDING_SINK (mark_pending_sink
        # transition without a subsequent claim).
        assert sorted(statuses_after) == ["leased", "leased", "pending_sink"]

    def test_drain_scheduler_transitions_use_injected_clock(self) -> None:
        """Durable scheduler state transitions must be deterministic under MockClock."""
        db, factory = _make_factory()
        clock = MockClock(start=1_700_000_000.0)
        transform_node = NodeID("transform-1")
        factory.data_flow.register_node(
            run_id="test-run",
            plugin_name="resume-transform",
            node_type=NodeType.TRANSFORM,
            plugin_version="1.0",
            config={},
            node_id=str(transform_node),
            schema_config=_DYNAMIC_SCHEMA,
        )
        source_payload = make_row({"value": 42})
        row = factory.data_flow.create_row(
            run_id="test-run",
            source_node_id="source-0",
            row_index=0,
            source_row_index=0,
            ingest_sequence=0,
            data=source_payload.to_dict(),
        )
        token = factory.data_flow.create_token(row.row_id, token_id="token-clock")
        factory.scheduler.enqueue_ready(
            run_id="test-run",
            token_id=token.token_id,
            row_id=row.row_id,
            node_id=str(transform_node),
            step_index=1,
            ingest_sequence=0,
            row_payload_json=factory.scheduler.serialize_row_payload(source_payload),
            available_at=clock.now_utc(),
        )

        transform = _make_mock_transform(node_id=str(transform_node), on_success="default")
        processor = _make_processor(
            factory,
            node_step_map={NodeID("source-0"): 0, transform_node: 1},
            node_to_next={NodeID("source-0"): transform_node, transform_node: None},
            node_to_plugin={transform_node: transform},
            scheduler=factory.scheduler,
            clock=clock,
        )
        ctx = make_context(landscape=factory.plugin_audit_writer())
        success_result = TransformResult.success(
            make_row({"value": 42, "resumed": True}),
            success_reason={"action": "resume_drain"},
        )

        def executor_side_effect(*, transform, token, ctx, attempt=0):
            clock.advance(5.0)
            return (success_result, token, None)

        with patch.object(processor._transform_executor, "execute_transform", side_effect=executor_side_effect):
            processor.drain_scheduled_work(ctx)

        from sqlalchemy import select

        from elspeth.core.landscape.schema import token_work_items_table

        with db.connection() as conn:
            status, updated_at = conn.execute(
                select(token_work_items_table.c.status, token_work_items_table.c.updated_at).where(
                    token_work_items_table.c.token_id == "token-clock"
                )
            ).one()
        assert status == "pending_sink"
        assert updated_at.replace(tzinfo=UTC) == clock.now_utc()

    def test_recovers_expired_lease_then_drains_without_source_replay(self) -> None:
        """Expired LEASED scheduler work is recovered and advanced by a fresh processor."""
        db, factory = _make_factory()
        transform_node = NodeID("transform-1")
        factory.data_flow.register_node(
            run_id="test-run",
            plugin_name="resume-transform",
            node_type=NodeType.TRANSFORM,
            plugin_version="1.0",
            config={},
            node_id=str(transform_node),
            schema_config=_DYNAMIC_SCHEMA,
        )
        source_payload = make_row({"value": 43})
        row = factory.data_flow.create_row(
            run_id="test-run",
            source_node_id="source-0",
            row_index=0,
            source_row_index=0,
            ingest_sequence=0,
            data=source_payload.to_dict(),
        )
        token = factory.data_flow.create_token(row.row_id, token_id="token-expired")
        past = datetime.now(UTC) - timedelta(hours=1)
        factory.scheduler.enqueue_ready(
            run_id="test-run",
            token_id=token.token_id,
            row_id=row.row_id,
            node_id=str(transform_node),
            step_index=1,
            ingest_sequence=0,
            row_payload_json=factory.scheduler.serialize_row_payload(source_payload),
            available_at=past,
        )
        _register_test_worker(factory, "dead-worker")
        claimed = factory.scheduler.claim_ready(
            run_id="test-run",
            lease_owner="dead-worker",
            lease_seconds=1,
            now=past,
        )
        assert claimed is not None

        transform = _make_mock_transform(node_id=str(transform_node), on_success="default")
        processor = _make_processor(
            factory,
            node_step_map={NodeID("source-0"): 0, transform_node: 1},
            node_to_next={NodeID("source-0"): transform_node, transform_node: None},
            node_to_plugin={transform_node: transform},
            scheduler=factory.scheduler,
        )
        ctx = make_context(landscape=factory.plugin_audit_writer())
        success_result = TransformResult.success(
            make_row({"value": 43, "resumed": True}),
            success_reason={"action": "expired_lease_recovered"},
        )

        def executor_side_effect(*, transform, token, ctx, attempt=0):
            assert token.token_id == "token-expired"
            assert token.row_data.to_dict() == {"value": 43}
            assert attempt == 1
            return (success_result, token, None)

        with patch.object(processor._transform_executor, "execute_transform", side_effect=executor_side_effect):
            results = processor.drain_scheduled_work(ctx)

        assert len(results) == 1
        assert results[0].token.token_id == "token-expired"
        from sqlalchemy import select

        from elspeth.core.landscape.schema import token_work_items_table

        with db.connection() as conn:
            statuses = (
                conn.execute(select(token_work_items_table.c.status).where(token_work_items_table.c.run_id == "test-run")).scalars().all()
            )
        assert statuses == ["pending_sink"]

    def test_drain_raises_when_resultless_work_has_no_scheduler_release_key(self) -> None:
        """Durable work cannot be BLOCKED unless resume has a release key for it."""
        _db, factory = _make_factory()
        transform_node = NodeID("transform-1")
        factory.data_flow.register_node(
            run_id="test-run",
            plugin_name="resume-transform",
            node_type=NodeType.TRANSFORM,
            plugin_version="1.0",
            config={},
            node_id=str(transform_node),
            schema_config=_DYNAMIC_SCHEMA,
        )
        source_payload = make_row({"value": 99})
        row = factory.data_flow.create_row(
            run_id="test-run",
            source_node_id="source-0",
            row_index=0,
            source_row_index=0,
            ingest_sequence=0,
            data=source_payload.to_dict(),
        )
        token = factory.data_flow.create_token(row.row_id, token_id="token-stranded")
        work_item = factory.scheduler.enqueue_ready(
            run_id="test-run",
            token_id=token.token_id,
            row_id=row.row_id,
            node_id=str(transform_node),
            step_index=1,
            ingest_sequence=0,
            row_payload_json=factory.scheduler.serialize_row_payload(source_payload),
            available_at=datetime.now(UTC),
        )

        transform = _make_mock_transform(node_id=str(transform_node), on_success="default")
        processor = _make_processor(
            factory,
            node_step_map={NodeID("source-0"): 0, transform_node: 1},
            node_to_next={NodeID("source-0"): transform_node, transform_node: None},
            node_to_plugin={transform_node: transform},
            scheduler=factory.scheduler,
        )
        ctx = make_context(landscape=factory.plugin_audit_writer())

        with (
            patch.object(processor, "_process_single_token", return_value=(None, [])),
            pytest.raises(
                OrchestrationInvariantError,
                match=(
                    rf"Work item '{work_item.work_item_id}'.*token='token-stranded'.*"
                    rf"node={str(transform_node)!r}.*no queue or barrier key"
                ),
            ),
        ):
            processor.drain_scheduled_work(ctx)

    def test_resume_restores_branch_lineage_for_direct_sink_work(self) -> None:
        """Direct branch sink routing must survive durable scheduler resume."""
        _db, factory = _make_factory()
        source_payload = make_row({"value": 44})
        row = factory.data_flow.create_row(
            run_id="test-run",
            source_node_id="source-0",
            row_index=0,
            source_row_index=0,
            ingest_sequence=0,
            data=source_payload.to_dict(),
        )
        token = TokenInfo(
            row_id=row.row_id,
            token_id="token-direct-branch",
            row_data=source_payload,
            branch_name="direct",
            fork_group_id="fork-1",
            join_group_id="join-1",
            expand_group_id="expand-1",
        )
        factory.data_flow.create_token(row.row_id, token_id=token.token_id)
        factory.scheduler.enqueue_ready(
            run_id="test-run",
            token_id=token.token_id,
            row_id=row.row_id,
            node_id=None,
            step_index=99,
            ingest_sequence=0,
            row_payload_json=factory.scheduler.serialize_row_payload(source_payload),
            available_at=datetime.now(UTC),
            on_success_sink="source_sink",
            branch_name=token.branch_name,
            fork_group_id=token.fork_group_id,
            join_group_id=token.join_group_id,
            expand_group_id=token.expand_group_id,
        )

        processor = _make_processor(
            factory,
            source_on_success="source_sink",
            branch_to_sink={BranchName("direct"): "branch_sink"},
            scheduler=factory.scheduler,
        )
        ctx = make_context(landscape=factory.plugin_audit_writer())

        results = processor.drain_scheduled_work(ctx)

        assert len(results) == 1
        assert results[0].sink_name == "branch_sink"
        assert results[0].token.branch_name == "direct"
        assert results[0].token.fork_group_id == "fork-1"
        assert results[0].token.join_group_id == "join-1"
        assert results[0].token.expand_group_id == "expand-1"

    def test_durable_scheduler_failure_result_marks_work_failed(self) -> None:
        """Durable failure outcomes remain distinguishable from successful terminal work."""
        db, factory = _make_factory()
        transform_node = NodeID("transform-1")
        factory.data_flow.register_node(
            run_id="test-run",
            plugin_name="resume-transform",
            node_type=NodeType.TRANSFORM,
            plugin_version="1.0",
            config={},
            node_id=str(transform_node),
            schema_config=_DYNAMIC_SCHEMA,
        )
        source_payload = make_row({"value": 45})
        row = factory.data_flow.create_row(
            run_id="test-run",
            source_node_id="source-0",
            row_index=0,
            source_row_index=0,
            ingest_sequence=0,
            data=source_payload.to_dict(),
        )
        token = factory.data_flow.create_token(row.row_id, token_id="token-failed")
        factory.scheduler.enqueue_ready(
            run_id="test-run",
            token_id=token.token_id,
            row_id=row.row_id,
            node_id=str(transform_node),
            step_index=1,
            ingest_sequence=0,
            row_payload_json=factory.scheduler.serialize_row_payload(source_payload),
            available_at=datetime.now(UTC),
        )
        transform = _make_mock_transform(node_id=str(transform_node), on_success="default")
        processor = _make_processor(
            factory,
            node_step_map={NodeID("source-0"): 0, transform_node: 1},
            node_to_next={NodeID("source-0"): transform_node, transform_node: None},
            node_to_plugin={transform_node: transform},
            scheduler=factory.scheduler,
        )
        failed_result = RowResult(
            token=TokenInfo(row_id=row.row_id, token_id=token.token_id, row_data=source_payload),
            final_data=source_payload,
            outcome=TerminalOutcome.FAILURE,
            path=TerminalPath.UNROUTED,
        )
        ctx = make_context(landscape=factory.plugin_audit_writer())

        with patch.object(processor, "_process_single_token", return_value=(failed_result, [])):
            results = processor.drain_scheduled_work(ctx)

        assert results == [failed_result]
        from sqlalchemy import select

        from elspeth.core.landscape.schema import token_work_items_table

        with db.connection() as conn:
            status, run_id, work_item_id = conn.execute(
                select(
                    token_work_items_table.c.status,
                    token_work_items_table.c.run_id,
                    token_work_items_table.c.work_item_id,
                ).where(token_work_items_table.c.token_id == "token-failed")
            ).one()
        assert status == "failed"
        assert run_id == "test-run"
        assert work_item_id

    def test_durable_scheduler_on_error_routed_result_marks_pending_sink(self) -> None:
        """ON_ERROR_ROUTED failures are sink-bound until error-sink durability."""
        db, factory = _make_factory()
        transform_node = NodeID("transform-1")
        factory.data_flow.register_node(
            run_id="test-run",
            plugin_name="resume-transform",
            node_type=NodeType.TRANSFORM,
            plugin_version="1.0",
            config={},
            node_id=str(transform_node),
            schema_config=_DYNAMIC_SCHEMA,
        )
        source_payload = make_row({"value": 45})
        row = factory.data_flow.create_row(
            run_id="test-run",
            source_node_id="source-0",
            row_index=0,
            source_row_index=0,
            ingest_sequence=0,
            data=source_payload.to_dict(),
        )
        token = factory.data_flow.create_token(row.row_id, token_id="token-on-error-routed")
        factory.scheduler.enqueue_ready(
            run_id="test-run",
            token_id=token.token_id,
            row_id=row.row_id,
            node_id=str(transform_node),
            step_index=1,
            ingest_sequence=0,
            row_payload_json=factory.scheduler.serialize_row_payload(source_payload),
            available_at=datetime.now(UTC),
        )
        transform = _make_mock_transform(node_id=str(transform_node), on_success="default", on_error="errors")
        processor = _make_processor(
            factory,
            node_step_map={NodeID("source-0"): 0, transform_node: 1},
            node_to_next={NodeID("source-0"): transform_node, transform_node: None},
            node_to_plugin={transform_node: transform},
            scheduler=factory.scheduler,
        )
        routed_result = RowResult(
            token=TokenInfo(row_id=row.row_id, token_id=token.token_id, row_data=source_payload),
            final_data=source_payload,
            outcome=TerminalOutcome.FAILURE,
            path=TerminalPath.ON_ERROR_ROUTED,
            sink_name="errors",
            error=FailureInfo(exception_type="TransformError", message="bad row"),
        )
        ctx = make_context(landscape=factory.plugin_audit_writer())

        with patch.object(processor, "_process_single_token", return_value=(routed_result, [])):
            results = processor.drain_scheduled_work(ctx)

        assert len(results) == 1
        assert results[0].outcome == TerminalOutcome.FAILURE
        assert results[0].path == TerminalPath.ON_ERROR_ROUTED
        assert results[0].sink_name == "errors"
        assert results[0].scheduler_pending_sink is True
        from sqlalchemy import select

        from elspeth.core.landscape.schema import token_work_items_table

        with db.connection() as conn:
            row_state = conn.execute(
                select(
                    token_work_items_table.c.status,
                    token_work_items_table.c.pending_sink_name,
                    token_work_items_table.c.pending_outcome,
                    token_work_items_table.c.pending_path,
                    token_work_items_table.c.row_payload_json,
                ).where(token_work_items_table.c.token_id == "token-on-error-routed")
            ).one()
        assert row_state.status == "pending_sink"
        assert row_state.pending_sink_name == "errors"
        assert row_state.pending_outcome == TerminalOutcome.FAILURE.value
        assert row_state.pending_path == TerminalPath.ON_ERROR_ROUTED.value
        assert factory.scheduler.deserialize_row_payload(row_state.row_payload_json).to_dict() == source_payload.to_dict()

    def test_durable_scheduler_tuple_failure_for_claimed_token_marks_work_failed(self) -> None:
        """Tuple results must classify the claimed token, not fall through to terminal."""
        db, factory = _make_factory()
        transform_node = NodeID("transform-1")
        factory.data_flow.register_node(
            run_id="test-run",
            plugin_name="resume-transform",
            node_type=NodeType.TRANSFORM,
            plugin_version="1.0",
            config={},
            node_id=str(transform_node),
            schema_config=_DYNAMIC_SCHEMA,
        )
        source_payload = make_row({"value": 45})
        row = factory.data_flow.create_row(
            run_id="test-run",
            source_node_id="source-0",
            row_index=0,
            source_row_index=0,
            ingest_sequence=0,
            data=source_payload.to_dict(),
        )
        token = factory.data_flow.create_token(row.row_id, token_id="token-failed-tuple")
        factory.scheduler.enqueue_ready(
            run_id="test-run",
            token_id=token.token_id,
            row_id=row.row_id,
            node_id=str(transform_node),
            step_index=1,
            ingest_sequence=0,
            row_payload_json=factory.scheduler.serialize_row_payload(source_payload),
            available_at=datetime.now(UTC),
        )
        transform = _make_mock_transform(node_id=str(transform_node), on_success="default")
        processor = _make_processor(
            factory,
            node_step_map={NodeID("source-0"): 0, transform_node: 1},
            node_to_next={NodeID("source-0"): transform_node, transform_node: None},
            node_to_plugin={transform_node: transform},
            scheduler=factory.scheduler,
        )
        failed_result = RowResult(
            token=TokenInfo(row_id=row.row_id, token_id=token.token_id, row_data=source_payload),
            final_data=source_payload,
            outcome=TerminalOutcome.FAILURE,
            path=TerminalPath.UNROUTED,
        )
        sibling_success = RowResult(
            token=TokenInfo(row_id=row.row_id, token_id="sibling-token", row_data=source_payload),
            final_data=source_payload,
            outcome=TerminalOutcome.SUCCESS,
            path=TerminalPath.DEFAULT_FLOW,
            sink_name="default",
        )
        ctx = make_context(landscape=factory.plugin_audit_writer())

        with patch.object(processor, "_process_single_token", return_value=((sibling_success, failed_result), [])):
            results = processor.drain_scheduled_work(ctx)

        assert results == [sibling_success, failed_result]
        from sqlalchemy import select

        from elspeth.core.landscape.schema import token_work_items_table

        with db.connection() as conn:
            status = conn.execute(
                select(token_work_items_table.c.status).where(token_work_items_table.c.token_id == "token-failed-tuple")
            ).scalar_one()
        assert status == "failed"

    def test_durable_scheduler_tuple_failure_for_sibling_keeps_claimed_work_terminal(self) -> None:
        """Sibling failures returned with a claimed token must not poison the claimed work item."""
        db, factory = _make_factory()
        transform_node = NodeID("transform-1")
        factory.data_flow.register_node(
            run_id="test-run",
            plugin_name="resume-transform",
            node_type=NodeType.TRANSFORM,
            plugin_version="1.0",
            config={},
            node_id=str(transform_node),
            schema_config=_DYNAMIC_SCHEMA,
        )
        source_payload = make_row({"value": 45})
        row = factory.data_flow.create_row(
            run_id="test-run",
            source_node_id="source-0",
            row_index=0,
            source_row_index=0,
            ingest_sequence=0,
            data=source_payload.to_dict(),
        )
        token = factory.data_flow.create_token(row.row_id, token_id="token-claimed-success")
        factory.scheduler.enqueue_ready(
            run_id="test-run",
            token_id=token.token_id,
            row_id=row.row_id,
            node_id=str(transform_node),
            step_index=1,
            ingest_sequence=0,
            row_payload_json=factory.scheduler.serialize_row_payload(source_payload),
            available_at=datetime.now(UTC),
        )
        transform = _make_mock_transform(node_id=str(transform_node), on_success="default")
        processor = _make_processor(
            factory,
            node_step_map={NodeID("source-0"): 0, transform_node: 1},
            node_to_next={NodeID("source-0"): transform_node, transform_node: None},
            node_to_plugin={transform_node: transform},
            scheduler=factory.scheduler,
        )
        claimed_success = RowResult(
            token=TokenInfo(row_id=row.row_id, token_id=token.token_id, row_data=source_payload),
            final_data=source_payload,
            outcome=TerminalOutcome.SUCCESS,
            path=TerminalPath.DEFAULT_FLOW,
            sink_name="default",
        )
        sibling_failure = RowResult(
            token=TokenInfo(row_id=row.row_id, token_id="sibling-token", row_data=source_payload),
            final_data=source_payload,
            outcome=TerminalOutcome.FAILURE,
            path=TerminalPath.UNROUTED,
        )
        ctx = make_context(landscape=factory.plugin_audit_writer())

        with patch.object(processor, "_process_single_token", return_value=((sibling_failure, claimed_success), [])):
            results = processor.drain_scheduled_work(ctx)

        assert len(results) == 2
        assert results[0] == sibling_failure
        assert results[1].token.token_id == claimed_success.token.token_id
        assert results[1].outcome == claimed_success.outcome
        assert results[1].path == claimed_success.path
        assert results[1].sink_name == claimed_success.sink_name
        assert results[1].scheduler_pending_sink is True
        from sqlalchemy import select

        from elspeth.core.landscape.schema import token_work_items_table

        with db.connection() as conn:
            status = conn.execute(
                select(token_work_items_table.c.status).where(token_work_items_table.c.token_id == "token-claimed-success")
            ).scalar_one()
        assert status == "pending_sink"

    def test_scheduler_failure_write_preserves_processing_exception_context(self) -> None:
        """If marking scheduler work failed also fails, diagnostics keep both causes."""
        _db, factory = _make_factory()
        transform_node = NodeID("transform-1")
        factory.data_flow.register_node(
            run_id="test-run",
            plugin_name="resume-transform",
            node_type=NodeType.TRANSFORM,
            plugin_version="1.0",
            config={},
            node_id=str(transform_node),
            schema_config=_DYNAMIC_SCHEMA,
        )
        source_payload = make_row({"value": 46})
        row = factory.data_flow.create_row(
            run_id="test-run",
            source_node_id="source-0",
            row_index=0,
            source_row_index=0,
            ingest_sequence=0,
            data=source_payload.to_dict(),
        )
        token = factory.data_flow.create_token(row.row_id, token_id="token-crash")
        factory.scheduler.enqueue_ready(
            run_id="test-run",
            token_id=token.token_id,
            row_id=row.row_id,
            node_id=str(transform_node),
            step_index=1,
            ingest_sequence=0,
            row_payload_json=factory.scheduler.serialize_row_payload(source_payload),
            available_at=datetime.now(UTC),
        )
        transform = _make_mock_transform(node_id=str(transform_node), on_success="default")
        processor = _make_processor(
            factory,
            node_step_map={NodeID("source-0"): 0, transform_node: 1},
            node_to_next={NodeID("source-0"): transform_node, transform_node: None},
            node_to_plugin={transform_node: transform},
            scheduler=factory.scheduler,
        )
        ctx = make_context(landscape=factory.plugin_audit_writer())

        with (
            patch.object(processor, "_process_single_token", side_effect=ValueError("transform exploded")),
            patch.object(factory.scheduler, "mark_failed", side_effect=AuditIntegrityError("scheduler write rejected")),
            pytest.raises(
                AuditIntegrityError,
                match=r"original processing exception ValueError: transform exploded.*scheduler failure write.*scheduler write rejected",
            ) as exc_info,
        ):
            processor.drain_scheduled_work(ctx)

        assert isinstance(exc_info.value.__cause__, AuditIntegrityError)

    def test_resume_restores_coalesce_cursor_for_held_branch_work(self) -> None:
        """Coalesce-held branch work must rehydrate both token lineage and work-item cursor."""
        db, factory = _make_factory()
        coalesce_node = NodeID("coalesce::merge")
        source_payload = make_row({"value": 45})
        row = factory.data_flow.create_row(
            run_id="test-run",
            source_node_id="source-0",
            row_index=0,
            source_row_index=0,
            ingest_sequence=0,
            data=source_payload.to_dict(),
        )
        factory.data_flow.register_node(
            run_id="test-run",
            plugin_name="coalesce:merge",
            node_type=NodeType.COALESCE,
            plugin_version="1.0",
            config={},
            node_id=str(coalesce_node),
            schema_config=_DYNAMIC_SCHEMA,
        )
        token = TokenInfo(
            row_id=row.row_id,
            token_id="token-held-branch",
            row_data=source_payload,
            branch_name="path_a",
            fork_group_id="fork-2",
        )
        factory.data_flow.create_token(row.row_id, token_id=token.token_id)
        factory.scheduler.enqueue_ready(
            run_id="test-run",
            token_id=token.token_id,
            row_id=row.row_id,
            node_id=str(coalesce_node),
            step_index=1,
            ingest_sequence=0,
            row_payload_json=factory.scheduler.serialize_row_payload(source_payload),
            available_at=datetime.now(UTC),
            branch_name=token.branch_name,
            fork_group_id=token.fork_group_id,
            coalesce_node_id=str(coalesce_node),
            coalesce_name="merge",
        )
        coalesce = create_autospec(CoalesceExecutor, instance=True)
        coalesce.accept.return_value = CoalesceOutcome(held=True, merged_token=None)
        processor = _make_processor(
            factory,
            coalesce_executor=coalesce,
            coalesce_node_ids={CoalesceName("merge"): coalesce_node},
            node_step_map={coalesce_node: 1},
            node_to_next={NodeID("source-0"): coalesce_node, coalesce_node: None},
            coalesce_on_success_map={CoalesceName("merge"): "merged_sink"},
            scheduler=factory.scheduler,
        )
        ctx = make_context(landscape=factory.plugin_audit_writer())

        results = processor.drain_scheduled_work(ctx)

        assert results == []
        coalesce.accept.assert_called_once()
        accepted = coalesce.accept.call_args.kwargs["token"]
        assert accepted.token_id == "token-held-branch"
        assert accepted.branch_name == "path_a"
        assert coalesce.accept.call_args.kwargs["coalesce_name"] == CoalesceName("merge")
        from sqlalchemy import select

        from elspeth.core.landscape.schema import token_work_items_table

        with db.connection() as conn:
            status, barrier_key = conn.execute(
                select(token_work_items_table.c.status, token_work_items_table.c.barrier_key).where(
                    token_work_items_table.c.token_id == "token-held-branch"
                )
            ).one()
        assert (status, barrier_key) == ("blocked", "merge")

    def _make_real_coalesce_executor(self, factory: RecorderFactory, coalesce_node: NodeID) -> Any:
        """Build a production CoalesceExecutor over the factory's real repositories."""
        from elspeth.core.config import CoalesceSettings
        from elspeth.engine.coalesce_executor import CoalesceExecutor
        from elspeth.engine.tokens import TokenManager

        def step_resolver(_node_id: NodeID) -> int:
            return 1

        executor = CoalesceExecutor(
            execution=factory.execution,
            span_factory=SpanFactory(),
            token_manager=TokenManager(factory.data_flow, step_resolver=step_resolver),
            run_id="test-run",
            step_resolver=step_resolver,
            data_flow=factory.data_flow,
        )
        executor.register_coalesce(
            CoalesceSettings(name="merge", branches=["path_a", "path_b"], policy="require_all", merge="union"),
            coalesce_node,
        )
        return executor

    def _seed_held_coalesce_branch(self, factory: RecorderFactory, coalesce_node: NodeID) -> str:
        """Register the coalesce node and enqueue a path_a branch token at it.

        Returns the row_id of the forked source row.
        """
        source_payload = make_row({"value": 45})
        row = factory.data_flow.create_row(
            run_id="test-run",
            source_node_id="source-0",
            row_index=0,
            source_row_index=0,
            ingest_sequence=0,
            data=source_payload.to_dict(),
        )
        factory.data_flow.register_node(
            run_id="test-run",
            plugin_name="coalesce:merge",
            node_type=NodeType.COALESCE,
            plugin_version="1.0",
            config={},
            node_id=str(coalesce_node),
            schema_config=_DYNAMIC_SCHEMA,
        )
        factory.data_flow.create_token(row.row_id, token_id="token-held-a", branch_name="path_a", fork_group_id="fork-1")
        factory.scheduler.enqueue_ready(
            run_id="test-run",
            token_id="token-held-a",
            row_id=row.row_id,
            node_id=str(coalesce_node),
            step_index=1,
            ingest_sequence=0,
            row_payload_json=factory.scheduler.serialize_row_payload(source_payload),
            available_at=datetime.now(UTC),
            branch_name="path_a",
            fork_group_id="fork-1",
            coalesce_node_id=str(coalesce_node),
            coalesce_name="merge",
        )
        return str(row.row_id)

    def test_resume_restores_coalesce_pending_from_blocked_rows(self) -> None:
        """F1 restore inversion: resume rebuilds coalesce pendings FROM journal BLOCKED rows.

        Arrange drives the PRODUCTION hold path (drain -> accept holds ->
        mark_blocked stamps barrier_blocked_at; accept writes the OPEN
        node_state hold at attempt 0). Act builds a fresh processor +
        executor with barrier_restore. The pending entry must carry the
        journal token, the audit-derived hold state_id, and the
        max_attempt+1 offset — with no new journal rows.
        """
        from sqlalchemy import func, select

        from elspeth.core.landscape.schema import token_work_items_table

        db, factory = _make_factory()
        coalesce_node = NodeID("coalesce::merge")
        row_id = self._seed_held_coalesce_branch(factory, coalesce_node)
        arrange_executor = self._make_real_coalesce_executor(factory, coalesce_node)
        processor_kwargs: dict[str, Any] = {
            "coalesce_node_ids": {CoalesceName("merge"): coalesce_node},
            "node_step_map": {NodeID("source-0"): 0, coalesce_node: 1},
            "node_to_next": {NodeID("source-0"): coalesce_node, coalesce_node: None},
            "coalesce_on_success_map": {CoalesceName("merge"): "merged_sink"},
        }
        processor1 = _make_processor(factory, coalesce_executor=arrange_executor, **processor_kwargs)
        ctx = make_context(landscape=factory.plugin_audit_writer())

        results = processor1.drain_scheduled_work(ctx)
        assert results == []
        assert ("merge", row_id) in arrange_executor._pending

        # The hold node_state written by accept() — the state_id the restore must rediscover.
        held_states = [
            state
            for state in factory.query.get_node_states_for_token("token-held-a")
            if str(state.node_id) == str(coalesce_node) and state.status is NodeStateStatus.OPEN
        ]
        assert len(held_states) == 1
        hold_state_id = held_states[0].state_id
        assert held_states[0].attempt == 0

        with db.connection() as conn:
            rows_before = conn.execute(select(func.count()).select_from(token_work_items_table)).scalar_one()
            status, barrier_key = conn.execute(
                select(token_work_items_table.c.status, token_work_items_table.c.barrier_key).where(
                    token_work_items_table.c.token_id == "token-held-a"
                )
            ).one()
        assert (status, barrier_key) == ("blocked", "merge")

        # ---- resume: fresh executor + processor rebuild FROM the journal ----
        resumed_executor = self._make_real_coalesce_executor(factory, coalesce_node)
        _make_processor(
            factory,
            coalesce_executor=resumed_executor,
            barrier_restore=BarrierJournalRestoreContext(
                resume_checkpoint_id="ckpt-resume-1",
                barrier_scalars=None,
                batch_id_remap={},
            ),
            **processor_kwargs,
        )

        pending = resumed_executor._pending[("merge", row_id)]
        assert set(pending.branches) == {"path_a"}
        entry = pending.branches["path_a"]
        assert entry.token.token_id == "token-held-a"
        assert entry.token.branch_name == "path_a"
        assert entry.token.fork_group_id == "fork-1"
        assert entry.state_id == hold_state_id  # state_id resolved from the OPEN node_state hold
        assert entry.token.resume_attempt_offset == 1  # max_attempt(0)+1
        assert entry.token.resume_checkpoint_id == "ckpt-resume-1"
        assert pending.lost_branches == {}
        # NO new journal rows: the BLOCKED row is reused as-is.
        with db.connection() as conn:
            rows_after = conn.execute(select(func.count()).select_from(token_work_items_table)).scalar_one()
        assert rows_after == rows_before

    def test_resume_restore_recovers_adopted_coalesce_row_without_hold_state(self) -> None:
        """Adopted BLOCKED coalesce row with no OPEN node_state hold is a crash-window state.

        Under ADR-030 §E.2 journal-first ordering the adoption CAS commits in
        one transaction and accept() (which writes the PENDING hold) runs in a
        separate transaction.  A leader crash between them leaves an adopted row
        with no OPEN state_id — a reachable, legitimate crash state, not
        corruption.

        The restore reconcile (§E.3/§E.4 crash-window recovery) must:
        - detect the holdless adopted row,
        - determine the key is NOT completed (normal adoption crash, not a late
          arrival), and
        - reset barrier_adopted_epoch to NULL so the row becomes intake-pending
          again and the first drain iteration's journal-first intake processes it
          via the normal adopt + accept + trigger path.

        The processor MUST be created successfully (no AuditIntegrityError).
        After creation, the row must be intake-pending (barrier_adopted_epoch IS
        NULL) so the next drain iteration's journal-first intake picks it up.

        Re-pin of test_resume_restore_rejects_blocked_coalesce_row_without_hold_state:
        that pin pre-dated §E.2 and claimed "hold written BEFORE journal row blocks",
        which is inverted under journal-first ordering — the adoption CAS commits
        first, accept() runs after. A refusal on this crash state permanently
        wedges every subsequent resume.
        """
        db, factory = _make_factory()
        coalesce_node = NodeID("coalesce::merge")
        self._seed_held_coalesce_branch(factory, coalesce_node)
        now = datetime.now(UTC)
        # Block the row through the production claim path WITHOUT the
        # accept()-written hold node_state (simulates a crash between adoption
        # CAS commit and accept() in _intake_adopt_coalesce_row).
        claimed = factory.scheduler.claim_ready(run_id="test-run", lease_owner="seeder", lease_seconds=60, now=now)
        assert claimed is not None
        factory.scheduler.mark_blocked(
            work_item_id=claimed.work_item_id,
            queue_key=None,
            barrier_key="merge",
            now=now,
            expected_lease_owner="seeder",
        )

        resumed_executor = self._make_real_coalesce_executor(factory, coalesce_node)
        # Under §E.2 crash-window recovery the processor MUST be created without
        # error; the holdless adopted row is reset to intake-pending.
        _make_processor(
            factory,
            coalesce_executor=resumed_executor,
            coalesce_node_ids={CoalesceName("merge"): coalesce_node},
            node_step_map={NodeID("source-0"): 0, coalesce_node: 1},
            node_to_next={NodeID("source-0"): coalesce_node, coalesce_node: None},
            coalesce_on_success_map={CoalesceName("merge"): "merged_sink"},
            barrier_restore=BarrierJournalRestoreContext(
                resume_checkpoint_id="ckpt-resume-1",
                barrier_scalars=None,
                batch_id_remap={},
            ),
        )
        # After crash-window recovery: barrier_adopted_epoch must be NULL
        # (reset by the restore reconcile) so the next intake re-adopts the row.
        from sqlalchemy import select as _select

        from elspeth.core.landscape.schema import token_work_items_table as _twi

        with db.connection() as conn:
            row = conn.execute(_select(_twi.c.barrier_adopted_epoch).where(_twi.c.work_item_id == claimed.work_item_id)).one()
        assert row.barrier_adopted_epoch is None, (
            "crash-window recovery must reset barrier_adopted_epoch to NULL so the intake re-processes the row"
        )

    def test_aggregation_buffering_leaves_scheduler_work_blocked(self) -> None:
        """Buffered aggregation tokens remain active until a flush consumes them."""
        db, factory = _make_factory()
        source_node = NodeID("source-0")
        agg_node = NodeID("agg-1")
        factory.data_flow.register_node(
            run_id="test-run",
            plugin_name="agg-transform",
            node_type=NodeType.AGGREGATION,
            plugin_version="1.0",
            config={},
            node_id=str(agg_node),
            schema_config=_DYNAMIC_SCHEMA,
        )
        transform = _make_mock_transform(node_id=str(agg_node), name="agg-transform", is_batch_aware=True)
        processor = _make_processor(
            factory,
            node_step_map={source_node: 0, agg_node: 1},
            node_to_next={source_node: agg_node, agg_node: None},
            node_to_plugin={agg_node: transform},
            aggregation_settings={
                agg_node: AggregationSettings(
                    name="batch_agg",
                    plugin="agg-transform",
                    input="default",
                    on_error="discard",
                    trigger={"count": 2},
                ),
            },
            scheduler=factory.scheduler,
        )
        ctx = make_context(landscape=factory.plugin_audit_writer())

        results = processor.process_row(
            row_index=0,
            source_row=_make_source_row({"value": 1}),
            transforms=[transform],
            ctx=ctx,
            source_row_index=0,
            ingest_sequence=0,
        )

        assert len(results) == 1
        _assert_outcome_pair(results[0], None, TerminalPath.BUFFERED)
        from sqlalchemy import select

        from elspeth.core.landscape.schema import token_work_items_table

        with db.connection() as conn:
            status, barrier_key = conn.execute(
                select(token_work_items_table.c.status, token_work_items_table.c.barrier_key).where(
                    token_work_items_table.c.token_id == results[0].token.token_id
                )
            ).one()
        assert (status, barrier_key) == ("blocked", str(agg_node))

    def test_aggregation_intake_adopts_follower_blocked_work_before_flush(self) -> None:
        """A fresh leader journal-adopts a follower-blocked aggregation row."""
        db, factory = _make_factory()
        source_node = NodeID("source-0")
        agg_node = NodeID("agg-1")
        factory.data_flow.register_node(
            run_id="test-run",
            plugin_name="agg-transform",
            node_type=NodeType.AGGREGATION,
            plugin_version="1.0",
            config={},
            node_id=str(agg_node),
            schema_config=_DYNAMIC_SCHEMA,
        )
        transform = _make_mock_transform(
            node_id=str(agg_node),
            name="agg-transform",
            is_batch_aware=True,
            on_success="agg_sink",
            result=TransformResult.success(make_row({"total": 3}), success_reason={"action": "aggregate"}),
        )
        processor = _make_processor(
            factory,
            node_step_map={source_node: 0, agg_node: 1},
            node_to_next={source_node: agg_node, agg_node: None},
            node_to_plugin={agg_node: transform},
            aggregation_settings={
                agg_node: AggregationSettings(
                    name="batch_agg",
                    plugin="agg-transform",
                    input="default",
                    on_error="discard",
                    trigger={"count": 2},
                ),
            },
            scheduler=factory.scheduler,
        )
        ctx = make_context(landscape=factory.plugin_audit_writer())

        first_results = processor.process_row(
            row_index=0,
            source_row=_make_source_row({"value": 1}),
            transforms=[transform],
            ctx=ctx,
            source_row_index=0,
            ingest_sequence=0,
        )
        first_token_id = first_results[0].token.token_id

        stray_payload = make_row({"value": 99})
        stray_row = factory.data_flow.create_row(
            run_id="test-run",
            source_node_id="source-0",
            row_index=99,
            source_row_index=99,
            ingest_sequence=99,
            data=stray_payload.to_dict(),
        )
        stray_token = factory.data_flow.create_token(stray_row.row_id, token_id="stray-buffered-token")
        stray_work = factory.scheduler.enqueue_ready(
            run_id="test-run",
            token_id=stray_token.token_id,
            row_id=stray_row.row_id,
            node_id=str(agg_node),
            step_index=1,
            ingest_sequence=99,
            row_payload_json=factory.scheduler.serialize_row_payload(stray_payload),
            available_at=datetime.now(UTC),
            barrier_key=str(agg_node),
        )
        _register_test_worker(factory, "test-worker")
        stray_claim = factory.scheduler.claim_ready(
            run_id="test-run",
            lease_owner="test-worker",
            lease_seconds=30,
            now=datetime.now(UTC),
        )
        assert stray_claim is not None
        assert stray_claim.work_item_id == stray_work.work_item_id
        factory.scheduler.mark_blocked(
            work_item_id=stray_work.work_item_id,
            queue_key=None,
            barrier_key=str(agg_node),
            now=datetime.now(UTC),
            expected_lease_owner="test-worker",
        )

        second_results = processor.process_row(
            row_index=1,
            source_row=_make_source_row({"value": 2}),
            transforms=[transform],
            ctx=ctx,
            source_row_index=1,
            ingest_sequence=1,
        )

        # The follower-shaped durable row is not an orphan: it is accepted
        # from the journal, triggers the count flush with the leader's own
        # buffered row, and the current row then buffers normally.
        assert [(r.outcome, r.path) for r in second_results] == [
            (TerminalOutcome.SUCCESS, TerminalPath.DEFAULT_FLOW),
            (None, TerminalPath.BUFFERED),
        ]
        assert second_results[0].sink_name == "agg_sink"
        assert second_results[0].scheduler_pending_sink is True
        flushed_token_id = second_results[0].token.token_id
        current_token_id = second_results[1].token.token_id

        from sqlalchemy import select

        from elspeth.core.landscape.schema import token_work_items_table

        # The consumed journal rows are both terminalized by the atomic flush.
        with db.connection() as conn:
            rows = {
                row.token_id: row
                for row in conn.execute(
                    select(
                        token_work_items_table.c.token_id,
                        token_work_items_table.c.status,
                        token_work_items_table.c.barrier_adopted_epoch,
                    ).where(
                        token_work_items_table.c.token_id.in_(
                            [
                                first_token_id,
                                stray_token.token_id,
                            ]
                        )
                    )
                )
            }
        assert {token_id: (row.status, row.barrier_adopted_epoch) for token_id, row in rows.items()} == {
            first_token_id: ("terminal", 1),
            stray_token.token_id: ("terminal", 1),
        }

        with db.connection() as conn:
            statuses = {
                row.token_id: row.status
                for row in conn.execute(
                    select(token_work_items_table.c.token_id, token_work_items_table.c.status).where(
                        token_work_items_table.c.token_id.in_([flushed_token_id, current_token_id])
                    )
                )
            }
        assert statuses == {
            flushed_token_id: "pending_sink",
            current_token_id: "blocked",
        }

    def test_passthrough_aggregation_sink_flush_marks_all_outputs_pending_sink(self) -> None:
        """Every sink-bound passthrough flush token needs a durable sink handoff."""
        db, factory = _make_factory()
        source_node = NodeID("source-0")
        agg_node = NodeID("agg-1")
        contract = _make_contract()
        transform = _make_mock_transform(
            node_id=str(agg_node),
            name="agg-transform",
            is_batch_aware=True,
            on_success="agg_sink",
            result=TransformResult.success_multi(
                (
                    make_row({"value": 11}, contract=contract),
                    make_row({"value": 22}, contract=contract),
                ),
                success_reason={"action": "passthrough"},
            ),
        )
        transform.passes_through_input = True
        processor = _make_processor(
            factory,
            node_step_map={source_node: 0, agg_node: 1},
            node_to_next={source_node: agg_node, agg_node: None},
            node_to_plugin={agg_node: transform},
            aggregation_settings={
                agg_node: AggregationSettings(
                    name="batch_agg",
                    plugin="agg-transform",
                    input="default",
                    on_error="discard",
                    trigger={"count": 2},
                    output_mode="passthrough",
                ),
            },
            scheduler=factory.scheduler,
        )
        ctx = make_context(landscape=factory.plugin_audit_writer())

        first_results = processor.process_row(
            row_index=0,
            source_row=_make_source_row({"value": 1}),
            transforms=[transform],
            ctx=ctx,
            source_row_index=0,
            ingest_sequence=0,
        )
        assert len(first_results) == 1
        _assert_outcome_pair(first_results[0], None, TerminalPath.BUFFERED)

        second_results = processor.process_row(
            row_index=1,
            source_row=_make_source_row({"value": 2}),
            transforms=[transform],
            ctx=ctx,
            source_row_index=1,
            ingest_sequence=1,
        )

        # F1 Task 4.3 (rows_buffered unification): the triggering token's
        # buffer-accept surfaces as a synthetic (None, BUFFERED) result
        # alongside the two terminal flush outputs, so every accepted batch
        # member contributes exactly one BUFFERED result to the live counter.
        assert len(second_results) == 3
        buffered_results = [result for result in second_results if result.path is TerminalPath.BUFFERED]
        assert len(buffered_results) == 1
        assert buffered_results[0].outcome is None
        flush_results = [result for result in second_results if result.path is not TerminalPath.BUFFERED]
        assert len(flush_results) == 2
        flushed_token_ids = tuple(result.token.token_id for result in flush_results)
        # The triggering token both buffers (synthetic) and flushes (terminal).
        assert buffered_results[0].token.token_id in flushed_token_ids
        assert first_results[0].token.token_id in flushed_token_ids
        assert len(frozenset(flushed_token_ids)) == 2
        assert all(result.sink_name == "agg_sink" for result in flush_results)
        assert all(result.scheduler_pending_sink for result in flush_results)

        from sqlalchemy import select

        from elspeth.core.landscape.schema import token_work_items_table

        with db.connection() as conn:
            rows = conn.execute(
                select(
                    token_work_items_table.c.token_id,
                    token_work_items_table.c.status,
                    token_work_items_table.c.pending_sink_name,
                    token_work_items_table.c.pending_outcome,
                    token_work_items_table.c.pending_path,
                ).where(token_work_items_table.c.token_id.in_(flushed_token_ids))
            ).all()
        assert {(row.token_id, row.status, row.pending_sink_name, row.pending_outcome, row.pending_path) for row in rows} == {
            (token_id, "pending_sink", "agg_sink", TerminalOutcome.SUCCESS.value, TerminalPath.DEFAULT_FLOW.value)
            for token_id in flushed_token_ids
        }

        resumed_processor = _make_processor(
            factory,
            node_step_map={source_node: 0, agg_node: 1},
            node_to_next={source_node: agg_node, agg_node: None},
            node_to_plugin={agg_node: transform},
            aggregation_settings={
                agg_node: AggregationSettings(
                    name="batch_agg",
                    plugin="agg-transform",
                    input="default",
                    on_error="discard",
                    trigger={"count": 2},
                    output_mode="passthrough",
                ),
            },
            scheduler=factory.scheduler,
            scheduler_lease_owner="resume-worker",
        )
        with patch.object(
            resumed_processor._transform_executor,
            "execute_transform",
            side_effect=AssertionError("transform replayed during aggregation pending-sink recovery"),
        ):
            recovered_results = resumed_processor.drain_scheduled_work(ctx)

        recovered_token_ids = tuple(result.token.token_id for result in recovered_results)
        assert len(recovered_token_ids) == len(frozenset(recovered_token_ids))
        assert sorted(recovered_token_ids) == sorted(flushed_token_ids)
        assert all(result.sink_name == "agg_sink" for result in recovered_results)
        assert all(result.scheduler_pending_sink for result in recovered_results)

        with db.connection() as conn:
            resumed_rows = conn.execute(
                select(
                    token_work_items_table.c.token_id,
                    token_work_items_table.c.status,
                    token_work_items_table.c.lease_owner,
                ).where(token_work_items_table.c.token_id.in_(flushed_token_ids))
            ).all()
        assert {(row.token_id, row.status, row.lease_owner) for row in resumed_rows} == {
            (token_id, "leased", "resume-worker") for token_id in flushed_token_ids
        }

    def test_drain_scheduled_work_no_longer_refuses_on_peer_lease(self) -> None:
        """Slice 4 demotion: peer unexpired leases are diagnostic, not refusals.

        ADR-030 §G slice 4 demotes ``peer_active_leases`` from a hard refusal
        to a debug-level diagnostic. The correctness gate is now the
        ``active_worker_fence_clause`` membership fence compiled into the claim
        verbs — a non-member's claim CAS fails the EXISTS fence. A concurrent
        peer holding an unexpired lease is a NORMAL multi-worker state, not a
        precondition violation (filigree elspeth-66be4216cd, G3).

        This test was previously ``test_drain_refuses_when_peer_worker_holds_active_lease``
        and asserted an ``AuditIntegrityError`` raise. It now asserts the
        OPPOSITE: ``drain_scheduled_work`` must NOT raise. Worker B's drain
        enters the claim loop, finds the row still LEASED under peer-worker-A
        (``claim_ready`` returns None — no READY rows), and exits cleanly with
        an empty result list.
        """
        db, factory = _make_factory()
        transform_node = NodeID("transform-1")
        factory.data_flow.register_node(
            run_id="test-run",
            plugin_name="resume-transform",
            node_type=NodeType.TRANSFORM,
            plugin_version="1.0",
            config={},
            node_id=str(transform_node),
            schema_config=_DYNAMIC_SCHEMA,
        )
        source_payload = make_row({"value": 1})
        row = factory.data_flow.create_row(
            run_id="test-run",
            source_node_id="source-0",
            row_index=0,
            source_row_index=0,
            ingest_sequence=0,
            data=source_payload.to_dict(),
        )
        token = factory.data_flow.create_token(row.row_id, token_id="token-peer-held")
        factory.scheduler.enqueue_ready(
            run_id="test-run",
            token_id=token.token_id,
            row_id=row.row_id,
            node_id=str(transform_node),
            step_index=1,
            ingest_sequence=0,
            row_payload_json=factory.scheduler.serialize_row_payload(source_payload),
            available_at=datetime.now(UTC),
        )

        # Peer worker A claims the row under its own lease_owner. Lease window
        # is wide enough that ``peer_active_leases`` sees it as unexpired.
        peer_claim_now = datetime.now(UTC)
        _register_test_worker(factory, "peer-worker-A")
        peer_claim = factory.scheduler.claim_ready(
            run_id="test-run",
            lease_owner="peer-worker-A",
            lease_seconds=600,
            now=peer_claim_now,
        )
        assert peer_claim is not None
        assert peer_claim.lease_owner == "peer-worker-A"

        # Worker B's drain now logs the peer diagnostic but does NOT raise.
        worker_b_processor = _make_processor(
            factory,
            node_step_map={NodeID("source-0"): 0, transform_node: 1},
            node_to_next={NodeID("source-0"): transform_node, transform_node: None},
            node_to_plugin={transform_node: _make_mock_transform(node_id=str(transform_node), on_success="default")},
            scheduler=factory.scheduler,
            scheduler_lease_owner="peer-worker-B",
        )
        ctx = make_context(landscape=factory.plugin_audit_writer())

        # No raise: peer lease is diagnostic, not a refusal.
        results = worker_b_processor.drain_scheduled_work(ctx)
        # claim_ready found no READY rows (row is LEASED under peer-worker-A).
        assert results == []

        # Peer's row is still owned by peer-worker-A — B's drain was a no-op.
        from sqlalchemy import select

        from elspeth.core.landscape.schema import token_work_items_table

        with db.connection() as conn:
            status, lease_owner = conn.execute(
                select(token_work_items_table.c.status, token_work_items_table.c.lease_owner).where(
                    token_work_items_table.c.token_id == "token-peer-held"
                )
            ).one()
        assert status == "leased"
        assert lease_owner == "peer-worker-A"

    def test_drain_proceeds_when_peer_lease_expired(self) -> None:
        """An expired peer lease is recoverable, not a precondition violation.

        Single-active-resume blocks unexpired peer leases. An expired peer lease
        is the normal crashed-prior-worker recovery path: the surviving worker
        sweeps it via ``recover_expired_leases`` and proceeds. This test pins
        that the precondition does NOT over-reject — only unexpired peers are
        treated as live.
        """
        clock = MockClock(start=1_700_000_000.0)
        _db, factory = _make_factory()
        transform_node = NodeID("transform-1")
        factory.data_flow.register_node(
            run_id="test-run",
            plugin_name="resume-transform",
            node_type=NodeType.TRANSFORM,
            plugin_version="1.0",
            config={},
            node_id=str(transform_node),
            schema_config=_DYNAMIC_SCHEMA,
        )
        source_payload = make_row({"value": 1})
        row = factory.data_flow.create_row(
            run_id="test-run",
            source_node_id="source-0",
            row_index=0,
            source_row_index=0,
            ingest_sequence=0,
            data=source_payload.to_dict(),
        )
        token = factory.data_flow.create_token(row.row_id, token_id="token-stale-peer")
        factory.scheduler.enqueue_ready(
            run_id="test-run",
            token_id=token.token_id,
            row_id=row.row_id,
            node_id=str(transform_node),
            step_index=1,
            ingest_sequence=0,
            row_payload_json=factory.scheduler.serialize_row_payload(source_payload),
            available_at=clock.now_utc(),
        )

        # Crashed peer A held a short lease that has since expired.
        _register_test_worker(factory, "crashed-peer", heartbeat_expires_at=clock.now_utc() - timedelta(seconds=120))
        factory.scheduler.claim_ready(
            run_id="test-run",
            lease_owner="crashed-peer",
            lease_seconds=30,
            now=clock.now_utc(),
        )
        clock.advance(60.0)

        success_result = TransformResult.success(
            make_row({"value": 1, "resumed": True}),
            success_reason={"action": "resume_drain"},
        )
        worker_b_processor = _make_processor(
            factory,
            node_step_map={NodeID("source-0"): 0, transform_node: 1},
            node_to_next={NodeID("source-0"): transform_node, transform_node: None},
            node_to_plugin={transform_node: _make_mock_transform(node_id=str(transform_node), on_success="default")},
            scheduler=factory.scheduler,
            scheduler_lease_owner="recovery-worker",
            clock=clock,
        )
        ctx = make_context(landscape=factory.plugin_audit_writer())

        def executor_side_effect(*, transform, token, ctx, attempt=0):
            return (success_result, token, None)

        with patch.object(worker_b_processor._transform_executor, "execute_transform", side_effect=executor_side_effect):
            results = worker_b_processor.drain_scheduled_work(ctx)

        # Expired peer lease was recovered; row drained under recovery-worker.
        assert len(results) == 1
        assert results[0].token.token_id == "token-stale-peer"

    def test_drain_ignores_callers_own_active_lease(self) -> None:
        """A caller's own unexpired lease must not trigger the peer precondition.

        ``peer_active_leases`` filters ``lease_owner != caller_owner``. A worker
        re-entering its own drain (e.g. after a transform that left its own row
        LEASED in an earlier iteration) must not self-block.
        """
        _db, factory = _make_factory()
        transform_node = NodeID("transform-1")
        factory.data_flow.register_node(
            run_id="test-run",
            plugin_name="resume-transform",
            node_type=NodeType.TRANSFORM,
            plugin_version="1.0",
            config={},
            node_id=str(transform_node),
            schema_config=_DYNAMIC_SCHEMA,
        )
        source_payload = make_row({"value": 1})
        row = factory.data_flow.create_row(
            run_id="test-run",
            source_node_id="source-0",
            row_index=0,
            source_row_index=0,
            ingest_sequence=0,
            data=source_payload.to_dict(),
        )
        token = factory.data_flow.create_token(row.row_id, token_id="token-self-held")
        factory.scheduler.enqueue_ready(
            run_id="test-run",
            token_id=token.token_id,
            row_id=row.row_id,
            node_id=str(transform_node),
            step_index=1,
            ingest_sequence=0,
            row_payload_json=factory.scheduler.serialize_row_payload(source_payload),
            available_at=datetime.now(UTC),
        )

        # The caller's lease_owner pre-claims the row before the drain entry.
        _register_test_worker(factory, "own-worker")
        factory.scheduler.claim_ready(
            run_id="test-run",
            lease_owner="own-worker",
            lease_seconds=600,
            now=datetime.now(UTC),
        )

        worker_processor = _make_processor(
            factory,
            node_step_map={NodeID("source-0"): 0, transform_node: 1},
            node_to_next={NodeID("source-0"): transform_node, transform_node: None},
            node_to_plugin={transform_node: _make_mock_transform(node_id=str(transform_node), on_success="default")},
            scheduler=factory.scheduler,
            scheduler_lease_owner="own-worker",
        )
        ctx = make_context(landscape=factory.plugin_audit_writer())

        # peer_active_leases must return () because the only LEASED row is
        # owned by the caller; drain must not raise. It will reach the
        # claim_ready loop, find no READY work (the row is LEASED by self),
        # and exit cleanly with an empty result list.
        results = worker_processor.drain_scheduled_work(ctx)
        assert results == []


# =============================================================================
# _process_single_token: Inner Traversal Cycle Guard
# =============================================================================


class TestInnerTraversalCycleGuard:
    """Tests for the inner_iterations cycle guard in _process_single_token."""

    def test_cycle_in_node_to_next_raises(self) -> None:
        """Cycle in node_to_next map triggers inner traversal iteration guard.

        Uses structural nodes (not in node_to_plugin) to avoid DB interactions
        during traversal — structural nodes are traversed but not executed.
        """
        _db, factory = _make_factory()

        # Build a topology with a cycle of structural nodes: s-1 -> s-2 -> s-1
        s1 = NodeID("s-1")
        s2 = NodeID("s-2")

        processor = _make_processor(
            factory,
            node_to_plugin={},  # no plugin nodes — all structural
            node_to_next={
                NodeID("source-0"): s1,
                s1: s2,
                s2: s1,  # cycle back
            },
            node_step_map={NodeID("source-0"): 0, s1: 1, s2: 2},
            structural_node_ids=frozenset({NodeID("source-0"), s1, s2}),
        )
        ctx = make_context(landscape=factory.plugin_audit_writer())
        token = make_token_info(data={"value": 1})

        with pytest.raises(OrchestrationInvariantError, match=r"Inner traversal exceeded.*Possible cycle"):
            processor._process_single_token(
                token=token,
                ctx=ctx,
                current_node_id=s1,
            )


# =============================================================================
# _execute_transform_with_retry: No retry manager
# =============================================================================


class TestExecuteTransformNoRetry:
    """Tests for _execute_transform_with_retry when no retry_manager is set."""

    def _setup(self) -> tuple[LandscapeDB, RecorderFactory, RowProcessor]:
        db, factory = _make_factory()

        processor = _make_processor(factory, retry_manager=None)
        return db, factory, processor

    def test_single_attempt_success(self) -> None:
        """Without retry_manager, executes single attempt."""
        _, factory, processor = self._setup()
        transform = _make_mock_transform()
        token = make_token_info(data={"value": 42})
        ctx = make_context(landscape=factory.plugin_audit_writer())
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
        _, _factory, processor = self._setup()
        transform = _make_mock_transform(node_id="t1", on_error="discard")
        token = make_token_info(data={"value": 42})
        ctx = make_context()

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
        _, _factory, processor = self._setup()
        transform = _make_mock_transform(node_id="t1", on_error="discard")
        token = make_token_info(data={"value": 42})
        ctx = make_context()

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

    def test_non_retryable_llm_error_produces_error_result(self) -> None:
        """Regression: elspeth-27a8298a24 — non-retryable PluginRetryableError
        must produce an error result, not crash the run."""
        _, _factory, processor = self._setup()
        transform = _make_mock_transform(node_id="t1", on_error="discard")
        token = make_token_info(data={"value": 42})
        ctx = make_context()

        llm_error = LLMClientError("content policy", retryable=False)
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

    def test_interrupted_error_converts_to_shutdown_requested_result(self) -> None:
        """InterruptedError from a transform (e.g. a shutdown-cancelled batch
        waiter, elspeth-14571961a6) converts to the row-scoped
        shutdown_requested error result at the PROCESSOR layer — end-state
        parity with the sync path's retry.py InterruptedError convention."""
        _, _factory, processor = self._setup()
        transform = _make_mock_transform(node_id="t1", on_error="discard")
        token = make_token_info(data={"value": 42})
        ctx = make_context()

        with patch.object(
            processor._transform_executor,
            "execute_transform",
            side_effect=InterruptedError("Shutdown requested while waiting for token tok-1 (state st-1)"),
        ):
            result, _out_token, error_sink = processor._execute_transform_with_retry(
                transform=transform,
                token=token,
                ctx=ctx,
            )

        assert result.status == "error"
        assert result.reason is not None
        assert result.reason["reason"] == "shutdown_requested"
        assert result.retryable is False
        assert error_sink == "discard"

    def test_transient_connection_error_with_on_error(self) -> None:
        """ConnectionError with on_error returns error result (no re-raise)."""
        _, _factory, processor = self._setup()
        transform = _make_mock_transform(node_id="t1", on_error="discard")
        token = make_token_info(data={"value": 42})
        ctx = make_context()

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
        _, _factory, processor = self._setup()
        transform = _make_mock_transform(node_id="t1", on_error="discard")
        token = make_token_info(data={"value": 42})
        ctx = make_context()

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
        _, _factory, processor = self._setup()
        transform = _make_mock_transform(node_id="t1", on_error="discard")
        token = make_token_info(data={"value": 42})
        ctx = make_context()

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
        _, _factory, processor = self._setup()
        transform = _make_mock_transform(node_id="t1", on_error="discard")
        token = make_token_info(data={"value": 42})
        ctx = make_context()

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
        _, factory, processor = self._setup()
        # Set up error edge so the routing path is reachable
        processor._error_edge_ids = {NodeID("t1"): "error-edge-1"}

        transform = _make_mock_transform(node_id="t1", on_error="error-sink")
        token = make_token_info(data={"value": 42})
        ctx = make_context(state_id="state-123")

        llm_error = LLMClientError("rate limited", retryable=True)
        # Mock record_routing_event since we don't have a real state_id in DB
        with (
            patch.object(processor._transform_executor, "execute_transform", side_effect=llm_error),
            patch.object(factory.execution, "record_routing_event"),
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
        _, _factory, processor = self._setup()
        # No error edges configured
        processor._error_edge_ids = {}

        transform = _make_mock_transform(node_id="t1", on_error="error-sink")
        token = make_token_info(data={"value": 42})
        ctx = make_context(state_id="state-123")

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
        _, factory = _make_factory()

        retry_manager = Mock(spec=RetryManager)
        processor = _make_processor(factory, retry_manager=retry_manager)

        expected = (
            TransformResult.success(make_row({"value": 42}), success_reason={"action": "test"}),
            make_token_info(data={"value": 42}),
            None,
        )
        retry_manager.execute_with_retry.return_value = expected

        transform = _make_mock_transform()
        token = make_token_info(data={"value": 42})
        ctx = make_context(landscape=factory.plugin_audit_writer())

        result = processor._execute_transform_with_retry(
            transform=transform,
            token=token,
            ctx=ctx,
        )

        retry_manager.execute_with_retry.assert_called_once()
        assert result == expected

    def test_is_retryable_accepts_retryable_llm_error(self) -> None:
        """is_retryable callback returns True for retryable LLMClientError."""
        _, factory = _make_factory()

        retry_manager = Mock(spec=RetryManager)
        processor = _make_processor(factory, retry_manager=retry_manager)

        # Capture the is_retryable callback
        retry_manager.execute_with_retry.return_value = (
            TransformResult.success(make_row({}), success_reason={"action": "t"}),
            make_token_info(),
            None,
        )

        transform = _make_mock_transform()
        token = make_token_info()
        ctx = make_context(landscape=factory.plugin_audit_writer())

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
        _, factory = _make_factory()
        processor = _make_processor(factory, coalesce_executor=None)
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
        _, factory = _make_factory()
        coalesce = create_autospec(CoalesceExecutor, instance=True)
        processor = _make_processor(
            factory,
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
        _, factory = _make_factory()
        coalesce = create_autospec(CoalesceExecutor, instance=True)
        processor = _make_processor(
            factory,
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
        _, factory = _make_factory()
        coalesce = create_autospec(CoalesceExecutor, instance=True)
        coalesce.accept.return_value = CoalesceOutcome(held=True, merged_token=None)
        processor = _make_processor(
            factory,
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
        """When executor already recorded FAILED outcome, the intake must not record again.

        Slice 3 re-pin (ADR-030 §E.2): the accept-time failure surfaces from
        the journal-first intake (the arrival blocked first, then the
        adoption-time executor accept produced the failure), not in-claim.
        """
        _, factory = _make_factory()
        token = TokenInfo(
            row_id="row-1",
            token_id="token-1",
            row_data=make_row({}),
            branch_name="path_a",
        )
        coalesce = create_autospec(CoalesceExecutor, instance=True)
        coalesce.accept.return_value = CoalesceOutcome(
            held=False,
            merged_token=None,
            failure_reason="merge_failed:path_b_lost",
            consumed_tokens=(token,),
            outcomes_recorded=True,
        )
        processor = _make_processor(
            factory,
            coalesce_executor=coalesce,
            coalesce_node_ids={CoalesceName("merge"): NodeID("coalesce::merge")},
            node_step_map={NodeID("coalesce::merge"): 2},
        )
        ctx = make_context(landscape=factory.plugin_audit_writer())
        _persist_blocked_scheduler_work(factory, processor, token, node_id=NodeID("coalesce::merge"), barrier_key="merge", adopted=False)
        processor._live_barrier_holds[token.token_id] = _LiveBarrierHold(token=token, barrier_key="merge")

        with (
            patch.object(factory.data_flow, "record_token_outcome") as record_outcome,
            # The intake path emits through the BarrierIntakeCoordinator's
            # construction-bound seam, not the processor attribute.
            patch.object(processor._barrier_intake, "_emit_token_completed") as emit_token_completed,
        ):
            results, child_items = processor._run_barrier_intake_pass(ctx)

        assert child_items == []
        assert len(results) == 1
        _assert_outcome_pair(results[0], TerminalOutcome.FAILURE, TerminalPath.UNROUTED)
        record_outcome.assert_not_called()
        emit_token_completed.assert_called_once()
        # Backdated accept timing (§H 476): the intake passed an explicit
        # monotonic arrival anchor derived from barrier_blocked_at.
        assert coalesce.accept.call_args.kwargs["arrival_time"] is not None

    def test_coalesce_merged_at_terminal_emits_pending_sink_handoff(self) -> None:
        """All branches arrived at terminal coalesce → durable COALESCED handoff.

        Slice 3 re-pin (ADR-030 §E.2/§E.3): the merge fires from the
        journal-first intake; for a TERMINAL coalesce the COALESCED output is
        emitted as a fresh PENDING_SINK row atomically with the consumed
        branches (F1/D6) — the old in-claim memory-only ride between the
        consumption and the claim disposition is gone.
        """
        db, factory = _make_factory()
        row = factory.data_flow.create_row(
            run_id="test-run",
            source_node_id="source-0",
            row_index=0,
            source_row_index=0,
            ingest_sequence=0,
            data={},
        )
        merged_token = make_token_info(row_id=row.row_id, token_id="merged-1", data={"merged": True})
        factory.data_flow.create_token(row.row_id, token_id="merged-1")
        token = TokenInfo(
            row_id=row.row_id,
            token_id="token-1",
            row_data=make_row({}),
            branch_name="path_a",
        )
        coalesce = create_autospec(CoalesceExecutor, instance=True)
        coalesce.accept.return_value = CoalesceOutcome(held=False, merged_token=merged_token, consumed_tokens=(token,))
        processor = _make_processor(
            factory,
            coalesce_executor=coalesce,
            coalesce_node_ids={CoalesceName("merge"): NodeID("coalesce::merge")},
            node_step_map={NodeID("coalesce::merge"): 3},
            coalesce_on_success_map={CoalesceName("merge"): "output"},
        )
        ctx = make_context(landscape=factory.plugin_audit_writer())
        _persist_blocked_scheduler_work(factory, processor, token, node_id=NodeID("coalesce::merge"), barrier_key="merge", adopted=False)
        processor._live_barrier_holds[token.token_id] = _LiveBarrierHold(token=token, barrier_key="merge")

        results, child_items = processor._run_barrier_intake_pass(ctx)

        # Terminal coalesce: the COALESCED sink-bound result is emitted as a
        # fresh PENDING_SINK row in the SAME atomic completion (no READY hop)
        # — the merged output is journal-durable before the sink write.
        assert child_items == []
        assert len(results) == 1
        _assert_outcome_pair(results[0], TerminalOutcome.SUCCESS, TerminalPath.COALESCED)
        assert results[0].sink_name == "output"
        assert results[0].scheduler_pending_sink is True
        from sqlalchemy import select

        from elspeth.core.landscape.schema import token_work_items_table

        with db.connection() as conn:
            rows = dict(
                conn.execute(
                    select(token_work_items_table.c.token_id, token_work_items_table.c.status).where(
                        token_work_items_table.c.run_id == "test-run"
                    )
                ).all()
            )
        assert rows["merged-1"] == "pending_sink"
        assert rows["token-1"] == "terminal"

    def test_coalesce_merged_at_terminal_missing_sink_mapping_raises(self) -> None:
        """Terminal coalesce merge without sink mapping is an internal bug.

        Slice 3 re-pin (ADR-030 §E.2): the terminal fire resolves the sink to
        build the durable PENDING_SINK emission, so the invariant raises from
        the intake pass.
        """
        _, factory = _make_factory()
        row = factory.data_flow.create_row(
            run_id="test-run",
            source_node_id="source-0",
            row_index=0,
            source_row_index=0,
            ingest_sequence=0,
            data={},
        )
        merged_token = make_token_info(row_id=row.row_id, token_id="merged-1", data={"merged": True})
        factory.data_flow.create_token(row.row_id, token_id="merged-1")
        token = TokenInfo(
            row_id=row.row_id,
            token_id="token-1",
            row_data=make_row({}),
            branch_name="path_a",
        )
        coalesce = create_autospec(CoalesceExecutor, instance=True)
        coalesce.accept.return_value = CoalesceOutcome(held=False, merged_token=merged_token, consumed_tokens=(token,))
        processor = _make_processor(
            factory,
            coalesce_executor=coalesce,
            coalesce_node_ids={CoalesceName("merge"): NodeID("coalesce::merge")},
            node_step_map={NodeID("coalesce::merge"): 3},
            # Intentionally omit coalesce_on_success_map
        )
        ctx = make_context(landscape=factory.plugin_audit_writer())
        _persist_blocked_scheduler_work(factory, processor, token, node_id=NodeID("coalesce::merge"), barrier_key="merge", adopted=False)
        processor._live_barrier_holds[token.token_id] = _LiveBarrierHold(token=token, barrier_key="merge")

        with pytest.raises(OrchestrationInvariantError, match="Coalesce 'merge' not in on_success map"):
            processor._run_barrier_intake_pass(ctx)

    def test_coalesce_merged_at_non_terminal_queues_work(self) -> None:
        """Merged at non-terminal step → child work item added, no result.

        F1/D6: the merged child's READY continuation is journal-durable the
        moment the coalesce fires (inserted by the atomic barrier completion),
        so the merged token and its row must exist in the audit DB.
        """
        db, factory = _make_factory()
        row = factory.data_flow.create_row(
            run_id="test-run",
            source_node_id="source-0",
            row_index=0,
            source_row_index=0,
            ingest_sequence=0,
            data={},
        )
        factory.data_flow.register_node(
            run_id="test-run",
            plugin_name="coalesce:merge",
            node_type=NodeType.COALESCE,
            plugin_version="1.0",
            config={},
            node_id="coalesce::merge",
            schema_config=_DYNAMIC_SCHEMA,
        )
        merged_token = make_token_info(row_id=row.row_id, token_id="merged-1", data={"merged": True})
        factory.data_flow.create_token(row.row_id, token_id="merged-1")
        token = TokenInfo(
            row_id=row.row_id,
            token_id="token-1",
            row_data=make_row({}),
            branch_name="path_a",
        )
        coalesce = create_autospec(CoalesceExecutor, instance=True)
        coalesce.accept.return_value = CoalesceOutcome(held=False, merged_token=merged_token, consumed_tokens=(token,))
        processor = _make_processor(
            factory,
            coalesce_executor=coalesce,
            coalesce_node_ids={CoalesceName("merge"): NodeID("coalesce::merge")},
            node_step_map={NodeID("coalesce::merge"): 2},
            node_to_next={NodeID("coalesce::merge"): NodeID("transform-5")},
        )
        ctx = make_context(landscape=factory.plugin_audit_writer())
        # Slice 3 re-pin (ADR-030 §E.2): the merge fires from the
        # journal-first intake, not from an in-claim accept.
        _persist_blocked_scheduler_work(factory, processor, token, node_id=NodeID("coalesce::merge"), barrier_key="merge", adopted=False)
        processor._live_barrier_holds[token.token_id] = _LiveBarrierHold(token=token, barrier_key="merge")

        results, child_items = processor._run_barrier_intake_pass(ctx)

        assert results == []
        assert len(child_items) == 1
        assert child_items[0].current_node_id == NodeID("coalesce::merge")
        # The merged child's continuation is already journal-durable (F1/D6).
        from sqlalchemy import select

        from elspeth.core.landscape.schema import token_work_items_table

        with db.connection() as conn:
            status, node_id = conn.execute(
                select(token_work_items_table.c.status, token_work_items_table.c.node_id).where(
                    token_work_items_table.c.token_id == "merged-1"
                )
            ).one()
        assert (status, node_id) == ("ready", "coalesce::merge")

    def test_invalid_coalesce_outcome_state_raises_invariant(self) -> None:
        """CoalesceOutcome must be held, merged, or failed; empty state is invalid.

        Slice 3 re-pin (ADR-030 §E.2): the executor accept runs at intake, so
        the invalid-state invariant fires from the intake pass.
        """
        _db, factory = _make_factory()
        coalesce = create_autospec(CoalesceExecutor, instance=True)
        coalesce.accept.return_value = SimpleNamespace(
            held=False,
            merged_token=None,
            failure_reason=None,
            outcomes_recorded=False,
            late_arrival=False,
        )
        processor = _make_processor(
            factory,
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
        ctx = make_context(landscape=factory.plugin_audit_writer())
        _persist_blocked_scheduler_work(factory, processor, token, node_id=NodeID("coalesce::merge"), barrier_key="merge", adopted=False)
        processor._live_barrier_holds[token.token_id] = _LiveBarrierHold(token=token, barrier_key="merge")

        with pytest.raises(OrchestrationInvariantError, match="invalid state"):
            processor._run_barrier_intake_pass(ctx)


# =============================================================================
# complete_coalesce_merge (out-of-claim timeout/EOF coalesce fire, F1/D6)
# =============================================================================


class TestCompleteCoalesceMerge:
    """Real-journal proof of the orchestrator-facing atomic coalesce merge verb."""

    def test_complete_coalesce_merge_consumes_siblings_and_drives_merged_child(self) -> None:
        """ONE atomic transition: held branches -> TERMINAL, merged child READY+claimed.

        Out-of-claim fires (timeout/EOF) have no LEASED triggering token: every
        held branch is BLOCKED, and the merged child must be journal-durable
        before the consumed branches are terminalized. The drain's initial
        enqueue then reconciles idempotently against the READY row inserted by
        the barrier completion (strict field equality) and claims it.
        """
        db, factory = _make_factory()
        coalesce_node = NodeID("coalesce::merge")
        factory.data_flow.register_node(
            run_id="test-run",
            plugin_name="coalesce:merge",
            node_type=NodeType.COALESCE,
            plugin_version="1.0",
            config={},
            node_id=str(coalesce_node),
            schema_config=_DYNAMIC_SCHEMA,
        )
        payload = make_row({"value": 7})
        row = factory.data_flow.create_row(
            run_id="test-run",
            source_node_id="source-0",
            row_index=0,
            source_row_index=0,
            ingest_sequence=0,
            data=payload.to_dict(),
        )
        # One held branch, BLOCKED at the coalesce barrier through the
        # production verbs (enqueue -> claim -> mark_blocked).
        factory.data_flow.create_token(row.row_id, token_id="token-held-a", branch_name="path_a", fork_group_id="fork-1")
        now = datetime.now(UTC)
        factory.scheduler.enqueue_ready(
            run_id="test-run",
            token_id="token-held-a",
            row_id=row.row_id,
            node_id=str(coalesce_node),
            step_index=1,
            ingest_sequence=0,
            row_payload_json=factory.scheduler.serialize_row_payload(payload),
            available_at=now,
            branch_name="path_a",
            fork_group_id="fork-1",
            coalesce_node_id=str(coalesce_node),
            coalesce_name="merge",
        )
        claimed = factory.scheduler.claim_ready(run_id="test-run", lease_owner="seeder", lease_seconds=60, now=now)
        assert claimed is not None and claimed.token_id == "token-held-a"
        factory.scheduler.mark_blocked(
            work_item_id=claimed.work_item_id,
            queue_key=None,
            barrier_key="merge",
            now=now,
            expected_lease_owner="seeder",
        )
        held_token = TokenInfo(
            row_id=row.row_id,
            token_id="token-held-a",
            row_data=payload,
            branch_name="path_a",
            fork_group_id="fork-1",
        )
        merged_token = make_token_info(row_id=row.row_id, token_id="merged-1", data={"value": 7})
        factory.data_flow.create_token(row.row_id, token_id="merged-1", join_group_id="join-1")

        processor = _make_processor(
            factory,
            coalesce_node_ids={CoalesceName("merge"): coalesce_node},
            node_step_map={NodeID("source-0"): 0, coalesce_node: 1},
            node_to_next={NodeID("source-0"): coalesce_node, coalesce_node: None},
            coalesce_on_success_map={CoalesceName("merge"): "merged_sink"},
            scheduler=factory.scheduler,
        )
        ctx = make_context(landscape=factory.plugin_audit_writer())

        results = processor.complete_coalesce_merge(
            coalesce_name=CoalesceName("merge"),
            consumed_tokens=(held_token,),
            merged_token=merged_token,
            coalesce_node_id=coalesce_node,
            ctx=ctx,
        )

        # The merged token was driven to its terminal coalesce sink handoff
        # (same continuation semantics as the old process_token hop: the
        # merged token resolves the coalesce on_success sink).
        assert len(results) == 1
        assert results[0].token.token_id == "merged-1"
        assert (results[0].outcome, results[0].path) == (TerminalOutcome.SUCCESS, TerminalPath.DEFAULT_FLOW)
        assert results[0].sink_name == "merged_sink"
        assert results[0].scheduler_pending_sink is True

        from sqlalchemy import select

        from elspeth.core.landscape.schema import token_work_items_table

        with db.connection() as conn:
            statuses = dict(
                conn.execute(
                    select(token_work_items_table.c.token_id, token_work_items_table.c.status).where(
                        token_work_items_table.c.token_id.in_(["token-held-a", "merged-1"])
                    )
                ).all()
            )
        assert statuses == {
            "token-held-a": "terminal",
            "merged-1": "pending_sink",
        }


# =============================================================================
# resume_incomplete_token
# =============================================================================


class TestResumeIncompleteToken:
    """Tests for re-driving reconstructed incomplete tokens from the correct DAG node."""

    def test_expanded_child_inside_coalesced_branch_resumes_after_expand_node(self) -> None:
        """An expanded branch child must resume after expand, not at branch entry."""
        _, factory = _make_factory()
        ctx = make_context(landscape=factory.plugin_audit_writer())

        source_node = NodeID("source-0")
        branch_first_node = NodeID("branch-first")
        expand_node = NodeID("expand-branch")
        after_expand_node = NodeID("after-expand")
        coalesce_node = NodeID("coalesce::merge")

        processor = _make_processor(
            factory,
            node_step_map={
                source_node: 0,
                branch_first_node: 1,
                expand_node: 2,
                after_expand_node: 3,
                coalesce_node: 4,
            },
            node_to_next={
                source_node: branch_first_node,
                branch_first_node: expand_node,
                expand_node: after_expand_node,
                after_expand_node: coalesce_node,
                coalesce_node: None,
            },
            branch_to_coalesce={BranchName("path_a"): CoalesceName("merge")},
            coalesce_node_ids={CoalesceName("merge"): coalesce_node},
        )
        spec = IncompleteTokenSpec(
            token_id="token-expanded-child",
            row_id="row-1",
            branch_name="path_a",
            fork_group_id=None,
            join_group_id=None,
            expand_group_id="expand-1",
            token_data_ref="payload-1",
            step_in_pipeline=2,
            max_attempt=0,
        )

        with (
            patch.object(processor._nav, "resolve_branch_first_node", return_value=branch_first_node),
            patch.object(processor, "process_token", return_value=[]) as process_token,
        ):
            processor.resume_incomplete_token(
                spec,
                make_pipeline_row({"value": 42}),
                ctx,
                resume_checkpoint_id="checkpoint-1",
            )

        process_token.assert_called_once()
        _token_arg, _ctx_arg = process_token.call_args.args
        assert process_token.call_args.kwargs == {"current_node_id": after_expand_node}


# =============================================================================
# _notify_coalesce_of_lost_branch
# =============================================================================


class TestNotifyCoalesceOfLostBranch:
    """Tests for the branch loss notification to coalesce executor."""

    def test_no_coalesce_executor_returns_empty(self) -> None:
        """Without coalesce_executor, returns empty list."""
        _, factory = _make_factory()
        processor = _make_processor(factory, coalesce_executor=None)
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
        _, factory = _make_factory()
        coalesce = create_autospec(CoalesceExecutor, instance=True)
        processor = _make_processor(factory, coalesce_executor=coalesce)
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
        _, factory = _make_factory()
        coalesce = create_autospec(CoalesceExecutor, instance=True)
        processor = _make_processor(
            factory,
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
        _, factory = _make_factory()
        sibling_token = make_token_info(data={"value": 99})
        coalesce = create_autospec(CoalesceExecutor, instance=True)
        coalesce.notify_branch_lost.return_value = CoalesceOutcome(
            held=False,
            merged_token=None,
            failure_reason="not enough branches",
            consumed_tokens=(sibling_token,),
        )
        processor = _make_processor(
            factory,
            coalesce_executor=coalesce,
            branch_to_coalesce={BranchName("path_a"): CoalesceName("merge")},
            coalesce_node_ids={CoalesceName("merge"): NodeID("coalesce::merge")},
            node_step_map={NodeID("coalesce::merge"): 3},
        )
        _persist_blocked_scheduler_work(
            factory,
            processor,
            sibling_token,
            node_id=NodeID("coalesce::merge"),
            barrier_key="merge",
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
        _assert_outcome_pair(results[0], TerminalOutcome.FAILURE, TerminalPath.UNROUTED)
        assert results[0].error is not None
        assert "not enough branches" in results[0].error.message

    def test_lost_branch_triggers_terminal_merge(self) -> None:
        """Branch loss triggers merge at terminal coalesce → COALESCED result."""
        _, factory = _make_factory()
        merged_token = make_token_info(data={"merged": True})
        coalesce = create_autospec(CoalesceExecutor, instance=True)
        coalesce.notify_branch_lost.return_value = CoalesceOutcome(
            held=False,
            merged_token=merged_token,
            failure_reason=None,
            consumed_tokens=(),
        )
        processor = _make_processor(
            factory,
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
        _assert_outcome_pair(results[0], TerminalOutcome.SUCCESS, TerminalPath.COALESCED)
        assert results[0].sink_name == "output"

    def test_lost_branch_terminal_merge_missing_sink_mapping_raises(self) -> None:
        """Terminal coalesce merge from branch loss must have sink mapping."""
        _, factory = _make_factory()
        merged_token = make_token_info(data={"merged": True})
        coalesce = create_autospec(CoalesceExecutor, instance=True)
        coalesce.notify_branch_lost.return_value = CoalesceOutcome(
            held=False,
            merged_token=merged_token,
            failure_reason=None,
            consumed_tokens=(),
        )
        processor = _make_processor(
            factory,
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
        """Branch loss triggers merge at non-terminal step → queues work.

        F1/D6: the merged child's READY continuation is journal-durable the
        moment the merge fires, so the merged token and its row must exist
        in the audit DB.
        """
        db, factory = _make_factory()
        row = factory.data_flow.create_row(
            run_id="test-run",
            source_node_id="source-0",
            row_index=0,
            source_row_index=0,
            ingest_sequence=0,
            data={},
        )
        factory.data_flow.register_node(
            run_id="test-run",
            plugin_name="coalesce:merge",
            node_type=NodeType.COALESCE,
            plugin_version="1.0",
            config={},
            node_id="coalesce::merge",
            schema_config=_DYNAMIC_SCHEMA,
        )
        merged_token = make_token_info(row_id=row.row_id, token_id="merged-1", data={"merged": True})
        factory.data_flow.create_token(row.row_id, token_id="merged-1")
        coalesce = create_autospec(CoalesceExecutor, instance=True)
        coalesce.notify_branch_lost.return_value = CoalesceOutcome(
            held=False,
            merged_token=merged_token,
            failure_reason=None,
            consumed_tokens=(),
        )
        child_items: list[WorkItem] = []
        processor = _make_processor(
            factory,
            coalesce_executor=coalesce,
            branch_to_coalesce={BranchName("path_a"): CoalesceName("merge")},
            coalesce_node_ids={CoalesceName("merge"): NodeID("coalesce::merge")},
            node_step_map={NodeID("coalesce::merge"): 3},
            node_to_next={NodeID("coalesce::merge"): NodeID("transform-4")},
        )
        token = TokenInfo(
            row_id=row.row_id,
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
        # The merged child's continuation is already journal-durable (F1/D6).
        from sqlalchemy import select

        from elspeth.core.landscape.schema import token_work_items_table

        with db.connection() as conn:
            status, node_id = conn.execute(
                select(token_work_items_table.c.status, token_work_items_table.c.node_id).where(
                    token_work_items_table.c.token_id == "merged-1"
                )
            ).one()
        assert (status, node_id) == ("ready", "coalesce::merge")


# =============================================================================
# Unknown transform type
# =============================================================================


class TestUnknownTransformType:
    """Tests for the TypeError guard on unknown transform types."""

    def test_unknown_type_raises_type_error(self) -> None:
        """Transform that is neither TransformProtocol nor GateSettings raises TypeError."""
        _db, factory = _make_factory()
        source_row = _make_source_row()
        ctx = make_context(landscape=factory.plugin_audit_writer())

        # Create an object that is NOT a transform or gate
        class FakePlugin:
            node_id = "fake-node"

        fake_plugin = FakePlugin()
        source_node = NodeID("source-0")
        fake_node = NodeID(fake_plugin.node_id)
        processor = _make_processor(
            factory,
            node_step_map={source_node: 0, fake_node: 1},
            node_to_next={source_node: fake_node, fake_node: None},
            node_to_plugin={fake_node: fake_plugin},
        )

        with pytest.raises(TypeError, match="Unknown transform type"):
            processor.process_row(
                row_index=0,
                source_row=source_row,
                transforms=[fake_plugin],
                ctx=ctx,
                source_row_index=0,
                ingest_sequence=0,
            )


class TestRoutingInvariantFailures:
    """Regression tests for strict fail-closed routing invariants."""

    def test_unhandled_config_gate_routing_kind_raises(self) -> None:
        """ROUTE outcome with no sink_name or next_node_id is caught at GateOutcome construction."""
        with pytest.raises(ValueError, match="ROUTE action must have exactly one of sink_name, next_node_id, or discarded=True"):
            GateOutcome(
                result=GateResult(
                    row={"value": 10},
                    action=RoutingAction.route("branch_a"),
                    contract=_make_contract(),
                ),
                updated_token=make_token_info(data={"value": 10}),
                sink_name=None,
                next_node_id=None,
                child_tokens=(),
            )

    def test_missing_effective_sink_raises_invariant(self) -> None:
        """Terminal completion must not fall back when no sink can be resolved."""
        _db, factory = _make_factory()
        source_row = _make_source_row({"value": 10})
        ctx = make_context(landscape=factory.plugin_audit_writer())

        processor = _make_processor(
            factory,
            source_on_success="   ",
            node_step_map={NodeID("source-0"): 0},
            node_to_next={NodeID("source-0"): None},
        )

        with pytest.raises(OrchestrationInvariantError, match="No effective sink for token"):
            processor.process_row(
                row_index=0,
                source_row=source_row,
                transforms=[],
                ctx=ctx,
                source_row_index=0,
                ingest_sequence=0,
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
        _, factory = _make_factory()
        processor = _make_processor(factory)

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
        _, factory = _make_factory()
        processor = _make_processor(factory)

        with patch.object(
            processor._aggregation_executor,
            "get_buffer_count",
            return_value=5,
        ):
            count = processor.get_aggregation_buffer_count(NodeID("agg-1"))

        assert count == 5

    def test_get_barrier_scalars_composes_aggregation_executor(self) -> None:
        """get_barrier_scalars composes the aggregation executor's latches (F1 Task 2.4).

        Aggregation entries are re-keyed by str(node_id); with no coalesce
        executor wired, the coalesce side is empty.
        """
        from elspeth.contracts.barrier_scalars import AggregationNodeScalars, BarrierScalars

        _, factory = _make_factory()
        processor = _make_processor(factory)

        node_scalars = AggregationNodeScalars(count_fire_offset=0.5, condition_fire_offset=None)
        with patch.object(
            processor._aggregation_executor,
            "get_barrier_scalars",
            return_value={NodeID("agg-1"): node_scalars},
        ):
            result = processor.get_barrier_scalars()

        assert isinstance(result, BarrierScalars)
        assert dict(result.aggregation) == {"agg-1": node_scalars}
        assert dict(result.coalesce) == {}  # no coalesce executor → empty

    def test_get_barrier_scalars_composes_coalesce_executor(self) -> None:
        """get_barrier_scalars folds in the coalesce executor's lost-branch scalars."""
        from elspeth.contracts.barrier_scalars import BarrierScalars, CoalescePendingScalars

        _, factory = _make_factory()
        coalesce_executor = create_autospec(CoalesceExecutor, instance=True)
        pending_scalars = CoalescePendingScalars(lost_branches={"branch_b": "transform_failed"})
        coalesce_executor.get_barrier_scalars.return_value = {("merge", "row-1"): pending_scalars}
        processor = _make_processor(factory, coalesce_executor=coalesce_executor)

        with patch.object(
            processor._aggregation_executor,
            "get_barrier_scalars",
            return_value={},
        ):
            result = processor.get_barrier_scalars()

        assert isinstance(result, BarrierScalars)
        assert dict(result.aggregation) == {}
        assert dict(result.coalesce) == {("merge", "row-1"): pending_scalars}


# =============================================================================
# Telemetry emission (optional)
# =============================================================================


class TestTelemetryEmission:
    """Tests for telemetry emission behavior."""

    def test_no_telemetry_manager_does_not_crash(self) -> None:
        """Without telemetry_manager, _emit_telemetry is a no-op."""
        _, factory = _make_factory()
        processor = _make_processor(factory)
        # Should not raise
        processor._emit_telemetry(SimpleNamespace())

    def test_telemetry_manager_receives_events(self) -> None:
        """With telemetry_manager, events are forwarded."""
        _, factory = _make_factory()

        telemetry = create_autospec(TelemetryManagerProtocol, instance=True)
        processor = _make_processor(factory, telemetry_manager=telemetry)

        event = SimpleNamespace()
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
        _db, factory = _make_factory()
        source_row = _make_source_row()
        ctx = make_context(landscape=factory.plugin_audit_writer())

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
            factory,
            source_on_success="source_sink",
            node_step_map={source_node: 0, transform_node: 1},
            node_to_next={source_node: transform_node, transform_node: None},
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
                source_row_index=0,
                ingest_sequence=0,
            )

        # Parent should be EXPANDED (no sink_name)
        expanded = [r for r in results if (r.outcome, r.path) == (TerminalOutcome.TRANSIENT, TerminalPath.EXPAND_PARENT)]
        assert len(expanded) == 1

        # Children should be COMPLETED with transform's on_success sink
        completed = [r for r in results if (r.outcome, r.path) == (TerminalOutcome.SUCCESS, TerminalPath.DEFAULT_FLOW)]
        assert len(completed) == 2
        for r in completed:
            assert r.sink_name == "transform_sink", (
                f"Expected 'transform_sink' but got '{r.sink_name}'. "
                f"Terminal deagg children must inherit the transform's on_success, "
                f"not source_on_success."
            )

    def test_mid_chain_deagg_children_process_through_remaining_transforms(self) -> None:
        """Mid-chain multi-row expansion: children continue to downstream transforms."""
        _db, factory = _make_factory()
        source_row = _make_source_row()
        ctx = make_context(landscape=factory.plugin_audit_writer())

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
            factory,
            source_on_success="source_sink",
            node_step_map={source_node: 0, expander_node: 1, terminal_node: 2},
            node_to_next={source_node: expander_node, expander_node: terminal_node, terminal_node: None},
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
                source_row_index=0,
                ingest_sequence=0,
            )

        # Should have 1 EXPANDED + 2 COMPLETED (children processed through terminal)
        completed = [r for r in results if (r.outcome, r.path) == (TerminalOutcome.SUCCESS, TerminalPath.DEFAULT_FLOW)]
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
        _db, factory = _make_factory()
        ctx = make_context(landscape=factory.plugin_audit_writer())

        # Build DAG: source → transform → coalesce → downstream
        source_node = NodeID("source-0")
        transform_node = NodeID("transform-1")
        coalesce_node = NodeID("coalesce-2")
        downstream_node = NodeID("downstream-3")

        # Register coalesce node for FK constraints
        factory.data_flow.register_node(
            run_id="test-run",
            plugin_name="coalesce",
            node_type=NodeType.COALESCE,
            plugin_version="1.0",
            config={},
            node_id="coalesce-2",
            schema_config=_DYNAMIC_SCHEMA,
        )
        factory.data_flow.register_node(
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
            factory,
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
        _db, factory = _make_factory()
        ctx = make_context(landscape=factory.plugin_audit_writer())

        source_node = NodeID("source-0")
        coalesce_node = NodeID("coalesce-1")

        factory.data_flow.register_node(
            run_id="test-run",
            plugin_name="coalesce",
            node_type=NodeType.COALESCE,
            plugin_version="1.0",
            config={},
            node_id="coalesce-1",
            schema_config=_DYNAMIC_SCHEMA,
        )

        processor = _make_processor(
            factory,
            source_on_success="output",
            node_step_map={source_node: 0, coalesce_node: 1},
            node_to_next={source_node: coalesce_node, coalesce_node: None},
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
        # ADR-030 §B (slice 5): without coalesce_executor (follower mode),
        # _maybe_coalesce_token returns (True, None) → mark_blocked hand-off.
        # result is None and child_items is [] — the outer drain calls
        # mark_blocked so the leader's next intake adopts the arrival.
        result, child_items = processor._process_single_token(
            token=token,
            ctx=ctx,
            current_node_id=coalesce_node,
            coalesce_node_id=coalesce_node,
            coalesce_name=CoalesceName("merge"),
        )
        # Follower coalesce barrier: (None, []) → mark_blocked, not a completion.
        assert result is None
        assert child_items == []


class TestTerminalWorkItemInvariant:
    """Tests for current_node_id=None work-item validation."""

    def test_none_current_node_without_sink_context_raises(self) -> None:
        """None current_node_id must not default to source_on_success silently."""
        _db, factory = _make_factory()
        ctx = make_context(landscape=factory.plugin_audit_writer())
        processor = _make_processor(factory, source_on_success="source_sink")
        token = make_token_info(data={"value": 1})

        with pytest.raises(OrchestrationInvariantError, match="current_node_id=None"):
            processor._process_single_token(
                token=token,
                ctx=ctx,
                current_node_id=None,
            )

    def test_none_current_node_with_inherited_sink_is_allowed(self) -> None:
        """Explicit on_success_sink context allows terminal completion with None node."""
        _db, factory = _make_factory()
        ctx = make_context(landscape=factory.plugin_audit_writer())
        processor = _make_processor(factory, source_on_success="source_sink")
        token = make_token_info(data={"value": 1})

        result, _child_items = processor._process_single_token(
            token=token,
            ctx=ctx,
            current_node_id=None,
            on_success_sink="terminal_sink",
        )

        assert result is not None
        assert not isinstance(result, tuple)
        _assert_outcome_pair(result, TerminalOutcome.SUCCESS, TerminalPath.DEFAULT_FLOW)
        assert result.sink_name == "terminal_sink"


# =============================================================================
# Regression: P1-2026-02-14 — Gate sink routing notifies coalesce of lost branch
# =============================================================================


class TestGateSinkRoutingNotifiesCoalesce:
    """Regression tests for P1-2026-02-14: gate-routed-to-sink tokens on fork
    branches must notify coalesce of the lost branch, same as transform error
    paths (max retries, quarantine, error-routed)."""

    def test_gate_sink_route_notifies_coalesce_of_lost_branch(self) -> None:
        """Gate routing a fork-branch token to a sink must call _notify_coalesce_of_lost_branch.

        Before fix: gate sink routing returned immediately without notifying coalesce,
        causing sibling branches to remain held until timeout/end-of-source.
        After fix: coalesce is notified with reason 'gate_routed_to_sink:<sink_name>'.
        """
        _db, factory = _make_factory()
        ctx = make_context(landscape=factory.plugin_audit_writer())

        source_node = NodeID("source-0")
        gate_node = NodeID("gate-1")

        factory.data_flow.register_node(
            run_id="test-run",
            plugin_name="router-gate",
            node_type=NodeType.GATE,
            plugin_version="1.0",
            config={},
            node_id="gate-1",
            schema_config=_DYNAMIC_SCHEMA,
        )

        gate_config = GateSettings(
            name="router-gate",
            input="default",
            condition="True",
            routes={"true": "error_sink", "false": "default"},
        )

        coalesce = create_autospec(CoalesceExecutor, instance=True)
        # notify_branch_lost returns None = no immediate consequence
        coalesce.notify_branch_lost.return_value = None

        processor = _make_processor(
            factory,
            source_on_success="default",
            node_step_map={source_node: 0, gate_node: 1},
            node_to_next={source_node: gate_node, gate_node: None},
            node_to_plugin={gate_node: gate_config},
            coalesce_executor=coalesce,
            branch_to_coalesce={BranchName("path_a"): CoalesceName("merge")},
            coalesce_node_ids={CoalesceName("merge"): NodeID("coalesce::merge")},
        )

        # Create a fork-branch token
        token = TokenInfo(
            row_id="row-1",
            token_id="tok-branch-a",
            row_data=make_row({"value": 42}),
            branch_name="path_a",
        )

        # Mock gate executor to return a sink routing outcome
        sink_outcome = GateOutcome(
            result=GateResult(
                row={"value": 42},
                action=RoutingAction.route("true"),
                contract=_make_contract(),
            ),
            updated_token=token,
            sink_name="error_sink",
        )

        with patch.object(
            processor._gate_executor,
            "execute_config_gate",
            return_value=sink_outcome,
        ):
            result, _child_items = processor._process_single_token(
                token=token,
                ctx=ctx,
                current_node_id=gate_node,
                coalesce_node_id=NodeID("coalesce::merge"),
                coalesce_name=CoalesceName("merge"),
            )

        # Gate should produce ROUTED result
        if isinstance(result, tuple):
            routed = [r for r in result if (r.outcome, r.path) == (TerminalOutcome.SUCCESS, TerminalPath.GATE_ROUTED)]
            assert len(routed) == 1
            assert routed[0].sink_name == "error_sink"
        else:
            assert result is not None
            _assert_outcome_pair(result, TerminalOutcome.SUCCESS, TerminalPath.GATE_ROUTED)
            assert result.sink_name == "error_sink"

        # Coalesce must have been notified of the lost branch
        coalesce.notify_branch_lost.assert_called_once_with(
            coalesce_name=CoalesceName("merge"),
            row_id="row-1",
            lost_branch="path_a",
            reason="gate_routed_to_sink:error_sink",
        )

    def test_gate_sink_route_with_coalesce_failure_returns_sibling_results(self) -> None:
        """Gate sink routing that triggers coalesce failure returns sibling FAILED results."""
        _db, factory = _make_factory()
        ctx = make_context(landscape=factory.plugin_audit_writer())

        source_node = NodeID("source-0")
        gate_node = NodeID("gate-1")

        factory.data_flow.register_node(
            run_id="test-run",
            plugin_name="router-gate",
            node_type=NodeType.GATE,
            plugin_version="1.0",
            config={},
            node_id="gate-1",
            schema_config=_DYNAMIC_SCHEMA,
        )

        gate_config = GateSettings(
            name="router-gate",
            input="default",
            condition="True",
            routes={"true": "error_sink", "false": "default"},
        )

        sibling_token = make_token_info(data={"value": 99})
        coalesce = create_autospec(CoalesceExecutor, instance=True)
        coalesce.notify_branch_lost.return_value = CoalesceOutcome(
            held=False,
            merged_token=None,
            failure_reason="require_all policy violated",
            consumed_tokens=(sibling_token,),
        )

        processor = _make_processor(
            factory,
            source_on_success="default",
            node_step_map={source_node: 0, gate_node: 1},
            node_to_next={source_node: gate_node, gate_node: None},
            node_to_plugin={gate_node: gate_config},
            coalesce_executor=coalesce,
            branch_to_coalesce={BranchName("path_a"): CoalesceName("merge")},
            coalesce_node_ids={CoalesceName("merge"): NodeID("coalesce::merge")},
        )
        _persist_blocked_scheduler_work(
            factory,
            processor,
            sibling_token,
            node_id=NodeID("coalesce::merge"),
            barrier_key="merge",
        )

        token = TokenInfo(
            row_id="row-1",
            token_id="tok-branch-a",
            row_data=make_row({"value": 42}),
            branch_name="path_a",
        )

        sink_outcome = GateOutcome(
            result=GateResult(
                row={"value": 42},
                action=RoutingAction.route("true"),
                contract=_make_contract(),
            ),
            updated_token=token,
            sink_name="error_sink",
        )

        with patch.object(
            processor._gate_executor,
            "execute_config_gate",
            return_value=sink_outcome,
        ):
            result, _child_items = processor._process_single_token(
                token=token,
                ctx=ctx,
                current_node_id=gate_node,
                coalesce_node_id=NodeID("coalesce::merge"),
                coalesce_name=CoalesceName("merge"),
            )

        # Result must be a list: ROUTED (current) + FAILED (sibling)
        assert isinstance(result, tuple)
        assert len(result) == 2

        routed = [r for r in result if (r.outcome, r.path) == (TerminalOutcome.SUCCESS, TerminalPath.GATE_ROUTED)]
        failed = [r for r in result if (r.outcome, r.path) == (TerminalOutcome.FAILURE, TerminalPath.UNROUTED)]
        assert len(routed) == 1
        assert routed[0].sink_name == "error_sink"
        assert len(failed) == 1
        assert failed[0].error is not None
        assert "require_all" in failed[0].error.message

    def test_gate_sink_route_without_branch_skips_coalesce(self) -> None:
        """Gate sink routing a non-fork token does not attempt coalesce notification."""
        _db, factory = _make_factory()
        ctx = make_context(landscape=factory.plugin_audit_writer())

        source_node = NodeID("source-0")
        gate_node = NodeID("gate-1")

        factory.data_flow.register_node(
            run_id="test-run",
            plugin_name="router-gate",
            node_type=NodeType.GATE,
            plugin_version="1.0",
            config={},
            node_id="gate-1",
            schema_config=_DYNAMIC_SCHEMA,
        )

        gate_config = GateSettings(
            name="router-gate",
            input="default",
            condition="True",
            routes={"true": "error_sink", "false": "default"},
        )

        coalesce = create_autospec(CoalesceExecutor, instance=True)

        processor = _make_processor(
            factory,
            source_on_success="default",
            node_step_map={source_node: 0, gate_node: 1},
            node_to_next={source_node: gate_node, gate_node: None},
            node_to_plugin={gate_node: gate_config},
            coalesce_executor=coalesce,
        )

        # Non-fork token: branch_name is None
        token = TokenInfo(
            row_id="row-1",
            token_id="tok-1",
            row_data=make_row({"value": 42}),
            branch_name=None,
        )

        sink_outcome = GateOutcome(
            result=GateResult(
                row={"value": 42},
                action=RoutingAction.route("true"),
                contract=_make_contract(),
            ),
            updated_token=token,
            sink_name="error_sink",
        )

        with patch.object(
            processor._gate_executor,
            "execute_config_gate",
            return_value=sink_outcome,
        ):
            result, _child_items = processor._process_single_token(
                token=token,
                ctx=ctx,
                current_node_id=gate_node,
            )

        # Should still route correctly
        assert result is not None
        assert not isinstance(result, tuple)
        _assert_outcome_pair(result, TerminalOutcome.SUCCESS, TerminalPath.GATE_ROUTED)
        assert result.sink_name == "error_sink"

        # notify_branch_lost should NOT have been called (no branch_name)
        coalesce.notify_branch_lost.assert_not_called()

    def test_gate_discard_records_terminal_discard_outcome(self) -> None:
        """Gate route target 'discard' records a terminal audit outcome without a sink."""
        _db, factory = _make_factory()
        ctx = make_context(landscape=factory.plugin_audit_writer())

        source_node = NodeID("source-0")
        gate_node = NodeID("gate-1")

        factory.data_flow.register_node(
            run_id="test-run",
            plugin_name="drop-gate",
            node_type=NodeType.GATE,
            plugin_version="1.0",
            config={},
            node_id="gate-1",
            schema_config=_DYNAMIC_SCHEMA,
        )

        gate_config = GateSettings(
            name="drop-gate",
            input="default",
            condition="False",
            routes={"true": "main", "false": "discard"},
        )

        processor = _make_processor(
            factory,
            source_on_success="default",
            node_step_map={source_node: 0, gate_node: 1},
            node_to_next={source_node: gate_node, gate_node: None},
            node_to_plugin={gate_node: gate_config},
            coalesce_executor=create_autospec(CoalesceExecutor, instance=True),
        )
        token = TokenInfo(
            row_id="row-1",
            token_id="tok-1",
            row_data=make_row({"value": 42}),
            branch_name=None,
        )
        discard_outcome = GateOutcome(
            result=GateResult(
                row={"value": 42},
                action=RoutingAction.route("false"),
                contract=_make_contract(),
            ),
            updated_token=token,
            discarded=True,
        )

        with (
            patch.object(processor._gate_executor, "execute_config_gate", return_value=discard_outcome),
            patch.object(factory.data_flow, "record_token_outcome") as record_outcome,
        ):
            result, _child_items = processor._process_single_token(
                token=token,
                ctx=ctx,
                current_node_id=gate_node,
            )

        assert result is not None
        assert not isinstance(result, tuple)
        _assert_outcome_pair(result, TerminalOutcome.SUCCESS, TerminalPath.GATE_DISCARDED)
        assert result.sink_name is None
        record_outcome.assert_called_once()
        assert record_outcome.call_args.kwargs["outcome"] == TerminalOutcome.SUCCESS
        assert record_outcome.call_args.kwargs["path"] == TerminalPath.GATE_DISCARDED


# =============================================================================
# Bug D3: Gate jump past coalesce ordering invariant
# =============================================================================


class TestGateJumpPastCoalesceInvariant:
    """Regression tests for Bug D3: coalesce ordering invariant must be
    re-validated after a gate jump sets node_id = outcome.next_node_id.

    Without the fix, a gate jump past the coalesce node would silently
    bypass join handling because _maybe_coalesce_token only triggers
    on exact node_id equality with coalesce_node_id.
    """

    def test_gate_jump_past_coalesce_raises_invariant_error(self) -> None:
        """Gate routing past a coalesce node must raise OrchestrationInvariantError.

        Setup: source -> gate(step=1) -> coalesce(step=2) -> transform(step=3)
        The gate jumps directly to transform(step=3), skipping coalesce(step=2).
        This must raise because the token would never hit the coalesce barrier.
        """
        _db, factory = _make_factory()
        ctx = make_context(landscape=factory.plugin_audit_writer())

        source_node = NodeID("source-0")
        gate_node = NodeID("gate-1")
        coalesce_node = NodeID("coalesce::merge")
        past_coalesce_node = NodeID("transform-3")

        config_gate = GateSettings(
            name="router",
            input="in_conn",
            condition="'skip_ahead'",
            routes={"skip_ahead": "skip_conn"},
        )

        # Need a transform at the jump target with on_success so that
        # resolve_jump_target_sink succeeds before the coalesce check.
        past_coalesce_transform = _make_mock_transform(
            node_id=str(past_coalesce_node),
            name="past-coalesce",
            on_success="some_sink",
        )

        processor = _make_processor(
            factory,
            source_on_success="default",
            node_step_map={
                source_node: 0,
                gate_node: 1,
                coalesce_node: 2,
                past_coalesce_node: 3,
            },
            node_to_next={
                source_node: gate_node,
                gate_node: coalesce_node,
                coalesce_node: past_coalesce_node,
                past_coalesce_node: None,
            },
            node_to_plugin={
                gate_node: config_gate,
                past_coalesce_node: past_coalesce_transform,
            },
            coalesce_node_ids={CoalesceName("merge"): coalesce_node},
        )

        gate_contract = _make_contract()
        gate_result = GateResult(
            row={"value": 42},
            action=RoutingAction.route("skip_ahead"),
            contract=gate_contract,
        )

        def config_gate_side_effect(*, gate_config, node_id, token, ctx, token_manager=None):
            return GateOutcome(
                result=gate_result,
                updated_token=token,
                next_node_id=past_coalesce_node,  # Jump past coalesce!
            )

        with (
            patch.object(
                processor._gate_executor,
                "execute_config_gate",
                side_effect=config_gate_side_effect,
            ),
            pytest.raises(
                OrchestrationInvariantError,
                match=r"Gate jump moved token.*past its coalesce node",
            ),
        ):
            # Process with coalesce metadata so the invariant check triggers
            token = make_token_info(
                row_id="row-1",
                token_id="tok-1",
                data={"value": 42},
                branch_name="branch_a",
            )
            processor._process_single_token(
                token=token,
                ctx=ctx,
                current_node_id=gate_node,
                coalesce_node_id=coalesce_node,
                coalesce_name=CoalesceName("merge"),
            )

    def test_gate_jump_before_coalesce_is_allowed(self) -> None:
        """Gate jump to a node BEFORE the coalesce node must NOT raise.

        Setup: source -> gate(step=1) -> transform(step=2) -> coalesce(step=3)
        The gate jumps to transform(step=2) which is before coalesce(step=3).
        This is fine — the token still has to pass through the coalesce.
        """
        _db, factory = _make_factory()
        ctx = make_context(landscape=factory.plugin_audit_writer())

        source_node = NodeID("source-0")
        gate_node = NodeID("gate-1")
        transform_node = NodeID("transform-2")
        coalesce_node = NodeID("coalesce::merge")

        config_gate = GateSettings(
            name="router",
            input="in_conn",
            condition="'jump_ok'",
            routes={"jump_ok": "ok_conn"},
        )
        transform = _make_mock_transform(
            node_id=str(transform_node),
            name="passthrough",
            on_success="out_sink",
            result=TransformResult.success(
                make_row({"value": 42}),
                success_reason={"action": "pass"},
            ),
        )

        coalesce_exec = create_autospec(CoalesceExecutor, instance=True)
        coalesce_exec.accept.return_value = CoalesceOutcome(held=True, merged_token=None)

        processor = _make_processor(
            factory,
            source_on_success="default",
            node_step_map={
                source_node: 0,
                gate_node: 1,
                transform_node: 2,
                coalesce_node: 3,
            },
            node_to_next={
                source_node: gate_node,
                gate_node: None,
                transform_node: coalesce_node,
                coalesce_node: None,
            },
            node_to_plugin={
                gate_node: config_gate,
                transform_node: transform,
            },
            coalesce_node_ids={CoalesceName("merge"): coalesce_node},
            coalesce_executor=coalesce_exec,
            coalesce_on_success_map={CoalesceName("merge"): "merge_sink"},
        )

        gate_contract = _make_contract()
        gate_result = GateResult(
            row={"value": 42},
            action=RoutingAction.route("jump_ok"),
            contract=gate_contract,
        )

        def config_gate_side_effect(*, gate_config, node_id, token, ctx, token_manager=None):
            return GateOutcome(
                result=gate_result,
                updated_token=token,
                next_node_id=transform_node,  # Jump to BEFORE coalesce — allowed
            )

        def transform_side_effect(*, transform, token, ctx, attempt=0):
            return (
                TransformResult.success(
                    make_row({"value": 42}),
                    success_reason={"action": "pass"},
                ),
                token,
                None,
            )

        with (
            patch.object(
                processor._gate_executor,
                "execute_config_gate",
                side_effect=config_gate_side_effect,
            ),
            patch.object(
                processor._transform_executor,
                "execute_transform",
                side_effect=transform_side_effect,
            ),
        ):
            token = make_token_info(
                row_id="row-1",
                token_id="tok-1",
                data={"value": 42},
                branch_name="branch_a",
            )
            # Should NOT raise — jump target is before coalesce
            result, _child_items = processor._process_single_token(
                token=token,
                ctx=ctx,
                current_node_id=gate_node,
                coalesce_node_id=coalesce_node,
                coalesce_name=CoalesceName("merge"),
            )

        # Token should be held at the coalesce node without emitting a terminal result.
        assert result is None
        assert _child_items == []
        # Slice 3 re-pin (ADR-030 §E.2): acceptance is journal-first — the
        # in-claim accept is gone. The hold is stashed for the next drain
        # iteration's intake (which runs the executor accept post-adoption).
        coalesce_exec.accept.assert_not_called()
        hold = processor._live_barrier_holds["tok-1"]
        assert hold.barrier_key == "merge"
        assert hold.token.token_id == "tok-1"


class TestFlushContextImmutability:
    """_FlushContext.buffered_tokens must be truly immutable."""

    def test_buffered_tokens_is_immutable(self) -> None:
        """buffered_tokens must not be mutable after construction.

        Bug: frozen=True prevents reassignment but list contents remain mutable.
        Fix: store as tuple instead of list.
        """
        token = make_token_info(data={"value": 1})
        original_list = [token]

        fctx = _FlushContext(
            node_id=NodeID("node-1"),
            transform=SimpleNamespace(node_id="node-1"),
            settings=AggregationSettings(
                name="agg",
                plugin="batch-plugin",
                input="source",
                on_error="discard",
                trigger={"count": 1},
            ),
            buffered_tokens=tuple(original_list),
            batch_id="batch-1",
            error_msg="test",
            expand_parent_token=token,
            triggering_token=None,
            coalesce_node_id=None,
            coalesce_name=None,
        )

        # buffered_tokens must be a tuple (immutable), not a list
        assert isinstance(fctx.buffered_tokens, tuple)

        # Mutating the original list must not affect the frozen context
        original_list.append(make_token_info(data={"value": 2}))
        assert len(fctx.buffered_tokens) == 1


# =============================================================================
# _handle_transform_error_status: ROUTED_ON_ERROR producer-site invariants
# =============================================================================


class TestHandleTransformErrorStatusRoutedOnError:
    """Pin the no-fabricated-audit-data rule on the ROUTED_ON_ERROR producer site.

    Plan elspeth-5069612f3c, Task 4 Step 1: when transform_result.reason is
    falsy and the row is routed to an error sink (not "discard"), the producer
    must crash with OrchestrationInvariantError BEFORE constructing FailureInfo
    or the RowResult. Refusing to fabricate "unknown_error" is required to
    avoid a deterministic error_hash collision across unrelated falsy-error
    failures (Tier-1 audit fabrication rule).
    """

    def _setup(self) -> tuple[LandscapeDB, RecorderFactory, RowProcessor]:
        db, factory = _make_factory()
        processor = _make_processor(factory, retry_manager=None)
        return db, factory, processor

    def test_falsy_reason_with_error_sink_raises_invariant_error(self) -> None:
        """Empty reason on ROUTED_ON_ERROR path must raise before any RowResult is built.

        Construct a TransformResult with a valid reason to satisfy
        __post_init__, then mutate `reason` to an empty dict (the producer-side
        invariant must catch this even though TransformResult itself rejects
        empty reasons at construction time — defense in depth at the routing
        boundary).
        """
        _, _factory, processor = self._setup()
        token = make_token_info(data={"value": 42})

        # Build a valid error TransformResult, then forge a falsy reason to
        # exercise the producer-side guard. TransformResult is not frozen, so
        # direct attribute mutation is sufficient; we cannot use the factory
        # method with an empty dict because TransformResult.__post_init__
        # rejects it.
        tr = TransformResult(
            status="error",
            row=None,
            reason={"reason": "needs forging"},
            rows=None,
        )
        tr.reason = {}  # falsy — triggers the offensive guard

        with pytest.raises(
            OrchestrationInvariantError,
            match=r"ROUTED_ON_ERROR requires transform_result\.reason",
        ):
            processor._handle_transform_error_status(
                transform_result=tr,
                current_token=token,
                error_sink="error-sink",
                child_items=[],
            )

    def test_none_reason_with_error_sink_raises_invariant_error(self) -> None:
        """`reason=None` on the ROUTED_ON_ERROR path must also raise.

        TransformResult.__post_init__ rejects status='error' with reason=None,
        so we forge it post-construction. The producer-side guard must still
        catch the case before any FailureInfo is created.
        """
        _, _factory, processor = self._setup()
        token = make_token_info(data={"value": 42})

        tr = TransformResult(
            status="error",
            row=None,
            reason={"reason": "needs forging"},
            rows=None,
        )
        tr.reason = None  # falsy — triggers the offensive guard

        with pytest.raises(
            OrchestrationInvariantError,
            match=r"ROUTED_ON_ERROR requires transform_result\.reason",
        ):
            processor._handle_transform_error_status(
                transform_result=tr,
                current_token=token,
                error_sink="error-sink",
                child_items=[],
            )

    def test_valid_reason_routed_on_error_emits_routed_on_error_with_failure(
        self,
    ) -> None:
        """Non-falsy reason produces ROUTED_ON_ERROR with FailureInfo populated.

        Verifies the new shape: outcome is ROUTED_ON_ERROR, sink_name is set,
        and error.message is the str() of transform_result.reason. Pins that
        we never use 'unknown_error' as message — error_hash must be derived
        from the real upstream error.
        """
        from elspeth.contracts.results import FailureInfo

        _, _factory, processor = self._setup()
        token = make_token_info(data={"value": 42})
        tr = TransformResult(
            status="error",
            row=None,
            reason={"reason": "transform_failed_for_reasons"},
            rows=None,
        )

        terminal = processor._handle_transform_error_status(
            transform_result=tr,
            current_token=token,
            error_sink="error-sink",
            child_items=[],
        )
        result = terminal.result
        assert not isinstance(result, tuple)
        _assert_outcome_pair(result, TerminalOutcome.FAILURE, TerminalPath.ON_ERROR_ROUTED)
        assert result.sink_name == "error-sink"
        assert isinstance(result.error, FailureInfo)
        assert result.error.exception_type == "TransformError"
        # str({'reason': 'transform_failed_for_reasons'}) — the message is the
        # str() of the whole reason dict, not literal "unknown_error".
        assert "transform_failed_for_reasons" in result.error.message
        assert result.error.message != "unknown_error"


# =============================================================================
# F1 derivation-parity pins (Task 3.4 review follow-ups)
# =============================================================================


class TestReadyEmissionEnqueueParity:
    """Pin: the READY emission mirrors ``_enqueue_scheduler_work_item``.

    The merged-coalesce READY emission (inserted atomically by
    ``complete_barrier`` via ``_insert_ready_emission``) and the drain loop's
    idempotent ``enqueue_ready`` for the SAME WorkItem must derive identical
    field values: the enqueue reconciles against the emission-inserted row by
    deterministic ``work_item_id`` + strict field equality, so any derivation
    drift fails in production at reconciliation time. This pin makes that
    drift fail in CI instead.

    Both lanes now derive from ONE ``SchedulerWorkCodec`` (the codec's own
    round-trip invariants live in test_scheduler_work_codec.py); this test
    pins the processor-side WIRING end-to-end — that the enqueue path passes
    the codec bundle through unmodified — using the repository mapper
    (``_ready_work_item_values``, shared by ``enqueue_ready`` AND
    ``_insert_ready_emission``) as the common projection onto the full
    journal-row column set.
    """

    @pytest.mark.parametrize(
        "flavor",
        [
            # Maximal coalesce-cursor item: full fork/join/expand lineage +
            # coalesce fields set (queue_key derives None — queue blocking and
            # coalesce barriers are mutually exclusive by derivation).
            "coalesce_cursor",
            # Structural-queue item: full lineage, no coalesce fields, current
            # node structural (in node_to_next, no plugin) so queue_key derives
            # the node id.
            "structural_queue",
        ],
    )
    def test_ready_emission_mirrors_enqueue_work_item_fields(self, flavor: str, monkeypatch: pytest.MonkeyPatch) -> None:
        _, factory = _make_factory()
        coalesce_node = NodeID("coalesce::merge")
        continue_node = NodeID("after-merge")
        processor = _make_processor(
            factory,
            coalesce_node_ids={CoalesceName("merge"): coalesce_node},
            node_step_map={NodeID("source-0"): 0, coalesce_node: 1, continue_node: 2},
            node_to_next={NodeID("source-0"): coalesce_node, coalesce_node: continue_node, continue_node: None},
            coalesce_on_success_map={CoalesceName("merge"): "merged_sink"},
            # after-merge models a QUEUE node: structural by explicit
            # allowlist (production classifies queues by node type).
            structural_node_ids=frozenset({NodeID("source-0"), continue_node}),
        )

        # Lineage note: the audit trail enforces fork_group_id XOR
        # join_group_id at token creation, so the two flavors split the
        # lineage fields between them — together they pin every lineage
        # column non-None at least once.
        if flavor == "coalesce_cursor":
            token = TokenInfo(
                row_id="row-1",
                token_id="token-merged-1",
                row_data=make_pipeline_row({"value": 42}),
                branch_name="path_a",
                fork_group_id="fork-1",
                expand_group_id="expand-1",
            )
            _persist_token_for_scheduler(factory, token)
            item = WorkItem(
                token=token,
                current_node_id=continue_node,
                coalesce_node_id=coalesce_node,
                coalesce_name=CoalesceName("merge"),
                on_success_sink="merged_sink",
            )
        else:
            token = TokenInfo(
                row_id="row-1",
                token_id="token-merged-1",
                row_data=make_pipeline_row({"value": 42}),
                join_group_id="join-1",
                expand_group_id="expand-1",
            )
            _persist_token_for_scheduler(factory, token)
            item = WorkItem(
                token=token,
                current_node_id=continue_node,
                on_success_sink="merged_sink",
            )

        emission = processor._work_codec.ready_emission(item)

        captured: dict[str, Any] = {}
        real_enqueue = factory.scheduler.enqueue_ready

        def _capturing_enqueue(**kwargs: Any) -> Any:
            captured.update(kwargs)
            return real_enqueue(**kwargs)

        monkeypatch.setattr(factory.scheduler, "enqueue_ready", _capturing_enqueue)
        scheduled = processor._enqueue_scheduler_work_item(item, {})
        assert captured, "enqueue_ready was not invoked"

        # Project BOTH derivations through the production journal-row mapper
        # (_ready_work_item_values — the single mapper used by enqueue_ready
        # AND complete_barrier's _insert_ready_emission) with a pinned clock,
        # then require strict equality across the FULL column set.
        pinned_now = datetime(2026, 1, 1, tzinfo=UTC)
        enqueue_kwargs = dict(captured)
        enqueue_kwargs["available_at"] = pinned_now
        enqueue_kwargs.setdefault("attempt", 1)
        # worker_id is an enqueue_ready-only membership-fence kwarg; strip it
        # before projecting through the journal-row mapper which does not accept it.
        enqueue_kwargs.pop("worker_id", None)
        values_from_enqueue = factory.scheduler._ready_work_item_values(**enqueue_kwargs)

        # Mirror _insert_ready_emission's emission -> values mapping exactly.
        values_from_emission = factory.scheduler._ready_work_item_values(
            run_id=processor.run_id,
            token_id=emission.token_id,
            row_id=emission.row_id,
            node_id=emission.node_id,
            step_index=emission.step_index,
            ingest_sequence=emission.ingest_sequence,
            row_payload_json=emission.row_payload_json,
            available_at=pinned_now,
            attempt=emission.attempt,
            queue_key=emission.queue_key,
            barrier_key=emission.barrier_key,
            on_success_sink=emission.on_success_sink,
            branch_name=emission.branch_name,
            fork_group_id=emission.fork_group_id,
            join_group_id=emission.join_group_id,
            expand_group_id=emission.expand_group_id,
            coalesce_node_id=emission.coalesce_node_id,
            coalesce_name=emission.coalesce_name,
        )

        assert values_from_emission == values_from_enqueue
        # Pin the projected column count: adding a journal column to ONE of
        # the two builders (or to the mapper) must force this pin to be
        # revisited rather than silently desync the reconciliation contract.
        assert len(values_from_emission) == 29

        # Spot-check the per-flavor derived keys so a failure localizes.
        if flavor == "coalesce_cursor":
            assert values_from_emission["queue_key"] is None
            assert values_from_emission["barrier_key"] == "merge"
            assert values_from_emission["coalesce_node_id"] == str(coalesce_node)
            assert values_from_emission["coalesce_name"] == "merge"
            assert values_from_emission["branch_name"] == "path_a"
            assert values_from_emission["fork_group_id"] == "fork-1"
        else:
            assert values_from_emission["queue_key"] == str(continue_node)
            assert values_from_emission["barrier_key"] is None
            assert values_from_emission["coalesce_node_id"] is None
            assert values_from_emission["coalesce_name"] is None
            assert values_from_emission["join_group_id"] == "join-1"
        assert values_from_emission["expand_group_id"] == "expand-1"
        assert values_from_emission["step_index"] == 2

        # And the live reconciliation accepted the enqueue against the same
        # deterministic work_item_id (no drift at the idempotent insert).
        assert scheduled.work_item_id == values_from_emission["work_item_id"]


class TestPendingSinkAttemptOffsetSinkStepScoping:
    """Pin: PENDING_SINK re-drive attempt offsets are SINK-step scoped.

    A re-driven PENDING_SINK token only ever writes ONE further node_state —
    the sink write. Its attempt offset must therefore derive from
    ``get_max_node_state_attempts(..., step_index=sink_step)``: attempts
    recorded at PRODUCER steps (e.g. a transform retried twice before the
    crash) must not inflate the sink-write attempt number.
    """

    def test_producer_attempts_do_not_inflate_pending_sink_redrive_offset(self) -> None:
        from elspeth.contracts.scheduler import TokenWorkItem, TokenWorkStatus

        _, factory = _make_factory()
        producer_node = NodeID("transform-1")
        processor = _make_processor(
            factory,
            node_step_map={NodeID("source-0"): 0, producer_node: 1},
            node_to_next={NodeID("source-0"): producer_node, producer_node: None},
            barrier_restore=BarrierJournalRestoreContext(
                resume_checkpoint_id="ckpt-resume-1",
                barrier_scalars=None,
                batch_id_remap={},
            ),
        )
        sink_step = processor.resolve_sink_step()
        assert sink_step == 2

        token = TokenInfo(
            row_id="row-1",
            token_id="token-redrive-1",
            row_data=make_pipeline_row({"value": 42}),
        )
        _persist_token_for_scheduler(factory, token)

        # Register a sink node so sink-step node_states satisfy the FK.
        factory.data_flow.register_node(
            run_id="test-run",
            plugin_name="collect-sink",
            node_type=NodeType.SINK,
            plugin_version="1.0",
            config={},
            node_id="sink-1",
            schema_config=_DYNAMIC_SCHEMA,
        )

        # Audited history: the PRODUCER step retried twice (attempts 0..2)...
        for attempt in range(3):
            factory.execution.begin_node_state(
                token_id=token.token_id,
                node_id=str(producer_node),
                run_id="test-run",
                step_index=1,
                input_data={"value": 42},
                attempt=attempt,
            )
        # ...while the SINK step only opened attempt 0 (the crashed write).
        factory.execution.begin_node_state(
            token_id=token.token_id,
            node_id="sink-1",
            run_id="test-run",
            step_index=sink_step,
            input_data={"value": 42},
            attempt=0,
        )

        # Discriminator precondition: an UNSCOPED max over all steps WOULD
        # see the producer's attempt 2 and over-bump the offset to 3.
        unscoped = factory.execution.get_max_node_state_attempts("test-run", [token.token_id])
        assert unscoped[token.token_id] == 2

        now = datetime.now(UTC)
        scheduled = TokenWorkItem(
            work_item_id="wi-redrive-1",
            run_id="test-run",
            token_id=token.token_id,
            row_id=token.row_id,
            node_id=None,
            step_index=sink_step,
            ingest_sequence=0,
            row_payload_json=factory.scheduler.serialize_row_payload(token.row_data),
            status=TokenWorkStatus.PENDING_SINK,
            attempt=1,
            available_at=now,
            created_at=now,
            updated_at=now,
            pending_sink_name="default",
            pending_outcome=TerminalOutcome.SUCCESS.value,
            pending_path=TerminalPath.DEFAULT_FLOW.value,
        )

        result = processor._row_result_from_pending_sink(scheduled)

        # Sink-step scoped: max sink attempt 0 -> offset 1. NOT the
        # producer-inflated 3.
        assert result.token.resume_attempt_offset == 1
        assert result.token.resume_checkpoint_id == "ckpt-resume-1"
        assert result.scheduler_pending_sink is True
        assert result.sink_name == "default"
