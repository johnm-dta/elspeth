"""Resume-path execution for the orchestrator.

This module contains the full resume code path:
- setup_resume_context: rebuild GraphArtifacts from existing Landscape records
  (the resume-path equivalent of graph node/edge registration)
- run_resume_processing_loop: iterate the unprocessed rows of a resumed run,
  transform/flush/accumulate, with end-of-source aggregation + coalesce flushes
  honoured only when the resume source is truly exhausted
- ResumeCoordinator: the resume orchestration that wires the two functions
  above together (``reconstruct_resume_state``, ``resume``,
  ``process_resumed_rows``)

The two module-level functions operate on external state passed via parameters
- they don't maintain internal state. This follows the same pattern as
aggregation.py and outcomes.py: pure delegation targets.

The module-level functions were extracted from ``Orchestrator`` (where they
lived as ``_setup_resume_context`` and ``_run_resume_processing_loop``) to
shrink ``core.py``; ``ResumeCoordinator`` completes that extraction by moving
the resume orchestration off ``Orchestrator`` (which now delegates its public
``resume()`` here).
"""

from __future__ import annotations

import json
import threading
import time
from collections.abc import Callable, Mapping, Sequence
from contextlib import nullcontext
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any

from elspeth.contracts import PipelineRow, ResumedRow, RunStatus
from elspeth.contracts.config import RuntimeRetryConfig
from elspeth.contracts.coordination import (
    DEFAULT_RUN_LIVENESS_WINDOW_SECONDS,
    CoordinationToken,
    mint_worker_id,
)
from elspeth.contracts.errors import (
    AuditIntegrityError,
    EmptyResumeStateError,
    GracefulShutdownError,
    IncompleteSourceResumeError,
    OrchestrationInvariantError,
)
from elspeth.contracts.freeze import freeze_fields
from elspeth.contracts.runtime_val_manifest import build_runtime_val_manifest
from elspeth.contracts.types import NodeID
from elspeth.core.canonical import canonical_json
from elspeth.core.checkpoint.compatibility import CheckpointCompatibilityValidator
from elspeth.core.checkpoint.recovery import NonResumableRunError, check_run_status_resumable
from elspeth.core.landscape.factory import RecorderFactory

# The immutable-success family (COMPLETED / COMPLETED_WITH_FAILURES / EMPTY)
# is deliberately imported from its single source of truth rather than
# duplicated: the resume() entry guard refuses these statuses as "Run is
# terminal" (the §H loser-after-winner contract), with the durable
# immutable-success backstops retained beneath (the acquire_run_leadership
# takeover CAS and the run_lifecycle conditional UPDATEs).
from elspeth.core.landscape.run_lifecycle_repository import _IMMUTABLE_SUCCESS_RUN_STATUSES
from elspeth.core.landscape.schema import RunSourceLifecycleState
from elspeth.engine._best_effort import best_effort
from elspeth.engine.barrier_coordination import BarrierJournalRestoreContext
from elspeth.engine.orchestrator.aggregation import check_aggregation_timeouts
from elspeth.engine.orchestrator.bootstrap import prepare_for_run
from elspeth.engine.orchestrator.cleanup import cleanup_plugins
from elspeth.engine.orchestrator.graph_wiring import build_source_id_map, load_edge_map
from elspeth.engine.orchestrator.heartbeat import RunHeartbeatThread
from elspeth.engine.orchestrator.leader_drain import run_end_of_input_barrier_flush
from elspeth.engine.orchestrator.outcomes import (
    accumulate_row_outcomes,
    handle_coalesce_timeouts,
)
from elspeth.engine.orchestrator.run_state import (
    GraphArtifacts,
    LoopContext,
    ResumeState,
    _RunFailedWithPartialResultError,
)
from elspeth.engine.orchestrator.run_status import (
    cli_completion_for,
    derive_resume_terminal_status_from_audit,
)
from elspeth.engine.orchestrator.runtime_preflight import run_transform_runtime_preflights
from elspeth.engine.orchestrator.schema_reconstruction import reconstruct_schema_from_json
from elspeth.engine.orchestrator.shutdown import shutdown_handler_context
from elspeth.engine.orchestrator.types import (
    ExecutionCounters,
)
from elspeth.engine.orchestrator.validation import (
    validate_pipeline_route_targets,
)
from elspeth.engine.retry import RetryManager

if TYPE_CHECKING:
    from elspeth.contracts import ResumePoint, SchemaContract
    from elspeth.contracts.payload_store import PayloadStore
    from elspeth.core.checkpoint import CheckpointManager
    from elspeth.core.checkpoint.recovery import IncompleteTokenSpec, RecoveryManager
    from elspeth.core.config import ElspethSettings
    from elspeth.core.dag import ExecutionGraph
    from elspeth.core.events import EventBusProtocol
    from elspeth.core.landscape import LandscapeDB
    from elspeth.core.landscape.execution_repository import ExecutionRepository
    from elspeth.engine.orchestrator.ceremony import RunCeremony
    from elspeth.engine.orchestrator.checkpointing import CheckpointCoordinator
    from elspeth.engine.orchestrator.run_context_factory import RunContextFactory
    from elspeth.engine.orchestrator.sink_flush import SinkFlushCoordinator
    from elspeth.engine.orchestrator.types import PipelineConfig, RunResult


_SOURCE_COMPLETE_LIFECYCLE_STATES = frozenset(
    {
        RunSourceLifecycleState.EXHAUSTED.value,
        RunSourceLifecycleState.LOADED.value,
    }
)


def setup_resume_context(
    factory: RecorderFactory,
    run_id: str,
    config: PipelineConfig,
    graph: ExecutionGraph,
) -> GraphArtifacts:
    """Resume-path equivalent of _register_graph_nodes_and_edges().

    Loads node ID maps and edge_map from database records instead of
    registering new ones. The graph is the same as the original run,
    but nodes/edges already exist in Landscape.

    Returns:
        GraphArtifacts populated from existing Landscape records.
    """
    # Get explicit node ID mappings from graph via the SAME loader the leader
    # and follower use (elspeth-07b2031e41). Resume must preserve every
    # source root from the original multi-source DAG; a singleton
    # ``graph.get_source()`` lookup is intentionally single-source-only.
    source_id_map = build_source_id_map(graph)
    source_id = next(iter(source_id_map.values()))
    sink_id_map = graph.get_sink_id_map()
    transform_id_map = graph.get_transform_id_map()
    config_gate_id_map = graph.get_config_gate_id_map()
    coalesce_id_map = graph.get_coalesce_id_map()

    # Load edge_map from database via the shared loader (real edge IDs
    # registered in the original run — FK integrity for routing events).
    edge_map = load_edge_map(factory.data_flow, run_id)

    validate_pipeline_route_targets(
        config=config,
        route_resolution_map=graph.get_route_resolution_map(),
        transform_id_map=transform_id_map,
        config_gate_id_map=config_gate_id_map,
    )

    return GraphArtifacts(
        edge_map=edge_map,
        source_id=source_id,
        source_id_map=source_id_map,
        sink_id_map=sink_id_map,
        transform_id_map=transform_id_map,
        config_gate_id_map=config_gate_id_map,
        coalesce_id_map=coalesce_id_map,
    )


def run_resume_processing_loop(
    loop_ctx: LoopContext,
    unprocessed_rows: Sequence[ResumedRow],
    *,
    schema_contracts_by_source: Mapping[NodeID, SchemaContract],
    source_on_success_by_source: Mapping[NodeID, str] | None = None,
    incomplete_by_row: Mapping[str, Sequence[IncompleteTokenSpec]],
    recovery_manager: RecoveryManager,
    payload_store: PayloadStore,
    run_id: str,
    resume_checkpoint_id: str,
    shutdown_event: threading.Event | None = None,
    check_coordination_latch: Callable[[], None] | None = None,
) -> bool:
    """Run the resume processing loop: iterate unprocessed rows, transform, flush, accumulate.

    Includes end-of-loop aggregation/coalesce flushes only when the resume
    source is actually exhausted. On graceful shutdown we keep buffered state
    pending rather than forcing end-of-source semantics.

    Simpler than the main loop:
    - No quarantine handling (rows already validated)
    - No field resolution (already recorded in original run)
    - No schema contract recording (passed via parameter)
    - No operation_id lifecycle (no source track_operation)
    - No progress emission (known gap — see design doc)

    Per-row dispatch (F1 fix):
    - If the row has incomplete child tokens (partial fork/expand/coalesce):
      drive ONLY the incomplete children via resume_incomplete_token.
      Restarting from source (process_existing_row) would re-fork to ALL branches
      and re-emit the completed ones (the F1 double-emission defect).
    - Otherwise (never started, or fully linear): whole-row restart from source
      via process_existing_row is correct.

    Per ADR-025 §3, ``schema_contracts_by_source`` is the plural-by-source
    resume contract surface. Every ``ResumedRow`` carries a non-optional
    ``source_node_id``; missing entries are audit corruption and resume refuses
    instead of choosing a default.

    Parameters
    ----------
    check_coordination_latch:
        Optional zero-argument callable that raises
        :class:`~elspeth.contracts.errors.RunWorkerEvictedError` if the
        heartbeat thread has detected seat deposition or registry eviction.
        Called at the same boundary as the ``shutdown_event`` check — once
        per row, after row processing completes.  Pass
        ``RunHeartbeatThread.check_and_raise`` here.  ``None`` (the default)
        disables latch polling (non-coordinated runs or tests that do not start
        a heartbeat thread).

    Returns:
        True if interrupted by shutdown, False otherwise.
    """
    # Destructure loop_ctx for local access
    config = loop_ctx.config
    ctx = loop_ctx.ctx
    processor = loop_ctx.processor
    counters = loop_ctx.counters
    pending_tokens = loop_ctx.pending_tokens
    coalesce_executor = loop_ctx.coalesce_executor
    coalesce_node_map = dict(loop_ctx.coalesce_node_map)
    agg_transform_lookup = dict(loop_ctx.agg_transform_lookup)

    # A buffered-only resume can have zero unprocessed rows but still carry
    # restored aggregation/coalesce state. If shutdown is already requested,
    # honor it before any end-of-source flush work so buffered state is
    # checkpointed again instead of being flushed to sinks.
    interrupted_by_shutdown = shutdown_event is not None and shutdown_event.is_set()

    if not interrupted_by_shutdown and processor.has_scheduled_work():
        recovered_row_ids = frozenset(row.row_id for row in unprocessed_rows)
        scheduled_row_ids = processor.active_scheduled_row_ids()
        uncovered_row_ids = recovered_row_ids - scheduled_row_ids
        if uncovered_row_ids:
            formatted_uncovered = ", ".join(sorted(uncovered_row_ids))
            formatted_scheduled = ", ".join(sorted(scheduled_row_ids)) or "<none>"
            raise AuditIntegrityError(
                "Resume scheduler coverage is incomplete: active scheduler work exists, "
                "but recovered rows are not represented by scheduler work items. "
                f"Uncovered row_id(s): {formatted_uncovered}. "
                f"Scheduled row_id(s): {formatted_scheduled}. "
                "Refusing mixed scheduler/source replay to avoid skipped or duplicated rows."
            )
        results = processor.drain_scheduled_work(ctx)
        counters.rows_processed += len({result.token.row_id for result in results})
        accumulate_row_outcomes(results, counters, pending_tokens)
        unprocessed_rows = ()

    # Process each unprocessed row. Rows already exist in DB; only tokens need to
    # be created. Dispatch: partial-fork/expand/coalesce rows use mid-DAG continuation;
    # never-started and fully-linear rows use whole-row restart (process_existing_row).
    for resumed_row in unprocessed_rows:
        if interrupted_by_shutdown:
            break
        row_id = resumed_row.row_id
        source_node_id = resumed_row.source_node_id
        row_data = resumed_row.row_data
        if source_node_id not in schema_contracts_by_source:
            raise OrchestrationInvariantError(
                f"Cannot resume row {row_id!r} from source node {source_node_id!r}: "
                "source-scoped schema contract is missing from resume state "
                f"(available source_node_ids: {sorted(schema_contracts_by_source)}). "
                "The audit trail recorded the row under a source whose contract was "
                "not restored; resume refuses rather than validate under an arbitrary contract."
            )
        row_contract = schema_contracts_by_source[source_node_id]
        if source_on_success_by_source is None or source_node_id not in source_on_success_by_source:
            raise OrchestrationInvariantError(
                f"Cannot resume row {row_id!r} from source node {source_node_id!r}: "
                "source-scoped on_success routing is missing from resume state."
            )
        source_on_success = source_on_success_by_source[source_node_id]
        counters.rows_processed += 1

        # ─────────────────────────────────────────────────────────────────
        # Check for timed-out aggregations BEFORE processing this row
        # Ensures timeout flushes OLD batch before processing new row
        # ─────────────────────────────────────────────────────────────────
        # Call module function directly (no wrapper method)
        timeout_result = check_aggregation_timeouts(
            config=config,
            processor=processor,
            ctx=ctx,
            pending_tokens=pending_tokens,
            agg_transform_lookup=agg_transform_lookup,
        )
        counters.accumulate_flush_result(timeout_result)

        # Wrap row_data in PipelineRow with contract. ResumedRow.row_data may be
        # a frozen mapping; PipelineRow intentionally requires a plain dict at
        # this boundary.
        ctx.contract = row_contract
        pipeline_row = PipelineRow(data=dict(row_data), contract=row_contract)

        # F1 fix: dispatch on whether this row has incomplete fork/expand/coalesce child tokens.
        #
        # incomplete_by_row ⊆ unprocessed_rows by construction of the
        # RecoveryManager resume work set: "incomplete non-delegation token" is
        # Case 2 of row replay selection, so every partial-fork/expand/coalesce
        # row IS visited by this loop and its specs are found here.
        #
        # Lineage-field filter: get_incomplete_tokens_by_row returns ALL incomplete
        # non-delegation tokens — including linear-pipeline tokens that were interrupted
        # mid-transform (branch_name=None, fork_group_id=None, expand_group_id=None,
        # join_group_id=None). Those linear tokens are correctly handled by
        # process_existing_row (whole-row restart mints a fresh token); routing them to
        # resume_incomplete_token raises OrchestrationInvariantError (F1 regression).
        # Only dispatch specs that are provably fork/expand/coalesce children (at least
        # one lineage field set).
        # Direct key check (not .get()) — incomplete_by_row is our pre-built index
        # (Tier-1 audit data), not an external boundary. A missing key is the normal
        # "no incomplete children for this row" case.
        fork_expand_coalesce_specs = (
            [
                s
                for s in incomplete_by_row[row_id]
                if s.branch_name is not None or s.fork_group_id is not None or s.expand_group_id is not None or s.join_group_id is not None
            ]
            if row_id in incomplete_by_row
            else []
        )

        if fork_expand_coalesce_specs:
            # Partial fork/expand/coalesce completion: drive ONLY the incomplete
            # children to completion under the original parent. Restarting from
            # source (process_existing_row) would re-fork to ALL branches and
            # re-emit the completed ones (F1 double-emission defect).
            results = []
            for spec in fork_expand_coalesce_specs:
                token_row = recovery_manager.reconstruct_token_row(spec, run_id, source_row=pipeline_row, payload_store=payload_store)
                results.extend(processor.resume_incomplete_token(spec, token_row, ctx, resume_checkpoint_id=resume_checkpoint_id))
        else:
            # No incomplete fork/expand/coalesce tokens for this row (never started,
            # fully linear, or interrupted linear token): whole-row restart from source
            # is correct. process_existing_row mints a fresh token and re-traverses.
            results = processor.process_existing_row(
                row_id=row_id,
                row_data=pipeline_row,
                transforms=config.transforms,
                ctx=ctx,
                source_node_id=source_node_id,
                source_on_success=source_on_success,
            )

        # Handle all results from this row
        accumulate_row_outcomes(results, counters, pending_tokens)

        # ─────────────────────────────────────────────────────────────────
        # Check for timed-out coalesces after processing each row
        # Must check coalesce timeouts after each row to flush stale barriers
        # ─────────────────────────────────────────────────────────────────
        if coalesce_executor is not None:
            handle_coalesce_timeouts(
                coalesce_executor=coalesce_executor,
                coalesce_node_map=coalesce_node_map,
                processor=processor,
                ctx=ctx,
                counters=counters,
                pending_tokens=pending_tokens,
            )

        # ─────────────────────────────────────────────────────────────
        # COORDINATION LATCH CHECK (ADR-030 §A.3 / §C.2, slice 4)
        # Poll the heartbeat thread's latch before the shutdown check so
        # a deposed resume-takeover-leader raises RunWorkerEvictedError
        # proactively at each row boundary, without waiting for the next
        # fenced write to refuse.  An optimization on top of the
        # epoch/membership fences — both independently refuse the same
        # writes — but this surfaces the condition on the drain thread.
        # ─────────────────────────────────────────────────────────────
        if check_coordination_latch is not None:
            check_coordination_latch()

        # ─────────────────────────────────────────────────────────────
        # GRACEFUL SHUTDOWN CHECK
        # Check between row iterations — current row is fully
        # processed, outcomes recorded, safe to stop here.
        # No quarantine path in resume (rows already validated).
        # ─────────────────────────────────────────────────────────────
        if shutdown_event is not None and shutdown_event.is_set():
            interrupted_by_shutdown = True
            break

    if not interrupted_by_shutdown:
        # CRITICAL: Flush remaining barriers only at true end-of-source.
        # ADR-030 §D steps 2-3 (slice 3): journal-quiescence gate, then the
        # intake -> trigger evaluation -> flush loop until no BLOCKED barrier
        # holds remain (same helper as _post_source_iteration_work).
        run_end_of_input_barrier_flush(
            config=config,
            processor=processor,
            ctx=ctx,
            counters=counters,
            pending_tokens=pending_tokens,
            coalesce_executor=coalesce_executor,
            coalesce_node_map=coalesce_node_map,
        )

        if processor.has_unresolved_scheduler_work():
            active_work = "; ".join(processor.summarize_unresolved_scheduler_work()) or "<unknown>"
            raise OrchestrationInvariantError(
                f"Resume for run '{processor.run_id}' left non-terminal scheduler work after end-of-source flush. "
                "Blocked scheduler state must be recovered explicitly before run completion. "
                f"Active scheduler work: {active_work}."
            )

    return interrupted_by_shutdown


def _resume_failure_result_from_baseline(
    run_id: str,
    *,
    baseline: ExecutionCounters | None,
    partial_result: RunResult,
) -> RunResult:
    """Build FAILED ceremony counters from pre-resume audit + resume-local partials."""
    if baseline is None:
        return partial_result

    counters = ExecutionCounters(
        rows_processed=baseline.rows_processed + partial_result.rows_processed,
        rows_succeeded=baseline.rows_succeeded + partial_result.rows_succeeded,
        rows_failed=baseline.rows_failed + partial_result.rows_failed,
        rows_routed_success=baseline.rows_routed_success + partial_result.rows_routed_success,
        rows_routed_failure=baseline.rows_routed_failure + partial_result.rows_routed_failure,
        rows_quarantined=baseline.rows_quarantined + partial_result.rows_quarantined,
        rows_forked=baseline.rows_forked + partial_result.rows_forked,
        rows_coalesced=baseline.rows_coalesced + partial_result.rows_coalesced,
        rows_coalesce_failed=baseline.rows_coalesce_failed + partial_result.rows_coalesce_failed,
        rows_expanded=baseline.rows_expanded + partial_result.rows_expanded,
        rows_buffered=baseline.rows_buffered + partial_result.rows_buffered,
        rows_diverted=baseline.rows_diverted + partial_result.rows_diverted,
    )
    counters.routed_destinations.update(baseline.routed_destinations)
    counters.routed_destinations.update(partial_result.routed_destinations)
    return counters.to_run_result(run_id, RunStatus.FAILED)


def _derive_resume_failure_counter_baseline(factory: RecorderFactory, run_id: str) -> ExecutionCounters | None:
    """Best-effort counter baseline for FAILED resume ceremony enrichment."""
    try:
        _terminal_status, counters = derive_resume_terminal_status_from_audit(factory, run_id)
    except Exception:
        return None
    return counters


@dataclass(frozen=True, slots=True)
class _ResumeAuditSnapshot:
    """READ-ONLY resume reconstruction results, assembled BEFORE the seat CAS.

    ``reconstruct_resume_state`` used to interleave read-only reconstruction
    (manifest drift, source-lifecycle completeness, per-source schema/contract
    maps, incomplete-token index) with durable mutation (the leadership CAS and
    incomplete-batch rewrite) in one method, so a reader could not tell read
    from write at the boundary (elspeth-e4f1eb6038). This snapshot is the
    output of the read-only stage: every field is derived from read-only
    audit/recovery queries and NOTHING here has mutated durable state. The
    caller composes the durable stages (``_acquire_resume_leadership``,
    unprocessed-row restore, ``_repair_resume_batches``) on top of it, in order.
    """

    factory: RecorderFactory
    recovery: RecoveryManager
    run_id: str
    worker_id: str
    unprocessed_row_ids: Sequence[str]
    incomplete_by_row: Mapping[str, Sequence[IncompleteTokenSpec]]
    schema_contracts_by_source: Mapping[NodeID, SchemaContract]
    source_names_by_source: Mapping[NodeID, str]
    source_lifecycle_by_source: Mapping[NodeID, str]
    source_schema_classes: Mapping[NodeID, type[Any]]

    def __post_init__(self) -> None:
        freeze_fields(
            self,
            "unprocessed_row_ids",
            "incomplete_by_row",
            "schema_contracts_by_source",
            "source_names_by_source",
            "source_lifecycle_by_source",
            "source_schema_classes",
        )


class ResumeCoordinator:
    """Resume-path orchestration extracted from ``Orchestrator``.

    Composes the module-level resume helpers (``setup_resume_context`` and
    ``run_resume_processing_loop``) into the full resume flow: reconstruct
    resume state from the audit trail, process the unprocessed rows, and
    finalize the run with audit-derived terminal status. The Orchestrator
    delegates its public ``resume()`` here.
    """

    def __init__(
        self,
        *,
        db: LandscapeDB,
        events: EventBusProtocol,
        ceremony: RunCeremony,
        checkpoints: CheckpointCoordinator,
        context_factory: RunContextFactory,
        sink_flush: SinkFlushCoordinator,
        checkpoint_manager: CheckpointManager | None,
    ) -> None:
        self._db = db
        self._events = events
        self._ceremony = ceremony
        self._checkpoints = checkpoints
        self._context_factory = context_factory
        self._sink_flush = sink_flush
        self._checkpoint_manager = checkpoint_manager

    def reconstruct_resume_state(
        self,
        resume_point: ResumePoint,
        payload_store: PayloadStore,
        *,
        worker_id: str | None = None,
    ) -> ResumeState:
        """Reconstruct state needed to process resumed rows.

        Creates a fresh factory, validates resumability (read-only), then —
        as the resume path's FIRST durable act (epoch 21, ADR-030 §B.4) —
        executes the seat-acquisition CAS ``acquire_run_leadership``: one
        BEGIN IMMEDIATE transaction carrying the seat takeover, the
        FAILED/INTERRUPTED→RUNNING run-status flip, and the identity-eviction
        of the deposed leader. A CAS loser is refused with
        ``NonResumableRunError`` and ZERO mutation — this closes the
        documented resume TOCTOU. Only after winning the seat does this
        method rewrite incomplete batches. (Barrier state is NOT restored
        here — that happens at processor construction, which rebuilds
        executors from journal BLOCKED rows plus checkpoint scalars.)

        Args:
            resume_point: ResumePoint from RecoveryManager.get_resume_point()
            payload_store: PayloadStore for retrieving row data
            worker_id: §A.1 worker identity minted by ``resume()``;
                self-minted when None (direct repository-level callers).

        Returns:
            ResumeState with all reconstruction results, including the
            leader ``coordination_token`` minted by the takeover CAS.

        Raises:
            ValueError: If checkpoint_manager is not initialized.
            OrchestrationInvariantError: If schema contract is missing from audit trail.
            NonResumableRunError: If the resume checkpoint format is incompatible.
            NonResumableRunError: If the seat CAS loses to a live leader.
            AuditIntegrityError: If the run is terminally successful
                (immutable-success durable backstop inside the CAS).
        """
        format_check = CheckpointCompatibilityValidator().validate_format_version(resume_point.checkpoint)
        if not format_check.can_resume:
            assert format_check.reason is not None
            raise NonResumableRunError(resume_point.checkpoint.run_id, format_check.reason)

        # Stage 1 — READ-ONLY reconstruction (no durable mutation).
        snapshot = self._load_resume_audit_snapshot(resume_point, payload_store, worker_id=worker_id)

        # Stage 2 — THE FIRST DURABLE ACT: the seat-acquisition CAS (epoch 21,
        # ADR-030 §B.4 — TOCTOU closure). A CAS loser raises NonResumableRunError
        # with zero mutation. See _acquire_resume_leadership.
        coordination_token = self._acquire_resume_leadership(snapshot)

        # Unprocessed-row payload restore, AFTER the seat CAS (elspeth-e3d1310b93).
        # get_unprocessed_row_data_by_source retrieves + json-decodes +
        # Pydantic-validates every unprocessed payload blob from the payload
        # store, so running it after acquire_run_leadership means a LOSING resume
        # contender is refused (NonResumableRunError, zero mutation) BEFORE paying
        # that read cost — the CAS now guards the expensive input boundary, not
        # just durable mutation. The cheap read-only refusals (manifest drift,
        # source-lifecycle completeness) stay pre-CAS in _load_resume_audit_snapshot;
        # nothing between the CAS and this restore consumes unprocessed_rows, and
        # ResumeState is still assembled with it below.
        unprocessed_rows = snapshot.recovery.get_unprocessed_row_data_by_source(
            snapshot.run_id,
            payload_store,
            source_schema_classes=snapshot.source_schema_classes,
            row_ids=snapshot.unprocessed_row_ids,
        )

        # Stage 3 — durable post-CAS repair (only the seat winner may run it):
        # rewrite incomplete batches + detect restored barrier work.
        batch_id_remap, has_restored_barrier_work = self._repair_resume_batches(snapshot)

        return ResumeState(
            factory=snapshot.factory,
            run_id=snapshot.run_id,
            unprocessed_rows=unprocessed_rows,
            incomplete_by_row=snapshot.incomplete_by_row,
            recovery_manager=snapshot.recovery,
            schema_contracts_by_source=snapshot.schema_contracts_by_source,
            source_names_by_source=snapshot.source_names_by_source,
            source_lifecycle_by_source=snapshot.source_lifecycle_by_source,
            has_restored_barrier_work=has_restored_barrier_work,
            batch_id_remap=batch_id_remap,
            coordination_token=coordination_token,
        )

    def _load_resume_audit_snapshot(
        self, resume_point: ResumePoint, payload_store: PayloadStore, *, worker_id: str | None
    ) -> _ResumeAuditSnapshot:
        """READ-ONLY resume reconstruction (elspeth-e4f1eb6038 stage 1).

        Create a fresh factory, verify resumability (runtime-VAL manifest drift
        + source-lifecycle completeness), and reconstruct the per-source
        schema/contract maps and the incomplete-token index. Performs NO durable
        mutation — the seat CAS (``_acquire_resume_leadership``), the
        unprocessed-row restore, and batch repair (``_repair_resume_batches``)
        all run in the caller AFTER this returns. Incomplete-source refusal is an
        operator-facing "start fresh or use a source-aware resume path" outcome;
        it must not strand the run as RUNNING or rewrite retry batches merely
        because the operator probed resume.
        """
        run_id = resume_point.checkpoint.run_id
        worker_id = worker_id or mint_worker_id(run_id)

        # Create fresh factory (stateless, like run()); pass payload_store for
        # external call payload persistence.
        factory = RecorderFactory(self._db, payload_store=payload_store)

        from elspeth.core.checkpoint import RecoveryManager

        if self._checkpoint_manager is None:
            raise OrchestrationInvariantError(
                "CheckpointManager is required for resume - Orchestrator must be initialized with checkpoint_manager"
            )
        recovery = RecoveryManager(self._db, self._checkpoint_manager)

        # Resume replays persisted PipelineRow payloads through NullSource rather
        # than re-opening the original source plugin, so source-boundary evidence
        # is inherited from the original run. That is only sound if the current
        # declaration-contract and Tier-1 registries still exactly match the
        # manifest captured at original run start.
        recorded_runtime_val_manifest = factory.run_lifecycle.get_runtime_val_manifest(run_id)
        current_runtime_val_manifest = canonical_json(build_runtime_val_manifest())
        if current_runtime_val_manifest != recorded_runtime_val_manifest:
            raise OrchestrationInvariantError(
                f"Cannot resume run '{run_id}': runtime VAL manifest drift detected. "
                "The current contract registry no longer matches the registry "
                "captured in the original run header, so inherited source-boundary "
                "evidence is no longer trustworthy. Resume requires identical "
                "declaration-contract and Tier-1 registries."
            )

        # ADR-025 §3 Decision 5 (G6): schema contracts are plural-by-source and
        # live exclusively in ``run_sources``. ``verify_contract_integrity``
        # (called via ``can_resume`` → ``get_resume_point`` before this method
        # runs) already raises ``EmptyResumeStateError`` when ``run_sources`` is
        # empty, so by the time we land here every declared source has a contract
        # record. We still assert the postcondition defensively against future
        # call-path changes: an empty map at resume time is Tier-1 audit corruption.
        source_lifecycle_records = factory.run_lifecycle.get_run_source_lifecycle_records(run_id)
        if not source_lifecycle_records:
            raise EmptyResumeStateError(run_id=run_id)
        incomplete_sources = {
            record.source_name: record.lifecycle_state
            for record in source_lifecycle_records.values()
            if record.lifecycle_state not in _SOURCE_COMPLETE_LIFECYCLE_STATES
        }
        if incomplete_sources:
            raise IncompleteSourceResumeError(run_id, incomplete_sources)

        # F1 fix: pre-compute resume work so the resume loop can dispatch
        # partial-fork/expand/coalesce rows via mid-DAG continuation rather
        # than whole-row restart (which would re-emit completed branches).
        # The same query boundary also yields unprocessed row IDs that are
        # reused by post-CAS row hydration instead of re-running the predicates.
        resume_workset = recovery.get_resume_workset(run_id)

        source_records = factory.run_lifecycle.get_run_source_resume_records(run_id)
        source_schema_classes: dict[NodeID, type[Any]] = {}
        schema_contracts_by_source: dict[NodeID, SchemaContract] = {}
        source_names_by_source: dict[NodeID, str] = {}
        source_lifecycle_by_source: dict[NodeID, str] = {}
        for raw_source_node_id, source_record in source_records.items():
            source_node_id = NodeID(str(raw_source_node_id))
            schema_dict = json.loads(source_record.source_schema_json)
            source_schema_classes[source_node_id] = reconstruct_schema_from_json(schema_dict)
            schema_contracts_by_source[source_node_id] = source_record.schema_contract
            source_names_by_source[source_node_id] = str(source_record.source_name)
            source_lifecycle_by_source[source_node_id] = str(source_record.lifecycle_state)

        return _ResumeAuditSnapshot(
            factory=factory,
            recovery=recovery,
            run_id=run_id,
            worker_id=worker_id,
            unprocessed_row_ids=resume_workset.row_ids,
            incomplete_by_row=resume_workset.incomplete_by_row,
            schema_contracts_by_source=schema_contracts_by_source,
            source_names_by_source=source_names_by_source,
            source_lifecycle_by_source=source_lifecycle_by_source,
            source_schema_classes=source_schema_classes,
        )

    def _acquire_resume_leadership(self, snapshot: _ResumeAuditSnapshot) -> CoordinationToken:
        """THE FIRST DURABLE ACT of resume (epoch 21, ADR-030 §B.4 — TOCTOU
        closure): the seat-acquisition CAS. One BEGIN IMMEDIATE transaction =
        seat takeover (leader_epoch+1) + the FAILED/INTERRUPTED→RUNNING
        run-status flip (which subsumed the old update_run_status(RUNNING)
        first-durable-write) + identity-eviction of the deposed leader +
        leader_acquire/worker_register/worker_evict events. A CAS loser raises
        NonResumableRunError with zero mutation; a terminally-successful run is
        refused by the immutable-success durable backstop (AuditIntegrityError);
        a held WAL write lock surfaces as the operator-actionable
        WriteLockHeldError naming the registered workers' pids.
        """
        return snapshot.factory.run_coordination.acquire_run_leadership(
            run_id=snapshot.run_id,
            worker_id=snapshot.worker_id,
            now=datetime.now(UTC),
            window_seconds=DEFAULT_RUN_LIVENESS_WINDOW_SECONDS,
            entry_point="resume",
        )

    def _repair_resume_batches(self, snapshot: _ResumeAuditSnapshot) -> tuple[Mapping[str, str], bool]:
        """Durable post-CAS repair — runs strictly AFTER
        ``_acquire_resume_leadership`` (only the seat winner may rewrite retry
        batches).

        Rewrites incomplete batches — the returned old→retry batch_id mapping
        feeds the processor's journal-based barrier restore (BUFFERED
        token_outcomes still carry the dead original batch ids after a
        flush-interrupting crash) — and reports whether the scheduler journal
        carries BLOCKED barrier rows. (F1: barrier restore itself runs in
        PROCESSOR CONSTRUCTION; a run whose remaining work all sits at barriers
        has zero unprocessed rows but must still run the processing path so the
        restored buffers flush, so the resume quiescence gate consults this flag.)
        """
        batch_id_remap = handle_incomplete_batches(snapshot.factory.execution, snapshot.run_id)
        has_restored_barrier_work = snapshot.recovery.count_blocked_barrier_items(snapshot.run_id) > 0
        return batch_id_remap, has_restored_barrier_work

    def _finalize_successful_resume(
        self,
        *,
        factory: RecorderFactory,
        run_id: str,
        coordination_token: CoordinationToken,
        heartbeat: RunHeartbeatThread,
        duration_seconds: float,
    ) -> RunResult:
        """Complete the shared successful-resume ceremony."""
        terminal_status, audit_counters = derive_resume_terminal_status_from_audit(factory, run_id)
        factory.run_lifecycle.finalize_run(run_id, status=terminal_status, token=coordination_token)
        result = audit_counters.to_run_result(run_id, terminal_status)

        # Delete checkpoints on successful completion. LEADER WORK: the
        # delete is epoch-fenced (ADR-030 §C.4 row 5) and must run BEFORE
        # the seat release vacates the fence's CAS target.
        self._checkpoints.delete_checkpoints(run_id)

        # ADR-030 §A.3: stop the heartbeat thread BEFORE releasing the
        # seat — the thread must not beat the seat after it is vacated.
        heartbeat.stop()

        # Seat hygiene (ADR-030 §D): release the seat AFTER the terminal
        # finalize succeeded. Best-effort.
        with best_effort("Seat release after resume finalize", run_id=run_id):
            factory.run_coordination.release_seat(token=coordination_token, now=datetime.now(UTC))

        self._ceremony.emit_run_finished(
            run_id=run_id,
            status=terminal_status,
            row_count=result.rows_processed,
            duration_seconds=duration_seconds,
        )

        cli_status, exit_code = cli_completion_for(terminal_status)
        self._ceremony.emit_run_summary(
            run_id=run_id,
            status=cli_status,
            rows_processed=result.rows_processed,
            rows_succeeded=result.rows_succeeded,
            rows_failed=result.rows_failed,
            rows_quarantined=result.rows_quarantined,
            duration_seconds=duration_seconds,
            exit_code=exit_code,
            rows_routed_success=result.rows_routed_success,
            rows_routed_failure=result.rows_routed_failure,
            routed_destinations=result.routed_destinations,
        )

        return result

    def resume(
        self,
        resume_point: ResumePoint,
        config: PipelineConfig,
        graph: ExecutionGraph,
        *,
        payload_store: PayloadStore,
        settings: ElspethSettings | None = None,
        shutdown_event: threading.Event | None = None,
    ) -> RunResult:
        """Resume a failed run from a checkpoint.

        STATELESS: Like run(), creates fresh factory and processor internally.
        This mirrors the reality that recovery happens in a new process.

        Args:
            resume_point: ResumePoint from RecoveryManager.get_resume_point()
            config: Same PipelineConfig used for original run()
            graph: Same ExecutionGraph used for original run()
            payload_store: PayloadStore for retrieving row data (required)
            settings: Full settings (optional, for retry config etc.)

        Returns:
            RunResult with recovery outcome

        Raises:
            ValueError: If payload_store is not provided
        """
        if payload_store is None:
            raise OrchestrationInvariantError("payload_store is required for resume - row data must be retrieved from stored payloads")

        # ---- resume() entry guard (elspeth-2f23292372, operator option b) ----
        # resume() historically trusted callers to honor the ADVISORY
        # RecoveryManager.can_resume() and never re-checked run status itself,
        # so a competing resume against a RUNNING run was ADMITTED. Re-check
        # here via the SAME shared implementation can_resume() uses, BEFORE
        # the first mutation. The RUNNING arm's live-seat precision (§B.3:
        # naming the incumbent + `elspeth join` direction) lives INSIDE
        # check_run_status_resumable so the advisory and enforcing surfaces
        # never drift.
        #
        # Terminal-run arm (epoch 21, ADR-030 §H test #1 flip): the
        # immutable-success family (COMPLETED / COMPLETED_WITH_FAILURES /
        # EMPTY) is now refused HERE, at the entry guard, with a "Run is
        # terminal" NonResumableRunError — the designed loser-after-winner
        # contract. The durable immutability guards stay BENEATH as the
        # backstop (the immutable-success arm inside the
        # acquire_run_leadership takeover CAS, and update_run_status /
        # complete_run's conditional UPDATEs — independently pinned in
        # tests/unit/core/landscape/), so a caller that skips this guard
        # still cannot mutate a successful terminal run.
        #
        # The old TOCTOU residual (two resumes both observing FAILED here) is
        # CLOSED at epoch 21: the first durable act of resume() is the
        # seat-acquisition CAS in reconstruct_resume_state — exactly one of
        # two racing resumes commits it; the loser is refused with zero
        # mutation. Check-then-act at THIS guard is therefore acceptable —
        # the leadership CAS is the arbiter (design §B.3).
        guarded_run_id = resume_point.checkpoint.run_id
        run_status, status_check = check_run_status_resumable(self._db, guarded_run_id)
        if not status_check.can_resume:
            if run_status is not None and run_status in _IMMUTABLE_SUCCESS_RUN_STATUSES:
                refusal_reason = f"Run is terminal (status {run_status.value!r}); successful terminal runs are immutable"
            else:
                refusal_reason = status_check.reason or f"Run status {run_status!r} precludes resume"
            raise NonResumableRunError(guarded_run_id, refusal_reason)

        # ---- resume() entry guard, part 2: checkpoint currency + topology ----
        # (elspeth-5129406607) RecoveryManager.get_resume_point() is ADVISORY
        # like can_resume() — a caller may hand-build a ResumePoint — so
        # re-verify at the enforcing boundary that (a) the supplied checkpoint
        # is the run's LATEST resume baseline (resuming a superseded one would
        # replay work the run has already progressed past) and (b) the stored
        # latest checkpoint's recorded topology matches the graph this resume
        # runs under (one run_id = one configuration). The topology check must
        # use the database-loaded latest checkpoint, not caller-supplied
        # checkpoint fields from a hand-built ResumePoint. Both are READ-ONLY
        # refusals fired before the first mutation (prepare_for_run /
        # rebase_sequence / the seat CAS in reconstruct_resume_state),
        # mirroring the status guard above. A format-incompatible checkpoint
        # row (IncompatibleCheckpointError from get_latest_checkpoint)
        # propagates as-is — structured, fail-closed.
        if self._checkpoint_manager is None:
            raise OrchestrationInvariantError(
                "CheckpointManager is required for resume - Orchestrator must be initialized with checkpoint_manager"
            )
        latest_checkpoint = self._checkpoint_manager.get_latest_checkpoint(guarded_run_id)
        if latest_checkpoint is None:
            raise NonResumableRunError(
                guarded_run_id,
                "run has no checkpoint rows; the supplied resume point cannot be validated as the run's resume baseline",
            )
        if (
            latest_checkpoint.checkpoint_id != resume_point.checkpoint.checkpoint_id
            or latest_checkpoint.sequence_number != resume_point.sequence_number
        ):
            raise NonResumableRunError(
                guarded_run_id,
                f"supplied checkpoint '{resume_point.checkpoint.checkpoint_id}' (sequence {resume_point.sequence_number}) "
                f"is not the run's latest resume point '{latest_checkpoint.checkpoint_id}' "
                f"(sequence {latest_checkpoint.sequence_number})",
            )
        topology_check = CheckpointCompatibilityValidator().validate(latest_checkpoint, graph)
        if not topology_check.can_resume:
            raise NonResumableRunError(
                guarded_run_id,
                topology_check.reason or "checkpoint topology is incompatible with the current execution graph",
            )

        # ADR-010 §Decision 3: freeze both registries at bootstrap, mirroring
        # run(). Recovery happens in a new process — the module import chain
        # registers PassThroughDeclarationContract, but without this call the
        # registries are never frozen, leaving a window where
        # register_declaration_contract() could succeed post-bootstrap on the
        # resume path.
        prepare_for_run()

        self._checkpoints.rebase_sequence(resume_point.sequence_number)
        # §A.1: mint this process's single-use worker identity; the takeover
        # CAS inside reconstruct_resume_state registers it as the new leader
        # and returns the fencing token on ResumeState.
        resume_worker_id = mint_worker_id(resume_point.checkpoint.run_id)
        state = self.reconstruct_resume_state(resume_point, payload_store, worker_id=resume_worker_id)
        run_id = state.run_id
        factory = state.factory
        coordination_token = state.coordination_token
        # acquire_run_leadership always returns a token (or raises), so
        # coordination_token is never None at this point.  Assert here to
        # narrow the type for the heartbeat thread constructor (which requires
        # a non-optional CoordinationToken) and to surface bugs early.
        if coordination_token is None:
            raise OrchestrationInvariantError(
                f"Resume for run '{state.run_id}': coordination_token is None after "
                "reconstruct_resume_state — acquire_run_leadership must always return "
                "a token or raise; a None result is an orchestration invariant violation."
            )
        # Thread the token to the collaborators the slice-2 step-4 fences
        # consume (checkpoint writes, finalize, ceremonies).
        self._checkpoints.bind_coordination(coordination_token)
        schema_contracts_by_source = state.schema_contracts_by_source
        unprocessed_rows = state.unprocessed_rows
        # F1 fix: pre-computed by _reconstruct_resume_state; forwarded to the loop.
        incomplete_by_row = state.incomplete_by_row
        recovery_manager = state.recovery_manager
        resume_checkpoint_id = resume_point.checkpoint.checkpoint_id
        resume_start_time = time.perf_counter()

        # ADR-030 §A.3 (slice 4): start the dedicated heartbeat thread AFTER
        # the seat is minted (coordination_token is live) and the token is
        # bound, BEFORE the try/except block.  Mirrors the run() path in
        # core.py.  The thread beats both the run_workers row and the
        # run_coordination seat in ONE BEGIN IMMEDIATE transaction so the two
        # liveness clocks cannot skew.
        #
        # Correct sequencing (design §A.3 "joined before release_seat"):
        # stop() is called as the FIRST statement in every except handler that
        # calls release_seat AND in the success path just before release_seat.
        # The finally block additionally calls stop() as a safety net
        # (idempotent) to cover any exit path that did not already stop.
        _heartbeat = RunHeartbeatThread(
            factory.run_coordination,
            token=coordination_token,
        )
        _heartbeat.start()

        # 5. Process unprocessed rows (with graceful shutdown support)

        # When shutdown_event is provided (testing), skip signal handler
        # installation and use the caller's event directly.
        shutdown_ctx = nullcontext(shutdown_event) if shutdown_event is not None else shutdown_handler_context()
        resume_failure_counter_baseline: ExecutionCounters | None = None

        try:
            incomplete_sources = {
                state.source_names_by_source[source_node_id]: lifecycle_state
                for source_node_id, lifecycle_state in state.source_lifecycle_by_source.items()
                if lifecycle_state not in _SOURCE_COMPLETE_LIFECYCLE_STATES
            }
            if incomplete_sources:
                raise IncompleteSourceResumeError(run_id, incomplete_sources)

            if unprocessed_rows or state.has_restored_barrier_work:
                resume_failure_counter_baseline = _derive_resume_failure_counter_baseline(factory, run_id)

            # F1 QUIESCENCE GATE (co-repointed with the buffered-token
            # exclusion, Task 3.2): journal BLOCKED barrier rows are excluded
            # from ``unprocessed_rows`` because they are RESTORED at processor
            # construction — so a fully-buffered crashed run (all remaining
            # work sitting at barriers) legitimately has zero unprocessed
            # rows. Early-completing here would finalize the run and delete
            # checkpoints WITHOUT ever constructing the
            # BarrierJournalRestoreContext, silently dropping the buffered
            # batch. The no-work arm therefore also requires the journal to
            # carry no restored barrier work.
            if not unprocessed_rows and not state.has_restored_barrier_work:
                factory.data_flow.sweep_deferred_invariants_or_crash(run_id)

                # All rows were processed - complete the run.
                #
                # Phase 2.2 (elspeth-0de989c56d): the resume's local counters
                # are 0 here because nothing was reprocessed, but the audit DB
                # carries the truth.  Aggregate token_outcomes to derive the
                # correct four-value terminal status and feed it to both the
                # Landscape finalize and the local RunResult.
                return self._finalize_successful_resume(
                    factory=factory,
                    run_id=run_id,
                    coordination_token=coordination_token,
                    heartbeat=_heartbeat,
                    duration_seconds=0.0,
                )

            with shutdown_ctx as active_event:
                # F1: bundle the journal-restore inputs (checkpoint scalars +
                # batch remap) for the processor's construction-time restore
                # sweep (BarrierRecoveryCoordinator.restore_from_journal).
                barrier_restore = BarrierJournalRestoreContext(
                    resume_checkpoint_id=resume_checkpoint_id,
                    barrier_scalars=resume_point.barrier_scalars,
                    batch_id_remap=state.batch_id_remap,
                )
                # Return value (the loop's resume-local counters) is deliberately
                # unused: finalization derives everything from the audit trail
                # below (single bookkeeper, elspeth-7294de558e).
                self.process_resumed_rows(
                    factory=factory,
                    run_id=run_id,
                    config=config,
                    graph=graph,
                    unprocessed_rows=unprocessed_rows,
                    barrier_restore=barrier_restore,
                    settings=settings,
                    payload_store=payload_store,
                    incomplete_by_row=incomplete_by_row,
                    recovery_manager=recovery_manager,
                    resume_checkpoint_id=resume_checkpoint_id,
                    schema_contracts_by_source=schema_contracts_by_source,
                    shutdown_event=active_event,
                    coordination_token=coordination_token,
                    check_coordination_latch=_heartbeat.check_and_raise,
                )

            # 6. Complete the run with reproducibility grade
            # SUCCESS PATH: Must be inside try block so RunFinished is emitted
            # BEFORE the finally block flushes telemetry to exporters.
            # Fix: elspeth-rapid-sg0q — previously this was after the finally block,
            # meaning RunFinished was emitted after telemetry flush (never exported).
            #
            # F2 (resume-fork-reemit) — UNIFY both resume branches on the audit
            # trail.  This with-unprocessed-rows branch previously derived its
            # terminal status + counters from the resume loop's *local* counters
            # (only what THIS resume call reprocessed), so a resumed run's
            # RunResult disagreed field-for-field with an uninterrupted run
            # (e.g. a resumed 1-row 2-branch fork reported rows_succeeded=1,
            # rows_forked=0 instead of the cumulative 2, 1) — while the
            # no-unprocessed-rows branch already reconstructed cumulative
            # counters from token_outcomes.  Both branches now finalize from the
            # SAME audit-derived cumulative (status, counters).
            #
            # ORDERING: this runs AFTER _process_resumed_rows returned, i.e.
            # after run_resume_processing_loop's end-of-source aggregation /
            # coalesce flushes, after flush_and_write_sinks recorded sink
            # diversions, and after sweep_deferred_invariants_or_crash — so every
            # outcome this resume wrote is committed and visible to the derive
            # query.  Deriving before those flushes commit would undercount.
            #
            # SINGLE BOOKKEEPER (elspeth-7294de558e): every counter field —
            # including rows_coalesce_failed — is now derived from the durable
            # audit trail. rows_coalesce_failed historically had no audit arm
            # (the coalesce-operation roll-up went to telemetry only) and was
            # GRAFTED here from the resume loop's live `result` counter, which
            # only saw failures during THIS resume's re-drive and forgot run-1
            # failures consumed before the interrupt. derive() now reconstructs
            # it cumulatively from the FAILED node_states _fail_pending writes
            # at the run's coalesce nodes (one barrier == one DISTINCT
            # (node, row) pair; see QueryRepository.count_failed_coalesce_
            # barrier_rows), so the audit-derived value IS the final value and
            # the terminal status is already a pure function of the final
            # counters — same construction as the uninterrupted path. Parity
            # pin: test_adr_019_resume_counter_parity.py::
            # test_resume_derives_rows_coalesce_failed_from_durable_audit
            # (resumed run_B == uninterrupted oracle run_A).
            return self._finalize_successful_resume(
                factory=factory,
                run_id=run_id,
                coordination_token=coordination_token,
                heartbeat=_heartbeat,
                duration_seconds=time.perf_counter() - resume_start_time,
            )
        except GracefulShutdownError as shutdown_exc:
            # ADR-030 §A.3: stop the heartbeat thread before the seat is released.
            _heartbeat.stop()
            with best_effort("Interrupted ceremony on resume graceful shutdown", run_id=run_id):
                self._ceremony.emit_interrupted_ceremony(run_id, factory, shutdown_exc, resume_start_time, token=coordination_token)
                # Seat hygiene: only AFTER the INTERRUPTED finalize succeeded
                # (same best_effort block); without this a failed resume's
                # seat wedges retries for the liveness window.
                if coordination_token is not None:
                    factory.run_coordination.release_seat(token=coordination_token, now=datetime.now(UTC))
            raise  # Propagate to CLI
        except _RunFailedWithPartialResultError as failed_exc:
            # ADR-030 §A.3: stop the heartbeat thread before the seat is released.
            _heartbeat.stop()
            failed_result = _resume_failure_result_from_baseline(
                run_id,
                baseline=resume_failure_counter_baseline,
                partial_result=failed_exc.partial_result,
            )
            with best_effort("Partial-result failure ceremony on resume", run_id=run_id):
                self._ceremony.emit_failed_ceremony(
                    run_id,
                    factory,
                    resume_start_time,
                    failed_result,
                    token=coordination_token,
                )
                # Seat hygiene: after the FAILED finalize succeeded.
                if coordination_token is not None:
                    factory.run_coordination.release_seat(token=coordination_token, now=datetime.now(UTC))
            raise failed_exc.original_error.with_traceback(failed_exc.original_traceback) from None
        except Exception:
            # Finalize as FAILED to prevent the run from being stuck in RUNNING
            # permanently (which blocks future resume attempts). The outer broad-except
            # is justified — any unhandled exception during resume needs ceremony.
            # ADR-030 §A.3: stop the heartbeat thread before the seat is released.
            _heartbeat.stop()
            with best_effort("Generic failure ceremony on resume", run_id=run_id):
                self._ceremony.emit_failed_ceremony(run_id, factory, resume_start_time, token=coordination_token)
                # Seat hygiene: after the FAILED finalize succeeded.
                if coordination_token is not None:
                    factory.run_coordination.release_seat(token=coordination_token, now=datetime.now(UTC))
            raise
        finally:
            # ADR-030 §A.3: safety-net stop (idempotent) — covers any exit
            # path that did not already stop the thread (e.g. an exception
            # raised before any except handler ran release_seat).
            _heartbeat.stop()
            self._ceremony.safe_flush_telemetry()

    def process_resumed_rows(
        self,
        factory: RecorderFactory,
        run_id: str,
        config: PipelineConfig,
        graph: ExecutionGraph,
        unprocessed_rows: Sequence[ResumedRow],
        barrier_restore: BarrierJournalRestoreContext | None,
        settings: ElspethSettings | None = None,
        *,
        payload_store: PayloadStore,
        incomplete_by_row: Mapping[str, Sequence[IncompleteTokenSpec]],
        recovery_manager: RecoveryManager,
        resume_checkpoint_id: str,
        schema_contracts_by_source: Mapping[NodeID, SchemaContract],
        shutdown_event: threading.Event | None = None,
        coordination_token: CoordinationToken | None = None,
        check_coordination_latch: Callable[[], None] | None = None,
    ) -> RunResult:
        """Process unprocessed rows during resume.

        Mirrors _execute_run() structure but with resume-specific divergences
        documented in the accounting block below. Returns RunStatus.RUNNING —
        the public resume() wrapper transitions to COMPLETED after finalize_run().
        """
        # ─────────────────────────────────────────────────────────────────
        # Divergence accounting: _process_resumed_rows vs _execute_run
        #
        # Source on_start():       Skipped (include_source_on_start=False)
        # Graph registration:     Loads from DB (setup_resume_context)
        # Quarantine routing:     Not applicable (rows already validated)
        # Field resolution:       Skipped (loaded from DB in original run)
        # Schema contract:        Skipped (passed via parameter)
        # operation_id lifecycle: Not applicable (no source track_operation)
        # Progress emission:      None (known gap — T24 follow-up)
        # Checkpointing:          Same post-sink + shutdown semantics as run()
        # ─────────────────────────────────────────────────────────────────

        self._checkpoints.set_active_graph(graph)
        try:
            # 1. Setup (loads graph artifacts from original run's DB records)
            artifacts = setup_resume_context(factory, run_id, config, graph)

            # 2. Initialize context + processor (source on_start skipped)
            run_ctx = self._context_factory.initialize_run_context(
                factory,
                run_id,
                config,
                graph,
                settings,
                artifacts,
                payload_store,
                include_source_on_start=False,
                barrier_restore=barrier_restore,
                shutdown_event=shutdown_event,
                coordination_token=coordination_token,
            )

            # ADR-025 §3: schema contracts are plural-by-source on resume.
            # ``ctx.contract`` is set per-row inside the resume loop via the
            # per-source lookup; the previous singular write here was a dead
            # assignment that the loop's per-row reassignment overwrote on
            # the first row anyway. ``run_transform_runtime_preflights``
            # does not read ``ctx.contract`` (it only sets ``ctx.node_id``
            # per-transform). Setting ``ctx.contract`` to None here is not
            # required — the field carries the prior run's value through
            # nullcontext if no rows are reprocessed, which is irrelevant
            # because the early-exit path skips this method entirely.
            preflight_retry_manager = RetryManager(RuntimeRetryConfig.from_settings(settings.retry)) if settings is not None else None
            try:
                run_transform_runtime_preflights(
                    factory,
                    run_id,
                    config,
                    run_ctx.ctx,
                    retry_manager=preflight_retry_manager,
                    shutdown_event=shutdown_event,
                )
            except BaseException:
                cleanup_plugins(config, run_ctx.ctx, include_source=False)
                raise

            loop_ctx = LoopContext(
                counters=ExecutionCounters(),
                pending_tokens={name: [] for name in config.sinks},
                processor=run_ctx.processor,
                ctx=run_ctx.ctx,
                config=config,
                agg_transform_lookup=run_ctx.agg_transform_lookup,
                coalesce_executor=run_ctx.coalesce_executor,
                coalesce_node_map=run_ctx.coalesce_node_map,
            )
            source_on_success_by_source: dict[NodeID, str] = {}
            for source_name, source_id in artifacts.source_id_map.items():
                source_on_success = config.sources[source_name].on_success
                if source_on_success is None:
                    raise OrchestrationInvariantError(
                        f"Cannot resume rows from source {source_name!r}: source on_success routing is missing."
                    )
                source_on_success_by_source[source_id] = source_on_success

            try:
                # 3. Process loop (resume path)
                interrupted = run_resume_processing_loop(
                    loop_ctx,
                    unprocessed_rows,
                    incomplete_by_row=incomplete_by_row,
                    recovery_manager=recovery_manager,
                    payload_store=payload_store,
                    run_id=run_id,
                    resume_checkpoint_id=resume_checkpoint_id,
                    schema_contracts_by_source=schema_contracts_by_source,
                    source_on_success_by_source=source_on_success_by_source,
                    shutdown_event=shutdown_event,
                    check_coordination_latch=check_coordination_latch,
                )

                # 4. Flush + write sinks with checkpoint advancement
                self._sink_flush.flush_and_write_sinks(
                    factory,
                    run_id,
                    loop_ctx,
                    artifacts.sink_id_map,
                    artifacts.edge_map,
                    interrupted,
                    on_token_written_factory=self._checkpoints.make_checkpoint_after_sink_factory(run_id, run_ctx.processor),
                    scheduler_terminalizer=run_ctx.processor,
                )

                # ADR-019 Phase 4: resumed row processing reaches stable I1a/I1b
                # postconditions only after resume sink writes finish.
                factory.data_flow.sweep_deferred_invariants_or_crash(run_id)
            except GracefulShutdownError:
                raise
            except Exception as exc:
                raise _RunFailedWithPartialResultError(
                    original_error=exc,
                    partial_result=loop_ctx.counters.to_run_result(run_id, status=RunStatus.FAILED),
                ) from exc

            finally:
                cleanup_plugins(config, run_ctx.ctx, include_source=False)

            return loop_ctx.counters.to_run_result(run_id, status=RunStatus.RUNNING)
        finally:
            self._checkpoints.set_active_graph(None)


def handle_incomplete_batches(
    execution: ExecutionRepository,
    run_id: str,
) -> dict[str, str]:
    """Find and handle incomplete batches for recovery.

    - EXECUTING batches: Mark as failed (crash interrupted), then retry
    - FAILED batches: Retry with incremented attempt
    - DRAFT batches: Leave as-is (collection continues)

    Args:
        execution: ExecutionRepository for database operations
        run_id: Run being recovered

    Returns:
        Mapping of old_batch_id to new_batch_id for retried batches.
        Callers must use this to rebind batch_ids in restored checkpoint
        state so that resumed execution references the retry batches,
        not the dead originals.
    """
    from elspeth.contracts.enums import BatchStatus

    incomplete = execution.get_incomplete_batches(run_id)
    batch_id_mapping: dict[str, str] = {}

    for batch in incomplete:
        if batch.status == BatchStatus.EXECUTING:
            # Crash interrupted mid-execution, mark failed then retry
            execution.update_batch_status(batch.batch_id, BatchStatus.FAILED)
            retry = execution.retry_batch(batch.batch_id)
            batch_id_mapping[batch.batch_id] = retry.batch_id
        elif batch.status == BatchStatus.FAILED:
            # Previous failure, retry
            retry = execution.retry_batch(batch.batch_id)
            batch_id_mapping[batch.batch_id] = retry.batch_id
        # DRAFT batches continue normally (collection resumes)

    return batch_id_mapping
