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
from collections.abc import Mapping, Sequence
from contextlib import nullcontext
from datetime import UTC, datetime
from types import SimpleNamespace
from typing import TYPE_CHECKING, Any

from elspeth.contracts import PipelineRow, ResumedRow, RunStatus
from elspeth.contracts.config import RuntimeRetryConfig
from elspeth.contracts.errors import (
    AuditIntegrityError,
    EmptyResumeStateError,
    GracefulShutdownError,
    IncompleteSourceResumeError,
    OrchestrationInvariantError,
)
from elspeth.contracts.events import RunFinished, RunSummary
from elspeth.contracts.run_result import derive_terminal_run_status
from elspeth.contracts.runtime_val_manifest import build_runtime_val_manifest
from elspeth.contracts.types import NodeID
from elspeth.core.canonical import canonical_json
from elspeth.core.checkpoint.recovery import NonResumableRunError, check_run_status_resumable
from elspeth.core.landscape.factory import RecorderFactory

# The immutable-success family (COMPLETED / COMPLETED_WITH_FAILURES / EMPTY)
# is deliberately imported from its single source of truth rather than
# duplicated: the resume() entry guard defers these statuses to the durable
# run-immutability guard in RunLifecycleRepository.update_run_status().
from elspeth.core.landscape.run_lifecycle_repository import _IMMUTABLE_SUCCESS_RUN_STATUSES
from elspeth.core.landscape.schema import RunSourceLifecycleState
from elspeth.engine._best_effort import best_effort
from elspeth.engine.orchestrator.aggregation import (
    check_aggregation_timeouts,
    flush_remaining_aggregation_buffers,
    handle_incomplete_batches,
)
from elspeth.engine.orchestrator.cleanup import cleanup_plugins
from elspeth.engine.orchestrator.export import reconstruct_schema_from_json
from elspeth.engine.orchestrator.outcomes import (
    accumulate_row_outcomes,
    flush_coalesce_pending,
    handle_coalesce_timeouts,
)
from elspeth.engine.orchestrator.run_status import (
    cli_completion_for,
    derive_resume_terminal_status_from_audit,
)
from elspeth.engine.orchestrator.runtime_preflight import run_transform_runtime_preflights
from elspeth.engine.orchestrator.shutdown import shutdown_handler_context
from elspeth.engine.orchestrator.types import (
    ExecutionCounters,
    GraphArtifacts,
    LoopContext,
    ResumeState,
)
from elspeth.engine.orchestrator.validation import (
    validate_route_destinations,
    validate_sink_failsink_destinations,
    validate_source_quarantine_destination,
    validate_transform_error_sinks,
)
from elspeth.engine.processor import BarrierJournalRestoreContext
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
    from elspeth.engine.orchestrator.ceremony import RunCeremony
    from elspeth.engine.orchestrator.checkpointing import CheckpointCoordinator
    from elspeth.engine.orchestrator.run_core import RunExecutionCore
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
    # Get explicit node ID mappings from graph. Resume must preserve every
    # source root from the original multi-source DAG; a singleton
    # ``graph.get_source()`` lookup is intentionally single-source-only.
    source_id_map: dict[str, NodeID] = {}
    for candidate_source_id in graph.get_sources():
        source_info = graph.get_node_info(candidate_source_id)
        if "source_name" not in source_info.config:
            raise OrchestrationInvariantError(
                f"DAG source node '{source_info.node_id}' is missing 'source_name' in its config. "
                f"Per ADR-025 §2 the DAG builder MUST set source_name on every source node. "
                f"This is a graph-construction bug — node config keys: {sorted(source_info.config.keys())}."
            )
        source_id_map[str(source_info.config["source_name"])] = candidate_source_id
    source_id = next(iter(source_id_map.values()))
    sink_id_map = graph.get_sink_id_map()
    transform_id_map = graph.get_transform_id_map()
    config_gate_id_map = graph.get_config_gate_id_map()
    coalesce_id_map = graph.get_coalesce_id_map()

    # Build edge_map from database (load real edge IDs registered in original run)
    # CRITICAL: Must use real edge_ids for FK integrity when recording routing events
    # Convert keys from (str, str) to (NodeID, str) to match RowProcessor's type
    raw_edge_map = factory.data_flow.get_edge_map(run_id)
    edge_map: dict[tuple[NodeID, str], str] = {(NodeID(k[0]), k[1]): v for k, v in raw_edge_map.items()}

    # Get route resolution map for validation
    route_resolution_map = graph.get_route_resolution_map()

    # Validate route destinations (config may have changed since original run)
    # This catches config errors early instead of after partial processing
    # Call module function directly (no wrapper method)
    validate_route_destinations(
        route_resolution_map=route_resolution_map,
        available_sinks=set(config.sinks.keys()),
        transform_id_map=transform_id_map,
        transforms=config.transforms,
        config_gate_id_map=config_gate_id_map,
        config_gates=config.gates,
    )

    # Validate transform error sink destinations
    # Call module function directly (no wrapper method)
    validate_transform_error_sinks(
        transforms=config.transforms,
        available_sinks=set(config.sinks.keys()),
    )

    # Validate source quarantine destinations
    # Call module function directly (no wrapper method)
    for source in config.sources.values():
        validate_source_quarantine_destination(
            source=source,
            available_sinks=set(config.sinks.keys()),
        )

    # Validate sink failsink destinations
    sink_validation_stubs = {name: SimpleNamespace(on_write_failure=sink._on_write_failure) for name, sink in config.sinks.items()}
    sink_plugins = {name: sink.name for name, sink in config.sinks.items()}
    validate_sink_failsink_destinations(
        sink_configs=sink_validation_stubs,
        available_sinks=set(config.sinks.keys()),
        sink_plugins=sink_plugins,
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
        # incomplete_by_row ⊆ unprocessed_rows by construction (both queries share
        # _DELEGATION_PATHS and "incomplete non-delegation token" is Case 2 of
        # get_unprocessed_rows), so every partial-fork/expand/coalesce row IS visited
        # by this loop and its specs are found here.
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
        # GRACEFUL SHUTDOWN CHECK
        # Check between row iterations — current row is fully
        # processed, outcomes recorded, safe to stop here.
        # No quarantine path in resume (rows already validated).
        # ─────────────────────────────────────────────────────────────
        if shutdown_event is not None and shutdown_event.is_set():
            interrupted_by_shutdown = True
            break

    if not interrupted_by_shutdown:
        # CRITICAL: Flush remaining aggregation buffers only at true end-of-source.
        if config.aggregation_settings:
            # Call module function directly (no wrapper method)
            flush_result = flush_remaining_aggregation_buffers(
                config=config,
                processor=processor,
                ctx=ctx,
                pending_tokens=pending_tokens,
            )
            counters.accumulate_flush_result(flush_result)

            # TERMINAL GUARANTEE: same assertion as _post_source_iteration_work.
            for agg_node_id_str in config.aggregation_settings:
                remaining = processor.get_aggregation_buffer_count(NodeID(agg_node_id_str))
                if remaining > 0:
                    raise OrchestrationInvariantError(
                        f"Aggregation buffer for node '{agg_node_id_str}' still has "
                        f"{remaining} tokens after end-of-source flush. "
                        f"These tokens would never reach a terminal state."
                    )

        # Flush pending coalesce operations only when resume processing exhausted all rows.
        if coalesce_executor is not None:
            flush_coalesce_pending(
                coalesce_executor=coalesce_executor,
                coalesce_node_map=coalesce_node_map,
                processor=processor,
                ctx=ctx,
                counters=counters,
                pending_tokens=pending_tokens,
            )

        if processor.has_unresolved_scheduler_work():
            active_work = "; ".join(processor.summarize_unresolved_scheduler_work()) or "<unknown>"
            raise OrchestrationInvariantError(
                f"Resume for run '{processor.run_id}' left non-terminal scheduler work after end-of-source flush. "
                "Blocked scheduler state must be recovered explicitly before run completion. "
                f"Active scheduler work: {active_work}."
            )

    return interrupted_by_shutdown


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
        run_core: RunExecutionCore,
        checkpoint_manager: CheckpointManager | None,
    ) -> None:
        self._db = db
        self._events = events
        self._ceremony = ceremony
        self._checkpoints = checkpoints
        self._run_core = run_core
        self._checkpoint_manager = checkpoint_manager

    def reconstruct_resume_state(
        self,
        resume_point: ResumePoint,
        payload_store: PayloadStore,
    ) -> ResumeState:
        """Reconstruct state needed to process resumed rows.

        Creates a fresh factory, handles incomplete batches, restores aggregation state,
        deserializes the source schema for type fidelity, validates the schema contract,
        and retrieves unprocessed rows from the payload store.

        Args:
            resume_point: ResumePoint from RecoveryManager.get_resume_point()
            payload_store: PayloadStore for retrieving row data

        Returns:
            ResumeState with all reconstruction results.

        Raises:
            ValueError: If checkpoint_manager is not initialized.
            OrchestrationInvariantError: If schema contract is missing from audit trail.
        """
        run_id = resume_point.checkpoint.run_id

        # Create fresh factory (stateless, like run())
        # Pass payload_store for external call payload persistence
        factory = RecorderFactory(self._db, payload_store=payload_store)

        # Validate resumability before mutating the run header or batch state.
        # Incomplete-source refusal is an operator-facing "start fresh or use a
        # source-aware resume path" outcome; it must not strand the run as
        # RUNNING or rewrite retry batches merely because the operator probed
        # resume.
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

        unprocessed_rows: Sequence[ResumedRow]
        schema_contracts_by_source: dict[NodeID, SchemaContract]

        # ADR-025 §3 Decision 5 (G6): schema contracts are plural-by-source
        # and live exclusively in ``run_sources``. The legacy single-source
        # fallback that read ``runs.schema_contract_json`` was deleted along
        # with the column itself — readers and writers are now symmetric on
        # ``run_sources``. ``verify_contract_integrity`` (called via
        # ``can_resume`` → ``get_resume_point`` before this method runs)
        # already raises ``EmptyResumeStateError`` when ``run_sources`` is
        # empty, so by the time we land here every declared source has a
        # contract record. We still assert the postcondition defensively
        # against future call-path changes: an empty map at resume time is
        # Tier-1 audit corruption.
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

        # F1 fix: pre-compute incomplete child tokens so the resume loop can dispatch
        # partial-fork/expand/coalesce rows via mid-DAG continuation rather than
        # whole-row restart (which would re-emit already-completed branches).
        incomplete_by_row = recovery.get_incomplete_tokens_by_row(run_id)

        source_records = factory.run_lifecycle.get_run_source_resume_records(run_id)
        source_schema_classes: dict[NodeID, type[Any]] = {}
        schema_contracts_by_source = {}
        source_names_by_source: dict[NodeID, str] = {}
        source_lifecycle_by_source: dict[NodeID, str] = {}
        for raw_source_node_id, source_record in source_records.items():
            source_node_id = NodeID(str(raw_source_node_id))
            schema_dict = json.loads(source_record.source_schema_json)
            source_schema_classes[source_node_id] = reconstruct_schema_from_json(schema_dict)
            schema_contracts_by_source[source_node_id] = source_record.schema_contract
            source_names_by_source[source_node_id] = str(source_record.source_name)
            source_lifecycle_by_source[source_node_id] = str(source_record.lifecycle_state)

        unprocessed_rows = recovery.get_unprocessed_row_data_by_source(
            run_id,
            payload_store,
            source_schema_classes=source_schema_classes,
        )

        # 1. Handle incomplete batches - call module function directly.
        # The returned old→retry batch_id mapping feeds the processor's
        # journal-based barrier restore (BUFFERED token_outcomes still carry
        # the dead original batch ids after a flush-interrupting crash),
        # threaded through ResumeState.
        batch_id_remap = handle_incomplete_batches(factory.execution, run_id)

        # 2. Update run status to running after validation has succeeded.
        factory.run_lifecycle.update_run_status(run_id, RunStatus.RUNNING)

        # 3. F1: barrier restore runs in PROCESSOR CONSTRUCTION — resume()
        # bundles a BarrierJournalRestoreContext (checkpoint scalars + the
        # batch_id_remap captured above) and RowProcessor.__init__ rebuilds
        # the executors from journal BLOCKED rows + audit tables. The
        # quiescence gate in resume() therefore consults the JOURNAL: a run
        # whose remaining work all sits at barriers has zero unprocessed rows
        # but must still run the processing path so the restored buffers flush.
        has_restored_barrier_work = recovery.count_blocked_barrier_items(run_id) > 0

        return ResumeState(
            factory=factory,
            run_id=run_id,
            unprocessed_rows=unprocessed_rows,
            incomplete_by_row=incomplete_by_row,
            recovery_manager=recovery,
            schema_contracts_by_source=schema_contracts_by_source,
            source_names_by_source=source_names_by_source,
            source_lifecycle_by_source=source_lifecycle_by_source,
            has_restored_barrier_work=has_restored_barrier_work,
            batch_id_remap=batch_id_remap,
        )

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
        # Deferred import: core.py imports ResumeCoordinator from this module, so a
        # module-level import here would create a cycle. These two symbols stay in
        # core.py because the normal-run path also uses them.
        from elspeth.engine.orchestrator.core import _RunFailedWithPartialResultError, prepare_for_run

        if payload_store is None:
            raise OrchestrationInvariantError("payload_store is required for resume - row data must be retrieved from stored payloads")

        # ---- resume() entry guard (elspeth-2f23292372, operator option b) ----
        # resume() historically trusted callers to honor the ADVISORY
        # RecoveryManager.can_resume() and never re-checked run status itself,
        # so a competing resume against a RUNNING run was ADMITTED. Re-check
        # here via the SAME shared implementation can_resume() uses, BEFORE
        # the first mutation (rebase_sequence below, then
        # reconstruct_resume_state's batch + run-header writes).
        #
        # The immutable-success family (COMPLETED / COMPLETED_WITH_FAILURES /
        # EMPTY) is deliberately NOT intercepted here: the run-immutability
        # guard in RunLifecycleRepository.update_run_status() already refuses
        # those durably with AuditIntegrityError ("Successful terminal runs
        # are immutable"), and that refusal is the pinned loser-after-winner
        # contract (tests/e2e/recovery/test_concurrent_resume.py). This guard
        # closes the caller-convention gap for everything else — RUNNING
        # above all, plus runs that do not exist.
        #
        # KNOWN RESIDUAL (TOCTOU): two resumes can BOTH observe FAILED here
        # before either flips the run to RUNNING in reconstruct_resume_state;
        # closing that check-then-act window requires cross-process
        # coordination (operator option c — a separate post-F1 effort,
        # deliberately not attempted here). The immutability guard remains
        # the durable backstop for the completed half of that window.
        guarded_run_id = resume_point.checkpoint.run_id
        run_status, status_check = check_run_status_resumable(self._db, guarded_run_id)
        if not status_check.can_resume and run_status not in _IMMUTABLE_SUCCESS_RUN_STATUSES:
            raise NonResumableRunError(
                guarded_run_id,
                status_check.reason or f"Run status {run_status!r} precludes resume",
            )

        # ADR-010 §Decision 3: freeze both registries at bootstrap, mirroring
        # run(). Recovery happens in a new process — the module import chain
        # registers PassThroughDeclarationContract, but without this call the
        # registries are never frozen, leaving a window where
        # register_declaration_contract() could succeed post-bootstrap on the
        # resume path.
        prepare_for_run()

        self._checkpoints.rebase_sequence(resume_point.sequence_number)
        state = self.reconstruct_resume_state(resume_point, payload_store)
        run_id = state.run_id
        factory = state.factory
        schema_contracts_by_source = state.schema_contracts_by_source
        unprocessed_rows = state.unprocessed_rows
        # F1 fix: pre-computed by _reconstruct_resume_state; forwarded to the loop.
        incomplete_by_row = state.incomplete_by_row
        recovery_manager = state.recovery_manager
        resume_checkpoint_id = resume_point.checkpoint.checkpoint_id
        resume_start_time = time.perf_counter()

        # 5. Process unprocessed rows (with graceful shutdown support)

        # When shutdown_event is provided (testing), skip signal handler
        # installation and use the caller's event directly.
        shutdown_ctx = nullcontext(shutdown_event) if shutdown_event is not None else shutdown_handler_context()

        try:
            incomplete_sources = {
                state.source_names_by_source[source_node_id]: lifecycle_state
                for source_node_id, lifecycle_state in state.source_lifecycle_by_source.items()
                if lifecycle_state not in _SOURCE_COMPLETE_LIFECYCLE_STATES
            }
            if incomplete_sources:
                raise IncompleteSourceResumeError(run_id, incomplete_sources)

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
                terminal_status, audit_counters = derive_resume_terminal_status_from_audit(factory, run_id)
                factory.run_lifecycle.finalize_run(run_id, status=terminal_status)

                # Emit RunFinished telemetry (matching the normal completion path)
                self._ceremony.emit_telemetry(
                    RunFinished(
                        timestamp=datetime.now(UTC),
                        run_id=run_id,
                        status=terminal_status,
                        row_count=audit_counters.rows_processed,
                        duration_ms=0.0,
                    )
                )

                # Emit RunSummary event
                cli_status, exit_code = cli_completion_for(terminal_status)
                self._events.emit(
                    RunSummary(
                        run_id=run_id,
                        status=cli_status,
                        total_rows=audit_counters.rows_processed,
                        succeeded=audit_counters.rows_succeeded,
                        failed=audit_counters.rows_failed,
                        quarantined=audit_counters.rows_quarantined,
                        duration_seconds=0.0,
                        exit_code=exit_code,
                        routed_success=audit_counters.rows_routed_success,
                        routed_failure=audit_counters.rows_routed_failure,
                        routed_destinations=tuple(audit_counters.routed_destinations.items()),
                    )
                )

                # Delete checkpoints on successful completion
                self._checkpoints.delete_checkpoints(run_id)

                return audit_counters.to_run_result(run_id, terminal_status)

            with shutdown_ctx as active_event:
                # F1: bundle the journal-restore inputs (checkpoint scalars +
                # batch remap) for the processor's construction-time restore
                # sweep (RowProcessor._restore_barriers_from_journal).
                barrier_restore = BarrierJournalRestoreContext(
                    resume_checkpoint_id=resume_checkpoint_id,
                    barrier_scalars=resume_point.barrier_scalars,
                    batch_id_remap=state.batch_id_remap,
                )
                result = self.process_resumed_rows(
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
            # coalesce flushes, after _flush_and_write_sinks recorded sink
            # diversions, and after sweep_deferred_invariants_or_crash — so every
            # outcome this resume wrote is committed and visible to the derive
            # query.  Deriving before those flushes commit would undercount.
            # The status returned here is computed from the audit-only counters
            # (rows_coalesce_failed == 0, see graft below); it is intentionally
            # discarded and RECOMPUTED post-graft so terminal_status stays a pure
            # function of the FINAL reconciled counters — the same set the
            # uninterrupted path derives from. This is correctness-by-symmetry,
            # NOT crash-prevention: a coalesce failure always co-increments
            # rows_failed (outcomes.py flush_coalesce_pending does both, and the
            # consumed branches land in audit as UNROUTED), so derive's audit-only
            # rows_failed is already > 0 → failure_indicator already True → the
            # pre-graft status is already COMPLETED_WITH_FAILURES, never COMPLETED.
            # So grafting rows_coalesce_failed does not flip the status in any
            # real flow and the RunResult biconditional's COMPLETED+failure arm
            # cannot fire here. Recomputing is still the honest, future-proof
            # construction (status derived from the same final counters as the
            # uninterrupted path) and guards against a future counter whose graft
            # WOULD be status-bearing.
            _audit_only_status, audit_counters = derive_resume_terminal_status_from_audit(factory, run_id)

            # F2 — per-field "best available source" graft.
            #
            # Each counter field is taken from its best available source. For
            # the 11 fields the audit trail records per-token, that source is
            # derive_resume_terminal_status_from_audit (cumulative, queryable
            # from token_outcomes). rows_coalesce_failed is the LONE field the
            # audit trail does NOT record: a failed coalesce records per-branch
            # FAILURE/UNROUTED outcomes (so rows_failed reconstructs) but the
            # coalesce-operation roll-up is emitted to TELEMETRY ONLY
            # (outcomes.py flush_coalesce_pending → _emit_failed_coalesce_telemetry;
            # there is no queryable token_outcomes signal for it). derive()
            # therefore has no match arm for it and returns 0. For the with-rows
            # branch its best source is the live re-drive counter captured by
            # flush_coalesce_pending into the resume loop's `result`, so graft it
            # back over the audit-derived 0.
            #
            # NOTE (scope + reachability): this recovers only coalesce failures
            # that occurred DURING THIS RESUME's re-drive. Coalesce failures from
            # run-1 (before the interrupt) were live-counter-only — never
            # persisted as a queryable signal — and are unrecoverable here.
            # Making rows_coalesce_failed reconstructable cumulatively (incl.
            # run-1) is a schema/epoch change tracked as an operator follow-up;
            # the no-rows branch already ships the 0 for the same structural
            # reason.
            #
            # This graft is a LIVE REGRESSION FIX, not future-proofing. The
            # during-re-drive coalesce failure is CONFIRMED reachable: the resume
            # loop calls handle_coalesce_timeouts → CoalesceExecutor.check_timeouts
            # PER-ROW (resume.py:268-276), and a coalesce that times out before
            # quorum during re-drive increments rows_coalesce_failed
            # (outcomes.py:447/462, "quorum_not_met_at_timeout") in the live
            # `result`. Pre-F2 that count was reported; F2 (pre-graft) discarded it
            # by replacing `result` with audit-derived counters; this graft
            # restores it. End-to-end regression test (with observed removed-graft
            # red / restored-graft green):
            # test_adr_019_resume_counter_parity.py::
            #   test_resume_grafts_rows_coalesce_failed_from_timeout_redrive.
            #
            # The other increment site — flush_pending (end-of-source,
            # "incomplete_branches") — does NOT produce a during-re-drive failure
            # in deterministic flows (lost branches fail immediately via
            # notify_branch_lost without touching this counter; buffered branches
            # are restored-to-completion by restore_from_checkpoint; an unproduced
            # coalesce branch is DAG-rejected), so on that path the graft copies
            # 0-over-0 and is a no-op. The timeout path is where it earns its keep.
            audit_counters.rows_coalesce_failed = result.rows_coalesce_failed

            # Recompute terminal_status from the final reconciled counters (now
            # carrying the grafted rows_coalesce_failed) via the pure L0 function
            # — NOT by re-calling derive_resume_terminal_status_from_audit, which
            # would re-query token_outcomes and silently re-zero the graft.
            terminal_status = derive_terminal_run_status(
                rows_processed=audit_counters.rows_processed,
                rows_succeeded=audit_counters.rows_succeeded,
                rows_failed=audit_counters.rows_failed,
                rows_routed_success=audit_counters.rows_routed_success,
                rows_routed_failure=audit_counters.rows_routed_failure,
                rows_quarantined=audit_counters.rows_quarantined,
                rows_coalesce_failed=audit_counters.rows_coalesce_failed,
            )

            factory.run_lifecycle.finalize_run(run_id, status=terminal_status)
            result = audit_counters.to_run_result(run_id, terminal_status)

            # 7. Emit RunFinished telemetry
            resume_duration_ms = (time.perf_counter() - resume_start_time) * 1000
            self._ceremony.emit_telemetry(
                RunFinished(
                    timestamp=datetime.now(UTC),
                    run_id=run_id,
                    status=terminal_status,
                    row_count=result.rows_processed,
                    duration_ms=resume_duration_ms,
                )
            )

            # 8. Emit RunSummary event
            cli_status, exit_code = cli_completion_for(terminal_status)
            total_duration = time.perf_counter() - resume_start_time
            self._events.emit(
                RunSummary(
                    run_id=run_id,
                    status=cli_status,
                    total_rows=result.rows_processed,
                    succeeded=result.rows_succeeded,
                    failed=result.rows_failed,
                    quarantined=result.rows_quarantined,
                    duration_seconds=total_duration,
                    exit_code=exit_code,
                    routed_success=result.rows_routed_success,
                    routed_failure=result.rows_routed_failure,
                    routed_destinations=tuple(result.routed_destinations.items()),
                )
            )

            # 9. Delete checkpoints on successful completion
            self._checkpoints.delete_checkpoints(run_id)

            return result
        except GracefulShutdownError as shutdown_exc:
            with best_effort("Interrupted ceremony on resume graceful shutdown", run_id=run_id):
                self._ceremony.emit_interrupted_ceremony(run_id, factory, shutdown_exc, resume_start_time)
            raise  # Propagate to CLI
        except _RunFailedWithPartialResultError as failed_exc:
            with best_effort("Partial-result failure ceremony on resume", run_id=run_id):
                self._ceremony.emit_failed_ceremony(
                    run_id,
                    factory,
                    resume_start_time,
                    failed_exc.partial_result,
                )
            raise failed_exc.original_error.with_traceback(failed_exc.original_traceback) from None
        except Exception:
            # Finalize as FAILED to prevent the run from being stuck in RUNNING
            # permanently (which blocks future resume attempts). The outer broad-except
            # is justified — any unhandled exception during resume needs ceremony.
            with best_effort("Generic failure ceremony on resume", run_id=run_id):
                self._ceremony.emit_failed_ceremony(run_id, factory, resume_start_time)
            raise
        finally:
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

        # 1. Setup (loads graph artifacts from original run's DB records)
        artifacts = setup_resume_context(factory, run_id, config, graph)

        # 2. Initialize context + processor (source on_start skipped)
        run_ctx = self._run_core.initialize_run_context(
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
        run_transform_runtime_preflights(
            factory,
            run_id,
            config,
            run_ctx.ctx,
            retry_manager=preflight_retry_manager,
            shutdown_event=shutdown_event,
        )

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
                source_on_success_by_source={
                    source_id: config.sources[source_name].on_success for source_name, source_id in artifacts.source_id_map.items()
                },
                shutdown_event=shutdown_event,
            )

            # 4. Flush + write sinks with checkpoint advancement
            self._run_core.flush_and_write_sinks(
                factory,
                run_id,
                loop_ctx,
                artifacts.sink_id_map,
                artifacts.edge_map,
                interrupted,
                on_token_written_factory=self._checkpoints.make_checkpoint_after_sink_factory(run_id, run_ctx.processor),
            )

            # ADR-019 Phase 4: resumed row processing reaches stable I1a/I1b
            # postconditions only after resume sink writes finish.
            factory.data_flow.sweep_deferred_invariants_or_crash(run_id)
        except GracefulShutdownError:
            raise
        except Exception as exc:
            # Deferred import: breaks the core.py <-> resume.py cycle (see reconstruct_resume_state).
            from elspeth.engine.orchestrator.core import _RunFailedWithPartialResultError

            raise _RunFailedWithPartialResultError(
                original_error=exc,
                partial_result=loop_ctx.counters.to_run_result(run_id, status=RunStatus.FAILED),
            ) from exc

        finally:
            cleanup_plugins(config, run_ctx.ctx, include_source=False)

        self._checkpoints.set_active_graph(None)
        return loop_ctx.counters.to_run_result(run_id, status=RunStatus.RUNNING)
