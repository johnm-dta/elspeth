"""RowProcessor: Orchestrates row processing through pipeline.

Coordinates:
- Token creation
- Transform execution
- Gate evaluation (config-driven)
- Aggregation handling
- Final outcome recording
"""

from __future__ import annotations

import logging
import uuid
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, replace
from datetime import UTC, datetime
from types import MappingProxyType
from typing import TYPE_CHECKING, Any, cast

from elspeth.contracts import RouteDestination, RowResult, SourceRow, TokenInfo, TransformResult
from elspeth.contracts.audit import TokenRef
from elspeth.contracts.audit_evidence import AuditEvidenceBase
from elspeth.contracts.barrier_scalars import AggregationNodeScalars, BarrierScalars, CoalescePendingScalars
from elspeth.contracts.coordination import DEFAULT_ITEM_STALL_BUDGET_SECONDS
from elspeth.contracts.freeze import deep_freeze
from elspeth.contracts.schema_contract import PipelineRow
from elspeth.contracts.types import BranchName, CoalesceName, NodeID, SinkName, StepResolver
from elspeth.engine._best_effort import best_effort
from elspeth.engine._error_hash import compute_error_hash
from elspeth.engine.dag_navigator import DAGNavigator, WorkItem

if TYPE_CHECKING:
    # TypeIs (PEP 742) lives in typing on 3.13 but only typing_extensions on
    # 3.12; it is used solely in an annotation (lazy under `from __future__
    # import annotations`), so a type-checking-only import works on both.
    from typing import TypeIs

    from sqlalchemy.engine import Connection

    from elspeth.contracts import Batch
    from elspeth.contracts.audit import Row as AuditRow
    from elspeth.contracts.audit import Token as AuditToken
    from elspeth.contracts.coordination import CoordinationToken
    from elspeth.contracts.events import TelemetryEvent
    from elspeth.contracts.payload_store import PayloadStore
    from elspeth.engine.clock import Clock
    from elspeth.engine.coalesce_executor import CoalesceExecutor, CoalesceOutcome
    from elspeth.engine.executors import GateOutcome
    from elspeth.engine.orchestrator.types import RowPlugin, TelemetryManagerProtocol

from elspeth.contracts import BatchTransformProtocol, SourceProtocol, TransformProtocol
from elspeth.contracts.declaration_contracts import (
    AggregateDeclarationContractViolation,
    BatchFlushInputs,
    BatchFlushOutputs,
    BoundaryInputs,
    BoundaryOutputs,
    DeclarationContractViolation,
)
from elspeth.contracts.enums import (
    BatchStatus,
    NodeStateStatus,
    OutputMode,
    RoutingKind,
    RoutingMode,
    TerminalOutcome,
    TerminalPath,
    TriggerType,
)
from elspeth.contracts.errors import (
    AuditIntegrityError,
    CapacityError,
    ExecutionError,
    FrameworkBugError,
    MaxRetriesExceeded,
    OrchestrationInvariantError,
    PassThroughContractViolation,
    PluginContractViolation,
    PluginRetryableError,
    RunWorkerEvictedError,
    SchedulerLeaseLostError,
    TransformErrorCategory,
    TransformErrorReason,
)
from elspeth.contracts.plugin_context import PluginContext
from elspeth.contracts.results import FailureInfo
from elspeth.contracts.scheduler import (
    BarrierEmission,
    BatchMembershipSpec,
    BranchLossSpec,
    BufferedOutcomeSpec,
    TokenWorkItem,
    TokenWorkStatus,
)
from elspeth.core.checkpoint.recovery import IncompleteTokenSpec
from elspeth.core.config import AggregationSettings, GateSettings
from elspeth.core.landscape._helpers import generate_id
from elspeth.core.landscape.data_flow_repository import DataFlowRepository
from elspeth.core.landscape.errors import LandscapeRecordError
from elspeth.core.landscape.execution_repository import ExecutionRepository
from elspeth.core.landscape.run_coordination_repository import RunCoordinationRepository
from elspeth.core.landscape.scheduler_repository import (
    TokenSchedulerRepository,
    token_from_journal_item,
)
from elspeth.engine.clock import DEFAULT_CLOCK
from elspeth.engine.executors import (
    AggregationExecutor,
    GateExecutor,
    TransformExecutor,
)
from elspeth.engine.executors.can_drop_rows import verify_zero_emission_declaration_path
from elspeth.engine.executors.declaration_dispatch import run_batch_flush_checks, run_boundary_checks
from elspeth.engine.executors.transform import record_transform_error_with_routing
from elspeth.engine.retry import RetryManager
from elspeth.engine.spans import SpanFactory
from elspeth.engine.tokens import TokenManager

# Iteration guard to prevent infinite loops from bugs.
# This counts dequeued work items, so it must exceed the largest supported
# single-row fan-out; otherwise legal expansion trips the safety valve before
# the final child runs.
MAX_WORK_QUEUE_ITERATIONS = 100_000
SCHEDULER_MAINTENANCE_INTERVAL = 64
logger = logging.getLogger(__name__)

type _SourceBoundaryFailure = (
    DeclarationContractViolation
    | AggregateDeclarationContractViolation
    | PluginContractViolation
    | FrameworkBugError
    | OrchestrationInvariantError
)


@dataclass(frozen=True, slots=True)
class DAGTraversalContext:
    """Precomputed DAG traversal data for the processor. Built by orchestrator.

    All dict fields are stored as MappingProxyType to enforce true
    immutability — frozen=True only prevents attribute reassignment,
    not mutation of mutable values held by those attributes.
    """

    node_step_map: Mapping[NodeID, int]
    node_to_plugin: Mapping[NodeID, RowPlugin | GateSettings]
    node_to_next: Mapping[NodeID, NodeID | None]
    coalesce_node_map: Mapping[CoalesceName, NodeID]
    branch_first_node: Mapping[str, NodeID] = MappingProxyType({})
    # Explicit allowlist of non-plugin traversal nodes (sources, queues,
    # coalesce points). Nodes in node_to_next that are neither plugin-bearing
    # nor in this set fail closed at resolution instead of being skipped.
    # Coalesce nodes are structural by definition and always unioned in.
    structural_node_ids: frozenset[NodeID] = frozenset()

    def __post_init__(self) -> None:
        object.__setattr__(self, "node_step_map", deep_freeze(self.node_step_map))
        object.__setattr__(self, "node_to_plugin", deep_freeze(self.node_to_plugin))
        object.__setattr__(self, "node_to_next", deep_freeze(self.node_to_next))
        object.__setattr__(self, "coalesce_node_map", deep_freeze(self.coalesce_node_map))
        object.__setattr__(self, "branch_first_node", deep_freeze(self.branch_first_node))
        object.__setattr__(
            self,
            "structural_node_ids",
            frozenset(self.structural_node_ids) | frozenset(self.coalesce_node_map.values()),
        )


@dataclass(frozen=True, slots=True)
class _AggregationRestorePlan:
    """Derived restore inputs for one aggregation node (journal restore).

    Built during the derivation phase of ``_restore_barriers_from_journal``
    so every audit read completes (and can raise) before any executor
    mutation runs.
    """

    node_id: NodeID
    items: Sequence[TokenWorkItem]
    member_order: Sequence[str]
    batch_id: str | None
    accepted_count_total: int
    completed_flush_count: int
    scalars: AggregationNodeScalars

    def __post_init__(self) -> None:
        object.__setattr__(self, "items", deep_freeze(self.items))
        object.__setattr__(self, "member_order", deep_freeze(self.member_order))


@dataclass(frozen=True, slots=True)
class _LiveBarrierHold:
    """In-memory companion of one durable BLOCKED barrier hold (ADR-030 §E.2).

    Stashed by the processor at the moment a claimed token is about to block
    at a barrier (aggregation buffering / coalesce hold) and consumed by the
    next drain iteration's journal-first intake: the LIVE token preserves the
    exact post-transform payload and resume provenance the old in-claim accept
    used (N=1 parity). Inherited rows with no stash entry (leader takeover)
    fall back to journal rehydration with audit-derived attempt offsets —
    the same semantics as the restore path.
    """

    token: TokenInfo
    barrier_key: str


@dataclass(frozen=True, slots=True)
class BarrierJournalRestoreContext:
    """Resume inputs for the journal-based barrier restore (F1 design D3).

    Built by the resume path (``ResumeCoordinator``) and handed to
    ``RowProcessor.__init__``; its presence IS the resume signal — a normal
    run passes ``None`` and no restore sweep runs.

    Attributes:
        resume_checkpoint_id: Checkpoint id stamped on every journal-restored
            token (resume provenance).
        barrier_scalars: Underivable scalar barrier metadata from the
            checkpoint row (trigger latches / lost_branches). ``None`` means
            the checkpoint carried no scalars — every node restores with
            unlatched / no-losses defaults. The scalars snapshot is
            NON-transactional vs the journal (D3 staleness model): an absent
            entry always means not-fired / re-derivable, never corruption.
        batch_id_remap: old->retry batch_id mapping returned by
            ``handle_incomplete_batches`` — BUFFERED token_outcomes still
            reference the dead original batch ids, so the restored in-progress
            batch id must be read through this remap.
    """

    resume_checkpoint_id: str
    barrier_scalars: BarrierScalars | None
    batch_id_remap: Mapping[str, str]

    def __post_init__(self) -> None:
        if not self.resume_checkpoint_id:
            raise ValueError("BarrierJournalRestoreContext.resume_checkpoint_id must not be empty")
        object.__setattr__(self, "batch_id_remap", deep_freeze(self.batch_id_remap))


@dataclass(frozen=True, slots=True)
class _FlushContext:
    """Parametric context for aggregation flush handling.

    Captures the differences between timeout/end-of-source flushes
    (handle_timeout_flush) and count-triggered flushes
    (_process_batch_aggregation_node) so shared helpers can handle both.

    Parametric differences:
    - error_msg: "...during timeout flush" vs "Batch transform failed"
    - expand_parent_token: buffered_tokens[0] (timeout) vs current_token (count)
    - triggering_token: None (timeout) vs current_token (count)
    - coalesce info: derived from tokens (timeout) vs passed from WorkItem (count)
    - CONSUMED_IN_BATCH recording: not needed (timeout) vs needed for triggering token (count)
    """

    node_id: NodeID
    transform: TransformProtocol
    settings: AggregationSettings
    buffered_tokens: tuple[TokenInfo, ...]
    batch_id: str
    error_msg: str
    expand_parent_token: TokenInfo
    triggering_token: TokenInfo | None
    coalesce_node_id: NodeID | None
    coalesce_name: CoalesceName | None

    def __post_init__(self) -> None:
        if not self.node_id:
            raise ValueError("_FlushContext.node_id must not be empty")
        # Freeze before validation so emptiness check works on generators too
        object.__setattr__(self, "buffered_tokens", tuple(self.buffered_tokens))
        if not self.buffered_tokens:
            raise ValueError("_FlushContext.buffered_tokens must not be empty")
        if not self.batch_id:
            raise ValueError("_FlushContext.batch_id must not be empty")
        # coalesce_node_id and coalesce_name must be both-or-neither
        has_id = self.coalesce_node_id is not None
        has_name = self.coalesce_name is not None
        if has_id != has_name:
            raise ValueError(
                f"_FlushContext: coalesce_node_id and coalesce_name must be both set or both None, "
                f"got node_id={self.coalesce_node_id!r}, name={self.coalesce_name!r}"
            )


def _validated_quarantined_indices(result: TransformResult, *, buffered_token_count: int, aggregation_name: str) -> set[int]:
    """Extract and validate batch-transform quarantine metadata."""
    if result.success_reason is None or "metadata" not in result.success_reason:
        return set()

    metadata = result.success_reason["metadata"]
    if type(metadata) is not dict:
        raise OrchestrationInvariantError(
            f"Aggregation {aggregation_name!r} returned success_reason.metadata={metadata!r}; "
            f"expected dict when quarantine metadata is present"
        )
    if "quarantined_indices" not in metadata:
        return set()

    raw_indices = metadata["quarantined_indices"]
    if type(raw_indices) is not list:
        raise OrchestrationInvariantError(
            f"Aggregation {aggregation_name!r} returned quarantined_indices={raw_indices!r}; expected list[int]"
        )

    quarantined_index_set: set[int] = set()
    for position, raw_index in enumerate(raw_indices):
        if type(raw_index) is not int:
            raise OrchestrationInvariantError(
                f"Aggregation {aggregation_name!r} returned quarantined_indices[{position}]={raw_index!r}; expected int"
            )
        if raw_index < 0 or raw_index >= buffered_token_count:
            raise OrchestrationInvariantError(
                f"Aggregation {aggregation_name!r} returned quarantined_indices[{position}]={raw_index}; "
                f"valid index range is 0..{buffered_token_count - 1}"
            )
        quarantined_index_set.add(raw_index)
    return quarantined_index_set


# --- Discriminated union types for _process_single_token extraction ---


@dataclass(frozen=True, slots=True)
class _TransformContinue:
    """Token should advance to the next node in the DAG."""

    updated_token: TokenInfo
    updated_sink: str


@dataclass(frozen=True, slots=True)
class _TransformTerminal:
    """Token has reached a terminal state (completed, failed, quarantined, etc.)."""

    result: RowResult | tuple[RowResult, ...]


type _TransformOutcome = _TransformContinue | _TransformTerminal


@dataclass(frozen=True, slots=True)
class _GateContinue:
    """Gate says advance to next node (or jump to a specific node)."""

    updated_token: TokenInfo
    updated_sink: str
    next_node_id: NodeID | None = None  # None = next structural node


@dataclass(frozen=True, slots=True)
class _GateTerminal:
    """Gate has routed, forked, or diverted the token to a terminal state."""

    result: RowResult | tuple[RowResult, ...]


type _GateOutcome = _GateContinue | _GateTerminal


def make_step_resolver(
    node_step_map: Mapping[NodeID, int],
    source_node_id: NodeID,
) -> StepResolver:
    """Create a StepResolver from a precomputed step map.

    Single source of truth for audit step resolution. Used by both RowProcessor
    (for its internal executors) and the orchestrator (for CoalesceExecutor and
    its TokenManager, which are constructed before the processor).

    Resolution order:
    1. Known node in step_map → return mapped step
    2. Source node (not in map) → return 0
    3. Unknown node → raise OrchestrationInvariantError
    """
    # Defensive copy so callers can't mutate the map after creation
    _map = dict(node_step_map)
    _source = source_node_id

    def resolve(node_id: NodeID) -> int:
        if node_id in _map:
            return _map[node_id]
        if node_id == _source:
            return 0
        raise OrchestrationInvariantError(f"Node ID '{node_id}' missing from traversal step map")

    return resolve


def _is_result_tuple(result: RowResult | tuple[RowResult, ...]) -> TypeIs[tuple[RowResult, ...]]:
    """Narrow a scheduler result to the buffered-tuple shape.

    A ``TypeIs`` guard (PEP 742) so mypy narrows BOTH branches: callers that
    fall through (or take the ``else`` of a ternary) get ``RowResult``, not the
    union. ``type(result) is tuple`` rather than ``isinstance`` because the
    discrimination is between a frozen ``RowResult`` dataclass and a concrete
    aggregation-buffer ``tuple`` — an exact-type test, not a subtype check.
    """
    return type(result) is tuple


class RowProcessor:
    """Processes rows through the DAG-defined pipeline topology.

    Processing follows the DAG topology built from explicit input/on_success
    connections. Transforms, gates, and aggregations are interleaved per their
    declared wiring — there is no fixed "transforms first, then gates" order.

    Handles:
    1. Creating initial tokens from source rows
    2. Executing transforms, gates, and aggregations per DAG traversal order
    3. Routing tokens to sinks or downstream processing nodes
    4. Recording final outcomes via Landscape audit trail

    Example:
        processor = RowProcessor(
            execution, data_flow, span_factory, run_id, source_node_id,
            traversal=traversal_context,
        )

        result = processor.process_row(
            row_index=0,
            source_row_index=0,
            ingest_sequence=0,
            source_row=SourceRow.valid({"value": 42}, contract=contract, source_row_index=0),
            transforms=[transform1, transform2],
            ctx=ctx,
        )
    """

    def __init__(
        self,
        execution: ExecutionRepository,
        data_flow: DataFlowRepository,
        span_factory: SpanFactory,
        run_id: str,
        source_node_id: NodeID,
        *,
        source_on_success: str,
        source_plugin: SourceProtocol | None = None,
        edge_map: dict[tuple[NodeID, str], str] | None = None,
        route_resolution_map: dict[tuple[NodeID, str], RouteDestination] | None = None,
        traversal: DAGTraversalContext,
        aggregation_settings: dict[NodeID, AggregationSettings] | None = None,
        retry_manager: RetryManager | None = None,
        coalesce_executor: CoalesceExecutor | None = None,
        branch_to_coalesce: dict[BranchName, CoalesceName] | None = None,
        branch_to_sink: dict[BranchName, SinkName] | None = None,
        sink_names: frozenset[str] | None = None,
        coalesce_on_success_map: dict[CoalesceName, str] | None = None,
        barrier_restore: BarrierJournalRestoreContext | None = None,
        payload_store: PayloadStore | None = None,
        clock: Clock | None = None,
        max_workers: int | None = None,
        telemetry_manager: TelemetryManagerProtocol | None = None,
        scheduler: TokenSchedulerRepository,
        scheduler_lease_owner: str | None = None,
        scheduler_lease_seconds: int = 300,
        scheduler_heartbeat_seconds: int = 60,
        coordination_token: CoordinationToken | None = None,
        run_coordination: RunCoordinationRepository | None = None,
        follower_barrier_node_ids: frozenset[NodeID] | None = None,
    ) -> None:
        """Initialize processor.

        Args:
            execution: Execution repository for node states, routing, operations
            data_flow: Data flow repository for token outcomes, schema contracts
            span_factory: Span factory for tracing
            run_id: Current run ID
            source_node_id: Source node ID
            source_on_success: Source's on_success sink name for COMPLETED routing
            source_plugin: Optional source plugin instance. Production
                orchestrator passes the concrete source so source-boundary
                contracts can evaluate runtime declarations after token creation.
            edge_map: Map of (node_id, label) -> edge_id
            route_resolution_map: Map of (node_id, label) -> resolved route destination
            traversal: Precomputed DAG traversal context from orchestrator
            aggregation_settings: Map of node_id -> AggregationSettings for trigger evaluation
            retry_manager: Optional retry manager for transform execution
            coalesce_executor: Optional coalesce executor for fork/join operations
            branch_to_coalesce: Map of branch_name -> coalesce_name for fork/join routing
            sink_names: Set of valid sink names for route resolution validation.
                If None, sink validation on jump-target resolution is skipped.
            coalesce_on_success_map: Map of coalesce_name -> terminal sink_name
                for COALESCED outcomes produced at terminal coalesce points
            barrier_restore: Resume-only inputs for the journal-based barrier
                restore (F1). Non-None means this processor is being built on
                the resume path: barrier buffers are rebuilt from journal
                BLOCKED rows + audit tables before the first row is processed.
            payload_store: Optional PayloadStore for persisting source row payloads
            clock: Optional clock for time access. Defaults to system clock.
                   Inject MockClock for deterministic testing.
            max_workers: Maximum concurrent workers for transform execution (None = no limit)
            telemetry_manager: Optional TelemetryManager for emitting telemetry events.
                               If None, telemetry emission is disabled.
            scheduler: Durable token scheduler. Every continuation is persisted,
                leased, and terminally marked as it advances.
            coordination_token: Leader fencing token (epoch 21, ADR-030).
                Carried by value for the slice-2 step-4 fenced verbs this
                processor drives (repair sweep, recover_expired_leases, the
                fenced ingest verb). When provided, the orchestrator also
                passes ``scheduler_lease_owner=token.worker_id`` — the §A.1
                registered worker identity IS the lease owner; the
                ``row-processor:{run_id}:{uuid}`` mint below remains the
                fallback for direct repository-level construction.
            run_coordination: Optional RunCoordinationRepository for leader
                housekeeping (§C.2 path 1, slice 4): enumerating dead non-leader
                workers and calling ``evict_worker`` for each. None = no
                housekeeping sweep (N=1 without the coordination substrate, or
                direct repository-level construction in tests).
            follower_barrier_node_ids: ADR-030 §B (slice 5, follower aggregation
                barrier hand-off): frozenset of aggregation node IDs that this
                follower must NOT execute locally.  When a batch-aware transform
                is encountered at one of these node IDs, the processor returns
                (None, []) which triggers mark_blocked so the leader's next
                journal-intake adopts the arrival and runs trigger evaluation
                (§B.2: trigger evaluation is leader-only).  Non-follower
                processors leave this None (no-op path).
        """
        if scheduler is None:
            raise TypeError("scheduler repository is required; RowProcessor no longer supports the legacy in-memory drain")

        self._execution = execution
        self._data_flow = data_flow
        self._spans = span_factory
        self._run_id = run_id
        self._source_node_id: NodeID = source_node_id
        self._source_on_success: str = source_on_success
        self._traversal = traversal
        self._node_step_map: Mapping[NodeID, int] = traversal.node_step_map
        self._step_resolver: StepResolver = make_step_resolver(traversal.node_step_map, source_node_id)
        self._node_to_plugin: Mapping[NodeID, RowPlugin | GateSettings] = traversal.node_to_plugin
        # Traversal metadata intentionally excludes the source node. Callers
        # that want source-boundary checks must pass the concrete source plugin.
        self._source_plugin: SourceProtocol | None = source_plugin
        self._node_to_next: Mapping[NodeID, NodeID | None] = traversal.node_to_next
        self._retry_manager = retry_manager
        self._coalesce_executor = coalesce_executor
        self._coalesce_node_ids: dict[CoalesceName, NodeID] = dict(traversal.coalesce_node_map)
        self._coalesce_name_by_node_id: dict[NodeID, CoalesceName] = {
            node_id: coalesce_name for coalesce_name, node_id in self._coalesce_node_ids.items()
        }
        # Explicit allowlist from the traversal context (elspeth-c522931bd1):
        # never derived as the complement of node_to_plugin, which silently
        # skipped plugin nodes missing from the mapping (fail-open).
        self._structural_node_ids: frozenset[NodeID] = traversal.structural_node_ids
        self._branch_to_coalesce: dict[BranchName, CoalesceName] = branch_to_coalesce or {}
        self._branch_to_sink: dict[BranchName, SinkName] = branch_to_sink or {}
        overlap = set(self._branch_to_coalesce.keys()) & set(self._branch_to_sink.keys())
        if overlap:
            raise OrchestrationInvariantError(
                f"Branch names {sorted(overlap)} appear in both branch_to_coalesce and branch_to_sink. "
                "A fork branch must route to EITHER a coalesce node OR a direct sink, not both."
            )
        self._sink_names: frozenset[str] = sink_names or frozenset()
        self._coalesce_on_success_map: dict[CoalesceName, str] = coalesce_on_success_map or {}
        self._aggregation_settings: dict[NodeID, AggregationSettings] = aggregation_settings or {}
        # ADR-030 §B (slice 5): aggregation node IDs that a follower must NOT
        # execute locally — these are barrier nodes whose trigger evaluation is
        # leader-only.  Empty frozenset on any non-follower processor (no-op).
        self._follower_barrier_node_ids: frozenset[NodeID] = follower_barrier_node_ids or frozenset()
        self._clock = clock if clock is not None else DEFAULT_CLOCK

        # DAG navigator: pure topology queries extracted from RowProcessor
        self._nav = DAGNavigator(
            node_to_plugin=self._node_to_plugin,
            node_to_next=self._node_to_next,
            coalesce_node_ids=self._coalesce_node_ids,
            structural_node_ids=self._structural_node_ids,
            coalesce_name_by_node_id=self._coalesce_name_by_node_id,
            coalesce_on_success_map=self._coalesce_on_success_map,
            sink_names=self._sink_names,
            branch_first_node=dict(traversal.branch_first_node),
        )

        # Build error edge map: transform node_id -> DIVERT edge_id.
        # Scans edge_map for __error_{name}__ labels (created by dag.py for transforms
        # with on_error pointing to a real sink, not "discard").
        _edge_map = edge_map or {}
        error_edge_ids: dict[NodeID, str] = {}
        for (node_id, label), edge_id in _edge_map.items():
            if label.startswith("__error_") and label.endswith("__"):
                error_edge_ids[node_id] = edge_id
        self._error_edge_ids = error_edge_ids

        self._token_manager = TokenManager(
            data_flow,
            step_resolver=self._step_resolver,
        )
        self._transform_executor = TransformExecutor(
            execution,
            span_factory,
            self._step_resolver,
            max_workers=max_workers,
            error_edge_ids=error_edge_ids,
            data_flow=data_flow,
        )
        self._gate_executor = GateExecutor(execution, span_factory, self._step_resolver, edge_map, route_resolution_map)
        self._aggregation_executor = AggregationExecutor(
            execution,
            span_factory,
            self._step_resolver,
            run_id,
            aggregation_settings=aggregation_settings,
            clock=self._clock,
        )
        self._telemetry_manager = telemetry_manager
        self._scheduler = scheduler
        self._coordination_token = coordination_token
        self._run_coordination = run_coordination
        # ADR-030 §G (slice 5): _scheduler_lease_owner_registered is True when
        # the lease owner is a run_workers identity. Production paths pass the
        # owner explicitly; direct tests often pass only the coordination token,
        # whose worker_id is the same registered leader identity.
        resolved_scheduler_lease_owner = scheduler_lease_owner
        if resolved_scheduler_lease_owner is None and coordination_token is not None:
            resolved_scheduler_lease_owner = coordination_token.worker_id
        self._scheduler_lease_owner_registered: bool = resolved_scheduler_lease_owner is not None
        self._scheduler_lease_owner = resolved_scheduler_lease_owner or f"row-processor:{run_id}:{uuid.uuid4().hex}"
        self._scheduler_lease_seconds = scheduler_lease_seconds
        if scheduler_heartbeat_seconds <= 0:
            raise OrchestrationInvariantError(f"scheduler_heartbeat_seconds must be positive, got {scheduler_heartbeat_seconds}")
        if scheduler_heartbeat_seconds >= scheduler_lease_seconds:
            raise OrchestrationInvariantError(
                f"scheduler_heartbeat_seconds ({scheduler_heartbeat_seconds}) must be less than "
                f"scheduler_lease_seconds ({scheduler_lease_seconds}); otherwise the heartbeat "
                "cannot refresh the lease before it expires under any slow-work scenario."
            )
        self._scheduler_heartbeat_seconds = scheduler_heartbeat_seconds
        # Active scheduler claim state for in-loop heartbeat refresh
        # (ADR-026 RC6 multi-worker, filigree elspeth-ddde8144b6). These fields
        # are non-None only inside ``_drain_scheduler_claims`` between
        # ``claim_ready``/``claim_pending_sink`` and the terminal ``mark_*``.
        # ``_process_single_token`` calls ``_heartbeat_active_claim`` on each
        # node-iteration boundary so an alive-but-slow worker's lease does not
        # expire under a peer reaper. RowProcessor is single-threaded per row,
        # so this instance state has no concurrent access.
        self._active_claim_work_item_id: str | None = None
        self._last_heartbeat_at: datetime | None = None
        self._scheduler_drains_since_maintenance = 0

        # ADR-030 §E.2 (slice 3, journal-first barrier acceptance): live-token
        # stash keyed by token_id. `_process_batch_aggregation_node` and
        # `_maybe_coalesce_token` deposit the CURRENT (post-transform) token
        # plus its barrier_key the moment they decide to block; the drain
        # reads the barrier_key for mark_blocked and the next iteration's
        # intake (`_run_barrier_intake_pass`) pops the token to feed executor
        # memory with exact N=1 parity. Entries for rows this process never
        # adopts (lease lost to a peer mid-claim) linger harmlessly for the
        # processor's lifetime — bounded by run size.
        # RowProcessor is single-threaded per row, so no concurrent access.
        self._live_barrier_holds: dict[str, _LiveBarrierHold] = {}
        # §E.5 record-then-notify: BranchLossSpecs accumulated by
        # `_notify_coalesce_of_lost_branch` during the current claim/flush;
        # consumed by the disposition that commits the branch's terminal state
        # (the drain's mark_failed/mark_pending_sink/mark_terminal, or the
        # flush's complete_barrier) so the durable loss record rides the SAME
        # transaction as the disposition.
        self._pending_branch_losses: list[BranchLossSpec] = []

        # F1 (THE RESTORE INVERSION): on resume, barrier buffers are rebuilt
        # FROM journal BLOCKED rows + audit tables. The old direction —
        # materializing journal rows from checkpoint blobs — is gone; the
        # checkpoint contributes only scalars (via barrier_restore).
        # The checkpoint id is retained for re-drive provenance: a PENDING_SINK
        # re-drive whose token already has node_states (the original run's
        # crashed sink attempt) must stamp resume_attempt_offset/-checkpoint_id
        # so the re-driven sink node_state does not collide at attempt 0.
        self._resume_checkpoint_id: str | None = barrier_restore.resume_checkpoint_id if barrier_restore is not None else None
        if barrier_restore is not None:
            self._restore_barriers_from_journal(barrier_restore)

    @property
    def token_manager(self) -> TokenManager:
        """Expose token manager for orchestrator to create tokens for quarantined rows."""
        return self._token_manager

    @property
    def coordination_token(self) -> CoordinationToken | None:
        """The leader fencing token bound at construction (ADR-030).

        Exposed so source-iteration helpers (quarantine ingest) can thread it
        into their own fenced rows writes. None only for direct
        repository-level construction (the unfenced legacy arm).
        """
        return self._coordination_token

    def _restore_barriers_from_journal(self, restore: BarrierJournalRestoreContext) -> None:
        """Rebuild aggregation buffers and coalesce pendings from journal BLOCKED rows.

        F1 resume path. The journal (token_work_items BLOCKED rows with a
        non-NULL barrier_key) is authoritative for buffered/held token
        payloads; counters, batch membership, hold state ids and attempt
        offsets derive from audit tables; the checkpoint contributes only
        the underivable scalars (``restore.barrier_scalars``).

        Discipline: ALL derivations complete (and raise, if they must) before
        any executor restore call mutates state. The coalesce restore is a
        single all-or-nothing call (its contract — a second call would discard
        the first); aggregation restores run per node afterwards, each
        internally validate-before-mutate.

        BLOCKED rows manufactured by the pre-epoch-20 blob-materialization
        restore path lacked ``barrier_blocked_at``; that writer is deleted and
        the epoch-20 delete-the-DB policy retired its rows, so the executors'
        NULL-means-corruption assert is honest — no live DB carries them.

        Raises:
            AuditIntegrityError: On any journal/audit disagreement (orphan
                barrier_key, missing/foreign batch ids, membership mismatch,
                NULL barrier_blocked_at, missing hold state ...).
        """
        now = self._clock.now_utc()
        # ADR-030 §E.4 belt: one run-wide duplicate-acceptance sweep at restore
        # entry. token_outcomes has NO non-terminal uniqueness — the adoption
        # CAS is the structural guard; >1 live BUFFERED rows for a token means
        # a deposed leader's unfenced intake wrote a second acceptance.
        duplicate_acceptances = self._data_flow.find_duplicate_live_buffered_outcomes(self._run_id)
        if duplicate_acceptances:
            details = ", ".join(f"{token_id} ({count} live BUFFERED)" for token_id, count in duplicate_acceptances)
            raise AuditIntegrityError(
                f"Barrier journal restore for run {self._run_id!r} (resume checkpoint "
                f"{restore.resume_checkpoint_id!r}) found duplicate live BUFFERED acceptances: {details}. "
                "A deposed leader's unfenced intake wrote a second acceptance — refusing silent latest-wins."
            )
        items = self._scheduler.list_blocked_barrier_items(run_id=self._run_id)
        scalars = restore.barrier_scalars if restore.barrier_scalars is not None else BarrierScalars(aggregation={}, coalesce={})

        # ---- Partition (design D1) ----------------------------------------
        # A BLOCKED row's barrier KIND is decided by its barrier_key ONLY:
        # barrier_key == coalesce_name        -> coalesce barrier
        # barrier_key == str(aggregation node_id) -> aggregation barrier
        # NEVER partition on node_id (a BLOCKED row's node_id is the
        # enqueue-time cursor, not the barrier owner) and NEVER on
        # coalesce_name (aggregation rows may carry non-NULL coalesce_name
        # LINEAGE for tokens that will coalesce after the flush).
        coalesce_keys: set[str] = {str(name) for name in self._coalesce_node_ids}
        aggregation_keys: set[str] = {str(node_id) for node_id in self._aggregation_settings}
        ambiguous = coalesce_keys & aggregation_keys
        if ambiguous:
            raise OrchestrationInvariantError(
                f"Barrier-key namespace collision between coalesce names and aggregation node ids: {sorted(ambiguous)}. "
                "The journal restore partition cannot disambiguate BLOCKED rows for these keys."
            )

        agg_items_by_node: dict[NodeID, list[TokenWorkItem]] = {}
        coalesce_items: list[TokenWorkItem] = []
        intake_pending_count = 0
        for item in items:
            if item.barrier_key is None:  # pragma: no cover - excluded by the query contract
                raise AuditIntegrityError(
                    f"list_blocked_barrier_items returned a row without barrier_key "
                    f"(work_item_id={item.work_item_id!r}, run {self._run_id!r})."
                )
            if item.barrier_key not in coalesce_keys and item.barrier_key not in aggregation_keys:
                raise AuditIntegrityError(
                    f"BLOCKED journal row for token {item.token_id!r} (run {self._run_id!r}, resume "
                    f"checkpoint {restore.resume_checkpoint_id!r}) carries orphan barrier_key "
                    f"{item.barrier_key!r}: not a configured coalesce ({sorted(coalesce_keys)}) "
                    f"or aggregation node ({sorted(aggregation_keys)}). The journal references a "
                    "barrier this pipeline no longer has — refusing to resume."
                )
            if item.barrier_adopted_epoch is None:
                # ADR-030 §E.2/§E.4: an intake-pending row was deposited but
                # never adopted (crash between mark_blocked and adoption, or a
                # mid-adoption rollback). It has NO batch_members/BUFFERED
                # rows and NO executor memory to restore — the legitimate
                # disposition is the next drain iteration's journal-first
                # intake, which adopts it under THIS leader's epoch.
                intake_pending_count += 1
                continue
            if item.barrier_key in coalesce_keys:
                coalesce_items.append(item)
            else:
                agg_items_by_node.setdefault(NodeID(item.barrier_key), []).append(item)
        if intake_pending_count:
            logger.info(
                "barrier journal restore: %d intake-pending BLOCKED row(s) left for the journal-first intake (run %s)",
                intake_pending_count,
                self._run_id,
            )
        # Adopted rows only from here down: derivations (attempt offsets,
        # batch ids, hold state ids) cover restored memory, not intake-pending
        # rows.
        items = [*coalesce_items, *(item for node_items in agg_items_by_node.values() for item in node_items)]
        if coalesce_items and self._coalesce_executor is None:
            raise OrchestrationInvariantError(
                f"Journal has {len(coalesce_items)} BLOCKED coalesce rows but no CoalesceExecutor is configured."
            )

        # ---- Audit derivations (no mutation yet) ---------------------------
        # Attempt offsets: max node_states attempt per journal token, + 1.
        # Derived here with ONE focused query rather than plumbed from
        # recovery's incomplete_by_row map: that map's exclusion set reads
        # journal BLOCKED rows (so the resume loop does not re-drive blocked
        # tokens), which excludes exactly the tokens this restore needs
        # offsets for.
        token_ids = [item.token_id for item in items]
        max_attempts = self._execution.get_max_node_state_attempts(self._run_id, token_ids) if token_ids else {}
        attempt_offsets: dict[str, int] = {
            token_id: (max_attempts[token_id] if token_id in max_attempts else -1) + 1 for token_id in token_ids
        }

        # Per-node batch metadata for every configured aggregation node.
        agg_plans: list[_AggregationRestorePlan] = []
        if self._aggregation_settings:
            members_by_batch: dict[str, list[str]] = {}
            for member in self._execution.get_all_batch_members_for_run(self._run_id):
                members_by_batch.setdefault(member.batch_id, []).append(member.token_id)
            batches_by_node: dict[str, list[Batch]] = {}
            for batch in self._execution.get_batches(self._run_id):
                batches_by_node.setdefault(str(batch.aggregation_node_id), []).append(batch)

            for node_id in sorted(self._aggregation_settings, key=str):
                node_items = agg_items_by_node[node_id] if node_id in agg_items_by_node else []
                # Reconciliation join (configured nodes against audit batches): a
                # configured node with no batches yet (batches are created
                # lazily on first row arrival) legitimately has zero rows, so
                # absence is an empty bucket, not Tier-1 corruption.
                node_batches = batches_by_node[str(node_id)] if str(node_id) in batches_by_node else []
                # COUNT(DISTINCT token_id), not raw COUNT: retry_batch COPIES
                # members into the retry batch, and handle_incomplete_batches
                # runs BEFORE this derivation on resume — a raw count would
                # double-count every member of a retried batch.
                accepted_count_total = len(
                    {
                        member_token
                        for node_batch in node_batches
                        for member_token in (members_by_batch[node_batch.batch_id] if node_batch.batch_id in members_by_batch else ())
                    }
                )
                completed_flush_count = sum(1 for node_batch in node_batches if node_batch.status is BatchStatus.COMPLETED)
                node_scalars = (
                    scalars.aggregation[str(node_id)] if str(node_id) in scalars.aggregation else AggregationNodeScalars(None, None)
                )
                # ---- ADR-030 §E.3a aggregation reconcile (elspeth-55546a6fd6) ---
                # A FAILED out-of-claim flush records terminal FAILURE/UNROUTED
                # token_outcomes for every buffered token (_handle_flush_error)
                # and THEN releases their BLOCKED scheduler rows in a SEPARATE
                # transaction (_mark_buffered_scheduler_work_terminal). A crash
                # between the two strands durable BLOCKED rows whose tokens are
                # already terminally failed: they carry NO live BUFFERED outcome,
                # so _derive_restored_batch_id below would refuse loudly and
                # brick EVERY resume attempt. Mirror the coalesce §E.3a holdless
                # path: the tokens are done — journal-release their orphaned
                # BLOCKED rows here (under this leader's coordination token) and
                # drop them from the restore set so the deriver sees only live
                # tokens. A fully-reconciled node then falls through to the
                # counter-only branch below ("flushes all FAILED" — exactly the
                # state that branch already anticipates).
                #
                # Scoped to (FAILURE, UNROUTED): the success-path BATCH_CONSUMED
                # crash residual (elspeth-3977d8ab60) still owes a sink output
                # and is NOT swept here — it keeps hitting the loud refusal.
                if node_items:
                    failed_terminal_ids = self._data_flow.get_failed_unrouted_terminal_token_ids(
                        self._run_id, [item.token_id for item in node_items]
                    )
                    if failed_terminal_ids:
                        reconciled = [item for item in node_items if item.token_id in failed_terminal_ids]
                        released = self._scheduler.mark_blocked_barrier_terminal(
                            run_id=self._run_id,
                            barrier_key=str(node_id),
                            token_ids=tuple(item.token_id for item in reconciled),
                            now=now,
                            coordination_token=self._coordination_token,
                            release_context={
                                "reason": "failed_flush_crash_reconcile",
                                "released_by": self._scheduler_lease_owner,
                                "restore_reconcile": True,
                            },
                        )
                        if released != len(reconciled):
                            raise AuditIntegrityError(
                                f"Restore §E.3a aggregation reconcile: FAILED-flush release at aggregation node "
                                f"{node_id!r} (run {self._run_id!r}, resume checkpoint {restore.resume_checkpoint_id!r}) "
                                f"terminalized {released} rows; expected exactly {len(reconciled)} orphaned "
                                "terminally-failed BLOCKED row(s)."
                            )
                        logger.info(
                            "barrier journal restore: §E.3a aggregation reconcile released %d orphaned BLOCKED row(s) "
                            "with terminal FAILURE/UNROUTED outcomes at node %s (run %s)",
                            len(reconciled),
                            node_id,
                            self._run_id,
                        )
                        node_items = [item for item in node_items if item.token_id not in failed_terminal_ids]
                if node_items:
                    batch_id = self._derive_restored_batch_id(node_id, node_items, restore)
                    agg_plans.append(
                        _AggregationRestorePlan(
                            node_id=node_id,
                            items=node_items,
                            member_order=(members_by_batch[batch_id] if batch_id in members_by_batch else []),
                            batch_id=batch_id,
                            accepted_count_total=accepted_count_total,
                            completed_flush_count=completed_flush_count,
                            scalars=node_scalars,
                        )
                    )
                elif completed_flush_count > 0 or accepted_count_total > 0 or str(node_id) in scalars.aggregation:
                    # Counter-only node: nothing buffered, but the audit trail
                    # shows prior activity — completed flushes, accepted rows
                    # (a node whose flushes all FAILED has accepted > 0 with
                    # zero COMPLETED batches and an empty buffer), or a stale
                    # scalars entry. Restore the derived counters so post-flush
                    # pagination metadata survives the resume.
                    agg_plans.append(
                        _AggregationRestorePlan(
                            node_id=node_id,
                            items=[],
                            member_order=[],
                            batch_id=None,
                            accepted_count_total=accepted_count_total,
                            completed_flush_count=completed_flush_count,
                            scalars=node_scalars,
                        )
                    )

        # Coalesce hold state ids: the OPEN node_state written at accept()
        # time at the coalesce node (the executor calls it the PENDING hold).
        coalesce_state_ids: Mapping[str, str] = {}
        if coalesce_items:
            coalesce_state_ids = self._execution.get_open_node_state_ids(
                self._run_id,
                node_ids=[str(node_id) for node_id in self._coalesce_node_ids.values()],
                token_ids=[item.token_id for item in coalesce_items],
            )

        # ---- ADR-030 §E.3a/§E.4 crash-window reconcile (findings 1 & 3) -----
        # Adopted coalesce rows with no OPEN state_id are in a crash window:
        # the adoption CAS committed (barrier_adopted_epoch non-NULL) but
        # accept() never wrote the PENDING hold node_state (the leader died
        # between steps 1 and 2 of _intake_adopt_coalesce_row).
        #
        # Two sub-cases, identified by whether the row's (coalesce_name, row_id)
        # key is Landscape-completed:
        #
        # a. Key completed (late-arrival crash §E.3a): the group already
        #    resolved; journal-release the row here at restore exactly like the
        #    live §E.3a path (mark_blocked_barrier_terminal with late_arrival
        #    context), using the new leader's coordination token.
        #
        # b. Key NOT completed (normal adoption crash §E.3): accept() never ran
        #    — re-run it after restore_from_journal populates executor state,
        #    writing the missing hold node_state under a fresh attempt offset.
        #    The row was already adopted, so the intake IS-NULL filter won't
        #    pick it up; this post-restore accept is the only recovery path.
        #
        # This replaces the old hard-refusal in restore_from_journal's
        # state_ids check, which fired on both reachable crash states.
        coalesce_holdless_items: list[TokenWorkItem] = []
        if coalesce_items:
            holdless = [item for item in coalesce_items if item.token_id not in coalesce_state_ids]
            if holdless:
                # Resolve the Landscape completed set once for all holdless rows.
                node_id_to_coalesce_name: dict[str, str] = {str(nid): str(name) for name, nid in self._coalesce_node_ids.items()}
                completed_pairs = self._execution.get_completed_row_ids_for_nodes(
                    self._run_id,
                    frozenset(node_id_to_coalesce_name.keys()),
                )
                completed_keys_set: frozenset[tuple[str, str]] = frozenset(
                    (node_id_to_coalesce_name[node_id_str], row_id)
                    for node_id_str, row_id in completed_pairs
                    if node_id_str in node_id_to_coalesce_name
                )
                for item in holdless:
                    coalesce_name_str = item.coalesce_name or item.barrier_key
                    key = (str(coalesce_name_str), item.row_id)
                    if key in completed_keys_set:
                        # §E.3a at restore: adopted-but-unreleased late row
                        # against a completed key — journal-release it now.
                        released = self._scheduler.mark_blocked_barrier_terminal(
                            run_id=self._run_id,
                            barrier_key=str(coalesce_name_str),
                            token_ids=(item.token_id,),
                            now=now,
                            coordination_token=self._coordination_token,
                            release_context={
                                "late_arrival": True,
                                "reason": "late_arrival_after_merge",
                                "released_by": self._scheduler_lease_owner,
                                "scope_row_id": item.row_id,
                                "restore_reconcile": True,
                            },
                        )
                        if released != 1:
                            raise AuditIntegrityError(
                                f"Restore §E.3a reconcile: late-arrival release for token {item.token_id!r} "
                                f"at coalesce {coalesce_name_str!r} (run {self._run_id!r}) terminalized "
                                f"{released} rows; expected exactly one."
                            )
                        logger.info(
                            "barrier journal restore: §E.3a reconcile released adopted-holdless late-arrival "
                            "token %s at coalesce %s/%s (run %s)",
                            item.token_id,
                            coalesce_name_str,
                            item.row_id,
                            self._run_id,
                        )
                    else:
                        # Normal adoption crash: accept() never ran — collect for
                        # post-restore accept (after restore_from_journal populates
                        # completed_keys so the executor can do late-arrival detection).
                        coalesce_holdless_items.append(item)
                # Remove ALL holdless items from the restore list; reconciled ones
                # are journal-released above, deferred ones handled post-restore.
                coalesce_items = [item for item in coalesce_items if item.token_id in coalesce_state_ids]

        # §E.5/§E.4: the durable coalesce_branch_losses ledger is the restore
        # source of truth for lost_branches across takeover; the D3 checkpoint
        # scalar is retained as a cross-check only (union, table wins on a
        # reason disagreement — the first durable record may already have
        # fired a must-fail policy).
        effective_coalesce_scalars: dict[tuple[str, str], CoalescePendingScalars] = dict(scalars.coalesce)
        if self._coalesce_executor is not None:
            for loss in self._scheduler.list_coalesce_branch_losses(run_id=self._run_id):
                key = (loss.coalesce_name, loss.row_id)
                existing = effective_coalesce_scalars[key] if key in effective_coalesce_scalars else None
                lost_branches = dict(existing.lost_branches) if existing is not None else {}
                checkpoint_reason = lost_branches[loss.branch_name] if loss.branch_name in lost_branches else None
                if checkpoint_reason is not None and checkpoint_reason != loss.reason:
                    logger.warning(
                        "coalesce restore: checkpoint lost-branch reason %r for %s/%s/%s disagrees with the durable "
                        "loss ledger %r; the ledger wins",
                        checkpoint_reason,
                        loss.coalesce_name,
                        loss.row_id,
                        loss.branch_name,
                        loss.reason,
                    )
                lost_branches[loss.branch_name] = loss.reason
                effective_coalesce_scalars[key] = CoalescePendingScalars(lost_branches=lost_branches)

        # ---- Mutate ---------------------------------------------------------
        # Coalesce first: ONE call for the whole executor (a second call would
        # discard this one — see CoalesceExecutor.restore_from_journal's caller
        # obligations). Called whenever a coalesce executor exists so completed
        # keys are reconstructed from the Landscape even with nothing pending,
        # and stale lost-branch scalars are dropped-with-log.
        if self._coalesce_executor is not None:
            # ---- §E.3 crash-window recovery: reset holdless non-completed rows ---
            # Adopted BLOCKED coalesce rows with no OPEN state_id whose key is NOT
            # completed are in the crash window between adopt_blocked_barrier_item
            # commit and CoalesceExecutor.accept() — accept() never wrote the hold.
            # Resetting barrier_adopted_epoch to NULL re-classifies them as
            # intake-pending: the first drain iteration's journal-first intake adopts
            # them afresh and runs the full accept + trigger path (merge, failure,
            # late-arrival), which the restore phase cannot safely produce (accept()
            # may fire a merge whose RowResult the __init__ path cannot commit to the
            # journal).  This reset is safe because the takeover CAS has already
            # committed — the old leader cannot concurrently re-adopt these rows.
            if coalesce_holdless_items:
                reset_count = self._scheduler.reset_adoption_marker_to_pending(
                    work_item_ids=[item.work_item_id for item in coalesce_holdless_items],
                    run_id=self._run_id,
                )
                if reset_count != len(coalesce_holdless_items):
                    logger.warning(
                        "barrier journal restore: §E.3 crash-window reset expected %d rows but "
                        "reset %d (run %s); the intake will re-classify any missed rows",
                        len(coalesce_holdless_items),
                        reset_count,
                        self._run_id,
                    )
                else:
                    logger.info(
                        "barrier journal restore: §E.3 crash-window reset %d holdless-non-completed "
                        "BLOCKED coalesce row(s) to intake-pending (run %s)",
                        reset_count,
                        self._run_id,
                    )
            self._coalesce_executor.restore_from_journal(
                items=coalesce_items,
                scalars=effective_coalesce_scalars,
                state_ids=coalesce_state_ids,
                attempt_offsets=attempt_offsets,
                resume_checkpoint_id=restore.resume_checkpoint_id,
                now=now,
            )

        for plan in agg_plans:
            self._aggregation_executor.restore_from_journal(
                node_id=plan.node_id,
                items=plan.items,
                member_order=plan.member_order,
                batch_id=plan.batch_id,
                accepted_count_total=plan.accepted_count_total,
                completed_flush_count=plan.completed_flush_count,
                scalars=plan.scalars,
                attempt_offsets=attempt_offsets,
                resume_checkpoint_id=restore.resume_checkpoint_id,
                now=now,
            )

    def _derive_restored_batch_id(
        self,
        node_id: NodeID,
        node_items: Sequence[TokenWorkItem],
        restore: BarrierJournalRestoreContext,
    ) -> str:
        """Resolve the in-progress batch id for one aggregation node's BLOCKED rows.

        Source of truth: each buffered token's BUFFERED token_outcome carries
        the batch_id it was accepted into (written by the fenced adoption
        verb since ADR-030 §E.2), read through the
        ``handle_incomplete_batches`` old->retry remap because the audit
        outcomes still reference the dead original batch when a crash
        interrupted a flush.

        Raises:
            AuditIntegrityError: If any token lacks a BUFFERED outcome with a
                batch_id, the group's tokens disagree on the (remapped)
                batch_id, the batch row is missing, or the batch belongs to a
                different aggregation node.
        """
        batch_id: str | None = None
        first_token_id: str | None = None
        for item in node_items:
            live_buffered = self._data_flow.get_live_buffered_outcomes(TokenRef(token_id=item.token_id, run_id=self._run_id))
            if len(live_buffered) > 1:
                # ADR-030 §C.4 row 6a / §E.4: token_outcomes has no non-terminal
                # uniqueness; >1 live BUFFERED rows means a deposed leader's
                # unfenced intake wrote a second acceptance. Refuse loudly —
                # never the historical silent latest-wins.
                duplicates = "; ".join(
                    f"outcome_id={o.outcome_id!r} batch_id={o.batch_id!r} "
                    f"recorded_at={o.recorded_at.isoformat()} context={o.context_json!r}"
                    for o in live_buffered
                )
                raise AuditIntegrityError(
                    f"BLOCKED journal row for token {item.token_id!r} at aggregation node {node_id!r} "
                    f"(run {self._run_id!r}, resume checkpoint {restore.resume_checkpoint_id!r}) has "
                    f"{len(live_buffered)} live BUFFERED token_outcomes — duplicate acceptances; a deposed "
                    "leader's unfenced intake wrote a second acceptance; refusing silent latest-wins. "
                    f"{duplicates}"
                )
            outcome = live_buffered[0] if live_buffered else None
            if outcome is None or outcome.batch_id is None:
                raise AuditIntegrityError(
                    f"BLOCKED journal row for token {item.token_id!r} at aggregation node {node_id!r} "
                    f"(run {self._run_id!r}, resume checkpoint {restore.resume_checkpoint_id!r}) has no "
                    f"matching BUFFERED token_outcome with a batch_id (got {outcome!r}) — the journal "
                    "and the audit trail disagree about this token being buffered."
                )
            resolved = restore.batch_id_remap.get(outcome.batch_id, outcome.batch_id)
            if batch_id is None:
                batch_id = resolved
                first_token_id = item.token_id
            elif resolved != batch_id:
                raise AuditIntegrityError(
                    f"BLOCKED journal rows at aggregation node {node_id!r} (run {self._run_id!r}, resume "
                    f"checkpoint {restore.resume_checkpoint_id!r}) split across batches: token "
                    f"{first_token_id!r} resolves batch_id={batch_id!r} but token {item.token_id!r} "
                    f"resolves batch_id={resolved!r}. One node has exactly one in-progress batch."
                )
        if batch_id is None:  # pragma: no cover - callers pass non-empty node_items
            raise AuditIntegrityError(f"_derive_restored_batch_id called with no journal rows for node {node_id!r}.")
        batch = self._execution.get_batch(batch_id)
        if batch is None:
            raise AuditIntegrityError(
                f"Restored batch_id {batch_id!r} for aggregation node {node_id!r} (run {self._run_id!r}) "
                "has no batches row — audit data corruption."
            )
        if str(batch.aggregation_node_id) != str(node_id):
            raise AuditIntegrityError(
                f"Restored batch_id {batch_id!r} belongs to aggregation node "
                f"{batch.aggregation_node_id!r}, but the journal BLOCKED rows carry barrier_key "
                f"{str(node_id)!r} (run {self._run_id!r}) — journal/audit disagreement."
            )
        return batch_id

    @property
    def run_id(self) -> str:
        """Run identifier owned by this processor."""
        return self._run_id

    def resolve_node_step(self, node_id: NodeID) -> int:
        """Resolve a node ID to processor step index (0-indexed)."""
        if node_id not in self._node_step_map:
            raise OrchestrationInvariantError(f"Node ID '{node_id}' missing from traversal step map")
        return self._node_step_map[node_id]

    def resolve_sink_step(self) -> int:
        """Resolve the audit step index for sink writes.

        Sinks are always the last step in processing, after all transforms,
        gates, aggregations, and coalesce nodes. Returns max(step_map) + 1.
        """
        if not self._node_step_map:
            raise OrchestrationInvariantError(
                "Cannot resolve sink step: node step map is empty. Pipeline must have at least one processing node."
            )
        return max(self._node_step_map.values()) + 1

    def _resolve_audit_step_for_node(self, node_id: NodeID) -> int:
        """Resolve 1-indexed audit step for a processing node.

        Delegates to the factory-produced StepResolver (make_step_resolver).
        """
        return self._step_resolver(node_id)

    def _emit_telemetry(self, event: TelemetryEvent) -> None:
        """Emit telemetry event if manager is configured.

        Telemetry is emitted AFTER Landscape recording succeeds. Landscape is
        the legal record; telemetry is operational visibility.

        Args:
            event: The telemetry event to emit
        """
        if self._telemetry_manager is not None:
            self._telemetry_manager.handle_event(event)

    def _emit_transform_completed(
        self,
        token: TokenInfo,
        transform: TransformProtocol,
        transform_result: TransformResult,
    ) -> None:
        """Emit TransformCompleted telemetry event.

        Called AFTER Landscape recording succeeds in TransformExecutor.

        Args:
            token: Token that was processed
            transform: Transform that was executed
            transform_result: Result from the transform execution
        """
        if self._telemetry_manager is None:
            return

        from datetime import datetime

        from elspeth.contracts import TransformCompleted
        from elspeth.contracts.enums import NodeStateStatus

        status = NodeStateStatus.COMPLETED if transform_result.status == "success" else NodeStateStatus.FAILED

        # node_id is assigned during DAG construction in from_plugin_instances()
        if transform.node_id is None:
            raise OrchestrationInvariantError("node_id must be assigned by DAG construction before execution")
        self._emit_telemetry(
            TransformCompleted(
                timestamp=datetime.now(UTC),
                run_id=self._run_id,
                row_id=token.row_id,
                token_id=token.token_id,
                node_id=transform.node_id,
                plugin_name=transform.name,
                status=status,
                duration_ms=transform_result.duration_ms if transform_result.duration_ms is not None else 0.0,
                input_hash=transform_result.input_hash,
                output_hash=transform_result.output_hash,
            )
        )

    def _emit_gate_evaluated(
        self,
        token: TokenInfo,
        gate_name: str,
        gate_node_id: str,
        routing_mode: RoutingMode,
        destinations: tuple[str, ...],
    ) -> None:
        """Emit GateEvaluated telemetry event.

        Called AFTER Landscape recording succeeds in GateExecutor.

        Args:
            token: Token that was routed
            gate_name: Name of the gate (from GateSettings)
            gate_node_id: Node ID of the gate
            routing_mode: How routing was performed (move, copy)
            destinations: Destination node/sink names
        """
        if self._telemetry_manager is None:
            return

        from datetime import datetime

        from elspeth.contracts import GateEvaluated

        self._emit_telemetry(
            GateEvaluated(
                timestamp=datetime.now(UTC),
                run_id=self._run_id,
                row_id=token.row_id,
                token_id=token.token_id,
                node_id=gate_node_id,
                plugin_name=gate_name,
                routing_mode=routing_mode,
                destinations=destinations,
            )
        )

    def _emit_token_completed(
        self,
        token: TokenInfo,
        *,
        outcome: TerminalOutcome | None,
        path: TerminalPath,
        sink_name: str | None = None,
    ) -> None:
        """Emit TokenCompleted telemetry event.

        Called AFTER Landscape recording succeeds (record_token_outcome).

        Args:
            token: Token that reached terminal state
            outcome: Lifecycle outcome (None for non-terminal BUFFERED)
            path: Terminal provenance path
            sink_name: Destination sink if applicable
        """
        if self._telemetry_manager is None:
            return

        from datetime import datetime

        from elspeth.contracts import TokenCompleted

        self._emit_telemetry(
            TokenCompleted(
                timestamp=datetime.now(UTC),
                run_id=self._run_id,
                row_id=token.row_id,
                token_id=token.token_id,
                outcome=outcome,
                path=path,
                sink_name=sink_name,
            )
        )

    def _get_gate_destinations(self, outcome: GateOutcome) -> tuple[str, ...]:
        """Extract destination names from gate outcome for telemetry.

        Args:
            outcome: The gate outcome containing routing information

        Returns:
            Tuple of destination names (sink names or path names for forks)
        """
        if outcome.sink_name is not None:
            return (outcome.sink_name,)
        elif outcome.discarded is True:
            return ("discard",)
        elif outcome.result.action.kind == RoutingKind.FORK_TO_PATHS:
            # For forks, return the branch names of child tokens
            return tuple(child.branch_name for child in outcome.child_tokens if child.branch_name)
        elif outcome.next_node_id is not None and outcome.result.action.kind == RoutingKind.ROUTE:
            # For route-label processing branches, report the chosen route label.
            return outcome.result.action.destinations
        else:
            # Continue routing - destination is "continue"
            return ("continue",)

    # ─────────────────────────────────────────────────────────────────────────
    # Public facade for aggregation timeout checking
    # Provides clean API for orchestrator timeout checks
    # ─────────────────────────────────────────────────────────────────────────

    def check_aggregation_timeout(self, node_id: NodeID) -> tuple[bool, TriggerType | None]:
        """Check if an aggregation should flush due to timeout.

        This is a public facade for orchestrator to check timeout conditions
        without directly accessing private _aggregation_executor.

        Note: This method is called in the hot path (before every row is processed),
        so it uses the optimized check_flush_status() which does a single dict
        lookup instead of two separate calls.

        Args:
            node_id: The aggregation node ID to check

        Returns:
            Tuple of (should_flush, trigger_type):
            - should_flush: True if trigger condition is met
            - trigger_type: The type of trigger that fired (TIMEOUT, COUNT, etc.) or None
        """
        return self._aggregation_executor.check_flush_status(node_id)

    def get_aggregation_buffer_count(self, node_id: NodeID) -> int:
        """Get the number of rows buffered in an aggregation.

        Args:
            node_id: The aggregation node ID

        Returns:
            Number of rows currently buffered
        """
        return self._aggregation_executor.get_buffer_count(node_id)

    def get_barrier_scalars(self) -> BarrierScalars:
        """Compose the underivable barrier scalars for the checkpoint row.

        F1 design D3: the checkpoint persists only scalar barrier metadata —
        buffered tokens live in journal BLOCKED rows; counters and state ids
        derive from audit tables at restore. The live executors each emit only
        nodes/keys with actual state (latched trigger fire offsets, recorded
        lost branches — see the executors' ``get_barrier_scalars``), so a
        quiescent pipeline composes an empty BarrierScalars (``has_state``
        False), which the checkpoint manager serializes as NULL.

        On restore a missing aggregation entry means unlatched ``(None, None)``;
        a missing coalesce key means no recorded losses.

        Returns:
            BarrierScalars with aggregation latches keyed by str(node_id) and
            coalesce lost-branch records keyed by (coalesce_name, row_id).
        """
        coalesce: dict[tuple[str, str], CoalescePendingScalars] = (
            dict(self._coalesce_executor.get_barrier_scalars()) if self._coalesce_executor is not None else {}
        )
        return BarrierScalars(
            aggregation={str(node_id): scalars for node_id, scalars in self._aggregation_executor.get_barrier_scalars().items()},
            coalesce=coalesce,
        )

    # ─────────────────────────────────────────────────────────────────────────
    # Aggregation flush helpers (shared by handle_timeout_flush and
    # _process_batch_aggregation_node flush path)
    # ─────────────────────────────────────────────────────────────────────────

    def _derive_coalesce_from_tokens(
        self,
        buffered_tokens: list[TokenInfo],
    ) -> tuple[NodeID | None, CoalesceName | None]:
        """Derive coalesce metadata from buffered tokens' branch_name.

        For timeout/end-of-source flushes, coalesce info isn't passed in from
        a WorkItem — it must be derived from the tokens' branch membership.
        """
        if buffered_tokens:
            branch_name = buffered_tokens[0].branch_name
            if branch_name and BranchName(branch_name) in self._branch_to_coalesce:
                coalesce_name = self._branch_to_coalesce[BranchName(branch_name)]
                return self._coalesce_node_ids[coalesce_name], coalesce_name
        return None, None

    def _handle_flush_error(
        self,
        fctx: _FlushContext,
    ) -> tuple[RowResult, ...]:
        """Handle failed aggregation flush for both passthrough and transform modes.

        Both modes now have BUFFERED (non-terminal) at buffer time,
        so FAILED can be recorded as the terminal outcome for all tokens.
        """
        error_hash = compute_error_hash(fctx.error_msg, exception_type="TransformError")
        results: list[RowResult] = []
        failure = FailureInfo(exception_type="TransformError", message=fctx.error_msg)

        for token in fctx.buffered_tokens:
            try:
                self._data_flow.record_token_outcome(
                    ref=TokenRef(token_id=token.token_id, run_id=self._run_id),
                    outcome=TerminalOutcome.FAILURE,
                    path=TerminalPath.UNROUTED,
                    error_hash=error_hash,
                )
            except LandscapeRecordError as record_failure:
                raise AuditIntegrityError(
                    f"Failed to record FAILED outcome for token {token.token_id!r} "
                    f"during batch flush failure handling "
                    f"(transform={fctx.transform.name!r}, node={fctx.node_id!r}). "
                    f"Audit trail is INCOMPLETE — some buffered tokens may already "
                    f"be terminalized while others remain BUFFERED. "
                    f"Recorder failure: {type(record_failure).__name__}: {record_failure}. "
                    f"Original flush error: {fctx.error_msg}"
                ) from record_failure
            with best_effort(
                "TokenCompleted telemetry after batch-flush FAILED audit",
                run_id=self._run_id,
                token_id=token.token_id,
                transform_node_id=fctx.node_id,
                transform_name=fctx.transform.name,
            ):
                self._emit_token_completed(
                    token,
                    outcome=TerminalOutcome.FAILURE,
                    path=TerminalPath.UNROUTED,
                )
            results.append(
                RowResult(
                    token=token,
                    final_data=token.row_data,
                    outcome=TerminalOutcome.FAILURE,
                    path=TerminalPath.UNROUTED,
                    error=failure,
                )
            )

        return tuple(results)

    def _cross_check_flush_output(
        self,
        fctx: _FlushContext,
        result: TransformResult,
    ) -> None:
        """Batch-flush declaration dispatch before any terminal emissions.

        ADR-009 §Clause 2 — this closes the gap ADR-008 left open. The batch
        aggregation flush path previously trusted the static annotation; a
        mis-annotated batch-aware transform (e.g., ``BatchReplicate``) could
        silently drop fields from emitted rows without any audit record.

        Semantic decisions (ADR-009 §§2.3, 2.4):

        - **PASSTHROUGH mode (1:1).** Each output token pairs with exactly one
          input token. The cross-check walks pairs and uses that specific
          input token's contract fields as ``input_fields``. A heterogeneous
          batch is not a hazard — each pair is checked independently.
        - **TRANSFORM mode (N:M, batch-homogeneous).** Every output row is
          checked against the intersection of all buffered input contracts
          (ADR-007 table line 53). This is the weakest shared guarantee — a
          transform claiming ``passes_through_input=True`` must preserve what
          every input contributed.

        Called BEFORE ``_emit_transform_completed`` and the routing methods
        (§2.5): a failed cross-check must not follow a COMPLETED or
        CONSUMED_IN_BATCH terminal-state emission on any token, which would
        violate CLAUDE.md's "every row reaches exactly one terminal state"
        invariant.

        Raises:
            FrameworkBugError: A buffered token has no input contract.
            DeclarationContractViolation | PluginContractViolation:
                Any batch-flush declaration contract fires.
                ``_record_flush_violation`` writes per-token FAILED audit
                entries before re-raising.
        """
        # Gather emitted rows uniformly across both output modes.
        if result.is_multi_row:
            emitted: list[PipelineRow] = list(result.rows) if result.rows is not None else []
        elif result.row is not None:
            emitted = [result.row]
        else:
            emitted = []
        used_success_empty = result.rows is not None and len(result.rows) == 0

        identity_token = fctx.triggering_token or fctx.buffered_tokens[0]
        transform_node_id_str = str(fctx.node_id)

        try:
            verify_zero_emission_declaration_path(
                plugin=fctx.transform,
                plugin_name=fctx.transform.name,
                node_id=transform_node_id_str,
                run_id=self._run_id,
                row_id=identity_token.row_id,
                token_id=identity_token.token_id,
                emitted_count=len(emitted),
                used_success_empty=used_success_empty,
            )

            # _FlushContext.__post_init__ guarantees buffered_tokens is non-empty;
            # no defensive emptiness guard (CLAUDE.md: defensive programming
            # forbidden for internal paths).
            for i, token in enumerate(fctx.buffered_tokens):
                if token.row_data.contract is None:
                    raise FrameworkBugError(
                        f"Batch flush: buffered token {i} "
                        f"(token_id={token.token_id!r}) has no contract "
                        f"(transform={fctx.transform.name!r}, node={fctx.node_id!r}). "
                        "Framework invariant violated."
                    )

            per_input_field_sets = [
                frozenset(fc.normalized_name for fc in token.row_data.contract.fields) for token in fctx.buffered_tokens
            ]

            static_contract = fctx.transform.effective_static_contract()

            if fctx.settings.output_mode == OutputMode.PASSTHROUGH:
                # 1:1 pairing — routing enforces len(emitted) == len(buffered).
                # Dispatch each pair through the audit-complete batch-flush
                # dispatcher (ADR-010 §Semantics amendment 2026-04-20). Each
                # pair's effective_input_fields is derived per-token — the
                # PASSTHROUGH carve-out preserves per-token identity.
                if len(emitted) == len(fctx.buffered_tokens):
                    for token, emitted_row, token_fields in zip(
                        fctx.buffered_tokens,
                        emitted,
                        per_input_field_sets,
                        strict=True,
                    ):
                        run_batch_flush_checks(
                            inputs=BatchFlushInputs(
                                plugin=fctx.transform,
                                node_id=transform_node_id_str,
                                run_id=self._run_id,
                                row_id=token.row_id,
                                token_id=token.token_id,
                                buffered_tokens=(token,),
                                static_contract=static_contract,
                                effective_input_fields=token_fields,
                            ),
                            outputs=BatchFlushOutputs(emitted_rows=(emitted_row,)),
                        )
                elif len(emitted) == 0:
                    # Zero-emission success has no 1:1 pairing witness, but the
                    # dispatcher still must evaluate governance contracts and the
                    # pass-through empty-emission path. The honest batch-level
                    # surface is the shared intersection across buffered tokens.
                    input_fields = frozenset.intersection(*per_input_field_sets)
                    identity_token = fctx.triggering_token or fctx.buffered_tokens[0]
                    run_batch_flush_checks(
                        inputs=BatchFlushInputs(
                            plugin=fctx.transform,
                            node_id=transform_node_id_str,
                            run_id=self._run_id,
                            row_id=identity_token.row_id,
                            token_id=identity_token.token_id,
                            buffered_tokens=tuple(fctx.buffered_tokens),
                            static_contract=static_contract,
                            effective_input_fields=input_fields,
                        ),
                        outputs=BatchFlushOutputs(emitted_rows=()),
                    )
                else:
                    # Count mismatch is ``_route_passthrough_results``'s
                    # concern; pass through unchecked so routing can surface
                    # the OrchestrationInvariantError with its own message.
                    pass
            else:
                # TRANSFORM mode: batch-homogeneous intersection (ADR-009 §Clause 2).
                # Every emitted row must preserve the intersection of every
                # buffered token's input contract — the weakest shared guarantee.
                # The batch-flush dispatcher surfaces the intersection via
                # ``BatchFlushInputs.effective_input_fields`` (panel F1
                # resolution: caller-computed; contracts don't re-derive).
                input_fields = frozenset.intersection(*per_input_field_sets)
                run_batch_flush_checks(
                    inputs=BatchFlushInputs(
                        plugin=fctx.transform,
                        node_id=transform_node_id_str,
                        run_id=self._run_id,
                        row_id=identity_token.row_id,
                        token_id=identity_token.token_id,
                        buffered_tokens=tuple(fctx.buffered_tokens),
                        static_contract=static_contract,
                        effective_input_fields=input_fields,
                    ),
                    outputs=BatchFlushOutputs(emitted_rows=tuple(emitted)),
                )
        except PluginContractViolation as violation:
            self._record_flush_violation(fctx, violation)
            raise
        except DeclarationContractViolation as violation:
            self._record_flush_violation(fctx, violation)
            raise
        except AggregateDeclarationContractViolation as aggregate:
            # Audit-complete multi-fire case: every buffered token gets a
            # FAILED outcome carrying the aggregate evidence bundle.
            self._record_flush_violation(fctx, aggregate)
            raise

    def _record_flush_violation(
        self,
        fctx: _FlushContext,
        violation: DeclarationContractViolation | PluginContractViolation | AggregateDeclarationContractViolation,
    ) -> None:
        """Record FAILED audit entries for every buffered token on flush failure.

        The violation is semantically batch-level but the audit trail must
        capture per-token evidence for every buffered token. ``per_token_audit_payload``
        is rebuilt inside the loop so ``$.context.token_id`` reflects the
        row's own token, not the triggering token's.

        If ``record_token_outcome`` raises mid-loop, the audit trail is
        incomplete. Rather than silently swallow the failure and re-raise the
        original violation, crash loudly with ``AuditIntegrityError`` so the
        operator learns about the audit-write failure. The primary violation
        is preserved via ``__context__`` (Python automatically sets it
        because this is inside ``except``).
        """
        if isinstance(violation, PassThroughContractViolation):
            violation_summary = f"PassThroughContractViolation:{fctx.transform.name}:{sorted(violation.divergence_set)}"
        else:
            violation_summary = f"{type(violation).__name__}:{fctx.transform.name}"
        error_hash = compute_error_hash(violation_summary)
        base_audit = violation.to_audit_dict()

        for token in fctx.buffered_tokens:
            per_token_audit_payload: dict[str, object] = {
                **base_audit,
                "token_id": token.token_id,
                "row_id": token.row_id,
                "triggering_token_id": (fctx.triggering_token.token_id if fctx.triggering_token is not None else None),
            }
            try:
                self._data_flow.record_token_outcome(
                    ref=TokenRef(token_id=token.token_id, run_id=self._run_id),
                    outcome=TerminalOutcome.FAILURE,
                    path=TerminalPath.UNROUTED,
                    error_hash=error_hash,
                    context=per_token_audit_payload,
                )
            except LandscapeRecordError as record_failure:
                raise AuditIntegrityError(
                    f"Failed to record {type(violation).__name__} FAILED outcome "
                    f"for token {token.token_id!r} in batch flush "
                    f"(transform={fctx.transform.name!r}, node={fctx.node_id!r}). "
                    f"Audit trail is INCOMPLETE — FAILED records may exist for some "
                    f"buffered tokens but not others. "
                    f"Recorder failure: {type(record_failure).__name__}: {record_failure}. "
                    f"Original violation: {violation!s}"
                ) from record_failure
            with best_effort(
                "TokenCompleted telemetry after batch-flush violation audit",
                run_id=self._run_id,
                token_id=token.token_id,
                transform_node_id=fctx.node_id,
                transform_name=fctx.transform.name,
            ):
                self._emit_token_completed(
                    token,
                    outcome=TerminalOutcome.FAILURE,
                    path=TerminalPath.UNROUTED,
                )

    def _record_dropped_by_filter_outcome(
        self,
        *,
        token: TokenInfo,
        transform_name: str,
        node_id: NodeID,
        path_label: str,
    ) -> None:
        """Record DROPPED_BY_FILTER or raise AuditIntegrityError on recorder failure."""
        try:
            self._data_flow.record_token_outcome(
                ref=TokenRef(token_id=token.token_id, run_id=self._run_id),
                outcome=TerminalOutcome.SUCCESS,
                path=TerminalPath.FILTER_DROPPED,
            )
        except LandscapeRecordError as record_failure:
            raise AuditIntegrityError(
                f"Failed to record DROPPED_BY_FILTER outcome for token {token.token_id!r} "
                f"{path_label} (transform={transform_name!r}, node={node_id!r}). "
                f"Audit trail is INCOMPLETE — the transform node state is already COMPLETED "
                f"but the terminal token_outcome is missing. Recorder failure: "
                f"{type(record_failure).__name__}: {record_failure}"
            ) from record_failure

    def _record_gate_discarded_outcome(
        self,
        *,
        token: TokenInfo,
        gate_name: str,
        node_id: NodeID,
    ) -> None:
        """Record terminal gate discard outcome or raise on audit failure."""
        try:
            self._data_flow.record_token_outcome(
                ref=TokenRef(token_id=token.token_id, run_id=self._run_id),
                outcome=TerminalOutcome.SUCCESS,
                path=TerminalPath.GATE_DISCARDED,
            )
        except LandscapeRecordError as record_failure:
            raise AuditIntegrityError(
                f"Failed to record GATE_DISCARDED outcome for token {token.token_id!r} "
                f"(gate={gate_name!r}, node={node_id!r}). "
                f"Audit trail is INCOMPLETE — the gate node state is already COMPLETED "
                f"but the terminal token_outcome is missing. Recorder failure: "
                f"{type(record_failure).__name__}: {record_failure}"
            ) from record_failure

    def _route_empty_emission_results(
        self,
        fctx: _FlushContext,
    ) -> tuple[tuple[RowResult, ...], list[WorkItem]]:
        """Record terminal outcomes for a successful batch flush with zero rows.

        If these buffered tokens were fork branches awaiting a downstream
        coalesce, each dropped branch must still notify the coalesce executor
        so joins do not strand.
        """
        results: list[RowResult] = []
        child_items: list[WorkItem] = []
        for token in fctx.buffered_tokens:
            self._record_dropped_by_filter_outcome(
                token=token,
                transform_name=fctx.transform.name,
                node_id=fctx.node_id,
                path_label="during empty batch flush",
            )
            with best_effort(
                "TokenCompleted telemetry after empty batch-flush audit",
                run_id=self._run_id,
                token_id=token.token_id,
                transform_node_id=fctx.node_id,
                transform_name=fctx.transform.name,
            ):
                self._emit_token_completed(
                    token,
                    outcome=TerminalOutcome.SUCCESS,
                    path=TerminalPath.FILTER_DROPPED,
                )
            results.append(
                RowResult(
                    token=token,
                    final_data=token.row_data,
                    outcome=TerminalOutcome.SUCCESS,
                    path=TerminalPath.FILTER_DROPPED,
                )
            )
            results.extend(
                self._notify_coalesce_of_lost_branch(
                    token,
                    "dropped_by_filter",
                    child_items,
                )
            )
        return tuple(results), child_items

    def _route_passthrough_results(
        self,
        fctx: _FlushContext,
        result: TransformResult,
    ) -> tuple[tuple[RowResult, ...], list[WorkItem]]:
        """Route passthrough aggregation results after successful flush.

        Passthrough mode: original tokens continue with enriched data.
        Validates 1:1 row count, updates token data, and routes to
        downstream processing or COMPLETED outcome.
        """
        if not result.is_multi_row:
            raise OrchestrationInvariantError(
                f"Passthrough mode requires multi-row result, "
                f"but transform '{fctx.transform.name}' returned single row. "
                f"Use TransformResult.success_multi() for passthrough."
            )
        if result.rows is None:
            raise RuntimeError("Multi-row result has rows=None")
        if len(result.rows) == 0:
            return self._route_empty_emission_results(fctx)
        if len(result.rows) != len(fctx.buffered_tokens):
            raise OrchestrationInvariantError(
                f"Passthrough mode requires same number of output rows "
                f"as input rows. Transform '{fctx.transform.name}' returned "
                f"{len(result.rows)} rows but received {len(fctx.buffered_tokens)} input rows."
            )

        pipeline_rows = list(result.rows)
        has_downstream = self._nav.resolve_next_node(fctx.node_id) is not None
        first_branch = fctx.buffered_tokens[0].branch_name if fctx.buffered_tokens else None
        needs_coalesce = fctx.coalesce_node_id is not None and fctx.coalesce_name is not None and first_branch is not None

        results: list[RowResult] = []
        child_items: list[WorkItem] = []

        if has_downstream or needs_coalesce:
            work_item_coalesce_name = fctx.coalesce_name if needs_coalesce else None
            for token, enriched_data in zip(fctx.buffered_tokens, pipeline_rows, strict=True):
                updated_token = token.with_updated_data(enriched_data)
                child_items.append(
                    self._nav.create_continuation_work_item(
                        token=updated_token,
                        current_node_id=fctx.node_id,
                        coalesce_name=work_item_coalesce_name,
                    )
                )
        else:
            for token, enriched_data in zip(fctx.buffered_tokens, pipeline_rows, strict=True):
                updated_token = token.with_updated_data(enriched_data)
                results.append(
                    RowResult(
                        token=updated_token,
                        final_data=enriched_data,
                        outcome=TerminalOutcome.SUCCESS,
                        path=TerminalPath.DEFAULT_FLOW,
                        sink_name=fctx.transform.on_success,
                    )
                )

        return tuple(results), child_items

    def _route_transform_results(
        self,
        fctx: _FlushContext,
        result: TransformResult,
    ) -> tuple[tuple[RowResult, ...], list[WorkItem]]:
        """Route transform-mode aggregation results after successful flush.

        Transform mode: N input rows → M output rows with new tokens via expand_token.
        Records per-token terminal outcomes (CONSUMED_IN_BATCH or QUARANTINED),
        emits deferred TokenCompleted telemetry, then routes expanded tokens downstream.

        Batch transforms can quarantine individual rows. Quarantined tokens
        get QUARANTINED terminal state instead of CONSUMED_IN_BATCH, identified
        via quarantined_indices in the result's success_reason metadata.
        """
        quarantined_index_set = _validated_quarantined_indices(
            result,
            buffered_token_count=len(fctx.buffered_tokens),
            aggregation_name=fctx.settings.name,
        )

        # Extract output rows
        if result.is_multi_row:
            if result.rows is None:
                raise RuntimeError("Multi-row result has rows=None")
            output_rows = result.rows
        else:
            if result.row is None:
                raise RuntimeError(
                    f"Aggregation transform '{fctx.transform.name}' returned None for result.row "
                    f"in 'transform' mode. Batch-aware transforms must return a row via "
                    f"TransformResult.success(row) or rows via TransformResult.success_multi(rows). "
                    f"This is a plugin bug."
                )
            output_rows = (result.row,)
        if len(output_rows) == 0:
            return self._route_empty_emission_results(fctx)

        # Enforce expected_output_count if configured
        if fctx.settings.expected_output_count is not None:
            actual_count = len(output_rows)
            if actual_count != fctx.settings.expected_output_count:
                raise RuntimeError(
                    f"Aggregation '{fctx.settings.name}' produced {actual_count} output row(s), "
                    f"but expected_output_count={fctx.settings.expected_output_count}. "
                    f"This is a plugin contract violation."
                )

        results: list[RowResult] = []
        child_items: list[WorkItem] = []

        if fctx.buffered_tokens:
            non_quarantined_tokens = tuple(token for index, token in enumerate(fctx.buffered_tokens) if index not in quarantined_index_set)
            if not non_quarantined_tokens:
                raise OrchestrationInvariantError(
                    f"Aggregation {fctx.settings.name!r} emitted {len(output_rows)} output row(s) "
                    f"but all {len(fctx.buffered_tokens)} buffered token(s) were quarantined"
                )
            expand_parent_token = (
                fctx.expand_parent_token
                if any(token.token_id == fctx.expand_parent_token.token_id for token in non_quarantined_tokens)
                else non_quarantined_tokens[0]
            )
            output_contract = output_rows[0].contract
            expanded_tokens, _expand_group_id = self._token_manager.expand_token(
                parent_token=expand_parent_token,
                expanded_rows=[row.to_dict() for row in output_rows],
                output_contract=output_contract,
                node_id=fctx.node_id,
                run_id=self._run_id,
                record_parent_outcome=False,
            )

            # Record terminal outcomes for ALL buffered tokens AFTER expand_token
            # succeeds. Recording before validation/expansion would leave parent
            # tokens in a terminal state (CONSUMED_IN_BATCH/QUARANTINED) with no
            # child tokens if a later step fails — recovery would skip them.
            for i, token in enumerate(fctx.buffered_tokens):
                if i in quarantined_index_set:
                    error_hash = compute_error_hash(f"quarantined_in_batch:{fctx.batch_id}:{i}")
                    batch_id = None
                    outcome = TerminalOutcome.FAILURE
                    path = TerminalPath.QUARANTINED_AT_SOURCE
                else:
                    error_hash = None
                    batch_id = fctx.batch_id
                    outcome = TerminalOutcome.TRANSIENT
                    path = TerminalPath.BATCH_CONSUMED
                try:
                    self._data_flow.record_token_outcome(
                        ref=TokenRef(token_id=token.token_id, run_id=self._run_id),
                        outcome=outcome,
                        path=path,
                        error_hash=error_hash,
                        batch_id=batch_id,
                    )
                except LandscapeRecordError as record_failure:
                    raise AuditIntegrityError(
                        f"Failed to record batch parent terminal outcome for token {token.token_id!r} "
                        f"during transform-mode aggregation routing "
                        f"(transform={fctx.transform.name!r}, node={fctx.node_id!r}, batch_id={fctx.batch_id!r}). "
                        f"Audit trail is INCOMPLETE — expanded child tokens were already created and "
                        f"some buffered parents may already be terminalized while others remain BUFFERED. "
                        f"Recorder failure: {type(record_failure).__name__}: {record_failure}."
                    ) from record_failure
                self._emit_token_completed(
                    token,
                    outcome=outcome,
                    path=path,
                )

            # Build triggering RowResult if applicable (count-triggered only).
            # The triggering token is always the last buffered token (buffered
            # immediately before flush), so its index is len(buffered_tokens) - 1.
            # Its outcome must match what the recorder loop recorded —
            # QUARANTINED if in quarantined_index_set, CONSUMED_IN_BATCH otherwise.
            if fctx.triggering_token is not None:
                triggering_index = len(fctx.buffered_tokens) - 1
                if triggering_index in quarantined_index_set:
                    triggering_outcome = TerminalOutcome.FAILURE
                    triggering_path = TerminalPath.QUARANTINED_AT_SOURCE
                else:
                    triggering_outcome = TerminalOutcome.TRANSIENT
                    triggering_path = TerminalPath.BATCH_CONSUMED
                results.append(
                    RowResult(
                        token=fctx.triggering_token,
                        final_data=fctx.triggering_token.row_data,
                        outcome=triggering_outcome,
                        path=triggering_path,
                    )
                )

            if quarantined_index_set:
                triggering_index_val = len(fctx.buffered_tokens) - 1 if fctx.triggering_token is not None else -1
                for i, token in enumerate(fctx.buffered_tokens):
                    if i in quarantined_index_set and i != triggering_index_val:
                        results.append(
                            RowResult(
                                token=token,
                                final_data=token.row_data,
                                outcome=TerminalOutcome.FAILURE,
                                path=TerminalPath.QUARANTINED_AT_SOURCE,
                            )
                        )

            # Route expanded tokens downstream
            has_downstream = self._nav.resolve_next_node(fctx.node_id) is not None
            first_expanded_branch = expanded_tokens[0].branch_name if expanded_tokens else None
            needs_coalesce = fctx.coalesce_node_id is not None and fctx.coalesce_name is not None and first_expanded_branch is not None

            if has_downstream or needs_coalesce:
                work_item_coalesce_name = fctx.coalesce_name if needs_coalesce else None
                for token in expanded_tokens:
                    child_items.append(
                        self._nav.create_continuation_work_item(
                            token=token,
                            current_node_id=fctx.node_id,
                            coalesce_name=work_item_coalesce_name,
                        )
                    )
            else:
                for token in expanded_tokens:
                    results.append(
                        RowResult(
                            token=token,
                            final_data=token.row_data,
                            outcome=TerminalOutcome.SUCCESS,
                            path=TerminalPath.DEFAULT_FLOW,
                            sink_name=fctx.transform.on_success,
                        )
                    )

        return tuple(results), child_items

    def handle_timeout_flush(
        self,
        node_id: NodeID,
        transform: TransformProtocol,
        ctx: PluginContext,
        trigger_type: TriggerType,
    ) -> tuple[tuple[RowResult, ...], list[WorkItem]]:
        """Handle an aggregation flush triggered outside normal row processing.

        Handles TIMEOUT (between row arrivals) and END_OF_SOURCE (remaining buffers)
        flushes. Delegates to shared flush helpers after building _FlushContext.

        Args:
            node_id: The aggregation node ID
            transform: The batch-aware transform to execute
            ctx: Plugin context
            trigger_type: The trigger type (TIMEOUT or END_OF_SOURCE)

        Returns:
            Tuple of (results, work_items):
            - results: RowResults for completed tokens (terminal state)
            - work_items: WorkItem list for tokens needing further processing
        """
        settings = self._aggregation_settings[node_id]

        result, buffered_tokens, batch_id = self._aggregation_executor.execute_flush(
            node_id=node_id,
            transform=cast(BatchTransformProtocol, transform),
            ctx=ctx,
            trigger_type=trigger_type,
        )

        coalesce_node_id, coalesce_name = self._derive_coalesce_from_tokens(buffered_tokens)

        fctx = _FlushContext(
            node_id=node_id,
            transform=transform,
            settings=settings,
            buffered_tokens=tuple(buffered_tokens),
            batch_id=batch_id,
            error_msg="Batch transform failed during timeout flush",
            expand_parent_token=buffered_tokens[0],
            triggering_token=None,
            coalesce_node_id=coalesce_node_id,
            coalesce_name=coalesce_name,
        )

        if result.status != "success":
            flush_error = self._handle_flush_error(fctx)
            self._mark_buffered_scheduler_work_terminal(node_id, tuple(buffered_tokens))
            return flush_error, []

        # ADR-009 §Clause 2: runtime cross-check for passes_through_input
        # transforms on the batch-aware flush path. MUST run BEFORE
        # _emit_transform_completed so a failed cross-check does not follow
        # a COMPLETED terminal-state emission on any token.
        self._cross_check_flush_output(fctx, result)

        # Emit TransformCompleted telemetry for all buffered tokens
        for token in buffered_tokens:
            self._emit_transform_completed(token=token, transform=transform, transform_result=result)

        # Out-of-claim (timeout/EOF) flush: ONE atomic journal transition
        # consumes the buffered BLOCKED rows and makes every sink-bound flush
        # output journal-durable (PENDING_SINK) before the in-process sink
        # write runs (F1/D6). The returned flush_results (tagged
        # ``scheduler_pending_sink=True`` by ``_complete_aggregation_flush``)
        # feed the in-process sink write; the post-sink callback terminalizes
        # the emitted rows.
        if settings.output_mode == OutputMode.PASSTHROUGH:
            flush_results, child_items = self._route_passthrough_results(fctx, result)
            flush_results, _pending_sink_token_ids = self._complete_aggregation_flush(node_id, flush_results, buffered_tokens)
            return flush_results, child_items
        if settings.output_mode == OutputMode.TRANSFORM:
            flush_results, child_items = self._route_transform_results(fctx, result)
            flush_results, _pending_sink_token_ids = self._complete_aggregation_flush(node_id, flush_results, buffered_tokens)
            return flush_results, child_items
        raise OrchestrationInvariantError(f"Unknown output_mode: {settings.output_mode}")

    def _process_batch_aggregation_node(
        self,
        transform: TransformProtocol,
        current_token: TokenInfo,
        ctx: PluginContext,
        child_items: list[WorkItem],
        coalesce_node_id: NodeID | None = None,
        coalesce_name: CoalesceName | None = None,
    ) -> tuple[RowResult | tuple[RowResult, ...], list[WorkItem]]:
        """Process a row at an aggregation node using engine buffering.

        Engine buffers rows and calls transform.process(rows: list[dict])
        when the trigger fires. Flush handling is delegated to shared helpers
        (_handle_flush_error, _route_passthrough_results, _route_transform_results).

        TEMPORAL DECOUPLING:

        Both modes now record BUFFERED (non-terminal) at buffer time, with
        terminal outcomes deferred to flush time. This enables per-token
        QUARANTINED recording when batch transforms quarantine individual rows.

        - **Landscape (audit trail)**: Records BUFFERED at buffer time.
          Terminal outcome (CONSUMED_IN_BATCH, QUARANTINED, FAILED) at flush.

        - **Telemetry (observability)**: Emits TokenCompleted at flush time.
          Deferred to maintain ordering invariant (TransformCompleted before
          TokenCompleted for each token).

        Args:
            transform: The batch-aware transform
            current_token: Current row token
            ctx: Plugin context
            child_items: Work items to return with result
            coalesce_node_id: Node ID at which fork children should coalesce (optional)
            coalesce_name: Name of the coalesce point for merging (optional)

        Returns:
            (RowResult or list[RowResult], child_items) tuple
            - Single RowResult for non-flush buffering
            - List of RowResults for flush (passthrough or transform mode)
        """
        raw_node_id = transform.node_id
        if raw_node_id is None:
            raise OrchestrationInvariantError("Node ID is None during edge resolution")
        node_id = NodeID(raw_node_id)
        if node_id not in self._aggregation_settings:
            raise OrchestrationInvariantError(f"Batch-aware transform {transform.name!r} at node {node_id!r} has no aggregation settings.")

        # ADR-030 §E.2 (slice 3, THE owned flush-timing change): record NOTHING
        # durable in-claim. The arriving token returns a (None, BUFFERED)
        # RowResult unconditionally; the drain marks its journal row BLOCKED
        # (the durable arrival), and the NEXT drain iteration's journal-first
        # intake adopts it (batch_members + BUFFERED token_outcome inside the
        # fenced adoption transaction, backdated to barrier_blocked_at) and
        # feeds executor memory. Count/condition triggers fire from that
        # intake step — the in-claim flush arm is deleted; the trigger fire
        # time is invariant under takeover (§H 476).
        #
        # The live token is stashed so the intake feeds the executor the exact
        # post-transform payload the old in-claim accept used (N=1 parity);
        # the barrier_key rides the same stash for the drain's mark_blocked
        # (the historical derivation read the in-claim BUFFERED outcome's
        # batch_id, which no longer exists at block time).
        #
        # NOTE: Do NOT emit TokenCompleted telemetry here!
        # TokenCompleted must be deferred to flush time so that
        # TransformCompleted can be emitted first.
        self._live_barrier_holds[current_token.token_id] = _LiveBarrierHold(
            token=current_token,
            barrier_key=str(node_id),
        )
        return (
            RowResult(
                token=current_token,
                final_data=current_token.row_data,
                outcome=None,
                path=TerminalPath.BUFFERED,
            ),
            child_items,
        )

    def _convert_retryable_to_error_result(
        self,
        exc: Exception,
        transform: Any,
        token: TokenInfo,
        ctx: Any,
        reason: TransformErrorCategory,
        *,
        retryable: bool = True,
    ) -> tuple[TransformResult, TokenInfo, str | None]:
        """Convert a retryable exception to a TransformResult.error when no retry manager is configured.

        Shared handler for PluginRetryableError (retryable) and transient exceptions
        (ConnectionError, TimeoutError, OSError, CapacityError). Records the
        error in the audit trail and emits a DIVERT routing event if on_error
        routes to a sink.
        """
        on_error = transform.on_error
        # on_error is always set (required by TransformSettings) — Tier 1 invariant
        if on_error is None:
            raise OrchestrationInvariantError(
                f"Transform '{transform.name}' has on_error=None — this should be impossible since TransformSettings requires on_error"
            )

        error_details: TransformErrorReason = {"reason": reason, "error": str(exc)}
        # Shared error-audit routine (elspeth-aeb0a8f756): identical
        # transform_error + DIVERT routing_event recording as the executor's
        # error-result branch. Here the guard already auto-failed the state on
        # exception exit, so recording happens after completion by design —
        # the helper is sequencing-agnostic. ctx.state_id was set by
        # TransformExecutor.execute_transform before the exception propagated.
        record_transform_error_with_routing(
            ctx=ctx,
            execution=self._execution,
            error_edge_ids=self._error_edge_ids,
            state_id=ctx.state_id,
            token=token,
            transform=transform,
            row=token.row_data,
            error_details=error_details,
            on_error=on_error,
        )

        return (
            TransformResult.error(error_details, retryable=retryable),
            token,
            on_error,
        )

    def _execute_transform_with_retry(
        self,
        transform: Any,
        token: TokenInfo,
        ctx: PluginContext,
        *,
        attempt_offset: int = 0,
    ) -> tuple[TransformResult, TokenInfo, str | None]:
        """Execute transform with optional retry for transient failures.

        Retry behavior:
        - If retry_manager is None: single attempt, no retry
        - If retry_manager is set: retry on transient exceptions

        Each attempt is recorded separately in the audit trail with attempt number.

        Note: TransformResult.error() is NOT retried - that's a processing error,
        not a transient failure. Only exceptions trigger retry.

        Resume state (attempt offset and checkpoint provenance) is read from the
        token itself (token.resume_attempt_offset, token.resume_checkpoint_id) and
        flows through to execute_transform. No explicit threading needed.

        Args:
            transform: Transform to execute
            token: Current token; token.resume_attempt_offset and
                token.resume_checkpoint_id carry the resume state.
            ctx: Plugin context

        Returns:
            Tuple of (TransformResult, updated TokenInfo, error_sink)
        """
        if self._retry_manager is None:
            # No retry configured - single attempt
            # Must still catch retryable exceptions and convert to error results
            # to keep failures row-scoped (don't abort entire run)
            try:
                return self._transform_executor.execute_transform(
                    transform=transform,
                    token=token,
                    ctx=ctx,
                    attempt=attempt_offset,
                )
            except InterruptedError as e:
                return self._convert_retryable_to_error_result(
                    e,
                    transform,
                    token,
                    ctx,
                    reason="shutdown_requested",
                    retryable=False,
                )
            except PluginRetryableError as e:
                return self._convert_retryable_to_error_result(
                    e,
                    transform,
                    token,
                    ctx,
                    reason="transient_error_no_retry" if e.retryable else "permanent_error",
                )
            except (ConnectionError, TimeoutError, OSError, CapacityError) as e:
                return self._convert_retryable_to_error_result(
                    e,
                    transform,
                    token,
                    ctx,
                    reason="transient_error_no_retry",
                )

        # Track attempt number for audit
        attempt_tracker = {"current": attempt_offset}

        def execute_attempt() -> tuple[TransformResult, TokenInfo, str | None]:
            attempt = attempt_tracker["current"]
            attempt_tracker["current"] += 1
            return self._transform_executor.execute_transform(
                transform=transform,
                token=token,
                ctx=ctx,
                attempt=attempt,
            )

        def is_retryable(e: BaseException) -> bool:
            if isinstance(e, InterruptedError):
                return False
            if isinstance(e, PluginRetryableError):
                return e.retryable
            return isinstance(e, ConnectionError | TimeoutError | OSError | CapacityError)

        try:
            return self._retry_manager.execute_with_retry(
                operation=execute_attempt,
                is_retryable=is_retryable,
                shutdown_event=ctx.shutdown_event,
            )
        except InterruptedError as e:
            return self._convert_retryable_to_error_result(
                e,
                transform,
                token,
                ctx,
                reason="shutdown_requested",
                retryable=False,
            )

    def _record_source_node_state(
        self,
        *,
        token: TokenInfo,
        input_data: dict[str, object],
        status: NodeStateStatus,
        source_node_id: NodeID | None = None,
        error: ExecutionError | None = None,
    ) -> None:
        """Record the source node state for a token.

        Source "processing" already happened in the plugin iterator, so the
        state is recorded immediately as COMPLETED or FAILED with duration 0.
        """
        effective_source_node_id = source_node_id or self._source_node_id
        if status == NodeStateStatus.COMPLETED:
            self._execution.record_completed_node_state(
                token_id=token.token_id,
                node_id=effective_source_node_id,
                run_id=self._run_id,
                step_index=0,
                input_data=input_data,
                output_data=input_data,
                duration_ms=0,
            )
            return
        source_state = self._execution.begin_node_state(
            token_id=token.token_id,
            node_id=effective_source_node_id,
            run_id=self._run_id,
            step_index=0,
            input_data=input_data,
        )
        if status == NodeStateStatus.FAILED:
            self._execution.complete_node_state(
                state_id=source_state.state_id,
                status=NodeStateStatus.FAILED,
                duration_ms=0,
                error=error,
            )
            return
        raise OrchestrationInvariantError(f"Source node states may only be recorded as COMPLETED or FAILED, not {status!r}.")

    def _record_source_boundary_failure(
        self,
        *,
        token: TokenInfo,
        input_data: dict[str, object],
        failure: _SourceBoundaryFailure,
        source_node_id: NodeID | None = None,
    ) -> None:
        """Record terminal audit evidence for a source-boundary failure.

        Source boundary validation runs after token creation so the failure
        can use the real row/token identity. Because the failure happens before
        DAG traversal begins, the processor must record BOTH the terminal token
        outcome and the FAILED source node state before re-raising the Tier 1
        exception. If either audit write fails, raise ``AuditIntegrityError`` so
        the recorder failure outranks the original failure.

        AuditEvidenceBase exceptions contribute structured context via
        ``to_audit_dict()``; framework/orchestration bugs are still recorded
        as FAILED outcomes and node states, but without a fabricated context
        payload. ``TokenCompleted`` telemetry is emitted only after both audit
        writes succeed. Telemetry is operational visibility, not part of the
        source-boundary audit pair; telemetry failures are logged and never
        outrank the original failure or a recorder failure.
        """
        effective_source_node_id = source_node_id or self._source_node_id
        audit_context = failure.to_audit_dict() if isinstance(failure, AuditEvidenceBase) else None
        original_label = (
            "violation"
            if isinstance(
                failure,
                (
                    DeclarationContractViolation,
                    AggregateDeclarationContractViolation,
                    PluginContractViolation,
                ),
            )
            else "failure"
        )
        error_hash = compute_error_hash(f"{type(failure).__name__}:{effective_source_node_id}")
        try:
            self._data_flow.record_token_outcome(
                ref=TokenRef(token_id=token.token_id, run_id=self._run_id),
                outcome=TerminalOutcome.FAILURE,
                path=TerminalPath.UNROUTED,
                error_hash=error_hash,
                context=audit_context,
            )
        except LandscapeRecordError as record_failure:
            raise AuditIntegrityError(
                f"Failed to record {type(failure).__name__} FAILED outcome for token {token.token_id!r} "
                f"on source boundary (node={effective_source_node_id!r}). Audit trail is INCOMPLETE — "
                f"the FAILED token outcome may be missing. Recorder failure: "
                f"{type(record_failure).__name__}: {record_failure}. Original {original_label}: {failure!s}"
            ) from record_failure
        try:
            self._record_source_node_state(
                token=token,
                input_data=input_data,
                status=NodeStateStatus.FAILED,
                source_node_id=effective_source_node_id,
                error=ExecutionError(
                    exception=str(failure),
                    exception_type=type(failure).__name__,
                    phase="source_boundary_check",
                    context=audit_context,
                ),
            )
        except LandscapeRecordError as record_failure:
            raise AuditIntegrityError(
                f"Failed to record FAILED source node state for token {token.token_id!r} "
                f"on source boundary (node={effective_source_node_id!r}). Audit trail is INCOMPLETE — "
                f"the FAILED source node state may be missing. Recorder failure: "
                f"{type(record_failure).__name__}: {record_failure}. Original {original_label}: {failure!s}"
            ) from record_failure
        with best_effort(
            "TokenCompleted telemetry after source-boundary FAILED audit",
            run_id=self._run_id,
            token_id=token.token_id,
            source_node_id=effective_source_node_id,
        ):
            self._emit_token_completed(
                token,
                outcome=TerminalOutcome.FAILURE,
                path=TerminalPath.UNROUTED,
            )

    def _record_source_and_start_traversal(
        self,
        token: TokenInfo,
        input_data: dict[str, object],
        transforms: Sequence[Any],
        ctx: PluginContext,
        *,
        source_node_id: NodeID | None = None,
        source_on_success: str | None = None,
        coalesce_node_id: NodeID | None,
        coalesce_name: CoalesceName | None,
    ) -> list[RowResult]:
        """Record source node_state and start pipeline traversal.

        Implementation for process_existing_row (the resume re-drive path);
        process_row inlines the equivalent sequence so the fenced ingest can
        journal the initial cursor in one transaction (ADR-030 §C.4 row 9).
        Records the source node as immediately COMPLETED (duration_ms=0)
        since source "processing" already happened in the plugin iterator.

        Args:
            token: Token for the row being processed
            input_data: Row data dict for audit hashing (must be plain dict)
            transforms: List of transform plugins (for invariant check)
            ctx: Plugin context
            source_on_success: Source-specific terminal sink for rows that do
                not traverse any processing nodes. Defaults to the processor's
                configured source sink for single-source callers.
            coalesce_node_id: Node ID at which fork children should coalesce
            coalesce_name: Name of the coalesce point for merging

        Returns:
            List of RowResults, one per terminal token
        """
        effective_source_node_id = source_node_id or self._source_node_id
        self._record_source_node_state(
            token=token,
            input_data=input_data,
            status=NodeStateStatus.COMPLETED,
            source_node_id=effective_source_node_id,
        )

        effective_source_on_success = source_on_success if source_on_success is not None else self._source_on_success
        return self._drain_work_queue(
            self._initial_work_item_for_source_token(
                token=token,
                transforms=transforms,
                source_node_id=effective_source_node_id,
                source_on_success=effective_source_on_success,
                coalesce_node_id=coalesce_node_id,
                coalesce_name=coalesce_name,
            ),
            ctx,
        )

    def _initial_work_item_for_source_token(
        self,
        *,
        token: TokenInfo,
        transforms: Sequence[Any],
        source_node_id: NodeID,
        source_on_success: str,
        coalesce_node_id: NodeID | None,
        coalesce_name: CoalesceName | None,
    ) -> WorkItem:
        """Resolve the source continuation and build the initial WorkItem.

        Shared by ``process_row`` (which needs the WorkItem BEFORE the fenced
        ingest so the composed transaction can journal the initial cursor)
        and ``_record_source_and_start_traversal`` (the resume/existing-row
        path). Per ADR-025 §2 the DAG builder always populates node_to_next
        for every source node — missing entries are a construction bug, not
        a state we silently work around with a "first transform" fallback.
        """
        if source_node_id not in self._node_to_next:
            raise OrchestrationInvariantError(
                f"Traversal context is missing source continuation for {source_node_id!r}. "
                "This is a graph construction bug — every source node must have a node_to_next entry."
            )
        initial_node_id = self._node_to_next[source_node_id]
        if transforms and initial_node_id == source_node_id:
            raise OrchestrationInvariantError("Traversal context is missing a source continuation for non-empty transform pipeline")
        return self._nav.create_work_item(
            token=token,
            current_node_id=initial_node_id,
            coalesce_node_id=coalesce_node_id,
            coalesce_name=coalesce_name,
            on_success_sink=source_on_success if initial_node_id is None else None,
        )

    def _ingest_source_row_with_initial_claim(
        self,
        *,
        item: WorkItem,
        source_node_id: NodeID,
        row_index: int,
        source_row_index: int,
        ingest_sequence: int,
        data: Mapping[str, object],
    ) -> TokenWorkItem:
        """Drive the fenced leader INGEST verb for one source row (§C.4 row 9).

        Composes the rows+tokens inserts (via the data-flow repository's
        connection-accepting closure) with the initial enqueue-and-claim in
        ONE epoch-fenced IMMEDIATE transaction. Field derivation mirrors
        ``_enqueue_scheduler_work_item`` exactly (deterministic
        work_item_id + strict field equality reconciliation downstream).
        """
        coordination_token = self._coordination_token
        if coordination_token is None:
            raise OrchestrationInvariantError(
                "Fenced source ingest requires a coordination token; the unfenced arm must not reach this helper."
            )
        token = item.token
        now = self._clock.now_utc()

        def insert_row_and_token(conn: Connection) -> tuple[AuditRow, AuditToken]:
            return self._data_flow.insert_row_with_token_on(
                conn,
                run_id=self._run_id,
                source_node_id=str(source_node_id),
                row_index=row_index,
                data=data,
                source_row_index=source_row_index,
                ingest_sequence=ingest_sequence,
                row_id=token.row_id,
                token_id=token.token_id,
            )

        _row, _token_record, scheduled = self._scheduler.ingest_row_with_initial_claim(
            coordination_token=coordination_token,
            now=now,
            insert_row_and_token=insert_row_and_token,
            token_id=token.token_id,
            row_id=token.row_id,
            node_id=self._scheduler_node_id(item.current_node_id),
            step_index=self._scheduler_step_index(item.current_node_id),
            ingest_sequence=ingest_sequence,
            row_payload_json=self._scheduler.serialize_row_payload(token.row_data),
            lease_owner=self._scheduler_lease_owner,
            lease_seconds=self._scheduler_lease_seconds,
            queue_key=self._queue_key_for_blocked_item(item),
            barrier_key=self._barrier_key_for_blocked_item(item),
            on_success_sink=item.on_success_sink,
            branch_name=token.branch_name,
            fork_group_id=token.fork_group_id,
            join_group_id=token.join_group_id,
            expand_group_id=token.expand_group_id,
            coalesce_node_id=str(item.coalesce_node_id) if item.coalesce_node_id is not None else None,
            coalesce_name=str(item.coalesce_name) if item.coalesce_name is not None else None,
        )
        return scheduled

    def process_row(
        self,
        row_index: int,
        source_row: SourceRow,
        transforms: Sequence[Any],
        ctx: PluginContext,
        *,
        coalesce_node_id: NodeID | None = None,
        coalesce_name: CoalesceName | None = None,
        source_node_id: NodeID | None = None,
        source_plugin: SourceProtocol | None = None,
        source_on_success: str | None = None,
        source_row_index: int,
        ingest_sequence: int,
    ) -> list[RowResult]:
        """Process a row through all transforms.

        Uses a work queue to handle fork operations - when a fork creates
        child tokens, they are added to the queue and processed through
        the remaining transforms.

        Args:
            row_index: Position in source
            source_row: SourceRow from source (must have contract)
            transforms: List of transform plugins
            ctx: Plugin context
            coalesce_node_id: Node ID at which fork children should coalesce
            coalesce_name: Name of the coalesce point for merging
            source_on_success: Source-specific terminal sink. Required by
                multi-source callers when a source row may terminalize without
                entering a transform.
            source_row_index: Position within the active source stream. Required
                to preserve per-source row identity in the audit trail.
            ingest_sequence: Global ingest-order position across all sources.
                Required to preserve deterministic resume ordering.

        Returns:
            List of RowResults, one per terminal token (parent + children)
        """
        effective_source_node_id = source_node_id or self._source_node_id
        # Pre-mint the row/token identities (client-generated ids) so the
        # source-boundary checks can run BEFORE any durable write (ADR-030
        # §C.4 row 9): a boundary-failed row keeps its rows+tokens audit
        # record but gets NO scheduler row; the happy path composes rows +
        # tokens + initial enqueue into ONE fenced IMMEDIATE transaction.
        if source_row.contract is None:
            raise OrchestrationInvariantError(
                "SourceRow must have contract to create token. Source plugins must set contract on all valid rows."
            )
        pipeline_row = source_row.to_pipeline_row()
        token = TokenInfo(
            row_id=generate_id(),
            token_id=generate_id(),
            row_data=pipeline_row,
        )

        # Valid SourceRows always carry mapping-shaped row payloads; once the
        # row enters the processor we treat the values as opaque objects.
        source_input = cast(dict[str, object], source_row.row)
        effective_source_plugin = source_plugin or self._source_plugin
        if effective_source_plugin is not None:
            try:
                run_boundary_checks(
                    inputs=BoundaryInputs(
                        plugin=effective_source_plugin,
                        node_id=str(effective_source_node_id),
                        run_id=self._run_id,
                        row_id=token.row_id,
                        token_id=token.token_id,
                        static_contract=effective_source_plugin.declared_guaranteed_fields,
                        row_data=source_input,
                        row_contract=source_row.contract,
                    ),
                    outputs=BoundaryOutputs(),
                )
            except (
                DeclarationContractViolation,
                AggregateDeclarationContractViolation,
                PluginContractViolation,
                FrameworkBugError,
                OrchestrationInvariantError,
            ) as failure:
                # The failed row still gets its durable rows+tokens audit
                # record (fenced when a coordination token is held) — but
                # never a scheduler row.
                self._token_manager.create_initial_token(
                    run_id=self._run_id,
                    source_node_id=effective_source_node_id,
                    row_index=row_index,
                    source_row_index=source_row_index,
                    ingest_sequence=ingest_sequence,
                    source_row=source_row,
                    row_id=token.row_id,
                    token_id=token.token_id,
                    coordination_token=self._coordination_token,
                )
                self._record_source_boundary_failure(
                    token=token,
                    input_data=source_input,
                    failure=failure,
                    source_node_id=effective_source_node_id,
                )
                raise

        effective_source_on_success = source_on_success if source_on_success is not None else self._source_on_success
        initial_item = self._initial_work_item_for_source_token(
            token=token,
            transforms=transforms,
            source_node_id=effective_source_node_id,
            source_on_success=effective_source_on_success,
            coalesce_node_id=coalesce_node_id,
            coalesce_name=coalesce_name,
        )

        preclaimed: TokenWorkItem | None = None
        if self._coordination_token is not None:
            # Fenced leader INGEST (§C.4 row 9): rows insert + tokens insert
            # + initial enqueue-and-claim in ONE IMMEDIATE transaction; a
            # stale epoch rolls the whole ingest back (no orphan rows row).
            preclaimed = self._ingest_source_row_with_initial_claim(
                item=initial_item,
                source_node_id=effective_source_node_id,
                row_index=row_index,
                source_row_index=source_row_index,
                ingest_sequence=ingest_sequence,
                data=pipeline_row.to_dict(),
            )
        else:
            # Legacy unfenced arm (direct repository-level construction, no
            # coordination token): rows+tokens in their own transaction; the
            # drain performs the initial enqueue as before.
            self._token_manager.create_initial_token(
                run_id=self._run_id,
                source_node_id=effective_source_node_id,
                row_index=row_index,
                source_row_index=source_row_index,
                ingest_sequence=ingest_sequence,
                source_row=source_row,
                row_id=token.row_id,
                token_id=token.token_id,
            )

        self._record_source_node_state(
            token=token,
            input_data=source_input,
            status=NodeStateStatus.COMPLETED,
            source_node_id=effective_source_node_id,
        )
        return self._drain_work_queue(initial_item, ctx, preclaimed=preclaimed)

    def process_existing_row(
        self,
        row_id: str,
        row_data: PipelineRow,
        transforms: Sequence[Any],
        ctx: PluginContext,
        *,
        coalesce_node_id: NodeID | None = None,
        coalesce_name: CoalesceName | None = None,
        source_node_id: NodeID | None = None,
        source_on_success: str | None = None,
    ) -> list[RowResult]:
        """Process an existing row (row already in database, create new token only).

        Used during resume when rows were created in the original run
        but need to be reprocessed. Unlike process_row(), this does NOT
        create a new row record - only a new token.

        Resume intentionally does NOT re-run source-boundary contracts here.
        The resumed row payload already crossed the source boundary in the
        original run, and resume replays persisted ``PipelineRow`` payloads
        through ``NullSource`` rather than reopening the original source
        plugin. The resume path therefore inherits source-boundary evidence
        from the original run and must verify runtime-VAL manifest equality
        before any resumed rows are loaded.

        Args:
            row_id: Existing row ID in the database
            row_data: Row data (retrieved from payload store)
            transforms: List of transform plugins
            ctx: Plugin context
            coalesce_node_id: Node ID at which fork children should coalesce
            coalesce_name: Name of the coalesce point for merging
            source_node_id: Source node that originally ingested this row.
                Multi-source resume must pass this so replayed source states
                remain attributable to the correct root.
            source_on_success: Source-specific terminal sink for rows that do
                not traverse any processing nodes.

        Returns:
            List of RowResults, one per terminal token (parent + children)
        """
        # Create token for existing row (NOT a new row)
        token = self._token_manager.create_token_for_existing_row(
            row_id=row_id,
            row_data=row_data,
        )

        # The row already exists from the original run, but this new token
        # needs its own source state for complete audit lineage.
        resumed_input = row_data.to_dict()
        return self._record_source_and_start_traversal(
            token=token,
            input_data=resumed_input,
            transforms=transforms,
            ctx=ctx,
            source_node_id=source_node_id,
            source_on_success=source_on_success,
            coalesce_node_id=coalesce_node_id,
            coalesce_name=coalesce_name,
        )

    def _terminal_coalesce_row_result(
        self,
        token: TokenInfo,
        coalesce_name: CoalesceName,
        *,
        context: str,
    ) -> RowResult:
        """Build the terminal-coalesce RowResult (SUCCESS/COALESCED routed to the coalesce sink).

        Single source of truth for the three terminal-coalesce sites (barrier-fire in
        _maybe_coalesce_token, lost-branch in _notify_coalesce_of_lost_branch, and resume
        re-drive in resume_incomplete_token) so the audit RowResult shape cannot drift between them.

        This constructs ONLY the RowResult — it does NOT emit telemetry or record outcomes.
        Each call site retains its own telemetry handling (e.g. _notify_coalesce_of_lost_branch
        deliberately does not emit TokenCompleted here, deferring to accumulate_row_outcomes).
        """
        sink_name = self._nav.resolve_coalesce_sink(coalesce_name, context=context)
        return RowResult(
            token=token,
            final_data=token.row_data,
            outcome=TerminalOutcome.SUCCESS,
            path=TerminalPath.COALESCED,
            sink_name=sink_name,
        )

    def process_token(
        self,
        token: TokenInfo,
        ctx: PluginContext,
        *,
        current_node_id: NodeID | None,
        coalesce_node_id: NodeID | None = None,
        coalesce_name: CoalesceName | None = None,
    ) -> list[RowResult]:
        """Process an existing token through the pipeline starting at current_node_id.

        current_node_id=None is valid only when sink routing is explicit: either the
        token has a branch_name present in _branch_to_sink, or on_success_sink is set
        (via an inherited WorkItem). _process_single_token enforces this invariant and
        raises OrchestrationInvariantError if neither is satisfied. Used for mid-pipeline
        coalesce merges that must continue processing, and for resume of fork→sink tokens.
        """
        return self._drain_work_queue(
            self._nav.create_work_item(
                token=token,
                current_node_id=current_node_id,
                coalesce_node_id=coalesce_node_id,
                coalesce_name=coalesce_name,
            ),
            ctx,
        )

    def _resolve_step_node(self, spec: IncompleteTokenSpec) -> NodeID:
        """Map an incomplete token's step_in_pipeline back to the NodeID that created it.

        _node_step_map is a bijection (unique monotonic step per node assigned by
        build_step_map via enumerate(..., start=1)), so the inverse is well-defined.
        Used to find the expand/coalesce node so the re-drive can continue from the
        node AFTER it (via resolve_next_node).

        Raises:
            OrchestrationInvariantError: If no node maps to spec.step_in_pipeline.
                Indicates audit/DAG inconsistency — step was persisted for a token
                but the current DAG has no node at that step position.
        """
        target_step = spec.step_in_pipeline
        if target_step is None:
            raise OrchestrationInvariantError(
                f"Incomplete token {spec.token_id} has step_in_pipeline=None — "
                "cannot resolve node ID for mid-DAG resume. Audit/DAG inconsistency."
            )
        for node_id, step in self._node_step_map.items():
            if step == target_step:
                return node_id
        raise OrchestrationInvariantError(
            f"No node maps to step_in_pipeline={target_step} for incomplete token "
            f"{spec.token_id} — _node_step_map has no such step. Audit/DAG inconsistency."
        )

    def resume_incomplete_token(
        self,
        spec: IncompleteTokenSpec,
        row_data: PipelineRow,
        ctx: PluginContext,
        *,
        resume_checkpoint_id: str,
    ) -> list[RowResult]:
        """Drive one reconstructed incomplete child token to completion in place.

        Reuses the persisted token id (continuing under the ORIGINAL parent) and re-drives
        from the correct mid-DAG node. The TokenInfo carries resume_attempt_offset =
        spec.max_attempt + 1 and resume_checkpoint_id, so every node_state it writes is at
        the bumped attempt and stamped with provenance (ADDENDUM 4 — carried on the token,
        NOT passed as params to process_token).

        Dispatch cases (expand_group_id checked first because expanded children inherit
        their parent's branch_name; their persisted step still identifies the expand node):

        1. expand child: expand_group_id set → re-drive from the node AFTER the expand node.
        2. fork → sink terminal branch: branch_name in _branch_to_sink → current_node_id=None
           (process_token's None-path routes via branch_to_sink to the terminal sink).
        3. fork → coalesce, crashed before barrier: branch_name in _branch_to_coalesce →
           re-run the branch from its first processing node with coalesce context.
        4. post-coalesce merged token, crashed after barrier (B1 review finding): join_group_id
           set AND fork_group_id None AND branch_name None →
           - Non-terminal coalesce (next node exists): process_token from node after coalesce.
           - Terminal coalesce (no next node): reconstruct the COALESCED RowResult directly,
             mirroring _maybe_coalesce_token's terminal-coalesce path (the correct routing
             mechanism is resolve_coalesce_sink; process_token(None) is NOT valid for a
             branchless merged token without on_success_sink context).

        Raises:
            OrchestrationInvariantError: If the token's lineage fields do not match any
                known resume-start pattern — indicates audit/DAG inconsistency.
        """
        token = TokenInfo(
            row_id=spec.row_id,
            token_id=spec.token_id,
            row_data=row_data,
            branch_name=spec.branch_name,
            fork_group_id=spec.fork_group_id,
            join_group_id=spec.join_group_id,
            expand_group_id=spec.expand_group_id,
            resume_attempt_offset=spec.max_attempt + 1,
            resume_checkpoint_id=resume_checkpoint_id,
        )
        branch = spec.branch_name

        if spec.expand_group_id is not None:
            # expand child: re-drive from the node AFTER the expand node.
            # Expanded children inherit branch_name from fork branches, including
            # coalesce-bound branches, so this must run before branch dispatch.
            # expand is never terminal; an `after` of None here is an audit/DAG inconsistency
            # that process_token's None-enforcement raises on (no branch_to_sink / on_success_sink).
            after = self._nav.resolve_next_node(self._resolve_step_node(spec))
            return self.process_token(token, ctx, current_node_id=after)

        if branch is not None and BranchName(branch) in self._branch_to_sink:
            # fork → sink terminal branch: straight to the sink via None-path routing.
            return self.process_token(token, ctx, current_node_id=None)

        if branch is not None and BranchName(branch) in self._branch_to_coalesce:
            # fork → coalesce, crashed BEFORE the barrier: re-run the branch from its
            # first node with coalesce context so _maybe_coalesce_token fires at the barrier.
            coalesce_name = self._branch_to_coalesce[BranchName(branch)]
            first_node = self._nav.resolve_branch_first_node(branch)
            return self.process_token(
                token,
                ctx,
                current_node_id=first_node,
                coalesce_name=coalesce_name,
            )

        if spec.join_group_id is not None and spec.fork_group_id is None and branch is None:
            # post-coalesce merged token, crashed AFTER the barrier (B1 review finding):
            # step_in_pipeline is the coalesce node's step. Re-drive downstream of the
            # coalesce node, or reconstruct the terminal COALESCED RowResult if the coalesce
            # was terminal (no next node exists).
            coalesce_node_id = self._resolve_step_node(spec)
            after = self._nav.resolve_next_node(coalesce_node_id)
            if after is not None:
                return self.process_token(token, ctx, current_node_id=after)
            # Terminal coalesce: no downstream processing nodes.
            # process_token(current_node_id=None) is NOT valid for a branchless merged token
            # (no branch_to_sink entry, no on_success_sink). Mirror _maybe_coalesce_token's
            # terminal-coalesce path: resolve the sink and return the COALESCED RowResult
            # directly for the caller (orchestrator) to route to sink.
            #
            # _resolve_step_node guarantees coalesce_node_id is in _node_step_map but NOT
            # that it is in _coalesce_name_by_node_id — wrap the lookup so a mismatch is an
            # uncontexted-KeyError-free audit-grade invariant failure.
            try:
                coalesce_name = self._coalesce_name_by_node_id[coalesce_node_id]
            except KeyError as exc:
                raise OrchestrationInvariantError(
                    f"Post-coalesce token {spec.token_id} resolved to node {coalesce_node_id!r} "
                    f"which is not a known coalesce node (known: {sorted(self._coalesce_name_by_node_id)}). "
                    f"Audit/DAG inconsistency."
                ) from exc
            return [
                self._terminal_coalesce_row_result(
                    token,
                    coalesce_name,
                    context=f"terminal coalesce resume for incomplete token '{spec.token_id}'",
                )
            ]

        raise OrchestrationInvariantError(
            f"Incomplete token {spec.token_id} has branch_name={branch!r}, "
            f"fork_group_id={spec.fork_group_id!r}, join_group_id={spec.join_group_id!r}, "
            f"expand_group_id={spec.expand_group_id!r} — no resume-start node resolvable. "
            f"Audit/DAG inconsistency."
        )

    def _maybe_coalesce_token(
        self,
        current_token: TokenInfo,
        *,
        current_node_id: NodeID,
        coalesce_node_id: NodeID | None,
        coalesce_name: CoalesceName | None,
        child_items: list[WorkItem],
    ) -> tuple[bool, RowResult | None]:
        # Structural guard: only handle when we're actually at the coalesce node
        # for this branch token's assigned coalesce point.
        if current_token.branch_name is None or coalesce_name is None or coalesce_node_id is None or current_node_id != coalesce_node_id:
            return False, None

        # ADR-030 §B (slice 5, follower barrier hand-off): a follower has no
        # in-process CoalesceExecutor.  When a follower arrives at a coalesce
        # node it must mark_blocked (durable barrier hold, no in-memory accept)
        # so the leader's next journal-intake adopts the arrival and runs the
        # trigger evaluation (§B.2: trigger evaluation is leader-only).
        # Returning (True, None) with no child items triggers the
        # `result is None and not child_items` arm of _drain_scheduler_claims
        # which calls mark_blocked (§E.2).
        if self._coalesce_executor is None:
            logger.debug(
                "follower: coalesce barrier hold for token %r at node %r (coalesce=%r) — marking blocked; leader adopts via journal-intake",
                current_token.token_id,
                current_node_id,
                coalesce_name,
            )
            return True, None

        # ADR-030 §E.2 (slice 3, journal-first barrier acceptance): the
        # arriving branch token is NOT accepted in-claim. It is held
        # unconditionally — the drain marks its journal row BLOCKED under
        # barrier_key=coalesce_name (the durable arrival) and the NEXT drain
        # iteration's intake adopts the row (fenced CAS marker; the coalesce
        # arm's durable payload is the marker alone) and runs the executor
        # accept with the backdated arrival time. Merge fires, group failures
        # and late-arrival releases (§E.3a) all surface from that intake step.
        # The live token is stashed so intake feeds the executor the exact
        # post-transform token the old in-claim accept used (N=1 parity).
        self._live_barrier_holds[current_token.token_id] = _LiveBarrierHold(
            token=current_token,
            barrier_key=str(coalesce_name),
        )
        return True, None

    def _notify_coalesce_of_lost_branch(
        self,
        current_token: TokenInfo,
        reason: str,
        child_items: list[WorkItem],
    ) -> list[RowResult]:
        """Notify the coalesce executor that a forked branch was diverted.

        Called when a forked token exits the pipeline early (error-routed,
        quarantined, or failed). The coalesce executor re-evaluates merge
        conditions and may trigger an immediate merge or failure for held
        sibling tokens.

        Args:
            current_token: The forked token being diverted
            reason: Machine-readable reason for the diversion
            child_items: Mutable work queue — merged tokens are appended here

        Returns:
            List of RowResults for sibling tokens that failed as a consequence
            of the branch loss, or a COALESCED RowResult if the merge triggered
            at a terminal coalesce step. Empty if no consequences yet.
        """
        if current_token.branch_name is None:
            return []

        branch_name = BranchName(current_token.branch_name)
        if branch_name not in self._branch_to_coalesce:
            return []
        coalesce_name = self._branch_to_coalesce[branch_name]

        coalesce_node_id = self._coalesce_node_ids[coalesce_name]
        # §E.5 record-then-notify: stage the durable loss record BEFORE the
        # in-memory notify, and UNCONDITIONALLY (regardless of whether this
        # worker has a coalesce_executor). A follower has no in-process
        # CoalesceExecutor, but it MUST still write the durable branch-loss row
        # in the same transaction as its mark_failed/divert so the leader's
        # next journal-intake can see the loss (design §E.5:366-374).
        # The spec rides the branch token's own disposition transaction (the
        # drain's mark_failed / mark_pending_sink / mark_terminal for the
        # claimed token, or the flush's complete_barrier for empty-emission
        # losses) — committed iff the disposition commits, idempotent on the
        # natural key.
        self._pending_branch_losses.append(
            BranchLossSpec(
                coalesce_name=str(coalesce_name),
                row_id=current_token.row_id,
                branch_name=str(branch_name),
                token_id=current_token.token_id,
                reason=reason,
                recorded_by=self._scheduler_lease_owner,
            )
        )

        # In-memory notify: only possible when this worker has a coalesce
        # executor (leader).  Followers skip the in-memory notify; the durable
        # record above is sufficient for the leader's next intake.
        if self._coalesce_executor is None:
            return []

        outcome = self._coalesce_executor.notify_branch_lost(
            coalesce_name=coalesce_name,
            row_id=current_token.row_id,
            lost_branch=current_token.branch_name,
            reason=reason,
        )

        if outcome is None:
            return []

        if outcome.merged_token is not None:
            if self._nav.resolve_next_node(coalesce_node_id) is None:
                self._complete_coalesce_fire(
                    coalesce_name=coalesce_name,
                    consumed_tokens=tuple(outcome.consumed_tokens),
                    scope_row_id=current_token.row_id,
                )
                # Terminal coalesce — no downstream transforms.
                # Do NOT emit TokenCompleted here: the merged token still
                # needs to flow through the sink write for durable recording.
                # Telemetry is emitted later by accumulate_row_outcomes.
                return [
                    self._terminal_coalesce_row_result(
                        outcome.merged_token,
                        coalesce_name,
                        context=f"branch-loss notification for row '{current_token.row_id}'",
                    ),
                ]
            # Non-terminal — consume held siblings and emit the merged child's
            # READY continuation in ONE atomic journal transition (F1/D6),
            # then resume the merged token at the coalesce step.
            merged_item = self._nav.create_work_item(
                token=outcome.merged_token,
                current_node_id=coalesce_node_id,
            )
            self._complete_coalesce_fire(
                coalesce_name=coalesce_name,
                consumed_tokens=tuple(outcome.consumed_tokens),
                scope_row_id=current_token.row_id,
                merged_item=merged_item,
            )
            child_items.append(merged_item)
            return []

        if outcome.failure_reason:
            self._mark_coalesce_consumed_scheduler_work_terminal(
                coalesce_name=coalesce_name,
                consumed_tokens=tuple(outcome.consumed_tokens),
            )
            # Merge failed — build RowResults for held sibling tokens.
            # DB outcomes are already recorded by the executor (outcomes_recorded=True).
            # These RowResults propagate to the orchestrator for counter accounting.
            sibling_results: list[RowResult] = []
            for consumed_token in outcome.consumed_tokens:
                self._emit_token_completed(
                    consumed_token,
                    outcome=TerminalOutcome.FAILURE,
                    path=TerminalPath.UNROUTED,
                )
                sibling_results.append(
                    RowResult(
                        token=consumed_token,
                        final_data=consumed_token.row_data,
                        outcome=TerminalOutcome.FAILURE,
                        path=TerminalPath.UNROUTED,
                        error=FailureInfo(
                            exception_type="CoalesceFailure",
                            message=outcome.failure_reason,
                        ),
                    )
                )
            return sibling_results

        return []

    def _drain_work_queue(
        self,
        initial_item: WorkItem,
        ctx: PluginContext,
        *,
        preclaimed: TokenWorkItem | None = None,
    ) -> list[RowResult]:
        """Drain scheduler-backed work, processing tokens until empty.

        Each work item is durably persisted and leased before it advances;
        child continuations are persisted as READY work before the current
        lease is marked terminal.

        ``preclaimed`` (ADR-030 §C.4 row 9): the fenced ingest verb already
        persisted AND claimed the initial item inside its composed
        transaction; passing the returned LEASED row here skips the drain's
        own enqueue-and-claim.
        """
        return self._drain_durable_work_queue(initial_item, ctx, preclaimed=preclaimed)

    def drain_scheduled_work(self, ctx: PluginContext) -> list[RowResult]:
        """Advance already-persisted READY scheduler work for this run.

        This is the recovery entry point: a fresh processor can resume work
        items that were persisted by an earlier process without re-reading the
        source or re-running source-boundary validation.

        **Admission gate (ADR-030 §G, slice 4).** Under multi-worker
        deployment, concurrent workers are EXPECTED to share a run_id.
        Registry admission (the ``active_worker_fence_clause`` membership fence
        compiled into ``claim_ready`` / ``claim_pending_sink`` / ``enqueue_ready``
        CAS UPDATEs) is the correctness mechanism: a non-member's claim fails the
        EXISTS fence and surfaces ``RunWorkerEvictedError``, so a concurrent peer
        is a normal multi-worker state, not a precondition violation.

        ``peer_active_leases`` is **diagnostic only** from slice 4 onward. Its
        result is logged for observability but no longer causes a refusal. The
        old ADR-026 Precondition #9 single-active-resume enforcement is replaced
        by the membership fence on the claim verbs (filigree elspeth-66be4216cd,
        G3 — the original concern was duplicate RowResult emission; the fence CAS
        prevents non-members from claiming, closing that race structurally).
        """
        peer_owners = self._scheduler.peer_active_leases(
            run_id=self._run_id,
            caller_owner=self._scheduler_lease_owner,
            now=self._clock.now_utc(),
        )
        if peer_owners:
            logger.debug(
                "drain_scheduled_work: peer workers %r hold unexpired leases on run_id=%r (diagnostic only; "
                "admission fence on claim verbs is the correctness gate)",
                list(peer_owners),
                self._run_id,
            )
        return self._drain_scheduler_claims(ctx=ctx, pending_items={}, recover_pending_sinks=True)

    def has_scheduled_work(self) -> bool:
        """Return whether this run has non-terminal durable scheduler work."""
        return self._scheduler.count_active_work(run_id=self._run_id) > 0

    def has_peer_active_leases(self) -> bool:
        """Return True if any peer worker holds an unexpired LEASED item.

        ADR-030 multi-worker: the leader polls this after its source loop to
        detect follower workers that are still processing items (LEASED).  The
        leader must not call finalize_run while followers still hold leases,
        because those items will become PENDING_SINK and block the quiescence
        predicate in complete_run.
        """
        return bool(
            self._scheduler.peer_active_leases(
                run_id=self._run_id,
                caller_owner=self._scheduler_lease_owner,
                now=self._clock.now_utc(),
            )
        )

    def peer_lease_wait_budget_seconds(self) -> float:
        """Return the bounded wait budget for peer-held active item leases."""
        return float(self._scheduler_lease_seconds) + DEFAULT_ITEM_STALL_BUDGET_SECONDS

    def peer_active_lease_owners(self) -> tuple[str, ...]:
        """Return the distinct peer lease_owners holding unexpired LEASED rows.

        Diagnostic surface for the leader's bounded peer-wait: names the peers
        still blocking finalization when the wait times out.
        """
        return self._scheduler.peer_active_leases(
            run_id=self._run_id,
            caller_owner=self._scheduler_lease_owner,
            now=self._clock.now_utc(),
        )

    def reap_expired_peer_leases(self) -> int:
        """Drive lease maintenance once so dead peers are actively reaped.

        ADR-030: the leader's bounded peer-wait calls this each iteration so a
        peer that died mid-lease (heartbeat stopped, lease expired) is recovered
        to READY within the liveness window instead of waiting out the full item
        lease TTL.  Returns the number of leases recovered this pass.
        """
        return self._run_scheduler_maintenance(self._clock.now_utc())

    def active_scheduled_row_ids(self) -> frozenset[str]:
        """Return row IDs currently represented by active scheduler work."""
        return self._scheduler.active_row_ids(run_id=self._run_id)

    def summarize_scheduled_work(self) -> tuple[str, ...]:
        """Return grouped active scheduler work for invariant diagnostics."""
        return self._scheduler.summarize_active_work(run_id=self._run_id)

    def has_unresolved_scheduler_work(self) -> bool:
        """Return whether scheduler work remains short of a durable sink handoff.

        PENDING_SINK handoffs (and pending sinks re-claimed during resume)
        are excluded: they are terminalized only after sink durability via
        ``mark_sink_bound_scheduler_terminal_many``, which runs during sink
        writes — after the pre-sink completion invariants that call this.
        """
        return self._scheduler.count_unresolved_work(run_id=self._run_id) > 0

    def summarize_unresolved_scheduler_work(self) -> tuple[str, ...]:
        """Return grouped unresolved scheduler work for invariant diagnostics."""
        return self._scheduler.summarize_unresolved_work(run_id=self._run_id)

    def mark_blocked_barrier_terminal(self, barrier_key: str, token_ids: tuple[str, ...]) -> int:
        """Mark durable scheduler work consumed by a barrier as terminal."""
        expected_count = len(frozenset(token_ids))
        if not token_ids:
            raise AuditIntegrityError(f"Scheduler barrier terminalization for barrier_key={barrier_key!r} requires live token_ids.")
        if expected_count != len(token_ids):
            raise AuditIntegrityError(
                f"Scheduler barrier terminalization received duplicate live token_ids for barrier_key={barrier_key!r}: {token_ids!r}"
            )
        terminalized_count = self._scheduler.mark_blocked_barrier_terminal(
            run_id=self._run_id,
            barrier_key=barrier_key,
            token_ids=token_ids,
            now=self._clock.now_utc(),
            coordination_token=self._coordination_token,
        )
        if expected_count and terminalized_count != expected_count:
            raise AuditIntegrityError(
                f"Scheduler barrier terminalization mismatch for run_id={self._run_id!r} barrier_key={barrier_key!r}: "
                f"live consumed {expected_count} token(s), but durable scheduler terminalized {terminalized_count}."
            )
        return terminalized_count

    def _mark_coalesce_consumed_scheduler_work_terminal(
        self,
        *,
        coalesce_name: CoalesceName,
        consumed_tokens: tuple[TokenInfo, ...],
    ) -> None:
        """Terminalize scheduler rows for coalesce branches consumed by a failure.

        FAILURE arms only (merge failed / branch loss without a merged
        output): no emission accompanies the consumption, and the legacy
        partial-release wrapper semantics are kept. Successful coalesce fires
        go through ``_complete_coalesce_fire`` — ONE atomic journal
        transition (F1/D6).

        §E.2 (journal-first intake): every consumed branch — including the
        arrival whose intake-time accept produced this failure — holds a
        BLOCKED journal row, so the whole consumed set is released here. (The
        historical LEASED-arrival exclusion died with the in-claim arms.)
        """
        blocked_token_ids = tuple(token.token_id for token in consumed_tokens)
        if blocked_token_ids:
            self.mark_blocked_barrier_terminal(str(coalesce_name), blocked_token_ids)

    def _mark_buffered_scheduler_work_terminal(
        self,
        node_id: NodeID,
        tokens: Sequence[TokenInfo],
    ) -> None:
        """Mark scheduler work for aggregation-buffered tokens consumed by a FAILED flush.

        Failure arm only (the flush produced no outputs to emit). Successful
        flush completions go through ``_complete_aggregation_flush`` — ONE
        atomic journal transition per barrier completion (F1/D6).
        """
        blocked_token_ids = tuple(token.token_id for token in tokens)
        if not blocked_token_ids:
            return
        self.mark_blocked_barrier_terminal(
            str(node_id),
            blocked_token_ids,
        )

    def _sink_emission_from_result(self, result: RowResult) -> BarrierEmission:
        """Build the sink-bound barrier emission for one flush output result.

        Carries the full insert bundle: for a buffered passthrough token only
        the handoff fields are read (BLOCKED -> PENDING_SINK in place); for a
        generated output (transform-mode aggregate, with NO blocked row) the
        cursor fields (``row_id``/``step_index``/``ingest_sequence``,
        ``node_id=None``) drive a fresh PENDING_SINK insert on the terminal
        lane. The emission's ``token_id`` is the result token's REAL token_id,
        so the post-sink callback (``mark_sink_bound_scheduler_terminal_many``)
        terminalizes exactly the rows inserted here.
        """
        token = result.token
        return BarrierEmission(
            token_id=token.token_id,
            row_payload_json=self._scheduler.serialize_row_payload(token.row_data),
            sink_name=self._require_scheduler_sink_name(result),
            outcome=self._require_scheduler_outcome(result).value,
            path=result.path.value,
            error_hash=self._scheduler_error_hash(result),
            error_message=self._scheduler_error_message(result),
            row_id=token.row_id,
            node_id=None,
            step_index=self._scheduler_step_index(None),
            ingest_sequence=self._data_flow.resolve_row_ingest_sequence(token.row_id),
            branch_name=token.branch_name,
            fork_group_id=token.fork_group_id,
            join_group_id=token.join_group_id,
            expand_group_id=token.expand_group_id,
        )

    def _complete_aggregation_flush(
        self,
        node_id: NodeID,
        results: tuple[RowResult, ...],
        buffered_tokens: Sequence[TokenInfo],
    ) -> tuple[tuple[RowResult, ...], frozenset[str]]:
        """Complete a successful aggregation flush as ONE atomic journal transition.

        Consumes the buffered tokens' BLOCKED rows and emits every sink-bound
        flush output as a durable PENDING_SINK row in the same transaction
        (F1/D6): the moment the inputs are consumed, the outputs are journal-
        durable, so a crash before the sink write can no longer strand a
        flush output in memory while its inputs are TERMINAL.

        ORDERING vs batch status: ``execute_flush`` finalizes the batches row
        (``complete_batch`` -> COMPLETED) BEFORE this journal transition. A
        crash in that window leaves BLOCKED rows with a COMPLETED batch:
        transform-mode flushes have already recorded terminal token_outcomes
        (BATCH_CONSUMED), so the journal restore REFUSES loudly
        (``_derive_restored_batch_id`` requires a live BUFFERED outcome);
        passthrough flushes still carry BUFFERED outcomes and re-flush from
        the rebuilt buffer. Do not reorder the two writes without re-deriving
        the restore arms. This residual crash window is tracked:
        elspeth-3977d8ab60.

        ``intake_snapshot_token_ids`` (ADR-030 §E.3, slice 3): the firing
        group is exactly this batch's adopted members (``buffered_tokens`` —
        executor memory is fed ONLY by the journal-first intake, so memory ==
        adopted set by construction). Durable BLOCKED rows outside the
        snapshot are late arrivals that legitimately stay BLOCKED and join the
        NEXT batch at the next intake. Every in-claim exclusion arm
        (``leased_token_id`` / ``emit_sink_outputs=False``) died with the
        §E.2 journal-first intake — flushes always run out-of-claim and every
        member, including the trigger arrival, is a consumed BLOCKED row.

        §E.5: branch-loss records staged by empty-emission routing
        (``_route_empty_emission_results`` -> ``_notify_coalesce_of_lost_branch``)
        ride this completion transaction.

        Returns the results (sink-handoff results tagged
        ``scheduler_pending_sink=True``) and the emitted token_ids.
        """
        emissions: list[BarrierEmission] = []
        emitted_token_ids: set[str] = set()
        for result in results:
            token_id = result.token.token_id
            if not self._is_scheduler_sink_bound_result(result):
                continue
            if token_id in emitted_token_ids:
                raise AuditIntegrityError(
                    f"Aggregation flush for node {node_id!r} produced duplicate sink-bound result for token_id={token_id!r}; "
                    "cannot create an unambiguous scheduler pending-sink handoff."
                )
            emissions.append(self._sink_emission_from_result(result))
            emitted_token_ids.add(token_id)

        consumed_token_ids = tuple(token.token_id for token in buffered_tokens if token.token_id not in emitted_token_ids)

        self._scheduler.complete_barrier(
            run_id=self._run_id,
            barrier_key=str(node_id),
            consumed_token_ids=consumed_token_ids,
            emitted_pending_sink=tuple(emissions),
            emitted_ready=(),
            now=self._clock.now_utc(),
            # §E.3 per-firing-group snapshot: this batch's adopted members.
            intake_snapshot_token_ids=frozenset(token.token_id for token in buffered_tokens),
            coordination_token=self._require_coordination_token(),
            # Attributed park (ADR-030): the post-sink strict owner CAS
            # terminalizes these handoffs under this worker's lease identity.
            pending_sink_lease_owner=self._scheduler_lease_owner,
            branch_losses=self._take_pending_branch_losses(),
        )

        if not emitted_token_ids:
            return results, frozenset()
        pending_sink_token_ids = frozenset(emitted_token_ids)
        tagged_results = cast(
            tuple[RowResult, ...],
            self._with_scheduler_pending_sink_handoffs(results, pending_sink_token_ids),
        )
        return tagged_results, pending_sink_token_ids

    def _ready_emission_from_work_item(self, item: WorkItem) -> BarrierEmission:
        """Build the READY continuation emission for a merged-coalesce work item.

        Field derivation MUST mirror ``_enqueue_scheduler_work_item`` exactly:
        the drain loop (or ``_drain_work_queue``'s initial claim) later runs
        the idempotent enqueue for the same WorkItem, which reconciles against
        the row inserted here by deterministic ``work_item_id`` and strict
        field equality.
        """
        token = item.token
        return BarrierEmission(
            token_id=token.token_id,
            row_payload_json=self._scheduler.serialize_row_payload(token.row_data),
            row_id=token.row_id,
            node_id=self._scheduler_node_id(item.current_node_id),
            step_index=self._scheduler_step_index(item.current_node_id),
            ingest_sequence=self._data_flow.resolve_row_ingest_sequence(token.row_id),
            queue_key=self._queue_key_for_blocked_item(item),
            barrier_key=self._barrier_key_for_blocked_item(item),
            on_success_sink=item.on_success_sink,
            branch_name=token.branch_name,
            fork_group_id=token.fork_group_id,
            join_group_id=token.join_group_id,
            expand_group_id=token.expand_group_id,
            coalesce_node_id=str(item.coalesce_node_id) if item.coalesce_node_id is not None else None,
            coalesce_name=str(item.coalesce_name) if item.coalesce_name is not None else None,
        )

    def _complete_coalesce_fire(
        self,
        *,
        coalesce_name: CoalesceName,
        consumed_tokens: tuple[TokenInfo, ...],
        scope_row_id: str,
        merged_item: WorkItem | None = None,
        merged_sink_result: RowResult | None = None,
    ) -> None:
        """Complete a fired coalesce barrier as ONE atomic journal transition.

        Consumes the fired group's BLOCKED rows and emits the merged output in
        the same transaction (F1/D6) — a crash after the branches are consumed
        can no longer lose the merged output. Non-terminal coalesces pass
        ``merged_item`` (a READY continuation); intake-fired TERMINAL
        coalesces pass ``merged_sink_result`` (the COALESCED sink-bound
        result), emitted as a fresh PENDING_SINK row on the terminal lane so
        the merged output is journal-durable before the in-process sink write
        (closing the historical in-claim memory-only ride).

        COALESCE barriers share ONE ``barrier_key`` (the coalesce_name) across
        every row pending at the coalesce, so ``scope_row_id`` (the firing
        group's row_id) is mandatory: it narrows both the membership and
        exhaustiveness checks to this row's pending group.

        §E.2/§E.3 (journal-first intake): every consumed branch — including
        the arrival whose intake-time accept fired the merge — holds a BLOCKED
        journal row; the firing-group snapshot is exactly the consumed set
        (executor memory is fed only by adoption, so memory == adopted set by
        construction). A branch row blocked-but-not-yet-adopted at fire time
        is a late arrival: it stays BLOCKED and is released by the next
        intake's late-arrival arm (§E.3a). (The historical LEASED-trigger
        ``active_token_id`` exclusion died with the in-claim arms.)
        """
        consumed_token_ids = tuple(token.token_id for token in consumed_tokens)
        if not consumed_token_ids and merged_item is None and merged_sink_result is None:
            return
        self._scheduler.complete_barrier(
            run_id=self._run_id,
            barrier_key=str(coalesce_name),
            consumed_token_ids=consumed_token_ids,
            emitted_pending_sink=() if merged_sink_result is None else (self._sink_emission_from_result(merged_sink_result),),
            emitted_ready=() if merged_item is None else (self._ready_emission_from_work_item(merged_item),),
            now=self._clock.now_utc(),
            # §E.3 per-firing-group snapshot: the fired group's adopted branches.
            intake_snapshot_token_ids=frozenset(consumed_token_ids),
            scope_row_id=scope_row_id,
            coordination_token=self._require_coordination_token(),
            # Attributed park (ADR-030): the post-sink strict owner CAS
            # terminalizes the merged handoff under this worker's identity.
            pending_sink_lease_owner=self._scheduler_lease_owner,
        )

    def complete_coalesce_merge(
        self,
        *,
        coalesce_name: CoalesceName,
        consumed_tokens: tuple[TokenInfo, ...],
        merged_token: TokenInfo,
        coalesce_node_id: NodeID,
        ctx: PluginContext,
    ) -> list[RowResult]:
        """Atomically complete an out-of-claim coalesce fire and drive the merged token.

        Orchestrator-facing verb for timeout/EOF coalesce fires (outcomes.py):
        the consumed branches' BLOCKED rows and the merged child's READY row
        transition in ONE ``complete_barrier`` call, then the merged token is
        processed from the coalesce node. The drain's initial enqueue
        reconciles idempotently against the READY row inserted here (same
        deterministic work_item_id and field derivation).
        """
        merged_item = self._nav.create_work_item(
            token=merged_token,
            current_node_id=coalesce_node_id,
            coalesce_node_id=coalesce_node_id,
            coalesce_name=coalesce_name,
        )
        self._complete_coalesce_fire(
            coalesce_name=coalesce_name,
            consumed_tokens=consumed_tokens,
            scope_row_id=merged_token.row_id,
            merged_item=merged_item,
        )
        return self._drain_work_queue(merged_item, ctx)

    def _drain_durable_work_queue(
        self,
        initial_item: WorkItem,
        ctx: PluginContext,
        *,
        preclaimed: TokenWorkItem | None = None,
    ) -> list[RowResult]:
        """Drain the scheduler-backed work queue for a single source token."""
        pending_items: dict[str, WorkItem] = {}

        if preclaimed is not None:
            # The fenced ingest already enqueued AND claimed this item in its
            # composed transaction. Mirror _enqueue_scheduler_work_item's
            # pending-item registration so the drain rehydrates the live
            # payload instead of deserializing the journal row.
            initial_claim = preclaimed
            if initial_claim.status is TokenWorkStatus.READY or (
                initial_claim.status is TokenWorkStatus.LEASED and initial_claim.lease_owner == self._scheduler_lease_owner
            ):
                pending_items[initial_claim.work_item_id] = initial_item
        else:
            initial_claim = self._enqueue_scheduler_work_item(initial_item, pending_items, claim_immediately=True)
        preclaimed_items = (
            [initial_claim]
            if initial_claim.status is TokenWorkStatus.LEASED and initial_claim.lease_owner == self._scheduler_lease_owner
            else []
        )

        with self._spans.row_span(initial_item.token.row_id, initial_item.token.token_id):
            results = self._drain_scheduler_claims(
                ctx=ctx,
                pending_items=pending_items,
                recover_pending_sinks=False,
                preclaimed_items=preclaimed_items,
            )

        return results

    def _run_scheduler_maintenance(self, now: datetime) -> int:
        """Evict dead members then recover expired peer leases (§C.2 path 1).

        Ordering: evict-before-reap (§C.2 :232) ensures that when we rotate
        an expired item lease the owner's registry row already carries
        status='evicted' (arm b of owner_registry_dead), so the reap is
        silent (no worker_stalled emitted for already-evicted workers).

        The stall-arm reap (item_stall_budget) is the documented exception:
        it may rotate BEFORE eviction for a live-heartbeat-but-wedged-drain
        worker, and emits worker_stalled in the same transaction (§A.5 :145).

        Leader-only: the evict sweep is gated on ``_coordination_token`` being
        set (followers never evict; unfenced/test arm skips silently).

        ADR-030 §C.2/§C.3 (slice 5): followers must NOT run
        ``recover_expired_leases``.  A follower passes
        ``coordination_token=None``, which takes the UNFENCED/LEGACY arm of
        ``recover_expired_leases`` — that arm unconditionally reaps ALL
        expired non-caller leases, defeating the liveness-aware gate that
        protects the leader's and peers' in-flight item leases from spurious
        rotation (§A.5/§C.1).  Followers are drain-only workers; lease
        recovery and eviction are the leader's responsibility (§C.2 path 1,
        §C.3: "followers drain what is claimable, then idle/exit").
        The follower path is identified by both ``_coordination_token is None``
        AND ``_scheduler_lease_owner_registered`` (registered worker identity
        from run_workers) AND ``_run_coordination is None``.  All three must
        be true simultaneously for the follower build path (follower.py); in
        all other None-token cases (N=1 test arm, direct repo construction)
        the legacy reap is correct and must be preserved.
        """
        # §C.2 path 1: leader evicts dead non-leader members before reaping.
        # Individual, not bulk — one evict_worker call per dead member (§B.4,
        # §C.2 :233). evict_worker is idempotent (benign skip on CAS miss).
        if self._coordination_token is not None and self._run_coordination is not None:
            from elspeth.contracts.coordination import DEFAULT_RUN_LIVENESS_WINDOW_SECONDS

            grace = DEFAULT_RUN_LIVENESS_WINDOW_SECONDS
            dead_members = self._run_coordination.dead_non_leader_workers(
                run_id=self._run_id,
                leader_worker_id=self._coordination_token.worker_id,
                now=now,
                grace_seconds=grace,
            )
            for target_worker_id in dead_members:
                self._run_coordination.evict_worker(
                    token=self._coordination_token,
                    target_worker_id=target_worker_id,
                    now=now,
                    grace_seconds=grace,
                    window_seconds=DEFAULT_RUN_LIVENESS_WINDOW_SECONDS,
                )

        # ADR-030 §C.3: followers must not run recover_expired_leases.
        # The follower build path (follower.py:build_follower_processor) has
        # coordination_token=None AND run_coordination=None AND a registered
        # lease owner (scheduler_lease_owner_registered=True).  In that case
        # skipping the reap is CORRECT; the leader owns the reap sweep.
        # All other None-token paths (N=1, test arm) still run the legacy reap.
        is_follower_path = self._coordination_token is None and self._run_coordination is None and self._scheduler_lease_owner_registered
        if is_follower_path:
            self._scheduler_drains_since_maintenance = 0
            return 0

        recovered = self._scheduler.recover_expired_leases(
            run_id=self._run_id,
            now=now,
            caller_owner=self._scheduler_lease_owner,
            coordination_token=self._coordination_token,
        )
        self._scheduler_drains_since_maintenance = 0
        return recovered

    def _scheduler_maintenance_due(self, *, recover_pending_sinks: bool) -> bool:
        """Return whether this drain should run scheduler maintenance up front."""
        if recover_pending_sinks:
            return True
        self._scheduler_drains_since_maintenance += 1
        return self._scheduler_drains_since_maintenance >= SCHEDULER_MAINTENANCE_INTERVAL

    # ─────────────────────────────────────────────────────────────────────────
    # Journal-first barrier intake (ADR-030 §E.2/§E.3/§E.3a/§E.5, slice 3)
    # ─────────────────────────────────────────────────────────────────────────

    def _require_coordination_token(self) -> CoordinationToken:
        """The leader fencing token, REQUIRED for the slice-3 adoption verbs."""
        if self._coordination_token is None:
            raise OrchestrationInvariantError(
                "Journal-first barrier intake requires the leader coordination token (ADR-030 §E.2): "
                "adopt_blocked_barrier_item / adopt_coalesce_branch_losses are fenced verbs with no "
                "unfenced arm. Construct RowProcessor with coordination_token — the orchestrator "
                "binds it at begin_run (epoch 1) or at the resume takeover CAS."
            )
        return self._coordination_token

    def _backdated_accept_monotonic(self, row: TokenWorkItem) -> float:
        """Convert a row's durable ``barrier_blocked_at`` onto the monotonic scale.

        §E.2 backdated accept timing — the EXACT clamped wall->monotonic
        transform the journal restore uses (coalesce restore_from_journal /
        aggregation elapsed_age derivation): trigger latches and coalesce
        arrival anchors are pure functions of durable state + config, hence
        invariant under leader takeover (§H 476).
        """
        if row.barrier_blocked_at is None:
            raise AuditIntegrityError(
                f"BLOCKED journal row for token {row.token_id!r} (run {self._run_id!r}) has NULL "
                "barrier_blocked_at — the backdated accept instant cannot be derived; journal "
                "corruption (mark_blocked stamps every hold)."
            )
        now_wall = self._clock.now_utc()
        return self._clock.monotonic() - max(0.0, (now_wall - row.barrier_blocked_at).total_seconds())

    def _token_for_intake(self, row: TokenWorkItem) -> TokenInfo:
        """Resolve the TokenInfo to feed executor memory for one adopted row.

        Live stash first (N=1 parity: the exact post-transform token the old
        in-claim accept used, with its original resume provenance). Without a
        live stash, the durable row is still authoritative: fresh leaders use
        offset zero for normal follower handoffs, while resume leaders use the
        audit-derived offset stamped with checkpoint provenance.
        """
        hold = self._live_barrier_holds.pop(row.token_id, None)
        if hold is not None:
            return hold.token
        if self._resume_checkpoint_id is None:
            return token_from_journal_item(row, attempt_offset=0, resume_checkpoint_id=None)
        max_attempts = self._execution.get_max_node_state_attempts(self._run_id, [row.token_id])
        return token_from_journal_item(
            row,
            attempt_offset=max_attempts.get(row.token_id, -1) + 1,
            resume_checkpoint_id=self._resume_checkpoint_id,
        )

    def _run_barrier_intake_pass(self, ctx: PluginContext) -> tuple[list[RowResult], list[WorkItem]]:
        """One §E.2 intake pass: adopt arrivals, replay losses, fire triggers.

        Runs at the top of every drain iteration (and from the orchestrator's
        EOF loop via ``run_barrier_intake``). Steps, in design order:

        1. arrival intake — every intake-pending BLOCKED barrier row
           (``barrier_adopted_epoch IS NULL``) is adopted via the fenced
           backdated adoption verb and fed into executor memory; coalesce
           adoption runs the executor accept, surfacing merge fires, group
           failures and late-arrival releases (§E.3a) here;
        2. per-adoption aggregation trigger evaluation — count/condition
           triggers fire from the SAME intake step as the triggering
           arrival's adoption (the §E.2 replacement for the deleted in-claim
           flush arm; batch composition is preserved because the check runs
           after EACH adoption, exactly like the old accept-then-check);
        3. branch-loss replay (§E.5) — unadopted durable losses are marked
           (journal-first) and replayed through ``notify_branch_lost``
           before the next trigger evaluation; at N=1 this is a structural
           no-op (record-then-notify already ran in-claim).

        Returns (results, child_items): results append to the drain's result
        set; child_items are continuations whose READY rows were inserted
        atomically by ``complete_barrier`` — the caller enqueues them
        (idempotent reconcile) into its own loop.
        """
        results: list[RowResult] = []
        child_items: list[WorkItem] = []
        if not self._aggregation_settings and self._coalesce_executor is None:
            return results, child_items

        # ADR-030 §E.2 intake scan shape: only intake-pending rows
        # (barrier_adopted_epoch IS NULL).  The SQL predicate avoids
        # materializing adopted rows on every drain iteration — without it a
        # filling count-N batch costs O(N²/2) full-row hydrations (finding 2).
        pending_rows = self._scheduler.list_pending_blocked_barrier_items(run_id=self._run_id)
        if pending_rows:
            coalesce_keys = {str(name) for name in self._coalesce_node_ids}
            aggregation_keys = {str(node_id) for node_id in self._aggregation_settings}
            for row in pending_rows:
                if row.barrier_key in aggregation_keys:
                    self._intake_adopt_aggregation_row(row, ctx, results, child_items)
                elif row.barrier_key in coalesce_keys:
                    self._intake_adopt_coalesce_row(row, results, child_items)
                else:
                    raise AuditIntegrityError(
                        f"Intake-pending BLOCKED row for token {row.token_id!r} (run {self._run_id!r}) carries "
                        f"orphan barrier_key {row.barrier_key!r}: not a configured coalesce "
                        f"({sorted(coalesce_keys)}) or aggregation node ({sorted(aggregation_keys)})."
                    )

        self._intake_replay_branch_losses(results, child_items)
        return results, child_items

    def _intake_adopt_aggregation_row(
        self,
        row: TokenWorkItem,
        ctx: PluginContext,
        results: list[RowResult],
        child_items: list[WorkItem],
    ) -> None:
        """Adopt one aggregation barrier row, then evaluate the node's trigger."""
        if row.barrier_key is None:  # pragma: no cover - excluded by the query contract
            raise AuditIntegrityError(f"Intake aggregation row {row.work_item_id!r} has no barrier_key.")
        node_id = NodeID(row.barrier_key)
        coordination_token = self._require_coordination_token()
        # Resolve the token BEFORE the fenced verb so an invalid journal row is
        # refused with ZERO durable mutation. Valid follower handoffs can be
        # rebuilt from the durable row even without a live stash entry.
        token = self._token_for_intake(row)
        batch_id, ordinal = self._aggregation_executor.open_batch_membership(node_id)
        adoption = self._scheduler.adopt_blocked_barrier_item(
            run_id=self._run_id,
            work_item_id=row.work_item_id,
            token_id=row.token_id,
            barrier_key=row.barrier_key,
            membership=BatchMembershipSpec(batch_id=batch_id, ordinal=ordinal),
            buffered_outcome=BufferedOutcomeSpec(batch_id=batch_id),
            now=self._clock.now_utc(),
            coordination_token=coordination_token,
        )
        if not adoption.adopted:
            # Idempotent success-SKIP: already adopted (a racing duplicate of
            # this leader's own pass). MUST NOT re-feed memory (§C.4 row 6a).
            return
        self._aggregation_executor.accept_adopted_row(node_id, token, accept_time=self._backdated_accept_monotonic(row))

        # Step 2: per-adoption trigger evaluation — the §E.2 home of the old
        # in-claim count/condition flush decision (accept-then-check), so
        # batch composition is byte-identical to the in-claim era.
        should_flush, trigger_type = self._aggregation_executor.check_flush_status(node_id)
        if not should_flush:
            return
        transform = self._nav.resolve_plugin_for_node(node_id)
        if not isinstance(transform, TransformProtocol) or not transform.is_batch_aware:
            raise OrchestrationInvariantError(
                f"Aggregation node {node_id!r} fired a {trigger_type} trigger at intake but resolves to "
                f"{type(transform).__name__!r}, not a batch-aware transform. DAG/config inconsistency."
            )
        flush_results, flush_child_items = self.handle_timeout_flush(
            node_id=node_id,
            transform=transform,
            ctx=ctx,
            trigger_type=trigger_type if trigger_type is not None else TriggerType.COUNT,
        )
        results.extend(flush_results)
        child_items.extend(flush_child_items)

    def _intake_adopt_coalesce_row(
        self,
        row: TokenWorkItem,
        results: list[RowResult],
        child_items: list[WorkItem],
    ) -> None:
        """Adopt one coalesce barrier row and run the intake-time accept."""
        if row.barrier_key is None:  # pragma: no cover - excluded by the query contract
            raise AuditIntegrityError(f"Intake coalesce row {row.work_item_id!r} has no barrier_key.")
        if self._coalesce_executor is None:  # pragma: no cover - partition guarantees a coalesce key
            raise OrchestrationInvariantError(f"Intake coalesce row for {row.barrier_key!r} but no CoalesceExecutor is configured.")
        coalesce_name = CoalesceName(row.barrier_key)
        coordination_token = self._require_coordination_token()
        # Resolve the token BEFORE the fenced verb (refusal-before-mutation
        # for invalid journal rows — see the aggregation arm).
        token = self._token_for_intake(row)
        adoption = self._scheduler.adopt_blocked_barrier_item(
            run_id=self._run_id,
            work_item_id=row.work_item_id,
            token_id=row.token_id,
            barrier_key=row.barrier_key,
            membership=None,
            buffered_outcome=None,
            now=self._clock.now_utc(),
            coordination_token=coordination_token,
        )
        if not adoption.adopted:
            return
        outcome = self._coalesce_executor.accept(
            token=token,
            coalesce_name=str(coalesce_name),
            arrival_time=self._backdated_accept_monotonic(row),
        )

        if outcome.held:
            return

        if outcome.late_arrival:
            # §E.3a: the group already completed — release THIS row alone in
            # the same drain iteration, with forensic late-arrival context.
            # The executor already recorded the FAILED node_state + FAILURE
            # outcome (outcomes_recorded=True on the late arm).
            released = self._scheduler.mark_blocked_barrier_terminal(
                run_id=self._run_id,
                barrier_key=str(coalesce_name),
                token_ids=(token.token_id,),
                now=self._clock.now_utc(),
                coordination_token=self._coordination_token,
                release_context={
                    "late_arrival": True,
                    "reason": outcome.failure_reason,
                    "released_by": self._scheduler_lease_owner,
                    "scope_row_id": row.row_id,
                },
            )
            if released != 1:
                raise AuditIntegrityError(
                    f"Late-arrival release for token {token.token_id!r} at coalesce {coalesce_name!r} "
                    f"(run {self._run_id!r}) terminalized {released} rows; expected exactly one."
                )
            self._emit_token_completed(token, outcome=TerminalOutcome.FAILURE, path=TerminalPath.UNROUTED)
            results.append(
                RowResult(
                    token=token,
                    final_data=token.row_data,
                    outcome=TerminalOutcome.FAILURE,
                    path=TerminalPath.UNROUTED,
                    error=FailureInfo(exception_type="CoalesceFailure", message=outcome.failure_reason or "late_arrival_after_merge"),
                )
            )
            return

        if outcome.merged_token is not None:
            self._fire_intake_coalesce_merge(coalesce_name, outcome, scope_row_id=row.row_id, results=results, child_items=child_items)
            return

        if outcome.failure_reason:
            # Group failure completed by this arrival: every consumed branch
            # (this one included) holds a BLOCKED row — release them all.
            self._mark_coalesce_consumed_scheduler_work_terminal(
                coalesce_name=coalesce_name,
                consumed_tokens=tuple(outcome.consumed_tokens),
            )
            error_msg = outcome.failure_reason
            # Bug 9z8 fix: only record if CoalesceExecutor didn't already record.
            if not outcome.outcomes_recorded:
                self._data_flow.record_token_outcome(
                    ref=TokenRef(token_id=token.token_id, run_id=self._run_id),
                    outcome=TerminalOutcome.FAILURE,
                    path=TerminalPath.UNROUTED,
                    error_hash=compute_error_hash(error_msg),
                )
            # Emit TokenCompleted telemetry AFTER Landscape recording. Only
            # the arriving token surfaces a RowResult (the held siblings'
            # outcomes were recorded by the executor) — the pre-§E.2 shape.
            self._emit_token_completed(token, outcome=TerminalOutcome.FAILURE, path=TerminalPath.UNROUTED)
            results.append(
                RowResult(
                    token=token,
                    final_data=token.row_data,
                    outcome=TerminalOutcome.FAILURE,
                    path=TerminalPath.UNROUTED,
                    error=FailureInfo(exception_type="CoalesceFailure", message=error_msg),
                )
            )
            return

        raise OrchestrationInvariantError(
            f"CoalesceOutcome for token {token.token_id} in coalesce '{coalesce_name}' is in invalid state: "
            f"held={outcome.held}, merged_token={outcome.merged_token is not None}, "
            f"failure_reason={outcome.failure_reason!r}"
        )

    def _fire_intake_coalesce_merge(
        self,
        coalesce_name: CoalesceName,
        outcome: CoalesceOutcome,
        *,
        scope_row_id: str,
        results: list[RowResult],
        child_items: list[WorkItem],
    ) -> None:
        """Complete an intake-time coalesce merge fire (terminal or not).

        Non-terminal: mirrors ``complete_coalesce_merge``'s shape — the merged
        child's READY continuation is inserted atomically with the consumption
        (F1/D6) and the same WorkItem is handed back for the caller's
        idempotent enqueue.

        Terminal: the COALESCED sink-bound result is emitted as a fresh
        PENDING_SINK row in the SAME atomic completion — the merged output is
        journal-durable the moment its inputs are consumed (the pre-§E.2
        in-claim ride to ``mark_pending_sink`` left it memory-only between the
        consumption and the claim disposition).
        """
        if outcome.merged_token is None:  # pragma: no cover - caller checks
            raise OrchestrationInvariantError("merged_token is None in _fire_intake_coalesce_merge")
        coalesce_node_id = self._coalesce_node_ids[coalesce_name]
        if self._nav.resolve_next_node(coalesce_node_id) is None:
            terminal_result = self._terminal_coalesce_row_result(
                outcome.merged_token,
                coalesce_name,
                context=f"intake coalesce fire for token '{outcome.merged_token.token_id}'",
            )
            self._complete_coalesce_fire(
                coalesce_name=coalesce_name,
                consumed_tokens=tuple(outcome.consumed_tokens),
                scope_row_id=scope_row_id,
                merged_sink_result=terminal_result,
            )
            results.append(replace(terminal_result, scheduler_pending_sink=True))
            return
        merged_item = self._nav.create_work_item(
            token=outcome.merged_token,
            current_node_id=coalesce_node_id,
            coalesce_node_id=coalesce_node_id,
            coalesce_name=coalesce_name,
        )
        self._complete_coalesce_fire(
            coalesce_name=coalesce_name,
            consumed_tokens=tuple(outcome.consumed_tokens),
            scope_row_id=scope_row_id,
            merged_item=merged_item,
        )
        child_items.append(merged_item)

    def _intake_replay_branch_losses(
        self,
        results: list[RowResult],
        child_items: list[WorkItem],
    ) -> None:
        """§E.5 loss intake: mark unadopted durable losses, replay into memory.

        Journal-first: the fenced cursor mark commits BEFORE the in-memory
        replay — a crash between mark and replay loses nothing because the
        takeover restore derives lost_branches from the FULL loss table. At
        N=1 the replay arm is structurally idle: the producer already
        notified in-claim (record-then-notify), so ``has_recorded_branch_loss``
        (or the executor's completed-keys check) dedups every row.
        """
        if self._coalesce_executor is None:
            return
        losses = self._scheduler.list_unadopted_coalesce_branch_losses(run_id=self._run_id)
        if not losses:
            return
        coordination_token = self._require_coordination_token()
        self._scheduler.adopt_coalesce_branch_losses(
            run_id=self._run_id,
            loss_ids=[loss.loss_id for loss in losses],
            now=self._clock.now_utc(),
            coordination_token=coordination_token,
        )
        for loss in losses:
            if self._coalesce_executor.has_recorded_branch_loss(loss.coalesce_name, loss.row_id, loss.branch_name):
                continue
            outcome = self._coalesce_executor.notify_branch_lost(
                coalesce_name=loss.coalesce_name,
                row_id=loss.row_id,
                lost_branch=loss.branch_name,
                reason=loss.reason,
            )
            if outcome is None:
                continue
            coalesce_name = CoalesceName(loss.coalesce_name)
            if outcome.merged_token is not None:
                self._fire_intake_coalesce_merge(coalesce_name, outcome, scope_row_id=loss.row_id, results=results, child_items=child_items)
                continue
            if outcome.failure_reason:
                self._mark_coalesce_consumed_scheduler_work_terminal(
                    coalesce_name=coalesce_name,
                    consumed_tokens=tuple(outcome.consumed_tokens),
                )
                # Replayed must-fail (§E.5: a must-fail group fails within one
                # drain iteration of the loss becoming visible): mirror the
                # branch-loss notification failure arm — RowResults for the
                # held siblings the failure consumed.
                for consumed_token in outcome.consumed_tokens:
                    self._emit_token_completed(consumed_token, outcome=TerminalOutcome.FAILURE, path=TerminalPath.UNROUTED)
                    results.append(
                        RowResult(
                            token=consumed_token,
                            final_data=consumed_token.row_data,
                            outcome=TerminalOutcome.FAILURE,
                            path=TerminalPath.UNROUTED,
                            error=FailureInfo(exception_type="CoalesceFailure", message=outcome.failure_reason),
                        )
                    )
                continue
            raise OrchestrationInvariantError(
                f"Replayed branch loss {loss.loss_id!r} ({loss.coalesce_name!r}/{loss.row_id!r}/{loss.branch_name!r}) "
                f"produced an invalid CoalesceOutcome: held={outcome.held}, merged=None, failure_reason=None."
            )

    def run_barrier_intake(self, ctx: PluginContext) -> list[RowResult]:
        """Public §E.2 intake entry for the orchestrator's EOF loop (§D step 3).

        Runs one intake pass and drives any continuation items (merged
        coalesce children, flush continuations) through the durable work
        queue, returning every produced RowResult for the caller's outcome
        accumulation.
        """
        results, child_items = self._run_barrier_intake_pass(ctx)
        for child_item in child_items:
            results.extend(self._drain_work_queue(child_item, ctx))
        return results

    def has_blocked_barrier_work(self) -> bool:
        """Whether any durable BLOCKED barrier holds remain (§D step-3 loop condition)."""
        return bool(self._scheduler.list_blocked_barrier_items(run_id=self._run_id))

    def count_unquiesced_scheduler_work(self) -> int:
        """§D step-2 journal quiescence: READY + LEASED (non-pending-sink) rows."""
        return self._scheduler.count_unquiesced_work(run_id=self._run_id)

    def summarize_unquiesced_scheduler_work(self) -> tuple[str, ...]:
        """Grouped §D step-2 unquiesced work for invariant diagnostics."""
        return self._scheduler.summarize_unquiesced_work(run_id=self._run_id)

    def _drain_scheduler_claims(
        self,
        *,
        ctx: PluginContext,
        pending_items: dict[str, WorkItem],
        recover_pending_sinks: bool,
        preclaimed_items: list[TokenWorkItem] | None = None,
        before_claim: Callable[[], None] | None = None,
    ) -> list[RowResult]:
        """Claim and advance scheduler work until no READY work remains.

        When ``recover_pending_sinks=True`` (the resume entry point), any
        PENDING_SINK rows left durable by a prior crashed worker are drained
        up front via ``claim_pending_sink`` BEFORE the main claim_ready loop
        runs. This is the only path that re-claims PENDING_SINK rows inside
        a drain.

        PENDING_SINK rows produced by the main loop itself (via
        ``mark_pending_sink`` after a transform success at sink handoff) are
        NOT re-claimed by this drain — their RowResult is already in
        ``results`` from the ``claim_ready`` path that produced them, and
        their durable scheduler row is terminalized later by
        ``mark_sink_bound_scheduler_terminal`` via the sink-write callback
        (``_make_checkpoint_after_sink_factory`` in orchestrator/core.py).
        Re-claiming them here would emit a duplicate
        ``_row_result_from_pending_sink`` for the same token_id. See
        filigree elspeth-5c5e88b071 (G3).

        Note: if a prior worker's lease on a sink-bound row is still active
        (not yet expired), ``claim_pending_sink`` won't find it (status is
        LEASED, not PENDING_SINK) and ``recover_expired_leases`` won't touch
        it (lease not yet expired). It will be stranded until the lease
        ages out, at which point a subsequent ``drain_scheduled_work`` call
        from a later resume attempt recovers it. That is correct behavior:
        the prior worker may still be alive and finishing the sink write.
        Forcing recovery here would race against an in-flight sink writer
        and risk duplicate sink emission.
        """
        results: list[RowResult] = []

        if recover_pending_sinks:
            self._drain_preexisting_pending_sinks(results)

        iterations = 0
        maintenance_iteration = 0
        preclaimed_queue = list(preclaimed_items or ())

        if self._scheduler_maintenance_due(recover_pending_sinks=recover_pending_sinks):
            self._run_scheduler_maintenance(self._clock.now_utc())

        while True:
            iterations += 1
            if iterations > MAX_WORK_QUEUE_ITERATIONS:
                raise RuntimeError(f"Work queue exceeded {MAX_WORK_QUEUE_ITERATIONS} iterations. Possible infinite loop in pipeline.")
            if before_claim is not None:
                before_claim()

            # ADR-030 §E.2 (slice 3): per-iteration journal-first intake —
            # adopt intake-pending BLOCKED barrier rows into executor memory,
            # replay durable branch losses (§E.5), release late arrivals
            # (§E.3a) and evaluate count/condition triggers over post-intake
            # memory, ALL before this iteration's claim. Flush/merge outputs
            # append to the drain's results; continuation items enqueue into
            # the current loop (their READY rows were inserted atomically by
            # complete_barrier; the enqueue reconciles idempotently).
            intake_results, intake_child_items = self._run_barrier_intake_pass(ctx)
            results.extend(intake_results)
            for intake_child in intake_child_items:
                self._enqueue_scheduler_work_item(intake_child, pending_items)

            # The claim timestamp is read AFTER intake: rows the intake just
            # emitted carry available_at stamps later than an iteration-top
            # reading, and claim_ready's available_at <= now predicate must
            # see them.
            now = self._clock.now_utc()
            if iterations - maintenance_iteration >= SCHEDULER_MAINTENANCE_INTERVAL:
                self._run_scheduler_maintenance(now)
                maintenance_iteration = iterations

            claimed: TokenWorkItem | None
            if preclaimed_queue:
                claimed = preclaimed_queue.pop(0)
            else:
                claimed = self._scheduler.claim_ready(
                    run_id=self._run_id,
                    lease_owner=self._scheduler_lease_owner,
                    lease_seconds=self._scheduler_lease_seconds,
                    now=now,
                )
            if claimed is None:
                if recover_pending_sinks:
                    recovered = self._run_scheduler_maintenance(self._clock.now_utc())
                    if recovered:
                        continue
                if pending_items:
                    # ADR-030 multi-worker: peer followers may have claimed some of
                    # the READY children enqueued by this leader (e.g. json_explode
                    # or per-LLM-call continuations).  Once a peer claims a child, the
                    # leader's in-memory pending entry can no longer be claimed READY:
                    # the peer drives it LEASED → PENDING_SINK (lease_owner kept) →
                    # TERMINAL/FAILED.  The leader may relinquish such peer-owned
                    # continuations, but ONLY under a discriminator that keeps the
                    # single-worker invariant and the audit backstops intact:
                    #
                    #   (1) NONE of the pending items are still READY
                    #       (count_ready_in_set == 0) — a still-READY item is a
                    #       genuine stranded continuation and must raise; and
                    #   (2) NONE of the pending items are FAILED
                    #       (count_failed_in_set == 0) — FAILED is the ONLY status
                    #       absent from BOTH backstops (count_active_work AND
                    #       complete_run's quiescence CAS cover READY/LEASED/BLOCKED/
                    #       PENDING_SINK but NOT FAILED), so a self-FAILED stray would
                    #       be silently lost; refusing on any FAILED pending row keeps
                    #       it loud (this is the M1 residual fix, and it lands at N=1
                    #       where the leader's OWN item is FAILED with no peer); and
                    #   (3) a peer is/was carrying work — has_peer_owned_work: some
                    #       OTHER lease_owner holds a LEASED or PENDING_SINK row on
                    #       this run.  PENDING_SINK is included because
                    #       mark_pending_sink KEEPS the claimant's lease_owner, so a
                    #       follower that has already parked all its claims as
                    #       PENDING_SINK and let its active LEASES lapse is STILL
                    #       detectable (the in-claim drain races AGAINST this: by the
                    #       time the leader's claim_ready returns None, the follower
                    #       may hold zero active leases yet own many PENDING_SINK
                    #       rows).  An N=1 leader's own rows carry the leader's owner
                    #       (BLOCKED rows carry none), so has_peer_owned_work is False
                    #       for a solo leader → it never relinquishes → its
                    #       self-stranded LEASED/BLOCKED/PENDING_SINK rows still hit
                    #       the raise here or the run-level backstop — never silent.
                    #
                    # The surviving relinquish set is therefore exactly "non-READY,
                    # non-FAILED continuations a peer is carrying" — every one of
                    # which is covered by the run-level and quiescence backstops if
                    # the peer somehow fails to finish it.
                    if self._scheduler_lease_owner_registered:
                        pending_ids = list(pending_items.keys())
                        ready_count = self._scheduler.count_ready_in_set(run_id=self._run_id, work_item_ids=pending_ids)
                        failed_count = self._scheduler.count_failed_in_set(run_id=self._run_id, work_item_ids=pending_ids)
                        if (
                            ready_count == 0
                            and failed_count == 0
                            and self._scheduler.has_peer_owned_work(run_id=self._run_id, caller_owner=self._scheduler_lease_owner)
                        ):
                            relinquished = {
                                work_item_id: f"{item.token.token_id}@{item.current_node_id}"
                                for work_item_id, item in pending_items.items()
                            }
                            logger.info(
                                "Relinquishing %d non-READY/non-FAILED pending continuation(s) to a peer worker "
                                "(run_id=%r, work_item_ids=%r, observed_statuses=%r)",
                                len(pending_items),
                                self._run_id,
                                list(relinquished.keys()),
                                self._scheduler.summarize_active_work(run_id=self._run_id),
                            )
                            pending_items.clear()
                            break
                    stranded = ", ".join(f"{item.token.token_id}@{item.current_node_id}" for item in pending_items.values())
                    active = "; ".join(self._scheduler.summarize_active_work(run_id=self._run_id)) or "<none>"
                    raise OrchestrationInvariantError(
                        f"Scheduler has {len(pending_items)} in-memory continuations for run {self._run_id!r} "
                        f"but no READY work item could be claimed. Stranded: {stranded}. Active journal work: {active}."
                    )
                break

            if claimed.work_item_id in pending_items:
                item = pending_items.pop(claimed.work_item_id)
            else:
                item = self._work_item_from_scheduler(claimed)
            claimed_lease_owner = self._claimed_scheduler_lease_owner(claimed)
            # Mark this claim active so ``_process_single_token``'s per-node
            # heartbeat refreshes the right lease (filigree elspeth-ddde8144b6).
            # Initial last_heartbeat_at is now: claim_ready just set
            # lease_expires_at = now + lease_seconds, so the first heartbeat
            # only needs to fire once heartbeat_seconds has elapsed.
            self._active_claim_work_item_id = claimed.work_item_id
            self._last_heartbeat_at = self._clock.now_utc()
            try:
                try:
                    result, child_items = self._process_single_token(
                        token=item.token,
                        ctx=ctx,
                        current_node_id=item.current_node_id,
                        coalesce_node_id=item.coalesce_node_id,
                        coalesce_name=item.coalesce_name,
                        on_success_sink=item.on_success_sink,
                        attempt_offset=max(claimed.attempt - 1, 0),
                    )
                except SchedulerLeaseLostError as exc:
                    # The lease was reaped by a peer mid-processing. The
                    # original ``work_item_id`` no longer exists (peer rewrote
                    # it under a bumped attempt) or no longer carries this
                    # worker's ``lease_owner``. Issuing ``mark_failed`` would
                    # CAS-fail and cascade into Tier-1 AuditIntegrityError —
                    # the exact failure mode this primitive exists to
                    # eliminate. Abandon the in-flight work, do NOT emit the
                    # (lost) result, and return the results that were already
                    # proven before this lease was lost. The caller's
                    # post-sink scheduler invariant check will refuse run
                    # completion if active work remains. Staged §E.5 loss
                    # records are discarded with the abandoned claim (the
                    # peer's re-drive re-stages them with its own disposition).
                    self._pending_branch_losses.clear()
                    exc.add_note("scheduler lease lost during row processing; in-flight token result was abandoned")
                    return results
                except Exception as processing_exc:
                    try:
                        self._scheduler.mark_failed(
                            work_item_id=claimed.work_item_id,
                            now=self._clock.now_utc(),
                            expected_lease_owner=claimed_lease_owner,
                            branch_loss=self._take_claim_branch_loss(claimed.token_id),
                            worker_id=self._disposition_fence_worker_id(),
                        )
                    except RunWorkerEvictedError as evicted_exc:
                        # The membership fence refused the failure bookkeeping:
                        # this worker was evicted mid-processing, so the peer
                        # reap path owns the item now. Propagate the eviction
                        # signal (followers exit on it) instead of wrapping it
                        # as an audit-integrity crash.
                        evicted_exc.add_note(f"processing exception {type(processing_exc).__name__} was superseded by the eviction refusal")
                        raise
                    except Exception as scheduler_exc:
                        raise AuditIntegrityError(
                            f"Scheduler failed to mark work_item_id={claimed.work_item_id!r} failed after original "
                            f"processing exception {type(processing_exc).__name__}: {processing_exc}. "
                            f"The scheduler failure write raised {type(scheduler_exc).__name__}: {scheduler_exc}."
                        ) from scheduler_exc
                    raise
            finally:
                self._active_claim_work_item_id = None
                self._last_heartbeat_at = None

            for child_item in child_items:
                self._enqueue_scheduler_work_item(child_item, pending_items)

            if result is not None and self._is_buffered_scheduler_result(result):
                self._mark_claimed_scheduler_work_blocked(
                    claimed,
                    item,
                    now=self._clock.now_utc(),
                    queue_key=None,
                    barrier_key=self._barrier_key_for_live_hold(claimed.token_id),
                )
                if isinstance(result, tuple):
                    results.extend(result)
                else:
                    results.append(result)
                # §E.2: ALWAYS take another iteration — the next iteration's
                # journal-first intake adopts the row just marked BLOCKED (and
                # fires any count/condition trigger it satisfies) before the
                # drain may exit.
                continue

            if result is None and not child_items:
                self._mark_claimed_scheduler_work_blocked(claimed, item, now=self._clock.now_utc())
                # §E.2: ALWAYS take another iteration (see the buffered arm).
                continue

            if (sink_bound_result := self._scheduler_sink_bound_result_for_claimed_token(result, claimed.token_id)) is not None:
                self._scheduler.mark_pending_sink(
                    work_item_id=claimed.work_item_id,
                    row_payload_json=self._scheduler.serialize_row_payload(sink_bound_result.token.row_data),
                    sink_name=self._require_scheduler_sink_name(sink_bound_result),
                    outcome=self._require_scheduler_outcome(sink_bound_result).value,
                    path=sink_bound_result.path.value,
                    error_hash=self._scheduler_error_hash(sink_bound_result),
                    error_message=self._scheduler_error_message(sink_bound_result),
                    now=self._clock.now_utc(),
                    expected_lease_owner=claimed_lease_owner,
                    branch_loss=self._take_claim_branch_loss(claimed.token_id),
                    worker_id=self._disposition_fence_worker_id(),
                )
                result = self._with_scheduler_pending_sink_handoff(result, claimed.token_id)
            elif self._scheduler_result_failed_claimed_token(result, claimed.token_id):
                self._scheduler.mark_failed(
                    work_item_id=claimed.work_item_id,
                    now=self._clock.now_utc(),
                    expected_lease_owner=claimed_lease_owner,
                    branch_loss=self._take_claim_branch_loss(claimed.token_id),
                    worker_id=self._disposition_fence_worker_id(),
                )
            else:
                self._scheduler.mark_terminal(
                    work_item_id=claimed.work_item_id,
                    now=self._clock.now_utc(),
                    expected_lease_owner=claimed_lease_owner,
                    branch_loss=self._take_claim_branch_loss(claimed.token_id),
                    worker_id=self._disposition_fence_worker_id(),
                )

            if result is not None:
                if isinstance(result, tuple):
                    results.extend(result)
                else:
                    results.append(result)

            if not recover_pending_sinks and not pending_items:
                break

        return results

    def _drain_preexisting_pending_sinks(self, results: list[RowResult]) -> None:
        """Re-emit PENDING_SINK rows left durable by a prior crashed worker.

        Called once at the top of a recovery drain (``recover_pending_sinks=True``).
        Each claim transitions a PENDING_SINK row to LEASED under this worker's
        lease_owner and appends a reconstructed RowResult to ``results``;
        terminalization happens later via the sink-write callback
        (``mark_sink_bound_scheduler_terminal``).

        ``recover_expired_leases`` is run on every iteration so that a prior
        worker's expired sink-bound lease (recovered as PENDING_SINK by
        scheduler_repository.recover_expired_leases) is picked up in the same
        drain pass. Iteration is bounded by ``MAX_WORK_QUEUE_ITERATIONS``
        because real fault-injected runs can produce hundreds of stranded
        pending-sinks; the cap prevents an unbounded loop if some other code
        path keeps recreating PENDING_SINK rows behind us.

        Only the resume entry point (``drain_scheduled_work``) calls into this
        path. Inside the main claim_ready loop, PENDING_SINK rows produced by
        ``mark_pending_sink`` MUST NOT be re-claimed here — see
        ``_drain_scheduler_claims`` docstring.
        """
        iterations = 0
        while True:
            iterations += 1
            if iterations > MAX_WORK_QUEUE_ITERATIONS:
                raise RuntimeError(
                    f"Pending-sink recovery exceeded {MAX_WORK_QUEUE_ITERATIONS} iterations. "
                    "Possible infinite loop in PENDING_SINK recovery."
                )

            now = self._clock.now_utc()
            self._scheduler.recover_expired_leases(
                run_id=self._run_id,
                now=now,
                caller_owner=self._scheduler_lease_owner,
                coordination_token=self._coordination_token,
            )
            repaired = self._scheduler.terminalize_pending_sinks_with_terminal_outcomes(
                run_id=self._run_id,
                now=now,
                caller_owner=self._scheduler_lease_owner,
                coordination_token=self._require_coordination_token(),
            )
            if repaired:
                continue
            pending_sink = self._scheduler.claim_pending_sink(
                run_id=self._run_id,
                lease_owner=self._scheduler_lease_owner,
                lease_seconds=self._scheduler_lease_seconds,
                now=now,
            )
            if pending_sink is None:
                return
            results.append(self._row_result_from_pending_sink(pending_sink))

    def mark_sink_bound_scheduler_terminal(self, token_id: str) -> None:
        """Terminalize scheduler work after sink outcome durability."""
        terminalized = self._scheduler.mark_pending_sink_terminal(
            run_id=self._run_id,
            token_id=token_id,
            now=self._clock.now_utc(),
            expected_lease_owner=self._scheduler_lease_owner,
            coordination_token=self._require_coordination_token(),
        )
        if terminalized != 1:
            raise AuditIntegrityError(
                f"Scheduler pending-sink terminalization for run_id={self._run_id!r} token_id={token_id!r} "
                f"terminalized {terminalized} rows; expected exactly one durable scheduler handoff."
            )

    def mark_sink_bound_scheduler_terminal_many(self, token_ids: tuple[str, ...]) -> None:
        """Terminalize a durable sink batch after sink outcome durability."""
        terminalized = self._scheduler.mark_pending_sink_terminal_many(
            run_id=self._run_id,
            token_ids=token_ids,
            now=self._clock.now_utc(),
            expected_lease_owner=self._scheduler_lease_owner,
            coordination_token=self._require_coordination_token(),
        )
        if terminalized != len(token_ids):
            raise AuditIntegrityError(
                f"Scheduler pending-sink batch terminalization for run_id={self._run_id!r} terminalized "
                f"{terminalized} rows for {len(token_ids)} durable sink outcomes."
            )

    def _row_result_from_pending_sink(self, scheduled: TokenWorkItem) -> RowResult:
        """Rebuild a sink-bound row result without re-running its producer node."""
        if scheduled.pending_sink_name is None or scheduled.pending_outcome is None or scheduled.pending_path is None:
            raise AuditIntegrityError(f"Scheduler pending sink work_item_id={scheduled.work_item_id!r} is missing sink outcome metadata.")
        # Attempt-offset derivation: if the original run already opened a SINK
        # node_state for this token (the sink write itself crashed after
        # opening attempt 0), the re-driven sink write must run at the bumped
        # attempt or its node_state insert collides with audited history.
        # Scoped to the sink step — the only step a pending-sink re-drive
        # writes; producer-node attempts must not inflate the offset.
        max_attempts = self._execution.get_max_node_state_attempts(
            self._run_id,
            [scheduled.token_id],
            step_index=self.resolve_sink_step(),
        )
        attempt_offset = max_attempts.get(scheduled.token_id, -1) + 1
        if attempt_offset > 0 and self._resume_checkpoint_id is None:
            raise AuditIntegrityError(
                f"Scheduler pending sink token {scheduled.token_id!r} (run {self._run_id!r}) already has "
                f"node_states up to attempt {attempt_offset - 1} but this processor has no resume checkpoint "
                "provenance; re-driving without a resume_checkpoint_id would write an unattributed retry attempt."
            )
        token = TokenInfo(
            row_id=scheduled.row_id,
            token_id=scheduled.token_id,
            row_data=self._scheduler.deserialize_row_payload(scheduled.row_payload_json),
            branch_name=scheduled.branch_name,
            fork_group_id=scheduled.fork_group_id,
            join_group_id=scheduled.join_group_id,
            expand_group_id=scheduled.expand_group_id,
            resume_attempt_offset=attempt_offset,
            resume_checkpoint_id=self._resume_checkpoint_id if attempt_offset > 0 else None,
        )
        is_on_error_routed = scheduled.pending_path == TerminalPath.ON_ERROR_ROUTED.value
        if is_on_error_routed and not scheduled.pending_error_hash:
            # The parking disposition always persists the originating error
            # hash for routed failures, so its absence is audit corruption —
            # refuse to replay with a recomputed (synthetic) hash
            # (filigree elspeth-d74d19f901).
            raise AuditIntegrityError(
                f"Scheduler pending sink work_item_id={scheduled.work_item_id!r} is ON_ERROR_ROUTED but carries no "
                "pending_error_hash; the replayed outcome cannot preserve the originally-audited error hash."
            )
        return RowResult(
            token=token,
            final_data=token.row_data,
            outcome=TerminalOutcome(scheduled.pending_outcome),
            path=TerminalPath(scheduled.pending_path),
            sink_name=scheduled.pending_sink_name,
            error=FailureInfo(exception_type="ResumedPendingSink", message=scheduled.pending_error_message or "")
            if is_on_error_routed
            else None,
            scheduler_pending_sink=True,
            authoritative_error_hash=scheduled.pending_error_hash if is_on_error_routed else None,
        )

    def _mark_claimed_scheduler_work_blocked(
        self,
        claimed: TokenWorkItem,
        item: WorkItem,
        *,
        now: datetime,
        queue_key: str | None = None,
        barrier_key: str | None = None,
    ) -> None:
        """Persist BLOCKED state only when resume has a durable release key."""
        queue_key = self._queue_key_for_blocked_item(item) if queue_key is None and barrier_key is None else queue_key
        barrier_key = self._barrier_key_for_blocked_item(item) if queue_key is None and barrier_key is None else barrier_key
        if queue_key is None and barrier_key is None:
            raise OrchestrationInvariantError(
                f"Work item {claimed.work_item_id!r} (token={item.token.token_id!r}, node={item.current_node_id!r}) "
                "produced no result and no children, but has no queue or barrier key; cannot be unblocked. "
                "This is a processor bug."
            )
        self._scheduler.mark_blocked(
            work_item_id=claimed.work_item_id,
            queue_key=queue_key,
            barrier_key=barrier_key,
            now=now,
            expected_lease_owner=self._claimed_scheduler_lease_owner(claimed),
            worker_id=self._disposition_fence_worker_id(),
        )

    def _disposition_fence_worker_id(self) -> str | None:
        """Membership-fence identity for drain disposition writes.

        ADR-030 §G parity (filigree elspeth-ba7b2cc25d): registered workers
        thread their identity so an evicted worker's dispositions are refused
        at the scheduler; unregistered (N=0 / legacy / test-fixture) builds
        pass None and stay unfenced — the same gating ``enqueue_ready``'s
        slice-5 fence uses.
        """
        return self._scheduler_lease_owner if self._scheduler_lease_owner_registered else None

    @staticmethod
    def _claimed_scheduler_lease_owner(claimed: TokenWorkItem) -> str:
        """Return the proven lease owner for a claimed scheduler item."""
        if claimed.lease_owner is None:
            raise AuditIntegrityError(
                f"Scheduler claimed work_item_id={claimed.work_item_id!r} without a lease_owner; "
                "cannot perform an owner-fenced state transition."
            )
        return claimed.lease_owner

    def _take_pending_branch_losses(self) -> tuple[BranchLossSpec, ...]:
        """Take-and-clear every staged §E.5 branch-loss record.

        Consumed by ``_complete_aggregation_flush`` so empty-emission losses
        ride the flush's barrier-completion transaction.
        """
        if not self._pending_branch_losses:
            return ()
        losses = tuple(self._pending_branch_losses)
        self._pending_branch_losses.clear()
        return losses

    def _take_claim_branch_loss(self, claimed_token_id: str) -> BranchLossSpec | None:
        """Take the staged §E.5 loss record for the claim being disposed.

        A single claim loses at most ONE branch (the claimed token reaches
        exactly one lossy terminal arm); the record rides the claim's own
        ``mark_failed`` / ``mark_pending_sink`` / ``mark_terminal``
        transaction. More than one staged record, or a record for a different
        token, is a processor bug.
        """
        if not self._pending_branch_losses:
            return None
        if len(self._pending_branch_losses) > 1:
            staged = [(spec.token_id, spec.branch_name) for spec in self._pending_branch_losses]
            raise OrchestrationInvariantError(
                f"Claim disposition for token {claimed_token_id!r} found {len(self._pending_branch_losses)} staged "
                f"branch-loss records ({staged!r}); one claim loses at most one branch. Processor bug."
            )
        spec = self._pending_branch_losses.pop()
        if spec.token_id != claimed_token_id:
            raise OrchestrationInvariantError(
                f"Claim disposition for token {claimed_token_id!r} found a staged branch-loss record for "
                f"token {spec.token_id!r} (branch {spec.branch_name!r}); the loss must ride its own token's "
                "disposition. Processor bug."
            )
        return spec

    def _heartbeat_active_claim(self) -> None:
        """Refresh the active scheduler lease if heartbeat interval has elapsed.

        Called from ``_process_single_token`` on every node-iteration boundary
        (ADR-026 RC6 multi-worker, filigree elspeth-ddde8144b6). The actual
        DB write fires at most once per ``scheduler_heartbeat_seconds`` so
        fast plugin chains do not incur a write per node.

        No-op when no claim is active or the interval has not yet elapsed.

        Raises:
            SchedulerLeaseLostError: the lease was reaped or reassigned by a
                peer between the claim and this heartbeat. The caller (the
                drain loop) catches this specifically and skips both the
                in-flight result emission and the terminal ``mark_*`` write —
                issuing either would CAS-fail and cascade into a Tier-1
                AuditIntegrityError, which is the exact failure mode this
                primitive exists to eliminate.

        **Single-plugin-call limitation.** This heartbeat fires *between*
        plugin calls, not *during* a single plugin call. If one plugin call
        exceeds ``scheduler_lease_seconds`` on its own, the lease still
        expires while that call is in-flight. The operator must size
        ``scheduler_lease_seconds`` to bracket the longest expected
        single-plugin call. Sub-call-level protection (option b watchdog or
        thread-based heartbeat) is a separate concern and not in scope for
        the ticket's option (a).
        """
        if self._active_claim_work_item_id is None:
            return
        now = self._clock.now_utc()
        if self._last_heartbeat_at is not None and (now - self._last_heartbeat_at).total_seconds() < self._scheduler_heartbeat_seconds:
            return
        self._scheduler.heartbeat_lease(
            run_id=self._run_id,
            work_item_id=self._active_claim_work_item_id,
            lease_owner=self._scheduler_lease_owner,
            lease_seconds=self._scheduler_lease_seconds,
            now=now,
        )
        self._last_heartbeat_at = now

    def _barrier_key_for_live_hold(self, token_id: str) -> str:
        """Resolve the barrier that owns a token about to be marked BLOCKED.

        §E.2: the historical derivation read the in-claim BUFFERED outcome's
        batch_id, which no longer exists at block time — the producer
        (``_process_batch_aggregation_node`` / ``_maybe_coalesce_token``)
        stashed the barrier_key alongside the live token instead.
        """
        try:
            hold = self._live_barrier_holds[token_id]
        except KeyError:
            raise AuditIntegrityError(
                f"Buffered scheduler result for token {token_id!r} has no live barrier hold stash; "
                "cannot persist a durable release barrier. Processor bug."
            ) from None
        return hold.barrier_key

    @staticmethod
    def _is_buffered_scheduler_result(result: RowResult | tuple[RowResult, ...] | None) -> bool:
        """Return whether a scheduler result is an active aggregation buffer."""
        if result is None:
            return False
        if _is_result_tuple(result):
            return bool(result) and all(item.outcome is None and item.path is TerminalPath.BUFFERED for item in result)
        return result.outcome is None and result.path is TerminalPath.BUFFERED

    @staticmethod
    def _scheduler_result_failed_claimed_token(result: RowResult | tuple[RowResult, ...] | None, claimed_token_id: str) -> bool:
        """Return whether the claimed scheduler token itself reached FAILURE."""
        if result is None:
            return False
        result_items = result if _is_result_tuple(result) else (result,)
        return any(item.token.token_id == claimed_token_id and item.outcome is TerminalOutcome.FAILURE for item in result_items)

    @staticmethod
    def _is_scheduler_sink_bound_result(result: RowResult) -> bool:
        return result.sink_name is not None and result.path in {
            TerminalPath.DEFAULT_FLOW,
            TerminalPath.GATE_ROUTED,
            TerminalPath.ON_ERROR_ROUTED,
            TerminalPath.COALESCED,
        }

    @classmethod
    def _scheduler_sink_bound_result_for_claimed_token(
        cls,
        result: RowResult | tuple[RowResult, ...] | None,
        claimed_token_id: str,
    ) -> RowResult | None:
        """Return the claimed token's sink-bound result, if any."""
        if result is None:
            return None
        result_items = result if _is_result_tuple(result) else (result,)
        for item in result_items:
            if item.token.token_id != claimed_token_id:
                continue
            if cls._is_scheduler_sink_bound_result(item):
                return item
        return None

    @staticmethod
    def _with_scheduler_pending_sink_handoffs(
        result: RowResult | tuple[RowResult, ...] | None,
        token_ids: frozenset[str],
    ) -> RowResult | tuple[RowResult, ...]:
        """Mark every requested token result as scheduler pending-sink backed."""
        if result is None:
            raise OrchestrationInvariantError(
                f"Cannot mark scheduler pending-sink handoffs for token_ids={sorted(token_ids)!r}: result is None."
            )
        if not token_ids:
            return result
        if isinstance(result, tuple):
            tagged: list[RowResult] = []
            matched_token_ids: set[str] = set()
            for item in result:
                if item.token.token_id in token_ids:
                    tagged.append(replace(item, scheduler_pending_sink=True))
                    matched_token_ids.add(item.token.token_id)
                else:
                    tagged.append(item)
            missing_token_ids = token_ids - matched_token_ids
            if missing_token_ids:
                raise OrchestrationInvariantError(
                    "Cannot mark scheduler pending-sink handoffs: "
                    f"token_ids={sorted(missing_token_ids)!r} were not present in tuple result."
                )
            return tuple(tagged)
        if result.token.token_id not in token_ids:
            raise OrchestrationInvariantError(
                "Cannot mark scheduler pending-sink handoff: "
                f"token_ids={sorted(token_ids)!r} do not include result token {result.token.token_id!r}."
            )
        return replace(result, scheduler_pending_sink=True)

    @staticmethod
    def _with_scheduler_pending_sink_handoff(
        result: RowResult | tuple[RowResult, ...] | None,
        claimed_token_id: str,
    ) -> RowResult | tuple[RowResult, ...]:
        """Mark only the claimed token result as scheduler pending-sink backed."""
        return RowProcessor._with_scheduler_pending_sink_handoffs(result, frozenset((claimed_token_id,)))

    @staticmethod
    def _require_scheduler_sink_name(result: RowResult) -> str:
        if result.sink_name is None:
            raise OrchestrationInvariantError(f"Scheduler sink-bound result missing sink_name for token {result.token.token_id!r}")
        return result.sink_name

    @staticmethod
    def _require_scheduler_outcome(result: RowResult) -> TerminalOutcome:
        if result.outcome is None:
            raise OrchestrationInvariantError(f"Scheduler sink-bound result missing terminal outcome for token {result.token.token_id!r}")
        return result.outcome

    @staticmethod
    def _scheduler_error_hash(result: RowResult) -> str | None:
        if result.path is not TerminalPath.ON_ERROR_ROUTED:
            return None
        if result.error is None:
            raise OrchestrationInvariantError(f"Scheduler ON_ERROR_ROUTED result missing error for token {result.token.token_id!r}")
        return compute_error_hash(result.error.message, exception_type=result.error.exception_type)

    @staticmethod
    def _scheduler_error_message(result: RowResult) -> str | None:
        if result.path is not TerminalPath.ON_ERROR_ROUTED:
            return None
        if result.error is None:
            raise OrchestrationInvariantError(f"Scheduler ON_ERROR_ROUTED result missing error for token {result.token.token_id!r}")
        return result.error.message

    def _enqueue_scheduler_work_item(
        self,
        item: WorkItem,
        pending_items: dict[str, WorkItem],
        *,
        claim_immediately: bool = False,
    ) -> TokenWorkItem:
        """Persist a READY scheduler item and retain the live token payload."""
        available_at = self._clock.now_utc()
        node_id = self._scheduler_node_id(item.current_node_id)
        step_index = self._scheduler_step_index(item.current_node_id)
        ingest_sequence = self._data_flow.resolve_row_ingest_sequence(item.token.row_id)
        row_payload_json = self._scheduler.serialize_row_payload(item.token.row_data)
        queue_key = self._queue_key_for_blocked_item(item)
        barrier_key = self._barrier_key_for_blocked_item(item)
        coalesce_node_id = str(item.coalesce_node_id) if item.coalesce_node_id is not None else None
        coalesce_name = str(item.coalesce_name) if item.coalesce_name is not None else None
        if claim_immediately:
            scheduled = self._scheduler.enqueue_ready_claimed(
                run_id=self._run_id,
                token_id=item.token.token_id,
                row_id=item.token.row_id,
                node_id=node_id,
                step_index=step_index,
                ingest_sequence=ingest_sequence,
                row_payload_json=row_payload_json,
                available_at=available_at,
                queue_key=queue_key,
                barrier_key=barrier_key,
                on_success_sink=item.on_success_sink,
                branch_name=item.token.branch_name,
                fork_group_id=item.token.fork_group_id,
                join_group_id=item.token.join_group_id,
                expand_group_id=item.token.expand_group_id,
                coalesce_node_id=coalesce_node_id,
                coalesce_name=coalesce_name,
                lease_owner=self._scheduler_lease_owner,
                lease_seconds=self._scheduler_lease_seconds,
                now=available_at,
            )
        else:
            scheduled = self._scheduler.enqueue_ready(
                run_id=self._run_id,
                token_id=item.token.token_id,
                row_id=item.token.row_id,
                node_id=node_id,
                step_index=step_index,
                ingest_sequence=ingest_sequence,
                row_payload_json=row_payload_json,
                available_at=available_at,
                queue_key=queue_key,
                barrier_key=barrier_key,
                on_success_sink=item.on_success_sink,
                branch_name=item.token.branch_name,
                fork_group_id=item.token.fork_group_id,
                join_group_id=item.token.join_group_id,
                expand_group_id=item.token.expand_group_id,
                coalesce_node_id=coalesce_node_id,
                coalesce_name=coalesce_name,
                # Membership fence (ADR-030 §G, slice 5): thread the registered
                # worker identity so an evicted RowProcessor cannot enqueue READY
                # items that no active worker will claim. The fence is active only
                # when scheduler_lease_owner was explicitly registered in run_workers
                # (production multi-worker path: leaders via run_core.py, followers
                # via follower.py). Legacy / single-worker / test-fixture builds
                # pass scheduler_lease_owner=None → auto-generate an unregistered
                # identity → _scheduler_lease_owner_registered=False → fence skipped.
                worker_id=self._scheduler_lease_owner if self._scheduler_lease_owner_registered else None,
            )
        if scheduled.status is TokenWorkStatus.READY or (
            scheduled.status is TokenWorkStatus.LEASED and scheduled.lease_owner == self._scheduler_lease_owner
        ):
            pending_items[scheduled.work_item_id] = item
        return scheduled

    def _work_item_from_scheduler(self, scheduled: TokenWorkItem) -> WorkItem:
        """Rehydrate a scheduler work item from its durable payload snapshot."""
        current_node_id = None if scheduled.node_id is None or scheduled.node_id == "__terminal__" else NodeID(scheduled.node_id)
        token = TokenInfo(
            row_id=scheduled.row_id,
            token_id=scheduled.token_id,
            row_data=self._scheduler.deserialize_row_payload(scheduled.row_payload_json),
            branch_name=scheduled.branch_name,
            fork_group_id=scheduled.fork_group_id,
            join_group_id=scheduled.join_group_id,
            expand_group_id=scheduled.expand_group_id,
        )
        return self._nav.create_work_item(
            token=token,
            current_node_id=current_node_id,
            coalesce_node_id=NodeID(scheduled.coalesce_node_id) if scheduled.coalesce_node_id is not None else None,
            coalesce_name=CoalesceName(scheduled.coalesce_name) if scheduled.coalesce_name is not None else None,
            on_success_sink=scheduled.on_success_sink,
        )

    def _scheduler_node_id(self, node_id: NodeID | None) -> str | None:
        """Return the persisted node cursor for a work item."""
        if node_id is None:
            return None
        return str(node_id)

    def _scheduler_step_index(self, node_id: NodeID | None) -> int:
        """Return the durable scheduler ordering step for a work item."""
        if node_id is None:
            return max(self._node_step_map.values(), default=0) + 1
        if node_id in self._node_step_map:
            return self._node_step_map[node_id]
        if node_id == self._source_node_id:
            return 0
        raise OrchestrationInvariantError(f"Cannot schedule unknown node cursor {node_id!r}")

    def _queue_key_for_blocked_item(self, item: WorkItem) -> str | None:
        """Return a queue key for structural queue blocking, if applicable."""
        if item.current_node_id is None:
            return None
        if item.current_node_id in self._structural_node_ids and item.coalesce_name is None:
            return str(item.current_node_id)
        return None

    def _barrier_key_for_blocked_item(self, item: WorkItem) -> str | None:
        """Return a barrier key for coalesce/aggregation blocking, if applicable.

        On a leader this checks ``_aggregation_settings`` (which is populated).
        On a follower ``_aggregation_settings`` is empty but
        ``_follower_barrier_node_ids`` carries the aggregation node IDs; both
        paths produce the same barrier key (``str(node_id)``).
        """
        if item.current_node_id in self._aggregation_settings:
            return str(item.current_node_id)
        # ADR-030 §B (slice 5): follower path — aggregation_settings is empty
        # but follower_barrier_node_ids carries the aggregation node IDs.
        if item.current_node_id in self._follower_barrier_node_ids:
            return str(item.current_node_id)
        if item.coalesce_name is not None:
            return str(item.coalesce_name)
        return None

    def _handle_transform_node(
        self,
        transform: TransformProtocol,
        current_token: TokenInfo,
        ctx: PluginContext,
        node_id: NodeID,
        child_items: list[WorkItem],
        coalesce_node_id: NodeID | None,
        coalesce_name: CoalesceName | None,
        current_on_success_sink: str,
        attempt_offset: int = 0,
    ) -> _TransformOutcome:
        """Handle a single transform node: execute with retry, route errors, handle multi-row.

        Args:
            transform: The transform plugin to execute.
            current_token: Token being processed through the DAG.
            ctx: Plugin context for the current run.
            node_id: Current DAG node ID (needed for deaggregation expand_token() and
                child work item creation via create_continuation_work_item()).
            child_items: Mutable list — deaggregation appends child work items here.
            coalesce_node_id: Coalesce barrier node for fork branches (or None).
            coalesce_name: Coalesce point name for fork branches (or None).
            current_on_success_sink: Current sink name, may be updated by transform.on_success.

        Resume state (attempt offset and checkpoint provenance) is carried on
        current_token.resume_attempt_offset and current_token.resume_checkpoint_id
        and flow through to execute_transform without explicit threading.

        Returns:
            _TransformContinue: Token should advance to next node (updated token + updated sink).
            _TransformTerminal: Token reached terminal state (FAILED, QUARANTINED, ROUTED, or EXPANDED).
        """
        # 1. Execute transform with retry
        try:
            transform_result, current_token, error_sink = self._execute_transform_with_retry(
                transform=transform,
                token=current_token,
                ctx=ctx,
                attempt_offset=attempt_offset,
            )
            # Emit TransformCompleted telemetry AFTER Landscape recording succeeds
            # (Landscape recording happens inside _execute_transform_with_retry)
            self._emit_transform_completed(
                token=current_token,
                transform=transform,
                transform_result=transform_result,
            )
        except MaxRetriesExceeded as e:
            # All retries exhausted - return FAILED outcome
            error_hash = compute_error_hash(str(e), exception_type=type(e).__name__)
            self._data_flow.record_token_outcome(
                ref=TokenRef(token_id=current_token.token_id, run_id=self._run_id),
                outcome=TerminalOutcome.FAILURE,
                path=TerminalPath.UNROUTED,
                error_hash=error_hash,
            )
            # Emit TokenCompleted telemetry AFTER Landscape recording
            self._emit_token_completed(
                current_token,
                outcome=TerminalOutcome.FAILURE,
                path=TerminalPath.UNROUTED,
            )
            # Notify coalesce if this is a forked branch
            sibling_results = self._notify_coalesce_of_lost_branch(
                current_token,
                f"max_retries_exceeded:{e}",
                child_items,
            )
            current_result = RowResult(
                token=current_token,
                final_data=current_token.row_data,
                outcome=TerminalOutcome.FAILURE,
                path=TerminalPath.UNROUTED,
                error=FailureInfo.from_max_retries_exceeded(e),
            )
            if sibling_results:
                return _TransformTerminal(result=(current_result, *sibling_results))
            return _TransformTerminal(result=current_result)

        # 2. Handle error status
        if transform_result.status == "error":
            return self._handle_transform_error_status(
                transform_result,
                current_token,
                error_sink,
                child_items,
            )

        # 3. Track on_success for sink routing at end of chain
        updated_sink = current_on_success_sink
        if transform.on_success is not None:
            updated_sink = transform.on_success

        # 4. Handle multi-row output (deaggregation)
        # NOTE: This is ONLY for non-aggregation transforms. Aggregation
        # transforms route through _process_batch_aggregation_node() above.
        if transform_result.is_multi_row:
            if transform_result.rows is None:
                raise OrchestrationInvariantError("is_multi_row guarantees rows is not None")
            if len(transform_result.rows) == 0:
                self._record_dropped_by_filter_outcome(
                    token=current_token,
                    transform_name=transform.name,
                    node_id=node_id,
                    path_label="after success_empty()",
                )
                self._emit_token_completed(
                    current_token,
                    outcome=TerminalOutcome.SUCCESS,
                    path=TerminalPath.FILTER_DROPPED,
                )
                sibling_results = self._notify_coalesce_of_lost_branch(
                    current_token,
                    "dropped_by_filter",
                    child_items,
                )
                current_result = RowResult(
                    token=current_token,
                    final_data=current_token.row_data,
                    outcome=TerminalOutcome.SUCCESS,
                    path=TerminalPath.FILTER_DROPPED,
                )
                if sibling_results:
                    return _TransformTerminal(result=(current_result, *sibling_results))
                return _TransformTerminal(result=current_result)

            # Validate transform is allowed to create tokens
            if not transform.creates_tokens:
                raise RuntimeError(
                    f"Transform '{transform.name}' returned multi-row result "
                    f"but has creates_tokens=False. Either set creates_tokens=True "
                    f"or return single row via TransformResult.success(row). "
                    f"(Multi-row is allowed in aggregation passthrough mode.)"
                )

            # Deaggregation: create child tokens for each output row
            # NOTE: Parent EXPANDED outcome is recorded atomically in expand_token()
            # Contract consistency is enforced by TransformResult.success_multi()
            output_contract = transform_result.rows[0].contract
            child_tokens, _expand_group_id = self._token_manager.expand_token(
                parent_token=current_token,
                expanded_rows=[r.to_dict() for r in transform_result.rows],
                output_contract=output_contract,
                node_id=node_id,
                run_id=self._run_id,
            )

            # Queue each child for continued processing.
            # Pass updated_sink so terminal children inherit the
            # expanding transform's sink instead of defaulting to source_on_success.
            # Children born during a re-drive get fresh token_ids with no prior node_states,
            # so they use the default resume_attempt_offset=0 / resume_checkpoint_id=None.
            for child_token in child_tokens:
                child_coalesce_name = coalesce_name if coalesce_name is not None and child_token.branch_name is not None else None
                child_items.append(
                    self._nav.create_continuation_work_item(
                        token=child_token,
                        current_node_id=node_id,
                        coalesce_name=child_coalesce_name,
                        on_success_sink=updated_sink,
                    )
                )

            # NOTE: Parent EXPANDED outcome is recorded atomically in expand_token()
            # to eliminate crash window between child creation and outcome recording.
            return _TransformTerminal(
                result=RowResult(
                    token=current_token,
                    final_data=current_token.row_data,
                    outcome=TerminalOutcome.TRANSIENT,
                    path=TerminalPath.EXPAND_PARENT,
                )
            )

        # 5. Single row success — continue to next node
        # (current_token already updated by _execute_transform_with_retry)
        return _TransformContinue(updated_token=current_token, updated_sink=updated_sink)

    def _handle_transform_error_status(
        self,
        transform_result: TransformResult,
        current_token: TokenInfo,
        error_sink: str | None,
        child_items: list[WorkItem],
    ) -> _TransformTerminal:
        """Handle transform error status: quarantine (discard) or route to error sink.

        Args:
            transform_result: The failed transform result.
            current_token: Token that failed processing.
            error_sink: "discard" for quarantine, or a sink name for error routing.
            child_items: Mutable list — coalesce notifications may append child work items.

        Returns:
            _TransformTerminal with QUARANTINED or ROUTED_ON_ERROR outcome.
        """
        if error_sink == "discard":
            # Intentionally discarded - QUARANTINED
            # The QUARANTINED path tolerates an "unknown_error" fallback for
            # historical reasons; do NOT extend that fallback to ROUTED_ON_ERROR
            # below — see the offensive guard in the routed branch.
            error_detail = str(transform_result.reason) if transform_result.reason else "unknown_error"
            quarantine_error_hash = compute_error_hash(error_detail)
            self._data_flow.record_token_outcome(
                ref=TokenRef(token_id=current_token.token_id, run_id=self._run_id),
                outcome=TerminalOutcome.FAILURE,
                path=TerminalPath.QUARANTINED_AT_SOURCE,
                error_hash=quarantine_error_hash,
            )
            # Emit TokenCompleted telemetry AFTER Landscape recording
            self._emit_token_completed(
                current_token,
                outcome=TerminalOutcome.FAILURE,
                path=TerminalPath.QUARANTINED_AT_SOURCE,
            )
            # Notify coalesce if this is a forked branch
            sibling_results = self._notify_coalesce_of_lost_branch(
                current_token,
                f"quarantined:{error_detail}",
                child_items,
            )
            current_result = RowResult(
                token=current_token,
                final_data=current_token.row_data,
                outcome=TerminalOutcome.FAILURE,
                path=TerminalPath.QUARANTINED_AT_SOURCE,
            )
            if sibling_results:
                return _TransformTerminal(result=(current_result, *sibling_results))
            return _TransformTerminal(result=current_result)

        # Routed to error sink — emit ROUTED_ON_ERROR (DIVERT semantics).
        # NOTE: Do NOT record the outcome here - the token hasn't been written yet.
        # SinkExecutor.write() records the outcome AFTER sink durability is achieved.
        #
        # Offensive: refuse to fabricate Tier-1 audit data. If the upstream
        # transform did not provide a reason, that is a producer bug; crashing
        # here is correct because emitting `FailureInfo.message="unknown_error"`
        # would create a deterministic error_hash collision across unrelated
        # falsy-error failures and falsify the audit trail.
        if not transform_result.reason:
            raise OrchestrationInvariantError(
                "ROUTED_ON_ERROR requires transform_result.reason; refusing to "
                "fabricate FailureInfo.message='unknown_error' for audit hashing"
            )
        error_detail = str(transform_result.reason)

        sibling_results = self._notify_coalesce_of_lost_branch(
            current_token,
            f"error_routed:{error_detail}",
            child_items,
        )
        # Capture the originating transform error so the audit trail records both
        # sink_name and error_hash on the ROUTED_ON_ERROR outcome (mirror of DIVERTED's
        # contract). The accumulator converts FailureInfo.message -> error_hash before
        # the pending-sink record is handed to SinkExecutor for durable recording.
        failure = FailureInfo(exception_type="TransformError", message=error_detail)
        current_result = RowResult(
            token=current_token,
            final_data=current_token.row_data,
            outcome=TerminalOutcome.FAILURE,
            path=TerminalPath.ON_ERROR_ROUTED,
            sink_name=error_sink,
            error=failure,
        )
        if sibling_results:
            return _TransformTerminal(result=(current_result, *sibling_results))
        return _TransformTerminal(result=current_result)

    def _handle_gate_node(
        self,
        gate: GateSettings,
        current_token: TokenInfo,
        ctx: PluginContext,
        node_id: NodeID,
        child_items: list[WorkItem],
        coalesce_node_id: NodeID | None,
        coalesce_name: CoalesceName | None,
        current_on_success_sink: str,
    ) -> _GateOutcome:
        """Handle a gate node: evaluate, then fork/route/divert/continue.

        Args:
            gate: Gate configuration to evaluate.
            current_token: Token being processed through the DAG.
            ctx: Plugin context for the current run.
            node_id: Current DAG node ID (passed to gate executor and used for
                fork child work item creation).
            child_items: Mutable list — fork paths append child work items here.
            coalesce_node_id: Coalesce barrier node for fork branches (or None).
            coalesce_name: Coalesce point name for fork branches (or None).
            current_on_success_sink: Current sink name, carried forward or overridden by jumps.

        Returns:
            _GateTerminal: Gate routed to sink, forked to paths, or diverted (contains result + child_items populated).
            _GateContinue: Gate says continue — updated_token, updated_sink, and optional next_node_id for jumps.
        """
        # 1. Execute gate
        outcome = self._gate_executor.execute_config_gate(
            gate_config=gate,
            node_id=node_id,
            token=current_token,
            ctx=ctx,
            token_manager=self._token_manager,
        )
        current_token = outcome.updated_token

        # 2. Emit GateEvaluated telemetry AFTER Landscape recording succeeds
        # (Landscape recording happens inside execute_config_gate)
        self._emit_gate_evaluated(
            token=current_token,
            gate_name=gate.name,
            gate_node_id=node_id,
            routing_mode=outcome.result.action.mode,
            destinations=self._get_gate_destinations(outcome),
        )

        # 3. Check if gate routed to a sink
        if outcome.sink_name is not None:
            # NOTE: Do NOT record ROUTED outcome here - the token hasn't been written yet.
            # SinkExecutor.write() records the outcome AFTER sink durability is achieved.
            # Notify coalesce if this is a forked branch
            sibling_results = self._notify_coalesce_of_lost_branch(
                current_token,
                f"gate_routed_to_sink:{outcome.sink_name}",
                child_items,
            )
            current_result = RowResult(
                token=current_token,
                final_data=current_token.row_data,
                outcome=TerminalOutcome.SUCCESS,
                path=TerminalPath.GATE_ROUTED,
                sink_name=outcome.sink_name,
            )
            if sibling_results:
                return _GateTerminal(result=(current_result, *sibling_results))
            return _GateTerminal(result=current_result)

        if outcome.discarded:
            self._record_gate_discarded_outcome(
                token=current_token,
                gate_name=gate.name,
                node_id=node_id,
            )
            with best_effort(
                "TokenCompleted telemetry after gate discard audit",
                run_id=self._run_id,
                token_id=current_token.token_id,
                gate_node_id=node_id,
                gate_name=gate.name,
            ):
                self._emit_token_completed(
                    current_token,
                    outcome=TerminalOutcome.SUCCESS,
                    path=TerminalPath.GATE_DISCARDED,
                )
            sibling_results = self._notify_coalesce_of_lost_branch(
                current_token,
                "gate_discarded",
                child_items,
            )
            current_result = RowResult(
                token=current_token,
                final_data=current_token.row_data,
                outcome=TerminalOutcome.SUCCESS,
                path=TerminalPath.GATE_DISCARDED,
            )
            if sibling_results:
                return _GateTerminal(result=(current_result, *sibling_results))
            return _GateTerminal(result=current_result)

        # 4. Fork to paths
        if outcome.result.action.kind == RoutingKind.FORK_TO_PATHS:
            return self._handle_gate_fork(outcome, current_token, node_id, child_items)

        # 5. Jump to specific node
        if outcome.next_node_id is not None:
            # Validate jump target exists in the DAG (our data — crash on invariant violation).
            # Without this check, a nonexistent target silently passes the coalesce ordering
            # check below (both .get() calls return None → condition is False) and only fails
            # one iteration later with a less informative error from resolve_plugin_for_node().
            if outcome.next_node_id not in self._node_step_map:
                raise OrchestrationInvariantError(
                    f"Gate at node '{node_id}' jumped token '{current_token.token_id}' to "
                    f"node '{outcome.next_node_id}' which is not in the DAG step map. "
                    f"Known nodes: {sorted(self._node_step_map.keys())}"
                )

            updated_sink = current_on_success_sink
            resolved_sink = self._nav.resolve_jump_target_sink(outcome.next_node_id)
            if resolved_sink is not None:
                updated_sink = resolved_sink

            # Re-validate coalesce ordering invariant after gate jump.
            # The initial check at entry only validates the starting node.
            # A gate jump can move the token past its coalesce node,
            # which would silently bypass join handling.
            #
            # IMPORTANT: Use outcome.next_node_id (not the caller's node_id param)
            # because we're validating the JUMP TARGET, not the current position.
            if coalesce_node_id is not None:
                jump_target_step = self._node_step_map[outcome.next_node_id]
                coalesce_barrier_step = self._node_step_map[coalesce_node_id]
                if jump_target_step > coalesce_barrier_step:
                    raise OrchestrationInvariantError(
                        f"Gate jump moved token '{current_token.token_id}' to node '{outcome.next_node_id}' "
                        f"(step {jump_target_step}) which is past its coalesce node '{coalesce_node_id}' "
                        f"(step {coalesce_barrier_step}). This would bypass join handling."
                    )

            return _GateContinue(
                updated_token=current_token,
                updated_sink=updated_sink,
                next_node_id=outcome.next_node_id,
            )

        # 6. CONTINUE: config gate says "proceed to next structural node."
        if outcome.result.action.kind != RoutingKind.CONTINUE:
            raise OrchestrationInvariantError(
                f"Unhandled config gate routing kind {outcome.result.action.kind!r} "
                f"for token {current_token.token_id} at node '{node_id}'. "
                f"Expected CONTINUE when no sink_name, fork, or next_node_id is set."
            )
        return _GateContinue(updated_token=current_token, updated_sink=current_on_success_sink)

    def _handle_gate_fork(
        self,
        outcome: GateOutcome,
        current_token: TokenInfo,
        node_id: NodeID,
        child_items: list[WorkItem],
    ) -> _GateTerminal:
        """Handle fork-to-paths routing: build child work items for each fork branch.

        Iterates child tokens from the gate outcome, resolves coalesce info for each
        branch, and appends continuation or terminal work items to child_items.

        Args:
            outcome: Config gate outcome containing child tokens and routing info.
            current_token: Parent token being forked.
            node_id: Current gate node ID for continuation work items.
            child_items: Mutable list — fork paths append child work items here.

        Returns:
            _GateTerminal with FORKED outcome for the parent token.
        """
        for child_token in outcome.child_tokens:
            # Look up coalesce info for this branch
            cfg_branch_name = child_token.branch_name
            cfg_coalesce_name: CoalesceName | None = None

            if cfg_branch_name and BranchName(cfg_branch_name) in self._branch_to_coalesce:
                cfg_coalesce_name = self._branch_to_coalesce[BranchName(cfg_branch_name)]

            # See config gate fork handler above for routing logic.
            # Children born during a re-drive get fresh token_ids with no prior node_states,
            # so they use the default resume_attempt_offset=0 / resume_checkpoint_id=None.
            if cfg_coalesce_name is None and cfg_branch_name and BranchName(cfg_branch_name) in self._branch_to_sink:
                child_items.append(
                    self._nav.create_work_item(
                        token=child_token,
                        current_node_id=None,
                    )
                )
            else:
                child_items.append(
                    self._nav.create_continuation_work_item(
                        token=child_token,
                        current_node_id=node_id,
                        coalesce_name=cfg_coalesce_name,
                    )
                )

        # NOTE: Parent FORKED outcome is now recorded atomically in fork_token()
        # to eliminate crash window between child creation and outcome recording.
        return _GateTerminal(
            result=RowResult(
                token=current_token,
                final_data=current_token.row_data,
                outcome=TerminalOutcome.TRANSIENT,
                path=TerminalPath.FORK_PARENT,
            )
        )

    def _validate_coalesce_ordering(
        self,
        token: TokenInfo,
        current_node_id: NodeID | None,
        coalesce_node_id: NodeID | None,
        coalesce_name: CoalesceName | None,
    ) -> None:
        """Validate that tokens with coalesce metadata don't start downstream of their coalesce point.

        A malformed work item starting past the coalesce node would silently skip coalesce handling
        because _maybe_coalesce_token only triggers on exact node equality.

        Raises:
            OrchestrationInvariantError: If the token's starting node is downstream of its coalesce barrier.
        """
        if (
            coalesce_node_id is not None
            and current_node_id is not None
            and coalesce_name is not None
            and current_node_id != coalesce_node_id
            and current_node_id in self._node_step_map
            and coalesce_node_id in self._node_step_map
        ):
            current_step = self._node_step_map[current_node_id]
            coalesce_step = self._node_step_map[coalesce_node_id]
            if current_step > coalesce_step:
                raise OrchestrationInvariantError(
                    f"Token {token.token_id} started at node '{current_node_id}' (step {current_step}), "
                    f"which is downstream of coalesce '{coalesce_name}' (step {coalesce_step}). "
                    f"Work items with coalesce metadata must start at or before the coalesce point."
                )

    def _handle_terminal_token(
        self,
        current_token: TokenInfo,
        current_on_success_sink: str,
    ) -> RowResult:
        """Handle a token that has traversed all nodes: resolve final sink, return result.

        Determines the effective sink from:
        1. branch_to_sink mapping (for fork branches routing directly to sinks)
        2. last_on_success_sink (inherited from transforms or source)

        If the token has a branch_name that maps to a direct sink via _branch_to_sink,
        that takes precedence. Otherwise, the accumulated on_success sink is used.

        Raises:
            OrchestrationInvariantError: If no effective sink can be determined (indicates
                a DAG construction or on_success configuration bug).

        Returns:
            RowResult with COMPLETED outcome and resolved sink_name.
        """
        # Determine sink name from explicit routing maps. Fork children
        # targeting direct sinks are resolved via _branch_to_sink (built from
        # DAG COPY edges at construction time). Non-fork tokens use the last
        # transform's on_success or the source's on_success.
        effective_sink = current_on_success_sink
        if current_token.branch_name is not None:
            branch = BranchName(current_token.branch_name)
            if branch in self._branch_to_sink:
                effective_sink = self._branch_to_sink[branch]

        if not effective_sink or not effective_sink.strip():
            raise OrchestrationInvariantError(
                f"No effective sink for token {current_token.token_id}: "
                f"last_on_success_sink={current_on_success_sink!r}, "
                f"branch_name={current_token.branch_name!r}. "
                f"This indicates a DAG construction or on_success configuration bug."
            )

        return RowResult(
            token=current_token,
            final_data=current_token.row_data,
            outcome=TerminalOutcome.SUCCESS,
            path=TerminalPath.DEFAULT_FLOW,
            sink_name=effective_sink,
        )

    def _process_single_token(
        self,
        token: TokenInfo,
        ctx: PluginContext,
        current_node_id: NodeID | None,
        coalesce_node_id: NodeID | None = None,
        coalesce_name: CoalesceName | None = None,
        on_success_sink: str | None = None,
        attempt_offset: int = 0,
    ) -> tuple[RowResult | tuple[RowResult, ...] | None, list[WorkItem]]:
        """Process a single token through processing nodes starting at node_id.

        Args:
            token: Token to process; token.resume_attempt_offset and
                token.resume_checkpoint_id carry the resume state for this token
                and propagate automatically through all node_state writes.
            ctx: Plugin context
            current_node_id: Node ID to start processing from. None is valid only
                for terminal work items that already have explicit sink context
                (inherited on_success_sink or branch_to_sink mapping).
            coalesce_node_id: Node ID at which fork children should coalesce
            coalesce_name: Name of the coalesce point for merging
            on_success_sink: Inherited sink from parent (e.g. terminal deagg parent's on_success)
            attempt_offset: Starting audit attempt offset for lease-recovered work

        Returns:
            Tuple of (RowResult or list of RowResults or None if held for coalesce,
                      list of child WorkItems to queue)
            - Single RowResult for most operations
            - List of RowResults for passthrough aggregation mode
            - None for held coalesce tokens
        """
        current_token = token
        # MUTATION CONTRACT: child_items is passed by reference to _handle_transform_node(),
        # _handle_gate_node(), _notify_coalesce_of_lost_branch(), and _maybe_coalesce_token().
        # These methods append child WorkItems (fork paths, deaggregation, coalesce merges)
        # directly into this list. The caller returns child_items alongside the RowResult.
        # Do NOT replace with return-value-based patterns without updating all call sites.
        child_items: list[WorkItem] = []

        # current_node_id=None skips traversal loop entirely, so only allow it
        # when sink routing is explicit (inherited sink or branch->sink map).
        if current_node_id is None:
            has_branch_sink = current_token.branch_name is not None and BranchName(current_token.branch_name) in self._branch_to_sink
            if on_success_sink is None and not has_branch_sink:
                raise OrchestrationInvariantError(
                    f"Token {token.token_id} has current_node_id=None without explicit terminal sink context. "
                    "Expected inherited on_success_sink or branch_to_sink mapping."
                )

        last_on_success_sink: str = on_success_sink if on_success_sink is not None else self._source_on_success
        if coalesce_name is not None and current_node_id is not None:
            coalesce_node_id_for_name = self._coalesce_node_ids[coalesce_name]
            if coalesce_node_id_for_name == current_node_id and self._nav.resolve_next_node(current_node_id) is None:
                last_on_success_sink = self._nav.resolve_coalesce_sink(
                    coalesce_name,
                    context=f"start of token processing for token '{token.token_id}'",
                )

        self._validate_coalesce_ordering(token, current_node_id, coalesce_node_id, coalesce_name)

        node_id: NodeID | None = current_node_id
        max_inner_iterations = len(self._node_to_next) + 1
        inner_iterations = 0
        while node_id is not None:
            inner_iterations += 1
            if inner_iterations > max_inner_iterations:
                raise OrchestrationInvariantError(
                    f"Inner traversal exceeded {max_inner_iterations} iterations for token "
                    f"{token.token_id}. Possible cycle in node_to_next map."
                )
            # Refresh active scheduler lease (filigree elspeth-ddde8144b6).
            # No-op when no claim is active. Raises SchedulerLeaseLostError
            # when the lease was reaped by a peer — propagates up to
            # ``_drain_scheduler_claims`` which catches it specifically and
            # abandons this iteration cleanly.
            self._heartbeat_active_claim()
            handled, result = self._maybe_coalesce_token(
                current_token,
                current_node_id=node_id,
                coalesce_node_id=coalesce_node_id,
                coalesce_name=coalesce_name,
                child_items=child_items,
            )
            if handled:
                return (result, child_items)

            next_node_id = self._nav.resolve_next_node(node_id)
            plugin = self._nav.resolve_plugin_for_node(node_id)
            if plugin is None:
                # Non-processing structural nodes (e.g. coalesce) are traversed but not executed.
                node_id = next_node_id
                continue

            # Type-safe plugin detection using protocols
            if isinstance(plugin, TransformProtocol):
                row_transform = plugin
                # Check if this is a batch-aware transform at an aggregation node
                transform_node_id = row_transform.node_id
                if row_transform.is_batch_aware and transform_node_id is not None and transform_node_id in self._aggregation_settings:
                    # Use engine buffering for aggregation
                    return self._process_batch_aggregation_node(
                        transform=row_transform,
                        current_token=current_token,
                        ctx=ctx,
                        child_items=child_items,
                        coalesce_node_id=coalesce_node_id,
                        coalesce_name=coalesce_name,
                    )

                # ADR-030 §B (slice 5, follower aggregation barrier hand-off):
                # a follower has no AggregationSettings (trigger evaluation is
                # leader-only per §B.2).  If this batch-aware transform sits at
                # a known aggregation node, the follower must NOT execute it
                # row-wise — doing so produces wrong aggregate output and
                # bypasses the leader's barrier.  Return (None, []) so that
                # _drain_scheduler_claims hits the ``result is None and not
                # child_items`` arm (line 4241) and calls mark_blocked with the
                # aggregation barrier key.  The leader's next journal-intake
                # adopts the arrival and runs trigger evaluation.
                if row_transform.is_batch_aware and transform_node_id is not None and transform_node_id in self._follower_barrier_node_ids:
                    logger.debug(
                        "follower: aggregation barrier hold for token %r at node %r — marking blocked; leader adopts via journal-intake",
                        current_token.token_id,
                        transform_node_id,
                    )
                    return None, child_items

                # NOTE: child_items is mutated inside (deagg appends, coalesce notifications).
                transform_outcome = self._handle_transform_node(
                    row_transform,
                    current_token,
                    ctx,
                    node_id,
                    child_items,
                    coalesce_node_id,
                    coalesce_name,
                    last_on_success_sink,
                    attempt_offset,
                )
                if isinstance(transform_outcome, _TransformTerminal):
                    return transform_outcome.result, child_items
                current_token = transform_outcome.updated_token
                last_on_success_sink = transform_outcome.updated_sink
            elif isinstance(plugin, GateSettings):
                # NOTE: child_items is mutated inside (fork paths, coalesce notifications).
                gate_outcome = self._handle_gate_node(
                    plugin,
                    current_token,
                    ctx,
                    node_id,
                    child_items,
                    coalesce_node_id,
                    coalesce_name,
                    last_on_success_sink,
                )
                if isinstance(gate_outcome, _GateTerminal):
                    return gate_outcome.result, child_items
                current_token = gate_outcome.updated_token
                last_on_success_sink = gate_outcome.updated_sink
                if gate_outcome.next_node_id is not None:
                    node_id = gate_outcome.next_node_id
                    continue

            else:
                raise TypeError(f"Unknown transform type: {type(plugin).__name__}. Expected TransformProtocol or GateSettings.")

            node_id = next_node_id

        result = self._handle_terminal_token(current_token, last_on_success_sink)
        return result, child_items
