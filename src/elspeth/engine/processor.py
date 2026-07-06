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
from dataclasses import dataclass
from datetime import UTC
from types import MappingProxyType
from typing import TYPE_CHECKING, Any, cast

from elspeth.contracts import RouteDestination, RowResult, SourceRow, TokenInfo, TransformResult
from elspeth.contracts.audit import TokenRef
from elspeth.contracts.audit_evidence import AuditEvidenceBase
from elspeth.contracts.barrier_scalars import BarrierScalars, CoalescePendingScalars
from elspeth.contracts.coordination import DEFAULT_ITEM_STALL_BUDGET_SECONDS
from elspeth.contracts.freeze import deep_freeze
from elspeth.contracts.schema_contract import PipelineRow
from elspeth.contracts.types import BranchName, CoalesceName, NodeID, SinkName, StepResolver
from elspeth.engine._best_effort import best_effort
from elspeth.engine._error_hash import compute_error_hash
from elspeth.engine.barrier_coordination import (
    BarrierIntakeCoordinator,
    BarrierJournalRestoreContext,
    BarrierRecoveryCoordinator,
    _LiveBarrierHold,
)
from elspeth.engine.dag_navigator import DAGNavigator
from elspeth.engine.scheduler_drain import (
    MAX_WORK_QUEUE_ITERATIONS as MAX_WORK_QUEUE_ITERATIONS,  # re-export: tests import from here
)
from elspeth.engine.scheduler_drain import (
    SCHEDULER_MAINTENANCE_INTERVAL as SCHEDULER_MAINTENANCE_INTERVAL,  # re-export: tests import from here
)
from elspeth.engine.scheduler_drain import (
    ProcessorMode,
    SchedulerDrainCoordinator,
    is_scheduler_sink_bound_result,
    require_scheduler_outcome,
    require_scheduler_sink_name,
    scheduler_error_hash,
    scheduler_error_message,
    with_scheduler_pending_sink_handoffs,
)
from elspeth.engine.scheduler_work_codec import SchedulerWorkCodec
from elspeth.engine.token_traversal import TokenTraversalEngine
from elspeth.engine.token_traversal import (
    _GateContinue as _GateContinue,  # re-export: callers import the union types from here
)
from elspeth.engine.token_traversal import (
    _GateOutcome as _GateOutcome,  # re-export
)
from elspeth.engine.token_traversal import (
    _GateTerminal as _GateTerminal,  # re-export
)
from elspeth.engine.token_traversal import (
    _TransformContinue as _TransformContinue,  # re-export: tests import from here
)
from elspeth.engine.token_traversal import (
    _TransformOutcome as _TransformOutcome,  # re-export
)
from elspeth.engine.token_traversal import (
    _TransformTerminal as _TransformTerminal,  # re-export
)
from elspeth.engine.work_items import WorkItem, WorkItemFactory

if TYPE_CHECKING:
    from sqlalchemy.engine import Connection

    from elspeth.contracts.audit import Row as AuditRow
    from elspeth.contracts.audit import Token as AuditToken
    from elspeth.contracts.coordination import CoordinationToken
    from elspeth.contracts.events import TelemetryEvent
    from elspeth.contracts.payload_store import PayloadStore
    from elspeth.engine.clock import Clock
    from elspeth.engine.coalesce_executor import CoalesceExecutor
    from elspeth.engine.executors import GateOutcome
    from elspeth.engine.orchestrator.plugin_types import RowPlugin
    from elspeth.engine.orchestrator.ports import TelemetryManagerProtocol

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
    OrchestrationInvariantError,
    PassThroughContractViolation,
    PluginContractViolation,
    PluginRetryableError,
    TransformErrorCategory,
    TransformErrorReason,
)
from elspeth.contracts.plugin_context import PluginContext
from elspeth.contracts.results import FailureInfo
from elspeth.contracts.scheduler import (
    BarrierEmission,
    BranchLossSpec,
    TokenWorkItem,
    TokenWorkStatus,
)
from elspeth.contracts.secret_scrub import scrub_text_for_audit
from elspeth.core.checkpoint.recovery import IncompleteTokenSpec
from elspeth.core.config import AggregationSettings, GateSettings
from elspeth.core.landscape._helpers import generate_id
from elspeth.core.landscape.data_flow_repository import DataFlowRepository
from elspeth.core.landscape.errors import LandscapeRecordError
from elspeth.core.landscape.execution_repository import ExecutionRepository
from elspeth.core.landscape.run_coordination_repository import RunCoordinationRepository
from elspeth.core.landscape.scheduler_repository import (
    TokenSchedulerRepository,
)
from elspeth.engine.clock import DEFAULT_CLOCK
from elspeth.engine.executors import (
    AggregationExecutor,
    GateExecutor,
    TransformExecutor,
)
from elspeth.engine.executors.declaration_dispatch import run_batch_flush_checks, run_boundary_checks
from elspeth.engine.executors.state_guard import stamped_node_state_id
from elspeth.engine.executors.transform import record_transform_error_with_routing
from elspeth.engine.retry import RetryManager
from elspeth.engine.spans import SpanFactory
from elspeth.engine.tokens import TokenManager

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
        mode: ProcessorMode = ProcessorMode.LEADER,
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
            mode: Explicit processor role (elspeth-577179bba1). LEADER (the
                default) covers the fenced leader, the N=1 arm, and direct
                repository-level construction — within it, behavior still
                keys on genuine coordination_token/run_coordination presence.
                FOLLOWER is validated fail-closed below: it requires
                ``coordination_token=None``, ``run_coordination=None``, and
                an explicit ``scheduler_lease_owner``; it gates the public
                :meth:`drain_follower_ready_work` surface and the follower
                skip of scheduler maintenance (ADR-030 §C.3).
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
        self._nav = DAGNavigator.from_traversal_context(
            traversal,
            coalesce_on_success_map=self._coalesce_on_success_map,
            sink_names=self._sink_names,
        )
        self._work_items = WorkItemFactory(self._nav)

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
        # One codec for the WorkItem <-> durable scheduler payload mapping
        # (elspeth-6291c51766): ingest, enqueue, READY barrier emission, and
        # rehydrate must derive byte-identical field bundles (deterministic
        # work_item_id + strict field-equality reconciliation), so the
        # derivation lives in one place instead of four hand-synced encoders.
        self._work_codec = SchedulerWorkCodec(
            serialize_row_payload=scheduler.serialize_row_payload,
            deserialize_row_payload=scheduler.deserialize_row_payload,
            resolve_node_cursor=self._scheduler_node_id,
            resolve_step_index=self._scheduler_step_index,
            resolve_ingest_sequence=data_flow.resolve_row_ingest_sequence,
            queue_key_for_item=self._queue_key_for_blocked_item,
            barrier_key_for_item=self._barrier_key_for_blocked_item,
            create_work_item=self._work_items.create,
        )
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
        # Explicit processor role (elspeth-577179bba1): follower-ness is a
        # STORED construction-time decision, never re-derived from the
        # coordination_token/run_coordination None-sentinels. On the follower
        # path those sentinels remain correct ABSENCES (no epoch fence, no
        # §C.2 housekeeping) — the mode flag, not their None-ness, drives
        # branch selection. FOLLOWER wiring is validated fail-closed HERE so
        # a wrong-mode construction crashes instead of silently taking a
        # leader-only arm. (The follower barrier-intake no-op needs no mode
        # gate: it is already structural — empty aggregation_settings plus
        # coalesce_executor=None make the intake pass a no-op.)
        self._mode = mode
        if mode is ProcessorMode.FOLLOWER:
            if coordination_token is not None:
                raise OrchestrationInvariantError(
                    "ProcessorMode.FOLLOWER forbids a coordination_token: a follower must never "
                    "present an epoch fence (ADR-030 §B.1 — the fenced verbs are leader-only). "
                    "A follower carrying a leader fence is the wrong-mode bug this flag exists to catch."
                )
            if run_coordination is not None:
                raise OrchestrationInvariantError(
                    "ProcessorMode.FOLLOWER forbids run_coordination: a follower must never run "
                    "the §C.2 housekeeping/eviction sweep (leader-only). A follower holding the "
                    "coordination repository is the wrong-mode bug this flag exists to catch."
                )
            if not self._scheduler_lease_owner_registered:
                raise OrchestrationInvariantError(
                    "ProcessorMode.FOLLOWER requires an explicit scheduler_lease_owner: the "
                    "follower's registered run_workers identity IS its lease owner and "
                    "membership-fence key (ADR-030 §A.1/§G); an anonymous minted owner cannot "
                    "pass the claim fence."
                )
        self._scheduler_lease_seconds = scheduler_lease_seconds
        if scheduler_heartbeat_seconds <= 0:
            raise OrchestrationInvariantError(f"scheduler_heartbeat_seconds must be positive, got {scheduler_heartbeat_seconds}")
        if scheduler_heartbeat_seconds >= scheduler_lease_seconds:
            raise OrchestrationInvariantError(
                f"scheduler_heartbeat_seconds ({scheduler_heartbeat_seconds}) must be less than "
                f"scheduler_lease_seconds ({scheduler_lease_seconds}); otherwise the heartbeat "
                "cannot refresh the lease before it expires under any slow-work scenario."
            )

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
        # Barrier subsystem (elspeth-e76a186916): the intake and recovery
        # coordinators own the crash-window adoption/restore ordering (open
        # batch -> fenced adopt -> feed memory -> evaluate trigger) behind one
        # boundary; the processor keeps thin delegates plus the continuation
        # seams (flush execution, coalesce fire completion, telemetry).
        self._barrier_intake = BarrierIntakeCoordinator(
            run_id=run_id,
            scheduler=scheduler,
            data_flow=data_flow,
            execution=execution,
            aggregation_executor=self._aggregation_executor,
            coalesce_executor=self._coalesce_executor,
            nav=self._nav,
            work_items=self._work_items,
            clock=self._clock,
            aggregation_settings=self._aggregation_settings,
            coalesce_node_ids=self._coalesce_node_ids,
            coordination_token=coordination_token,
            scheduler_lease_owner=self._scheduler_lease_owner,
            live_barrier_holds=self._live_barrier_holds,
            resume_checkpoint_id=self._resume_checkpoint_id,
            flush_batch=self.handle_timeout_flush,
            complete_coalesce_fire=self._complete_coalesce_fire,
            terminal_coalesce_row_result=self._terminal_coalesce_row_result,
            emit_token_completed=self._emit_token_completed,
            mark_coalesce_consumed_terminal=self._mark_coalesce_consumed_scheduler_work_terminal,
        )
        # Scheduler-drain subsystem (elspeth-c49f33d6e4 component 3): the
        # coordinator owns the durable claim/drain loop, dispositions,
        # pending-sink recovery, the active-claim heartbeat, and the
        # maintenance cadence. It shares the live-holds dict and the
        # branch-loss stage BY REFERENCE with the barrier subsystem and the
        # processor-side producers, and drives token processing through this
        # processor at call time (the traversal engine is component 4 of the
        # same split).
        self._scheduler_drain = SchedulerDrainCoordinator(
            processor=self,
            mode=mode,
            run_id=run_id,
            scheduler=scheduler,
            work_codec=self._work_codec,
            execution=execution,
            clock=self._clock,
            run_coordination=run_coordination,
            coordination_token=coordination_token,
            scheduler_lease_owner=self._scheduler_lease_owner,
            scheduler_lease_seconds=scheduler_lease_seconds,
            scheduler_heartbeat_seconds=scheduler_heartbeat_seconds,
            scheduler_lease_owner_registered=self._scheduler_lease_owner_registered,
            resume_checkpoint_id=self._resume_checkpoint_id,
            live_barrier_holds=self._live_barrier_holds,
            pending_branch_losses=self._pending_branch_losses,
        )
        # Component 4 (c49): the DAG token-traversal state machine. Holds only a
        # back-reference and resolves processor seams at call time, so tests that
        # patch _process_single_token / _handle_transform_node on the processor
        # still intercept traversal.
        self._token_traversal = TokenTraversalEngine(self)
        if barrier_restore is not None:
            BarrierRecoveryCoordinator(
                run_id=run_id,
                scheduler=scheduler,
                data_flow=data_flow,
                execution=execution,
                aggregation_executor=self._aggregation_executor,
                coalesce_executor=self._coalesce_executor,
                clock=self._clock,
                aggregation_settings=self._aggregation_settings,
                coalesce_node_ids=self._coalesce_node_ids,
                coordination_token=coordination_token,
                scheduler_lease_owner=self._scheduler_lease_owner,
            ).restore_from_journal(barrier_restore)

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
                            outputs=BatchFlushOutputs(
                                emitted_rows=(emitted_row,),
                                used_success_empty=used_success_empty,
                            ),
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
                        outputs=BatchFlushOutputs(
                            emitted_rows=(),
                            used_success_empty=used_success_empty,
                        ),
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
                    outputs=BatchFlushOutputs(
                        emitted_rows=tuple(emitted),
                        used_success_empty=used_success_empty,
                    ),
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
                    self._work_items.create_continuation(
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
                        self._work_items.create_continuation(
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
        state_id: str | None,
        retryable: bool = True,
    ) -> tuple[TransformResult, TokenInfo, str | None]:
        """Convert a retryable exception to a TransformResult.error when no retry manager is configured.

        Shared handler for PluginRetryableError (retryable) and transient exceptions
        (ConnectionError, TimeoutError, OSError, CapacityError). Records the
        error in the audit trail and emits a DIVERT routing event if on_error
        routes to a sink.

        ``state_id`` is the failed attempt's node-state id, carried out of the
        executor on the exception via NodeStateGuard's stamp (ctx.state_id is
        scope-restored during unwind and must not be read here). None is only
        possible when no attempt ever opened a node state (e.g. shutdown
        before the first retry attempt); record_transform_error_with_routing
        rejects None for named sinks.
        """
        on_error = transform.on_error
        # on_error is always set (required by TransformSettings) — Tier 1 invariant
        if on_error is None:
            raise OrchestrationInvariantError(
                f"Transform '{transform.name}' has on_error=None — this should be impossible since TransformSettings requires on_error"
            )

        error_details: TransformErrorReason = {"reason": reason, "error": scrub_text_for_audit(str(exc))}
        # Shared error-audit routine (elspeth-aeb0a8f756): identical
        # transform_error + DIVERT routing_event recording as the executor's
        # error-result branch. Here the guard already auto-failed the state on
        # exception exit, so recording happens after completion by design —
        # the helper is sequencing-agnostic.
        record_transform_error_with_routing(
            ctx=ctx,
            execution=self._execution,
            error_edge_ids=self._error_edge_ids,
            state_id=state_id,
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
                    state_id=stamped_node_state_id(e),
                    retryable=False,
                )
            except PluginRetryableError as e:
                return self._convert_retryable_to_error_result(
                    e,
                    transform,
                    token,
                    ctx,
                    reason="transient_error_no_retry" if e.retryable else "permanent_error",
                    state_id=stamped_node_state_id(e),
                )
            except (ConnectionError, TimeoutError, OSError, CapacityError) as e:
                return self._convert_retryable_to_error_result(
                    e,
                    transform,
                    token,
                    ctx,
                    reason="transient_error_no_retry",
                    state_id=stamped_node_state_id(e),
                )

        # Track attempt number for audit; track the last failed attempt's
        # node-state id so a shutdown InterruptedError raised INSIDE the
        # RetryManager (pre-attempt check or backoff wait — never crosses a
        # NodeStateGuard, so it carries no stamp) can still attribute its
        # DIVERT routing_event to the state that actually failed.
        attempt_tracker = {"current": attempt_offset}
        state_tracker: dict[str, str | None] = {"last_failed_state_id": None}

        def execute_attempt() -> tuple[TransformResult, TokenInfo, str | None]:
            attempt = attempt_tracker["current"]
            attempt_tracker["current"] += 1
            try:
                return self._transform_executor.execute_transform(
                    transform=transform,
                    token=token,
                    ctx=ctx,
                    attempt=attempt,
                )
            except BaseException as e:
                stamped = stamped_node_state_id(e)
                if stamped is not None:
                    state_tracker["last_failed_state_id"] = stamped
                raise

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
                # Stamped when the shutdown fired inside execute_transform
                # (batch waiter); otherwise the RetryManager raised it and the
                # last failed attempt's state is the divert attribution point.
                state_id=stamped_node_state_id(e) or state_tracker["last_failed_state_id"],
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
        return self._work_items.create(
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
        ONE epoch-fenced IMMEDIATE transaction. Scheduler fields come from
        the shared work codec (deterministic work_item_id + strict field
        equality reconciliation downstream); ``ingest_sequence`` is passed
        explicitly because the row is inserted in this same transaction and
        is not yet resolvable.
        """
        coordination_token = self._coordination_token
        if coordination_token is None:
            raise OrchestrationInvariantError(
                "Fenced source ingest requires a coordination token; the unfenced arm must not reach this helper."
            )
        token = item.token
        now = self._clock.now_utc()
        fields = self._work_codec.ready_fields(item, ingest_sequence=ingest_sequence)

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
            token_id=fields.token_id,
            row_id=fields.row_id,
            node_id=fields.node_id,
            step_index=fields.step_index,
            ingest_sequence=fields.ingest_sequence,
            row_payload_json=fields.row_payload_json,
            lease_owner=self._scheduler_lease_owner,
            lease_seconds=self._scheduler_lease_seconds,
            queue_key=fields.queue_key,
            barrier_key=fields.barrier_key,
            on_success_sink=fields.on_success_sink,
            branch_name=fields.branch_name,
            fork_group_id=fields.fork_group_id,
            join_group_id=fields.join_group_id,
            expand_group_id=fields.expand_group_id,
            coalesce_node_id=fields.coalesce_node_id,
            coalesce_name=fields.coalesce_name,
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
            self._work_items.create(
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
            merged_item = self._work_items.create(
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
        return self._scheduler_drain.run_maintenance(self._clock.now_utc())

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
            sink_name=require_scheduler_sink_name(result),
            outcome=require_scheduler_outcome(result).value,
            path=result.path.value,
            error_hash=scheduler_error_hash(result),
            error_message=scheduler_error_message(result),
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
            if not is_scheduler_sink_bound_result(result):
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
            with_scheduler_pending_sink_handoffs(results, pending_sink_token_ids),
        )
        return tagged_results, pending_sink_token_ids

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
            # READY continuation fields come from the shared work codec: the
            # drain loop (or ``_drain_work_queue``'s initial claim) later runs
            # the idempotent enqueue for the same WorkItem, which reconciles
            # against the row inserted here by deterministic ``work_item_id``
            # and strict field equality.
            emitted_ready=() if merged_item is None else (self._work_codec.ready_emission(merged_item),),
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
        merged_item = self._work_items.create(
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

    def _run_barrier_intake_pass(self, ctx: PluginContext) -> tuple[list[RowResult], list[WorkItem]]:
        """One §E.2 intake pass, delegated to the barrier subsystem.

        The BarrierIntakeCoordinator owns the adoption/replay/trigger
        ordering (elspeth-e76a186916) and returns typed dispositions;
        flattening them preserves the historical (results, child_items)
        contract for the drain loop and the orchestrator's EOF loop.
        """
        outcome = self._barrier_intake.run_intake_pass(ctx)
        return outcome.results, outcome.child_items

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

    def drain_follower_ready_work(
        self,
        ctx: PluginContext,
        *,
        before_claim: Callable[[], None] | None = None,
    ) -> list[RowResult]:
        """Follower drain surface: claim and advance READY work only (ADR-030 §B.1/§C.3).

        The explicit follower entry point driven by orchestrator/follower.py's
        drain loop. The follower-mode contract is owned HERE, not by caller
        flag wiring: ``claim_ready`` only — never ``claim_pending_sink`` or
        pending-sink recovery (sink work is leader-only) — and no pre-claimed
        items. ``before_claim`` is the follower's leader-liveness probe,
        invoked before every claim attempt.

        Fails closed on mode: driving a LEADER-mode processor through the
        follower surface is exactly the wrong-mode bug this named contract
        exists to catch (elspeth-577179bba1).
        """
        if self._mode is not ProcessorMode.FOLLOWER:
            raise OrchestrationInvariantError(
                f"drain_follower_ready_work called on a {self._mode.value!r}-mode RowProcessor: "
                "the follower drain surface requires ProcessorMode.FOLLOWER. Leader/N=1 paths "
                "drain via process_row / drain_scheduled_work."
            )
        return self._scheduler_drain.drain_claims(
            ctx=ctx,
            pending_items={},
            recover_pending_sinks=False,
            before_claim=before_claim,
        )

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

        Thin delegate to the scheduler-drain subsystem (elspeth-c49f33d6e4):
        ``SchedulerDrainCoordinator.drain_claims`` owns the claim loop, the
        per-iteration journal-first barrier intake ordering, the disposition
        arms, and pending-sink recovery. The keyword-only signature here is
        load-bearing — the ADR-030 invariant-guard tests call this private
        name. (orchestrator/follower.py now enters via the public
        :meth:`drain_follower_ready_work` surface instead.)
        """
        return self._scheduler_drain.drain_claims(
            ctx=ctx,
            pending_items=pending_items,
            recover_pending_sinks=recover_pending_sinks,
            preclaimed_items=preclaimed_items,
            before_claim=before_claim,
        )

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
        """Rebuild a sink-bound row result without re-running its producer node (delegate)."""
        return self._scheduler_drain.row_result_from_pending_sink(scheduled)

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
        """Take the staged §E.5 loss record for the claim being disposed (delegate)."""
        return self._scheduler_drain.take_claim_branch_loss(claimed_token_id)

    def _heartbeat_active_claim(self) -> None:
        """Refresh the active scheduler lease if the heartbeat interval elapsed.

        Delegate: ``SchedulerDrainCoordinator.heartbeat_active_claim`` owns the
        active-claim state and the at-most-once-per-interval write.
        ``_process_single_token`` calls this on every node-iteration boundary
        (ADR-026 RC6, filigree elspeth-ddde8144b6); it raises
        ``SchedulerLeaseLostError`` when the lease was reaped by a peer.
        """
        self._scheduler_drain.heartbeat_active_claim()

    def _barrier_key_for_live_hold(self, token_id: str) -> str:
        """Resolve the barrier owning a token about to be marked BLOCKED (delegate)."""
        return self._scheduler_drain.barrier_key_for_live_hold(token_id)

    def _enqueue_scheduler_work_item(
        self,
        item: WorkItem,
        pending_items: dict[str, WorkItem],
        *,
        claim_immediately: bool = False,
    ) -> TokenWorkItem:
        """Persist a READY scheduler item and retain the live token payload (delegate)."""
        return self._scheduler_drain.enqueue_work_item(item, pending_items, claim_immediately=claim_immediately)

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

    # --- TokenTraversalEngine delegates (c49 component 4) ---
    # The traversal state machine lives in engine/token_traversal.py. These three
    # names are the cluster's external surface and stay on RowProcessor: the
    # SchedulerDrainHost seam calls self._processor._process_single_token at call
    # time (10+ tests patch it), and test_processor.py drives _handle_transform_node
    # / _handle_transform_error_status directly. Every other handler in the family
    # is engine-internal.

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
        return self._token_traversal.process_single_token(
            token,
            ctx,
            current_node_id,
            coalesce_node_id,
            coalesce_name,
            on_success_sink,
            attempt_offset,
        )

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
        return self._token_traversal.handle_transform_node(
            transform,
            current_token,
            ctx,
            node_id,
            child_items,
            coalesce_node_id,
            coalesce_name,
            current_on_success_sink,
            attempt_offset,
        )

    def _handle_transform_error_status(
        self,
        transform_result: TransformResult,
        current_token: TokenInfo,
        error_sink: str | None,
        child_items: list[WorkItem],
    ) -> _TransformTerminal:
        return self._token_traversal.handle_transform_error_status(
            transform_result,
            current_token,
            error_sink,
            child_items,
        )
