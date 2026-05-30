"""RowProcessor: Orchestrates row processing through pipeline.

Coordinates:
- Token creation
- Transform execution
- Gate evaluation (config-driven)
- Aggregation handling
- Final outcome recording
"""

from __future__ import annotations

import hashlib
import logging
from collections import deque
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from types import MappingProxyType
from typing import TYPE_CHECKING, Any, cast

from elspeth.contracts import RouteDestination, RowResult, SourceRow, TokenInfo, TransformResult
from elspeth.contracts.audit import TokenRef
from elspeth.contracts.audit_evidence import AuditEvidenceBase
from elspeth.contracts.freeze import deep_freeze
from elspeth.contracts.schema_contract import PipelineRow
from elspeth.contracts.types import BranchName, CoalesceName, NodeID, SinkName, StepResolver
from elspeth.engine._best_effort import best_effort
from elspeth.engine.dag_navigator import DAGNavigator, WorkItem

if TYPE_CHECKING:
    from elspeth.contracts.aggregation_checkpoint import AggregationCheckpointState
    from elspeth.contracts.coalesce_checkpoint import CoalesceCheckpointState
    from elspeth.contracts.events import TelemetryEvent
    from elspeth.contracts.payload_store import PayloadStore
    from elspeth.engine.clock import Clock
    from elspeth.engine.coalesce_executor import CoalesceExecutor
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
    TransformErrorCategory,
    TransformErrorReason,
)
from elspeth.contracts.plugin_context import PluginContext
from elspeth.contracts.results import FailureInfo
from elspeth.core.checkpoint.recovery import IncompleteTokenSpec
from elspeth.core.config import AggregationSettings, GateSettings
from elspeth.core.landscape.data_flow_repository import DataFlowRepository
from elspeth.core.landscape.errors import LandscapeRecordError
from elspeth.core.landscape.execution_repository import ExecutionRepository
from elspeth.engine.clock import DEFAULT_CLOCK
from elspeth.engine.executors import (
    AggregationExecutor,
    GateExecutor,
    TransformExecutor,
)
from elspeth.engine.executors.can_drop_rows import verify_zero_emission_declaration_path
from elspeth.engine.executors.declaration_dispatch import run_batch_flush_checks, run_boundary_checks
from elspeth.engine.retry import RetryManager
from elspeth.engine.spans import SpanFactory
from elspeth.engine.tokens import TokenManager

# Iteration guard to prevent infinite loops from bugs.
# This counts dequeued work items, so it must exceed the largest supported
# single-row fan-out; otherwise legal expansion trips the safety valve before
# the final child runs.
MAX_WORK_QUEUE_ITERATIONS = 100_000
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
    first_transform_node_id: NodeID | None
    node_to_next: Mapping[NodeID, NodeID | None]
    coalesce_node_map: Mapping[CoalesceName, NodeID]
    branch_first_node: Mapping[str, NodeID] = MappingProxyType({})

    def __post_init__(self) -> None:
        object.__setattr__(self, "node_step_map", deep_freeze(self.node_step_map))
        object.__setattr__(self, "node_to_plugin", deep_freeze(self.node_to_plugin))
        object.__setattr__(self, "node_to_next", deep_freeze(self.node_to_next))
        object.__setattr__(self, "coalesce_node_map", deep_freeze(self.coalesce_node_map))
        object.__setattr__(self, "branch_first_node", deep_freeze(self.branch_first_node))


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
            source_row=SourceRow.valid({"value": 42}, contract=contract),
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
        restored_aggregation_state: Mapping[NodeID, AggregationCheckpointState] | None = None,
        payload_store: PayloadStore | None = None,
        clock: Clock | None = None,
        max_workers: int | None = None,
        telemetry_manager: TelemetryManagerProtocol | None = None,
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
            restored_aggregation_state: Map of node_id -> state dict for crash recovery
            payload_store: Optional PayloadStore for persisting source row payloads
            clock: Optional clock for time access. Defaults to system clock.
                   Inject MockClock for deterministic testing.
            max_workers: Maximum concurrent workers for transform execution (None = no limit)
            telemetry_manager: Optional TelemetryManager for emitting telemetry events.
                               If None, telemetry emission is disabled.
        """
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
        self._first_transform_node_id: NodeID | None = traversal.first_transform_node_id
        self._node_to_next: Mapping[NodeID, NodeID | None] = traversal.node_to_next
        self._retry_manager = retry_manager
        self._coalesce_executor = coalesce_executor
        self._coalesce_node_ids: dict[CoalesceName, NodeID] = dict(traversal.coalesce_node_map)
        self._coalesce_name_by_node_id: dict[NodeID, CoalesceName] = {
            node_id: coalesce_name for coalesce_name, node_id in self._coalesce_node_ids.items()
        }
        self._structural_node_ids: frozenset[NodeID] = frozenset(nid for nid in self._node_to_next if nid not in self._node_to_plugin)
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

        # Restore aggregation state if provided (crash recovery / resume).
        # Multiple node_id keys may map to the same state — deduplicate by
        # content equality (not id()) to handle both shared references and
        # independently deserialized copies.
        if restored_aggregation_state:
            unique_states: list[AggregationCheckpointState] = []
            for state in restored_aggregation_state.values():
                if state not in unique_states:
                    unique_states.append(state)
            for state in unique_states:
                self._aggregation_executor.restore_from_checkpoint(state)

    @property
    def token_manager(self) -> TokenManager:
        """Expose token manager for orchestrator to create tokens for quarantined rows."""
        return self._token_manager

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

        from datetime import UTC, datetime

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

        from datetime import UTC, datetime

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

        from datetime import UTC, datetime

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

    def get_aggregation_checkpoint_state(self) -> AggregationCheckpointState:
        """Get checkpoint state for all aggregation buffers.

        Returns complete state of all aggregation nodes (buffers + triggers)
        for persistence during checkpointing. This enables crash recovery
        without losing buffered rows.

        Returns:
            Typed checkpoint state suitable for passing to create_checkpoint().
        """
        return self._aggregation_executor.get_checkpoint_state()

    def get_coalesce_checkpoint_state(self) -> CoalesceCheckpointState | None:
        """Get checkpoint state for pending coalesces."""
        if self._coalesce_executor is None:
            return None
        return self._coalesce_executor.get_checkpoint_state()

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
        error_hash = hashlib.sha256(fctx.error_msg.encode()).hexdigest()[:16]
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
        error_hash = hashlib.sha256(violation_summary.encode()).hexdigest()[:16]
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
                    error_hash = hashlib.sha256(f"quarantined_in_batch:{fctx.batch_id}:{i}".encode()).hexdigest()[:16]
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
            return self._handle_flush_error(fctx), []

        # ADR-009 §Clause 2: runtime cross-check for passes_through_input
        # transforms on the batch-aware flush path. MUST run BEFORE
        # _emit_transform_completed so a failed cross-check does not follow
        # a COMPLETED terminal-state emission on any token.
        self._cross_check_flush_output(fctx, result)

        # Emit TransformCompleted telemetry for all buffered tokens
        for token in buffered_tokens:
            self._emit_transform_completed(token=token, transform=transform, transform_result=result)

        if settings.output_mode == OutputMode.PASSTHROUGH:
            return self._route_passthrough_results(fctx, result)
        if settings.output_mode == OutputMode.TRANSFORM:
            return self._route_transform_results(fctx, result)
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

        settings = self._aggregation_settings[node_id]
        output_mode = settings.output_mode

        # Buffer the row
        self._aggregation_executor.buffer_row(node_id, current_token)

        # Record BUFFERED for this token BEFORE checking flush.
        # On count-threshold flush, the triggering token would otherwise have
        # no BUFFERED record — it goes directly to CONSUMED_IN_BATCH/FAILED.
        # Recording here ensures BUFFERED → terminal for every aggregation token.
        buf_batch_id = self._aggregation_executor.get_batch_id(node_id)
        if buf_batch_id is None:
            raise OrchestrationInvariantError(f"batch_id is None after buffer_row() for node {node_id}")
        self._data_flow.record_token_outcome(
            ref=TokenRef(token_id=current_token.token_id, run_id=self._run_id),
            outcome=None,
            path=TerminalPath.BUFFERED,
            batch_id=buf_batch_id,
        )

        # Check if we should flush
        if self._aggregation_executor.should_flush(node_id):
            trigger_type = self._aggregation_executor.get_trigger_type(node_id)
            if trigger_type is None:
                trigger_type = TriggerType.COUNT

            result, buffered_tokens, batch_id = self._aggregation_executor.execute_flush(
                node_id=node_id,
                transform=cast(BatchTransformProtocol, transform),
                ctx=ctx,
                trigger_type=trigger_type,
            )

            fctx = _FlushContext(
                node_id=node_id,
                transform=transform,
                settings=settings,
                buffered_tokens=tuple(buffered_tokens),
                batch_id=batch_id,
                error_msg="Batch transform failed",
                expand_parent_token=current_token,
                triggering_token=current_token,
                coalesce_node_id=coalesce_node_id,
                coalesce_name=coalesce_name,
            )

            if result.status != "success":
                return self._handle_flush_error(fctx), child_items

            # ADR-009 §Clause 2: runtime cross-check for passes_through_input
            # transforms on the batch-aware flush path. MUST run BEFORE
            # _emit_transform_completed so a failed cross-check does not
            # follow a COMPLETED terminal-state emission on any token.
            self._cross_check_flush_output(fctx, result)

            # Emit TransformCompleted telemetry for all buffered tokens
            for token in buffered_tokens:
                self._emit_transform_completed(token=token, transform=transform, transform_result=result)

            if output_mode == OutputMode.PASSTHROUGH:
                flush_results, flush_child_items = self._route_passthrough_results(fctx, result)
                child_items.extend(flush_child_items)
                return flush_results, child_items
            if output_mode == OutputMode.TRANSFORM:
                flush_results, flush_child_items = self._route_transform_results(fctx, result)
                child_items.extend(flush_child_items)
                return flush_results, child_items
            raise OrchestrationInvariantError(f"Unknown output_mode: {output_mode}")

        # Not flushing yet — BUFFERED already recorded above.
        # Terminal outcome is deferred to flush time for both modes:
        # - passthrough: BUFFERED → COMPLETED/FAILED at flush
        # - transform: BUFFERED → CONSUMED_IN_BATCH/QUARANTINED/FAILED at flush
        # NOTE: Do NOT emit TokenCompleted telemetry here!
        # TokenCompleted must be deferred to flush time so that
        # TransformCompleted can be emitted first.
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
        ctx.record_transform_error(
            token_id=token.token_id,
            transform_id=transform.node_id,
            row=token.row_data,
            error_details=error_details,
            destination=on_error,
        )

        # Record DIVERT routing_event using ctx.state_id (set by
        # TransformExecutor.execute_transform before the exception propagated).
        if on_error != "discard":
            try:
                error_edge_id = self._error_edge_ids[NodeID(transform.node_id)]
            except KeyError as key_err:
                raise OrchestrationInvariantError(
                    f"Transform '{transform.node_id}' has on_error={on_error!r} but no DIVERT edge registered."
                ) from key_err
            if ctx.state_id is None:
                raise OrchestrationInvariantError(
                    f"ctx.state_id must be set by TransformExecutor before exception propagated (transform={transform.node_id})"
                )
            self._execution.record_routing_event(
                state_id=ctx.state_id,
                edge_id=error_edge_id,
                mode=RoutingMode.DIVERT,
                reason=error_details,
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
                    attempt=0,
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
        attempt_tracker = {"current": 0}

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
        error: ExecutionError | None = None,
    ) -> None:
        """Record the source node state for a token.

        Source "processing" already happened in the plugin iterator, so the
        state is recorded immediately as COMPLETED or FAILED with duration 0.
        """
        source_state = self._execution.begin_node_state(
            token_id=token.token_id,
            node_id=self._source_node_id,
            run_id=self._run_id,
            step_index=0,
            input_data=input_data,
        )
        if status == NodeStateStatus.COMPLETED:
            self._execution.complete_node_state(
                state_id=source_state.state_id,
                status=NodeStateStatus.COMPLETED,
                output_data=input_data,
                duration_ms=0,
            )
            return
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
        error_hash = hashlib.sha256(f"{type(failure).__name__}:{self._source_node_id}".encode()).hexdigest()[:16]
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
                f"on source boundary (node={self._source_node_id!r}). Audit trail is INCOMPLETE — "
                f"the FAILED token outcome may be missing. Recorder failure: "
                f"{type(record_failure).__name__}: {record_failure}. Original {original_label}: {failure!s}"
            ) from record_failure
        try:
            self._record_source_node_state(
                token=token,
                input_data=input_data,
                status=NodeStateStatus.FAILED,
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
                f"on source boundary (node={self._source_node_id!r}). Audit trail is INCOMPLETE — "
                f"the FAILED source node state may be missing. Recorder failure: "
                f"{type(record_failure).__name__}: {record_failure}. Original {original_label}: {failure!s}"
            ) from record_failure
        with best_effort(
            "TokenCompleted telemetry after source-boundary FAILED audit",
            run_id=self._run_id,
            token_id=token.token_id,
            source_node_id=self._source_node_id,
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
        coalesce_node_id: NodeID | None,
        coalesce_name: CoalesceName | None,
    ) -> list[RowResult]:
        """Record source node_state and start pipeline traversal.

        Shared implementation for process_row and process_existing_row.
        Records the source node as immediately COMPLETED (duration_ms=0)
        since source "processing" already happened in the plugin iterator.

        Args:
            token: Token for the row being processed
            input_data: Row data dict for audit hashing (must be plain dict)
            transforms: List of transform plugins (for invariant check)
            ctx: Plugin context
            coalesce_node_id: Node ID at which fork children should coalesce
            coalesce_name: Name of the coalesce point for merging

        Returns:
            List of RowResults, one per terminal token
        """
        self._record_source_node_state(
            token=token,
            input_data=input_data,
            status=NodeStateStatus.COMPLETED,
        )

        if transforms and self._first_transform_node_id is None:
            raise OrchestrationInvariantError("Traversal context is missing first_transform_node_id for non-empty transform pipeline")
        initial_node_id = self._first_transform_node_id if self._first_transform_node_id is not None else self._source_node_id
        return self._drain_work_queue(
            self._nav.create_work_item(
                token=token,
                current_node_id=initial_node_id,
                coalesce_node_id=coalesce_node_id,
                coalesce_name=coalesce_name,
            ),
            ctx,
        )

    def process_row(
        self,
        row_index: int,
        source_row: SourceRow,
        transforms: Sequence[Any],
        ctx: PluginContext,
        *,
        coalesce_node_id: NodeID | None = None,
        coalesce_name: CoalesceName | None = None,
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

        Returns:
            List of RowResults, one per terminal token (parent + children)
        """
        # Create initial token from SourceRow
        # TokenManager.create_initial_token() expects SourceRow and converts to PipelineRow
        token = self._token_manager.create_initial_token(
            run_id=self._run_id,
            source_node_id=self._source_node_id,
            row_index=row_index,
            source_row=source_row,
        )

        # Valid SourceRows always carry mapping-shaped row payloads; once the
        # row enters the processor we treat the values as opaque objects.
        source_input = cast(dict[str, object], source_row.row)
        if self._source_plugin is not None:
            try:
                run_boundary_checks(
                    inputs=BoundaryInputs(
                        plugin=self._source_plugin,
                        node_id=str(self._source_node_id),
                        run_id=self._run_id,
                        row_id=token.row_id,
                        token_id=token.token_id,
                        static_contract=self._source_plugin.declared_guaranteed_fields,
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
                self._record_source_boundary_failure(
                    token=token,
                    input_data=source_input,
                    failure=failure,
                )
                raise
        return self._record_source_and_start_traversal(
            token=token,
            input_data=source_input,
            transforms=transforms,
            ctx=ctx,
            coalesce_node_id=coalesce_node_id,
            coalesce_name=coalesce_name,
        )

    def process_existing_row(
        self,
        row_id: str,
        row_data: PipelineRow,
        transforms: Sequence[Any],
        ctx: PluginContext,
        *,
        coalesce_node_id: NodeID | None = None,
        coalesce_name: CoalesceName | None = None,
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

        Dispatch cases (branch_name checked first to avoid confusing a fork→coalesce
        before-barrier token with a post-coalesce merged token that also has join_group_id):

        1. fork → sink terminal branch: branch_name in _branch_to_sink → current_node_id=None
           (process_token's None-path routes via branch_to_sink to the terminal sink).
        2. fork → coalesce, crashed before barrier: branch_name in _branch_to_coalesce →
           re-run the branch from its first processing node with coalesce context.
        3. expand child: expand_group_id set → re-drive from the node AFTER the expand node.
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

        if spec.expand_group_id is not None:
            # expand child: re-drive from the node AFTER the expand node.
            # expand is never terminal; an `after` of None here is an audit/DAG inconsistency
            # that process_token's None-enforcement raises on (no branch_to_sink / on_success_sink).
            after = self._nav.resolve_next_node(self._resolve_step_node(spec))
            return self.process_token(token, ctx, current_node_id=after)

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
        if (
            self._coalesce_executor is None
            or current_token.branch_name is None
            or coalesce_name is None
            or coalesce_node_id is None
            or current_node_id != coalesce_node_id
        ):
            return False, None

        coalesce_outcome = self._coalesce_executor.accept(
            token=current_token,
            coalesce_name=coalesce_name,
        )

        if coalesce_outcome.held:
            return True, None

        if coalesce_outcome.merged_token is not None:
            if self._nav.resolve_next_node(coalesce_node_id) is None:
                if coalesce_name is None:
                    raise OrchestrationInvariantError("Terminal coalesce outcome missing coalesce_name")
                return (
                    True,
                    self._terminal_coalesce_row_result(
                        coalesce_outcome.merged_token,
                        coalesce_name,
                        context=f"terminal coalesce outcome for token '{coalesce_outcome.merged_token.token_id}'",
                    ),
                )

            coalesce_node_id = self._coalesce_node_ids[coalesce_name]
            child_items.append(
                self._nav.create_work_item(
                    token=coalesce_outcome.merged_token,
                    current_node_id=coalesce_node_id,
                )
            )
            return True, None

        if coalesce_outcome.failure_reason:
            error_msg = coalesce_outcome.failure_reason
            error_hash = hashlib.sha256(error_msg.encode()).hexdigest()[:16]

            # Bug 9z8 fix: Only record if CoalesceExecutor didn't already record
            if not coalesce_outcome.outcomes_recorded:
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

            return (
                True,
                RowResult(
                    token=current_token,
                    final_data=current_token.row_data,
                    outcome=TerminalOutcome.FAILURE,
                    path=TerminalPath.UNROUTED,
                    error=FailureInfo(
                        exception_type="CoalesceFailure",
                        message=error_msg,
                    ),
                ),
            )

        raise OrchestrationInvariantError(
            f"CoalesceOutcome for token {current_token.token_id} in coalesce '{coalesce_name}' "
            f"is in invalid state: held={coalesce_outcome.held}, "
            f"merged_token={coalesce_outcome.merged_token is not None}, "
            f"failure_reason={coalesce_outcome.failure_reason!r}"
        )

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
        if self._coalesce_executor is None or current_token.branch_name is None:
            return []

        branch_name = BranchName(current_token.branch_name)
        if branch_name not in self._branch_to_coalesce:
            return []
        coalesce_name = self._branch_to_coalesce[branch_name]

        coalesce_node_id = self._coalesce_node_ids[coalesce_name]
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
            # Non-terminal — resume merged token at coalesce step
            child_items.append(
                self._nav.create_work_item(
                    token=outcome.merged_token,
                    current_node_id=coalesce_node_id,
                )
            )
            return []

        if outcome.failure_reason:
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
    ) -> list[RowResult]:
        """Drain the work queue, processing tokens until empty.

        Implements breadth-first DAG traversal. Each _process_single_token call
        may produce child work items (from forks, expansions, etc.) which are
        appended to the queue.
        """
        work_queue: deque[WorkItem] = deque([initial_item])
        results: list[RowResult] = []
        iterations = 0

        with self._spans.row_span(initial_item.token.row_id, initial_item.token.token_id):
            while work_queue:
                iterations += 1
                if iterations > MAX_WORK_QUEUE_ITERATIONS:
                    raise RuntimeError(f"Work queue exceeded {MAX_WORK_QUEUE_ITERATIONS} iterations. Possible infinite loop in pipeline.")

                item = work_queue.popleft()
                result, child_items = self._process_single_token(
                    token=item.token,
                    ctx=ctx,
                    current_node_id=item.current_node_id,
                    coalesce_node_id=item.coalesce_node_id,
                    coalesce_name=item.coalesce_name,
                    on_success_sink=item.on_success_sink,
                )

                if result is not None:
                    if isinstance(result, tuple):
                        results.extend(result)
                    else:
                        results.append(result)

                work_queue.extend(child_items)

        return results

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
            error_hash = hashlib.sha256(str(e).encode()).hexdigest()[:16]
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
            quarantine_error_hash = hashlib.sha256(error_detail.encode()).hexdigest()[:16]
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
