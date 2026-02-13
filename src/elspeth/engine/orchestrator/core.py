# src/elspeth/engine/orchestrator/core.py
"""Core Orchestrator class for pipeline execution.

Coordinates:
- Run initialization
- Source loading
- Row processing
- Sink writing
- Run completion
- Post-run audit export (when configured)

The Orchestrator is the main entry point for running ELSPETH pipelines.
It delegates to focused helper modules for:
- Validation: Route and sink validation (validation.py)
- Export: Landscape export functionality (export.py)
- Aggregation: Timeout and flush handling (aggregation.py)
"""

from __future__ import annotations

import hashlib
import json
import signal
import threading
import time
from collections.abc import Callable, Iterator
from contextlib import contextmanager, nullcontext
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from elspeth.contracts.events import TelemetryEvent
    from elspeth.contracts.payload_store import PayloadStore
    from elspeth.core.events import EventBusProtocol
    from elspeth.telemetry import TelemetryManager

from elspeth import __version__ as ENGINE_VERSION
from elspeth.contracts import (
    BatchPendingError,
    ExportStatus,
    NodeType,
    PendingOutcome,
    PipelineRow,
    RouteDestination,
    RowOutcome,
    RunStatus,
    SchemaContract,
    TokenInfo,
)
from elspeth.contracts.cli import ProgressEvent
from elspeth.contracts.config import RuntimeRetryConfig
from elspeth.contracts.enums import NodeStateStatus, RoutingMode
from elspeth.contracts.errors import (
    ExecutionError,
    GracefulShutdownError,
    OrchestrationInvariantError,
    SourceQuarantineReason,
)
from elspeth.contracts.events import (
    PhaseAction,
    PhaseCompleted,
    PhaseError,
    PhaseStarted,
    PipelinePhase,
    RunCompletionStatus,
    RunSummary,
)
from elspeth.contracts.plugin_context import PluginContext
from elspeth.contracts.types import (
    AggregationName,
    BranchName,
    CoalesceName,
    GateName,
    NodeID,
    SinkName,
)
from elspeth.core.canonical import repr_hash, sanitize_for_canonical, stable_hash
from elspeth.core.config import AggregationSettings
from elspeth.core.dag import ExecutionGraph
from elspeth.core.landscape import LandscapeDB, LandscapeRecorder
from elspeth.core.operations import track_operation

# Import module functions from orchestrator submodules
from elspeth.engine.orchestrator.aggregation import (
    check_aggregation_timeouts,
    flush_remaining_aggregation_buffers,
    handle_incomplete_batches,
)
from elspeth.engine.orchestrator.export import (
    export_landscape,
    reconstruct_schema_from_json,
)
from elspeth.engine.orchestrator.outcomes import (
    accumulate_row_outcomes,
    flush_coalesce_pending,
    handle_coalesce_timeouts,
)
from elspeth.engine.orchestrator.types import (
    ExecutionCounters,
    PipelineConfig,
    RouteValidationError,
    RowPlugin,
    RunResult,
)
from elspeth.engine.orchestrator.validation import (
    validate_route_destinations,
    validate_source_quarantine_destination,
    validate_transform_error_sinks,
)
from elspeth.engine.processor import DAGTraversalContext, RowProcessor, make_step_resolver
from elspeth.engine.retry import RetryManager
from elspeth.engine.spans import SpanFactory
from elspeth.plugins.protocols import SinkProtocol, SourceProtocol, TransformProtocol

if TYPE_CHECKING:
    from elspeth.contracts import ResumePoint
    from elspeth.contracts.config.runtime import RuntimeCheckpointConfig, RuntimeConcurrencyConfig
    from elspeth.core.checkpoint import CheckpointManager
    from elspeth.core.config import ElspethSettings, GateSettings
    from elspeth.core.rate_limit import RateLimitRegistry
    from elspeth.engine.clock import Clock
    from elspeth.engine.coalesce_executor import CoalesceExecutor


class Orchestrator:
    """Orchestrates full pipeline runs.

    Manages the complete lifecycle:
    1. Begin run in Landscape
    2. Register all nodes (and set node_id on each plugin instance)
    3. Load rows from source
    4. Process rows through transforms
    5. Write to sinks
    6. Complete run

    The Orchestrator sets node_id on each plugin instance AFTER registering
    it with Landscape. This is part of the plugin protocol contract - all
    plugins define node_id: str | None and the orchestrator populates it.
    """

    def __init__(
        self,
        db: LandscapeDB,
        *,
        event_bus: EventBusProtocol | None = None,
        canonical_version: str = "sha256-rfc8785-v1",
        checkpoint_manager: CheckpointManager | None = None,
        checkpoint_config: RuntimeCheckpointConfig | None = None,
        clock: Clock | None = None,
        rate_limit_registry: RateLimitRegistry | None = None,
        concurrency_config: RuntimeConcurrencyConfig | None = None,
        telemetry_manager: TelemetryManager | None = None,
        coalesce_completed_keys_limit: int = 10000,
    ) -> None:
        from elspeth.core.events import NullEventBus
        from elspeth.engine.clock import DEFAULT_CLOCK

        self._db = db
        self._events = event_bus if event_bus is not None else NullEventBus()
        self._canonical_version = canonical_version
        self._span_factory = SpanFactory()
        self._checkpoint_manager = checkpoint_manager
        self._checkpoint_config = checkpoint_config
        self._clock = clock if clock is not None else DEFAULT_CLOCK
        self._rate_limit_registry = rate_limit_registry
        self._concurrency_config = concurrency_config
        self._coalesce_completed_keys_limit = coalesce_completed_keys_limit
        self._sequence_number = 0  # Monotonic counter for checkpoint ordering
        self._current_graph: ExecutionGraph | None = None  # Set during execution for checkpointing
        self._telemetry = telemetry_manager  # Optional, disabled by default

    def _emit_telemetry(self, event: TelemetryEvent) -> None:
        """Emit telemetry event if manager is configured.

        Telemetry is emitted AFTER Landscape recording succeeds. Landscape is
        the legal record; telemetry is operational visibility.

        Args:
            event: The telemetry event to emit
        """
        if self._telemetry is not None:
            self._telemetry.handle_event(event)

    def _flush_telemetry(self) -> None:
        """Flush telemetry events if manager is configured.

        Ensures queued telemetry is exported before returning control to caller.
        """
        if self._telemetry is not None:
            self._telemetry.flush()

    def _maybe_checkpoint(
        self,
        run_id: str,
        token_id: str,
        node_id: str,
        aggregation_state: dict[str, Any] | None = None,
    ) -> None:
        """Create checkpoint if configured.

        Called after a token has been durably written to its terminal sink.
        The checkpoint represents a durable progress marker - recovery can
        safely skip any row whose token has a checkpoint with a sink node_id.

        IMPORTANT: Checkpoints are created AFTER sink writes, not during
        the main processing loop. This ensures the checkpoint represents
        actual durable output, not just processing completion.

        Args:
            run_id: Current run ID
            token_id: Token that was just written to sink
            node_id: Sink node that received the token
            aggregation_state: Current aggregation buffer/trigger state for crash recovery
        """
        if not self._checkpoint_config or not self._checkpoint_config.enabled:
            return
        if self._checkpoint_manager is None:
            return
        if self._current_graph is None:
            # Should never happen - graph is set during execution
            raise RuntimeError("Cannot create checkpoint: execution graph not available")

        self._sequence_number += 1

        # RuntimeCheckpointConfig.frequency is an int:
        # - 1 = every_row
        # - 0 = aggregation_only
        # - N = every N rows
        frequency = self._checkpoint_config.frequency
        should_checkpoint = False
        if frequency == 1:
            should_checkpoint = True  # every_row
        elif frequency > 1:
            should_checkpoint = (self._sequence_number % frequency) == 0  # every_n
        # frequency == 0: aggregation_only - checkpointed separately in aggregation flush

        if should_checkpoint:
            self._checkpoint_manager.create_checkpoint(
                run_id=run_id,
                token_id=token_id,
                node_id=node_id,
                sequence_number=self._sequence_number,
                graph=self._current_graph,
                aggregation_state=aggregation_state,
            )

    def _delete_checkpoints(self, run_id: str) -> None:
        """Delete all checkpoints for a run after successful completion.

        Args:
            run_id: Run to clean up checkpoints for
        """
        if self._checkpoint_manager is not None:
            self._checkpoint_manager.delete_checkpoints(run_id)

    def _write_pending_to_sinks(
        self,
        recorder: LandscapeRecorder,
        run_id: str,
        config: PipelineConfig,
        ctx: PluginContext,
        pending_tokens: dict[str, list[tuple[TokenInfo, PendingOutcome | None]]],
        sink_id_map: dict[SinkName, NodeID],
        sink_step: int,
        *,
        on_token_written_factory: Callable[[str], Callable[[TokenInfo], None]] | None = None,
    ) -> None:
        """Write pending tokens to sinks using SinkExecutor.

        Extracted from _execute_run() and _process_resumed_rows() to eliminate
        duplication of the sink write orchestration pattern.

        Args:
            recorder: LandscapeRecorder for audit trail
            run_id: Current run ID
            config: Pipeline configuration
            ctx: Plugin context
            pending_tokens: Dict of sink_name -> list of (token, pending_outcome) pairs
            sink_id_map: Maps SinkName -> NodeID for checkpoint callbacks
            sink_step: Audit step index for sink writes (from processor.resolve_sink_step())
            on_token_written_factory: Optional factory that creates per-sink checkpoint
                callbacks. Takes sink_node_id, returns callback(TokenInfo) -> None.
                When None (resume path), no checkpoint callbacks are used.
        """
        from itertools import groupby

        from elspeth.engine.executors import SinkExecutor

        sink_executor = SinkExecutor(recorder, self._span_factory, run_id)
        step = sink_step

        for sink_name, token_outcome_pairs in pending_tokens.items():
            if token_outcome_pairs and sink_name in config.sinks:
                sink = config.sinks[sink_name]
                sink_node_id = sink_id_map[SinkName(sink_name)]

                # Group tokens by pending_outcome for separate write() calls
                # (sink_executor.write() takes a single PendingOutcome for all tokens in a batch)
                # Fix: P1-2026-01-31 - PendingOutcome carries error_hash for QUARANTINED
                def pending_sort_key(pair: tuple[TokenInfo, PendingOutcome | None]) -> tuple[bool, str, str]:
                    pending = pair[1]
                    if pending is None:
                        return (True, "", "")  # None sorts first
                    return (False, pending.outcome.value, pending.error_hash or "")

                sorted_pairs = sorted(token_outcome_pairs, key=pending_sort_key)

                # Build on_token_written callback (or None for resume)
                on_token_written: Callable[[TokenInfo], None] | None = None
                if on_token_written_factory is not None:
                    on_token_written = on_token_written_factory(sink_node_id)

                for pending_outcome, group in groupby(sorted_pairs, key=lambda x: x[1]):
                    group_tokens = [token for token, _ in group]
                    sink_executor.write(
                        sink=sink,
                        tokens=group_tokens,
                        ctx=ctx,
                        step_in_pipeline=step,
                        sink_name=sink_name,
                        pending_outcome=pending_outcome,
                        on_token_written=on_token_written,
                    )

    def _cleanup_plugins(
        self,
        config: PipelineConfig,
        ctx: PluginContext,
        *,
        include_source: bool = True,
    ) -> None:
        """Clean up all plugins in the finally block.

        Implements the lifecycle teardown contract:
        1. on_complete(ctx) on all plugins (transforms, sinks, optionally source)
        2. close() on all plugins (source, transforms, sinks)

        on_complete() is called even on pipeline error -- it signals "processing
        is done" (success or failure), not "processing succeeded". close() is
        pure resource teardown and always follows on_complete().

        Each call is individually try/excepted so one plugin's failure does not
        prevent other plugins from cleaning up. All errors are collected and
        raised together after all cleanup completes.

        Extracted from _execute_run() and _process_resumed_rows() to eliminate
        duplication of the finally-block cleanup pattern.

        Args:
            config: Pipeline configuration
            ctx: Plugin context
            include_source: If True (default), calls on_complete() and close()
                on the source. Set to False for resume path where source wasn't opened.

        Raises:
            RuntimeError: If any plugin cleanup hook fails. Chained from the
                pending exception if one exists.
        """
        import sys

        import structlog

        logger = structlog.get_logger()
        pending_exc = sys.exc_info()[1]
        cleanup_errors: list[str] = []

        def record_cleanup_error(hook: str, plugin_name: str, error: Exception) -> None:
            logger.warning(
                "Plugin cleanup hook failed",
                hook=hook,
                plugin=plugin_name,
                error=str(error),
                error_type=type(error).__name__,
            )
            cleanup_errors.append(f"{hook}({plugin_name}): {type(error).__name__}: {error}")

        # Call on_complete for all plugins (even on error)
        # Base classes provide no-op implementations, so no hasattr needed
        for transform in config.transforms:
            try:
                transform.on_complete(ctx)
            except Exception as e:
                record_cleanup_error("transform.on_complete", transform.name, e)
        for sink in config.sinks.values():
            try:
                sink.on_complete(ctx)
            except Exception as e:
                record_cleanup_error("sink.on_complete", sink.name, e)
        if include_source:
            try:
                config.source.on_complete(ctx)
            except Exception as e:
                record_cleanup_error("source.on_complete", config.source.name, e)

        # Close source (if included) and all sinks
        if include_source:
            try:
                config.source.close()
            except Exception as e:
                record_cleanup_error("source.close", config.source.name, e)

        # Close all transforms (release resources - file handles, connections, etc.)
        for transform in config.transforms:
            try:
                transform.close()
            except Exception as e:
                record_cleanup_error("transform.close", transform.name, e)

        # Close all sinks
        for sink in config.sinks.values():
            try:
                sink.close()
            except Exception as e:
                record_cleanup_error("sink.close", sink.name, e)

        if cleanup_errors:
            error_summary = "; ".join(cleanup_errors)
            if pending_exc is not None:
                raise RuntimeError(f"Plugin cleanup failed: {error_summary}") from pending_exc
            raise RuntimeError(f"Plugin cleanup failed: {error_summary}")

    def _assign_plugin_node_ids(
        self,
        source: SourceProtocol,
        transforms: list[RowPlugin],
        sinks: dict[str, SinkProtocol],
        source_id: NodeID,
        transform_id_map: dict[int, NodeID],
        sink_id_map: dict[SinkName, NodeID],
    ) -> None:
        """Explicitly assign node_id to all plugins with validation.

        This is part of the plugin protocol contract - all plugins define
        node_id: str | None and the orchestrator populates it.

        Args:
            source: Source plugin instance
            transforms: List of transform plugins
            sinks: Dict of sink_name -> sink plugin
            source_id: Node ID for source
            transform_id_map: Maps transform sequence -> node_id
            sink_id_map: Maps sink_name -> node_id

        Raises:
            ValueError: If transform/sink not in ID map
        """
        # Set node_id on source
        source.node_id = source_id

        # Set node_id on transforms
        # Note: Aggregation transforms already have node_id set by CLI (mapped from
        # aggregation_id_map), so only assign for transforms without node_id.
        for seq, transform in enumerate(transforms):
            if transform.node_id is not None:
                # Already has node_id (e.g., aggregation transform) - skip
                continue
            if seq not in transform_id_map:
                raise ValueError(
                    f"Transform at sequence {seq} not found in graph. Graph has mappings for sequences: {list(transform_id_map.keys())}"
                )
            transform.node_id = transform_id_map[seq]

        # Set node_id on sinks
        # Note: Sinks not in graph are skipped (e.g., export sinks used post-run)
        for sink_name, sink in sinks.items():
            sink_name_typed = SinkName(sink_name)
            if sink_name_typed not in sink_id_map:
                # Sink not in execution graph - skip silently
                # This happens for post-run sinks (e.g., landscape.export.sink)
                continue
            sink.node_id = sink_id_map[sink_name_typed]

    def _build_dag_traversal_context(
        self,
        graph: ExecutionGraph,
        config: PipelineConfig,
        config_gate_id_map: dict[GateName, NodeID],
    ) -> DAGTraversalContext:
        """Build traversal context for RowProcessor from graph + pipeline config."""
        node_step_map = graph.build_step_map()
        node_to_plugin: dict[NodeID, RowPlugin | GateSettings] = {}

        for transform in config.transforms:
            node_id_raw = transform.node_id
            if node_id_raw is None:
                raise OrchestrationInvariantError(f"Transform '{transform.name}' missing node_id for traversal context")
            node_to_plugin[NodeID(node_id_raw)] = transform

        for gate in config.gates:
            gate_node_id = config_gate_id_map[GateName(gate.name)]
            node_to_plugin[gate_node_id] = gate

        node_to_next: dict[NodeID, NodeID | None] = {}
        source_id = graph.get_source()
        if source_id is not None:
            node_to_next[source_id] = graph.get_next_node(source_id)
        for node_id in graph.get_pipeline_node_sequence():
            node_to_next[node_id] = graph.get_next_node(node_id)
        for coalesce_node_id in graph.get_coalesce_id_map().values():
            node_to_next[coalesce_node_id] = graph.get_next_node(coalesce_node_id)

        return DAGTraversalContext(
            node_step_map=node_step_map,
            node_to_plugin=node_to_plugin,
            first_transform_node_id=graph.get_first_transform_node(),
            node_to_next=node_to_next,
            coalesce_node_map=graph.get_coalesce_id_map(),
            branch_first_node=graph.get_branch_first_nodes(),
        )

    def _build_processor(
        self,
        *,
        graph: ExecutionGraph,
        config: PipelineConfig,
        settings: ElspethSettings | None,
        recorder: LandscapeRecorder,
        run_id: str,
        source_id: NodeID,
        edge_map: dict[tuple[NodeID, str], str],
        route_resolution_map: dict[tuple[NodeID, str], RouteDestination] | None,
        config_gate_id_map: dict[GateName, NodeID],
        coalesce_id_map: dict[CoalesceName, NodeID],
        payload_store: PayloadStore,
        restored_aggregation_state: dict[NodeID, dict[str, Any]] | None = None,
    ) -> tuple[RowProcessor, dict[CoalesceName, NodeID], CoalesceExecutor | None]:
        """Build a RowProcessor with all supporting infrastructure.

        Constructs the retry manager, coalesce executor, traversal context,
        and coalesce routing maps, then assembles a RowProcessor. Used by
        both the main run path and the resume path.

        Returns:
            Tuple of (processor, coalesce_node_map, coalesce_executor).
        """
        from elspeth.engine.coalesce_executor import CoalesceExecutor
        from elspeth.engine.tokens import TokenManager

        retry_manager: RetryManager | None = None
        if settings is not None:
            retry_manager = RetryManager(RuntimeRetryConfig.from_settings(settings.retry))

        # Derive coalesce routing from graph topology unconditionally.
        # If the graph has coalesce nodes, the processor needs branch_to_coalesce
        # regardless of whether settings is available.
        branch_to_coalesce: dict[BranchName, CoalesceName] = graph.get_branch_to_coalesce_map()
        coalesce_node_map: dict[CoalesceName, NodeID] = graph.get_coalesce_id_map()

        # Build traversal context BEFORE CoalesceExecutor/TokenManager so that
        # node_step_map is available for the step_resolver closure they require.
        traversal = self._build_dag_traversal_context(graph, config, config_gate_id_map)

        # Build step_resolver from shared factory (single source of truth).
        # Same factory is used by RowProcessor internally for its executors.
        step_resolver = make_step_resolver(traversal.node_step_map, source_id)

        coalesce_executor: CoalesceExecutor | None = None

        if coalesce_node_map:
            # Graph has coalesce nodes — settings.coalesce is required for
            # CoalesceExecutor registration (merge policy, timeout, etc.)
            if settings is None or not settings.coalesce:
                raise OrchestrationInvariantError(
                    "Graph contains coalesce nodes but settings.coalesce is missing. "
                    "Coalesce settings are required when the pipeline has fork/join patterns."
                )

            # payload_store intentionally omitted: CoalesceExecutor's TokenManager only
            # calls coalesce_tokens(), which does not persist payloads (payloads are
            # recorded by the RowProcessor's TokenManager during initial token creation).
            token_manager = TokenManager(recorder, step_resolver=step_resolver)
            coalesce_executor = CoalesceExecutor(
                recorder=recorder,
                span_factory=self._span_factory,
                token_manager=token_manager,
                run_id=run_id,
                step_resolver=step_resolver,
                clock=self._clock,
                max_completed_keys=self._coalesce_completed_keys_limit,
            )

            for coalesce_settings_entry in settings.coalesce:
                coalesce_node_id = coalesce_id_map[CoalesceName(coalesce_settings_entry.name)]
                coalesce_executor.register_coalesce(coalesce_settings_entry, coalesce_node_id)

        # Derive coalesce on_success from graph's terminal sink map (graph-authoritative),
        # falling back to settings for non-terminal coalesce nodes.
        terminal_sink_map = graph.get_terminal_sink_map()
        coalesce_on_success_map: dict[CoalesceName, str] = {}
        for cname, cnode_id in coalesce_node_map.items():
            if cnode_id in terminal_sink_map:
                coalesce_on_success_map[cname] = terminal_sink_map[cnode_id]
            elif settings is not None and settings.coalesce:
                for coalesce_settings_entry in settings.coalesce:
                    if CoalesceName(coalesce_settings_entry.name) == cname and coalesce_settings_entry.on_success is not None:
                        coalesce_on_success_map[cname] = coalesce_settings_entry.on_success

        branch_to_sink = graph.get_branch_to_sink_map()
        typed_aggregation_settings: dict[NodeID, AggregationSettings] = {NodeID(k): v for k, v in config.aggregation_settings.items()}

        processor = RowProcessor(
            recorder=recorder,
            span_factory=self._span_factory,
            run_id=run_id,
            source_node_id=source_id,
            source_on_success=config.source.on_success,
            edge_map=edge_map,
            route_resolution_map=route_resolution_map,
            traversal=traversal,
            aggregation_settings=typed_aggregation_settings,
            retry_manager=retry_manager,
            coalesce_executor=coalesce_executor,
            branch_to_coalesce=branch_to_coalesce,
            branch_to_sink=branch_to_sink,
            sink_names=frozenset(config.sinks),
            coalesce_on_success_map=coalesce_on_success_map,
            restored_aggregation_state=restored_aggregation_state,
            payload_store=payload_store,
            clock=self._clock,
            max_workers=self._concurrency_config.max_workers if self._concurrency_config else None,
            telemetry_manager=self._telemetry,
        )

        return processor, coalesce_node_map, coalesce_executor

    @contextmanager
    def _shutdown_handler_context(self) -> Iterator[threading.Event]:
        """Install SIGINT/SIGTERM handlers that set a shutdown event.

        On first signal: sets the event, restores default SIGINT handler
        (so second Ctrl-C force-kills via KeyboardInterrupt).

        When called from a non-main thread (e.g., programmatic/embedded usage),
        signal registration is skipped — Python raises ValueError if
        signal.signal() is called outside the main thread.  The returned
        Event still works; it just won't be triggered by OS signals.

        Yields the Event for the processing loop to check.
        Restores original handlers in finally block (main thread only).
        """
        shutdown_event = threading.Event()

        # signal.signal() can only be called from the main thread.
        # In embedded/programmatic usage the orchestrator may run on a
        # worker thread — fall back to a plain event without handlers.
        if threading.current_thread() is not threading.main_thread():
            yield shutdown_event
            return

        original_sigint = signal.getsignal(signal.SIGINT)
        original_sigterm = signal.getsignal(signal.SIGTERM)

        def _handler(signum: int, frame: Any) -> None:
            shutdown_event.set()
            # Restore default SIGINT so second Ctrl-C force-kills
            signal.signal(signal.SIGINT, signal.default_int_handler)

        signal.signal(signal.SIGINT, _handler)
        signal.signal(signal.SIGTERM, _handler)
        try:
            yield shutdown_event
        finally:
            signal.signal(signal.SIGINT, original_sigint)
            signal.signal(signal.SIGTERM, original_sigterm)

    def run(
        self,
        config: PipelineConfig,
        graph: ExecutionGraph | None = None,
        settings: ElspethSettings | None = None,
        batch_checkpoints: dict[str, dict[str, Any]] | None = None,
        *,
        payload_store: PayloadStore,
        secret_resolutions: list[dict[str, Any]] | None = None,
        shutdown_event: threading.Event | None = None,
    ) -> RunResult:
        """Execute a pipeline run.

        Args:
            config: Pipeline configuration with plugins
            graph: Pre-validated execution graph (required)
            settings: Full settings (for post-run hooks like export)
            batch_checkpoints: Batch transform checkpoints to restore (from
                previous BatchPendingError). Maps node_id -> checkpoint_data.
                Used when retrying a run after a batch transform raised
                BatchPendingError.
            payload_store: PayloadStore for persisting source row payloads.
                Required for audit compliance (CLAUDE.md: "Source entry - Raw data
                stored before any processing").
            secret_resolutions: Optional list of secret resolution records from
                load_secrets_from_config(). When provided, these are recorded
                in the audit trail after run creation. Each record contains
                env_var_name, source, vault_url, secret_name, timestamp, latency_ms,
                and secret_value (for fingerprinting, never stored).
            shutdown_event: Optional pre-created shutdown event for testing.
                When provided, signal handler installation is skipped and this
                event is passed directly to _execute_run(). Production callers
                should omit this (signal handlers are installed automatically).

        Raises:
            ValueError: If graph or payload_store is not provided
        """
        if graph is None:
            raise ValueError("ExecutionGraph is required. Build with ExecutionGraph.from_plugin_instances()")
        if payload_store is None:
            raise ValueError("PayloadStore is required for audit compliance.")

        # Schema validation now happens in ExecutionGraph.validate() during graph construction

        # Local imports for telemetry events - consolidated here to avoid repeated imports
        from elspeth.telemetry import (
            PhaseChanged,
            RunFinished,
            RunStarted,
        )

        # DATABASE phase - create recorder and begin run
        phase_start = time.perf_counter()
        try:
            self._events.emit(PhaseStarted(phase=PipelinePhase.DATABASE, action=PhaseAction.CONNECTING))

            # Serialize source schema for resume type restoration
            # This enables proper type coercion (datetime/Decimal) when resuming from JSON payloads
            # SourceProtocol requires output_schema - all sources have schemas (even dynamic ones)
            source_schema_json = json.dumps(config.source.output_schema.model_json_schema())

            # Get source schema contract for resume PipelineRow wrapping
            # This enables proper contract propagation when resuming from stored payloads
            source_contract = config.source.get_schema_contract()

            recorder = LandscapeRecorder(self._db, payload_store=payload_store)
            run = recorder.begin_run(
                config=config.config,
                canonical_version=self._canonical_version,
                source_schema_json=source_schema_json,
                schema_contract=source_contract,
            )

            # Record secret resolutions in audit trail (deferred from pre-run loading)
            # Resolutions already contain pre-computed fingerprints (no plaintext values)
            if secret_resolutions:
                recorder.record_secret_resolutions(
                    run_id=run.run_id,
                    resolutions=secret_resolutions,
                )

            # Emit telemetry AFTER Landscape succeeds - Landscape is the legal record
            self._emit_telemetry(
                RunStarted(
                    timestamp=datetime.now(UTC),
                    run_id=run.run_id,
                    config_hash=run.config_hash,
                    source_plugin=config.source.name,
                )
            )

            self._events.emit(PhaseCompleted(phase=PipelinePhase.DATABASE, duration_seconds=time.perf_counter() - phase_start))
        except Exception as e:
            self._events.emit(PhaseError(phase=PipelinePhase.DATABASE, error=e))
            raise  # CRITICAL: Always re-raise - database connection failure is fatal

        run_completed = False
        run_start_time = time.perf_counter()
        try:
            # When shutdown_event is provided (testing), skip signal handler
            # installation and use the caller's event directly.
            shutdown_ctx = nullcontext(shutdown_event) if shutdown_event is not None else self._shutdown_handler_context()
            with self._span_factory.run_span(run.run_id), shutdown_ctx as active_event:
                result = self._execute_run(
                    recorder,
                    run.run_id,
                    config,
                    graph,
                    settings,
                    batch_checkpoints,
                    payload_store=payload_store,
                    shutdown_event=active_event,
                )

            # Complete run with reproducibility grade computation
            recorder.finalize_run(run.run_id, status=RunStatus.COMPLETED)
            result.status = RunStatus.COMPLETED
            run_completed = True

            # Emit telemetry AFTER Landscape finalize succeeds
            run_duration_ms = (time.perf_counter() - run_start_time) * 1000
            self._emit_telemetry(
                RunFinished(
                    timestamp=datetime.now(UTC),
                    run_id=run.run_id,
                    status=RunStatus.COMPLETED,
                    row_count=result.rows_processed,
                    duration_ms=run_duration_ms,
                )
            )

            # Delete checkpoints on successful completion
            # (checkpoints are for recovery, not needed after success)
            self._delete_checkpoints(run.run_id)

            # EXPORT phase - post-run landscape export (if enabled)
            if settings is not None and settings.landscape.export.enabled:
                export_config = settings.landscape.export
                recorder.set_export_status(
                    run.run_id,
                    status=ExportStatus.PENDING,
                    export_format=export_config.format,
                    export_sink=export_config.sink,
                )

                phase_start = time.perf_counter()
                try:
                    self._events.emit(PhaseStarted(phase=PipelinePhase.EXPORT, action=PhaseAction.EXPORTING, target=export_config.sink))

                    # Emit telemetry PhaseChanged for EXPORT
                    self._emit_telemetry(
                        PhaseChanged(
                            timestamp=datetime.now(UTC),
                            run_id=run.run_id,
                            phase=PipelinePhase.EXPORT,
                            action=PhaseAction.EXPORTING,
                        )
                    )

                    # Call module function directly (no wrapper method)
                    export_landscape(self._db, run.run_id, settings, config.sinks)

                    recorder.set_export_status(run.run_id, status=ExportStatus.COMPLETED)
                    self._events.emit(PhaseCompleted(phase=PipelinePhase.EXPORT, duration_seconds=time.perf_counter() - phase_start))
                except Exception as export_error:
                    self._events.emit(PhaseError(phase=PipelinePhase.EXPORT, error=export_error, target=export_config.sink))
                    recorder.set_export_status(
                        run.run_id,
                        status=ExportStatus.FAILED,
                        error=str(export_error),
                    )
                    # Re-raise so caller knows export failed
                    # (run is still "completed" in Landscape)
                    raise

            # Emit RunSummary event with final metrics
            total_duration = time.perf_counter() - run_start_time
            self._events.emit(
                RunSummary(
                    run_id=run.run_id,
                    status=RunCompletionStatus.COMPLETED,
                    total_rows=result.rows_processed,
                    succeeded=result.rows_succeeded,
                    failed=result.rows_failed,
                    quarantined=result.rows_quarantined,
                    duration_seconds=total_duration,
                    exit_code=0,
                    routed=result.rows_routed,
                    routed_destinations=tuple(result.routed_destinations.items()),
                )
            )

            return result

        except BatchPendingError:
            # BatchPendingError is a CONTROL-FLOW SIGNAL, not an error.
            # A batch transform has submitted work that isn't complete yet.
            # DO NOT mark run as failed - it's pending, not failed.
            # DO NOT emit RunSummary - run isn't done yet.
            # Re-raise for caller to schedule retry based on check_after_seconds.
            raise
        except GracefulShutdownError as shutdown_exc:
            # Graceful shutdown: all in-flight work flushed, checkpoints created.
            # Mark run INTERRUPTED (resumable via `elspeth resume`).
            total_duration = time.perf_counter() - run_start_time
            recorder.finalize_run(run.run_id, status=RunStatus.INTERRUPTED)

            run_duration_ms = total_duration * 1000
            self._emit_telemetry(
                RunFinished(
                    timestamp=datetime.now(UTC),
                    run_id=run.run_id,
                    status=RunStatus.INTERRUPTED,
                    row_count=shutdown_exc.rows_processed,
                    duration_ms=run_duration_ms,
                )
            )

            self._events.emit(
                RunSummary(
                    run_id=run.run_id,
                    status=RunCompletionStatus.INTERRUPTED,
                    total_rows=shutdown_exc.rows_processed,
                    succeeded=shutdown_exc.rows_succeeded,
                    failed=shutdown_exc.rows_failed,
                    quarantined=shutdown_exc.rows_quarantined,
                    duration_seconds=total_duration,
                    exit_code=3,
                    routed=shutdown_exc.rows_routed,
                    routed_destinations=tuple(shutdown_exc.routed_destinations.items()),
                )
            )

            raise  # Propagate to CLI
        except Exception:
            # Emit RunSummary with failure status
            total_duration = time.perf_counter() - run_start_time

            if run_completed:
                # Export failed after successful run - emit PARTIAL status
                # NOTE: RunFinished was already emitted at lines 604-612
                # before the export attempt, so we only emit the EventBus event here
                self._events.emit(
                    RunSummary(
                        run_id=run.run_id,
                        status=RunCompletionStatus.PARTIAL,
                        total_rows=result.rows_processed,
                        succeeded=result.rows_succeeded,
                        failed=result.rows_failed,
                        quarantined=result.rows_quarantined,
                        duration_seconds=total_duration,
                        exit_code=1,
                        routed=result.rows_routed,
                        routed_destinations=tuple(result.routed_destinations.items()),
                    )
                )
            else:
                # Run failed before completion - emit FAILED status with zero metrics
                recorder.finalize_run(run.run_id, status=RunStatus.FAILED)

                # Emit telemetry AFTER Landscape finalize succeeds
                self._emit_telemetry(
                    RunFinished(
                        timestamp=datetime.now(UTC),
                        run_id=run.run_id,
                        status=RunStatus.FAILED,
                        row_count=0,
                        duration_ms=total_duration * 1000,
                    )
                )

                self._events.emit(
                    RunSummary(
                        run_id=run.run_id,
                        status=RunCompletionStatus.FAILED,
                        total_rows=0,
                        succeeded=0,
                        failed=0,
                        quarantined=0,
                        duration_seconds=total_duration,
                        exit_code=2,  # exit_code: 0=success, 1=partial, 2=total failure
                        routed=0,
                        routed_destinations=(),
                    )
                )
            raise  # CRITICAL: Always re-raise - observability doesn't suppress errors
        finally:
            # CRITICAL: Telemetry flush must not mask run errors or skip cleanup.
            # If _flush_telemetry() raises TelemetryExporterError (fail_on_total=True),
            # we must still run cleanup and preserve any pending exception.
            import sys

            import structlog

            from elspeth.telemetry.errors import TelemetryExporterError

            logger = structlog.get_logger()
            pending_exc = sys.exc_info()[0]

            try:
                self._flush_telemetry()
            except TelemetryExporterError as e:
                logger.warning(
                    "Telemetry flush failed - will raise after cleanup if no other exception pending",
                    exporter=e.exporter_name,
                    error=e.message,
                )
                if pending_exc is None:
                    raise
            # NOTE: Transform/sink/source cleanup is handled by _cleanup_plugins()
            # in _execute_run()'s finally block. No need for separate cleanup here.

    def _execute_run(
        self,
        recorder: LandscapeRecorder,
        run_id: str,
        config: PipelineConfig,
        graph: ExecutionGraph,
        settings: ElspethSettings | None = None,
        batch_checkpoints: dict[str, dict[str, Any]] | None = None,
        *,
        payload_store: PayloadStore,
        shutdown_event: threading.Event | None = None,
    ) -> RunResult:
        """Execute the run using the execution graph.

        The graph provides:
        - Node IDs and metadata via topological_order() and get_node_info()
        - Edges via get_edges()
        - Explicit ID mappings via get_sink_id_map() and get_transform_id_map()

        Args:
            recorder: LandscapeRecorder for audit trail
            run_id: Run identifier
            config: Pipeline configuration
            graph: Execution graph
            settings: Full settings (optional)
            batch_checkpoints: Restored batch checkpoints (maps node_id -> checkpoint_data)
            payload_store: Optional PayloadStore for persisting source row payloads
            shutdown_event: Optional threading.Event set by signal handler on SIGINT/SIGTERM.
                When set, the processing loop breaks after the current row completes.
                All pending work (aggregation flush, sink writes) is still performed.
        """
        # Store graph for checkpointing during execution
        self._current_graph = graph

        # Local imports for telemetry events - consolidated here to avoid repeated imports
        from elspeth.telemetry import (
            FieldResolutionApplied,
            PhaseChanged,
            RowCreated,
        )

        # Get execution order from graph
        execution_order = graph.topological_order()

        # Build node_id -> plugin instance mapping for metadata extraction
        # Source: single plugin from config.source
        source_id = graph.get_source()
        transform_id_map: dict[int, NodeID] = graph.get_transform_id_map()
        sink_id_map: dict[SinkName, NodeID] = graph.get_sink_id_map()
        config_gate_id_map: dict[GateName, NodeID] = graph.get_config_gate_id_map()
        aggregation_id_map: dict[AggregationName, NodeID] = graph.get_aggregation_id_map()

        # Build node ID sets for special node types
        config_gate_node_ids: set[NodeID] = set(config_gate_id_map.values())
        aggregation_node_ids: set[NodeID] = set(aggregation_id_map.values())

        # Map plugin instances to their node IDs for metadata extraction
        # Config gates and coalesce nodes don't have plugin instances (they're structural)
        # Aggregation transforms DO have instances - they're in config.transforms with node_id set
        node_to_plugin: dict[NodeID, Any] = {}
        if source_id is not None:
            node_to_plugin[source_id] = config.source
        for seq, transform in enumerate(config.transforms):
            if seq in transform_id_map:
                # Regular transform - mapped by sequence number
                node_to_plugin[transform_id_map[seq]] = transform
            elif transform.node_id is not None and NodeID(transform.node_id) in aggregation_node_ids:
                # Aggregation transform - has node_id set by CLI, not in transform_id_map
                node_to_plugin[NodeID(transform.node_id)] = transform
        for sink_name, sink in config.sinks.items():
            if SinkName(sink_name) in sink_id_map:
                node_to_plugin[sink_id_map[SinkName(sink_name)]] = sink
        coalesce_id_map: dict[CoalesceName, NodeID] = graph.get_coalesce_id_map()
        coalesce_node_ids: set[NodeID] = set(coalesce_id_map.values())

        # GRAPH phase - register nodes and edges in Landscape
        phase_start = time.perf_counter()
        try:
            self._events.emit(PhaseStarted(phase=PipelinePhase.GRAPH, action=PhaseAction.BUILDING))

            # Emit telemetry PhaseChanged - we now have run_id from begin_run
            self._emit_telemetry(
                PhaseChanged(
                    timestamp=datetime.now(UTC),
                    run_id=run_id,
                    phase=PipelinePhase.GRAPH,
                    action=PhaseAction.BUILDING,
                )
            )

            # Register nodes with Landscape using graph's node IDs and actual plugin metadata
            from elspeth.contracts import Determinism
            from elspeth.contracts.schema import SchemaConfig

            for node_id in execution_order:
                node_info = graph.get_node_info(node_id)

                # Config gates and coalesce nodes are structural (no plugin instances)
                # Aggregations have plugin instances in node_to_plugin (transforms with metadata)
                if node_id in config_gate_node_ids:
                    # Config gates are deterministic (expression evaluation is deterministic)
                    # Use engine version to track which version of ExpressionParser was used
                    plugin_version = f"engine:{ENGINE_VERSION}"
                    determinism = Determinism.DETERMINISTIC
                elif node_id in coalesce_node_ids:
                    # Coalesce nodes merge tokens from parallel paths - deterministic operation
                    # Use engine version to track which version of the coalesce logic was used
                    plugin_version = f"engine:{ENGINE_VERSION}"
                    determinism = Determinism.DETERMINISTIC
                else:
                    # Direct access - if node_id is in execution_order (from graph.topological_order()),
                    # it MUST be in node_to_plugin (built from the same graph's source, transforms, sinks).
                    # A KeyError here indicates a bug in graph construction or node_to_plugin building.
                    plugin = node_to_plugin[NodeID(node_id)]

                    # Extract plugin metadata - all protocols define these attributes,
                    # all base classes provide defaults. Direct access is safe.
                    plugin_version = plugin.plugin_version
                    determinism = plugin.determinism

                # Get schema_config — prefer computed output_schema_config
                # (includes guaranteed_fields, audit_fields from LLM transforms)
                # over raw config["schema"] which may omit computed contract fields.
                if node_info.output_schema_config is not None:
                    schema_config = node_info.output_schema_config
                else:
                    schema_dict = node_info.config["schema"]
                    schema_config = SchemaConfig.from_dict(schema_dict)

                # Get output_contract for source nodes
                # Sources have get_schema_contract() method that returns their output contract
                output_contract = None
                if node_id == source_id:
                    output_contract = config.source.get_schema_contract()

                recorder.register_node(
                    run_id=run_id,
                    node_id=node_id,
                    plugin_name=node_info.plugin_name,
                    node_type=NodeType(node_info.node_type),  # Already lowercase
                    plugin_version=plugin_version,
                    config=node_info.config,
                    determinism=determinism,
                    schema_config=schema_config,
                    output_contract=output_contract,
                )

            # Register edges from graph - key by (from_node, label) for lookup
            # Gates return route labels, so edge_map is keyed by label
            edge_map: dict[tuple[NodeID, str], str] = {}

            for edge_info in graph.get_edges():
                edge = recorder.register_edge(
                    run_id=run_id,
                    from_node_id=edge_info.from_node,
                    to_node_id=edge_info.to_node,
                    label=edge_info.label,
                    mode=edge_info.mode,
                )
                # Key by edge label - gates return route labels, transforms use "continue"
                edge_map[(NodeID(edge_info.from_node), edge_info.label)] = edge.edge_id

            # Get route resolution map - maps (gate_node, label) -> "continue" | sink_name
            route_resolution_map = graph.get_route_resolution_map()

            # Validate all route destinations BEFORE processing any rows
            # This catches config errors early instead of after partial processing
            # Note: config gates also add to route_resolution_map, validated the same way
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

            # Validate source quarantine destination
            # Call module function directly (no wrapper method)
            validate_source_quarantine_destination(
                source=config.source,
                available_sinks=set(config.sinks.keys()),
            )

            self._events.emit(PhaseCompleted(phase=PipelinePhase.GRAPH, duration_seconds=time.perf_counter() - phase_start))
        except Exception as e:
            self._events.emit(PhaseError(phase=PipelinePhase.GRAPH, error=e))
            raise  # CRITICAL: Always re-raise - graph validation failure is fatal

        # Get explicit node ID mappings from graph
        source_id = graph.get_source()
        if source_id is None:
            raise ValueError("Graph has no source node")
        sink_id_map = graph.get_sink_id_map()
        transform_id_map = graph.get_transform_id_map()
        config_gate_id_map = graph.get_config_gate_id_map()
        coalesce_id_map = graph.get_coalesce_id_map()

        # Assign node_ids to all plugins
        self._assign_plugin_node_ids(
            source=config.source,
            transforms=config.transforms,
            sinks=config.sinks,
            source_id=source_id,
            transform_id_map=transform_id_map,
            sink_id_map=sink_id_map,
        )

        # Create context with the LandscapeRecorder
        # Restore batch checkpoints if provided (from previous BatchPendingError)
        ctx = PluginContext(
            run_id=run_id,
            config=config.config,
            landscape=recorder,
            rate_limit_registry=self._rate_limit_registry,
            concurrency_config=self._concurrency_config,
            _batch_checkpoints=batch_checkpoints or {},
            telemetry_emit=self._emit_telemetry,
        )

        # Set node_id on context for source validation error attribution
        # This must be set BEFORE source.load() so that any validation errors
        # (e.g., malformed CSV rows) can be attributed to the source node
        ctx.node_id = source_id

        # Call on_start for all plugins BEFORE processing.
        # Order: source -> transforms (pipeline order) -> sinks.
        # Base classes provide no-op implementations, so no hasattr needed.
        # NOTE: on_start is called OUTSIDE the try/finally that calls
        # _cleanup_plugins. If on_start raises, on_complete/close are NOT called.
        config.source.on_start(ctx)
        for transform in config.transforms:
            transform.on_start(ctx)
        for sink in config.sinks.values():
            sink.on_start(ctx)

        processor, coalesce_node_map, coalesce_executor = self._build_processor(
            graph=graph,
            config=config,
            settings=settings,
            recorder=recorder,
            run_id=run_id,
            source_id=source_id,
            edge_map=edge_map,
            route_resolution_map=route_resolution_map,
            config_gate_id_map=config_gate_id_map,
            coalesce_id_map=coalesce_id_map,
            payload_store=payload_store,
        )

        # Process rows - Buffer TOKENS, not dicts, to preserve identity
        counters = ExecutionCounters()
        # Track (token, pending_outcome) pairs for deferred outcome recording
        # Outcomes are recorded by SinkExecutor.write() AFTER sink durability is achieved
        # Fix: P1-2026-01-31 - use PendingOutcome to carry error_hash for QUARANTINED
        pending_tokens: dict[str, list[tuple[TokenInfo, PendingOutcome | None]]] = {name: [] for name in config.sinks}

        # Pre-compute aggregation transform lookup for O(1) access per timeout check
        # Maps node_id_str -> (transform, aggregation_node_id)
        agg_transform_lookup: dict[str, tuple[TransformProtocol, NodeID]] = {}
        if config.aggregation_settings:
            for t in config.transforms:
                if isinstance(t, TransformProtocol) and t.is_batch_aware and t.node_id in config.aggregation_settings:
                    agg_transform_lookup[t.node_id] = (t, NodeID(t.node_id))

        # Progress tracking - hybrid timing: emit on 100 rows OR 5 seconds
        progress_interval = 100
        progress_time_interval = 5.0  # seconds
        start_time = time.perf_counter()
        last_progress_time = start_time

        # Compute default last_node_id for end-of-source checkpointing
        # (e.g., flush_pending when no rows were processed in the main loop)
        # This mirrors the in-loop logic for consistency
        default_last_node_id: str
        if config.gates:
            last_gate_name = config.gates[-1].name
            default_last_node_id = config_gate_id_map[GateName(last_gate_name)]
        elif config.transforms:
            transform_node_id = config.transforms[-1].node_id
            if transform_node_id is None:
                raise OrchestrationInvariantError("Last transform in pipeline has no node_id")
            default_last_node_id = transform_node_id
        else:
            default_last_node_id = source_id

        # SOURCE phase - initialize source and begin loading
        phase_start = time.perf_counter()
        self._events.emit(PhaseStarted(phase=PipelinePhase.SOURCE, action=PhaseAction.INITIALIZING, target=config.source.name))

        # Emit telemetry PhaseChanged for SOURCE
        self._emit_telemetry(
            PhaseChanged(
                timestamp=datetime.now(UTC),
                run_id=run_id,
                phase=PipelinePhase.SOURCE,
                action=PhaseAction.INITIALIZING,
            )
        )

        try:
            # Begin source_load operation to track external calls during load/iteration
            # This operation covers the entire source consumption lifecycle
            with track_operation(
                recorder=recorder,
                run_id=run_id,
                node_id=source_id,
                operation_type="source_load",
                ctx=ctx,
                input_data={"source_plugin": config.source.name},
            ) as source_op_handle:
                # Capture operation_id for restoration during iteration
                # Generator-based sources execute code during next() calls, so we need
                # to restore operation_id at the end of each iteration before the for
                # loop calls enumerate() again.
                source_operation_id = source_op_handle.operation.operation_id

                # Nested try for SOURCE phase to catch load() failures separately from PROCESS errors
                try:
                    with self._span_factory.source_span(config.source.name):
                        # Invoke load() to get iterator - any immediate failures (file not found) happen here
                        source_iterator = config.source.load(ctx)
                except Exception as e:
                    # SOURCE phase error (file not found, auth failure, etc.)
                    self._events.emit(PhaseError(phase=PipelinePhase.SOURCE, error=e, target=config.source.name))
                    raise  # Re-raise to propagate SOURCE failures (cleanup will still run via outer finally)

                self._events.emit(PhaseCompleted(phase=PipelinePhase.SOURCE, duration_seconds=time.perf_counter() - phase_start))

                # Track whether field resolution has been recorded (must happen after first iteration)
                field_resolution_recorded = False
                # Track whether schema contract has been recorded (must happen after first VALID row)
                # Separate from field_resolution because first row might be quarantined
                schema_contract_recorded = False

                # PROCESS phase - iterate through rows
                phase_start = time.perf_counter()
                self._events.emit(PhaseStarted(phase=PipelinePhase.PROCESS, action=PhaseAction.PROCESSING))

                # Emit telemetry PhaseChanged for PROCESS
                self._emit_telemetry(
                    PhaseChanged(
                        timestamp=datetime.now(UTC),
                        run_id=run_id,
                        phase=PipelinePhase.PROCESS,
                        action=PhaseAction.PROCESSING,
                    )
                )

                # Nested try for PROCESS phase to catch iteration/processing failures
                interrupted_by_shutdown = False
                try:
                    for row_index, source_item in enumerate(source_iterator):
                        counters.rows_processed += 1

                        # Record field resolution mapping on first iteration
                        # Must happen AFTER iterator advances because generators (like CSVSource.load())
                        # only execute their body when iterated. The _field_resolution assignment in
                        # CSVSource happens inside the generator, not when load() is called.
                        if not field_resolution_recorded:
                            field_resolution_recorded = True
                            field_resolution = config.source.get_field_resolution()
                            if field_resolution is not None:
                                resolution_mapping, normalization_version = field_resolution
                                recorder.record_source_field_resolution(
                                    run_id=run_id,
                                    resolution_mapping=resolution_mapping,
                                    normalization_version=normalization_version,
                                )
                                # Emit telemetry AFTER Landscape succeeds
                                self._emit_telemetry(
                                    FieldResolutionApplied(
                                        timestamp=datetime.now(UTC),
                                        run_id=run_id,
                                        source_plugin=config.source.name,
                                        field_count=len(resolution_mapping),
                                        normalization_version=normalization_version,
                                        resolution_mapping=resolution_mapping,
                                    )
                                )

                        # Handle quarantined source rows - route directly to sink
                        if source_item.is_quarantined:
                            counters.rows_quarantined += 1
                            # Route quarantined row to configured sink
                            # Per CLAUDE.md: plugin bugs must crash, no silent drops
                            quarantine_sink = source_item.quarantine_destination

                            # Validate destination exists - crash on plugin bug
                            if not quarantine_sink:
                                raise RouteValidationError(
                                    f"Source '{config.source.name}' yielded quarantined row "
                                    f"(row_index={row_index}) with missing quarantine_destination. "
                                    f"This is a plugin bug: quarantined rows MUST specify a destination. "
                                    f"Use SourceRow.quarantined(row, error, destination) factory method."
                                )
                            if quarantine_sink not in config.sinks:
                                raise RouteValidationError(
                                    f"Source '{config.source.name}' yielded quarantined row "
                                    f"(row_index={row_index}) with invalid quarantine_destination='{quarantine_sink}'. "
                                    f"No sink named '{quarantine_sink}' exists. "
                                    f"Available sinks: {sorted(config.sinks.keys())}. "
                                    f"This is a plugin bug: quarantine_destination must match "
                                    f"source._on_validation_failure='{config.source._on_validation_failure}'."
                                )

                            # Destination validated - proceed with routing.
                            # Sanitize quarantine data at Tier-3 boundary: replace non-finite
                            # floats (NaN, Infinity) with None so downstream canonical JSON
                            # and stable_hash operations succeed. The quarantine_error records
                            # what was originally wrong with the data.
                            source_item.row = sanitize_for_canonical(source_item.row)

                            # Create a token for the quarantined row using specialized method
                            # (quarantine rows don't have contracts - they failed validation)
                            quarantine_token = processor.token_manager.create_quarantine_token(
                                run_id=run_id,
                                source_node_id=source_id,
                                row_index=row_index,
                                source_row=source_item,
                            )

                            # Record source node_state (step_index=0) for quarantine audit lineage.
                            # Status is FAILED because the source validation rejected this row.
                            quarantine_data = source_item.row if isinstance(source_item.row, dict) else {"_raw": source_item.row}
                            quarantine_error_msg = source_item.quarantine_error or "unknown_validation_error"
                            source_state = recorder.begin_node_state(
                                token_id=quarantine_token.token_id,
                                node_id=source_id,
                                run_id=run_id,
                                step_index=0,
                                input_data=quarantine_data,
                                quarantined=True,
                            )
                            recorder.complete_node_state(
                                state_id=source_state.state_id,
                                status=NodeStateStatus.FAILED,
                                duration_ms=0,
                                error=ExecutionError(
                                    exception=quarantine_error_msg,
                                    type="ValidationError",
                                ),
                            )

                            # Record DIVERT routing_event for the quarantine edge.
                            # The __quarantine__ edge MUST exist — DAG creates it in
                            # the source quarantine edge block of from_plugin_instances().
                            quarantine_edge_key = (source_id, "__quarantine__")
                            try:
                                quarantine_edge_id = edge_map[quarantine_edge_key]
                            except KeyError:
                                raise OrchestrationInvariantError(
                                    f"Quarantine row reached orchestrator but no __quarantine__ "
                                    f"DIVERT edge exists in DAG for source '{source_id}'. "
                                    f"This is a DAG construction bug — "
                                    f"on_validation_failure should have created a DIVERT edge "
                                    f"in from_plugin_instances()."
                                ) from None
                            recorder.record_routing_event(
                                state_id=source_state.state_id,
                                edge_id=quarantine_edge_id,
                                mode=RoutingMode.DIVERT,
                                reason=SourceQuarantineReason(
                                    quarantine_error=quarantine_error_msg,
                                ),
                            )

                            # Emit RowCreated telemetry AFTER Landscape recording succeeds
                            # Quarantined rows are Tier-3 data that may contain non-canonical
                            # values (NaN, Infinity). Use stable_hash when possible, fall back
                            # to repr_hash for non-canonical data.
                            try:
                                quarantine_content_hash = stable_hash(source_item.row)
                            except (ValueError, TypeError):
                                quarantine_content_hash = repr_hash(source_item.row)
                            self._emit_telemetry(
                                RowCreated(
                                    timestamp=datetime.now(UTC),
                                    run_id=run_id,
                                    row_id=quarantine_token.row_id,
                                    token_id=quarantine_token.token_id,
                                    content_hash=quarantine_content_hash,
                                )
                            )

                            # Compute error_hash for QUARANTINED outcome audit trail
                            # Per CLAUDE.md: every row must reach exactly one terminal state
                            # Fix: P1-2026-01-31 - Do NOT record outcome here!
                            # Record outcome AFTER sink durability in SinkExecutor.write()
                            error_detail = source_item.quarantine_error or "unknown_validation_error"
                            quarantine_error_hash = hashlib.sha256(error_detail.encode()).hexdigest()[:16]

                            # Pass PendingOutcome with error_hash - outcome recorded after sink durability
                            pending_tokens[quarantine_sink].append(
                                (quarantine_token, PendingOutcome(RowOutcome.QUARANTINED, quarantine_error_hash))
                            )
                            # Emit progress before continue (ensures quarantined rows trigger updates)
                            # Hybrid timing: emit on first row, every 100 rows, or every 5 seconds
                            current_time = time.perf_counter()
                            time_since_last_progress = current_time - last_progress_time
                            should_emit = (
                                counters.rows_processed == 1  # First row - immediate feedback
                                or counters.rows_processed % progress_interval == 0  # Every 100 rows
                                or time_since_last_progress >= progress_time_interval  # Every 5 seconds
                            )
                            if should_emit:
                                elapsed = current_time - start_time
                                self._events.emit(
                                    ProgressEvent(
                                        rows_processed=counters.rows_processed,
                                        # Include routed rows in success count - they reached their destination
                                        rows_succeeded=counters.rows_succeeded + counters.rows_routed,
                                        rows_failed=counters.rows_failed,
                                        rows_quarantined=counters.rows_quarantined,
                                        elapsed_seconds=elapsed,
                                    )
                                )
                                last_progress_time = current_time
                            # Restore operation_id before next iteration
                            # (generator may execute external calls on next() call)
                            ctx.operation_id = source_operation_id

                            # Shutdown check for quarantine path — without this,
                            # a stream of quarantined rows would never hit the
                            # normal-path shutdown check (line ~1605) because
                            # `continue` skips it.
                            if shutdown_event is not None and shutdown_event.is_set():
                                interrupted_by_shutdown = True
                                break

                            # Skip normal processing - row is already handled
                            continue

                        # ─────────────────────────────────────────────────────────────────
                        # Record schema contract after first VALID row
                        # (BUG FIX: mwwo + c1v5 - contract only exists after first valid row)
                        #
                        # For OBSERVED/FLEXIBLE modes, the source's schema contract is set
                        # when the first valid row is processed. Quarantined rows don't
                        # trigger contract population. Recording must happen here, not on
                        # the first iteration which may be a quarantined row.
                        # ─────────────────────────────────────────────────────────────────
                        if not schema_contract_recorded:
                            schema_contract = config.source.get_schema_contract()
                            if schema_contract is not None:
                                schema_contract_recorded = True
                                # Update run-level contract
                                recorder.update_run_contract(run_id, schema_contract)
                                # Update source node's output_contract (was NULL at registration)
                                recorder.update_node_output_contract(run_id, source_id, schema_contract)
                                # Make contract available to transforms via context
                                # This enables contract-aware template access (original header names)
                                ctx.contract = schema_contract

                        # ─────────────────────────────────────────────────────────────────
                        # CRITICAL: Clear operation_id now that source item is fetched.
                        # Generator-based sources (e.g., AzureBlobSource) execute during
                        # iteration - their external calls (blob downloads) happen inside
                        # the for loop at the enumerate() call. By this point, the source
                        # item is fully fetched and any source-side calls are recorded
                        # with operation_id. Now we must clear it so transforms can set
                        # their own state_id without triggering the XOR constraint.
                        # ─────────────────────────────────────────────────────────────────
                        ctx.operation_id = None

                        # ─────────────────────────────────────────────────────────────────
                        # Check for timed-out aggregations BEFORE processing this row
                        # (BUG FIX: P1-2026-01-22 - ensures timeout flushes OLD batch)
                        #
                        # This is the critical fix: checking timeout BEFORE buffering the
                        # new row ensures the timed-out batch contains only previously
                        # buffered rows, not the row that just arrived.
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

                        results = processor.process_row(
                            row_index=row_index,
                            source_row=source_item,
                            transforms=config.transforms,
                            ctx=ctx,
                        )

                        # Handle all results from this source row (includes fork children)
                        #
                        # Note: Counters track processing outcomes (how many rows reached each state).
                        # Sink durability is tracked separately via checkpoints, which are created
                        # AFTER successful sink writes. A crash before sink write means:
                        # - Counters may be inflated (row counted but not persisted)
                        # - But recovery will correctly identify the unwritten rows
                        accumulate_row_outcomes(results, counters, config.sinks, pending_tokens)

                        # ─────────────────────────────────────────────────────────────────
                        # Check for timed-out coalesces after processing each row
                        # (BUG FIX: P1-2026-01-22 - check_timeouts was never called)
                        # ─────────────────────────────────────────────────────────────────
                        if coalesce_executor is not None:
                            handle_coalesce_timeouts(
                                coalesce_executor=coalesce_executor,
                                coalesce_node_map=coalesce_node_map,
                                processor=processor,
                                config_sinks=config.sinks,
                                ctx=ctx,
                                counters=counters,
                                pending_tokens=pending_tokens,
                            )

                        # Emit progress every N rows or every M seconds (after outcome counters are updated)
                        # Hybrid timing: emit on first row, every 100 rows, or every 5 seconds
                        current_time = time.perf_counter()
                        time_since_last_progress = current_time - last_progress_time
                        should_emit = (
                            counters.rows_processed == 1  # First row - immediate feedback
                            or counters.rows_processed % progress_interval == 0  # Every 100 rows
                            or time_since_last_progress >= progress_time_interval  # Every 5 seconds
                        )
                        if should_emit:
                            elapsed = current_time - start_time
                            self._events.emit(
                                ProgressEvent(
                                    rows_processed=counters.rows_processed,
                                    # Include routed rows in success count - they reached their destination
                                    rows_succeeded=counters.rows_succeeded + counters.rows_routed,
                                    rows_failed=counters.rows_failed,
                                    rows_quarantined=counters.rows_quarantined,
                                    elapsed_seconds=elapsed,
                                )
                            )
                            last_progress_time = current_time

                        # ─────────────────────────────────────────────────────────────────
                        # GRACEFUL SHUTDOWN CHECK
                        # Check between row iterations — current row is fully
                        # processed, outcomes recorded, safe to stop here.
                        # ─────────────────────────────────────────────────────────────────
                        if shutdown_event is not None and shutdown_event.is_set():
                            interrupted_by_shutdown = True
                            break

                        # ─────────────────────────────────────────────────────────────────
                        # CRITICAL: Restore operation_id before next iteration.
                        # Generator-based sources execute during next() calls in the for
                        # loop. Any external calls (blob downloads, API fetches) must be
                        # attributed to the source_load operation.
                        # ─────────────────────────────────────────────────────────────────
                        ctx.operation_id = source_operation_id

                    # ─────────────────────────────────────────────────────────────────
                    # CRITICAL: Flush remaining aggregation buffers at end-of-source
                    # ─────────────────────────────────────────────────────────────────
                    if config.aggregation_settings:
                        # Build checkpoint callback if checkpointing is enabled
                        checkpoint_callback: Callable[[TokenInfo], None] | None = None
                        if self._checkpoint_config and self._checkpoint_config.enabled and self._checkpoint_manager:

                            def make_checkpoint_callback() -> Callable[[TokenInfo], None]:
                                # Closure captures: run_id, default_last_node_id, processor, self
                                captured_run_id = run_id
                                captured_node_id = default_last_node_id

                                def callback(token: TokenInfo) -> None:
                                    agg_state = processor.get_aggregation_checkpoint_state()
                                    self._maybe_checkpoint(
                                        run_id=captured_run_id,
                                        token_id=token.token_id,
                                        node_id=captured_node_id,
                                        aggregation_state=agg_state,
                                    )

                                return callback

                            checkpoint_callback = make_checkpoint_callback()

                        # Call module function directly (no wrapper method)
                        flush_result = flush_remaining_aggregation_buffers(
                            config=config,
                            processor=processor,
                            ctx=ctx,
                            pending_tokens=pending_tokens,
                            checkpoint_callback=checkpoint_callback,
                        )
                        counters.accumulate_flush_result(flush_result)

                    # Flush pending coalesce operations at end-of-source
                    if coalesce_executor is not None:
                        flush_coalesce_pending(
                            coalesce_executor=coalesce_executor,
                            coalesce_node_map=coalesce_node_map,
                            processor=processor,
                            config_sinks=config.sinks,
                            ctx=ctx,
                            counters=counters,
                            pending_tokens=pending_tokens,
                        )

                    # Source iteration complete - for loop ends here

                    # ─────────────────────────────────────────────────────────────────
                    # Record field resolution for empty sources (header-only files).
                    # For sources with rows, this is recorded inside the loop on the
                    # first iteration. For empty sources, the loop never executes, but
                    # the source may still have computed field resolution (e.g., CSV
                    # sources read headers before yielding data rows).
                    # ─────────────────────────────────────────────────────────────────
                    if not field_resolution_recorded:
                        field_resolution = config.source.get_field_resolution()
                        if field_resolution is not None:
                            resolution_mapping, normalization_version = field_resolution
                            recorder.record_source_field_resolution(
                                run_id=run_id,
                                resolution_mapping=resolution_mapping,
                                normalization_version=normalization_version,
                            )
                            # Emit telemetry AFTER Landscape succeeds
                            self._emit_telemetry(
                                FieldResolutionApplied(
                                    timestamp=datetime.now(UTC),
                                    run_id=run_id,
                                    source_plugin=config.source.name,
                                    field_count=len(resolution_mapping),
                                    normalization_version=normalization_version,
                                    resolution_mapping=resolution_mapping,
                                )
                            )
                            field_resolution_recorded = True

                    # ─────────────────────────────────────────────────────────────────
                    # Record schema contract for runs with no valid source rows.
                    #
                    # In-loop recording happens on the first VALID row. For all-invalid
                    # or empty inputs, that branch never executes. Sources may still
                    # finalize a locked contract at end-of-load (e.g. FLEXIBLE with
                    # declared fields, OBSERVED/FLEXIBLE empty input). Persist it here
                    # so resume invariants still hold.
                    # ─────────────────────────────────────────────────────────────────
                    if not schema_contract_recorded:
                        schema_contract = config.source.get_schema_contract()
                        if schema_contract is not None:
                            schema_contract_recorded = True
                            # Update run-level contract
                            recorder.update_run_contract(run_id, schema_contract)
                            # Update source node's output_contract (was NULL at registration)
                            recorder.update_node_output_contract(run_id, source_id, schema_contract)
                            # Keep context contract aligned with recorded contract
                            ctx.contract = schema_contract

                except BatchPendingError:
                    # BatchPendingError is a control-flow signal, not an error.
                    # Don't emit PhaseError - the run isn't failing, it's just waiting.
                    raise  # Re-raise immediately for caller to handle retry
                except Exception as e:
                    # PROCESS phase error (iteration or processing failures)
                    self._events.emit(PhaseError(phase=PipelinePhase.PROCESS, error=e, target=config.source.name))
                    raise  # CRITICAL: Always re-raise - exceptions in PROCESS phase must propagate

            # track_operation ended - source_load operation is now complete
            # Source duration is now accurately measured (excludes sink I/O)

            # ─────────────────────────────────────────────────────────────────────────
            # SINK WRITES - Outside source_load track_operation context
            # Each sink write has its own track_operation (sink_write) in SinkExecutor.
            # This ensures sink failures are not misattributed to the source operation.
            # ─────────────────────────────────────────────────────────────────────────

            # Create checkpoint callback factory for post-sink checkpointing
            # Captures processor to get aggregation state for crash recovery
            def checkpoint_after_sink(sink_node_id: str) -> Callable[[TokenInfo], None]:
                def callback(token: TokenInfo) -> None:
                    agg_state = processor.get_aggregation_checkpoint_state()
                    self._maybe_checkpoint(
                        run_id=run_id,
                        token_id=token.token_id,
                        node_id=sink_node_id,
                        aggregation_state=agg_state,
                    )

                return callback

            self._write_pending_to_sinks(
                recorder=recorder,
                run_id=run_id,
                config=config,
                ctx=ctx,
                pending_tokens=pending_tokens,
                sink_id_map=sink_id_map,
                sink_step=processor.resolve_sink_step(),
                on_token_written_factory=checkpoint_after_sink,
            )

            # If shutdown interrupted the loop, raise after all pending work is flushed.
            # At this point: aggregation buffers flushed, coalesce flushed, sink writes done.
            if interrupted_by_shutdown:
                raise GracefulShutdownError(
                    rows_processed=counters.rows_processed,
                    run_id=run_id,
                    rows_succeeded=counters.rows_succeeded,
                    rows_failed=counters.rows_failed,
                    rows_quarantined=counters.rows_quarantined,
                    rows_routed=counters.rows_routed,
                    routed_destinations=dict(counters.routed_destinations),
                )

            # Emit final progress if we haven't emitted recently or row count not on interval
            # (RunSummary will show final summary regardless, but progress shows intermediate state)
            current_time = time.perf_counter()
            time_since_last_progress = current_time - last_progress_time
            # Emit if: not on progress_interval boundary OR >1s since last emission
            if counters.rows_processed % progress_interval != 0 or time_since_last_progress >= 1.0:
                elapsed = current_time - start_time
                self._events.emit(
                    ProgressEvent(
                        rows_processed=counters.rows_processed,
                        # Include routed rows in success count - they reached their destination
                        rows_succeeded=counters.rows_succeeded + counters.rows_routed,
                        rows_failed=counters.rows_failed,
                        rows_quarantined=counters.rows_quarantined,
                        elapsed_seconds=elapsed,
                    )
                )

            # PROCESS phase completed successfully
            self._events.emit(PhaseCompleted(phase=PipelinePhase.PROCESS, duration_seconds=time.perf_counter() - phase_start))

        finally:
            self._cleanup_plugins(config, ctx, include_source=True)

        # Clear graph after execution completes
        self._current_graph = None

        return counters.to_run_result(run_id)

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

        STATELESS: Like run(), creates fresh recorder and processor internally.
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
            raise ValueError("payload_store is required for resume - row data must be retrieved from stored payloads")

        run_id = resume_point.checkpoint.run_id

        # Create fresh recorder (stateless, like run())
        # Pass payload_store for external call payload persistence
        recorder = LandscapeRecorder(self._db, payload_store=payload_store)

        # 1. Handle incomplete batches - call module function directly
        handle_incomplete_batches(recorder, run_id)

        # 2. Update run status to running
        recorder.update_run_status(run_id, RunStatus.RUNNING)

        # 3. Build restored aggregation state map
        restored_state: dict[str, dict[str, Any]] = {}
        if resume_point.aggregation_state is not None:
            restored_state[resume_point.node_id] = resume_point.aggregation_state

        # 4. Get unprocessed row data from payload store
        from elspeth.core.checkpoint import RecoveryManager

        if self._checkpoint_manager is None:
            raise ValueError("CheckpointManager is required for resume - Orchestrator must be initialized with checkpoint_manager")
        recovery = RecoveryManager(self._db, self._checkpoint_manager)

        # TYPE FIDELITY: Retrieve source schema from audit trail for type restoration
        # Resume must use the ORIGINAL run's schema, not the current source's schema
        # This enables proper type coercion (datetime/Decimal) from JSON payload strings
        source_schema_json = recorder.get_source_schema(run_id)

        # Deserialize schema and recreate Pydantic model class with full type fidelity
        # Call module function directly (no wrapper method)
        schema_dict = json.loads(source_schema_json)
        source_schema_class = reconstruct_schema_from_json(schema_dict)

        # PIPELINEROW MIGRATION: Retrieve contract from audit trail for row wrapping
        # During resume, we need to wrap plain dicts in PipelineRow with contract
        # This ensures type fidelity and maintains the same data structures as main run
        schema_contract = recorder.get_run_contract(run_id)
        if schema_contract is None:
            # TIER-1 AUDIT INTEGRITY: Crash if contract is missing from audit trail
            # Per CLAUDE.md: "Bad data in the audit trail = crash immediately"
            # Inferring a contract from row data would:
            # 1. Mask missing/corrupt audit data (evidence tampering)
            # 2. Produce incomplete contracts (fields appearing later are omitted)
            # 3. Violate the NO LEGACY CODE POLICY (no backward compatibility shims)
            raise OrchestrationInvariantError(
                f"Cannot resume run '{run_id}': schema contract is missing from audit trail. "
                f"This indicates either:\n"
                f"  1. The audit database is corrupt or incomplete\n"
                f"  2. The run was started with a version that didn't record contracts\n"
                f"Resume cannot proceed safely without the schema contract. "
                f"The audit trail must be complete and trustworthy."
            )

        unprocessed_rows = recovery.get_unprocessed_row_data(run_id, payload_store, source_schema_class=source_schema_class)

        if not unprocessed_rows:
            # All rows were processed - complete the run
            recorder.finalize_run(run_id, status=RunStatus.COMPLETED)

            # Delete checkpoints on successful completion (Bug #8 fix)
            self._delete_checkpoints(run_id)

            return RunResult(
                run_id=run_id,
                status=RunStatus.COMPLETED,
                rows_processed=0,
                rows_succeeded=0,
                rows_failed=0,
                rows_routed=0,
                routed_destinations={},
            )

        # 5. Process unprocessed rows (with graceful shutdown support)
        from elspeth.telemetry import RunFinished

        resume_start_time = time.perf_counter()

        # When shutdown_event is provided (testing), skip signal handler
        # installation and use the caller's event directly.
        shutdown_ctx = nullcontext(shutdown_event) if shutdown_event is not None else self._shutdown_handler_context()

        try:
            with shutdown_ctx as active_event:
                result = self._process_resumed_rows(
                    recorder=recorder,
                    run_id=run_id,
                    config=config,
                    graph=graph,
                    unprocessed_rows=unprocessed_rows,
                    restored_aggregation_state=restored_state,
                    settings=settings,
                    payload_store=payload_store,
                    schema_contract=schema_contract,
                    shutdown_event=active_event,
                )
        except GracefulShutdownError as shutdown_exc:
            # Graceful shutdown: all in-flight work flushed, sinks written.
            # Mark run INTERRUPTED (resumable via `elspeth resume`).
            total_duration = time.perf_counter() - resume_start_time
            recorder.finalize_run(run_id, status=RunStatus.INTERRUPTED)

            run_duration_ms = total_duration * 1000
            self._emit_telemetry(
                RunFinished(
                    timestamp=datetime.now(UTC),
                    run_id=run_id,
                    status=RunStatus.INTERRUPTED,
                    row_count=shutdown_exc.rows_processed,
                    duration_ms=run_duration_ms,
                )
            )

            self._events.emit(
                RunSummary(
                    run_id=run_id,
                    status=RunCompletionStatus.INTERRUPTED,
                    total_rows=shutdown_exc.rows_processed,
                    succeeded=shutdown_exc.rows_succeeded,
                    failed=shutdown_exc.rows_failed,
                    quarantined=shutdown_exc.rows_quarantined,
                    duration_seconds=total_duration,
                    exit_code=3,
                    routed=shutdown_exc.rows_routed,
                    routed_destinations=tuple(shutdown_exc.routed_destinations.items()),
                )
            )

            raise  # Propagate to CLI

        # 6. Complete the run with reproducibility grade
        recorder.finalize_run(run_id, status=RunStatus.COMPLETED)
        result.status = RunStatus.COMPLETED

        # 7. Emit RunFinished telemetry (Bug fix #3: resume was missing this)
        resume_duration_ms = (time.perf_counter() - resume_start_time) * 1000
        self._emit_telemetry(
            RunFinished(
                timestamp=datetime.now(UTC),
                run_id=run_id,
                status=RunStatus.COMPLETED,
                row_count=result.rows_processed,
                duration_ms=resume_duration_ms,
            )
        )

        # 8. Emit RunSummary event (Bug fix #4: resume was missing this)
        total_duration = time.perf_counter() - resume_start_time
        self._events.emit(
            RunSummary(
                run_id=run_id,
                status=RunCompletionStatus.COMPLETED,
                total_rows=result.rows_processed,
                succeeded=result.rows_succeeded,
                failed=result.rows_failed,
                quarantined=result.rows_quarantined,
                duration_seconds=total_duration,
                exit_code=0,
                routed=result.rows_routed,
                routed_destinations=tuple(result.routed_destinations.items()),
            )
        )

        # 9. Delete checkpoints on successful completion
        self._delete_checkpoints(run_id)

        return result

    def _process_resumed_rows(
        self,
        recorder: LandscapeRecorder,
        run_id: str,
        config: PipelineConfig,
        graph: ExecutionGraph,
        unprocessed_rows: list[tuple[str, int, dict[str, Any]]],
        restored_aggregation_state: dict[str, dict[str, Any]],
        settings: ElspethSettings | None = None,
        *,
        payload_store: PayloadStore,
        schema_contract: SchemaContract,
        shutdown_event: threading.Event | None = None,
    ) -> RunResult:
        """Process unprocessed rows during resume.

        Follows the same pattern as _execute_run() but:
        - Row data comes from unprocessed_rows (not source)
        - Source plugin is NOT called (data already recorded)
        - Restored aggregation state is passed to processor
        - Uses process_existing_row() instead of process_row()

        Args:
            recorder: LandscapeRecorder for audit trail
            run_id: Run being resumed
            config: Pipeline configuration
            graph: Execution graph
            unprocessed_rows: List of (row_id, row_index, row_data) tuples
            restored_aggregation_state: Map of node_id -> state dict
            settings: Full settings (optional)
            payload_store: Optional PayloadStore for persisting source row payloads
            schema_contract: SchemaContract for wrapping row data in PipelineRow

        Returns:
            RunResult with processing counts
        """
        # Store graph for checkpointing during execution
        self._current_graph = graph

        # Get explicit node ID mappings from graph
        source_id = graph.get_source()
        if source_id is None:
            raise ValueError("Graph has no source node")
        sink_id_map = graph.get_sink_id_map()
        transform_id_map = graph.get_transform_id_map()
        config_gate_id_map = graph.get_config_gate_id_map()
        coalesce_id_map = graph.get_coalesce_id_map()

        # Build edge_map from database (load real edge IDs registered in original run)
        # CRITICAL: Must use real edge_ids for FK integrity when recording routing events
        # Convert keys from (str, str) to (NodeID, str) to match RowProcessor's type
        raw_edge_map = recorder.get_edge_map(run_id)
        edge_map: dict[tuple[NodeID, str], str] = {(NodeID(k[0]), k[1]): v for k, v in raw_edge_map.items()}

        # Validate: If graph has edges, database MUST have matching edges (Tier 1 trust)
        # Missing edges = data corruption or incomplete original run registration
        graph_edges = graph.get_edges()
        if graph_edges and not edge_map:
            raise ValueError(
                f"Resume failed: Graph has {len(graph_edges)} edges but no edges found in database "
                f"for run_id '{run_id}'. This indicates data corruption or incomplete edge registration "
                f"in the original run. Cannot resume without edge data."
            )

        # Get route resolution map
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

        # Validate source quarantine destination
        # Call module function directly (no wrapper method)
        validate_source_quarantine_destination(
            source=config.source,
            available_sinks=set(config.sinks.keys()),
        )

        # Assign node_ids to all plugins
        self._assign_plugin_node_ids(
            source=config.source,
            transforms=config.transforms,
            sinks=config.sinks,
            source_id=source_id,
            transform_id_map=transform_id_map,
            sink_id_map=sink_id_map,
        )

        # Create context
        ctx = PluginContext(
            run_id=run_id,
            config=config.config,
            landscape=recorder,
            rate_limit_registry=self._rate_limit_registry,
            concurrency_config=self._concurrency_config,
            telemetry_emit=self._emit_telemetry,
        )

        # Restore contract from run for transforms (was recorded during original run)
        # This enables contract-aware template access (original header names) during resume
        ctx.contract = recorder.get_run_contract(run_id)

        # Call on_start for transforms and sinks.
        # Source's on_start/on_complete are intentionally skipped because:
        # 1. Source's load() is not called - row data comes from stored payloads
        # 2. The source used during resume is NullSource, which has no resources to manage
        # 3. If a real source with resources were used in the future (e.g., holding
        #    a database connection), on_start/on_complete would need to be called here
        for transform in config.transforms:
            transform.on_start(ctx)
        for sink in config.sinks.values():
            sink.on_start(ctx)

        processor, coalesce_node_map, coalesce_executor = self._build_processor(
            graph=graph,
            config=config,
            settings=settings,
            recorder=recorder,
            run_id=run_id,
            source_id=source_id,
            edge_map=edge_map,
            route_resolution_map=route_resolution_map,
            config_gate_id_map=config_gate_id_map,
            coalesce_id_map=coalesce_id_map,
            payload_store=payload_store,
            restored_aggregation_state={NodeID(k): v for k, v in restored_aggregation_state.items()},
        )

        # Process rows - Buffer TOKENS
        counters = ExecutionCounters()
        # Track (token, pending_outcome) pairs for deferred outcome recording
        # Outcomes are recorded by SinkExecutor.write() AFTER sink durability is achieved
        # Fix: P1-2026-01-31 - use PendingOutcome to carry error_hash for QUARANTINED
        pending_tokens: dict[str, list[tuple[TokenInfo, PendingOutcome | None]]] = {name: [] for name in config.sinks}

        # Pre-compute aggregation transform lookup for O(1) access per timeout check
        agg_transform_lookup: dict[str, tuple[TransformProtocol, NodeID]] = {}
        if config.aggregation_settings:
            for t in config.transforms:
                if isinstance(t, TransformProtocol) and t.is_batch_aware and t.node_id in config.aggregation_settings:
                    agg_transform_lookup[t.node_id] = (t, NodeID(t.node_id))

        interrupted_by_shutdown = False

        try:
            # Process each unprocessed row using process_existing_row
            # (rows already exist in DB, only tokens need to be created)
            #
            # NOTE: No checkpointing during resume processing.
            # This is intentional for the following reasons:
            # 1. Resume typically handles few rows (those after the original checkpoint)
            # 2. Adding checkpointing during resume increases complexity significantly
            # 3. If resume crashes, re-running from the original checkpoint is acceptable
            # 4. For very large resume scenarios, a future enhancement could add checkpoint
            #    support, but the current design prioritizes simplicity over edge-case
            #    optimization
            for row_id, _row_index, row_data in unprocessed_rows:
                counters.rows_processed += 1

                # ─────────────────────────────────────────────────────────────────
                # Check for timed-out aggregations BEFORE processing this row
                # (BUG FIX: P1-2026-01-22 - ensures timeout flushes OLD batch)
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

                # Wrap row_data in PipelineRow with contract (PIPELINEROW MIGRATION)
                # Row data from resume is a plain dict, but process_existing_row expects PipelineRow
                pipeline_row = PipelineRow(data=row_data, contract=schema_contract)

                results = processor.process_existing_row(
                    row_id=row_id,
                    row_data=pipeline_row,
                    transforms=config.transforms,
                    ctx=ctx,
                )

                # Handle all results from this row
                accumulate_row_outcomes(results, counters, config.sinks, pending_tokens)

                # ─────────────────────────────────────────────────────────────────
                # Check for timed-out coalesces after processing each row
                # (BUG FIX: P1-2026-01-22 - check_timeouts was never called)
                # ─────────────────────────────────────────────────────────────────
                if coalesce_executor is not None:
                    handle_coalesce_timeouts(
                        coalesce_executor=coalesce_executor,
                        coalesce_node_map=coalesce_node_map,
                        processor=processor,
                        config_sinks=config.sinks,
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

            # ─────────────────────────────────────────────────────────────────
            # CRITICAL: Flush remaining aggregation buffers at end-of-source
            # ─────────────────────────────────────────────────────────────────
            if config.aggregation_settings:
                # Call module function directly (no wrapper method)
                # No checkpointing during resume
                flush_result = flush_remaining_aggregation_buffers(
                    config=config,
                    processor=processor,
                    ctx=ctx,
                    pending_tokens=pending_tokens,
                    checkpoint_callback=None,
                )
                counters.accumulate_flush_result(flush_result)

            # Flush pending coalesce operations
            if coalesce_executor is not None:
                flush_coalesce_pending(
                    coalesce_executor=coalesce_executor,
                    coalesce_node_map=coalesce_node_map,
                    processor=processor,
                    config_sinks=config.sinks,
                    ctx=ctx,
                    counters=counters,
                    pending_tokens=pending_tokens,
                )

            # Write to sinks (no checkpoint callbacks for resume path)
            self._write_pending_to_sinks(
                recorder=recorder,
                run_id=run_id,
                config=config,
                ctx=ctx,
                pending_tokens=pending_tokens,
                sink_id_map=sink_id_map,
                sink_step=processor.resolve_sink_step(),
            )

            # If shutdown interrupted the loop, raise after all pending work is flushed.
            # At this point: aggregation buffers flushed, coalesce flushed, sink writes done.
            if interrupted_by_shutdown:
                raise GracefulShutdownError(
                    rows_processed=counters.rows_processed,
                    run_id=run_id,
                    rows_succeeded=counters.rows_succeeded,
                    rows_failed=counters.rows_failed,
                    rows_quarantined=counters.rows_quarantined,
                    rows_routed=counters.rows_routed,
                    routed_destinations=dict(counters.routed_destinations),
                )

        finally:
            self._cleanup_plugins(config, ctx, include_source=False)

        # Clear graph after execution completes
        self._current_graph = None

        return counters.to_run_result(run_id)
