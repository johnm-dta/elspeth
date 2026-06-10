"""RunExecutionCore: run-context construction and sink-writing cluster.

Extracted from Orchestrator (core.py) — these methods own the construction
of the per-run :class:`RunContext` (node-id assignment, ``on_start`` plugin
lifecycle, processor assembly) and the flushing of pending tokens to sinks.
They are shared verbatim by both the main run path (``_execute_run``) and the
resume path (``_process_resumed_rows``).

Dependencies held by this core:
- ``_ceremony``: RunCeremony for telemetry emission
- ``_checkpoints``: CheckpointCoordinator for interrupted-progress checkpoints
- ``_span_factory``: SpanFactory for executor spans
- ``_clock``: Clock injected into processors/executors
- ``_concurrency_config``: RuntimeConcurrencyConfig for worker counts / context
- ``_rate_limit_registry``: RateLimitRegistry for the PluginContext
- ``_coalesce_completed_keys_limit``: bound on CoalesceExecutor completed keys
- ``_telemetry``: TelemetryManagerProtocol passed directly to RowProcessor
"""

from __future__ import annotations

import threading
from collections.abc import Mapping
from typing import TYPE_CHECKING

from elspeth.contracts import (
    TransformProtocol,
)
from elspeth.contracts.config import RuntimeRetryConfig
from elspeth.contracts.errors import (
    FrameworkBugError,
    GracefulShutdownError,
    OrchestrationInvariantError,
)
from elspeth.contracts.plugin_context import PluginContext
from elspeth.contracts.schema_contract_factory import create_contract_from_config
from elspeth.contracts.types import (
    CoalesceName,
    NodeID,
    SinkName,
)
from elspeth.core.config import AggregationSettings
from elspeth.core.landscape.factory import RecorderFactory
from elspeth.engine.executors.sink import DiversionCounts
from elspeth.engine.orchestrator.cleanup import cleanup_plugins
from elspeth.engine.orchestrator.graph_wiring import (
    assign_plugin_node_ids,
    build_dag_traversal_context,
)
from elspeth.engine.orchestrator.outcomes import (
    reconcile_sink_write_diversions,
)
from elspeth.engine.orchestrator.types import (
    AggNodeEntry,
    RunContext,
)
from elspeth.engine.processor import RowProcessor, make_step_resolver
from elspeth.engine.retry import RetryManager

if TYPE_CHECKING:
    from elspeth.contracts import (
        PendingOutcome,
        RouteDestination,
        SinkProtocol,
        TokenInfo,
    )
    from elspeth.contracts.aggregation_checkpoint import AggregationCheckpointState
    from elspeth.contracts.coalesce_checkpoint import CoalesceCheckpointState
    from elspeth.contracts.config.runtime import RuntimeConcurrencyConfig
    from elspeth.contracts.payload_store import PayloadStore
    from elspeth.contracts.types import (
        BranchName,
        GateName,
    )
    from elspeth.core.config import ElspethSettings
    from elspeth.core.dag import ExecutionGraph
    from elspeth.core.rate_limit import RateLimitRegistry
    from elspeth.engine.clock import Clock
    from elspeth.engine.coalesce_executor import CoalesceExecutor
    from elspeth.engine.orchestrator.ceremony import RunCeremony
    from elspeth.engine.orchestrator.checkpointing import CheckpointCoordinator
    from elspeth.engine.orchestrator.types import (
        CheckpointAfterSinkCallback,
        ExecutionCounters,
        GraphArtifacts,
        LoopContext,
        PendingTokenMap,
        PipelineConfig,
        TelemetryManagerProtocol,
        _CheckpointFactory,
    )
    from elspeth.engine.spans import SpanFactory


class RunExecutionCore:
    """Owns run-context construction and the sink-write cluster of an Orchestrator run.

    Extracted from Orchestrator — holds the four methods that are shared
    verbatim by both the main run path and the resume path: run-context
    initialization, processor construction, sink flushing, and the underlying
    pending-token sink writer.
    """

    def __init__(
        self,
        *,
        ceremony: RunCeremony,
        checkpoints: CheckpointCoordinator,
        span_factory: SpanFactory,
        clock: Clock,
        concurrency_config: RuntimeConcurrencyConfig | None,
        rate_limit_registry: RateLimitRegistry | None,
        coalesce_completed_keys_limit: int,
        telemetry: TelemetryManagerProtocol | None,
    ) -> None:
        self._ceremony = ceremony
        self._checkpoints = checkpoints
        self._span_factory = span_factory
        self._clock = clock
        self._concurrency_config = concurrency_config
        self._rate_limit_registry = rate_limit_registry
        self._coalesce_completed_keys_limit = coalesce_completed_keys_limit
        self._telemetry = telemetry

    def write_pending_to_sinks(
        self,
        factory: RecorderFactory,
        run_id: str,
        config: PipelineConfig,
        ctx: PluginContext,
        counters: ExecutionCounters,
        pending_tokens: PendingTokenMap,
        sink_id_map: dict[SinkName, NodeID],
        edge_map: Mapping[tuple[NodeID, str], str],
        sink_step: int,
        *,
        on_token_written_factory: _CheckpointFactory | None = None,
    ) -> DiversionCounts:
        """Write pending tokens to sinks using SinkExecutor.

        Extracted from _execute_run() and _process_resumed_rows() to eliminate
        duplication of the sink write orchestration pattern.

        Args:
            factory: RecorderFactory for audit trail
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

        from elspeth.engine.executors.sink import DiversionCounts, SinkExecutor

        sink_executor = SinkExecutor(factory.execution, factory.data_flow, self._span_factory, run_id)
        step = sink_step
        total_diversions = DiversionCounts()

        for sink_name, token_outcome_pairs in pending_tokens.items():
            if not token_outcome_pairs:
                continue
            if sink_name not in config.sinks:
                raise OrchestrationInvariantError(
                    f"Sink '{sink_name}' in pending_tokens not found in config.sinks. "
                    f"Available: {sorted(config.sinks.keys())}. "
                    f"This indicates a token routing bug."
                )
            sink = config.sinks[sink_name]
            sink_node_id = sink_id_map[SinkName(sink_name)]

            # Resolve failsink reference (if configured and not 'discard')

            failsink: SinkProtocol | None = None
            failsink_config_name: str | None = None
            failsink_edge_id: str | None = None
            on_write_failure = sink._on_write_failure
            if on_write_failure is not None and on_write_failure != "discard":
                if on_write_failure not in config.sinks:
                    raise OrchestrationInvariantError(
                        f"Sink '{sink_name}' on_write_failure references '{on_write_failure}' "
                        f"which passed validation but is not in config.sinks at runtime. "
                        f"Available: {sorted(config.sinks.keys())}."
                    )
                failsink = config.sinks[on_write_failure]
                failsink_config_name = on_write_failure
                failsink_edge_key = (sink_node_id, "__failsink__")
                try:
                    failsink_edge_id = edge_map[failsink_edge_key]
                except KeyError as exc:
                    raise OrchestrationInvariantError(
                        f"Sink '{sink_name}' on_write_failure='{on_write_failure}' "
                        f"but no __failsink__ DIVERT edge exists in DAG for node '{sink_node_id}'. "
                        f"This is a DAG construction bug — on_write_failure should have "
                        f"created a DIVERT edge in from_plugin_instances()."
                    ) from exc

            # Group tokens by pending_outcome for separate write() calls
            # (sink_executor.write() takes a single PendingOutcome for all tokens in a batch)
            # PendingOutcome carries error_hash for QUARANTINED tokens
            def pending_sort_key(pair: tuple[TokenInfo, PendingOutcome | None]) -> tuple[bool, str, str, str, bool]:
                pending = pair[1]
                if pending is None:
                    return (True, "", "", "", False)  # None sorts last
                outcome_value = pending.outcome.value if pending.outcome is not None else ""
                return (False, outcome_value, pending.path.value, pending.error_hash or "", pending.scheduler_pending_sink)

            sorted_pairs = sorted(token_outcome_pairs, key=pending_sort_key)

            for _group_key, group in groupby(sorted_pairs, key=pending_sort_key):
                group_pairs = list(group)
                pending_outcome = group_pairs[0][1]
                group_tokens = [token for token, _pending in group_pairs]
                # Only tokens with a proven durable PENDING_SINK handoff are
                # terminalized after sink durability. Generated terminal
                # outputs (aggregation/coalesce flushes) and source-quarantine
                # rows still need checkpoints but have no scheduler row to
                # close.
                terminalize_scheduler = bool(pending_outcome is not None and pending_outcome.scheduler_pending_sink)
                on_token_written: CheckpointAfterSinkCallback | None = None
                if on_token_written_factory is not None:
                    on_token_written = on_token_written_factory(sink_node_id, terminalize_scheduler=terminalize_scheduler)
                _, diversion_counts = sink_executor.write(
                    sink=sink,
                    tokens=group_tokens,
                    ctx=ctx,
                    step_in_pipeline=step,
                    sink_name=sink_name,
                    pending_outcome=pending_outcome,
                    failsink=failsink,
                    failsink_name=failsink_config_name,
                    failsink_edge_id=failsink_edge_id,
                    on_token_written=on_token_written,
                )
                if on_token_written is not None:
                    on_token_written.flush()
                reconcile_sink_write_diversions(
                    counters=counters,
                    sink_name=sink_name,
                    pending_outcome=pending_outcome,
                    diversion_count=diversion_counts.total,
                )
                total_diversions = DiversionCounts(
                    failsink_mode=total_diversions.failsink_mode + diversion_counts.failsink_mode,
                    discard_mode=total_diversions.discard_mode + diversion_counts.discard_mode,
                )

        return total_diversions

    def build_processor(
        self,
        *,
        graph: ExecutionGraph,
        config: PipelineConfig,
        settings: ElspethSettings | None,
        factory: RecorderFactory,
        run_id: str,
        source_id: NodeID,
        edge_map: dict[tuple[NodeID, str], str],
        route_resolution_map: dict[tuple[NodeID, str], RouteDestination] | None,
        config_gate_id_map: dict[GateName, NodeID],
        coalesce_id_map: dict[CoalesceName, NodeID],
        payload_store: PayloadStore,
        restored_aggregation_state: Mapping[NodeID, AggregationCheckpointState] | None = None,
        restored_coalesce_state: CoalesceCheckpointState | None = None,
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
        traversal = build_dag_traversal_context(graph, config, config_gate_id_map)

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
            token_manager = TokenManager(factory.data_flow, step_resolver=step_resolver)
            coalesce_executor = CoalesceExecutor(
                execution=factory.execution,
                span_factory=self._span_factory,
                token_manager=token_manager,
                run_id=run_id,
                step_resolver=step_resolver,
                clock=self._clock,
                max_completed_keys=self._coalesce_completed_keys_limit,
                data_flow=factory.data_flow,
            )

            for coalesce_settings_entry in settings.coalesce:
                coalesce_node_id = coalesce_id_map[CoalesceName(coalesce_settings_entry.name)]
                # Extract guaranteed fields from branch schemas for lost-branch audit trail.
                # Returns dict[branch_name, SchemaConfig]; we extract guaranteed fields.
                branch_schema_configs = graph.get_coalesce_branch_schemas(CoalesceName(coalesce_settings_entry.name))
                branch_schemas: dict[str, tuple[str, ...]] | None = None
                if branch_schema_configs:
                    branch_schemas = {
                        branch_name: tuple(sorted(schema.get_effective_guaranteed_fields()))
                        for branch_name, schema in branch_schema_configs.items()
                    }

                # Retrieve pre-computed output schema from DAG builder (P2 fix).
                # This ensures runtime contracts match build-time schema computation,
                # preserving nullable semantics from the P1 fix.
                coalesce_node_info = graph.get_node_info(coalesce_node_id)
                if coalesce_node_info.output_schema_config is None:
                    raise FrameworkBugError(
                        f"Coalesce node '{coalesce_node_id}' has no output_schema_config. "
                        f"The DAG builder must populate output_schema_config for all coalesce "
                        f"nodes via _assign_schema(). This indicates a builder bug."
                    )
                output_schema = create_contract_from_config(coalesce_node_info.output_schema_config)

                coalesce_executor.register_coalesce(
                    coalesce_settings_entry,
                    coalesce_node_id,
                    branch_schemas=branch_schemas,
                    output_schema=output_schema,
                )
            if restored_coalesce_state is not None:
                coalesce_executor.restore_from_checkpoint(restored_coalesce_state)

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

        # RowProcessor still carries a run-level source view for legacy helper
        # surfaces. Per-row processing passes the active source explicitly.
        first_source = next(iter(config.sources.values()))
        processor = RowProcessor(
            execution=factory.execution,
            data_flow=factory.data_flow,
            span_factory=self._span_factory,
            run_id=run_id,
            source_node_id=source_id,
            source_on_success=first_source.on_success,
            source_plugin=first_source,
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
            restored_coalesce_state=restored_coalesce_state,
            payload_store=payload_store,
            clock=self._clock,
            max_workers=self._concurrency_config.max_workers if self._concurrency_config else None,
            telemetry_manager=self._telemetry,
            scheduler=factory.scheduler,
        )

        return processor, coalesce_node_map, coalesce_executor

    def initialize_run_context(
        self,
        factory: RecorderFactory,
        run_id: str,
        config: PipelineConfig,
        graph: ExecutionGraph,
        settings: ElspethSettings | None,
        artifacts: GraphArtifacts,
        payload_store: PayloadStore,
        *,
        include_source_on_start: bool = True,
        restored_aggregation_state: Mapping[str, AggregationCheckpointState] | None = None,
        restored_coalesce_state: CoalesceCheckpointState | None = None,
        shutdown_event: threading.Event | None = None,
    ) -> RunContext:
        """Initialize run context: assign node IDs, create PluginContext, call on_start, build processor.

        Args:
            include_source_on_start: If True, call source.on_start(). False for resume
                (source was fully consumed in original run).
            restored_aggregation_state: Map of node_id -> state for resume path.
            restored_coalesce_state: Pending coalesce state for resume path.

        Returns:
            RunContext with ctx, processor, coalesce_executor, coalesce_node_map,
            and agg_transform_lookup.
        """
        source_id = artifacts.source_id
        sink_id_map = dict(artifacts.sink_id_map)
        transform_id_map = dict(artifacts.transform_id_map)
        config_gate_id_map = dict(artifacts.config_gate_id_map)
        coalesce_id_map = dict(artifacts.coalesce_id_map)
        edge_map = dict(artifacts.edge_map)
        route_resolution_map = graph.get_route_resolution_map()

        # Assign node_ids to all plugins
        assign_plugin_node_ids(
            sources=config.sources,
            transforms=config.transforms,
            sinks=config.sinks,
            source_id_map=artifacts.source_id_map,
            transform_id_map=transform_id_map,
            sink_id_map=sink_id_map,
        )

        # Create context with the PluginAuditWriter
        ctx = PluginContext(
            run_id=run_id,
            config=config.config,
            landscape=factory.plugin_audit_writer(),
            payload_store=factory.payload_store,
            rate_limit_registry=self._rate_limit_registry,
            concurrency_config=self._concurrency_config,
            telemetry_emit=self._ceremony.emit_telemetry,
            shutdown_event=shutdown_event,
        )

        # Set node_id on context for source validation error attribution
        # This must be set BEFORE source.load() so that any validation errors
        # (e.g., malformed CSV rows) can be attributed to the source node
        ctx.node_id = source_id

        try:
            if include_source_on_start:
                for source_name, source in config.sources.items():
                    ctx.node_id = artifacts.source_id_map[source_name]
                    source.on_start(ctx)
                ctx.node_id = source_id
            for transform in config.transforms:
                transform.on_start(ctx)
            for sink in config.sinks.values():
                sink.on_start(ctx)

            processor, coalesce_node_map, coalesce_executor = self.build_processor(
                graph=graph,
                config=config,
                settings=settings,
                factory=factory,
                run_id=run_id,
                source_id=source_id,
                edge_map=edge_map,
                route_resolution_map=route_resolution_map,
                config_gate_id_map=config_gate_id_map,
                coalesce_id_map=coalesce_id_map,
                payload_store=payload_store,
                restored_aggregation_state={NodeID(k): v for k, v in restored_aggregation_state.items()}
                if restored_aggregation_state
                else None,
                restored_coalesce_state=restored_coalesce_state,
            )
        except Exception:
            cleanup_plugins(config, ctx, include_source=include_source_on_start)
            raise

        # Pre-compute aggregation transform lookup for O(1) access per timeout check
        agg_transform_lookup: dict[str, AggNodeEntry] = {}
        if config.aggregation_settings:
            for t in config.transforms:
                if (
                    isinstance(t, TransformProtocol)
                    and t.is_batch_aware
                    and t.node_id is not None
                    and t.node_id in config.aggregation_settings
                ):
                    agg_transform_lookup[t.node_id] = AggNodeEntry(transform=t, node_id=NodeID(t.node_id))

        return RunContext(
            ctx=ctx,
            processor=processor,
            coalesce_executor=coalesce_executor,
            coalesce_node_map=coalesce_node_map,
            agg_transform_lookup=agg_transform_lookup,
        )

    def flush_and_write_sinks(
        self,
        factory: RecorderFactory,
        run_id: str,
        loop_ctx: LoopContext,
        sink_id_map: Mapping[SinkName, NodeID],
        edge_map: Mapping[tuple[NodeID, str], str],
        interrupted_by_shutdown: bool,
        *,
        on_token_written_factory: _CheckpointFactory | None = None,
    ) -> None:
        """Write all pending tokens to sinks and handle post-loop bookkeeping.

        IMPORTANT: Aggregation flush and coalesce flush are NOT in this method.
        They stay inside the processing loop because they must execute inside
        the track_operation(source_load) context to preserve audit attribution.

        Handles:
        1. Write pending tokens to sinks (each sink has its own track_operation)
        2. Raise GracefulShutdownError if interrupted
        """
        counters = loop_ctx.counters

        diversion_counts = self.write_pending_to_sinks(
            factory=factory,
            run_id=run_id,
            config=loop_ctx.config,
            ctx=loop_ctx.ctx,
            counters=loop_ctx.counters,
            pending_tokens=loop_ctx.pending_tokens,
            sink_id_map=dict(sink_id_map),
            edge_map=edge_map,
            sink_step=loop_ctx.processor.resolve_sink_step(),
            on_token_written_factory=on_token_written_factory,
        )
        # ADR-019: failsink-mode diversions are TRANSIENT structural evidence;
        # discard-mode diversions are FAILURE predicate inputs as well.
        loop_ctx.counters.rows_diverted += diversion_counts.total
        loop_ctx.counters.rows_failed += diversion_counts.discard_mode

        # If shutdown interrupted the loop, raise after all pending work is flushed.
        # At this point: sink writes are done, and any buffered aggregation/coalesce
        # state that we intentionally preserved can be checkpointed for resume.
        if interrupted_by_shutdown:
            self._checkpoints.checkpoint_interrupted_progress(
                run_id=run_id,
                loop_ctx=loop_ctx,
            )
            raise GracefulShutdownError(
                rows_processed=counters.rows_processed,
                run_id=run_id,
                rows_succeeded=counters.rows_succeeded,
                rows_failed=counters.rows_failed,
                rows_quarantined=counters.rows_quarantined,
                rows_routed_success=counters.rows_routed_success,
                rows_routed_failure=counters.rows_routed_failure,
                routed_destinations=dict(counters.routed_destinations),
            )
