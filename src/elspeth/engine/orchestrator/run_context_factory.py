"""RunContextFactory: per-run PluginContext construction and lifecycle startup.

Split out of the old ``RunExecutionCore`` by lifecycle boundary
(elspeth-b53a093321): this module owns the construction of the per-run
:class:`RunContext` — node-id assignment, PluginContext creation, the
``on_start`` plugin lifecycle (with cleanup on failure), and processor
assembly via the injected :class:`ProcessorFactory`. Shared verbatim by the
main run path (``LeaderDrainCoordinator.execute_run``) and the resume path
(``ResumeCoordinator.process_resumed_rows``).

Dependencies held by the factory:
- ``_ceremony``: RunCeremony for telemetry emission into the PluginContext
- ``_rate_limit_registry``: RateLimitRegistry for the PluginContext
- ``_concurrency_config``: RuntimeConcurrencyConfig for the PluginContext
- ``_processor_factory``: ProcessorFactory for traversal/coalesce assembly
"""

from __future__ import annotations

import threading
from typing import TYPE_CHECKING

from elspeth.contracts import (
    TransformProtocol,
)
from elspeth.contracts.plugin_context import PluginContext
from elspeth.contracts.sink_effects import SinkEffectInputKind
from elspeth.contracts.types import NodeID
from elspeth.engine.orchestrator.cleanup import cleanup_plugins, plugin_node_scope
from elspeth.engine.orchestrator.graph_wiring import assign_plugin_node_ids
from elspeth.engine.orchestrator.preflight import validate_pipeline_sink_effect_capabilities
from elspeth.engine.orchestrator.run_state import (
    AggNodeEntry,
    RunContext,
)

if TYPE_CHECKING:
    from elspeth.contracts.config.runtime import RuntimeConcurrencyConfig
    from elspeth.contracts.coordination import CoordinationToken
    from elspeth.contracts.payload_store import PayloadStore
    from elspeth.contracts.plugin_protocols import SinkProtocol, SourceProtocol
    from elspeth.core.config import ElspethSettings
    from elspeth.core.dag import ExecutionGraph
    from elspeth.core.landscape.factory import RecorderFactory
    from elspeth.core.rate_limit import RateLimitRegistry
    from elspeth.engine.barrier_coordination import BarrierJournalRestoreContext
    from elspeth.engine.orchestrator.ceremony import RunCeremony
    from elspeth.engine.orchestrator.processor_factory import ProcessorFactory
    from elspeth.engine.orchestrator.run_state import (
        GraphArtifacts,
    )
    from elspeth.engine.orchestrator.types import (
        PipelineConfig,
    )


class RunContextFactory:
    """Builds the per-run RunContext: node ids, PluginContext, on_start, processor.

    Owns exactly the run-context construction lifecycle — everything that
    must happen between graph registration and the first processed row. Sink
    writing lives in :class:`~elspeth.engine.orchestrator.sink_flush.SinkFlushCoordinator`;
    processor assembly detail lives in the injected
    :class:`~elspeth.engine.orchestrator.processor_factory.ProcessorFactory`.
    """

    def __init__(
        self,
        *,
        ceremony: RunCeremony,
        rate_limit_registry: RateLimitRegistry | None,
        concurrency_config: RuntimeConcurrencyConfig | None,
        processor_factory: ProcessorFactory,
    ) -> None:
        self._ceremony = ceremony
        self._rate_limit_registry = rate_limit_registry
        self._concurrency_config = concurrency_config
        self._processor_factory = processor_factory

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
        barrier_restore: BarrierJournalRestoreContext | None = None,
        shutdown_event: threading.Event | None = None,
        coordination_token: CoordinationToken | None = None,
    ) -> RunContext:
        """Initialize run context: assign node IDs, create PluginContext, call on_start, build processor.

        Args:
            include_source_on_start: If True, call source.on_start(). False for resume
                (source was fully consumed in original run).
            barrier_restore: Resume-only journal-restore inputs (F1); None on
                the normal run path.
            coordination_token: Leader fencing token (ADR-030). Threaded into
                the RowProcessor so its worker identity doubles as the
                scheduler lease_owner and so the slice-2 step-4 fenced verbs
                (repair sweep, ingest) can present it.

        Returns:
            RunContext with ctx, processor, coalesce_executor, coalesce_node_map,
            and agg_transform_lookup.
        """
        # Effect capability is a local/declarative gate over already-resolved
        # sink instances. It must precede node assignment, restricted-context
        # construction, every plugin on_start(), reservation, inspection,
        # sink client initialization, and target I/O on both fresh and resume
        # paths.
        validate_pipeline_sink_effect_capabilities(
            config.sinks,
            required_input_kind=SinkEffectInputKind.PIPELINE_MEMBERS,
        )

        source_id = artifacts.source_id
        sink_id_map = dict(artifacts.sink_id_map)
        transform_id_map = dict(artifacts.transform_id_map)
        config_gate_id_map = dict(artifacts.config_gate_id_map)
        coalesce_id_map = dict(artifacts.coalesce_id_map)
        edge_map = dict(artifacts.edge_map)
        route_resolution_map = graph.get_route_resolution_map()
        aggregation_node_ids = frozenset(graph.get_aggregation_id_map().values())

        # Assign node_ids to all plugins
        assign_plugin_node_ids(
            sources=config.sources,
            transforms=config.transforms,
            sinks=config.sinks,
            source_id_map=artifacts.source_id_map,
            transform_id_map=transform_id_map,
            sink_id_map=sink_id_map,
            aggregation_node_ids=aggregation_node_ids,
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

        started_sources: dict[str, SourceProtocol] = {}
        started_transforms: list[TransformProtocol] = []
        started_sinks: dict[str, SinkProtocol] = {}
        try:
            if include_source_on_start:
                for source_name, source in config.sources.items():
                    with plugin_node_scope(ctx, source.node_id):
                        source.on_start(ctx)
                    started_sources[source_name] = source
                ctx.node_id = source_id
            for transform in config.transforms:
                with plugin_node_scope(ctx, transform.node_id):
                    transform.on_start(ctx)
                started_transforms.append(transform)
            for sink_name, sink in config.sinks.items():
                with plugin_node_scope(ctx, sink.node_id):
                    sink.on_start(ctx)
                started_sinks[sink_name] = sink

            processor, coalesce_node_map, coalesce_executor = self._processor_factory.build_processor(
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
                barrier_restore=barrier_restore,
                coordination_token=coordination_token,
            )
        except Exception:
            cleanup_plugins(
                config,
                ctx,
                include_source=include_source_on_start,
                started_sources=started_sources,
                started_transforms=tuple(started_transforms),
                started_sinks=started_sinks,
            )
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
