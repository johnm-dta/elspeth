"""GraphRegistrationService: GRAPH-phase node/edge registration + validation.

Extracted from ``Orchestrator._register_graph_nodes_and_edges`` and
``Orchestrator._record_declared_sources_ready`` (filigree elspeth-9e71ae82a4).
The facade keeps a thin ``_register_graph_nodes_and_edges`` delegator so tests
that stub the method on the orchestrator instance keep working.

Behaviour-preserving: the registration order (nodes -> declared sources ->
edges -> route/error-sink/quarantine/failsink validation), the
PhaseStarted/PhaseChanged/PhaseCompleted event ceremony, and the
emit_phase_error + re-raise failure arm are unchanged.
"""

from __future__ import annotations

import json
import time
from collections.abc import Mapping
from datetime import UTC, datetime
from types import SimpleNamespace
from typing import TYPE_CHECKING

from elspeth.contracts.events import (
    PhaseAction,
    PhaseChanged,
    PhaseCompleted,
    PhaseStarted,
    PipelinePhase,
)
from elspeth.contracts.types import (
    AggregationName,
    CoalesceName,
    GateName,
    NodeID,
    SinkName,
)
from elspeth.core.canonical import stable_hash
from elspeth.engine.orchestrator.graph_wiring import build_source_id_map
from elspeth.engine.orchestrator.landscape_registration import (
    register_nodes_with_landscape,
    resolve_node_audit_metadata,
)
from elspeth.engine.orchestrator.types import GraphArtifacts
from elspeth.engine.orchestrator.validation import (
    validate_route_destinations,
    validate_sink_failsink_destinations,
    validate_source_quarantine_destination,
    validate_transform_error_sinks,
)

if TYPE_CHECKING:
    from elspeth.core.dag import ExecutionGraph
    from elspeth.core.events import EventBusProtocol
    from elspeth.core.landscape.factory import RecorderFactory
    from elspeth.engine.orchestrator.ceremony import RunCeremony
    from elspeth.engine.orchestrator.types import PipelineConfig


class GraphRegistrationService:
    """Owns the GRAPH phase: registering nodes/edges and validating routing."""

    def __init__(self, *, events: EventBusProtocol, ceremony: RunCeremony) -> None:
        self._events = events
        self._ceremony = ceremony

    def register_graph_nodes_and_edges(
        self,
        factory: RecorderFactory,
        run_id: str,
        config: PipelineConfig,
        graph: ExecutionGraph,
    ) -> GraphArtifacts:
        """Register all graph nodes and edges in Landscape. Returns artifacts for subsequent phases.

        Performs the GRAPH phase:
        1. Build node_to_plugin mapping from config
        2. Register each node with Landscape (metadata, determinism, schema)
        3. Register edges and build edge_map
        4. Validate route destinations, error sinks, quarantine destinations

        Args:
            factory: RecorderFactory for audit trail
            run_id: Run identifier
            config: Pipeline configuration
            graph: Execution graph

        Returns:
            GraphArtifacts with edge_map, source_id, and all ID mappings
        """

        # Get execution order from graph
        execution_order = graph.topological_order()

        # Build source-name -> node-id mapping via the SAME loader resume and
        # follower use (elspeth-07b2031e41); it fails closed on a source node
        # missing ADR-025 §2 source_name (which would silently collide
        # entries across multiple sources).
        source_id_map = build_source_id_map(graph)
        source_id = next(iter(source_id_map.values()))
        transform_id_map: dict[int, NodeID] = graph.get_transform_id_map()
        sink_id_map: dict[SinkName, NodeID] = graph.get_sink_id_map()
        config_gate_id_map: dict[GateName, NodeID] = graph.get_config_gate_id_map()
        aggregation_id_map: dict[AggregationName, NodeID] = graph.get_aggregation_id_map()

        # Build node ID sets for special node types
        config_gate_node_ids: set[NodeID] = set(config_gate_id_map.values())
        aggregation_node_ids: set[NodeID] = set(aggregation_id_map.values())

        coalesce_id_map: dict[CoalesceName, NodeID] = graph.get_coalesce_id_map()
        coalesce_node_ids: set[NodeID] = set(coalesce_id_map.values())
        audit_metadata_by_node = resolve_node_audit_metadata(
            config,
            graph,
            source_id_map=source_id_map,
            transform_id_map=transform_id_map,
            sink_id_map=sink_id_map,
            config_gate_node_ids=config_gate_node_ids,
            aggregation_node_ids=aggregation_node_ids,
            coalesce_node_ids=coalesce_node_ids,
        )

        # GRAPH phase - register nodes and edges in Landscape
        phase_start = time.perf_counter()
        try:
            self._events.emit(PhaseStarted(phase=PipelinePhase.GRAPH, action=PhaseAction.BUILDING))

            # Emit telemetry PhaseChanged - we now have run_id from begin_run
            self._ceremony.emit_telemetry(
                PhaseChanged(
                    timestamp=datetime.now(UTC),
                    run_id=run_id,
                    phase=PipelinePhase.GRAPH,
                    action=PhaseAction.BUILDING,
                )
            )

            # Register nodes with Landscape using graph's node IDs and actual plugin metadata
            register_nodes_with_landscape(
                factory,
                run_id,
                config,
                graph,
                execution_order,
                audit_metadata_by_node,
            )
            self._record_declared_sources_ready(
                factory=factory,
                run_id=run_id,
                config=config,
                source_id_map=source_id_map,
            )

            # Register edges from graph - key by (from_node, label) for lookup
            # Gates return route labels, so edge_map is keyed by label
            edge_map: dict[tuple[NodeID, str], str] = {}

            for edge_info in graph.get_edges():
                edge = factory.data_flow.register_edge(
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

            # NOTE — value-source compliance is enforced at the entry-point
            # boundary, NOT here. The walker
            # (``engine/orchestrator/preflight.validate_value_source_compliance``)
            # runs inside ``runtime_factory.instantiate_plugins_from_config`` and
            # the composer/web-execution validate paths
            # (``web/execution/validation.validate_pipeline``,
            # ``web/execution/service._run_pipeline``). Every legitimate caller
            # that builds a ``PipelineConfig`` passes through one of those
            # surfaces, so by the time we reach ``Orchestrator.run`` the bundle
            # has already been gated. If you add a new entry point that
            # constructs a ``PipelineConfig`` directly (test harness,
            # programmatic API, etc.), call ``validate_value_source_compliance``
            # at that boundary too — the orchestrator does NOT re-validate
            # value-source declarations per run, and a bypassing entry point
            # would silently skip the check otherwise.
            #
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

            self._events.emit(PhaseCompleted(phase=PipelinePhase.GRAPH, duration_seconds=time.perf_counter() - phase_start))
        except Exception as e:
            self._ceremony.emit_phase_error(PipelinePhase.GRAPH, e)
            raise  # CRITICAL: Always re-raise - graph validation failure is fatal

        return GraphArtifacts(
            edge_map=edge_map,
            source_id=source_id,
            source_id_map=source_id_map,
            sink_id_map=sink_id_map,
            transform_id_map=transform_id_map,
            config_gate_id_map=config_gate_id_map,
            coalesce_id_map=coalesce_id_map,
        )

    def _record_declared_sources_ready(
        self,
        *,
        factory: RecorderFactory,
        run_id: str,
        config: PipelineConfig,
        source_id_map: Mapping[str, NodeID],
    ) -> None:
        """Seed run_sources for every declared source before iteration starts.

        A hard kill between source lifecycles must leave audit evidence that
        later sources were declared but never exhausted. The source-specific
        loop updates the active source from ready -> loading -> exhausted ->
        loaded, or interrupted; unstarted later sources remain ready and resume
        refuses rather than fabricating source exhaustion.
        """
        for source_name, source_node_id in source_id_map.items():
            source = config.sources[source_name]
            factory.run_lifecycle.record_run_source(
                run_id=run_id,
                source_node_id=source_node_id,
                source_name=source_name,
                plugin_name=source.name,
                config_hash=stable_hash(source.config),
                source_schema_json=json.dumps(source.output_schema.model_json_schema()),
                schema_contract=source.get_schema_contract(),
                lifecycle_state="ready",
            )
