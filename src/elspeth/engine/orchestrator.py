# src/elspeth/engine/orchestrator.py
"""Orchestrator: Full run lifecycle management.

Coordinates:
- Run initialization
- Source loading
- Row processing
- Sink writing
- Run completion
- Post-run audit export (when configured)
"""

import os
from contextlib import suppress
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from elspeth.core.dag import ExecutionGraph
from elspeth.core.landscape import LandscapeDB, LandscapeRecorder
from elspeth.engine.processor import RowProcessor
from elspeth.engine.spans import SpanFactory
from elspeth.plugins.context import PluginContext
from elspeth.plugins.enums import NodeType

if TYPE_CHECKING:
    from elspeth.core.config import ElspethSettings


@dataclass
class PipelineConfig:
    """Configuration for a pipeline run."""

    source: Any  # SourceProtocol
    transforms: list[Any]  # List of transform/gate plugins
    sinks: dict[str, Any]  # sink_name -> SinkProtocol
    config: dict[str, Any] = field(default_factory=dict)


@dataclass
class RunResult:
    """Result of a pipeline run."""

    run_id: str
    status: str  # completed, failed
    rows_processed: int
    rows_succeeded: int
    rows_failed: int
    rows_routed: int


class Orchestrator:
    """Orchestrates full pipeline runs.

    Manages the complete lifecycle:
    1. Begin run in Landscape
    2. Register all nodes (and set node_id on each plugin instance)
    3. Load rows from source
    4. Process rows through transforms
    5. Write to sinks
    6. Complete run

    NOTE on node_id: Plugin protocols don't define node_id as an attribute.
    The Orchestrator sets node_id on each plugin instance AFTER registering
    it with Landscape:

        node = recorder.register_node(...)
        transform.node_id = node.node_id  # Set by Orchestrator

    This allows executors to access node_id without requiring plugins
    to know their node_id at construction time.
    """

    def __init__(
        self,
        db: LandscapeDB,
        *,
        canonical_version: str = "sha256-rfc8785-v1",
    ) -> None:
        self._db = db
        self._canonical_version = canonical_version
        self._span_factory = SpanFactory()

    def run(
        self,
        config: PipelineConfig,
        graph: ExecutionGraph | None = None,
        settings: "ElspethSettings | None" = None,
    ) -> RunResult:
        """Execute a pipeline run.

        Args:
            config: Pipeline configuration with plugins
            graph: Pre-validated execution graph (required)
            settings: Full settings (for post-run hooks like export)

        Raises:
            ValueError: If graph is not provided
        """
        if graph is None:
            raise ValueError(
                "ExecutionGraph is required. "
                "Build with ExecutionGraph.from_config(settings)"
            )

        recorder = LandscapeRecorder(self._db)

        # Begin run
        run = recorder.begin_run(
            config=config.config,
            canonical_version=self._canonical_version,
        )

        try:
            with self._span_factory.run_span(run.run_id):
                result = self._execute_run(recorder, run.run_id, config, graph)

            # Complete run
            recorder.complete_run(run.run_id, status="completed")
            result.status = "completed"

            # Post-run export
            if settings is not None and settings.landscape.export.enabled:
                self._export_landscape(
                    run_id=run.run_id,
                    settings=settings,
                    sinks=config.sinks,
                )

            return result

        except Exception:
            recorder.complete_run(run.run_id, status="failed")
            raise

    def _execute_run(
        self,
        recorder: LandscapeRecorder,
        run_id: str,
        config: PipelineConfig,
        graph: ExecutionGraph,
    ) -> RunResult:
        """Execute the run using the execution graph.

        The graph provides:
        - Node IDs and metadata via topological_order() and get_node_info()
        - Edges via get_edges()
        - Explicit ID mappings via get_sink_id_map() and get_transform_id_map()
        """

        # Get execution order from graph
        execution_order = graph.topological_order()

        # Register nodes with Landscape using graph's node IDs
        for node_id in execution_order:
            node_info = graph.get_node_info(node_id)
            recorder.register_node(
                run_id=run_id,
                node_id=node_id,  # Use graph's ID
                plugin_name=node_info.plugin_name,
                node_type=NodeType(node_info.node_type),  # Already lowercase
                plugin_version="1.0.0",
                config=node_info.config,
            )

        # Register edges from graph - key by (from_node, label) for lookup
        # Gates return route labels, so edge_map is keyed by label
        edge_map: dict[tuple[str, str], str] = {}

        for from_id, to_id, edge_data in graph.get_edges():
            edge = recorder.register_edge(
                run_id=run_id,
                from_node_id=from_id,
                to_node_id=to_id,
                label=edge_data["label"],
                mode=edge_data["mode"],
            )
            # Key by edge label - gates return route labels, transforms use "continue"
            edge_map[(from_id, edge_data["label"])] = edge.edge_id

        # Get route resolution map - maps (gate_node, label) -> "continue" | sink_name
        route_resolution_map = graph.get_route_resolution_map()

        # Get explicit node ID mappings from graph
        source_id = graph.get_source()
        if source_id is None:
            raise ValueError("Graph has no source node")
        sink_id_map = graph.get_sink_id_map()
        transform_id_map = graph.get_transform_id_map()
        output_sink_name = graph.get_output_sink()

        # Set node_id on source plugin
        config.source.node_id = source_id

        # Set node_id on transforms using graph's transform_id_map
        for seq, transform in enumerate(config.transforms):
            if seq not in transform_id_map:
                raise ValueError(
                    f"Transform at sequence {seq} not found in graph. "
                    f"Graph has mappings for sequences: {list(transform_id_map.keys())}"
                )
            transform.node_id = transform_id_map[seq]

        # Set node_id on sinks using explicit mapping
        for sink_name, sink in config.sinks.items():
            if sink_name not in sink_id_map:
                raise ValueError(
                    f"Sink '{sink_name}' not found in graph. "
                    f"Available sinks: {list(sink_id_map.keys())}"
                )
            sink.node_id = sink_id_map[sink_name]

        # Create context
        # Note: landscape field uses Any type since PluginContext.LandscapeRecorder
        # is a protocol placeholder for Phase 2, while we pass the real recorder
        ctx = PluginContext(
            run_id=run_id,
            config=config.config,
            landscape=recorder,  # type: ignore[arg-type]
        )

        # Call on_start for all plugins BEFORE processing
        # Lifecycle hooks are optional - plugins may or may not implement them
        for transform in config.transforms:
            if hasattr(transform, "on_start"):
                transform.on_start(ctx)

        # Create processor
        processor = RowProcessor(
            recorder=recorder,
            span_factory=self._span_factory,
            run_id=run_id,
            source_node_id=source_id,
            edge_map=edge_map,
            route_resolution_map=route_resolution_map,
        )

        # Process rows - Buffer TOKENS, not dicts, to preserve identity
        from elspeth.engine.executors import SinkExecutor
        from elspeth.engine.tokens import TokenInfo

        rows_processed = 0
        rows_succeeded = 0
        rows_failed = 0
        rows_routed = 0
        pending_tokens: dict[str, list[TokenInfo]] = {name: [] for name in config.sinks}

        try:
            with self._span_factory.source_span(config.source.name):
                for row_index, row_data in enumerate(config.source.load(ctx)):
                    rows_processed += 1

                    result = processor.process_row(
                        row_index=row_index,
                        row_data=row_data,
                        transforms=config.transforms,
                        ctx=ctx,
                    )

                    if result.outcome == "completed":
                        rows_succeeded += 1
                        pending_tokens[output_sink_name].append(result.token)
                    elif result.outcome == "routed":
                        rows_routed += 1
                        if result.sink_name and result.sink_name in config.sinks:
                            pending_tokens[result.sink_name].append(result.token)
                        elif result.sink_name:
                            # Gate routed to non-existent sink - configuration error
                            raise ValueError(
                                f"Gate routed to unknown sink '{result.sink_name}'. "
                                f"Available sinks: {list(config.sinks.keys())}"
                            )
                        else:
                            # sink_name is None but outcome is "routed" - should not happen
                            raise RuntimeError(
                                f"Row outcome is 'routed' but sink_name is None "
                                f"for token {result.token.token_id}"
                            )
                    elif result.outcome == "failed":
                        rows_failed += 1

            # Write to sinks using SinkExecutor
            sink_executor = SinkExecutor(recorder, self._span_factory, run_id)
            step = len(config.transforms) + 1

            for sink_name, tokens in pending_tokens.items():
                if tokens and sink_name in config.sinks:
                    sink = config.sinks[sink_name]
                    sink_executor.write(
                        sink=sink,
                        tokens=tokens,
                        ctx=ctx,
                        step_in_pipeline=step,
                    )

        finally:
            # Call on_complete for all plugins (even on error)
            # Lifecycle hooks are optional - plugins may or may not implement them
            # suppress(Exception) ensures one plugin failure doesn't prevent others from cleanup
            for transform in config.transforms:
                if hasattr(transform, "on_complete"):
                    with suppress(Exception):
                        transform.on_complete(ctx)

            # Close source and all sinks
            # SinkProtocol requires close() - if missing, that's a bug
            config.source.close()
            for sink in config.sinks.values():
                sink.close()

        return RunResult(
            run_id=run_id,
            status="running",  # Will be updated
            rows_processed=rows_processed,
            rows_succeeded=rows_succeeded,
            rows_failed=rows_failed,
            rows_routed=rows_routed,
        )

    def _export_landscape(
        self,
        run_id: str,
        settings: "ElspethSettings",
        sinks: dict[str, Any],
    ) -> None:
        """Export audit trail to configured sink after run completion.

        Args:
            run_id: The completed run ID
            settings: Full settings containing export configuration
            sinks: Dict of sink_name -> sink instance from PipelineConfig

        Raises:
            ValueError: If signing requested but ELSPETH_SIGNING_KEY not set,
                       or if configured sink not found
        """
        from elspeth.core.landscape.exporter import LandscapeExporter

        export_config = settings.landscape.export

        # Get signing key from environment if signing enabled
        signing_key: bytes | None = None
        if export_config.sign:
            key_str = os.environ.get("ELSPETH_SIGNING_KEY")
            if not key_str:
                raise ValueError(
                    "ELSPETH_SIGNING_KEY environment variable required for signed export"
                )
            signing_key = key_str.encode("utf-8")

        # Create exporter
        exporter = LandscapeExporter(self._db, signing_key=signing_key)

        # Get target sink
        sink_name = export_config.sink
        if sink_name not in sinks:
            raise ValueError(f"Export sink '{sink_name}' not found in sinks")
        sink = sinks[sink_name]

        # Create context for sink writes
        ctx = PluginContext(run_id=run_id, config={}, landscape=None)

        # Export records to sink - write as list to match sink.write() signature
        records = list(exporter.export_run(run_id, sign=export_config.sign))
        sink.write(records, ctx)

        # Flush sink to ensure all records are written
        sink.flush()
