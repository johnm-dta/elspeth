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
from elspeth.core.landscape.models import RunStatus
from elspeth.engine.processor import RowProcessor
from elspeth.engine.spans import SpanFactory
from elspeth.plugins.base import BaseAggregation, BaseGate, BaseTransform
from elspeth.plugins.context import PluginContext
from elspeth.plugins.enums import NodeType
from elspeth.plugins.protocols import SinkProtocol, SourceProtocol
from elspeth.plugins.results import RowOutcome

# Type alias for transform-like plugins
TransformLike = BaseTransform | BaseGate | BaseAggregation

if TYPE_CHECKING:
    from elspeth.core.checkpoint import CheckpointManager
    from elspeth.core.config import CheckpointSettings, ElspethSettings


@dataclass
class PipelineConfig:
    """Configuration for a pipeline run.

    All plugin fields are now properly typed for IDE support and
    static type checking.
    """

    source: SourceProtocol
    transforms: list[TransformLike]
    sinks: dict[str, SinkProtocol]
    config: dict[str, Any] = field(default_factory=dict)


@dataclass
class RunResult:
    """Result of a pipeline run."""

    run_id: str
    status: RunStatus
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
        checkpoint_manager: "CheckpointManager | None" = None,
        checkpoint_settings: "CheckpointSettings | None" = None,
    ) -> None:
        self._db = db
        self._canonical_version = canonical_version
        self._span_factory = SpanFactory()
        self._checkpoint_manager = checkpoint_manager
        self._checkpoint_settings = checkpoint_settings
        self._sequence_number = 0  # Monotonic counter for checkpoint ordering

    def _maybe_checkpoint(self, run_id: str, token_id: str, node_id: str) -> None:
        """Create checkpoint if configured.

        Called after each row is processed. Creates a checkpoint based on
        the configured frequency:
        - every_row: Always checkpoint
        - every_n: Checkpoint at intervals (e.g., every 10 rows)
        - aggregation_only: No per-row checkpoints (handled separately)

        Args:
            run_id: Current run ID
            token_id: Token that was just processed
            node_id: Node that processed it (last in chain before sink)
        """
        if not self._checkpoint_settings or not self._checkpoint_settings.enabled:
            return
        if self._checkpoint_manager is None:
            return

        self._sequence_number += 1

        should_checkpoint = False
        if self._checkpoint_settings.frequency == "every_row":
            should_checkpoint = True
        elif self._checkpoint_settings.frequency == "every_n":
            interval = self._checkpoint_settings.checkpoint_interval
            # interval is validated in CheckpointSettings when frequency="every_n"
            assert interval is not None  # Validated by CheckpointSettings model
            should_checkpoint = (self._sequence_number % interval) == 0
        # aggregation_only: checkpointed separately in aggregation flush

        if should_checkpoint:
            self._checkpoint_manager.create_checkpoint(
                run_id=run_id,
                token_id=token_id,
                node_id=node_id,
                sequence_number=self._sequence_number,
            )

    def _delete_checkpoints(self, run_id: str) -> None:
        """Delete all checkpoints for a run after successful completion.

        Args:
            run_id: Run to clean up checkpoints for
        """
        if self._checkpoint_manager is not None:
            self._checkpoint_manager.delete_checkpoints(run_id)

    def _cleanup_transforms(self, config: PipelineConfig) -> None:
        """Call close() on all transforms and gates.

        Called in finally block to ensure cleanup happens even on failure.
        Logs but doesn't raise if individual cleanup fails.

        The hasattr check is acceptable here because old plugins might not
        have close() yet (graceful degradation at plugin trust boundary).
        """
        import structlog

        logger = structlog.get_logger()

        for transform in config.transforms:
            try:
                if hasattr(transform, "close"):
                    transform.close()
            except Exception as e:
                # Log but don't raise - cleanup should be best-effort
                logger.warning(
                    "Transform cleanup failed",
                    transform=getattr(transform, "name", str(transform)),
                    error=str(e),
                )

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

        run_completed = False
        try:
            with self._span_factory.run_span(run.run_id):
                result = self._execute_run(recorder, run.run_id, config, graph)

            # Complete run
            recorder.complete_run(run.run_id, status="completed")
            result.status = RunStatus.COMPLETED
            run_completed = True

            # Delete checkpoints on successful completion
            # (checkpoints are for recovery, not needed after success)
            self._delete_checkpoints(run.run_id)

            # Post-run export (separate from run status - export failures
            # don't change run status)
            if settings is not None and settings.landscape.export.enabled:
                export_config = settings.landscape.export
                recorder.set_export_status(
                    run.run_id,
                    status="pending",
                    export_format=export_config.format,
                    export_sink=export_config.sink,
                )
                try:
                    self._export_landscape(
                        run_id=run.run_id,
                        settings=settings,
                        sinks=config.sinks,
                    )
                    recorder.set_export_status(run.run_id, status="completed")
                except Exception as export_error:
                    recorder.set_export_status(
                        run.run_id,
                        status="failed",
                        error=str(export_error),
                    )
                    # Re-raise so caller knows export failed
                    # (run is still "completed" in Landscape)
                    raise

            return result

        except Exception:
            # Only mark run as failed if it didn't complete successfully
            # (export failures are tracked separately)
            if not run_completed:
                recorder.complete_run(run.run_id, status="failed")
            raise
        finally:
            # Always clean up transforms, even on failure
            self._cleanup_transforms(config)

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
        from elspeth.plugins.enums import Determinism

        # Get execution order from graph
        execution_order = graph.topological_order()

        # Build node_id -> plugin instance mapping for metadata extraction
        # Source: single plugin from config.source
        source_id = graph.get_source()
        transform_id_map = graph.get_transform_id_map()
        sink_id_map = graph.get_sink_id_map()

        node_to_plugin: dict[str, Any] = {}
        if source_id is not None:
            node_to_plugin[source_id] = config.source
        for seq, transform in enumerate(config.transforms):
            if seq in transform_id_map:
                node_to_plugin[transform_id_map[seq]] = transform
        for sink_name, sink in config.sinks.items():
            if sink_name in sink_id_map:
                node_to_plugin[sink_id_map[sink_name]] = sink

        # Register nodes with Landscape using graph's node IDs and actual plugin metadata
        for node_id in execution_order:
            node_info = graph.get_node_info(node_id)

            # Get plugin metadata from actual plugin instance (with defaults)
            # Uses getattr + isinstance to safely extract attributes from duck-typed plugins
            plugin = node_to_plugin.get(node_id)
            plugin_version = "0.0.0"
            determinism: Determinism | str = Determinism.DETERMINISTIC

            if plugin is not None:
                # Extract plugin_version if defined and valid
                raw_version = getattr(plugin, "plugin_version", None)
                if isinstance(raw_version, str):
                    plugin_version = raw_version

                # Extract determinism if defined and valid (Determinism enum or string)
                raw_determinism = getattr(plugin, "determinism", None)
                if isinstance(raw_determinism, Determinism | str):
                    determinism = raw_determinism

            recorder.register_node(
                run_id=run_id,
                node_id=node_id,  # Use graph's ID
                plugin_name=node_info.plugin_name,
                node_type=NodeType(node_info.node_type),  # Already lowercase
                plugin_version=plugin_version,
                config=node_info.config,
                determinism=determinism,
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
        # NOTE: node_id is set dynamically by Orchestrator, not defined in protocols
        # See class docstring for rationale
        config.source.node_id = source_id  # type: ignore[attr-defined]

        # Set node_id on transforms using graph's transform_id_map
        for seq, transform in enumerate(config.transforms):
            if seq not in transform_id_map:
                raise ValueError(
                    f"Transform at sequence {seq} not found in graph. "
                    f"Graph has mappings for sequences: {list(transform_id_map.keys())}"
                )
            transform.node_id = transform_id_map[seq]  # type: ignore[union-attr]

        # Set node_id on sinks using explicit mapping
        for sink_name, sink in config.sinks.items():
            if sink_name not in sink_id_map:
                raise ValueError(
                    f"Sink '{sink_name}' not found in graph. "
                    f"Available sinks: {list(sink_id_map.keys())}"
                )
            sink.node_id = sink_id_map[sink_name]  # type: ignore[attr-defined]

        # Create context
        # Note: landscape field uses Any type since PluginContext.LandscapeRecorder
        # is a protocol placeholder for Phase 2, while we pass the real recorder
        ctx = PluginContext(
            run_id=run_id,
            config=config.config,
            landscape=recorder,  # type: ignore[arg-type]  # TODO: align PluginContext.landscape type with LandscapeRecorder
        )

        # Call on_start for all plugins BEFORE processing
        # Base classes provide no-op implementations, so no hasattr needed
        config.source.on_start(ctx)
        for transform in config.transforms:
            transform.on_start(ctx)
        for sink in config.sinks.values():
            sink.on_start(ctx)

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

                    # Determine the last node that processed this row
                    # (used for checkpoint to know where to resume from)
                    last_node_id = (
                        config.transforms[-1].node_id  # type: ignore[union-attr]
                        if config.transforms
                        else source_id
                    )

                    if result.outcome == RowOutcome.COMPLETED:
                        rows_succeeded += 1
                        pending_tokens[output_sink_name].append(result.token)
                        # Checkpoint after successful row processing
                        self._maybe_checkpoint(
                            run_id=run_id,
                            token_id=result.token.token_id,
                            node_id=last_node_id,
                        )
                    elif result.outcome == RowOutcome.ROUTED:
                        rows_routed += 1
                        if result.sink_name and result.sink_name in config.sinks:
                            pending_tokens[result.sink_name].append(result.token)
                            # Checkpoint after successful routing
                            self._maybe_checkpoint(
                                run_id=run_id,
                                token_id=result.token.token_id,
                                node_id=last_node_id,
                            )
                        elif result.sink_name:
                            # Gate routed to non-existent sink - configuration error
                            raise ValueError(
                                f"Gate routed to unknown sink '{result.sink_name}'. "
                                f"Available sinks: {list(config.sinks.keys())}"
                            )
                        else:
                            # sink_name is None but outcome is ROUTED - should not happen
                            raise RuntimeError(
                                f"Row outcome is ROUTED but sink_name is None "
                                f"for token {result.token.token_id}"
                            )
                    elif result.outcome == RowOutcome.FAILED:
                        rows_failed += 1

            # Write to sinks using SinkExecutor
            sink_executor = SinkExecutor(recorder, self._span_factory, run_id)
            step = len(config.transforms) + 1

            for sink_name, tokens in pending_tokens.items():
                if tokens and sink_name in config.sinks:
                    sink = config.sinks[sink_name]
                    # NOTE: PipelineConfig.sinks is typed as SinkProtocol for API clarity,
                    # but at runtime receives SinkAdapter instances (SinkLike interface)
                    sink_executor.write(
                        sink=sink,  # type: ignore[arg-type]
                        tokens=tokens,
                        ctx=ctx,
                        step_in_pipeline=step,
                    )

        finally:
            # Call on_complete for all plugins (even on error)
            # Base classes provide no-op implementations, so no hasattr needed
            # suppress(Exception) ensures one plugin failure doesn't prevent others from cleanup
            for transform in config.transforms:
                with suppress(Exception):
                    transform.on_complete(ctx)
            for sink in config.sinks.values():
                with suppress(Exception):
                    sink.on_complete(ctx)
            with suppress(Exception):
                config.source.on_complete(ctx)

            # Close source and all sinks
            # SinkProtocol requires close() - if missing, that's a bug
            config.source.close()
            for sink in config.sinks.values():
                sink.close()

        return RunResult(
            run_id=run_id,
            status=RunStatus.RUNNING,  # Will be updated to COMPLETED
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

        For JSON format: writes all records to a single sink (records are
        heterogeneous but JSON handles that naturally).

        For CSV format: writes separate files per record_type to a directory,
        since CSV requires homogeneous schemas per file.

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

        # Get target sink config
        sink_name = export_config.sink
        if sink_name not in sinks:
            raise ValueError(f"Export sink '{sink_name}' not found in sinks")
        sink = sinks[sink_name]

        # Create context for sink writes
        ctx = PluginContext(run_id=run_id, config={}, landscape=None)

        # Unwrap SinkAdapter if present (adapter expects bulk writes,
        # but export writes records individually)
        from elspeth.engine.adapters import SinkAdapter

        # Extract artifact path from SinkAdapter using public API before unwrapping
        artifact_path: str | None = None
        if isinstance(sink, SinkAdapter):
            artifact_path = sink.artifact_path
            raw_sink = sink._sink
        else:
            raw_sink = sink

        if export_config.format == "csv":
            # Multi-file CSV export: one file per record type
            self._export_csv_multifile(
                exporter=exporter,
                run_id=run_id,
                artifact_path=artifact_path,
                sign=export_config.sign,
                ctx=ctx,
            )
        else:
            # JSON export: write records one at a time to single sink
            for record in exporter.export_run(run_id, sign=export_config.sign):
                raw_sink.write(record, ctx)
            raw_sink.flush()
            raw_sink.close()  # Finalize file (JSONSink writes array on close)

    def _export_csv_multifile(
        self,
        exporter: Any,  # LandscapeExporter (avoid circular import in type hint)
        run_id: str,
        artifact_path: str | None,
        sign: bool,
        ctx: PluginContext,
    ) -> None:
        """Export audit trail as multiple CSV files (one per record type).

        Creates a directory at the artifact path, then writes
        separate CSV files for each record type (run.csv, nodes.csv, etc.).

        Args:
            exporter: LandscapeExporter instance
            run_id: The completed run ID
            artifact_path: Path from SinkAdapter.artifact_path (file sinks only)
            sign: Whether to sign records
            ctx: Plugin context for sink operations

        Raises:
            ValueError: If artifact_path is None (non-file sink used for CSV export)
        """
        import csv
        from pathlib import Path

        from elspeth.core.landscape.formatters import CSVFormatter

        # Validate that we have a file-based sink path
        if artifact_path is None:
            raise ValueError(
                "CSV export requires a file-based sink with a configured path"
            )

        export_dir = Path(artifact_path)
        if export_dir.suffix:
            # Remove file extension if present, treat as directory
            export_dir = export_dir.with_suffix("")

        export_dir.mkdir(parents=True, exist_ok=True)

        # Get records grouped by type
        grouped = exporter.export_run_grouped(run_id, sign=sign)
        formatter = CSVFormatter()

        # Write each record type to its own CSV file
        for record_type, records in grouped.items():
            if not records:
                continue

            csv_path = export_dir / f"{record_type}.csv"

            # Flatten all records for CSV
            flat_records = [formatter.format(r) for r in records]

            # Get union of all keys (some records may have optional fields)
            all_keys: set[str] = set()
            for rec in flat_records:
                all_keys.update(rec.keys())
            fieldnames = sorted(all_keys)  # Sorted for determinism

            with open(csv_path, "w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()
                for rec in flat_records:
                    writer.writerow(rec)
