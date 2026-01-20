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
from collections.abc import Callable
from contextlib import suppress
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from elspeth.contracts import NodeType, RowOutcome, RunStatus, TokenInfo
from elspeth.core.config import AggregationSettings, CoalesceSettings, GateSettings
from elspeth.core.dag import ExecutionGraph
from elspeth.core.landscape import LandscapeDB, LandscapeRecorder
from elspeth.engine.processor import RowProcessor
from elspeth.engine.retry import RetryConfig, RetryManager
from elspeth.engine.schema_validator import validate_pipeline_schemas
from elspeth.engine.spans import SpanFactory
from elspeth.plugins.base import BaseGate, BaseTransform
from elspeth.plugins.context import PluginContext
from elspeth.plugins.llm.batch_errors import BatchPendingError
from elspeth.plugins.protocols import SinkProtocol, SourceProtocol

# Type alias for row-processing plugins in the transforms pipeline
# NOTE: BaseAggregation was DELETED - aggregation is now handled by
# batch-aware transforms (is_batch_aware=True on BaseTransform)
RowPlugin = BaseTransform | BaseGate
"""Union of all row-processing plugin types for pipeline transforms list."""

if TYPE_CHECKING:
    from elspeth.core.checkpoint import CheckpointManager
    from elspeth.core.checkpoint.recovery import ResumePoint
    from elspeth.core.config import CheckpointSettings, ElspethSettings


@dataclass
class PipelineConfig:
    """Configuration for a pipeline run.

    All plugin fields are now properly typed for IDE support and
    static type checking.

    Attributes:
        source: Source plugin instance
        transforms: List of transform/gate plugin instances (processed first)
        sinks: Dict of sink_name -> sink plugin instance
        config: Additional run configuration
        gates: Config-driven gates (processed AFTER transforms, BEFORE sinks)
        aggregation_settings: Dict of node_id -> AggregationSettings
        coalesce_settings: List of coalesce configurations for merging fork paths
    """

    source: SourceProtocol
    transforms: list[RowPlugin]
    sinks: dict[str, SinkProtocol]  # Sinks implement batch write directly
    config: dict[str, Any] = field(default_factory=dict)
    gates: list[GateSettings] = field(default_factory=list)
    aggregation_settings: dict[str, AggregationSettings] = field(default_factory=dict)
    coalesce_settings: list[CoalesceSettings] = field(default_factory=list)


@dataclass
class RunResult:
    """Result of a pipeline run."""

    run_id: str
    status: RunStatus
    rows_processed: int
    rows_succeeded: int
    rows_failed: int
    rows_routed: int
    rows_quarantined: int = 0
    rows_forked: int = 0
    rows_coalesced: int = 0


class RouteValidationError(Exception):
    """Raised when route configuration is invalid.

    This error is raised at pipeline initialization, before any rows are
    processed. It indicates a configuration problem that would cause
    failures during processing.
    """

    pass


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
        """
        import structlog

        logger = structlog.get_logger()

        for transform in config.transforms:
            try:
                transform.close()
            except Exception as e:
                # Log but don't raise - cleanup should be best-effort
                logger.warning(
                    "Transform cleanup failed",
                    transform=transform.name,
                    error=str(e),
                )

    def _validate_route_destinations(
        self,
        route_resolution_map: dict[tuple[str, str], str],
        available_sinks: set[str],
        transform_id_map: dict[int, str],
        transforms: list[RowPlugin],
        config_gate_id_map: dict[str, str] | None = None,
        config_gates: list[GateSettings] | None = None,
    ) -> None:
        """Validate all route destinations reference existing sinks.

        Called at pipeline initialization, BEFORE any rows are processed.
        This catches config errors early instead of failing mid-run.

        Args:
            route_resolution_map: Maps (gate_node_id, route_label) -> destination
            available_sinks: Set of sink names from PipelineConfig
            transform_id_map: Maps transform sequence -> node_id
            transforms: List of transform plugins
            config_gate_id_map: Maps config gate name -> node_id
            config_gates: List of config gate settings

        Raises:
            RouteValidationError: If any route references a non-existent sink
        """
        # Build reverse lookup: node_id -> gate name
        node_id_to_gate_name: dict[str, str] = {}
        for seq, transform in enumerate(transforms):
            if isinstance(transform, BaseGate):
                node_id = transform_id_map.get(seq)
                if node_id is not None:
                    node_id_to_gate_name[node_id] = transform.name

        # Add config gates to the lookup
        if config_gate_id_map and config_gates:
            for gate_config in config_gates:
                node_id = config_gate_id_map.get(gate_config.name)
                if node_id is not None:
                    node_id_to_gate_name[node_id] = gate_config.name

        # Check each route destination
        for (gate_node_id, route_label), destination in route_resolution_map.items():
            # "continue" means proceed to next transform, not a sink
            if destination == "continue":
                continue

            # "fork" means fork to multiple paths, not a sink
            if destination == "fork":
                continue

            # destination should be a sink name
            if destination not in available_sinks:
                gate_name = node_id_to_gate_name.get(gate_node_id, gate_node_id)
                raise RouteValidationError(
                    f"Gate '{gate_name}' can route to '{destination}' "
                    f"(via route label '{route_label}') but no sink named "
                    f"'{destination}' exists. Available sinks: {sorted(available_sinks)}"
                )

    def _validate_transform_error_sinks(
        self,
        transforms: list[RowPlugin],
        available_sinks: set[str],
        _transform_id_map: dict[int, str],
    ) -> None:
        """Validate all transform on_error destinations reference existing sinks.

        Called at pipeline initialization, BEFORE any rows are processed.
        This catches config errors early instead of failing mid-run with KeyError.

        Args:
            transforms: List of transform plugins
            available_sinks: Set of sink names from PipelineConfig
            _transform_id_map: Maps transform sequence -> node_id (unused, kept for
                API consistency with _validate_route_destinations)

        Raises:
            RouteValidationError: If any transform on_error references a non-existent sink
        """
        for transform in transforms:
            # Only BaseTransform has _on_error; BaseGate uses routing, not error sinks
            if not isinstance(transform, BaseTransform):
                continue

            # Access _on_error directly - defined in TransformProtocol and BaseTransform
            on_error = transform._on_error

            if on_error is None:
                # No error routing configured - that's fine
                continue

            if on_error == "discard":
                # "discard" is a special value, not a sink name
                continue

            # on_error should reference an existing sink
            if on_error not in available_sinks:
                raise RouteValidationError(
                    f"Transform '{transform.name}' has on_error='{on_error}' "
                    f"but no sink named '{on_error}' exists. "
                    f"Available sinks: {sorted(available_sinks)}. "
                    f"Use 'discard' to drop error rows without routing."
                )

    def _assign_plugin_node_ids(
        self,
        source: SourceProtocol,
        transforms: list[RowPlugin],
        sinks: dict[str, SinkProtocol],
        source_id: str,
        transform_id_map: dict[int, str],
        sink_id_map: dict[str, str],
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
        for sink_name, sink in sinks.items():
            if sink_name not in sink_id_map:
                raise ValueError(f"Sink '{sink_name}' not found in graph. Available sinks: {list(sink_id_map.keys())}")
            sink.node_id = sink_id_map[sink_name]

    def run(
        self,
        config: PipelineConfig,
        graph: ExecutionGraph | None = None,
        settings: "ElspethSettings | None" = None,
        batch_checkpoints: dict[str, dict[str, Any]] | None = None,
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

        Raises:
            ValueError: If graph is not provided
        """
        if graph is None:
            raise ValueError("ExecutionGraph is required. Build with ExecutionGraph.from_config(settings)")

        # Validate schema compatibility
        # Schemas are required by plugin protocols - access directly
        source_output = config.source.output_schema
        transform_inputs = [t.input_schema for t in config.transforms]
        transform_outputs = [t.output_schema for t in config.transforms]
        sink_inputs = [s.input_schema for s in config.sinks.values()]

        schema_errors = validate_pipeline_schemas(
            source_output=source_output,
            transform_inputs=transform_inputs,  # type: ignore[arg-type]
            transform_outputs=transform_outputs,  # type: ignore[arg-type]
            sink_inputs=sink_inputs,  # type: ignore[arg-type]
        )
        if schema_errors:
            raise ValueError(f"Pipeline schema incompatibility: {'; '.join(schema_errors)}")

        recorder = LandscapeRecorder(self._db)

        # Begin run
        run = recorder.begin_run(
            config=config.config,
            canonical_version=self._canonical_version,
        )

        run_completed = False
        try:
            with self._span_factory.run_span(run.run_id):
                result = self._execute_run(recorder, run.run_id, config, graph, settings, batch_checkpoints)

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

        except BatchPendingError:
            # BatchPendingError is a CONTROL-FLOW SIGNAL, not an error.
            # A batch transform has submitted work that isn't complete yet.
            # DO NOT mark run as failed - it's pending, not failed.
            # Re-raise for caller to schedule retry based on check_after_seconds.
            # The run remains in its current state (the caller should manage
            # run status transitions for pending/retry scenarios).
            raise
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
        settings: "ElspethSettings | None" = None,
        batch_checkpoints: dict[str, dict[str, Any]] | None = None,
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
        """
        # Get execution order from graph
        execution_order = graph.topological_order()

        # Build node_id -> plugin instance mapping for metadata extraction
        # Source: single plugin from config.source
        source_id = graph.get_source()
        transform_id_map = graph.get_transform_id_map()
        sink_id_map = graph.get_sink_id_map()
        config_gate_id_map = graph.get_config_gate_id_map()
        aggregation_id_map = graph.get_aggregation_id_map()

        # Map plugin instances (not config gates or aggregations - they don't have instances)
        node_to_plugin: dict[str, Any] = {}
        if source_id is not None:
            node_to_plugin[source_id] = config.source
        for seq, transform in enumerate(config.transforms):
            if seq in transform_id_map:
                node_to_plugin[transform_id_map[seq]] = transform
        for sink_name, sink in config.sinks.items():
            if sink_name in sink_id_map:
                node_to_plugin[sink_id_map[sink_name]] = sink

        # Config gates, aggregations, and coalesce nodes are identified by their node IDs (no plugin instances)
        config_gate_node_ids = set(config_gate_id_map.values())
        aggregation_node_ids = set(aggregation_id_map.values())
        coalesce_id_map = graph.get_coalesce_id_map()
        coalesce_node_ids = set(coalesce_id_map.values())

        # Register nodes with Landscape using graph's node IDs and actual plugin metadata
        from elspeth.contracts import Determinism
        from elspeth.contracts.schema import SchemaConfig

        for node_id in execution_order:
            node_info = graph.get_node_info(node_id)

            # Config gates, aggregations, and coalesce nodes have metadata in graph node, not plugin instances
            if node_id in config_gate_node_ids:
                # Config gates are deterministic (expression evaluation is deterministic)
                plugin_version = "1.0.0"
                determinism = Determinism.DETERMINISTIC
            elif node_id in aggregation_node_ids:
                # Aggregations use batch-aware transforms - determinism depends on the transform
                # Default to deterministic (statistical operations are typically deterministic)
                plugin_version = "1.0.0"
                determinism = Determinism.DETERMINISTIC
            elif node_id in coalesce_node_ids:
                # Coalesce nodes merge tokens from parallel paths - deterministic operation
                plugin_version = "1.0.0"
                determinism = Determinism.DETERMINISTIC
            else:
                # Direct access - if node_id is in execution_order (from graph.topological_order()),
                # it MUST be in node_to_plugin (built from the same graph's source, transforms, sinks).
                # A KeyError here indicates a bug in graph construction or node_to_plugin building.
                plugin = node_to_plugin[node_id]

                # Extract plugin metadata - all protocols define these attributes,
                # all base classes provide defaults. Direct access is safe.
                plugin_version = plugin.plugin_version
                determinism = plugin.determinism

            # Get schema_config from node_info config or default to dynamic
            # Schema is specified in pipeline config, not plugin attributes
            schema_dict = node_info.config.get("schema", {"fields": "dynamic"})
            schema_config = SchemaConfig.from_dict(schema_dict)

            recorder.register_node(
                run_id=run_id,
                node_id=node_id,  # Use graph's ID
                plugin_name=node_info.plugin_name,
                node_type=NodeType(node_info.node_type),  # Already lowercase
                plugin_version=plugin_version,
                config=node_info.config,
                determinism=determinism,
                schema_config=schema_config,
            )

        # Register edges from graph - key by (from_node, label) for lookup
        # Gates return route labels, so edge_map is keyed by label
        edge_map: dict[tuple[str, str], str] = {}

        for edge_info in graph.get_edges():
            edge = recorder.register_edge(
                run_id=run_id,
                from_node_id=edge_info.from_node,
                to_node_id=edge_info.to_node,
                label=edge_info.label,
                mode=edge_info.mode,
            )
            # Key by edge label - gates return route labels, transforms use "continue"
            edge_map[(edge_info.from_node, edge_info.label)] = edge.edge_id

        # Get route resolution map - maps (gate_node, label) -> "continue" | sink_name
        route_resolution_map = graph.get_route_resolution_map()

        # Validate all route destinations BEFORE processing any rows
        # This catches config errors early instead of after partial processing
        # Note: config gates also add to route_resolution_map, validated the same way
        self._validate_route_destinations(
            route_resolution_map=route_resolution_map,
            available_sinks=set(config.sinks.keys()),
            transform_id_map=transform_id_map,
            transforms=config.transforms,
            config_gate_id_map=config_gate_id_map,
            config_gates=config.gates,
        )

        # Validate transform error sink destinations
        self._validate_transform_error_sinks(
            transforms=config.transforms,
            available_sinks=set(config.sinks.keys()),
            _transform_id_map=transform_id_map,
        )

        # Get explicit node ID mappings from graph
        source_id = graph.get_source()
        if source_id is None:
            raise ValueError("Graph has no source node")
        sink_id_map = graph.get_sink_id_map()
        transform_id_map = graph.get_transform_id_map()
        config_gate_id_map = graph.get_config_gate_id_map()
        output_sink_name = graph.get_output_sink()

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
            _batch_checkpoints=batch_checkpoints or {},
        )

        # Call on_start for all plugins BEFORE processing
        # Base classes provide no-op implementations, so no hasattr needed
        config.source.on_start(ctx)
        for transform in config.transforms:
            transform.on_start(ctx)
        for sink in config.sinks.values():
            sink.on_start(ctx)

        # Create retry manager from settings if available
        retry_manager: RetryManager | None = None
        if settings is not None:
            retry_manager = RetryManager(RetryConfig.from_settings(settings.retry))

        # Create coalesce executor if config has coalesce settings
        from elspeth.engine.coalesce_executor import CoalesceExecutor
        from elspeth.engine.tokens import TokenManager

        coalesce_executor: CoalesceExecutor | None = None
        branch_to_coalesce: dict[str, str] = {}

        if settings is not None and settings.coalesce:
            branch_to_coalesce = graph.get_branch_to_coalesce_map()
            token_manager = TokenManager(recorder)

            coalesce_executor = CoalesceExecutor(
                recorder=recorder,
                span_factory=self._span_factory,
                token_manager=token_manager,
                run_id=run_id,
            )

            # Register each coalesce point
            # Direct access: graph was built from same settings, so all coalesce names
            # must exist in map. KeyError here indicates a bug in graph construction.
            for coalesce_settings in settings.coalesce:
                coalesce_node_id = coalesce_id_map[coalesce_settings.name]
                coalesce_executor.register_coalesce(coalesce_settings, coalesce_node_id)

        # Compute coalesce step positions
        # Coalesce step = after all transforms and gates
        coalesce_step_map: dict[str, int] = {}
        if settings is not None and settings.coalesce:
            base_step = len(config.transforms) + len(config.gates)
            for i, cs in enumerate(settings.coalesce):
                # Each coalesce gets its own step (in case of multiple)
                coalesce_step_map[cs.name] = base_step + i

        # Create processor with config gates info
        processor = RowProcessor(
            recorder=recorder,
            span_factory=self._span_factory,
            run_id=run_id,
            source_node_id=source_id,
            edge_map=edge_map,
            route_resolution_map=route_resolution_map,
            config_gates=config.gates,
            config_gate_id_map=config_gate_id_map,
            aggregation_settings=config.aggregation_settings,
            retry_manager=retry_manager,
            coalesce_executor=coalesce_executor,
            coalesce_node_ids=coalesce_id_map,
            branch_to_coalesce=branch_to_coalesce,
            coalesce_step_map=coalesce_step_map,
        )

        # Process rows - Buffer TOKENS, not dicts, to preserve identity
        from elspeth.contracts import TokenInfo
        from elspeth.engine.executors import SinkExecutor

        rows_processed = 0
        rows_succeeded = 0
        rows_failed = 0
        rows_routed = 0
        rows_quarantined = 0
        rows_forked = 0
        rows_coalesced = 0
        pending_tokens: dict[str, list[TokenInfo]] = {name: [] for name in config.sinks}

        # Compute default last_node_id for end-of-source checkpointing
        # (e.g., flush_pending when no rows were processed in the main loop)
        # This mirrors the in-loop logic for consistency
        default_last_node_id: str
        if config.gates:
            last_gate_name = config.gates[-1].name
            default_last_node_id = config_gate_id_map[last_gate_name]
        elif config.transforms:
            transform_node_id = config.transforms[-1].node_id
            assert transform_node_id is not None
            default_last_node_id = transform_node_id
        else:
            default_last_node_id = source_id

        try:
            with self._span_factory.source_span(config.source.name):
                for row_index, source_item in enumerate(config.source.load(ctx)):
                    rows_processed += 1

                    # Handle quarantined source rows - route directly to sink
                    if source_item.is_quarantined:
                        rows_quarantined += 1
                        # Route quarantined row to configured sink if it exists
                        quarantine_sink = source_item.quarantine_destination
                        if quarantine_sink and quarantine_sink in config.sinks:
                            # Create a token for the quarantined row
                            quarantine_token = processor.token_manager.create_initial_token(
                                run_id=run_id,
                                source_node_id=source_id,
                                row_index=row_index,
                                row_data=source_item.row,
                            )
                            pending_tokens[quarantine_sink].append(quarantine_token)
                        # Skip normal processing - row is already handled
                        continue

                    # Extract row data from SourceRow (all source items are SourceRow)
                    row_data: dict[str, Any] = source_item.row

                    results = processor.process_row(
                        row_index=row_index,
                        row_data=row_data,
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
                    for result in results:
                        if result.outcome == RowOutcome.COMPLETED:
                            rows_succeeded += 1
                            # Fork children route to branch-named sink if it exists
                            sink_name = output_sink_name
                            if result.token.branch_name is not None and result.token.branch_name in config.sinks:
                                sink_name = result.token.branch_name
                            pending_tokens[sink_name].append(result.token)
                        elif result.outcome == RowOutcome.ROUTED:
                            rows_routed += 1
                            # GateExecutor contract: ROUTED outcome always has sink_name set
                            assert result.sink_name is not None
                            pending_tokens[result.sink_name].append(result.token)
                        elif result.outcome == RowOutcome.FAILED:
                            rows_failed += 1
                        elif result.outcome == RowOutcome.QUARANTINED:
                            rows_quarantined += 1
                        elif result.outcome == RowOutcome.FORKED:
                            rows_forked += 1
                            # Children are counted separately when they reach terminal state
                            pass
                        elif result.outcome == RowOutcome.CONSUMED_IN_BATCH:
                            # Aggregated - will be counted when batch flushes
                            pass
                        elif result.outcome == RowOutcome.COALESCED:
                            # Merged token from coalesce - route to output sink
                            rows_coalesced += 1
                            pending_tokens[output_sink_name].append(result.token)

            # ─────────────────────────────────────────────────────────────────
            # CRITICAL: Flush remaining aggregation buffers at end-of-source
            # ─────────────────────────────────────────────────────────────────
            if config.aggregation_settings:
                agg_succeeded, agg_failed = self._flush_remaining_aggregation_buffers(
                    config=config,
                    processor=processor,
                    ctx=ctx,
                    pending_tokens=pending_tokens,
                    output_sink_name=output_sink_name,
                    run_id=run_id,
                    checkpoint=False,  # Checkpointing now happens after sink write
                    last_node_id=default_last_node_id,
                )
                rows_succeeded += agg_succeeded
                rows_failed += agg_failed

            # Flush pending coalesce operations at end-of-source
            if coalesce_executor is not None:
                # Step for coalesce flush = after all transforms and gates
                flush_step = len(config.transforms) + len(config.gates)
                pending_outcomes = coalesce_executor.flush_pending(flush_step)

                # Handle any merged tokens from flush
                for outcome in pending_outcomes:
                    if outcome.merged_token is not None:
                        # Successful merge - route to output sink
                        rows_coalesced += 1
                        pending_tokens[output_sink_name].append(outcome.merged_token)
                    elif outcome.failure_reason:
                        # Coalesce failed (timeout, missing branches, etc.)
                        # Failure is recorded in audit trail by executor.
                        # Not counted as rows_failed since the individual fork children
                        # were already counted when they reached their terminal states.
                        pass

            # Write to sinks using SinkExecutor
            sink_executor = SinkExecutor(recorder, self._span_factory, run_id)
            # Step = transforms + config gates + 1 (for sink)
            step = len(config.transforms) + len(config.gates) + 1

            # Create checkpoint callback for post-sink checkpointing
            def checkpoint_after_sink(sink_node_id: str) -> Callable[[TokenInfo], None]:
                def callback(token: TokenInfo) -> None:
                    self._maybe_checkpoint(
                        run_id=run_id,
                        token_id=token.token_id,
                        node_id=sink_node_id,
                    )

                return callback

            for sink_name, tokens in pending_tokens.items():
                if tokens and sink_name in config.sinks:
                    sink = config.sinks[sink_name]
                    sink_node_id = sink_id_map[sink_name]

                    sink_executor.write(
                        sink=sink,
                        tokens=tokens,
                        ctx=ctx,
                        step_in_pipeline=step,
                        on_token_written=checkpoint_after_sink(sink_node_id),
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
            rows_quarantined=rows_quarantined,
            rows_forked=rows_forked,
            rows_coalesced=rows_coalesced,
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
            try:
                key_str = os.environ["ELSPETH_SIGNING_KEY"]
            except KeyError:
                raise ValueError("ELSPETH_SIGNING_KEY environment variable required for signed export") from None
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

        if export_config.format == "csv":
            # Multi-file CSV export: one file per record type
            # CSV export writes files directly (not via sink.write), so we need
            # the path from sink config. CSV format requires file-based sink.
            if "path" not in sink.config:
                raise ValueError(
                    f"CSV export requires file-based sink with 'path' in config, but sink '{sink_name}' has no path configured"
                )
            artifact_path: str = sink.config["path"]
            self._export_csv_multifile(
                exporter=exporter,
                run_id=run_id,
                artifact_path=artifact_path,
                sign=export_config.sign,
                ctx=ctx,
            )
        else:
            # JSON export: batch all records for single write
            records = list(exporter.export_run(run_id, sign=export_config.sign))
            if records:
                # Capture ArtifactDescriptor for audit trail (future use)
                _artifact_descriptor = sink.write(records, ctx)
            sink.flush()
            sink.close()

    def _export_csv_multifile(
        self,
        exporter: Any,  # LandscapeExporter (avoid circular import in type hint)
        run_id: str,
        artifact_path: str,
        sign: bool,
        ctx: PluginContext,  # - reserved for future use
    ) -> None:
        """Export audit trail as multiple CSV files (one per record type).

        Creates a directory at the artifact path, then writes
        separate CSV files for each record type (run.csv, nodes.csv, etc.).

        Args:
            exporter: LandscapeExporter instance
            run_id: The completed run ID
            artifact_path: Path from sink config (validated by caller)
            sign: Whether to sign records
            ctx: Plugin context for sink operations (reserved for future use)
        """
        import csv
        from pathlib import Path

        from elspeth.core.landscape.formatters import CSVFormatter

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

    def resume(
        self,
        resume_point: "ResumePoint",
        config: PipelineConfig,
        graph: ExecutionGraph,
        *,
        payload_store: Any = None,
        settings: "ElspethSettings | None" = None,
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
        recorder = LandscapeRecorder(self._db)

        # 1. Handle incomplete batches
        self._handle_incomplete_batches(recorder, run_id)

        # 2. Update run status to running
        self._update_run_status(recorder, run_id, RunStatus.RUNNING)

        # 3. Build restored aggregation state map
        restored_state: dict[str, dict[str, Any]] = {}
        if resume_point.aggregation_state is not None:
            restored_state[resume_point.node_id] = resume_point.aggregation_state

        # 4. Get unprocessed row data from payload store
        from elspeth.core.checkpoint import RecoveryManager

        if self._checkpoint_manager is None:
            raise ValueError("CheckpointManager is required for resume - Orchestrator must be initialized with checkpoint_manager")
        recovery = RecoveryManager(self._db, self._checkpoint_manager)
        unprocessed_rows = recovery.get_unprocessed_row_data(run_id, payload_store)

        if not unprocessed_rows:
            # All rows were processed - complete the run
            recorder.complete_run(run_id, status="completed")
            return RunResult(
                run_id=run_id,
                status=RunStatus.COMPLETED,
                rows_processed=0,
                rows_succeeded=0,
                rows_failed=0,
                rows_routed=0,
            )

        # 5. Process unprocessed rows
        result = self._process_resumed_rows(
            recorder=recorder,
            run_id=run_id,
            config=config,
            graph=graph,
            unprocessed_rows=unprocessed_rows,
            restored_aggregation_state=restored_state,
            settings=settings,
        )

        # 6. Complete the run
        recorder.complete_run(run_id, status="completed")
        result.status = RunStatus.COMPLETED

        # 7. Delete checkpoints on successful completion
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
        settings: "ElspethSettings | None" = None,
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

        Returns:
            RunResult with processing counts
        """
        # Get explicit node ID mappings from graph
        source_id = graph.get_source()
        if source_id is None:
            raise ValueError("Graph has no source node")
        sink_id_map = graph.get_sink_id_map()
        transform_id_map = graph.get_transform_id_map()
        config_gate_id_map = graph.get_config_gate_id_map()
        coalesce_id_map = graph.get_coalesce_id_map()
        output_sink_name = graph.get_output_sink()

        # Build edge_map from graph edges
        edge_map: dict[tuple[str, str], str] = {}
        for i, edge_info in enumerate(graph.get_edges()):
            # Generate synthetic edge_id for resume (edges were registered in original run)
            edge_id = f"resume_edge_{i}"
            edge_map[(edge_info.from_node, edge_info.label)] = edge_id

        # Get route resolution map
        route_resolution_map = graph.get_route_resolution_map()

        # Validate route destinations (config may have changed since original run)
        # This catches config errors early instead of after partial processing
        self._validate_route_destinations(
            route_resolution_map=route_resolution_map,
            available_sinks=set(config.sinks.keys()),
            transform_id_map=transform_id_map,
            transforms=config.transforms,
            config_gate_id_map=config_gate_id_map,
            config_gates=config.gates,
        )

        # Validate transform error sink destinations
        self._validate_transform_error_sinks(
            transforms=config.transforms,
            available_sinks=set(config.sinks.keys()),
            _transform_id_map=transform_id_map,
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
        )

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

        # Create retry manager from settings if available
        retry_manager: RetryManager | None = None
        if settings is not None:
            retry_manager = RetryManager(RetryConfig.from_settings(settings.retry))

        # Create coalesce executor if config has coalesce settings
        from elspeth.engine.coalesce_executor import CoalesceExecutor
        from elspeth.engine.tokens import TokenManager

        coalesce_executor: CoalesceExecutor | None = None
        branch_to_coalesce: dict[str, str] = {}

        if settings is not None and settings.coalesce:
            branch_to_coalesce = graph.get_branch_to_coalesce_map()
            token_manager = TokenManager(recorder)

            coalesce_executor = CoalesceExecutor(
                recorder=recorder,
                span_factory=self._span_factory,
                token_manager=token_manager,
                run_id=run_id,
            )

            for coalesce_settings in settings.coalesce:
                coalesce_node_id = coalesce_id_map[coalesce_settings.name]
                coalesce_executor.register_coalesce(coalesce_settings, coalesce_node_id)

        # Compute coalesce step positions
        coalesce_step_map: dict[str, int] = {}
        if settings is not None and settings.coalesce:
            base_step = len(config.transforms) + len(config.gates)
            for i, cs in enumerate(settings.coalesce):
                coalesce_step_map[cs.name] = base_step + i

        # Create processor with restored aggregation state
        processor = RowProcessor(
            recorder=recorder,
            span_factory=self._span_factory,
            run_id=run_id,
            source_node_id=source_id,
            edge_map=edge_map,
            route_resolution_map=route_resolution_map,
            config_gates=config.gates,
            config_gate_id_map=config_gate_id_map,
            aggregation_settings=config.aggregation_settings,
            retry_manager=retry_manager,
            coalesce_executor=coalesce_executor,
            coalesce_node_ids=coalesce_id_map,
            branch_to_coalesce=branch_to_coalesce,
            coalesce_step_map=coalesce_step_map,
            restored_aggregation_state=restored_aggregation_state,
        )

        # Process rows - Buffer TOKENS
        from elspeth.contracts import TokenInfo
        from elspeth.engine.executors import SinkExecutor

        rows_processed = 0
        rows_succeeded = 0
        rows_failed = 0
        rows_routed = 0
        rows_quarantined = 0
        rows_forked = 0
        rows_coalesced = 0
        pending_tokens: dict[str, list[TokenInfo]] = {name: [] for name in config.sinks}

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
                rows_processed += 1

                results = processor.process_existing_row(
                    row_id=row_id,
                    row_data=row_data,
                    transforms=config.transforms,
                    ctx=ctx,
                )

                # Handle all results from this row
                for result in results:
                    if result.outcome == RowOutcome.COMPLETED:
                        rows_succeeded += 1
                        sink_name = output_sink_name
                        if result.token.branch_name is not None and result.token.branch_name in config.sinks:
                            sink_name = result.token.branch_name
                        pending_tokens[sink_name].append(result.token)
                    elif result.outcome == RowOutcome.ROUTED:
                        rows_routed += 1
                        assert result.sink_name is not None
                        pending_tokens[result.sink_name].append(result.token)
                    elif result.outcome == RowOutcome.FAILED:
                        rows_failed += 1
                    elif result.outcome == RowOutcome.QUARANTINED:
                        rows_quarantined += 1
                    elif result.outcome == RowOutcome.FORKED:
                        rows_forked += 1
                    elif result.outcome == RowOutcome.CONSUMED_IN_BATCH:
                        pass
                    elif result.outcome == RowOutcome.COALESCED:
                        rows_coalesced += 1
                        pending_tokens[output_sink_name].append(result.token)

            # ─────────────────────────────────────────────────────────────────
            # CRITICAL: Flush remaining aggregation buffers at end-of-source
            # ─────────────────────────────────────────────────────────────────
            if config.aggregation_settings:
                agg_succeeded, agg_failed = self._flush_remaining_aggregation_buffers(
                    config=config,
                    processor=processor,
                    ctx=ctx,
                    pending_tokens=pending_tokens,
                    output_sink_name=output_sink_name,
                    run_id=run_id,
                    checkpoint=False,  # No checkpointing during resume
                )
                rows_succeeded += agg_succeeded
                rows_failed += agg_failed

            # Flush pending coalesce operations
            if coalesce_executor is not None:
                flush_step = len(config.transforms) + len(config.gates)
                pending_outcomes = coalesce_executor.flush_pending(flush_step)

                for outcome in pending_outcomes:
                    if outcome.merged_token is not None:
                        rows_coalesced += 1
                        pending_tokens[output_sink_name].append(outcome.merged_token)

            # Write to sinks using SinkExecutor
            sink_executor = SinkExecutor(recorder, self._span_factory, run_id)
            step = len(config.transforms) + len(config.gates) + 1

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
            for transform in config.transforms:
                with suppress(Exception):
                    transform.on_complete(ctx)
            for sink in config.sinks.values():
                with suppress(Exception):
                    sink.on_complete(ctx)

            # Close all sinks (NOT source - wasn't opened)
            for sink in config.sinks.values():
                sink.close()

        return RunResult(
            run_id=run_id,
            status=RunStatus.RUNNING,  # Will be updated by caller
            rows_processed=rows_processed,
            rows_succeeded=rows_succeeded,
            rows_failed=rows_failed,
            rows_routed=rows_routed,
            rows_quarantined=rows_quarantined,
            rows_forked=rows_forked,
            rows_coalesced=rows_coalesced,
        )

    def _handle_incomplete_batches(
        self,
        recorder: LandscapeRecorder,
        run_id: str,
    ) -> None:
        """Find and handle incomplete batches for recovery.

        - EXECUTING batches: Mark as failed (crash interrupted), then retry
        - FAILED batches: Retry with incremented attempt
        - DRAFT batches: Leave as-is (collection continues)

        Args:
            recorder: LandscapeRecorder for database operations
            run_id: Run being recovered
        """
        from elspeth.contracts.enums import BatchStatus

        incomplete = recorder.get_incomplete_batches(run_id)

        for batch in incomplete:
            if batch.status == BatchStatus.EXECUTING:
                # Crash interrupted mid-execution, mark failed then retry
                recorder.update_batch_status(batch.batch_id, "failed")
                recorder.retry_batch(batch.batch_id)
            elif batch.status == BatchStatus.FAILED:
                # Previous failure, retry
                recorder.retry_batch(batch.batch_id)
            # DRAFT batches continue normally (collection resumes)

    def _update_run_status(
        self,
        recorder: LandscapeRecorder,
        run_id: str,
        status: RunStatus,
    ) -> None:
        """Update run status without completing the run.

        Used during recovery to set status back to RUNNING.

        Args:
            recorder: LandscapeRecorder for database operations
            run_id: Run to update
            status: New status
        """
        from elspeth.core.landscape.schema import runs_table

        with self._db.connection() as conn:
            conn.execute(runs_table.update().where(runs_table.c.run_id == run_id).values(status=status.value))

    def _flush_remaining_aggregation_buffers(
        self,
        config: PipelineConfig,
        processor: RowProcessor,
        ctx: PluginContext,
        pending_tokens: dict[str, list[TokenInfo]],
        output_sink_name: str,
        run_id: str,
        checkpoint: bool = True,
        last_node_id: str | None = None,
    ) -> tuple[int, int]:
        """Flush remaining aggregation buffers at end-of-source.

        Without this, rows buffered but not yet flushed (e.g., 50 rows
        when trigger is count=100) would be silently lost.

        Args:
            config: Pipeline configuration with aggregation_settings
            processor: RowProcessor with aggregation executor
            ctx: Plugin context for transform execution
            pending_tokens: Dict of sink_name -> tokens to append results to
            output_sink_name: Default sink for aggregation output
            run_id: Current run ID (for checkpointing)
            checkpoint: Whether to create checkpoints for flushed tokens
                       (True for _execute_run, False for _process_resumed_rows)
            last_node_id: Node ID to use for checkpointing (required if checkpoint=True)

        Returns:
            Tuple of (rows_succeeded, rows_failed) from flushing

        Raises:
            RuntimeError: If no batch-aware transform found for an aggregation
                         (indicates bug in graph construction or pipeline config)
        """
        from elspeth.contracts import TokenInfo
        from elspeth.contracts.enums import TriggerType

        rows_succeeded = 0
        rows_failed = 0

        for agg_node_id, agg_settings in config.aggregation_settings.items():
            # aggregation_settings is keyed by node_id (set in cli.py)
            # The aggregation name is available via agg_settings.name
            agg_name = agg_settings.name

            # Check if there are buffered rows
            buffered_count = processor._aggregation_executor.get_buffer_count(agg_node_id)
            if buffered_count == 0:
                continue

            # Find the batch-aware transform for this aggregation
            # Only BaseTransform can have is_batch_aware (gates cannot)
            agg_transform: BaseTransform | None = None
            for t in config.transforms:
                if isinstance(t, BaseTransform) and t.node_id == agg_node_id and t.is_batch_aware:
                    agg_transform = t
                    break

            if agg_transform is None:
                raise RuntimeError(
                    f"No batch-aware transform found for aggregation '{agg_name}' "
                    f"(node_id={agg_node_id}). This indicates a bug in graph construction "
                    f"or pipeline configuration."
                )

            # Compute step_in_pipeline for this aggregation
            agg_step = next(
                (i for i, t in enumerate(config.transforms) if t.node_id == agg_node_id),
                len(config.transforms),
            )

            # Execute flush with END_OF_SOURCE trigger
            flush_result, buffered_tokens = processor._aggregation_executor.execute_flush(
                node_id=agg_node_id,
                transform=agg_transform,
                ctx=ctx,
                step_in_pipeline=agg_step,
                trigger_type=TriggerType.END_OF_SOURCE,
            )

            # Handle the flushed batch result
            if flush_result.status == "success":
                if flush_result.row is not None and buffered_tokens:
                    # Single row output - reuse first buffered token's metadata
                    output_token = TokenInfo(
                        token_id=buffered_tokens[0].token_id,
                        row_id=buffered_tokens[0].row_id,
                        row_data=flush_result.row,
                        branch_name=buffered_tokens[0].branch_name,
                    )
                    pending_tokens[output_sink_name].append(output_token)
                    rows_succeeded += 1

                    # Checkpoint the flushed aggregation token
                    if checkpoint and last_node_id is not None:
                        self._maybe_checkpoint(
                            run_id=run_id,
                            token_id=output_token.token_id,
                            node_id=last_node_id,
                        )
                elif flush_result.rows is not None and buffered_tokens:
                    # Multiple row output - use expand_token for proper audit
                    expanded = processor.token_manager.expand_token(
                        parent_token=buffered_tokens[0],
                        expanded_rows=flush_result.rows,
                        step_in_pipeline=agg_step,
                    )
                    for exp_token in expanded:
                        pending_tokens[output_sink_name].append(exp_token)
                        rows_succeeded += 1

                        # Checkpoint each expanded token
                        if checkpoint and last_node_id is not None:
                            self._maybe_checkpoint(
                                run_id=run_id,
                                token_id=exp_token.token_id,
                                node_id=last_node_id,
                            )
            else:
                # Flush failed
                rows_failed += len(buffered_tokens)

        return rows_succeeded, rows_failed
