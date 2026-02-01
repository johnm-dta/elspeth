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

from __future__ import annotations

import hashlib
import json
import os
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from elspeth.contracts.events import TelemetryEvent
    from elspeth.contracts.payload_store import PayloadStore
    from elspeth.core.events import EventBusProtocol
    from elspeth.telemetry import TelemetryManager

from elspeth import __version__ as ENGINE_VERSION
from elspeth.contracts import BatchPendingError, ExportStatus, NodeType, PendingOutcome, RowOutcome, RunStatus, TokenInfo
from elspeth.contracts.cli import ProgressEvent
from elspeth.contracts.config import RuntimeRetryConfig
from elspeth.contracts.enums import TriggerType
from elspeth.contracts.errors import OrchestrationInvariantError
from elspeth.contracts.events import (
    PhaseAction,
    PhaseCompleted,
    PhaseError,
    PhaseStarted,
    PipelinePhase,
    RunCompletionStatus,
    RunSummary,
)
from elspeth.contracts.types import (
    AggregationName,
    BranchName,
    CoalesceName,
    GateName,
    NodeID,
    SinkName,
)
from elspeth.core.canonical import stable_hash
from elspeth.core.config import AggregationSettings, CoalesceSettings, GateSettings
from elspeth.core.dag import ExecutionGraph
from elspeth.core.landscape import LandscapeDB, LandscapeRecorder
from elspeth.core.operations import track_operation
from elspeth.engine.processor import RowProcessor
from elspeth.engine.retry import RetryManager
from elspeth.engine.spans import SpanFactory
from elspeth.plugins.context import PluginContext
from elspeth.plugins.protocols import GateProtocol, SinkProtocol, SourceProtocol, TransformProtocol

# Type alias for row-processing plugins in the transforms pipeline
# NOTE: BaseAggregation was DELETED - aggregation is now handled by
# batch-aware transforms (is_batch_aware=True on TransformProtocol)
# Using protocols instead of base classes to support protocol-only plugins.
RowPlugin = TransformProtocol | GateProtocol
"""Union of all row-processing plugin types for pipeline transforms list."""

if TYPE_CHECKING:
    from elspeth.contracts import ResumePoint
    from elspeth.contracts.config.runtime import RuntimeCheckpointConfig, RuntimeConcurrencyConfig
    from elspeth.core.checkpoint import CheckpointManager
    from elspeth.core.config import ElspethSettings
    from elspeth.core.rate_limit import RateLimitRegistry
    from elspeth.engine.clock import Clock


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
    rows_coalesce_failed: int = 0  # Coalesce failures (quorum_not_met, incomplete_branches)
    rows_expanded: int = 0  # Deaggregation parent tokens
    rows_buffered: int = 0  # Passthrough mode buffered tokens
    routed_destinations: dict[str, int] = field(default_factory=dict)  # sink_name -> count


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
        event_bus: EventBusProtocol = None,  # type: ignore[assignment]
        canonical_version: str = "sha256-rfc8785-v1",
        checkpoint_manager: CheckpointManager | None = None,
        checkpoint_config: RuntimeCheckpointConfig | None = None,
        clock: Clock | None = None,
        rate_limit_registry: RateLimitRegistry | None = None,
        concurrency_config: RuntimeConcurrencyConfig | None = None,
        telemetry_manager: TelemetryManager | None = None,
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

        EXCEPTION TO "PLUGIN BUGS SHOULD CRASH" PRINCIPLE:
        ---------------------------------------------------
        Cleanup is best-effort even if individual close() fails. This violates
        the general principle that plugin method exceptions should crash, but
        attempts all cleanups, then raises if any failed. Per CLAUDE.md, plugins
        are system-owned code and bugs must crash - but we collect errors first
        to ensure all plugins get cleanup attempts before failing.
        """
        import structlog

        logger = structlog.get_logger()
        cleanup_errors: list[tuple[str, Exception]] = []

        for transform in config.transforms:
            try:
                transform.close()
            except Exception as e:
                # Collect error but continue to attempt other cleanups
                logger.warning(
                    "Transform cleanup failed - plugin close() raised exception",
                    transform=transform.name,
                    error=str(e),
                    error_type=type(e).__name__,
                )
                cleanup_errors.append((transform.name, e))

        # After attempting all cleanups, raise if any failed (plugins are system code)
        if cleanup_errors:
            error_summary = "; ".join(f"{name}: {type(e).__name__}: {e}" for name, e in cleanup_errors)
            raise RuntimeError(f"Plugin cleanup failed for {len(cleanup_errors)} transform(s): {error_summary}")

    def _validate_route_destinations(
        self,
        route_resolution_map: dict[tuple[NodeID, str], str],
        available_sinks: set[str],
        transform_id_map: dict[int, NodeID],
        transforms: list[RowPlugin],
        config_gate_id_map: dict[GateName, NodeID] | None = None,
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
        # All gates in transforms and config_gates MUST have entries in their ID maps
        # (graph construction bug if missing)
        node_id_to_gate_name: dict[str, str] = {}
        for seq, transform in enumerate(transforms):
            if isinstance(transform, GateProtocol):
                # Graph must have ID for every transform - crash if missing
                node_id = transform_id_map[seq]
                node_id_to_gate_name[node_id] = transform.name

        # Add config gates to the lookup
        if config_gate_id_map and config_gates:
            for gate_config in config_gates:
                # Graph must have ID for every config gate - crash if missing
                node_id = config_gate_id_map[GateName(gate_config.name)]
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
                # Every gate in route_resolution_map MUST have a name mapping
                gate_name = node_id_to_gate_name[gate_node_id]
                raise RouteValidationError(
                    f"Gate '{gate_name}' can route to '{destination}' "
                    f"(via route label '{route_label}') but no sink named "
                    f"'{destination}' exists. Available sinks: {sorted(available_sinks)}"
                )

    def _validate_transform_error_sinks(
        self,
        transforms: list[RowPlugin],
        available_sinks: set[str],
        _transform_id_map: dict[int, NodeID],
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
            # Only TransformProtocol has _on_error; GateProtocol uses routing, not error sinks
            if not isinstance(transform, TransformProtocol):
                continue

            # Access _on_error directly - defined in TransformProtocol
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

    def _validate_source_quarantine_destination(
        self,
        source: SourceProtocol,
        available_sinks: set[str],
    ) -> None:
        """Validate source quarantine destination references an existing sink.

        Called at pipeline initialization, BEFORE any rows are processed.
        This catches config errors early instead of silently dropping quarantined
        rows at runtime (P2-2026-01-19-source-quarantine-silent-drop).

        Args:
            source: Source plugin instance
            available_sinks: Set of sink names from PipelineConfig

        Raises:
            RouteValidationError: If source on_validation_failure references
                a non-existent sink
        """
        # _on_validation_failure is required by SourceProtocol
        on_validation_failure = source._on_validation_failure

        if on_validation_failure == "discard":
            # "discard" is a special value, not a sink name
            return

        # on_validation_failure should reference an existing sink
        if on_validation_failure not in available_sinks:
            raise RouteValidationError(
                f"Source '{source.name}' has on_validation_failure='{on_validation_failure}' "
                f"but no sink named '{on_validation_failure}' exists. "
                f"Available sinks: {sorted(available_sinks)}. "
                f"Use 'discard' to drop invalid rows without routing."
            )

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
        for sink_name, sink in sinks.items():
            if SinkName(sink_name) not in sink_id_map:
                raise ValueError(f"Sink '{sink_name}' not found in graph. Available sinks: {list(sink_id_map.keys())}")
            sink.node_id = sink_id_map[SinkName(sink_name)]

    def _compute_coalesce_step_map(
        self,
        graph: ExecutionGraph,
        config: PipelineConfig,
        settings: ElspethSettings | None,
    ) -> dict[CoalesceName, int]:
        """Compute coalesce step positions aligned with graph topology.

        Coalesce step = gate_idx + 1 (step after the producing fork gate)

        This ensures:
        1. Fork children skip to coalesce step and merge before downstream processing
        2. Merged tokens continue from the coalesce step, executing downstream nodes
        3. Execution path matches graph topology (coalesce → downstream → sink)

        The graph's coalesce_gate_index provides the pipeline position of each
        coalesce's producing fork gate. The coalesce step is one position after
        that gate, allowing merged tokens to traverse downstream nodes.

        Args:
            graph: The execution graph (provides coalesce gate positions)
            config: Pipeline configuration
            settings: Elspeth settings (may be None)

        Returns:
            Dict mapping coalesce name to its step index in the pipeline
        """
        coalesce_step_map: dict[CoalesceName, int] = {}
        if settings is not None and settings.coalesce:
            # Get actual gate positions from graph topology
            coalesce_gate_index = graph.get_coalesce_gate_index()

            for cs in settings.coalesce:
                coalesce_name = CoalesceName(cs.name)
                # Coalesce step is one AFTER the fork gate
                # This allows merged tokens to continue downstream processing
                gate_idx = coalesce_gate_index[coalesce_name]
                coalesce_step_map[coalesce_name] = gate_idx + 1

        return coalesce_step_map

    def run(
        self,
        config: PipelineConfig,
        graph: ExecutionGraph | None = None,
        settings: ElspethSettings | None = None,
        batch_checkpoints: dict[str, dict[str, Any]] | None = None,
        *,
        payload_store: PayloadStore,
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

            recorder = LandscapeRecorder(self._db, payload_store=payload_store)
            run = recorder.begin_run(
                config=config.config,
                canonical_version=self._canonical_version,
                source_schema_json=source_schema_json,
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
            with self._span_factory.run_span(run.run_id):
                result = self._execute_run(
                    recorder,
                    run.run_id,
                    config,
                    graph,
                    settings,
                    batch_checkpoints,
                    payload_store=payload_store,
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

                    self._export_landscape(
                        run_id=run.run_id,
                        settings=settings,
                        sinks=config.sinks,
                    )

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
            finally:
                # Always clean up transforms, even on failure
                self._cleanup_transforms(config)

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
        """
        # Store graph for checkpointing during execution
        self._current_graph = graph

        # Local imports for telemetry events - consolidated here to avoid repeated imports
        from elspeth.telemetry import (
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

                # Get schema_config from node_info config
                # DataPluginConfig enforces that all data plugins have schema
                schema_dict = node_info.config["schema"]
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

            # Validate source quarantine destination
            self._validate_source_quarantine_destination(
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
        default_sink_name = graph.get_default_sink()

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
            retry_manager = RetryManager(RuntimeRetryConfig.from_settings(settings.retry))

        # Create coalesce executor if config has coalesce settings
        from elspeth.engine.coalesce_executor import CoalesceExecutor
        from elspeth.engine.tokens import TokenManager

        coalesce_executor: CoalesceExecutor | None = None
        branch_to_coalesce: dict[BranchName, CoalesceName] = {}

        if settings is not None and settings.coalesce:
            branch_to_coalesce = graph.get_branch_to_coalesce_map()
            token_manager = TokenManager(recorder)

            coalesce_executor = CoalesceExecutor(
                recorder=recorder,
                span_factory=self._span_factory,
                token_manager=token_manager,
                run_id=run_id,
                clock=self._clock,
            )

            # Register each coalesce point
            # Direct access: graph was built from same settings, so all coalesce names
            # must exist in map. KeyError here indicates a bug in graph construction.
            for coalesce_settings in settings.coalesce:
                coalesce_node_id = coalesce_id_map[CoalesceName(coalesce_settings.name)]
                coalesce_executor.register_coalesce(coalesce_settings, coalesce_node_id)

        # Compute coalesce step positions FROM GRAPH TOPOLOGY
        coalesce_step_map = self._compute_coalesce_step_map(graph, config, settings)

        # Convert aggregation_settings keys from str to NodeID
        typed_aggregation_settings: dict[NodeID, AggregationSettings] = {NodeID(k): v for k, v in config.aggregation_settings.items()}

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
            aggregation_settings=typed_aggregation_settings,
            retry_manager=retry_manager,
            coalesce_executor=coalesce_executor,
            coalesce_node_ids=coalesce_id_map,
            branch_to_coalesce=branch_to_coalesce,
            coalesce_step_map=coalesce_step_map,
            payload_store=payload_store,
            clock=self._clock,
            max_workers=self._concurrency_config.max_workers if self._concurrency_config else None,
            telemetry_manager=self._telemetry,
        )

        # Process rows - Buffer TOKENS, not dicts, to preserve identity
        from elspeth.engine.executors import SinkExecutor

        rows_processed = 0
        rows_succeeded = 0
        rows_failed = 0
        rows_routed = 0
        rows_quarantined = 0
        rows_forked = 0
        rows_coalesced = 0
        rows_coalesce_failed = 0
        rows_expanded = 0
        rows_buffered = 0
        routed_destinations: dict[str, int] = {}  # Track routing destinations
        # Track (token, pending_outcome) pairs for deferred outcome recording
        # Outcomes are recorded by SinkExecutor.write() AFTER sink durability is achieved
        # Fix: P1-2026-01-31 - use PendingOutcome to carry error_hash for QUARANTINED
        pending_tokens: dict[str, list[tuple[TokenInfo, PendingOutcome | None]]] = {name: [] for name in config.sinks}

        # Pre-compute aggregation transform lookup for O(1) access per timeout check
        # Maps node_id_str -> (transform, step_in_pipeline)
        # NOTE: Steps are 0-indexed here; handle_timeout_flush converts to 1-indexed
        # for audit recording (to avoid node_state step_index collisions)
        agg_transform_lookup: dict[str, tuple[TransformProtocol, int]] = {}
        if config.aggregation_settings:
            for i, t in enumerate(config.transforms):
                if isinstance(t, TransformProtocol) and t.is_batch_aware and t.node_id in config.aggregation_settings:
                    agg_transform_lookup[t.node_id] = (t, i)  # 0-indexed, converted in handle_timeout_flush

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
                try:
                    for row_index, source_item in enumerate(source_iterator):
                        rows_processed += 1

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

                        # Handle quarantined source rows - route directly to sink
                        if source_item.is_quarantined:
                            rows_quarantined += 1
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

                            # Destination validated - proceed with routing
                            # Create a token for the quarantined row
                            quarantine_token = processor.token_manager.create_initial_token(
                                run_id=run_id,
                                source_node_id=source_id,
                                row_index=row_index,
                                row_data=source_item.row,
                            )

                            # Emit RowCreated telemetry AFTER Landscape recording succeeds
                            self._emit_telemetry(
                                RowCreated(
                                    timestamp=datetime.now(UTC),
                                    run_id=run_id,
                                    row_id=quarantine_token.row_id,
                                    token_id=quarantine_token.token_id,
                                    content_hash=stable_hash(source_item.row),
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
                                rows_processed == 1  # First row - immediate feedback
                                or rows_processed % progress_interval == 0  # Every 100 rows
                                or time_since_last_progress >= progress_time_interval  # Every 5 seconds
                            )
                            if should_emit:
                                elapsed = current_time - start_time
                                self._events.emit(
                                    ProgressEvent(
                                        rows_processed=rows_processed,
                                        # Include routed rows in success count - they reached their destination
                                        rows_succeeded=rows_succeeded + rows_routed,
                                        rows_failed=rows_failed,
                                        rows_quarantined=rows_quarantined,
                                        elapsed_seconds=elapsed,
                                    )
                                )
                                last_progress_time = current_time
                            # Restore operation_id before next iteration
                            # (generator may execute external calls on next() call)
                            ctx.operation_id = source_operation_id
                            # Skip normal processing - row is already handled
                            continue

                        # Extract row data from SourceRow (all source items are SourceRow)
                        row_data: dict[str, Any] = source_item.row

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
                        (
                            pre_agg_succ,
                            pre_agg_fail,
                            pre_agg_routed,
                            pre_agg_quarantined,
                            pre_agg_coalesced,
                            pre_agg_forked,
                            pre_agg_expanded,
                            pre_agg_buffered,
                            pre_agg_routed_dests,
                        ) = self._check_aggregation_timeouts(
                            config=config,
                            processor=processor,
                            ctx=ctx,
                            pending_tokens=pending_tokens,
                            default_sink_name=default_sink_name,
                            agg_transform_lookup=agg_transform_lookup,
                        )
                        rows_succeeded += pre_agg_succ
                        rows_failed += pre_agg_fail
                        rows_routed += pre_agg_routed
                        rows_quarantined += pre_agg_quarantined
                        rows_coalesced += pre_agg_coalesced
                        rows_forked += pre_agg_forked
                        rows_expanded += pre_agg_expanded
                        rows_buffered += pre_agg_buffered
                        for dest, count in pre_agg_routed_dests.items():
                            routed_destinations[dest] = routed_destinations.get(dest, 0) + count

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
                                sink_name = default_sink_name
                                if result.token.branch_name is not None and result.token.branch_name in config.sinks:
                                    sink_name = result.token.branch_name

                                # NOTE: COMPLETED outcome is recorded by SinkExecutor.write() AFTER
                                # sink durability is achieved. Do NOT record here - that would violate
                                # Invariant 3: "COMPLETED implies token has completed sink node_state"

                                pending_tokens[sink_name].append((result.token, PendingOutcome(RowOutcome.COMPLETED)))
                            elif result.outcome == RowOutcome.ROUTED:
                                rows_routed += 1
                                # GateExecutor contract: ROUTED outcome always has sink_name set
                                if result.sink_name is None:
                                    raise RuntimeError("ROUTED outcome requires sink_name")
                                routed_destinations[result.sink_name] = routed_destinations.get(result.sink_name, 0) + 1
                                pending_tokens[result.sink_name].append((result.token, PendingOutcome(RowOutcome.ROUTED)))
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
                                rows_succeeded += 1
                                # NOTE: COMPLETED outcome is recorded by SinkExecutor.write() AFTER
                                # sink durability is achieved. Consumed tokens have COALESCED recorded
                                # by CoalesceExecutor. The merged token's lineage is in join_group_id.
                                pending_tokens[default_sink_name].append((result.token, PendingOutcome(RowOutcome.COMPLETED)))
                            elif result.outcome == RowOutcome.EXPANDED:
                                # Deaggregation parent token - children counted separately
                                rows_expanded += 1
                            elif result.outcome == RowOutcome.BUFFERED:
                                # Passthrough mode buffered token
                                rows_buffered += 1

                        # ─────────────────────────────────────────────────────────────────
                        # Check for timed-out coalesces after processing each row
                        # (BUG FIX: P1-2026-01-22 - check_timeouts was never called)
                        # ─────────────────────────────────────────────────────────────────
                        if coalesce_executor is not None:
                            total_steps = len(config.transforms) + len(config.gates)
                            for coalesce_name_str in coalesce_executor.get_registered_names():
                                coalesce_name = CoalesceName(coalesce_name_str)
                                coalesce_step = coalesce_step_map[coalesce_name]
                                timed_out = coalesce_executor.check_timeouts(
                                    coalesce_name=coalesce_name_str,
                                    step_in_pipeline=coalesce_step,
                                )
                                for outcome in timed_out:
                                    if outcome.merged_token is not None:
                                        rows_coalesced += 1
                                        # Check if merged token should continue downstream processing
                                        if coalesce_step < total_steps:
                                            # Continue processing through downstream nodes
                                            continuation_results = processor.process_token(
                                                token=outcome.merged_token,
                                                transforms=config.transforms,
                                                ctx=ctx,
                                                start_step=coalesce_step,
                                            )
                                            for cont_result in continuation_results:
                                                if cont_result.outcome == RowOutcome.COMPLETED:
                                                    rows_succeeded += 1
                                                    sink_name = default_sink_name
                                                    if (
                                                        cont_result.token.branch_name is not None
                                                        and cont_result.token.branch_name in config.sinks
                                                    ):
                                                        sink_name = cont_result.token.branch_name
                                                    pending_tokens[sink_name].append(
                                                        (cont_result.token, PendingOutcome(RowOutcome.COMPLETED))
                                                    )
                                                elif cont_result.outcome == RowOutcome.ROUTED:
                                                    rows_routed += 1
                                                    # sink_name is guaranteed non-None for ROUTED outcome
                                                    routed_sink = cont_result.sink_name or default_sink_name
                                                    routed_destinations[routed_sink] = routed_destinations.get(routed_sink, 0) + 1
                                                    pending_tokens[routed_sink].append(
                                                        (cont_result.token, PendingOutcome(RowOutcome.ROUTED))
                                                    )
                                                elif cont_result.outcome == RowOutcome.QUARANTINED:
                                                    rows_quarantined += 1
                                                elif cont_result.outcome == RowOutcome.FAILED:
                                                    rows_failed += 1
                                        else:
                                            # No downstream nodes - send directly to sink
                                            # NOTE: COMPLETED outcome is recorded by SinkExecutor.write()
                                            # AFTER sink durability is achieved.
                                            rows_succeeded += 1
                                            pending_tokens[default_sink_name].append(
                                                (outcome.merged_token, PendingOutcome(RowOutcome.COMPLETED))
                                            )
                                    elif outcome.failure_reason:
                                        rows_coalesce_failed += 1

                        # Emit progress every N rows or every M seconds (after outcome counters are updated)
                        # Hybrid timing: emit on first row, every 100 rows, or every 5 seconds
                        current_time = time.perf_counter()
                        time_since_last_progress = current_time - last_progress_time
                        should_emit = (
                            rows_processed == 1  # First row - immediate feedback
                            or rows_processed % progress_interval == 0  # Every 100 rows
                            or time_since_last_progress >= progress_time_interval  # Every 5 seconds
                        )
                        if should_emit:
                            elapsed = current_time - start_time
                            self._events.emit(
                                ProgressEvent(
                                    rows_processed=rows_processed,
                                    # Include routed rows in success count - they reached their destination
                                    rows_succeeded=rows_succeeded + rows_routed,
                                    rows_failed=rows_failed,
                                    rows_quarantined=rows_quarantined,
                                    elapsed_seconds=elapsed,
                                )
                            )
                            last_progress_time = current_time

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
                        (
                            agg_succeeded,
                            agg_failed,
                            agg_routed,
                            agg_quarantined,
                            agg_coalesced,
                            agg_forked,
                            agg_expanded,
                            agg_buffered,
                            agg_routed_dests,
                        ) = self._flush_remaining_aggregation_buffers(
                            config=config,
                            processor=processor,
                            ctx=ctx,
                            pending_tokens=pending_tokens,
                            default_sink_name=default_sink_name,
                            run_id=run_id,
                            recorder=recorder,
                            checkpoint=False,  # Checkpointing now happens after sink write
                            last_node_id=default_last_node_id,
                        )
                        rows_succeeded += agg_succeeded
                        rows_failed += agg_failed
                        rows_routed += agg_routed
                        rows_quarantined += agg_quarantined
                        rows_coalesced += agg_coalesced
                        rows_forked += agg_forked
                        rows_expanded += agg_expanded
                        rows_buffered += agg_buffered
                        for dest, count in agg_routed_dests.items():
                            routed_destinations[dest] = routed_destinations.get(dest, 0) + count

                    # Flush pending coalesce operations at end-of-source
                    if coalesce_executor is not None:
                        total_steps = len(config.transforms) + len(config.gates)
                        # Convert CoalesceName -> str for CoalesceExecutor API
                        flush_step_map = {str(name): step for name, step in coalesce_step_map.items()}
                        pending_outcomes = coalesce_executor.flush_pending(flush_step_map)

                        # Handle any merged tokens from flush
                        for outcome in pending_outcomes:
                            if outcome.merged_token is not None:
                                # Successful merge
                                rows_coalesced += 1
                                # Get the correct step for this coalesce
                                # outcome.coalesce_name is guaranteed non-None when merged_token is not None
                                coalesce_name = CoalesceName(outcome.coalesce_name)  # type: ignore[arg-type]
                                coalesce_step = coalesce_step_map[coalesce_name]
                                # Check if merged token should continue downstream processing
                                if coalesce_step < total_steps:
                                    # Continue processing through downstream nodes
                                    continuation_results = processor.process_token(
                                        token=outcome.merged_token,
                                        transforms=config.transforms,
                                        ctx=ctx,
                                        start_step=coalesce_step,
                                    )
                                    for cont_result in continuation_results:
                                        if cont_result.outcome == RowOutcome.COMPLETED:
                                            rows_succeeded += 1
                                            sink_name = default_sink_name
                                            if cont_result.token.branch_name is not None and cont_result.token.branch_name in config.sinks:
                                                sink_name = cont_result.token.branch_name
                                            pending_tokens[sink_name].append((cont_result.token, PendingOutcome(RowOutcome.COMPLETED)))
                                        elif cont_result.outcome == RowOutcome.ROUTED:
                                            rows_routed += 1
                                            # sink_name is guaranteed non-None for ROUTED outcome
                                            routed_sink = cont_result.sink_name or default_sink_name
                                            routed_destinations[routed_sink] = routed_destinations.get(routed_sink, 0) + 1
                                            pending_tokens[routed_sink].append((cont_result.token, PendingOutcome(RowOutcome.ROUTED)))
                                        elif cont_result.outcome == RowOutcome.QUARANTINED:
                                            rows_quarantined += 1
                                        elif cont_result.outcome == RowOutcome.FAILED:
                                            rows_failed += 1
                                else:
                                    # No downstream nodes - send directly to sink
                                    # NOTE: COMPLETED outcome is recorded by SinkExecutor.write()
                                    # AFTER sink durability is achieved.
                                    rows_succeeded += 1
                                    pending_tokens[default_sink_name].append((outcome.merged_token, PendingOutcome(RowOutcome.COMPLETED)))
                            elif outcome.failure_reason:
                                # Coalesce failed (quorum_not_met, incomplete_branches)
                                # Audit trail recorded by executor: each consumed token has
                                # node_state with status="failed" and error_json explaining why.
                                # Count failed coalesces for observability.
                                rows_coalesce_failed += 1

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
                            field_resolution_recorded = True

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

            for sink_name, token_outcome_pairs in pending_tokens.items():
                if token_outcome_pairs and sink_name in config.sinks:
                    sink = config.sinks[sink_name]
                    sink_node_id = sink_id_map[SinkName(sink_name)]

                    # Group tokens by pending_outcome for separate write() calls
                    # (sink_executor.write() takes a single PendingOutcome for all tokens in a batch)
                    from itertools import groupby

                    # Sort by (outcome, error_hash) to enable groupby (None sorts first)
                    # Fix: P1-2026-01-31 - PendingOutcome carries error_hash for QUARANTINED
                    def pending_sort_key(pair: tuple[TokenInfo, PendingOutcome | None]) -> tuple[bool, str, str]:
                        pending = pair[1]
                        if pending is None:
                            return (True, "", "")  # None sorts first
                        return (False, pending.outcome.value, pending.error_hash or "")

                    sorted_pairs = sorted(token_outcome_pairs, key=pending_sort_key)
                    for pending_outcome, group in groupby(sorted_pairs, key=lambda x: x[1]):
                        group_tokens = [token for token, _ in group]
                        sink_executor.write(
                            sink=sink,
                            tokens=group_tokens,
                            ctx=ctx,
                            step_in_pipeline=step,
                            sink_name=sink_name,
                            pending_outcome=pending_outcome,
                            on_token_written=checkpoint_after_sink(sink_node_id),
                        )

            # Emit final progress if we haven't emitted recently or row count not on interval
            # (RunSummary will show final summary regardless, but progress shows intermediate state)
            current_time = time.perf_counter()
            time_since_last_progress = current_time - last_progress_time
            # Emit if: not on progress_interval boundary OR >1s since last emission
            if rows_processed % progress_interval != 0 or time_since_last_progress >= 1.0:
                elapsed = current_time - start_time
                self._events.emit(
                    ProgressEvent(
                        rows_processed=rows_processed,
                        # Include routed rows in success count - they reached their destination
                        rows_succeeded=rows_succeeded + rows_routed,
                        rows_failed=rows_failed,
                        rows_quarantined=rows_quarantined,
                        elapsed_seconds=elapsed,
                    )
                )

            # PROCESS phase completed successfully
            self._events.emit(PhaseCompleted(phase=PipelinePhase.PROCESS, duration_seconds=time.perf_counter() - phase_start))

        finally:
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
            try:
                config.source.on_complete(ctx)
            except Exception as e:
                record_cleanup_error("source.on_complete", config.source.name, e)

            # Close source and all sinks
            # SinkProtocol requires close() - if missing, that's a bug
            try:
                config.source.close()
            except Exception as e:
                record_cleanup_error("source.close", config.source.name, e)
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

        # Clear graph after execution completes
        self._current_graph = None

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
            rows_coalesce_failed=rows_coalesce_failed,
            rows_expanded=rows_expanded,
            rows_buffered=rows_buffered,
            routed_destinations=routed_destinations,
        )

    def _export_landscape(
        self,
        run_id: str,
        settings: ElspethSettings,
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

    def _reconstruct_schema_from_json(self, schema_dict: dict[str, Any]) -> type:
        """Reconstruct Pydantic schema class from JSON schema dict.

        Handles complete Pydantic JSON schema including:
        - Primitive types: string, integer, number, boolean
        - datetime: string with format="date-time"
        - Decimal: anyOf with number/string (for precision preservation)
        - Arrays: type="array" with items schema
        - Nested objects: type="object" with properties schema

        Args:
            schema_dict: Pydantic JSON schema dict (from model_json_schema())

        Returns:
            Dynamically created Pydantic model class

        Raises:
            ValueError: If schema is malformed, empty, or contains unsupported types
        """

        from pydantic import create_model

        from elspeth.contracts import PluginSchema

        # Extract field definitions from Pydantic JSON schema
        # This is OUR data (from Landscape DB) - crash if malformed
        if "properties" not in schema_dict:
            raise ValueError(
                "Resume failed: Schema JSON has no 'properties' field. This indicates a malformed schema. Cannot reconstruct types."
            )
        properties = schema_dict["properties"]

        if not properties:
            raise ValueError(
                "Resume failed: Schema has zero fields defined. "
                "Cannot resume with empty schema - this would silently discard all row data. "
                "The original source schema must have at least one field."
            )

        # "required" is optional in JSON Schema spec - empty list is valid default
        if "required" in schema_dict:
            required_fields = set(schema_dict["required"])
        else:
            required_fields = set()

        # Build field definitions for create_model
        field_definitions: dict[str, Any] = {}

        for field_name, field_info in properties.items():
            # Determine Python type from JSON schema
            field_type = self._json_schema_to_python_type(field_name, field_info)

            # Handle optional vs required fields
            if field_name in required_fields:
                field_definitions[field_name] = (field_type, ...)  # Required field
            else:
                field_definitions[field_name] = (field_type, None)  # Optional field

        # Recreate the schema class dynamically
        return create_model("RestoredSourceSchema", __base__=PluginSchema, **field_definitions)

    def _json_schema_to_python_type(self, field_name: str, field_info: dict[str, Any]) -> type:
        """Map Pydantic JSON schema field to Python type.

        Handles Pydantic's type mapping including special cases:
        - datetime: {"type": "string", "format": "date-time"}
        - Decimal: {"anyOf": [{"type": "number"}, {"type": "string"}]}
        - list[T]: {"type": "array", "items": {...}}
        - dict: {"type": "object"} without properties

        Args:
            field_name: Field name (for error messages)
            field_info: JSON schema field definition

        Returns:
            Python type for Pydantic field

        Raises:
            ValueError: If field type is not supported (prevents silent degradation)
        """
        from datetime import datetime
        from decimal import Decimal

        # Check for datetime first (string with format annotation)
        # "format" is optional in JSON Schema, so check with "in" first
        if "type" in field_info and field_info["type"] == "string" and "format" in field_info and field_info["format"] == "date-time":
            return datetime

        # Check for Decimal (anyOf pattern)
        if "anyOf" in field_info:
            # Pydantic emits: {"anyOf": [{"type": "number"}, {"type": "string"}]}
            # This indicates Decimal (accepts both for parsing flexibility)
            any_of_types = field_info["anyOf"]
            # Only consider items that have "type" key, then access directly
            type_strs = {item["type"] for item in any_of_types if "type" in item}
            if {"number", "string"}.issubset(type_strs):
                return Decimal

        # Get basic type - required for all non-anyOf fields
        if "type" not in field_info:
            raise ValueError(
                f"Resume failed: Field '{field_name}' has no 'type' in schema. "
                f"Schema definition: {field_info}. "
                f"Cannot determine Python type for field."
            )
        field_type_str = field_info["type"]

        # Handle array types
        if field_type_str == "array":
            # "items" is optional in JSON Schema arrays
            if "items" not in field_info:
                # Generic list without item type constraint
                return list
            # items_schema = field_info["items"]  # Available if needed for recursive handling
            # For typed arrays, we'd need recursive handling
            # For now, return list (Pydantic will validate items at parse time)
            return list

        # Handle nested object types
        if field_type_str == "object":
            # Generic dict (no specific structure)
            return dict

        # Handle primitive types
        primitive_type_map = {
            "string": str,
            "integer": int,
            "number": float,
            "boolean": bool,
        }

        if field_type_str in primitive_type_map:
            return primitive_type_map[field_type_str]

        # Unknown type - CRASH instead of silent degradation
        raise ValueError(
            f"Resume failed: Field '{field_name}' has unsupported type '{field_type_str}'. "
            f"Supported types: string, integer, number, boolean, date-time, Decimal, array, object. "
            f"Schema definition: {field_info}. "
            f"This is a bug in schema reconstruction - please report this."
        )

    def resume(
        self,
        resume_point: ResumePoint,
        config: PipelineConfig,
        graph: ExecutionGraph,
        *,
        payload_store: PayloadStore,
        settings: ElspethSettings | None = None,
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

        # 1. Handle incomplete batches
        self._handle_incomplete_batches(recorder, run_id)

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
        schema_dict = json.loads(source_schema_json)
        source_schema_class = self._reconstruct_schema_from_json(schema_dict)

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

        # 5. Process unprocessed rows
        result = self._process_resumed_rows(
            recorder=recorder,
            run_id=run_id,
            config=config,
            graph=graph,
            unprocessed_rows=unprocessed_rows,
            restored_aggregation_state=restored_state,
            settings=settings,
            payload_store=payload_store,
        )

        # 6. Complete the run with reproducibility grade
        recorder.finalize_run(run_id, status=RunStatus.COMPLETED)
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
        settings: ElspethSettings | None = None,
        *,
        payload_store: PayloadStore,
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
        default_sink_name = graph.get_default_sink()

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

        # Validate source quarantine destination
        self._validate_source_quarantine_destination(
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
            retry_manager = RetryManager(RuntimeRetryConfig.from_settings(settings.retry))

        # Create coalesce executor if config has coalesce settings
        from elspeth.engine.coalesce_executor import CoalesceExecutor
        from elspeth.engine.tokens import TokenManager

        coalesce_executor: CoalesceExecutor | None = None
        branch_to_coalesce: dict[BranchName, CoalesceName] = {}

        if settings is not None and settings.coalesce:
            branch_to_coalesce = graph.get_branch_to_coalesce_map()
            token_manager = TokenManager(recorder)

            coalesce_executor = CoalesceExecutor(
                recorder=recorder,
                span_factory=self._span_factory,
                token_manager=token_manager,
                run_id=run_id,
                clock=self._clock,
            )

            for coalesce_settings in settings.coalesce:
                coalesce_node_id = coalesce_id_map[CoalesceName(coalesce_settings.name)]
                coalesce_executor.register_coalesce(coalesce_settings, coalesce_node_id)

        # Compute coalesce step positions FROM GRAPH TOPOLOGY (same as main run path)
        coalesce_step_map = self._compute_coalesce_step_map(graph, config, settings)

        # Convert aggregation_settings keys from str to NodeID
        typed_aggregation_settings: dict[NodeID, AggregationSettings] = {NodeID(k): v for k, v in config.aggregation_settings.items()}

        # Convert restored_aggregation_state keys from str to NodeID
        typed_restored_state: dict[NodeID, dict[str, Any]] = {NodeID(k): v for k, v in restored_aggregation_state.items()}

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
            aggregation_settings=typed_aggregation_settings,
            retry_manager=retry_manager,
            coalesce_executor=coalesce_executor,
            coalesce_node_ids=coalesce_id_map,
            branch_to_coalesce=branch_to_coalesce,
            coalesce_step_map=coalesce_step_map,
            restored_aggregation_state=typed_restored_state,
            payload_store=payload_store,
            clock=self._clock,
            max_workers=self._concurrency_config.max_workers if self._concurrency_config else None,
            telemetry_manager=self._telemetry,
        )

        # Process rows - Buffer TOKENS
        from elspeth.engine.executors import SinkExecutor

        rows_processed = 0
        rows_succeeded = 0
        rows_failed = 0
        rows_routed = 0
        rows_quarantined = 0
        rows_forked = 0
        rows_coalesced = 0
        rows_coalesce_failed = 0
        rows_expanded = 0
        rows_buffered = 0
        routed_destinations: dict[str, int] = {}  # Track routing destinations
        # Track (token, pending_outcome) pairs for deferred outcome recording
        # Outcomes are recorded by SinkExecutor.write() AFTER sink durability is achieved
        # Fix: P1-2026-01-31 - use PendingOutcome to carry error_hash for QUARANTINED
        pending_tokens: dict[str, list[tuple[TokenInfo, PendingOutcome | None]]] = {name: [] for name in config.sinks}

        # Pre-compute aggregation transform lookup for O(1) access per timeout check
        # NOTE: Steps are 0-indexed here; handle_timeout_flush converts to 1-indexed
        # for audit recording (to avoid node_state step_index collisions)
        agg_transform_lookup: dict[str, tuple[TransformProtocol, int]] = {}
        if config.aggregation_settings:
            for i, t in enumerate(config.transforms):
                if isinstance(t, TransformProtocol) and t.is_batch_aware and t.node_id in config.aggregation_settings:
                    agg_transform_lookup[t.node_id] = (t, i)  # 0-indexed, converted in handle_timeout_flush

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

                # ─────────────────────────────────────────────────────────────────
                # Check for timed-out aggregations BEFORE processing this row
                # (BUG FIX: P1-2026-01-22 - ensures timeout flushes OLD batch)
                # ─────────────────────────────────────────────────────────────────
                (
                    pre_agg_succ,
                    pre_agg_fail,
                    pre_agg_routed,
                    pre_agg_quarantined,
                    pre_agg_coalesced,
                    pre_agg_forked,
                    pre_agg_expanded,
                    pre_agg_buffered,
                    pre_agg_routed_dests,
                ) = self._check_aggregation_timeouts(
                    config=config,
                    processor=processor,
                    ctx=ctx,
                    pending_tokens=pending_tokens,
                    default_sink_name=default_sink_name,
                    agg_transform_lookup=agg_transform_lookup,
                )
                rows_succeeded += pre_agg_succ
                rows_failed += pre_agg_fail
                rows_routed += pre_agg_routed
                rows_quarantined += pre_agg_quarantined
                rows_coalesced += pre_agg_coalesced
                rows_forked += pre_agg_forked
                rows_expanded += pre_agg_expanded
                rows_buffered += pre_agg_buffered
                for dest, count in pre_agg_routed_dests.items():
                    routed_destinations[dest] = routed_destinations.get(dest, 0) + count

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
                        sink_name = default_sink_name
                        if result.token.branch_name is not None and result.token.branch_name in config.sinks:
                            sink_name = result.token.branch_name

                        # NOTE: COMPLETED outcome is recorded by SinkExecutor.write() AFTER
                        # sink durability is achieved. Do NOT record here - that would violate
                        # Invariant 3: "COMPLETED implies token has completed sink node_state"

                        pending_tokens[sink_name].append((result.token, PendingOutcome(RowOutcome.COMPLETED)))
                    elif result.outcome == RowOutcome.ROUTED:
                        rows_routed += 1
                        if result.sink_name is None:
                            raise RuntimeError("ROUTED outcome requires sink_name")
                        routed_destinations[result.sink_name] = routed_destinations.get(result.sink_name, 0) + 1
                        pending_tokens[result.sink_name].append((result.token, PendingOutcome(RowOutcome.ROUTED)))
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
                        rows_succeeded += 1
                        # NOTE: COMPLETED outcome is recorded by SinkExecutor.write() AFTER
                        # sink durability is achieved.
                        pending_tokens[default_sink_name].append((result.token, PendingOutcome(RowOutcome.COMPLETED)))
                    elif result.outcome == RowOutcome.EXPANDED:
                        rows_expanded += 1
                    elif result.outcome == RowOutcome.BUFFERED:
                        rows_buffered += 1

                # ─────────────────────────────────────────────────────────────────
                # Check for timed-out coalesces after processing each row
                # (BUG FIX: P1-2026-01-22 - check_timeouts was never called)
                # ─────────────────────────────────────────────────────────────────
                if coalesce_executor is not None:
                    total_steps = len(config.transforms) + len(config.gates)
                    for coalesce_name_str in coalesce_executor.get_registered_names():
                        coalesce_name = CoalesceName(coalesce_name_str)
                        coalesce_step = coalesce_step_map[coalesce_name]
                        timed_out = coalesce_executor.check_timeouts(
                            coalesce_name=coalesce_name_str,
                            step_in_pipeline=coalesce_step,
                        )
                        for outcome in timed_out:
                            if outcome.merged_token is not None:
                                rows_coalesced += 1
                                # Check if merged token should continue downstream processing
                                if coalesce_step < total_steps:
                                    # Continue processing through downstream nodes
                                    continuation_results = processor.process_token(
                                        token=outcome.merged_token,
                                        transforms=config.transforms,
                                        ctx=ctx,
                                        start_step=coalesce_step,
                                    )
                                    for cont_result in continuation_results:
                                        if cont_result.outcome == RowOutcome.COMPLETED:
                                            sink_name = default_sink_name
                                            if cont_result.token.branch_name is not None and cont_result.token.branch_name in config.sinks:
                                                sink_name = cont_result.token.branch_name
                                            pending_tokens[sink_name].append((cont_result.token, PendingOutcome(RowOutcome.COMPLETED)))
                                        elif cont_result.outcome == RowOutcome.ROUTED:
                                            rows_routed += 1
                                            # sink_name is guaranteed non-None for ROUTED outcome
                                            routed_sink = cont_result.sink_name or default_sink_name
                                            routed_destinations[routed_sink] = routed_destinations.get(routed_sink, 0) + 1
                                            pending_tokens[routed_sink].append((cont_result.token, PendingOutcome(RowOutcome.ROUTED)))
                                        elif cont_result.outcome == RowOutcome.QUARANTINED:
                                            rows_quarantined += 1
                                        elif cont_result.outcome == RowOutcome.FAILED:
                                            rows_failed += 1
                                else:
                                    # No downstream nodes - send directly to sink
                                    # NOTE: COMPLETED outcome is recorded by SinkExecutor.write()
                                    # AFTER sink durability is achieved.
                                    pending_tokens[default_sink_name].append((outcome.merged_token, PendingOutcome(RowOutcome.COMPLETED)))
                            elif outcome.failure_reason:
                                rows_coalesce_failed += 1

            # ─────────────────────────────────────────────────────────────────
            # CRITICAL: Flush remaining aggregation buffers at end-of-source
            # ─────────────────────────────────────────────────────────────────
            if config.aggregation_settings:
                (
                    agg_succeeded,
                    agg_failed,
                    agg_routed,
                    agg_quarantined,
                    agg_coalesced,
                    agg_forked,
                    agg_expanded,
                    agg_buffered,
                    agg_routed_dests,
                ) = self._flush_remaining_aggregation_buffers(
                    config=config,
                    processor=processor,
                    ctx=ctx,
                    pending_tokens=pending_tokens,
                    default_sink_name=default_sink_name,
                    run_id=run_id,
                    recorder=recorder,
                    checkpoint=False,  # No checkpointing during resume
                )
                rows_succeeded += agg_succeeded
                rows_failed += agg_failed
                rows_routed += agg_routed
                rows_quarantined += agg_quarantined
                rows_coalesced += agg_coalesced
                rows_forked += agg_forked
                rows_expanded += agg_expanded
                rows_buffered += agg_buffered
                for dest, count in agg_routed_dests.items():
                    routed_destinations[dest] = routed_destinations.get(dest, 0) + count

            # Flush pending coalesce operations
            if coalesce_executor is not None:
                total_steps = len(config.transforms) + len(config.gates)
                # Convert CoalesceName -> str for CoalesceExecutor API
                flush_step_map = {str(name): step for name, step in coalesce_step_map.items()}
                pending_outcomes = coalesce_executor.flush_pending(flush_step_map)

                for outcome in pending_outcomes:
                    if outcome.merged_token is not None:
                        rows_coalesced += 1
                        # Get the correct step for this coalesce
                        # outcome.coalesce_name is guaranteed non-None when merged_token is not None
                        coalesce_name = CoalesceName(outcome.coalesce_name)  # type: ignore[arg-type]
                        coalesce_step = coalesce_step_map[coalesce_name]
                        # Check if merged token should continue downstream processing
                        if coalesce_step < total_steps:
                            # Continue processing through downstream nodes
                            continuation_results = processor.process_token(
                                token=outcome.merged_token,
                                transforms=config.transforms,
                                ctx=ctx,
                                start_step=coalesce_step,
                            )
                            for cont_result in continuation_results:
                                if cont_result.outcome == RowOutcome.COMPLETED:
                                    rows_succeeded += 1
                                    sink_name = default_sink_name
                                    if cont_result.token.branch_name is not None and cont_result.token.branch_name in config.sinks:
                                        sink_name = cont_result.token.branch_name
                                    pending_tokens[sink_name].append((cont_result.token, PendingOutcome(RowOutcome.COMPLETED)))
                                elif cont_result.outcome == RowOutcome.ROUTED:
                                    rows_routed += 1
                                    # sink_name is guaranteed non-None for ROUTED outcome
                                    routed_sink = cont_result.sink_name or default_sink_name
                                    routed_destinations[routed_sink] = routed_destinations.get(routed_sink, 0) + 1
                                    pending_tokens[routed_sink].append((cont_result.token, PendingOutcome(RowOutcome.ROUTED)))
                                elif cont_result.outcome == RowOutcome.QUARANTINED:
                                    rows_quarantined += 1
                                elif cont_result.outcome == RowOutcome.FAILED:
                                    rows_failed += 1
                        else:
                            # No downstream nodes - send directly to sink
                            # NOTE: COMPLETED outcome is recorded by SinkExecutor.write()
                            # AFTER sink durability is achieved.
                            rows_succeeded += 1
                            pending_tokens[default_sink_name].append((outcome.merged_token, PendingOutcome(RowOutcome.COMPLETED)))
                    elif outcome.failure_reason:
                        # Coalesce failed - audit trail already recorded by executor
                        rows_coalesce_failed += 1

            # Write to sinks using SinkExecutor
            sink_executor = SinkExecutor(recorder, self._span_factory, run_id)
            step = len(config.transforms) + len(config.gates) + 1

            for sink_name, token_outcome_pairs in pending_tokens.items():
                if token_outcome_pairs and sink_name in config.sinks:
                    sink = config.sinks[sink_name]

                    # Group tokens by pending_outcome for separate write() calls
                    # Fix: P1-2026-01-31 - PendingOutcome carries error_hash for QUARANTINED
                    from itertools import groupby

                    def pending_sort_key(pair: tuple[TokenInfo, PendingOutcome | None]) -> tuple[bool, str, str]:
                        pending = pair[1]
                        if pending is None:
                            return (True, "", "")
                        return (False, pending.outcome.value, pending.error_hash or "")

                    sorted_pairs = sorted(token_outcome_pairs, key=pending_sort_key)
                    for pending_outcome, group in groupby(sorted_pairs, key=lambda x: x[1]):
                        group_tokens = [token for token, _ in group]
                        sink_executor.write(
                            sink=sink,
                            tokens=group_tokens,
                            ctx=ctx,
                            step_in_pipeline=step,
                            sink_name=sink_name,
                            pending_outcome=pending_outcome,
                        )

        finally:
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

            # Close all transforms (release resources - file handles, connections, etc.)
            # Mirrors _cleanup_transforms() pattern from _execute_run()
            for transform in config.transforms:
                try:
                    transform.close()
                except Exception as e:
                    record_cleanup_error("transform.close", transform.name, e)

            # Close all sinks (NOT source - wasn't opened)
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

        # Clear graph after execution completes
        self._current_graph = None

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
            rows_coalesce_failed=rows_coalesce_failed,
            rows_expanded=rows_expanded,
            rows_buffered=rows_buffered,
            routed_destinations=routed_destinations,
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
                recorder.update_batch_status(batch.batch_id, BatchStatus.FAILED)
                recorder.retry_batch(batch.batch_id)
            elif batch.status == BatchStatus.FAILED:
                # Previous failure, retry
                recorder.retry_batch(batch.batch_id)
            # DRAFT batches continue normally (collection resumes)

    def _find_aggregation_transform(
        self,
        config: PipelineConfig,
        agg_node_id_str: str,
        agg_name: str,
    ) -> tuple[TransformProtocol, int]:
        """Find the batch-aware transform for an aggregation node.

        Args:
            config: Pipeline configuration with transforms
            agg_node_id_str: The aggregation node ID as string
            agg_name: Human-readable aggregation name (for error messages)

        Returns:
            Tuple of (transform, step_index) where step_index is the 0-indexed
            position in the pipeline

        Raises:
            RuntimeError: If no batch-aware transform found for the aggregation
        """
        agg_transform: TransformProtocol | None = None
        agg_step = len(config.transforms)

        for i, t in enumerate(config.transforms):
            if isinstance(t, TransformProtocol) and t.node_id == agg_node_id_str and t.is_batch_aware:
                agg_transform = t
                agg_step = i
                break

        if agg_transform is None:
            raise RuntimeError(
                f"No batch-aware transform found for aggregation '{agg_name}' "
                f"(node_id={agg_node_id_str}). This indicates a bug in graph construction "
                f"or pipeline configuration. "
                f"Available transforms: {[t.node_id for t in config.transforms]}"
            )

        return agg_transform, agg_step

    def _check_aggregation_timeouts(
        self,
        config: PipelineConfig,
        processor: RowProcessor,
        ctx: PluginContext,
        pending_tokens: dict[str, list[tuple[TokenInfo, PendingOutcome | None]]],
        default_sink_name: str,
        agg_transform_lookup: dict[str, tuple[TransformProtocol, int]] | None = None,
    ) -> tuple[int, int, int, int, int, int, int, int, dict[str, int]]:
        """Check and flush any aggregations whose timeout has expired.

        Called BEFORE processing each row to ensure timeouts fire during active
        processing, not just at end-of-source. Checking BEFORE buffering ensures
        timed-out batches don't include the newly arriving row.

        Bug fix: P1-2026-01-22-aggregation-timeout-idle-never-fires
        Before this fix, should_flush() was only called from buffer_row(),
        meaning timeouts never fired during idle periods between rows.

        KNOWN LIMITATION (True Idle):
        Timeouts fire when the next row arrives, not during "true idle" periods.
        If no rows arrive, buffered data will not flush until either:
        1. A new row arrives (triggering this timeout check), or
        2. The source completes (triggering _flush_remaining_aggregation_buffers)

        Example: If timeout_seconds=5 and rows stop arriving at T=10, the batch
        won't flush until either a new row arrives or the source ends. For
        streaming sources that may never end, consider using count triggers or
        implementing periodic polling at the source level.

        Args:
            config: Pipeline configuration with aggregation_settings
            processor: RowProcessor with public aggregation timeout API
            ctx: Plugin context for transform execution
            pending_tokens: Dict of sink_name -> tokens to append results to
            default_sink_name: Default sink for aggregation output
            agg_transform_lookup: Pre-computed dict mapping node_id_str -> (transform, step).
                If None, lookup is computed on each call (less efficient).

        Returns:
            Tuple of (rows_succeeded, rows_failed, rows_routed, rows_quarantined, rows_coalesced,
                      rows_forked, rows_expanded, rows_buffered, routed_destinations)
        """
        rows_succeeded = 0
        rows_failed = 0
        rows_routed = 0
        rows_quarantined = 0
        rows_coalesced = 0
        rows_forked = 0
        rows_expanded = 0
        rows_buffered = 0
        routed_destinations: dict[str, int] = {}

        for agg_node_id_str, agg_settings in config.aggregation_settings.items():
            agg_node_id = NodeID(agg_node_id_str)

            # Use public facade method to check timeout (no private member access)
            should_flush, trigger_type = processor.check_aggregation_timeout(agg_node_id)

            if not should_flush:
                continue

            # Skip if not a timeout trigger - count triggers are handled in buffer_row
            if trigger_type != TriggerType.TIMEOUT:
                continue

            # Check if there are buffered rows
            buffered_count = processor.get_aggregation_buffer_count(agg_node_id)
            if buffered_count == 0:
                continue

            # Get transform and step from pre-computed lookup (O(1)) or compute (O(n))
            if agg_transform_lookup and agg_node_id_str in agg_transform_lookup:
                agg_transform, agg_step = agg_transform_lookup[agg_node_id_str]
            else:
                # Fallback: use helper method if lookup not provided
                agg_transform, agg_step = self._find_aggregation_transform(config, agg_node_id_str, agg_settings.name)

            # Use handle_timeout_flush for proper output_mode handling
            # This correctly routes through remaining transforms and gates
            total_steps = len(config.transforms)
            completed_results, work_items = processor.handle_timeout_flush(
                node_id=agg_node_id,
                transform=agg_transform,
                ctx=ctx,
                step=agg_step,
                total_steps=total_steps,
                trigger_type=TriggerType.TIMEOUT,
            )

            # Handle completed results (no more transforms - go to sink)
            for result in completed_results:
                if result.outcome == RowOutcome.FAILED:
                    rows_failed += 1
                else:
                    # Route to appropriate sink based on branch_name if set
                    sink_name = result.token.branch_name or default_sink_name
                    if sink_name not in pending_tokens:
                        sink_name = default_sink_name
                    pending_tokens[sink_name].append((result.token, PendingOutcome(result.outcome)))
                    rows_succeeded += 1

            # Process work items through remaining transforms
            # These tokens need to continue through the pipeline
            for work_item in work_items:
                # Determine start_step: if coalesce is set, use it directly
                # Otherwise, add 1 to current position to get next transform
                if work_item.coalesce_at_step is not None:
                    continuation_start = work_item.coalesce_at_step
                else:
                    continuation_start = work_item.start_step + 1
                downstream_results = processor.process_token(
                    token=work_item.token,
                    transforms=config.transforms,
                    ctx=ctx,
                    start_step=continuation_start,
                    coalesce_at_step=work_item.coalesce_at_step,
                    coalesce_name=work_item.coalesce_name,
                )

                for result in downstream_results:
                    if result.outcome == RowOutcome.FAILED:
                        rows_failed += 1
                    elif result.outcome == RowOutcome.COMPLETED:
                        # Route to appropriate sink
                        sink_name = result.token.branch_name or default_sink_name
                        if sink_name not in pending_tokens:
                            sink_name = default_sink_name
                        pending_tokens[sink_name].append((result.token, PendingOutcome(result.outcome)))
                        rows_succeeded += 1
                    elif result.outcome == RowOutcome.ROUTED:
                        # Gate routed to named sink - MUST enqueue or row is lost
                        # GateExecutor contract: ROUTED outcome always has sink_name set
                        rows_routed += 1
                        routed_sink = result.sink_name or default_sink_name
                        routed_destinations[routed_sink] = routed_destinations.get(routed_sink, 0) + 1
                        pending_tokens[routed_sink].append((result.token, PendingOutcome(RowOutcome.ROUTED)))
                    elif result.outcome == RowOutcome.QUARANTINED:
                        # Row quarantined by downstream transform - already recorded
                        rows_quarantined += 1
                    elif result.outcome == RowOutcome.COALESCED:
                        # Merged token from terminal coalesce - route to output sink
                        # This handles the case where coalesce is the last step
                        rows_coalesced += 1
                        rows_succeeded += 1
                        pending_tokens[default_sink_name].append((result.token, PendingOutcome(RowOutcome.COMPLETED)))
                    elif result.outcome == RowOutcome.FORKED:
                        # Parent token split into multiple paths - children counted separately
                        rows_forked += 1
                    elif result.outcome == RowOutcome.EXPANDED:
                        # Deaggregation parent token - children counted separately
                        rows_expanded += 1
                    elif result.outcome == RowOutcome.BUFFERED:
                        # Passthrough mode buffered token (into downstream aggregation)
                        rows_buffered += 1
                    # CONSUMED_IN_BATCH is handled within process_token

        return (
            rows_succeeded,
            rows_failed,
            rows_routed,
            rows_quarantined,
            rows_coalesced,
            rows_forked,
            rows_expanded,
            rows_buffered,
            routed_destinations,
        )

    def _flush_remaining_aggregation_buffers(
        self,
        config: PipelineConfig,
        processor: RowProcessor,
        ctx: PluginContext,
        pending_tokens: dict[str, list[tuple[TokenInfo, PendingOutcome | None]]],
        default_sink_name: str,
        run_id: str,
        recorder: LandscapeRecorder,
        checkpoint: bool = True,
        last_node_id: str | None = None,
    ) -> tuple[int, int, int, int, int, int, int, int, dict[str, int]]:
        """Flush remaining aggregation buffers at end-of-source.

        Without this, rows buffered but not yet flushed (e.g., 50 rows
        when trigger is count=100) would be silently lost.

        Uses handle_timeout_flush with END_OF_SOURCE trigger to properly handle
        all output_mode semantics (single, passthrough, transform) and route
        tokens through remaining transforms if any exist after the aggregation.

        Args:
            config: Pipeline configuration with aggregation_settings
            processor: RowProcessor with public aggregation facades
            ctx: Plugin context for transform execution
            pending_tokens: Dict of sink_name -> tokens to append results to
            default_sink_name: Default sink for aggregation output
            run_id: Current run ID (for checkpointing)
            recorder: LandscapeRecorder for recording outcomes
            checkpoint: Whether to create checkpoints for flushed tokens
                       (True for _execute_run, False for _process_resumed_rows)
            last_node_id: Node ID to use for checkpointing (required if checkpoint=True)

        Returns:
            Tuple of (rows_succeeded, rows_failed, rows_routed, rows_quarantined, rows_coalesced,
                      rows_forked, rows_expanded, rows_buffered, routed_destinations)

        Raises:
            RuntimeError: If no batch-aware transform found for an aggregation
                         (indicates bug in graph construction or pipeline config)
        """
        rows_succeeded = 0
        rows_failed = 0
        rows_routed = 0
        rows_quarantined = 0
        rows_coalesced = 0
        rows_forked = 0
        rows_expanded = 0
        rows_buffered = 0
        routed_destinations: dict[str, int] = {}
        total_steps = len(config.transforms)

        for agg_node_id_str, agg_settings in config.aggregation_settings.items():
            agg_node_id = NodeID(agg_node_id_str)

            # Use public facade (not private member)
            buffered_count = processor.get_aggregation_buffer_count(agg_node_id)
            if buffered_count == 0:
                continue

            # Use helper method for transform lookup
            agg_transform, agg_step = self._find_aggregation_transform(config, agg_node_id_str, agg_settings.name)

            # Use handle_timeout_flush with END_OF_SOURCE trigger
            # This properly handles output_mode and routes through remaining transforms
            completed_results, work_items = processor.handle_timeout_flush(
                node_id=agg_node_id,
                transform=agg_transform,
                ctx=ctx,
                step=agg_step,
                total_steps=total_steps,
                trigger_type=TriggerType.END_OF_SOURCE,
            )

            # Handle completed results (terminal tokens - go to sink)
            for result in completed_results:
                if result.outcome == RowOutcome.FAILED:
                    rows_failed += 1
                else:
                    # Route to appropriate sink based on branch_name if set
                    sink_name = result.token.branch_name or default_sink_name
                    if sink_name not in pending_tokens:
                        sink_name = default_sink_name
                    pending_tokens[sink_name].append((result.token, PendingOutcome(result.outcome)))
                    rows_succeeded += 1

                    # Checkpoint if enabled
                    if checkpoint and last_node_id is not None:
                        self._maybe_checkpoint(
                            run_id=run_id,
                            token_id=result.token.token_id,
                            node_id=last_node_id,
                        )

            # Process work items through remaining transforms
            # These tokens need to continue through the pipeline
            for work_item in work_items:
                # Determine start_step: if coalesce is set, use it directly
                # Otherwise, add 1 to current position to get next transform
                if work_item.coalesce_at_step is not None:
                    continuation_start = work_item.coalesce_at_step
                else:
                    continuation_start = work_item.start_step + 1
                downstream_results = processor.process_token(
                    token=work_item.token,
                    transforms=config.transforms,
                    ctx=ctx,
                    start_step=continuation_start,
                    coalesce_at_step=work_item.coalesce_at_step,
                    coalesce_name=work_item.coalesce_name,
                )

                for result in downstream_results:
                    if result.outcome == RowOutcome.FAILED:
                        rows_failed += 1
                    elif result.outcome == RowOutcome.COMPLETED:
                        # Route to appropriate sink
                        sink_name = result.token.branch_name or default_sink_name
                        if sink_name not in pending_tokens:
                            sink_name = default_sink_name
                        pending_tokens[sink_name].append((result.token, PendingOutcome(result.outcome)))
                        rows_succeeded += 1

                        # Checkpoint if enabled
                        if checkpoint and last_node_id is not None:
                            self._maybe_checkpoint(
                                run_id=run_id,
                                token_id=result.token.token_id,
                                node_id=last_node_id,
                            )
                    elif result.outcome == RowOutcome.ROUTED:
                        # Gate routed to named sink - MUST enqueue or row is lost
                        # GateExecutor contract: ROUTED outcome always has sink_name set
                        rows_routed += 1
                        routed_sink = result.sink_name or default_sink_name
                        routed_destinations[routed_sink] = routed_destinations.get(routed_sink, 0) + 1
                        pending_tokens[routed_sink].append((result.token, PendingOutcome(RowOutcome.ROUTED)))

                        # Checkpoint if enabled
                        if checkpoint and last_node_id is not None:
                            self._maybe_checkpoint(
                                run_id=run_id,
                                token_id=result.token.token_id,
                                node_id=last_node_id,
                            )
                    elif result.outcome == RowOutcome.QUARANTINED:
                        # Row quarantined by downstream transform - already recorded
                        rows_quarantined += 1
                    elif result.outcome == RowOutcome.COALESCED:
                        # Merged token from terminal coalesce - route to output sink
                        # This handles the case where coalesce is the last step
                        rows_coalesced += 1
                        rows_succeeded += 1
                        pending_tokens[default_sink_name].append((result.token, PendingOutcome(RowOutcome.COMPLETED)))

                        # Checkpoint if enabled
                        if checkpoint and last_node_id is not None:
                            self._maybe_checkpoint(
                                run_id=run_id,
                                token_id=result.token.token_id,
                                node_id=last_node_id,
                            )
                    elif result.outcome == RowOutcome.FORKED:
                        # Parent token split into multiple paths - children counted separately
                        rows_forked += 1
                    elif result.outcome == RowOutcome.EXPANDED:
                        # Deaggregation parent token - children counted separately
                        rows_expanded += 1
                    elif result.outcome == RowOutcome.BUFFERED:
                        # Passthrough mode buffered token (into downstream aggregation)
                        rows_buffered += 1
                    # CONSUMED_IN_BATCH is handled within process_token

        return (
            rows_succeeded,
            rows_failed,
            rows_routed,
            rows_quarantined,
            rows_coalesced,
            rows_forked,
            rows_expanded,
            rows_buffered,
            routed_destinations,
        )
