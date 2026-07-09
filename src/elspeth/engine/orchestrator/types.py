"""Pipeline configuration and result types.

These types define the interface for pipeline execution:
- PipelineConfig: Input configuration for a run
- RunResult: Output statistics from a run
- RouteValidationError: Configuration validation failure
- AggregationFlushResult: Result of flushing aggregation buffers

IMPORTANT: Import Cycle Prevention
----------------------------------
The canonical definitions in this module are leaf data definitions - they must
not import runtime orchestration helpers such as validation.py, export.py,
aggregation.py, or core.py.

Pre-1.0 compatibility re-exports remain available for older callers. New code
should import internal run-state, value-source, and port types from their
canonical modules instead of adding new definitions here.

Keep the public config/result/counter surface here with minimal dependencies.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import TYPE_CHECKING, Any

from elspeth.contracts import RunStatus
from elspeth.contracts.freeze import deep_thaw, freeze_fields
from elspeth.contracts.run_result import RunResult as RunResult  # re-exported
from elspeth.engine.orchestrator.plugin_types import RowPlugin
from elspeth.engine.orchestrator.ports import TelemetryManagerProtocol
from elspeth.engine.orchestrator.run_state import (
    AggNodeEntry,
    GraphArtifacts,
    LoopContext,
    LoopResult,
    PendingTokenMap,
    ResumeState,
    RunContext,
    _RunFailedWithPartialResultError,
)
from elspeth.engine.orchestrator.value_source_validation import ValueSourceFinding, ValueSourceValidationError

if TYPE_CHECKING:
    from elspeth.contracts import SinkProtocol, SourceProtocol
    from elspeth.core.config import AggregationSettings, CoalesceSettings, GateSettings

__all__ = [
    "AggNodeEntry",
    "AggregationFlushResult",
    "ExecutionCounters",
    "GraphArtifacts",
    "LoopContext",
    "LoopResult",
    "PendingTokenMap",
    "PipelineConfig",
    "ResumeState",
    "RouteValidationError",
    "RowPlugin",
    "RunContext",
    "RunResult",
    "TelemetryManagerProtocol",
    "ValueSourceFinding",
    "ValueSourceValidationError",
    "_RunFailedWithPartialResultError",
]


@dataclass(frozen=True, slots=True)
class PipelineConfig:
    """Configuration for a pipeline run.

    All plugin fields are now properly typed for IDE support and
    static type checking. Frozen after construction — pipeline
    configuration must not change during execution.

    The ``frozen=True`` decorator prevents field reassignment after
    construction, ensuring pipeline config is immutable during a run.

    Per ADR-025 Decision §1, the pipeline source surface is plural by
    contract and by code. Callers iterate ``sources.items()`` and pass
    the active ``SourceProtocol`` as an explicit parameter to each
    per-source method; PipelineConfig itself no longer exposes a
    "current source" pseudo-attribute.

    Attributes:
        sources: Source plugin instances keyed by stable source name (non-empty).
        transforms: Transform plugin instances (processed in DAG order)
        sinks: Dict of sink_name -> sink plugin instance
        config: Additional run configuration
        gates: Config-driven gates (processed AFTER transforms, BEFORE sinks)
        aggregation_settings: Dict of node_id -> AggregationSettings
        coalesce_settings: Coalesce configurations for merging fork paths
    """

    sources: Mapping[str, SourceProtocol]
    transforms: Sequence[RowPlugin]
    sinks: Mapping[str, SinkProtocol]
    config: Mapping[str, Any] = field(default_factory=dict)
    gates: Sequence[GateSettings] = field(default_factory=list)
    aggregation_settings: Mapping[str, AggregationSettings] = field(default_factory=dict)
    coalesce_settings: Sequence[CoalesceSettings] = field(default_factory=list)

    def __post_init__(self) -> None:
        from elspeth.contracts.errors import OrchestrationInvariantError

        if not self.sinks:
            raise OrchestrationInvariantError("PipelineConfig requires at least one sink")
        if not self.sources:
            raise OrchestrationInvariantError("PipelineConfig requires at least one source")
        # Freeze mutable container fields. freeze_fields deep-freezes recursively,
        # converting nested dicts/lists to MappingProxyType/tuple throughout.
        # transforms/gates/coalesce_settings contain frozen dataclass instances
        # (scalars only) so tuple() is sufficient — no nested containers to freeze.
        object.__setattr__(self, "transforms", tuple(self.transforms))
        object.__setattr__(self, "gates", tuple(self.gates))
        object.__setattr__(self, "coalesce_settings", tuple(self.coalesce_settings))
        freeze_fields(self, "sources", "sinks", "config", "aggregation_settings")


@dataclass(frozen=True, slots=True)
class AggregationFlushResult:
    """Result of flushing aggregation buffers.

    Replaces a wide tuple return type with named fields for clarity and
    type safety. Using frozen dataclass prevents accidental mutation.
    """

    rows_succeeded: int = 0
    rows_failed: int = 0
    rows_routed_success: int = 0
    rows_routed_failure: int = 0
    rows_quarantined: int = 0
    rows_coalesced: int = 0
    rows_forked: int = 0
    rows_expanded: int = 0
    rows_buffered: int = 0
    rows_diverted: int = 0
    routed_destinations: Mapping[str, int] = field(default_factory=lambda: MappingProxyType({}))

    def __post_init__(self) -> None:
        freeze_fields(self, "routed_destinations")

    def to_dict(self) -> dict[str, Any]:
        """Serialize to a plain dict for JSON export.

        Replaces ``dataclasses.asdict()`` which cannot deep-copy
        ``MappingProxyType`` fields.
        """
        return {
            "rows_succeeded": self.rows_succeeded,
            "rows_failed": self.rows_failed,
            "rows_routed_success": self.rows_routed_success,
            "rows_routed_failure": self.rows_routed_failure,
            "rows_quarantined": self.rows_quarantined,
            "rows_coalesced": self.rows_coalesced,
            "rows_forked": self.rows_forked,
            "rows_expanded": self.rows_expanded,
            "rows_buffered": self.rows_buffered,
            "rows_diverted": self.rows_diverted,
            "routed_destinations": deep_thaw(self.routed_destinations),
        }

    def __add__(self, other: AggregationFlushResult) -> AggregationFlushResult:
        """Combine two results by summing all counters."""
        combined_destinations: Counter[str] = Counter(self.routed_destinations)
        combined_destinations.update(other.routed_destinations)
        return AggregationFlushResult(
            rows_succeeded=self.rows_succeeded + other.rows_succeeded,
            rows_failed=self.rows_failed + other.rows_failed,
            rows_routed_success=self.rows_routed_success + other.rows_routed_success,
            rows_routed_failure=self.rows_routed_failure + other.rows_routed_failure,
            rows_quarantined=self.rows_quarantined + other.rows_quarantined,
            rows_coalesced=self.rows_coalesced + other.rows_coalesced,
            rows_forked=self.rows_forked + other.rows_forked,
            rows_expanded=self.rows_expanded + other.rows_expanded,
            rows_buffered=self.rows_buffered + other.rows_buffered,
            rows_diverted=self.rows_diverted + other.rows_diverted,
            routed_destinations=MappingProxyType(dict(combined_destinations)),
        )


@dataclass
class ExecutionCounters:
    """Mutable counters accumulated during pipeline execution.

    Replaces the loose counter variables + routed_destinations Counter
    that were duplicated in both _execute_run() and _process_resumed_rows().

    Mutable (not frozen) because counters are incremented row-by-row during
    the processing loop. Frozen would require creating new instances on
    every update.
    """

    rows_processed: int = 0
    rows_succeeded: int = 0
    rows_failed: int = 0
    rows_routed_success: int = 0
    rows_routed_failure: int = 0
    rows_quarantined: int = 0
    rows_forked: int = 0
    rows_coalesced: int = 0
    rows_coalesce_failed: int = 0
    rows_expanded: int = 0
    rows_buffered: int = 0
    rows_diverted: int = 0
    routed_destinations: Counter[str] = field(default_factory=Counter)

    def accumulate_flush_result(self, result: AggregationFlushResult) -> None:
        """Merge an AggregationFlushResult into these counters.

        Replaces the manual per-counter additions that appeared after every
        check_aggregation_timeouts() and flush_remaining_aggregation_buffers() call.
        """
        self.rows_succeeded += result.rows_succeeded
        self.rows_failed += result.rows_failed
        self.rows_routed_success += result.rows_routed_success
        self.rows_routed_failure += result.rows_routed_failure
        self.rows_quarantined += result.rows_quarantined
        self.rows_coalesced += result.rows_coalesced
        self.rows_forked += result.rows_forked
        self.rows_expanded += result.rows_expanded
        self.rows_buffered += result.rows_buffered
        self.rows_diverted += result.rows_diverted
        for dest, count in result.routed_destinations.items():
            self.routed_destinations[dest] += count

    def to_flush_result(self) -> AggregationFlushResult:
        """Build an AggregationFlushResult from these counters.

        Mirrors ``to_run_result()`` for the aggregation flush path.
        """
        return AggregationFlushResult(
            rows_succeeded=self.rows_succeeded,
            rows_failed=self.rows_failed,
            rows_routed_success=self.rows_routed_success,
            rows_routed_failure=self.rows_routed_failure,
            rows_quarantined=self.rows_quarantined,
            rows_coalesced=self.rows_coalesced,
            rows_forked=self.rows_forked,
            rows_expanded=self.rows_expanded,
            rows_buffered=self.rows_buffered,
            rows_diverted=self.rows_diverted,
            routed_destinations=dict(self.routed_destinations),
        )

    def to_run_result(self, run_id: str, status: RunStatus) -> RunResult:
        """Build a RunResult from these counters.

        Args:
            run_id: The run identifier.
            status: Run status (callers must be explicit).
        """
        return RunResult(
            run_id=run_id,
            status=status,
            rows_processed=self.rows_processed,
            rows_succeeded=self.rows_succeeded,
            rows_failed=self.rows_failed,
            rows_routed_success=self.rows_routed_success,
            rows_routed_failure=self.rows_routed_failure,
            rows_quarantined=self.rows_quarantined,
            rows_forked=self.rows_forked,
            rows_coalesced=self.rows_coalesced,
            rows_coalesce_failed=self.rows_coalesce_failed,
            rows_expanded=self.rows_expanded,
            rows_buffered=self.rows_buffered,
            rows_diverted=self.rows_diverted,
            routed_destinations=dict(self.routed_destinations),
        )


class RouteValidationError(Exception):
    """Raised when route configuration is invalid.

    This error is raised at pipeline initialization, before any rows are
    processed. It indicates a configuration problem that would cause
    failures during processing.
    """
