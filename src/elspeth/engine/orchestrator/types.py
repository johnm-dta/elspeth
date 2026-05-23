"""Pipeline configuration and result types.

These types define the interface for pipeline execution:
- PipelineConfig: Input configuration for a run
- RunResult: Output statistics from a run
- RouteValidationError: Configuration validation failure
- AggregationFlushResult: Result of flushing aggregation buffers

IMPORTANT: Import Cycle Prevention
----------------------------------
This module is a LEAF MODULE - it must NOT import from other orchestrator
submodules (validation.py, export.py, aggregation.py, core.py).

Other modules import FROM here (e.g., validation.py imports RouteValidationError).
If types.py were to import from those modules, a circular import would occur.

Keep types.py as pure data definitions with minimal dependencies.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import TYPE_CHECKING, Any, Protocol

from elspeth.contracts.freeze import deep_thaw, freeze_fields
from elspeth.contracts.run_result import RunResult as RunResult  # re-exported

if TYPE_CHECKING:
    from elspeth.contracts import PendingOutcome, SinkProtocol, SourceProtocol, TokenInfo
    from elspeth.contracts.aggregation_checkpoint import AggregationCheckpointState
    from elspeth.contracts.checkpoint import ResumedRow
    from elspeth.contracts.coalesce_checkpoint import CoalesceCheckpointState
    from elspeth.contracts.plugin_context import PluginContext
    from elspeth.contracts.schema_contract import SchemaContract
    from elspeth.contracts.types import CoalesceName, GateName, NodeID, SinkName
    from elspeth.core.config import AggregationSettings, CoalesceSettings, GateSettings
    from elspeth.core.landscape.factory import RecorderFactory
    from elspeth.engine.coalesce_executor import CoalesceExecutor

# Import protocols at runtime (not TYPE_CHECKING) because RowPlugin type alias
# is used in runtime annotations and isinstance() checks
from elspeth.contracts import RunStatus, TransformProtocol

# Type alias for pending tokens accumulated during row processing.
# Keys are sink names, values are lists of (token, optional outcome) pairs.
# Used across LoopContext, accumulate_row_outcomes, flush functions, etc.
PendingTokenMap = dict[str, list[tuple["TokenInfo", "PendingOutcome | None"]]]

# Type alias for row-processing plugins in the transforms pipeline
# NOTE: BaseAggregation was DELETED - aggregation is now handled by
# batch-aware transforms (is_batch_aware=True on TransformProtocol)
RowPlugin = TransformProtocol
"""Row-processing plugin type for pipeline transforms list."""


class RowProcessorHandle(Protocol):
    """Orchestrator-facing processor contract stored in run/loop contexts."""

    @property
    def token_manager(self) -> Any:
        raise NotImplementedError

    def process_row(self, *args: Any, **kwargs: Any) -> Any:
        raise NotImplementedError

    def process_existing_row(self, *args: Any, **kwargs: Any) -> Any:
        raise NotImplementedError

    def process_token(self, *args: Any, **kwargs: Any) -> Any:
        raise NotImplementedError

    def check_aggregation_timeout(self, *args: Any, **kwargs: Any) -> Any:
        raise NotImplementedError

    def get_aggregation_buffer_count(self, *args: Any, **kwargs: Any) -> int:
        raise NotImplementedError

    def handle_timeout_flush(self, *args: Any, **kwargs: Any) -> Any:
        raise NotImplementedError

    def active_scheduled_row_ids(self) -> frozenset[str]:
        """Return row IDs represented by active durable scheduler work."""
        ...

    def summarize_scheduled_work(self) -> tuple[str, ...]:
        """Return grouped active scheduler work for invariant diagnostics."""
        ...

    def mark_blocked_barrier_terminal(self, barrier_key: str, token_ids: tuple[str, ...]) -> int:
        """Mark durable scheduler work consumed by a barrier as terminal."""
        ...

    def mark_sink_bound_scheduler_terminal(self, token_id: str) -> None:
        """Mark scheduler sink handoff complete after sink outcome durability."""
        ...

    def get_aggregation_checkpoint_state(self) -> AggregationCheckpointState:
        raise NotImplementedError

    def get_coalesce_checkpoint_state(self) -> CoalesceCheckpointState | None:
        raise NotImplementedError

    def resolve_sink_step(self) -> int:
        raise NotImplementedError


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


@dataclass(frozen=True, slots=True)
class ValueSourceFinding:
    """Structured per-field violation report from the value-source walker.

    Each finding pairs the offending ``component_id`` (the operator-facing
    transform name, e.g. ``openrouter_llm_node_1``) with the ``field_name``
    that violated its declaration and a human-readable ``reason``.

    Carrying the three fields directly — rather than encoding them into a
    formatted string and reverse-parsing at the consumer — eliminates the
    silent-attribution failure mode where a future format change would
    have produced ``ValidationError(component_id=None)`` records the
    composer UI cannot tie back to a specific node.

    All fields are scalars (per CLAUDE.md "Scalar-Only Fields Need No
    Guard"); ``frozen=True, slots=True`` is sufficient — no freeze guard
    is required.
    """

    component_id: str
    field_name: str
    reason: str

    def __post_init__(self) -> None:
        if not self.component_id:
            raise ValueError("ValueSourceFinding.component_id must be non-empty")
        if not self.field_name:
            raise ValueError("ValueSourceFinding.field_name must be non-empty")
        if not self.reason:
            raise ValueError("ValueSourceFinding.reason must be non-empty")

    def format(self) -> str:
        """Render as a human-readable string for log/check-detail surfaces.

        The single point of stringification — anything wanting a flat
        message synthesises it here. Keeps the format coupled to the
        finding's own fields rather than scattering ``f"component '{...}'"``
        templates across the codebase.
        """
        return f"component '{self.component_id}' field '{self.field_name}': {self.reason}"


class ValueSourceValidationError(Exception):
    """Raised when a plugin-config field violates its value-source declaration.

    Examples:
    - An OpenRouter LLM transform's ``model`` field is set to a string
      that does not appear in the registered catalog.
    - An Azure LLM transform's ``model`` field has been overridden to a
      value that does not match its ``deployment_name`` sibling.

    Like :class:`RouteValidationError`, this error fires at pipeline
    initialization (pre-token), so the failure is per-pipeline rather
    than per-row.

    ``findings`` carries one :class:`ValueSourceFinding` per offending
    field. Consumers (e.g. the composer ``/validate`` path) read
    ``finding.component_id`` directly to attribute each violation to its
    node — no string parsing.
    """

    def __init__(
        self,
        message: str,
        *,
        findings: tuple[ValueSourceFinding, ...] = (),
    ) -> None:
        super().__init__(message)
        self.findings = findings


# --- Extraction return types ---


@dataclass(frozen=True, slots=True)
class GraphArtifacts:
    """Return type for _register_graph_nodes_and_edges().

    Named fields eliminate positional-swap hazards — several members share
    compatible Mapping[..., NodeID] types that mypy cannot distinguish in a tuple.

    All mapping fields are wrapped in MappingProxyType via __post_init__
    to enforce deep immutability, matching the DAGTraversalContext precedent.
    """

    edge_map: Mapping[tuple[NodeID, str], str]
    source_id: NodeID
    source_id_map: Mapping[str, NodeID]
    sink_id_map: Mapping[SinkName, NodeID]
    transform_id_map: Mapping[int, NodeID]
    config_gate_id_map: Mapping[GateName, NodeID]
    coalesce_id_map: Mapping[CoalesceName, NodeID]

    def __post_init__(self) -> None:
        freeze_fields(
            self,
            "edge_map",
            "source_id_map",
            "sink_id_map",
            "transform_id_map",
            "config_gate_id_map",
            "coalesce_id_map",
        )


@dataclass(frozen=True, slots=True)
class AggNodeEntry:
    """Named pair for aggregation lookup values.

    Replaces tuple[TransformProtocol, NodeID] to prevent positional-swap bugs,
    applying the same rationale as GraphArtifacts.
    """

    transform: TransformProtocol
    node_id: NodeID


@dataclass(frozen=True, slots=True)
class RunContext:
    """Return type for _initialize_run_context().

    Bundles the five objects created during run initialization that are
    consumed by subsequent phases. Short-lived: consumed immediately to
    build LoopContext. Mapping fields are wrapped in MappingProxyType
    for consistency with GraphArtifacts.
    """

    ctx: PluginContext
    processor: RowProcessorHandle
    coalesce_executor: CoalesceExecutor | None
    coalesce_node_map: Mapping[CoalesceName, NodeID]
    agg_transform_lookup: Mapping[str, AggNodeEntry]

    def __post_init__(self) -> None:
        freeze_fields(self, "coalesce_node_map", "agg_transform_lookup")


@dataclass(slots=True)
class LoopContext:
    """Parameter bundle for _run_main_processing_loop() and _flush_and_write_sinks().

    Reduces 10+ parameter signatures to (self, loop_ctx, ...) and prevents
    parameter-list growth as the loop acquires new concerns.

    NOT frozen: ``counters`` and ``pending_tokens`` are mutated in place
    throughout the processing loop.

    Convention: fields below the "Read-only" separator are never reassigned
    after construction. They are not frozen because ``counters`` and
    ``pending_tokens`` require in-place mutation. Treat read-only fields as
    if they were on a frozen dataclass — mappings are wrapped in
    MappingProxyType at construction time.
    """

    # --- Mutable state (updated row-by-row) ---
    counters: ExecutionCounters
    pending_tokens: PendingTokenMap

    # --- Read-only after construction (not reassigned) ---
    processor: RowProcessorHandle
    ctx: PluginContext
    config: PipelineConfig
    agg_transform_lookup: Mapping[str, AggNodeEntry]
    coalesce_executor: CoalesceExecutor | None
    coalesce_node_map: Mapping[CoalesceName, NodeID]
    last_token_id: str | None = None
    last_token_source_id: NodeID | None = None

    def __post_init__(self) -> None:
        freeze_fields(self, "agg_transform_lookup", "coalesce_node_map")


@dataclass(frozen=True, slots=True)
class LoopResult:
    """Return value from _run_main_processing_loop().

    Carries timing state back to the caller so that final progress emission
    and PhaseCompleted can be emitted AFTER sink writes (not before).
    The resume loop does not use this — it has no progress or phase events.
    """

    interrupted: bool
    start_time: float
    phase_start: float
    last_progress_time: float


@dataclass(frozen=True, slots=True)
class ResumeState:
    """Return type for _reconstruct_resume_state().

    Bundles the state reconstruction results needed to process resumed rows.
    Short-lived: consumed immediately by the resume method.

    Per ADR-025 Decision §3, schema contracts are plural-by-source.
    ``schema_contracts_by_source`` is non-optional **and never empty** —
    resume reconstruction either populates one contract per source from
    ``run_sources`` (RC6 audit DBs) or one contract keyed by the
    single-source NodeID derived from ``rows.source_node_id`` (pre-RC6
    audit DBs). The previous singular ``schema_contract`` field has
    been deleted; consumers look up each row's contract via
    ``schema_contracts_by_source[row.source_node_id]``.

    The empty case — a run that failed before any row was committed
    and before any ``run_sources`` records were written (``on_start``
    failure, source-level abort, infrastructure crash pre-ingest) — is
    refused **upstream** in ``_reconstruct_resume_state`` via
    :class:`EmptyResumeStateError`. That exception is the interpretable
    "nothing to resume" outcome; the construction-time guard below is
    the chokepoint that pins the invariant against future regressions
    where some caller bypasses the upstream check.
    """

    factory: RecorderFactory
    run_id: str
    restored_aggregation_state: Mapping[str, AggregationCheckpointState]
    restored_coalesce_state: CoalesceCheckpointState | None
    unprocessed_rows: Sequence[ResumedRow]
    schema_contracts_by_source: Mapping[NodeID, SchemaContract]

    def __post_init__(self) -> None:
        # Local import to avoid hoisting OrchestrationInvariantError into the
        # module header — it's only referenced inside this guard, and the
        # contracts package is already an L0 dependency so this is not a
        # layer-architecture concern.
        from elspeth.contracts.errors import OrchestrationInvariantError

        freeze_fields(self, "restored_aggregation_state", "schema_contracts_by_source")
        # unprocessed_rows is a Sequence of ResumedRow instances. Each
        # ResumedRow is fully deep-frozen in its own __post_init__ (row_data
        # is MappingProxyType via freeze_fields, not a plain dict). Tuple-
        # ifying the outer Sequence is sufficient — further deep_freeze
        # traversal is not needed because every leaf is already immutable.
        # Consumers that need row_data as a plain dict (e.g. PipelineRow) call
        # dict(row.row_data) explicitly at the construction boundary (see
        # engine/orchestrator/core.py _reconstruct_resume_state loop).
        if not isinstance(self.unprocessed_rows, tuple):
            object.__setattr__(self, "unprocessed_rows", tuple(self.unprocessed_rows))
        # ADR-025 §3: schema_contracts_by_source is non-empty by invariant.
        # The empty case (no rows committed, no run_sources records) is
        # handled upstream by ``_reconstruct_resume_state`` via
        # :class:`EmptyResumeStateError` — ResumeState is never
        # constructed in that case. This guard pins the invariant so a
        # future caller that bypasses the upstream check fails loudly
        # rather than silently picking an arbitrary contract.
        if not self.schema_contracts_by_source:
            raise OrchestrationInvariantError(
                "ResumeState.schema_contracts_by_source must not be empty. "
                "Empty-state resume should have been refused upstream via "
                "EmptyResumeStateError before ResumeState was constructed. "
                "If you're hitting this, the upstream check is missing."
            )
        # ADR-025 §3: resume rejects rather than picks an arbitrary
        # contract. Every row's ``source_node_id`` must have a
        # corresponding entry in ``schema_contracts_by_source`` —
        # otherwise the loop would have to pick a default, which is
        # the failure mode this ADR closes.
        missing = {row.source_node_id for row in self.unprocessed_rows} - set(self.schema_contracts_by_source)
        if missing:
            raise OrchestrationInvariantError(
                "ResumeState.schema_contracts_by_source is missing entries for "
                f"source_node_id(s): {sorted(missing)}. Available keys: "
                f"{sorted(self.schema_contracts_by_source)}. Resume rejects rather "
                "than picks an arbitrary contract (ADR-025 §3)."
            )


# Factory that creates a per-sink checkpoint callback.
# Takes a sink_node_id (str) and returns a callback invoked after each
# token is written to that sink.
type _CheckpointFactory = Callable[[str], Callable[[TokenInfo], None]]
