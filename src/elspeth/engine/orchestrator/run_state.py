"""Internal graph, run, loop, and resume state bundles for orchestrator code."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from elspeth.contracts.freeze import freeze_fields

if TYPE_CHECKING:
    from elspeth.contracts import PendingOutcome, TokenInfo, TransformProtocol
    from elspeth.contracts.checkpoint import ResumedRow
    from elspeth.contracts.coordination import CoordinationToken
    from elspeth.contracts.plugin_context import PluginContext
    from elspeth.contracts.run_result import RunResult
    from elspeth.contracts.schema_contract import SchemaContract
    from elspeth.contracts.types import CoalesceName, GateName, NodeID, SinkName
    from elspeth.core.checkpoint.recovery import IncompleteTokenSpec, RecoveryManager
    from elspeth.core.landscape.factory import RecorderFactory
    from elspeth.engine.coalesce_executor import CoalesceExecutor
    from elspeth.engine.orchestrator.ports import RowProcessorHandle
    from elspeth.engine.orchestrator.types import ExecutionCounters, PipelineConfig


# Type alias for pending tokens accumulated during row processing.
# Keys are sink names, values are lists of (token, optional outcome) pairs.
# Used across LoopContext, accumulate_row_outcomes, flush functions, etc.
PendingTokenMap = dict[str, list[tuple["TokenInfo", "PendingOutcome | None"]]]


class _RunFailedWithPartialResultError(Exception):
    """Internal wrapper used to move partial counters to outer failure handlers.

    Raised by the leader run-phase sequencing (``leader_drain.py``) and the
    resume path, caught and translated by the run-lifecycle failure ceremony
    (``run_lifecycle.py``). ``core.py`` re-exports it for the legacy
    ``...orchestrator.core`` import path.
    """

    def __init__(self, original_error: Exception, partial_result: RunResult) -> None:
        super().__init__(str(original_error))
        self.original_error = original_error
        self.partial_result = partial_result
        self.original_traceback = original_error.__traceback__


@dataclass(frozen=True, slots=True)
class GraphArtifacts:
    """Return type for _register_graph_nodes_and_edges().

    Named fields eliminate positional-swap hazards - several members share
    compatible Mapping[..., NodeID] types that mypy cannot distinguish in a tuple.

    All mapping fields are wrapped by __post_init__ to enforce deep
    immutability, matching the DAGTraversalContext precedent.
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
    """Return type for RunContextFactory.initialize_run_context().

    Bundles the five objects created during run initialization that are
    consumed by subsequent phases. Short-lived: consumed immediately to
    build LoopContext. Mapping fields are wrapped for consistency with
    GraphArtifacts.
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
    """Parameter bundle for the main processing loop and sink flush coordinator.

    Reduces 10+ parameter signatures to (self, loop_ctx, ...) and prevents
    parameter-list growth as the loop acquires new concerns.

    NOT frozen: ``counters`` and ``pending_tokens`` are mutated in place
    throughout the processing loop.

    Convention: fields below the "Read-only" separator are never reassigned
    after construction. They are not frozen because ``counters`` and
    ``pending_tokens`` require in-place mutation. Treat read-only fields as
    if they were on a frozen dataclass - mappings are wrapped in __post_init__.
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

    def __post_init__(self) -> None:
        freeze_fields(self, "agg_transform_lookup", "coalesce_node_map")


@dataclass(frozen=True, slots=True)
class LoopResult:
    """Return value from _run_main_processing_loop().

    Carries timing state back to the caller so that final progress emission
    and PhaseCompleted can be emitted AFTER sink writes. The resume loop does
    not use this because it has no progress or phase events.
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

    Per ADR-025 Decision section 3, schema contracts are plural-by-source.
    ``schema_contracts_by_source`` is non-optional and never empty - resume
    reconstruction either populates one contract per source from ``run_sources``
    or one contract keyed by the single-source NodeID derived from
    ``rows.source_node_id``. The previous singular ``schema_contract`` field
    has been deleted; consumers look up each row's contract via
    ``schema_contracts_by_source[row.source_node_id]``.

    The empty case - a run that failed before any row was committed and before
    any ``run_sources`` records were written - is refused upstream in
    ``_reconstruct_resume_state`` via :class:`EmptyResumeStateError`. That
    exception is the interpretable "nothing to resume" outcome; the
    construction-time guard below is the chokepoint that pins the invariant
    against future regressions where some caller bypasses the upstream check.
    """

    factory: RecorderFactory
    run_id: str
    unprocessed_rows: Sequence[ResumedRow]
    # F1 fix: incomplete child tokens grouped by row_id - used by the resume loop
    # to dispatch partial-fork/expand/coalesce rows via mid-DAG continuation
    # instead of whole-row restart (which re-emits completed branches).
    incomplete_by_row: Mapping[str, Sequence[IncompleteTokenSpec]]
    # RecoveryManager needed by resume loop for reconstruct_token_row.
    recovery_manager: RecoveryManager
    schema_contracts_by_source: Mapping[NodeID, SchemaContract]
    source_names_by_source: Mapping[NodeID, str]
    source_lifecycle_by_source: Mapping[NodeID, str]
    # F1: True when the scheduler journal carries BLOCKED barrier rows for the
    # run. Those tokens are EXCLUDED from unprocessed_rows (they are restored
    # into executor buffers at processor construction, not re-driven), so the
    # resume quiescence gate must consult this flag - a fully-buffered crashed
    # run has zero unprocessed rows but must still run the processing path so
    # the restored buffers flush.
    has_restored_barrier_work: bool = False
    # F1: old->retry batch_id mapping from handle_incomplete_batches; consumed
    # by the processor's journal restore.
    batch_id_remap: Mapping[str, str] = field(default_factory=dict)
    # ADR-030: the leader fencing token minted by resume()'s seat-acquisition
    # CAS (acquire_run_leadership). Carried by value out of reconstruct_resume_state
    # to the processor / checkpoint / finalize collaborators; never re-read mid-run.
    coordination_token: CoordinationToken | None = None

    def __post_init__(self) -> None:
        # Local import to avoid hoisting OrchestrationInvariantError into the
        # module header - it's only referenced inside this guard, and the
        # contracts package is already an L0 dependency.
        from elspeth.contracts.errors import OrchestrationInvariantError

        # incomplete_by_row is a dict[str, list[IncompleteTokenSpec]] of frozen specs;
        # deep_freeze converts it to MappingProxyType[str, tuple[IncompleteTokenSpec, ...]].
        # The resume loop consumes it via membership (`row_id in incomplete_by_row`) and
        # iteration (`for s in incomplete_by_row[row_id]`), both fine on the frozen shape.
        # recovery_manager is a live service object (not a container) - NOT frozen here.
        freeze_fields(
            self,
            "incomplete_by_row",
            "schema_contracts_by_source",
            "source_names_by_source",
            "source_lifecycle_by_source",
            "batch_id_remap",
        )
        # unprocessed_rows is a Sequence of ResumedRow instances. Each
        # ResumedRow is fully deep-frozen in its own __post_init__ (row_data
        # is MappingProxyType via freeze_fields, not a plain dict). Tuple-
        # ifying the outer Sequence is sufficient.
        if not isinstance(self.unprocessed_rows, tuple):
            object.__setattr__(self, "unprocessed_rows", tuple(self.unprocessed_rows))
        # ADR-025 section 3: schema_contracts_by_source is non-empty by invariant.
        # The empty case (no rows committed, no run_sources records) is
        # handled upstream by ``_reconstruct_resume_state`` via
        # :class:`EmptyResumeStateError` - ResumeState is never constructed in
        # that case. This guard pins the invariant so a future caller that
        # bypasses the upstream check fails loudly rather than silently picking
        # an arbitrary contract.
        if not self.schema_contracts_by_source:
            raise OrchestrationInvariantError(
                "ResumeState.schema_contracts_by_source must not be empty. "
                "Empty-state resume should have been refused upstream via "
                "EmptyResumeStateError before ResumeState was constructed. "
                "If you're hitting this, the upstream check is missing."
            )
        # ADR-025 section 3: resume rejects rather than picks an arbitrary
        # contract. Every row's ``source_node_id`` must have a corresponding
        # entry in ``schema_contracts_by_source``.
        missing = {row.source_node_id for row in self.unprocessed_rows} - set(self.schema_contracts_by_source)
        if missing:
            raise OrchestrationInvariantError(
                "ResumeState.schema_contracts_by_source is missing entries for "
                f"source_node_id(s): {sorted(missing)}. Available keys: "
                f"{sorted(self.schema_contracts_by_source)}. Resume rejects rather "
                "than picks an arbitrary contract (ADR-025 section 3)."
            )
