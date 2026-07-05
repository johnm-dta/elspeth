"""Processor-facing orchestration ports.

These protocols describe capabilities the orchestrator needs from the row
processor. They live outside ``types.py`` so the data-definition module remains
a runtime leaf while call sites can depend on focused capability slices.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Protocol

if TYPE_CHECKING:
    from elspeth.contracts import RowResult, TokenInfo
    from elspeth.contracts.barrier_scalars import BarrierScalars
    from elspeth.contracts.coordination import CoordinationToken
    from elspeth.contracts.plugin_context import PluginContext
    from elspeth.contracts.schema_contract import PipelineRow
    from elspeth.contracts.types import CoalesceName, NodeID
    from elspeth.core.checkpoint.recovery import IncompleteTokenSpec


class RunIdentityPort(Protocol):
    """Processor surface that exposes the active run id."""

    @property
    def run_id(self) -> str:
        """Expose the run identifier for diagnostics."""
        ...


class TokenCreationPort(RunIdentityPort, Protocol):
    """Processor surface needed when source quarantine creates a token."""

    @property
    def token_manager(self) -> Any:
        raise NotImplementedError

    @property
    def coordination_token(self) -> CoordinationToken | None:
        """Leader fencing token bound at construction."""
        ...


class RowProcessingPort(Protocol):
    """Processor surface for source/resume row and token execution."""

    def process_row(self, *args: Any, **kwargs: Any) -> Any:
        raise NotImplementedError

    def process_existing_row(self, *args: Any, **kwargs: Any) -> Any:
        raise NotImplementedError

    def process_token(self, *args: Any, **kwargs: Any) -> Any:
        raise NotImplementedError


class AggregationProcessorPort(Protocol):
    """Processor surface needed by aggregation timeout and EOF flushing."""

    def process_token(self, *args: Any, **kwargs: Any) -> Any:
        raise NotImplementedError

    def check_aggregation_timeout(self, *args: Any, **kwargs: Any) -> Any:
        raise NotImplementedError

    def get_aggregation_buffer_count(self, *args: Any, **kwargs: Any) -> int:
        raise NotImplementedError

    def handle_timeout_flush(self, *args: Any, **kwargs: Any) -> Any:
        raise NotImplementedError


class SchedulerDrainPort(Protocol):
    """Processor surface for durable scheduler drains."""

    def drain_scheduled_work(self, ctx: PluginContext) -> list[RowResult]:
        """Drain recoverable durable scheduler work during resume."""
        ...

    def has_scheduled_work(self) -> bool:
        """Return whether the durable scheduler has active non-terminal work."""
        ...

    def active_scheduled_row_ids(self) -> frozenset[str]:
        """Return row IDs represented by active durable scheduler work."""
        ...

    def summarize_scheduled_work(self) -> tuple[str, ...]:
        """Return grouped active scheduler work for invariant diagnostics."""
        ...


class SchedulerQuiescencePort(RunIdentityPort, Protocol):
    """Processor surface for scheduler quiescence and peer-lease checks."""

    def has_unresolved_scheduler_work(self) -> bool:
        """Return whether scheduler work remains short of a durable sink handoff."""
        ...

    def has_peer_active_leases(self) -> bool:
        """Return whether any peer worker holds an unexpired LEASED item."""
        ...

    def peer_lease_wait_budget_seconds(self) -> float:
        """Return bounded wait budget for peer-held active item leases."""
        ...

    def peer_active_lease_owners(self) -> tuple[str, ...]:
        """Return distinct peer lease owners holding unexpired LEASED rows."""
        ...

    def reap_expired_peer_leases(self) -> int:
        """Drive lease maintenance once so dead peers are reaped."""
        ...

    def summarize_unresolved_scheduler_work(self) -> tuple[str, ...]:
        """Return grouped unresolved scheduler work for invariant diagnostics."""
        ...


class SchedulerJournalPort(SchedulerDrainPort, SchedulerQuiescencePort, Protocol):
    """Processor surface for scheduler journal drains and quiescence checks."""


class BarrierIntakePort(RunIdentityPort, Protocol):
    """Processor surface for journal-first barrier intake and quiescence."""

    def run_barrier_intake(self, ctx: PluginContext) -> list[RowResult]:
        """Run one journal-first barrier intake pass."""
        ...

    def has_blocked_barrier_work(self) -> bool:
        """Return whether durable BLOCKED barrier holds remain."""
        ...

    def count_unquiesced_scheduler_work(self) -> int:
        """Count scheduler work that could still deposit barrier arrivals."""
        ...

    def summarize_unquiesced_scheduler_work(self) -> tuple[str, ...]:
        """Return grouped unquiesced scheduler work for invariant diagnostics."""
        ...


class CoalesceCompletionPort(Protocol):
    """Processor surface for durable coalesce barrier completion."""

    def mark_blocked_barrier_terminal(self, barrier_key: str, token_ids: tuple[str, ...]) -> int:
        """Mark durable scheduler work consumed by a barrier as terminal."""
        ...

    def complete_coalesce_merge(
        self,
        *,
        coalesce_name: CoalesceName,
        consumed_tokens: tuple[TokenInfo, ...],
        merged_token: TokenInfo,
        coalesce_node_id: NodeID,
        ctx: PluginContext,
    ) -> list[RowResult]:
        """Atomically consume coalesce inputs, emit the merge, and continue."""
        raise NotImplementedError


class SchedulerTerminalizer(Protocol):
    """Processor surface for terminalizing durable sink scheduler handoffs."""

    def mark_sink_bound_scheduler_terminal_many(self, token_ids: tuple[str, ...]) -> None:
        """Mark scheduler sink handoffs complete after durable batch sink outcomes."""
        ...


class SinkTerminalizationPort(SchedulerTerminalizer, Protocol):
    """Full processor surface for sink-bound scheduler terminalization."""

    def mark_sink_bound_scheduler_terminal(self, token_id: str) -> None:
        """Mark one scheduler sink handoff complete after sink outcome durability."""
        ...


class BarrierScalarsSource(Protocol):
    """Narrow processor surface the checkpoint-progress callback needs."""

    def get_barrier_scalars(self) -> BarrierScalars:
        """Compose the underivable barrier scalars from the live executors."""
        ...


class ResumeContinuationPort(Protocol):
    """Processor surface for incomplete-token resume continuation."""

    def resume_incomplete_token(
        self,
        spec: IncompleteTokenSpec,
        row_data: PipelineRow,
        ctx: PluginContext,
        *,
        resume_checkpoint_id: str,
    ) -> list[RowResult]:
        raise NotImplementedError


class SinkStepResolver(Protocol):
    """Processor surface for sink audit step resolution."""

    def resolve_sink_step(self) -> int:
        raise NotImplementedError


class EndOfInputBarrierProcessorPort(
    AggregationProcessorPort,
    BarrierIntakePort,
    CoalesceCompletionPort,
    Protocol,
):
    """Combined surface needed by the end-of-input barrier flush loop."""


class RowProcessorHandle(
    TokenCreationPort,
    RowProcessingPort,
    AggregationProcessorPort,
    SchedulerJournalPort,
    BarrierIntakePort,
    CoalesceCompletionPort,
    SinkTerminalizationPort,
    BarrierScalarsSource,
    ResumeContinuationPort,
    SinkStepResolver,
    Protocol,
):
    """Composed full processor contract stored in run/loop contexts."""


class CheckpointAfterSinkCallback(Protocol):
    """Post-sink callback with an explicit batch flush boundary."""

    def __call__(self, token: TokenInfo) -> None:
        """Record per-token checkpoint progress after durable sink handling."""
        ...

    def flush(self) -> None:
        """Flush batched post-sink work after callback use."""
        ...


class _CheckpointFactory(Protocol):
    """Factory that creates per-sink checkpoint-PROGRESS callbacks."""

    def __call__(self, sink_node_id: str) -> CheckpointAfterSinkCallback:
        """Return a callback invoked after each token is written to a sink."""
        ...
