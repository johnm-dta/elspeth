"""CoalesceExecutor: Merges tokens from parallel fork paths.

Coalesce is a stateful barrier that holds tokens until merge conditions are met.
Tokens are correlated by row_id (same source row that was forked).
"""

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from elspeth.contracts import TokenInfo
from elspeth.core.config import CoalesceSettings
from elspeth.core.landscape import LandscapeRecorder
from elspeth.engine.spans import SpanFactory

if TYPE_CHECKING:
    from elspeth.engine.tokens import TokenManager


@dataclass
class CoalesceOutcome:
    """Result of a coalesce accept operation.

    Attributes:
        held: True if token is being held waiting for more branches
        merged_token: The merged token if merge is complete, None if held
        consumed_tokens: Tokens that were merged (marked COALESCED)
        coalesce_metadata: Audit metadata about the merge (branches, policy, etc.)
        failure_reason: Reason for failure if merge failed (timeout, missing branches)
    """

    held: bool
    merged_token: TokenInfo | None = None
    consumed_tokens: list[TokenInfo] = field(default_factory=list)
    coalesce_metadata: dict[str, Any] | None = None
    failure_reason: str | None = None


@dataclass
class _PendingCoalesce:
    """Tracks pending tokens for a single row_id at a coalesce point."""

    arrived: dict[str, TokenInfo]  # branch_name -> token
    arrival_times: dict[str, float]  # branch_name -> monotonic time
    first_arrival: float  # For timeout calculation


class CoalesceExecutor:
    """Executes coalesce operations with audit recording.

    Maintains state for pending coalesce operations:
    - Tracks which tokens have arrived for each row_id
    - Evaluates merge conditions based on policy
    - Merges row data according to strategy
    - Records audit trail via LandscapeRecorder

    Example:
        executor = CoalesceExecutor(recorder, span_factory, token_manager, run_id)

        # Configure coalesce point
        executor.register_coalesce(settings, node_id)

        # Accept tokens as they arrive
        for token in arriving_tokens:
            outcome = executor.accept(token, "coalesce_name", step_in_pipeline)
            if outcome.merged_token:
                # Merged token continues through pipeline
                work_queue.append(outcome.merged_token)
    """

    def __init__(
        self,
        recorder: LandscapeRecorder,
        span_factory: SpanFactory,
        token_manager: "TokenManager",
        run_id: str,
    ) -> None:
        """Initialize executor.

        Args:
            recorder: Landscape recorder for audit trail
            span_factory: Span factory for tracing
            token_manager: TokenManager for creating merged tokens
            run_id: Run identifier for audit context
        """
        self._recorder = recorder
        self._spans = span_factory
        self._token_manager = token_manager
        self._run_id = run_id

        # Coalesce configuration: name -> settings
        self._settings: dict[str, CoalesceSettings] = {}
        # Node IDs: coalesce_name -> node_id
        self._node_ids: dict[str, str] = {}
        # Pending tokens: (coalesce_name, row_id) -> _PendingCoalesce
        self._pending: dict[tuple[str, str], _PendingCoalesce] = {}

    def register_coalesce(
        self,
        settings: CoalesceSettings,
        node_id: str,
    ) -> None:
        """Register a coalesce point.

        Args:
            settings: Coalesce configuration
            node_id: Node ID assigned by orchestrator
        """
        self._settings[settings.name] = settings
        self._node_ids[settings.name] = node_id

    def get_registered_names(self) -> list[str]:
        """Get names of all registered coalesce points.

        Used by processor for timeout checking loop.

        Returns:
            List of registered coalesce names
        """
        return list(self._settings.keys())
