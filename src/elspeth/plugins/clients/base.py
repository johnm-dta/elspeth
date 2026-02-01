# src/elspeth/plugins/clients/base.py
"""Base class for audited clients."""

from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from elspeth.core.landscape.recorder import LandscapeRecorder
    from elspeth.core.rate_limit import NoOpLimiter
    from elspeth.core.rate_limit.limiter import RateLimiter
    from elspeth.telemetry.events import ExternalCallCompleted

# Type alias for telemetry emit callback.
# When telemetry is disabled, orchestrator provides a no-op function.
# Clients always call this - never check for None.
TelemetryEmitCallback = Callable[["ExternalCallCompleted"], None]


class AuditedClientBase:
    """Base class for clients that automatically record to audit trail.

    Provides common infrastructure for tracking external calls:
    - Reference to LandscapeRecorder for audit storage and call index allocation
    - State ID linking calls to the current processing state
    - Run ID and telemetry callback for operational visibility
    - Optional rate limiter for throttling external calls

    Subclasses implement specific client protocols (LLM, HTTP, etc.)
    while inheriting automatic audit recording, telemetry emission, and rate limiting.

    Thread Safety:
        The _next_call_index method delegates to LandscapeRecorder.allocate_call_index(),
        which is thread-safe. Multiple threads and multiple client types can safely
        call this method concurrently without risk of call_index collisions.

    Call Index Coordination:
        Call indices are allocated centrally by the LandscapeRecorder, ensuring
        UNIQUE(state_id, call_index) across all client types (HTTP, LLM) and retry
        attempts. This prevents IntegrityError when multiple clients share the same state_id.

    Telemetry:
        Clients emit ExternalCallCompleted events after successful Landscape recording.
        The telemetry_emit callback is always present - when telemetry is disabled,
        orchestrator provides a no-op. Clients never check for None.

    Rate Limiting:
        Clients optionally accept a rate limiter. When provided, _acquire_rate_limit()
        blocks until the rate limit allows the request. When None, no throttling occurs.
        Subclasses should call _acquire_rate_limit() before making external calls.
    """

    def __init__(
        self,
        recorder: LandscapeRecorder,
        state_id: str,
        run_id: str,
        telemetry_emit: TelemetryEmitCallback,
        *,
        limiter: RateLimiter | NoOpLimiter | None = None,
    ) -> None:
        """Initialize audited client.

        Args:
            recorder: LandscapeRecorder for audit trail storage and call index allocation
            state_id: Node state ID to associate calls with
            run_id: Pipeline run ID for telemetry correlation
            telemetry_emit: Callback to emit telemetry events (no-op when disabled)
            limiter: Optional rate limiter for throttling requests (from RateLimitRegistry)
        """
        self._recorder = recorder
        self._state_id = state_id
        self._run_id = run_id
        self._telemetry_emit = telemetry_emit
        self._limiter = limiter

    def _next_call_index(self) -> int:
        """Get next call index for this state (thread-safe).

        Delegates to LandscapeRecorder for centralized call index allocation.
        This ensures unique indices across all client types sharing the same state_id.

        Returns:
            Sequential call index, unique within this state_id (not just this client)
        """
        return self._recorder.allocate_call_index(self._state_id)

    def _acquire_rate_limit(self) -> None:
        """Acquire rate limit permission before making external call.

        Blocks until the rate limiter allows the request. If no limiter
        is configured, returns immediately (no throttling).

        Subclasses should call this at the start of their external call
        methods (e.g., chat_completion, post) before making the actual request.
        """
        if self._limiter is not None:
            self._limiter.acquire()

    def close(self) -> None:
        """Release any resources held by the client.

        Default implementation is a no-op. Subclasses may override
        to close underlying connections or resources.
        """
        pass
