"""Base class for audited clients."""

from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING

from elspeth.contracts.errors import FrameworkBugError

if TYPE_CHECKING:
    from elspeth.contracts import Call, CallStatus, CallType
    from elspeth.contracts.audit_protocols import CallRecorder
    from elspeth.contracts.call_data import CallPayload
    from elspeth.contracts.contexts import LimiterProtocol
    from elspeth.contracts.events import ExternalCallCompleted

# Type alias for telemetry emit callback.
# When telemetry is disabled, orchestrator provides a no-op function.
# Clients always call this - never check for None.
TelemetryEmitCallback = Callable[["ExternalCallCompleted"], None]


class AuditedClientBase:
    """Base class for clients that automatically record to audit trail.

    Provides common infrastructure for tracking external calls:
    - Reference to ExecutionRepository for audit storage and call index allocation
    - State ID or operation ID linking calls to their audit parent
    - Run ID and telemetry callback for operational visibility
    - Optional rate limiter for throttling external calls

    Subclasses implement specific client protocols (LLM, HTTP, etc.)
    while inheriting automatic audit recording, telemetry emission, and rate limiting.

    Thread Safety:
        The _next_call_index method delegates to ExecutionRepository.allocate_call_index(),
        which is thread-safe. Multiple threads and multiple client types can safely
        call this method concurrently without risk of call_index collisions.

    Call Index Coordination:
        Call indices are allocated centrally by the ExecutionRepository, ensuring
        UNIQUE(parent_id, call_index) across all client types (HTTP, LLM) and retry
        attempts. This prevents IntegrityError when multiple clients share the same
        state_id or operation_id.

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
        execution: CallRecorder,
        state_id: str | None,
        run_id: str,
        telemetry_emit: TelemetryEmitCallback,
        *,
        operation_id: str | None = None,
        limiter: LimiterProtocol | None = None,
        token_id: str | None = None,
    ) -> None:
        """Initialize audited client.

        Args:
            execution: CallRecorder for audit trail storage and call index allocation
            state_id: Node state ID to associate calls with, if this client is
                bound to row processing
            run_id: Pipeline run ID for telemetry correlation
            telemetry_emit: Callback to emit telemetry events (no-op when disabled)
            operation_id: Operation ID to associate calls with, if this client
                is bound to run/node-level work such as runtime preflight
            limiter: Optional rate limiter for throttling requests (from RateLimitRegistry)
            token_id: Optional token identity for transform-context correlation
        """
        if (state_id is None) == (operation_id is None):
            raise FrameworkBugError("AuditedClientBase requires exactly one of state_id or operation_id")
        self._execution = execution
        self._state_id = state_id
        self._operation_id = operation_id
        self._run_id = run_id
        self._telemetry_emit = telemetry_emit
        self._limiter = limiter
        self._token_id = token_id

    def _telemetry_token_id(self) -> str | None:
        """Get token_id for telemetry correlation when available."""
        return self._token_id

    def _telemetry_state_id(self) -> str | None:
        """Get state_id for telemetry parentage when this is a state call."""
        return self._state_id

    def _telemetry_operation_id(self) -> str | None:
        """Get operation_id for telemetry parentage when this is an operation call."""
        return self._operation_id

    def _next_call_index(self) -> int:
        """Get next call index for this audit parent (thread-safe).

        Delegates to ExecutionRepository for centralized call index allocation.
        This ensures unique indices across all client types sharing the same parent.

        Returns:
            Sequential call index, unique within this state_id or operation_id
            (not just this client)
        """
        if self._operation_id is not None:
            return self._execution.allocate_operation_call_index(self._operation_id)
        if self._state_id is None:
            raise FrameworkBugError("Audited client has neither state_id nor operation_id")
        return self._execution.allocate_call_index(self._state_id)

    def _record_call(
        self,
        *,
        call_index: int,
        call_type: CallType,
        status: CallStatus,
        request_data: CallPayload,
        response_data: CallPayload | None = None,
        error: CallPayload | None = None,
        latency_ms: float | None = None,
        resolved_prompt_template_hash: str | None = None,
    ) -> Call:
        """Record a call under the configured audit parent.

        ``resolved_prompt_template_hash`` is the Phase 5b Task 9 cross-DB
        anchor: when an LLM transform is downstream of a resolved
        interpretation event, the runtime reads the SHA-256 from
        ``options.resolved_prompt_template_hash`` on the node config and
        forwards it here. ``None`` for non-LLM calls and for LLM transforms
        that never went through an interpretation surface.
        """
        if self._operation_id is not None:
            return self._execution.record_operation_call(
                operation_id=self._operation_id,
                call_index=call_index,
                call_type=call_type,
                status=status,
                request_data=request_data,
                response_data=response_data,
                error=error,
                latency_ms=latency_ms,
                resolved_prompt_template_hash=resolved_prompt_template_hash,
            )
        if self._state_id is None:
            raise FrameworkBugError("Audited client has neither state_id nor operation_id")
        return self._execution.record_call(
            state_id=self._state_id,
            call_index=call_index,
            call_type=call_type,
            status=status,
            request_data=request_data,
            response_data=response_data,
            error=error,
            latency_ms=latency_ms,
            resolved_prompt_template_hash=resolved_prompt_template_hash,
        )

    def _acquire_rate_limit(self) -> None:
        """Acquire rate limit permission before making external call.

        Blocks until the rate limiter allows the request. If no limiter
        is configured, returns immediately (no throttling).

        Subclasses should call this at the start of their external call
        methods (e.g., chat_completion, post) before making the actual request.
        """
        if self._limiter is not None:
            self._limiter.acquire()

    def update_call_context(self, state_id: str, token_id: str | None = None) -> None:
        """Update the per-call audit scoping on a shared client.

        Used by providers that reuse a single client instance across multiple
        rows (e.g., AzureSearchProvider). Safe because row processing is serial
        within a transform — no concurrent calls to this method.

        Args:
            state_id: New state_id for subsequent calls
            token_id: New token_id for subsequent calls (None to clear)
        """
        self._state_id = state_id
        self._operation_id = None
        self._token_id = token_id

    def update_operation_call_context(self, operation_id: str) -> None:
        """Update the audit scoping to an operation parent."""
        self._state_id = None
        self._operation_id = operation_id
        self._token_id = None

    def close(self) -> None:
        """Release any resources held by the client.

        Default implementation is a no-op. Subclasses may override
        to close underlying connections or resources.
        """
        pass
