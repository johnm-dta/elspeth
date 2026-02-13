# src/elspeth/contracts/plugin_context.py
"""Plugin execution context.

The PluginContext carries everything a plugin might need during execution.
Phase 2 includes Optional placeholders for Phase 3 integrations.

Phase 3 Integration Points:
- landscape: LandscapeRecorder for audit trail
- tracer: OpenTelemetry Tracer for distributed tracing
- payload_store: PayloadStore for large blob storage
"""

from __future__ import annotations

import copy
import logging
from collections.abc import Callable
from contextlib import AbstractContextManager, nullcontext
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    # These types are available in Phase 3
    # Using string annotations to avoid import errors in Phase 2
    from opentelemetry.trace import Span, Tracer

    from elspeth.contracts import Call, CallStatus, CallType, PayloadStore, TransformErrorReason
    from elspeth.contracts.config.runtime import RuntimeConcurrencyConfig
    from elspeth.contracts.errors import ContractViolation
    from elspeth.contracts.identity import TokenInfo
    from elspeth.contracts.schema_contract import PipelineRow, SchemaContract
    from elspeth.core.landscape.recorder import LandscapeRecorder
    from elspeth.core.rate_limit import RateLimitRegistry
    from elspeth.plugins.clients.http import AuditedHTTPClient
    from elspeth.plugins.clients.llm import AuditedLLMClient

logger = logging.getLogger(__name__)


@dataclass
class ValidationErrorToken:
    """Token returned when recording a validation error.

    Allows tracking the quarantined row through the audit trail.
    """

    row_id: str
    node_id: str
    error_id: str | None = None  # Set if recorded to landscape
    destination: str = "discard"  # Sink name or "discard"


@dataclass
class TransformErrorToken:
    """Token returned when recording a transform error.

    Allows tracking the errored row through the audit trail.
    This is for LEGITIMATE processing errors, not transform bugs.
    """

    token_id: str
    transform_id: str
    error_id: str | None = None  # Set if recorded to landscape
    destination: str = "discard"  # Sink name or "discard"


@dataclass
class PluginContext:
    """Context passed to every plugin operation.

    Provides access to:
    - Run metadata (run_id, config)
    - Phase 3 integrations (landscape, tracer, payload_store)
    - Utility methods (get config values, start spans)

    Example:
        def process(self, row: PipelineRow, ctx: PluginContext) -> TransformResult:
            threshold = ctx.get("threshold", default=0.5)
            with ctx.start_span("my_operation"):
                result = do_work(row, threshold)
            return TransformResult.success(result, success_reason={"action": "processed"})
    """

    run_id: str
    config: dict[str, Any]

    # === Phase 3 Integration Points ===
    # Optional in Phase 2, populated by engine in Phase 3
    # Use string annotations to avoid import errors at runtime
    landscape: LandscapeRecorder | None = None
    tracer: Tracer | None = None
    payload_store: PayloadStore | None = None
    rate_limit_registry: RateLimitRegistry | None = None
    concurrency_config: RuntimeConcurrencyConfig | None = None

    # Additional metadata
    node_id: str | None = field(default=None)
    plugin_name: str | None = field(default=None)

    # === Row-Level Pipelining (BatchTransformMixin) ===
    # Set by orchestrator/executor when calling accept() on batch transforms.
    # Used by RowReorderBuffer for FIFO ordering and audit attribution.
    # IMPORTANT: This is derivative state - the executor must keep it synchronized
    # with the authoritative token flowing through the pipeline.
    token: TokenInfo | None = field(default=None)

    # === Batch Token Identity (Aggregation) ===
    # Set by AggregationExecutor.execute_flush() before calling batch-aware transforms.
    # Maps row index in the batch to the originating token_id. Batch transforms
    # use this to pass per-row token_id to audited clients for correct telemetry
    # attribution. When None, the transform falls back to ctx.token (single-token mode).
    batch_token_ids: list[str] | None = field(default=None)

    # === Schema Contract (Phase 3: Transform/Sink Integration) ===
    # Set by executor when processing transforms to enable contract-aware template
    # access (original header names). When transforms receive a plain dict (not
    # PipelineRow), they can still access the contract via ctx.contract.
    # This allows templates using {{ row["Original Header"] }} to resolve correctly.
    contract: SchemaContract | None = field(default=None)

    # === Phase 6: State & Call Recording ===
    # Set by executor to enable transforms to record external calls
    # Exactly one of state_id or operation_id should be set when recording calls
    state_id: str | None = field(default=None)  # For transform calls (via node_states)
    operation_id: str | None = field(default=None)  # For source/sink calls (via operations)
    # Note: call_index allocation is delegated to LandscapeRecorder.allocate_call_index()
    # to ensure coordination with audited clients. See P1-2026-01-31-context-record-call-bypasses-allocator.

    # === Phase 6: Audited Clients ===
    # Set by executor when processing LLM transforms
    llm_client: AuditedLLMClient | None = None
    http_client: AuditedHTTPClient | None = None

    # === Phase 6: Telemetry Callback ===
    # Callback to emit telemetry events for external calls.
    # Always present - when telemetry is disabled, orchestrator sets this to a no-op.
    # Plugins ALWAYS call this after successful Landscape recording - no None checks.
    telemetry_emit: Callable[[Any], None] = field(default=lambda event: None)

    # === Phase 6: Checkpoint API ===
    # Used by batch transforms (e.g., azure_batch_llm) for crash recovery.
    # The checkpoint stores batch_id, row_mapping, etc. between invocations.
    #
    # Checkpoints are keyed by node_id to support multiple batch transforms.
    # The orchestrator restores these from the BatchPendingError.checkpoint
    # when scheduling retries.
    _checkpoint: dict[str, Any] = field(default_factory=dict)

    # Batch checkpoints restored from previous BatchPendingError
    # Maps node_id -> checkpoint_data for each batch transform
    _batch_checkpoints: dict[str, dict[str, Any]] = field(default_factory=dict)

    def get_checkpoint(self) -> dict[str, Any] | None:
        """Get checkpoint state for batch transforms.

        Used by batch transforms to recover state after crashes.
        Returns None if no checkpoint exists (empty dict = no checkpoint).

        First checks for a restored batch checkpoint (from a previous
        BatchPendingError), then falls back to the local checkpoint.

        Returns:
            Checkpoint dict with batch state, or None if empty
        """
        # First check for restored batch checkpoint (keyed by node_id)
        if self.node_id and self.node_id in self._batch_checkpoints:
            restored = self._batch_checkpoints[self.node_id]
            if restored:
                return restored

        # Fall back to local checkpoint
        return self._checkpoint if self._checkpoint else None

    def update_checkpoint(self, data: dict[str, Any]) -> None:
        """Update checkpoint state with new data.

        Merges the provided data into the existing checkpoint.
        Used by batch transforms to save progress after submission.

        Args:
            data: Checkpoint data to merge (batch_id, row_mapping, etc.)
        """
        self._checkpoint.update(data)

    def clear_checkpoint(self) -> None:
        """Clear checkpoint state after batch completion.

        Called when batch processing completes successfully
        or when starting fresh after a failure.

        Clears both the local checkpoint and any restored batch checkpoint
        for the current node to prevent stale data on subsequent batches.
        """
        self._checkpoint.clear()
        # Also clear restored batch checkpoint to prevent stale resume data
        if self.node_id and self.node_id in self._batch_checkpoints:
            del self._batch_checkpoints[self.node_id]

    def get(self, key: str, *, default: Any = None) -> Any:
        """Get a config value by dotted path.

        Args:
            key: Dotted path like "nested.key"
            default: Value if key not found

        Returns:
            Config value or default
        """
        parts = key.split(".")
        value: Any = self.config
        for part in parts:
            if isinstance(value, dict) and part in value:
                value = value[part]
            else:
                return default
        return value

    def start_span(self, name: str) -> AbstractContextManager[Span | None]:
        """Start an OpenTelemetry span.

        Returns nullcontext if tracer not configured.

        Usage:
            with ctx.start_span("operation_name"):
                do_work()
        """
        if self.tracer is None:
            return nullcontext()
        return self.tracer.start_as_current_span(name)

    def record_call(
        self,
        call_type: CallType,
        status: CallStatus,
        request_data: dict[str, Any],
        response_data: dict[str, Any] | None = None,
        error: dict[str, Any] | None = None,
        latency_ms: float | None = None,
        *,
        provider: str = "unknown",
    ) -> Call | None:
        """Record an external API call to the audit trail and emit telemetry.

        Provides a convenient way for plugins to record external calls
        without managing call indices manually. Routes to the appropriate
        recorder method based on whether state_id or operation_id is set.

        After recording to Landscape (the legal record), emits an
        ExternalCallCompleted telemetry event for operational visibility.

        Args:
            call_type: Type of call (LLM, HTTP, SQL, FILESYSTEM)
            status: Outcome (SUCCESS, ERROR)
            request_data: Request payload (will be hashed)
            response_data: Response payload (optional for errors)
            error: Error details if status is ERROR
            latency_ms: Call duration in milliseconds
            provider: Provider name for telemetry (e.g., "openrouter", "azure")

        Returns:
            The recorded Call, or None if landscape not configured

        Raises:
            FrameworkBugError: If neither or both of state_id and operation_id are set
        """
        from elspeth.contracts import FrameworkBugError

        if self.landscape is None:
            logger.warning("External call not recorded (no landscape)")
            return None

        # Enforce XOR: exactly one of state_id or operation_id must be set
        has_state = self.state_id is not None
        has_operation = self.operation_id is not None

        if has_state and has_operation:
            raise FrameworkBugError(
                f"record_call() called with BOTH state_id and operation_id set. "
                f"state_id={self.state_id}, operation_id={self.operation_id}. "
                f"This is a framework bug - context should have exactly one parent."
            )

        if not has_state and not has_operation:
            raise FrameworkBugError(
                f"record_call() called without state_id or operation_id. "
                f"Context state: run_id={self.run_id}, node_id={self.node_id}. "
                f"This is a framework bug - context should have been set by orchestrator/executor."
            )

        # Route to appropriate recorder method
        if has_state:
            # Delegate call_index allocation to centralized LandscapeRecorder.
            # This ensures UNIQUE(state_id, call_index) when mixing ctx.record_call()
            # with audited clients (AuditedLLMClient, AuditedHTTPClient), which also
            # use recorder.allocate_call_index(). See P1-2026-01-31-context-record-call-bypasses-allocator.
            assert self.state_id is not None  # Guarded by has_state check above
            call_index = self.landscape.allocate_call_index(self.state_id)

            recorded_call = self.landscape.record_call(
                state_id=self.state_id,
                call_index=call_index,
                call_type=call_type,
                status=status,
                request_data=request_data,
                response_data=response_data,
                error=error,
                latency_ms=latency_ms,
            )
            parent_id: str = self.state_id
        else:
            # Operation call - recorder handles call index allocation
            assert self.operation_id is not None  # Guarded by has_operation check above
            recorded_call = self.landscape.record_operation_call(
                operation_id=self.operation_id,
                call_type=call_type,
                status=status,
                request_data=request_data,
                response_data=response_data,
                error=error,
                latency_ms=latency_ms,
                provider=provider,
            )
            parent_id = self.operation_id

        # Emit telemetry AFTER successful Landscape recording
        # Wrapped in try/except to prevent telemetry failures from affecting callers
        try:
            from elspeth.contracts.enums import CallType as CallTypeEnum
            from elspeth.contracts.events import ExternalCallCompleted
            from elspeth.core.canonical import stable_hash

            # Snapshot payloads so async telemetry exports can't drift from call-time hashes.
            request_snapshot = copy.deepcopy(request_data)
            response_snapshot = copy.deepcopy(response_data) if response_data is not None else None

            # Extract token usage for LLM calls if available
            token_usage = None
            if call_type == CallTypeEnum.LLM and response_snapshot is not None:
                usage = response_snapshot.get("usage")
                if usage and isinstance(usage, dict):
                    token_usage = usage

            token_id = None
            if has_state:
                if self.token is not None:
                    token_id = self.token.token_id
                elif self.state_id is not None:
                    node_state = self.landscape.get_node_state(self.state_id)
                    if node_state is not None:
                        token_id = node_state.token_id

            self.telemetry_emit(
                ExternalCallCompleted(
                    timestamp=datetime.now(UTC),
                    run_id=self.run_id,
                    # Use correct field based on context type
                    state_id=self.state_id if has_state else None,
                    operation_id=self.operation_id if has_operation else None,
                    token_id=token_id,
                    call_type=call_type,
                    provider=provider,
                    status=status,
                    latency_ms=latency_ms or 0.0,
                    request_hash=stable_hash(request_snapshot),
                    response_hash=stable_hash(response_snapshot) if response_snapshot is not None else None,
                    request_payload=request_snapshot,  # Full request snapshot for observability
                    response_payload=response_snapshot,  # Full response snapshot for observability
                    token_usage=token_usage,
                )
            )
        except Exception as tel_err:
            # Telemetry failure must not corrupt the call recording
            logger.warning(
                "telemetry_emit_failed in record_call",
                extra={
                    "error": str(tel_err),
                    "error_type": type(tel_err).__name__,
                    "run_id": self.run_id,
                    "parent_id": parent_id,
                },
            )

        return recorded_call

    def record_validation_error(
        self,
        row: Any,
        error: str,
        schema_mode: str,
        destination: str,
        *,
        contract_violation: ContractViolation | None = None,
    ) -> ValidationErrorToken:
        """Record a validation error for audit trail.

        Called by sources when row validation fails. The row will be
        quarantined (not processed further) but the error is recorded
        for complete audit coverage.

        Args:
            row: The row data that failed validation (may be non-dict for
                 malformed external data like JSON arrays containing primitives)
            error: Description of the validation failure
            schema_mode: "fixed", "flexible", "observed", or "parse" (parse = file-level parse error)
            destination: Sink name where row is routed, or "discard"
            contract_violation: Optional contract violation details for structured auditing

        Returns:
            ValidationErrorToken for tracking the quarantined row
        """
        from elspeth.core.canonical import repr_hash, stable_hash

        # Generate row_id from content hash if not present
        # External data may be non-dict (e.g., JSON array containing primitives),
        # so we must check isinstance before accessing dict keys
        if isinstance(row, dict) and "id" in row:
            row_id = str(row["id"])
        else:
            # Try canonical hash first, fall back to repr() hash for non-serializable data
            # This is Tier-3 (external data) - we must record what we saw, even if malformed
            try:
                row_id = stable_hash(row)[:16]
            except (ValueError, TypeError) as e:
                # Non-canonical data (NaN, Infinity, or other non-serializable types)
                # Hash the repr() instead - not canonical, but preserves audit trail
                row_preview = repr(row)[:200] + "..." if len(repr(row)) > 200 else repr(row)
                logger.warning(
                    "Row data not canonically serializable, using repr() hash: %s | Row preview: %s",
                    str(e),
                    row_preview,
                )
                row_id = repr_hash(row)[:16]

        if self.landscape is None:
            logger.warning(
                "Validation error not recorded (no landscape): %s",
                error,
            )
            return ValidationErrorToken(
                row_id=row_id,
                node_id=self.node_id or "unknown",
                destination=destination,
            )

        # Record to landscape audit trail
        error_id = self.landscape.record_validation_error(
            run_id=self.run_id,
            node_id=self.node_id,
            row_data=row,
            error=error,
            schema_mode=schema_mode,
            destination=destination,
            contract_violation=contract_violation,
        )

        return ValidationErrorToken(
            row_id=row_id,
            node_id=self.node_id or "unknown",
            error_id=error_id,
            destination=destination,
        )

    def record_transform_error(
        self,
        token_id: str,
        transform_id: str,
        row: dict[str, Any] | PipelineRow,
        error_details: TransformErrorReason,
        destination: str,
    ) -> TransformErrorToken:
        """Record a transform processing error for audit trail.

        Called when a transform returns TransformResult.error().
        This is for legitimate errors, NOT transform bugs (which crash).

        Args:
            token_id: Token ID for the row being processed
            transform_id: Transform that returned the error
            row: The row data that could not be processed
            error_details: Error details from TransformResult.error() (TransformErrorReason TypedDict)
            destination: Sink name where row is routed, or "discard"

        Returns:
            TransformErrorToken for tracking
        """
        if self.landscape is None:
            logger.warning(
                "Transform error not recorded (no landscape): %s - %s",
                transform_id,
                error_details,
            )
            return TransformErrorToken(
                token_id=token_id,
                transform_id=transform_id,
                destination=destination,
            )

        error_id = self.landscape.record_transform_error(
            run_id=self.run_id,
            token_id=token_id,
            transform_id=transform_id,
            row_data=row,
            error_details=error_details,
            destination=destination,
        )

        return TransformErrorToken(
            token_id=token_id,
            transform_id=transform_id,
            error_id=error_id,
            destination=destination,
        )
