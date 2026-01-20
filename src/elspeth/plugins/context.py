# src/elspeth/plugins/context.py
"""Plugin execution context.

The PluginContext carries everything a plugin might need during execution.
Phase 2 includes Optional placeholders for Phase 3 integrations.

Phase 3 Integration Points:
- landscape: LandscapeRecorder for audit trail
- tracer: OpenTelemetry Tracer for distributed tracing
- payload_store: PayloadStore for large blob storage
"""

import logging
from contextlib import AbstractContextManager, nullcontext
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    # These types are available in Phase 3
    # Using string annotations to avoid import errors in Phase 2
    from opentelemetry.trace import Span, Tracer

    from elspeth.contracts import Call, CallStatus, CallType
    from elspeth.core.landscape.recorder import LandscapeRecorder
    from elspeth.core.payload_store import PayloadStore
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
        def process(self, row: dict, ctx: PluginContext) -> TransformResult:
            threshold = ctx.get("threshold", default=0.5)
            with ctx.start_span("my_operation"):
                result = do_work(row, threshold)
            return TransformResult.success(result)
    """

    run_id: str
    config: dict[str, Any]

    # === Phase 3 Integration Points ===
    # Optional in Phase 2, populated by engine in Phase 3
    # Use string annotations to avoid import errors at runtime
    landscape: "LandscapeRecorder | None" = None
    tracer: "Tracer | None" = None
    payload_store: "PayloadStore | None" = None

    # Additional metadata
    node_id: str | None = field(default=None)
    plugin_name: str | None = field(default=None)

    # === Phase 6: State & Call Recording ===
    # Set by executor to enable transforms to record external calls
    state_id: str | None = field(default=None)
    _call_index: int = field(default=0)

    # === Phase 6: Audited Clients ===
    # Set by executor when processing LLM transforms
    llm_client: "AuditedLLMClient | None" = None
    http_client: "AuditedHTTPClient | None" = None

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

    def start_span(self, name: str) -> AbstractContextManager["Span | None"]:
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
        call_type: "CallType",
        status: "CallStatus",
        request_data: dict[str, Any],
        response_data: dict[str, Any] | None = None,
        error: dict[str, Any] | None = None,
        latency_ms: float | None = None,
    ) -> "Call | None":
        """Record an external API call to the audit trail.

        Provides a convenient way for transforms to record external calls
        without managing call indices manually. Requires state_id to be set.

        Args:
            call_type: Type of call (LLM, HTTP, SQL, FILESYSTEM)
            status: Outcome (SUCCESS, ERROR)
            request_data: Request payload (will be hashed)
            response_data: Response payload (optional for errors)
            error: Error details if status is ERROR
            latency_ms: Call duration in milliseconds

        Returns:
            The recorded Call, or None if landscape not configured

        Raises:
            RuntimeError: If state_id is not set (transform not in execution context)
        """
        if self.landscape is None:
            logger.warning("External call not recorded (no landscape)")
            return None

        if self.state_id is None:
            raise RuntimeError("Cannot record call: state_id not set. Ensure transform is being executed through the engine.")

        call_index = self._call_index
        self._call_index += 1

        return self.landscape.record_call(
            state_id=self.state_id,
            call_index=call_index,
            call_type=call_type,
            status=status,
            request_data=request_data,
            response_data=response_data,
            error=error,
            latency_ms=latency_ms,
        )

    def record_validation_error(
        self,
        row: Any,
        error: str,
        schema_mode: str,
        destination: str,
    ) -> ValidationErrorToken:
        """Record a validation error for audit trail.

        Called by sources when row validation fails. The row will be
        quarantined (not processed further) but the error is recorded
        for complete audit coverage.

        Args:
            row: The row data that failed validation (may be non-dict for
                 malformed external data like JSON arrays containing primitives)
            error: Description of the validation failure
            schema_mode: "strict", "free", or "dynamic"
            destination: Sink name where row is routed, or "discard"

        Returns:
            ValidationErrorToken for tracking the quarantined row
        """
        from elspeth.core.canonical import stable_hash

        # Generate row_id from content hash if not present
        # External data may be non-dict (e.g., JSON array containing primitives),
        # so we must check isinstance before accessing dict keys
        if isinstance(row, dict) and "id" in row:
            row_id = str(row["id"])
        else:
            row_id = stable_hash(row)[:16]

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
        row: dict[str, Any],
        error_details: dict[str, Any],
        destination: str,
    ) -> TransformErrorToken:
        """Record a transform processing error for audit trail.

        Called when a transform returns TransformResult.error().
        This is for legitimate errors, NOT transform bugs (which crash).

        Args:
            token_id: Token ID for the row being processed
            transform_id: Transform that returned the error
            row: The row data that could not be processed
            error_details: Error details from TransformResult.error()
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

    def route_to_sink(
        self,
        sink_name: str,
        row: dict[str, Any],
        metadata: dict[str, Any] | None = None,
    ) -> None:
        """Route a row to a named sink.

        NOTE: This is a Phase 2 stub. Currently logs the routing action.
        Full implementation will integrate with DAG executor to actually
        deliver rows to the named sink.

        Args:
            sink_name: Name of the destination sink
            row: The row data to route
            metadata: Optional metadata about why row was routed
        """
        # Phase 2 stub - log the action
        logger.info(
            "route_to_sink: %s -> %s (metadata=%s)",
            self.node_id,
            sink_name,
            metadata,
        )
