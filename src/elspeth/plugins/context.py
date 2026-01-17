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

    from elspeth.core.landscape.recorder import LandscapeRecorder
    from elspeth.core.payload_store import PayloadStore

logger = logging.getLogger(__name__)


@dataclass
class ValidationErrorToken:
    """Token returned when recording a validation error.

    Allows tracking the quarantined row through the audit trail.
    """

    row_id: str
    node_id: str
    error_id: str | None = None  # Set if recorded to landscape


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

    def record_validation_error(
        self,
        row: dict[str, Any],
        error: str,
        schema_mode: str,
    ) -> ValidationErrorToken:
        """Record a validation error for audit trail.

        Called by sources when row validation fails. The row will be
        quarantined (not processed further) but the error is recorded
        for complete audit coverage.

        Args:
            row: The row data that failed validation
            error: Description of the validation failure
            schema_mode: "strict", "free", or "dynamic"

        Returns:
            ValidationErrorToken for tracking the quarantined row
        """
        from elspeth.core.canonical import stable_hash

        # Generate row_id from content hash if not present
        row_id = str(row["id"]) if "id" in row else stable_hash(row)[:16]

        if self.landscape is None:
            logger.warning(
                "Validation error not recorded (no landscape): %s",
                error,
            )
            return ValidationErrorToken(
                row_id=row_id,
                node_id=self.node_id or "unknown",
            )

        # Record to landscape audit trail
        error_id = self.landscape.record_validation_error(
            run_id=self.run_id,
            node_id=self.node_id,
            row_data=row,
            error=error,
            schema_mode=schema_mode,
        )

        return ValidationErrorToken(
            row_id=row_id,
            node_id=self.node_id or "unknown",
            error_id=error_id,
        )
