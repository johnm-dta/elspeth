# src/elspeth/plugins/context.py
"""Plugin execution context.

The PluginContext carries everything a plugin might need during execution.
Phase 2 includes Optional placeholders for Phase 3 integrations.

Phase 3 Integration Points:
- landscape: LandscapeRecorder for audit trail
- tracer: OpenTelemetry Tracer for distributed tracing
- payload_store: PayloadStore for large blob storage
"""

from contextlib import nullcontext
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, ContextManager

if TYPE_CHECKING:
    # These types are available in Phase 3
    # Using string annotations to avoid import errors in Phase 2
    from opentelemetry.trace import Span, Tracer

    from elspeth.core.payload_store import PayloadStore


# Protocol for Landscape recorder (Phase 3)
# Defined here so plugins can type-hint against it
class LandscapeRecorder:
    """Protocol for Landscape audit recording.

    Implemented in Phase 3. Plugins can optionally use this
    for custom audit events.
    """

    def record_event(self, event_type: str, data: dict[str, Any]) -> None:
        """Record a custom audit event."""
        ...


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
    landscape: LandscapeRecorder | None = None
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

    def start_span(self, name: str) -> ContextManager["Span | None"]:
        """Start an OpenTelemetry span.

        Returns nullcontext if tracer not configured.

        Usage:
            with ctx.start_span("operation_name"):
                do_work()
        """
        if self.tracer is None:
            return nullcontext()
        return self.tracer.start_as_current_span(name)
