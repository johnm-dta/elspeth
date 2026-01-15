# src/elspeth/engine/spans.py
"""OpenTelemetry span factory for SDA Engine.

Provides structured span creation for pipeline execution.
Falls back to no-op mode when no tracer is configured.

Span Hierarchy:
    run:{run_id}
    ├── source:{source_name}
    │   └── load
    ├── row:{row_id}
    │   ├── transform:{transform_name}
    │   ├── gate:{gate_name}
    │   └── sink:{sink_name}
    └── aggregation:{agg_name}
        └── flush
"""

from collections.abc import Iterator
from contextlib import contextmanager
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from opentelemetry.trace import Span, Tracer


class NoOpSpan:
    """No-op span for when tracing is disabled."""

    def set_attribute(self, key: str, value: Any) -> None:
        """No-op."""
        pass

    def set_status(self, status: Any) -> None:
        """No-op."""
        pass

    def record_exception(self, exception: Exception) -> None:
        """No-op."""
        pass

    def is_recording(self) -> bool:
        """Always False for no-op."""
        return False


class SpanFactory:
    """Factory for creating OpenTelemetry spans.

    When no tracer is provided, all span methods return no-op contexts.

    Example:
        factory = SpanFactory(tracer=opentelemetry.trace.get_tracer("elspeth"))

        with factory.run_span("run-001") as span:
            with factory.row_span("row-001", "token-001") as row_span:
                with factory.transform_span("my_transform") as transform_span:
                    # Do work
                    pass
    """

    # Singleton no-op span to avoid repeated allocations
    _NOOP_SPAN = NoOpSpan()

    def __init__(self, tracer: "Tracer | None" = None) -> None:
        """Initialize with optional tracer.

        Args:
            tracer: OpenTelemetry tracer. If None, spans are no-ops.
        """
        self._tracer = tracer

    @property
    def enabled(self) -> bool:
        """Whether tracing is enabled."""
        return self._tracer is not None

    @contextmanager
    def run_span(self, run_id: str) -> Iterator["Span | NoOpSpan"]:
        """Create a span for the entire run.

        Args:
            run_id: Run identifier

        Yields:
            Span or NoOpSpan if tracing disabled (never None - uniform interface)
        """
        if self._tracer is None:
            yield self._NOOP_SPAN
            return

        with self._tracer.start_as_current_span(f"run:{run_id}") as span:
            span.set_attribute("run.id", run_id)
            yield span

    @contextmanager
    def source_span(self, source_name: str) -> Iterator["Span | NoOpSpan"]:
        """Create a span for source loading.

        Args:
            source_name: Name of the source plugin

        Yields:
            Span or NoOpSpan
        """
        if self._tracer is None:
            yield self._NOOP_SPAN
            return

        with self._tracer.start_as_current_span(f"source:{source_name}") as span:
            span.set_attribute("plugin.name", source_name)
            span.set_attribute("plugin.type", "source")
            yield span

    @contextmanager
    def row_span(
        self,
        row_id: str,
        token_id: str,
    ) -> Iterator["Span | NoOpSpan"]:
        """Create a span for processing a row.

        Args:
            row_id: Row identifier
            token_id: Token identifier

        Yields:
            Span or NoOpSpan
        """
        if self._tracer is None:
            yield self._NOOP_SPAN
            return

        with self._tracer.start_as_current_span(f"row:{row_id}") as span:
            span.set_attribute("row.id", row_id)
            span.set_attribute("token.id", token_id)
            yield span

    @contextmanager
    def transform_span(
        self,
        transform_name: str,
        *,
        input_hash: str | None = None,
    ) -> Iterator["Span | NoOpSpan"]:
        """Create a span for a transform operation.

        Args:
            transform_name: Name of the transform plugin
            input_hash: Optional input data hash

        Yields:
            Span or NoOpSpan
        """
        if self._tracer is None:
            yield self._NOOP_SPAN
            return

        with self._tracer.start_as_current_span(f"transform:{transform_name}") as span:
            span.set_attribute("plugin.name", transform_name)
            span.set_attribute("plugin.type", "transform")
            if input_hash:
                span.set_attribute("input.hash", input_hash)
            yield span

    @contextmanager
    def gate_span(
        self,
        gate_name: str,
        *,
        input_hash: str | None = None,
    ) -> Iterator["Span | NoOpSpan"]:
        """Create a span for a gate operation.

        Args:
            gate_name: Name of the gate plugin
            input_hash: Optional input data hash

        Yields:
            Span or NoOpSpan
        """
        if self._tracer is None:
            yield self._NOOP_SPAN
            return

        with self._tracer.start_as_current_span(f"gate:{gate_name}") as span:
            span.set_attribute("plugin.name", gate_name)
            span.set_attribute("plugin.type", "gate")
            if input_hash:
                span.set_attribute("input.hash", input_hash)
            yield span

    @contextmanager
    def aggregation_span(
        self,
        aggregation_name: str,
        *,
        batch_id: str | None = None,
    ) -> Iterator["Span | NoOpSpan"]:
        """Create a span for an aggregation flush.

        Args:
            aggregation_name: Name of the aggregation plugin
            batch_id: Optional batch identifier

        Yields:
            Span or NoOpSpan
        """
        if self._tracer is None:
            yield self._NOOP_SPAN
            return

        with self._tracer.start_as_current_span(
            f"aggregation:{aggregation_name}"
        ) as span:
            span.set_attribute("plugin.name", aggregation_name)
            span.set_attribute("plugin.type", "aggregation")
            if batch_id:
                span.set_attribute("batch.id", batch_id)
            yield span

    @contextmanager
    def sink_span(
        self,
        sink_name: str,
    ) -> Iterator["Span | NoOpSpan"]:
        """Create a span for a sink write.

        Args:
            sink_name: Name of the sink plugin

        Yields:
            Span or NoOpSpan
        """
        if self._tracer is None:
            yield self._NOOP_SPAN
            return

        with self._tracer.start_as_current_span(f"sink:{sink_name}") as span:
            span.set_attribute("plugin.name", sink_name)
            span.set_attribute("plugin.type", "sink")
            yield span
