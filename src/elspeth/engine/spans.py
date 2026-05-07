"""OpenTelemetry span factory for SDA Engine.

Provides structured span creation for pipeline execution.
Falls back to no-op mode when no tracer is configured.

Span Hierarchy (span names are static; plugin identity is in attributes):
    run                          [run.id=<run_id>]
    ├── source                   [plugin.name=<name>]
    ├── row                      [row.id=<row_id>, token.id=<token_id>]
    │   ├── transform            [plugin.name=<name>]
    │   └── sink                 [plugin.name=<name>]
    └── aggregation              [plugin.name=<name>]
"""

from collections.abc import Iterator, Sequence
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
    def _make_span(
        self,
        name: str,
        attributes: dict[str, Any],
    ) -> Iterator["Span | NoOpSpan"]:
        """Create a span with attributes, or yield no-op if tracing is disabled.

        Args:
            name: Span name (e.g., "run", "source:csv", "transform:field_mapper")
            attributes: Key-value pairs to set on the span. Values must already
                be in their final form (e.g., token_ids already converted to tuple).
        """
        if self._tracer is None:
            yield self._NOOP_SPAN
            return

        with self._tracer.start_as_current_span(name) as span:
            for key, value in attributes.items():
                span.set_attribute(key, value)
            yield span

    @contextmanager
    def run_span(self, run_id: str) -> Iterator["Span | NoOpSpan"]:
        """Create a span for the entire run.

        Args:
            run_id: Run identifier

        Yields:
            Span or NoOpSpan if tracing disabled (never None - uniform interface)
        """
        with self._make_span("run", {"run.id": run_id}) as span:
            yield span

    @contextmanager
    def source_span(self, source_name: str) -> Iterator["Span | NoOpSpan"]:
        """Create a span for source loading.

        Args:
            source_name: Name of the source plugin

        Yields:
            Span or NoOpSpan
        """
        with self._make_span("source", {"plugin.name": source_name, "plugin.type": "source"}) as span:
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
        with self._make_span("row", {"row.id": row_id, "token.id": token_id}) as span:
            yield span

    @contextmanager
    def transform_span(
        self,
        transform_name: str,
        *,
        node_id: str | None = None,
        input_hash: str | None = None,
        token_id: str | None = None,
        token_ids: Sequence[str] | None = None,
    ) -> Iterator["Span | NoOpSpan"]:
        """Create a span for a transform operation.

        Args:
            transform_name: Name of the transform plugin
            node_id: Unique node identifier for disambiguation
            input_hash: Optional input data hash
            token_id: Token identifier for single-row transforms
            token_ids: Token identifiers for batch transforms (aggregation flush)

        Note:
            Use token_id for single-row transforms (most common case).
            Use token_ids for batch/aggregation transforms that process multiple tokens.
            These are mutually exclusive - if both provided, token_ids takes precedence.

            node_id enables correlation with Landscape node_states when multiple
            instances of the same plugin type exist in a pipeline.

        Yields:
            Span or NoOpSpan
        """
        attrs: dict[str, Any] = {"plugin.name": transform_name, "plugin.type": "transform"}
        if node_id is not None:
            attrs["node.id"] = node_id
        if input_hash is not None:
            attrs["input.hash"] = input_hash
        # Token tracking for accurate child token attribution
        if token_ids is not None:
            attrs["token.ids"] = tuple(token_ids)
        elif token_id is not None:
            attrs["token.id"] = token_id
        with self._make_span("transform", attrs) as span:
            yield span

    @contextmanager
    def gate_span(
        self,
        gate_name: str,
        *,
        node_id: str | None = None,
        input_hash: str | None = None,
        token_id: str | None = None,
    ) -> Iterator["Span | NoOpSpan"]:
        """Create a span for a gate operation.

        Args:
            gate_name: Name of the gate (from GateSettings)
            node_id: Unique node identifier for disambiguation
            input_hash: Optional input data hash
            token_id: Token identifier for the token being evaluated

        Yields:
            Span or NoOpSpan
        """
        attrs: dict[str, Any] = {"plugin.name": gate_name, "plugin.type": "gate"}
        if node_id is not None:
            attrs["node.id"] = node_id
        if input_hash is not None:
            attrs["input.hash"] = input_hash
        if token_id is not None:
            attrs["token.id"] = token_id
        with self._make_span("gate", attrs) as span:
            yield span

    @contextmanager
    def aggregation_span(
        self,
        aggregation_name: str,
        *,
        node_id: str | None = None,
        input_hash: str | None = None,
        batch_id: str | None = None,
        token_ids: Sequence[str] | None = None,
    ) -> Iterator["Span | NoOpSpan"]:
        """Create a span for an aggregation flush.

        Args:
            aggregation_name: Name of the aggregation plugin
            node_id: Unique node identifier for disambiguation
            input_hash: Input data hash for trace-to-audit correlation
            batch_id: Optional batch identifier
            token_ids: Token identifiers in the batch

        Note:
            Aggregation batches process multiple tokens, so this uses token_ids (plural).
            The token.ids attribute is a tuple for OpenTelemetry compatibility.

        Yields:
            Span or NoOpSpan
        """
        attrs: dict[str, Any] = {"plugin.name": aggregation_name, "plugin.type": "aggregation"}
        if node_id is not None:
            attrs["node.id"] = node_id
        if input_hash is not None:
            attrs["input.hash"] = input_hash
        if batch_id is not None:
            attrs["batch.id"] = batch_id
        if token_ids is not None:
            attrs["token.ids"] = tuple(token_ids)
        with self._make_span("aggregation", attrs) as span:
            yield span

    @contextmanager
    def sink_span(
        self,
        sink_name: str,
        *,
        node_id: str | None = None,
        token_ids: Sequence[str] | None = None,
    ) -> Iterator["Span | NoOpSpan"]:
        """Create a span for a sink write.

        Args:
            sink_name: Name of the sink plugin
            node_id: Unique node identifier for disambiguation
            token_ids: Token identifiers being written in this batch

        Note:
            Sinks batch-write multiple tokens, so this uses token_ids (plural).
            The token.ids attribute is a tuple for OpenTelemetry compatibility.

        Yields:
            Span or NoOpSpan
        """
        attrs: dict[str, Any] = {"plugin.name": sink_name, "plugin.type": "sink"}
        if node_id is not None:
            attrs["node.id"] = node_id
        if token_ids is not None:
            attrs["token.ids"] = tuple(token_ids)
        with self._make_span("sink", attrs) as span:
            yield span
