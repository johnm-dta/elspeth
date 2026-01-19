# src/elspeth/plugins/base.py
"""Base classes for plugin implementations.

These provide common functionality and ensure proper interface compliance.
Plugins can subclass these for convenience, or implement protocols directly.

Phase 3 Integration:
- Lifecycle hooks (on_start, on_complete) are called by engine
- PluginContext is provided by engine with landscape/tracer/payload_store
"""

from abc import ABC, abstractmethod
from collections.abc import Iterator
from typing import Any

from elspeth.contracts import ArtifactDescriptor, Determinism, PluginSchema, SourceRow
from elspeth.plugins.context import PluginContext
from elspeth.plugins.results import (
    GateResult,
    TransformResult,
)


class BaseTransform(ABC):
    """Base class for stateless row transforms.

    Subclass and implement process() to create a transform.

    For batch-aware transforms (used in aggregation nodes):
    - Set is_batch_aware = True
    - process() will receive list[dict] when used in aggregation

    Example:
        class MyTransform(BaseTransform):
            name = "my_transform"
            input_schema = InputSchema
            output_schema = OutputSchema

            def process(self, row: dict, ctx: PluginContext) -> TransformResult:
                return TransformResult.success({**row, "new_field": "value"})
    """

    name: str
    input_schema: type[PluginSchema]
    output_schema: type[PluginSchema]
    node_id: str | None = None  # Set by orchestrator after registration

    # Metadata for Phase 3 audit/reproducibility
    determinism: Determinism = Determinism.DETERMINISTIC
    plugin_version: str = "0.0.0"

    # Batch support - override to True for batch-aware transforms
    # When True, engine may pass list[dict] instead of single dict to process()
    is_batch_aware: bool = False

    # Error routing configuration (WP-11.99b)
    # Transforms extending TransformDataConfig override this from config.
    # None means: transform doesn't return errors, OR errors are bugs.
    _on_error: str | None = None

    def __init__(self, config: dict[str, Any]) -> None:
        """Initialize with configuration."""
        self.config = config

    @abstractmethod
    def process(
        self,
        row: dict[str, Any],
        ctx: PluginContext,
    ) -> TransformResult:
        """Process a single row.

        Args:
            row: Input row matching input_schema
            ctx: Plugin context

        Returns:
            TransformResult with processed row or error
        """
        ...

    def close(self) -> None:  # noqa: B027
        """Clean up resources after pipeline completion.

        Called once after all rows have been processed. Override for closing
        connections, flushing buffers, or releasing external resources.
        """

    # === Lifecycle Hooks (Phase 3) ===
    # These are intentionally empty - optional hooks for subclasses to override

    def on_start(self, ctx: PluginContext) -> None:  # noqa: B027
        """Called at the start of each run.

        Override for per-run initialization.
        """

    def on_complete(self, ctx: PluginContext) -> None:  # noqa: B027
        """Called at the end of each run.

        Override for cleanup.
        """


class BaseGate(ABC):
    """Base class for gate transforms (routing decisions).

    Subclass and implement evaluate() to create a gate.

    Example:
        class SafetyGate(BaseGate):
            name = "safety"
            input_schema = RowSchema
            output_schema = RowSchema

            def evaluate(self, row: dict, ctx: PluginContext) -> GateResult:
                if self._is_suspicious(row):
                    return GateResult(
                        row=row,
                        action=RoutingAction.route("suspicious"),  # Resolved via routes config
                    )
                return GateResult(row=row, action=RoutingAction.route("normal"))
    """

    name: str
    input_schema: type[PluginSchema]
    output_schema: type[PluginSchema]
    node_id: str | None = None  # Set by orchestrator after registration

    # Metadata for Phase 3 audit/reproducibility
    determinism: Determinism = Determinism.DETERMINISTIC
    plugin_version: str = "0.0.0"

    def __init__(self, config: dict[str, Any]) -> None:
        """Initialize with configuration."""
        self.config = config

    @abstractmethod
    def evaluate(
        self,
        row: dict[str, Any],
        ctx: PluginContext,
    ) -> GateResult:
        """Evaluate a row and decide routing.

        Args:
            row: Input row
            ctx: Plugin context

        Returns:
            GateResult with routing decision
        """
        ...

    def close(self) -> None:  # noqa: B027
        """Clean up resources after pipeline completion.

        Called once after all rows have been processed. Override for closing
        connections, flushing buffers, or releasing external resources.
        """

    # === Lifecycle Hooks (Phase 3) ===

    def on_start(self, ctx: PluginContext) -> None:  # noqa: B027
        """Called at start of run."""

    def on_complete(self, ctx: PluginContext) -> None:  # noqa: B027
        """Called at end of run."""


# NOTE: BaseAggregation was DELETED in aggregation structural cleanup.
# Aggregation is now fully structural:
# - Engine buffers rows internally
# - Engine evaluates triggers (WP-06)
# - Engine calls batch-aware Transform.process(rows: list[dict])
# Use is_batch_aware=True on BaseTransform for batch processing.


class BaseSink(ABC):
    """Base class for sink plugins.

    Subclass and implement write(), flush(), close().

    Example:
        class CSVSink(BaseSink):
            name = "csv"
            input_schema = RowSchema
            idempotent = False

            def write(self, rows: list[dict], ctx: PluginContext) -> ArtifactDescriptor:
                for row in rows:
                    self._writer.writerow(row)
                return ArtifactDescriptor.for_file(
                    path=self._path,
                    content_hash=self._compute_hash(),
                    size_bytes=self._file.tell(),
                )

            def flush(self) -> None:
                self._file.flush()

            def close(self) -> None:
                self._file.close()
    """

    name: str
    input_schema: type[PluginSchema]
    idempotent: bool = False
    node_id: str | None = None  # Set by orchestrator after registration

    # Metadata for Phase 3 audit/reproducibility
    determinism: Determinism = Determinism.IO_WRITE
    plugin_version: str = "0.0.0"

    def __init__(self, config: dict[str, Any]) -> None:
        """Initialize with configuration."""
        self.config = config

    @abstractmethod
    def write(
        self,
        rows: list[dict[str, Any]],
        ctx: PluginContext,
    ) -> ArtifactDescriptor:
        """Write a batch of rows to the sink.

        Args:
            rows: List of row dicts to write
            ctx: Plugin context

        Returns:
            ArtifactDescriptor with content_hash and size_bytes
        """
        ...

    @abstractmethod
    def flush(self) -> None:
        """Flush buffered data."""
        ...

    @abstractmethod
    def close(self) -> None:
        """Close and release resources."""
        ...

    # === Lifecycle Hooks (Phase 3) ===

    def on_start(self, ctx: PluginContext) -> None:  # noqa: B027
        """Called at start of run."""

    def on_complete(self, ctx: PluginContext) -> None:  # noqa: B027
        """Called at end of run (before close)."""


class BaseSource(ABC):
    """Base class for source plugins.

    Subclass and implement load() and close().

    Example:
        class CSVSource(BaseSource):
            name = "csv"
            output_schema = RowSchema

            def load(self, ctx: PluginContext) -> Iterator[SourceRow]:
                with open(self.config["path"]) as f:
                    reader = csv.DictReader(f)
                    for row in reader:
                        yield SourceRow.valid(row)

            def close(self) -> None:
                pass  # File already closed by context manager
    """

    name: str
    output_schema: type[PluginSchema]
    node_id: str | None = None  # Set by orchestrator after registration

    # Metadata for Phase 3 audit/reproducibility
    determinism: Determinism = Determinism.IO_READ
    plugin_version: str = "0.0.0"

    def __init__(self, config: dict[str, Any]) -> None:
        """Initialize with configuration."""
        self.config = config

    @abstractmethod
    def load(self, ctx: PluginContext) -> Iterator[SourceRow]:
        """Load and yield rows from the source.

        Args:
            ctx: Plugin context

        Yields:
            SourceRow for each row - either SourceRow.valid() for rows that
            passed validation, or SourceRow.quarantined() for invalid rows.
        """
        ...

    @abstractmethod
    def close(self) -> None:
        """Clean up resources."""
        ...

    # === Lifecycle Hooks (Phase 3) ===

    def on_start(self, ctx: PluginContext) -> None:  # noqa: B027
        """Called before load()."""

    def on_complete(self, ctx: PluginContext) -> None:  # noqa: B027
        """Called after load() completes (before close)."""
