# src/elspeth/plugins/base.py
"""Base classes for plugin implementations.

These provide common functionality and ensure proper interface compliance.
Plugins can subclass these for convenience, or implement protocols directly.

Phase 3 Integration:
- Lifecycle hooks (on_register, on_start, on_complete) are called by engine
- PluginContext is provided by engine with landscape/tracer/payload_store
"""

from abc import ABC, abstractmethod
from collections.abc import Iterator
from typing import Any

from elspeth.contracts import ArtifactDescriptor, Determinism, PluginSchema
from elspeth.plugins.context import PluginContext
from elspeth.plugins.results import (
    AcceptResult,
    GateResult,
    TransformResult,
)


class BaseTransform(ABC):
    """Base class for stateless row transforms.

    Subclass and implement process() to create a transform.

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

    # === Lifecycle Hooks (Phase 3) ===
    # These are intentionally empty - optional hooks for subclasses to override

    def on_register(self, ctx: PluginContext) -> None:  # noqa: B027
        """Called when plugin is registered with the engine.

        Override for one-time setup.
        """

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

    # === Lifecycle Hooks (Phase 3) ===

    def on_register(self, ctx: PluginContext) -> None:  # noqa: B027
        """Called when plugin is registered."""

    def on_start(self, ctx: PluginContext) -> None:  # noqa: B027
        """Called at start of run."""

    def on_complete(self, ctx: PluginContext) -> None:  # noqa: B027
        """Called at end of run."""


class BaseAggregation(ABC):
    """Base class for aggregation transforms (stateful batching).

    Subclass and implement accept(), should_trigger(), flush().

    Phase 3 Integration:
    - Engine creates Landscape batch on first accept()
    - Engine persists batch membership on every accept()
    - Engine manages batch state transitions

    Example:
        class StatsAggregation(BaseAggregation):
            name = "stats"
            input_schema = InputSchema
            output_schema = StatsSchema

            def __init__(self, config):
                super().__init__(config)
                self._values = []

            def accept(self, row, ctx) -> AcceptResult:
                self._values.append(row["value"])
                return AcceptResult(
                    accepted=True,
                    trigger=len(self._values) >= 100,
                )

            def flush(self, ctx) -> list[dict]:
                result = {"mean": statistics.mean(self._values)}
                self._values = []
                return [result]
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
    def accept(
        self,
        row: dict[str, Any],
        ctx: PluginContext,
    ) -> AcceptResult:
        """Accept a row into the batch."""
        ...

    @abstractmethod
    def should_trigger(self) -> bool:
        """Check if batch should flush."""
        ...

    @abstractmethod
    def flush(self, ctx: PluginContext) -> list[dict[str, Any]]:
        """Process batch and return results."""
        ...

    def reset(self) -> None:  # noqa: B027
        """Reset internal state.

        Override if you have state beyond what __init__ sets up.
        """

    # === Lifecycle Hooks (Phase 3) ===

    def on_register(self, ctx: PluginContext) -> None:  # noqa: B027
        """Called when plugin is registered."""

    def on_start(self, ctx: PluginContext) -> None:  # noqa: B027
        """Called at start of run."""

    def on_complete(self, ctx: PluginContext) -> None:  # noqa: B027
        """Called at end of run."""


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

    def on_register(self, ctx: PluginContext) -> None:  # noqa: B027
        """Called when plugin is registered."""

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

            def load(self, ctx: PluginContext) -> Iterator[dict]:
                with open(self.config["path"]) as f:
                    reader = csv.DictReader(f)
                    yield from reader

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
    def load(self, ctx: PluginContext) -> Iterator[dict[str, Any]]:
        """Load and yield rows from the source.

        Args:
            ctx: Plugin context

        Yields:
            Row dicts matching output_schema
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
