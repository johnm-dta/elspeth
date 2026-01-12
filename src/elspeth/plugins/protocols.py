# src/elspeth/plugins/protocols.py
"""Plugin protocols defining the contracts for each plugin type.

These protocols define what methods plugins must implement.
They're used for type checking, not runtime enforcement (that's pluggy's job).

Plugin Types:
- Source: Loads data into the system (one per run)
- Transform: Processes rows (stateless)
- Gate: Routes rows to destinations (stateless)
- Aggregation: Accumulates rows, flushes batches (stateful)
- Coalesce: Merges parallel paths (stateful)
- Sink: Outputs data (one or more per run)
"""

from typing import TYPE_CHECKING, Any, Iterator, Protocol, runtime_checkable

if TYPE_CHECKING:
    from elspeth.plugins.context import PluginContext
    from elspeth.plugins.results import AcceptResult, GateResult, TransformResult
    from elspeth.plugins.schemas import PluginSchema


@runtime_checkable
class SourceProtocol(Protocol):
    """Protocol for source plugins.

    Sources load data into the system. There is exactly one source per run.

    Lifecycle:
    1. __init__(config) - Plugin instantiation
    2. on_start(ctx) - Called before loading (optional)
    3. load(ctx) - Yields rows
    4. close() - Cleanup

    Example:
        class CSVSource:
            name = "csv"
            output_schema = RowSchema

            def load(self, ctx: PluginContext) -> Iterator[dict]:
                with open(self.path) as f:
                    reader = csv.DictReader(f)
                    for row in reader:
                        yield row
    """

    name: str
    output_schema: type["PluginSchema"]

    def __init__(self, config: dict[str, Any]) -> None:
        """Initialize with configuration."""
        ...

    def load(self, ctx: "PluginContext") -> Iterator[dict[str, Any]]:
        """Load and yield rows from the source.

        Args:
            ctx: Plugin context with run metadata

        Yields:
            Row dicts matching output_schema
        """
        ...

    def close(self) -> None:
        """Clean up resources.

        Called after all rows are loaded or on error.
        """
        ...

    # === Optional Lifecycle Hooks ===

    def on_start(self, ctx: "PluginContext") -> None:
        """Called before load(). Override for setup."""
        ...


@runtime_checkable
class TransformProtocol(Protocol):
    """Protocol for stateless row transforms.

    Transforms process one row and emit one row (possibly modified).
    They are stateless between rows.

    Example:
        class EnrichTransform:
            name = "enrich"
            input_schema = InputSchema
            output_schema = OutputSchema

            def process(self, row: dict, ctx: PluginContext) -> TransformResult:
                enriched = {**row, "timestamp": datetime.now().isoformat()}
                return TransformResult.success(enriched)
    """

    name: str
    input_schema: type["PluginSchema"]
    output_schema: type["PluginSchema"]

    def __init__(self, config: dict[str, Any]) -> None:
        """Initialize with configuration."""
        ...

    def process(
        self,
        row: dict[str, Any],
        ctx: "PluginContext",
    ) -> "TransformResult":
        """Process a single row.

        Args:
            row: Input row matching input_schema
            ctx: Plugin context

        Returns:
            TransformResult with processed row or error
        """
        ...

    # === Optional Lifecycle Hooks ===

    def on_register(self, ctx: "PluginContext") -> None:
        """Called when plugin is registered."""
        ...

    def on_start(self, ctx: "PluginContext") -> None:
        """Called at start of run."""
        ...

    def on_complete(self, ctx: "PluginContext") -> None:
        """Called at end of run."""
        ...


@runtime_checkable
class GateProtocol(Protocol):
    """Protocol for gate transforms (routing decisions).

    Gates evaluate rows and decide routing. They can:
    - Continue to next transform
    - Route to a named sink
    - Fork to multiple parallel paths

    Example:
        class SafetyGate:
            name = "safety"
            input_schema = InputSchema
            output_schema = OutputSchema

            def evaluate(self, row: dict, ctx: PluginContext) -> GateResult:
                if row.get("suspicious"):
                    return GateResult(
                        row=row,
                        action=RoutingAction.route_to_sink("review_queue"),
                    )
                return GateResult(row=row, action=RoutingAction.continue_())
    """

    name: str
    input_schema: type["PluginSchema"]
    output_schema: type["PluginSchema"]

    def __init__(self, config: dict[str, Any]) -> None:
        """Initialize with configuration."""
        ...

    def evaluate(
        self,
        row: dict[str, Any],
        ctx: "PluginContext",
    ) -> "GateResult":
        """Evaluate a row and decide routing.

        Args:
            row: Input row
            ctx: Plugin context

        Returns:
            GateResult with (possibly modified) row and routing action
        """
        ...

    # === Optional Lifecycle Hooks ===

    def on_register(self, ctx: "PluginContext") -> None:
        """Called when plugin is registered."""
        ...

    def on_start(self, ctx: "PluginContext") -> None:
        """Called at start of run."""
        ...

    def on_complete(self, ctx: "PluginContext") -> None:
        """Called at end of run."""
        ...
