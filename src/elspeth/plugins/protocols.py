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
