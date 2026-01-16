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

from collections.abc import Iterator
from enum import Enum
from typing import TYPE_CHECKING, Any, Protocol, runtime_checkable

from elspeth.contracts import Determinism

if TYPE_CHECKING:
    from elspeth.contracts import PluginSchema
    from elspeth.plugins.context import PluginContext
    from elspeth.plugins.results import AcceptResult, GateResult, TransformResult


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
    node_id: str | None  # Set by orchestrator after registration

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

    def on_complete(self, ctx: "PluginContext") -> None:
        """Called after load() completes. Override for cleanup before close()."""
        ...


@runtime_checkable
class TransformProtocol(Protocol):
    """Protocol for stateless row transforms.

    Transforms process one row and emit one row (possibly modified).
    They are stateless between rows.

    Lifecycle:
        - __init__(config): Called once at pipeline construction
        - process(row, ctx): Called for each row
        - close(): Called at pipeline completion for cleanup

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
    node_id: str | None  # Set by orchestrator after registration

    # Metadata for Phase 3 audit/reproducibility
    determinism: Determinism
    plugin_version: str

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

    def close(self) -> None:
        """Clean up resources after pipeline completion.

        Called once after all rows have been processed. Use for closing
        connections, flushing buffers, or releasing external resources.
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

    Lifecycle:
        - __init__(config): Called once at pipeline construction
        - evaluate(row, ctx): Called for each row
        - close(): Called at pipeline completion for cleanup

    Example:
        class SafetyGate:
            name = "safety"
            input_schema = InputSchema
            output_schema = OutputSchema

            def evaluate(self, row: dict, ctx: PluginContext) -> GateResult:
                # Direct field access - schema guarantees field exists
                if row["suspicious"]:
                    return GateResult(
                        row=row,
                        action=RoutingAction.route("review"),  # Resolved via routes config
                    )
                return GateResult(row=row, action=RoutingAction.route("normal"))
    """

    name: str
    input_schema: type["PluginSchema"]
    output_schema: type["PluginSchema"]
    node_id: str | None  # Set by orchestrator after registration

    # Metadata for Phase 3 audit/reproducibility
    determinism: Determinism
    plugin_version: str

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

    def close(self) -> None:
        """Clean up resources after pipeline completion.

        Called once after all rows have been processed. Use for closing
        connections, flushing buffers, or releasing external resources.
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


class CoalescePolicy(Enum):
    """How coalesce handles partial arrivals."""

    REQUIRE_ALL = "require_all"  # Wait for all branches; any failure fails
    QUORUM = "quorum"  # Merge if >= n branches succeed
    BEST_EFFORT = "best_effort"  # Merge whatever arrives by timeout


@runtime_checkable
class AggregationProtocol(Protocol):
    """Protocol for aggregation transforms (stateful batching).

    Aggregations accumulate rows until a trigger condition, then flush.

    Phase 3 Integration:
    - Engine creates Landscape batch on first accept()
    - Engine persists batch membership on every accept()
    - Engine transitions batch status on flush()

    Example:
        class StatsAggregation:
            name = "stats"
            input_schema = InputSchema
            output_schema = StatsSchema

            def accept(self, row, ctx) -> AcceptResult:
                self._values.append(row["value"])
                return AcceptResult(
                    accepted=True,
                    trigger=len(self._values) >= self.batch_size,
                )

            def flush(self, ctx) -> list[dict]:
                result = {"mean": statistics.mean(self._values)}
                self._values = []
                return [result]
    """

    name: str
    input_schema: type["PluginSchema"]
    output_schema: type["PluginSchema"]
    node_id: str | None  # Set by orchestrator after registration

    # Metadata for Phase 3 audit/reproducibility
    determinism: Determinism
    plugin_version: str

    def __init__(self, config: dict[str, Any]) -> None:
        """Initialize with configuration."""
        ...

    def accept(
        self,
        row: dict[str, Any],
        ctx: "PluginContext",
    ) -> "AcceptResult":
        """Accept a row into the batch.

        Called for each row. Implementation should:
        1. Store the row in internal buffer
        2. Return trigger=True when batch should flush

        Note: In Phase 3, the engine wraps this to manage Landscape batches.

        Args:
            row: Input row
            ctx: Plugin context

        Returns:
            AcceptResult indicating acceptance and trigger state
        """
        ...

    def should_trigger(self) -> bool:
        """Check if batch should flush now.

        Called by engine to check trigger condition outside of accept().
        """
        ...

    def flush(self, ctx: "PluginContext") -> list[dict[str, Any]]:
        """Process accumulated rows and return results.

        Called when trigger condition is met or at end of source.
        Should reset internal state after processing.

        Note: In Phase 3, the engine wraps this to:
        1. Transition batch to "executing"
        2. Record results
        3. Transition batch to "completed" or "failed"

        Args:
            ctx: Plugin context

        Returns:
            List of output rows (usually one aggregate result)
        """
        ...

    def reset(self) -> None:
        """Reset internal state.

        Called by engine on error recovery.
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
class CoalesceProtocol(Protocol):
    """Protocol for coalesce transforms (merge parallel paths).

    Coalesce merges results from parallel branches back into a single path.

    Configuration:
    - policy: How to handle partial arrivals
    - quorum_threshold: Minimum branches for QUORUM policy (None otherwise)
    - inputs: Which branches to expect
    - key: How to correlate branch outputs (Phase 3 engine concern)

    Example:
        class SimpleCoalesce:
            name = "merge"
            policy = CoalescePolicy.REQUIRE_ALL
            quorum_threshold = None  # Only used for QUORUM policy

            def merge(self, branch_outputs, ctx) -> dict:
                merged = {}
                for branch_name, output in branch_outputs.items():
                    merged.update(output)
                return merged

        class QuorumCoalesce:
            name = "quorum_merge"
            policy = CoalescePolicy.QUORUM
            quorum_threshold = 2  # Proceed if >= 2 branches arrive
    """

    name: str
    policy: CoalescePolicy
    quorum_threshold: int | None  # Required if policy == QUORUM
    expected_branches: list[str]
    output_schema: type["PluginSchema"]
    node_id: str | None  # Set by orchestrator after registration

    # Metadata for Phase 3 audit/reproducibility
    determinism: Determinism
    plugin_version: str

    def __init__(self, config: dict[str, Any]) -> None:
        """Initialize with configuration."""
        ...

    def merge(
        self,
        branch_outputs: dict[str, dict[str, Any]],
        ctx: "PluginContext",
    ) -> dict[str, Any]:
        """Merge outputs from multiple branches.

        Args:
            branch_outputs: Map of branch_name -> output_row
            ctx: Plugin context

        Returns:
            Merged output row
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
class SinkProtocol(Protocol):
    """Protocol for sink plugins.

    Sinks output data to external destinations.
    There can be multiple sinks per run.

    Idempotency:
    - Sinks receive idempotency keys: {run_id}:{row_id}:{sink_name}
    - Sinks that cannot guarantee idempotency should set idempotent=False

    Example:
        class CSVSink:
            name = "csv"
            input_schema = RowSchema
            idempotent = False  # Appends are not idempotent

            def write(self, row: dict, ctx: PluginContext) -> None:
                self._writer.writerow(row)

            def flush(self) -> None:
                self._file.flush()
    """

    name: str
    input_schema: type["PluginSchema"]
    idempotent: bool  # Can this sink handle retries safely?
    node_id: str | None  # Set by orchestrator after registration

    # Metadata for Phase 3 audit/reproducibility
    determinism: Determinism
    plugin_version: str

    def __init__(self, config: dict[str, Any]) -> None:
        """Initialize with configuration."""
        ...

    def write(
        self,
        row: dict[str, Any],
        ctx: "PluginContext",
    ) -> None:
        """Write a row to the sink.

        Args:
            row: Row data to write
            ctx: Plugin context
        """
        ...

    def flush(self) -> None:
        """Flush any buffered data.

        Called periodically and at end of run.
        """
        ...

    def close(self) -> None:
        """Close the sink and release resources.

        Called at end of run or on error.
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
        """Called at end of run (before close)."""
        ...
