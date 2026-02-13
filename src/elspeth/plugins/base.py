# src/elspeth/plugins/base.py
"""Base classes for plugin implementations.

These provide common functionality and ensure proper interface compliance.
Plugins MUST subclass these base classes (BaseSource, BaseTransform, BaseSink).
Gate routing is handled by config-driven GateSettings + ExpressionParser, not plugin classes.

Why base class inheritance is required:
- Plugin discovery uses issubclass() checks against base classes
- Python's Protocol with non-method members (name, determinism, etc.) cannot
  support issubclass() - only isinstance() on already-instantiated objects
- Base classes enforce self-consistency via __init_subclass__ hooks
- Per CLAUDE.md "Plugin Ownership", all plugins are system code, not user extensions

The protocol definitions (SourceProtocol, TransformProtocol, SinkProtocol) exist
for type-checking purposes only - they define the interface contract but cannot
be used for runtime discovery.

Lifecycle Contract (all hooks called on main thread by orchestrator):
    on_start(ctx) -> [process/load/write] -> on_complete(ctx) -> close()

- on_start: Per-run initialization (acquire resources, capture context).
  If on_start raises, neither on_complete nor close is called.
- on_complete: Processing finished (success or error). Receives PluginContext
  for landscape/telemetry interaction. Called even on pipeline crash.
- close: Pure resource teardown (no PluginContext). Called even on pipeline
  crash. Each plugin's cleanup is individually protected.
- Call order across plugin types (normal run):
  source.on_start -> transforms.on_start -> sinks.on_start -> [processing]
  -> transforms.on_complete -> sinks.on_complete -> source.on_complete
  -> source.close -> transforms.close -> sinks.close
- Resume runs skip source lifecycle entirely (NullSource is used).
"""

from abc import ABC, abstractmethod
from collections.abc import Iterator
from typing import TYPE_CHECKING, Any

from elspeth.contracts import ArtifactDescriptor, Determinism, PluginSchema, SourceRow
from elspeth.contracts.schema_contract import PipelineRow

if TYPE_CHECKING:
    from elspeth.contracts.schema_contract import SchemaContract
    from elspeth.contracts.sink import OutputValidationResult
from elspeth.contracts.plugin_context import PluginContext
from elspeth.plugins.results import (
    TransformResult,
)


class BaseTransform(ABC):
    """Base class for all row transforms.

    Execution Models
    ----------------
    Transforms use one of three execution models depending on concurrency needs.
    The TransformExecutor in engine/executors/transform.py dispatches automatically
    based on isinstance checks (BatchTransformMixin) and the is_batch_aware flag.

    **1. Synchronous (process) -- standard row-by-row processing**

        The engine calls process() once per row and receives a TransformResult
        synchronously. Use this for CPU-bound, fast, or deterministic transforms
        (field mapping, truncation, validation, etc.).

        class MyTransform(BaseTransform):
            name = "my_transform"
            input_schema = InputSchema
            output_schema = OutputSchema

            def process(self, row: PipelineRow, ctx: PluginContext) -> TransformResult:
                return TransformResult.success(
                    {**row.to_dict(), "new_field": "value"},
                    success_reason={"action": "processed"},
                )

    **2. Streaming (accept) -- row-level pipelining via BatchTransformMixin**

        The engine calls accept() per row. Processing happens asynchronously in a
        worker pool; results are emitted in FIFO order through an OutputPort. The
        engine blocks until each row's result arrives (sequential across rows,
        concurrent within each row). Use this for I/O-bound transforms that benefit
        from concurrency (LLM calls, HTTP APIs, multi-query evaluation).

        Requires inheriting both BaseTransform and BatchTransformMixin:

        class MyLLMTransform(BaseTransform, BatchTransformMixin):
            name = "my_llm"

            def accept(self, row: PipelineRow, ctx: PluginContext) -> None:
                self.accept_row(row, ctx, self._do_work)

            def connect_output(self, output: OutputPort, max_pending: int = 30) -> None:
                self.init_batch_processing(max_pending=max_pending, output=output)

            def _do_work(self, row: PipelineRow, ctx: PluginContext) -> TransformResult:
                # Runs in worker thread
                return TransformResult.success(...)

        Streaming transforms override process() to raise NotImplementedError,
        directing callers to use accept(). The TransformExecutor detects the mixin
        via isinstance(transform, BatchTransformMixin) and routes to accept()
        automatically -- plugin authors never need to worry about dispatch.

    **3. Batch-aware (process with is_batch_aware=True) -- aggregation batches**

        The engine buffers rows until an aggregation trigger fires, then calls
        process() with list[PipelineRow]. Use this for transforms inside
        aggregation nodes (batch LLM calls, statistical aggregations).

        class MyBatchTransform(BaseTransform):
            name = "my_batch"
            is_batch_aware = True

            def process(self, row, ctx):  # row is list[PipelineRow] in batch mode
                if isinstance(row, list):
                    return self._process_batch(row, ctx)
                return self._process_single(row, ctx)

    When to Use Which
    -----------------
    - process() alone: Simple, fast transforms (field mapping, filtering, etc.)
    - accept() + BatchTransformMixin: I/O-bound per-row work needing concurrency
      (LLM API calls, HTTP requests, multi-query evaluation)
    - process() + is_batch_aware: Aggregation-stage transforms that receive
      pre-buffered batches from the engine's trigger system
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

    # Token creation flag for deaggregation transforms
    # When True AND process() returns success_multi(), the processor creates
    # new token_ids for each output row with parent linkage to input token.
    # When False AND success_multi() is returned, the processor expects
    # passthrough mode (same number of outputs as inputs, preserve token_ids).
    # Default: False (most transforms don't create new tokens)
    creates_tokens: bool = False

    # Schema evolution flag (P1-2026-02-05)
    # When True, transform adds fields during execution and evolved contract
    # should be recorded to audit trail (input fields + added fields).
    # When False (default), transform does not add fields to schema.
    transforms_adds_fields: bool = False

    # Error routing configuration (WP-11.99b)
    # Transforms extending TransformDataConfig override this from config.
    # Always non-None at runtime (TransformSettings requires on_error).
    # Base class default is None because injection happens post-construction
    # via cli_helpers bridge (set from TransformSettings.on_error).
    on_error: str | None = None

    # Success routing configuration
    # Terminal transforms set this to the output sink name.
    # Always non-None at runtime (TransformSettings requires on_success).
    # Base class default is None because injection happens post-construction
    # via cli_helpers bridge (set from TransformSettings.on_success).
    on_success: str | None = None

    def __init__(self, config: dict[str, Any]) -> None:
        """Initialize with configuration.

        Args:
            config: Plugin configuration
        """
        self.config = config

    def process(
        self,
        row: PipelineRow,
        ctx: PluginContext,
    ) -> TransformResult:
        """Process a single row.

        Single-row transforms must override this method.
        Batch-aware transforms (is_batch_aware=True) should override with
        signature: process(self, rows: list[PipelineRow], ctx) -> TransformResult

        Args:
            row: Input row as PipelineRow (immutable, supports dual-name access)
            ctx: Plugin context

        Returns:
            TransformResult with processed row dict or error
        """
        raise NotImplementedError(
            f"{self.__class__.__name__} must implement process(). "
            f"Single-row transforms: process(row: PipelineRow, ctx) -> TransformResult. "
            f"Batch-aware transforms: process(rows: list[PipelineRow], ctx) -> TransformResult."
        )

    def close(self) -> None:  # noqa: B027 - optional override, not abstract
        """Release resources (connections, file handles, thread pools).

        Called once per run after on_complete(), inside a finally block.
        No PluginContext is available -- this is pure resource teardown.

        Guaranteed to be called if on_start() succeeded, even when the
        pipeline crashes mid-processing. NOT called if on_start() itself
        raises. Each plugin's close() is individually protected: one
        plugin's failure does not prevent others from closing.

        Called on the main thread.
        """
        pass

    # === Lifecycle Hooks ===
    # These are intentionally empty - optional hooks for subclasses to override.
    #
    # Call ordering (orchestrator, main thread):
    #   1. on_start(ctx)    -- before any rows are processed
    #   2. process(row, ctx) -- per row (or per batch)
    #   3. on_complete(ctx)  -- after all rows, or after pipeline error
    #   4. close()           -- resource teardown (always after on_complete)
    #
    # on_complete() and close() run inside a finally block, so they execute
    # even when the pipeline crashes. However, if on_start() raises, neither
    # on_complete() nor close() is called for ANY plugin.
    #
    # Resume path: on_start/on_complete/close are called normally for
    # transforms during resume runs.

    def on_start(self, ctx: PluginContext) -> None:  # noqa: B027 - optional hook
        """Called once before any rows are processed.

        Override for per-run initialization: capturing the recorder,
        acquiring rate limiters, initializing tracing, etc.

        Called on the main thread. If this raises, the pipeline aborts
        and neither on_complete() nor close() will be called.
        """
        pass

    def on_complete(self, ctx: PluginContext) -> None:  # noqa: B027 - optional hook
        """Called after all rows are processed (or after pipeline error).

        Override for recording final metrics, flushing application-level
        buffers, or updating audit state. Always called before close().

        Called on the main thread. Receives PluginContext so it can
        interact with the landscape and telemetry. Individually protected:
        if this raises, other plugins still get their on_complete/close calls.
        """
        pass


# NOTE: BaseAggregation was DELETED in aggregation structural cleanup.
# Aggregation is now fully structural:
# - Engine buffers rows internally
# - Engine evaluates triggers (WP-06)
# - Engine calls batch-aware Transform.process(rows: list[dict])
# Use is_batch_aware=True on BaseTransform for batch processing.


class BaseSink(ABC):
    """Base class for sink plugins.

    Subclass and implement write(), flush(), close().

    Lifecycle (called by the orchestrator on the main thread):

        1. on_start(ctx)     -- per-run initialization (before any writes)
        2. write(rows, ctx)  -- called in batches as rows reach this sink
        3. flush()           -- called before checkpoints to guarantee durability
        4. on_complete(ctx)  -- all rows written (or pipeline errored)
        5. close()           -- release resources (file handles, connections)

    Guarantees:
        - on_start() is called once before any write() call.
        - on_complete() and close() run inside a finally block, so they
          execute even when the pipeline crashes mid-processing. However,
          if on_start() raises, neither on_complete() nor close() is called.
        - on_complete() is called before close(). Both are called regardless
          of whether processing succeeded or failed.
        - Each plugin's on_complete()/close() is individually protected: one
          plugin's cleanup failure does not prevent other plugins from
          cleaning up.

    on_complete vs close:
        - on_complete(ctx): "Processing is done." Use for finalizing output
          format (e.g., writing JSON array closing bracket), recording metrics,
          or updating audit state. Receives PluginContext.
        - close(): "Release all resources." Use for closing file handles or
          network connections. No PluginContext -- pure resource teardown.

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

    # Resume capability (Phase 5 - Checkpoint/Resume)
    # Default: sinks don't support resume. Override in subclasses that can append.
    supports_resume: bool = False

    def configure_for_resume(self) -> None:
        """Configure sink for resume mode (append instead of truncate).

        Called by engine when resuming a run. Override in sinks that support
        resume to switch from truncate mode to append mode.

        Default implementation raises NotImplementedError. Subclasses that
        set supports_resume=True MUST override this method.

        Raises:
            NotImplementedError: If sink cannot be resumed.
        """
        raise NotImplementedError(
            f"{self.__class__.__name__} does not support resume. "
            f"To make this sink resumable, set supports_resume=True and "
            f"implement configure_for_resume()."
        )

    def validate_output_target(self) -> "OutputValidationResult":
        """Validate existing output target matches configured schema.

        Called by engine/CLI before write operations in append/resume mode.
        Default returns valid=True (dynamic schema or no existing target).

        Sinks that set supports_resume=True SHOULD override this to validate
        that the existing output target (file/table) matches the schema.

        Returns:
            OutputValidationResult indicating compatibility.
        """
        from elspeth.contracts.sink import OutputValidationResult

        return OutputValidationResult.success()

    def set_resume_field_resolution(self, resolution_mapping: dict[str, str]) -> None:
        """Set field resolution mapping for resume validation.

        Default is a no-op. Only sinks that support restore_source_headers
        (CSVSink, JSONSink) override this to use the mapping for validation.

        Args:
            resolution_mapping: Dict mapping original header name -> normalized field name.
        """
        # Intentional no-op - most sinks don't use restore_source_headers
        _ = resolution_mapping  # Explicitly consume the argument

    # Output contract for schema-aware sinks (Phase 3)
    _output_contract: "SchemaContract | None" = None

    def __init__(self, config: dict[str, Any]) -> None:
        """Initialize with configuration.

        Args:
            config: Plugin configuration
        """
        self.config = config
        self._output_contract = None

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
        """Release resources (file handles, connections).

        Called once per run after on_complete(), inside a finally block.
        Guaranteed to be called if on_start() succeeded, even on pipeline
        crash. NOT called if on_start() itself raises. Each plugin's
        close() is individually protected. Called on the main thread.
        """
        ...

    # === Output Contract Support (Phase 3) ===

    def get_output_contract(self) -> "SchemaContract | None":
        """Get the current output contract.

        Returns:
            SchemaContract if set, None otherwise
        """
        return self._output_contract

    def set_output_contract(self, contract: "SchemaContract") -> None:
        """Set or update the output contract.

        Used for schema-aware sinks that need field metadata (e.g., for
        restoring original header names in CSV output).

        Args:
            contract: The schema contract to use for output operations
        """
        self._output_contract = contract

    # === Lifecycle Hooks ===
    # Call ordering: on_start -> write/flush -> on_complete -> close
    # See class docstring for full lifecycle contract and guarantees.

    def on_start(self, ctx: PluginContext) -> None:  # noqa: B027 - optional hook
        """Called once before any write() call.

        Override for per-run initialization. Called on the main thread.
        If this raises, the pipeline aborts and neither on_complete()
        nor close() will be called.
        """
        pass

    def on_complete(self, ctx: PluginContext) -> None:  # noqa: B027 - optional hook
        """Called after all rows are written (or after pipeline error), before close().

        Override for finalizing output format, recording metrics, or
        updating audit state. Called on the main thread. Individually
        protected: if this raises, other plugins still get their calls.
        """
        pass


class BaseSource(ABC):
    """Base class for source plugins.

    Subclass and implement load() and close().

    Lifecycle (called by the orchestrator on the main thread):

        1. on_start(ctx)  -- per-run initialization (before load)
        2. load(ctx)      -- yields SourceRow instances
        3. on_complete(ctx) -- source exhausted (or pipeline errored)
        4. close()        -- release resources (file handles, connections)

    Guarantees:
        - on_start() is called once before load().
        - on_complete() and close() run inside a finally block, so they
          execute even when the pipeline crashes mid-processing. However,
          if on_start() raises, neither on_complete() nor close() is called.
        - on_complete() is called before close(). Both are called regardless
          of whether processing succeeded or failed.
        - Each plugin's on_complete()/close() is individually protected.

    Resume path: Source lifecycle hooks (on_start, on_complete, close) are
    skipped during resume runs because NullSource is used and row data comes
    from stored payloads, not from the original source.

    on_complete vs close:
        - on_complete(ctx): "Loading is done." Receives PluginContext.
        - close(): "Release all resources." No PluginContext -- pure teardown.

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

    # Sink name for quarantined rows, or "discard" to drop invalid rows
    # All sources must set this - config-based sources get it from SourceDataConfig
    _on_validation_failure: str

    # Success routing: sink name for rows that pass source validation
    # All sources must set this - config-based sources get it from SourceDataConfig
    on_success: str

    # Schema contract for row validation (Phase 2)
    _schema_contract: "SchemaContract | None" = None

    def __init__(self, config: dict[str, Any]) -> None:
        """Initialize with configuration.

        Args:
            config: Plugin configuration
        """
        self.config = config
        self._schema_contract = None

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
        """Release resources (file handles, connections).

        Called once per run after on_complete(), inside a finally block.
        Guaranteed to be called if on_start() succeeded, even on pipeline
        crash. NOT called if on_start() itself raises. Each plugin's
        close() is individually protected. Called on the main thread.

        Skipped during resume runs (source is not opened).
        """
        ...

    # === Schema Contract Support (Phase 2) ===

    def get_schema_contract(self) -> "SchemaContract | None":
        """Get the current schema contract.

        Returns:
            SchemaContract if set, None otherwise
        """
        return self._schema_contract

    def set_schema_contract(self, contract: "SchemaContract") -> None:
        """Set or update the schema contract.

        Called during initialization for explicit schemas (FIXED/FLEXIBLE),
        or after first-row inference for OBSERVED mode.

        Args:
            contract: The schema contract to use for validation
        """
        self._schema_contract = contract

    # === Lifecycle Hooks ===
    # Call ordering: on_start -> load -> on_complete -> close
    # See class docstring for full lifecycle contract and guarantees.
    # Skipped entirely during resume runs (NullSource is used instead).

    def on_start(self, ctx: PluginContext) -> None:  # noqa: B027 - optional hook
        """Called once before load().

        Override for per-run initialization. Called on the main thread.
        If this raises, the pipeline aborts and neither on_complete()
        nor close() will be called.

        Skipped during resume runs.
        """
        pass

    def on_complete(self, ctx: PluginContext) -> None:  # noqa: B027 - optional hook
        """Called after load() completes (or after pipeline error), before close().

        Override for recording final metrics or updating audit state.
        Called on the main thread. Individually protected: if this raises,
        other plugins still get their on_complete/close calls.

        Skipped during resume runs.
        """
        pass

    # === Audit Trail Metadata ===

    def get_field_resolution(self) -> tuple[dict[str, str], str | None] | None:
        """Return field resolution mapping computed during load().

        Sources that perform field normalization (e.g., CSVSource with normalize_fields)
        should override this to return the mapping from original header names to final
        field names. This enables audit trail to recover original headers.

        Must be called AFTER load() has been invoked (resolution is computed lazily
        when file headers are read).

        Returns:
            Tuple of (resolution_mapping, normalization_version) if field resolution
            was performed, or None if no normalization occurred. The resolution_mapping
            is a dict mapping original header name → final field name.
        """
        return None  # Default: no field resolution metadata
