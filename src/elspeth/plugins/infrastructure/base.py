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
- on_complete: Processing finished (success or error). Receives LifecycleContext
  for landscape/telemetry interaction. Called even on pipeline crash.
- close: Pure resource teardown (no context). Called even on pipeline
  crash. Each plugin's cleanup is individually protected.
- Call order across plugin types (normal run):
  source.on_start -> transforms.on_start -> sinks.on_start -> [processing]
  -> transforms.on_complete -> sinks.on_complete -> source.on_complete
  -> source.close -> transforms.close -> sinks.close
- Resume runs skip source lifecycle entirely (NullSource is used).
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Iterator, Mapping
from typing import TYPE_CHECKING, Any, ClassVar

from elspeth.contracts import Determinism, PluginSchema, SourceRow
from elspeth.contracts.diversion import RowDiversion, SinkWriteResult
from elspeth.contracts.errors import FrameworkBugError
from elspeth.contracts.schema_contract import FieldContract, PipelineRow, SchemaContract

if TYPE_CHECKING:
    from elspeth.contracts.contexts import LifecycleContext, SinkContext, SourceContext, TransformContext
    from elspeth.contracts.header_modes import HeaderMode
    from elspeth.contracts.plugin_assistance import PluginAssistance
    from elspeth.contracts.plugin_semantics import (
        InputSemanticRequirements,
        OutputSemanticDeclaration,
    )
    from elspeth.contracts.schema import SchemaConfig
    from elspeth.contracts.schema_contract import SchemaContract
    from elspeth.contracts.sink import OutputValidationResult
    from elspeth.plugins.infrastructure.config_base import PluginConfig, TransformDataConfig
from elspeth.plugins.infrastructure.results import (
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

            def process(self, row: PipelineRow, ctx: TransformContext) -> TransformResult:
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

            def accept(self, row: PipelineRow, ctx: TransformContext) -> None:
                self.accept_row(row, ctx, self._do_work)

            def connect_output(self, output: OutputPort, max_pending: int = 30) -> None:
                self.init_batch_processing(max_pending=max_pending, output=output)

            def _do_work(self, row: PipelineRow, ctx: TransformContext) -> TransformResult:
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

    # Audit metadata
    determinism: Determinism = Determinism.DETERMINISTIC
    plugin_version: str = "0.0.0"
    source_file_hash: str | None = None

    # Config model — each subclass sets this to its Pydantic config class.
    # get_config_model() is the public API; override it for dynamic dispatch
    # (e.g. provider-based LLM config selection).
    config_model: ClassVar[type[PluginConfig] | None] = None

    @classmethod
    def get_config_model(cls, config: dict[str, Any] | None = None) -> type[PluginConfig] | None:
        """Return the Pydantic config model for this plugin type.

        Override in subclasses that need dynamic dispatch (e.g. LLMTransform
        selects a provider-specific model based on config["provider"]).
        """
        return cls.config_model

    @classmethod
    def get_config_schema(cls) -> dict[str, Any]:
        """Default ``get_config_schema`` — renders a single Pydantic model.

        See :meth:`~elspeth.contracts.plugin_protocols.TransformProtocol.get_config_schema`
        for the canonical contract, including the MUST-override rule for
        plugins whose effective configuration is a discriminated union.
        """
        if cls.config_model is None:
            return {}
        schema: dict[str, Any] = cls.config_model.model_json_schema()
        return schema

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

    # Pass-through contract flag (ADR-007).
    # True iff process() UNCONDITIONALLY emits rows containing every field
    # present on the input row plus declared_output_fields, regardless of
    # row content or runtime state. False if the transform may drop, rename,
    # filter, or conditionally omit input fields. Conditional drops based on
    # row content are forbidden under this annotation — they would pass static
    # and Hypothesis tests and crash production via PassThroughContractViolation.
    # Annotation is verified at runtime by TransformExecutor's pass-through
    # cross-check; mis-annotation raises PassThroughContractViolation (TIER_1).
    passes_through_input: bool = False

    # Empty-emission governance declaration (ADR-012).
    # True means the transform may intentionally emit zero rows on success.
    # False means empty success output is governance-significant for
    # passes_through_input=True transforms and is checked by the
    # can_drop_rows declaration contract.
    can_drop_rows: bool = False

    # Field collision enforcement (centralized in TransformExecutor).
    # Transforms that add fields to the output row declare WHAT fields they add
    # at init time. The executor checks these against input keys BEFORE running
    # the transform. Empty frozenset = no fields added = no check needed.
    declared_output_fields: frozenset[str] = frozenset()

    # Input-field declaration for ADR-013.
    # Normalized from TransformDataConfig.required_input_fields at construction
    # time via _initialize_declared_input_fields(). Empty frozenset means the
    # transform declares no pre-emission required-input contract.
    declared_input_fields: frozenset[str] = frozenset()

    # Error routing configuration.
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

    # DAG contract for output field validation (centralized in DAG builder).
    # Transforms that add fields must set this via _build_output_schema_config()
    # so the DAG builder can validate downstream required_input_fields.
    # None = no output contract provided (acceptable for shape-preserving transforms).
    _output_schema_config: SchemaConfig | None

    def __init__(self, config: dict[str, Any]) -> None:
        """Initialize with configuration.

        Args:
            config: Plugin configuration
        """
        self.config = config
        # Per-instance, not per-class — class-level defaults would be shared across instances.
        self._on_start_called: bool = False
        self.declared_input_fields = frozenset()
        self._output_schema_config: SchemaConfig | None = None

    def _initialize_declared_input_fields(self, validated_config: TransformDataConfig) -> None:
        """Populate ADR-013's runtime input-field declaration from config.

        Call from the transform's authoritative config-validation path —
        immediately after ``<Config>.from_dict(...)`` succeeds. This preserves
        existing per-plugin validation/error semantics while centralizing the
        runtime normalization and batch-aware fail-closed guard.
        """
        declared_input_fields = validated_config.declared_input_fields
        if declared_input_fields and self.is_batch_aware:
            raise FrameworkBugError(
                f"Transform {self.name!r} declares declared_input_fields "
                f"{sorted(declared_input_fields)!r} but is batch-aware. No "
                f"batch-pre-execution dispatch site exists; ADR-013 scopes "
                f"DeclaredRequiredFieldsContract to non-batch transforms until "
                f"an ADR-010 amendment lands."
            )
        self.declared_input_fields = declared_input_fields

    def effective_static_contract(self) -> frozenset[str]:
        """Return the transform's public static output guarantee surface.

        Runtime declaration checks record this value in audit evidence. Missing
        output schema config is acceptable only for shape-preserving transforms
        that declare no added fields. A field-adding transform without a schema
        config would falsely state its static guarantees and must crash.
        """
        output_schema_config = self._output_schema_config
        if output_schema_config is None:
            if not self.declared_output_fields:
                return frozenset()
            raise FrameworkBugError(
                f"Cannot derive effective static contract for transform {self.name!r}: "
                "_output_schema_config is missing. Concrete transforms must "
                "initialize their output schema config during construction "
                "before the engine can run declaration-contract checks."
            )
        return output_schema_config.get_effective_guaranteed_fields()

    def _align_output_contract(self, contract: SchemaContract) -> SchemaContract:
        """Normalize emitted contract mode/lock state to declared output semantics.

        ADR-014 compares emitted ``PipelineRow.contract`` semantics to the
        transform's ``_output_schema_config`` declaration. Once a contract is
        attached to an emitted row it is expected to be locked, even for
        ``flexible``/``observed`` config modes whose pre-emission builders may
        begin unlocked.
        """
        output_schema_config = self._output_schema_config
        if output_schema_config is None:
            return contract

        from elspeth.contracts.schema_contract import SchemaContract
        from elspeth.contracts.schema_contract_factory import map_schema_mode

        expected_mode = map_schema_mode(output_schema_config.mode)
        if contract.mode == expected_mode and contract.locked:
            return contract

        return SchemaContract(
            mode=expected_mode,
            fields=contract.fields,
            locked=True,
        )

    def _apply_declared_output_field_contracts(self, contract: SchemaContract) -> SchemaContract:
        """Apply declared output field metadata to an emitted row contract.

        Contract propagation infers newly added fields from runtime values, which
        marks them as ``source="inferred"`` and ``required=False``. When a
        transform has an explicit ``_output_schema_config`` field declaration,
        ADR-014 expects emitted contracts to carry that declared metadata.
        """
        output_schema_config = self._output_schema_config
        if output_schema_config is None or output_schema_config.fields is None:
            return contract

        from elspeth.contracts.schema_contract_factory import create_contract_from_config

        declared_fields = {field.normalized_name: field for field in create_contract_from_config(output_schema_config).fields}
        fields: list[FieldContract] = []
        changed = False
        for field in contract.fields:
            if field.normalized_name in declared_fields:
                fields.append(declared_fields[field.normalized_name])
                changed = True
            else:
                fields.append(field)

        if not changed:
            return contract

        return SchemaContract(
            mode=contract.mode,
            fields=tuple(fields),
            locked=contract.locked,
        )

    def _align_output_row_contract(self, row: PipelineRow) -> PipelineRow:
        """Return ``row`` with contract semantics aligned to this transform."""
        if row.contract is None:
            raise FrameworkBugError(f"Transform {self.name!r} emitted PipelineRow with no contract. Framework invariant violated.")

        aligned_contract = self._align_output_contract(row.contract)
        if aligned_contract is row.contract:
            return row
        return PipelineRow(row.to_dict(), aligned_contract)

    @classmethod
    def probe_config(cls) -> dict[str, Any]:
        """Return a minimal config dict sufficient to instantiate this transform
        for invariant probing (ADR-009 §Clause 4).

        Transforms annotated with ``passes_through_input=True`` MUST override
        this method. The default raises ``NotImplementedError`` so that a
        transform that later gains the annotation is caught immediately by
        the skip-rate budget test, not silently excluded from governance.

        The returned dict is passed directly to ``cls(...)`` using the same
        positional constructor contract as ``PluginManager.create_transform()``.
        It must not require external services, network calls, or credentials —
        the invariant harness runs in CI and probes transforms in isolation.
        """
        raise NotImplementedError(
            f"{cls.__name__}.probe_config() is not implemented. "
            "Transforms with passes_through_input=True must declare how to "
            "instantiate in isolation. Implement probe_config() or remove the annotation."
        )

    def forward_invariant_probe_rows(self, probe: PipelineRow) -> list[PipelineRow]:
        """Return representative input rows for ADR-009's forward invariant.

        The default harness drives annotated pass-through transforms with a
        single scalar ``probe`` row. Config-sensitive transforms can override
        this to add the specific fields/values their ``probe_config()``
        requires while preserving the randomized background row shape.
        """
        return [probe]

    def backward_invariant_probe_rows(self, probe: PipelineRow) -> list[PipelineRow]:
        """Return representative input rows for ADR-009's backward invariant.

        The default harness drives non-pass-through transforms with a single
        scalar ``probe`` row. Batch-aware transforms whose non-pass-through
        semantics only appear under mixed-validity batches can override this
        to supply a more representative shape.
        """
        return [probe]

    def execute_forward_invariant_probe(
        self,
        probe_rows: list[PipelineRow],
        ctx: Any,
    ) -> TransformResult:
        """Execute the forward invariant probe using the production path.

        Default behavior mirrors runtime dispatch:
        - batch-aware transforms receive the full probe row list
        - single-row transforms must receive exactly one probe row

        Transforms with transport or concurrency seams that cannot be exercised
        via plain ``process()`` override this hook rather than teaching the
        invariant harness about plugin-specific internals.
        """
        if self.is_batch_aware:
            return self.process(probe_rows, ctx)  # type: ignore[arg-type]
        if len(probe_rows) != 1:
            raise FrameworkBugError(
                f"{self.__class__.__name__}.execute_forward_invariant_probe() received {len(probe_rows)} rows for a non-batch transform."
            )
        return self.process(probe_rows[0], ctx)

    def execute_backward_invariant_probe(
        self,
        probe_rows: list[PipelineRow],
        ctx: Any,
    ) -> TransformResult:
        """Execute the backward invariant probe.

        Defaults to the same execution path as the forward probe. Non-pass-through
        transforms can override this when their representative drop path needs a
        special local seam.
        """
        return self.execute_forward_invariant_probe(probe_rows, ctx)

    @staticmethod
    def _augment_invariant_probe_row(
        probe: PipelineRow,
        *,
        field_name: str,
        value: Any,
    ) -> PipelineRow:
        """Return ``probe`` plus one guaranteed field for invariant helpers."""
        from elspeth.contracts.contract_propagation import propagate_contract

        output = probe.to_dict().copy()
        output[field_name] = value
        contract = propagate_contract(
            probe.contract,
            output,
            transform_adds_fields=True,
        )
        return PipelineRow(output, contract)

    @staticmethod
    def _create_schemas(
        schema_config: Any,
        name: str,
        *,
        adds_fields: bool = False,
    ) -> tuple[type[PluginSchema], type[PluginSchema]]:
        """Create input/output schema pair from config.

        Reduces boilerplate for the common two-schema pattern:
        - Shape-preserving transforms: input and output share the same schema.
        - Shape-changing transforms: output uses observed mode (accepts any fields).

        Args:
            schema_config: The plugin's SchemaConfig instance.
            name: Plugin name for schema class naming.
            adds_fields: If True, output schema uses observed mode
                (accepts any fields since output shape is dynamic).

        Returns:
            Tuple of (input_schema, output_schema).
        """
        from elspeth.contracts.schema import SchemaConfig
        from elspeth.plugins.infrastructure.schema_factory import create_schema_from_config

        input_schema = create_schema_from_config(
            schema_config,
            f"{name}Input",
            allow_coercion=False,
        )
        if adds_fields:
            output_schema = create_schema_from_config(
                SchemaConfig.from_dict({"mode": "observed"}),
                f"{name}Output",
                allow_coercion=False,
            )
        else:
            output_schema = input_schema
        return input_schema, output_schema

    def _build_output_schema_config(self, schema_config: SchemaConfig) -> SchemaConfig:
        """Build output schema config for DAG contract propagation.

        Merges the transform's declared_output_fields into guaranteed_fields
        so the DAG builder can validate downstream field requirements.

        Default assumes output ⊇ input fields (additive transforms). Subclasses
        with reductive output — where emitted rows do NOT carry the user's
        input ``fields``/``required_fields``/``guaranteed_fields`` — MUST
        override this method to drop input-side declarations. See
        ``BatchStats._build_output_schema_config`` (elspeth-f5f798f797) for the
        canonical override pattern.

        Args:
            schema_config: The transform's input schema config (base fields).

        Returns:
            SchemaConfig with guaranteed_fields including declared output fields.
        """
        from elspeth.contracts.schema import SchemaConfig

        base_guaranteed = set(schema_config.guaranteed_fields or ())
        output_fields = base_guaranteed | self.declared_output_fields

        # Preserve None-vs-empty-tuple semantics: None = abstain, () = explicitly empty.
        # If upstream declared guarantees or we computed non-empty output, declare explicitly.
        upstream_declared = schema_config.guaranteed_fields is not None
        if upstream_declared or output_fields:
            guaranteed_fields_result = tuple(sorted(output_fields))
        else:
            guaranteed_fields_result = None

        return SchemaConfig(
            mode=schema_config.mode,
            fields=schema_config.fields,
            guaranteed_fields=guaranteed_fields_result,
            audit_fields=schema_config.audit_fields,
            required_fields=schema_config.required_fields,
        )

    def process(
        self,
        row: PipelineRow,
        ctx: TransformContext,
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
        No context is available -- this is pure resource teardown.

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

    def on_start(self, ctx: LifecycleContext) -> None:
        """Called once before any rows are processed.

        Override for per-run initialization: capturing the recorder,
        acquiring rate limiters, initializing tracing, etc.

        Called on the main thread. If this raises, the pipeline aborts
        and neither on_complete() nor close() will be called.

        Subclasses MUST call super().on_start(ctx) to set the lifecycle flag.
        """
        self._on_start_called = True

    def on_complete(self, ctx: LifecycleContext) -> None:  # noqa: B027 - optional hook
        """Called after all rows are processed (or after pipeline error).

        Override for recording final metrics, flushing application-level
        buffers, or updating audit state. Always called before close().

        Called on the main thread. Receives LifecycleContext so it can
        interact with the landscape and telemetry. Individually protected:
        if this raises, other plugins still get their on_complete/close calls.
        """
        pass

    # ── Plugin-declared semantics (Phase 1: optional, default empty) ──
    # Override on a subclass to declare what the plugin emits / requires.
    # The generic semantic validator compares producer facts to consumer
    # requirements when the configured field names match.

    def output_semantics(self) -> OutputSemanticDeclaration:
        """Return semantic facts for the fields this transform emits.

        Default returns an empty declaration: the transform makes no
        semantic claims beyond what the schema contract already
        expresses. Override to declare ContentKind/TextFraming for
        configured output fields.
        """
        from elspeth.contracts.plugin_semantics import OutputSemanticDeclaration

        return OutputSemanticDeclaration()

    def input_semantic_requirements(self) -> InputSemanticRequirements:
        """Return semantic requirements for fields this transform consumes.

        Default returns no requirements. Override to declare that a
        configured input field must satisfy specific ContentKind /
        TextFraming acceptance sets.
        """
        from elspeth.contracts.plugin_semantics import InputSemanticRequirements

        return InputSemanticRequirements()

    @classmethod
    def get_agent_assistance(
        cls,
        *,
        issue_code: str | None = None,
    ) -> PluginAssistance | None:
        """Return deterministic guidance keyed by an issue code.

        Default returns None: the plugin offers no specific guidance.
        Override to return a PluginAssistance instance describing fixes
        for this plugin's issue codes. Validators attach the issue
        code; the plugin owns the prose.
        """
        return None


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
          or updating audit state. Receives LifecycleContext.
        - close(): "Release all resources." Use for closing file handles or
          network connections. No context -- pure resource teardown.

    Example:
        class CSVSink(BaseSink):
            name = "csv"
            input_schema = RowSchema
            idempotent = False

            def write(self, rows: list[dict], ctx: SinkContext) -> SinkWriteResult:
                for row in rows:
                    self._writer.writerow(row)
                return SinkWriteResult(
                    artifact=ArtifactDescriptor.for_file(
                        path=self._path,
                        content_hash=self._compute_hash(),
                        size_bytes=self._file.tell(),
                    ),
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

    # Audit metadata
    determinism: Determinism = Determinism.IO_WRITE
    plugin_version: str = "0.0.0"
    source_file_hash: str | None = None

    # Config model — each subclass sets this to its Pydantic config class.
    config_model: ClassVar[type[PluginConfig] | None] = None

    @classmethod
    def get_config_model(cls, config: dict[str, Any] | None = None) -> type[PluginConfig] | None:
        """Return the Pydantic config model for this plugin type."""
        return cls.config_model

    @classmethod
    def get_config_schema(cls) -> dict[str, Any]:
        """Default ``get_config_schema`` — renders a single Pydantic model.

        See :meth:`~elspeth.contracts.plugin_protocols.SinkProtocol.get_config_schema`
        for the canonical contract, including the MUST-override rule for
        plugins whose effective configuration is a discriminated union.
        """
        if cls.config_model is None:
            return {}
        schema: dict[str, Any] = cls.config_model.model_json_schema()
        return schema

    # Default: sinks don't support resume. Override in subclasses that can append.
    supports_resume: bool = False

    # Required-field enforcement (centralized in SinkExecutor).
    # Sinks set this from schema_config.get_effective_required_fields() at init.
    # Empty frozenset = no required-field check.
    declared_required_fields: frozenset[str] = frozenset()

    # Failsink infrastructure — set by orchestrator from SinkSettings.on_write_failure.
    # None until injected at pipeline startup; "discard" or sink name at runtime.
    _on_write_failure: str | None

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

    def validate_output_target(self) -> OutputValidationResult:
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

    @property
    def needs_resume_field_resolution(self) -> bool:
        """Whether this sink needs field resolution mapping for resume.

        True when headers mode is ORIGINAL — the CLI resume path must
        provide the source field resolution mapping before validation.

        Set by init_display_headers(). Sinks that don't use display headers
        return False (the default).
        """
        return self._needs_resume_field_resolution

    def set_resume_field_resolution(self, resolution_mapping: dict[str, str]) -> None:
        """Set field resolution mapping for resume validation.

        Default is a no-op. Only sinks with headers: original mode
        override this to use the mapping for validation.

        Args:
            resolution_mapping: Dict mapping original header name -> normalized field name.
        """
        # Intentional no-op - most sinks don't use headers: original
        _ = resolution_mapping  # Explicitly consume the argument

    # Output contract for schema-aware sinks
    _output_contract: SchemaContract | None = None

    # Display header state — set by init_display_headers() in subclass __init__.
    # Declared here for mypy structural typing against DisplayHeaderHost protocol.
    _headers_mode: HeaderMode
    _headers_custom_mapping: dict[str, str] | None
    _resolved_display_headers: dict[str, str] | None
    _display_headers_resolved: bool
    _needs_resume_field_resolution: bool

    def __init__(self, config: dict[str, Any]) -> None:
        """Initialize with configuration.

        Args:
            config: Plugin configuration
        """
        self.config = config
        self._on_write_failure: str | None = None
        self._output_contract = None
        self._needs_resume_field_resolution = False
        self._diversion_log: list[RowDiversion] = []

    @abstractmethod
    def write(
        self,
        rows: list[dict[str, Any]],
        ctx: SinkContext,
    ) -> SinkWriteResult:
        """Write a batch of rows to the sink.

        Args:
            rows: List of row dicts to write
            ctx: Sink context with run identity and recording methods

        Returns:
            SinkWriteResult with artifact descriptor and optional diversions
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

    # === Output Contract Support ===

    def get_output_contract(self) -> SchemaContract | None:
        """Get the current output contract.

        Returns:
            SchemaContract if set, None otherwise
        """
        return self._output_contract

    def set_output_contract(self, contract: SchemaContract) -> None:
        """Set or update the output contract.

        Used for schema-aware sinks that need field metadata (e.g., for
        restoring original header names in CSV output).

        Args:
            contract: The schema contract to use for output operations
        """
        self._output_contract = contract

    # === Diversion Infrastructure ===

    def _divert_row(self, row_data: dict[str, Any], *, row_index: int, reason: str) -> None:
        """Record a row diversion during write().

        Called by concrete sinks when an individual row fails at the
        external system boundary (Tier 2 -> External). Both "discard"
        and failsink modes accumulate to _diversion_log. The executor
        reads the log after write() returns and handles the actual
        discard-vs-write decision.

        Args:
            row_data: The row dict that couldn't be written.
            row_index: Index in the original batch (for token correlation).
            reason: Human-readable reason for the diversion.

        Raises:
            FrameworkBugError: If _on_write_failure has not been set
                (plugin bug — calling _divert_row before orchestrator injection).
        """
        if self._on_write_failure is None:
            raise FrameworkBugError(
                f"Sink '{self.name}' called _divert_row() but _on_write_failure "
                f"is not set. Configure on_write_failure in pipeline YAML or "
                f"re-raise the exception to crash the pipeline."
            )
        self._diversion_log.append(RowDiversion(row_index=row_index, reason=reason, row_data=row_data))

    def _reset_diversion_log(self) -> None:
        """Clear the diversion log. Called by SinkExecutor before each write()."""
        self._diversion_log.clear()

    def _get_diversions(self) -> tuple[RowDiversion, ...]:
        """Return accumulated diversions as an immutable tuple."""
        return tuple(self._diversion_log)

    # === Lifecycle Hooks ===
    # Call ordering: on_start -> write/flush -> on_complete -> close
    # See class docstring for full lifecycle contract and guarantees.

    def on_start(self, ctx: LifecycleContext) -> None:  # noqa: B027 - optional hook
        """Called once before any write() call.

        Override for per-run initialization. Called on the main thread.
        If this raises, the pipeline aborts and neither on_complete()
        nor close() will be called.
        """
        pass

    def on_complete(self, ctx: LifecycleContext) -> None:  # noqa: B027 - optional hook
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
        - on_complete(ctx): "Loading is done." Receives LifecycleContext.
        - close(): "Release all resources." No context -- pure teardown.

    Example:
        class CSVSource(BaseSource):
            name = "csv"
            output_schema = RowSchema

            def load(self, ctx: SourceContext) -> Iterator[SourceRow]:
                with open(self.config["path"]) as f:
                    reader = csv.DictReader(f)
                    for row in reader:
                        yield SourceRow.valid(row, contract=contract)

            def close(self) -> None:
                pass  # File already closed by context manager
    """

    name: str
    output_schema: type[PluginSchema]
    node_id: str | None = None  # Set by orchestrator after registration

    # Audit metadata
    determinism: Determinism = Determinism.IO_READ
    plugin_version: str = "0.0.0"
    source_file_hash: str | None = None

    # Config model — each subclass sets this to its Pydantic config class.
    # NullSource sets this to None (no config validation needed).
    config_model: ClassVar[type[PluginConfig] | None] = None

    @classmethod
    def get_config_model(cls, config: dict[str, Any] | None = None) -> type[PluginConfig] | None:
        """Return the Pydantic config model for this plugin type.

        Returns None for sources with no config (e.g. NullSource).
        """
        return cls.config_model

    @classmethod
    def get_config_schema(cls) -> dict[str, Any]:
        """Default ``get_config_schema`` — renders a single Pydantic model.

        See :meth:`~elspeth.contracts.plugin_protocols.SourceProtocol.get_config_schema`
        for the canonical contract, including the MUST-override rule for
        plugins whose effective configuration is a discriminated union.
        """
        if cls.config_model is None:
            return {}
        schema: dict[str, Any] = cls.config_model.model_json_schema()
        return schema

    # Sink name for quarantined rows, or "discard" to drop invalid rows
    # All sources must set this - config-based sources get it from SourceDataConfig
    _on_validation_failure: str

    # Success routing: sink name for rows that pass source validation
    # All sources must set this - config-based sources get it from SourceDataConfig
    on_success: str

    # Guaranteed-field enforcement (centralized in the source boundary contract).
    # Sources set this from schema_config.get_effective_guaranteed_fields() at init.
    # Empty frozenset = no guaranteed-field contract.
    declared_guaranteed_fields: frozenset[str] = frozenset()

    # Schema contract for row validation
    _schema_contract: SchemaContract | None = None

    def __init__(self, config: dict[str, Any]) -> None:
        """Initialize with configuration.

        Args:
            config: Plugin configuration
        """
        self.config = config
        self._schema_contract = None
        self.declared_guaranteed_fields = frozenset()

    @abstractmethod
    def load(self, ctx: SourceContext) -> Iterator[SourceRow]:
        """Load and yield rows from the source.

        Args:
            ctx: Source context with run metadata and recording methods

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

    # === Schema Contract Support ===

    def get_schema_contract(self) -> SchemaContract | None:
        """Get the current schema contract.

        Returns:
            SchemaContract if set, None otherwise
        """
        return self._schema_contract

    def require_schema_contract(self) -> SchemaContract:
        """Return the current schema contract or crash on framework invariant failure."""
        contract = self.get_schema_contract()
        if contract is None:
            raise FrameworkBugError(
                f"{type(self).__name__} attempted to yield SourceRow.valid() before establishing "
                "a schema contract. Source plugins must call set_schema_contract() before "
                "emitting valid rows."
            )
        return contract

    def set_schema_contract(self, contract: SchemaContract) -> None:
        """Set or update the schema contract.

        Called during initialization for explicit schemas (FIXED/FLEXIBLE),
        or after first-row inference for OBSERVED mode.

        Args:
            contract: The schema contract to use for validation
        """
        self._schema_contract = contract

    def _initialize_declared_guaranteed_fields(self, schema_config: SchemaConfig) -> None:
        """Normalize the source's runtime guarantee declaration from SchemaConfig.

        Call this after any source-specific schema rewrite so the runtime
        contract surface matches the source's effective guarantees, not the
        caller's raw config dict.
        """
        self.declared_guaranteed_fields = schema_config.get_effective_guaranteed_fields()

    # === Lifecycle Hooks ===
    # Call ordering: on_start -> load -> on_complete -> close
    # See class docstring for full lifecycle contract and guarantees.
    # Skipped entirely during resume runs (NullSource is used instead).

    def on_start(self, ctx: LifecycleContext) -> None:  # noqa: B027 - optional hook
        """Called once before load().

        Override for per-run initialization. Called on the main thread.
        If this raises, the pipeline aborts and neither on_complete()
        nor close() will be called.

        Skipped during resume runs.
        """
        pass

    def on_complete(self, ctx: LifecycleContext) -> None:  # noqa: B027 - optional hook
        """Called after load() completes (or after pipeline error), before close().

        Override for recording final metrics or updating audit state.
        Called on the main thread. Individually protected: if this raises,
        other plugins still get their on_complete/close calls.

        Skipped during resume runs.
        """
        pass

    # === Audit Trail Metadata ===

    def get_field_resolution(self) -> tuple[Mapping[str, str], str | None] | None:
        """Return field resolution mapping computed during load().

        Sources that perform field normalization (e.g., CSVSource with field normalization)
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
