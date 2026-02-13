"""Base class for multi-query LLM transforms.

Provides shared logic for case_studies x criteria cross-product evaluation.
Subclasses implement provider-specific client management and LLM calls.

Azure and OpenRouter multi-query transforms inherit from BaseMultiQueryTransform
and implement the abstract methods for their respective API protocols.
"""

from __future__ import annotations

import time
from abc import ABC, abstractmethod
from collections.abc import Callable
from typing import TYPE_CHECKING, Any, cast

from elspeth.contracts import Determinism, TransformResult, propagate_contract
from elspeth.contracts.errors import QueryFailureDetail
from elspeth.contracts.plugin_context import PluginContext
from elspeth.contracts.schema import SchemaConfig
from elspeth.contracts.schema_contract import FieldContract, PipelineRow, SchemaContract
from elspeth.plugins.base import BaseTransform
from elspeth.plugins.batching import BatchTransformMixin, OutputPort
from elspeth.plugins.clients.llm import LLMClientError
from elspeth.plugins.llm import get_llm_audit_fields, get_multi_query_guaranteed_fields
from elspeth.plugins.llm.multi_query import OutputFieldConfig, OutputFieldType, QuerySpec, ResponseFormat
from elspeth.plugins.llm.templates import PromptTemplate
from elspeth.plugins.llm.tracing import LangfuseTracingConfig, TracingConfig, parse_tracing_config
from elspeth.plugins.pooling import CapacityError, PooledExecutor
from elspeth.plugins.schema_factory import create_schema_from_config

if TYPE_CHECKING:
    from elspeth.core.landscape.recorder import LandscapeRecorder


class BaseMultiQueryTransform(BaseTransform, BatchTransformMixin, ABC):
    """Abstract base class for multi-query LLM transforms.

    Provides the shared pipeline for case_studies x criteria cross-product
    evaluation. Subclasses implement provider-specific concerns:

    - Client creation and lifecycle (_get_llm_client / _get_http_client)
    - Single-query execution (_process_single_query)
    - Tracing setup (_setup_tracing)
    - Rate limiter service name (_get_rate_limiter_service_name)

    Architecture:
        Uses two layers of concurrency:
        1. Row-level pipelining (BatchTransformMixin): Multiple rows in flight,
           FIFO output ordering, backpressure when buffer is full.
        2. Query-level concurrency (PooledExecutor): Parallel LLM queries within
           each row, AIMD backoff on rate limits.

        Flow:
            Orchestrator -> accept() -> [RowReorderBuffer] -> [Worker Pool]
                -> _process_row() -> PooledExecutor -> _process_single_query()
                -> emit() -> OutputPort (sink or next transform)
    """

    creates_tokens = False  # Does not create new tokens (1 row in -> 1 row out)
    transforms_adds_fields = True  # Multi-query adds output_prefix result fields per query spec
    determinism: Determinism = Determinism.NON_DETERMINISTIC
    plugin_version = "1.0.0"

    # Subclasses MUST set this in __init__ before calling _init_multi_query
    _model: str

    def _init_multi_query(self, cfg: Any) -> None:
        """Initialize shared multi-query state from parsed config.

        Call from subclass __init__ after parsing provider-specific config
        and setting self._model.

        Args:
            cfg: Parsed config satisfying both LLMConfig and MultiQueryConfigMixin.
                Must have: template, template_source, lookup, lookup_source,
                system_prompt, system_prompt_source, temperature, max_tokens,
                max_capacity_retry_seconds, schema_config, pool_config,
                output_mapping, response_format, expand_queries(),
                build_response_format(), tracing.
        """
        # Store template settings
        self._template = PromptTemplate(
            cfg.template,
            template_source=cfg.template_source,
            lookup_data=cfg.lookup,
            lookup_source=cfg.lookup_source,
        )
        self._system_prompt = cfg.system_prompt
        self._system_prompt_source = cfg.system_prompt_source
        self._temperature = cfg.temperature
        self._max_tokens = cfg.max_tokens
        self._max_capacity_retry_seconds = cfg.max_capacity_retry_seconds

        # Multi-query specific settings
        self._output_mapping: dict[str, OutputFieldConfig] = cfg.output_mapping
        self._response_format: ResponseFormat = cfg.response_format
        self._response_format_dict: dict[str, Any] = cfg.build_response_format()

        # Pre-expand query specs (case_studies x criteria)
        self._query_specs: list[QuerySpec] = cfg.expand_queries()

        # Build output schema config with field categorization
        schema_config = cfg.schema_config
        all_guaranteed: set[str] = set()
        for spec in self._query_specs:
            all_guaranteed.update(get_multi_query_guaranteed_fields(spec.output_prefix))
            for field_config in self._output_mapping.values():
                all_guaranteed.add(f"{spec.output_prefix}_{field_config.suffix}")

        all_audit = {field for spec in self._query_specs for field in get_llm_audit_fields(spec.output_prefix)}

        base_guaranteed = schema_config.guaranteed_fields or ()
        base_audit = schema_config.audit_fields or ()

        self._output_schema_config = SchemaConfig(
            mode=schema_config.mode,
            fields=schema_config.fields,
            guaranteed_fields=tuple(set(base_guaranteed) | all_guaranteed),
            audit_fields=tuple(set(base_audit) | all_audit),
            required_fields=schema_config.required_fields,
        )

        # Create schema from config
        schema = create_schema_from_config(
            schema_config,
            f"{self.name}Schema",
            allow_coercion=False,
        )
        self.input_schema = schema
        self.output_schema = schema

        # Pooled execution setup
        if cfg.pool_config is not None:
            self._executor: PooledExecutor | None = PooledExecutor(cfg.pool_config)
        else:
            self._executor = None

        # Runtime state (set in on_start)
        self._recorder: LandscapeRecorder | None = None
        self._run_id: str = ""
        self._telemetry_emit: Callable[[Any], None] = lambda event: None
        self._limiter: Any = None  # RateLimiter | NoOpLimiter | None

        # Tier 2: Plugin-internal tracing
        self._tracing_config: TracingConfig | None = parse_tracing_config(cfg.tracing)
        self._tracing_active: bool = False
        self._langfuse_client: Any = None

        # Batch processing state (initialized by connect_output)
        self._batch_initialized = False

    # ------------------------------------------------------------------
    # Concrete methods: pipeline lifecycle
    # ------------------------------------------------------------------

    def connect_output(
        self,
        output: OutputPort,
        max_pending: int = 30,
    ) -> None:
        """Connect output port and initialize batch processing.

        Call this after __init__ but before accept(). The output port
        receives results in FIFO order (submission order, not completion order).

        Args:
            output: Output port to emit results to (sink adapter or next transform)
            max_pending: Maximum rows in flight before accept() blocks (backpressure)

        Raises:
            RuntimeError: If called more than once
        """
        if self._batch_initialized:
            raise RuntimeError("connect_output() already called")

        self.init_batch_processing(
            max_pending=max_pending,
            output=output,
            name=self.name,
            max_workers=max_pending,
            batch_wait_timeout=float(self._max_capacity_retry_seconds),
        )
        self._batch_initialized = True

    def on_start(self, ctx: PluginContext) -> None:
        """Capture recorder, telemetry, and rate limit context for pooled execution."""
        self._recorder = ctx.landscape
        self._run_id = ctx.run_id
        self._telemetry_emit = ctx.telemetry_emit
        service = self._get_rate_limiter_service_name()
        self._limiter = ctx.rate_limit_registry.get_limiter(service) if ctx.rate_limit_registry is not None else None

        if self._tracing_config is not None:
            self._setup_tracing()

    def accept(self, row: PipelineRow, ctx: PluginContext) -> None:
        """Accept a row for processing.

        Submits the row to the batch processing pipeline. Returns quickly
        unless backpressure is applied (buffer full). Results flow through
        the output port in FIFO order.

        Args:
            row: Input row with all case study fields as PipelineRow
            ctx: Plugin context with token and landscape

        Raises:
            RuntimeError: If connect_output() was not called
        """
        if not self._batch_initialized:
            raise RuntimeError("connect_output() must be called before accept(). This wires up the output port for result emission.")

        if self._recorder is None and ctx.landscape is not None:
            self._recorder = ctx.landscape

        self.accept_row(row, ctx, self._process_row)

    def process(
        self,
        row: PipelineRow,
        ctx: PluginContext,
    ) -> TransformResult:
        """Not supported - use accept() for row-level pipelining.

        Raises:
            NotImplementedError: Always, directing callers to use accept()
        """
        raise NotImplementedError(
            f"{self.__class__.__name__} uses row-level pipelining. Use accept() instead of process(). See class docstring for usage."
        )

    def close(self) -> None:
        """Release resources and flush tracing."""
        if self._tracing_active:
            self._flush_tracing()

        if self._batch_initialized:
            self.shutdown_batch_processing()

        if self._executor is not None:
            self._executor.shutdown(wait=True)

        self._recorder = None
        self._close_all_clients()
        self._langfuse_client = None

    # ------------------------------------------------------------------
    # Concrete methods: row processing pipeline
    # ------------------------------------------------------------------

    def _process_row(
        self,
        row: PipelineRow,
        ctx: PluginContext,
    ) -> TransformResult:
        """Process a single row with all queries. Called by worker threads.

        Args:
            row: Input row with all case study fields as PipelineRow
            ctx: Plugin context with state_id for audit trail

        Returns:
            TransformResult with all query results merged, or error
        """
        if ctx.state_id is None:
            raise RuntimeError("state_id is required for batch processing. Ensure transform is executed through the engine.")

        row_data = row.to_dict()
        token_id = ctx.token.token_id if ctx.token is not None else "unknown"
        input_contract = row.contract
        start_time = time.monotonic()

        try:
            result = self._process_single_row_internal(row, row_data, ctx.state_id, token_id, input_contract)

            # Row-level Langfuse trace (overridable hook)
            latency_ms = (time.monotonic() - start_time) * 1000
            self._record_row_langfuse_trace(token_id, result, latency_ms)

            # Contract propagation
            if result.status == "success" and result.row is not None:
                output_row = result.row.to_dict()
                output_contract = propagate_contract(
                    input_contract=input_contract,
                    output_row=output_row,
                    transform_adds_fields=True,
                )
                assert result.success_reason is not None, "success status guarantees success_reason"
                return TransformResult.success(
                    PipelineRow(output_row, output_contract),
                    success_reason=result.success_reason,
                    context_after=result.context_after,
                )
            return result
        finally:
            self._cleanup_clients(ctx.state_id)

    def _process_single_row_internal(
        self,
        row_for_queries: PipelineRow | dict[str, Any],
        row_data: dict[str, Any],
        state_id: str,
        token_id: str,
        input_contract: SchemaContract | None,
    ) -> TransformResult:
        """Internal row processing: execute all queries and merge results.

        Args:
            row_for_queries: Input row preserving PipelineRow semantics for query field access
            row_data: Raw normalized row data used for output row assembly
            state_id: State ID for audit trail
            token_id: Token ID for tracing correlation
            input_contract: Schema contract for template dual-name access

        Returns:
            TransformResult with all query results merged, or error
        """
        pool_context: dict[str, Any] | None = None
        if self._executor is not None:
            results, pool_context = self._execute_queries_parallel(row_for_queries, state_id, token_id, input_contract)
        else:
            results = self._execute_queries_sequential(row_for_queries, state_id, token_id, input_contract)

        # Check for failures (all-or-nothing for this row)
        failed = [(spec, r) for spec, r in zip(self._query_specs, results, strict=True) if r.status != "success"]
        if failed:
            return TransformResult.error(
                {
                    "reason": "query_failed",
                    "failed_queries": [
                        cast(
                            QueryFailureDetail,
                            {
                                "query": spec.output_prefix,
                                "error": (r.reason.get("error", r.reason.get("reason", "unknown")) if r.reason is not None else "unknown"),
                            },
                        )
                        for spec, r in failed
                    ],
                    "succeeded_count": len(results) - len(failed),
                    "total_count": len(results),
                },
                context_after=pool_context,
            )

        # Merge all results into output row
        output = row_data.copy()
        for result in results:
            if result.row is not None:
                output.update(result.row)

        all_fields_added = [
            f"{spec.output_prefix}_{field_config.suffix}" for spec in self._query_specs for field_config in self._output_mapping.values()
        ]
        observed = SchemaContract(
            mode="OBSERVED",
            fields=tuple(
                FieldContract(
                    k,
                    k,
                    type(v) if v is not None and type(v) in (int, str, float, bool) else object,
                    False,
                    "inferred",
                )
                for k, v in output.items()
            ),
            locked=True,
        )
        return TransformResult.success(
            PipelineRow(output, observed),
            success_reason={"action": "enriched", "fields_added": all_fields_added},
            context_after=pool_context,
        )

    def _execute_queries_parallel(
        self,
        row: PipelineRow | dict[str, Any],
        state_id: str,
        token_id: str,
        input_contract: SchemaContract | None,
    ) -> tuple[list[TransformResult], dict[str, Any]]:
        """Execute queries in parallel via PooledExecutor with AIMD retry.

        Args:
            row: The input row data
            state_id: State ID for audit trail
            token_id: Token ID for tracing correlation
            input_contract: Schema contract for template dual-name access

        Returns:
            Tuple of (results in query spec order, pool context for audit trail)
        """
        from elspeth.plugins.pooling.executor import RowContext

        if self._executor is None:
            raise RuntimeError("LLM executor not initialized - call initialize() first")

        contexts = [
            RowContext(
                row={"original_row": row, "spec": spec, "token_id": token_id, "input_contract": input_contract},
                state_id=state_id,
                row_index=i,
            )
            for i, spec in enumerate(self._query_specs)
        ]

        entries = self._executor.execute_batch(
            contexts=contexts,
            process_fn=lambda work, work_state_id: self._process_single_query(
                work["original_row"],
                work["spec"],
                work_state_id,
                work["token_id"],
                work["input_contract"],
            ),
        )

        pool_stats = self._executor.get_stats()
        query_ordering = [
            {
                "submit_index": entry.submit_index,
                "complete_index": entry.complete_index,
                "buffer_wait_ms": entry.buffer_wait_ms,
            }
            for entry in entries
        ]
        pool_context = {
            "pool_config": pool_stats["pool_config"],
            "pool_stats": pool_stats["pool_stats"],
            "query_ordering": query_ordering,
        }

        return [entry.result for entry in entries], pool_context

    def _execute_queries_sequential(
        self,
        row: PipelineRow | dict[str, Any],
        state_id: str,
        token_id: str,
        input_contract: SchemaContract | None,
    ) -> list[TransformResult]:
        """Execute queries sequentially (fallback when no executor).

        Without PooledExecutor, capacity errors are not retried - they immediately
        fail the query. This is acceptable for the fallback path.

        Args:
            row: The input row data
            state_id: State ID for audit trail
            token_id: Token ID for tracing correlation
            input_contract: Schema contract for template dual-name access

        Returns:
            List of TransformResults in query spec order
        """
        results: list[TransformResult] = []

        for spec in self._query_specs:
            try:
                result = self._process_single_query(row, spec, state_id, token_id, input_contract)
            except CapacityError as e:
                # No retry in sequential mode - fail immediately
                result = TransformResult.error(
                    {
                        "reason": "rate_limited",
                        "error": str(e),
                        "query": spec.output_prefix,
                    }
                )
            except LLMClientError as e:
                # In concurrent mode, retryable errors are re-raised for pool retry.
                # In sequential mode (no pool), return error result directly.
                result = TransformResult.error(
                    {
                        "reason": "llm_call_failed",
                        "error": str(e),
                        "error_type": type(e).__name__,
                        "query": spec.output_prefix,
                    },
                    retryable=e.retryable,
                )
            results.append(result)

        return results

    # ------------------------------------------------------------------
    # Concrete methods: shared utilities
    # ------------------------------------------------------------------

    def _validate_field_type(
        self,
        field_name: str,
        value: Any,
        field_config: OutputFieldConfig,
    ) -> str | None:
        """Validate that a parsed value matches the expected type.

        Args:
            field_name: Name of the field (for error messages)
            value: The parsed JSON value
            field_config: Expected type configuration

        Returns:
            Error message if validation fails, None if valid
        """
        expected_type = field_config.type

        if expected_type == OutputFieldType.STRING:
            if not isinstance(value, str):
                return f"expected string, got {type(value).__name__}"

        elif expected_type == OutputFieldType.INTEGER:
            if isinstance(value, bool):
                return "expected integer, got boolean"
            if isinstance(value, int) or (isinstance(value, float) and value.is_integer()):
                pass
            else:
                return f"expected integer, got {type(value).__name__}"

        elif expected_type == OutputFieldType.NUMBER:
            if isinstance(value, bool):
                return "expected number, got boolean"
            if not isinstance(value, (int, float)):
                return f"expected number, got {type(value).__name__}"

        elif expected_type == OutputFieldType.BOOLEAN:
            if not isinstance(value, bool):
                return f"expected boolean, got {type(value).__name__}"

        elif expected_type == OutputFieldType.ENUM:
            if not isinstance(value, str):
                return f"expected string (enum), got {type(value).__name__}"
            if field_config.values and value not in field_config.values:
                return f"value '{value}' not in allowed values: {field_config.values}"

        return None

    def _flush_tracing(self) -> None:
        """Flush any pending tracing data."""
        import structlog

        logger = structlog.get_logger(__name__)

        if self._langfuse_client is not None:
            try:
                self._langfuse_client.flush()
                logger.debug("Langfuse tracing flushed")
            except Exception as e:
                logger.warning("Failed to flush Langfuse tracing", error=str(e))

    # ------------------------------------------------------------------
    # Overridable hook: row-level Langfuse tracing
    # ------------------------------------------------------------------

    def _record_row_langfuse_trace(
        self,
        token_id: str,
        result: TransformResult,
        latency_ms: float,
    ) -> None:
        """Record aggregate Langfuse trace for the row.

        Default: records query count and success count (OpenRouter pattern).
        Override to no-op for transforms that trace per-query (Azure pattern).

        Args:
            token_id: Token ID for correlation
            result: The row processing result
            latency_ms: Total row processing latency in milliseconds
        """
        if not self._tracing_active or self._langfuse_client is None:
            return
        if not isinstance(self._tracing_config, LangfuseTracingConfig):
            return

        query_count = len(self._query_specs)
        if result.status == "success":
            succeeded = query_count
        else:
            succeeded = result.reason.get("succeeded_count", 0) if result.reason else 0

        try:
            with (
                self._langfuse_client.start_as_current_observation(
                    as_type="span",
                    name=f"elspeth.{self.name}",
                    metadata={
                        "token_id": token_id,
                        "plugin": self.name,
                        "model": self._model,
                        "query_count": query_count,
                    },
                ),
                self._langfuse_client.start_as_current_observation(
                    as_type="generation",
                    name="multi_query_batch",
                    model=self._model,
                    input=[{"role": "user", "content": f"{query_count} queries"}],
                ) as generation,
            ):
                update_kwargs: dict[str, Any] = {
                    "output": f"{succeeded}/{query_count} succeeded",
                }

                if result.status == "success" and result.row is not None:
                    # Aggregate usage from result row if available
                    total_usage: dict[str, int] = {}
                    for spec in self._query_specs:
                        usage = result.row.get(f"{spec.output_prefix}_usage")
                        if isinstance(usage, dict):
                            for key in ("prompt_tokens", "completion_tokens"):
                                val = usage.get(key, 0)
                                if isinstance(val, int):
                                    total_usage[key] = total_usage.get(key, 0) + val
                    if total_usage:
                        prompt_tokens = total_usage.get("prompt_tokens", 0)
                        completion_tokens = total_usage.get("completion_tokens", 0)
                        if isinstance(prompt_tokens, int) and isinstance(completion_tokens, int):
                            update_kwargs["usage_details"] = {
                                "input": prompt_tokens,
                                "output": completion_tokens,
                            }

                metadata: dict[str, Any] = {
                    "query_count": query_count,
                    "succeeded_count": succeeded,
                    "latency_ms": latency_ms,
                }
                update_kwargs["metadata"] = metadata

                generation.update(**update_kwargs)
        except Exception as e:
            import structlog

            logger = structlog.get_logger(__name__)
            logger.warning("Failed to record Langfuse trace", error=str(e))

    # ------------------------------------------------------------------
    # Abstract methods: subclass must implement
    # ------------------------------------------------------------------

    @abstractmethod
    def _get_rate_limiter_service_name(self) -> str:
        """Return the rate limiter service name for this provider.

        Returns:
            Service name string (e.g., "azure_openai", "openrouter")
        """
        ...

    @abstractmethod
    def _process_single_query(
        self,
        row: PipelineRow | dict[str, Any],
        spec: QuerySpec,
        state_id: str,
        token_id: str,
        input_contract: SchemaContract | None,
    ) -> TransformResult:
        """Process a single query (one case_study x criterion pair).

        Args:
            row: Full input row
            spec: Query specification with input field mapping
            state_id: State ID for audit trail
            token_id: Token ID for tracing correlation
            input_contract: Schema contract for template dual-name access

        Returns:
            TransformResult with mapped output fields

        Raises:
            CapacityError: On rate limit (for pooled retry)
            LLMClientError: On retryable LLM errors (for pooled retry)
        """
        ...

    @abstractmethod
    def _cleanup_clients(self, state_id: str) -> None:
        """Clean up cached clients for a state_id after row processing.

        Called in the finally block of _process_row.

        Args:
            state_id: State ID whose clients should be cleaned up
        """
        ...

    @abstractmethod
    def _close_all_clients(self) -> None:
        """Close all cached clients during transform shutdown.

        Called by close(). Must be thread-safe.
        """
        ...

    @abstractmethod
    def _setup_tracing(self) -> None:
        """Initialize Tier 2 tracing based on provider configuration.

        Called from on_start() when self._tracing_config is not None.
        Must set self._tracing_active = True on success.
        """
        ...
