"""Azure Multi-Query LLM transform for case study x criteria evaluation.

Executes multiple LLM queries per row in parallel, merging all results
into a single output row with all-or-nothing error handling.

Uses BatchTransformMixin for row-level pipelining (multiple rows in flight
with FIFO output ordering) and PooledExecutor for query-level concurrency
(parallel LLM queries within each row).
"""

from __future__ import annotations

from threading import Lock
from typing import TYPE_CHECKING, Any, cast

from elspeth.contracts import TransformErrorCategory, TransformResult
from elspeth.contracts.schema_contract import FieldContract, PipelineRow, SchemaContract
from elspeth.plugins.clients.llm import AuditedLLMClient, LLMClientError, RateLimitError
from elspeth.plugins.llm.base_multi_query import BaseMultiQueryTransform
from elspeth.plugins.llm.multi_query import (
    MultiQueryConfig,
    QuerySpec,
    ResponseFormat,
)
from elspeth.plugins.llm.templates import TemplateError
from elspeth.plugins.llm.tracing import (
    AzureAITracingConfig,
    LangfuseTracingConfig,
    TracingConfig,
    validate_tracing_config,
)
from elspeth.plugins.llm.validation import ValidationSuccess, validate_json_object_response
from elspeth.plugins.pooling import CapacityError

if TYPE_CHECKING:
    from openai import AzureOpenAI

    from elspeth.contracts import TransformErrorReason


class AzureMultiQueryLLMTransform(BaseMultiQueryTransform):
    """LLM transform that executes case_studies x criteria queries per row.

    For each row, expands the cross-product of case studies and criteria
    into individual LLM queries. All queries run in parallel (up to pool_size),
    with all-or-nothing error semantics (if any query fails, the row fails).

    Architecture:
        Uses two layers of concurrency:
        1. Row-level pipelining (BatchTransformMixin): Multiple rows in flight,
           FIFO output ordering, backpressure when buffer is full.
        2. Query-level concurrency (PooledExecutor): Parallel LLM queries within
           each row, AIMD backoff on rate limits.

    Configuration example:
        transforms:
          - plugin: azure_multi_query_llm
            options:
              deployment_name: "gpt-4o"
              endpoint: "${AZURE_OPENAI_ENDPOINT}"
              api_key: "${AZURE_OPENAI_KEY}"
              template: |
                Case: {{ input_1 }}, {{ input_2 }}
                Criterion: {{ criterion.name }}
              case_studies:
                - name: cs1
                  input_fields: [cs1_bg, cs1_sym]
              criteria:
                - name: diagnosis
                  code: DIAG
              response_format: structured
              output_mapping:
                score:
                  suffix: score
                  type: integer
              pool_size: 4
              schema:
                mode: observed
    """

    name = "azure_multi_query_llm"

    def __init__(self, config: dict[str, Any]) -> None:
        """Initialize transform with multi-query configuration."""
        super().__init__(config)

        cfg = MultiQueryConfig.from_dict(config)

        # Azure-specific connection settings
        self._azure_endpoint = cfg.endpoint
        self._azure_api_key: str | None = cfg.api_key
        self._azure_api_version = cfg.api_version
        self._deployment_name = cfg.deployment_name
        self._model = cfg.model or cfg.deployment_name

        # Shared multi-query init
        self._init_multi_query(cfg)

        # Azure-specific client caching
        self._llm_clients: dict[str, AuditedLLMClient] = {}
        self._llm_clients_lock = Lock()
        self._underlying_client: AzureOpenAI | None = None
        self._underlying_client_lock = Lock()

    # ------------------------------------------------------------------
    # Abstract method implementations
    # ------------------------------------------------------------------

    def _get_rate_limiter_service_name(self) -> str:
        return "azure_openai"

    def _cleanup_clients(self, state_id: str) -> None:
        with self._llm_clients_lock:
            self._llm_clients.pop(state_id, None)

    def _close_all_clients(self) -> None:
        with self._llm_clients_lock:
            self._llm_clients.clear()
        with self._underlying_client_lock:
            client = self._underlying_client
            self._underlying_client = None
        if client is not None:
            client.close()

    def _record_row_langfuse_trace(
        self,
        token_id: str,
        result: TransformResult,
        latency_ms: float,
    ) -> None:
        """Azure traces per-query in _process_single_query, no row-level trace needed."""

    def _setup_tracing(self) -> None:
        """Initialize Tier 2 tracing based on provider."""
        import structlog

        logger = structlog.get_logger(__name__)

        assert self._tracing_config is not None
        tracing_config = self._tracing_config

        errors = validate_tracing_config(tracing_config)
        if errors:
            for error in errors:
                logger.warning("Tracing configuration error", error=error)
            return

        match tracing_config.provider:
            case "azure_ai":
                self._setup_azure_ai_tracing(logger, tracing_config)
            case "langfuse":
                self._setup_langfuse_tracing(logger, tracing_config)
            case "none":
                pass
            case _:
                logger.warning(
                    "Unknown tracing provider encountered after validation - tracing disabled",
                    provider=tracing_config.provider,
                )

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
        """
        import time

        start_time = time.monotonic()
        # 1. Build synthetic row for PromptTemplate
        synthetic_row = spec.build_template_context(row)

        # 2. Render template using PromptTemplate (preserves audit metadata)
        try:
            rendered = self._template.render_with_metadata(synthetic_row, contract=input_contract)
        except TemplateError as e:
            return TransformResult.error(
                {
                    "reason": "template_rendering_failed",
                    "error": str(e),
                    "query": spec.output_prefix,
                    "template_hash": self._template.template_hash,
                }
            )

        # 3. Build messages
        messages: list[dict[str, str]] = []
        if self._system_prompt:
            messages.append({"role": "system", "content": self._system_prompt})
        messages.append({"role": "user", "content": rendered.prompt})

        # 4. Get LLM client
        llm_client = self._get_llm_client(state_id, token_id=token_id)

        # 5. Call LLM (EXTERNAL - wrap, raise CapacityError for retry)
        effective_max_tokens = spec.max_tokens or self._max_tokens

        llm_kwargs: dict[str, Any] = {
            "model": self._model,
            "messages": messages,
            "temperature": self._temperature,
            "max_tokens": effective_max_tokens,
        }
        llm_kwargs["response_format"] = self._response_format_dict

        try:
            response = llm_client.chat_completion(**llm_kwargs)
        except RateLimitError as e:
            latency_ms = (time.monotonic() - start_time) * 1000
            self._record_langfuse_trace_for_error(
                token_id=token_id,
                query_prefix=spec.output_prefix,
                prompt=rendered.prompt,
                error_message=str(e),
                latency_ms=latency_ms,
            )
            raise CapacityError(429, str(e)) from e
        except LLMClientError as e:
            latency_ms = (time.monotonic() - start_time) * 1000
            self._record_langfuse_trace_for_error(
                token_id=token_id,
                query_prefix=spec.output_prefix,
                prompt=rendered.prompt,
                error_message=str(e),
                latency_ms=latency_ms,
            )

            if e.retryable:
                raise
            return TransformResult.error(
                {
                    "reason": "llm_call_failed",
                    "error": str(e),
                    "error_type": type(e).__name__,
                    "query": spec.output_prefix,
                },
                retryable=False,
            )

        # Record in Langfuse (per-query trace)
        latency_ms = (time.monotonic() - start_time) * 1000
        self._record_langfuse_trace(
            token_id=token_id,
            query_prefix=spec.output_prefix,
            prompt=rendered.prompt,
            response_content=response.content,
            usage=response.usage,
            latency_ms=latency_ms,
        )

        # 6. Check for response truncation BEFORE parsing
        completion_tokens = response.usage.get("completion_tokens", 0)
        if effective_max_tokens is not None and completion_tokens > 0 and completion_tokens >= effective_max_tokens:
            truncation_error: TransformErrorReason = {
                "reason": "response_truncated",
                "error": (
                    f"LLM response was truncated at {completion_tokens} tokens "
                    f"(max_tokens={effective_max_tokens}). "
                    f"Increase max_tokens for query '{spec.output_prefix}' or shorten your prompt."
                ),
                "query": spec.output_prefix,
                "max_tokens": effective_max_tokens,
                "completion_tokens": completion_tokens,
                "prompt_tokens": response.usage.get("prompt_tokens", 0),
            }
            if response.content:
                truncation_error["raw_response_preview"] = response.content[:500]
            return TransformResult.error(truncation_error)

        # 7. Parse JSON response (THEIR DATA - wrap)
        content = response.content.strip()

        if self._response_format == ResponseFormat.STANDARD and content.startswith("```"):
            first_newline = content.find("\n")
            if first_newline != -1:
                content = content[first_newline + 1 :]
            if content.endswith("```"):
                content = content[:-3].strip()

        validation_result = validate_json_object_response(content)
        if not isinstance(validation_result, ValidationSuccess):
            error_info: TransformErrorReason = {
                "reason": cast(TransformErrorCategory, validation_result.reason),
                "query": spec.output_prefix,
            }
            if response.content:
                error_info["raw_response"] = response.content[:500]
            if validation_result.detail:
                error_info["error"] = validation_result.detail
                error_info["content_after_fence_strip"] = content
                error_info["usage"] = response.usage
            if validation_result.expected:
                error_info["expected"] = validation_result.expected
            if validation_result.actual:
                error_info["actual"] = validation_result.actual
            return TransformResult.error(error_info)

        parsed = validation_result.data

        # 8. Map and validate output fields
        output: dict[str, Any] = {}
        for json_field, field_config in self._output_mapping.items():
            output_key = f"{spec.output_prefix}_{field_config.suffix}"
            if json_field not in parsed:
                return TransformResult.error(
                    {
                        "reason": "missing_output_field",
                        "field": json_field,
                        "query": spec.output_prefix,
                    }
                )

            value = parsed[json_field]

            type_error = self._validate_field_type(json_field, value, field_config)
            if type_error is not None:
                return TransformResult.error(
                    {
                        "reason": "type_mismatch",
                        "field": json_field,
                        "expected": field_config.type.value,
                        "actual": type(value).__name__,
                        "value": str(value)[:100],
                        "query": spec.output_prefix,
                    }
                )

            output[output_key] = value

        # 9. Add metadata for audit trail
        output[f"{spec.output_prefix}_usage"] = (
            response.usage
            if response.usage
            else {
                "prompt_tokens": 0,
                "completion_tokens": 0,
            }
        )
        output[f"{spec.output_prefix}_model"] = response.model
        output[f"{spec.output_prefix}_template_hash"] = rendered.template_hash
        output[f"{spec.output_prefix}_variables_hash"] = rendered.variables_hash
        output[f"{spec.output_prefix}_template_source"] = rendered.template_source
        output[f"{spec.output_prefix}_lookup_hash"] = rendered.lookup_hash
        output[f"{spec.output_prefix}_lookup_source"] = rendered.lookup_source
        output[f"{spec.output_prefix}_system_prompt_source"] = self._system_prompt_source

        fields_added = [f"{spec.output_prefix}_{fc.suffix}" for fc in self._output_mapping.values()]
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
            success_reason={"action": "enriched", "fields_added": fields_added},
        )

    # ------------------------------------------------------------------
    # Azure-specific client management
    # ------------------------------------------------------------------

    def _get_underlying_client(self) -> AzureOpenAI:
        """Get or create the underlying Azure OpenAI client.

        Thread-safe: protected by _underlying_client_lock.
        """
        with self._underlying_client_lock:
            if self._underlying_client is None:
                from openai import AzureOpenAI

                self._underlying_client = AzureOpenAI(
                    azure_endpoint=self._azure_endpoint,
                    api_key=self._azure_api_key,
                    api_version=self._azure_api_version,
                )
                # Clear plaintext key — SDK client holds its own copy internally
                self._azure_api_key = None
            return self._underlying_client

    def _get_llm_client(self, state_id: str, *, token_id: str | None = None) -> AuditedLLMClient:
        """Get or create LLM client for a state_id."""
        with self._llm_clients_lock:
            if state_id not in self._llm_clients:
                if self._recorder is None:
                    raise RuntimeError("Transform requires recorder. Ensure on_start was called.")
                self._llm_clients[state_id] = AuditedLLMClient(
                    recorder=self._recorder,
                    state_id=state_id,
                    run_id=self._run_id,
                    telemetry_emit=self._telemetry_emit,
                    underlying_client=self._get_underlying_client(),
                    provider="azure",
                    limiter=self._limiter,
                    token_id=token_id,
                )
            return self._llm_clients[state_id]

    # ------------------------------------------------------------------
    # Azure-specific tracing
    # ------------------------------------------------------------------

    def _setup_azure_ai_tracing(self, logger: Any, tracing_config: TracingConfig) -> None:
        """Initialize Azure AI tracing."""
        try:
            from opentelemetry import trace as otel_trace

            if otel_trace.get_tracer_provider().__class__.__name__ != "ProxyTracerProvider":
                logger.warning(
                    "Existing OpenTelemetry tracer detected - Azure AI tracing may conflict",
                    existing_provider=otel_trace.get_tracer_provider().__class__.__name__,
                )

            from elspeth.plugins.llm.azure import _configure_azure_monitor

            success = _configure_azure_monitor(tracing_config)
            if success:
                self._tracing_active = True
                logger.info(
                    "Azure AI tracing initialized",
                    provider="azure_ai",
                    content_recording=tracing_config.enable_content_recording if isinstance(tracing_config, AzureAITracingConfig) else None,
                )

        except ImportError:
            logger.warning(
                "Azure AI tracing requested but package not installed",
                hint="Install with: uv pip install elspeth[tracing-azure]",
            )

    def _setup_langfuse_tracing(self, logger: Any, tracing_config: TracingConfig) -> None:
        """Initialize Langfuse tracing (v3 API)."""
        try:
            from langfuse import Langfuse  # type: ignore[import-not-found,import-untyped]  # optional dep, no stubs

            if not isinstance(tracing_config, LangfuseTracingConfig):
                return

            self._langfuse_client = Langfuse(
                public_key=tracing_config.public_key,
                secret_key=tracing_config.secret_key,
                host=tracing_config.host,
                tracing_enabled=tracing_config.tracing_enabled,
            )
            self._tracing_active = True

            logger.info(
                "Langfuse tracing initialized (v3)",
                provider="langfuse",
                host=tracing_config.host,
                tracing_enabled=tracing_config.tracing_enabled,
            )

        except ImportError:
            logger.warning(
                "Langfuse tracing requested but package not installed",
                hint="Install with: uv pip install elspeth[tracing-langfuse]",
            )

    def _record_langfuse_trace(
        self,
        token_id: str,
        query_prefix: str,
        prompt: str,
        response_content: str,
        usage: dict[str, int] | None,
        latency_ms: float | None,
    ) -> None:
        """Record LLM call to Langfuse using v3 nested context managers."""
        if not self._tracing_active or self._langfuse_client is None:
            return
        if not isinstance(self._tracing_config, LangfuseTracingConfig):
            return

        try:
            with (
                self._langfuse_client.start_as_current_observation(
                    as_type="span",
                    name=f"elspeth.{self.name}",
                    metadata={"token_id": token_id, "plugin": self.name, "query": query_prefix},
                ),
                self._langfuse_client.start_as_current_observation(
                    as_type="generation",
                    name="llm_call",
                    model=self._model,
                    input=[{"role": "user", "content": prompt}],
                ) as generation,
            ):
                update_kwargs: dict[str, Any] = {"output": response_content}

                if usage:
                    prompt_tokens = usage.get("prompt_tokens", 0)
                    completion_tokens = usage.get("completion_tokens", 0)
                    if isinstance(prompt_tokens, int) and isinstance(completion_tokens, int):
                        update_kwargs["usage_details"] = {
                            "input": prompt_tokens,
                            "output": completion_tokens,
                        }

                if latency_ms is not None:
                    update_kwargs["metadata"] = {"latency_ms": latency_ms}

                generation.update(**update_kwargs)
        except Exception as e:
            import structlog

            logger = structlog.get_logger(__name__)
            logger.warning("Failed to record Langfuse trace", error=str(e), query=query_prefix)

    def _record_langfuse_trace_for_error(
        self,
        token_id: str,
        query_prefix: str,
        prompt: str,
        error_message: str,
        latency_ms: float | None,
    ) -> None:
        """Record failed LLM call to Langfuse with ERROR level."""
        if not self._tracing_active or self._langfuse_client is None:
            return
        if not isinstance(self._tracing_config, LangfuseTracingConfig):
            return

        try:
            with (
                self._langfuse_client.start_as_current_observation(
                    as_type="span",
                    name=f"elspeth.{self.name}",
                    metadata={"token_id": token_id, "plugin": self.name, "query": query_prefix},
                ),
                self._langfuse_client.start_as_current_observation(
                    as_type="generation",
                    name="llm_call",
                    model=self._model,
                    input=[{"role": "user", "content": prompt}],
                ) as generation,
            ):
                update_kwargs: dict[str, Any] = {
                    "level": "ERROR",
                    "status_message": error_message,
                }

                if latency_ms is not None:
                    update_kwargs["metadata"] = {"latency_ms": latency_ms}

                generation.update(**update_kwargs)
        except Exception as e:
            import structlog

            logger = structlog.get_logger(__name__)
            logger.warning("Failed to record Langfuse error trace", error=str(e), query=query_prefix)
