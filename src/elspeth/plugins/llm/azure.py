# src/elspeth/plugins/llm/azure.py
"""Azure OpenAI LLM transform with row-level pipelining.

Self-contained transform that creates its own AuditedLLMClient using
the context's landscape and state_id. Uses BatchTransformMixin for
concurrent row processing with FIFO output ordering.
"""

from __future__ import annotations

from collections.abc import Callable
from threading import Lock
from typing import TYPE_CHECKING, Any, Self

from pydantic import Field, model_validator

from elspeth.contracts import Determinism, TransformErrorReason, TransformResult, propagate_contract
from elspeth.contracts.plugin_context import PluginContext
from elspeth.contracts.schema import SchemaConfig
from elspeth.contracts.schema_contract import PipelineRow
from elspeth.plugins.base import BaseTransform
from elspeth.plugins.batching import BatchTransformMixin, OutputPort
from elspeth.plugins.clients.llm import AuditedLLMClient, LLMClientError
from elspeth.plugins.llm import get_llm_audit_fields, get_llm_guaranteed_fields
from elspeth.plugins.llm.base import LLMConfig
from elspeth.plugins.llm.templates import PromptTemplate, TemplateError
from elspeth.plugins.llm.tracing import (
    AzureAITracingConfig,
    LangfuseTracingConfig,
    TracingConfig,
    parse_tracing_config,
    validate_tracing_config,
)
from elspeth.plugins.schema_factory import create_schema_from_config

if TYPE_CHECKING:
    from openai import AzureOpenAI

    from elspeth.core.landscape.recorder import LandscapeRecorder


class AzureOpenAIConfig(LLMConfig):
    """Azure OpenAI-specific configuration.

    Extends LLMConfig with Azure-specific settings:
    - deployment_name: Azure deployment name (required) - used as model identifier
    - endpoint: Azure OpenAI endpoint URL (required)
    - api_key: Azure OpenAI API key (required)
    - api_version: Azure API version (default: 2024-10-21)

    Pooling options (inherited from LLMConfig):
    - pool_size: Number of concurrent workers (1=sequential, >1=pooled)
    - max_dispatch_delay_ms: Maximum AIMD backoff delay
    - max_capacity_retry_seconds: Timeout for capacity error retries

    Note: The 'model' field from LLMConfig is automatically set to
    deployment_name if not explicitly provided.
    """

    # Override model to make it optional - will default to deployment_name
    model: str = Field(default="", description="Model identifier (defaults to deployment_name)")

    deployment_name: str = Field(..., description="Azure deployment name")
    endpoint: str = Field(..., description="Azure OpenAI endpoint URL")
    api_key: str = Field(..., description="Azure OpenAI API key")
    api_version: str = Field(default="2024-10-21", description="Azure API version")

    # Tier 2: Plugin-internal tracing (optional)
    # Use environment variables for secrets: ${APPLICATIONINSIGHTS_CONNECTION_STRING}
    tracing: dict[str, Any] | None = Field(
        default=None,
        description="Tier 2 tracing configuration (azure_ai, langfuse, or none)",
    )

    @model_validator(mode="after")
    def _set_model_from_deployment(self) -> Self:
        """Set model to deployment_name if not explicitly provided."""
        if not self.model:
            self.model = self.deployment_name
        return self


class AzureLLMTransform(BaseTransform, BatchTransformMixin):
    """LLM transform using Azure OpenAI with row-level pipelining.

    Self-contained transform that creates its own AuditedLLMClient
    internally using ctx.landscape and ctx.state_id. Uses BatchTransformMixin
    for concurrent row processing with FIFO output ordering.

    Architecture:
        Uses BatchTransformMixin for row-level pipelining:
        - Multiple rows in flight simultaneously
        - FIFO output ordering (results emitted in submission order)
        - Backpressure when buffer is full

        Flow:
            Orchestrator → accept() → [RowReorderBuffer] → [Worker Pool]
                → _process_row() → Azure LLM API
                → emit() → OutputPort (sink or next transform)

    Usage:
        # 1. Instantiate
        transform = AzureLLMTransform(config)

        # 2. Connect output port (required before accept())
        transform.connect_output(output_port, max_pending=30)

        # 3. Feed rows (blocks on backpressure)
        for row in source:
            transform.accept(row, ctx)

        # 4. Flush and close
        transform.flush_batch_processing()
        transform.close()

    Configuration example:
        transforms:
          - plugin: azure_llm
            options:
              deployment_name: "my-gpt4o-deployment"
              endpoint: "${AZURE_OPENAI_ENDPOINT}"
              api_key: "${AZURE_OPENAI_KEY}"
              template: |
                Analyze: {{ row.text }}
              schema:
                mode: observed
    """

    name = "azure_llm"

    # LLM transforms are non-deterministic by nature
    determinism: Determinism = Determinism.NON_DETERMINISTIC

    def __init__(self, config: dict[str, Any]) -> None:
        """Initialize Azure LLM transform.

        Args:
            config: Transform configuration dictionary
        """
        super().__init__(config)

        # Parse Azure-specific config to validate all required fields
        cfg = AzureOpenAIConfig.from_dict(config)

        # Store Azure-specific config
        self._azure_endpoint = cfg.endpoint
        self._azure_api_key: str | None = cfg.api_key
        self._azure_api_version = cfg.api_version
        self._deployment_name = cfg.deployment_name

        # Store common LLM settings (from LLMConfig)
        self._pool_size = cfg.pool_size
        self._max_capacity_retry_seconds = cfg.max_capacity_retry_seconds
        self._model = cfg.model or cfg.deployment_name
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
        self._response_field = cfg.response_field

        # Schema from config (TransformDataConfig guarantees schema_config is not None)
        schema_config = cfg.schema_config
        schema = create_schema_from_config(
            schema_config,
            f"{self.name}Schema",
            allow_coercion=False,  # Transforms do NOT coerce
        )
        self.input_schema = schema
        self.output_schema = schema

        # Build output schema config with field categorization
        guaranteed = get_llm_guaranteed_fields(self._response_field)
        audit = get_llm_audit_fields(self._response_field)

        # Merge with any existing fields from base schema
        base_guaranteed = schema_config.guaranteed_fields or ()
        base_audit = schema_config.audit_fields or ()

        self._output_schema_config = SchemaConfig(
            mode=schema_config.mode,
            fields=schema_config.fields,
            guaranteed_fields=tuple(set(base_guaranteed) | set(guaranteed)),
            audit_fields=tuple(set(base_audit) | set(audit)),
            required_fields=schema_config.required_fields,
        )

        # Recorder, telemetry, and rate limit context (set in on_start)
        self._recorder: LandscapeRecorder | None = None
        self._run_id: str = ""
        self._telemetry_emit: Callable[[Any], None] = lambda event: None
        self._limiter: Any = None  # RateLimiter | NoOpLimiter | None

        # LLM client cache - ensures call_index uniqueness
        # Each state_id gets its own client with monotonically increasing call indices
        self._llm_clients: dict[str, AuditedLLMClient] = {}
        self._llm_clients_lock = Lock()
        # Cache underlying Azure clients to avoid recreating them
        self._underlying_client: AzureOpenAI | None = None
        self._underlying_client_lock = Lock()

        # Batch processing state (initialized by connect_output)
        self._batch_initialized = False

        # Tier 2: Plugin-internal tracing
        self._tracing_config: TracingConfig | None = parse_tracing_config(cfg.tracing)
        self._tracing_active: bool = False
        self._langfuse_client: Any = None  # Langfuse client if configured

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
            max_workers=max_pending,  # Match workers to max_pending
            batch_wait_timeout=float(self._max_capacity_retry_seconds),
        )
        self._batch_initialized = True

    def on_start(self, ctx: PluginContext) -> None:
        """Capture recorder, telemetry, rate limit context, and initialize tracing.

        Called by the engine at pipeline start. Captures the landscape
        recorder, run_id, telemetry callback, and rate limiter for use in worker threads.
        Also initializes Tier 2 tracing if configured.
        """
        self._recorder = ctx.landscape
        self._run_id = ctx.run_id
        self._telemetry_emit = ctx.telemetry_emit
        # Get rate limiter for Azure OpenAI service (None if rate limiting disabled)
        self._limiter = ctx.rate_limit_registry.get_limiter("azure_openai") if ctx.rate_limit_registry is not None else None

        # Initialize Tier 2 tracing if configured
        if self._tracing_config is not None:
            self._setup_tracing()

    def _setup_tracing(self) -> None:
        """Initialize Tier 2 tracing based on provider.

        Tracing is optional - if the required SDK is not installed,
        we log a warning and continue without tracing.
        """
        import structlog

        logger = structlog.get_logger(__name__)

        # Type narrowing for mypy - caller guarantees this is not None
        assert self._tracing_config is not None
        tracing_config = self._tracing_config

        # Validate configuration completeness
        errors = validate_tracing_config(tracing_config)
        if errors:
            for error in errors:
                logger.warning("Tracing configuration error", error=error)
            return  # Don't attempt setup with incomplete config

        match tracing_config.provider:
            case "azure_ai":
                self._setup_azure_ai_tracing(logger, tracing_config)
            case "langfuse":
                self._setup_langfuse_tracing(logger, tracing_config)
            case "none":
                pass  # No tracing
            case _:
                logger.warning(
                    "Unknown tracing provider encountered after validation - tracing disabled",
                    provider=tracing_config.provider,
                )

    def _setup_azure_ai_tracing(self, logger: Any, tracing_config: TracingConfig) -> None:
        """Initialize Azure AI / Application Insights tracing.

        Azure Monitor OpenTelemetry auto-instruments the OpenAI SDK.
        No manual instrumentation needed after configure_azure_monitor().

        WARNING: This is process-level configuration. Multiple plugins
        with azure_ai tracing will share the same configuration.
        """
        try:
            # Check for existing OTEL configuration that might conflict
            from opentelemetry import trace as otel_trace

            if otel_trace.get_tracer_provider().__class__.__name__ != "ProxyTracerProvider":
                logger.warning(
                    "Existing OpenTelemetry tracer detected - Azure AI tracing may conflict with Tier 1 telemetry",
                    existing_provider=otel_trace.get_tracer_provider().__class__.__name__,
                )

            success = _configure_azure_monitor(tracing_config)
            if success:
                self._tracing_active = True
                logger.info(
                    "Azure AI tracing initialized",
                    provider="azure_ai",
                    content_recording=tracing_config.enable_content_recording if isinstance(tracing_config, AzureAITracingConfig) else None,
                    live_metrics=tracing_config.enable_live_metrics if isinstance(tracing_config, AzureAITracingConfig) else None,
                )

        except ImportError:
            logger.warning(
                "Azure AI tracing requested but package not installed",
                provider="azure_ai",
                hint="Install with: uv pip install elspeth[tracing-azure]",
            )

    def _setup_langfuse_tracing(self, logger: Any, tracing_config: TracingConfig) -> None:
        """Initialize Langfuse tracing (v3 API).

        Langfuse v3 uses OpenTelemetry-based context managers for lifecycle.
        The Langfuse client is stored for use in _record_langfuse_trace().
        """
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
                provider="langfuse",
                hint="Install with: uv pip install elspeth[tracing-langfuse]",
            )

    def accept(self, row: PipelineRow, ctx: PluginContext) -> None:
        """Accept a row for processing.

        This is the pipeline entry point. Rows are processed concurrently
        with FIFO output ordering. Blocks when buffer is full (backpressure).

        Args:
            row: Row to process as PipelineRow
            ctx: Plugin context with landscape and state_id

        Raises:
            RuntimeError: If connect_output() was not called
        """
        if not self._batch_initialized:
            raise RuntimeError("connect_output() must be called before accept()")

        self.accept_row(row, ctx, self._process_row)

    def process(
        self,
        row: PipelineRow,
        ctx: PluginContext,
    ) -> TransformResult:
        """Not supported - use accept() for row-level pipelining.

        This transform uses BatchTransformMixin for concurrent row processing
        with FIFO output ordering. Call accept() instead of process().

        Raises:
            NotImplementedError: Always, directing callers to use accept()
        """
        raise NotImplementedError(
            f"{self.__class__.__name__} uses row-level pipelining. Use accept() instead of process(). See class docstring for usage."
        )

    def _process_row(self, row: PipelineRow, ctx: PluginContext) -> TransformResult:
        """Process a single row through Azure OpenAI.

        Called by worker threads from the BatchTransformMixin. Each row is
        processed independently with its own LLM client.

        Error handling follows Three-Tier Trust Model:
        1. Template rendering (THEIR DATA) - wrap, return error
        2. LLM call (EXTERNAL) - wrap, return error
        3. Internal logic (OUR CODE) - let crash

        Args:
            row: Row to process as PipelineRow
            ctx: Plugin context with landscape and state_id

        Returns:
            TransformResult with processed row or error
        """
        # 1. Render template with row data (THEIR DATA - wrap)
        try:
            rendered = self._template.render_with_metadata(row, contract=row.contract)
        except TemplateError as e:
            error_reason: TransformErrorReason = {
                "reason": "template_rendering_failed",
                "error": str(e),
                "template_hash": self._template.template_hash,
            }
            if self._template.template_source:
                error_reason["template_file_path"] = self._template.template_source
            return TransformResult.error(error_reason)

        # 2. Build messages
        messages: list[dict[str, str]] = []
        if self._system_prompt:
            messages.append({"role": "system", "content": self._system_prompt})
        messages.append({"role": "user", "content": rendered.prompt})

        # 3. Get LLM client (cached per state_id for call_index uniqueness)
        if ctx.state_id is None:
            raise RuntimeError("Azure LLM transform requires state_id. Ensure transform is executed through the engine.")

        try:
            import time

            if ctx.token is None:
                raise RuntimeError("Azure LLM transform requires ctx.token. Ensure transform is executed through the engine.")
            token_id = ctx.token.token_id
            llm_client = self._get_llm_client(ctx.state_id, token_id=token_id)

            # 4. Call LLM (EXTERNAL - wrap)
            # Retryable errors (RateLimitError, NetworkError, ServerError) are re-raised
            # to let the engine's RetryManager handle them. Non-retryable errors
            # (ContentPolicyError, ContextLengthError) return TransformResult.error().
            start_time = time.monotonic()

            try:
                response = llm_client.chat_completion(
                    model=self._model,
                    messages=messages,
                    temperature=self._temperature,
                    max_tokens=self._max_tokens,
                )
            except LLMClientError as e:
                # Record failed call in Langfuse for observability (before re-raising or returning)
                latency_ms = (time.monotonic() - start_time) * 1000
                self._record_langfuse_trace_for_error(
                    ctx=ctx,
                    token_id=token_id,
                    prompt=rendered.prompt,
                    error_message=str(e),
                    latency_ms=latency_ms,
                )

                if e.retryable:
                    # Re-raise for engine retry (RateLimitError, NetworkError, ServerError)
                    raise
                # Non-retryable error - return error result
                return TransformResult.error(
                    {"reason": "llm_call_failed", "error": str(e)},
                    retryable=False,
                )

            # Record in Langfuse using v3 nested context managers (after successful call)
            latency_ms = (time.monotonic() - start_time) * 1000
            self._record_langfuse_trace(
                ctx=ctx,
                token_id=token_id,
                prompt=rendered.prompt,
                response_content=response.content,
                usage=response.usage,
                latency_ms=latency_ms,
            )

            # 5. Build output row (OUR CODE - let exceptions crash)
            output = row.to_dict()
            output[self._response_field] = response.content
            output[f"{self._response_field}_usage"] = response.usage
            output[f"{self._response_field}_template_hash"] = rendered.template_hash
            output[f"{self._response_field}_variables_hash"] = rendered.variables_hash
            output[f"{self._response_field}_template_source"] = rendered.template_source
            output[f"{self._response_field}_lookup_hash"] = rendered.lookup_hash
            output[f"{self._response_field}_lookup_source"] = rendered.lookup_source
            output[f"{self._response_field}_system_prompt_source"] = self._system_prompt_source
            output[f"{self._response_field}_model"] = response.model

            # 6. Propagate contract (always present in PipelineRow)
            output_contract = propagate_contract(
                input_contract=row.contract,
                output_row=output,
                transform_adds_fields=True,  # LLM transforms add response field + metadata
            )

            return TransformResult.success(
                PipelineRow(output, output_contract),
                success_reason={"action": "enriched", "fields_added": [self._response_field]},
            )
        finally:
            # Clean up cached client for this state_id to prevent unbounded growth
            with self._llm_clients_lock:
                self._llm_clients.pop(ctx.state_id, None)

    def _get_underlying_client(self) -> AzureOpenAI:
        """Get or create the underlying Azure OpenAI client.

        Thread-safe: protected by _underlying_client_lock to prevent
        duplicate client creation from concurrent worker threads.
        """
        with self._underlying_client_lock:
            if self._underlying_client is None:
                # Import here to avoid hard dependency on openai package
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
        """Get or create LLM client for a state_id.

        Clients are cached to preserve call_index across retries.
        This ensures uniqueness of (state_id, call_index) even when
        the pooled executor retries after CapacityError.

        Thread-safe: multiple workers can call this concurrently.
        """
        with self._llm_clients_lock:
            if state_id not in self._llm_clients:
                if self._recorder is None:
                    raise RuntimeError("Azure transform requires recorder. Ensure on_start was called.")
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

    @property
    def azure_config(self) -> dict[str, Any]:
        """Azure configuration (for reference/debugging).

        Returns:
            Dict containing endpoint, api_version, and provider
        """
        return {
            "endpoint": self._azure_endpoint,
            "api_version": self._azure_api_version,
            "provider": "azure",
        }

    @property
    def deployment_name(self) -> str:
        """Azure deployment name (used as model in API calls)."""
        return self._deployment_name

    def close(self) -> None:
        """Release resources and flush tracing."""
        # Flush Tier 2 tracing if active
        if self._tracing_active:
            self._flush_tracing()

        # Shutdown batch processing infrastructure
        if self._batch_initialized:
            self.shutdown_batch_processing()

        self._recorder = None
        # Clear cached LLM clients
        with self._llm_clients_lock:
            self._llm_clients.clear()
            self._underlying_client = None
        self._langfuse_client = None

    def _flush_tracing(self) -> None:
        """Flush any pending tracing data."""
        import structlog

        logger = structlog.get_logger(__name__)

        # Langfuse needs explicit flush
        if self._langfuse_client is not None:
            try:
                self._langfuse_client.flush()
                logger.debug("Langfuse tracing flushed")
            except Exception as e:
                logger.warning("Failed to flush Langfuse tracing", error=str(e))

        # Azure Monitor handles its own batching/flushing
        # No explicit flush needed

    def _record_langfuse_trace(
        self,
        ctx: PluginContext,
        token_id: str,
        prompt: str,
        response_content: str,
        usage: dict[str, int] | None,
        latency_ms: float | None,
    ) -> None:
        """Record LLM call to Langfuse using v3 nested context managers.

        Langfuse v3 uses OpenTelemetry-based context managers. The span and generation
        are created with start_as_current_observation() and auto-close on exit.

        Args:
            ctx: Plugin context for telemetry emission
            token_id: Token ID for correlation
            prompt: The prompt sent to the LLM
            response_content: The response received
            usage: Token usage dict with prompt_tokens/completion_tokens
            latency_ms: Call latency in milliseconds
        """
        if not self._tracing_active or self._langfuse_client is None:
            return
        if not isinstance(self._tracing_config, LangfuseTracingConfig):
            return

        try:
            with (
                self._langfuse_client.start_as_current_observation(
                    as_type="span",
                    name=f"elspeth.{self.name}",
                    metadata={"token_id": token_id, "plugin": self.name, "deployment": self._deployment_name},
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
                    # Validate types at external boundary (Tier 3 data from LLM API)
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
            # No Silent Failures: emit telemetry event for trace failure
            ctx.telemetry_emit(
                {
                    "event": "langfuse_trace_failed",
                    "plugin": self.name,
                    "error": str(e),
                }
            )
            import structlog

            logger = structlog.get_logger(__name__)
            logger.warning("Failed to record Langfuse trace", error=str(e))

    def _record_langfuse_trace_for_error(
        self,
        ctx: PluginContext,
        token_id: str,
        prompt: str,
        error_message: str,
        latency_ms: float | None,
    ) -> None:
        """Record failed LLM call to Langfuse with ERROR level.

        Called when an LLM call fails (rate limit, policy error, etc.) to ensure
        failed attempts are visible in Langfuse for debugging and correlation.

        Args:
            ctx: Plugin context for telemetry emission
            token_id: Token ID for correlation
            prompt: The prompt that was sent (or attempted)
            error_message: Error description
            latency_ms: Time elapsed before failure in milliseconds
        """
        if not self._tracing_active or self._langfuse_client is None:
            return
        if not isinstance(self._tracing_config, LangfuseTracingConfig):
            return

        try:
            with (
                self._langfuse_client.start_as_current_observation(
                    as_type="span",
                    name=f"elspeth.{self.name}",
                    metadata={"token_id": token_id, "plugin": self.name, "deployment": self._deployment_name},
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
            # No Silent Failures: emit telemetry event for trace failure
            ctx.telemetry_emit(
                {
                    "event": "langfuse_error_trace_failed",
                    "plugin": self.name,
                    "error": str(e),
                }
            )
            import structlog

            logger = structlog.get_logger(__name__)
            logger.warning("Failed to record Langfuse error trace", error=str(e))


def _configure_azure_monitor(config: TracingConfig) -> bool:
    """Configure Azure Monitor (module-level to allow mocking).

    Returns True on success, False on failure.
    """
    from azure.monitor.opentelemetry import (
        configure_azure_monitor,  # type: ignore[import-not-found,import-untyped,attr-defined]  # optional dep: azure-monitor-opentelemetry
    )

    if not isinstance(config, AzureAITracingConfig):
        return False

    configure_azure_monitor(
        connection_string=config.connection_string,
        enable_live_metrics=config.enable_live_metrics,
    )
    return True
