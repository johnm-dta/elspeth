# src/elspeth/plugins/llm/openrouter.py
"""OpenRouter LLM transform with row-level pipelining.

Self-contained transform that accesses 100+ models via single API.
Uses BatchTransformMixin for concurrent row processing with FIFO output ordering.
"""

from __future__ import annotations

from collections.abc import Callable
from threading import Lock
from typing import TYPE_CHECKING, Any

import httpx
from pydantic import Field

from elspeth.contracts import Determinism, TransformErrorReason, TransformResult, propagate_contract
from elspeth.contracts.plugin_context import PluginContext
from elspeth.contracts.schema import SchemaConfig
from elspeth.contracts.schema_contract import PipelineRow
from elspeth.plugins.base import BaseTransform
from elspeth.plugins.batching import BatchTransformMixin, OutputPort
from elspeth.plugins.clients.http import AuditedHTTPClient
from elspeth.plugins.clients.llm import NetworkError, RateLimitError, ServerError
from elspeth.plugins.llm import get_llm_audit_fields, get_llm_guaranteed_fields
from elspeth.plugins.llm.base import LLMConfig
from elspeth.plugins.llm.templates import PromptTemplate, TemplateError
from elspeth.plugins.llm.tracing import (
    LangfuseTracingConfig,
    TracingConfig,
    parse_tracing_config,
    validate_tracing_config,
)
from elspeth.plugins.schema_factory import create_schema_from_config

if TYPE_CHECKING:
    from elspeth.core.landscape.recorder import LandscapeRecorder


class OpenRouterConfig(LLMConfig):
    """OpenRouter-specific configuration.

    Extends LLMConfig with OpenRouter API settings:
    - api_key: OpenRouter API key (required)
    - base_url: API base URL (default: https://openrouter.ai/api/v1)
    - timeout_seconds: Request timeout (default: 60.0)

    Tier 2 tracing:
    - tracing: Optional Langfuse configuration (azure_ai not supported for OpenRouter)
    """

    api_key: str = Field(..., description="OpenRouter API key")
    base_url: str = Field(
        default="https://openrouter.ai/api/v1",
        description="OpenRouter API base URL",
    )
    timeout_seconds: float = Field(default=60.0, gt=0, description="Request timeout")

    # Tier 2: Plugin-internal tracing (optional, Langfuse only)
    # Azure AI tracing is NOT supported - it auto-instruments the OpenAI SDK,
    # but OpenRouter uses HTTP directly via httpx.
    tracing: dict[str, Any] | None = Field(
        default=None,
        description="Tier 2 tracing configuration (langfuse only - azure_ai not supported)",
    )


class OpenRouterLLMTransform(BaseTransform, BatchTransformMixin):
    """LLM transform using OpenRouter API with row-level pipelining.

    OpenRouter provides access to 100+ models via a unified API.
    Uses BatchTransformMixin for concurrent row processing with FIFO output ordering.

    Architecture:
        Uses BatchTransformMixin for row-level pipelining:
        - Multiple rows in flight simultaneously
        - FIFO output ordering (results emitted in submission order)
        - Backpressure when buffer is full

        Flow:
            Orchestrator → accept() → [RowReorderBuffer] → [Worker Pool]
                → _process_row() → OpenRouter API
                → emit() → OutputPort (sink or next transform)

    Usage:
        # 1. Instantiate
        transform = OpenRouterLLMTransform(config)

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
          - plugin: openrouter_llm
            options:
              model: "anthropic/claude-3-opus"
              template: |
                Analyze: {{ row.text }}
              api_key: "${OPENROUTER_API_KEY}"
              schema:
                mode: observed
    """

    name = "openrouter_llm"

    # LLM transforms are non-deterministic by nature
    determinism: Determinism = Determinism.NON_DETERMINISTIC

    def __init__(self, config: dict[str, Any]) -> None:
        """Initialize OpenRouter LLM transform.

        Args:
            config: Transform configuration dictionary
        """
        super().__init__(config)

        # Parse OpenRouter-specific config (includes all LLMConfig fields)
        cfg = OpenRouterConfig.from_dict(config)

        # Pre-build auth headers — avoids storing the raw API key as a named attribute
        self._request_headers = {
            "Authorization": f"Bearer {cfg.api_key}",
            "HTTP-Referer": "https://github.com/elspeth-rapid",  # Required by OpenRouter
        }
        self._base_url = cfg.base_url
        self._timeout = cfg.timeout_seconds

        # Store common LLM settings (from LLMConfig)
        self._pool_size = cfg.pool_size
        self._max_capacity_retry_seconds = cfg.max_capacity_retry_seconds
        self._model = cfg.model
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

        # Recorder reference (set in on_start or first accept)
        self._recorder: LandscapeRecorder | None = None
        self._run_id: str = ""
        self._telemetry_emit: Callable[[Any], None] = lambda event: None
        self._limiter: Any = None  # RateLimiter | NoOpLimiter | None

        # HTTP client cache - ensures call_index uniqueness across retries
        # Each state_id gets its own client with monotonically increasing call indices
        self._http_clients: dict[str, AuditedHTTPClient] = {}
        self._http_clients_lock = Lock()

        # Batch processing state (initialized by connect_output)
        self._batch_initialized = False

        # Tier 2: Plugin-internal tracing (Langfuse only)
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
        # Get rate limiter for OpenRouter service (None if rate limiting disabled)
        self._limiter = ctx.rate_limit_registry.get_limiter("openrouter") if ctx.rate_limit_registry is not None else None

        # Initialize Tier 2 tracing if configured
        if self._tracing_config is not None:
            self._setup_tracing()

    def _setup_tracing(self) -> None:
        """Initialize Tier 2 tracing based on provider.

        OpenRouter uses HTTP directly (not the OpenAI SDK), so Azure AI
        auto-instrumentation is NOT supported. Only Langfuse (manual spans)
        is available.
        """
        import structlog

        logger = structlog.get_logger(__name__)

        # Type narrowing - caller ensures config is not None
        if self._tracing_config is None:
            return

        # Validate configuration completeness
        errors = validate_tracing_config(self._tracing_config)
        if errors:
            for error in errors:
                logger.warning("Tracing configuration error", error=error)
            return  # Don't attempt setup with incomplete config

        match self._tracing_config.provider:
            case "azure_ai":
                # Azure AI tracing NOT supported for OpenRouter
                logger.warning(
                    "Azure AI tracing not supported for OpenRouter - use Langfuse instead",
                    provider="azure_ai",
                    hint="Azure AI auto-instruments the OpenAI SDK; OpenRouter uses HTTP directly",
                )
                return
            case "langfuse":
                self._setup_langfuse_tracing(logger)
            case "none":
                pass  # No tracing
            case _:
                logger.warning(
                    "Unknown tracing provider encountered after validation - tracing disabled",
                    provider=self._tracing_config.provider,
                )

    def _setup_langfuse_tracing(self, logger: Any) -> None:
        """Initialize Langfuse tracing (v3 API).

        Langfuse v3 uses OpenTelemetry-based context managers for lifecycle.
        The Langfuse client is stored for use in _record_langfuse_trace().
        """
        try:
            from langfuse import Langfuse  # type: ignore[import-not-found,import-untyped]  # optional dep, no stubs

            cfg = self._tracing_config
            if not isinstance(cfg, LangfuseTracingConfig):
                return

            self._langfuse_client = Langfuse(
                public_key=cfg.public_key,
                secret_key=cfg.secret_key,
                host=cfg.host,
                tracing_enabled=cfg.tracing_enabled,
            )
            self._tracing_active = True

            logger.info(
                "Langfuse tracing initialized (v3)",
                provider="langfuse",
                host=cfg.host,
                tracing_enabled=cfg.tracing_enabled,
            )

        except ImportError:
            logger.warning(
                "Langfuse tracing requested but package not installed",
                provider="langfuse",
                hint="Install with: uv pip install elspeth[tracing-langfuse]",
            )

    def _record_langfuse_trace(
        self,
        ctx: PluginContext,
        token_id: str,
        prompt: str,
        response_content: str,
        model: str,
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
            model: Model name
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
                    metadata={"token_id": token_id, "plugin": self.name, "model": model},
                ),
                self._langfuse_client.start_as_current_observation(
                    as_type="generation",
                    name="llm_call",
                    model=model,
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

        Called when an LLM call fails (rate limit, HTTP error, etc.) to ensure
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
                    metadata={"token_id": token_id, "plugin": self.name, "model": self._model},
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
        """Process a single row through OpenRouter API.

        Called by worker threads from the BatchTransformMixin. Each row is
        processed independently with its own HTTP client.

        Error handling follows Three-Tier Trust Model:
        1. Template rendering (THEIR DATA) - wrap, return error
        2. HTTP API call (EXTERNAL) - wrap, return error
        3. Response parsing (EXTERNAL DATA) - wrap, return error

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

        # 2. Build request
        messages: list[dict[str, str]] = []
        if self._system_prompt:
            messages.append({"role": "system", "content": self._system_prompt})
        messages.append({"role": "user", "content": rendered.prompt})

        request_body: dict[str, Any] = {
            "model": self._model,
            "messages": messages,
            "temperature": self._temperature,
        }
        if self._max_tokens is not None:
            request_body["max_tokens"] = self._max_tokens

        # 3. Get HTTP client (cached per state_id for call_index uniqueness)
        if ctx.state_id is None:
            raise RuntimeError("OpenRouter LLM transform requires state_id. Ensure transform is executed through the engine.")

        try:
            import time

            token_id_for_client = ctx.token.token_id if ctx.token is not None else None
            http_client = self._get_http_client(ctx.state_id, token_id=token_id_for_client)

            # 4. Call OpenRouter API (EXTERNAL - wrap)
            token_id = ctx.token.token_id if ctx.token else "unknown"
            start_time = time.monotonic()

            try:
                response = http_client.post(
                    "/chat/completions",
                    json=request_body,
                    headers={"Content-Type": "application/json"},
                )
                response.raise_for_status()
            except httpx.HTTPStatusError as e:
                # Record error trace before handling (for observability)
                latency_ms = (time.monotonic() - start_time) * 1000
                self._record_langfuse_trace_for_error(
                    ctx=ctx,
                    token_id=token_id,
                    prompt=rendered.prompt,
                    error_message=str(e),
                    latency_ms=latency_ms,
                )

                # Retryable HTTP errors (429, 503) must RAISE exceptions for engine RetryManager
                # Matching Azure pattern: non-retryable errors return TransformResult.error()
                status_code = e.response.status_code
                if status_code == 429:
                    raise RateLimitError(f"Rate limited: {e}") from e
                elif status_code >= 500:
                    raise ServerError(f"Server error ({status_code}): {e}") from e
                # Non-retryable HTTP errors (4xx except 429) return error result
                return TransformResult.error(
                    {"reason": "api_call_failed", "error": str(e), "status_code": status_code},
                    retryable=False,
                )
            except httpx.RequestError as e:
                # Record error trace before raising (for observability)
                latency_ms = (time.monotonic() - start_time) * 1000
                self._record_langfuse_trace_for_error(
                    ctx=ctx,
                    token_id=token_id,
                    prompt=rendered.prompt,
                    error_message=str(e),
                    latency_ms=latency_ms,
                )
                # Network errors (timeout, connection refused) are retryable
                raise NetworkError(f"Network error: {e}") from e

            # 5. Parse JSON response (EXTERNAL DATA - wrap)
            try:
                data = response.json()
            except (ValueError, TypeError) as e:
                error_reason_json: TransformErrorReason = {
                    "reason": "invalid_json_response",
                    "error": f"Response is not valid JSON: {e}",
                    "content_type": response.headers.get("content-type", "unknown"),
                }
                if response.text:
                    error_reason_json["body_preview"] = response.text[:500]
                return TransformResult.error(error_reason_json, retryable=False)

            # 6. Extract content from response (EXTERNAL DATA - wrap)
            try:
                choices = data["choices"]
                if not choices:
                    return TransformResult.error(
                        {"reason": "empty_choices", "response": data},
                        retryable=False,
                    )
                content = choices[0]["message"]["content"]
            except (KeyError, IndexError, TypeError) as e:
                return TransformResult.error(
                    {
                        "reason": "malformed_response",
                        "error": f"{type(e).__name__}: {e}",
                        "response_keys": list(data.keys()) if isinstance(data, dict) else None,
                    },
                    retryable=False,
                )

            # 6b. Check for content filtering (null content from provider)
            if content is None:
                return TransformResult.error(
                    {
                        "reason": "content_filtered",
                        "error": "LLM returned null content (likely content-filtered by provider)",
                    },
                    retryable=False,
                )

            # OpenRouter can return {"usage": null} or omit usage entirely.
            # Use `or {}` to handle both missing AND null cases.
            usage = data.get("usage") or {}

            # Record in Langfuse using v3 nested context managers (after successful call)
            latency_ms = (time.monotonic() - start_time) * 1000
            self._record_langfuse_trace(
                ctx=ctx,
                token_id=token_id,
                prompt=rendered.prompt,
                response_content=content,
                model=data.get("model", self._model),
                usage=usage,
                latency_ms=latency_ms,
            )

            # 7. Build output row (OUR CODE - let exceptions crash)
            output = row.to_dict()
            output[self._response_field] = content
            output[f"{self._response_field}_usage"] = usage
            output[f"{self._response_field}_template_hash"] = rendered.template_hash
            output[f"{self._response_field}_variables_hash"] = rendered.variables_hash
            output[f"{self._response_field}_template_source"] = rendered.template_source
            output[f"{self._response_field}_lookup_hash"] = rendered.lookup_hash
            output[f"{self._response_field}_lookup_source"] = rendered.lookup_source
            output[f"{self._response_field}_system_prompt_source"] = self._system_prompt_source
            output[f"{self._response_field}_model"] = data.get("model", self._model)

            # 8. Propagate contract (always present in PipelineRow)
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
            with self._http_clients_lock:
                client = self._http_clients.pop(ctx.state_id, None)
            if client is not None:
                client.close()

    def _get_http_client(self, state_id: str, *, token_id: str | None = None) -> AuditedHTTPClient:
        """Get or create HTTP client for a state_id.

        Clients are cached to preserve call_index across retries.
        This ensures uniqueness of (state_id, call_index) even when
        the pooled executor retries after CapacityError.

        Thread-safe: multiple workers can call this concurrently.
        """
        with self._http_clients_lock:
            if state_id not in self._http_clients:
                if self._recorder is None:
                    raise RuntimeError("OpenRouter transform requires recorder. Ensure on_start was called.")
                self._http_clients[state_id] = AuditedHTTPClient(
                    recorder=self._recorder,
                    state_id=state_id,
                    run_id=self._run_id,
                    telemetry_emit=self._telemetry_emit,
                    timeout=self._timeout,
                    base_url=self._base_url,
                    headers=self._request_headers,
                    limiter=self._limiter,
                    token_id=token_id,
                )
            return self._http_clients[state_id]

    def close(self) -> None:
        """Release resources and flush tracing."""
        # Flush Tier 2 tracing if active
        if self._tracing_active:
            self._flush_tracing()

        # Shutdown batch processing infrastructure
        if self._batch_initialized:
            self.shutdown_batch_processing()

        self._recorder = None
        # Close and clear cached HTTP clients
        with self._http_clients_lock:
            for client in self._http_clients.values():
                client.close()
            self._http_clients.clear()
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
