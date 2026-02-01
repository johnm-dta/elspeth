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

from elspeth.contracts import Determinism, TransformErrorReason, TransformResult
from elspeth.contracts.schema import SchemaConfig
from elspeth.plugins.base import BaseTransform
from elspeth.plugins.batching import BatchTransformMixin, OutputPort
from elspeth.plugins.clients.http import AuditedHTTPClient
from elspeth.plugins.clients.llm import NetworkError, RateLimitError, ServerError
from elspeth.plugins.context import PluginContext
from elspeth.plugins.llm import get_llm_audit_fields, get_llm_guaranteed_fields
from elspeth.plugins.llm.base import LLMConfig
from elspeth.plugins.llm.templates import PromptTemplate, TemplateError
from elspeth.plugins.schema_factory import create_schema_from_config

if TYPE_CHECKING:
    from elspeth.core.landscape.recorder import LandscapeRecorder


class OpenRouterConfig(LLMConfig):
    """OpenRouter-specific configuration.

    Extends LLMConfig with OpenRouter API settings:
    - api_key: OpenRouter API key (required)
    - base_url: API base URL (default: https://openrouter.ai/api/v1)
    - timeout_seconds: Request timeout (default: 60.0)
    """

    api_key: str = Field(..., description="OpenRouter API key")
    base_url: str = Field(
        default="https://openrouter.ai/api/v1",
        description="OpenRouter API base URL",
    )
    timeout_seconds: float = Field(default=60.0, gt=0, description="Request timeout")


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
                fields: dynamic
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

        # Store OpenRouter-specific settings
        self._api_key = cfg.api_key
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
        self._on_error = cfg.on_error

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
            is_dynamic=schema_config.is_dynamic,
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
        """Capture recorder, telemetry, and rate limit context.

        Called by the engine at pipeline start. Captures the landscape
        recorder, run_id, telemetry callback, and rate limiter for use in worker threads.
        """
        self._recorder = ctx.landscape
        self._run_id = ctx.run_id
        self._telemetry_emit = ctx.telemetry_emit
        # Get rate limiter for OpenRouter service (None if rate limiting disabled)
        self._limiter = ctx.rate_limit_registry.get_limiter("openrouter") if ctx.rate_limit_registry is not None else None

    def accept(self, row: dict[str, Any], ctx: PluginContext) -> None:
        """Accept a row for processing.

        This is the pipeline entry point. Rows are processed concurrently
        with FIFO output ordering. Blocks when buffer is full (backpressure).

        Args:
            row: Row to process
            ctx: Plugin context with landscape and state_id

        Raises:
            RuntimeError: If connect_output() was not called
        """
        if not self._batch_initialized:
            raise RuntimeError("connect_output() must be called before accept()")

        self.accept_row(row, ctx, self._process_row)

    def process(
        self,
        row: dict[str, Any],
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

    def _process_row(self, row: dict[str, Any], ctx: PluginContext) -> TransformResult:
        """Process a single row through OpenRouter API.

        Called by worker threads from the BatchTransformMixin. Each row is
        processed independently with its own HTTP client.

        Error handling follows Three-Tier Trust Model:
        1. Template rendering (THEIR DATA) - wrap, return error
        2. HTTP API call (EXTERNAL) - wrap, return error
        3. Response parsing (EXTERNAL DATA) - wrap, return error

        Args:
            row: Row to process
            ctx: Plugin context with landscape and state_id

        Returns:
            TransformResult with processed row or error
        """
        # 1. Render template with row data (THEIR DATA - wrap)
        try:
            rendered = self._template.render_with_metadata(row)
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
        if self._max_tokens:
            request_body["max_tokens"] = self._max_tokens

        # 3. Get HTTP client (cached per state_id for call_index uniqueness)
        if ctx.state_id is None:
            raise RuntimeError("OpenRouter LLM transform requires state_id. Ensure transform is executed through the engine.")

        try:
            http_client = self._get_http_client(ctx.state_id)

            # 4. Call OpenRouter API (EXTERNAL - wrap)
            try:
                response = http_client.post(
                    "/chat/completions",
                    json=request_body,
                    headers={"Content-Type": "application/json"},
                )
                response.raise_for_status()
            except httpx.HTTPStatusError as e:
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

            # OpenRouter can return {"usage": null} or omit usage entirely.
            # Use `or {}` to handle both missing AND null cases.
            usage = data.get("usage") or {}

            # 7. Build output row (OUR CODE - let exceptions crash)
            output = dict(row)
            output[self._response_field] = content
            output[f"{self._response_field}_usage"] = usage
            output[f"{self._response_field}_template_hash"] = rendered.template_hash
            output[f"{self._response_field}_variables_hash"] = rendered.variables_hash
            output[f"{self._response_field}_template_source"] = rendered.template_source
            output[f"{self._response_field}_lookup_hash"] = rendered.lookup_hash
            output[f"{self._response_field}_lookup_source"] = rendered.lookup_source
            output[f"{self._response_field}_system_prompt_source"] = self._system_prompt_source
            output[f"{self._response_field}_model"] = data.get("model", self._model)

            return TransformResult.success(
                output,
                success_reason={"action": "enriched", "fields_added": [self._response_field]},
            )
        finally:
            # Clean up cached client for this state_id to prevent unbounded growth
            with self._http_clients_lock:
                self._http_clients.pop(ctx.state_id, None)

    def _get_http_client(self, state_id: str) -> AuditedHTTPClient:
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
                    headers={
                        "Authorization": f"Bearer {self._api_key}",
                        "HTTP-Referer": "https://github.com/elspeth-rapid",  # Required by OpenRouter
                    },
                    limiter=self._limiter,
                )
            return self._http_clients[state_id]

    def close(self) -> None:
        """Release resources."""
        # Shutdown batch processing infrastructure
        if self._batch_initialized:
            self.shutdown_batch_processing()

        self._recorder = None
        # Clear cached HTTP clients
        with self._http_clients_lock:
            self._http_clients.clear()
