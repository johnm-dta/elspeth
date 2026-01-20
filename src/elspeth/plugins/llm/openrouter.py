# src/elspeth/plugins/llm/openrouter.py
"""OpenRouter LLM transform - access 100+ models via single API."""

from __future__ import annotations

from threading import Lock
from typing import TYPE_CHECKING, Any

import httpx
from pydantic import Field

from elspeth.contracts import Determinism, TransformResult
from elspeth.plugins.base import BaseTransform
from elspeth.plugins.clients.http import AuditedHTTPClient
from elspeth.plugins.context import PluginContext
from elspeth.plugins.llm.base import LLMConfig
from elspeth.plugins.llm.capacity_errors import CapacityError, is_capacity_error
from elspeth.plugins.llm.pooled_executor import PooledExecutor, RowContext
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


class OpenRouterLLMTransform(BaseTransform):
    """LLM transform using OpenRouter API.

    OpenRouter provides access to 100+ models via a unified API.
    Uses audited HTTP client for call recording.

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
    is_batch_aware = True  # Enable aggregation buffering

    # LLM transforms are non-deterministic by nature
    determinism: Determinism = Determinism.NON_DETERMINISTIC

    def __init__(self, config: dict[str, Any]) -> None:
        # Call BaseTransform.__init__ to store raw config
        super().__init__(config)

        # Parse OpenRouter-specific config (includes all LLMConfig fields)
        cfg = OpenRouterConfig.from_dict(config)

        # Store OpenRouter-specific settings
        self._api_key = cfg.api_key
        self._base_url = cfg.base_url
        self._timeout = cfg.timeout_seconds

        # Store common LLM settings (from LLMConfig)
        self._model = cfg.model
        self._template = PromptTemplate(
            cfg.template,
            template_source=cfg.template_source,
            lookup_data=cfg.lookup,
            lookup_source=cfg.lookup_source,
        )
        self._system_prompt = cfg.system_prompt
        self._temperature = cfg.temperature
        self._max_tokens = cfg.max_tokens
        self._response_field = cfg.response_field
        self._on_error = cfg.on_error

        # Schema from config
        # TransformDataConfig validates schema_config is not None
        assert cfg.schema_config is not None
        schema = create_schema_from_config(
            cfg.schema_config,
            f"{self.name}Schema",
            allow_coercion=False,  # Transforms do NOT coerce
        )
        self.input_schema = schema
        self.output_schema = schema

        # Recorder reference for pooled execution (set in on_start)
        self._recorder: LandscapeRecorder | None = None

        # Create pooled executor if pool_size > 1
        if cfg.pool_config is not None:
            self._executor: PooledExecutor | None = PooledExecutor(cfg.pool_config)
        else:
            self._executor = None

        # HTTP client cache for pooled execution - ensures call_index uniqueness across retries
        # Each state_id gets its own client with monotonically increasing call indices
        self._http_clients: dict[str, AuditedHTTPClient] = {}
        self._http_clients_lock = Lock()

    def on_start(self, ctx: PluginContext) -> None:
        """Capture recorder reference for pooled execution."""
        self._recorder = ctx.landscape

    def process(self, row: dict[str, Any], ctx: PluginContext) -> TransformResult:
        """Process row via OpenRouter API using audited HTTP client.

        Routes to pooled or sequential execution based on pool_size config.

        Error handling follows Three-Tier Trust Model:
        1. Template rendering (THEIR DATA) - wrap, return error
        2. HTTP API call (EXTERNAL) - wrap, return error
        3. Response parsing (OUR CODE) - let crash if malformed
        """
        # Route to pooled execution if configured
        if self._executor is not None:
            if ctx.landscape is None or ctx.state_id is None:
                raise RuntimeError(
                    "Pooled execution requires landscape recorder and state_id. Ensure transform is executed through the engine."
                )
            row_ctx = RowContext(row=row, state_id=ctx.state_id, row_index=0)
            try:
                results = self._executor.execute_batch(
                    contexts=[row_ctx],
                    process_fn=self._process_single_with_state,
                )
                return results[0]
            finally:
                # Evict cached client after row completes to prevent unbounded memory growth
                # The client is only needed during retry loops within execute_batch()
                with self._http_clients_lock:
                    self._http_clients.pop(ctx.state_id, None)

        # Sequential execution path
        # 1. Render template (THEIR DATA - wrap)
        try:
            rendered = self._template.render_with_metadata(row)
        except TemplateError as e:
            return TransformResult.error(
                {
                    "reason": "template_rendering_failed",
                    "error": str(e),
                    "template_hash": self._template.template_hash,
                    "template_source": self._template.template_source,
                }
            )

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

        # 3. Call via audited HTTP client (EXTERNAL - wrap)
        # Create client using context's recorder and state_id
        if ctx.landscape is None or ctx.state_id is None:
            raise RuntimeError(
                "OpenRouter transform requires landscape recorder and state_id. Ensure transform is executed through the engine."
            )

        http_client = AuditedHTTPClient(
            recorder=ctx.landscape,
            state_id=ctx.state_id,
            timeout=self._timeout,
            base_url=self._base_url,
            headers={"Authorization": f"Bearer {self._api_key}"},
        )

        try:
            response = http_client.post(
                "/chat/completions",
                json=request_body,
                headers={"Content-Type": "application/json"},
            )
            response.raise_for_status()
        except httpx.HTTPStatusError as e:
            # HTTP error (4xx, 5xx) - check for capacity errors (429/503/529)
            # Use is_capacity_error() for consistency with pooled execution path
            return TransformResult.error(
                {"reason": "api_call_failed", "error": str(e)},
                retryable=is_capacity_error(e.response.status_code),
            )
        except httpx.RequestError as e:
            # Network/connection errors - not retryable by default
            return TransformResult.error(
                {"reason": "api_call_failed", "error": str(e)},
                retryable=False,
            )

        # 4. Parse JSON response (EXTERNAL DATA - wrap)
        # OpenRouter/proxy may return non-JSON (e.g., HTML error page) with HTTP 200
        try:
            data = response.json()
        except (ValueError, TypeError) as e:
            # JSONDecodeError is a subclass of ValueError
            return TransformResult.error(
                {
                    "reason": "invalid_json_response",
                    "error": f"Response is not valid JSON: {e}",
                    "content_type": response.headers.get("content-type", "unknown"),
                    "body_preview": response.text[:500] if response.text else None,
                },
                retryable=False,
            )

        # 5. Extract content from response (EXTERNAL DATA - wrap)
        # OpenRouter may return malformed responses: empty choices, error JSON
        # with HTTP 200, or unexpected structure from various providers
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

        usage = data.get("usage", {})

        output = dict(row)
        output[self._response_field] = content
        output[f"{self._response_field}_usage"] = usage
        output[f"{self._response_field}_template_hash"] = rendered.template_hash
        output[f"{self._response_field}_variables_hash"] = rendered.variables_hash
        output[f"{self._response_field}_template_source"] = rendered.template_source
        output[f"{self._response_field}_lookup_hash"] = rendered.lookup_hash
        output[f"{self._response_field}_lookup_source"] = rendered.lookup_source
        output[f"{self._response_field}_model"] = data.get("model", self._model)

        return TransformResult.success(output)

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
                    timeout=self._timeout,
                    base_url=self._base_url,
                    headers={"Authorization": f"Bearer {self._api_key}"},
                )
            return self._http_clients[state_id]

    def _process_single_with_state(self, row: dict[str, Any], state_id: str) -> TransformResult:
        """Process a single row via OpenRouter API with explicit state_id.

        This is used by the pooled executor where each row has its own state.
        Uses cached HTTP clients to ensure call_index uniqueness across retries.

        Args:
            row: The row data to process
            state_id: The state ID for audit trail recording

        Returns:
            TransformResult with processed row or error

        Raises:
            CapacityError: On 429/503/529 HTTP errors (for pooled retry)
        """
        # 1. Render template (THEIR DATA - wrap)
        try:
            rendered = self._template.render_with_metadata(row)
        except TemplateError as e:
            return TransformResult.error(
                {
                    "reason": "template_rendering_failed",
                    "error": str(e),
                    "template_hash": self._template.template_hash,
                    "template_source": self._template.template_source,
                }
            )

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

        # 3. Get cached HTTP client (preserves call_index across retries)
        http_client = self._get_http_client(state_id)

        try:
            response = http_client.post(
                "/chat/completions",
                json=request_body,
                headers={"Content-Type": "application/json"},
            )
            response.raise_for_status()
        except httpx.HTTPStatusError as e:
            # Check for capacity error
            if is_capacity_error(e.response.status_code):
                raise CapacityError(e.response.status_code, str(e)) from e
            # Non-capacity HTTP error
            return TransformResult.error(
                {"reason": "api_call_failed", "error": str(e)},
                retryable=False,
            )
        except httpx.RequestError as e:
            return TransformResult.error(
                {"reason": "api_call_failed", "error": str(e)},
                retryable=False,
            )

        # 4. Parse JSON response (EXTERNAL DATA - wrap)
        try:
            data = response.json()
        except (ValueError, TypeError) as e:
            return TransformResult.error(
                {
                    "reason": "invalid_json_response",
                    "error": f"Response is not valid JSON: {e}",
                    "content_type": response.headers.get("content-type", "unknown"),
                    "body_preview": response.text[:500] if response.text else None,
                },
                retryable=False,
            )

        # 5. Extract content
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

        usage = data.get("usage", {})

        output = dict(row)
        output[self._response_field] = content
        output[f"{self._response_field}_usage"] = usage
        output[f"{self._response_field}_template_hash"] = rendered.template_hash
        output[f"{self._response_field}_variables_hash"] = rendered.variables_hash
        output[f"{self._response_field}_template_source"] = rendered.template_source
        output[f"{self._response_field}_lookup_hash"] = rendered.lookup_hash
        output[f"{self._response_field}_lookup_source"] = rendered.lookup_source
        output[f"{self._response_field}_model"] = data.get("model", self._model)

        return TransformResult.success(output)

    def close(self) -> None:
        """Release resources."""
        if self._executor is not None:
            self._executor.shutdown(wait=True)
        self._recorder = None
        # Clear cached HTTP clients
        with self._http_clients_lock:
            self._http_clients.clear()
