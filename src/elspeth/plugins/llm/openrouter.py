# src/elspeth/plugins/llm/openrouter.py
"""OpenRouter LLM transform - access 100+ models via single API."""

from __future__ import annotations

from typing import Any

import httpx
from pydantic import Field

from elspeth.contracts import Determinism, TransformResult
from elspeth.plugins.base import BaseTransform
from elspeth.plugins.context import PluginContext
from elspeth.plugins.llm.base import LLMConfig
from elspeth.plugins.llm.templates import PromptTemplate, TemplateError
from elspeth.plugins.schema_factory import create_schema_from_config


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
                Analyze: {{ text }}
              api_key: "${OPENROUTER_API_KEY}"
              schema:
                fields: dynamic
    """

    name = "openrouter_llm"

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
        self._template = PromptTemplate(cfg.template)
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

    def process(self, row: dict[str, Any], ctx: PluginContext) -> TransformResult:
        """Process row via OpenRouter API using audited HTTP client.

        Error handling follows Three-Tier Trust Model:
        1. Template rendering (THEIR DATA) - wrap, return error
        2. HTTP API call (EXTERNAL) - wrap, return error
        3. Response parsing (OUR CODE) - let crash if malformed
        """
        # 1. Render template (THEIR DATA - wrap)
        try:
            rendered = self._template.render_with_metadata(**row)
        except TemplateError as e:
            return TransformResult.error(
                {
                    "reason": "template_rendering_failed",
                    "error": str(e),
                    "template_hash": self._template.template_hash,
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
        if ctx.http_client is None:
            # This is OUR BUG - executor should have provided client
            raise RuntimeError("HTTP client not available in PluginContext")

        try:
            response = ctx.http_client.post(
                f"{self._base_url}/chat/completions",
                json=request_body,
                headers={
                    "Authorization": f"Bearer {self._api_key}",
                    "Content-Type": "application/json",
                },
                timeout=self._timeout,
            )
            response.raise_for_status()
            data = response.json()

        except httpx.HTTPStatusError as e:
            # HTTP error (4xx, 5xx) - check for rate limit
            is_rate_limit = e.response.status_code == 429
            return TransformResult.error(
                {"reason": "api_call_failed", "error": str(e)},
                retryable=is_rate_limit,
            )
        except httpx.RequestError as e:
            # Network/connection errors - not retryable by default
            return TransformResult.error(
                {"reason": "api_call_failed", "error": str(e)},
                retryable=False,
            )

        # 4. Extract content from response (EXTERNAL DATA - wrap)
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
                    "response_keys": list(data.keys())
                    if isinstance(data, dict)
                    else None,
                },
                retryable=False,
            )

        usage = data.get("usage", {})

        output = dict(row)
        output[self._response_field] = content
        output[f"{self._response_field}_usage"] = usage
        output[f"{self._response_field}_template_hash"] = rendered.template_hash
        output[f"{self._response_field}_variables_hash"] = rendered.variables_hash
        output[f"{self._response_field}_model"] = data.get("model", self._model)

        return TransformResult.success(output)

    def close(self) -> None:
        """Release resources."""
        pass
