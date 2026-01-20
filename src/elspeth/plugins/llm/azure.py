# src/elspeth/plugins/llm/azure.py
"""Azure OpenAI LLM transform - single call per row.

Self-contained transform that creates its own AuditedLLMClient using
the context's landscape and state_id. This ensures transforms can be
containerized without external client dependencies.
"""

from __future__ import annotations

from typing import Any, Self

from pydantic import Field, model_validator

from elspeth.contracts import Determinism, TransformResult
from elspeth.plugins.base import BaseTransform
from elspeth.plugins.clients.llm import AuditedLLMClient, LLMClientError, RateLimitError
from elspeth.plugins.context import PluginContext
from elspeth.plugins.llm.base import LLMConfig
from elspeth.plugins.llm.templates import PromptTemplate, TemplateError
from elspeth.plugins.schema_factory import create_schema_from_config


class AzureOpenAIConfig(LLMConfig):
    """Azure OpenAI-specific configuration.

    Extends LLMConfig with Azure-specific settings:
    - deployment_name: Azure deployment name (required) - used as model identifier
    - endpoint: Azure OpenAI endpoint URL (required)
    - api_key: Azure OpenAI API key (required)
    - api_version: Azure API version (default: 2024-10-21)

    Note: The 'model' field from LLMConfig is automatically set to
    deployment_name if not explicitly provided.
    """

    # Override model to make it optional - will default to deployment_name
    model: str = Field(
        default="", description="Model identifier (defaults to deployment_name)"
    )

    deployment_name: str = Field(..., description="Azure deployment name")
    endpoint: str = Field(..., description="Azure OpenAI endpoint URL")
    api_key: str = Field(..., description="Azure OpenAI API key")
    api_version: str = Field(default="2024-10-21", description="Azure API version")

    @model_validator(mode="after")
    def _set_model_from_deployment(self) -> Self:
        """Set model to deployment_name if not explicitly provided."""
        if not self.model:
            self.model = self.deployment_name
        return self


class AzureLLMTransform(BaseTransform):
    """LLM transform using Azure OpenAI.

    Self-contained transform that creates its own AuditedLLMClient
    internally using ctx.landscape and ctx.state_id. This ensures
    the transform can be containerized without external dependencies.

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
                fields: dynamic
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
        self._azure_api_key = cfg.api_key
        self._azure_api_version = cfg.api_version
        self._deployment_name = cfg.deployment_name

        # Store common LLM settings (from LLMConfig)
        self._model = cfg.model or cfg.deployment_name
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
        assert cfg.schema_config is not None
        schema = create_schema_from_config(
            cfg.schema_config,
            f"{self.name}Schema",
            allow_coercion=False,  # Transforms do NOT coerce
        )
        self.input_schema = schema
        self.output_schema = schema

    def process(self, row: dict[str, Any], ctx: PluginContext) -> TransformResult:
        """Process a row through Azure OpenAI.

        Creates its own AuditedLLMClient using ctx.landscape and ctx.state_id.

        Error handling follows Three-Tier Trust Model:
        1. Template rendering (THEIR DATA) - wrap, return error
        2. LLM call (EXTERNAL) - wrap, return error
        3. Internal logic (OUR CODE) - let crash

        Args:
            row: Input row matching input_schema
            ctx: Plugin context with landscape and state_id

        Returns:
            TransformResult with processed row or error
        """
        # 1. Render template with row data (THEIR DATA - wrap)
        try:
            rendered = self._template.render_with_metadata(row)
        except TemplateError as e:
            return TransformResult.error(
                {
                    "reason": "template_rendering_failed",
                    "error": str(e),
                    "template_hash": self._template.template_hash,
                }
            )

        # 2. Build messages
        messages: list[dict[str, str]] = []
        if self._system_prompt:
            messages.append({"role": "system", "content": self._system_prompt})
        messages.append({"role": "user", "content": rendered.prompt})

        # 3. Create audited LLM client (self-contained)
        if ctx.landscape is None or ctx.state_id is None:
            raise RuntimeError(
                "Azure LLM transform requires landscape recorder and state_id. "
                "Ensure transform is executed through the engine."
            )

        # Import here to avoid hard dependency on openai package
        from openai import AzureOpenAI

        underlying_client = AzureOpenAI(
            azure_endpoint=self._azure_endpoint,
            api_key=self._azure_api_key,
            api_version=self._azure_api_version,
        )

        llm_client = AuditedLLMClient(
            recorder=ctx.landscape,
            state_id=ctx.state_id,
            underlying_client=underlying_client,
            provider="azure",
        )

        # 4. Call LLM (EXTERNAL - wrap)
        try:
            response = llm_client.chat_completion(
                model=self._model,
                messages=messages,
                temperature=self._temperature,
                max_tokens=self._max_tokens,
            )
        except RateLimitError as e:
            return TransformResult.error(
                {"reason": "rate_limited", "error": str(e)},
                retryable=True,
            )
        except LLMClientError as e:
            return TransformResult.error(
                {"reason": "llm_call_failed", "error": str(e)},
                retryable=e.retryable,
            )

        # 5. Build output row (OUR CODE - let exceptions crash)
        output = dict(row)
        output[self._response_field] = response.content
        output[f"{self._response_field}_usage"] = response.usage
        output[f"{self._response_field}_template_hash"] = rendered.template_hash
        output[f"{self._response_field}_variables_hash"] = rendered.variables_hash
        output[f"{self._response_field}_template_source"] = rendered.template_source
        output[f"{self._response_field}_lookup_hash"] = rendered.lookup_hash
        output[f"{self._response_field}_lookup_source"] = rendered.lookup_source
        output[f"{self._response_field}_model"] = response.model

        return TransformResult.success(output)

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
        """Release resources."""
        pass
