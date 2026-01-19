# src/elspeth/plugins/llm/azure.py
"""Azure OpenAI LLM transform - single call per row."""

from __future__ import annotations

from typing import Any, Self

from pydantic import Field, model_validator

from elspeth.plugins.llm.base import BaseLLMTransform, LLMConfig


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


class AzureLLMTransform(BaseLLMTransform):
    """LLM transform using Azure OpenAI.

    Inherits from BaseLLMTransform - uses ctx.llm_client for calls.
    The executor configures the audited client with Azure credentials.

    Unlike OpenRouterLLMTransform (which uses HTTP directly), Azure
    uses the OpenAI SDK. The executor reads `azure_config` property
    to create an AuditedLLMClient with the correct Azure credentials.

    Configuration example:
        transforms:
          - plugin: azure_llm
            options:
              deployment_name: "my-gpt4o-deployment"
              endpoint: "${AZURE_OPENAI_ENDPOINT}"
              api_key: "${AZURE_OPENAI_KEY}"
              template: |
                Analyze: {{ text }}
              schema:
                fields: dynamic
    """

    name = "azure_llm"

    def __init__(self, config: dict[str, Any]) -> None:
        """Initialize Azure LLM transform.

        Parses Azure-specific configuration and prepares config for
        BaseLLMTransform by setting model to deployment_name.

        Args:
            config: Transform configuration dictionary
        """
        # Parse Azure-specific config to validate all required fields
        cfg = AzureOpenAIConfig.from_dict(config)

        # Store Azure-specific config for executor to use
        self._azure_endpoint = cfg.endpoint
        self._azure_api_key = cfg.api_key
        self._azure_api_version = cfg.api_version
        self._deployment_name = cfg.deployment_name

        # Build config for BaseLLMTransform - strip Azure-specific fields
        # since LLMConfig uses extra="forbid"
        base_config = dict(config)
        base_config["model"] = cfg.deployment_name
        # Remove Azure-specific fields that LLMConfig doesn't know about
        for azure_field in ("deployment_name", "endpoint", "api_key", "api_version"):
            base_config.pop(azure_field, None)

        super().__init__(base_config)

    @property
    def azure_config(self) -> dict[str, Any]:
        """Azure configuration for executor to create audited client.

        The executor reads this to create an AuditedLLMClient with
        the correct Azure credentials via openai.AzureOpenAI().

        Returns:
            Dict containing endpoint, api_key, api_version, and provider
        """
        return {
            "endpoint": self._azure_endpoint,
            "api_key": self._azure_api_key,
            "api_version": self._azure_api_version,
            "provider": "azure",
        }

    @property
    def deployment_name(self) -> str:
        """Azure deployment name (used as model in API calls)."""
        return self._deployment_name
