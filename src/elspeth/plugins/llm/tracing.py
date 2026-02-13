# src/elspeth/plugins/llm/tracing.py
"""Tier 2 tracing configuration models for LLM plugins.

This module provides configuration dataclasses for plugin-internal tracing.
Each plugin that wants Tier 2 telemetry uses these to parse its tracing config.

Design Philosophy:
    The framework provides NOTHING for plugin-internal tracing. Plugins are
    autonomous - they bring their own SDK dependencies and configure their
    own observability. This module provides CONVENTIONS, not enforcement.

Supported Providers:
    - azure_ai: Azure Monitor / Application Insights (auto-instruments OpenAI SDK)
    - langfuse: Langfuse LLM observability platform (manual spans)
    - none: No tracing (default)

Example YAML Configuration:
    transforms:
      - plugin: azure_llm
        options:
          deployment_name: gpt-4
          endpoint: ${AZURE_OPENAI_ENDPOINT}
          api_key: ${AZURE_OPENAI_KEY}

          # Tier 2: Plugin-specific tracing (optional)
          # Use environment variables for secrets!
          tracing:
            provider: azure_ai
            connection_string: ${APPLICATIONINSIGHTS_CONNECTION_STRING}
            enable_content_recording: true
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

SUPPORTED_TRACING_PROVIDERS = frozenset({"none", "azure_ai", "langfuse"})


@dataclass(frozen=True, slots=True)
class TracingConfig:
    """Base tracing configuration.

    Attributes:
        provider: Tracing provider identifier ('azure_ai', 'langfuse', 'none')
    """

    provider: str = "none"


@dataclass(frozen=True, slots=True)
class AzureAITracingConfig(TracingConfig):
    """Azure AI / Application Insights tracing configuration.

    Azure Monitor OpenTelemetry auto-instruments the OpenAI SDK, capturing:
    - Full prompts and responses (if enable_content_recording=True)
    - Token usage metrics
    - Latency and error rates
    - Custom dimensions for filtering

    WARNING: Azure Monitor is process-level. If multiple plugins configure
    azure_ai tracing, the first one to initialize wins.

    Attributes:
        provider: Always 'azure_ai'
        connection_string: Application Insights connection string
            (REQUIRED - use ${APPLICATIONINSIGHTS_CONNECTION_STRING} in YAML)
        enable_content_recording: Whether to capture full prompts/responses
            (default: True - enables prompt debugging)
        enable_live_metrics: Whether to enable Live Metrics Stream
            (default: False - adds overhead)
    """

    provider: str = "azure_ai"
    connection_string: str | None = None
    enable_content_recording: bool = True
    enable_live_metrics: bool = False


@dataclass(frozen=True, slots=True)
class LangfuseTracingConfig(TracingConfig):
    """Langfuse tracing configuration.

    Langfuse provides LLM-specific observability:
    - Prompt engineering analytics
    - Cost tracking and attribution
    - Evaluation scores and feedback
    - A/B testing for prompts

    Langfuse uses per-instance clients, so multiple plugins can have
    different Langfuse configurations (e.g., different hosts).

    Attributes:
        provider: Always 'langfuse'
        public_key: Langfuse public API key (REQUIRED - use ${LANGFUSE_PUBLIC_KEY})
        secret_key: Langfuse secret API key (REQUIRED - use ${LANGFUSE_SECRET_KEY})
        host: Langfuse host URL (default: cloud.langfuse.com)
        tracing_enabled: Whether tracing is enabled (v3 parameter, default: True)
    """

    provider: str = "langfuse"
    public_key: str | None = None
    secret_key: str | None = None
    host: str = "https://cloud.langfuse.com"
    tracing_enabled: bool = True


def parse_tracing_config(config: dict[str, Any] | None) -> TracingConfig | None:
    """Parse tracing configuration from dict.

    Args:
        config: Tracing configuration dict from plugin options,
            or None if tracing is not configured.

    Returns:
        Appropriate TracingConfig subclass based on provider,
        or None if config is None.

    Example:
        >>> config = {"provider": "langfuse", "host": "https://my.langfuse.com"}
        >>> result = parse_tracing_config(config)
        >>> isinstance(result, LangfuseTracingConfig)
        True
    """
    if config is None:
        return None

    provider = config.get("provider", "none")

    match provider:
        case "azure_ai":
            return AzureAITracingConfig(
                connection_string=config.get("connection_string"),
                enable_content_recording=config.get("enable_content_recording", True),
                enable_live_metrics=config.get("enable_live_metrics", False),
            )
        case "langfuse":
            return LangfuseTracingConfig(
                public_key=config.get("public_key"),
                secret_key=config.get("secret_key"),
                host=config.get("host", "https://cloud.langfuse.com"),
                tracing_enabled=config.get("tracing_enabled", True),
            )
        case _:
            return TracingConfig(provider=provider)


def validate_tracing_config(config: TracingConfig) -> list[str]:
    """Validate tracing configuration completeness.

    Args:
        config: Parsed tracing configuration

    Returns:
        List of validation error messages (empty if valid)
    """
    errors: list[str] = []

    if config.provider not in SUPPORTED_TRACING_PROVIDERS:
        errors.append(f"Unknown tracing provider '{config.provider}'. Supported providers: azure_ai, langfuse, none.")
        return errors

    if isinstance(config, AzureAITracingConfig):
        if config.connection_string is None:
            errors.append("azure_ai tracing requires connection_string. Use ${APPLICATIONINSIGHTS_CONNECTION_STRING} in YAML.")

    elif isinstance(config, LangfuseTracingConfig):
        if config.public_key is None:
            errors.append("langfuse tracing requires public_key. Use ${LANGFUSE_PUBLIC_KEY} in YAML.")
        if config.secret_key is None:
            errors.append("langfuse tracing requires secret_key. Use ${LANGFUSE_SECRET_KEY} in YAML.")

    return errors
