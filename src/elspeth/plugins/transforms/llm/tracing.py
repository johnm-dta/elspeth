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
      - plugin: llm
        options:
          provider: azure
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

    def __post_init__(self) -> None:
        if self.provider != "azure_ai":
            raise ValueError(f"AzureAITracingConfig requires provider='azure_ai', got {self.provider!r}")
        if self.connection_string is None:
            raise ValueError("AzureAITracingConfig requires connection_string. Use ${APPLICATIONINSIGHTS_CONNECTION_STRING} in YAML.")


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
        host: Langfuse host URL (REQUIRED - operator must specify)
        tracing_enabled: Whether tracing is enabled (v3 parameter, default: True)
    """

    provider: str = "langfuse"
    public_key: str | None = None
    secret_key: str | None = None
    host: str | None = None
    tracing_enabled: bool = True

    def __post_init__(self) -> None:
        if self.provider != "langfuse":
            raise ValueError(f"LangfuseTracingConfig requires provider='langfuse', got {self.provider!r}")
        if self.public_key is None:
            raise ValueError("LangfuseTracingConfig requires public_key. Use ${LANGFUSE_PUBLIC_KEY} in YAML.")
        if self.secret_key is None:
            raise ValueError("LangfuseTracingConfig requires secret_key. Use ${LANGFUSE_SECRET_KEY} in YAML.")
        if self.host is None:
            raise ValueError("LangfuseTracingConfig requires host (e.g. 'https://cloud.langfuse.com' or your on-prem URL).")


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

    # "provider" absence is a valid shorthand for "no tracing" — the only
    # key where a .get() default is justified (it's a discriminator, not data).
    provider = config.get("provider", "none")

    # Pass remaining config fields through to the dataclass, which owns all
    # defaults and required-field validation via __post_init__.  This avoids
    # duplicating dataclass defaults in .get() calls that can silently diverge.
    fields = {k: v for k, v in config.items() if k != "provider"}

    match provider:
        case "azure_ai":
            return AzureAITracingConfig(**fields)
        case "langfuse":
            return LangfuseTracingConfig(**fields)
        case "none":
            return TracingConfig(provider="none")
        case _:
            raise ValueError(
                f"Unknown tracing provider '{provider}'. Supported providers: {', '.join(sorted(SUPPORTED_TRACING_PROVIDERS))}"
            )


def validate_tracing_config(config: TracingConfig) -> list[str]:
    """Validate tracing configuration completeness.

    Note: Required-field and discriminator checks are now enforced at
    construction time via ``__post_init__`` on each subclass.  This
    function is retained for callers that import it, but only checks
    provider validity (the one check not covered by construction).

    Args:
        config: Parsed tracing configuration

    Returns:
        List of validation error messages (empty if valid)
    """
    if config.provider not in SUPPORTED_TRACING_PROVIDERS:
        return [f"Unknown tracing provider '{config.provider}'. Supported providers: azure_ai, langfuse, none."]
    return []
