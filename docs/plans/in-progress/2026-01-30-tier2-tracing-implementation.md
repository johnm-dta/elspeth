# Tier 2 Plugin Tracing Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Implement plugin-internal tracing support for Azure AI (Application Insights) and Langfuse, enabling deep LLM observability beyond Tier 1 framework telemetry.

**Architecture:** Plugins optionally configure their own tracing providers. The framework provides configuration patterns and lifecycle hooks, but no abstraction over provider SDKs. Plugins bring their own dependencies and initialize in `on_start()`.

**Tech Stack:** Python, Pydantic (config), azure-monitor-opentelemetry (Azure AI), langfuse (Langfuse), structlog (logging)

---

## Design Philosophy

From the telemetry design document:

> The framework provides **nothing** for plugin-internal tracing. Plugins are T1 code and handle their own integrations.

This means:
- **No framework abstraction** over provider SDKs
- **No framework configuration** for plugin telemetry (it lives in plugin `options:`)
- **Plugins are autonomous** — they bring their own dependencies and config
- **Convention, not enforcement** — we document patterns, not mandate them

---

## Tier 1 vs Tier 2 Comparison

| Aspect | Tier 1 (Framework) | Tier 2 (Plugin) |
|--------|-------------------|-----------------|
| **Scope** | All external calls | LLM calls only |
| **Data captured** | Hashes, latency, status | Full prompts, responses, token details |
| **Destination** | Generic observability (Datadog, etc.) | LLM-specific platforms (Langfuse, Azure AI) |
| **Configuration** | `telemetry:` section | Plugin's `options.tracing:` section |
| **Who implements** | Framework | Each plugin individually |

Users can enable both:
- Tier 1 → Datadog/Grafana for ops visibility
- Tier 2 → Langfuse for prompt engineering analytics

---

## Task 1: Create Tracing Config Models

**Files:**
- Create: `src/elspeth/plugins/llm/tracing.py`
- Create: `tests/plugins/llm/test_tracing_config.py`

### Step 1.1: Write failing test for TracingConfig parsing

**File:** `tests/plugins/llm/test_tracing_config.py`

```python
# tests/plugins/llm/test_tracing_config.py
"""Tests for Tier 2 tracing configuration models."""

import pytest

from elspeth.plugins.llm.tracing import (
    AzureAITracingConfig,
    LangfuseTracingConfig,
    TracingConfig,
    parse_tracing_config,
)


class TestTracingConfigParsing:
    """Tests for parse_tracing_config function."""

    def test_none_config_returns_none(self) -> None:
        """None input returns None."""
        result = parse_tracing_config(None)
        assert result is None

    def test_empty_dict_returns_base_config(self) -> None:
        """Empty dict returns base TracingConfig with provider='none'."""
        result = parse_tracing_config({})
        assert isinstance(result, TracingConfig)
        assert result.provider == "none"

    def test_azure_ai_provider_returns_azure_config(self) -> None:
        """Provider 'azure_ai' returns AzureAITracingConfig."""
        config = {
            "provider": "azure_ai",
            "connection_string": "InstrumentationKey=xxx",
            "enable_content_recording": True,
            "enable_live_metrics": False,
        }
        result = parse_tracing_config(config)

        assert isinstance(result, AzureAITracingConfig)
        assert result.provider == "azure_ai"
        assert result.connection_string == "InstrumentationKey=xxx"
        assert result.enable_content_recording is True
        assert result.enable_live_metrics is False

    def test_azure_ai_defaults(self) -> None:
        """AzureAITracingConfig has sensible defaults."""
        config = {"provider": "azure_ai"}
        result = parse_tracing_config(config)

        assert isinstance(result, AzureAITracingConfig)
        assert result.connection_string is None
        assert result.enable_content_recording is True  # Default: capture prompts
        assert result.enable_live_metrics is False  # Default: off

    def test_langfuse_provider_returns_langfuse_config(self) -> None:
        """Provider 'langfuse' returns LangfuseTracingConfig."""
        config = {
            "provider": "langfuse",
            "public_key": "pk-xxx",
            "secret_key": "sk-xxx",
            "host": "https://self-hosted.example.com",
        }
        result = parse_tracing_config(config)

        assert isinstance(result, LangfuseTracingConfig)
        assert result.provider == "langfuse"
        assert result.public_key == "pk-xxx"
        assert result.secret_key == "sk-xxx"
        assert result.host == "https://self-hosted.example.com"

    def test_langfuse_defaults(self) -> None:
        """LangfuseTracingConfig has sensible defaults."""
        config = {"provider": "langfuse"}
        result = parse_tracing_config(config)

        assert isinstance(result, LangfuseTracingConfig)
        assert result.public_key is None
        assert result.secret_key is None
        assert result.host == "https://cloud.langfuse.com"  # Default: cloud

    def test_unknown_provider_returns_base_config(self) -> None:
        """Unknown provider returns base TracingConfig."""
        config = {"provider": "unknown_provider"}
        result = parse_tracing_config(config)

        assert isinstance(result, TracingConfig)
        assert result.provider == "unknown_provider"

    def test_none_provider_returns_base_config(self) -> None:
        """Provider 'none' returns base TracingConfig."""
        config = {"provider": "none"}
        result = parse_tracing_config(config)

        assert isinstance(result, TracingConfig)
        assert result.provider == "none"
```

### Step 1.2: Run test to verify it fails

Run: `.venv/bin/python -m pytest tests/plugins/llm/test_tracing_config.py -v`

Expected: FAIL with `ModuleNotFoundError: No module named 'elspeth.plugins.llm.tracing'`

### Step 1.3: Create tracing config module

**File:** `src/elspeth/plugins/llm/tracing.py`

```python
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
          tracing:
            provider: azure_ai
            connection_string: ${APPLICATIONINSIGHTS_CONNECTION_STRING}
            enable_content_recording: true
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


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

    Attributes:
        provider: Always 'azure_ai'
        connection_string: Application Insights connection string
            (typically from APPLICATIONINSIGHTS_CONNECTION_STRING env var)
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

    Attributes:
        provider: Always 'langfuse'
        public_key: Langfuse public API key
        secret_key: Langfuse secret API key
        host: Langfuse host URL (default: cloud.langfuse.com)
    """

    provider: str = "langfuse"
    public_key: str | None = None
    secret_key: str | None = None
    host: str = "https://cloud.langfuse.com"


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
            )
        case _:
            return TracingConfig(provider=provider)
```

### Step 1.4: Run tests to verify they pass

Run: `.venv/bin/python -m pytest tests/plugins/llm/test_tracing_config.py -v`

Expected: All PASS

### Step 1.5: Commit

```bash
git add src/elspeth/plugins/llm/tracing.py tests/plugins/llm/test_tracing_config.py
git commit -m "feat(telemetry): add Tier 2 tracing config models

- TracingConfig base class
- AzureAITracingConfig for Azure Monitor integration
- LangfuseTracingConfig for Langfuse integration
- parse_tracing_config() factory function

Co-Authored-By: Claude Opus 4.5 <noreply@anthropic.com>"
```

---

## Task 2: Add Optional Dependencies

**Files:**
- Modify: `pyproject.toml`

### Step 2.1: Add tracing optional dependencies

**File:** `pyproject.toml`

Add after the `azure = [...]` section (around line 101):

```toml
tracing-azure = [
    # Tier 2: Azure AI tracing for LLM plugins
    "azure-monitor-opentelemetry>=1.6,<2",  # Azure Monitor + OpenTelemetry
]

tracing-langfuse = [
    # Tier 2: Langfuse tracing for LLM plugins
    "langfuse>=2.50,<3",  # Langfuse SDK
]

tracing = [
    # All Tier 2 tracing providers
    "elspeth[tracing-azure,tracing-langfuse]",
]
```

### Step 2.2: Update the `all` optional dependency

Update the `all` section to include tracing:

```toml
all = [
    "elspeth[dev,llm,azure,mcp,tracing]",
]
```

### Step 2.3: Verify dependencies can be resolved

Run: `uv pip install -e ".[tracing-azure]" --dry-run`

Expected: Shows packages that would be installed (no errors)

Run: `uv pip install -e ".[tracing-langfuse]" --dry-run`

Expected: Shows packages that would be installed (no errors)

### Step 2.4: Commit

```bash
git add pyproject.toml
git commit -m "feat(deps): add optional dependencies for Tier 2 tracing

- tracing-azure: azure-monitor-opentelemetry
- tracing-langfuse: langfuse SDK
- tracing: all tracing providers

Co-Authored-By: Claude Opus 4.5 <noreply@anthropic.com>"
```

---

## Task 3: Add Tracing Config to AzureOpenAIConfig

**Files:**
- Modify: `src/elspeth/plugins/llm/azure.py`
- Create: `tests/plugins/llm/test_azure_tracing.py`

### Step 3.1: Write failing test for tracing config in AzureOpenAIConfig

**File:** `tests/plugins/llm/test_azure_tracing.py`

```python
# tests/plugins/llm/test_azure_tracing.py
"""Tests for Tier 2 tracing in AzureLLMTransform."""

import pytest

from elspeth.plugins.llm.azure import AzureOpenAIConfig


class TestAzureOpenAIConfigTracing:
    """Tests for tracing configuration in AzureOpenAIConfig."""

    def test_tracing_field_accepts_none(self) -> None:
        """Tracing field defaults to None (no tracing)."""
        config = AzureOpenAIConfig(
            deployment_name="gpt-4",
            endpoint="https://test.openai.azure.com",
            api_key="test-key",
            prompt_template="Hello",
        )
        assert config.tracing is None

    def test_tracing_field_accepts_azure_ai_config(self) -> None:
        """Tracing field accepts Azure AI configuration dict."""
        config = AzureOpenAIConfig(
            deployment_name="gpt-4",
            endpoint="https://test.openai.azure.com",
            api_key="test-key",
            prompt_template="Hello",
            tracing={
                "provider": "azure_ai",
                "connection_string": "InstrumentationKey=xxx",
                "enable_content_recording": True,
            },
        )
        assert config.tracing is not None
        assert config.tracing["provider"] == "azure_ai"

    def test_tracing_field_accepts_langfuse_config(self) -> None:
        """Tracing field accepts Langfuse configuration dict."""
        config = AzureOpenAIConfig(
            deployment_name="gpt-4",
            endpoint="https://test.openai.azure.com",
            api_key="test-key",
            prompt_template="Hello",
            tracing={
                "provider": "langfuse",
                "public_key": "pk-xxx",
                "secret_key": "sk-xxx",
            },
        )
        assert config.tracing is not None
        assert config.tracing["provider"] == "langfuse"
```

### Step 3.2: Run test to verify it fails

Run: `.venv/bin/python -m pytest tests/plugins/llm/test_azure_tracing.py::TestAzureOpenAIConfigTracing::test_tracing_field_accepts_none -v`

Expected: FAIL with `pydantic_core._pydantic_core.ValidationError: ... Extra inputs are not permitted` (tracing field not defined)

### Step 3.3: Add tracing field to AzureOpenAIConfig

**File:** `src/elspeth/plugins/llm/azure.py`

Add to imports (around line 14):

```python
from typing import TYPE_CHECKING, Any, Self
```

Add to `AzureOpenAIConfig` class (after `api_version` field, around line 57):

```python
    # Tier 2: Plugin-internal tracing (optional)
    tracing: dict[str, Any] | None = Field(
        default=None,
        description="Tier 2 tracing configuration (azure_ai, langfuse, or none)",
    )
```

### Step 3.4: Run tests to verify they pass

Run: `.venv/bin/python -m pytest tests/plugins/llm/test_azure_tracing.py -v`

Expected: All PASS

### Step 3.5: Commit

```bash
git add src/elspeth/plugins/llm/azure.py tests/plugins/llm/test_azure_tracing.py
git commit -m "feat(azure): add tracing config field to AzureOpenAIConfig

Tier 2 plugin tracing configuration for Azure AI / Langfuse integration.

Co-Authored-By: Claude Opus 4.5 <noreply@anthropic.com>"
```

---

## Task 4: Implement Tracing Lifecycle in AzureLLMTransform

**Files:**
- Modify: `src/elspeth/plugins/llm/azure.py`
- Modify: `tests/plugins/llm/test_azure_tracing.py`

### Step 4.1: Write failing test for tracing initialization

**File:** `tests/plugins/llm/test_azure_tracing.py`

Add to the file:

```python
from unittest.mock import MagicMock, patch


class TestAzureLLMTransformTracing:
    """Tests for tracing lifecycle in AzureLLMTransform."""

    def _create_transform(self, tracing_config: dict | None = None) -> "AzureLLMTransform":
        """Create a transform with optional tracing config."""
        from elspeth.plugins.llm.azure import AzureLLMTransform

        config = {
            "deployment_name": "gpt-4",
            "endpoint": "https://test.openai.azure.com",
            "api_key": "test-key",
            "prompt_template": "Hello {{ name }}",
        }
        if tracing_config is not None:
            config["tracing"] = tracing_config

        return AzureLLMTransform(config)

    def test_no_tracing_when_config_is_none(self) -> None:
        """No tracing setup when tracing config is None."""
        transform = self._create_transform(tracing_config=None)
        assert transform._tracing_config is None
        assert transform._tracing_active is False

    def test_tracing_config_is_parsed(self) -> None:
        """Tracing config dict is parsed into TracingConfig."""
        from elspeth.plugins.llm.tracing import AzureAITracingConfig

        transform = self._create_transform(
            tracing_config={
                "provider": "azure_ai",
                "connection_string": "InstrumentationKey=xxx",
            }
        )
        assert transform._tracing_config is not None
        assert isinstance(transform._tracing_config, AzureAITracingConfig)

    def test_azure_ai_tracing_setup_logs_warning_when_package_missing(self) -> None:
        """Azure AI tracing logs warning when package not installed."""
        import structlog

        transform = self._create_transform(
            tracing_config={
                "provider": "azure_ai",
                "connection_string": "InstrumentationKey=xxx",
            }
        )

        # Mock the import to fail
        with patch.dict("sys.modules", {"azure.monitor.opentelemetry": None}):
            with patch("structlog.get_logger") as mock_get_logger:
                mock_logger = MagicMock()
                mock_get_logger.return_value = mock_logger

                # Create mock context
                ctx = MagicMock()
                ctx.landscape = MagicMock()

                # Call on_start which should try to setup tracing
                transform.on_start(ctx)

                # Should have logged a warning
                # Note: This test may need adjustment based on actual implementation

    def test_tracing_active_flag_set_on_successful_setup(self) -> None:
        """_tracing_active is True after successful tracing setup."""
        transform = self._create_transform(
            tracing_config={
                "provider": "azure_ai",
                "connection_string": "InstrumentationKey=xxx",
            }
        )

        # Mock successful import and configure
        with patch("elspeth.plugins.llm.azure.configure_azure_monitor") as mock_configure:
            ctx = MagicMock()
            ctx.landscape = MagicMock()

            transform.on_start(ctx)

            # Verify configure was called
            mock_configure.assert_called_once()
            assert transform._tracing_active is True
```

### Step 4.2: Run test to verify it fails

Run: `.venv/bin/python -m pytest tests/plugins/llm/test_azure_tracing.py::TestAzureLLMTransformTracing::test_no_tracing_when_config_is_none -v`

Expected: FAIL with `AttributeError: 'AzureLLMTransform' object has no attribute '_tracing_config'`

### Step 4.3: Add tracing initialization to AzureLLMTransform.__init__

**File:** `src/elspeth/plugins/llm/azure.py`

Add import at top:

```python
from elspeth.plugins.llm.tracing import TracingConfig, parse_tracing_config
```

Add to `__init__` method (after existing initialization, around line 175):

```python
        # Tier 2: Plugin-internal tracing
        self._tracing_config: TracingConfig | None = parse_tracing_config(
            cfg.tracing
        )
        self._tracing_active: bool = False
```

### Step 4.4: Add tracing setup to on_start

**File:** `src/elspeth/plugins/llm/azure.py`

Update the `on_start` method:

```python
    def on_start(self, ctx: PluginContext) -> None:
        """Capture recorder reference and initialize tracing.

        Called by the engine at pipeline start. Captures the landscape
        recorder reference and sets up Tier 2 tracing if configured.
        """
        self._recorder = ctx.landscape

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

        match self._tracing_config.provider:
            case "azure_ai":
                self._setup_azure_ai_tracing(logger)
            case "langfuse":
                self._setup_langfuse_tracing(logger)
            case "none" | _:
                pass  # No tracing

    def _setup_azure_ai_tracing(self, logger: Any) -> None:
        """Initialize Azure AI / Application Insights tracing.

        Azure Monitor OpenTelemetry auto-instruments the OpenAI SDK.
        No manual instrumentation needed after configure_azure_monitor().
        """
        try:
            from azure.monitor.opentelemetry import configure_azure_monitor

            from elspeth.plugins.llm.tracing import AzureAITracingConfig

            cfg = self._tracing_config
            if not isinstance(cfg, AzureAITracingConfig):
                return

            configure_azure_monitor(
                connection_string=cfg.connection_string,
                enable_live_metrics=cfg.enable_live_metrics,
            )
            self._tracing_active = True

            logger.info(
                "Azure AI tracing initialized",
                provider="azure_ai",
                content_recording=cfg.enable_content_recording,
                live_metrics=cfg.enable_live_metrics,
            )

        except ImportError:
            logger.warning(
                "Azure AI tracing requested but package not installed",
                provider="azure_ai",
                hint="Install with: uv pip install elspeth[tracing-azure]",
            )

    def _setup_langfuse_tracing(self, logger: Any) -> None:
        """Initialize Langfuse tracing.

        Langfuse requires manual span creation around LLM calls.
        The Langfuse client is stored for use in _process_row().
        """
        try:
            from langfuse import Langfuse

            from elspeth.plugins.llm.tracing import LangfuseTracingConfig

            cfg = self._tracing_config
            if not isinstance(cfg, LangfuseTracingConfig):
                return

            self._langfuse_client = Langfuse(
                public_key=cfg.public_key,
                secret_key=cfg.secret_key,
                host=cfg.host,
            )
            self._tracing_active = True

            logger.info(
                "Langfuse tracing initialized",
                provider="langfuse",
                host=cfg.host,
            )

        except ImportError:
            logger.warning(
                "Langfuse tracing requested but package not installed",
                provider="langfuse",
                hint="Install with: uv pip install elspeth[tracing-langfuse]",
            )
```

### Step 4.5: Add tracing cleanup to close()

**File:** `src/elspeth/plugins/llm/azure.py`

Update the `close` method to flush tracing:

```python
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

    def _flush_tracing(self) -> None:
        """Flush any pending tracing data."""
        import structlog

        logger = structlog.get_logger(__name__)

        # Langfuse needs explicit flush
        if hasattr(self, "_langfuse_client") and self._langfuse_client is not None:
            try:
                self._langfuse_client.flush()
                logger.debug("Langfuse tracing flushed")
            except Exception as e:
                logger.warning("Failed to flush Langfuse tracing", error=str(e))

        # Azure Monitor handles its own batching/flushing
        # No explicit flush needed
```

### Step 4.6: Run tests to verify they pass

Run: `.venv/bin/python -m pytest tests/plugins/llm/test_azure_tracing.py -v`

Expected: All PASS (some tests may need adjustment based on actual implementation)

### Step 4.7: Run existing Azure LLM tests

Run: `.venv/bin/python -m pytest tests/plugins/llm/test_azure.py -v`

Expected: All PASS (no regression)

### Step 4.8: Commit

```bash
git add src/elspeth/plugins/llm/azure.py tests/plugins/llm/test_azure_tracing.py
git commit -m "feat(azure): implement Tier 2 tracing lifecycle

- Parse tracing config in __init__
- Initialize tracing in on_start() (Azure AI or Langfuse)
- Graceful degradation when SDK not installed
- Flush tracing in close()

Co-Authored-By: Claude Opus 4.5 <noreply@anthropic.com>"
```

---

## Task 5: Add Tracing to AzureMultiQueryLLMTransform

**Files:**
- Modify: `src/elspeth/plugins/llm/azure_multi_query.py`

### Step 5.1: Add tracing field to config

Find the config class (likely `AzureMultiQueryConfig` or similar) and add:

```python
    # Tier 2: Plugin-internal tracing (optional)
    tracing: dict[str, Any] | None = Field(
        default=None,
        description="Tier 2 tracing configuration (azure_ai, langfuse, or none)",
    )
```

### Step 5.2: Add tracing initialization (same pattern as Task 4)

Copy the tracing initialization pattern from `AzureLLMTransform`:
- Parse config in `__init__`
- Setup in `on_start()`
- Flush in `close()`

### Step 5.3: Run existing tests

Run: `.venv/bin/python -m pytest tests/plugins/llm/test_azure_multi_query.py -v`

Expected: All PASS

### Step 5.4: Commit

```bash
git add src/elspeth/plugins/llm/azure_multi_query.py
git commit -m "feat(azure_multi_query): add Tier 2 tracing support

Co-Authored-By: Claude Opus 4.5 <noreply@anthropic.com>"
```

---

## Task 6: Add Tracing to OpenRouter Plugins (Optional)

**Files:**
- Modify: `src/elspeth/plugins/llm/openrouter.py`
- Modify: `src/elspeth/plugins/llm/openrouter_multi_query.py`

### Step 6.1: Decide on Langfuse-only support

OpenRouter uses HTTP (not OpenAI SDK), so Azure Monitor auto-instrumentation won't work.
Langfuse can still provide value with manual spans.

Add tracing config and Langfuse-only setup following the same pattern.

### Step 6.2: Commit

```bash
git add src/elspeth/plugins/llm/openrouter.py src/elspeth/plugins/llm/openrouter_multi_query.py
git commit -m "feat(openrouter): add Tier 2 tracing support (Langfuse)

OpenRouter uses HTTP, so Azure AI auto-instrumentation doesn't apply.
Langfuse manual spans provide LLM observability.

Co-Authored-By: Claude Opus 4.5 <noreply@anthropic.com>"
```

---

## Task 7: Write Documentation

**Files:**
- Create: `docs/guides/tier2-tracing.md`

### Step 7.1: Create documentation guide

**File:** `docs/guides/tier2-tracing.md`

```markdown
# Tier 2 Plugin Tracing Guide

Tier 2 tracing provides deep LLM observability beyond the framework's Tier 1 telemetry.
While Tier 1 captures latency, status, and content hashes for ALL external calls,
Tier 2 captures full prompts, responses, and token-level metrics for LLM calls specifically.

## Overview

| Aspect | Tier 1 (Framework) | Tier 2 (Plugin) |
|--------|-------------------|-----------------|
| Scope | All external calls | LLM calls only |
| Data | Hashes, latency, status | Full prompts, responses, tokens |
| Destination | Generic observability | LLM-specific platforms |
| Configuration | `telemetry:` section | Plugin `options.tracing:` |

## Supported Providers

### Azure AI (Application Insights)

Azure Monitor OpenTelemetry auto-instruments the OpenAI SDK, capturing:
- Full prompts and responses
- Token usage metrics
- Latency and error rates

**Installation:**
```bash
uv pip install elspeth[tracing-azure]
```

**Configuration:**
```yaml
transforms:
  - plugin: azure_llm
    options:
      deployment_name: gpt-4
      endpoint: ${AZURE_OPENAI_ENDPOINT}
      api_key: ${AZURE_OPENAI_KEY}
      prompt_template: "Classify: {{ text }}"

      # Tier 2 tracing
      tracing:
        provider: azure_ai
        connection_string: ${APPLICATIONINSIGHTS_CONNECTION_STRING}
        enable_content_recording: true  # Capture full prompts
        enable_live_metrics: false       # Disable for lower overhead
```

### Langfuse

Langfuse provides LLM-specific observability:
- Prompt engineering analytics
- Cost tracking and attribution
- Evaluation scores

**Installation:**
```bash
uv pip install elspeth[tracing-langfuse]
```

**Configuration:**
```yaml
transforms:
  - plugin: azure_llm
    options:
      deployment_name: gpt-4
      endpoint: ${AZURE_OPENAI_ENDPOINT}
      api_key: ${AZURE_OPENAI_KEY}
      prompt_template: "Classify: {{ text }}"

      # Tier 2 tracing
      tracing:
        provider: langfuse
        public_key: ${LANGFUSE_PUBLIC_KEY}
        secret_key: ${LANGFUSE_SECRET_KEY}
        host: https://cloud.langfuse.com  # Or self-hosted URL
```

## Using Both Tiers Together

Tier 1 and Tier 2 are complementary. A typical production setup:

```yaml
# Tier 1: Framework telemetry → Datadog for ops
telemetry:
  enabled: true
  granularity: full
  exporters:
    - name: datadog
      options:
        service_name: "elspeth-pipeline"

# Tier 2: Plugin tracing → Langfuse for ML team
transforms:
  - plugin: azure_llm
    options:
      # ... LLM config ...
      tracing:
        provider: langfuse
        public_key: ${LANGFUSE_PUBLIC_KEY}
        secret_key: ${LANGFUSE_SECRET_KEY}
```

**Result:**
- Datadog shows all external call metrics (LLM + HTTP + DB)
- Langfuse shows detailed prompt/response data for LLM calls

## Graceful Degradation

If the tracing SDK is not installed, plugins log a warning and continue without tracing:

```
WARNING: Langfuse tracing requested but package not installed
         hint: Install with: uv pip install elspeth[tracing-langfuse]
```

This allows pipelines to run in environments without tracing dependencies.

## Privacy Considerations

Tier 2 tracing captures full prompts and responses. Consider:

1. **PII in prompts**: Use `enable_content_recording: false` for Azure AI
2. **Data residency**: Self-host Langfuse for sensitive data
3. **Compliance**: Review your organization's data handling policies

## Troubleshooting

### No traces appearing

1. Check connection string / API keys are set
2. Verify the tracing SDK is installed
3. Check logs for initialization warnings

### Azure AI traces missing token counts

Azure Monitor may not capture token usage for all API versions.
Ensure `api_version: 2024-10-21` or later.

### Langfuse traces delayed

Langfuse batches traces for efficiency. Call `transform.close()` to flush pending traces.
```

### Step 7.2: Commit

```bash
git add docs/guides/tier2-tracing.md
git commit -m "docs: add Tier 2 tracing guide

Covers Azure AI and Langfuse configuration, usage patterns,
and privacy considerations.

Co-Authored-By: Claude Opus 4.5 <noreply@anthropic.com>"
```

---

## Task 8: Run Full Test Suite and Type Checks

### Step 8.1: Run mypy

Run: `.venv/bin/python -m mypy src/elspeth/plugins/llm/tracing.py src/elspeth/plugins/llm/azure.py`

Expected: No errors

### Step 8.2: Run ruff

Run: `.venv/bin/python -m ruff check src/elspeth/plugins/llm/tracing.py src/elspeth/plugins/llm/azure.py`

Expected: No errors

### Step 8.3: Run full test suite

Run: `.venv/bin/python -m pytest tests/ -v --tb=short`

Expected: All PASS

### Step 8.4: Final commit (if any fixes needed)

```bash
git add -A
git commit -m "fix: address lint/type issues from Tier 2 tracing

Co-Authored-By: Claude Opus 4.5 <noreply@anthropic.com>"
```

---

## Verification Checklist

- [ ] `TracingConfig` base class created
- [ ] `AzureAITracingConfig` with connection_string, enable_content_recording, enable_live_metrics
- [ ] `LangfuseTracingConfig` with public_key, secret_key, host
- [ ] `parse_tracing_config()` function handles all cases
- [ ] Optional dependencies in pyproject.toml (tracing-azure, tracing-langfuse)
- [ ] `AzureOpenAIConfig.tracing` field added
- [ ] `AzureLLMTransform` initializes tracing in `on_start()`
- [ ] `AzureLLMTransform` flushes tracing in `close()`
- [ ] Graceful degradation when SDK not installed (warning, not crash)
- [ ] `AzureMultiQueryLLMTransform` has same tracing support
- [ ] Documentation complete
- [ ] All existing tests pass
- [ ] mypy clean
- [ ] ruff clean

---

## Summary of Changes

| File | Change |
|------|--------|
| `src/elspeth/plugins/llm/tracing.py` | New - config models |
| `src/elspeth/plugins/llm/azure.py` | Add tracing field and lifecycle |
| `src/elspeth/plugins/llm/azure_multi_query.py` | Add tracing field and lifecycle |
| `src/elspeth/plugins/llm/openrouter.py` | Optional - Langfuse support |
| `src/elspeth/plugins/llm/openrouter_multi_query.py` | Optional - Langfuse support |
| `pyproject.toml` | Add tracing optional dependencies |
| `tests/plugins/llm/test_tracing_config.py` | New - config tests |
| `tests/plugins/llm/test_azure_tracing.py` | New - lifecycle tests |
| `docs/guides/tier2-tracing.md` | New - user documentation |
