# tests/unit/plugins/llm/test_provider_lifecycle.py
"""Tests for LLMTransform._create_provider() lifecycle.

Verifies that the unified LLMTransform correctly dispatches to the right
provider class based on config type (AzureOpenAIConfig vs OpenRouterConfig),
and that lifecycle ordering is enforced (recorder must be set before provider
creation).
"""

from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock

import pytest

from elspeth.plugins.transforms.llm.provider import LLMProvider
from elspeth.plugins.transforms.llm.providers.azure import AzureLLMProvider, AzureOpenAIConfig
from elspeth.plugins.transforms.llm.providers.openrouter import (
    OpenRouterConfig,
    OpenRouterLLMProvider,
)
from elspeth.plugins.transforms.llm.transform import LLMTransform

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

DYNAMIC_SCHEMA = {"mode": "observed"}


def _make_azure_config() -> dict[str, Any]:
    """Build minimal valid Azure LLMTransform config."""
    return {
        "provider": "azure",
        "deployment_name": "gpt-4o",
        "endpoint": "https://test.openai.azure.com",
        "api_key": "test-key",
        "prompt_template": "Test: {{ row.text }}",
        "schema": DYNAMIC_SCHEMA,
        "required_input_fields": ["text"],
    }


def _make_openrouter_config() -> dict[str, Any]:
    """Build minimal valid OpenRouter LLMTransform config."""
    return {
        "provider": "openrouter",
        "model": "openai/gpt-4o",
        "api_key": "test-key",
        "prompt_template": "Test: {{ row.text }}",
        "schema": DYNAMIC_SCHEMA,
        "required_input_fields": ["text"],
    }


def _prepare_transform_for_provider_creation(transform: LLMTransform) -> None:
    """Set the minimal internal state required for _create_provider().

    Simulates what on_start() would set (recorder, run_id, telemetry_emit)
    without going through the full lifecycle. This lets us test
    _create_provider() in isolation.
    """
    transform._recorder = MagicMock()
    transform._run_id = "test-run"
    transform._telemetry_emit = lambda event: None


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestCreateProviderDispatch:
    """Verify _create_provider() dispatches to correct provider class."""

    def test_azure_config_produces_azure_provider(self) -> None:
        """AzureOpenAIConfig should produce an AzureLLMProvider instance."""
        transform = LLMTransform(_make_azure_config())
        _prepare_transform_for_provider_creation(transform)

        provider = transform._create_provider()

        assert isinstance(provider, AzureLLMProvider)
        # Also verify it satisfies the runtime-checkable LLMProvider protocol
        assert isinstance(provider, LLMProvider)

    def test_openrouter_config_produces_openrouter_provider(self) -> None:
        """OpenRouterConfig should produce an OpenRouterLLMProvider instance."""
        transform = LLMTransform(_make_openrouter_config())
        _prepare_transform_for_provider_creation(transform)

        provider = transform._create_provider()

        assert isinstance(provider, OpenRouterLLMProvider)
        assert isinstance(provider, LLMProvider)

    def test_azure_config_type_is_azure_openai_config(self) -> None:
        """Verify the parsed config is actually AzureOpenAIConfig."""
        transform = LLMTransform(_make_azure_config())

        assert isinstance(transform._config, AzureOpenAIConfig)

    def test_openrouter_config_type_is_openrouter_config(self) -> None:
        """Verify the parsed config is actually OpenRouterConfig."""
        transform = LLMTransform(_make_openrouter_config())

        assert isinstance(transform._config, OpenRouterConfig)


class TestCreateProviderLifecycleEnforcement:
    """Verify lifecycle ordering is enforced."""

    def test_create_provider_raises_before_recorder_set(self) -> None:
        """_create_provider() raises RuntimeError if called before on_start()."""
        transform = LLMTransform(_make_azure_config())
        # Do NOT set _recorder — it stays None from __init__

        with pytest.raises(RuntimeError, match="before on_start"):
            transform._create_provider()

    def test_create_provider_raises_for_openrouter_before_recorder(self) -> None:
        """Same lifecycle check applies to OpenRouter config path."""
        transform = LLMTransform(_make_openrouter_config())

        with pytest.raises(RuntimeError, match="before on_start"):
            transform._create_provider()


class TestOnStartSetsProvider:
    """Verify on_start() calls _create_provider() and sets self._provider."""

    def test_on_start_sets_provider_for_azure(self) -> None:
        """on_start() should create and store the provider."""
        transform = LLMTransform(_make_azure_config())

        # Before on_start, provider is None
        assert transform._provider is None

        # Create a mock LifecycleContext with the required attributes
        mock_ctx = MagicMock()
        mock_ctx.landscape = MagicMock()
        mock_ctx.run_id = "test-run"
        mock_ctx.telemetry_emit = lambda event: None
        mock_ctx.rate_limit_registry = None

        transform.on_start(mock_ctx)

        assert transform._provider is not None
        # _provider is typed as LLMProvider (Protocol), but at runtime it's a concrete
        # AzureLLMProvider. Mypy considers this unreachable because the Protocol and
        # concrete class are nominally unrelated — but that's what we're verifying.
        assert isinstance(transform._provider, AzureLLMProvider)  # type: ignore[unreachable]
        assert isinstance(transform._provider, LLMProvider)

    def test_on_start_sets_provider_for_openrouter(self) -> None:
        """on_start() should create OpenRouterLLMProvider for openrouter config."""
        transform = LLMTransform(_make_openrouter_config())

        assert transform._provider is None

        mock_ctx = MagicMock()
        mock_ctx.landscape = MagicMock()
        mock_ctx.run_id = "test-run"
        mock_ctx.telemetry_emit = lambda event: None
        mock_ctx.rate_limit_registry = None

        transform.on_start(mock_ctx)

        assert transform._provider is not None
        # Same as above — verifying concrete provider type behind Protocol interface.
        assert isinstance(transform._provider, OpenRouterLLMProvider)  # type: ignore[unreachable]
        assert isinstance(transform._provider, LLMProvider)

    def test_on_start_stores_recorder(self) -> None:
        """on_start() should capture the recorder from context."""
        transform = LLMTransform(_make_azure_config())
        mock_landscape = MagicMock()

        mock_ctx = MagicMock()
        mock_ctx.landscape = mock_landscape
        mock_ctx.run_id = "test-run-123"
        mock_ctx.telemetry_emit = lambda event: None
        mock_ctx.rate_limit_registry = None

        transform.on_start(mock_ctx)

        assert transform._recorder is mock_landscape
        assert transform._run_id == "test-run-123"
