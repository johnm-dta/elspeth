# tests/unit/plugins/llm/test_plugin_registration.py
"""Tests for unified LLM plugin registration and validation dispatch (Task 10).

Verifies that:
- "llm" plugin dispatches to provider-specific config models
- Old plugin names raise helpful migration errors
- Discovery finds the new LLMTransform plugin
"""

from __future__ import annotations

import pytest

from elspeth.contracts import Determinism
from elspeth.plugins.infrastructure.validation import get_transform_config_model


class TestLLMPluginConfigDispatch:
    """Tests for get_transform_config_model provider dispatch."""

    def test_llm_plugin_dispatches_to_azure_config(self) -> None:
        """verify get_transform_config_model("llm", {"provider": "azure"}) returns AzureOpenAIConfig."""
        from elspeth.plugins.transforms.llm.providers.azure import AzureOpenAIConfig

        config_model = get_transform_config_model("llm", {"provider": "azure"})
        assert config_model is AzureOpenAIConfig

    def test_llm_plugin_dispatches_to_openrouter_config(self) -> None:
        """verify get_transform_config_model("llm", {"provider": "openrouter"}) returns OpenRouterConfig."""
        from elspeth.plugins.transforms.llm.providers.openrouter import OpenRouterConfig

        config_model = get_transform_config_model("llm", {"provider": "openrouter"})
        assert config_model is OpenRouterConfig

    def test_llm_plugin_missing_provider_falls_back_to_base(self) -> None:
        """verify missing provider key returns LLMConfig (Pydantic catches the Literal validation)."""
        from elspeth.plugins.transforms.llm.base import LLMConfig

        config_model = get_transform_config_model("llm", {})
        assert config_model is LLMConfig

    def test_llm_plugin_unknown_provider_raises(self) -> None:
        """verify unknown provider raises ValueError with valid providers listed."""
        with pytest.raises(ValueError, match="Unknown LLM provider 'fake'"):
            get_transform_config_model("llm", {"provider": "fake"})

    def test_llm_plugin_none_config_falls_back_to_base(self) -> None:
        """verify None config falls back to LLMConfig."""
        from elspeth.plugins.transforms.llm.base import LLMConfig

        config_model = get_transform_config_model("llm", None)
        assert config_model is LLMConfig


class TestOldPluginNamesRejected:
    """Old plugin names are rejected as unknown types."""

    @pytest.mark.parametrize(
        "old_name",
        ["azure_llm", "openrouter_llm", "azure_multi_query_llm", "openrouter_multi_query_llm"],
    )
    def test_old_plugin_names_raise_unknown_type(self, old_name: str) -> None:
        """Old plugin names should raise ValueError as unknown transform type."""
        with pytest.raises(ValueError, match="Unknown transform type"):
            get_transform_config_model(old_name)


class TestLLMPluginDiscovery:
    """Verify the unified LLMTransform is discovered correctly."""

    def test_llm_plugin_discovered_with_correct_name(self) -> None:
        from elspeth.plugins.infrastructure.discovery import discover_all_plugins

        discovered = discover_all_plugins()
        transform_names = [cls.name for cls in discovered["transforms"]]  # type: ignore[attr-defined]
        assert "llm" in transform_names

    def test_llm_plugin_is_non_deterministic(self) -> None:
        from elspeth.plugins.transforms.llm.transform import LLMTransform

        assert LLMTransform.determinism == Determinism.NON_DETERMINISTIC
