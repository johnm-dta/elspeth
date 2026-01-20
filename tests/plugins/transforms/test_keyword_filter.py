"""Tests for KeywordFilter transform."""

import pytest

from elspeth.plugins.config_base import PluginConfigError


class TestKeywordFilterConfig:
    """Tests for KeywordFilterConfig validation."""

    def test_config_requires_fields(self) -> None:
        """Config must specify fields - no defaults allowed."""
        from elspeth.plugins.transforms.keyword_filter import KeywordFilterConfig

        with pytest.raises(PluginConfigError) as exc_info:
            KeywordFilterConfig.from_dict(
                {
                    "blocked_patterns": ["test"],
                    "schema": {"fields": "dynamic"},
                }
            )
        assert "fields" in str(exc_info.value).lower()

    def test_config_requires_blocked_patterns(self) -> None:
        """Config must specify blocked_patterns - no defaults allowed."""
        from elspeth.plugins.transforms.keyword_filter import KeywordFilterConfig

        with pytest.raises(PluginConfigError) as exc_info:
            KeywordFilterConfig.from_dict(
                {
                    "fields": ["content"],
                    "schema": {"fields": "dynamic"},
                }
            )
        assert "blocked_patterns" in str(exc_info.value).lower()

    def test_config_accepts_single_field(self) -> None:
        """Config accepts single field as string."""
        from elspeth.plugins.transforms.keyword_filter import KeywordFilterConfig

        cfg = KeywordFilterConfig.from_dict(
            {
                "fields": "content",
                "blocked_patterns": ["test"],
                "schema": {"fields": "dynamic"},
            }
        )
        assert cfg.fields == "content"

    def test_config_accepts_field_list(self) -> None:
        """Config accepts list of fields."""
        from elspeth.plugins.transforms.keyword_filter import KeywordFilterConfig

        cfg = KeywordFilterConfig.from_dict(
            {
                "fields": ["content", "subject"],
                "blocked_patterns": ["test"],
                "schema": {"fields": "dynamic"},
            }
        )
        assert cfg.fields == ["content", "subject"]

    def test_config_accepts_all_keyword(self) -> None:
        """Config accepts 'all' to scan all string fields."""
        from elspeth.plugins.transforms.keyword_filter import KeywordFilterConfig

        cfg = KeywordFilterConfig.from_dict(
            {
                "fields": "all",
                "blocked_patterns": ["test"],
                "schema": {"fields": "dynamic"},
            }
        )
        assert cfg.fields == "all"

    def test_config_validates_patterns_not_empty(self) -> None:
        """Config rejects empty blocked_patterns list."""
        from elspeth.plugins.transforms.keyword_filter import KeywordFilterConfig

        with pytest.raises(PluginConfigError) as exc_info:
            KeywordFilterConfig.from_dict(
                {
                    "fields": ["content"],
                    "blocked_patterns": [],
                    "schema": {"fields": "dynamic"},
                }
            )
        assert "blocked_patterns" in str(exc_info.value).lower()


class TestKeywordFilterInstantiation:
    """Tests for KeywordFilter transform instantiation."""

    def test_transform_has_required_attributes(self) -> None:
        """Transform has all protocol-required attributes."""
        from elspeth.plugins.transforms.keyword_filter import KeywordFilter

        transform = KeywordFilter(
            {
                "fields": ["content"],
                "blocked_patterns": ["test"],
                "schema": {"fields": "dynamic"},
            }
        )

        assert transform.name == "keyword_filter"
        assert transform.determinism.value == "deterministic"
        assert transform.plugin_version == "1.0.0"
        assert transform.is_batch_aware is False
        assert transform.creates_tokens is False
        assert transform.input_schema is not None
        assert transform.output_schema is not None

    def test_transform_compiles_patterns_at_init(self) -> None:
        """Transform compiles regex patterns at initialization."""
        from elspeth.plugins.transforms.keyword_filter import KeywordFilter

        transform = KeywordFilter(
            {
                "fields": ["content"],
                "blocked_patterns": [r"\bpassword\b", r"(?i)secret"],
                "schema": {"fields": "dynamic"},
            }
        )

        # Patterns should be compiled (implementation detail, but important for perf)
        assert len(transform._compiled_patterns) == 2

    def test_transform_rejects_invalid_regex(self) -> None:
        """Transform fails at init if regex pattern is invalid."""
        import re

        from elspeth.plugins.transforms.keyword_filter import KeywordFilter

        with pytest.raises(re.error):
            KeywordFilter(
                {
                    "fields": ["content"],
                    "blocked_patterns": ["[invalid(regex"],
                    "schema": {"fields": "dynamic"},
                }
            )
