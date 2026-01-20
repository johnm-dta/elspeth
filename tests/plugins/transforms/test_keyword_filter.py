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
