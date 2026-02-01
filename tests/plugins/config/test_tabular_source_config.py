# tests/plugins/config/test_tabular_source_config.py
"""Tests for TabularSourceDataConfig validation."""

import pytest

from elspeth.plugins.config_base import PluginConfigError


class TestTabularSourceDataConfigValidation:
    """Tests for field normalization config option validation."""

    def test_normalize_with_columns_raises(self) -> None:
        """normalize_fields=True with columns raises error."""
        from elspeth.plugins.config_base import TabularSourceDataConfig

        with pytest.raises(PluginConfigError, match="cannot be used with columns"):
            TabularSourceDataConfig.from_dict(
                {
                    "path": "/tmp/test.csv",
                    "schema": {"fields": "dynamic"},
                    "on_validation_failure": "quarantine",
                    "columns": ["a", "b"],
                    "normalize_fields": True,
                }
            )

    def test_mapping_without_normalize_or_columns_raises(self) -> None:
        """field_mapping without normalize_fields or columns raises error."""
        from elspeth.plugins.config_base import TabularSourceDataConfig

        with pytest.raises(PluginConfigError, match="requires normalize_fields"):
            TabularSourceDataConfig.from_dict(
                {
                    "path": "/tmp/test.csv",
                    "schema": {"fields": "dynamic"},
                    "on_validation_failure": "quarantine",
                    "field_mapping": {"a": "b"},
                }
            )

    def test_columns_with_python_keyword_raises(self) -> None:
        """columns entry that is Python keyword raises error."""
        from elspeth.plugins.config_base import TabularSourceDataConfig

        with pytest.raises(PluginConfigError, match="Python keyword"):
            TabularSourceDataConfig.from_dict(
                {
                    "path": "/tmp/test.csv",
                    "schema": {"fields": "dynamic"},
                    "on_validation_failure": "quarantine",
                    "columns": ["id", "class", "name"],
                }
            )

    def test_columns_with_invalid_identifier_raises(self) -> None:
        """columns entry that is invalid identifier raises error."""
        from elspeth.plugins.config_base import TabularSourceDataConfig

        with pytest.raises(PluginConfigError, match=r"valid.*identifier"):
            TabularSourceDataConfig.from_dict(
                {
                    "path": "/tmp/test.csv",
                    "schema": {"fields": "dynamic"},
                    "on_validation_failure": "quarantine",
                    "columns": ["id", "123_bad", "name"],
                }
            )

    def test_columns_with_duplicates_raises(self) -> None:
        """columns with duplicate entries raises error."""
        from elspeth.plugins.config_base import TabularSourceDataConfig

        with pytest.raises(PluginConfigError, match=r"[Dd]uplicate"):
            TabularSourceDataConfig.from_dict(
                {
                    "path": "/tmp/test.csv",
                    "schema": {"fields": "dynamic"},
                    "on_validation_failure": "quarantine",
                    "columns": ["id", "name", "id"],
                }
            )

    def test_field_mapping_value_is_keyword_raises(self) -> None:
        """field_mapping value that is Python keyword raises error."""
        from elspeth.plugins.config_base import TabularSourceDataConfig

        with pytest.raises(PluginConfigError, match="Python keyword"):
            TabularSourceDataConfig.from_dict(
                {
                    "path": "/tmp/test.csv",
                    "schema": {"fields": "dynamic"},
                    "on_validation_failure": "quarantine",
                    "normalize_fields": True,
                    "field_mapping": {"user_id": "class"},
                }
            )

    def test_valid_config_with_normalize_fields(self) -> None:
        """Valid config with normalize_fields passes."""
        from elspeth.plugins.config_base import TabularSourceDataConfig

        cfg = TabularSourceDataConfig.from_dict(
            {
                "path": "/tmp/test.csv",
                "schema": {"fields": "dynamic"},
                "on_validation_failure": "quarantine",
                "normalize_fields": True,
            }
        )
        assert cfg.normalize_fields is True
        assert cfg.field_mapping is None
        assert cfg.columns is None

    def test_valid_config_with_columns(self) -> None:
        """Valid config with columns passes."""
        from elspeth.plugins.config_base import TabularSourceDataConfig

        cfg = TabularSourceDataConfig.from_dict(
            {
                "path": "/tmp/test.csv",
                "schema": {"fields": "dynamic"},
                "on_validation_failure": "quarantine",
                "columns": ["id", "name", "amount"],
            }
        )
        assert cfg.columns == ["id", "name", "amount"]
        assert cfg.normalize_fields is False

    def test_valid_config_with_normalize_and_mapping(self) -> None:
        """Valid config with normalize_fields + field_mapping passes."""
        from elspeth.plugins.config_base import TabularSourceDataConfig

        cfg = TabularSourceDataConfig.from_dict(
            {
                "path": "/tmp/test.csv",
                "schema": {"fields": "dynamic"},
                "on_validation_failure": "quarantine",
                "normalize_fields": True,
                "field_mapping": {"user_id": "uid"},
            }
        )
        assert cfg.normalize_fields is True
        assert cfg.field_mapping == {"user_id": "uid"}

    def test_empty_field_mapping_treated_as_none(self) -> None:
        """Empty field_mapping dict should behave same as None."""
        from elspeth.plugins.config_base import TabularSourceDataConfig

        cfg = TabularSourceDataConfig.from_dict(
            {
                "path": "/tmp/test.csv",
                "schema": {"fields": "dynamic"},
                "on_validation_failure": "quarantine",
                "normalize_fields": True,
                "field_mapping": {},
            }
        )
        # Empty dict should be allowed (treated as no mapping)
        assert cfg.field_mapping == {}

    def test_single_column_headerless_mode(self) -> None:
        """Columns with single entry should work."""
        from elspeth.plugins.config_base import TabularSourceDataConfig

        cfg = TabularSourceDataConfig.from_dict(
            {
                "path": "/tmp/test.csv",
                "schema": {"fields": "dynamic"},
                "on_validation_failure": "quarantine",
                "columns": ["id"],
            }
        )
        assert cfg.columns == ["id"]
