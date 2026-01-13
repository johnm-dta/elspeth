# tests/core/test_config.py
"""Tests for configuration schema and loading."""

from pathlib import Path

import pytest
from pydantic import ValidationError


class TestDatabaseSettings:
    """Database configuration validation."""

    def test_valid_sqlite_url(self) -> None:
        from elspeth.core.config import DatabaseSettings

        settings = DatabaseSettings(url="sqlite:///audit.db")
        assert settings.url == "sqlite:///audit.db"
        assert settings.pool_size == 5  # default

    def test_valid_postgres_url(self) -> None:
        from elspeth.core.config import DatabaseSettings

        settings = DatabaseSettings(
            url="postgresql://user:pass@localhost/db",
            pool_size=10,
        )
        assert settings.pool_size == 10

    def test_pool_size_must_be_positive(self) -> None:
        from elspeth.core.config import DatabaseSettings

        with pytest.raises(ValidationError):
            DatabaseSettings(url="sqlite:///test.db", pool_size=0)

    def test_settings_are_frozen(self) -> None:
        from elspeth.core.config import DatabaseSettings

        settings = DatabaseSettings(url="sqlite:///test.db")
        with pytest.raises(ValidationError):
            settings.url = "sqlite:///other.db"  # type: ignore[misc]


class TestRetrySettings:
    """Retry configuration validation."""

    def test_defaults(self) -> None:
        from elspeth.core.config import RetrySettings

        settings = RetrySettings()
        assert settings.max_attempts == 3
        assert settings.initial_delay_seconds == 1.0
        assert settings.max_delay_seconds == 60.0
        assert settings.exponential_base == 2.0

    def test_max_attempts_must_be_positive(self) -> None:
        from elspeth.core.config import RetrySettings

        with pytest.raises(ValidationError):
            RetrySettings(max_attempts=0)

    def test_delays_must_be_positive(self) -> None:
        from elspeth.core.config import RetrySettings

        with pytest.raises(ValidationError):
            RetrySettings(initial_delay_seconds=-1.0)


class TestElspethSettings:
    """Top-level settings validation."""

    def test_minimal_valid_config(self) -> None:
        from elspeth.core.config import ElspethSettings

        settings = ElspethSettings(
            database={"url": "sqlite:///audit.db"},
        )
        assert settings.database.url == "sqlite:///audit.db"
        assert settings.retry.max_attempts == 3  # default

    def test_nested_config(self) -> None:
        from elspeth.core.config import ElspethSettings

        settings = ElspethSettings(
            database={"url": "sqlite:///audit.db", "pool_size": 10},
            retry={"max_attempts": 5},
        )
        assert settings.database.pool_size == 10
        assert settings.retry.max_attempts == 5

    def test_run_id_prefix_default(self) -> None:
        from elspeth.core.config import ElspethSettings

        settings = ElspethSettings(database={"url": "sqlite:///audit.db"})
        assert settings.run_id_prefix == "run"


class TestLoadSettings:
    """Test Dynaconf-based settings loading."""

    def test_load_from_yaml_file(self, tmp_path: Path) -> None:
        from elspeth.core.config import load_settings

        config_file = tmp_path / "settings.yaml"
        config_file.write_text("""
database:
  url: "sqlite:///test.db"
  pool_size: 10
retry:
  max_attempts: 5
""")
        settings = load_settings(config_file)
        assert settings.database.url == "sqlite:///test.db"
        assert settings.database.pool_size == 10
        assert settings.retry.max_attempts == 5

    def test_load_with_env_override(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        from elspeth.core.config import load_settings

        config_file = tmp_path / "settings.yaml"
        config_file.write_text("""
database:
  url: "sqlite:///original.db"
""")
        # Environment variable should override YAML
        monkeypatch.setenv("ELSPETH_DATABASE__URL", "sqlite:///from_env.db")

        settings = load_settings(config_file)
        assert settings.database.url == "sqlite:///from_env.db"

    def test_load_validates_schema(self, tmp_path: Path) -> None:
        from elspeth.core.config import load_settings

        config_file = tmp_path / "settings.yaml"
        config_file.write_text("""
database:
  url: "sqlite:///test.db"
  pool_size: -1
""")
        with pytest.raises(ValidationError):
            load_settings(config_file)

    def test_load_missing_required_field(self, tmp_path: Path) -> None:
        from elspeth.core.config import load_settings

        config_file = tmp_path / "settings.yaml"
        config_file.write_text("""
retry:
  max_attempts: 5
""")
        # database.url is required
        with pytest.raises(ValidationError):
            load_settings(config_file)

    def test_load_missing_file_raises_file_not_found(self, tmp_path: Path) -> None:
        from elspeth.core.config import load_settings

        missing_file = tmp_path / "nonexistent.yaml"
        with pytest.raises(FileNotFoundError, match="Config file not found"):
            load_settings(missing_file)


class TestDatasourceSettings:
    """DatasourceSettings matches architecture specification."""

    def test_datasource_settings_structure(self) -> None:
        """DatasourceSettings has plugin and options."""
        from elspeth.core.config import DatasourceSettings

        ds = DatasourceSettings(plugin="csv_local", options={"path": "data/input.csv"})
        assert ds.plugin == "csv_local"
        assert ds.options == {"path": "data/input.csv"}

    def test_datasource_settings_options_default_empty(self) -> None:
        """Options defaults to empty dict."""
        from elspeth.core.config import DatasourceSettings

        ds = DatasourceSettings(plugin="csv")
        assert ds.options == {}

    def test_datasource_settings_frozen(self) -> None:
        """DatasourceSettings is immutable."""
        from elspeth.core.config import DatasourceSettings

        ds = DatasourceSettings(plugin="csv")
        with pytest.raises(ValidationError):
            ds.plugin = "json"


class TestRowPluginSettings:
    """RowPluginSettings matches architecture specification."""

    def test_row_plugin_settings_structure(self) -> None:
        """RowPluginSettings has plugin, type, options, routes."""
        from elspeth.core.config import RowPluginSettings

        rp = RowPluginSettings(
            plugin="threshold_gate",
            type="gate",
            options={"field": "confidence", "min": 0.8},
            routes={"pass": "continue", "fail": "quarantine"},
        )
        assert rp.plugin == "threshold_gate"
        assert rp.type == "gate"
        assert rp.options == {"field": "confidence", "min": 0.8}
        assert rp.routes == {"pass": "continue", "fail": "quarantine"}

    def test_row_plugin_settings_defaults(self) -> None:
        """RowPluginSettings defaults: type=transform, no routes."""
        from elspeth.core.config import RowPluginSettings

        rp = RowPluginSettings(plugin="field_mapper")
        assert rp.type == "transform"
        assert rp.options == {}
        assert rp.routes is None

    def test_row_plugin_settings_type_validation(self) -> None:
        """Type must be 'transform' or 'gate'."""
        from elspeth.core.config import RowPluginSettings

        with pytest.raises(ValidationError):
            RowPluginSettings(plugin="test", type="invalid")
