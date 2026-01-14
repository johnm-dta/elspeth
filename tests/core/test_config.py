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
            datasource={"plugin": "csv"},
            sinks={"output": {"plugin": "csv"}},
            output_sink="output",
        )
        assert settings.datasource.plugin == "csv"
        assert settings.retry.max_attempts == 3  # default

    def test_nested_config(self) -> None:
        from elspeth.core.config import ElspethSettings

        settings = ElspethSettings(
            datasource={"plugin": "csv"},
            sinks={"output": {"plugin": "csv"}},
            output_sink="output",
            retry={"max_attempts": 5},
        )
        assert settings.retry.max_attempts == 5

    def test_settings_are_frozen(self) -> None:
        from elspeth.core.config import ElspethSettings

        settings = ElspethSettings(
            datasource={"plugin": "csv"},
            sinks={"output": {"plugin": "csv"}},
            output_sink="output",
        )
        with pytest.raises(ValidationError):
            settings.output_sink = "other"  # type: ignore[misc]


class TestLoadSettings:
    """Test Dynaconf-based settings loading."""

    def test_load_from_yaml_file(self, tmp_path: Path) -> None:
        from elspeth.core.config import load_settings

        config_file = tmp_path / "settings.yaml"
        config_file.write_text("""
datasource:
  plugin: "csv"
  options:
    path: "input.csv"
sinks:
  output:
    plugin: "csv"
    options:
      path: "output.csv"
output_sink: "output"
retry:
  max_attempts: 5
""")
        settings = load_settings(config_file)
        assert settings.datasource.plugin == "csv"
        assert settings.datasource.options == {"path": "input.csv"}
        assert settings.retry.max_attempts == 5

    def test_load_with_env_override(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        from elspeth.core.config import load_settings

        config_file = tmp_path / "settings.yaml"
        config_file.write_text("""
datasource:
  plugin: "csv"
sinks:
  output:
    plugin: "csv"
output_sink: "output"
""")
        # Environment variable should override YAML
        monkeypatch.setenv("ELSPETH_DATASOURCE__PLUGIN", "json")

        settings = load_settings(config_file)
        assert settings.datasource.plugin == "json"

    def test_load_validates_schema(self, tmp_path: Path) -> None:
        from elspeth.core.config import load_settings

        config_file = tmp_path / "settings.yaml"
        config_file.write_text("""
datasource:
  plugin: "csv"
sinks:
  output:
    plugin: "csv"
output_sink: "output"
concurrency:
  max_workers: -1
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
        # datasource, sinks, output_sink are required
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


class TestSinkSettings:
    """SinkSettings matches architecture specification."""

    def test_sink_settings_structure(self) -> None:
        """SinkSettings has plugin and options."""
        from elspeth.core.config import SinkSettings

        sink = SinkSettings(plugin="csv", options={"path": "output/results.csv"})
        assert sink.plugin == "csv"
        assert sink.options == {"path": "output/results.csv"}

    def test_sink_settings_options_default_empty(self) -> None:
        """Options defaults to empty dict."""
        from elspeth.core.config import SinkSettings

        sink = SinkSettings(plugin="database")
        assert sink.options == {}


class TestLandscapeExportSettings:
    """LandscapeExportSettings for audit trail export configuration."""

    def test_landscape_export_config_defaults(self) -> None:
        """Export config should have sensible defaults."""
        from elspeth.core.config import LandscapeSettings

        settings = LandscapeSettings()
        assert settings.export is not None
        assert settings.export.enabled is False
        assert settings.export.format == "csv"
        assert settings.export.sign is False

    def test_landscape_export_config_with_sink(self) -> None:
        """Export config should accept sink reference."""
        from elspeth.core.config import LandscapeSettings

        settings = LandscapeSettings(
            export={
                "enabled": True,
                "sink": "audit_archive",
                "format": "csv",
                "sign": True,
            }
        )
        assert settings.export.enabled is True
        assert settings.export.sink == "audit_archive"
        assert settings.export.sign is True

    def test_landscape_export_format_validation(self) -> None:
        """Format must be 'csv' or 'json'."""
        from elspeth.core.config import LandscapeExportSettings

        with pytest.raises(ValidationError):
            LandscapeExportSettings(format="xml")

    def test_landscape_export_settings_frozen(self) -> None:
        """LandscapeExportSettings is immutable."""
        from elspeth.core.config import LandscapeExportSettings

        export = LandscapeExportSettings()
        with pytest.raises(ValidationError):
            export.enabled = True


class TestLandscapeSettings:
    """LandscapeSettings matches architecture specification."""

    def test_landscape_settings_structure(self) -> None:
        """LandscapeSettings has enabled, backend, url."""
        from elspeth.core.config import LandscapeSettings

        ls = LandscapeSettings(enabled=True, backend="sqlite", url="sqlite:///./runs/audit.db")
        assert ls.enabled is True
        assert ls.backend == "sqlite"
        assert ls.url == "sqlite:///./runs/audit.db"

    def test_landscape_settings_defaults(self) -> None:
        """LandscapeSettings has sensible defaults."""
        from elspeth.core.config import LandscapeSettings

        ls = LandscapeSettings()
        assert ls.enabled is True
        assert ls.backend == "sqlite"
        assert ls.url == "sqlite:///./runs/audit.db"

    def test_landscape_settings_postgresql_url(self) -> None:
        """LandscapeSettings accepts PostgreSQL DSNs without mangling."""
        from elspeth.core.config import LandscapeSettings

        # This would fail with pathlib.Path which mangles // as UNC paths
        pg_url = "postgresql://user:pass@localhost:5432/elspeth_audit"
        ls = LandscapeSettings(enabled=True, backend="postgresql", url=pg_url)
        assert ls.url == pg_url  # Preserved exactly

    def test_landscape_settings_backend_validation(self) -> None:
        """Backend must be sqlite or postgresql."""
        from elspeth.core.config import LandscapeSettings

        with pytest.raises(ValidationError):
            LandscapeSettings(backend="mysql")


class TestConcurrencySettings:
    """ConcurrencySettings matches architecture specification."""

    def test_concurrency_settings_structure(self) -> None:
        """ConcurrencySettings has max_workers."""
        from elspeth.core.config import ConcurrencySettings

        cs = ConcurrencySettings(max_workers=16)
        assert cs.max_workers == 16

    def test_concurrency_settings_default(self) -> None:
        """Default max_workers is 4 per architecture."""
        from elspeth.core.config import ConcurrencySettings

        cs = ConcurrencySettings()
        assert cs.max_workers == 4

    def test_concurrency_settings_validation(self) -> None:
        """max_workers must be positive."""
        from elspeth.core.config import ConcurrencySettings

        with pytest.raises(ValidationError):
            ConcurrencySettings(max_workers=0)
        with pytest.raises(ValidationError):
            ConcurrencySettings(max_workers=-1)


class TestLoadSettingsArchitecture:
    """load_settings() parses architecture-compliant YAML."""

    def test_load_readme_example(self, tmp_path: Path) -> None:
        """Load the exact example from README.md."""
        from elspeth.core.config import load_settings

        config_file = tmp_path / "settings.yaml"
        config_file.write_text("""
datasource:
  plugin: csv_local
  options:
    path: data/submissions.csv

sinks:
  results:
    plugin: csv
    options:
      path: output/results.csv
  flagged:
    plugin: csv
    options:
      path: output/flagged_for_review.csv

row_plugins:
  - plugin: pattern_gate
    type: gate
    options:
      patterns:
        - "ignore previous"
        - "disregard instructions"
    routes:
      suspicious: flagged
      clean: continue

output_sink: results

landscape:
  enabled: true
  backend: sqlite
  url: sqlite:///./runs/audit.db
""")

        settings = load_settings(config_file)

        assert settings.datasource.plugin == "csv_local"
        assert settings.datasource.options["path"] == "data/submissions.csv"
        assert len(settings.sinks) == 2
        assert len(settings.row_plugins) == 1
        assert settings.row_plugins[0].type == "gate"
        assert settings.output_sink == "results"
        assert settings.landscape.backend == "sqlite"

    def test_load_minimal_config(self, tmp_path: Path) -> None:
        """Minimal valid configuration."""
        from elspeth.core.config import load_settings

        config_file = tmp_path / "settings.yaml"
        config_file.write_text("""
datasource:
  plugin: csv

sinks:
  output:
    plugin: csv

output_sink: output
""")

        settings = load_settings(config_file)

        assert settings.datasource.plugin == "csv"
        assert settings.landscape.enabled is True  # Default
        assert settings.concurrency.max_workers == 4  # Default

    def test_load_invalid_output_sink(self, tmp_path: Path) -> None:
        """Error when output_sink doesn't exist."""
        from pydantic import ValidationError

        from elspeth.core.config import load_settings

        config_file = tmp_path / "settings.yaml"
        config_file.write_text("""
datasource:
  plugin: csv

sinks:
  results:
    plugin: csv

output_sink: nonexistent
""")

        with pytest.raises(ValidationError) as exc_info:
            load_settings(config_file)

        assert "output_sink" in str(exc_info.value)


class TestExportSinkValidation:
    """Validation that export.sink references a defined sink."""

    def test_export_sink_must_exist_when_enabled(self) -> None:
        """If export.enabled=True, export.sink must reference a defined sink."""
        from elspeth.core.config import ElspethSettings

        with pytest.raises(ValidationError) as exc_info:
            ElspethSettings(
                datasource={"plugin": "csv", "options": {"path": "input.csv"}},
                sinks={"output": {"plugin": "csv", "options": {"path": "out.csv"}}},
                output_sink="output",
                landscape={
                    "export": {
                        "enabled": True,
                        "sink": "nonexistent_sink",  # Not in sinks
                    }
                },
            )

        assert "export.sink 'nonexistent_sink' not found in sinks" in str(exc_info.value)

    def test_export_sink_not_required_when_disabled(self) -> None:
        """If export.enabled=False, sink can be None."""
        from elspeth.core.config import ElspethSettings

        # Should not raise
        settings = ElspethSettings(
            datasource={"plugin": "csv", "options": {"path": "input.csv"}},
            sinks={"output": {"plugin": "csv", "options": {"path": "out.csv"}}},
            output_sink="output",
            landscape={
                "export": {"enabled": False}  # No sink required
            },
        )
        assert settings.landscape.export.sink is None

    def test_export_sink_required_when_enabled(self) -> None:
        """If export.enabled=True, sink cannot be None."""
        from elspeth.core.config import ElspethSettings

        with pytest.raises(ValidationError) as exc_info:
            ElspethSettings(
                datasource={"plugin": "csv", "options": {"path": "input.csv"}},
                sinks={"output": {"plugin": "csv", "options": {"path": "out.csv"}}},
                output_sink="output",
                landscape={
                    "export": {
                        "enabled": True,
                        # sink is None (not provided)
                    }
                },
            )

        assert "landscape.export.sink is required when export is enabled" in str(exc_info.value)

    def test_export_sink_valid_reference(self) -> None:
        """If export.sink references a valid sink, no error."""
        from elspeth.core.config import ElspethSettings

        # Should not raise
        settings = ElspethSettings(
            datasource={"plugin": "csv", "options": {"path": "input.csv"}},
            sinks={
                "output": {"plugin": "csv", "options": {"path": "out.csv"}},
                "audit_archive": {"plugin": "csv", "options": {"path": "audit.csv"}},
            },
            output_sink="output",
            landscape={
                "export": {
                    "enabled": True,
                    "sink": "audit_archive",  # Valid sink
                }
            },
        )
        assert settings.landscape.export.sink == "audit_archive"


class TestCheckpointSettings:
    """Tests for checkpoint configuration."""

    def test_checkpoint_settings_defaults(self) -> None:
        from elspeth.core.config import CheckpointSettings

        settings = CheckpointSettings()

        assert settings.enabled is True
        assert settings.frequency == "every_row"
        assert settings.aggregation_boundaries is True

    def test_checkpoint_frequency_options(self) -> None:
        from elspeth.core.config import CheckpointSettings

        # Every row (safest, slowest)
        s1 = CheckpointSettings(frequency="every_row")
        assert s1.frequency == "every_row"

        # Every N rows (balanced)
        s2 = CheckpointSettings(frequency="every_n", checkpoint_interval=100)
        assert s2.frequency == "every_n"
        assert s2.checkpoint_interval == 100

        # Aggregation boundaries only (fastest, less safe)
        s3 = CheckpointSettings(frequency="aggregation_only")
        assert s3.frequency == "aggregation_only"

    def test_checkpoint_settings_validation(self) -> None:
        from pydantic import ValidationError

        from elspeth.core.config import CheckpointSettings

        # every_n requires checkpoint_interval
        with pytest.raises(ValidationError):
            CheckpointSettings(frequency="every_n", checkpoint_interval=None)

    def test_checkpoint_interval_must_be_positive(self) -> None:
        """checkpoint_interval must be > 0 when provided."""
        from pydantic import ValidationError

        from elspeth.core.config import CheckpointSettings

        # Zero is invalid
        with pytest.raises(ValidationError):
            CheckpointSettings(frequency="every_n", checkpoint_interval=0)

        # Negative is invalid
        with pytest.raises(ValidationError):
            CheckpointSettings(frequency="every_n", checkpoint_interval=-1)

    def test_checkpoint_settings_frozen(self) -> None:
        """CheckpointSettings is immutable."""
        from elspeth.core.config import CheckpointSettings

        settings = CheckpointSettings()
        with pytest.raises(ValidationError):
            settings.enabled = False  # type: ignore[misc]

    def test_checkpoint_settings_invalid_frequency(self) -> None:
        """Frequency must be a valid option."""
        from pydantic import ValidationError

        from elspeth.core.config import CheckpointSettings

        with pytest.raises(ValidationError):
            CheckpointSettings(frequency="invalid")


class TestElspethSettingsArchitecture:
    """Top-level settings matches architecture specification."""

    def test_elspeth_settings_required_fields(self) -> None:
        """ElspethSettings requires datasource, sinks, output_sink."""
        from elspeth.core.config import ElspethSettings

        # Missing required fields
        with pytest.raises(ValidationError) as exc_info:
            ElspethSettings()

        errors = exc_info.value.errors()
        missing_fields = {e["loc"][0] for e in errors if e["type"] == "missing"}
        assert "datasource" in missing_fields
        assert "sinks" in missing_fields
        assert "output_sink" in missing_fields

    def test_elspeth_settings_minimal_valid(self) -> None:
        """Minimal valid configuration."""
        from elspeth.core.config import (
            DatasourceSettings,
            ElspethSettings,
            SinkSettings,
        )

        settings = ElspethSettings(
            datasource=DatasourceSettings(plugin="csv", options={"path": "in.csv"}),
            sinks={"results": SinkSettings(plugin="csv", options={"path": "out.csv"})},
            output_sink="results",
        )

        assert settings.datasource.plugin == "csv"
        assert "results" in settings.sinks
        assert settings.output_sink == "results"
        # Defaults applied
        assert settings.row_plugins == []
        assert settings.landscape.enabled is True
        assert settings.concurrency.max_workers == 4

    def test_elspeth_settings_output_sink_must_exist(self) -> None:
        """output_sink must reference a defined sink."""
        from elspeth.core.config import (
            DatasourceSettings,
            ElspethSettings,
            SinkSettings,
        )

        with pytest.raises(ValidationError) as exc_info:
            ElspethSettings(
                datasource=DatasourceSettings(plugin="csv"),
                sinks={"results": SinkSettings(plugin="csv")},
                output_sink="nonexistent",  # Not in sinks!
            )

        assert "output_sink" in str(exc_info.value)

    def test_elspeth_settings_at_least_one_sink(self) -> None:
        """At least one sink is required."""
        from elspeth.core.config import DatasourceSettings, ElspethSettings

        with pytest.raises(ValidationError) as exc_info:
            ElspethSettings(
                datasource=DatasourceSettings(plugin="csv"),
                sinks={},  # Empty!
                output_sink="results",
            )

        assert "sink" in str(exc_info.value).lower()
