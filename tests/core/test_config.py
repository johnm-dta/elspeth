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

    def test_load_with_env_override(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
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
            ds.plugin = "json"  # type: ignore[misc]


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
            export.enabled = True  # type: ignore[misc]


class TestLandscapeSettings:
    """LandscapeSettings matches architecture specification."""

    def test_landscape_settings_structure(self) -> None:
        """LandscapeSettings has enabled, backend, url."""
        from elspeth.core.config import LandscapeSettings

        ls = LandscapeSettings(
            enabled=True, backend="sqlite", url="sqlite:///./runs/audit.db"
        )
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

        assert "export.sink 'nonexistent_sink' not found in sinks" in str(
            exc_info.value
        )

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

        assert "landscape.export.sink is required when export is enabled" in str(
            exc_info.value
        )

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
            ElspethSettings()  # type: ignore[call-arg]

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


class TestRateLimitSettings:
    """Tests for rate limit configuration."""

    def test_rate_limit_settings_defaults(self) -> None:
        from elspeth.core.config import RateLimitSettings

        settings = RateLimitSettings()

        assert settings.enabled is True
        assert settings.default_requests_per_second == 10
        assert settings.persistence_path is None

    def test_rate_limit_per_service(self) -> None:
        from elspeth.core.config import RateLimitSettings, ServiceRateLimit

        settings = RateLimitSettings(
            services={
                "openai": ServiceRateLimit(
                    requests_per_second=5,
                    requests_per_minute=100,
                ),
                "weather_api": ServiceRateLimit(
                    requests_per_second=20,
                ),
            }
        )

        assert settings.services["openai"].requests_per_second == 5
        assert settings.services["openai"].requests_per_minute == 100
        assert settings.services["weather_api"].requests_per_second == 20

    def test_rate_limit_get_service_config(self) -> None:
        from elspeth.core.config import RateLimitSettings, ServiceRateLimit

        settings = RateLimitSettings(
            default_requests_per_second=10,
            services={
                "openai": ServiceRateLimit(requests_per_second=5),
            },
        )

        # Configured service
        openai_config = settings.get_service_config("openai")
        assert openai_config.requests_per_second == 5

        # Unconfigured service falls back to default
        other_config = settings.get_service_config("other_api")
        assert other_config.requests_per_second == 10

    def test_rate_limit_settings_frozen(self) -> None:
        """RateLimitSettings is immutable."""
        from elspeth.core.config import RateLimitSettings

        settings = RateLimitSettings()
        with pytest.raises(ValidationError):
            settings.enabled = False  # type: ignore[misc]

    def test_service_rate_limit_frozen(self) -> None:
        """ServiceRateLimit is immutable."""
        from elspeth.core.config import ServiceRateLimit

        limit = ServiceRateLimit(requests_per_second=10)
        with pytest.raises(ValidationError):
            limit.requests_per_second = 20  # type: ignore[misc]

    def test_service_rate_limit_requests_per_second_must_be_positive(self) -> None:
        """requests_per_second must be > 0."""
        from elspeth.core.config import ServiceRateLimit

        with pytest.raises(ValidationError):
            ServiceRateLimit(requests_per_second=0)

        with pytest.raises(ValidationError):
            ServiceRateLimit(requests_per_second=-1)

    def test_service_rate_limit_requests_per_minute_must_be_positive(self) -> None:
        """requests_per_minute must be > 0 when provided."""
        from elspeth.core.config import ServiceRateLimit

        with pytest.raises(ValidationError):
            ServiceRateLimit(requests_per_second=10, requests_per_minute=0)

        with pytest.raises(ValidationError):
            ServiceRateLimit(requests_per_second=10, requests_per_minute=-1)

    def test_rate_limit_settings_default_requests_per_second_must_be_positive(
        self,
    ) -> None:
        """default_requests_per_second must be > 0."""
        from elspeth.core.config import RateLimitSettings

        with pytest.raises(ValidationError):
            RateLimitSettings(default_requests_per_second=0)

        with pytest.raises(ValidationError):
            RateLimitSettings(default_requests_per_second=-1)

    def test_rate_limit_settings_default_requests_per_minute_must_be_positive(
        self,
    ) -> None:
        """default_requests_per_minute must be > 0 when provided."""
        from elspeth.core.config import RateLimitSettings

        with pytest.raises(ValidationError):
            RateLimitSettings(default_requests_per_minute=0)

        with pytest.raises(ValidationError):
            RateLimitSettings(default_requests_per_minute=-1)


class TestResolveConfig:
    """Tests for resolve_config function."""

    def test_resolve_config_returns_dict(self) -> None:
        """resolve_config converts ElspethSettings to dict."""
        from elspeth.core.config import ElspethSettings, resolve_config

        settings = ElspethSettings(
            datasource={"plugin": "csv", "options": {"path": "input.csv"}},
            sinks={"output": {"plugin": "csv", "options": {"path": "output.csv"}}},
            output_sink="output",
        )

        resolved = resolve_config(settings)

        assert isinstance(resolved, dict)
        assert "datasource" in resolved
        assert resolved["datasource"]["plugin"] == "csv"
        assert "output_sink" in resolved
        assert resolved["output_sink"] == "output"

    def test_resolve_config_includes_defaults(self) -> None:
        """resolve_config includes default values for audit completeness."""
        from elspeth.core.config import ElspethSettings, resolve_config

        settings = ElspethSettings(
            datasource={"plugin": "csv"},
            sinks={"output": {"plugin": "csv"}},
            output_sink="output",
        )

        resolved = resolve_config(settings)

        # Should include defaults
        assert "landscape" in resolved
        assert resolved["landscape"]["enabled"] is True
        assert "concurrency" in resolved
        assert resolved["concurrency"]["max_workers"] == 4
        assert "retry" in resolved
        assert resolved["retry"]["max_attempts"] == 3

    def test_resolve_config_json_serializable(self) -> None:
        """resolve_config output is JSON-serializable for Landscape storage."""
        import json

        from elspeth.core.config import ElspethSettings, resolve_config

        settings = ElspethSettings(
            datasource={"plugin": "csv", "options": {"path": "input.csv"}},
            sinks={"output": {"plugin": "csv", "options": {"path": "output.csv"}}},
            output_sink="output",
        )

        resolved = resolve_config(settings)

        # Should not raise - must be JSON serializable
        json_str = json.dumps(resolved)
        assert isinstance(json_str, str)
        assert len(json_str) > 0

    def test_resolve_config_preserves_row_plugins(self) -> None:
        """resolve_config includes row_plugins configuration."""
        from elspeth.core.config import ElspethSettings, resolve_config

        settings = ElspethSettings(
            datasource={"plugin": "csv"},
            sinks={"output": {"plugin": "csv"}},
            output_sink="output",
            row_plugins=[
                {
                    "plugin": "field_mapper",
                    "type": "transform",
                    "options": {"mapping": {"a": "b"}},
                },
            ],
        )

        resolved = resolve_config(settings)

        assert "row_plugins" in resolved
        assert len(resolved["row_plugins"]) == 1
        assert resolved["row_plugins"][0]["plugin"] == "field_mapper"


class TestGateSettings:
    """Tests for engine-level gate configuration (WP-09)."""

    def test_gate_settings_minimal_valid(self) -> None:
        """GateSettings with required fields only."""
        from elspeth.core.config import GateSettings

        gate = GateSettings(
            name="quality_check",
            condition="row['confidence'] >= 0.85",
            routes={"pass": "continue", "fail": "review_sink"},
        )
        assert gate.name == "quality_check"
        assert gate.condition == "row['confidence'] >= 0.85"
        assert gate.routes == {"pass": "continue", "fail": "review_sink"}
        assert gate.fork_to is None

    def test_gate_settings_with_fork(self) -> None:
        """GateSettings with fork_to for parallel paths."""
        from elspeth.core.config import GateSettings

        gate = GateSettings(
            name="parallel_analysis",
            condition="True",
            routes={"all": "fork"},
            fork_to=["path_a", "path_b"],
        )
        assert gate.name == "parallel_analysis"
        assert gate.routes == {"all": "fork"}
        assert gate.fork_to == ["path_a", "path_b"]

    def test_gate_settings_frozen(self) -> None:
        """GateSettings is immutable."""
        from elspeth.core.config import GateSettings

        gate = GateSettings(
            name="test_gate",
            condition="row['x'] > 0",
            routes={"yes": "continue"},
        )
        with pytest.raises(ValidationError):
            gate.name = "other"  # type: ignore[misc]

    def test_gate_settings_invalid_condition_syntax(self) -> None:
        """Condition must be valid Python syntax."""
        from elspeth.core.config import GateSettings

        with pytest.raises(ValidationError) as exc_info:
            GateSettings(
                name="bad_gate",
                condition="row['x' >=",  # Invalid syntax
                routes={"yes": "continue"},
            )
        assert "Invalid condition syntax" in str(exc_info.value)

    def test_gate_settings_forbidden_condition_construct(self) -> None:
        """Condition must not contain forbidden constructs."""
        from elspeth.core.config import GateSettings

        # Lambda is forbidden
        with pytest.raises(ValidationError) as exc_info:
            GateSettings(
                name="bad_gate",
                condition="(lambda x: x)(row['field'])",
                routes={"yes": "continue"},
            )
        assert "Forbidden construct" in str(exc_info.value)

    def test_gate_settings_forbidden_name_in_condition(self) -> None:
        """Condition must only use 'row' as a name."""
        from elspeth.core.config import GateSettings

        with pytest.raises(ValidationError) as exc_info:
            GateSettings(
                name="bad_gate",
                condition="os.system('rm -rf /')",  # Forbidden name
                routes={"yes": "continue"},
            )
        assert "Forbidden" in str(exc_info.value)

    def test_gate_settings_empty_routes(self) -> None:
        """Routes must have at least one entry."""
        from elspeth.core.config import GateSettings

        with pytest.raises(ValidationError) as exc_info:
            GateSettings(
                name="bad_gate",
                condition="row['x'] > 0",
                routes={},
            )
        assert "at least one entry" in str(exc_info.value)

    def test_gate_settings_invalid_route_destination(self) -> None:
        """Route destination must be 'continue', 'fork', or valid identifier."""
        from elspeth.core.config import GateSettings

        with pytest.raises(ValidationError) as exc_info:
            GateSettings(
                name="bad_gate",
                condition="row['x'] > 0",
                routes={"yes": "123invalid"},  # Starts with number
            )
        assert "valid identifier" in str(exc_info.value)

    def test_gate_settings_route_destination_special_chars(self) -> None:
        """Route destination cannot have special characters."""
        from elspeth.core.config import GateSettings

        with pytest.raises(ValidationError) as exc_info:
            GateSettings(
                name="bad_gate",
                condition="row['x'] > 0",
                routes={"yes": "sink-name"},  # Has hyphen
            )
        assert "valid identifier" in str(exc_info.value)

    def test_gate_settings_fork_requires_fork_to(self) -> None:
        """fork_to is required when routes use 'fork' destination."""
        from elspeth.core.config import GateSettings

        with pytest.raises(ValidationError) as exc_info:
            GateSettings(
                name="bad_gate",
                condition="True",
                routes={"all": "fork"},
                # Missing fork_to
            )
        assert "fork_to is required" in str(exc_info.value)

    def test_gate_settings_fork_to_requires_fork_route(self) -> None:
        """fork_to is only valid when a route destination is 'fork'."""
        from elspeth.core.config import GateSettings

        with pytest.raises(ValidationError) as exc_info:
            GateSettings(
                name="bad_gate",
                condition="row['x'] > 0",
                routes={"yes": "continue"},
                fork_to=["path_a", "path_b"],  # No fork route
            )
        assert "fork_to is only valid" in str(exc_info.value)

    def test_gate_settings_valid_identifiers(self) -> None:
        """Valid identifier sink names are accepted."""
        from elspeth.core.config import GateSettings

        gate = GateSettings(
            name="multi_route",
            condition="row['category'] in ['a', 'b', 'c']",
            routes={
                "a": "sink_a",
                "b": "Sink_B",
                "c": "_private_sink",
                "d": "continue",
            },
        )
        assert len(gate.routes) == 4

    def test_gate_settings_complex_condition(self) -> None:
        """Complex expressions are validated correctly."""
        from elspeth.core.config import GateSettings

        gate = GateSettings(
            name="complex_gate",
            condition="row['confidence'] >= 0.85 and row.get('category', 'unknown') != 'spam'",
            routes={"pass": "continue", "fail": "quarantine"},
        )
        assert "and" in gate.condition


class TestElspethSettingsWithGates:
    """Tests for ElspethSettings with engine-level gates."""

    def test_elspeth_settings_gates_default_empty(self) -> None:
        """Gates defaults to empty list."""
        from elspeth.core.config import ElspethSettings

        settings = ElspethSettings(
            datasource={"plugin": "csv"},
            sinks={"output": {"plugin": "csv"}},
            output_sink="output",
        )
        assert settings.gates == []

    def test_elspeth_settings_with_gates(self) -> None:
        """ElspethSettings accepts gates configuration."""
        from elspeth.core.config import ElspethSettings

        settings = ElspethSettings(
            datasource={"plugin": "csv"},
            sinks={"output": {"plugin": "csv"}, "review": {"plugin": "csv"}},
            output_sink="output",
            gates=[
                {
                    "name": "quality_check",
                    "condition": "row['confidence'] >= 0.85",
                    "routes": {"pass": "continue", "fail": "review"},
                },
            ],
        )
        assert len(settings.gates) == 1
        assert settings.gates[0].name == "quality_check"

    def test_elspeth_settings_multiple_gates(self) -> None:
        """ElspethSettings accepts multiple gates."""
        from elspeth.core.config import ElspethSettings

        settings = ElspethSettings(
            datasource={"plugin": "csv"},
            sinks={"output": {"plugin": "csv"}},
            output_sink="output",
            gates=[
                {
                    "name": "gate_1",
                    "condition": "row['x'] > 0",
                    "routes": {"yes": "continue"},
                },
                {
                    "name": "gate_2",
                    "condition": "row['y'] < 100",
                    "routes": {"yes": "continue"},
                },
            ],
        )
        assert len(settings.gates) == 2
        assert settings.gates[0].name == "gate_1"
        assert settings.gates[1].name == "gate_2"

    def test_resolve_config_includes_gates(self) -> None:
        """resolve_config preserves gates configuration."""
        from elspeth.core.config import ElspethSettings, resolve_config

        settings = ElspethSettings(
            datasource={"plugin": "csv"},
            sinks={"output": {"plugin": "csv"}},
            output_sink="output",
            gates=[
                {
                    "name": "quality_check",
                    "condition": "row['confidence'] >= 0.85",
                    "routes": {"pass": "continue", "fail": "output"},
                },
            ],
        )

        resolved = resolve_config(settings)

        assert "gates" in resolved
        assert len(resolved["gates"]) == 1
        assert resolved["gates"][0]["name"] == "quality_check"
        assert resolved["gates"][0]["condition"] == "row['confidence'] >= 0.85"


class TestLoadSettingsWithGates:
    """Tests for loading YAML with gates configuration."""

    def test_load_settings_with_gates(self, tmp_path: Path) -> None:
        """Load YAML with gates section."""
        from elspeth.core.config import load_settings

        config_file = tmp_path / "settings.yaml"
        config_file.write_text("""
datasource:
  plugin: csv

sinks:
  output:
    plugin: csv
  review:
    plugin: csv

output_sink: output

gates:
  - name: quality_check
    condition: "row['confidence'] >= 0.85"
    routes:
      pass: continue
      fail: review
""")
        settings = load_settings(config_file)

        assert len(settings.gates) == 1
        assert settings.gates[0].name == "quality_check"
        assert settings.gates[0].condition == "row['confidence'] >= 0.85"
        assert settings.gates[0].routes == {"pass": "continue", "fail": "review"}

    def test_load_settings_with_fork_gate(self, tmp_path: Path) -> None:
        """Load YAML with fork gate."""
        from elspeth.core.config import load_settings

        config_file = tmp_path / "settings.yaml"
        config_file.write_text("""
datasource:
  plugin: csv

sinks:
  output:
    plugin: csv

output_sink: output

gates:
  - name: parallel_analysis
    condition: "True"
    routes:
      all: fork
    fork_to:
      - path_a
      - path_b
""")
        settings = load_settings(config_file)

        assert len(settings.gates) == 1
        assert settings.gates[0].name == "parallel_analysis"
        assert settings.gates[0].routes == {"all": "fork"}
        assert settings.gates[0].fork_to == ["path_a", "path_b"]


class TestCoalesceSettings:
    """Test CoalesceSettings configuration model."""

    def test_coalesce_settings_basic(self) -> None:
        """Basic coalesce configuration should validate."""
        from elspeth.core.config import CoalesceSettings

        settings = CoalesceSettings(
            name="merge_results",
            branches=["path_a", "path_b"],
            policy="require_all",
            merge="union",
        )

        assert settings.name == "merge_results"
        assert settings.branches == ["path_a", "path_b"]
        assert settings.policy == "require_all"
        assert settings.merge == "union"
        assert settings.timeout_seconds is None
        assert settings.quorum_count is None

    def test_coalesce_settings_quorum_requires_count(self) -> None:
        """Quorum policy requires quorum_count."""
        from elspeth.core.config import CoalesceSettings

        with pytest.raises(ValidationError, match="quorum_count"):
            CoalesceSettings(
                name="quorum_merge",
                branches=["a", "b", "c"],
                policy="quorum",
                merge="union",
                # Missing quorum_count
            )

    def test_coalesce_settings_quorum_with_count(self) -> None:
        """Quorum policy with count should validate."""
        from elspeth.core.config import CoalesceSettings

        settings = CoalesceSettings(
            name="quorum_merge",
            branches=["a", "b", "c"],
            policy="quorum",
            merge="union",
            quorum_count=2,
        )

        assert settings.quorum_count == 2

    def test_coalesce_settings_best_effort_requires_timeout(self) -> None:
        """Best effort policy requires timeout."""
        from elspeth.core.config import CoalesceSettings

        with pytest.raises(ValidationError, match="timeout"):
            CoalesceSettings(
                name="best_effort_merge",
                branches=["a", "b"],
                policy="best_effort",
                merge="union",
                # Missing timeout_seconds
            )

    def test_coalesce_settings_nested_merge_strategy(self) -> None:
        """Nested merge strategy should validate."""
        from elspeth.core.config import CoalesceSettings

        settings = CoalesceSettings(
            name="nested_merge",
            branches=["sentiment", "entities"],
            policy="require_all",
            merge="nested",
        )

        assert settings.merge == "nested"

    def test_coalesce_settings_select_merge_strategy(self) -> None:
        """Select merge requires select_branch."""
        from elspeth.core.config import CoalesceSettings

        with pytest.raises(ValidationError, match="select_branch"):
            CoalesceSettings(
                name="select_merge",
                branches=["a", "b"],
                policy="require_all",
                merge="select",
                # Missing select_branch
            )

    def test_coalesce_settings_select_with_branch(self) -> None:
        """Select merge with branch should validate."""
        from elspeth.core.config import CoalesceSettings

        settings = CoalesceSettings(
            name="select_merge",
            branches=["primary", "fallback"],
            policy="require_all",
            merge="select",
            select_branch="primary",
        )

        assert settings.select_branch == "primary"

    def test_coalesce_settings_quorum_count_cannot_exceed_branches(self) -> None:
        """Quorum count cannot exceed number of branches."""
        from elspeth.core.config import CoalesceSettings

        with pytest.raises(ValidationError, match="cannot exceed"):
            CoalesceSettings(
                name="quorum_merge",
                branches=["a", "b"],
                policy="quorum",
                merge="union",
                quorum_count=3,  # More than 2 branches
            )

    def test_coalesce_settings_select_branch_must_be_in_branches(self) -> None:
        """Select branch must be one of the defined branches."""
        from elspeth.core.config import CoalesceSettings

        with pytest.raises(ValidationError, match="must be one of"):
            CoalesceSettings(
                name="select_merge",
                branches=["a", "b"],
                policy="require_all",
                merge="select",
                select_branch="c",  # Not in branches
            )

    def test_coalesce_settings_branches_minimum_two(self) -> None:
        """Branches must have at least 2 entries."""
        from elspeth.core.config import CoalesceSettings

        with pytest.raises(ValidationError):
            CoalesceSettings(
                name="single_branch",
                branches=["only_one"],
                policy="require_all",
                merge="union",
            )

    def test_coalesce_settings_timeout_must_be_positive(self) -> None:
        """Timeout must be positive when provided."""
        from elspeth.core.config import CoalesceSettings

        with pytest.raises(ValidationError):
            CoalesceSettings(
                name="bad_timeout",
                branches=["a", "b"],
                policy="best_effort",
                merge="union",
                timeout_seconds=0,
            )

    def test_coalesce_settings_quorum_count_must_be_positive(self) -> None:
        """Quorum count must be positive when provided."""
        from elspeth.core.config import CoalesceSettings

        with pytest.raises(ValidationError):
            CoalesceSettings(
                name="bad_quorum",
                branches=["a", "b", "c"],
                policy="quorum",
                merge="union",
                quorum_count=0,
            )

    def test_coalesce_settings_frozen(self) -> None:
        """CoalesceSettings is immutable."""
        from elspeth.core.config import CoalesceSettings

        settings = CoalesceSettings(
            name="merge_results",
            branches=["a", "b"],
            policy="require_all",
            merge="union",
        )
        with pytest.raises(ValidationError):
            settings.name = "other"  # type: ignore[misc]

    def test_coalesce_settings_first_policy(self) -> None:
        """First policy should validate without additional requirements."""
        from elspeth.core.config import CoalesceSettings

        settings = CoalesceSettings(
            name="first_wins",
            branches=["fast_model", "slow_model"],
            policy="first",
            merge="union",
        )

        assert settings.policy == "first"

    def test_coalesce_settings_best_effort_with_timeout(self) -> None:
        """Best effort policy with timeout should validate."""
        from elspeth.core.config import CoalesceSettings

        settings = CoalesceSettings(
            name="best_effort_merge",
            branches=["a", "b"],
            policy="best_effort",
            merge="union",
            timeout_seconds=30.0,
        )

        assert settings.timeout_seconds == 30.0

    def test_coalesce_settings_timeout_negative_rejected(self) -> None:
        """Negative timeout values should be rejected."""
        from elspeth.core.config import CoalesceSettings

        with pytest.raises(ValidationError):
            CoalesceSettings(
                name="test",
                branches=["branch_a", "branch_b"],
                policy="best_effort",
                merge="union",
                timeout_seconds=-1.0,
            )

    def test_coalesce_settings_quorum_count_negative_rejected(self) -> None:
        """Negative quorum count should be rejected."""
        from elspeth.core.config import CoalesceSettings

        with pytest.raises(ValidationError):
            CoalesceSettings(
                name="test",
                branches=["branch_a", "branch_b"],
                policy="quorum",
                merge="union",
                quorum_count=-1,
            )
