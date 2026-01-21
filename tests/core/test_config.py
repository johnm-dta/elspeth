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
            ds.plugin = "json"  # type: ignore[misc]


class TestRowPluginSettings:
    """RowPluginSettings matches architecture specification."""

    def test_row_plugin_settings_structure(self) -> None:
        """RowPluginSettings has plugin and options (transforms only)."""
        from elspeth.core.config import RowPluginSettings

        # RowPluginSettings is now transform-only (gates are config-driven)
        rp = RowPluginSettings(
            plugin="field_mapper",
            options={"field": "confidence", "min": 0.8},
        )
        assert rp.plugin == "field_mapper"
        assert rp.options == {"field": "confidence", "min": 0.8}

    def test_row_plugin_settings_defaults(self) -> None:
        """RowPluginSettings defaults: options is empty dict."""
        from elspeth.core.config import RowPluginSettings

        rp = RowPluginSettings(plugin="passthrough")
        assert rp.plugin == "passthrough"
        assert rp.options == {}

    def test_row_plugin_settings_requires_plugin(self) -> None:
        """Plugin name is required."""
        from elspeth.core.config import RowPluginSettings

        with pytest.raises(ValidationError):
            RowPluginSettings(options={})  # type: ignore[call-arg]


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
        """Load config with config-driven gates."""
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

gates:
  - name: safety_check
    condition: "row['suspicious'] == True"
    routes:
      "true": flagged
      "false": continue

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
        assert len(settings.gates) == 1
        assert settings.gates[0].name == "safety_check"
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
            routes={"true": "continue", "false": "review_sink"},
        )
        assert gate.name == "quality_check"
        assert gate.condition == "row['confidence'] >= 0.85"
        assert gate.routes == {"true": "continue", "false": "review_sink"}
        assert gate.fork_to is None

    def test_gate_settings_with_fork(self) -> None:
        """GateSettings with fork_to for parallel paths."""
        from elspeth.core.config import GateSettings

        gate = GateSettings(
            name="parallel_analysis",
            condition="True",
            routes={"true": "fork", "false": "continue"},
            fork_to=["path_a", "path_b"],
        )
        assert gate.name == "parallel_analysis"
        assert gate.routes == {"true": "fork", "false": "continue"}
        assert gate.fork_to == ["path_a", "path_b"]

    def test_gate_settings_frozen(self) -> None:
        """GateSettings is immutable."""
        from elspeth.core.config import GateSettings

        gate = GateSettings(
            name="test_gate",
            condition="row['x'] > 0",
            routes={"true": "continue", "false": "reject_sink"},
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

    def test_gate_settings_hyphenated_sink_destination_accepted(self) -> None:
        """Route destination can be any sink name, including hyphenated.

        Regression test for gate-route-destination-name-validation-mismatch bug.
        Sink names don't need to match identifier pattern - they're just dict keys.
        The DAG builder validates that destinations are actual sink keys.
        """
        from elspeth.core.config import GateSettings

        # Hyphenated sink names should be accepted at GateSettings level
        gate = GateSettings(
            name="hyphen_gate",
            condition="row['x'] > 0",
            routes={"true": "output-sink", "false": "continue"},
        )
        assert gate.routes["true"] == "output-sink"

    def test_gate_settings_numeric_prefix_sink_destination_accepted(self) -> None:
        """Route destination can start with numbers (not identifier-constrained).

        Sink names are dict keys, not Python identifiers. Names like "123_sink"
        are valid sink names and should be accepted in gate routes.
        """
        from elspeth.core.config import GateSettings

        gate = GateSettings(
            name="numeric_gate",
            condition="row['x'] > 0",
            routes={"true": "123_sink", "false": "continue"},
        )
        assert gate.routes["true"] == "123_sink"

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

        # Non-boolean condition allows custom route labels
        gate = GateSettings(
            name="multi_route",
            condition="row['category']",  # Returns string, not boolean
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
            routes={"true": "continue", "false": "quarantine"},
        )
        assert "and" in gate.condition

    def test_gate_settings_reserved_route_label_rejected(self) -> None:
        """Route label 'continue' is reserved and must be rejected.

        Using 'continue' as a route label would cause edge_map collisions
        in the orchestrator (the DAG builder already uses 'continue' for
        edges between sequential nodes), leading to routing events being
        recorded against the wrong edge - corrupting the audit trail.
        """
        from elspeth.core.config import GateSettings

        with pytest.raises(ValidationError) as exc_info:
            GateSettings(
                name="bad_gate",
                condition="row['score'] >= 0.5",
                routes={"continue": "some_sink"},  # 'continue' as label is forbidden
            )
        assert "reserved" in str(exc_info.value).lower()
        assert "continue" in str(exc_info.value)

    def test_gate_settings_reserved_fork_branch_rejected(self) -> None:
        """Fork branch 'continue' is reserved and must be rejected.

        Fork branches become edge labels, so using 'continue' would cause
        the same edge_map collision issue as reserved route labels.
        """
        from elspeth.core.config import GateSettings

        with pytest.raises(ValidationError) as exc_info:
            GateSettings(
                name="bad_fork_gate",
                condition="True",
                routes={"all": "fork"},
                fork_to=["path_a", "continue"],  # 'continue' as branch is forbidden
            )
        assert "reserved" in str(exc_info.value).lower()
        assert "continue" in str(exc_info.value)

    def test_gate_settings_continue_as_destination_allowed(self) -> None:
        """'continue' as a route DESTINATION (not label) is still valid.

        The restriction is on route LABELS (dict keys), not destinations.
        'continue' as a destination means "proceed to next node" which is
        the expected semantic.
        """
        from elspeth.core.config import GateSettings

        # This should NOT raise - 'continue' is the destination, not the label
        gate = GateSettings(
            name="valid_gate",
            condition="row['valid']",
            routes={"pass": "continue", "fail": "error_sink"},
        )
        assert gate.routes["pass"] == "continue"

    def test_gate_settings_boolean_condition_requires_true_false_routes(self) -> None:
        """Boolean conditions must use 'true'/'false' route labels.

        A condition like `row['amount'] > 1000` evaluates to True or False,
        not 'above' or 'below'. Using arbitrary labels is a config error.
        """
        from elspeth.core.config import GateSettings

        # This should fail - using 'above'/'below' for a boolean condition
        with pytest.raises(ValidationError) as exc_info:
            GateSettings(
                name="threshold_gate",
                condition="row['amount'] > 1000",
                routes={"above": "high_sink", "below": "continue"},
            )
        error_msg = str(exc_info.value)
        assert "boolean condition" in error_msg
        assert "true" in error_msg.lower()
        assert "false" in error_msg.lower()

    def test_gate_settings_non_boolean_allows_custom_routes(self) -> None:
        """Non-boolean conditions (field access, ternary) allow custom labels.

        A condition like `row['category']` returns the field value directly,
        so routes can be labeled with expected values like 'high'/'low'.
        """
        from elspeth.core.config import GateSettings

        # Field access returns string, not boolean
        gate = GateSettings(
            name="category_router",
            condition="row['priority']",
            routes={"high": "urgent_sink", "medium": "continue", "low": "archive_sink"},
        )
        assert len(gate.routes) == 3

        # Ternary returns the branch value, not boolean
        gate2 = GateSettings(
            name="ternary_router",
            condition="'high' if row['score'] > 0.8 else 'low'",
            routes={"high": "priority_sink", "low": "continue"},
        )
        assert gate2.condition.startswith("'high'")


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
                    "routes": {"true": "continue", "false": "review"},
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
                    "routes": {"true": "continue", "false": "continue"},
                },
                {
                    "name": "gate_2",
                    "condition": "row['y'] < 100",
                    "routes": {"true": "continue", "false": "continue"},
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
                    "routes": {"true": "continue", "false": "output"},
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
      "true": continue
      "false": review
""")
        settings = load_settings(config_file)

        assert len(settings.gates) == 1
        assert settings.gates[0].name == "quality_check"
        assert settings.gates[0].condition == "row['confidence'] >= 0.85"
        assert settings.gates[0].routes == {"true": "continue", "false": "review"}

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
      "true": fork
      "false": continue
    fork_to:
      - path_a
      - path_b
""")
        settings = load_settings(config_file)

        assert len(settings.gates) == 1
        assert settings.gates[0].name == "parallel_analysis"
        assert settings.gates[0].routes == {"true": "fork", "false": "continue"}
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


class TestElspethSettingsWithCoalesce:
    """Tests for ElspethSettings with coalesce configuration."""

    def test_elspeth_settings_with_coalesce(self) -> None:
        """ElspethSettings should accept coalesce configuration."""
        from elspeth.core.config import (
            CoalesceSettings,
            DatasourceSettings,
            ElspethSettings,
            SinkSettings,
        )

        settings = ElspethSettings(
            datasource=DatasourceSettings(plugin="csv_local", options={"path": "test.csv"}),
            sinks={"default": SinkSettings(plugin="csv", options={"path": "out.csv"})},
            output_sink="default",
            coalesce=[
                CoalesceSettings(
                    name="merge_results",
                    branches=["path_a", "path_b"],
                    policy="require_all",
                    merge="union",
                ),
            ],
        )

        assert len(settings.coalesce) == 1
        assert settings.coalesce[0].name == "merge_results"

    def test_elspeth_settings_coalesce_default_empty(self) -> None:
        """Coalesce defaults to empty list."""
        from elspeth.core.config import ElspethSettings

        settings = ElspethSettings(
            datasource={"plugin": "csv"},
            sinks={"output": {"plugin": "csv"}},
            output_sink="output",
        )
        assert settings.coalesce == []

    def test_resolve_config_includes_coalesce(self) -> None:
        """resolve_config preserves coalesce configuration."""
        from elspeth.core.config import ElspethSettings, resolve_config

        settings = ElspethSettings(
            datasource={"plugin": "csv"},
            sinks={"output": {"plugin": "csv"}},
            output_sink="output",
            coalesce=[
                {
                    "name": "merge_results",
                    "branches": ["path_a", "path_b"],
                    "policy": "require_all",
                    "merge": "union",
                },
            ],
        )

        resolved = resolve_config(settings)

        assert "coalesce" in resolved
        assert len(resolved["coalesce"]) == 1
        assert resolved["coalesce"][0]["name"] == "merge_results"


class TestSecretFieldFingerprinting:
    """Test that secret fields are preserved at load time and fingerprinted for audit.

    IMPORTANT: Secrets are kept in ElspethSettings for runtime use (transforms
    need actual credentials). Fingerprinting happens only in resolve_config()
    when creating the audit copy for Landscape storage.
    """

    def test_api_key_preserved_at_load_time(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        """API keys in config should be preserved for runtime use."""
        from elspeth.core.config import load_settings

        monkeypatch.setenv("ELSPETH_FINGERPRINT_KEY", "test-key")

        config_file = tmp_path / "settings.yaml"
        config_file.write_text("""
datasource:
  plugin: http_source
  options:
    api_key: sk-secret-key-12345
    url: https://api.example.com
sinks:
  output:
    plugin: csv_sink
    options:
      path: output.csv
output_sink: output
""")

        settings = load_settings(config_file)

        # API key should be preserved for runtime (transforms need it!)
        assert "api_key" in settings.datasource.options
        assert settings.datasource.options["api_key"] == "sk-secret-key-12345"

    def test_api_key_is_fingerprinted_in_resolve_config(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        """API keys should be fingerprinted when creating audit copy."""
        from elspeth.core.config import load_settings, resolve_config

        monkeypatch.setenv("ELSPETH_FINGERPRINT_KEY", "test-key")

        config_file = tmp_path / "settings.yaml"
        config_file.write_text("""
datasource:
  plugin: http_source
  options:
    api_key: sk-secret-key-12345
    url: https://api.example.com
sinks:
  output:
    plugin: csv_sink
    options:
      path: output.csv
output_sink: output
""")

        settings = load_settings(config_file)
        audit_config = resolve_config(settings)

        # API key should be removed in audit copy
        assert "api_key" not in audit_config["datasource"]["options"]
        # Should have a 64-char hex fingerprint
        fingerprint = audit_config["datasource"]["options"].get("api_key_fingerprint")
        assert fingerprint is not None
        assert len(fingerprint) == 64
        assert all(c in "0123456789abcdef" for c in fingerprint)

    def test_token_preserved_at_load_time(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        """Token fields should be preserved for runtime use."""
        from elspeth.core.config import load_settings

        monkeypatch.setenv("ELSPETH_FINGERPRINT_KEY", "test-key")

        config_file = tmp_path / "settings.yaml"
        config_file.write_text("""
datasource:
  plugin: webhook_source
  options:
    token: bearer-token-xyz
sinks:
  output:
    plugin: csv_sink
output_sink: output
""")

        settings = load_settings(config_file)

        assert "token" in settings.datasource.options
        assert settings.datasource.options["token"] == "bearer-token-xyz"

    def test_token_is_fingerprinted_in_resolve_config(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        """Token fields should be fingerprinted in audit copy."""
        from elspeth.core.config import load_settings, resolve_config

        monkeypatch.setenv("ELSPETH_FINGERPRINT_KEY", "test-key")

        config_file = tmp_path / "settings.yaml"
        config_file.write_text("""
datasource:
  plugin: webhook_source
  options:
    token: bearer-token-xyz
sinks:
  output:
    plugin: csv_sink
output_sink: output
""")

        settings = load_settings(config_file)
        audit_config = resolve_config(settings)

        assert "token" not in audit_config["datasource"]["options"]
        assert "token_fingerprint" in audit_config["datasource"]["options"]

    def test_secret_suffix_preserved_at_load_time(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        """Fields ending in _secret should be preserved for runtime."""
        from elspeth.core.config import load_settings

        monkeypatch.setenv("ELSPETH_FINGERPRINT_KEY", "test-key")

        config_file = tmp_path / "settings.yaml"
        config_file.write_text("""
datasource:
  plugin: custom_source
  options:
    database_secret: my-db-password
sinks:
  output:
    plugin: csv_sink
output_sink: output
""")

        settings = load_settings(config_file)

        assert "database_secret" in settings.datasource.options
        assert settings.datasource.options["database_secret"] == "my-db-password"

    def test_secret_suffix_is_fingerprinted_in_resolve_config(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        """Fields ending in _secret should be fingerprinted in audit copy."""
        from elspeth.core.config import load_settings, resolve_config

        monkeypatch.setenv("ELSPETH_FINGERPRINT_KEY", "test-key")

        config_file = tmp_path / "settings.yaml"
        config_file.write_text("""
datasource:
  plugin: custom_source
  options:
    database_secret: my-db-password
sinks:
  output:
    plugin: csv_sink
output_sink: output
""")

        settings = load_settings(config_file)
        audit_config = resolve_config(settings)

        assert "database_secret" not in audit_config["datasource"]["options"]
        assert "database_secret_fingerprint" in audit_config["datasource"]["options"]

    def test_sink_options_preserved_at_load_time(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        """Secret fields in sink options should be preserved for runtime."""
        from elspeth.core.config import load_settings

        monkeypatch.setenv("ELSPETH_FINGERPRINT_KEY", "test-key")

        config_file = tmp_path / "settings.yaml"
        config_file.write_text("""
datasource:
  plugin: csv_source
sinks:
  output:
    plugin: database_sink
    options:
      password: super-secret-password
output_sink: output
""")

        settings = load_settings(config_file)

        assert "password" in settings.sinks["output"].options
        assert settings.sinks["output"].options["password"] == "super-secret-password"

    def test_sink_options_are_fingerprinted_in_resolve_config(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        """Secret fields in sink options should be fingerprinted in audit copy."""
        from elspeth.core.config import load_settings, resolve_config

        monkeypatch.setenv("ELSPETH_FINGERPRINT_KEY", "test-key")

        config_file = tmp_path / "settings.yaml"
        config_file.write_text("""
datasource:
  plugin: csv_source
sinks:
  output:
    plugin: database_sink
    options:
      password: super-secret-password
output_sink: output
""")

        settings = load_settings(config_file)
        audit_config = resolve_config(settings)

        assert "password" not in audit_config["sinks"]["output"]["options"]
        assert "password_fingerprint" in audit_config["sinks"]["output"]["options"]

    def test_non_secret_fields_preserved(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        """Non-secret fields should remain unchanged."""
        from elspeth.core.config import load_settings

        monkeypatch.setenv("ELSPETH_FINGERPRINT_KEY", "test-key")

        config_file = tmp_path / "settings.yaml"
        config_file.write_text("""
datasource:
  plugin: csv_source
  options:
    path: input.csv
    delimiter: ","
sinks:
  output:
    plugin: csv_sink
    options:
      path: output.csv
output_sink: output
""")

        settings = load_settings(config_file)

        # Regular fields should be preserved
        assert settings.datasource.options["path"] == "input.csv"
        assert settings.datasource.options["delimiter"] == ","

    def test_row_plugin_options_preserved_at_load_time(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        """Secret fields in row_plugins options should be preserved for runtime."""
        from elspeth.core.config import load_settings

        monkeypatch.setenv("ELSPETH_FINGERPRINT_KEY", "test-key")

        config_file = tmp_path / "settings.yaml"
        config_file.write_text("""
datasource:
  plugin: csv_source
sinks:
  output:
    plugin: csv_sink
output_sink: output
row_plugins:
  - plugin: llm_transform
    options:
      api_key: openai-key-123
      model: gpt-4
""")

        settings = load_settings(config_file)

        # Secrets preserved for runtime
        assert "api_key" in settings.row_plugins[0].options
        assert settings.row_plugins[0].options["api_key"] == "openai-key-123"
        # Non-secret field preserved
        assert settings.row_plugins[0].options["model"] == "gpt-4"

    def test_row_plugin_options_are_fingerprinted_in_resolve_config(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        """Secret fields in row_plugins options should be fingerprinted in audit copy."""
        from elspeth.core.config import load_settings, resolve_config

        monkeypatch.setenv("ELSPETH_FINGERPRINT_KEY", "test-key")

        config_file = tmp_path / "settings.yaml"
        config_file.write_text("""
datasource:
  plugin: csv_source
sinks:
  output:
    plugin: csv_sink
output_sink: output
row_plugins:
  - plugin: llm_transform
    options:
      api_key: openai-key-123
      model: gpt-4
""")

        settings = load_settings(config_file)
        audit_config = resolve_config(settings)

        assert "api_key" not in audit_config["row_plugins"][0]["options"]
        assert "api_key_fingerprint" in audit_config["row_plugins"][0]["options"]
        # Non-secret field preserved
        assert audit_config["row_plugins"][0]["options"]["model"] == "gpt-4"

    # === Tests for recursive/nested fingerprinting ===

    def test_nested_secrets_are_fingerprinted(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Secrets in nested dicts should be fingerprinted."""
        from elspeth.core.config import _fingerprint_secrets

        monkeypatch.setenv("ELSPETH_FINGERPRINT_KEY", "test-key")

        options = {
            "api_key": "sk-top-level",
            "auth": {
                "api_key": "sk-nested",
                "nested": {"token": "nested-token"},
            },
        }

        result = _fingerprint_secrets(options)

        # Top-level secret fingerprinted
        assert "api_key" not in result
        assert "api_key_fingerprint" in result

        # Nested secrets fingerprinted
        assert "api_key" not in result["auth"]
        assert "api_key_fingerprint" in result["auth"]
        assert "token" not in result["auth"]["nested"]
        assert "token_fingerprint" in result["auth"]["nested"]

    def test_secrets_in_lists_are_fingerprinted(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Secrets inside list elements should be fingerprinted."""
        from elspeth.core.config import _fingerprint_secrets

        monkeypatch.setenv("ELSPETH_FINGERPRINT_KEY", "test-key")

        options = {
            "providers": [
                {"name": "openai", "api_key": "sk-openai"},
                {"name": "anthropic", "api_key": "sk-anthropic"},
            ]
        }

        result = _fingerprint_secrets(options)

        for provider in result["providers"]:
            assert "api_key" not in provider
            assert "api_key_fingerprint" in provider

    # === Tests for fail-closed behavior ===

    def test_missing_key_raises_error_on_fingerprint(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Missing fingerprint key should raise SecretFingerprintError."""
        from elspeth.core.config import SecretFingerprintError, _fingerprint_secrets

        monkeypatch.delenv("ELSPETH_FINGERPRINT_KEY", raising=False)
        monkeypatch.delenv("ELSPETH_ALLOW_RAW_SECRETS", raising=False)

        options = {"api_key": "sk-secret"}

        with pytest.raises(SecretFingerprintError) as exc_info:
            _fingerprint_secrets(options, fail_if_no_key=True)

        assert "ELSPETH_FINGERPRINT_KEY" in str(exc_info.value)
        assert "api_key" in str(exc_info.value)

    def test_load_settings_allows_secrets_without_fingerprint_key(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        """load_settings should allow secrets without fingerprint key (for runtime).

        Secrets are preserved at load time for transforms to use. The
        fingerprint key is only required when calling resolve_config()
        to create the audit copy.
        """
        from elspeth.core.config import load_settings

        monkeypatch.delenv("ELSPETH_FINGERPRINT_KEY", raising=False)
        monkeypatch.delenv("ELSPETH_ALLOW_RAW_SECRETS", raising=False)

        config_file = tmp_path / "settings.yaml"
        config_file.write_text("""
datasource:
  plugin: http_source
  options:
    api_key: sk-secret-key
sinks:
  output:
    plugin: csv_sink
output_sink: output
""")

        # load_settings succeeds even without fingerprint key
        settings = load_settings(config_file)
        assert settings.datasource.options["api_key"] == "sk-secret-key"

    def test_missing_key_raises_error_on_resolve_config(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        """resolve_config should raise SecretFingerprintError when key missing."""
        from elspeth.core.config import (
            SecretFingerprintError,
            load_settings,
            resolve_config,
        )

        monkeypatch.delenv("ELSPETH_FINGERPRINT_KEY", raising=False)
        monkeypatch.delenv("ELSPETH_ALLOW_RAW_SECRETS", raising=False)

        config_file = tmp_path / "settings.yaml"
        config_file.write_text("""
datasource:
  plugin: http_source
  options:
    api_key: sk-secret-key
sinks:
  output:
    plugin: csv_sink
output_sink: output
""")

        settings = load_settings(config_file)

        with pytest.raises(SecretFingerprintError) as exc_info:
            resolve_config(settings)

        assert "ELSPETH_FINGERPRINT_KEY" in str(exc_info.value)

    def test_dev_mode_keeps_secrets_unchanged(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """ELSPETH_ALLOW_RAW_SECRETS=true should keep secrets as-is for dev use."""
        from elspeth.core.config import _fingerprint_secrets

        monkeypatch.delenv("ELSPETH_FINGERPRINT_KEY", raising=False)
        monkeypatch.setenv("ELSPETH_ALLOW_RAW_SECRETS", "true")

        options = {"api_key": "sk-secret"}

        result = _fingerprint_secrets(options, fail_if_no_key=False)

        # In dev mode, secrets are kept unchanged so plugins can use them
        assert result.get("api_key") == "sk-secret"
        assert "api_key_redacted" not in result

    def test_dev_mode_allows_load_settings(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        """ELSPETH_ALLOW_RAW_SECRETS=true should allow load and keep secrets as-is."""
        from elspeth.core.config import load_settings

        monkeypatch.delenv("ELSPETH_FINGERPRINT_KEY", raising=False)
        monkeypatch.setenv("ELSPETH_ALLOW_RAW_SECRETS", "true")

        config_file = tmp_path / "settings.yaml"
        config_file.write_text("""
datasource:
  plugin: http_source
  options:
    api_key: sk-secret-key
sinks:
  output:
    plugin: csv_sink
output_sink: output
""")

        settings = load_settings(config_file)

        # In dev mode, secrets are kept unchanged so plugins can use them
        assert settings.datasource.options.get("api_key") == "sk-secret-key"
        assert "api_key_redacted" not in settings.datasource.options

    # === Tests for DSN password handling ===

    def test_dsn_password_sanitized(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """DSN passwords should be removed and fingerprinted."""
        from elspeth.core.config import _sanitize_dsn

        monkeypatch.setenv("ELSPETH_FINGERPRINT_KEY", "test-key")

        url = "postgresql://user:secret_password@localhost:5432/mydb"
        sanitized, fingerprint, had_password = _sanitize_dsn(url)

        assert "secret_password" not in sanitized
        assert "user@localhost" in sanitized
        assert "***" not in sanitized  # Should NOT have placeholder
        assert fingerprint is not None
        assert len(fingerprint) == 64  # SHA256 hex
        assert had_password is True

    def test_dsn_without_password_unchanged(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """DSN without password should pass through."""
        from elspeth.core.config import _sanitize_dsn

        monkeypatch.setenv("ELSPETH_FINGERPRINT_KEY", "test-key")

        url = "sqlite:///path/to/db.sqlite"
        sanitized, fingerprint, had_password = _sanitize_dsn(url)

        assert sanitized == url
        assert fingerprint is None
        assert had_password is False

    def test_dsn_password_raises_without_key(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """DSN with password should raise when no fingerprint key."""
        from elspeth.core.config import SecretFingerprintError, _sanitize_dsn

        monkeypatch.delenv("ELSPETH_FINGERPRINT_KEY", raising=False)
        monkeypatch.delenv("ELSPETH_ALLOW_RAW_SECRETS", raising=False)

        url = "postgresql://user:secret@localhost/db"

        with pytest.raises(SecretFingerprintError) as exc_info:
            _sanitize_dsn(url, fail_if_no_key=True)

        assert "ELSPETH_FINGERPRINT_KEY" in str(exc_info.value)

    def test_dsn_password_redacted_in_dev_mode(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """DSN password should be removed (not fingerprinted) in dev mode."""
        from elspeth.core.config import _sanitize_dsn

        monkeypatch.delenv("ELSPETH_FINGERPRINT_KEY", raising=False)

        url = "postgresql://user:secret@localhost/db"
        sanitized, fingerprint, had_password = _sanitize_dsn(url, fail_if_no_key=False)

        assert "secret" not in sanitized
        assert fingerprint is None
        assert had_password is True

    def test_landscape_url_password_fingerprinted(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """landscape.url password should be fingerprinted in audit copy."""
        from elspeth.core.config import _fingerprint_config_for_audit

        monkeypatch.setenv("ELSPETH_FINGERPRINT_KEY", "test-key")

        config_dict = {
            "landscape": {"url": "postgresql://user:mysecret@host/db"},
            "datasource": {"plugin": "csv", "options": {}},
            "sinks": {"output": {"plugin": "csv_sink"}},
            "output_sink": "output",
        }

        result = _fingerprint_config_for_audit(config_dict)

        assert "mysecret" not in result["landscape"]["url"]
        assert "***" not in result["landscape"]["url"]
        assert "url_password_fingerprint" in result["landscape"]
        assert len(result["landscape"]["url_password_fingerprint"]) == 64

    def test_landscape_url_password_redacted_in_dev_mode(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """landscape.url password should be redacted (with flag) in dev mode."""
        from elspeth.core.config import _fingerprint_config_for_audit

        monkeypatch.delenv("ELSPETH_FINGERPRINT_KEY", raising=False)
        monkeypatch.setenv("ELSPETH_ALLOW_RAW_SECRETS", "true")

        config_dict = {
            "landscape": {"url": "postgresql://user:mysecret@host/db"},
            "datasource": {"plugin": "csv", "options": {}},
            "sinks": {"output": {"plugin": "csv_sink"}},
            "output_sink": "output",
        }

        result = _fingerprint_config_for_audit(config_dict)

        assert "mysecret" not in result["landscape"]["url"]
        assert "url_password_fingerprint" not in result["landscape"]
        assert result["landscape"]["url_password_redacted"] is True


class TestRunModeSettings:
    """Tests for run_mode configuration."""

    def test_run_mode_defaults_to_live(self) -> None:
        """Default run_mode should be 'live'."""
        from elspeth.contracts.enums import RunMode
        from elspeth.core.config import ElspethSettings

        settings = ElspethSettings(
            datasource={"plugin": "csv"},
            sinks={"output": {"plugin": "csv"}},
            output_sink="output",
        )

        assert settings.run_mode == RunMode.LIVE

    def test_run_mode_accepts_all_valid_modes(self) -> None:
        """All valid run_mode values should be accepted."""
        from elspeth.contracts.enums import RunMode
        from elspeth.core.config import ElspethSettings

        # Test live mode
        settings_live = ElspethSettings(
            datasource={"plugin": "csv"},
            sinks={"output": {"plugin": "csv"}},
            output_sink="output",
            run_mode="live",
        )
        assert settings_live.run_mode == RunMode.LIVE

        # Test replay mode (with required source run ID)
        settings_replay = ElspethSettings(
            datasource={"plugin": "csv"},
            sinks={"output": {"plugin": "csv"}},
            output_sink="output",
            run_mode="replay",
            replay_source_run_id="run-abc123",
        )
        assert settings_replay.run_mode == RunMode.REPLAY

        # Test verify mode (with required source run ID)
        settings_verify = ElspethSettings(
            datasource={"plugin": "csv"},
            sinks={"output": {"plugin": "csv"}},
            output_sink="output",
            run_mode="verify",
            replay_source_run_id="run-abc123",
        )
        assert settings_verify.run_mode == RunMode.VERIFY

    def test_replay_mode_requires_source_run_id(self) -> None:
        """Replay mode requires replay_source_run_id."""
        from elspeth.core.config import ElspethSettings

        with pytest.raises(ValidationError) as exc_info:
            ElspethSettings(
                datasource={"plugin": "csv"},
                sinks={"output": {"plugin": "csv"}},
                output_sink="output",
                run_mode="replay",
                # Missing replay_source_run_id
            )

        assert "replay_source_run_id is required" in str(exc_info.value)
        assert "replay" in str(exc_info.value)

    def test_verify_mode_requires_source_run_id(self) -> None:
        """Verify mode requires replay_source_run_id."""
        from elspeth.core.config import ElspethSettings

        with pytest.raises(ValidationError) as exc_info:
            ElspethSettings(
                datasource={"plugin": "csv"},
                sinks={"output": {"plugin": "csv"}},
                output_sink="output",
                run_mode="verify",
                # Missing replay_source_run_id
            )

        assert "replay_source_run_id is required" in str(exc_info.value)
        assert "verify" in str(exc_info.value)

    def test_live_mode_ignores_source_run_id(self) -> None:
        """Live mode doesn't require replay_source_run_id."""
        from elspeth.contracts.enums import RunMode
        from elspeth.core.config import ElspethSettings

        # Should not raise - live mode doesn't require source run ID
        settings = ElspethSettings(
            datasource={"plugin": "csv"},
            sinks={"output": {"plugin": "csv"}},
            output_sink="output",
            run_mode="live",
            # No replay_source_run_id
        )

        assert settings.run_mode == RunMode.LIVE
        assert settings.replay_source_run_id is None

    def test_live_mode_accepts_source_run_id(self) -> None:
        """Live mode accepts but ignores replay_source_run_id if provided."""
        from elspeth.contracts.enums import RunMode
        from elspeth.core.config import ElspethSettings

        # Should not raise - live mode ignores source run ID
        settings = ElspethSettings(
            datasource={"plugin": "csv"},
            sinks={"output": {"plugin": "csv"}},
            output_sink="output",
            run_mode="live",
            replay_source_run_id="run-abc123",  # Provided but not required
        )

        assert settings.run_mode == RunMode.LIVE
        assert settings.replay_source_run_id == "run-abc123"

    def test_run_mode_invalid_value_rejected(self) -> None:
        """Invalid run_mode values should be rejected."""
        from elspeth.core.config import ElspethSettings

        with pytest.raises(ValidationError):
            ElspethSettings(
                datasource={"plugin": "csv"},
                sinks={"output": {"plugin": "csv"}},
                output_sink="output",
                run_mode="invalid_mode",
            )

    def test_run_mode_settings_frozen(self) -> None:
        """RunMode settings are immutable."""
        from elspeth.core.config import ElspethSettings

        settings = ElspethSettings(
            datasource={"plugin": "csv"},
            sinks={"output": {"plugin": "csv"}},
            output_sink="output",
        )

        with pytest.raises(ValidationError):
            settings.run_mode = "replay"  # type: ignore[misc,assignment]

    def test_resolve_config_includes_run_mode(self) -> None:
        """resolve_config includes run_mode settings."""
        from elspeth.core.config import ElspethSettings, resolve_config

        settings = ElspethSettings(
            datasource={"plugin": "csv"},
            sinks={"output": {"plugin": "csv"}},
            output_sink="output",
            run_mode="replay",
            replay_source_run_id="run-abc123",
        )

        resolved = resolve_config(settings)

        assert "run_mode" in resolved
        assert resolved["run_mode"] == "replay"
        assert "replay_source_run_id" in resolved
        assert resolved["replay_source_run_id"] == "run-abc123"


class TestExpandTemplateFiles:
    """Tests for _expand_template_files function."""

    def test_expand_template_file_not_found(self, tmp_path: Path) -> None:
        """Missing template file raises TemplateFileError."""
        from elspeth.core.config import TemplateFileError, _expand_template_files

        settings_path = tmp_path / "settings.yaml"
        config = {"template_file": "prompts/missing.j2"}

        with pytest.raises(TemplateFileError, match="not found"):
            _expand_template_files(config, settings_path)

    def test_expand_template_file_with_inline_raises(self, tmp_path: Path) -> None:
        """Cannot specify both template and template_file."""
        from elspeth.core.config import TemplateFileError, _expand_template_files

        settings_path = tmp_path / "settings.yaml"
        config = {
            "template": "inline template",
            "template_file": "prompts/test.j2",
        }

        with pytest.raises(TemplateFileError, match="Cannot specify both"):
            _expand_template_files(config, settings_path)

    def test_expand_lookup_file_invalid_yaml(self, tmp_path: Path) -> None:
        """Invalid YAML in lookup file raises TemplateFileError."""
        from elspeth.core.config import TemplateFileError, _expand_template_files

        lookup_file = tmp_path / "bad.yaml"
        lookup_file.write_text("invalid: yaml: content: [")

        settings_path = tmp_path / "settings.yaml"
        config = {
            "template": "test",
            "lookup_file": "bad.yaml",
        }

        with pytest.raises(TemplateFileError, match="Invalid YAML"):
            _expand_template_files(config, settings_path)

    def test_expand_template_file(self, tmp_path: Path) -> None:
        """template_file is expanded to template content at config time."""
        from elspeth.core.config import _expand_template_files

        # Create template file
        template_file = tmp_path / "prompts" / "test.j2"
        template_file.parent.mkdir(parents=True)
        template_file.write_text("Hello, {{ row.name }}!")

        # Create settings file path (for relative resolution)
        settings_path = tmp_path / "settings.yaml"

        config = {
            "template_file": "prompts/test.j2",
        }

        expanded = _expand_template_files(config, settings_path)

        assert "template" in expanded
        assert expanded["template"] == "Hello, {{ row.name }}!"
        assert expanded["template_source"] == "prompts/test.j2"
        assert "template_file" not in expanded  # Original key removed

    def test_expand_lookup_file_with_inline_raises(self, tmp_path: Path) -> None:
        """Cannot specify both lookup and lookup_file."""
        from elspeth.core.config import TemplateFileError, _expand_template_files

        settings_path = tmp_path / "settings.yaml"
        config = {
            "template": "test",
            "lookup": {"existing": "data"},
            "lookup_file": "prompts/lookups.yaml",
        }

        with pytest.raises(TemplateFileError, match="Cannot specify both"):
            _expand_template_files(config, settings_path)

    def test_expand_lookup_file(self, tmp_path: Path) -> None:
        """lookup_file is expanded to parsed YAML at config time."""
        from elspeth.core.config import _expand_template_files

        # Create lookup file
        lookup_file = tmp_path / "prompts" / "lookups.yaml"
        lookup_file.parent.mkdir(parents=True, exist_ok=True)
        lookup_file.write_text("categories:\n  - Electronics\n  - Clothing\n")

        settings_path = tmp_path / "settings.yaml"

        config = {
            "template": "{{ lookup.categories }}",
            "lookup_file": "prompts/lookups.yaml",
        }

        expanded = _expand_template_files(config, settings_path)

        assert expanded["lookup"] == {"categories": ["Electronics", "Clothing"]}
        assert expanded["lookup_source"] == "prompts/lookups.yaml"
        assert "lookup_file" not in expanded

    def test_expand_template_and_lookup_files(self, tmp_path: Path) -> None:
        """Both template_file and lookup_file expand together."""
        from elspeth.core.config import _expand_template_files

        prompts_dir = tmp_path / "prompts"
        prompts_dir.mkdir()

        (prompts_dir / "classify.j2").write_text("Category: {{ lookup.cats[row.id] }}")
        (prompts_dir / "lookups.yaml").write_text("cats:\n  0: A\n  1: B\n")

        settings_path = tmp_path / "settings.yaml"

        config = {
            "template_file": "prompts/classify.j2",
            "lookup_file": "prompts/lookups.yaml",
        }

        expanded = _expand_template_files(config, settings_path)

        assert expanded["template"] == "Category: {{ lookup.cats[row.id] }}"
        assert expanded["template_source"] == "prompts/classify.j2"
        assert expanded["lookup"] == {"cats": {0: "A", 1: "B"}}
        assert expanded["lookup_source"] == "prompts/lookups.yaml"

    def test_expand_system_prompt_file_not_found(self, tmp_path: Path) -> None:
        """Missing system_prompt file raises TemplateFileError."""
        from elspeth.core.config import TemplateFileError, _expand_template_files

        settings_path = tmp_path / "settings.yaml"
        config = {
            "template": "test",
            "system_prompt_file": "prompts/missing.txt",
        }

        with pytest.raises(TemplateFileError, match="not found"):
            _expand_template_files(config, settings_path)

    def test_expand_system_prompt_file_with_inline_raises(self, tmp_path: Path) -> None:
        """Cannot specify both system_prompt and system_prompt_file."""
        from elspeth.core.config import TemplateFileError, _expand_template_files

        settings_path = tmp_path / "settings.yaml"
        config = {
            "template": "test",
            "system_prompt": "You are a helpful assistant.",
            "system_prompt_file": "prompts/system.txt",
        }

        with pytest.raises(TemplateFileError, match="Cannot specify both"):
            _expand_template_files(config, settings_path)

    def test_expand_system_prompt_file(self, tmp_path: Path) -> None:
        """system_prompt_file is expanded to system_prompt content at config time."""
        from elspeth.core.config import _expand_template_files

        # Create system prompt file
        system_file = tmp_path / "prompts" / "system.txt"
        system_file.parent.mkdir(parents=True, exist_ok=True)
        system_file.write_text("You are an expert classifier. Be precise and consistent.")

        settings_path = tmp_path / "settings.yaml"

        config = {
            "template": "Classify: {{ row.text }}",
            "system_prompt_file": "prompts/system.txt",
        }

        expanded = _expand_template_files(config, settings_path)

        assert "system_prompt" in expanded
        assert expanded["system_prompt"] == "You are an expert classifier. Be precise and consistent."
        assert expanded["system_prompt_source"] == "prompts/system.txt"
        assert "system_prompt_file" not in expanded  # Original key removed

    def test_expand_all_file_types(self, tmp_path: Path) -> None:
        """All file types (template, lookup, system_prompt) expand together."""
        from elspeth.core.config import _expand_template_files

        prompts_dir = tmp_path / "prompts"
        prompts_dir.mkdir()

        (prompts_dir / "classify.j2").write_text("Classify: {{ row.text }}")
        (prompts_dir / "lookups.yaml").write_text("categories:\n  - A\n  - B\n")
        (prompts_dir / "system.txt").write_text("You are an expert.")

        settings_path = tmp_path / "settings.yaml"

        config = {
            "template_file": "prompts/classify.j2",
            "lookup_file": "prompts/lookups.yaml",
            "system_prompt_file": "prompts/system.txt",
        }

        expanded = _expand_template_files(config, settings_path)

        # Template expanded
        assert expanded["template"] == "Classify: {{ row.text }}"
        assert expanded["template_source"] == "prompts/classify.j2"
        assert "template_file" not in expanded

        # Lookup expanded
        assert expanded["lookup"] == {"categories": ["A", "B"]}
        assert expanded["lookup_source"] == "prompts/lookups.yaml"
        assert "lookup_file" not in expanded

        # System prompt expanded
        assert expanded["system_prompt"] == "You are an expert."
        assert expanded["system_prompt_source"] == "prompts/system.txt"
        assert "system_prompt_file" not in expanded


class TestLoadSettingsWithRunMode:
    """Tests for loading YAML with run_mode configuration."""

    def test_load_settings_with_live_mode(self, tmp_path: Path) -> None:
        """Load YAML with live mode (default)."""
        from elspeth.contracts.enums import RunMode
        from elspeth.core.config import load_settings

        config_file = tmp_path / "settings.yaml"
        config_file.write_text("""
datasource:
  plugin: csv

sinks:
  output:
    plugin: csv

output_sink: output
run_mode: live
""")
        settings = load_settings(config_file)

        assert settings.run_mode == RunMode.LIVE
        assert settings.replay_source_run_id is None

    def test_load_settings_with_replay_mode(self, tmp_path: Path) -> None:
        """Load YAML with replay mode."""
        from elspeth.contracts.enums import RunMode
        from elspeth.core.config import load_settings

        config_file = tmp_path / "settings.yaml"
        config_file.write_text("""
datasource:
  plugin: csv

sinks:
  output:
    plugin: csv

output_sink: output
run_mode: replay
replay_source_run_id: run-abc123
""")
        settings = load_settings(config_file)

        assert settings.run_mode == RunMode.REPLAY
        assert settings.replay_source_run_id == "run-abc123"

    def test_load_settings_with_verify_mode(self, tmp_path: Path) -> None:
        """Load YAML with verify mode."""
        from elspeth.contracts.enums import RunMode
        from elspeth.core.config import load_settings

        config_file = tmp_path / "settings.yaml"
        config_file.write_text("""
datasource:
  plugin: csv

sinks:
  output:
    plugin: csv

output_sink: output
run_mode: verify
replay_source_run_id: run-xyz789
""")
        settings = load_settings(config_file)

        assert settings.run_mode == RunMode.VERIFY
        assert settings.replay_source_run_id == "run-xyz789"

    def test_load_settings_replay_without_source_run_id_fails(self, tmp_path: Path) -> None:
        """Loading replay mode without source_run_id should fail."""
        from elspeth.core.config import load_settings

        config_file = tmp_path / "settings.yaml"
        config_file.write_text("""
datasource:
  plugin: csv

sinks:
  output:
    plugin: csv

output_sink: output
run_mode: replay
""")
        with pytest.raises(ValidationError) as exc_info:
            load_settings(config_file)

        assert "replay_source_run_id is required" in str(exc_info.value)


class TestLoadSettingsTemplateFileExpansion:
    """Tests for template file expansion during load_settings."""

    def test_load_settings_expands_template_files(self, tmp_path: Path) -> None:
        """load_settings expands template_file in row_plugins."""
        from elspeth.core.config import load_settings

        # Create directory structure
        prompts_dir = tmp_path / "prompts"
        prompts_dir.mkdir()
        (prompts_dir / "test.j2").write_text("Hello {{ row.name }}")
        (prompts_dir / "lookups.yaml").write_text("greetings:\n  - Hello\n")

        # Create settings file
        settings_file = tmp_path / "settings.yaml"
        settings_file.write_text("""
datasource:
  plugin: csv_local
  options:
    path: test.csv

sinks:
  output:
    plugin: csv_local
    options:
      path: out.csv

output_sink: output

row_plugins:
  - plugin: openrouter_llm
    options:
      model: test
      template_file: prompts/test.j2
      lookup_file: prompts/lookups.yaml
      schema:
        fields: dynamic
""")

        settings = load_settings(settings_file)

        # Check that files were expanded
        plugin_opts = settings.row_plugins[0].options
        assert plugin_opts["template"] == "Hello {{ row.name }}"
        assert plugin_opts["template_source"] == "prompts/test.j2"
        assert plugin_opts["lookup"] == {"greetings": ["Hello"]}
        assert plugin_opts["lookup_source"] == "prompts/lookups.yaml"
