"""Tests for elspeth validate command."""

from pathlib import Path

import pytest
import yaml
from typer.testing import CliRunner

from elspeth.cli import app

runner = CliRunner()


class TestValidateCommand:
    """Tests for validate command with new config format."""

    @pytest.fixture
    def valid_config(self, tmp_path: Path) -> Path:
        """Create a valid pipeline config using new schema."""
        config = {
            "datasource": {
                "plugin": "csv",
                "options": {"path": "/data/input.csv"},
            },
            "sinks": {
                "output": {
                    "plugin": "json",
                    "options": {"path": "/data/output.json"},
                },
            },
            "output_sink": "output",
        }
        config_file = tmp_path / "valid.yaml"
        config_file.write_text(yaml.dump(config))
        return config_file

    @pytest.fixture
    def invalid_yaml(self, tmp_path: Path) -> Path:
        """Create invalid YAML file."""
        config_file = tmp_path / "invalid.yaml"
        config_file.write_text("datasource:\n  plugin: csv\n  options: [invalid")
        return config_file

    @pytest.fixture
    def missing_datasource_config(self, tmp_path: Path) -> Path:
        """Create config missing required datasource."""
        config = {
            "sinks": {
                "output": {
                    "plugin": "json",
                    "options": {"path": "/data/output.json"},
                },
            },
            "output_sink": "output",
        }
        config_file = tmp_path / "missing_datasource.yaml"
        config_file.write_text(yaml.dump(config))
        return config_file

    @pytest.fixture
    def invalid_output_sink_config(self, tmp_path: Path) -> Path:
        """Create config with invalid output_sink reference."""
        config = {
            "datasource": {
                "plugin": "csv",
                "options": {"path": "/data/input.csv"},
            },
            "sinks": {
                "output": {
                    "plugin": "json",
                    "options": {"path": "/data/output.json"},
                },
            },
            "output_sink": "nonexistent",  # References non-existent sink
        }
        config_file = tmp_path / "invalid_output_sink.yaml"
        config_file.write_text(yaml.dump(config))
        return config_file

    def test_validate_valid_config(self, valid_config: Path) -> None:
        """Valid config passes validation."""
        result = runner.invoke(app, ["validate", "-s", str(valid_config)])
        assert result.exit_code == 0
        assert "valid" in result.stdout.lower()

    def test_validate_file_not_found(self) -> None:
        """Nonexistent file shows error."""
        result = runner.invoke(app, ["validate", "-s", "/nonexistent/file.yaml"])
        assert result.exit_code != 0
        # Use result.output to include stderr
        assert "not found" in result.output.lower()

    def test_validate_invalid_yaml(self, invalid_yaml: Path) -> None:
        """Invalid YAML raises exception (Dynaconf handles parsing)."""
        # Dynaconf raises its own YAML parser error which isn't caught
        # by our ValidationError handler, so this test expects an exception
        result = runner.invoke(app, ["validate", "-s", str(invalid_yaml)])
        # The exception may not produce output - just verify non-zero exit
        assert result.exit_code != 0 or result.exception is not None

    def test_validate_missing_datasource(self, missing_datasource_config: Path) -> None:
        """Missing datasource shows error."""
        result = runner.invoke(app, ["validate", "-s", str(missing_datasource_config)])
        assert result.exit_code != 0
        assert "datasource" in result.output.lower()

    def test_validate_invalid_output_sink(self, invalid_output_sink_config: Path) -> None:
        """Invalid output_sink reference shows error."""
        result = runner.invoke(app, ["validate", "-s", str(invalid_output_sink_config)])
        assert result.exit_code != 0
        assert "nonexistent" in result.output.lower() or "output_sink" in result.output.lower()


class TestValidateCommandGraphValidation:
    """Validate command validates graph structure."""

    def test_validate_detects_invalid_route(self, tmp_path: Path) -> None:
        """Validate command catches gate routing to nonexistent sink."""
        config_file = tmp_path / "settings.yaml"
        config_file.write_text("""
datasource:
  plugin: csv

sinks:
  output:
    plugin: csv

row_plugins:
  - plugin: my_gate
    type: gate
    routes:
      bad_route: nonexistent_sink

output_sink: output
""")

        result = runner.invoke(app, ["validate", "-s", str(config_file)])

        assert result.exit_code != 0
        output = result.stdout + (result.stderr or "")
        assert "nonexistent_sink" in output.lower()

    def test_validate_shows_graph_info(self, tmp_path: Path) -> None:
        """Validate command shows graph structure on success."""
        config_file = tmp_path / "settings.yaml"
        config_file.write_text("""
datasource:
  plugin: csv

sinks:
  results:
    plugin: csv
  flagged:
    plugin: csv

row_plugins:
  - plugin: classifier
    type: gate
    routes:
      suspicious: flagged
      clean: continue

output_sink: results
""")

        result = runner.invoke(app, ["validate", "-s", str(config_file)])

        assert result.exit_code == 0
        # Should show graph info with node and edge counts
        assert "graph" in result.stdout.lower()
        assert "node" in result.stdout.lower()
        assert "edge" in result.stdout.lower()
