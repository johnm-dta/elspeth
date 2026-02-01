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
            "source": {
                "plugin": "csv",
                "options": {
                    "path": "/data/input.csv",
                    "on_validation_failure": "quarantine",
                    "schema": {"fields": "dynamic"},
                },
            },
            "sinks": {
                "output": {
                    "plugin": "json",
                    "options": {
                        "path": "/data/output.json",
                        "schema": {"fields": "dynamic"},
                    },
                },
            },
            "default_sink": "output",
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
                    "options": {
                        "path": "/data/output.json",
                        "schema": {"fields": "dynamic"},
                    },
                },
            },
            "default_sink": "output",
        }
        config_file = tmp_path / "missing_datasource.yaml"
        config_file.write_text(yaml.dump(config))
        return config_file

    @pytest.fixture
    def invalid_output_sink_config(self, tmp_path: Path) -> Path:
        """Create config with invalid default_sink reference."""
        config = {
            "source": {
                "plugin": "csv",
                "options": {
                    "path": "/data/input.csv",
                    "on_validation_failure": "quarantine",
                    "schema": {"fields": "dynamic"},
                },
            },
            "sinks": {
                "output": {
                    "plugin": "json",
                    "options": {
                        "path": "/data/output.json",
                        "schema": {"fields": "dynamic"},
                    },
                },
            },
            "default_sink": "nonexistent",  # References non-existent sink
        }
        config_file = tmp_path / "invalid_output_sink.yaml"
        config_file.write_text(yaml.dump(config))
        return config_file

    def test_validate_valid_config(self, valid_config: Path) -> None:
        """Valid config passes validation."""
        result = runner.invoke(app, ["validate", "-s", str(valid_config)])
        assert result.exit_code == 0
        # Use exact phrase to avoid matching "invalid"
        assert "pipeline configuration valid" in result.stdout.lower(), (
            f"Expected 'Pipeline configuration valid' in output, got: {result.stdout}"
        )

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
        """Missing source shows error."""
        result = runner.invoke(app, ["validate", "-s", str(missing_datasource_config)])
        assert result.exit_code != 0
        assert "source" in result.output.lower()

    def test_validate_invalid_output_sink(self, invalid_output_sink_config: Path) -> None:
        """Invalid default_sink reference shows error."""
        result = runner.invoke(app, ["validate", "-s", str(invalid_output_sink_config)])
        assert result.exit_code != 0
        assert "nonexistent" in result.output.lower() or "default_sink" in result.output.lower()


class TestValidateCommandGraphValidation:
    """Validate command validates graph structure."""

    def test_validate_detects_invalid_route(self, tmp_path: Path) -> None:
        """Validate command catches gate routing to nonexistent sink."""
        config_file = tmp_path / "settings.yaml"
        config_file.write_text("""
source:
  plugin: csv
  options:
    path: /data/input.csv
    on_validation_failure: quarantine
    schema:
      fields: dynamic

sinks:
  output:
    plugin: csv
    options:
      path: /data/output.csv
      schema:
        mode: strict
        fields:
          - "data: str"

gates:
  - name: my_gate
    condition: "True"
    routes:
      "true": nonexistent_sink
      "false": continue

default_sink: output
""")

        result = runner.invoke(app, ["validate", "-s", str(config_file)])

        assert result.exit_code != 0
        output = result.stdout + (result.stderr or "")
        assert "nonexistent_sink" in output.lower()

    def test_validate_shows_graph_info(self, tmp_path: Path) -> None:
        """Validate command shows graph structure on success."""
        config_file = tmp_path / "settings.yaml"
        config_file.write_text("""
source:
  plugin: csv
  options:
    path: /data/input.csv
    on_validation_failure: quarantine
    schema:
      fields: dynamic

sinks:
  results:
    plugin: csv
    options:
      path: /data/results.csv
      schema:
        mode: strict
        fields:
          - "data: str"
  flagged:
    plugin: csv
    options:
      path: /data/flagged.csv
      schema:
        mode: strict
        fields:
          - "data: str"

gates:
  - name: classifier
    condition: "row['suspicious'] == True"
    routes:
      "true": flagged
      "false": continue

default_sink: results
""")

        result = runner.invoke(app, ["validate", "-s", str(config_file)])

        assert result.exit_code == 0

        # Should show graph info with actual node and edge counts
        import re

        output_lower = result.stdout.lower()
        assert "graph:" in output_lower, f"Expected 'Graph:' in output, got: {result.stdout}"

        # Parse node and edge counts from output
        # Format: "Graph: N nodes, M edges"
        match = re.search(r"graph:\s*(\d+)\s+nodes?,\s*(\d+)\s+edges?", output_lower)
        assert match is not None, f"Expected 'Graph: N nodes, M edges' format, got: {result.stdout}"

        node_count = int(match.group(1))
        edge_count = int(match.group(2))

        # For this config: 1 source + 1 gate + 2 sinks = 4 nodes
        # Edges: source→gate, gate→results, gate→flagged = 3 edges
        assert node_count == 4, f"Expected 4 nodes (source+gate+2sinks), got {node_count}"
        assert edge_count == 3, f"Expected 3 edges (source→gate→2sinks), got {edge_count}"
