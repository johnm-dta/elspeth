"""Tests for elspeth validate command."""

from pathlib import Path

import pytest
import yaml
from typer.testing import CliRunner

runner = CliRunner()


class TestValidateCommand:
    """Tests for validate command."""

    @pytest.fixture
    def valid_config(self, tmp_path: Path) -> Path:
        """Create a valid pipeline config."""
        config = {
            "source": {"plugin": "csv", "path": "/data/input.csv"},
            "sinks": {
                "output": {"plugin": "json", "path": "/data/output.json"},
            },
        }
        config_file = tmp_path / "valid.yaml"
        config_file.write_text(yaml.dump(config))
        return config_file

    @pytest.fixture
    def invalid_yaml(self, tmp_path: Path) -> Path:
        """Create invalid YAML file."""
        config_file = tmp_path / "invalid.yaml"
        config_file.write_text("source:\n  plugin: csv\n  path: [invalid")
        return config_file

    @pytest.fixture
    def missing_source_config(self, tmp_path: Path) -> Path:
        """Create config missing required source."""
        config = {
            "sinks": {
                "output": {"plugin": "json", "path": "/data/output.json"},
            },
        }
        config_file = tmp_path / "missing_source.yaml"
        config_file.write_text(yaml.dump(config))
        return config_file

    @pytest.fixture
    def unknown_plugin_config(self, tmp_path: Path) -> Path:
        """Create config with unknown plugin."""
        config = {
            "source": {"plugin": "unknown_source", "path": "/data/input.csv"},
            "sinks": {
                "output": {"plugin": "json", "path": "/data/output.json"},
            },
        }
        config_file = tmp_path / "unknown_plugin.yaml"
        config_file.write_text(yaml.dump(config))
        return config_file

    def test_validate_valid_config(self, valid_config: Path) -> None:
        """Valid config passes validation."""
        from elspeth.cli import app

        result = runner.invoke(app, ["validate", "-s", str(valid_config)])
        assert result.exit_code == 0
        assert "valid" in result.stdout.lower()

    def test_validate_file_not_found(self) -> None:
        """Nonexistent file shows error."""
        from elspeth.cli import app

        result = runner.invoke(app, ["validate", "-s", "/nonexistent/file.yaml"])
        assert result.exit_code != 0
        # Use result.output to include stderr
        assert "not found" in result.output.lower()

    def test_validate_invalid_yaml(self, invalid_yaml: Path) -> None:
        """Invalid YAML shows parse error."""
        from elspeth.cli import app

        result = runner.invoke(app, ["validate", "-s", str(invalid_yaml)])
        assert result.exit_code != 0
        assert "yaml" in result.output.lower() or "error" in result.output.lower()

    def test_validate_missing_source(self, missing_source_config: Path) -> None:
        """Missing source shows error."""
        from elspeth.cli import app

        result = runner.invoke(app, ["validate", "-s", str(missing_source_config)])
        assert result.exit_code != 0
        assert "source" in result.output.lower()

    def test_validate_unknown_plugin(self, unknown_plugin_config: Path) -> None:
        """Unknown plugin shows error."""
        from elspeth.cli import app

        result = runner.invoke(app, ["validate", "-s", str(unknown_plugin_config)])
        assert result.exit_code != 0
        assert "unknown" in result.output.lower()
