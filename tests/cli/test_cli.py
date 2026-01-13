"""Tests for ELSPETH CLI."""

from pathlib import Path

import pytest
from typer.testing import CliRunner

# Note: In Click 8.0+, mix_stderr is no longer a CliRunner parameter.
# Stderr output is combined with stdout by default when using CliRunner.invoke()
runner = CliRunner()


class TestCLIBasics:
    """Test basic CLI functionality."""

    def test_cli_exists(self) -> None:
        """CLI app can be imported."""
        from elspeth.cli import app

        assert app is not None

    def test_version_flag(self) -> None:
        """--version shows version info."""
        from elspeth.cli import app

        result = runner.invoke(app, ["--version"])
        assert result.exit_code == 0
        assert "elspeth" in result.stdout.lower()

    def test_help_flag(self) -> None:
        """--help shows available commands."""
        from elspeth.cli import app

        result = runner.invoke(app, ["--help"])
        assert result.exit_code == 0
        assert "run" in result.stdout
        assert "explain" in result.stdout
        assert "validate" in result.stdout
        assert "plugins" in result.stdout


class TestRunCommandExecutesTransforms:
    """Verify row_plugins are actually executed."""

    def test_transforms_from_config_are_instantiated(self, tmp_path: Path) -> None:
        """Transforms in row_plugins are instantiated and passed to orchestrator."""
        from typer.testing import CliRunner

        from elspeth.cli import app

        runner = CliRunner()

        # Create input CSV
        input_file = tmp_path / "input.csv"
        input_file.write_text("id,value\n1,hello\n2,world\n")

        output_file = tmp_path / "output.csv"
        audit_db = tmp_path / "audit.db"

        # Config with a passthrough transform
        config_file = tmp_path / "settings.yaml"
        config_file.write_text(f"""
datasource:
  plugin: csv
  options:
    path: "{input_file}"

sinks:
  results:
    plugin: csv
    options:
      path: "{output_file}"

row_plugins:
  - plugin: passthrough

output_sink: results

landscape:
  enabled: true
  backend: sqlite
  url: "sqlite:///{audit_db}"
"""
        )

        result = runner.invoke(app, ["run", "-s", str(config_file), "--execute", "-v"])

        assert result.exit_code == 0, f"CLI failed: {result.stdout}"
        # Output should exist with data processed
        assert output_file.exists()

    def test_field_mapper_transform_modifies_output(self, tmp_path: Path) -> None:
        """Field mapper should rename columns - proves transform actually runs."""
        from typer.testing import CliRunner

        from elspeth.cli import app

        runner = CliRunner()

        # Create input CSV
        input_file = tmp_path / "input.csv"
        input_file.write_text("id,old_name\n1,hello\n2,world\n")

        output_file = tmp_path / "output.csv"
        audit_db = tmp_path / "audit.db"

        # Config with field_mapper that renames 'old_name' to 'new_name'
        config_file = tmp_path / "settings.yaml"
        config_file.write_text(f"""
datasource:
  plugin: csv
  options:
    path: "{input_file}"

sinks:
  results:
    plugin: csv
    options:
      path: "{output_file}"

row_plugins:
  - plugin: field_mapper
    options:
      mapping:
        old_name: new_name

output_sink: results

landscape:
  enabled: true
  backend: sqlite
  url: "sqlite:///{audit_db}"
"""
        )

        result = runner.invoke(app, ["run", "-s", str(config_file), "--execute", "-v"])

        assert result.exit_code == 0, f"CLI failed: {result.stdout}"
        assert output_file.exists()

        # Read output and verify the transform was applied
        output_content = output_file.read_text()
        # If transform ran, we should have 'new_name' column, not 'old_name'
        assert "new_name" in output_content, (
            f"Field mapper should have renamed 'old_name' to 'new_name'. "
            f"Output was: {output_content}"
        )
