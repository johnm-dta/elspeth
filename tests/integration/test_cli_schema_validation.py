"""Integration tests for CLI schema validation."""

import tempfile
from pathlib import Path

from typer.testing import CliRunner

from elspeth.cli import app


def test_cli_run_detects_schema_incompatibility():
    """Verify CLI run command detects schema incompatibility."""
    runner = CliRunner()

    config_yaml = """
source:
  plugin: csv
  options:
    path: test_input.csv
    schema:
      mode: strict
      fields:
        field_a: {type: str}

transforms:
  - plugin: passthrough
    options:
      schema:
        fields:
          field_a: {type: str}

sinks:
  output:
    plugin: csv
    options:
      path: test_output.csv
      schema:
        mode: strict
        fields:
          field_b: {type: int}  # INCOMPATIBLE

default_sink: output
"""

    with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
        f.write(config_yaml)
        config_file = Path(f.name)

    try:
        result = runner.invoke(app, ["run", "--settings", str(config_file)])

        # Should fail with schema error
        assert result.exit_code != 0
        assert "schema" in result.output.lower() or "field_b" in result.output.lower()

    finally:
        config_file.unlink()


def test_cli_validate_detects_schema_incompatibility():
    """Verify CLI validate command detects schema incompatibility."""
    runner = CliRunner()

    config_yaml = """
source:
  plugin: csv
  options:
    path: test_input.csv
    schema:
      mode: strict
      fields:
        field_a: {type: str}

transforms:
  - plugin: passthrough
    options:
      schema:
        fields:
          field_a: {type: str}

sinks:
  output:
    plugin: csv
    options:
      path: test_output.csv
      schema:
        mode: strict
        fields:
          field_b: {type: int}  # INCOMPATIBLE

default_sink: output
"""

    with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
        f.write(config_yaml)
        config_file = Path(f.name)

    try:
        result = runner.invoke(app, ["validate", "--settings", str(config_file)])

        # Should fail with schema error
        assert result.exit_code != 0
        assert "schema" in result.output.lower() or "field_b" in result.output.lower()

    finally:
        config_file.unlink()
