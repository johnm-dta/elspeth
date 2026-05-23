# tests/integration/config/test_cli_schema_validation.py
"""Integration tests for CLI schema validation."""

import tempfile
from pathlib import Path

from typer.testing import CliRunner

from elspeth.cli import app


def test_cli_run_detects_schema_incompatibility():
    """Verify CLI run command detects schema incompatibility between source and sink."""
    runner = CliRunner()

    config_yaml = """
sources:
  primary:
    plugin: csv
    options:
      path: test_input.csv
      schema:
        mode: fixed
        fields:
          - "field_a: str"
      on_success: output

transforms:
  - plugin: passthrough
    options:
      on_success: output

sinks:
  output:
    plugin: csv
    options:
      path: test_output.csv
      schema:
        mode: fixed
        fields:
          - "field_b: int"
"""

    with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
        f.write(config_yaml)
        config_file = Path(f.name)

    try:
        result = runner.invoke(app, ["run", "--settings", str(config_file)])

        assert result.exit_code != 0
        # Must fail specifically due to schema or field issue, not generic crash
        output_lower = result.output.lower()
        assert "schema" in output_lower or "field" in output_lower, f"Expected schema-related error, got: {result.output!r}"

    finally:
        config_file.unlink()


def test_cli_validate_detects_schema_incompatibility():
    """Verify CLI validate command detects schema incompatibility between source and sink."""
    runner = CliRunner()

    config_yaml = """
sources:
  primary:
    plugin: csv
    options:
      path: test_input.csv
      schema:
        mode: fixed
        fields:
          - "field_a: str"
      on_success: output

transforms:
  - plugin: passthrough
    options:
      on_success: output

sinks:
  output:
    plugin: csv
    options:
      path: test_output.csv
      schema:
        mode: fixed
        fields:
          - "field_b: int"
"""

    with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
        f.write(config_yaml)
        config_file = Path(f.name)

    try:
        result = runner.invoke(app, ["validate", "--settings", str(config_file)])

        assert result.exit_code != 0
        # Must fail specifically due to schema or field issue, not generic crash
        output_lower = result.output.lower()
        assert "schema" in output_lower or "field" in output_lower, f"Expected schema-related error, got: {result.output!r}"

    finally:
        config_file.unlink()
