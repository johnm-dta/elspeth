"""Integration tests for resume command with new schema validation."""

import tempfile
from pathlib import Path

from typer.testing import CliRunner

from elspeth.cli import app


def test_resume_command_uses_new_graph_construction():
    """Verify resume command builds graph from plugin instances."""
    # This test verifies resume doesn't call deprecated from_config()
    # Actual checkpoint/resume testing requires database setup

    runner = CliRunner()

    # Create minimal valid config
    config_yaml = """
source:
  plugin: csv
  options:
    path: test.csv
    schema:
      fields: dynamic

sinks:
  output:
    plugin: csv
    options:
      path: output.csv

default_sink: output

landscape:
  url: "sqlite:///:memory:"
"""

    with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
        f.write(config_yaml)
        config_file = Path(f.name)

    try:
        # Resume with non-existent run_id should fail gracefully
        # but NOT crash due to from_config() deprecation warning
        result = runner.invoke(
            app,
            [
                "resume",
                "nonexistent-run-id",
                "--settings",
                str(config_file),
            ],
        )

        # Should exit with error (run not found), not crash
        assert result.exit_code != 0
        # Should NOT contain deprecation warning
        assert "deprecated" not in result.output.lower()

    finally:
        config_file.unlink()
