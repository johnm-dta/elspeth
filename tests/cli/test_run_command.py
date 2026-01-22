"""Tests for elspeth run command."""

from pathlib import Path

import pytest
import yaml
from typer.testing import CliRunner

from elspeth.cli import app

runner = CliRunner()


class TestRunCommand:
    """Tests for run command with new config format."""

    @pytest.fixture
    def sample_data(self, tmp_path: Path) -> Path:
        """Create sample input data."""
        csv_file = tmp_path / "input.csv"
        csv_file.write_text("id,name,value\n1,alice,100\n2,bob,200\n")
        return csv_file

    @pytest.fixture
    def pipeline_settings(self, tmp_path: Path, sample_data: Path) -> Path:
        """Create a complete pipeline configuration using new schema."""
        output_file = tmp_path / "output.json"
        landscape_db = tmp_path / "landscape.db"
        settings = {
            "datasource": {
                "plugin": "csv",
                "options": {
                    "path": str(sample_data),
                    "on_validation_failure": "discard",
                    "schema": {"fields": "dynamic"},
                },
            },
            "sinks": {
                "default": {
                    "plugin": "json",
                    "options": {
                        "path": str(output_file),
                        "schema": {"fields": "dynamic"},
                    },
                },
            },
            "output_sink": "default",
            "landscape": {"url": f"sqlite:///{landscape_db}"},
        }
        settings_file = tmp_path / "settings.yaml"
        settings_file.write_text(yaml.dump(settings))
        return settings_file

    def test_run_executes_pipeline(self, pipeline_settings: Path, tmp_path: Path) -> None:
        """run --execute executes pipeline and creates output."""
        result = runner.invoke(app, ["run", "--settings", str(pipeline_settings), "--execute"])
        assert result.exit_code == 0

        # Check output was created
        output_file = tmp_path / "output.json"
        assert output_file.exists()

    def test_run_shows_summary(self, pipeline_settings: Path) -> None:
        """run --execute shows execution summary."""
        result = runner.invoke(app, ["run", "--settings", str(pipeline_settings), "--execute"])
        assert result.exit_code == 0
        # Use result.output to include both stdout and stderr
        assert "completed" in result.output.lower() or "rows" in result.output.lower()

    def test_run_without_execute_shows_warning(self, pipeline_settings: Path) -> None:
        """run without --execute shows warning and exits non-zero."""
        result = runner.invoke(app, ["run", "--settings", str(pipeline_settings)])
        assert result.exit_code != 0
        assert "--execute" in result.output

    def test_run_missing_settings(self) -> None:
        """run exits non-zero for missing settings file."""
        result = runner.invoke(app, ["run", "--settings", "/nonexistent.yaml"])
        assert result.exit_code != 0

    def test_run_dry_run_mode(self, pipeline_settings: Path) -> None:
        """run --dry-run validates without executing."""
        result = runner.invoke(app, ["run", "--settings", str(pipeline_settings), "--dry-run"])
        assert result.exit_code == 0
        assert "dry" in result.output.lower() or "would" in result.output.lower()


class TestRunCommandWithNewConfig:
    """Run command uses load_settings() for config."""

    def test_run_with_readme_config(self, tmp_path: Path) -> None:
        """Run command accepts README-style config."""
        config_file = tmp_path / "settings.yaml"
        config_file.write_text("""
datasource:
  plugin: csv
  options:
    path: input.csv
    on_validation_failure: discard
    schema:
      fields: dynamic

sinks:
  results:
    plugin: csv
    options:
      path: output.csv
      schema:
        fields: dynamic

output_sink: results
""")

        result = runner.invoke(app, ["run", "-s", str(config_file), "--dry-run"])

        assert result.exit_code == 0
        assert "csv" in result.stdout.lower()

    def test_run_rejects_old_config_format(self, tmp_path: Path) -> None:
        """Run command rejects old 'source' format."""
        config_file = tmp_path / "settings.yaml"
        config_file.write_text("""
source:
  plugin: csv
  path: input.csv
""")

        result = runner.invoke(app, ["run", "-s", str(config_file), "--dry-run"])

        # Should fail - 'source' is not valid, must be 'datasource'
        assert result.exit_code != 0

    def test_run_shows_pydantic_errors(self, tmp_path: Path) -> None:
        """Run shows Pydantic validation errors clearly."""
        config_file = tmp_path / "settings.yaml"
        config_file.write_text("""
datasource:
  plugin: csv

sinks:
  output:
    plugin: csv

output_sink: nonexistent

concurrency:
  max_workers: -5
""")

        result = runner.invoke(app, ["run", "-s", str(config_file), "--dry-run"])

        assert result.exit_code != 0
        # Should show helpful error messages with field path
        output = result.stdout + (result.stderr or "")
        # Pydantic validates fields before model validators, so max_workers error shows first
        assert "configuration errors" in output.lower()
        assert "concurrency.max_workers" in output or "max_workers" in output

    def test_run_shows_output_sink_error(self, tmp_path: Path) -> None:
        """Run shows output_sink validation error when it references nonexistent sink."""
        config_file = tmp_path / "settings.yaml"
        config_file.write_text("""
datasource:
  plugin: csv

sinks:
  output:
    plugin: csv

output_sink: nonexistent
""")

        result = runner.invoke(app, ["run", "-s", str(config_file), "--dry-run"])

        assert result.exit_code != 0
        # Should show helpful error about output_sink
        output = result.stdout + (result.stderr or "")
        assert "nonexistent" in output.lower() or "output_sink" in output.lower()


class TestRunCommandGraphValidation:
    """Run command validates graph before execution."""

    def test_run_validates_graph_before_execution(self, tmp_path: Path) -> None:
        """Run command validates graph before any execution."""
        config_file = tmp_path / "settings.yaml"
        config_file.write_text("""
datasource:
  plugin: csv

sinks:
  output:
    plugin: csv

gates:
  - name: bad_gate
    condition: "True"
    routes:
      "true": missing_sink
      "false": continue

output_sink: output
""")

        result = runner.invoke(app, ["run", "-s", str(config_file), "--execute"])

        # Should fail at validation, not during execution
        assert result.exit_code != 0
        output = result.stdout + (result.stderr or "")
        assert "missing_sink" in output.lower() or "graph" in output.lower()

    def test_dry_run_shows_graph_info(self, tmp_path: Path) -> None:
        """Dry run shows graph structure."""
        config_file = tmp_path / "settings.yaml"
        config_file.write_text("""
datasource:
  plugin: csv

sinks:
  results:
    plugin: csv
  flagged:
    plugin: csv

gates:
  - name: classifier
    condition: "row['suspicious'] == True"
    routes:
      "true": flagged
      "false": continue

output_sink: results
""")

        result = runner.invoke(app, ["run", "-s", str(config_file), "--dry-run", "-v"])

        assert result.exit_code == 0
        # Verbose should show graph info
        assert "node" in result.stdout.lower() or "edge" in result.stdout.lower()


class TestRunCommandResourceCleanup:
    """Tests that run command properly cleans up resources.

    Regression tests for:
    - docs/bugs/closed/P2-2026-01-20-cli-run-does-not-close-landscape-db.md
    """

    def test_run_closes_database_after_success(self, tmp_path: Path) -> None:
        """Database connection is closed after successful pipeline execution.

        Verifies that LandscapeDB.close() is called even on success, preventing
        resource leaks in embedded/test contexts where process doesn't exit.
        """
        from unittest.mock import patch

        # Create minimal valid settings
        csv_file = tmp_path / "input.csv"
        csv_file.write_text("id,name\n1,alice\n")
        output_file = tmp_path / "output.json"
        landscape_db = tmp_path / "landscape.db"

        settings = {
            "datasource": {
                "plugin": "csv",
                "options": {
                    "path": str(csv_file),
                    "on_validation_failure": "discard",
                    "schema": {"fields": "dynamic"},
                },
            },
            "sinks": {
                "default": {
                    "plugin": "json",
                    "options": {
                        "path": str(output_file),
                        "schema": {"fields": "dynamic"},
                    },
                },
            },
            "output_sink": "default",
            "landscape": {"url": f"sqlite:///{landscape_db}"},
        }
        settings_file = tmp_path / "settings.yaml"
        import yaml

        settings_file.write_text(yaml.dump(settings))

        # Track if close was called
        close_called = []
        original_close = None

        def track_close(self: object) -> None:
            close_called.append(True)
            if original_close:
                original_close(self)

        from elspeth.core.landscape import LandscapeDB

        original_close = LandscapeDB.close

        with patch.object(LandscapeDB, "close", track_close):
            result = runner.invoke(app, ["run", "--settings", str(settings_file), "--execute"])

        assert result.exit_code == 0, f"Run failed: {result.output}"
        assert len(close_called) >= 1, (
            "LandscapeDB.close() was not called after successful pipeline execution. "
            "This is a resource leak - see P2-2026-01-20-cli-run-does-not-close-landscape-db.md"
        )

    def test_run_closes_database_after_failure(self, tmp_path: Path) -> None:
        """Database connection is closed even when pipeline fails.

        Verifies that LandscapeDB.close() is called in finally block, ensuring
        cleanup happens regardless of success or failure.
        """
        from unittest.mock import patch

        # Create settings that will fail during execution (non-existent file)
        landscape_db = tmp_path / "landscape.db"

        settings = {
            "datasource": {
                "plugin": "csv",
                "options": {
                    "path": str(tmp_path / "nonexistent.csv"),  # Will fail
                    "on_validation_failure": "discard",
                    "schema": {"fields": "dynamic"},
                },
            },
            "sinks": {
                "default": {
                    "plugin": "json",
                    "options": {
                        "path": str(tmp_path / "output.json"),
                        "schema": {"fields": "dynamic"},
                    },
                },
            },
            "output_sink": "default",
            "landscape": {"url": f"sqlite:///{landscape_db}"},
        }
        settings_file = tmp_path / "settings.yaml"
        import yaml

        settings_file.write_text(yaml.dump(settings))

        # Track if close was called
        close_called = []
        original_close = None

        def track_close(self: object) -> None:
            close_called.append(True)
            if original_close:
                original_close(self)

        from elspeth.core.landscape import LandscapeDB

        original_close = LandscapeDB.close

        with patch.object(LandscapeDB, "close", track_close):
            result = runner.invoke(app, ["run", "--settings", str(settings_file), "--execute"])

        # Expect failure (file doesn't exist)
        assert result.exit_code != 0

        # But close should still be called (finally block)
        assert len(close_called) >= 1, (
            "LandscapeDB.close() was not called after pipeline failure. The try/finally block should ensure cleanup on all code paths."
        )


class TestRunCommandProgress:
    """Tests for progress output during pipeline execution."""

    @pytest.fixture
    def multi_row_data(self, tmp_path: Path) -> Path:
        """Create sample input data with 150 rows for progress testing."""
        csv_file = tmp_path / "multi_row_input.csv"
        lines = ["id,name,value"]
        for i in range(150):
            lines.append(f"{i},item_{i},{i * 10}")
        csv_file.write_text("\n".join(lines))
        return csv_file

    @pytest.fixture
    def progress_settings(self, tmp_path: Path, multi_row_data: Path) -> Path:
        """Create pipeline configuration for progress testing."""
        output_file = tmp_path / "progress_output.json"
        landscape_db = tmp_path / "landscape.db"
        settings = {
            "datasource": {
                "plugin": "csv",
                "options": {
                    "path": str(multi_row_data),
                    "on_validation_failure": "discard",
                    "schema": {"fields": "dynamic"},
                },
            },
            "sinks": {
                "default": {
                    "plugin": "json",
                    "options": {
                        "path": str(output_file),
                        "schema": {"fields": "dynamic"},
                    },
                },
            },
            "output_sink": "default",
            "landscape": {"url": f"sqlite:///{landscape_db}"},
        }
        settings_file = tmp_path / "settings.yaml"
        settings_file.write_text(yaml.dump(settings))
        return settings_file

    def test_run_shows_progress_output(self, progress_settings: Path) -> None:
        """run --execute shows progress lines during execution."""
        result = runner.invoke(app, ["run", "--settings", str(progress_settings), "--execute"])
        assert result.exit_code == 0

        # Check for progress output format: "Processing: X rows | ..."
        assert "Processing:" in result.output
        assert "rows" in result.output
        assert "rows/sec" in result.output

    def test_run_progress_shows_counters(self, progress_settings: Path) -> None:
        """run --execute progress shows success/fail/quarantine counters."""
        result = runner.invoke(app, ["run", "--settings", str(progress_settings), "--execute"])
        assert result.exit_code == 0

        # Check for counter symbols in output
        # Format: ✓N ✗N ⚠N
        assert "✓" in result.output  # success
        assert "✗" in result.output  # failed
        assert "⚠" in result.output  # quarantined
