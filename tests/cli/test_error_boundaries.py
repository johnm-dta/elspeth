"""Tests for CLI error boundary handling.

These tests verify that the CLI handles error conditions gracefully,
showing helpful messages instead of raw tracebacks.

RC-1 Blocker: CLI error paths are under-tested (test-gaps-analysis.md).
"""

from __future__ import annotations

import json
import os
import stat
from pathlib import Path

import pytest
import yaml
from typer.testing import CliRunner

from elspeth.cli import app

runner = CliRunner()


class TestYamlParsingErrors:
    """Test that YAML syntax errors show helpful messages, not tracebacks."""

    def test_run_yaml_syntax_error_shows_helpful_message(self, tmp_path: Path) -> None:
        """Invalid YAML syntax shows 'syntax error' message, not raw traceback.

        This is the key error boundary test - users should see:
          "YAML syntax error in settings.yaml: ..."
        Not:
          "Traceback (most recent call last): ..."
        """
        config_file = tmp_path / "invalid_syntax.yaml"
        # Unclosed bracket is a scanner error
        config_file.write_text("source:\n  plugin: csv\n  options: [invalid")

        result = runner.invoke(app, ["run", "-s", str(config_file)])

        assert result.exit_code == 1
        # Should show helpful message, not traceback
        output = result.output.lower()
        assert ("yaml" in output and "syntax" in output) or "error" in output
        # Should NOT show Python traceback
        assert "traceback" not in output.lower()

    def test_validate_yaml_syntax_error_shows_helpful_message(self, tmp_path: Path) -> None:
        """Validate command also catches YAML syntax errors gracefully."""
        config_file = tmp_path / "invalid_syntax.yaml"
        config_file.write_text("source:\n  plugin: csv\n  options: {unclosed")

        result = runner.invoke(app, ["validate", "-s", str(config_file)])

        assert result.exit_code == 1
        output = result.output.lower()
        assert ("yaml" in output and "syntax" in output) or "error" in output
        assert "traceback" not in output.lower()

    def test_run_yaml_with_tabs_shows_helpful_error(self, tmp_path: Path) -> None:
        """Tab indentation (common mistake) shows helpful error.

        YAML doesn't allow tabs for indentation - this is a frequent user error.
        The error message should help them understand the problem.
        """
        config_file = tmp_path / "tabs.yaml"
        # Use actual tab character
        config_file.write_text("source:\n\tplugin: csv")  # Tab before 'plugin'

        result = runner.invoke(app, ["run", "-s", str(config_file)])

        assert result.exit_code == 1
        output = result.output.lower()
        # Should mention the YAML error (tabs not allowed)
        assert "yaml" in output or "tab" in output or "syntax" in output or "error" in output
        assert "traceback" not in output.lower()

    def test_run_yaml_duplicate_key_error(self, tmp_path: Path) -> None:
        """Duplicate YAML keys show helpful error."""
        config_file = tmp_path / "duplicate_keys.yaml"
        config_file.write_text("""
source:
  plugin: csv
source:
  plugin: json
""")

        result = runner.invoke(app, ["run", "-s", str(config_file)])

        # Should exit non-zero (either caught as YAML error or validation error)
        assert result.exit_code != 0
        assert "traceback" not in result.output.lower()


class TestDatabaseConnectionErrors:
    """Test database connection error handling."""

    def test_run_sqlite_path_not_writable(self, tmp_path: Path) -> None:
        """Unwritable database path shows clear error message.

        When the database path can't be written to (e.g., permission denied),
        users should see a clear error, not a traceback.
        """
        csv_file = tmp_path / "input.csv"
        csv_file.write_text("id,value\n1,100\n")

        # Create a directory we can't write to
        readonly_dir = tmp_path / "readonly"
        readonly_dir.mkdir()

        settings = {
            "source": {
                "plugin": "csv",
                "options": {
                    "path": str(csv_file),
                    "on_validation_failure": "discard",
                    "schema": {"fields": "dynamic"},
                },
            },
            "sinks": {
                "output": {
                    "plugin": "json",
                    "options": {
                        "path": str(tmp_path / "output.json"),
                        "schema": {"fields": "dynamic"},
                    },
                },
            },
            "default_sink": "output",
            # Put database in unwritable directory
            "landscape": {"url": f"sqlite:///{readonly_dir}/cannot_write.db"},
        }
        settings_file = tmp_path / "settings.yaml"
        settings_file.write_text(yaml.dump(settings))

        # Make directory read-only
        original_mode = readonly_dir.stat().st_mode
        try:
            readonly_dir.chmod(stat.S_IRUSR | stat.S_IXUSR)

            result = runner.invoke(app, ["run", "-s", str(settings_file), "--execute"])

            # Should fail (can't create database)
            assert result.exit_code != 0
            # Error message should be present (not silent failure)
            assert result.output.strip() != ""
        finally:
            # Restore permissions for cleanup
            readonly_dir.chmod(original_mode)

    @pytest.mark.skipif(
        os.geteuid() == 0,
        reason="Root user can write anywhere, permission test won't work",
    )
    def test_run_sqlite_path_permission_denied(self, tmp_path: Path) -> None:
        """Permission denied on database shows clear error.

        Skip on root - root can write anywhere so the test would fail.
        """
        csv_file = tmp_path / "input.csv"
        csv_file.write_text("id,value\n1,100\n")

        # Create database file and make it read-only
        db_file = tmp_path / "readonly.db"
        db_file.touch()
        original_mode = db_file.stat().st_mode
        db_file.chmod(stat.S_IRUSR)

        settings = {
            "source": {
                "plugin": "csv",
                "options": {
                    "path": str(csv_file),
                    "on_validation_failure": "discard",
                    "schema": {"fields": "dynamic"},
                },
            },
            "sinks": {
                "output": {
                    "plugin": "json",
                    "options": {
                        "path": str(tmp_path / "output.json"),
                        "schema": {"fields": "dynamic"},
                    },
                },
            },
            "default_sink": "output",
            "landscape": {"url": f"sqlite:///{db_file}"},
        }
        settings_file = tmp_path / "settings.yaml"
        settings_file.write_text(yaml.dump(settings))

        try:
            result = runner.invoke(app, ["run", "-s", str(settings_file), "--execute"])

            # Should fail (can't write to read-only file)
            assert result.exit_code != 0
        finally:
            # Restore permissions for cleanup
            db_file.chmod(original_mode)


class TestSourceFileErrors:
    """Test source file error handling."""

    def test_run_source_file_not_found_shows_path(self, tmp_path: Path) -> None:
        """Missing source file shows the actual path in the error.

        Users need to know WHICH file is missing, not just "file not found".
        """
        missing_file = tmp_path / "nonexistent_data.csv"

        settings = {
            "source": {
                "plugin": "csv",
                "options": {
                    "path": str(missing_file),
                    "on_validation_failure": "discard",
                    "schema": {"fields": "dynamic"},
                },
            },
            "sinks": {
                "output": {
                    "plugin": "json",
                    "options": {
                        "path": str(tmp_path / "output.json"),
                        "schema": {"fields": "dynamic"},
                    },
                },
            },
            "default_sink": "output",
            "landscape": {"url": f"sqlite:///{tmp_path / 'landscape.db'}"},
        }
        settings_file = tmp_path / "settings.yaml"
        settings_file.write_text(yaml.dump(settings))

        result = runner.invoke(app, ["run", "-s", str(settings_file), "--execute"])

        assert result.exit_code != 0
        # Error should mention the missing file path
        assert "nonexistent_data.csv" in result.output or "not found" in result.output.lower()

    @pytest.mark.skipif(
        os.geteuid() == 0,
        reason="Root user can read anything, permission test won't work",
    )
    def test_run_source_file_permission_denied(self, tmp_path: Path) -> None:
        """Unreadable source file shows permission error."""
        # Create file but make it unreadable
        csv_file = tmp_path / "unreadable.csv"
        csv_file.write_text("id,value\n1,100\n")
        original_mode = csv_file.stat().st_mode
        csv_file.chmod(0)  # No permissions

        settings = {
            "source": {
                "plugin": "csv",
                "options": {
                    "path": str(csv_file),
                    "on_validation_failure": "discard",
                    "schema": {"fields": "dynamic"},
                },
            },
            "sinks": {
                "output": {
                    "plugin": "json",
                    "options": {
                        "path": str(tmp_path / "output.json"),
                        "schema": {"fields": "dynamic"},
                    },
                },
            },
            "default_sink": "output",
            "landscape": {"url": f"sqlite:///{tmp_path / 'landscape.db'}"},
        }
        settings_file = tmp_path / "settings.yaml"
        settings_file.write_text(yaml.dump(settings))

        try:
            result = runner.invoke(app, ["run", "-s", str(settings_file), "--execute"])

            assert result.exit_code != 0
            # Should show some error about the file
            output_lower = result.output.lower()
            assert "permission" in output_lower or "denied" in output_lower or "error" in output_lower
        finally:
            # Restore permissions for cleanup
            csv_file.chmod(original_mode)


class TestExitCodeConsistency:
    """Test that exit codes are consistent across different error types."""

    def test_exit_code_zero_on_success(self, tmp_path: Path) -> None:
        """Successful run returns exit code 0."""
        csv_file = tmp_path / "input.csv"
        csv_file.write_text("id,value\n1,100\n")

        settings = {
            "source": {
                "plugin": "csv",
                "options": {
                    "path": str(csv_file),
                    "on_validation_failure": "discard",
                    "schema": {"fields": "dynamic"},
                },
            },
            "sinks": {
                "output": {
                    "plugin": "json",
                    "options": {
                        "path": str(tmp_path / "output.json"),
                        "schema": {"fields": "dynamic"},
                    },
                },
            },
            "default_sink": "output",
            "landscape": {"url": f"sqlite:///{tmp_path / 'landscape.db'}"},
        }
        settings_file = tmp_path / "settings.yaml"
        settings_file.write_text(yaml.dump(settings))

        result = runner.invoke(app, ["run", "-s", str(settings_file), "--execute"])

        assert result.exit_code == 0

    def test_exit_code_one_on_config_error(self, tmp_path: Path) -> None:
        """Configuration validation error returns exit code 1."""
        config_file = tmp_path / "invalid.yaml"
        # Missing required 'source' field
        config_file.write_text("""
sinks:
  output:
    plugin: json
default_sink: output
""")

        result = runner.invoke(app, ["run", "-s", str(config_file)])

        assert result.exit_code == 1

    def test_exit_code_one_on_file_not_found(self) -> None:
        """Missing settings file returns exit code 1."""
        result = runner.invoke(app, ["run", "-s", "/nonexistent/settings.yaml"])

        assert result.exit_code == 1

    def test_exit_code_one_on_execution_error(self, tmp_path: Path) -> None:
        """Runtime error during execution returns exit code 1."""
        # Reference a missing source file to cause execution error
        settings = {
            "source": {
                "plugin": "csv",
                "options": {
                    "path": str(tmp_path / "missing.csv"),
                    "on_validation_failure": "discard",
                    "schema": {"fields": "dynamic"},
                },
            },
            "sinks": {
                "output": {
                    "plugin": "json",
                    "options": {
                        "path": str(tmp_path / "output.json"),
                        "schema": {"fields": "dynamic"},
                    },
                },
            },
            "default_sink": "output",
            "landscape": {"url": f"sqlite:///{tmp_path / 'landscape.db'}"},
        }
        settings_file = tmp_path / "settings.yaml"
        settings_file.write_text(yaml.dump(settings))

        result = runner.invoke(app, ["run", "-s", str(settings_file), "--execute"])

        assert result.exit_code == 1

    def test_exit_code_one_for_missing_database(self) -> None:
        """explain --json returns exit code 1 when database can't be resolved.

        Without a --database or settings.yaml, the command returns an error.
        """
        result = runner.invoke(app, ["explain", "--run", "test-run", "--token", "tok-abc", "--json"])

        assert result.exit_code == 1

    def test_exit_code_one_for_explain_no_tui_missing_database(self) -> None:
        """explain --no-tui returns exit code 1 when database can't be resolved."""
        result = runner.invoke(app, ["explain", "--run", "test-run", "--token", "tok-abc", "--no-tui"])

        assert result.exit_code == 1


class TestJsonModeErrors:
    """Test that errors in JSON mode output valid JSON."""

    def test_json_mode_error_is_valid_json(self, tmp_path: Path) -> None:
        """Errors with --format json return structured JSON on stderr.

        When the pipeline fails with --format json, the error should be
        valid JSON that can be parsed by automation tools.
        """
        # Use missing source file to trigger execution error
        settings = {
            "source": {
                "plugin": "csv",
                "options": {
                    "path": str(tmp_path / "missing.csv"),
                    "on_validation_failure": "discard",
                    "schema": {"fields": "dynamic"},
                },
            },
            "sinks": {
                "output": {
                    "plugin": "json",
                    "options": {
                        "path": str(tmp_path / "output.json"),
                        "schema": {"fields": "dynamic"},
                    },
                },
            },
            "default_sink": "output",
            "landscape": {"url": f"sqlite:///{tmp_path / 'landscape.db'}"},
        }
        settings_file = tmp_path / "settings.yaml"
        settings_file.write_text(yaml.dump(settings))

        result = runner.invoke(app, ["run", "-s", str(settings_file), "--execute", "--format", "json"])

        assert result.exit_code != 0
        # The error output should contain valid JSON with error info
        # Look for JSON object in output
        output = result.output
        # Find JSON error object - might be on stderr or stdout
        if '{"event": "error"' in output:
            # Extract the JSON line
            for line in output.split("\n"):
                if '{"event": "error"' in line:
                    error_obj = json.loads(line)
                    assert error_obj["event"] == "error"
                    assert "error" in error_obj
                    break

    def test_explain_json_returns_valid_json_on_error(self) -> None:
        """explain --json returns valid JSON even for error responses."""
        result = runner.invoke(app, ["explain", "--run", "test-run", "--token", "tok-abc", "--json"])

        assert result.exit_code == 1
        # Output should be valid JSON with error key
        response = json.loads(result.output)
        assert "error" in response


class TestValidateCommandErrorBoundaries:
    """Test validate command error boundaries."""

    def test_validate_shows_pydantic_errors_clearly(self, tmp_path: Path) -> None:
        """Pydantic validation errors show field paths clearly."""
        config_file = tmp_path / "invalid.yaml"
        config_file.write_text("""
source:
  plugin: csv
sinks:
  output:
    plugin: json
default_sink: nonexistent
concurrency:
  max_workers: -5
""")

        result = runner.invoke(app, ["validate", "-s", str(config_file)])

        assert result.exit_code == 1
        output = result.output.lower()
        # Should show validation failure and field paths
        assert "configuration validation failed" in output or "validation" in output
        # Should show the specific field that failed
        assert "concurrency.max_workers" in output

    def test_validate_file_not_found_shows_path(self) -> None:
        """Missing settings file shows the attempted path."""
        result = runner.invoke(app, ["validate", "-s", "/nonexistent/path/settings.yaml"])

        assert result.exit_code == 1
        output = result.output.lower()
        assert "not found" in output
        # Should show the path
        assert "nonexistent" in output or "settings.yaml" in output
