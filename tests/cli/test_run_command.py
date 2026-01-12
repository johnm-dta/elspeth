"""Tests for elspeth run command."""

from pathlib import Path

import pytest
import yaml
from typer.testing import CliRunner

runner = CliRunner()


class TestRunCommand:
    """Tests for run command."""

    @pytest.fixture
    def sample_data(self, tmp_path: Path) -> Path:
        """Create sample input data."""
        csv_file = tmp_path / "input.csv"
        csv_file.write_text("id,name,value\n1,alice,100\n2,bob,200\n")
        return csv_file

    @pytest.fixture
    def pipeline_settings(self, tmp_path: Path, sample_data: Path) -> Path:
        """Create a complete pipeline configuration."""
        output_file = tmp_path / "output.json"
        landscape_db = tmp_path / "landscape.db"
        settings = {
            "source": {"plugin": "csv", "path": str(sample_data)},
            # Use "default" sink name - Orchestrator routes completed rows to "default"
            "sinks": {"default": {"plugin": "json", "path": str(output_file)}},
            # Use temp-path DB to avoid polluting CWD during tests
            "landscape": {"url": f"sqlite:///{landscape_db}"},
        }
        settings_file = tmp_path / "settings.yaml"
        settings_file.write_text(yaml.dump(settings))
        return settings_file

    def test_run_executes_pipeline(
        self, pipeline_settings: Path, tmp_path: Path
    ) -> None:
        """run executes pipeline and creates output."""
        from elspeth.cli import app

        result = runner.invoke(app, ["run", "--settings", str(pipeline_settings)])
        assert result.exit_code == 0

        # Check output was created
        output_file = tmp_path / "output.json"
        assert output_file.exists()

    def test_run_shows_summary(self, pipeline_settings: Path) -> None:
        """run shows execution summary."""
        from elspeth.cli import app

        result = runner.invoke(app, ["run", "--settings", str(pipeline_settings)])
        assert result.exit_code == 0
        # Use result.output to include both stdout and stderr
        assert "completed" in result.output.lower() or "rows" in result.output.lower()

    def test_run_missing_settings(self) -> None:
        """run exits non-zero for missing settings file."""
        from elspeth.cli import app

        result = runner.invoke(app, ["run", "--settings", "/nonexistent.yaml"])
        assert result.exit_code != 0

    def test_run_dry_run_mode(self, pipeline_settings: Path) -> None:
        """run --dry-run validates without executing."""
        from elspeth.cli import app

        result = runner.invoke(
            app, ["run", "--settings", str(pipeline_settings), "--dry-run"]
        )
        assert result.exit_code == 0
        assert "dry" in result.output.lower() or "would" in result.output.lower()
