"""Unit tests for the CLI purge command."""

from pathlib import Path

import pytest
import typer
from typer.testing import CliRunner

from elspeth.cli import _validate_existing_sqlite_db_url, app

runner = CliRunner()


def _write_minimal_settings(settings_path: Path, *, landscape_url: str) -> None:
    settings_path.write_text(
        f"""
landscape:
  url: {landscape_url}
source:
  plugin: csv
  on_success: output
  options:
    path: input.csv
    on_validation_failure: discard
sinks:
  output:
    plugin: csv
    options:
      path: output.csv
"""
    )


class TestPurgeCommandDatabaseResolution:
    """Tests for purge database URL resolution behavior."""

    def test_purge_fails_for_missing_sqlite_file_from_settings(self, tmp_path: Path, monkeypatch) -> None:
        """settings.yaml SQLite URL must point to an existing file."""
        state_dir = tmp_path / "state"
        state_dir.mkdir()
        missing_db = state_dir / "missing.db"

        settings_path = tmp_path / "settings.yaml"
        _write_minimal_settings(settings_path, landscape_url="sqlite:///./state/missing.db")

        monkeypatch.chdir(tmp_path)
        result = runner.invoke(app, ["purge", "--dry-run"])

        assert result.exit_code == 1
        assert "Database file not found (settings.yaml)" in result.output
        assert not missing_db.exists()

    def test_purge_uses_existing_sqlite_file_from_settings(self, tmp_path: Path, monkeypatch) -> None:
        """settings.yaml SQLite URL works when the DB file already exists."""
        state_dir = tmp_path / "state"
        state_dir.mkdir()
        existing_db = state_dir / "audit.db"
        existing_db.touch()

        settings_path = tmp_path / "settings.yaml"
        _write_minimal_settings(settings_path, landscape_url="sqlite:///./state/audit.db")

        monkeypatch.chdir(tmp_path)
        result = runner.invoke(app, ["purge", "--dry-run"])

        assert result.exit_code == 0
        assert "Using database from settings.yaml: sqlite:///./state/audit.db" in result.output
        assert "No payloads older than" in result.output
        assert existing_db.exists()


class TestPurgeSqliteUriValidation:
    """Tests for URI-style SQLite URL validation in purge preflight."""

    def test_accepts_existing_file_uri_database(self, tmp_path: Path) -> None:
        """sqlite:///file:/abs/path.db?uri=true should pass when file exists."""
        db_path = tmp_path / "audit.db"
        db_path.touch()
        url = f"sqlite:///file:{db_path}?uri=true"

        _validate_existing_sqlite_db_url(url, source="settings.yaml")

    def test_rejects_missing_file_uri_database(self, tmp_path: Path) -> None:
        """sqlite:///file:/abs/path.db?uri=true should fail when file is missing."""
        db_path = tmp_path / "missing.db"
        url = f"sqlite:///file:{db_path}?uri=true"

        with pytest.raises(typer.Exit):
            _validate_existing_sqlite_db_url(url, source="settings.yaml")

    def test_accepts_file_memory_uri_database(self) -> None:
        """file::memory URI should not be treated as a filesystem path."""
        _validate_existing_sqlite_db_url("sqlite:///file::memory:?cache=shared&uri=true", source="settings.yaml")
