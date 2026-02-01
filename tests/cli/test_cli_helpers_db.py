"""Tests for CLI database resolution helpers."""

from pathlib import Path

import pytest


class TestResolveDatabaseUrl:
    """Tests for resolve_database_url helper."""

    def test_explicit_database_path_takes_precedence(self, tmp_path: Path) -> None:
        """--database CLI option overrides settings.yaml."""
        from elspeth.cli_helpers import resolve_database_url

        db_path = tmp_path / "explicit.db"
        db_path.touch()

        url, _ = resolve_database_url(database=str(db_path), settings_path=None)

        assert url == f"sqlite:///{db_path.resolve()}"

    def test_raises_when_database_file_not_found(self, tmp_path: Path) -> None:
        """Raises ValueError when --database points to nonexistent file."""
        from elspeth.cli_helpers import resolve_database_url

        nonexistent = tmp_path / "nonexistent.db"

        with pytest.raises(ValueError, match="not found"):
            resolve_database_url(database=str(nonexistent), settings_path=None)

    def test_loads_from_settings_yaml(self, tmp_path: Path) -> None:
        """Falls back to settings.yaml landscape.url."""
        from elspeth.cli_helpers import resolve_database_url

        settings_file = tmp_path / "settings.yaml"
        settings_file.write_text("""
landscape:
  url: sqlite:///./state/audit.db
source:
  plugin: csv
  options:
    path: input.csv
sinks:
  output:
    plugin: csv
    options:
      path: output.csv
default_sink: output
""")

        url, _ = resolve_database_url(database=None, settings_path=settings_file)

        assert url == "sqlite:///./state/audit.db"

    def test_uses_default_landscape_url_when_not_specified(self, tmp_path: Path) -> None:
        """Uses default landscape.url when not explicitly set in settings."""
        from elspeth.cli_helpers import resolve_database_url

        settings_file = tmp_path / "settings.yaml"
        # Valid config without explicit landscape - should use default
        settings_file.write_text("""
source:
  plugin: csv
  options:
    path: input.csv
sinks:
  output:
    plugin: csv
    options:
      path: output.csv
default_sink: output
""")

        url, config = resolve_database_url(database=None, settings_path=settings_file)

        # Should get the default landscape URL
        assert url == "sqlite:///./state/audit.db"
        assert config is not None

    def test_uses_default_settings_yaml_when_no_explicit_path(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        """Falls back to default settings.yaml when no explicit --settings provided."""
        from elspeth.cli_helpers import resolve_database_url

        monkeypatch.chdir(tmp_path)
        settings_file = tmp_path / "settings.yaml"
        settings_file.write_text("""
landscape:
  url: sqlite:///./state/audit.db
source:
  plugin: csv
  options:
    path: input.csv
sinks:
  output:
    plugin: csv
    options:
      path: output.csv
default_sink: output
""")

        url, _ = resolve_database_url(database=None, settings_path=None)

        assert url == "sqlite:///./state/audit.db"

    def test_raises_when_no_database_and_no_settings(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        """Raises ValueError when neither database nor settings provided."""
        from elspeth.cli_helpers import resolve_database_url

        monkeypatch.chdir(tmp_path)

        with pytest.raises(ValueError, match="No database specified"):
            resolve_database_url(database=None, settings_path=None)

    def test_raises_when_explicit_settings_path_missing(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        """Raises ValueError when --settings path is provided but missing."""
        from elspeth.cli_helpers import resolve_database_url

        monkeypatch.chdir(tmp_path)
        settings_file = tmp_path / "settings.yaml"
        settings_file.write_text("""
landscape:
  url: sqlite:///./state/audit.db
source:
  plugin: csv
  options:
    path: input.csv
sinks:
  output:
    plugin: csv
    options:
      path: output.csv
default_sink: output
""")

        missing_settings = tmp_path / "explicit.yaml"

        with pytest.raises(ValueError, match="Settings file not found"):
            resolve_database_url(database=None, settings_path=missing_settings)

    def test_raises_when_default_settings_fails(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        """Raises ValueError with context when default settings.yaml fails to load."""
        from elspeth.cli_helpers import resolve_database_url

        # Create invalid settings file in current directory
        monkeypatch.chdir(tmp_path)
        settings_file = tmp_path / "settings.yaml"
        settings_file.write_text("invalid: yaml: content: [")

        with pytest.raises(ValueError, match=r"settings\.yaml"):
            resolve_database_url(database=None, settings_path=None)


class TestResolveLatestRunId:
    """Tests for resolve_latest_run_id helper."""

    def test_returns_most_recent_run(self) -> None:
        """'latest' resolves to most recently started run."""
        from elspeth.cli_helpers import resolve_latest_run_id
        from elspeth.core.landscape import LandscapeDB, LandscapeRecorder

        db = LandscapeDB.in_memory()
        recorder = LandscapeRecorder(db)

        # Create two runs - second one is "latest"
        run1 = recorder.begin_run(config={}, canonical_version="v1")
        run2 = recorder.begin_run(config={}, canonical_version="v1")

        result = resolve_latest_run_id(recorder)

        # run2 was created after run1, so it's latest
        assert result == run2.run_id
        # Sanity check: run1 should be different
        assert run1.run_id != run2.run_id

    def test_returns_none_when_no_runs(self) -> None:
        """Returns None when database has no runs."""
        from elspeth.cli_helpers import resolve_latest_run_id
        from elspeth.core.landscape import LandscapeDB, LandscapeRecorder

        db = LandscapeDB.in_memory()
        recorder = LandscapeRecorder(db)

        result = resolve_latest_run_id(recorder)

        assert result is None

    def test_passthrough_non_latest_run_id(self) -> None:
        """Non-'latest' run_id passed through unchanged."""
        from elspeth.cli_helpers import resolve_run_id
        from elspeth.core.landscape import LandscapeDB, LandscapeRecorder

        db = LandscapeDB.in_memory()
        recorder = LandscapeRecorder(db)

        result = resolve_run_id("explicit-run-id", recorder)

        assert result == "explicit-run-id"

    def test_latest_keyword_resolved(self) -> None:
        """'latest' keyword is resolved to actual run_id."""
        from elspeth.cli_helpers import resolve_run_id
        from elspeth.core.landscape import LandscapeDB, LandscapeRecorder

        db = LandscapeDB.in_memory()
        recorder = LandscapeRecorder(db)
        run = recorder.begin_run(config={}, canonical_version="v1")

        result = resolve_run_id("latest", recorder)

        assert result == run.run_id

    def test_latest_keyword_case_insensitive(self) -> None:
        """'LATEST' keyword is case insensitive."""
        from elspeth.cli_helpers import resolve_run_id
        from elspeth.core.landscape import LandscapeDB, LandscapeRecorder

        db = LandscapeDB.in_memory()
        recorder = LandscapeRecorder(db)
        run = recorder.begin_run(config={}, canonical_version="v1")

        result = resolve_run_id("LATEST", recorder)

        assert result == run.run_id
