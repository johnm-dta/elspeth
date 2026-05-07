# tests/unit/cli/test_cli_helpers_db.py
"""Tests for CLI database resolution helpers.

Migrated from tests/cli/test_cli_helpers_db.py.
TestResolveLatestRunId is deferred to integration tier (uses LandscapeDB).
"""

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

    def test_explicit_database_still_rejects_missing_explicit_settings(self, tmp_path: Path) -> None:
        """--database does not silently discard an explicit missing --settings path."""
        from elspeth.cli_helpers import resolve_database_url

        db_path = tmp_path / "explicit.db"
        db_path.touch()

        with pytest.raises(ValueError, match="Settings file not found"):
            resolve_database_url(database=str(db_path), settings_path=tmp_path / "missing.yaml")

    def test_explicit_database_existing_settings_defers_secret_loading_to_cli(self, tmp_path: Path) -> None:
        """--database validates --settings exists but leaves secret loading to explain."""
        from elspeth.cli_helpers import resolve_database_url

        db_path = tmp_path / "explicit.db"
        db_path.touch()
        settings_file = tmp_path / "settings.yaml"
        settings_file.write_text("""
landscape:
  url: sqlite:///./ignored-settings.db
source:
  plugin: csv
  on_success: output
  options:
    path: input.csv
    on_validation_failure: discard
sinks:
  output:
    plugin: csv
    on_write_failure: discard
    options:
      path: output.csv
""")

        url, config = resolve_database_url(database=str(db_path), settings_path=settings_file)

        assert url == f"sqlite:///{db_path.resolve()}"
        assert config is None

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
  on_success: output
  options:
    path: input.csv
    on_validation_failure: discard
sinks:
  output:
    plugin: csv
    on_write_failure: discard
    options:
      path: output.csv
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
  on_success: output
  options:
    path: input.csv
    on_validation_failure: discard
sinks:
  output:
    plugin: csv
    on_write_failure: discard
    options:
      path: output.csv
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
  on_success: output
  options:
    path: input.csv
    on_validation_failure: discard
sinks:
  output:
    plugin: csv
    on_write_failure: discard
    options:
      path: output.csv
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
  on_success: output
  options:
    path: input.csv
    on_validation_failure: discard
sinks:
  output:
    plugin: csv
    on_write_failure: discard
    options:
      path: output.csv
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


class TestResolveAuditPassphrase:
    """Tests for resolve_audit_passphrase helper."""

    def test_returns_passphrase_for_sqlcipher_backend(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """When backend=sqlcipher, reads the configured env var."""
        from unittest.mock import MagicMock

        from elspeth.cli_helpers import resolve_audit_passphrase

        settings = MagicMock()
        settings.backend = "sqlcipher"
        settings.encryption_key_env = "MY_CUSTOM_KEY"
        monkeypatch.setenv("MY_CUSTOM_KEY", "secret-pass")

        result = resolve_audit_passphrase(settings)
        assert result == "secret-pass"

    def test_raises_when_sqlcipher_env_not_set(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """When backend=sqlcipher but env var is missing, raises RuntimeError."""
        from unittest.mock import MagicMock

        from elspeth.cli_helpers import resolve_audit_passphrase

        settings = MagicMock()
        settings.backend = "sqlcipher"
        settings.encryption_key_env = "MISSING_KEY_VAR"
        monkeypatch.delenv("MISSING_KEY_VAR", raising=False)

        with pytest.raises(RuntimeError, match="MISSING_KEY_VAR"):
            resolve_audit_passphrase(settings)

    def test_returns_none_for_sqlite_backend(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Non-sqlcipher backends return None (no encryption)."""
        from unittest.mock import MagicMock

        from elspeth.cli_helpers import resolve_audit_passphrase

        settings = MagicMock()
        settings.backend = "sqlite"
        # Even if ELSPETH_AUDIT_KEY is set, non-sqlcipher backend returns None
        monkeypatch.setenv("ELSPETH_AUDIT_KEY", "should-be-ignored")

        result = resolve_audit_passphrase(settings)
        assert result is None

    def test_no_settings_returns_none_even_with_env_var(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Regression: settings=None must return None even when ELSPETH_AUDIT_KEY is set.

        Without explicit backend=sqlcipher config, passing a passphrase to a
        plain SQLite database causes 'file is not a database'.
        """
        from elspeth.cli_helpers import resolve_audit_passphrase

        monkeypatch.setenv("ELSPETH_AUDIT_KEY", "should-be-ignored")

        result = resolve_audit_passphrase(None)
        assert result is None

    def test_no_settings_returns_none_without_env_var(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """When settings=None and no env var, returns None."""
        from elspeth.cli_helpers import resolve_audit_passphrase

        monkeypatch.delenv("ELSPETH_AUDIT_KEY", raising=False)

        result = resolve_audit_passphrase(None)
        assert result is None

    def test_custom_encryption_key_env(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Regression: custom encryption_key_env is respected, not just ELSPETH_AUDIT_KEY."""
        from unittest.mock import MagicMock

        from elspeth.cli_helpers import resolve_audit_passphrase

        settings = MagicMock()
        settings.backend = "sqlcipher"
        settings.encryption_key_env = "CUSTOM_AUDIT_SECRET"
        monkeypatch.setenv("CUSTOM_AUDIT_SECRET", "custom-value")
        monkeypatch.setenv("ELSPETH_AUDIT_KEY", "default-value")

        result = resolve_audit_passphrase(settings)
        # Must use the custom var, not the default
        assert result == "custom-value"
