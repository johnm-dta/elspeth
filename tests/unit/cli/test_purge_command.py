"""Unit tests for the CLI purge command."""

import sqlite3
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
import typer
from typer.testing import CliRunner

from elspeth.cli import _validate_existing_sqlite_db_url, app
from elspeth.core.landscape import LandscapeDB
from elspeth.core.retention.purge import PurgeResult

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
    on_write_failure: discard
    options:
      path: output.csv
"""
    )


def _make_purge_mocks(
    *,
    expired_refs: list[str] | None = None,
    purge_result: PurgeResult | None = None,
    exists_side_effect: bool | None = True,
):
    """Create standard mocks for purge command tests.

    Returns (mock_db, mock_payload_store, mock_purge_manager, patches_dict).
    """
    mock_db = MagicMock()
    mock_payload_store = MagicMock()
    mock_purge_manager = MagicMock()

    if expired_refs is not None:
        mock_purge_manager.find_expired_payload_refs.return_value = expired_refs

    if purge_result is not None:
        mock_purge_manager.purge_payloads.return_value = purge_result

    if exists_side_effect is not None:
        mock_payload_store.exists.return_value = exists_side_effect

    return mock_db, mock_payload_store, mock_purge_manager


_PATCH_LANDSCAPE_DB = "elspeth.core.landscape.LandscapeDB"
_PATCH_FS_PAYLOAD_STORE = "elspeth.core.payload_store.FilesystemPayloadStore"
_PATCH_PURGE_MANAGER = "elspeth.core.retention.purge.PurgeManager"
_PATCH_RESOLVE_PASSPHRASE = "elspeth.cli_helpers.resolve_audit_passphrase"


def _invoke_purge_with_mocks(
    tmp_path: Path,
    mock_db: MagicMock,
    mock_payload_store: MagicMock,
    mock_purge_manager: MagicMock,
    *,
    extra_args: list[str] | None = None,
    db_file_name: str = "audit.db",
):
    """Invoke the purge CLI command with mocked internals.

    Creates a fake DB file, patches LandscapeDB, FilesystemPayloadStore,
    PurgeManager, and resolve_audit_passphrase, then invokes the CLI.
    """
    db_path = tmp_path / db_file_name
    db_path.touch()

    args = ["purge", "--database", str(db_path), "--yes"]
    if extra_args:
        args.extend(extra_args)

    with (
        patch(_PATCH_LANDSCAPE_DB) as mock_ldb_cls,
        patch(_PATCH_FS_PAYLOAD_STORE, return_value=mock_payload_store),
        patch(_PATCH_PURGE_MANAGER, return_value=mock_purge_manager),
        patch(_PATCH_RESOLVE_PASSPHRASE, return_value=None),
    ):
        mock_ldb_cls.from_url.return_value = mock_db
        result = runner.invoke(app, args)

    return result


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
        """settings.yaml SQLite URL works when the file is a real audit database."""
        state_dir = tmp_path / "state"
        state_dir.mkdir()
        existing_db = state_dir / "audit.db"
        db = LandscapeDB.from_url(f"sqlite:///{existing_db}")
        db.close()

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


class TestPurgeBasicFlow:
    """Tests for the core purge execution path."""

    def test_purge_deletes_expired_payloads(self, tmp_path: Path) -> None:
        """Basic purge deletes expired payloads and reports results."""
        refs = ["aabb" * 16, "ccdd" * 16]
        result_obj = PurgeResult(
            deleted_count=2,
            skipped_count=0,
            failed_refs=(),
            grade_update_failures=(),
            duration_seconds=0.42,
        )
        mock_db, mock_ps, mock_pm = _make_purge_mocks(expired_refs=refs, purge_result=result_obj)

        result = _invoke_purge_with_mocks(tmp_path, mock_db, mock_ps, mock_pm)

        assert result.exit_code == 0
        assert "Purge completed" in result.output
        assert "Deleted: 2" in result.output
        assert "Skipped (not found): 0" in result.output
        mock_pm.purge_payloads.assert_called_once_with(refs)
        mock_db.close.assert_called_once()

    def test_purge_no_expired_payloads(self, tmp_path: Path) -> None:
        """When nothing is expired, purge exits cleanly with a message."""
        mock_db, mock_ps, mock_pm = _make_purge_mocks(expired_refs=[])

        result = _invoke_purge_with_mocks(tmp_path, mock_db, mock_ps, mock_pm)

        assert result.exit_code == 0
        assert "No payloads older than" in result.output
        mock_pm.purge_payloads.assert_not_called()
        mock_db.close.assert_called_once()

    def test_purge_reports_failed_refs(self, tmp_path: Path) -> None:
        """Failed deletions are reported in the output."""
        bad_ref = "dead" * 16
        refs = [bad_ref]
        result_obj = PurgeResult(
            deleted_count=0,
            skipped_count=0,
            failed_refs=(bad_ref,),
            grade_update_failures=(),
            duration_seconds=0.01,
        )
        mock_db, mock_ps, mock_pm = _make_purge_mocks(expired_refs=refs, purge_result=result_obj)

        result = _invoke_purge_with_mocks(tmp_path, mock_db, mock_ps, mock_pm)

        assert result.exit_code == 0
        assert "Failed: 1" in result.output
        assert bad_ref[:16] in result.output

    def test_purge_reports_skipped_refs(self, tmp_path: Path) -> None:
        """Skipped (already-purged) refs are reported."""
        refs = ["abcd" * 16]
        result_obj = PurgeResult(
            deleted_count=0,
            skipped_count=1,
            failed_refs=(),
            grade_update_failures=(),
            duration_seconds=0.05,
        )
        mock_db, mock_ps, mock_pm = _make_purge_mocks(expired_refs=refs, purge_result=result_obj)

        result = _invoke_purge_with_mocks(tmp_path, mock_db, mock_ps, mock_pm)

        assert result.exit_code == 0
        assert "Skipped (not found): 1" in result.output

    def test_purge_closes_db_on_success(self, tmp_path: Path) -> None:
        """Database is closed even on successful purge."""
        mock_db, mock_ps, mock_pm = _make_purge_mocks(expired_refs=[])

        _invoke_purge_with_mocks(tmp_path, mock_db, mock_ps, mock_pm)

        mock_db.close.assert_called_once()

    def test_purge_closes_db_on_purge_error(self, tmp_path: Path) -> None:
        """Database is closed even when purge_payloads raises."""
        refs = ["aabb" * 16]
        mock_db, mock_ps, mock_pm = _make_purge_mocks(expired_refs=refs)
        mock_pm.purge_payloads.side_effect = OSError("disk full")

        _invoke_purge_with_mocks(tmp_path, mock_db, mock_ps, mock_pm)

        # The exception propagates (not caught by CLI), but finally block runs
        mock_db.close.assert_called_once()


class TestPurgeDryRun:
    """Tests for --dry-run mode."""

    def test_dry_run_does_not_delete(self, tmp_path: Path) -> None:
        """Dry run shows what would be deleted but does NOT call purge_payloads."""
        refs = ["aabb" * 16, "ccdd" * 16]
        mock_db, mock_ps, mock_pm = _make_purge_mocks(expired_refs=refs)

        result = _invoke_purge_with_mocks(tmp_path, mock_db, mock_ps, mock_pm, extra_args=["--dry-run"])

        assert result.exit_code == 0
        assert "Would delete 2 payload(s)" in result.output
        mock_pm.purge_payloads.assert_not_called()

    def test_dry_run_shows_existence_status(self, tmp_path: Path) -> None:
        """Dry run checks existence of each ref and shows status."""
        refs = ["aabb" * 16, "ccdd" * 16]
        mock_db, mock_ps, mock_pm = _make_purge_mocks(expired_refs=refs)
        # First ref exists, second does not
        mock_ps.exists.side_effect = [True, False]

        result = _invoke_purge_with_mocks(tmp_path, mock_db, mock_ps, mock_pm, extra_args=["--dry-run"])

        assert result.exit_code == 0
        assert "exists" in result.output
        assert "already deleted" in result.output

    def test_dry_run_truncates_long_list(self, tmp_path: Path) -> None:
        """Dry run shows first 10 refs and a count of remaining."""
        refs = [f"{i:064x}" for i in range(15)]
        mock_db, mock_ps, mock_pm = _make_purge_mocks(expired_refs=refs)

        result = _invoke_purge_with_mocks(tmp_path, mock_db, mock_ps, mock_pm, extra_args=["--dry-run"])

        assert result.exit_code == 0
        assert "... and 5 more" in result.output

    def test_dry_run_no_expired_payloads(self, tmp_path: Path) -> None:
        """Dry run with nothing expired shows the 'no payloads' message."""
        mock_db, mock_ps, mock_pm = _make_purge_mocks(expired_refs=[])

        result = _invoke_purge_with_mocks(tmp_path, mock_db, mock_ps, mock_pm, extra_args=["--dry-run"])

        assert result.exit_code == 0
        assert "No payloads older than" in result.output


class TestPurgeErrorHandling:
    """Tests for error conditions in the purge command."""

    def test_purge_rejects_non_landscape_db_without_mutating_file(self, tmp_path: Path) -> None:
        """Wrong SQLite files must fail fast without schema creation side effects."""
        db_path = tmp_path / "wrong.db"
        conn = sqlite3.connect(str(db_path))
        conn.execute("CREATE TABLE unrelated (id INTEGER PRIMARY KEY)")
        conn.commit()
        before_tables = [row[0] for row in conn.execute("SELECT name FROM sqlite_master WHERE type='table' ORDER BY name")]
        before_user_version = conn.execute("PRAGMA user_version").fetchone()[0]
        conn.close()

        with patch(_PATCH_RESOLVE_PASSPHRASE, return_value=None):
            result = runner.invoke(app, ["purge", "--database", str(db_path), "--dry-run"])

        conn = sqlite3.connect(str(db_path))
        after_tables = [row[0] for row in conn.execute("SELECT name FROM sqlite_master WHERE type='table' ORDER BY name")]
        after_user_version = conn.execute("PRAGMA user_version").fetchone()[0]
        conn.close()

        assert result.exit_code == 1
        assert "does not appear to be an ELSPETH audit database" in result.output
        assert after_tables == before_tables
        assert after_user_version == before_user_version == 0

    def test_missing_database_file(self, tmp_path: Path) -> None:
        """--database pointing to nonexistent file fails with clear error."""
        missing = tmp_path / "nonexistent.db"

        result = runner.invoke(app, ["purge", "--database", str(missing), "--dry-run"])

        assert result.exit_code == 1
        assert "Database file not found" in result.output

    def test_no_database_no_settings(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        """No --database and no settings.yaml fails with guidance."""
        monkeypatch.chdir(tmp_path)

        result = runner.invoke(app, ["purge", "--dry-run"])

        assert result.exit_code == 1
        assert "No settings.yaml found" in result.output
        assert "--database" in result.output

    def test_retention_days_zero_rejected(self, tmp_path: Path) -> None:
        """--retention-days 0 is rejected."""
        db_path = tmp_path / "audit.db"
        db_path.touch()

        with patch(_PATCH_RESOLVE_PASSPHRASE, return_value=None):
            result = runner.invoke(
                app,
                ["purge", "--database", str(db_path), "--retention-days", "0", "--dry-run"],
            )

        assert result.exit_code == 1
        assert "must be greater than 0" in result.output

    def test_retention_days_negative_rejected(self, tmp_path: Path) -> None:
        """--retention-days with negative value is rejected."""
        db_path = tmp_path / "audit.db"
        db_path.touch()

        with patch(_PATCH_RESOLVE_PASSPHRASE, return_value=None):
            result = runner.invoke(
                app,
                ["purge", "--database", str(db_path), "--retention-days", "-5", "--dry-run"],
            )

        assert result.exit_code == 1
        assert "must be greater than 0" in result.output

    def test_database_connection_failure(self, tmp_path: Path) -> None:
        """Database connection error is caught and reported."""
        db_path = tmp_path / "audit.db"
        db_path.touch()

        with (
            patch(_PATCH_LANDSCAPE_DB) as mock_ldb_cls,
            patch(_PATCH_RESOLVE_PASSPHRASE, return_value=None),
        ):
            mock_ldb_cls.from_url.side_effect = RuntimeError("cannot open database")
            result = runner.invoke(app, ["purge", "--database", str(db_path), "--dry-run"])

        assert result.exit_code == 1
        assert "Error connecting to database" in result.output

    def test_yaml_syntax_error_without_database_flag(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        """Malformed settings.yaml without --database fails with YAML error."""
        settings = tmp_path / "settings.yaml"
        settings.write_text(":\n  - :\n  bad: [unclosed")
        monkeypatch.chdir(tmp_path)

        result = runner.invoke(app, ["purge", "--dry-run"])

        assert result.exit_code == 1

    def test_yaml_syntax_error_with_database_flag_warns(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        """Malformed settings.yaml WITH --database continues with warning."""
        settings = tmp_path / "settings.yaml"
        settings.write_text(":\n  - :\n  bad: [unclosed")
        monkeypatch.chdir(tmp_path)

        db_path = tmp_path / "audit.db"
        db_path.touch()

        mock_db, mock_ps, mock_pm = _make_purge_mocks(expired_refs=[])

        with (
            patch(_PATCH_LANDSCAPE_DB) as mock_ldb_cls,
            patch(_PATCH_FS_PAYLOAD_STORE, return_value=mock_ps),
            patch(_PATCH_PURGE_MANAGER, return_value=mock_pm),
            patch(_PATCH_RESOLVE_PASSPHRASE, return_value=None),
        ):
            mock_ldb_cls.from_url.return_value = mock_db
            result = runner.invoke(app, ["purge", "--database", str(db_path), "--dry-run"])

        assert result.exit_code == 0
        assert "Warning" in result.output


class TestPurgeConfirmation:
    """Tests for the --yes / confirmation prompt behavior."""

    def test_purge_aborted_on_decline(self, tmp_path: Path) -> None:
        """Declining confirmation aborts purge."""
        refs = ["aabb" * 16]
        mock_db, mock_ps, mock_pm = _make_purge_mocks(expired_refs=refs)

        db_path = tmp_path / "audit.db"
        db_path.touch()

        with (
            patch(_PATCH_LANDSCAPE_DB) as mock_ldb_cls,
            patch(_PATCH_FS_PAYLOAD_STORE, return_value=mock_ps),
            patch(_PATCH_PURGE_MANAGER, return_value=mock_pm),
            patch(_PATCH_RESOLVE_PASSPHRASE, return_value=None),
        ):
            mock_ldb_cls.from_url.return_value = mock_db
            # No --yes, answer "n" to the prompt
            result = runner.invoke(
                app,
                ["purge", "--database", str(db_path)],
                input="n\n",
            )

        assert result.exit_code == 1
        assert "Aborted" in result.output
        mock_pm.purge_payloads.assert_not_called()


class TestPurgeRetentionDays:
    """Tests for retention days resolution."""

    def test_custom_retention_days(self, tmp_path: Path) -> None:
        """--retention-days is passed through to find_expired_payload_refs."""
        mock_db, mock_ps, mock_pm = _make_purge_mocks(expired_refs=[])

        _invoke_purge_with_mocks(tmp_path, mock_db, mock_ps, mock_pm, extra_args=["--retention-days", "30"])

        mock_pm.find_expired_payload_refs.assert_called_once_with(30)

    def test_default_retention_days(self, tmp_path: Path) -> None:
        """Without --retention-days, defaults to 90."""
        mock_db, mock_ps, mock_pm = _make_purge_mocks(expired_refs=[])

        _invoke_purge_with_mocks(tmp_path, mock_db, mock_ps, mock_pm)

        mock_pm.find_expired_payload_refs.assert_called_once_with(90)


class TestPurgePayloadPath:
    """Tests for payload directory resolution."""

    def test_payload_dir_defaults_to_sibling_of_database(self, tmp_path: Path) -> None:
        """Without --payload-dir, defaults to payloads/ next to database."""
        db_path = tmp_path / "audit.db"
        db_path.touch()

        with (
            patch(_PATCH_LANDSCAPE_DB) as mock_ldb_cls,
            patch(_PATCH_FS_PAYLOAD_STORE) as mock_fps_cls,
            patch(_PATCH_PURGE_MANAGER) as mock_pm_cls,
            patch(_PATCH_RESOLVE_PASSPHRASE, return_value=None),
        ):
            mock_ldb_cls.from_url.return_value = MagicMock()
            mock_pm_instance = MagicMock()
            mock_pm_instance.find_expired_payload_refs.return_value = []
            mock_pm_cls.return_value = mock_pm_instance

            runner.invoke(app, ["purge", "--database", str(db_path), "--yes", "--dry-run"])

            # FilesystemPayloadStore should be created with parent/payloads
            mock_fps_cls.assert_called_once()
            actual_path = mock_fps_cls.call_args[0][0]
            assert actual_path == db_path.parent / "payloads"

    def test_explicit_payload_dir(self, tmp_path: Path) -> None:
        """--payload-dir overrides the default payload path."""
        db_path = tmp_path / "audit.db"
        db_path.touch()
        custom_dir = tmp_path / "my_payloads"

        with (
            patch(_PATCH_LANDSCAPE_DB) as mock_ldb_cls,
            patch(_PATCH_FS_PAYLOAD_STORE) as mock_fps_cls,
            patch(_PATCH_PURGE_MANAGER) as mock_pm_cls,
            patch(_PATCH_RESOLVE_PASSPHRASE, return_value=None),
        ):
            mock_ldb_cls.from_url.return_value = MagicMock()
            mock_pm_instance = MagicMock()
            mock_pm_instance.find_expired_payload_refs.return_value = []
            mock_pm_cls.return_value = mock_pm_instance

            runner.invoke(
                app,
                [
                    "purge",
                    "--database",
                    str(db_path),
                    "--payload-dir",
                    str(custom_dir),
                    "--yes",
                    "--dry-run",
                ],
            )

            mock_fps_cls.assert_called_once()
            actual_path = mock_fps_cls.call_args[0][0]
            assert actual_path == custom_dir
