"""Tests for ELSPETH CLI."""

from pathlib import Path

from typer.testing import CliRunner

# Note: In Click 8.0+, mix_stderr is no longer a CliRunner parameter.
# Stderr output is combined with stdout by default when using CliRunner.invoke()
runner = CliRunner()


class TestCLIBasics:
    """Test basic CLI functionality."""

    def test_cli_exists(self) -> None:
        """CLI app can be imported."""
        from elspeth.cli import app

        assert app is not None

    def test_version_flag(self) -> None:
        """--version shows version info."""
        from elspeth.cli import app

        result = runner.invoke(app, ["--version"])
        assert result.exit_code == 0
        assert "elspeth" in result.stdout.lower()

    def test_help_flag(self) -> None:
        """--help shows available commands."""
        from elspeth.cli import app

        result = runner.invoke(app, ["--help"])
        assert result.exit_code == 0
        assert "run" in result.stdout
        assert "explain" in result.stdout
        assert "validate" in result.stdout
        assert "plugins" in result.stdout
        assert "resume" in result.stdout
        assert "purge" in result.stdout


class TestRunCommandExecutesTransforms:
    """Verify row_plugins are actually executed."""

    def test_transforms_from_config_are_instantiated(self, tmp_path: Path) -> None:
        """Transforms in row_plugins are instantiated and passed to orchestrator."""
        from typer.testing import CliRunner

        from elspeth.cli import app

        runner = CliRunner()

        # Create input CSV
        input_file = tmp_path / "input.csv"
        input_file.write_text("id,value\n1,hello\n2,world\n")

        output_file = tmp_path / "output.csv"
        audit_db = tmp_path / "audit.db"

        # Config with a passthrough transform
        config_file = tmp_path / "settings.yaml"
        config_file.write_text(f"""
datasource:
  plugin: csv
  options:
    path: "{input_file}"
    schema:
      fields: dynamic

sinks:
  results:
    plugin: csv
    options:
      path: "{output_file}"
      schema:
        fields: dynamic

row_plugins:
  - plugin: passthrough

output_sink: results

landscape:
  enabled: true
  backend: sqlite
  url: "sqlite:///{audit_db}"
""")

        result = runner.invoke(app, ["run", "-s", str(config_file), "--execute", "-v"])

        assert result.exit_code == 0, f"CLI failed: {result.stdout}"
        # Output should exist with data processed
        assert output_file.exists()

    def test_field_mapper_transform_modifies_output(self, tmp_path: Path) -> None:
        """Field mapper should rename columns - proves transform actually runs."""
        from typer.testing import CliRunner

        from elspeth.cli import app

        runner = CliRunner()

        # Create input CSV
        input_file = tmp_path / "input.csv"
        input_file.write_text("id,old_name\n1,hello\n2,world\n")

        output_file = tmp_path / "output.csv"
        audit_db = tmp_path / "audit.db"

        # Config with field_mapper that renames 'old_name' to 'new_name'
        config_file = tmp_path / "settings.yaml"
        config_file.write_text(f"""
datasource:
  plugin: csv
  options:
    path: "{input_file}"
    schema:
      fields: dynamic

sinks:
  results:
    plugin: csv
    options:
      path: "{output_file}"
      schema:
        fields: dynamic

row_plugins:
  - plugin: field_mapper
    options:
      mapping:
        old_name: new_name

output_sink: results

landscape:
  enabled: true
  backend: sqlite
  url: "sqlite:///{audit_db}"
""")

        result = runner.invoke(app, ["run", "-s", str(config_file), "--execute", "-v"])

        assert result.exit_code == 0, f"CLI failed: {result.stdout}"
        assert output_file.exists()

        # Read output and verify the transform was applied
        output_content = output_file.read_text()
        # If transform ran, we should have 'new_name' column, not 'old_name'
        assert "new_name" in output_content, (
            f"Field mapper should have renamed 'old_name' to 'new_name'. "
            f"Output was: {output_content}"
        )


class TestPurgeCommand:
    """Tests for purge CLI command."""

    def test_purge_help(self) -> None:
        """purge --help shows usage."""
        from elspeth.cli import app

        result = runner.invoke(app, ["purge", "--help"])

        assert result.exit_code == 0
        assert "retention" in result.stdout.lower() or "days" in result.stdout.lower()

    def test_purge_dry_run(self, tmp_path: Path) -> None:
        """purge --dry-run shows what would be deleted."""
        from elspeth.cli import app

        result = runner.invoke(
            app,
            [
                "purge",
                "--dry-run",
                "--database",
                str(tmp_path / "test.db"),
            ],
        )

        assert result.exit_code == 0
        assert "would delete" in result.stdout.lower() or "0" in result.stdout

    def test_purge_with_retention_override(self, tmp_path: Path) -> None:
        """purge --retention-days overrides default."""
        from elspeth.cli import app

        result = runner.invoke(
            app,
            [
                "purge",
                "--dry-run",
                "--retention-days",
                "30",
                "--database",
                str(tmp_path / "test.db"),
            ],
        )

        assert result.exit_code == 0

    def test_purge_requires_confirmation(self, tmp_path: Path) -> None:
        """purge without --yes asks for confirmation."""
        from datetime import UTC, datetime, timedelta

        from sqlalchemy import insert

        from elspeth.cli import app
        from elspeth.core.landscape import LandscapeDB
        from elspeth.core.landscape.schema import nodes_table, rows_table, runs_table
        from elspeth.core.payload_store import FilesystemPayloadStore

        # Set up database with old completed run so there's something to purge
        db_file = tmp_path / "landscape.db"
        db_url = f"sqlite:///{db_file}"
        db = LandscapeDB.from_url(db_url)

        # Create payload store with some data
        payload_dir = tmp_path / "payloads"
        store = FilesystemPayloadStore(payload_dir)
        content_hash = store.store(b"test payload data")

        # Create old run (100 days ago) so it's older than retention
        old_date = datetime.now(UTC) - timedelta(days=100)
        with db.connection() as conn:
            conn.execute(
                insert(runs_table).values(
                    run_id="old-run-for-confirm",
                    status="completed",
                    started_at=old_date,
                    completed_at=old_date,
                    config_hash="abc123",
                    settings_json="{}",
                    canonical_version="1.0.0",
                )
            )
            conn.execute(
                insert(nodes_table).values(
                    node_id="source-node-1",
                    run_id="old-run-for-confirm",
                    plugin_name="csv",
                    node_type="source",
                    plugin_version="1.0.0",
                    determinism="deterministic",
                    config_hash="def456",
                    config_json="{}",
                    registered_at=old_date,
                )
            )
            conn.execute(
                insert(rows_table).values(
                    row_id="row-confirm-1",
                    run_id="old-run-for-confirm",
                    source_node_id="source-node-1",
                    row_index=0,
                    source_data_hash="hash1",
                    source_data_ref=content_hash,
                    created_at=old_date,
                )
            )
        db.close()

        result = runner.invoke(
            app,
            ["purge", "--database", str(db_file), "--payload-dir", str(payload_dir)],
            input="n\n",  # Say no to confirmation
        )

        assert result.exit_code == 1
        assert "abort" in result.stdout.lower() or "cancel" in result.stdout.lower()

    def test_purge_with_yes_flag_skips_confirmation(self, tmp_path: Path) -> None:
        """purge --yes skips confirmation prompt."""
        from elspeth.cli import app

        result = runner.invoke(
            app,
            [
                "purge",
                "--yes",
                "--database",
                str(tmp_path / "test.db"),
            ],
        )

        # Should complete without asking for confirmation
        assert result.exit_code == 0
        # Should not ask for confirmation
        assert "confirm" not in result.stdout.lower() or "yes" in result.stdout.lower()

    def test_purge_with_payloads_to_delete(self, tmp_path: Path) -> None:
        """purge deletes expired payloads when present."""
        from datetime import UTC, datetime, timedelta

        from sqlalchemy import insert

        from elspeth.cli import app
        from elspeth.core.landscape import LandscapeDB
        from elspeth.core.landscape.schema import nodes_table, rows_table, runs_table
        from elspeth.core.payload_store import FilesystemPayloadStore

        # Set up database with old completed run
        db_file = tmp_path / "landscape.db"
        db_url = f"sqlite:///{db_file}"
        db = LandscapeDB.from_url(db_url)

        try:
            # Create payload store
            payload_dir = tmp_path / "payloads"
            store = FilesystemPayloadStore(payload_dir)
            content_hash = store.store(b"test payload data")

            # Create old run (100 days ago)
            old_date = datetime.now(UTC) - timedelta(days=100)
            with db.connection() as conn:
                conn.execute(
                    insert(runs_table).values(
                        run_id="old-run-123",
                        status="completed",
                        started_at=old_date,
                        completed_at=old_date,
                        config_hash="abc123",
                        settings_json="{}",
                        canonical_version="1.0.0",
                    )
                )
                conn.execute(
                    insert(nodes_table).values(
                        node_id="source-node-purge",
                        run_id="old-run-123",
                        plugin_name="csv",
                        node_type="source",
                        plugin_version="1.0.0",
                        determinism="deterministic",
                        config_hash="def456",
                        config_json="{}",
                        registered_at=old_date,
                    )
                )
                conn.execute(
                    insert(rows_table).values(
                        run_id="old-run-123",
                        row_id="row-1",
                        source_node_id="source-node-purge",
                        row_index=0,
                        source_data_hash="hash1",
                        source_data_ref=content_hash,
                        created_at=old_date,
                    )
                )

            # Verify payload exists
            assert store.exists(content_hash)

            result = runner.invoke(
                app,
                [
                    "purge",
                    "--yes",
                    "--retention-days",
                    "90",
                    "--database",
                    str(db_file),
                    "--payload-dir",
                    str(payload_dir),
                ],
            )

            assert result.exit_code == 0
            assert "deleted" in result.stdout.lower() or "1" in result.stdout
            # Verify payload was deleted
            assert not store.exists(content_hash)
        finally:
            db.close()


class TestResumeCommand:
    """Tests for resume CLI command."""

    def test_resume_help(self) -> None:
        """resume --help shows usage."""
        from elspeth.cli import app

        result = runner.invoke(app, ["resume", "--help"])

        assert result.exit_code == 0
        assert "run" in result.stdout.lower()

    def test_resume_nonexistent_run(self, tmp_path: Path) -> None:
        """resume fails gracefully for nonexistent run."""
        from elspeth.cli import app

        db_file = tmp_path / "test.db"

        result = runner.invoke(
            app,
            [
                "resume",
                "nonexistent-run-id",
                "--database",
                str(db_file),
            ],
        )

        assert result.exit_code != 0
        output = result.output.lower()
        assert "not found" in output or "error" in output

    def test_resume_completed_run(self, tmp_path: Path) -> None:
        """resume fails for already-completed run."""
        from datetime import UTC, datetime

        from sqlalchemy import insert

        from elspeth.cli import app
        from elspeth.core.landscape import LandscapeDB
        from elspeth.core.landscape.schema import runs_table

        # Set up database with a completed run
        db_file = tmp_path / "test.db"
        db_url = f"sqlite:///{db_file}"
        db = LandscapeDB.from_url(db_url)

        now = datetime.now(UTC)
        with db.connection() as conn:
            conn.execute(
                insert(runs_table).values(
                    run_id="completed-run-001",
                    status="completed",
                    started_at=now,
                    completed_at=now,
                    config_hash="abc123",
                    settings_json="{}",
                    canonical_version="1.0.0",
                )
            )
        db.close()

        result = runner.invoke(
            app,
            [
                "resume",
                "completed-run-001",
                "--database",
                str(db_file),
            ],
        )

        assert result.exit_code != 0
        output = result.output.lower()
        assert "completed" in output or "cannot resume" in output

    def test_resume_running_run(self, tmp_path: Path) -> None:
        """resume fails for still-running run."""
        from datetime import UTC, datetime

        from sqlalchemy import insert

        from elspeth.cli import app
        from elspeth.core.landscape import LandscapeDB
        from elspeth.core.landscape.schema import runs_table

        # Set up database with a running run
        db_file = tmp_path / "test.db"
        db_url = f"sqlite:///{db_file}"
        db = LandscapeDB.from_url(db_url)

        now = datetime.now(UTC)
        with db.connection() as conn:
            conn.execute(
                insert(runs_table).values(
                    run_id="running-run-001",
                    status="running",
                    started_at=now,
                    completed_at=None,
                    config_hash="abc123",
                    settings_json="{}",
                    canonical_version="1.0.0",
                )
            )
        db.close()

        result = runner.invoke(
            app,
            [
                "resume",
                "running-run-001",
                "--database",
                str(db_file),
            ],
        )

        assert result.exit_code != 0
        output = result.output.lower()
        assert "in progress" in output or "running" in output

    def test_resume_failed_run_without_checkpoint(self, tmp_path: Path) -> None:
        """resume fails for failed run that has no checkpoint."""
        from datetime import UTC, datetime

        from sqlalchemy import insert

        from elspeth.cli import app
        from elspeth.core.landscape import LandscapeDB
        from elspeth.core.landscape.schema import runs_table

        # Set up database with a failed run (no checkpoint)
        db_file = tmp_path / "test.db"
        db_url = f"sqlite:///{db_file}"
        db = LandscapeDB.from_url(db_url)

        now = datetime.now(UTC)
        with db.connection() as conn:
            conn.execute(
                insert(runs_table).values(
                    run_id="failed-no-checkpoint-001",
                    status="failed",
                    started_at=now,
                    completed_at=now,
                    config_hash="abc123",
                    settings_json="{}",
                    canonical_version="1.0.0",
                )
            )
        db.close()

        result = runner.invoke(
            app,
            [
                "resume",
                "failed-no-checkpoint-001",
                "--database",
                str(db_file),
            ],
        )

        assert result.exit_code != 0
        output = result.output.lower()
        assert "checkpoint" in output or "no checkpoint" in output

    def test_resume_shows_resume_point_info(self, tmp_path: Path) -> None:
        """resume shows checkpoint info for resumable run."""
        from datetime import UTC, datetime

        from sqlalchemy import insert

        from elspeth.cli import app
        from elspeth.core.landscape import LandscapeDB
        from elspeth.core.landscape.schema import checkpoints_table, runs_table

        # Set up database with a failed run that has a checkpoint
        db_file = tmp_path / "test.db"
        db_url = f"sqlite:///{db_file}"
        db = LandscapeDB.from_url(db_url)

        now = datetime.now(UTC)
        with db.connection() as conn:
            conn.execute(
                insert(runs_table).values(
                    run_id="failed-with-checkpoint-001",
                    status="failed",
                    started_at=now,
                    completed_at=now,
                    config_hash="abc123",
                    settings_json="{}",
                    canonical_version="1.0.0",
                )
            )
            conn.execute(
                insert(checkpoints_table).values(
                    checkpoint_id="cp-test123",
                    run_id="failed-with-checkpoint-001",
                    token_id="token-abc",
                    node_id="transform-xyz",
                    sequence_number=42,
                    created_at=now,
                    aggregation_state_json=None,
                )
            )
        db.close()

        result = runner.invoke(
            app,
            [
                "resume",
                "failed-with-checkpoint-001",
                "--database",
                str(db_file),
            ],
        )

        # Should succeed and show resume point info
        assert result.exit_code == 0
        # Should show checkpoint info
        assert "token" in result.stdout.lower() or "node" in result.stdout.lower()
        assert "42" in result.stdout or "sequence" in result.stdout.lower()
