"""Tests for ELSPETH CLI."""

from pathlib import Path

import pytest
from typer.testing import CliRunner

from elspeth.contracts import Determinism, NodeType, RunStatus
from elspeth.core.landscape.reproducibility import ReproducibilityGrade

from .conftest import verify_audit_trail

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
source:
  plugin: csv
  options:
    path: "{input_file}"
    on_validation_failure: discard
    schema:
      fields: dynamic

sinks:
  results:
    plugin: csv
    options:
      path: "{output_file}"
      schema:
        mode: strict
        fields:
          - "id: str"
          - "value: str"

transforms:
  - plugin: passthrough
    options:
      schema:
        fields: dynamic

default_sink: results

landscape:
  enabled: true
  backend: sqlite
  url: "sqlite:///{audit_db}"
""")

        result = runner.invoke(app, ["run", "-s", str(config_file), "--execute", "-v"])

        assert result.exit_code == 0, f"CLI failed: {result.stdout}"
        # Output should exist with data processed
        assert output_file.exists()

        # Verify audit trail integrity - 2 rows processed through passthrough transform
        verify_audit_trail(audit_db, expected_row_count=2)

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
source:
  plugin: csv
  options:
    path: "{input_file}"
    on_validation_failure: discard
    schema:
      fields: dynamic

sinks:
  results:
    plugin: csv
    options:
      path: "{output_file}"
      schema:
        mode: strict
        fields:
          - "id: str"
          - "new_name: str"

transforms:
  - plugin: field_mapper
    options:
      schema:
        fields: dynamic
      mapping:
        old_name: new_name

default_sink: results

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
        assert "new_name" in output_content, f"Field mapper should have renamed 'old_name' to 'new_name'. Output was: {output_content}"

        # Verify audit trail integrity - 2 rows processed through field_mapper transform
        verify_audit_trail(audit_db, expected_row_count=2)


class TestPurgeCommand:
    """Tests for purge CLI command."""

    def test_purge_help(self) -> None:
        """purge --help shows usage."""
        from elspeth.cli import app

        result = runner.invoke(app, ["purge", "--help"])

        assert result.exit_code == 0
        assert "retention" in result.stdout.lower() or "days" in result.stdout.lower()

    def test_purge_fails_on_missing_database(self, tmp_path: Path) -> None:
        """purge fails with clear error when database file doesn't exist.

        Prevents silent creation of empty DB on typoed --database paths.
        See: BUG-AUDIT-01 / P3-cli-purge-resume-silently-create-db
        """
        from elspeth.cli import app

        nonexistent_db = tmp_path / "does-not-exist.db"

        result = runner.invoke(
            app,
            ["purge", "--dry-run", "--database", str(nonexistent_db)],
        )

        assert result.exit_code == 1
        assert "database file not found" in result.output.lower()
        # Verify no file was created
        assert not nonexistent_db.exists(), "Should not create database file on missing path"

    def test_purge_dry_run(self, tmp_path: Path) -> None:
        """purge --dry-run shows what would be deleted or that nothing expired."""
        from elspeth.cli import app
        from elspeth.core.landscape import LandscapeDB

        # Pre-create database (purge requires existing DB, doesn't auto-create)
        db_file = tmp_path / "test.db"
        LandscapeDB.from_url(f"sqlite:///{db_file}")

        result = runner.invoke(
            app,
            [
                "purge",
                "--dry-run",
                "--database",
                str(db_file),
            ],
        )

        assert result.exit_code == 0
        # CLI outputs either "would delete N payload(s)" or "No payloads older than X days found"
        output_lower = result.stdout.lower()
        assert "would delete" in output_lower or "no payloads older than" in output_lower, (
            f"Expected purge dry-run message, got: {result.stdout}"
        )

    def test_purge_with_retention_override(self, tmp_path: Path) -> None:
        """purge --retention-days overrides default."""
        from elspeth.cli import app
        from elspeth.core.landscape import LandscapeDB

        # Pre-create database (purge requires existing DB, doesn't auto-create)
        db_file = tmp_path / "test.db"
        LandscapeDB.from_url(f"sqlite:///{db_file}")

        result = runner.invoke(
            app,
            [
                "purge",
                "--dry-run",
                "--retention-days",
                "30",
                "--database",
                str(db_file),
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
                    status=RunStatus.COMPLETED,
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
                    node_type=NodeType.SOURCE,
                    plugin_version="1.0.0",
                    determinism=Determinism.DETERMINISTIC,
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
        from elspeth.core.landscape import LandscapeDB

        # Pre-create database (purge requires existing DB, doesn't auto-create)
        db_file = tmp_path / "test.db"
        LandscapeDB.from_url(f"sqlite:///{db_file}")

        result = runner.invoke(
            app,
            [
                "purge",
                "--yes",
                "--database",
                str(db_file),
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
                        status=RunStatus.COMPLETED,
                        started_at=old_date,
                        completed_at=old_date,
                        config_hash="abc123",
                        settings_json="{}",
                        reproducibility_grade=ReproducibilityGrade.REPLAY_REPRODUCIBLE.value,
                        canonical_version="1.0.0",
                    )
                )
                conn.execute(
                    insert(nodes_table).values(
                        node_id="source-node-purge",
                        run_id="old-run-123",
                        plugin_name="csv",
                        node_type=NodeType.SOURCE,
                        plugin_version="1.0.0",
                        determinism=Determinism.DETERMINISTIC,
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

    def test_purge_uses_config_payload_store_settings(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        """purge uses payload_store settings from config when settings.yaml exists.

        Bug: P1-2026-01-20-cli-purge-ignores-payload-store-settings
        """
        import yaml

        from elspeth.cli import app

        # Change to temp directory so settings.yaml is found
        monkeypatch.chdir(tmp_path)

        # Create minimal settings.yaml with custom payload_store config
        custom_payload_path = tmp_path / "custom_payloads"
        custom_payload_path.mkdir()
        settings = {
            "source": {"plugin": "csv", "path": "test.csv"},
            "sinks": {"output": {"plugin": "csv", "path": "output.csv"}},
            "default_sink": "output",
            "landscape": {"url": f"sqlite:///{tmp_path / 'landscape.db'}"},
            "payload_store": {
                "backend": "filesystem",
                "base_path": str(custom_payload_path),
                "retention_days": 45,  # Custom retention
            },
        }
        with open("settings.yaml", "w") as f:
            yaml.dump(settings, f)

        # Run purge --dry-run (no --payload-dir, no --retention-days)
        result = runner.invoke(
            app,
            ["purge", "--dry-run"],
        )

        assert result.exit_code == 0
        # Should show it's using the configured payload directory
        assert "custom_payloads" in result.stdout or str(custom_payload_path) in result.stdout
        # Should show it's using the configured retention_days
        assert "45" in result.stdout or "retention" in result.stdout.lower()

    def test_purge_rejects_unsupported_backend(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        """purge fails fast when payload_store.backend is not 'filesystem'.

        Bug: P1-2026-01-20-cli-purge-ignores-payload-store-settings
        """
        import yaml

        from elspeth.cli import app

        # Change to temp directory so settings.yaml is found
        monkeypatch.chdir(tmp_path)

        # Create settings.yaml with unsupported backend
        settings = {
            "source": {"plugin": "csv", "path": "test.csv"},
            "sinks": {"output": {"plugin": "csv", "path": "output.csv"}},
            "default_sink": "output",
            "landscape": {"url": f"sqlite:///{tmp_path / 'landscape.db'}"},
            "payload_store": {
                "backend": "azure_blob",  # Unsupported backend
                "base_path": str(tmp_path / "payloads"),
            },
        }
        with open("settings.yaml", "w") as f:
            yaml.dump(settings, f)

        # Run purge --dry-run
        result = runner.invoke(
            app,
            ["purge", "--dry-run"],
        )

        # Should fail with clear error message
        assert result.exit_code == 1
        # Error is written to stderr (result.output includes both stdout and stderr)
        output = result.output.lower() if result.output else ""
        assert "azure_blob" in output or "not supported" in output


class TestResumeCommand:
    """Tests for resume CLI command."""

    def test_resume_help(self) -> None:
        """resume --help shows usage."""
        from elspeth.cli import app

        result = runner.invoke(app, ["resume", "--help"])

        assert result.exit_code == 0
        assert "run" in result.stdout.lower()

    def test_resume_fails_on_missing_database(self, tmp_path: Path) -> None:
        """resume fails with clear error when database file doesn't exist.

        Prevents silent creation of empty DB on typoed --database paths.
        See: BUG-AUDIT-01 / P3-cli-purge-resume-silently-create-db
        """
        from elspeth.cli import app

        nonexistent_db = tmp_path / "does-not-exist.db"

        # Create a minimal settings file (resume requires settings)
        settings_file = tmp_path / "settings.yaml"
        settings_file.write_text("""
source:
  plugin: csv
  options:
    path: dummy.csv
    schema:
      mode: free
sinks:
  output:
    plugin: csv
    options:
      path: output.csv
default_sink: output
landscape:
  url: sqlite:///./runs/audit.db
""")

        result = runner.invoke(
            app,
            [
                "resume",
                "some-run-id",
                "--database",
                str(nonexistent_db),
                "--settings",
                str(settings_file),
            ],
        )

        assert result.exit_code == 1
        assert "database file not found" in result.output.lower()
        # Verify no file was created
        assert not nonexistent_db.exists(), "Should not create database file on missing path"

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

        # Create minimal settings file for resume command
        settings_file = tmp_path / "settings.yaml"
        settings_file.write_text(f"""
landscape:
  url: "sqlite:///{tmp_path / "test.db"}"
source:
  plugin: csv
  options:
    path: dummy.csv
    on_validation_failure: quarantine
    schema:
      fields: dynamic
transforms:
  - plugin: passthrough
    options:
      schema:
        fields: dynamic
sinks:
  output:
    plugin: csv
    options:
      path: output.csv
      schema:
        mode: strict
        fields:
          - "data: str"
default_sink: output
""")

        # Set up database with a completed run
        db_file = tmp_path / "test.db"
        db_url = f"sqlite:///{db_file}"
        db = LandscapeDB.from_url(db_url)

        now = datetime.now(UTC)
        with db.connection() as conn:
            conn.execute(
                insert(runs_table).values(
                    run_id="completed-run-001",
                    status=RunStatus.COMPLETED,
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
                "--settings",
                str(settings_file),
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

        # Create minimal settings file for resume command
        settings_file = tmp_path / "settings.yaml"
        settings_file.write_text(f"""
landscape:
  url: "sqlite:///{tmp_path / "test.db"}"
source:
  plugin: csv
  options:
    path: dummy.csv
    on_validation_failure: quarantine
    schema:
      fields: dynamic
transforms:
  - plugin: passthrough
    options:
      schema:
        fields: dynamic
sinks:
  output:
    plugin: csv
    options:
      path: output.csv
      schema:
        mode: strict
        fields:
          - "data: str"
default_sink: output
""")

        # Set up database with a running run
        db_file = tmp_path / "test.db"
        db_url = f"sqlite:///{db_file}"
        db = LandscapeDB.from_url(db_url)

        now = datetime.now(UTC)
        with db.connection() as conn:
            conn.execute(
                insert(runs_table).values(
                    run_id="running-run-001",
                    status=RunStatus.RUNNING,
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
                "--settings",
                str(settings_file),
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
                    status=RunStatus.FAILED,
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
        from elspeth.core.landscape.schema import (
            nodes_table,
            rows_table,
            runs_table,
            tokens_table,
        )

        # Create minimal settings file for resume command
        settings_file = tmp_path / "settings.yaml"
        settings_file.write_text(f"""
landscape:
  url: "sqlite:///{tmp_path / "test.db"}"
source:
  plugin: csv
  options:
    path: dummy.csv
    on_validation_failure: quarantine
    schema:
      fields: dynamic
transforms:
  - plugin: passthrough
    options:
      schema:
        fields: dynamic
sinks:
  output:
    plugin: csv
    options:
      path: output.csv
      schema:
        mode: strict
        fields:
          - "data: str"
default_sink: output
""")

        # Build graph to get actual node IDs (use ORIGINAL source for validation)
        from elspeth.core.config import load_settings
        from elspeth.core.dag import ExecutionGraph

        settings_config = load_settings(settings_file)

        # Build graph using original source (not NullSource) for topology validation
        from elspeth.cli_helpers import instantiate_plugins_from_config

        plugins = instantiate_plugins_from_config(settings_config)

        graph = ExecutionGraph.from_plugin_instances(
            source=plugins["source"],  # Use original source, NOT NullSource
            transforms=plugins["transforms"],
            sinks=plugins["sinks"],
            aggregations=plugins["aggregations"],
            gates=list(settings_config.gates),
            default_sink=settings_config.default_sink,
            coalesce_settings=list(settings_config.coalesce) if settings_config.coalesce else None,
        )
        graph.validate()

        # Get actual node IDs from graph
        all_nodes = list(graph.get_nx_graph().nodes())
        source_node_id = next(n for n in all_nodes if graph.get_node_info(n).node_type == "source")
        # Use sink node for checkpoint (transforms may not exist in simple pipeline)
        sink_node_id = next(n for n in all_nodes if graph.get_node_info(n).node_type == "sink")
        checkpoint_node_id = sink_node_id

        # Set up database with a failed run that has a checkpoint
        db_file = tmp_path / "test.db"
        db_url = f"sqlite:///{db_file}"
        db = LandscapeDB.from_url(db_url)

        run_id = "failed-with-checkpoint-001"
        now = datetime.now(UTC)
        with db.connection() as conn:
            # Insert run
            conn.execute(
                insert(runs_table).values(
                    run_id=run_id,
                    status=RunStatus.FAILED,
                    started_at=now,
                    completed_at=now,
                    config_hash="abc123",
                    settings_json="{}",
                    canonical_version="1.0.0",
                )
            )
            # Insert nodes (source and sink) for FK integrity using actual node IDs
            source_info = graph.get_node_info(source_node_id)
            conn.execute(
                insert(nodes_table).values(
                    node_id=source_node_id,
                    run_id=run_id,
                    plugin_name=source_info.plugin_name,
                    node_type=source_info.node_type,
                    plugin_version="1.0.0",
                    determinism=Determinism.DETERMINISTIC,
                    config_hash="hash1",
                    config_json="{}",
                    registered_at=now,
                )
            )
            sink_info = graph.get_node_info(sink_node_id)
            conn.execute(
                insert(nodes_table).values(
                    node_id=sink_node_id,
                    run_id=run_id,
                    plugin_name=sink_info.plugin_name,
                    node_type=sink_info.node_type,
                    plugin_version="1.0.0",
                    determinism=Determinism.DETERMINISTIC,
                    config_hash="hash2",
                    config_json="{}",
                    registered_at=now,
                )
            )
            # Insert row (references actual source node)
            conn.execute(
                insert(rows_table).values(
                    row_id="row-001",
                    run_id=run_id,
                    source_node_id=source_node_id,
                    row_index=0,
                    source_data_hash="hash123",
                    created_at=now,
                )
            )
            # Insert token (references row)
            conn.execute(
                insert(tokens_table).values(
                    token_id="token-abc",
                    row_id="row-001",
                    created_at=now,
                )
            )
            conn.commit()

        # Create checkpoint with topology validation
        from elspeth.core.checkpoint.manager import CheckpointManager

        checkpoint_manager = CheckpointManager(db)
        checkpoint_manager.create_checkpoint(
            run_id=run_id,
            token_id="token-abc",
            node_id=checkpoint_node_id,
            sequence_number=42,
            graph=graph,
        )

        db.close()

        result = runner.invoke(
            app,
            [
                "resume",
                "failed-with-checkpoint-001",
                "--settings",
                str(settings_file),
            ],
        )

        # Should succeed and show resume point info
        assert result.exit_code == 0
        # Should show checkpoint info
        assert "token" in result.stdout.lower() or "node" in result.stdout.lower()
        assert "42" in result.stdout or "sequence" in result.stdout.lower()


class TestTildeExpansion:
    """Tests that CLI path options expand ~ to home directory.

    Regression tests for:
    - docs/bugs/closed/P2-2026-01-20-cli-paths-no-tilde-expansion.md
    """

    def test_run_expands_tilde_in_settings_path(self, tmp_path: Path) -> None:
        """run command expands ~ in --settings path.

        Creates a file in a temp dir, then constructs a path using ~ that
        resolves to the same location, verifying expansion works.
        """
        from unittest.mock import patch

        from elspeth.cli import app

        # Create a settings file
        settings_content = """
source:
  plugin: csv
  options:
    path: input.csv
sinks:
  default:
    plugin: json
    options:
      path: output.json
default_sink: default
"""
        settings_file = tmp_path / "settings.yaml"
        settings_file.write_text(settings_content)

        # Mock expanduser to return our temp path
        # This simulates ~ expanding to our test directory
        def mock_expanduser(self: Path) -> Path:
            if str(self).startswith("~"):
                return tmp_path / str(self)[2:]  # Replace ~/x with tmp_path/x
            return self

        with patch.object(Path, "expanduser", mock_expanduser):
            result = runner.invoke(app, ["run", "-s", "~/settings.yaml"])

        # Should find the file (even if validation fails for other reasons)
        # The key is that it doesn't say "file not found" for the tilde path
        assert "Settings file not found: ~/settings.yaml" not in result.output

    def test_validate_expands_tilde_in_settings_path(self, tmp_path: Path) -> None:
        """validate command expands ~ in --settings path."""
        from unittest.mock import patch

        from elspeth.cli import app

        settings_content = """
source:
  plugin: csv
  options:
    path: input.csv
sinks:
  default:
    plugin: json
    options:
      path: output.json
default_sink: default
"""
        settings_file = tmp_path / "settings.yaml"
        settings_file.write_text(settings_content)

        def mock_expanduser(self: Path) -> Path:
            if str(self).startswith("~"):
                return tmp_path / str(self)[2:]
            return self

        with patch.object(Path, "expanduser", mock_expanduser):
            result = runner.invoke(app, ["validate", "-s", "~/settings.yaml"])

        # Should find the file
        assert "Settings file not found: ~/settings.yaml" not in result.output
