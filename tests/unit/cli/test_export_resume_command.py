# tests/unit/cli/test_export_resume_command.py
"""CLI production driver for resuming a finalized run's unfinished audit export.

elspeth-8fd1f415b9: a crash or transient target failure after run finalization
leaves the run's export PENDING/FAILED and its durable sink effect
PREPARED/IN_FLIGHT. ``elspeth export-resume`` is the supported operator path
that reconciles the unfinished effect for completed runs — the checkpoint
``resume`` command deliberately refuses completed runs.
"""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path
from unittest.mock import patch

from typer.testing import CliRunner

from elspeth.cli import app

runner = CliRunner()

_COMPLETED_AT = datetime(2026, 7, 16, 7, 8, 9, 123456, tzinfo=UTC)


def _write_settings(tmp_path: Path, db_path: Path, *, export_enabled: bool = True) -> Path:
    export_block = """
  export:
    enabled: true
    sink: output
    format: json
    signing_mode: unsigned
    signer_key_id: UNSIGNED
    total_record_limit: 10000
    total_byte_limit: 10485760
    chunk_limit: 100
    per_chunk_record_limit: 100
    per_chunk_byte_limit: 1048576
    spool_root: .elspeth/audit-export-spool/cli-test
    content_store:
      content_store_id: audit-store-v1
      namespace: audit/export
      root: .elspeth/audit-export-content-store/cli-test
      policy_version: v1
      retention_days: 30
      durability: fsync
"""
    settings_content = f"""
sources:
  primary:
    plugin: csv
    on_success: default
    options:
      path: input.csv
      on_validation_failure: discard
      schema: {{mode: observed}}
sinks:
  default:
    plugin: json
    on_write_failure: discard
    options:
      path: output.json
      schema: {{mode: observed}}
  output:
    plugin: json
    on_write_failure: discard
    options:
      path: audit-export.json
      schema: {{mode: observed}}
landscape:
  url: "sqlite:///{db_path}"{export_block if export_enabled else ""}
"""
    settings_file = tmp_path / "settings.yaml"
    settings_file.write_text(settings_content)
    return settings_file


def _make_landscape_db_with_run(
    db_path: Path,
    *,
    run_status: str = "completed",
    export_status: str | None = "failed",
    export_error: str | None = "publication response lost",
) -> None:
    from elspeth.core.landscape.database import LandscapeDB
    from elspeth.core.landscape.schema import runs_table

    db = LandscapeDB(f"sqlite:///{db_path}")
    try:
        with db.engine.begin() as connection:
            connection.execute(
                runs_table.insert().values(
                    run_id="run-export",
                    started_at=_COMPLETED_AT,
                    completed_at=_COMPLETED_AT if run_status != "running" else None,
                    config_hash="0" * 64,
                    settings_json="{}",
                    canonical_version="v1",
                    status=run_status,
                    export_status=export_status,
                    export_error=export_error,
                    openrouter_catalog_sha256="1" * 64,
                    openrouter_catalog_source="bundled",
                )
            )
    finally:
        db.close()


class TestExportResumeCommand:
    def test_dry_run_reports_eligible_failed_export(self, tmp_path: Path) -> None:
        db_path = tmp_path / "landscape.db"
        _make_landscape_db_with_run(db_path)
        settings_file = _write_settings(tmp_path, db_path)

        result = runner.invoke(app, ["export-resume", "run-export", "-s", str(settings_file)])

        assert result.exit_code == 0, f"Expected exit 0, got {result.exit_code}. Output: {result.output}"
        assert "Export status: failed" in result.output
        assert "can be resumed" in result.output
        assert "--execute" in result.output

    def test_dry_run_refuses_completed_export(self, tmp_path: Path) -> None:
        db_path = tmp_path / "landscape.db"
        _make_landscape_db_with_run(db_path, export_status="completed", export_error=None)
        settings_file = _write_settings(tmp_path, db_path)

        result = runner.invoke(app, ["export-resume", "run-export", "-s", str(settings_file)])

        assert result.exit_code == 1, f"Expected exit 1, got {result.exit_code}. Output: {result.output}"
        assert "already completed" in result.output

    def test_dry_run_refuses_non_terminal_run(self, tmp_path: Path) -> None:
        db_path = tmp_path / "landscape.db"
        _make_landscape_db_with_run(db_path, run_status="running", export_status=None, export_error=None)
        settings_file = _write_settings(tmp_path, db_path)

        result = runner.invoke(app, ["export-resume", "run-export", "-s", str(settings_file)])

        assert result.exit_code == 1, f"Expected exit 1, got {result.exit_code}. Output: {result.output}"
        assert "not export-terminal" in result.output

    def test_refuses_when_export_disabled(self, tmp_path: Path) -> None:
        db_path = tmp_path / "landscape.db"
        _make_landscape_db_with_run(db_path)
        settings_file = _write_settings(tmp_path, db_path, export_enabled=False)

        result = runner.invoke(app, ["export-resume", "run-export", "-s", str(settings_file)])

        assert result.exit_code == 1, f"Expected exit 1, got {result.exit_code}. Output: {result.output}"
        assert "not enabled" in result.output

    def test_refuses_missing_run(self, tmp_path: Path) -> None:
        db_path = tmp_path / "landscape.db"
        _make_landscape_db_with_run(db_path)
        settings_file = _write_settings(tmp_path, db_path)

        result = runner.invoke(app, ["export-resume", "run-other", "-s", str(settings_file)])

        assert result.exit_code == 1, f"Expected exit 1, got {result.exit_code}. Output: {result.output}"
        assert "not found" in result.output

    def test_execute_invokes_engine_resume_driver(self, tmp_path: Path, monkeypatch) -> None:
        monkeypatch.chdir(tmp_path)
        db_path = tmp_path / "landscape.db"
        _make_landscape_db_with_run(db_path)
        settings_file = _write_settings(tmp_path, db_path)
        payload_dir = tmp_path / ".elspeth" / "payloads"
        payload_dir.mkdir(parents=True)
        payload_dir.chmod(0o700)

        with patch("elspeth.engine.orchestrator.export.resume_audit_export") as resume:
            result = runner.invoke(
                app,
                ["export-resume", "run-export", "-s", str(settings_file), "--execute"],
            )

        assert result.exit_code == 0, f"Expected exit 0, got {result.exit_code}. Output: {result.output}"
        assert "resumed successfully" in result.output
        resume.assert_called_once()
        call = resume.call_args
        assert call.args[1] == "run-export"
        assert call.kwargs["worker_id"].startswith("worker:run-export:")
        assert call.kwargs["audit_export_content_store"].content_store_id == "audit-store-v1"

    def test_execute_reports_export_failure(self, tmp_path: Path, monkeypatch) -> None:
        monkeypatch.chdir(tmp_path)
        db_path = tmp_path / "landscape.db"
        _make_landscape_db_with_run(db_path)
        settings_file = _write_settings(tmp_path, db_path)
        payload_dir = tmp_path / ".elspeth" / "payloads"
        payload_dir.mkdir(parents=True)
        payload_dir.chmod(0o700)

        with patch(
            "elspeth.engine.orchestrator.export.resume_audit_export",
            side_effect=RuntimeError("target still unreachable"),
        ):
            result = runner.invoke(
                app,
                ["export-resume", "run-export", "-s", str(settings_file), "--execute"],
            )

        assert result.exit_code == 1, f"Expected exit 1, got {result.exit_code}. Output: {result.output}"
        assert "target still unreachable" in result.output
