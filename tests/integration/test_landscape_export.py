# tests/integration/test_landscape_export.py
"""Integration tests for landscape audit export functionality.

Note: Uses JSON format for audit export because audit records are heterogeneous
(different record types have different fields). The CSV sink requires homogeneous
records with consistent field names. For CSV export, use separate files per
record type (not yet implemented).
"""

import json
from pathlib import Path

import pytest
import yaml
from typer.testing import CliRunner

runner = CliRunner()


class TestLandscapeExport:
    """End-to-end tests for landscape export to sink."""

    @pytest.fixture
    def export_settings_yaml(self, tmp_path: Path) -> Path:
        """Create settings file with export enabled using JSON sink.

        Uses JSON sink because audit records have heterogeneous schemas
        (run, node, row, token records have different fields).
        """
        # Create input CSV
        input_csv = tmp_path / "input.csv"
        input_csv.write_text("id,name,value\n1,Alice,100\n2,Bob,200\n")

        output_csv = tmp_path / "output.csv"
        audit_json = tmp_path / "audit_export.json"
        db_path = tmp_path / "audit.db"

        config = {
            "datasource": {"plugin": "csv", "options": {"path": str(input_csv)}},
            "sinks": {
                "output": {
                    "plugin": "csv",
                    "options": {"path": str(output_csv)},
                },
                "audit_export": {
                    "plugin": "json",
                    "options": {"path": str(audit_json)},
                },
            },
            "output_sink": "output",
            "landscape": {
                "url": f"sqlite:///{db_path}",
                "export": {
                    "enabled": True,
                    "sink": "audit_export",
                    "format": "json",  # Must use JSON for heterogeneous records
                    "sign": False,
                },
            },
        }

        settings_file = tmp_path / "settings.yaml"
        settings_file.write_text(yaml.dump(config))
        return settings_file

    def test_run_with_export_creates_audit_file(
        self, export_settings_yaml: Path, tmp_path: Path
    ) -> None:
        """Running pipeline with export enabled should create audit JSON."""
        from elspeth.cli import app

        # Run pipeline with --execute flag
        result = runner.invoke(
            app, ["run", "-s", str(export_settings_yaml), "--execute"]
        )
        assert result.exit_code == 0, f"CLI failed: {result.stdout}"
        assert "completed" in result.stdout.lower()

        # Check audit export was created
        audit_json = tmp_path / "audit_export.json"
        assert audit_json.exists(), "Audit export file was not created"

        # Read and verify content is valid JSON
        content = audit_json.read_text()
        records = json.loads(content)
        assert isinstance(records, list), "Export should be a JSON array"
        assert len(records) > 0, "Export should contain records"

        # Check for expected structure
        record_types = {r["record_type"] for r in records}
        assert "run" in record_types, "Missing run record"
        assert "row" in record_types, "Missing row records"

    def test_export_contains_all_record_types(
        self, export_settings_yaml: Path, tmp_path: Path
    ) -> None:
        """Export should contain run, node, row, token, and node_state records."""
        from elspeth.cli import app

        result = runner.invoke(
            app, ["run", "-s", str(export_settings_yaml), "--execute"]
        )
        assert result.exit_code == 0, f"CLI failed: {result.stdout}"

        # Read audit JSON
        audit_json = tmp_path / "audit_export.json"
        records = json.loads(audit_json.read_text())

        # Extract record types
        record_types = {r["record_type"] for r in records}

        # Must have core record types
        assert "run" in record_types, "Missing 'run' record type"
        assert "node" in record_types, "Missing 'node' record type"
        assert "row" in record_types, "Missing 'row' record type"
        assert "token" in record_types, "Missing 'token' record type"

    def test_export_run_record_has_required_fields(
        self, export_settings_yaml: Path, tmp_path: Path
    ) -> None:
        """Run record should have run_id, status, and timestamps."""
        from elspeth.cli import app

        result = runner.invoke(
            app, ["run", "-s", str(export_settings_yaml), "--execute"]
        )
        assert result.exit_code == 0, f"CLI failed: {result.stdout}"

        # Find run record
        audit_json = tmp_path / "audit_export.json"
        records = json.loads(audit_json.read_text())
        run_records = [r for r in records if r["record_type"] == "run"]

        assert len(run_records) == 1, "Should have exactly one run record"
        run_record = run_records[0]

        # Check required fields
        assert "run_id" in run_record, "Missing run_id"
        assert "status" in run_record, "Missing status"
        assert run_record["status"] == "completed", "Status should be completed"

    @pytest.fixture
    def export_disabled_settings(self, tmp_path: Path) -> Path:
        """Create settings file with export disabled."""
        input_csv = tmp_path / "input.csv"
        input_csv.write_text("id,name\n1,Test\n")

        output_csv = tmp_path / "output.csv"
        audit_json = tmp_path / "audit_export.json"
        db_path = tmp_path / "audit.db"

        config = {
            "datasource": {"plugin": "csv", "options": {"path": str(input_csv)}},
            "sinks": {
                "output": {
                    "plugin": "csv",
                    "options": {"path": str(output_csv)},
                },
                "audit_export": {
                    "plugin": "json",
                    "options": {"path": str(audit_json)},
                },
            },
            "output_sink": "output",
            "landscape": {
                "url": f"sqlite:///{db_path}",
                "export": {
                    "enabled": False,  # Export disabled
                    "sink": "audit_export",
                },
            },
        }

        settings_file = tmp_path / "settings.yaml"
        settings_file.write_text(yaml.dump(config))
        return settings_file

    def test_export_disabled_does_not_create_file(
        self, export_disabled_settings: Path, tmp_path: Path
    ) -> None:
        """When export is disabled, audit file should not be created."""
        from elspeth.cli import app

        result = runner.invoke(
            app, ["run", "-s", str(export_disabled_settings), "--execute"]
        )
        assert result.exit_code == 0, f"CLI failed: {result.stdout}"

        # Audit export should NOT be created
        audit_json = tmp_path / "audit_export.json"
        assert not audit_json.exists(), "Audit file should not exist when export disabled"
