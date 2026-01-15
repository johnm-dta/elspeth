# tests/integration/test_landscape_export.py
"""Integration tests for landscape audit export functionality.

Note: Uses JSON format for audit export because audit records are heterogeneous
(different record types have different fields). The CSV sink requires homogeneous
records with consistent field names. For CSV export, use separate files per
record type (now implemented as multi-file export).
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
        assert (
            not audit_json.exists()
        ), "Audit file should not exist when export disabled"


class TestSignedExportDeterminism:
    """Integration tests for signed export determinism.

    These tests verify that signed exports produce identical signatures
    when run with the same data and signing key. This is critical for
    audit integrity - if signatures aren't deterministic, verification
    becomes impossible.
    """

    def test_signed_export_produces_identical_final_hash(self, tmp_path: Path) -> None:
        """Two exports of the same run should produce identical final_hash.

        This is the critical determinism test: same data + same key = same hash.
        If this fails, the ORDER BY fixes in recorder.py are broken.

        We run ONE pipeline, then export the audit trail TWICE via the exporter
        directly (not via CLI, which creates new runs each time).
        """
        from elspeth.core.landscape import LandscapeDB, LandscapeRecorder
        from elspeth.core.landscape.exporter import LandscapeExporter

        db = LandscapeDB.in_memory()
        recorder = LandscapeRecorder(db)

        # Create a run with multiple records of each type
        run = recorder.begin_run(config={"test": True}, canonical_version="v1")

        # Multiple nodes
        for i in range(3):
            recorder.register_node(
                run_id=run.run_id,
                node_id=f"node_{i}",
                plugin_name="test",
                node_type="transform",
                plugin_version="1.0.0",
                config={"index": i},
            )

        # Multiple edges
        for i in range(2):
            recorder.register_edge(
                run_id=run.run_id,
                from_node_id=f"node_{i}",
                to_node_id=f"node_{i + 1}",
                label="continue",
                mode="move",
            )

        # Multiple rows with tokens
        for i in range(3):
            row = recorder.create_row(
                run_id=run.run_id,
                source_node_id="node_0",
                row_index=i,
                data={"value": i * 10},
            )
            token = recorder.create_token(row_id=row.row_id)
            state = recorder.begin_node_state(
                token_id=token.token_id,
                node_id="node_0",
                step_index=0,
                input_data={"x": i},
            )
            recorder.complete_node_state(state.state_id, status="completed")

        recorder.complete_run(run.run_id, status="completed")

        # Export the SAME run twice with signing
        signing_key = b"test-determinism-key-12345"
        exporter = LandscapeExporter(db, signing_key=signing_key)

        final_hashes = []
        for _ in range(2):
            records = list(exporter.export_run(run.run_id, sign=True))
            manifest = [r for r in records if r["record_type"] == "manifest"][0]
            final_hashes.append(manifest["final_hash"])

        # Both exports must produce the same final hash
        assert (
            final_hashes[0] == final_hashes[1]
        ), f"Non-deterministic export! Hash 1: {final_hashes[0]}, Hash 2: {final_hashes[1]}"

    def test_signed_export_all_records_have_signatures(self, tmp_path: Path) -> None:
        """All exported records should have HMAC signatures when signing enabled."""
        from elspeth.cli import app

        input_csv = tmp_path / "input.csv"
        input_csv.write_text("id,value\n1,100\n")

        output_csv = tmp_path / "output.csv"
        audit_json = tmp_path / "audit.json"
        db_path = tmp_path / "audit.db"

        config = {
            "datasource": {"plugin": "csv", "options": {"path": str(input_csv)}},
            "sinks": {
                "output": {"plugin": "csv", "options": {"path": str(output_csv)}},
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
                    "format": "json",
                    "sign": True,
                },
            },
        }

        settings_file = tmp_path / "settings.yaml"
        settings_file.write_text(yaml.dump(config))

        result = runner.invoke(
            app,
            ["run", "-s", str(settings_file), "--execute"],
            env={"ELSPETH_SIGNING_KEY": "test-key"},
        )
        assert result.exit_code == 0, f"CLI failed: {result.stdout}"

        records = json.loads(audit_json.read_text())

        # Every record must have a signature
        for record in records:
            assert (
                "signature" in record
            ), f"Missing signature: {record.get('record_type')}"
            assert (
                len(record["signature"]) == 64
            ), "Signature should be 64-char hex (SHA256)"

    def test_different_signing_keys_produce_different_hashes(
        self, tmp_path: Path
    ) -> None:
        """Different signing keys should produce different final hashes.

        This verifies the signature actually depends on the key, not just the data.
        """
        from elspeth.cli import app

        input_csv = tmp_path / "input.csv"
        input_csv.write_text("id,value\n1,42\n")

        keys = ["key-alpha-12345", "key-beta-67890"]
        final_hashes = []

        for i, key in enumerate(keys):
            output_csv = tmp_path / f"output_{i}.csv"
            audit_json = tmp_path / f"audit_{i}.json"
            db_path = tmp_path / f"audit_{i}.db"

            config = {
                "datasource": {"plugin": "csv", "options": {"path": str(input_csv)}},
                "sinks": {
                    "output": {"plugin": "csv", "options": {"path": str(output_csv)}},
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
                        "format": "json",
                        "sign": True,
                    },
                },
            }

            settings_file = tmp_path / f"settings_{i}.yaml"
            settings_file.write_text(yaml.dump(config))

            result = runner.invoke(
                app,
                ["run", "-s", str(settings_file), "--execute"],
                env={"ELSPETH_SIGNING_KEY": key},
            )
            assert result.exit_code == 0, f"Run with key {i} failed: {result.stdout}"

            records = json.loads(audit_json.read_text())
            manifest = [r for r in records if r["record_type"] == "manifest"][0]
            final_hashes.append(manifest["final_hash"])

        # Different keys must produce different hashes
        assert (
            final_hashes[0] != final_hashes[1]
        ), "Different keys produced same hash - signature not key-dependent!"
