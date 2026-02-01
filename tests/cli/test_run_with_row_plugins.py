"""End-to-end tests for run command with row_plugins (transforms and gates).

These tests verify that data actually flows through transforms and gates
correctly when run through the full CLI pipeline.
"""

import csv
from pathlib import Path

import pytest
import yaml
from typer.testing import CliRunner

from .conftest import verify_audit_trail

runner = CliRunner()

# Dynamic schema config for tests - PathConfig now requires schema
DYNAMIC_SCHEMA = {"fields": "dynamic"}

# Strict schemas for CSV sinks (CSVSink requires fixed columns)
# Passthrough sink - same as input: id, name, score
PASSTHROUGH_SINK_SCHEMA = {"mode": "strict", "fields": ["id: str", "name: str", "score: str"]}
# Field mapper sink - renames name→full_name, score→test_score
FIELD_MAPPER_SINK_SCHEMA = {"mode": "strict", "fields": ["id: str", "full_name: str", "test_score: str"]}
# Chained transforms sink - renames name→person_name
CHAINED_SINK_SCHEMA = {"mode": "strict", "fields": ["id: str", "person_name: str", "score: str"]}


class TestRunWithTransforms:
    """Test run command with transform plugins."""

    @pytest.fixture
    def sample_csv(self, tmp_path: Path) -> Path:
        """Create sample CSV input file."""
        csv_file = tmp_path / "input.csv"
        csv_file.write_text("id,name,score\n1,alice,75\n2,bob,45\n3,carol,90\n")
        return csv_file

    @pytest.fixture
    def output_csv(self, tmp_path: Path) -> Path:
        """Output CSV path."""
        return tmp_path / "output.csv"

    @pytest.fixture
    def settings_with_passthrough(self, tmp_path: Path, sample_csv: Path, output_csv: Path) -> Path:
        """Config with passthrough transform."""
        config = {
            "source": {
                "plugin": "csv",
                "options": {
                    "path": str(sample_csv),
                    "on_validation_failure": "discard",
                    "schema": DYNAMIC_SCHEMA,
                },
            },
            "transforms": [
                {
                    "plugin": "passthrough",
                    "type": "transform",
                    "options": {"schema": DYNAMIC_SCHEMA},
                }
            ],
            "sinks": {
                "output": {
                    "plugin": "csv",
                    "options": {"path": str(output_csv), "schema": PASSTHROUGH_SINK_SCHEMA},
                }
            },
            "default_sink": "output",
            "landscape": {"url": f"sqlite:///{tmp_path}/audit.db"},
        }
        settings_file = tmp_path / "settings.yaml"
        settings_file.write_text(yaml.dump(config))
        return settings_file

    @pytest.fixture
    def settings_with_field_mapper(self, tmp_path: Path, sample_csv: Path, output_csv: Path) -> Path:
        """Config with field_mapper transform that renames columns."""
        config = {
            "source": {
                "plugin": "csv",
                "options": {
                    "path": str(sample_csv),
                    "on_validation_failure": "discard",
                    "schema": DYNAMIC_SCHEMA,
                },
            },
            "transforms": [
                {
                    "plugin": "field_mapper",
                    "type": "transform",
                    "options": {
                        "schema": DYNAMIC_SCHEMA,
                        "mapping": {"name": "full_name", "score": "test_score"},
                    },
                }
            ],
            "sinks": {
                "output": {
                    "plugin": "csv",
                    "options": {"path": str(output_csv), "schema": FIELD_MAPPER_SINK_SCHEMA},
                }
            },
            "default_sink": "output",
            "landscape": {"url": f"sqlite:///{tmp_path}/audit.db"},
        }
        settings_file = tmp_path / "settings.yaml"
        settings_file.write_text(yaml.dump(config))
        return settings_file

    @pytest.fixture
    def settings_with_chained_transforms(self, tmp_path: Path, sample_csv: Path, output_csv: Path) -> Path:
        """Config with multiple transforms chained together."""
        config = {
            "source": {
                "plugin": "csv",
                "options": {
                    "path": str(sample_csv),
                    "on_validation_failure": "discard",
                    "schema": DYNAMIC_SCHEMA,
                },
            },
            "transforms": [
                {
                    "plugin": "passthrough",
                    "type": "transform",
                    "options": {"schema": DYNAMIC_SCHEMA},
                },
                {
                    "plugin": "field_mapper",
                    "type": "transform",
                    "options": {
                        "schema": DYNAMIC_SCHEMA,
                        "mapping": {"name": "person_name"},
                    },
                },
            ],
            "sinks": {
                "output": {
                    "plugin": "csv",
                    "options": {"path": str(output_csv), "schema": CHAINED_SINK_SCHEMA},
                }
            },
            "default_sink": "output",
            "landscape": {"url": f"sqlite:///{tmp_path}/audit.db"},
        }
        settings_file = tmp_path / "settings.yaml"
        settings_file.write_text(yaml.dump(config))
        return settings_file

    def test_run_with_passthrough_preserves_data(self, settings_with_passthrough: Path, output_csv: Path, tmp_path: Path) -> None:
        """Passthrough transform preserves all input data."""
        from elspeth.cli import app

        result = runner.invoke(app, ["run", "-s", str(settings_with_passthrough), "-x"])
        assert result.exit_code == 0, f"Failed with: {result.output}"

        # Parse CSV for structured validation instead of substring checks
        with output_csv.open(newline="") as f:
            rows = list(csv.DictReader(f))

        # Verify exact row count and column names
        assert len(rows) == 3, f"Expected 3 rows, got {len(rows)}"
        expected_columns = {"id", "name", "score"}
        actual_columns = set(rows[0].keys())
        assert actual_columns == expected_columns, f"Expected columns {expected_columns}, got {actual_columns}"

        # Verify all expected data is present with exact values
        names = {row["name"] for row in rows}
        assert names == {"alice", "bob", "carol"}, f"Expected names {{alice, bob, carol}}, got {names}"
        scores = {row["score"] for row in rows}
        assert scores == {"75", "45", "90"}, f"Expected scores {{75, 45, 90}}, got {scores}"

        # Verify audit trail integrity
        audit_db = tmp_path / "audit.db"
        verify_audit_trail(audit_db, expected_row_count=3)

    def test_run_with_field_mapper_renames_columns(self, settings_with_field_mapper: Path, output_csv: Path, tmp_path: Path) -> None:
        """Field mapper transform renames columns correctly."""
        from elspeth.cli import app

        result = runner.invoke(app, ["run", "-s", str(settings_with_field_mapper), "-x"])
        assert result.exit_code == 0, f"Failed with: {result.output}"

        # Parse CSV for structured validation
        with output_csv.open(newline="") as f:
            rows = list(csv.DictReader(f))

        # Verify row count and NEW column names (not old ones)
        assert len(rows) == 3, f"Expected 3 rows, got {len(rows)}"
        actual_columns = set(rows[0].keys())

        # Must have the renamed columns
        assert "full_name" in actual_columns, f"Missing 'full_name' column in {actual_columns}"
        assert "test_score" in actual_columns, f"Missing 'test_score' column in {actual_columns}"

        # Must NOT have the old column names (verifies rename, not copy)
        assert "name" not in actual_columns, "Old 'name' column should be renamed, not present"
        assert "score" not in actual_columns, "Old 'score' column should be renamed, not present"

        # Verify data integrity under new column names
        names = {row["full_name"] for row in rows}
        assert names == {"alice", "bob", "carol"}, f"Data not preserved under 'full_name': {names}"
        scores = {row["test_score"] for row in rows}
        assert scores == {"75", "45", "90"}, f"Data not preserved under 'test_score': {scores}"

        # Verify audit trail integrity
        audit_db = tmp_path / "audit.db"
        verify_audit_trail(audit_db, expected_row_count=3)

    def test_run_with_chained_transforms(self, settings_with_chained_transforms: Path, output_csv: Path, tmp_path: Path) -> None:
        """Multiple transforms in chain all execute in order."""
        from elspeth.cli import app

        result = runner.invoke(app, ["run", "-s", str(settings_with_chained_transforms), "-x"])
        assert result.exit_code == 0, f"Failed with: {result.output}"

        # Parse CSV for structured validation
        with output_csv.open(newline="") as f:
            rows = list(csv.DictReader(f))

        # Verify row count
        assert len(rows) == 3, f"Expected 3 rows, got {len(rows)}"
        actual_columns = set(rows[0].keys())

        # Second transform renamed 'name' to 'person_name'
        assert "person_name" in actual_columns, f"Expected 'person_name' column after chained transform, got {actual_columns}"

        # Original 'name' column should be gone (renamed, not copied)
        assert "name" not in actual_columns, "Old 'name' column should be renamed to 'person_name', not present"

        # Verify data integrity
        names = {row["person_name"] for row in rows}
        assert names == {"alice", "bob", "carol"}, f"Data not preserved under 'person_name': {names}"

        # Verify audit trail integrity
        audit_db = tmp_path / "audit.db"
        verify_audit_trail(audit_db, expected_row_count=3)


# NOTE: TestRunWithGates and TestRunWithTransformAndGate classes removed in WP-02
# Gate plugins (threshold_gate, field_match_gate, filter_gate) were deleted.
# WP-09 will introduce engine-level gates with new tests.
