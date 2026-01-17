"""End-to-end tests for run command with row_plugins (transforms and gates).

These tests verify that data actually flows through transforms and gates
correctly when run through the full CLI pipeline.
"""

from pathlib import Path

import pytest
import yaml
from typer.testing import CliRunner

runner = CliRunner()

# Dynamic schema config for tests - PathConfig now requires schema
DYNAMIC_SCHEMA = {"fields": "dynamic"}


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
    def settings_with_passthrough(
        self, tmp_path: Path, sample_csv: Path, output_csv: Path
    ) -> Path:
        """Config with passthrough transform."""
        config = {
            "datasource": {
                "plugin": "csv",
                "options": {
                    "path": str(sample_csv),
                    "on_validation_failure": "discard",
                    "schema": DYNAMIC_SCHEMA,
                },
            },
            "row_plugins": [
                {
                    "plugin": "passthrough",
                    "type": "transform",
                    "options": {"schema": DYNAMIC_SCHEMA},
                }
            ],
            "sinks": {
                "output": {
                    "plugin": "csv",
                    "options": {"path": str(output_csv), "schema": DYNAMIC_SCHEMA},
                }
            },
            "output_sink": "output",
            "landscape": {"url": f"sqlite:///{tmp_path}/audit.db"},
        }
        settings_file = tmp_path / "settings.yaml"
        settings_file.write_text(yaml.dump(config))
        return settings_file

    @pytest.fixture
    def settings_with_field_mapper(
        self, tmp_path: Path, sample_csv: Path, output_csv: Path
    ) -> Path:
        """Config with field_mapper transform that renames columns."""
        config = {
            "datasource": {
                "plugin": "csv",
                "options": {
                    "path": str(sample_csv),
                    "on_validation_failure": "discard",
                    "schema": DYNAMIC_SCHEMA,
                },
            },
            "row_plugins": [
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
                    "options": {"path": str(output_csv), "schema": DYNAMIC_SCHEMA},
                }
            },
            "output_sink": "output",
            "landscape": {"url": f"sqlite:///{tmp_path}/audit.db"},
        }
        settings_file = tmp_path / "settings.yaml"
        settings_file.write_text(yaml.dump(config))
        return settings_file

    @pytest.fixture
    def settings_with_chained_transforms(
        self, tmp_path: Path, sample_csv: Path, output_csv: Path
    ) -> Path:
        """Config with multiple transforms chained together."""
        config = {
            "datasource": {
                "plugin": "csv",
                "options": {
                    "path": str(sample_csv),
                    "on_validation_failure": "discard",
                    "schema": DYNAMIC_SCHEMA,
                },
            },
            "row_plugins": [
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
                    "options": {"path": str(output_csv), "schema": DYNAMIC_SCHEMA},
                }
            },
            "output_sink": "output",
            "landscape": {"url": f"sqlite:///{tmp_path}/audit.db"},
        }
        settings_file = tmp_path / "settings.yaml"
        settings_file.write_text(yaml.dump(config))
        return settings_file

    def test_run_with_passthrough_preserves_data(
        self, settings_with_passthrough: Path, output_csv: Path
    ) -> None:
        """Passthrough transform preserves all input data."""
        from elspeth.cli import app

        result = runner.invoke(app, ["run", "-s", str(settings_with_passthrough), "-x"])
        assert result.exit_code == 0, f"Failed with: {result.output}"

        output_content = output_csv.read_text()
        assert "alice" in output_content
        assert "bob" in output_content
        assert "carol" in output_content
        assert "75" in output_content
        assert "45" in output_content
        assert "90" in output_content

    def test_run_with_field_mapper_renames_columns(
        self, settings_with_field_mapper: Path, output_csv: Path
    ) -> None:
        """Field mapper transform renames columns correctly."""
        from elspeth.cli import app

        result = runner.invoke(
            app, ["run", "-s", str(settings_with_field_mapper), "-x"]
        )
        assert result.exit_code == 0, f"Failed with: {result.output}"

        output_content = output_csv.read_text()
        # Should have new column names
        assert "full_name" in output_content
        assert "test_score" in output_content
        # Data should still be present
        assert "alice" in output_content
        assert "75" in output_content

    def test_run_with_chained_transforms(
        self, settings_with_chained_transforms: Path, output_csv: Path
    ) -> None:
        """Multiple transforms in chain all execute in order."""
        from elspeth.cli import app

        result = runner.invoke(
            app, ["run", "-s", str(settings_with_chained_transforms), "-x"]
        )
        assert result.exit_code == 0, f"Failed with: {result.output}"

        output_content = output_csv.read_text()
        # Second transform renamed 'name' to 'person_name'
        assert "person_name" in output_content
        assert "alice" in output_content


# NOTE: TestRunWithGates and TestRunWithTransformAndGate classes removed in WP-02
# Gate plugins (threshold_gate, field_match_gate, filter_gate) were deleted.
# WP-09 will introduce engine-level gates with new tests.
