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
                "options": {"path": str(sample_csv), "schema": DYNAMIC_SCHEMA},
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
                "options": {"path": str(sample_csv), "schema": DYNAMIC_SCHEMA},
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
                "options": {"path": str(sample_csv), "schema": DYNAMIC_SCHEMA},
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


class TestRunWithGates:
    """Test run command with gate plugins."""

    @pytest.fixture
    def sample_csv(self, tmp_path: Path) -> Path:
        """Create sample CSV with varied scores for filtering."""
        csv_file = tmp_path / "input.csv"
        csv_file.write_text("id,name,score\n1,alice,75\n2,bob,45\n3,carol,90\n")
        return csv_file

    @pytest.fixture
    def settings_with_threshold_gate(self, tmp_path: Path, sample_csv: Path) -> Path:
        """Config with threshold gate that routes high/low scores to different sinks."""
        high_output = tmp_path / "high_scores.csv"
        low_output = tmp_path / "low_scores.csv"
        config = {
            "datasource": {
                "plugin": "csv",
                "options": {"path": str(sample_csv), "schema": DYNAMIC_SCHEMA},
            },
            "row_plugins": [
                {
                    "plugin": "threshold_gate",
                    "type": "gate",
                    "options": {
                        "field": "score",
                        "threshold": 60,
                    },
                    # Routes map gate labels ("above"/"below") to sinks
                    "routes": {
                        "above": "high",
                        "below": "low",
                    },
                }
            ],
            "sinks": {
                "high": {
                    "plugin": "csv",
                    "options": {"path": str(high_output), "schema": DYNAMIC_SCHEMA},
                },
                "low": {
                    "plugin": "csv",
                    "options": {"path": str(low_output), "schema": DYNAMIC_SCHEMA},
                },
            },
            "output_sink": "high",  # Default, but gate routes explicitly
            "landscape": {"url": f"sqlite:///{tmp_path}/audit.db"},
        }
        settings_file = tmp_path / "settings.yaml"
        settings_file.write_text(yaml.dump(config))
        return settings_file

    @pytest.fixture
    def settings_with_field_match_gate(self, tmp_path: Path) -> Path:
        """Config with field match gate that routes by name pattern."""
        csv_file = tmp_path / "input.csv"
        csv_file.write_text("id,name,dept\n1,alice,eng\n2,bob,sales\n3,carol,eng\n")
        eng_output = tmp_path / "engineering.csv"
        other_output = tmp_path / "other.csv"
        config = {
            "datasource": {
                "plugin": "csv",
                "options": {"path": str(csv_file), "schema": DYNAMIC_SCHEMA},
            },
            "row_plugins": [
                {
                    "plugin": "field_match_gate",
                    "type": "gate",
                    "options": {
                        "field": "dept",
                        # matches maps field values to route labels
                        "matches": {"eng": "engineering"},
                        "default_label": "other",
                    },
                    # Routes map gate labels to sinks
                    "routes": {
                        "engineering": "engineering",
                        "other": "other",
                    },
                }
            ],
            "sinks": {
                "engineering": {
                    "plugin": "csv",
                    "options": {"path": str(eng_output), "schema": DYNAMIC_SCHEMA},
                },
                "other": {
                    "plugin": "csv",
                    "options": {"path": str(other_output), "schema": DYNAMIC_SCHEMA},
                },
            },
            "output_sink": "other",
            "landscape": {"url": f"sqlite:///{tmp_path}/audit.db"},
        }
        settings_file = tmp_path / "settings.yaml"
        settings_file.write_text(yaml.dump(config))
        return settings_file

    def test_run_with_threshold_gate_routes_correctly(
        self, tmp_path: Path, settings_with_threshold_gate: Path
    ) -> None:
        """Threshold gate routes rows to correct sinks based on score."""
        from elspeth.cli import app

        result = runner.invoke(
            app, ["run", "-s", str(settings_with_threshold_gate), "-x"]
        )
        assert result.exit_code == 0, f"Failed with: {result.output}"

        high_output = (tmp_path / "high_scores.csv").read_text()
        low_output = (tmp_path / "low_scores.csv").read_text()

        # alice (75) and carol (90) should be in high
        assert "alice" in high_output
        assert "carol" in high_output
        # bob (45) should be in low
        assert "bob" in low_output
        # bob should NOT be in high
        assert "bob" not in high_output

    def test_run_with_field_match_gate_routes_by_pattern(
        self, tmp_path: Path, settings_with_field_match_gate: Path
    ) -> None:
        """Field match gate routes rows based on field value matching."""
        from elspeth.cli import app

        result = runner.invoke(
            app, ["run", "-s", str(settings_with_field_match_gate), "-x"]
        )
        assert result.exit_code == 0, f"Failed with: {result.output}"

        eng_output = (tmp_path / "engineering.csv").read_text()
        other_output = (tmp_path / "other.csv").read_text()

        # alice and carol (dept=eng) should be in engineering
        assert "alice" in eng_output
        assert "carol" in eng_output
        # bob (dept=sales) should be in other
        assert "bob" in other_output
        # bob should NOT be in engineering
        assert "bob" not in eng_output


class TestRunWithTransformAndGate:
    """Test pipelines combining transforms and gates."""

    @pytest.fixture
    def settings_with_transform_then_gate(self, tmp_path: Path) -> Path:
        """Config with transform followed by gate."""
        csv_file = tmp_path / "input.csv"
        csv_file.write_text("id,name,points\n1,alice,75\n2,bob,45\n3,carol,90\n")
        pass_output = tmp_path / "passed.csv"
        fail_output = tmp_path / "failed.csv"
        config = {
            "datasource": {
                "plugin": "csv",
                "options": {"path": str(csv_file), "schema": DYNAMIC_SCHEMA},
            },
            "row_plugins": [
                # First: rename points -> score
                {
                    "plugin": "field_mapper",
                    "type": "transform",
                    "options": {
                        "schema": DYNAMIC_SCHEMA,
                        "mapping": {"points": "score"},
                    },
                },
                # Then: filter by the renamed field
                {
                    "plugin": "threshold_gate",
                    "type": "gate",
                    "options": {
                        "field": "score",
                        "threshold": 50,
                    },
                    # Routes map gate labels to sinks
                    "routes": {
                        "above": "passed",
                        "below": "failed",
                    },
                },
            ],
            "sinks": {
                "passed": {
                    "plugin": "csv",
                    "options": {"path": str(pass_output), "schema": DYNAMIC_SCHEMA},
                },
                "failed": {
                    "plugin": "csv",
                    "options": {"path": str(fail_output), "schema": DYNAMIC_SCHEMA},
                },
            },
            "output_sink": "passed",
            "landscape": {"url": f"sqlite:///{tmp_path}/audit.db"},
        }
        settings_file = tmp_path / "settings.yaml"
        settings_file.write_text(yaml.dump(config))
        return settings_file

    def test_transform_then_gate_pipeline(
        self, tmp_path: Path, settings_with_transform_then_gate: Path
    ) -> None:
        """Transform executes before gate, gate uses transformed data."""
        from elspeth.cli import app

        result = runner.invoke(
            app, ["run", "-s", str(settings_with_transform_then_gate), "-x"]
        )
        assert result.exit_code == 0, f"Failed with: {result.output}"

        pass_output = (tmp_path / "passed.csv").read_text()
        fail_output = (tmp_path / "failed.csv").read_text()

        # alice (75) and carol (90) pass threshold
        assert "alice" in pass_output
        assert "carol" in pass_output
        # bob (45) fails threshold
        assert "bob" in fail_output
        # Verify the field was renamed (score not points)
        assert "score" in pass_output
