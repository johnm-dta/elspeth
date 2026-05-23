# tests/integration/cli/test_cli.py
"""Integration tests for CLI end-to-end workflow."""

import json
from pathlib import Path

import pytest
import yaml
from typer.testing import CliRunner

# Note: In Click 8.0+, mix_stderr is no longer a CliRunner parameter.
# Stderr output is combined with stdout by default when using CliRunner.invoke()
runner = CliRunner()


class TestCLIIntegration:
    """End-to-end CLI integration tests."""

    @pytest.fixture
    def sample_csv(self, tmp_path: Path) -> Path:
        """Create sample CSV data."""
        csv_file = tmp_path / "data.csv"
        csv_file.write_text("id,name,score\n1,alice,95\n2,bob,87\n3,carol,92\n")
        return csv_file

    @pytest.fixture
    def pipeline_config(self, tmp_path: Path, sample_csv: Path) -> Path:
        """Create pipeline configuration.

        Note: Uses "default" as primary sink - the Orchestrator routes
        all completed rows to the "default" sink via output_sink.
        """
        config = {
            "sources": {
                "primary": {
                    "plugin": "csv",
                    "on_success": "default",
                    "options": {
                        "path": str(sample_csv),
                        "on_validation_failure": "discard",
                        "schema": {"mode": "observed"},
                    },
                }
            },
            "sinks": {
                # "default" is required - Orchestrator routes completed rows here
                "default": {
                    "plugin": "json",
                    "on_write_failure": "discard",
                    "options": {
                        "path": str(tmp_path / "output.json"),
                        "schema": {"mode": "observed"},
                    },
                },
            },
            # Use temp-path DB to avoid polluting CWD during tests
            "landscape": {"url": f"sqlite:///{tmp_path / 'landscape.db'}"},
        }
        config_file = tmp_path / "settings.yaml"
        config_file.write_text(yaml.dump(config))
        return config_file

    def test_full_workflow_csv_to_json(self, pipeline_config: Path, tmp_path: Path) -> None:
        """Complete workflow: validate, run with --execute, check output."""
        from elspeth.cli import app

        # Step 1: Validate configuration
        result = runner.invoke(app, ["validate", "-s", str(pipeline_config)])
        assert result.exit_code == 0
        assert "valid" in result.stdout.lower()

        # Step 2: Run pipeline with --execute flag (required for safety)
        result = runner.invoke(app, ["run", "-s", str(pipeline_config), "--execute"])
        assert result.exit_code == 0
        assert "completed" in result.stdout.lower()

        # Step 3: Check output exists and is valid
        output_file = tmp_path / "output.json"
        assert output_file.exists()

        data = json.loads(output_file.read_text())
        assert len(data) == 3
        assert data[0]["name"] == "alice"

    def test_plugins_list_shows_all_types(self) -> None:
        """plugins list shows sources and sinks."""
        from elspeth.cli import app

        result = runner.invoke(app, ["plugins", "list"])
        assert result.exit_code == 0

        # Sources
        assert "csv" in result.stdout
        assert "json" in result.stdout

        # Sinks
        assert "database" in result.stdout

    def test_dry_run_does_not_create_output(self, pipeline_config: Path, tmp_path: Path) -> None:
        """dry-run does not create output files."""
        from elspeth.cli import app

        output_file = tmp_path / "output.json"
        assert not output_file.exists()

        result = runner.invoke(app, ["run", "-s", str(pipeline_config), "--dry-run"])
        assert result.exit_code == 0

        # Output should NOT be created
        assert not output_file.exists()

    def test_run_without_flags_exits_with_warning(self, pipeline_config: Path) -> None:
        """run without --execute shows warning and exits non-zero."""
        from elspeth.cli import app

        result = runner.invoke(app, ["run", "-s", str(pipeline_config)])

        # Should exit with code 1 (safety feature)
        assert result.exit_code == 1
        # Should tell user to add --execute flag (in stderr, captured in output)
        assert "--execute" in result.output


class TestSourceQuarantineRouting:
    """Integration tests for source quarantine routing to sinks.

    Verifies that invalid source rows with on_validation_failure configured
    to a sink name are actually routed to that sink (not silently dropped).
    """

    @pytest.fixture
    def csv_with_invalid_rows(self, tmp_path: Path) -> Path:
        """Create CSV with mixed valid and invalid rows.

        Uses strict schema requiring id:int, name:str, score:int.
        Row 2 has score='bad' which fails int validation.
        """
        csv_file = tmp_path / "data.csv"
        csv_file.write_text(
            "id,name,score\n"
            "1,alice,95\n"
            "2,bob,bad\n"  # Invalid: score is not an int
            "3,carol,92\n"
        )
        return csv_file

    @pytest.fixture
    def quarantine_pipeline_config(self, tmp_path: Path, csv_with_invalid_rows: Path) -> Path:
        """Create pipeline with quarantine sink for invalid rows."""
        config = {
            "sources": {
                "primary": {
                    "plugin": "csv",
                    "on_success": "default",
                    "options": {
                        "path": str(csv_with_invalid_rows),
                        "on_validation_failure": "quarantine",  # Route to quarantine sink
                        "schema": {
                            "mode": "fixed",
                            "fields": ["id: int", "name: str", "score: int"],
                        },
                    },
                }
            },
            "sinks": {
                "default": {
                    "plugin": "json",
                    "on_write_failure": "discard",
                    "options": {
                        "path": str(tmp_path / "output.json"),
                        "schema": {"mode": "observed"},
                    },
                },
                "quarantine": {
                    "plugin": "json",
                    "on_write_failure": "discard",
                    "options": {
                        "path": str(tmp_path / "quarantine.json"),
                        "schema": {"mode": "observed"},
                    },
                },
            },
            "landscape": {"url": f"sqlite:///{tmp_path / 'landscape.db'}"},
        }
        config_file = tmp_path / "settings.yaml"
        config_file.write_text(yaml.dump(config))
        return config_file

    def test_invalid_rows_routed_failure_to_quarantine_sink(self, quarantine_pipeline_config: Path, tmp_path: Path) -> None:
        """Invalid source rows are written to the quarantine sink.

        Post elspeth-5069612f3c (rows_routed counter split): invalid source rows
        traverse the on_validation_failure DIVERT path, which increments
        rows_routed_failure (NOT rows_routed_success).  The test name reflects
        the new failure-side semantics.

        This is the key acceptance test for the source quarantine routing feature.
        Before this fix, route_to_sink() was a stub and invalid rows were dropped.
        """
        from elspeth.cli import app

        # Run the pipeline
        result = runner.invoke(app, ["run", "-s", str(quarantine_pipeline_config), "--execute"])
        assert result.exit_code == 0

        # Check valid rows went to default output
        output_file = tmp_path / "output.json"
        assert output_file.exists()
        data = json.loads(output_file.read_text())
        assert len(data) == 2  # alice and carol (valid rows)
        assert {d["name"] for d in data} == {"alice", "carol"}

        # Check invalid row went to quarantine sink
        quarantine_file = tmp_path / "quarantine.json"
        assert quarantine_file.exists(), "Quarantine sink should receive invalid rows"
        quarantine_data = json.loads(quarantine_file.read_text())
        assert len(quarantine_data) == 1  # bob (invalid row)
        assert quarantine_data[0]["name"] == "bob"
        assert quarantine_data[0]["score"] == "bad"  # Original value preserved

    def test_discard_does_not_write_to_sink(self, tmp_path: Path, csv_with_invalid_rows: Path) -> None:
        """When on_validation_failure='discard', invalid rows are not written."""
        config = {
            "sources": {
                "primary": {
                    "plugin": "csv",
                    "on_success": "default",
                    "options": {
                        "path": str(csv_with_invalid_rows),
                        "on_validation_failure": "discard",  # Intentionally drop
                        "schema": {
                            "mode": "fixed",
                            "fields": ["id: int", "name: str", "score: int"],
                        },
                    },
                }
            },
            "sinks": {
                "default": {
                    "plugin": "json",
                    "on_write_failure": "discard",
                    "options": {
                        "path": str(tmp_path / "output.json"),
                        "schema": {"mode": "observed"},
                    },
                },
            },
            "landscape": {"url": f"sqlite:///{tmp_path / 'landscape.db'}"},
        }
        config_file = tmp_path / "settings.yaml"
        config_file.write_text(yaml.dump(config))

        from elspeth.cli import app

        result = runner.invoke(app, ["run", "-s", str(config_file), "--execute"])
        assert result.exit_code == 0

        # Only valid rows in output
        output_file = tmp_path / "output.json"
        data = json.loads(output_file.read_text())
        assert len(data) == 2  # alice and carol only


class TestTransformErrorSinkRouting:
    """Tests for sinks only reachable via transform on_error divert edges.

    Verifies that a sink referenced only by on_error doesn't break DAG
    validation. The __error_N__ divert edge makes it reachable in the graph.
    """

    def test_dedicated_error_sink_does_not_break_graph(self, tmp_path: Path) -> None:
        """A sink only referenced by on_error must not break DAG validation."""
        input_csv = tmp_path / "input.csv"
        input_csv.write_text("id,value\n1,good\n2,bad\n3,ok\n")

        config = {
            "sources": {
                "primary": {
                    "plugin": "csv",
                    "on_success": "passthrough_input",
                    "options": {
                        "path": str(input_csv),
                        "on_validation_failure": "discard",
                        "schema": {"mode": "observed"},
                    },
                }
            },
            "transforms": [
                {
                    "name": "passthrough_0",
                    "plugin": "passthrough",
                    "input": "passthrough_input",
                    "on_success": "default",
                    "on_error": "errors",
                    "options": {"schema": {"mode": "observed"}},
                },
            ],
            "sinks": {
                "default": {
                    "plugin": "json",
                    "on_write_failure": "discard",
                    "options": {
                        "path": str(tmp_path / "output.json"),
                        "schema": {"mode": "observed"},
                    },
                },
                "errors": {
                    "plugin": "json",
                    "on_write_failure": "discard",
                    "options": {
                        "path": str(tmp_path / "errors.json"),
                        "schema": {"mode": "observed"},
                    },
                },
            },
            "landscape": {"url": f"sqlite:///{tmp_path / 'landscape.db'}"},
        }

        config_path = tmp_path / "settings.yaml"
        config_path.write_text(yaml.dump(config))

        from elspeth.cli import app

        result = runner.invoke(app, ["run", "-s", str(config_path), "--execute"])
        assert result.exit_code == 0, f"Pipeline failed: {result.output}"
