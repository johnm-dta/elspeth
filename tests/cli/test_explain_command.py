"""Tests for elspeth explain command.

Validates explain command exit codes, output format, and error messages.
"""

import json
from pathlib import Path

import pytest
from typer.testing import CliRunner

from elspeth.cli import app
from elspeth.contracts import NodeType, RowOutcome, RunStatus
from elspeth.contracts.schema import SchemaConfig
from elspeth.core.landscape import (
    LandscapeDB,
    LandscapeRecorder,
    dataclass_to_dict,
)
from elspeth.core.landscape import explain as explain_lineage

runner = CliRunner()

DYNAMIC_SCHEMA = SchemaConfig.from_dict({"fields": "dynamic"})


class TestExplainCommandBasics:
    """Basic CLI tests for explain command."""

    def test_explain_requires_run_id(self) -> None:
        """explain requires --run option and exits with specific error."""
        result = runner.invoke(app, ["explain"])
        # Typer exits with code 2 for missing required options
        assert result.exit_code == 2, f"Expected exit code 2 for missing --run, got {result.exit_code}"
        # Error message should mention the missing option (output includes stderr)
        output = result.output.lower()
        assert "missing option" in output or "--run" in output, f"Expected error about missing --run option, got: {output}"


class TestExplainJsonMode:
    """Tests for explain --json mode with real data."""

    @pytest.fixture
    def db_with_run(self, tmp_path: Path) -> tuple[Path, str, str]:
        """Create database with a simple completed run."""
        db_path = tmp_path / "audit.db"
        db = LandscapeDB.from_url(f"sqlite:///{db_path}")
        recorder = LandscapeRecorder(db)

        run = recorder.begin_run(config={}, canonical_version="v1")
        source_node = recorder.register_node(
            run_id=run.run_id,
            plugin_name="csv",
            node_type=NodeType.SOURCE,
            plugin_version="1.0",
            config={},
            schema_config=DYNAMIC_SCHEMA,
        )
        row = recorder.create_row(
            run_id=run.run_id,
            source_node_id=source_node.node_id,
            row_index=0,
            data={"id": 1, "name": "test"},
        )
        token = recorder.create_token(row_id=row.row_id)
        recorder.record_token_outcome(
            token_id=token.token_id,
            run_id=run.run_id,
            outcome=RowOutcome.COMPLETED,
            sink_name="output",
        )
        recorder.complete_run(run.run_id, status=RunStatus.COMPLETED)
        db.close()

        return db_path, run.run_id, token.token_id

    @pytest.fixture
    def db_with_forked_row(self, tmp_path: Path) -> tuple[Path, str, str]:
        """Create database with a row that forked to multiple sinks."""
        db_path = tmp_path / "audit.db"
        db = LandscapeDB.from_url(f"sqlite:///{db_path}")
        recorder = LandscapeRecorder(db)

        run = recorder.begin_run(config={}, canonical_version="v1")
        source_node = recorder.register_node(
            run_id=run.run_id,
            plugin_name="csv",
            node_type=NodeType.SOURCE,
            plugin_version="1.0",
            config={},
            schema_config=DYNAMIC_SCHEMA,
        )
        row = recorder.create_row(
            run_id=run.run_id,
            source_node_id=source_node.node_id,
            row_index=0,
            data={"id": 1},
        )

        # Fork to two different sinks
        token_a = recorder.create_token(row_id=row.row_id)
        token_b = recorder.create_token(row_id=row.row_id)

        recorder.record_token_outcome(
            token_id=token_a.token_id,
            run_id=run.run_id,
            outcome=RowOutcome.ROUTED,
            sink_name="sink_a",
        )
        recorder.record_token_outcome(
            token_id=token_b.token_id,
            run_id=run.run_id,
            outcome=RowOutcome.ROUTED,
            sink_name="sink_b",
        )
        recorder.complete_run(run.run_id, status=RunStatus.COMPLETED)
        db.close()

        return db_path, run.run_id, row.row_id

    def test_json_output_returns_lineage(self, db_with_run: tuple[Path, str, str]) -> None:
        """--json returns real lineage data."""
        db_path, run_id, token_id = db_with_run

        result = runner.invoke(app, ["explain", "--run", run_id, "--token", token_id, "--database", str(db_path), "--json"])

        assert result.exit_code == 0, f"Expected exit 0, got {result.exit_code}: {result.output}"
        data = json.loads(result.output)

        assert data["token"]["token_id"] == token_id
        assert data["source_row"]["row_id"] is not None
        assert data["outcome"]["outcome"] == "completed"

    def test_json_output_with_row_id(self, db_with_run: tuple[Path, str, str]) -> None:
        """--json with --row returns lineage."""
        db_path, run_id, token_id = db_with_run

        # Get row_id from the token we created
        db = LandscapeDB.from_url(f"sqlite:///{db_path}")
        recorder = LandscapeRecorder(db)
        token = recorder.get_token(token_id)
        row_id = token.row_id
        db.close()

        result = runner.invoke(app, ["explain", "--run", run_id, "--row", row_id, "--database", str(db_path), "--json"])

        assert result.exit_code == 0
        data = json.loads(result.output)
        assert data["source_row"]["row_id"] == row_id

    def test_json_output_nonexistent_token(self, db_with_run: tuple[Path, str, str]) -> None:
        """--json with nonexistent token returns error JSON."""
        db_path, run_id, _ = db_with_run

        result = runner.invoke(app, ["explain", "--run", run_id, "--token", "nonexistent", "--database", str(db_path), "--json"])

        assert result.exit_code == 1
        data = json.loads(result.output)
        assert "error" in data

    def test_json_output_latest_run(self, db_with_run: tuple[Path, str, str]) -> None:
        """--run latest resolves to most recent run."""
        db_path, _run_id, token_id = db_with_run

        result = runner.invoke(app, ["explain", "--run", "latest", "--token", token_id, "--database", str(db_path), "--json"])

        assert result.exit_code == 0
        data = json.loads(result.output)
        assert data["token"]["token_id"] == token_id

    def test_json_output_ambiguous_row_without_sink(self, db_with_forked_row: tuple[Path, str, str]) -> None:
        """--json with ambiguous row (multiple tokens) returns helpful error."""
        db_path, run_id, row_id = db_with_forked_row

        result = runner.invoke(app, ["explain", "--run", run_id, "--row", row_id, "--database", str(db_path), "--json"])

        assert result.exit_code == 1
        data = json.loads(result.output)
        assert "error" in data
        # Error should mention multiple tokens or need for sink disambiguation
        error_lower = data["error"].lower()
        assert "terminal tokens" in error_lower or "sink" in error_lower or "multiple" in error_lower

    def test_json_output_matches_backend_explain(self, db_with_run: tuple[Path, str, str]) -> None:
        """CLI JSON output should match direct explain() call (round-trip test)."""
        db_path, run_id, token_id = db_with_run

        # Get backend result
        db = LandscapeDB.from_url(f"sqlite:///{db_path}")
        recorder = LandscapeRecorder(db)
        backend_result = explain_lineage(recorder, run_id=run_id, token_id=token_id)
        backend_json = dataclass_to_dict(backend_result)
        db.close()

        # Get CLI result
        result = runner.invoke(app, ["explain", "--run", run_id, "--token", token_id, "--database", str(db_path), "--json"])
        assert result.exit_code == 0
        cli_json = json.loads(result.output)

        # Should be identical (deep comparison)
        assert cli_json == backend_json, "CLI JSON output differs from backend explain() result"

    def test_json_requires_token_or_row(self, db_with_run: tuple[Path, str, str]) -> None:
        """--json without --token or --row returns error."""
        db_path, run_id, _ = db_with_run

        result = runner.invoke(app, ["explain", "--run", run_id, "--database", str(db_path), "--json"])

        assert result.exit_code == 1
        data = json.loads(result.output)
        assert "error" in data
        assert "token" in data["error"].lower() or "row" in data["error"].lower()

    def test_json_database_not_found(self, tmp_path: Path) -> None:
        """--json with nonexistent database returns error JSON."""
        nonexistent = tmp_path / "does_not_exist.db"

        result = runner.invoke(app, ["explain", "--run", "any", "--token", "any", "--database", str(nonexistent), "--json"])

        assert result.exit_code == 1
        data = json.loads(result.output)
        assert "error" in data

    def test_json_no_runs_in_database(self, tmp_path: Path) -> None:
        """--run latest with empty database returns error JSON."""
        db_path = tmp_path / "empty.db"
        db = LandscapeDB.from_url(f"sqlite:///{db_path}")
        db.close()

        result = runner.invoke(app, ["explain", "--run", "latest", "--token", "any", "--database", str(db_path), "--json"])

        assert result.exit_code == 1
        data = json.loads(result.output)
        assert "error" in data
        assert "no runs" in data["error"].lower()


class TestExplainTextMode:
    """Tests for explain --no-tui text mode."""

    @pytest.fixture
    def db_with_run(self, tmp_path: Path) -> tuple[Path, str, str]:
        """Create database with a simple completed run."""
        db_path = tmp_path / "audit.db"
        db = LandscapeDB.from_url(f"sqlite:///{db_path}")
        recorder = LandscapeRecorder(db)

        run = recorder.begin_run(config={}, canonical_version="v1")
        source_node = recorder.register_node(
            run_id=run.run_id,
            plugin_name="csv",
            node_type=NodeType.SOURCE,
            plugin_version="1.0",
            config={},
            schema_config=DYNAMIC_SCHEMA,
        )
        row = recorder.create_row(
            run_id=run.run_id,
            source_node_id=source_node.node_id,
            row_index=0,
            data={"id": 1, "name": "test"},
        )
        token = recorder.create_token(row_id=row.row_id)
        recorder.record_token_outcome(
            token_id=token.token_id,
            run_id=run.run_id,
            outcome=RowOutcome.COMPLETED,
            sink_name="output",
        )
        recorder.complete_run(run.run_id, status=RunStatus.COMPLETED)
        db.close()

        return db_path, run.run_id, token.token_id

    def test_no_tui_shows_lineage_report(self, db_with_run: tuple[Path, str, str]) -> None:
        """--no-tui shows text lineage report."""
        db_path, run_id, token_id = db_with_run

        result = runner.invoke(app, ["explain", "--run", run_id, "--token", token_id, "--database", str(db_path), "--no-tui"])

        assert result.exit_code == 0, f"Expected exit 0, got {result.exit_code}: {result.output}"
        assert "LINEAGE REPORT" in result.output
        assert token_id in result.output

    def test_no_tui_error_shows_message(self, db_with_run: tuple[Path, str, str]) -> None:
        """--no-tui with nonexistent token shows error message."""
        db_path, run_id, _ = db_with_run

        result = runner.invoke(app, ["explain", "--run", run_id, "--token", "nonexistent", "--database", str(db_path), "--no-tui"])

        assert result.exit_code == 1
        # Error message should be on stderr
        assert "not found" in result.output.lower() or "error" in result.output.lower()
