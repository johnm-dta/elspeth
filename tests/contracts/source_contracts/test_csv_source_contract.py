# tests/contracts/source_contracts/test_csv_source_contract.py
"""Contract tests for CSVSource plugin.

Verifies CSVSource honors the SourceProtocol contract.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import pytest

from elspeth.contracts import NodeType, SourceRow
from elspeth.contracts.schema import SchemaConfig
from elspeth.core.canonical import CANONICAL_VERSION, stable_hash
from elspeth.core.landscape import LandscapeDB, LandscapeRecorder
from elspeth.plugins.context import PluginContext
from elspeth.plugins.sources.csv_source import CSVSource

from .test_source_protocol import SourceContractPropertyTestBase

if TYPE_CHECKING:
    from elspeth.plugins.protocols import SourceProtocol


class TestCSVSourceContract(SourceContractPropertyTestBase):
    """Contract tests for CSVSource."""

    @pytest.fixture
    def source_data(self, tmp_path: Path) -> Path:
        """Create a test CSV file."""
        csv_file = tmp_path / "test_data.csv"
        csv_file.write_text("id,name,value\n1,Alice,100\n2,Bob,200\n3,Charlie,300\n")
        return csv_file

    @pytest.fixture
    def source(self, source_data: Path) -> SourceProtocol:
        """Create a CSVSource instance with dynamic schema."""
        return CSVSource(
            {
                "path": str(source_data),
                "schema": {"fields": "dynamic"},
                "on_validation_failure": "discard",
            }
        )

    # Additional CSVSource-specific contract tests

    def test_csv_source_respects_delimiter(self, tmp_path: Path) -> None:
        """CSVSource MUST respect delimiter configuration."""
        tsv_file = tmp_path / "data.tsv"
        tsv_file.write_text("id\tname\n1\tAlice\n2\tBob\n")

        source = CSVSource(
            {
                "path": str(tsv_file),
                "delimiter": "\t",
                "schema": {"fields": "dynamic"},
                "on_validation_failure": "discard",
            }
        )
        ctx = PluginContext(run_id="test", config={})

        rows = list(source.load(ctx))
        assert len(rows) == 2
        for row in rows:
            assert isinstance(row, SourceRow)
            if not row.is_quarantined:
                # Should have correctly parsed columns
                assert "id" in row.row
                assert "name" in row.row

    def test_csv_source_handles_empty_file(self, tmp_path: Path) -> None:
        """CSVSource: Empty files return no rows gracefully.

        Note: After the csv.reader refactor, empty files are handled
        gracefully by detecting StopIteration on the first next() call
        (no headers available). This is better than raising EmptyDataError.
        """
        empty_file = tmp_path / "empty.csv"
        empty_file.write_text("")

        source = CSVSource(
            {
                "path": str(empty_file),
                "schema": {"fields": "dynamic"},
                "on_validation_failure": "discard",
            }
        )
        ctx = PluginContext(run_id="test", config={})

        # Empty file returns no rows gracefully (no error)
        rows = list(source.load(ctx))
        assert len(rows) == 0

    def test_csv_source_handles_header_only(self, tmp_path: Path) -> None:
        """CSVSource MUST handle files with only headers."""
        header_only = tmp_path / "header_only.csv"
        header_only.write_text("id,name,value\n")

        source = CSVSource(
            {
                "path": str(header_only),
                "schema": {"fields": "dynamic"},
                "on_validation_failure": "discard",
            }
        )
        ctx = PluginContext(run_id="test", config={})

        rows = list(source.load(ctx))
        assert rows == []


class TestCSVSourceQuarantineContract(SourceContractPropertyTestBase):
    """Contract tests for CSVSource quarantine behavior.

    Verifies that validation failures produce proper SourceRow.quarantined() results
    and are recorded in the audit trail.
    """

    @pytest.fixture
    def source_data_with_invalid(self, tmp_path: Path) -> Path:
        """Create a CSV file with rows that will fail strict validation."""
        csv_file = tmp_path / "mixed_data.csv"
        csv_file.write_text("id,name\n1,Alice\nnot_an_int,Bob\n3,Charlie\n")
        return csv_file

    @pytest.fixture
    def source(self, source_data_with_invalid: Path) -> SourceProtocol:
        """Create a CSVSource with strict schema that will quarantine bad rows."""
        return CSVSource(
            {
                "path": str(source_data_with_invalid),
                "schema": {
                    "mode": "strict",
                    "fields": ["id: int", "name: str"],
                },
                "on_validation_failure": "quarantine_sink",
            }
        )

    def test_invalid_rows_are_quarantined(self, source: SourceProtocol) -> None:
        """Contract: Invalid rows MUST be yielded as SourceRow.quarantined()."""
        db = LandscapeDB.in_memory()
        recorder = LandscapeRecorder(db)
        run = recorder.begin_run(config={}, canonical_version=CANONICAL_VERSION, run_id="test-quarantine")

        recorder.register_node(
            run_id=run.run_id,
            plugin_name="csv",
            node_type=NodeType.SOURCE,
            plugin_version="1.0",
            config={},
            schema_config=SchemaConfig.from_dict({"fields": "dynamic"}),
            node_id="csv_source",
            sequence=0,
        )

        ctx = PluginContext(
            run_id=run.run_id,
            config={},
            landscape=recorder,
            node_id="csv_source",
        )
        rows = list(source.load(ctx))

        valid_rows = [r for r in rows if not r.is_quarantined]
        quarantined_rows = [r for r in rows if r.is_quarantined]

        assert len(valid_rows) == 2, f"Expected 2 valid rows, got {len(valid_rows)}"
        assert len(quarantined_rows) == 1, f"Expected 1 quarantined row, got {len(quarantined_rows)}"

        q_row = quarantined_rows[0]
        assert q_row.is_quarantined is True
        assert q_row.row == {"id": "not_an_int", "name": "Bob"}
        assert q_row.quarantine_error is not None, "quarantine_error should be present"
        assert "id" in q_row.quarantine_error.lower()
        assert q_row.quarantine_destination == "quarantine_sink"

        row_hash = stable_hash({"id": "not_an_int", "name": "Bob"})
        errors = recorder.get_validation_errors_for_row(run.run_id, row_hash)
        assert len(errors) == 1
        assert errors[0].schema_mode == "strict"
        assert errors[0].destination == "quarantine_sink"


class TestCSVSourceDiscardContract:
    """Contract tests for CSVSource discard behavior.

    When on_validation_failure="discard", invalid rows should not be yielded
    but MUST still be recorded in the audit trail.
    """

    @pytest.fixture
    def source_data_with_invalid(self, tmp_path: Path) -> Path:
        """Create a CSV file with rows that will fail strict validation."""
        csv_file = tmp_path / "mixed_data.csv"
        csv_file.write_text("id,name\n1,Alice\nnot_an_int,Bob\n3,Charlie\n")
        return csv_file

    def test_discarded_rows_not_yielded_but_recorded(self, source_data_with_invalid: Path) -> None:
        """Contract: When discard mode, invalid rows NOT yielded but MUST be recorded."""
        db = LandscapeDB.in_memory()
        recorder = LandscapeRecorder(db)
        run = recorder.begin_run(config={}, canonical_version=CANONICAL_VERSION, run_id="test-discard")

        recorder.register_node(
            run_id=run.run_id,
            plugin_name="csv",
            node_type=NodeType.SOURCE,
            plugin_version="1.0",
            config={},
            schema_config=SchemaConfig.from_dict({"fields": "dynamic"}),
            node_id="csv_source",
            sequence=0,
        )

        source = CSVSource(
            {
                "path": str(source_data_with_invalid),
                "schema": {
                    "mode": "strict",
                    "fields": ["id: int", "name: str"],
                },
                "on_validation_failure": "discard",
            }
        )
        ctx = PluginContext(
            run_id=run.run_id,
            config={},
            landscape=recorder,
            node_id="csv_source",
        )

        rows = list(source.load(ctx))

        assert len(rows) == 2
        for row in rows:
            assert not row.is_quarantined

        row_hash = stable_hash({"id": "not_an_int", "name": "Bob"})
        errors = recorder.get_validation_errors_for_row(run.run_id, row_hash)
        assert len(errors) == 1
        assert errors[0].schema_mode == "strict"
        assert errors[0].destination == "discard"


class TestCSVSourceFileNotFoundContract:
    """Contract tests for CSVSource file error handling."""

    def test_file_not_found_raises(self, tmp_path: Path) -> None:
        """Contract: Non-existent file MUST raise FileNotFoundError."""
        source = CSVSource(
            {
                "path": str(tmp_path / "nonexistent.csv"),
                "schema": {"fields": "dynamic"},
                "on_validation_failure": "discard",
            }
        )
        ctx = PluginContext(run_id="test", config={})

        with pytest.raises(FileNotFoundError):
            list(source.load(ctx))
