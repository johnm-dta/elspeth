# tests/contracts/source_contracts/test_csv_source_contract.py
"""Contract tests for CSVSource plugin.

Verifies CSVSource honors the SourceProtocol contract.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import pytest

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
        from elspeth.contracts import SourceRow
        from elspeth.plugins.context import PluginContext

        # Create tab-delimited file
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
        """CSVSource: Empty files raise EmptyDataError (pandas behavior).

        Note: Truly empty CSV files (no headers, no data) cannot be parsed
        by pandas - this is expected behavior. Use header-only files for
        "no data" scenarios.
        """
        import pandas as pd

        from elspeth.plugins.context import PluginContext

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

        # Pandas cannot parse a truly empty file - this is expected
        with pytest.raises(pd.errors.EmptyDataError):
            list(source.load(ctx))

    def test_csv_source_handles_header_only(self, tmp_path: Path) -> None:
        """CSVSource MUST handle files with only headers."""
        from elspeth.plugins.context import PluginContext

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

    Verifies that validation failures produce proper SourceRow.quarantined() results.
    """

    @pytest.fixture
    def source_data_with_invalid(self, tmp_path: Path) -> Path:
        """Create a CSV file with rows that will fail strict validation."""
        csv_file = tmp_path / "mixed_data.csv"
        # id should be int, but "not_an_int" will fail
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
        from elspeth.plugins.context import PluginContext

        ctx = PluginContext(run_id="test", config={})
        rows = list(source.load(ctx))

        # Should have 2 valid rows and 1 quarantined
        valid_rows = [r for r in rows if not r.is_quarantined]
        quarantined_rows = [r for r in rows if r.is_quarantined]

        assert len(valid_rows) == 2, f"Expected 2 valid rows, got {len(valid_rows)}"
        assert (
            len(quarantined_rows) == 1
        ), f"Expected 1 quarantined row, got {len(quarantined_rows)}"

        # Verify quarantined row has proper attributes
        q_row = quarantined_rows[0]
        assert q_row.quarantine_error is not None
        assert q_row.quarantine_destination == "quarantine_sink"
        assert q_row.row is not None  # Original row data preserved


class TestCSVSourceDiscardContract:
    """Contract tests for CSVSource discard behavior.

    When on_validation_failure="discard", invalid rows should not be yielded.
    """

    @pytest.fixture
    def source_data_with_invalid(self, tmp_path: Path) -> Path:
        """Create a CSV file with rows that will fail strict validation."""
        csv_file = tmp_path / "mixed_data.csv"
        csv_file.write_text("id,name\n1,Alice\nnot_an_int,Bob\n3,Charlie\n")
        return csv_file

    def test_discarded_rows_not_yielded(self, source_data_with_invalid: Path) -> None:
        """Contract: When discard mode, invalid rows MUST NOT be yielded."""
        from elspeth.plugins.context import PluginContext

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
        ctx = PluginContext(run_id="test", config={})

        rows = list(source.load(ctx))

        # Should only have 2 valid rows (invalid one discarded)
        assert len(rows) == 2
        for row in rows:
            assert not row.is_quarantined


class TestCSVSourceFileNotFoundContract:
    """Contract tests for CSVSource file error handling."""

    def test_file_not_found_raises(self, tmp_path: Path) -> None:
        """Contract: Non-existent file MUST raise FileNotFoundError."""
        from elspeth.plugins.context import PluginContext

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
