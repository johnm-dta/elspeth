# tests/unit/contracts/source_contracts/test_csv_source_contract.py
"""Contract tests for CSVSource plugin.

Verifies CSVSource honors the SourceProtocol contract.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any

import pytest

from elspeth.contracts import SourceRow
from elspeth.core.canonical import CANONICAL_VERSION, stable_hash
from elspeth.plugins.sources.csv_source import CSVSource
from tests.fixtures.factories import make_context
from tests.fixtures.landscape import make_factory, make_landscape_db, make_recorder_with_run

from .test_source_protocol import SourceContractPropertyTestBase

if TYPE_CHECKING:
    from elspeth.contracts import SourceProtocol


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
        source = CSVSource(
            {
                "path": str(source_data),
                "schema": {"mode": "observed"},
                "on_validation_failure": "discard",
            }
        )
        source.on_success = "output"
        return source

    # Additional CSVSource-specific contract tests

    def test_csv_source_respects_delimiter(self, tmp_path: Path) -> None:
        """CSVSource MUST respect delimiter configuration.

        Asserts column-presence unconditionally for every yielded row and
        explicitly asserts no rows are quarantined. A delimiter regression
        (e.g., comma used when ``\\t`` was configured) would produce
        single-column rows whose only field is the entire ``id\\tname`` blob
        — these would either fail "id"/"name" key membership or, in stricter
        modes, be quarantined. Either failure mode is detectable here.
        """
        tsv_file = tmp_path / "data.tsv"
        tsv_file.write_text("id\tname\n1\tAlice\n2\tBob\n")

        source = CSVSource(
            {
                "path": str(tsv_file),
                "delimiter": "\t",
                "schema": {"mode": "observed"},
                "on_validation_failure": "discard",
            }
        )
        source.on_success = "output"
        db = make_landscape_db()
        factory = make_factory(db)
        ctx = make_context(landscape=factory.plugin_audit_writer())

        rows = list(source.load(ctx))
        assert len(rows) == 2
        for row in rows:
            assert isinstance(row, SourceRow)
            # Happy-path TSV with valid rows: no row should be quarantined.
            # If the delimiter is mishandled the row will either be
            # quarantined (strict modes) or arrive with a single combined
            # column — both are caught below.
            assert not row.is_quarantined, f"Unexpected quarantine for row {row.row!r}: {row.quarantine_error!r}"
            # Column-presence assertions run UNCONDITIONALLY — a delimiter
            # regression that produced a single 'id\tname' column would
            # fail these even if the row weren't quarantined.
            assert "id" in row.row, f"Expected 'id' column in row {row.row!r}; delimiter likely mishandled"
            assert "name" in row.row, f"Expected 'name' column in row {row.row!r}; delimiter likely mishandled"

        # Stronger contract: the actual parsed values must match the TSV.
        assert [dict(r.row) for r in rows] == [
            {"id": "1", "name": "Alice"},
            {"id": "2", "name": "Bob"},
        ]

    def test_csv_source_handles_empty_file(self, tmp_path: Path) -> None:
        """CSVSource: Empty files return no rows gracefully.

        Note: After the csv.reader refactor, empty files are handled
        gracefully by detecting StopIteration on the first next() call
        (no headers available). No schema contract or validation error is
        fabricated because there is no header or row payload to audit.
        """
        empty_file = tmp_path / "empty.csv"
        empty_file.write_text("")

        setup = make_recorder_with_run(
            run_id="test-empty-file",
            source_node_id="csv_source",
            source_plugin_name="csv",
            canonical_version=CANONICAL_VERSION,
        )
        source = CSVSource(
            {
                "path": str(empty_file),
                "schema": {"mode": "observed"},
                "on_validation_failure": "discard",
            }
        )
        source.on_success = "output"
        ctx = make_context(
            run_id=setup.run_id,
            landscape=setup.factory.plugin_audit_writer(),
            node_id=setup.source_node_id,
        )

        rows = list(source.load(ctx))
        assert rows == []
        assert source.get_schema_contract() is None
        assert source.get_field_resolution() is None
        assert setup.factory.data_flow.get_validation_errors_for_run(setup.run_id) == []

    def test_csv_source_handles_header_only(self, tmp_path: Path) -> None:
        """CSVSource MUST handle files with only headers."""
        header_only = tmp_path / "header_only.csv"
        header_only.write_text("id,name,value\n")

        source = CSVSource(
            {
                "path": str(header_only),
                "schema": {"mode": "observed"},
                "on_validation_failure": "discard",
            }
        )
        source.on_success = "output"
        db = make_landscape_db()
        factory = make_factory(db)
        ctx = make_context(landscape=factory.plugin_audit_writer())

        rows = list(source.load(ctx))
        assert rows == []
        contract = source.get_schema_contract()
        assert contract is not None
        assert contract.mode == "OBSERVED"
        assert contract.locked is True
        assert contract.fields == ()
        assert source.get_field_resolution() == (
            {"id": "id", "name": "name", "value": "value"},
            "1.0.1",
        )

    def test_csv_source_load_is_deterministic_for_same_file(self, source_data: Path) -> None:
        """CSVSource MUST emit stable row and contract data for the same file."""

        def load_snapshot(run_id: str) -> list[tuple[Any, bool, dict[str, Any] | None]]:
            setup = make_recorder_with_run(
                run_id=run_id,
                source_node_id="csv_source",
                source_plugin_name="csv",
                canonical_version=CANONICAL_VERSION,
            )
            source = CSVSource(
                {
                    "path": str(source_data),
                    "schema": {"mode": "observed"},
                    "on_validation_failure": "discard",
                }
            )
            source.on_success = "output"
            ctx = make_context(
                run_id=setup.run_id,
                landscape=setup.factory.plugin_audit_writer(),
                node_id=setup.source_node_id,
            )

            return [
                (
                    row.row,
                    row.is_quarantined,
                    row.contract.to_checkpoint_format() if row.contract is not None else None,
                )
                for row in source.load(ctx)
            ]

        assert load_snapshot("test-determinism-a") == load_snapshot("test-determinism-b")


class TestCSVSourceQuarantineContract:
    """Quarantine-specific behaviour for CSVSource.

    Standalone (no protocol-base inheritance) — full SourceProtocol coverage is
    already exercised by ``TestCSVSourceContract`` against a CSV source. Re-running
    the entire property suite under a quarantine-configured fixture added no
    protocol coverage; this class now contains only the quarantine-specific
    assertion that validation failures produce proper SourceRow.quarantined()
    results and are recorded in the audit trail.
    """

    def test_invalid_rows_are_quarantined(self, tmp_path: Path) -> None:
        """Contract: Invalid rows MUST be yielded as SourceRow.quarantined()."""
        csv_file = tmp_path / "mixed_data.csv"
        csv_file.write_text("id,name\n1,Alice\nnot_an_int,Bob\n3,Charlie\n")

        source = CSVSource(
            {
                "path": str(csv_file),
                "schema": {
                    "mode": "fixed",
                    "fields": ["id: int", "name: str"],
                },
                "on_validation_failure": "quarantine_sink",
            }
        )
        source.on_success = "output"

        setup = make_recorder_with_run(
            run_id="test-quarantine",
            source_node_id="csv_source",
            source_plugin_name="csv",
            canonical_version=CANONICAL_VERSION,
        )
        factory, run_id = setup.factory, setup.run_id

        ctx = make_context(
            run_id=run_id,
            landscape=factory.plugin_audit_writer(),
            node_id=setup.source_node_id,
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
        errors = factory.data_flow.get_validation_errors_for_row(run_id, row_hash)
        assert len(errors) == 1
        assert errors[0].schema_mode == "fixed"
        assert errors[0].destination == "quarantine_sink"
        assert ctx.pop_pending_quarantine_validation_error_id(q_row.row) == errors[0].error_id
        assert ctx.pop_pending_quarantine_validation_error_id(q_row.row) is None


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
        setup = make_recorder_with_run(
            run_id="test-discard",
            source_node_id="csv_source",
            source_plugin_name="csv",
            canonical_version=CANONICAL_VERSION,
        )
        factory, run_id = setup.factory, setup.run_id

        source = CSVSource(
            {
                "path": str(source_data_with_invalid),
                "schema": {
                    "mode": "fixed",
                    "fields": ["id: int", "name: str"],
                },
                "on_validation_failure": "discard",
            }
        )
        source.on_success = "output"
        ctx = make_context(
            run_id=run_id,
            landscape=factory.plugin_audit_writer(),
            node_id=setup.source_node_id,
        )

        rows = list(source.load(ctx))

        assert len(rows) == 2
        for row in rows:
            assert not row.is_quarantined

        row_hash = stable_hash({"id": "not_an_int", "name": "Bob"})
        errors = factory.data_flow.get_validation_errors_for_row(run_id, row_hash)
        assert len(errors) == 1
        assert errors[0].schema_mode == "fixed"
        assert errors[0].destination == "discard"


class TestCSVSourceFileNotFoundContract:
    """Contract tests for CSVSource file error handling."""

    def test_file_not_found_raises(self, tmp_path: Path) -> None:
        """Contract: Non-existent file MUST raise FileNotFoundError."""
        source = CSVSource(
            {
                "path": str(tmp_path / "nonexistent.csv"),
                "schema": {"mode": "observed"},
                "on_validation_failure": "discard",
            }
        )
        source.on_success = "output"
        db = make_landscape_db()
        factory = make_factory(db)
        ctx = make_context(landscape=factory.plugin_audit_writer())

        with pytest.raises(FileNotFoundError):
            list(source.load(ctx))
