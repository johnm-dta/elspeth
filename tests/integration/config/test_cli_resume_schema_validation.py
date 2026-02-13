# tests/integration/config/test_cli_resume_schema_validation.py
"""Integration tests for CLI resume schema validation.

Tests that CLI resume properly validates output target schema compatibility
before proceeding with resume operations. This prevents data corruption from
schema drift (P1-2026-01-21-csvsink-append-schema-mismatch).

The validation happens AFTER configure_for_resume() but BEFORE any writes,
ensuring fail-fast behavior when schemas don't match.
"""

import csv
import json
from pathlib import Path

from elspeth.contracts.plugin_context import PluginContext
from elspeth.plugins.sinks.csv_sink import CSVSink
from elspeth.plugins.sinks.database_sink import DatabaseSink
from elspeth.plugins.sinks.json_sink import JSONSink


class TestCSVSinkResumeSchemaValidation:
    """Tests for CSV sink resume schema validation."""

    def test_resume_fails_on_csv_schema_mismatch(self, tmp_path: Path):
        """CLI resume should fail when CSV headers don't match schema."""
        csv_path = tmp_path / "output.csv"

        # Create CSV with wrong headers
        with open(csv_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=["wrong", "headers"])
            writer.writeheader()

        # Create sink with different schema
        sink = CSVSink(
            {
                "path": str(csv_path),
                "schema": {"mode": "fixed", "fields": ["id: int", "name: str"]},
            }
        )

        # Simulate CLI resume flow: configure_for_resume() then validate
        sink.configure_for_resume()
        validation = sink.validate_output_target()

        # CLI would check this and exit with error
        assert validation.valid is False
        assert validation.error_message is not None and "fixed mode" in validation.error_message
        assert "id" in validation.missing_fields
        assert "name" in validation.missing_fields

    def test_resume_succeeds_on_matching_csv_schema(self, tmp_path: Path):
        """CLI resume should proceed when CSV headers match schema."""
        csv_path = tmp_path / "output.csv"

        # Create CSV with correct headers
        with open(csv_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=["id", "name"])
            writer.writeheader()

        sink = CSVSink(
            {
                "path": str(csv_path),
                "schema": {"mode": "fixed", "fields": ["id: int", "name: str"]},
            }
        )

        sink.configure_for_resume()
        validation = sink.validate_output_target()

        assert validation.valid is True


class TestDatabaseSinkResumeSchemaValidation:
    """Tests for database sink resume schema validation."""

    def test_resume_fails_on_table_schema_mismatch(self, tmp_path: Path):
        """CLI resume should fail when table columns don't match schema."""
        from sqlalchemy import Column, MetaData, String, Table, create_engine

        db_path = tmp_path / "test.db"
        url = f"sqlite:///{db_path}"

        # Create table with wrong columns
        engine = create_engine(url)
        metadata = MetaData()
        Table("output_data", metadata, Column("wrong", String), Column("columns", String))
        metadata.create_all(engine)
        engine.dispose()

        sink = DatabaseSink(
            {
                "url": url,
                "table": "output_data",
                "schema": {"mode": "fixed", "fields": ["id: int", "name: str"]},
            }
        )

        sink.configure_for_resume()
        validation = sink.validate_output_target()

        assert validation.valid is False
        assert validation.error_message is not None and "fixed mode" in validation.error_message
        sink.close()

    def test_resume_succeeds_on_matching_table_schema(self, tmp_path: Path):
        """CLI resume should proceed when table columns match schema."""
        from sqlalchemy import Column, Integer, MetaData, String, Table, create_engine

        db_path = tmp_path / "test.db"
        url = f"sqlite:///{db_path}"

        # Create table with correct columns
        engine = create_engine(url)
        metadata = MetaData()
        Table("output_data", metadata, Column("id", Integer), Column("name", String))
        metadata.create_all(engine)
        engine.dispose()

        sink = DatabaseSink(
            {
                "url": url,
                "table": "output_data",
                "schema": {"mode": "fixed", "fields": ["id: int", "name: str"]},
            }
        )

        sink.configure_for_resume()
        validation = sink.validate_output_target()

        assert validation.valid is True
        sink.close()

    def test_resume_validate_then_write_succeeds(self, tmp_path: Path):
        """After validate_output_target(), first write should not fail initialization."""
        from sqlalchemy import Column, Integer, MetaData, String, Table, create_engine, text

        db_path = tmp_path / "test.db"
        url = f"sqlite:///{db_path}"

        # Create table with correct columns for resume validation.
        engine = create_engine(url)
        metadata = MetaData()
        Table("output_data", metadata, Column("id", Integer), Column("name", String))
        metadata.create_all(engine)
        engine.dispose()

        sink = DatabaseSink(
            {
                "url": url,
                "table": "output_data",
                "schema": {"mode": "fixed", "fields": ["id: int", "name: str"]},
            }
        )

        sink.configure_for_resume()
        validation = sink.validate_output_target()
        assert validation.valid is True

        # Regression check: write must not raise RuntimeError after validation.
        sink.write([{"id": 1, "name": "alice"}], PluginContext(run_id="test-run", config={}))
        sink.close()

        engine = create_engine(url)
        with engine.connect() as conn:
            count = conn.execute(text("SELECT COUNT(*) FROM output_data")).scalar_one()
        engine.dispose()
        assert count == 1


class TestJSONLSinkResumeSchemaValidation:
    """Tests for JSONL sink resume schema validation."""

    def test_resume_fails_on_jsonl_schema_mismatch(self, tmp_path: Path):
        """CLI resume should fail when JSONL records don't match schema."""
        jsonl_path = tmp_path / "output.jsonl"

        # Create JSONL with wrong fields
        with open(jsonl_path, "w") as f:
            f.write(json.dumps({"wrong": 1, "fields": 2}) + "\n")

        sink = JSONSink(
            {
                "path": str(jsonl_path),
                "schema": {"mode": "fixed", "fields": ["id: int", "name: str"]},
                "format": "jsonl",
            }
        )

        sink.configure_for_resume()
        validation = sink.validate_output_target()

        assert validation.valid is False
        assert validation.error_message is not None and "fixed mode" in validation.error_message

    def test_resume_succeeds_on_matching_jsonl_schema(self, tmp_path: Path):
        """CLI resume should proceed when JSONL records match schema."""
        jsonl_path = tmp_path / "output.jsonl"

        # Create JSONL with correct fields
        with open(jsonl_path, "w") as f:
            f.write(json.dumps({"id": 1, "name": "test"}) + "\n")

        sink = JSONSink(
            {
                "path": str(jsonl_path),
                "schema": {"mode": "fixed", "fields": ["id: int", "name: str"]},
                "format": "jsonl",
            }
        )

        sink.configure_for_resume()
        validation = sink.validate_output_target()

        assert validation.valid is True


class TestResumeSchemaValidationOrder:
    """Tests verifying validation happens after configure_for_resume."""

    def test_validation_uses_configured_mode(self, tmp_path: Path):
        """Validation should use the mode set by configure_for_resume."""
        csv_path = tmp_path / "output.csv"

        # Create CSV with matching headers
        with open(csv_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=["id", "name"])
            writer.writeheader()

        sink = CSVSink(
            {
                "path": str(csv_path),
                "schema": {"mode": "fixed", "fields": ["id: int", "name: str"]},
                "mode": "write",  # Original mode
            }
        )

        # Before configure_for_resume
        assert sink._mode == "write"

        # Simulate CLI flow
        sink.configure_for_resume()
        assert sink._mode == "append"

        # Validation should work with append mode
        validation = sink.validate_output_target()
        assert validation.valid is True

    def test_validation_result_provides_diagnostic_info(self, tmp_path: Path):
        """Validation failure should include diagnostic information for CLI."""
        csv_path = tmp_path / "output.csv"

        # Create CSV with partially matching headers
        with open(csv_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=["id", "extra"])
            writer.writeheader()

        sink = CSVSink(
            {
                "path": str(csv_path),
                "schema": {"mode": "fixed", "fields": ["id: int", "name: str"]},
            }
        )

        sink.configure_for_resume()
        validation = sink.validate_output_target()

        # CLI uses these fields for error messages
        assert validation.valid is False
        assert validation.error_message is not None
        assert "name" in validation.missing_fields
        assert "extra" in validation.extra_fields
        assert list(validation.target_fields) == ["id", "extra"]
        assert list(validation.schema_fields) == ["id", "name"]


class TestResumeWithOriginalHeaders:
    """Tests for resume validation when headers: original is enabled."""

    def test_csv_resume_with_original_headers_succeeds(self, tmp_path: Path):
        """Resume succeeds when CLI provides field resolution for headers: original."""
        csv_path = tmp_path / "output.csv"

        # Create CSV with original headers (as if previous run used headers: original)
        with open(csv_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=["User ID", "Amount (USD)"])
            writer.writeheader()
            writer.writerow({"User ID": "u1", "Amount (USD)": "100.0"})

        sink = CSVSink(
            {
                "path": str(csv_path),
                "schema": {"mode": "fixed", "fields": ["user_id: str", "amount_usd: float"]},
                "headers": "original",
            }
        )

        sink.configure_for_resume()

        field_resolution = {
            "User ID": "user_id",
            "Amount (USD)": "amount_usd",
        }

        sink.set_resume_field_resolution(field_resolution)

        validation = sink.validate_output_target()
        assert validation.valid is True, f"Validation failed: {validation.error_message}"

    def test_csv_resume_with_original_headers_fails_without_resolution(self, tmp_path: Path):
        """Resume fails when CLI doesn't provide field resolution (bug scenario)."""
        csv_path = tmp_path / "output.csv"

        with open(csv_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=["User ID", "Amount (USD)"])
            writer.writeheader()

        sink = CSVSink(
            {
                "path": str(csv_path),
                "schema": {"mode": "fixed", "fields": ["user_id: str", "amount_usd: float"]},
                "headers": "original",
            }
        )

        sink.configure_for_resume()

        validation = sink.validate_output_target()
        assert validation.valid is False
        assert validation.missing_fields is not None
        assert len(validation.missing_fields) > 0

    def test_jsonl_resume_with_original_headers_succeeds(self, tmp_path: Path):
        """JSONL resume succeeds when CLI provides field resolution."""
        jsonl_path = tmp_path / "output.jsonl"

        with open(jsonl_path, "w") as f:
            f.write(json.dumps({"User ID": "u1", "Amount (USD)": 100.0}) + "\n")

        sink = JSONSink(
            {
                "path": str(jsonl_path),
                "schema": {"mode": "fixed", "fields": ["user_id: str", "amount_usd: float"]},
                "format": "jsonl",
                "headers": "original",
            }
        )

        sink.configure_for_resume()

        field_resolution = {
            "User ID": "user_id",
            "Amount (USD)": "amount_usd",
        }
        sink.set_resume_field_resolution(field_resolution)

        validation = sink.validate_output_target()
        assert validation.valid is True, f"Validation failed: {validation.error_message}"

    def test_no_op_when_not_using_original_headers(self, tmp_path: Path):
        """set_resume_field_resolution() is a no-op when headers mode is not original."""
        csv_path = tmp_path / "output.csv"

        with open(csv_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=["user_id", "amount_usd"])
            writer.writeheader()

        sink = CSVSink(
            {
                "path": str(csv_path),
                "schema": {"mode": "fixed", "fields": ["user_id: str", "amount_usd: float"]},
            }
        )

        sink.configure_for_resume()

        sink.set_resume_field_resolution({"Some Header": "some_field"})

        validation = sink.validate_output_target()
        assert validation.valid is True
