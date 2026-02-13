"""Tests for CSVSink append mode."""

from pathlib import Path

import pytest

from elspeth.contracts.plugin_context import PluginContext
from elspeth.plugins.sinks.csv_sink import CSVSink

# Strict schema config for tests - CSVSink requires fixed columns
STRICT_SCHEMA = {"mode": "fixed", "fields": ["id: int", "value: str"]}


@pytest.fixture
def ctx() -> PluginContext:
    """Create test context."""
    return PluginContext(run_id="test", config={})


class TestCSVSinkAppendMode:
    """Tests for CSVSink append mode."""

    def test_append_mode_adds_to_existing_file(self, tmp_path: Path, ctx: PluginContext) -> None:
        """Append mode should add rows without rewriting header."""
        output_path = tmp_path / "output.csv"

        # First write in normal mode
        sink1 = CSVSink(
            {
                "path": str(output_path),
                "schema": STRICT_SCHEMA,
            }
        )
        sink1.write([{"id": 1, "value": "a"}], ctx)
        sink1.flush()
        sink1.close()

        # Verify first write
        content1 = output_path.read_text()
        lines1 = content1.strip().split("\n")
        assert len(lines1) == 2  # header + 1 row
        assert "id" in lines1[0] and "value" in lines1[0]

        # Second write in append mode
        sink2 = CSVSink(
            {
                "path": str(output_path),
                "schema": STRICT_SCHEMA,
                "mode": "append",
            }
        )
        sink2.write([{"id": 2, "value": "b"}], ctx)
        sink2.flush()
        sink2.close()

        # Should have both rows, one header
        content2 = output_path.read_text()
        lines2 = content2.strip().split("\n")
        assert len(lines2) == 3  # header + 2 rows

    def test_append_mode_reads_headers_from_existing_file(self, tmp_path: Path, ctx: PluginContext) -> None:
        """Append mode should use headers from existing file."""
        output_path = tmp_path / "output.csv"

        # Use a schema matching the test data
        name_age_schema = {"mode": "fixed", "fields": ["name: str", "age: int"]}

        # First write with specific column order
        sink1 = CSVSink(
            {
                "path": str(output_path),
                "schema": name_age_schema,
            }
        )
        sink1.write([{"name": "Alice", "age": 30}], ctx)
        sink1.flush()
        sink1.close()

        # Get the original header order
        content1 = output_path.read_text()
        original_header = content1.strip().split("\n")[0]

        # Append with same fields (order might differ in dict)
        sink2 = CSVSink(
            {
                "path": str(output_path),
                "schema": name_age_schema,
                "mode": "append",
            }
        )
        sink2.write([{"age": 25, "name": "Bob"}], ctx)  # Different order
        sink2.flush()
        sink2.close()

        # Columns should match original order
        content = output_path.read_text()
        lines = content.strip().split("\n")
        # Header should be preserved from original file
        assert lines[0] == original_header
        # Data rows should follow header column order
        assert len(lines) == 3  # header + 2 rows

    def test_append_mode_creates_file_if_not_exists(self, tmp_path: Path, ctx: PluginContext) -> None:
        """Append mode should create file with header if it doesn't exist."""
        output_path = tmp_path / "new_file.csv"
        assert not output_path.exists()

        sink = CSVSink(
            {
                "path": str(output_path),
                "schema": STRICT_SCHEMA,
                "mode": "append",
            }
        )
        sink.write([{"id": 1, "value": "created"}], ctx)
        sink.flush()
        sink.close()

        # Should create file with header
        content = output_path.read_text()
        lines = content.strip().split("\n")
        assert len(lines) == 2  # header + row
        assert "id" in lines[0]

    def test_default_mode_is_write(self, tmp_path: Path, ctx: PluginContext) -> None:
        """Default mode should be 'write' (truncate)."""
        output_path = tmp_path / "output.csv"

        # First write
        sink1 = CSVSink(
            {
                "path": str(output_path),
                "schema": STRICT_SCHEMA,
            }
        )
        sink1.write([{"id": 1, "value": "first"}], ctx)
        sink1.flush()
        sink1.close()

        # Second write without mode (should truncate)
        sink2 = CSVSink(
            {
                "path": str(output_path),
                "schema": STRICT_SCHEMA,
            }
        )
        sink2.write([{"id": 2, "value": "second"}], ctx)
        sink2.flush()
        sink2.close()

        # Should only have second row
        content = output_path.read_text()
        lines = content.strip().split("\n")
        assert len(lines) == 2  # header + 1 row
        assert "2" in lines[1]
        assert "1" not in lines[1]  # First row data should be gone

    def test_append_mode_with_empty_existing_file(self, tmp_path: Path, ctx: PluginContext) -> None:
        """Append mode with empty file should add header and data."""
        output_path = tmp_path / "empty.csv"
        output_path.touch()  # Create empty file

        sink = CSVSink(
            {
                "path": str(output_path),
                "schema": STRICT_SCHEMA,
                "mode": "append",
            }
        )
        sink.write([{"id": 1, "value": "test"}], ctx)
        sink.flush()
        sink.close()

        # Should have header and data (treated like new file)
        content = output_path.read_text()
        lines = content.strip().split("\n")
        assert len(lines) == 2  # header + row
        assert "id" in lines[0] and "value" in lines[0]

    def test_append_mode_multiple_appends(self, tmp_path: Path, ctx: PluginContext) -> None:
        """Multiple append operations should accumulate rows."""
        output_path = tmp_path / "output.csv"

        # Initial write
        sink1 = CSVSink(
            {
                "path": str(output_path),
                "schema": STRICT_SCHEMA,
            }
        )
        sink1.write([{"id": 1, "value": "first"}], ctx)
        sink1.flush()
        sink1.close()

        # First append
        sink2 = CSVSink(
            {
                "path": str(output_path),
                "schema": STRICT_SCHEMA,
                "mode": "append",
            }
        )
        sink2.write([{"id": 2, "value": "second"}], ctx)
        sink2.flush()
        sink2.close()

        # Second append
        sink3 = CSVSink(
            {
                "path": str(output_path),
                "schema": STRICT_SCHEMA,
                "mode": "append",
            }
        )
        sink3.write([{"id": 3, "value": "third"}], ctx)
        sink3.flush()
        sink3.close()

        # Should have all rows
        content = output_path.read_text()
        lines = content.strip().split("\n")
        assert len(lines) == 4  # header + 3 rows

    def test_append_mode_missing_required_field_preserves_existing_file(self, tmp_path: Path, ctx: PluginContext) -> None:
        """Append with missing required field fails before mutating existing output."""
        output_path = tmp_path / "output.csv"

        sink1 = CSVSink(
            {
                "path": str(output_path),
                "schema": STRICT_SCHEMA,
            }
        )
        sink1.write([{"id": 1, "value": "a"}], ctx)
        sink1.flush()
        sink1.close()
        original_content = output_path.read_text()

        sink2 = CSVSink(
            {
                "path": str(output_path),
                "schema": STRICT_SCHEMA,
                "mode": "append",
            }
        )
        with pytest.raises(ValueError, match="missing required fields"):
            sink2.write([{"id": 2}], ctx)  # Missing required "value"
        sink2.close()

        assert output_path.read_text() == original_content


class TestCSVSinkAppendExplicitSchema:
    """Tests for CSVSink append mode with explicit schema.

    Bug: P1-2026-01-21-csvsink-append-schema-mismatch
    In append mode, CSVSink reads existing CSV headers and uses them
    without validating against the configured explicit schema.
    """

    def test_append_explicit_schema_rejects_missing_fields(self, tmp_path: Path, ctx: PluginContext) -> None:
        """Append mode should fail fast when file headers are missing schema fields.

        When schema is explicit (strict mode), append mode must validate that
        existing file headers contain all required schema fields. Missing fields
        should raise a clear error at file open time, not during write.
        """
        import csv

        output_path = tmp_path / "output.csv"

        # Create file with ONLY 'id' column (missing 'score' from schema)
        with open(output_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=["id"])
            writer.writeheader()
            writer.writerow({"id": 1})

        # Strict schema requires both 'id' and 'score'
        strict_schema = {
            "mode": "fixed",
            "fields": ["id: int", "score: float?"],
        }

        sink = CSVSink(
            {
                "path": str(output_path),
                "schema": strict_schema,
                "mode": "append",
            }
        )

        # Should fail fast at write time (when _open_file is called)
        # with a clear error about schema mismatch
        with pytest.raises(ValueError, match=r"schema.*mismatch|missing.*field|Missing"):
            sink.write([{"id": 2, "score": 1.5}], ctx)

    def test_append_explicit_schema_rejects_wrong_order_strict(self, tmp_path: Path, ctx: PluginContext) -> None:
        """Append mode with strict schema should reject headers in wrong order.

        Strict mode requires exact field order match, not just presence.
        """
        import csv

        output_path = tmp_path / "output.csv"

        # Create file with fields in different order than schema
        with open(output_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=["score", "id"])  # Wrong order
            writer.writeheader()
            writer.writerow({"score": 1.0, "id": 1})

        # Strict schema requires exact order: id, score
        strict_schema = {
            "mode": "fixed",
            "fields": ["id: int", "score: float"],
        }

        sink = CSVSink(
            {
                "path": str(output_path),
                "schema": strict_schema,
                "mode": "append",
            }
        )

        # Should fail because order doesn't match
        with pytest.raises(ValueError, match=r"schema.*mismatch|order"):
            sink.write([{"id": 2, "score": 2.0}], ctx)

    def test_append_explicit_schema_accepts_matching_headers(self, tmp_path: Path, ctx: PluginContext) -> None:
        """Append mode should succeed when file headers match explicit schema."""
        import csv

        output_path = tmp_path / "output.csv"

        # Create file with headers matching schema
        with open(output_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=["id", "score"])
            writer.writeheader()
            writer.writerow({"id": 1, "score": 1.0})

        # Strict schema matches file headers
        strict_schema = {
            "mode": "fixed",
            "fields": ["id: int", "score: float"],
        }

        sink = CSVSink(
            {
                "path": str(output_path),
                "schema": strict_schema,
                "mode": "append",
            }
        )

        # Should succeed - headers match schema
        artifact = sink.write([{"id": 2, "score": 2.0}], ctx)
        sink.close()

        assert artifact.size_bytes > 0

        # Verify both rows present
        content = output_path.read_text()
        lines = content.strip().split("\n")
        assert len(lines) == 3  # header + 2 rows
