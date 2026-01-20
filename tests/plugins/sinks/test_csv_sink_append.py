"""Tests for CSVSink append mode."""

from pathlib import Path

import pytest

from elspeth.plugins.context import PluginContext
from elspeth.plugins.sinks.csv_sink import CSVSink

# Dynamic schema config for tests
DYNAMIC_SCHEMA = {"fields": "dynamic"}


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
                "schema": DYNAMIC_SCHEMA,
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
                "schema": DYNAMIC_SCHEMA,
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

        # First write with specific column order
        sink1 = CSVSink(
            {
                "path": str(output_path),
                "schema": DYNAMIC_SCHEMA,
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
                "schema": DYNAMIC_SCHEMA,
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
                "schema": DYNAMIC_SCHEMA,
                "mode": "append",
            }
        )
        sink.write([{"id": 1}], ctx)
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
                "schema": DYNAMIC_SCHEMA,
            }
        )
        sink1.write([{"id": 1}], ctx)
        sink1.flush()
        sink1.close()

        # Second write without mode (should truncate)
        sink2 = CSVSink(
            {
                "path": str(output_path),
                "schema": DYNAMIC_SCHEMA,
            }
        )
        sink2.write([{"id": 2}], ctx)
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
                "schema": DYNAMIC_SCHEMA,
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
                "schema": DYNAMIC_SCHEMA,
            }
        )
        sink1.write([{"id": 1}], ctx)
        sink1.flush()
        sink1.close()

        # First append
        sink2 = CSVSink(
            {
                "path": str(output_path),
                "schema": DYNAMIC_SCHEMA,
                "mode": "append",
            }
        )
        sink2.write([{"id": 2}], ctx)
        sink2.flush()
        sink2.close()

        # Second append
        sink3 = CSVSink(
            {
                "path": str(output_path),
                "schema": DYNAMIC_SCHEMA,
                "mode": "append",
            }
        )
        sink3.write([{"id": 3}], ctx)
        sink3.flush()
        sink3.close()

        # Should have all rows
        content = output_path.read_text()
        lines = content.strip().split("\n")
        assert len(lines) == 4  # header + 3 rows
