"""Tests for CSVSink resume capability."""

from pathlib import Path

import pytest

from elspeth.contracts.plugin_context import PluginContext
from elspeth.plugins.sinks.csv_sink import CSVSink
from tests.fixtures.base_classes import inject_write_failure
from tests.fixtures.factories import make_context
from tests.fixtures.landscape import make_factory, make_landscape_db

# Strict schema for tests - CSVSink requires fixed columns
STRICT_SCHEMA = {"mode": "fixed", "fields": ["id: int"]}


@pytest.fixture
def ctx() -> PluginContext:
    """Create test context."""
    db = make_landscape_db()
    factory = make_factory(db)
    return make_context(landscape=factory.plugin_audit_writer())


class TestCSVSinkResumeContract:
    """Tests for CSVSink resume contract (public API)."""

    def test_csv_sink_supports_resume(self) -> None:
        """CSVSink should declare supports_resume=True."""
        assert CSVSink.supports_resume is True


class TestCSVSinkResumeEndToEnd:
    """End-to-end tests for CSVSink resume capability."""

    def test_resume_appends_to_partial_output(self, tmp_path: Path, ctx: PluginContext) -> None:
        """Resume should append to existing partial output without duplicating headers."""
        output_path = tmp_path / "output.csv"

        # Phase 1: Initial write (simulating partial run before crash)
        sink1 = inject_write_failure(
            CSVSink(
                {
                    "path": str(output_path),
                    "schema": {"mode": "fixed", "fields": ["id: int", "value: str"]},
                }
            )
        )
        sink1.write([{"id": 1, "value": "first"}, {"id": 2, "value": "second"}], ctx)
        sink1.flush()
        sink1.close()

        # Verify partial state
        content_partial = output_path.read_text()
        lines_partial = content_partial.strip().split("\n")
        assert len(lines_partial) == 3  # header + 2 rows
        assert "id" in lines_partial[0] and "value" in lines_partial[0]

        # Phase 2: Resume write (simulating resumed run)
        sink2 = inject_write_failure(
            CSVSink(
                {
                    "path": str(output_path),
                    "schema": {"mode": "fixed", "fields": ["id: int", "value: str"]},
                }
            )
        )
        sink2.configure_for_resume()  # CRITICAL: Switch to append mode
        sink2.write([{"id": 3, "value": "third"}, {"id": 4, "value": "fourth"}], ctx)
        sink2.flush()
        sink2.close()

        # Verify final state: all data present, one header
        content_final = output_path.read_text()
        lines_final = content_final.strip().split("\n")
        assert len(lines_final) == 5  # header + 4 rows

        # Verify header appears only once
        header_count = sum(1 for line in lines_final if "id" in line and "value" in line)
        assert header_count == 1

        # Verify all data is present
        assert "first" in content_final
        assert "second" in content_final
        assert "third" in content_final
        assert "fourth" in content_final

    def test_resume_with_multiple_batches(self, tmp_path: Path, ctx: PluginContext) -> None:
        """Resume should handle multiple write batches correctly."""
        output_path = tmp_path / "output.csv"

        # Phase 1: Initial write with multiple batches
        sink1 = inject_write_failure(
            CSVSink(
                {
                    "path": str(output_path),
                    "schema": STRICT_SCHEMA,
                }
            )
        )
        sink1.write([{"id": 1}], ctx)
        sink1.write([{"id": 2}, {"id": 3}], ctx)  # Second batch
        sink1.flush()
        sink1.close()

        # Phase 2: Resume with multiple batches
        sink2 = inject_write_failure(
            CSVSink(
                {
                    "path": str(output_path),
                    "schema": STRICT_SCHEMA,
                }
            )
        )
        sink2.configure_for_resume()
        sink2.write([{"id": 4}], ctx)
        sink2.write([{"id": 5}, {"id": 6}], ctx)  # Second batch in resume
        sink2.flush()
        sink2.close()

        # Verify all data present
        content = output_path.read_text()
        lines = content.strip().split("\n")
        assert len(lines) == 7  # header + 6 rows

        # Verify all IDs present
        for i in range(1, 7):
            assert str(i) in content

    def test_resume_on_empty_file_creates_headers(self, tmp_path: Path, ctx: PluginContext) -> None:
        """Resume on empty file should create headers like normal write."""
        output_path = tmp_path / "output.csv"

        # Create empty file
        output_path.touch()

        # Resume write to empty file
        sink = inject_write_failure(
            CSVSink(
                {
                    "path": str(output_path),
                    "schema": {"mode": "fixed", "fields": ["id: int", "name: str"]},
                }
            )
        )
        sink.configure_for_resume()
        sink.write([{"id": 1, "name": "Alice"}], ctx)
        sink.flush()
        sink.close()

        # Verify headers were written
        content = output_path.read_text()
        lines = content.strip().split("\n")
        assert len(lines) == 2  # header + 1 row
        assert "id" in lines[0] and "name" in lines[0]

    def test_resume_on_nonexistent_file_creates_file(self, tmp_path: Path, ctx: PluginContext) -> None:
        """Resume on nonexistent file should create it like normal write."""
        output_path = tmp_path / "output.csv"

        # Ensure file doesn't exist
        assert not output_path.exists()

        # Resume write to nonexistent file
        sink = inject_write_failure(
            CSVSink(
                {
                    "path": str(output_path),
                    "schema": {"mode": "fixed", "fields": ["id: int"]},
                }
            )
        )
        sink.configure_for_resume()
        sink.write([{"id": 1}, {"id": 2}], ctx)
        sink.flush()
        sink.close()

        # Verify file was created with headers
        assert output_path.exists()
        content = output_path.read_text()
        lines = content.strip().split("\n")
        assert len(lines) == 3  # header + 2 rows
        assert lines[0] == "id"

    def test_resume_preserves_column_order(self, tmp_path: Path, ctx: PluginContext) -> None:
        """Resume should preserve existing column order from file."""
        output_path = tmp_path / "output.csv"

        # Phase 1: Write with specific column order
        sink1 = inject_write_failure(
            CSVSink(
                {
                    "path": str(output_path),
                    "schema": {"mode": "fixed", "fields": ["name: str", "age: int", "city: str"]},
                }
            )
        )
        sink1.write([{"name": "Alice", "age": 30, "city": "NYC"}], ctx)
        sink1.flush()
        sink1.close()

        # Get original header order
        content1 = output_path.read_text()
        original_header = content1.strip().split("\n")[0]

        # Phase 2: Resume (dict order might differ)
        sink2 = inject_write_failure(
            CSVSink(
                {
                    "path": str(output_path),
                    "schema": {"mode": "fixed", "fields": ["name: str", "age: int", "city: str"]},
                }
            )
        )
        sink2.configure_for_resume()
        # Write with different dict key order
        sink2.write([{"city": "LA", "name": "Bob", "age": 25}], ctx)
        sink2.flush()
        sink2.close()

        # Verify header order preserved
        content_final = output_path.read_text()
        lines_final = content_final.strip().split("\n")
        assert lines_final[0] == original_header

    def test_resume_without_configure_overwrites_file(self, tmp_path: Path, ctx: PluginContext) -> None:
        """Forgetting to call configure_for_resume should overwrite (demonstrating need for it)."""
        output_path = tmp_path / "output.csv"

        # Phase 1: Initial write
        sink1 = inject_write_failure(
            CSVSink(
                {
                    "path": str(output_path),
                    "schema": STRICT_SCHEMA,
                }
            )
        )
        sink1.write([{"id": 1}, {"id": 2}], ctx)
        sink1.flush()
        sink1.close()

        # Phase 2: Write without configure_for_resume (default mode="write")
        sink2 = inject_write_failure(
            CSVSink(
                {
                    "path": str(output_path),
                    "schema": STRICT_SCHEMA,
                }
            )
        )
        # NO configure_for_resume() call - should truncate
        sink2.write([{"id": 3}], ctx)
        sink2.flush()
        sink2.close()

        # Verify only new data (old data lost)
        content = output_path.read_text()
        lines = content.strip().split("\n")
        assert len(lines) == 2  # header + 1 row
        assert "1" not in content  # Old data gone
        assert "2" not in content  # Old data gone
        assert "3" in content  # Only new data present
