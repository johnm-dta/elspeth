"""Tests for JSONSink resume capability."""

import json
from pathlib import Path

import pytest

from elspeth.contracts.plugin_context import PluginContext
from elspeth.plugins.sinks.json_sink import JSONSink
from tests.fixtures.base_classes import inject_write_failure
from tests.fixtures.factories import make_context
from tests.fixtures.landscape import make_factory, make_landscape_db


@pytest.fixture
def ctx() -> PluginContext:
    """Create test context."""
    db = make_landscape_db()
    factory = make_factory(db)
    return make_context(landscape=factory.plugin_audit_writer())


class TestJSONSinkResumeCapability:
    """Tests for JSONSink resume declaration."""

    def test_jsonl_sink_supports_resume(self):
        """JSONL format should support resume."""
        sink = inject_write_failure(
            JSONSink(
                {
                    "path": "/tmp/test.jsonl",
                    "schema": {"mode": "observed"},
                    "format": "jsonl",
                }
            )
        )
        assert sink.supports_resume is True

    def test_json_array_sink_does_not_support_resume(self):
        """JSON array format should NOT support resume."""
        sink = inject_write_failure(
            JSONSink(
                {
                    "path": "/tmp/test.json",
                    "schema": {"mode": "observed"},
                    "format": "json",
                }
            )
        )
        assert sink.supports_resume is False

    def test_json_sink_auto_detect_jsonl_supports_resume(self):
        """Auto-detected JSONL format should support resume."""
        sink = inject_write_failure(
            JSONSink(
                {
                    "path": "/tmp/test.jsonl",  # .jsonl extension
                    "schema": {"mode": "observed"},
                    # No format specified - auto-detect
                }
            )
        )
        assert sink.supports_resume is True

    def test_json_sink_auto_detect_json_does_not_support_resume(self):
        """Auto-detected JSON array format should NOT support resume."""
        sink = inject_write_failure(
            JSONSink(
                {
                    "path": "/tmp/test.json",  # .json extension
                    "schema": {"mode": "observed"},
                    # No format specified - auto-detect
                }
            )
        )
        assert sink.supports_resume is False


class TestJSONSinkConfigureForResume:
    """Tests for JSONSink configure_for_resume error contract."""

    def test_json_array_configure_for_resume_raises(self):
        """JSON array sink configure_for_resume should raise NotImplementedError."""
        sink = inject_write_failure(
            JSONSink(
                {
                    "path": "/tmp/test.json",
                    "schema": {"mode": "observed"},
                    "format": "json",
                }
            )
        )

        with pytest.raises(NotImplementedError):
            sink.configure_for_resume()


class TestJSONSinkResumeEndToEnd:
    """End-to-end tests for JSONL resume functionality."""

    def test_jsonl_resume_appends_to_existing_file(self, tmp_path: Path, ctx: PluginContext) -> None:
        """JSONL resume should append rows to existing file."""
        output_path = tmp_path / "output.jsonl"

        # First write - initial rows
        sink1 = inject_write_failure(
            JSONSink(
                {
                    "path": str(output_path),
                    "schema": {"mode": "observed"},
                    "format": "jsonl",
                }
            )
        )
        sink1.write([{"id": 1, "value": "a"}, {"id": 2, "value": "b"}], ctx)
        sink1.flush()
        sink1.close()

        # Verify first write - 2 rows
        content1 = output_path.read_text()
        lines1 = content1.strip().split("\n")
        assert len(lines1) == 2
        assert json.loads(lines1[0]) == {"id": 1, "value": "a"}
        assert json.loads(lines1[1]) == {"id": 2, "value": "b"}

        # Second write - configure for resume and append more rows
        sink2 = inject_write_failure(
            JSONSink(
                {
                    "path": str(output_path),
                    "schema": {"mode": "observed"},
                    "format": "jsonl",
                }
            )
        )
        sink2.configure_for_resume()
        sink2.write([{"id": 3, "value": "c"}, {"id": 4, "value": "d"}], ctx)
        sink2.flush()
        sink2.close()

        # Verify all rows present - 4 total
        content2 = output_path.read_text()
        lines2 = content2.strip().split("\n")
        assert len(lines2) == 4
        assert json.loads(lines2[0]) == {"id": 1, "value": "a"}
        assert json.loads(lines2[1]) == {"id": 2, "value": "b"}
        assert json.loads(lines2[2]) == {"id": 3, "value": "c"}
        assert json.loads(lines2[3]) == {"id": 4, "value": "d"}

    def test_jsonl_resume_with_multiple_appends(self, tmp_path: Path, ctx: PluginContext) -> None:
        """Multiple resume operations should accumulate rows."""
        output_path = tmp_path / "output.jsonl"

        # Initial write
        sink1 = inject_write_failure(
            JSONSink(
                {
                    "path": str(output_path),
                    "schema": {"mode": "observed"},
                    "format": "jsonl",
                }
            )
        )
        sink1.write([{"id": 1}], ctx)
        sink1.flush()
        sink1.close()

        # First resume
        sink2 = inject_write_failure(
            JSONSink(
                {
                    "path": str(output_path),
                    "schema": {"mode": "observed"},
                    "format": "jsonl",
                }
            )
        )
        sink2.configure_for_resume()
        sink2.write([{"id": 2}], ctx)
        sink2.flush()
        sink2.close()

        # Second resume
        sink3 = inject_write_failure(
            JSONSink(
                {
                    "path": str(output_path),
                    "schema": {"mode": "observed"},
                    "format": "jsonl",
                }
            )
        )
        sink3.configure_for_resume()
        sink3.write([{"id": 3}], ctx)
        sink3.flush()
        sink3.close()

        # Verify all rows present
        content = output_path.read_text()
        lines = content.strip().split("\n")
        assert len(lines) == 3
        assert json.loads(lines[0]) == {"id": 1}
        assert json.loads(lines[1]) == {"id": 2}
        assert json.loads(lines[2]) == {"id": 3}

    def test_jsonl_resume_creates_file_if_not_exists(self, tmp_path: Path, ctx: PluginContext) -> None:
        """Resume on non-existent file should create it."""
        output_path = tmp_path / "new_file.jsonl"
        assert not output_path.exists()

        sink = inject_write_failure(
            JSONSink(
                {
                    "path": str(output_path),
                    "schema": {"mode": "observed"},
                    "format": "jsonl",
                }
            )
        )
        sink.configure_for_resume()
        sink.write([{"id": 1, "value": "test"}], ctx)
        sink.flush()
        sink.close()

        # Should create file with data
        assert output_path.exists()
        content = output_path.read_text()
        lines = content.strip().split("\n")
        assert len(lines) == 1
        assert json.loads(lines[0]) == {"id": 1, "value": "test"}

    def test_jsonl_resume_with_empty_existing_file(self, tmp_path: Path, ctx: PluginContext) -> None:
        """Resume with empty file should add data."""
        output_path = tmp_path / "empty.jsonl"
        output_path.touch()  # Create empty file

        sink = inject_write_failure(
            JSONSink(
                {
                    "path": str(output_path),
                    "schema": {"mode": "observed"},
                    "format": "jsonl",
                }
            )
        )
        sink.configure_for_resume()
        sink.write([{"id": 1, "value": "test"}], ctx)
        sink.flush()
        sink.close()

        # Should have data
        content = output_path.read_text()
        lines = content.strip().split("\n")
        assert len(lines) == 1
        assert json.loads(lines[0]) == {"id": 1, "value": "test"}

    def test_jsonl_auto_detect_supports_resume(self, tmp_path: Path, ctx: PluginContext) -> None:
        """Auto-detected JSONL format should support resume."""
        output_path = tmp_path / "output.jsonl"

        # First write - format auto-detected from extension
        sink1 = inject_write_failure(
            JSONSink(
                {
                    "path": str(output_path),
                    "schema": {"mode": "observed"},
                    # No format specified - auto-detect
                }
            )
        )
        sink1.write([{"id": 1}], ctx)
        sink1.flush()
        sink1.close()

        # Resume should work with auto-detected format
        sink2 = inject_write_failure(
            JSONSink(
                {
                    "path": str(output_path),
                    "schema": {"mode": "observed"},
                }
            )
        )
        sink2.configure_for_resume()
        sink2.write([{"id": 2}], ctx)
        sink2.flush()
        sink2.close()

        # Verify both rows present
        content = output_path.read_text()
        lines = content.strip().split("\n")
        assert len(lines) == 2
        assert json.loads(lines[0]) == {"id": 1}
        assert json.loads(lines[1]) == {"id": 2}

    def test_jsonl_resume_preserves_field_order(self, tmp_path: Path, ctx: PluginContext) -> None:
        """Resume should preserve field order from original writes."""
        output_path = tmp_path / "output.jsonl"

        # First write with specific field order
        sink1 = inject_write_failure(
            JSONSink(
                {
                    "path": str(output_path),
                    "schema": {"mode": "observed"},
                    "format": "jsonl",
                }
            )
        )
        sink1.write([{"name": "Alice", "age": 30, "city": "NYC"}], ctx)
        sink1.flush()
        sink1.close()

        # Resume with different field order in input dict
        sink2 = inject_write_failure(
            JSONSink(
                {
                    "path": str(output_path),
                    "schema": {"mode": "observed"},
                    "format": "jsonl",
                }
            )
        )
        sink2.configure_for_resume()
        sink2.write([{"city": "LA", "name": "Bob", "age": 25}], ctx)
        sink2.flush()
        sink2.close()

        # Both rows should be valid JSON objects
        content = output_path.read_text()
        lines = content.strip().split("\n")
        assert len(lines) == 2

        row1 = json.loads(lines[0])
        row2 = json.loads(lines[1])

        assert row1 == {"name": "Alice", "age": 30, "city": "NYC"}
        assert row2 == {"city": "LA", "name": "Bob", "age": 25}

    def test_jsonl_write_mode_truncates_existing_file(self, tmp_path: Path, ctx: PluginContext) -> None:
        """Write mode (default) should truncate existing file, not append."""
        output_path = tmp_path / "output.jsonl"

        # First write
        sink1 = inject_write_failure(
            JSONSink(
                {
                    "path": str(output_path),
                    "schema": {"mode": "observed"},
                    "format": "jsonl",
                }
            )
        )
        sink1.write([{"id": 1}], ctx)
        sink1.flush()
        sink1.close()

        # Second write without configure_for_resume (should truncate)
        sink2 = inject_write_failure(
            JSONSink(
                {
                    "path": str(output_path),
                    "schema": {"mode": "observed"},
                    "format": "jsonl",
                }
            )
        )
        sink2.write([{"id": 2}], ctx)
        sink2.flush()
        sink2.close()

        # Should only have second row
        content = output_path.read_text()
        lines = content.strip().split("\n")
        assert len(lines) == 1
        assert json.loads(lines[0]) == {"id": 2}
