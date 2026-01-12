"""Tests for JSON sink plugin."""

import json
from pathlib import Path

import pytest

from elspeth.plugins.context import PluginContext
from elspeth.plugins.protocols import SinkProtocol


class TestJSONSink:
    """Tests for JSONSink plugin."""

    @pytest.fixture
    def ctx(self) -> PluginContext:
        """Create a minimal plugin context."""
        return PluginContext(run_id="test-run", config={})

    def test_implements_protocol(self) -> None:
        """JSONSink implements SinkProtocol."""
        from elspeth.plugins.sinks.json_sink import JSONSink

        sink = JSONSink({"path": "/tmp/test.json"})
        assert isinstance(sink, SinkProtocol)

    def test_has_required_attributes(self) -> None:
        """JSONSink has name and input_schema."""
        from elspeth.plugins.sinks.json_sink import JSONSink

        assert JSONSink.name == "json"
        assert hasattr(JSONSink, "input_schema")

    def test_write_json_array(self, tmp_path: Path, ctx: PluginContext) -> None:
        """Write rows as JSON array."""
        from elspeth.plugins.sinks.json_sink import JSONSink

        output_file = tmp_path / "output.json"
        sink = JSONSink({"path": str(output_file), "format": "json"})

        sink.write({"id": 1, "name": "alice"}, ctx)
        sink.write({"id": 2, "name": "bob"}, ctx)
        sink.flush()
        sink.close()

        data = json.loads(output_file.read_text())
        assert isinstance(data, list)
        assert len(data) == 2
        assert data[0]["name"] == "alice"

    def test_write_jsonl(self, tmp_path: Path, ctx: PluginContext) -> None:
        """Write rows as JSONL (one per line)."""
        from elspeth.plugins.sinks.json_sink import JSONSink

        output_file = tmp_path / "output.jsonl"
        sink = JSONSink({"path": str(output_file), "format": "jsonl"})

        sink.write({"id": 1, "name": "alice"}, ctx)
        sink.write({"id": 2, "name": "bob"}, ctx)
        sink.flush()
        sink.close()

        lines = output_file.read_text().strip().split("\n")
        assert len(lines) == 2
        assert json.loads(lines[0])["name"] == "alice"
        assert json.loads(lines[1])["name"] == "bob"

    def test_auto_detect_format_from_extension(
        self, tmp_path: Path, ctx: PluginContext
    ) -> None:
        """Auto-detect JSONL format from .jsonl extension."""
        from elspeth.plugins.sinks.json_sink import JSONSink

        # .jsonl extension should default to jsonl format
        output_file = tmp_path / "output.jsonl"
        sink = JSONSink({"path": str(output_file)})

        sink.write({"id": 1}, ctx)
        sink.flush()
        sink.close()

        # Should be JSONL format (one object per line, not array)
        content = output_file.read_text().strip()
        data = json.loads(content)
        assert data == {"id": 1}  # Single object, not array

    def test_json_extension_defaults_to_array(
        self, tmp_path: Path, ctx: PluginContext
    ) -> None:
        """Auto-detect JSON array format from .json extension."""
        from elspeth.plugins.sinks.json_sink import JSONSink

        output_file = tmp_path / "output.json"
        sink = JSONSink({"path": str(output_file)})

        sink.write({"id": 1}, ctx)
        sink.flush()
        sink.close()

        data = json.loads(output_file.read_text())
        assert isinstance(data, list)
        assert data == [{"id": 1}]

    def test_close_is_idempotent(self, tmp_path: Path, ctx: PluginContext) -> None:
        """close() can be called multiple times."""
        from elspeth.plugins.sinks.json_sink import JSONSink

        output_file = tmp_path / "output.json"
        sink = JSONSink({"path": str(output_file)})

        sink.write({"id": 1}, ctx)
        sink.close()
        sink.close()  # Should not raise

    def test_pretty_print_option(self, tmp_path: Path, ctx: PluginContext) -> None:
        """Support pretty-printed JSON output."""
        from elspeth.plugins.sinks.json_sink import JSONSink

        output_file = tmp_path / "output.json"
        sink = JSONSink({"path": str(output_file), "format": "json", "indent": 2})

        sink.write({"id": 1}, ctx)
        sink.flush()
        sink.close()

        content = output_file.read_text()
        assert "\n" in content  # Pretty-printed has newlines
        assert "  " in content  # Has indentation
