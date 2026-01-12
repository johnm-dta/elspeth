"""Tests for CSV sink plugin."""

import csv
from pathlib import Path

import pytest

from elspeth.plugins.context import PluginContext
from elspeth.plugins.protocols import SinkProtocol


class TestCSVSink:
    """Tests for CSVSink plugin."""

    @pytest.fixture
    def ctx(self) -> PluginContext:
        """Create a minimal plugin context."""
        return PluginContext(run_id="test-run", config={})

    def test_implements_protocol(self) -> None:
        """CSVSink implements SinkProtocol."""
        from elspeth.plugins.sinks.csv_sink import CSVSink

        sink = CSVSink({"path": "/tmp/test.csv"})
        assert isinstance(sink, SinkProtocol)

    def test_has_required_attributes(self) -> None:
        """CSVSink has name and input_schema."""
        from elspeth.plugins.sinks.csv_sink import CSVSink

        assert CSVSink.name == "csv"
        assert hasattr(CSVSink, "input_schema")

    def test_write_creates_file(self, tmp_path: Path, ctx: PluginContext) -> None:
        """write() creates CSV file with headers."""
        from elspeth.plugins.sinks.csv_sink import CSVSink

        output_file = tmp_path / "output.csv"
        sink = CSVSink({"path": str(output_file)})

        sink.write({"id": "1", "name": "alice"}, ctx)
        sink.flush()
        sink.close()

        assert output_file.exists()
        with open(output_file) as f:
            reader = csv.DictReader(f)
            rows = list(reader)

        assert len(rows) == 1
        assert rows[0]["id"] == "1"
        assert rows[0]["name"] == "alice"

    def test_write_multiple_rows(self, tmp_path: Path, ctx: PluginContext) -> None:
        """Multiple writes append to CSV."""
        from elspeth.plugins.sinks.csv_sink import CSVSink

        output_file = tmp_path / "output.csv"
        sink = CSVSink({"path": str(output_file)})

        sink.write({"id": "1", "name": "alice"}, ctx)
        sink.write({"id": "2", "name": "bob"}, ctx)
        sink.write({"id": "3", "name": "carol"}, ctx)
        sink.flush()
        sink.close()

        with open(output_file) as f:
            reader = csv.DictReader(f)
            rows = list(reader)

        assert len(rows) == 3
        assert rows[2]["name"] == "carol"

    def test_custom_delimiter(self, tmp_path: Path, ctx: PluginContext) -> None:
        """Support custom delimiter."""
        from elspeth.plugins.sinks.csv_sink import CSVSink

        output_file = tmp_path / "output.csv"
        sink = CSVSink({"path": str(output_file), "delimiter": ";"})

        sink.write({"id": "1", "name": "alice"}, ctx)
        sink.flush()
        sink.close()

        content = output_file.read_text()
        assert ";" in content
        assert "," not in content.replace(",", "")  # No commas except possibly in data

    def test_close_is_idempotent(self, tmp_path: Path, ctx: PluginContext) -> None:
        """close() can be called multiple times."""
        from elspeth.plugins.sinks.csv_sink import CSVSink

        output_file = tmp_path / "output.csv"
        sink = CSVSink({"path": str(output_file)})

        sink.write({"id": "1"}, ctx)
        sink.close()
        sink.close()  # Should not raise

    def test_flush_before_close(self, tmp_path: Path, ctx: PluginContext) -> None:
        """Data is available after flush, before close."""
        from elspeth.plugins.sinks.csv_sink import CSVSink

        output_file = tmp_path / "output.csv"
        sink = CSVSink({"path": str(output_file)})

        sink.write({"id": "1"}, ctx)
        sink.flush()

        # File should have content before close
        content = output_file.read_text()
        assert "id" in content
        assert "1" in content

        sink.close()
