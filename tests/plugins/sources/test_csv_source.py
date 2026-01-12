"""Tests for CSV source plugin."""

from pathlib import Path
from typing import Iterator

import pytest

from elspeth.plugins.context import PluginContext
from elspeth.plugins.protocols import SourceProtocol


class TestCSVSource:
    """Tests for CSVSource plugin."""

    @pytest.fixture
    def sample_csv(self, tmp_path: Path) -> Path:
        """Create a sample CSV file."""
        csv_file = tmp_path / "sample.csv"
        csv_file.write_text("id,name,value\n1,alice,100\n2,bob,200\n3,carol,300\n")
        return csv_file

    @pytest.fixture
    def ctx(self) -> PluginContext:
        """Create a minimal plugin context."""
        return PluginContext(run_id="test-run", config={})

    def test_implements_protocol(self) -> None:
        """CSVSource implements SourceProtocol."""
        from elspeth.plugins.sources.csv_source import CSVSource

        assert isinstance(CSVSource, type)
        # Runtime check via Protocol
        source = CSVSource({"path": "/tmp/test.csv"})
        assert isinstance(source, SourceProtocol)

    def test_has_required_attributes(self) -> None:
        """CSVSource has name and output_schema."""
        from elspeth.plugins.sources.csv_source import CSVSource

        assert CSVSource.name == "csv"
        assert hasattr(CSVSource, "output_schema")

    def test_load_yields_rows(self, sample_csv: Path, ctx: PluginContext) -> None:
        """load() yields dict rows from CSV."""
        from elspeth.plugins.sources.csv_source import CSVSource

        source = CSVSource({"path": str(sample_csv)})
        rows = list(source.load(ctx))

        assert len(rows) == 3
        assert rows[0] == {"id": "1", "name": "alice", "value": "100"}
        assert rows[1]["name"] == "bob"
        assert rows[2]["value"] == "300"

    def test_load_with_delimiter(self, tmp_path: Path, ctx: PluginContext) -> None:
        """CSV with custom delimiter."""
        from elspeth.plugins.sources.csv_source import CSVSource

        csv_file = tmp_path / "semicolon.csv"
        csv_file.write_text("id;name;value\n1;alice;100\n")

        source = CSVSource({"path": str(csv_file), "delimiter": ";"})
        rows = list(source.load(ctx))

        assert len(rows) == 1
        assert rows[0]["name"] == "alice"

    def test_load_with_encoding(self, tmp_path: Path, ctx: PluginContext) -> None:
        """CSV with non-UTF8 encoding."""
        from elspeth.plugins.sources.csv_source import CSVSource

        csv_file = tmp_path / "latin1.csv"
        csv_file.write_bytes(b"id,name\n1,caf\xe9\n")

        source = CSVSource({"path": str(csv_file), "encoding": "latin-1"})
        rows = list(source.load(ctx))

        assert rows[0]["name"] == "caf\xe9"

    def test_close_is_idempotent(self, sample_csv: Path, ctx: PluginContext) -> None:
        """close() can be called multiple times."""
        from elspeth.plugins.sources.csv_source import CSVSource

        source = CSVSource({"path": str(sample_csv)})
        list(source.load(ctx))  # Consume iterator
        source.close()
        source.close()  # Should not raise

    def test_file_not_found_raises(self, ctx: PluginContext) -> None:
        """Missing file raises FileNotFoundError."""
        from elspeth.plugins.sources.csv_source import CSVSource

        source = CSVSource({"path": "/nonexistent/file.csv"})
        with pytest.raises(FileNotFoundError):
            list(source.load(ctx))
