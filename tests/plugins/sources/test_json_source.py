"""Tests for JSON source plugin."""

import json
from pathlib import Path

import pytest

from elspeth.plugins.context import PluginContext
from elspeth.plugins.protocols import SourceProtocol

# Dynamic schema config for tests - PathConfig now requires schema
DYNAMIC_SCHEMA = {"fields": "dynamic"}


class TestJSONSource:
    """Tests for JSONSource plugin."""

    @pytest.fixture
    def ctx(self) -> PluginContext:
        """Create a minimal plugin context."""
        return PluginContext(run_id="test-run", config={})

    def test_implements_protocol(self) -> None:
        """JSONSource implements SourceProtocol."""
        from elspeth.plugins.sources.json_source import JSONSource

        source = JSONSource({"path": "/tmp/test.json", "schema": DYNAMIC_SCHEMA})
        assert isinstance(source, SourceProtocol)

    def test_has_required_attributes(self) -> None:
        """JSONSource has name and output_schema."""
        from elspeth.plugins.sources.json_source import JSONSource

        assert JSONSource.name == "json"
        # output_schema is an instance attribute (set based on config)
        source = JSONSource({"path": "/tmp/test.json", "schema": DYNAMIC_SCHEMA})
        assert hasattr(source, "output_schema")

    def test_load_json_array(self, tmp_path: Path, ctx: PluginContext) -> None:
        """Load rows from JSON array file."""
        from elspeth.plugins.sources.json_source import JSONSource

        json_file = tmp_path / "data.json"
        data = [
            {"id": 1, "name": "alice"},
            {"id": 2, "name": "bob"},
        ]
        json_file.write_text(json.dumps(data))

        source = JSONSource({"path": str(json_file), "schema": DYNAMIC_SCHEMA})
        rows = list(source.load(ctx))

        assert len(rows) == 2
        assert rows[0] == {"id": 1, "name": "alice"}
        assert rows[1]["name"] == "bob"

    def test_load_jsonl(self, tmp_path: Path, ctx: PluginContext) -> None:
        """Load rows from JSONL file."""
        from elspeth.plugins.sources.json_source import JSONSource

        jsonl_file = tmp_path / "data.jsonl"
        jsonl_file.write_text(
            '{"id": 1, "name": "alice"}\n'
            '{"id": 2, "name": "bob"}\n'
            '{"id": 3, "name": "carol"}\n'
        )

        source = JSONSource(
            {"path": str(jsonl_file), "format": "jsonl", "schema": DYNAMIC_SCHEMA}
        )
        rows = list(source.load(ctx))

        assert len(rows) == 3
        assert rows[2]["name"] == "carol"

    def test_auto_detect_jsonl_by_extension(
        self, tmp_path: Path, ctx: PluginContext
    ) -> None:
        """Auto-detect JSONL format from .jsonl extension."""
        from elspeth.plugins.sources.json_source import JSONSource

        jsonl_file = tmp_path / "data.jsonl"
        jsonl_file.write_text('{"id": 1}\n{"id": 2}\n')

        source = JSONSource(
            {"path": str(jsonl_file), "schema": DYNAMIC_SCHEMA}
        )  # No format specified
        rows = list(source.load(ctx))

        assert len(rows) == 2

    def test_json_object_with_data_key(
        self, tmp_path: Path, ctx: PluginContext
    ) -> None:
        """Load rows from nested JSON object using data_key."""
        from elspeth.plugins.sources.json_source import JSONSource

        json_file = tmp_path / "wrapped.json"
        data = {
            "metadata": {"count": 2},
            "results": [{"id": 1}, {"id": 2}],
        }
        json_file.write_text(json.dumps(data))

        source = JSONSource(
            {"path": str(json_file), "data_key": "results", "schema": DYNAMIC_SCHEMA}
        )
        rows = list(source.load(ctx))

        assert len(rows) == 2
        assert rows[0] == {"id": 1}

    def test_empty_lines_ignored_in_jsonl(
        self, tmp_path: Path, ctx: PluginContext
    ) -> None:
        """Empty lines in JSONL are ignored."""
        from elspeth.plugins.sources.json_source import JSONSource

        jsonl_file = tmp_path / "sparse.jsonl"
        jsonl_file.write_text('{"id": 1}\n\n{"id": 2}\n\n')

        source = JSONSource(
            {"path": str(jsonl_file), "format": "jsonl", "schema": DYNAMIC_SCHEMA}
        )
        rows = list(source.load(ctx))

        assert len(rows) == 2

    def test_file_not_found_raises(self, ctx: PluginContext) -> None:
        """Missing file raises FileNotFoundError."""
        from elspeth.plugins.sources.json_source import JSONSource

        source = JSONSource(
            {"path": "/nonexistent/file.json", "schema": DYNAMIC_SCHEMA}
        )
        with pytest.raises(FileNotFoundError):
            list(source.load(ctx))

    def test_non_array_json_raises(self, tmp_path: Path, ctx: PluginContext) -> None:
        """Non-array JSON raises ValueError."""
        from elspeth.plugins.sources.json_source import JSONSource

        json_file = tmp_path / "object.json"
        json_file.write_text('{"not": "an_array"}')

        source = JSONSource({"path": str(json_file), "schema": DYNAMIC_SCHEMA})
        with pytest.raises(ValueError, match="Expected JSON array"):
            list(source.load(ctx))

    def test_close_is_idempotent(self, tmp_path: Path, ctx: PluginContext) -> None:
        """close() can be called multiple times."""
        from elspeth.plugins.sources.json_source import JSONSource

        json_file = tmp_path / "data.json"
        json_file.write_text("[]")

        source = JSONSource({"path": str(json_file), "schema": DYNAMIC_SCHEMA})
        list(source.load(ctx))
        source.close()
        source.close()  # Should not raise
