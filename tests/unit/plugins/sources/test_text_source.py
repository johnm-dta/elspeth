"""Tests for the plain-text line source plugin."""

from pathlib import Path

import pytest

from elspeth.contracts.plugin_context import PluginContext
from tests.fixtures.factories import make_context, make_source_context

DYNAMIC_SCHEMA = {"mode": "observed"}
QUARANTINE_SINK = "quarantine"


class TestTextSource:
    """Tests for TextSource plugin."""

    @pytest.fixture
    def ctx(self) -> PluginContext:
        return make_source_context(plugin_name="text")

    def test_has_required_attributes(self) -> None:
        from elspeth.plugins.sources.text_source import TextSource

        assert TextSource.name == "text"
        source = TextSource(
            {
                "path": "/tmp/test.txt",
                "column": "url",
                "schema": DYNAMIC_SCHEMA,
                "on_validation_failure": QUARANTINE_SINK,
            }
        )
        assert source.output_schema is not None

    def test_loads_one_row_per_line(self, tmp_path: Path, ctx: PluginContext) -> None:
        from elspeth.plugins.sources.text_source import TextSource

        text_file = tmp_path / "urls.txt"
        text_file.write_text("https://a.example\nhttps://b.example\n", encoding="utf-8")

        source = TextSource(
            {
                "path": str(text_file),
                "column": "url",
                "schema": DYNAMIC_SCHEMA,
                "on_validation_failure": QUARANTINE_SINK,
            }
        )

        rows = list(source.load(ctx))

        assert [row.row for row in rows] == [
            {"url": "https://a.example"},
            {"url": "https://b.example"},
        ]
        assert all(row.is_quarantined is False for row in rows)

    def test_strips_whitespace_by_default(self, tmp_path: Path, ctx: PluginContext) -> None:
        from elspeth.plugins.sources.text_source import TextSource

        text_file = tmp_path / "items.md"
        text_file.write_text("  first  \n\tsecond\t\n", encoding="utf-8")

        source = TextSource(
            {
                "path": str(text_file),
                "column": "item",
                "schema": DYNAMIC_SCHEMA,
                "on_validation_failure": QUARANTINE_SINK,
            }
        )

        rows = list(source.load(ctx))

        assert [row.row for row in rows] == [{"item": "first"}, {"item": "second"}]

    def test_skips_blank_lines_by_default(self, tmp_path: Path, ctx: PluginContext) -> None:
        from elspeth.plugins.sources.text_source import TextSource

        text_file = tmp_path / "urls.txt"
        text_file.write_text("https://a.example\n\n   \nhttps://b.example\n", encoding="utf-8")

        source = TextSource(
            {
                "path": str(text_file),
                "column": "url",
                "schema": DYNAMIC_SCHEMA,
                "on_validation_failure": QUARANTINE_SINK,
            }
        )

        rows = list(source.load(ctx))

        assert [row.row for row in rows] == [
            {"url": "https://a.example"},
            {"url": "https://b.example"},
        ]

    def test_can_preserve_blank_lines(self, tmp_path: Path, ctx: PluginContext) -> None:
        from elspeth.plugins.sources.text_source import TextSource

        text_file = tmp_path / "items.txt"
        text_file.write_text("first\n\nsecond\n", encoding="utf-8")

        source = TextSource(
            {
                "path": str(text_file),
                "column": "item",
                "schema": DYNAMIC_SCHEMA,
                "skip_blank_lines": False,
                "on_validation_failure": QUARANTINE_SINK,
            }
        )

        rows = list(source.load(ctx))

        assert [row.row for row in rows] == [{"item": "first"}, {"item": ""}, {"item": "second"}]

    def test_invalid_encoding_line_quarantines_but_valid_lines_continue(self, tmp_path: Path, ctx: PluginContext) -> None:
        """Malformed bytes should quarantine one line, not abort the whole file."""
        from elspeth.plugins.sources.text_source import TextSource

        text_file = tmp_path / "bad.txt"
        text_file.write_bytes(b"good1\n\xff\ngood3\n")

        source = TextSource(
            {
                "path": str(text_file),
                "column": "line",
                "schema": DYNAMIC_SCHEMA,
                "on_validation_failure": QUARANTINE_SINK,
            }
        )

        rows = list(source.load(ctx))

        assert len(rows) == 3
        assert rows[0].is_quarantined is False
        assert rows[0].row == {"line": "good1"}

        quarantined = rows[1]
        assert quarantined.is_quarantined is True
        assert quarantined.quarantine_error is not None
        assert "line 2" in quarantined.quarantine_error
        assert "utf-8" in quarantined.quarantine_error.lower()
        assert quarantined.row["__line_number__"] == 2
        assert "__raw_bytes_hex__" in quarantined.row

        assert rows[2].is_quarantined is False
        assert rows[2].row == {"line": "good3"}

    def test_skipped_blank_lines_record_audit_summary(self, tmp_path: Path) -> None:
        """Configured blank-line skipping should still leave audit evidence."""
        from elspeth.plugins.sources.text_source import TextSource

        text_file = tmp_path / "blank.txt"
        text_file.write_text("first\n\n   \nsecond\n", encoding="utf-8")

        source = TextSource(
            {
                "path": str(text_file),
                "column": "line",
                "schema": DYNAMIC_SCHEMA,
                "on_validation_failure": QUARANTINE_SINK,
            }
        )
        ctx = make_context(node_id="source")

        rows = list(source.load(ctx))

        assert [row.row for row in rows] == [{"line": "first"}, {"line": "second"}]
        assert ctx.landscape.record_validation_error.call_count == 1

        call = ctx.landscape.record_validation_error.call_args
        assert call.kwargs["row_data"]["__skipped_blank_lines__"] == 2
        assert call.kwargs["destination"] == "discard"
        assert "skipped 2 blank line" in call.kwargs["error"].lower()

    def test_file_not_found_raises(self, ctx: PluginContext) -> None:
        from elspeth.plugins.sources.text_source import TextSource

        source = TextSource(
            {
                "path": "/nonexistent/file.txt",
                "column": "value",
                "schema": DYNAMIC_SCHEMA,
                "on_validation_failure": QUARANTINE_SINK,
            }
        )

        with pytest.raises(FileNotFoundError):
            list(source.load(ctx))

    def test_invalid_schema_quarantines_rows(self, tmp_path: Path, ctx: PluginContext) -> None:
        from elspeth.plugins.sources.text_source import TextSource

        text_file = tmp_path / "numbers.txt"
        text_file.write_text("123\nabc\n", encoding="utf-8")

        source = TextSource(
            {
                "path": str(text_file),
                "column": "value",
                "schema": {"mode": "fixed", "fields": ["value: int"]},
                "on_validation_failure": QUARANTINE_SINK,
            }
        )

        rows = list(source.load(ctx))

        assert len(rows) == 2
        assert rows[0].is_quarantined is False
        assert rows[0].row == {"value": 123}
        assert rows[1].is_quarantined is True

    def test_has_plugin_version(self) -> None:
        from elspeth.plugins.sources.text_source import TextSource

        assert isinstance(TextSource.plugin_version, str)
        assert TextSource.plugin_version != "0.0.0"
